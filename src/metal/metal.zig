//! Metal GPU backend for zgml.
//!
//! Provides `MetalBackend`, a struct that compiles MSL shaders at startup and
//! dispatches GEMM (f32, Q8_0, Q4_0) to the Apple GPU via the ObjC runtime
//! bridge in `metal_objc.zig`.
//!
//! Usage follows the same global-singleton pattern as the BLAS path:
//!
//! ```zig
//! const backend = try metal.getBackend(allocator);
//! defer metal.deinitBackend();
//!
//! const buf = try backend.uploadWeights(weight_ptr, byte_len);
//! if (MetalBackend.shouldDispatchGpu(M, N, K)) {
//!     backend.gemm(dst_ptr, input_ptr, buf, M, N, K);
//! }
//! ```

const std = @import("std");
const objc = @import("metal_objc.zig");
const shaders = @import("metal_shaders.zig");

const log = std.log.scoped(.metal);

pub const MetalBackend = struct {
    device: objc.id,
    queue: objc.id,
    f32_pipeline: objc.id,
    q8_pipeline: objc.id,
    q4_pipeline: objc.id,
    alloc: std.mem.Allocator,

    /// Weight buffer registry: map data pointer -> MTLBuffer.
    /// Allows reuse of already-uploaded weight buffers across calls.
    weight_buffers: std.AutoHashMapUnmanaged([*]const u8, objc.id),

    /// Minimum flop count to dispatch on GPU (avoid launch-overhead on tiny matmuls).
    /// 2*M*N*K >= min_gpu_flops  =>  use GPU.
    const min_gpu_flops: usize = 1024 * 1024;

    // -----------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------

    pub fn init(alloc: std.mem.Allocator) !MetalBackend {
        // 1. Get the default Metal device (GPU).
        const device = objc.createSystemDefaultDevice();
        if (device == null) {
            log.err("No Metal device found", .{});
            return error.MetalDeviceNotFound;
        }

        // Log GPU name for diagnostics.
        logDeviceName(device);

        // 2. Create command queue.
        const queue = objc.newCommandQueue(device);
        if (queue == null) {
            log.err("Failed to create Metal command queue", .{});
            return error.MetalCommandQueueFailed;
        }

        // 3. Compile all shaders from the combined source string.
        const source_ns = try objc_nsStringFromSlice(alloc, shaders.all_shaders);
        defer objc.release(source_ns);

        const library = objc.newLibraryWithSource(device, source_ns);
        if (library == null) {
            log.err("Failed to compile Metal shaders", .{});
            return error.MetalShaderCompilationFailed;
        }
        defer objc.release(library);

        // 4. Extract kernel functions and build pipeline states.
        const f32_pipeline = try buildPipeline(alloc, device, library, "f32_matmul");
        errdefer objc.release(f32_pipeline);

        const q8_pipeline = try buildPipeline(alloc, device, library, "q8_matmul");
        errdefer objc.release(q8_pipeline);

        const q4_pipeline = try buildPipeline(alloc, device, library, "q4_matmul");
        errdefer objc.release(q4_pipeline);

        return .{
            .device = device,
            .queue = queue,
            .f32_pipeline = f32_pipeline,
            .q8_pipeline = q8_pipeline,
            .q4_pipeline = q4_pipeline,
            .alloc = alloc,
            .weight_buffers = .{},
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        // Release all cached weight buffers.
        var it = self.weight_buffers.iterator();
        while (it.next()) |entry| {
            objc.release(entry.value_ptr.*);
        }
        self.weight_buffers.deinit(self.alloc);

        objc.release(self.q4_pipeline);
        objc.release(self.q8_pipeline);
        objc.release(self.f32_pipeline);
        objc.release(self.queue);
        objc.release(self.device);
        self.* = undefined;
    }

    // -----------------------------------------------------------------
    // Weight upload
    // -----------------------------------------------------------------

    /// Upload weight data to a Metal buffer (shared memory = zero-copy on Apple Silicon).
    /// The buffer is cached by data pointer so repeated calls with the same
    /// pointer are free.
    pub fn uploadWeights(self: *MetalBackend, data: [*]const u8, len: usize) !objc.id {
        // Check cache first.
        if (self.weight_buffers.get(data)) |existing| {
            return existing;
        }

        const buf = objc.newBufferWithBytes(self.device, data, len);
        if (buf == null) {
            log.err("Failed to create Metal buffer for weights ({d} bytes)", .{len});
            return error.MetalBufferAllocationFailed;
        }

        try self.weight_buffers.put(self.alloc, data, buf);
        return buf;
    }

    // -----------------------------------------------------------------
    // GEMM dispatch
    // -----------------------------------------------------------------

    /// Dispatch f32 GEMM: C[M,N] = A[M,K] * B[K,N].
    ///
    /// `input`      — pointer to A (f32, M*K elements, row-major)
    /// `weight_buf` — pre-uploaded MTLBuffer for B (f32, K*N elements)
    /// `dst`        — pointer to C (f32, M*N elements, written on return)
    pub fn gemm(
        self: *MetalBackend,
        dst: [*]u8,
        input: [*]const u8,
        weight_buf: objc.id,
        M: usize,
        N: usize,
        K: usize,
    ) void {
        // 1. Create temporary input buffer (A).
        const input_bytes = M * K * @sizeOf(f32);
        const input_buf = objc.newBufferWithBytes(self.device, input, input_bytes);
        if (input_buf == null) {
            log.err("Metal: failed to allocate input buffer ({d} bytes)", .{input_bytes});
            return;
        }
        defer objc.release(input_buf);

        // 2. Create temporary output buffer (C).
        const dst_bytes = M * N * @sizeOf(f32);
        const dst_buf = objc.newBuffer(self.device, dst_bytes);
        if (dst_buf == null) {
            log.err("Metal: failed to allocate output buffer ({d} bytes)", .{dst_bytes});
            return;
        }
        defer objc.release(dst_buf);

        // 3. Encode compute command.
        const cmd_buf = objc.newCommandBuffer(self.queue);
        if (cmd_buf == null) {
            log.err("Metal: failed to create command buffer", .{});
            return;
        }

        const encoder = objc.newComputeEncoder(cmd_buf);
        if (encoder == null) {
            log.err("Metal: failed to create compute encoder", .{});
            return;
        }

        // 4. Set pipeline state (f32 GEMM).
        objc.setComputePipelineState(encoder, self.f32_pipeline);

        // 5. Bind buffers: A=0, B=1, C=2.
        objc.setBuffer(encoder, input_buf, 0, 0);
        objc.setBuffer(encoder, weight_buf, 0, 1);
        objc.setBuffer(encoder, dst_buf, 0, 2);

        // 6. Set dimension constants M, N, K as raw bytes at indices 3, 4, 5.
        var m32: u32 = @intCast(M);
        var n32: u32 = @intCast(N);
        var k32: u32 = @intCast(K);
        objc.setBytes(encoder, @ptrCast(&m32), @sizeOf(u32), 3);
        objc.setBytes(encoder, @ptrCast(&n32), @sizeOf(u32), 4);
        objc.setBytes(encoder, @ptrCast(&k32), @sizeOf(u32), 5);

        // 7. Dispatch threadgroups.
        //    Threadgroup size: (32, 4, 1) — 128 threads.
        //    Each thread computes THREAD_TILE=8 rows for one column (4*8 = 32 rows).
        //    Grid: ceil(N/32) x ceil(M/32) x 1 threadgroups.
        const grid = objc.MTLSize{
            .width = (N + 31) / 32,
            .height = (M + 31) / 32,
            .depth = 1,
        };
        const group = objc.MTLSize{
            .width = 32,
            .height = 4,
            .depth = 1,
        };
        objc.dispatchThreadgroups(encoder, grid, group);

        // 8. End encoding, commit, and wait.
        objc.endEncoding(encoder);
        objc.commit(cmd_buf);
        objc.waitUntilCompleted(cmd_buf);

        // 9. Copy results back to caller's destination.
        const gpu_ptr = objc.bufferContents(dst_buf);
        @memcpy(dst[0..dst_bytes], gpu_ptr[0..dst_bytes]);
    }

    /// Dispatch Q8_0 GEMM: C[M,N] = A[M,K] * dequant(W[K,N]).
    ///
    /// `input`      — pointer to A (f32, M*K elements)
    /// `weight_buf` — pre-uploaded MTLBuffer for W (Q8_0 block format)
    /// `dst`        — pointer to C (f32, M*N elements)
    pub fn gemmQ8(
        self: *MetalBackend,
        dst: [*]u8,
        input: [*]const u8,
        weight_buf: objc.id,
        M: usize,
        N: usize,
        K: usize,
    ) void {
        self.dispatchQuantizedGemm(self.q8_pipeline, dst, input, weight_buf, M, N, K);
    }

    /// Dispatch Q4_0 GEMM: C[M,N] = A[M,K] * dequant(W[K,N]).
    pub fn gemmQ4(
        self: *MetalBackend,
        dst: [*]u8,
        input: [*]const u8,
        weight_buf: objc.id,
        M: usize,
        N: usize,
        K: usize,
    ) void {
        self.dispatchQuantizedGemm(self.q4_pipeline, dst, input, weight_buf, M, N, K);
    }

    /// Shared dispatch logic for quantized GEMM kernels (Q8_0 / Q4_0).
    /// These kernels use a simple 1-thread-per-output-element grid.
    fn dispatchQuantizedGemm(
        self: *MetalBackend,
        pipeline: objc.id,
        dst: [*]u8,
        input: [*]const u8,
        weight_buf: objc.id,
        M: usize,
        N: usize,
        K: usize,
    ) void {
        const input_bytes = M * K * @sizeOf(f32);
        const input_buf = objc.newBufferWithBytes(self.device, input, input_bytes);
        if (input_buf == null) {
            log.err("Metal: failed to allocate input buffer ({d} bytes)", .{input_bytes});
            return;
        }
        defer objc.release(input_buf);

        const dst_bytes = M * N * @sizeOf(f32);
        const dst_buf = objc.newBuffer(self.device, dst_bytes);
        if (dst_buf == null) {
            log.err("Metal: failed to allocate output buffer ({d} bytes)", .{dst_bytes});
            return;
        }
        defer objc.release(dst_buf);

        const cmd_buf = objc.newCommandBuffer(self.queue);
        if (cmd_buf == null) {
            log.err("Metal: failed to create command buffer", .{});
            return;
        }

        const encoder = objc.newComputeEncoder(cmd_buf);
        if (encoder == null) {
            log.err("Metal: failed to create compute encoder", .{});
            return;
        }

        objc.setComputePipelineState(encoder, pipeline);

        objc.setBuffer(encoder, input_buf, 0, 0);
        objc.setBuffer(encoder, weight_buf, 0, 1);
        objc.setBuffer(encoder, dst_buf, 0, 2);

        var m32: u32 = @intCast(M);
        var n32: u32 = @intCast(N);
        var k32: u32 = @intCast(K);
        objc.setBytes(encoder, @ptrCast(&m32), @sizeOf(u32), 3);
        objc.setBytes(encoder, @ptrCast(&n32), @sizeOf(u32), 4);
        objc.setBytes(encoder, @ptrCast(&k32), @sizeOf(u32), 5);

        // Simple grid: one thread per output element, 16x16 = 256 threads per group.
        const grid = objc.MTLSize{
            .width = (N + 15) / 16,
            .height = (M + 15) / 16,
            .depth = 1,
        };
        const group = objc.MTLSize{
            .width = 16,
            .height = 16,
            .depth = 1,
        };
        objc.dispatchThreadgroups(encoder, grid, group);

        objc.endEncoding(encoder);
        objc.commit(cmd_buf);
        objc.waitUntilCompleted(cmd_buf);

        const gpu_ptr = objc.bufferContents(dst_buf);
        @memcpy(dst[0..dst_bytes], gpu_ptr[0..dst_bytes]);
    }

    // -----------------------------------------------------------------
    // Heuristic
    // -----------------------------------------------------------------

    /// Check if a matmul is large enough to benefit from GPU dispatch.
    /// Returns true when 2*M*N*K >= min_gpu_flops.
    pub fn shouldDispatchGpu(M: usize, N: usize, K: usize) bool {
        // Overflow-safe: check components individually before multiplying.
        const flops = @as(u128, 2) * @as(u128, M) * @as(u128, N) * @as(u128, K);
        return flops >= min_gpu_flops;
    }

    // -----------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------

    /// Build a compute pipeline for the named kernel function.
    fn buildPipeline(_: std.mem.Allocator, device: objc.id, library: objc.id, name: [*:0]const u8) !objc.id {
        const name_ns = objc_nsStringLiteral(name);
        defer objc.release(name_ns);

        const func = objc.newFunctionWithName(library, name_ns);
        if (func == null) {
            log.err("Metal: kernel function '{s}' not found in library", .{name});
            return error.MetalKernelNotFound;
        }
        defer objc.release(func);

        const pso = objc.newComputePipelineState(device, func);
        if (pso == null) {
            log.err("Metal: failed to create pipeline state for '{s}'", .{name});
            return error.MetalPipelineCreationFailed;
        }

        log.info("Metal: pipeline '{s}' ready (max threads/group: {d})", .{
            name, objc.maxTotalThreadsPerThreadgroup(pso),
        });

        return pso;
    }
};

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// Global Metal backend instance (matches the `use_blas` singleton pattern).
var global_backend: ?MetalBackend = null;

/// Get (or lazily initialize) the global Metal backend.
pub fn getBackend(alloc: std.mem.Allocator) !*MetalBackend {
    if (global_backend) |*b| return b;

    global_backend = try MetalBackend.init(alloc);
    return &global_backend.?;
}

/// Tear down the global backend and release all Metal resources.
pub fn deinitBackend() void {
    if (global_backend) |*b| {
        b.deinit();
        global_backend = null;
    }
}

// ---------------------------------------------------------------------------
// NSString helpers (internal)
// ---------------------------------------------------------------------------

/// Create an NSString from a comptime-known sentinel-terminated string literal.
fn objc_nsStringLiteral(str: [*:0]const u8) objc.id {
    const c = @cImport({
        @cInclude("objc/message.h");
        @cInclude("objc/runtime.h");
    });

    const class = c.objc_getClass("NSString");
    const alloc_sel = c.sel_registerName("alloc");
    const init_sel = c.sel_registerName("initWithUTF8String:");

    const alloc_fn: *const fn (?*anyopaque, c.SEL) callconv(.C) ?*anyopaque = @ptrCast(&c.objc_msgSend);
    const raw = alloc_fn(@ptrCast(class), alloc_sel);

    const init_fn: *const fn (?*anyopaque, c.SEL, [*:0]const u8) callconv(.C) ?*anyopaque = @ptrCast(&c.objc_msgSend);
    return init_fn(raw, init_sel, str);
}

/// Create an NSString from a Zig slice (allocates a temporary null-terminated copy).
fn objc_nsStringFromSlice(alloc: std.mem.Allocator, str: []const u8) !objc.id {
    const z = try alloc.dupeZ(u8, str);
    defer alloc.free(z);
    return objc_nsStringLiteral(z.ptr);
}

/// Log the GPU device name to the scoped logger.
fn logDeviceName(device: objc.id) void {
    const c = @cImport({
        @cInclude("objc/message.h");
        @cInclude("objc/runtime.h");
    });

    const name_sel = c.sel_registerName("name");
    const utf8_sel = c.sel_registerName("UTF8String");

    const msg_fn: *const fn (?*anyopaque, c.SEL) callconv(.C) ?*anyopaque = @ptrCast(&c.objc_msgSend);
    const name_ns = msg_fn(device, name_sel);

    if (name_ns) |ns| {
        const utf8_fn: *const fn (?*anyopaque, c.SEL) callconv(.C) [*:0]const u8 = @ptrCast(&c.objc_msgSend);
        const name_cstr = utf8_fn(ns, utf8_sel);
        log.info("Metal GPU: {s}", .{name_cstr});
    } else {
        log.info("Metal GPU: (unknown)", .{});
    }
}
