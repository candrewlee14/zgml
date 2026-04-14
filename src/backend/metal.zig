//! Metal GPU backend for macOS / Apple Silicon.
//!
//! Uses shared memory (MTLResourceStorageModeShared) so upload/download
//! are plain memcpy — CPU and GPU see the same physical pages.
//! Each dispatch is synchronous (commit + waitUntilCompleted).

const std = @import("std");
const backend_mod = @import("../backend.zig");

const c = @cImport(@cInclude("metal_shim.h"));

// ── Tile size for simdgroup kernel ────────────────────────────────

const TILE: u32 = 32; // output tile per threadgroup (TILE x TILE)
// 4 simdgroups per threadgroup (128 threads), each handles 8x8 sub-tiles
// Shared memory per K step: TILE*8 + 8*TILE = 512 floats = 2 KB

// ── Metal shader source (compiled at init time) ───────────────────

const shader_source =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\constant uint TILE = 32;
    \\constant uint NSUB = 4; // 2x2 arrangement of 8x8 sub-tiles per simdgroup
    \\
    \\struct MatMulParams {
    \\    uint M; uint N; uint K;
    \\    uint a_row_stride; uint a_col_stride;
    \\    uint b_row_stride; uint b_col_stride;
    \\    uint a_offset; uint b_offset;
    \\    uint dst_offset; uint dst_row_stride;
    \\};
    \\
    \\// Simdgroup-accelerated matmul: 4 simdgroups per threadgroup,
    \\// each computing a 2x2 grid of 8x8 tiles = 16x16 per simdgroup,
    \\// 32x32 per threadgroup. Uses hardware matrix multiply-accumulate.
    \\kernel void matmul_f32(
    \\    device const float* A [[buffer(0)]],
    \\    device const float* B [[buffer(1)]],
    \\    device float* C       [[buffer(2)]],
    \\    constant MatMulParams& p [[buffer(3)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\
    \\    // Each simdgroup owns a 16x16 quadrant (2x2 of 8x8 sub-tiles).
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tA[TILE * 8];
    \\    threadgroup float tB[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        // Cooperatively load A[gRow:gRow+32, kt:kt+8] — 256 elems, 128 threads → 2 each
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ar = gRow + r, ac = kt + c;
    \\            tA[i] = (ar < p.M && ac < p.K)
    \\                ? A[p.a_offset + ar * p.a_row_stride + ac * p.a_col_stride] : 0.0f;
    \\        }
    \\        // Cooperatively load B[kt:kt+8, gCol:gCol+32] — 256 elems
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint br = kt + r, bc = gCol + c;
    \\            tB[i] = (br < p.K && bc < p.N)
    \\                ? B[p.b_offset + br * p.b_row_stride + bc * p.b_col_stride] : 0.0f;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        // Each simdgroup: 2x2 sub-tiles within its 16x16 quadrant
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tA + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tA + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tB + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tB + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    // Store 2x2 sub-tiles via threadgroup memory
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Write to global — 1024 elems, 128 threads → 8 each
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N)
    \\            C[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\struct QMatMulParams {
    \\    uint M; uint N; uint K;
    \\    uint block_size;
    \\};
    \\
    \\// Simdgroup-accelerated quantized matmul: dequantize during tile load.
    \\kernel void qmatmul_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    constant QMatMulParams& p [[buffer(4)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[ir * p.K + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < p.K && nc < p.N) {
    \\                uint w_idx = kr * p.N + nc;
    \\                tW[i] = float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N)
    \\            output[cr * p.N + cc] = tC[i];
    \\    }
    \\}
;

// ── Kernel param structs (must match MSL layout) ──────────────────

const MatMulParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const QMatMulParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
};

// ── MetalBackend ──────────────────────────────────────────────────

pub const MetalBackend = struct {
    device: *anyopaque,
    queue: *anyopaque,
    matmul_pipeline: *anyopaque,
    qmatmul_pipeline: *anyopaque,
    library: *anyopaque,
    active_commands: ?*anyopaque = null,

    pub fn init() !MetalBackend {
        const device = c.mtl_create_device() orelse return error.MetalNotAvailable;
        errdefer c.mtl_release(device);

        const queue = c.mtl_create_queue(device) orelse return error.MetalInitFailed;
        errdefer c.mtl_release(queue);

        const library = c.mtl_compile_source(device, shader_source.ptr, shader_source.len) orelse return error.ShaderCompileFailed;
        errdefer c.mtl_release(library);

        const matmul_pipeline = c.mtl_create_pipeline(device, library, "matmul_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matmul_pipeline);

        const qmatmul_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_f32") orelse return error.PipelineCreateFailed;

        return .{
            .device = device,
            .queue = queue,
            .matmul_pipeline = matmul_pipeline,
            .qmatmul_pipeline = qmatmul_pipeline,
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
        c.mtl_release(self.qmatmul_pipeline);
        c.mtl_release(self.matmul_pipeline);
        c.mtl_release(self.library);
        c.mtl_release(self.queue);
        c.mtl_release(self.device);
    }

    /// Ensure a command session is active, creating one if needed.
    fn ensureCommands(self: *MetalBackend) *anyopaque {
        if (self.active_commands == null) {
            self.active_commands = c.mtl_begin_commands(self.queue);
        }
        return self.active_commands.?;
    }

    /// Commit and wait on any active command session.
    fn flushCommands(self: *MetalBackend) void {
        if (self.active_commands) |cmds| {
            c.mtl_commit_and_wait(cmds);
            self.active_commands = null;
        }
    }

    pub fn backend(self: *MetalBackend) backend_mod.Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
            .name_str = "metal",
            .device_type = .metal,
            .capabilities = .{ .device_buffers = true },
        };
    }
};

// ── VTable implementation ─────────────────────────────────────────

fn getState(ctx: *anyopaque) *MetalBackend {
    return @ptrCast(@alignCast(ctx));
}

// Host kernel dispatch — delegate to BLAS (same as CPU backend).
fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const forward = @import("../tensor/forward.zig");
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

fn quantizedMatMulF32(_: *anyopaque, spec: backend_mod.QuantizedMatMulSpecF32) bool {
    if (spec.weight.rows != spec.K or spec.weight.cols != spec.N) return false;
    const quant = @import("../quant.zig");
    const Weight = quant.QuantizedWeight(f32);
    const weight = Weight{
        .data = spec.weight.data,
        .scales = spec.weight.scales,
        .rows = spec.weight.rows,
        .cols = spec.weight.cols,
        .block_size = spec.weight.block_size,
    };
    weight.matmul(spec.input, spec.dst, spec.M, spec.N, spec.K);
    return true;
}

// Buffer management — shared memory, so upload/download are memcpy.
fn allocBuffer(ctx: *anyopaque, size: usize) ?backend_mod.DeviceBuffer {
    const self = getState(ctx);
    const buf = c.mtl_create_buffer(self.device, size) orelse return null;
    return .{ .ptr = buf, .size = size };
}

fn freeBuffer(_: *anyopaque, buf: backend_mod.DeviceBuffer) void {
    c.mtl_release(buf.ptr);
}

fn uploadFn(_: *anyopaque, dst: backend_mod.DeviceBuffer, dst_byte_offset: usize, src: []const u8) void {
    const ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(dst.ptr));
    @memcpy(ptr[dst_byte_offset..][0..src.len], src);
}

fn downloadFn(ctx: *anyopaque, dst: []u8, src: backend_mod.DeviceBuffer, src_byte_offset: usize) void {
    const self = getState(ctx);
    self.flushCommands();
    const ptr: [*]const u8 = @ptrCast(c.mtl_buffer_contents(src.ptr));
    @memcpy(dst, ptr[src_byte_offset..][0..dst.len]);
}

fn syncFn(ctx: *anyopaque) void {
    const self = getState(ctx);
    self.flushCommands();
}

// Device kernel dispatch — run on GPU.
fn deviceMatMulF32(ctx: *anyopaque, spec: backend_mod.DeviceMatMulSpecF32) bool {
    const self = getState(ctx);
    const g = spec.geom;

    const params = MatMulParams{
        .M = @intCast(g.M),
        .N = @intCast(g.N),
        .K = @intCast(g.K),
        .a_row_stride = @intCast(g.a_row_stride),
        .a_col_stride = @intCast(g.a_col_stride),
        .b_row_stride = @intCast(g.b_row_stride),
        .b_col_stride = @intCast(g.b_col_stride),
        .a_offset = @intCast(g.a_offset),
        .b_offset = @intCast(g.b_offset),
        .dst_offset = @intCast(g.dst_offset),
        .dst_row_stride = @intCast(g.dst_row_stride),
    };

    var bufs = [_]?*anyopaque{ spec.a.ptr, spec.b.ptr, spec.dst.ptr };
    const grid_x: u32 = @intCast((g.N + TILE - 1) / TILE);
    const grid_y: u32 = @intCast((g.M + TILE - 1) / TILE);

    const cmds = self.ensureCommands();
    c.mtl_encode_dispatch(cmds, self.matmul_pipeline, @ptrCast(&bufs), 3, &params, @sizeOf(MatMulParams), 3, grid_x, grid_y, 128, 1);
    return true;
}

fn deviceQuantizedMatMulF32(ctx: *anyopaque, spec: backend_mod.DeviceQuantizedMatMulSpecF32) bool {
    const self = getState(ctx);
    const w = spec.weight;

    const params = QMatMulParams{
        .M = @intCast(spec.M),
        .N = @intCast(spec.N),
        .K = @intCast(spec.K),
        .block_size = @intCast(w.block_size),
    };

    var bufs = [_]?*anyopaque{ w.data.ptr, w.scales.ptr, spec.input.ptr, spec.dst.ptr };
    const grid_x: u32 = @intCast((spec.N + TILE - 1) / TILE);
    const grid_y: u32 = @intCast((spec.M + TILE - 1) / TILE);

    const cmds = self.ensureCommands();
    c.mtl_encode_dispatch(cmds, self.qmatmul_pipeline, @ptrCast(&bufs), 4, &params, @sizeOf(QMatMulParams), 4, grid_x, grid_y, 128, 1);
    return true;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .quantized_matmul_f32 = quantizedMatMulF32,
    .alloc_buffer = allocBuffer,
    .free_buffer = freeBuffer,
    .upload = uploadFn,
    .download = downloadFn,
    .sync = syncFn,
    .device_matmul_f32 = deviceMatMulF32,
    .device_quantized_matmul_f32 = deviceQuantizedMatMulF32,
};

// ── Tests ─────────────────────────────────────────────────────────

test "metal backend init and device buffer round-trip" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return, // skip on non-Metal systems
        else => return err,
    };
    defer metal.deinit();
    const be = metal.backend();

    const buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(buf);

    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    be.uploadSlice(f32, buf, 0, &src);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, buf, 0);

    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "metal backend dense matmul on GPU" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    const be = metal.backend();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const a_buf = be.allocSlice(f32, 6) orelse return error.OutOfMemory;
    defer be.freeBuffer(a_buf);
    const b_buf = be.allocSlice(f32, 6) orelse return error.OutOfMemory;
    defer be.freeBuffer(b_buf);
    const dst_buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(dst_buf);

    be.uploadSlice(f32, a_buf, 0, &a_data);
    be.uploadSlice(f32, b_buf, 0, &b_data);

    const ok = be.deviceMatMul(.{
        .dst = dst_buf,
        .a = a_buf,
        .b = b_buf,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });
    try std.testing.expect(ok);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, dst_buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "metal backend quantized matmul on GPU" {
    const quant = @import("../quant.zig");
    const alloc = std.testing.allocator;

    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    const be = metal.backend();

    const weights = [_]f32{ 1.0, 0.5, -0.5, 1.0, 0.25, -0.25 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var expected = [_]f32{0} ** 4;

    var qw = try quant.QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);
    qw.matmul(&input, &expected, 2, 2, 3);

    // Upload quantized weights to device.
    const w_data_buf = be.allocBuffer(qw.data.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(w_data_buf);
    const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
    be.uploadBytes(w_data_buf, 0, i8_as_u8[0..qw.data.len]);

    const w_scales_buf = be.allocSlice(f32, qw.scales.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(w_scales_buf);
    be.uploadSlice(f32, w_scales_buf, 0, qw.scales);

    const input_buf = be.allocSlice(f32, input.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(input_buf);
    be.uploadSlice(f32, input_buf, 0, &input);

    const dst_buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(dst_buf);

    const ok = be.deviceQuantizedMatMul(.{
        .dst = dst_buf,
        .input = input_buf,
        .weight = .{
            .data = w_data_buf,
            .scales = w_scales_buf,
            .rows = 3,
            .cols = 2,
            .block_size = 32,
        },
        .M = 2,
        .N = 2,
        .K = 3,
    });
    try std.testing.expect(ok);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, dst_buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &expected, &dst);
}
