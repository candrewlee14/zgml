//! WebGPU backend via wgpu-native.
//!
//! Portable GPU acceleration: Metal (macOS), Vulkan (Linux/Windows),
//! D3D12 (Windows) from a single set of WGSL shaders.
//!
//! Key difference from Metal: no shared memory. Uses wgpuQueueWriteBuffer
//! for uploads and a staging buffer + wgpuBufferMapAsync for output readback.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const program_mod = @import("program.zig");

const c = @cImport({
    @cInclude("webgpu/webgpu.h");
    @cInclude("webgpu/wgpu.h");
});

// ── Tile sizes (must match WGSL shaders) ──────────────────────────

const MATMUL_BM: u32 = 64; // matmul output rows per workgroup
const MATMUL_BN: u32 = 64; // matmul output cols per workgroup
const WG_SIZE: u32 = 256; // compute.wgsl workgroup size
const ATTN_MAX_SEQ_KV: u32 = 4096; // bounded workgroup score buffer
const ATTN_MAX_D_HEAD: u32 = 512; // bounded workgroup query cache

// ── Shader sources (embedded at comptime) ─────────────────────────

const matmul_wgsl = @embedFile("shaders/matmul.wgsl");
const matmul_f16_wgsl = @embedFile("shaders/matmul_f16.wgsl");
const qmatmul_wgsl = @embedFile("shaders/qmatmul.wgsl");
const compute_wgsl = @embedFile("shaders/compute.wgsl");
const attention_wgsl = @embedFile("shaders/attention.wgsl");

// ── Kernel param structs (must match WGSL layout, std140 rules) ───

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
    _pad: u32 = 0, // align to 16 bytes (3 vec4s → 48 bytes)
};

const MatMulF16Params = extern struct {
    M: u32,
    N: u32,
    K: u32,
    a_offset: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0, // align to 16 bytes (2 vec4s → 32 bytes)
};

const QMatMulParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
};

const ComputeParams = extern struct {
    op: u32,
    n_elements: u32,
    dst_ne: [4]u32,
    dst_strides: [4]u32,
    dst_offset: u32,
    src0_ne: [4]u32,
    src0_strides: [4]u32,
    src0_offset: u32,
    src1_ne: [4]u32,
    src1_strides: [4]u32,
    src1_offset: u32,
    _pad: u32 = 0, // align to 16 bytes
};

const AttentionParams = extern struct {
    dims0: [4]u32,
    scale_pad: [4]f32,
    offsets0: [4]u32,
    offsets1: [4]u32,
    strides0: [4]u32,
    strides1: [4]u32,
};

const ComputePipelineState = struct {
    pipeline: c.WGPUComputePipeline,
    bgl: c.WGPUBindGroupLayout,

    fn init(device: c.WGPUDevice, module: c.WGPUShaderModule) ?ComputePipelineState {
        const desc = c.WGPUComputePipelineDescriptor{
            .nextInChain = null,
            .label = .{ .data = null, .length = 0 },
            .layout = null, // auto layout
            .compute = .{
                .nextInChain = null,
                .module = module,
                .entryPoint = .{ .data = "main", .length = 4 },
                .constantCount = 0,
                .constants = null,
            },
        };
        const pipeline = c.wgpuDeviceCreateComputePipeline(device, &desc) orelse return null;
        errdefer c.wgpuComputePipelineRelease(pipeline);

        const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0) orelse return null;
        return .{
            .pipeline = pipeline,
            .bgl = bgl,
        };
    }

    fn deinit(self: ComputePipelineState) void {
        c.wgpuBindGroupLayoutRelease(self.bgl);
        c.wgpuComputePipelineRelease(self.pipeline);
    }
};

const DispatchGrid = struct {
    gx: u32,
    gy: u32 = 1,
};

const ComputeDispatchSpec = struct {
    params: ComputeParams,
    src0: u16,
    src1: u16,
    dst: u16,
    grid: DispatchGrid,
};

fn matmulGrid(M: anytype, N: anytype) DispatchGrid {
    return .{
        .gx = @intCast((N + MATMUL_BN - 1) / MATMUL_BN),
        .gy = @intCast((M + MATMUL_BM - 1) / MATMUL_BM),
    };
}

fn linearGrid(n: anytype) u32 {
    return @intCast((n + WG_SIZE - 1) / WG_SIZE);
}

fn matmulParams(geom: backend_mod.MatMulGeometry) MatMulParams {
    return .{
        .M = @intCast(geom.M),
        .N = @intCast(geom.N),
        .K = @intCast(geom.K),
        .a_row_stride = @intCast(geom.a_row_stride),
        .a_col_stride = @intCast(geom.a_col_stride),
        .b_row_stride = @intCast(geom.b_row_stride),
        .b_col_stride = @intCast(geom.b_col_stride),
        .a_offset = @intCast(geom.a_offset),
        .b_offset = @intCast(geom.b_offset),
        .dst_offset = @intCast(geom.dst_offset),
        .dst_row_stride = @intCast(geom.dst_row_stride),
    };
}

fn matmulF16Params(geom: backend_mod.MatMulGeometry) MatMulF16Params {
    return .{
        .M = @intCast(geom.M),
        .N = @intCast(geom.N),
        .K = @intCast(geom.K),
        .a_offset = @intCast(geom.a_offset),
        .dst_offset = @intCast(geom.dst_offset),
        .dst_row_stride = @intCast(geom.dst_row_stride),
    };
}

fn qmatmulParams(q: anytype, block_size: usize) QMatMulParams {
    return .{
        .M = q.M,
        .N = q.N,
        .K = q.K,
        .block_size = @intCast(block_size),
    };
}

// ── WgpuBackend ───────────────────────────────────────────────────

pub const WgpuBackend = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,

    matmul: ComputePipelineState,
    matmul_f16: ComputePipelineState,
    qmatmul: ComputePipelineState,
    compute: ComputePipelineState,
    attention: ComputePipelineState,

    // Scratch buffers for denseMatMulF32 — reused across calls, grown as needed.
    scratch_a: c.WGPUBuffer = null,
    scratch_b: c.WGPUBuffer = null,
    scratch_c: c.WGPUBuffer = null,
    scratch_staging: c.WGPUBuffer = null,
    scratch_a_size: usize = 0,
    scratch_b_size: usize = 0,
    scratch_c_size: usize = 0,
    scratch_staging_size: usize = 0,

    pub fn init() !WgpuBackend {
        // Create instance.
        const instance = c.wgpuCreateInstance(&c.WGPUInstanceDescriptor{}) orelse return error.WgpuInitFailed;
        errdefer c.wgpuInstanceRelease(instance);

        // Request adapter (synchronous via wgpu-native poll).
        var adapter: c.WGPUAdapter = null;
        _ = c.wgpuInstanceRequestAdapter(instance, &c.WGPURequestAdapterOptions{
            .powerPreference = c.WGPUPowerPreference_HighPerformance,
        }, .{
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = onAdapterReady,
            .userdata1 = @ptrCast(&adapter),
        });
        // wgpu-native fires the callback synchronously during poll.
        _ = c.wgpuInstanceProcessEvents(instance);
        if (adapter == null) return error.WgpuAdapterNotAvailable;
        errdefer c.wgpuAdapterRelease(adapter);

        // Request device (synchronous via wgpu-native poll).
        var device: c.WGPUDevice = null;
        _ = c.wgpuAdapterRequestDevice(adapter, &c.WGPUDeviceDescriptor{
            .label = .{ .data = "zgml", .length = 4 },
        }, .{
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = onDeviceReady,
            .userdata1 = @ptrCast(&device),
        });
        _ = c.wgpuInstanceProcessEvents(instance);
        if (device == null) return error.WgpuDeviceNotAvailable;
        errdefer c.wgpuDeviceRelease(device);

        const queue = c.wgpuDeviceGetQueue(device) orelse return error.WgpuInitFailed;
        errdefer c.wgpuQueueRelease(queue);

        // Compile shader modules + pipelines.
        const matmul_mod = createShaderModule(device, matmul_wgsl) orelse return error.ShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(matmul_mod);
        const matmul_f16_mod = createShaderModule(device, matmul_f16_wgsl) orelse return error.ShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(matmul_f16_mod);
        const qmatmul_mod = createShaderModule(device, qmatmul_wgsl) orelse return error.ShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(qmatmul_mod);
        const compute_mod = createShaderModule(device, compute_wgsl) orelse return error.ShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(compute_mod);
        const attention_mod = createShaderModule(device, attention_wgsl) orelse return error.ShaderCompileFailed;
        defer c.wgpuShaderModuleRelease(attention_mod);

        const matmul = ComputePipelineState.init(device, matmul_mod) orelse return error.PipelineCreateFailed;
        errdefer matmul.deinit();
        const matmul_f16 = ComputePipelineState.init(device, matmul_f16_mod) orelse return error.PipelineCreateFailed;
        errdefer matmul_f16.deinit();
        const qmatmul = ComputePipelineState.init(device, qmatmul_mod) orelse return error.PipelineCreateFailed;
        errdefer qmatmul.deinit();
        const compute = ComputePipelineState.init(device, compute_mod) orelse return error.PipelineCreateFailed;
        errdefer compute.deinit();
        const attention = ComputePipelineState.init(device, attention_mod) orelse return error.PipelineCreateFailed;

        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
            .matmul = matmul,
            .matmul_f16 = matmul_f16,
            .qmatmul = qmatmul,
            .compute = compute,
            .attention = attention,
        };
    }

    pub fn deinit(self: *WgpuBackend) void {
        if (self.scratch_a != null) c.wgpuBufferRelease(self.scratch_a);
        if (self.scratch_b != null) c.wgpuBufferRelease(self.scratch_b);
        if (self.scratch_c != null) c.wgpuBufferRelease(self.scratch_c);
        if (self.scratch_staging != null) c.wgpuBufferRelease(self.scratch_staging);
        self.attention.deinit();
        self.compute.deinit();
        self.qmatmul.deinit();
        self.matmul_f16.deinit();
        self.matmul.deinit();
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuInstanceRelease(self.instance);
    }

    pub fn backend(self: *WgpuBackend) backend_mod.Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
            .name_str = "wgpu",
            .device_type = .wgpu,
            .capabilities = backend_mod.Capabilities.wgpu,
        };
    }
};

// ── Helper: create shader module from WGSL source ─────────────────

fn createShaderModule(device: c.WGPUDevice, src: [*:0]const u8) ?c.WGPUShaderModule {
    const wgsl_desc = c.WGPUShaderSourceWGSL{
        .chain = .{ .next = null, .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = .{ .data = src, .length = std.mem.len(src) },
    };
    const desc = c.WGPUShaderModuleDescriptor{
        .nextInChain = @ptrCast(@constCast(&wgsl_desc.chain)),
    };
    return c.wgpuDeviceCreateShaderModule(device, &desc);
}

// ── Adapter/device request callbacks (wgpu-native sync pattern) ───

fn onAdapterReady(status: c.WGPURequestAdapterStatus, adapter: c.WGPUAdapter, _: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    if (status == c.WGPURequestAdapterStatus_Success) {
        const out: *c.WGPUAdapter = @ptrCast(@alignCast(userdata1));
        out.* = adapter;
    }
}

fn onDeviceReady(status: c.WGPURequestDeviceStatus, device: c.WGPUDevice, _: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    if (status == c.WGPURequestDeviceStatus_Success) {
        const out: *c.WGPUDevice = @ptrCast(@alignCast(userdata1));
        out.* = device;
    }
}

// ── VTable implementation ─────────────────────────────────────────

fn getState(ctx: *anyopaque) *WgpuBackend {
    return @ptrCast(@alignCast(ctx));
}

fn ensureScratch(self: *WgpuBackend, buf: *c.WGPUBuffer, cur_size: *usize, needed: usize, usage: c.WGPUBufferUsage) void {
    if (needed <= cur_size.*) return;
    if (buf.* != null) c.wgpuBufferRelease(buf.*);
    buf.* = createGpuBuffer(self.device, needed, usage);
    cur_size.* = needed;
}

/// Host matmul override — dispatches matmul on the GPU using scratch buffers.
fn denseMatMulF32(ctx: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const self = getState(ctx);

    const a_size = spec.a.len * 4;
    const b_size = spec.b.len * 4;
    const c_size = spec.dst.len * 4;

    // Grow scratch buffers if needed.
    const storage_dst = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst;
    const storage_dst_src = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc;
    ensureScratch(self, &self.scratch_a, &self.scratch_a_size, a_size, storage_dst);
    ensureScratch(self, &self.scratch_b, &self.scratch_b_size, b_size, storage_dst);
    ensureScratch(self, &self.scratch_c, &self.scratch_c_size, c_size, storage_dst_src);
    ensureScratch(self, &self.scratch_staging, &self.scratch_staging_size, c_size, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);

    // Upload A and B.
    c.wgpuQueueWriteBuffer(self.queue, self.scratch_a, 0, @ptrCast(spec.a.ptr), a_size);
    c.wgpuQueueWriteBuffer(self.queue, self.scratch_b, 0, @ptrCast(spec.b.ptr), b_size);

    // Create temporary uniform buffer with matmul params.
    const params = matmulParams(spec.geom);
    const ubuf = createUniformBuffer(self.device, self.queue, std.mem.asBytes(&params));
    defer c.wgpuBufferRelease(ubuf);

    // Create temporary bind group.
    const entries = [_]c.WGPUBindGroupEntry{
        bufEntry(0, self.scratch_a, self.scratch_a_size),
        bufEntry(1, self.scratch_b, self.scratch_b_size),
        bufEntry(2, self.scratch_c, self.scratch_c_size),
        bufEntry(3, ubuf, @sizeOf(MatMulParams)),
    };
    const bind_group = createBindGroup(self.device, self.matmul.bgl, &entries);
    defer c.wgpuBindGroupRelease(bind_group);

    // Encode compute pass.
    const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, &c.WGPUCommandEncoderDescriptor{
        .nextInChain = null,
        .label = .{ .data = null, .length = 0 },
    }) orelse return false;

    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, &c.WGPUComputePassDescriptor{
        .nextInChain = null,
        .label = .{ .data = null, .length = 0 },
        .timestampWrites = null,
    }) orelse {
        c.wgpuCommandEncoderRelease(encoder);
        return false;
    };

    c.wgpuComputePassEncoderSetPipeline(pass, self.matmul.pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, null);
    const grid = matmulGrid(spec.geom.M, spec.geom.N);
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, grid.gx, grid.gy, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    // Copy result to staging buffer.
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, self.scratch_c, 0, self.scratch_staging, 0, c_size);

    const cmd_buf = c.wgpuCommandEncoderFinish(encoder, &c.WGPUCommandBufferDescriptor{
        .nextInChain = null,
        .label = .{ .data = null, .length = 0 },
    });
    c.wgpuCommandEncoderRelease(encoder);

    if (cmd_buf == null) return false;
    var cmds = [_]c.WGPUCommandBuffer{cmd_buf};
    c.wgpuQueueSubmit(self.queue, 1, &cmds);
    c.wgpuCommandBufferRelease(cmd_buf);

    // Map staging buffer; the poll loop waits for both GPU completion
    // and the map in a single sync point (avoids two separate blocking waits).
    var state = MapState{};
    _ = c.wgpuBufferMapAsync(self.scratch_staging, c.WGPUMapMode_Read, 0, c_size, .{
        .mode = c.WGPUCallbackMode_AllowSpontaneous,
        .callback = onMapComplete,
        .userdata1 = @ptrCast(&state),
    });
    while (!state.done) {
        _ = c.wgpuDevicePoll(self.device, @intFromBool(true), null);
    }
    if (state.status == c.WGPUMapAsyncStatus_Success) {
        const mapped: [*]const f32 = @ptrCast(@alignCast(c.wgpuBufferGetConstMappedRange(self.scratch_staging, 0, c_size)));
        @memcpy(spec.dst, mapped[0..spec.dst.len]);
    }
    c.wgpuBufferUnmap(self.scratch_staging);

    return true;
}

// ── GPU buffer wrapper ────────────────────────────────────────────

const DeviceBuffer = struct {
    buf: c.WGPUBuffer,
    size: usize,
    f16_buf: c.WGPUBuffer = null, // f16 shadow for weight buffers
    f16_size: usize = 0,
};

const DeviceQWeight = struct {
    data: DeviceBuffer,
    scales: DeviceBuffer,
    block_size: usize,
};

// ── Pre-built dispatch (one per op, built at compile time) ────────

const PrebuiltDispatch = struct {
    pipeline: c.WGPUComputePipeline, // borrowed from WgpuBackend — not released
    bind_group: c.WGPUBindGroup, // owned — release on deinit
    uniform_buf: c.WGPUBuffer, // owned — release on deinit
    gx: u32,
    gy: u32,
};

fn releaseDispatches(dispatches: []const PrebuiltDispatch) void {
    for (dispatches) |d| {
        c.wgpuBindGroupRelease(d.bind_group);
        c.wgpuBufferRelease(d.uniform_buf);
    }
}

fn releaseDeviceBuffers(device_bufs: []const DeviceBuffer) void {
    for (device_bufs) |db| {
        if (db.f16_buf != null) c.wgpuBufferRelease(db.f16_buf);
        c.wgpuBufferRelease(db.buf);
    }
}

fn releaseQWeightViews(qweight_views: []const DeviceQWeight) void {
    for (qweight_views) |qw| {
        c.wgpuBufferRelease(qw.data.buf);
        c.wgpuBufferRelease(qw.scales.buf);
    }
}

// ── Compiled program ──────────────────────────────────────────────

const CompiledProgram = struct {
    be: *WgpuBackend,
    device_bufs: []DeviceBuffer,
    qweight_views: []DeviceQWeight,
    dispatches: []PrebuiltDispatch,
    dynamic_buf: c.WGPUBuffer,
    staging_buf: c.WGPUBuffer,
    staging_size: usize,
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledProgram) void {
        releaseDispatches(self.dispatches);
        self.alloc.free(self.dispatches);
        if (self.dynamic_buf != null) c.wgpuBufferRelease(self.dynamic_buf);
        if (self.staging_buf != null) c.wgpuBufferRelease(self.staging_buf);
        releaseDeviceBuffers(self.device_bufs);
        releaseQWeightViews(self.qweight_views);
        self.alloc.free(self.device_bufs);
        if (self.qweight_views.len > 0) self.alloc.free(self.qweight_views);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        const be = self.be;

        // Upload per-step inputs via wgpuQueueWriteBuffer.
        for (inputs) |io| {
            const db = self.device_bufs[io.buf_idx];
            c.wgpuQueueWriteBuffer(be.queue, db.buf, io.offset, io.host_ptr, io.size);
        }

        // Encode all dispatches — zero allocation, everything pre-built.
        const encoder = c.wgpuDeviceCreateCommandEncoder(be.device, &c.WGPUCommandEncoderDescriptor{
            .nextInChain = null,
            .label = .{ .data = null, .length = 0 },
        }) orelse return;

        const pass = c.wgpuCommandEncoderBeginComputePass(encoder, &c.WGPUComputePassDescriptor{
            .nextInChain = null,
            .label = .{ .data = null, .length = 0 },
            .timestampWrites = null,
        }) orelse {
            c.wgpuCommandEncoderRelease(encoder);
            return;
        };

        for (self.dispatches) |d| {
            c.wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            c.wgpuComputePassEncoderSetBindGroup(pass, 0, d.bind_group, 0, null);
            c.wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, 1);
        }

        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);

        // Copy outputs to staging buffer at sequential offsets.
        var staging_used: u64 = 0;
        for (outputs) |io| {
            const db = self.device_bufs[io.buf_idx];
            c.wgpuCommandEncoderCopyBufferToBuffer(encoder, db.buf, io.offset, self.staging_buf, staging_used, io.size);
            staging_used += std.mem.alignForward(u64, io.size, 4);
        }

        const cmd_buf = c.wgpuCommandEncoderFinish(encoder, &c.WGPUCommandBufferDescriptor{
            .nextInChain = null,
            .label = .{ .data = null, .length = 0 },
        });
        c.wgpuCommandEncoderRelease(encoder);

        if (cmd_buf == null) return;
        var cmds = [_]c.WGPUCommandBuffer{cmd_buf};
        c.wgpuQueueSubmit(be.queue, 1, &cmds);
        c.wgpuCommandBufferRelease(cmd_buf);

        if (staging_used == 0) return; // No readback — queue serializes next submit.

        // Map staging buffer; the poll loop waits for both GPU completion
        // and the map in a single sync point (avoids two separate blocking waits).
        var state = MapState{};
        _ = c.wgpuBufferMapAsync(self.staging_buf, c.WGPUMapMode_Read, 0, staging_used, .{
            .mode = c.WGPUCallbackMode_AllowSpontaneous,
            .callback = onMapComplete,
            .userdata1 = @ptrCast(&state),
        });
        while (!state.done) {
            _ = c.wgpuDevicePoll(be.device, @intFromBool(true), null);
        }
        if (state.status == c.WGPUMapAsyncStatus_Success) {
            const mapped: [*]const u8 = @ptrCast(c.wgpuBufferGetConstMappedRange(self.staging_buf, 0, staging_used));
            var read_offset: usize = 0;
            for (outputs) |io| {
                @memcpy(io.host_ptr[0..io.size], mapped[read_offset..][0..io.size]);
                read_offset += std.mem.alignForward(usize, io.size, 4);
            }
        }
        c.wgpuBufferUnmap(self.staging_buf);
    }
};

// ── GPU buffer helpers ────────────────────────────────────────────

fn createGpuBuffer(device: c.WGPUDevice, size: usize, usage: c.WGPUBufferUsage) c.WGPUBuffer {
    // WebGPU requires buffer size >= 1.
    const actual_size: u64 = @intCast(@max(size, 4));
    return c.wgpuDeviceCreateBuffer(device, &c.WGPUBufferDescriptor{
        .nextInChain = null,
        .label = .{ .data = null, .length = 0 },
        .usage = usage,
        .size = actual_size,
        .mappedAtCreation = @intFromBool(false),
    });
}

fn createUniformBuffer(device: c.WGPUDevice, queue: c.WGPUQueue, data: []const u8) c.WGPUBuffer {
    const buf = createGpuBuffer(device, data.len, c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
    c.wgpuQueueWriteBuffer(queue, buf, 0, data.ptr, data.len);
    return buf;
}

fn bufEntry(binding: u32, buffer: c.WGPUBuffer, size: usize) c.WGPUBindGroupEntry {
    return .{
        .nextInChain = null,
        .binding = binding,
        .buffer = buffer,
        .offset = 0,
        .size = @intCast(size),
        .sampler = null,
        .textureView = null,
    };
}

fn deviceBufEntry(binding: u32, db: DeviceBuffer) c.WGPUBindGroupEntry {
    return bufEntry(binding, db.buf, db.size);
}

fn createBindGroup(device: c.WGPUDevice, layout: c.WGPUBindGroupLayout, entries: []const c.WGPUBindGroupEntry) c.WGPUBindGroup {
    return c.wgpuDeviceCreateBindGroup(device, &c.WGPUBindGroupDescriptor{
        .nextInChain = null,
        .label = .{ .data = null, .length = 0 },
        .layout = layout,
        .entryCount = entries.len,
        .entries = entries.ptr,
    });
}

// ── Staging buffer map-read ───────────────────────────────────────

const MapState = struct {
    done: bool = false,
    status: c.WGPUMapAsyncStatus = 0,
};

fn onMapComplete(status: c.WGPUMapAsyncStatus, _: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    const state: *MapState = @ptrCast(@alignCast(userdata1));
    state.status = status;
    state.done = true;
}

// ── Dispatch builder ────────��─────────────────────────────────────
//
// Pre-builds a PrebuiltDispatch for each DeviceOp at compile time:
// uniform buffer (pre-filled), bind group, and workgroup counts.
// The execute loop is just setPipeline → setBindGroup → dispatch.

fn dispatchWithPipeline(
    be: *WgpuBackend,
    pipeline_state: ComputePipelineState,
    ubuf: c.WGPUBuffer,
    entries: []const c.WGPUBindGroupEntry,
    gx: u32,
    gy: u32,
) PrebuiltDispatch {
    return .{
        .pipeline = pipeline_state.pipeline,
        .bind_group = createBindGroup(be.device, pipeline_state.bgl, entries),
        .uniform_buf = ubuf,
        .gx = gx,
        .gy = gy,
    };
}

fn dispatchWithUniformParams(
    comptime ParamsType: type,
    comptime PrefixN: usize,
    comptime SuffixN: usize,
    be: *WgpuBackend,
    pipeline_state: ComputePipelineState,
    params: ParamsType,
    prefix_entries: [PrefixN]c.WGPUBindGroupEntry,
    uniform_binding: u32,
    suffix_entries: [SuffixN]c.WGPUBindGroupEntry,
    grid: DispatchGrid,
) PrebuiltDispatch {
    const ubuf = createUniformBuffer(be.device, be.queue, std.mem.asBytes(&params));
    var entries: [PrefixN + 1 + SuffixN]c.WGPUBindGroupEntry = undefined;
    @memcpy(entries[0..PrefixN], prefix_entries[0..]);
    entries[PrefixN] = bufEntry(uniform_binding, ubuf, @sizeOf(ParamsType));
    @memcpy(entries[PrefixN + 1 ..], suffix_entries[0..]);
    return dispatchWithPipeline(be, pipeline_state, ubuf, &entries, grid.gx, grid.gy);
}

fn computeDispatch(
    be: *WgpuBackend,
    bufs: []const DeviceBuffer,
    spec: ComputeDispatchSpec,
    dynamic_buf: c.WGPUBuffer,
) PrebuiltDispatch {
    return dispatchWithUniformParams(
        ComputeParams,
        3,
        1,
        be,
        be.compute,
        spec.params,
        .{
            deviceBufEntry(0, bufs[spec.src0]),
            deviceBufEntry(1, bufs[spec.src1]),
            deviceBufEntry(2, bufs[spec.dst]),
        },
        3,
        .{
            bufEntry(4, dynamic_buf, @sizeOf(program_mod.StepDynamicParams)),
        },
        spec.grid,
    );
}

fn baseComputeParams(op_code: u32, n_elements: u32, dst_offset: u32) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = op_code;
    p.n_elements = n_elements;
    p.dst_offset = dst_offset;
    return p;
}

fn rowComputeParams(op_code: u32, rows: u32, cols: u32, src_offset: u32, dst_offset: u32) ComputeParams {
    var p = baseComputeParams(op_code, rows, dst_offset);
    p.src0_ne[0] = cols;
    p.src0_offset = src_offset;
    return p;
}

fn epsilonRowComputeParams(op_code: u32, rows: u32, cols: u32, eps: f32, src_offset: u32, dst_offset: u32) ComputeParams {
    var p = rowComputeParams(op_code, rows, cols, src_offset, dst_offset);
    p.src1_ne[0] = @bitCast(eps);
    return p;
}

fn elementwiseComputeParams(e: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(e.op), e.n, e.dst_offset);
    p.src0_offset = e.src0_offset;
    p.src1_offset = e.src1_offset;
    return p;
}

fn reduceComputeParams(r: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(r.op), r.n_out, r.dst_offset);
    p.src0_ne[0] = r.reduce_size;
    p.src0_offset = r.src_offset;
    return p;
}

fn repeatComputeParams(rp: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(backend_mod.Op.repeat), rp.n, rp.dst_offset);
    p.dst_ne = rp.dst_ne;
    p.dst_strides = rp.dst_strides;
    p.src0_ne = rp.src_ne;
    p.src0_strides = rp.src_strides;
    p.src0_offset = rp.src_offset;
    return p;
}

fn isSupportedElementwiseOp(op: backend_mod.Op) bool {
    return switch (op) {
        .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => true,
        else => false,
    };
}

fn computeDispatchSpec(op: backend_mod.DeviceOp) ?ComputeDispatchSpec {
    switch (op) {
        .elementwise => |e| {
            if (!isSupportedElementwiseOp(e.op)) return null;
            return .{
                .params = elementwiseComputeParams(e),
                .src0 = e.src0,
                .src1 = e.src1,
                .dst = e.dst,
                .grid = .{ .gx = linearGrid(e.n) },
            };
        },
        .softmax => |s| return .{
            .params = rowComputeParams(program_mod.compute_op_softmax, s.rows, s.cols, s.src_offset, s.dst_offset),
            .src0 = s.src,
            .src1 = s.src,
            .dst = s.dst,
            .grid = .{ .gx = linearGrid(s.rows) },
        },
        .layernorm => |l| return .{
            .params = epsilonRowComputeParams(program_mod.compute_op_layernorm, l.rows, l.cols, l.eps, l.src_offset, l.dst_offset),
            .src0 = l.src,
            .src1 = l.src,
            .dst = l.dst,
            .grid = .{ .gx = linearGrid(l.rows) },
        },
        .rmsnorm => |r| return .{
            .params = epsilonRowComputeParams(program_mod.compute_op_rmsnorm, r.rows, r.cols, r.eps, r.src_offset, r.dst_offset),
            .src0 = r.src,
            .src1 = r.src,
            .dst = r.dst,
            .grid = .{ .gx = linearGrid(r.rows) },
        },
        .reduce => |r| return .{
            .params = reduceComputeParams(r),
            .src0 = r.src,
            .src1 = r.dst,
            .dst = r.dst,
            .grid = .{ .gx = linearGrid(r.n_out) },
        },
        .repeat => |rp| return .{
            .params = repeatComputeParams(rp),
            .src0 = rp.src,
            .src1 = rp.src,
            .dst = rp.dst,
            .grid = .{ .gx = linearGrid(rp.n) },
        },
        .slice_assign => |sa| {
            const params = sliceAssignComputeParams(sa);
            return .{
                .params = params,
                .src0 = sa.src,
                .src1 = sa.src,
                .dst = sa.dst,
                .grid = .{ .gx = linearGrid(params.n_elements) },
            };
        },
        .rope => |rr| {
            const params = ropeComputeParams(rr);
            return .{
                .params = params,
                .src0 = rr.src,
                .src1 = rr.cos_sin,
                .dst = rr.dst,
                .grid = .{ .gx = linearGrid(params.n_elements) },
            };
        },
        else => return null,
    }
}

fn sliceAssignComputeParams(sa: anytype) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = @intFromEnum(backend_mod.Op.slice_assign);
    p.n_elements = sa.rows * sa.cols;
    p.src0_ne[0] = sa.rows;
    p.dst_strides[0] = sa.dst_row_stride;
    p.dst_strides[1] = sa.dst_col_stride;
    p.dst_offset = sa.dst_base_offset;
    p.src0_strides[0] = sa.src_row_stride;
    p.src0_strides[1] = sa.src_col_stride;
    p.src0_offset = sa.src_offset;
    p.src1_ne[0] = sa.patch_stride;
    return p;
}

fn ropeComputeParams(rr: anytype) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = @intFromEnum(backend_mod.Op.rope);
    p.n_elements = rr.half_d * rr.seq_len;
    p.dst_offset = rr.dst_off;
    p.src0_ne[0] = rr.half_d;
    p.src0_ne[1] = rr.seq_len;
    p.src0_strides[0] = rr.src_rs;
    p.src0_strides[1] = rr.src_cs;
    p.src0_offset = rr.src_off;
    p.src1_strides[0] = rr.cs_cs;
    p.src1_offset = rr.cs_off;
    return p;
}

fn attentionParams(att: anytype) AttentionParams {
    return .{
        .dims0 = .{ att.d_head, att.seq_q, @intFromBool(att.has_mask), 0 },
        .scale_pad = .{ att.scale, 0, 0, 0 },
        .offsets0 = .{ att.q_off, att.k_off, att.v_off, att.mask_off },
        .offsets1 = .{ att.dst_off, att.q_rs, att.q_cs, 0 },
        .strides0 = .{ att.k_rs, att.k_cs, att.v_rs, att.v_cs },
        .strides1 = .{ att.mask_rs, att.mask_cs, att.dst_rs, att.dst_cs },
    };
}

fn buildDispatch(
    be: *WgpuBackend,
    bufs: []const DeviceBuffer,
    qweights: []const DeviceQWeight,
    op: backend_mod.DeviceOp,
    dynamic_buf: c.WGPUBuffer,
) ?PrebuiltDispatch {
    if (computeDispatchSpec(op)) |spec| {
        return computeDispatch(be, bufs, spec, dynamic_buf);
    }
    switch (op) {
        .matmul => |m| {
            // F16 weight path: B buffer has a pre-packed f16 shadow.
            if (bufs[m.b].f16_buf) |f16_b| {
                const grid = matmulGrid(m.geom.M, m.geom.N);
                const params = matmulF16Params(m.geom);
                return dispatchWithUniformParams(
                    MatMulF16Params,
                    3,
                    0,
                    be,
                    be.matmul_f16,
                    params,
                    .{
                        deviceBufEntry(0, bufs[m.a]),
                        bufEntry(1, f16_b, bufs[m.b].f16_size),
                        deviceBufEntry(2, bufs[m.dst]),
                    },
                    3,
                    .{},
                    grid,
                );
            }
            // F32 path.
            const grid = matmulGrid(m.geom.M, m.geom.N);
            const params = matmulParams(m.geom);
            return dispatchWithUniformParams(
                MatMulParams,
                3,
                0,
                be,
                be.matmul,
                params,
                .{
                    deviceBufEntry(0, bufs[m.a]),
                    deviceBufEntry(1, bufs[m.b]),
                    deviceBufEntry(2, bufs[m.dst]),
                },
                3,
                .{},
                grid,
            );
        },
        .qmatmul => |q| {
            const w = qweights[q.weight_idx];
            const grid = matmulGrid(q.M, q.N);
            const params = qmatmulParams(q, w.block_size);
            return dispatchWithUniformParams(
                QMatMulParams,
                4,
                0,
                be,
                be.qmatmul,
                params,
                .{
                    deviceBufEntry(0, w.data),
                    deviceBufEntry(1, w.scales),
                    deviceBufEntry(2, bufs[q.input]),
                    deviceBufEntry(3, bufs[q.dst]),
                },
                4,
                .{},
                grid,
            );
        },
        .attention => |att| {
            if (att.seq_kv > ATTN_MAX_SEQ_KV or att.d_head > ATTN_MAX_D_HEAD) return null;

            const params = attentionParams(att);
            return dispatchWithUniformParams(
                AttentionParams,
                5,
                1,
                be,
                be.attention,
                params,
                .{
                    deviceBufEntry(0, bufs[att.q]),
                    deviceBufEntry(1, bufs[att.k]),
                    deviceBufEntry(2, bufs[att.v]),
                    deviceBufEntry(3, bufs[att.mask]),
                    deviceBufEntry(4, bufs[att.dst]),
                },
                5,
                .{
                    bufEntry(6, dynamic_buf, @sizeOf(program_mod.StepDynamicParams)),
                },
                .{ .gx = att.seq_q },
            );
        },
        .elementwise, .softmax, .layernorm, .rmsnorm, .reduce, .repeat, .slice_assign, .rope => return null,
        // Fused/batched ops: fall back to individual dispatches or skip.
        .fused_elementwise => return null,
    }
}

// ── VTable functions ──────────────────────────────────────────────

fn compileProgramFn(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = std.heap.page_allocator;

    // Allocate device buffers (Storage + CopyDst + CopySrc).
    const device_bufs = alloc.alloc(DeviceBuffer, program.n_buffers) catch return null;
    errdefer alloc.free(device_bufs);
    var n_device_bufs: usize = 0;
    errdefer releaseDeviceBuffers(device_bufs[0..n_device_bufs]);
    for (device_bufs, program.buffer_sizes) |*db, size| {
        const byte_size = size * @sizeOf(f32);
        const buf = createGpuBuffer(
            self.device,
            byte_size,
            c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc,
        );
        if (buf == null) return null;
        db.* = .{ .buf = buf, .size = byte_size };
        n_device_bufs += 1;
    }

    // Upload initial data (weights, KV cache zeros).
    for (program.initial_uploads) |io| {
        const db = device_bufs[io.buf_idx];
        c.wgpuQueueWriteBuffer(self.queue, db.buf, io.offset, io.host_ptr, io.size);
    }

    // Upload quantized weights.
    const qweight_views = alloc.alloc(DeviceQWeight, program.qweights.len) catch return null;
    errdefer {
        if (qweight_views.len > 0) alloc.free(qweight_views);
    }
    var n_qweight_views: usize = 0;
    errdefer releaseQWeightViews(qweight_views[0..n_qweight_views]);
    for (program.qweights, 0..) |qw, i| {
        const data_buf = createGpuBuffer(
            self.device,
            qw.data.len,
            c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst,
        );
        if (data_buf == null) return null;
        const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
        c.wgpuQueueWriteBuffer(self.queue, data_buf, 0, i8_as_u8, qw.data.len);

        const scales_size = qw.scales.len * @sizeOf(f32);
        const scales_buf = createGpuBuffer(
            self.device,
            scales_size,
            c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst,
        );
        if (scales_buf == null) return null;
        c.wgpuQueueWriteBuffer(self.queue, scales_buf, 0, @ptrCast(qw.scales.ptr), scales_size);

        qweight_views[i] = .{
            .data = .{ .buf = data_buf, .size = qw.data.len },
            .scales = .{ .buf = scales_buf, .size = scales_size },
            .block_size = qw.block_size,
        };
        n_qweight_views += 1;
    }

    // Auto-promote weight buffers to f16: detect matmul B operands with initial uploads.
    for (program.ops) |op| {
        const b_idx = switch (op) {
            .matmul => |m| m.b,
            else => continue,
        };
        // Find the initial upload for this buffer (if any).
        const upload = for (program.initial_uploads) |io| {
            if (io.buf_idx == b_idx) break io;
        } else continue;
        // Already promoted? (same buffer used in multiple matmuls)
        if (device_bufs[b_idx].f16_buf != null) continue;

        // Pack f32 → f16 using matmul geometry for stride-aware reads.
        const geom = op.matmul.geom;
        const n_elems = geom.K * geom.N;
        const f16_bytes = std.mem.alignForward(usize, n_elems * 2, 4);
        const f16_tmp = alloc.alloc(u16, n_elems + (n_elems & 1)) catch continue;
        defer alloc.free(f16_tmp);

        const f32_ptr: [*]const f32 = @ptrCast(@alignCast(upload.host_ptr));
        for (0..geom.K) |row| {
            for (0..geom.N) |col| {
                const src_idx = geom.b_offset + row * geom.b_row_stride + col * geom.b_col_stride;
                f16_tmp[row * geom.N + col] = @bitCast(@as(f16, @floatCast(f32_ptr[src_idx])));
            }
        }
        if (n_elems & 1 != 0) f16_tmp[n_elems] = 0;

        const f16_gpu = createGpuBuffer(self.device, f16_bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
        if (f16_gpu == null) continue;
        c.wgpuQueueWriteBuffer(self.queue, f16_gpu, 0, @ptrCast(f16_tmp.ptr), f16_bytes);
        device_bufs[b_idx].f16_buf = f16_gpu;
        device_bufs[b_idx].f16_size = f16_bytes;
    }

    const dynamic_state = program_mod.stepDynamicStateFromOps(program.ops);
    const dynamic_buf = createUniformBuffer(self.device, self.queue, std.mem.asBytes(&dynamic_state.params));
    errdefer if (dynamic_buf != null) c.wgpuBufferRelease(dynamic_buf);

    // Pre-build all dispatches (uniform buffers + bind groups).
    var dispatch_list: std.ArrayListUnmanaged(PrebuiltDispatch) = .empty;
    errdefer {
        releaseDispatches(dispatch_list.items);
        dispatch_list.deinit(alloc);
    }
    for (program.ops) |op| {
        if (buildDispatch(self, device_bufs, qweight_views, op, dynamic_buf)) |d| {
            dispatch_list.append(alloc, d) catch return null;
        } else {
            return null;
        }
    }
    const dispatches = dispatch_list.toOwnedSlice(alloc) catch return null;
    dispatch_list = .empty;
    errdefer alloc.free(dispatches);
    errdefer releaseDispatches(dispatches);

    // Staging buffer — conservatively sized to the largest device buffer.
    var max_buf_size: usize = 0;
    for (program.buffer_sizes) |s| {
        const bs = s * @sizeOf(f32);
        if (bs > max_buf_size) max_buf_size = bs;
    }

    const staging = createGpuBuffer(
        self.device,
        max_buf_size,
        c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst,
    );
    errdefer if (staging != null) c.wgpuBufferRelease(staging);

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .be = self,
        .device_bufs = device_bufs,
        .qweight_views = qweight_views,
        .dispatches = dispatches,
        .dynamic_buf = dynamic_buf,
        .staging_buf = staging,
        .staging_size = max_buf_size,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn executeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn refreshProgramFn(ctx: *anyopaque, handle: backend_mod.Backend.CompiledHandle, ops: []const backend_mod.DeviceOp) void {
    const self = getState(ctx);
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const dynamic_state = program_mod.stepDynamicStateFromOps(ops);
    if (compiled.dynamic_buf != null and dynamic_state.needsUpload()) {
        c.wgpuQueueWriteBuffer(self.queue, compiled.dynamic_buf, 0, std.mem.asBytes(&dynamic_state.params).ptr, @sizeOf(program_mod.StepDynamicParams));
    }
}

fn freeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn getRuntimeProfileFn(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) ?*@import("../profile.zig").RuntimeProfile {
    return null;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgramFn,
    .refresh_program = refreshProgramFn,
    .execute_program = executeProgramFn,
    .free_program = freeProgramFn,
    .get_runtime_profile = getRuntimeProfileFn,
};

test "wgpu backend compute dispatch spec groups compute ops only" {
    const layernorm = backend_mod.DeviceOp{ .layernorm = .{
        .dst = 3,
        .src = 1,
        .rows = 8,
        .cols = 16,
        .eps = 1e-5,
        .src_offset = 4,
        .dst_offset = 12,
    } };
    const spec = computeDispatchSpec(layernorm).?;
    try std.testing.expectEqual(@as(u16, 1), spec.src0);
    try std.testing.expectEqual(@as(u16, 1), spec.src1);
    try std.testing.expectEqual(@as(u16, 3), spec.dst);
    try std.testing.expectEqual(@as(u32, program_mod.compute_op_layernorm), spec.params.op);
    try std.testing.expectEqual(@as(u32, 8), spec.params.n_elements);
    try std.testing.expectEqual(@as(u32, 16), spec.params.src0_ne[0]);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1e-5))), spec.params.src1_ne[0]);
    try std.testing.expectEqual(@as(u32, linearGrid(8)), spec.grid.gx);
    try std.testing.expectEqual(@as(u32, 1), spec.grid.gy);

    const unsupported = backend_mod.DeviceOp{ .elementwise = .{
        .op = .none,
        .dst = 0,
        .src0 = 1,
        .src1 = 2,
        .n = 4,
    } };
    try std.testing.expect(computeDispatchSpec(unsupported) == null);

    const matmul = backend_mod.DeviceOp{ .matmul = .{
        .dst = 0,
        .a = 1,
        .b = 2,
        .geom = .{
            .M = 1,
            .N = 1,
            .K = 1,
            .a_row_stride = 1,
            .a_col_stride = 1,
            .b_row_stride = 1,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
        },
    } };
    try std.testing.expect(computeDispatchSpec(matmul) == null);
}

// ── Tests ─────────────────────────────────────────────────────────

test "wgpu backend compiled program matmul" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    // Program: buf0(A) × buf1(B) → buf2(dst). 2x3 × 3x2 = 2x2.
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 6, 6, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&a_data), .size = 6 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&b_data), .size = 6 * 4 },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "wgpu backend f16 matmul (auto-promoted)" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    // Program: buf0(A) × buf1(B) → buf2(dst). 2x3 × 3x2 = 2x2.
    // B has an initial upload, so the backend auto-promotes it to f16.
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // Expected: [[58,64],[139,154]]
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 6, 6, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&a_data), .size = 6 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&b_data), .size = 6 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 3,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst_buf: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst_buf), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    // f16 has limited precision, but small integers are exact.
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst_buf);
}

test "wgpu backend elementwise add" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 10, 20, 30, 40 };
    const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const buf_sizes = [_]usize{ 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&a_data), .size = 4 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&b_data), .size = 4 * 4 },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, &dst);
}

test "wgpu backend rope op" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    var src_data = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var cs_data = [_]f32{
        1, 1, 0, 0,
        0, 0, 1, 1,
    };

    const ops = [_]backend_mod.DeviceOp{.{ .rope = .{
        .dst = 2,
        .src = 0,
        .cos_sin = 1,
        .half_d = 2,
        .seq_len = 2,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 4,
    } }};
    const buf_sizes = [_]usize{ 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&src_data), .size = src_data.len * @sizeOf(f32) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&cs_data), .size = cs_data.len * @sizeOf(f32) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = dst.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, -7, -8, 5, 6 }, &dst);
}

test "wgpu backend slice assign refreshes position" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    var src_data = [_]f32{ 1, 2, 3, 4 };
    var dst_data = [_]f32{0} ** 16;

    var ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 4,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } }};
    const buf_sizes = [_]usize{ src_data.len, dst_data.len };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&src_data), .size = src_data.len * @sizeOf(f32) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&dst_data), .size = dst_data.len * @sizeOf(f32) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst = [_]f32{0} ** 16;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&dst), .size = dst.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &out);
    try std.testing.expectEqualSlices(f32, &.{
        0, 0, 0, 0,
        1, 2, 3, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
    }, &dst);

    ops[0].slice_assign.dst_offset = 12;
    be.refreshProgram(handle, &ops);
    be.executeProgram(handle, &.{}, &out);
    try std.testing.expectEqualSlices(f32, &.{
        0, 0, 0, 0,
        1, 2, 3, 4,
        0, 0, 0, 0,
        1, 2, 3, 4,
    }, &dst);
}

test "wgpu backend attention op refreshes seq_kv" {
    var wgpu = WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    const be = wgpu.backend();

    var q_data = [_]f32{
        1.0, 0.0, 1.0, 0.0,
        0.5, 1.0, 0.0, 1.0,
    };
    var k_data = [_]f32{
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0,
    };
    var v_data = [_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
    };
    var mask_data = [_]f32{
        0.0,                0.0,                0.0,                -std.math.inf(f32),
        -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32),
    };

    var ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 2,
        .seq_kv = 4,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 4,
        .dst_rs = 1,
        .dst_cs = 4,
    } }};
    const buf_sizes = [_]usize{ q_data.len, k_data.len, v_data.len, mask_data.len, q_data.len };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_data), .size = q_data.len * @sizeOf(f32) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&k_data), .size = k_data.len * @sizeOf(f32) },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&v_data), .size = v_data.len * @sizeOf(f32) },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&mask_data), .size = mask_data.len * @sizeOf(f32) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const Ref = struct {
        fn run(dst: []f32, seq_kv: usize, q: []const f32, k: []const f32, v: []const f32, mask: []const f32) void {
            const neg_inf = -std.math.inf(f32);
            const d_head = 4;
            const seq_q = 2;
            const scale: f32 = 0.5;
            for (0..seq_q) |qi| {
                const q_off = qi * d_head;
                var max_score = neg_inf;
                var weights: [4]f32 = .{0} ** 4;
                for (0..seq_kv) |s| {
                    const mask_add = mask[qi * 4 + s];
                    if (!std.math.isFinite(mask_add)) {
                        weights[s] = neg_inf;
                        continue;
                    }
                    const k_off = s * d_head;
                    var dot: f32 = 0.0;
                    for (0..d_head) |r| dot += q[q_off + r] * k[k_off + r];
                    const score = dot * scale + mask_add;
                    weights[s] = score;
                    max_score = @max(max_score, score);
                }
                var sum: f32 = 0.0;
                if (std.math.isFinite(max_score)) {
                    for (0..seq_kv) |s| {
                        if (weights[s] == neg_inf) {
                            weights[s] = 0.0;
                        } else {
                            weights[s] = @exp(weights[s] - max_score);
                            sum += weights[s];
                        }
                    }
                } else {
                    @memset(weights[0..seq_kv], 0.0);
                }
                const inv_sum: f32 = if (sum > 0.0) 1.0 / sum else 0.0;
                for (0..d_head) |r| {
                    var acc: f32 = 0.0;
                    for (0..seq_kv) |s| {
                        acc += weights[s] * inv_sum * v[s * d_head + r];
                    }
                    dst[qi * d_head + r] = acc;
                }
            }
        }
    };

    var expected_full = [_]f32{0} ** 8;
    Ref.run(&expected_full, 4, &q_data, &k_data, &v_data, &mask_data);

    var dst_full = [_]f32{0} ** 8;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 4, .host_ptr = @ptrCast(&dst_full), .size = dst_full.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &out);

    for (dst_full, expected_full) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }

    ops[0].attention.seq_kv = 2;
    be.refreshProgram(handle, &ops);

    var expected_short = [_]f32{0} ** 8;
    Ref.run(&expected_short, 2, &q_data, &k_data, &v_data, &mask_data);

    var dst_short = [_]f32{0} ** 8;
    out[0].host_ptr = @ptrCast(&dst_short);
    be.executeProgram(handle, &.{}, &out);

    for (dst_short, expected_short) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }
}
