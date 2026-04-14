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
    \\
    \\// ── Generic compute kernels ──────────────────────────────
    \\
    \\struct ComputeParams {
    \\    uint op;
    \\    uint n_elements;
    \\    uint4 dst_ne;    uint4 dst_strides;   uint dst_offset;
    \\    uint4 src0_ne;   uint4 src0_strides;   uint src0_offset;
    \\    uint4 src1_ne;   uint4 src1_strides;   uint src1_offset;
    \\};
    \\
    \\// Op enum values (must match op.zig exactly):
    \\//   add=7 mul=8 neg=9 sqrt=13 recip=14 exp=15 gelu=17
    \\//   sum=18 max=19 repeat=20 slice_assign=27
    \\
    \\// Elementwise: contiguous fast path, one thread per element.
    \\kernel void elementwise_f32(
    \\    device const float* src0 [[buffer(0)]],
    \\    device const float* src1 [[buffer(1)]],
    \\    device float* dst        [[buffer(2)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    float a = src0[p.src0_offset + gid];
    \\    switch (p.op) {
    \\        case 7:  dst[p.dst_offset + gid] = a + src1[p.src1_offset + gid]; break;
    \\        case 8:  dst[p.dst_offset + gid] = a * src1[p.src1_offset + gid]; break;
    \\        case 9:  dst[p.dst_offset + gid] = -a; break;
    \\        case 13: dst[p.dst_offset + gid] = sqrt(a); break;
    \\        case 14: dst[p.dst_offset + gid] = 1.0f / a; break;
    \\        case 15: dst[p.dst_offset + gid] = exp(a); break;
    \\        case 17: {
    \\            float c = 0.7978845608f * (a + 0.044715f * a * a * a);
    \\            dst[p.dst_offset + gid] = 0.5f * a * (1.0f + precise::tanh(c));
    \\            break;
    \\        }
    \\        default: break;
    \\    }
    \\}
    \\
    \\// Reduce: sum or max along innermost contiguous dimension.
    \\kernel void reduce_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst       [[buffer(1)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    uint reduce_size = p.src0_ne[0];
    \\    uint src_base = p.src0_offset + gid * reduce_size;
    \\    float val = (p.op == 19) ? -INFINITY : 0.0f; // 19=max, 18=sum
    \\    for (uint k = 0; k < reduce_size; k++) {
    \\        float v = src[src_base + k];
    \\        if (p.op == 18) val += v;
    \\        else val = max(val, v);
    \\    }
    \\    dst[p.dst_offset + gid] = val;
    \\}
    \\
    \\// Repeat: broadcast src to dst shape via modular indexing.
    \\kernel void repeat_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst       [[buffer(1)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    uint idx = gid;
    \\    uint src_idx = p.src0_offset;
    \\    for (int d = 3; d >= 0; d--) {
    \\        uint ud = uint(d);
    \\        uint coord = idx / p.dst_strides[ud];
    \\        idx %= p.dst_strides[ud];
    \\        src_idx += (coord % p.src0_ne[ud]) * p.src0_strides[ud];
    \\    }
    \\    dst[p.dst_offset + gid] = src[src_idx];
    \\}
    \\
    \\// Slice assign: write column into KV cache at dst_offset.
    \\kernel void slice_assign_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst       [[buffer(1)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    dst[p.dst_offset + gid * p.dst_strides[0]] = src[p.src0_offset + gid * p.src0_strides[0]];
    \\}
    \\
    \\// ── Fused kernels ────────────────────────────────────────
    \\
    \\struct FusedParams { uint rows; uint cols; uint src_offset; uint dst_offset; };
    \\
    \\// Fused softmax: max → shift → exp → sum → normalize. One thread per row.
    \\kernel void fused_softmax_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst       [[buffer(1)]],
    \\    constant FusedParams& p [[buffer(2)]],
    \\    uint rid [[thread_position_in_grid]]
    \\) {
    \\    if (rid >= p.rows) return;
    \\    uint src_base = p.src_offset + rid * p.cols;
    \\    uint dst_base = p.dst_offset + rid * p.cols;
    \\    float m = -INFINITY;
    \\    for (uint j = 0; j < p.cols; j++) m = max(m, src[src_base + j]);
    \\    float s = 0.0f;
    \\    for (uint j = 0; j < p.cols; j++) {
    \\        float e = exp(src[src_base + j] - m);
    \\        dst[dst_base + j] = e;
    \\        s += e;
    \\    }
    \\    float inv = 1.0f / s;
    \\    for (uint j = 0; j < p.cols; j++) dst[dst_base + j] *= inv;
    \\}
    \\
    \\// Fused layer norm: mean → center → var → normalize → scale + shift.
    \\// One thread per row. gamma at buffer(2), beta at buffer(3).
    \\kernel void fused_layernorm_f32(
    \\    device const float* src   [[buffer(0)]],
    \\    device float* dst         [[buffer(1)]],
    \\    device const float* gamma [[buffer(2)]],
    \\    device const float* beta  [[buffer(3)]],
    \\    constant FusedParams& p   [[buffer(4)]],
    \\    uint rid [[thread_position_in_grid]]
    \\) {
    \\    if (rid >= p.rows) return;
    \\    uint base = p.src_offset + rid * p.cols;
    \\    uint dbase = p.dst_offset + rid * p.cols;
    \\    float mu = 0.0f;
    \\    for (uint j = 0; j < p.cols; j++) mu += src[base + j];
    \\    mu /= float(p.cols);
    \\    float v = 0.0f;
    \\    for (uint j = 0; j < p.cols; j++) {
    \\        float d = src[base + j] - mu;
    \\        v += d * d;
    \\    }
    \\    float inv_std = 1.0f / sqrt(v / float(p.cols) + 1e-5f);
    \\    for (uint j = 0; j < p.cols; j++)
    \\        dst[dbase + j] = (src[base + j] - mu) * inv_std * gamma[j] + beta[j];
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

const FusedParams = extern struct {
    rows: u32,
    cols: u32,
    src_offset: u32,
    dst_offset: u32,
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
};

// ── MetalBackend ──────────────────────────────────────────────────

pub const MetalBackend = struct {
    device: *anyopaque,
    queue: *anyopaque,
    matmul_pipeline: *anyopaque,
    qmatmul_pipeline: *anyopaque,
    elementwise_pipeline: *anyopaque,
    reduce_pipeline: *anyopaque,
    repeat_pipeline: *anyopaque,
    slice_assign_pipeline: *anyopaque,
    fused_softmax_pipeline: *anyopaque,
    fused_layernorm_pipeline: *anyopaque,
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
        errdefer c.mtl_release(qmatmul_pipeline);
        const elementwise_pipeline = c.mtl_create_pipeline(device, library, "elementwise_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(elementwise_pipeline);
        const reduce_pipeline = c.mtl_create_pipeline(device, library, "reduce_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(reduce_pipeline);
        const repeat_pipeline = c.mtl_create_pipeline(device, library, "repeat_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(repeat_pipeline);
        const slice_assign_pipeline = c.mtl_create_pipeline(device, library, "slice_assign_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(slice_assign_pipeline);
        const fused_softmax_pipeline = c.mtl_create_pipeline(device, library, "fused_softmax_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(fused_softmax_pipeline);
        const fused_layernorm_pipeline = c.mtl_create_pipeline(device, library, "fused_layernorm_f32") orelse return error.PipelineCreateFailed;

        return .{
            .device = device,
            .queue = queue,
            .matmul_pipeline = matmul_pipeline,
            .qmatmul_pipeline = qmatmul_pipeline,
            .elementwise_pipeline = elementwise_pipeline,
            .reduce_pipeline = reduce_pipeline,
            .repeat_pipeline = repeat_pipeline,
            .slice_assign_pipeline = slice_assign_pipeline,
            .fused_softmax_pipeline = fused_softmax_pipeline,
            .fused_layernorm_pipeline = fused_layernorm_pipeline,
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
        c.mtl_release(self.fused_layernorm_pipeline);
        c.mtl_release(self.fused_softmax_pipeline);
        c.mtl_release(self.slice_assign_pipeline);
        c.mtl_release(self.repeat_pipeline);
        c.mtl_release(self.reduce_pipeline);
        c.mtl_release(self.elementwise_pipeline);
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

// ── Compiled program execution ────────────────────────────────────

const CompiledProgram = struct {
    backend: *MetalBackend,
    device_bufs: []backend_mod.DeviceBuffer,
    qweight_views: []backend_mod.DeviceQuantizedWeightView,
    ops: []const backend_mod.DeviceOp,
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledProgram) void {
        for (self.device_bufs) |buf| c.mtl_release(buf.ptr);
        self.alloc.free(self.device_bufs);
        if (self.qweight_views.len > 0) self.alloc.free(self.qweight_views);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        const be = self.backend;

        // Upload per-step inputs (token embed, pos, mask).
        for (inputs) |io| {
            const buf = self.device_bufs[io.buf_idx];
            const ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(buf.ptr));
            @memcpy(ptr[io.offset..][0..io.size], io.host_ptr[0..io.size]);
        }

        // Encode all ops into one command buffer.
        const cmds = be.ensureCommands();
        for (self.ops) |op| self.encodeOp(cmds, op);
        be.flushCommands();

        // Download outputs (logits).
        for (outputs) |io| {
            const buf = self.device_bufs[io.buf_idx];
            const ptr: [*]const u8 = @ptrCast(c.mtl_buffer_contents(buf.ptr));
            @memcpy(io.host_ptr[0..io.size], ptr[io.offset..][0..io.size]);
        }
    }

    fn encodeOp(self: *CompiledProgram, cmds: *anyopaque, op: backend_mod.DeviceOp) void {
        const be = self.backend;
        switch (op) {
            .matmul => |m| {
                const params = MatMulParams{
                    .M = @intCast(m.geom.M), .N = @intCast(m.geom.N), .K = @intCast(m.geom.K),
                    .a_row_stride = @intCast(m.geom.a_row_stride), .a_col_stride = @intCast(m.geom.a_col_stride),
                    .b_row_stride = @intCast(m.geom.b_row_stride), .b_col_stride = @intCast(m.geom.b_col_stride),
                    .a_offset = @intCast(m.geom.a_offset), .b_offset = @intCast(m.geom.b_offset),
                    .dst_offset = @intCast(m.geom.dst_offset), .dst_row_stride = @intCast(m.geom.dst_row_stride),
                };
                var bufs = [_]?*anyopaque{ self.device_bufs[m.a].ptr, self.device_bufs[m.b].ptr, self.device_bufs[m.dst].ptr };
                const grid_x: u32 = @intCast((m.geom.N + TILE - 1) / TILE);
                const grid_y: u32 = @intCast((m.geom.M + TILE - 1) / TILE);
                c.mtl_encode_dispatch(cmds, be.matmul_pipeline, @ptrCast(&bufs), 3, &params, @sizeOf(MatMulParams), 3, grid_x, grid_y, 128, 1);
            },
            .qmatmul => |q| {
                const w = self.qweight_views[q.weight_idx];
                const params = QMatMulParams{ .M = q.M, .N = q.N, .K = q.K, .block_size = @intCast(w.block_size) };
                var bufs = [_]?*anyopaque{ w.data.ptr, w.scales.ptr, self.device_bufs[q.input].ptr, self.device_bufs[q.dst].ptr };
                const grid_x: u32 = (q.N + TILE - 1) / TILE;
                const grid_y: u32 = (q.M + TILE - 1) / TILE;
                c.mtl_encode_dispatch(cmds, be.qmatmul_pipeline, @ptrCast(&bufs), 4, &params, @sizeOf(QMatMulParams), 4, grid_x, grid_y, 128, 1);
            },
            .elementwise => |e| {
                const pipeline = switch (e.op) {
                    .add, .mul, .neg, .exp, .sqrt, .recip, .gelu => be.elementwise_pipeline,
                    else => return,
                };
                const params = ComputeParams{
                    .op = @intFromEnum(e.op), .n_elements = e.n,
                    .dst_ne = .{ 0, 0, 0, 0 }, .dst_strides = .{ 0, 0, 0, 0 }, .dst_offset = e.dst_offset,
                    .src0_ne = .{ 0, 0, 0, 0 }, .src0_strides = .{ 0, 0, 0, 0 }, .src0_offset = e.src0_offset,
                    .src1_ne = .{ 0, 0, 0, 0 }, .src1_strides = .{ 0, 0, 0, 0 }, .src1_offset = e.src1_offset,
                };
                var bufs = [_]?*anyopaque{ self.device_bufs[e.src0].ptr, self.device_bufs[e.src1].ptr, self.device_bufs[e.dst].ptr };
                const grid: u32 = (e.n + 255) / 256;
                c.mtl_encode_dispatch(cmds, pipeline, @ptrCast(&bufs), 3, &params, @sizeOf(ComputeParams), 3, grid, 1, 256, 1);
            },
            .softmax => |s| {
                const params = FusedParams{ .rows = s.rows, .cols = s.cols, .src_offset = s.src_offset, .dst_offset = s.dst_offset };
                var bufs = [_]?*anyopaque{ self.device_bufs[s.src].ptr, self.device_bufs[s.dst].ptr };
                const grid: u32 = (s.rows + 255) / 256;
                c.mtl_encode_dispatch(cmds, be.fused_softmax_pipeline, @ptrCast(&bufs), 2, &params, @sizeOf(FusedParams), 2, grid, 1, @min(s.rows, 256), 1);
            },
            .layernorm => |l| {
                const params = FusedParams{ .rows = l.rows, .cols = l.cols, .src_offset = l.src_offset, .dst_offset = l.dst_offset };
                var bufs = [_]?*anyopaque{ self.device_bufs[l.src].ptr, self.device_bufs[l.dst].ptr, self.device_bufs[l.src].ptr, self.device_bufs[l.src].ptr };
                const grid: u32 = (l.rows + 255) / 256;
                c.mtl_encode_dispatch(cmds, be.fused_layernorm_pipeline, @ptrCast(&bufs), 4, &params, @sizeOf(FusedParams), 4, grid, 1, @min(l.rows, 256), 1);
            },
            .reduce => |r| {
                const params = ComputeParams{
                    .op = @intFromEnum(r.op), .n_elements = r.n_out,
                    .dst_ne = .{ 0, 0, 0, 0 }, .dst_strides = .{ 0, 0, 0, 0 }, .dst_offset = r.dst_offset,
                    .src0_ne = .{ r.reduce_size, 0, 0, 0 }, .src0_strides = .{ 0, 0, 0, 0 }, .src0_offset = r.src_offset,
                    .src1_ne = .{ 0, 0, 0, 0 }, .src1_strides = .{ 0, 0, 0, 0 }, .src1_offset = 0,
                };
                var bufs = [_]?*anyopaque{ self.device_bufs[r.src].ptr, self.device_bufs[r.dst].ptr };
                const grid: u32 = (r.n_out + 255) / 256;
                c.mtl_encode_dispatch(cmds, be.reduce_pipeline, @ptrCast(&bufs), 2, &params, @sizeOf(ComputeParams), 3, grid, 1, 256, 1);
            },
            .repeat => |rp| {
                const params = ComputeParams{
                    .op = @intFromEnum(backend_mod.Op.repeat), .n_elements = rp.n,
                    .dst_ne = rp.dst_ne, .dst_strides = rp.dst_strides, .dst_offset = rp.dst_offset,
                    .src0_ne = rp.src_ne, .src0_strides = rp.src_strides, .src0_offset = rp.src_offset,
                    .src1_ne = .{ 0, 0, 0, 0 }, .src1_strides = .{ 0, 0, 0, 0 }, .src1_offset = 0,
                };
                var bufs = [_]?*anyopaque{ self.device_bufs[rp.src].ptr, self.device_bufs[rp.dst].ptr };
                const grid: u32 = (rp.n + 255) / 256;
                c.mtl_encode_dispatch(cmds, be.repeat_pipeline, @ptrCast(&bufs), 2, &params, @sizeOf(ComputeParams), 3, grid, 1, 256, 1);
            },
            .slice_assign => |sa| {
                const params = ComputeParams{
                    .op = @intFromEnum(backend_mod.Op.slice_assign), .n_elements = sa.n,
                    .dst_ne = .{ 0, 0, 0, 0 }, .dst_strides = .{ sa.dst_stride, 0, 0, 0 }, .dst_offset = sa.dst_offset,
                    .src0_ne = .{ 0, 0, 0, 0 }, .src0_strides = .{ sa.src_stride, 0, 0, 0 }, .src0_offset = sa.src_offset,
                    .src1_ne = .{ 0, 0, 0, 0 }, .src1_strides = .{ 0, 0, 0, 0 }, .src1_offset = 0,
                };
                var bufs = [_]?*anyopaque{ self.device_bufs[sa.src].ptr, self.device_bufs[sa.dst].ptr };
                const grid: u32 = (sa.n + 255) / 256;
                c.mtl_encode_dispatch(cmds, be.slice_assign_pipeline, @ptrCast(&bufs), 2, &params, @sizeOf(ComputeParams), 3, grid, 1, 256, 1);
            },
        }
    }
};

fn compileProgramFn(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = std.heap.page_allocator;

    // Allocate device buffers.
    const device_bufs = alloc.alloc(backend_mod.DeviceBuffer, program.n_buffers) catch return null;
    for (device_bufs, program.buffer_sizes) |*buf, size| {
        buf.* = .{ .ptr = c.mtl_create_buffer(self.device, size * @sizeOf(f32)) orelse return null, .size = size * @sizeOf(f32) };
    }

    // Upload initial data (weights, KV cache zeros).
    for (program.initial_uploads) |io| {
        const buf = device_bufs[io.buf_idx];
        const ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(buf.ptr));
        @memcpy(ptr[io.offset..][0..io.size], io.host_ptr[0..io.size]);
    }

    // Upload quantized weights.
    const qweight_views = alloc.alloc(backend_mod.DeviceQuantizedWeightView, program.qweights.len) catch return null;
    for (program.qweights, 0..) |qw, i| {
        const data_buf: backend_mod.DeviceBuffer = .{ .ptr = c.mtl_create_buffer(self.device, qw.data.len) orelse return null, .size = qw.data.len };
        const data_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
        @memcpy(data_ptr[0..qw.data.len], i8_as_u8[0..qw.data.len]);

        const scales_buf: backend_mod.DeviceBuffer = .{ .ptr = c.mtl_create_buffer(self.device, qw.scales.len * @sizeOf(f32)) orelse return null, .size = qw.scales.len * @sizeOf(f32) };
        const scales_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(scales_buf.ptr));
        @memcpy(scales_ptr[0 .. qw.scales.len * @sizeOf(f32)], std.mem.sliceAsBytes(qw.scales));

        qweight_views[i] = .{ .data = data_buf, .scales = scales_buf, .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
    }

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .backend = self,
        .device_bufs = device_bufs,
        .qweight_views = qweight_views,
        .ops = program.ops,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn executeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn freeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .quantized_matmul_f32 = quantizedMatMulF32,
    .alloc_buffer = allocBuffer,
    .free_buffer = freeBuffer,
    .upload = uploadFn,
    .download = downloadFn,
    .sync = syncFn,
    .compile_program = compileProgramFn,
    .execute_program = executeProgramFn,
    .free_program = freeProgramFn,
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

test "metal backend compiled program matmul" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    const be = metal.backend();

    // Program: buf0(A) × buf1(B) → buf2(dst). 2x3 × 3x2 = 2x2.
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2, .a = 0, .b = 1,
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
