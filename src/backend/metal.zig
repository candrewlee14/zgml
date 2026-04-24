//! Metal GPU backend for macOS / Apple Silicon.
//!
//! Uses shared memory (MTLResourceStorageModeShared) so upload/download
//! are plain memcpy — CPU and GPU see the same physical pages.
//! Each dispatch is synchronous (commit + waitUntilCompleted).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const profile_mod = @import("../profile.zig");
const reference = @import("reference.zig");
const program_mod = @import("program.zig");
const c = @cImport(@cInclude("metal_shim.h"));

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
}

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
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint dst_row_stride;
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
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[p.input_offset + ir * p.input_row_stride + ic] : 0.0f;
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
    \\            output[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\// ── F16 matvec: M==1, one thread per output element ────
    \\
    \\struct MatVecParams {
    \\    uint N; uint K;
    \\    uint a_offset; uint dst_offset;
    \\};
    \\
    \\kernel void matvec_f16(
    \\    device const float* A  [[buffer(0)]],
    \\    device const half*  B  [[buffer(1)]],
    \\    device float*       C  [[buffer(2)]],
    \\    constant MatVecParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.N) return;
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K; k++) {
    \\        sum += A[p.a_offset + k] * float(B[k * p.N + gid]);
    \\    }
    \\    C[p.dst_offset + gid] = sum;
    \\}
    \\
    \\// ── F16 tiled matmul: M>1, simdgroup with half precision ─
    \\
    \\struct MatMulF16Params {
    \\    uint M; uint N; uint K;
    \\    uint a_row_stride; uint a_col_stride;
    \\    uint a_offset; uint dst_offset; uint dst_row_stride;
    \\};
    \\
    \\kernel void matmul_f16(
    \\    device const float* A  [[buffer(0)]],
    \\    device const half*  B  [[buffer(1)]],
    \\    device float*       C  [[buffer(2)]],
    \\    constant MatMulF16Params& p [[buffer(3)]],
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
    \\    threadgroup half tA[TILE * 8];
    \\    threadgroup half tB[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        // Load A (f32 global) → half threadgroup
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ar = gRow + r, ac = kt + c;
    \\            tA[i] = (ar < p.M && ac < p.K)
    \\                ? half(A[p.a_offset + ar * p.a_row_stride + ac * p.a_col_stride]) : half(0);
    \\        }
    \\        // Load B (f16 global, packed K×N) → half threadgroup
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint br = kt + r, bc = gCol + c;
    \\            tB[i] = (br < p.K && bc < p.N)
    \\                ? B[br * p.N + bc] : half(0);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_half8x8 a0, a1, b0, b1;
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
    \\            C[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\// ── RoPE: rotary position encoding ─────────────────────
    \\
    \\struct RopeParams {
    \\    uint half_d; uint seq_len;
    \\    uint src_off; uint cs_off; uint dst_off;
    \\    uint src_rs; uint src_cs; uint cs_cs;
    \\};
    \\
    \\kernel void rope_f32(
    \\    device const float* src     [[buffer(0)]],
    \\    device const float* cos_sin [[buffer(1)]],
    \\    device float*       dst     [[buffer(2)]],
    \\    constant RopeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.half_d * p.seq_len) return;
    \\    uint col = gid / p.half_d;
    \\    uint i   = gid % p.half_d;
    \\    uint d   = p.half_d * 2;
    \\
    \\    float cos_val = cos_sin[p.cs_off + col * p.cs_cs + i];
    \\    float sin_val = cos_sin[p.cs_off + col * p.cs_cs + p.half_d + i];
    \\    float x_lo = src[p.src_off + col * p.src_cs + i * p.src_rs];
    \\    float x_hi = src[p.src_off + col * p.src_cs + (i + p.half_d) * p.src_rs];
    \\
    \\    uint dst_base = p.dst_off + col * d;
    \\    dst[dst_base + i]            = x_lo * cos_val - x_hi * sin_val;
    \\    dst[dst_base + i + p.half_d] = x_hi * cos_val + x_lo * sin_val;
    \\}
    \\
    \\// ── Attention: fused softmax(Q@K^T * scale + mask) @ V ─
    \\
    \\struct AttentionParams {
    \\    uint d_head; uint seq_kv;
    \\    float scale;
    \\    uint q_off; uint k_off; uint v_off; uint mask_off; uint dst_off;
    \\    uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\};
    \\
    \\constant uint MAX_SEQ = 4096;
    \\
    \\// Single-query attention (seq_q=1). One threadgroup per head.
    \\kernel void attention_f32(
    \\    device const float* Q    [[buffer(0)]],
    \\    device const float* K    [[buffer(1)]],
    \\    device const float* V    [[buffer(2)]],
    \\    device const float* mask [[buffer(3)]],
    \\    device float*       dst  [[buffer(4)]],
    \\    constant AttentionParams& p [[buffer(5)]],
    \\    uint tid     [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    // Phase 1: Compute Q·K scores (threads share the work across KV positions).
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[p.mask_off + s];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += Q[p.q_off + r] * K[p.k_off + r * p.k_rs + s * p.k_cs];
    \\        scores[s] = dot * p.scale + mv;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 2: Max reduction.
    \\    float local_max = -INFINITY;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        local_max = max(local_max, scores[s]);
    \\    scratch[tid] = local_max;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float global_max = scratch[0];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 3: Softmax (exp + sum reduction + normalize).
    \\    float local_sum = 0.0f;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float w = (scores[s] == -INFINITY) ? 0.0f : exp(scores[s] - global_max);
    \\        scores[s] = w;
    \\        local_sum += w;
    \\    }
    \\    scratch[tid] = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] += scratch[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv_sum = (scratch[0] > 0.0f) ? 1.0f / scratch[0] : 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        scores[s] *= inv_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 4: Weighted sum of V columns → output[d_head].
    \\    for (uint r = tid; r < p.d_head; r += tg_size) {
    \\        float val = 0.0f;
    \\        for (uint s = 0; s < p.seq_kv; s++)
    \\            val += scores[s] * V[p.v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[p.dst_off + r] = val;
    \\    }
    \\}
    \\
    \\// ── Unified compute kernel ──────────────────────────────
    \\
    \\struct ComputeParams {
    \\    uint op;
    \\    uint n_elements;
    \\    uint dst_ne[4];    uint dst_strides[4];   uint dst_offset;
    \\    uint src0_ne[4];   uint src0_strides[4];   uint src0_offset;
    \\    uint src1_ne[4];   uint src1_strides[4];   uint src1_offset;
    \\};
    \\
    \\// Op codes — elementwise match op.zig values directly:
    \\//   add=7 mul=8 neg=9 abs=10 sgn=11 step=12 relu=13
    \\//   sqrt=14 recip=15 exp=16 log=17 gelu=18
    \\//   sum=19 max=20 repeat=21 slice_assign=27
    \\// Fused ops use codes 100+:
    \\//   fused_softmax=100 fused_layernorm=101 fused_rmsnorm=102
    \\
    \\kernel void compute_f32(
    \\    device const float* src0 [[buffer(0)]],
    \\    device const float* src1 [[buffer(1)]],
    \\    device float* dst        [[buffer(2)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    switch (p.op) {
    \\        // ── Elementwise (one thread per element) ──
    \\        case 7:  dst[p.dst_offset + gid] = src0[p.src0_offset + gid] + src1[p.src1_offset + gid]; break;
    \\        case 8:  dst[p.dst_offset + gid] = src0[p.src0_offset + gid] * src1[p.src1_offset + gid]; break;
    \\        case 9:  dst[p.dst_offset + gid] = -src0[p.src0_offset + gid]; break;
    \\        case 10: dst[p.dst_offset + gid] = abs(src0[p.src0_offset + gid]); break;
    \\        case 11: dst[p.dst_offset + gid] = sign(src0[p.src0_offset + gid]); break;
    \\        case 12: dst[p.dst_offset + gid] = (src0[p.src0_offset + gid] > 0.0f) ? 1.0f : 0.0f; break;
    \\        case 13: dst[p.dst_offset + gid] = max(src0[p.src0_offset + gid], 0.0f); break;
    \\        case 14: dst[p.dst_offset + gid] = sqrt(src0[p.src0_offset + gid]); break;
    \\        case 15: dst[p.dst_offset + gid] = 1.0f / src0[p.src0_offset + gid]; break;
    \\        case 16: dst[p.dst_offset + gid] = exp(src0[p.src0_offset + gid]); break;
    \\        case 17: dst[p.dst_offset + gid] = log(src0[p.src0_offset + gid]); break;
    \\        case 18: {
    \\            float a = src0[p.src0_offset + gid];
    \\            float c = 0.7978845608f * (a + 0.044715f * a * a * a);
    \\            dst[p.dst_offset + gid] = 0.5f * a * (1.0f + precise::tanh(c));
    \\            break;
    \\        }
    \\        // ── Reduce: sum(19) or max(20), one thread per output ──
    \\        case 19: case 20: {
    \\            uint reduce_size = p.src0_ne[0];
    \\            uint src_base = p.src0_offset + gid * reduce_size;
    \\            float val = (p.op == 20) ? -INFINITY : 0.0f;
    \\            for (uint k = 0; k < reduce_size; k++) {
    \\                float v = src0[src_base + k];
    \\                if (p.op == 19) val += v; else val = max(val, v);
    \\            }
    \\            dst[p.dst_offset + gid] = val;
    \\            break;
    \\        }
    \\        // ── Repeat: broadcast via modular indexing ──
    \\        case 21: {
    \\            uint idx = gid;
    \\            uint src_idx = p.src0_offset;
    \\            for (int d = 3; d >= 0; d--) {
    \\                uint ud = uint(d);
    \\                uint coord = idx / p.dst_strides[ud];
    \\                idx %= p.dst_strides[ud];
    \\                src_idx += (coord % p.src0_ne[ud]) * p.src0_strides[ud];
    \\            }
    \\            dst[p.dst_offset + gid] = src0[src_idx];
    \\            break;
    \\        }
    \\        // ── Slice assign: strided copy ──
    \\        case 27: {
    \\            uint row = gid % p.src0_ne[0];
    \\            uint col = gid / p.src0_ne[0];
    \\            dst[p.dst_offset + row * p.dst_strides[0] + col * p.dst_strides[1]] =
    \\                src0[p.src0_offset + row * p.src0_strides[0] + col * p.src0_strides[1]];
    \\            break;
    \\        }
    \\        // ── Fused softmax: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 100: {
    \\            uint cols = p.src0_ne[0];
    \\            uint src_base = p.src0_offset + gid * cols;
    \\            uint dst_base = p.dst_offset + gid * cols;
    \\            float m = -INFINITY;
    \\            for (uint j = 0; j < cols; j++) m = max(m, src0[src_base + j]);
    \\            float s = 0.0f;
    \\            for (uint j = 0; j < cols; j++) {
    \\                float e = exp(src0[src_base + j] - m);
    \\                dst[dst_base + j] = e;
    \\                s += e;
    \\            }
    \\            float inv = 1.0f / s;
    \\            for (uint j = 0; j < cols; j++) dst[dst_base + j] *= inv;
    \\            break;
    \\        }
    \\        // ── Fused layer norm: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 101: {
    \\            uint cols = p.src0_ne[0];
    \\            uint base = p.src0_offset + gid * cols;
    \\            uint dbase = p.dst_offset + gid * cols;
    \\            float eps = as_type<float>(p.src1_ne[0]);
    \\            float mu = 0.0f;
    \\            for (uint j = 0; j < cols; j++) mu += src0[base + j];
    \\            mu /= float(cols);
    \\            float v = 0.0f;
    \\            for (uint j = 0; j < cols; j++) {
    \\                float d = src0[base + j] - mu;
    \\                v += d * d;
    \\            }
    \\            float inv_std = 1.0f / sqrt(v / float(cols) + eps);
    \\            for (uint j = 0; j < cols; j++)
    \\                dst[dbase + j] = (src0[base + j] - mu) * inv_std;
    \\            break;
    \\        }
    \\        // ── Fused RMS norm: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 102: {
    \\            uint cols = p.src0_ne[0];
    \\            uint base = p.src0_offset + gid * cols;
    \\            uint dbase = p.dst_offset + gid * cols;
    \\            float eps = as_type<float>(p.src1_ne[0]);
    \\            float ss = 0.0f;
    \\            for (uint j = 0; j < cols; j++) ss += src0[base + j] * src0[base + j];
    \\            float inv_rms = 1.0f / sqrt(ss / float(cols) + eps);
    \\            for (uint j = 0; j < cols; j++)
    \\                dst[dbase + j] = src0[base + j] * inv_rms;
    \\            break;
    \\        }
    \\        default: break;
    \\    }
    \\}
    \\
    \\constant uint MAX_FUSED_EW_STEPS = 8;
    \\constant uint MAX_FUSED_EW_SECONDARIES = 8;
    \\
    \\struct FusedEwParams {
    \\    uint n_elements;
    \\    uint n_steps;
    \\    uint dst_offset;
    \\    uint src_offset;
    \\    uint op[MAX_FUSED_EW_STEPS];
    \\    uint is_swapped[MAX_FUSED_EW_STEPS];
    \\    uint secondary_slot[MAX_FUSED_EW_STEPS];
    \\    uint secondary_offset[MAX_FUSED_EW_STEPS];
    \\};
    \\
    \\float fused_secondary(
    \\    uint slot,
    \\    uint idx,
    \\    device const float* s0,
    \\    device const float* s1,
    \\    device const float* s2,
    \\    device const float* s3,
    \\    device const float* s4,
    \\    device const float* s5,
    \\    device const float* s6,
    \\    device const float* s7
    \\) {
    \\    switch (slot) {
    \\        case 0: return s0[idx];
    \\        case 1: return s1[idx];
    \\        case 2: return s2[idx];
    \\        case 3: return s3[idx];
    \\        case 4: return s4[idx];
    \\        case 5: return s5[idx];
    \\        case 6: return s6[idx];
    \\        default: return s7[idx];
    \\    }
    \\}
    \\
    \\float fused_unary(uint op, float v) {
    \\    switch (op) {
    \\        case 9: return -v;
    \\        case 10: return abs(v);
    \\        case 11: return sign(v);
    \\        case 12: return (v > 0.0f) ? 1.0f : 0.0f;
    \\        case 13: return max(v, 0.0f);
    \\        case 14: return sqrt(v);
    \\        case 15: return 1.0f / v;
    \\        case 16: return exp(v);
    \\        case 17: return log(v);
    \\        case 18: {
    \\            float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    \\            return 0.5f * v * (1.0f + precise::tanh(c));
    \\        }
    \\        default: return v;
    \\    }
    \\}
    \\
    \\kernel void fused_elementwise_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst [[buffer(1)]],
    \\    device const float* s0 [[buffer(2)]],
    \\    device const float* s1 [[buffer(3)]],
    \\    device const float* s2 [[buffer(4)]],
    \\    device const float* s3 [[buffer(5)]],
    \\    device const float* s4 [[buffer(6)]],
    \\    device const float* s5 [[buffer(7)]],
    \\    device const float* s6 [[buffer(8)]],
    \\    device const float* s7 [[buffer(9)]],
    \\    constant FusedEwParams& p [[buffer(10)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    float v = src[p.src_offset + gid];
    \\    for (uint step = 0; step < p.n_steps; step++) {
    \\        uint op = p.op[step];
    \\        if (op == 7 || op == 8) {
    \\            uint sec_idx = p.secondary_offset[step] + gid;
    \\            float other = fused_secondary(p.secondary_slot[step], sec_idx, s0, s1, s2, s3, s4, s5, s6, s7);
    \\            if (op == 7) v = (p.is_swapped[step] != 0) ? other + v : v + other;
    \\            else v = (p.is_swapped[step] != 0) ? other * v : v * other;
    \\        } else {
    \\            v = fused_unary(op, v);
    \\        }
    \\    }
    \\    dst[p.dst_offset + gid] = v;
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
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const MatVecParams = extern struct {
    N: u32,
    K: u32,
    a_offset: u32,
    dst_offset: u32,
};

const MatMulF16Params = extern struct {
    M: u32,
    N: u32,
    K: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    a_offset: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const RopeParams = extern struct {
    half_d: u32,
    seq_len: u32,
    src_off: u32,
    cs_off: u32,
    dst_off: u32,
    src_rs: u32,
    src_cs: u32,
    cs_cs: u32,
};

const AttentionParams = extern struct {
    d_head: u32,
    seq_kv: u32,
    scale: f32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    mask_off: u32,
    dst_off: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
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

const MAX_FUSED_EW_STEPS: usize = 8;
const MAX_FUSED_EW_SECONDARIES: usize = 8;

const FusedEwParams = extern struct {
    n_elements: u32,
    n_steps: u32,
    dst_offset: u32,
    src_offset: u32,
    op: [MAX_FUSED_EW_STEPS]u32,
    is_swapped: [MAX_FUSED_EW_STEPS]u32,
    secondary_slot: [MAX_FUSED_EW_STEPS]u32,
    secondary_offset: [MAX_FUSED_EW_STEPS]u32,
};

const WG_SIZE: u32 = 256;
const MATMUL_THREADS: u32 = 128;

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
        .gx = @intCast((N + TILE - 1) / TILE),
        .gy = @intCast((M + TILE - 1) / TILE),
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

fn qmatmulParams(q: anytype, block_size: usize) QMatMulParams {
    return .{
        .M = q.M,
        .N = q.N,
        .K = q.K,
        .block_size = @intCast(block_size),
        .input_offset = q.input_offset,
        .input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K,
        .dst_offset = q.dst_offset,
        .dst_row_stride = if (q.dst_row_stride != 0) q.dst_row_stride else q.N,
    };
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

fn sliceAssignComputeParams(sa: anytype) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = @intFromEnum(backend_mod.Op.slice_assign);
    p.n_elements = sa.rows * sa.cols;
    p.src0_ne[0] = sa.rows;
    p.dst_strides[0] = sa.dst_row_stride;
    p.dst_strides[1] = sa.dst_col_stride;
    p.dst_offset = sa.dst_offset;
    p.src0_strides[0] = sa.src_row_stride;
    p.src0_strides[1] = sa.src_col_stride;
    p.src0_offset = sa.src_offset;
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
        .reduce => |r| {
            if (r.op != .sum and r.op != .max) return null;
            return .{
                .params = reduceComputeParams(r),
                .src0 = r.src,
                .src1 = r.dst,
                .dst = r.dst,
                .grid = .{ .gx = linearGrid(r.n_out) },
            };
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
        else => return null,
    }
}

fn attentionParams(att: anytype) AttentionParams {
    return .{
        .d_head = att.d_head,
        .seq_kv = att.seq_kv,
        .scale = att.scale,
        .q_off = att.q_off,
        .k_off = att.k_off,
        .v_off = att.v_off,
        .mask_off = att.mask_off,
        .dst_off = att.dst_off,
        .k_rs = att.k_rs,
        .k_cs = att.k_cs,
        .v_rs = att.v_rs,
        .v_cs = att.v_cs,
    };
}

// ── MetalBackend ──────────────────────────────────────────────────

pub const MetalBackend = struct {
    device: *anyopaque,
    queue: *anyopaque,
    matmul_pipeline: *anyopaque,
    qmatmul_pipeline: *anyopaque,
    matvec_f16_pipeline: *anyopaque,
    matmul_f16_pipeline: *anyopaque,
    rope_pipeline: *anyopaque,
    attention_pipeline: *anyopaque,
    compute_pipeline: *anyopaque,
    fused_ew_pipeline: *anyopaque,
    library: *anyopaque,
    active_commands: ?*anyopaque = null,
    fine_grained_program_dispatch: bool = false,

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
        const matvec_f16_pipeline = c.mtl_create_pipeline(device, library, "matvec_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matvec_f16_pipeline);
        const matmul_f16_pipeline = c.mtl_create_pipeline(device, library, "matmul_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matmul_f16_pipeline);
        const rope_pipeline = c.mtl_create_pipeline(device, library, "rope_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rope_pipeline);
        const attention_pipeline = c.mtl_create_pipeline(device, library, "attention_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(attention_pipeline);
        const compute_pipeline = c.mtl_create_pipeline(device, library, "compute_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(compute_pipeline);
        const fused_ew_pipeline = c.mtl_create_pipeline(device, library, "fused_elementwise_f32") orelse return error.PipelineCreateFailed;

        return .{
            .device = device,
            .queue = queue,
            .matmul_pipeline = matmul_pipeline,
            .qmatmul_pipeline = qmatmul_pipeline,
            .matvec_f16_pipeline = matvec_f16_pipeline,
            .matmul_f16_pipeline = matmul_f16_pipeline,
            .rope_pipeline = rope_pipeline,
            .attention_pipeline = attention_pipeline,
            .compute_pipeline = compute_pipeline,
            .fused_ew_pipeline = fused_ew_pipeline,
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
        c.mtl_release(self.fused_ew_pipeline);
        c.mtl_release(self.compute_pipeline);
        c.mtl_release(self.attention_pipeline);
        c.mtl_release(self.rope_pipeline);
        c.mtl_release(self.matmul_f16_pipeline);
        c.mtl_release(self.matvec_f16_pipeline);
        c.mtl_release(self.qmatmul_pipeline);
        c.mtl_release(self.matmul_pipeline);
        c.mtl_release(self.library);
        c.mtl_release(self.queue);
        c.mtl_release(self.device);
    }

    /// Enable one-dispatch-per-DeviceOp execution for experiments and
    /// conformance tests. Decode keeps this off by default until layer-level
    /// fusion removes the tiny-dispatch overhead.
    pub fn setFineGrainedProgramDispatch(self: *MetalBackend, enabled: bool) void {
        self.fine_grained_program_dispatch = enabled;
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
            .capabilities = backend_mod.Capabilities.metal,
        };
    }
};

// ── VTable implementation ─────────────────────────────────────────

fn getState(ctx: *anyopaque) *MetalBackend {
    return @ptrCast(@alignCast(ctx));
}

// Host kernel dispatch — delegate to BLAS.
fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const forward = @import("../tensor/forward.zig");
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

// ── Compiled program execution ────────────────────────────────────

/// Opaque handle to a Metal buffer + size.
const DeviceBuffer = struct {
    ptr: *anyopaque,
    size: usize,
};

/// Device-resident quantized weight (data + scales as Metal buffers).
const DeviceQWeight = struct {
    data: DeviceBuffer,
    scales: DeviceBuffer,
    block_size: usize,
};

fn releaseDeviceBuffers(device_bufs: []const DeviceBuffer) void {
    for (device_bufs) |buf| c.mtl_release(buf.ptr);
}

fn releaseQWeightViews(qweight_views: []const DeviceQWeight) void {
    for (qweight_views) |qw| {
        c.mtl_release(qw.data.ptr);
        c.mtl_release(qw.scales.ptr);
    }
}

const CompiledProgram = struct {
    backend: *MetalBackend,
    device_bufs: []DeviceBuffer,
    ref_buffers: []reference.Buffer,
    qweight_views: []DeviceQWeight,
    ref_qweights: []reference.QWeight,
    ops: []const backend_mod.DeviceOp,
    alloc: std.mem.Allocator,
    runtime_profile: profile_mod.RuntimeProfile = .{},

    fn deinit(self: *CompiledProgram) void {
        releaseDeviceBuffers(self.device_bufs);
        releaseQWeightViews(self.qweight_views);
        self.alloc.free(self.ref_buffers);
        self.alloc.free(self.device_bufs);
        if (self.qweight_views.len > 0) self.alloc.free(self.qweight_views);
        if (self.ref_qweights.len > 0) self.alloc.free(self.ref_qweights);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        // Upload per-step inputs (token embed, pos, mask) via shared memory.
        reference.uploadToBuffers(self.ref_buffers, inputs);

        for (self.ops) |op| {
            const tag: usize = @intFromEnum(op);
            const t0 = nowNs();
            if (!self.tryEncodeGpuOp(op)) {
                self.backend.flushCommands();
                reference.executeOp(self.ref_buffers, self.ref_qweights, op);
            }
            self.runtime_profile.time_ns[tag] +%= @intCast(nowNs() - t0);
        }
        self.backend.flushCommands();
        self.runtime_profile.call_count += 1;

        // Download outputs (logits) via shared memory.
        reference.downloadFromBuffers(self.ref_buffers, outputs);
    }

    fn encode(
        self: *CompiledProgram,
        pipeline: *anyopaque,
        buffers: []const DeviceBuffer,
        params: *const anyopaque,
        params_size: usize,
        params_index: u32,
        grid: DispatchGrid,
        threads_x: u32,
    ) void {
        var raw_buffers: [16]?*anyopaque = undefined;
        for (buffers, 0..) |buf, i| raw_buffers[i] = buf.ptr;
        c.mtl_encode_dispatch(
            self.backend.ensureCommands(),
            pipeline,
            @ptrCast(&raw_buffers),
            @intCast(buffers.len),
            params,
            params_size,
            params_index,
            grid.gx,
            grid.gy,
            threads_x,
            1,
        );
    }

    fn encodeTyped(
        self: *CompiledProgram,
        comptime Params: type,
        pipeline: *anyopaque,
        buffers: []const DeviceBuffer,
        params: Params,
        params_index: u32,
        grid: DispatchGrid,
        threads_x: u32,
    ) void {
        self.encode(pipeline, buffers, &params, @sizeOf(Params), params_index, grid, threads_x);
    }

    fn tryEncodeFusedElementwise(self: *CompiledProgram, fe: anytype) bool {
        if (!self.backend.fine_grained_program_dispatch) return false;
        if (fe.steps.len > MAX_FUSED_EW_STEPS) return false;

        var params = std.mem.zeroes(FusedEwParams);
        params.n_elements = fe.n;
        params.n_steps = @intCast(fe.steps.len);
        params.dst_offset = fe.dst_offset;
        params.src_offset = fe.src_offset;

        var buffers: [2 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer = undefined;
        buffers[0] = self.device_bufs[fe.src];
        buffers[1] = self.device_bufs[fe.dst];
        for (buffers[2..]) |*buf| buf.* = self.device_bufs[fe.src];

        var secondary_bufs: [MAX_FUSED_EW_SECONDARIES]u16 = undefined;
        var secondary_count: usize = 0;

        for (fe.steps, 0..) |step, i| {
            params.op[i] = @intFromEnum(step.op);
            params.is_swapped[i] = @intFromBool(step.is_swapped);
            params.secondary_offset[i] = step.secondary_offset;

            if (step.op.isBinary()) {
                const slot = for (secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                    if (buf_idx == step.secondary_buf) break slot_idx;
                } else blk: {
                    if (secondary_count >= MAX_FUSED_EW_SECONDARIES) return false;
                    const next = secondary_count;
                    secondary_bufs[next] = step.secondary_buf;
                    buffers[2 + next] = self.device_bufs[step.secondary_buf];
                    secondary_count += 1;
                    break :blk next;
                };
                params.secondary_slot[i] = @intCast(slot);
            }
        }

        self.encodeTyped(
            FusedEwParams,
            self.backend.fused_ew_pipeline,
            buffers[0..],
            params,
            10,
            .{ .gx = linearGrid(fe.n) },
            WG_SIZE,
        );
        return true;
    }

    fn tryEncodeGpuOp(self: *CompiledProgram, op: backend_mod.DeviceOp) bool {
        const fine_grained = self.backend.fine_grained_program_dispatch;

        if (fine_grained) {
            if (computeDispatchSpec(op)) |spec| {
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[spec.src0],
                    self.device_bufs[spec.src1],
                    self.device_bufs[spec.dst],
                };
                self.encodeTyped(ComputeParams, self.backend.compute_pipeline, &buffers, spec.params, 3, spec.grid, WG_SIZE);
                return true;
            }
        }

        switch (op) {
            .matmul => |m| {
                if (!fine_grained and m.geom.M < 16) return false;
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[m.a],
                    self.device_bufs[m.b],
                    self.device_bufs[m.dst],
                };
                self.encodeTyped(MatMulParams, self.backend.matmul_pipeline, &buffers, matmulParams(m.geom), 3, matmulGrid(m.geom.M, m.geom.N), MATMUL_THREADS);
                return true;
            },
            .qmatmul => |q| {
                if (!fine_grained and q.M < 16) return false;
                const w = self.qweight_views[q.weight_idx];
                const buffers = [_]DeviceBuffer{
                    w.data,
                    w.scales,
                    self.device_bufs[q.input],
                    self.device_bufs[q.dst],
                };
                self.encodeTyped(QMatMulParams, self.backend.qmatmul_pipeline, &buffers, qmatmulParams(q, w.block_size), 4, matmulGrid(q.M, q.N), MATMUL_THREADS);
                return true;
            },
            .rope => |rr| {
                if (!fine_grained) return false;
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[rr.src],
                    self.device_bufs[rr.cos_sin],
                    self.device_bufs[rr.dst],
                };
                const params = RopeParams{
                    .half_d = rr.half_d,
                    .seq_len = rr.seq_len,
                    .src_off = rr.src_off,
                    .cs_off = rr.cs_off,
                    .dst_off = rr.dst_off,
                    .src_rs = rr.src_rs,
                    .src_cs = rr.src_cs,
                    .cs_cs = rr.cs_cs,
                };
                self.encodeTyped(RopeParams, self.backend.rope_pipeline, &buffers, params, 3, .{ .gx = linearGrid(rr.half_d * rr.seq_len) }, WG_SIZE);
                return true;
            },
            .attention => |att| {
                if (!fine_grained) return false;
                if (att.seq_q != 1 or att.seq_kv > 4096 or att.d_head > 512) return false;
                if (!att.has_mask or att.q_rs != 1 or att.mask_rs != 1 or att.dst_rs != 1) return false;
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[att.q],
                    self.device_bufs[att.k],
                    self.device_bufs[att.v],
                    self.device_bufs[att.mask],
                    self.device_bufs[att.dst],
                };
                self.encodeTyped(AttentionParams, self.backend.attention_pipeline, &buffers, attentionParams(att), 5, .{ .gx = 1 }, WG_SIZE);
                return true;
            },
            .elementwise, .softmax, .layernorm, .rmsnorm, .reduce, .repeat, .slice_assign => return false,
            .fused_elementwise => |fe| return self.tryEncodeFusedElementwise(fe),
        }
    }
};

fn compileProgramFn(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = std.heap.page_allocator;

    // Allocate device buffers.
    const device_bufs = alloc.alloc(DeviceBuffer, program.n_buffers) catch return null;
    errdefer alloc.free(device_bufs);
    var n_device_bufs: usize = 0;
    errdefer releaseDeviceBuffers(device_bufs[0..n_device_bufs]);
    for (device_bufs, program.buffer_sizes) |*buf, size| {
        const byte_size = size * @sizeOf(f32);
        const ptr = c.mtl_create_buffer(self.device, byte_size) orelse return null;
        buf.* = .{ .ptr = ptr, .size = byte_size };
        n_device_bufs += 1;
    }

    // Upload initial data (weights, KV cache zeros).
    for (program.initial_uploads) |io| {
        const buf = device_bufs[io.buf_idx];
        const ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(buf.ptr));
        @memcpy(ptr[io.offset..][0..io.size], io.host_ptr[0..io.size]);
    }

    // Upload quantized weights.
    const qweight_views = alloc.alloc(DeviceQWeight, program.qweights.len) catch return null;
    errdefer if (qweight_views.len > 0) alloc.free(qweight_views);
    const ref_qweights = alloc.alloc(reference.QWeight, program.qweights.len) catch return null;
    errdefer if (ref_qweights.len > 0) alloc.free(ref_qweights);
    var n_qweight_views: usize = 0;
    errdefer releaseQWeightViews(qweight_views[0..n_qweight_views]);
    for (program.qweights, 0..) |qw, i| {
        const data_raw = c.mtl_create_buffer(self.device, qw.data.len) orelse return null;
        const data_buf: DeviceBuffer = .{ .ptr = data_raw, .size = qw.data.len };
        const data_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
        @memcpy(data_ptr[0..qw.data.len], i8_as_u8[0..qw.data.len]);

        const scales_size = qw.scales.len * @sizeOf(f32);
        const scales_raw = c.mtl_create_buffer(self.device, scales_size) orelse {
            c.mtl_release(data_buf.ptr);
            return null;
        };
        const scales_buf: DeviceBuffer = .{ .ptr = scales_raw, .size = scales_size };
        const scales_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(scales_buf.ptr));
        @memcpy(scales_ptr[0..scales_size], std.mem.sliceAsBytes(qw.scales));

        qweight_views[i] = .{ .data = data_buf, .scales = scales_buf, .block_size = qw.block_size };
        const ref_data: [*]const i8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const ref_scales: [*]const f32 = @ptrCast(@alignCast(c.mtl_buffer_contents(scales_buf.ptr)));
        ref_qweights[i] = .{ .data = ref_data[0..qw.data.len], .scales = ref_scales[0..qw.scales.len], .block_size = qw.block_size };
        n_qweight_views += 1;
    }

    // Cache mtl_buffer_contents pointers — stable for shared-memory buffers.
    const ref_buffers = alloc.alloc(reference.Buffer, program.n_buffers) catch return null;
    errdefer alloc.free(ref_buffers);
    for (ref_buffers, device_bufs) |*rb, buf| {
        rb.* = .{ .ptr = @ptrCast(@alignCast(c.mtl_buffer_contents(buf.ptr))), .len = buf.size / @sizeOf(f32) };
    }

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .backend = self,
        .device_bufs = device_bufs,
        .ref_buffers = ref_buffers,
        .qweight_views = qweight_views,
        .ref_qweights = ref_qweights,
        .ops = program.ops,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn executeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn refreshProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, ops: []const backend_mod.DeviceOp) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.ops = ops;
}

fn freeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn getRuntimeProfileFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?*profile_mod.RuntimeProfile {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return &compiled.runtime_profile;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgramFn,
    .refresh_program = refreshProgramFn,
    .execute_program = executeProgramFn,
    .free_program = freeProgramFn,
    .get_runtime_profile = getRuntimeProfileFn,
};

// ── Tests ─────────────────────────────────────────────────────────

test "metal backend compiled program matmul" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    const be = metal.backend();

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
