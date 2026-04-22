//! Metal GPU backend for macOS / Apple Silicon.
//!
//! Uses shared memory (MTLResourceStorageModeShared) so upload/download
//! are plain memcpy — CPU and GPU see the same physical pages.
//! Each dispatch is synchronous (commit + waitUntilCompleted).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const profile_mod = @import("../profile.zig");
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
    \\    float sin_val = cos_sin[p.cs_off + col * p.cs_cs + d + i];
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
    \\    uint4 dst_ne;    uint4 dst_strides;   uint dst_offset;
    \\    uint4 src0_ne;   uint4 src0_strides;   uint src0_offset;
    \\    uint4 src1_ne;   uint4 src1_strides;   uint src1_offset;
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
    \\            dst[p.dst_offset + gid * p.dst_strides[0]] = src0[p.src0_offset + gid * p.src0_strides[0]];
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
        const matvec_f16_pipeline = c.mtl_create_pipeline(device, library, "matvec_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matvec_f16_pipeline);
        const matmul_f16_pipeline = c.mtl_create_pipeline(device, library, "matmul_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matmul_f16_pipeline);
        const rope_pipeline = c.mtl_create_pipeline(device, library, "rope_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rope_pipeline);
        const attention_pipeline = c.mtl_create_pipeline(device, library, "attention_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(attention_pipeline);
        const compute_pipeline = c.mtl_create_pipeline(device, library, "compute_f32") orelse return error.PipelineCreateFailed;

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
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
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

const CompiledProgram = struct {
    backend: *MetalBackend,
    device_bufs: []DeviceBuffer,
    buf_ptrs: [][*]f32, // cached mtl_buffer_contents pointers
    qweight_views: []DeviceQWeight,
    ops: []const backend_mod.DeviceOp,
    alloc: std.mem.Allocator,
    runtime_profile: profile_mod.RuntimeProfile = .{},

    fn deinit(self: *CompiledProgram) void {
        for (self.device_bufs) |buf| c.mtl_release(buf.ptr);
        self.alloc.free(self.buf_ptrs);
        self.alloc.free(self.device_bufs);
        if (self.qweight_views.len > 0) self.alloc.free(self.qweight_views);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        // Upload per-step inputs (token embed, pos, mask) via shared memory.
        for (inputs) |io| {
            const ptr: [*]u8 = @ptrCast(self.buf_ptrs[io.buf_idx]);
            @memcpy(ptr[io.offset..][0..io.size], io.host_ptr[0..io.size]);
        }

        // All ops on CPU using shared-memory buffers.
        // At decode batch_size=1, BLAS matmuls + SIMD ops beat GPU dispatch
        // overhead (~210 commits/token). Weights are pre-loaded in device
        // buffers, avoiding graph traversal and pointer chasing per token.
        for (self.ops) |op| {
            const tag: usize = @intFromEnum(op);
            const t0 = nowNs();
            self.executeCpuOp(op);
            self.runtime_profile.time_ns[tag] +%= @intCast(nowNs() - t0);
        }
        self.runtime_profile.call_count += 1;

        // Download outputs (logits) via shared memory.
        for (outputs) |io| {
            const ptr: [*]const u8 = @ptrCast(self.buf_ptrs[io.buf_idx]);
            @memcpy(io.host_ptr[0..io.size], ptr[io.offset..][0..io.size]);
        }
    }

    // ── CPU execution ──────────────────────────────────────────────

    fn bufF32(self: *const CompiledProgram, idx: u16) [*]f32 {
        return self.buf_ptrs[idx];
    }

    fn executeCpuOp(self: *const CompiledProgram, op: backend_mod.DeviceOp) void {
        switch (op) {
            .matmul => |m| self.cpuMatmul(m),
            .qmatmul => |q| self.cpuQmatmul(q),
            .elementwise => |e| self.cpuElementwise(e),
            .softmax => |s| self.cpuSoftmax(s),
            .layernorm => |l| self.cpuLayernorm(l),
            .rmsnorm => |r| self.cpuRmsnorm(r),
            .reduce => |rd| self.cpuReduce(rd),
            .repeat => |rp| self.cpuRepeat(rp),
            .slice_assign => |sa| self.cpuSliceAssign(sa),
            .rope => |rr| self.cpuRope(rr),
            .attention => |att| self.cpuAttention(att),
            .fused_elementwise => |fe| self.cpuFusedElementwise(fe),
        }
    }

    const V = 8; // SIMD lane count — 2× NEON width for superscalar fill.

    fn simdBinaryLoop(dst: [*]f32, src0: [*]const f32, src1: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32), @Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src0[i..][0..V].*;
            const b: VecT = src1[i..][0..V].*;
            dst[i..][0..V].* = op(a, b);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src0[i])), @as(VecT, @splat(src1[i])))[0];
    }

    fn simdUnaryLoop(dst: [*]f32, src: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src[i..][0..V].*;
            dst[i..][0..V].* = op(a);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src[i])))[0];
    }

    fn cpuElementwise(self: *const CompiledProgram, e: anytype) void {
        const dst = self.bufF32(e.dst) + @as(usize, e.dst_offset);
        const src0 = self.bufF32(e.src0) + @as(usize, e.src0_offset);
        const src1 = self.bufF32(e.src1) + @as(usize, e.src1_offset);
        const n: usize = e.n;
        switch (e.op) {
            .add => simdBinaryLoop(dst, src0, src1, n, struct { fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) { return a + b; } }.f),
            .mul => simdBinaryLoop(dst, src0, src1, n, struct { fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) { return a * b; } }.f),
            .neg => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return -a; } }.f),
            .abs => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @abs(a); } }.f),
            .relu => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @max(a, @as(@Vector(V, f32), @splat(0.0))); } }.f),
            .sqrt => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @sqrt(a); } }.f),
            .recip => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @as(@Vector(V, f32), @splat(@as(f32, 1.0))) / a; } }.f),
            .exp => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @exp(a); } }.f),
            .log => simdUnaryLoop(dst, src0, n, struct { fn f(a: @Vector(V, f32)) @Vector(V, f32) { return @log(a); } }.f),
            .gelu => {
                const VecT = @Vector(V, f32);
                const k0: VecT = @splat(0.7978845608);
                const k1: VecT = @splat(0.044715);
                const half: VecT = @splat(0.5);
                const one: VecT = @splat(1.0);
                var i: usize = 0;
                while (i + V <= n) : (i += V) {
                    const a: VecT = src0[i..][0..V].*;
                    const k = k0 * (a + k1 * a * a * a);
                    // tanh via (exp(2x)-1)/(exp(2x)+1)
                    const e2k = @exp(k + k);
                    dst[i..][0..V].* = half * a * (one + (e2k - one) / (e2k + one));
                }
                while (i < n) : (i += 1) {
                    const a = src0[i];
                    const kk = 0.7978845608 * (a + 0.044715 * a * a * a);
                    dst[i] = 0.5 * a * (1.0 + std.math.tanh(kk));
                }
            },
            else => @memcpy(dst[0..n], src0[0..n]),
        }
    }

    fn cpuFusedElementwise(self: *const CompiledProgram, fe: anytype) void {
        const dst = self.bufF32(fe.dst) + @as(usize, fe.dst_offset);
        const src = self.bufF32(fe.src) + @as(usize, fe.src_offset);
        const n: usize = fe.n;
        for (0..n) |i| {
            var v = src[i];
            for (fe.steps) |step| {
                switch (step.op) {
                    .neg => v = -v,
                    .abs => v = @abs(v),
                    .relu => v = @max(v, 0.0),
                    .sqrt => v = @sqrt(v),
                    .recip => v = 1.0 / v,
                    .exp => v = @exp(v),
                    .log => v = @log(v),
                    .gelu => {
                        const kk = 0.7978845608 * (v + 0.044715 * v * v * v);
                        v = 0.5 * v * (1.0 + std.math.tanh(kk));
                    },
                    .add => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] + v else v + s_ptr[i];
                    },
                    .mul => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] * v else v * s_ptr[i];
                    },
                    else => {},
                }
            }
            dst[i] = v;
        }
    }

    fn cpuSoftmax(self: *const CompiledProgram, s: anytype) void {
        const src = self.bufF32(s.src);
        const dst = self.bufF32(s.dst);
        const cols: usize = s.cols;
        for (0..@as(usize, s.rows)) |row| {
            const sb: usize = @as(usize, s.src_offset) + row * cols;
            const db: usize = @as(usize, s.dst_offset) + row * cols;
            var m: f32 = -std.math.inf(f32);
            for (0..cols) |j| m = @max(m, src[sb + j]);
            var sum: f32 = 0;
            for (0..cols) |j| {
                const v = @exp(src[sb + j] - m);
                dst[db + j] = v;
                sum += v;
            }
            const inv = if (sum > 0.0) 1.0 / sum else 0.0;
            for (0..cols) |j| dst[db + j] *= inv;
        }
    }

    fn cpuLayernorm(self: *const CompiledProgram, l: anytype) void {
        const src = self.bufF32(l.src);
        const dst = self.bufF32(l.dst);
        const cols: usize = l.cols;
        for (0..@as(usize, l.rows)) |row| {
            const base: usize = @as(usize, l.src_offset) + row * cols;
            const dbase: usize = @as(usize, l.dst_offset) + row * cols;
            var mu: f32 = 0;
            for (0..cols) |j| mu += src[base + j];
            mu /= @as(f32, @floatFromInt(cols));
            var v: f32 = 0;
            for (0..cols) |j| {
                const diff = src[base + j] - mu;
                v += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(v / @as(f32, @floatFromInt(cols)) + l.eps);
            for (0..cols) |j| dst[dbase + j] = (src[base + j] - mu) * inv_std;
        }
    }

    fn cpuRmsnorm(self: *const CompiledProgram, r: anytype) void {
        const src = self.bufF32(r.src);
        const dst = self.bufF32(r.dst);
        const cols: usize = r.cols;
        const VecT = @Vector(V, f32);
        for (0..@as(usize, r.rows)) |row| {
            const s = src + @as(usize, r.src_offset) + row * cols;
            const d = dst + @as(usize, r.dst_offset) + row * cols;
            var acc: VecT = @splat(0);
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                acc += v * v;
            }
            var ss: f32 = @reduce(.Add, acc);
            while (i < cols) : (i += 1) ss += s[i] * s[i];
            const inv_rms: VecT = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(cols)) + r.eps));
            i = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                d[i..][0..V].* = v * inv_rms;
            }
            const inv_s = inv_rms[0];
            while (i < cols) : (i += 1) d[i] = s[i] * inv_s;
        }
    }

    fn cpuReduce(self: *const CompiledProgram, rd: anytype) void {
        const src = self.bufF32(rd.src);
        const dst = self.bufF32(rd.dst);
        const rs: usize = rd.reduce_size;
        for (0..@as(usize, rd.n_out)) |i| {
            const sb: usize = @as(usize, rd.src_offset) + i * rs;
            var val: f32 = if (rd.op == .max) -std.math.inf(f32) else 0.0;
            for (0..rs) |k| {
                const v = src[sb + k];
                val = if (rd.op == .max) @max(val, v) else val + v;
            }
            dst[@as(usize, rd.dst_offset) + i] = val;
        }
    }

    fn cpuRepeat(self: *const CompiledProgram, rp: anytype) void {
        const src = self.bufF32(rp.src);
        const dst = self.bufF32(rp.dst);
        const n: usize = rp.n;
        const d = dst + @as(usize, rp.dst_offset);
        const s = src + @as(usize, rp.src_offset);

        const src_n: usize = @as(usize, rp.src_ne[0]) * @as(usize, rp.src_ne[1]) *
            @as(usize, rp.src_ne[2]) * @as(usize, rp.src_ne[3]);

        if (src_n == 1) {
            // Scalar broadcast → fill.
            @memset(d[0..n], s[0]);
            return;
        }
        if (src_n >= n) {
            // Same-size or larger source → plain copy.
            @memcpy(d[0..n], s[0..n]);
            return;
        }
        // Tiled contiguous copy: source is contiguous and tiles evenly into dst.
        if (n % src_n == 0 and rp.src_strides[0] == 1 and
            (rp.src_ne[1] <= 1 or rp.src_strides[1] == rp.src_ne[0]) and
            (rp.src_ne[2] <= 1 or rp.src_strides[2] == @as(u32, rp.src_ne[0]) * rp.src_ne[1]) and
            (rp.src_ne[3] <= 1 or rp.src_strides[3] == @as(u32, rp.src_ne[0]) * @as(u32, rp.src_ne[1]) * rp.src_ne[2]))
        {
            var off: usize = 0;
            while (off + src_n <= n) : (off += src_n) {
                @memcpy(d[off..][0..src_n], s[0..src_n]);
            }
            return;
        }

        // Generic 4D modular indexing fallback.
        for (0..n) |gid| {
            var idx = gid;
            var src_idx: usize = rp.src_offset;
            var dim: usize = 4;
            while (dim > 0) {
                dim -= 1;
                const coord = idx / @as(usize, rp.dst_strides[dim]);
                idx = idx % @as(usize, rp.dst_strides[dim]);
                src_idx += (coord % @as(usize, rp.src_ne[dim])) * @as(usize, rp.src_strides[dim]);
            }
            dst[@as(usize, rp.dst_offset) + gid] = src[src_idx];
        }
    }

    fn cpuSliceAssign(self: *const CompiledProgram, sa: anytype) void {
        const src = self.bufF32(sa.src);
        const dst = self.bufF32(sa.dst);
        const n: usize = sa.n;
        const doff: usize = sa.dst_offset;
        const soff: usize = sa.src_offset;
        if (sa.dst_stride == 1 and sa.src_stride == 1) {
            @memcpy(dst[doff..][0..n], src[soff..][0..n]);
        } else {
            const ds: usize = sa.dst_stride;
            const ss: usize = sa.src_stride;
            for (0..n) |i| dst[doff + i * ds] = src[soff + i * ss];
        }
    }

    fn cpuRope(self: *const CompiledProgram, rr: anytype) void {
        const src = self.bufF32(rr.src);
        const cs = self.bufF32(rr.cos_sin);
        const dst = self.bufF32(rr.dst);
        const hd: usize = rr.half_d;
        const s_off: usize = rr.src_off;
        const c_off: usize = rr.cs_off;
        const d_off: usize = rr.dst_off;
        const s_rs: usize = rr.src_rs;
        const s_cs: usize = rr.src_cs;
        const c_cs: usize = rr.cs_cs;
        for (0..@as(usize, rr.seq_len)) |col| {
            for (0..hd) |pair| {
                const x_lo = src[s_off + pair * s_rs + col * s_cs];
                const x_hi = src[s_off + (pair + hd) * s_rs + col * s_cs];
                const cos_v = cs[c_off + pair + col * c_cs];
                const sin_v = cs[c_off + pair + hd + col * c_cs];
                dst[d_off + pair + col * 2 * hd] = x_lo * cos_v - x_hi * sin_v;
                dst[d_off + pair + hd + col * 2 * hd] = x_hi * cos_v + x_lo * sin_v;
            }
        }
    }

    fn cpuMatmul(self: *const CompiledProgram, m: anytype) void {
        // Always use BLAS on the f32 buffer — f16 shadows are for GPU kernels only.
        const a_ptr = self.bufF32(m.a);
        const b_ptr = self.bufF32(m.b);
        const dst_ptr = self.bufF32(m.dst);
        const a_slice = a_ptr[0 .. self.device_bufs[m.a].size / @sizeOf(f32)];
        const b_slice = b_ptr[0 .. self.device_bufs[m.b].size / @sizeOf(f32)];
        const dst_slice = dst_ptr[0 .. self.device_bufs[m.dst].size / @sizeOf(f32)];
        const forward = @import("../tensor/forward.zig");
        forward.blasSgemm(dst_slice, a_slice, b_slice, m.geom.M, m.geom.N, m.geom.K, m.geom.a_row_stride, m.geom.a_col_stride, m.geom.b_row_stride, m.geom.b_col_stride, m.geom.a_offset, m.geom.b_offset, m.geom.dst_offset, m.geom.dst_row_stride);
    }

    fn cpuQmatmul(self: *const CompiledProgram, q: anytype) void {
        const input = self.bufF32(q.input);
        const dst_ptr = self.bufF32(q.dst);
        const w = self.qweight_views[q.weight_idx];
        const qdata: [*]const i8 = @ptrCast(c.mtl_buffer_contents(w.data.ptr));
        const scales: [*]const f32 = @ptrCast(@alignCast(c.mtl_buffer_contents(w.scales.ptr)));
        const M: usize = q.M;
        const N: usize = q.N;
        const K: usize = q.K;
        const bs: usize = w.block_size;
        const n_blocks = K / bs;

        for (0..M) |row| {
            for (0..N) |col| {
                var sum: f32 = 0;
                for (0..n_blocks) |blk| {
                    const scale = scales[col * n_blocks + blk];
                    for (0..bs) |k_off| {
                        const k = blk * bs + k_off;
                        sum += input[row * K + k] * @as(f32, @floatFromInt(qdata[col * K + k])) * scale;
                    }
                }
                dst_ptr[row * N + col] = sum;
            }
        }
    }

    fn cpuAttention(self: *const CompiledProgram, att: anytype) void {
        const q_ptr = self.bufF32(att.q);
        const k_ptr = self.bufF32(att.k);
        const v_ptr = self.bufF32(att.v);
        const mask_ptr = self.bufF32(att.mask);
        const dst = self.bufF32(att.dst);
        const dh: usize = att.d_head;
        const skv: usize = att.seq_kv;
        const q_off: usize = att.q_off;
        const k_off: usize = att.k_off;
        const v_off: usize = att.v_off;
        const m_off: usize = att.mask_off;
        const d_off: usize = att.dst_off;
        const krs: usize = att.k_rs;
        const kcs: usize = att.k_cs;
        const vrs: usize = att.v_rs;
        const vcs: usize = att.v_cs;
        const VecT = @Vector(V, f32);
        const neg_inf = -std.math.inf(f32);

        // Online softmax: single pass fusing Q·K, softmax, and V accumulation.
        var m_val: f32 = neg_inf;
        var l: f32 = 0;
        var acc_buf: [512]f32 = undefined; // d_head max
        const acc = acc_buf[0..dh];
        @memset(acc, 0);

        const unit_k = (krs == 1);
        const unit_v = (vrs == 1);

        for (0..skv) |s| {
            const mask_add = mask_ptr[m_off + s];
            if (!std.math.isFinite(mask_add)) continue;

            // Q·K dot product.
            var dot: f32 = 0;
            if (unit_k) {
                var dot_v: VecT = @splat(0);
                var r: usize = 0;
                const kb = k_off + s * kcs;
                while (r + V <= dh) : (r += V) {
                    const qv: VecT = q_ptr[q_off + r ..][0..V].*;
                    const kv: VecT = k_ptr[kb + r ..][0..V].*;
                    dot_v += qv * kv;
                }
                dot = @reduce(.Add, dot_v);
                while (r < dh) : (r += 1) dot += q_ptr[q_off + r] * k_ptr[kb + r];
            } else {
                for (0..dh) |r| dot += q_ptr[q_off + r] * k_ptr[k_off + r * krs + s * kcs];
            }

            const score = dot * att.scale + mask_add;
            if (!std.math.isFinite(score)) continue;

            // Online softmax update.
            const new_m = @max(m_val, score);
            const alpha = if (m_val == neg_inf) @as(f32, 0) else @exp(m_val - new_m);
            const w = @exp(score - new_m);
            l = l * alpha + w;
            m_val = new_m;

            // Accumulate V with rescaling: acc = acc * alpha + w * V[s].
            if (unit_v) {
                const alpha_v: VecT = @splat(alpha);
                const w_v: VecT = @splat(w);
                var r: usize = 0;
                const vb = v_off + s * vcs;
                while (r + V <= dh) : (r += V) {
                    const av: VecT = acc[r..][0..V].*;
                    const vv: VecT = v_ptr[vb + r ..][0..V].*;
                    acc[r..][0..V].* = av * alpha_v + w_v * vv;
                }
                while (r < dh) : (r += 1) {
                    acc[r] = acc[r] * alpha + w * v_ptr[vb + r];
                }
            } else {
                for (0..dh) |r| {
                    acc[r] = acc[r] * alpha + w * v_ptr[v_off + r * vrs + s * vcs];
                }
            }
        }

        // Normalize and write output.
        const inv_l = if (l > 0) 1.0 / l else @as(f32, 0);
        if (unit_v) {
            const inv_v: VecT = @splat(inv_l);
            var r: usize = 0;
            while (r + V <= dh) : (r += V) {
                const av: VecT = acc[r..][0..V].*;
                dst[d_off + r ..][0..V].* = av * inv_v;
            }
            while (r < dh) : (r += 1) dst[d_off + r] = acc[r] * inv_l;
        } else {
            for (0..dh) |r| dst[d_off + r] = acc[r] * inv_l;
        }
    }
};

fn compileProgramFn(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = std.heap.page_allocator;

    // Allocate device buffers.
    const device_bufs = alloc.alloc(DeviceBuffer, program.n_buffers) catch return null;
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
    const qweight_views = alloc.alloc(DeviceQWeight, program.qweights.len) catch return null;
    for (program.qweights, 0..) |qw, i| {
        const data_buf: DeviceBuffer = .{ .ptr = c.mtl_create_buffer(self.device, qw.data.len) orelse return null, .size = qw.data.len };
        const data_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
        @memcpy(data_ptr[0..qw.data.len], i8_as_u8[0..qw.data.len]);

        const scales_buf: DeviceBuffer = .{ .ptr = c.mtl_create_buffer(self.device, qw.scales.len * @sizeOf(f32)) orelse return null, .size = qw.scales.len * @sizeOf(f32) };
        const scales_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(scales_buf.ptr));
        @memcpy(scales_ptr[0 .. qw.scales.len * @sizeOf(f32)], std.mem.sliceAsBytes(qw.scales));

        qweight_views[i] = .{ .data = data_buf, .scales = scales_buf, .block_size = qw.block_size };
    }

    // Cache mtl_buffer_contents pointers — stable for shared-memory buffers.
    const buf_ptrs = alloc.alloc([*]f32, program.n_buffers) catch return null;
    for (buf_ptrs, device_bufs) |*bp, buf| {
        bp.* = @ptrCast(@alignCast(c.mtl_buffer_contents(buf.ptr)));
    }

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .backend = self,
        .device_bufs = device_bufs,
        .buf_ptrs = buf_ptrs,
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

fn getRuntimeProfileFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?*profile_mod.RuntimeProfile {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return &compiled.runtime_profile;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgramFn,
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
