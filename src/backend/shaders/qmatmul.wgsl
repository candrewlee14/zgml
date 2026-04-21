// Register-tiled quantized int8 matmul — 2D block tiling with per-thread accumulation.
// B weights stored as packed array<i32> (4 × int8 per i32), dequantized during
// cooperative load into shared memory. Accumulation loop is pure f32.

struct QMatMulParams {
    M: u32, N: u32, K: u32,
    block_size: u32,
};

@group(0) @binding(0) var<storage, read>       weight_data   : array<i32>;
@group(0) @binding(1) var<storage, read>       weight_scales : array<f32>;
@group(0) @binding(2) var<storage, read>       input         : array<f32>;
@group(0) @binding(3) var<storage, read_write> output        : array<f32>;
@group(0) @binding(4) var<uniform>             p             : QMatMulParams;

const BM: u32 = 64u;     // output rows per workgroup
const BN: u32 = 64u;     // output cols per workgroup
const BK: u32 = 16u;     // K-dimension block size
const TM: u32 = 4u;      // output rows per thread
const TN: u32 = 4u;      // output cols per thread
const N_THREADS: u32 = 256u;

// Shared memory: tI padded to stride BK+1=17 to avoid bank conflicts.
var<workgroup> tI : array<f32, 1088>;  // BM * (BK+1) = 64 * 17
var<workgroup> tW : array<f32, 1024>;  // BK * BN     = 16 * 64

// Extract one signed byte from a packed i32.
fn extract_i8(packed: i32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte_val = (packed >> shift) & 0xFF;
    let signed_val = select(byte_val, byte_val - 256, byte_val >= 128);
    return f32(signed_val);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id)         group_id : vec3<u32>,
    @builtin(local_invocation_id)  local_id : vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let tid = ty * 16u + tx;

    let row0 = group_id.y * BM;
    let col0 = group_id.x * BN;

    let thread_row = ty * TM;
    let thread_col = tx * TN;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let LOADS_PER_THREAD: u32 = 4u;
    let BK_PAD: u32 = BK + 1u;

    for (var kt: u32 = 0u; kt < p.K; kt += BK) {

        // ── Cooperative load input tile [BM × BK] into tI (padded stride 17) ──
        for (var i = 0u; i < LOADS_PER_THREAD; i++) {
            let flat = tid + i * N_THREADS;
            let i_r = flat / BK;
            let i_c = flat % BK;
            let g_row = row0 + i_r;
            let g_col = kt + i_c;
            if (g_row < p.M && g_col < p.K) {
                tI[i_r * BK_PAD + i_c] = input[g_row * p.K + g_col];
            } else {
                tI[i_r * BK_PAD + i_c] = 0.0;
            }
        }

        // ── Cooperative load weight tile [BK × BN] with dequantization ──
        for (var i = 0u; i < LOADS_PER_THREAD; i++) {
            let flat = tid + i * N_THREADS;
            let w_r = flat / BN;
            let w_c = flat % BN;
            let g_row = kt + w_r;
            let g_col = col0 + w_c;
            if (g_row < p.K && g_col < p.N) {
                let w_idx = g_row * p.N + g_col;
                let packed_idx = w_idx / 4u;
                let byte_lane  = w_idx % 4u;
                let packed = weight_data[packed_idx];
                let scale  = weight_scales[w_idx / p.block_size];
                tW[w_r * BN + w_c] = extract_i8(packed, byte_lane) * scale;
            } else {
                tW[w_r * BN + w_c] = 0.0;
            }
        }

        workgroupBarrier();

        // ── Register accumulation: outer product over BK ──
        for (var k = 0u; k < BK; k++) {
            var a_reg: array<f32, 4>;
            a_reg[0] = tI[(thread_row + 0u) * BK_PAD + k];
            a_reg[1] = tI[(thread_row + 1u) * BK_PAD + k];
            a_reg[2] = tI[(thread_row + 2u) * BK_PAD + k];
            a_reg[3] = tI[(thread_row + 3u) * BK_PAD + k];

            var b_reg: array<f32, 4>;
            let b_base = k * BN + thread_col;
            b_reg[0] = tW[b_base + 0u];
            b_reg[1] = tW[b_base + 1u];
            b_reg[2] = tW[b_base + 2u];
            b_reg[3] = tW[b_base + 3u];

            acc[0]  += a_reg[0] * b_reg[0];
            acc[1]  += a_reg[0] * b_reg[1];
            acc[2]  += a_reg[0] * b_reg[2];
            acc[3]  += a_reg[0] * b_reg[3];
            acc[4]  += a_reg[1] * b_reg[0];
            acc[5]  += a_reg[1] * b_reg[1];
            acc[6]  += a_reg[1] * b_reg[2];
            acc[7]  += a_reg[1] * b_reg[3];
            acc[8]  += a_reg[2] * b_reg[0];
            acc[9]  += a_reg[2] * b_reg[1];
            acc[10] += a_reg[2] * b_reg[2];
            acc[11] += a_reg[2] * b_reg[3];
            acc[12] += a_reg[3] * b_reg[0];
            acc[13] += a_reg[3] * b_reg[1];
            acc[14] += a_reg[3] * b_reg[2];
            acc[15] += a_reg[3] * b_reg[3];
        }

        workgroupBarrier();
    }

    // ── Write results with bounds checking ──
    for (var tm = 0u; tm < TM; tm++) {
        let g_row = row0 + thread_row + tm;
        if (g_row >= p.M) { continue; }
        for (var tn = 0u; tn < TN; tn++) {
            let g_col = col0 + thread_col + tn;
            if (g_col < p.N) {
                output[g_row * p.N + g_col] = acc[tm * TN + tn];
            }
        }
    }
}
