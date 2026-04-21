// Register-tiled f32 matmul — 2D block tiling with per-thread accumulation.
// BM×BN output tile per workgroup, each thread computes TM×TN outputs.
// 256 threads per workgroup (16×16), 16 output elements per thread.
//
// TODO: when wgpu-native ships cooperative matrix support (v30+), add an
// alternate kernel using `enable wgpu_cooperative_matrix;` with 8×8 f32
// coop_mat tiles and coopMatrixMulAdd(). This maps to simdgroup MMA on
// Metal and SPV_KHR_cooperative_matrix on Vulkan, closing the gap with
// the native Metal backend's simdgroup_multiply_accumulate path.

struct MatMulParams {
    M: u32, N: u32, K: u32,
    a_row_stride: u32, a_col_stride: u32,
    b_row_stride: u32, b_col_stride: u32,
    a_offset: u32, b_offset: u32,
    dst_offset: u32, dst_row_stride: u32,
};

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             p : MatMulParams;

const BM: u32 = 64u;     // output rows per workgroup
const BN: u32 = 64u;     // output cols per workgroup
const BK: u32 = 16u;     // K-dimension block size
const TM: u32 = 4u;      // output rows per thread
const TN: u32 = 4u;      // output cols per thread
const N_THREADS: u32 = 256u;  // (BM/TM) * (BN/TN) = 16 * 16
const ROWS_PER_WG: u32 = 16u; // BM / TM
const COLS_PER_WG: u32 = 16u; // BN / TN

// Shared memory: tA padded to stride BK+1=17 to avoid bank conflicts.
var<workgroup> tA : array<f32, 1088>;  // BM * (BK+1) = 64 * 17
var<workgroup> tB : array<f32, 1024>;  // BK * BN     = 16 * 64

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id)         group_id : vec3<u32>,
    @builtin(local_invocation_id)  local_id : vec3<u32>,
) {
    let tx = local_id.x;  // 0..15
    let ty = local_id.y;  // 0..15
    let tid = ty * 16u + tx;  // flat thread index 0..255

    // Global output tile origin.
    let row0 = group_id.y * BM;
    let col0 = group_id.x * BN;

    // Thread's output sub-tile origin within the workgroup tile.
    let thread_row = ty * TM;  // 0, 4, 8, ..., 60
    let thread_col = tx * TN;  // 0, 4, 8, ..., 60

    // Register accumulators: TM * TN = 16 values per thread.
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    // Cooperative load: 1024 elements per tile, 256 threads → 4 per thread.
    let LOADS_PER_THREAD: u32 = 4u;  // (BM * BK) / N_THREADS = 1024/256

    let BK_PAD: u32 = BK + 1u;  // padded stride for tA

    for (var kt: u32 = 0u; kt < p.K; kt += BK) {

        // ── Cooperative load A tile [BM × BK] into tA (padded stride 17) ──
        for (var i = 0u; i < LOADS_PER_THREAD; i++) {
            let flat = tid + i * N_THREADS;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let g_row = row0 + a_r;
            let g_col = kt + a_c;
            if (g_row < p.M && g_col < p.K) {
                tA[a_r * BK_PAD + a_c] = A[p.a_offset + g_row * p.a_row_stride + g_col * p.a_col_stride];
            } else {
                tA[a_r * BK_PAD + a_c] = 0.0;
            }
        }

        // ── Cooperative load B tile [BK × BN] into tB ──
        for (var i = 0u; i < LOADS_PER_THREAD; i++) {
            let flat = tid + i * N_THREADS;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let g_row = kt + b_r;
            let g_col = col0 + b_c;
            if (g_row < p.K && g_col < p.N) {
                tB[b_r * BN + b_c] = B[p.b_offset + g_row * p.b_row_stride + g_col * p.b_col_stride];
            } else {
                tB[b_r * BN + b_c] = 0.0;
            }
        }

        workgroupBarrier();

        // ── Register accumulation: outer product over BK ──
        for (var k = 0u; k < BK; k++) {
            // Load TM values from tA column k.
            var a_reg: array<f32, 4>;
            a_reg[0] = tA[(thread_row + 0u) * BK_PAD + k];
            a_reg[1] = tA[(thread_row + 1u) * BK_PAD + k];
            a_reg[2] = tA[(thread_row + 2u) * BK_PAD + k];
            a_reg[3] = tA[(thread_row + 3u) * BK_PAD + k];

            // Load TN values from tB row k.
            var b_reg: array<f32, 4>;
            let b_base = k * BN + thread_col;
            b_reg[0] = tB[b_base + 0u];
            b_reg[1] = tB[b_base + 1u];
            b_reg[2] = tB[b_base + 2u];
            b_reg[3] = tB[b_base + 3u];

            // Outer product: TM × TN = 16 FMAs.
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
                C[p.dst_offset + g_row * p.dst_row_stride + g_col] = acc[tm * TN + tn];
            }
        }
    }
}
