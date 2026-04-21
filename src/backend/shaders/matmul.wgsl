// Tiled f32 matmul — workgroup-based (no subgroup intrinsics).
// 16×16 output tile per workgroup, K blocked in steps of 16.
// Each thread computes one output element.

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

const TILE: u32 = 16u;
const BK: u32   = 16u;

var<workgroup> tA : array<f32, 256>;  // TILE * BK = 16 * 16
var<workgroup> tB : array<f32, 256>;  // BK * TILE = 16 * 16

@compute @workgroup_size(TILE, TILE)
fn main(
    @builtin(workgroup_id)         group_id : vec3<u32>,
    @builtin(local_invocation_id)  local_id : vec3<u32>,
) {
    let row = group_id.y * TILE + local_id.y;
    let col = group_id.x * TILE + local_id.x;
    let lr  = local_id.y;
    let lc  = local_id.x;

    var acc: f32 = 0.0;

    for (var kt: u32 = 0u; kt < p.K; kt += BK) {
        // Load A tile: tA[lr][lc] = A[row, kt + lc]
        let a_row = row;
        let a_col = kt + lc;
        if (a_row < p.M && a_col < p.K) {
            tA[lr * BK + lc] = A[p.a_offset + a_row * p.a_row_stride + a_col * p.a_col_stride];
        } else {
            tA[lr * BK + lc] = 0.0;
        }

        // Load B tile: tB[lr][lc] = B[kt + lr, col]
        let b_row = kt + lr;
        let b_col = col;
        if (b_row < p.K && b_col < p.N) {
            tB[lr * TILE + lc] = B[p.b_offset + b_row * p.b_row_stride + b_col * p.b_col_stride];
        } else {
            tB[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate: dot product over shared tiles
        for (var k: u32 = 0u; k < BK; k++) {
            acc += tA[lr * BK + k] * tB[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < p.M && col < p.N) {
        C[p.dst_offset + row * p.dst_row_stride + col] = acc;
    }
}
