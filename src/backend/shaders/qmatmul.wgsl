// Quantized int8 matmul — workgroup tiled, on-the-fly dequantization.
// B weights stored as packed array<i32> (4 × int8 per i32).
// Same tiling as matmul.wgsl: 16×16 output tile, K blocked in steps of 16.

struct QMatMulParams {
    M: u32, N: u32, K: u32,
    block_size: u32,
};

@group(0) @binding(0) var<storage, read>       weight_data   : array<i32>;
@group(0) @binding(1) var<storage, read>       weight_scales : array<f32>;
@group(0) @binding(2) var<storage, read>       input         : array<f32>;
@group(0) @binding(3) var<storage, read_write> output        : array<f32>;
@group(0) @binding(4) var<uniform>             p             : QMatMulParams;

const TILE: u32 = 16u;
const BK: u32   = 16u;

var<workgroup> tI : array<f32, 256>;  // TILE * BK
var<workgroup> tW : array<f32, 256>;  // BK * TILE

// Extract one signed byte from a packed i32.
// lane = 0..3 selects byte within the 4-byte word.
fn extract_i8(packed: i32, lane: u32) -> f32 {
    let shift = lane * 8u;
    // Shift right to put target byte in low 8 bits, mask, sign-extend.
    let byte_val = (packed >> shift) & 0xFF;
    // Sign-extend from 8 bits: if bit 7 set, subtract 256.
    let signed_val = select(byte_val, byte_val - 256, byte_val >= 128);
    return f32(signed_val);
}

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
        // Load input tile: tI[lr][lc] = input[row, kt + lc]
        let i_row = row;
        let i_col = kt + lc;
        if (i_row < p.M && i_col < p.K) {
            tI[lr * BK + lc] = input[i_row * p.K + i_col];
        } else {
            tI[lr * BK + lc] = 0.0;
        }

        // Load weight tile with dequantization:
        // Weight layout is row-major [K × N], stored as bytes packed 4-per-i32.
        // w_idx = (kt + lr) * N + col is the linear byte index.
        let w_row = kt + lr;
        let w_col = col;
        if (w_row < p.K && w_col < p.N) {
            let w_idx = w_row * p.N + w_col;
            let packed_idx = w_idx / 4u;
            let byte_lane  = w_idx % 4u;
            let packed = weight_data[packed_idx];
            let scale  = weight_scales[w_idx / p.block_size];
            tW[lr * TILE + lc] = extract_i8(packed, byte_lane) * scale;
        } else {
            tW[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BK; k++) {
            acc += tI[lr * BK + k] * tW[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < p.M && col < p.N) {
        output[row * p.N + col] = acc;
    }
}
