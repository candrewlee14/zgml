// Unified elementwise / reduce / fused compute kernel.
// Op codes match op.zig enum values:
//   add=7 mul=8 neg=9 abs=10 sgn=11 step=12 relu=13
//   sqrt=14 recip=15 exp=16 log=17 gelu=18
//   sum=19 max=20 repeat=21 slice_assign=27
//   fused_softmax=100 fused_layernorm=101 fused_rmsnorm=102

struct ComputeParams {
    op: u32,
    n_elements: u32,
    dst_ne0: u32, dst_ne1: u32, dst_ne2: u32, dst_ne3: u32,
    dst_str0: u32, dst_str1: u32, dst_str2: u32, dst_str3: u32,
    dst_offset: u32,
    src0_ne0: u32, src0_ne1: u32, src0_ne2: u32, src0_ne3: u32,
    src0_str0: u32, src0_str1: u32, src0_str2: u32, src0_str3: u32,
    src0_offset: u32,
    src1_ne0: u32, src1_ne1: u32, src1_ne2: u32, src1_ne3: u32,
    src1_str0: u32, src1_str1: u32, src1_str2: u32, src1_str3: u32,
    src1_offset: u32,
};

@group(0) @binding(0) var<storage, read>       src0 : array<f32>;
@group(0) @binding(1) var<storage, read>       src1 : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(3) var<uniform>             p    : ComputeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid_v : vec3<u32>) {
    let gid = gid_v.x;
    if (gid >= p.n_elements) { return; }

    switch p.op {
        // ── Elementwise (one thread per element) ──
        case 7u: {  // add
            dst[p.dst_offset + gid] = src0[p.src0_offset + gid] + src1[p.src1_offset + gid];
        }
        case 8u: {  // mul
            dst[p.dst_offset + gid] = src0[p.src0_offset + gid] * src1[p.src1_offset + gid];
        }
        case 9u: {  // neg
            dst[p.dst_offset + gid] = -src0[p.src0_offset + gid];
        }
        case 10u: { // abs
            dst[p.dst_offset + gid] = abs(src0[p.src0_offset + gid]);
        }
        case 11u: { // sgn
            let v = src0[p.src0_offset + gid];
            dst[p.dst_offset + gid] = sign(v);
        }
        case 12u: { // step
            dst[p.dst_offset + gid] = select(0.0, 1.0, src0[p.src0_offset + gid] > 0.0);
        }
        case 13u: { // relu
            dst[p.dst_offset + gid] = max(src0[p.src0_offset + gid], 0.0);
        }
        case 14u: { // sqrt
            dst[p.dst_offset + gid] = sqrt(src0[p.src0_offset + gid]);
        }
        case 15u: { // recip
            dst[p.dst_offset + gid] = 1.0 / src0[p.src0_offset + gid];
        }
        case 16u: { // exp
            dst[p.dst_offset + gid] = exp(src0[p.src0_offset + gid]);
        }
        case 17u: { // log
            dst[p.dst_offset + gid] = log(src0[p.src0_offset + gid]);
        }
        case 18u: { // gelu
            let a = src0[p.src0_offset + gid];
            let c = 0.7978845608 * (a + 0.044715 * a * a * a);
            dst[p.dst_offset + gid] = 0.5 * a * (1.0 + tanh(c));
        }
        // ── Reduce: sum(19) or max(20), one thread per output ──
        case 19u: {
            let reduce_size = p.src0_ne0;
            let src_base = p.src0_offset + gid * reduce_size;
            var val: f32 = 0.0;
            for (var k: u32 = 0u; k < reduce_size; k++) {
                val += src0[src_base + k];
            }
            dst[p.dst_offset + gid] = val;
        }
        case 20u: {
            let reduce_size = p.src0_ne0;
            let src_base = p.src0_offset + gid * reduce_size;
            var val: f32 = -3.402823466e+38;  // -FLT_MAX
            for (var k: u32 = 0u; k < reduce_size; k++) {
                val = max(val, src0[src_base + k]);
            }
            dst[p.dst_offset + gid] = val;
        }
        // ── Repeat: broadcast via modular indexing ──
        case 21u: {
            var idx = gid;
            var src_idx = p.src0_offset;

            let dst_ne  = array<u32, 4>(p.dst_ne0, p.dst_ne1, p.dst_ne2, p.dst_ne3);
            let dst_str  = array<u32, 4>(p.dst_str0, p.dst_str1, p.dst_str2, p.dst_str3);
            let src0_ne = array<u32, 4>(p.src0_ne0, p.src0_ne1, p.src0_ne2, p.src0_ne3);
            let src0_str = array<u32, 4>(p.src0_str0, p.src0_str1, p.src0_str2, p.src0_str3);

            // Unroll from dim 3 down to 0.
            var coord3 = idx / dst_str[3];
            idx = idx % dst_str[3];
            src_idx += (coord3 % src0_ne[3]) * src0_str[3];

            var coord2 = idx / dst_str[2];
            idx = idx % dst_str[2];
            src_idx += (coord2 % src0_ne[2]) * src0_str[2];

            var coord1 = idx / dst_str[1];
            idx = idx % dst_str[1];
            src_idx += (coord1 % src0_ne[1]) * src0_str[1];

            var coord0 = idx / dst_str[0];
            src_idx += (coord0 % src0_ne[0]) * src0_str[0];

            dst[p.dst_offset + gid] = src0[src_idx];
        }
        // ── Slice assign: 2D strided copy ──
        case 27u: {
            let row = gid % p.src0_ne0;
            let col = gid / p.src0_ne0;
            let dst_idx = p.dst_offset + row * p.dst_str0 + col * p.dst_str1;
            let src_idx = p.src0_offset + row * p.src0_str0 + col * p.src0_str1;
            dst[dst_idx] = src0[src_idx];
        }
        // ── Fused softmax: one thread per row ──
        // n_elements = rows, src0_ne0 = cols
        case 100u: {
            let cols = p.src0_ne0;
            let src_base = p.src0_offset + gid * cols;
            let dst_base = p.dst_offset + gid * cols;

            var m: f32 = -3.402823466e+38;
            for (var j: u32 = 0u; j < cols; j++) {
                m = max(m, src0[src_base + j]);
            }
            var s: f32 = 0.0;
            for (var j: u32 = 0u; j < cols; j++) {
                let e = exp(src0[src_base + j] - m);
                dst[dst_base + j] = e;
                s += e;
            }
            let inv = 1.0 / s;
            for (var j: u32 = 0u; j < cols; j++) {
                dst[dst_base + j] *= inv;
            }
        }
        // ── Fused layer norm: one thread per row ──
        // n_elements = rows, src0_ne0 = cols
        // eps passed via bitcast<f32>(p.src1_ne0)
        case 101u: {
            let cols = p.src0_ne0;
            let base  = p.src0_offset + gid * cols;
            let dbase = p.dst_offset + gid * cols;
            let eps = bitcast<f32>(p.src1_ne0);

            var mu: f32 = 0.0;
            for (var j: u32 = 0u; j < cols; j++) {
                mu += src0[base + j];
            }
            mu /= f32(cols);

            var v: f32 = 0.0;
            for (var j: u32 = 0u; j < cols; j++) {
                let d = src0[base + j] - mu;
                v += d * d;
            }
            let inv_std = 1.0 / sqrt(v / f32(cols) + eps);

            for (var j: u32 = 0u; j < cols; j++) {
                dst[dbase + j] = (src0[base + j] - mu) * inv_std;
            }
        }
        // ── Fused RMS norm: one thread per row ──
        // n_elements = rows, src0_ne0 = cols
        // eps passed via bitcast<f32>(p.src1_ne0)
        case 102u: {
            let cols = p.src0_ne0;
            let base = p.src0_offset + gid * cols;
            let dbase = p.dst_offset + gid * cols;
            let eps = bitcast<f32>(p.src1_ne0);
            var ss: f32 = 0.0;
            for (var j: u32 = 0u; j < cols; j++) {
                let v = src0[base + j];
                ss += v * v;
            }
            let inv_rms = 1.0 / sqrt(ss / f32(cols) + eps);
            for (var j: u32 = 0u; j < cols; j++) {
                dst[dbase + j] = src0[base + j] * inv_rms;
            }
        }
        default: {}
    }
}
