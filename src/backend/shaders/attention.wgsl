// One workgroup computes one attention query row.
//
// The graph already emits one fused attention op per head, so a compact
// per-row kernel is a good fit here: it keeps the backend small while still
// using the whole workgroup across the KV axis and output dimensions.

struct AttentionParams {
    dims0: vec4<u32>,
    scale_pad: vec4<f32>,
    offsets0: vec4<u32>,
    offsets1: vec4<u32>,
    strides0: vec4<u32>,
    strides1: vec4<u32>,
};

const WG_SIZE: u32 = 64u;
const MAX_SEQ_KV: u32 = 4096u;
const MAX_D_HEAD: u32 = 512u;
const NEG_INF: f32 = -3.402823466e+38;
const FLT_MAX: f32 = 3.402823466e+38;

@group(0) @binding(0) var<storage, read>       Q    : array<f32>;
@group(0) @binding(1) var<storage, read>       K    : array<f32>;
@group(0) @binding(2) var<storage, read>       V    : array<f32>;
@group(0) @binding(3) var<storage, read>       MASK : array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT  : array<f32>;
@group(0) @binding(5) var<uniform>             p    : AttentionParams;

var<workgroup> reduce_buf : array<f32, WG_SIZE>;
var<workgroup> score_buf  : array<f32, MAX_SEQ_KV>;
var<workgroup> q_buf      : array<f32, MAX_D_HEAD>;

fn isFiniteVal(x: f32) -> bool {
    return x == x && abs(x) <= FLT_MAX;
}

fn scoreFor(qi: u32, s: u32) -> f32 {
    let d_head = p.dims0.x;
    let has_mask = p.dims0.w;
    let scale = p.scale_pad.x;
    let k_off = p.offsets0.y;
    let mask_off = p.offsets0.w;
    let k_rs = p.strides0.x;
    let k_cs = p.strides0.y;
    let mask_rs = p.strides1.x;
    let mask_cs = p.strides1.y;

    var dot: f32 = 0.0;
    for (var r: u32 = 0u; r < d_head; r = r + 1u) {
        let k_idx = k_off + r * k_rs + s * k_cs;
        dot += q_buf[r] * K[k_idx];
    }

    var score = dot * scale;
    if (has_mask != 0u) {
        let mask_add = MASK[mask_off + qi * mask_cs + s * mask_rs];
        if (!isFiniteVal(mask_add)) {
            return NEG_INF;
        }
        score += mask_add;
        if (!isFiniteVal(score)) {
            return NEG_INF;
        }
    }
    return score;
}

fn reduceMax(tid: u32) {
    if (tid < 32u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 32u]); }
    workgroupBarrier();
    if (tid < 16u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 16u]); }
    workgroupBarrier();
    if (tid < 8u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 8u]); }
    workgroupBarrier();
    if (tid < 4u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 4u]); }
    workgroupBarrier();
    if (tid < 2u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 2u]); }
    workgroupBarrier();
    if (tid < 1u) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 1u]); }
    workgroupBarrier();
}

fn reduceSum(tid: u32) {
    if (tid < 32u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 1u]; }
    workgroupBarrier();
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let seq_q = p.dims0.y;
    let seq_kv = p.dims0.z;
    let d_head = p.dims0.x;
    let q_off = p.offsets0.x;
    let v_off = p.offsets0.z;
    let dst_off = p.offsets1.x;
    let q_rs = p.offsets1.y;
    let q_cs = p.offsets1.z;
    let v_rs = p.strides0.z;
    let v_cs = p.strides0.w;
    let dst_rs = p.strides1.z;
    let dst_cs = p.strides1.w;
    let qi = workgroup_id.x;
    let tid = local_id.x;
    if (qi >= seq_q || seq_kv > MAX_SEQ_KV || d_head > MAX_D_HEAD) {
        return;
    }

    for (var r = tid; r < d_head; r = r + WG_SIZE) {
        q_buf[r] = Q[q_off + r * q_rs + qi * q_cs];
    }
    workgroupBarrier();

    var local_max = NEG_INF;
    for (var s = tid; s < seq_kv; s = s + WG_SIZE) {
        let score = scoreFor(qi, s);
        score_buf[s] = score;
        local_max = max(local_max, score);
    }
    reduce_buf[tid] = local_max;
    workgroupBarrier();
    reduceMax(tid);
    let max_score = reduce_buf[0];

    var local_sum: f32 = 0.0;
    if (max_score != NEG_INF) {
        for (var s = tid; s < seq_kv; s = s + WG_SIZE) {
            let w = exp(score_buf[s] - max_score);
            score_buf[s] = w;
            local_sum += w;
        }
    } else {
        for (var s = tid; s < seq_kv; s = s + WG_SIZE) {
            score_buf[s] = 0.0;
        }
    }
    reduce_buf[tid] = local_sum;
    workgroupBarrier();
    reduceSum(tid);

    let sum_weights = reduce_buf[0];
    let inv_sum = select(0.0, 1.0 / sum_weights, sum_weights > 0.0);

    for (var r = tid; r < d_head; r = r + WG_SIZE) {
        var acc: f32 = 0.0;
        for (var s: u32 = 0u; s < seq_kv; s = s + 1u) {
            let w = score_buf[s] * inv_sum;
            let v_idx = v_off + r * v_rs + s * v_cs;
            acc += w * V[v_idx];
        }
        OUT[dst_off + r * dst_rs + qi * dst_cs] = acc;
    }
}
