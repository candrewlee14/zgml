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

struct StepDynamicParams {
    slice_pos: u32,
    seq_kv: u32,
    _pad0: u32,
    _pad1: u32,
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
@group(0) @binding(6) var<uniform>             d    : StepDynamicParams;

var<workgroup> reduce_buf : array<f32, WG_SIZE>;
var<workgroup> score_buf  : array<f32, MAX_SEQ_KV>;
var<workgroup> q_buf      : array<f32, MAX_D_HEAD>;

fn isFiniteVal(x: f32) -> bool {
    return x == x && abs(x) <= FLT_MAX;
}

fn scoreFor(qi: u32, s: u32) -> f32 {
    let d_head = p.dims0.x;
    let has_mask = p.dims0.z;
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

fn scoreForContiguous(s: u32, d_head: u32, scale: f32, has_mask: u32, mask_off: u32, mask_rs: u32, mask_cs: u32) -> f32 {
    let k_base = p.offsets0.y + s * d_head;
    var dot_acc: f32 = 0.0;
    var r: u32 = 0u;
    while (r + 8u <= d_head) {
        let qv0 = vec4(
            q_buf[r],
            q_buf[r + 1u],
            q_buf[r + 2u],
            q_buf[r + 3u],
        );
        let kv0 = vec4(
            K[k_base + r],
            K[k_base + r + 1u],
            K[k_base + r + 2u],
            K[k_base + r + 3u],
        );
        let qv1 = vec4(
            q_buf[r + 4u],
            q_buf[r + 5u],
            q_buf[r + 6u],
            q_buf[r + 7u],
        );
        let kv1 = vec4(
            K[k_base + r + 4u],
            K[k_base + r + 5u],
            K[k_base + r + 6u],
            K[k_base + r + 7u],
        );
        dot_acc += dot(qv0, kv0) + dot(qv1, kv1);
        r += 8u;
    }
    while (r + 4u <= d_head) {
        let qv = vec4(
            q_buf[r],
            q_buf[r + 1u],
            q_buf[r + 2u],
            q_buf[r + 3u],
        );
        let kv = vec4(
            K[k_base + r],
            K[k_base + r + 1u],
            K[k_base + r + 2u],
            K[k_base + r + 3u],
        );
        dot_acc += dot(qv, kv);
        r += 4u;
    }
    while (r < d_head) {
        dot_acc += q_buf[r] * K[k_base + r];
        r += 1u;
    }

    var score = dot_acc * scale;
    if (has_mask != 0u) {
        let mask_add = MASK[mask_off + s * mask_rs];
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
    let seq_kv = d.seq_kv;
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

    let contiguous = q_rs == 1u && q_cs == d_head &&
        p.strides0.x == 1u && p.strides0.y == d_head &&
        v_rs == 1u && v_cs == d_head &&
        dst_rs == 1u && dst_cs == d_head;
    let scale = p.scale_pad.x;
    let has_mask = p.dims0.z;
    let mask_off = p.offsets0.w + qi * p.strides1.y;
    let mask_rs = p.strides1.x;

    for (var r = tid; r < d_head; r = r + WG_SIZE) {
        if (contiguous) {
            q_buf[r] = Q[q_off + qi * d_head + r];
        } else {
            q_buf[r] = Q[q_off + r * q_rs + qi * q_cs];
        }
    }
    workgroupBarrier();

    var local_max = NEG_INF;
    for (var s = tid; s < seq_kv; s = s + WG_SIZE) {
        var score: f32 = 0.0;
        if (contiguous) {
            score = scoreForContiguous(s, d_head, scale, has_mask, mask_off, mask_rs, p.strides1.y);
        } else {
            score = scoreFor(qi, s);
        }
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
    for (var s = tid; s < seq_kv; s = s + WG_SIZE) {
        score_buf[s] = score_buf[s] * inv_sum;
    }
    workgroupBarrier();

    if (contiguous) {
        for (var r = tid; r < d_head; r = r + WG_SIZE) {
            var acc: f32 = 0.0;
            var s: u32 = 0u;
            while (s + 4u <= seq_kv) {
                let v_base = v_off + s * d_head + r;
                acc += score_buf[s] * V[v_base];
                acc += score_buf[s + 1u] * V[v_base + d_head];
                acc += score_buf[s + 2u] * V[v_base + 2u * d_head];
                acc += score_buf[s + 3u] * V[v_base + 3u * d_head];
                s += 4u;
            }
            while (s < seq_kv) {
                acc += score_buf[s] * V[v_off + s * d_head + r];
                s += 1u;
            }
            OUT[dst_off + qi * d_head + r] = acc;
        }
    } else {
        for (var r = tid; r < d_head; r = r + WG_SIZE) {
            var acc: f32 = 0.0;
            for (var s: u32 = 0u; s < seq_kv; s = s + 1u) {
                let v_idx = v_off + r * v_rs + s * v_cs;
                acc += score_buf[s] * V[v_idx];
            }
            OUT[dst_off + r * dst_rs + qi * dst_cs] = acc;
        }
    }
}
