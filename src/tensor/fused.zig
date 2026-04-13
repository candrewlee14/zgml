//! Comptime-generated fused elementwise kernels.
//!
//! Given a comptime slice of `Op`s, `FusedKernel` generates a branch-free
//! inner loop that applies the entire chain per element. The `inline for`
//! over the op slice is fully unrolled by the compiler — the resulting
//! machine code is equivalent to a hand-written fused function.
//!
//! For chains detected at runtime, `executeFusedChain` dispatches to the
//! correct comptime-specialized kernel via bounded `inline for` lookup.

const std = @import("std");
const Op = @import("../op.zig").Op;
const Tensor = @import("../tensor.zig").Tensor;

const GELU_COEF_A: comptime_float = 0.044715;
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

/// All ops that can participate in a fused chain.
pub const fusible_ops = [_]Op{
    .neg, .abs, .sgn, .step, .sqrt, .recip, .exp, .log, .gelu, .add, .mul,
};

pub const FusionKind = enum {
    elementwise_chain,
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
};

pub fn ElementwiseChainPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        nodes: []const *Tensor(T),

        pub fn output(self: @This()) *Tensor(T) {
            return self.nodes[self.nodes.len - 1];
        }
    };
}

pub fn SoftmaxPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        max_node: *Tensor(T),
        rep_max: *Tensor(T),
        neg_rep_max: *Tensor(T),
        shifted: *Tensor(T),
        exp_node: *Tensor(T),
        sum_node: *Tensor(T),
        rep_sum: *Tensor(T),
        recip_rep_sum: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn LogSoftmaxPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        max_node: *Tensor(T),
        rep_max: *Tensor(T),
        neg_rep_max: *Tensor(T),
        shifted: *Tensor(T),
        exp_node: *Tensor(T),
        sum_node: *Tensor(T),
        log_node: *Tensor(T),
        rep_log: *Tensor(T),
        neg_rep_log: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn CrossEntropyPlan(comptime T: type) type {
    return struct {
        log_softmax: LogSoftmaxPlan(T),
        targets: *Tensor(T),
        picked: *Tensor(T),
        neg_picked: *Tensor(T),
        sum_node: *Tensor(T),
        mean_node: *Tensor(T),
    };
}

pub fn LayerNormPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        sum_node: *Tensor(T),
        mean_node: *Tensor(T),
        rep_mean: *Tensor(T),
        neg_rep_mean: *Tensor(T),
        centered: *Tensor(T),
        sqr_node: *Tensor(T),
        var_sum: *Tensor(T),
        var_node: *Tensor(T),
        eps_like: *Tensor(T),
        var_eps: *Tensor(T),
        sqrt_node: *Tensor(T),
        recip_node: *Tensor(T),
        rep_std_inv: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn FusionPayload(comptime T: type) type {
    return union(FusionKind) {
        elementwise_chain: ElementwiseChainPlan(T),
        softmax: SoftmaxPlan(T),
        log_softmax: LogSoftmaxPlan(T),
        cross_entropy: CrossEntropyPlan(T),
        layer_norm: LayerNormPlan(T),
    };
}

/// A runtime descriptor for a detected fused chain.
pub fn FusionPlan(comptime T: type) type {
    return struct {
        output_idx: usize,
        payload: FusionPayload(T),

        pub fn kind(self: @This()) FusionKind {
            return std.meta.activeTag(self.payload);
        }
    };
}

/// Apply a single op to a value. Comptime-dispatched — no runtime branching.
fn applyOp(comptime T: type, comptime op: Op, val: T, node: anytype, i: usize) T {
    return switch (op) {
        .neg => -val,
        .abs => @abs(val),
        .sgn => if (val > 0) 1 else if (val < 0) @as(T, -1) else 0,
        .step => if (val > 0) @as(T, 1) else 0,
        .sqrt => @sqrt(val),
        .recip => 1.0 / val,
        .exp => @exp(val),
        .log => @log(val),
        .gelu => 0.5 * val * (1.0 + std.math.tanh(
            @as(T, SQRT_2_OVER_PI) * val * (1.0 + @as(T, GELU_COEF_A) * val * val),
        )),
        .add => blk: {
            const src1 = node.src1.?;
            break :blk val + if (src1.isScalar()) src1.data[0] else src1.data[i];
        },
        .mul => blk: {
            // sqr: src0 == src1, both operands are the chain value
            if (node.src0.? == node.src1.?) break :blk val * val;
            const src1 = node.src1.?;
            break :blk val * if (src1.isScalar()) src1.data[0] else src1.data[i];
        },
        else => unreachable,
    };
}

/// A comptime-specialized fused kernel for a known op sequence.
/// The `inline for` is fully unrolled — the inner loop has zero branches.
pub fn FusedKernel(comptime T: type, comptime ops: []const Op) type {
    return struct {
        pub fn execute(nodes: []const *Tensor(T)) void {
            std.debug.assert(nodes[0].source0().?.n_dims <= 4);
            const n_elems = nodes[nodes.len - 1].nElems();
            const input_data = nodes[0].source0().?.data;

            for (0..n_elems) |i| {
                var val: T = input_data[i];
                inline for (ops, 0..) |op, k| {
                    val = applyOp(T, op, val, nodes[k], i);
                    nodes[k].data[i] = val;
                }
            }
        }
    };
}

/// Runtime interpreter fallback for chains longer than the comptime dispatch limit.
/// Still one memory pass, but the inner loop has a runtime switch.
pub fn executeFusedGeneric(comptime T: type, nodes: []const *Tensor(T)) void {
    std.debug.assert(nodes[0].source0().?.n_dims <= 4);
    const n_elems = nodes[nodes.len - 1].nElems();
    const input_data = nodes[0].source0().?.data;

    for (0..n_elems) |i| {
        var val: T = input_data[i];
        for (nodes) |node| {
            val = switch (node.op) {
                .neg => -val,
                .abs => @abs(val),
                .sgn => if (val > 0) 1 else if (val < 0) @as(T, -1) else 0,
                .step => if (val > 0) @as(T, 1) else 0,
                .sqrt => @sqrt(val),
                .recip => 1.0 / val,
                .exp => @exp(val),
                .log => @log(val),
                .gelu => 0.5 * val * (1.0 + std.math.tanh(
                    @as(T, SQRT_2_OVER_PI) * val * (1.0 + @as(T, GELU_COEF_A) * val * val),
                )),
                .add => blk: {
                    const src1 = node.src1.?;
                    break :blk val + if (src1.isScalar()) src1.data[0] else src1.data[i];
                },
                .mul => blk: {
                    if (node.src0.? == node.src1.?) break :blk val * val;
                    const src1 = node.src1.?;
                    break :blk val * if (src1.isScalar()) src1.data[0] else src1.data[i];
                },
                else => unreachable,
            };
            node.data[i] = val;
        }
    }
}

/// Dispatch a fused chain to a comptime-specialized kernel.
/// For chains of length 2-3, enumerates all fusible op combinations
/// at comptime. Falls back to the generic interpreter for longer chains.
pub fn executeFusedChain(comptime T: type, plan: ElementwiseChainPlan(T)) void {
    switch (plan.nodes.len) {
        2 => executeFused2(T, plan.nodes),
        3 => executeFused3(T, plan.nodes),
        else => executeFusedGeneric(T, plan.nodes),
    }
}

fn offset4(strides: [@import("../tensor.zig").max_dims]usize, c0: usize, c1: usize, c2: usize, c3: usize) usize {
    return c0 * strides[0] + c1 * strides[1] + c2 * strides[2] + c3 * strides[3];
}

fn executeSoftmaxPlanBase(comptime T: type, plan: anytype, comptime log_mode: bool) void {
    const input = plan.input;
    const max_node = plan.max_node;
    const rep_max = plan.rep_max;
    const neg_rep_max = plan.neg_rep_max;
    const shifted = plan.shifted;
    const exp_node = plan.exp_node;
    const sum_node = plan.sum_node;

    @memset(max_node.data, -std.math.inf(T));
    for (0..input.ne[3]) |d3| {
        for (0..input.ne[2]) |d2| {
            for (0..input.ne[1]) |d1| {
                for (0..input.ne[0]) |d0| {
                    const input_idx = offset4(input.strides, d0, d1, d2, d3);
                    const red_idx = offset4(max_node.strides, d0 % max_node.ne[0], d1 % max_node.ne[1], d2 % max_node.ne[2], d3 % max_node.ne[3]);
                    max_node.data[red_idx] = @max(max_node.data[red_idx], input.data[input_idx]);
                }
            }
        }
    }

    @memset(sum_node.data, 0);
    for (0..input.ne[3]) |d3| {
        for (0..input.ne[2]) |d2| {
            for (0..input.ne[1]) |d1| {
                for (0..input.ne[0]) |d0| {
                    const input_idx = offset4(input.strides, d0, d1, d2, d3);
                    const full_idx = offset4(shifted.strides, d0, d1, d2, d3);
                    const red_idx = offset4(max_node.strides, d0 % max_node.ne[0], d1 % max_node.ne[1], d2 % max_node.ne[2], d3 % max_node.ne[3]);
                    const maxv = max_node.data[red_idx];
                    const shifted_val = input.data[input_idx] - maxv;
                    const exp_val = @exp(shifted_val);

                    rep_max.data[full_idx] = maxv;
                    neg_rep_max.data[full_idx] = -maxv;
                    shifted.data[full_idx] = shifted_val;
                    exp_node.data[full_idx] = exp_val;
                    sum_node.data[red_idx] += exp_val;
                }
            }
        }
    }

    if (log_mode) {
        const log_node = plan.log_node;
        const rep_log = plan.rep_log;
        const neg_rep_log = plan.neg_rep_log;
        const output = plan.output;

        for (log_node.data, sum_node.data) |*dst, src| dst.* = @log(src);

        for (0..input.ne[3]) |d3| {
            for (0..input.ne[2]) |d2| {
                for (0..input.ne[1]) |d1| {
                    for (0..input.ne[0]) |d0| {
                        const full_idx = offset4(output.strides, d0, d1, d2, d3);
                        const red_idx = offset4(log_node.strides, d0 % log_node.ne[0], d1 % log_node.ne[1], d2 % log_node.ne[2], d3 % log_node.ne[3]);
                        const logv = log_node.data[red_idx];
                        rep_log.data[full_idx] = logv;
                        neg_rep_log.data[full_idx] = -logv;
                        output.data[full_idx] = shifted.data[full_idx] - logv;
                    }
                }
            }
        }
    } else {
        const rep_sum = plan.rep_sum;
        const recip_rep_sum = plan.recip_rep_sum;
        const output = plan.output;

        for (0..input.ne[3]) |d3| {
            for (0..input.ne[2]) |d2| {
                for (0..input.ne[1]) |d1| {
                    for (0..input.ne[0]) |d0| {
                        const full_idx = offset4(output.strides, d0, d1, d2, d3);
                        const red_idx = offset4(sum_node.strides, d0 % sum_node.ne[0], d1 % sum_node.ne[1], d2 % sum_node.ne[2], d3 % sum_node.ne[3]);
                        const denom = sum_node.data[red_idx];
                        const recip = 1.0 / denom;
                        rep_sum.data[full_idx] = denom;
                        recip_rep_sum.data[full_idx] = recip;
                        output.data[full_idx] = exp_node.data[full_idx] * recip;
                    }
                }
            }
        }
    }
}

fn executeSoftmaxPlan(comptime T: type, plan: SoftmaxPlan(T)) void {
    executeSoftmaxPlanBase(T, plan, false);
}

fn executeLogSoftmaxPlan(comptime T: type, plan: LogSoftmaxPlan(T)) void {
    executeSoftmaxPlanBase(T, plan, true);
}

fn executeCrossEntropyPlan(comptime T: type, plan: CrossEntropyPlan(T)) void {
    const logits = plan.log_softmax.input;
    const targets = plan.targets;

    executeLogSoftmaxPlan(T, plan.log_softmax);

    const picked = plan.picked;
    const neg_picked = plan.neg_picked;
    const sum_node = plan.sum_node;
    const mean_node = plan.mean_node;

    const batch = picked.ne[0];
    var total: T = 0;
    for (0..batch) |row| {
        const class_idx = if (@typeInfo(T) == .float)
            @as(usize, @intFromFloat(targets.data[row]))
        else
            @as(usize, @intCast(targets.data[row]));
        std.debug.assert(class_idx < logits.ne[0]);
        const log_probs = plan.log_softmax.output;
        const val = log_probs.data[row * log_probs.strides[1] + class_idx];
        picked.data[row] = val;
        neg_picked.data[row] = -val;
        total += -val;
    }
    sum_node.data[0] = total;
    mean_node.data[0] = total * mean_node.src1.?.data[0];
}

fn scalarValueFromBroadcast(comptime T: type, node: *Tensor(T)) T {
    if (node.isScalar()) return node.data[0];
    std.debug.assert(node.isOp(.repeat));
    std.debug.assert(node.source0().?.isScalar());
    return node.source0().?.data[0];
}

fn findNode(comptime T: type, nodes: []const *Tensor(T), op: Op) ?*Tensor(T) {
    for (nodes) |node| {
        if (node.isOp(op)) return node;
    }
    return null;
}

fn findNodeAfter(comptime T: type, nodes: []const *Tensor(T), start_idx: usize, op: Op, src: *Tensor(T)) ?*Tensor(T) {
    for (nodes[start_idx..]) |node| {
        if (!node.isOp(op)) continue;
        if (node.source0() == src or node.source1() == src) return node;
    }
    return null;
}

fn indexOfNode(comptime T: type, nodes: []const *Tensor(T), needle: *Tensor(T)) usize {
    for (nodes, 0..) |node, i| {
        if (node == needle) return i;
    }
    unreachable;
}

fn executeLayerNormPlan(comptime T: type, plan: LayerNormPlan(T)) void {
    const input = plan.input;
    const sum_node = plan.sum_node;
    const mean_node = plan.mean_node;
    const rep_mean = plan.rep_mean;
    const neg_rep_mean = plan.neg_rep_mean;
    const centered = plan.centered;
    const sqr_node = plan.sqr_node;
    const var_sum = plan.var_sum;
    const var_node = plan.var_node;
    const rep_eps = plan.eps_like;
    const var_eps = plan.var_eps;
    const sqrt_node = plan.sqrt_node;
    const recip_node = plan.recip_node;
    const rep_std_inv = plan.rep_std_inv;
    const output = plan.output;

    const mean_scale = scalarValueFromBroadcast(T, mean_node.source1().?);
    const var_scale = scalarValueFromBroadcast(T, var_node.source1().?);
    const eps = scalarValueFromBroadcast(T, rep_eps);

    @memset(sum_node.data, 0);
    for (0..input.ne[3]) |d3| {
        for (0..input.ne[2]) |d2| {
            for (0..input.ne[1]) |d1| {
                for (0..input.ne[0]) |d0| {
                    const input_idx = offset4(input.strides, d0, d1, d2, d3);
                    const red_idx = offset4(sum_node.strides, d0 % sum_node.ne[0], d1 % sum_node.ne[1], d2 % sum_node.ne[2], d3 % sum_node.ne[3]);
                    sum_node.data[red_idx] += input.data[input_idx];
                }
            }
        }
    }
    for (mean_node.data, sum_node.data) |*dst, src| dst.* = src * mean_scale;

    @memset(var_sum.data, 0);
    for (0..input.ne[3]) |d3| {
        for (0..input.ne[2]) |d2| {
            for (0..input.ne[1]) |d1| {
                for (0..input.ne[0]) |d0| {
                    const full_idx = offset4(centered.strides, d0, d1, d2, d3);
                    const red_idx = offset4(mean_node.strides, d0 % mean_node.ne[0], d1 % mean_node.ne[1], d2 % mean_node.ne[2], d3 % mean_node.ne[3]);
                    const mu = mean_node.data[red_idx];
                    const centered_val = input.data[offset4(input.strides, d0, d1, d2, d3)] - mu;
                    const sq = centered_val * centered_val;
                    rep_mean.data[full_idx] = mu;
                    neg_rep_mean.data[full_idx] = -mu;
                    centered.data[full_idx] = centered_val;
                    sqr_node.data[full_idx] = sq;
                    var_sum.data[red_idx] += sq;
                }
            }
        }
    }
    for (var_node.data, var_sum.data) |*dst, src| dst.* = src * var_scale;
    for (0..var_node.data.len) |i| {
        rep_eps.data[i] = eps;
        var_eps.data[i] = var_node.data[i] + eps;
        sqrt_node.data[i] = @sqrt(var_eps.data[i]);
        recip_node.data[i] = 1.0 / sqrt_node.data[i];
    }

    for (0..input.ne[3]) |d3| {
        for (0..input.ne[2]) |d2| {
            for (0..input.ne[1]) |d1| {
                for (0..input.ne[0]) |d0| {
                    const full_idx = offset4(output.strides, d0, d1, d2, d3);
                    const red_idx = offset4(recip_node.strides, d0 % recip_node.ne[0], d1 % recip_node.ne[1], d2 % recip_node.ne[2], d3 % recip_node.ne[3]);
                    const std_inv = recip_node.data[red_idx];
                    rep_std_inv.data[full_idx] = std_inv;
                    output.data[full_idx] = centered.data[full_idx] * std_inv;
                }
            }
        }
    }
}

pub fn executeFusionPlan(comptime T: type, plan: FusionPlan(T)) void {
    switch (plan.payload) {
        .elementwise_chain => |chain_plan| executeFusedChain(T, chain_plan),
        .softmax => |softmax_plan| executeSoftmaxPlan(T, softmax_plan),
        .log_softmax => |log_softmax_plan| executeLogSoftmaxPlan(T, log_softmax_plan),
        .cross_entropy => |cross_entropy_plan| executeCrossEntropyPlan(T, cross_entropy_plan),
        .layer_norm => |layer_norm_plan| executeLayerNormPlan(T, layer_norm_plan),
    }
}

fn executeFused2(comptime T: type, nodes: []const *Tensor(T)) void {
    inline for (fusible_ops) |op0| {
        inline for (fusible_ops) |op1| {
            if (nodes[0].op == op0 and nodes[1].op == op1) {
                FusedKernel(T, &.{ op0, op1 }).execute(nodes);
                return;
            }
        }
    }
    executeFusedGeneric(T, nodes);
}

fn executeFused3(comptime T: type, nodes: []const *Tensor(T)) void {
    @setEvalBranchQuota(20000);
    inline for (fusible_ops) |op0| {
        inline for (fusible_ops) |op1| {
            inline for (fusible_ops) |op2| {
                if (nodes[0].op == op0 and nodes[1].op == op1 and nodes[2].op == op2) {
                    FusedKernel(T, &.{ op0, op1, op2 }).execute(nodes);
                    return;
                }
            }
        }
    }
    executeFusedGeneric(T, nodes);
}
