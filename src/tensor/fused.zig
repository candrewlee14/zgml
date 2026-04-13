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
};

/// A runtime descriptor for a detected fused chain.
pub fn FusionPlan(comptime T: type) type {
    const Tensor = @import("../tensor.zig").Tensor;
    return struct {
        /// Ordered nodes in the chain. nodes[0].src0 is the chain input.
        nodes: []const *Tensor(T),
        /// Index of the chain's output node in the graph's node list.
        output_idx: usize,
        kind: FusionKind = .elementwise_chain,
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
    const Tensor = @import("../tensor.zig").Tensor;
    return struct {
        pub fn execute(nodes: []const *Tensor(T)) void {
            const n_elems = nodes[nodes.len - 1].nElems();
            const input_data = nodes[0].src0.?.data;

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
pub fn executeFusedGeneric(comptime T: type, nodes: []const *@import("../tensor.zig").Tensor(T)) void {
    const n_elems = nodes[nodes.len - 1].nElems();
    const input_data = nodes[0].src0.?.data;

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
pub fn executeFusedChain(comptime T: type, chain_nodes: []const *@import("../tensor.zig").Tensor(T)) void {
    switch (chain_nodes.len) {
        2 => executeFused2(T, chain_nodes),
        3 => executeFused3(T, chain_nodes),
        else => executeFusedGeneric(T, chain_nodes),
    }
}

fn offset4(strides: [4]usize, c0: usize, c1: usize, c2: usize, c3: usize) usize {
    return c0 * strides[0] + c1 * strides[1] + c2 * strides[2] + c3 * strides[3];
}

fn executeSoftmaxPlan(comptime T: type, nodes: []const *@import("../tensor.zig").Tensor(T), comptime log_mode: bool) void {
    const input = nodes[0].src0.?;
    const max_node = nodes[0];
    const rep_max = nodes[1];
    const neg_rep_max = nodes[2];
    const shifted = nodes[3];
    const exp_node = nodes[4];
    const sum_node = nodes[5];

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
        const log_node = nodes[6];
        const rep_log = nodes[7];
        const neg_rep_log = nodes[8];
        const output = nodes[9];

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
        const rep_sum = nodes[6];
        const recip_rep_sum = nodes[7];
        const output = nodes[8];

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

pub fn executeFusionPlan(comptime T: type, plan: FusionPlan(T)) void {
    switch (plan.kind) {
        .elementwise_chain => executeFusedChain(T, plan.nodes),
        .softmax => executeSoftmaxPlan(T, plan.nodes, false),
        .log_softmax => executeSoftmaxPlan(T, plan.nodes, true),
    }
}

fn executeFused2(comptime T: type, nodes: []const *@import("../tensor.zig").Tensor(T)) void {
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

fn executeFused3(comptime T: type, nodes: []const *@import("../tensor.zig").Tensor(T)) void {
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
