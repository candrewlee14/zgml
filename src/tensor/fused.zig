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
const SQRT_2_OVER_PI: comptime_float = 0.79788456080286535587989211986876;

/// All ops that can participate in a fused chain.
pub const fusible_ops = [_]Op{
    .neg, .abs, .sgn, .step, .sqrt, .recip, .exp, .log, .gelu, .add, .mul,
};

/// A runtime descriptor for a detected fused chain.
pub fn FusedChain(comptime T: type) type {
    const Tensor = @import("../tensor.zig").Tensor;
    return struct {
        /// Ordered nodes in the chain. nodes[0].src0 is the chain input.
        nodes: []const *Tensor(T),
        /// Index of the chain's output node in the graph's node list.
        output_idx: usize,
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
