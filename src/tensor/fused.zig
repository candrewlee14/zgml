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
const compiler = @import("../compiler.zig");
const Op = @import("../op.zig").Op;
const Tensor = @import("../tensor.zig").Tensor;
const forward = @import("forward.zig");

const GELU_COEF_A: comptime_float = 0.044715;
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

/// All ops that can participate in a fused chain.
pub const fusible_ops = [_]Op{
    .neg, .abs, .sgn, .step, .sqrt, .recip, .exp, .log, .gelu, .add, .mul,
};

pub const FusionKind = enum {
    elementwise_chain,
    conv2d,
    conv2d_bwd_input,
    conv2d_bwd_kernel,
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

pub fn Conv2dPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        kernel: *Tensor(T),
        input_view: *Tensor(T),
        kernel_view: *Tensor(T),
        bias: ?*Tensor(T),
        bias_node: ?*Tensor(T),
        activation: ?*Tensor(T),
        mul_node: *Tensor(T),
        sum_node: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn Conv2dBwdInputPlan(comptime T: type) type {
    return struct {
        output_grad: *Tensor(T),
        kernel: *Tensor(T),
        reshape_node: *Tensor(T),
        repeat_node: *Tensor(T),
        mul_node: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn Conv2dBwdKernelPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        output_grad: *Tensor(T),
        reshape_node: *Tensor(T),
        repeat_node: *Tensor(T),
        mul_node: *Tensor(T),
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
        conv2d: Conv2dPlan(T),
        conv2d_bwd_input: Conv2dBwdInputPlan(T),
        conv2d_bwd_kernel: Conv2dBwdKernelPlan(T),
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

/// Validate that a softmax-family plan's intermediate nodes have consistent shapes.
/// The execution kernel iterates over input dimensions and indexes into all
/// full-sized intermediates — if any have a different shape, we'd go OOB.
fn validateSoftmaxPlan(comptime T: type, plan: SoftmaxPlan(T)) bool {
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_max, plan.neg_rep_max, plan.shifted, plan.exp_node, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}

fn validateLogSoftmaxPlan(comptime T: type, plan: LogSoftmaxPlan(T)) bool {
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_max, plan.neg_rep_max, plan.shifted, plan.exp_node, plan.rep_log, plan.neg_rep_log, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}

fn validateLayerNormPlan(comptime T: type, plan: LayerNormPlan(T)) bool {
    if (plan.mean_node.source1() == null or plan.var_node.source1() == null) return false;
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_mean, plan.neg_rep_mean, plan.centered, plan.sqr_node, plan.rep_std_inv, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}

pub fn CompilerMappedPlan(comptime T: type) type {
    return struct {
        start_idx: usize,
        plan: FusionPlan(T),
    };
}

fn nodeForValueId(comptime T: type, value_to_tensor: []const ?*Tensor(T), id: compiler.ValueId) ?*Tensor(T) {
    const idx = @intFromEnum(id);
    if (idx >= value_to_tensor.len) return null;
    return value_to_tensor[idx];
}

pub fn mapCompilerPattern(comptime T: type, forward_nodes: []const *Tensor(T), value_to_tensor: []const ?*Tensor(T), pattern: compiler.KernelPattern) ?CompilerMappedPlan(T) {
    return switch (pattern) {
        .softmax => |spec| blk: {
            const max_node = nodeForValueId(T, value_to_tensor, spec.max_node) orelse return null;
            const rep_max = nodeForValueId(T, value_to_tensor, spec.rep_max) orelse return null;
            const neg_rep_max = nodeForValueId(T, value_to_tensor, spec.neg_rep_max) orelse return null;
            const shifted = nodeForValueId(T, value_to_tensor, spec.shifted) orelse return null;
            const exp_node = nodeForValueId(T, value_to_tensor, spec.exp_node) orelse return null;
            const sum_node = nodeForValueId(T, value_to_tensor, spec.sum_node) orelse return null;
            const rep_sum = nodeForValueId(T, value_to_tensor, spec.rep_sum) orelse return null;
            const recip_rep_sum = nodeForValueId(T, value_to_tensor, spec.recip_rep_sum) orelse return null;
            const output = nodeForValueId(T, value_to_tensor, spec.output) orelse return null;
            const start_idx = indexOfNode(T, forward_nodes, max_node);
            const output_idx = indexOfNode(T, forward_nodes, output);

            const softmax_plan = SoftmaxPlan(T){
                .input = max_node.source0().?,
                .max_node = max_node,
                .rep_max = rep_max,
                .neg_rep_max = neg_rep_max,
                .shifted = shifted,
                .exp_node = exp_node,
                .sum_node = sum_node,
                .rep_sum = rep_sum,
                .recip_rep_sum = recip_rep_sum,
                .output = output,
            };
            if (!validateSoftmaxPlan(T, softmax_plan)) return null;
            break :blk .{ .start_idx = start_idx, .plan = .{
                .output_idx = output_idx,
                .payload = .{ .softmax = softmax_plan },
            } };
        },
        .log_softmax => |spec| blk: {
            const max_node = nodeForValueId(T, value_to_tensor, spec.max_node) orelse return null;
            const rep_max = nodeForValueId(T, value_to_tensor, spec.rep_max) orelse return null;
            const neg_rep_max = nodeForValueId(T, value_to_tensor, spec.neg_rep_max) orelse return null;
            const shifted = nodeForValueId(T, value_to_tensor, spec.shifted) orelse return null;
            const exp_node = nodeForValueId(T, value_to_tensor, spec.exp_node) orelse return null;
            const sum_node = nodeForValueId(T, value_to_tensor, spec.sum_node) orelse return null;
            const log_node = nodeForValueId(T, value_to_tensor, spec.log_node) orelse return null;
            const rep_log = nodeForValueId(T, value_to_tensor, spec.rep_log) orelse return null;
            const neg_rep_log = nodeForValueId(T, value_to_tensor, spec.neg_rep_log) orelse return null;
            const output = nodeForValueId(T, value_to_tensor, spec.output) orelse return null;
            const start_idx = indexOfNode(T, forward_nodes, max_node);
            const output_idx = indexOfNode(T, forward_nodes, output);

            const log_softmax_plan = LogSoftmaxPlan(T){
                .input = max_node.source0().?,
                .max_node = max_node,
                .rep_max = rep_max,
                .neg_rep_max = neg_rep_max,
                .shifted = shifted,
                .exp_node = exp_node,
                .sum_node = sum_node,
                .log_node = log_node,
                .rep_log = rep_log,
                .neg_rep_log = neg_rep_log,
                .output = output,
            };
            if (!validateLogSoftmaxPlan(T, log_softmax_plan)) return null;
            break :blk .{ .start_idx = start_idx, .plan = .{
                .output_idx = output_idx,
                .payload = .{ .log_softmax = log_softmax_plan },
            } };
        },
        .cross_entropy => |spec| blk: {
            const max_node = nodeForValueId(T, value_to_tensor, spec.log_softmax.max_node) orelse return null;
            const rep_max = nodeForValueId(T, value_to_tensor, spec.log_softmax.rep_max) orelse return null;
            const neg_rep_max = nodeForValueId(T, value_to_tensor, spec.log_softmax.neg_rep_max) orelse return null;
            const shifted = nodeForValueId(T, value_to_tensor, spec.log_softmax.shifted) orelse return null;
            const exp_node = nodeForValueId(T, value_to_tensor, spec.log_softmax.exp_node) orelse return null;
            const sum_node = nodeForValueId(T, value_to_tensor, spec.log_softmax.sum_node) orelse return null;
            const log_node = nodeForValueId(T, value_to_tensor, spec.log_softmax.log_node) orelse return null;
            const rep_log = nodeForValueId(T, value_to_tensor, spec.log_softmax.rep_log) orelse return null;
            const neg_rep_log = nodeForValueId(T, value_to_tensor, spec.log_softmax.neg_rep_log) orelse return null;
            const log_softmax_output = nodeForValueId(T, value_to_tensor, spec.log_softmax.output) orelse return null;
            const picked = nodeForValueId(T, value_to_tensor, spec.picked) orelse return null;
            const neg_picked = nodeForValueId(T, value_to_tensor, spec.neg_picked) orelse return null;
            const sum_node_ce = nodeForValueId(T, value_to_tensor, spec.sum_node) orelse return null;
            const mean_node = nodeForValueId(T, value_to_tensor, spec.mean_node) orelse return null;
            const start_idx = indexOfNode(T, forward_nodes, max_node);
            const output_idx = indexOfNode(T, forward_nodes, mean_node);

            const inner_log_softmax = LogSoftmaxPlan(T){
                .input = max_node.source0().?,
                .max_node = max_node,
                .rep_max = rep_max,
                .neg_rep_max = neg_rep_max,
                .shifted = shifted,
                .exp_node = exp_node,
                .sum_node = sum_node,
                .log_node = log_node,
                .rep_log = rep_log,
                .neg_rep_log = neg_rep_log,
                .output = log_softmax_output,
            };
            if (!validateLogSoftmaxPlan(T, inner_log_softmax)) return null;
            if (picked.source1() == null or mean_node.source1() == null) return null;
            break :blk .{ .start_idx = start_idx, .plan = .{
                .output_idx = output_idx,
                .payload = .{ .cross_entropy = .{
                    .log_softmax = inner_log_softmax,
                    .targets = picked.source1().?,
                    .picked = picked,
                    .neg_picked = neg_picked,
                    .sum_node = sum_node_ce,
                    .mean_node = mean_node,
                } },
            } };
        },
        .layer_norm => |spec| blk: {
            const sum_node = nodeForValueId(T, value_to_tensor, spec.sum_node) orelse return null;
            const mean_node = nodeForValueId(T, value_to_tensor, spec.mean_node) orelse return null;
            const rep_mean = nodeForValueId(T, value_to_tensor, spec.rep_mean) orelse return null;
            const neg_rep_mean = nodeForValueId(T, value_to_tensor, spec.neg_rep_mean) orelse return null;
            const centered = nodeForValueId(T, value_to_tensor, spec.centered) orelse return null;
            const sqr_node = nodeForValueId(T, value_to_tensor, spec.sqr_node) orelse return null;
            const var_sum = nodeForValueId(T, value_to_tensor, spec.var_sum) orelse return null;
            const var_node = nodeForValueId(T, value_to_tensor, spec.var_node) orelse return null;
            const eps_like = nodeForValueId(T, value_to_tensor, spec.eps_like) orelse return null;
            const var_eps = nodeForValueId(T, value_to_tensor, spec.var_eps) orelse return null;
            const sqrt_node = nodeForValueId(T, value_to_tensor, spec.sqrt_node) orelse return null;
            const recip_node = nodeForValueId(T, value_to_tensor, spec.recip_node) orelse return null;
            const rep_std_inv = nodeForValueId(T, value_to_tensor, spec.rep_std_inv) orelse return null;
            const output = nodeForValueId(T, value_to_tensor, spec.output) orelse return null;
            const start_idx = indexOfNode(T, forward_nodes, sum_node);
            const output_idx = indexOfNode(T, forward_nodes, output);

            if (sum_node.source0() == null) return null;
            const ln_plan = LayerNormPlan(T){
                .input = sum_node.source0().?,
                .sum_node = sum_node,
                .mean_node = mean_node,
                .rep_mean = rep_mean,
                .neg_rep_mean = neg_rep_mean,
                .centered = centered,
                .sqr_node = sqr_node,
                .var_sum = var_sum,
                .var_node = var_node,
                .eps_like = eps_like,
                .var_eps = var_eps,
                .sqrt_node = sqrt_node,
                .recip_node = recip_node,
                .rep_std_inv = rep_std_inv,
                .output = output,
            };
            if (!validateLayerNormPlan(T, ln_plan)) return null;
            break :blk .{ .start_idx = start_idx, .plan = .{
                .output_idx = output_idx,
                .payload = .{ .layer_norm = ln_plan },
            } };
        },
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

fn executeConv2dPlan(comptime T: type, plan: Conv2dPlan(T)) void {
    _ = plan.input_view;
    _ = plan.kernel_view;
    _ = plan.mul_node;

    const input = plan.input;
    const kernel = plan.kernel;
    const output = plan.output;

    const out_w = output.ne[0];
    const out_h = output.ne[1];
    const c_out = output.ne[2];
    const batch = output.ne[3];
    const kw = kernel.ne[0];
    const kh = kernel.ne[1];
    const c_in = kernel.ne[2];
    const in_w = input.ne[0];

    const K = kw * kh * c_in; // rows of im2col / inner dimension
    const N = out_w * out_h; // cols of im2col / spatial positions

    // im2col scratch buffer [K, N] — one batch element at a time.
    const col_buf = std.heap.page_allocator.alloc(T, K * N) catch {
        executeConv2dNaive(T, plan);
        return;
    };
    defer std.heap.page_allocator.free(col_buf);

    // Kernel layout: [kw, kh, c_in, c_out] contiguous.
    // As [c_out, K] matrix: row oc at offset oc*K, elements contiguous.
    // a_m_stride = K, a_k_stride = 1.

    // Output layout: [out_w, out_h, c_out, batch] contiguous.
    // Per batch slice is [c_out, N] at offset n*N*c_out with row stride N.
    // This matches the matmul's natural output layout.

    const matmul = forward.selectMatMulKernel(T);
    const contiguous_x = (input.strides[0] == 1);

    for (0..batch) |n| {
        // --- Build im2col column matrix ---
        // col_buf[k, col] = input[ox+kx, oy+ky, ic, n]
        // where k = kx + ky*kw + ic*kw*kh, col = ox + oy*out_w
        if (contiguous_x) {
            // Fast path: memcpy entire rows of out_w elements.
            for (0..c_in) |ic| {
                for (0..kh) |ky| {
                    for (0..kw) |kx| {
                        const k = kx + ky * kw + ic * kw * kh;
                        const row_off = k * N;
                        for (0..out_h) |oy| {
                            const src = kx + (oy + ky) * in_w + ic * input.strides[2] + n * input.strides[3];
                            const dst = row_off + oy * out_w;
                            @memcpy(col_buf[dst..][0..out_w], input.data[src..][0..out_w]);
                        }
                    }
                }
            }
        } else {
            for (0..c_in) |ic| {
                for (0..kh) |ky| {
                    for (0..kw) |kx| {
                        const k = kx + ky * kw + ic * kw * kh;
                        const row_off = k * N;
                        for (0..out_h) |oy| {
                            for (0..out_w) |ox| {
                                col_buf[row_off + oy * out_w + ox] = input.data[
                                    (ox + kx) * input.strides[0] +
                                        (oy + ky) * input.strides[1] +
                                        ic * input.strides[2] +
                                        n * input.strides[3]
                                ];
                            }
                        }
                    }
                }
            }
        }

        // --- Matmul: kernel[c_out, K] @ col_buf[K, N] → output[c_out, N] ---
        matmul(
            output.data, // dst
            kernel.data, // A = kernel as [c_out, K]
            col_buf, // B = im2col  [K, N]
            c_out, // M
            N, // N
            K, // K
            K, // a_m_stride (kernel row stride = kw*kh*c_in)
            1, // a_k_stride
            N, // b_k_stride (col_buf row stride)
            1, // b_n_stride
            0, // a_base
            0, // b_base
            n * N * c_out, // d_base (batch offset into output)
            N, // d_row_stride
        );

        if (plan.bias != null or plan.activation != null) {
            for (0..c_out) |oc| {
                for (0..N) |col| {
                    const out_idx = n * N * c_out + oc * N + col;
                    var val = output.data[out_idx];
                    if (plan.bias) |bias| val += bias.data[oc];
                    if (plan.activation != null) val = if (val > 0) val else 0;
                    output.data[out_idx] = val;
                }
            }
        }
    }
}

fn executeConv2dBwdInputPlan(comptime T: type, plan: Conv2dBwdInputPlan(T)) void {
    _ = plan.reshape_node;
    _ = plan.repeat_node;
    _ = plan.mul_node;

    const out_grad = plan.output_grad;
    const kernel = plan.kernel;
    const output = plan.output;

    const out_w = out_grad.ne[0];
    const out_h = out_grad.ne[1];
    const c_out = out_grad.ne[2];
    const batch = out_grad.ne[3];
    const kw = kernel.ne[0];
    const kh = kernel.ne[1];
    const c_in = kernel.ne[2];

    @memset(output.data, 0);
    for (0..batch) |n| {
        for (0..c_out) |oc| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    const grad_idx = ox * out_grad.strides[0] + oy * out_grad.strides[1] + oc * out_grad.strides[2] + n * out_grad.strides[3];
                    const grad_val = out_grad.data[grad_idx];
                    for (0..c_in) |ic| {
                        for (0..kh) |ky| {
                            for (0..kw) |kx| {
                                const out_idx = (ox + kx) * output.strides[0] +
                                    (oy + ky) * output.strides[1] +
                                    ic * output.strides[2] +
                                    n * output.strides[3];
                                const kernel_idx = kx * kernel.strides[0] +
                                    ky * kernel.strides[1] +
                                    ic * kernel.strides[2] +
                                    oc * kernel.strides[3];
                                output.data[out_idx] += grad_val * kernel.data[kernel_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn executeConv2dBwdKernelPlan(comptime T: type, plan: Conv2dBwdKernelPlan(T)) void {
    _ = plan.reshape_node;
    _ = plan.repeat_node;
    _ = plan.mul_node;

    const input = plan.input;
    const out_grad = plan.output_grad;
    const output = plan.output;

    const out_w = out_grad.ne[0];
    const out_h = out_grad.ne[1];
    const c_out = out_grad.ne[2];
    const batch = out_grad.ne[3];
    const kw = output.ne[0];
    const kh = output.ne[1];
    const c_in = output.ne[2];

    @memset(output.data, 0);
    for (0..c_out) |oc| {
        for (0..c_in) |ic| {
            for (0..kh) |ky| {
                for (0..kw) |kx| {
                    var acc: T = 0;
                    for (0..batch) |n| {
                        for (0..out_h) |oy| {
                            for (0..out_w) |ox| {
                                const input_idx = (ox + kx) * input.strides[0] +
                                    (oy + ky) * input.strides[1] +
                                    ic * input.strides[2] +
                                    n * input.strides[3];
                                const grad_idx = ox * out_grad.strides[0] + oy * out_grad.strides[1] + oc * out_grad.strides[2] + n * out_grad.strides[3];
                                acc += input.data[input_idx] * out_grad.data[grad_idx];
                            }
                        }
                    }
                    const out_idx = kx * output.strides[0] + ky * output.strides[1] + ic * output.strides[2] + oc * output.strides[3];
                    output.data[out_idx] = acc;
                }
            }
        }
    }
}

/// Fallback naive conv2d if scratch allocation fails.
fn executeConv2dNaive(comptime T: type, plan: Conv2dPlan(T)) void {
    const input = plan.input;
    const kernel = plan.kernel;
    const output = plan.output;

    const out_w = output.ne[0];
    const out_h = output.ne[1];
    const c_out = output.ne[2];
    const batch = output.ne[3];
    const kw = kernel.ne[0];
    const kh = kernel.ne[1];
    const c_in = kernel.ne[2];

    for (0..batch) |n| {
        for (0..c_out) |oc| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var acc: T = 0;
                    for (0..c_in) |ic| {
                        for (0..kh) |ky| {
                            for (0..kw) |kx| {
                                acc += input.data[
                                    (ox + kx) * input.strides[0] +
                                        (oy + ky) * input.strides[1] +
                                        ic * input.strides[2] +
                                        n * input.strides[3]
                                ] *
                                    kernel.data[
                                        kx * kernel.strides[0] +
                                            ky * kernel.strides[1] +
                                            ic * kernel.strides[2] +
                                            oc * kernel.strides[3]
                                    ];
                            }
                        }
                    }
                    output.data[ox + oy * out_w + oc * out_w * out_h + n * out_w * out_h * c_out] = acc;
                }
            }
        }
    }
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
        .conv2d => |conv2d_plan| executeConv2dPlan(T, conv2d_plan),
        .conv2d_bwd_input => |conv2d_plan| executeConv2dBwdInputPlan(T, conv2d_plan),
        .conv2d_bwd_kernel => |conv2d_plan| executeConv2dBwdKernelPlan(T, conv2d_plan),
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
