//! Comptime-generated fused elementwise kernels.
//!
//! Given a comptime slice of `Op`s, `FusedKernel` generates a branch-free
//! inner loop that applies the entire chain per element. The `inline for`
//! over the op slice is fully unrolled by the compiler — the resulting
//! machine code is equivalent to a hand-written fused function.
//!
//! Schedule-owned elementwise regions reuse `executeFusedChain`, which
//! dispatches to the correct comptime-specialized kernel via bounded
//! `inline for` lookup.

const std = @import("std");
const Op = @import("../op.zig").Op;
const Tensor = @import("../tensor.zig").Tensor;
const forward = @import("forward.zig");

const GELU_COEF_A: comptime_float = 0.044715;
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

const FusedOp = enum {
    neg,
    abs,
    sgn,
    step,
    sqrt,
    recip,
    exp,
    log,
    gelu,
    add_src1,
    add_src0,
    mul_src1,
    mul_src0,
};

/// All ops that can participate in a fused chain.
pub const fusible_ops = [_]FusedOp{
    .neg,
    .abs,
    .sgn,
    .step,
    .sqrt,
    .recip,
    .exp,
    .log,
    .gelu,
    .add_src1,
    .add_src0,
    .mul_src1,
    .mul_src0,
};

pub const FusionKind = enum {
    elementwise_chain,
    conv2d,
    conv2d_bwd_input,
    conv2d_bwd_kernel,
    max_pool2d,
    max_pool2d_bwd,
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
};

pub const BinaryOperandRole = enum {
    src0,
    src1,
};

pub fn ElementwiseFusionPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        nodes: []const *Tensor(T),
        other_operand_roles: []const BinaryOperandRole = &.{},

        pub fn output(self: @This()) *Tensor(T) {
            return self.nodes[self.nodes.len - 1];
        }

        pub fn otherOperandRole(self: @This(), idx: usize) BinaryOperandRole {
            return if (idx < self.other_operand_roles.len) self.other_operand_roles[idx] else .src1;
        }

        pub fn otherOperand(self: @This(), idx: usize) ?*Tensor(T) {
            if (idx >= self.nodes.len) return null;
            return switch (self.otherOperandRole(idx)) {
                .src0 => self.nodes[idx].source0(),
                .src1 => self.nodes[idx].source1(),
            };
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
        scratch: ?[]T = null, // pre-allocated im2col buffer [K, N]
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
        scratch: ?[]T = null, // pre-allocated col buffer [K, N]
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
        scratch: ?[]T = null, // pre-allocated im2col buffer [K, N*batch]
    };
}

pub fn MaxPool2dPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T),
        strided: *Tensor(T),
        max_node: *Tensor(T),
        output: *Tensor(T),
    };
}

pub fn MaxPool2dBwdPlan(comptime T: type) type {
    return struct {
        input: *Tensor(T), // forward input [in_w, in_h, C, N]
        output_grad: *Tensor(T), // gradient of pool output [out_w, out_h, C, N]
        output: *Tensor(T), // gradient of pool input (scatter target) [in_w, in_h, C, N]
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
        elementwise_chain: ElementwiseFusionPlan(T),
        conv2d: Conv2dPlan(T),
        conv2d_bwd_input: Conv2dBwdInputPlan(T),
        conv2d_bwd_kernel: Conv2dBwdKernelPlan(T),
        max_pool2d: MaxPool2dPlan(T),
        max_pool2d_bwd: MaxPool2dBwdPlan(T),
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

        /// Return tensors that fused kernels read from at execution time.
        /// DCE must not eliminate these even if they have no graph consumers.
        pub fn liveRefs(self: @This()) []const *Tensor(T) {
            return switch (self.payload) {
                .conv2d_bwd_kernel => |p| &.{ p.input, p.output_grad },
                .conv2d_bwd_input => |p| &.{ p.output_grad, p.kernel },
                .max_pool2d_bwd => |p| &.{ p.input, p.output_grad },
                .elementwise_chain => |p| &.{p.input},
                else => &.{},
            };
        }

        /// Pre-allocate scratch buffers from the given arena so execution
        /// doesn't hit page_allocator (mmap/munmap) on every call.
        pub fn allocScratchBuffers(self: *@This(), alloc: std.mem.Allocator) !void {
            switch (self.payload) {
                .conv2d => |*p| {
                    const d = Conv2dDims(T).fromForward(p.*);
                    const NB = d.N * d.batch;
                    // im2col [K, NB] + matmul temp [c_out, NB]
                    p.scratch = try alloc.alloc(T, (d.K + d.c_out) * NB);
                },
                .conv2d_bwd_input => |*p| {
                    const d = Conv2dDims(T).fromBwdInput(p.*);
                    p.scratch = try alloc.alloc(T, d.K * d.N);
                },
                .conv2d_bwd_kernel => |*p| {
                    const d = Conv2dDims(T).fromBwdKernel(p.*);
                    // Batched: [K, N*batch] — one im2col for all samples
                    p.scratch = try alloc.alloc(T, d.K * d.N * d.batch);
                },
                else => {},
            }
        }
    };
}

pub fn cloneFusionPlan(comptime T: type, alloc: std.mem.Allocator, plan: FusionPlan(T)) !FusionPlan(T) {
    return switch (plan.payload) {
        .elementwise_chain => |elementwise| .{
            .output_idx = plan.output_idx,
            .payload = .{ .elementwise_chain = .{
                .input = elementwise.input,
                .nodes = try alloc.dupe(*Tensor(T), elementwise.nodes),
                .other_operand_roles = if (elementwise.other_operand_roles.len == 0)
                    &.{}
                else
                    try alloc.dupe(BinaryOperandRole, elementwise.other_operand_roles),
            } },
        },
        else => plan,
    };
}

pub fn deinitFusionPlan(comptime T: type, alloc: std.mem.Allocator, plan: FusionPlan(T)) void {
    switch (plan.payload) {
        .elementwise_chain => |elementwise| {
            alloc.free(elementwise.nodes);
            if (elementwise.other_operand_roles.len != 0) alloc.free(elementwise.other_operand_roles);
        },
        else => {},
    }
}

/// Validate that a softmax-family plan's intermediate nodes have consistent shapes.
/// The execution kernel iterates over input dimensions and indexes into all
/// full-sized intermediates — if any have a different shape, we'd go OOB.
pub fn validateSoftmaxPlan(comptime T: type, plan: SoftmaxPlan(T)) bool {
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_max, plan.neg_rep_max, plan.shifted, plan.exp_node, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}

pub fn validateLogSoftmaxPlan(comptime T: type, plan: LogSoftmaxPlan(T)) bool {
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_max, plan.neg_rep_max, plan.shifted, plan.exp_node, plan.rep_log, plan.neg_rep_log, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}

pub fn validateLayerNormPlan(comptime T: type, plan: LayerNormPlan(T)) bool {
    if (plan.mean_node.source1() == null or plan.var_node.source1() == null) return false;
    const input = plan.input;
    const full_nodes = [_]*Tensor(T){ plan.rep_mean, plan.neg_rep_mean, plan.centered, plan.sqr_node, plan.rep_std_inv, plan.output };
    for (full_nodes) |node| {
        if (!node.isSameShape(input)) return false;
    }
    return true;
}


fn fusedOpForNode(comptime T: type, plan: ElementwiseFusionPlan(T), idx: usize) ?FusedOp {
    const node = plan.nodes[idx];
    return switch (node.op) {
        .neg => .neg,
        .abs => .abs,
        .sgn => .sgn,
        .step => .step,
        .sqrt => .sqrt,
        .recip => .recip,
        .exp => .exp,
        .log => .log,
        .gelu => .gelu,
        .add => if (plan.otherOperandRole(idx) == .src0) .add_src0 else .add_src1,
        .mul => if (plan.otherOperandRole(idx) == .src0) .mul_src0 else .mul_src1,
        else => null,
    };
}

/// Apply a single op to a value. Comptime-dispatched — no runtime branching.
fn applyOp(comptime T: type, comptime op: FusedOp, val: T, node: anytype, i: usize) T {
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
        .add_src1 => blk: {
            const other = node.src1.?;
            break :blk val + if (other.data.len <= 1) other.data[0] else other.data[i];
        },
        .add_src0 => blk: {
            const other = node.src0.?;
            break :blk val + if (other.data.len <= 1) other.data[0] else other.data[i];
        },
        .mul_src1 => blk: {
            // sqr: src0 == src1, both operands are the chain value
            if (node.src0.? == node.src1.?) break :blk val * val;
            const other = node.src1.?;
            break :blk val * if (other.data.len <= 1) other.data[0] else other.data[i];
        },
        .mul_src0 => blk: {
            if (node.src0.? == node.src1.?) break :blk val * val;
            const other = node.src0.?;
            break :blk val * if (other.data.len <= 1) other.data[0] else other.data[i];
        },
    };
}

/// A comptime-specialized fused kernel for a known op sequence.
/// The `inline for` is fully unrolled — the inner loop has zero branches.
pub fn FusedKernel(comptime T: type, comptime ops: []const FusedOp) type {
    return struct {
        pub fn execute(nodes: []const *Tensor(T)) void {
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
pub fn executeFusedGeneric(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    const nodes = plan.nodes;
    const n_elems = nodes[nodes.len - 1].nElems();
    const input_data = plan.input.data;

    for (0..n_elems) |i| {
        var val: T = input_data[i];
        for (nodes, 0..) |node, node_idx| {
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
                    const other = plan.otherOperand(node_idx).?;
                    break :blk val + if (other.data.len <= 1) other.data[0] else other.data[i];
                },
                .mul => blk: {
                    if (node.src0.? == node.src1.?) break :blk val * val;
                    const other = plan.otherOperand(node_idx).?;
                    break :blk val * if (other.data.len <= 1) other.data[0] else other.data[i];
                },
                else => unreachable,
            };
            node.data[i] = val;
        }
    }
}

/// Dispatch a fused chain to a comptime-specialized kernel.
///
/// Length 2-3: enumerates all op combinations at comptime (169 / 2197 kernels)
///             producing zero-branch inner loops.
/// Length 4+: single-pass generic interpreter (one runtime switch per op per element).
///            Memory-bound for large tensors, so branch overhead is negligible.
pub fn executeFusedChain(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    if (!isSafeElementwiseChain(T, plan)) {
        for (plan.nodes) |node| node.compute();
        return;
    }
    switch (plan.nodes.len) {
        2 => executeFused2(T, plan),
        3 => executeFused3(T, plan),
        else => executeFusedGeneric(T, plan),
    }
}

fn executeFused2(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    const op0 = fusedOpForNode(T, plan, 0) orelse return executeFusedGeneric(T, plan);
    const op1 = fusedOpForNode(T, plan, 1) orelse return executeFusedGeneric(T, plan);
    inline for (fusible_ops) |c0| {
        inline for (fusible_ops) |c1| {
            if (op0 == c0 and op1 == c1) {
                FusedKernel(T, &.{ c0, c1 }).execute(plan.nodes);
                return;
            }
        }
    }
    executeFusedGeneric(T, plan);
}

fn executeFused3(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    @setEvalBranchQuota(20000);
    const op0 = fusedOpForNode(T, plan, 0) orelse return executeFusedGeneric(T, plan);
    const op1 = fusedOpForNode(T, plan, 1) orelse return executeFusedGeneric(T, plan);
    const op2 = fusedOpForNode(T, plan, 2) orelse return executeFusedGeneric(T, plan);
    inline for (fusible_ops) |c0| {
        inline for (fusible_ops) |c1| {
            inline for (fusible_ops) |c2| {
                if (op0 == c0 and op1 == c1 and op2 == c2) {
                    FusedKernel(T, &.{ c0, c1, c2 }).execute(plan.nodes);
                    return;
                }
            }
        }
    }
    executeFusedGeneric(T, plan);
}

pub fn isSafeElementwiseChain(comptime T: type, plan: ElementwiseFusionPlan(T)) bool {
    // Fused kernel indexes data[i] linearly — requires all tensors to be contiguous.
    if (!plan.input.isContiguous()) return false;
    var prev = plan.input;
    const n_elems = plan.nodes[plan.nodes.len - 1].nElems();
    for (plan.nodes, 0..) |node, node_idx| {
        if (!node.isSameShape(prev)) return false;
        if (!node.isContiguous()) return false;
        const other = plan.otherOperand(node_idx);
        if (other) |src| {
            // Must be scalar (1 elem) or contiguous with matching element count
            if (src.data.len > 1 and (!src.isContiguous() or src.data.len < n_elems)) return false;
        }
        prev = node;
    }
    return true;
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

// ---------------------------------------------------------------------------
// MaxPool2d execution
// ---------------------------------------------------------------------------

fn executeMaxPool2dBwd(comptime T: type, plan: MaxPool2dBwdPlan(T)) void {
    const input = plan.input;
    const out_grad = plan.output_grad;
    const dst = plan.output;
    const in_w = input.ne[0];
    const out_w = out_grad.ne[0];
    const out_h = out_grad.ne[1];
    const channels = out_grad.ne[2];
    const batch = out_grad.ne[3];
    const in_stride_h = in_w;
    const in_stride_c = in_w * input.ne[1];
    const in_stride_n = in_stride_c * channels;
    const out_stride_h = out_w;
    const out_stride_c = out_w * out_h;
    const out_stride_n = out_stride_c * channels;
    @memset(dst.data, 0);
    for (0..batch) |n| {
        for (0..channels) |ch| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    const ix = ox * 2;
                    const iy = oy * 2;
                    const base = ix + iy * in_stride_h + ch * in_stride_c + n * in_stride_n;
                    // Find argmax in 2x2 window
                    var best = base;
                    var val = input.data[base];
                    const offsets = [_]usize{ 1, in_stride_h, in_stride_h + 1 };
                    for (offsets) |off| {
                        if (input.data[base + off] > val) {
                            val = input.data[base + off];
                            best = base + off;
                        }
                    }
                    // Scatter gradient to argmax position
                    dst.data[best] += out_grad.data[ox + oy * out_stride_h + ch * out_stride_c + n * out_stride_n];
                }
            }
        }
    }
}

fn executeMaxPool2d(comptime T: type, plan: MaxPool2dPlan(T)) void {
    const input = plan.input;
    const dst = plan.output;
    const in_w = input.ne[0];
    const out_w = dst.ne[0];
    const out_h = dst.ne[1];
    const channels = dst.ne[2];
    const batch = dst.ne[3];
    const in_stride_h = in_w;
    const in_stride_c = in_w * input.ne[1];
    const in_stride_n = in_stride_c * channels;
    const out_stride_h = out_w;
    const out_stride_c = out_w * out_h;
    const out_stride_n = out_stride_c * channels;
    for (0..batch) |n| {
        for (0..channels) |ch| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    const ix = ox * 2;
                    const iy = oy * 2;
                    const base = ix + iy * in_stride_h + ch * in_stride_c + n * in_stride_n;
                    var val = input.data[base];
                    val = @max(val, input.data[base + 1]);
                    val = @max(val, input.data[base + in_stride_h]);
                    val = @max(val, input.data[base + in_stride_h + 1]);
                    dst.data[ox + oy * out_stride_h + ch * out_stride_c + n * out_stride_n] = val;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conv2d shared helpers: im2col, col2im, geometry
// ---------------------------------------------------------------------------

/// Shared dimension info for conv2d forward and backward.
fn Conv2dDims(comptime T: type) type {
    const max_dims = @import("../tensor.zig").max_dims;
    return struct {
        kw: usize,
        kh: usize,
        c_in: usize,
        c_out: usize,
        out_w: usize,
        out_h: usize,
        in_w: usize,
        batch: usize,
        K: usize, // kw * kh * c_in (patch size / inner matmul dim)
        N: usize, // out_w * out_h  (spatial positions)

        input_data: []const T,
        input_strides: [max_dims]usize,
        kernel_data: []const T,

        fn fromForward(plan: Conv2dPlan(T)) @This() {
            return init(plan.input.data, plan.input.strides, plan.kernel, plan.output);
        }

        fn fromBwdInput(plan: Conv2dBwdInputPlan(T)) @This() {
            return init(plan.output.data, plan.output.strides, plan.kernel, plan.output_grad);
        }

        fn fromBwdKernel(plan: Conv2dBwdKernelPlan(T)) @This() {
            return init(plan.input.data, plan.input.strides, plan.output, plan.output_grad);
        }

        fn init(
            input_data: []const T,
            input_strides: [max_dims]usize,
            kernel: *const Tensor(T),
            out_ref: *const Tensor(T),
        ) @This() {
            const kw = kernel.ne[0];
            const kh = kernel.ne[1];
            const c_in = kernel.ne[2];
            return .{
                .kw = kw,
                .kh = kh,
                .c_in = c_in,
                .c_out = out_ref.ne[2],
                .out_w = out_ref.ne[0],
                .out_h = out_ref.ne[1],
                .in_w = input_strides[1], // for contiguous layout, strides[1] == in_w
                .batch = out_ref.ne[3],
                .K = kw * kh * c_in,
                .N = out_ref.ne[0] * out_ref.ne[1],
                .input_data = input_data,
                .input_strides = input_strides,
                .kernel_data = kernel.data,
            };
        }
    };
}

/// Extract sliding-window patches into a column matrix.
///
/// Writes to col_buf with row stride `col_stride` (number of columns per row).
/// For non-batched use: col_stride = N, col_offset = 0.
/// For batched use:     col_stride = N*batch, col_offset = n*N.
///
/// col_buf[k * col_stride + col_offset + col] = input[ox+kx, oy+ky, ic, n]
///   k = kx + ky*kw + ic*kw*kh,  col = ox + oy*out_w
fn im2col(comptime T: type, d: Conv2dDims(T), col_buf: []T, n: usize, col_stride: usize, col_offset: usize) void {
    if (d.input_strides[0] == 1) {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    const row = (kx + ky * d.kw + ic * d.kw * d.kh) * col_stride + col_offset;
                    for (0..d.out_h) |oy| {
                        const src = kx + (oy + ky) * d.in_w + ic * d.input_strides[2] + n * d.input_strides[3];
                        @memcpy(col_buf[row + oy * d.out_w ..][0..d.out_w], d.input_data[src..][0..d.out_w]);
                    }
                }
            }
        }
    } else {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    const row = (kx + ky * d.kw + ic * d.kw * d.kh) * col_stride + col_offset;
                    for (0..d.out_h) |oy| {
                        for (0..d.out_w) |ox| {
                            col_buf[row + oy * d.out_w + ox] = d.input_data[
                                (ox + kx) * d.input_strides[0] +
                                    (oy + ky) * d.input_strides[1] +
                                    ic * d.input_strides[2] +
                                    n * d.input_strides[3]
                            ];
                        }
                    }
                }
            }
        }
    }
}

/// Scatter-add columns back to image layout (inverse of im2col).
/// Accumulates col_buf[K, N] into dst_data at the positions that im2col would read from.
fn col2im(comptime T: type, d: Conv2dDims(T), dst_data: []T, dst_strides: [@import("../tensor.zig").max_dims]usize, col_buf: []const T, n: usize) void {
    if (dst_strides[0] == 1) {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    const row = (kx + ky * d.kw + ic * d.kw * d.kh) * d.N;
                    for (0..d.out_h) |oy| {
                        const dst_base = kx + (oy + ky) * d.in_w + ic * dst_strides[2] + n * dst_strides[3];
                        const src_base = row + oy * d.out_w;
                        for (0..d.out_w) |ox| {
                            dst_data[dst_base + ox] += col_buf[src_base + ox];
                        }
                    }
                }
            }
        }
    } else {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    const row = (kx + ky * d.kw + ic * d.kw * d.kh) * d.N;
                    for (0..d.out_h) |oy| {
                        for (0..d.out_w) |ox| {
                            dst_data[
                                (ox + kx) * dst_strides[0] + (oy + ky) * dst_strides[1] +
                                    ic * dst_strides[2] + n * dst_strides[3]
                            ] += col_buf[row + oy * d.out_w + ox];
                        }
                    }
                }
            }
        }
    }
}

/// Allocate scratch from page_allocator, returning null on failure.
fn allocScratch(comptime T: type, len: usize) ?[]T {
    return std.heap.page_allocator.alloc(T, len) catch null;
}

fn freeScratch(comptime T: type, buf: []T) void {
    std.heap.page_allocator.free(buf);
}

// ---------------------------------------------------------------------------
// Conv2d execution: forward, backward-input, backward-kernel
// ---------------------------------------------------------------------------

fn executeConv2dPlan(comptime T: type, plan: Conv2dPlan(T)) void {
    const d = Conv2dDims(T).fromForward(plan);
    const NB = d.N * d.batch;
    const total_scratch = (d.K + d.c_out) * NB;
    const scratch = plan.scratch orelse allocScratch(T, total_scratch) orelse {
        conv2dNaive(T, d, plan.output.data);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, scratch);

    const col_buf = scratch[0 .. d.K * NB];
    const mm_temp = scratch[d.K * NB ..][0 .. d.c_out * NB];

    // 1. Batched im2col: all samples into [K, N*batch].
    for (0..d.batch) |n| {
        im2col(T, d, col_buf, n, NB, n * d.N);
    }

    // 2. Single matmul: kernel[c_out, K] @ col_buf[K, NB] → mm_temp[c_out, NB]
    const mm = forward.selectMatMulKernel(T);
    mm(mm_temp, d.kernel_data, col_buf.ptr[0..col_buf.len], d.c_out, NB, d.K, d.K, 1, NB, 1, 0, 0, 0, NB);

    // 3. Rearrange [c_out, NB] → [batch, c_out, N] with fused bias + activation.
    const output = plan.output.data;
    if (plan.bias) |bias| {
        if (plan.activation != null) {
            for (0..d.batch) |n| {
                for (0..d.c_out) |oc| {
                    const b = bias.data[oc];
                    const src = mm_temp[oc * NB + n * d.N ..][0..d.N];
                    const dst = output[n * d.N * d.c_out + oc * d.N ..][0..d.N];
                    for (dst, src) |*dv, sv| dv.* = @max(sv + b, 0);
                }
            }
        } else {
            for (0..d.batch) |n| {
                for (0..d.c_out) |oc| {
                    const b = bias.data[oc];
                    const src = mm_temp[oc * NB + n * d.N ..][0..d.N];
                    const dst = output[n * d.N * d.c_out + oc * d.N ..][0..d.N];
                    for (dst, src) |*dv, sv| dv.* = sv + b;
                }
            }
        }
    } else if (plan.activation != null) {
        for (0..d.batch) |n| {
            for (0..d.c_out) |oc| {
                const src = mm_temp[oc * NB + n * d.N ..][0..d.N];
                const dst = output[n * d.N * d.c_out + oc * d.N ..][0..d.N];
                for (dst, src) |*dv, sv| dv.* = @max(sv, 0);
            }
        }
    } else {
        for (0..d.batch) |n| {
            for (0..d.c_out) |oc| {
                const src = mm_temp[oc * NB + n * d.N ..][0..d.N];
                const dst = output[n * d.N * d.c_out + oc * d.N ..][0..d.N];
                @memcpy(dst, src);
            }
        }
    }
}

fn executeConv2dBwdInputPlan(comptime T: type, plan: Conv2dBwdInputPlan(T)) void {
    const d = Conv2dDims(T).fromBwdInput(plan);
    const col_buf = plan.scratch orelse allocScratch(T, d.K * d.N) orelse {
        conv2dBwdInputNaive(T, d, plan);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, col_buf);

    const mm = forward.selectMatMulKernel(T);
    @memset(plan.output.data, 0);

    for (0..d.batch) |n| {
        // kernel^T[K, c_out] @ out_grad_n[c_out, N] → col_buf[K, N]
        mm(col_buf, d.kernel_data, plan.output_grad.data, d.K, d.N, d.c_out, 1, d.K, d.N, 1, 0, n * d.N * d.c_out, 0, d.N);
        col2im(T, d, plan.output.data, plan.output.strides, col_buf, n);
    }
}

fn executeConv2dBwdKernelPlan(comptime T: type, plan: Conv2dBwdKernelPlan(T)) void {
    const d = Conv2dDims(T).fromBwdKernel(plan);
    const NB = d.N * d.batch;

    const col_buf = plan.scratch orelse allocScratch(T, d.K * NB) orelse {
        conv2dBwdKernelNaive(T, d, plan);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, col_buf);

    // Build batched im2col: [K, N*batch] — all samples concatenated.
    for (0..d.batch) |n| {
        im2col(T, d, col_buf, n, NB, n * d.N);
    }

    // Per-batch matmul: out_grad_n[c_out, N] @ col_n^T[N, K] → accumulate into output[c_out, K].
    // Layout is [out_w, out_h, c_out, batch] — channels are interleaved with batches,
    // so we can't do a single batched matmul.
    const mm = forward.selectMatMulKernel(T);

    // First batch: write directly to output.
    // col_buf layout is [K, NB] — rows have stride NB, not N.
    mm(plan.output.data, plan.output_grad.data, col_buf, d.c_out, d.K, d.N, d.N, 1, 1, NB, 0, 0, 0, d.K);

    // Remaining batches: matmul into temp, then accumulate.
    if (d.batch > 1) {
        const temp = allocScratch(T, d.c_out * d.K) orelse {
            conv2dBwdKernelNaive(T, d, plan);
            return;
        };
        defer freeScratch(T, temp);
        for (1..d.batch) |n| {
            mm(temp, plan.output_grad.data, col_buf, d.c_out, d.K, d.N, d.N, 1, 1, NB, n * d.N * d.c_out, n * d.N, 0, d.K);
            for (0..d.c_out * d.K) |j| plan.output.data[j] += temp[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Naive fallbacks (allocation failure only)
// ---------------------------------------------------------------------------

fn conv2dNaive(comptime T: type, d: Conv2dDims(T), output_data: []T) void {
    for (0..d.batch) |n| {
        for (0..d.c_out) |oc| {
            for (0..d.out_h) |oy| {
                for (0..d.out_w) |ox| {
                    var acc: T = 0;
                    for (0..d.c_in) |ic| {
                        for (0..d.kh) |ky| {
                            for (0..d.kw) |kx| {
                                acc += d.input_data[(ox + kx) * d.input_strides[0] + (oy + ky) * d.input_strides[1] + ic * d.input_strides[2] + n * d.input_strides[3]] *
                                    d.kernel_data[kx + ky * d.kw + ic * d.kw * d.kh + oc * d.K];
                            }
                        }
                    }
                    output_data[ox + oy * d.out_w + oc * d.N + n * d.N * d.c_out] = acc;
                }
            }
        }
    }
}

fn conv2dBwdInputNaive(comptime T: type, d: Conv2dDims(T), plan: Conv2dBwdInputPlan(T)) void {
    const out_grad = plan.output_grad;
    const output = plan.output;
    @memset(output.data, 0);
    for (0..d.batch) |n| {
        for (0..d.c_out) |oc| {
            for (0..d.out_h) |oy| {
                for (0..d.out_w) |ox| {
                    const g = out_grad.data[offset4(out_grad.strides, ox, oy, oc, n)];
                    for (0..d.c_in) |ic| {
                        for (0..d.kh) |ky| {
                            for (0..d.kw) |kx| {
                                output.data[offset4(output.strides, ox + kx, oy + ky, ic, n)] +=
                                    g * d.kernel_data[kx + ky * d.kw + ic * d.kw * d.kh + oc * d.K];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn conv2dBwdKernelNaive(comptime T: type, d: Conv2dDims(T), plan: Conv2dBwdKernelPlan(T)) void {
    const out_grad = plan.output_grad;
    const output = plan.output;
    @memset(output.data, 0);
    for (0..d.c_out) |oc| {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    var acc: T = 0;
                    for (0..d.batch) |n| {
                        for (0..d.out_h) |oy| {
                            for (0..d.out_w) |ox| {
                                acc += d.input_data[(ox + kx) * d.input_strides[0] + (oy + ky) * d.input_strides[1] + ic * d.input_strides[2] + n * d.input_strides[3]] *
                                    out_grad.data[offset4(out_grad.strides, ox, oy, oc, n)];
                            }
                        }
                    }
                    output.data[offset4(output.strides, kx, ky, ic, oc)] = acc;
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

/// Walk backward through a tensor's source chain to find the earliest
/// dependency that exists in the node list, bounded by the output index.
/// Used to determine the start of a fused region that replaces an entire
/// backward chain (e.g., conv2d backward: reshape → broadcast → mul → scatter).

pub fn indexOfNodeMaybe(comptime T: type, nodes: []const *Tensor(T), needle: *Tensor(T)) ?usize {
    for (nodes, 0..) |node, i| {
        if (node == needle) return i;
    }
    return null;
}

fn indexOfNode(comptime T: type, nodes: []const *Tensor(T), needle: *Tensor(T)) usize {
    return indexOfNodeMaybe(T, nodes, needle) orelse unreachable;
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
        .max_pool2d => |pool_plan| executeMaxPool2d(T, pool_plan),
        .max_pool2d_bwd => |pool_plan| executeMaxPool2dBwd(T, pool_plan),
        .softmax => |softmax_plan| executeSoftmaxPlan(T, softmax_plan),
        .log_softmax => |log_softmax_plan| executeLogSoftmaxPlan(T, log_softmax_plan),
        .cross_entropy => |cross_entropy_plan| executeCrossEntropyPlan(T, cross_entropy_plan),
        .layer_norm => |layer_norm_plan| executeLayerNormPlan(T, layer_norm_plan),
    }
}


test "fused - specialized swapped commutative 2-op chain matches generic" {
    const T = f32;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const input_fused = try Tensor(T).init(a, &.{3});
    input_fused.setData(&.{ 1, 2, 3 });
    const scalar_fused = try Tensor(T).initScalar(a, 2);
    const exp_fused = input_fused.exp();
    const add_fused = scalar_fused.repeatLike(exp_fused).add(exp_fused);

    const input_generic = try Tensor(T).init(a, &.{3});
    input_generic.setData(&.{ 1, 2, 3 });
    const scalar_generic = try Tensor(T).initScalar(a, 2);
    const exp_generic = input_generic.exp();
    const add_generic = scalar_generic.repeatLike(exp_generic).add(exp_generic);

    const fused_plan = ElementwiseFusionPlan(T){
        .input = input_fused,
        .nodes = &.{ exp_fused, add_fused },
        .other_operand_roles = &.{ .src1, .src0 },
    };
    const generic_plan = ElementwiseFusionPlan(T){
        .input = input_generic,
        .nodes = &.{ exp_generic, add_generic },
        .other_operand_roles = &.{ .src1, .src0 },
    };

    executeFusedChain(T, fused_plan);
    executeFusedGeneric(T, generic_plan);

    try std.testing.expectEqualSlices(T, add_generic.data, add_fused.data);
}

test "fused - specialized swapped commutative 3-op chain matches generic" {
    const T = f32;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const input_fused = try Tensor(T).init(a, &.{3});
    input_fused.setData(&.{ 1, 2, 3 });
    const scalar_fused = try Tensor(T).initScalar(a, 2);
    const exp_fused = input_fused.exp();
    const add_fused = scalar_fused.repeatLike(exp_fused).add(exp_fused);
    const log_fused = add_fused.log();

    const input_generic = try Tensor(T).init(a, &.{3});
    input_generic.setData(&.{ 1, 2, 3 });
    const scalar_generic = try Tensor(T).initScalar(a, 2);
    const exp_generic = input_generic.exp();
    const add_generic = scalar_generic.repeatLike(exp_generic).add(exp_generic);
    const log_generic = add_generic.log();

    const fused_plan = ElementwiseFusionPlan(T){
        .input = input_fused,
        .nodes = &.{ exp_fused, add_fused, log_fused },
        .other_operand_roles = &.{ .src1, .src0, .src1 },
    };
    const generic_plan = ElementwiseFusionPlan(T){
        .input = input_generic,
        .nodes = &.{ exp_generic, add_generic, log_generic },
        .other_operand_roles = &.{ .src1, .src0, .src1 },
    };

    executeFusedChain(T, fused_plan);
    executeFusedGeneric(T, generic_plan);

    try std.testing.expectEqualSlices(T, log_generic.data, log_fused.data);
}
