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

const conv_workspace_target_bytes: usize = 16 * 1024 * 1024;
const conv_workspace_min_tiled_k: usize = 64;

pub const ConvPhaseProfile = struct {
    fwd_im2col_ns: u64 = 0,
    fwd_gemm_ns: u64 = 0,
    fwd_epilogue_ns: u64 = 0,
    bwd_input_rearrange_ns: u64 = 0,
    bwd_input_gemm_ns: u64 = 0,
    bwd_input_col2im_ns: u64 = 0,
    bwd_kernel_im2col_ns: u64 = 0,
    bwd_kernel_rearrange_ns: u64 = 0,
    bwd_kernel_gemm_ns: u64 = 0,
};

fn addConvPhase(dst: ?*ConvPhaseProfile, comptime field: []const u8, ns: u64) void {
    if (dst) |profile| {
        @field(profile, field) += ns;
    }
}

const GELU_COEF_A: comptime_float = 0.044715;
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

const FusedOp = enum {
    neg,
    abs,
    sgn,
    step,
    relu,
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
    .relu,
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
                    const ws = ConvWorkspacePlan(T).init(.forward, d);
                    p.scratch = try alloc.alloc(T, ws.total_elems);
                },
                .conv2d_bwd_input => |*p| {
                    const d = Conv2dDims(T).fromBwdInput(p.*);
                    const ws = ConvWorkspacePlan(T).init(.bwd_input, d);
                    p.scratch = try alloc.alloc(T, ws.total_elems);
                },
                .conv2d_bwd_kernel => |*p| {
                    const d = Conv2dDims(T).fromBwdKernel(p.*);
                    const ws = ConvWorkspacePlan(T).init(.bwd_kernel, d);
                    p.scratch = try alloc.alloc(T, ws.total_elems);
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
        .relu => .relu,
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

// ---------------------------------------------------------------------------
// SIMD + stride helpers
// ---------------------------------------------------------------------------

/// SIMD vector width (in lanes) for type T. Targets 256-bit.
fn vecSize(comptime T: type) comptime_int {
    const lanes = 32 / @sizeOf(T);
    return if (lanes >= 4) lanes else 4;
}

const max_dims = @import("../tensor.zig").max_dims;

/// Advance an N-dimensional coordinate by 1, returning false when done.
inline fn nextCoord(coords: []usize, shape: []const usize) bool {
    var d: usize = 0;
    while (d < coords.len) : (d += 1) {
        coords[d] += 1;
        if (coords[d] < shape[d]) return true;
        coords[d] = 0;
    }
    return false;
}

/// Compute strided offset for a set of coordinates.
inline fn stridedOffset(strides: []const usize, coords: []const usize, base: usize) usize {
    var off: usize = base;
    for (strides, coords) |s, c| off += s * c;
    return off;
}

/// Load one scalar element from a chain "other" operand.
inline fn loadOther(comptime T: type, other: anytype, i: usize) T {
    return if (other.nElems() <= 1) other.data[other.storage_offset] else other.data[i + other.storage_offset];
}

/// Load a SIMD vector from a chain "other" operand: splat if scalar, load if dense.
inline fn loadOtherVec(comptime T: type, comptime V: comptime_int, other: anytype, i: usize) @Vector(V, T) {
    return if (other.nElems() <= 1)
        @splat(other.data[other.storage_offset])
    else
        other.data[i + other.storage_offset ..][0..V].*;
}

/// Apply a single op to a scalar value. Comptime-dispatched — no runtime branching.
fn applyOp(comptime T: type, comptime op: FusedOp, val: T, node: anytype, i: usize) T {
    return switch (op) {
        .neg => -val,
        .abs => @abs(val),
        .sgn => if (val > 0) 1 else if (val < 0) @as(T, -1) else 0,
        .step => if (val > 0) @as(T, 1) else 0,
        .relu => if (val > 0) val else 0,
        .sqrt => @sqrt(val),
        .recip => 1.0 / val,
        .exp => @exp(val),
        .log => @log(val),
        .gelu => 0.5 * val * (1.0 + std.math.tanh(
            @as(T, SQRT_2_OVER_PI) * val * (1.0 + @as(T, GELU_COEF_A) * val * val),
        )),
        .add_src1 => val + loadOther(T, node.src1.?, i),
        .add_src0 => val + loadOther(T, node.src0.?, i),
        .mul_src1 => if (node.src0.? == node.src1.?) val * val else val * loadOther(T, node.src1.?, i),
        .mul_src0 => if (node.src0.? == node.src1.?) val * val else val * loadOther(T, node.src0.?, i),
    };
}

/// Apply a single op to a SIMD vector. Comptime-dispatched — no runtime branching.
fn applyOpVec(comptime T: type, comptime V: comptime_int, comptime op: FusedOp, val: @Vector(V, T), node: anytype, i: usize) @Vector(V, T) {
    const VecT = @Vector(V, T);
    return switch (op) {
        .neg => -val,
        .abs => @abs(val),
        .sgn => blk: {
            const zero: VecT = @splat(0);
            break :blk @select(T, val > zero, @as(VecT, @splat(@as(T, 1))), @select(T, val < zero, @as(VecT, @splat(@as(T, -1))), zero));
        },
        .step => @select(T, val > @as(VecT, @splat(@as(T, 0))), @as(VecT, @splat(@as(T, 1))), @as(VecT, @splat(@as(T, 0)))),
        .relu => @max(val, @as(VecT, @splat(@as(T, 0)))),
        .sqrt => @sqrt(val),
        .recip => @as(VecT, @splat(@as(T, 1))) / val,
        .exp => @exp(val),
        .log => @log(val),
        .gelu => blk: {
            const one: VecT = @splat(@as(T, 1));
            const two: VecT = @splat(@as(T, 2));
            const inner = @as(VecT, @splat(@as(T, SQRT_2_OVER_PI))) * val * (one + @as(VecT, @splat(@as(T, GELU_COEF_A))) * val * val);
            const e2 = @exp(two * inner);
            break :blk @as(VecT, @splat(@as(T, 0.5))) * val * (one + (e2 - one) / (e2 + one));
        },
        .add_src1 => val + loadOtherVec(T, V, node.src1.?, i),
        .add_src0 => val + loadOtherVec(T, V, node.src0.?, i),
        .mul_src1 => if (node.src0.? == node.src1.?) val * val else val * loadOtherVec(T, V, node.src1.?, i),
        .mul_src0 => if (node.src0.? == node.src1.?) val * val else val * loadOtherVec(T, V, node.src0.?, i),
    };
}

/// A comptime-specialized fused kernel for a known op sequence.
/// The `inline for` is fully unrolled — the inner loop has zero branches.
pub fn FusedKernel(comptime T: type, comptime ops: []const FusedOp) type {
    return struct {
        pub fn execute(nodes: []const *Tensor(T)) void {
            executeRange(nodes, 0, nodes[nodes.len - 1].nElems());
        }

        /// SIMD-vectorized execution for dense (contiguous) inputs only.
        /// Non-dense inputs must be handled by the generic interpreter.
        pub fn executeRange(nodes: []const *Tensor(T), start: usize, end: usize) void {
            const V = comptime vecSize(T);
            const VecT = @Vector(V, T);
            const input = nodes[0].source0().?;
            const input_data = input.data;
            const input_off = input.storage_offset;

            var i: usize = start;
            while (i + V <= end) : (i += V) {
                var val: VecT = input_data[i + input_off ..][0..V].*;
                inline for (ops, 0..) |op, k| {
                    val = applyOpVec(T, V, op, val, nodes[k], i);
                    nodes[k].data[i + nodes[k].storage_offset ..][0..V].* = val;
                }
            }
            while (i < end) : (i += 1) {
                var val: T = input_data[i + input_off];
                inline for (ops, 0..) |op, k| {
                    val = applyOp(T, op, val, nodes[k], i);
                    nodes[k].data[i + nodes[k].storage_offset] = val;
                }
            }
        }
    };
}

/// Runtime interpreter for a range of elements. Uses SIMD for the main loop
/// with a scalar tail. Used by both the generic fallback and parallel executor.
fn executeFusedGenericRange(comptime T: type, plan: ElementwiseFusionPlan(T), start: usize, end: usize) void {
    const V = comptime vecSize(T);
    const VecT = @Vector(V, T);
    const nodes = plan.nodes;
    const input = plan.input;
    const input_data = input.data;
    const input_dense = input.isDenseLayout();

    if (input_dense) {
        const input_off = input.storage_offset;
        var i: usize = start;
        while (i + V <= end) : (i += V) {
            var val: VecT = input_data[i + input_off ..][0..V].*;
            for (nodes, 0..) |node, node_idx| {
                const fop = fusedOpForNode(T, plan, node_idx).?;
                inline for (fusible_ops) |c| if (fop == c) {
                    val = applyOpVec(T, V, c, val, node, i);
                };
                node.data[i + node.storage_offset ..][0..V].* = val;
            }
        }
        while (i < end) : (i += 1) {
            var val: T = input_data[i + input_off];
            for (nodes, 0..) |node, node_idx| {
                const fop = fusedOpForNode(T, plan, node_idx).?;
                inline for (fusible_ops) |c| if (fop == c) {
                    val = applyOp(T, c, val, node, i);
                };
                node.data[i + node.storage_offset] = val;
            }
        }
    } else {
        // Non-dense input: coordinate-based gather, SIMD for chain ops.
        const n_dims = input.n_dims;
        var coords: [max_dims]usize = [_]usize{0} ** max_dims;
        if (start > 0) {
            var remaining = start;
            var d: usize = 0;
            while (d < n_dims) : (d += 1) {
                coords[d] = remaining % input.ne[d];
                remaining /= input.ne[d];
            }
        }
        var i: usize = start;
        while (i + V <= end) : (i += V) {
            var val: VecT = undefined;
            inline for (0..V) |lane| {
                val[lane] = input_data[stridedOffset(input.strides[0..n_dims], coords[0..n_dims], input.storage_offset)];
                _ = nextCoord(coords[0..n_dims], input.ne[0..n_dims]);
            }
            for (nodes, 0..) |node, node_idx| {
                const fop = fusedOpForNode(T, plan, node_idx).?;
                inline for (fusible_ops) |c| if (fop == c) {
                    val = applyOpVec(T, V, c, val, node, i);
                };
                node.data[i + node.storage_offset ..][0..V].* = val;
            }
        }
        while (i < end) : (i += 1) {
            var val: T = input_data[stridedOffset(input.strides[0..n_dims], coords[0..n_dims], input.storage_offset)];
            _ = nextCoord(coords[0..n_dims], input.ne[0..n_dims]);
            for (nodes, 0..) |node, node_idx| {
                const fop = fusedOpForNode(T, plan, node_idx).?;
                inline for (fusible_ops) |c| if (fop == c) {
                    val = applyOp(T, c, val, node, i);
                };
                node.data[i + node.storage_offset] = val;
            }
        }
    }
}

/// Parallel elementwise chain execution. Splits the element range across threads.
pub fn executeFusedChainParallel(comptime T: type, plan: ElementwiseFusionPlan(T), pool: *std.Thread.Pool) void {
    if (!isSafeElementwiseChain(T, plan)) {
        for (plan.nodes) |node| node.compute();
        return;
    }

    const n_elems = plan.nodes[plan.nodes.len - 1].nElems();
    const min_chunk = 1024; // below this, not worth threading
    const n_workers = pool.threads.len + 1;

    if (n_elems < min_chunk * 2 or n_workers <= 1) {
        executeFusedChain(T, plan);
        return;
    }

    const chunk = @max(min_chunk, (n_elems + n_workers - 1) / n_workers);
    var wg = std.Thread.WaitGroup{};
    var start: usize = chunk;
    while (start < n_elems) {
        const s = start;
        const e = @min(start + chunk, n_elems);
        pool.spawnWg(&wg, struct {
            fn run(p: ElementwiseFusionPlan(T), cs: usize, ce: usize) void {
                executeFusedGenericRange(T, p, cs, ce);
            }
        }.run, .{ plan, s, e });
        start = e;
    }

    // Caller thread does the first chunk (comptime-specialized when possible)
    executeFusedGenericRange(T, plan, 0, @min(chunk, n_elems));
    wg.wait();
}

/// Runtime interpreter fallback for chains longer than the comptime dispatch limit.
/// Still one memory pass, but the inner loop has a runtime switch.
pub fn executeFusedGeneric(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    const n_elems = plan.nodes[plan.nodes.len - 1].nElems();
    executeFusedGenericRange(T, plan, 0, n_elems);
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
    // On WASM, the comptime-enumerated dispatch (13^N inline for) is
    // miscompiled at optimization levels above Debug. Use the generic
    // interpreter which is correct and still single-pass.
    if (comptime @import("builtin").target.cpu.arch == .wasm32 or
        @import("builtin").target.cpu.arch == .wasm64)
    {
        executeFusedGeneric(T, plan);
        return;
    }
    switch (plan.nodes.len) {
        2 => executeFused2(T, plan),
        3 => executeFused3(T, plan),
        else => executeFusedGeneric(T, plan),
    }
}

fn executeFused2(comptime T: type, plan: ElementwiseFusionPlan(T)) void {
    // Comptime-specialized kernels require dense input for linear SIMD access.
    if (!plan.input.isDenseLayout()) return executeFusedGeneric(T, plan);
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
    // Comptime-specialized kernels require dense input for linear SIMD access.
    if (!plan.input.isDenseLayout()) return executeFusedGeneric(T, plan);
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
    // Chain nodes and other operands are indexed linearly and must be dense.
    // The input may be non-dense (permuted strides); execution handles it.
    var prev = plan.input;
    const n_elems = plan.nodes[plan.nodes.len - 1].nElems();
    for (plan.nodes, 0..) |node, node_idx| {
        if (!node.isSameShape(prev)) return false;
        if (!node.isDenseLayout()) return false;
        const other = plan.otherOperand(node_idx);
        if (other) |src| {
            if (src.nElems() > 1 and (!src.isDenseLayout() or src.nElems() < n_elems)) return false;
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
                    const grad = out_grad.data[ox + oy * out_stride_h + ch * out_stride_c + n * out_stride_n];
                    const v00 = input.data[base];
                    const v10 = input.data[base + 1];
                    const v01 = input.data[base + in_stride_h];
                    const v11 = input.data[base + in_stride_h + 1];
                    const max_val = @max(@max(v00, v10), @max(v01, v11));
                    const tie_count_usize: usize =
                        @as(usize, @intFromBool(v00 == max_val)) +
                        @as(usize, @intFromBool(v10 == max_val)) +
                        @as(usize, @intFromBool(v01 == max_val)) +
                        @as(usize, @intFromBool(v11 == max_val));
                    const share = grad / @as(T, @floatFromInt(tie_count_usize));

                    // Symmetric subgradient for max: split upstream gradient
                    // evenly across all tied maxima in the pooling window.
                    if (v00 == max_val) dst.data[base] += share;
                    if (v10 == max_val) dst.data[base + 1] += share;
                    if (v01 == max_val) dst.data[base + in_stride_h] += share;
                    if (v11 == max_val) dst.data[base + in_stride_h + 1] += share;
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

const ConvWorkspaceKind = enum {
    forward,
    bwd_input,
    bwd_kernel,
};

fn ConvWorkspacePlan(comptime T: type) type {
    return struct {
        batch_tile: usize,
        col_stride: usize,
        col_elems: usize,
        aux_elems: usize,
        partial_elems: usize,
        total_elems: usize,

        fn chooseBatchTile(batch: usize, per_batch_elems: usize, k: usize) usize {
            if (k < conv_workspace_min_tiled_k) return batch;
            if (!(T == f32 and @import("zgml_options").use_blas)) return batch;
            const target_elems = conv_workspace_target_bytes / @sizeOf(T);
            var tile = batch;
            while (tile > 1 and per_batch_elems * tile > target_elems) {
                tile = (tile + 1) / 2;
            }
            return tile;
        }

        fn init(kind: ConvWorkspaceKind, d: Conv2dDims(T)) @This() {
            const partial_elems = if (kind == .bwd_kernel) d.c_out * d.K else 0;
            const per_batch_elems = (d.K + d.c_out) * d.N;
            const batch_tile = chooseBatchTile(d.batch, per_batch_elems, d.K);
            const col_stride = d.N * batch_tile;
            const col_elems = d.K * col_stride;
            const aux_elems = d.c_out * col_stride;
            return .{
                .batch_tile = batch_tile,
                .col_stride = col_stride,
                .col_elems = col_elems,
                .aux_elems = aux_elems,
                .partial_elems = if (batch_tile < d.batch) partial_elems else 0,
                .total_elems = col_elems + aux_elems + if (batch_tile < d.batch) partial_elems else 0,
            };
        }

        fn colBuf(self: @This(), scratch: []T) []T {
            return scratch[0..self.col_elems];
        }

        fn auxBuf(self: @This(), scratch: []T) []T {
            return scratch[self.col_elems..][0..self.aux_elems];
        }

        fn partialBuf(self: @This(), scratch: []T) []T {
            return scratch[self.col_elems + self.aux_elems ..][0..self.partial_elems];
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
                        forward.simdCopy(T, col_buf[row + oy * d.out_w ..][0..d.out_w], d.input_data[src..][0..d.out_w]);
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
/// Accumulates col_buf into dst_data at the positions that im2col would read from.
///
/// col_stride: number of columns per row in col_buf (N for non-batched, N*batch for batched).
/// col_offset: column offset within each row (0 for non-batched, n*N for batched).
fn col2im(comptime T: type, d: Conv2dDims(T), dst_data: []T, dst_strides: [@import("../tensor.zig").max_dims]usize, col_buf: []const T, n: usize, col_stride: usize, col_offset: usize) void {
    if (dst_strides[0] == 1) {
        for (0..d.c_in) |ic| {
            for (0..d.kh) |ky| {
                for (0..d.kw) |kx| {
                    const row = (kx + ky * d.kw + ic * d.kw * d.kh) * col_stride + col_offset;
                    for (0..d.out_h) |oy| {
                        const dst_base = kx + (oy + ky) * d.in_w + ic * dst_strides[2] + n * dst_strides[3];
                        const src_base = row + oy * d.out_w;
                        forward.simdAccumulate(T, dst_data[dst_base..][0..d.out_w], col_buf[src_base..][0..d.out_w]);
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

/// Allocate scratch memory, returning null on failure.
/// Uses page_allocator on native targets; falls back to wasm_allocator on WASM.
const scratch_allocator = switch (@import("builtin").cpu.arch) {
    .wasm32, .wasm64 => std.heap.wasm_allocator,
    else => std.heap.page_allocator,
};

fn allocScratch(comptime T: type, len: usize) ?[]T {
    return scratch_allocator.alloc(T, len) catch null;
}

fn freeScratch(comptime T: type, buf: []T) void {
    scratch_allocator.free(buf);
}

/// Select the best matmul for fused conv2d: BLAS when available (f32), else tiled.
fn selectConvMatMul(comptime T: type) forward.MatMulFnType(T) {
    if (T == f32) return &forward.blasSgemm;
    return forward.selectMatMulKernel(T);
}

// ---------------------------------------------------------------------------
// Conv2d execution: forward, backward-input, backward-kernel
// ---------------------------------------------------------------------------

fn executeConv2dPlan(comptime T: type, plan: Conv2dPlan(T), phase_profile: ?*ConvPhaseProfile) void {
    const d = Conv2dDims(T).fromForward(plan);
    const ws = ConvWorkspacePlan(T).init(.forward, d);
    const total_scratch = ws.total_elems;
    const scratch = plan.scratch orelse allocScratch(T, total_scratch) orelse {
        conv2dNaive(T, d, plan.output.data);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, scratch);

    const col_buf = ws.colBuf(scratch);
    const mm_temp = ws.auxBuf(scratch);
    var timer = std.time.Timer.start() catch @panic("timer");

    // 1. Batched im2col: all samples into [K, N*batch].
    timer.reset();
    const mm = selectConvMatMul(T);
    const output = plan.output.data;
    const has_bias = plan.bias != null;
    const has_relu = plan.activation != null;
    var batch_start: usize = 0;
    while (batch_start < d.batch) {
        const tile_batch = @min(ws.batch_tile, d.batch - batch_start);
        const tile_cols = d.N * tile_batch;

        timer.reset();
        for (0..tile_batch) |local_n| {
            const n = batch_start + local_n;
            im2col(T, d, col_buf, n, tile_cols, local_n * d.N);
        }
        addConvPhase(phase_profile, "fwd_im2col_ns", timer.read());

        timer.reset();
        mm(mm_temp[0 .. d.c_out * tile_cols], d.kernel_data, col_buf[0 .. d.K * tile_cols], d.c_out, tile_cols, d.K, d.K, 1, tile_cols, 1, 0, 0, 0, tile_cols);
        addConvPhase(phase_profile, "fwd_gemm_ns", timer.read());

        timer.reset();
        for (0..tile_batch) |local_n| {
            const n = batch_start + local_n;
            for (0..d.c_out) |oc| {
                const src = mm_temp[oc * tile_cols + local_n * d.N ..][0..d.N];
                const dst = output[n * d.N * d.c_out + oc * d.N ..][0..d.N];
                const b = if (has_bias) plan.bias.?.data[oc] else 0;
                if (has_bias or has_relu) {
                    rearrangeSimd(T, dst, src, b, has_relu);
                } else {
                    forward.simdCopy(T, dst, src);
                }
            }
        }
        addConvPhase(phase_profile, "fwd_epilogue_ns", timer.read());

        batch_start += tile_batch;
    }
}

/// SIMD copy with fused bias add + optional ReLU.
fn rearrangeSimd(comptime T: type, dst: []T, src: []const T, bias: T, relu: bool) void {
    const vl = comptime forward.simdVecSize(T);
    const V = @Vector(vl, T);
    const bv: V = @splat(bias);
    const len = dst.len;
    const zero: V = @splat(0);

    var i: usize = 0;
    while (i + vl <= len) : (i += vl) {
        var v = src[i..][0..vl].* + bv;
        if (relu) v = @max(v, zero);
        dst[i..][0..vl].* = v;
    }
    while (i < len) : (i += 1) {
        var v = src[i] + bias;
        if (relu and v < 0) v = 0;
        dst[i] = v;
    }
}

fn executeConv2dBwdInputPlan(comptime T: type, plan: Conv2dBwdInputPlan(T), phase_profile: ?*ConvPhaseProfile) void {
    const d = Conv2dDims(T).fromBwdInput(plan);
    const ws = ConvWorkspacePlan(T).init(.bwd_input, d);
    const total_scratch = ws.total_elems;
    const scratch = plan.scratch orelse allocScratch(T, total_scratch) orelse {
        conv2dBwdInputNaive(T, d, plan);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, scratch);

    const col_buf = ws.colBuf(scratch);
    const grad_buf = ws.auxBuf(scratch);
    var timer = std.time.Timer.start() catch @panic("timer");
    @memset(plan.output.data, 0);
    const mm = selectConvMatMul(T);
    var batch_start: usize = 0;

    while (batch_start < d.batch) {
        const tile_batch = @min(ws.batch_tile, d.batch - batch_start);
        const tile_cols = d.N * tile_batch;

        // 1. Rearrange output_grad from [N, c_out, batch] to [c_out, N*tile]
        timer.reset();
        for (0..d.c_out) |oc| {
            for (0..tile_batch) |local_n| {
                const n = batch_start + local_n;
                forward.simdCopy(T, grad_buf[oc * tile_cols + local_n * d.N ..][0..d.N], plan.output_grad.data[n * d.N * d.c_out + oc * d.N ..][0..d.N]);
            }
        }
        addConvPhase(phase_profile, "bwd_input_rearrange_ns", timer.read());

        // 2. GEMM: kernel^T[K, c_out] @ grad_buf[c_out, N*tile] → col_buf[K, N*tile]
        timer.reset();
        mm(col_buf[0 .. d.K * tile_cols], d.kernel_data, grad_buf[0 .. d.c_out * tile_cols], d.K, tile_cols, d.c_out, 1, d.K, tile_cols, 1, 0, 0, 0, tile_cols);
        addConvPhase(phase_profile, "bwd_input_gemm_ns", timer.read());

        // 3. col2im back into the original input layout for each batch slice.
        timer.reset();
        for (0..tile_batch) |local_n| {
            const n = batch_start + local_n;
            col2im(T, d, plan.output.data, plan.output.strides, col_buf[0 .. d.K * tile_cols], n, tile_cols, local_n * d.N);
        }
        addConvPhase(phase_profile, "bwd_input_col2im_ns", timer.read());

        batch_start += tile_batch;
    }
}

fn executeConv2dBwdKernelPlan(comptime T: type, plan: Conv2dBwdKernelPlan(T), phase_profile: ?*ConvPhaseProfile) void {
    const d = Conv2dDims(T).fromBwdKernel(plan);
    const ws = ConvWorkspacePlan(T).init(.bwd_kernel, d);
    const total_scratch = ws.total_elems;

    const scratch = plan.scratch orelse allocScratch(T, total_scratch) orelse {
        conv2dBwdKernelNaive(T, d, plan);
        return;
    };
    defer if (plan.scratch == null) freeScratch(T, scratch);

    const col_buf = ws.colBuf(scratch);
    const grad_buf = ws.auxBuf(scratch);
    const partial = ws.partialBuf(scratch);
    var timer = std.time.Timer.start() catch @panic("timer");
    @memset(plan.output.data, 0);
    const use_partial = ws.batch_tile < d.batch;

    const mm = selectConvMatMul(T);
    var batch_start: usize = 0;
    while (batch_start < d.batch) {
        const tile_batch = @min(ws.batch_tile, d.batch - batch_start);
        const tile_cols = d.N * tile_batch;

        timer.reset();
        for (0..tile_batch) |local_n| {
            const n = batch_start + local_n;
            im2col(T, d, col_buf, n, tile_cols, local_n * d.N);
        }
        addConvPhase(phase_profile, "bwd_kernel_im2col_ns", timer.read());

        timer.reset();
        for (0..d.c_out) |oc| {
            for (0..tile_batch) |local_n| {
                const n = batch_start + local_n;
                forward.simdCopy(T, grad_buf[oc * tile_cols + local_n * d.N ..][0..d.N], plan.output_grad.data[n * d.N * d.c_out + oc * d.N ..][0..d.N]);
            }
        }
        addConvPhase(phase_profile, "bwd_kernel_rearrange_ns", timer.read());

        const gemm_dst = if (use_partial) partial[0 .. d.c_out * d.K] else plan.output.data;
        timer.reset();
        mm(gemm_dst, grad_buf[0 .. d.c_out * tile_cols], col_buf[0 .. d.K * tile_cols], d.c_out, d.K, tile_cols, tile_cols, 1, 1, tile_cols, 0, 0, 0, d.K);
        addConvPhase(phase_profile, "bwd_kernel_gemm_ns", timer.read());

        if (use_partial) {
            timer.reset();
            forward.simdAccumulate(T, plan.output.data, partial[0 .. d.c_out * d.K]);
            addConvPhase(phase_profile, "bwd_kernel_rearrange_ns", timer.read());
        }

        batch_start += tile_batch;
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

pub fn executeFusionPlan(comptime T: type, plan: FusionPlan(T), phase_profile: ?*ConvPhaseProfile) void {
    switch (plan.payload) {
        .elementwise_chain => |chain_plan| executeFusedChain(T, chain_plan),
        .conv2d => |conv2d_plan| executeConv2dPlan(T, conv2d_plan, phase_profile),
        .conv2d_bwd_input => |conv2d_plan| executeConv2dBwdInputPlan(T, conv2d_plan, phase_profile),
        .conv2d_bwd_kernel => |conv2d_plan| executeConv2dBwdKernelPlan(T, conv2d_plan, phase_profile),
        .max_pool2d => |pool_plan| executeMaxPool2d(T, pool_plan),
        .max_pool2d_bwd => |pool_plan| executeMaxPool2dBwd(T, pool_plan),
        .softmax => |softmax_plan| executeSoftmaxPlan(T, softmax_plan),
        .log_softmax => |log_softmax_plan| executeLogSoftmaxPlan(T, log_softmax_plan),
        .cross_entropy => |cross_entropy_plan| executeCrossEntropyPlan(T, cross_entropy_plan),
        .layer_norm => |layer_norm_plan| executeLayerNormPlan(T, layer_norm_plan),
    }
}

test "fused - swapped commutative 2-op chain" {
    const T = f32;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Chain: exp(x), then add(scalar_repeat, exp) with swapped operands
    // Expected: exp(x) + 2 for each element
    const input = try Tensor(T).init(a, &.{3});
    input.setData(&.{ 1, 2, 3 });
    const scalar = try Tensor(T).initScalar(a, 2);
    const scalar_rep = scalar.repeatLike(input);
    scalar_rep.compute(); // materialize the repeat so fused kernel can read it
    const exp_node = input.exp();
    const add_node = scalar_rep.add(exp_node);

    const plan = ElementwiseFusionPlan(T){
        .input = input,
        .nodes = &.{ exp_node, add_node },
        .other_operand_roles = &.{ .src1, .src0 },
    };

    executeFusedChain(T, plan);

    // Verify against hand-computed reference
    const expected = [_]T{ @exp(@as(T, 1)) + 2, @exp(@as(T, 2)) + 2, @exp(@as(T, 3)) + 2 };
    for (add_node.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}

test "fused - swapped commutative 3-op chain" {
    const T = f32;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Chain: exp(x), add(scalar_repeat, exp), log(add)
    // Expected: log(exp(x) + 2) for each element
    const input = try Tensor(T).init(a, &.{3});
    input.setData(&.{ 1, 2, 3 });
    const scalar = try Tensor(T).initScalar(a, 2);
    const scalar_rep = scalar.repeatLike(input);
    scalar_rep.compute(); // materialize the repeat
    const exp_node = input.exp();
    const add_node = scalar_rep.add(exp_node);
    const log_node = add_node.log();

    const plan = ElementwiseFusionPlan(T){
        .input = input,
        .nodes = &.{ exp_node, add_node, log_node },
        .other_operand_roles = &.{ .src1, .src0, .src1 },
    };

    executeFusedChain(T, plan);

    const expected = [_]T{
        @log(@exp(@as(T, 1)) + 2),
        @log(@exp(@as(T, 2)) + 2),
        @log(@exp(@as(T, 3)) + 2),
    };
    for (log_node.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}
