const std = @import("std");
const Op = @import("op.zig").Op;
const tensorlib = @import("tensor.zig");
const losslib = @import("loss.zig");
const max_dims = tensorlib.max_dims;
const Alloc = std.mem.Allocator;

pub const ValueId = enum(u32) { _ };
pub const Axis = u8;

pub const DType = enum {
    f16,
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,

    pub fn fromType(comptime T: type) DType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            i32 => .i32,
            i64 => .i64,
            u32 => .u32,
            u64 => .u64,
            else => @compileError("unsupported dtype for compiler IR: " ++ @typeName(T)),
        };
    }
};

pub const Shape = struct {
    ndims: u8,
    dims: [max_dims]usize,

    pub fn init(dims: []const usize) Shape {
        std.debug.assert(dims.len <= max_dims);
        var out = Shape{ .ndims = @intCast(dims.len), .dims = [_]usize{1} ** max_dims };
        for (dims, 0..) |d, i| out.dims[i] = d;
        return out;
    }

    pub fn fromTensor(comptime T: type, tensor: *const tensorlib.Tensor(T)) Shape {
        return .{ .ndims = tensor.n_dims, .dims = tensor.ne };
    }

    pub fn nElems(self: Shape) usize {
        var n: usize = 1;
        for (self.dims[0..self.ndims]) |d| n *= d;
        return n;
    }
};

pub const Strides = struct {
    values: [max_dims]usize,

    pub fn contiguous(shape: Shape) Strides {
        var out = Strides{ .values = [_]usize{0} ** max_dims };
        out.values[0] = 1;
        var i: usize = 1;
        while (i < max_dims) : (i += 1) {
            out.values[i] = out.values[i - 1] * shape.dims[i - 1];
        }
        return out;
    }
};

pub const UnaryOp = enum {
    neg,
    abs,
    sqrt,
    recip,
    exp,
    log,
    gelu,

    pub fn fromOp(op: Op) ?UnaryOp {
        return switch (op) {
            .neg => .neg,
            .abs => .abs,
            .sqrt => .sqrt,
            .recip => .recip,
            .exp => .exp,
            .log => .log,
            .gelu => .gelu,
            else => null,
        };
    }
};

pub const BinaryOp = enum {
    add,
    mul,

    pub fn fromOp(op: Op) ?BinaryOp {
        return switch (op) {
            .add => .add,
            .mul => .mul,
            else => null,
        };
    }
};

pub const ReduceOp = enum {
    sum,
    max,

    pub fn fromOp(op: Op) ?ReduceOp {
        return switch (op) {
            .sum => .sum,
            .max => .max,
            else => null,
        };
    }
};

pub const ScaleSpec = struct {
    input: ValueId,
    scalar: ValueId,
};

pub const SoftmaxSpec = struct {
    input: ValueId,
    max_reduce: ValueId,
    shifted: ValueId,
    exp: ValueId,
    denom: ValueId,
    output: ValueId,
};

pub const LogSoftmaxSpec = struct {
    input: ValueId,
    max_reduce: ValueId,
    shifted: ValueId,
    exp: ValueId,
    sum: ValueId,
    log_norm: ValueId,
    output: ValueId,
};

pub const CrossEntropySpec = struct {
    input: ValueId,
    log_softmax: LogSoftmaxSpec,
    targets: ValueId,
    picked: ValueId,
    neg_picked: ValueId,
    sum: ValueId,
    mean: ValueId,
};

pub const LayerNormSpec = struct {
    input: ValueId,
    sum: ValueId,
    mean: ValueId,
    centered: ValueId,
    sqr: ValueId,
    var_sum: ValueId,
    variance: ValueId,
    variance_plus_eps: ValueId,
    sqrt_node: ValueId,
    recip_node: ValueId,
    output: ValueId,
};

pub const SoftmaxKernelSpec = struct {
    input: ValueId,
    max_node: ValueId,
    rep_max: ValueId,
    neg_rep_max: ValueId,
    shifted: ValueId,
    exp_node: ValueId,
    sum_node: ValueId,
    rep_sum: ValueId,
    recip_rep_sum: ValueId,
    output: ValueId,
};

pub const LogSoftmaxKernelSpec = struct {
    input: ValueId,
    max_node: ValueId,
    rep_max: ValueId,
    neg_rep_max: ValueId,
    shifted: ValueId,
    exp_node: ValueId,
    sum_node: ValueId,
    log_node: ValueId,
    rep_log: ValueId,
    neg_rep_log: ValueId,
    output: ValueId,
};

pub const CrossEntropyKernelSpec = struct {
    log_softmax: LogSoftmaxKernelSpec,
    targets: ValueId,
    picked: ValueId,
    neg_picked: ValueId,
    sum_node: ValueId,
    mean_node: ValueId,
};

pub const LayerNormKernelSpec = struct {
    input: ValueId,
    sum_node: ValueId,
    mean_node: ValueId,
    rep_mean: ValueId,
    neg_rep_mean: ValueId,
    centered: ValueId,
    sqr_node: ValueId,
    var_sum: ValueId,
    var_node: ValueId,
    eps_like: ValueId,
    var_eps: ValueId,
    sqrt_node: ValueId,
    recip_node: ValueId,
    rep_std_inv: ValueId,
    output: ValueId,
};

pub const KernelPatternKind = enum {
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
};

pub const KernelPattern = union(KernelPatternKind) {
    softmax: SoftmaxKernelSpec,
    log_softmax: LogSoftmaxKernelSpec,
    cross_entropy: CrossEntropyKernelSpec,
    layer_norm: LayerNormKernelSpec,

    pub fn output(self: @This()) ValueId {
        return switch (self) {
            .softmax => |spec| spec.output,
            .log_softmax => |spec| spec.output,
            .cross_entropy => |spec| spec.mean_node,
            .layer_norm => |spec| spec.output,
        };
    }

    pub fn markCovered(self: @This(), claimed: []bool) void {
        switch (self) {
            .softmax => |spec| {
                claimed[@intFromEnum(spec.max_node)] = true;
                claimed[@intFromEnum(spec.rep_max)] = true;
                claimed[@intFromEnum(spec.neg_rep_max)] = true;
                claimed[@intFromEnum(spec.shifted)] = true;
                claimed[@intFromEnum(spec.exp_node)] = true;
                claimed[@intFromEnum(spec.sum_node)] = true;
                claimed[@intFromEnum(spec.rep_sum)] = true;
                claimed[@intFromEnum(spec.recip_rep_sum)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .log_softmax => |spec| {
                claimed[@intFromEnum(spec.max_node)] = true;
                claimed[@intFromEnum(spec.rep_max)] = true;
                claimed[@intFromEnum(spec.neg_rep_max)] = true;
                claimed[@intFromEnum(spec.shifted)] = true;
                claimed[@intFromEnum(spec.exp_node)] = true;
                claimed[@intFromEnum(spec.sum_node)] = true;
                claimed[@intFromEnum(spec.log_node)] = true;
                claimed[@intFromEnum(spec.rep_log)] = true;
                claimed[@intFromEnum(spec.neg_rep_log)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .cross_entropy => |spec| {
                const log_softmax_pattern = KernelPattern{ .log_softmax = spec.log_softmax };
                log_softmax_pattern.markCovered(claimed);
                claimed[@intFromEnum(spec.picked)] = true;
                claimed[@intFromEnum(spec.neg_picked)] = true;
                claimed[@intFromEnum(spec.sum_node)] = true;
                claimed[@intFromEnum(spec.mean_node)] = true;
            },
            .layer_norm => |spec| {
                claimed[@intFromEnum(spec.sum_node)] = true;
                claimed[@intFromEnum(spec.mean_node)] = true;
                claimed[@intFromEnum(spec.rep_mean)] = true;
                claimed[@intFromEnum(spec.neg_rep_mean)] = true;
                claimed[@intFromEnum(spec.centered)] = true;
                claimed[@intFromEnum(spec.sqr_node)] = true;
                claimed[@intFromEnum(spec.var_sum)] = true;
                claimed[@intFromEnum(spec.var_node)] = true;
                claimed[@intFromEnum(spec.eps_like)] = true;
                claimed[@intFromEnum(spec.var_eps)] = true;
                claimed[@intFromEnum(spec.sqrt_node)] = true;
                claimed[@intFromEnum(spec.recip_node)] = true;
                claimed[@intFromEnum(spec.rep_std_inv)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
        }
    }

    pub fn overlapsClaimed(self: @This(), claimed: []const bool) bool {
        var tmp = claimed;
        _ = &tmp;
        return switch (self) {
            .softmax => |spec| claimed[@intFromEnum(spec.max_node)] or
                claimed[@intFromEnum(spec.rep_max)] or
                claimed[@intFromEnum(spec.neg_rep_max)] or
                claimed[@intFromEnum(spec.shifted)] or
                claimed[@intFromEnum(spec.exp_node)] or
                claimed[@intFromEnum(spec.sum_node)] or
                claimed[@intFromEnum(spec.rep_sum)] or
                claimed[@intFromEnum(spec.recip_rep_sum)] or
                claimed[@intFromEnum(spec.output)],
            .log_softmax => |spec| claimed[@intFromEnum(spec.max_node)] or
                claimed[@intFromEnum(spec.rep_max)] or
                claimed[@intFromEnum(spec.neg_rep_max)] or
                claimed[@intFromEnum(spec.shifted)] or
                claimed[@intFromEnum(spec.exp_node)] or
                claimed[@intFromEnum(spec.sum_node)] or
                claimed[@intFromEnum(spec.log_node)] or
                claimed[@intFromEnum(spec.rep_log)] or
                claimed[@intFromEnum(spec.neg_rep_log)] or
                claimed[@intFromEnum(spec.output)],
            .cross_entropy => |spec| blk: {
                const log_softmax_pattern = KernelPattern{ .log_softmax = spec.log_softmax };
                break :blk log_softmax_pattern.overlapsClaimed(claimed) or
                    claimed[@intFromEnum(spec.picked)] or
                    claimed[@intFromEnum(spec.neg_picked)] or
                    claimed[@intFromEnum(spec.sum_node)] or
                    claimed[@intFromEnum(spec.mean_node)];
            },
            .layer_norm => |spec| claimed[@intFromEnum(spec.sum_node)] or
                claimed[@intFromEnum(spec.mean_node)] or
                claimed[@intFromEnum(spec.rep_mean)] or
                claimed[@intFromEnum(spec.neg_rep_mean)] or
                claimed[@intFromEnum(spec.centered)] or
                claimed[@intFromEnum(spec.sqr_node)] or
                claimed[@intFromEnum(spec.var_sum)] or
                claimed[@intFromEnum(spec.var_node)] or
                claimed[@intFromEnum(spec.eps_like)] or
                claimed[@intFromEnum(spec.var_eps)] or
                claimed[@intFromEnum(spec.sqrt_node)] or
                claimed[@intFromEnum(spec.recip_node)] or
                claimed[@intFromEnum(spec.rep_std_inv)] or
                claimed[@intFromEnum(spec.output)],
        };
    }
};

pub const KernelPatternRecord = struct {
    output: ValueId,
    pattern: KernelPattern,
};

const kernel_pattern_priority = [_]KernelPatternKind{
    .cross_entropy,
    .layer_norm,
    .log_softmax,
    .softmax,
};

pub const RmsNormSpec = struct {
    input: ValueId,
    sqr: ValueId,
    mean: ValueId,
    variance_plus_eps: ValueId,
    sqrt_node: ValueId,
    recip_node: ValueId,
    output: ValueId,
};

pub const RewriteKind = enum {
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
};

pub const RewriteResult = union(RewriteKind) {
    softmax: SoftmaxSpec,
    log_softmax: LogSoftmaxSpec,
    cross_entropy: CrossEntropySpec,
    layer_norm: LayerNormSpec,
};

pub const RewritePass = struct {
    kind: RewriteKind,

    pub fn apply(self: RewritePass, graph: *const CanonicalGraph, output: ValueId) ?RewriteResult {
        return switch (self.kind) {
            .softmax => if (graph.detectSoftmax(output)) |spec| .{ .softmax = spec } else null,
            .log_softmax => if (graph.detectLogSoftmax(output)) |spec| .{ .log_softmax = spec } else null,
            .cross_entropy => if (graph.detectCrossEntropy(output)) |spec| .{ .cross_entropy = spec } else null,
            .layer_norm => if (graph.detectLayerNorm(output)) |spec| .{ .layer_norm = spec } else null,
        };
    }
};

pub const CanonicalPatternKind = enum {
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
};

pub const CanonicalPattern = union(CanonicalPatternKind) {
    softmax: SoftmaxSpec,
    log_softmax: LogSoftmaxSpec,
    cross_entropy: CrossEntropySpec,
    layer_norm: LayerNormSpec,
};

const SoftmaxOutputParts = struct {
    exp: ValueId,
    denom: ValueId,
};

pub const ViewKind = enum {
    reshape,
    transpose,
    broadcast,
};

pub const ViewSpec = struct {
    kind: ViewKind,
    input: ValueId,
    shape: Shape,
    strides: ?Strides = null,
};

pub const BinarySpec = struct {
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
};

pub const MatmulSpec = struct {
    lhs: ValueId,
    rhs: ValueId,
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
};

pub const GatherSpec = struct {
    table: ValueId,
    indices: ValueId,
    axis: Axis,
};

pub const ScatterAddSpec = struct {
    dst_shape: Shape,
    indices: ValueId,
    updates: ValueId,
    axis: Axis,
};

pub const CanonicalExpr = union(enum) {
    input,
    constant,
    unary: struct { op: UnaryOp, input: ValueId },
    binary: BinarySpec,
    reduce: struct { op: ReduceOp, input: ValueId, axes: []const Axis },
    scale: ScaleSpec,
    view: ViewSpec,
    gather: GatherSpec,
    scatter_add: ScatterAddSpec,
    matmul: MatmulSpec,
};

pub const CanonicalValue = struct {
    id: ValueId,
    dtype: DType,
    shape: Shape,
    expr: CanonicalExpr,
};

pub const CanonicalGraph = struct {
    alloc: std.mem.Allocator,
    values: std.ArrayList(CanonicalValue),
    reduction_axes_storage: std.ArrayList([]Axis),

    pub fn init(alloc: std.mem.Allocator) CanonicalGraph {
        return .{ .alloc = alloc, .values = .{}, .reduction_axes_storage = .{} };
    }

    pub fn deinit(self: *CanonicalGraph) void {
        for (self.reduction_axes_storage.items) |axes| self.alloc.free(axes);
        self.reduction_axes_storage.deinit(self.alloc);
        self.values.deinit(self.alloc);
    }

    pub fn addValue(self: *CanonicalGraph, dtype: DType, shape: Shape, expr: CanonicalExpr) !ValueId {
        const id: ValueId = @enumFromInt(self.values.items.len);
        try self.values.append(self.alloc, .{ .id = id, .dtype = dtype, .shape = shape, .expr = expr });
        return id;
    }

    pub fn dupeAxes(self: *CanonicalGraph, axes: []const Axis) ![]const Axis {
        const dup = try self.alloc.dupe(Axis, axes);
        try self.reduction_axes_storage.append(self.alloc, dup);
        return dup;
    }

    pub fn value(self: *const CanonicalGraph, id: ValueId) *const CanonicalValue {
        return &self.values.items[@intFromEnum(id)];
    }

    pub fn canonicalize(self: *const CanonicalGraph, alloc: Alloc) !CanonicalGraph {
        var out = CanonicalGraph.init(alloc);
        errdefer out.deinit();

        for (self.values.items) |item| {
            const expr = try canonicalizeExpr(self, &out, item);
            _ = try out.addValue(item.dtype, item.shape, expr);
        }
        return out;
    }

    pub fn detectSoftmax(self: *const CanonicalGraph, output: ValueId) ?SoftmaxSpec {
        const out_v = self.value(output);
        const output_parts: SoftmaxOutputParts = switch (out_v.expr) {
            .scale => |s| .{ .exp = s.input, .denom = s.scalar },
            .binary => |b| blk: {
                if (b.op != .mul) return null;
                const rhs = self.value(b.rhs);
                if (rhs.expr != .unary or rhs.expr.unary.op != .recip) return null;
                const denom_bcast = self.value(rhs.expr.unary.input);
                if (denom_bcast.expr != .view or denom_bcast.expr.view.kind != .broadcast) return null;
                break :blk .{ .exp = b.lhs, .denom = denom_bcast.expr.view.input };
            },
            else => return null,
        };

        const exp_v = self.value(output_parts.exp);
        const denom_v = self.value(output_parts.denom);
        if (exp_v.expr != .unary or exp_v.expr.unary.op != .exp) return null;
        if (denom_v.expr != .reduce or denom_v.expr.reduce.op != .sum) return null;
        if (denom_v.expr.reduce.input != output_parts.exp) return null;

        const shifted = exp_v.expr.unary.input;
        const shifted_v = self.value(shifted);
        if (shifted_v.expr != .binary or shifted_v.expr.binary.op != .add) return null;

        const lhs = self.value(shifted_v.expr.binary.lhs);
        const rhs = self.value(shifted_v.expr.binary.rhs);

        const neg_rep = if (rhs.expr == .unary and rhs.expr.unary.op == .neg) rhs else return null;
        const rep_max = self.value(neg_rep.expr.unary.input);
        if (rep_max.expr != .view or rep_max.expr.view.kind != .broadcast) return null;

        const max_reduce = rep_max.expr.view.input;
        const max_v = self.value(max_reduce);
        if (max_v.expr != .reduce or max_v.expr.reduce.op != .max) return null;
        if (max_v.expr.reduce.input != shifted_v.expr.binary.lhs) return null;
        _ = lhs;

        return .{
            .input = shifted_v.expr.binary.lhs,
            .max_reduce = max_reduce,
            .shifted = shifted,
            .exp = output_parts.exp,
            .denom = output_parts.denom,
            .output = output,
        };
    }

    pub fn detectLogSoftmax(self: *const CanonicalGraph, output: ValueId) ?LogSoftmaxSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .add) return null;

        const shifted = out_v.expr.binary.lhs;
        const neg_log = out_v.expr.binary.rhs;
        const neg_log_v = self.value(neg_log);
        if (neg_log_v.expr != .unary or neg_log_v.expr.unary.op != .neg) return null;

        const rep_log = neg_log_v.expr.unary.input;
        const rep_log_v = self.value(rep_log);
        if (rep_log_v.expr != .view or rep_log_v.expr.view.kind != .broadcast) return null;

        const log_norm = rep_log_v.expr.view.input;
        const log_norm_v = self.value(log_norm);
        if (log_norm_v.expr != .unary or log_norm_v.expr.unary.op != .log) return null;

        const sum = log_norm_v.expr.unary.input;
        const sum_v = self.value(sum);
        if (sum_v.expr != .reduce or sum_v.expr.reduce.op != .sum) return null;

        const exp = sum_v.expr.reduce.input;
        const exp_v = self.value(exp);
        if (exp_v.expr != .unary or exp_v.expr.unary.op != .exp) return null;
        if (exp_v.expr.unary.input != shifted) return null;

        const shifted_v = self.value(shifted);
        if (shifted_v.expr != .binary or shifted_v.expr.binary.op != .add) return null;

        const rhs = self.value(shifted_v.expr.binary.rhs);
        if (rhs.expr != .unary or rhs.expr.unary.op != .neg) return null;
        const rep_max = self.value(rhs.expr.unary.input);
        if (rep_max.expr != .view or rep_max.expr.view.kind != .broadcast) return null;

        const max_reduce = rep_max.expr.view.input;
        const max_v = self.value(max_reduce);
        if (max_v.expr != .reduce or max_v.expr.reduce.op != .max) return null;
        if (max_v.expr.reduce.input != shifted_v.expr.binary.lhs) return null;

        return .{
            .input = shifted_v.expr.binary.lhs,
            .max_reduce = max_reduce,
            .shifted = shifted,
            .exp = exp,
            .sum = sum,
            .log_norm = log_norm,
            .output = output,
        };
    }

    pub fn detectCrossEntropy(self: *const CanonicalGraph, output: ValueId) ?CrossEntropySpec {
        const out_v = self.value(output);
        const scaled = switch (out_v.expr) {
            .scale => |s| s,
            .binary => |b| blk: {
                if (b.op != .mul) return null;
                if (isScalarBroadcastValue(self, b.rhs)) {
                    break :blk ScaleSpec{ .input = b.lhs, .scalar = scalarSourceValueId(self, b.rhs) };
                }
                if (isScalarBroadcastValue(self, b.lhs)) {
                    break :blk ScaleSpec{ .input = b.rhs, .scalar = scalarSourceValueId(self, b.lhs) };
                }
                return null;
            },
            else => return null,
        };

        const sum = scaled.input;
        const scale_scalar = scaled.scalar;
        const sum_v = self.value(sum);
        if (sum_v.expr != .reduce or sum_v.expr.reduce.op != .sum) return null;

        const neg_picked = sum_v.expr.reduce.input;
        const neg_picked_v = self.value(neg_picked);
        if (neg_picked_v.expr != .unary or neg_picked_v.expr.unary.op != .neg) return null;

        const picked = neg_picked_v.expr.unary.input;
        const picked_v = self.value(picked);
        if (picked_v.expr != .gather or picked_v.expr.gather.axis != 0) return null;

        const log_softmax_out = picked_v.expr.gather.table;
        const log_softmax = self.detectLogSoftmax(log_softmax_out) orelse return null;

        const scalar_v = self.value(scale_scalar);
        if (scalar_v.shape.nElems() != 1) return null;

        return .{
            .input = log_softmax.input,
            .log_softmax = log_softmax,
            .targets = picked_v.expr.gather.indices,
            .picked = picked,
            .neg_picked = neg_picked,
            .sum = sum,
            .mean = output,
        };
    }

    pub fn detectRmsNorm(self: *const CanonicalGraph, output: ValueId) ?RmsNormSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .mul) return null;

        const lhs = out_v.expr.binary.lhs;
        const rhs = out_v.expr.binary.rhs;
        const rhs_v = self.value(rhs);
        if (rhs_v.expr != .view or rhs_v.expr.view.kind != .broadcast) return null;

        const recip = rhs_v.expr.view.input;
        const recip_v = self.value(recip);
        if (recip_v.expr != .unary or recip_v.expr.unary.op != .recip) return null;

        const sqrt_node = recip_v.expr.unary.input;
        const sqrt_v = self.value(sqrt_node);
        if (sqrt_v.expr != .unary or sqrt_v.expr.unary.op != .sqrt) return null;

        const variance_plus_eps = sqrt_v.expr.unary.input;
        const vpe_v = self.value(variance_plus_eps);
        if (vpe_v.expr != .binary or vpe_v.expr.binary.op != .add) return null;

        const mean = vpe_v.expr.binary.lhs;
        const eps_like = vpe_v.expr.binary.rhs;
        const eps_v = self.value(eps_like);
        if (eps_v.expr != .view or eps_v.expr.view.kind != .broadcast) return null;

        const mean_v = self.value(mean);
        if (mean_v.expr != .scale) return null;
        const mean_input = mean_v.expr.scale.input;

        const sqr_v = self.value(mean_input);
        if (sqr_v.expr != .binary or sqr_v.expr.binary.op != .mul) return null;
        if (sqr_v.expr.binary.lhs != lhs or sqr_v.expr.binary.rhs != lhs) return null;

        return .{
            .input = lhs,
            .sqr = mean_input,
            .mean = mean,
            .variance_plus_eps = variance_plus_eps,
            .sqrt_node = sqrt_node,
            .recip_node = recip,
            .output = output,
        };
    }

    pub fn detectLayerNorm(self: *const CanonicalGraph, output: ValueId) ?LayerNormSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .mul) return null;

        const centered = out_v.expr.binary.lhs;
        const rep_std_inv = out_v.expr.binary.rhs;
        const rep_std_inv_v = self.value(rep_std_inv);
        if (rep_std_inv_v.expr != .view or rep_std_inv_v.expr.view.kind != .broadcast) return null;

        const recip_node = rep_std_inv_v.expr.view.input;
        const recip_v = self.value(recip_node);
        if (recip_v.expr != .unary or recip_v.expr.unary.op != .recip) return null;

        const sqrt_node = recip_v.expr.unary.input;
        const sqrt_v = self.value(sqrt_node);
        if (sqrt_v.expr != .unary or sqrt_v.expr.unary.op != .sqrt) return null;

        const variance_plus_eps = sqrt_v.expr.unary.input;
        const variance_plus_eps_v = self.value(variance_plus_eps);
        if (variance_plus_eps_v.expr != .binary or variance_plus_eps_v.expr.binary.op != .add) return null;

        const variance = variance_plus_eps_v.expr.binary.lhs;
        const eps_like = variance_plus_eps_v.expr.binary.rhs;
        if (!isScalarBroadcastValue(self, eps_like)) return null;

        const variance_v = self.value(variance);
        const variance_scale = switch (variance_v.expr) {
            .scale => |s| s,
            .binary => |b| blk: {
                if (b.op != .mul) return null;
                if (isScalarBroadcastValue(self, b.rhs)) {
                    break :blk ScaleSpec{ .input = b.lhs, .scalar = scalarSourceValueId(self, b.rhs) };
                }
                if (isScalarBroadcastValue(self, b.lhs)) {
                    break :blk ScaleSpec{ .input = b.rhs, .scalar = scalarSourceValueId(self, b.lhs) };
                }
                return null;
            },
            else => return null,
        };

        const var_sum = variance_scale.input;
        const var_sum_v = self.value(var_sum);
        if (var_sum_v.expr != .reduce or var_sum_v.expr.reduce.op != .sum) return null;

        const sqr = var_sum_v.expr.reduce.input;
        const sqr_v = self.value(sqr);
        if (sqr_v.expr != .binary or sqr_v.expr.binary.op != .mul) return null;
        if (sqr_v.expr.binary.lhs != centered or sqr_v.expr.binary.rhs != centered) return null;

        const centered_v = self.value(centered);
        if (centered_v.expr != .binary or centered_v.expr.binary.op != .add) return null;

        const input = centered_v.expr.binary.lhs;
        const neg_rep_mean = centered_v.expr.binary.rhs;
        const neg_rep_mean_v = self.value(neg_rep_mean);
        if (neg_rep_mean_v.expr != .unary or neg_rep_mean_v.expr.unary.op != .neg) return null;

        const rep_mean = neg_rep_mean_v.expr.unary.input;
        const rep_mean_v = self.value(rep_mean);
        if (rep_mean_v.expr != .view or rep_mean_v.expr.view.kind != .broadcast) return null;

        const mean = rep_mean_v.expr.view.input;
        const mean_v = self.value(mean);
        const mean_scale = switch (mean_v.expr) {
            .scale => |s| s,
            .binary => |b| blk: {
                if (b.op != .mul) return null;
                if (isScalarBroadcastValue(self, b.rhs)) {
                    break :blk ScaleSpec{ .input = b.lhs, .scalar = scalarSourceValueId(self, b.rhs) };
                }
                if (isScalarBroadcastValue(self, b.lhs)) {
                    break :blk ScaleSpec{ .input = b.rhs, .scalar = scalarSourceValueId(self, b.lhs) };
                }
                return null;
            },
            else => return null,
        };

        const sum = mean_scale.input;
        const sum_v = self.value(sum);
        if (sum_v.expr != .reduce or sum_v.expr.reduce.op != .sum) return null;
        if (sum_v.expr.reduce.input != input) return null;

        return .{
            .input = input,
            .sum = sum,
            .mean = mean,
            .centered = centered,
            .sqr = sqr,
            .var_sum = var_sum,
            .variance = variance,
            .variance_plus_eps = variance_plus_eps,
            .sqrt_node = sqrt_node,
            .recip_node = recip_node,
            .output = output,
        };
    }

    pub fn detectPattern(self: *const CanonicalGraph, kind: CanonicalPatternKind, output: ValueId) ?CanonicalPattern {
        return switch (kind) {
            .softmax => if (self.detectSoftmax(output)) |spec| .{ .softmax = spec } else null,
            .log_softmax => if (self.detectLogSoftmax(output)) |spec| .{ .log_softmax = spec } else null,
            .cross_entropy => if (self.detectCrossEntropy(output)) |spec| .{ .cross_entropy = spec } else null,
            .layer_norm => if (self.detectLayerNorm(output)) |spec| .{ .layer_norm = spec } else null,
        };
    }

    pub fn detectPatterns(self: *const CanonicalGraph, output: ValueId, alloc: Alloc) !std.ArrayList(CanonicalPattern) {
        var out = std.ArrayList(CanonicalPattern){};
        errdefer out.deinit(alloc);

        inline for (std.meta.tags(CanonicalPatternKind)) |kind| {
            if (self.detectPattern(kind, output)) |pattern| {
                try out.append(alloc, pattern);
            }
        }
        return out;
    }

    pub fn applyRewritePasses(self: *const CanonicalGraph, output: ValueId, alloc: Alloc) !std.ArrayList(RewriteResult) {
        var out = std.ArrayList(RewriteResult){};
        errdefer out.deinit(alloc);

        inline for (std.meta.tags(RewriteKind)) |kind| {
            const pass = RewritePass{ .kind = kind };
            if (pass.apply(self, output)) |result| {
                try out.append(alloc, result);
            }
        }
        return out;
    }

    pub fn detectKernelPattern(self: *const CanonicalGraph, kind: KernelPatternKind, output: ValueId) ?KernelPattern {
        return switch (kind) {
            .softmax => if (self.detectSoftmax(output)) |spec| .{ .softmax = lowerSoftmaxKernelSpec(self, spec) orelse return null } else null,
            .log_softmax => if (self.detectLogSoftmax(output)) |spec| .{ .log_softmax = lowerLogSoftmaxKernelSpec(self, spec) orelse return null } else null,
            .cross_entropy => if (self.detectCrossEntropy(output)) |spec| .{ .cross_entropy = lowerCrossEntropyKernelSpec(self, spec) orelse return null } else null,
            .layer_norm => if (self.detectLayerNorm(output)) |spec| .{ .layer_norm = lowerLayerNormKernelSpec(self, spec) orelse return null } else null,
        };
    }

    pub fn collectKernelPatterns(self: *const CanonicalGraph, alloc: Alloc) !std.ArrayList(KernelPatternRecord) {
        var out = std.ArrayList(KernelPatternRecord){};
        errdefer out.deinit(alloc);

        const claimed = try alloc.alloc(bool, self.values.items.len);
        defer alloc.free(claimed);
        @memset(claimed, false);

        var idx = self.values.items.len;
        while (idx > 0) {
            idx -= 1;
            const output: ValueId = @enumFromInt(idx);
            const pattern = blk: {
                inline for (kernel_pattern_priority) |kind| {
                    if (self.detectKernelPattern(kind, output)) |p| {
                        if (!p.overlapsClaimed(claimed)) break :blk p;
                    }
                }
                break :blk null;
            };
            if (pattern) |p| {
                p.markCovered(claimed);
                try out.append(alloc, .{ .output = p.output(), .pattern = p });
            }
        }

        std.mem.sort(KernelPatternRecord, out.items, {}, sortKernelPatternRecords);
        return out;
    }
};

pub const KernelExpr = union(enum) {
    map: struct { op: UnaryOp, input: ValueId },
    zip: struct { op: BinaryOp, lhs: ValueId, rhs: ValueId },
    reduce: struct { op: ReduceOp, input: ValueId, axes: []const Axis },
    broadcast: ViewSpec,
    gather: GatherSpec,
    scatter_add: ScatterAddSpec,
    matmul: MatmulSpec,
};

pub const KernelValue = struct {
    id: ValueId,
    dtype: DType,
    shape: Shape,
    expr: KernelExpr,
};

pub const KernelAnnotation = struct {
    output: ValueId,
    rewrite: RewriteResult,
};

fn sortKernelPatternRecords(_: void, lhs: KernelPatternRecord, rhs: KernelPatternRecord) bool {
    return @intFromEnum(lhs.output) < @intFromEnum(rhs.output);
}

pub const KernelPlan = struct {
    alloc: std.mem.Allocator,
    values: std.ArrayList(KernelValue),
    outputs: std.ArrayList(ValueId),
    annotations: std.ArrayList(KernelAnnotation),
    patterns: std.ArrayList(KernelPatternRecord),

    pub fn init(alloc: std.mem.Allocator) KernelPlan {
        return .{
            .alloc = alloc,
            .values = .{},
            .outputs = .{},
            .annotations = .{},
            .patterns = .{},
        };
    }

    pub fn deinit(self: *KernelPlan) void {
        self.values.deinit(self.alloc);
        self.outputs.deinit(self.alloc);
        self.annotations.deinit(self.alloc);
        self.patterns.deinit(self.alloc);
    }

    pub fn lowerFromCanonical(alloc: Alloc, graph: *const CanonicalGraph) !KernelPlan {
        var plan = KernelPlan.init(alloc);
        errdefer plan.deinit();

        for (graph.values.items) |value| {
            const expr = switch (value.expr) {
                .input, .constant => continue,
                .unary => |u| KernelExpr{ .map = .{ .op = u.op, .input = u.input } },
                .binary => |b| KernelExpr{ .zip = .{ .op = b.op, .lhs = b.lhs, .rhs = b.rhs } },
                .reduce => |r| KernelExpr{ .reduce = .{ .op = r.op, .input = r.input, .axes = r.axes } },
                .scale => |s| KernelExpr{ .zip = .{ .op = .mul, .lhs = s.input, .rhs = s.scalar } },
                .view => |v| KernelExpr{ .broadcast = v },
                .gather => |g| KernelExpr{ .gather = g },
                .scatter_add => |s| KernelExpr{ .scatter_add = s },
                .matmul => |m| KernelExpr{ .matmul = m },
            };

            try plan.values.append(alloc, .{
                .id = value.id,
                .dtype = value.dtype,
                .shape = value.shape,
                .expr = expr,
            });
        }

        if (graph.values.items.len > 0) {
            try plan.outputs.append(alloc, graph.values.items[graph.values.items.len - 1].id);
        }
        return plan;
    }

    pub fn addRewriteAnnotations(self: *KernelPlan, alloc: Alloc, output: ValueId, rewrites: []const RewriteResult) !void {
        for (rewrites) |rewrite| {
            try self.annotations.append(alloc, .{ .output = output, .rewrite = rewrite });
        }
    }

    pub fn buildPatterns(self: *KernelPlan, alloc: Alloc, graph: *const CanonicalGraph) !void {
        self.patterns = try graph.collectKernelPatterns(alloc);
    }
};

pub fn Pipeline(comptime T: type) type {
    const Tensor = tensorlib.Tensor(T);
    return struct {
        const Self = @This();

        alloc: Alloc,
        lowering: Lowering(T),
        canonical: ?CanonicalGraph = null,
        rewrites: std.ArrayList(RewriteResult),
        kernel: ?KernelPlan = null,

        pub fn init(alloc: Alloc) Self {
            return .{
                .alloc = alloc,
                .lowering = Lowering(T).init(alloc),
                .rewrites = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.kernel) |*k| k.deinit();
            if (self.canonical) |*c| c.deinit();
            self.rewrites.deinit(self.alloc);
            self.lowering.deinit();
        }

        pub fn compile(self: *Self, root: *const Tensor) (Alloc.Error || error{UnsupportedOp})!void {
            _ = try self.lowering.lowerRoot(root);

            self.canonical = try self.lowering.graph.canonicalize(self.alloc);
            errdefer if (self.canonical) |*c| c.deinit();

            const output: ValueId = @enumFromInt(self.canonical.?.values.items.len - 1);
            self.rewrites = try self.canonical.?.applyRewritePasses(output, self.alloc);
            errdefer self.rewrites.deinit(self.alloc);

            self.kernel = try KernelPlan.lowerFromCanonical(self.alloc, &self.canonical.?);
            try self.kernel.?.addRewriteAnnotations(self.alloc, output, self.rewrites.items);
            try self.kernel.?.buildPatterns(self.alloc, &self.canonical.?);
        }
    };
}

fn lowerSoftmaxKernelSpec(graph: *const CanonicalGraph, spec: SoftmaxSpec) ?SoftmaxKernelSpec {
    const output_v = graph.value(spec.output);
    const recip_rep_sum = switch (output_v.expr) {
        .scale => |s| s.scalar,
        .binary => |b| blk: {
            if (b.op != .mul) return null;
            const rhs = graph.value(b.rhs);
            if (rhs.expr != .unary or rhs.expr.unary.op != .recip) return null;
            break :blk b.rhs;
        },
        else => return null,
    };

    const recip_v = graph.value(recip_rep_sum);
    if (recip_v.expr != .unary or recip_v.expr.unary.op != .recip) return null;
    const rep_sum = recip_v.expr.unary.input;
    const rep_sum_v = graph.value(rep_sum);
    if (rep_sum_v.expr != .view or rep_sum_v.expr.view.kind != .broadcast) return null;
    if (rep_sum_v.expr.view.input != spec.denom) return null;

    const exp_v = graph.value(spec.exp);
    if (exp_v.expr != .unary or exp_v.expr.unary.op != .exp) return null;

    const shifted_v = graph.value(spec.shifted);
    if (shifted_v.expr != .binary or shifted_v.expr.binary.op != .add) return null;
    const neg_rep_max = shifted_v.expr.binary.rhs;
    const neg_rep_max_v = graph.value(neg_rep_max);
    if (neg_rep_max_v.expr != .unary or neg_rep_max_v.expr.unary.op != .neg) return null;
    const rep_max = neg_rep_max_v.expr.unary.input;

    return .{
        .input = spec.input,
        .max_node = spec.max_reduce,
        .rep_max = rep_max,
        .neg_rep_max = neg_rep_max,
        .shifted = spec.shifted,
        .exp_node = spec.exp,
        .sum_node = spec.denom,
        .rep_sum = rep_sum,
        .recip_rep_sum = recip_rep_sum,
        .output = spec.output,
    };
}

fn lowerLogSoftmaxKernelSpec(graph: *const CanonicalGraph, spec: LogSoftmaxSpec) ?LogSoftmaxKernelSpec {
    const output_v = graph.value(spec.output);
    if (output_v.expr != .binary or output_v.expr.binary.op != .add) return null;
    const neg_rep_log = output_v.expr.binary.rhs;
    const neg_rep_log_v = graph.value(neg_rep_log);
    if (neg_rep_log_v.expr != .unary or neg_rep_log_v.expr.unary.op != .neg) return null;
    const rep_log = neg_rep_log_v.expr.unary.input;

    const shifted_v = graph.value(spec.shifted);
    if (shifted_v.expr != .binary or shifted_v.expr.binary.op != .add) return null;
    const neg_rep_max = shifted_v.expr.binary.rhs;
    const neg_rep_max_v = graph.value(neg_rep_max);
    if (neg_rep_max_v.expr != .unary or neg_rep_max_v.expr.unary.op != .neg) return null;
    const rep_max = neg_rep_max_v.expr.unary.input;

    return .{
        .input = spec.input,
        .max_node = spec.max_reduce,
        .rep_max = rep_max,
        .neg_rep_max = neg_rep_max,
        .shifted = spec.shifted,
        .exp_node = spec.exp,
        .sum_node = spec.sum,
        .log_node = spec.log_norm,
        .rep_log = rep_log,
        .neg_rep_log = neg_rep_log,
        .output = spec.output,
    };
}

fn lowerCrossEntropyKernelSpec(graph: *const CanonicalGraph, spec: CrossEntropySpec) ?CrossEntropyKernelSpec {
    return .{
        .log_softmax = lowerLogSoftmaxKernelSpec(graph, spec.log_softmax) orelse return null,
        .targets = spec.targets,
        .picked = spec.picked,
        .neg_picked = spec.neg_picked,
        .sum_node = spec.sum,
        .mean_node = spec.mean,
    };
}

fn lowerLayerNormKernelSpec(graph: *const CanonicalGraph, spec: LayerNormSpec) ?LayerNormKernelSpec {
    const output_v = graph.value(spec.output);
    if (output_v.expr != .binary or output_v.expr.binary.op != .mul) return null;
    const rep_std_inv = output_v.expr.binary.rhs;

    const centered_v = graph.value(spec.centered);
    if (centered_v.expr != .binary or centered_v.expr.binary.op != .add) return null;
    const neg_rep_mean = centered_v.expr.binary.rhs;
    const neg_rep_mean_v = graph.value(neg_rep_mean);
    if (neg_rep_mean_v.expr != .unary or neg_rep_mean_v.expr.unary.op != .neg) return null;
    const rep_mean = neg_rep_mean_v.expr.unary.input;

    const variance_plus_eps_v = graph.value(spec.variance_plus_eps);
    if (variance_plus_eps_v.expr != .binary or variance_plus_eps_v.expr.binary.op != .add) return null;
    const eps_like = variance_plus_eps_v.expr.binary.rhs;

    return .{
        .input = spec.input,
        .sum_node = spec.sum,
        .mean_node = spec.mean,
        .rep_mean = rep_mean,
        .neg_rep_mean = neg_rep_mean,
        .centered = spec.centered,
        .sqr_node = spec.sqr,
        .var_sum = spec.var_sum,
        .var_node = spec.variance,
        .eps_like = eps_like,
        .var_eps = spec.variance_plus_eps,
        .sqrt_node = spec.sqrt_node,
        .recip_node = spec.recip_node,
        .rep_std_inv = rep_std_inv,
        .output = spec.output,
    };
}

pub fn Lowering(comptime T: type) type {
    const Tensor = tensorlib.Tensor(T);
    return struct {
        const Self = @This();

        alloc: Alloc,
        graph: CanonicalGraph,
        memo: std.AutoHashMap(*const Tensor, ValueId),

        pub fn init(alloc: Alloc) Self {
            return .{
                .alloc = alloc,
                .graph = CanonicalGraph.init(alloc),
                .memo = std.AutoHashMap(*const Tensor, ValueId).init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.memo.deinit();
            self.graph.deinit();
        }

        pub fn lowerRoot(self: *Self, root: *const Tensor) (Alloc.Error || error{UnsupportedOp})!ValueId {
            return self.lowerTensor(root);
        }

        fn lowerTensor(self: *Self, tensor: *const Tensor) (Alloc.Error || error{UnsupportedOp})!ValueId {
            if (self.memo.get(tensor)) |id| return id;

            const dtype = DType.fromType(T);
            const shape = Shape.fromTensor(T, tensor);
            const expr = try self.lowerExpr(tensor);
            const id = try self.graph.addValue(dtype, shape, expr);
            try self.memo.put(tensor, id);
            return id;
        }

        fn lowerExpr(self: *Self, tensor: *const Tensor) (Alloc.Error || error{UnsupportedOp})!CanonicalExpr {
            const op = tensor.opTag();

            if (op == .none) {
                return if (tensor.source0() == null and tensor.source1() == null)
                    .input
                else
                    .constant;
            }

            if (UnaryOp.fromOp(op)) |uop| {
                return .{ .unary = .{ .op = uop, .input = try self.lowerTensor(tensor.source0().?) } };
            }

            if (BinaryOp.fromOp(op)) |bop| {
                return .{ .binary = .{
                    .op = bop,
                    .lhs = try self.lowerTensor(tensor.source0().?),
                    .rhs = try self.lowerTensor(tensor.source1().?),
                } };
            }

            if (ReduceOp.fromOp(op)) |rop| {
                const axes = try inferReductionAxes(self.graph.alloc, tensor.source0().?, tensor);
                defer self.graph.alloc.free(axes);
                return .{ .reduce = .{
                    .op = rop,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .axes = try self.graph.dupeAxes(axes),
                } };
            }

            return switch (op) {
                .view => .{ .view = .{
                    .kind = .broadcast,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .reshape => .{ .view = .{
                    .kind = .reshape,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .transpose => .{ .view = .{
                    .kind = .transpose,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .repeat => .{ .view = .{
                    .kind = .broadcast,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .gather_rows => .{ .gather = .{
                    .table = try self.lowerTensor(tensor.source0().?),
                    .indices = try self.lowerTensor(tensor.source1().?),
                    .axis = 1,
                } },
                .pick_rows => .{ .gather = .{
                    .table = try self.lowerTensor(tensor.source0().?),
                    .indices = try self.lowerTensor(tensor.source1().?),
                    .axis = 0,
                } },
                .scatter_add_rows => .{ .scatter_add = .{
                    .dst_shape = Shape.fromTensor(T, tensor),
                    .indices = try self.lowerTensor(tensor.source1().?),
                    .updates = try self.lowerTensor(tensor.source0().?),
                    .axis = 1,
                } },
                .scatter_add_picks => .{ .scatter_add = .{
                    .dst_shape = Shape.fromTensor(T, tensor),
                    .indices = try self.lowerTensor(tensor.source1().?),
                    .updates = try self.lowerTensor(tensor.source0().?),
                    .axis = 0,
                } },
                .matmul => .{ .matmul = .{
                    .lhs = try self.lowerTensor(tensor.source0().?),
                    .rhs = try self.lowerTensor(tensor.source1().?),
                } },
                .matmul_t0 => .{ .matmul = .{
                    .lhs = try self.lowerTensor(tensor.source0().?),
                    .rhs = try self.lowerTensor(tensor.source1().?),
                    .transpose_lhs = true,
                } },
                .matmul_t1 => .{ .matmul = .{
                    .lhs = try self.lowerTensor(tensor.source0().?),
                    .rhs = try self.lowerTensor(tensor.source1().?),
                    .transpose_rhs = true,
                } },
                .matmul_t0t1 => .{ .matmul = .{
                    .lhs = try self.lowerTensor(tensor.source0().?),
                    .rhs = try self.lowerTensor(tensor.source1().?),
                    .transpose_lhs = true,
                    .transpose_rhs = true,
                } },
                else => return error.UnsupportedOp,
            };
        }
    };
}

fn inferReductionAxes(alloc: Alloc, src: anytype, dst: anytype) ![]Axis {
    var buf = std.ArrayList(Axis){};
    defer buf.deinit(alloc);

    const ndims = @max(src.n_dims, dst.n_dims);
    for (0..ndims) |i| {
        const src_dim = src.ne[i];
        const dst_dim = dst.ne[i];
        if (src_dim != dst_dim) {
            std.debug.assert(dst_dim != 0);
            std.debug.assert(src_dim % dst_dim == 0);
            try buf.append(alloc, @intCast(i));
        }
    }
    return try alloc.dupe(Axis, buf.items);
}

fn canonicalizeExpr(src_graph: *const CanonicalGraph, dst_graph: *CanonicalGraph, value: CanonicalValue) !CanonicalExpr {
    return switch (value.expr) {
        .input, .constant, .gather, .scatter_add, .matmul, .scale => value.expr,
        .unary => |u| .{ .unary = .{ .op = u.op, .input = u.input } },
        .reduce => |r| .{ .reduce = .{ .op = r.op, .input = r.input, .axes = try dst_graph.dupeAxes(r.axes) } },
        .view => |v| canonicalizeView(src_graph, v),
        .binary => |b| canonicalizeBinary(src_graph, b),
    };
}

fn canonicalizeView(src_graph: *const CanonicalGraph, view: ViewSpec) CanonicalExpr {
    const input_value = src_graph.value(view.input);

    if (input_value.expr == .view) {
        const parent = input_value.expr.view;
        if (view.kind == .reshape and parent.kind == .reshape) {
            return .{ .view = .{
                .kind = .reshape,
                .input = parent.input,
                .shape = view.shape,
                .strides = view.strides,
            } };
        }
        if (view.kind == .broadcast and parent.kind == .broadcast) {
            return .{ .view = .{
                .kind = .broadcast,
                .input = parent.input,
                .shape = view.shape,
                .strides = view.strides,
            } };
        }
    }

    return .{ .view = view };
}

fn canonicalizeBinary(src_graph: *const CanonicalGraph, binary: BinarySpec) CanonicalExpr {
    if (binary.op == .mul) {
        const lhs = src_graph.value(binary.lhs);
        const rhs = src_graph.value(binary.rhs);

        if (isScalarBroadcastValue(src_graph, binary.rhs)) {
            return .{ .scale = .{ .input = binary.lhs, .scalar = scalarSourceValueId(src_graph, binary.rhs) } };
        }
        if (isScalarBroadcastValue(src_graph, binary.lhs)) {
            return .{ .scale = .{ .input = binary.rhs, .scalar = scalarSourceValueId(src_graph, binary.lhs) } };
        }

        _ = lhs;
        _ = rhs;
    }

    return .{ .binary = binary };
}

fn isScalarBroadcastValue(graph: *const CanonicalGraph, id: ValueId) bool {
    const value = graph.value(id);
    if (value.shape.nElems() == 1) return true;
    return switch (value.expr) {
        .view => |v| v.kind == .broadcast and graph.value(v.input).shape.nElems() == 1,
        else => false,
    };
}

fn scalarSourceValueId(graph: *const CanonicalGraph, id: ValueId) ValueId {
    const value = graph.value(id);
    if (value.shape.nElems() == 1) return id;
    return switch (value.expr) {
        .view => |v| v.input,
        else => unreachable,
    };
}

test "compiler - shape helpers" {
    const shape = Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(u8, 3), shape.ndims);
    try std.testing.expectEqual(@as(usize, 24), shape.nElems());

    const strides = Strides.contiguous(shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 6, 24, 24, 24, 24, 24 }, &strides.values);
}

test "compiler - canonical graph append" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input_id = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .input);
    const reduced_id = try graph.addValue(.f32, Shape.init(&.{ 1, 3 }), .{ .reduce = .{
        .op = .sum,
        .input = input_id,
        .axes = &.{0},
    } });

    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(input_id));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(reduced_id));
    try std.testing.expectEqual(@as(usize, 2), graph.values.items.len);
}

test "compiler - lower unary binary reduce graph" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{3});
    defer x.deinit();
    x.setData(&.{ 1, 2, 3 });

    const exp_x = x.exp();
    defer exp_x.deinit();
    const sum_x = exp_x.add(x);
    defer sum_x.deinit();
    const y = sum_x.sumAll();
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();

    const root_id = try lowering.lowerRoot(y);
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(root_id));
    try std.testing.expectEqual(@as(usize, 4), lowering.graph.values.items.len);
    try std.testing.expect(lowering.graph.values.items[0].expr == .input);
    try std.testing.expect(lowering.graph.values.items[1].expr == .unary);
    try std.testing.expect(lowering.graph.values.items[2].expr == .binary);
    try std.testing.expect(lowering.graph.values.items[3].expr == .reduce);
}

test "compiler - lower matmul and transpose" {
    const Tensor = tensorlib.Tensor(f32);

    var a = try Tensor.init(std.testing.allocator, &.{ 2, 3 });
    defer a.deinit();
    var b = try Tensor.init(std.testing.allocator, &.{ 2, 3 });
    defer b.deinit();

    const out = a.matMul(false, b, true);
    defer out.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();

    const root_id = try lowering.lowerRoot(out);
    const root = lowering.graph.values.items[@intFromEnum(root_id)];
    try std.testing.expect(root.expr == .matmul);
    try std.testing.expect(root.expr.matmul.transpose_rhs);
    try std.testing.expect(!root.expr.matmul.transpose_lhs);
}

test "compiler - canonicalize scale from mul broadcast scalar" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .input);
    const scalar = try graph.addValue(.f32, Shape.init(&.{1}), .constant);
    const scalar_bcast = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .{ .view = .{
        .kind = .broadcast,
        .input = scalar,
        .shape = Shape.init(&.{ 4, 3 }),
        .strides = null,
    } });
    _ = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .{ .binary = .{
        .op = .mul,
        .lhs = input,
        .rhs = scalar_bcast,
    } });

    var canon = try graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.values.items[3].expr == .scale);
    try std.testing.expectEqual(input, canon.values.items[3].expr.scale.input);
    try std.testing.expectEqual(scalar, canon.values.items[3].expr.scale.scalar);
}

test "compiler - canonicalize nested reshape views" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 2, 3 }), .input);
    const r1 = try graph.addValue(.f32, Shape.init(&.{ 6, 1 }), .{ .view = .{
        .kind = .reshape,
        .input = input,
        .shape = Shape.init(&.{ 6, 1 }),
        .strides = null,
    } });
    _ = try graph.addValue(.f32, Shape.init(&.{ 3, 2 }), .{ .view = .{
        .kind = .reshape,
        .input = r1,
        .shape = Shape.init(&.{ 3, 2 }),
        .strides = null,
    } });

    var canon = try graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.values.items[2].expr == .view);
    try std.testing.expectEqual(input, canon.values.items[2].expr.view.input);
    try std.testing.expectEqual(ViewKind.reshape, canon.values.items[2].expr.view.kind);
}

test "compiler - detect canonical softmax pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{3});
    defer x.deinit();
    x.setData(&.{ 1, 2, 3 });

    const max_t = x.max(&.{1});
    defer max_t.deinit();
    const rep_max = max_t.repeatLike(x);
    defer rep_max.deinit();
    const neg_rep = rep_max.neg();
    defer neg_rep.deinit();
    const shifted = x.add(neg_rep);
    defer shifted.deinit();
    const exps = shifted.exp();
    defer exps.deinit();
    const denom = exps.sum(&.{1});
    defer denom.deinit();
    const rep_denom = denom.repeatLike(exps);
    defer rep_denom.deinit();
    const recip = rep_denom.recip();
    defer recip.deinit();
    const y = exps.mul(recip);
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectSoftmax(@enumFromInt(canon.values.items.len - 1)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(spec.input));
    try std.testing.expectEqual(@as(u32, @intCast(canon.values.items.len - 1)), @intFromEnum(spec.output));
}

test "compiler - rewrite layer detects softmax pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{3});
    defer x.deinit();
    x.setData(&.{ 1, 2, 3 });

    const max_t = x.max(&.{1});
    defer max_t.deinit();
    const rep_max = max_t.repeatLike(x);
    defer rep_max.deinit();
    const neg_rep = rep_max.neg();
    defer neg_rep.deinit();
    const shifted = x.add(neg_rep);
    defer shifted.deinit();
    const exps = shifted.exp();
    defer exps.deinit();
    const denom = exps.sum(&.{1});
    defer denom.deinit();
    const rep_denom = denom.repeatLike(exps);
    defer rep_denom.deinit();
    const recip = rep_denom.recip();
    defer recip.deinit();
    const y = exps.mul(recip);
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    var patterns = try canon.detectPatterns(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer patterns.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), patterns.items.len);
    try std.testing.expect(patterns.items[0] == .softmax);
}

test "compiler - rewrite layer detects logSoftmax pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{3});
    defer x.deinit();
    x.setData(&.{ 1, 2, 3 });

    const max_t = x.max(&.{1});
    defer max_t.deinit();
    const rep_max = max_t.repeatLike(x);
    defer rep_max.deinit();
    const neg_rep = rep_max.neg();
    defer neg_rep.deinit();
    const shifted = x.add(neg_rep);
    defer shifted.deinit();
    const exps = shifted.exp();
    defer exps.deinit();
    const sum = exps.sum(&.{1});
    defer sum.deinit();
    const log_norm = sum.log();
    defer log_norm.deinit();
    const rep_log = log_norm.repeatLike(shifted);
    defer rep_log.deinit();
    const neg_rep_log = rep_log.neg();
    defer neg_rep_log.deinit();
    const y = shifted.add(neg_rep_log);
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    var patterns = try canon.detectPatterns(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer patterns.deinit(std.testing.allocator);

    var found = false;
    for (patterns.items) |p| {
        if (p == .log_softmax) {
            found = true;
        }
    }
    try std.testing.expect(found);

    var rewrites = try canon.applyRewritePasses(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer rewrites.deinit(std.testing.allocator);

    var found_rewrite = false;
    for (rewrites.items) |r| {
        if (r == .log_softmax) {
            found_rewrite = true;
        }
    }
    try std.testing.expect(found_rewrite);
}

test "compiler - rewrite layer detects crossEntropy pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var logits = try Tensor.init(std.testing.allocator, &.{ 3, 2 });
    defer logits.deinit();
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });

    var targets = try Tensor.init(std.testing.allocator, &.{2});
    defer targets.deinit();
    targets.setData(&.{ 0, 1 });

    const max_t = logits.max(&.{ 1, 2 });
    defer max_t.deinit();
    const rep_max = max_t.repeatLike(logits);
    defer rep_max.deinit();
    const neg_rep_max = rep_max.neg();
    defer neg_rep_max.deinit();
    const shifted = logits.add(neg_rep_max);
    defer shifted.deinit();
    const exps = shifted.exp();
    defer exps.deinit();
    const sum = exps.sum(&.{ 1, 2 });
    defer sum.deinit();
    const log_norm = sum.log();
    defer log_norm.deinit();
    const rep_log = log_norm.repeatLike(shifted);
    defer rep_log.deinit();
    const neg_rep_log = rep_log.neg();
    defer neg_rep_log.deinit();
    const log_probs = shifted.add(neg_rep_log);
    defer log_probs.deinit();
    const picked = log_probs.pickRows(targets);
    defer picked.deinit();
    const neg_picked = picked.neg();
    defer neg_picked.deinit();
    const ce_sum = neg_picked.sum(&.{1});
    defer ce_sum.deinit();
    const ce = ce_sum.scaleByVal(0.5);
    defer ce.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(ce);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.detectCrossEntropy(@enumFromInt(canon.values.items.len - 1)) != null);

    var rewrites = try canon.applyRewritePasses(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer rewrites.deinit(std.testing.allocator);

    var found_rewrite = false;
    for (rewrites.items) |r| {
        if (r == .cross_entropy) {
            found_rewrite = true;
        }
    }
    try std.testing.expect(found_rewrite);
}

test "compiler - rewrite layer detects layerNorm pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });

    const y = x.layerNorm(&.{ 1, 3 }, 1e-5);

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.detectLayerNorm(@enumFromInt(canon.values.items.len - 1)) != null);

    var rewrites = try canon.applyRewritePasses(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer rewrites.deinit(std.testing.allocator);

    var found_rewrite = false;
    for (rewrites.items) |r| {
        if (r == .layer_norm) {
            found_rewrite = true;
        }
    }
    try std.testing.expect(found_rewrite);
}

test "compiler - detects rmsNorm canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{ 4, 3 });
    defer x.deinit();
    x.setData(&.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });

    const y = x.rmsNorm(&.{ 1, 3 }, 1e-5);
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.detectRmsNorm(@enumFromInt(canon.values.items.len - 1)) == null);
}

test "compiler - rewrite passes detect softmax" {
    const Tensor = tensorlib.Tensor(f32);

    var x = try Tensor.init(std.testing.allocator, &.{3});
    defer x.deinit();
    x.setData(&.{ 1, 2, 3 });

    const max_t = x.max(&.{1});
    defer max_t.deinit();
    const rep_max = max_t.repeatLike(x);
    defer rep_max.deinit();
    const neg_rep = rep_max.neg();
    defer neg_rep.deinit();
    const shifted = x.add(neg_rep);
    defer shifted.deinit();
    const exps = shifted.exp();
    defer exps.deinit();
    const denom = exps.sum(&.{1});
    defer denom.deinit();
    const rep_denom = denom.repeatLike(exps);
    defer rep_denom.deinit();
    const recip = rep_denom.recip();
    defer recip.deinit();
    const y = exps.mul(recip);
    defer y.deinit();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    var rewrites = try canon.applyRewritePasses(@enumFromInt(canon.values.items.len - 1), std.testing.allocator);
    defer rewrites.deinit(std.testing.allocator);

    var found = false;
    for (rewrites.items) |r| {
        if (r == .softmax) {
            found = true;
        }
    }
    try std.testing.expect(found);
}

test "compiler - lower canonical graph to kernel plan" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .input);
    const scalar = try graph.addValue(.f32, Shape.init(&.{1}), .constant);
    _ = try graph.addValue(.f32, Shape.init(&.{ 4, 3 }), .{ .scale = .{ .input = input, .scalar = scalar } });

    var plan = try KernelPlan.lowerFromCanonical(std.testing.allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.values.items.len);
    try std.testing.expect(plan.values.items[0].expr == .zip);
    try std.testing.expectEqual(@as(usize, 1), plan.outputs.items.len);
}

test "compiler - pipeline compiles softmax graph end-to-end" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.softmax(&.{1});

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.canonical != null);
    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.rewrites.items.len >= 1);
    try std.testing.expectEqual(pipeline.rewrites.items.len, pipeline.kernel.?.annotations.items.len);

    var found = false;
    for (pipeline.rewrites.items) |r| {
        if (r == .softmax) found = true;
    }
    try std.testing.expect(found);
    try std.testing.expect(pipeline.kernel.?.annotations.items[0].rewrite == .softmax);
}

test "compiler - pipeline compiles logSoftmax graph end-to-end" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.logSoftmax(&.{1});

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.canonical != null);
    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.rewrites.items.len >= 1);
    try std.testing.expectEqual(pipeline.rewrites.items.len, pipeline.kernel.?.annotations.items.len);

    var found = false;
    for (pipeline.rewrites.items) |r| {
        if (r == .log_softmax) found = true;
    }
    try std.testing.expect(found);
    try std.testing.expect(pipeline.kernel.?.annotations.items[0].rewrite == .log_softmax);
}

test "compiler - pipeline compiles crossEntropy graph end-to-end" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var logits = try Tensor.init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });

    var targets = try Tensor.init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const y = losslib.crossEntropy(f32, logits, targets);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.canonical != null);
    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.rewrites.items.len >= 1);
    try std.testing.expectEqual(pipeline.rewrites.items.len, pipeline.kernel.?.annotations.items.len);

    var found = false;
    for (pipeline.rewrites.items) |r| {
        if (r == .cross_entropy) found = true;
    }
    try std.testing.expect(found);
    try std.testing.expect(pipeline.kernel.?.annotations.items[0].rewrite == .cross_entropy);
}

test "compiler - pipeline compiles layerNorm graph end-to-end" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });

    const y = x.layerNorm(&.{ 1, 3 }, 1e-5);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.canonical != null);
    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.rewrites.items.len >= 1);
    try std.testing.expectEqual(pipeline.rewrites.items.len, pipeline.kernel.?.annotations.items.len);

    var found = false;
    for (pipeline.rewrites.items) |r| {
        if (r == .layer_norm) found = true;
    }
    try std.testing.expect(found);
    try std.testing.expect(pipeline.kernel.?.annotations.items[0].rewrite == .layer_norm);
}

test "compiler - kernel plan builds explicit fusion patterns" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.softmax(&.{1});

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expectEqual(@as(usize, 1), pipeline.kernel.?.patterns.items.len);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .softmax);
}

test "compiler - kernel patterns prefer outer fused region" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var logits = try Tensor.init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });

    var targets = try Tensor.init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const y = losslib.crossEntropy(f32, logits, targets);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expectEqual(@as(usize, 1), pipeline.kernel.?.patterns.items.len);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .cross_entropy);
}
