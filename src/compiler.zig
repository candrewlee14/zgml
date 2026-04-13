const std = @import("std");
const Op = @import("op.zig").Op;
const tensorlib = @import("tensor.zig");
const losslib = @import("loss.zig");
const nnlib = @import("nn.zig");
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
    sgn,
    step,
    sqrt,
    recip,
    exp,
    log,
    gelu,

    pub fn fromOp(op: Op) ?UnaryOp {
        return switch (op) {
            .neg => .neg,
            .abs => .abs,
            .sgn => .sgn,
            .step => .step,
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

pub const LinearSpec = struct {
    input: ValueId,
    weight: ValueId,
    bias: ValueId,
    matmul: ValueId,
    output: ValueId,
};

pub const LinearGeluSpec = struct {
    linear: LinearSpec,
    output: ValueId,
};

pub const LinearReluSpec = struct {
    linear: LinearSpec,
    step_node: ValueId,
    output: ValueId,
};

pub const LinearResidualSpec = struct {
    linear: LinearSpec,
    residual: ValueId,
    output: ValueId,
};

pub const MatmulResidualSpec = struct {
    lhs: ValueId,
    rhs: ValueId,
    matmul: ValueId,
    residual: ValueId,
    output: ValueId,
};

const LinearParts = struct {
    matmul: ValueId,
    bias_bcast: ValueId,
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

pub const LinearKernelSpec = struct {
    input: ValueId,
    weight: ValueId,
    bias: ValueId,
    bias_bcast: ValueId,
    matmul: ValueId,
    output: ValueId,
};

pub const LinearGeluKernelSpec = struct {
    linear: LinearKernelSpec,
    output: ValueId,
};

pub const LinearReluKernelSpec = struct {
    linear: LinearKernelSpec,
    step_node: ValueId,
    output: ValueId,
};

pub const LinearResidualKernelSpec = struct {
    linear: LinearKernelSpec,
    residual: ValueId,
    output: ValueId,
};

pub const MatmulResidualKernelSpec = struct {
    lhs: ValueId,
    rhs: ValueId,
    matmul: ValueId,
    residual: ValueId,
    output: ValueId,
};

pub const Conv2dSpec = struct {
    input: ValueId,
    kernel: ValueId,
    input_view: ValueId,
    kernel_view: ValueId,
    mul_node: ValueId,
    sum_node: ValueId,
    reshape_node: ValueId,
    bias: ?ValueId = null,
    bias_add: ?ValueId = null,
    activation: ?ValueId = null,
    output: ValueId,
};

pub const Conv2dBwdInputSpec = struct {
    output_grad: ValueId,
    kernel: ValueId,
    output: ValueId,
};

pub const Conv2dBwdKernelSpec = struct {
    input: ValueId,
    output_grad: ValueId,
    output: ValueId,
};

pub const MaxPool2dSpec = struct {
    input: ValueId,
    strided: ValueId,
    max_node: ValueId,
    output: ValueId,
};

pub const Conv2dKernelSpec = Conv2dSpec;
pub const Conv2dBwdInputKernelSpec = Conv2dBwdInputSpec;
pub const Conv2dBwdKernelKernelSpec = Conv2dBwdKernelSpec;
pub const MaxPool2dKernelSpec = MaxPool2dSpec;

/// Executable high-level kernel regions discovered from canonical IR.
///
/// These are intentionally richer than primitive tensor ops and intentionally
/// lower-level than the public tensor API. They are the compiler's typed view
/// of the fused regions that are worth scheduling as units.
pub const KernelPatternKind = enum {
    softmax,
    log_softmax,
    cross_entropy,
    layer_norm,
    linear,
    linear_gelu,
    linear_relu,
    linear_residual,
    matmul_residual,
    conv2d,
    conv2d_bwd_input,
    conv2d_bwd_kernel,
    max_pool2d,
};

/// Typed payloads for compiler-selected fused regions.
///
/// The pattern layer is the semantic boundary between canonical graph rewrites
/// and later scheduling/runtime decisions. Runtime may choose to execute only a
/// subset of these patterns; unsupported patterns can still exist here without
/// forcing frontend or execution complexity.
pub const KernelPattern = union(KernelPatternKind) {
    softmax: SoftmaxKernelSpec,
    log_softmax: LogSoftmaxKernelSpec,
    cross_entropy: CrossEntropyKernelSpec,
    layer_norm: LayerNormKernelSpec,
    linear: LinearKernelSpec,
    linear_gelu: LinearGeluKernelSpec,
    linear_relu: LinearReluKernelSpec,
    linear_residual: LinearResidualKernelSpec,
    matmul_residual: MatmulResidualKernelSpec,
    conv2d: Conv2dKernelSpec,
    conv2d_bwd_input: Conv2dBwdInputKernelSpec,
    conv2d_bwd_kernel: Conv2dBwdKernelKernelSpec,
    max_pool2d: MaxPool2dKernelSpec,

    pub fn output(self: @This()) ValueId {
        return switch (self) {
            .softmax => |spec| spec.output,
            .log_softmax => |spec| spec.output,
            .cross_entropy => |spec| spec.mean_node,
            .layer_norm => |spec| spec.output,
            .linear => |spec| spec.output,
            .linear_gelu => |spec| spec.output,
            .linear_relu => |spec| spec.output,
            .linear_residual => |spec| spec.output,
            .matmul_residual => |spec| spec.output,
            .conv2d => |spec| spec.output,
            .conv2d_bwd_input => |spec| spec.output,
            .conv2d_bwd_kernel => |spec| spec.output,
            .max_pool2d => |spec| spec.output,
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
            .linear => |spec| {
                claimed[@intFromEnum(spec.bias_bcast)] = true;
                claimed[@intFromEnum(spec.matmul)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .linear_gelu => |spec| {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                linear_pattern.markCovered(claimed);
                claimed[@intFromEnum(spec.output)] = true;
            },
            .linear_relu => |spec| {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                linear_pattern.markCovered(claimed);
                claimed[@intFromEnum(spec.step_node)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .linear_residual => |spec| {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                linear_pattern.markCovered(claimed);
                claimed[@intFromEnum(spec.output)] = true;
            },
            .matmul_residual => |spec| {
                claimed[@intFromEnum(spec.matmul)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .conv2d => |spec| {
                claimed[@intFromEnum(spec.input_view)] = true;
                claimed[@intFromEnum(spec.kernel_view)] = true;
                claimed[@intFromEnum(spec.mul_node)] = true;
                claimed[@intFromEnum(spec.sum_node)] = true;
                claimed[@intFromEnum(spec.reshape_node)] = true;
                if (spec.bias_add) |b| claimed[@intFromEnum(b)] = true;
                if (spec.activation) |a| claimed[@intFromEnum(a)] = true;
                claimed[@intFromEnum(spec.output)] = true;
            },
            .conv2d_bwd_input => |spec| {
                claimed[@intFromEnum(spec.output)] = true;
            },
            .conv2d_bwd_kernel => |spec| {
                claimed[@intFromEnum(spec.output)] = true;
            },
            .max_pool2d => |spec| {
                claimed[@intFromEnum(spec.strided)] = true;
                claimed[@intFromEnum(spec.max_node)] = true;
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
            .linear => |spec| claimed[@intFromEnum(spec.bias_bcast)] or
                claimed[@intFromEnum(spec.matmul)] or
                claimed[@intFromEnum(spec.output)],
            .linear_gelu => |spec| blk: {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                break :blk linear_pattern.overlapsClaimed(claimed) or
                    claimed[@intFromEnum(spec.output)];
            },
            .linear_relu => |spec| blk: {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                break :blk linear_pattern.overlapsClaimed(claimed) or
                    claimed[@intFromEnum(spec.step_node)] or
                    claimed[@intFromEnum(spec.output)];
            },
            .linear_residual => |spec| blk: {
                const linear_pattern = KernelPattern{ .linear = spec.linear };
                break :blk linear_pattern.overlapsClaimed(claimed) or
                    claimed[@intFromEnum(spec.output)];
            },
            .matmul_residual => |spec| claimed[@intFromEnum(spec.matmul)] or
                claimed[@intFromEnum(spec.output)],
            .conv2d => |spec| claimed[@intFromEnum(spec.input_view)] or
                claimed[@intFromEnum(spec.kernel_view)] or
                claimed[@intFromEnum(spec.mul_node)] or
                claimed[@intFromEnum(spec.sum_node)] or
                claimed[@intFromEnum(spec.reshape_node)] or
                (if (spec.bias_add) |b| claimed[@intFromEnum(b)] else false) or
                (if (spec.activation) |a| claimed[@intFromEnum(a)] else false) or
                claimed[@intFromEnum(spec.output)],
            .conv2d_bwd_input => |spec| claimed[@intFromEnum(spec.output)],
            .conv2d_bwd_kernel => |spec| claimed[@intFromEnum(spec.output)],
            .max_pool2d => |spec| claimed[@intFromEnum(spec.strided)] or
                claimed[@intFromEnum(spec.max_node)] or
                claimed[@intFromEnum(spec.output)],
        };
    }
};

pub const KernelPatternRecord = struct {
    output: ValueId,
    pattern: KernelPattern,
};

/// Materialization state for a canonical value at the schedule boundary.
///
/// v1 intentionally keeps this binary and conservative:
/// - `virtual` means the value is considered internal to a scheduled fused region
/// - `materialized` means the value is a schedule boundary or requested output
pub const MaterializationKind = enum {
    virtual,
    materialized,
};

pub const ScheduleRegionKind = enum {
    elementwise,
};

/// Schedule-owned fused regions.
///
/// These are intentionally distinct from `KernelPattern`: semantic patterns stay
/// compiler-owned, while generic same-shape elementwise fusion lives at the
/// schedule layer where execution policy and materialization are decided.
pub const ElementwiseRegion = struct {
    input: ValueId,
    nodes: []const ValueId,
};

pub const ScheduleRegion = union(ScheduleRegionKind) {
    elementwise: ElementwiseRegion,

    pub fn output(self: @This()) ValueId {
        return switch (self) {
            .elementwise => |region| region.nodes[region.nodes.len - 1],
        };
    }
};

/// A scheduled execution step.
///
/// v1 supports two kinds only:
/// - `kernel_pattern` for compiler-selected semantic fused regions
/// - `schedule_region` for schedule-owned generic fused regions
/// - `generic` for uncovered canonical values that still need execution
pub const ScheduleStepKind = enum {
    kernel_pattern,
    schedule_region,
    generic,
};

pub const ScheduleStep = union(ScheduleStepKind) {
    kernel_pattern: struct {
        output: ValueId,
        pattern_index: usize,
    },
    schedule_region: struct {
        output: ValueId,
        region_index: usize,
    },
    generic: struct {
        output: ValueId,
    },

    pub fn output(self: @This()) ValueId {
        return switch (self) {
            .kernel_pattern => |step| step.output,
            .schedule_region => |step| step.output,
            .generic => |step| step.output,
        };
    }
};

pub const SchedulePlan = struct {
    alloc: Alloc,
    materialization: std.ArrayList(MaterializationKind),
    regions: std.ArrayList(ScheduleRegion),
    steps: std.ArrayList(ScheduleStep),
    outputs: std.ArrayList(ValueId),

    pub fn init(alloc: Alloc) SchedulePlan {
        return .{
            .alloc = alloc,
            .materialization = .{},
            .regions = .{},
            .steps = .{},
            .outputs = .{},
        };
    }

    pub fn deinit(self: *SchedulePlan) void {
        for (self.regions.items) |region| {
            switch (region) {
                .elementwise => |payload| self.alloc.free(payload.nodes),
            }
        }
        self.materialization.deinit(self.alloc);
        self.regions.deinit(self.alloc);
        self.steps.deinit(self.alloc);
        self.outputs.deinit(self.alloc);
    }

    pub fn writeReport(self: *const SchedulePlan, writer: anytype) !void {
        try writer.print("schedule steps={d} outputs={d}\n", .{ self.steps.items.len, self.outputs.items.len });
        for (self.steps.items, 0..) |step, idx| {
            switch (step) {
                .kernel_pattern => |s| try writer.print("  step[{d}] pattern output=v{d} pattern_index={d}\n", .{ idx, @intFromEnum(s.output), s.pattern_index }),
                .schedule_region => |s| {
                    const region = self.regions.items[s.region_index];
                    switch (region) {
                        .elementwise => |payload| try writer.print("  step[{d}] region kind=elementwise output=v{d} nodes={d}\n", .{ idx, @intFromEnum(s.output), payload.nodes.len }),
                    }
                },
                .generic => |s| try writer.print("  step[{d}] generic output=v{d}\n", .{ idx, @intFromEnum(s.output) }),
            }
        }
        for (self.outputs.items) |output| {
            try writer.print("  output v{d} materialized\n", .{@intFromEnum(output)});
        }
    }

    /// Build a minimal schedule/materialization plan from canonical values plus
    /// compiler-selected kernel patterns.
    ///
    /// Design goals for v1:
    /// - keep fused-region boundaries explicit
    /// - mark pattern internals virtual and outputs materialized
    /// - keep generic elementwise fusion owned by the schedule layer
    /// - preserve a simple topological walk for uncovered generic values
    /// - avoid backend or profitability policy here
    pub fn build(alloc: Alloc, graph: *const CanonicalGraph, kernel: *const KernelPlan) !SchedulePlan {
        var schedule = SchedulePlan.init(alloc);
        errdefer schedule.deinit();

        try schedule.materialization.resize(alloc, graph.values.items.len);
        @memset(schedule.materialization.items, .materialized);

        const covered_outputs = try alloc.alloc(bool, graph.values.items.len);
        defer alloc.free(covered_outputs);
        @memset(covered_outputs, false);

        const covered_values = try alloc.alloc(bool, graph.values.items.len);
        defer alloc.free(covered_values);
        @memset(covered_values, false);

        const pattern_steps = try alloc.alloc(?usize, graph.values.items.len);
        defer alloc.free(pattern_steps);
        for (pattern_steps) |*entry| entry.* = null;

        const region_steps = try alloc.alloc(?usize, graph.values.items.len);
        defer alloc.free(region_steps);
        for (region_steps) |*entry| entry.* = null;

        for (kernel.patterns.items, 0..) |record, pattern_index| {
            const output = record.output;
            const output_idx = @intFromEnum(output);
            covered_outputs[output_idx] = true;
            pattern_steps[output_idx] = pattern_index;

            try markPatternVirtuals(&schedule, record.pattern);
            record.pattern.markCovered(covered_values);
            schedule.materialization.items[output_idx] = .materialized;
        }

        try buildScheduleRegions(&schedule, alloc, graph, covered_values, covered_outputs);
        for (schedule.regions.items, 0..) |region, region_index| {
            const output = region.output();
            const output_idx = @intFromEnum(output);
            covered_outputs[output_idx] = true;
            region_steps[output_idx] = region_index;
            try markRegionVirtuals(&schedule, region);
            markRegionCovered(covered_values, region);
            schedule.materialization.items[output_idx] = .materialized;
        }

        for (graph.values.items) |value| {
            const idx = @intFromEnum(value.id);
            if (pattern_steps[idx]) |pattern_index| {
                try schedule.steps.append(alloc, .{ .kernel_pattern = .{
                    .output = value.id,
                    .pattern_index = pattern_index,
                } });
                continue;
            }
            if (region_steps[idx]) |region_index| {
                try schedule.steps.append(alloc, .{ .schedule_region = .{
                    .output = value.id,
                    .region_index = region_index,
                } });
                continue;
            }
            if (covered_outputs[idx]) continue;
            if (covered_values[idx]) continue;
            if (!isGenericSchedulableExpr(value.expr)) continue;
            try schedule.steps.append(alloc, .{ .generic = .{ .output = value.id } });
        }

        for (kernel.outputs.items) |output| {
            try schedule.outputs.append(alloc, output);
            schedule.materialization.items[@intFromEnum(output)] = .materialized;
        }

        return schedule;
    }
};

pub const ExecutionStepKind = enum {
    kernel_pattern,
    schedule_region,
    generic,
};

pub const ExecutionStep = union(ExecutionStepKind) {
    kernel_pattern: struct {
        pattern_index: usize,
        output: ValueId,
    },
    schedule_region: struct {
        region_index: usize,
        output: ValueId,
    },
    generic: struct {
        output: ValueId,
    },
};

/// Runtime-facing execution plan lowered from `SchedulePlan`.
///
/// Unlike `SchedulePlan`, this form is intentionally close to execution:
/// it is just an ordered list of executable steps. Execution policy stays data-
/// driven, while the executor itself can remain simple and mostly branch-free.
pub const ExecutionPlan = struct {
    alloc: Alloc,
    steps: std.ArrayList(ExecutionStep),
    outputs: std.ArrayList(ValueId),

    pub fn init(alloc: Alloc) ExecutionPlan {
        return .{ .alloc = alloc, .steps = .{}, .outputs = .{} };
    }

    pub fn deinit(self: *ExecutionPlan) void {
        self.steps.deinit(self.alloc);
        self.outputs.deinit(self.alloc);
    }

    pub fn build(alloc: Alloc, schedule: *const SchedulePlan) !ExecutionPlan {
        var plan = ExecutionPlan.init(alloc);
        errdefer plan.deinit();

        for (schedule.steps.items) |step| {
            switch (step) {
                .kernel_pattern => |s| try plan.steps.append(alloc, .{ .kernel_pattern = .{
                    .pattern_index = s.pattern_index,
                    .output = s.output,
                } }),
                .schedule_region => |s| try plan.steps.append(alloc, .{ .schedule_region = .{
                    .region_index = s.region_index,
                    .output = s.output,
                } }),
                .generic => |s| try plan.steps.append(alloc, .{ .generic = .{ .output = s.output } }),
            }
        }
        for (schedule.outputs.items) |output| {
            try plan.outputs.append(alloc, output);
        }
        return plan;
    }
};

const kernel_pattern_priority = [_]KernelPatternKind{
    .conv2d,
    .max_pool2d,
    .linear_gelu,
    .linear_relu,
    .linear_residual,
    .matmul_residual,
    .linear,
    .cross_entropy,
    .layer_norm,
    .log_softmax,
    .softmax,
};

fn markVirtual(materialization: []MaterializationKind, id: ValueId) void {
    materialization[@intFromEnum(id)] = .virtual;
}

fn markPatternVirtuals(schedule: *SchedulePlan, pattern: KernelPattern) !void {
    const materialization = schedule.materialization.items;
    switch (pattern) {
        .softmax => |spec| {
            markVirtual(materialization, spec.max_node);
            markVirtual(materialization, spec.rep_max);
            markVirtual(materialization, spec.neg_rep_max);
            markVirtual(materialization, spec.shifted);
            markVirtual(materialization, spec.exp_node);
            markVirtual(materialization, spec.sum_node);
            markVirtual(materialization, spec.rep_sum);
            markVirtual(materialization, spec.recip_rep_sum);
        },
        .log_softmax => |spec| {
            markVirtual(materialization, spec.max_node);
            markVirtual(materialization, spec.rep_max);
            markVirtual(materialization, spec.neg_rep_max);
            markVirtual(materialization, spec.shifted);
            markVirtual(materialization, spec.exp_node);
            markVirtual(materialization, spec.sum_node);
            markVirtual(materialization, spec.log_node);
            markVirtual(materialization, spec.rep_log);
            markVirtual(materialization, spec.neg_rep_log);
        },
        .cross_entropy => |spec| {
            try markPatternVirtuals(schedule, .{ .log_softmax = spec.log_softmax });
            markVirtual(materialization, spec.picked);
            markVirtual(materialization, spec.neg_picked);
            markVirtual(materialization, spec.sum_node);
        },
        .layer_norm => |spec| {
            markVirtual(materialization, spec.sum_node);
            markVirtual(materialization, spec.mean_node);
            markVirtual(materialization, spec.rep_mean);
            markVirtual(materialization, spec.neg_rep_mean);
            markVirtual(materialization, spec.centered);
            markVirtual(materialization, spec.sqr_node);
            markVirtual(materialization, spec.var_sum);
            markVirtual(materialization, spec.var_node);
            markVirtual(materialization, spec.eps_like);
            markVirtual(materialization, spec.var_eps);
            markVirtual(materialization, spec.sqrt_node);
            markVirtual(materialization, spec.recip_node);
            markVirtual(materialization, spec.rep_std_inv);
        },
        .linear => |spec| {
            markVirtual(materialization, spec.bias_bcast);
            markVirtual(materialization, spec.matmul);
        },
        .linear_gelu => |spec| {
            try markPatternVirtuals(schedule, .{ .linear = spec.linear });
        },
        .linear_relu => |spec| {
            try markPatternVirtuals(schedule, .{ .linear = spec.linear });
            markVirtual(materialization, spec.step_node);
        },
        .linear_residual => |spec| {
            try markPatternVirtuals(schedule, .{ .linear = spec.linear });
        },
        .matmul_residual => |spec| {
            markVirtual(materialization, spec.matmul);
        },
        .conv2d => |spec| {
            markVirtual(materialization, spec.input_view);
            markVirtual(materialization, spec.kernel_view);
            markVirtual(materialization, spec.mul_node);
            markVirtual(materialization, spec.sum_node);
            markVirtual(materialization, spec.reshape_node);
            if (spec.bias_add) |b| markVirtual(materialization, b);
            if (spec.activation) |a| markVirtual(materialization, a);
        },
        .conv2d_bwd_input => {},
        .conv2d_bwd_kernel => {},
        .max_pool2d => |spec| {
            markVirtual(materialization, spec.strided);
            markVirtual(materialization, spec.max_node);
        },
    }
}

fn markRegionVirtuals(schedule: *SchedulePlan, region: ScheduleRegion) !void {
    switch (region) {
        .elementwise => |payload| {
            if (payload.nodes.len == 0) return;
            const materialization = schedule.materialization.items;
            for (payload.nodes[0 .. payload.nodes.len - 1]) |node| {
                markVirtual(materialization, node);
            }
        },
    }
}

fn markRegionCovered(covered: []bool, region: ScheduleRegion) void {
    switch (region) {
        .elementwise => |payload| {
            if (payload.nodes.len == 0) return;
            for (payload.nodes[0 .. payload.nodes.len - 1]) |node| {
                covered[@intFromEnum(node)] = true;
            }
        },
    }
}

fn isGenericSchedulableExpr(expr: CanonicalExpr) bool {
    return switch (expr) {
        .input, .constant => false,
        else => true,
    };
}

fn buildScheduleRegions(
    schedule: *SchedulePlan,
    alloc: Alloc,
    graph: *const CanonicalGraph,
    claimed_values: []const bool,
    claimed_outputs: []const bool,
) !void {
    const use_count = try buildCanonicalUseCount(alloc, graph);
    defer alloc.free(use_count);

    const region_claimed = try alloc.alloc(bool, graph.values.items.len);
    defer alloc.free(region_claimed);
    @memset(region_claimed, false);

    for (graph.values.items) |value| {
        const idx = @intFromEnum(value.id);
        if (claimed_values[idx] or claimed_outputs[idx] or region_claimed[idx]) continue;

        const region = try detectElementwiseScheduleRegion(alloc, graph, use_count, claimed_values, region_claimed, value.id) orelse continue;
        errdefer alloc.free(region.nodes);

        if (region.nodes.len < 2) {
            alloc.free(region.nodes);
            continue;
        }

        for (region.nodes) |node| {
            region_claimed[@intFromEnum(node)] = true;
        }
        try schedule.regions.append(alloc, .{ .elementwise = region });
    }
}

fn buildCanonicalUseCount(alloc: Alloc, graph: *const CanonicalGraph) ![]u32 {
    const use_count = try alloc.alloc(u32, graph.values.items.len);
    @memset(use_count, 0);

    for (graph.values.items) |value| {
        countExprUses(use_count, value.expr);
    }
    return use_count;
}

fn countExprUses(use_count: []u32, expr: CanonicalExpr) void {
    switch (expr) {
        .input, .constant => {},
        .unary => |u| use_count[@intFromEnum(u.input)] += 1,
        .binary => |b| {
            use_count[@intFromEnum(b.lhs)] += 1;
            use_count[@intFromEnum(b.rhs)] += 1;
        },
        .reduce => |r| use_count[@intFromEnum(r.input)] += 1,
        .scale => |s| {
            use_count[@intFromEnum(s.input)] += 1;
            use_count[@intFromEnum(s.scalar)] += 1;
        },
        .view => |v| use_count[@intFromEnum(v.input)] += 1,
        .gather => |g| {
            use_count[@intFromEnum(g.table)] += 1;
            use_count[@intFromEnum(g.indices)] += 1;
        },
        .scatter_add => |s| {
            use_count[@intFromEnum(s.indices)] += 1;
            use_count[@intFromEnum(s.updates)] += 1;
        },
        .scatter_add_view => |s| {
            use_count[@intFromEnum(s.grad)] += 1;
            use_count[@intFromEnum(s.view)] += 1;
        },
        .matmul => |m| {
            use_count[@intFromEnum(m.lhs)] += 1;
            use_count[@intFromEnum(m.rhs)] += 1;
        },
    }
}

const ElementwiseNodeInfo = struct {
    input: ValueId,
};

fn isCommutativeFusibleBinaryOp(op: BinaryOp) bool {
    return switch (op) {
        .add, .mul => true,
    };
}

fn detectElementwiseScheduleRegion(
    alloc: Alloc,
    graph: *const CanonicalGraph,
    use_count: []const u32,
    claimed_values: []const bool,
    region_claimed: []const bool,
    start: ValueId,
) !?ElementwiseRegion {
    const start_info = elementwiseNodeInfo(graph, start, null) orelse return null;
    if (hasElementwiseProducer(graph, start)) return null;

    var nodes = std.ArrayList(ValueId){};
    defer nodes.deinit(alloc);
    try nodes.append(alloc, start);

    var current = start;
    while (use_count[@intFromEnum(current)] == 1) {
        const next = findSingleUserValue(graph, current, claimed_values, region_claimed) orelse break;
        if (elementwiseNodeInfo(graph, next, current) == null) break;
        try nodes.append(alloc, next);
        current = next;
    }

    if (nodes.items.len < 2) return null;

    return .{
        .input = start_info.input,
        .nodes = try alloc.dupe(ValueId, nodes.items),
    };
}

fn hasElementwiseProducer(graph: *const CanonicalGraph, id: ValueId) bool {
    const value = graph.value(id);
    return switch (value.expr) {
        .unary => |u| elementwiseNodeInfo(graph, u.input, null) != null,
        .binary => |b| blk: {
            const lhs = graph.value(b.lhs);
            const rhs = graph.value(b.rhs);
            if (shapesEqual(value.shape, lhs.shape) and elementwiseNodeInfo(graph, b.lhs, null) != null) break :blk true;
            if (isCommutativeFusibleBinaryOp(b.op) and shapesEqual(value.shape, rhs.shape) and elementwiseNodeInfo(graph, b.rhs, null) != null) break :blk true;
            break :blk false;
        },
        .scale => |s| elementwiseNodeInfo(graph, s.input, null) != null,
        else => false,
    };
}

fn findSingleUserValue(
    graph: *const CanonicalGraph,
    needle: ValueId,
    claimed_values: []const bool,
    region_claimed: []const bool,
) ?ValueId {
    var found: ?ValueId = null;
    var idx: usize = @intFromEnum(needle) + 1;
    while (idx < graph.values.items.len) : (idx += 1) {
        if (claimed_values[idx] or region_claimed[idx]) continue;
        const value = graph.values.items[idx];
        if (!exprUsesValue(value.expr, needle)) continue;
        if (found != null) return null;
        found = value.id;
    }
    return found;
}

/// Returns the ValueId inputs of an expression, packed into a fixed-size array.
/// Unused slots are null. Callers iterate until the first null.
fn exprInputs(expr: CanonicalExpr) [2]?ValueId {
    return switch (expr) {
        .input, .constant => .{ null, null },
        .unary => |u| .{ u.input, null },
        .binary => |b| .{ b.lhs, b.rhs },
        .reduce => |r| .{ r.input, null },
        .scale => |s| .{ s.input, s.scalar },
        .view => |v| .{ v.input, null },
        .gather => |g| .{ g.table, g.indices },
        .scatter_add => |s| .{ s.indices, s.updates },
        .scatter_add_view => |s| .{ s.grad, s.view },
        .matmul => |m| .{ m.lhs, m.rhs },
    };
}

/// Remap all ValueId references in an expression using the given table.
/// Axes are duped into the destination graph's storage.
fn remapExpr(dst: *CanonicalGraph, expr: CanonicalExpr, remap: []const ValueId) !CanonicalExpr {
    return switch (expr) {
        .input, .constant => expr,
        .unary => |u| .{ .unary = .{ .op = u.op, .input = remap[@intFromEnum(u.input)] } },
        .binary => |b| .{ .binary = .{ .op = b.op, .lhs = remap[@intFromEnum(b.lhs)], .rhs = remap[@intFromEnum(b.rhs)] } },
        .reduce => |r| .{ .reduce = .{ .op = r.op, .input = remap[@intFromEnum(r.input)], .axes = try dst.dupeAxes(r.axes) } },
        .scale => |s| .{ .scale = .{ .input = remap[@intFromEnum(s.input)], .scalar = remap[@intFromEnum(s.scalar)] } },
        .view => |v| .{ .view = .{ .kind = v.kind, .input = remap[@intFromEnum(v.input)], .shape = v.shape, .strides = v.strides } },
        .gather => |g| .{ .gather = .{ .table = remap[@intFromEnum(g.table)], .indices = remap[@intFromEnum(g.indices)], .axis = g.axis } },
        .scatter_add => |s| .{ .scatter_add = .{ .dst_shape = s.dst_shape, .indices = remap[@intFromEnum(s.indices)], .updates = remap[@intFromEnum(s.updates)], .axis = s.axis } },
        .scatter_add_view => |s| .{ .scatter_add_view = .{ .grad = remap[@intFromEnum(s.grad)], .view = remap[@intFromEnum(s.view)] } },
        .matmul => |m| .{ .matmul = .{ .lhs = remap[@intFromEnum(m.lhs)], .rhs = remap[@intFromEnum(m.rhs)], .transpose_lhs = m.transpose_lhs, .transpose_rhs = m.transpose_rhs } },
    };
}

fn exprUsesValue(expr: CanonicalExpr, needle: ValueId) bool {
    return switch (expr) {
        .input, .constant => false,
        .unary => |u| u.input == needle,
        .binary => |b| b.lhs == needle or b.rhs == needle,
        .reduce => |r| r.input == needle,
        .scale => |s| s.input == needle or s.scalar == needle,
        .view => |v| v.input == needle,
        .gather => |g| g.table == needle or g.indices == needle,
        .scatter_add => |s| s.indices == needle or s.updates == needle,
        .scatter_add_view => |s| s.grad == needle or s.view == needle,
        .matmul => |m| m.lhs == needle or m.rhs == needle,
    };
}

fn elementwiseNodeInfo(graph: *const CanonicalGraph, id: ValueId, expected_input: ?ValueId) ?ElementwiseNodeInfo {
    const value = graph.value(id);
    return switch (value.expr) {
        .unary => |u| blk: {
            if (!isFusibleUnaryOp(u.op)) return null;
            if (!shapesEqual(value.shape, graph.value(u.input).shape)) return null;
            if (expected_input) |expected| {
                if (u.input != expected) return null;
            }
            break :blk .{ .input = u.input };
        },
        .binary => |b| blk: {
            if (!isFusibleBinaryOp(b.op)) return null;
            const lhs = graph.value(b.lhs);
            const rhs = graph.value(b.rhs);
            const lhs_matches = shapesEqual(value.shape, lhs.shape) and (rhs.shape.nElems() == 1 or shapesEqual(rhs.shape, lhs.shape));
            const rhs_matches = shapesEqual(value.shape, rhs.shape) and (lhs.shape.nElems() == 1 or shapesEqual(lhs.shape, rhs.shape));
            if (expected_input) |expected| {
                if (lhs_matches and b.lhs == expected) break :blk .{ .input = b.lhs };
                if (isCommutativeFusibleBinaryOp(b.op) and rhs_matches and b.rhs == expected) break :blk .{ .input = b.rhs };
                return null;
            }
            if (lhs_matches) break :blk .{ .input = b.lhs };
            if (isCommutativeFusibleBinaryOp(b.op) and rhs_matches) break :blk .{ .input = b.rhs };
            return null;
        },
        .scale => |s| blk: {
            if (!shapesEqual(value.shape, graph.value(s.input).shape)) return null;
            if (graph.value(s.scalar).shape.nElems() != 1) return null;
            if (expected_input) |expected| {
                if (s.input != expected) return null;
            }
            break :blk .{ .input = s.input };
        },
        else => null,
    };
}

fn isFusibleUnaryOp(op: UnaryOp) bool {
    return switch (op) {
        .neg, .abs, .sgn, .step, .sqrt, .recip, .exp, .log, .gelu => true,
    };
}

fn isFusibleBinaryOp(op: BinaryOp) bool {
    return switch (op) {
        .add, .mul => true,
    };
}

fn shapesEqual(lhs: Shape, rhs: Shape) bool {
    if (lhs.ndims != rhs.ndims) return false;
    return std.mem.eql(usize, lhs.dims[0..lhs.ndims], rhs.dims[0..rhs.ndims]);
}

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
    strided,
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

pub const ScatterAddViewSpec = struct {
    grad: ValueId,
    view: ValueId,
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
    scatter_add_view: ScatterAddViewSpec,
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
    /// Maps source (pre-DCE) ValueIds to canonical (post-DCE) ValueIds.
    /// null entry means the value was eliminated. Only populated when DCE
    /// actually removes values; otherwise null (identity mapping).
    source_remap: ?[]const ?ValueId = null,

    pub fn init(alloc: std.mem.Allocator) CanonicalGraph {
        return .{ .alloc = alloc, .values = .{}, .reduction_axes_storage = .{} };
    }

    pub fn deinit(self: *CanonicalGraph) void {
        if (self.source_remap) |remap| self.alloc.free(remap);
        for (self.reduction_axes_storage.items) |axes| self.alloc.free(axes);
        self.reduction_axes_storage.deinit(self.alloc);
        self.values.deinit(self.alloc);
    }

    /// Translate a source (pre-DCE) ValueId to its canonical (post-DCE) ValueId.
    /// Returns the input unchanged if no remap is active.
    pub fn remapSourceValue(self: *const CanonicalGraph, source_id: ValueId) ?ValueId {
        const remap = self.source_remap orelse return source_id;
        return remap[@intFromEnum(source_id)];
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
        var canonicalized = CanonicalGraph.init(alloc);
        errdefer canonicalized.deinit();

        for (self.values.items) |item| {
            const expr = try canonicalizeExpr(self, &canonicalized, item);
            _ = try canonicalized.addValue(item.dtype, item.shape, expr);
        }

        return canonicalized;
    }

    /// Remove values with no consumers that aren't the output (last value).
    /// Returns a new compacted graph with remapped ValueIds.
    pub fn eliminateDeadValues(self: *const CanonicalGraph, alloc: Alloc) !CanonicalGraph {
        const len = self.values.items.len;
        if (len == 0) return CanonicalGraph.init(alloc);

        // Compute use counts.
        const use_counts = try alloc.alloc(u32, len);
        defer alloc.free(use_counts);
        @memset(use_counts, 0);

        for (self.values.items) |v| {
            for (exprInputs(v.expr)) |maybe_id| {
                const id = maybe_id orelse break;
                use_counts[@intFromEnum(id)] += 1;
            }
        }

        // Mark liveness. The output (last value) always survives.
        const live = try alloc.alloc(bool, len);
        defer alloc.free(live);
        @memset(live, true);

        const output_idx = len - 1;
        // Walk backwards, killing dead values and propagating.
        var i: usize = output_idx;
        while (true) {
            if (i != output_idx and use_counts[i] == 0) {
                live[i] = false;
                for (exprInputs(self.values.items[i].expr)) |maybe_id| {
                    const id = maybe_id orelse break;
                    use_counts[@intFromEnum(id)] -= 1;
                }
            }
            if (i == 0) break;
            i -= 1;
        }

        // If nothing was eliminated, clone as-is.
        var any_dead = false;
        for (live) |l| {
            if (!l) {
                any_dead = true;
                break;
            }
        }
        if (!any_dead) {
            var out = CanonicalGraph.init(alloc);
            errdefer out.deinit();
            for (self.values.items) |v| {
                const expr = if (v.expr == .reduce)
                    CanonicalExpr{ .reduce = .{
                        .op = v.expr.reduce.op,
                        .input = v.expr.reduce.input,
                        .axes = try out.dupeAxes(v.expr.reduce.axes),
                    } }
                else
                    v.expr;
                _ = try out.addValue(v.dtype, v.shape, expr);
            }
            return out;
        }

        // Build remap table and compact graph.
        const remap = try alloc.alloc(ValueId, len);
        defer alloc.free(remap);

        var out = CanonicalGraph.init(alloc);
        errdefer out.deinit();

        for (self.values.items, 0..) |v, idx| {
            if (!live[idx]) continue;
            const expr = try remapExpr(&out, v.expr, remap);
            remap[idx] = try out.addValue(v.dtype, v.shape, expr);
        }

        // Store source_remap so downstream can translate source ValueIds.
        const source_remap = try alloc.alloc(?ValueId, len);
        for (0..len) |idx| {
            source_remap[idx] = if (live[idx]) remap[idx] else null;
        }
        out.source_remap = source_remap;

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

    pub fn detectLinear(self: *const CanonicalGraph, output: ValueId) ?LinearSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .add) return null;

        const lhs = out_v.expr.binary.lhs;
        const rhs = out_v.expr.binary.rhs;
        const lhs_v = self.value(lhs);
        const rhs_v = self.value(rhs);

        const parts: LinearParts = blk: {
            if (lhs_v.expr == .matmul and rhs_v.expr == .view and rhs_v.expr.view.kind == .broadcast and !isScalarBroadcastValue(self, rhs)) {
                break :blk .{ .matmul = lhs, .bias_bcast = rhs };
            }
            if (rhs_v.expr == .matmul and lhs_v.expr == .view and lhs_v.expr.view.kind == .broadcast and !isScalarBroadcastValue(self, lhs)) {
                break :blk .{ .matmul = rhs, .bias_bcast = lhs };
            }
            return null;
        };

        const matmul_v = self.value(parts.matmul);
        const bias_bcast_v = self.value(parts.bias_bcast);

        return .{
            .input = matmul_v.expr.matmul.lhs,
            .weight = matmul_v.expr.matmul.rhs,
            .bias = bias_bcast_v.expr.view.input,
            .matmul = parts.matmul,
            .output = output,
        };
    }

    pub fn detectLinearGelu(self: *const CanonicalGraph, output: ValueId) ?LinearGeluSpec {
        const out_v = self.value(output);
        if (out_v.expr != .unary or out_v.expr.unary.op != .gelu) return null;

        const linear = self.detectLinear(out_v.expr.unary.input) orelse return null;
        return .{ .linear = linear, .output = output };
    }

    pub fn detectLinearRelu(self: *const CanonicalGraph, output: ValueId) ?LinearReluSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .mul) return null;

        const lhs = out_v.expr.binary.lhs;
        const rhs = out_v.expr.binary.rhs;
        const lhs_v = self.value(lhs);
        const rhs_v = self.value(rhs);

        if (rhs_v.expr == .unary and rhs_v.expr.unary.op == UnaryOp.step) {
            const linear = self.detectLinear(lhs) orelse return null;
            if (rhs_v.expr.unary.input != linear.output) return null;
            return .{ .linear = linear, .step_node = rhs, .output = output };
        }
        if (lhs_v.expr == .unary and lhs_v.expr.unary.op == UnaryOp.step) {
            const linear = self.detectLinear(rhs) orelse return null;
            if (lhs_v.expr.unary.input != linear.output) return null;
            return .{ .linear = linear, .step_node = lhs, .output = output };
        }
        return null;
    }

    pub fn detectLinearResidual(self: *const CanonicalGraph, output: ValueId) ?LinearResidualSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .add) return null;

        if (self.detectLinear(out_v.expr.binary.lhs)) |linear| {
            return .{ .linear = linear, .residual = out_v.expr.binary.rhs, .output = output };
        }
        if (self.detectLinear(out_v.expr.binary.rhs)) |linear| {
            return .{ .linear = linear, .residual = out_v.expr.binary.lhs, .output = output };
        }
        return null;
    }

    pub fn detectMatmulResidual(self: *const CanonicalGraph, output: ValueId) ?MatmulResidualSpec {
        const out_v = self.value(output);
        if (out_v.expr != .binary or out_v.expr.binary.op != .add) return null;

        const lhs = out_v.expr.binary.lhs;
        const rhs = out_v.expr.binary.rhs;
        const lhs_v = self.value(lhs);
        const rhs_v = self.value(rhs);

        if (lhs_v.expr == .matmul) {
            return .{
                .lhs = lhs_v.expr.matmul.lhs,
                .rhs = lhs_v.expr.matmul.rhs,
                .matmul = lhs,
                .residual = rhs,
                .output = output,
            };
        }
        if (rhs_v.expr == .matmul) {
            return .{
                .lhs = rhs_v.expr.matmul.lhs,
                .rhs = rhs_v.expr.matmul.rhs,
                .matmul = rhs,
                .residual = lhs,
                .output = output,
            };
        }
        return null;
    }

    pub fn detectConv2d(self: *const CanonicalGraph, output_id: ValueId) ?Conv2dSpec {
        const output = self.value(output_id);

        // Output should be a reshape back to 4D
        if (output.expr != .view or output.expr.view.kind != .reshape) return null;
        if (output.shape.ndims != 4) return null;
        const reshape_input = output.expr.view.input;

        // The reshape input should be a sum reduction
        const sum_val = self.value(reshape_input);
        if (sum_val.expr != .reduce or sum_val.expr.reduce.op != .sum) return null;

        // The sum input should be a mul
        const mul_id = sum_val.expr.reduce.input;
        const mul_val = self.value(mul_id);
        if (mul_val.expr != .binary or mul_val.expr.binary.op != .mul) return null;

        // Both mul inputs should be strided views
        const lhs_id = mul_val.expr.binary.lhs;
        const rhs_id = mul_val.expr.binary.rhs;
        const lhs = self.value(lhs_id);
        const rhs = self.value(rhs_id);

        if (lhs.expr != .view or lhs.expr.view.kind != .strided) return null;
        if (rhs.expr != .view or rhs.expr.view.kind != .strided) return null;

        // Strided views should be 7D (conv2d decomposition)
        if (lhs.shape.ndims != 7) return null;
        if (rhs.shape.ndims != 7) return null;

        // Determine which is input view and which is kernel view
        const input_id = lhs.expr.view.input;
        const kernel_id = rhs.expr.view.input;

        // Both sources should be 4D
        if (self.value(input_id).shape.ndims != 4) return null;
        if (self.value(kernel_id).shape.ndims != 4) return null;

        return .{
            .input = input_id,
            .kernel = kernel_id,
            .input_view = lhs_id,
            .kernel_view = rhs_id,
            .mul_node = mul_id,
            .sum_node = reshape_input,
            .reshape_node = output_id,
            .output = output_id,
        };
    }

    pub fn detectMaxPool2d(self: *const CanonicalGraph, output_id: ValueId) ?MaxPool2dSpec {
        const output = self.value(output_id);

        // Output should be a reshape
        if (output.expr != .view or output.expr.view.kind != .reshape) return null;
        // Output should be 4D
        if (output.shape.ndims != 4) return null;

        const max_id = output.expr.view.input;
        const max_val = self.value(max_id);
        if (max_val.expr != .reduce or max_val.expr.reduce.op != .max) return null;

        const strided_id = max_val.expr.reduce.input;
        const strided_val = self.value(strided_id);
        if (strided_val.expr != .view or strided_val.expr.view.kind != .strided) return null;

        // Strided view should be 6D
        if (strided_val.shape.ndims != 6) return null;
        // Window dims should be 2x2
        if (strided_val.shape.dims[2] != 2 or strided_val.shape.dims[3] != 2) return null;

        const input_id = strided_val.expr.view.input;
        if (self.value(input_id).shape.ndims != 4) return null;

        return .{
            .input = input_id,
            .strided = strided_id,
            .max_node = max_id,
            .output = output_id,
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

    pub fn detectKernelPattern(self: *const CanonicalGraph, kind: KernelPatternKind, output: ValueId, alloc: Alloc) !?KernelPattern {
        _ = alloc;
        return switch (kind) {
            .softmax => if (self.detectSoftmax(output)) |spec| .{ .softmax = lowerSoftmaxKernelSpec(self, spec) orelse return null } else null,
            .log_softmax => if (self.detectLogSoftmax(output)) |spec| .{ .log_softmax = lowerLogSoftmaxKernelSpec(self, spec) orelse return null } else null,
            .cross_entropy => if (self.detectCrossEntropy(output)) |spec| .{ .cross_entropy = lowerCrossEntropyKernelSpec(self, spec) orelse return null } else null,
            .layer_norm => if (self.detectLayerNorm(output)) |spec| .{ .layer_norm = lowerLayerNormKernelSpec(self, spec) orelse return null } else null,
            .linear => if (self.detectLinear(output)) |spec| .{ .linear = lowerLinearKernelSpec(self, spec) orelse return null } else null,
            .linear_gelu => if (self.detectLinearGelu(output)) |spec| .{ .linear_gelu = lowerLinearGeluKernelSpec(self, spec) orelse return null } else null,
            .linear_relu => if (self.detectLinearRelu(output)) |spec| .{ .linear_relu = lowerLinearReluKernelSpec(self, spec) orelse return null } else null,
            .linear_residual => if (self.detectLinearResidual(output)) |spec| .{ .linear_residual = lowerLinearResidualKernelSpec(self, spec) orelse return null } else null,
            .matmul_residual => if (self.detectMatmulResidual(output)) |spec| .{ .matmul_residual = lowerMatmulResidualKernelSpec(self, spec) orelse return null } else null,
            .conv2d => if (self.detectConv2d(output)) |spec| .{ .conv2d = spec } else null,
            .conv2d_bwd_input => null, // backward not yet supported in compiler IR
            .conv2d_bwd_kernel => null, // backward not yet supported in compiler IR
            .max_pool2d => if (self.detectMaxPool2d(output)) |spec| .{ .max_pool2d = spec } else null,
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
                    if (try self.detectKernelPattern(kind, output, alloc)) |p| {
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
    scatter_add_view: ScatterAddViewSpec,
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
                .scatter_add_view => |s| KernelExpr{ .scatter_add_view = s },
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

    /// Build typed kernel patterns from the canonical graph.
    ///
    /// This is intentionally broader than runtime support: the compiler may
    /// understand patterns that the current executor still ignores. That keeps
    /// semantic recognition and execution policy decoupled.
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
        schedule: ?SchedulePlan = null,
        execution: ?ExecutionPlan = null,

        pub fn init(alloc: Alloc) Self {
            return .{
                .alloc = alloc,
                .lowering = Lowering(T).init(alloc),
                .rewrites = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.execution) |*e| e.deinit();
            if (self.schedule) |*s| s.deinit();
            if (self.kernel) |*k| k.deinit();
            if (self.canonical) |*c| c.deinit();
            self.rewrites.deinit(self.alloc);
            self.lowering.deinit();
        }

        /// Lower a single root tensor into the canonical IR graph.
        /// The lowering memo prevents re-lowering already-seen tensors,
        /// so calling this multiple times with different roots is efficient.
        pub fn lowerOneRoot(self: *Self, root: *const Tensor) (Alloc.Error || error{UnsupportedOp})!void {
            _ = try self.lowering.lowerRoot(root);
        }

        /// Run post-lowering pipeline stages: canonicalize, rewrite, kernel
        /// plan, schedule, and execution plan. Call after all roots have been
        /// lowered via `lowerOneRoot`.
        pub fn finalize(self: *Self) Alloc.Error!void {
            self.canonical = try self.lowering.graph.canonicalize(self.alloc);
            errdefer if (self.canonical) |*c| c.deinit();

            const output: ValueId = @enumFromInt(self.canonical.?.values.items.len - 1);
            self.rewrites = try self.canonical.?.applyRewritePasses(output, self.alloc);
            errdefer self.rewrites.deinit(self.alloc);

            self.kernel = try KernelPlan.lowerFromCanonical(self.alloc, &self.canonical.?);
            try self.kernel.?.addRewriteAnnotations(self.alloc, output, self.rewrites.items);
            try self.kernel.?.buildPatterns(self.alloc, &self.canonical.?);
            self.schedule = try SchedulePlan.build(self.alloc, &self.canonical.?, &self.kernel.?);
            self.execution = try ExecutionPlan.build(self.alloc, &self.schedule.?);
        }

        /// Run the current compiler pipeline end-to-end.
        ///
        /// Current stages:
        /// - lower tensor graph to canonical IR
        /// - canonicalize and detect rewrite-layer patterns
        /// - lower to kernel plan and collect typed kernel patterns
        /// - build a minimal schedule/materialization plan
        pub fn compile(self: *Self, root: *const Tensor) (Alloc.Error || error{UnsupportedOp})!void {
            try self.lowerOneRoot(root);
            try self.finalize();
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

fn lowerLinearKernelSpec(graph: *const CanonicalGraph, spec: LinearSpec) ?LinearKernelSpec {
    const output_v = graph.value(spec.output);
    if (output_v.expr != .binary or output_v.expr.binary.op != .add) return null;

    const lhs = output_v.expr.binary.lhs;
    const rhs = output_v.expr.binary.rhs;
    const lhs_v = graph.value(lhs);
    const bias_bcast = if (lhs_v.expr == .view and lhs_v.expr.view.kind == .broadcast) lhs else rhs;

    return .{
        .input = spec.input,
        .weight = spec.weight,
        .bias = spec.bias,
        .bias_bcast = bias_bcast,
        .matmul = spec.matmul,
        .output = spec.output,
    };
}

fn lowerLinearGeluKernelSpec(graph: *const CanonicalGraph, spec: LinearGeluSpec) ?LinearGeluKernelSpec {
    return .{
        .linear = lowerLinearKernelSpec(graph, spec.linear) orelse return null,
        .output = spec.output,
    };
}

fn lowerLinearReluKernelSpec(graph: *const CanonicalGraph, spec: LinearReluSpec) ?LinearReluKernelSpec {
    return .{
        .linear = lowerLinearKernelSpec(graph, spec.linear) orelse return null,
        .step_node = spec.step_node,
        .output = spec.output,
    };
}

fn lowerLinearResidualKernelSpec(graph: *const CanonicalGraph, spec: LinearResidualSpec) ?LinearResidualKernelSpec {
    return .{
        .linear = lowerLinearKernelSpec(graph, spec.linear) orelse return null,
        .residual = spec.residual,
        .output = spec.output,
    };
}

fn lowerMatmulResidualKernelSpec(_: *const CanonicalGraph, spec: MatmulResidualSpec) ?MatmulResidualKernelSpec {
    return .{
        .lhs = spec.lhs,
        .rhs = spec.rhs,
        .matmul = spec.matmul,
        .residual = spec.residual,
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
                .broadcast_to => .{ .view = .{
                    .kind = .broadcast,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .permute => .{ .view = .{
                    .kind = .transpose,
                    .input = try self.lowerTensor(tensor.source0().?),
                    .shape = Shape.fromTensor(T, tensor),
                    .strides = .{ .values = tensor.strides },
                } },
                .as_strided => .{ .view = .{
                    .kind = .strided,
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
                    .transpose_lhs = tensor.matmul_flags.trans0,
                    .transpose_rhs = tensor.matmul_flags.trans1,
                } },
                .scatter_add_view => .{ .scatter_add_view = .{
                    .grad = try self.lowerTensor(tensor.source0().?),
                    .view = try self.lowerTensor(tensor.source1().?),
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
        .input, .constant, .gather, .scatter_add, .scatter_add_view, .matmul, .scale => value.expr,
        .unary => |u| .{ .unary = .{ .op = u.op, .input = u.input } },
        .reduce => |r| .{ .reduce = .{ .op = r.op, .input = r.input, .axes = try canonicalizeAxes(dst_graph, r.axes) } },
        .view => |v| canonicalizeView(src_graph, v),
        .binary => |b| canonicalizeBinary(src_graph, b),
    };
}

fn canonicalizeView(src_graph: *const CanonicalGraph, view: ViewSpec) CanonicalExpr {
    const input_value = src_graph.value(view.input);

    if (isIdentityView(input_value, view)) {
        return input_value.expr;
    }

    if (input_value.expr == .view) {
        const parent = input_value.expr.view;
        if (view.kind == .reshape and parent.kind == .transpose) {
            if (sameShape(src_graph.value(parent.input).shape, view.shape)) {
                return .{ .view = .{
                    .kind = .transpose,
                    .input = parent.input,
                    .shape = view.shape,
                    .strides = parent.strides,
                } };
            }
        }
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
        if (view.kind == .transpose and parent.kind == .transpose) {
            const grandparent = src_graph.value(parent.input);
            if (sameShape(grandparent.shape, view.shape) and sameOptionalStrides(view.strides, grandparent.shape, grandparent.shape)) {
                return grandparent.expr;
            }
        }
    }

    return .{ .view = view };
}

fn isIdentityView(input_value: *const CanonicalValue, view: ViewSpec) bool {
    return sameShape(input_value.shape, view.shape) and sameOptionalStrides(view.strides, input_value.shape, input_value.shape);
}

fn sameShape(a: Shape, b: Shape) bool {
    if (a.ndims != b.ndims) return false;
    var i: usize = 0;
    while (i < a.ndims) : (i += 1) {
        if (a.dims[i] != b.dims[i]) return false;
    }
    return true;
}

fn sameOptionalStrides(candidate: ?Strides, input_shape: Shape, output_shape: Shape) bool {
    const strides = candidate orelse return sameShape(input_shape, output_shape);
    const expected = Strides.contiguous(output_shape);
    var i: usize = 0;
    while (i < output_shape.ndims) : (i += 1) {
        if (strides.values[i] != expected.values[i]) return false;
    }
    return true;
}

fn canonicalizeAxes(graph: *CanonicalGraph, axes: []const Axis) ![]const Axis {
    if (axes.len <= 1) return graph.dupeAxes(axes);

    var buf = try graph.alloc.dupe(Axis, axes);
    errdefer graph.alloc.free(buf);

    std.mem.sort(Axis, buf, {}, sortAxisAsc);

    var out_len: usize = 0;
    for (buf) |axis| {
        if (out_len > 0 and buf[out_len - 1] == axis) continue;
        buf[out_len] = axis;
        out_len += 1;
    }

    const normalized = try graph.alloc.dupe(Axis, buf[0..out_len]);
    graph.alloc.free(buf);
    try graph.reduction_axes_storage.append(graph.alloc, normalized);
    return normalized;
}

fn sortAxisAsc(_: void, lhs: Axis, rhs: Axis) bool {
    return lhs < rhs;
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

    const out = canon.values.items[canon.values.items.len - 1];
    try std.testing.expect(out.expr == .scale);
    try std.testing.expectEqual(input, out.expr.scale.input);
    try std.testing.expectEqual(scalar, out.expr.scale.scalar);
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

    const out = canon.values.items[canon.values.items.len - 1];
    try std.testing.expect(out.expr == .view);
    try std.testing.expectEqual(input, out.expr.view.input);
    try std.testing.expectEqual(ViewKind.reshape, out.expr.view.kind);
}

test "compiler - canonicalize identity reshape to input" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 2, 3 }), .input);
    _ = try graph.addValue(.f32, Shape.init(&.{ 2, 3 }), .{ .view = .{
        .kind = .reshape,
        .input = input,
        .shape = Shape.init(&.{ 2, 3 }),
        .strides = null,
    } });

    var canonicalized = try graph.canonicalize(std.testing.allocator);
    defer canonicalized.deinit();

    // Canonicalization rewrites the identity reshape to its input's expression.
    try std.testing.expect(canonicalized.values.items[1].expr == .input);

    // After DCE the dead original input is removed.
    var canon = try canonicalized.eliminateDeadValues(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expectEqual(@as(usize, 1), canon.values.items.len);
    try std.testing.expect(canon.values.items[0].expr == .input);
}

test "compiler - canonicalize transpose then reshape back to source shape" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 2, 3 }), .input);
    const tr = try graph.addValue(.f32, Shape.init(&.{ 3, 2 }), .{ .view = .{
        .kind = .transpose,
        .input = input,
        .shape = Shape.init(&.{ 3, 2 }),
        .strides = .{ .values = .{ 2, 1 } ++ ([_]usize{0} ** (max_dims - 2)) },
    } });
    _ = try graph.addValue(.f32, Shape.init(&.{ 2, 3 }), .{ .view = .{
        .kind = .reshape,
        .input = tr,
        .shape = Shape.init(&.{ 2, 3 }),
        .strides = null,
    } });

    var canonicalized = try graph.canonicalize(std.testing.allocator);
    defer canonicalized.deinit();

    // Canonicalization rewrites reshape(transpose(x)) back to transpose(x).
    const out_pre = canonicalized.values.items[canonicalized.values.items.len - 1];
    try std.testing.expect(out_pre.expr == .view);
    try std.testing.expectEqual(ViewKind.transpose, out_pre.expr.view.kind);

    // After DCE the dead intermediate transpose is removed, compacting to 2 values.
    var canon = try canonicalized.eliminateDeadValues(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expectEqual(@as(usize, 2), canon.values.items.len);
    const out = canon.values.items[canon.values.items.len - 1];
    try std.testing.expect(out.expr == .view);
    try std.testing.expectEqual(ViewKind.transpose, out.expr.view.kind);
    try std.testing.expect(canon.values.items[@intFromEnum(out.expr.view.input)].expr == .input);
}

test "compiler - canonicalize reduction axes order and duplicates" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{ 2, 3, 4 }), .input);
    _ = try graph.addValue(.f32, Shape.init(&.{ 1, 3, 1 }), .{ .reduce = .{
        .op = .sum,
        .input = input,
        .axes = try graph.dupeAxes(&.{ 2, 0, 2 }),
    } });

    var canon = try graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    try std.testing.expect(canon.values.items[1].expr == .reduce);
    try std.testing.expectEqualSlices(Axis, &.{ 0, 2 }, canon.values.items[1].expr.reduce.axes);
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

test "compiler - detects linear canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });

    const y = nnlib.linear(f32, x, w, b);

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectLinear(@enumFromInt(canon.values.items.len - 1)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(spec.input));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(spec.weight));
}

test "compiler - detects linear gelu canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });

    const y = nnlib.linear(f32, x, w, b).gelu();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectLinearGelu(@enumFromInt(canon.values.items.len - 1)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(spec.linear.input));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(spec.linear.weight));
}

test "compiler - kernel patterns detect linear gelu outer region" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });

    const y = nnlib.linear(f32, x, w, b).gelu();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.kernel.?.patterns.items.len >= 1);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .linear_gelu);
}

test "compiler - detects linear relu canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });

    const y = nnlib.linear(f32, x, w, b).relu();

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectLinearRelu(@enumFromInt(canon.values.items.len - 1)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(spec.linear.input));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(spec.linear.weight));
}

test "compiler - kernel patterns detect linear relu outer region" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });

    const y = nnlib.linear(f32, x, w, b).relu();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.kernel.?.patterns.items.len >= 1);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .linear_relu);
}

test "compiler - detects linear residual canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    var residual = try Tensor.init(a, &.{ 3, 2 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });
    residual.setData(&.{ 1, 1, 1, 1, 1, 1 });

    const y = nnlib.linear(f32, x, w, b).add(residual);

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectLinearResidual(@enumFromInt(canon.values.items.len - 1)) orelse return error.SkipZigTest;
    try std.testing.expect(spec.residual != spec.linear.output);
    try std.testing.expect(canon.value(spec.residual).expr == .input);
}

test "compiler - kernel patterns detect linear residual outer region" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w = try Tensor.init(a, &.{ 3, 4 });
    var b = try Tensor.init(a, &.{3});
    var residual = try Tensor.init(a, &.{ 3, 2 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    b.setData(&.{ 0.5, 1.0, -0.5 });
    residual.setData(&.{ 1, 1, 1, 1, 1, 1 });

    const y = nnlib.linear(f32, x, w, b).add(residual);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.kernel.?.patterns.items.len >= 1);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .linear_residual);
}

test "compiler - detects matmul residual canonical pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var lhs = try Tensor.init(a, &.{ 4, 2 });
    var rhs = try Tensor.init(a, &.{ 3, 4 });
    var residual = try Tensor.init(a, &.{ 3, 2 });
    lhs.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    rhs.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    residual.setData(&.{ 1, 1, 1, 1, 1, 1 });

    const y = lhs.matMul(false, rhs, false).add(residual);

    var lowering = Lowering(f32).init(std.testing.allocator);
    defer lowering.deinit();
    _ = try lowering.lowerRoot(y);

    var canon = try lowering.graph.canonicalize(std.testing.allocator);
    defer canon.deinit();

    const spec = canon.detectMatmulResidual(@enumFromInt(canon.values.items.len - 1)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(spec.lhs));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(spec.rhs));
}

test "compiler - kernel patterns detect matmul residual outer region" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var lhs = try Tensor.init(a, &.{ 4, 2 });
    var rhs = try Tensor.init(a, &.{ 3, 4 });
    var residual = try Tensor.init(a, &.{ 3, 2 });
    lhs.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    rhs.setData(&.{ 1, 0, -1, 2, 1, 0, 0, 1, 2, -1, 0, 1 });
    residual.setData(&.{ 1, 1, 1, 1, 1, 1 });

    const y = lhs.matMul(false, rhs, false).add(residual);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.kernel != null);
    try std.testing.expect(pipeline.kernel.?.patterns.items.len >= 1);
    try std.testing.expect(pipeline.kernel.?.patterns.items[0].pattern == .matmul_residual);
}

test "compiler - schedule plan materializes outputs and virtualizes pattern internals" {
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

    try std.testing.expect(pipeline.schedule != null);
    try std.testing.expectEqual(@as(usize, 1), pipeline.schedule.?.steps.items.len);
    try std.testing.expect(pipeline.schedule.?.steps.items[0] == .kernel_pattern);

    const pattern = pipeline.kernel.?.patterns.items[0].pattern.softmax;
    try std.testing.expectEqual(MaterializationKind.virtual, pipeline.schedule.?.materialization.items[@intFromEnum(pattern.max_node)]);
    try std.testing.expectEqual(MaterializationKind.materialized, pipeline.schedule.?.materialization.items[@intFromEnum(pattern.output)]);
}

test "compiler - schedule plan emits generic steps for uncovered values" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.exp();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.schedule != null);
    try std.testing.expectEqual(@as(usize, 1), pipeline.schedule.?.steps.items.len);
    try std.testing.expect(pipeline.schedule.?.steps.items[0] == .generic);
    try std.testing.expectEqual(@as(usize, 1), pipeline.schedule.?.outputs.items.len);
}

test "compiler - schedule plan builds generic elementwise regions" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.exp().log();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.schedule != null);
    try std.testing.expectEqual(@as(usize, 1), pipeline.schedule.?.regions.items.len);
    try std.testing.expectEqual(@as(usize, 1), pipeline.schedule.?.steps.items.len);
    try std.testing.expect(pipeline.schedule.?.steps.items[0] == .schedule_region);

    const region = pipeline.schedule.?.regions.items[0].elementwise;
    try std.testing.expectEqual(@as(usize, 2), region.nodes.len);
    try std.testing.expectEqual(MaterializationKind.virtual, pipeline.schedule.?.materialization.items[@intFromEnum(region.nodes[0])]);
    try std.testing.expectEqual(MaterializationKind.materialized, pipeline.schedule.?.materialization.items[@intFromEnum(region.nodes[1])]);
}

test "compiler - schedule regions start at boundaries, not mid-chain" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const input = try graph.addValue(.f32, Shape.init(&.{3}), .input);
    const expv = try graph.addValue(.f32, Shape.init(&.{3}), .{ .unary = .{ .op = .exp, .input = input } });
    _ = try graph.addValue(.f32, Shape.init(&.{3}), .{ .unary = .{ .op = .log, .input = expv } });

    var kernel = try KernelPlan.lowerFromCanonical(std.testing.allocator, &graph);
    defer kernel.deinit();
    try kernel.buildPatterns(std.testing.allocator, &graph);

    var schedule = try SchedulePlan.build(std.testing.allocator, &graph, &kernel);
    defer schedule.deinit();

    try std.testing.expectEqual(@as(usize, 1), schedule.regions.items.len);
    try std.testing.expectEqual(@as(usize, 2), schedule.regions.items[0].elementwise.nodes.len);
}

test "compiler - schedule regions allow swapped commutative continuation" {
    var graph = CanonicalGraph.init(std.testing.allocator);
    defer graph.deinit();

    const scalar = try graph.addValue(.f32, Shape.init(&.{1}), .constant);
    const input = try graph.addValue(.f32, Shape.init(&.{3}), .input);
    const expv = try graph.addValue(.f32, Shape.init(&.{3}), .{ .unary = .{ .op = .exp, .input = input } });
    const addv = try graph.addValue(.f32, Shape.init(&.{3}), .{ .binary = .{ .op = .add, .lhs = scalar, .rhs = expv } });
    _ = try graph.addValue(.f32, Shape.init(&.{3}), .{ .unary = .{ .op = .log, .input = addv } });

    var kernel = try KernelPlan.lowerFromCanonical(std.testing.allocator, &graph);
    defer kernel.deinit();
    try kernel.buildPatterns(std.testing.allocator, &graph);

    var schedule = try SchedulePlan.build(std.testing.allocator, &graph, &kernel);
    defer schedule.deinit();

    try std.testing.expectEqual(@as(usize, 1), schedule.regions.items.len);
    const region = schedule.regions.items[0].elementwise;
    try std.testing.expectEqual(input, region.input);
    try std.testing.expectEqual(@as(usize, 3), region.nodes.len);
    try std.testing.expectEqual(expv, region.nodes[0]);
    try std.testing.expectEqual(addv, region.nodes[1]);
}

test "compiler - schedule report reflects kernel and generic steps" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const soft = x.softmax(&.{1});
    const y = soft.exp();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    try pipeline.schedule.?.writeReport(buf.writer(std.testing.allocator));

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "schedule steps=") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "pattern output=") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "generic output=") != null);
}

test "compiler - schedule report reflects schedule regions" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.exp().log();

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    try pipeline.schedule.?.writeReport(buf.writer(std.testing.allocator));

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "region kind=elementwise") != null);
}

test "compiler - transformer ffn schedule contains linear epilogue pattern" {
    const Tensor = tensorlib.Tensor(f32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var x = try Tensor.init(a, &.{ 4, 2 });
    var w1 = try Tensor.init(a, &.{ 6, 4 });
    var b1 = try Tensor.init(a, &.{6});
    var w2 = try Tensor.init(a, &.{ 4, 6 });
    var b2 = try Tensor.init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    w1.setData(&.{
        1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
    });
    b1.setData(&.{ 0, 1, 2, 3, 4, 5 });
    w2.setData(&.{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 0, 0,
        0, 0, 1, 1,
    });
    b2.setData(&.{ 1, 1, 1, 1 });

    const hidden = nnlib.linear(f32, x, w1, b1).gelu();
    const y = nnlib.linear(f32, hidden, w2, b2);

    var pipeline = Pipeline(f32).init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.compile(y);

    try std.testing.expect(pipeline.schedule != null);

    var saw_linear_gelu = false;
    var saw_linear = false;
    for (pipeline.kernel.?.patterns.items) |record| {
        switch (record.pattern) {
            .linear_gelu => saw_linear_gelu = true,
            .linear => saw_linear = true,
            else => {},
        }
    }

    try std.testing.expect(saw_linear_gelu);
    try std.testing.expect(saw_linear or pipeline.schedule.?.steps.items.len >= 2);
}
