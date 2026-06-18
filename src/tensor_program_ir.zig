const std = @import("std");
const Op = @import("op.zig").Op;
const tensor_mod = @import("tensor.zig");
const fused_mod = @import("tensor/fused.zig");

pub const Inspection = struct {
    source_node_count: usize,
    step_count: usize,
    op_count: usize,
    value_count: usize,
    dense_value_count: usize,
    input_edge_count: usize,
    elementwise_substep_count: usize,
    node_step_count: usize,
    fusion_step_count: usize,
    value_step_count: usize,
    structural_step_count: usize,
    effect_step_count: usize,
    elementwise_chain_step_count: usize,
    layer_norm_step_count: usize,
    log_softmax_step_count: usize,
    unsupported_fusion_step_count: usize,
    max_rank: usize,
    total_output_elements: usize,
    normalized_step_hash: u64,
    normalized_shape_hash: u64,
};

pub fn TensorProgramIr(comptime T: type) type {
    const Tensor = tensor_mod.Tensor(T);
    const ComputeGraph = @import("graph.zig").ComputeGraph(T);
    return struct {
        const Ir = @This();
        pub const TensorFusionPayload = fused_mod.FusionPayload(T);
        pub const TensorFusionKind = fused_mod.FusionKind;
        pub const TensorMaxDims = tensor_mod.max_dims;
        pub const ValueId = u32;

        pub const StepKind = enum(u8) {
            node,
            fusion,
        };

        pub const StepInfo = struct {
            kind: StepKind,
            fusion_kind: ?TensorFusionKind = null,
            op: Op,
            input_count: usize,
            rank: usize,
            element_count: usize,
            storage_offset: usize,
            ne: [TensorMaxDims]usize,
            strides: [TensorMaxDims]usize,

            fn init(kind: StepKind, fusion_kind: ?TensorFusionKind, tensor: *const Tensor, input_count: usize) StepInfo {
                return .{
                    .kind = kind,
                    .fusion_kind = fusion_kind,
                    .op = tensor.opTag(),
                    .input_count = input_count,
                    .rank = tensor.n_dims,
                    .element_count = tensor.nElems(),
                    .storage_offset = tensor.storage_offset,
                    .ne = tensor.ne,
                    .strides = tensor.strides,
                };
            }

            fn fromNode(node: *const Tensor) StepInfo {
                return StepInfo.init(.node, null, node, nodeInputCount(node));
            }

            fn fromFusion(payload: TensorFusionPayload) StepInfo {
                return StepInfo.init(.fusion, std.meta.activeTag(payload), fusionOutput(payload), fusionInputCount(payload));
            }

            pub fn isStructural(self: StepInfo) bool {
                return switch (self.op) {
                    .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to => true,
                    else => false,
                };
            }

            pub fn isEffectful(self: StepInfo) bool {
                return switch (self.op) {
                    .slice_assign, .slice_assign_rows, .scatter_add_rows, .scatter_add_picks, .scatter_add_view => true,
                    else => false,
                };
            }

            pub fn isValue(self: StepInfo) bool {
                return self.op != .none and !self.isStructural();
            }
        };

        pub const IrShape = struct {
            rank: usize,
            ne: [TensorMaxDims]usize,
            strides: [TensorMaxDims]usize,
            storage_offset: usize,
            element_count: usize,
            dense: bool,

            fn fromTensor(tensor: *const Tensor) IrShape {
                return .{
                    .rank = tensor.n_dims,
                    .ne = tensor.ne,
                    .strides = tensor.strides,
                    .storage_offset = tensor.storage_offset,
                    .element_count = tensor.nElems(),
                    .dense = tensor.isDenseLayout(),
                };
            }
        };

        pub const IrValue = struct {
            source_tensor: *const Tensor,
            shape: IrShape,
        };

        pub const IrOp = struct {
            source_step: usize,
            info: StepInfo,
            attrs: IrAttrs,
            output: ValueId,
            input_start: usize,
            input_count: usize,

            fn inputs(self: IrOp, ir: Ir) []const ValueId {
                return ir.op_inputs[self.input_start..][0..self.input_count];
            }
        };

        pub const IrAttrs = union(enum) {
            none,
            elementwise_chain: struct {
                input: ValueId,
                step_start: usize,
                step_count: usize,
            },
            log_softmax: struct {
                op_start: usize,
                op_count: usize,
            },
            layer_norm: struct {
                eps: f32,
            },
        };

        pub const IrElementwiseStep = struct {
            op: Op,
            output: ValueId,
            secondary: ?ValueId = null,
            is_swapped: bool = false,
        };

        pub const Step = struct {
            op_index: usize,
        };

        nodes: []const *Tensor,
        steps: []const Step,
        ops: []const IrOp,
        op_inputs: []const ValueId,
        elementwise_steps: []const IrElementwiseStep,
        values: []const IrValue,
        source_node_count: usize,

        fn nodeInputCount(node: *const Tensor) usize {
            var count: usize = 0;
            if (node.src0 != null) count += 1;
            if (node.src1 != null) count += 1;
            if (node.src2 != null) count += 1;
            if (node.src3 != null) count += 1;
            return count;
        }

        fn fusionOutput(payload: TensorFusionPayload) *Tensor {
            return switch (payload) {
                .elementwise_chain => |plan| plan.output(),
                .conv2d => |plan| plan.output,
                .conv2d_bwd_input => |plan| plan.output,
                .conv2d_bwd_kernel => |plan| plan.output,
                .max_pool2d => |plan| plan.output,
                .avg_pool2d => |plan| plan.output,
                .max_pool2d_bwd => |plan| plan.output,
                .log_softmax => |plan| plan.output,
                .cross_entropy => |plan| plan.mean_node,
                .layer_norm => |plan| plan.output,
            };
        }

        fn fusionInputCount(payload: TensorFusionPayload) usize {
            return switch (payload) {
                .elementwise_chain => |plan| blk: {
                    var count: usize = 1;
                    for (plan.nodes, 0..) |node, i| {
                        if (node.opTag().isBinary() and plan.otherOperand(i) != null) count += 1;
                    }
                    break :blk count;
                },
                .conv2d => |plan| 2 + @as(usize, @intFromBool(plan.bias != null)),
                .conv2d_bwd_input => 2,
                .conv2d_bwd_kernel => 2,
                .max_pool2d => 1,
                .avg_pool2d => 1,
                .max_pool2d_bwd => 2,
                .log_softmax => 1,
                .cross_entropy => 2,
                .layer_norm => 1,
            };
        }

        fn hashIrInt(comptime U: type, hasher: *std.hash.Wyhash, value: U) void {
            var v = value;
            hasher.update(std.mem.asBytes(&v));
        }

        fn hashAttrs(hasher: *std.hash.Wyhash, ir: Ir, attrs: IrAttrs) void {
            hashIrInt(u8, hasher, @intFromEnum(std.meta.activeTag(attrs)));
            switch (attrs) {
                .none => {},
                .elementwise_chain => |chain| {
                    hashIrInt(u32, hasher, chain.input);
                    hashIrInt(u64, hasher, @intCast(chain.step_start));
                    hashIrInt(u64, hasher, @intCast(chain.step_count));
                    for (ir.elementwise_steps[chain.step_start..][0..chain.step_count]) |step| {
                        hashIrInt(u8, hasher, @intFromEnum(step.op));
                        hashIrInt(u32, hasher, step.output);
                        hashIrInt(u8, hasher, @intFromBool(step.is_swapped));
                        if (step.secondary) |secondary| {
                            hashIrInt(u8, hasher, 1);
                            hashIrInt(u32, hasher, secondary);
                        } else {
                            hashIrInt(u8, hasher, 0);
                        }
                    }
                },
                .log_softmax => |log_softmax| {
                    hashIrInt(u64, hasher, @intCast(log_softmax.op_start));
                    hashIrInt(u64, hasher, @intCast(log_softmax.op_count));
                },
                .layer_norm => |ln| hashIrInt(u32, hasher, @bitCast(ln.eps)),
            }
        }

        fn hashOp(hasher: *std.hash.Wyhash, ir: Ir, op: IrOp, index: usize) void {
            const step = op.info;
            hashIrInt(u64, hasher, @intCast(index));
            hashIrInt(u64, hasher, @intCast(op.source_step));
            hashIrInt(u8, hasher, @intFromEnum(step.kind));
            hashIrInt(u8, hasher, if (step.fusion_kind) |kind| @intFromEnum(kind) else std.math.maxInt(u8));
            hashIrInt(u8, hasher, @intFromEnum(step.op));
            hashAttrs(hasher, ir, op.attrs);
            hashIrInt(u32, hasher, op.output);
            hashIrInt(u64, hasher, @intCast(step.input_count));
            for (op.inputs(ir)) |input| hashIrInt(u32, hasher, input);
            hashIrInt(u64, hasher, @intCast(step.rank));
            hashIrInt(u64, hasher, @intCast(step.element_count));
            hashIrInt(u64, hasher, @intCast(step.storage_offset));
        }

        fn hashStepShape(hasher: *std.hash.Wyhash, step: StepInfo, index: usize) void {
            hashIrInt(u64, hasher, @intCast(index));
            hashIrInt(u64, hasher, @intCast(step.rank));
            hashIrInt(u64, hasher, @intCast(step.element_count));
            for (step.ne) |dim| hashIrInt(u64, hasher, @intCast(dim));
            for (step.strides) |stride| hashIrInt(u64, hasher, @intCast(stride));
            hashIrInt(u64, hasher, @intCast(step.storage_offset));
        }

        fn hashValueShape(hasher: *std.hash.Wyhash, value: IrValue, index: usize) void {
            hashIrInt(u64, hasher, @intCast(index));
            hashIrInt(u64, hasher, @intCast(value.shape.rank));
            hashIrInt(u64, hasher, @intCast(value.shape.element_count));
            hashIrInt(u8, hasher, @intFromBool(value.shape.dense));
            for (value.shape.ne) |dim| hashIrInt(u64, hasher, @intCast(dim));
            for (value.shape.strides) |stride| hashIrInt(u64, hasher, @intCast(stride));
            hashIrInt(u64, hasher, @intCast(value.shape.storage_offset));
        }

        const IrBuilder = struct {
            alloc: std.mem.Allocator,
            values: std.ArrayListUnmanaged(IrValue) = .empty,
            ops: std.ArrayListUnmanaged(IrOp) = .empty,
            op_inputs: std.ArrayListUnmanaged(ValueId) = .empty,
            elementwise_steps: std.ArrayListUnmanaged(IrElementwiseStep) = .empty,
            value_ids: std.AutoHashMap(*const Tensor, ValueId),

            fn init(alloc: std.mem.Allocator) IrBuilder {
                return .{
                    .alloc = alloc,
                    .value_ids = std.AutoHashMap(*const Tensor, ValueId).init(alloc),
                };
            }

            fn deinit(self: *IrBuilder) void {
                self.values.deinit(self.alloc);
                self.ops.deinit(self.alloc);
                self.op_inputs.deinit(self.alloc);
                self.elementwise_steps.deinit(self.alloc);
                self.value_ids.deinit();
            }

            fn ensureValue(self: *IrBuilder, tensor: *const Tensor) !ValueId {
                if (self.value_ids.get(tensor)) |id| return id;
                const id = std.math.cast(ValueId, self.values.items.len) orelse return error.UnsupportedDeviceOp;
                try self.value_ids.put(tensor, id);
                try self.values.append(self.alloc, .{
                    .source_tensor = tensor,
                    .shape = IrShape.fromTensor(tensor),
                });
                return id;
            }

            fn ensureMaybeValue(self: *IrBuilder, maybe_tensor: ?*const Tensor) !void {
                const tensor = maybe_tensor orelse return;
                _ = try self.ensureValue(tensor);
            }

            fn materializeGraphValues(self: *IrBuilder, nodes: []const *Tensor) !void {
                for (nodes) |node| {
                    _ = try self.ensureValue(node);
                    try self.ensureMaybeValue(node.src0);
                    try self.ensureMaybeValue(node.src1);
                    try self.ensureMaybeValue(node.src2);
                    try self.ensureMaybeValue(node.src3);
                }
            }

            fn appendInput(self: *IrBuilder, tensor: *const Tensor) !void {
                try self.op_inputs.append(self.alloc, try self.ensureValue(tensor));
            }

            fn appendNodeInputs(self: *IrBuilder, node: *const Tensor) !void {
                try self.appendMaybeInput(node.src0);
                try self.appendMaybeInput(node.src1);
                try self.appendMaybeInput(node.src2);
                try self.appendMaybeInput(node.src3);
            }

            fn appendMaybeInput(self: *IrBuilder, maybe_tensor: ?*const Tensor) !void {
                const tensor = maybe_tensor orelse return;
                try self.appendInput(tensor);
            }

            fn appendFusionInputs(self: *IrBuilder, payload: TensorFusionPayload) !void {
                switch (payload) {
                    .elementwise_chain => |plan| {
                        try self.appendInput(plan.input);
                        for (plan.nodes, 0..) |node, i| {
                            if (node.opTag().isBinary()) {
                                if (plan.otherOperand(i)) |other| try self.appendInput(other);
                            }
                        }
                    },
                    .conv2d => |plan| {
                        try self.appendInput(plan.input);
                        try self.appendInput(plan.kernel);
                        try self.appendMaybeInput(plan.bias);
                    },
                    .conv2d_bwd_input => |plan| {
                        try self.appendInput(plan.output_grad);
                        try self.appendInput(plan.kernel);
                    },
                    .conv2d_bwd_kernel => |plan| {
                        try self.appendInput(plan.input);
                        try self.appendInput(plan.output_grad);
                    },
                    .max_pool2d => |plan| try self.appendInput(plan.input),
                    .avg_pool2d => |plan| try self.appendInput(plan.input),
                    .max_pool2d_bwd => |plan| {
                        try self.appendInput(plan.input);
                        try self.appendInput(plan.output_grad);
                    },
                    .log_softmax => |plan| try self.appendInput(plan.input),
                    .cross_entropy => |plan| {
                        try self.appendInput(plan.log_softmax.input);
                        try self.appendInput(plan.targets);
                    },
                    .layer_norm => |plan| try self.appendInput(plan.input),
                }
            }

            fn appendNodeOpTo(self: *IrBuilder, op_list: *std.ArrayListUnmanaged(IrOp), source_step: usize, node: *Tensor) !usize {
                const input_start = self.op_inputs.items.len;
                try self.appendNodeInputs(node);
                const input_count = self.op_inputs.items.len - input_start;
                const output = try self.ensureValue(node);
                const op_index = op_list.items.len;
                try op_list.append(self.alloc, .{
                    .source_step = source_step,
                    .info = StepInfo.fromNode(node),
                    .attrs = .none,
                    .output = output,
                    .input_start = input_start,
                    .input_count = input_count,
                });
                return op_index;
            }

            fn appendNodeOp(self: *IrBuilder, source_step: usize, node: *Tensor) !usize {
                return self.appendNodeOpTo(&self.ops, source_step, node);
            }

            fn appendLogSoftmaxSubOps(self: *IrBuilder, source_step: usize, plan: fused_mod.LogSoftmaxPlan(T)) !IrAttrs {
                const op_start = self.ops.items.len;
                const ordered_nodes = [_]*Tensor{
                    plan.max_node,
                    plan.rep_max,
                    plan.neg_rep_max,
                    plan.shifted,
                    plan.exp_node,
                    plan.sum_node,
                    plan.log_node,
                    plan.rep_log,
                    plan.neg_rep_log,
                    plan.output,
                };
                for (&ordered_nodes) |node| _ = try self.appendNodeOp(source_step, node);
                return .{ .log_softmax = .{
                    .op_start = op_start,
                    .op_count = self.ops.items.len - op_start,
                } };
            }

            fn appendFusionAttrs(self: *IrBuilder, source_step: usize, payload: TensorFusionPayload) !IrAttrs {
                return switch (payload) {
                    .elementwise_chain => |plan| blk: {
                        const step_start = self.elementwise_steps.items.len;
                        for (plan.nodes, 0..) |node, i| {
                            const node_op = node.opTag();
                            const secondary = if (node_op.isBinary()) sec: {
                                const other = plan.otherOperand(i) orelse break :sec null;
                                break :sec try self.ensureValue(other);
                            } else null;
                            try self.elementwise_steps.append(self.alloc, .{
                                .op = node_op,
                                .output = try self.ensureValue(node),
                                .secondary = secondary,
                                .is_swapped = node_op.isBinary() and plan.otherOperandRole(i) == .src0,
                            });
                        }
                        break :blk .{ .elementwise_chain = .{
                            .input = try self.ensureValue(plan.input),
                            .step_start = step_start,
                            .step_count = self.elementwise_steps.items.len - step_start,
                        } };
                    },
                    .log_softmax => |plan| try self.appendLogSoftmaxSubOps(source_step, plan),
                    .layer_norm => |plan| .{ .layer_norm = .{ .eps = @floatCast(plan.eps_like.data[0]) } },
                    else => .none,
                };
            }

            fn appendFusionOp(self: *IrBuilder, source_step: usize, payload: TensorFusionPayload) !usize {
                const input_start = self.op_inputs.items.len;
                try self.appendFusionInputs(payload);
                const input_count = self.op_inputs.items.len - input_start;
                const attrs = try self.appendFusionAttrs(source_step, payload);
                const output = try self.ensureValue(fusionOutput(payload));
                const op_index = self.ops.items.len;
                try self.ops.append(self.alloc, .{
                    .source_step = source_step,
                    .info = StepInfo.fromFusion(payload),
                    .attrs = attrs,
                    .output = output,
                    .input_start = input_start,
                    .input_count = input_count,
                });
                return op_index;
            }
        };

        fn nodeStep(op_index: usize) Step {
            return .{
                .op_index = op_index,
            };
        }

        fn fusionStep(op_index: usize) Step {
            return .{
                .op_index = op_index,
            };
        }

        pub fn init(alloc: std.mem.Allocator, graph: *ComputeGraph) !Ir {
            const nodes = graph.nodes.items[0..graph.forward_node_count];
            var builder = IrBuilder.init(alloc);
            defer builder.deinit();
            try builder.materializeGraphValues(nodes);
            const graph_steps = graph.forward_execution_steps.items;
            const steps = if (graph_steps.len > 0) blk: {
                const out = try alloc.alloc(Step, graph_steps.len);
                errdefer alloc.free(out);
                for (graph_steps, out, 0..) |graph_step, *step, i| {
                    step.* = switch (graph_step) {
                        .fusion => |idx| blk_step: {
                            const payload = graph.fused_chains.items[idx].payload;
                            break :blk_step fusionStep(try builder.appendFusionOp(i, payload));
                        },
                        .node => |node| nodeStep(try builder.appendNodeOp(i, node)),
                    };
                }
                break :blk out;
            } else blk: {
                const out = try alloc.alloc(Step, nodes.len);
                errdefer alloc.free(out);
                for (nodes, out, 0..) |node, *step, i| step.* = nodeStep(try builder.appendNodeOp(i, node));
                break :blk out;
            };
            errdefer alloc.free(steps);

            const values = try builder.values.toOwnedSlice(alloc);
            builder.values = .empty;
            errdefer alloc.free(values);
            const ops = try builder.ops.toOwnedSlice(alloc);
            builder.ops = .empty;
            errdefer alloc.free(ops);
            const op_inputs = try builder.op_inputs.toOwnedSlice(alloc);
            builder.op_inputs = .empty;
            errdefer alloc.free(op_inputs);
            const elementwise_steps = try builder.elementwise_steps.toOwnedSlice(alloc);
            builder.elementwise_steps = .empty;
            errdefer alloc.free(elementwise_steps);

            return .{
                .nodes = nodes,
                .steps = steps,
                .ops = ops,
                .op_inputs = op_inputs,
                .elementwise_steps = elementwise_steps,
                .values = values,
                .source_node_count = nodes.len,
            };
        }

        pub fn deinit(self: *Ir, alloc: std.mem.Allocator) void {
            alloc.free(self.steps);
            alloc.free(self.ops);
            alloc.free(self.op_inputs);
            alloc.free(self.elementwise_steps);
            alloc.free(self.values);
            self.* = undefined;
        }

        pub fn inspect(self: Ir) Inspection {
            var step_hasher = std.hash.Wyhash.init(0);
            step_hasher.update("zgml-tensor-program-ir-steps-v1");
            var shape_hasher = std.hash.Wyhash.init(0);
            shape_hasher.update("zgml-tensor-program-ir-shapes-v1");
            var out = Inspection{
                .source_node_count = self.source_node_count,
                .step_count = self.steps.len,
                .op_count = self.ops.len,
                .value_count = self.values.len,
                .dense_value_count = 0,
                .input_edge_count = self.op_inputs.len,
                .elementwise_substep_count = self.elementwise_steps.len,
                .node_step_count = 0,
                .fusion_step_count = 0,
                .value_step_count = 0,
                .structural_step_count = 0,
                .effect_step_count = 0,
                .elementwise_chain_step_count = 0,
                .layer_norm_step_count = 0,
                .log_softmax_step_count = 0,
                .unsupported_fusion_step_count = 0,
                .max_rank = 0,
                .total_output_elements = 0,
                .normalized_step_hash = 0,
                .normalized_shape_hash = 0,
            };

            for (self.values, 0..) |value, i| {
                if (value.shape.dense) out.dense_value_count += 1;
                hashValueShape(&shape_hasher, value, i);
            }

            for (self.ops, 0..) |op, i| {
                const info = op.info;
                hashOp(&step_hasher, self, op, i);
                hashStepShape(&shape_hasher, info, i);
                out.max_rank = @max(out.max_rank, info.rank);
                out.total_output_elements += info.element_count;
                if (info.isStructural()) out.structural_step_count += 1;
                if (info.isEffectful()) out.effect_step_count += 1;
                if (info.isValue()) out.value_step_count += 1;
                switch (info.kind) {
                    .node => out.node_step_count += 1,
                    .fusion => {
                        out.fusion_step_count += 1;
                        switch (info.fusion_kind.?) {
                            .elementwise_chain => out.elementwise_chain_step_count += 1,
                            .layer_norm => out.layer_norm_step_count += 1,
                            .log_softmax => out.log_softmax_step_count += 1,
                            .conv2d_bwd_input, .conv2d_bwd_kernel, .max_pool2d_bwd, .cross_entropy => out.unsupported_fusion_step_count += 1,
                            .conv2d, .max_pool2d, .avg_pool2d => {},
                        }
                    },
                }
            }
            out.normalized_step_hash = step_hasher.final();
            out.normalized_shape_hash = shape_hasher.final();

            return out;
        }

        pub fn collectBuffers(
            self: Ir,
            buffers: anytype,
            input_tensors: []const *const Tensor,
            persistent_tensors: []const *const Tensor,
            output_tensors: []const *const Tensor,
        ) !void {
            for (persistent_tensors) |t| try buffers.skipInitialUpload(t);

            for (self.values) |value| try buffers.ensure(value.source_tensor);
            for (input_tensors) |t| try buffers.ensure(t);
            for (persistent_tensors) |t| try buffers.ensure(t);
            for (output_tensors) |t| try buffers.ensure(t);
        }

        pub fn valueTensor(self: Ir, id: ValueId) *Tensor {
            return @constCast(self.values[@intCast(id)].source_tensor);
        }

        pub fn opOutputTensor(self: Ir, op: IrOp) *Tensor {
            return self.valueTensor(op.output);
        }

        pub fn opInputTensor(self: Ir, op: IrOp, index: usize) ?*Tensor {
            if (index >= op.input_count) return null;
            return self.valueTensor(op.inputs(self)[index]);
        }
    };
}
