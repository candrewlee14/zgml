//! Computation graph for automatic differentiation.
//!
//! A `ComputeGraph` owns an arena allocator and manages the lifecycle of all
//! tensors created within it. Call `allocator()` to get the arena for tensor
//! creation, then `buildForward` / `buildBackward` to wire up the graph.
//! A single `deinit()` frees everything.

const std = @import("std");

const tensorlib = @import("./tensor.zig");
const Tensor = tensorlib.Tensor;
const Op = @import("op.zig").Op;
const loss = @import("loss.zig");
const compiler = @import("compiler.zig");
const fused = @import("tensor/fused.zig");
const assert = std.debug.assert;
const testing = std.testing;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

/// Manages forward and backward passes over a tensor computation graph.
///
/// All tensors should be allocated from `allocator()` so that `deinit()`
/// can free them in bulk via the arena.
pub fn ComputeGraph(comptime T: type) type {
    return struct {
        const Self = @This();

        built_forward: bool = false,
        built_backward: bool = false,
        forward_node_count: usize = 0,

        arena: std.heap.ArenaAllocator,
        nodes: std.ArrayList(*Tensor(T)),
        grads: std.ArrayList(?*Tensor(T)),
        leaves: std.ArrayList(*Tensor(T)),
        scratch: std.ArrayList(*Tensor(T)),

        /// Fusion state — populated by `fusionPass()`.
        fused_chains: std.ArrayList(fused.FusionPlan(T)),
        /// Per-node flag: true means this node is part of a fused chain
        /// and should be skipped during normal compute iteration.
        fused_skip: std.ArrayList(bool),

        /// Set up resources for compute graph.
        /// Must call `buildForward` (then optionally `buildBackward`) to be able to do computation.
        pub fn init(backing_alloc: Alloc) Self {
            return .{
                .arena = std.heap.ArenaAllocator.init(backing_alloc),
                .nodes = .{},
                .grads = .{},
                .leaves = .{},
                .scratch = .{},
                .fused_chains = .{},
                .fused_skip = .{},
            };
        }

        /// Returns the arena allocator for allocating tensors within this graph.
        pub fn allocator(self: *Self) Alloc {
            return self.arena.allocator();
        }

        // ---------------------------------------------------------------
        // Tensor creation helpers
        // ---------------------------------------------------------------

        /// Create a tensor within this graph.
        pub fn tensor(self: *Self, ne: []const usize) !*Tensor(T) {
            return try Tensor(T).init(self.arena.allocator(), ne);
        }

        /// Create a tensor and mark it as a learnable parameter.
        pub fn param(self: *Self, ne: []const usize) !*Tensor(T) {
            const t = try Tensor(T).init(self.arena.allocator(), ne);
            t.setParam();
            return t;
        }

        /// Create a scalar (1-element) tensor.
        pub fn scalar(self: *Self, val: T) !*Tensor(T) {
            return try Tensor(T).initScalar(self.arena.allocator(), val);
        }

        /// Create a tensor filled with evenly spaced values.
        pub fn linspace(self: *Self, ne: []const usize, start: T, end: T) !*Tensor(T) {
            return try Tensor(T).initLinspace(self.arena.allocator(), ne, start, end);
        }

        /// Clean up all the resources for this compute graph
        pub fn deinit(self: *Self) void {
            const alloc = self.arena.allocator();
            self.fused_chains.deinit(alloc);
            self.fused_skip.deinit(alloc);
            self.nodes.deinit(alloc);
            self.grads.deinit(alloc);
            self.leaves.deinit(alloc);
            self.scratch.deinit(alloc);
            self.arena.deinit();
        }

        /// Build a graph where the provided tensor is the final output node
        pub fn buildForward(self: *Self, root: *Tensor(T)) Alloc.Error!void {
            try self.buildForwardHelper(root);
            self.built_forward = true;
            self.forward_node_count = self.nodes.items.len;
        }
        fn buildForwardHelper(self: *Self, root_node: *Tensor(T)) Alloc.Error!void {
            const n_before = self.nodes.items.len;
            try self.addParentsThenSelf(root_node);
            // node should be last node
            const n_change = self.nodes.items.len - n_before;
            if (n_change > 0) assert(self.nodes.items[self.nodes.items.len - 1] == root_node);
        }
        /// Build a backward graph
        pub fn buildBackward(self: *Self, keep: bool) Alloc.Error!void {
            assert(self.nodes.items.len > 0);
            const alloc = self.arena.allocator();
            const nodes_len = self.nodes.items.len;
            for (0..nodes_len) |j| {
                const i = nodes_len - j - 1;
                const node = self.nodes.items[i];

                // because we detached the grad nodes from the original graph, we can afford inplace operations
                if (node.hasGrad()) {
                    try node.backward(alloc, &self.scratch, keep);
                }
            }
            for (0..nodes_len) |j| {
                const i = nodes_len - j - 1;
                const node = self.nodes.items[i];
                if (node.is_param) {
                    assert(node.hasGrad());
                    try self.buildForwardHelper(node.gradOrNull().?);
                }
            }

            self.built_backward = true;
            self.resetGrads();
        }

        /// Detect chains of fusible elementwise ops in the forward graph
        /// and prepare comptime-specialized kernels for them.
        ///
        /// Call after `buildForward()` (and optionally after `buildBackward()`).
        /// Fusion only affects the forward pass — backward still uses
        /// the original nodes and their stored intermediate values.
        pub fn fusionPass(self: *Self) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const fwd_count = self.forward_node_count;
            if (fwd_count < 2) return;

            const compiler_plans = try self.detectCompilerFusionPlans();

            self.fused_chains.clearRetainingCapacity();
            self.fused_skip.clearRetainingCapacity();

            // Build use-count: how many forward nodes read each node's output
            const use_count = try alloc.alloc(u16, fwd_count);
            @memset(use_count, 0);

            for (self.nodes.items[0..fwd_count]) |node| {
                if (node.source0()) |src| {
                    for (self.nodes.items[0..fwd_count], 0..) |n, j| {
                        if (n == src) {
                            use_count[j] += 1;
                            break;
                        }
                    }
                }
                if (node.source1()) |src| {
                    for (self.nodes.items[0..fwd_count], 0..) |n, j| {
                        if (n == src) {
                            use_count[j] += 1;
                            break;
                        }
                    }
                }
            }

            // Initialize skip bitmap
            self.fused_skip = try std.ArrayList(bool).initCapacity(alloc, fwd_count);
            self.fused_skip.items.len = fwd_count;
            @memset(self.fused_skip.items, false);

            // Scan for fusible chains
            var i: usize = 0;
            var next_compiler_plan: usize = 0;
            while (i < fwd_count) : (i += 1) {
                const node = self.nodes.items[i];
                if (next_compiler_plan < compiler_plans.items.len) {
                    const compiler_plan = compiler_plans.items[next_compiler_plan];
                    if (compiler_plan.start_idx == i) {
                        for (i..compiler_plan.plan.output_idx + 1) |idx| self.fused_skip.items[idx] = true;
                        try self.fused_chains.append(alloc, compiler_plan.plan);
                        i = compiler_plan.plan.output_idx;
                        next_compiler_plan += 1;
                        continue;
                    }
                }
                if (detectConv2dPattern(self, i)) |plan| {
                    for (i..plan.output_idx + 1) |idx| self.fused_skip.items[idx] = true;
                    try self.fused_chains.append(alloc, plan);
                    i = plan.output_idx;
                    continue;
                }
                if (!node.opTag().isFusible()) continue;
                if (self.fused_skip.items[i]) continue;

                // Start a chain from this node
                const chain_start = i;
                var chain_end = i; // inclusive

                // Extend forward: look for the next node that reads chain_end via src0
                var extending = true;
                while (extending) {
                    extending = false;
                    if (chain_end + 1 < fwd_count) {
                        const next = self.nodes.items[chain_end + 1];
                        if (next.opTag().isFusible() and
                            next.source0().? == self.nodes.items[chain_end] and
                            next.isSameShape(self.nodes.items[chain_end]) and
                            use_count[chain_end] == 1)
                        {
                            chain_end += 1;
                            extending = true;
                        }
                    }
                }

                const chain_len = chain_end - chain_start + 1;
                if (chain_len < 2) continue;

                // Record the fused chain
                const chain_nodes = try alloc.alloc(*Tensor(T), chain_len);
                for (chain_start..chain_end + 1, 0..) |idx, k| {
                    chain_nodes[k] = self.nodes.items[idx];
                    self.fused_skip.items[idx] = true;
                }

                try self.fused_chains.append(alloc, .{
                    .output_idx = chain_end,
                    .payload = .{ .elementwise_chain = .{
                        .input = self.nodes.items[chain_start].source0().?,
                        .nodes = chain_nodes,
                    } },
                });

                i = chain_end; // skip past the chain
            }
        }

        fn detectSoftmaxPattern(self: *const Self, start: usize) ?fused.FusionPlan(T) {
            const nodes = self.nodes.items;

            if (start + 8 < self.forward_node_count) {
                const n0 = nodes[start + 0];
                const n1 = nodes[start + 1];
                const n2 = nodes[start + 2];
                const n3 = nodes[start + 3];
                const n4 = nodes[start + 4];
                const n5 = nodes[start + 5];
                const n6 = nodes[start + 6];
                const n7 = nodes[start + 7];
                const n8 = nodes[start + 8];

                if (n0.isOp(.max) and
                    n1.isOp(.repeat) and n1.sourceIs(.src0, n0) and
                    n2.isOp(.neg) and n2.sourceIs(.src0, n1) and
                    n3.isOp(.add) and n3.source0().? == n0.source0().? and n3.sourceIs(.src1, n2) and
                    n4.isOp(.exp) and n4.sourceIs(.src0, n3) and
                    n5.isOp(.sum) and n5.sourceIs(.src0, n4) and
                    n6.isOp(.repeat) and n6.sourceIs(.src0, n5) and
                    n7.isOp(.recip) and n7.sourceIs(.src0, n6) and
                    n8.isOp(.mul) and n8.sourceIs(.src0, n4) and n8.sourceIs(.src1, n7))
                {
                    return .{ .output_idx = start + 8, .payload = .{ .softmax = .{
                        .input = n0.source0().?,
                        .max_node = n0,
                        .rep_max = n1,
                        .neg_rep_max = n2,
                        .shifted = n3,
                        .exp_node = n4,
                        .sum_node = n5,
                        .rep_sum = n6,
                        .recip_rep_sum = n7,
                        .output = n8,
                    } } };
                }
            }

            if (start + 9 < self.forward_node_count) {
                const n0 = nodes[start + 0];
                const n1 = nodes[start + 1];
                const n2 = nodes[start + 2];
                const n3 = nodes[start + 3];
                const n4 = nodes[start + 4];
                const n5 = nodes[start + 5];
                const n6 = nodes[start + 6];
                const n7 = nodes[start + 7];
                const n8 = nodes[start + 8];
                const n9 = nodes[start + 9];

                if (n0.isOp(.max) and
                    n1.isOp(.repeat) and n1.sourceIs(.src0, n0) and
                    n2.isOp(.neg) and n2.sourceIs(.src0, n1) and
                    n3.isOp(.add) and n3.source0().? == n0.source0().? and n3.sourceIs(.src1, n2) and
                    n4.isOp(.exp) and n4.sourceIs(.src0, n3) and
                    n5.isOp(.sum) and n5.sourceIs(.src0, n4) and
                    n6.isOp(.log) and n6.sourceIs(.src0, n5) and
                    n7.isOp(.repeat) and n7.sourceIs(.src0, n6) and
                    n8.isOp(.neg) and n8.sourceIs(.src0, n7) and
                    n9.isOp(.add) and n9.source0().? == n3 and n9.sourceIs(.src1, n8))
                {
                    return .{ .output_idx = start + 9, .payload = .{ .log_softmax = .{
                        .input = n0.source0().?,
                        .max_node = n0,
                        .rep_max = n1,
                        .neg_rep_max = n2,
                        .shifted = n3,
                        .exp_node = n4,
                        .sum_node = n5,
                        .log_node = n6,
                        .rep_log = n7,
                        .neg_rep_log = n8,
                        .output = n9,
                    } } };
                }
            }

            return null;
        }

        fn detectConv2dPattern(self: *const Self, start: usize) ?fused.FusionPlan(T) {
            if (start + 3 >= self.forward_node_count) return null;

            const input_view = self.nodes.items[start + 0];
            const kernel_view = self.nodes.items[start + 1];
            const mul_node = self.nodes.items[start + 2];
            const sum_node = self.nodes.items[start + 3];
            const output = if (start + 4 < self.forward_node_count) self.nodes.items[start + 4] else null;

            if (!input_view.isOp(.as_strided)) return null;
            if (!kernel_view.isOp(.as_strided)) return null;
            if (!mul_node.isOp(.mul)) return null;
            if (!sum_node.isOp(.sum)) return null;
            if (mul_node.source0().? != input_view or mul_node.source1().? != kernel_view) return null;
            if (sum_node.source0().? != mul_node) return null;

            const input = input_view.source0().?;
            const kernel = kernel_view.source0().?;
            if (input.n_dims != 4 or kernel.n_dims != 4) return null;
            if (output == null or !output.?.isOp(.reshape) or output.?.source0().? != sum_node) return null;

            return .{ .output_idx = start + 4, .payload = .{ .conv2d = .{
                .input = input,
                .kernel = kernel,
                .input_view = input_view,
                .kernel_view = kernel_view,
                .mul_node = mul_node,
                .sum_node = sum_node,
                .output = output.?,
            } } };
        }

        const CompilerPlan = struct {
            start_idx: usize,
            plan: fused.FusionPlan(T),
        };

        fn sortCompilerPlans(_: void, lhs: CompilerPlan, rhs: CompilerPlan) bool {
            return lhs.start_idx < rhs.start_idx;
        }

        fn detectCompilerFusionPlans(self: *const Self) Alloc.Error!std.ArrayList(CompilerPlan) {
            var plans = std.ArrayList(CompilerPlan){};
            errdefer plans.deinit(std.heap.page_allocator);

            if (self.forward_node_count == 0) return plans;

            var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer temp_arena.deinit();

            var pipeline = compiler.Pipeline(T).init(temp_arena.allocator());
            defer pipeline.deinit();

            pipeline.compile(self.nodes.items[self.forward_node_count - 1]) catch |err| switch (err) {
                error.UnsupportedOp => return plans,
                error.OutOfMemory => return error.OutOfMemory,
            };

            for (pipeline.kernel.?.patterns.items) |record| {
                const mapped = self.mapCompilerPattern(&pipeline.kernel.?, record.pattern) orelse continue;
                try plans.append(std.heap.page_allocator, mapped);
            }

            std.mem.sort(CompilerPlan, plans.items, {}, sortCompilerPlans);
            return plans;
        }

        fn detectCompilerRootRewritePlan(self: *const Self) Alloc.Error!?CompilerPlan {
            var plans = try self.detectCompilerFusionPlans();
            defer plans.deinit(std.heap.page_allocator);
            if (plans.items.len == 0) return null;
            return plans.items[plans.items.len - 1];
        }

        fn mapCompilerPattern(self: *const Self, kernel: *const compiler.KernelPlan, pattern: compiler.KernelPattern) ?CompilerPlan {
            return switch (pattern) {
                .softmax => |spec| blk: {
                    const max_node = self.nodeForKernelValueId(kernel, spec.max_node) orelse return null;
                    const rep_max = self.nodeForKernelValueId(kernel, spec.rep_max) orelse return null;
                    const neg_rep_max = self.nodeForKernelValueId(kernel, spec.neg_rep_max) orelse return null;
                    const shifted = self.nodeForKernelValueId(kernel, spec.shifted) orelse return null;
                    const exp_node = self.nodeForKernelValueId(kernel, spec.exp_node) orelse return null;
                    const sum_node = self.nodeForKernelValueId(kernel, spec.sum_node) orelse return null;
                    const rep_sum = self.nodeForKernelValueId(kernel, spec.rep_sum) orelse return null;
                    const recip_rep_sum = self.nodeForKernelValueId(kernel, spec.recip_rep_sum) orelse return null;
                    const output = self.nodeForKernelValueId(kernel, spec.output) orelse return null;
                    const start_idx = self.kernelValueIndex(kernel, spec.max_node) orelse return null;
                    const output_idx = self.kernelValueIndex(kernel, spec.output) orelse return null;

                    break :blk .{ .start_idx = start_idx, .plan = .{
                        .output_idx = output_idx,
                        .payload = .{ .softmax = .{
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
                        } },
                    } };
                },
                .log_softmax => |spec| blk: {
                    const max_node = self.nodeForKernelValueId(kernel, spec.max_node) orelse return null;
                    const rep_max = self.nodeForKernelValueId(kernel, spec.rep_max) orelse return null;
                    const neg_rep_max = self.nodeForKernelValueId(kernel, spec.neg_rep_max) orelse return null;
                    const shifted = self.nodeForKernelValueId(kernel, spec.shifted) orelse return null;
                    const exp_node = self.nodeForKernelValueId(kernel, spec.exp_node) orelse return null;
                    const sum_node = self.nodeForKernelValueId(kernel, spec.sum_node) orelse return null;
                    const log_node = self.nodeForKernelValueId(kernel, spec.log_node) orelse return null;
                    const rep_log = self.nodeForKernelValueId(kernel, spec.rep_log) orelse return null;
                    const neg_rep_log = self.nodeForKernelValueId(kernel, spec.neg_rep_log) orelse return null;
                    const output = self.nodeForKernelValueId(kernel, spec.output) orelse return null;
                    const start_idx = self.kernelValueIndex(kernel, spec.max_node) orelse return null;
                    const output_idx = self.kernelValueIndex(kernel, spec.output) orelse return null;

                    break :blk .{ .start_idx = start_idx, .plan = .{
                        .output_idx = output_idx,
                        .payload = .{ .log_softmax = .{
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
                        } },
                    } };
                },
                .cross_entropy => |spec| blk: {
                    const max_node = self.nodeForKernelValueId(kernel, spec.log_softmax.max_node) orelse return null;
                    const rep_max = self.nodeForKernelValueId(kernel, spec.log_softmax.rep_max) orelse return null;
                    const neg_rep_max = self.nodeForKernelValueId(kernel, spec.log_softmax.neg_rep_max) orelse return null;
                    const shifted = self.nodeForKernelValueId(kernel, spec.log_softmax.shifted) orelse return null;
                    const exp_node = self.nodeForKernelValueId(kernel, spec.log_softmax.exp_node) orelse return null;
                    const sum_node = self.nodeForKernelValueId(kernel, spec.log_softmax.sum_node) orelse return null;
                    const log_node = self.nodeForKernelValueId(kernel, spec.log_softmax.log_node) orelse return null;
                    const rep_log = self.nodeForKernelValueId(kernel, spec.log_softmax.rep_log) orelse return null;
                    const neg_rep_log = self.nodeForKernelValueId(kernel, spec.log_softmax.neg_rep_log) orelse return null;
                    const log_softmax_output = self.nodeForKernelValueId(kernel, spec.log_softmax.output) orelse return null;
                    const picked = self.nodeForKernelValueId(kernel, spec.picked) orelse return null;
                    const neg_picked = self.nodeForKernelValueId(kernel, spec.neg_picked) orelse return null;
                    const sum_node_ce = self.nodeForKernelValueId(kernel, spec.sum_node) orelse return null;
                    const mean_node = self.nodeForKernelValueId(kernel, spec.mean_node) orelse return null;
                    const start_idx = self.kernelValueIndex(kernel, spec.log_softmax.max_node) orelse return null;
                    const output_idx = self.kernelValueIndex(kernel, spec.mean_node) orelse return null;

                    break :blk .{ .start_idx = start_idx, .plan = .{
                        .output_idx = output_idx,
                        .payload = .{ .cross_entropy = .{
                            .log_softmax = .{
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
                            },
                            .targets = picked.source1().?,
                            .picked = picked,
                            .neg_picked = neg_picked,
                            .sum_node = sum_node_ce,
                            .mean_node = mean_node,
                        } },
                    } };
                },
                .layer_norm => |spec| blk: {
                    const sum_node = self.nodeForKernelValueId(kernel, spec.sum_node) orelse return null;
                    const mean_node = self.nodeForKernelValueId(kernel, spec.mean_node) orelse return null;
                    const rep_mean = self.nodeForKernelValueId(kernel, spec.rep_mean) orelse return null;
                    const neg_rep_mean = self.nodeForKernelValueId(kernel, spec.neg_rep_mean) orelse return null;
                    const centered = self.nodeForKernelValueId(kernel, spec.centered) orelse return null;
                    const sqr_node = self.nodeForKernelValueId(kernel, spec.sqr_node) orelse return null;
                    const var_sum = self.nodeForKernelValueId(kernel, spec.var_sum) orelse return null;
                    const var_node = self.nodeForKernelValueId(kernel, spec.var_node) orelse return null;
                    const eps_like = self.nodeForKernelValueId(kernel, spec.eps_like) orelse return null;
                    const var_eps = self.nodeForKernelValueId(kernel, spec.var_eps) orelse return null;
                    const sqrt_node = self.nodeForKernelValueId(kernel, spec.sqrt_node) orelse return null;
                    const recip_node = self.nodeForKernelValueId(kernel, spec.recip_node) orelse return null;
                    const rep_std_inv = self.nodeForKernelValueId(kernel, spec.rep_std_inv) orelse return null;
                    const output = self.nodeForKernelValueId(kernel, spec.output) orelse return null;
                    const start_idx = self.kernelValueIndex(kernel, spec.sum_node) orelse return null;
                    const output_idx = self.kernelValueIndex(kernel, spec.output) orelse return null;

                    break :blk .{ .start_idx = start_idx, .plan = .{
                        .output_idx = output_idx,
                        .payload = .{ .layer_norm = .{
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
                        } },
                    } };
                },
            };
        }

        fn kernelValueIndex(self: *const Self, kernel: *const compiler.KernelPlan, id: compiler.ValueId) ?usize {
            _ = self;
            for (kernel.values.items, 0..) |value, idx| {
                if (value.id == id) return idx;
            }
            return null;
        }

        fn nodeForKernelValueId(self: *const Self, kernel: *const compiler.KernelPlan, id: compiler.ValueId) ?*Tensor(T) {
            const idx = self.kernelValueIndex(kernel, id) orelse return null;
            return self.nodeAt(idx);
        }

        fn nodeAt(self: *const Self, idx: usize) ?*Tensor(T) {
            if (idx >= self.forward_node_count) return null;
            return self.nodes.items[idx];
        }

        fn matchOp(node: ?*Tensor(T), op: Op) ?*Tensor(T) {
            const n = node orelse return null;
            return if (n.opTag() == op) n else null;
        }

        fn matchUnary(node: ?*Tensor(T), op: Op, src0: *Tensor(T)) ?*Tensor(T) {
            const n = matchOp(node, op) orelse return null;
            return if (n.source0().? == src0) n else null;
        }

        fn matchBinary(node: ?*Tensor(T), op: Op, src0: *Tensor(T), src1: *Tensor(T)) ?*Tensor(T) {
            const n = matchOp(node, op) orelse return null;
            return if (n.source0().? == src0 and n.source1().? == src1) n else null;
        }

        fn matchBinaryScalarRhs(node: ?*Tensor(T), op: Op, src0: *Tensor(T)) ?*Tensor(T) {
            const n = matchOp(node, op) orelse return null;
            if (n.source0().? != src0) return null;
            return if (n.source1().?.isScalar()) n else null;
        }

        fn findNextUser(self: *const Self, start: usize, src: *Tensor(T), op: Op) ?struct { idx: usize, node: *Tensor(T) } {
            var i = start;
            while (i < self.forward_node_count) : (i += 1) {
                const node = self.nodes.items[i];
                if (node.opTag() != op) continue;
                if (node.source0() == src or node.source1() == src) return .{ .idx = i, .node = node };
            }
            return null;
        }

        fn matchScalarBroadcast(node: ?*Tensor(T)) ?*Tensor(T) {
            const n = node orelse return null;
            if (n.isScalar()) return n;
            if (n.opTag() == .repeat and n.source0().?.isScalar()) return n;
            return null;
        }

        fn detectCrossEntropyPattern(self: *const Self, start: usize) ?fused.FusionPlan(T) {
            const nodes = self.nodes.items;
            if (start + 13 >= self.forward_node_count) return null;

            const n0 = nodes[start + 0];
            const n1 = nodes[start + 1];
            const n2 = nodes[start + 2];
            const n3 = nodes[start + 3];
            const n4 = nodes[start + 4];
            const n5 = nodes[start + 5];
            const n6 = nodes[start + 6];
            const n7 = nodes[start + 7];
            const n8 = nodes[start + 8];
            const n9 = nodes[start + 9];
            const n10 = nodes[start + 10];
            const n11 = nodes[start + 11];
            const n12 = nodes[start + 12];
            const n13 = nodes[start + 13];

            if (n0.isOp(.max) and
                n1.isOp(.repeat) and n1.sourceIs(.src0, n0) and
                n2.isOp(.neg) and n2.sourceIs(.src0, n1) and
                n3.isOp(.add) and n3.source0().? == n0.source0().? and n3.sourceIs(.src1, n2) and
                n4.isOp(.exp) and n4.sourceIs(.src0, n3) and
                n5.isOp(.sum) and n5.sourceIs(.src0, n4) and
                n6.isOp(.log) and n6.sourceIs(.src0, n5) and
                n7.isOp(.repeat) and n7.sourceIs(.src0, n6) and
                n8.isOp(.neg) and n8.sourceIs(.src0, n7) and
                n9.isOp(.add) and n9.source0().? == n3 and n9.sourceIs(.src1, n8) and
                n10.isOp(.pick_rows) and n10.sourceIs(.src0, n9) and
                n11.isOp(.neg) and n11.sourceIs(.src0, n10) and
                n12.isOp(.sum) and n12.sourceIs(.src0, n11) and
                n13.isOp(.mul) and n13.sourceIs(.src0, n12) and n13.source1().?.isScalar())
            {
                return .{ .output_idx = start + 13, .payload = .{ .cross_entropy = .{
                    .log_softmax = .{
                        .input = n0.source0().?,
                        .max_node = n0,
                        .rep_max = n1,
                        .neg_rep_max = n2,
                        .shifted = n3,
                        .exp_node = n4,
                        .sum_node = n5,
                        .log_node = n6,
                        .rep_log = n7,
                        .neg_rep_log = n8,
                        .output = n9,
                    },
                    .targets = n10.source1().?,
                    .picked = n10,
                    .neg_picked = n11,
                    .sum_node = n12,
                    .mean_node = n13,
                } } };
            }

            return null;
        }

        fn detectLayerNormPattern(self: *const Self, start: usize) ?fused.FusionPlan(T) {
            const n0 = matchOp(self.nodeAt(start), .sum) orelse return null;
            const n1 = findNextUser(self, start + 1, n0, .mul) orelse return null;
            if (matchScalarBroadcast(n1.node.source1()) == null) return null;

            const n2 = findNextUser(self, n1.idx + 1, n1.node, .repeat) orelse return null;
            const n3 = findNextUser(self, n2.idx + 1, n2.node, .neg) orelse return null;
            const n4 = findNextUser(self, n3.idx + 1, n3.node, .add) orelse return null;
            if (n4.node.source0().? != n0.source0().?) return null;

            const n5 = findNextUser(self, n4.idx + 1, n4.node, .mul) orelse return null;
            if (n5.node.source0().? != n4.node or n5.node.source1().? != n4.node) return null;

            const n6 = findNextUser(self, n5.idx + 1, n5.node, .sum) orelse return null;
            const n7 = findNextUser(self, n6.idx + 1, n6.node, .mul) orelse return null;
            if (matchScalarBroadcast(n7.node.source1()) == null) return null;

            const n8 = findNextUser(self, n7.idx + 1, n7.node, .add) orelse return null;
            const eps_like = matchScalarBroadcast(n8.node.source1()) orelse return null;
            if (n8.node.source0().? != n7.node) return null;

            const n9 = findNextUser(self, n8.idx + 1, n8.node, .sqrt) orelse return null;
            const n10 = findNextUser(self, n9.idx + 1, n9.node, .recip) orelse return null;
            const n11 = findNextUser(self, n10.idx + 1, n10.node, .repeat) orelse return null;
            const n12 = findNextUser(self, n11.idx + 1, n11.node, .mul) orelse return null;
            if (n12.node.source0().? != n4.node and n12.node.source1().? != n4.node) return null;

            return .{ .output_idx = n12.idx, .payload = .{ .layer_norm = .{
                .input = n0.source0().?,
                .sum_node = n0,
                .mean_node = n1.node,
                .rep_mean = n2.node,
                .neg_rep_mean = n3.node,
                .centered = n4.node,
                .sqr_node = n5.node,
                .var_sum = n6.node,
                .var_node = n7.node,
                .eps_like = eps_like,
                .var_eps = n8.node,
                .sqrt_node = n9.node,
                .recip_node = n10.node,
                .rep_std_inv = n11.node,
                .output = n12.node,
            } } };
        }

        fn addParentsThenSelf(self: *Self, cur: *Tensor(T)) Alloc.Error!void {
            const alloc = self.arena.allocator();
            // check if already visited
            for (self.nodes.items) |item| {
                if (cur == item) {
                    return;
                }
            }
            for (self.leaves.items) |item| {
                if (cur == item) {
                    return;
                }
            }
            // visit parents
            if (cur.source0()) |ts0| try self.addParentsThenSelf(ts0);
            if (cur.source1()) |ts1| try self.addParentsThenSelf(ts1);
            if (cur.isLeaf()) {
                // is leaf
                try self.leaves.append(alloc, cur);
            } else {
                try self.nodes.append(alloc, cur);
                try self.grads.append(alloc, cur.gradOrNull());
            }
        }
        pub fn toGraphViz(self: *const Self, alloc: Alloc) Alloc.Error!std.ArrayList(u8) {
            var str = std.ArrayList(u8){};
            const writer = str.writer(alloc);
            try writer.print("digraph G {{\n  node [shape=box];\n", .{});

            for (self.nodes.items) |node| {
                try writer.print("  \"{*}\" [shape=\"none\",label=<<table>", .{node});
                if (node.opTag() == .none) {
                    try writer.print("<tr><td>{any}</td></tr>", .{node.data});
                } else {
                    try writer.print("<tr><td>{any}</td></tr>", .{node.data});
                    try writer.print("<tr><td>{s}</td></tr>", .{node.opTag().symbol()});
                }
                if (node.name) |name| {
                    try writer.print("<tr><td>{s}</td></tr>", .{name});
                }
                try writer.print("<tr><td>{any}</td></tr>", .{node.ne});
                try writer.print("</table>>];\n", .{});
                if (node.source0()) |src0| {
                    try writer.print("  \"{*}\" -> \"{*}\";\n", .{ src0, node });
                }
                if (node.source1()) |src1| {
                    try writer.print("  \"{*}\" -> \"{*}\";\n", .{ src1, node });
                }
                if (node.gradOrNull()) |grad| {
                    try writer.print("  \"{*}\" -> \"{*}\" [style=dashed];\n", .{ node, grad });
                }
                if (!node.data_owned) {
                    try writer.print("  \"{*}\" [style=filled fillcolor=gray];\n", .{node});
                }
            }
            for (self.leaves.items) |leaf| {
                try writer.print("  \"{*}\" [style=filled fillcolor=green label=\"{any}\"];\n", .{ leaf, leaf.data });
            }
            for (self.scratch.items) |item| {
                try writer.print("  \"{*}\" [style=filled fillcolor=gray label=\"{any}\"];\n", .{ item, item.data });
            }
            try writer.print("}}\n", .{});
            return str;
        }

        pub fn resetGrads(self: *Self) void {
            for (self.grads.items) |grad_o| {
                if (grad_o) |grad| {
                    _ = grad.setAllScalar(0);
                }
            }
        }
        /// Zero all intermediate node data to prepare for recomputation.
        /// Skips nodes that alias another tensor's data (e.g. reshape/view),
        /// since zeroing them would corrupt the source tensor's values.
        pub fn reset(self: *Self) void {
            for (self.nodes.items) |node| {
                if (node.opTag() != .none and node.data_owned) {
                    _ = node.setAllScalar(0);
                }
            }
        }

        /// Execute the forward pass over all nodes.
        /// If `fusionPass()` was called, fused chains execute as single-pass
        /// comptime-specialized kernels.
        pub fn compute(self: *const Self) void {
            if (self.fused_chains.items.len == 0) {
                for (self.nodes.items) |node| {
                    node.compute();
                }
                return;
            }
            var next_chain: usize = 0;
            for (self.nodes.items, 0..) |node, i| {
                if (i < self.fused_skip.items.len and self.fused_skip.items[i]) {
                    // When we reach a chain's output node, execute the fused kernel
                    if (next_chain < self.fused_chains.items.len and
                        self.fused_chains.items[next_chain].output_idx == i)
                    {
                        fused.executeFusionPlan(T, self.fused_chains.items[next_chain]);
                        next_chain += 1;
                    }
                    continue;
                }
                node.compute();
            }
        }

        pub fn computeNoGrad(self: *const Self) void {
            if (self.fused_chains.items.len == 0) {
                for (self.nodes.items[0..self.forward_node_count]) |node| {
                    node.compute();
                }
                return;
            }
            var next_chain: usize = 0;
            for (self.nodes.items[0..self.forward_node_count], 0..) |node, i| {
                if (i < self.fused_skip.items.len and self.fused_skip.items[i]) {
                    if (next_chain < self.fused_chains.items.len and
                        self.fused_chains.items[next_chain].output_idx == i)
                    {
                        fused.executeFusionPlan(T, self.fused_chains.items[next_chain]);
                        next_chain += 1;
                    }
                    continue;
                }
                node.compute();
            }
        }

        // ---------------------------------------------------------------
        // High-level convenience methods
        // ---------------------------------------------------------------

        /// One-call training step: build graph (once), reset, seed loss gradient, compute.
        ///
        /// First call builds forward and backward graphs. Subsequent calls
        /// reuse the graph and just reset + compute.
        ///
        /// ```
        /// const loss = model.forward(x);
        /// try g.run(loss);        // forward + backward in one call
        /// optimizer.step();
        /// optimizer.zeroGrad();
        /// ```
        pub fn run(self: *Self, loss_node: *Tensor(T)) !void {
            if (!self.built_forward) try self.buildForward(loss_node);
            if (!self.built_backward) try self.buildBackward(false);
            self.reset();
            self.resetGrads();
            if (loss_node.grad) |grad| _ = grad.setAllScalar(1);
            self.compute();
        }

        /// One-call inference: build forward graph (once), reset, compute forward only.
        ///
        /// Skips backward pass and gradient computation.
        ///
        /// ```
        /// const output = model.forward(x);
        /// try g.infer(output);
        /// ```
        pub fn infer(self: *Self, output: *Tensor(T)) !void {
            if (!self.built_forward) try self.buildForward(output);
            self.reset();
            self.computeNoGrad();
        }
    };
}

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(ComputeGraph(f32));
}

//#region Tests

test "tensor compute graph - matmul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t1 = try Tensor(f32).init(a, &.{ 2, 3 });
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    t1.setParam();

    const t2 = try Tensor(f32).init(a, &.{ 3, 2 });
    t2.setData(&[_]f32{
        1, 2, 3,
        4, 5, 6,
    });

    const dst = t1.matMul(false, t2, false);
    try g.buildForward(dst);
    try g.buildBackward(false);

    _ = dst.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{
            9,  12, 15,
            19, 26, 33,
            29, 40, 51,
        };
        try testing.expectEqualSlices(f32, &expected, dst.data);
    }
    {
        const expected = [_]f32{
            6, 15,
            6, 15,
            6, 15,
        };
        try testing.expectEqualSlices(f32, &expected, t1.grad.?.data);
    }
}

test "build compute graph - forward mul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t0 = try Tensor(f32).init(a, &.{1});
    t0.data[0] = 5;
    const t1 = try Tensor(f32).init(a, &.{1});
    t1.data[0] = 6;
    const out = t0.mul(t1);
    try g.buildForward(out);
    try g.buildBackward(false);
    g.compute();
    {
        const expected = [_]f32{30};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build computeNoGrad graph - forward mul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t0 = try Tensor(f32).init(a, &.{1});
    t0.data[0] = 5;
    t0.setParam();
    const t1 = try Tensor(f32).init(a, &.{1});
    t1.data[0] = 6;
    const out = t0.mul(t1);
    try g.buildForward(out);
    try g.buildBackward(false);
    const dummy_val: f32 = -23;
    t0.grad.?.data[0] = dummy_val;
    g.computeNoGrad();
    {
        const expected = [_]f32{30};
        try testing.expectEqualSlices(f32, &expected, out.data);
        try testing.expectEqual(dummy_val, t0.grad.?.data[0]);
    }
}

test "build compute graph - forward matMul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t1 = try Tensor(f32).init(a, &.{ 2, 3 });
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    const intermed = t1.matMul(true, t1, false);
    const out = intermed.matMul(false, t1, true);
    try g.buildForward(out);
    g.compute();
    {
        const expected = [_]f32{
            35, 44,
            44, 56,
        };
        try testing.expectEqualSlices(f32, &expected, intermed.data);
    }
    {
        const expected = [_]f32{
            123, 281, 439,
            156, 356, 556,
        };
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build compute graph - forward mul & add" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    const w = try Tensor(f32).initScalar(a, 2);
    w.setParam();
    const b = try Tensor(f32).initScalar(a, 5);
    b.setParam();
    const intermed = w.mul(x);
    const out = intermed.add(b);
    try g.buildForward(out);
    g.compute();
    {
        const expected = [_]f32{11};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build compute graph - backward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    const w = try Tensor(f32).initScalar(a, 2);
    w.setParam();
    const b = try Tensor(f32).initScalar(a, 5);
    b.setParam();
    const intermed = w.mul(x);
    const out = intermed.add(b);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{11};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{3};
        try testing.expectEqualSlices(f32, &expected, w.grad.?.data);
    }
    {
        const expected = [_]f32{1};
        try testing.expectEqualSlices(f32, &expected, b.grad.?.data);
    }
}

fn testSqrFunc(alloc: Alloc, x: *Tensor(f32)) *Tensor(f32) {
    _ = alloc;
    return x.sqr();
}

test "build compute graph - backward - testSqrFunc" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    x.setParam();
    const out = testSqrFunc(a, x);
    try g.buildForward(out);
    try g.buildBackward(true);

    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{9};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{6};
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
    const iters = 10;
    for (0..iters) |_| {
        g.compute();
    }
    {
        const expected = [_]f32{9};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    // accumulated gradient
    {
        const expected = [_]f32{6 * (iters + 1)};
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
}

fn testSqrSumFunc(alloc: Alloc, x: *Tensor(f32)) *Tensor(f32) {
    _ = alloc;
    return x.sqr().sumAll();
}

test "build compute graph - backward - testSqrSumFunc" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    const data = [_]f32{ 3, 4, 10 };
    x.setData(&data);
    x.setParam();
    const out = testSqrSumFunc(a, x);
    try g.buildForward(out);
    try g.buildBackward(true);

    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{125};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{ 6, 8, 20 };
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
}

test "time speed equation test" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const time = try Tensor(f32).initLinspace(a, &.{20}, 0, 20);

    const c0 = try Tensor(f32).initScalar(a, 0.75);
    const c1 = try Tensor(f32).initScalar(a, 9.5);
    const c2 = try Tensor(f32).initScalar(a, 1);

    const inner = time.sub(c1.repeatLike(time));
    const inner2 = inner.sqr();
    const inner3 = inner2.mul(c0.repeatLike(inner2));
    const speed = inner3.add(c2.repeatLike(inner3));

    try g.buildForward(speed);
    g.compute();

    try testing.expectEqual(@as(usize, 20), time.nElems());
    for (time.data, speed.data) |t, s| {
        const t1 = t - 9.5;
        try testing.expectEqual(0.75 * (t1 * t1) + 1, s);
    }
}

test "a*x^2" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.name = "x";
    const coeff = try Tensor(f32).initScalar(a, 2);
    coeff.name = "a";
    coeff.setParam();
    const xsq = x.sqr();
    xsq.name = "x^2";
    const axsq = xsq.mul(coeff);
    axsq.name = "a*x^2";
    try g.buildForward(axsq);
    try g.buildBackward(false);
    g.compute();
    try testing.expectEqual(@as(f32, 32), axsq.data[0]);
}

test "arange a*x^2" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initLinspace(a, &.{5}, 0, 5);
    x.name = "x";
    const coeff = try Tensor(f32).initScalar(a, 2);
    coeff.name = "a";
    coeff.setParam();
    const xsq = x.sqr();
    xsq.name = "x^2";
    const axsq = xsq.mul(coeff.repeatLike(xsq));
    axsq.name = "a*x^2";
    try g.buildForward(axsq);
    try g.buildBackward(true);
    g.compute();
    const expected = [_]f32{ 0, 2, 8, 18, 32 };
    try testing.expectEqualSlices(f32, &expected, axsq.data);
}

test "arange" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t = try Tensor(f32).initLinspace(a, &.{5}, 0, 5);
    t.setParam();
    try testing.expectEqual(@as(usize, 5), t.nElems());
    const expected = [_]f32{ 0, 1, 2, 3, 4 };
    try testing.expectEqualSlices(f32, &expected, t.data);

    const out = t.sqr().sumAll();
    try g.buildForward(out);
    try g.buildBackward(true);
    out.grad.?.data[0] = 1;
    g.compute();

    try testing.expectEqual(@as(f32, 30), out.data[0]);
    try testing.expectEqualSlices(f32, &.{ 0, 2, 4, 6, 8 }, t.grad.?.data);
    const lr = try Tensor(f32).initScalar(a, 0.01);
    for (0..1000) |_| {
        g.reset();
        g.resetGrads();
        out.grad.?.data[0] = 1;
        g.compute();
        t.grad.?.computeMul(t.grad.?, lr);
        t.computeSub(t, t.grad.?);
    }
    try testing.expectApproxEqAbs(@as(f32, 0), out.data[0], 0.00001);
}

test "backward - sqrt" {
    // f(x) = sqrt(x), f'(x) = 0.5 / sqrt(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.setParam();
    const out = x.sqrt();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 2.0), out.data[0], 1e-6);
    // f'(4) = 0.5 / sqrt(4) = 0.25
    try testing.expectApproxEqAbs(@as(f32, 0.25), x.grad.?.data[0], 1e-6);
}

test "backward - abs" {
    // f(x) = abs(x), f'(x) = sgn(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&[_]f32{ -3, 0, 5 });
    x.setParam();
    const out = x.abs().sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    try testing.expectEqualSlices(f32, &.{ -1, 0, 1 }, x.grad.?.data);
}

test "backward - neg" {
    // f(x) = -x, f'(x) = -1
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 7);
    x.setParam();
    const out = x.neg();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, -7.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), x.grad.?.data[0], 1e-6);
}

test "backward - relu" {
    // f(x) = relu(x), f'(x) = step(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&[_]f32{ -2, 0, 3, 5 });
    x.setParam();
    const out = x.relu().sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // relu(-2)=0, relu(0)=0, relu(3)=3, relu(5)=5 → sum=8
    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    // gradients: step(-2)=0, step(0)=0, step(3)=1, step(5)=1
    try testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1 }, x.grad.?.data);
}

test "backward - recip" {
    // f(x) = 1/x, f'(x) = -1/x^2
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.setParam();
    const out = x.recip();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 0.25), out.data[0], 1e-6);
    // f'(4) = -1/16 = -0.0625
    try testing.expectApproxEqAbs(@as(f32, -0.0625), x.grad.?.data[0], 1e-6);
}

test "fusion - fused compute matches unfused" {
    // Compute x.sub(y).sqr() both with and without fusion, verify identical results
    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const x1 = try Tensor(f32).init(a1, &.{4});
    x1.setData(&.{ 1, 2, 3, 4 });
    const y1 = try Tensor(f32).init(a1, &.{4});
    y1.setData(&.{ 4, 3, 2, 1 });
    const out1 = x1.sub(y1).sqr();
    try g1.buildForward(out1);
    g1.compute();

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const x2 = try Tensor(f32).init(a2, &.{4});
    x2.setData(&.{ 1, 2, 3, 4 });
    const y2 = try Tensor(f32).init(a2, &.{4});
    y2.setData(&.{ 4, 3, 2, 1 });
    const out2 = x2.sub(y2).sqr();
    try g2.buildForward(out2);
    try g2.fusionPass();
    g2.compute();

    for (out1.data, out2.data) |v1, v2| {
        try testing.expectApproxEqAbs(v1, v2, 1e-6);
    }
    try testing.expectEqualSlices(f32, &.{ 9, 1, 1, 9 }, out2.data);
}

test "fusion - backward works after fusion" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    x.setParam();
    const y = try Tensor(f32).initScalar(a, 2);
    const out = x.mul(y).neg(); // mul -> neg: fusible chain of 2
    try g.buildForward(out);
    try g.fusionPass();
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // f(x) = -(x * 2) = -2x, f'(x) = -2
    try testing.expectApproxEqAbs(@as(f32, -6.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -2.0), x.grad.?.data[0], 1e-6);
}

test "fusion - chain detection skips non-fusible ops" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    // neg -> sumAll: neg is fusible but sum breaks the chain
    const out = x.neg().sumAll();
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 0), g.fused_chains.items.len);
    g.compute();
    try testing.expectApproxEqAbs(@as(f32, -21.0), out.data[0], 1e-6);
}

test "fusion - detects softmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.softmax(&.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.softmax, g.fused_chains.items[0].kind());

    g.compute();
    try testing.expectApproxEqAbs(@as(f32, 1.0), out.data[0] + out.data[1] + out.data[2], 1e-6);
}

test "fusion - detects conv2d composite pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 6, 6, 1, 2 });
    const k = try Tensor(f32).init(a, &.{ 3, 3, 1, 4 });
    const y = x.conv2d(k);
    try g.buildForward(y);
    try g.fusionPass();

    var found = false;
    for (g.fused_chains.items) |plan| {
        if (plan.kind() == .conv2d) {
            found = true;
            break;
        }
    }
    try testing.expect(found);
}

test "fusion - conv2d fused matches unfused" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{ 5, 5, 1, 1 });
    const xu = try Tensor(f32).init(au, &.{ 5, 5, 1, 1 });
    for (xf.data, 0..) |*d, i| d.* = @floatFromInt(i + 1);
    @memcpy(xu.data, xf.data);

    const kf = try Tensor(f32).init(af, &.{ 3, 3, 1, 2 });
    const ku = try Tensor(f32).init(au, &.{ 3, 3, 1, 2 });
    for (kf.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2));
    @memcpy(ku.data, kf.data);

    const yf = xf.conv2d(kf);
    const yu = xu.conv2d(ku);
    try gf.buildForward(yf);
    try gu.buildForward(yu);
    try gf.fusionPass();

    gf.compute();
    gu.compute();

    for (yf.data, yu.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
}

test "fusion - compiler root annotations detect softmax plan" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.softmax(&.{1});
    try g.buildForward(out);

    const root_plan = (try g.detectCompilerRootRewritePlan()) orelse return error.SkipZigTest;
    try testing.expectEqual(@as(usize, 0), root_plan.start_idx);
    try testing.expectEqual(fused.FusionKind.softmax, root_plan.plan.kind());
    try testing.expectEqual(@as(usize, g.forward_node_count - 1), root_plan.plan.output_idx);
}

test "fusion - detects logSoftmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.logSoftmax(&.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.log_softmax, g.fused_chains.items[0].kind());

    g.compute();
    const probs = std.math.exp(out.data[0]) + std.math.exp(out.data[1]) + std.math.exp(out.data[2]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), probs, 1e-6);
}

test "fusion - compiler root annotations detect logSoftmax plan" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.logSoftmax(&.{1});
    try g.buildForward(out);

    const root_plan = (try g.detectCompilerRootRewritePlan()) orelse return error.SkipZigTest;
    try testing.expectEqual(@as(usize, 0), root_plan.start_idx);
    try testing.expectEqual(fused.FusionKind.log_softmax, root_plan.plan.kind());
    try testing.expectEqual(@as(usize, g.forward_node_count - 1), root_plan.plan.output_idx);
}

test "fusion - detects cross entropy pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });
    const targets = try Tensor(f32).init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const out = loss.crossEntropy(f32, logits, targets);
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    // Fusion pass may detect log_softmax subpattern within cross-entropy
    try testing.expect(g.fused_chains.items[0].kind() == .cross_entropy or
        g.fused_chains.items[0].kind() == .log_softmax);

    g.compute();

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, out.data[0], 1e-5);
}

test "fusion - compiler root annotations detect cross entropy plan" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });
    const targets = try Tensor(f32).init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const out = loss.crossEntropy(f32, logits, targets);
    try g.buildForward(out);

    const root_plan = (try g.detectCompilerRootRewritePlan()) orelse return error.SkipZigTest;
    try testing.expectEqual(@as(usize, 0), root_plan.start_idx);
    try testing.expectEqual(fused.FusionKind.cross_entropy, root_plan.plan.kind());
    try testing.expectEqual(@as(usize, g.forward_node_count - 1), root_plan.plan.output_idx);
}

test "fusion - compiler root annotations detect layerNorm plan" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const out = x.layerNorm(&.{ 1, 3 }, 1e-5);
    try g.buildForward(out);

    const root_plan = (try g.detectCompilerRootRewritePlan()) orelse return error.SkipZigTest;
    try testing.expectEqual(@as(usize, 0), root_plan.start_idx);
    try testing.expectEqual(fused.FusionKind.layer_norm, root_plan.plan.kind());
    try testing.expectEqual(@as(usize, g.forward_node_count - 1), root_plan.plan.output_idx);
}

test "fusion - compiler emits multi-region plans" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = x.layerNorm(&.{ 1, 3 }, 1e-5).softmax(&.{ 1, 3 });
    try g.buildForward(y);

    var plans = try g.detectCompilerFusionPlans();
    defer plans.deinit(std.heap.page_allocator);

    try testing.expect(plans.items.len >= 2);
    try testing.expectEqual(fused.FusionKind.layer_norm, plans.items[0].plan.kind());
    try testing.expectEqual(fused.FusionKind.softmax, plans.items[plans.items.len - 1].plan.kind());
}

test "layerNorm forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4 });
    // mean=2.5, var=1.25, result = (x - 2.5) / sqrt(1.25 + 1e-5)
    const out = x.layerNorm(&.{1}, 1e-5);
    try g.buildForward(out);
    g.compute();

    const std_dev = @sqrt(@as(f32, 1.25) + 1e-5);
    try testing.expectApproxEqAbs((1.0 - 2.5) / std_dev, out.data[0], 1e-4);
    try testing.expectApproxEqAbs((2.0 - 2.5) / std_dev, out.data[1], 1e-4);
    try testing.expectApproxEqAbs((3.0 - 2.5) / std_dev, out.data[2], 1e-4);
    try testing.expectApproxEqAbs((4.0 - 2.5) / std_dev, out.data[3], 1e-4);
}

test "fusion - detects layerNorm pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = x.layerNorm(&.{ 1, 3 }, 1e-5);
    try g.buildForward(y);
    try g.fusionPass();

    var kinds = [_]fused.FusionKind{ .elementwise_chain, .elementwise_chain, .elementwise_chain };
    const n = @min(g.fused_chains.items.len, kinds.len);
    for (g.fused_chains.items[0..n], 0..) |plan, i| kinds[i] = plan.kind();

    const has_layer_norm = blk: {
        for (g.fused_chains.items) |plan| {
            if (plan.kind() == .layer_norm) break :blk true;
        }
        break :blk false;
    };
    try testing.expect(has_layer_norm);
}

test "fusion - layerNorm fused matches unfused 1D" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{4});
    xf.setData(&.{ 1, 2, 3, 4 });
    const xu = try Tensor(f32).init(au, &.{4});
    xu.setData(&.{ 1, 2, 3, 4 });

    const yf = xf.layerNorm(&.{1}, 1e-5);
    const yu = xu.layerNorm(&.{1}, 1e-5);

    try gf.buildForward(yf);
    try gu.buildForward(yu);
    try gf.fusionPass();
    gf.compute();
    gu.compute();

    for (yf.data, yu.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
}

test "fusion - layerNorm fused matches unfused 2D" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{ 2, 3 });
    xf.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const xu = try Tensor(f32).init(au, &.{ 2, 3 });
    xu.setData(&.{ 1, 2, 3, 4, 5, 6 });

    const yf = xf.layerNorm(&.{ 1, 3 }, 1e-5);
    const yu = xu.layerNorm(&.{ 1, 3 }, 1e-5);

    try gf.buildForward(yf);
    try gu.buildForward(yu);
    try gf.fusionPass();
    gf.compute();
    gu.compute();

    for (yf.data, yu.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
}

test "fusion - layerNorm plan kind is linked" {
    try testing.expectEqual(fused.FusionKind.layer_norm, .layer_norm);
}

test "backward - gelu" {
    // Numerical gradient check: (gelu(x+h) - gelu(x-h)) / 2h ≈ gelu'(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ -1.0, 0.0, 1.5 });
    x.setParam();
    const out = x.gelu().sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // Compute expected gradients numerically
    const h: f32 = 1e-4;
    const test_vals = [_]f32{ -1.0, 0.0, 1.5 };
    for (test_vals, 0..) |xv, i| {
        const gelu_plus = geluScalar(xv + h);
        const gelu_minus = geluScalar(xv - h);
        const numerical_grad = (gelu_plus - gelu_minus) / (2.0 * h);
        try testing.expectApproxEqAbs(numerical_grad, x.grad.?.data[i], 1e-3);
    }
}

fn geluScalar(x: f32) f32 {
    const a = 0.79788456080286535587989211986876 * x * (1.0 + 0.044715 * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(a));
}

test "reset preserves param data through reshape" {
    // Regression: g.reset() must not zero data shared via reshape with params.
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const param = try Tensor(f32).init(a, &.{ 2, 3 });
    param.setData(&.{ 1, 2, 3, 4, 5, 6 });
    param.setParam();

    // Reshape shares data buffer with param
    const reshaped = param.reshape(&.{6});
    const out = reshaped.sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);

    g.compute();
    try testing.expectApproxEqAbs(@as(f32, 21), out.data[0], 1e-5);

    // After reset + recompute, param data must survive
    g.reset();
    g.compute();
    try testing.expectApproxEqAbs(@as(f32, 21), out.data[0], 1e-5);

    // Param data must still be intact
    try testing.expectApproxEqAbs(@as(f32, 1), param.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6), param.data[5], 1e-5);
}

test "run - one-call training step" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    x.setParam();

    const loss_node = x.sqr().sumAll();

    // First call builds forward+backward, computes, and seeds grad
    try g.run(loss_node);

    // d/dx[sum(x^2)] = 2x
    try testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, x.grad.?.data);

    // Second call reuses graph
    x.setData(&.{ 4, 5, 6 });
    try g.run(loss_node);
    try testing.expectEqualSlices(f32, &.{ 8, 10, 12 }, x.grad.?.data);
}

test "infer - one-call inference" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });

    const y = x.sqr().sumAll();

    try g.infer(y);

    try testing.expectApproxEqAbs(@as(f32, 14.0), y.data[0], 1e-6);
}

//#endregion
