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
/// can free them in bulk via the arena. Graph discovery uses a visited set,
/// so shared subgraphs are recorded once even when reachable through multiple
/// parent paths.
pub fn ComputeGraph(comptime T: type) type {
    return struct {
        const Self = @This();

        built_forward: bool = false,
        built_backward: bool = false,
        built_fusion: bool = false,
        forward_node_count: usize = 0,

        arena: std.heap.ArenaAllocator,
        nodes: std.ArrayList(*Tensor(T)),
        grads: std.ArrayList(?*Tensor(T)),
        leaves: std.ArrayList(*Tensor(T)),
        visited_nodes: std.AutoHashMapUnmanaged(*Tensor(T), void),
        scratch: std.ArrayList(*Tensor(T)),

        /// Fusion state — populated by `fusionPass()`.
        fused_chains: std.ArrayList(fused.FusionPlan(T)),
        /// Per-node flag: true means this node is part of a fused chain
        /// and should be skipped during normal compute iteration.
        fused_skip: std.ArrayList(bool),
        execution_steps: std.ArrayList(ExecutionStep),
        forward_execution_steps: std.ArrayList(ExecutionStep),

        const ExecutionStep = union(enum) {
            fusion: usize,
            node: *Tensor(T),
        };

        /// Set up resources for compute graph.
        /// Must call `buildForward` (then optionally `buildBackward`) to be able to do computation.
        pub fn init(backing_alloc: Alloc) Self {
            return .{
                .arena = std.heap.ArenaAllocator.init(backing_alloc),
                .nodes = .{},
                .grads = .{},
                .leaves = .{},
                .visited_nodes = .empty,
                .scratch = .{},
                .fused_chains = .{},
                .fused_skip = .{},
                .execution_steps = .{},
                .forward_execution_steps = .{},
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
            self.deinitFusedChains();
            self.fused_skip.deinit(alloc);
            self.execution_steps.deinit(alloc);
            self.forward_execution_steps.deinit(alloc);
            self.nodes.deinit(alloc);
            self.grads.deinit(alloc);
            self.leaves.deinit(alloc);
            self.visited_nodes.deinit(alloc);
            self.scratch.deinit(alloc);
            self.arena.deinit();
        }

        /// Build a graph where the provided tensor is the final output node.
        /// Shared subgraphs are deduplicated during the traversal.
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
            if (n_change > 0) self.invalidateExecutionPlans();
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
                if (node.isParam()) {
                    assert(node.hasGrad());
                    try self.buildForwardHelper(node.gradOrNull().?);
                }
            }

            self.built_backward = true;
            self.resetGrads();
            self.invalidateExecutionPlans();
        }

        /// Detect chains of fusible elementwise ops in the built graph
        /// and prepare comptime-specialized kernels for them.
        ///
        /// Call after `buildForward()` (and optionally after `buildBackward()`).
        /// Lowering applies to the full executable graph, so backward regions
        /// can be fused too once they have been built.
        pub fn fusionPass(self: *Self) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const node_count = self.nodes.items.len;
            if (node_count < 2) return;

            var compiler_plans = try self.detectCompilerFusionPlansWithAlloc(alloc);
            defer self.deinitCompilerPlans(alloc, &compiler_plans);

            self.deinitFusedChains();
            self.fused_skip.clearRetainingCapacity();

            // Initialize skip bitmap
            self.fused_skip = try std.ArrayList(bool).initCapacity(alloc, node_count);
            self.fused_skip.items.len = node_count;
            @memset(self.fused_skip.items, false);

            // Apply compiler plans to the skip bitmap and fused chains
            {
                var ci: usize = 0;
                var next_compiler_plan: usize = 0;
                while (ci < node_count) : (ci += 1) {
                    if (next_compiler_plan < compiler_plans.items.len) {
                        const compiler_plan = compiler_plans.items[next_compiler_plan];
                        if (compiler_plan.start_idx == ci) {
                            for (ci..compiler_plan.plan.output_idx + 1) |idx| self.fused_skip.items[idx] = true;
                            try self.fused_chains.append(alloc, try fused.cloneFusionPlan(T, alloc, compiler_plan.plan));
                            ci = compiler_plan.plan.output_idx;
                            next_compiler_plan += 1;
                            continue;
                        }
                    }
                }
            }

            // Build node → index map (used by backward conv detection and elementwise chains)
            var ptr_to_idx = std.AutoHashMap(*Tensor(T), usize).init(alloc);
            defer ptr_to_idx.deinit();
            try ptr_to_idx.ensureTotalCapacity(@intCast(node_count));
            for (self.nodes.items, 0..) |node, idx| {
                ptr_to_idx.putAssumeCapacity(node, idx);
            }

            // Detect backward conv2d patterns directly on the tensor graph.
            // scatter_add_view nodes in the backward pass are produced by the
            // backward rule for as_strided. When the as_strided came from a
            // conv2d decomposition, the scatter implements either bwd_input or
            // bwd_kernel depending on which conv operand the view referenced.
            for (self.nodes.items[self.forward_node_count..], self.forward_node_count..) |node, idx| {
                if (self.fused_skip.items[idx]) continue;
                if (detectConv2dBwdPlan(node, idx)) |_| {
                    // TODO: fused kernel produces wrong results; detection works
                    // but execution needs debugging. Skipping for now.
                }
            }

            const use_count = try alloc.alloc(u16, node_count);
            @memset(use_count, 0);
            for (self.nodes.items) |node| {
                if (node.source0()) |src| {
                    if (ptr_to_idx.get(src)) |j| use_count[j] += 1;
                }
                if (node.source1()) |src| {
                    if (ptr_to_idx.get(src)) |j| use_count[j] += 1;
                }
            }

            // Detect elementwise chains (covers both forward and backward)
            {
                var i: usize = 0;
                while (i < node_count) : (i += 1) {
                    const node = self.nodes.items[i];
                    if (!node.opTag().isFusible()) continue;
                    if (self.fused_skip.items[i]) continue;
                    // For binary ops, non-chain source must be scalar or
                    // same-shaped AND contiguous (fused kernel indexes data[i] directly).
                    if (node.source1()) |src1| {
                        if (!src1.isScalar() and !(node.isSameShape(src1) and src1.isContiguous() and src1.data.len == node.nElems())) continue;
                    }

                    const chain_start = i;
                    var chain_end = i;
                    var extending = true;
                    while (extending) {
                        extending = false;
                        if (chain_end + 1 < node_count) {
                            const next = self.nodes.items[chain_end + 1];
                            const fusible = next.opTag().isFusible() and
                                !self.fused_skip.items[chain_end + 1] and
                                next.source0().? == self.nodes.items[chain_end] and
                                next.isSameShape(self.nodes.items[chain_end]) and
                                use_count[chain_end] == 1;
                            // For binary ops, the non-chain source must be scalar
                            // or same-shaped to be safely indexable in the fused kernel.
                            const binary_ok = if (next.source1()) |src1|
                                (src1.isScalar() or (next.isSameShape(src1) and src1.isContiguous() and src1.data.len == next.nElems()))
                            else
                                true;
                            if (fusible and binary_ok) {
                                chain_end += 1;
                                extending = true;
                            }
                        }
                    }

                    const chain_len = chain_end - chain_start + 1;
                    if (chain_len < 2) continue;

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

                    i = chain_end;
                }
            }

            // Pre-allocate scratch buffers from the arena so conv2d plans
            // don't hit mmap/munmap on every execution.
            for (self.fused_chains.items) |*chain| {
                try chain.allocScratchBuffers(alloc);
            }

            try self.buildExecutionSteps(alloc, node_count, &self.execution_steps);
            try self.buildExecutionSteps(alloc, self.forward_node_count, &self.forward_execution_steps);
        }

        /// Detect conv2d backward pattern from a scatter_add_view node.
        ///
        /// The backward rule for `as_strided` produces `out_grad.scatterAddView(view)`.
        /// For conv2d decompositions, the view (scatter.src1) is a 7D as_strided of a 4D
        /// tensor. We walk the gradient chain (scatter.src0) to find the mul and broadcast
        /// nodes, then determine direction based on whether the view's source is a
        /// parameter (bwd_kernel: gradient w.r.t. kernel) or not (bwd_input).
        ///
        /// The gradient chain has add-accumulation nodes interleaved. We use `findPastAdd`
        /// to look through exactly one level of add on each step.
        fn detectConv2dBwdPlan(scatter: *Tensor(T), scatter_idx: usize) ?fused.FusionPlan(T) {
            if (scatter.opTag() != .scatter_add_view) return null;

            // scatter.src1 must be an as_strided view of a 4D tensor (from forward conv2d)
            const view = scatter.source1() orelse return null;
            if (view.opTag() != .as_strided) return null;
            const view_source = view.source0() orelse return null;
            if (view_source.n_dims != 4) return null;
            // Conv2d decomposition uses 7D views
            if (view.n_dims != 7) return null;

            // Walk gradient chain: scatter.src0 → [add] → mul
            const mul_node = findPastAdd(scatter.source0() orelse return null, .mul) orelse return null;

            // mul has two operands (each possibly behind one add):
            //   - a broadcast_to (gradient path from output_grad)
            //   - an as_strided (the other conv2d operand from forward)
            const bcast = findPastAdd(mul_node.source0() orelse return null, .broadcast_to) orelse
                findPastAdd(mul_node.source1() orelse return null, .broadcast_to) orelse return null;

            const other_view = findPastAdd(mul_node.source0() orelse return null, .as_strided) orelse
                findPastAdd(mul_node.source1() orelse return null, .as_strided) orelse return null;
            const other_source = other_view.source0() orelse return null;
            if (other_source.n_dims != 4) return null;

            // Walk broadcast → [add] → reshape → output_grad (4D)
            const reshape = findPastAdd(bcast.source0() orelse return null, .reshape) orelse return null;
            const output_grad = reshape.source0() orelse return null;
            if (output_grad.n_dims != 4) return null;

            // Determine direction: if view_source is a parameter, this scatter
            // writes the kernel gradient (bwd_kernel). Otherwise it's bwd_input.
            if (view_source.isParam()) {
                // bwd_kernel: scatter writes to kernel grad, other_view is the input
                return .{ .output_idx = scatter_idx, .payload = .{ .conv2d_bwd_kernel = .{
                    .input = other_source,
                    .output_grad = output_grad,
                    .reshape_node = reshape,
                    .repeat_node = bcast,
                    .mul_node = mul_node,
                    .output = scatter,
                } } };
            } else {
                // bwd_input: scatter writes to input grad, other_view is the kernel
                return .{ .output_idx = scatter_idx, .payload = .{ .conv2d_bwd_input = .{
                    .output_grad = output_grad,
                    .kernel = other_source,
                    .reshape_node = reshape,
                    .repeat_node = bcast,
                    .mul_node = mul_node,
                    .output = scatter,
                } } };
            }
        }

        /// Look through one level of add-accumulation to find a node with the target op.
        fn findPastAdd(start: *Tensor(T), target_op: Op) ?*Tensor(T) {
            if (start.opTag() == target_op) return start;
            if (start.opTag() != .add) return null;
            if (start.source0()) |s| { if (s.opTag() == target_op) return s; }
            if (start.source1()) |s| { if (s.opTag() == target_op) return s; }
            return null;
        }

        /// Mark the scatter_add_view node as skipped. The fused kernel
        /// replaces the scatter computation; intermediate backward nodes
        /// (reshape, broadcast, mul) still run but their results are unused
        /// by the fused kernel.
        fn markBwdConvSkip(self: *Self, _: *Tensor(T), scatter_idx: usize, _: *const std.AutoHashMap(*Tensor(T), usize)) void {
            self.fused_skip.items[scatter_idx] = true;
        }

        const CompilerPlan = struct {
            start_idx: usize,
            plan: fused.FusionPlan(T),
        };

        fn fromMappedPlan(mapped: fused.CompilerMappedPlan(T)) CompilerPlan {
            return .{ .start_idx = mapped.start_idx, .plan = mapped.plan };
        }

        fn sortCompilerPlans(_: void, lhs: CompilerPlan, rhs: CompilerPlan) bool {
            return lhs.start_idx < rhs.start_idx;
        }

        fn detectCompilerFusionPlans(self: *const Self) Alloc.Error!std.ArrayList(CompilerPlan) {
            return self.detectCompilerFusionPlansWithAlloc(std.heap.page_allocator);
        }

        fn detectCompilerFusionPlansWithAlloc(self: *const Self, plan_alloc: Alloc) Alloc.Error!std.ArrayList(CompilerPlan) {
            var plans = std.ArrayList(CompilerPlan){};
            errdefer self.deinitCompilerPlans(plan_alloc, &plans);

            if (self.forward_node_count == 0) return plans;

            var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer temp_arena.deinit();

            var pipeline = compiler.Pipeline(T).init(temp_arena.allocator());
            defer pipeline.deinit();

            // Lower forward root
            pipeline.lowerOneRoot(self.nodes.items[self.forward_node_count - 1]) catch |err| switch (err) {
                error.UnsupportedOp => return plans,
                error.OutOfMemory => return error.OutOfMemory,
            };

            // Lower backward roots (parameter gradients)
            if (self.built_backward) {
                for (self.nodes.items[0..self.forward_node_count]) |node| {
                    if (node.isParam()) {
                        if (node.gradOrNull()) |grad| {
                            pipeline.lowerOneRoot(grad) catch |err| switch (err) {
                                error.UnsupportedOp => continue,
                                error.OutOfMemory => return error.OutOfMemory,
                            };
                        }
                    }
                }
            }

            // Run post-lowering pipeline stages
            pipeline.finalize() catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
            };

            const value_to_tensor = try self.buildValueToTensorMap(temp_arena.allocator(), &pipeline.lowering, &pipeline.canonical.?);
            const all_nodes = self.nodes.items;

            for (pipeline.schedule.?.steps.items) |step| {
                const mapped = switch (step) {
                    .kernel_pattern => |s| fused.mapCompilerPattern(T, self.nodes.items, value_to_tensor, pipeline.kernel.?.patterns.items[s.pattern_index].pattern),
                    else => null,
                } orelse continue;
                try plans.append(plan_alloc, fromMappedPlan(mapped));
            }

            // Schedule-owned elementwise regions (forward-only — backward
            // chains use the simpler graph-level chain detector instead).
            for (pipeline.schedule.?.steps.items) |step| {
                const mapped = switch (step) {
                    .schedule_region => |s| fused.mapCompilerRegion(T, plan_alloc, all_nodes, value_to_tensor, pipeline.schedule.?.regions.items[s.region_index]),
                    else => null,
                } orelse continue;
                if (mapped.start_idx >= self.forward_node_count) continue;
                if (compilerPlanOverlaps(plans.items, mapped.start_idx, mapped.plan.output_idx)) continue;
                if (!isElementwiseChainSafe(mapped.plan)) continue;
                try plans.append(plan_alloc, fromMappedPlan(mapped));
            }

            std.mem.sort(CompilerPlan, plans.items, {}, sortCompilerPlans);
            return plans;
        }

        fn deinitCompilerPlans(self: *const Self, alloc: Alloc, plans: *std.ArrayList(CompilerPlan)) void {
            _ = self;
            for (plans.items) |plan| {
                fused.deinitFusionPlan(T, alloc, plan.plan);
            }
            plans.deinit(alloc);
        }

        /// Validate that every binary op in an elementwise chain has a
        /// non-chain operand that is either scalar or a same-shaped contiguous
        /// buffer with one element per output position. This guards against
        /// broadcast operands that would cause out-of-bounds accesses in the
        /// fused kernel.
        fn isElementwiseChainSafe(plan: fused.FusionPlan(T)) bool {
            const chain = switch (plan.payload) {
                .elementwise_chain => |c| c,
                else => return true,
            };
            for (chain.nodes, 0..) |node, idx| {
                const other = chain.otherOperand(idx) orelse continue;
                if (!other.isScalar() and !(node.isSameShape(other) and other.isContiguous() and other.data.len == node.nElems())) {
                    return false;
                }
            }
            for (chain.nodes) |node| {
                if (node.data.len != chain.input.nElems()) {
                    return false;
                }
            }
            return true;
        }

        fn compilerPlanOverlaps(existing: []const CompilerPlan, start_idx: usize, output_idx: usize) bool {
            for (existing) |plan| {
                if (start_idx <= plan.plan.output_idx and output_idx >= plan.start_idx) return true;
            }
            return false;
        }

        fn nextFusedChainAtOrAfter(self: *const Self, start: usize, limit: usize) ?usize {
            for (self.fused_chains.items, 0..) |plan, i| {
                if (plan.output_idx < start) continue;
                if (plan.output_idx >= limit) return null;
                return i;
            }
            return null;
        }

        pub const FusionSummary = struct {
            node_count: usize,
            forward_node_count: usize,
            fused_region_count: usize,
        };

        pub fn fusionSummary(self: *const Self) FusionSummary {
            return .{
                .node_count = self.nodes.items.len,
                .forward_node_count = self.forward_node_count,
                .fused_region_count = self.fused_chains.items.len,
            };
        }

        pub fn dumpFusionReport(self: *const Self, writer: anytype) !void {
            const summary = self.fusionSummary();
            try writer.print("nodes total={} forward={} fused_regions={}\n", .{
                summary.node_count,
                summary.forward_node_count,
                summary.fused_region_count,
            });

            for (self.fused_chains.items, 0..) |plan, i| {
                try writer.print("fused[{d}] kind={s} output_idx={}\n", .{ i, @tagName(plan.kind()), plan.output_idx });
            }

            for (self.nodes.items, 0..) |node, i| {
                try writer.print("node[{d}] {s} shape={any}\n", .{ i, @tagName(node.opTag()), node.ne[0..node.n_dims] });
            }
        }

        pub fn dumpTensorLineage(self: *const Self, writer: anytype, needle: *Tensor(T)) !void {
            const idx = fused.indexOfNodeMaybe(T, self.nodes.items, needle) orelse {
                try writer.print("tensor not found in graph\n", .{});
                return;
            };
            try writer.print("lineage for node[{d}] {s} shape={any}\n", .{ idx, @tagName(needle.opTag()), needle.ne[0..needle.n_dims] });

            var visited = std.AutoHashMap(*Tensor(T), void).init(std.heap.page_allocator);
            defer visited.deinit();
            try self.dumpTensorLineageRecur(writer, needle, 1, &visited);
        }

        fn dumpTensorLineageRecur(self: *const Self, writer: anytype, node: *Tensor(T), depth: usize, visited: *std.AutoHashMap(*Tensor(T), void)) !void {
            const gop = try visited.getOrPut(node);
            if (gop.found_existing) return;
            gop.value_ptr.* = {};

            const srcs = [_]?*Tensor(T){ node.source0(), node.source1() };
            for (srcs) |src_o| {
                const src = src_o orelse continue;
                const node_idx = fused.indexOfNodeMaybe(T, self.nodes.items, src) orelse continue;
                var j: usize = 0;
                while (j < depth * 2) : (j += 1) try writer.writeByte(' ');
                try writer.print("node[{d}] {s} shape={any}\n", .{ node_idx, @tagName(src.opTag()), src.ne[0..src.n_dims] });
                try self.dumpTensorLineageRecur(writer, src, depth + 1, visited);
            }
        }

        fn detectCompilerRootRewritePlan(self: *const Self) Alloc.Error!?CompilerPlan {
            var plans = try self.detectCompilerFusionPlans();
            defer plans.deinit(std.heap.page_allocator);
            if (plans.items.len == 0) return null;
            return plans.items[plans.items.len - 1];
        }

        fn buildValueToTensorMap(self: *const Self, alloc: Alloc, lowering: *const compiler.Lowering(T), canonical: *const compiler.CanonicalGraph) Alloc.Error![]const ?*Tensor(T) {
            const len = canonical.values.items.len;
            const mapping = try alloc.alloc(?*Tensor(T), len);
            @memset(mapping, null);

            var it = lowering.memo.iterator();
            while (it.next()) |entry| {
                const canonical_id = canonical.remapSourceValue(entry.value_ptr.*) orelse continue;
                mapping[@intFromEnum(canonical_id)] = @constCast(entry.key_ptr.*);
            }

            _ = self;
            return mapping;
        }

        fn addParentsThenSelf(self: *Self, cur: *Tensor(T)) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const visited = try self.visited_nodes.getOrPut(alloc, cur);
            if (visited.found_existing) return;
            errdefer _ = self.visited_nodes.remove(cur);
            visited.value_ptr.* = {};
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
                if (!node.ownsData()) {
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
                if (node.opTag() != .none and node.ownsData()) {
                    _ = node.setAllScalar(0);
                }
            }
        }

        /// Execute the forward pass over all nodes.
        /// If `fusionPass()` was called, fused chains execute as single-pass
        /// comptime-specialized kernels.
        pub fn compute(self: *const Self) void {
            if (self.execution_steps.items.len == 0) {
                for (self.nodes.items) |node| node.compute();
                return;
            }
            self.executeGraphPlan(self.execution_steps.items);
        }

        pub fn computeNoGrad(self: *const Self) void {
            if (self.forward_execution_steps.items.len == 0) {
                for (self.nodes.items[0..self.forward_node_count]) |node| node.compute();
                return;
            }
            self.executeGraphPlan(self.forward_execution_steps.items);
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
            if (!self.built_fusion) {
                try self.fusionPass();
                self.built_fusion = true;
            }
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
            if (!self.built_fusion) {
                try self.fusionPass();
                self.built_fusion = true;
            }
            self.reset();
            self.computeNoGrad();
        }

        fn executeGraphPlan(self: *const Self, steps: []const ExecutionStep) void {
            for (steps) |step| {
                switch (step) {
                    .fusion => |idx| fused.executeFusionPlan(T, self.fused_chains.items[idx]),
                    .node => |node| node.compute(),
                }
            }
        }

        fn buildExecutionSteps(self: *Self, alloc: Alloc, limit: usize, out: *std.ArrayList(ExecutionStep)) Alloc.Error!void {
            out.clearRetainingCapacity();
            var next_chain: usize = self.nextFusedChainAtOrAfter(0, limit) orelse self.fused_chains.items.len;
            for (self.nodes.items[0..limit], 0..) |node, i| {
                if (i < self.fused_skip.items.len and self.fused_skip.items[i]) {
                    if (next_chain < self.fused_chains.items.len and self.fused_chains.items[next_chain].output_idx == i) {
                        try out.append(alloc, .{ .fusion = next_chain });
                        next_chain = self.nextFusedChainAtOrAfter(i + 1, limit) orelse self.fused_chains.items.len;
                    }
                    continue;
                }
                try out.append(alloc, .{ .node = node });
            }
        }

        fn invalidateExecutionPlans(self: *Self) void {
            self.built_fusion = false;
            self.execution_steps.clearRetainingCapacity();
            self.forward_execution_steps.clearRetainingCapacity();
            self.deinitFusedChains();
            self.fused_skip.clearRetainingCapacity();
        }

        fn deinitFusedChains(self: *Self) void {
            const alloc = self.arena.allocator();
            for (self.fused_chains.items) |plan| {
                fused.deinitFusionPlan(T, alloc, plan);
            }
            self.fused_chains.clearRetainingCapacity();
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

test "build compute graph avoids duplicate visits across shared subgraphs" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    const shared = x.sqr();
    const out = shared.add(shared);

    try g.buildForward(out);

    try testing.expectEqual(@as(usize, 2), g.nodes.items.len);
    try testing.expectEqual(@as(usize, 1), g.leaves.items.len);
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

test "fusion - compiler selects softmax region" {
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

test "fusion - conv2d backward fused matches unfused" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{ 4, 4, 1, 1 });
    const xu = try Tensor(f32).init(au, &.{ 4, 4, 1, 1 });
    for (xf.data, 0..) |*d, i| d.* = @floatFromInt(i + 1);
    @memcpy(xu.data, xf.data);
    xf.setParam();
    xu.setParam();

    const kf = try Tensor(f32).init(af, &.{ 2, 2, 1, 2 });
    const ku = try Tensor(f32).init(au, &.{ 2, 2, 1, 2 });
    for (kf.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 4)) - 1));
    @memcpy(ku.data, kf.data);
    kf.setParam();
    ku.setParam();

    const yf = xf.conv2d(kf).sumAll();
    const yu = xu.conv2d(ku).sumAll();

    try gf.buildForward(yf);
    try gf.buildBackward(false);
    try gf.fusionPass();
    _ = yf.grad.?.setAllScalar(1);

    try gu.buildForward(yu);
    try gu.buildBackward(false);
    _ = yu.grad.?.setAllScalar(1);

    gf.compute();
    gu.compute();

    for (xf.grad.?.data, xu.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
    for (kf.grad.?.data, ku.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
}

test "fusion - compiler root plan maps softmax" {
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

test "fusion - compiler selects logSoftmax region" {
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

test "fusion - compiler root plan maps logSoftmax" {
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

test "fusion - compiler selects cross entropy region" {
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
    try testing.expectEqual(fused.FusionKind.cross_entropy, g.fused_chains.items[0].kind());

    g.compute();

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, out.data[0], 1e-5);
}

test "fusion - compiler root plan maps cross entropy" {
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

test "fusion - compiler root plan maps layerNorm" {
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

test "fusion - compiler maps schedule-owned elementwise region" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.exp().log();
    try g.buildForward(y);

    var plans = try g.detectCompilerFusionPlans();
    defer g.deinitCompilerPlans(std.heap.page_allocator, &plans);

    try testing.expectEqual(@as(usize, 1), plans.items.len);
    try testing.expectEqual(fused.FusionKind.elementwise_chain, plans.items[0].plan.kind());
    try testing.expectEqual(@as(usize, 0), plans.items[0].start_idx);
    try testing.expectEqual(@as(usize, g.forward_node_count - 1), plans.items[0].plan.output_idx);
}

test "compute uses compiler execution plan for softmax" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.softmax(&.{1});

    try g.buildForward(y);
    try g.fusionPass();
    try testing.expect(g.fused_chains.items.len > 0);

    g.computeNoGrad();
    try testing.expectApproxEqAbs(@as(f32, 0.09003057), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.24472848), y.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.66524094), y.data[2], 1e-5);
}

test "compute uses schedule-owned elementwise fusion" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{3});
    xf.setData(&.{ 1, 2, 3 });
    const xu = try Tensor(f32).init(au, &.{3});
    xu.setData(&.{ 1, 2, 3 });

    const yf = xf.exp().log();
    const yu = xu.exp().log();

    try gf.buildForward(yf);
    try gu.buildForward(yu);
    try gf.fusionPass();

    try testing.expectEqual(@as(usize, 1), gf.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.elementwise_chain, gf.fused_chains.items[0].kind());

    gf.computeNoGrad();
    gu.computeNoGrad();

    try testing.expectEqualSlices(f32, yu.data, yf.data);
}

test "fusion - compiler maps swapped commutative schedule-owned elementwise region" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const scalar = try Tensor(f32).initScalar(a, 2);
    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = scalar.repeatLike(x).add(x.exp()).log();
    try g.buildForward(y);

    var plans = try g.detectCompilerFusionPlans();
    defer g.deinitCompilerPlans(std.heap.page_allocator, &plans);

    try testing.expectEqual(@as(usize, 1), plans.items.len);
    try testing.expectEqual(fused.FusionKind.elementwise_chain, plans.items[0].plan.kind());
    const chain = plans.items[0].plan.payload.elementwise_chain;
    try testing.expectEqual(@as(usize, 3), chain.nodes.len);
    try testing.expectEqual(fused.BinaryOperandRole.src0, chain.otherOperandRole(1));
}

test "compute uses swapped commutative schedule-owned elementwise fusion" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const sf = try Tensor(f32).initScalar(af, 2);
    const su = try Tensor(f32).initScalar(au, 2);
    const xf = try Tensor(f32).init(af, &.{3});
    xf.setData(&.{ 1, 2, 3 });
    const xu = try Tensor(f32).init(au, &.{3});
    xu.setData(&.{ 1, 2, 3 });

    const yf = sf.repeatLike(xf).add(xf.exp()).log();
    const yu = su.repeatLike(xu).add(xu.exp()).log();

    try gf.buildForward(yf);
    try gu.buildForward(yu);
    try gf.fusionPass();

    try testing.expectEqual(@as(usize, 1), gf.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.elementwise_chain, gf.fused_chains.items[0].kind());
    try testing.expectEqual(fused.BinaryOperandRole.src0, gf.fused_chains.items[0].payload.elementwise_chain.otherOperandRole(1));

    gf.computeNoGrad();
    gu.computeNoGrad();

    try testing.expectEqualSlices(f32, yu.data, yf.data);
}

test "fusion report reflects built graph" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 6, 6, 1, 1 });
    const k = try Tensor(f32).init(a, &.{ 3, 3, 1, 2 });
    const y = x.conv2d(k).sumAll();
    try g.buildForward(y);
    try g.fusionPass();

    const summary = g.fusionSummary();
    try testing.expect(summary.node_count >= summary.forward_node_count);
    try testing.expect(summary.fused_region_count > 0);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(tac);
    try g.dumpFusionReport(buf.writer(tac));
    try testing.expect(std.mem.indexOf(u8, buf.items, "fused[") != null);
    try testing.expect(std.mem.indexOf(u8, buf.items, "node[") != null);
}

test "tensor lineage reports reachable node chain" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 4, 4, 1, 1 });
    const k = try Tensor(f32).init(a, &.{ 2, 2, 1, 1 });
    const y = x.conv2d(k).sumAll();
    try g.buildForward(y);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(tac);
    try g.dumpTensorLineage(buf.writer(tac), y);
    try testing.expect(std.mem.indexOf(u8, buf.items, "lineage for node[") != null);
    try testing.expect(std.mem.indexOf(u8, buf.items, "sum") != null);
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

test "fusion - compiler selects layerNorm region" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = x.layerNorm(&.{ 1, 3 }, 1e-5);
    try g.buildForward(y);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.layer_norm, g.fused_chains.items[0].kind());
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
