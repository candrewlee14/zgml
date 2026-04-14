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
const fusion = @import("fusion.zig");
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
        backward_inplace: bool = false,
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

        /// Optional thread pool for parallel matmul and elementwise ops.
        /// Null by default (single-threaded). Call `enableThreading()` to activate.
        thread_pool: ?std.Thread.Pool = null,

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
            if (self.thread_pool) |*tp| tp.deinit();
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

        /// Enable multi-threaded execution for matmul and elementwise ops.
        /// Uses all available CPU cores. Safe to call multiple times (no-op if already enabled).
        pub fn enableThreading(self: *Self) !void {
            if (self.thread_pool != null) return;
            var pool: std.Thread.Pool = undefined;
            try pool.init(.{
                .allocator = std.heap.page_allocator,
                .track_ids = false,
            });
            self.thread_pool = pool;
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
            self.backward_inplace = keep;
            self.resetGrads();
            self.invalidateExecutionPlans();
        }

        /// Detect fusible patterns in the built graph and prepare
        /// specialized kernels for them.
        ///
        /// Call after `buildForward()` (and optionally after `buildBackward()`).
        /// Uses the unified FusionDetector which works directly on the tensor
        /// graph — no intermediate IR round-trip.
        pub fn fusionPass(self: *Self) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const node_count = self.nodes.items.len;
            if (node_count < 2) return;

            self.deinitFusedChains();
            self.fused_skip.clearRetainingCapacity();

            // Run unified fusion detection
            var detector = try fusion.FusionDetector(T).init(alloc, self.nodes.items, self.forward_node_count);
            defer detector.deinit(alloc);
            try detector.detect(alloc);

            // Transfer skip bitmap
            self.fused_skip = try std.ArrayList(bool).initCapacity(alloc, node_count);
            self.fused_skip.items.len = node_count;
            @memcpy(self.fused_skip.items, detector.fused_skip);

            // Transfer fused chains (clone elementwise plans that own allocations)
            for (detector.fused_chains.items) |plan| {
                try self.fused_chains.append(alloc, try fused.cloneFusionPlan(T, alloc, plan));
            }

            // Dead code elimination: remove backward nodes whose outputs
            // are never read.
            {
                var live_roots = std.AutoHashMapUnmanaged(*Tensor(T), void).empty;
                for (self.nodes.items[0..self.forward_node_count]) |node| {
                    if (node.isParam()) {
                        if (node.gradOrNull()) |grad| {
                            try live_roots.put(alloc, grad, {});
                        }
                    }
                }
                for (self.fused_chains.items) |chain| {
                    for (chain.liveRefs()) |ref| {
                        try live_roots.put(alloc, ref, {});
                    }
                }
                defer live_roots.deinit(alloc);

                // Build node→index map for DCE
                var ptr_to_idx = std.AutoHashMap(*Tensor(T), usize).init(alloc);
                defer ptr_to_idx.deinit();
                try ptr_to_idx.ensureTotalCapacity(@intCast(node_count));
                for (self.nodes.items, 0..) |node_ptr, idx| {
                    ptr_to_idx.putAssumeCapacity(node_ptr, idx);
                }

                // Count source references from ALL nodes (including fused).
                // Fused nodes still consume their sources during fused execution —
                // only the fused nodes themselves are candidates for removal, not
                // their dependencies.
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

                var changed = true;
                while (changed) {
                    changed = false;
                    for (self.nodes.items[self.forward_node_count..], self.forward_node_count..) |node, idx| {
                        if (self.fused_skip.items[idx]) continue;
                        if (use_count[idx] > 0) continue;
                        if (live_roots.contains(node)) continue;
                        self.fused_skip.items[idx] = true;
                        if (node.source0()) |src| {
                            if (ptr_to_idx.get(src)) |j| use_count[j] -= 1;
                        }
                        if (node.source1()) |src| {
                            if (ptr_to_idx.get(src)) |j| use_count[j] -= 1;
                        }
                        changed = true;
                    }
                }
            }

            for (self.fused_chains.items) |*chain| {
                try chain.allocScratchBuffers(alloc);
            }

            try self.buildExecutionSteps(alloc, node_count, &self.execution_steps);
            try self.buildExecutionSteps(alloc, self.forward_node_count, &self.forward_execution_steps);
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
            leaf_count: usize,
            param_count: usize,
            aux_count: usize,
        };

        pub fn fusionSummary(self: *const Self) FusionSummary {
            var param_count: usize = 0;
            var aux_count: usize = 0;
            for (self.nodes.items[0..self.forward_node_count]) |node| {
                if (node.isParam()) param_count += 1;
                if (node.isInternalAux()) aux_count += 1;
            }
            return .{
                .node_count = self.nodes.items.len,
                .forward_node_count = self.forward_node_count,
                .fused_region_count = self.fused_chains.items.len,
                .leaf_count = self.leaves.items.len,
                .param_count = param_count,
                .aux_count = aux_count,
            };
        }

        pub const ReportOptions = struct {
            include_nodes: bool = true,
            include_execution: bool = true,
        };

        pub const ExecutionPlanSummary = struct {
            forward_step_count: usize,
            backward_step_count: usize,
        };

        pub const NodePhase = enum {
            forward,
            backward,

            fn label(self: @This()) []const u8 {
                return switch (self) {
                    .forward => "fwd",
                    .backward => "bwd",
                };
            }
        };

        pub const NodeExecutionKind = enum {
            node,
            fused_internal,
            fused_output,

            fn label(self: @This()) []const u8 {
                return switch (self) {
                    .node => "node",
                    .fused_internal => "fused internal",
                    .fused_output => "fused output",
                };
            }
        };

        pub const NodeDisposition = enum {
            covered_by_fused_region,
            fused_region_output,
            non_fusible_primitive,
            fusible_terminal,
            binary_operand_not_directly_indexable,
            consumer_chain_not_linear,
            consumer_changes_shape,
            consumer_binary_operand_not_directly_indexable,
            multiple_consumers_prevent_chain_fusion,
            fusible_pending_schedule,
            fusible_isolated,

            fn description(self: @This()) []const u8 {
                return switch (self) {
                    .covered_by_fused_region => "covered by fused region",
                    .fused_region_output => "region output",
                    .non_fusible_primitive => "non-fusible primitive",
                    .fusible_terminal => "fusible but terminal",
                    .binary_operand_not_directly_indexable => "binary operand is not directly indexable",
                    .consumer_chain_not_linear => "consumer chain is not linear",
                    .consumer_changes_shape => "consumer changes shape",
                    .consumer_binary_operand_not_directly_indexable => "consumer binary operand is not directly indexable",
                    .multiple_consumers_prevent_chain_fusion => "multiple consumers prevent chain fusion",
                    .fusible_pending_schedule => "fusible but chain candidate depends on later scheduling",
                    .fusible_isolated => "fusible but isolated",
                };
            }
        };

        pub const TensorRef = union(enum) {
            node: usize,
            leaf,
            external,

            fn render(self: @This(), writer: anytype) !void {
                switch (self) {
                    .node => |idx| try writer.print("node[{}]", .{idx}),
                    .leaf => try writer.writeAll("leaf"),
                    .external => try writer.writeAll("external"),
                }
            }
        };

        pub const NodeReport = struct {
            index: usize,
            phase: NodePhase,
            execution_kind: NodeExecutionKind,
            disposition: NodeDisposition,
            fusion_region: ?usize,
            op: Op,
            n_dims: usize,
            shape: [tensorlib.max_dims]usize,
            elem_count: usize,
            src0: ?TensorRef,
            src1: ?TensorRef,
            has_grad: bool,
            is_param: bool,
            is_aux: bool,
            owns_data: bool,

            pub fn render(self: @This(), writer: anytype) !void {
                try writer.print(
                    "node[{d}] phase={s} exec={s} op={s} shape={any} elems={} ",
                    .{
                        self.index,
                        self.phase.label(),
                        self.execution_kind.label(),
                        @tagName(self.op),
                        self.shape[0..self.n_dims],
                        self.elem_count,
                    },
                );
                try writer.writeAll("src0=");
                if (self.src0) |src0| {
                    try src0.render(writer);
                } else {
                    try writer.writeAll("none");
                }
                try writer.writeAll(" src1=");
                if (self.src1) |src1| {
                    try src1.render(writer);
                } else {
                    try writer.writeAll("none");
                }
                try writer.print(
                    " grad={any} param={any} aux={any} owns_data={any} note={s}\n",
                    .{
                        self.has_grad,
                        self.is_param,
                        self.is_aux,
                        self.owns_data,
                        self.disposition.description(),
                    },
                );
            }
        };

        pub const FusionRegionDetail = union(fused.FusionKind) {
            elementwise_chain: struct {
                input: TensorRef,
                chain_len: usize,
            },
            conv2d: struct {
                input: TensorRef,
                kernel: TensorRef,
                has_bias: bool,
                has_activation: bool,
            },
            conv2d_bwd_input: struct {
                kernel: TensorRef,
            },
            conv2d_bwd_kernel: struct {
                input: TensorRef,
            },
            max_pool2d: struct {
                input: TensorRef,
            },
            max_pool2d_bwd: struct {
                input: TensorRef,
            },
            softmax: struct {
                input: TensorRef,
            },
            log_softmax: struct {
                input: TensorRef,
            },
            cross_entropy: struct {
                logits: TensorRef,
                targets: TensorRef,
            },
            layer_norm: struct {
                input: TensorRef,
            },

            fn render(self: @This(), writer: anytype, output_idx: usize) !void {
                switch (self) {
                    .elementwise_chain => |payload| {
                        try writer.writeAll("  input=");
                        try payload.input.render(writer);
                        try writer.print(" chain_len={} output=node[{}]\n", .{ payload.chain_len, output_idx });
                    },
                    .conv2d => |payload| {
                        try writer.writeAll("  conv input=");
                        try payload.input.render(writer);
                        try writer.writeAll(" kernel=");
                        try payload.kernel.render(writer);
                        try writer.print(" bias={} activation={} output=node[{}]\n", .{ payload.has_bias, payload.has_activation, output_idx });
                    },
                    .conv2d_bwd_input => |payload| {
                        try writer.writeAll("  backward input_grad kernel=");
                        try payload.kernel.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .conv2d_bwd_kernel => |payload| {
                        try writer.writeAll("  backward kernel_grad input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .max_pool2d => |payload| {
                        try writer.writeAll("  max_pool input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .max_pool2d_bwd => |payload| {
                        try writer.writeAll("  backward pool_grad input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .softmax => |payload| {
                        try writer.writeAll("  softmax input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .log_softmax => |payload| {
                        try writer.writeAll("  log_softmax input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .cross_entropy => |payload| {
                        try writer.writeAll("  cross_entropy logits=");
                        try payload.logits.render(writer);
                        try writer.writeAll(" targets=");
                        try payload.targets.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                    .layer_norm => |payload| {
                        try writer.writeAll("  layer_norm input=");
                        try payload.input.render(writer);
                        try writer.print(" output=node[{}]\n", .{output_idx});
                    },
                }
            }
        };

        pub const FusionRegionReport = struct {
            region_index: usize,
            kind: fused.FusionKind,
            start_idx: usize,
            output_idx: usize,
            detail: FusionRegionDetail,

            pub fn render(self: @This(), writer: anytype) !void {
                try writer.print("fused[{d}] kind={s} range={}..{}\n", .{ self.region_index, @tagName(self.kind), self.start_idx, self.output_idx });
                try self.detail.render(writer, self.output_idx);
            }
        };

        pub const GraphReport = struct {
            alloc: Alloc,
            summary: FusionSummary,
            execution: ?ExecutionPlanSummary,
            fused_regions: []FusionRegionReport,
            nodes: []NodeReport,

            pub fn deinit(self: *@This()) void {
                self.alloc.free(self.fused_regions);
                self.alloc.free(self.nodes);
                self.* = undefined;
            }

            pub fn render(self: *const @This(), writer: anytype) !void {
                try writer.print("graph nodes={} forward_nodes={} backward_nodes={} leaves={} params={} aux={} fused_regions={}\n", .{
                    self.summary.node_count,
                    self.summary.forward_node_count,
                    self.summary.node_count - self.summary.forward_node_count,
                    self.summary.leaf_count,
                    self.summary.param_count,
                    self.summary.aux_count,
                    self.summary.fused_region_count,
                });

                for (self.fused_regions) |region| {
                    try region.render(writer);
                }

                if (self.execution) |execution| {
                    try writer.print("execution forward_steps={} backward_steps={}\n", .{ execution.forward_step_count, execution.backward_step_count });
                }

                for (self.nodes) |node| {
                    try node.render(writer);
                }
            }
        };

        pub const ExecutionProfileOptions = struct {
            reset: bool = true,
            reset_grads: bool = true,
            loss_grad: ?*Tensor(T) = null,
            forward_only: bool = false,
        };

        pub const ExecutionProfile = struct {
            reset_ns: u64 = 0,
            reset_grads_ns: u64 = 0,
            seed_loss_grad_ns: u64 = 0,
            forward_ns: u64 = 0,
            backward_ns: u64 = 0,
            total_ns: u64 = 0,
            node_count: usize = 0,
            forward_node_count: usize = 0,
            fused_region_count: usize = 0,
            forward_step_count: usize = 0,
            backward_step_count: usize = 0,

            pub fn render(self: @This(), writer: anytype) !void {
                try writer.print(
                    "profile nodes={} forward_nodes={} fused_regions={} forward_steps={} backward_steps={}\n",
                    .{ self.node_count, self.forward_node_count, self.fused_region_count, self.forward_step_count, self.backward_step_count },
                );
                try writer.print(
                    "  reset={d:.3}ms reset_grads={d:.3}ms seed_loss_grad={d:.3}ms forward={d:.3}ms backward={d:.3}ms total={d:.3}ms\n",
                    .{
                        nsToMs(self.reset_ns),
                        nsToMs(self.reset_grads_ns),
                        nsToMs(self.seed_loss_grad_ns),
                        nsToMs(self.forward_ns),
                        nsToMs(self.backward_ns),
                        nsToMs(self.total_ns),
                    },
                );
            }

            pub fn dump(self: @This(), writer: anytype) !void {
                try self.render(writer);
            }
        };

        pub fn buildReport(self: *const Self, alloc: Alloc, options: ReportOptions) Alloc.Error!GraphReport {
            const fused_regions = try alloc.alloc(FusionRegionReport, self.fused_chains.items.len);
            errdefer alloc.free(fused_regions);
            for (self.fused_chains.items, 0..) |plan, i| {
                fused_regions[i] = self.buildFusionRegionReport(i, plan);
            }

            const nodes = if (options.include_nodes)
                try alloc.alloc(NodeReport, self.nodes.items.len)
            else
                try alloc.alloc(NodeReport, 0);
            errdefer alloc.free(nodes);

            if (options.include_nodes) {
                for (self.nodes.items, 0..) |node, i| {
                    nodes[i] = self.buildNodeReport(i, node, fused_regions);
                }
            }

            return .{
                .alloc = alloc,
                .summary = self.fusionSummary(),
                .execution = if (options.include_execution) .{
                    .forward_step_count = self.stepCount(true),
                    .backward_step_count = self.stepCount(false),
                } else null,
                .fused_regions = fused_regions,
                .nodes = nodes,
            };
        }

        pub fn dumpFusionReport(self: *const Self, writer: anytype) !void {
            return self.dumpReport(writer, .{});
        }

        pub fn dumpReport(self: *const Self, writer: anytype, options: ReportOptions) !void {
            var report = try self.buildReport(self.arena.child_allocator, options);
            defer report.deinit();
            try report.render(writer);
        }

        pub fn profileExecution(self: *Self, options: ExecutionProfileOptions) !ExecutionProfile {
            var timer = try std.time.Timer.start();
            var profile = ExecutionProfile{
                .node_count = self.nodes.items.len,
                .forward_node_count = self.forward_node_count,
                .fused_region_count = self.fused_chains.items.len,
                .forward_step_count = self.stepCount(true),
                .backward_step_count = self.stepCount(false),
            };

            timer.reset();
            if (options.reset) self.reset();
            profile.reset_ns = timer.read();

            timer.reset();
            if (options.reset_grads) self.resetGrads();
            profile.reset_grads_ns = timer.read();

            timer.reset();
            if (options.loss_grad) |grad| _ = grad.setAllScalar(1);
            profile.seed_loss_grad_ns = timer.read();

            timer.reset();
            self.computeNoGrad();
            profile.forward_ns = timer.read();

            if (!options.forward_only) {
                timer.reset();
                self.computeBackward();
                profile.backward_ns = timer.read();
            }

            profile.total_ns = profile.reset_ns + profile.reset_grads_ns + profile.seed_loss_grad_ns + profile.forward_ns + profile.backward_ns;
            return profile;
        }

        pub fn dumpTensorLineage(self: *const Self, writer: anytype, needle: *Tensor(T)) !void {
            const idx = fused.indexOfNodeMaybe(T, self.nodes.items, needle) orelse {
                try writer.print("tensor not found in graph\n", .{});
                return;
            };
            try writer.print("lineage for node[{d}] {s} shape={any}\n", .{ idx, @tagName(needle.opTag()), needle.ne[0..needle.n_dims] });

            var visited = std.AutoHashMap(*Tensor(T), void).init(self.arena.child_allocator);
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

        fn nsToMs(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        }

        fn stepCount(self: *const Self, comptime forward_only: bool) usize {
            if (forward_only) {
                return if (self.forward_execution_steps.items.len != 0) self.forward_execution_steps.items.len else self.forward_node_count;
            }
            return if (self.execution_steps.items.len != 0)
                self.execution_steps.items.len - self.stepCount(true)
            else if (self.nodes.items.len >= self.forward_node_count)
                self.nodes.items.len - self.forward_node_count
            else
                0;
        }

        fn fusionStartIdx(self: *const Self, plan: fused.FusionPlan(T)) usize {
            return fused.indexOfNodeMaybe(T, self.nodes.items, self.fusionAnchorNode(plan) orelse return plan.output_idx) orelse plan.output_idx;
        }

        fn fusionAnchorNode(_: *const Self, plan: fused.FusionPlan(T)) ?*Tensor(T) {
            return switch (plan.payload) {
                .elementwise_chain => |payload| payload.nodes[0],
                .conv2d => |payload| payload.input_view,
                .conv2d_bwd_input => |payload| payload.output,
                .conv2d_bwd_kernel => |payload| payload.output,
                .max_pool2d => |payload| payload.strided,
                .max_pool2d_bwd => |payload| payload.output,
                .softmax => |payload| payload.max_node,
                .log_softmax => |payload| payload.max_node,
                .cross_entropy => |payload| payload.log_softmax.max_node,
                .layer_norm => |payload| payload.sum_node,
            };
        }

        fn fusionCoverageAt(idx: usize, fused_regions: []const FusionRegionReport) ?struct { region_idx: usize, is_output: bool } {
            for (fused_regions) |region| {
                if (idx < region.start_idx or idx > region.output_idx) continue;
                return .{ .region_idx = region.region_index, .is_output = idx == region.output_idx };
            }
            return null;
        }

        fn buildFusionRegionReport(self: *const Self, region_index: usize, plan: fused.FusionPlan(T)) FusionRegionReport {
            return .{
                .region_index = region_index,
                .kind = plan.kind(),
                .start_idx = self.fusionStartIdx(plan),
                .output_idx = plan.output_idx,
                .detail = switch (plan.payload) {
                    .elementwise_chain => |payload| .{ .elementwise_chain = .{
                        .input = self.tensorRef(payload.input),
                        .chain_len = payload.nodes.len,
                    } },
                    .conv2d => |payload| .{ .conv2d = .{
                        .input = self.tensorRef(payload.input),
                        .kernel = self.tensorRef(payload.kernel),
                        .has_bias = payload.bias != null,
                        .has_activation = payload.activation != null,
                    } },
                    .conv2d_bwd_input => |payload| .{ .conv2d_bwd_input = .{
                        .kernel = self.tensorRef(payload.kernel),
                    } },
                    .conv2d_bwd_kernel => |payload| .{ .conv2d_bwd_kernel = .{
                        .input = self.tensorRef(payload.input),
                    } },
                    .max_pool2d => |payload| .{ .max_pool2d = .{
                        .input = self.tensorRef(payload.input),
                    } },
                    .max_pool2d_bwd => |payload| .{ .max_pool2d_bwd = .{
                        .input = self.tensorRef(payload.input),
                    } },
                    .softmax => |payload| .{ .softmax = .{
                        .input = self.tensorRef(payload.input),
                    } },
                    .log_softmax => |payload| .{ .log_softmax = .{
                        .input = self.tensorRef(payload.input),
                    } },
                    .cross_entropy => |payload| .{ .cross_entropy = .{
                        .logits = self.tensorRef(payload.log_softmax.input),
                        .targets = self.tensorRef(payload.targets),
                    } },
                    .layer_norm => |payload| .{ .layer_norm = .{
                        .input = self.tensorRef(payload.input),
                    } },
                },
            };
        }

        fn buildNodeReport(self: *const Self, idx: usize, node: *Tensor(T), fused_regions: []const FusionRegionReport) NodeReport {
            const coverage = fusionCoverageAt(idx, fused_regions);
            const execution_kind: NodeExecutionKind = if (coverage) |c|
                if (c.is_output) .fused_output else .fused_internal
            else
                .node;
            const disposition: NodeDisposition = if (coverage) |c|
                if (c.is_output) .fused_region_output else .covered_by_fused_region
            else
                self.nodeDisposition(idx, node);

            return .{
                .index = idx,
                .phase = if (idx < self.forward_node_count) .forward else .backward,
                .execution_kind = execution_kind,
                .disposition = disposition,
                .fusion_region = if (coverage) |c| c.region_idx else null,
                .op = node.opTag(),
                .n_dims = node.n_dims,
                .shape = node.ne,
                .elem_count = node.nElems(),
                .src0 = if (node.source0()) |src| self.tensorRef(src) else null,
                .src1 = if (node.source1()) |src| self.tensorRef(src) else null,
                .has_grad = node.hasGrad(),
                .is_param = node.isParam(),
                .is_aux = node.isInternalAux(),
                .owns_data = node.ownsData(),
            };
        }

        fn tensorRef(self: *const Self, node_ptr: *Tensor(T)) TensorRef {
            if (fused.indexOfNodeMaybe(T, self.nodes.items, node_ptr)) |idx| return .{ .node = idx };
            for (self.leaves.items) |leaf| {
                if (leaf == node_ptr) return .leaf;
            }
            return .external;
        }

        fn nodeDisposition(self: *const Self, idx: usize, node: *Tensor(T)) NodeDisposition {
            if (!node.opTag().isFusible()) return .non_fusible_primitive;
            if (idx + 1 >= self.nodes.items.len) return .fusible_terminal;

            if (node.source1()) |src1| {
                if (!src1.isScalar() and !(node.isSameShape(src1) and src1.isContiguous() and src1.data.len == node.nElems())) {
                    return .binary_operand_not_directly_indexable;
                }
            }

            const next = self.nodes.items[idx + 1];
            if (next.opTag().isFusible()) {
                if (next.source0() != node) return .consumer_chain_not_linear;
                if (!next.isSameShape(node)) return .consumer_changes_shape;
                if (next.source1()) |src1| {
                    if (!src1.isScalar() and !(next.isSameShape(src1) and src1.isContiguous() and src1.data.len == next.nElems())) {
                        return .consumer_binary_operand_not_directly_indexable;
                    }
                }
                var uses: usize = 0;
                for (self.nodes.items) |candidate| {
                    if (candidate.source0() == node or candidate.source1() == node) uses += 1;
                    if (uses > 1) return .multiple_consumers_prevent_chain_fusion;
                }
                return .fusible_pending_schedule;
            }
            return .fusible_isolated;
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
        /// Zero all intermediate node data to prepare for recomputation.
        /// Skips fused_skip nodes (their data is never read) and nodes
        /// that alias another tensor's data (e.g. reshape/view).
        pub fn reset(self: *Self) void {
            for (self.nodes.items, 0..) |node, i| {
                if (node.opTag() != .none and node.ownsData()) {
                    if (i < self.fused_skip.items.len and self.fused_skip.items[i]) continue;
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

        /// Execute only the backward portion of the graph (nodes after forward_node_count).
        /// Uses fused execution plans when available.
        pub fn computeBackward(self: *const Self) void {
            if (self.execution_steps.items.len == 0) {
                for (self.nodes.items[self.forward_node_count..]) |node| node.compute();
                return;
            }
            // Backward steps are everything after the forward execution steps.
            self.executeGraphPlan(self.execution_steps.items[self.forward_execution_steps.items.len..]);
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
            const pool = if (self.thread_pool) |*tp| @constCast(tp) else null;
            for (steps) |step| {
                switch (step) {
                    .fusion => |idx| {
                        const plan = self.fused_chains.items[idx];
                        if (pool != null and plan.kind() == .elementwise_chain) {
                            fused.executeFusedChainParallel(T, plan.payload.elementwise_chain, pool.?);
                        } else {
                            fused.executeFusionPlan(T, plan);
                        }
                    },
                    .node => |node| {
                        if (pool != null and node.opTag() == .matmul) {
                            const flags = node.matmul_flags;
                            const s0 = node.src0.?;
                            const s1 = node.src1.?;
                            if (flags.trans0) {
                                if (flags.trans1) Tensor(T).computeMatMulParallel(node, s0, true, s1, true, pool.?) else Tensor(T).computeMatMulParallel(node, s0, true, s1, false, pool.?);
                            } else {
                                if (flags.trans1) Tensor(T).computeMatMulParallel(node, s0, false, s1, true, pool.?) else Tensor(T).computeMatMulParallel(node, s0, false, s1, false, pool.?);
                            }
                        } else {
                            node.compute();
                        }
                    },
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

test "fusion - detects softmax region" {
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

test "fusion - detects conv2d bias relu composite pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 6, 6, 1, 2 });
    const k = try Tensor(f32).init(a, &.{ 3, 3, 1, 4 });
    const b = try Tensor(f32).init(a, &.{4});
    const conv = x.conv2d(k);
    const b4 = b.reshape(&.{ 1, 1, 4, 1 });
    const y = conv.add(b4.repeat(conv.ne[0..conv.n_dims])).relu();
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

test "fusion - conv2d bias relu fused matches unfused" {
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

    const bf = try Tensor(f32).init(af, &.{2});
    const bu = try Tensor(f32).init(au, &.{2});
    bf.setData(&.{ -1.5, 0.5 });
    bu.setData(&.{ -1.5, 0.5 });

    const convf = xf.conv2d(kf);
    const convu = xu.conv2d(ku);
    const bff = bf.reshape(&.{ 1, 1, 2, 1 });
    const bfu = bu.reshape(&.{ 1, 1, 2, 1 });
    const yf = convf.add(bff.repeat(convf.ne[0..convf.n_dims])).relu();
    const yu = convu.add(bfu.repeat(convu.ne[0..convu.n_dims])).relu();
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

test "fusion - detects softmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.softmax(&.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 1);
    try testing.expectEqual(fused.FusionKind.softmax, g.fused_chains.items[g.fused_chains.items.len - 1].kind());
}

test "fusion - detects logSoftmax region" {
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

test "fusion - detects logSoftmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.logSoftmax(&.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 1);
    try testing.expectEqual(fused.FusionKind.log_softmax, g.fused_chains.items[g.fused_chains.items.len - 1].kind());
}

test "fusion - detects cross entropy region" {
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

    try testing.expect(g.fused_chains.items.len >= 1);
    // Cross entropy should be one of the detected patterns
    var has_cross_entropy = false;
    for (g.fused_chains.items) |chain| {
        if (chain.kind() == .cross_entropy) has_cross_entropy = true;
    }
    try testing.expect(has_cross_entropy);

    g.compute();

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, out.data[0], 1e-5);
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

    try testing.expect(g.fused_chains.items.len >= 1);
    try testing.expectEqual(fused.FusionKind.cross_entropy, g.fused_chains.items[g.fused_chains.items.len - 1].kind());
}

test "fusion - detects layerNorm pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const out = x.layerNorm(&.{ 1, 3 }, 1e-5);
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 1);
    try testing.expectEqual(fused.FusionKind.layer_norm, g.fused_chains.items[g.fused_chains.items.len - 1].kind());
}

test "fusion - detects multi-region plans" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = x.layerNorm(&.{ 1, 3 }, 1e-5).softmax(&.{ 1, 3 });
    try g.buildForward(y);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 2);
    // Should have both layer_norm and softmax patterns
    var has_layer_norm = false;
    var has_softmax = false;
    for (g.fused_chains.items) |chain| {
        if (chain.kind() == .layer_norm) has_layer_norm = true;
        if (chain.kind() == .softmax) has_softmax = true;
    }
    try testing.expect(has_layer_norm);
    try testing.expect(has_softmax);
}

test "fusion - detects elementwise chain" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.exp().log();
    try g.buildForward(y);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.elementwise_chain, g.fused_chains.items[0].kind());
}

test "compute uses fused execution plan for softmax" {
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

test "fusion - detects swapped commutative elementwise chain" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const scalar = try Tensor(f32).initScalar(a, 2);
    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = scalar.repeatLike(x).add(x.exp()).log();
    try g.buildForward(y);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 1);
    // Should find an elementwise chain
    var found_chain = false;
    for (g.fused_chains.items) |chain| {
        if (chain.kind() == .elementwise_chain) found_chain = true;
    }
    try testing.expect(found_chain);
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
    try testing.expect(summary.leaf_count > 0);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(tac);
    try g.dumpFusionReport(buf.writer(tac));
    try testing.expect(std.mem.indexOf(u8, buf.items, "fused[") != null);
    try testing.expect(std.mem.indexOf(u8, buf.items, "node[") != null);
    try testing.expect(std.mem.indexOf(u8, buf.items, "forward_steps=") != null);
}

test "profile execution reports phase timings" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    const y = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    y.setData(&.{ 3, 2, 1 });
    x.setParam();
    const loss_node = x.add(y).sumAll();
    try g.buildForward(loss_node);
    try g.buildBackward(false);
    try g.fusionPass();

    const profile = try g.profileExecution(.{ .loss_grad = loss_node.grad });
    try testing.expect(profile.total_ns >= profile.forward_ns + profile.backward_ns);
    try testing.expect(profile.node_count == g.nodes.items.len);
    try testing.expect(profile.forward_step_count > 0);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(tac);
    try profile.dump(buf.writer(tac));
    try testing.expect(std.mem.indexOf(u8, buf.items, "profile nodes=") != null);
    try testing.expect(std.mem.indexOf(u8, buf.items, "forward=") != null);
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

test "fusion - detects layerNorm region" {
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

test "fusion - conv2d backward kernel gradient fused matches unfused" {
    // Build two identical graphs — one with fusionPass, one without.
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    // Small conv: input [4,4,1,2], kernel [3,3,1,1] → output [2,2,1,2]
    const xf = try gf.param(&.{ 4, 4, 1, 2 });
    const kf = try gf.param(&.{ 3, 3, 1, 1 });
    const xu = try gu.param(&.{ 4, 4, 1, 2 });
    const ku = try gu.param(&.{ 3, 3, 1, 1 });

    // Same data for both
    for (xf.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
    @memcpy(xu.data, xf.data);
    for (kf.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.05 + 0.1;
    @memcpy(ku.data, kf.data);

    const yf = xf.conv2d(kf).sumAll();
    const yu = xu.conv2d(ku).sumAll();

    // Fused path
    try gf.buildForward(yf);
    try gf.buildBackward(true);
    try gf.fusionPass();
    gf.resetGrads();
    if (yf.grad) |g| _ = g.setAllScalar(1);
    gf.compute();

    // Unfused path — no fusionPass
    try gu.buildForward(yu);
    try gu.buildBackward(true);
    gu.resetGrads();
    if (yu.grad) |g| _ = g.setAllScalar(1);
    gu.compute();

    // Compare kernel gradients
    for (kf.gradOrNull().?.data, ku.gradOrNull().?.data) |fg, ug| {
        try testing.expectApproxEqAbs(ug, fg, 1e-4);
    }
}

test "fusion - conv2d backward gradients fused matches unfused (multi-filter)" {
    // Larger test: input [8,8,1,8], kernel [3,3,1,4] — matches conv classifier dims.
    // Verifies both kernel and input gradients.
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const xf = try gf.param(&.{ 8, 8, 1, 8 });
    const kf = try gf.param(&.{ 3, 3, 1, 4 });
    const xu = try gu.param(&.{ 8, 8, 1, 8 });
    const ku = try gu.param(&.{ 3, 3, 1, 4 });

    for (xf.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 37)) * 0.1 - 1.5;
    @memcpy(xu.data, xf.data);
    for (kf.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.05 + 0.1;
    @memcpy(ku.data, kf.data);

    const yf = xf.conv2d(kf).sumAll();
    const yu = xu.conv2d(ku).sumAll();

    try gf.buildForward(yf);
    try gf.buildBackward(true);
    try gf.fusionPass();
    gf.resetGrads();
    if (yf.grad) |g| _ = g.setAllScalar(1);
    gf.compute();

    try gu.buildForward(yu);
    try gu.buildBackward(true);
    gu.resetGrads();
    if (yu.grad) |g| _ = g.setAllScalar(1);
    gu.compute();

    for (kf.gradOrNull().?.data, ku.gradOrNull().?.data) |fg, ug| {
        try testing.expectApproxEqAbs(ug, fg, 1e-4);
    }
    for (xf.gradOrNull().?.data, xu.gradOrNull().?.data) |fg, ug| {
        try testing.expectApproxEqAbs(ug, fg, 1e-4);
    }
}

test "threaded elementwise matches single-threaded" {
    // Use 4096 elements to exceed min_chunk=1024 and trigger actual threading
    var gs = ComputeGraph(f32).init(tac);
    defer gs.deinit();
    var gt = ComputeGraph(f32).init(tac);
    defer gt.deinit();
    try gt.enableThreading();

    const xs = try Tensor(f32).init(gs.allocator(), &.{4096});
    const xt = try Tensor(f32).init(gt.allocator(), &.{4096});
    for (xs.data, xt.data, 0..) |*a, *b, i| {
        a.* = @as(f32, @floatFromInt(i)) * 0.001;
        b.* = @as(f32, @floatFromInt(i)) * 0.001;
    }

    const ys = xs.exp().neg().add(xs.gelu());
    const yt = xt.exp().neg().add(xt.gelu());

    try gs.buildForward(ys);
    try gs.fusionPass();
    gs.computeNoGrad();

    try gt.buildForward(yt);
    try gt.fusionPass();
    gt.computeNoGrad();

    for (ys.data, yt.data) |sv, tv| {
        try testing.expectApproxEqAbs(sv, tv, 1e-6);
    }
}

test "threaded matmul matches single-threaded" {
    var gs = ComputeGraph(f32).init(tac);
    defer gs.deinit();
    var gt = ComputeGraph(f32).init(tac);
    defer gt.deinit();
    try gt.enableThreading();

    // 64x32 @ 32x16 — large enough to split across threads
    const as_ = try Tensor(f32).init(gs.allocator(), &.{ 32, 64 });
    const bs_ = try Tensor(f32).init(gs.allocator(), &.{ 16, 32 });
    const at = try Tensor(f32).init(gt.allocator(), &.{ 32, 64 });
    const bt = try Tensor(f32).init(gt.allocator(), &.{ 16, 32 });

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (as_.data, at.data) |*a, *b| {
        const v = rand.float(f32) * 2.0 - 1.0;
        a.* = v;
        b.* = v;
    }
    for (bs_.data, bt.data) |*a, *b| {
        const v = rand.float(f32) * 2.0 - 1.0;
        a.* = v;
        b.* = v;
    }

    const ys = as_.mm(bs_);
    const yt = at.mm(bt);

    try gs.buildForward(ys);
    try gs.fusionPass();
    gs.computeNoGrad();

    try gt.buildForward(yt);
    try gt.fusionPass();
    gt.computeNoGrad();

    for (ys.data, yt.data) |sv, tv| {
        try testing.expectApproxEqAbs(sv, tv, 1e-4);
    }
}

//#endregion
