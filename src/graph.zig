//! Computation graph for automatic differentiation.
//!
//! A `ComputeGraph` owns an arena allocator and manages the lifecycle of all
//! tensors created within it. Call `allocator()` to get the arena for tensor
//! creation, then use `infer()` for forward-only execution or `run()` for a
//! training step. A single `deinit()` frees everything.

const std = @import("std");
const builtin = @import("builtin");

const tensorlib = @import("./tensor.zig");
const Tensor = tensorlib.Tensor;
const tensor_forward = @import("tensor/forward.zig");
const tensor_backward = @import("tensor/backward.zig");
const backend_mod = @import("backend.zig");
const Op = @import("op.zig").Op;
const fusion = @import("fusion.zig");
const fused = @import("tensor/fused.zig");
const thread_pool_mod = @import("thread_pool.zig");
const assert = std.debug.assert;
const testing = std.testing;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

fn testCrossEntropy(comptime T: type, logits: *Tensor(T), targets: *Tensor(T)) *Tensor(T) {
    std.debug.assert(logits.isMatrix());
    std.debug.assert(targets.isVector());
    std.debug.assert(logits.ne[1] == targets.ne[0]);

    const log_probs = logits.logSoftmax(&.{ 1, logits.ne[1] });
    return log_probs.pickRows(targets).neg().mean(&.{1});
}

pub fn setBackend(comptime T: type, graph: *ComputeGraph(T), backend: backend_mod.Backend) void {
    graph.backend = backend;
}

pub fn executeNode(comptime T: type, graph: *const ComputeGraph(T), node: *Tensor(T), n_workers: usize) void {
    graph.executeNode(node, n_workers);
}

pub fn computeNoGrad(comptime T: type, graph: *const ComputeGraph(T)) void {
    graph.computeNoGrad();
}

pub fn buildForward(comptime T: type, graph: *ComputeGraph(T), root: *Tensor(T)) Alloc.Error!void {
    return graph.buildForward(root);
}

pub fn buildBackward(comptime T: type, graph: *ComputeGraph(T), keep: bool) Alloc.Error!void {
    return graph.buildBackward(keep);
}

pub fn compute(comptime T: type, graph: *const ComputeGraph(T)) void {
    graph.compute();
}

/// Manages forward and backward passes over a tensor computation graph.
///
/// All tensors should be allocated from `allocator()` so that `deinit()`
/// can free them in bulk via the arena. Graph discovery uses a visited set,
/// so shared subgraphs are recorded once even when reachable through multiple
/// parent paths.
pub fn ComputeGraph(comptime T: type) type {
    return struct {
        const Self = @This();
        const ForwardOps = tensor_forward.Ops(Tensor(T), T);

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

        /// Fusion state — populated by the graph's optimization plan.
        fused_chains: std.ArrayList(fused.FusionPlan(T)),
        /// Per-node flag: true means this node is part of a fused chain
        /// and should be skipped during normal compute iteration.
        fused_skip: std.ArrayList(bool),
        execution_steps: std.ArrayList(ExecutionStep),
        forward_execution_steps: std.ArrayList(ExecutionStep),

        /// Optional thread pool for parallel matmul and elementwise ops.
        /// Null by default (single-threaded). Call `enableThreading()` to activate.
        n_threads: usize = 1,
        thread_pool: ?*thread_pool_mod.ThreadPool = null,
        backend: ?backend_mod.Backend = null,

        const ExecutionStep = union(enum) {
            fusion: usize,
            node: *Tensor(T),
        };

        /// Set up resources for a compute graph.
        /// Use `infer()` for forward-only execution or `run()` for training.
        pub fn init(backing_alloc: Alloc) Self {
            return .{
                .arena = std.heap.ArenaAllocator.init(backing_alloc),
                .nodes = .empty,
                .grads = .empty,
                .leaves = .empty,
                .visited_nodes = .empty,
                .scratch = .empty,
                .fused_chains = .empty,
                .fused_skip = .empty,
                .execution_steps = .empty,
                .forward_execution_steps = .empty,
                .backend = null,
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

        /// Create a graph-owned tensor from a slice.
        pub fn fromSlice(self: *Self, ne: []const usize, data: []const T) !*Tensor(T) {
            return try Tensor(T).fromSlice(self.arena.allocator(), ne, data);
        }

        /// Create a graph-owned tensor filled with `val`.
        pub fn full(self: *Self, ne: []const usize, val: T) !*Tensor(T) {
            return try Tensor(T).full(self.arena.allocator(), ne, val);
        }

        /// Create a graph-owned tensor filled with zeros.
        pub fn zeros(self: *Self, ne: []const usize) !*Tensor(T) {
            return try Tensor(T).zeros(self.arena.allocator(), ne);
        }

        /// Create a graph-owned tensor filled with ones.
        pub fn ones(self: *Self, ne: []const usize) !*Tensor(T) {
            return try Tensor(T).ones(self.arena.allocator(), ne);
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

        /// Create a rank-1 graph-owned tensor containing `[start, end)` with `step`.
        pub fn arange(self: *Self, start: T, end: T, step: T) !*Tensor(T) {
            return try Tensor(T).arange(self.arena.allocator(), start, end, step);
        }

        /// Create a tensor filled with evenly spaced values.
        pub fn linspace(self: *Self, ne: []const usize, start: T, end: T) !*Tensor(T) {
            return try Tensor(T).initLinspace(self.arena.allocator(), ne, start, end);
        }

        /// Create a graph-owned tensor filled with uniform random values in `[0, 1)`.
        pub fn rand(self: *Self, rng: *std.Random, ne: []const usize) !*Tensor(T) {
            return try Tensor(T).rand(self.arena.allocator(), rng, ne);
        }

        /// Create a graph-owned tensor filled with standard normal random values.
        pub fn randn(self: *Self, rng: *std.Random, ne: []const usize) !*Tensor(T) {
            return try Tensor(T).randn(self.arena.allocator(), rng, ne);
        }

        /// Clean up all the resources for this compute graph
        pub fn deinit(self: *Self) void {
            const alloc = self.arena.allocator();
            self.syncAndRestoreParamGrads();
            if (self.thread_pool) |pool| {
                pool.deinit();
                alloc.destroy(pool);
                self.thread_pool = null;
            }
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

        fn syncAndRestoreParamGrads(self: *Self) void {
            for (self.nodes.items) |node| {
                if (node.isParam()) node.syncAndRestoreParamGrad();
            }
        }

        /// Enable multi-threaded execution for matmul and elementwise ops.
        /// Uses all available CPU cores. Safe to call multiple times (no-op if already enabled).
        pub fn enableThreading(self: *Self) void {
            if (comptime builtin.single_threaded) return;
            if (self.n_threads > 1) return;
            const n_threads = std.Thread.getCpuCount() catch 1;
            if (n_threads <= 1) return;

            const alloc = self.arena.allocator();
            const pool = alloc.create(thread_pool_mod.ThreadPool) catch {
                self.n_threads = n_threads;
                return;
            };
            pool.* = .{};
            pool.init(alloc, n_threads - 1) catch {
                alloc.destroy(pool);
                self.n_threads = n_threads;
                return;
            };

            self.thread_pool = pool;
            self.n_threads = pool.threadCount();
        }

        /// Build a graph where the provided tensor is the final output node.
        /// Shared subgraphs are deduplicated during the traversal.
        fn buildForward(self: *Self, root: *Tensor(T)) Alloc.Error!void {
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
        fn buildBackward(self: *Self, keep: bool) Alloc.Error!void {
            assert(self.nodes.items.len > 0);
            const alloc = self.arena.allocator();
            const nodes_len = self.nodes.items.len;
            for (0..nodes_len) |j| {
                const i = nodes_len - j - 1;
                const node = self.nodes.items[i];

                // because we detached the grad nodes from the original graph, we can afford inplace operations
                if (node.hasGrad()) {
                    try tensor_backward.Ops(Tensor(T), T).backward(node, alloc, &self.scratch, keep);
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
        fn fusionPass(self: *Self) Alloc.Error!void {
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

        const FusionSummary = struct {
            node_count: usize,
            forward_node_count: usize,
            fused_region_count: usize,
            leaf_count: usize,
            param_count: usize,
            aux_count: usize,
        };

        fn fusionSummary(self: *const Self) FusionSummary {
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

        fn addParentsThenSelf(self: *Self, cur: *Tensor(T)) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const visited = try self.visited_nodes.getOrPut(alloc, cur);
            if (visited.found_existing) return;
            errdefer _ = self.visited_nodes.remove(cur);
            visited.value_ptr.* = {};
            // visit parents
            if (cur.source0()) |ts0| try self.addParentsThenSelf(ts0);
            if (cur.source1()) |ts1| try self.addParentsThenSelf(ts1);
            if (cur.source2()) |ts2| try self.addParentsThenSelf(ts2);
            if (cur.source3()) |ts3| try self.addParentsThenSelf(ts3);
            if (cur.isLeaf()) {
                // is leaf
                try self.leaves.append(alloc, cur);
            } else {
                try self.nodes.append(alloc, cur);
                try self.grads.append(alloc, cur.gradOrNull());
            }
        }
        fn resetGrads(self: *Self) void {
            for (self.grads.items) |grad_o| {
                if (grad_o) |grad| {
                    _ = grad.setAllScalar(0);
                }
            }
        }
        /// Zero all intermediate node data to prepare for recomputation.
        /// Skips fused_skip nodes (their data is never read) and nodes
        /// that alias another tensor's data (e.g. reshape/view).
        fn reset(self: *Self) void {
            for (self.nodes.items, 0..) |node, i| {
                if (node.opTag() != .none and node.ownsData()) {
                    if (i < self.fused_skip.items.len and self.fused_skip.items[i]) continue;
                    _ = node.setAllScalar(0);
                }
            }
        }

        /// Execute the forward pass over all nodes.
        /// If optimization plans were built, fused chains execute as
        /// single-pass comptime-specialized kernels.
        fn compute(self: *const Self) void {
            if (self.execution_steps.items.len == 0) {
                const nw = self.n_threads;
                for (self.nodes.items) |node| self.executeNode(node, nw);
                return;
            }
            self.executeGraphPlan(self.execution_steps.items);
        }

        fn computeNoGrad(self: *const Self) void {
            if (self.forward_execution_steps.items.len == 0) {
                const nw = self.n_threads;
                for (self.nodes.items[0..self.forward_node_count]) |node| self.executeNode(node, nw);
                return;
            }
            self.executeGraphPlan(self.forward_execution_steps.items);
        }

        /// Execute only the backward portion of the graph (nodes after forward_node_count).
        /// Uses fused execution plans when available.
        fn computeBackward(self: *const Self) void {
            if (self.execution_steps.items.len == 0) {
                const nw = self.n_threads;
                for (self.nodes.items[self.forward_node_count..]) |node| self.executeNode(node, nw);
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
            const nw = self.n_threads;
            for (steps) |step| {
                switch (step) {
                    .fusion => |idx| {
                        const plan = self.fused_chains.items[idx];
                        if (nw > 1 and plan.kind() == .elementwise_chain) {
                            fused.executeFusedChainParallel(T, plan.payload.elementwise_chain, nw);
                        } else {
                            fused.executeFusionPlan(T, plan);
                        }
                    },
                    .node => |node| {
                        self.executeNode(node, nw);
                    },
                }
            }
        }

        /// Execute a single node, routing matmul through the best available path.
        ///
        /// Priority: backend > thread pool > single-threaded compute.
        /// When a backend is set it owns matmul execution entirely (the backend
        /// may use BLAS/GPU threading internally). The thread pool is a
        /// framework-level parallelism strategy for when no backend is attached.
        fn executeNode(self: *const Self, node: *Tensor(T), n_workers: usize) void {
            if (node.opTag() == .matmul) {
                const flags = node.matmul_flags;
                const s0 = node.src0.?;
                const s1 = node.src1.?;

                if (self.backend) |be| {
                    dispatchMatMul(node, s0, s1, flags, be);
                    return;
                }

                if (n_workers > 1) {
                    if (self.thread_pool) |pool| {
                        self.dispatchMatMulThreadPool(node, s0, s1, flags, pool);
                        return;
                    }
                    dispatchMatMulParallel(node, s0, s1, flags, n_workers);
                    return;
                }
            }

            node.compute();
        }

        fn dispatchMatMulThreadPool(self: *const Self, node: *Tensor(T), s0: *const Tensor(T), s1: *const Tensor(T), flags: Tensor(T).MatMulFlags, pool: *thread_pool_mod.ThreadPool) void {
            _ = self;
            if (flags.trans0) {
                if (flags.trans1) computeMatMulThreadPool(node, s0, true, s1, true, pool) else computeMatMulThreadPool(node, s0, true, s1, false, pool);
            } else {
                if (flags.trans1) computeMatMulThreadPool(node, s0, false, s1, true, pool) else computeMatMulThreadPool(node, s0, false, s1, false, pool);
            }
        }

        fn computeMatMulThreadPool(dst: *Tensor(T), src0: *const Tensor(T), comptime trans0: bool, src1: *const Tensor(T), comptime trans1: bool, pool: *thread_pool_mod.ThreadPool) void {
            ForwardOps.assertValidMatMulDims(dst, src0, trans0, src1, trans1);
            assert(dst.strides[0] == 1);

            const M = if (trans0) src0.ne[0] else src0.ne[1];
            const N = if (trans1) src1.ne[1] else src1.ne[0];
            const K = if (trans0) src0.ne[1] else src0.ne[0];
            const a_m_stride = if (trans0) src0.strides[0] else src0.strides[1];
            const a_k_stride = if (trans0) src0.strides[1] else src0.strides[0];
            const b_n_stride = if (trans1) src1.strides[1] else src1.strides[0];
            const b_k_stride = if (trans1) src1.strides[0] else src1.strides[1];
            const kernel = tensor_forward.selectMatMulRangeKernel(T);

            const min_rows_per_thread = 4;
            const n_workers = pool.threadCount();

            const Context = struct {
                dst_data: []T,
                src0_data: []const T,
                src1_data: []const T,
                chunk: usize,
                M: usize,
                N: usize,
                K: usize,
                a_m_stride: usize,
                a_k_stride: usize,
                b_k_stride: usize,
                b_n_stride: usize,
                a_base: usize,
                b_base: usize,
                d_base: usize,
                d_row_stride: usize,
                kernel: tensor_forward.MatMulRangeFnType(T),

                fn run(ctx: *@This(), task_index: usize) void {
                    const m_start = task_index * ctx.chunk;
                    const m_end = @min(m_start + ctx.chunk, ctx.M);
                    if (m_start >= m_end) return;
                    ctx.kernel(
                        ctx.dst_data,
                        ctx.src0_data,
                        ctx.src1_data,
                        m_start,
                        m_end,
                        ctx.N,
                        ctx.K,
                        ctx.a_m_stride,
                        ctx.a_k_stride,
                        ctx.b_k_stride,
                        ctx.b_n_stride,
                        ctx.a_base,
                        ctx.b_base,
                        ctx.d_base,
                        ctx.d_row_stride,
                    );
                }
            };

            for (0..src0.ne[3]) |b3| {
                for (0..src0.ne[2]) |b2| {
                    const a_base = b3 * src0.strides[3] + b2 * src0.strides[2];
                    const b_base = b3 * src1.strides[3] + b2 * src1.strides[2];
                    const d_base = b3 * dst.strides[3] + b2 * dst.strides[2];

                    if (M < min_rows_per_thread * 2 or n_workers <= 1) {
                        kernel(dst.data, src0.data, src1.data, 0, M, N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, dst.strides[1]);
                        continue;
                    }

                    const chunk = @max(min_rows_per_thread, (M + n_workers - 1) / n_workers);
                    const task_count = (M + chunk - 1) / chunk;
                    var ctx = Context{
                        .dst_data = dst.data,
                        .src0_data = src0.data,
                        .src1_data = src1.data,
                        .chunk = chunk,
                        .M = M,
                        .N = N,
                        .K = K,
                        .a_m_stride = a_m_stride,
                        .a_k_stride = a_k_stride,
                        .b_k_stride = b_k_stride,
                        .b_n_stride = b_n_stride,
                        .a_base = a_base,
                        .b_base = b_base,
                        .d_base = d_base,
                        .d_row_stride = dst.strides[1],
                        .kernel = kernel,
                    };
                    pool.parallelFor(Context, &ctx, task_count, Context.run);
                }
            }
        }

        fn dispatchMatMul(node: *Tensor(T), s0: *const Tensor(T), s1: *const Tensor(T), flags: Tensor(T).MatMulFlags, be: backend_mod.Backend) void {
            if (flags.trans0) {
                if (flags.trans1) ForwardOps.computeMatMulWithBackend(node, s0, true, s1, true, be) else ForwardOps.computeMatMulWithBackend(node, s0, true, s1, false, be);
            } else {
                if (flags.trans1) ForwardOps.computeMatMulWithBackend(node, s0, false, s1, true, be) else ForwardOps.computeMatMulWithBackend(node, s0, false, s1, false, be);
            }
        }

        fn dispatchMatMulParallel(node: *Tensor(T), s0: *const Tensor(T), s1: *const Tensor(T), flags: Tensor(T).MatMulFlags, n_workers: usize) void {
            if (flags.trans0) {
                if (flags.trans1) ForwardOps.computeMatMulParallel(node, s0, true, s1, true, n_workers) else ForwardOps.computeMatMulParallel(node, s0, true, s1, false, n_workers);
            } else {
                if (flags.trans1) ForwardOps.computeMatMulParallel(node, s0, false, s1, true, n_workers) else ForwardOps.computeMatMulParallel(node, s0, false, s1, false, n_workers);
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
    _ = testing.refAllDecls(ComputeGraph(f32));
}

//#region Tests

test "compute graph tensor factory helpers" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();

    const z = try g.zeros(&.{ 2, 2 });
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, z.data);

    const o = try g.ones(&.{3});
    try testing.expectEqualSlices(f32, &.{ 1, 1, 1 }, o.data);

    const f = try g.full(&.{2}, 7);
    try testing.expectEqualSlices(f32, &.{ 7, 7 }, f.data);

    const s = try g.fromSlice(&.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, s.data);

    const a = try g.arange(1, 6, 2);
    try testing.expectEqualSlices(f32, &.{ 1, 3, 5 }, a.data);

    var prng = std.Random.DefaultPrng.init(0);
    var rng = prng.random();
    const r = try g.randn(&rng, &.{4});
    for (r.data) |v| try testing.expect(std.math.isFinite(v));
}

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

test "threaded graph matmul matches single threaded" {
    var g_single = ComputeGraph(f32).init(tac);
    defer g_single.deinit();
    var g_threaded = ComputeGraph(f32).init(tac);
    defer g_threaded.deinit();

    const lhs_single = try Tensor(f32).init(g_single.allocator(), &.{ 5, 16 });
    const rhs_single = try Tensor(f32).init(g_single.allocator(), &.{ 7, 5 });
    const lhs_threaded = try Tensor(f32).init(g_threaded.allocator(), &.{ 5, 16 });
    const rhs_threaded = try Tensor(f32).init(g_threaded.allocator(), &.{ 7, 5 });

    for (lhs_single.data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt((i % 13) + 1)) * 0.125;
        lhs_threaded.data[i] = value.*;
    }
    for (rhs_single.data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt((i % 17) + 1)) * 0.0625;
        rhs_threaded.data[i] = value.*;
    }

    const out_single = lhs_single.matMul(false, rhs_single, false);
    const out_threaded = lhs_threaded.matMul(false, rhs_threaded, false);
    try g_single.buildForward(out_single);
    try g_threaded.buildForward(out_threaded);

    g_single.computeNoGrad();
    g_threaded.enableThreading();
    g_threaded.computeNoGrad();

    for (out_single.data, out_threaded.data) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-6);
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

test "backward - max splits ties evenly" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&[_]f32{ 2, 5, 5, 1 });
    x.setParam();
    const out = x.maxAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 5.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), x.grad.?.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), x.grad.?.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), x.grad.?.data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), x.grad.?.data[3], 1e-6);
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

test "fusion - maxpool2d backward fused matches unfused on ties" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const xf = try Tensor(f32).init(af, &.{ 4, 4, 1, 1 });
    const xu = try Tensor(f32).init(au, &.{ 4, 4, 1, 1 });
    const input = [_]f32{
        1, 3, 2, 2,
        3, 0, 0, 1,
        5, 4, 1, 1,
        5, 5, 1, 1,
    };
    xf.setData(&input);
    xu.setData(&input);
    xf.setParam();
    xu.setParam();

    const pooled_f = xf.maxPool2d();
    const pooled_u = xu.maxPool2d();
    const out_f = pooled_f.sumAll();
    const out_u = pooled_u.sumAll();

    try gf.buildForward(out_f);
    try gf.buildBackward(false);
    try gf.fusionPass();
    _ = out_f.grad.?.setAllScalar(1);

    try gu.buildForward(out_u);
    try gu.buildBackward(false);
    _ = out_u.grad.?.setAllScalar(1);

    var found_fwd = false;
    var found_bwd = false;
    for (gf.fused_chains.items) |plan| {
        switch (plan.kind()) {
            .max_pool2d => found_fwd = true,
            .max_pool2d_bwd => found_bwd = true,
            else => {},
        }
    }
    try testing.expect(found_fwd);
    try testing.expect(found_bwd);

    gf.compute();
    gu.compute();

    try testing.expectEqualSlices(f32, pooled_u.data, pooled_f.data);
    try testing.expectEqualSlices(f32, &.{ 3, 2, 5, 1 }, pooled_f.data);

    try testing.expectEqualSlices(f32, xu.grad.?.data, xf.grad.?.data);
    try testing.expectEqualSlices(f32, &.{
        0,         0.5,       0.5,  0.5,
        0.5,       0,         0,    0,
        1.0 / 3.0, 0,         0.25, 0.25,
        1.0 / 3.0, 1.0 / 3.0, 0.25, 0.25,
    }, xf.grad.?.data);
}

test "fusion - conv2d bias relu crossEntropy backward fused matches unfused" {
    var gf = ComputeGraph(f32).init(tac);
    defer gf.deinit();
    var gu = ComputeGraph(f32).init(tac);
    defer gu.deinit();

    const af = gf.allocator();
    const au = gu.allocator();

    const ks: usize = 3;
    const n_filters: usize = 2;
    const n_classes: usize = 3;
    const batch_size: usize = 2;
    const in_w: usize = 4;
    const in_h: usize = 4;
    const conv_w = in_w - ks + 1;
    const conv_h = in_h - ks + 1;
    const flat_dim = (conv_w / 2) * (conv_h / 2) * n_filters;

    const conv_k_data = [_]f32{
        0.15235855,  -0.51999205, 0.3752256,    0.47028235,  -0.9755176, -0.6510897,
        0.0639202,   -0.1581213,  -0.008400579, -0.42652196, 0.439699,   0.38889596,
        0.033015348, 0.5636206,   0.23375466,   -0.42964622, 0.18437539, -0.47944131,
    };
    const fc_w_data = [_]f32{ 0.43922514, -0.024962956, -0.09243118, -0.34046477, 0.61127067, -0.07726474 };
    const xs_data = [_]f32{
        -0.42832783, -0.35213354, 0.5323092,   0.36544406,
        0.4127326,   0.430821,    2.1416476,   -0.40641502,
        -0.51224273, -0.81377274, 0.61597943,  1.1289723,
        -0.11394746, -0.8401565,  -0.8244812,  0.6505928,
        0.7432542,   0.54315424,  -0.6655097,  0.23216133,
        0.11668581,  0.21868859,  0.8714288,   0.22359554,
        0.67891353,  0.06757907,  0.2891194,   0.63128823,
        -1.4571558,  -0.3196712,  -0.47037265, -0.63887787,
    };
    const ys_data = [_]f32{ 0.0, 2.0 };

    const conv_k_f = try Tensor(f32).init(af, &.{ ks, ks, 1, n_filters });
    const conv_k_u = try Tensor(f32).init(au, &.{ ks, ks, 1, n_filters });
    conv_k_f.setData(&conv_k_data);
    conv_k_u.setData(&conv_k_data);
    conv_k_f.setParam();
    conv_k_u.setParam();

    const conv_b_f = try Tensor(f32).init(af, &.{n_filters});
    const conv_b_u = try Tensor(f32).init(au, &.{n_filters});
    conv_b_f.setData(&.{ 0.0, 0.0 });
    conv_b_u.setData(&.{ 0.0, 0.0 });
    conv_b_f.setParam();
    conv_b_u.setParam();

    const fc_w_f = try Tensor(f32).init(af, &.{ n_classes, flat_dim });
    const fc_w_u = try Tensor(f32).init(au, &.{ n_classes, flat_dim });
    fc_w_f.setData(&fc_w_data);
    fc_w_u.setData(&fc_w_data);
    fc_w_f.setParam();
    fc_w_u.setParam();

    const fc_b_f = try Tensor(f32).init(af, &.{n_classes});
    const fc_b_u = try Tensor(f32).init(au, &.{n_classes});
    fc_b_f.setData(&.{ 0.0, 0.0, 0.0 });
    fc_b_u.setData(&.{ 0.0, 0.0, 0.0 });
    fc_b_f.setParam();
    fc_b_u.setParam();

    const xs_f = try Tensor(f32).init(af, &.{ in_w, in_h, 1, batch_size });
    const xs_u = try Tensor(f32).init(au, &.{ in_w, in_h, 1, batch_size });
    xs_f.setData(&xs_data);
    xs_u.setData(&xs_data);

    const ys_f = try Tensor(f32).init(af, &.{batch_size});
    const ys_u = try Tensor(f32).init(au, &.{batch_size});
    ys_f.setData(&ys_data);
    ys_u.setData(&ys_data);

    const conv_out_f = xs_f.conv2d(conv_k_f);
    const conv_out_u = xs_u.conv2d(conv_k_u);
    const cb4_f = conv_b_f.reshape(&.{ 1, 1, n_filters, 1 });
    const cb4_u = conv_b_u.reshape(&.{ 1, 1, n_filters, 1 });
    const act_f = conv_out_f.add(cb4_f.repeat(conv_out_f.ne[0..conv_out_f.n_dims])).relu();
    const act_u = conv_out_u.add(cb4_u.repeat(conv_out_u.ne[0..conv_out_u.n_dims])).relu();
    const flat_f = act_f.maxPool2d().reshape(&.{ flat_dim, batch_size });
    const flat_u = act_u.maxPool2d().reshape(&.{ flat_dim, batch_size });
    const logits_f = flat_f.matMul(false, fc_w_f, false).addBias(fc_b_f);
    const logits_u = flat_u.matMul(false, fc_w_u, false).addBias(fc_b_u);
    const out_f = testCrossEntropy(f32, logits_f, ys_f);
    const out_u = testCrossEntropy(f32, logits_u, ys_u);

    try gf.buildForward(out_f);
    try gf.buildBackward(false);
    try gf.fusionPass();
    _ = out_f.grad.?.setAllScalar(1);

    try gu.buildForward(out_u);
    try gu.buildBackward(false);
    _ = out_u.grad.?.setAllScalar(1);

    gf.compute();
    gu.compute();

    try testing.expectApproxEqAbs(out_u.data[0], out_f.data[0], 1e-6);
    for (logits_f.data, logits_u.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-6);
    }
    for (conv_k_f.grad.?.data, conv_k_u.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
    for (conv_b_f.grad.?.data, conv_b_u.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
    for (fc_w_f.grad.?.data, fc_w_u.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
    for (fc_b_f.grad.?.data, fc_b_u.grad.?.data) |a_out, b_out| {
        try testing.expectApproxEqAbs(a_out, b_out, 1e-5);
    }
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

test "fusion - detects logSoftmax after linear bias" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).fromSlice(a, &.{ 2, 2 }, &.{ 1, 2, -1, 0.5 });
    const w = try Tensor(f32).fromSlice(a, &.{ 3, 2 }, &.{ 0.5, 1, -1, 0.25, 1, -0.75 });
    const b = try Tensor(f32).fromSlice(a, &.{3}, &.{ 0.1, -0.2, 0.3 });
    const logits = @import("nn.zig").linear(f32, x, w, b);
    const out = logits.logSoftmaxDim(0);

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
    const targets = try Tensor(f32).initIndexVectorCopy(a, &.{ 0, 1 });

    const out = testCrossEntropy(f32, logits, targets);
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

    const out = testCrossEntropy(f32, logits, targets);
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
    const y = x.layerNorm(&.{ 1, 3 }, 1e-5).relu().gelu();
    try g.buildForward(y);
    try g.fusionPass();

    try testing.expect(g.fused_chains.items.len >= 2);
    var has_layer_norm = false;
    var has_elementwise = false;
    for (g.fused_chains.items) |chain| {
        if (chain.kind() == .layer_norm) has_layer_norm = true;
        if (chain.kind() == .elementwise_chain) has_elementwise = true;
    }
    try testing.expect(has_layer_norm);
    try testing.expect(has_elementwise);
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

test "compute softmax via composite op" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.softmax(&.{1});

    try g.buildForward(y);
    try g.fusionPass();
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

test "fusion summary reflects built graph" {
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

// Threading correctness is verified via benchmarks (enableThreading + large matmul).
// Unit tests with std.Thread.Pool under the test allocator deadlock reliably,
// so we test the single-threaded path here and rely on integration tests for threading.

//#endregion
