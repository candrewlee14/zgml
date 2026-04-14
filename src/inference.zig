//! Persistent inference for GPT models.
//!
//! `InferenceSession` is the main entry point.  It builds the computation
//! graph once, then re-executes it on every token with zero per-step
//! allocation — no graph rebuild, no weight copies, no KV-cache memcpy.
//!
//! ```
//! const Session = zgml.inference.InferenceSession(f32, config);
//!
//! var session = try Session.init(allocator);
//! defer session.deinit();
//!
//! // Load weights (writes directly into the session's model tensors).
//! checkpoint.load(f32, &session.model.params(), path);
//!
//! // Generate tokens.
//! for (0..max_tokens) |_| {
//!     const logits = try session.step(next_token);
//!     next_token = argmax(logits);
//! }
//!
//! // Start a new sequence.
//! session.reset();
//! ```
//!
//! Internally the session owns:
//!   - model weights (persistent arena)
//!   - KV caches (persistent arena, mutated in-place)
//!   - an `InferencePlan` (frozen `ComputeGraph` with workspace reuse)
//!
//! The plan patches four bound inputs each step:
//!   1. token embedding   (`[d_model, 1]`)
//!   2. positional encoding (`[d_model, 1]`)
//!   3. causal mask        (`[max_seq_len, 1]`, 0 / -inf)
//!   4. KV-cache write positions (storage_offset on slice_assign nodes)
//!
//! Then it calls `reset()` + `computeNoGrad()` on the frozen graph.

const std = @import("std");

const backend_mod = @import("backend.zig");
const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const GPT = @import("models/gpt.zig").GPT;
const GPTConfig = @import("models/gpt.zig").GPTConfig;
const QuantizedWeight = @import("quant.zig").QuantizedWeight;

/// Frozen forward-only execution plan.
///
/// Built once from a `ComputeGraph` trace of `GPT.forwardCachedMasked`.
/// Re-executed each step by patching bound inputs (token embedding,
/// positional encoding, causal mask, KV-cache write positions) and
/// calling `computeNoGrad` — no graph rebuild, no fusion re-analysis,
/// no per-step allocation.
///
/// Workspace liveness analysis assigns shared buffer slots to
/// intermediate tensors so dead buffers are recycled automatically.
pub fn InferencePlan(comptime T: type, comptime config: GPTConfig) type {
    const Model = GPT(T, config);
    const d_model = config.d_model;
    const max_seq = config.max_seq_len;

    return struct {
        const Self = @This();

        graph: ComputeGraph(T),
        backing_alloc: std.mem.Allocator,

        // Bound inputs — data patched before each execution.
        token_input: *Tensor(T),
        pos_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Cached slice_assign nodes — avoids linear scan per step.
        slice_assign_nodes: []const *Tensor(T),

        // Output tensor (read after execution).
        logits: *Tensor(T),
        logits_buf: []T,

        // Workspace reuse state.
        workspace_bufs: [][]T,

        // Quantization state (empty until quantize() is called).
        quant_weights: []QuantizedWeight(T),
        quant_map: std.AutoHashMapUnmanaged(*Tensor(T), usize),

        // Device state for quantized inference (populated by quantize()
        // when a backend with device_buffers is attached).
        device_state: ?DeviceState = null,

        const DeviceState = struct {
            be: backend_mod.Backend,
            alloc: std.mem.Allocator,
            weight_views: []backend_mod.DeviceQuantizedWeightView,
            weight_data_bufs: []backend_mod.DeviceBuffer,
            weight_scale_bufs: []backend_mod.DeviceBuffer,
            input_staging: backend_mod.DeviceBuffer,
            output_staging: backend_mod.DeviceBuffer,

            fn deinit(self: *DeviceState) void {
                for (self.weight_data_bufs) |buf| self.be.freeBuffer(buf);
                for (self.weight_scale_bufs) |buf| self.be.freeBuffer(buf);
                self.be.freeBuffer(self.input_staging);
                self.be.freeBuffer(self.output_staging);
                self.alloc.free(self.weight_views);
                self.alloc.free(self.weight_data_bufs);
                self.alloc.free(self.weight_scale_bufs);
            }
        };

        /// Build a frozen plan from an existing model and KV caches.
        ///
        /// Traces one forward pass through `forwardCachedMasked`, builds the
        /// graph, runs fusion, and optionally optimises the workspace.  The
        /// model weights and KV caches must outlive the plan (they are
        /// referenced, not copied).
        pub fn init(
            model: *const Model,
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            backing_alloc: std.mem.Allocator,
        ) !Self {
            return Self.initWithBackend(model, k_caches, v_caches, backing_alloc, null);
        }

        pub fn initWithBackend(
            model: *const Model,
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            backing_alloc: std.mem.Allocator,
            backend: ?backend_mod.Backend,
        ) !Self {
            var graph = ComputeGraph(T).init(backing_alloc);
            errdefer graph.deinit();
            if (backend) |b| graph.setBackend(b);
            const a = graph.allocator();

            // Bound-input placeholders (leaves: op=.none, never zeroed by reset).
            const token_input = try Tensor(T).init(a, &.{ d_model, 1 });
            const pos_input = try Tensor(T).init(a, &.{ d_model, 1 });
            const attn_mask = try Tensor(T).init(a, &.{ max_seq, 1 });
            @memset(attn_mask.data, -std.math.inf(T));

            const x = token_input.add(pos_input);
            const logits = model.forwardCachedMasked(x, k_caches, v_caches, 0, attn_mask);

            // Build forward graph + fusion (one-time cost).
            try graph.infer(logits);

            // Cache slice_assign node pointers for fast per-step patching.
            var sa_count: usize = 0;
            for (graph.nodes.items) |node| {
                if (node.opTag() == .slice_assign) sa_count += 1;
            }
            const sa_nodes = try backing_alloc.alloc(*Tensor(T), sa_count);
            errdefer backing_alloc.free(sa_nodes);
            var sa_idx: usize = 0;
            for (graph.nodes.items) |node| {
                if (node.opTag() == .slice_assign) {
                    sa_nodes[sa_idx] = node;
                    sa_idx += 1;
                }
            }

            // Stable output buffer — survives across steps.
            const logits_buf = try backing_alloc.alloc(T, config.vocab_size);
            errdefer backing_alloc.free(logits_buf);

            // Workspace reuse: share buffers among temporaries whose
            // lifetimes don't overlap.  Failure is non-fatal — the plan
            // works without it, just uses more memory.
            var bufs: [][]T = &.{};
            optimizeWorkspace(&graph, backing_alloc, &bufs) catch {};

            return .{
                .graph = graph,
                .backing_alloc = backing_alloc,
                .token_input = token_input,
                .pos_input = pos_input,
                .attn_mask = attn_mask,
                .slice_assign_nodes = sa_nodes,
                .logits = logits,
                .logits_buf = logits_buf,
                .workspace_bufs = bufs,
                .quant_weights = &.{},
                .quant_map = .empty,
            };
        }

        /// Free the graph, workspace buffers, and output buffer.
        /// Does NOT free the model weights or KV caches (caller owns those).
        pub fn deinit(self: *Self) void {
            if (self.device_state) |*ds| ds.deinit();
            for (self.quant_weights) |qw| qw.deinit(self.backing_alloc);
            if (self.quant_weights.len > 0) self.backing_alloc.free(self.quant_weights);
            self.quant_map.deinit(self.backing_alloc);
            for (self.workspace_bufs) |buf| self.backing_alloc.free(buf);
            if (self.workspace_bufs.len > 0) self.backing_alloc.free(self.workspace_bufs);
            self.backing_alloc.free(self.logits_buf);
            self.backing_alloc.free(self.slice_assign_nodes);
            self.graph.deinit();
        }

        /// Quantize eligible weight matmuls to int8.  Subsequent `execute()`
        /// calls use quantized matmul for those nodes — 4x weight memory
        /// reduction.  Activations, norms, and KV caches stay in f32.
        pub fn quantize(self: *Self, block_size: usize) !void {
            const alloc = self.backing_alloc;
            const nodes = self.graph.nodes.items[0..self.graph.forward_node_count];

            // Find matmul nodes where one source is a parameter.
            var count: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and isWeightMatmul(node)) count += 1;
            }
            if (count == 0) return;

            const qw = try alloc.alloc(QuantizedWeight(T), count);
            errdefer alloc.free(qw);

            // Build pointer → index map and quantize weights in one pass.
            var map: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            try map.ensureTotalCapacity(alloc, @intCast(count));

            var idx: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and isWeightMatmul(node)) {
                    const weight = if (node.src1.?.isParam()) node.src1.? else node.src0.?;
                    qw[idx] = try QuantizedWeight(T).fromTensor(alloc, weight, block_size);
                    map.putAssumeCapacity(node, idx);
                    idx += 1;
                }
            }

            // Free any prior quantization.
            for (self.quant_weights) |w| w.deinit(alloc);
            if (self.quant_weights.len > 0) alloc.free(self.quant_weights);
            self.quant_map.deinit(alloc);

            self.quant_weights = qw;
            self.quant_map = map;

            // Upload quantized weights to device if the backend supports it.
            if (self.device_state) |*ds| ds.deinit();
            self.device_state = null;
            if (self.graph.backend) |be| {
                if (be.caps().device_buffers) {
                    self.device_state = try self.initDeviceState(be);
                }
            }
        }

        /// Custom execution: iterate forward nodes, dispatching quantized
        /// matmul for weight-matmul nodes and standard compute for the rest.
        /// Bypasses fusion (minor cost — the big wins are in quantized matmul).
        fn computeQuantized(self: *Self) void {
            if (self.device_state) |*ds| {
                self.computeQuantizedDevice(ds);
                return;
            }
            for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                if (self.quant_map.get(node)) |qi| {
                    const weight = &self.quant_weights[qi];
                    if (!backend_mod.tryQuantizedMatMul(T, self.graph.backend, .{
                        .dst = node.data,
                        .input = if (node.src1.?.isParam()) node.src0.?.data else node.src1.?.data,
                        .weight = backend_mod.quantizedWeightViewF32(weight.*),
                        .M = if (node.matmul_flags.trans0) node.src0.?.ne[0] else node.src0.?.ne[1],
                        .N = if (node.matmul_flags.trans1) node.src1.?.ne[1] else node.src1.?.ne[0],
                        .K = if (node.matmul_flags.trans0) node.src0.?.ne[1] else node.src0.?.ne[0],
                    })) {
                        executeQuantizedMatmul(node, weight);
                    }
                } else {
                    self.graph.executeNode(node, null);
                }
            }
        }

        /// Device-accelerated quantized execution: quantized matmul ops
        /// dispatch through device buffers (weights resident, activations
        /// staged per-op). All other ops fall back to CPU.
        fn computeQuantizedDevice(self: *Self, ds: *DeviceState) void {
            const be = ds.be;
            for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                if (self.quant_map.get(node)) |qi| {
                    const src0 = node.src0.?;
                    const src1 = node.src1.?;
                    const flags = node.matmul_flags;
                    const input = if (src1.isParam()) src0 else src1;
                    const M = if (flags.trans0) src0.ne[0] else src0.ne[1];
                    const N = if (flags.trans1) src1.ne[1] else src1.ne[0];
                    const K = if (flags.trans0) src0.ne[1] else src0.ne[0];

                    // Upload activation input to staging buffer.
                    be.uploadSlice(T, ds.input_staging, 0, input.data[0 .. M * K]);

                    // Dispatch device quantized matmul.
                    _ = be.vtable.device_quantized_matmul_f32(be.ctx, .{
                        .dst = ds.output_staging,
                        .input = ds.input_staging,
                        .weight = ds.weight_views[qi],
                        .M = M,
                        .N = N,
                        .K = K,
                    });

                    // Download result (sync: blocks until kernel completes).
                    be.downloadSlice(T, node.data[0 .. M * N], ds.output_staging, 0);
                } else {
                    self.graph.executeNode(node, null);
                }
            }
            be.sync();
        }

        /// Upload quantized weights to device and allocate staging buffers.
        fn initDeviceState(self: *Self, be: backend_mod.Backend) !DeviceState {
            const alloc = self.backing_alloc;
            const n = self.quant_weights.len;

            const data_bufs = try alloc.alloc(backend_mod.DeviceBuffer, n);
            errdefer alloc.free(data_bufs);
            const scale_bufs = try alloc.alloc(backend_mod.DeviceBuffer, n);
            errdefer alloc.free(scale_bufs);
            const views = try alloc.alloc(backend_mod.DeviceQuantizedWeightView, n);
            errdefer alloc.free(views);

            var initialized: usize = 0;
            errdefer for (0..initialized) |j| {
                be.freeBuffer(data_bufs[j]);
                be.freeBuffer(scale_bufs[j]);
            };

            for (self.quant_weights, 0..) |qw, i| {
                // Upload i8 weight data.
                data_bufs[i] = be.allocBuffer(qw.data.len) orelse return error.OutOfMemory;
                const i8_ptr: [*]const u8 = @ptrCast(qw.data.ptr);
                be.uploadBytes(data_bufs[i], 0, i8_ptr[0..qw.data.len]);

                // Upload f32 scales.
                scale_bufs[i] = be.allocSlice(T, qw.scales.len) orelse {
                    be.freeBuffer(data_bufs[i]);
                    return error.OutOfMemory;
                };
                be.uploadSlice(T, scale_bufs[i], 0, qw.scales);

                views[i] = .{
                    .data = data_bufs[i],
                    .scales = scale_bufs[i],
                    .rows = qw.rows,
                    .cols = qw.cols,
                    .block_size = qw.block_size,
                };
                initialized += 1;
            }

            // Size staging buffers to the largest quantized matmul.
            var max_input: usize = 0;
            var max_output: usize = 0;
            for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                if (self.quant_map.get(node) != null) {
                    const flags = node.matmul_flags;
                    const s0 = node.src0.?;
                    const s1 = node.src1.?;
                    const M = if (flags.trans0) s0.ne[0] else s0.ne[1];
                    const N = if (flags.trans1) s1.ne[1] else s1.ne[0];
                    const K = if (flags.trans0) s0.ne[1] else s0.ne[0];
                    max_input = @max(max_input, M * K);
                    max_output = @max(max_output, M * N);
                }
            }

            const input_staging = be.allocSlice(T, @max(max_input, 1)) orelse return error.OutOfMemory;
            errdefer be.freeBuffer(input_staging);
            const output_staging = be.allocSlice(T, @max(max_output, 1)) orelse return error.OutOfMemory;

            return .{
                .be = be,
                .alloc = alloc,
                .weight_views = views,
                .weight_data_bufs = data_bufs,
                .weight_scale_bufs = scale_bufs,
                .input_staging = input_staging,
                .output_staging = output_staging,
            };
        }

        fn executeQuantizedMatmul(node: *Tensor(T), qw: *const QuantizedWeight(T)) void {
            const src0 = node.src0.?;
            const src1 = node.src1.?;
            const flags = node.matmul_flags;

            const input = if (src1.isParam()) src0 else src1;
            const M = if (flags.trans0) src0.ne[0] else src0.ne[1];
            const N = if (flags.trans1) src1.ne[1] else src1.ne[0];
            const K = if (flags.trans0) src0.ne[1] else src0.ne[0];

            // The quantized matmul expects:
            //   input: [M, K] row-major (= col-major [K, M])
            //   weight: [K, N] row-major (= col-major [N, K]) — stored in qw
            //   dst: [M, N] row-major (= col-major [N, M])
            //
            // Our col-major tensors have this exact flat layout, so we pass
            // data pointers directly.
            qw.matmul(input.data, node.data, M, N, K);
        }

        fn isWeightMatmul(node: *Tensor(T)) bool {
            if (node.src0) |s| { if (s.isParam()) return true; }
            if (node.src1) |s| { if (s.isParam()) return true; }
            return false;
        }

        /// Execute one step: patch inputs, reset intermediates, compute.
        /// Returns a stable logits buffer that remains valid until the next step.
        pub fn execute(
            self: *Self,
            model: *const Model,
            token_id: usize,
            pos: usize,
        ) []const T {
            std.debug.assert(token_id < config.vocab_size);
            std.debug.assert(pos < config.max_seq_len);

            // 1. Patch token embedding.
            const tok_data = model.embed.token_embed.inner.data;
            @memcpy(
                self.token_input.data[0..d_model],
                tok_data[token_id * d_model ..][0..d_model],
            );

            // 2. Patch positional encoding.
            const pe_data = if (config.learnable_pos_embed)
                model.embed.pos_encode.inner.data
            else
                model.embed.pos_encode.data;
            @memcpy(
                self.pos_input.data[0..d_model],
                pe_data[pos * d_model ..][0..d_model],
            );

            // 3. Update causal mask: 0 for positions ≤ pos, -inf otherwise.
            for (self.attn_mask.data[0..max_seq], 0..) |*v, i| {
                v.* = if (i <= pos) 0 else -std.math.inf(T);
            }

            // 4. Patch KV-cache write positions.
            for (self.slice_assign_nodes) |node| node.storage_offset = pos;

            // 5. Reset intermediates and execute.
            self.graph.reset();
            if (self.quant_weights.len > 0) {
                self.computeQuantized();
            } else {
                self.graph.computeNoGrad();
            }

            // 6. Copy to stable output buffer.
            @memcpy(self.logits_buf, self.logits.data[0..config.vocab_size]);
            return self.logits_buf;
        }

        // ---------------------------------------------------------------
        // Workspace optimisation
        // ---------------------------------------------------------------

        /// Analyse tensor liveness across the forward schedule and assign
        /// shared workspace buffers so that dead temporaries are recycled.
        fn optimizeWorkspace(
            graph: *ComputeGraph(T),
            alloc: std.mem.Allocator,
            out_bufs: *[][]T,
        ) !void {
            const nodes = graph.nodes.items[0..graph.forward_node_count];
            if (nodes.len == 0) return;

            // Build pointer → index map.
            var ptr_to_idx = std.AutoHashMap(*Tensor(T), u32).init(alloc);
            defer ptr_to_idx.deinit();
            try ptr_to_idx.ensureTotalCapacity(@intCast(nodes.len));
            for (nodes, 0..) |node, i| ptr_to_idx.putAssumeCapacity(node, @intCast(i));

            // Compute last-use step for each node.
            const last_use = try alloc.alloc(u32, nodes.len);
            defer alloc.free(last_use);
            for (0..nodes.len) |i| last_use[i] = @intCast(i);

            for (nodes, 0..) |node, step| {
                inline for (.{ node.src0, node.src1 }) |maybe_src| {
                    if (maybe_src) |s| {
                        if (ptr_to_idx.get(s)) |idx| {
                            last_use[idx] = @max(last_use[idx], @as(u32, @intCast(step)));
                        }
                    }
                }
            }

            // Mark tensors that are parents of views — their data buffers
            // must not be relocated because child views hold raw pointers
            // into them.
            const is_view_parent = try alloc.alloc(bool, nodes.len);
            defer alloc.free(is_view_parent);
            @memset(is_view_parent, false);
            for (nodes) |node| {
                const op = node.opTag();
                if (op == .as_strided or op == .view) {
                    if (node.src0) |parent| {
                        if (ptr_to_idx.get(parent)) |idx| is_view_parent[idx] = true;
                    }
                }
            }

            // Identify workspace-eligible nodes: intermediate, owned data,
            // not a view parent, not fused-skip.
            const eligible = try alloc.alloc(bool, nodes.len);
            defer alloc.free(eligible);
            for (nodes, 0..) |node, i| {
                eligible[i] = node.opTag() != .none and
                    node.ownsData() and
                    !is_view_parent[i] and
                    !(i < graph.fused_skip.items.len and graph.fused_skip.items[i]);
            }

            // Slot assignment: best-fit (smallest adequate free slot).
            const Slot = struct { free_after: u32, size: usize };
            var slots: std.ArrayList(Slot) = .{};
            defer slots.deinit(alloc);

            const assignment = try alloc.alloc(i32, nodes.len);
            defer alloc.free(assignment);
            @memset(assignment, -1);

            for (nodes, 0..) |node, step_usize| {
                if (!eligible[step_usize]) continue;
                const step: u32 = @intCast(step_usize);
                const size = node.nElems();

                var best: ?usize = null;
                for (slots.items, 0..) |slot, si| {
                    if (slot.free_after < step and slot.size >= size) {
                        if (best == null or slot.size < slots.items[best.?].size) {
                            best = si;
                        }
                    }
                }

                if (best) |si| {
                    slots.items[si].free_after = last_use[step_usize];
                    assignment[step_usize] = @intCast(si);
                } else {
                    assignment[step_usize] = @intCast(slots.items.len);
                    try slots.append(alloc, .{ .free_after = last_use[step_usize], .size = size });
                }
            }

            if (slots.items.len == 0) return;

            // Allocate workspace buffers and redirect tensor data pointers.
            const bufs = try alloc.alloc([]T, slots.items.len);
            errdefer {
                for (bufs) |b| if (b.len > 0) alloc.free(b);
                alloc.free(bufs);
            }
            for (slots.items, 0..) |slot, si| {
                bufs[si] = try alloc.alloc(T, slot.size);
            }

            for (nodes, 0..) |node, i| {
                if (assignment[i] < 0) continue;
                const si: usize = @intCast(assignment[i]);
                node.data = bufs[si][0..node.nElems()];
            }

            out_bufs.* = bufs;
        }
    };
}

/// Persistent inference session backed by a frozen `InferencePlan`.
///
/// Owns model weights, KV caches, and the plan.  Each `step` call
/// patches bound inputs and re-executes — no graph rebuild, no
/// per-token allocation.
///
/// ```
/// const Session = InferenceSession(f32, my_config);
/// var s = try Session.init(allocator);
/// defer s.deinit();
///
/// // Optionally load pretrained weights into s.model
///
/// const logits = try s.step(token_id);   // [vocab_size]
/// s.reset();                              // clear KV caches, rewind to pos 0
/// ```
pub fn InferenceSession(comptime T: type, comptime config: GPTConfig) type {
    const Model = GPT(T, config);
    const Plan = InferencePlan(T, config);

    return struct {
        const Self = @This();

        backing_alloc: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        model: Model,
        k_caches: [config.n_layers]*Tensor(T),
        v_caches: [config.n_layers]*Tensor(T),
        plan: Plan,
        pos: usize,

        /// Create a session with freshly initialised weights.
        ///
        /// To use pretrained weights, overwrite them after init:
        /// ```
        /// var s = try Session.init(alloc);
        /// checkpoint.load(f32, &s.model.params(), path);
        /// ```
        pub fn init(backing_alloc: std.mem.Allocator) !Self {
            return Self.initWithBackend(backing_alloc, null);
        }

        pub fn initWithBackend(backing_alloc: std.mem.Allocator, backend: ?backend_mod.Backend) !Self {
            var arena = std.heap.ArenaAllocator.init(backing_alloc);
            errdefer arena.deinit();
            const a = arena.allocator();

            const model = try Model.init(a);

            var k_caches: [config.n_layers]*Tensor(T) = undefined;
            var v_caches: [config.n_layers]*Tensor(T) = undefined;
            for (0..config.n_layers) |l| {
                k_caches[l] = try Tensor(T).init(a, &.{ config.d_model, config.max_seq_len });
                v_caches[l] = try Tensor(T).init(a, &.{ config.d_model, config.max_seq_len });
                @memset(k_caches[l].data, 0);
                @memset(v_caches[l].data, 0);
            }

            var plan = try Plan.initWithBackend(&model, k_caches, v_caches, backing_alloc, backend);
            errdefer plan.deinit();

            return .{
                .backing_alloc = backing_alloc,
                .arena = arena,
                .model = model,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .plan = plan,
                .pos = 0,
            };
        }

        /// Free the plan, model arena, and all owned memory.
        pub fn deinit(self: *Self) void {
            self.plan.deinit();
            self.arena.deinit();
        }

        /// Clear KV caches and rewind to position 0.
        /// Call this to start generating a new sequence.
        pub fn reset(self: *Self) void {
            self.pos = 0;
            for (0..config.n_layers) |l| {
                @memset(self.k_caches[l].data, 0);
                @memset(self.v_caches[l].data, 0);
            }
        }

        /// Current sequence position (number of tokens processed so far).
        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        /// Quantize eligible weight matrices to int8.
        /// Call after loading weights, before the first `step()`.
        pub fn quantize(self: *Self) !void {
            try self.plan.quantize(@import("quant.zig").default_block_size);
        }

        /// Process one token and return logits over the vocabulary.
        ///
        /// Returns a `[vocab_size]` slice that remains valid until the next
        /// call to `step`.  Returns `error.SequenceTooLong` if the KV cache
        /// is full (`pos >= max_seq_len`).
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;
            const logits = self.plan.execute(&self.model, token_id, self.pos);
            self.pos += 1;
            return logits;
        }

        /// Process multiple prompt tokens at once, returning logits for the
        /// last token (used to predict the next token after the prompt).
        ///
        /// This is semantically equivalent to calling `step()` for each token
        /// in sequence but expresses the intent of bulk prompt processing.
        /// The returned logits slice remains valid until the next call to
        /// `step` or `prefill`.
        ///
        /// Returns `error.SequenceTooLong` if the prompt would exceed the
        /// maximum sequence length.
        ///
        /// TODO: Replace the sequential step() loop with a true batched
        /// forward pass using `GPT.forward()` (full-sequence attention in a
        /// single `[seq, seq]` matmul instead of `seq` separate `[pos, 1]`
        /// matmuls).  This requires either (a) capturing intermediate K/V
        /// projections from the batched forward to populate the KV caches,
        /// or (b) adding a `forwardWithKVCapture` variant to
        /// `TransformerBlock`.  For prompts of 100+ tokens the batched
        /// approach would be significantly faster.
        pub fn prefill(self: *Self, token_ids: []const usize) ![]const T {
            if (token_ids.len == 0) return self.plan.logits_buf;
            if (self.pos + token_ids.len > config.max_seq_len) return error.SequenceTooLong;

            for (token_ids) |tok| {
                _ = try self.step(tok);
            }

            return self.plan.logits_buf;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "InferenceSession produces valid logits" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const tokens = [_]usize{ 0, 3, 1 };
    for (tokens) |tok| {
        const logits = try session.step(tok);
        try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
        for (logits) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "InferenceSession matches manual forward" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };
    const Model = GPT(f32, cfg);

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    // Build a reference model with identical weights in a separate arena.
    var ref_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer ref_arena.deinit();
    const ra = ref_arena.allocator();
    const ref_model = try Model.init(ra);
    for (session.model.params(), ref_model.params()) |src, dst| @memcpy(dst.data, src.data);

    var ref_k: [cfg.n_layers]*Tensor(f32) = undefined;
    var ref_v: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        ref_k[l] = try Tensor(f32).init(ra, &.{ cfg.d_model, cfg.max_seq_len });
        ref_v[l] = try Tensor(f32).init(ra, &.{ cfg.d_model, cfg.max_seq_len });
        @memset(ref_k[l].data, 0);
        @memset(ref_v[l].data, 0);
    }

    // Build a reference attn mask in the ref arena.
    const ref_mask = try Tensor(f32).init(ra, &.{ cfg.max_seq_len, 1 });

    const tokens = [_]usize{ 2, 5, 0, 7, 1 };
    for (tokens, 0..) |tok, pos| {
        // Reference: manual frozen forward in a temporary graph.
        var g = ComputeGraph(f32).init(testing.allocator);
        defer g.deinit();
        const ga = g.allocator();

        const tok_input = try Tensor(f32).init(ga, &.{ cfg.d_model, 1 });
        const pos_input = try Tensor(f32).init(ga, &.{ cfg.d_model, 1 });
        const tok_data = ref_model.embed.token_embed.inner.data;
        @memcpy(tok_input.data[0..cfg.d_model], tok_data[tok * cfg.d_model ..][0..cfg.d_model]);
        const pe_data = ref_model.embed.pos_encode.data;
        @memcpy(pos_input.data[0..cfg.d_model], pe_data[pos * cfg.d_model ..][0..cfg.d_model]);
        for (ref_mask.data[0..cfg.max_seq_len], 0..) |*v, i| {
            v.* = if (i <= pos) 0 else -std.math.inf(f32);
        }

        const x = tok_input.add(pos_input);
        const ref_logits = ref_model.forwardCachedMasked(x, ref_k, ref_v, pos, ref_mask);
        try g.infer(ref_logits);

        // Frozen session step.
        const session_logits = try session.step(tok);
        for (session_logits, ref_logits.data[0..cfg.vocab_size]) |got, want| {
            try testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

test "InferenceSession reset replays identical outputs" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const tokens = [_]usize{ 1, 2, 3 };
    var first: [tokens.len][cfg.vocab_size]f32 = undefined;

    for (tokens, 0..) |tok, i| {
        const logits = try session.step(tok);
        @memcpy(&first[i], logits);
    }

    session.reset();

    for (tokens, 0..) |tok, i| {
        const logits = try session.step(tok);
        for (logits, first[i][0..]) |got, want| {
            try testing.expectApproxEqAbs(want, got, 1e-6);
        }
    }
}

test "InferencePlan workspace reuse reduces slot count" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const n_slots = session.plan.workspace_bufs.len;
    const n_forward = session.plan.graph.forward_node_count;

    var n_intermediates: usize = 0;
    for (session.plan.graph.nodes.items[0..n_forward]) |node| {
        if (node.opTag() != .none and node.ownsData()) n_intermediates += 1;
    }

    if (n_intermediates > 2) {
        try testing.expect(n_slots < n_intermediates);
    }
}

test "Quantized InferenceSession produces valid logits" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();
    try session.quantize();

    const tokens = [_]usize{ 0, 3, 1, 5, 7 };
    for (tokens) |tok| {
        const logits = try session.step(tok);
        try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
        for (logits) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "Quantized logits are close to f32 logits" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var f32_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer f32_session.deinit();

    var q_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer q_session.deinit();

    // Same weights.
    for (f32_session.model.params(), q_session.model.params()) |src, dst| {
        @memcpy(dst.data, src.data);
    }
    try q_session.quantize();

    const tokens = [_]usize{ 2, 5, 0 };
    for (tokens) |tok| {
        const f32_logits = try f32_session.step(tok);
        const q_logits = try q_session.step(tok);
        for (f32_logits, q_logits) |f, q| {
            // int8 quantization error for small weights — allow generous tolerance.
            try testing.expectApproxEqAbs(f, q, 0.5);
        }
    }
}

test "prefill matches sequential step() calls" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    const tokens = [_]usize{ 2, 5, 0, 7, 1 };

    // Run 1: sequential step() calls.
    var step_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer step_session.deinit();

    var step_logits: [cfg.vocab_size]f32 = undefined;
    for (tokens) |tok| {
        const logits = try step_session.step(tok);
        @memcpy(&step_logits, logits);
    }

    // Run 2: single prefill() call with same weights.
    var prefill_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer prefill_session.deinit();

    // Copy weights so both sessions are identical.
    for (step_session.model.params(), prefill_session.model.params()) |src, dst| {
        @memcpy(dst.data, src.data);
    }

    const prefill_logits = try prefill_session.prefill(&tokens);

    // Logits from the last token must match exactly.
    for (step_logits[0..], prefill_logits) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-6);
    }

    // Position must be advanced correctly.
    try testing.expectEqual(tokens.len, prefill_session.pos);

    // A subsequent step() after prefill must work and match.
    const next_tok: usize = 3;
    const step_next = try step_session.step(next_tok);
    const prefill_next = try prefill_session.step(next_tok);
    for (step_next, prefill_next) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "prefill with empty slice is a no-op" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const empty: []const usize = &.{};
    _ = try session.prefill(empty);
    try testing.expectEqual(@as(usize, 0), session.pos);
}

test "prefill returns SequenceTooLong when exceeding max_seq_len" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    // 5 tokens exceeds max_seq_len of 4.
    const tokens = [_]usize{ 0, 1, 2, 3, 0 };
    const result = session.prefill(&tokens);
    try testing.expectError(error.SequenceTooLong, result);

    // Position should be unchanged after the error.
    try testing.expectEqual(@as(usize, 0), session.pos);
}
