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
const quant = @import("quant.zig");
const QuantizedWeight = quant.QuantizedWeight;
const inference_utils = @import("inference_utils.zig");

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
        gemv_pool: ?quant.GemvPool(T) = null,

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
            inference_utils.optimizeWorkspace(T, &graph, backing_alloc, &bufs) catch {};

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
            if (self.gemv_pool) |*p| p.deinit(self.backing_alloc);
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
                if (node.opTag() == .matmul and inference_utils.isWeightMatmul(T, node)) count += 1;
            }
            if (count == 0) return;

            const qw = try alloc.alloc(QuantizedWeight(T), count);
            errdefer alloc.free(qw);

            // Build pointer → index map and quantize weights in one pass.
            var map: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            try map.ensureTotalCapacity(alloc, @intCast(count));

            var idx: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and inference_utils.isWeightMatmul(T, node)) {
                    const weight = if (node.src1.?.isParam()) node.src1.? else node.src0.?;
                    qw[idx] = try QuantizedWeight(T).fromTensor(alloc, weight, block_size);
                    try qw[idx].prepareTransposed(alloc);
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
            const n_workers = std.Thread.getCpuCount() catch 1;
            if (self.gemv_pool) |*p| p.deinit(alloc);
            self.gemv_pool = try quant.GemvPool(T).init(alloc, n_workers);
        }

        /// Execute forward pass with quantized matmul dispatch.
        /// Uses the graph's execution plan (including fused chains) and
        /// only intercepts matmul nodes that have quantized weights.
        fn computeQuantized(self: *Self) void {
            const pool: ?*quant.GemvPool(T) = if (self.gemv_pool != null) &self.gemv_pool.? else null;
            const steps = self.graph.forward_execution_steps.items;
            if (steps.len == 0) {
                for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                    if (self.quant_map.get(node)) |qi| {
                        inference_utils.executeQuantizedMatmul(T, node, &self.quant_weights[qi], pool);
                    } else {
                        self.graph.executeNode(node, 1);
                    }
                }
                return;
            }
            for (steps) |step| {
                switch (step) {
                    .fusion => |idx| {
                        const plan = self.graph.fused_chains.items[idx];
                        @import("tensor/fused.zig").executeFusionPlan(T, plan, null);
                    },
                    .node => |node| {
                        if (self.quant_map.get(node)) |qi| {
                            inference_utils.executeQuantizedMatmul(T, node, &self.quant_weights[qi], pool);
                        } else {
                            self.graph.executeNode(node, 1);
                        }
                    },
                }
            }
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
            @memcpy(self.logits_buf, self.logits.data[0..config.vocab_size]);

            return self.logits_buf;
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
