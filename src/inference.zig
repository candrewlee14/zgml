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

const opts = @import("zgml_options");
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

        // Shape regime for this plan. 1 for decode, N for batched prefill.
        token_len: usize,

        // Bound inputs — data patched before each execution.
        token_input: *Tensor(T),
        pos_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Identified graph nodes returned by `model.forwardCachedMaskedTrace`.
        trace: Model.CachedForwardTrace,
        logits_buf: []T,

        // Workspace reuse state.
        workspace_bufs: [][]T,

        // Quantization state (empty until quantize() is called).
        quant_weights: []QuantizedWeight(T),
        quant_map: std.AutoHashMapUnmanaged(*Tensor(T), usize),
        gemv_pool: ?quant.GemvPool(T) = null,
        // Shared f32 scratch for BLAS-backed M>1 quantized matmul.
        quant_scratch: []T = &.{},

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
            return Self.initWithBackend(model, k_caches, v_caches, backing_alloc, null, 1);
        }

        pub fn initWithBackend(
            model: *const Model,
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            backing_alloc: std.mem.Allocator,
            backend: ?backend_mod.Backend,
            token_len: usize,
        ) !Self {
            std.debug.assert(token_len >= 1 and token_len <= max_seq);
            var graph = ComputeGraph(T).init(backing_alloc);
            errdefer graph.deinit();
            if (backend) |b| graph.setBackend(b);
            const a = graph.allocator();

            // Bound-input placeholders (leaves: op=.none, never zeroed by reset).
            const token_input = try Tensor(T).init(a, &.{ d_model, token_len });
            const pos_input = try Tensor(T).init(a, &.{ d_model, token_len });
            const attn_mask = try Tensor(T).init(a, &.{ max_seq, token_len });
            @memset(attn_mask.data, -std.math.inf(T));

            const x = token_input.add(pos_input);
            const trace = model.forwardCachedMaskedTrace(x, k_caches, v_caches, 0, attn_mask);

            // Build forward graph + fusion (one-time cost).
            try graph.infer(trace.logits);

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
                .token_len = token_len,
                .token_input = token_input,
                .pos_input = pos_input,
                .attn_mask = attn_mask,
                .trace = trace,
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
            if (self.quant_scratch.len > 0) self.backing_alloc.free(self.quant_scratch);
            self.quant_map.deinit(self.backing_alloc);
            for (self.workspace_bufs) |buf| self.backing_alloc.free(buf);
            if (self.workspace_bufs.len > 0) self.backing_alloc.free(self.workspace_bufs);
            self.backing_alloc.free(self.logits_buf);
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

            if (self.quant_scratch.len > 0) alloc.free(self.quant_scratch);
            self.quant_scratch = &.{};
            if (comptime (T == f32 and opts.use_blas)) {
                var max_kn: usize = 0;
                for (qw) |w| max_kn = @max(max_kn, w.rows * w.cols);
                if (max_kn > 0) self.quant_scratch = try alloc.alloc(T, max_kn);
            }
        }

        /// Execute forward pass with quantized matmul dispatch.
        /// Uses the graph's execution plan (including fused chains) and
        /// only intercepts matmul nodes that have quantized weights.
        fn computeQuantized(self: *Self) void {
            const pool: ?*quant.GemvPool(T) = if (self.gemv_pool != null) &self.gemv_pool.? else null;
            const scratch: ?[]T = if (self.quant_scratch.len > 0) self.quant_scratch else null;
            const steps = self.graph.forward_execution_steps.items;
            if (steps.len == 0) {
                for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                    if (self.quant_map.get(node)) |qi| {
                        inference_utils.executeQuantizedMatmul(T, node, &self.quant_weights[qi], pool, scratch);
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
                            inference_utils.executeQuantizedMatmul(T, node, &self.quant_weights[qi], pool, scratch);
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
            token_ids: []const usize,
            pos: usize,
        ) []const T {
            std.debug.assert(token_ids.len == self.token_len);
            std.debug.assert(pos + self.token_len <= config.max_seq_len);
            const n = self.token_len;

            // 1. Patch token embeddings for the N positions.
            const tok_data = model.embed.token_embed.inner.data;
            for (token_ids, 0..) |token_id, i| {
                std.debug.assert(token_id < config.vocab_size);
                @memcpy(
                    self.token_input.data[i * d_model ..][0..d_model],
                    tok_data[token_id * d_model ..][0..d_model],
                );
            }

            // 2. Patch positional encodings for positions [pos..pos+n).
            const pe_data = if (config.learnable_pos_embed)
                model.embed.pos_encode.inner.data
            else
                model.embed.pos_encode.data;
            for (0..n) |i| {
                @memcpy(
                    self.pos_input.data[i * d_model ..][0..d_model],
                    pe_data[(pos + i) * d_model ..][0..d_model],
                );
            }

            // 3. Update causal mask. Column j (query at pos+j) can attend
            //    through KV position pos+j and masks the rest of the cache.
            for (0..n) |j| {
                const col = self.attn_mask.data[j * max_seq ..][0..max_seq];
                const valid_upto = pos + j + 1;
                @memset(col[0..valid_upto], 0);
                if (valid_upto < max_seq) {
                    @memset(col[valid_upto..], -std.math.inf(T));
                }
            }

            // 4. Patch KV-cache write positions.
            for (self.trace.layers) |layer_trace| {
                layer_trace.k_write.storage_offset = pos;
                layer_trace.v_write.storage_offset = pos;
            }

            // 5. Reset intermediates and execute.
            self.graph.reset();
            if (self.quant_weights.len > 0) {
                self.computeQuantized();
            } else {
                self.graph.computeNoGrad();
            }
            const last_off = (n - 1) * config.vocab_size;
            @memcpy(self.logits_buf, self.trace.logits.data[last_off..][0..config.vocab_size]);

            return self.logits_buf;
        }

    };
}

/// Default GPT prefill chunk size in tokens. The session builds one reusable
/// plan at this shape and streams full prompt chunks through it; shorter tails
/// keep using the decode plan.
pub const default_prefill_chunk: usize = 128;

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
        /// Heap-allocated so its address is stable across moves of the
        /// Session struct. Model tensors capture this allocator by pointer,
        /// so the arena must not move after tensors are created.
        arena: *std.heap.ArenaAllocator,
        backend: ?backend_mod.Backend,
        model: Model,
        k_caches: [config.n_layers]*Tensor(T),
        v_caches: [config.n_layers]*Tensor(T),
        plan: Plan,
        prefill_plan: ?Plan,
        /// Tune before `prefill()` to trade plan memory for prompt throughput.
        /// Values outside [1, max_seq_len] are clamped at use sites.
        prefill_chunk: usize,
        quant_block_size: ?usize,
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
            const arena = try backing_alloc.create(std.heap.ArenaAllocator);
            arena.* = std.heap.ArenaAllocator.init(backing_alloc);
            errdefer {
                arena.deinit();
                backing_alloc.destroy(arena);
            }
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

            var plan = try Plan.initWithBackend(&model, k_caches, v_caches, backing_alloc, backend, 1);
            errdefer plan.deinit();

            return .{
                .backing_alloc = backing_alloc,
                .arena = arena,
                .backend = backend,
                .model = model,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .plan = plan,
                .prefill_plan = null,
                .prefill_chunk = @min(default_prefill_chunk, config.max_seq_len),
                .quant_block_size = null,
                .pos = 0,
            };
        }

        /// Free the plan, model arena, and all owned memory.
        pub fn deinit(self: *Self) void {
            if (self.prefill_plan) |*p| p.deinit();
            self.plan.deinit();
            self.arena.deinit();
            self.backing_alloc.destroy(self.arena);
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
            const bs = @import("quant.zig").default_block_size;
            try self.plan.quantize(bs);
            if (self.prefill_plan) |*p| try p.quantize(bs);
            self.quant_block_size = bs;
        }

        /// Process one token and return logits over the vocabulary.
        ///
        /// Returns a `[vocab_size]` slice that remains valid until the next
        /// call to `step`.  Returns `error.SequenceTooLong` if the KV cache
        /// is full (`pos >= max_seq_len`).
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;
            const logits = self.plan.execute(&self.model, &.{token_id}, self.pos);
            self.pos += 1;
            return logits;
        }

        /// Process multiple prompt tokens and return logits for the final
        /// position. Full `prefill_chunk` runs use a reusable batched plan;
        /// a shorter tail falls through the decode plan, preserving one
        /// simple public API while avoiding graph rebuilds for long prompts.
        pub fn prefill(self: *Self, token_ids: []const usize) ![]const T {
            if (token_ids.len == 0) return self.plan.logits_buf;
            if (self.pos + token_ids.len > config.max_seq_len) return error.SequenceTooLong;

            if (self.prefill_chunk == 0) self.prefill_chunk = @min(default_prefill_chunk, config.max_seq_len);
            if (self.prefill_chunk > config.max_seq_len) self.prefill_chunk = config.max_seq_len;
            const chunk = self.prefill_chunk;

            if (token_ids.len < chunk) {
                var last: []const T = &.{};
                for (token_ids) |tok| last = try self.step(tok);
                return last;
            }

            const pp = try self.getOrBuildPrefillPlan(chunk);
            var processed: usize = 0;
            var last: []const T = &.{};
            while (processed + chunk <= token_ids.len) : (processed += chunk) {
                last = pp.execute(&self.model, token_ids[processed..][0..chunk], self.pos);
                self.pos += chunk;
            }
            while (processed < token_ids.len) : (processed += 1) {
                last = try self.step(token_ids[processed]);
            }
            return last;
        }

        fn getOrBuildPrefillPlan(self: *Self, n: usize) !*Plan {
            if (self.prefill_plan) |*p| {
                if (p.token_len == n) return p;
                p.deinit();
                self.prefill_plan = null;
            }
            var pp = try Plan.initWithBackend(
                &self.model,
                self.k_caches,
                self.v_caches,
                self.backing_alloc,
                self.backend,
                n,
            );
            errdefer pp.deinit();
            if (self.quant_block_size) |bs| try pp.quantize(bs);
            self.prefill_plan = pp;
            return &self.prefill_plan.?;
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
    prefill_session.prefill_chunk = 2;

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
    try testing.expect(prefill_session.prefill_plan != null);
    try testing.expectEqual(@as(usize, 2), prefill_session.prefill_plan.?.token_len);

    // A subsequent step() after prefill must work and match.
    const next_tok: usize = 3;
    const step_next = try step_session.step(next_tok);
    const prefill_next = try prefill_session.step(next_tok);
    for (step_next, prefill_next) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "quantized prefill stays close to quantized sequential step() calls" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 16,
    };

    const tokens = [_]usize{ 2, 5, 0, 7 };

    var step_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer step_session.deinit();
    try step_session.quantize();

    var prefill_session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer prefill_session.deinit();
    prefill_session.prefill_chunk = 2;

    for (step_session.model.params(), prefill_session.model.params()) |src, dst| {
        @memcpy(dst.data, src.data);
    }
    try prefill_session.quantize();

    var step_logits: [cfg.vocab_size]f32 = undefined;
    for (tokens) |tok| {
        const logits = try step_session.step(tok);
        @memcpy(&step_logits, logits);
    }

    const prefill_logits = try prefill_session.prefill(&tokens);
    for (step_logits[0..], prefill_logits) |want, got| {
        try testing.expectApproxEqAbs(want, got, 0.05);
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
