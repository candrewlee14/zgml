//! Persistent inference for LLaMA models.
//!
//! `LlamaInferenceSession` is the main entry point. It builds the computation
//! graph once, then re-executes it on every token with zero per-step
//! allocation — no graph rebuild, no weight copies, no KV-cache memcpy.
//!
//! ```
//! const Session = zgml.llama_inference.LlamaInferenceSession(f32, config);
//!
//! var session = try Session.init(allocator);
//! defer session.deinit();
//!
//! // Load weights.
//! var sf = try SafetensorsFile.open(alloc, path, io);
//! try llama_loader.loadLlama(f32, config, &session.model, &sf);
//!
//! // Generate tokens.
//! for (0..max_tokens) |_| {
//!     const logits = try session.step(next_token);
//!     next_token = argmax(logits);
//! }
//!
//! session.reset();
//! ```
//!
//! Key differences from GPT InferencePlan:
//!   - Per-head KV caches: [n_layers][n_kv_heads] * [d_head, max_seq_len]
//!   - No positional encoding input (RoPE is internal to each block)
//!   - Token embedding lookup done externally, fed as [d_model, 1]

const std = @import("std");

const backend_mod = @import("backend.zig");
const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const LLaMA = @import("models/llama.zig").LLaMA;
const LlamaConfig = @import("models/llama.zig").LlamaConfig;
const quant = @import("quant.zig");
const QuantizedWeight = quant.QuantizedWeight;
const inference_utils = @import("inference_utils.zig");

/// Frozen forward-only execution plan for LLaMA models.
///
/// Built once from a `ComputeGraph` trace of `LLaMA.forwardCachedMasked`.
/// Re-executed each step by patching the token input, attention mask, and
/// KV-cache write positions.
pub fn LlamaInferencePlan(comptime T: type, comptime config: LlamaConfig) type {
    const Model = LLaMA(T, config);
    const d_model = config.d_model;
    const d_head = d_model / config.n_heads;
    const max_seq = config.max_seq_len;

    return struct {
        const Self = @This();

        graph: ComputeGraph(T),
        backing_alloc: std.mem.Allocator,

        // Bound inputs.
        token_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Cached slice_assign nodes for position patching.
        slice_assign_nodes: []const *Tensor(T),

        // RoPE packed cos_sin leaf nodes: one [2*d_head, 1] per layer.
        // Patched each step with values for the current position.
        rope_nodes: [config.n_layers]*Tensor(T),

        // Output tensor.
        logits: *Tensor(T),

        // Workspace reuse state.
        workspace_bufs: [][]T,

        // Quantization state.
        quant_weights: []QuantizedWeight(T),
        quant_map: std.AutoHashMapUnmanaged(*Tensor(T), usize),
        gemv_pool: ?quant.GemvPool(T) = null,

        pub fn init(
            model: *const Model,
            k_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
            v_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
            backing_alloc: std.mem.Allocator,
        ) !Self {
            return Self.initWithBackend(model, k_caches, v_caches, backing_alloc, null);
        }

        pub fn initWithBackend(
            model: *const Model,
            k_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
            v_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
            backing_alloc: std.mem.Allocator,
            backend: ?backend_mod.Backend,
        ) !Self {
            var graph = ComputeGraph(T).init(backing_alloc);
            errdefer graph.deinit();
            if (backend) |b| graph.setBackend(b);
            const a = graph.allocator();

            // Bound-input placeholder: token embedding [d_model, 1].
            const token_input = try Tensor(T).init(a, &.{ d_model, 1 });
            const attn_mask = try Tensor(T).init(a, &.{ max_seq, 1 });
            @memset(attn_mask.data, -std.math.inf(T));

            const logits = model.forwardCachedMasked(token_input, k_caches, v_caches, 0, attn_mask);

            // Build forward graph + fusion.
            try graph.infer(logits);

            // Cache slice_assign nodes.
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

            // Find RoPE packed cos_sin leaf nodes in the graph's leaf list.
            // They are non-param, shape [2*d_head, 1], not token_input or attn_mask.
            var rope_nodes: [config.n_layers]*Tensor(T) = undefined;
            var rope_idx: usize = 0;
            for (graph.leaves.items) |leaf| {
                if (!leaf.isParam() and
                    leaf != token_input and leaf != attn_mask and
                    leaf.ne[0] == 2 * d_head and leaf.ne[1] == 1)
                {
                    rope_nodes[rope_idx] = leaf;
                    rope_idx += 1;
                }
            }
            std.debug.assert(rope_idx == config.n_layers);

            var bufs: [][]T = &.{};
            inference_utils.optimizeWorkspace(T, &graph, backing_alloc, &bufs) catch {};

            return .{
                .graph = graph,
                .backing_alloc = backing_alloc,
                .token_input = token_input,
                .attn_mask = attn_mask,
                .slice_assign_nodes = sa_nodes,
                .rope_nodes = rope_nodes,
                .logits = logits,
                .workspace_bufs = bufs,
                .quant_weights = &.{},
                .quant_map = .empty,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.gemv_pool) |*p| p.deinit(self.backing_alloc);
            for (self.quant_weights) |qw| qw.deinit(self.backing_alloc);
            if (self.quant_weights.len > 0) self.backing_alloc.free(self.quant_weights);
            self.quant_map.deinit(self.backing_alloc);
            for (self.workspace_bufs) |buf| self.backing_alloc.free(buf);
            if (self.workspace_bufs.len > 0) self.backing_alloc.free(self.workspace_bufs);
            self.backing_alloc.free(self.slice_assign_nodes);
            self.graph.deinit();
        }

        /// Quantize eligible weight matmuls to int8.
        pub fn quantize(self: *Self, block_size: usize) !void {
            const alloc = self.backing_alloc;
            const nodes = self.graph.nodes.items[0..self.graph.forward_node_count];

            var count: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and inference_utils.isWeightMatmul(T, node)) count += 1;
            }
            if (count == 0) return;

            const qw = try alloc.alloc(QuantizedWeight(T), count);
            errdefer alloc.free(qw);

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

            for (self.quant_weights) |w| w.deinit(alloc);
            if (self.quant_weights.len > 0) alloc.free(self.quant_weights);
            self.quant_map.deinit(alloc);

            self.quant_weights = qw;
            self.quant_map = map;
            const n_workers = std.Thread.getCpuCount() catch 1;
            if (self.gemv_pool) |*p| p.deinit(alloc);
            self.gemv_pool = try quant.GemvPool(T).init(alloc, n_workers);
        }

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
            for (steps) |step_item| {
                switch (step_item) {
                    .fusion => |idx| {
                        const fplan = self.graph.fused_chains.items[idx];
                        @import("tensor/fused.zig").executeFusionPlan(T, fplan, null);
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
        pub fn execute(
            self: *Self,
            model: *const Model,
            token_id: usize,
            pos: usize,
        ) []const T {
            std.debug.assert(token_id < config.vocab_size);
            std.debug.assert(pos < config.max_seq_len);

            // 1. Patch token embedding.
            const tok_data = model.token_embed.inner.data;
            @memcpy(
                self.token_input.data[0..d_model],
                tok_data[token_id * d_model ..][0..d_model],
            );

            // 2. Extend the causal mask for the new position.
            self.attn_mask.data[pos] = 0;

            // 3. Patch RoPE packed cos_sin for current position.
            const rope = &model.blocks[0].rope;
            for (0..config.n_layers) |l| {
                @memcpy(self.rope_nodes[l].data[0..d_head], rope.cos_table.data[pos * d_head ..][0..d_head]);
                @memcpy(self.rope_nodes[l].data[d_head .. 2 * d_head], rope.sin_table.data[pos * d_head ..][0..d_head]);
            }

            // 4. Patch KV-cache write positions.
            for (self.slice_assign_nodes) |node| node.storage_offset = pos;

            // 5. Execute. Inference nodes fully overwrite their outputs, so we
            // can skip the graph-wide zeroing pass here.
            if (self.quant_weights.len > 0) {
                self.computeQuantized();
            } else {
                self.graph.computeNoGrad();
            }

            return self.logits.data[0..config.vocab_size];
        }
    };
}

/// Persistent inference session for LLaMA models.
///
/// Owns model weights, per-head KV caches, and the frozen plan.
pub fn LlamaInferenceSession(comptime T: type, comptime config: LlamaConfig) type {
    const Model = LLaMA(T, config);
    const Plan = LlamaInferencePlan(T, config);
    const d_head = config.d_model / config.n_heads;

    return struct {
        const Self = @This();

        backing_alloc: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        model: Model,
        k_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
        v_caches: [config.n_layers][config.n_kv_heads]*Tensor(T),
        plan: Plan,
        pos: usize,

        pub fn init(backing_alloc: std.mem.Allocator) !Self {
            return Self.initWithBackend(backing_alloc, null);
        }

        pub fn initWithBackend(backing_alloc: std.mem.Allocator, backend: ?backend_mod.Backend) !Self {
            var arena = std.heap.ArenaAllocator.init(backing_alloc);
            errdefer arena.deinit();
            const a = arena.allocator();

            const model = try Model.init(a);

            var k_caches: [config.n_layers][config.n_kv_heads]*Tensor(T) = undefined;
            var v_caches: [config.n_layers][config.n_kv_heads]*Tensor(T) = undefined;
            for (0..config.n_layers) |l| {
                for (0..config.n_kv_heads) |h| {
                    k_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    v_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    @memset(k_caches[l][h].data, 0);
                    @memset(v_caches[l][h].data, 0);
                }
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

        pub fn deinit(self: *Self) void {
            self.plan.deinit();
            self.arena.deinit();
        }

        /// Clear KV caches and rewind to position 0.
        pub fn reset(self: *Self) void {
            self.pos = 0;
            @memset(self.plan.attn_mask.data, -std.math.inf(T));
            for (0..config.n_layers) |l| {
                for (0..config.n_kv_heads) |h| {
                    @memset(self.k_caches[l][h].data, 0);
                    @memset(self.v_caches[l][h].data, 0);
                }
            }
        }

        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        /// Quantize eligible weight matrices to int8.
        pub fn quantize(self: *Self) !void {
            try self.plan.quantize(@import("quant.zig").default_block_size);
        }

        /// Process one token and return logits [vocab_size].
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;
            const logits = self.plan.execute(&self.model, token_id, self.pos);
            self.pos += 1;
            return logits;
        }

        /// Process multiple prompt tokens, returning logits for the last token.
        pub fn prefill(self: *Self, token_ids: []const usize) ![]const T {
            if (token_ids.len == 0) return self.plan.logits.data[0..config.vocab_size];
            if (self.pos + token_ids.len > config.max_seq_len) return error.SequenceTooLong;

            for (token_ids) |tok| {
                _ = try self.step(tok);
            }

            return self.plan.logits.data[0..config.vocab_size];
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "LlamaInferenceSession produces valid logits" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
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

test "LlamaInferenceSession reset replays identical outputs" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
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

test "LlamaInferenceSession GQA (n_kv_heads < n_heads)" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const tokens = [_]usize{ 0, 3, 7 };
    for (tokens) |tok| {
        const logits = try session.step(tok);
        try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
        for (logits) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "LlamaInferenceSession matches manual forward" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };
    const Model = LLaMA(f32, cfg);
    const d_head = cfg.d_model / cfg.n_heads;

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    // Reference model with identical weights.
    var ref_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer ref_arena.deinit();
    const ra = ref_arena.allocator();
    const ref_model = try Model.init(ra);
    for (session.model.params(), ref_model.params()) |src, dst| @memcpy(dst.data, src.data);

    var ref_k: [cfg.n_layers][cfg.n_kv_heads]*Tensor(f32) = undefined;
    var ref_v: [cfg.n_layers][cfg.n_kv_heads]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        for (0..cfg.n_kv_heads) |h| {
            ref_k[l][h] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len });
            ref_v[l][h] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len });
            @memset(ref_k[l][h].data, 0);
            @memset(ref_v[l][h].data, 0);
        }
    }

    const ref_mask = try Tensor(f32).init(ra, &.{ cfg.max_seq_len, 1 });

    const tokens = [_]usize{ 2, 5, 0, 7, 1 };
    for (tokens, 0..) |tok, pos| {
        var g = ComputeGraph(f32).init(testing.allocator);
        defer g.deinit();
        const ga = g.allocator();

        const tok_input = try Tensor(f32).init(ga, &.{ cfg.d_model, 1 });
        const tok_data = ref_model.token_embed.inner.data;
        @memcpy(tok_input.data[0..cfg.d_model], tok_data[tok * cfg.d_model ..][0..cfg.d_model]);
        for (ref_mask.data[0..cfg.max_seq_len], 0..) |*v, i| {
            v.* = if (i <= pos) 0 else -std.math.inf(f32);
        }

        const ref_logits = ref_model.forwardCachedMasked(tok_input, ref_k, ref_v, pos, ref_mask);
        try g.infer(ref_logits);

        const session_logits = try session.step(tok);
        for (session_logits, ref_logits.data[0..cfg.vocab_size]) |got, want| {
            try testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

test "LlamaInferencePlan workspace reuse reduces slot count" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
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

test "LlamaInferenceSession prefill matches sequential step()" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    const tokens = [_]usize{ 2, 5, 0, 7, 1 };

    var step_session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer step_session.deinit();

    var step_logits: [cfg.vocab_size]f32 = undefined;
    for (tokens) |tok| {
        const logits = try step_session.step(tok);
        @memcpy(&step_logits, logits);
    }

    var prefill_session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer prefill_session.deinit();

    for (step_session.model.params(), prefill_session.model.params()) |src, dst| {
        @memcpy(dst.data, src.data);
    }

    const prefill_logits = try prefill_session.prefill(&tokens);

    for (step_logits[0..], prefill_logits) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-6);
    }

    try testing.expectEqual(tokens.len, prefill_session.pos);

    const next_tok: usize = 3;
    const step_next = try step_session.step(next_tok);
    const prefill_next = try prefill_session.step(next_tok);
    for (step_next, prefill_next) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-6);
    }
}
