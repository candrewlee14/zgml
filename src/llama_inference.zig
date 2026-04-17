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
//!   - Consolidated KV cache: [n_layers] * [d_head, max_seq_len * n_kv_heads]
//!   - No positional encoding input (RoPE is internal to each block)
//!   - Token embedding lookup done externally, fed as [d_model, 1]

const std = @import("std");

const opts = @import("zgml_options");
const backend_mod = @import("backend.zig");
const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const LLaMA = @import("models/llama.zig").LLaMA;
const LlamaConfig = @import("models/llama.zig").LlamaConfig;
const quant = @import("quant.zig");
const QuantizedWeight = quant.QuantizedWeight;
const QuantizedKVCache = quant.QuantizedKVCache;
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

    const SAQuant = struct {
        cache: *QuantizedKVCache(T),
        col_offset: usize,
    };
    const AttnQuant = struct {
        k_cache: *QuantizedKVCache(T),
        v_cache: *QuantizedKVCache(T),
        k_col_offset: usize,
        v_col_offset: usize,
    };

    return struct {
        const Self = @This();

        graph: ComputeGraph(T),
        backing_alloc: std.mem.Allocator,

        // Shape regime for this plan: number of query timesteps processed per
        // execute() call. 1 for decode, N for batched prefill. Frozen at init.
        token_len: usize,

        // Bound inputs.
        token_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Base KV cache tensors (borrowed from session) — used to identify
        // slice_assign/attention nodes that target the caches.
        k_cache_tensors: [config.n_layers]*Tensor(T),
        v_cache_tensors: [config.n_layers]*Tensor(T),

        // Cached slice_assign nodes for position patching.
        slice_assign_nodes: []const *Tensor(T),

        // RoPE packed cos_sin leaf nodes: one [2*d_head, token_len] per layer.
        // Patched each call with values for the current position range.
        rope_nodes: [config.n_layers]*Tensor(T),

        // Output tensor.
        logits: *Tensor(T),

        // Workspace reuse state.
        workspace_bufs: [][]T,

        // Weight-quantization state.
        quant_weights: []QuantizedWeight(T),
        quant_map: std.AutoHashMapUnmanaged(*Tensor(T), usize),
        gemv_pool: ?quant.GemvPool(T) = null,
        // Scratch buffer for BLAS-backed M>1 quantized matmul: sized to
        // max(K*N) across quant_weights. Empty when no quant weights or
        // BLAS is disabled.
        quant_scratch: []T = &.{},

        // KV-cache-quantization state.
        k_quant_caches: []QuantizedKVCache(T),
        v_quant_caches: []QuantizedKVCache(T),
        quant_sa_map: std.AutoHashMapUnmanaged(*Tensor(T), SAQuant),
        quant_attn_map: std.AutoHashMapUnmanaged(*Tensor(T), AttnQuant),

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

            // Bound-input placeholder: token embeddings [d_model, token_len].
            const token_input = try Tensor(T).init(a, &.{ d_model, token_len });
            const attn_mask = try Tensor(T).init(a, &.{ max_seq, token_len });
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
            // They are non-param, shape [2*d_head, token_len], not token_input or attn_mask.
            var rope_nodes: [config.n_layers]*Tensor(T) = undefined;
            var rope_idx: usize = 0;
            for (graph.leaves.items) |leaf| {
                if (!leaf.isParam() and
                    leaf != token_input and leaf != attn_mask and
                    leaf.ne[0] == 2 * d_head and leaf.ne[1] == token_len)
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
                .token_len = token_len,
                .token_input = token_input,
                .attn_mask = attn_mask,
                .k_cache_tensors = k_caches,
                .v_cache_tensors = v_caches,
                .slice_assign_nodes = sa_nodes,
                .rope_nodes = rope_nodes,
                .logits = logits,
                .workspace_bufs = bufs,
                .quant_weights = &.{},
                .quant_map = .empty,
                .k_quant_caches = &.{},
                .v_quant_caches = &.{},
                .quant_sa_map = .empty,
                .quant_attn_map = .empty,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.gemv_pool) |*p| p.deinit(self.backing_alloc);
            for (self.quant_weights) |qw| qw.deinit(self.backing_alloc);
            if (self.quant_weights.len > 0) self.backing_alloc.free(self.quant_weights);
            if (self.quant_scratch.len > 0) self.backing_alloc.free(self.quant_scratch);
            self.quant_map.deinit(self.backing_alloc);
            for (self.k_quant_caches) |*c| c.deinit(self.backing_alloc);
            if (self.k_quant_caches.len > 0) self.backing_alloc.free(self.k_quant_caches);
            for (self.v_quant_caches) |*c| c.deinit(self.backing_alloc);
            if (self.v_quant_caches.len > 0) self.backing_alloc.free(self.v_quant_caches);
            self.quant_sa_map.deinit(self.backing_alloc);
            self.quant_attn_map.deinit(self.backing_alloc);
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

            // Size a shared scratch buffer for BLAS-backed M>1 matmul to
            // the largest K*N across quantized weights. Only allocated when
            // BLAS is built in — otherwise the fallback kernel is used and
            // scratch would be unused.
            if (self.quant_scratch.len > 0) alloc.free(self.quant_scratch);
            self.quant_scratch = &.{};
            if (comptime (T == f32 and opts.use_blas)) {
                var max_kn: usize = 0;
                for (qw) |w| max_kn = @max(max_kn, w.rows * w.cols);
                if (max_kn > 0) self.quant_scratch = try alloc.alloc(T, max_kn);
            }
        }

        /// Quantize each layer's KV cache to int8 with per-block f32 scales.
        /// Slice-assigns into the cache and attention reads from the cache are
        /// rerouted through `QuantizedKVCache` in the inference step — the
        /// f32 cache tensors still exist but are no longer touched.
        pub fn quantizeKV(self: *Self, block_size: usize) !void {
            const alloc = self.backing_alloc;
            const n_cols = config.max_seq_len * config.n_kv_heads;

            const k_caches = try alloc.alloc(QuantizedKVCache(T), config.n_layers);
            errdefer alloc.free(k_caches);
            const v_caches = try alloc.alloc(QuantizedKVCache(T), config.n_layers);
            errdefer alloc.free(v_caches);

            var ki: usize = 0;
            errdefer for (k_caches[0..ki]) |*c| c.deinit(alloc);
            while (ki < config.n_layers) : (ki += 1) {
                k_caches[ki] = try QuantizedKVCache(T).init(alloc, d_head, n_cols, block_size);
            }
            var vi: usize = 0;
            errdefer for (v_caches[0..vi]) |*c| c.deinit(alloc);
            while (vi < config.n_layers) : (vi += 1) {
                v_caches[vi] = try QuantizedKVCache(T).init(alloc, d_head, n_cols, block_size);
            }

            var base_k: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            defer base_k.deinit(alloc);
            var base_v: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            defer base_v.deinit(alloc);
            try base_k.ensureTotalCapacity(alloc, @intCast(config.n_layers));
            try base_v.ensureTotalCapacity(alloc, @intCast(config.n_layers));
            for (self.k_cache_tensors, 0..) |t, l| base_k.putAssumeCapacity(t, l);
            for (self.v_cache_tensors, 0..) |t, l| base_v.putAssumeCapacity(t, l);

            var sa_map: std.AutoHashMapUnmanaged(*Tensor(T), SAQuant) = .empty;
            errdefer sa_map.deinit(alloc);
            var attn_map: std.AutoHashMapUnmanaged(*Tensor(T), AttnQuant) = .empty;
            errdefer attn_map.deinit(alloc);

            const nodes = self.graph.nodes.items[0..self.graph.forward_node_count];
            for (nodes) |node| {
                switch (node.opTag()) {
                    .slice_assign => {
                        const dest = node.src1.?;
                        const info = walkToCacheBase(dest) orelse continue;
                        if (base_k.get(info.base)) |layer| {
                            try sa_map.put(alloc, node, .{
                                .cache = &k_caches[layer],
                                .col_offset = info.col_offset,
                            });
                        } else if (base_v.get(info.base)) |layer| {
                            try sa_map.put(alloc, node, .{
                                .cache = &v_caches[layer],
                                .col_offset = info.col_offset,
                            });
                        }
                    },
                    .attention => {
                        const k_info = walkToCacheBase(node.src1.?) orelse continue;
                        const v_info = walkToCacheBase(node.src2.?) orelse continue;
                        const k_layer = base_k.get(k_info.base) orelse continue;
                        const v_layer = base_v.get(v_info.base) orelse continue;
                        try attn_map.put(alloc, node, .{
                            .k_cache = &k_caches[k_layer],
                            .v_cache = &v_caches[v_layer],
                            .k_col_offset = k_info.col_offset,
                            .v_col_offset = v_info.col_offset,
                        });
                    },
                    else => {},
                }
            }

            // Swap in new state.
            for (self.k_quant_caches) |*c| c.deinit(alloc);
            if (self.k_quant_caches.len > 0) alloc.free(self.k_quant_caches);
            for (self.v_quant_caches) |*c| c.deinit(alloc);
            if (self.v_quant_caches.len > 0) alloc.free(self.v_quant_caches);
            self.quant_sa_map.deinit(alloc);
            self.quant_attn_map.deinit(alloc);

            self.k_quant_caches = k_caches;
            self.v_quant_caches = v_caches;
            self.quant_sa_map = sa_map;
            self.quant_attn_map = attn_map;
        }

        const CacheLoc = struct { base: *Tensor(T), col_offset: usize };

        /// Walk through view/slice_assign nodes to reach an allocated base tensor,
        /// and return the column offset of `start` within that base. Returns null
        /// if the chain doesn't terminate at an op-less leaf or the offset isn't
        /// column-aligned.
        fn walkToCacheBase(start: *Tensor(T)) ?CacheLoc {
            var cur = start;
            while (true) {
                const op = cur.opTag();
                if (op == .view) {
                    cur = cur.src0 orelse return null;
                } else if (op == .slice_assign or op == .slice_assign_rows) {
                    cur = cur.src1 orelse return null;
                } else break;
            }
            if (cur.data.len == 0) return null;
            const base_ptr = @intFromPtr(cur.data.ptr);
            const start_ptr = @intFromPtr(start.data.ptr);
            if (start_ptr < base_ptr) return null;
            const byte_offset = start_ptr - base_ptr;
            const row_bytes = cur.ne[0] * @sizeOf(T);
            if (row_bytes == 0 or byte_offset % row_bytes != 0) return null;
            return .{ .base = cur, .col_offset = byte_offset / row_bytes };
        }

        fn executeOneQuantized(self: *Self, node: *Tensor(T), pool: ?*quant.GemvPool(T)) void {
            if (self.quant_map.get(node)) |qi| {
                const scratch: ?[]T = if (self.quant_scratch.len > 0) self.quant_scratch else null;
                inference_utils.executeQuantizedMatmul(T, node, &self.quant_weights[qi], pool, scratch);
                return;
            }
            if (self.quant_sa_map.get(node)) |sa| {
                // Quantize src0 (shape [d_head, n_write]) into consecutive cache columns
                // starting at col_offset + storage_offset. n_write is 1 for decode and
                // N for batched prefill.
                const src = node.src0.?;
                const d = sa.cache.d_head;
                const n_write = if (src.n_dims >= 2) src.ne[1] else 1;
                const base = sa.col_offset + node.storage_offset;
                for (0..n_write) |i| {
                    sa.cache.storeColumn(base + i, src.data[i * d ..][0..d]);
                }
                return;
            }
            if (self.quant_attn_map.get(node)) |aq| {
                const q = node.src0.?;
                const mask = node.src3;
                const seq_kv = node.src1.?.ne[1];
                const mask_col_stride: usize = blk: {
                    const m = mask orelse break :blk 0;
                    if (m.ne[1] <= 1) break :blk 0;
                    break :blk m.strides[1];
                };
                quant.attentionQuantized(
                    T,
                    node.data, node.strides[1],
                    q.data, q.strides[1],
                    q.ne[0], q.ne[1],
                    aq.k_cache, aq.k_col_offset,
                    aq.v_cache, aq.v_col_offset,
                    seq_kv,
                    if (mask) |m| m.data else null,
                    if (mask) |m| m.strides[0] else 0,
                    mask_col_stride,
                    node.op_scale,
                );
                return;
            }
            self.graph.executeNode(node, 1);
        }

        fn computeQuantized(self: *Self) void {
            const pool: ?*quant.GemvPool(T) = if (self.gemv_pool != null) &self.gemv_pool.? else null;
            const steps = self.graph.forward_execution_steps.items;
            if (steps.len == 0) {
                for (self.graph.nodes.items[0..self.graph.forward_node_count]) |node| {
                    self.executeOneQuantized(node, pool);
                }
                return;
            }
            for (steps) |step_item| {
                switch (step_item) {
                    .fusion => |idx| {
                        const fplan = self.graph.fused_chains.items[idx];
                        @import("tensor/fused.zig").executeFusionPlan(T, fplan, null);
                    },
                    .node => |node| self.executeOneQuantized(node, pool),
                }
            }
        }

        /// Execute one step: patch inputs, reset intermediates, compute.
        /// Returns logits for the last (newest) query position. For decode
        /// plans (token_len=1) that's the only column; for prefill plans
        /// (token_len=N) it's the final column of the [vocab, N] output.
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
            const tok_data = model.token_embed.inner.data;
            for (token_ids, 0..) |tid, i| {
                std.debug.assert(tid < config.vocab_size);
                @memcpy(
                    self.token_input.data[i * d_model ..][0..d_model],
                    tok_data[tid * d_model ..][0..d_model],
                );
            }

            // 2. Causal mask across [max_seq, N]: column j (position pos+j) can
            //    attend to kv positions [0..pos+j]. Column-major mask; column j
            //    starts at offset j * max_seq.
            for (0..n) |j| {
                const col = self.attn_mask.data[j * config.max_seq_len ..][0..config.max_seq_len];
                const valid_upto = pos + j + 1;
                @memset(col[0..valid_upto], 0);
                if (valid_upto < config.max_seq_len) {
                    @memset(col[valid_upto..], -std.math.inf(T));
                }
            }

            // 3. Patch RoPE packed cos_sin for positions [pos..pos+n).
            const rope = &model.blocks[0].rope;
            for (0..config.n_layers) |l| {
                const buf = self.rope_nodes[l].data;
                for (0..n) |j| {
                    @memcpy(buf[j * 2 * d_head ..][0..d_head], rope.cos_table.data[(pos + j) * d_head ..][0..d_head]);
                    @memcpy(buf[j * 2 * d_head + d_head ..][0..d_head], rope.sin_table.data[(pos + j) * d_head ..][0..d_head]);
                }
            }

            // 4. KV-cache write offsets: each slice_assign writes N contiguous
            //    columns starting at pos.
            for (self.slice_assign_nodes) |node| node.storage_offset = pos;

            // 5. Execute. Inference nodes fully overwrite their outputs, so we
            //    can skip the graph-wide zeroing pass here.
            if (self.quant_weights.len > 0 or self.k_quant_caches.len > 0) {
                self.computeQuantized();
            } else {
                self.graph.computeNoGrad();
            }

            // Last-column logits: [vocab, N] column-major → last col at offset (N-1)*vocab.
            const last_off = (n - 1) * config.vocab_size;
            return self.logits.data[last_off..][0..config.vocab_size];
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
        /// Heap-allocated so its address is stable across moves of the
        /// Session struct. Model tensors capture this allocator by pointer,
        /// so the arena must not move after tensors are created.
        arena: *std.heap.ArenaAllocator,
        backend: ?backend_mod.Backend,
        model: Model,
        k_caches: [config.n_layers]*Tensor(T),
        v_caches: [config.n_layers]*Tensor(T),

        /// Decode plan: token_len=1, used by `step()`. Always present.
        plan: Plan,
        /// Lazily built prefill plan for batched prompt ingestion. Rebuilt
        /// when `prefill()` is called with a different token count.
        prefill_plan: ?Plan,
        /// Remembered weight-quant block size so prefill plans, when built
        /// lazily, can match the decode plan's quantization state.
        quant_block_size: ?usize,
        /// KV-cache quantization is active on the decode plan. When set,
        /// `prefill()` falls back to sequential step() — the two plans would
        /// each own a distinct QuantizedKVCache and drift out of sync.
        quant_kv_active: bool,
        pos: usize,

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
                k_caches[l] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len * config.n_kv_heads });
                v_caches[l] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len * config.n_kv_heads });
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
                .quant_block_size = null,
                .quant_kv_active = false,
                .pos = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.prefill_plan) |*p| p.deinit();
            self.plan.deinit();
            self.arena.deinit();
            self.backing_alloc.destroy(self.arena);
        }

        /// Clear KV caches and rewind to position 0.
        pub fn reset(self: *Self) void {
            self.pos = 0;
            @memset(self.plan.attn_mask.data, -std.math.inf(T));
            if (self.prefill_plan) |*p| @memset(p.attn_mask.data, -std.math.inf(T));
            for (0..config.n_layers) |l| {
                @memset(self.k_caches[l].data, 0);
                @memset(self.v_caches[l].data, 0);
            }
            for (self.plan.k_quant_caches) |*c| c.clear();
            for (self.plan.v_quant_caches) |*c| c.clear();
        }

        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        /// Quantize eligible weight matrices to int8 on all plans.
        pub fn quantize(self: *Self) !void {
            const bs = @import("quant.zig").default_block_size;
            try self.plan.quantize(bs);
            if (self.prefill_plan) |*p| try p.quantize(bs);
            self.quant_block_size = bs;
        }

        /// Quantize the decode plan's per-layer KV caches to int8. Any
        /// existing prefill plan is dropped: each plan owns its own
        /// QuantizedKVCache, so they can't share state without a larger
        /// refactor. Subsequent `prefill()` calls fall back to looping
        /// `step()` instead of using a batched prefill plan.
        pub fn quantizeKV(self: *Self) !void {
            try self.plan.quantizeKV(@import("quant.zig").default_block_size);
            self.quant_kv_active = true;
            if (self.prefill_plan) |*p| {
                p.deinit();
                self.prefill_plan = null;
            }
        }

        /// Process one token and return logits [vocab_size].
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;
            const logits = self.plan.execute(&self.model, &.{token_id}, self.pos);
            self.pos += 1;
            return logits;
        }

        /// Process a prompt of N tokens in a single batched forward, and
        /// return logits [vocab_size] for the final position. The batched
        /// path uses a lazily-built prefill plan of token_len=N. Plans are
        /// reused across calls with matching N; a different N rebuilds it.
        ///
        /// Falls back to sequential `step()` when N<=1 or when KV quant is
        /// active (see `quantizeKV`).
        pub fn prefill(self: *Self, token_ids: []const usize) ![]const T {
            if (token_ids.len == 0) return self.plan.logits.data[0..config.vocab_size];
            if (self.pos + token_ids.len > config.max_seq_len) return error.SequenceTooLong;

            if (token_ids.len == 1 or self.quant_kv_active) {
                var last: []const T = &.{};
                for (token_ids) |tok| last = try self.step(tok);
                return last;
            }

            const pp = try self.getOrBuildPrefillPlan(token_ids.len);
            const logits = pp.execute(&self.model, token_ids, self.pos);
            self.pos += token_ids.len;
            return logits;
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

    var ref_k: [cfg.n_layers]*Tensor(f32) = undefined;
    var ref_v: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        ref_k[l] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len * cfg.n_kv_heads });
        ref_v[l] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len * cfg.n_kv_heads });
        @memset(ref_k[l].data, 0);
        @memset(ref_v[l].data, 0);
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

test "LlamaInferenceSession quantizeKV approximates f32 path" {
    // d_head must divide evenly into block_size=32.
    const cfg = LlamaConfig{
        .vocab_size = 16,
        .d_model = 32,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 32,
        .n_layers = 2,
        .max_seq_len = 8,
    };

    var ref_session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer ref_session.deinit();

    var quant_session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer quant_session.deinit();

    for (ref_session.model.params(), quant_session.model.params()) |src, dst| {
        @memcpy(dst.data, src.data);
    }
    try quant_session.quantizeKV();

    const tokens = [_]usize{ 3, 7, 1, 0 };
    for (tokens) |tok| {
        const ref_logits = try ref_session.step(tok);
        const q_logits = try quant_session.step(tok);
        try testing.expectEqual(ref_logits.len, q_logits.len);

        // Cosine similarity: logit directions should stay close even when
        // per-element noise from Q8 KV dominates a toy model's tiny weights.
        var dot: f32 = 0;
        var nr: f32 = 0;
        var nq: f32 = 0;
        for (ref_logits, q_logits) |r, q| {
            try testing.expect(!std.math.isNan(q) and !std.math.isInf(q));
            dot += r * q;
            nr += r * r;
            nq += q * q;
        }
        const cos = dot / (@sqrt(nr) * @sqrt(nq));
        try testing.expect(cos > 0.99);
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
        try testing.expectApproxEqAbs(want, got, 1e-4);
    }

    try testing.expectEqual(tokens.len, prefill_session.pos);

    const next_tok: usize = 3;
    const step_next = try step_session.step(next_tok);
    const prefill_next = try prefill_session.step(next_tok);
    for (step_next, prefill_next) |want, got| {
        try testing.expectApproxEqAbs(want, got, 1e-4);
    }
}
