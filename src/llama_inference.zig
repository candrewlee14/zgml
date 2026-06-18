//! Persistent inference for LLaMA models.
//!
//! `LlamaInferenceSession` is the main entry point. It builds the computation
//! graph once, then re-executes it on every token with zero per-step
//! allocation — no graph rebuild, no weight copies, no KV-cache memcpy.
//!
//! ```
//! const Session = internal.llama_inference.LlamaInferenceSession(f32, config);
//!
//! var session = try Session.init(allocator);
//! defer session.deinit();
//!
//! try session.load(io, path);
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
//! Plan shape:
//!   - Consolidated KV cache: [n_layers] * [d_head, context_len * n_kv_heads]
//!   - No positional encoding input (RoPE is internal to each block)
//!   - Token embedding lookup done externally, fed as [d_model, 1]

const std = @import("std");

const opts = @import("zgml_options");
const backend_mod = @import("backend.zig");
const Tensor = @import("tensor.zig").Tensor;
const graph_mod = @import("graph.zig");
const ComputeGraph = graph_mod.ComputeGraph;
const LLaMA = @import("models/llama.zig").LLaMA;
pub const LlamaConfig = @import("models/llama.zig").LlamaConfig;
const gguf_mod = @import("gguf.zig");
const gguf_loader = @import("models/gguf_loader.zig");
const llama_loader = @import("models/llama_loader.zig");
const quant = @import("quant.zig");
const QuantizedWeight = quant.QuantizedWeight;
const QuantizedKVCache = quant.QuantizedKVCache;
const inference_utils = @import("inference_utils.zig");
const safetensors = @import("safetensors.zig");
const device_inference = @import("device_inference.zig");
const profile = @import("profile.zig");

/// Frozen forward-only execution plan for LLaMA models.
///
/// Built once from a `ComputeGraph` trace of `LLaMA.forwardCachedMasked`.
/// Re-executed each step by patching the token input, attention mask, and
/// KV-cache write positions.
fn LlamaInferencePlan(comptime T: type, comptime config: LlamaConfig) type {
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
        pub const SemanticShape = struct {
            semantic_stage_count: usize,
            semantic_token_count: usize,
            semantic_layers: usize,
            semantic_heads: usize,
            semantic_kv_heads: usize,
            semantic_layer_stage_count: usize,
            semantic_terminal_stage_count: usize,
            semantic_runtime_patch_holes: usize,
            semantic_runtime_patch_cache_write_pos_holes: usize,
            semantic_runtime_patch_attention_seq_kv_holes: usize,

            fn fromCounts(
                token_count: usize,
                layer_count: usize,
                cache_write_pos_holes: usize,
                attention_seq_kv_holes: usize,
            ) SemanticShape {
                return .{
                    .semantic_stage_count = layer_count * semantic_layer_stage_count + semantic_terminal_stage_count,
                    .semantic_token_count = token_count,
                    .semantic_layers = layer_count,
                    .semantic_heads = config.n_heads,
                    .semantic_kv_heads = config.n_kv_heads,
                    .semantic_layer_stage_count = semantic_layer_stage_count,
                    .semantic_terminal_stage_count = semantic_terminal_stage_count,
                    .semantic_runtime_patch_holes = cache_write_pos_holes + attention_seq_kv_holes,
                    .semantic_runtime_patch_cache_write_pos_holes = cache_write_pos_holes,
                    .semantic_runtime_patch_attention_seq_kv_holes = attention_seq_kv_holes,
                };
            }
        };

        graph: ComputeGraph(T),
        backing_alloc: std.mem.Allocator,

        // Shape regime for this plan: number of query timesteps processed per
        // execute() call. 1 for decode, N for batched prefill. Frozen at init.
        token_len: usize,

        // Bound inputs.
        token_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Identified graph nodes returned by `model.forwardCachedMasked`.
        // The trace is the single source of truth for dynamic inputs, KV-cache
        // writes, and attention nodes; no graph walking or shape guessing.
        trace: Model.CachedForwardTrace,
        k_caches: [config.n_layers]*Tensor(T),
        v_caches: [config.n_layers]*Tensor(T),
        device_logits_output: *Tensor(T),
        context_len: usize,

        // Workspace reuse state.
        workspace_bufs: [][]T,

        // Weight-quantization state.
        quant_weights: []QuantizedWeight(T),
        owns_quant_weights: bool,
        quant_map: std.AutoHashMapUnmanaged(*Tensor(T), usize),
        gemv_pool: ?quant.GemvPool(T) = null,
        // Scratch buffer for BLAS-backed M>1 quantized matmul: sized to
        // max(K*N) across quant_weights. Empty when no quant weights or
        // BLAS is disabled.
        quant_scratch: []T = &.{},

        // KV-cache-quantization state. The caches themselves are session-owned
        // so the decode and prefill plans share the same physical storage;
        // plans only hold per-graph-node maps into them.
        uses_quant_kv: bool,
        quant_sa_map: std.AutoHashMapUnmanaged(*Tensor(T), SAQuant),
        quant_attn_map: std.AutoHashMapUnmanaged(*Tensor(T), AttnQuant),

        const semantic_layer_stage_count = 7;
        const semantic_terminal_stage_count = 2;

        const PatchIntent = enum {
            graph_execute,
            device_prefill,
            device_decode,
        };

        const DeviceMode = enum { decode, prefill };

        const DeviceCompileBinding = struct {
            graph: *ComputeGraph(T),
            input_tensors: [3]*const Tensor(T),
            input_count: usize,
            output_tensor: *const Tensor(T),
            quant_weights: []const QuantizedWeight(T),
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor(T), usize),
            expected_runtime_patch_shape: backend_mod.RuntimePatchShape,

            fn inputs(self: *const DeviceCompileBinding) []const *const Tensor(T) {
                return self.input_tensors[0..self.input_count];
            }
        };

        fn cacheContextLen(k_caches: [config.n_layers]*Tensor(T), v_caches: [config.n_layers]*Tensor(T)) !usize {
            if (config.n_kv_heads == 0) return error.ShapeMismatch;
            const first = k_caches[0];
            if (first.n_dims < 2 or first.ne[0] != d_head or first.ne[1] % config.n_kv_heads != 0) return error.ShapeMismatch;
            const context_len = first.ne[1] / config.n_kv_heads;
            if (context_len == 0 or context_len > max_seq) return error.InvalidContextLength;
            for (0..config.n_layers) |l| {
                if (k_caches[l].n_dims < 2 or v_caches[l].n_dims < 2) return error.ShapeMismatch;
                if (k_caches[l].ne[0] != d_head or v_caches[l].ne[0] != d_head) return error.ShapeMismatch;
                if (k_caches[l].ne[1] != context_len * config.n_kv_heads or v_caches[l].ne[1] != context_len * config.n_kv_heads) return error.ShapeMismatch;
            }
            return context_len;
        }

        fn initWithBackend(
            model: *const Model,
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            backing_alloc: std.mem.Allocator,
            backend: ?backend_mod.Backend,
            token_len: usize,
        ) !Self {
            const context_len = try cacheContextLen(k_caches, v_caches);
            if (token_len == 0 or token_len > context_len) return error.InvalidPrefillLength;
            var graph = ComputeGraph(T).init(backing_alloc);
            errdefer graph.deinit();
            if (backend) |b| graph_mod.setBackend(T, &graph, b);
            const a = graph.allocator();

            // Bound-input placeholder: token embeddings [d_model, token_len].
            const token_input = try Tensor(T).init(a, &.{ d_model, token_len });
            const attn_mask = try Tensor(T).init(a, &.{ context_len, token_len });
            @memset(attn_mask.data, -std.math.inf(T));

            const trace = model.forwardCachedMasked(token_input, k_caches, v_caches, 0, attn_mask);

            // Build forward graph + fusion.
            try graph.infer(trace.logits);

            var bufs: [][]T = &.{};
            inference_utils.optimizeWorkspace(T, &graph, backing_alloc, &bufs) catch {};
            const device_logits_output = if (token_len == 1)
                trace.logits
            else
                trace.logits.sliceColumns(token_len - 1, token_len);

            return .{
                .graph = graph,
                .backing_alloc = backing_alloc,
                .token_len = token_len,
                .token_input = token_input,
                .attn_mask = attn_mask,
                .trace = trace,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .device_logits_output = device_logits_output,
                .context_len = context_len,
                .workspace_bufs = bufs,
                .quant_weights = &.{},
                .owns_quant_weights = true,
                .quant_map = .empty,
                .uses_quant_kv = false,
                .quant_sa_map = .empty,
                .quant_attn_map = .empty,
            };
        }

        fn deinit(self: *Self) void {
            self.clearQuantization();
            self.quant_sa_map.deinit(self.backing_alloc);
            self.quant_attn_map.deinit(self.backing_alloc);
            for (self.workspace_bufs) |buf| self.backing_alloc.free(buf);
            if (self.workspace_bufs.len > 0) self.backing_alloc.free(self.workspace_bufs);
            self.graph.deinit();
        }

        fn clearQuantization(self: *Self) void {
            const alloc = self.backing_alloc;
            if (self.gemv_pool) |*p| p.deinit(alloc);
            self.gemv_pool = null;
            if (self.owns_quant_weights) {
                for (self.quant_weights) |qw| qw.deinit(alloc);
                if (self.quant_weights.len > 0) alloc.free(self.quant_weights);
            }
            self.quant_weights = &.{};
            self.owns_quant_weights = true;
            if (self.quant_scratch.len > 0) alloc.free(self.quant_scratch);
            self.quant_scratch = &.{};
            self.quant_map.deinit(alloc);
            self.quant_map = .empty;
        }

        fn setupQuantRuntime(self: *Self) !void {
            const alloc = self.backing_alloc;
            const n_workers = std.Thread.getCpuCount() catch 1;
            self.gemv_pool = try quant.GemvPool(T).init(alloc, n_workers);

            if (comptime (T == f32 and opts.use_blas)) {
                var max_kn: usize = 0;
                for (self.quant_weights) |w| max_kn = @max(max_kn, w.rows * w.cols);
                if (max_kn > 0) self.quant_scratch = try alloc.alloc(T, max_kn);
            }
        }

        /// Quantize eligible weight matmuls to int8.
        fn quantize(self: *Self, block_size: usize) !void {
            const alloc = self.backing_alloc;
            const nodes = self.graph.nodes.items[0..self.graph.forward_node_count];

            var count: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and inference_utils.isWeightMatmul(T, node)) count += 1;
            }
            if (count == 0) return;

            const qw = try alloc.alloc(QuantizedWeight(T), count);
            var qw_transferred = false;
            errdefer if (!qw_transferred) alloc.free(qw);
            var initialized: usize = 0;
            errdefer if (!qw_transferred) {
                for (qw[0..initialized]) |w| w.deinit(alloc);
            };

            var map: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            errdefer map.deinit(alloc);
            try map.ensureTotalCapacity(alloc, @intCast(count));

            var idx: usize = 0;
            for (nodes) |node| {
                if (node.opTag() == .matmul and inference_utils.isWeightMatmul(T, node)) {
                    const weight = if (node.src1.?.isParam()) node.src1.? else node.src0.?;
                    qw[idx] = try QuantizedWeight(T).fromTensor(alloc, weight, block_size);
                    initialized += 1;
                    try qw[idx].prepareTransposed(alloc);
                    map.putAssumeCapacity(node, idx);
                    idx += 1;
                }
            }

            self.clearQuantization();
            self.quant_weights = qw;
            self.owns_quant_weights = true;
            self.quant_map = map;
            map = .empty;
            initialized = 0;
            qw_transferred = true;
            try self.setupQuantRuntime();
        }

        fn externalQuantWeightIndex(
            weights: []QuantizedWeight(T),
            param_map: *const std.AutoHashMapUnmanaged(*Tensor(T), quant.QuantizedWeightBinding),
            node: *Tensor(T),
        ) ?usize {
            if (node.opTag() != .matmul) return null;
            const src0 = node.src0.?;
            const src1 = node.src1.?;
            const flags = node.matmul_flags;
            const N = if (flags.trans1) src1.ne[1] else src1.ne[0];
            const K = if (flags.trans0) src0.ne[1] else src0.ne[0];

            if (!src1.isParam()) return null;
            const required_layout: quant.QuantizedWeightLayout = if (flags.trans1) .transposed_param else .native_param;
            const binding = param_map.get(src1) orelse return null;
            if (binding.layout != required_layout) return null;
            if (binding.index >= weights.len) return null;
            if (weights[binding.index].rows != K or weights[binding.index].cols != N) return null;
            return binding.index;
        }

        fn useExternalQuantWeights(
            self: *Self,
            weights: []QuantizedWeight(T),
            param_map: *const std.AutoHashMapUnmanaged(*Tensor(T), quant.QuantizedWeightBinding),
        ) !void {
            const alloc = self.backing_alloc;
            const nodes = self.graph.nodes.items[0..self.graph.forward_node_count];

            var count: usize = 0;
            for (nodes) |node| {
                if (externalQuantWeightIndex(weights, param_map, node) != null) count += 1;
            }
            if (count == 0) return;

            var map: std.AutoHashMapUnmanaged(*Tensor(T), usize) = .empty;
            errdefer map.deinit(alloc);
            try map.ensureTotalCapacity(alloc, @intCast(count));

            for (nodes) |node| {
                if (externalQuantWeightIndex(weights, param_map, node)) |qi| map.putAssumeCapacity(node, qi);
            }

            self.clearQuantization();
            self.quant_weights = weights;
            self.owns_quant_weights = false;
            self.quant_map = map;
            map = .empty;
            try self.setupQuantRuntime();
        }

        /// Build the slice_assign / attention → QuantizedKVCache maps for this
        /// plan's graph nodes using caches owned by the session. Plans share
        /// the same physical caches so a single session can hold both a
        /// decode plan and a prefill plan without duplicated state.
        ///
        /// The trace tells us which graph node is which KV write / attention
        /// read for which (layer, kv_head), so the mapping is a direct index
        /// lookup — no graph walking, no pointer arithmetic on tensor data.
        fn quantizeKV(
            self: *Self,
            k_caches: []QuantizedKVCache(T),
            v_caches: []QuantizedKVCache(T),
        ) !void {
            std.debug.assert(k_caches.len == config.n_layers);
            std.debug.assert(v_caches.len == config.n_layers);
            const alloc = self.backing_alloc;

            // One entry per (layer, kv_h) write and per (layer, head) read.
            const sa_capacity = 2 * config.n_layers * config.n_kv_heads;
            const attn_capacity = config.n_layers * config.n_heads;

            var sa_map: std.AutoHashMapUnmanaged(*Tensor(T), SAQuant) = .empty;
            errdefer sa_map.deinit(alloc);
            try sa_map.ensureTotalCapacity(alloc, @intCast(sa_capacity));

            var attn_map: std.AutoHashMapUnmanaged(*Tensor(T), AttnQuant) = .empty;
            errdefer attn_map.deinit(alloc);
            try attn_map.ensureTotalCapacity(alloc, @intCast(attn_capacity));

            const n_rep = config.n_heads / config.n_kv_heads;
            for (self.trace.layers, 0..) |layer_trace, l| {
                for (0..config.n_kv_heads) |kv_h| {
                    const col = kv_h * self.context_len;
                    sa_map.putAssumeCapacity(layer_trace.k_write[kv_h], .{
                        .cache = &k_caches[l],
                        .col_offset = col,
                    });
                    sa_map.putAssumeCapacity(layer_trace.v_write[kv_h], .{
                        .cache = &v_caches[l],
                        .col_offset = col,
                    });
                }
                for (0..config.n_heads) |h| {
                    const kv_h = h / n_rep;
                    const col = kv_h * self.context_len;
                    attn_map.putAssumeCapacity(layer_trace.attention[h], .{
                        .k_cache = &k_caches[l],
                        .v_cache = &v_caches[l],
                        .k_col_offset = col,
                        .v_col_offset = col,
                    });
                }
            }

            self.quant_sa_map.deinit(alloc);
            self.quant_attn_map.deinit(alloc);
            self.quant_sa_map = sa_map;
            self.quant_attn_map = attn_map;
            self.uses_quant_kv = true;
        }

        fn runtimePatchHoleCounts(self: *const Self) struct { cache_write_pos: usize, attention_seq_kv: usize } {
            var cache_write_pos: usize = 0;
            var attention_seq_kv: usize = 0;
            for (self.trace.layers) |layer_trace| {
                cache_write_pos += layer_trace.k_write.len + layer_trace.v_write.len;
                attention_seq_kv += layer_trace.attention.len;
            }
            return .{
                .cache_write_pos = cache_write_pos,
                .attention_seq_kv = attention_seq_kv,
            };
        }

        pub fn semanticShape(self: *const Self) SemanticShape {
            const holes = self.runtimePatchHoleCounts();
            return SemanticShape.fromCounts(
                self.token_len,
                self.trace.layers.len,
                holes.cache_write_pos,
                holes.attention_seq_kv,
            );
        }

        fn semanticShapeForTokenCount(token_count: usize) SemanticShape {
            return SemanticShape.fromCounts(
                token_count,
                config.n_layers,
                config.n_layers * 2 * config.n_kv_heads,
                config.n_layers * config.n_heads,
            );
        }

        fn expectedRuntimePatchShape(self: *const Self) !backend_mod.RuntimePatchShape {
            const shape = self.semanticShape();
            const cache_holes = std.math.cast(u32, shape.semantic_runtime_patch_cache_write_pos_holes) orelse return error.RuntimePatchShapeMismatch;
            const attention_holes = std.math.cast(u32, shape.semantic_runtime_patch_attention_seq_kv_holes) orelse return error.RuntimePatchShapeMismatch;
            const expected = backend_mod.RuntimePatchShape.expectedDynamic(cache_holes, attention_holes);
            if (expected.runtime_patch_holes != shape.semantic_runtime_patch_holes) return error.RuntimePatchShapeMismatch;
            return expected;
        }

        fn deviceCompileBinding(self: *Self, mode: DeviceMode) !DeviceCompileBinding {
            if (mode == .decode) @memset(self.attn_mask.data, 0);
            const decode_inputs = [3]*const Tensor(T){ self.token_input, self.trace.rope, self.trace.rope };
            const prefill_inputs = [3]*const Tensor(T){ self.token_input, self.attn_mask, self.trace.rope };
            return .{
                .graph = &self.graph,
                .input_tensors = if (mode == .decode) decode_inputs else prefill_inputs,
                .input_count = if (mode == .decode) 2 else 3,
                .output_tensor = self.device_logits_output,
                .quant_weights = self.quant_weights,
                .quant_map = &self.quant_map,
                .expected_runtime_patch_shape = try self.expectedRuntimePatchShape(),
            };
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
                    node.data,
                    node.strides[1],
                    q.data,
                    q.strides[1],
                    q.ne[0],
                    q.ne[1],
                    aq.k_cache,
                    aq.k_col_offset,
                    aq.v_cache,
                    aq.v_col_offset,
                    seq_kv,
                    if (mask) |m| m.data else null,
                    if (mask) |m| m.strides[0] else 0,
                    mask_col_stride,
                    node.op_scale,
                );
                return;
            }
            graph_mod.executeNode(T, &self.graph, node, 1);
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
                        @import("tensor/fused.zig").executeFusionPlan(T, fplan);
                    },
                    .node => |node| self.executeOneQuantized(node, pool),
                }
            }
        }

        fn patchExecutionInputs(self: *Self, model: *const Model, token_ids: []const usize, pos: usize, intent: PatchIntent) !void {
            const end = std.math.add(usize, pos, token_ids.len) catch return error.SequenceTooLong;
            if (end > self.context_len) return error.SequenceTooLong;
            try self.patchTokenInputs(model, token_ids);
            if (intent != .device_decode) self.patchCausalMask(pos);
            self.patchRope(model, pos);
            if (intent == .graph_execute) self.patchKvWriteOffsets(pos);
        }

        fn patchTokenInputs(self: *Self, model: *const Model, token_ids: []const usize) !void {
            if (token_ids.len != self.token_len) return error.InvalidTokenWindow;
            for (token_ids) |tid| {
                if (tid >= config.vocab_size) return error.TokenIdOutOfRange;
            }
            const tok_data = model.token_embed.data;
            for (token_ids, 0..) |tid, i| {
                @memcpy(
                    self.token_input.data[i * d_model ..][0..d_model],
                    tok_data[tid * d_model ..][0..d_model],
                );
            }
        }

        fn patchCausalMask(self: *Self, pos: usize) void {
            std.debug.assert(pos + self.token_len <= self.context_len);
            for (0..self.token_len) |j| {
                const col = self.attn_mask.data[j * self.attn_mask.strides[1] ..][0..self.context_len];
                const valid_upto = pos + j + 1;
                @memset(col[0..valid_upto], 0);
                if (valid_upto < self.context_len) @memset(col[valid_upto..], -std.math.inf(T));
            }
        }

        fn patchRope(self: *Self, model: *const Model, pos: usize) void {
            std.debug.assert(pos + self.token_len <= config.max_seq_len);
            const rope = &model.blocks[0].rope;
            const buf = self.trace.rope.data;
            for (0..self.token_len) |j| {
                @memcpy(buf[j * 2 * d_head ..][0..d_head], rope.cos_table.data[(pos + j) * d_head ..][0..d_head]);
                @memcpy(buf[j * 2 * d_head + d_head ..][0..d_head], rope.sin_table.data[(pos + j) * d_head ..][0..d_head]);
            }
        }

        /// Patch host graph offsets for host/quantized-host execution. Compiled
        /// backend programs receive token windows through `RuntimeWindow`.
        fn patchKvWriteOffsets(self: *Self, pos: usize) void {
            for (self.trace.layers) |layer_trace| {
                for (layer_trace.k_write) |node| node.storage_offset = pos;
                for (layer_trace.v_write) |node| node.storage_offset = pos;
            }
        }

        /// Execute one step: patch inputs, reset intermediates, compute.
        /// Returns logits for the last (newest) query position. For decode
        /// plans (token_len=1) that's the only column; for prefill plans
        /// (token_len=N) it's the final column of the [vocab, N] output.
        fn execute(
            self: *Self,
            model: *const Model,
            token_ids: []const usize,
            pos: usize,
        ) ![]const T {
            try self.patchExecutionInputs(model, token_ids, pos, .graph_execute);

            // 5. Execute. Inference nodes fully overwrite their outputs, so we
            //    can skip the graph-wide zeroing pass here.
            if (self.quant_weights.len > 0 or self.uses_quant_kv) {
                self.computeQuantized();
            } else {
                graph_mod.computeNoGrad(T, &self.graph);
            }

            // Last-column logits: [vocab, N] column-major → last col at offset (N-1)*vocab.
            const n = self.token_len;
            const last_off = (n - 1) * config.vocab_size;
            return self.trace.logits.data[last_off..][0..config.vocab_size];
        }
    };
}

/// Default prefill chunk size in tokens. A single prefill plan is built once
/// at this shape and reused across all prompts; tails shorter than the chunk
/// flow through `step()`. 128 balances SGEMM efficiency, plan-build cost, and
/// memory.
pub const default_prefill_chunk: usize = 128;

/// Persistent inference session for LLaMA models.
///
/// Owns model weights, per-head KV caches, and the frozen plan.
pub fn LlamaInferenceSession(comptime T: type, comptime config: LlamaConfig) type {
    const Model = LLaMA(T, config);
    const Plan = LlamaInferencePlan(T, config);
    const d_head = config.d_model / config.n_heads;

    return struct {
        const Self = @This();
        pub const DeviceDecodeProgram = LlamaDeviceProgram(T, config);
        pub const DeviceDecodeSession = LlamaDeviceSession(T, config);
        backing_alloc: std.mem.Allocator,
        /// Heap-allocated so its address is stable across moves of the
        /// Session struct. Model tensors capture this allocator by pointer,
        /// so the arena must not move after tensors are created.
        arena: *std.heap.ArenaAllocator,
        backend: ?backend_mod.Backend,
        context_len: usize,
        model: Model,
        k_caches: [config.n_layers]*Tensor(T),
        v_caches: [config.n_layers]*Tensor(T),

        /// Session-owned quantized KV caches, shared by all plans. Empty
        /// until `quantizeKV()` is called.
        k_quant_caches: []QuantizedKVCache(T),
        v_quant_caches: []QuantizedKVCache(T),

        /// Decode plan: token_len=1, used by `step()`. Always present.
        plan: Plan,
        /// Lazily built fixed-size prefill plan reused across prompts.
        prefill_plan: ?Plan,
        /// Remembered weight-quant block size so prefill plans, when built
        /// lazily, can match the decode plan's quantization state.
        quant_block_size: ?usize,
        direct_quant_weights: []QuantizedWeight(T),
        direct_quant_param_map: std.AutoHashMapUnmanaged(*Tensor(T), quant.QuantizedWeightBinding),
        pos: usize,

        pub fn init(backing_alloc: std.mem.Allocator) !Self {
            return Self.initWithBackend(backing_alloc, null);
        }

        pub fn initWithBackend(backing_alloc: std.mem.Allocator, backend: ?backend_mod.Backend) !Self {
            return Self.initWithBackendAndContext(backing_alloc, backend, config.max_seq_len);
        }

        pub fn initWithBackendAndContext(backing_alloc: std.mem.Allocator, backend: ?backend_mod.Backend, context_len: usize) !Self {
            if (context_len == 0 or context_len > config.max_seq_len) return error.InvalidContextLength;
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
                k_caches[l] = try Tensor(T).init(a, &.{ d_head, context_len * config.n_kv_heads });
                v_caches[l] = try Tensor(T).init(a, &.{ d_head, context_len * config.n_kv_heads });
                @memset(k_caches[l].data, 0);
                @memset(v_caches[l].data, 0);
            }

            var plan = try Plan.initWithBackend(&model, k_caches, v_caches, backing_alloc, backend, 1);
            errdefer plan.deinit();

            return .{
                .backing_alloc = backing_alloc,
                .arena = arena,
                .backend = backend,
                .context_len = context_len,
                .model = model,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .k_quant_caches = &.{},
                .v_quant_caches = &.{},
                .plan = plan,
                .prefill_plan = null,
                .quant_block_size = null,
                .direct_quant_weights = &.{},
                .direct_quant_param_map = .empty,
                .pos = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.prefill_plan) |*p| p.deinit();
            self.plan.deinit();
            self.deinitDirectQuantStorage();
            for (self.k_quant_caches) |*c| c.deinit(self.backing_alloc);
            if (self.k_quant_caches.len > 0) self.backing_alloc.free(self.k_quant_caches);
            for (self.v_quant_caches) |*c| c.deinit(self.backing_alloc);
            if (self.v_quant_caches.len > 0) self.backing_alloc.free(self.v_quant_caches);
            self.arena.deinit();
            self.backing_alloc.destroy(self.arena);
        }

        fn deinitDirectQuantStorage(self: *Self) void {
            for (self.direct_quant_weights) |qw| qw.deinit(self.backing_alloc);
            if (self.direct_quant_weights.len > 0) self.backing_alloc.free(self.direct_quant_weights);
            self.direct_quant_weights = &.{};
            self.direct_quant_param_map.deinit(self.backing_alloc);
            self.direct_quant_param_map = .empty;
        }

        fn copyModelParams(src: *const Model, dst: *Model) !void {
            const src_params = src.params();
            const dst_params = dst.params();
            for (src_params, dst_params) |src_param, dst_param| {
                if (src_param.data.len != dst_param.data.len) return error.ShapeMismatch;
                @memcpy(dst_param.data, src_param.data);
            }
        }

        fn cloneDirectQuantStorageFrom(self: *Self, source: *const Self) !void {
            const alloc = self.backing_alloc;
            const src_weights = source.direct_quant_weights;
            if (src_weights.len == 0) return;

            const cloned_weights = try alloc.alloc(QuantizedWeight(T), src_weights.len);
            var cloned_count: usize = 0;
            errdefer {
                for (cloned_weights[0..cloned_count]) |qw| qw.deinit(alloc);
                alloc.free(cloned_weights);
            }
            for (src_weights, 0..) |qw, i| {
                cloned_weights[i] = try qw.clone(alloc);
                cloned_count += 1;
            }

            var cloned_map: std.AutoHashMapUnmanaged(*Tensor(T), quant.QuantizedWeightBinding) = .empty;
            errdefer cloned_map.deinit(alloc);
            try cloned_map.ensureTotalCapacity(alloc, @intCast(source.direct_quant_param_map.count()));

            const src_params = source.model.params();
            const dst_params = self.model.params();
            var mapped: usize = 0;
            for (src_params, dst_params) |src_param, dst_param| {
                if (source.direct_quant_param_map.get(src_param)) |binding| {
                    cloned_map.putAssumeCapacity(dst_param, binding);
                    mapped += 1;
                }
            }
            if (mapped != source.direct_quant_param_map.count()) return error.ShapeMismatch;

            try self.plan.useExternalQuantWeights(cloned_weights, &cloned_map);
            self.direct_quant_weights = cloned_weights;
            self.direct_quant_param_map = cloned_map;
            cloned_map = .empty;
            cloned_count = 0;
        }

        /// Bind a fresh runtime session with the same persistent weights and
        /// backend placement as this session, but independent KV caches,
        /// position, plans, and runtime profile state.
        pub fn bindRuntime(self: *const Self, backing_alloc: std.mem.Allocator) !Self {
            return self.bindRuntimeWithContext(backing_alloc, self.context_len);
        }

        pub fn bindRuntimeWithContext(self: *const Self, backing_alloc: std.mem.Allocator, context_len: usize) !Self {
            var runtime = try Self.initWithBackendAndContext(backing_alloc, self.backend, context_len);
            errdefer runtime.deinit();

            try copyModelParams(&self.model, &runtime.model);
            if (self.direct_quant_weights.len > 0) {
                try runtime.cloneDirectQuantStorageFrom(self);
                runtime.quant_block_size = null;
            } else if (self.quant_block_size) |bs| {
                try runtime.plan.quantize(bs);
                runtime.quant_block_size = bs;
            }
            if (self.k_quant_caches.len > 0) try runtime.quantizeKV();
            runtime.reset();
            return runtime;
        }

        fn clearWeightQuantization(self: *Self) void {
            self.plan.clearQuantization();
            if (self.prefill_plan) |*p| p.clearQuantization();
            self.deinitDirectQuantStorage();
            self.quant_block_size = null;
        }

        /// Clear KV caches and rewind to position 0. Attention masks are
        /// fully rewritten on every `execute()` call, so they need no reset.
        pub fn reset(self: *Self) void {
            self.pos = 0;
            for (0..config.n_layers) |l| {
                @memset(self.k_caches[l].data, 0);
                @memset(self.v_caches[l].data, 0);
            }
            for (self.k_quant_caches) |*c| c.clear();
            for (self.v_quant_caches) |*c| c.clear();
        }

        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        pub fn semanticShape(self: *const Self) Plan.SemanticShape {
            return self.plan.semanticShape();
        }

        /// Quantize eligible weight matrices to int8 on all plans.
        pub fn quantize(self: *Self) !void {
            if (self.direct_quant_weights.len > 0) return error.DirectQuantizedWeightsActive;
            const bs = @import("quant.zig").default_block_size;
            try self.plan.quantize(bs);
            if (self.prefill_plan) |*p| try p.quantize(bs);
            self.quant_block_size = bs;
        }

        /// Load GGUF tensors for inference while keeping supported quantized
        /// matmul weights compressed. Non-matmul tensors and unsupported
        /// formats are dequantized into the model's f32 parameter tensors.
        fn loadGGUFDirectQuantized(self: *Self, gf: *const gguf_mod.GGUFFile) !void {
            var loaded = try gguf_loader.loadDirectQuantized(T, config, self.backing_alloc, &self.model, gf);
            errdefer loaded.deinit(self.backing_alloc);

            self.clearWeightQuantization();
            try self.plan.useExternalQuantWeights(loaded.weights, &loaded.param_map);
            var plan_installed = true;
            errdefer if (plan_installed) self.plan.clearQuantization();
            if (self.prefill_plan) |*p| {
                try p.useExternalQuantWeights(loaded.weights, &loaded.param_map);
                var prefill_installed = true;
                errdefer if (prefill_installed) p.clearQuantization();
                prefill_installed = false;
            }

            self.direct_quant_weights = loaded.weights;
            self.direct_quant_param_map = loaded.param_map;
            loaded.weights = &.{};
            loaded.param_map = .empty;
            self.quant_block_size = null;
            self.reset();
            plan_installed = false;
        }

        pub fn loadSafetensorsFile(self: *Self, sf: *const safetensors.SafetensorsFile) !void {
            self.clearWeightQuantization();
            try llama_loader.loadLlama(T, config, &self.model, sf);
            self.reset();
        }

        fn loadSafetensors(self: *Self, io: std.Io, path: []const u8) !void {
            var sf = try safetensors.SafetensorsFile.open(self.backing_alloc, path, io);
            defer sf.deinit();
            try self.loadSafetensorsFile(&sf);
        }

        pub fn loadSafetensorsBytes(self: *Self, bytes: []const u8) !void {
            var sf = try safetensors.SafetensorsFile.fromBytes(self.backing_alloc, bytes);
            defer sf.deinit();
            try self.loadSafetensorsFile(&sf);
        }

        fn loadGGUF(self: *Self, io: std.Io, path: []const u8) !void {
            var gf = try gguf_mod.GGUFFile.open(self.backing_alloc, io, path);
            defer gf.deinit();
            try self.loadGGUFDirectQuantized(&gf);
        }

        pub fn load(self: *Self, io: std.Io, path: []const u8) !void {
            if (std.ascii.endsWithIgnoreCase(path, ".safetensors")) return self.loadSafetensors(io, path);
            if (std.ascii.endsWithIgnoreCase(path, ".gguf")) return self.loadGGUF(io, path);
            return error.UnknownModelFormat;
        }

        /// Quantize the session's per-layer KV caches to int8. The caches are
        /// session-owned and shared by the decode plan and any prefill plan,
        /// so `prefill()` can still use the batched path after this call.
        pub fn quantizeKV(self: *Self) !void {
            const alloc = self.backing_alloc;
            const block_size = @import("quant.zig").default_block_size;
            const n_cols = self.context_len * config.n_kv_heads;

            // Caches are allocated at most once per session; calling quantizeKV
            // twice is a no-op on the storage and just rebuilds the plan maps.
            if (self.k_quant_caches.len == 0) {
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

                self.k_quant_caches = k_caches;
                self.v_quant_caches = v_caches;
            }

            try self.plan.quantizeKV(self.k_quant_caches, self.v_quant_caches);
            if (self.prefill_plan) |*p| try p.quantizeKV(self.k_quant_caches, self.v_quant_caches);
        }

        fn patchDeviceInputs(
            self: *Self,
            plan_to_patch: *Plan,
            token_ids: []const usize,
            pos_to_patch: usize,
            mode: Plan.DeviceMode,
        ) !backend_mod.RuntimeWindow {
            try plan_to_patch.patchExecutionInputs(
                &self.model,
                token_ids,
                pos_to_patch,
                switch (mode) {
                    .decode => .device_decode,
                    .prefill => .device_prefill,
                },
            );
            return backend_mod.RuntimeWindow.init(pos_to_patch, token_ids.len);
        }

        /// Process one token and return logits [vocab_size].
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= self.context_len) return error.SequenceTooLong;
            const logits = try self.plan.execute(&self.model, &.{token_id}, self.pos);
            self.pos += 1;
            return logits;
        }

        fn validateTokenIds(token_ids: []const usize) !void {
            for (token_ids) |tok| {
                if (tok >= config.vocab_size) return error.TokenIdOutOfRange;
            }
        }

        /// Ingest a prompt of N tokens and return logits [vocab_size] for
        /// the final position.
        ///
        /// The prompt is split into fixed-size chunks. Full chunks are
        /// executed through a single batched plan built once and reused for
        /// every prompt; a tail shorter than the chunk falls back to `step()`.
        /// Two plans exist for the session's lifetime — decode and prefill —
        /// so varying prompt lengths never trigger graph rebuilds.
        pub fn prefill(self: *Self, token_ids: []const usize) ![]const T {
            if (token_ids.len == 0) return self.plan.trace.logits.data[0..config.vocab_size];
            const end = std.math.add(usize, self.pos, token_ids.len) catch return error.SequenceTooLong;
            if (end > self.context_len) return error.SequenceTooLong;
            try validateTokenIds(token_ids);

            const chunk = @min(default_prefill_chunk, self.context_len);
            // Short prompts flow entirely through step(). Building a
            // chunk-sized plan would be wasted work.
            if (token_ids.len < chunk) {
                var last: []const T = &.{};
                for (token_ids) |tok| last = try self.step(tok);
                return last;
            }

            const pp = try self.getOrBuildPrefillPlan(chunk);
            var processed: usize = 0;
            var last: []const T = &.{};
            while (processed + chunk <= token_ids.len) : (processed += chunk) {
                last = try pp.execute(&self.model, token_ids[processed..][0..chunk], self.pos);
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
            if (self.direct_quant_weights.len > 0) {
                try pp.useExternalQuantWeights(self.direct_quant_weights, &self.direct_quant_param_map);
            } else if (self.quant_block_size) |bs| {
                try pp.quantize(bs);
            }
            if (self.k_quant_caches.len > 0) {
                try pp.quantizeKV(self.k_quant_caches, self.v_quant_caches);
            }
            self.prefill_plan = pp;
            return &self.prefill_plan.?;
        }

        /// Ensure a fixed-size batched prefill plan exists and return it.
        /// This is primarily for backend/device execution layers that want the
        /// same frozen graph as `prefill()` without first running a prompt.
        fn ensurePrefillPlan(self: *Self, n: usize) !*Plan {
            if (n == 0 or n > self.context_len) return error.InvalidPrefillLength;
            return self.getOrBuildPrefillPlan(n);
        }

        pub fn compileDeviceDecode(self: *Self, backend: backend_mod.Backend, alloc: std.mem.Allocator, logits_buf: []T) !LlamaDevice(T, config) {
            return LlamaDevice(T, config).initDecode(.{
                .session = self,
                .backend = backend,
                .alloc = alloc,
                .logits_buf = logits_buf,
            });
        }

        pub fn compileDeviceDecodeProgram(self: *Self, backend: backend_mod.Backend, alloc: std.mem.Allocator) !LlamaDeviceProgram(T, config) {
            return LlamaDeviceProgram(T, config).initDecode(.{
                .session = self,
                .backend = backend,
                .alloc = alloc,
            });
        }

        pub fn compileDevicePrefillProgram(self: *Self, backend: backend_mod.Backend, alloc: std.mem.Allocator, chunk_tokens: usize) !LlamaDeviceProgram(T, config) {
            return LlamaDeviceProgram(T, config).initPrefill(.{
                .session = self,
                .backend = backend,
                .alloc = alloc,
                .chunk_tokens = chunk_tokens,
            });
        }

        pub fn compileDevicePrefill(self: *Self, backend: backend_mod.Backend, alloc: std.mem.Allocator, chunk_tokens: usize, logits_buf: []T) !LlamaDevice(T, config) {
            return LlamaDevice(T, config).initPrefill(.{
                .session = self,
                .backend = backend,
                .alloc = alloc,
                .chunk_tokens = chunk_tokens,
                .logits_buf = logits_buf,
            });
        }
    };
}

pub fn LlamaDeviceProgram(comptime T: type, comptime config: LlamaConfig) type {
    const TensorT = Tensor(T);
    const Session = LlamaInferenceSession(T, config);
    const Plan = LlamaInferencePlan(T, config);
    const Device = device_inference.DeviceInference(T);

    return struct {
        const Self = @This();

        owner: *Session,
        alloc: std.mem.Allocator,
        plan: *Plan,
        mode: Plan.DeviceMode,
        program: Device.Program,
        input_tensors: [3]*const TensorT,
        input_count: usize,
        output_tensor: *const TensorT,
        persistent_tensors: []*const TensorT,

        const DecodeOptions = struct {
            session: *Session,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
        };

        const PrefillOptions = struct {
            session: *Session,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            chunk_tokens: usize,
        };

        const InitOptions = struct {
            session: *Session,
            plan: *Plan,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            mode: Plan.DeviceMode,
        };

        fn cacheElementCountForContextChecked(context_len: usize) ?usize {
            const d_head = config.d_model / config.n_heads;
            const columns = std.math.mul(usize, context_len, config.n_kv_heads) catch return null;
            return std.math.mul(usize, d_head, columns) catch null;
        }

        pub const full_cache_element_count = cacheElementCountForContextChecked(config.max_seq_len) orelse @compileError("LLaMA KV cache element count overflows usize");

        pub fn cacheElementCountForContext(context_len: usize) !usize {
            if (context_len == 0 or context_len > config.max_seq_len) return error.InvalidContextLength;
            return cacheElementCountForContextChecked(context_len) orelse error.ShapeMismatch;
        }

        pub const CacheResourceBindings = struct {
            k: [config.n_layers]backend_mod.ProgramIO.ExternalResource,
            v: [config.n_layers]backend_mod.ProgramIO.ExternalResource,
            element_count: usize = full_cache_element_count,
        };

        pub const CacheHostBindings = struct {
            k: [config.n_layers][*]T,
            v: [config.n_layers][*]T,
            element_count: usize = full_cache_element_count,
        };

        pub const CacheBindings = union(enum) {
            host: CacheHostBindings,
            resource: CacheResourceBindings,
        };

        pub const BindOptions = struct {
            logits_buf: []T,
            output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
            cache_bindings: ?CacheBindings = null,
        };

        pub fn initDecode(options: DecodeOptions) !Self {
            return init(.{
                .session = options.session,
                .plan = &options.session.plan,
                .backend = options.backend,
                .alloc = options.alloc,
                .mode = .decode,
            });
        }

        pub fn initPrefill(options: PrefillOptions) !Self {
            const plan = try options.session.ensurePrefillPlan(options.chunk_tokens);
            return init(.{
                .session = options.session,
                .plan = plan,
                .backend = options.backend,
                .alloc = options.alloc,
                .mode = .prefill,
            });
        }

        fn init(options: InitOptions) !Self {
            const binding = try options.plan.deviceCompileBinding(options.mode);
            const params = options.session.model.params();
            const cache_tensor_count = config.n_layers * 2;
            const persistent_tensors = try options.alloc.alloc(*const TensorT, params.len + cache_tensor_count);
            errdefer options.alloc.free(persistent_tensors);
            for (params, 0..) |param, i| persistent_tensors[i] = param;
            for (0..config.n_layers) |l| {
                persistent_tensors[params.len + l] = options.plan.k_caches[l];
                persistent_tensors[params.len + config.n_layers + l] = options.plan.v_caches[l];
            }

            var program = try Device.Program.compile(.{
                .graph = binding.graph,
                .be = options.backend,
                .alloc = options.alloc,
                .input_tensors = binding.inputs(),
                .persistent_tensors = persistent_tensors,
                .output_tensors = &.{binding.output_tensor},
                .quant_weights = binding.quant_weights,
                .quant_map = binding.quant_map,
                .expected_runtime_patch_shape = binding.expected_runtime_patch_shape,
            });
            errdefer program.deinit();

            return .{
                .owner = options.session,
                .alloc = options.alloc,
                .plan = options.plan,
                .mode = options.mode,
                .program = program,
                .input_tensors = binding.input_tensors,
                .input_count = binding.input_count,
                .output_tensor = binding.output_tensor,
                .persistent_tensors = persistent_tensors,
            };
        }

        pub fn deinit(self: *Self) void {
            self.program.deinit();
            self.alloc.free(self.persistent_tensors);
            self.* = undefined;
        }

        pub fn inspectExecutable(self: *const Self) Device.ProgramInspection {
            return self.program.inspect();
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.program.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile.RuntimeProfile) void {
            self.program.addRuntimeProfileTo(dest);
        }

        pub fn semanticShape(self: *const Self) Plan.SemanticShape {
            return self.plan.semanticShape();
        }

        pub fn bind(self: *Self, session: *Session, logits_buf: []T) !LlamaDeviceSession(T, config) {
            return self.bindWithOptions(session, .{ .logits_buf = logits_buf });
        }

        pub fn bindWithOptions(self: *Self, session: *Session, options: BindOptions) !LlamaDeviceSession(T, config) {
            if (options.logits_buf.len < config.vocab_size) return error.OutputBufferTooSmall;
            const runtime_plan = switch (self.mode) {
                .decode => &session.plan,
                .prefill => try session.ensurePrefillPlan(self.plan.token_len),
            };
            const runtime_binding = try runtime_plan.deviceCompileBinding(self.mode);
            if (runtime_binding.input_count != self.input_count) return error.InvalidProgramIO;

            var input_bindings: [3]Device.TensorBinding = undefined;
            for (0..self.input_count) |i| {
                const runtime_tensor = runtime_binding.input_tensors[i];
                input_bindings[i] = .{
                    .tensor = self.input_tensors[i],
                    .host_ptr = runtime_tensor.data.ptr,
                    .element_count = runtime_tensor.data.len,
                };
            }

            const runtime_params = session.model.params();
            const cache_tensor_count = config.n_layers * 2;
            const expected_persistent = runtime_params.len + cache_tensor_count;
            if (self.persistent_tensors.len != expected_persistent) return error.InvalidProgramIO;
            const persistent_bindings = try self.alloc.alloc(Device.TensorBinding, expected_persistent);
            defer self.alloc.free(persistent_bindings);
            for (runtime_params, 0..) |runtime_param, i| {
                const compile_param = self.persistent_tensors[i];
                if (runtime_param.data.len != compile_param.data.len) return error.ShapeMismatch;
                persistent_bindings[i] = .{
                    .tensor = compile_param,
                    .host_ptr = runtime_param.data.ptr,
                    .element_count = runtime_param.data.len,
                };
            }
            for (0..config.n_layers) |l| {
                const k_index = runtime_params.len + l;
                const v_index = runtime_params.len + config.n_layers + l;
                const compile_k = self.persistent_tensors[k_index];
                const compile_v = self.persistent_tensors[v_index];
                const runtime_k = session.k_caches[l];
                const runtime_v = session.v_caches[l];
                if (runtime_k.data.len != compile_k.data.len or runtime_v.data.len != compile_v.data.len) return error.ShapeMismatch;
                if (options.cache_bindings) |cache_bindings| switch (cache_bindings) {
                    .resource => |resources| {
                        if (resources.element_count == 0 or resources.element_count > compile_k.data.len or resources.element_count > compile_v.data.len) return error.ShapeMismatch;
                        persistent_bindings[k_index] = Device.TensorBinding.externalResource(compile_k, resources.k[l], resources.element_count);
                        persistent_bindings[v_index] = Device.TensorBinding.externalResource(compile_v, resources.v[l], resources.element_count);
                    },
                    .host => |buffers| {
                        if (buffers.element_count == 0 or buffers.element_count > compile_k.data.len or buffers.element_count > compile_v.data.len) return error.ShapeMismatch;
                        persistent_bindings[k_index] = Device.TensorBinding.host(compile_k, buffers.k[l], buffers.element_count);
                        persistent_bindings[v_index] = Device.TensorBinding.host(compile_v, buffers.v[l], buffers.element_count);
                    },
                } else {
                    persistent_bindings[k_index] = .{
                        .tensor = compile_k,
                        .host_ptr = runtime_k.data.ptr,
                        .element_count = runtime_k.data.len,
                    };
                    persistent_bindings[v_index] = .{
                        .tensor = compile_v,
                        .host_ptr = runtime_v.data.ptr,
                        .element_count = runtime_v.data.len,
                    };
                }
            }

            const output_binding = if (options.output_resource) |resource|
                Device.TensorBinding.externalResource(self.output_tensor, resource, config.vocab_size)
            else
                null;
            var device_session = try self.program.bind(.{
                .input_tensors = self.input_tensors[0..self.input_count],
                .input_bindings = input_bindings[0..self.input_count],
                .persistent_tensors = self.persistent_tensors,
                .persistent_bindings = persistent_bindings,
                .quant_weights = if (self.program.be.capabilities.runtime_qweights) runtime_binding.quant_weights else &.{},
                .output_tensor = self.output_tensor,
                .output_host_buf = options.logits_buf.ptr,
                .output_len = config.vocab_size,
                .output_binding = output_binding,
            });
            errdefer device_session.deinit();

            if (self.mode == .prefill) {
                const original_outputs = device_session.bindings.step_outputs;
                if (original_outputs.len != 1) return error.InvalidProgramIO;
                const outputs = try self.alloc.alloc(backend_mod.ProgramIO, 1 + cache_tensor_count);
                var outputs_owned_by_session = false;
                errdefer if (!outputs_owned_by_session) self.alloc.free(outputs);
                outputs[0] = original_outputs[0];
                for (0..config.n_layers) |l| {
                    if (options.cache_bindings) |cache_bindings| switch (cache_bindings) {
                        .resource => |resources| {
                            if (resources.element_count == 0 or resources.element_count > self.plan.k_caches[l].data.len or resources.element_count > self.plan.v_caches[l].data.len) return error.ShapeMismatch;
                            outputs[1 + l] = try self.program.tensorResourceBindingIO(self.plan.k_caches[l], resources.k[l], resources.element_count);
                            outputs[1 + config.n_layers + l] = try self.program.tensorResourceBindingIO(self.plan.v_caches[l], resources.v[l], resources.element_count);
                        },
                        .host => |buffers| {
                            if (buffers.element_count == 0 or buffers.element_count > self.plan.k_caches[l].data.len or buffers.element_count > self.plan.v_caches[l].data.len) return error.ShapeMismatch;
                            outputs[1 + l] = try self.program.tensorBindingIO(self.plan.k_caches[l], buffers.k[l], buffers.element_count);
                            outputs[1 + config.n_layers + l] = try self.program.tensorBindingIO(self.plan.v_caches[l], buffers.v[l], buffers.element_count);
                        },
                    } else {
                        outputs[1 + l] = try self.program.tensorBindingIO(self.plan.k_caches[l], session.k_caches[l].data.ptr, session.k_caches[l].data.len);
                        outputs[1 + config.n_layers + l] = try self.program.tensorBindingIO(self.plan.v_caches[l], session.v_caches[l].data.ptr, session.v_caches[l].data.len);
                    }
                }
                self.alloc.free(original_outputs);
                device_session.bindings.step_outputs = outputs;
                outputs_owned_by_session = true;
                try self.program.configureBindings(&device_session);
            }

            return .{
                .program = self,
                .session = session,
                .plan = runtime_plan,
                .device_session = device_session,
                .logits_buf = options.logits_buf[0..config.vocab_size],
                .default_output_is_host = options.output_resource == null,
            };
        }
    };
}

pub fn LlamaDeviceSession(comptime T: type, comptime config: LlamaConfig) type {
    const Session = LlamaInferenceSession(T, config);
    const DeviceProgram = LlamaDeviceProgram(T, config);
    const Device = device_inference.DeviceInference(T);
    const Plan = LlamaInferencePlan(T, config);

    return struct {
        const Self = @This();

        program: *DeviceProgram,
        session: *Session,
        plan: *Plan,
        device_session: Device.Session,
        logits_buf: []T,
        default_output_is_host: bool,

        pub fn deinit(self: *Self) void {
            self.device_session.deinit();
            self.* = undefined;
        }

        pub fn refreshCacheBindings(self: *Self) !void {
            const param_count = self.session.model.params().len;
            try self.device_session.uploadPersistentRange(&self.program.program, param_count, config.n_layers * 2);
        }

        pub fn step(self: *Self, token_id: usize) ![]const T {
            const logits = try self.executeAt(&.{token_id}, self.session.pos);
            self.session.pos += 1;
            return logits;
        }

        pub fn stepInto(self: *Self, output: []T, token_id: usize) ![]T {
            if (output.len < config.vocab_size) return error.OutputBufferTooSmall;
            const out = output[0..config.vocab_size];
            try self.executeAtInto(out, &.{token_id}, self.session.pos);
            self.session.pos += 1;
            return out;
        }

        pub fn stepBoundOutput(self: *Self, token_id: usize) !void {
            try self.executeAtBoundOutput(&.{token_id}, self.session.pos);
            self.session.pos += 1;
        }

        pub fn advance(self: *Self, token_id: usize) !void {
            try self.executeAtNoOutput(&.{token_id}, self.session.pos);
            self.session.pos += 1;
        }

        pub fn advanceTokens(self: *Self, tokens: []const usize) !void {
            try self.executeAtNoOutput(tokens, self.session.pos);
            self.session.pos += tokens.len;
        }

        pub fn prefill(self: *Self, tokens: []const usize) ![]const T {
            const logits = try self.executeAt(tokens, self.session.pos);
            self.session.pos += tokens.len;
            return logits;
        }

        pub fn prefillInto(self: *Self, output: []T, tokens: []const usize) ![]T {
            if (output.len < config.vocab_size) return error.OutputBufferTooSmall;
            const out = output[0..config.vocab_size];
            try self.executeAtInto(out, tokens, self.session.pos);
            self.session.pos += tokens.len;
            return out;
        }

        pub fn prefillBoundOutput(self: *Self, tokens: []const usize) !void {
            try self.executeAtBoundOutput(tokens, self.session.pos);
            self.session.pos += tokens.len;
        }

        fn executeAt(self: *Self, tokens: []const usize, position: usize) ![]const T {
            if (!self.default_output_is_host) return error.UnsupportedResourceBinding;
            const window = try self.patchInputs(tokens, position);
            try self.program.program.executeStep(&self.device_session, .{ .window = window });
            return self.logits_buf;
        }

        fn executeAtInto(self: *Self, output: []T, tokens: []const usize, position: usize) !void {
            const window = try self.patchInputs(tokens, position);
            if (self.device_session.bindings.step_outputs.len < 1) return error.InvalidProgramIO;
            const original_output = self.device_session.bindings.step_outputs[0];
            defer self.device_session.bindings.step_outputs[0] = original_output;
            self.device_session.bindings.step_outputs[0] = try self.program.program.tensorBindingIO(self.program.output_tensor, output.ptr, output.len);
            try self.program.program.executeStep(&self.device_session, .{ .window = window, .dynamic_io = true });
        }

        fn executeAtBoundOutput(self: *Self, tokens: []const usize, position: usize) !void {
            const window = try self.patchInputs(tokens, position);
            try self.program.program.executeStep(&self.device_session, .{ .window = window, .download_outputs = false });
        }

        fn executeAtNoOutput(self: *Self, tokens: []const usize, position: usize) !void {
            const window = try self.patchInputs(tokens, position);
            if (self.program.mode == .prefill and self.device_session.bindings.step_outputs.len > 1) {
                const original_outputs = self.device_session.bindings.step_outputs;
                self.device_session.bindings.step_outputs = original_outputs[1..];
                defer self.device_session.bindings.step_outputs = original_outputs;
                try self.program.program.executeStep(&self.device_session, .{ .window = window, .dynamic_io = true });
                return;
            }
            if (!self.default_output_is_host and self.device_session.bindings.step_outputs.len > 0) {
                const original_outputs = self.device_session.bindings.step_outputs;
                self.device_session.bindings.step_outputs = original_outputs[0..0];
                defer self.device_session.bindings.step_outputs = original_outputs;
                try self.program.program.executeStep(&self.device_session, .{ .window = window, .dynamic_io = true });
                return;
            }
            try self.program.program.executeStep(&self.device_session, .{ .window = window, .download_outputs = false });
        }

        fn patchInputs(self: *Self, tokens: []const usize, position: usize) !backend_mod.RuntimeWindow {
            return self.session.patchDeviceInputs(self.plan, tokens, position, self.program.mode);
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.device_session.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile.RuntimeProfile) void {
            self.device_session.addRuntimeProfileTo(dest);
        }

        pub fn semanticShape(self: *const Self) Plan.SemanticShape {
            return self.program.semanticShape();
        }
    };
}

/// Shared compiled-window runner for LLaMA decode and prefill.
///
/// Kept in this file so the LLaMA Session owns private plan construction,
/// host input patching, and position advancement for both host and device
/// execution paths.
fn LlamaDevice(comptime T: type, comptime config: LlamaConfig) type {
    const Session = LlamaInferenceSession(T, config);
    const Plan = LlamaInferencePlan(T, config);
    const Device = device_inference.DeviceInference(T);

    return struct {
        const Self = @This();

        session: *Session,
        plan: *Plan,
        mode: Plan.DeviceMode,
        device: Device,
        logits_buf: []T,

        const DecodeOptions = struct {
            session: *Session,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            logits_buf: []T,
        };

        const PrefillOptions = struct {
            session: *Session,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            chunk_tokens: usize,
            logits_buf: []T,
        };

        const InitOptions = struct {
            session: *Session,
            plan: *Plan,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            mode: Plan.DeviceMode,
            logits_buf: []T,
        };

        pub fn initDecode(options: DecodeOptions) !Self {
            return init(.{
                .session = options.session,
                .plan = &options.session.plan,
                .backend = options.backend,
                .alloc = options.alloc,
                .mode = .decode,
                .logits_buf = options.logits_buf,
            });
        }

        pub fn initPrefill(options: PrefillOptions) !Self {
            const plan = try options.session.ensurePrefillPlan(options.chunk_tokens);

            return init(.{
                .session = options.session,
                .plan = plan,
                .backend = options.backend,
                .alloc = options.alloc,
                .mode = .prefill,
                .logits_buf = options.logits_buf,
            });
        }

        fn init(options: InitOptions) !Self {
            if (options.logits_buf.len < config.vocab_size) return error.OutputBufferTooSmall;
            const binding = try options.plan.deviceCompileBinding(options.mode);

            var device = try Device.init(.{
                .graph = binding.graph,
                .be = options.backend,
                .alloc = options.alloc,
                .input_tensors = binding.inputs(),
                .output_tensor = binding.output_tensor,
                .output_host_buf = options.logits_buf.ptr,
                .output_len = config.vocab_size,
                .quant_weights = binding.quant_weights,
                .quant_map = binding.quant_map,
                .expected_runtime_patch_shape = binding.expected_runtime_patch_shape,
            });
            errdefer device.deinit();

            return .{
                .session = options.session,
                .plan = options.plan,
                .mode = options.mode,
                .device = device,
                .logits_buf = options.logits_buf[0..config.vocab_size],
            };
        }

        pub fn deinit(self: *Self) void {
            self.device.deinit();
        }

        pub fn step(self: *Self, token_id: usize) ![]const T {
            const logits = try self.executeAt(&.{token_id}, self.session.pos);
            self.session.pos += 1;
            return logits;
        }

        pub fn advance(self: *Self, token_id: usize) !void {
            try self.executeAtNoOutput(&.{token_id}, self.session.pos);
            self.session.pos += 1;
        }

        pub fn advanceTokens(self: *Self, tokens: []const usize) !void {
            try self.executeAtNoOutput(tokens, self.session.pos);
            self.session.pos += tokens.len;
        }

        /// Execute one fixed-size prefill window at the session's current
        /// position and advance the session position.
        pub fn prefill(self: *Self, tokens: []const usize) ![]const T {
            const logits = try self.executeAt(tokens, self.session.pos);
            self.session.pos += tokens.len;
            return logits;
        }

        fn executeAt(self: *Self, tokens: []const usize, position: usize) ![]const T {
            const window = try self.patchInputs(tokens, position);
            try self.device.executeStep(.{ .window = window });
            return self.logits_buf;
        }

        fn executeAtNoOutput(self: *Self, tokens: []const usize, position: usize) !void {
            const window = try self.patchInputs(tokens, position);
            try self.device.executeStep(.{ .window = window, .download_outputs = false });
        }

        fn patchInputs(self: *Self, tokens: []const usize, position: usize) !backend_mod.RuntimeWindow {
            return self.session.patchDeviceInputs(self.plan, tokens, position, self.mode);
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.device.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile.RuntimeProfile) void {
            self.device.addRuntimeProfileTo(dest);
        }

        pub fn inspectExecutable(self: *const Self) Device.ProgramInspection {
            return self.device.program.inspect();
        }

        pub fn semanticShape(self: *const Self) Plan.SemanticShape {
            return self.plan.semanticShape();
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

test "LlamaInferenceSession rejects unknown model file formats" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    try testing.expectError(error.UnknownModelFormat, session.load(std.Io.Threaded.global_single_threaded.io(), "model.bin"));
}

test "SmolLM semantic copy-and-patch shape matches benchmark contract" {
    const cfg = LlamaConfig{
        .vocab_size = 49152,
        .d_model = 576,
        .n_heads = 9,
        .n_kv_heads = 3,
        .d_ff = 1536,
        .n_layers = 30,
        .max_seq_len = 2048,
    };
    const shape = LlamaInferencePlan(f32, cfg).semanticShapeForTokenCount(128);

    try testing.expectEqual(@as(usize, 212), shape.semantic_stage_count);
    try testing.expectEqual(@as(usize, 128), shape.semantic_token_count);
    try testing.expectEqual(@as(usize, 7), shape.semantic_layer_stage_count);
    try testing.expectEqual(@as(usize, 2), shape.semantic_terminal_stage_count);
    try testing.expectEqual(@as(usize, 450), shape.semantic_runtime_patch_holes);
    try testing.expectEqual(@as(usize, 180), shape.semantic_runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(@as(usize, 270), shape.semantic_runtime_patch_attention_seq_kv_holes);
}

test "LlamaInferenceSession rejects invalid token ids before advancing position" {
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

    try testing.expectError(error.TokenIdOutOfRange, session.step(cfg.vocab_size));
    try testing.expectEqual(@as(usize, 0), session.position());
}

test "LlamaInferenceSession rejects invalid prefill tokens before advancing position" {
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

    try testing.expectError(error.TokenIdOutOfRange, session.prefill(&.{ 1, 2, cfg.vocab_size }));
    try testing.expectEqual(@as(usize, 0), session.position());
}

test "LlamaInferenceSession rejects invalid device tokens before patching host inputs" {
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

    const prefill = try session.ensurePrefillPlan(2);
    const sentinel: f32 = -123.0;
    @memset(prefill.token_input.data, sentinel);

    try testing.expectError(
        error.TokenIdOutOfRange,
        session.patchDeviceInputs(prefill, &.{ 1, cfg.vocab_size }, 0, .prefill),
    );
    for (prefill.token_input.data) |v| {
        try testing.expectEqual(sentinel, v);
    }
    try testing.expectEqual(@as(usize, 0), session.position());
}

const patch_test_config = LlamaConfig{
    .vocab_size = 16,
    .d_model = 8,
    .n_heads = 2,
    .n_kv_heads = 1,
    .d_ff = 16,
    .n_layers = 2,
    .max_seq_len = 8,
};

fn zeroQuantizedWeightPayloads(comptime T: type, qweights: []QuantizedWeight(T)) void {
    for (qweights) |*qw| {
        @memset(@constCast(qw.data), 0);
        if (qw.t_data) |data| @memset(@constCast(data), 0);
    }
}

test "llama device programs expose semantic runtime patch width" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var session = try Session.init(testing.allocator);
    defer session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    const decode_shape = session.plan.semanticShape();
    const expected: u32 = @intCast(decode_shape.semantic_runtime_patch_holes);
    const expected_cache: u32 = @intCast(decode_shape.semantic_runtime_patch_cache_write_pos_holes);
    const expected_attention: u32 = @intCast(decode_shape.semantic_runtime_patch_attention_seq_kv_holes);

    const logits = try testing.allocator.alloc(f32, patch_test_config.vocab_size);
    defer testing.allocator.free(logits);

    var decode = try session.compileDeviceDecode(cpu.backend(), testing.allocator, logits);
    defer decode.deinit();

    var decode_profile = profile.RuntimeProfile{};
    decode.addRuntimeProfileTo(&decode_profile);
    try testing.expectEqual(expected, decode_profile.runtime_patch_shape.runtime_patch_holes);
    try testing.expectEqual(expected_cache, decode_profile.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(expected_attention, decode_profile.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);
    try testing.expect(decode_profile.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try testing.expectEqual(decode_shape, decode.semanticShape());

    var prefill = try session.compileDevicePrefill(cpu.backend(), testing.allocator, 3, logits);
    defer prefill.deinit();

    var prefill_profile = profile.RuntimeProfile{};
    prefill.addRuntimeProfileTo(&prefill_profile);
    try testing.expectEqual(expected, prefill_profile.runtime_patch_shape.runtime_patch_holes);
    try testing.expectEqual(expected_cache, prefill_profile.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(expected_attention, prefill_profile.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);
    try testing.expect(prefill_profile.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try testing.expectEqual(prefill.plan.semanticShape(), prefill.semanticShape());
}

test "llama persistent decode program binds independent executable sessions" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var program = try model_session.compileDeviceDecodeProgram(cpu.backend(), testing.allocator);
    defer program.deinit();

    const executable = program.inspectExecutable();
    const semantic = model_session.semanticShape();
    try testing.expect(executable.command_shape.command_count > 0);
    try testing.expect(executable.command_shape.command_stencil_hash != 0);
    try testing.expectEqual(semantic.semantic_runtime_patch_holes, executable.runtime_patch_shape.runtime_patch_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_cache_write_pos_holes, executable.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_attention_seq_kv_holes, executable.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);

    var expected_session = try model_session.bindRuntime(testing.allocator);
    defer expected_session.deinit();
    const expected = try testing.allocator.dupe(f32, try expected_session.step(0));
    defer testing.allocator.free(expected);
    const expected_next = try testing.allocator.dupe(f32, try expected_session.step(1));
    defer testing.allocator.free(expected_next);

    var runtime_a = try model_session.bindRuntime(testing.allocator);
    defer runtime_a.deinit();
    var runtime_b = try model_session.bindRuntime(testing.allocator);
    defer runtime_b.deinit();

    var one_shot_runtime = try model_session.bindRuntime(testing.allocator);
    defer one_shot_runtime.deinit();
    var one_shot_logits = [_]f32{0} ** patch_test_config.vocab_size;
    var one_shot = try one_shot_runtime.compileDeviceDecode(cpu.backend(), testing.allocator, &one_shot_logits);
    defer one_shot.deinit();
    const one_shot_first = try one_shot.step(0);
    try testing.expectEqualSlices(f32, expected, one_shot_first);
    const one_shot_next = try one_shot.step(1);
    try testing.expectEqualSlices(f32, expected_next, one_shot_next);

    var modified_expected_session = try model_session.bindRuntime(testing.allocator);
    defer modified_expected_session.deinit();
    @memset(modified_expected_session.model.out_proj.data, 0);
    const expected_rebound = try testing.allocator.dupe(f32, try modified_expected_session.step(0));
    defer testing.allocator.free(expected_rebound);
    try testing.expect(!std.mem.eql(f32, expected, expected_rebound));

    var modified_runtime = try model_session.bindRuntime(testing.allocator);
    defer modified_runtime.deinit();
    @memset(modified_runtime.model.out_proj.data, 0);
    var rebound_logits = [_]f32{-555} ** patch_test_config.vocab_size;
    var rebound_exec = try program.bind(&modified_runtime, &rebound_logits);
    defer rebound_exec.deinit();
    const rebound = try rebound_exec.step(0);
    try testing.expectEqualSlices(f32, expected_rebound, rebound);

    var logits_a = [_]f32{-999} ** patch_test_config.vocab_size;
    var logits_b = [_]f32{-777} ** patch_test_config.vocab_size;
    var exec_a = try program.bind(&runtime_a, &logits_a);
    defer exec_a.deinit();
    var exec_b = try program.bind(&runtime_b, &logits_b);
    defer exec_b.deinit();

    const a0 = try exec_a.step(0);
    try testing.expectEqualSlices(f32, expected, a0);
    try testing.expectEqual(@as(usize, 1), runtime_a.position());
    try testing.expectEqual(@as(usize, 0), runtime_b.position());
    const a1 = try exec_a.step(1);
    try testing.expectEqualSlices(f32, expected_next, a1);
    try testing.expectEqual(@as(usize, 2), runtime_a.position());
    try testing.expectEqual(@as(usize, 0), runtime_b.position());

    const b0 = try exec_b.step(0);
    try testing.expectEqualSlices(f32, expected, b0);
    try testing.expectEqual(@as(usize, 2), runtime_a.position());
    try testing.expectEqual(@as(usize, 1), runtime_b.position());
}

test "llama persistent decode executable hot path does not allocate" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    var model_session = try Session.init(alloc);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var program = try model_session.compileDeviceDecodeProgram(cpu.backend(), alloc);
    defer program.deinit();

    var runtime = try model_session.bindRuntime(alloc);
    defer runtime.deinit();

    var logits = [_]f32{-999} ** patch_test_config.vocab_size;
    var exec = try program.bind(&runtime, &logits);
    defer exec.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    const first = try exec.stepInto(&logits, 0);
    try testing.expectEqual(@as(usize, patch_test_config.vocab_size), first.len);
    try exec.advance(1);
    const second = try exec.stepInto(&logits, 2);
    try testing.expectEqual(@as(usize, patch_test_config.vocab_size), second.len);
    try testing.expectEqual(@as(usize, 3), runtime.position());

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);
}

test "llama resource-bound decode output requires explicit host override" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var backend = cpu.backend();
    backend.device_type = .webgpu;
    backend.capabilities.external_resources = true;

    var program = try model_session.compileDeviceDecodeProgram(backend, testing.allocator);
    defer program.deinit();

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();

    var scratch_logits = [_]f32{-999} ** patch_test_config.vocab_size;
    const resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 77,
        .byte_offset = 16,
        .byte_len = 16 + @as(u32, patch_test_config.vocab_size * @sizeOf(f32)),
        .access = .write_only,
    };
    var exec = try program.bindWithOptions(&runtime, .{
        .logits_buf = &scratch_logits,
        .output_resource = resource,
    });
    defer exec.deinit();

    try testing.expectEqual(@as(usize, 1), exec.device_session.bindings.step_outputs.len);
    try testing.expect(!exec.device_session.bindings.step_outputs[0].isHost());
    const bound_resource = exec.device_session.bindings.step_outputs[0].resource.?;
    try testing.expectEqual(backend_mod.Device.webgpu, bound_resource.placement);
    try testing.expectEqual(@as(usize, 77), bound_resource.handle);
    try testing.expectEqual(@as(u32, 16), bound_resource.byte_offset);
    try testing.expect(!bound_resource.canRead());
    try testing.expect(bound_resource.canWrite());

    try testing.expectError(error.UnsupportedResourceBinding, exec.step(0));
    try testing.expectEqual(@as(usize, 0), runtime.position());

    var host_logits = [_]f32{-555} ** patch_test_config.vocab_size;
    const out = try exec.stepInto(&host_logits, 0);
    try testing.expectEqual(@as(usize, patch_test_config.vocab_size), out.len);
    try testing.expectEqual(@as(usize, 1), runtime.position());
    try testing.expect(exec.device_session.bindings.step_outputs[0].resource != null);
    try testing.expectEqual(@as(usize, 77), exec.device_session.bindings.step_outputs[0].resource.?.handle);

    try exec.advance(1);
    try testing.expectEqual(@as(usize, 2), runtime.position());
    try testing.expect(exec.device_session.bindings.step_outputs[0].resource != null);
    try testing.expectEqual(@as(usize, 77), exec.device_session.bindings.step_outputs[0].resource.?.handle);
}

test "llama executable sessions can bind KV caches as external resources" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var backend = cpu.backend();
    backend.device_type = .webgpu;
    backend.capabilities.external_resources = true;

    var decode_program = try model_session.compileDeviceDecodeProgram(backend, testing.allocator);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(backend, testing.allocator, 2);
    defer prefill_program.deinit();

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();

    const DeviceProgram = Session.DeviceDecodeProgram;
    var cache_resources: DeviceProgram.CacheResourceBindings = undefined;
    const cache_element_count = try DeviceProgram.cacheElementCountForContext(4);
    const cache_byte_len: u32 = @intCast(cache_element_count * @sizeOf(f32));
    cache_resources.element_count = cache_element_count;
    for (0..patch_test_config.n_layers) |l| {
        cache_resources.k[l] = .{
            .placement = .webgpu,
            .handle = 1000 + l,
            .byte_len = cache_byte_len,
            .access = .read_write,
        };
        cache_resources.v[l] = .{
            .placement = .webgpu,
            .handle = 2000 + l,
            .byte_offset = 4,
            .byte_len = 4 + cache_byte_len,
            .access = .read_write,
        };
    }

    var wrong_resources = cache_resources;
    wrong_resources.k[0].placement = .metal;
    var scratch_logits = [_]f32{-999} ** patch_test_config.vocab_size;
    try testing.expectError(error.UnsupportedResourceBinding, decode_program.bindWithOptions(&runtime, .{
        .logits_buf = &scratch_logits,
        .cache_bindings = .{ .resource = wrong_resources },
    }));

    var write_only_decode_resources = cache_resources;
    write_only_decode_resources.k[0].access = .write_only;
    try testing.expectError(error.UnsupportedResourceBinding, decode_program.bindWithOptions(&runtime, .{
        .logits_buf = &scratch_logits,
        .cache_bindings = .{ .resource = write_only_decode_resources },
    }));

    var decode_exec = try decode_program.bindWithOptions(&runtime, .{
        .logits_buf = &scratch_logits,
        .cache_bindings = .{ .resource = cache_resources },
    });
    defer decode_exec.deinit();

    const param_count = runtime.model.params().len;
    try testing.expectEqual(param_count + patch_test_config.n_layers * 2, decode_exec.device_session.bindings.persistent_inputs.len);
    for (0..patch_test_config.n_layers) |l| {
        const k_io = decode_exec.device_session.bindings.persistent_inputs[param_count + l];
        const v_io = decode_exec.device_session.bindings.persistent_inputs[param_count + patch_test_config.n_layers + l];
        try testing.expect(!k_io.isHost());
        try testing.expect(!v_io.isHost());
        try testing.expect(k_io.hostSlice() == null);
        try testing.expect(v_io.hostSlice() == null);
        try testing.expectEqual(@as(usize, 1000 + l), k_io.resource.?.handle);
        try testing.expectEqual(@as(usize, 2000 + l), v_io.resource.?.handle);
        try testing.expect(k_io.resource.?.canRead());
        try testing.expect(k_io.resource.?.canWrite());
        try testing.expect(v_io.resource.?.canRead());
        try testing.expect(v_io.resource.?.canWrite());
        try testing.expectEqual(cache_byte_len, k_io.size);
        try testing.expectEqual(cache_byte_len, v_io.size);
    }

    var prefill_logits = [_]f32{-777} ** patch_test_config.vocab_size;
    var read_only_prefill_resources = cache_resources;
    read_only_prefill_resources.k[0].access = .read_only;
    try testing.expectError(error.UnsupportedResourceBinding, prefill_program.bindWithOptions(&runtime, .{
        .logits_buf = &prefill_logits,
        .cache_bindings = .{ .resource = read_only_prefill_resources },
    }));

    var prefill_exec = try prefill_program.bindWithOptions(&runtime, .{
        .logits_buf = &prefill_logits,
        .cache_bindings = .{ .resource = cache_resources },
    });
    defer prefill_exec.deinit();

    try testing.expectEqual(@as(usize, 1 + patch_test_config.n_layers * 2), prefill_exec.device_session.bindings.step_outputs.len);
    for (0..patch_test_config.n_layers) |l| {
        const k_output = prefill_exec.device_session.bindings.step_outputs[1 + l];
        const v_output = prefill_exec.device_session.bindings.step_outputs[1 + patch_test_config.n_layers + l];
        try testing.expect(!k_output.isHost());
        try testing.expect(!v_output.isHost());
        try testing.expectEqual(@as(usize, 1000 + l), k_output.resource.?.handle);
        try testing.expectEqual(@as(usize, 2000 + l), v_output.resource.?.handle);
        try testing.expectEqual(cache_byte_len, k_output.size);
        try testing.expectEqual(cache_byte_len, v_output.size);
    }
}

test "llama persistent prefill executable hot path does not allocate" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    var model_session = try Session.init(alloc);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    const tokens = [_]usize{ 0, 1, 2 };
    var program = try model_session.compileDevicePrefillProgram(cpu.backend(), alloc, tokens.len);
    defer program.deinit();

    var runtime = try model_session.bindRuntime(alloc);
    defer runtime.deinit();

    var logits = [_]f32{-999} ** patch_test_config.vocab_size;
    var exec = try program.bind(&runtime, &logits);
    defer exec.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    const prefilled = try exec.prefillInto(&logits, &tokens);
    try testing.expectEqual(@as(usize, patch_test_config.vocab_size), prefilled.len);
    try testing.expectEqual(@as(usize, tokens.len), runtime.position());

    runtime.reset();
    try exec.advanceTokens(&tokens);
    try testing.expectEqual(@as(usize, tokens.len), runtime.position());

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);
}

test "llama persistent decode program binds runtime quantized weights" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();
    try model_session.quantize();
    try testing.expect(model_session.plan.quant_weights.len > 0);

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var program = try model_session.compileDeviceDecodeProgram(cpu.backend(), testing.allocator);
    defer program.deinit();

    var expected_session = try model_session.bindRuntime(testing.allocator);
    defer expected_session.deinit();
    const expected = try testing.allocator.dupe(f32, try expected_session.step(0));
    defer testing.allocator.free(expected);

    var modified_expected_session = try model_session.bindRuntime(testing.allocator);
    defer modified_expected_session.deinit();
    zeroQuantizedWeightPayloads(f32, modified_expected_session.plan.quant_weights);
    const expected_rebound = try testing.allocator.dupe(f32, try modified_expected_session.step(0));
    defer testing.allocator.free(expected_rebound);
    try testing.expect(!std.mem.eql(f32, expected, expected_rebound));

    var modified_runtime = try model_session.bindRuntime(testing.allocator);
    defer modified_runtime.deinit();
    zeroQuantizedWeightPayloads(f32, modified_runtime.plan.quant_weights);
    var rebound_logits = [_]f32{-555} ** patch_test_config.vocab_size;
    var rebound_exec = try program.bind(&modified_runtime, &rebound_logits);
    defer rebound_exec.deinit();
    const rebound = try rebound_exec.step(0);
    try testing.expectEqualSlices(f32, expected_rebound, rebound);
}

test "llama persistent prefill program binds independent executable sessions" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    const tokens = [_]usize{ 0, 1, 2 };
    var program = try model_session.compileDevicePrefillProgram(cpu.backend(), testing.allocator, tokens.len);
    defer program.deinit();

    const executable = program.inspectExecutable();
    const semantic = program.semanticShape();
    try testing.expect(executable.command_shape.command_count > 0);
    try testing.expect(executable.command_shape.command_stencil_hash != 0);
    try testing.expectEqual(@as(usize, tokens.len), semantic.semantic_token_count);
    try testing.expectEqual(semantic.semantic_runtime_patch_holes, executable.runtime_patch_shape.runtime_patch_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_cache_write_pos_holes, executable.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_attention_seq_kv_holes, executable.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);

    var expected_session = try model_session.bindRuntime(testing.allocator);
    defer expected_session.deinit();
    const expected = try testing.allocator.dupe(f32, try expected_session.prefill(&tokens));
    defer testing.allocator.free(expected);
    try testing.expectEqual(@as(usize, tokens.len), expected_session.position());

    var modified_expected_session = try model_session.bindRuntime(testing.allocator);
    defer modified_expected_session.deinit();
    @memset(modified_expected_session.model.out_proj.data, 0);
    const expected_rebound = try testing.allocator.dupe(f32, try modified_expected_session.prefill(&tokens));
    defer testing.allocator.free(expected_rebound);
    try testing.expect(!std.mem.eql(f32, expected, expected_rebound));

    var modified_runtime = try model_session.bindRuntime(testing.allocator);
    defer modified_runtime.deinit();
    @memset(modified_runtime.model.out_proj.data, 0);
    var rebound_logits = [_]f32{-555} ** patch_test_config.vocab_size;
    var rebound_exec = try program.bind(&modified_runtime, &rebound_logits);
    defer rebound_exec.deinit();
    const rebound = try rebound_exec.prefill(&tokens);
    try testing.expectEqualSlices(f32, expected_rebound, rebound);
    try testing.expectEqual(@as(usize, tokens.len), modified_runtime.position());

    var runtime_a = try model_session.bindRuntime(testing.allocator);
    defer runtime_a.deinit();
    var runtime_b = try model_session.bindRuntime(testing.allocator);
    defer runtime_b.deinit();

    var logits_a = [_]f32{-999} ** patch_test_config.vocab_size;
    var logits_b = [_]f32{-777} ** patch_test_config.vocab_size;
    var exec_a = try program.bind(&runtime_a, &logits_a);
    defer exec_a.deinit();
    var exec_b = try program.bind(&runtime_b, &logits_b);
    defer exec_b.deinit();

    const a0 = try exec_a.prefill(&tokens);
    try testing.expectEqualSlices(f32, expected, a0);
    try testing.expectEqual(@as(usize, tokens.len), runtime_a.position());
    try testing.expectEqual(@as(usize, 0), runtime_b.position());

    var caller_logits = [_]f32{-321} ** (patch_test_config.vocab_size + 1);
    const b0 = try exec_b.prefillInto(&caller_logits, &tokens);
    try testing.expect(b0.ptr == caller_logits[0..patch_test_config.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -321), caller_logits[patch_test_config.vocab_size]);
    try testing.expectEqualSlices(f32, expected, b0);
    try testing.expectEqual(@as(usize, tokens.len), runtime_a.position());
    try testing.expectEqual(@as(usize, tokens.len), runtime_b.position());
}

test "llama prefill cache side effects refresh bound decode session" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    const prefix = [_]usize{ 0, 1 };
    const full = [_]usize{ 0, 1, 2 };

    var expected_session = try model_session.bindRuntime(testing.allocator);
    defer expected_session.deinit();
    const expected = try testing.allocator.dupe(f32, try expected_session.prefill(&full));
    defer testing.allocator.free(expected);

    var mutated_weight_session = try model_session.bindRuntime(testing.allocator);
    defer mutated_weight_session.deinit();
    _ = try mutated_weight_session.prefill(&prefix);
    @memset(mutated_weight_session.model.out_proj.data, 0);
    const mutated_weight_expected = try testing.allocator.dupe(f32, try mutated_weight_session.step(2));
    defer testing.allocator.free(mutated_weight_expected);
    try testing.expect(!std.mem.eql(f32, expected, mutated_weight_expected));

    var decode_program = try model_session.compileDeviceDecodeProgram(cpu.backend(), testing.allocator);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(cpu.backend(), testing.allocator, prefix.len);
    defer prefill_program.deinit();

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();

    var decode_logits = [_]f32{-999} ** patch_test_config.vocab_size;
    var decode_exec = try decode_program.bind(&runtime, &decode_logits);
    defer decode_exec.deinit();
    const decode_runtime_handle = decode_exec.device_session.runtime_handle;

    var prefill_logits = [_]f32{-777} ** patch_test_config.vocab_size;
    var prefill_exec = try prefill_program.bind(&runtime, &prefill_logits);
    defer prefill_exec.deinit();

    _ = try prefill_exec.prefill(&prefix);
    try testing.expectEqual(@as(usize, prefix.len), runtime.position());
    @memset(runtime.model.out_proj.data, 0);

    try decode_exec.refreshCacheBindings();
    try testing.expect(decode_runtime_handle == decode_exec.device_session.runtime_handle);

    const got = try decode_exec.step(2);
    try testing.expectEqualSlices(f32, expected, got);
    try testing.expectEqual(@as(usize, full.len), runtime.position());
}

test "llama device rejects invalid runtime inputs before advancing position" {
    const Session = LlamaInferenceSession(f32, patch_test_config);
    var session = try Session.init(testing.allocator);
    defer session.deinit();

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    const logits = try testing.allocator.alloc(f32, patch_test_config.vocab_size);
    defer testing.allocator.free(logits);

    var decode = try session.compileDeviceDecode(cpu.backend(), testing.allocator, logits);
    defer decode.deinit();

    try testing.expectError(error.TokenIdOutOfRange, decode.step(patch_test_config.vocab_size));
    try testing.expectEqual(@as(usize, 0), session.position());

    session.pos = std.math.maxInt(usize);
    try testing.expectError(error.SequenceTooLong, decode.step(0));
    try testing.expectEqual(std.math.maxInt(usize), session.position());
}

test "LlamaInferencePlan rejects overflowing host patch windows" {
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

    try testing.expectError(
        error.SequenceTooLong,
        session.plan.execute(&session.model, &.{0}, std.math.maxInt(usize)),
    );
    try testing.expectEqual(@as(usize, 0), session.position());
}

test "LlamaInferencePlan rejects invalid token window at construction" {
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

    try testing.expectError(
        error.InvalidPrefillLength,
        LlamaInferencePlan(f32, cfg).initWithBackend(
            &session.model,
            session.k_caches,
            session.v_caches,
            testing.allocator,
            null,
            0,
        ),
    );
    try testing.expectError(
        error.InvalidPrefillLength,
        LlamaInferencePlan(f32, cfg).initWithBackend(
            &session.model,
            session.k_caches,
            session.v_caches,
            testing.allocator,
            null,
            cfg.max_seq_len + 1,
        ),
    );
}

test "LlamaDeviceProgram KV cache sizing is packed by compiled context" {
    const cfg = LlamaConfig{
        .vocab_size = 16,
        .d_model = 12,
        .n_heads = 3,
        .n_kv_heads = 3,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 8,
    };
    const Program = LlamaInferenceSession(f32, cfg).DeviceDecodeProgram;

    const context_len: usize = 3;
    const d_head = cfg.d_model / cfg.n_heads;
    const packed_context = d_head * context_len * cfg.n_kv_heads;

    try testing.expectEqual(packed_context, try Program.cacheElementCountForContext(context_len));
    try testing.expectEqual(d_head * cfg.max_seq_len * cfg.n_kv_heads, try Program.cacheElementCountForContext(cfg.max_seq_len));
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
        const tok_data = ref_model.token_embed.data;
        @memcpy(tok_input.data[0..cfg.d_model], tok_data[tok * cfg.d_model ..][0..cfg.d_model]);
        for (ref_mask.data[0..cfg.max_seq_len], 0..) |*v, i| {
            v.* = if (i <= pos) 0 else -std.math.inf(f32);
        }

        const ref_trace = ref_model.forwardCachedMasked(tok_input, ref_k, ref_v, pos, ref_mask);
        try g.infer(ref_trace.logits);

        const session_logits = try session.step(tok);
        for (session_logits, ref_trace.logits.data[0..cfg.vocab_size]) |got, want| {
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

test "LlamaInferenceSession clearWeightQuantization drops generated plan weights" {
    const cfg = LlamaConfig{
        .vocab_size = 16,
        .d_model = 32,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 32,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const prefill = try session.ensurePrefillPlan(2);
    try session.quantize();
    try testing.expect(session.plan.quant_weights.len > 0);
    try testing.expect(prefill.quant_weights.len > 0);
    try testing.expect(session.quant_block_size != null);

    session.clearWeightQuantization();
    try testing.expectEqual(@as(usize, 0), session.plan.quant_weights.len);
    try testing.expectEqual(@as(usize, 0), prefill.quant_weights.len);
    try testing.expect(session.plan.owns_quant_weights);
    try testing.expect(prefill.owns_quant_weights);
    try testing.expectEqual(@as(?usize, null), session.quant_block_size);
}

test "LlamaInferencePlan maps external tied embedding qweight to logits" {
    const cfg = LlamaConfig{
        .vocab_size = 4,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
        .tied_lm_head = true,
    };

    var session = try LlamaInferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    var weights = [_]QuantizedWeight(f32){
        try QuantizedWeight(f32).fromTensor(testing.allocator, session.model.token_embed, quant.default_block_size),
        try QuantizedWeight(f32).fromTransposedTensor(testing.allocator, session.model.token_embed, quant.default_block_size),
    };
    defer weights[1].deinit(testing.allocator);
    defer weights[0].deinit(testing.allocator);

    var param_map: std.AutoHashMapUnmanaged(*Tensor(f32), quant.QuantizedWeightBinding) = .empty;
    defer param_map.deinit(testing.allocator);
    try param_map.put(testing.allocator, session.model.token_embed, .{ .index = 0, .layout = .native_param });

    try session.plan.useExternalQuantWeights(weights[0..], &param_map);
    try testing.expect(session.plan.quant_map.get(session.plan.trace.logits) == null);

    try param_map.put(testing.allocator, session.model.token_embed, .{ .index = 1, .layout = .transposed_param });
    try session.plan.useExternalQuantWeights(weights[0..], &param_map);
    defer session.plan.clearQuantization();
    try testing.expectEqual(@as(usize, 1), session.plan.quant_map.get(session.plan.trace.logits).?);

    const prefill = try session.ensurePrefillPlan(2);
    try prefill.useExternalQuantWeights(weights[0..], &param_map);
    defer prefill.clearQuantization();
    try testing.expectEqual(@as(usize, 1), prefill.quant_map.get(prefill.trace.logits).?);
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

test "LlamaInferenceSession fixed-chunk prefill matches sequential step()" {
    // max_seq_len above default_prefill_chunk exercises one batched chunk plus
    // a tail without exposing the chunk size as a mutable session field.
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 130,
    };

    var tokens: [default_prefill_chunk + 1]usize = undefined;
    for (&tokens, 0..) |*tok, i| tok.* = (i * 3 + 2) % cfg.vocab_size;

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
