//! LLaMA decoder-only transformer model.
//!
//! Stacks token embeddings (no positional encoding — RoPE is per-layer),
//! N LLaMA transformer blocks, a final RMSNorm, and an output projection.
//!
//! ```
//! const model = try LLaMA(f32, .{
//!     .vocab_size = 32000,
//!     .d_model = 4096,
//!     .n_heads = 32,
//!     .n_kv_heads = 8,
//!     .d_ff = 11008,
//!     .n_layers = 32,
//!     .max_seq_len = 4096,
//! }).init(alloc);
//!
//! const indices = try Tensor(f32).initIndexVectorCopy(alloc, &.{1, 15043, 29871});
//! const logits = model.forward(indices);  // -> [vocab_size, seq_len]
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;

const LlamaBlock = @import("llama_transformer.zig").LlamaBlock;
const LlamaBlockConfig = @import("llama_transformer.zig").LlamaBlockConfig;
const nn = @import("../nn.zig");

pub const LlamaConfig = struct {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_seq_len: usize,
    rope_base: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
    tied_lm_head: bool = false,
};

pub fn LLaMA(comptime T: type, comptime config: LlamaConfig) type {
    const block_cfg = LlamaBlockConfig{
        .d_model = config.d_model,
        .n_heads = config.n_heads,
        .n_kv_heads = config.n_kv_heads,
        .d_ff = config.d_ff,
        .max_seq_len = config.max_seq_len,
        .rope_base = config.rope_base,
        .rms_norm_eps = config.rms_norm_eps,
    };
    const Block = LlamaBlock(T, block_cfg);
    const d_model = config.d_model;
    const EmbedShape = Shaped(T, .{ d_model, config.vocab_size });
    const NormShape = Shaped(T, .{d_model});
    const OutProjShape = Shaped(T, .{ config.vocab_size, d_model });

    const params_per_block = Block.n_block_params;
    const out_proj_params: usize = if (config.tied_lm_head) 0 else 1;
    const total_params = 1 + params_per_block * config.n_layers + 1 + out_proj_params;

    return struct {
        const Self = @This();

        /// Everything a frozen inference plan needs to operate without
        /// re-scanning the graph: `logits` plus per-layer `CachedLayerTrace`s
        /// that pin down RoPE leaves, KV-cache writes, and attention nodes.
        pub const CachedForwardTrace = struct {
            logits: *Tensor(T),
            layers: [config.n_layers]Block.CachedLayerTrace,
        };

        token_embed: EmbedShape,
        blocks: [config.n_layers]Block,
        rms_norm_f: NormShape,
        out_proj: if (!config.tied_lm_head) OutProjShape else void,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            self.token_embed = try EmbedShape.init(alloc);
            const scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(d_model)));
            for (self.token_embed.inner.data, 0..) |*d, i| {
                const fi: T = @floatFromInt(i);
                d.* = scale * @sin(fi * 0.1 + 0.3) * @cos(fi * 0.07 + 0.5);
            }
            self.token_embed.inner.setParam();

            for (0..config.n_layers) |i| {
                self.blocks[i] = try Block.init(alloc);
            }

            self.rms_norm_f = try NormShape.init(alloc);
            _ = self.rms_norm_f.inner.setAllScalar(1);
            self.rms_norm_f.inner.setParam();

            if (!config.tied_lm_head) {
                self.out_proj = try OutProjShape.init(alloc);
                nn.kaimingUniform(T, self.out_proj.inner, 99);
                self.out_proj.inner.setParam();
            }

            return self;
        }

        /// Forward pass: token indices -> vocabulary logits.
        /// Returns logits of shape [vocab_size, seq_len].
        pub fn forward(self: *const Self, token_indices: *Tensor(T)) *Tensor(T) {
            const seq_len = token_indices.ne[0];

            // Token embedding (no positional encoding — RoPE is in each block)
            var x = self.token_embed.inner.gatherRows(token_indices);

            // Transformer blocks
            for (0..config.n_layers) |i| {
                x = self.blocks[i].forward(x);
            }

            // Final RMSNorm
            var norm_reduce = [_]usize{ 1, seq_len };
            x = x.rmsNorm(&norm_reduce, @floatCast(config.rms_norm_eps));
            x = x.mul(self.rms_norm_f.inner.repeatLike(x));

            // Output projection
            if (config.tied_lm_head) {
                return x.matMul(false, self.token_embed.inner, true);
            } else {
                return x.matMul(false, self.out_proj.inner, false);
            }
        }

        /// Frozen cached forward for persistent inference plans.
        ///
        /// `x_in`: [d_model, 1] — token embedding (caller does the lookup).
        /// `k_caches`/`v_caches`: one consolidated [d_head, max_seq_len * n_kv_heads]
        /// tensor per layer; head `kv_h`'s slab is a contiguous column range.
        /// `attn_mask`: [max_seq_len, 1] — 0 for valid, -inf for masked.
        pub fn forwardCachedMasked(
            self: *const Self,
            x_in: *Tensor(T),
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            pos: usize,
            attn_mask: *Tensor(T),
        ) CachedForwardTrace {
            var x = x_in;
            var layers: [config.n_layers]Block.CachedLayerTrace = undefined;
            for (0..config.n_layers) |i| {
                layers[i] = self.blocks[i].forwardCachedMasked(x, k_caches[i], v_caches[i], pos, attn_mask);
                x = layers[i].output;
            }

            var norm_reduce = [_]usize{ 1, x.ne[1] };
            x = x.rmsNorm(&norm_reduce, @floatCast(config.rms_norm_eps));
            x = x.mul(self.rms_norm_f.inner.repeatLike(x));

            const logits = if (config.tied_lm_head)
                x.matMul(false, self.token_embed.inner, true)
            else
                x.matMul(false, self.out_proj.inner, false);

            return .{ .logits = logits, .layers = layers };
        }

        /// Return all learnable parameters.
        pub fn params(self: *const Self) [total_params]*Tensor(T) {
            var result: [total_params]*Tensor(T) = undefined;
            var idx: usize = 0;

            result[idx] = self.token_embed.inner;
            idx += 1;

            for (0..config.n_layers) |i| {
                for (self.blocks[i].params()) |p| {
                    result[idx] = p;
                    idx += 1;
                }
            }

            result[idx] = self.rms_norm_f.inner;
            idx += 1;

            if (!config.tied_lm_head) {
                result[idx] = self.out_proj.inner;
                idx += 1;
            }

            std.debug.assert(idx == total_params);
            return result;
        }

        /// Per-head KV cache row dimension for external allocation.
        pub const kv_d_head = d_model / config.n_heads;
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "LLaMA - forward produces valid logits" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try LLaMA(f32, .{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 32,
    }).init(a);

    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 0, 3, 1 });
    const logits = model.forward(indices);

    try g.infer(logits);

    try testing.expectEqual(@as(usize, 16), logits.ne[0]);
    try testing.expectEqual(@as(usize, 3), logits.ne[1]);

    for (logits.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "LLaMA - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try LLaMA(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 16,
    }).init(a);

    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 2, 5 });
    const loss = model.forward(indices).sumAll();

    try g.run(loss);

    for (model.params()) |param| {
        if (param.grad) |grad| {
            for (grad.data) |v| {
                try testing.expect(!std.math.isNan(v));
                try testing.expect(!std.math.isInf(v));
            }
        }
    }
}

test "LLaMA - GQA model forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try LLaMA(f32, .{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 32,
    }).init(a);

    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 0, 7, 3 });
    const logits = model.forward(indices);

    try g.infer(logits);

    try testing.expectEqual(@as(usize, 16), logits.ne[0]);
    try testing.expectEqual(@as(usize, 3), logits.ne[1]);

    for (logits.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "LLaMA - frozen cached masked forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };
    const Model = LLaMA(f32, cfg);
    const model = try Model.init(a);

    const d_h = Model.kv_d_head;
    var k_caches: [cfg.n_layers]*Tensor(f32) = undefined;
    var v_caches: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        k_caches[l] = try Tensor(f32).init(a, &.{ d_h, cfg.max_seq_len * cfg.n_kv_heads });
        v_caches[l] = try Tensor(f32).init(a, &.{ d_h, cfg.max_seq_len * cfg.n_kv_heads });
        @memset(k_caches[l].data, 0);
        @memset(v_caches[l].data, 0);
    }

    const attn_mask = try Tensor(f32).init(a, &.{ cfg.max_seq_len, 1 });
    @memset(attn_mask.data, -std.math.inf(f32));
    attn_mask.data[0] = 0;

    // Token embedding lookup (manual for frozen plan)
    const x = try Tensor(f32).init(a, &.{ cfg.d_model, 1 });
    for (0..cfg.d_model) |i| {
        x.data[i] = model.token_embed.inner.data[0 * cfg.d_model + i]; // token 0
    }

    const trace = model.forwardCachedMasked(x, k_caches, v_caches, 0, attn_mask);
    try g.infer(trace.logits);

    try testing.expectEqual(@as(usize, cfg.vocab_size), trace.logits.ne[0]);
    try testing.expectEqual(@as(usize, 1), trace.logits.ne[1]);
    for (trace.logits.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "LLaMA - param count is correct" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const model = try LLaMA(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    }).init(a);

    const p = model.params();
    // 1 (embed) + 2 * 9 (blocks) + 1 (rms_norm_f) + 1 (out_proj) = 21
    try testing.expectEqual(@as(usize, 21), p.len);
}
