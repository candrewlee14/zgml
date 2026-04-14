//! GPT-style decoder-only transformer model.
//!
//! Stacks token/positional embeddings, N causal transformer blocks,
//! a final layer norm, and an output projection to vocabulary logits.
//!
//! ```
//! const model = try GPT(f32, .{
//!     .vocab_size = 256,
//!     .d_model = 64,
//!     .n_heads = 4,
//!     .d_ff = 256,
//!     .n_layers = 4,
//!     .max_seq_len = 128,
//! }).init(alloc);
//!
//! const indices = try Tensor(f32).initIndexVectorCopy(alloc, &.{5, 12, 0, 42});
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

const TransformerBlock = @import("transformer.zig").TransformerBlock;
const Embedding = @import("embedding.zig").Embedding;
const nn = @import("../nn.zig");

pub const GPTConfig = struct {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_seq_len: usize,
    // Pretrained model support (defaults preserve existing behavior)
    learnable_pos_embed: bool = false,
    learnable_ln: bool = false,
    attn_bias: bool = false,
    tied_lm_head: bool = false,
};

/// GPT-style decoder-only transformer.
///
/// Architecture:
///   1. Token embedding + sinusoidal positional encoding
///   2. N x causal TransformerBlock (pre-norm, multi-head attention + FFN)
///   3. Final layer normalization
///   4. Linear projection to vocabulary logits
pub fn GPT(comptime T: type, comptime config: GPTConfig) type {
    const Block = TransformerBlock(T, config.d_model, config.n_heads, config.d_ff, true, config.learnable_ln, config.attn_bias);
    const Embed = Embedding(T, config.vocab_size, config.d_model, config.max_seq_len, config.learnable_pos_embed);

    // Output projection weight shape: [vocab_size, d_model]
    const OutProjShape = Shaped(T, .{ config.vocab_size, config.d_model });
    const LnShape = Shaped(T, .{config.d_model});

    // Number of params per block
    const params_per_block = Block.n_block_params;
    const embed_params = Embed.n_params;
    const ln_f_params: usize = if (config.learnable_ln) 2 else 0;
    const out_proj_params: usize = if (config.tied_lm_head) 0 else 1;
    // Total: embedding + blocks + final_ln + output_proj
    const total_params = embed_params + params_per_block * config.n_layers + ln_f_params + out_proj_params;

    return struct {
        const Self = @This();

        embed: Embed,
        blocks: [config.n_layers]Block,
        out_proj: if (!config.tied_lm_head) OutProjShape else void,
        ln_f_gamma: if (config.learnable_ln) LnShape else void,
        ln_f_beta: if (config.learnable_ln) LnShape else void,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            self.embed = try Embed.init(alloc);

            for (0..config.n_layers) |i| {
                self.blocks[i] = try Block.init(alloc);
            }

            if (!config.tied_lm_head) {
                self.out_proj = try OutProjShape.init(alloc);
                nn.kaimingUniform(T, self.out_proj.inner, 99);
                self.out_proj.inner.setParam();
            }

            if (config.learnable_ln) {
                self.ln_f_gamma = try LnShape.init(alloc);
                self.ln_f_beta = try LnShape.init(alloc);
                _ = self.ln_f_gamma.inner.setAllScalar(1);
                _ = self.ln_f_beta.inner.setAllScalar(0);
                self.ln_f_gamma.inner.setParam();
                self.ln_f_beta.inner.setParam();
            }

            return self;
        }

        /// Forward pass: token indices -> vocabulary logits.
        ///
        /// `token_indices` is a 1-D index tensor of shape [seq_len].
        /// Returns logits of shape [vocab_size, seq_len].
        pub fn forward(self: *const Self, token_indices: *Tensor(T)) *Tensor(T) {
            const seq_len = token_indices.ne[0];

            // 1. Embedding: token lookup + positional encoding -> [d_model, seq_len]
            var x = self.embed.forward(token_indices);

            // 2. Transformer blocks
            for (0..config.n_layers) |i| {
                x = self.blocks[i].forward(x);
            }

            // 3. Final layer norm (optionally affine)
            var ln_reduce = [_]usize{ 1, seq_len };
            x = x.layerNorm(&ln_reduce, 1e-5);
            if (config.learnable_ln) {
                x = x.mul(self.ln_f_gamma.inner.repeatLike(x)).addBias(self.ln_f_beta.inner);
            }

            // 4. Output projection: [d_model, seq] -> [vocab_size, seq]
            if (config.tied_lm_head) {
                return x.matMul(false, self.embed.token_embed.inner, true);
            } else {
                return x.matMul(false, self.out_proj.inner, false);
            }
        }

        /// Cached forward for autoregressive generation (O(1) per token).
        ///
        /// `k_caches`/`v_caches`: packed per-layer [d_model, max_seq] tensors.
        /// Returns logits [vocab_size, 1].
        pub fn forwardCached(
            self: *const Self,
            alloc: Alloc,
            token_id: usize,
            pos: usize,
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
        ) *Tensor(T) {
            var x = self.embed.forwardAt(alloc, token_id, pos);
            for (0..config.n_layers) |i| {
                x = self.blocks[i].forwardCached(x, k_caches[i], v_caches[i], pos);
            }

            var ln_reduce = [_]usize{ 1, 1 };
            x = x.layerNorm(&ln_reduce, 1e-5);
            if (config.learnable_ln) {
                x = x.mul(self.ln_f_gamma.inner.repeatLike(x)).addBias(self.ln_f_beta.inner);
            }

            if (config.tied_lm_head) {
                return x.matMul(false, self.embed.token_embed.inner, true);
            } else {
                return x.matMul(false, self.out_proj.inner, false);
            }
        }

        /// Frozen cached forward for persistent inference plans.
        ///
        /// Like `forwardCached`, but attends over the full KV cache with an
        /// explicit mask.  All shapes are position-independent, so the graph
        /// can be built once and re-executed.
        ///
        /// `x`: [d_model, 1] — token embedding + positional encoding.
        /// `attn_mask`: [max_seq_len, 1] — 0 for valid, -inf for masked.
        pub fn forwardCachedFrozen(
            self: *const Self,
            x_in: *Tensor(T),
            k_caches: [config.n_layers]*Tensor(T),
            v_caches: [config.n_layers]*Tensor(T),
            pos: usize,
            attn_mask: *Tensor(T),
        ) *Tensor(T) {
            var x = x_in;
            for (0..config.n_layers) |i| {
                x = self.blocks[i].forwardCachedFrozen(x, k_caches[i], v_caches[i], pos, attn_mask);
            }

            var ln_reduce = [_]usize{ 1, 1 };
            x = x.layerNorm(&ln_reduce, 1e-5);
            if (config.learnable_ln) {
                x = x.mul(self.ln_f_gamma.inner.repeatLike(x)).addBias(self.ln_f_beta.inner);
            }

            if (config.tied_lm_head) {
                return x.matMul(false, self.embed.token_embed.inner, true);
            } else {
                return x.matMul(false, self.out_proj.inner, false);
            }
        }

        /// Return all learnable parameters.
        pub fn params(self: *const Self) [total_params]*Tensor(T) {
            var result: [total_params]*Tensor(T) = undefined;
            var idx: usize = 0;

            // Embedding params
            for (self.embed.params()) |p| {
                result[idx] = p;
                idx += 1;
            }

            // Block params
            for (0..config.n_layers) |i| {
                for (self.blocks[i].params()) |p| {
                    result[idx] = p;
                    idx += 1;
                }
            }

            // Final layer norm
            if (config.learnable_ln) {
                result[idx] = self.ln_f_gamma.inner;
                idx += 1;
                result[idx] = self.ln_f_beta.inner;
                idx += 1;
            }

            // Output projection
            if (!config.tied_lm_head) {
                result[idx] = self.out_proj.inner;
                idx += 1;
            }

            std.debug.assert(idx == total_params);
            return result;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "GPT - forward produces valid logits" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try GPT(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    }).init(a);

    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 0, 3, 1 });
    const logits = model.forward(indices);

    try g.infer(logits);

    // Output should be [vocab_size=8, seq_len=3]
    try testing.expectEqual(@as(usize, 8), logits.ne[0]);
    try testing.expectEqual(@as(usize, 3), logits.ne[1]);

    for (logits.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "GPT - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try GPT(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
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

test "GPT - cached forward produces valid logits" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };
    const model = try GPT(f32, cfg).init(a);

    // Packed per-layer KV caches: [d_model, max_seq]
    var k_caches: [cfg.n_layers]*Tensor(f32) = undefined;
    var v_caches: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        k_caches[l] = try Tensor(f32).init(a, &.{ cfg.d_model, cfg.max_seq_len });
        v_caches[l] = try Tensor(f32).init(a, &.{ cfg.d_model, cfg.max_seq_len });
    }

    const tokens = [_]usize{ 0, 3, 1 };
    for (tokens, 0..) |tok, pos| {
        const logits = model.forwardCached(a, tok, pos, k_caches, v_caches);
        try g.infer(logits);

        try testing.expectEqual(@as(usize, cfg.vocab_size), logits.ne[0]);
        try testing.expectEqual(@as(usize, 1), logits.ne[1]);
        for (logits.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }

        g.built_forward = false;
        g.nodes.clearRetainingCapacity();
        g.visited_nodes = .empty;
    }
}

test "GPT - cached forward is consistent across steps" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 16,
    };
    const Model = GPT(f32, cfg);
    const tokens = [_]usize{ 2, 5, 0 };

    // Run 1
    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const model = try Model.init(a1);

    var k1: [cfg.n_layers]*Tensor(f32) = undefined;
    var v1: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        k1[l] = try Tensor(f32).init(a1, &.{ cfg.d_model, cfg.max_seq_len });
        v1[l] = try Tensor(f32).init(a1, &.{ cfg.d_model, cfg.max_seq_len });
    }

    var logits1: *Tensor(f32) = undefined;
    for (tokens, 0..) |tok, pos| {
        logits1 = model.forwardCached(a1, tok, pos, k1, v1);
        try g1.infer(logits1);
        g1.built_forward = false;
        g1.nodes.clearRetainingCapacity();
        g1.visited_nodes = .empty;
    }

    // Run 2 — fresh caches, same weights
    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const model2 = try Model.init(a2);
    for (model.params(), model2.params()) |src, dst| @memcpy(dst.data, src.data);

    var k2: [cfg.n_layers]*Tensor(f32) = undefined;
    var v2: [cfg.n_layers]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        k2[l] = try Tensor(f32).init(a2, &.{ cfg.d_model, cfg.max_seq_len });
        v2[l] = try Tensor(f32).init(a2, &.{ cfg.d_model, cfg.max_seq_len });
    }

    var logits2: *Tensor(f32) = undefined;
    for (tokens, 0..) |tok, pos| {
        logits2 = model2.forwardCached(a2, tok, pos, k2, v2);
        try g2.infer(logits2);
        g2.built_forward = false;
        g2.nodes.clearRetainingCapacity();
        g2.visited_nodes = .empty;
    }

    for (logits1.data[0..cfg.vocab_size], logits2.data[0..cfg.vocab_size]) |a, b| {
        try testing.expectApproxEqAbs(a, b, 1e-5);
    }
}

test "GPT - param count is correct" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const model = try GPT(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    }).init(a);

    const p = model.params();
    // 1 (embed) + 2 * (2 + 4) (blocks: w_qkv + w_o + FFN) + 1 (out_proj) = 1 + 12 + 1 = 14
    try testing.expectEqual(@as(usize, 14), p.len);
}
