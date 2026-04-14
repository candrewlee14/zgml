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
};

/// GPT-style decoder-only transformer.
///
/// Architecture:
///   1. Token embedding + sinusoidal positional encoding
///   2. N x causal TransformerBlock (pre-norm, multi-head attention + FFN)
///   3. Final layer normalization
///   4. Linear projection to vocabulary logits
pub fn GPT(comptime T: type, comptime config: GPTConfig) type {
    const Block = TransformerBlock(T, config.d_model, config.n_heads, config.d_ff, true);
    const Embed = Embedding(T, config.vocab_size, config.d_model, config.max_seq_len);

    // Output projection weight shape: [vocab_size, d_model]
    const OutProjShape = Shaped(T, .{ config.vocab_size, config.d_model });

    // Number of params per block
    const params_per_block = 4 * config.n_heads + 4;
    // Total: embedding(1) + blocks(params_per_block * n_layers) + output_proj(1)
    const total_params = 1 + params_per_block * config.n_layers + 1;

    return struct {
        const Self = @This();

        embed: Embed,
        blocks: [config.n_layers]Block,
        out_proj: OutProjShape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            self.embed = try Embed.init(alloc);

            for (0..config.n_layers) |i| {
                self.blocks[i] = try Block.init(alloc);
            }

            self.out_proj = try OutProjShape.init(alloc);
            nn.kaimingUniform(T, self.out_proj.inner, 99);
            self.out_proj.inner.setParam();

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

            // 3. Final layer norm
            var ln_reduce = [_]usize{ 1, seq_len };
            x = x.layerNorm(&ln_reduce, 1e-5);

            // 4. Output projection: [d_model, seq] @ [vocab_size, d_model] = [vocab_size, seq]
            return x.matMul(false, self.out_proj.inner, false);
        }

        /// Cached forward for autoregressive generation (O(1) per token).
        ///
        /// Processes a single token at position `pos` using KV caches.
        /// On first call (pos=0), also prefills the cache. For generation,
        /// call with each new token sequentially: pos=0,1,2,...
        ///
        /// `token_id`: vocabulary index of the new token.
        /// `pos`: current sequence position.
        /// `alloc`: allocator for intermediate tensors.
        /// `k_caches`/`v_caches`: per-layer, per-head [d_head, max_seq] tensors.
        ///
        /// Returns logits [vocab_size, 1].
        pub fn forwardCached(
            self: *const Self,
            alloc: Alloc,
            token_id: usize,
            pos: usize,
            k_caches: [config.n_layers][config.n_heads]*Tensor(T),
            v_caches: [config.n_layers][config.n_heads]*Tensor(T),
        ) *Tensor(T) {
            // 1. Single-token embedding at position
            var x = self.embed.forwardAt(alloc, token_id, pos);

            // 2. Transformer blocks with KV caches
            for (0..config.n_layers) |i| {
                x = self.blocks[i].forwardCached(x, k_caches[i], v_caches[i], pos);
            }

            // 3. Final layer norm (single token: reduce over [1,1])
            var ln_reduce = [_]usize{ 1, 1 };
            x = x.layerNorm(&ln_reduce, 1e-5);

            // 4. Output projection: [d_model, 1] @ [vocab_size, d_model] = [vocab_size, 1]
            return x.matMul(false, self.out_proj.inner, false);
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

            // Output projection
            result[idx] = self.out_proj.inner;
            idx += 1;

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
    const Model = GPT(f32, cfg);
    const model = try Model.init(a);

    const d_head = cfg.d_model / cfg.n_heads;

    // Allocate KV caches: [n_layers][n_heads] of [d_head, max_seq]
    var k_caches: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    var v_caches: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| {
        for (0..cfg.n_heads) |h| {
            k_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, cfg.max_seq_len });
            v_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, cfg.max_seq_len });
        }
    }

    // Run 3 tokens through cached forward
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

        // Reset graph for next step (KV caches persist in tensor data)
        g.built_forward = false;
        g.nodes.clearRetainingCapacity();
        g.visited_nodes = .empty;
    }
}

test "GPT - cached forward is consistent across steps" {
    // Verify that running the same prompt through cached forward twice
    // (with reset cache) produces identical results.
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 16,
    };
    const Model = GPT(f32, cfg);
    const d_head = cfg.d_model / cfg.n_heads;
    const tokens = [_]usize{ 2, 5, 0 };

    // Run 1
    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const model = try Model.init(a1);

    var k1: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    var v1: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| for (0..cfg.n_heads) |h| {
        k1[l][h] = try Tensor(f32).init(a1, &.{ d_head, cfg.max_seq_len });
        v1[l][h] = try Tensor(f32).init(a1, &.{ d_head, cfg.max_seq_len });
    };

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

    var k2: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    var v2: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| for (0..cfg.n_heads) |h| {
        k2[l][h] = try Tensor(f32).init(a2, &.{ d_head, cfg.max_seq_len });
        v2[l][h] = try Tensor(f32).init(a2, &.{ d_head, cfg.max_seq_len });
    };

    var logits2: *Tensor(f32) = undefined;
    for (tokens, 0..) |tok, pos| {
        logits2 = model2.forwardCached(a2, tok, pos, k2, v2);
        try g2.infer(logits2);
        g2.built_forward = false;
        g2.nodes.clearRetainingCapacity();
        g2.visited_nodes = .empty;
    }

    // Same weights + same tokens → identical logits
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
    // 1 (embed) + 2 * (4*2 + 4) (blocks) + 1 (out_proj) = 1 + 24 + 1 = 26
    try testing.expectEqual(@as(usize, 26), p.len);
}
