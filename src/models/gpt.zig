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
            // Initialize output projection
            const scale: T = 0.02;
            for (self.out_proj.inner.data, 0..) |*d, i| {
                d.* = scale * @as(T, @floatFromInt(i % 7 + 1)) * if (i % 3 == 0) @as(T, -1) else @as(T, 1);
            }
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

    try g.buildForward(logits);
    g.compute();

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
    const logits = model.forward(indices);
    const loss = logits.sumAll();

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    // All weight params should have finite gradients
    for (model.params()) |param| {
        if (param.grad) |grad| {
            for (grad.data) |v| {
                try testing.expect(!std.math.isNan(v));
                try testing.expect(!std.math.isInf(v));
            }
        }
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
