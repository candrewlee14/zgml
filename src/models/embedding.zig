//! Token + positional embedding layer.
//!
//! Combines a learnable token embedding table with sinusoidal positional
//! encoding (Vaswani et al., 2017). Token lookup uses `gatherRows` so
//! gradients flow back to the embedding weights during training.
//!
//! ```
//! const embed = try Embedding(f32, 256, 64, 128).init(alloc);
//! const indices = try Tensor(f32).initIndexVectorCopy(alloc, &.{5, 12, 0, 42});
//! const x = embed.forward(indices);  // -> [d_model=64, seq_len=4]
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;

/// Token embedding + sinusoidal positional encoding.
///
/// Type parameters:
///   - `T`: element type (f32, f64)
///   - `vocab_size`: number of tokens in the vocabulary
///   - `d_model`: embedding/model dimension
///   - `max_seq_len`: maximum supported sequence length for positional encoding
pub fn Embedding(comptime T: type, comptime vocab_size: usize, comptime d_model: usize, comptime max_seq_len: usize) type {
    return struct {
        const Self = @This();

        /// Learnable token embedding table: [d_model, vocab_size].
        /// Row i (column in zgml layout) contains the embedding for token i.
        token_embed: Shaped(T, .{ d_model, vocab_size }),

        /// Pre-computed sinusoidal positional encoding: [d_model, max_seq_len].
        /// This is a constant (not a learnable parameter).
        pos_encode: *Tensor(T),

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            // Initialize token embeddings as learnable parameters
            self.token_embed = try Shaped(T, .{ d_model, vocab_size }).init(alloc);
            // Xavier-style initialization: scale by 1/sqrt(d_model)
            const scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(d_model)));
            for (self.token_embed.inner.data, 0..) |*d, i| {
                // Deterministic pseudo-random initialization
                const fi: T = @floatFromInt(i);
                d.* = scale * @sin(fi * 0.1 + 0.3) * @cos(fi * 0.07 + 0.5);
            }
            self.token_embed.inner.setParam();

            // Pre-compute sinusoidal positional encoding
            self.pos_encode = try Tensor(T).init(alloc, &.{ d_model, max_seq_len });
            for (0..max_seq_len) |pos| {
                for (0..d_model) |i| {
                    const p: T = @floatFromInt(pos);
                    const dim_pair: T = @floatFromInt(2 * (i / 2)); // floor to even
                    const dm: T = @floatFromInt(d_model);
                    const angle = p / std.math.pow(T, 10000.0, dim_pair / dm);
                    self.pos_encode.data[pos * d_model + i] = if (i % 2 == 0)
                        @sin(angle)
                    else
                        @cos(angle);
                }
            }
            // pos_encode is NOT a parameter — no gradients needed

            return self;
        }

        /// Forward pass: look up token embeddings and add positional encoding.
        ///
        /// `token_indices` is a 1-D index tensor of shape [seq_len] where each
        /// element is a token ID in [0, vocab_size). Create it with
        /// `Tensor(T).initIndexVectorCopy(alloc, &.{id0, id1, ...})`.
        ///
        /// Returns a tensor of shape [d_model, seq_len].
        pub fn forward(self: *const Self, token_indices: *Tensor(T)) *Tensor(T) {
            const seq_len = token_indices.ne[0];
            std.debug.assert(seq_len <= max_seq_len);

            // Gather token embeddings: [d_model, vocab_size] gather [seq_len] -> [d_model, seq_len]
            const tok_embed = self.token_embed.inner.gatherRows(token_indices);

            // Slice positional encoding to seq_len.
            // We create a [d_model, seq_len] view by reshaping the first d_model*seq_len elements.
            const alloc = token_indices.alloc.?;
            const pos_enc = Tensor(T).init(alloc, &.{ d_model, seq_len }) catch unreachable;
            const elems = d_model * seq_len;
            @memcpy(pos_enc.data[0..elems], self.pos_encode.data[0..elems]);
            // pos_enc is a constant — no grad

            return tok_embed.add(pos_enc.repeatLike(tok_embed));
        }

        /// Return all learnable parameters.
        pub fn params(self: *const Self) [1]*Tensor(T) {
            return .{self.token_embed.inner};
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "embedding - forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const embed = try Embedding(f32, 10, 4, 8).init(a);

    // Token indices: [3] (3 tokens from vocab of 10)
    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 2, 5, 0 });

    const output = embed.forward(indices);

    try g.infer(output);

    try testing.expectEqual(@as(usize, 4), output.ne[0]); // d_model
    try testing.expectEqual(@as(usize, 3), output.ne[1]); // seq_len

    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "embedding - positional encoding varies by position" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const embed = try Embedding(f32, 10, 4, 8).init(a);

    // Same token at different positions should produce different embeddings
    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 3, 3, 3 });

    const output = embed.forward(indices);

    try g.infer(output);

    // Positions 0 and 1 should differ (because pos encoding differs)
    var any_diff = false;
    for (0..4) |feat| {
        if (output.data[0 * 4 + feat] != output.data[1 * 4 + feat]) {
            any_diff = true;
            break;
        }
    }
    try testing.expect(any_diff);
}

test "embedding - backward produces gradients for token embeddings" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const embed = try Embedding(f32, 10, 4, 8).init(a);
    const indices = try Tensor(f32).initIndexVectorCopy(a, &.{ 1, 4 });

    const loss = embed.forward(indices).sumAll();

    try g.run(loss);

    // Token embedding should have gradients
    const grad = embed.token_embed.inner.grad.?;
    var has_nonzero = false;
    for (grad.data) |v| {
        try testing.expect(!std.math.isNan(v));
        if (v != 0) has_nonzero = true;
    }
    try testing.expect(has_nonzero);
}
