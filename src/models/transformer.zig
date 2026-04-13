//! Minimal transformer block: layerNorm → attention → residual → layerNorm → FFN → residual.
//!
//! This validates the full zgml stack: matmul, softmax, layerNorm, gelu, add.
//! All ops decompose into primitives; backward works via the chain rule.
//!
//! ```
//! const block = try TransformerBlock(f32, 4, 8).init(alloc);
//! const out = block.forward(alloc, input); // input: [d_model, seq_len]
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;

/// A single transformer block with self-attention and a feed-forward network.
///
/// Architecture (GPT-2 style, pre-norm):
///   1. x = x + attention(layerNorm(x))
///   2. x = x + ffn(layerNorm(x))
///
/// Where ffn(x) = gelu(x @ W1 + b1) @ W2 + b2
pub fn TransformerBlock(comptime T: type, comptime d_model: usize, comptime d_ff: usize) type {
    return struct {
        const Self = @This();

        // Attention weights: W_q, W_k, W_v, W_o all [d_model, d_model]
        w_q: *Tensor(T),
        w_k: *Tensor(T),
        w_v: *Tensor(T),
        w_o: *Tensor(T),

        // FFN weights
        w1: *Tensor(T), // [d_model, d_ff]
        b1: *Tensor(T), // [d_ff]
        w2: *Tensor(T), // [d_ff, d_model]
        b2: *Tensor(T), // [d_model]

        /// Initialize with small random-ish values (zeros for simplicity).
        /// In practice you'd use proper initialization (Xavier, etc.)
        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            // Attention projections
            self.w_q = try Tensor(T).init(alloc, &.{ d_model, d_model });
            self.w_k = try Tensor(T).init(alloc, &.{ d_model, d_model });
            self.w_v = try Tensor(T).init(alloc, &.{ d_model, d_model });
            self.w_o = try Tensor(T).init(alloc, &.{ d_model, d_model });

            // FFN: W1 maps d_model→d_ff, W2 maps d_ff→d_model
            // Shape convention: ne = [cols, rows]. For X @ W: X.ne[0] == W.ne[1].
            self.w1 = try Tensor(T).init(alloc, &.{ d_ff, d_model });
            self.b1 = try Tensor(T).init(alloc, &.{d_ff});
            self.w2 = try Tensor(T).init(alloc, &.{ d_model, d_ff });
            self.b2 = try Tensor(T).init(alloc, &.{d_model});

            // Initialize all to small values
            const scale: T = 0.02;
            for ([_]*Tensor(T){ self.w_q, self.w_k, self.w_v, self.w_o, self.w1, self.w2 }) |w| {
                for (w.data) |*d| d.* = scale;
                w.setParam(alloc);
            }
            for ([_]*Tensor(T){ self.b1, self.b2 }) |b| {
                _ = b.setAllScalar(0);
                b.setParam(alloc);
            }

            return self;
        }

        /// Forward pass: input shape [d_model, seq_len].
        /// Returns output of the same shape.
        pub fn forward(self: *const Self, alloc: Alloc, x: *Tensor(T)) *Tensor(T) {
            // --- Self-attention with pre-norm ---
            const norm1 = x.layerNorm(alloc, &.{1}, 1e-5);

            // Project to Q, K, V: input @ W = [d_model, seq].matMul([d_model, d_model]) = [d_model, seq]
            const q = norm1.matMul(alloc, false, self.w_q, false);
            const k = norm1.matMul(alloc, false, self.w_k, false);
            const v = norm1.matMul(alloc, false, self.w_v, false);

            // Scaled dot-product attention: softmax(Q^T @ K / sqrt(d_k)) @ V^T
            const d_k: T = @floatFromInt(d_model);
            const scores = q.matMul(alloc, true, k, false); // [d_model,seq]^T @ [d_model,seq] = [seq, seq]
            const scale_t = Tensor(T).initScalar(alloc, 1.0 / @sqrt(d_k)) catch unreachable;
            const scaled = scores.mul(alloc, scale_t.repeatLike(alloc, scores));
            const attn_weights = scaled.softmax(alloc, &.{1}); // softmax over columns

            // attn_out = V @ weights^T = [d_model, seq] @ [seq, seq] = [d_model, seq]
            const attn_out = v.matMul(alloc, false, attn_weights, false);
            const projected = attn_out.matMul(alloc, false, self.w_o, false);

            // Residual connection
            const after_attn = x.add(alloc, projected);

            // --- Feed-forward with pre-norm ---
            const norm2 = after_attn.layerNorm(alloc, &.{1}, 1e-5);

            // FFN: gelu(norm2 @ W1 + b1) @ W2 + b2
            // norm2 [d_model, seq] @ W1 [d_model, d_ff] = [d_ff, seq]
            const hidden = norm2.matMul(alloc, false, self.w1, false);
            const b1_rep = self.b1.repeat(alloc, &hidden.ne);
            const activated = hidden.add(alloc, b1_rep).gelu(alloc);

            // activated [d_ff, seq] @ W2 [d_ff, d_model] = [d_model, seq]
            const ff_out = activated.matMul(alloc, false, self.w2, false);
            const b2_rep = self.b2.repeat(alloc, &ff_out.ne);

            // Residual connection
            return after_attn.add(alloc, ff_out.add(alloc, b2_rep));
        }

        /// Return all learnable parameters.
        pub fn params(self: *const Self) [8]*Tensor(T) {
            return .{ self.w_q, self.w_k, self.w_v, self.w_o, self.w1, self.b1, self.w2, self.b2 };
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "transformer block - forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 8).init(a);

    // Input: [d_model=4, seq_len=3]
    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(a, input);

    try g.buildForward(output);
    g.compute();

    // Output should be same shape as input
    try testing.expectEqual(@as(usize, 4), output.ne[0]);
    try testing.expectEqual(@as(usize, 3), output.ne[1]);

    // Values should be finite
    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - backward computes gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 8).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(a, input);
    const loss = output.sumAll(a);

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    // All parameters should have finite gradients
    for (block.params()) |param| {
        for (param.grad.?.data) |v| {
            try testing.expect(!std.math.isNan(v));
        }
    }
}
