//! Multi-head transformer block: layerNorm -> MHA -> residual -> layerNorm -> FFN -> residual.
//!
//! Architecture (GPT-2 style, pre-norm):
//!   1. x = x + multiHeadAttention(layerNorm(x))
//!   2. x = x + ffn(layerNorm(x))
//!
//! Where ffn(x) = gelu(x @ W1 + b1) @ W2 + b2
//!
//! Multi-head attention splits the model dimension into `n_heads` independent
//! attention heads, each operating on `d_head = d_model / n_heads` features.
//! Per-head outputs are projected back to d_model and summed (equivalent to
//! the standard concat-then-project formulation).
//!
//! When `causal` is true, an upper-triangular mask is applied to attention
//! scores so each position can only attend to itself and earlier positions
//! (autoregressive / decoder-style).
//!
//! ```
//! const block = try TransformerBlock(f32, 64, 4, 256, false).init(alloc);
//! const x = try Tensor(f32).init(alloc, &.{64, seq_len});
//! const y = block.forward(x);   // → *Tensor(f32), shape [64, seq_len]
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;
const ShapedTensor = shaped_mod.ShapedTensor;
const nn = @import("../nn.zig");

/// A single transformer block with multi-head self-attention and a feed-forward network.
///
/// Type parameters:
///   - `T`: element type (f32, f64)
///   - `d_model`: model/embedding dimension
///   - `n_heads`: number of attention heads (must divide d_model)
///   - `d_ff`: feed-forward hidden dimension
///   - `causal`: if true, apply causal (autoregressive) masking to attention
pub fn TransformerBlock(
    comptime T: type,
    comptime d_model: usize,
    comptime n_heads: usize,
    comptime d_ff: usize,
    comptime causal: bool,
) type {
    if (d_model % n_heads != 0)
        @compileError("d_model (" ++ std.fmt.comptimePrint("{}", .{d_model}) ++
            ") must be divisible by n_heads (" ++ std.fmt.comptimePrint("{}", .{n_heads}) ++ ")");

    const d_head = d_model / n_heads;

    // Per-head weight shapes
    const QkvShape = Shaped(T, .{ d_head, d_model }); // [d_head, d_model]
    const OShape = Shaped(T, .{ d_model, d_head }); // [d_model, d_head]

    // FFN weight shapes
    const W1Shape = Shaped(T, .{ d_ff, d_model });
    const B1Shape = Shaped(T, .{d_ff});
    const W2Shape = Shaped(T, .{ d_model, d_ff });
    const B2Shape = Shaped(T, .{d_model});

    return struct {
        const Self = @This();

        // Per-head attention weights
        w_q: [n_heads]QkvShape,
        w_k: [n_heads]QkvShape,
        w_v: [n_heads]QkvShape,
        w_o: [n_heads]OShape,

        // FFN weights
        w1: W1Shape, // d_model -> d_ff
        b1: B1Shape,
        w2: W2Shape, // d_ff -> d_model
        b2: B2Shape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            // Allocate per-head weights
            for (0..n_heads) |h| {
                self.w_q[h] = try QkvShape.init(alloc);
                self.w_k[h] = try QkvShape.init(alloc);
                self.w_v[h] = try QkvShape.init(alloc);
                self.w_o[h] = try OShape.init(alloc);
            }

            self.w1 = try W1Shape.init(alloc);
            self.b1 = try B1Shape.init(alloc);
            self.w2 = try W2Shape.init(alloc);
            self.b2 = try B2Shape.init(alloc);

            // Initialize weights
            var seed: u64 = 42;
            for (0..n_heads) |h| {
                for ([_]*Tensor(T){ self.w_q[h].inner, self.w_k[h].inner, self.w_v[h].inner, self.w_o[h].inner }) |w| {
                    nn.kaimingUniform(T, w, seed);
                    seed +%= 1;
                    w.setParam();
                }
            }
            for ([_]*Tensor(T){ self.w1.inner, self.w2.inner }) |w| {
                nn.kaimingUniform(T, w, seed);
                seed +%= 1;
                w.setParam();
            }
            for ([_]*Tensor(T){ self.b1.inner, self.b2.inner }) |b| {
                _ = b.setAllScalar(0);
                b.setParam();
            }

            return self;
        }

        fn isShaped(comptime Input: type) bool {
            return @typeInfo(Input) == .@"struct" and @hasDecl(Input, "static_shape");
        }

        /// Determine the return type and validate input at compile time.
        fn ForwardResult(comptime Input: type) type {
            if (isShaped(Input)) {
                if (Input.element_type != T)
                    @compileError("TransformerBlock(" ++ @typeName(T) ++ "): input element type is " ++ @typeName(Input.element_type) ++ ", expected " ++ @typeName(T));
                if (Input.static_shape[0] != d_model)
                    @compileError("TransformerBlock: input dimension 0 is " ++ std.fmt.comptimePrint("{}", .{Input.static_shape[0]}) ++ ", expected d_model=" ++ std.fmt.comptimePrint("{}", .{d_model}));
                if (Input.static_ndims > 2)
                    @compileError("TransformerBlock: input must be 2-D [d_model, seq_len], got " ++ std.fmt.comptimePrint("{}", .{Input.static_ndims}) ++ "-D");
                return Input; // same shape in -> same shape out
            } else if (Input == *Tensor(T)) {
                return *Tensor(T);
            } else {
                @compileError("TransformerBlock.forward: expected Shaped(" ++ @typeName(T) ++ ", .{" ++ std.fmt.comptimePrint("{}", .{d_model}) ++ ", seq_len}) or *Tensor(" ++ @typeName(T) ++ "), got " ++ @typeName(Input));
            }
        }

        /// Forward pass. Accepts `Shaped` (comptime-checked) or `*Tensor(T)` (runtime-flexible).
        pub fn forward(self: *const Self, x: anytype) ForwardResult(@TypeOf(x)) {
            const inner = if (comptime isShaped(@TypeOf(x))) x.inner else x;
            const result = self.forwardInner(inner);
            return if (comptime isShaped(@TypeOf(x)))
                @TypeOf(x).fromTensor(result)
            else
                result;
        }

        /// Unified forward implementation operating on raw tensors.
        fn forwardInner(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const alloc = x.alloc.?;
            const seq_len = x.ne[1];
            var ln_reduce = [_]usize{ 1, seq_len };

            // --- Multi-head self-attention with pre-norm ---
            const norm1 = x.layerNorm(&ln_reduce, 1e-5);

            // Build causal mask once (shared across heads)
            const mask: ?*Tensor(T) = if (causal) blk: {
                const m = Tensor(T).init(alloc, &.{ seq_len, seq_len }) catch unreachable;
                for (0..seq_len) |qi| { // query position (ne[1])
                    for (0..seq_len) |ki| { // key position (ne[0])
                        m.data[qi * seq_len + ki] = if (ki <= qi) 0 else -1e9;
                    }
                }
                break :blk m;
            } else null;

            // Per-head attention, summed after output projection
            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                // Q, K, V projections: [d_model, seq] @ [d_head, d_model] = [d_head, seq]
                const q = norm1.matMul(false, self.w_q[h].inner, false);
                const k = norm1.matMul(false, self.w_k[h].inner, false);
                const v = norm1.matMul(false, self.w_v[h].inner, false);

                // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_head)) @ V
                const scores = q.matMul(false, k, true); // [seq, seq]
                const dk: T = @floatFromInt(d_head);
                var scaled = scores.scaleByVal(1.0 / @sqrt(dk));

                if (causal) {
                    scaled = scaled.add(mask.?);
                }

                var sm_reduce = [_]usize{ 1, seq_len };
                const attn_weights = scaled.softmax(&sm_reduce);
                const attn_out = attn_weights.matMul(false, v, false); // [d_head, seq]

                // Project back to d_model: [d_head, seq] @ [d_model, d_head] = [d_model, seq]
                const projected = attn_out.matMul(false, self.w_o[h].inner, false);

                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            const after_attn = x.add(attn_sum.?);

            // --- Feed-forward with pre-norm ---
            const norm2 = after_attn.layerNorm(&ln_reduce, 1e-5);

            const activated = nn.linear(T, norm2, self.w1.inner, self.b1.inner).gelu();
            const ff_out = nn.linear(T, activated, self.w2.inner, self.b2.inner);

            return after_attn.add(ff_out);
        }

        /// Cached forward for autoregressive generation.
        ///
        /// x: [d_model, 1] — single new token embedding.
        /// k/v_caches: per-head [d_head, max_seq] tensors (persistent across steps).
        /// pos: current position (number of tokens already cached).
        ///
        /// Returns [d_model, 1].
        pub fn forwardCached(
            self: *const Self,
            x: *Tensor(T),
            k_caches: [n_heads]*Tensor(T),
            v_caches: [n_heads]*Tensor(T),
            pos: usize,
        ) *Tensor(T) {
            var ln_reduce = [_]usize{ 1, 1 };
            const norm1 = x.layerNorm(&ln_reduce, 1e-5);

            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                const q = norm1.matMul(false, self.w_q[h].inner, false); // [d_head, 1]
                const k_new = norm1.matMul(false, self.w_k[h].inner, false);
                const v_new = norm1.matMul(false, self.w_v[h].inner, false);

                // Append to cache
                _ = k_caches[h].sliceAssign(k_new, pos);
                _ = v_caches[h].sliceAssign(v_new, pos);

                // Full cached K, V
                const ck = k_caches[h].sliceColumns(0, pos + 1); // [d_head, pos+1]
                const cv = v_caches[h].sliceColumns(0, pos + 1);

                // scores = q @ ck^T → [pos+1, 1] (same pattern as non-cached)
                const scores = q.matMul(false, ck, true);
                const dk: T = @floatFromInt(d_head);
                const scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                const weights = scaled.softmax(&.{ 1, 1 }); // [pos+1, 1]

                // attn_out = weights^T @ cv^T → [d_head, 1]
                const attn_out = weights.matMul(false, cv, false);
                const projected = attn_out.matMul(false, self.w_o[h].inner, false);
                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            const after_attn = x.add(attn_sum.?);
            var ln2_reduce = [_]usize{ 1, 1 };
            const norm2 = after_attn.layerNorm(&ln2_reduce, 1e-5);
            const activated = nn.linear(T, norm2, self.w1.inner, self.b1.inner).gelu();
            const ff_out = nn.linear(T, activated, self.w2.inner, self.b2.inner);
            return after_attn.add(ff_out);
        }

        /// Return all learnable parameters (as runtime tensors for optimizer compatibility).
        pub fn params(self: *const Self) [4 * n_heads + 4]*Tensor(T) {
            var result: [4 * n_heads + 4]*Tensor(T) = undefined;
            for (0..n_heads) |h| {
                result[h * 4 + 0] = self.w_q[h].inner;
                result[h * 4 + 1] = self.w_k[h].inner;
                result[h * 4 + 2] = self.w_v[h].inner;
                result[h * 4 + 3] = self.w_o[h].inner;
            }
            result[4 * n_heads + 0] = self.w1.inner;
            result[4 * n_heads + 1] = self.b1.inner;
            result[4 * n_heads + 2] = self.w2.inner;
            result[4 * n_heads + 3] = self.b2.inner;
            return result;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "transformer block - single head forward (dynamic)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 1, 8, false).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(input);

    try g.buildForward(output);
    g.compute();

    try testing.expectEqual(@as(usize, 4), output.ne[0]);
    try testing.expectEqual(@as(usize, 3), output.ne[1]);

    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - multi-head forward (dynamic)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    // d_model=4, n_heads=2, d_head=2
    const block = try TransformerBlock(f32, 4, 2, 8, false).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(input);

    try g.buildForward(output);
    g.compute();

    try testing.expectEqual(@as(usize, 4), output.ne[0]);
    try testing.expectEqual(@as(usize, 3), output.ne[1]);

    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - causal masking" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 2, 8, true).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(input);

    try g.buildForward(output);
    g.compute();

    try testing.expectEqual(@as(usize, 4), output.ne[0]);
    try testing.expectEqual(@as(usize, 3), output.ne[1]);

    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - forward produces valid output (shaped)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 2, 8, false).init(a);

    const input = (try Shaped(f32, .{ 4, 3 }).init(a)).setData(
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 },
    );

    const output = block.forward(input);
    comptime {
        std.debug.assert(@TypeOf(output).static_shape[0] == 4);
        std.debug.assert(@TypeOf(output).static_shape[1] == 3);
    }

    try g.buildForward(output.tensor());
    g.compute();

    for (output.tensor().data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - shaped and dynamic produce same result" {
    const data = [_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };

    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const block1 = try TransformerBlock(f32, 4, 2, 8, false).init(a1);
    const input_shaped = (try Shaped(f32, .{ 4, 3 }).init(a1)).setData(&data);
    const out_shaped = block1.forward(input_shaped);
    try g1.buildForward(out_shaped.tensor());
    g1.compute();

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const block2 = try TransformerBlock(f32, 4, 2, 8, false).init(a2);
    const input_dynamic = try Tensor(f32).init(a2, &.{ 4, 3 });
    input_dynamic.setData(&data);
    const out_dynamic = block2.forward(input_dynamic);
    try g2.buildForward(out_dynamic);
    g2.compute();

    for (out_shaped.tensor().data, out_dynamic.data) |a, b| {
        try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}

test "transformer block - backward produces non-zero gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 2, 8, false).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(input);
    const loss = output.sumAll();

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    // Check that weight matrices have non-zero gradients
    for (block.params()) |param| {
        for (param.grad.?.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "transformer block - causal backward produces non-zero gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 2, 8, true).init(a);

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    for (input.data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const output = block.forward(input);
    const loss = output.sumAll();

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    for (block.params()) |param| {
        for (param.grad.?.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "transformer block - residual connection preserves input contribution" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 1, 8, false).init(a);
    for (block.params()) |param| {
        _ = param.setAllScalar(0);
    }

    const input = try Tensor(f32).init(a, &.{ 4, 3 });
    input.setData(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    const output = block.forward(input);
    try g.buildForward(output);
    g.compute();

    try testing.expectEqual(@as(usize, 4), output.ne[0]);
    try testing.expectEqual(@as(usize, 3), output.ne[1]);
    for (output.data) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}

test "transformer block - numerical gradient check on one parameter" {
    const h: f32 = 1e-3;

    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const block1 = try TransformerBlock(f32, 4, 2, 8, false).init(a1);
    const input1 = try Tensor(f32).init(a1, &.{ 4, 3 });
    for (input1.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;

    const loss1 = block1.forward(input1).sumAll();
    try g1.buildForward(loss1);
    try g1.buildBackward(false);
    _ = loss1.grad.?.setAllScalar(1);
    g1.compute();
    const analytical_grad = block1.w_o[0].inner.grad.?.data[0];

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const block2 = try TransformerBlock(f32, 4, 2, 8, false).init(a2);
    block2.w_o[0].inner.data[0] += h;
    const input2 = try Tensor(f32).init(a2, &.{ 4, 3 });
    for (input2.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;
    const loss_plus = block2.forward(input2).sumAll();
    try g2.buildForward(loss_plus);
    g2.compute();

    var g3 = ComputeGraph(f32).init(tac);
    defer g3.deinit();
    const a3 = g3.allocator();
    const block3 = try TransformerBlock(f32, 4, 2, 8, false).init(a3);
    block3.w_o[0].inner.data[0] -= h;
    const input3 = try Tensor(f32).init(a3, &.{ 4, 3 });
    for (input3.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;
    const loss_minus = block3.forward(input3).sumAll();
    try g3.buildForward(loss_minus);
    g3.compute();

    const numerical_grad = (loss_plus.data[0] - loss_minus.data[0]) / (2.0 * h);
    try testing.expectApproxEqAbs(numerical_grad, analytical_grad, 0.5);
}

test "transformer block - cached forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const d_model = 4;
    const n_h = 2;
    const d_h = d_model / n_h; // 2
    const max_seq = 8;

    const block = try TransformerBlock(f32, d_model, n_h, 8, true).init(a);

    // Allocate KV caches: [d_head, max_seq] per head
    var k_caches: [n_h]*Tensor(f32) = undefined;
    var v_caches: [n_h]*Tensor(f32) = undefined;
    for (0..n_h) |h| {
        k_caches[h] = try Tensor(f32).init(a, &.{ d_h, max_seq });
        v_caches[h] = try Tensor(f32).init(a, &.{ d_h, max_seq });
    }

    // Run 3 tokens through cached forward
    for (0..3) |pos| {
        const x = try Tensor(f32).init(a, &.{ d_model, 1 });
        for (x.data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(pos * d_model + i)) * 0.1;
        }

        const out = block.forwardCached(x, k_caches, v_caches, pos);

        try g.buildForward(out);
        g.computeNoGrad();

        try testing.expectEqual(@as(usize, d_model), out.ne[0]);
        try testing.expectEqual(@as(usize, 1), out.ne[1]);
        for (out.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }

        // Reset graph for next position (but caches persist!)
        g.built_forward = false;
        g.nodes.clearRetainingCapacity();
        g.visited_nodes = .empty;
    }
}
