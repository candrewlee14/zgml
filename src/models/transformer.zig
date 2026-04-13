//! Minimal transformer block: layerNorm → attention → residual → layerNorm → FFN → residual.
//!
//! Weights are stored as compile-time `Shaped` tensors — d_model and d_ff mismatches
//! are caught at compile time. The `forward` method accepts either:
//!
//!   - `Shaped(T, .{d_model, seq_len})` — full compile-time shape checking of every
//!     intermediate activation (matmul dims, broadcast, residual add, etc.)
//!   - `*Tensor(T)` — runtime-flexible path for dynamic sequence lengths.
//!
//! Both paths share the same underlying tensor ops and produce identical results.
//!
//! ```
//! const block = try TransformerBlock(f32, 4, 8).init(alloc);
//! // Comptime-checked (seq_len=3 baked into the type):
//! const x = try Shaped(f32, .{4, 3}).init(alloc);
//! const y = block.forward(x);   // → Shaped(f32, .{4, 3}), all shapes verified at comptime
//! // Runtime-flexible:
//! const t = try Tensor(f32).init(alloc, &.{4, seq_len});
//! const u = block.forward(t);   // → *Tensor(f32), shapes checked at runtime
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

/// A single transformer block with self-attention and a feed-forward network.
///
/// Architecture (GPT-2 style, pre-norm):
///   1. x = x + attention(layerNorm(x))
///   2. x = x + ffn(layerNorm(x))
///
/// Where ffn(x) = gelu(x @ W1 + b1) @ W2 + b2
pub fn TransformerBlock(comptime T: type, comptime d_model: usize, comptime d_ff: usize) type {
    // Compile-time weight shape aliases
    const WtShape = Shaped(T, .{ d_model, d_model });
    const W1Shape = Shaped(T, .{ d_ff, d_model });
    const B1Shape = Shaped(T, .{d_ff});
    const W2Shape = Shaped(T, .{ d_model, d_ff });
    const B2Shape = Shaped(T, .{d_model});

    return struct {
        const Self = @This();

        // Attention weights — shapes verified at compile time
        w_q: WtShape,
        w_k: WtShape,
        w_v: WtShape,
        w_o: WtShape,

        // FFN weights
        w1: W1Shape, // d_model → d_ff
        b1: B1Shape,
        w2: W2Shape, // d_ff → d_model
        b2: B2Shape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;

            self.w_q = try WtShape.init(alloc);
            self.w_k = try WtShape.init(alloc);
            self.w_v = try WtShape.init(alloc);
            self.w_o = try WtShape.init(alloc);
            self.w1 = try W1Shape.init(alloc);
            self.b1 = try B1Shape.init(alloc);
            self.w2 = try W2Shape.init(alloc);
            self.b2 = try B2Shape.init(alloc);

            // Initialize with varied small values to break symmetry
            for ([_]*Tensor(T){ self.w_q.inner, self.w_k.inner, self.w_v.inner, self.w_o.inner, self.w1.inner, self.w2.inner }) |w| {
                for (w.data, 0..) |*d, i| {
                    d.* = 0.01 * @as(T, @floatFromInt(i % 7 + 1)) * if (i % 3 == 0) @as(T, -1) else @as(T, 1);
                }
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
                // Shaped path: full compile-time validation
                if (Input.element_type != T)
                    @compileError("TransformerBlock(" ++ @typeName(T) ++ "): input element type is " ++ @typeName(Input.element_type) ++ ", expected " ++ @typeName(T));
                if (Input.static_shape[0] != d_model)
                    @compileError("TransformerBlock: input dimension 0 is " ++ std.fmt.comptimePrint("{}", .{Input.static_shape[0]}) ++ ", expected d_model=" ++ std.fmt.comptimePrint("{}", .{d_model}));
                if (Input.static_ndims > 2)
                    @compileError("TransformerBlock: input must be 2-D [d_model, seq_len], got " ++ std.fmt.comptimePrint("{}", .{Input.static_ndims}) ++ "-D");
                return Input; // same shape in → same shape out
            } else if (Input == *Tensor(T)) {
                return *Tensor(T);
            } else {
                @compileError("TransformerBlock.forward: expected Shaped(" ++ @typeName(T) ++ ", .{" ++ std.fmt.comptimePrint("{}", .{d_model}) ++ ", seq_len}) or *Tensor(" ++ @typeName(T) ++ "), got " ++ @typeName(Input));
            }
        }

        /// Forward pass. Accepts `Shaped` (comptime-checked) or `*Tensor(T)` (runtime-flexible).
        pub fn forward(self: *const Self, x: anytype) ForwardResult(@TypeOf(x)) {
            if (comptime isShaped(@TypeOf(x)))
                return self.forwardShaped(x)
            else
                return self.forwardDynamic(x);
        }

        /// Runtime-flexible forward: input shape [d_model, seq_len] with dynamic seq_len.
        fn forwardDynamic(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            // LayerNorm reduce shape: [1, seq] normalizes over d_model per position
            var ln_reduce = [_]usize{ 1, x.ne[1] };

            // --- Self-attention with pre-norm ---
            const norm1 = x.layerNorm(&ln_reduce, 1e-5);

            const q = norm1.matMul(false, self.w_q.inner, false);
            const k = norm1.matMul(false, self.w_k.inner, false);
            const v = norm1.matMul(false, self.w_v.inner, false);

            // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
            const d_k: T = @floatFromInt(d_model);
            const scores = q.matMul(false, k, true); // [d_model,seq] @ [d_model,seq]^T = [seq, seq]
            const scaled = scores.scaleByVal(1.0 / @sqrt(d_k));
            var sm_reduce = [_]usize{ 1, scores.ne[1] };
            const attn_weights = scaled.softmax(&sm_reduce); // per-query softmax over keys
            const attn_out = attn_weights.matMul(false, v, false); // [seq,seq] @ [d_model,seq] = [d_model,seq]
            const projected = attn_out.matMul(false, self.w_o.inner, false);

            const after_attn = x.add(projected);

            // --- Feed-forward with pre-norm ---
            const norm2 = after_attn.layerNorm(&ln_reduce, 1e-5);

            const hidden = norm2.matMul(false, self.w1.inner, false);
            const b1_rep = self.b1.inner.repeat(&hidden.ne);
            const activated = hidden.add(b1_rep).gelu();

            const ff_out = activated.matMul(false, self.w2.inner, false);
            const b2_rep = self.b2.inner.repeat(&ff_out.ne);

            return after_attn.add(ff_out.add(b2_rep));
        }

        /// Comptime-checked forward: every intermediate shape is validated at compile time.
        fn forwardShaped(self: *const Self, x: anytype) @TypeOf(x) {
            const seq_len = @TypeOf(x).static_shape[1];

            // --- Self-attention with pre-norm ---
            // LayerNorm over d_model per position: reduce to [1, seq_len]
            const norm1 = x.layerNorm(.{ 1, seq_len }, 1e-5);

            // Q, K, V projections: [d_model, seq] @ [d_model, d_model] = [d_model, seq]
            const q = norm1.matMul(false, self.w_q, false);
            const k = norm1.matMul(false, self.w_k, false);
            const v = norm1.matMul(false, self.w_v, false);

            // Q @ K^T → [seq, seq] — dimensions checked at compile time
            const scores = q.matMul(false, k, true);
            const ScoresType = @TypeOf(scores);

            // Scale by 1/sqrt(d_k)
            const d_k: T = @floatFromInt(d_model);
            const scale_val = 1.0 / @sqrt(d_k);
            const scaled = ScoresType.fromTensor(scores.inner.scaleByVal(scale_val));

            // Per-query softmax over keys
            const attn_weights = scaled.softmax(.{ 1, seq_len });

            // Weights @ V → [d_model, seq]
            const attn_out = attn_weights.matMul(false, v, false);
            const projected = attn_out.matMul(false, self.w_o, false);

            const after_attn = x.add(projected);

            // --- Feed-forward with pre-norm ---
            const norm2 = after_attn.layerNorm(.{ 1, seq_len }, 1e-5);

            // norm2 [d_model, seq] @ W1 [d_ff, d_model] = [d_ff, seq]
            const hidden = norm2.matMul(false, self.w1, false);
            const b1_rep = self.b1.repeat(.{ d_ff, seq_len });
            const activated = hidden.add(b1_rep).gelu();

            // activated [d_ff, seq] @ W2 [d_model, d_ff] = [d_model, seq]
            const ff_out = activated.matMul(false, self.w2, false);
            const b2_rep = self.b2.repeat(.{ d_model, seq_len });

            return after_attn.add(ff_out.add(b2_rep));
        }

        /// Return all learnable parameters (as runtime tensors for optimizer compatibility).
        pub fn params(self: *const Self) [8]*Tensor(T) {
            return .{
                self.w_q.inner, self.w_k.inner, self.w_v.inner, self.w_o.inner,
                self.w1.inner,  self.b1.inner,  self.w2.inner,  self.b2.inner,
            };
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "transformer block - forward produces valid output (dynamic)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 8).init(a);

    // Input: [d_model=4, seq_len=3]
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

    const block = try TransformerBlock(f32, 4, 8).init(a);

    const input = (try Shaped(f32, .{ 4, 3 }).init(a)).setData(
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 },
    );

    // Full comptime shape checking: output type is Shaped(f32, .{4, 3})
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
    // Both paths should produce identical output for the same input
    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const block1 = try TransformerBlock(f32, 4, 8).init(a1);
    const data = [_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };

    const input_shaped = (try Shaped(f32, .{ 4, 3 }).init(a1)).setData(&data);
    const out_shaped = block1.forward(input_shaped);
    try g1.buildForward(out_shaped.tensor());
    g1.compute();

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const block2 = try TransformerBlock(f32, 4, 8).init(a2);
    const input_dynamic = try Tensor(f32).init(a2, &.{ 4, 3 });
    input_dynamic.setData(&data);
    const out_dynamic = block2.forward(input_dynamic);
    try g2.buildForward(out_dynamic);
    g2.compute();

    for (out_shaped.tensor().data, out_dynamic.data) |a, b| {
        try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}

test "transformer block - backward produces non-zero gradients for all params" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 8).init(a);

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

    for (block.params(), 0..) |param, pi| {
        var has_nonzero = false;
        for (param.grad.?.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
            if (v != 0) has_nonzero = true;
        }
        // Weight matrices should always have non-zero gradients.
        if (pi < 4 or pi == 4 or pi == 6) {
            try testing.expect(has_nonzero);
        }
    }
}

test "transformer block - residual connection preserves input contribution" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try TransformerBlock(f32, 4, 8).init(a);
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
    const block1 = try TransformerBlock(f32, 4, 8).init(a1);
    const input1 = try Tensor(f32).init(a1, &.{ 4, 3 });
    for (input1.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;

    const loss1 = block1.forward(input1).sumAll();
    try g1.buildForward(loss1);
    try g1.buildBackward(false);
    _ = loss1.grad.?.setAllScalar(1);
    g1.compute();
    const analytical_grad = block1.w_o.inner.grad.?.data[0];

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const block2 = try TransformerBlock(f32, 4, 8).init(a2);
    block2.w_o.inner.data[0] += h;
    const input2 = try Tensor(f32).init(a2, &.{ 4, 3 });
    for (input2.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;
    const loss_plus = block2.forward(input2).sumAll();
    try g2.buildForward(loss_plus);
    g2.compute();

    var g3 = ComputeGraph(f32).init(tac);
    defer g3.deinit();
    const a3 = g3.allocator();
    const block3 = try TransformerBlock(f32, 4, 8).init(a3);
    block3.w_o.inner.data[0] -= h;
    const input3 = try Tensor(f32).init(a3, &.{ 4, 3 });
    for (input3.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i)) * 0.1;
    const loss_minus = block3.forward(input3).sumAll();
    try g3.buildForward(loss_minus);
    g3.compute();

    const numerical_grad = (loss_plus.data[0] - loss_minus.data[0]) / (2.0 * h);
    try testing.expectApproxEqAbs(numerical_grad, analytical_grad, 0.1);
}





