//! Neural network building blocks.
//!
//! Utility layers that decompose into primitive tensor ops. These don't
//! introduce new ops to the IR — they build graph subgraphs from existing
//! primitives, so backpropagation works automatically.
//!
//! Also provides free functions for common patterns:
//!   - `linear(T, x, w, b)` — fully-connected layer (matmul + optional bias)
//!   - `kaimingUniform(T, tensor, seed)` — weight initialization
//!   - `uniform(T, tensor, low, high, seed)` — uniform random initialization

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Alloc = std.mem.Allocator;

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Fully-connected layer: `x @ w + b`.
///
/// Computes `x.matMul(w)` and adds bias via broadcasting if `b` is non-null.
/// This is the standard linear/dense layer used in most networks.
///
/// ```
/// const h = nn.linear(f32, x, w1, b1); // x @ w1 + b1
/// const o = nn.linear(f32, h, w2, null); // h @ w2 (no bias)
/// ```
pub fn linear(comptime T: type, x: *Tensor(T), w: *Tensor(T), b: ?*Tensor(T)) *Tensor(T) {
    const h = x.matMul(false, w, false);
    return if (b) |bias| h.addBias(bias) else h;
}

/// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
///
/// Composed from primitives (neg, exp, add, recip, mul) so gradients
/// flow automatically. Used by SwiGLU FFN in LLaMA-style models.
pub fn silu(comptime T: type, x: *Tensor(T)) *Tensor(T) {
    const alloc = x.alloc.?;
    const one = Tensor(T).initScalar(alloc, 1) catch unreachable;
    const exp_neg = x.neg().exp();
    const sigmoid = exp_neg.add(one.repeatLike(exp_neg)).recip();
    return x.mul(sigmoid);
}

/// Build a lower-triangular causal attention mask [seq_len, seq_len].
///
/// Returns 0 for valid (attend) positions and `mask_val` (typically -1e9)
/// for masked positions.
pub fn buildCausalMask(comptime T: type, alloc: Alloc, seq_len: usize) *Tensor(T) {
    const mask = Tensor(T).init(alloc, &.{ seq_len, seq_len }) catch unreachable;
    for (0..seq_len) |qi| {
        for (0..seq_len) |ki| {
            mask.data[qi * seq_len + ki] = if (ki <= qi) 0 else -1e9;
        }
    }
    return mask;
}

/// Batch normalization for 4D tensors [W, H, C, N].
///
/// Normalizes per-channel across the spatial and batch dimensions,
/// then applies learnable scale (gamma) and shift (beta).
/// Uses the same composed-primitive approach as layerNorm.
///
/// ```
/// const bn_g = try g.param(&.{n_filters}); // gamma (scale)
/// const bn_b = try g.param(&.{n_filters}); // beta (shift)
/// // init gamma=1, beta=0
/// const y = nn.batchNorm2d(f32, conv_out, bn_g, bn_b, 1e-5);
/// ```
pub fn batchNorm2d(comptime T: type, x: *Tensor(T), gamma: *Tensor(T), beta: *Tensor(T), eps: T) *Tensor(T) {
    std.debug.assert(x.n_dims == 4);
    const C = x.ne[2];
    // Normalize per-channel: reduce over W, H, N → [1, 1, C, 1]
    const normalized = x.layerNorm(&.{ 1, 1, C, 1 }, eps);
    // Scale and shift with learnable parameters
    const g4 = gamma.reshape(&.{ 1, 1, C, 1 });
    const b4 = beta.reshape(&.{ 1, 1, C, 1 });
    var out_ne = [_]usize{ x.ne[0], x.ne[1], x.ne[2], x.ne[3] };
    return normalized.mul(g4.repeat(out_ne[0..])).add(b4.repeat(out_ne[0..]));
}

/// Initialize tensor weights with Kaiming uniform distribution.
///
/// Draws from U(-bound, +bound) where bound = sqrt(6 / fan_in).
/// Standard initialization for layers followed by ReLU.
///
/// For 2D weights {out_features, in_features}: fan_in = in_features.
/// For ≥3D conv kernels {kw, kh, c_in, c_out}: fan_in = kw * kh * c_in.
pub fn kaimingUniform(comptime T: type, tensor: *Tensor(T), seed: u64) void {
    const fan_in: usize = if (tensor.n_dims == 2)
        tensor.ne[1]
    else blk: {
        var fi: usize = 1;
        for (tensor.ne[0 .. tensor.n_dims - 1]) |d| fi *= d;
        break :blk fi;
    };
    const bound: T = @sqrt(6.0 / @as(T, @floatFromInt(fan_in)));
    var rng = std.Random.DefaultPrng.init(seed);
    var random = rng.random();
    for (tensor.data) |*d| {
        d.* = (random.float(T) * 2.0 - 1.0) * bound;
    }
}

/// Initialize tensor with uniform random values in [low, high).
pub fn uniform(comptime T: type, tensor: *Tensor(T), low: T, high: T, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    var random = rng.random();
    const range = high - low;
    for (tensor.data) |*d| {
        d.* = random.float(T) * range + low;
    }
}

/// Argmax over rows: for each column (sample), find the row index with the largest value.
///
/// `logits` has shape `{n_classes, batch}`. Writes one predicted class index per sample
/// into `preds[0..batch]`.
pub fn argmax(comptime T: type, logits: *const Tensor(T), preds: []usize) void {
    const n_classes = logits.ne[0];
    const batch = logits.ne[1];
    std.debug.assert(preds.len >= batch);
    for (0..batch) |s| {
        var best_class: usize = 0;
        var best_val: T = logits.data[s * n_classes];
        for (1..n_classes) |c| {
            const val = logits.data[s * n_classes + c];
            if (val > best_val) {
                best_val = val;
                best_class = c;
            }
        }
        preds[s] = best_class;
    }
}

// ---------------------------------------------------------------------------
// Generic training loops
// ---------------------------------------------------------------------------

const ComputeGraph = @import("graph.zig").ComputeGraph;

/// Generic supervised training loop: batched epochs over (xs, ys) pairs.
///
/// Handles batching, memcpy into graph slots, forward+backward, and optimizer step.
/// Works with any model that exposes its graph, loss, and batch tensors.
///
/// ```
/// // Instead of writing a train() method on every model:
/// nn.trainSupervised(f32, &model.g, model.loss, model.xs_batch, model.ys_batch,
///                    xs, ys, 500, &optimizer);
/// ```
pub fn trainSupervised(
    comptime T: type,
    g: *ComputeGraph(T),
    loss_node: *Tensor(T),
    xs_batch: *Tensor(T),
    ys_batch: *Tensor(T),
    xs: *Tensor(T),
    ys: *Tensor(T),
    n_epochs: usize,
    optimizer: anytype,
) !void {
    const n_elems_x = xs_batch.nElems();
    const n_elems_y = ys_batch.nElems();
    std.debug.assert(xs.nElems() % n_elems_x == 0);
    std.debug.assert(ys.nElems() % n_elems_y == 0);
    const n_batches = xs.nElems() / n_elems_x;
    for (0..n_epochs) |_| {
        for (0..n_batches) |b_idx| {
            @memcpy(xs_batch.data, xs.data[b_idx * n_elems_x ..][0..n_elems_x]);
            @memcpy(ys_batch.data, ys.data[b_idx * n_elems_y ..][0..n_elems_y]);
            try g.run(loss_node);
            optimizer.step();
        }
    }
}

/// Generic unsupervised training loop: batched epochs over xs only.
///
/// For autoencoders, VAEs, and other models where the loss is computed
/// from the input alone (e.g. reconstruction loss).
pub fn trainUnsupervised(
    comptime T: type,
    g: *ComputeGraph(T),
    loss_node: *Tensor(T),
    xs_batch: *Tensor(T),
    xs: *Tensor(T),
    n_epochs: usize,
    optimizer: anytype,
) !void {
    const n_elems = xs_batch.nElems();
    std.debug.assert(xs.nElems() % n_elems == 0);
    const n_batches = xs.nElems() / n_elems;
    for (0..n_epochs) |_| {
        for (0..n_batches) |b_idx| {
            @memcpy(xs_batch.data, xs.data[b_idx * n_elems ..][0..n_elems]);
            try g.run(loss_node);
            optimizer.step();
        }
    }
}

/// Inverted dropout: randomly zeroes elements during training and scales
/// surviving elements by `1/(1-p)` so expected values are preserved.
///
/// During inference (training=false), acts as identity.
///
/// Usage:
/// ```
/// var drop = Dropout(f32).init(0.1, 42);
/// const y = drop.forward(x, true);  // training mode
/// ```
pub fn Dropout(comptime T: type) type {
    return struct {
        const Self = @This();

        p: T, // drop probability
        rng: std.Random.DefaultPrng,

        pub fn init(p: T, seed: u64) Self {
            std.debug.assert(p >= 0 and p < 1);
            return .{
                .p = p,
                .rng = std.Random.DefaultPrng.init(seed),
            };
        }

        /// Apply dropout to tensor `x`.
        /// When `training` is true, randomly zeroes elements with probability `p`
        /// and scales survivors by `1/(1-p)`.
        /// When `training` is false, returns `x` unchanged.
        pub fn forward(self: *Self, x: *Tensor(T), training: bool) *Tensor(T) {
            if (!training or self.p == 0) return x;

            const alloc = x.alloc.?;
            const mask = Tensor(T).init(alloc, &x.ne) catch unreachable;
            const scale = 1.0 / (1.0 - self.p);
            var random = self.rng.random();

            for (mask.data) |*d| {
                const r: T = @floatCast(random.float(f64));
                d.* = if (r >= self.p) scale else 0;
            }
            // mask is a constant (no grad) — gradients flow through mul to x
            return x.mul(mask.repeatLike(x));
        }
    };
}

/// Rotary Position Embeddings (RoPE) — Su et al., 2021.
///
/// Encodes position information by rotating pairs of features in Q/K vectors.
/// Fully composed from existing ops: `matMul` (permutation), `mul`, `add`.
///
/// For each pair of features (x_{2i}, x_{2i+1}) at position m:
///   output_{2i}   = x_{2i} * cos(θ) - x_{2i+1} * sin(θ)
///   output_{2i+1} = x_{2i} * sin(θ) + x_{2i+1} * cos(θ)
/// where θ = m / base^(2i/d).
///
/// The rotation is expressed as: `x * cos_table + (x @ P) * sin_table`
/// where P is a permutation matrix that swaps and negates pairs.
///
/// Usage:
/// ```
/// var rope = try RoPE(f32, 64, 128).init(alloc);
/// const q_rotated = rope.forward(q, seq_len);
/// const k_rotated = rope.forward(k, seq_len);
/// ```
pub fn RoPE(comptime T: type, comptime d: usize, comptime max_seq_len: usize) type {
    if (d % 2 != 0)
        @compileError("RoPE: d must be even, got " ++ std.fmt.comptimePrint("{}", .{d}));

    return struct {
        const Self = @This();

        /// Pre-computed cos values [d, max_seq_len].
        cos_table: *Tensor(T),

        /// Pre-computed sin values [d, max_seq_len].
        sin_table: *Tensor(T),

        pub fn init(alloc: Alloc, base: T) !Self {
            var self: Self = undefined;

            // Build cos/sin frequency tables.
            // Half-split pairing: pair (i, i+d/2) shares the same frequency,
            // matching HuggingFace's LlamaRotaryEmbedding.
            self.cos_table = try Tensor(T).init(alloc, &.{ d, max_seq_len });
            self.sin_table = try Tensor(T).init(alloc, &.{ d, max_seq_len });

            for (0..max_seq_len) |pos| {
                for (0..d / 2) |i| {
                    const p: T = @floatFromInt(pos);
                    const dim: T = @floatFromInt(2 * i);
                    const dm: T = @floatFromInt(d);
                    const freq = p / std.math.pow(T, base, dim / dm);
                    const cos_val = @cos(freq);
                    const sin_val = @sin(freq);
                    self.cos_table.data[pos * d + i] = cos_val;
                    self.cos_table.data[pos * d + i + d / 2] = cos_val;
                    self.sin_table.data[pos * d + i] = sin_val;
                    self.sin_table.data[pos * d + i + d / 2] = sin_val;
                }
            }

            return self;
        }

        /// Pack cos/sin into a single [2*d, seq_len] tensor for the fused rope op.
        fn packCosSin(alloc: Alloc, cos: *Tensor(T), sin: *Tensor(T)) *Tensor(T) {
            const seq = cos.ne[1];
            const cs_buf = Tensor(T).init(alloc, &.{ 2 * d, seq }) catch unreachable;
            // Pack column by column: for each column, cos rows then sin rows.
            for (0..seq) |col| {
                @memcpy(cs_buf.data[col * 2 * d ..][0..d], cos.data[col * d ..][0..d]);
                @memcpy(cs_buf.data[col * 2 * d + d ..][0..d], sin.data[col * d ..][0..d]);
            }
            return cs_buf;
        }

        /// Prepare packed cos_sin [2*d, seq_len] for positions [0, seq_len).
        pub fn getCosSinPacked(self: *const Self, alloc: Alloc, seq_len: usize) *Tensor(T) {
            std.debug.assert(seq_len <= max_seq_len);
            const cos = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const sin = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const elems = d * seq_len;
            @memcpy(cos.data[0..elems], self.cos_table.data[0..elems]);
            @memcpy(sin.data[0..elems], self.sin_table.data[0..elems]);
            return packCosSin(alloc, cos, sin);
        }

        /// Prepare packed cos_sin [2*d, 1] for a single position.
        pub fn getCosSinPackedAtPos(self: *const Self, alloc: Alloc, pos: usize) *Tensor(T) {
            std.debug.assert(pos < max_seq_len);
            const cs_buf = Tensor(T).init(alloc, &.{ 2 * d, 1 }) catch unreachable;
            @memcpy(cs_buf.data[0..d], self.cos_table.data[pos * d ..][0..d]);
            @memcpy(cs_buf.data[d .. 2 * d], self.sin_table.data[pos * d ..][0..d]);
            return cs_buf;
        }

        /// Apply fused RoPE rotation: x * cos + rotate_half(x) * sin.
        pub fn apply(_: *const Self, x: *Tensor(T), cos_sin: *Tensor(T)) *Tensor(T) {
            return x.ropeRotate(cos_sin);
        }

        /// Apply rotary position embeddings to x [d, seq_len] from position 0.
        pub fn forward(self: *const Self, x: *Tensor(T), seq_len: usize) *Tensor(T) {
            std.debug.assert(x.ne[0] == d);
            std.debug.assert(x.ne[1] == seq_len);
            const cs = self.getCosSinPacked(x.alloc.?, seq_len);
            return self.apply(x, cs);
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const tac = testing.allocator;

test "dropout - inference mode is identity" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var drop = Dropout(f32).init(0.5, 42);

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4 });

    const y = drop.forward(x, false); // inference — identity
    try g.buildForward(y);
    g.compute();

    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, y.data);
}

test "dropout - training mode zeroes some elements" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var drop = Dropout(f32).init(0.5, 123);

    const x = try Tensor(f32).init(a, &.{100});
    for (x.data) |*d| d.* = 1.0;

    const y = drop.forward(x, true);
    try g.buildForward(y);
    g.compute();

    var n_zero: usize = 0;
    var n_scaled: usize = 0;
    for (y.data) |v| {
        if (v == 0) n_zero += 1;
        if (@abs(v - 2.0) < 1e-6) n_scaled += 1; // 1 * (1/0.5) = 2
    }

    // With p=0.5, roughly half should be zero and half scaled to 2
    try testing.expect(n_zero > 20 and n_zero < 80);
    try testing.expect(n_scaled > 20 and n_scaled < 80);
    try testing.expectEqual(@as(usize, 100), n_zero + n_scaled);
}

test "dropout - zero probability is identity" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var drop = Dropout(f32).init(0.0, 42);

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4 });

    const y = drop.forward(x, true);
    try testing.expectEqual(x, y); // should be same pointer (no-op)
}

test "dropout - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var drop = Dropout(f32).init(0.3, 42);

    const x = try Tensor(f32).init(a, &.{8});
    for (x.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i + 1));
    x.setParam();

    const y = drop.forward(x, true);
    const loss = y.sumAll();

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    // Gradients should be either 0 (dropped) or 1/(1-p) (scaled)
    const scale = 1.0 / (1.0 - @as(f32, 0.3));
    for (x.grad.?.data) |v| {
        try testing.expect(v == 0 or @abs(v - scale) < 1e-5);
    }
}

// -- RoPE tests --

test "rope - position 0 is identity" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);

    // At position 0, all angles are 0: cos=1, sin=0 → output = x * 1 + rotated * 0 = x
    const x = try Tensor(f32).init(a, &.{ 4, 1 });
    x.setData(&.{ 1, 2, 3, 4 });

    const y = rope.forward(x, 1);
    try g.buildForward(y);
    g.compute();

    for (x.data, y.data) |xv, yv| {
        try testing.expectApproxEqAbs(xv, yv, 1e-5);
    }
}

test "rope - different positions produce different outputs" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);

    // Same vector at two positions should differ
    const x = try Tensor(f32).init(a, &.{ 4, 2 });
    x.setData(&.{ 1, 2, 3, 4, 1, 2, 3, 4 }); // same features at pos 0 and 1

    const y = rope.forward(x, 2);
    try g.buildForward(y);
    g.compute();

    // Position 0 and position 1 should differ
    var any_diff = false;
    for (0..4) |feat| {
        if (@abs(y.data[0 * 4 + feat] - y.data[1 * 4 + feat]) > 1e-6) {
            any_diff = true;
            break;
        }
    }
    try testing.expect(any_diff);
}

test "rope - preserves norm (rotation is orthogonal)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);

    const x = try Tensor(f32).init(a, &.{ 4, 1 });
    x.setData(&.{ 1, 2, 3, 4 });

    const y = rope.forward(x, 1);
    try g.buildForward(y);
    g.compute();

    // Rotation preserves L2 norm
    var x_norm: f32 = 0;
    var y_norm: f32 = 0;
    for (x.data, y.data) |xv, yv| {
        x_norm += xv * xv;
        y_norm += yv * yv;
    }
    try testing.expectApproxEqAbs(@sqrt(x_norm), @sqrt(y_norm), 1e-4);
}

test "rope - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);

    const x = try Tensor(f32).init(a, &.{ 4, 2 });
    for (x.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i + 1)) * 0.5;
    x.setParam();

    const y = rope.forward(x, 2);
    const loss = y.sumAll();

    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    var has_nonzero = false;
    for (x.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
        if (v != 0) has_nonzero = true;
    }
    try testing.expect(has_nonzero);
}

// -- linear tests --

test "linear - matmul plus bias" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    // Use same shapes as graph matmul test: x=[2,3] @ w=[3,2] → [3,3]
    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const b = try Tensor(f32).init(a, &.{3});
    b.setData(&.{ 10, 20, 30 });

    const y = linear(f32, x, w, b);
    try g.buildForward(y);
    g.compute();

    // matmul gives {9,12,15, 19,26,33, 29,40,51}, bias [10,20,30] broadcasts per-row
    try testing.expectEqualSlices(f32, &.{ 19, 32, 45, 29, 46, 63, 39, 60, 81 }, y.data);
}

test "linear - no bias" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 2, 3, 4, 5, 6 });

    const y = linear(f32, x, w, null);
    try g.buildForward(y);
    g.compute();

    try testing.expectEqualSlices(f32, &.{ 9, 12, 15, 19, 26, 33, 29, 40, 51 }, y.data);
}

test "linear - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 1 });
    x.setData(&.{ 1, 2 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 0, 0, 1, 1, 1 });
    w.setParam();
    const b = try Tensor(f32).init(a, &.{3});
    b.setData(&.{ 0, 0, 0 });
    b.setParam();

    const loss = linear(f32, x, w, b).sumAll();
    try g.buildForward(loss);
    try g.buildBackward(false);
    _ = loss.grad.?.setAllScalar(1);
    g.compute();

    // w and b should have non-zero gradients
    var has_nonzero_w = false;
    for (w.grad.?.data) |v| if (v != 0) {
        has_nonzero_w = true;
        break;
    };
    try testing.expect(has_nonzero_w);

    var has_nonzero_b = false;
    for (b.grad.?.data) |v| if (v != 0) {
        has_nonzero_b = true;
        break;
    };
    try testing.expect(has_nonzero_b);
}

// -- init tests --

test "kaimingUniform - values within expected bounds" {
    const t = try Tensor(f32).init(tac, &.{ 16, 4 });
    defer t.deinit();

    kaimingUniform(f32, t, 42);

    const bound: f32 = @sqrt(6.0 / 4.0);
    for (t.data) |v| {
        try testing.expect(v >= -bound and v <= bound);
    }
}

test "uniform - values within range" {
    const t = try Tensor(f32).init(tac, &.{100});
    defer t.deinit();

    uniform(f32, t, -0.5, 0.5, 123);

    for (t.data) |v| {
        try testing.expect(v >= -0.5 and v < 0.5);
    }
}
