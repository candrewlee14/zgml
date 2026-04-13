//! Neural network building blocks.
//!
//! Utility layers that decompose into primitive tensor ops. These don't
//! introduce new ops to the IR — they build graph subgraphs from existing
//! primitives, so backpropagation works automatically.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Alloc = std.mem.Allocator;

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

        /// Permutation matrix [d, d]: swaps pairs and negates first of each pair.
        /// P[2i, 2i+1] = -1,  P[2i+1, 2i] = 1,  all others 0.
        /// So (x @ P)[2i] = -x[2i+1],  (x @ P)[2i+1] = x[2i].
        perm: *Tensor(T),

        /// Pre-computed cos values [d, max_seq_len].
        cos_table: *Tensor(T),

        /// Pre-computed sin values [d, max_seq_len].
        sin_table: *Tensor(T),

        pub fn init(alloc: Alloc, base: T) !Self {
            var self: Self = undefined;

            // Build permutation matrix: P @ x rotates pairs
            self.perm = try Tensor(T).init(alloc, &.{ d, d });
            _ = self.perm.setAllScalar(0);
            for (0..d / 2) |i| {
                // P[2i, 2i+1] = -1  → maps x[2i+1] to position 2i, negated
                self.perm.data[(2 * i + 1) * d + (2 * i)] = -1;
                // P[2i+1, 2i] = 1   → maps x[2i] to position 2i+1
                self.perm.data[(2 * i) * d + (2 * i + 1)] = 1;
            }
            // perm is a constant — no grad, no setParam

            // Build cos/sin frequency tables
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
                    // Both features in a pair get the same cos/sin
                    self.cos_table.data[pos * d + 2 * i] = cos_val;
                    self.cos_table.data[pos * d + 2 * i + 1] = cos_val;
                    self.sin_table.data[pos * d + 2 * i] = sin_val;
                    self.sin_table.data[pos * d + 2 * i + 1] = sin_val;
                }
            }

            return self;
        }

        /// Apply rotary position embeddings to x [d, seq_len].
        ///
        /// Returns: x * cos + rotate_half(x) * sin
        /// where rotate_half(x) = x @ P (permutation matrix).
        ///
        /// Gradients flow through x via the mul and matMul ops automatically.
        pub fn forward(self: *const Self, x: *Tensor(T), seq_len: usize) *Tensor(T) {
            std.debug.assert(x.ne[0] == d);
            std.debug.assert(seq_len <= max_seq_len);
            std.debug.assert(x.ne[1] == seq_len);

            const alloc = x.alloc.?;

            // Slice cos/sin tables to actual seq_len
            const cos = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const sin = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const elems = d * seq_len;
            @memcpy(cos.data[0..elems], self.cos_table.data[0..elems]);
            @memcpy(sin.data[0..elems], self.sin_table.data[0..elems]);
            // cos, sin are constants — no grad

            // rotate_half(x) = x @ P
            const rotated = x.matMul(false, self.perm, false);

            // output = x * cos + rotated * sin
            return x.mul(cos.repeatLike(x)).add(rotated.mul(sin.repeatLike(rotated)));
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const tac = testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

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
