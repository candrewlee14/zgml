//! Compile-time shape-tracked tensor wrapper.
//!
//! `Shaped(T, .{cols, rows, ...})` wraps a runtime `Tensor(T)` but carries its
//! shape in the type system. Shape mismatches (wrong matmul dims, incompatible
//! broadcast, etc.) become compile errors instead of runtime panics.
//!
//! ```
//! const S = @import("shaped.zig").Shaped;
//! const W = try S(f32, .{4, 3}).init(alloc);
//! const x = try S(f32, .{4, 1}).init(alloc);
//! const y = W.matMul(alloc, true, x, false);  // → Shaped(f32, .{3, 1})
//! // const bad = W.matMul(alloc, false, x, false);  // compile error!
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;
const tensorlib = @import("tensor.zig");
const Tensor = tensorlib.Tensor;
pub const max_dims = tensorlib.max_dims;

// ---------------------------------------------------------------------------
// Comptime shape helpers
// ---------------------------------------------------------------------------

/// Normalize a shape tuple (e.g. `.{3, 4}`) into a full `[max_dims]usize` array
/// padded with 1s.
pub fn normalizeShape(comptime ne: anytype) [max_dims]usize {
    comptime {
        var result = [_]usize{1} ** max_dims;
        const fields = @typeInfo(@TypeOf(ne)).@"struct".fields;
        for (fields, 0..) |f, i| {
            result[i] = @field(ne, f.name);
        }
        return result;
    }
}

fn nElemsOf(comptime s: [max_dims]usize) usize {
    comptime {
        var n: usize = 1;
        for (s) |d| n *= d;
        return n;
    }
}

fn transposeShape(comptime s: [max_dims]usize) [max_dims]usize {
    return .{ s[1], s[0], s[2], s[3] };
}

fn matMulOutputShape(
    comptime a: [max_dims]usize,
    comptime trans_a: bool,
    comptime b: [max_dims]usize,
    comptime trans_b: bool,
) [max_dims]usize {
    comptime {
        const a_inner = if (trans_a) a[1] else a[0];
        const b_inner = if (trans_b) b[0] else b[1];
        if (a_inner != b_inner)
            @compileError("matMul: inner dimensions don't match");
        if (a[2] != b[2]) @compileError("matMul: batch dimension mismatch");
        if (a[3] != b[3]) @compileError("matMul: channel dimension mismatch");

        const out_cols = if (trans_b) b[1] else b[0];
        const out_rows = if (trans_a) a[0] else a[1];
        return .{ out_cols, out_rows, a[2], a[3] };
    }
}

/// Number of dimensions (ignoring trailing 1s).
fn ndims(comptime s: [max_dims]usize) u8 {
    comptime {
        var n: u8 = max_dims;
        while (n > 1 and s[n - 1] == 1) n -= 1;
        return n;
    }
}

// ---------------------------------------------------------------------------
// Shaped tensor
// ---------------------------------------------------------------------------

/// Convenience alias: `Shaped(f32, .{3, 4})` is a compile-time-tracked 3x4 f32 tensor.
pub fn Shaped(comptime T: type, comptime dims: anytype) type {
    return ShapedTensor(T, normalizeShape(dims));
}

/// A tensor whose shape is part of the type. Wraps `*Tensor(T)` and delegates
/// all compute, but validates shape compatibility at compile time.
pub fn ShapedTensor(comptime T: type, comptime shape: [max_dims]usize) type {
    return struct {
        const Self = @This();

        pub const element_type = T;
        pub const static_shape = shape;
        pub const static_ndims = ndims(shape);
        pub const static_nelems = nElemsOf(shape);

        inner: *Tensor(T),

        // -- Construction --------------------------------------------------

        /// Allocate a new zero-initialized shaped tensor.
        pub fn init(alloc: Alloc) Alloc.Error!Self {
            return .{ .inner = try Tensor(T).init(alloc, &shape) };
        }

        /// Create a shaped tensor filled with evenly spaced values.
        pub fn initLinspace(alloc: Alloc, start: T, end: T) Alloc.Error!Self {
            return .{ .inner = try Tensor(T).initLinspace(alloc, &shape, start, end) };
        }

        /// Wrap an existing runtime tensor (asserts shape match in debug builds).
        pub fn fromTensor(t: *Tensor(T)) Self {
            std.debug.assert(t.hasShape(&shape));
            return .{ .inner = t };
        }

        /// Unwrap to the underlying runtime tensor.
        pub fn tensor(self: Self) *Tensor(T) {
            return self.inner;
        }

        /// Set all elements to `val`.
        pub fn setAllScalar(self: Self, val: T) Self {
            _ = self.inner.setAllScalar(val);
            return self;
        }

        /// Set data from a slice.
        pub fn setData(self: Self, data: []const T) Self {
            self.inner.setData(data);
            return self;
        }

        /// Mark as a learnable parameter (allocates gradient).
        pub fn setParam(self: Self, alloc: Alloc) Self {
            self.inner.setParam(alloc);
            return self;
        }

        /// Free this tensor.
        pub fn deinit(self: Self, alloc: Alloc) void {
            self.inner.deinit(alloc);
        }

        // -- Shape-preserving elementwise ops ------------------------------

        /// Element-wise addition. Both operands must have the same (static) shape.
        pub fn add(self: Self, alloc: Alloc, other: Self) Self {
            return .{ .inner = self.inner.add(alloc, other.inner) };
        }

        /// Element-wise multiplication.
        pub fn mul(self: Self, alloc: Alloc, other: Self) Self {
            return .{ .inner = self.inner.mul(alloc, other.inner) };
        }

        pub fn neg(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.neg(alloc) };
        }

        pub fn sqrt(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.sqrt(alloc) };
        }

        pub fn abs(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.abs(alloc) };
        }

        pub fn recip(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.recip(alloc) };
        }

        pub fn exp(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.exp(alloc) };
        }

        pub fn log(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.log(alloc) };
        }

        pub fn sgn(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.sgn(alloc) };
        }

        pub fn step(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.step(alloc) };
        }

        pub fn gelu(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.gelu(alloc) };
        }

        // -- Sugar (decomposed, shape-preserving) --------------------------

        pub fn sub(self: Self, alloc: Alloc, other: Self) Self {
            return .{ .inner = self.inner.sub(alloc, other.inner) };
        }

        pub fn div(self: Self, alloc: Alloc, other: Self) Self {
            return .{ .inner = self.inner.div(alloc, other.inner) };
        }

        pub fn sqr(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.sqr(alloc) };
        }

        pub fn relu(self: Self, alloc: Alloc) Self {
            return .{ .inner = self.inner.relu(alloc) };
        }

        // -- Shape-changing ops --------------------------------------------

        /// Matrix multiply with compile-time shape inference.
        ///
        /// ```
        /// const W = Shaped(f32, .{4, 3});  // 4 cols, 3 rows
        /// const x = Shaped(f32, .{4, 1});  // 4 cols, 1 row
        /// const y = W.matMul(alloc, true, x, false);  // → Shaped(f32, .{3, 1})
        /// ```
        pub fn matMul(
            self: Self,
            alloc: Alloc,
            comptime trans_self: bool,
            other: anytype,
            comptime trans_other: bool,
        ) ShapedTensor(T, matMulOutputShape(shape, trans_self, @TypeOf(other).static_shape, trans_other)) {
            return .{ .inner = self.inner.matMul(alloc, trans_self, other.inner, trans_other) };
        }

        /// Sum all elements to a scalar.
        pub fn sumAll(self: Self, alloc: Alloc) ShapedTensor(T, normalizeShape(.{1})) {
            return .{ .inner = self.inner.sumAll(alloc) };
        }

        /// Max-reduce all elements to a scalar.
        pub fn maxAll(self: Self, alloc: Alloc) ShapedTensor(T, normalizeShape(.{1})) {
            return .{ .inner = self.inner.maxAll(alloc) };
        }

        /// Sum (reduce) to a target shape. Each dim of target must divide self's dim.
        pub fn sum(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, normalizeShape(target)) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("sum: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.sum(alloc, &ne) };
        }

        /// Mean (reduce) to a target shape.
        pub fn mean(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, normalizeShape(target)) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("mean: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.mean(alloc, &ne) };
        }

        /// Max-reduce to a target shape.
        pub fn max(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, normalizeShape(target)) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("max: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.max(alloc, &ne) };
        }

        pub fn softmax(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, shape) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("softmax: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.softmax(alloc, &ne) };
        }

        pub fn logSoftmax(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, shape) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("logSoftmax: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.logSoftmax(alloc, &ne) };
        }

        /// Layer normalization: `(x - mean) / sqrt(var + eps)`.
        /// Normalizes over dimensions reduced to `target`.
        pub fn layerNorm(self: Self, alloc: Alloc, comptime target: anytype, eps: T) Self {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (s % t != 0) @compileError("layerNorm: target shape dimension does not divide source");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.layerNorm(alloc, &ne, eps) };
        }

        /// Broadcast to a larger shape. Each dim of self must divide target's dim.
        pub fn repeat(self: Self, alloc: Alloc, comptime target: anytype) ShapedTensor(T, normalizeShape(target)) {
            const target_shape = comptime normalizeShape(target);
            comptime {
                for (shape, target_shape) |s, t| {
                    if (t % s != 0) @compileError("repeat: source dimension does not divide target");
                }
            }
            var ne = target_shape;
            return .{ .inner = self.inner.repeat(alloc, &ne) };
        }

        /// Broadcast to match another shaped tensor's shape.
        pub fn repeatLike(self: Self, alloc: Alloc, comptime Other: type) ShapedTensor(T, Other.static_shape) {
            return self.repeat(alloc, Other.static_shape);
        }

        /// Reshape to a new shape with the same total elements.
        pub fn reshape(self: Self, alloc: Alloc, comptime new_dims: anytype) ShapedTensor(T, normalizeShape(new_dims)) {
            const new_shape = comptime normalizeShape(new_dims);
            comptime {
                if (nElemsOf(shape) != nElemsOf(new_shape))
                    @compileError("reshape: element count mismatch");
            }
            var ne = new_shape;
            return .{ .inner = self.inner.reshape(alloc, &ne) };
        }

        /// Transpose the first two dimensions.
        pub fn transpose(self: Self, alloc: Alloc) ShapedTensor(T, transposeShape(shape)) {
            return .{ .inner = self.inner.transpose(alloc) };
        }

        // -- Fused elementwise operations ----------------------------------
        //
        // Apply a user-provided function in a single pass over memory.
        // No intermediate tensors, no graph nodes — just one loop,
        // auto-vectorized by LLVM in release builds.

        /// Apply a unary function element-wise: `dst[i] = f(self[i])`.
        pub fn map(self: Self, alloc: Alloc, comptime f: fn (T) T) Alloc.Error!Self {
            return .{ .inner = try self.inner.map(alloc, f) };
        }

        /// Apply a binary function element-wise: `dst[i] = f(self[i], other[i])`.
        pub fn map2(self: Self, alloc: Alloc, other: Self, comptime f: fn (T, T) T) Alloc.Error!Self {
            return .{ .inner = try self.inner.map2(alloc, other.inner, f) };
        }

        // -- High-level composite operations --------------------------------

        /// Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`.
        ///
        /// Q, K, V must have shape [d_k, seq_len]. The output has the same shape as V.
        /// d_k (the key dimension) is taken from Q's first dimension and used for scaling.
        ///
        /// All dimensions are validated at compile time.
        pub fn attention(
            q: Self,
            alloc: Alloc,
            k: anytype,
            v: anytype,
        ) ShapedTensor(T, matMulOutputShape(
            matMulOutputShape(shape, false, @TypeOf(k).static_shape, true),
            false,
            @TypeOf(v).static_shape,
            false,
        )) {
            const d_k: T = @floatFromInt(shape[0]);
            // Q @ K^T → [seq_q, seq_k]
            const scores = q.matMul(alloc, false, k, true);
            // Scale by 1/sqrt(d_k)
            const ScoresType = @TypeOf(scores);
            const scale_val = 1.0 / @sqrt(d_k);
            const scale_tensor = ScoresType.fromTensor(
                (Tensor(T).initScalar(alloc, scale_val) catch unreachable)
                    .repeatLike(alloc, scores.inner),
            );
            const scaled = scores.mul(alloc, scale_tensor);
            // Softmax over the last key dimension (reduce to one column)
            const softmax_ne = comptime blk: {
                var ne = ScoresType.static_shape;
                ne[0] = 1;
                break :blk ne;
            };
            const weights = ShapedTensor(T, ScoresType.static_shape).fromTensor(
                scaled.inner.softmax(alloc, &softmax_ne),
            );
            // Weights @ V → same shape as V
            return weights.matMul(alloc, false, v, false);
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const tac = testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

test "shaped - init and basic ops" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const x = (try Shaped(f32, .{3}).init(a)).setAllScalar(2);
    const y = (try Shaped(f32, .{3}).init(a)).setAllScalar(3);
    const z = x.add(a, y);
    z.tensor().compute();

    try testing.expectEqualSlices(f32, &.{ 5, 5, 5 }, z.tensor().data);
}

test "shaped - matmul shape inference" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    // A: [2, 3] (2 cols, 3 rows), B: [3, 2] (3 cols, 2 rows)
    // A @ B: inner = A.ne[0](2) == B.ne[1](2) ✓. Output: [B.ne[0](3), A.ne[1](3)] = [3, 3]
    const A = (try Shaped(f32, .{ 2, 3 }).init(a)).setData(&.{ 1, 2, 3, 4, 5, 6 });
    const B = (try Shaped(f32, .{ 3, 2 }).init(a)).setData(&.{ 1, 2, 3, 4, 5, 6 });

    const C = A.matMul(a, false, B, false);

    // Verify the type carries the right shape
    comptime {
        const out_shape = @TypeOf(C).static_shape;
        std.debug.assert(out_shape[0] == 3); // cols
        std.debug.assert(out_shape[1] == 3); // rows
    }

    try g.buildForward(C.tensor());
    g.compute();

    // Same matmul we tested before: [2,3] @ [3,2] → [3,3]
    try testing.expectEqualSlices(f32, &.{ 9, 12, 15, 19, 26, 33, 29, 40, 51 }, C.tensor().data);
}

test "shaped - sum and repeat" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const x = (try Shaped(f32, .{ 2, 3 }).init(a)).setData(&.{ 1, 2, 3, 4, 5, 6 });

    // Sum to scalar
    const s = x.sumAll(a);
    comptime std.debug.assert(@TypeOf(s).static_shape[0] == 1);

    // Repeat scalar back
    const r = s.repeat(a, .{ 2, 3 });
    comptime std.debug.assert(@TypeOf(r).static_shape[0] == 2);
    comptime std.debug.assert(@TypeOf(r).static_shape[1] == 3);
}

test "shaped - softmax shape preserved" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = (try Shaped(f32, .{3}).init(a)).setData(&.{ 1, 2, 3 });
    const y = x.softmax(a, .{1});
    comptime std.debug.assert(@TypeOf(y).static_shape[0] == 3);

    try g.buildForward(y.tensor());
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 1.0), y.tensor().data[0] + y.tensor().data[1] + y.tensor().data[2], 1e-6);
}

test "shaped - reshape" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const x = try Shaped(f32, .{ 2, 3 }).init(a);
    const y = x.reshape(a, .{ 3, 2 });
    comptime {
        std.debug.assert(@TypeOf(y).static_nelems == 6);
        std.debug.assert(@TypeOf(y).static_shape[0] == 3);
        std.debug.assert(@TypeOf(y).static_shape[1] == 2);
    }
}

test "shaped - transpose" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const x = try Shaped(f32, .{ 2, 3 }).init(a);
    const y = x.transpose(a);
    comptime {
        std.debug.assert(@TypeOf(y).static_shape[0] == 3);
        std.debug.assert(@TypeOf(y).static_shape[1] == 2);
    }
}

test "shaped - backward through graph" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = (try Shaped(f32, .{3}).init(a)).setData(&.{ 1, 2, 3 }).setParam(a);
    const loss = x.sqr(a).sumAll(a);

    try g.buildForward(loss.tensor());
    try g.buildBackward(false);
    _ = loss.tensor().grad.?.setAllScalar(1);
    g.compute();

    // d/dx[sum(x^2)] = 2x
    try testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, x.tensor().grad.?.data);
}

test "map - unary fusion" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const x = (try Shaped(f32, .{4}).init(a)).setData(&.{ 1, 4, 9, 16 });

    // Fused: sqrt then negate — one pass, zero intermediates
    const y = try x.map(a, struct {
        fn f(xi: f32) f32 {
            return -std.math.sqrt(xi);
        }
    }.f);

    try testing.expectEqualSlices(f32, &.{ -1, -2, -3, -4 }, y.tensor().data);
}

test "map2 - binary fusion (sub+sqr = MSE kernel)" {
    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const a = arena.allocator();

    const pred = (try Shaped(f32, .{4}).init(a)).setData(&.{ 1, 2, 3, 4 });
    const target = (try Shaped(f32, .{4}).init(a)).setData(&.{ 1, 3, 3, 6 });

    // Fused (pred - target)^2 — one pass
    const sq_err = try pred.map2(a, target, struct {
        fn f(p: f32, t: f32) f32 {
            const d = p - t;
            return d * d;
        }
    }.f);

    // (0)^2, (−1)^2, (0)^2, (−2)^2 = 0, 1, 0, 4
    try testing.expectEqualSlices(f32, &.{ 0, 1, 0, 4 }, sq_err.tensor().data);
}

test "map2 - fused result feeds into graph" {
    // Verify a map2 result (eager) can feed into the lazy graph for autodiff
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = (try Shaped(f32, .{3}).init(a)).setData(&.{ 2, 3, 4 });

    // Eager: fused double
    const doubled = try x.map(a, struct {
        fn f(xi: f32) f32 {
            return xi * 2;
        }
    }.f);

    // Feed into lazy graph for sum
    const out = doubled.sumAll(a);
    try g.buildForward(out.tensor());
    g.compute();

    // 2+3+4 doubled = 18
    try testing.expectApproxEqAbs(@as(f32, 18.0), out.tensor().data[0], 1e-6);
}

test "shaped - attention" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    // Q, K, V all [d_k=2, seq=3]
    const Q = (try Shaped(f32, .{ 2, 3 }).init(a)).setData(&.{ 1, 0, 0, 1, 1, 1 });
    const K = (try Shaped(f32, .{ 2, 3 }).init(a)).setData(&.{ 1, 0, 0, 1, 1, 1 });
    const V = (try Shaped(f32, .{ 2, 3 }).init(a)).setData(&.{ 1, 2, 3, 4, 5, 6 });

    // attention output should be [d_k=2, seq=3] (same as V)
    const out = Q.attention(a, K, V);
    comptime {
        std.debug.assert(@TypeOf(out).static_shape[0] == 2);
        std.debug.assert(@TypeOf(out).static_shape[1] == 3);
    }

    try g.buildForward(out.tensor());
    g.compute();

    // Verify output is valid (weighted combination of V rows)
    // Each output value should be finite and reasonable
    for (out.tensor().data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
