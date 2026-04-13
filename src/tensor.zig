//! Core tensor type for the zgml machine learning library.
//!
//! A `Tensor` is a multi-dimensional array (up to 4 dimensions) that supports
//! lazy computation graph construction, forward evaluation, and reverse-mode
//! automatic differentiation.

const std = @import("std");
const assert = std.debug.assert;
const builtin = @import("builtin");
const Alloc = std.mem.Allocator;

const Op = @import("op.zig").Op;

/// Maximum number of dimensions a tensor can have.
pub const max_dims = 4;

/// Generic tensor parameterized on element type `T` (typically `f32` or `f64`).
///
/// Tensors form the nodes of a computation graph. "Lazy" operations (e.g. `add`,
/// `mul`, `matMul`) record the operation without computing it — the actual math
/// runs when `compute()` is called (usually via `ComputeGraph`).
///
/// Tensors that are marked as parameters (via `setParam`) track gradients for
/// use with optimizers.
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        // -- delegation to split-out implementations --
        const fwd = @import("tensor/forward.zig").Ops(Self, T);
        const bwd = @import("tensor/backward.zig").Ops(Self);

        /// Number of dimensions (1–4).
        n_dims: u8,
        /// Number of elements per axis, in [cols, rows, batch, channel] order.
        ne: [max_dims]usize,
        /// Memory stride per axis.
        strides: [max_dims]usize,
        /// The operation that produced this tensor (`.none` for user-created tensors).
        op: Op,
        /// Whether this tensor is a learnable parameter.
        is_param: bool,
        /// Gradient tensor, populated during backward pass. Only present for parameters
        /// and intermediate nodes that contribute to a parameter's gradient.
        grad: ?*Self,
        /// First source tensor (left operand or sole input).
        src0: ?*Self,
        /// Second source tensor (right operand for binary ops).
        src1: ?*Self,
        /// Debug label for visualization.
        name: ?[]const u8,
        /// Raw element data.
        data: []T,
        /// Whether this tensor owns (and should free) its data slice.
        data_owned: bool,

        // ---------------------------------------------------------------
        // Initialization
        // ---------------------------------------------------------------

        /// Create a tensor with the given shape.
        ///
        /// `ne` specifies the size of each dimension in [cols, rows, batch, channel]
        /// order. The number of dimensions is inferred from `ne.len` (max 4).
        pub fn init(alloc: Alloc, ne: []const usize) Alloc.Error!*Self {
            return try initHelper(alloc, ne, null);
        }

        /// Create a tensor filled with evenly spaced values in `[start, end)`.
        pub fn initLinspace(alloc: Alloc, ne: []const usize, start: T, end: T) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, ne);
            const denom: T = @floatFromInt(tensor.nElems());
            const diff = (end - start) / denom;
            for (tensor.data, 0..) |*d, i| {
                const num: T = @floatFromInt(i);
                d.* = start + diff * num;
            }
            return tensor;
        }

        /// Create a scalar (1-element) tensor with value `val`.
        pub fn initScalar(alloc: Alloc, val: T) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, &.{1});
            return tensor.setAllScalar(val);
        }

        /// Create a tensor filled with uniform random values in `[0, 1)`.
        pub fn initRand(alloc: Alloc, rng: *std.Random, ne: []const usize) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, ne);
            for (tensor.data) |*d| {
                d.* = rng.float(T);
            }
            return tensor;
        }

        fn initHelper(alloc: Alloc, ne: []const usize, data_buf: ?[]T) Alloc.Error!*Self {
            std.debug.assert(ne.len <= max_dims);
            const tensor: *Self = try alloc.create(Self);
            tensor.* = .{
                .n_dims = @truncate(ne.len),
                .ne = .{1} ** max_dims,
                .strides = .{0} ** max_dims,
                .op = .none,
                .is_param = false,
                .grad = null,
                .src0 = null,
                .src1 = null,
                .data = undefined,
                .name = null,
                .data_owned = data_buf == null,
            };
            for (ne, 0..) |shape_item, i| {
                tensor.ne[i] = shape_item;
            }
            tensor.strides[0] = 1;
            for (1..max_dims) |i| {
                tensor.strides[i] = tensor.strides[i - 1] * tensor.ne[i - 1];
            }
            tensor.data = if (data_buf) |d| d else try alloc.alloc(T, tensor.nElems());
            return tensor;
        }

        /// Free this tensor and its owned data.
        pub fn deinit(self: *Self, alloc: Alloc) void {
            if (self.data_owned) alloc.free(self.data);
            alloc.destroy(self);
        }

        /// Mark this tensor as a learnable parameter, allocating a gradient tensor.
        pub fn setParam(self: *Self, alloc: Alloc) void {
            self.is_param = true;
            assert(self.grad == null);
            self.grad = self.copyTensorShape(alloc);
        }

        // ---------------------------------------------------------------
        // Lazy graph-building operations
        //
        // These methods build the computation graph without performing any math.
        // They allocate new tensor nodes via `catch unreachable` — this is
        // intentional: lazy ops are designed to be infallible so they can be
        // composed fluently (e.g. `x.sub(a, y).sqr(a).mean(a, &.{1})`).
        //
        // When used with a ComputeGraph's arena allocator, allocation failure
        // is not expected. If you need fallible allocation, use the `init*`
        // family directly.
        // ---------------------------------------------------------------

        /// Create a view of this tensor's data (shared memory, no copy).
        pub fn view(self: *Self, alloc: Alloc) *Self {
            var t = Self.initHelper(alloc, &self.ne, self.data) catch unreachable;
            t.op = .view;
            t.src0 = self;
            t.src1 = null;
            t.grad = if (self.grad) |grad| grad.view(alloc) else null;
            return t;
        }

        /// Allocate a new tensor with the same shape but uninitialized data.
        pub fn copyTensorShape(self: *Self, alloc: Alloc) *Self {
            return Self.initHelper(alloc, &self.ne, null) catch unreachable;
        }

        fn unaryOp(self: *Self, alloc: Alloc, op: Op, inplace: bool) *Self {
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) self.view(alloc) else self.copyTensorShape(alloc);
            res.op = op;
            res.grad = if (is_node) self.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn binaryOp(self: *Self, alloc: Alloc, other: *Self, op: Op, inplace: bool) *Self {
            assert(self.isSameShape(other));
            const is_node: bool = !inplace and (self.grad != null or other.grad != null);
            const res: *Self = if (inplace) self.view(alloc) else self.copyTensorShape(alloc);
            res.op = op;
            res.grad = if (is_node) self.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }

        /// Duplicate this tensor.
        pub fn dup(self: *Self, alloc: Alloc) *Self {
            return self.unaryOp(alloc, .dup, false);
        }
        pub fn dupInplace(self: *Self, alloc: Alloc) *Self {
            return self.unaryOp(alloc, .dup, true);
        }

        fn addImpl(self: *Self, alloc: Alloc, other: *Self, inplace: bool) *Self {
            assert(self.isSameShape(other));
            return self.binaryOp(alloc, other, .add, inplace);
        }
        /// Element-wise addition.
        pub fn add(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.addImpl(alloc, other, false);
        }
        pub fn addInplace(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.addImpl(alloc, other, true);
        }

        fn subImpl(self: *Self, alloc: Alloc, other: *Self, inplace: bool) *Self {
            return self.binaryOp(alloc, other, .sub, inplace);
        }
        /// Element-wise subtraction.
        pub fn sub(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.subImpl(alloc, other, false);
        }
        pub fn subInplace(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.subImpl(alloc, other, true);
        }

        /// Element-wise multiplication.
        pub fn mul(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.binaryOp(alloc, other, .mul, false);
        }
        pub fn mulInplace(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.binaryOp(alloc, other, .mul, true);
        }

        /// Element-wise division.
        pub fn div(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.binaryOp(alloc, other, .div, false);
        }
        pub fn divInplace(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.binaryOp(alloc, other, .div, true);
        }

        /// Element-wise square.
        pub fn sqr(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sqr, false); }
        pub fn sqrInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sqr, true); }
        /// Element-wise square root.
        pub fn sqrt(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sqrt, false); }
        pub fn sqrtInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sqrt, true); }
        /// Element-wise reciprocal: 1/x.
        pub fn recip(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .recip, false); }
        pub fn recipInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .recip, true); }
        /// Element-wise absolute value.
        pub fn abs(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .abs, false); }
        pub fn absInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .abs, true); }
        /// Element-wise sign (-1, 0, or 1).
        pub fn sgn(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sgn, false); }
        pub fn sgnInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .sgn, true); }
        /// Element-wise negation.
        pub fn neg(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .neg, false); }
        pub fn negInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .neg, true); }
        /// Element-wise step function (1 if positive, 0 otherwise).
        pub fn step(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .step, false); }
        pub fn stepInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .step, true); }
        /// Element-wise ReLU: max(0, x).
        pub fn relu(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .relu, false); }
        pub fn reluInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .relu, true); }
        /// Element-wise GeLU approximation.
        pub fn gelu(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .gelu, false); }
        pub fn geluInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .gelu, true); }
        /// Row-wise normalization.
        pub fn norm(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .norm, false); }
        pub fn normInplace(self: *Self, alloc: Alloc) *Self { return self.unaryOp(alloc, .norm, true); }

        /// Sum all elements into a scalar.
        pub fn sumAll(self: *Self, alloc: Alloc) *Self {
            return sum(self, alloc, &.{1});
        }

        /// Sum (reduce) elements into the given target shape `ne`.
        pub fn sum(self: *Self, alloc: Alloc, ne: []const usize) *Self {
            assert(ne.len <= max_dims);
            assert(canSumToShape(self, ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .sum;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Sum into another tensor's shape (inverse of repeat).
        pub fn sumInto(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view(alloc);
            res.op = .sum;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Mean (reduce) elements into the given target shape.
        pub fn mean(self: *Self, alloc: Alloc, ne: []const usize) *Self {
            assert(ne.len <= max_dims);
            assert(canSumToShape(self, ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .mean;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn meanInto(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view(alloc);
            res.op = .mean;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn repeatInto(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(self.canRepeatTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view(alloc);
            res.op = .repeat;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Broadcast (repeat) this tensor to the given shape.
        /// Each dimension in `ne` must be a multiple of the corresponding dimension in self.
        pub fn repeat(self: *Self, alloc: Alloc, ne: []usize) *Self {
            assert(ne.len <= max_dims);
            assert(self.canRepeatToShape(ne));
            const is_node: bool = self.grad != null;
            if (self.hasShape(ne) and !is_node) return self;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .repeat;
            res.grad = if (self.grad) |grad| grad.repeat(alloc, ne) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Broadcast this tensor to match `other`'s shape.
        pub fn repeatLike(self: *Self, alloc: Alloc, other: *Self) *Self {
            return repeat(self, alloc, &other.ne);
        }

        /// Matrix multiplication with optional transposition of either operand.
        pub fn matMul(self: *Self, alloc: Alloc, trans_self: bool, other: *Self, trans_other: bool) *Self {
            assert(self.canMatMul(trans_self, other, trans_other));
            const is_node = self.grad != null or other.grad != null;
            assert(max_dims == 4);

            const out_ne: [max_dims]usize = lbl: {
                break :lbl if (!trans_self and !trans_other)
                    .{ other.ne[0], self.ne[1], self.ne[2], other.ne[3] }
                else if (trans_self and !trans_other)
                    .{ other.ne[0], self.ne[0], self.ne[2], other.ne[3] }
                else if (!trans_self and trans_other)
                    .{ other.ne[1], self.ne[1], self.ne[2], other.ne[3] }
                else
                    .{ other.ne[1], self.ne[0], self.ne[2], other.ne[3] };
            };
            const res = Self.init(alloc, out_ne[0..@min(self.n_dims, other.n_dims)]) catch unreachable;
            res.op = if (trans_self) if (trans_other) .matmul_t0t1 else .matmul_t0 else if (trans_other) .matmul_t1 else .matmul;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            res.assertValidMatMulDims(self, trans_self, other, trans_other);
            return res;
        }

        /// Scale by a scalar tensor.
        pub fn scale(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(other.isScalar());
            return self.binaryOp(alloc, other, .scale, false);
        }
        pub fn scaleInplace(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(other.isScalar());
            return self.binaryOp(alloc, other, .scale, true);
        }

        fn cpyImpl(self: *Self, alloc: Alloc, other: *Self, inplace: bool) *Self {
            assert(self.nElems() == other.nElems());
            const is_node = !inplace and (self.grad != null or other.grad != null);
            assert(!is_node);
            const res = if (is_node) other.copyTensorShape(alloc) else other.view(alloc);
            res.op = .cpy;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }
        pub fn cpyTo(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.cpyImpl(alloc, other, true);
        }
        pub fn cpyInplaceTo(self: *Self, alloc: Alloc, other: *Self) *Self {
            return self.cpyImpl(alloc, other, true);
        }

        /// Reshape this tensor's data to match `other`'s shape.
        pub fn reshapeLike(self: *Self, alloc: Alloc, other: *Self) *Self {
            assert(self.isContiguous());
            assert(other.isContiguous());
            assert(self.nElems() == other.nElems());
            const is_node = (self.grad != null or other.grad != null);
            assert(!is_node);
            const res = Self.initHelper(alloc, other.ne[0..other.n_dims], self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Reshape this tensor to a new shape (must have same total elements).
        pub fn reshape(self: *Self, alloc: Alloc, ne: []const usize) *Self {
            assert(self.isContiguous());
            const neProd = lbl: {
                var prod: usize = 1;
                for (ne) |item| prod *= item;
                break :lbl prod;
            };
            assert(self.nElems() == neProd);
            const is_node = self.grad != null;
            assert(!is_node);
            const res = Self.initHelper(alloc, ne, self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        /// Transpose the first two dimensions (swap rows and columns).
        pub fn transpose(self: *Self, alloc: Alloc) *Self {
            const is_node = self.grad != null;
            assert(!is_node);
            const res = self.view(alloc);
            res.ne[0] = self.ne[1];
            res.ne[1] = self.ne[0];
            res.strides[0] = self.strides[1];
            res.strides[1] = self.strides[0];
            res.op = .transpose;
            res.grad = if (is_node) res.copyTensorShape(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        // ---------------------------------------------------------------
        // Forward compute — delegated to tensor/forward.zig
        // ---------------------------------------------------------------

        pub const computeDup = fwd.computeDup;
        pub const computeAdd = fwd.computeAdd;
        pub const computeSub = fwd.computeSub;
        pub const computeMul = fwd.computeMul;
        pub const computeDiv = fwd.computeDiv;
        pub const computeMean = fwd.computeMean;
        pub const computeSum = fwd.computeSum;
        pub const computeRepeat = fwd.computeRepeat;
        pub const computeSqr = fwd.computeSqr;
        pub const computeSqrt = fwd.computeSqrt;
        pub const computeRecip = fwd.computeRecip;
        pub const computeAbs = fwd.computeAbs;
        pub const computeSgn = fwd.computeSgn;
        pub const computeNeg = fwd.computeNeg;
        pub const computeStep = fwd.computeStep;
        pub const computeRelu = fwd.computeRelu;
        pub const computeGelu = fwd.computeGelu;
        pub const computeNorm = fwd.computeNorm;
        pub const computeRMSNorm = fwd.computeRMSNorm;
        pub const computeMatMul = fwd.computeMatMul;
        pub const assertValidMatMulDims = fwd.assertValidMatMulDims;
        pub const compute = fwd.compute;

        // ---------------------------------------------------------------
        // Backward — delegated to tensor/backward.zig
        // ---------------------------------------------------------------

        pub const backward = bwd.backward;

        // ---------------------------------------------------------------
        // Utility methods
        // ---------------------------------------------------------------

        /// Overwrite this tensor's data from a slice. Length must match.
        pub fn setData(self: *Self, data: []const T) void {
            assert(@as(usize, data.len) == self.nElems());
            @memcpy(self.data, data);
        }

        /// Set every element to `val`. Returns self for chaining.
        pub fn setAllScalar(self: *Self, val: T) *Self {
            @memset(self.data, val);
            return self;
        }

        /// Total number of elements across all dimensions.
        pub fn nElems(self: *Self) usize {
            var res: usize = 1;
            for (&self.ne) |shape_item| res *= shape_item;
            return res;
        }

        /// True if this tensor is a single scalar value (all dims == 1).
        pub fn isScalar(self: *Self) bool {
            for (self.ne[0..]) |s| { if (s != 1) return false; }
            return true;
        }

        /// True if this is a 1-D vector (dims 1+ are all 1).
        pub fn isVector(self: *Self) bool {
            for (self.ne[1..]) |s| { if (s != 1) return false; }
            return true;
        }

        /// True if this is a 2-D matrix (dims 2+ are all 1).
        pub fn isMatrix(self: *Self) bool {
            for (self.ne[2..]) |s| { if (s != 1) return false; }
            return true;
        }

        /// True if self can be matrix-multiplied with `other`, accounting for transpositions.
        pub fn canMatMul(self: *Self, transSelf: bool, other: *Self, transOther: bool) bool {
            if (self.ne[3] != other.ne[3]) return false;
            if (self.ne[2] != other.ne[2]) return false;
            if (!transSelf and !transOther) return self.ne[0] == other.ne[1];
            if (transSelf and !transOther) return self.ne[1] == other.ne[1];
            if (!transSelf and transOther) return self.ne[0] == other.ne[0];
            return self.ne[1] == other.ne[0];
        }

        /// True if data is laid out contiguously in memory (no stride gaps).
        pub fn isContiguous(self: *Self) bool {
            if (self.strides[0] != 1) return false;
            for (1..max_dims) |i| {
                if (self.strides[i] != self.strides[i - 1] * self.ne[i - 1]) return false;
            }
            return true;
        }

        /// True if self can be broadcast (repeated) to match `other`'s shape.
        pub fn canRepeatTo(self: *Self, other: *Self) bool {
            return self.canRepeatToShape(&other.ne);
        }

        pub fn canRepeatToShape(self: *Self, other_ne: []const usize) bool {
            return shapeCanRepeatToShape(&self.ne, other_ne);
        }

        /// True if self can be reduced (summed) down to `other`'s shape.
        pub fn canSumTo(self: *Self, other: *Self) bool {
            return self.canSumToShape(&other.ne);
        }

        pub fn canSumToShape(self: *Self, other_ne: []const usize) bool {
            return shapeCanRepeatToShape(other_ne, &self.ne);
        }

        fn shapeCanRepeatToShape(self_ne: []const usize, other_ne: []const usize) bool {
            for (self_ne, 0..) |selfNe, i| {
                const otherNe = if (i < other_ne.len) other_ne[i] else 1;
                if (otherNe % selfNe != 0) return false;
            }
            return true;
        }

        /// Access the element at the given multi-dimensional coordinates.
        pub fn get(self: *Self, coords: []const usize) T {
            assert(coords.len == self.n_dims);
            var idx: usize = 0;
            for (coords, self.strides[0..coords.len]) |coord, stride| {
                idx += coord * stride;
            }
            return self.data[idx];
        }

        /// Print a debug summary of this tensor.
        pub fn print(self: *Self) void {
            std.debug.print("----{*}----\n", .{self});
            std.debug.print("shape: {any}\nstrides: {any}\ndata: {any}\n", .{ self.ne, self.strides, self.data });
            std.debug.print("--------------------------\n", .{});
        }

        /// True if self and other have compatible shapes for broadcasting (numpy-style).
        pub fn isBroadcastable(self: *Self, other: *Self) bool {
            for (self.ne, other.ne) |selfNe, otherNe| {
                if (selfNe != otherNe and selfNe != 1 and otherNe != 1) return false;
            }
            return true;
        }

        /// True if self and other have identical shapes.
        pub fn isSameShape(self: *Self, other: *Self) bool {
            return self.hasShape(&other.ne);
        }

        pub fn hasShape(self: *Self, other_ne: []const usize) bool {
            for (self.ne, 0..) |selfNe, i| {
                const otherNe = if (i < other_ne.len) other_ne[i] else 1;
                if (selfNe != otherNe) return false;
            }
            return true;
        }
    };
}

test {
    _ = @import("tensor/forward.zig");
    _ = @import("tensor/backward.zig");
}

const testing = std.testing;
const tac = std.testing.allocator;

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(Tensor(f32));
}

test "init" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit(tac);
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
        @memcpy(tensor.data, &data);
        try testing.expectEqual(@as(f32, 1), tensor.get(&.{ 0, 0 }));
        try testing.expectEqual(@as(f32, 3), tensor.get(&.{ 0, 1 }));
        try testing.expectEqual(@as(f32, 6), tensor.get(&.{ 1, 2 }));
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 5, 3, 2 });
        defer tensor.deinit(tac);
        try testing.expectEqual(@as(usize, 30), tensor.nElems());
    }
}

test "initLinspace" {
    {
        const t = try Tensor(f32).initLinspace(tac, &.{20}, 0, 20);
        defer t.deinit(tac);
        try testing.expectEqual(@as(usize, 20), t.nElems());
        for (t.data, 0..) |v, i| {
            try testing.expectEqual(@as(f32, @floatFromInt(i)), v);
        }
    }
    {
        const t = try Tensor(f32).initLinspace(tac, &.{20}, 0, 10);
        defer t.deinit(tac);
        try testing.expectEqual(@as(usize, 20), t.nElems());
        for (t.data, 0..) |v, i| {
            try testing.expectEqual(@as(f32, @floatFromInt(i)) * 0.5, v);
        }
    }
}

test "reshape" {
    const t = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t.deinit(tac);
    t.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const r = t.reshape(tac, &.{ 3, 2 });
    defer r.deinit(tac);

    try testing.expectEqual(@as(usize, 6), r.nElems());
    try testing.expectEqual(@as(u8, 2), r.n_dims);
    try testing.expectEqual(@as(usize, 3), r.ne[0]);
    try testing.expectEqual(@as(usize, 2), r.ne[1]);
    // data is shared (view), so values match
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, r.data);
}

test "isMatrix" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit(tac);
        try testing.expectEqual(true, tensor.isMatrix());
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3, 4 });
        defer tensor.deinit(tac);
        try testing.expectEqual(false, tensor.isMatrix());
    }
}

test "isSameShape" {
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer t1.deinit(tac);
        const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer t2.deinit(tac);
        try testing.expectEqual(false, t1.isSameShape(t2));
        try testing.expectEqual(true, t1.isSameShape(t1));
    }
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer t1.deinit(tac);
        const t2 = t1.view(tac);
        defer t2.deinit(tac);
        try testing.expectEqual(true, t1.isSameShape(t2));
    }
}

test "canRepeatTo" {
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer t1.deinit(tac);
        const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer t2.deinit(tac);
        try testing.expectEqual(false, t1.canRepeatTo(t2));
    }
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer t1.deinit(tac);
        const t2 = try Tensor(f32).init(tac, &.{ 4, 16, 9 });
        defer t2.deinit(tac);
        try testing.expectEqual(true, t1.canRepeatTo(t2));
    }
}

test "compute mean" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.mean(tac, &.{1});
    defer dst.deinit(tac);
    dst.computeMean(t1);

    try testing.expectApproxEqAbs(@as(f32, 3.5), dst.data[0], 1e-10);
}

test "compute matmul" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
    defer t2.deinit(tac);
    t2.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(tac, false, t2, false);
    defer dst.deinit(tac);
    dst.computeMatMul(t1, false, t2, false);

    try testing.expectEqualSlices(f32, &.{ 9, 12, 15, 19, 26, 33, 29, 40, 51 }, dst.data);
}

test "compute matmul_t0" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(tac, true, t1, false);
    defer dst.deinit(tac);
    dst.computeMatMul(t1, true, t1, false);

    try testing.expectEqualSlices(f32, &.{ 35, 44, 44, 56 }, dst.data);
}

test "compute matmul_t1 2D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(tac, false, t1, true);
    defer dst.deinit(tac);
    dst.computeMatMul(t1, false, t1, true);

    try testing.expectEqualSlices(f32, &.{ 5, 11, 17, 11, 25, 39, 17, 39, 61 }, dst.data);
}

test "compute matmul_t1 3D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 2, 2 });
    defer t1.deinit(tac);
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });

    const dst = t1.matMul(tac, false, t1, true);
    defer dst.deinit(tac);
    dst.computeMatMul(t1, false, t1, true);

    try testing.expectEqualSlices(f32, &.{ 5, 11, 11, 25, 61, 83, 83, 113 }, dst.data);
}
