//! Lazy graph-building operations for tensors.
//!
//! Every op uses the tensor's stored `.alloc` field — callers never pass
//! an allocator. The allocator is set automatically when a tensor is created
//! via `Tensor.init`, `ComputeGraph.tensor`, etc.
//!
//! These methods build the computation graph without performing any math.
//! They allocate new tensor nodes via `catch unreachable` — this is
//! intentional: lazy ops are designed to be composed fluently
//! (e.g. `x.sub(y).sqr().mean(&.{1})`).

const std = @import("std");
const assert = std.debug.assert;
const Alloc = std.mem.Allocator;
const Op = @import("../op.zig").Op;
const indexlib = @import("../index.zig");
const max_dims = @import("../tensor.zig").max_dims;

pub fn Api(comptime Self: type, comptime T: type) type {
    return struct {
        /// Get the tensor's allocator. Panics if not set.
        inline fn a(self: *Self) Alloc {
            return self.alloc.?;
        }

        fn wrapIndexTensor(self: *Self, indices: anytype) *Self {
            const Idx = @TypeOf(indices.*.data[0]);
            const IndexTensor = indexlib.IndexTensor(Idx);
            const typed_indices: *const IndexTensor = indices;
            const vals = typed_indices.toUsizeOwned(a(self)) catch unreachable;
            defer a(self).free(vals);
            return Self.initIndexVectorCopy(a(self), vals) catch unreachable;
        }

        // ---------------------------------------------------------------
        // Internal helpers
        // ---------------------------------------------------------------

        pub fn view(self: *Self) *Self {
            const alloc = a(self);
            var t = Self.initHelper(alloc, &self.ne, self.data) catch unreachable;
            t.op = .view;
            t.src0 = self;
            t.src1 = null;
            t.grad = if (self.grad) |grad| grad.view() else null;
            return t;
        }

        pub fn copyTensorShape(self: *Self) *Self {
            return Self.initHelper(a(self), &self.ne, null) catch unreachable;
        }

        fn unaryOp(self: *Self, op: Op, inplace: bool) *Self {
            const alloc = a(self);
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) self.view() else self.copyTensorShape();
            res.op = op;
            res.grad = if (is_node) Self.initHelper(alloc, &self.ne, null) catch unreachable else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn binaryOp(self: *Self, other: *Self, op: Op, inplace: bool) *Self {
            const alloc = a(self);
            assert(self.isSameShape(other));
            const is_node: bool = !inplace and (self.grad != null or other.grad != null);
            const res: *Self = if (inplace) self.view() else self.copyTensorShape();
            res.op = op;
            res.grad = if (is_node) Self.initHelper(alloc, &self.ne, null) catch unreachable else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }

        // ---------------------------------------------------------------
        // Element-wise binary ops
        // ---------------------------------------------------------------

        /// Element-wise addition.
        pub fn add(self: *Self, other: *Self) *Self {
            assert(self.isSameShape(other));
            return binaryOp(self, other, .add, false);
        }

        pub fn addInplace(self: *Self, other: *Self) *Self {
            assert(self.isSameShape(other));
            return binaryOp(self, other, .add, true);
        }

        /// Element-wise subtraction. Decomposes to `add(self, neg(other))`.
        pub fn sub(self: *Self, other: *Self) *Self {
            return self.add(other.neg());
        }

        pub fn subInplace(self: *Self, other: *Self) *Self {
            return self.addInplace(other.neg());
        }

        /// Element-wise multiplication.
        pub fn mul(self: *Self, other: *Self) *Self {
            return binaryOp(self, other, .mul, false);
        }

        pub fn mulInplace(self: *Self, other: *Self) *Self {
            return binaryOp(self, other, .mul, true);
        }

        /// Element-wise division. Decomposes to `mul(self, recip(other))`.
        pub fn div(self: *Self, other: *Self) *Self {
            return self.mul(other.recip());
        }

        pub fn divInplace(self: *Self, other: *Self) *Self {
            return self.mulInplace(other.recip());
        }

        /// Element-wise square. Decomposes to `mul(self, self)`.
        pub fn sqr(self: *Self) *Self {
            return self.mul(self);
        }

        // ---------------------------------------------------------------
        // Element-wise unary ops
        // ---------------------------------------------------------------

        pub fn sqrt(self: *Self) *Self {
            return unaryOp(self, .sqrt, false);
        }
        pub fn sqrtInplace(self: *Self) *Self {
            return unaryOp(self, .sqrt, true);
        }
        pub fn recip(self: *Self) *Self {
            return unaryOp(self, .recip, false);
        }
        pub fn recipInplace(self: *Self) *Self {
            return unaryOp(self, .recip, true);
        }
        pub fn exp(self: *Self) *Self {
            return unaryOp(self, .exp, false);
        }
        pub fn expInplace(self: *Self) *Self {
            return unaryOp(self, .exp, true);
        }
        pub fn log(self: *Self) *Self {
            return unaryOp(self, .log, false);
        }
        pub fn logInplace(self: *Self) *Self {
            return unaryOp(self, .log, true);
        }
        pub fn abs(self: *Self) *Self {
            return unaryOp(self, .abs, false);
        }
        pub fn absInplace(self: *Self) *Self {
            return unaryOp(self, .abs, true);
        }
        pub fn sgn(self: *Self) *Self {
            return unaryOp(self, .sgn, false);
        }
        pub fn sgnInplace(self: *Self) *Self {
            return unaryOp(self, .sgn, true);
        }
        pub fn neg(self: *Self) *Self {
            return unaryOp(self, .neg, false);
        }
        pub fn negInplace(self: *Self) *Self {
            return unaryOp(self, .neg, true);
        }
        pub fn step(self: *Self) *Self {
            return unaryOp(self, .step, false);
        }
        pub fn stepInplace(self: *Self) *Self {
            return unaryOp(self, .step, true);
        }
        pub fn gelu(self: *Self) *Self {
            return unaryOp(self, .gelu, false);
        }
        pub fn geluInplace(self: *Self) *Self {
            return unaryOp(self, .gelu, true);
        }

        /// Element-wise ReLU: max(0, x). Decomposes to `mul(self, step(self))`.
        pub fn relu(self: *Self) *Self {
            const mask = self.step();
            mask.grad = null;
            return self.mul(mask);
        }

        // ---------------------------------------------------------------
        // Reductions
        // ---------------------------------------------------------------

        /// Sum all elements into a scalar.
        pub fn sumAll(self: *Self) *Self {
            return sum(self, &.{1});
        }
        /// Max-reduce all elements into a scalar.
        pub fn maxAll(self: *Self) *Self {
            return max(self, &.{1});
        }

        /// Sum (reduce) elements into the given target shape.
        pub fn sum(self: *Self, ne: []const usize) *Self {
            const alloc = a(self);
            assert(ne.len <= max_dims);
            assert(self.canSumToShape(ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .sum;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            return res;
        }

        /// Max-reduce elements into the given target shape.
        pub fn max(self: *Self, ne: []const usize) *Self {
            const alloc = a(self);
            assert(ne.len <= max_dims);
            assert(self.canSumToShape(ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .max;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            return res;
        }

        pub fn maxInto(self: *Self, other: *Self) *Self {
            const alloc = a(self);
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view();
            res.op = .max;
            res.grad = if (is_node) Self.initHelper(alloc, &res.ne, null) catch unreachable else null;
            res.src0 = self;
            return res;
        }

        /// Sum into another tensor's shape.
        pub fn sumInto(self: *Self, other: *Self) *Self {
            const alloc = a(self);
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view();
            res.op = .sum;
            res.grad = if (is_node) Self.initHelper(alloc, &res.ne, null) catch unreachable else null;
            res.src0 = self;
            return res;
        }

        /// Mean (reduce). Decomposes to `sum * (1/count)`.
        pub fn mean(self: *Self, ne: []const usize) *Self {
            const alloc = a(self);
            const s = self.sum(ne);
            const count: T = @floatFromInt(self.nElems() / s.nElems());
            const inv_count = Self.initScalar(alloc, 1.0 / count) catch unreachable;
            return s.mul(inv_count.repeatLike(s));
        }

        pub fn meanInto(self: *Self, other: *Self) *Self {
            const alloc = a(self);
            const s = self.sumInto(other);
            const count: T = @floatFromInt(self.nElems() / s.nElems());
            const inv_count = Self.initScalar(alloc, 1.0 / count) catch unreachable;
            return s.mul(inv_count.repeatLike(s));
        }

        // ---------------------------------------------------------------
        // Broadcasting
        // ---------------------------------------------------------------

        /// Broadcast (repeat) this tensor to the given shape.
        pub fn repeat(self: *Self, ne: []usize) *Self {
            const alloc = a(self);
            assert(ne.len <= max_dims);
            assert(self.canRepeatToShape(ne));
            const is_node: bool = self.grad != null;
            if (self.hasShape(ne) and !is_node) return self;
            const res = Self.init(alloc, ne) catch unreachable;
            res.op = .repeat;
            res.grad = if (self.grad) |grad| grad.repeat(ne) else null;
            res.src0 = self;
            return res;
        }

        /// Broadcast this tensor to match `other`'s shape.
        pub fn repeatLike(self: *Self, other: *Self) *Self {
            return repeat(self, &other.ne);
        }

        // ---------------------------------------------------------------
        // Composite ops
        // ---------------------------------------------------------------

        /// Numerically stable softmax.
        pub fn softmax(self: *Self, ne: []const usize) *Self {
            const max_t = self.max(ne);
            const shifted = self.sub(max_t.repeatLike(self));
            const exps = shifted.exp();
            const denom = exps.sum(ne);
            return exps.div(denom.repeatLike(exps));
        }

        /// Numerically stable log-softmax.
        pub fn logSoftmax(self: *Self, ne: []const usize) *Self {
            const max_t = self.max(ne);
            const shifted = self.sub(max_t.repeatLike(self));
            const exps = shifted.exp();
            const log_norm = exps.sum(ne).log();
            return shifted.sub(log_norm.repeatLike(shifted));
        }

        /// Layer normalization: `(x - mean) / sqrt(var + eps)`.
        pub fn layerNorm(self: *Self, ne: []const usize, eps: T) *Self {
            const alloc = a(self);
            const mu = self.mean(ne);
            const centered = self.sub(mu.repeatLike(self));
            const variance = centered.sqr().mean(ne);
            const eps_t = Self.initScalar(alloc, eps) catch unreachable;
            const std_inv = variance.add(eps_t.repeatLike(variance)).sqrt().recip();
            return centered.mul(std_inv.repeatLike(centered));
        }

        /// Add a lower-dimensional bias tensor, auto-broadcasting to self's shape.
        pub fn addBias(self: *Self, bias: *Self) *Self {
            return self.add(bias.repeatLike(self));
        }

        pub fn scale(self: *Self, other: *Self) *Self {
            assert(other.isScalar());
            return self.mul(other.repeatLike(self));
        }

        pub fn scaleInplace(self: *Self, other: *Self) *Self {
            assert(other.isScalar());
            return self.mulInplace(other.repeatLike(self));
        }

        /// Scale every element by a scalar value.
        pub fn scaleByVal(self: *Self, val: T) *Self {
            const alloc = a(self);
            const s = Self.initScalar(alloc, val) catch unreachable;
            return self.mul(s.repeatLike(self));
        }

        // ---------------------------------------------------------------
        // Matrix multiplication
        // ---------------------------------------------------------------

        /// Full matmul with explicit transpose flags.
        pub fn matMul(self: *Self, trans_self: bool, other: *Self, trans_other: bool) *Self {
            const alloc = a(self);
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
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = other;
            res.assertValidMatMulDims(self, trans_self, other, trans_other);
            return res;
        }

        /// `self @ other` (no transpose).
        pub fn mm(self: *Self, other: *Self) *Self {
            return self.matMul(false, other, false);
        }
        /// `self^T @ other`.
        pub fn Tmm(self: *Self, other: *Self) *Self {
            return self.matMul(true, other, false);
        }
        /// `self @ other^T`.
        pub fn mmT(self: *Self, other: *Self) *Self {
            return self.matMul(false, other, true);
        }
        /// `self^T @ other^T`.
        pub fn TmmT(self: *Self, other: *Self) *Self {
            return self.matMul(true, other, true);
        }

        // ---------------------------------------------------------------
        // Shape ops
        // ---------------------------------------------------------------

        pub fn reshapeLike(self: *Self, other: *Self) *Self {
            const alloc = a(self);
            assert(self.isContiguous());
            assert(other.isContiguous());
            assert(self.nElems() == other.nElems());
            const is_node = (self.grad != null or other.grad != null);
            const res = Self.initHelper(alloc, other.ne[0..other.n_dims], self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            return res;
        }

        pub fn reshape(self: *Self, ne: []const usize) *Self {
            const alloc = a(self);
            assert(self.isContiguous());
            const ne_prod = blk: {
                var prod: usize = 1;
                for (ne) |item| prod *= item;
                break :blk prod;
            };
            assert(self.nElems() == ne_prod);
            const is_node = self.grad != null;
            const res = Self.initHelper(alloc, ne, self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            return res;
        }

        /// Transpose the first two dimensions.
        pub fn transpose(self: *Self) *Self {
            const alloc = a(self);
            const is_node = self.grad != null;
            const out_ne = [max_dims]usize{ self.ne[1], self.ne[0], self.ne[2], self.ne[3] };
            const res = Self.init(alloc, out_ne[0..self.n_dims]) catch unreachable;
            res.op = .transpose;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            return res;
        }

        // ---------------------------------------------------------------
        // Index ops
        // ---------------------------------------------------------------

        pub fn gatherRows(self: *Self, indices: *Self) *Self {
            const alloc = a(self);
            assert(self.isMatrix());
            assert(indices.isVector());
            assert(indices.hasIndexBuffer() or indices.data.len == indices.ne[0]);
            const is_node = self.grad != null;
            const res = Self.init(alloc, &.{ self.ne[0], indices.ne[0] }) catch unreachable;
            res.op = .gather_rows;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = indices;
            return res;
        }

        pub fn pickRows(self: *Self, indices: *Self) *Self {
            const alloc = a(self);
            assert(self.isMatrix());
            assert(indices.isVector());
            assert(indices.hasIndexBuffer() or indices.data.len == indices.ne[0]);
            assert(indices.ne[0] == self.ne[1]);
            const is_node = self.grad != null;
            const res = Self.init(alloc, &.{indices.ne[0]}) catch unreachable;
            res.op = .pick_rows;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = indices;
            return res;
        }

        pub fn scatterAddRows(self: *Self, indices: *Self, updates: *Self) *Self {
            const alloc = a(self);
            assert(self.isMatrix());
            assert(indices.isVector());
            assert(indices.hasIndexBuffer() or indices.data.len == indices.ne[0]);
            assert(updates.isMatrix());
            assert(updates.ne[0] == self.ne[0]);
            assert(updates.ne[1] == indices.ne[0]);
            const is_node = updates.grad != null;
            const res = Self.init(alloc, self.ne[0..self.n_dims]) catch unreachable;
            res.op = .scatter_add_rows;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = updates;
            res.src1 = indices;
            return res;
        }

        pub fn scatterAddPicks(self: *Self, indices: *Self, updates: *Self) *Self {
            const alloc = a(self);
            assert(self.isMatrix());
            assert(indices.isVector());
            assert(indices.hasIndexBuffer() or indices.data.len == indices.ne[0]);
            assert(updates.isVector());
            assert(indices.ne[0] == self.ne[1]);
            assert(updates.ne[0] == self.ne[1]);
            const is_node = updates.grad != null;
            const res = Self.init(alloc, self.ne[0..self.n_dims]) catch unreachable;
            res.op = .scatter_add_picks;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = updates;
            res.src1 = indices;
            return res;
        }

        pub fn gatherRowsIdx(self: *Self, indices: anytype) *Self {
            const idx_tensor = wrapIndexTensor(self, indices);
            return self.gatherRows(idx_tensor);
        }

        pub fn pickRowsIdx(self: *Self, indices: anytype) *Self {
            const idx_tensor = wrapIndexTensor(self, indices);
            return self.pickRows(idx_tensor);
        }
    };
}
