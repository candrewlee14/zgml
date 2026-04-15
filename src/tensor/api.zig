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

        fn scalarRepeatLike(self: *Self, val: T, other: *Self) *Self {
            const scalar = Self.initScalar(a(self), val) catch unreachable;
            _ = scalar.markInternalAux();
            const repeated = scalar.repeatLike(other);
            _ = repeated.markInternalAux();
            return repeated;
        }

        fn aux(node: *Self) *Self {
            return node.markInternalAux();
        }

        // ---------------------------------------------------------------
        // Internal helpers
        // ---------------------------------------------------------------

        pub fn view(self: *Self) *Self {
            const alloc = a(self);
            var t = Self.initHelper(alloc, self.ne[0..self.n_dims], self.data) catch unreachable;
            t.op = .view;
            t.src0 = self;
            t.src1 = null;
            t.strides = self.strides;
            t.storage_offset = self.storage_offset;
            t.grad = if (self.grad) |grad| grad.view() else null;
            return t;
        }

        /// Create a zero-initialized tensor with the same shape as self.
        /// Used for gradient accumulators — they must start at zero and
        /// stay zero (backward never writes to them, only reads).
        pub fn copyTensorShape(self: *Self) *Self {
            const t = Self.initHelper(a(self), self.ne[0..self.n_dims], null) catch unreachable;
            @memset(t.data, 0);
            return t;
        }

        fn structuralView(self: *Self, ne: []const usize, strides: [max_dims]usize, storage_offset: usize, op: Op) *Self {
            const alloc = a(self);
            const is_node = self.grad != null;
            const res = Self.initHelper(alloc, ne, self.data) catch unreachable;
            res.op = op;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            res.strides = strides;
            res.storage_offset = storage_offset;
            return res;
        }

        fn unaryOp(self: *Self, op: Op, inplace: bool) *Self {
            const alloc = a(self);
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) self.view() else self.copyTensorShape();
            res.op = op;
            res.grad = if (is_node) blk: {
                const g = Self.initHelper(alloc, &self.ne, null) catch unreachable;
                @memset(g.data, 0);
                break :blk g;
            } else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn binaryOp(self: *Self, other: *Self, op: Op, inplace: bool) *Self {
            const alloc = a(self);
            assert(self.isBroadcastable(other));
            const out_ndims = @max(self.n_dims, other.n_dims);
            var out_ne: [max_dims]usize = [_]usize{1} ** max_dims;
            for (0..out_ndims) |i| {
                const lhs = if (i < self.n_dims) self.ne[i] else 1;
                const rhs = if (i < other.n_dims) other.ne[i] else 1;
                out_ne[i] = @max(lhs, rhs);
            }
            const is_node: bool = !inplace and (self.grad != null or other.grad != null);
            const res: *Self = if (inplace) self.view() else Self.init(alloc, out_ne[0..out_ndims]) catch unreachable;
            res.op = op;
            res.grad = if (is_node) blk: {
                const g = Self.initHelper(alloc, out_ne[0..out_ndims], null) catch unreachable;
                @memset(g.data, 0);
                break :blk g;
            } else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }

        // ---------------------------------------------------------------
        // Element-wise binary ops
        // ---------------------------------------------------------------

        /// Element-wise addition.
        pub fn add(self: *Self, other: *Self) *Self {
            return binaryOp(self, other, .add, false);
        }

        pub fn addInplace(self: *Self, other: *Self) *Self {
            return binaryOp(self, other, .add, true);
        }

        /// Element-wise subtraction. Decomposes to `add(self, neg(other))`.
        pub fn sub(self: *Self, other: *Self) *Self {
            return self.add(aux(other.neg()));
        }

        /// Element-wise multiplication.
        pub fn mul(self: *Self, other: *Self) *Self {
            return binaryOp(self, other, .mul, false);
        }

        /// Element-wise division. Decomposes to `mul(self, recip(other))`.
        pub fn div(self: *Self, other: *Self) *Self {
            return self.mul(aux(other.recip()));
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
        pub fn recip(self: *Self) *Self {
            return unaryOp(self, .recip, false);
        }
        pub fn exp(self: *Self) *Self {
            return unaryOp(self, .exp, false);
        }
        pub fn log(self: *Self) *Self {
            return unaryOp(self, .log, false);
        }
        pub fn abs(self: *Self) *Self {
            return unaryOp(self, .abs, false);
        }
        pub fn sgn(self: *Self) *Self {
            return unaryOp(self, .sgn, false);
        }
        pub fn neg(self: *Self) *Self {
            return unaryOp(self, .neg, false);
        }
        pub fn step(self: *Self) *Self {
            return unaryOp(self, .step, false);
        }
        pub fn reluPrimitive(self: *Self) *Self {
            return unaryOp(self, .relu, false);
        }
        pub fn gelu(self: *Self) *Self {
            return unaryOp(self, .gelu, false);
        }
        /// Element-wise ReLU primitive.
        pub fn relu(self: *Self) *Self {
            return unaryOp(self, .relu, false);
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
            const s = aux(self.sum(ne));
            const count: T = @floatFromInt(self.nElems() / s.nElems());
            return s.mul(scalarRepeatLike(self, 1.0 / count, s));
        }

        pub fn meanInto(self: *Self, other: *Self) *Self {
            const s = aux(self.sumInto(other));
            const count: T = @floatFromInt(self.nElems() / s.nElems());
            return s.mul(scalarRepeatLike(self, 1.0 / count, s));
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

        pub fn broadcastTo(self: *Self, ne: []const usize) *Self {
            assert(ne.len <= max_dims);
            assert(self.canRepeatToShape(ne));

            var out_strides = self.strides;
            for (0..ne.len) |i| {
                const src_dim = if (i < self.n_dims) self.ne[i] else 1;
                if (src_dim == ne[i]) {
                    if (i >= self.n_dims) out_strides[i] = 0;
                } else {
                    assert(src_dim == 1);
                    out_strides[i] = 0;
                }
            }
            var i = ne.len;
            while (i < max_dims) : (i += 1) out_strides[i] = 0;
            return structuralView(self, ne, out_strides, self.storage_offset, .broadcast_to);
        }

        // ---------------------------------------------------------------
        // Composite ops
        // ---------------------------------------------------------------

        /// Numerically stable softmax.
        pub fn softmax(self: *Self, ne: []const usize) *Self {
            const max_t = aux(self.max(ne));
            const rep_max = aux(max_t.repeatLike(self));
            const shifted = aux(self.sub(rep_max));
            const exps = aux(shifted.exp());
            const denom = aux(exps.sum(ne));
            const rep_denom = aux(denom.repeatLike(exps));
            return exps.div(rep_denom);
        }

        /// Numerically stable log-softmax.
        pub fn logSoftmax(self: *Self, ne: []const usize) *Self {
            const max_t = aux(self.max(ne));
            const rep_max = aux(max_t.repeatLike(self));
            const shifted = aux(self.sub(rep_max));
            const exps = aux(shifted.exp());
            const sum_t = aux(exps.sum(ne));
            const log_norm = aux(sum_t.log());
            const rep_log = aux(log_norm.repeatLike(shifted));
            return shifted.sub(rep_log);
        }

        /// RMS normalization: `x / sqrt(mean(x²) + eps)`.
        ///
        /// Simpler than layerNorm (no mean subtraction). Used in LLaMA, Gemma, etc.
        /// Decomposes to: sqr -> mean -> add(eps) -> sqrt -> recip -> mul.
        pub fn rmsNorm(self: *Self, ne: []const usize, eps: T) *Self {
            const sq = aux(self.sqr());
            const ms = aux(sq.mean(ne));
            const ms_eps = aux(ms.add(scalarRepeatLike(self, eps, ms)));
            const ms_sqrt = aux(ms_eps.sqrt());
            const rms_inv = aux(ms_sqrt.recip());
            const rep_rms_inv = aux(rms_inv.repeatLike(self));
            return self.mul(rep_rms_inv);
        }

        /// Layer normalization: `(x - mean) / sqrt(var + eps)`.
        pub fn layerNorm(self: *Self, ne: []const usize, eps: T) *Self {
            const mu = aux(self.mean(ne));
            const rep_mu = aux(mu.repeatLike(self));
            const centered = aux(self.sub(rep_mu));
            const centered_sq = aux(centered.sqr());
            const variance = aux(centered_sq.mean(ne));
            const var_eps = aux(variance.add(scalarRepeatLike(self, eps, variance)));
            const var_sqrt = aux(var_eps.sqrt());
            const std_inv = aux(var_sqrt.recip());
            const rep_std_inv = aux(std_inv.repeatLike(centered));
            return centered.mul(rep_std_inv);
        }

        /// Add a lower-dimensional bias tensor, auto-broadcasting to self's shape.
        pub fn addBias(self: *Self, bias: *Self) *Self {
            return self.add(aux(bias.repeatLike(self)));
        }

        pub fn scale(self: *Self, other: *Self) *Self {
            assert(other.isScalar());
            return self.mul(aux(other.repeatLike(self)));
        }

        /// Scale every element by a scalar value.
        pub fn scaleByVal(self: *Self, val: T) *Self {
            return self.mul(scalarRepeatLike(self, val, self));
        }

        // ---------------------------------------------------------------
        // Matrix multiplication
        // ---------------------------------------------------------------

        /// Full matmul with explicit transpose flags.
        pub fn matMul(self: *Self, trans_self: bool, other: *Self, trans_other: bool) *Self {
            const alloc = a(self);
            assert(self.canMatMul(trans_self, other, trans_other));
            const is_node = self.grad != null or other.grad != null;
            assert(max_dims >= 4);

            const out_ne: [max_dims]usize = blk: {
                var dims: [max_dims]usize = [_]usize{1} ** max_dims;
                dims[0] = if (trans_other) other.ne[1] else other.ne[0];
                dims[1] = if (trans_self) self.ne[0] else self.ne[1];
                var i: usize = 2;
                while (i < max_dims) : (i += 1) dims[i] = self.ne[i];
                break :blk dims;
            };
            const res = Self.init(alloc, out_ne[0..@min(self.n_dims, other.n_dims)]) catch unreachable;
            res.op = .matmul;
            res.matmul_flags = .{ .trans0 = trans_self, .trans1 = trans_other };
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
            const target = if (self.isContiguous()) self else aux(self.contiguous());
            assert(target.nElems() == other.nElems());
            var strides: [max_dims]usize = [_]usize{0} ** max_dims;
            strides[0] = 1;
            var i: usize = 1;
            while (i < max_dims) : (i += 1) strides[i] = strides[i - 1] * (if (i - 1 < other.n_dims) other.ne[i - 1] else 1);
            return structuralView(target, other.ne[0..other.n_dims], strides, target.storage_offset, .reshape);
        }

        /// Return a contiguous copy of this tensor. If already contiguous, returns self.
        pub fn contiguous(self: *Self) *Self {
            if (self.isContiguous()) return self;
            const alloc = a(self);
            const res = Self.init(alloc, self.ne[0..self.n_dims]) catch unreachable;
            // Copy data respecting strides
            const n = self.nElems();
            for (0..n) |flat| {
                var src_offset: usize = self.storage_offset;
                var dim: usize = 0;
                while (dim < self.n_dims) : (dim += 1) {
                    const stride_prod = blk: {
                        var s: usize = 1;
                        for (0..dim) |d| s *= self.ne[d];
                        break :blk s;
                    };
                    const coord = (flat / stride_prod) % self.ne[dim];
                    src_offset += coord * self.strides[dim];
                }
                res.data[flat] = self.data[src_offset];
            }
            res.op = .view;
            res.src0 = self;
            return res;
        }

        pub fn reshape(self: *Self, ne: []const usize) *Self {
            const target = if (self.isContiguous()) self else aux(self.contiguous());
            const ne_prod = blk: {
                var prod: usize = 1;
                for (ne) |item| prod *= item;
                break :blk prod;
            };
            assert(target.nElems() == ne_prod);
            var strides: [max_dims]usize = [_]usize{0} ** max_dims;
            strides[0] = 1;
            var i: usize = 1;
            while (i < max_dims) : (i += 1) strides[i] = strides[i - 1] * (if (i - 1 < ne.len) ne[i - 1] else 1);
            return structuralView(target, ne, strides, target.storage_offset, .reshape);
        }

        /// Transpose the first two dimensions.
        pub fn transpose(self: *Self) *Self {
            var axes = [_]usize{0} ** max_dims;
            var i: usize = 0;
            while (i < max_dims) : (i += 1) axes[i] = i;
            axes[0] = 1;
            axes[1] = 0;
            return self.permute(axes[0..self.n_dims]);
        }

        pub fn permute(self: *Self, axes: []const usize) *Self {
            assert(axes.len == self.n_dims);
            var seen = [_]bool{false} ** max_dims;
            var out_ne = [_]usize{1} ** max_dims;
            var out_strides = [_]usize{0} ** max_dims;
            for (axes, 0..) |axis, i| {
                assert(axis < self.n_dims);
                assert(!seen[axis]);
                seen[axis] = true;
                out_ne[i] = self.ne[axis];
                out_strides[i] = self.strides[axis];
            }
            const res = structuralView(self, out_ne[0..axes.len], out_strides, self.storage_offset, .permute);
            const axes_tensor = Self.initIndexVectorCopy(a(self), axes) catch unreachable;
            res.src1 = axes_tensor;
            return res;
        }

        pub fn asStrided(self: *Self, ne: []const usize, strides: []const usize, storage_offset: usize) *Self {
            assert(ne.len <= max_dims);
            assert(ne.len == strides.len);
            var out_strides = [_]usize{0} ** max_dims;
            for (strides, 0..) |stride, i| out_strides[i] = stride;
            return structuralView(self, ne, out_strides, self.storage_offset + storage_offset, .as_strided);
        }

        /// Backward pass for as_strided: accumulate gradients back into the
        /// original tensor's shape by scatter-adding from the strided view.
        pub fn asStridedBackward(self: *Self, src: *Self) *Self {
            const alloc = a(self);
            const res = Self.init(alloc, src.ne[0..src.n_dims]) catch unreachable;
            _ = res.setAllScalar(0);
            // Accumulate self's gradient into res using src's strides
            const n = self.nElems();
            for (0..n) |i| {
                // Map flat index i through self's strides back to src's flat index
                var remaining = i;
                var src_idx: usize = src.storage_offset;
                var dim: usize = self.n_dims;
                while (dim > 0) {
                    dim -= 1;
                    const coord = remaining / self.strides[dim];
                    remaining %= self.strides[dim];
                    src_idx += coord * src.strides[dim];
                }
                if (src_idx < res.nElems()) {
                    res.data[src_idx] += self.data[i];
                }
            }
            return res;
        }

        pub fn scatterAddView(self: *Self, view_tensor: *Self) *Self {
            const alloc = a(self);
            const base = view_tensor.source0().?;
            const res = Self.init(alloc, base.ne[0..base.n_dims]) catch unreachable;
            res.op = .scatter_add_view;
            res.src0 = self;
            res.src1 = view_tensor;
            return res;
        }

        pub fn slidingWindow2d(self: *Self, kw: usize, kh: usize) *Self {
            assert(self.n_dims == 4);
            assert(kw <= self.ne[0]);
            assert(kh <= self.ne[1]);
            const out_w = self.ne[0] - kw + 1;
            const out_h = self.ne[1] - kh + 1;
            return self.asStrided(
                &.{ out_w, out_h, kw, kh, self.ne[2], self.ne[3] },
                &.{ self.strides[0], self.strides[1], self.strides[0], self.strides[1], self.strides[2], self.strides[3] },
                0,
            );
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

        // ---------------------------------------------------------------
        // Slice operations (for KV cache and general indexing)
        // ---------------------------------------------------------------

        /// Write `src` into column `pos` of `self`. Returns a node whose
        /// data aliases self's data (the modified buffer).
        ///
        /// self: [rows, cols]  (the destination, e.g. KV cache)
        /// src:  [rows, 1] or [rows]  (data to write)
        /// pos:  column index
        pub fn sliceAssign(self: *Self, src: *Self, pos: usize) *Self {
            const alloc = a(self);
            assert(self.n_dims >= 2);
            assert(pos < self.ne[1]);
            assert(src.ne[0] == self.ne[0]);
            const res = Self.initHelper(alloc, self.ne[0..self.n_dims], self.data) catch unreachable;
            res.op = .slice_assign;
            res.storage_offset = pos;
            res.grad = null; // inference-only
            res.src0 = src;
            res.src1 = self;
            return res;
        }

        /// Fused RoPE: result = x * cos + rotate_half(x) * sin.
        /// src0 = x [d, seq_len], src1 = cos_sin [2*d, seq_len] (cos in rows 0..d, sin in rows d..2d).
        pub fn ropeRotate(self: *Self, cos_sin: *Self) *Self {
            const alloc = a(self);
            const d = self.ne[0];
            assert(d % 2 == 0);
            assert(cos_sin.ne[0] == 2 * d);
            assert(cos_sin.ne[1] == self.ne[1]);
            const is_node = self.grad != null or cos_sin.grad != null;
            const res = Self.init(alloc, self.ne[0..self.n_dims]) catch unreachable;
            res.op = .rope;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = cos_sin;
            return res;
        }

        /// Write src into rows [offset..offset+src.ne[0]] of self, returning the mutated tensor.
        /// Analogous to sliceAssign but for row ranges instead of column positions.
        pub fn sliceAssignRows(self: *Self, src: *Self, row_offset: usize) *Self {
            const alloc = a(self);
            assert(self.n_dims >= 2);
            assert(src.ne[0] + row_offset <= self.ne[0]);
            assert(src.ne[1] == self.ne[1]);
            const res = Self.initHelper(alloc, self.ne[0..self.n_dims], self.data) catch unreachable;
            res.op = .slice_assign_rows;
            res.storage_offset = row_offset;
            res.grad = null; // inference-only
            res.src0 = src;
            res.src1 = self;
            return res;
        }

        /// View columns [start, end) of a 2D tensor. Returns a view with
        /// adjusted shape — no data copy.
        ///
        /// self: [rows, cols]
        /// Returns: [rows, end - start]
        pub fn sliceColumns(self: *Self, start: usize, end: usize) *Self {
            assert(self.n_dims >= 2);
            assert(end > start);
            assert(end <= self.ne[1]);
            const alloc = a(self);
            const new_cols = end - start;
            const res = Self.init(alloc, &.{ self.ne[0], new_cols }) catch unreachable;
            res.op = .view;
            res.src0 = self;
            // Point data to the start column: data[start * rows ..]
            res.data = self.data[start * self.ne[0] ..][0 .. new_cols * self.ne[0]];
            res.bookkeeping.data_ownership = .borrowed;
            return res;
        }

        /// View rows [start, end) of a 2D tensor.  Returns a strided view
        /// — no data copy.  The result has stride `[1, original_rows]` so
        /// matmul and other strided ops work correctly.
        ///
        /// self: [rows, cols]
        /// Returns: [end - start, cols]
        pub fn sliceRows(self: *Self, start: usize, end: usize) *Self {
            assert(self.n_dims >= 2);
            assert(end > start);
            assert(end <= self.ne[0]);
            const alloc = a(self);
            const new_rows = end - start;
            // Point data to the first element of the sliced row range.
            const res = Self.initHelper(alloc, &.{ new_rows, self.ne[1] }, self.data[start..]) catch unreachable;
            res.op = .as_strided;
            res.src0 = self;
            // Inherit parent's strides — the row slice is non-contiguous
            // when new_rows < original rows.
            res.strides[0] = self.strides[0];
            res.strides[1] = self.strides[1];
            res.storage_offset = 0;
            res.bookkeeping.data_ownership = .borrowed;
            return res;
        }

        // ---------------------------------------------------------------
        // Convolution & pooling
        // ---------------------------------------------------------------

        /// 2D convolution (valid, stride 1, no padding). Composite op.
        /// Decomposes to sliding-window view → broadcasted multiply → reduction.
        /// self: input  [W, H, C_in, N]
        /// kernel:      [kW, kH, C_in, C_out]
        /// result:      [W-kW+1, H-kH+1, C_out, N]
        pub fn conv2d(self: *Self, kernel: *Self) *Self {
            assert(self.n_dims == 4);
            assert(kernel.n_dims == 4);
            assert(self.ne[2] == kernel.ne[2]);
            const kw = kernel.ne[0];
            const kh = kernel.ne[1];
            const c_out = kernel.ne[3];
            const out_w = self.ne[0] - kw + 1;
            const out_h = self.ne[1] - kh + 1;
            const batch = self.ne[3];

            const windows_expanded = aux(self.asStrided(
                &.{ out_w, out_h, kw, kh, self.ne[2], c_out, batch },
                &.{ self.strides[0], self.strides[1], self.strides[0], self.strides[1], self.strides[2], 0, self.strides[3] },
                0,
            ));

            const kernel_view = aux(kernel.asStrided(
                &.{ out_w, out_h, kw, kh, self.ne[2], c_out, batch },
                &.{ 0, 0, kernel.strides[0], kernel.strides[1], kernel.strides[2], kernel.strides[3], 0 },
                0,
            ));

            const prod = aux(windows_expanded.mul(kernel_view));
            return prod.sum(&.{ out_w, out_h, 1, 1, 1, c_out, batch }).reshape(&.{ out_w, out_h, c_out, batch });
        }

        /// 2×2 max pooling with stride 2. Composite op.
        /// Decomposes to strided view of 2×2 windows → max reduction → reshape.
        /// self: input  [W, H, C, N]  (W and H must be even)
        /// result:      [W/2, H/2, C, N]
        pub fn maxPool2d(self: *Self) *Self {
            assert(self.n_dims == 4);
            assert(self.ne[0] % 2 == 0);
            assert(self.ne[1] % 2 == 0);
            const out_w = self.ne[0] / 2;
            const out_h = self.ne[1] / 2;
            const C = self.ne[2];
            const N = self.ne[3];

            // Create strided view of 2x2 windows: [out_w, out_h, 2, 2, C, N]
            const windows = aux(self.asStrided(
                &.{ out_w, out_h, 2, 2, C, N },
                &.{ self.strides[0] * 2, self.strides[1] * 2, self.strides[0], self.strides[1], self.strides[2], self.strides[3] },
                0,
            ));

            // Reduce max over the 2x2 window dims (dims 2,3): [out_w, out_h, 1, 1, C, N]
            return windows.max(&.{ out_w, out_h, 1, 1, C, N }).reshape(&.{ out_w, out_h, C, N });
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
