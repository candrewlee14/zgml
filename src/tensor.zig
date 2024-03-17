const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const builtin = @import("builtin");
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

const Op = @import("op.zig").Op;
const opts = @import("zgml_options");

const c = if (opts.use_blas)
    switch (builtin.os.tag) {
        .linux, .windows => @cImport(@cInclude("cblas.h")),
        .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
        else => @cImport(@compileError("Unsupported OS")),
    }
else
    void;

const max_dims = 4;
const max_opt = 4;
const GELU_COEF_A = 0.044715;
const SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Number of dimension
        n_dims: u8,
        /// Number of elements per axis
        /// Format like col, row, batch, channel
        ne: [max_dims]usize,
        /// Stride per axis
        strides: [max_dims]usize,
        /// Op that created this tensor.
        /// .none by default
        op: Op,
        is_param: bool,
        grad: ?*Self,
        src0: ?*Self,
        src1: ?*Self,
        opt: [max_dims]?*Self,
        /// Title of the tensor for debugging
        name: ?[]const u8,

        /// Data of the tensor
        data: []T,
        /// Whether the data slice is owned by this tensor
        data_owned: bool,

        alloc: Alloc,

        //#region Setup/Takedown methods

        /// Create a tensor.
        /// The sizes of each dimension are specified by `ne`.
        /// The length of `ne` must be less than or equal to `max_dims`.
        /// `ne` is in the format [#cols, #rows, batch, channel] when using all dimensions.
        /// The number of columns is the number of elements in a row, thus the data is stored row-major.
        /// The number of dimensions will be infered by `ne.len`.
        /// Must call `deinit` to free.
        pub fn init(alloc: Alloc, ne: []const usize) Alloc.Error!*Self {
            return try initHelper(alloc, ne, null);
        }

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

        /// Free this tensor and its owned resources
        pub fn deinit(self: *Self) void {
            if (self.data_owned) self.alloc.free(self.data);
            // if (self.grad) |grad| grad.deinit();
            self.alloc.destroy(self);
        }

        // Mark this tensor as an input variable to be used for AD & optim algorithms.
        pub fn setParam(self: *Self) void {
            self.is_param = true;
            assert(self.grad == null);
            self.grad = self.copyTensorShape();
        }

        /// Helper for init.
        /// if `data_buf` is null, a new data slice is allocated.
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
                .opt = .{null} ** max_opt,
                .alloc = alloc,
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

        /// Init a tensor with a single element `val`
        pub fn initScalar(alloc: Alloc, val: T) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, &.{1});
            return tensor.setAllScalar(val);
        }

        // init values in the interval [0, 1)
        pub fn initRand(alloc: Alloc, rng: *std.rand.Random, ne: []const usize) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, ne);
            for (tensor.data) |*d| {
                d.* = rng.float(T);
            }
            return tensor;
        }

        //#endregion

        //#region Lazy Operations

        /// Create a new tensor as a view of the current tensor's data
        pub fn view(self: *Self) *Self {
            var t = Self.initHelper(self.alloc, &self.ne, self.data) catch unreachable;
            t.op = .view;
            t.src0 = self;
            t.src1 = null;
            t.grad = if (self.grad) |grad| grad.view() else null;
            return t;
        }

        /// Duplicate this tensor (with its shape) without preserving the data
        pub fn copyTensorShape(self: *Self) *Self {
            return Self.initHelper(self.alloc, &self.ne, null) catch unreachable;
        }

        fn unaryOp(self: *Self, op: Op, inplace: bool) *Self {
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) self.view() else self.copyTensorShape();
            res.op = op;
            res.grad = if (is_node) self.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn binaryOp(self: *Self, other: *Self, op: Op, inplace: bool) *Self {
            assert(self.isSameShape(other));
            const is_node: bool = !inplace and (self.grad != null or other.grad != null);
            const res: *Self = if (inplace) self.view() else self.copyTensorShape();
            res.op = op;
            res.grad = if (is_node) self.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = other;

            return res;
        }
        pub fn dup(self: *Self) *Self {
            return self.unaryOp(.dup, false);
        }
        pub fn dupInplace(self: *Self) *Self {
            return self.unaryOp(.dup, true);
        }
        fn addImpl(self: *Self, other: *Self, inplace: bool) *Self {
            assert(self.isSameShape(other));
            return self.binaryOp(other, .add, inplace);
        }
        pub fn add(self: *Self, other: *Self) *Self {
            return self.addImpl(other, false);
        }
        pub fn addInplace(self: *Self, other: *Self) *Self {
            return self.addImpl(other, true);
        }
        fn subImpl(self: *Self, other: *Self, inplace: bool) *Self {
            return self.binaryOp(other, .sub, inplace);
        }
        pub fn sub(self: *Self, other: *Self) *Self {
            return self.subImpl(other, false);
        }
        pub fn subInplace(self: *Self, other: *Self) *Self {
            return self.subImpl(other, true);
        }
        /// Element-wise multiply
        pub fn mul(self: *Self, other: *Self) *Self {
            return self.binaryOp(other, .mul, false);
        }
        /// Element-wise multiply inplace
        pub fn mulInplace(self: *Self, other: *Self) *Self {
            return self.binaryOp(other, .mul, true);
        }
        pub fn div(self: *Self, other: *Self) *Self {
            return self.binaryOp(other, .div, false);
        }
        pub fn divInplace(self: *Self, other: *Self) *Self {
            return self.binaryOp(other, .div, true);
        }
        pub fn sqr(self: *Self) *Self {
            return self.unaryOp(.sqr, false);
        }
        pub fn sqrInplace(self: *Self) *Self {
            return self.unaryOp(.sqr, true);
        }
        pub fn sqrt(self: *Self) *Self {
            return self.unaryOp(.sqrt, false);
        }
        pub fn sqrtInplace(self: *Self) *Self {
            return self.unaryOp(.sqrt, true);
        }
        /// Sum all elements of a tensor
        pub fn sumAll(self: *Self) *Self {
            return sum(self, &.{1});
        }
        /// Sum elements of a tensor into the new shape defined by `ne`
        pub fn sum(self: *Self, ne: []const usize) *Self {
            assert(ne.len <= max_dims);
            assert(canSumToShape(self, ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(self.alloc, ne) catch unreachable;
            res.op = .sum;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn sumInto(self: *Self, other: *Self) *Self {
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view();
            res.op = .sum;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn mean(self: *Self, ne: []const usize) *Self {
            assert(ne.len <= max_dims);
            assert(canSumToShape(self, ne));
            const is_node: bool = self.grad != null;
            const res = Self.init(self.alloc, ne) catch unreachable;
            res.op = .mean;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn meanInto(self: *Self, other: *Self) *Self {
            assert(self.canSumTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view();
            res.op = .mean;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn repeatInto(self: *Self, other: *Self) *Self {
            assert(self.canRepeatTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = other.view();
            res.op = .repeat;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn repeat(self: *Self, ne: []usize) *Self {
            assert(ne.len <= max_dims);
            assert(self.canRepeatToShape(ne));
            const is_node: bool = self.grad != null;
            if (self.hasShape(ne) and !is_node) return self;
            const res = Self.init(self.alloc, ne) catch unreachable;
            res.op = .repeat;
            res.grad = if (self.grad) |grad| grad.repeat(ne) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }
        pub fn repeatLike(self: *Self, other: *Self) *Self {
            return repeat(self, &other.ne);
        }
        pub fn abs(self: *Self) *Self {
            return self.unaryOp(.abs, false);
        }
        pub fn absInplace(self: *Self) *Self {
            return self.unaryOp(.abs, true);
        }
        pub fn sgn(self: *Self) *Self {
            return self.unaryOp(.sgn, false);
        }
        pub fn sgnInplace(self: *Self) *Self {
            return self.unaryOp(.sgn, true);
        }
        pub fn neg(self: *Self) *Self {
            return self.unaryOp(.neg, false);
        }
        pub fn negInplace(self: *Self) *Self {
            return self.unaryOp(.neg, true);
        }
        pub fn step(self: *Self) *Self {
            return self.unaryOp(.step, false);
        }
        pub fn stepInplace(self: *Self) *Self {
            return self.unaryOp(.step, true);
        }
        pub fn relu(self: *Self) *Self {
            return self.unaryOp(.relu, false);
        }
        pub fn reluInplace(self: *Self) *Self {
            return self.unaryOp(.relu, true);
        }
        pub fn gelu(self: *Self) *Self {
            return self.unaryOp(.gelu, false);
        }
        pub fn geluInplace(self: *Self) *Self {
            return self.unaryOp(.gelu, true);
        }
        /// Normalize along rows
        pub fn norm(self: *Self) *Self {
            return self.unaryOp(.norm, false);
        }
        /// Normalize along rows inplace
        pub fn normInplace(self: *Self) *Self {
            // TODO: maybe store epsilon in src1?
            return self.unaryOp(.norm, true);
        }

        pub fn matMul(self: *Self, trans_self: bool, other: *Self, trans_other: bool) *Self {
            assert(self.canMatMul(trans_self, other, trans_other));
            const is_node = self.grad != null or other.grad != null;
            assert(max_dims == 4); // Need to update this function if max_dims changes

            const out_ne: [max_dims]usize = lbl: {
                break :lbl if (!trans_self and !trans_other)
                    // out #cols = other #cols, out #rows = self #rows
                    .{ other.ne[0], self.ne[1], self.ne[2], other.ne[3] }
                else if (trans_self and !trans_other)
                    // out #cols = other #cols, out #rows = self #cols
                    .{ other.ne[0], self.ne[0], self.ne[2], other.ne[3] }
                else if (!trans_self and trans_other)
                    // out #cols = other #rows, out #rows = self #rows
                    .{ other.ne[1], self.ne[1], self.ne[2], other.ne[3] }
                else
                    // out #cols = other #rows, out #rows = self #cols
                    .{ other.ne[1], self.ne[0], self.ne[2], other.ne[3] };
            };
            const res = Self.init(self.alloc, out_ne[0..@min(self.n_dims, other.n_dims)]) catch unreachable;
            res.op = if (trans_self) if (trans_other) .matmul_t0t1 else .matmul_t0 else if (trans_other) .matmul_t1 else .matmul;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = other;
            res.assertValidMatMulDims(self, trans_self, other, trans_other);
            return res;
        }
        pub fn scale(self: *Self, other: *Self) *Self {
            assert(other.isScalar());
            return self.binaryOp(other, .scale, false);
        }
        pub fn scaleInplace(self: *Self, other: *Self) *Self {
            assert(other.isScalar());
            return self.binaryOp(other, .scale, true);
        }
        fn cpyImpl(self: *Self, other: *Self, inplace: bool) *Self {
            assert(self.nElems() == other.nElems());
            const is_node = !inplace and (self.grad != null or other.grad != null);
            assert(!is_node); // TODO: implement backward
            const res = if (is_node) other.copyTensorShape() else other.view();
            res.op = .cpy;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }
        pub fn cpyTo(self: *Self, other: *Self) *Self {
            return self.cpyImpl(other, true);
        }
        pub fn cpyInplaceTo(self: *Self, other: *Self) *Self {
            return self.cpyImpl(other, true);
        }
        pub fn reshapeLike(self: *Self, other: *Self) *Self {
            assert(self.isContiguous());
            assert(other.isContiguous());
            assert(self.nElems() == other.nElems());
            const is_node = (self.grad != null or other.grad != null);
            assert(!is_node); // TODO: implement backward
            const res = Self.initHelper(self.alloc, other.ne[0..other.n_dims], self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }
        pub fn reshape(self: *Self, ne: []const usize) *Self {
            assert(self.isContiguous());
            const neProd = lbl: {
                var prod: usize = 0;
                for (ne) |item| {
                    prod *= item;
                }
                break :lbl prod;
            };
            assert(self.nElems() == neProd);
            const is_node = self.grad != null;
            assert(!is_node); // TODO: implement backward
            const res = Self.initHelper(self.alloc, ne, self.data) catch unreachable;
            res.op = .reshape;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn transpose(self: *Self) *Self {
            const is_node = self.grad != null;
            assert(!is_node); // TODO: implement backward
            const res = self.view();
            res.ne[0] = self.ne[1];
            res.ne[1] = self.ne[0];
            res.strides[0] = self.strides[1];
            res.strides[1] = self.strides[0];
            res.op = .transpose;
            res.grad = if (is_node) res.copyTensorShape() else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        //#endregion

        //#region Forward Computations for Operations

        pub fn computeDup(dst: *Self, src0: *Self) void {
            assert(dst.isContiguous());
            assert(dst.nElems() == src0.nElems());

            if (src0.isContiguous()) {
                @memcpy(dst.data, src0.data);
                return;
            }
            // TODO: implement non-contiguous dup
            @panic("Unimplemented forward dup for non-contiguous src");
        }
        pub fn computeAdd(dst: *Self, src0: *Self, src1: *Self) void {
            // TODO: add together elements at same position accounting for strides
            // TODO: this change needs to happen in all forward functions
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                    dst_item.* = src0_item + src1_item;
                }
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                for (src0.data, dst.data) |src0_item, *dst_item| {
                    dst_item.* = src0_item + src1.data[0];
                }
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                for (src1.data, dst.data) |src1_item, *dst_item| {
                    dst_item.* = src0.data[0] + src1_item;
                }
            }
        }
        pub fn computeSub(dst: *Self, src0: *Self, src1: *Self) void {
            // TODO: add together elements at same position accounting for strides
            // TODO: this change needs to happen in all forward functions
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                    dst_item.* = src0_item - src1_item;
                }
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                for (src0.data, dst.data) |src0_item, *dst_item| {
                    dst_item.* = src0_item - src1.data[0];
                }
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                for (src1.data, dst.data) |src1_item, *dst_item| {
                    dst_item.* = src0.data[0] - src1_item;
                }
            } else {
                @panic("Unimplemented forward sub for src sizes");
            }
        }
        pub fn computeMul(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                for (src1.data, dst.data) |src1_item, *dst_item| {
                    dst_item.* = src0.data[0] * src1_item;
                }
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                for (src0.data, dst.data) |src0_item, *dst_item| {
                    dst_item.* = src0_item * src1.data[0];
                }
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                    dst_item.* = src0_item * src1_item;
                }
            }
        }
        pub fn computeDiv(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                for (src1.data, dst.data) |src1_item, *dst_item| {
                    dst_item.* = src0.data[0] / src1_item;
                }
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                for (src0.data, dst.data) |src0_item, *dst_item| {
                    dst_item.* = src0_item / src1.data[0];
                }
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                    dst_item.* = src0_item / src1_item;
                }
            }
        }
        // inverse-broadcast mean from src0 to dst
        pub fn computeMean(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canSumTo(dst));
            const src0_ne_v: @Vector(4, usize) = src0.ne;
            const div_elems: T = @floatFromInt(@reduce(.Mul, src0_ne_v / dst.ne));

            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_nes = src0_nes % dst.ne;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);

                            dst.data[dst_idx] += src0.data[src0_idx] / div_elems; // we divide every iteration to avoid overflow, rather than summing and dividing once at the end
                        }
                    }
                }
            }
        }
        // inverse-broadcast sum from src0 to dst
        pub fn computeSum(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canSumTo(dst));
            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_nes = src0_nes % dst.ne;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);

                            dst.data[dst_idx] += src0.data[src0_idx];
                        }
                    }
                }
            }
        }
        // broadcast src0 to dst
        pub fn computeRepeat(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canRepeatTo(dst));
            for (0..dst.ne[3]) |ne3| {
                for (0..dst.ne[2]) |ne2| {
                    for (0..dst.ne[1]) |ne1| {
                        for (0..dst.ne[0]) |ne0| {
                            const nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const src0_nes = nes % src0.ne;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, nes * dst_stride_v);

                            dst.data[dst_idx] = src0.data[src0_idx];
                        }
                    }
                }
            }
        }

        pub fn computeSqr(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = src0_item * src0_item;
            }
        }
        pub fn computeSqrt(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = std.math.sqrt(src0_item);
            }
        }
        pub fn computeAbs(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = @abs(src0_item);
            }
        }
        pub fn computeSgn(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) 1 else if (src0_item < 0) -1 else 0;
            }
        }
        pub fn computeNeg(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = -src0_item;
            }
        }
        pub fn computeStep(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) 1 else 0;
            }
        }
        pub fn computeReLu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) src0_item else 0;
            }
        }
        pub fn computeGeLu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |x, *dst_item| {
                dst_item.* = 0.5 * x * (1 + std.math.tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)));
            }
        }
        pub fn computeNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }
        pub fn computeRMSNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }
        fn shouldUseBlasForMatMul(dst: *Self, src0: *Self, src1: *Self) bool {
            return src0.isContiguous() and src1.isContiguous() and
                (dst.ne[0] >= 32 and dst.ne[1] >= 32 and src1.ne[0] >= 32);
        }
        fn assertValidMatMulDims(dst: *Self, src0: *Self, trans0: bool, src1: *Self, trans1: bool) void {
            // src0 and src1 have same outer two dims
            assert(src0.ne[3] == src1.ne[3]);
            assert(src0.ne[2] == src1.ne[2]);

            assert(dst.ne[2] == src0.ne[2]);
            assert(dst.ne[3] == src0.ne[3]);

            if (!trans0 and !trans1) {
                // src0 and src1 can be matmul'd
                // src0 #cols match src1 #rows
                assert(src0.ne[0] == src1.ne[1]);
                // dst #rows match src0 #rows
                assert(dst.ne[1] == src0.ne[1]);
                // dst #cols match src1 #cols
                assert(dst.ne[0] == src1.ne[0]);
            } else if (!trans0 and trans1) {
                // same number of #cols (dot product of rows)
                assert(src0.ne[0] == src1.ne[0]);
                // dst #rows match src0 #rows
                assert(dst.ne[1] == src0.ne[1]);
                // dst #cols match src1 #rows
                assert(dst.ne[0] == src1.ne[1]);
            } else if (trans0 and !trans1) {
                // same #rows (dot product of columns)
                assert(src0.ne[1] == src1.ne[1]);
                // dst #rows match src0 #cols
                assert(dst.ne[1] == src0.ne[0]);
                // dst #cols match src1 #cols
                assert(dst.ne[0] == src1.ne[0]);
            } else if (trans0 and trans1) {
                // transposed src0 and transposed src1 can be matmul'd
                // src0 #rows match src1 #cols
                assert(src0.ne[1] == src1.ne[0]);
                // dst #rows match src0 #cols
                assert(dst.ne[1] == src0.ne[0]);
                // dst #cols match src1 #rows
                assert(dst.ne[0] == src1.ne[1]);
            }
        }
        pub fn computeMatMul(dst: *Self, src0: *Self, comptime trans0: bool, src1: *Self, comptime trans1: bool) void {
            assert(max_dims == 4); // must update this func if max_dims changes
            dst.assertValidMatMulDims(src0, trans0, src1, trans1);
            // dst is not transposed
            assert(dst.strides[0] == 1);
            assert(dst.strides[0] <= dst.strides[1]);
            assert(dst.strides[1] <= dst.strides[2]);
            assert(dst.strides[2] <= dst.strides[3]);

            const src0_ne3 = src0.ne[3];
            const src0_ne2 = src0.ne[2];
            const src0_ne1 = src0.ne[1];
            const src0_ne0 = src0.ne[0];

            const src1_ne1 = src1.ne[1];
            const src1_ne0 = src1.ne[0];

            const dst_ne0 = dst.ne[0];

            const src0_ne1c: c_int = @intCast(src0_ne1);
            const src0_ne0c: c_int = @intCast(src0_ne0);
            const src1_ne1c: c_int = @intCast(src1_ne1);
            const src1_ne0c: c_int = @intCast(src1_ne0);
            const dst_ne0c: c_int = @intCast(dst_ne0);

            for (0..src0_ne3) |src0_i3| {
                for (0..src0_ne2) |src0_i2| {
                    // mat mul
                    if (opts.use_blas and T == f32) {
                        // z = x * yT
                        c.cblas_sgemm(
                            c.CblasRowMajor,
                            if (trans0) c.CblasTrans else c.CblasNoTrans,
                            if (trans1) c.CblasTrans else c.CblasNoTrans,
                            if (trans0) src0_ne0c else src0_ne1c, // dst rows
                            if (trans1) src1_ne1c else src1_ne0c, // dst cols
                            if (trans0) src0_ne1c else src0_ne0c, // src0 row/col == src1 row/col
                            1.0, // alpha scaling factor
                            &src0.data[src0_i3 * src0.strides[3] + src0_i2 * src0.strides[2]],
                            src0_ne0c, // src0 first dim
                            &src1.data[src0_i3 * src1.strides[3] + src0_i2 * src1.strides[2]],
                            src1_ne0c, // src1 first dim
                            0.0, // beta scaling factor
                            &dst.data[src0_i3 * dst.strides[3] + src0_i2 * dst.strides[2]],
                            dst_ne0c, // dst first dim
                        );
                    } else if (!trans0 and !trans1) {
                        for (0..src0_ne1) |src0_i1| { // row0
                            for (0..src1_ne0) |src1_i0| { // col1
                                var matmul_sum: T = 0;
                                for (0..src0_ne0) |src0_i0| { // col0 == row1
                                    const src0_i_v = @Vector(4, usize){ src0_i0, src0_i1, src0_i2, src0_i3 };
                                    const src1_i_v = @Vector(4, usize){ src1_i0, src0_i0, src0_i2, src0_i3 };
                                    const src0_stride_v: @Vector(4, usize) = src0.strides;
                                    const src1_stride_v: @Vector(4, usize) = src1.strides;
                                    const src0_i = @reduce(.Add, src0_i_v * src0_stride_v);
                                    const src1_i = @reduce(.Add, src1_i_v * src1_stride_v);
                                    matmul_sum += src0.data[src0_i] * src1.data[src1_i];
                                }
                                // dst col = col1
                                // dst row = row0
                                const dst_i_v = @Vector(4, usize){ src1_i0, src0_i1, src0_i2, src0_i3 };
                                const dst_stride_v: @Vector(4, usize) = dst.strides;
                                const dst_i = @reduce(.Add, dst_i_v * dst_stride_v);
                                dst.data[dst_i] = matmul_sum;
                            }
                        }
                    } else if (!trans0 and trans1) {
                        for (0..src0_ne1) |src0_i1| { // row0
                            for (0..src1_ne1) |src1_i1| { // row1
                                var matmul_sum: T = 0;
                                for (0..src0_ne0) |src0_i0| { // col0 == col1
                                    const src0_i_v = @Vector(4, usize){ src0_i0, src0_i1, src0_i2, src0_i3 };
                                    const src1_i_v = @Vector(4, usize){ src0_i0, src1_i1, src0_i2, src0_i3 }; // different row
                                    const src0_stride_v: @Vector(4, usize) = src0.strides;
                                    const src1_stride_v: @Vector(4, usize) = src1.strides;
                                    const src0_i = @reduce(.Add, src0_i_v * src0_stride_v);
                                    const src1_i = @reduce(.Add, src1_i_v * src1_stride_v);
                                    matmul_sum += src0.data[src0_i] * src1.data[src1_i];
                                }
                                // dst col = row1
                                // dst row = row0
                                const dst_i_v = @Vector(4, usize){ src1_i1, src0_i1, src0_i2, src0_i3 };
                                const dst_stride_v: @Vector(4, usize) = dst.strides;
                                const dst_i = @reduce(.Add, dst_i_v * dst_stride_v);
                                dst.data[dst_i] = matmul_sum;
                            }
                        }
                    } else if (trans0 and !trans1) {
                        for (0..src0_ne0) |src0_i0| { // cos0
                            for (0..src1_ne0) |src1_i0| { // col1
                                var matmul_sum: T = 0;
                                for (0..src0_ne1) |src0_i1| { // row0 == row1
                                    const src0_i_v = @Vector(4, usize){ src0_i0, src0_i1, src0_i2, src0_i3 };
                                    const src1_i_v = @Vector(4, usize){ src1_i0, src0_i1, src0_i2, src0_i3 }; // different column
                                    const src0_stride_v: @Vector(4, usize) = src0.strides;
                                    const src1_stride_v: @Vector(4, usize) = src1.strides;
                                    const src0_i = @reduce(.Add, src0_i_v * src0_stride_v);
                                    const src1_i = @reduce(.Add, src1_i_v * src1_stride_v);
                                    matmul_sum += src0.data[src0_i] * src1.data[src1_i];
                                }
                                // dst col = col1
                                // dst row = col0
                                const dst_i_v = @Vector(4, usize){ src1_i0, src0_i0, src0_i2, src0_i3 };
                                const dst_stride_v: @Vector(4, usize) = dst.strides;
                                const dst_i = @reduce(.Add, dst_i_v * dst_stride_v);
                                dst.data[dst_i] = matmul_sum;
                            }
                        }
                    } else if (trans0 and trans1) {
                        for (0..src0_ne0) |src0_i0| { // col0
                            for (0..src1_ne1) |src1_i1| { // row1
                                var matmul_sum: T = 0;
                                for (0..src0_ne1) |src0_i1| { // col1 == row0
                                    const src0_i_v = @Vector(4, usize){ src0_i0, src0_i1, src0_i2, src0_i3 };
                                    const src1_i_v = @Vector(4, usize){ src0_i1, src1_i1, src0_i2, src0_i3 };
                                    const src0_stride_v: @Vector(4, usize) = src0.strides;
                                    const src1_stride_v: @Vector(4, usize) = src1.strides;
                                    const src0_i = @reduce(.Add, src0_i_v * src0_stride_v);
                                    const src1_i = @reduce(.Add, src1_i_v * src1_stride_v);
                                    matmul_sum += src0.data[src0_i] * src1.data[src1_i];
                                }
                                // dst col = row1
                                // dst row = col0
                                const dst_i_v = @Vector(4, usize){ src1_i1, src0_i0, src0_i2, src0_i3 };
                                const dst_stride_v: @Vector(4, usize) = dst.strides;
                                const dst_i = @reduce(.Add, dst_i_v * dst_stride_v);
                                dst.data[dst_i] = matmul_sum;
                            }
                        }
                    }
                }
            }
        }

        pub fn compute(tensor: *Tensor(T)) void {
            const src0 = tensor.src0;
            const src1 = tensor.src1;
            switch (tensor.op) {
                .none => {},
                .dup => tensor.computeDup(src0.?),
                .add => tensor.computeAdd(src0.?, src1.?),
                .sub => tensor.computeSub(src0.?, src1.?),
                .mul => tensor.computeMul(src0.?, src1.?),
                .div => tensor.computeDiv(src0.?, src1.?),
                .repeat => tensor.computeRepeat(src0.?),
                .sqr => tensor.computeSqr(src0.?),
                .sqrt => tensor.computeSqrt(src0.?),
                .sum => tensor.computeSum(src0.?),
                .mean => tensor.computeMean(src0.?),
                .abs => tensor.computeAbs(src0.?),
                .sgn => tensor.computeSgn(src0.?),
                .neg => tensor.computeNeg(src0.?),
                .step => tensor.computeStep(src0.?),
                .relu => tensor.computeReLu(src0.?),
                .gelu => tensor.computeGeLu(src0.?),
                .norm => tensor.computeNorm(src0.?),
                //
                .matmul => tensor.computeMatMul(src0.?, false, src1.?, false),
                .matmul_t0 => tensor.computeMatMul(src0.?, true, src1.?, false),
                .matmul_t1 => tensor.computeMatMul(src0.?, false, src1.?, true),
                .matmul_t0t1 => tensor.computeMatMul(src0.?, true, src1.?, true),
                .view => {},
                //
                // .scale => tensor.computeScale(src0.?),
                // .cpy => tensor.computeCpy(src0.?),
                // .reshape => tensor.computeReshape(src0.?),
                // .permute => tensor.computePermute(src0.?),
                // .transpose => tensor.computeTranspose(src0.?),
                // .get_rows,
                // diag_max_inf,
                // .soft_max,
                // .rope,
                else => @panic("Unimplemented forward OP"),
            }
        }

        //#endregion

        //#region Utility Methods

        /// Set data of this tensor to elements of data parameter.
        /// Number of elements must match.
        pub fn setData(self: *Self, data: []const T) void {
            assert(@as(usize, data.len) == self.nElems());
            @memcpy(self.data, data);
        }

        /// Sets all values in this tensor to `val`.
        /// Returns self for convenience.
        pub fn setAllScalar(self: *Self, val: T) *Self {
            @memset(self.data, val);
            return self;
        }

        /// Returns number of elements in this tensor
        pub fn nElems(self: *Self) usize {
            var res: usize = 1;
            for (&self.ne) |shape_item| {
                res *= shape_item;
            }
            return res;
        }

        /// Returns if this tensor is a single scalar value
        pub fn isScalar(self: *Self) bool {
            for (self.ne[0..]) |shape_item| {
                if (shape_item != 1) return false;
            }
            return true;
        }

        pub fn isVector(self: *Self) bool {
            for (self.ne[1..]) |shape_item| {
                if (shape_item != 1) return false;
            }
            return true;
        }

        pub fn isMatrix(self: *Self) bool {
            for (self.ne[2..]) |shape_item| {
                if (shape_item != 1) return false;
            }
            return true;
        }

        /// Returns if self can matmul with other.
        pub fn canMatMul(self: *Self, transSelf: bool, other: *Self, transOther: bool) bool {
            if (self.ne[3] != other.ne[3]) return false; // channels same
            if (self.ne[2] != other.ne[2]) return false; // batch same
            if (!transSelf and !transOther) {
                // self #cols == other #rows
                return self.ne[0] == other.ne[1];
            } else if (transSelf and !transOther) {
                // self #rows == other #rows (column dot product)
                return self.ne[1] == other.ne[1];
            } else if (!transSelf and transOther) {
                // self #cols == other #cols (row dot product)
                return self.ne[0] == other.ne[0];
            } else {
                // self #rows == other #cols
                return self.ne[1] == other.ne[0];
            }
        }

        pub fn isContiguous(self: *Self) bool {
            if (self.strides[0] != 1) {
                return false;
            }
            for (1..max_dims) |i| {
                if (self.strides[i] != self.strides[i - 1] * self.ne[i - 1]) return false;
            }
            return true;
        }

        /// Returns if this tensor can be repeated to match the shape of other.
        /// This is used for broadcasting.
        /// Unlike numpy, broadcasting works where each dimension in other is a multiple of the corresponding dimension in self.
        pub fn canRepeatTo(self: *Self, other: *Self) bool {
            return self.canRepeatToShape(&other.ne);
        }
        /// Returns if this tensor can be repeated to match the given shape.
        /// This is used for broadcasting.
        /// Unlike numpy, broadcasting works where each dimension in other is a multiple of the corresponding dimension in self.
        pub fn canRepeatToShape(self: *Self, other_ne: []const usize) bool {
            return shapeCanRepeatToShape(&self.ne, other_ne);
        }
        pub fn canSumTo(self: *Self, other: *Self) bool {
            return self.canSumToShape(&other.ne);
        }
        /// Returns if this tensor can be summed to match the given shape.
        /// This is the inverse of canRepeatToShape. It's the opposite of broadcasting.
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

        /// Returns the element at the given coordinates.
        /// Coordinates are given in the format [col#, row#, batch#, channel#]
        /// or [col#, row#, batch#] or [col#, row#] or [col#]
        pub fn get(self: *Self, coords: []const usize) T {
            assert(coords.len == self.n_dims);
            var idx: usize = 0;
            for (coords, self.strides[0..coords.len]) |coord, stride| {
                idx += coord * stride;
            }
            return self.data[idx];
        }

        /// Print out a summary of this tensor.
        pub fn print(self: *Self) void {
            std.debug.print("----{*}----\n", .{self});
            std.debug.print("shape: {any}\nstrides: {any}\ndata: {any}\n", .{ self.ne, self.strides, self.data });
            std.debug.print("--------------------------\n", .{});
        }
        /// Checks if two tensors are broadcastable.
        /// More info here: https://pytorch.org/docs/stable/notes/broadcasting.html
        pub fn isBroadcastable(self: *Self, other: *Self) bool {
            for (self.ne, other.ne) |selfNe, otherNe| {
                if (selfNe != otherNe and selfNe != 1 and otherNe != 1) {
                    return false;
                }
            }
            return true;
        }

        pub fn isSameShape(self: *Self, other: *Self) bool {
            return self.hasShape(&other.ne);
        }
        pub fn hasShape(self: *Self, other_ne: []const usize) bool {
            for (self.ne, 0..) |selfNe, i| {
                const otherNe = if (i < other_ne.len) other_ne[i] else 1;
                if (selfNe != otherNe) {
                    return false;
                }
            }
            return true;
        }

        //#endregion

        //#region Backward Computations for Operations

        fn addToScratchUniq(scratch: *std.ArrayList(*Tensor(T)), tensor: *Tensor(T)) Alloc.Error!void {
            for (scratch.items) |item| {
                if (item == tensor) return;
            }
            try scratch.append(tensor);
        }
        pub fn backward(tensor: *Tensor(T), scratch: *std.ArrayList(*Tensor(T)), inplace: bool) Alloc.Error!void {
            const src0_o = tensor.src0;
            const src1_o = tensor.src1;
            switch (tensor.op) {
                .none, .view => {},
                .dup => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const new_grad = grad.addImpl(tensor.grad.?, inplace);
                        assert(new_grad.isSameShape(grad));
                        src0.grad = new_grad;
                    }
                },
                .add => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        // TODO: since add can coerce shapes, we need to reduce back to the original shape if they are different
                        // TODO: this is a problem for all ops that can coerce shapes
                        // TODO: scalar + vector, for example, when added together, will have the same shape as the vector
                        // TODO: if the scalar had a grad, the grad would now be the same shape as the vector
                        // TODO: we need to reduce the grad back to the original shape of the scalar with average
                        // TODO: Consider broadcastable ops: https://pytorch.org/docs/stable/notes/broadcasting.html
                        // TODO: http://coldattic.info/post/116/
                        src0.grad = grad.addImpl(tensor.grad.?, inplace);
                    }
                    if (src1.grad) |grad| {
                        src1.grad = grad.addImpl(tensor.grad.?, inplace);
                    }
                },
                .sub => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = grad.addImpl(tensor.grad.?, inplace);
                    }
                    if (src1.grad) |grad| {
                        src1.grad = grad.subImpl(tensor.grad.?, inplace);
                    }
                },
                .mul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const src1_x = src1.mul(tensor.grad.?);
                        // remove grad
                        if (src1_x.grad) |gradp| {
                            gradp.deinit();
                            src1_x.grad = null;
                        }
                        src0.grad = grad.addImpl(src1_x, inplace);
                    }
                    if (src1.grad) |grad| {
                        const src0_x = src0.mul(tensor.grad.?);
                        // remove grad
                        if (src0_x.grad) |gradp| {
                            gradp.deinit();
                            src0_x.grad = null;
                        }
                        src1.grad = grad.addImpl(src0_x, inplace);
                    }
                },
                .div => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const src1_x = src1.div(tensor.grad.?);
                        // remove grad
                        if (src1_x.grad) |gradp| {
                            gradp.deinit();
                            src1_x.grad = null;
                        }
                        src0.grad = grad.addImpl(src1_x, inplace);
                    }
                    if (src1.grad) |grad| {
                        const src0_x = src0.div(tensor.grad.?);
                        // remove grad
                        if (src0_x.grad) |gradp| {
                            gradp.deinit();
                            src0_x.grad = null;
                        }
                        src1.grad = grad.addImpl(src0_x, inplace);
                    }
                },
                .repeat => {
                    const src0 = src0_o.?;
                    const t_grad = tensor.grad.?;
                    if (src0.grad) |grad| {
                        src0.grad = t_grad.sumInto(grad);
                    }
                },
                .sqr => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        // src_grad = 2 * src * out_grad
                        const t2 = try Self.initScalar(tensor.alloc, 2);
                        const t2_rep = t2.repeatLike(src0);
                        const src0_2 = src0.mul(t2_rep);
                        // remove grad
                        if (src0_2.grad) |gradp| {
                            gradp.deinit();
                            src0_2.grad = null;
                        }
                        const src0_2_grad = src0_2.mul(tensor.grad.?);
                        src0.grad = grad.addImpl(src0_2_grad, inplace);
                    }
                },
                // .sqrt,
                .sum, .mean => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = grad.addImpl(tensor.grad.?.repeatLike(grad), inplace);
                    }
                },
                // .abs,
                // .sgn,
                // .neg,
                // .step,
                // .relu,
                // .gelu,
                // .norm,
                //
                .matmul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = tensor.grad.?.matMul(false, src1, true);
                        // remove grad
                        if (z.grad) |gradp| {
                            gradp.deinit();
                            z.grad = null;
                        }
                        src0.grad = grad.addImpl(z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = src0.matMul(true, tensor.grad.?, false);
                        // remove grad
                        if (z.grad) |gradp| {
                            gradp.deinit();
                            z.grad = null;
                        }
                        src1.grad = grad.addImpl(z, inplace);
                    }
                },
                // TODO: remove add to scratch
                .matmul_t0 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(scratch, tensor.grad.?.matMul(false, src1, true));
                        src0.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(scratch, src0.matMul(false, tensor.grad.?, false));
                        src1.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                },
                .matmul_t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(scratch, tensor.grad.?.matMul(false, src1, false));
                        src0.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(scratch, src0.matMul(true, tensor.grad.?, false));
                        src1.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                },
                .matmul_t0t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(scratch, tensor.grad.?.matMul(true, src1, false));
                        src0.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(scratch, src0.matMul(false, tensor.grad.?, true));
                        src1.grad = grad.addImpl(scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(scratch, grad); // move the old one into scratch
                    }
                },
                // //
                // .scale,
                // .cpy,
                // .reshape,
                // .view,
                // .permute,
                // .transpose,
                // .get_rows,
                // // diag_max_inf,
                // .soft_max,
                // .rope,
                else => @panic("Unimplemented backward OP"),
            }
        }

        //#endregion
    };
}

//#region Tests

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(Tensor(f32));
}

//#region Setup/Takedown Tests

test "init" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        @memcpy(tensor.data, &data);
        try testing.expectEqual(@as(f32, 1), tensor.get(&.{ 0, 0 }));
        try testing.expectEqual(@as(f32, 3), tensor.get(&.{ 0, 1 }));
        try testing.expectEqual(@as(f32, 6), tensor.get(&.{ 1, 2 }));
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 5, 3, 2 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 30), tensor.nElems());
    }
}

test "initLinspace" {
    {
        const t = try Tensor(f32).initLinspace(tac, &.{20}, 0, 20);
        defer t.deinit();
        try testing.expectEqual(@as(usize, 20), t.nElems());
        for (t.data, 0..) |v, i| {
            try testing.expectEqual(@as(f32, @floatFromInt(i)), v);
        }
    }
    {
        const t = try Tensor(f32).initLinspace(tac, &.{20}, 0, 10);
        defer t.deinit();
        try testing.expectEqual(@as(usize, 20), t.nElems());
        for (t.data, 0..) |v, i| {
            try testing.expectEqual(@as(f32, @floatFromInt(i)) * 0.5, v);
        }
    }
}

//#endregion

//#region Utility Method Tests

test "isMatrix" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        try testing.expectEqual(true, tensor.isMatrix());
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3, 4 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 24), tensor.nElems());
        try testing.expectEqual(false, tensor.isMatrix());
    }
}

test "isSameShape" {
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit();
        const tensor2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer tensor2.deinit();
        try testing.expectEqual(false, tensor1.isSameShape(tensor2));
        try testing.expectEqual(false, tensor2.isSameShape(tensor1));
        try testing.expectEqual(true, tensor1.isSameShape(tensor1));
        try testing.expectEqual(true, tensor2.isSameShape(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer tensor1.deinit();
        const tensor2 = tensor1.view();
        defer tensor2.deinit();
        try testing.expectEqual(true, tensor1.isSameShape(tensor2));
    }
}

test "canRepeatTo" {
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit();
        const tensor2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer tensor2.deinit();
        try testing.expectEqual(false, tensor1.canRepeatTo(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer tensor1.deinit();
        const tensor2 = try Tensor(f32).init(tac, &.{ 4, 16, 9 });
        defer tensor2.deinit();
        try testing.expectEqual(true, tensor1.canRepeatTo(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit();
        const tensor2 = try Tensor(f32).init(tac, &.{ 2, 3, 5 });
        defer tensor2.deinit();
        try testing.expectEqual(true, tensor1.canRepeatTo(tensor2));
    }
}

//#endregion

//#region Forward Computation Tests

test "compute mean" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });

    const dst = t1.mean(&.{1});
    defer dst.deinit();

    dst.computeMean(t1);

    const expected = [_]f32{3.5};
    for (dst.data, 0..) |v, i| {
        try testing.expectApproxEqAbs(expected[i], v, 1e-10);
    }
}

test "compute matmul" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });

    const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
    defer t2.deinit();
    t2.setData(&[_]f32{
        1, 2, 3,
        4, 5, 6,
    });

    const dst = t1.matMul(false, t2, false);
    defer dst.deinit();

    dst.computeMatMul(t1, false, t2, false);

    const expected = [_]f32{
        9,  12, 15,
        19, 26, 33,
        29, 40, 51,
    };
    try testing.expectEqualSlices(f32, &expected, dst.data);
}

test "compute matmul_t0" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });

    const t2 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t2.deinit();
    t2.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });

    const dst = t1.matMul(true, t2, false);
    defer dst.deinit();

    dst.computeMatMul(t1, true, t2, false);

    const expected = [_]f32{
        35, 44,
        44, 56,
    };
    try testing.expectEqualSlices(f32, &expected, dst.data);
}

test "compute matmul_t1 2D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    defer t1.deinit();

    const t2 = try Tensor(f32).init(tac, &.{ 2, 3 });
    t2.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    defer t2.deinit();

    const dst = t1.matMul(false, t2, true);
    defer dst.deinit();

    dst.computeMatMul(t1, false, t2, true);

    const expected = [_]f32{
        5,  11, 17,
        11, 25, 39,
        17, 39, 61,
    };
    try testing.expectEqualSlices(f32, &expected, dst.data);
}

test "compute matmul_t1 3D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 2, 2 });
    defer t1.deinit();
    const data = [_]f32{
        1, 2,
        3, 4,
        //
        5, 6,
        7, 8,
    };
    t1.setData(&data);

    try testing.expectEqual(@as(f32, 4), t1.get(&.{ 1, 1, 0 }));
    try testing.expectEqual(@as(f32, 7), t1.get(&.{ 0, 1, 1 }));

    const dst = t1.matMul(false, t1, true);
    defer dst.deinit();

    dst.computeMatMul(t1, false, t1, true);
    const expected = [_]f32{
        5,  11,
        11, 25,
        //
        61, 83,
        83, 113,
    };
    try testing.expectEqualSlices(f32, &expected, dst.data);
}

//#endregion

//#endregion
