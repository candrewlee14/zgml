const std = @import("std");
const Op = @import("op.zig").Op;
const testing = std.testing;
const assert = std.debug.assert;
const builtin = @import("builtin");

const Opts = struct {
    use_blas: bool = false,
};
// const opts = .{.use_blas = false};
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
const max_nodes = 4096;
const max_params = 16;
const max_contexts = 64;
const max_opt = 4;
const GELU_COEF_A = 0.044715;
const SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

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

        /// Data of the tensor
        data: []T,
        /// Whether the data slice is owned by this tensor
        data_owned: bool,

        /// Create a tensor.
        /// The sizes of each dimension are specified by `ne`.
        /// The length of `ne` must be less than or equal to `max_dims`.
        /// `ne` is in the format [col, row, batch, channel] when using all dimensions.
        /// The number of dimensions will be infered by `ne.len`.
        /// Must call `deinit` to free.
        pub fn init(alloc: Alloc, ne: []const usize) Alloc.Error!*Self {
            return try initHelper(alloc, ne, null);
        }

        /// Free this tensor and its owned resources
        pub fn deinit(self: *Self, alloc: Alloc) void {
            if (self.data_owned) alloc.free(self.data);
            alloc.destroy(self);
        }

        // Mark this tensor as an input variable to be used for AD & optim algorithms.
        pub fn setParam(self: *Self, alloc: Alloc) Alloc.Error!void {
            self.is_param = true;
            assert(self.grad == null);
            self.grad = try self.dupTensor(alloc);
        }

        /// Helper for init.
        /// if `data_buf` is null, a new data slice is allocated.
        fn initHelper(alloc: Alloc, ne: []const usize, data_buf: ?[]T) Alloc.Error!*Self {
            std.debug.assert(ne.len <= max_dims);
            const tensor: *Self = try alloc.create(Self);
            tensor.* = .{
                .n_dims = @truncate(u8, ne.len),
                .ne = .{1} ** max_dims,
                .strides = .{0} ** max_dims,
                .op = .none,
                .is_param = false,
                .grad = null,
                .src0 = null,
                .src1 = null,
                .data = undefined,
                .data_owned = data_buf == null,
                .opt = .{null} ** max_opt,
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

        /// Create a new tensor as a view of the current tensor's data
        pub fn view(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try Self.initHelper(alloc, &self.ne, self.data);
        }

        /// Duplicate this tensor (with its shape) without preserving the data
        pub fn dupTensor(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return Self.initHelper(alloc, &self.ne, null);
        }

        fn unaryOp(self: *Self, alloc: Alloc, comptime op: Op, inplace: bool) Alloc.Error!*Self {
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) try self.view(alloc) else try self.dupTensor(alloc);
            res.op = op;
            res.grad = if (is_node) try self.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        fn binaryOp(self: *Self, alloc: Alloc, other: *Self, comptime op: Op, inplace: bool) Alloc.Error!*Self {
            var is_node: bool = !inplace and (self.grad != null or other.grad != null);
            switch (op) {
                .mul, .div, .scale => assert(!is_node), // TODO: implement backward
                else => {},
            }
            const res: *Self = if (inplace) try self.view(alloc) else try self.dup(alloc);
            res.op = op;
            res.grad = if (is_node) try self.dup(alloc) else null;
            res.src0 = self;
            res.src1 = other;

            return res;
        }

        pub fn dup(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .dup, false);
        }
        pub fn dupInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .dup, true);
        }
        pub fn add(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .add, false);
        }
        pub fn addInplace(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .add, true);
        }
        pub fn sub(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .sub, false);
        }
        pub fn subInplace(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .sub, true);
        }
        /// Element-wise multiply
        pub fn mul(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .mul, false);
        }
        /// Element-wise multiply inplace
        pub fn mulInplace(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .mul, true);
        }
        pub fn div(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .div, false);
        }
        pub fn divInplace(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            return try self.binaryOp(alloc, other, .div, true);
        }
        pub fn sqr(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sqr, false);
        }
        pub fn sqrInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sqr, true);
        }
        pub fn sqrt(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sqrt, false);
        }
        pub fn sqrtInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sqrt, true);
        }
        pub fn sum(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            const is_node: bool = self.grad != null;
            const res = try Self.init(alloc, &.{1});
            res.op = .sum;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn mean(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            const is_node: bool = self.grad != null;
            assert(!is_node); // TODO: implement
            var ne: [max_dims]usize = undefined;
            std.mem.copy(usize, &ne, &self.ne);
            ne[0] = 1;
            const res = try Self.init(alloc, &ne);
            res.op = .mean;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }
        pub fn repeatTo(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.canRepeatTo(other));
            const is_node: bool = self.grad != null;
            if (self.isSameShape(other) and !is_node) return self;
            const res = try Self.init(alloc, other.ne[0..other.n_dims]);
            res.op = .repeat;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }
        pub fn abs(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .abs, false);
        }
        pub fn absInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .abs, true);
        }
        pub fn sgn(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sgn, false);
        }
        pub fn sgnInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .sgn, true);
        }
        pub fn neg(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .neg, false);
        }
        pub fn negInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .neg, true);
        }
        pub fn step(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .step, false);
        }
        pub fn stepInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .step, true);
        }
        pub fn relu(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .relu, false);
        }
        pub fn reluInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .relu, true);
        }
        pub fn gelu(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .gelu, false);
        }
        pub fn geluInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .gelu, true);
        }
        /// Normalize along rows
        pub fn norm(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.unaryOp(alloc, .norm, false);
        }
        /// Normalize along rows inplace
        pub fn normInplace(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            // TODO: maybe store epsilon in src1?
            return try self.unaryOp(alloc, .norm, true);
        }

        pub fn matMul(self: *Self, trans_self: bool, alloc: Alloc, other: *Self, trans_other: bool) Alloc.Error!*Self {
            assert(self.canMatMul(trans_self, other, trans_other));
            const is_node = self.grad != null or other.grad != null;
            assert(max_dims == 4); // Need to update this function if max_dims changes

            const ne: [max_dims]usize = lbl: {
                if (trans_self) {
                    if (trans_other) {
                        // col, row
                        break :lbl .{ other.ne[1], self.ne[0], self.ne[2], other.ne[3] };
                    } else {
                        // col, col
                        break :lbl .{ other.ne[0], self.ne[0], self.ne[2], other.ne[3] };
                    }
                } else {
                    if (trans_other) {
                        // row, row
                        break :lbl .{ other.ne[1], self.ne[1], self.ne[2], other.ne[3] };
                    } else {
                        // row, col
                        break :lbl .{ other.ne[0], self.ne[1], self.ne[2], other.ne[3] };
                    }
                }
            };
            const res = try Self.init(alloc, ne[0..std.math.min(self.n_dims, other.n_dims)]);
            res.op = if (trans_self) if (trans_other) .matmul_t0t1 else .matmul_t0 else if (trans_other) .matmul_t1 else .matmul;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }
        pub fn scale(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(other.isScalar());
            return try self.binaryOp(alloc, other, .scale, false);
        }
        pub fn scaleInplace(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(other.isScalar());
            return try self.binaryOp(alloc, other, .scale, true);
        }
        fn cpyImpl(self: *Self, alloc: Alloc, other: *Self, inplace: bool) Alloc.Error!*Self {
            assert(self.nElems() == other.nElems());
            const is_node = !inplace and (self.grad != null or other.grad != null);
            assert(!is_node); // TODO: implement backward
            const res = if (is_node) try other.dupTensor(alloc) else try other.view(alloc);
            res.op = .cpy;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = other;
            return res;
        }
        pub fn cpyTo(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            return try self.cpyImpl(alloc, other, true);
        }
        pub fn cpyInplaceTo(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            return try self.cpyImpl(alloc, other, true);
        }
        pub fn reshapeLike(self: *Self, alloc: Alloc, other: *Self) Alloc.Error!*Self {
            assert(self.isContiguous());
            assert(other.isContiguous());
            assert(self.nElems() == other.nElems());
            const is_node = (self.grad != null or other.grad != null);
            assert(!is_node); // TODO: implement backward
            const res = try Self.initHelper(alloc, other.ne[0..other.n_dims], self.data);
            res.op = .reshape;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }
        pub fn reshape(self: *Self, alloc: Alloc, ne: []const usize) Alloc.Error!*Self {
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
            const res = try Self.initHelper(alloc, ne, self.data);
            res.op = .reshape;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn transpose(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            const is_node = self.grad != null;
            assert(!is_node); // TODO: implement backward
            const res = try self.view(alloc);
            res.ne[0] = self.ne[1];
            res.ne[1] = self.ne[0];
            res.strides[0] = self.strides[1];
            res.strides[1] = self.strides[0];
            res.op = .transpose;
            res.grad = if (is_node) try res.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }
        fn computeForwardDup(dst: *Self, src0: *Self) void {
            assert(dst.isContiguous());
            assert(dst.nElems() == src0.nElems());

            if (src0.isContiguous()) {
                std.mem.copy(T, &dst.data, &src0.data);
                return;
            }
            // TODO: implement non-contiguous dup
            @panic("Unimplemented forward dup for non-contiguous src");
        }
        fn computeForwardAdd(dst: *Self, src0: *Self, src1: *Self) void {
            assert(dst.isSameShape(src0));
            assert(src0.isSameShape(src1));
            for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                dst_item.* = src0_item + src1_item;
            }
        }
        fn computeForwardSub(dst: *Self, src0: *Self, src1: *Self) void {
            assert(dst.isSameShape(src0));
            assert(src0.isSameShape(src1));
            for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                dst_item.* = src0_item - src1_item;
            }
        }
        fn computeForwardMul(dst: *Self, src0: *Self, src1: *Self) void {
            assert(dst.isSameShape(src0));
            assert(src0.isSameShape(src1));
            for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                dst_item.* = src0_item * src1_item;
            }
        }
        fn computeForwardDiv(dst: *Self, src0: *Self, src1: *Self) void {
            assert(dst.isSameShape(src0));
            assert(src0.isSameShape(src1));
            for (src0.data, src1.data, dst.data) |src0_item, src1_item, *dst_item| {
                dst_item.* = src0_item / src1_item;
            }
        }
        fn computeForwardSqr(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = src0_item * src0_item;
            }
        }
        fn computeForwardSqrt(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = std.math.sqrt(src0_item);
            }
        }
        fn computeForwardSum(dst: *Self, src0: *Self) void {
            assert(dst.nElems() == 1);
            for (src0.data) |src0_item| {
                dst.data[0] += src0_item;
            }
        }
        fn computeForwardMean(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("not implemented");
        }
        fn computeForwardRepeat(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("not implemented");
        }
        fn computeForwardAbs(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = @fabs(src0_item);
            }
        }
        fn computeForwardSgn(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) 1 else if (src0_item < 0) -1 else 0;
            }
        }
        fn computeForwardNeg(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = -src0_item;
            }
        }
        fn computeForwardStep(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) 1 else 0;
            }
        }
        fn computeForwardReLu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |src0_item, *dst_item| {
                dst_item.* = if (src0_item > 0) src0_item else 0;
            }
        }
        fn computeForwardGeLu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |x, *dst_item| {
                dst_item.* = 0.5 * x * (1 + std.math.tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)));
            }
        }
        fn computeForwardNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }
        fn computeForwardRMSNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }
        fn shouldUseBlasForMatMul(dst: *Self, src0: *Self, src1: *Self) bool {
            return src0.isContiguous() and src1.isContiguous() and
                (dst.ne[0] >= 32 and dst.ne[1] >= 32 and src1.ne[0] >= 32);
        }
        fn computeForwardMatMul(dst: *Self, src0: *Self, comptime trans0: bool, src1: *Self, comptime trans1: bool) void {
            assert(max_dims == 4); // must update this func if max_dims changes

            // dst is not transposed
            assert(dst.strides[0] == 1);
            assert(dst.strides[0] <= dst.strides[1]);
            assert(dst.strides[1] <= dst.strides[2]);
            assert(dst.strides[2] <= dst.strides[3]);

            // src0 and src1 have same outer two dims
            assert(src0.ne[3] == src1.ne[3]);
            assert(src0.ne[2] == src1.ne[2]);

            assert(dst.ne[2] == src0.ne[2]);
            assert(dst.ne[3] == src0.ne[3]);

            if (!trans0 and !trans1) {
                // src0 and src1 can be matmul'd
                // src0 cols match src1 rows
                assert(src0.ne[0] == src1.ne[1]);
                // dst rows match src0 cols
                assert(dst.ne[1] == src0.ne[0]);
                // dst cols match src1 rows
                assert(dst.ne[0] == src1.ne[1]);
            } else if (!trans0 and trans1) {
                // src0 and transposed src1 can be matmul'd
                // same number of cols
                assert(src0.ne[0] == src1.ne[0]);
                // dst rows match src0 rows
                assert(dst.ne[1] == src0.ne[1]);
                // dst cols match src1 rows
                assert(dst.ne[0] == src1.ne[1]);
            } else if (trans0 and !trans1) {
                // transposed src0 and src1 can be matmul'd
                // same number of rows
                assert(src0.ne[1] == src1.ne[1]);
                // dst rows match src0 cols
                assert(dst.ne[1] == src0.ne[0]);
                // dst cols match src1 cols
                assert(dst.ne[0] == src1.ne[0]);
            } else if (trans0 and trans1) {
                // transposed src0 and transposed src1 can be matmul'd
                // src0 rows match src1 cols
                assert(src0.ne[1] == src1.ne[0]);
                // dst rows match src0 cols
                assert(dst.ne[1] == src0.ne[0]);
                // dst cols match src1 rows
                assert(dst.ne[0] == src1.ne[1]);
            }
            const src0_ne3 = src0.ne[3];
            const src0_ne2 = src0.ne[2];
            const src0_ne1 = src0.ne[1];
            const src0_ne0 = src0.ne[0];

            const src1_ne1 = src1.ne[1];
            const src1_ne0 = src1.ne[0];

            const dst_ne0 = dst.ne[0];

            const src0_ne1c = @intCast(c_int, src0_ne1);
            const src0_ne0c = @intCast(c_int, src0_ne0);
            const src1_ne1c = @intCast(c_int, src1_ne1);
            const src1_ne0c = @intCast(c_int, src1_ne0);
            const dst_ne0c = @intCast(c_int, dst_ne0);

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
                                var matmul_sum: f32 = 0;
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
                                var matmul_sum: f32 = 0;
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
                                var matmul_sum: f32 = 0;
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
                                var matmul_sum: f32 = 0;
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

        /// Sets all values in this tensor to `val`.
        /// Returns self for convenience.
        pub fn setAllScalar(self: *Self, val: T) *Self {
            std.mem.set(T, self.data, val);
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
            if (transSelf) {
                if (transOther) {
                    return self.ne[0] == other.ne[1];
                } else {
                    return self.ne[0] == other.ne[0];
                }
            } else {
                if (transOther) {
                    return self.ne[1] == other.ne[1];
                } else {
                    return self.ne[1] == other.ne[0];
                }
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

        pub fn canRepeatTo(self: *Self, other: *Self) bool {
            for (&self.ne, &other.ne) |selfNe, otherNe| {
                if (otherNe % selfNe != 0) return false;
            }
            return true;
        }

        /// Returns the element at the given coordinates.
        /// Coordinates are given in the format [col, row, batch, channel]
        /// or [col, row, batch] or [col, row] or [col]
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

        pub fn isSameShape(self: *Self, other: *Self) bool {
            for (self.ne, other.ne) |selfNe, otherNe| {
                if (selfNe != otherNe) {
                    return false;
                }
            }
            return true;
        }
    };
}

pub fn ComputeGraph(comptime T: type) type {
    return struct {
        n_nodes: usize,
        n_leafs: usize,
        n_threads: usize,

        work_size: usize,
        work: *Tensor(T),
        nodes: [max_nodes]*Tensor(T),
        grads: [max_nodes]*Tensor(T),
        leafs: [max_nodes]*Tensor(T),
    };
}

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(Tensor(f32));
    _ = testing.refAllDeclsRecursive(ComputeGraph(f32));
}

test "tensor init" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit(tac);
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        std.mem.copy(f32, tensor.data, &data);
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

test "tensor isMatrix" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit(tac);
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        try testing.expectEqual(true, tensor.isMatrix());
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3, 4 });
        defer tensor.deinit(tac);
        try testing.expectEqual(@as(usize, 24), tensor.nElems());
        try testing.expectEqual(false, tensor.isMatrix());
    }
}

test "tensor isSameShape" {
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit(tac);
        const tensor2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer tensor2.deinit(tac);
        try testing.expectEqual(false, tensor1.isSameShape(tensor2));
        try testing.expectEqual(false, tensor2.isSameShape(tensor1));
        try testing.expectEqual(true, tensor1.isSameShape(tensor1));
        try testing.expectEqual(true, tensor2.isSameShape(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer tensor1.deinit(tac);
        const tensor2 = try tensor1.view(tac);
        defer tensor2.deinit(tac);
        try testing.expectEqual(true, tensor1.isSameShape(tensor2));
    }
}

test "tensor canRepeatTo" {
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit(tac);
        const tensor2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer tensor2.deinit(tac);
        try testing.expectEqual(false, tensor1.canRepeatTo(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer tensor1.deinit(tac);
        const tensor2 = try Tensor(f32).init(tac, &.{ 4, 16, 9 });
        defer tensor2.deinit(tac);
        try testing.expectEqual(true, tensor1.canRepeatTo(tensor2));
    }
    {
        const tensor1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor1.deinit(tac);
        const tensor2 = try Tensor(f32).init(tac, &.{ 2, 3, 5 });
        defer tensor2.deinit(tac);
        try testing.expectEqual(true, tensor1.canRepeatTo(tensor2));
    }
}

test "tensor compute matmul_t0" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    {
        try testing.expectEqual(@as(usize, 6), t1.nElems());
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        std.mem.copy(f32, t1.data, &data);
    }
    const t2 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t2.deinit(tac);
    try testing.expectEqual(@as(usize, 6), t2.nElems());
    {
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        std.mem.copy(f32, t2.data, &data);
    }
    const dst = try t1.matMul(true, tac, t2, false);
    defer dst.deinit(tac);

    dst.computeForwardMatMul(t1, true, t2, false);

    const expected = [_]f32{
        35, 44,
        44, 56,
    };
    const expected_slice: []const f32 = expected[0..];
    try testing.expectEqualSlices(f32, expected_slice, dst.data);
}

test "tensor compute matmul_t1 2D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit(tac);
    {
        try testing.expectEqual(@as(usize, 6), t1.nElems());
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        std.mem.copy(f32, t1.data, &data);
    }
    const t2 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t2.deinit(tac);
    try testing.expectEqual(@as(usize, 6), t2.nElems());
    {
        const data = [_]f32{
            1, 2,
            3, 4,
            5, 6,
        };
        std.mem.copy(f32, t2.data, &data);
    }
    const dst = try t1.matMul(false, tac, t2, true);
    defer dst.deinit(tac);

    dst.computeForwardMatMul(t1, false, t2, true);

    const expected = [_]f32{
        5,  11, 17,
        11, 25, 39,
        17, 39, 61,
    };
    const expected_slice: []const f32 = expected[0..];
    try testing.expectEqualSlices(f32, expected_slice, dst.data);
}

test "tensor compute matmul_t1 3D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 2, 2 });
    defer t1.deinit(tac);
    {
        try testing.expectEqual(@as(usize, 8), t1.nElems());
        const data = [_]f32{
            1, 2,
            3, 4,
            //
            5, 6,
            7, 8,
        };
        std.mem.copy(f32, t1.data, &data);
        try testing.expectEqual(@as(f32, 4), t1.get(&.{ 1, 1, 0 }));
        try testing.expectEqual(@as(f32, 7), t1.get(&.{ 0, 1, 1 }));
    }
    const dst = try t1.matMul(false, tac, t1, true);
    defer dst.deinit(tac);

    dst.computeForwardMatMul(t1, false, t1, true);
    const expected = [_]f32{
        5,  11,
        11, 25,
        //
        61, 83,
        83, 113,
    };
    const expected_slice: []const f32 = expected[0..];
    try testing.expectEqualSlices(f32, expected_slice, dst.data);
}
