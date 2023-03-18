const std = @import("std");
const testing = std.testing;

const max_dims = 4;
const max_nodes = 4096;
const max_params = 16;
const max_contexts = 64;
const max_opt = 4;

const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

pub const Op = enum {
    none,
    dup,
    add,
    sub,
    mul,
    div,
    sqr,
    sqrt,
    sum,
    mean,
    repeat,
    abs,
    sgn,
    neg,
    relu,
    gelu,
    norm,
    mul_mat,
    count,
};

// const Obj = struct {
//   offset: usize,
//   size: usize,
//   next: ?*Obj,
// };
//
// pub const Ctx = struct {
//   buf: []u8,
//   buf_owned: bool,
//   n_objects: usize,
//   objects_begin: ?*Obj,
//   objects_end: ?*Obj,
// };

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        n_dims: u8,
        ne: [max_dims]usize,
        strides: [max_dims]usize,
        op: Op,
        is_param: bool,
        grad: ?*Self,
        src0: ?*Self,
        src1: ?*Self,
        opt: [max_dims]?*Self,

        data: []T,
        data_owned: bool,

        pub fn init(alloc: Alloc, ne: []const usize) Alloc.Error!*Self {
            return try initHelper(alloc, ne, null);
        }

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
            tensor.data = if (data_buf) |d| d else try alloc.alloc(T, tensor.nElems());
            for (ne, 0..) |neItem, i| {
                tensor.ne[i] = neItem;
            }
            for (1..max_dims) |i| {
                tensor.strides[i] = tensor.strides[i - 1] * tensor.ne[i - 1];
            }
            return tensor;
        }

        pub fn initScalar(alloc: Alloc, val: T) Alloc.Error!*Self {
            const tensor = try Self.init(alloc, &.{1});
            return tensor.setAllScalar(val);
        }

        pub fn view(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try Self.initHelper(alloc, &self.ne, self.data);
        }

        pub fn dupTensor(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return Self.initHelper(alloc, &self.ne, null);
        }

        fn dupImpl(self: *Self, alloc: Alloc, inplace: bool) Alloc.Error!*Self {
            const is_node: bool = !inplace and self.grad != null;
            const res = if (inplace) try self.view(alloc) else try self.dupTensor(alloc);
            res.op = .dup;
            res.grad = if (is_node) try self.dupTensor(alloc) else null;
            res.src0 = self;
            res.src1 = null;
            return res;
        }

        pub fn dup(self: *Self, alloc: Alloc) Alloc.Error!*Self {
            return try self.dupImpl(alloc, true);
        }

        pub fn deinit(self: *Self, alloc: Alloc) void {
            if (self.data_owned) alloc.free(self.data);
            alloc.destroy(self);
        }

        pub fn setAllScalar(self: *Self, val: T) *Self {
            std.mem.set(T, self.data, val);
            return self;
        }

        pub fn nElems(self: *Self) usize {
            var res: usize = 1;
            for (&self.ne) |neItem| {
                res *= neItem;
            }
            return res;
        }

        pub fn nRows(self: *Self) usize {
            var res: usize = 1;
            for (self.ne[1..]) |neItem| {
                res *= neItem;
            }
            return res;
        }

        pub fn isScalar(self: *Self) bool {
            for (self.ne[0..]) |neItem| {
                if (neItem != 1) return false;
            }
            return true;
        }

        pub fn isVector(self: *Self) bool {
            for (self.ne[1..]) |neItem| {
                if (neItem != 1) return false;
            }
            return true;
        }

        pub fn isMatrix(self: *Self) bool {
            for (self.ne[2..]) |neItem| {
                if (neItem != 1) return false;
            }
            return true;
        }

        pub fn canMulMat(self: *Self, other: *Self) bool {
            for (&self.ne, &other.ne, 0..) |selfNe, otherNe, i| {
                if (i != 1 and selfNe != otherNe) return false;
            }
            return true;
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

        pub fn get(self: *Self, coords: []usize) T {
            _ = coords;
            _ = self;
            // TODO: fix
            return 0;
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
