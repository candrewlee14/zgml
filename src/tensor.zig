//! Core tensor type for the zgml machine learning library.
//!
//! A `Tensor` is a multi-dimensional array (up to `max_dims` dimensions) that supports
//! lazy computation graph construction, forward evaluation, and reverse-mode
//! automatic differentiation.

const std = @import("std");
const assert = std.debug.assert;
const builtin = @import("builtin");
const Alloc = std.mem.Allocator;

const Op = @import("op.zig").Op;

/// Maximum number of dimensions a tensor can have.
pub const max_dims = 8;

/// Generic tensor parameterized on element type `T` (typically `f32` or `f64`).
///
/// Tensors form the nodes of a computation graph. "Lazy" operations (e.g. `add`,
/// `mul`, `matMul`) record the operation without computing it — the actual math
/// runs when `compute()` is called (usually via `ComputeGraph`).
///
/// The runtime layout deliberately keeps hot execution metadata (`ne`, `strides`,
/// `storage_offset`, `op`, `data`) separate from colder lifecycle bookkeeping
/// (`role`, data/index ownership, auxiliary index payloads).
///
/// Tensors that are marked as parameters (via `setParam`) track gradients for
/// use with optimizers.
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const MatMulFlags = packed struct {
            trans0: bool = false,
            trans1: bool = false,
        };

        /// Coarse tensor role used by lifecycle code.
        ///
        /// Invariants:
        /// - `.parameter` tensors are user-visible trainable leaves created via `setParam()`.
        /// - `.internal_aux` tensors are helper nodes owned by a parent edge and reclaimed
        ///   by that parent during `deinit()`.
        /// - All other tensors use `.plain`.
        pub const Role = enum {
            plain,
            parameter,
            internal_aux,
        };

        pub const Ownership = enum(u1) {
            borrowed,
            owned,
        };

        pub const IndexState = struct {
            data: ?[]usize = null,
            ownership: Ownership = .borrowed,

            pub fn has(self: IndexState) bool {
                return self.data != null;
            }

            pub fn owns(self: IndexState) bool {
                return self.ownership == .owned;
            }
        };

        /// Colder bookkeeping kept separate from the hot shape/op/data fields.
        ///
        /// Invariants:
        /// - `data_ownership` applies only to `.data`.
        /// - `index` ownership applies only when `index.data != null`.
        /// - `role` is the single source of truth for parameter vs internal-aux lifetime.
        pub const Bookkeeping = struct {
            role: Role = .plain,
            data_ownership: Ownership = .owned,
            index: IndexState = .{},

            pub fn ownsData(self: Bookkeeping) bool {
                return self.data_ownership == .owned;
            }
        };

        // -- delegation to split-out implementations --
        const api = @import("tensor/api.zig").Api(Self, T);
        const fwd = @import("tensor/forward.zig").Ops(Self, T);
        const bwd = @import("tensor/backward.zig").Ops(Self);

        /// Number of logical dimensions.
        n_dims: u8,
        /// Number of elements per axis.
        ne: [max_dims]usize,
        /// Memory stride per axis.
        strides: [max_dims]usize,
        /// Base element offset into the underlying storage.
        storage_offset: usize,
        /// The operation that produced this tensor (`.none` for user-created tensors).
        op: Op,
        /// Transpose flags for matmul ops (ignored for other ops).
        matmul_flags: MatMulFlags = .{},
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
        /// Colder ownership and role bookkeeping.
        bookkeeping: Bookkeeping,
        /// Allocator for creating intermediate tensors during lazy ops.
        /// Set automatically by `init` / graph creation so callers don't need
        /// to pass an allocator to every operation.
        alloc: ?Alloc = null,

        // ---------------------------------------------------------------
        // Initialization
        // ---------------------------------------------------------------

        /// Create a tensor with the given shape.
        ///
        /// The number of dimensions is inferred from `ne.len`.
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

        pub fn initHelper(alloc: Alloc, ne: []const usize, data_buf: ?[]T) Alloc.Error!*Self {
            std.debug.assert(ne.len <= max_dims);
            // When passed the full max_dims-length array (e.g. from &tensor.ne),
            // trim trailing 1s to get the effective dimensionality.
            // For shorter slices, the caller explicitly chose the dimension count.
            var effective_dims: u8 = @truncate(ne.len);
            if (ne.len == max_dims) {
                while (effective_dims > 1 and ne[effective_dims - 1] == 1) effective_dims -= 1;
            }
            const tensor: *Self = try alloc.create(Self);
            tensor.* = .{
                .n_dims = effective_dims,
                .ne = .{1} ** max_dims,
                .strides = .{0} ** max_dims,
                .storage_offset = 0,
                .op = .none,
                .grad = null,
                .src0 = null,
                .src1 = null,
                .name = null,
                .data = undefined,
                .bookkeeping = .{ .data_ownership = if (data_buf == null) .owned else .borrowed },
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

        /// Create an internal auxiliary 1-D index carrier backed by typed usize indices.
        pub fn initIndexVectorCopy(alloc: Alloc, idx: []const usize) Alloc.Error!*Self {
            const tensor: *Self = try alloc.create(Self);
            tensor.* = .{
                .n_dims = 1,
                .ne = blk: {
                    var dims: [max_dims]usize = [_]usize{1} ** max_dims;
                    dims[0] = idx.len;
                    break :blk dims;
                },
                .strides = blk: {
                    var strides: [max_dims]usize = [_]usize{0} ** max_dims;
                    strides[0] = 1;
                    var i: usize = 1;
                    while (i < max_dims) : (i += 1) strides[i] = idx.len;
                    break :blk strides;
                },
                .storage_offset = 0,
                .op = .none,
                .grad = null,
                .src0 = null,
                .src1 = null,
                .name = null,
                .data = &.{},
                .bookkeeping = .{
                    .role = .internal_aux,
                    .data_ownership = .borrowed,
                    .index = .{
                        .data = try alloc.dupe(usize, idx),
                        .ownership = .owned,
                    },
                },
                .alloc = alloc,
            };
            return tensor;
        }

        pub fn hasIndexBuffer(self: *const Self) bool {
            return self.bookkeeping.index.has();
        }

        pub fn indexData(self: *const Self) ?[]const usize {
            return self.bookkeeping.index.data;
        }

        pub fn source0(self: *const Self) ?*Self {
            return self.src0;
        }

        pub fn source1(self: *const Self) ?*Self {
            return self.src1;
        }

        pub fn setSources(self: *Self, src0: ?*Self, src1: ?*Self) void {
            self.src0 = src0;
            self.src1 = src1;
        }

        pub fn opTag(self: *const Self) Op {
            return self.op;
        }

        pub fn setOp(self: *Self, op: Op) void {
            self.op = op;
        }

        pub fn isOp(self: *const Self, op: Op) bool {
            return self.op == op;
        }

        pub fn sourceIs(self: *const Self, which: enum { src0, src1 }, other: *Self) bool {
            return switch (which) {
                .src0 => self.src0 == other,
                .src1 => self.src1 == other,
            };
        }

        pub fn gradOrNull(self: *const Self) ?*Self {
            return self.grad;
        }

        pub fn hasGrad(self: *const Self) bool {
            return self.grad != null;
        }

        pub fn setGrad(self: *Self, grad: ?*Self) void {
            self.grad = grad;
        }

        pub fn isParam(self: *const Self) bool {
            return self.bookkeeping.role == .parameter;
        }

        pub fn isInternalAux(self: *const Self) bool {
            return self.bookkeeping.role == .internal_aux;
        }

        pub fn markInternalAux(self: *Self) *Self {
            self.bookkeeping.role = .internal_aux;
            return self;
        }

        pub fn ownsData(self: *const Self) bool {
            return self.bookkeeping.ownsData();
        }

        pub fn ownsIndexData(self: *const Self) bool {
            return self.bookkeeping.index.owns();
        }

        fn ownsStandaloneParamGrad(self: *const Self) bool {
            return self.isParam() and self.grad != null and self.grad.?.op == .none;
        }

        pub fn isLeaf(self: *const Self) bool {
            return self.op == .none and self.grad == null;
        }

        /// Free this tensor and its owned data.
        ///
        /// Ownership rules:
        /// - views and structural aliases borrow `.data`
        /// - internal aux tensors are edge-owned helpers reclaimed here
        /// - standalone parameter grads are freed here, while graph-owned grads
        ///   are usually released by `ComputeGraph` arena teardown
        pub fn deinit(self: *Self) void {
            const al = self.alloc.?;
            // Only free grad for params we own — in graph contexts, the arena
            // handles cleanup. Grad tensors from buildBackward may have shared
            // references, so we only free the simple case (param with no sources).
            if (self.ownsStandaloneParamGrad()) {
                self.grad.?.deinit();
            }
            if (self.source0()) |src0| {
                if (src0.isInternalAux()) src0.deinit();
            }
            if (self.source1()) |src1| {
                if (src1.isInternalAux()) src1.deinit();
            }
            if (self.ownsIndexData()) {
                if (self.bookkeeping.index.data) |idx| al.free(idx);
            }
            if (self.ownsData()) al.free(self.data);
            al.destroy(self);
        }

        /// Mark this tensor as a learnable parameter, allocating a gradient tensor.
        pub fn setParam(self: *Self) void {
            self.bookkeeping.role = .parameter;
            assert(!self.hasGrad());
            self.setGrad(self.copyTensorShape());
        }

        // ---------------------------------------------------------------
        // Lazy graph-building operations
        //
        // These methods build the computation graph without performing any math.
        // They allocate new tensor nodes via `catch unreachable` — this is
        // intentional: lazy ops are designed to be infallible so they can be
        // composed fluently (e.g. `x.sub(y).sqr().mean(&.{1})`).
        //
        // When used with a ComputeGraph's arena allocator, allocation failure
        // is not expected. If you need fallible allocation, use the `init*`
        // family directly.
        // ---------------------------------------------------------------

        fn repeatInto(self: *Self, other: *Self) *Self {
            return api.repeatInto(self, other);
        }

        pub const view = api.view;
        pub const copyTensorShape = api.copyTensorShape;
        pub const add = api.add;
        pub const addInplace = api.addInplace;
        pub const sub = api.sub;
        pub const mul = api.mul;
        pub const div = api.div;
        pub const sqr = api.sqr;
        pub const sqrt = api.sqrt;
        pub const recip = api.recip;
        pub const exp = api.exp;
        pub const log = api.log;
        pub const abs = api.abs;
        pub const sgn = api.sgn; // Internal: used by backward pass only
        pub const neg = api.neg;
        pub const step = api.step; // Internal: used by backward pass only
        pub const relu = api.relu;
        pub const gelu = api.gelu;
        pub const sumAll = api.sumAll;
        pub const maxAll = api.maxAll;
        pub const sum = api.sum;
        pub const max = api.max;

        pub const sumInto = api.sumInto;
        pub const mean = api.mean;
        pub const softmax = api.softmax;
        pub const logSoftmax = api.logSoftmax;
        pub const rmsNorm = api.rmsNorm;
        pub const layerNorm = api.layerNorm;
        pub const meanInto = api.meanInto;
        pub const repeat = api.repeat;
        pub const repeatLike = api.repeatLike;
        pub const matMul = api.matMul;
        pub const mm = api.mm;
        pub const Tmm = api.Tmm;
        pub const mmT = api.mmT;
        pub const TmmT = api.TmmT;
        pub const gatherRows = api.gatherRows;
        pub const scatterAddRows = api.scatterAddRows; // Internal: used by backward pass only
        pub const pickRows = api.pickRows;
        pub const scatterAddPicks = api.scatterAddPicks; // Internal: used by backward pass only
        pub const gatherRowsIdx = api.gatherRowsIdx;
        pub const pickRowsIdx = api.pickRowsIdx;
        pub const addBias = api.addBias;
        pub const scale = api.scale;
        pub const scaleByVal = api.scaleByVal;
        pub const conv2d = api.conv2d;
        pub const maxPool2d = api.maxPool2d;
        pub const contiguous = api.contiguous;
        pub const reshapeLike = api.reshapeLike;
        pub const reshape = api.reshape;
        pub const transpose = api.transpose;
        pub const permute = api.permute;
        pub const asStrided = api.asStrided;
        pub const scatterAddView = api.scatterAddView; // Internal: used by backward pass only
        pub const broadcastTo = api.broadcastTo;
        pub const sliceAssign = api.sliceAssign;
        pub const sliceColumns = api.sliceColumns;
        pub const sliceRows = api.sliceRows;
        pub const slidingWindow2d = api.slidingWindow2d;

        // ---------------------------------------------------------------
        // Forward compute — delegated to tensor/forward.zig
        // ---------------------------------------------------------------

        /// Dispatch forward computation for this tensor's op.
        pub const compute = fwd.compute;

        // Primitive forward compute functions (used by dispatch and available for
        // direct imperative use, e.g. in optimizer step functions).
        pub const computeAdd = fwd.computeAdd;
        pub const computeMul = fwd.computeMul;
        pub const computeSub = fwd.computeSub;
        pub const computeDiv = fwd.computeDiv;
        pub const computeNeg = fwd.computeNeg;
        pub const computeAbs = fwd.computeAbs;
        pub const computeSgn = fwd.computeSgn;
        pub const computeStep = fwd.computeStep;
        pub const computeSqrt = fwd.computeSqrt;
        pub const computeRecip = fwd.computeRecip;
        pub const computeExp = fwd.computeExp;
        pub const computeLog = fwd.computeLog;
        pub const computeGelu = fwd.computeGelu;
        pub const computeSum = fwd.computeSum;
        pub const computeMax = fwd.computeMax;
        pub const computeRepeat = fwd.computeRepeat;
        pub const computeGatherRows = fwd.computeGatherRows;
        pub const computeScatterAddRows = fwd.computeScatterAddRows;
        pub const computePickRows = fwd.computePickRows;
        pub const computeScatterAddPicks = fwd.computeScatterAddPicks;
        pub const computeTranspose = fwd.computeTranspose;
        pub const computeMatMul = fwd.computeMatMul;
        pub const computeMatMulParallel = fwd.computeMatMulParallel;
        pub const assertValidMatMulDims = fwd.assertValidMatMulDims;

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
        pub fn nElems(self: *const Self) usize {
            var res: usize = 1;
            for (&self.ne) |shape_item| res *= shape_item;
            return res;
        }

        /// True if this tensor is a single scalar value (all dims == 1).
        pub fn isScalar(self: *const Self) bool {
            for (self.ne[0..]) |s| {
                if (s != 1) return false;
            }
            return true;
        }

        /// True if all strides are zero — a scalar broadcast to any shape.
        /// Every element maps to data[storage_offset].
        pub fn isBroadcastScalar(self: *const Self) bool {
            for (self.strides[0..]) |s| {
                if (s != 0) return false;
            }
            return true;
        }

        /// True if this is a 1-D vector (dims 1+ are all 1).
        pub fn isVector(self: *const Self) bool {
            for (self.ne[1..]) |s| {
                if (s != 1) return false;
            }
            return true;
        }

        /// True if this is a 2-D matrix (dims 2+ are all 1).
        pub fn isMatrix(self: *const Self) bool {
            for (self.ne[2..]) |s| {
                if (s != 1) return false;
            }
            return true;
        }

        /// True if self can be matrix-multiplied with `other`, accounting for transpositions.
        pub fn canMatMul(self: *const Self, transSelf: bool, other: *const Self, transOther: bool) bool {
            const self_contract = if (transSelf) self.ne[1] else self.ne[0];
            const other_contract = if (transOther) other.ne[0] else other.ne[1];
            if (self_contract != other_contract) return false;

            var i: usize = 2;
            while (i < max_dims) : (i += 1) {
                if (self.ne[i] != other.ne[i]) return false;
            }
            return true;
        }

        /// True if data is laid out contiguously in memory starting at offset 0.
        pub fn isContiguous(self: *const Self) bool {
            if (self.storage_offset != 0) return false;
            return self.isDenseLayout();
        }

        /// True if elements are densely packed (standard strides) but possibly
        /// at a non-zero storage_offset. Use `denseSlice()` to get the actual
        /// contiguous element range.
        pub fn isDenseLayout(self: *const Self) bool {
            if (self.strides[0] != 1) return false;
            for (1..max_dims) |i| {
                if (self.strides[i] != self.strides[i - 1] * self.ne[i - 1]) return false;
            }
            return true;
        }

        /// Returns the densely-packed element slice `data[offset..offset+nElems]`.
        /// Only valid when `isDenseLayout()` is true.
        pub fn denseSlice(self: *const Self) []T {
            return self.data[self.storage_offset..][0..self.nElems()];
        }

        /// Const version of `denseSlice()`.
        pub fn denseSliceConst(self: *const Self) []const T {
            return self.data[self.storage_offset..][0..self.nElems()];
        }

        /// True if self can be broadcast (repeated) to match `other`'s shape.
        pub fn canRepeatTo(self: *const Self, other: *const Self) bool {
            return self.canRepeatToShape(&other.ne);
        }

        pub fn canRepeatToShape(self: *const Self, other_ne: []const usize) bool {
            return shapeCanRepeatToShape(&self.ne, other_ne);
        }

        /// True if self can be reduced (summed) down to `other`'s shape.
        pub fn canSumTo(self: *const Self, other: *const Self) bool {
            return self.canSumToShape(&other.ne);
        }

        pub fn canSumToShape(self: *const Self, other_ne: []const usize) bool {
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
        pub fn get(self: *const Self, coords: []const usize) T {
            assert(coords.len == self.n_dims);
            var idx: usize = self.storage_offset;
            for (coords, self.strides[0..coords.len]) |coord, stride| {
                idx += coord * stride;
            }
            return self.data[idx];
        }

        /// Print a debug summary of this tensor.
        pub fn print(self: *const Self) void {
            std.debug.print("----{*}----\n", .{self});
            std.debug.print("shape: {any}\nstrides: {any}\ndata: {any}\n", .{ self.ne, self.strides, self.data });
            std.debug.print("--------------------------\n", .{});
        }

        /// True if self and other have compatible shapes for broadcasting (numpy-style).
        pub fn isBroadcastable(self: *const Self, other: *const Self) bool {
            const nd = @max(self.n_dims, other.n_dims);
            for (0..nd) |i| {
                const self_ne = if (i < self.n_dims) self.ne[i] else 1;
                const other_ne = if (i < other.n_dims) other.ne[i] else 1;
                if (self_ne != other_ne and self_ne != 1 and other_ne != 1) return false;
            }
            return true;
        }

        /// True if self and other have identical shapes.
        pub fn isSameShape(self: *const Self, other: *const Self) bool {
            return self.hasShape(&other.ne);
        }

        pub fn hasShape(self: *const Self, other_ne: []const usize) bool {
            for (self.ne, 0..) |selfNe, i| {
                const otherNe = if (i < other_ne.len) other_ne[i] else 1;
                if (selfNe != otherNe) return false;
            }
            return true;
        }

        // ---------------------------------------------------------------
        // Fused elementwise operations
        //
        // `map` and `map2` apply a user-provided function element-wise in a
        // single pass over memory. No intermediate tensors are allocated.
        // The result is an eagerly-computed leaf tensor (.op = .none).
        //
        // LLVM auto-vectorizes the loop in ReleaseFast builds. For explicit
        // SIMD control, use the compute* functions in tensor/forward.zig.
        // ---------------------------------------------------------------

        /// Apply a unary function element-wise: `dst[i] = f(self[i])`.
        /// Computes eagerly — no graph node is created.
        pub fn map(self: *Self, comptime f: fn (T) T) Alloc.Error!*Self {
            const dst = try Self.init(self.alloc.?, self.ne[0..self.n_dims]);
            for (self.data, dst.data) |x, *d| {
                d.* = f(x);
            }
            return dst;
        }

        /// Apply a binary function element-wise: `dst[i] = f(self[i], other[i])`.
        /// Both tensors must have the same shape. Computes eagerly.
        pub fn map2(self: *Self, other: *Self, comptime f: fn (T, T) T) Alloc.Error!*Self {
            assert(self.isSameShape(other));
            const dst = try Self.init(self.alloc.?, self.ne[0..self.n_dims]);
            for (self.data, other.data, dst.data) |aa, b, *d| {
                d.* = f(aa, b);
            }
            return dst;
        }
    };
}

test {
    _ = @import("tensor/forward.zig");
    _ = @import("tensor/backward.zig");
}

const testing = std.testing;
const tac = std.testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(Tensor(f32));
}

test "init" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 6), tensor.nElems());
        const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
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

test "reshape" {
    const t = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t.deinit();
    t.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const r = t.reshape(&.{ 3, 2 });
    defer r.deinit();

    try testing.expectEqual(@as(usize, 6), r.nElems());
    try testing.expectEqual(@as(u8, 2), r.n_dims);
    try testing.expectEqual(@as(usize, 3), r.ne[0]);
    try testing.expectEqual(@as(usize, 2), r.ne[1]);
    // data is shared (view), so values match
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, r.data);
}

test "permute view preserves logical indexing" {
    const t = try Tensor(f32).init(tac, &.{ 2, 3, 4 });
    defer t.deinit();
    for (t.data, 0..) |*d, i| d.* = @floatFromInt(i);

    const p = t.permute(&.{ 2, 0, 1 });
    defer p.deinit();

    try testing.expectEqual(@as(u8, 3), p.n_dims);
    try testing.expectEqual(@as(usize, 4), p.ne[0]);
    try testing.expectEqual(@as(usize, 2), p.ne[1]);
    try testing.expectEqual(@as(usize, 3), p.ne[2]);
    try testing.expectEqual(t.get(&.{ 1, 2, 3 }), p.get(&.{ 3, 1, 2 }));
}

test "broadcastTo creates zero-stride view" {
    const t = try Tensor(f32).init(tac, &.{ 1, 3 });
    defer t.deinit();
    t.setData(&.{ 10, 20, 30 });

    const b = t.broadcastTo(&.{ 2, 3 });
    defer b.deinit();

    try testing.expectEqual(@as(usize, 0), b.strides[0]);
    try testing.expectEqual(@as(f32, 10), b.get(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 10), b.get(&.{ 1, 0 }));
    try testing.expectEqual(@as(f32, 30), b.get(&.{ 1, 2 }));
}

test "slidingWindow2d exposes overlapping patches" {
    const t = try Tensor(f32).init(tac, &.{ 4, 4, 1, 1 });
    defer t.deinit();
    for (t.data, 0..) |*d, i| d.* = @floatFromInt(i);

    const w = t.slidingWindow2d(2, 2);
    defer w.deinit();

    try testing.expectEqual(@as(u8, 6), w.n_dims);
    try testing.expectEqual(@as(usize, 3), w.ne[0]);
    try testing.expectEqual(@as(usize, 3), w.ne[1]);
    try testing.expectEqual(@as(usize, 2), w.ne[2]);
    try testing.expectEqual(@as(usize, 2), w.ne[3]);
    try testing.expectEqual(@as(f32, 0), w.get(&.{ 0, 0, 0, 0, 0, 0 }));
    try testing.expectEqual(@as(f32, 1), w.get(&.{ 0, 0, 1, 0, 0, 0 }));
    try testing.expectEqual(@as(f32, 4), w.get(&.{ 0, 0, 0, 1, 0, 0 }));
    try testing.expectEqual(@as(f32, 5), w.get(&.{ 0, 0, 1, 1, 0, 0 }));
    try testing.expectEqual(@as(f32, 10), w.get(&.{ 1, 1, 1, 1, 0, 0 }));
}

test "compute conv2d composite view path" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 2, 1, 1 });
    x.setData(&.{
        1, 2,
        3, 4,
    });
    const k = try Tensor(f32).init(a, &.{ 1, 1, 1, 1 });
    k.setData(&.{
        2,
    });

    const y = x.conv2d(k);
    try g.buildForward(y);
    g.compute();

    try testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, y.data);
}

test "backward conv2d composite view path" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 2, 1, 1 });
    x.setData(&.{
        1, 2,
        3, 4,
    });
    x.setParam();

    const k = try Tensor(f32).init(a, &.{ 1, 1, 1, 1 });
    k.setData(&.{
        3,
    });
    k.setParam();

    const out = x.conv2d(k).sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 3), x.grad.?.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3), x.grad.?.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3), x.grad.?.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3), x.grad.?.data[3], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 10), k.grad.?.data[0], 1e-5);
}

test "isMatrix" {
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer tensor.deinit();
        try testing.expectEqual(true, tensor.isMatrix());
    }
    {
        const tensor = try Tensor(f32).init(tac, &.{ 2, 3, 4 });
        defer tensor.deinit();
        try testing.expectEqual(false, tensor.isMatrix());
    }
}

test "isSameShape" {
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer t1.deinit();
        const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer t2.deinit();
        try testing.expectEqual(false, t1.isSameShape(t2));
        try testing.expectEqual(true, t1.isSameShape(t1));
    }
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer t1.deinit();
        const t2 = t1.view();
        defer t2.deinit();
        try testing.expectEqual(true, t1.isSameShape(t2));
    }
}

test "canRepeatTo" {
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
        defer t1.deinit();
        const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
        defer t2.deinit();
        try testing.expectEqual(false, t1.canRepeatTo(t2));
    }
    {
        const t1 = try Tensor(f32).init(tac, &.{ 2, 4, 3 });
        defer t1.deinit();
        const t2 = try Tensor(f32).init(tac, &.{ 4, 16, 9 });
        defer t2.deinit();
        try testing.expectEqual(true, t1.canRepeatTo(t2));
    }
}

test "compute mean" {
    // mean decomposes into a subgraph, so use ComputeGraph to evaluate
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t1 = try Tensor(f32).init(a, &.{ 2, 3 });
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.mean(&.{1});
    try g.buildForward(dst);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 3.5), dst.data[0], 1e-10);
}

test "compute matmul" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const t2 = try Tensor(f32).init(tac, &.{ 3, 2 });
    defer t2.deinit();
    t2.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(false, t2, false);
    defer dst.deinit();
    dst.computeMatMul(t1, false, t2, false);

    try testing.expectEqualSlices(f32, &.{ 9, 12, 15, 19, 26, 33, 29, 40, 51 }, dst.data);
}

test "compute matmul_t0" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(true, t1, false);
    defer dst.deinit();
    dst.computeMatMul(t1, true, t1, false);

    try testing.expectEqualSlices(f32, &.{ 35, 44, 44, 56 }, dst.data);
}

test "compute matmul_t1 2D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t1.deinit();
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6 });

    const dst = t1.matMul(false, t1, true);
    defer dst.deinit();
    dst.computeMatMul(t1, false, t1, true);

    try testing.expectEqualSlices(f32, &.{ 5, 11, 17, 11, 25, 39, 17, 39, 61 }, dst.data);
}

test "compute matmul_t1 3D" {
    const t1 = try Tensor(f32).init(tac, &.{ 2, 2, 2 });
    defer t1.deinit();
    t1.setData(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });

    const dst = t1.matMul(false, t1, true);
    defer dst.deinit();
    dst.computeMatMul(t1, false, t1, true);

    try testing.expectEqualSlices(f32, &.{ 5, 11, 11, 25, 61, 83, 83, 113 }, dst.data);
}

// Validate tiled matmul at various sizes against a naive reference.
// Tests edge cases: sizes smaller than tile, non-divisible by tile_m/tile_n,
// rectangular matrices, and sizes that exercise all tile paths.
fn naiveMatMulRef(alloc: Alloc, M: usize, K: usize, N: usize, a: []const f32, b: []const f32) ![]f32 {
    const dst = try alloc.alloc(f32, M * N);
    for (0..M) |i| {
        for (0..N) |j| {
            var s: f32 = 0;
            for (0..K) |ki| s += a[i * K + ki] * b[ki * N + j];
            dst[i * N + j] = s;
        }
    }
    return dst;
}

test "matmul edge cases - various sizes" {
    // Sizes chosen to exercise: smaller than tile, not divisible by tile_m(6),
    // not divisible by tile_n(16) or vec_size(8), large enough for full tiles.
    const cases = [_][3]usize{
        .{ 1, 1, 1 }, // scalar
        .{ 3, 3, 3 }, // smaller than tile_m
        .{ 5, 5, 5 }, // smaller than tile_m, not power of 2
        .{ 7, 13, 5 }, // rectangular, M > tile_m, N < vec_size
        .{ 6, 6, 6 }, // exactly tile_m
        .{ 9, 9, 9 }, // just over tile_m, N > vec_size but < tile_n
        .{ 13, 7, 17 }, // rectangular, N > tile_n, not aligned
        .{ 16, 16, 16 }, // exactly 2*vec_size = tile_n
        .{ 17, 17, 17 }, // just over tile_n
        .{ 32, 32, 32 }, // multiple of everything
        .{ 33, 65, 17 }, // large rectangular, nothing aligned
    };

    for (cases) |c| {
        const M = c[0];
        const K = c[1];
        const N = c[2];

        const t_a = try Tensor(f32).init(tac, &.{ K, M });
        defer t_a.deinit();
        const t_b = try Tensor(f32).init(tac, &.{ N, K });
        defer t_b.deinit();

        // Fill with deterministic data
        var prng = std.Random.DefaultPrng.init(42 + M * 1000 + K * 100 + N);
        for (t_a.data) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;
        for (t_b.data) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;

        const t_dst = t_a.matMul(false, t_b, false);
        defer t_dst.deinit();
        t_dst.computeMatMul(t_a, false, t_b, false);

        const expected = try naiveMatMulRef(tac, M, K, N, t_a.data, t_b.data);
        defer tac.free(expected);

        for (expected, t_dst.data, 0..) |exp, got, i| {
            if (@abs(exp - got) > 1e-3) {
                std.debug.print("matmul {d}x{d}x{d}: mismatch at index {d}: expected {d}, got {d}\n", .{ M, K, N, i, exp, got });
                return error.TestExpectedApproxEqAbs;
            }
        }
    }
}

test "matmul edge cases - transposed" {
    const M = 11;
    const K = 9;
    const N = 7;

    const t_a = try Tensor(f32).init(tac, &.{ K, M });
    defer t_a.deinit();
    const t_b = try Tensor(f32).init(tac, &.{ N, K });
    defer t_b.deinit();

    var prng = std.Random.DefaultPrng.init(999);
    for (t_a.data) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;
    for (t_b.data) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;

    // Test all 4 transpose variants
    // t0: uses t_a as (K x M), transposed -> (M x K)
    // For matMul(trans_self=true, ...) the "logical A" is t_a transposed = M x K
    // So we need K rows, M cols in storage -> shape {M, K}
    const t_at = try Tensor(f32).init(tac, &.{ M, K });
    defer t_at.deinit();
    // Transpose t_a data manually: t_a is (K cols, M rows) row-major
    // t_at should be (M cols, K rows) such that t_at^T = t_a logically
    for (0..M) |i| {
        for (0..K) |j| {
            t_at.data[j * M + i] = t_a.data[i * K + j];
        }
    }

    const t_bt = try Tensor(f32).init(tac, &.{ K, N });
    defer t_bt.deinit();
    for (0..K) |i| {
        for (0..N) |j| {
            t_bt.data[j * K + i] = t_b.data[i * N + j];
        }
    }

    // Reference: A(MxK) * B(KxN)
    const expected = try naiveMatMulRef(tac, M, K, N, t_a.data, t_b.data);
    defer tac.free(expected);

    // Test t0: t_at^T * t_b (t_at is K x M in storage, transposed = M x K)
    {
        const dst = t_at.matMul(true, t_b, false);
        defer dst.deinit();
        dst.computeMatMul(t_at, true, t_b, false);
        for (expected, dst.data, 0..) |exp, got, i| {
            if (@abs(exp - got) > 1e-3) {
                std.debug.print("matmul_t0 mismatch at {d}: {d} vs {d}\n", .{ i, exp, got });
                return error.TestExpectedApproxEqAbs;
            }
        }
    }

    // Test t1: t_a * t_bt^T (t_bt is N x K in storage, transposed = K x N)
    {
        const dst = t_a.matMul(false, t_bt, true);
        defer dst.deinit();
        dst.computeMatMul(t_a, false, t_bt, true);
        for (expected, dst.data, 0..) |exp, got, i| {
            if (@abs(exp - got) > 1e-3) {
                std.debug.print("matmul_t1 mismatch at {d}: {d} vs {d}\n", .{ i, exp, got });
                return error.TestExpectedApproxEqAbs;
            }
        }
    }

    // Test t0t1: t_at^T * t_bt^T
    {
        const dst = t_at.matMul(true, t_bt, true);
        defer dst.deinit();
        dst.computeMatMul(t_at, true, t_bt, true);
        for (expected, dst.data, 0..) |exp, got, i| {
            if (@abs(exp - got) > 1e-3) {
                std.debug.print("matmul_t0t1 mismatch at {d}: {d} vs {d}\n", .{ i, exp, got });
                return error.TestExpectedApproxEqAbs;
            }
        }
    }
}

test "backward - exp" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 1.5);
    x.setParam();
    const out = x.exp();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    const expected = std.math.exp(@as(f32, 1.5));
    try testing.expectApproxEqAbs(expected, out.data[0], 1e-6);
    try testing.expectApproxEqAbs(expected, x.grad.?.data[0], 1e-6);
}

test "backward - log" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4.0);
    x.setParam();
    const out = x.log();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(std.math.log(f32, std.math.e, 4.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), x.grad.?.data[0], 1e-6);
}

test "backward - reshape" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    x.setParam();
    const reshaped = x.reshape(&.{ 3, 2 });
    const out = reshaped.sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, x.grad.?.data);
}

test "backward - transpose" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    x.setParam();
    const transposed = x.transpose();
    const weights = try Tensor(f32).init(a, &.{ 3, 2 });
    weights.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const out = transposed.mul(weights).sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, x.grad.?.data);
}

test "compute max reduction" {
    const t = try Tensor(f32).init(tac, &.{ 2, 3 });
    defer t.deinit();
    t.setData(&.{ 1, 5, 2, 4, 3, 6 });

    const dst = t.max(&.{ 1, 3 });
    defer dst.deinit();
    dst.compute();

    try testing.expectEqualSlices(f32, &.{ 5, 4, 6 }, dst.data);
}

test "compute softmax" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t = try Tensor(f32).init(a, &.{3});
    t.setData(&.{ 1, 2, 3 });
    const s = t.softmax(&.{1});
    try g.buildForward(s);
    g.compute();

    const e1 = std.math.exp(@as(f32, 1));
    const e2 = std.math.exp(@as(f32, 2));
    const e3 = std.math.exp(@as(f32, 3));
    const denom = e1 + e2 + e3;
    try testing.expectApproxEqAbs(e1 / denom, s.data[0], 1e-6);
    try testing.expectApproxEqAbs(e2 / denom, s.data[1], 1e-6);
    try testing.expectApproxEqAbs(e3 / denom, s.data[2], 1e-6);
}

test "compute logSoftmax" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t = try Tensor(f32).init(a, &.{3});
    t.setData(&.{ 1, 2, 3 });
    const ls = t.logSoftmax(&.{1});
    try g.buildForward(ls);
    g.compute();

    const e1 = std.math.exp(@as(f32, 1));
    const e2 = std.math.exp(@as(f32, 2));
    const e3 = std.math.exp(@as(f32, 3));
    const log_denom = std.math.log(f32, std.math.e, e1 + e2 + e3);
    try testing.expectApproxEqAbs(@as(f32, 1) - log_denom, ls.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2) - log_denom, ls.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3) - log_denom, ls.data[2], 1e-6);
}

test "compute gatherRows" {
    const table = try Tensor(f32).init(tac, &.{ 3, 4 });
    defer table.deinit();
    table.setData(&.{
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
    });

    const indices = try Tensor(f32).init(tac, &.{3});
    defer indices.deinit();
    indices.setData(&.{ 2, 0, 3 });

    const out = table.gatherRows(indices);
    defer out.deinit();
    out.compute();

    try testing.expectEqualSlices(f32, &.{
        30, 31, 32,
        10, 11, 12,
        40, 41, 42,
    }, out.data);
}

test "backward - gatherRows accumulates repeated indices" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const table = try Tensor(f32).init(a, &.{ 2, 4 });
    table.setData(&.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    table.setParam();

    const indices = try Tensor(f32).init(a, &.{3});
    indices.setData(&.{ 1, 3, 1 });

    const gathered = table.gatherRows(indices);
    const out = gathered.sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectEqualSlices(f32, &.{
        0, 0,
        2, 2,
        0, 0,
        1, 1,
    }, table.grad.?.data);
}

test "compute pickRows" {
    const logits = try Tensor(f32).init(tac, &.{ 4, 3 });
    defer logits.deinit();
    logits.setData(&.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });

    const indices = try Tensor(f32).init(tac, &.{3});
    defer indices.deinit();
    indices.setData(&.{ 3, 0, 2 });

    const picked = logits.pickRows(indices);
    defer picked.deinit();
    picked.compute();

    try testing.expectEqualSlices(f32, &.{ 4, 5, 11 }, picked.data);
}

test "backward - pickRows" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 4, 3 });
    logits.setData(&.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });
    logits.setParam();

    const indices = try Tensor(f32).init(a, &.{3});
    indices.setData(&.{ 3, 0, 2 });

    const picked = logits.pickRows(indices);
    const out = picked.sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectEqualSlices(f32, &.{
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 1, 0,
    }, logits.grad.?.data);
}

test "compute gatherRowsIdx" {
    const IndexTensor = @import("index.zig").IndexTensor;

    const table = try Tensor(f32).init(tac, &.{ 2, 4 });
    defer table.deinit();
    table.setData(&.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });

    const indices = try IndexTensor(i32).initCopy(tac, &.{ 3, 1 });
    defer indices.deinit(tac);

    const out = table.gatherRowsIdx(indices);
    defer out.deinit();
    out.compute();

    try testing.expectEqualSlices(f32, &.{ 7, 8, 3, 4 }, out.data);
}

test "compute pickRowsIdx" {
    const IndexTensor = @import("index.zig").IndexTensor;

    const logits = try Tensor(f32).init(tac, &.{ 4, 3 });
    defer logits.deinit();
    logits.setData(&.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });

    const indices = try IndexTensor(i32).initCopy(tac, &.{ 3, 0, 2 });
    defer indices.deinit(tac);

    const out = logits.pickRowsIdx(indices);
    defer out.deinit();
    out.compute();

    try testing.expectEqualSlices(f32, &.{ 4, 5, 11 }, out.data);
}

test "backward - gatherRowsIdx accumulates repeated indices" {
    const IndexTensor = @import("index.zig").IndexTensor;

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const table = try Tensor(f32).init(a, &.{ 2, 4 });
    table.setData(&.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    table.setParam();

    const indices = try IndexTensor(i32).initCopy(a, &.{ 1, 3, 1 });
    const gathered = table.gatherRowsIdx(indices);
    const out = gathered.sumAll();
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectEqualSlices(f32, &.{
        0, 0,
        2, 2,
        0, 0,
        1, 1,
    }, table.grad.?.data);
}
