const std = @import("std");
const Alloc = std.mem.Allocator;

pub fn IndexTensor(comptime I: type) type {
    return struct {
        const Self = @This();

        data: []I,

        pub fn init(alloc: Alloc, len: usize) Alloc.Error!*Self {
            const self = try alloc.create(Self);
            self.* = .{ .data = try alloc.alloc(I, len) };
            return self;
        }

        pub fn initCopy(alloc: Alloc, vals: []const I) Alloc.Error!*Self {
            const self = try Self.init(alloc, vals.len);
            @memcpy(self.data, vals);
            return self;
        }

        pub fn deinit(self: *Self, alloc: Alloc) void {
            alloc.free(self.data);
            alloc.destroy(self);
        }

        pub fn deinitWithStoredAlloc(self: *Self, alloc: Alloc) void {
            self.deinit(alloc);
        }

        pub fn nElems(self: *const Self) usize {
            return self.data.len;
        }

        pub fn toUsizeOwned(self: *const Self, alloc: Alloc) Alloc.Error![]usize {
            const out = try alloc.alloc(usize, self.data.len);
            for (self.data, 0..) |v, i| {
                const info = @typeInfo(I);
                out[i] = switch (info) {
                    .int, .comptime_int => blk: {
                        if (@typeInfo(I).int.signedness == .signed) std.debug.assert(v >= 0);
                        break :blk @intCast(v);
                    },
                    else => @compileError("IndexTensor only supports integer element types"),
                };
            }
            return out;
        }
    };
}

const testing = std.testing;
const tac = testing.allocator;

test "IndexTensor converts to usize" {
    const idx = try IndexTensor(i32).initCopy(tac, &.{ 2, 0, 3 });
    defer idx.deinit(tac);

    const vals = try idx.toUsizeOwned(tac);
    defer tac.free(vals);

    try testing.expectEqualSlices(usize, &.{ 2, 0, 3 }, vals);
}
