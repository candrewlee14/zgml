const std = @import("std");
const Alloc = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

pub fn meanSqErr(comptime T: type, x: *Tensor(T), y: *Tensor(T)) Alloc.Error!*Tensor(T) {
    const diff = try x.sub(y);
    const diff2 = try diff.sqr();
    return try diff2.mean(&.{1});
}
