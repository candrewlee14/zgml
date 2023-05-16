const std = @import("std");
const Alloc = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

pub fn meanSqErr(comptime T: type, x: *Tensor(T), y: *Tensor(T)) *Tensor(T) {
    return x.sub(y).sqr().mean(&.{1});
}
