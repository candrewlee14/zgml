//! Loss functions for training.

const std = @import("std");
const Alloc = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

/// Mean Squared Error: `mean((x - y)^2)`.
///
/// Returns a scalar tensor representing the loss. Both `x` and `y` must have
/// the same shape. The returned tensor participates in the computation graph
/// and supports backpropagation.
pub fn meanSqErr(comptime T: type, alloc: Alloc, x: *Tensor(T), y: *Tensor(T)) *Tensor(T) {
    return x.sub(alloc, y).sqr(alloc).mean(alloc, &.{1});
}
