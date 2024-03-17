const std = @import("std");
pub const adam = @import("optim/adam.zig");
pub const sgd = @import("optim/sgd.zig");

fn implementsOptimizer(comptime Optimizer: type) bool {
    return std.meta.hasFn(Optimizer, "init") and
        std.meta.hasFn(Optimizer, "deinit") and
        std.meta.hasFn(Optimizer, "step") and
        std.meta.hasFn(Optimizer, "zeroGrad");
}

test "sgd impls optim" {
    const Optimizer = sgd.SGD(f32);
    try std.testing.expect(implementsOptimizer(Optimizer));
}
