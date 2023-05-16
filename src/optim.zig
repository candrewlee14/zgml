const std = @import("std");
pub const adam = @import("optim/adam.zig");
pub const sgd = @import("optim/sgd.zig");

fn implementsOptimizer(comptime Optimizer: type) bool {
    return std.meta.trait.hasFn("init")(Optimizer) and
        std.meta.trait.hasFn("deinit")(Optimizer) and
        std.meta.trait.hasFn("step")(Optimizer) and
        std.meta.trait.hasFn("zeroGrad")(Optimizer);
}

test "sgd impls optim" {
    const Optimizer = sgd.SGD(f32);
    try std.testing.expect(implementsOptimizer(Optimizer));
}
