const std = @import("std");
pub const mnist = @import("data/mnist.zig");
pub const dataloader = @import("data/dataloader.zig");
pub const DataLoader = dataloader.DataLoader;

fn implementsDataLoader(comptime Loader: type) bool {
    return std.meta.hasFn(Loader, "init") and
        std.meta.hasFn(Loader, "deinit") and
        std.meta.hasFn(Loader, "shuffle") and
        std.meta.hasFn(Loader, "reset") and
        std.meta.hasFn(Loader, "next");
}

test "DataLoader(f32) impls data loader interface" {
    const Loader = DataLoader(f32);
    try std.testing.expect(implementsDataLoader(Loader));
}

test "ref all decls" {
    _ = @import("data/mnist.zig");
    _ = @import("data/dataloader.zig");
}
