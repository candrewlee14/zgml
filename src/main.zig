const std = @import("std");
const testing = std.testing;
pub usingnamespace @import("tensor.zig");
pub usingnamespace @import("graph.zig");

pub const models = @import("models.zig");
pub const optim = @import("optim.zig");

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(models);
    _ = testing.refAllDeclsRecursive(optim);
}
