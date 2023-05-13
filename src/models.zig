const std = @import("std");
const testing = std.testing;
pub const Quadratic = @import("models/quad.zig").Model;

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(Quadratic(f64));
}
