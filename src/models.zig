const std = @import("std");
const testing = std.testing;
pub const Quadratic = @import("models/quad.zig").Model;
pub const Linear = @import("models/linear.zig").Model;

test "ref all decls" {
    _ = testing.refAllDecls(Quadratic(f64));
    _ = testing.refAllDecls(Linear(f32));
}
