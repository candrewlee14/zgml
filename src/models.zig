const std = @import("std");
const testing = std.testing;
pub const Linear = @import("models/linear.zig").Model;
pub const Poly = @import("models/poly.zig").Model;

test "ref all decls" {
    _ = testing.refAllDecls(Linear(f32));
    _ = testing.refAllDecls(Poly(f32));
}
