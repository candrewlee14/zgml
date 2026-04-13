const std = @import("std");
const testing = std.testing;
pub const Tensor = @import("tensor.zig").Tensor;
pub const ComputeGraph = @import("graph.zig").ComputeGraph;

pub const models = @import("models.zig");
pub const optim = @import("optim.zig");

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(models);
    _ = testing.refAllDeclsRecursive(optim);
}
