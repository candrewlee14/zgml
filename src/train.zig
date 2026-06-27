//! Native training-loop helpers over ComputeGraph, Tensor losses, and optimizers.

const std = @import("std");
const ComputeGraph = @import("graph.zig").ComputeGraph;
const Tensor = @import("tensor.zig").Tensor;

pub const native_helper_manifest = .{
    .kind = "zgml-native-helper",
    .domain = "train",
    .product_frontend = false,
    .product_owner = "src/ts/**",
    .product_policy = "required-core",
    .role = "native-helper-substrate",
    .alignment_boundary = "JS/TS API -> Zig C ABI -> Program/Session kernels",
};

/// Run forward/backward for a scalar loss.
///
/// This is a native helper over `ComputeGraph.run`, so low-level examples and
/// bindings do not need to expose graph internals for the common case.
pub fn backward(comptime T: type, graph: *ComputeGraph(T), loss: *Tensor(T)) !void {
    try graph.run(loss);
}

/// Run one training step: backward pass, optimizer update, then gradient clear.
pub fn step(comptime T: type, graph: *ComputeGraph(T), loss: *Tensor(T), optimizer: anytype) !void {
    try backward(T, graph, loss);
    optimizer.step();
    optimizer.zeroGrad();
}

const testing = std.testing;
const tac = testing.allocator;
const nn = @import("nn.zig");
const loss_mod = @import("loss.zig");
const optim = @import("optim.zig");

test "train native helper surface stays curated" {
    const root = @This();
    const expected = .{
        "native_helper_manifest",
        "backward",
        "step",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
    try testing.expectEqualStrings("zgml-native-helper", native_helper_manifest.kind);
    try testing.expectEqualStrings("train", native_helper_manifest.domain);
    try testing.expectEqual(false, native_helper_manifest.product_frontend);
    try testing.expectEqualStrings("src/ts/**", native_helper_manifest.product_owner);
    try testing.expectEqualStrings("required-core", native_helper_manifest.product_policy);
    try testing.expectEqualStrings("native-helper-substrate", native_helper_manifest.role);
    try testing.expectEqualStrings("JS/TS API -> Zig C ABI -> Program/Session kernels", native_helper_manifest.alignment_boundary);
}

test "step updates optimizer and restores durable module gradients after graph teardown" {
    var model = try nn.Sequential(f32).init(tac, .{
        try nn.Linear(f32).init(tac, 2, 1, .{ .seed = 1 }),
    });
    defer model.deinit();

    const params = model.parameters();
    const grad_buf = params[0].paramGradOrNull().?;
    var opt = try optim.SGD(f32).init(tac, params, .{ .lr = 0.05 });
    defer opt.deinit();

    const before = params[0].data[0];
    {
        var g = ComputeGraph(f32).init(tac);
        defer g.deinit();
        const a = g.allocator();

        const x = try Tensor(f32).init(a, &.{ 2, 2 });
        x.setData(&.{ 1, 2, 3, 4 });
        const y = try Tensor(f32).init(a, &.{ 1, 2 });
        y.setData(&.{ 0, 1 });
        const pred = model.forward(x);
        const l = loss_mod.meanSquaredError(f32, pred, y);

        try step(f32, &g, l, &opt);
        try testing.expect(params[0].grad != null);
    }

    try testing.expect(params[0].grad.? == grad_buf);
    try testing.expect(params[0].data[0] != before);
    try testing.expectEqualSlices(f32, &.{ 0, 0 }, grad_buf.data);
}
