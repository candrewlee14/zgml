//! Native loss helpers over Tensor primitives.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Alloc = std.mem.Allocator;

pub const native_helper_manifest = .{
    .kind = "zgml-native-helper",
    .domain = "loss",
    .product_frontend = false,
    .product_owner = "src/ts/**",
    .product_policy = "forbidden",
    .role = "native-helper-substrate",
    .alignment_boundary = "Program/Session/ABI contracts",
};

/// Mean squared error reduced to a scalar loss.
pub fn meanSquaredError(comptime T: type, pred: *Tensor(T), target: *Tensor(T)) *Tensor(T) {
    return pred.sub(target).sqr().mean(&.{1});
}

/// Short PyTorch-style alias for mean squared error.
pub fn mse(comptime T: type, pred: *Tensor(T), target: *Tensor(T)) *Tensor(T) {
    return meanSquaredError(T, pred, target);
}

/// Cross-entropy over per-row class logits.
///
/// `logits` has shape `{n_classes, batch}` and `targets` has shape `{batch}`,
/// where each target is the class index for that sample.
pub fn crossEntropy(comptime T: type, logits: *Tensor(T), targets: *Tensor(T)) *Tensor(T) {
    std.debug.assert(logits.isMatrix());
    std.debug.assert(targets.isVector());
    std.debug.assert(logits.ne[1] == targets.ne[0]);

    const log_probs = logits.logSoftmax(&.{ 1, logits.ne[1] });
    return log_probs.pickRows(targets).neg().mean(&.{1});
}

/// Build typed class-index targets for `crossEntropy`.
pub fn classTargets(comptime T: type, alloc: Alloc, classes: []const usize) Alloc.Error!*Tensor(T) {
    return Tensor(T).initIndexVectorCopy(alloc, classes);
}

const testing = std.testing;
const tac = testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

test "loss native helper surface stays curated" {
    const root = @This();
    const expected = .{
        "native_helper_manifest",
        "meanSquaredError",
        "mse",
        "crossEntropy",
        "classTargets",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
    try testing.expectEqualStrings("zgml-native-helper", native_helper_manifest.kind);
    try testing.expectEqualStrings("loss", native_helper_manifest.domain);
    try testing.expectEqual(false, native_helper_manifest.product_frontend);
    try testing.expectEqualStrings("src/ts/**", native_helper_manifest.product_owner);
    try testing.expectEqualStrings("forbidden", native_helper_manifest.product_policy);
    try testing.expectEqualStrings("native-helper-substrate", native_helper_manifest.role);
    try testing.expectEqualStrings("Program/Session/ABI contracts", native_helper_manifest.alignment_boundary);
}

test "mse alias composes differentiable tensor primitives" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const pred = try Tensor(f32).init(a, &.{ 1, 2 });
    pred.setData(&.{ 2, -1 });
    pred.setParam();
    const target = try Tensor(f32).init(a, &.{ 1, 2 });
    target.setData(&.{ 1, 1 });

    const loss = mse(f32, pred, target);
    try g.run(loss);

    try testing.expectApproxEqAbs(@as(f32, 2.5), loss.data[0], 1e-6);
    try testing.expectEqualSlices(f32, &.{ 1, -2 }, pred.grad.?.data);
}

test "crossEntropy delegates class-index loss shape" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 1 });
    logits.setData(&.{ 2, 0, 1 });
    logits.setParam();
    const targets = try classTargets(f32, a, &.{0});

    const loss = crossEntropy(f32, logits, targets);
    try g.run(loss);

    try testing.expect(loss.data[0] > 0);
    for (logits.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
