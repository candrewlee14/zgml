//! Loss functions for training.

const std = @import("std");
const Alloc = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const testing = std.testing;
const tac = testing.allocator;

/// Mean Squared Error: `mean((x - y)^2)`.
///
/// Returns a scalar tensor representing the loss. Both `x` and `y` must have
/// the same shape. The returned tensor participates in the computation graph
/// and supports backpropagation.
pub fn meanSqErr(comptime T: type, alloc: Alloc, x: *Tensor(T), y: *Tensor(T)) *Tensor(T) {
    return x.sub(alloc, y).sqr(alloc).mean(alloc, &.{1});
}

/// Cross-entropy over per-row class logits.
///
/// `logits` has shape `{n_classes, batch}` and `targets` has shape `{batch}`,
/// where each target is the class index for that row/sample.
pub fn crossEntropy(comptime T: type, alloc: Alloc, logits: *Tensor(T), targets: *Tensor(T)) *Tensor(T) {
    std.debug.assert(logits.isMatrix());
    std.debug.assert(targets.isVector());
    std.debug.assert(logits.ne[1] == targets.ne[0]);

    const reduce_ne = [_]usize{ 1, logits.ne[1] };
    const log_probs = logits.logSoftmax(alloc, &reduce_ne);
    const picked = log_probs.pickRows(alloc, targets);
    return picked.neg(alloc).mean(alloc, &.{1});
}

test "crossEntropy forward and backward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });
    logits.setParam(a);

    const targets = try Tensor(f32).init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const ce = crossEntropy(f32, a, logits, targets);
    try g.buildForward(ce);
    try g.buildBackward(false);
    _ = ce.grad.?.setAllScalar(1);
    g.compute();

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, ce.data[0], 1e-5);

    for (logits.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
