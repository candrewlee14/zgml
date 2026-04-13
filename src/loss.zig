//! Loss functions for training.

const std = @import("std");
const Alloc = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const IndexTensor = @import("index.zig").IndexTensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const testing = std.testing;
const tac = testing.allocator;

/// Mean Squared Error: `mean((x - y)^2)`.
///
/// Returns a scalar tensor representing the loss. Both `x` and `y` must have
/// the same shape. The returned tensor participates in the computation graph
/// and supports backpropagation.
pub fn meanSqErr(comptime T: type, x: *Tensor(T), y: *Tensor(T)) *Tensor(T) {
    return x.sub(y).sqr().mean(&.{1});
}

/// Cross-entropy over per-row class logits.
///
/// `logits` has shape `{n_classes, batch}` and `targets` has shape `{batch}`,
/// where each target is the class index for that row/sample.
pub fn crossEntropy(comptime T: type, logits: *Tensor(T), targets: *Tensor(T)) *Tensor(T) {
    std.debug.assert(logits.isMatrix());
    std.debug.assert(targets.isVector());
    std.debug.assert(logits.ne[1] == targets.ne[0]);

    const reduce_ne = [_]usize{ 1, logits.ne[1] };
    const log_probs = logits.logSoftmax(&reduce_ne);
    const picked = log_probs.pickRows(targets);
    return picked.neg().mean(&.{1});
}

pub fn crossEntropyIdx(comptime T: type, comptime I: type, logits: *Tensor(T), targets: *IndexTensor(I)) *Tensor(T) {
    std.debug.assert(logits.isMatrix());
    std.debug.assert(logits.ne[1] == targets.nElems());

    const reduce_ne = [_]usize{ 1, logits.ne[1] };
    const log_probs = logits.logSoftmax(&reduce_ne);
    const picked = log_probs.pickRowsIdx(targets);
    return picked.neg().mean(&.{1});
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
    logits.setParam();

    const targets = try Tensor(f32).init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const ce = crossEntropy(f32, logits, targets);
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

test "crossEntropyIdx forward matches tensor target version" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });

    const targets_t = try Tensor(f32).init(a, &.{2});
    targets_t.setData(&.{ 0, 1 });
    const targets_i = try IndexTensor(i32).initCopy(a, &.{ 0, 1 });

    const ce_t = crossEntropy(f32, logits, targets_t);
    const ce_i = crossEntropyIdx(f32, i32, logits, targets_i);
    try g.buildForward(ce_t);
    try g.buildForward(ce_i);
    g.compute();

    try testing.expectApproxEqAbs(ce_t.data[0], ce_i.data[0], 1e-6);
}
