const std = @import("std");
const zgml = @import("main.zig");
const Op = @import("op.zig").Op;
const fused = @import("tensor/fused.zig");

const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const loss_mod = zgml.loss;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var g = ComputeGraph(f32).init(alloc);
    defer g.deinit();
    const a = g.allocator();

    const batch_size = 32;
    const k1 = try Tensor(f32).init(a, &.{ 5, 5, 1, 8 });
    const b1 = try Tensor(f32).init(a, &.{8});
    const fc_w = try Tensor(f32).init(a, &.{ 10, 12 * 12 * 8 });
    const fc_b = try Tensor(f32).init(a, &.{10});
    for ([_]*Tensor(f32){ k1, b1, fc_w, fc_b }) |p| p.setParam();

    const xs_batch = try Tensor(f32).init(a, &.{ 28, 28, 1, batch_size });
    const ys_batch = try Tensor(f32).init(a, &.{batch_size});

    const conv_out = xs_batch.conv2d(k1);
    const b1_4d = b1.reshape(&.{ 1, 1, 8, 1 });
    const b1_bc = b1_4d.broadcastTo(conv_out.ne[0..conv_out.n_dims]);
    const act1 = conv_out.add(b1_bc).relu();
    const pool1 = act1.maxPool2d();
    const flat = pool1.reshape(&.{ 12 * 12 * 8, batch_size });
    const fc_out = flat.matMul(false, fc_w, false);
    const fb_bc = fc_b.broadcastTo(fc_out.ne[0..fc_out.n_dims]);
    const logits = fc_out.add(fb_bc);
    const loss = loss_mod.crossEntropy(f32, logits, ys_batch);

    try g.buildForward(loss);
    try g.buildBackward(true);
    try g.fusionPass();

    std.debug.print("match bwd kernel @86 = {}\n", .{g.debugMatchesConv2dBwdKernelAt(86)});

    var op_counts = std.AutoHashMap(Op, usize).init(alloc);
    defer op_counts.deinit();
    for (g.nodes.items) |node| {
        const entry = try op_counts.getOrPut(node.opTag());
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var it = op_counts.iterator();
    while (it.next()) |e| {
        std.debug.print("op {s}: {}\n", .{ @tagName(e.key_ptr.*), e.value_ptr.* });
    }

    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);
    try g.dumpFusionReport(&stderr_writer.interface);
    try g.dumpTensorLineage(&stderr_writer.interface, k1.grad.?);
    try g.dumpTensorLineage(&stderr_writer.interface, b1.grad.?);
    try stderr_writer.interface.flush();

    for (86..93) |i| {
        const node = g.nodes.items[i];
        const s0 = if (node.source0()) |src| fused.indexOfNodeMaybe(f32, g.nodes.items, src) else null;
        const s1 = if (node.source1()) |src| fused.indexOfNodeMaybe(f32, g.nodes.items, src) else null;
        std.debug.print("detail node[{d}] {s} src0={any} src1={any}\n", .{ i, @tagName(node.opTag()), s0, s1 });
    }

    _ = xs_batch.setAllScalar(1);
    _ = ys_batch.setAllScalar(0);
    if (loss.grad) |grad| _ = grad.setAllScalar(1);

    var timer = try std.time.Timer.start();
    g.compute();
    const first_ns = timer.read();
    timer.reset();
    g.reset();
    g.resetGrads();
    if (loss.grad) |grad| _ = grad.setAllScalar(1);
    g.compute();
    const second_ns = timer.read();
    std.debug.print("compute1_ms={d:.3} compute2_ms={d:.3}\n", .{ @as(f64, @floatFromInt(first_ns)) / 1e6, @as(f64, @floatFromInt(second_ns)) / 1e6 });
}
