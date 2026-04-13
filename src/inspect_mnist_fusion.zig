const std = @import("std");
const zgml = @import("main.zig");
const Op = @import("op.zig").Op;
const fused = @import("tensor/fused.zig");

const Tensor = zgml.Tensor;
const ConvClassifier = zgml.models.ConvClassifier;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, 32);
    defer model.deinit();
    try model.g.fusionPass();

    // Op histogram
    var op_counts = std.AutoHashMap(Op, usize).init(alloc);
    defer op_counts.deinit();
    for (model.g.nodes.items) |node| {
        const entry = try op_counts.getOrPut(node.opTag());
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var it = op_counts.iterator();
    while (it.next()) |e| {
        std.debug.print("op {s}: {}\n", .{ @tagName(e.key_ptr.*), e.value_ptr.* });
    }

    // Fusion report
    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);
    try model.g.dumpReport(&stderr_writer.interface, .{ .include_nodes = true, .include_execution = true });
    try stderr_writer.interface.flush();

    // Timing
    _ = model.xs_batch.setAllScalar(1);
    _ = model.ys_batch.setAllScalar(0);
    if (model.loss.grad) |grad| _ = grad.setAllScalar(1);

    const first = try model.g.profileExecution(.{ .loss_grad = model.loss.grad });
    const second = try model.g.profileExecution(.{ .loss_grad = model.loss.grad });
    std.debug.print("first run\n", .{});
    try first.dump(&stderr_writer.interface);
    std.debug.print("second run\n", .{});
    try second.dump(&stderr_writer.interface);
    try stderr_writer.interface.flush();
}
