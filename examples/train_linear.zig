const std = @import("std");
const zgml = @import("zgml");

// Native helper smoke. Product-level model/training policy belongs in src/ts/**.
pub fn main() !void {
    var layer = try zgml.nn.Linear(f32).init(std.heap.page_allocator, 2, 1, .{});
    layer.weight.setData(&.{ 0.05, -0.05 });
    layer.bias.?.setData(&.{0});

    var model = try zgml.nn.Sequential(f32).init(std.heap.page_allocator, .{layer});
    defer model.deinit();

    const params = model.parameters();
    var opt = try zgml.optim.SGD(f32).init(std.heap.page_allocator, params, .{ .lr = 0.05 });
    defer opt.deinit();

    var g = zgml.ComputeGraph(f32).init(std.heap.page_allocator);
    defer g.deinit();

    const x = try g.tensor(&.{ 2, 4 });
    x.setData(&.{
        0, 1,
        1, 1,
        2, 2,
        3, 2,
    });
    const y = try g.tensor(&.{ 1, 4 });
    y.setData(&.{ -2, 0, -1, 1 });

    const pred = model.forward(x);
    const loss = zgml.loss.meanSquaredError(f32, pred, y);

    for (0..1200) |_| {
        try zgml.train.step(f32, &g, loss, &opt);
    }

    try zgml.train.backward(f32, &g, loss);
    std.debug.print("loss={d:.6} w=[{d:.3}, {d:.3}] b={d:.3}\n", .{ loss.data[0], params[0].data[0], params[0].data[1], params[1].data[0] });
    std.debug.assert(loss.data[0] < 1e-3);
}
