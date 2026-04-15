const std = @import("std");
const zgml = @import("zgml");

const ConvClassifier = zgml.models.ConvClassifier;
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const nn = zgml.nn;
const loss = zgml.loss;

const Config = struct {
    name: []const u8,
    in_w: usize,
    in_h: usize,
    kernel_size: usize,
    n_filters: usize,
    n_hidden: usize = 0,
    n_classes: usize,
    batch_size: usize,
    deep: bool = false,
    two_conv: bool = false,
    warmup_iters: usize = 3,
    bench_iters: usize = 30,
};

const BenchModel = struct {
    params: []const *Tensor(f32),
    backing_alloc: std.mem.Allocator,
    xs_batch: *Tensor(f32),
    ys_batch: *Tensor(f32),
    loss: *Tensor(f32),
    g: ComputeGraph(f32),

    fn deinit(self: *BenchModel) void {
        self.backing_alloc.free(self.params);
        self.g.deinit();
    }
};

fn buildTwoConvModel(alloc: std.mem.Allocator, cfg: Config) !BenchModel {
    var g = ComputeGraph(f32).init(alloc);
    const a = g.allocator();

    const conv1_k = try g.param(&.{ 3, 3, 1, cfg.n_filters });
    const conv1_b = try g.param(&.{cfg.n_filters});
    const conv2_k = try g.param(&.{ 3, 3, cfg.n_filters, cfg.n_filters });
    const conv2_b = try g.param(&.{cfg.n_filters});

    const conv1_w = cfg.in_w - 3 + 1;
    const conv1_h = cfg.in_h - 3 + 1;
    const conv2_w = conv1_w - 3 + 1;
    const conv2_h = conv1_h - 3 + 1;
    const pool_w = conv2_w / 2;
    const pool_h = conv2_h / 2;
    const flat_dim = pool_w * pool_h * cfg.n_filters;

    const fc_w = try g.param(&.{ cfg.n_classes, flat_dim });
    const fc_b = try g.param(&.{cfg.n_classes});

    nn.kaimingUniform(f32, conv1_k, 42);
    nn.kaimingUniform(f32, conv2_k, 43);
    nn.kaimingUniform(f32, fc_w, 44);

    const params = try alloc.dupe(*Tensor(f32), &.{ conv1_k, conv1_b, conv2_k, conv2_b, fc_w, fc_b });

    const xs_batch = try Tensor(f32).init(a, &.{ cfg.in_w, cfg.in_h, 1, cfg.batch_size });
    const ys_batch = try Tensor(f32).init(a, &.{cfg.batch_size});

    const conv1 = xs_batch.conv2d(conv1_k);
    const conv1_b4 = conv1_b.reshape(&.{ 1, 1, cfg.n_filters, 1 });
    var conv1_ne = [_]usize{ conv1.ne[0], conv1.ne[1], conv1.ne[2], conv1.ne[3] };
    const act1 = conv1.add(conv1_b4.repeat(conv1_ne[0..])).relu();

    const conv2 = act1.conv2d(conv2_k);
    const conv2_b4 = conv2_b.reshape(&.{ 1, 1, cfg.n_filters, 1 });
    var conv2_ne = [_]usize{ conv2.ne[0], conv2.ne[1], conv2.ne[2], conv2.ne[3] };
    const act2 = conv2.add(conv2_b4.repeat(conv2_ne[0..])).relu();

    const flat = act2.maxPool2d().reshape(&.{ flat_dim, cfg.batch_size });
    const logits = nn.linear(f32, flat, fc_w, fc_b);
    const loss_node = loss.crossEntropy(f32, logits, ys_batch);

    try g.buildForward(loss_node);
    try g.buildBackward(false);

    return .{
        .params = params,
        .backing_alloc = alloc,
        .xs_batch = xs_batch,
        .ys_batch = ys_batch,
        .loss = loss_node,
        .g = g,
    };
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn runCase(alloc: std.mem.Allocator, writer: anytype, cfg: Config) !void {
    var model = if (cfg.two_conv)
        try buildTwoConvModel(alloc, cfg)
    else if (cfg.deep) blk: {
        var m = try ConvClassifier(f32).buildDeep(alloc, cfg.in_w, cfg.in_h, cfg.n_filters, cfg.n_hidden, cfg.n_classes, cfg.batch_size);
        break :blk BenchModel{ .params = m.params(), .backing_alloc = alloc, .xs_batch = m.xs_batch, .ys_batch = m.ys_batch, .loss = m.loss, .g = m.g };
    } else blk: {
        var m = try ConvClassifier(f32).build(alloc, cfg.in_w, cfg.in_h, cfg.kernel_size, cfg.n_filters, cfg.n_classes, cfg.batch_size);
        break :blk BenchModel{ .params = m.params(), .backing_alloc = alloc, .xs_batch = m.xs_batch, .ys_batch = m.ys_batch, .loss = m.loss, .g = m.g };
    };
    defer model.deinit();
    try model.g.fusionPass();
    model.g.enableThreading() catch {};

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (model.xs_batch.data) |*v| v.* = rand.float(f32);
    for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, @intCast(cfg.n_classes - 1)));

    var sgd = try zgml.optim.sgd.SGD(f32).init(alloc, model.params, .{ .lr = 0.01, .momentum = 0.9 });
    defer sgd.deinit();

    for (0..cfg.warmup_iters) |_| {
        model.g.reset();
        model.g.resetGrads();
        if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
        model.g.compute();
        sgd.step();
    }

    var total_ns: u64 = 0;
    var forward_ns: u64 = 0;
    var backward_ns: u64 = 0;
    var fwd_im2col_ns: u64 = 0;
    var fwd_gemm_ns: u64 = 0;
    var fwd_epilogue_ns: u64 = 0;
    var bwd_input_rearrange_ns: u64 = 0;
    var bwd_input_gemm_ns: u64 = 0;
    var bwd_input_col2im_ns: u64 = 0;
    var bwd_kernel_im2col_ns: u64 = 0;
    var bwd_kernel_rearrange_ns: u64 = 0;
    var bwd_kernel_gemm_ns: u64 = 0;

    for (0..cfg.bench_iters) |_| {
        const profile = try model.g.profileExecution(.{ .loss_grad = model.loss.grad });
        total_ns += profile.total_ns;
        forward_ns += profile.forward_ns;
        backward_ns += profile.backward_ns;
        fwd_im2col_ns += profile.fused_conv_phases.fwd_im2col_ns;
        fwd_gemm_ns += profile.fused_conv_phases.fwd_gemm_ns;
        fwd_epilogue_ns += profile.fused_conv_phases.fwd_epilogue_ns;
        bwd_input_rearrange_ns += profile.fused_conv_phases.bwd_input_rearrange_ns;
        bwd_input_gemm_ns += profile.fused_conv_phases.bwd_input_gemm_ns;
        bwd_input_col2im_ns += profile.fused_conv_phases.bwd_input_col2im_ns;
        bwd_kernel_im2col_ns += profile.fused_conv_phases.bwd_kernel_im2col_ns;
        bwd_kernel_rearrange_ns += profile.fused_conv_phases.bwd_kernel_rearrange_ns;
        bwd_kernel_gemm_ns += profile.fused_conv_phases.bwd_kernel_gemm_ns;
        sgd.step();
    }

    const iters_f = @as(f64, @floatFromInt(cfg.bench_iters));
    try writer.print("{s}\n", .{cfg.name});
    try writer.print("  total mean:    {d:>7.3} ms\n", .{nsToMs(total_ns) / iters_f});
    try writer.print("  forward mean:  {d:>7.3} ms\n", .{nsToMs(forward_ns) / iters_f});
    try writer.print("  backward mean: {d:>7.3} ms\n", .{nsToMs(backward_ns) / iters_f});
    try writer.print("  conv fwd:      im2col={d:>7.3} gemm={d:>7.3} epilogue={d:>7.3} ms\n", .{
        nsToMs(fwd_im2col_ns) / iters_f,
        nsToMs(fwd_gemm_ns) / iters_f,
        nsToMs(fwd_epilogue_ns) / iters_f,
    });
    try writer.print("  conv bwd-in:   rearrange={d:>7.3} gemm={d:>7.3} col2im={d:>7.3} ms\n", .{
        nsToMs(bwd_input_rearrange_ns) / iters_f,
        nsToMs(bwd_input_gemm_ns) / iters_f,
        nsToMs(bwd_input_col2im_ns) / iters_f,
    });
    try writer.print("  conv bwd-k:    im2col={d:>7.3} rearrange={d:>7.3} gemm={d:>7.3} ms\n", .{
        nsToMs(bwd_kernel_im2col_ns) / iters_f,
        nsToMs(bwd_kernel_rearrange_ns) / iters_f,
        nsToMs(bwd_kernel_gemm_ns) / iters_f,
    });
    try writer.print("\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const stdout_file = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = stdout_file.writer(&buf);

    try runCase(alloc, &w.interface, .{
        .name = "MNIST-scale conv",
        .in_w = 28,
        .in_h = 28,
        .kernel_size = 5,
        .n_filters = 8,
        .n_classes = 10,
        .batch_size = 32,
    });
    try runCase(alloc, &w.interface, .{
        .name = "Larger conv",
        .in_w = 64,
        .in_h = 64,
        .kernel_size = 3,
        .n_filters = 32,
        .n_classes = 100,
        .batch_size = 32,
    });
    try runCase(alloc, &w.interface, .{
        .name = "Two-conv stack",
        .in_w = 64,
        .in_h = 64,
        .kernel_size = 3,
        .n_filters = 32,
        .n_classes = 100,
        .batch_size = 32,
        .two_conv = true,
    });
    w.interface.flush() catch {};
}
