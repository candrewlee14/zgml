const std = @import("std");
const zgml = @import("zgml");

const ConvClassifier = zgml.models.ConvClassifier;

const Config = struct {
    name: []const u8,
    in_w: usize,
    in_h: usize,
    kernel_size: usize,
    n_filters: usize,
    n_classes: usize,
    batch_size: usize,
    warmup_iters: usize = 3,
    bench_iters: usize = 30,
};

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn runCase(alloc: std.mem.Allocator, writer: anytype, cfg: Config) !void {
    var model = try ConvClassifier(f32).build(alloc, cfg.in_w, cfg.in_h, cfg.kernel_size, cfg.n_filters, cfg.n_classes, cfg.batch_size);
    defer model.deinit();
    try model.g.fusionPass();
    model.g.enableThreading() catch {};

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (model.xs_batch.data) |*v| v.* = rand.float(f32);
    for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, @intCast(cfg.n_classes - 1)));

    const p = model.params();
    var sgd = try zgml.optim.sgd.SGD(f32).init(alloc, p, .{ .lr = 0.01, .momentum = 0.9 });
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
    w.interface.flush() catch {};
}
