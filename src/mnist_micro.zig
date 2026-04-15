//! MNIST CNN training micro-benchmark.
//!
//! Isolates the forward + backward + optimizer step hotloop with synthetic
//! data. No file I/O, no accuracy tracking — just the compute path.
//!
//! Run with: zig build mnist-micro

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const nn = zgml.nn;
const ConvClassifier = zgml.models.ConvClassifier;
const time_compat = zgml.time_compat;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const batch_size: usize = 32;
    const warmup_iters: usize = 5;
    const bench_iters: usize = 100;

    // Simple model: Conv(5x5, 1->8) -> ReLU -> MaxPool(2x2) -> FC(1152->10)
    var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, batch_size);
    defer model.deinit();
    try model.g.fusionPass();

    // Enable threading if available
    model.g.enableThreading() catch {};

    const p = model.params();
    var sgd = try zgml.optim.sgd.SGD(f32).init(alloc, p, .{ .lr = 0.01, .momentum = 0.9 });
    defer sgd.deinit();

    // Fill with deterministic synthetic data
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (model.xs_batch.data) |*v| v.* = rand.float(f32);
    for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, 9));

    const stdout_file = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &buf);

    try w.interface.print("\nMNIST CNN Micro-Benchmark (batch_size={})\n", .{batch_size});
    try w.interface.print("==========================================\n\n", .{});

    // Print graph and fusion summary
    try model.g.dumpReport(&w.interface, .{ .include_nodes = false, .include_execution = true });

    // Warmup
    for (0..warmup_iters) |_| {
        model.g.reset();
        model.g.resetGrads();
        if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
        model.g.compute();
        sgd.step();
    }

    // Benchmark with phase breakdown
    const Phase = struct {
        reset: u64,
        reset_grads: u64,
        seed_loss_grad: u64,
        forward: u64,
        backward: u64,
        optim: u64,
        total: u64,
    };
    var times: [bench_iters]Phase = undefined;

    for (&times) |*t| {
        const profile = try model.g.profileExecution(.{ .loss_grad = model.loss.grad });
        t.reset = profile.reset_ns;
        t.reset_grads = profile.reset_grads_ns;
        t.seed_loss_grad = profile.seed_loss_grad_ns;
        t.forward = profile.forward_ns;
        t.backward = profile.backward_ns;

        var timer = time_compat.Timer.start();
        timer.reset();
        sgd.step();
        t.optim = timer.read();

        t.total = profile.total_ns + t.optim;
    }

    // Stats
    var totals: [bench_iters]u64 = undefined;
    var reset_times: [bench_iters]u64 = undefined;
    var reset_grad_times: [bench_iters]u64 = undefined;
    var seed_times: [bench_iters]u64 = undefined;
    var fwd_times: [bench_iters]u64 = undefined;
    var bwd_times: [bench_iters]u64 = undefined;
    for (0..bench_iters) |i| {
        totals[i] = times[i].total;
        reset_times[i] = times[i].reset;
        reset_grad_times[i] = times[i].reset_grads;
        seed_times[i] = times[i].seed_loss_grad;
        fwd_times[i] = times[i].forward;
        bwd_times[i] = times[i].backward;
    }
    std.mem.sort(u64, &totals, {}, std.sort.asc(u64));
    std.mem.sort(u64, &reset_times, {}, std.sort.asc(u64));
    std.mem.sort(u64, &reset_grad_times, {}, std.sort.asc(u64));
    std.mem.sort(u64, &seed_times, {}, std.sort.asc(u64));
    std.mem.sort(u64, &fwd_times, {}, std.sort.asc(u64));
    std.mem.sort(u64, &bwd_times, {}, std.sort.asc(u64));

    const p50 = totals[bench_iters / 2];
    const p90 = totals[bench_iters * 9 / 10];
    const p99 = totals[bench_iters * 99 / 100];
    var total_sum: u64 = 0;
    for (totals) |t| total_sum += t;

    w.interface.flush() catch {};
    try w.interface.print("Results ({} iterations, {} warmup):\n", .{ bench_iters, warmup_iters });
    try w.interface.print("  total:    min={d:>7.2}  p50={d:>7.2}  p90={d:>7.2}  p99={d:>7.2}  mean={d:>7.2} ms\n", .{
        ns_to_ms(totals[0]), ns_to_ms(p50), ns_to_ms(p90), ns_to_ms(p99), ns_to_ms(total_sum / bench_iters),
    });
    try w.interface.print("  reset:    min={d:>7.2}  p50={d:>7.2} ms\n", .{
        ns_to_ms(reset_times[0]), ns_to_ms(reset_times[bench_iters / 2]),
    });
    try w.interface.print("  zeroGrad: min={d:>7.2}  p50={d:>7.2} ms\n", .{
        ns_to_ms(reset_grad_times[0]), ns_to_ms(reset_grad_times[bench_iters / 2]),
    });
    try w.interface.print("  seedLoss: min={d:>7.2}  p50={d:>7.2} ms\n", .{
        ns_to_ms(seed_times[0]), ns_to_ms(seed_times[bench_iters / 2]),
    });
    try w.interface.print("  forward:  min={d:>7.2}  p50={d:>7.2} ms\n", .{
        ns_to_ms(fwd_times[0]), ns_to_ms(fwd_times[bench_iters / 2]),
    });
    try w.interface.print("  backward: min={d:>7.2}  p50={d:>7.2} ms\n", .{
        ns_to_ms(bwd_times[0]), ns_to_ms(bwd_times[bench_iters / 2]),
    });
    try w.interface.print("  throughput: {d:.0} img/s (at p50)\n\n", .{
        @as(f64, @floatFromInt(batch_size)) / (ns_to_ms(p50) / 1000.0),
    });
    const phase_profile = (try model.g.profileExecution(.{ .loss_grad = model.loss.grad })).fused_conv_phases;
    try w.interface.print("Conv phases (single profiled step, ms):\n", .{});
    try w.interface.print("  fwd im2col:      {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.fwd_im2col_ns)) / 1_000_000.0});
    try w.interface.print("  fwd gemm:        {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.fwd_gemm_ns)) / 1_000_000.0});
    try w.interface.print("  fwd epilogue:    {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.fwd_epilogue_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-in rearrange:{d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_input_rearrange_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-in gemm:     {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_input_gemm_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-in col2im:   {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_input_col2im_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-k im2col:    {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_kernel_im2col_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-k rearrange: {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_kernel_rearrange_ns)) / 1_000_000.0});
    try w.interface.print("  bwd-k gemm:      {d:>7.3}\n", .{@as(f64, @floatFromInt(phase_profile.bwd_kernel_gemm_ns)) / 1_000_000.0});
    try w.interface.print("  loss after bench: {d:.4}\n\n", .{model.loss.data[0]});
    w.interface.flush() catch {};
}

fn ns_to_ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
