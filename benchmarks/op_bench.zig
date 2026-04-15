//! Per-op microbenchmark for MNIST CNN ops.
//!
//! Uses the library's built-in `profileNodes` to get per-op timing on
//! the full ConvClassifier graph (unfused, so every primitive is visible).
//! Then runs the same profiling with fusion enabled for comparison.
//!
//! Run with: zig build op-bench

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const ConvClassifier = zgml.models.ConvClassifier;

const BATCH = 32;
const WARMUP = 5;
const ITERS = 20;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const stdout_file = std.Io.File.stdout();
    var buf: [16384]u8 = undefined;
    var w = stdout_file.writer(io, &buf);

    try w.interface.print("\nPer-Op Microbenchmark — MNIST CNN (batch={})\n", .{BATCH});
    try w.interface.print("===============================================\n\n", .{});

    // --- Unfused profiling ---
    {
        try w.interface.print("UNFUSED (every primitive op timed separately)\n", .{});
        try w.interface.print("----------------------------------------------\n", .{});

        var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, BATCH);
        defer model.deinit();
        // No fusionPass — we want to see every primitive

        // Fill synthetic data
        var prng = std.Random.DefaultPrng.init(42);
        const rand = prng.random();
        for (model.xs_batch.data) |*v| v.* = rand.float(f32);
        for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, 9));

        // Warmup
        for (0..WARMUP) |_| {
            model.g.reset();
            model.g.resetGrads();
            if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
            model.g.compute();
        }

        // Accumulate across ITERS runs
        const op_count = @typeInfo(zgml.Op).@"enum".fields.len;
        var accum_fwd = [_]u64{0} ** op_count;
        var accum_bwd = [_]u64{0} ** op_count;
        var accum_fwd_n = [_]u64{0} ** op_count;
        var accum_bwd_n = [_]u64{0} ** op_count;

        for (0..ITERS) |_| {
            model.g.reset();
            model.g.resetGrads();

            var np = try model.g.profileNodes(alloc, .{
                .loss_grad = model.loss.grad,
            });
            defer np.deinit();

            for (np.buckets, 0..) |b, i| {
                accum_fwd[i] += b.fwd_ns;
                accum_bwd[i] += b.bwd_ns;
                accum_fwd_n[i] = b.fwd_count; // count is same every iter
                accum_bwd_n[i] = b.bwd_count;
            }
        }

        // Build averaged profile and print
        const avg_buckets = try alloc.alloc(ComputeGraph(f32).OpBucket, op_count);
        defer alloc.free(avg_buckets);
        var avg_fwd_total: u64 = 0;
        var avg_bwd_total: u64 = 0;
        for (0..op_count) |i| {
            avg_buckets[i] = .{
                .tag = @enumFromInt(i),
                .fwd_count = accum_fwd_n[i],
                .bwd_count = accum_bwd_n[i],
                .fwd_ns = accum_fwd[i] / ITERS,
                .bwd_ns = accum_bwd[i] / ITERS,
            };
            avg_fwd_total += avg_buckets[i].fwd_ns;
            avg_bwd_total += avg_buckets[i].bwd_ns;
        }

        const avg_profile = ComputeGraph(f32).NodeProfile{
            .buckets = avg_buckets,
            .fwd_total_ns = avg_fwd_total,
            .bwd_total_ns = avg_bwd_total,
            .alloc = alloc,
        };
        try avg_profile.render(&w.interface);
        try w.interface.print("\n", .{});
        w.interface.flush() catch {};
    }

    // --- Fused per-step profiling ---
    {
        try w.interface.print("FUSED (per-step profiling)\n", .{});
        try w.interface.print("---------------------------------------------\n", .{});

        var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, BATCH);
        defer model.deinit();
        try model.g.fusionPass();

        var prng = std.Random.DefaultPrng.init(42);
        const rand = prng.random();
        for (model.xs_batch.data) |*v| v.* = rand.float(f32);
        for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, 9));

        // Warmup
        for (0..WARMUP) |_| {
            model.g.reset();
            model.g.resetGrads();
            if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
            model.g.compute();
        }

        // Single step profile run for detailed breakdown
        var sp = try model.g.profileSteps(alloc, .{ .loss_grad = model.loss.grad });
        defer sp.deinit();
        try sp.renderDetailed(&w.interface, 100.0); // show detail for steps > 100 us

        // Whole-graph timing (multiple iterations)
        var fwd_times: [ITERS]u64 = undefined;
        var bwd_times: [ITERS]u64 = undefined;
        for (0..ITERS) |i| {
            const profile = try model.g.profileExecution(.{ .loss_grad = model.loss.grad });
            fwd_times[i] = profile.forward_ns;
            bwd_times[i] = profile.backward_ns;
        }
        std.mem.sort(u64, &fwd_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, &bwd_times, {}, std.sort.asc(u64));
        var fwd_sum: u64 = 0;
        var bwd_sum: u64 = 0;
        for (fwd_times) |t| fwd_sum += t;
        for (bwd_times) |t| bwd_sum += t;

        try w.interface.print("  forward:  min={d:>8.1}  p50={d:>8.1}  mean={d:>8.1} us\n", .{
            @as(f64, @floatFromInt(fwd_times[0])) / 1000.0,
            @as(f64, @floatFromInt(fwd_times[ITERS / 2])) / 1000.0,
            @as(f64, @floatFromInt(fwd_sum / ITERS)) / 1000.0,
        });
        try w.interface.print("  backward: min={d:>8.1}  p50={d:>8.1}  mean={d:>8.1} us\n", .{
            @as(f64, @floatFromInt(bwd_times[0])) / 1000.0,
            @as(f64, @floatFromInt(bwd_times[ITERS / 2])) / 1000.0,
            @as(f64, @floatFromInt(bwd_sum / ITERS)) / 1000.0,
        });
        const total_mean = (fwd_sum + bwd_sum) / ITERS;
        try w.interface.print("  total:    mean={d:>8.1} us  ({d:.0} img/s)\n\n", .{
            @as(f64, @floatFromInt(total_mean)) / 1000.0,
            @as(f64, @floatFromInt(BATCH)) / (@as(f64, @floatFromInt(total_mean)) / 1_000_000_000.0),
        });
        w.interface.flush() catch {};
    }

    try w.interface.print("({} warmup + {} timed iterations)\n\n", .{ WARMUP, ITERS });
    w.interface.flush() catch {};
}
