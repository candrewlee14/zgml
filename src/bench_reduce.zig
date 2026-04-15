//! Microbenchmark for reduction (sum) and broadcast (repeat) operations.
//!
//! Isolates the hot path from BN backward: [24,24,32,32] → [1,1,32,1]
//! which is the per-channel reduction over spatial + batch dimensions.
//!
//! Run with: zig build bench-reduce

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const stdout_file = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &buf);

    // BN backward reduction: [24,24,C,32] → [1,1,C,1]
    // Test with C=8 (simple model) and C=32 (deep model)
    for ([_]usize{ 8, 32 }) |C| {
        const src = try Tensor(f32).init(alloc, &.{ 24, 24, C, 32 });
        defer src.deinit();
        const dst = try Tensor(f32).init(alloc, &.{ 1, 1, C, 1 });
        defer dst.deinit();

        // Fill with data
        for (src.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 97)) * 0.01;

        const n_elems = src.nElems();
        const warmup: usize = 50;
        const iters: usize = 500;

        // Warmup
        for (0..warmup) |_| src.computeSum(dst);

        // Benchmark sum: reduce [24,24,C,32] → [1,1,C,1]
        // dst.computeSum(src) — dst is target, src is source
        for (0..warmup) |_| dst.computeSum(src);
        var t0 = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..iters) |_| dst.computeSum(src);
        const sum_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

        // Benchmark repeat: broadcast [1,1,C,1] → [24,24,C,32]
        const rep_dst = try Tensor(f32).init(alloc, &.{ 24, 24, C, 32 });
        defer rep_dst.deinit();
        for (0..warmup) |_| rep_dst.computeRepeat(dst);

        t0 = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..iters) |_| rep_dst.computeRepeat(dst);
        const rep_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

        const sum_us = @as(f64, @floatFromInt(sum_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const rep_us = @as(f64, @floatFromInt(rep_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const sum_gbps = @as(f64, @floatFromInt(n_elems * 4)) / (@as(f64, @floatFromInt(sum_ns)) / @as(f64, @floatFromInt(iters))) * 1.0;
        const rep_gbps = @as(f64, @floatFromInt(n_elems * 4)) / (@as(f64, @floatFromInt(rep_ns)) / @as(f64, @floatFromInt(iters))) * 1.0;

        try w.interface.print("[24,24,{},32] ({} elems)\n", .{ C, n_elems });
        try w.interface.print("  sum:    {d:>8.1} us   {d:.2} GB/s\n", .{ sum_us, sum_gbps });
        try w.interface.print("  repeat: {d:>8.1} us   {d:.2} GB/s\n", .{ rep_us, rep_gbps });

        // Also benchmark a raw SIMD sum of the same data for comparison
        t0 = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..iters) |_| {
            var acc: f32 = 0;
            const vec_size = 8;
            const Vec = @Vector(vec_size, f32);
            var vacc: Vec = @splat(0);
            var j: usize = 0;
            while (j + vec_size <= n_elems) : (j += vec_size) {
                vacc += src.data[j..][0..vec_size].*;
            }
            acc = @reduce(.Add, vacc);
            while (j < n_elems) : (j += 1) acc += src.data[j];
            std.mem.doNotOptimizeAway(acc);
        }
        const raw_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);
        const raw_us = @as(f64, @floatFromInt(raw_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const raw_gbps = @as(f64, @floatFromInt(n_elems * 4)) / (@as(f64, @floatFromInt(raw_ns)) / @as(f64, @floatFromInt(iters))) * 1.0;
        try w.interface.print("  raw:    {d:>8.1} us   {d:.2} GB/s  (SIMD flat sum baseline)\n\n", .{ raw_us, raw_gbps });
    }
    w.interface.flush() catch {};
}
