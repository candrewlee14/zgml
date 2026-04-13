//! Benchmark suite for zgml tensor operations.
//!
//! Run with: zig build bench
//! Always built with ReleaseFast for meaningful measurements.

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const IoWriter = std.io.Writer;
const opts = @import("zgml_options");
const use_blas = opts.use_blas;

// ---------------------------------------------------------------------------
// Naive baselines (scalar, no SIMD, bounds-check-free)
// ---------------------------------------------------------------------------

fn naiveAdd(dst: []f32, a: []const f32, b: []const f32) void {
    @setRuntimeSafety(false);
    for (0..dst.len) |i| dst[i] = a[i] + b[i];
}

fn naiveMul(dst: []f32, a: []const f32, b: []const f32) void {
    @setRuntimeSafety(false);
    for (0..dst.len) |i| dst[i] = a[i] * b[i];
}

fn naiveRelu(dst: []f32, a: []const f32) void {
    @setRuntimeSafety(false);
    for (0..dst.len) |i| dst[i] = if (a[i] > 0) a[i] else 0;
}

fn naiveGelu(dst: []f32, a: []const f32) void {
    @setRuntimeSafety(false);
    const GELU_COEF_A = 0.044715;
    const SQRT_2_OVER_PI = 0.79788456080286535587989211986876;
    for (0..dst.len) |i| {
        const x = a[i];
        dst[i] = 0.5 * x * (1.0 + std.math.tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)));
    }
}

fn naiveMatMul(dst: []f32, a: []const f32, b: []const f32, M: usize, N: usize, K: usize) void {
    @setRuntimeSafety(false);
    for (0..M) |i| {
        for (0..N) |j| {
            var s: f32 = 0;
            for (0..K) |ki| s += a[i * K + ki] * b[ki * N + j];
            dst[i * N + j] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const WARMUP = 5;

fn benchMedian(times: []u64) u64 {
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    return times[times.len / 2];
}

fn doNotOptimize(ptr: anytype) void {
    const P = @TypeOf(ptr);
    const addr: *volatile P = @constCast(&ptr);
    _ = addr.*;
}

const ITERS = 200;

fn runBench(comptime func: anytype, args: anytype) u64 {
    for (0..WARMUP) |_| @call(.auto, func, args);
    var times: [ITERS]u64 = undefined;
    for (&times) |*t| {
        var timer = std.time.Timer.start() catch unreachable;
        @call(.auto, func, args);
        t.* = timer.read();
    }
    return benchMedian(&times);
}

fn computeWrapper(t: *Tensor(f32)) void {
    t.compute();
}

fn printElemResult(w: *IoWriter, name: []const u8, n: usize, naive_ns: u64, simd_ns: u64) void {
    const throughput = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(simd_ns, 1)));
    const speedup = @as(f64, @floatFromInt(naive_ns)) / @as(f64, @floatFromInt(@max(simd_ns, 1)));
    w.print("  {s:<5} n={d:<10}  naive={d:>8}ns  simd={d:>8}ns  {d:>6.2} Gelem/s  {d:.1}x\n", .{
        name, n, naive_ns, simd_ns, throughput, speedup,
    }) catch {};
}

// ---------------------------------------------------------------------------
// Element-wise benchmarks
// ---------------------------------------------------------------------------

fn benchElementwise(alloc: std.mem.Allocator, w: *IoWriter) !void {
    const sizes = [_]usize{ 64, 4096, 1_048_576 };

    for (sizes) |n| {
        const buf_a = try alloc.alloc(f32, n);
        defer alloc.free(buf_a);
        const buf_b = try alloc.alloc(f32, n);
        defer alloc.free(buf_b);
        const buf_dst = try alloc.alloc(f32, n);
        defer alloc.free(buf_dst);

        var prng = std.Random.DefaultPrng.init(42);
        for (buf_a) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;
        for (buf_b) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;

        const t_a = try Tensor(f32).init(alloc, &.{n});
        defer t_a.deinit();
        const t_b = try Tensor(f32).init(alloc, &.{n});
        defer t_b.deinit();
        @memcpy(t_a.data, buf_a);
        @memcpy(t_b.data, buf_b);

        const t_add = t_a.add(t_b);
        defer t_add.deinit();
        const t_mul = t_a.mul(t_b);
        defer t_mul.deinit();
        const t_relu = t_a.relu();
        defer t_relu.deinit();
        const t_gelu = t_a.gelu();
        defer t_gelu.deinit();

        {
            const naive = runBench(naiveAdd, .{ buf_dst, buf_a, buf_b });
            doNotOptimize(buf_dst.ptr);
            const simd = runBench(computeWrapper, .{t_add});
            doNotOptimize(t_add.data.ptr);
            printElemResult(w, "add", n, naive, simd);
        }
        {
            const naive = runBench(naiveMul, .{ buf_dst, buf_a, buf_b });
            doNotOptimize(buf_dst.ptr);
            const simd = runBench(computeWrapper, .{t_mul});
            doNotOptimize(t_mul.data.ptr);
            printElemResult(w, "mul", n, naive, simd);
        }
        {
            const naive = runBench(naiveRelu, .{ buf_dst, buf_a });
            doNotOptimize(buf_dst.ptr);
            const simd = runBench(computeWrapper, .{t_relu});
            doNotOptimize(t_relu.data.ptr);
            printElemResult(w, "relu", n, naive, simd);
        }
        {
            const naive = runBench(naiveGelu, .{ buf_dst, buf_a });
            doNotOptimize(buf_dst.ptr);
            const simd = runBench(computeWrapper, .{t_gelu});
            doNotOptimize(t_gelu.data.ptr);
            printElemResult(w, "gelu", n, naive, simd);
        }
        try w.print("\n", .{});
    }
}

// ---------------------------------------------------------------------------
// MatMul benchmarks
// ---------------------------------------------------------------------------

fn benchMatMul(alloc: std.mem.Allocator, w: *IoWriter) !void {
    const sizes = [_]usize{ 4, 32, 128, 512 };

    for (sizes) |dim| {
        const n = dim * dim;
        const buf_a = try alloc.alloc(f32, n);
        defer alloc.free(buf_a);
        const buf_b = try alloc.alloc(f32, n);
        defer alloc.free(buf_b);
        const buf_dst = try alloc.alloc(f32, n);
        defer alloc.free(buf_dst);

        var prng = std.Random.DefaultPrng.init(123);
        for (buf_a) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;
        for (buf_b) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;

        const t_a = try Tensor(f32).init(alloc, &.{ dim, dim });
        defer t_a.deinit();
        const t_b = try Tensor(f32).init(alloc, &.{ dim, dim });
        defer t_b.deinit();
        @memcpy(t_a.data, buf_a);
        @memcpy(t_b.data, buf_b);

        const t_dst = t_a.matMul(false, t_b, false);
        defer t_dst.deinit();

        const flops: f64 = 2.0 * @as(f64, @floatFromInt(dim)) * @as(f64, @floatFromInt(dim)) * @as(f64, @floatFromInt(dim));

        const naive = runBench(naiveMatMul, .{ buf_dst, buf_a, buf_b, dim, dim, dim });
        doNotOptimize(buf_dst.ptr);
        const tiled = runBench(computeWrapper, .{t_dst});
        doNotOptimize(t_dst.data.ptr);
        const gflops = flops / @as(f64, @floatFromInt(@max(tiled, 1)));
        const speedup = @as(f64, @floatFromInt(naive)) / @as(f64, @floatFromInt(@max(tiled, 1)));
        try w.print("  {d:>3}x{d:>3}x{d:>3}  naive={d:>10}ns  tiled={d:>10}ns  {d:>6.2} GFLOPS  {d:.1}x\n", .{
            dim, dim, dim, naive, tiled, gflops, speedup,
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const stdout_file = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    var file_writer = stdout_file.writer(&buf);
    var w = &file_writer.interface;

    try w.print("\nzgml benchmark suite", .{});
    if (use_blas) try w.print(" [BLAS enabled]", .{});
    try w.print("\n====================\n\n", .{});

    try w.print("Element-wise operations (SIMD vs naive scalar)\n", .{});
    try w.print("-----------------------------------------------\n", .{});
    try benchElementwise(alloc, w);

    if (use_blas) {
        try w.print("\nMatrix multiplication (OpenBLAS vs naive triple-loop)\n", .{});
    } else {
        try w.print("\nMatrix multiplication (tiled SIMD vs naive triple-loop)\n", .{});
    }
    try w.print("-------------------------------------------------------\n", .{});
    try benchMatMul(alloc, w);

    try w.print("\n", .{});
    file_writer.interface.flush() catch {};
}
