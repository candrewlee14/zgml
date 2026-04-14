//! Metal backend benchmark: compare CPU BLAS vs Metal GPU matmul.
//!
//! Run with: zig build bench-metal

const std = @import("std");
const zgml = @import("zgml");
const backend_mod = zgml.backend;
const CpuBackend = zgml.backend_cpu.CpuBackend;
const MetalBackend = zgml.backend_metal.MetalBackend;

const WARMUP = 5;
const ITERS = 50;

fn benchDenseMatMul(
    comptime label: []const u8,
    be: backend_mod.Backend,
    M: usize,
    N: usize,
    K: usize,
    writer: anytype,
) !void {
    const max_elems = @max(M * K, @max(K * N, M * N));
    const fill = try std.heap.page_allocator.alloc(f32, max_elems);
    defer std.heap.page_allocator.free(fill);
    @memset(fill, 1.0);

    const a_buf = be.allocSlice(f32, M * K) orelse return error.OutOfMemory;
    defer be.freeBuffer(a_buf);
    const b_buf = be.allocSlice(f32, K * N) orelse return error.OutOfMemory;
    defer be.freeBuffer(b_buf);
    const dst_buf = be.allocSlice(f32, M * N) orelse return error.OutOfMemory;
    defer be.freeBuffer(dst_buf);

    be.uploadSlice(f32, a_buf, 0, fill[0 .. M * K]);
    be.uploadSlice(f32, b_buf, 0, fill[0 .. K * N]);

    const spec = backend_mod.DeviceMatMulSpecF32{
        .dst = dst_buf,
        .a = a_buf,
        .b = b_buf,
        .geom = .{
            .M = M, .N = N, .K = K,
            .a_row_stride = K, .a_col_stride = 1,
            .b_row_stride = N, .b_col_stride = 1,
            .a_offset = 0, .b_offset = 0,
            .dst_offset = 0, .dst_row_stride = N,
        },
    };

    for (0..WARMUP) |_| _ = be.deviceMatMul(spec);
    be.sync();

    var timer = try std.time.Timer.start();
    for (0..ITERS) |_| _ = be.deviceMatMul(spec);
    be.sync();
    const elapsed_ns = timer.read();

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const gflops = flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));

    try writer.print("    {s:<12} {d:8.1} us/iter  {d:6.2} GFLOPS\n", .{ label, per_iter_us, gflops });
}

pub fn main() !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(&stdout_buf);

    try w.interface.print("\nMetal Backend Benchmark — Dense MatMul\n", .{});
    try w.interface.print("========================================\n", .{});
    try w.interface.print("  warmup={d}, iters={d}\n\n", .{ WARMUP, ITERS });

    const alloc = std.heap.page_allocator;

    var cpu_be = CpuBackend.init(alloc);
    const cpu = cpu_be.backend();

    var metal_be = MetalBackend.init() catch |err| {
        try w.interface.print("Metal init failed: {}\n", .{err});
        return;
    };
    defer metal_be.deinit();
    const metal = metal_be.backend();

    const sizes = [_][3]usize{
        .{ 64, 64, 64 },
        .{ 128, 128, 128 },
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
        .{ 2048, 2048, 2048 },
        // Inference-shaped: 1 token, large model dims
        .{ 1, 768, 768 },
        .{ 1, 2048, 768 },
        .{ 1, 768, 3072 },
    };

    for (sizes) |s| {
        try w.interface.print("  M={d}, N={d}, K={d}:\n", .{ s[0], s[1], s[2] });
        try benchDenseMatMul("CPU/BLAS", cpu, s[0], s[1], s[2], &w.interface);
        try benchDenseMatMul("Metal/GPU", metal, s[0], s[1], s[2], &w.interface);
    }

    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
