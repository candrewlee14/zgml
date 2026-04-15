//! Metal backend benchmark: compare CPU BLAS vs Metal GPU matmul.
//!
//! Run with: zig build bench-metal

const std = @import("std");
const zgml = @import("zgml");
const backend_mod = zgml.backend;
const CpuBackend = zgml.backend_cpu.CpuBackend;
const MetalBackend = zgml.backend_metal.MetalBackend;
const time_compat = zgml.time_compat;

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
    // Build a single-matmul program.
    const a_data = try std.heap.page_allocator.alloc(f32, M * K);
    defer std.heap.page_allocator.free(a_data);
    @memset(a_data, 1.0);
    const b_data = try std.heap.page_allocator.alloc(f32, K * N);
    defer std.heap.page_allocator.free(b_data);
    @memset(b_data, 1.0);

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2, .a = 0, .b = 1,
        .geom = .{ .M = M, .N = N, .K = K, .a_row_stride = K, .a_col_stride = 1, .b_row_stride = N, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = N },
    } }};
    const buf_sizes = [_]usize{ M * K, K * N, M * N };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(a_data.ptr), .size = @intCast(M * K * 4) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(b_data.ptr), .size = @intCast(K * N * 4) },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 3,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const dst = try std.heap.page_allocator.alloc(f32, M * N);
    defer std.heap.page_allocator.free(dst);
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(dst.ptr), .size = @intCast(M * N * 4) }};

    // Warmup
    for (0..WARMUP) |_| be.executeProgram(handle, &.{}, &out);

    // Timed
    var timer = time_compat.Timer.start();
    for (0..ITERS) |_| be.executeProgram(handle, &.{}, &out);
    const elapsed_ns = timer.read();

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const gflops = flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));

    try writer.print("    {s:<12} {d:8.1} us/iter  {d:6.2} GFLOPS\n", .{ label, per_iter_us, gflops });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &stdout_buf);

    try w.interface.print("\nMetal Backend Benchmark — Dense MatMul\n", .{});
    try w.interface.print("========================================\n", .{});
    try w.interface.print("  warmup={d}, iters={d}\n\n", .{ WARMUP, ITERS });

    var cpu_be = CpuBackend{};
    const cpu = cpu_be.backend();

    var metal_be = MetalBackend.init() catch |err| {
        try w.interface.print("Metal init failed: {}\n", .{err});
        return;
    };
    defer metal_be.deinit();
    const metal = metal_be.backend();

    const sizes = [_][3]usize{
        .{ 1024, 1024, 1024 },
        .{ 2048, 2048, 2048 },
        // Batch=1 inference
        .{ 1, 768, 768 },
        .{ 1, 768, 3072 },
        // Batched inference / prefill
        .{ 8, 768, 768 },
        .{ 8, 768, 3072 },
        .{ 32, 768, 768 },
        .{ 32, 768, 3072 },
        .{ 128, 768, 768 },
        .{ 128, 768, 3072 },
        .{ 512, 768, 768 },
    };

    for (sizes) |s| {
        try w.interface.print("  M={d}, N={d}, K={d}:\n", .{ s[0], s[1], s[2] });
        try benchDenseMatMul("CPU/BLAS", cpu, s[0], s[1], s[2], &w.interface);
        try benchDenseMatMul("Metal/GPU", metal, s[0], s[1], s[2], &w.interface);
    }

    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
