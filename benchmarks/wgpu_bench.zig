//! WebGPU backend benchmark: compare CPU BLAS vs WebGPU GPU matmul.
//!
//! Measures single-op latency (sync overhead included) and multi-op
//! pipeline throughput (sync amortized), reflecting real inference.
//!
//! Run with: zig build bench-wgpu -Duse-wgpu=true

const std = @import("std");
const zgml = @import("zgml");
const backend_mod = zgml.backend;
const CpuBackend = zgml.backend_cpu.CpuBackend;
const WgpuBackend = zgml.backend_wgpu.WgpuBackend;
const WARMUP = 5;
const ITERS = 50;

fn benchCpuMatMul(
    M: usize,
    N: usize,
    K: usize,
    writer: anytype,
    io: std.Io,
) !void {
    var cpu_be = CpuBackend{};
    const cpu = cpu_be.backend();

    const a_data = try std.heap.page_allocator.alloc(f32, M * K);
    defer std.heap.page_allocator.free(a_data);
    @memset(a_data, 1.0);
    const b_data = try std.heap.page_allocator.alloc(f32, K * N);
    defer std.heap.page_allocator.free(b_data);
    @memset(b_data, 1.0);
    const dst = try std.heap.page_allocator.alloc(f32, M * N);
    defer std.heap.page_allocator.free(dst);

    const geom = backend_mod.MatMulGeometry{
        .M = M, .N = N, .K = K,
        .a_row_stride = K, .a_col_stride = 1,
        .b_row_stride = N, .b_col_stride = 1,
        .a_offset = 0, .b_offset = 0,
        .dst_offset = 0, .dst_row_stride = N,
    };

    // Warmup
    for (0..WARMUP) |_| {
        _ = cpu.vtable.dense_matmul_f32(cpu.ctx, .{ .dst = dst, .a = a_data, .b = b_data, .geom = geom });
    }

    // Timed
    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    for (0..ITERS) |_| {
        _ = cpu.vtable.dense_matmul_f32(cpu.ctx, .{ .dst = dst, .a = a_data, .b = b_data, .geom = geom });
    }
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const gflops = flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));

    try writer.print("    {s:<12} {d:8.1} us/iter  {d:6.2} GFLOPS\n", .{ "CPU/BLAS", per_iter_us, gflops });
}

fn benchWgpuMatMul(
    be: backend_mod.Backend,
    M: usize,
    N: usize,
    K: usize,
    writer: anytype,
    io: std.Io,
) !void {
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
    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    for (0..ITERS) |_| be.executeProgram(handle, &.{}, &out);
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const gflops = flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));

    try writer.print("    {s:<12} {d:8.1} us/iter  {d:6.2} GFLOPS\n", .{ "WebGPU/GPU", per_iter_us, gflops });
}

fn benchWgpuF16MatMul(
    be: backend_mod.Backend,
    M: usize,
    N: usize,
    K: usize,
    writer: anytype,
    io: std.Io,
) !void {
    const a_data = try std.heap.page_allocator.alloc(f32, M * K);
    defer std.heap.page_allocator.free(a_data);
    @memset(a_data, 1.0);
    const b_data = try std.heap.page_allocator.alloc(f32, K * N);
    defer std.heap.page_allocator.free(b_data);
    @memset(b_data, 1.0);

    // Use regular matmul op — B is in initial_uploads so backend auto-promotes to f16.
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

    for (0..WARMUP) |_| be.executeProgram(handle, &.{}, &out);

    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    for (0..ITERS) |_| be.executeProgram(handle, &.{}, &out);
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const gflops = flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));

    try writer.print("    {s:<12} {d:8.1} us/iter  {d:6.2} GFLOPS\n", .{ "WebGPU/F16", per_iter_us, gflops });
}

/// Benchmark a multi-op pipeline: N_LAYERS × (matmul + layernorm) batched
/// in a single DeviceProgram, one sync per execute — like a real forward pass.
fn benchWgpuPipeline(
    be: backend_mod.Backend,
    comptime n_layers: usize,
    M: usize,
    N: usize,
    K: usize,
    writer: anytype,
    io: std.Io,
) !void {
    // Layout: buf 0 = A (M×K input), buf 1 = B (K×N weights, constant),
    // buf 2 = C (M×N matmul output / layernorm input), buf 3 = D (M×N layernorm output).
    // Each layer: matmul(A,B)->C, layernorm(C)->D, then the next layer
    // re-uses C as scratch by doing matmul(D,B)->C, layernorm(C)->D, etc.
    // (First layer reads from buf 0, subsequent layers read from buf 3.)
    const n_ops = n_layers * 2;
    var ops: [n_ops]backend_mod.DeviceOp = undefined;
    for (0..n_layers) |i| {
        const a_buf: u16 = if (i == 0) 0 else 3;
        ops[i * 2] = .{ .matmul = .{
            .dst = 2, .a = a_buf, .b = 1,
            .geom = .{
                .M = M, .N = N, .K = if (i == 0) K else N,
                .a_row_stride = if (i == 0) @intCast(K) else @intCast(N),
                .a_col_stride = 1,
                .b_row_stride = @intCast(N), .b_col_stride = 1,
                .a_offset = 0, .b_offset = 0,
                .dst_offset = 0, .dst_row_stride = @intCast(N),
            },
        } };
        ops[i * 2 + 1] = .{ .layernorm = .{
            .dst = 3, .src = 2,
            .rows = @intCast(M), .cols = @intCast(N),
        } };
    }

    const a_data = try std.heap.page_allocator.alloc(f32, M * K);
    defer std.heap.page_allocator.free(a_data);
    @memset(a_data, 1.0);
    // B must be K×N for layer 0 and N×N for subsequent layers.
    // Use max(K,N)×N so buf 1 is large enough for both.
    const b_rows = @max(K, N);
    const b_data = try std.heap.page_allocator.alloc(f32, b_rows * N);
    defer std.heap.page_allocator.free(b_data);
    @memset(b_data, 0.01);

    const buf_sizes = [_]usize{ M * K, b_rows * N, M * N, M * N };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(a_data.ptr), .size = @intCast(M * K * 4) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(b_data.ptr), .size = @intCast(b_rows * N * 4) },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 4,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const dst = try std.heap.page_allocator.alloc(f32, M * N);
    defer std.heap.page_allocator.free(dst);
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 3, .host_ptr = @ptrCast(dst.ptr), .size = @intCast(M * N * 4) }};

    for (0..WARMUP) |_| be.executeProgram(handle, &.{}, &out);

    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    for (0..ITERS) |_| be.executeProgram(handle, &.{}, &out);
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(ITERS));
    // Total FLOPS: n_layers matmuls. Layer 0: 2*M*N*K, layers 1+: 2*M*N*N.
    const flops_l0 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const flops_rest = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(N));
    const total_flops = flops_l0 + flops_rest * @as(f64, @floatFromInt(n_layers - 1));
    const gflops = total_flops * @as(f64, @floatFromInt(ITERS)) / @as(f64, @floatFromInt(elapsed_ns));
    // Per-matmul latency (sync amortized across n_layers matmuls).
    const per_matmul_us = per_iter_us / @as(f64, @floatFromInt(n_layers));

    try writer.print("    {d:>2}-layer     {d:8.1} us/iter  {d:6.2} GFLOPS  ({d:.1} us/matmul)\n", .{
        n_layers, per_iter_us, gflops, per_matmul_us,
    });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &stdout_buf);

    try w.interface.print("\nWebGPU Backend Benchmark\n", .{});
    try w.interface.print("========================\n", .{});
    try w.interface.print("  warmup={d}, iters={d}\n\n", .{ WARMUP, ITERS });

    var wgpu_be = WgpuBackend.init() catch |err| {
        try w.interface.print("WebGPU init failed: {}\n", .{err});
        return;
    };
    defer wgpu_be.deinit();
    const wgpu = wgpu_be.backend();

    // ── Single-op latency ────────────────────────────────────────────
    try w.interface.print("── Single-op latency (1 matmul per sync) ──\n\n", .{});

    const sizes = [_][3]usize{
        .{ 1024, 1024, 1024 },
        .{ 2048, 2048, 2048 },
        .{ 1, 768, 768 },
        .{ 1, 768, 3072 },
        .{ 32, 768, 768 },
        .{ 32, 768, 3072 },
        .{ 128, 768, 768 },
        .{ 512, 768, 768 },
    };

    for (sizes) |s| {
        try w.interface.print("  M={d}, N={d}, K={d}:\n", .{ s[0], s[1], s[2] });
        try benchCpuMatMul(s[0], s[1], s[2], &w.interface, io);
        try benchWgpuMatMul(wgpu, s[0], s[1], s[2], &w.interface, io);
        try benchWgpuF16MatMul(wgpu, s[0], s[1], s[2], &w.interface, io);
    }

    // ── Pipeline throughput ──────────────────────────────────────────
    try w.interface.print("\n── Pipeline throughput (N×matmul+layernorm, 1 sync) ──\n\n", .{});

    // Simulate transformer-like depth scaling at a fixed size.
    const pipe_m = 32;
    const pipe_n = 768;
    const pipe_k = 768;
    try w.interface.print("  M={d}, N={d}, K={d}:\n", .{ pipe_m, pipe_n, pipe_k });
    try benchWgpuPipeline(wgpu, 1, pipe_m, pipe_n, pipe_k, &w.interface, io);
    try benchWgpuPipeline(wgpu, 4, pipe_m, pipe_n, pipe_k, &w.interface, io);
    try benchWgpuPipeline(wgpu, 12, pipe_m, pipe_n, pipe_k, &w.interface, io);
    try benchWgpuPipeline(wgpu, 24, pipe_m, pipe_n, pipe_k, &w.interface, io);

    // Larger matmul pipeline.
    const pipe_m2 = 128;
    const pipe_n2 = 768;
    const pipe_k2 = 768;
    try w.interface.print("  M={d}, N={d}, K={d}:\n", .{ pipe_m2, pipe_n2, pipe_k2 });
    try benchWgpuPipeline(wgpu, 1, pipe_m2, pipe_n2, pipe_k2, &w.interface, io);
    try benchWgpuPipeline(wgpu, 12, pipe_m2, pipe_n2, pipe_k2, &w.interface, io);
    try benchWgpuPipeline(wgpu, 24, pipe_m2, pipe_n2, pipe_k2, &w.interface, io);

    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
