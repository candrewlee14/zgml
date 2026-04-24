const std = @import("std");
const builtin = @import("builtin");
const zgml = @import("zgml");
const opts = @import("zgml_options");

const backend_mod = zgml.backend;

const io = std.Io.Threaded.global_single_threaded.io();

fn nowNs() i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn benchProgram(
    be: backend_mod.Backend,
    label: []const u8,
    program: backend_mod.DeviceProgram,
    out_idx: u16,
    out_len: usize,
    iters: usize,
) !void {
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const alloc = std.heap.page_allocator;
    const out = try alloc.alloc(f32, out_len);
    defer alloc.free(out);
    var output = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(out.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};

    be.executeProgram(handle, &.{}, &output);

    const t0 = nowNs();
    for (0..iters) |_| be.executeProgram(handle, &.{}, &output);
    const elapsed_ns: u64 = @intCast(nowNs() - t0);
    const per_iter = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters));

    std.debug.print("{s} {s}: {d:.1} ns/iter\n", .{ be.name_str, label, per_iter });
}

fn benchMatmul(be: backend_mod.Backend) !void {
    const alloc = std.heap.page_allocator;
    const M = 1;
    const K = 512;
    const N = 512;
    const a = try alloc.alloc(f32, M * K);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, K * N);
    defer alloc.free(b);
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.03125;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.015625;

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = M, .N = N, .K = K, .a_row_stride = K, .a_col_stride = 1, .b_row_stride = N, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = N },
    } }};
    const sizes = [_]usize{ a.len, b.len, M * N };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(a.ptr), .size = @intCast(a.len * @sizeOf(f32)) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(b.ptr), .size = @intCast(b.len * @sizeOf(f32)) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
    try benchProgram(be, "matmul M1 K512", program, 2, M * N, 100);
}

fn benchElementwise(be: backend_mod.Backend) !void {
    const alloc = std.heap.page_allocator;
    const N = 4096;
    const a = try alloc.alloc(f32, N);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, N);
    defer alloc.free(b);
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 31)) * 0.01;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 37)) * 0.02;

    const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = N,
    } }};
    const sizes = [_]usize{ a.len, b.len, N };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(a.ptr), .size = @intCast(a.len * @sizeOf(f32)) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(b.ptr), .size = @intCast(b.len * @sizeOf(f32)) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
    try benchProgram(be, "add 4096", program, 2, N, 500);
}

fn benchRmsNormDecode(be: backend_mod.Backend) !void {
    const alloc = std.heap.page_allocator;
    const D = 4096;
    const src = try alloc.alloc(f32, D);
    defer alloc.free(src);
    for (src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) * 0.03125;

    const ops = [_]backend_mod.DeviceOp{.{ .rmsnorm = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = D,
        .eps = 1e-5,
    } }};
    const sizes = [_]usize{ src.len, D };
    const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(src.ptr), .size = @intCast(src.len * @sizeOf(f32)) }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
    try benchProgram(be, "rmsnorm decode D4096", program, 1, D, 500);
}

fn benchAttentionDecode(be: backend_mod.Backend) !void {
    const alloc = std.heap.page_allocator;
    const D = 64;
    const S = 128;
    const q = try alloc.alloc(f32, D);
    defer alloc.free(q);
    const k = try alloc.alloc(f32, D * S);
    defer alloc.free(k);
    const v = try alloc.alloc(f32, D * S);
    defer alloc.free(v);
    const mask = try alloc.alloc(f32, S);
    defer alloc.free(mask);

    for (q, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;
    for (k, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) * 0.02;
    for (v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.03;
    @memset(mask, 0);

    const ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = D,
        .seq_q = 1,
        .seq_kv = S,
        .scale = 0.125,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = D,
        .k_rs = 1,
        .k_cs = D,
        .v_rs = 1,
        .v_cs = D,
        .mask_rs = 1,
        .mask_cs = S,
        .dst_rs = 1,
        .dst_cs = D,
    } }};
    const sizes = [_]usize{ q.len, k.len, v.len, mask.len, D };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(q.ptr), .size = @intCast(q.len * @sizeOf(f32)) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(k.ptr), .size = @intCast(k.len * @sizeOf(f32)) },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(v.ptr), .size = @intCast(v.len * @sizeOf(f32)) },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(mask.ptr), .size = @intCast(mask.len * @sizeOf(f32)) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &sizes, .initial_uploads = &uploads };
    try benchProgram(be, "attention decode D64 S128", program, 4, D, 100);
}

fn benchBackend(be: backend_mod.Backend) !void {
    try benchMatmul(be);
    try benchElementwise(be);
    try benchRmsNormDecode(be);
    try benchAttentionDecode(be);
}

pub fn main() !void {
    std.debug.print("backend program decision bench\n", .{});

    var cpu = zgml.backend_cpu.CpuBackend{};
    try benchBackend(cpu.backend());

    if (builtin.os.tag == .macos) {
        var metal = zgml.backend_metal.MetalBackend.init() catch |err| switch (err) {
            error.MetalNotAvailable => {
                std.debug.print("metal unavailable\n", .{});
                return;
            },
            else => return err,
        };
        defer metal.deinit();
        try benchBackend(metal.backend());
    }

    if (opts.use_wgpu) {
        var wgpu = zgml.backend_wgpu.WgpuBackend.init() catch |err| switch (err) {
            error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => {
                std.debug.print("wgpu unavailable\n", .{});
                return;
            },
            else => return err,
        };
        defer wgpu.deinit();
        try benchBackend(wgpu.backend());
    }
}
