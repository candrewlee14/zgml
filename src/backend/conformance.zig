//! Backend conformance tests against the shared reference executor.

const std = @import("std");
const builtin = @import("builtin");
const opts = @import("zgml_options");

const backend_mod = @import("../backend.zig");
const reference = @import("reference.zig");
const cpu_mod = @import("cpu.zig");
const metal_mod = if (builtin.os.tag == .macos) @import("metal.zig") else struct {};
const wgpu_mod = if (opts.use_wgpu) @import("wgpu.zig") else struct {};

fn expectedOutput(alloc: std.mem.Allocator, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize) ![]f32 {
    var table = try reference.OwnedBufferTable.init(alloc, program.buffer_sizes);
    defer table.deinit();

    table.upload(program.initial_uploads);
    reference.executeProgram(table.buffers, &.{}, program.ops);

    const out = try alloc.alloc(f32, out_len);
    var io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(out.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
    table.download(&io);
    return out;
}

fn backendOutput(alloc: std.mem.Allocator, be: backend_mod.Backend, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize) ![]f32 {
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const out = try alloc.alloc(f32, out_len);
    var io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(out.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
    be.executeProgram(handle, &.{}, &io);
    return out;
}

fn expectApproxSlices(want: []const f32, got: []const f32, tol: f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectApproxEqAbs(w, g, tol);
    }
}

fn assertBackendMatchesReference(be: backend_mod.Backend, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize, tol: f32) !void {
    const alloc = std.testing.allocator;
    const expected = try expectedOutput(alloc, program, out_idx, out_len);
    defer alloc.free(expected);
    const got = try backendOutput(alloc, be, program, out_idx, out_len);
    defer alloc.free(got);
    try expectApproxSlices(expected, got, tol);
}

fn runCoreCases(be: backend_mod.Backend, tol: f32) !void {
    {
        var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
        const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
            .dst = 2,
            .a = 0,
            .b = 1,
            .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
        } }};
        const sizes = [_]usize{ 6, 6, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&a), .size = 6 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&b), .size = 6 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 4, tol);
    }

    {
        var a = [_]f32{ 1, 2, 3, 4 };
        var b = [_]f32{ 10, 20, 30, 40 };
        const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
            .op = .add,
            .dst = 2,
            .src0 = 0,
            .src1 = 1,
            .n = 4,
        } }};
        const sizes = [_]usize{ 4, 4, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&a), .size = 4 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&b), .size = 4 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 4, tol);
    }

    {
        var src = [_]f32{ 1, 2, 3, -1, 0, 1 };
        const ops = [_]backend_mod.DeviceOp{.{ .softmax = .{
            .dst = 1,
            .src = 0,
            .rows = 2,
            .cols = 3,
        } }};
        const sizes = [_]usize{ 6, 6 };
        const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 6 * @sizeOf(f32) }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 6, tol);
    }

    {
        var src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var cs = [_]f32{ 1, 1, 0, 0, 0, 0, 1, 1 };
        const ops = [_]backend_mod.DeviceOp{.{ .rope = .{
            .dst = 2,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } }};
        const sizes = [_]usize{ 8, 8, 8 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 8 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&cs), .size = 8 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 8, tol);
    }
}

test "cpu backend conforms to reference core ops" {
    var cpu = cpu_mod.CpuBackend{};
    try runCoreCases(cpu.backend(), 1e-5);
}

test "metal backend conforms to reference core ops" {
    if (builtin.os.tag != .macos) return;
    var metal = metal_mod.MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    try runCoreCases(metal.backend(), 1e-5);
}

test "wgpu backend conforms to reference core ops" {
    if (!opts.use_wgpu) return;
    var wgpu = wgpu_mod.WgpuBackend.init() catch |err| switch (err) {
        error.WgpuInitFailed, error.WgpuAdapterNotAvailable, error.WgpuDeviceNotAvailable => return,
        else => return err,
    };
    defer wgpu.deinit();
    try runCoreCases(wgpu.backend(), 1e-4);
}
