//! CPU backend: BLAS matmul override for graph execution.
//! Returns null from compile_program — the graph interpreter
//! is already fast on CPU (AMX handles matmul at hardware speed).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const forward = @import("../tensor/forward.zig");

pub const CpuBackend = struct {
    pub fn backend(self: *CpuBackend) backend_mod.Backend {
        _ = self;
        return .{
            .ctx = undefined,
            .vtable = &vtable,
            .name_str = "cpu",
            .device_type = .cpu,
        };
    }
};

fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

fn compileProgram(_: *anyopaque, _: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    return null;
}

fn executeProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.ProgramIO, _: []const backend_mod.ProgramIO) void {}

fn refreshProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.DeviceOp) void {}

fn freeProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) void {}

fn getRuntimeProfile(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) ?*@import("../profile.zig").RuntimeProfile {
    return null;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgram,
    .refresh_program = refreshProgram,
    .execute_program = executeProgram,
    .free_program = freeProgram,
    .get_runtime_profile = getRuntimeProfile,
};

test "cpu backend host dense matmul" {
    var cpu = CpuBackend{};
    var dst = [_]f32{0} ** 4;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ok = backend_mod.tryDenseMatMul(f32, cpu.backend(), .{
        .dst = &dst, .a = &a, .b = &b,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });
    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "cpu backend compile returns null" {
    var cpu = CpuBackend{};
    const be = cpu.backend();
    try std.testing.expect(be.compileProgram(.{ .ops = &.{}, .n_buffers = 0, .buffer_sizes = &.{}, .initial_uploads = &.{} }) == null);
}
