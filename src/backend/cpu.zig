//! CPU backend implementation.
//!
//! Provides both host kernel dispatch and device buffer management.
//! Device buffers are plain host allocations — this makes the CPU
//! backend a complete reference implementation for testing the device
//! buffer path without actual GPU hardware.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const forward = @import("../tensor/forward.zig");
const quant = @import("../quant.zig");

pub const CpuBackend = struct {
    alloc: ?std.mem.Allocator,

    /// Create a CPU backend with device buffer support.
    pub fn init(alloc: std.mem.Allocator) CpuBackend {
        return .{ .alloc = alloc };
    }

    /// Create a CPU backend without device buffer support (host dispatch only).
    pub fn initHostOnly() CpuBackend {
        return .{ .alloc = null };
    }

    pub fn backend(self: *CpuBackend) backend_mod.Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
            .name_str = "cpu",
            .device_type = .cpu,
            .capabilities = .{ .device_buffers = self.alloc != null },
        };
    }
};

// ── VTable implementation ──────────────────────────────────────────

fn getState(ctx: *anyopaque) *CpuBackend {
    return @ptrCast(@alignCast(ctx));
}

// ── Host kernel dispatch ───────────────────────────────────────────

fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

fn quantizedMatMulF32(_: *anyopaque, spec: backend_mod.QuantizedMatMulSpecF32) bool {
    if (spec.weight.rows != spec.K or spec.weight.cols != spec.N) return false;
    const Weight = quant.QuantizedWeight(f32);
    const weight = Weight{
        .data = spec.weight.data,
        .scales = spec.weight.scales,
        .rows = spec.weight.rows,
        .cols = spec.weight.cols,
        .block_size = spec.weight.block_size,
    };
    weight.matmul(spec.input, spec.dst, spec.M, spec.N, spec.K);
    return true;
}

// ── Device buffer management ───────────────────────────────────────

fn allocBuffer(ctx: *anyopaque, size: usize) ?backend_mod.DeviceBuffer {
    const self = getState(ctx);
    const alloc = self.alloc orelse return null;
    const mem = alloc.alloc(u8, size) catch return null;
    return .{ .ptr = @ptrCast(mem.ptr), .size = size };
}

fn freeBuffer(ctx: *anyopaque, buf: backend_mod.DeviceBuffer) void {
    const self = getState(ctx);
    const alloc = self.alloc orelse return;
    const ptr: [*]u8 = @ptrCast(buf.ptr);
    alloc.free(ptr[0..buf.size]);
}

fn upload(_: *anyopaque, dst: backend_mod.DeviceBuffer, dst_byte_offset: usize, src: []const u8) void {
    const ptr: [*]u8 = @ptrCast(dst.ptr);
    @memcpy(ptr[dst_byte_offset..][0..src.len], src);
}

fn download(_: *anyopaque, dst: []u8, src: backend_mod.DeviceBuffer, src_byte_offset: usize) void {
    const ptr: [*]const u8 = @ptrCast(src.ptr);
    @memcpy(dst, ptr[src_byte_offset..][0..dst.len]);
}

fn syncFn(_: *anyopaque) void {}

// ── Compiled programs ─────────────────────────────────────────────

fn compileProgram(_: *anyopaque, _: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    return null; // CPU backend doesn't compile device programs
}

fn executeProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.ProgramIO, _: []const backend_mod.ProgramIO) void {}

fn freeProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) void {}

// ── VTable ─────────────────────────────────────────────────────────

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .quantized_matmul_f32 = quantizedMatMulF32,
    .alloc_buffer = allocBuffer,
    .free_buffer = freeBuffer,
    .upload = upload,
    .download = download,
    .sync = syncFn,
    .compile_program = compileProgram,
    .execute_program = executeProgram,
    .free_program = freeProgram,
};

// ── Tests ──────────────────────────────────────────────────────────

test "cpu backend host dense matmul" {
    var cpu = CpuBackend.initHostOnly();
    var dst = [_]f32{0} ** 4;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const ok = backend_mod.tryDenseMatMul(f32, cpu.backend(), .{
        .dst = &dst,
        .a = &a,
        .b = &b,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });

    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "cpu backend host quantized matmul" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.initHostOnly();
    const weights = [_]f32{ 1.0, 0.5, -0.5, 1.0, 0.25, -0.25 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var dst = [_]f32{0} ** 4;
    var expected = [_]f32{0} ** 4;

    var qw = try quant.QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);
    qw.matmul(&input, &expected, 2, 2, 3);

    const ok = backend_mod.tryQuantizedMatMul(f32, cpu.backend(), .{
        .dst = &dst,
        .input = &input,
        .weight = backend_mod.quantizedWeightViewF32(qw),
        .M = 2,
        .N = 2,
        .K = 3,
    });

    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &expected, &dst);
}

test "cpu backend device buffer round-trip" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();

    try std.testing.expect(be.caps().device_buffers);

    const buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(buf);

    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    be.uploadSlice(f32, buf, 0, &src);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "cpu backend compile returns null (no device execution)" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();
    const program = backend_mod.DeviceProgram{ .ops = &.{}, .n_buffers = 0, .buffer_sizes = &.{}, .initial_uploads = &.{} };
    try std.testing.expect(be.compileProgram(program) == null);
}

test "host-only backend reports no device buffer support" {
    var cpu = CpuBackend.initHostOnly();
    const be = cpu.backend();
    try std.testing.expect(!be.caps().device_buffers);
    try std.testing.expect(be.allocBuffer(64) == null);
}
