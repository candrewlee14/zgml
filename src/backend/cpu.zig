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

// ── Device kernel dispatch ─────────────────────────────────────────

fn deviceBufSlice(comptime T: type, buf: backend_mod.DeviceBuffer) []T {
    const ptr: [*]T = @ptrCast(@alignCast(buf.ptr));
    return ptr[0 .. buf.size / @sizeOf(T)];
}

fn deviceBufConstSlice(comptime T: type, buf: backend_mod.DeviceBuffer) []const T {
    const ptr: [*]const T = @ptrCast(@alignCast(buf.ptr));
    return ptr[0 .. buf.size / @sizeOf(T)];
}

fn deviceMatMulF32(_: *anyopaque, spec: backend_mod.DeviceMatMulSpecF32) bool {
    const g = spec.geom;
    forward.blasSgemm(deviceBufSlice(f32, spec.dst), deviceBufConstSlice(f32, spec.a), deviceBufConstSlice(f32, spec.b), g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

fn deviceQuantizedMatMulF32(_: *anyopaque, spec: backend_mod.DeviceQuantizedMatMulSpecF32) bool {
    const Weight = quant.QuantizedWeight(f32);
    const w = spec.weight;
    const weight = Weight{
        .data = deviceBufConstSlice(i8, w.data),
        .scales = deviceBufConstSlice(f32, w.scales),
        .rows = w.rows,
        .cols = w.cols,
        .block_size = w.block_size,
    };
    weight.matmul(
        deviceBufConstSlice(f32, spec.input),
        deviceBufSlice(f32, spec.dst),
        spec.M,
        spec.N,
        spec.K,
    );
    return true;
}

// ── VTable ─────────────────────────────────────────────────────────

fn deviceCompute(_: *anyopaque, _: backend_mod.DeviceComputeSpec) bool {
    return false;
}

fn deviceFused(_: *anyopaque, _: backend_mod.DeviceFusedSpec) bool {
    return false;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .quantized_matmul_f32 = quantizedMatMulF32,
    .alloc_buffer = allocBuffer,
    .free_buffer = freeBuffer,
    .upload = upload,
    .download = download,
    .sync = syncFn,
    .device_matmul_f32 = deviceMatMulF32,
    .device_quantized_matmul_f32 = deviceQuantizedMatMulF32,
    .device_compute = deviceCompute,
    .device_fused = deviceFused,
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

test "cpu backend device dense matmul" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();

    // A: 2x3, B: 3x2, dst: 2x2
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const a_buf = be.allocSlice(f32, 6) orelse return error.OutOfMemory;
    defer be.freeBuffer(a_buf);
    const b_buf = be.allocSlice(f32, 6) orelse return error.OutOfMemory;
    defer be.freeBuffer(b_buf);
    const dst_buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(dst_buf);

    be.uploadSlice(f32, a_buf, 0, &a_data);
    be.uploadSlice(f32, b_buf, 0, &b_data);

    const ok = be.deviceMatMul(.{
        .dst = dst_buf,
        .a = a_buf,
        .b = b_buf,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });
    try std.testing.expect(ok);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, dst_buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "cpu backend device quantized matmul" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();

    // Quantize weights on host, upload to device, run device qmatmul.
    const weights = [_]f32{ 1.0, 0.5, -0.5, 1.0, 0.25, -0.25 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var expected = [_]f32{0} ** 4;

    var qw = try quant.QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);
    qw.matmul(&input, &expected, 2, 2, 3);

    // Upload quantized weight data and scales to device.
    const w_data_buf = be.allocBuffer(qw.data.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(w_data_buf);
    const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
    be.uploadBytes(w_data_buf, 0, i8_as_u8[0..qw.data.len]);

    const w_scales_buf = be.allocSlice(f32, qw.scales.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(w_scales_buf);
    be.uploadSlice(f32, w_scales_buf, 0, qw.scales);

    // Upload input, allocate output.
    const input_buf = be.allocSlice(f32, input.len) orelse return error.OutOfMemory;
    defer be.freeBuffer(input_buf);
    be.uploadSlice(f32, input_buf, 0, &input);

    const dst_buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(dst_buf);

    const ok = be.deviceQuantizedMatMul(.{
        .dst = dst_buf,
        .input = input_buf,
        .weight = .{
            .data = w_data_buf,
            .scales = w_scales_buf,
            .rows = 3,
            .cols = 2,
            .block_size = 32,
        },
        .M = 2,
        .N = 2,
        .K = 3,
    });
    try std.testing.expect(ok);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, dst_buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &expected, &dst);
}

test "host-only backend reports no device buffer support" {
    var cpu = CpuBackend.initHostOnly();
    const be = cpu.backend();
    try std.testing.expect(!be.caps().device_buffers);
    try std.testing.expect(be.allocBuffer(64) == null);
}
