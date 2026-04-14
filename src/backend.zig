//! Backend interface for kernel dispatch and device buffer management.
//!
//! Two levels of integration:
//!   1. Host kernel dispatch — override matmul/qmatmul with host pointers.
//!   2. Device buffers — alloc/free/transfer + device kernel dispatch.
//!
//! The graph and autodiff layers remain backend-agnostic. Only the
//! execution hot path and frozen inference plans use the backend.

const std = @import("std");

pub const Device = enum {
    cpu,
    metal,
    cuda,
    npu,
};

pub const BackendCaps = struct {
    device_buffers: bool = false,
};

// ── Shared matmul geometry ──────────────────────────────────────────

/// Stride/offset parameters shared by all matmul dispatch specs.
pub const MatMulGeometry = struct {
    M: usize,
    N: usize,
    K: usize,
    a_row_stride: usize,
    a_col_stride: usize,
    b_row_stride: usize,
    b_col_stride: usize,
    a_offset: usize,
    b_offset: usize,
    dst_offset: usize,
    dst_row_stride: usize,
};

// ── Host kernel specs ──────────────────────────────────────────────

/// Single dense GEMM call over already-resolved row/column strides.
pub const DenseMatMulSpecF32 = struct {
    dst: []f32,
    a: []const f32,
    b: []const f32,
    geom: MatMulGeometry,
};

pub const QuantizedWeightViewF32 = struct {
    data: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
    block_size: usize,
};

pub const QuantizedMatMulSpecF32 = struct {
    dst: []f32,
    input: []const f32,
    weight: QuantizedWeightViewF32,
    M: usize,
    N: usize,
    K: usize,
};

// ── Device buffers ─────────────────────────────────────────────────

/// Opaque handle to a backend-owned buffer (host or device memory).
/// For CPU backends this wraps a host allocation; for GPU backends
/// it wraps a device-resident buffer (e.g. MTLBuffer, CUdeviceptr).
pub const DeviceBuffer = struct {
    ptr: *anyopaque,
    size: usize, // bytes
};

// ── Device kernel specs ────────────────────────────────────────────

pub const DeviceMatMulSpecF32 = struct {
    dst: DeviceBuffer,
    a: DeviceBuffer,
    b: DeviceBuffer,
    geom: MatMulGeometry,
};

pub const DeviceQuantizedWeightView = struct {
    data: DeviceBuffer,
    scales: DeviceBuffer,
    rows: usize,
    cols: usize,
    block_size: usize,
};

pub const DeviceQuantizedMatMulSpecF32 = struct {
    dst: DeviceBuffer,
    input: DeviceBuffer,
    weight: DeviceQuantizedWeightView,
    M: usize,
    N: usize,
    K: usize,
};

// ── Backend ────────────────────────────────────────────────────────

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    /// Static identity — set once at construction, never changes.
    name_str: []const u8,
    device_type: Device,
    capabilities: BackendCaps,

    pub const VTable = struct {
        // Host kernel dispatch — returns true if handled.
        dense_matmul_f32: *const fn (ctx: *anyopaque, spec: DenseMatMulSpecF32) bool,
        quantized_matmul_f32: *const fn (ctx: *anyopaque, spec: QuantizedMatMulSpecF32) bool,

        // Device buffer lifecycle
        alloc_buffer: *const fn (ctx: *anyopaque, size: usize) ?DeviceBuffer,
        free_buffer: *const fn (ctx: *anyopaque, buf: DeviceBuffer) void,
        upload: *const fn (ctx: *anyopaque, dst: DeviceBuffer, dst_byte_offset: usize, src: []const u8) void,
        download: *const fn (ctx: *anyopaque, dst: []u8, src: DeviceBuffer, src_byte_offset: usize) void,
        sync: *const fn (ctx: *anyopaque) void,

        // Device kernel dispatch — returns true if handled.
        device_matmul_f32: *const fn (ctx: *anyopaque, spec: DeviceMatMulSpecF32) bool,
        device_quantized_matmul_f32: *const fn (ctx: *anyopaque, spec: DeviceQuantizedMatMulSpecF32) bool,
    };

    // ── Convenience methods ────────────────────────────────────

    pub fn name(self: Backend) []const u8 {
        return self.name_str;
    }

    pub fn device(self: Backend) Device {
        return self.device_type;
    }

    pub fn caps(self: Backend) BackendCaps {
        return self.capabilities;
    }

    pub fn allocBuffer(self: Backend, size: usize) ?DeviceBuffer {
        return self.vtable.alloc_buffer(self.ctx, size);
    }

    pub fn allocSlice(self: Backend, comptime T: type, len: usize) ?DeviceBuffer {
        return self.allocBuffer(len * @sizeOf(T));
    }

    pub fn freeBuffer(self: Backend, buf: DeviceBuffer) void {
        self.vtable.free_buffer(self.ctx, buf);
    }

    pub fn uploadBytes(self: Backend, dst: DeviceBuffer, dst_byte_offset: usize, src: []const u8) void {
        self.vtable.upload(self.ctx, dst, dst_byte_offset, src);
    }

    pub fn downloadBytes(self: Backend, dst: []u8, src: DeviceBuffer, src_byte_offset: usize) void {
        self.vtable.download(self.ctx, dst, src, src_byte_offset);
    }

    pub fn uploadSlice(self: Backend, comptime T: type, dst: DeviceBuffer, elem_offset: usize, src: []const T) void {
        self.uploadBytes(dst, elem_offset * @sizeOf(T), std.mem.sliceAsBytes(src));
    }

    pub fn downloadSlice(self: Backend, comptime T: type, dst: []T, src: DeviceBuffer, elem_offset: usize) void {
        self.downloadBytes(std.mem.sliceAsBytes(dst), src, elem_offset * @sizeOf(T));
    }

    pub fn sync(self: Backend) void {
        self.vtable.sync(self.ctx);
    }

    pub fn deviceMatMul(self: Backend, spec: DeviceMatMulSpecF32) bool {
        return self.vtable.device_matmul_f32(self.ctx, spec);
    }

    pub fn deviceQuantizedMatMul(self: Backend, spec: DeviceQuantizedMatMulSpecF32) bool {
        return self.vtable.device_quantized_matmul_f32(self.ctx, spec);
    }
};

// ── Dispatch helpers ───────────────────────────────────────────────

pub fn tryDenseMatMul(comptime T: type, backend_opt: ?Backend, spec: DenseMatMulSpecF32) bool {
    if (T != f32) return false;
    const be = backend_opt orelse return false;
    return be.vtable.dense_matmul_f32(be.ctx, spec);
}

pub fn tryQuantizedMatMul(comptime T: type, backend_opt: ?Backend, spec: QuantizedMatMulSpecF32) bool {
    if (T != f32) return false;
    const be = backend_opt orelse return false;
    return be.vtable.quantized_matmul_f32(be.ctx, spec);
}

pub fn quantizedWeightViewF32(weight: anytype) QuantizedWeightViewF32 {
    return .{
        .data = weight.data,
        .scales = weight.scales,
        .rows = weight.rows,
        .cols = weight.cols,
        .block_size = weight.block_size,
    };
}

// ── Tests ──────────────────────────────────────────────────────────

test "dispatch helpers return false when no backend is configured" {
    const dense = DenseMatMulSpecF32{
        .dst = &.{},
        .a = &.{},
        .b = &.{},
        .geom = .{ .M = 0, .N = 0, .K = 0, .a_row_stride = 0, .a_col_stride = 0, .b_row_stride = 0, .b_col_stride = 0, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 0 },
    };
    try std.testing.expect(!tryDenseMatMul(f32, null, dense));
}
