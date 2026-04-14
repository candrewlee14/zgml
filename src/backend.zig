//! Backend interface for kernel dispatch and compiled program execution.
//!
//! Three levels of integration:
//!   1. Host kernel dispatch — override matmul/qmatmul with host pointers.
//!   2. Device buffers — alloc/free/transfer for data management.
//!   3. Compiled programs — the framework builds a DeviceProgram (list of
//!      DeviceOps using buffer indices), the backend compiles it once, then
//!      executes it per token with minimal overhead.

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

// ── Compiled device programs ───────────────────────────────────────

pub const Op = @import("op.zig").Op;

/// A single operation in a device program. Uses buffer indices (u16)
/// instead of pointers — the backend maps indices to device buffers.
pub const DeviceOp = union(enum) {
    /// Elementwise: add, mul, neg, exp, sqrt, recip, gelu, etc.
    elementwise: struct { op: Op, dst: u16, src0: u16, src1: u16, n: u32, dst_offset: u32 = 0, src0_offset: u32 = 0, src1_offset: u32 = 0 },
    /// Dense matmul with full geometry.
    matmul: struct { dst: u16, a: u16, b: u16, geom: MatMulGeometry },
    /// Quantized matmul (weight index refers to a quantized weight, not a buffer).
    qmatmul: struct { dst: u16, input: u16, weight_idx: u16, M: u32, N: u32, K: u32 },
    /// Fused softmax: one kernel for max → shift → exp → sum → normalize.
    softmax: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    /// Fused layer norm: mean → center → var → normalize.
    layernorm: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    /// Reduce (sum, max) along innermost dim.
    reduce: struct { op: Op, dst: u16, src: u16, n_out: u32, reduce_size: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    /// Broadcast repeat.
    repeat: struct { dst: u16, src: u16, n: u32, src_ne: [4]u32, dst_ne: [4]u32, src_strides: [4]u32, dst_strides: [4]u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    /// KV cache column write.
    slice_assign: struct { dst: u16, src: u16, n: u32, dst_offset: u32, dst_stride: u32, src_offset: u32, src_stride: u32 },
};

/// Host↔device transfer for per-step inputs/outputs.
pub const ProgramIO = struct {
    buf_idx: u16,
    offset: u32 = 0,
    host_ptr: [*]u8,
    size: u32,
};

/// A compiled device program — built once from the execution plan,
/// executed per token. Backend-agnostic: expressed in buffer indices
/// and op codes. The backend compiles this into optimized execution.
pub const DeviceProgram = struct {
    ops: []const DeviceOp,
    n_buffers: u16,
    buffer_sizes: []const usize,
    /// Initial data to upload at compile time (weights, zeros).
    initial_uploads: []const ProgramIO,
    /// Quantized weight views for qmatmul ops (indexed by DeviceOp.qmatmul.weight_idx).
    qweights: []const QuantizedWeightUpload = &.{},
};

/// Quantized weight to upload at compile time (host data + device buffer info).
pub const QuantizedWeightUpload = struct {
    data: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
    block_size: usize,
};

// ── Backend ────────────────────────────────────────────────────────

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    /// Static identity — set once at construction, never changes.
    name_str: []const u8,
    device_type: Device,
    capabilities: BackendCaps,

    /// Opaque handle to a backend-compiled program.
    pub const CompiledHandle = *anyopaque;

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

        // Compiled program execution — replaces per-op device dispatch.
        // compile: build backend-optimized execution from a DeviceProgram.
        // execute: run the compiled program (upload inputs, dispatch, download outputs).
        // free: release compiled program resources.
        compile_program: *const fn (ctx: *anyopaque, program: DeviceProgram) ?CompiledHandle,
        execute_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void,
        free_program: *const fn (ctx: *anyopaque, handle: CompiledHandle) void,
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

    pub fn compileProgram(self: Backend, program: DeviceProgram) ?CompiledHandle {
        return self.vtable.compile_program(self.ctx, program);
    }

    pub fn executeProgram(self: Backend, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void {
        self.vtable.execute_program(self.ctx, handle, inputs, outputs);
    }

    pub fn freeProgram(self: Backend, handle: CompiledHandle) void {
        self.vtable.free_program(self.ctx, handle);
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
