//! Backend interface for kernel dispatch and compiled program execution.
//!
//! Two concerns:
//!   1. Host matmul override — swap BLAS implementation during graph execution.
//!   2. Compiled programs — the framework builds a DeviceProgram (list of
//!      DeviceOps with buffer indices), the backend compiles it once, then
//!      executes it per token. Buffer management is internal to the backend.

const std = @import("std");

pub const Device = enum { cpu, metal, cuda, npu, wgpu };

// ── Host kernel specs ──────────────────────────────────────────────

/// Stride/offset parameters for matmul dispatch.
pub const MatMulGeometry = struct {
    M: usize, N: usize, K: usize,
    a_row_stride: usize, a_col_stride: usize,
    b_row_stride: usize, b_col_stride: usize,
    a_offset: usize, b_offset: usize,
    dst_offset: usize, dst_row_stride: usize,
};

pub const DenseMatMulSpecF32 = struct {
    dst: []f32,
    a: []const f32,
    b: []const f32,
    geom: MatMulGeometry,
};

// ── Compiled device programs ───────────────────────────────────────

pub const Op = @import("op.zig").Op;

/// A single operation in a device program. Uses buffer indices (u16)
/// instead of pointers — the backend maps indices to its own buffers.
pub const DeviceOp = union(enum) {
    elementwise: struct { op: Op, dst: u16, src0: u16, src1: u16, n: u32, dst_offset: u32 = 0, src0_offset: u32 = 0, src1_offset: u32 = 0 },
    matmul: struct { dst: u16, a: u16, b: u16, geom: MatMulGeometry },
    qmatmul: struct { dst: u16, input: u16, weight_idx: u16, M: u32, N: u32, K: u32 },
    softmax: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    layernorm: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    reduce: struct { op: Op, dst: u16, src: u16, n_out: u32, reduce_size: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    repeat: struct { dst: u16, src: u16, n: u32, src_ne: [4]u32, dst_ne: [4]u32, src_strides: [4]u32, dst_strides: [4]u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    slice_assign: struct { dst: u16, src: u16, n: u32, dst_offset: u32, dst_stride: u32, src_offset: u32, src_stride: u32 },
};

/// Per-step host↔device transfer descriptor.
pub const ProgramIO = struct {
    buf_idx: u16,
    offset: u32 = 0,
    host_ptr: [*]u8,
    size: u32,
};

/// Quantized weight descriptor for compile-time upload.
pub const QuantizedWeightUpload = struct {
    data: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
    block_size: usize,
};

/// Backend-agnostic program IR. Built once from the execution plan,
/// compiled by the backend, executed per token.
pub const DeviceProgram = struct {
    ops: []const DeviceOp,
    n_buffers: u16,
    buffer_sizes: []const usize,
    initial_uploads: []const ProgramIO,
    qweights: []const QuantizedWeightUpload = &.{},
};

// ── Backend ────────────────────────────────────────────────────────

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    name_str: []const u8,
    device_type: Device,

    pub const CompiledHandle = *anyopaque;

    pub const VTable = struct {
        /// Override dense matmul during graph execution. Returns true if handled.
        dense_matmul_f32: *const fn (ctx: *anyopaque, spec: DenseMatMulSpecF32) bool,
        /// Compile a DeviceProgram into backend-optimized execution.
        compile_program: *const fn (ctx: *anyopaque, program: DeviceProgram) ?CompiledHandle,
        /// Execute a compiled program: upload inputs, dispatch, download outputs.
        execute_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void,
        /// Release compiled program resources.
        free_program: *const fn (ctx: *anyopaque, handle: CompiledHandle) void,
    };

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

// ── Dispatch helper ────────────────────────────────────────────────

pub fn tryDenseMatMul(comptime T: type, backend_opt: ?Backend, spec: DenseMatMulSpecF32) bool {
    if (T != f32) return false;
    const be = backend_opt orelse return false;
    return be.vtable.dense_matmul_f32(be.ctx, spec);
}

test "dispatch helper returns false when no backend is configured" {
    const dense = DenseMatMulSpecF32{
        .dst = &.{}, .a = &.{}, .b = &.{},
        .geom = .{ .M = 0, .N = 0, .K = 0, .a_row_stride = 0, .a_col_stride = 0, .b_row_stride = 0, .b_col_stride = 0, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 0 },
    };
    try std.testing.expect(!tryDenseMatMul(f32, null, dense));
}
