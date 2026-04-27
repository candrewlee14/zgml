//! Backend interface for kernel dispatch and compiled program execution.
//!
//! Two concerns:
//!   1. Host matmul override — swap BLAS implementation during graph execution.
//!   2. Compiled programs — the framework builds a DeviceProgram (list of
//!      DeviceOps with buffer indices), the backend compiles it once, then
//!      executes it per token. Buffer management is internal to the backend.

const std = @import("std");

pub const Device = enum { cpu, metal, cuda, npu, wgpu };
pub const Op = @import("op.zig").Op;

pub const Capabilities = struct {
    compiled_programs: bool = false,
    host_visible_program_memory: bool = false,
    dense_matmul_f32: bool = false,
    dense_matmul_f16: bool = false,
    qmatmul: bool = false,
    fused_elementwise: bool = false,
    max_fused_elementwise_steps: ?u32 = null,
    f16_weight_promotion: bool = false,
    dynamic_program_refresh: bool = false,
    prefill_attention: bool = false,
    decode_attention: bool = false,
    quantized_kv: bool = false,
    command_buffer_execution: bool = false,
    command_stream: CommandStream = .{},
    attention: Attention = .{},

    pub const Attention = struct {
        supported: bool = false,
        max_seq_kv: ?u32 = null,
        max_d_head: ?u32 = null,

        pub fn supports(self: Attention, seq_kv: u32, d_head: u32) bool {
            if (!self.supported) return false;
            if (self.max_seq_kv) |max| if (seq_kv > max) return false;
            if (self.max_d_head) |max| if (d_head > max) return false;
            return true;
        }
    };

    pub const CommandStream = struct {
        stage_commands: bool = false,
        qmatvec_group_size: u32 = 1,
        qmatmul_group_size: u32 = 1,
        qmatmul_sidecars: bool = false,
        qmatmul_cache_sidecars_per_anchor: u32 = 1,
        projection_rope_cache_sidecars: bool = false,
        max_rope_batch: u32 = 1,
        max_movement_batch: u32 = 1,
        max_attention_batch: u32 = 1,
        max_attention_store_batch: u32 = 1,
        max_rope_attention_store_batch: u32 = 1,
        max_elementwise_batch: u32 = 1,
        fuse_repeat_fused_elementwise: bool = false,
    };

    pub const reference_cpu = Capabilities{
        .compiled_programs = true,
        .host_visible_program_memory = true,
        .dense_matmul_f32 = true,
        .qmatmul = true,
        .fused_elementwise = true,
        .dynamic_program_refresh = true,
        .prefill_attention = true,
        .decode_attention = true,
        .attention = .{ .supported = true, .max_d_head = 512 },
    };

    pub const metal = Capabilities{
        .compiled_programs = true,
        .host_visible_program_memory = true,
        .dense_matmul_f32 = true,
        .dense_matmul_f16 = true,
        .qmatmul = true,
        .fused_elementwise = true,
        .max_fused_elementwise_steps = 8,
        .dynamic_program_refresh = true,
        .prefill_attention = true,
        .decode_attention = true,
        .command_buffer_execution = true,
        .command_stream = .{
            .stage_commands = true,
            .qmatvec_group_size = 4,
            .qmatmul_group_size = 4,
            .qmatmul_sidecars = true,
            .qmatmul_cache_sidecars_per_anchor = 8,
            .max_rope_batch = 16,
            .max_movement_batch = 16,
            .max_attention_batch = 16,
            .max_attention_store_batch = 4,
            .max_rope_attention_store_batch = 16,
            .max_elementwise_batch = 8,
            .fuse_repeat_fused_elementwise = true,
        },
        .attention = .{ .supported = true, .max_d_head = 512 },
    };

    pub const wgpu = Capabilities{
        .compiled_programs = true,
        .dense_matmul_f32 = true,
        .qmatmul = true,
        .f16_weight_promotion = true,
        .dynamic_program_refresh = true,
        .prefill_attention = true,
        .decode_attention = true,
        .command_buffer_execution = true,
        .attention = .{ .supported = true, .max_seq_kv = 4096, .max_d_head = 512 },
    };

    pub fn supportsElementwiseOp(_: Capabilities, op: Op) bool {
        return switch (op) {
            .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => true,
            else => false,
        };
    }

    pub fn supportsOp(self: Capabilities, op: DeviceOp) bool {
        if (!self.compiled_programs) return false;
        return switch (op) {
            .elementwise => |e| self.supportsElementwiseOp(e.op),
            .matmul => self.dense_matmul_f32,
            .qmatmul => self.qmatmul,
            .softmax, .layernorm, .rmsnorm, .repeat, .slice_assign, .rope => true,
            .reduce => |r| r.op == .sum or r.op == .max,
            .attention => |att| self.attention.supports(att.seq_kv, att.d_head),
            .fused_elementwise => |fe| {
                if (!self.fused_elementwise) return false;
                if (self.max_fused_elementwise_steps) |max| {
                    if (fe.steps.len > max) return false;
                }
                for (fe.steps) |step| {
                    if (!self.supportsElementwiseOp(step.op)) return false;
                }
                return true;
            },
        };
    }
};

// ── Host kernel specs ──────────────────────────────────────────────

/// Stride/offset parameters for matmul dispatch.
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

pub const DenseMatMulSpecF32 = struct {
    dst: []f32,
    a: []const f32,
    b: []const f32,
    geom: MatMulGeometry,
};

// ── Compiled device programs ───────────────────────────────────────

/// One step in a fused elementwise chain.
pub const FusedEwStep = struct {
    op: Op,
    is_swapped: bool, // true if chain value is in src1 position
    secondary_buf: u16, // buffer index of external operand (binary ops)
    secondary_offset: u32, // offset into that buffer
};

/// A single operation in a device program. Uses buffer indices (u16)
/// instead of pointers — the backend maps indices to its own buffers.
pub const DeviceOp = union(enum) {
    elementwise: struct { op: Op, dst: u16, src0: u16, src1: u16, n: u32, dst_offset: u32 = 0, src0_offset: u32 = 0, src1_offset: u32 = 0 },
    matmul: struct { dst: u16, a: u16, b: u16, geom: MatMulGeometry },
    qmatmul: struct {
        dst: u16,
        input: u16,
        weight_idx: u16,
        M: u32,
        N: u32,
        K: u32,
        input_offset: u32 = 0,
        input_row_stride: u32 = 0,
        dst_offset: u32 = 0,
        dst_row_stride: u32 = 0,
    },
    softmax: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    layernorm: struct { dst: u16, src: u16, rows: u32, cols: u32, eps: f32 = 1e-5, src_offset: u32 = 0, dst_offset: u32 = 0 },
    rmsnorm: struct { dst: u16, src: u16, rows: u32, cols: u32, eps: f32 = 1e-5, src_offset: u32 = 0, dst_offset: u32 = 0 },
    reduce: struct { op: Op, dst: u16, src: u16, n_out: u32, reduce_size: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    repeat: struct { dst: u16, src: u16, n: u32, src_ne: [4]u32, dst_ne: [4]u32, src_strides: [4]u32, dst_strides: [4]u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    slice_assign: struct {
        dst: u16,
        src: u16,
        rows: u32,
        cols: u32,
        dst_base_offset: u32,
        dst_offset: u32,
        dst_row_stride: u32,
        dst_col_stride: u32,
        src_offset: u32,
        src_row_stride: u32,
        src_col_stride: u32,
        patch_stride: u32,
    },
    rope: struct { dst: u16, src: u16, cos_sin: u16, half_d: u32, seq_len: u32, src_off: u32, cs_off: u32, dst_off: u32, src_rs: u32, src_cs: u32, cs_cs: u32 },
    attention: struct {
        dst: u16,
        q: u16,
        k: u16,
        v: u16,
        mask: u16,
        has_mask: bool,
        d_head: u32,
        seq_q: u32,
        seq_kv: u32,
        scale: f32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        mask_off: u32,
        dst_off: u32,
        q_rs: u32,
        q_cs: u32,
        k_rs: u32,
        k_cs: u32,
        v_rs: u32,
        v_cs: u32,
        mask_rs: u32,
        mask_cs: u32,
        dst_rs: u32,
        dst_cs: u32,
    },
    fused_elementwise: struct {
        steps: []const FusedEwStep,
        n: u32,
        dst: u16,
        src: u16,
        dst_offset: u32,
        src_offset: u32,
    },
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

    pub fn isSupportedBy(self: DeviceProgram, capabilities: Capabilities) bool {
        if (!capabilities.compiled_programs) return false;
        if (@as(usize, self.n_buffers) != self.buffer_sizes.len) return false;
        for (self.ops) |op| {
            if (!capabilities.supportsOp(op)) return false;
            if (!self.opBuffersValid(op)) return false;
            switch (op) {
                .qmatmul => |q| {
                    if (@as(usize, q.weight_idx) >= self.qweights.len) return false;
                    const qw = self.qweights[q.weight_idx];
                    if (qw.block_size == 0) return false;
                    if (qw.rows != q.K or qw.cols != q.N) return false;
                    const n_elems: usize = @as(usize, q.K) * @as(usize, q.N);
                    const n_blocks = (n_elems + qw.block_size - 1) / qw.block_size;
                    if (qw.data.len < n_elems or qw.scales.len < n_blocks) return false;
                },
                else => {},
            }
        }
        return true;
    }

    fn hasBuffer(self: DeviceProgram, idx: u16) bool {
        return @as(usize, idx) < self.buffer_sizes.len;
    }

    fn opBuffersValid(self: DeviceProgram, op: DeviceOp) bool {
        return switch (op) {
            .elementwise => |e| self.hasBuffer(e.dst) and self.hasBuffer(e.src0) and self.hasBuffer(e.src1),
            .matmul => |m| self.hasBuffer(m.dst) and self.hasBuffer(m.a) and self.hasBuffer(m.b),
            .qmatmul => |q| self.hasBuffer(q.dst) and self.hasBuffer(q.input),
            .softmax => |s| self.hasBuffer(s.dst) and self.hasBuffer(s.src),
            .layernorm => |l| self.hasBuffer(l.dst) and self.hasBuffer(l.src),
            .rmsnorm => |r| self.hasBuffer(r.dst) and self.hasBuffer(r.src),
            .reduce => |r| self.hasBuffer(r.dst) and self.hasBuffer(r.src),
            .repeat => |rp| self.hasBuffer(rp.dst) and self.hasBuffer(rp.src),
            .slice_assign => |sa| self.hasBuffer(sa.dst) and self.hasBuffer(sa.src),
            .rope => |rr| self.hasBuffer(rr.dst) and self.hasBuffer(rr.src) and self.hasBuffer(rr.cos_sin),
            .attention => |att| self.hasBuffer(att.dst) and self.hasBuffer(att.q) and
                self.hasBuffer(att.k) and self.hasBuffer(att.v) and self.hasBuffer(att.mask),
            .fused_elementwise => |fe| {
                if (!self.hasBuffer(fe.dst) or !self.hasBuffer(fe.src)) return false;
                for (fe.steps) |step| {
                    if (step.op.isBinary() and !self.hasBuffer(step.secondary_buf)) return false;
                }
                return true;
            },
        };
    }
};

// ── Backend ────────────────────────────────────────────────────────

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    name_str: []const u8,
    device_type: Device,
    capabilities: Capabilities,

    pub const CompiledHandle = *anyopaque;

    pub const VTable = struct {
        /// Override dense matmul during graph execution. Returns true if handled.
        dense_matmul_f32: *const fn (ctx: *anyopaque, spec: DenseMatMulSpecF32) bool,
        /// Compile a DeviceProgram into backend-optimized execution.
        compile_program: *const fn (ctx: *anyopaque, program: DeviceProgram) ?CompiledHandle,
        /// Refresh dynamic op parameters before execution.
        refresh_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, ops: []const DeviceOp) void,
        /// Execute a compiled program: upload inputs, dispatch, download outputs.
        execute_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void,
        /// Release compiled program resources.
        free_program: *const fn (ctx: *anyopaque, handle: CompiledHandle) void,
        /// Get pointer to accumulated runtime profile (null if unsupported).
        get_runtime_profile: *const fn (ctx: *anyopaque, handle: CompiledHandle) ?*@import("profile.zig").RuntimeProfile,
    };

    pub fn compileProgram(self: Backend, program: DeviceProgram) ?CompiledHandle {
        if (!self.supportsProgram(program)) return null;
        return self.vtable.compile_program(self.ctx, program);
    }

    pub fn supportsProgram(self: Backend, program: DeviceProgram) bool {
        return program.isSupportedBy(self.capabilities);
    }

    pub fn refreshProgram(self: Backend, handle: CompiledHandle, ops: []const DeviceOp) void {
        self.vtable.refresh_program(self.ctx, handle, ops);
    }

    pub fn executeProgram(self: Backend, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void {
        self.vtable.execute_program(self.ctx, handle, inputs, outputs);
    }

    pub fn freeProgram(self: Backend, handle: CompiledHandle) void {
        self.vtable.free_program(self.ctx, handle);
    }

    pub fn getRuntimeProfile(self: Backend, handle: CompiledHandle) ?*@import("profile.zig").RuntimeProfile {
        return self.vtable.get_runtime_profile(self.ctx, handle);
    }

    pub fn supportsAttention(self: Backend, seq_kv: u32, d_head: u32) bool {
        return self.capabilities.attention.supports(seq_kv, d_head);
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
        .dst = &.{},
        .a = &.{},
        .b = &.{},
        .geom = .{ .M = 0, .N = 0, .K = 0, .a_row_stride = 0, .a_col_stride = 0, .b_row_stride = 0, .b_col_stride = 0, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 0 },
    };
    try std.testing.expect(!tryDenseMatMul(f32, null, dense));
}

test "program support validates capabilities and qweight descriptors" {
    const fused_steps = [_]FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }};
    const fused_ops = [_]DeviceOp{.{ .fused_elementwise = .{ .steps = &fused_steps, .n = 1, .dst = 1, .src = 0, .dst_offset = 0, .src_offset = 0 } }};
    const fused_program = DeviceProgram{ .ops = &fused_ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 1 }, .initial_uploads = &.{} };
    try std.testing.expect(!fused_program.isSupportedBy(Capabilities.wgpu));
    try std.testing.expect(fused_program.isSupportedBy(Capabilities.metal));
    try std.testing.expect(fused_program.isSupportedBy(Capabilities.reference_cpu));

    const many_fused_steps = [_]FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }} ** 9;
    const too_large_fused = DeviceProgram{
        .ops = &.{.{ .fused_elementwise = .{ .steps = &many_fused_steps, .n = 1, .dst = 1, .src = 0, .dst_offset = 0, .src_offset = 0 } }},
        .n_buffers = 2,
        .buffer_sizes = &.{ 1, 1 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!too_large_fused.isSupportedBy(Capabilities.metal));
    try std.testing.expect(too_large_fused.isSupportedBy(Capabilities.reference_cpu));

    const qdata = [_]i8{ 1, 2, 3, 4 };
    const scales = [_]f32{1};
    const qops = [_]DeviceOp{.{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = 1, .N = 2, .K = 2 } }};
    const qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = 4 }};
    const valid_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &qweights };
    try std.testing.expect(valid_qprogram.isSupportedBy(Capabilities.wgpu));

    const bad_qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 2, .block_size = 4 }};
    const bad_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &bad_qweights };
    try std.testing.expect(!bad_qprogram.isSupportedBy(Capabilities.wgpu));
}

test "capability attention limits are explicit" {
    try std.testing.expect(Capabilities.reference_cpu.attention.supports(8192, 512));
    try std.testing.expect(!Capabilities.reference_cpu.attention.supports(1, 513));
    try std.testing.expect(Capabilities.wgpu.attention.supports(4096, 512));
    try std.testing.expect(!Capabilities.wgpu.attention.supports(4097, 512));
}
