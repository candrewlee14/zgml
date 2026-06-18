//! Backend interface for kernel dispatch and compiled program execution.
//!
//! Two concerns:
//!   1. Host matmul override — swap BLAS implementation during graph execution.
//!   2. Compiled programs — the framework builds a DeviceProgram (list of
//!      DeviceOps with buffer indices), the backend compiles it once, then
//!      executes it per token. Buffer management is internal to the backend.

const std = @import("std");

pub const Device = enum { cpu, metal, webgpu };
pub const Op = @import("op.zig").Op;

pub const Capabilities = struct {
    compiled_programs: bool = false,
    executes_programs: bool = false,
    dense_matmul_f32: bool = false,
    elementwise: bool = false,
    qmatmul: bool = false,
    runtime_qweights: bool = false,
    softmax: bool = false,
    logsoftmax: bool = false,
    layernorm: bool = false,
    rmsnorm: bool = false,
    reduce: bool = false,
    repeat: bool = false,
    conv2d: bool = false,
    gather_rows: bool = false,
    slice_assign: bool = false,
    rope: bool = false,
    external_resources: bool = false,
    fused_elementwise: bool = false,
    max_fused_elementwise_steps: ?u32 = null,
    attention: Attention = .{},

    const Attention = struct {
        supported: bool = false,
        max_seq_kv: ?u32 = null,
        max_d_head: ?u32 = null,

        fn supports(self: Attention, seq_kv: u32, d_head: u32) bool {
            if (!self.supported) return false;
            if (self.max_seq_kv) |max| if (seq_kv > max) return false;
            if (self.max_d_head) |max| if (d_head > max) return false;
            return true;
        }
    };

    pub const reference_cpu = Capabilities{
        .compiled_programs = true,
        .executes_programs = true,
        .dense_matmul_f32 = true,
        .elementwise = true,
        .qmatmul = true,
        .runtime_qweights = true,
        .softmax = true,
        .logsoftmax = true,
        .layernorm = true,
        .rmsnorm = true,
        .reduce = true,
        .repeat = true,
        .conv2d = true,
        .gather_rows = true,
        .slice_assign = true,
        .rope = true,
        .fused_elementwise = true,
        .attention = .{ .supported = true, .max_d_head = 512 },
    };

    pub const metal = Capabilities{
        .compiled_programs = true,
        .executes_programs = true,
        .dense_matmul_f32 = true,
        .elementwise = true,
        .qmatmul = true,
        .runtime_qweights = true,
        .softmax = true,
        .layernorm = true,
        .rmsnorm = true,
        .reduce = true,
        .repeat = true,
        .gather_rows = false,
        .slice_assign = true,
        .rope = true,
        .fused_elementwise = true,
        .max_fused_elementwise_steps = 8,
        .attention = .{ .supported = true, .max_d_head = 512 },
    };

    pub const webgpu = Capabilities{
        .compiled_programs = true,
        .executes_programs = true,
        .dense_matmul_f32 = true,
        .elementwise = true,
        .qmatmul = true,
        .runtime_qweights = true,
        .softmax = true,
        .layernorm = true,
        .rmsnorm = true,
        .reduce = true,
        .repeat = true,
        .gather_rows = false,
        .slice_assign = true,
        .rope = true,
        .fused_elementwise = true,
        .max_fused_elementwise_steps = 8,
        .attention = .{ .supported = true, .max_d_head = 512 },
    };

    pub const webgpu_compile_only = blk: {
        var caps = Capabilities.webgpu;
        caps.executes_programs = false;
        break :blk caps;
    };

    pub const webgpu_resource_plan = blk: {
        var caps = Capabilities.webgpu_compile_only;
        caps.external_resources = true;
        break :blk caps;
    };

    fn supportsElementwiseOp(_: Capabilities, op: Op) bool {
        return switch (op) {
            .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu, .sqr => true,
            else => false,
        };
    }

    pub fn supportsOp(self: Capabilities, op: DeviceOp) bool {
        if (!self.compiled_programs) return false;
        return switch (op) {
            .elementwise => |e| self.elementwise and self.supportsElementwiseOp(e.op),
            .matmul => self.dense_matmul_f32,
            .qmatmul => self.qmatmul,
            .softmax => self.softmax,
            .logsoftmax => self.logsoftmax,
            .layernorm => self.layernorm,
            .rmsnorm => self.rmsnorm,
            .repeat => self.repeat,
            .conv2d => self.conv2d,
            .max_pool2d => self.repeat,
            .avg_pool2d => self.repeat,
            .gather_rows => self.gather_rows,
            .slice_assign => self.slice_assign,
            .rope => self.rope,
            .reduce => |r| self.reduce and (r.op == .sum or r.op == .max),
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
    logsoftmax: struct { dst: u16, src: u16, rows: u32, cols: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    layernorm: struct { dst: u16, src: u16, rows: u32, cols: u32, eps: f32 = 1e-5, src_offset: u32 = 0, dst_offset: u32 = 0 },
    rmsnorm: struct { dst: u16, src: u16, rows: u32, cols: u32, eps: f32 = 1e-5, src_offset: u32 = 0, dst_offset: u32 = 0 },
    reduce: struct { op: Op, dst: u16, src: u16, n_out: u32, reduce_size: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    conv2d: struct {
        dst: u16,
        src: u16,
        weight: u16,
        bias: u16 = std.math.maxInt(u16),
        out_w: u32,
        out_h: u32,
        in_w: u32,
        in_h: u32,
        in_channels: u32,
        out_channels: u32,
        kernel_w: u32,
        kernel_h: u32,
        batch: u32,
        src_offset: u32 = 0,
        weight_offset: u32 = 0,
        bias_offset: u32 = 0,
        dst_offset: u32 = 0,
        relu: bool = false,
    },
    max_pool2d: struct { dst: u16, src: u16, out_w: u32, out_h: u32, channels: u32, batch: u32, src_w: u32, src_h: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    avg_pool2d: struct { dst: u16, src: u16, out_w: u32, out_h: u32, channels: u32, batch: u32, src_w: u32, src_h: u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    repeat: struct { dst: u16, src: u16, n: u32, src_ne: [4]u32, dst_ne: [4]u32, src_strides: [4]u32, dst_strides: [4]u32, src_offset: u32 = 0, dst_offset: u32 = 0 },
    gather_rows: struct { dst: u16, src: u16, indices: u16, width: u32, count: u32, src_rows: u32, src_offset: u32 = 0, indices_offset: u32 = 0, dst_offset: u32 = 0, src_row_stride: u32 = 0, dst_row_stride: u32 = 0 },
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
        patch_seq_kv: bool = false,
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

/// Token window patched between executions of a compiled program.
pub const RuntimeWindow = struct {
    position: u32,
    len: u32,

    pub fn init(position: usize, len: usize) !RuntimeWindow {
        const end = std.math.add(usize, position, len) catch return error.InvalidRuntimeWindow;
        if (end > std.math.maxInt(u32)) return error.InvalidRuntimeWindow;
        return .{
            .position = std.math.cast(u32, position) orelse return error.InvalidRuntimeWindow,
            .len = std.math.cast(u32, len) orelse return error.InvalidRuntimeWindow,
        };
    }

    pub fn attentionSeqKv(self: RuntimeWindow) ?u32 {
        return std.math.add(u32, self.position, self.len) catch null;
    }
};

pub const RuntimePatchStatus = enum {
    unchanged,
    changed,
    invalid,
};

/// Shape of the runtime copy-and-patch holes in a compiled program.
pub const RuntimePatchShape = struct {
    runtime_patch_holes: u32 = 0,
    runtime_patch_cache_write_pos_holes: u32 = 0,
    runtime_patch_attention_seq_kv_holes: u32 = 0,
    runtime_patch_stencil_hash: u64 = 0,

    fn fromCounts(cache_write_pos: u32, attention_seq_kv: u32, stencil_hash: u64) RuntimePatchShape {
        return .{
            .runtime_patch_holes = cache_write_pos + attention_seq_kv,
            .runtime_patch_cache_write_pos_holes = cache_write_pos,
            .runtime_patch_attention_seq_kv_holes = attention_seq_kv,
            .runtime_patch_stencil_hash = stencil_hash,
        };
    }

    pub fn expectedDynamic(cache_write_pos: u32, attention_seq_kv: u32) RuntimePatchShape {
        return fromCounts(cache_write_pos, attention_seq_kv, 0);
    }

    pub fn actual(cache_write_pos: u32, attention_seq_kv: u32, stencil_hash: u64) RuntimePatchShape {
        return fromCounts(cache_write_pos, attention_seq_kv, stencil_hash);
    }

    pub fn matches(self: RuntimePatchShape, observed: RuntimePatchShape) bool {
        if (observed.runtime_patch_holes != self.runtime_patch_holes or
            observed.runtime_patch_cache_write_pos_holes != self.runtime_patch_cache_write_pos_holes or
            observed.runtime_patch_attention_seq_kv_holes != self.runtime_patch_attention_seq_kv_holes) return false;
        if (self.runtime_patch_stencil_hash != 0) return observed.runtime_patch_stencil_hash == self.runtime_patch_stencil_hash;
        return if (self.runtime_patch_holes == 0)
            observed.runtime_patch_stencil_hash == 0
        else
            observed.runtime_patch_stencil_hash != 0;
    }

    pub fn merge(self: RuntimePatchShape, other: RuntimePatchShape) RuntimePatchShape {
        return .{
            .runtime_patch_holes = @max(self.runtime_patch_holes, other.runtime_patch_holes),
            .runtime_patch_cache_write_pos_holes = @max(self.runtime_patch_cache_write_pos_holes, other.runtime_patch_cache_write_pos_holes),
            .runtime_patch_attention_seq_kv_holes = @max(self.runtime_patch_attention_seq_kv_holes, other.runtime_patch_attention_seq_kv_holes),
            .runtime_patch_stencil_hash = if (self.runtime_patch_stencil_hash == 0)
                other.runtime_patch_stencil_hash
            else if (other.runtime_patch_stencil_hash == 0 or other.runtime_patch_stencil_hash == self.runtime_patch_stencil_hash)
                self.runtime_patch_stencil_hash
            else
                0,
        };
    }
};

/// Per-step host↔device transfer descriptor.
pub const ProgramIO = struct {
    buf_idx: u16,
    offset: u32 = 0,
    host_ptr: [*]u8 = undefined,
    resource: ?ExternalResource = null,
    size: u32,

    pub const ExternalResource = struct {
        placement: Device,
        handle: usize,
        byte_offset: u32 = 0,
        byte_len: u32,
        access: Access = .read_write,

        pub const Access = struct {
            read: bool = true,
            write: bool = true,

            pub const read_only: Access = .{ .read = true, .write = false };
            pub const write_only: Access = .{ .read = false, .write = true };
            pub const read_write: Access = .{ .read = true, .write = true };

            pub const read_flag: u32 = 1 << 0;
            pub const write_flag: u32 = 1 << 1;
            pub const read_write_flags: u32 = read_flag | write_flag;

            pub fn fromFlags(flags: u32) ?Access {
                const effective = if (flags == 0) read_write_flags else flags;
                if ((effective & ~read_write_flags) != 0) return null;
                return .{
                    .read = (effective & read_flag) != 0,
                    .write = (effective & write_flag) != 0,
                };
            }

            pub fn toFlags(self: Access) u32 {
                var flags: u32 = 0;
                if (self.read) flags |= read_flag;
                if (self.write) flags |= write_flag;
                return flags;
            }
        };

        pub fn canRead(self: ExternalResource) bool {
            return self.access.read;
        }

        pub fn canWrite(self: ExternalResource) bool {
            return self.access.write;
        }
    };

    pub fn host(buf_idx: u16, offset: u32, ptr: [*]u8, size: u32) ProgramIO {
        return .{ .buf_idx = buf_idx, .offset = offset, .host_ptr = ptr, .size = size };
    }

    pub fn external(buf_idx: u16, offset: u32, resource: ExternalResource, size: u32) ProgramIO {
        return .{ .buf_idx = buf_idx, .offset = offset, .resource = resource, .size = size };
    }

    pub fn isHost(self: ProgramIO) bool {
        return self.resource == null;
    }

    pub fn hostSlice(self: ProgramIO) ?[]u8 {
        if (!self.isHost()) return null;
        return self.host_ptr[0..self.size];
    }

    pub fn resourceFits(self: ProgramIO) bool {
        const resource = self.resource orelse return true;
        return self.size <= resource.byte_len and resource.byte_offset <= resource.byte_len - self.size;
    }

    pub fn resourceCanRead(self: ProgramIO) bool {
        const resource = self.resource orelse return true;
        return resource.canRead();
    }

    pub fn resourceCanWrite(self: ProgramIO) bool {
        const resource = self.resource orelse return true;
        return resource.canWrite();
    }
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

    fn isStructurallyValid(self: DeviceProgram, capabilities: Capabilities) bool {
        if (!capabilities.compiled_programs) return false;
        if (@as(usize, self.n_buffers) != self.buffer_sizes.len) return false;
        for (self.buffer_sizes) |size| _ = std.math.mul(usize, size, @sizeOf(f32)) catch return false;
        if (!BufferBounds.programIOListValid(self.buffer_sizes, self.initial_uploads)) return false;
        if (!capabilities.external_resources and !BufferBounds.programIOListHostOnly(self.initial_uploads)) return false;
        if (capabilities.external_resources and !BufferBounds.programIOListResourcesCanRead(self.initial_uploads)) return false;
        for (self.qweights) |qw| if (!qweightStorageValid(qw)) return false;
        for (self.ops) |op| {
            if (!self.opBuffersValid(op)) return false;
            switch (op) {
                .qmatmul => |q| {
                    if (@as(usize, q.weight_idx) >= self.qweights.len) return false;
                    const qw = self.qweights[q.weight_idx];
                    if (qw.rows != q.K or qw.cols != q.N) return false;
                },
                else => {},
            }
        }
        return true;
    }

    fn opsSupportedBy(self: DeviceProgram, capabilities: Capabilities) bool {
        for (self.ops) |op| if (!capabilities.supportsOp(op)) return false;
        return true;
    }

    fn isSupportedBy(self: DeviceProgram, capabilities: Capabilities) bool {
        return self.isStructurallyValid(capabilities) and self.opsSupportedBy(capabilities);
    }

    fn qweightStorageValid(qw: QuantizedWeightUpload) bool {
        if (qw.block_size == 0) return false;
        if (qw.block_size > std.math.maxInt(u32)) return false;
        const n_elems = std.math.mul(usize, qw.rows, qw.cols) catch return false;
        const n_blocks = if (n_elems == 0) 0 else ((n_elems - 1) / qw.block_size) + 1;
        if (qw.data.len < n_elems or qw.scales.len < n_blocks) return false;
        _ = std.math.mul(usize, qw.scales.len, @sizeOf(f32)) catch return false;
        const blocks_per_row = if (qw.rows == 0) 0 else ((qw.rows - 1) / qw.block_size) + 1;
        const transposed_scales = std.math.mul(usize, qw.cols, blocks_per_row) catch return false;
        _ = std.math.mul(usize, transposed_scales, @sizeOf(f32)) catch return false;
        return true;
    }

    fn hasBuffer(self: DeviceProgram, idx: u16) bool {
        return @as(usize, idx) < self.buffer_sizes.len;
    }

    fn rangeFits(self: DeviceProgram, idx: u16, offset: usize, len: usize) bool {
        return BufferBounds.rangeFits(self.buffer_sizes, idx, offset, len);
    }

    fn dense2Fits(self: DeviceProgram, idx: u16, offset: usize, rows: usize, cols: usize) bool {
        return BufferBounds.dense2Fits(self.buffer_sizes, idx, offset, rows, cols);
    }

    fn strided2Fits(self: DeviceProgram, idx: u16, offset: usize, rows: usize, cols: usize, row_stride: usize, col_stride: usize) bool {
        return BufferBounds.strided2Fits(self.buffer_sizes, idx, offset, rows, cols, row_stride, col_stride);
    }

    fn strided4Fits(self: DeviceProgram, idx: u16, offset: usize, ne: [4]u32, strides: [4]u32) bool {
        return BufferBounds.strided4Fits(self.buffer_sizes, idx, offset, ne, strides);
    }

    fn opBuffersValid(self: DeviceProgram, op: DeviceOp) bool {
        return switch (op) {
            .elementwise => |e| self.hasBuffer(e.dst) and self.hasBuffer(e.src0) and self.hasBuffer(e.src1) and
                self.rangeFits(e.dst, e.dst_offset, @intCast(e.n)) and self.rangeFits(e.src0, e.src0_offset, @intCast(e.n)) and
                (!e.op.isBinary() or self.rangeFits(e.src1, e.src1_offset, @intCast(e.n))),
            .matmul => |m| blk: {
                const g = m.geom;
                break :blk self.hasBuffer(m.dst) and self.hasBuffer(m.a) and self.hasBuffer(m.b) and
                    self.strided2Fits(m.a, g.a_offset, g.M, g.K, g.a_row_stride, g.a_col_stride) and
                    self.strided2Fits(m.b, g.b_offset, g.K, g.N, g.b_row_stride, g.b_col_stride) and
                    self.strided2Fits(m.dst, g.dst_offset, g.M, g.N, g.dst_row_stride, 1);
            },
            .qmatmul => |q| self.hasBuffer(q.dst) and self.hasBuffer(q.input) and
                self.strided2Fits(q.input, q.input_offset, @intCast(q.M), @intCast(q.K), if (q.input_row_stride != 0) @intCast(q.input_row_stride) else @intCast(q.K), 1) and
                self.strided2Fits(q.dst, q.dst_offset, @intCast(q.M), @intCast(q.N), if (q.dst_row_stride != 0) @intCast(q.dst_row_stride) else @intCast(q.N), 1),
            .softmax => |s| self.hasBuffer(s.dst) and self.hasBuffer(s.src) and
                self.dense2Fits(s.src, s.src_offset, @intCast(s.rows), @intCast(s.cols)) and
                self.dense2Fits(s.dst, s.dst_offset, @intCast(s.rows), @intCast(s.cols)),
            .logsoftmax => |s| self.hasBuffer(s.dst) and self.hasBuffer(s.src) and
                self.dense2Fits(s.src, s.src_offset, @intCast(s.rows), @intCast(s.cols)) and
                self.dense2Fits(s.dst, s.dst_offset, @intCast(s.rows), @intCast(s.cols)),
            .layernorm => |l| self.hasBuffer(l.dst) and self.hasBuffer(l.src) and
                self.dense2Fits(l.src, l.src_offset, @intCast(l.rows), @intCast(l.cols)) and
                self.dense2Fits(l.dst, l.dst_offset, @intCast(l.rows), @intCast(l.cols)),
            .rmsnorm => |r| self.hasBuffer(r.dst) and self.hasBuffer(r.src) and
                self.dense2Fits(r.src, r.src_offset, @intCast(r.rows), @intCast(r.cols)) and
                self.dense2Fits(r.dst, r.dst_offset, @intCast(r.rows), @intCast(r.cols)),
            .reduce => |r| self.hasBuffer(r.dst) and self.hasBuffer(r.src) and
                self.rangeFits(r.dst, r.dst_offset, @intCast(r.n_out)) and
                self.rangeFits(r.src, r.src_offset, std.math.mul(usize, r.n_out, r.reduce_size) catch return false),
            .conv2d => |c| self.hasBuffer(c.dst) and self.hasBuffer(c.src) and self.hasBuffer(c.weight) and
                (c.bias == std.math.maxInt(u16) or self.hasBuffer(c.bias)) and
                self.rangeFits(c.src, c.src_offset, std.math.mul(usize, std.math.mul(usize, c.in_w, c.in_h) catch return false, std.math.mul(usize, c.in_channels, c.batch) catch return false) catch return false) and
                self.rangeFits(c.weight, c.weight_offset, std.math.mul(usize, std.math.mul(usize, c.kernel_w, c.kernel_h) catch return false, std.math.mul(usize, c.in_channels, c.out_channels) catch return false) catch return false) and
                (c.bias == std.math.maxInt(u16) or self.rangeFits(c.bias, c.bias_offset, c.out_channels)) and
                self.rangeFits(c.dst, c.dst_offset, std.math.mul(usize, std.math.mul(usize, c.out_w, c.out_h) catch return false, std.math.mul(usize, c.out_channels, c.batch) catch return false) catch return false),
            .max_pool2d => |mp| self.hasBuffer(mp.dst) and self.hasBuffer(mp.src) and
                self.rangeFits(mp.src, mp.src_offset, std.math.mul(usize, std.math.mul(usize, mp.src_w, mp.src_h) catch return false, std.math.mul(usize, mp.channels, mp.batch) catch return false) catch return false) and
                self.rangeFits(mp.dst, mp.dst_offset, std.math.mul(usize, std.math.mul(usize, mp.out_w, mp.out_h) catch return false, std.math.mul(usize, mp.channels, mp.batch) catch return false) catch return false),
            .avg_pool2d => |mp| self.hasBuffer(mp.dst) and self.hasBuffer(mp.src) and
                self.rangeFits(mp.src, mp.src_offset, std.math.mul(usize, std.math.mul(usize, mp.src_w, mp.src_h) catch return false, std.math.mul(usize, mp.channels, mp.batch) catch return false) catch return false) and
                self.rangeFits(mp.dst, mp.dst_offset, std.math.mul(usize, std.math.mul(usize, mp.out_w, mp.out_h) catch return false, std.math.mul(usize, mp.channels, mp.batch) catch return false) catch return false),
            .repeat => |rp| self.hasBuffer(rp.dst) and self.hasBuffer(rp.src) and
                self.rangeFits(rp.dst, rp.dst_offset, @intCast(rp.n)) and
                self.rangeFits(rp.src, rp.src_offset, 1) and
                self.strided4Fits(rp.src, rp.src_offset, rp.src_ne, rp.src_strides) and
                self.strided4Fits(rp.dst, rp.dst_offset, rp.dst_ne, rp.dst_strides),
            .gather_rows => |g| self.hasBuffer(g.dst) and self.hasBuffer(g.src) and self.hasBuffer(g.indices) and
                self.rangeFits(g.indices, g.indices_offset, @intCast(g.count)) and
                self.strided2Fits(g.src, g.src_offset, @intCast(g.width), @intCast(g.src_rows), 1, if (g.src_row_stride != 0) @intCast(g.src_row_stride) else @intCast(g.width)) and
                self.strided2Fits(g.dst, g.dst_offset, @intCast(g.width), @intCast(g.count), 1, if (g.dst_row_stride != 0) @intCast(g.dst_row_stride) else @intCast(g.width)),
            .slice_assign => |sa| self.hasBuffer(sa.dst) and self.hasBuffer(sa.src) and
                self.strided2Fits(sa.src, sa.src_offset, @intCast(sa.rows), @intCast(sa.cols), @intCast(sa.src_row_stride), @intCast(sa.src_col_stride)) and
                self.strided2Fits(sa.dst, sa.dst_offset, @intCast(sa.rows), @intCast(sa.cols), @intCast(sa.dst_row_stride), @intCast(sa.dst_col_stride)),
            .rope => |rr| self.hasBuffer(rr.dst) and self.hasBuffer(rr.src) and self.hasBuffer(rr.cos_sin) and
                self.strided2Fits(rr.src, rr.src_off, @intCast(rr.half_d * 2), @intCast(rr.seq_len), @intCast(rr.src_rs), @intCast(rr.src_cs)) and
                self.strided2Fits(rr.cos_sin, rr.cs_off, @intCast(rr.half_d * 2), @intCast(rr.seq_len), 1, @intCast(rr.cs_cs)) and
                self.strided2Fits(rr.dst, rr.dst_off, @intCast(rr.half_d * 2), @intCast(rr.seq_len), 1, @intCast(rr.half_d * 2)),
            .attention => |att| self.hasBuffer(att.dst) and self.hasBuffer(att.q) and
                self.hasBuffer(att.k) and self.hasBuffer(att.v) and self.hasBuffer(att.mask) and
                self.strided2Fits(att.q, att.q_off, @intCast(att.d_head), @intCast(att.seq_q), @intCast(att.q_rs), @intCast(att.q_cs)) and
                self.strided2Fits(att.k, att.k_off, @intCast(att.d_head), @intCast(att.seq_kv), @intCast(att.k_rs), @intCast(att.k_cs)) and
                self.strided2Fits(att.v, att.v_off, @intCast(att.d_head), @intCast(att.seq_kv), @intCast(att.v_rs), @intCast(att.v_cs)) and
                (!att.has_mask or self.strided2Fits(att.mask, att.mask_off, @intCast(att.seq_kv), @intCast(att.seq_q), @intCast(att.mask_rs), @intCast(att.mask_cs))) and
                self.strided2Fits(att.dst, att.dst_off, @intCast(att.d_head), @intCast(att.seq_q), @intCast(att.dst_rs), @intCast(att.dst_cs)),
            .fused_elementwise => |fe| {
                if (!self.hasBuffer(fe.dst) or !self.hasBuffer(fe.src)) return false;
                if (!self.rangeFits(fe.dst, fe.dst_offset, @intCast(fe.n)) or !self.rangeFits(fe.src, fe.src_offset, @intCast(fe.n))) return false;
                for (fe.steps) |step| {
                    if (step.op.isBinary() and (!self.hasBuffer(step.secondary_buf) or !self.rangeFits(step.secondary_buf, step.secondary_offset, @intCast(fe.n)))) return false;
                }
                return true;
            },
        };
    }
};

pub const BufferBounds = struct {
    fn rangeFits(buffer_sizes: []const usize, idx: u16, offset: usize, len: usize) bool {
        const buf_idx: usize = idx;
        if (buf_idx >= buffer_sizes.len) return false;
        const size = buffer_sizes[buf_idx];
        return offset <= size and len <= size - offset;
    }

    fn dense2Fits(buffer_sizes: []const usize, idx: u16, offset: usize, rows: usize, cols: usize) bool {
        const len = std.math.mul(usize, rows, cols) catch return false;
        return rangeFits(buffer_sizes, idx, offset, len);
    }

    pub fn strided2Fits(buffer_sizes: []const usize, idx: u16, offset: usize, rows: usize, cols: usize, row_stride: usize, col_stride: usize) bool {
        if (rows == 0 or cols == 0) return rangeFits(buffer_sizes, idx, offset, 0);
        const row_span = std.math.mul(usize, rows - 1, row_stride) catch return false;
        const col_span = std.math.mul(usize, cols - 1, col_stride) catch return false;
        const last = std.math.add(usize, offset, row_span) catch return false;
        const end = std.math.add(usize, (std.math.add(usize, last, col_span) catch return false), 1) catch return false;
        return rangeFits(buffer_sizes, idx, 0, end);
    }

    fn strided4Fits(buffer_sizes: []const usize, idx: u16, offset: usize, ne: [4]u32, strides: [4]u32) bool {
        var last = offset;
        for (ne, strides) |dim, stride| {
            if (dim == 0) return rangeFits(buffer_sizes, idx, offset, 0);
            const span = std.math.mul(usize, @as(usize, dim - 1), @as(usize, stride)) catch return false;
            last = std.math.add(usize, last, span) catch return false;
        }
        return rangeFits(buffer_sizes, idx, 0, (std.math.add(usize, last, 1) catch return false));
    }

    fn programIOValid(buffer_sizes: []const usize, io: ProgramIO) bool {
        const idx: usize = io.buf_idx;
        if (idx >= buffer_sizes.len) return false;
        const byte_size = std.math.mul(usize, buffer_sizes[idx], @sizeOf(f32)) catch return false;
        const offset: usize = io.offset;
        const size: usize = io.size;
        return offset <= byte_size and size <= byte_size - offset and io.resourceFits();
    }

    fn programIOListValid(buffer_sizes: []const usize, ios: []const ProgramIO) bool {
        for (ios) |io| if (!programIOValid(buffer_sizes, io)) return false;
        return true;
    }

    pub fn programIOListHostOnly(ios: []const ProgramIO) bool {
        for (ios) |io| if (!io.isHost()) return false;
        return true;
    }

    pub fn programIOListResourcesMatchDevice(ios: []const ProgramIO, device: Device) bool {
        for (ios) |io| {
            const resource = io.resource orelse continue;
            if (resource.placement != device) return false;
        }
        return true;
    }

    pub fn programIOListResourcesCanRead(ios: []const ProgramIO) bool {
        for (ios) |io| if (!io.resourceCanRead()) return false;
        return true;
    }

    pub fn programIOListResourcesReadable(ios: []const ProgramIO, device: Device) bool {
        for (ios) |io| {
            const resource = io.resource orelse continue;
            if (resource.placement != device or !io.resourceCanRead()) return false;
        }
        return true;
    }

    pub fn programIOListResourcesWritable(ios: []const ProgramIO, device: Device) bool {
        for (ios) |io| {
            const resource = io.resource orelse continue;
            if (resource.placement != device or !io.resourceCanWrite()) return false;
        }
        return true;
    }

    pub fn programIOListsValid(buffer_sizes: []const usize, inputs: []const ProgramIO, outputs: []const ProgramIO) bool {
        return programIOListValid(buffer_sizes, inputs) and programIOListValid(buffer_sizes, outputs);
    }

    pub fn programIOListsHostOnly(inputs: []const ProgramIO, outputs: []const ProgramIO) bool {
        return programIOListHostOnly(inputs) and programIOListHostOnly(outputs);
    }

    pub fn programIOListsResourcesMatchDevice(inputs: []const ProgramIO, outputs: []const ProgramIO, device: Device) bool {
        return programIOListResourcesMatchDevice(inputs, device) and programIOListResourcesMatchDevice(outputs, device);
    }

    pub fn programIOListsResourcesUsable(inputs: []const ProgramIO, outputs: []const ProgramIO, device: Device) bool {
        return programIOListResourcesReadable(inputs, device) and programIOListResourcesWritable(outputs, device);
    }
};

// ── Backend ────────────────────────────────────────────────────────

pub const ExecutionFamilyCounts = struct {
    tiny_linear: u64 = 0,
    matmul: u64 = 0,
    matmul_elementwise: u64 = 0,
    matmul_fused_elementwise: u64 = 0,
    matvec_elementwise: u64 = 0,
    matvec_fused_elementwise: u64 = 0,
    matvec_slice: u64 = 0,
    matvec_rope_slice: u64 = 0,
    qmatmul: u64 = 0,
    qmatmul_elementwise: u64 = 0,
    qmatvec_slice: u64 = 0,
    qmatvec_rope_slice: u64 = 0,
    qmatvec_elementwise: u64 = 0,
    qmatvec_fused_elementwise: u64 = 0,
    qlinear: u64 = 0,
    elementwise: u64 = 0,
    fused_elementwise: u64 = 0,
    repeat: u64 = 0,
    layernorm: u64 = 0,
    rmsnorm: u64 = 0,
    softmax: u64 = 0,
    logsoftmax: u64 = 0,
    reduce: u64 = 0,
    rope: u64 = 0,
    slice_assign: u64 = 0,
    attention: u64 = 0,

    pub fn projectionCount(self: ExecutionFamilyCounts) u64 {
        return self.tiny_linear +
            self.matmul +
            self.matmul_elementwise +
            self.matmul_fused_elementwise +
            self.matvec_elementwise +
            self.matvec_fused_elementwise +
            self.matvec_slice +
            self.matvec_rope_slice +
            self.qmatmul +
            self.qmatmul_elementwise +
            self.qmatvec_slice +
            self.qmatvec_rope_slice +
            self.qmatvec_elementwise +
            self.qmatvec_fused_elementwise +
            self.qlinear;
    }

    pub fn quantizedProjectionCount(self: ExecutionFamilyCounts) u64 {
        return self.qmatmul +
            self.qmatmul_elementwise +
            self.qmatvec_slice +
            self.qmatvec_rope_slice +
            self.qmatvec_elementwise +
            self.qmatvec_fused_elementwise +
            self.qlinear;
    }

    pub fn rowCount(self: ExecutionFamilyCounts) u64 {
        return self.layernorm + self.rmsnorm + self.softmax + self.logsoftmax + self.reduce;
    }

    pub fn movementCount(self: ExecutionFamilyCounts) u64 {
        return self.repeat + self.slice_assign;
    }
};

pub const ExecutionPlanInspection = struct {
    supported: bool = false,
    total_op_count: u64 = 0,
    covered_op_count: u64 = 0,
    dispatch_count: u64 = 0,
    first_unsupported_op: ?u64 = null,
    max_dispatch_ops_per_dispatch: u64 = 0,
    family_counts: ExecutionFamilyCounts = .{},

    pub fn fullCoverage(self: ExecutionPlanInspection) bool {
        return self.supported and
            self.total_op_count > 0 and
            self.covered_op_count == self.total_op_count and
            self.first_unsupported_op == null;
    }
};

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    name_str: []const u8,
    device_type: Device,
    capabilities: Capabilities,

    pub const CompiledHandle = *anyopaque;
    pub const RuntimeHandle = *anyopaque;

    pub const VTable = struct {
        /// Override dense matmul during graph execution. Returns true if handled.
        dense_matmul_f32: *const fn (ctx: *anyopaque, spec: DenseMatMulSpecF32) bool,
        /// Optional backend-specific support check for compilers that only
        /// implement a subset of the coarse capability shape.
        supports_program: ?*const fn (ctx: *anyopaque, program: DeviceProgram) bool = null,
        /// Compile a DeviceProgram into backend-optimized execution.
        /// Backends must own any program state they need after this returns.
        compile_program: *const fn (ctx: *anyopaque, program: DeviceProgram) ?CompiledHandle,
        /// Optional backend-specific execution-plan evidence for a compiled Program.
        inspect_execution_plan: ?*const fn (ctx: *anyopaque, handle: CompiledHandle) ExecutionPlanInspection = null,
        /// Bind mutable runtime/session state for a compiled program.
        bind_program: *const fn (ctx: *anyopaque, handle: CompiledHandle) ?RuntimeHandle,
        /// Configure the complete Session I/O table for a bound runtime.
        /// Resource-capable backends use this to build bind groups/resource
        /// tables outside the hot step path. Host-only backends may no-op.
        configure_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, persistent_inputs: []const ProgramIO, step_inputs: []const ProgramIO, step_outputs: []const ProgramIO) bool,
        /// Patch a dynamic token window into bound runtime state before execution.
        patch_runtime_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, window: RuntimeWindow) RuntimePatchStatus,
        /// Upload persistent Session bindings into bound runtime state.
        upload_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, inputs: []const ProgramIO) void,
        /// Upload bound quantized weights into bound runtime state.
        upload_qweights: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, qweights: []const QuantizedWeightUpload) bool,
        /// Execute a compiled program against bound runtime state.
        execute_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void,
        /// Execute against the Session I/O table captured by configure_bindings.
        execute_configured_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, download_outputs: bool) void,
        /// Release bound runtime/session resources.
        free_bindings: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle) void,
        /// Patch a dynamic token window into a compiled program before execution.
        patch_runtime_window: *const fn (ctx: *anyopaque, handle: CompiledHandle, window: RuntimeWindow) RuntimePatchStatus,
        /// Upload persistent Session bindings without executing the program.
        upload_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, inputs: []const ProgramIO) void,
        /// Execute a compiled program: upload inputs, dispatch, download outputs.
        execute_program: *const fn (ctx: *anyopaque, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void,
        /// Release compiled program resources.
        free_program: *const fn (ctx: *anyopaque, handle: CompiledHandle) void,
        /// Reset accumulated runtime profile while preserving cold shape evidence.
        reset_runtime_profile: *const fn (ctx: *anyopaque, handle: CompiledHandle) void,
        /// Add a snapshot of accumulated runtime profile to `dest`.
        add_runtime_profile_to: *const fn (ctx: *anyopaque, handle: CompiledHandle, dest: *@import("profile.zig").RuntimeProfile) void,
        /// Reset accumulated runtime profile for a bound runtime/session.
        reset_runtime_bindings_profile: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle) void,
        /// Add a bound runtime/session profile snapshot to `dest`.
        add_runtime_bindings_profile_to: *const fn (ctx: *anyopaque, handle: CompiledHandle, runtime: RuntimeHandle, dest: *@import("profile.zig").RuntimeProfile) void,
    };

    pub fn compileProgram(self: Backend, program: DeviceProgram) ?CompiledHandle {
        if (!self.supportsProgram(program)) return null;
        return self.vtable.compile_program(self.ctx, program);
    }

    pub fn bindProgram(self: Backend, handle: CompiledHandle) ?RuntimeHandle {
        return self.vtable.bind_program(self.ctx, handle);
    }

    pub fn inspectExecutionPlan(self: Backend, handle: CompiledHandle) ExecutionPlanInspection {
        if (self.vtable.inspect_execution_plan) |inspect| return inspect(self.ctx, handle);
        return .{};
    }

    pub fn configureBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, persistent_inputs: []const ProgramIO, step_inputs: []const ProgramIO, step_outputs: []const ProgramIO) bool {
        if (!self.supportsProgramIO(persistent_inputs) or !self.supportsProgramIOLists(step_inputs, step_outputs)) return false;
        return self.vtable.configure_bindings(self.ctx, handle, runtime, persistent_inputs, step_inputs, step_outputs);
    }

    pub fn supportsProgram(self: Backend, program: DeviceProgram) bool {
        if (!program.isStructurallyValid(self.capabilities)) return false;
        if (!BufferBounds.programIOListResourcesReadable(program.initial_uploads, self.device_type)) return false;
        if (self.vtable.supports_program) |supports| return supports(self.ctx, program);
        return program.opsSupportedBy(self.capabilities);
    }

    pub fn patchRuntimeBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, window: RuntimeWindow) RuntimePatchStatus {
        return self.vtable.patch_runtime_bindings(self.ctx, handle, runtime, window);
    }

    pub fn uploadBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, inputs: []const ProgramIO) void {
        self.vtable.upload_bindings(self.ctx, handle, runtime, inputs);
    }

    pub fn uploadQWeights(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, qweights: []const QuantizedWeightUpload) bool {
        return self.vtable.upload_qweights(self.ctx, handle, runtime, qweights);
    }

    pub fn supportsProgramIO(self: Backend, ios: []const ProgramIO) bool {
        if (!self.capabilities.external_resources) return BufferBounds.programIOListHostOnly(ios);
        return BufferBounds.programIOListResourcesReadable(ios, self.device_type);
    }

    pub fn supportsProgramIOLists(self: Backend, inputs: []const ProgramIO, outputs: []const ProgramIO) bool {
        if (!self.capabilities.external_resources) return BufferBounds.programIOListsHostOnly(inputs, outputs);
        return BufferBounds.programIOListsResourcesUsable(inputs, outputs, self.device_type);
    }

    pub fn executeBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void {
        self.vtable.execute_bindings(self.ctx, handle, runtime, inputs, outputs);
    }

    pub fn executeConfiguredBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, download_outputs: bool) void {
        self.vtable.execute_configured_bindings(self.ctx, handle, runtime, download_outputs);
    }

    pub fn freeBindings(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle) void {
        self.vtable.free_bindings(self.ctx, handle, runtime);
    }

    pub fn patchRuntimeWindow(self: Backend, handle: CompiledHandle, window: RuntimeWindow) RuntimePatchStatus {
        return self.vtable.patch_runtime_window(self.ctx, handle, window);
    }

    pub fn uploadProgram(self: Backend, handle: CompiledHandle, inputs: []const ProgramIO) void {
        self.vtable.upload_program(self.ctx, handle, inputs);
    }

    pub fn executeProgram(self: Backend, handle: CompiledHandle, inputs: []const ProgramIO, outputs: []const ProgramIO) void {
        self.vtable.execute_program(self.ctx, handle, inputs, outputs);
    }

    pub fn freeProgram(self: Backend, handle: CompiledHandle) void {
        self.vtable.free_program(self.ctx, handle);
    }

    pub fn resetRuntimeProfile(self: Backend, handle: CompiledHandle) void {
        self.vtable.reset_runtime_profile(self.ctx, handle);
    }

    pub fn addRuntimeProfileTo(self: Backend, handle: CompiledHandle, dest: *@import("profile.zig").RuntimeProfile) void {
        self.vtable.add_runtime_profile_to(self.ctx, handle, dest);
    }

    pub fn resetRuntimeBindingsProfile(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle) void {
        self.vtable.reset_runtime_bindings_profile(self.ctx, handle, runtime);
    }

    pub fn addRuntimeBindingsProfileTo(self: Backend, handle: CompiledHandle, runtime: RuntimeHandle, dest: *@import("profile.zig").RuntimeProfile) void {
        self.vtable.add_runtime_bindings_profile_to(self.ctx, handle, runtime, dest);
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

test "runtime window carries only position and length" {
    const window = try RuntimeWindow.init(3, 2);
    try std.testing.expectEqual(@as(u32, 3), window.position);
    try std.testing.expectEqual(@as(u32, 2), window.len);
    try std.testing.expectEqual(@as(?u32, 5), window.attentionSeqKv());
    try std.testing.expectError(error.InvalidRuntimeWindow, RuntimeWindow.init(std.math.maxInt(usize), 1));
    try std.testing.expectError(error.InvalidRuntimeWindow, RuntimeWindow.init(@as(usize, std.math.maxInt(u32)) + 1, 0));
    try std.testing.expectEqual(@as(?u32, null), (RuntimeWindow{ .position = std.math.maxInt(u32), .len = 1 }).attentionSeqKv());
}

test "runtime patch shape wildcard requires stencil evidence for dynamic holes" {
    const expected_dynamic = RuntimePatchShape.expectedDynamic(1, 1);
    try std.testing.expect(expected_dynamic.matches(RuntimePatchShape.actual(1, 1, 99)));
    try std.testing.expect(!expected_dynamic.matches(RuntimePatchShape.actual(1, 1, 0)));
    try std.testing.expect(!expected_dynamic.matches(RuntimePatchShape.actual(2, 0, 99)));

    const expected_static = RuntimePatchShape.expectedDynamic(0, 0);
    try std.testing.expect(expected_static.matches(RuntimePatchShape.actual(0, 0, 0)));
    try std.testing.expect(!expected_static.matches(RuntimePatchShape.actual(0, 0, 99)));
}

test "webgpu compile-only capabilities do not claim execution" {
    try std.testing.expect(Capabilities.webgpu.executes_programs);
    try std.testing.expect(Capabilities.webgpu_compile_only.compiled_programs);
    try std.testing.expect(!Capabilities.webgpu_compile_only.executes_programs);
    try std.testing.expect(!Capabilities.webgpu_compile_only.external_resources);
    try std.testing.expect(Capabilities.webgpu_resource_plan.compiled_programs);
    try std.testing.expect(!Capabilities.webgpu_resource_plan.executes_programs);
    try std.testing.expect(Capabilities.webgpu_resource_plan.external_resources);
    try std.testing.expect(Capabilities.webgpu_resource_plan.qmatmul);
    try std.testing.expect(Capabilities.webgpu_resource_plan.attention.supported);
}

test "program support validates capabilities and qweight descriptors" {
    const fused_steps = [_]FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }};
    const fused_ops = [_]DeviceOp{.{ .fused_elementwise = .{ .steps = &fused_steps, .n = 1, .dst = 1, .src = 0, .dst_offset = 0, .src_offset = 0 } }};
    const fused_program = DeviceProgram{ .ops = &fused_ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 1 }, .initial_uploads = &.{} };
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
    try std.testing.expect(valid_qprogram.isSupportedBy(Capabilities.metal));

    const unused_bad_qweights = [_]QuantizedWeightUpload{
        qweights[0],
        .{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = 0 },
    };
    const unused_bad_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &unused_bad_qweights };
    try std.testing.expect(!unused_bad_qprogram.isSupportedBy(Capabilities.metal));

    const huge_block_qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = std.math.maxInt(usize) }};
    const huge_block_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &huge_block_qweights };
    try std.testing.expect(!huge_block_qprogram.isSupportedBy(Capabilities.metal));

    const bad_qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 2, .block_size = 4 }};
    const bad_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &bad_qweights };
    try std.testing.expect(!bad_qprogram.isSupportedBy(Capabilities.metal));

    const huge_scales = scales[0..].ptr[0..std.math.maxInt(usize)];
    const huge_scales_qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = huge_scales, .rows = 2, .cols = 2, .block_size = 4 }};
    const huge_scales_qprogram = DeviceProgram{ .ops = &qops, .n_buffers = 2, .buffer_sizes = &.{ 2, 2 }, .initial_uploads = &.{}, .qweights = &huge_scales_qweights };
    try std.testing.expect(!huge_scales_qprogram.isSupportedBy(Capabilities.metal));

    const patch_store = DeviceOp{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 4,
    } };
    const patch_program = DeviceProgram{ .ops = &.{patch_store}, .n_buffers = 2, .buffer_sizes = &.{ 1, 8 }, .initial_uploads = &.{} };
    try std.testing.expect(patch_program.isSupportedBy(Capabilities.metal));

    var bad_patch_store = patch_store;
    bad_patch_store.slice_assign.patch_stride = 0;
    const static_patch_program = DeviceProgram{ .ops = &.{bad_patch_store}, .n_buffers = 2, .buffer_sizes = &.{ 1, 8 }, .initial_uploads = &.{} };
    try std.testing.expect(static_patch_program.isSupportedBy(Capabilities.metal));
}

test "program support validates initial upload ranges" {
    var data = [_]f32{ 1, 2 };
    const upload = ProgramIO{ .buf_idx = 0, .host_ptr = @ptrCast(&data), .size = data.len * @sizeOf(f32) };
    const program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{upload} };
    try std.testing.expect(program.isSupportedBy(Capabilities.metal));

    const bad_buf = ProgramIO{ .buf_idx = 1, .host_ptr = @ptrCast(&data), .size = @sizeOf(f32) };
    const bad_buf_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{bad_buf} };
    try std.testing.expect(!bad_buf_program.isSupportedBy(Capabilities.metal));

    const bad_range = ProgramIO{ .buf_idx = 0, .offset = @sizeOf(f32), .host_ptr = @ptrCast(&data), .size = data.len * @sizeOf(f32) };
    const bad_range_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{bad_range} };
    try std.testing.expect(!bad_range_program.isSupportedBy(Capabilities.metal));

    const overflow_program = DeviceProgram{
        .ops = &.{},
        .n_buffers = 1,
        .buffer_sizes = &.{std.math.maxInt(usize)},
        .initial_uploads = &.{ProgramIO{ .buf_idx = 0, .host_ptr = @ptrCast(&data), .size = @sizeOf(f32) }},
    };
    try std.testing.expect(!overflow_program.isSupportedBy(Capabilities.metal));

    const no_io_overflow_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{std.math.maxInt(usize)}, .initial_uploads = &.{} };
    try std.testing.expect(!no_io_overflow_program.isSupportedBy(Capabilities.metal));
}

test "program IO distinguishes host memory from external resources" {
    var data = [_]f32{ 1, 2 };
    const host_upload = ProgramIO.host(0, 0, @ptrCast(&data), data.len * @sizeOf(f32));
    try std.testing.expect(host_upload.isHost());
    try std.testing.expect(host_upload.hostSlice() != null);

    const resource = ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 42,
        .byte_len = data.len * @sizeOf(f32),
    };
    const resource_upload = ProgramIO.external(0, 0, resource, data.len * @sizeOf(f32));
    try std.testing.expect(!resource_upload.isHost());
    try std.testing.expect(resource_upload.hostSlice() == null);
    try std.testing.expect(resource_upload.resourceFits());
    try std.testing.expect(BufferBounds.programIOListResourcesMatchDevice(&.{resource_upload}, .webgpu));
    try std.testing.expect(!BufferBounds.programIOListResourcesMatchDevice(&.{resource_upload}, .metal));
    try std.testing.expect(BufferBounds.programIOListResourcesReadable(&.{resource_upload}, .webgpu));
    try std.testing.expect(BufferBounds.programIOListResourcesWritable(&.{resource_upload}, .webgpu));

    const resource_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{resource_upload} };
    try std.testing.expect(!resource_program.isSupportedBy(Capabilities.metal));
    var resource_caps = Capabilities.metal;
    resource_caps.external_resources = true;
    try std.testing.expect(resource_program.isSupportedBy(resource_caps));

    var short_resource = resource;
    short_resource.byte_len -= 1;
    const short_resource_upload = ProgramIO.external(0, 0, short_resource, data.len * @sizeOf(f32));
    const short_resource_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{short_resource_upload} };
    try std.testing.expect(!short_resource_program.isSupportedBy(resource_caps));

    var write_only_resource = resource;
    write_only_resource.access = .write_only;
    const write_only_upload = ProgramIO.external(0, 0, write_only_resource, data.len * @sizeOf(f32));
    const write_only_program = DeviceProgram{ .ops = &.{}, .n_buffers = 1, .buffer_sizes = &.{data.len}, .initial_uploads = &.{write_only_upload} };
    try std.testing.expect(!write_only_upload.resourceCanRead());
    try std.testing.expect(!write_only_program.isSupportedBy(resource_caps));
    try std.testing.expect(!BufferBounds.programIOListResourcesReadable(&.{write_only_upload}, .webgpu));

    var read_only_resource = resource;
    read_only_resource.access = .read_only;
    const read_only_output = ProgramIO.external(0, 0, read_only_resource, data.len * @sizeOf(f32));
    try std.testing.expect(read_only_output.resourceCanRead());
    try std.testing.expect(!read_only_output.resourceCanWrite());
    try std.testing.expect(!BufferBounds.programIOListResourcesWritable(&.{read_only_output}, .webgpu));
    try std.testing.expect(!BufferBounds.programIOListsResourcesUsable(&.{resource_upload}, &.{read_only_output}, .webgpu));
}

test "program support validates linear op ranges" {
    const add = DeviceOp{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } };
    const valid_add = DeviceProgram{ .ops = &.{add}, .n_buffers = 3, .buffer_sizes = &.{ 4, 4, 4 }, .initial_uploads = &.{} };
    try std.testing.expect(valid_add.isSupportedBy(Capabilities.metal));
    var no_elementwise = Capabilities.metal;
    no_elementwise.elementwise = false;
    try std.testing.expect(!valid_add.isSupportedBy(no_elementwise));

    var bad_add = add;
    bad_add.elementwise.src1_offset = 2;
    const bad_add_program = DeviceProgram{ .ops = &.{bad_add}, .n_buffers = 3, .buffer_sizes = &.{ 4, 4, 4 }, .initial_uploads = &.{} };
    try std.testing.expect(!bad_add_program.isSupportedBy(Capabilities.metal));

    const relu = DeviceOp{ .elementwise = .{ .op = .relu, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } };
    var relu_ignores_src1_tail = relu;
    relu_ignores_src1_tail.elementwise.src1_offset = 4;
    const relu_program = DeviceProgram{ .ops = &.{relu_ignores_src1_tail}, .n_buffers = 3, .buffer_sizes = &.{ 4, 4, 4 }, .initial_uploads = &.{} };
    try std.testing.expect(relu_program.isSupportedBy(Capabilities.metal));

    const reduce = DeviceOp{ .reduce = .{ .op = .sum, .dst = 1, .src = 0, .n_out = 2, .reduce_size = 3 } };
    const valid_reduce = DeviceProgram{ .ops = &.{reduce}, .n_buffers = 2, .buffer_sizes = &.{ 6, 2 }, .initial_uploads = &.{} };
    try std.testing.expect(valid_reduce.isSupportedBy(Capabilities.metal));
    var no_reduce = Capabilities.metal;
    no_reduce.reduce = false;
    try std.testing.expect(!valid_reduce.isSupportedBy(no_reduce));
    const bad_reduce = DeviceProgram{ .ops = &.{reduce}, .n_buffers = 2, .buffer_sizes = &.{ 5, 2 }, .initial_uploads = &.{} };
    try std.testing.expect(!bad_reduce.isSupportedBy(Capabilities.metal));

    const steps = [_]FusedEwStep{.{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 2 }};
    const fused = DeviceOp{ .fused_elementwise = .{ .steps = &steps, .n = 4, .dst = 2, .src = 0, .dst_offset = 0, .src_offset = 0 } };
    const bad_fused = DeviceProgram{ .ops = &.{fused}, .n_buffers = 3, .buffer_sizes = &.{ 4, 5, 4 }, .initial_uploads = &.{} };
    try std.testing.expect(!bad_fused.isSupportedBy(Capabilities.metal));
}

test "program support validates strided op ranges" {
    const mat = DeviceOp{ .matmul = .{ .dst = 2, .a = 0, .b = 1, .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 } } };
    try std.testing.expect((DeviceProgram{ .ops = &.{mat}, .n_buffers = 3, .buffer_sizes = &.{ 6, 6, 4 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
    try std.testing.expect(!(DeviceProgram{ .ops = &.{mat}, .n_buffers = 3, .buffer_sizes = &.{ 5, 6, 4 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));

    const qmat = DeviceOp{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = 2, .N = 3, .K = 3, .input_offset = 1, .input_row_stride = 4, .dst_offset = 1, .dst_row_stride = 4 } };
    const qdata = [_]i8{1} ** 9;
    const qscales = [_]f32{1} ** 3;
    const qweights = [_]QuantizedWeightUpload{.{ .data = &qdata, .scales = &qscales, .rows = 3, .cols = 3, .block_size = 4 }};
    try std.testing.expect((DeviceProgram{ .ops = &.{qmat}, .n_buffers = 2, .buffer_sizes = &.{ 9, 9 }, .initial_uploads = &.{}, .qweights = &qweights }).isSupportedBy(Capabilities.metal));
    try std.testing.expect(!(DeviceProgram{ .ops = &.{qmat}, .n_buffers = 2, .buffer_sizes = &.{ 7, 9 }, .initial_uploads = &.{}, .qweights = &qweights }).isSupportedBy(Capabilities.metal));

    const slice = DeviceOp{ .slice_assign = .{ .dst = 1, .src = 0, .rows = 2, .cols = 2, .dst_base_offset = 0, .dst_offset = 2, .dst_row_stride = 1, .dst_col_stride = 2, .src_offset = 1, .src_row_stride = 1, .src_col_stride = 2, .patch_stride = 2 } };
    try std.testing.expect((DeviceProgram{ .ops = &.{slice}, .n_buffers = 2, .buffer_sizes = &.{ 6, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
    try std.testing.expect(!(DeviceProgram{ .ops = &.{slice}, .n_buffers = 2, .buffer_sizes = &.{ 4, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));

    const rope_op = DeviceOp{ .rope = .{ .dst = 2, .src = 0, .cos_sin = 1, .half_d = 2, .seq_len = 2, .src_off = 0, .cs_off = 0, .dst_off = 0, .src_rs = 1, .src_cs = 4, .cs_cs = 4 } };
    try std.testing.expect((DeviceProgram{ .ops = &.{rope_op}, .n_buffers = 3, .buffer_sizes = &.{ 8, 8, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
    try std.testing.expect(!(DeviceProgram{ .ops = &.{rope_op}, .n_buffers = 3, .buffer_sizes = &.{ 7, 8, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
    var no_rope = Capabilities.metal;
    no_rope.rope = false;
    try std.testing.expect(!(DeviceProgram{ .ops = &.{rope_op}, .n_buffers = 3, .buffer_sizes = &.{ 8, 8, 8 }, .initial_uploads = &.{} }).isSupportedBy(no_rope));

    const att = DeviceOp{ .attention = .{ .dst = 4, .q = 0, .k = 1, .v = 2, .mask = 3, .has_mask = true, .d_head = 4, .seq_q = 2, .seq_kv = 3, .scale = 0.5, .q_off = 0, .k_off = 0, .v_off = 0, .mask_off = 0, .dst_off = 0, .q_rs = 1, .q_cs = 4, .k_rs = 1, .k_cs = 4, .v_rs = 1, .v_cs = 4, .mask_rs = 1, .mask_cs = 3, .dst_rs = 1, .dst_cs = 4 } };
    try std.testing.expect((DeviceProgram{ .ops = &.{att}, .n_buffers = 5, .buffer_sizes = &.{ 8, 12, 12, 6, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
    try std.testing.expect(!(DeviceProgram{ .ops = &.{att}, .n_buffers = 5, .buffer_sizes = &.{ 8, 12, 12, 5, 8 }, .initial_uploads = &.{} }).isSupportedBy(Capabilities.metal));
}

test "capability attention limits are explicit" {
    try std.testing.expect(Capabilities.reference_cpu.attention.supports(8192, 512));
    try std.testing.expect(!Capabilities.reference_cpu.attention.supports(1, 513));
    try std.testing.expect(Capabilities.metal.attention.supports(8192, 512));
    try std.testing.expect(!Capabilities.metal.attention.supports(1, 513));
}
