//! Shared f32 reference executor for backend DevicePrograms.
//!
//! This module owns the CPU semantics for backend_mod.DeviceOp. Backends that
//! have host-visible buffers can reuse it instead of carrying their own copy of
//! elementwise, norm, cache update, RoPE, matmul, qmatmul, and attention logic.

const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../backend.zig");
const forward = @import("../tensor/forward.zig");
const profile_mod = @import("../profile.zig");
const program_mod = @import("program.zig");
const quant = @import("../quant.zig");

pub const Buffer = struct {
    ptr: [*]f32,
    len: usize,
};

pub const QWeight = struct {
    data: []const i8,
    scales: []const f32,
    block_size: usize,
    t_data: []const i8 = &.{},
    t_scales: []const f32 = &.{},
};

pub fn prepareTransposedQWeight(alloc: std.mem.Allocator, qw: backend_mod.QuantizedWeightUpload) !QWeight {
    const K = qw.rows;
    const N = qw.cols;
    const bs = qw.block_size;
    if (bs == 0 or bs > std.math.maxInt(u32)) return error.InvalidQuantizedWeight;
    const n_data = std.math.mul(usize, N, K) catch return error.InvalidQuantizedWeight;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / bs) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return error.InvalidQuantizedWeight;
    const blocks_per_row = if (K == 0) 0 else ((K - 1) / bs) + 1;
    const n_scales = std.math.mul(usize, N, blocks_per_row) catch return error.InvalidQuantizedWeight;

    const t_data = try alloc.alloc(i8, n_data);
    errdefer alloc.free(t_data);
    const t_scales = try alloc.alloc(f32, n_scales);
    errdefer alloc.free(t_scales);

    for (0..N) |n| {
        for (0..blocks_per_row) |b| {
            const k_start = b * bs;
            const k_end = if (bs > K - k_start) K else k_start + bs;

            var max_abs: f32 = 0;
            for (k_start..k_end) |k| {
                const orig_flat = k * N + n;
                const orig_block = orig_flat / bs;
                const val = @as(f32, @floatFromInt(qw.data[orig_flat])) * qw.scales[orig_block];
                max_abs = @max(max_abs, @abs(val));
            }

            const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
            const inv_scale = if (max_abs > 0) 127.0 / max_abs else 0.0;
            t_scales[n * blocks_per_row + b] = scale;

            for (k_start..k_end) |k| {
                const orig_flat = k * N + n;
                const orig_block = orig_flat / bs;
                const val = @as(f32, @floatFromInt(qw.data[orig_flat])) * qw.scales[orig_block];
                t_data[n * K + k] = @intFromFloat(std.math.clamp(val * inv_scale, -127.0, 127.0));
            }
        }
    }

    return .{
        .data = qw.data,
        .scales = qw.scales,
        .block_size = qw.block_size,
        .t_data = t_data,
        .t_scales = t_scales,
    };
}

pub fn deinitTransposedQWeight(alloc: std.mem.Allocator, qw: QWeight) void {
    if (qw.t_data.len > 0) alloc.free(@constCast(qw.t_data));
    if (qw.t_scales.len > 0) alloc.free(@constCast(qw.t_scales));
}

pub const OwnedBufferTable = struct {
    alloc: std.mem.Allocator,
    buffers: []Buffer,

    pub fn init(alloc: std.mem.Allocator, sizes: []const usize) !OwnedBufferTable {
        const buffers = try alloc.alloc(Buffer, sizes.len);
        var n_buffers: usize = 0;
        errdefer {
            for (buffers[0..n_buffers]) |buf| alloc.free(buf.ptr[0..buf.len]);
            alloc.free(buffers);
        }

        for (buffers, sizes) |*buf, size| {
            const storage = try alloc.alloc(f32, @max(size, 1));
            @memset(storage, 0);
            buf.* = .{ .ptr = storage.ptr, .len = storage.len };
            n_buffers += 1;
        }

        return .{ .alloc = alloc, .buffers = buffers };
    }

    pub fn deinit(self: *OwnedBufferTable) void {
        for (self.buffers) |buf| self.alloc.free(buf.ptr[0..buf.len]);
        self.alloc.free(self.buffers);
    }

    pub fn clone(self: OwnedBufferTable, alloc: std.mem.Allocator) !OwnedBufferTable {
        const sizes = try alloc.alloc(usize, self.buffers.len);
        defer alloc.free(sizes);
        for (self.buffers, 0..) |buf, i| sizes[i] = buf.len;

        var cloned = try OwnedBufferTable.init(alloc, sizes);
        errdefer cloned.deinit();
        for (self.buffers, cloned.buffers) |src, dst| {
            @memcpy(dst.ptr[0..dst.len], src.ptr[0..src.len]);
        }
        return cloned;
    }

    pub fn upload(self: OwnedBufferTable, inputs: []const backend_mod.ProgramIO) void {
        uploadToBuffers(self.buffers, inputs);
    }

    pub fn download(self: OwnedBufferTable, outputs: []const backend_mod.ProgramIO) void {
        downloadFromBuffers(self.buffers, outputs);
    }
};

pub fn uploadToBuffers(buffers: []const Buffer, inputs: []const backend_mod.ProgramIO) void {
    for (inputs) |io| {
        uploadBindingToBuffers(buffers, io);
    }
}

pub fn downloadFromBuffers(buffers: []const Buffer, outputs: []const backend_mod.ProgramIO) void {
    for (outputs) |io| {
        downloadBindingFromBuffers(buffers, io);
    }
}

pub fn uploadBindingToBuffers(buffers: []const Buffer, io: backend_mod.ProgramIO) void {
    const idx: usize = io.buf_idx;
    if (idx >= buffers.len) {
        std.debug.panic("reference backend upload buffer index out of range: {} >= {}", .{ idx, buffers.len });
    }
    const bytes = bufferBytes(buffers[idx]);
    const offset: usize = io.offset;
    const size: usize = io.size;
    if (offset > bytes.len or size > bytes.len - offset) {
        std.debug.panic("reference backend upload range out of bounds: offset={} size={} buffer_size={}", .{ offset, size, bytes.len });
    }
    const host = io.hostSlice() orelse std.debug.panic("reference backend upload received external resource binding", .{});
    @memcpy(bytes[offset..][0..size], host);
}

pub fn downloadBindingFromBuffers(buffers: []const Buffer, io: backend_mod.ProgramIO) void {
    const idx: usize = io.buf_idx;
    if (idx >= buffers.len) {
        std.debug.panic("reference backend download buffer index out of range: {} >= {}", .{ idx, buffers.len });
    }
    const bytes = bufferConstBytes(buffers[idx]);
    const offset: usize = io.offset;
    const size: usize = io.size;
    if (offset > bytes.len or size > bytes.len - offset) {
        std.debug.panic("reference backend download range out of bounds: offset={} size={} buffer_size={}", .{ offset, size, bytes.len });
    }
    const host = io.hostSlice() orelse std.debug.panic("reference backend download received external resource binding", .{});
    @memcpy(host, bytes[offset..][0..size]);
}

const ExecuteFn = *const fn (Context, backend_mod.DeviceOp) void;

pub const ExecutionTape = struct {
    entries: []Entry = &.{},
    commands: []Command = &.{},

    const Entry = struct {
        op_index: u32,
        execute: ExecuteFn,
    };

    const Command = struct {
        kind: program_mod.ProgramCommandKind,
        entry_start: u32,
        entry_count: u32,
    };

    pub fn init(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp) !ExecutionTape {
        return initWithCommandPolicy(alloc, ops, program_mod.CommandStreamPolicy.default());
    }

    pub fn initWithCommandPolicy(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp, policy: program_mod.CommandStreamPolicy) !ExecutionTape {
        var kernel_plan = try program_mod.Kernelizer.init(policy).kernelize(alloc, ops);
        defer kernel_plan.deinit(alloc);
        return initFromCommands(alloc, ops, kernel_plan.commands);
    }

    pub fn initFromCommands(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp, program_commands: []const program_mod.ProgramCommand) !ExecutionTape {
        var n_entries: usize = 0;
        for (program_commands) |command| {
            var iter = command.coveredIndexIterator();
            while (iter.next()) |_| n_entries += 1;
        }

        const entries = try alloc.alloc(Entry, n_entries);
        errdefer alloc.free(entries);
        const commands = try alloc.alloc(Command, program_commands.len);
        errdefer alloc.free(commands);

        var entry_index: usize = 0;
        for (program_commands, commands) |program_command, *command| {
            const entry_start = entry_index;
            var iter = program_command.coveredIndexIterator();
            while (iter.next()) |op_index| {
                if (op_index >= ops.len) return error.UnsupportedDeviceOp;
                entries[entry_index] = .{
                    .op_index = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp,
                    .execute = executeFnForOp(ops[op_index]),
                };
                entry_index += 1;
            }
            command.* = .{
                .kind = program_command.kind,
                .entry_start = std.math.cast(u32, entry_start) orelse return error.UnsupportedDeviceOp,
                .entry_count = std.math.cast(u32, entry_index - entry_start) orelse return error.UnsupportedDeviceOp,
            };
        }
        std.debug.assert(entry_index == entries.len);
        return .{ .entries = entries, .commands = commands };
    }

    pub fn deinit(self: *ExecutionTape, alloc: std.mem.Allocator) void {
        if (self.entries.len > 0) alloc.free(self.entries);
        if (self.commands.len > 0) alloc.free(self.commands);
        self.* = .{};
    }

    pub fn len(self: ExecutionTape) usize {
        return self.entries.len;
    }

    pub fn commandLen(self: ExecutionTape) usize {
        return self.commands.len;
    }

    pub fn execute(self: ExecutionTape, buffers: []const Buffer, qweights: []const QWeight, ops: []const backend_mod.DeviceOp) void {
        self.executeProfiled(buffers, qweights, ops, null);
    }

    pub fn executeProfiled(self: ExecutionTape, buffers: []const Buffer, qweights: []const QWeight, ops: []const backend_mod.DeviceOp, runtime_profile: ?*profile_mod.RuntimeProfile) void {
        const ctx = Context{ .buffers = buffers, .qweights = qweights };
        for (self.commands) |command| {
            if (runtime_profile) |profile| profile.recordProgramCommandAttempt(command.kind);
            const start: usize = command.entry_start;
            const count: usize = command.entry_count;
            const end = start + count;
            if (start > self.entries.len or end > self.entries.len) {
                if (runtime_profile) |profile| profile.recordProgramCommandFailed(command.kind);
                std.debug.panic("reference execution tape command range out of bounds: start={} count={} entries={}", .{ start, count, self.entries.len });
            }
            if (command.kind == .dense_projection_chain and count == 2) {
                const matmul_entry = self.entries[start];
                const sidecar_entry = self.entries[start + 1];
                const matmul_index: usize = matmul_entry.op_index;
                const sidecar_index: usize = sidecar_entry.op_index;
                if (matmul_index >= ops.len or sidecar_index >= ops.len) {
                    std.debug.panic("reference execution tape dense projection index out of range", .{});
                }
                if (ctx.denseProjectionChain(ops[matmul_index], ops[sidecar_index])) {
                    if (runtime_profile) |profile| {
                        profile.recordProgramCommand(command.kind);
                        profile.recordProgramCommandDispatch(command.kind);
                    }
                    continue;
                }
            }
            if (command.kind == .dense_projection_chain and count == 3) {
                const matmul_entry = self.entries[start];
                const repeat_entry = self.entries[start + 1];
                const sidecar_entry = self.entries[start + 2];
                const matmul_index: usize = matmul_entry.op_index;
                const repeat_index: usize = repeat_entry.op_index;
                const sidecar_index: usize = sidecar_entry.op_index;
                if (matmul_index >= ops.len or repeat_index >= ops.len or sidecar_index >= ops.len) {
                    std.debug.panic("reference execution tape dense projection bias index out of range", .{});
                }
                if (ctx.denseProjectionBiasChain(ops[matmul_index], ops[repeat_index], ops[sidecar_index])) {
                    if (runtime_profile) |profile| {
                        profile.recordProgramCommand(command.kind);
                        profile.recordProgramCommandDispatch(command.kind);
                    }
                    continue;
                }
            }
            if (command.kind == .dense_projection_chain and count == 4) {
                const matmul_entry = self.entries[start];
                const repeat_entry = self.entries[start + 1];
                const bias_entry = self.entries[start + 2];
                const tail_entry = self.entries[start + 3];
                const matmul_index: usize = matmul_entry.op_index;
                const repeat_index: usize = repeat_entry.op_index;
                const bias_index: usize = bias_entry.op_index;
                const tail_index: usize = tail_entry.op_index;
                if (matmul_index >= ops.len or repeat_index >= ops.len or bias_index >= ops.len or tail_index >= ops.len) {
                    std.debug.panic("reference execution tape dense projection bias activation index out of range", .{});
                }
                const executed = ctx.denseProjectionBiasActivationChain(ops[matmul_index], ops[repeat_index], ops[bias_index], ops[tail_index]) or
                    ctx.denseProjectionBiasLogSoftmaxChain(ops[matmul_index], ops[repeat_index], ops[bias_index], ops[tail_index]);
                if (executed) {
                    if (runtime_profile) |profile| {
                        profile.recordProgramCommand(command.kind);
                        profile.recordProgramCommandDispatch(command.kind);
                    }
                    continue;
                }
            }
            if (command.kind == .row_chain and count == 3) {
                const rms_entry = self.entries[start];
                const repeat_entry = self.entries[start + 1];
                const scale_entry = self.entries[start + 2];
                const rms_index: usize = rms_entry.op_index;
                const repeat_index: usize = repeat_entry.op_index;
                const scale_index: usize = scale_entry.op_index;
                if (rms_index >= ops.len or repeat_index >= ops.len or scale_index >= ops.len) {
                    std.debug.panic("reference execution tape rmsnorm scale index out of range", .{});
                }
                if (ctx.rmsnormScaleChain(ops[rms_index], ops[repeat_index], ops[scale_index])) {
                    if (runtime_profile) |profile| {
                        profile.recordProgramCommand(command.kind);
                        profile.recordProgramCommandDispatch(command.kind);
                    }
                    continue;
                }
            }
            if (command.kind == .row_chain and count == 4) {
                const rms_entry = self.entries[start];
                const repeat_entry = self.entries[start + 1];
                const scale_entry = self.entries[start + 2];
                const activation_entry = self.entries[start + 3];
                const rms_index: usize = rms_entry.op_index;
                const repeat_index: usize = repeat_entry.op_index;
                const scale_index: usize = scale_entry.op_index;
                const activation_index: usize = activation_entry.op_index;
                if (rms_index >= ops.len or repeat_index >= ops.len or scale_index >= ops.len or activation_index >= ops.len) {
                    std.debug.panic("reference execution tape rmsnorm scale activation index out of range", .{});
                }
                if (ctx.rmsnormScaleActivationChain(ops[rms_index], ops[repeat_index], ops[scale_index], ops[activation_index])) {
                    if (runtime_profile) |profile| {
                        profile.recordProgramCommand(command.kind);
                        profile.recordProgramCommandDispatch(command.kind);
                    }
                    continue;
                }
            }
            for (self.entries[start..end]) |entry| {
                const idx: usize = entry.op_index;
                if (idx >= ops.len) {
                    std.debug.panic("reference execution tape op index out of range: {} >= {}", .{ idx, ops.len });
                }
                entry.execute(ctx, ops[idx]);
            }
            if (runtime_profile) |profile| {
                profile.recordProgramCommand(command.kind);
                profile.recordProgramCommandDispatch(command.kind);
            }
        }
    }
};

pub fn executeProgram(buffers: []const Buffer, qweights: []const QWeight, ops: []const backend_mod.DeviceOp) void {
    for (ops) |op| executeOp(buffers, qweights, op);
}

pub fn executeOp(buffers: []const Buffer, qweights: []const QWeight, op: backend_mod.DeviceOp) void {
    const ctx = Context{ .buffers = buffers, .qweights = qweights };
    ctx.executeOp(op);
}

fn executeFnForOp(op: backend_mod.DeviceOp) ExecuteFn {
    return switch (op) {
        .matmul => executeMatmul,
        .qmatmul => executeQMatmul,
        .elementwise => executeElementwise,
        .softmax => executeSoftmax,
        .logsoftmax => executeLogSoftmax,
        .layernorm => executeLayerNorm,
        .rmsnorm => executeRmsNorm,
        .reduce => executeReduce,
        .conv2d => executeConv2d,
        .max_pool2d => executeMaxPool2d,
        .avg_pool2d => executeAvgPool2d,
        .repeat => executeRepeat,
        .gather_rows => executeGatherRows,
        .slice_assign => executeSliceAssign,
        .rope => executeRope,
        .attention => executeAttention,
        .fused_elementwise => executeFusedElementwise,
    };
}

fn executeMatmul(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.matmul(op.matmul);
}

fn executeQMatmul(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.qmatmul(op.qmatmul);
}

fn executeElementwise(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.elementwise(op.elementwise);
}

fn executeSoftmax(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.softmax(op.softmax);
}

fn executeLogSoftmax(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.logsoftmax(op.logsoftmax);
}

fn executeLayerNorm(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.layernorm(op.layernorm);
}

fn executeRmsNorm(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.rmsnorm(op.rmsnorm);
}

fn executeReduce(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.reduce(op.reduce);
}

fn executeConv2d(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.conv2d(op.conv2d);
}

fn executeMaxPool2d(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.maxPool2d(op.max_pool2d);
}

fn executeAvgPool2d(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.avgPool2d(op.avg_pool2d);
}

fn executeRepeat(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.repeat(op.repeat);
}

fn executeGatherRows(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.gatherRows(op.gather_rows);
}

fn executeSliceAssign(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.sliceAssign(op.slice_assign);
}

fn executeRope(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.rope(op.rope);
}

fn executeAttention(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.attention(op.attention);
}

fn executeFusedElementwise(ctx: Context, op: backend_mod.DeviceOp) void {
    ctx.fusedElementwise(op.fused_elementwise);
}

fn bufferBytes(buffer: Buffer) []u8 {
    const ptr: [*]u8 = @ptrCast(buffer.ptr);
    return ptr[0 .. buffer.len * @sizeOf(f32)];
}

fn bufferConstBytes(buffer: Buffer) []const u8 {
    const ptr: [*]const u8 = @ptrCast(buffer.ptr);
    return ptr[0 .. buffer.len * @sizeOf(f32)];
}

const Context = struct {
    buffers: []const Buffer,
    qweights: []const QWeight,

    fn bufF32(self: Context, idx: u16) [*]f32 {
        return self.buffers[@as(usize, idx)].ptr;
    }

    fn bufSlice(self: Context, idx: u16) []f32 {
        const b = self.buffers[@as(usize, idx)];
        return b.ptr[0..b.len];
    }

    fn executeOp(self: Context, op: backend_mod.DeviceOp) void {
        switch (op) {
            .matmul => |m| self.matmul(m),
            .qmatmul => |q| self.qmatmul(q),
            .elementwise => |e| self.elementwise(e),
            .softmax => |s| self.softmax(s),
            .logsoftmax => |s| self.logsoftmax(s),
            .layernorm => |l| self.layernorm(l),
            .rmsnorm => |r| self.rmsnorm(r),
            .reduce => |rd| self.reduce(rd),
            .conv2d => |c| self.conv2d(c),
            .max_pool2d => |mp| self.maxPool2d(mp),
            .avg_pool2d => |mp| self.avgPool2d(mp),
            .repeat => |rp| self.repeat(rp),
            .gather_rows => |g| self.gatherRows(g),
            .slice_assign => |sa| self.sliceAssign(sa),
            .rope => |rr| self.rope(rr),
            .attention => |att| self.attention(att),
            .fused_elementwise => |fe| self.fusedElementwise(fe),
        }
    }

    const V = 8;

    fn simdBinaryLoop(dst: [*]f32, src0: [*]const f32, src1: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32), @Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src0[i..][0..V].*;
            const b: VecT = src1[i..][0..V].*;
            dst[i..][0..V].* = op(a, b);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src0[i])), @as(VecT, @splat(src1[i])))[0];
    }

    fn simdUnaryLoop(dst: [*]f32, src: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src[i..][0..V].*;
            dst[i..][0..V].* = op(a);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src[i])))[0];
    }

    fn fusedUnaryStepCanSimd(op: backend_mod.Op) bool {
        return switch (op) {
            .neg, .abs, .sgn, .step, .relu, .sqr, .sqrt, .recip, .exp, .log, .gelu, .sigmoid, .silu, .tanh => true,
            else => false,
        };
    }

    fn fusedUnaryStepVec(op: backend_mod.Op, v: @Vector(V, f32)) @Vector(V, f32) {
        const VecT = @Vector(V, f32);
        const zero: VecT = @splat(0.0);
        const one: VecT = @splat(1.0);
        const two: VecT = @splat(2.0);
        return switch (op) {
            .neg => -v,
            .abs => @abs(v),
            .sgn => @select(f32, v > zero, one, @select(f32, v < zero, @as(VecT, @splat(-1.0)), zero)),
            .step => @select(f32, v > zero, one, zero),
            .relu => @max(v, zero),
            .sqr => v * v,
            .sqrt => @sqrt(v),
            .recip => one / v,
            .exp => @exp(v),
            .log => @log(v),
            .gelu => {
                return geluApproxVec(v);
            },
            .sigmoid => one / (one + @exp(-v)),
            .silu => v / (one + @exp(-v)),
            .tanh => {
                const e2 = @exp(two * v);
                return (e2 - one) / (e2 + one);
            },
            else => unreachable,
        };
    }

    fn fusedElementwiseCanSimd(fe: anytype) bool {
        if (fe.steps.len == 0) return false;
        for (fe.steps) |step| {
            if (!fusedUnaryStepCanSimd(step.op)) return false;
        }
        return true;
    }

    fn unsupportedElementwiseOp(op: backend_mod.Op) noreturn {
        std.debug.panic("reference backend reached unsupported elementwise op: {s}", .{@tagName(op)});
    }

    fn elementwise(self: Context, e: anytype) void {
        const dst = self.bufF32(e.dst) + @as(usize, e.dst_offset);
        const src0 = self.bufF32(e.src0) + @as(usize, e.src0_offset);
        const src1 = self.bufF32(e.src1) + @as(usize, e.src1_offset);
        const n: usize = e.n;
        switch (e.op) {
            .add => simdBinaryLoop(dst, src0, src1, n, struct {
                fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) {
                    return a + b;
                }
            }.f),
            .mul => simdBinaryLoop(dst, src0, src1, n, struct {
                fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) {
                    return a * b;
                }
            }.f),
            .neg => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return -a;
                }
            }.f),
            .abs => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @abs(a);
                }
            }.f),
            .sgn => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    const zero: @Vector(V, f32) = @splat(0.0);
                    const one: @Vector(V, f32) = @splat(1.0);
                    const neg_one: @Vector(V, f32) = @splat(-1.0);
                    return @select(f32, a > zero, one, @select(f32, a < zero, neg_one, zero));
                }
            }.f),
            .step => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    const zero: @Vector(V, f32) = @splat(0.0);
                    const one: @Vector(V, f32) = @splat(1.0);
                    return @select(f32, a > zero, one, zero);
                }
            }.f),
            .relu => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @max(a, @as(@Vector(V, f32), @splat(0.0)));
                }
            }.f),
            .sqr => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return a * a;
                }
            }.f),
            .sqrt => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @sqrt(a);
                }
            }.f),
            .recip => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @as(@Vector(V, f32), @splat(@as(f32, 1.0))) / a;
                }
            }.f),
            .exp => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @exp(a);
                }
            }.f),
            .log => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @log(a);
                }
            }.f),
            .gelu => {
                var i: usize = 0;
                while (i + V <= n) : (i += V) {
                    dst[i..][0..V].* = geluApproxVec(src0[i..][0..V].*);
                }
                while (i < n) : (i += 1) {
                    const a = src0[i];
                    const kk = 0.7978845608 * (a + 0.044715 * a * a * a);
                    dst[i] = 0.5 * a * (1.0 + std.math.tanh(kk));
                }
            },
            .sigmoid => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    const one: @Vector(V, f32) = @splat(1.0);
                    return one / (one + @exp(-a));
                }
            }.f),
            .silu => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    const one: @Vector(V, f32) = @splat(1.0);
                    return a * (one / (one + @exp(-a)));
                }
            }.f),
            .tanh => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    const one: @Vector(V, f32) = @splat(1.0);
                    const two: @Vector(V, f32) = @splat(2.0);
                    const e2 = @exp(two * a);
                    return (e2 - one) / (e2 + one);
                }
            }.f),
            else => unsupportedElementwiseOp(e.op),
        }
    }

    fn fusedElementwise(self: Context, fe: anytype) void {
        const dst = self.bufF32(fe.dst) + @as(usize, fe.dst_offset);
        const src = self.bufF32(fe.src) + @as(usize, fe.src_offset);
        const n: usize = fe.n;
        if (fusedElementwiseCanSimd(fe)) {
            const VecT = @Vector(V, f32);
            var i: usize = 0;
            while (i + V <= n) : (i += V) {
                var v: VecT = src[i..][0..V].*;
                for (fe.steps) |step| {
                    v = fusedUnaryStepVec(step.op, v);
                }
                dst[i..][0..V].* = v;
            }
            while (i < n) : (i += 1) {
                var v = src[i];
                for (fe.steps) |step| {
                    switch (step.op) {
                        .neg => v = -v,
                        .abs => v = @abs(v),
                        .sgn => v = if (v > 0) 1 else if (v < 0) -1 else 0,
                        .step => v = if (v > 0) 1 else 0,
                        .relu => v = @max(v, 0.0),
                        .sqr => v = v * v,
                        .sqrt => v = @sqrt(v),
                        .recip => v = 1.0 / v,
                        .exp => v = @exp(v),
                        .log => v = @log(v),
                        .gelu => {
                            const kk = 0.7978845608 * (v + 0.044715 * v * v * v);
                            v = 0.5 * v * (1.0 + std.math.tanh(kk));
                        },
                        .sigmoid => v = 1.0 / (1.0 + @exp(-v)),
                        .silu => v = v / (1.0 + @exp(-v)),
                        .tanh => v = std.math.tanh(v),
                        else => unreachable,
                    }
                }
                dst[i] = v;
            }
            return;
        }
        for (0..n) |i| {
            var v = src[i];
            for (fe.steps) |step| {
                switch (step.op) {
                    .neg => v = -v,
                    .abs => v = @abs(v),
                    .sgn => v = if (v > 0) 1 else if (v < 0) -1 else 0,
                    .step => v = if (v > 0) 1 else 0,
                    .relu => v = @max(v, 0.0),
                    .sqr => v = v * v,
                    .sqrt => v = @sqrt(v),
                    .recip => v = 1.0 / v,
                    .exp => v = @exp(v),
                    .log => v = @log(v),
                    .gelu => {
                        const kk = 0.7978845608 * (v + 0.044715 * v * v * v);
                        v = 0.5 * v * (1.0 + std.math.tanh(kk));
                    },
                    .sigmoid => v = 1.0 / (1.0 + @exp(-v)),
                    .silu => v = v / (1.0 + @exp(-v)),
                    .tanh => v = std.math.tanh(v),
                    .add => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] + v else v + s_ptr[i];
                    },
                    .mul => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] * v else v * s_ptr[i];
                    },
                    else => unsupportedElementwiseOp(step.op),
                }
            }
            dst[i] = v;
        }
    }

    fn softmax(self: Context, s: anytype) void {
        const src = self.bufF32(s.src);
        const dst = self.bufF32(s.dst);
        const cols: usize = s.cols;
        const inner: usize = s.inner;
        if (inner != 1) {
            for (0..@as(usize, s.rows)) |row| {
                const row_base: usize = @as(usize, s.src_offset) + row * cols * inner;
                const dst_base: usize = @as(usize, s.dst_offset) + row * cols * inner;
                for (0..inner) |lane| {
                    var max_v: f32 = -std.math.inf(f32);
                    var col: usize = 0;
                    while (col < cols) : (col += 1) {
                        max_v = @max(max_v, src[row_base + col * inner + lane]);
                    }
                    var sum: f32 = 0;
                    col = 0;
                    while (col < cols) : (col += 1) {
                        const value = @exp(src[row_base + col * inner + lane] - max_v);
                        dst[dst_base + col * inner + lane] = value;
                        sum += value;
                    }
                    const inv = if (sum > 0.0) 1.0 / sum else 0.0;
                    col = 0;
                    while (col < cols) : (col += 1) {
                        dst[dst_base + col * inner + lane] *= inv;
                    }
                }
            }
            return;
        }
        const VecT = @Vector(V, f32);
        for (0..@as(usize, s.rows)) |row| {
            const sb: usize = @as(usize, s.src_offset) + row * cols;
            const db: usize = @as(usize, s.dst_offset) + row * cols;
            const src_row = src[sb..][0..cols];
            const dst_row = dst[db..][0..cols];
            var max_v: VecT = @splat(-std.math.inf(f32));
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = src_row[i..][0..V].*;
                max_v = @max(max_v, v);
            }
            var m: f32 = @reduce(.Max, max_v);
            while (i < cols) : (i += 1) m = @max(m, src_row[i]);
            const m_v: VecT = @splat(m);
            var sum_v: VecT = @splat(0);
            i = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = @exp(src_row[i..][0..V].* - m_v);
                dst_row[i..][0..V].* = v;
                sum_v += v;
            }
            var sum: f32 = @reduce(.Add, sum_v);
            while (i < cols) : (i += 1) {
                const v = @exp(src_row[i] - m);
                dst_row[i] = v;
                sum += v;
            }
            const inv = if (sum > 0.0) 1.0 / sum else 0.0;
            const inv_v: VecT = @splat(inv);
            i = 0;
            while (i + V <= cols) : (i += V) {
                dst_row[i..][0..V].* = dst_row[i..][0..V].* * inv_v;
            }
            while (i < cols) : (i += 1) dst_row[i] *= inv;
        }
    }

    fn logsoftmax(self: Context, s: anytype) void {
        const src = self.bufF32(s.src);
        const dst = self.bufF32(s.dst);
        const cols: usize = s.cols;
        const inner: usize = s.inner;
        if (inner != 1) {
            for (0..@as(usize, s.rows)) |row| {
                const row_base: usize = @as(usize, s.src_offset) + row * cols * inner;
                const dst_base: usize = @as(usize, s.dst_offset) + row * cols * inner;
                for (0..inner) |lane| {
                    var max_v: f32 = -std.math.inf(f32);
                    var col: usize = 0;
                    while (col < cols) : (col += 1) {
                        max_v = @max(max_v, src[row_base + col * inner + lane]);
                    }
                    var sum: f32 = 0;
                    col = 0;
                    while (col < cols) : (col += 1) {
                        sum += @exp(src[row_base + col * inner + lane] - max_v);
                    }
                    const log_denom = max_v + @log(sum);
                    col = 0;
                    while (col < cols) : (col += 1) {
                        dst[dst_base + col * inner + lane] = src[row_base + col * inner + lane] - log_denom;
                    }
                }
            }
            return;
        }
        const VecT = @Vector(V, f32);
        for (0..@as(usize, s.rows)) |row| {
            const sb: usize = @as(usize, s.src_offset) + row * cols;
            const db: usize = @as(usize, s.dst_offset) + row * cols;
            const src_row = src[sb..][0..cols];
            const dst_row = dst[db..][0..cols];
            var max_v: VecT = @splat(-std.math.inf(f32));
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = src_row[i..][0..V].*;
                max_v = @max(max_v, v);
            }
            var m: f32 = @reduce(.Max, max_v);
            while (i < cols) : (i += 1) m = @max(m, src_row[i]);
            const m_v: VecT = @splat(m);
            var sum_v: VecT = @splat(0);
            i = 0;
            while (i + V <= cols) : (i += V) {
                sum_v += @exp(src_row[i..][0..V].* - m_v);
            }
            var sum: f32 = @reduce(.Add, sum_v);
            while (i < cols) : (i += 1) sum += @exp(src_row[i] - m);
            const log_denom = m + @log(sum);
            const log_denom_v: VecT = @splat(log_denom);
            i = 0;
            while (i + V <= cols) : (i += V) {
                dst_row[i..][0..V].* = src_row[i..][0..V].* - log_denom_v;
            }
            while (i < cols) : (i += 1) dst_row[i] = src_row[i] - log_denom;
        }
    }

    fn layernorm(self: Context, l: anytype) void {
        const src = self.bufF32(l.src);
        const dst = self.bufF32(l.dst);
        const cols: usize = l.cols;
        for (0..@as(usize, l.rows)) |row| {
            const base: usize = @as(usize, l.src_offset) + row * cols;
            const dbase: usize = @as(usize, l.dst_offset) + row * cols;
            var mu: f32 = 0;
            for (0..cols) |j| mu += src[base + j];
            mu /= @as(f32, @floatFromInt(cols));
            var v: f32 = 0;
            for (0..cols) |j| {
                const diff = src[base + j] - mu;
                v += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(v / @as(f32, @floatFromInt(cols)) + l.eps);
            for (0..cols) |j| dst[dbase + j] = (src[base + j] - mu) * inv_std;
        }
    }

    fn rmsnorm(self: Context, r: anytype) void {
        const src = self.bufF32(r.src);
        const dst = self.bufF32(r.dst);
        const cols: usize = r.cols;
        const VecT = @Vector(V, f32);
        for (0..@as(usize, r.rows)) |row| {
            const s = src + @as(usize, r.src_offset) + row * cols;
            const d = dst + @as(usize, r.dst_offset) + row * cols;
            var acc: VecT = @splat(0);
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                acc += v * v;
            }
            var ss: f32 = @reduce(.Add, acc);
            while (i < cols) : (i += 1) ss += s[i] * s[i];
            const inv_rms: VecT = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(cols)) + r.eps));
            i = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                d[i..][0..V].* = v * inv_rms;
            }
            const inv_s = inv_rms[0];
            while (i < cols) : (i += 1) d[i] = s[i] * inv_s;
        }
    }

    fn rmsnormScaleChain(self: Context, rms_op: backend_mod.DeviceOp, repeat_op: backend_mod.DeviceOp, scale_op: backend_mod.DeviceOp) bool {
        const r = switch (rms_op) {
            .rmsnorm => |r| r,
            else => return false,
        };
        const rp = switch (repeat_op) {
            .repeat => |rp| rp,
            else => return false,
        };
        const e = switch (scale_op) {
            .elementwise => |e| e,
            else => return false,
        };
        if (!program_mod.isRmsnormScaleChain(rms_op, repeat_op, scale_op)) return false;
        self.rmsnormScaleRows(r, rp, e.dst, e.dst_offset, null);
        return true;
    }

    fn rmsnormScaleActivationChain(
        self: Context,
        rms_op: backend_mod.DeviceOp,
        repeat_op: backend_mod.DeviceOp,
        scale_op: backend_mod.DeviceOp,
        activation_op: backend_mod.DeviceOp,
    ) bool {
        const r = switch (rms_op) {
            .rmsnorm => |r| r,
            else => return false,
        };
        const rp = switch (repeat_op) {
            .repeat => |rp| rp,
            else => return false,
        };
        const activation = switch (activation_op) {
            .elementwise => |e| e,
            else => return false,
        };
        if (!program_mod.isRmsnormScaleActivationChain(rms_op, repeat_op, scale_op, activation_op)) return false;
        self.rmsnormScaleRows(r, rp, activation.dst, activation.dst_offset, activation.op);
        return true;
    }

    fn rmsnormScaleRows(self: Context, r: anytype, rp: anytype, dst_buf: u16, dst_offset: u32, activation: ?backend_mod.Op) void {
        const src = self.bufF32(r.src);
        const dst = self.bufF32(dst_buf);
        const scale = self.bufF32(rp.src);
        const rows: usize = r.rows;
        const cols: usize = r.cols;
        const VecT = @Vector(V, f32);
        for (0..rows) |row| {
            const s = src + @as(usize, r.src_offset) + row * cols;
            const d = dst + @as(usize, dst_offset) + row * cols;
            const scale_row = scale[@as(usize, rp.src_offset)..][0..cols];
            var acc: VecT = @splat(0);
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                acc += v * v;
            }
            var ss: f32 = @reduce(.Add, acc);
            while (i < cols) : (i += 1) ss += s[i] * s[i];
            const inv_rms: VecT = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(cols)) + r.eps));
            i = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                const scale_v: VecT = scale_row[i..][0..V].*;
                const scaled = v * inv_rms * scale_v;
                d[i..][0..V].* = if (activation) |op| activationVec(op, scaled) else scaled;
            }
            const inv_s = inv_rms[0];
            while (i < cols) : (i += 1) {
                const scaled = s[i] * inv_s * scale_row[i];
                d[i] = if (activation) |op| activationScalar(op, scaled) else scaled;
            }
        }
    }

    fn activationVec(op: backend_mod.Op, x: @Vector(V, f32)) @Vector(V, f32) {
        const VecT = @Vector(V, f32);
        return switch (op) {
            .relu => @max(x, @as(VecT, @splat(0.0))),
            .gelu => blk: {
                break :blk geluApproxVec(x);
            },
            .silu => blk: {
                const one: VecT = @splat(1.0);
                break :blk x * (one / (one + fastExpApproxVec(-x)));
            },
            else => x,
        };
    }

    fn activationScalar(op: backend_mod.Op, x: f32) f32 {
        return switch (op) {
            .relu => @max(x, 0.0),
            .gelu => blk: {
                const k = 0.7978845608 * (x + 0.044715 * x * x * x);
                break :blk 0.5 * x * (1.0 + std.math.tanh(k));
            },
            .silu => x / (1.0 + @exp(-x)),
            else => x,
        };
    }

    fn reduce(self: Context, rd: anytype) void {
        const src = self.bufF32(rd.src);
        const dst = self.bufF32(rd.dst);
        const rs: usize = rd.reduce_size;
        for (0..@as(usize, rd.n_out)) |i| {
            const sb: usize = @as(usize, rd.src_offset) + i * rs;
            var val: f32 = switch (rd.op) {
                .sum => 0.0,
                .prod => 1.0,
                .max => -std.math.inf(f32),
                .min => std.math.inf(f32),
                .argmax => -std.math.inf(f32),
                .argmin => std.math.inf(f32),
                else => unreachable,
            };
            for (0..rs) |k| {
                const v = src[sb + k];
                switch (rd.op) {
                    .sum => val += v,
                    .prod => val *= v,
                    .max => val = @max(val, v),
                    .min => val = @min(val, v),
                    .argmax => if (v > val) {
                        val = v;
                        dst[@as(usize, rd.dst_offset) + i] = @floatFromInt(k);
                    },
                    .argmin => if (v < val) {
                        val = v;
                        dst[@as(usize, rd.dst_offset) + i] = @floatFromInt(k);
                    },
                    else => unreachable,
                }
            }
            if (rd.op == .sum or rd.op == .prod or rd.op == .max or rd.op == .min) dst[@as(usize, rd.dst_offset) + i] = val;
        }
    }

    fn maxPool2d(self: Context, mp: anytype) void {
        const src = self.bufF32(mp.src);
        const dst = self.bufF32(mp.dst);
        const out_w: usize = mp.out_w;
        const out_h: usize = mp.out_h;
        const channels: usize = mp.channels;
        const batch: usize = mp.batch;
        const src_w: usize = mp.src_w;
        const src_h: usize = mp.src_h;
        const src_base: usize = mp.src_offset;
        const dst_base: usize = mp.dst_offset;
        for (0..batch) |n| {
            const src_batch_base = src_base + n * src_w * src_h * channels;
            const dst_batch_base = dst_base + n * out_w * out_h * channels;
            for (0..channels) |c| {
                const src_channel_base = src_batch_base + c * src_w * src_h;
                const dst_channel_base = dst_batch_base + c * out_w * out_h;
                for (0..out_h) |y| {
                    const sy = y * 2;
                    const src_row0 = src_channel_base + sy * src_w;
                    const src_row1 = src_row0 + src_w;
                    const dst_row = dst_channel_base + y * out_w;
                    for (0..out_w) |x| {
                        const sx = x * 2;
                        const a = src[src_row0 + sx];
                        const b = src[src_row0 + sx + 1];
                        const c0 = src[src_row1 + sx];
                        const d = src[src_row1 + sx + 1];
                        dst[dst_row + x] = @max(@max(a, b), @max(c0, d));
                    }
                }
            }
        }
    }

    fn avgPool2d(self: Context, mp: anytype) void {
        const src = self.bufF32(mp.src);
        const dst = self.bufF32(mp.dst);
        const out_w: usize = mp.out_w;
        const out_h: usize = mp.out_h;
        const channels: usize = mp.channels;
        const batch: usize = mp.batch;
        const src_w: usize = mp.src_w;
        const src_h: usize = mp.src_h;
        const src_base: usize = mp.src_offset;
        const dst_base: usize = mp.dst_offset;
        for (0..batch) |n| {
            const src_batch_base = src_base + n * src_w * src_h * channels;
            const dst_batch_base = dst_base + n * out_w * out_h * channels;
            for (0..channels) |c| {
                const src_channel_base = src_batch_base + c * src_w * src_h;
                const dst_channel_base = dst_batch_base + c * out_w * out_h;
                for (0..out_h) |y| {
                    const sy = y * 2;
                    const src_row0 = src_channel_base + sy * src_w;
                    const src_row1 = src_row0 + src_w;
                    const dst_row = dst_channel_base + y * out_w;
                    for (0..out_w) |x| {
                        const sx = x * 2;
                        dst[dst_row + x] = (src[src_row0 + sx] + src[src_row0 + sx + 1] + src[src_row1 + sx] + src[src_row1 + sx + 1]) * 0.25;
                    }
                }
            }
        }
    }

    fn conv2d(self: Context, c: anytype) void {
        const src = self.bufF32(c.src);
        const weight = self.bufF32(c.weight);
        const dst = self.bufF32(c.dst);
        const has_bias = c.bias != std.math.maxInt(u16);
        const bias = if (has_bias) self.bufF32(c.bias) else undefined;
        const out_w: usize = c.out_w;
        const out_h: usize = c.out_h;
        const in_w: usize = c.in_w;
        const in_h: usize = c.in_h;
        const in_channels: usize = c.in_channels;
        const out_channels: usize = c.out_channels;
        const kernel_w: usize = c.kernel_w;
        const kernel_h: usize = c.kernel_h;
        const batch: usize = c.batch;
        const src_base: usize = c.src_offset;
        const weight_base: usize = c.weight_offset;
        const bias_base: usize = c.bias_offset;
        const dst_base: usize = c.dst_offset;
        const relu = c.relu;
        if (in_channels == 1 and kernel_w == 3 and kernel_h == 3) {
            for (0..batch) |n| {
                const src_batch_base = src_base + n * in_w * in_h;
                const dst_batch_base = dst_base + n * out_w * out_h * out_channels;
                for (0..out_channels) |oc| {
                    const b: f32 = if (has_bias) bias[bias_base + oc] else 0;
                    const weight_channel_base = weight_base + oc * 9;
                    const w0 = weight[weight_channel_base + 0];
                    const w1 = weight[weight_channel_base + 1];
                    const w2 = weight[weight_channel_base + 2];
                    const w3 = weight[weight_channel_base + 3];
                    const w4 = weight[weight_channel_base + 4];
                    const w5 = weight[weight_channel_base + 5];
                    const w6 = weight[weight_channel_base + 6];
                    const w7 = weight[weight_channel_base + 7];
                    const w8 = weight[weight_channel_base + 8];
                    const dst_channel_base = dst_batch_base + oc * out_w * out_h;
                    for (0..out_h) |oy| {
                        const r0 = src_batch_base + oy * in_w;
                        const r1 = r0 + in_w;
                        const r2 = r1 + in_w;
                        const drow = dst_channel_base + oy * out_w;
                        var ox: usize = 0;
                        while (ox + 1 < out_w) : (ox += 2) {
                            const r0x0 = src[r0 + ox];
                            const r0x1 = src[r0 + ox + 1];
                            const r0x2 = src[r0 + ox + 2];
                            const r0x3 = src[r0 + ox + 3];
                            const r1x0 = src[r1 + ox];
                            const r1x1 = src[r1 + ox + 1];
                            const r1x2 = src[r1 + ox + 2];
                            const r1x3 = src[r1 + ox + 3];
                            const r2x0 = src[r2 + ox];
                            const r2x1 = src[r2 + ox + 1];
                            const r2x2 = src[r2 + ox + 2];
                            const r2x3 = src[r2 + ox + 3];
                            var sum0 =
                                r0x0 * w0 + r0x1 * w1 + r0x2 * w2 +
                                r1x0 * w3 + r1x1 * w4 + r1x2 * w5 +
                                r2x0 * w6 + r2x1 * w7 + r2x2 * w8 + b;
                            var sum1 =
                                r0x1 * w0 + r0x2 * w1 + r0x3 * w2 +
                                r1x1 * w3 + r1x2 * w4 + r1x3 * w5 +
                                r2x1 * w6 + r2x2 * w7 + r2x3 * w8 + b;
                            if (relu) {
                                if (sum0 < 0) sum0 = 0;
                                if (sum1 < 0) sum1 = 0;
                            }
                            dst[drow + ox] = sum0;
                            dst[drow + ox + 1] = sum1;
                        }
                        if (ox < out_w) {
                            var sum =
                                src[r0 + ox] * w0 + src[r0 + ox + 1] * w1 + src[r0 + ox + 2] * w2 +
                                src[r1 + ox] * w3 + src[r1 + ox + 1] * w4 + src[r1 + ox + 2] * w5 +
                                src[r2 + ox] * w6 + src[r2 + ox + 1] * w7 + src[r2 + ox + 2] * w8 + b;
                            if (relu and sum < 0) sum = 0;
                            dst[drow + ox] = sum;
                        }
                    }
                }
            }
            return;
        }
        for (0..batch) |n| {
            const src_batch_base = src_base + n * in_w * in_h * in_channels;
            const dst_batch_base = dst_base + n * out_w * out_h * out_channels;
            for (0..out_channels) |oc| {
                const dst_channel_base = dst_batch_base + oc * out_w * out_h;
                for (0..out_h) |oy| {
                    for (0..out_w) |ox| {
                        var sum: f32 = if (has_bias) bias[bias_base + oc] else 0;
                        for (0..in_channels) |ic| {
                            const src_channel_base = src_batch_base + ic * in_w * in_h;
                            const weight_channel_base = weight_base + (((oc * in_channels + ic) * kernel_h) * kernel_w);
                            for (0..kernel_h) |ky| {
                                const src_row = src_channel_base + (oy + ky) * in_w;
                                const weight_row = weight_channel_base + ky * kernel_w;
                                for (0..kernel_w) |kx| {
                                    sum += src[src_row + ox + kx] * weight[weight_row + kx];
                                }
                            }
                        }
                        if (relu and sum < 0) sum = 0;
                        dst[dst_channel_base + oy * out_w + ox] = sum;
                    }
                }
            }
        }
    }

    fn repeat(self: Context, rp: anytype) void {
        const src = self.bufF32(rp.src);
        const dst = self.bufF32(rp.dst);
        const n: usize = rp.n;
        const d = dst + @as(usize, rp.dst_offset);
        const s = src + @as(usize, rp.src_offset);

        const src_n: usize = @as(usize, rp.src_ne[0]) * @as(usize, rp.src_ne[1]) *
            @as(usize, rp.src_ne[2]) * @as(usize, rp.src_ne[3]);

        if (src_n == 1) {
            @memset(d[0..n], s[0]);
            return;
        }

        const src_dense = rp.src_strides[0] == 1 and
            (rp.src_ne[1] <= 1 or rp.src_strides[1] == rp.src_ne[0]) and
            (rp.src_ne[2] <= 1 or rp.src_strides[2] == @as(u32, rp.src_ne[0]) * rp.src_ne[1]) and
            (rp.src_ne[3] <= 1 or rp.src_strides[3] == @as(u32, rp.src_ne[0]) * @as(u32, rp.src_ne[1]) * rp.src_ne[2]);
        const dst_dense = rp.dst_strides[0] == 1 and
            (rp.dst_ne[1] <= 1 or rp.dst_strides[1] == rp.dst_ne[0]) and
            (rp.dst_ne[2] <= 1 or rp.dst_strides[2] == @as(u32, rp.dst_ne[0]) * rp.dst_ne[1]) and
            (rp.dst_ne[3] <= 1 or rp.dst_strides[3] == @as(u32, rp.dst_ne[0]) * @as(u32, rp.dst_ne[1]) * rp.dst_ne[2]);
        if (src_dense and dst_dense) {
            var chunk: usize = 1;
            for (0..4) |dim| {
                if (rp.src_ne[dim] > 1) break;
                chunk *= rp.dst_ne[dim];
            }
            const flat_modulo_compatible = rp.src_ne[1] <= 1 and rp.src_ne[2] <= 1 and rp.src_ne[3] <= 1;
            if (chunk == 1 and flat_modulo_compatible) {
                for (d[0..n], 0..) |*out, i| out.* = s[i % src_n];
                return;
            }
            if (chunk != 1 and n % chunk == 0) {
                const groups = n / chunk;
                for (0..groups) |group| {
                    const base = group * chunk;
                    @memset(d[base..][0..chunk], s[group % src_n]);
                }
                return;
            }
        }

        for (0..n) |gid| {
            var idx = gid;
            var src_idx: usize = rp.src_offset;
            var dim: usize = 4;
            while (dim > 0) {
                dim -= 1;
                const coord = idx / @as(usize, rp.dst_strides[dim]);
                idx = idx % @as(usize, rp.dst_strides[dim]);
                src_idx += (coord % @as(usize, rp.src_ne[dim])) * @as(usize, rp.src_strides[dim]);
            }
            dst[@as(usize, rp.dst_offset) + gid] = src[src_idx];
        }
    }

    fn indexFromF32(v: f32) usize {
        std.debug.assert(v >= 0);
        const iv: usize = @intFromFloat(v);
        std.debug.assert(@as(f32, @floatFromInt(iv)) == v);
        return iv;
    }

    fn gatherRows(self: Context, g: anytype) void {
        const src = self.bufF32(g.src);
        const indices = self.bufF32(g.indices);
        const dst = self.bufF32(g.dst);
        const width: usize = g.width;
        const count: usize = g.count;
        const src_rows: usize = g.src_rows;
        const src_row_stride: usize = if (g.src_row_stride != 0) g.src_row_stride else width;
        const dst_row_stride: usize = if (g.dst_row_stride != 0) g.dst_row_stride else width;
        for (0..count) |out_row| {
            const src_row = indexFromF32(indices[@as(usize, g.indices_offset) + out_row]);
            std.debug.assert(src_row < src_rows);
            const src_off = @as(usize, g.src_offset) + src_row * src_row_stride;
            const dst_off = @as(usize, g.dst_offset) + out_row * dst_row_stride;
            @memcpy(dst[dst_off..][0..width], src[src_off..][0..width]);
        }
    }

    fn sliceAssign(self: Context, sa: anytype) void {
        const src = self.bufF32(sa.src);
        const dst = self.bufF32(sa.dst);
        const rows: usize = sa.rows;
        const cols: usize = sa.cols;
        const doff: usize = sa.dst_offset;
        const soff: usize = sa.src_offset;
        const drs: usize = sa.dst_row_stride;
        const dcs: usize = sa.dst_col_stride;
        const srs: usize = sa.src_row_stride;
        const scs: usize = sa.src_col_stride;
        if (drs == 1 and srs == 1 and dcs == rows and scs == rows) {
            @memcpy(dst[doff..][0 .. rows * cols], src[soff..][0 .. rows * cols]);
        } else {
            for (0..cols) |col| {
                for (0..rows) |row| {
                    dst[doff + row * drs + col * dcs] = src[soff + row * srs + col * scs];
                }
            }
        }
    }

    fn rope(self: Context, rr: anytype) void {
        const src = self.bufF32(rr.src);
        const cs = self.bufF32(rr.cos_sin);
        const dst = self.bufF32(rr.dst);
        const hd: usize = rr.half_d;
        const s_off: usize = rr.src_off;
        const c_off: usize = rr.cs_off;
        const d_off: usize = rr.dst_off;
        const s_rs: usize = rr.src_rs;
        const s_cs: usize = rr.src_cs;
        const c_cs: usize = rr.cs_cs;
        for (0..@as(usize, rr.seq_len)) |col| {
            for (0..hd) |pair| {
                const x_lo = src[s_off + pair * s_rs + col * s_cs];
                const x_hi = src[s_off + (pair + hd) * s_rs + col * s_cs];
                const cos_v = cs[c_off + pair + col * c_cs];
                const sin_v = cs[c_off + pair + 2 * hd + col * c_cs];
                dst[d_off + pair + col * 2 * hd] = x_lo * cos_v - x_hi * sin_v;
                dst[d_off + pair + hd + col * 2 * hd] = x_hi * cos_v + x_lo * sin_v;
            }
        }
    }

    fn matmul(self: Context, m: anytype) void {
        forward.blasSgemm(
            self.bufSlice(m.dst),
            self.bufSlice(m.a),
            self.bufSlice(m.b),
            m.geom.M,
            m.geom.N,
            m.geom.K,
            m.geom.a_row_stride,
            m.geom.a_col_stride,
            m.geom.b_row_stride,
            m.geom.b_col_stride,
            m.geom.a_offset,
            m.geom.b_offset,
            m.geom.dst_offset,
            m.geom.dst_row_stride,
        );
    }

    fn denseProjectionChain(self: Context, matmul_op: backend_mod.DeviceOp, sidecar_op: backend_mod.DeviceOp) bool {
        const m = switch (matmul_op) {
            .matmul => |m| m,
            else => return false,
        };
        switch (sidecar_op) {
            .fused_elementwise => |fe| {
                if (fe.src != m.dst or fe.src_offset != m.geom.dst_offset) return false;
                const expected_n = m.geom.M * m.geom.N;
                if (fe.n != expected_n) return false;
                forward.blasSgemm(
                    self.bufSlice(fe.dst),
                    self.bufSlice(m.a),
                    self.bufSlice(m.b),
                    m.geom.M,
                    m.geom.N,
                    m.geom.K,
                    m.geom.a_row_stride,
                    m.geom.a_col_stride,
                    m.geom.b_row_stride,
                    m.geom.b_col_stride,
                    m.geom.a_offset,
                    m.geom.b_offset,
                    fe.dst_offset,
                    m.geom.dst_row_stride,
                );
                self.fusedElementwise(.{
                    .steps = fe.steps,
                    .n = fe.n,
                    .dst = fe.dst,
                    .src = fe.dst,
                    .dst_offset = fe.dst_offset,
                    .src_offset = fe.dst_offset,
                });
                return true;
            },
            .elementwise => |e| {
                if (e.n != m.geom.M * m.geom.N) return false;
                const src_is_left = e.src0 == m.dst and e.src0_offset == m.geom.dst_offset;
                const src_is_right = e.src1 == m.dst and e.src1_offset == m.geom.dst_offset;
                if (!src_is_left and !src_is_right) return false;
                forward.blasSgemm(
                    self.bufSlice(e.dst),
                    self.bufSlice(m.a),
                    self.bufSlice(m.b),
                    m.geom.M,
                    m.geom.N,
                    m.geom.K,
                    m.geom.a_row_stride,
                    m.geom.a_col_stride,
                    m.geom.b_row_stride,
                    m.geom.b_col_stride,
                    m.geom.a_offset,
                    m.geom.b_offset,
                    e.dst_offset,
                    m.geom.dst_row_stride,
                );
                const step = backend_mod.FusedEwStep{
                    .op = e.op,
                    .is_swapped = src_is_right,
                    .secondary_buf = if (src_is_left) e.src1 else e.src0,
                    .secondary_offset = if (src_is_left) e.src1_offset else e.src0_offset,
                };
                self.fusedElementwise(.{
                    .steps = @as([]const backend_mod.FusedEwStep, &.{step}),
                    .n = e.n,
                    .dst = e.dst,
                    .src = e.dst,
                    .dst_offset = e.dst_offset,
                    .src_offset = e.dst_offset,
                });
                return true;
            },
            else => return false,
        }
    }

    fn denseProjectionBiasChain(self: Context, matmul_op: backend_mod.DeviceOp, repeat_op: backend_mod.DeviceOp, sidecar_op: backend_mod.DeviceOp) bool {
        const m = switch (matmul_op) {
            .matmul => |m| m,
            else => return false,
        };
        const rp = switch (repeat_op) {
            .repeat => |rp| rp,
            else => return false,
        };
        const e = switch (sidecar_op) {
            .elementwise => |e| e,
            else => return false,
        };
        if (!program_mod.matmulRepeatElementwiseBiasCompatible(m, repeat_op, sidecar_op)) return false;
        if (self.smallDenseProjectionBiasRows(m, rp, e)) return true;
        forward.blasSgemm(
            self.bufSlice(e.dst),
            self.bufSlice(m.a),
            self.bufSlice(m.b),
            m.geom.M,
            m.geom.N,
            m.geom.K,
            m.geom.a_row_stride,
            m.geom.a_col_stride,
            m.geom.b_row_stride,
            m.geom.b_col_stride,
            m.geom.a_offset,
            m.geom.b_offset,
            e.dst_offset,
            m.geom.dst_row_stride,
        );
        const dst = self.bufF32(e.dst);
        const bias = self.bufF32(rp.src);
        const M: usize = m.geom.M;
        const N: usize = m.geom.N;
        const dst_offset: usize = e.dst_offset;
        const bias_offset: usize = rp.src_offset;
        addBiasRows(dst, bias, M, N, m.geom.dst_row_stride, dst_offset, bias_offset);
        return true;
    }

    fn smallDenseProjectionBiasRows(self: Context, m: anytype, rp: anytype, e: anytype) bool {
        const g = m.geom;
        if (g.M > 256 or g.N < 64 or g.N > 64 or g.K > 64 or g.N % V != 0) return false;
        if (g.a_col_stride != 1 or g.b_col_stride != 1) return false;
        if (g.a_row_stride != g.K or g.b_row_stride != g.N or g.dst_row_stride != g.N) return false;

        const VecT = @Vector(V, f32);
        const input = self.bufF32(m.a);
        const weight = self.bufF32(m.b);
        const bias = self.bufF32(rp.src);
        const dst = self.bufF32(e.dst);
        for (0..g.M) |row| {
            const input_row = input[g.a_offset + row * g.a_row_stride ..][0..g.K];
            const dst_row = dst[e.dst_offset + row * g.dst_row_stride ..][0..g.N];
            var col: usize = 0;
            while (col < g.N) : (col += V) {
                var acc: VecT = bias[rp.src_offset + col ..][0..V].*;
                for (0..g.K) |k| {
                    const xv: VecT = @splat(input_row[k]);
                    const wv: VecT = weight[g.b_offset + k * g.b_row_stride + col ..][0..V].*;
                    acc += xv * wv;
                }
                dst_row[col..][0..V].* = acc;
            }
        }
        return true;
    }

    fn addBiasRows(
        dst: [*]f32,
        bias: [*]const f32,
        M: usize,
        N: usize,
        dst_row_stride: usize,
        dst_offset: usize,
        bias_offset: usize,
    ) void {
        const VecT = @Vector(V, f32);
        for (0..M) |row| {
            const dst_row = dst[dst_offset + row * dst_row_stride ..][0..N];
            const bias_row = bias[bias_offset..][0..N];
            var i: usize = 0;
            while (i + V <= N) : (i += V) {
                const out_v: VecT = dst_row[i..][0..V].*;
                const bias_v: VecT = bias_row[i..][0..V].*;
                dst_row[i..][0..V].* = out_v + bias_v;
            }
            while (i < N) : (i += 1) {
                dst_row[i] += bias_row[i];
            }
        }
    }

    fn denseProjectionBiasActivationChain(
        self: Context,
        matmul_op: backend_mod.DeviceOp,
        repeat_op: backend_mod.DeviceOp,
        bias_op: backend_mod.DeviceOp,
        activation_op: backend_mod.DeviceOp,
    ) bool {
        const m = switch (matmul_op) {
            .matmul => |m| m,
            else => return false,
        };
        const rp = switch (repeat_op) {
            .repeat => |rp| rp,
            else => return false,
        };
        const activation = switch (activation_op) {
            .elementwise => |e| e,
            else => return false,
        };
        if (!program_mod.matmulRepeatElementwiseBiasActivationCompatible(m, repeat_op, bias_op, activation_op)) return false;
        forward.blasSgemm(
            self.bufSlice(activation.dst),
            self.bufSlice(m.a),
            self.bufSlice(m.b),
            m.geom.M,
            m.geom.N,
            m.geom.K,
            m.geom.a_row_stride,
            m.geom.a_col_stride,
            m.geom.b_row_stride,
            m.geom.b_col_stride,
            m.geom.a_offset,
            m.geom.b_offset,
            activation.dst_offset,
            m.geom.dst_row_stride,
        );
        const dst = self.bufF32(activation.dst);
        const bias = self.bufF32(rp.src);
        const M: usize = m.geom.M;
        const N: usize = m.geom.N;
        const dst_offset: usize = activation.dst_offset;
        const bias_offset: usize = rp.src_offset;
        switch (activation.op) {
            .relu => addBiasReluRows(dst, bias, M, N, m.geom.dst_row_stride, dst_offset, bias_offset),
            .gelu => addBiasGeluRows(dst, bias, M, N, m.geom.dst_row_stride, dst_offset, bias_offset),
            .silu => addBiasSiluRows(dst, bias, M, N, m.geom.dst_row_stride, dst_offset, bias_offset),
            else => unreachable,
        }
        return true;
    }

    fn denseProjectionBiasLogSoftmaxChain(
        self: Context,
        matmul_op: backend_mod.DeviceOp,
        repeat_op: backend_mod.DeviceOp,
        bias_op: backend_mod.DeviceOp,
        logsoftmax_op: backend_mod.DeviceOp,
    ) bool {
        const m = switch (matmul_op) {
            .matmul => |m| m,
            else => return false,
        };
        const rp = switch (repeat_op) {
            .repeat => |rp| rp,
            else => return false,
        };
        const ls = switch (logsoftmax_op) {
            .logsoftmax => |s| s,
            else => return false,
        };
        if (!program_mod.matmulRepeatElementwiseBiasLogSoftmaxCompatible(m, repeat_op, bias_op, logsoftmax_op)) return false;
        forward.blasSgemm(
            self.bufSlice(ls.dst),
            self.bufSlice(m.a),
            self.bufSlice(m.b),
            m.geom.M,
            m.geom.N,
            m.geom.K,
            m.geom.a_row_stride,
            m.geom.a_col_stride,
            m.geom.b_row_stride,
            m.geom.b_col_stride,
            m.geom.a_offset,
            m.geom.b_offset,
            ls.dst_offset,
            m.geom.dst_row_stride,
        );
        addBiasLogSoftmaxRows(
            self.bufF32(ls.dst),
            self.bufF32(rp.src),
            m.geom.M,
            m.geom.N,
            m.geom.dst_row_stride,
            ls.dst_offset,
            rp.src_offset,
        );
        return true;
    }

    fn addBiasLogSoftmaxRows(
        dst: [*]f32,
        bias: [*]const f32,
        M: usize,
        N: usize,
        dst_row_stride: usize,
        dst_offset: usize,
        bias_offset: usize,
    ) void {
        const VecT = @Vector(V, f32);
        for (0..M) |row| {
            const dst_row = dst[dst_offset + row * dst_row_stride ..][0..N];
            const bias_row = bias[bias_offset..][0..N];
            var max_v: VecT = @splat(-std.math.inf(f32));
            var i: usize = 0;
            while (i + V <= N) : (i += V) {
                const dst_v: VecT = dst_row[i..][0..V].*;
                const bias_v: VecT = bias_row[i..][0..V].*;
                const logits = dst_v + bias_v;
                dst_row[i..][0..V].* = logits;
                max_v = @max(max_v, logits);
            }
            var max_scalar: f32 = @reduce(.Max, max_v);
            while (i < N) : (i += 1) {
                const logits = dst_row[i] + bias_row[i];
                dst_row[i] = logits;
                max_scalar = @max(max_scalar, logits);
            }

            const max_vec: VecT = @splat(max_scalar);
            var sum_v: VecT = @splat(0);
            i = 0;
            while (i + V <= N) : (i += V) {
                sum_v += fastExpApproxVec(dst_row[i..][0..V].* - max_vec);
            }
            var sum: f32 = @reduce(.Add, sum_v);
            while (i < N) : (i += 1) sum += @exp(dst_row[i] - max_scalar);

            const log_denom_v: VecT = @splat(max_scalar + @log(sum));
            i = 0;
            while (i + V <= N) : (i += V) {
                dst_row[i..][0..V].* = dst_row[i..][0..V].* - log_denom_v;
            }
            const log_denom = max_scalar + @log(sum);
            while (i < N) : (i += 1) dst_row[i] -= log_denom;
        }
    }

    fn addBiasReluRows(
        dst: [*]f32,
        bias: [*]const f32,
        M: usize,
        N: usize,
        dst_row_stride: usize,
        dst_offset: usize,
        bias_offset: usize,
    ) void {
        const VecT = @Vector(V, f32);
        const zero: VecT = @splat(0.0);
        for (0..M) |row| {
            const dst_row = dst[dst_offset + row * dst_row_stride ..][0..N];
            const bias_row = bias[bias_offset..][0..N];
            var i: usize = 0;
            while (i + V <= N) : (i += V) {
                const gemm_v: VecT = dst_row[i..][0..V].*;
                const bias_v: VecT = bias_row[i..][0..V].*;
                dst_row[i..][0..V].* = @max(gemm_v + bias_v, zero);
            }
            while (i < N) : (i += 1) {
                dst_row[i] = @max(dst_row[i] + bias_row[i], 0.0);
            }
        }
    }

    fn addBiasGeluRows(
        dst: [*]f32,
        bias: [*]const f32,
        M: usize,
        N: usize,
        dst_row_stride: usize,
        dst_offset: usize,
        bias_offset: usize,
    ) void {
        const VecT = @Vector(V, f32);
        for (0..M) |row| {
            const dst_row = dst[dst_offset + row * dst_row_stride ..][0..N];
            const bias_row = bias[bias_offset..][0..N];
            var i: usize = 0;
            while (i + V <= N) : (i += V) {
                const gemm_v: VecT = dst_row[i..][0..V].*;
                const bias_v: VecT = bias_row[i..][0..V].*;
                const a: VecT = gemm_v + bias_v;
                dst_row[i..][0..V].* = geluApproxVec(a);
            }
            while (i < N) : (i += 1) {
                const a = dst_row[i] + bias_row[i];
                const kk = 0.7978845608 * (a + 0.044715 * a * a * a);
                dst_row[i] = 0.5 * a * (1.0 + std.math.tanh(kk));
            }
        }
    }

    fn addBiasSiluRows(
        dst: [*]f32,
        bias: [*]const f32,
        M: usize,
        N: usize,
        dst_row_stride: usize,
        dst_offset: usize,
        bias_offset: usize,
    ) void {
        const VecT = @Vector(V, f32);
        const one: VecT = @splat(1.0);
        for (0..M) |row| {
            const dst_row = dst[dst_offset + row * dst_row_stride ..][0..N];
            const bias_row = bias[bias_offset..][0..N];
            var i: usize = 0;
            while (i + V <= N) : (i += V) {
                const gemm_v: VecT = dst_row[i..][0..V].*;
                const bias_v: VecT = bias_row[i..][0..V].*;
                const a: VecT = gemm_v + bias_v;
                dst_row[i..][0..V].* = a * (one / (one + fastExpApproxVec(-a)));
            }
            while (i < N) : (i += 1) {
                const a = dst_row[i] + bias_row[i];
                dst_row[i] = a / (1.0 + @exp(-a));
            }
        }
    }

    fn fastExpApproxVec(x: @Vector(V, f32)) @Vector(V, f32) {
        const VecT = @Vector(V, f32);
        const IVecT = @Vector(V, i32);
        const UVecT = @Vector(V, u32);
        const inv_ln2: VecT = @splat(1.4426950408889634);
        const ln2: VecT = @splat(0.6931471805599453);
        const half: VecT = @splat(0.5);
        const one: VecT = @splat(1.0);
        const kf = @floor(x * inv_ln2 + half);
        const r = x - kf * ln2;
        const r2 = r * r;
        const r3 = r2 * r;
        const r4 = r2 * r2;
        const r5 = r4 * r;
        const poly = one + r + r2 * @as(VecT, @splat(0.5)) + r3 * @as(VecT, @splat(0.16666666666666666)) + r4 * @as(VecT, @splat(0.041666666666666664)) + r5 * @as(VecT, @splat(0.008333333333333333));
        const ki: IVecT = @intFromFloat(kf);
        const exponent_bits: UVecT = @as(UVecT, @intCast(ki + @as(IVecT, @splat(127)))) << @as(UVecT, @splat(23));
        const pow2: VecT = @bitCast(exponent_bits);
        return poly * pow2;
    }

    fn geluApproxVec(x: @Vector(V, f32)) @Vector(V, f32) {
        const VecT = @Vector(V, f32);
        const k0: VecT = @splat(0.7978845608);
        const k1: VecT = @splat(0.044715);
        const half: VecT = @splat(0.5);
        const one: VecT = @splat(1.0);
        const k = k0 * (x + k1 * x * x * x);
        const e2k = fastExpApproxVec(k + k);
        return half * x * (one + (e2k - one) / (e2k + one));
    }

    fn qmatmul(self: Context, q: anytype) void {
        const input = self.bufF32(q.input);
        const dst_ptr = self.bufF32(q.dst);
        const w = self.qweights[@as(usize, q.weight_idx)];
        const M: usize = q.M;
        const N: usize = q.N;
        const K: usize = q.K;
        const bs: usize = w.block_size;
        const input_offset: usize = q.input_offset;
        const dst_offset: usize = q.dst_offset;
        const input_row_stride: usize = if (q.input_row_stride != 0) q.input_row_stride else K;
        const dst_row_stride: usize = if (q.dst_row_stride != 0) q.dst_row_stride else N;

        if (comptime (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .aarch64_be)) {
            const blocks_per_row = (K + bs - 1) / bs;
            if (M == 1 and input_row_stride == K and dst_row_stride == N and
                K <= 16384 and blocks_per_row <= 512 and
                w.t_data.len >= N * K and w.t_scales.len >= N * blocks_per_row)
            {
                const input_row = input[input_offset..][0..K];
                const dst_row = dst_ptr[dst_offset..][0..N];
                var inp_q_buf: [16384]i8 = undefined;
                var inp_scales_buf: [512]f32 = undefined;
                const inp_q = inp_q_buf[0..K];
                const inp_scales = inp_scales_buf[0..blocks_per_row];
                quant.QuantizedWeight(f32).quantizeInput(input_row, K, bs, inp_q, inp_scales);
                quant.QuantizedWeight(f32).gemvRange(w.t_data, w.t_scales, inp_q, inp_scales, dst_row, 0, N, K, bs);
                return;
            }
        }

        for (0..M) |row| {
            const input_row = input[input_offset + row * input_row_stride ..][0..K];
            const dst_row = dst_ptr[dst_offset + row * dst_row_stride ..][0..N];
            @memset(dst_row, 0);

            const vec_len = 8;
            const Vec = @Vector(vec_len, f32);
            const IVec = @Vector(vec_len, i32);
            const I8Vec = @Vector(vec_len, i8);

            for (0..K) |k| {
                const input_v = input_row[k];
                const w_base = k * N;

                var n: usize = 0;
                while (n < N) {
                    const flat = w_base + n;
                    const scale = w.scales[flat / bs] * input_v;
                    const scale_v: Vec = @splat(scale);
                    const block_rem = bs - (flat % bs);
                    const chunk = @min(block_rem, N - n);

                    var j: usize = 0;
                    while (j + vec_len <= chunk) : (j += vec_len) {
                        const w_vec: I8Vec = w.data[flat + j ..][0..vec_len].*;
                        const f_vec: Vec = @floatFromInt(@as(IVec, w_vec));
                        const d_vec: Vec = dst_row[n + j ..][0..vec_len].*;
                        dst_row[n + j ..][0..vec_len].* = d_vec + f_vec * scale_v;
                    }
                    while (j < chunk) : (j += 1) {
                        dst_row[n + j] += @as(f32, @floatFromInt(w.data[flat + j])) * scale;
                    }
                    n += chunk;
                }
            }
        }
    }

    fn attention(self: Context, att: anytype) void {
        const q_ptr = self.bufF32(att.q);
        const k_ptr = self.bufF32(att.k);
        const v_ptr = self.bufF32(att.v);
        const mask_ptr = self.bufF32(att.mask);
        const dst = self.bufF32(att.dst);
        const dh: usize = att.d_head;
        const sq: usize = att.seq_q;
        const skv: usize = att.seq_kv;
        const k_off: usize = att.k_off;
        const v_off: usize = att.v_off;
        const m_off: usize = att.mask_off;
        const qrs: usize = att.q_rs;
        const qcs: usize = att.q_cs;
        const krs: usize = att.k_rs;
        const kcs: usize = att.k_cs;
        const vrs: usize = att.v_rs;
        const vcs: usize = att.v_cs;
        const mrs: usize = att.mask_rs;
        const mcs: usize = att.mask_cs;
        const drs: usize = att.dst_rs;
        const dcs: usize = att.dst_cs;
        const VecT = @Vector(V, f32);
        const neg_inf = -std.math.inf(f32);

        std.debug.assert(dh <= 512);
        const unit_k = (krs == 1);
        const unit_v = (vrs == 1);
        const unit_q = (qrs == 1);
        const unit_dst = (drs == 1);

        for (0..sq) |qi| {
            const q_off: usize = @as(usize, att.q_off) + qi * qcs;
            const d_off: usize = @as(usize, att.dst_off) + qi * dcs;
            const mask_q_off = m_off + qi * mcs;

            var m_val: f32 = neg_inf;
            var l: f32 = 0;
            var acc_buf: [512]f32 = undefined;
            const acc = acc_buf[0..dh];
            @memset(acc, 0);

            for (0..skv) |s| {
                const mask_add = if (att.has_mask) mask_ptr[mask_q_off + s * mrs] else 0;
                if (!std.math.isFinite(mask_add)) continue;

                var dot: f32 = 0;
                if (unit_q and unit_k) {
                    var dot_v: VecT = @splat(0);
                    var r: usize = 0;
                    const kb = k_off + s * kcs;
                    while (r + V <= dh) : (r += V) {
                        const qv: VecT = q_ptr[q_off + r ..][0..V].*;
                        const kv: VecT = k_ptr[kb + r ..][0..V].*;
                        dot_v += qv * kv;
                    }
                    dot = @reduce(.Add, dot_v);
                    while (r < dh) : (r += 1) dot += q_ptr[q_off + r] * k_ptr[kb + r];
                } else {
                    for (0..dh) |r| dot += q_ptr[q_off + r * qrs] * k_ptr[k_off + r * krs + s * kcs];
                }

                const score = dot * att.scale + mask_add;
                if (!std.math.isFinite(score)) continue;

                const new_m = @max(m_val, score);
                const alpha = if (m_val == neg_inf) @as(f32, 0) else @exp(m_val - new_m);
                const w = @exp(score - new_m);
                l = l * alpha + w;
                m_val = new_m;

                if (unit_v) {
                    const alpha_v: VecT = @splat(alpha);
                    const w_v: VecT = @splat(w);
                    var r: usize = 0;
                    const vb = v_off + s * vcs;
                    while (r + V <= dh) : (r += V) {
                        const av: VecT = acc[r..][0..V].*;
                        const vv: VecT = v_ptr[vb + r ..][0..V].*;
                        acc[r..][0..V].* = av * alpha_v + w_v * vv;
                    }
                    while (r < dh) : (r += 1) {
                        acc[r] = acc[r] * alpha + w * v_ptr[vb + r];
                    }
                } else {
                    for (0..dh) |r| {
                        acc[r] = acc[r] * alpha + w * v_ptr[v_off + r * vrs + s * vcs];
                    }
                }
            }

            const inv_l = if (l > 0) 1.0 / l else @as(f32, 0);
            if (unit_dst) {
                const inv_v: VecT = @splat(inv_l);
                var r: usize = 0;
                while (r + V <= dh) : (r += V) {
                    const av: VecT = acc[r..][0..V].*;
                    dst[d_off + r ..][0..V].* = av * inv_v;
                }
                while (r < dh) : (r += 1) dst[d_off + r] = acc[r] * inv_l;
            } else {
                for (0..dh) |r| dst[d_off + r * drs] = acc[r] * inv_l;
            }
        }
    }
};

test "reference executor elementwise add" {
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 10, 20, 30, 40 };
    var dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &a, .len = a.len },
        .{ .ptr = &b, .len = b.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } });

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, &dst);
}

test "reference executor elementwise sgn and step" {
    var src = [_]f32{ -2, 0, 3, -0.5 };
    var sgn_dst = [_]f32{0} ** 4;
    var step_dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &sgn_dst, .len = sgn_dst.len },
        .{ .ptr = &step_dst, .len = step_dst.len },
    };

    executeOp(&buffers, &.{}, .{ .elementwise = .{ .op = .sgn, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } });
    executeOp(&buffers, &.{}, .{ .elementwise = .{ .op = .step, .dst = 2, .src0 = 0, .src1 = 0, .n = 4 } });

    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 1, -1 }, &sgn_dst);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 0 }, &step_dst);
}

test "reference executor fused elementwise sgn and step" {
    var src = [_]f32{ -2, 0, 3, -0.5 };
    var dst = [_]f32{9} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &dst, .len = dst.len },
    };
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .sgn, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .step, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };

    executeOp(&buffers, &.{}, .{ .fused_elementwise = .{ .steps = &steps, .dst = 1, .src = 0, .n = 4, .dst_offset = 0, .src_offset = 0 } });

    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 0 }, &dst);
}

test "reference executor logsoftmax normalizes rows in log space" {
    var src = [_]f32{ 1, 2, 3, 2, 0, -1 };
    var dst = [_]f32{9} ** 6;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .logsoftmax = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 3,
        .dst_offset = 0,
        .src_offset = 0,
    } });

    for (0..2) |row| {
        const base = row * 3;
        var max_value: f32 = -std.math.inf(f32);
        for (src[base..][0..3]) |value| max_value = @max(max_value, value);
        var sum: f32 = 0;
        for (src[base..][0..3]) |value| sum += @exp(value - max_value);
        const log_denom = max_value + @log(sum);
        for (0..3) |col| {
            try std.testing.expectApproxEqAbs(src[base + col] - log_denom, dst[base + col], 1e-6);
        }
    }
}

test "reference executor vector row softmax and logsoftmax handle full vector chunks" {
    var src = [_]f32{
        1, 2, 3, 4,  0, -1, -2, 5,
        2, 0, 1, -3, 4, 3,  -1, 6,
    };
    var softmax_dst = [_]f32{9} ** 16;
    var logsoftmax_dst = [_]f32{9} ** 16;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &softmax_dst, .len = softmax_dst.len },
        .{ .ptr = &logsoftmax_dst, .len = logsoftmax_dst.len },
    };

    executeOp(&buffers, &.{}, .{ .softmax = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 8,
        .dst_offset = 0,
        .src_offset = 0,
    } });
    executeOp(&buffers, &.{}, .{ .logsoftmax = .{
        .dst = 2,
        .src = 0,
        .rows = 2,
        .cols = 8,
        .dst_offset = 0,
        .src_offset = 0,
    } });

    for (0..2) |row| {
        const base = row * 8;
        var max_value: f32 = -std.math.inf(f32);
        for (src[base..][0..8]) |value| max_value = @max(max_value, value);
        var sum: f32 = 0;
        for (src[base..][0..8]) |value| sum += @exp(value - max_value);
        const log_denom = max_value + @log(sum);
        var softmax_sum: f32 = 0;
        for (0..8) |col| {
            const expected_softmax = @exp(src[base + col] - max_value) / sum;
            softmax_sum += softmax_dst[base + col];
            try std.testing.expectApproxEqAbs(expected_softmax, softmax_dst[base + col], 1e-6);
            try std.testing.expectApproxEqAbs(src[base + col] - log_denom, logsoftmax_dst[base + col], 1e-6);
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), softmax_sum, 1e-6);
    }
}

test "reference executor strided softmax and logsoftmax normalize inner lanes" {
    var src = [_]f32{ 1, 2, 3, 4, 0, -1 };
    var softmax_dst = [_]f32{9} ** 6;
    var logsoftmax_dst = [_]f32{9} ** 6;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &softmax_dst, .len = softmax_dst.len },
        .{ .ptr = &logsoftmax_dst, .len = logsoftmax_dst.len },
    };

    executeOp(&buffers, &.{}, .{ .softmax = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 2,
        .inner = 3,
    } });
    executeOp(&buffers, &.{}, .{ .logsoftmax = .{
        .dst = 2,
        .src = 0,
        .rows = 1,
        .cols = 2,
        .inner = 3,
    } });

    for (0..3) |lane| {
        const a = src[lane];
        const b = src[3 + lane];
        const max_value = @max(a, b);
        const denom = @exp(a - max_value) + @exp(b - max_value);
        const log_denom = max_value + @log(denom);
        try std.testing.expectApproxEqAbs(@exp(a - max_value) / denom, softmax_dst[lane], 1e-6);
        try std.testing.expectApproxEqAbs(@exp(b - max_value) / denom, softmax_dst[3 + lane], 1e-6);
        try std.testing.expectApproxEqAbs(a - log_denom, logsoftmax_dst[lane], 1e-6);
        try std.testing.expectApproxEqAbs(b - log_denom, logsoftmax_dst[3 + lane], 1e-6);
    }
}

test "reference executor matmul" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &a, .len = a.len },
        .{ .ptr = &b, .len = b.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } });

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "reference execution tape fuses dense projection bias logsoftmax" {
    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var weight = [_]f32{0} ** 24;
    var bias = [_]f32{0} ** 8;
    for (&weight, 0..) |*w, i| w.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 8.0;
    for (&bias, 0..) |*b, i| b.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 4)) / 16.0;
    var repeated_bias = [_]f32{0} ** 16;
    var logits = [_]f32{0} ** 16;
    var output = [_]f32{0} ** 16;
    const buffers = [_]Buffer{
        .{ .ptr = &input, .len = input.len },
        .{ .ptr = &weight, .len = weight.len },
        .{ .ptr = &bias, .len = bias.len },
        .{ .ptr = &repeated_bias, .len = repeated_bias.len },
        .{ .ptr = &logits, .len = logits.len },
        .{ .ptr = &output, .len = output.len },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 4,
            .a = 0,
            .b = 1,
            .geom = .{ .M = 2, .N = 8, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 8, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 8 },
        } },
        .{ .repeat = .{
            .dst = 3,
            .src = 2,
            .n = 16,
            .src_ne = .{ 8, 1, 1, 1 },
            .dst_ne = .{ 8, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 8, 1, 1 },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 3, .src0 = 4, .src1 = 3, .n = 16 } },
        .{ .logsoftmax = .{ .dst = 5, .src = 3, .rows = 2, .cols = 8 } },
    };

    var tape = try ExecutionTape.init(std.testing.allocator, &ops);
    defer tape.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), tape.commandLen());
    tape.execute(&buffers, &.{}, &ops);

    var expected_logits = [_]f32{0} ** 16;
    forward.blasSgemm(
        &expected_logits,
        &input,
        &weight,
        2,
        8,
        3,
        3,
        1,
        8,
        1,
        0,
        0,
        0,
        8,
    );
    for (0..2) |row| {
        const base = row * 8;
        var max_value: f32 = -std.math.inf(f32);
        for (0..8) |col| {
            expected_logits[base + col] += bias[col];
            max_value = @max(max_value, expected_logits[base + col]);
        }
        var denom: f32 = 0;
        for (0..8) |col| denom += @exp(expected_logits[base + col] - max_value);
        const log_denom = max_value + @log(denom);
        for (0..8) |col| {
            try std.testing.expectApproxEqAbs(expected_logits[base + col] - log_denom, output[base + col], 1e-4);
        }
    }
}

test "reference executor vector gelu stays within scalar tolerance" {
    var src = [_]f32{ -3, -1.5, -0.5, 0, 0.5, 1, 2, 3 };
    var dst = [_]f32{0} ** 8;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .elementwise = .{
        .op = .gelu,
        .dst = 1,
        .src0 = 0,
        .src1 = 0,
        .n = src.len,
        .dst_offset = 0,
        .src0_offset = 0,
        .src1_offset = 0,
    } });

    for (src, dst) |x, actual| {
        const k = 0.7978845608 * (x + 0.044715 * x * x * x);
        const expected = 0.5 * x * (1.0 + std.math.tanh(k));
        try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "reference execution tape uses patched op payloads" {
    var src = [_]f32{42};
    var dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &dst, .len = dst.len },
    };
    var ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
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
        .patch_stride = 1,
    } }};
    var tape = try ExecutionTape.init(std.testing.allocator, &ops);
    defer tape.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), tape.len());
    try std.testing.expectEqual(@as(usize, 1), tape.commandLen());

    ops[0].slice_assign.dst_offset = 2;
    var profile = profile_mod.RuntimeProfile{};
    tape.executeProfiled(&buffers, &.{}, &ops, &profile);

    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 42, 0 }, &dst);
    const op_command = @intFromEnum(program_mod.ProgramCommandKind.op);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_attempt_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_dispatch_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 0), profile.program_command_failed_counts[op_command]);
}

test "reference execution tape fuses rmsnorm scale row chain" {
    var src = [_]f32{ 3, 4, 0, 1, 2, 2 };
    var scale = [_]f32{ 1, 2, 3 };
    var norm_scratch = [_]f32{0} ** 6;
    var repeat_scratch = [_]f32{0} ** 6;
    var dst = [_]f32{0} ** 6;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &scale, .len = scale.len },
        .{ .ptr = &norm_scratch, .len = norm_scratch.len },
        .{ .ptr = &repeat_scratch, .len = repeat_scratch.len },
        .{ .ptr = &dst, .len = dst.len },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .rmsnorm = .{
            .dst = 2,
            .src = 0,
            .rows = 2,
            .cols = 3,
            .eps = 0,
            .src_offset = 0,
            .dst_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 3,
            .src = 1,
            .n = 6,
            .src_offset = 0,
            .dst_offset = 0,
            .src_ne = .{ 3, 1, 1, 1 },
            .dst_ne = .{ 3, 2, 1, 1 },
            .src_strides = .{ 1, 3, 3, 3 },
            .dst_strides = .{ 1, 3, 6, 6 },
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 4,
            .src0 = 2,
            .src1 = 3,
            .n = 6,
            .dst_offset = 0,
            .src0_offset = 0,
            .src1_offset = 0,
        } },
    };

    var tape = try ExecutionTape.init(std.testing.allocator, &ops);
    defer tape.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), tape.len());
    try std.testing.expectEqual(@as(usize, 1), tape.commandLen());

    var profile = profile_mod.RuntimeProfile{};
    tape.executeProfiled(&buffers, &.{}, &ops, &profile);

    for (0..2) |row| {
        const base = row * 3;
        const ss = src[base] * src[base] + src[base + 1] * src[base + 1] + src[base + 2] * src[base + 2];
        const inv_rms = 1.0 / @sqrt(ss / 3.0);
        for (0..3) |col| {
            try std.testing.expectApproxEqAbs(src[base + col] * inv_rms * scale[col], dst[base + col], 1e-6);
        }
    }
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &norm_scratch);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &repeat_scratch);

    const row_chain = @intFromEnum(program_mod.ProgramCommandKind.row_chain);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_attempt_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_dispatch_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 0), profile.program_command_failed_counts[row_chain]);
}

test "reference execution tape fuses rmsnorm scale activation row chain" {
    var src = [_]f32{ 3, 4, 0, 1, 2, 2 };
    var scale = [_]f32{ 1, 2, 3 };
    var norm_scratch = [_]f32{0} ** 6;
    var repeat_scratch = [_]f32{0} ** 6;
    var scale_scratch = [_]f32{0} ** 6;
    var dst = [_]f32{0} ** 6;
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &scale, .len = scale.len },
        .{ .ptr = &norm_scratch, .len = norm_scratch.len },
        .{ .ptr = &repeat_scratch, .len = repeat_scratch.len },
        .{ .ptr = &scale_scratch, .len = scale_scratch.len },
        .{ .ptr = &dst, .len = dst.len },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .rmsnorm = .{
            .dst = 2,
            .src = 0,
            .rows = 2,
            .cols = 3,
            .eps = 0,
            .src_offset = 0,
            .dst_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 3,
            .src = 1,
            .n = 6,
            .src_offset = 0,
            .dst_offset = 0,
            .src_ne = .{ 3, 1, 1, 1 },
            .dst_ne = .{ 3, 2, 1, 1 },
            .src_strides = .{ 1, 3, 3, 3 },
            .dst_strides = .{ 1, 3, 6, 6 },
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 4,
            .src0 = 2,
            .src1 = 3,
            .n = 6,
            .dst_offset = 0,
            .src0_offset = 0,
            .src1_offset = 0,
        } },
        .{ .elementwise = .{
            .op = .gelu,
            .dst = 5,
            .src0 = 4,
            .src1 = 4,
            .n = 6,
            .dst_offset = 0,
            .src0_offset = 0,
            .src1_offset = 0,
        } },
    };

    var tape = try ExecutionTape.init(std.testing.allocator, &ops);
    defer tape.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 4), tape.len());
    try std.testing.expectEqual(@as(usize, 1), tape.commandLen());

    var profile = profile_mod.RuntimeProfile{};
    tape.executeProfiled(&buffers, &.{}, &ops, &profile);

    for (0..2) |row| {
        const base = row * 3;
        const ss = src[base] * src[base] + src[base + 1] * src[base + 1] + src[base + 2] * src[base + 2];
        const inv_rms = 1.0 / @sqrt(ss / 3.0);
        for (0..3) |col| {
            const scaled = src[base + col] * inv_rms * scale[col];
            const k = 0.7978845608 * (scaled + 0.044715 * scaled * scaled * scaled);
            const expected = 0.5 * scaled * (1.0 + std.math.tanh(k));
            try std.testing.expectApproxEqAbs(expected, dst[base + col], 1e-6);
        }
    }
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &norm_scratch);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &repeat_scratch);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &scale_scratch);

    const row_chain = @intFromEnum(program_mod.ProgramCommandKind.row_chain);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_attempt_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_dispatch_counts[row_chain]);
    try std.testing.expectEqual(@as(u64, 0), profile.program_command_failed_counts[row_chain]);
}

test "reference executor conv2d relu handles multiple output channels" {
    const batch = 2;
    const in_w = 4;
    const in_h = 4;
    const out_w = 2;
    const out_h = 2;
    const out_channels = 2;
    var src = [_]f32{
        1,  -2,  3,   4,
        5,  6,   -7,  8,
        9,  10,  11,  -12,
        13, 14,  15,  16,

        -1, -2,  -3,  -4,
        5,  6,   7,   8,
        -9, 10,  -11, 12,
        13, -14, 15,  -16,
    };
    var weight = [_]f32{
        1,    0,    -1,
        0,    1,    0,
        -1,   0,    1,

        -1,   0.5,  0.25,
        0,    -0.5, 0,
        0.25, 0.5,  -1,
    };
    var bias = [_]f32{ -1.5, 0.75 };
    var dst = [_]f32{-99} ** (batch * out_channels * out_w * out_h);
    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &weight, .len = weight.len },
        .{ .ptr = &bias, .len = bias.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .conv2d = .{
        .dst = 3,
        .src = 0,
        .weight = 1,
        .bias = 2,
        .out_w = out_w,
        .out_h = out_h,
        .in_w = in_w,
        .in_h = in_h,
        .in_channels = 1,
        .out_channels = out_channels,
        .kernel_w = 3,
        .kernel_h = 3,
        .batch = batch,
        .relu = true,
    } });

    for (0..batch) |n| {
        for (0..out_channels) |oc| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var expected = bias[oc];
                    for (0..3) |ky| {
                        for (0..3) |kx| {
                            const src_idx = n * in_w * in_h + (oy + ky) * in_w + ox + kx;
                            const weight_idx = oc * 9 + ky * 3 + kx;
                            expected += src[src_idx] * weight[weight_idx];
                        }
                    }
                    if (expected < 0) expected = 0;
                    const dst_idx = n * out_channels * out_w * out_h + oc * out_w * out_h + oy * out_w + ox;
                    try std.testing.expectApproxEqAbs(expected, dst[dst_idx], 1e-6);
                }
            }
        }
    }
}

test "reference executor rope reads sine from packed second half" {
    var src = [_]f32{ 1, 2, 3, 4 };
    var cos_sin = [_]f32{ 0.5, 0.25, 0.5, 0.25, 1.0, 2.0, 1.0, 2.0 };
    var dst = [_]f32{0} ** 4;

    const buffers = [_]Buffer{
        .{ .ptr = &src, .len = src.len },
        .{ .ptr = &cos_sin, .len = cos_sin.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .rope = .{
        .dst = 2,
        .src = 0,
        .cos_sin = 1,
        .half_d = 2,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 8,
    } });

    try std.testing.expectApproxEqAbs(@as(f32, -2.5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -7.5), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), dst[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dst[3], 1e-6);
}

test "reference executor attention matches scalar two-token decode" {
    var q = [_]f32{ 0.2, -0.4, 0.6, -0.8 };
    var k = [_]f32{
        0.1,  0.3, -0.5, 0.7,
        -0.2, 0.4, 0.8,  -0.6,
    };
    var v = [_]f32{
        1.0,  2.0,  -1.0, 0.5,
        -0.5, 0.25, 1.5,  -2.0,
    };
    var mask = [_]f32{ 0, 0 };
    var dst = [_]f32{0} ** 4;

    const buffers = [_]Buffer{
        .{ .ptr = &q, .len = q.len },
        .{ .ptr = &k, .len = k.len },
        .{ .ptr = &v, .len = v.len },
        .{ .ptr = &mask, .len = mask.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = 2,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 2,
        .dst_rs = 1,
        .dst_cs = 4,
    } });

    const s0 = 0.5 * (q[0] * k[0] + q[1] * k[1] + q[2] * k[2] + q[3] * k[3]);
    const s1 = 0.5 * (q[0] * k[4] + q[1] * k[5] + q[2] * k[6] + q[3] * k[7]);
    const m = @max(s0, s1);
    const e0 = @exp(s0 - m);
    const e1 = @exp(s1 - m);
    const inv = 1.0 / (e0 + e1);
    const w0 = e0 * inv;
    const w1 = e1 * inv;

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(w0 * v[i] + w1 * v[4 + i], dst[i], 1e-6);
    }
}

test "reference executor qmatmul uses row-major quantized weights" {
    var input = [_]f32{ 1, 2, 3, -1, 0.5, 4 };
    var dst = [_]f32{0} ** 6;
    const data = [_]i8{ 2, -1, 3, 4, -2, 1, -3, 5, 2 };
    const scales = [_]f32{ 0.5, 0.25, 1.0 };
    const qweights = [_]QWeight{.{ .data = &data, .scales = &scales, .block_size = 4 }};
    const buffers = [_]Buffer{
        .{ .ptr = &input, .len = input.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &qweights, .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 3,
        .K = 3,
    } });

    try std.testing.expectApproxEqAbs(@as(f32, 2.75), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.25), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), dst[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), dst[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.25), dst[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.625), dst[5], 1e-6);

    var input_offset = [_]f32{ 99, 1, 2, 3, 99, -1, 0.5, 4, 99 };
    var dst_offset = [_]f32{-7} ** 9;
    const offset_buffers = [_]Buffer{
        .{ .ptr = &input_offset, .len = input_offset.len },
        .{ .ptr = &dst_offset, .len = dst_offset.len },
    };
    executeOp(&offset_buffers, &qweights, .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 3,
        .K = 3,
        .input_offset = 1,
        .input_row_stride = 4,
        .dst_offset = 1,
        .dst_row_stride = 4,
    } });
    try std.testing.expectApproxEqAbs(@as(f32, 2.75), dst_offset[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.25), dst_offset[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), dst_offset[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), dst_offset[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.25), dst_offset[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.625), dst_offset[7], 1e-6);
}
