//! Shared helpers for compiled backend programs.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const DeviceOpTag = std.meta.Tag(backend_mod.DeviceOp);

pub const RuntimePatchEnvelope = struct {
    max_cache_write_pos: ?u32 = std.math.maxInt(u32),
    max_attention_seq_kv: ?u32 = std.math.maxInt(u32),

    pub fn initProgram(program: backend_mod.DeviceProgram) !RuntimePatchEnvelope {
        var out = RuntimePatchEnvelope{};
        for (program.ops, 0..) |op, op_index| switch (op) {
            .slice_assign => |sa| if (sa.patch_stride != 0) {
                _ = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
                if (maxCacheWritePos(sa, program.buffer_sizes)) |max| {
                    if (out.max_cache_write_pos) |current| out.max_cache_write_pos = @min(current, max);
                } else {
                    out.max_cache_write_pos = null;
                }
            },
            .attention => |att| if (att.patch_seq_kv) {
                _ = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
                if (maxAttentionSeqKv(att, program.buffer_sizes)) |max| {
                    if (out.max_attention_seq_kv) |current| out.max_attention_seq_kv = @min(current, max);
                } else {
                    out.max_attention_seq_kv = null;
                }
            },
            else => {},
        };
        return out;
    }
};

const RuntimeBindings = struct {
    patch_op_indices: []u32 = &.{},
    patch_shape: backend_mod.RuntimePatchShape = .{},
    max_cache_write_pos: ?u32 = std.math.maxInt(u32),
    max_attention_seq_kv: ?u32 = std.math.maxInt(u32),
    window: backend_mod.RuntimeWindow = .{ .position = 0, .len = 0 },
    valid: bool = false,

    fn initProgram(alloc: std.mem.Allocator, program: backend_mod.DeviceProgram) !RuntimeBindings {
        const ops = program.ops;
        const counts = try countRuntimePatchHoles(ops);
        const n_cache: usize = @intCast(counts.cache_write_pos);
        const n_attention: usize = @intCast(counts.attention_seq_kv);
        const envelope = try RuntimePatchEnvelope.initProgram(program);

        const patch_op_indices = try alloc.alloc(u32, n_cache + n_attention);
        errdefer alloc.free(patch_op_indices);
        const cache_write_pos_ops = patch_op_indices[0..n_cache];
        const attention_seq_kv_ops = patch_op_indices[n_cache..];
        var cache_i: usize = 0;
        var h = RuntimeStencilHasher{};
        const has_dynamic_patches = counts.cache_write_pos != 0 or counts.attention_seq_kv != 0;
        if (has_dynamic_patches) h.add(counts.cache_write_pos);
        for (ops, 0..) |op, op_index| {
            switch (op) {
                .slice_assign => |sa| if (sa.patch_stride != 0) {
                    const idx = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
                    cache_write_pos_ops[cache_i] = idx;
                    cache_i += 1;
                    if (has_dynamic_patches) addSliceRuntimeStencil(&h, idx, sa);
                },
                else => {},
            }
        }
        var attention_i: usize = 0;
        if (has_dynamic_patches) h.add(counts.attention_seq_kv);
        for (ops, 0..) |op, op_index| {
            switch (op) {
                .attention => |att| if (att.patch_seq_kv) {
                    const idx = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
                    attention_seq_kv_ops[attention_i] = idx;
                    attention_i += 1;
                    if (has_dynamic_patches) addAttentionRuntimeStencil(&h, idx, att);
                },
                else => {},
            }
        }
        std.debug.assert(cache_i == n_cache);
        std.debug.assert(attention_i == n_attention);

        return .{
            .patch_op_indices = patch_op_indices,
            .patch_shape = backend_mod.RuntimePatchShape.actual(counts.cache_write_pos, counts.attention_seq_kv, if (has_dynamic_patches) h.state else 0),
            .max_cache_write_pos = envelope.max_cache_write_pos,
            .max_attention_seq_kv = envelope.max_attention_seq_kv,
        };
    }

    fn deinit(self: *RuntimeBindings, alloc: std.mem.Allocator) void {
        if (self.patch_op_indices.len > 0) alloc.free(self.patch_op_indices);
        self.* = .{};
    }

    fn windowFit(self: RuntimeBindings, window: backend_mod.RuntimeWindow) ?u32 {
        const max_cache_write_pos = self.max_cache_write_pos orelse return null;
        const max_attention_seq_kv = self.max_attention_seq_kv orelse return null;
        const attention_seq_kv = window.attentionSeqKv() orelse return null;
        if (window.position > max_cache_write_pos or attention_seq_kv > max_attention_seq_kv) return null;
        return attention_seq_kv;
    }

    fn apply(self: *RuntimeBindings, ops: []backend_mod.DeviceOp, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
        const attention_seq_kv = self.windowFit(window) orelse return .invalid;

        var changed = false;
        const patch_cache_write_pos = !self.valid or self.window.position != window.position;
        const patch_attention_seq_kv = !self.valid or self.window.attentionSeqKv().? != attention_seq_kv;
        const n_cache: usize = @intCast(self.patch_shape.runtime_patch_cache_write_pos_holes);

        if (patch_cache_write_pos) {
            for (self.patch_op_indices[0..n_cache]) |op_index| {
                const idx: usize = @intCast(op_index);
                std.debug.assert(idx < ops.len and ops[idx] == .slice_assign);
                const sa = &ops[idx].slice_assign;
                sa.dst_offset = sa.dst_base_offset + window.position * sa.patch_stride;
                changed = true;
            }
        }
        if (patch_attention_seq_kv) {
            for (self.patch_op_indices[n_cache..]) |op_index| {
                const idx: usize = @intCast(op_index);
                std.debug.assert(idx < ops.len and ops[idx] == .attention);
                ops[idx].attention.seq_kv = attention_seq_kv;
                changed = true;
            }
        }
        self.window = window;
        self.valid = true;
        return if (changed) .changed else .unchanged;
    }

    fn runtimePatchEnvelope(self: RuntimeBindings) RuntimePatchEnvelope {
        return .{
            .max_cache_write_pos = self.max_cache_write_pos,
            .max_attention_seq_kv = self.max_attention_seq_kv,
        };
    }
};

const RuntimePatchCounts = struct {
    cache_write_pos: u32 = 0,
    attention_seq_kv: u32 = 0,
};

fn countRuntimePatchHoles(ops: []const backend_mod.DeviceOp) !RuntimePatchCounts {
    var counts = RuntimePatchCounts{};
    for (ops, 0..) |op, op_index| switch (op) {
        .slice_assign => |sa| if (sa.patch_stride != 0) {
            _ = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
            counts.cache_write_pos = std.math.add(u32, counts.cache_write_pos, 1) catch return error.UnsupportedDeviceOp;
        },
        .attention => |att| if (att.patch_seq_kv) {
            _ = std.math.cast(u32, op_index) orelse return error.UnsupportedDeviceOp;
            counts.attention_seq_kv = std.math.add(u32, counts.attention_seq_kv, 1) catch return error.UnsupportedDeviceOp;
        },
        else => {},
    };
    return counts;
}

fn maxCacheWritePos(sa: anytype, buffer_sizes: []const usize) ?u32 {
    const size = bufferSize(buffer_sizes, sa.dst) orelse return null;
    const span = strided2Span(sa.rows, sa.cols, sa.dst_row_stride, sa.dst_col_stride) orelse return null;
    if (span > size or sa.dst_base_offset > size - span) return null;
    if (sa.patch_stride == 0) return std.math.maxInt(u32);
    const buffer_room = size - span - sa.dst_base_offset;
    const u32_room = std.math.maxInt(u32) - sa.dst_base_offset;
    return @intCast(@min(buffer_room / sa.patch_stride, u32_room / sa.patch_stride));
}

fn maxAttentionSeqKv(att: anytype, buffer_sizes: []const usize) ?u32 {
    if (!backend_mod.BufferBounds.strided2Fits(buffer_sizes, att.q, att.q_off, att.d_head, att.seq_q, att.q_rs, att.q_cs) or
        !backend_mod.BufferBounds.strided2Fits(buffer_sizes, att.dst, att.dst_off, att.d_head, att.seq_q, att.dst_rs, att.dst_cs)) return null;
    var max_seq_kv = maxDynamicExtent(buffer_sizes, att.k, att.k_off, att.d_head, att.k_rs, att.k_cs) orelse return null;
    max_seq_kv = @min(max_seq_kv, maxDynamicExtent(buffer_sizes, att.v, att.v_off, att.d_head, att.v_rs, att.v_cs) orelse return null);
    if (att.has_mask) {
        max_seq_kv = @min(max_seq_kv, maxDynamicExtent(buffer_sizes, att.mask, att.mask_off, att.seq_q, att.mask_cs, att.mask_rs) orelse return null);
    }
    return max_seq_kv;
}

fn bufferSize(buffer_sizes: []const usize, idx: u16) ?usize {
    const i: usize = idx;
    return if (i < buffer_sizes.len) buffer_sizes[i] else null;
}

fn strided2Span(rows: u32, cols: u32, row_stride: u32, col_stride: u32) ?usize {
    if (rows == 0 or cols == 0) return 0;
    const row_span = std.math.mul(usize, rows - 1, row_stride) catch return null;
    const col_span = std.math.mul(usize, cols - 1, col_stride) catch return null;
    return std.math.add(usize, std.math.add(usize, row_span, col_span) catch return null, 1) catch return null;
}

fn maxDynamicExtent(buffer_sizes: []const usize, idx: u16, offset: u32, fixed_extent: u32, fixed_stride: u32, dynamic_stride: u32) ?u32 {
    const size = bufferSize(buffer_sizes, idx) orelse return null;
    if (@as(usize, offset) > size) return null;
    if (fixed_extent == 0) return std.math.maxInt(u32);
    const fixed_span = std.math.mul(usize, fixed_extent - 1, fixed_stride) catch return null;
    const first_dynamic_end = std.math.add(usize, std.math.add(usize, offset, fixed_span) catch return null, 1) catch return null;
    if (first_dynamic_end > size) return 0;
    if (dynamic_stride == 0) return std.math.maxInt(u32);
    return @intCast(@min(@as(usize, std.math.maxInt(u32)), (size - first_dynamic_end) / dynamic_stride + 1));
}

fn dupeRuntimeOps(alloc: std.mem.Allocator, source_ops: []const backend_mod.DeviceOp) ![]backend_mod.DeviceOp {
    const ops = try alloc.dupe(backend_mod.DeviceOp, source_ops);
    errdefer {
        freeRuntimeOpPayloads(alloc, ops);
        alloc.free(ops);
    }
    for (ops) |*op| switch (op.*) {
        .fused_elementwise => |*fe| fe.steps = &.{},
        else => {},
    };
    for (ops, source_ops) |*op, source| switch (source) {
        .fused_elementwise => |fe| if (fe.steps.len > 0) {
            op.fused_elementwise.steps = try alloc.dupe(backend_mod.FusedEwStep, fe.steps);
        },
        else => {},
    };
    return ops;
}

fn freeRuntimeOpPayloads(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp) void {
    for (ops) |op| switch (op) {
        .fused_elementwise => |fe| if (fe.steps.len > 0) alloc.free(fe.steps),
        else => {},
    };
}

/// Deep compiler Module that schedules DeviceProgram ops into executable
/// backend commands.
pub const Kernelizer = struct {
    command_policy: CommandStreamPolicy,

    pub fn init(command_policy: CommandStreamPolicy) Kernelizer {
        return .{ .command_policy = command_policy };
    }

    pub fn default() Kernelizer {
        return init(CommandStreamPolicy.default());
    }

    pub fn kernelize(self: Kernelizer, alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp) !KernelPlan {
        return KernelPlan.init(alloc, ops, self.command_policy);
    }

    pub fn executionPlan(
        self: Kernelizer,
        alloc: std.mem.Allocator,
        ops: []const backend_mod.DeviceOp,
        schedule_policy: SchedulePolicy,
        stages: []const StagePolicy,
    ) !ExecutionPlan {
        return buildExecutionPlan(alloc, ops, schedule_policy, stages, self);
    }
};

/// Scheduled executable command plan for a copied DeviceOp tape.
///
/// KernelPlan is the cold compiler artifact between raw DeviceProgram ops and
/// ProgramStencil. It owns the command stream and its inspection shape, while
/// ProgramStencil adds runtime patching and Session-bound execution state.
pub const KernelPlan = struct {
    commands: []const ProgramCommand = &.{},
    command_shape: ProgramCommandStreamShape = .{},

    fn init(
        alloc: std.mem.Allocator,
        ops: []const backend_mod.DeviceOp,
        policy: CommandStreamPolicy,
    ) !KernelPlan {
        const commands = try buildProgramCommands(alloc, ops, policy);
        errdefer if (commands.len > 0) alloc.free(commands);
        const command_shape = try ProgramCommandStreamShape.fromCommands(commands);
        return .{
            .commands = commands,
            .command_shape = command_shape,
        };
    }

    pub fn clone(self: KernelPlan, alloc: std.mem.Allocator) !KernelPlan {
        const commands = try alloc.dupe(ProgramCommand, self.commands);
        return .{
            .commands = commands,
            .command_shape = self.command_shape,
        };
    }

    pub fn deinit(self: *KernelPlan, alloc: std.mem.Allocator) void {
        if (self.commands.len > 0) alloc.free(self.commands);
        self.* = .{};
    }
};

/// Backend-agnostic executable skeleton for a DeviceProgram.
///
/// A ProgramStencil owns the copied op tape, command stream, and bounded runtime
/// update table used to apply per-step windows. Inspection and execution should
/// both flow through this same object: command evidence describes the command
/// stream that backends lower, and runtime patch evidence describes the exact
/// mutable op tape that patchRuntimeWindow() updates before a backend step.
pub const ProgramStencil = struct {
    ops: []backend_mod.DeviceOp = &.{},
    buffer_sizes: []const usize = &.{},
    runtime_bindings: RuntimeBindings = .{},
    kernel_plan: KernelPlan = .{},

    pub const Inspection = struct {
        op_count: usize,
        buffer_count: usize,
        buffer_element_count: usize,
        runtime_patch_shape: backend_mod.RuntimePatchShape,
        runtime_patch_envelope: RuntimePatchEnvelope,
        command_shape: ProgramCommandStreamShape,
    };

    pub fn initProgram(alloc: std.mem.Allocator, program: backend_mod.DeviceProgram) !ProgramStencil {
        return initProgramWithKernelizer(alloc, program, Kernelizer.default());
    }

    pub fn initProgramWithKernelizer(
        alloc: std.mem.Allocator,
        program: backend_mod.DeviceProgram,
        kernelizer: Kernelizer,
    ) !ProgramStencil {
        const ops = try dupeRuntimeOps(alloc, program.ops);
        errdefer {
            freeRuntimeOpPayloads(alloc, ops);
            alloc.free(ops);
        }
        const buffer_sizes = try alloc.dupe(usize, program.buffer_sizes);
        errdefer alloc.free(buffer_sizes);
        var runtime_bindings = try RuntimeBindings.initProgram(alloc, program);
        errdefer runtime_bindings.deinit(alloc);
        const kernel_plan = try kernelizer.kernelize(alloc, ops);
        return .{
            .ops = ops,
            .buffer_sizes = buffer_sizes,
            .runtime_bindings = runtime_bindings,
            .kernel_plan = kernel_plan,
        };
    }

    pub fn deinit(self: *ProgramStencil, alloc: std.mem.Allocator) void {
        self.runtime_bindings.deinit(alloc);
        self.kernel_plan.deinit(alloc);
        if (self.buffer_sizes.len > 0) alloc.free(self.buffer_sizes);
        if (self.ops.len > 0) {
            freeRuntimeOpPayloads(alloc, self.ops);
            alloc.free(self.ops);
        }
        self.* = .{};
    }

    pub fn clone(self: ProgramStencil, alloc: std.mem.Allocator) !ProgramStencil {
        const ops = try dupeRuntimeOps(alloc, self.ops);
        errdefer {
            freeRuntimeOpPayloads(alloc, ops);
            alloc.free(ops);
        }
        const buffer_sizes = try alloc.dupe(usize, self.buffer_sizes);
        errdefer alloc.free(buffer_sizes);
        var runtime_bindings = try RuntimeBindings.initProgram(alloc, .{
            .ops = ops,
            .n_buffers = std.math.cast(u16, buffer_sizes.len) orelse return error.UnsupportedDeviceOp,
            .buffer_sizes = buffer_sizes,
            .initial_uploads = &.{},
        });
        errdefer runtime_bindings.deinit(alloc);
        const kernel_plan = try self.kernel_plan.clone(alloc);
        return .{
            .ops = ops,
            .buffer_sizes = buffer_sizes,
            .runtime_bindings = runtime_bindings,
            .kernel_plan = kernel_plan,
        };
    }

    pub fn patchRuntimeWindow(self: *ProgramStencil, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
        return self.runtime_bindings.apply(self.ops, window);
    }

    pub fn ioValid(self: ProgramStencil, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) bool {
        return backend_mod.BufferBounds.programIOListsValid(self.buffer_sizes, inputs, outputs);
    }

    pub fn runtimePatchShape(self: ProgramStencil) backend_mod.RuntimePatchShape {
        return self.runtime_bindings.patch_shape;
    }

    pub fn inspect(self: ProgramStencil) Inspection {
        return .{
            .op_count = self.ops.len,
            .buffer_count = self.buffer_sizes.len,
            .buffer_element_count = totalElements(self.buffer_sizes),
            .runtime_patch_shape = self.runtimePatchShape(),
            .runtime_patch_envelope = self.runtime_bindings.runtimePatchEnvelope(),
            .command_shape = self.kernel_plan.command_shape,
        };
    }
};

fn totalElements(buffer_sizes: []const usize) usize {
    var total: usize = 0;
    for (buffer_sizes) |len| total = std.math.add(usize, total, len) catch return std.math.maxInt(usize);
    return total;
}

test "kernel plan owns command stream evidence" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 3, .src0 = 2, .src1 = 1, .n = 4 } },
    };

    var plan = try Kernelizer.default().kernelize(std.testing.allocator, &ops);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expect(plan.commands.len > 0);
    try std.testing.expectEqual(plan.command_shape, try ProgramCommandStreamShape.fromCommands(plan.commands));

    var clone = try plan.clone(std.testing.allocator);
    defer clone.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(ProgramCommand, plan.commands, clone.commands);
    try std.testing.expectEqual(plan.command_shape, clone.command_shape);
}

const RuntimeStencilHasher = struct {
    state: u64 = 14695981039346656037,

    fn add(self: *RuntimeStencilHasher, value: u64) void {
        var v = value;
        for (0..8) |_| {
            self.state ^= v & 0xff;
            self.state *%= 1099511628211;
            v >>= 8;
        }
    }
};

fn addSliceRuntimeStencil(h: *RuntimeStencilHasher, op_index: u32, sa: anytype) void {
    h.add(1);
    h.add(op_index);
    h.add(sa.dst);
    h.add(sa.src);
    h.add(sa.rows);
    h.add(sa.cols);
    h.add(sa.dst_base_offset);
    h.add(sa.dst_row_stride);
    h.add(sa.dst_col_stride);
    h.add(sa.src_offset);
    h.add(sa.src_row_stride);
    h.add(sa.src_col_stride);
    h.add(sa.patch_stride);
}

fn addAttentionRuntimeStencil(h: *RuntimeStencilHasher, op_index: u32, att: anytype) void {
    h.add(2);
    h.add(op_index);
    h.add(att.dst);
    h.add(att.q);
    h.add(att.k);
    h.add(att.v);
    h.add(att.mask);
    h.add(@intFromBool(att.has_mask));
    h.add(att.d_head);
    h.add(att.seq_q);
    h.add(att.q_off);
    h.add(att.k_off);
    h.add(att.v_off);
    h.add(att.mask_off);
    h.add(att.dst_off);
    h.add(att.q_rs);
    h.add(att.q_cs);
    h.add(att.k_rs);
    h.add(att.k_cs);
    h.add(att.v_rs);
    h.add(att.v_cs);
    h.add(att.mask_rs);
    h.add(att.mask_cs);
    h.add(att.dst_rs);
    h.add(att.dst_cs);
}

fn testAttentionOp() backend_mod.DeviceOp {
    return .{ .attention = .{
        .dst = 4,
        .q = 1,
        .k = 2,
        .v = 3,
        .mask = 0,
        .has_mask = false,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = 1,
        .scale = 1.0,
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
        .mask_cs = 4,
        .dst_rs = 1,
        .dst_cs = 4,
    } };
}

test "program stencil owns dynamic patching state" {
    var attention = testAttentionOp();
    attention.attention.patch_seq_kv = true;
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 8,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        attention,
    };

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 4, 32, 20, 20, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(3, 2)));
    try std.testing.expectEqual(@as(u32, 20), program_stencil.ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(@as(u32, 5), program_stencil.ops[1].attention.seq_kv);
    try std.testing.expectEqual(backend_mod.RuntimePatchShape.actual(1, 1, program_stencil.runtimePatchShape().runtime_patch_stencil_hash), program_stencil.runtimePatchShape());
    try std.testing.expect(program_stencil.runtimePatchShape().runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(u32, 8), ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.unchanged, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(3, 2)));
}

test "program stencil inspection reports executable shape" {
    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 2,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 2,
    } }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &.{ 2, 8 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    const inspection = program_stencil.inspect();
    try std.testing.expectEqual(@as(usize, 1), inspection.op_count);
    try std.testing.expectEqual(@as(usize, 2), inspection.buffer_count);
    try std.testing.expectEqual(@as(usize, 10), inspection.buffer_element_count);
    try std.testing.expectEqual(backend_mod.RuntimePatchShape.actual(1, 0, inspection.runtime_patch_shape.runtime_patch_stencil_hash), inspection.runtime_patch_shape);
    try std.testing.expect(inspection.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(usize, 1), program_stencil.kernel_plan.commands.len);
    try std.testing.expectEqual(@as(u32, 1), inspection.command_shape.command_count);
    try std.testing.expectEqual(@as(u32, 1), inspection.command_shape.covered_ops);
    try std.testing.expectEqual(@as(u32, 0), inspection.command_shape.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), inspection.command_shape.command_kind_counts[@intFromEnum(ProgramCommandKind.op)]);
    try std.testing.expect(inspection.command_shape.command_stencil_hash != 0);
    try std.testing.expectEqual(inspection.command_shape, try ProgramCommandStreamShape.fromCommands(program_stencil.kernel_plan.commands));
}

test "runtime patch stencil hash changes when patch geometry changes" {
    const a_ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 1,
        .dst_base_offset = 4,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 2,
    } }};
    const b_ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 1,
        .dst_base_offset = 4,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 3,
    } }};

    var a = try ProgramStencil.initProgram(std.testing.allocator, .{ .ops = &a_ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 16 }, .initial_uploads = &.{} });
    defer a.deinit(std.testing.allocator);
    var b = try ProgramStencil.initProgram(std.testing.allocator, .{ .ops = &b_ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 16 }, .initial_uploads = &.{} });
    defer b.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), a.runtimePatchShape().runtime_patch_holes);
    try std.testing.expectEqual(a.runtimePatchShape().runtime_patch_holes, b.runtimePatchShape().runtime_patch_holes);
    try std.testing.expect(a.runtimePatchShape().runtime_patch_stencil_hash != b.runtimePatchShape().runtime_patch_stencil_hash);
}

test "program stencil command hash changes when command stream changes" {
    const a_ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } }};
    const b_ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 3, .src0 = 2, .src1 = 1, .n = 4 } },
    };

    var a = try ProgramStencil.initProgram(std.testing.allocator, .{ .ops = &a_ops, .n_buffers = 3, .buffer_sizes = &.{ 4, 4, 4 }, .initial_uploads = &.{} });
    defer a.deinit(std.testing.allocator);
    var b = try ProgramStencil.initProgram(std.testing.allocator, .{ .ops = &b_ops, .n_buffers = 4, .buffer_sizes = &.{ 4, 4, 4, 4 }, .initial_uploads = &.{} });
    defer b.deinit(std.testing.allocator);

    try std.testing.expect(a.inspect().command_shape.command_count > 0);
    try std.testing.expect(b.inspect().command_shape.command_count > 0);
    try std.testing.expect(a.inspect().command_shape.command_stencil_hash != b.inspect().command_shape.command_stencil_hash);
}

test "program stencil rejects invalid runtime windows" {
    var attention = testAttentionOp();
    attention.attention.patch_seq_kv = true;
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 8,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        attention,
    };

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 4, 32, 20, 20, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(.{
        .position = std.math.maxInt(u32),
        .len = 1,
    }));
    try std.testing.expectEqual(@as(u32, 8), program_stencil.ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(@as(u32, 1), program_stencil.ops[1].attention.seq_kv);
}

test "program stencil derives cache-write holes from slice assign stride" {
    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 1,
        .dst_base_offset = 4,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 2,
    } }};

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 8 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(1, 1)));
    try std.testing.expectEqual(@as(u32, 6), program_stencil.ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(@as(u32, 1), program_stencil.runtimePatchShape().runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(@as(u32, 0), program_stencil.runtimePatchShape().runtime_patch_attention_seq_kv_holes);
}

test "program stencil keeps static attention length without patch flag" {
    const ops = [_]backend_mod.DeviceOp{testAttentionOp()};

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 1, 4, 4, 4, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.unchanged, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(5, 3)));
    try std.testing.expectEqual(@as(u32, 1), program_stencil.ops[0].attention.seq_kv);
}

test "program stencil owns fused elementwise step payloads" {
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 3 },
    };
    const ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &steps,
        .n = 4,
        .dst = 1,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &.{ 4, 4, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    const runtime_steps = program_stencil.ops[0].fused_elementwise.steps;
    try std.testing.expect(runtime_steps.ptr != steps[0..].ptr);
    try std.testing.expectEqual(steps[0], runtime_steps[0]);
}

test "program stencil patch owned ops after source ops mutate" {
    var attention = testAttentionOp();
    attention.attention.patch_seq_kv = true;
    var ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 1,
            .cols = 1,
            .dst_base_offset = 4,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 2,
        } },
        attention,
    };

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 1, 32, 32, 32, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    ops[0] = .{ .elementwise = .{ .op = .add, .dst = 0, .src0 = 0, .src1 = 0, .n = 1 } };
    ops[1] = .{ .reduce = .{ .op = .sum, .dst = 0, .src = 0, .n_out = 1, .reduce_size = 1 } };

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(5, 3)));
    try std.testing.expectEqual(@as(u32, 14), program_stencil.ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(@as(u32, 8), program_stencil.ops[1].attention.seq_kv);
}

test "program stencil refuses cache patches outside compiled buffers" {
    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 2,
    } }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &.{ 2, 8 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);
    const inspection = program_stencil.inspect();
    try std.testing.expectEqual(@as(?u32, 3), inspection.runtime_patch_envelope.max_cache_write_pos);
    try std.testing.expectEqual(@as(?u32, std.math.maxInt(u32)), inspection.runtime_patch_envelope.max_attention_seq_kv);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(3, 1)));
    try std.testing.expectEqual(@as(u32, 6), program_stencil.ops[0].slice_assign.dst_offset);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(4, 1)));
    try std.testing.expectEqual(@as(u32, 6), program_stencil.ops[0].slice_assign.dst_offset);
}

test "program stencil rejects cache patches without buffer bounds" {
    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
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
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 0, .buffer_sizes = &.{}, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(0, 1)));
}

test "program stencil rejects cache patches beyond u32 offsets" {
    const near_u32_limit: u32 = std.math.maxInt(u32) - 1;
    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 1,
        .dst_base_offset = near_u32_limit,
        .dst_offset = near_u32_limit,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 2,
    } }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &.{ 1, @as(usize, std.math.maxInt(u32)) + 2 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(1, 1)));
    try std.testing.expectEqual(near_u32_limit, program_stencil.ops[0].slice_assign.dst_offset);
}

test "program stencil refuses attention patches outside compiled buffers" {
    var attention = testAttentionOp();
    attention.attention.patch_seq_kv = true;
    const ops = [_]backend_mod.DeviceOp{attention};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 1, 4, 8, 8, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(0, 2)));
    try std.testing.expectEqual(@as(u32, 2), program_stencil.ops[0].attention.seq_kv);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(0, 3)));
    try std.testing.expectEqual(@as(u32, 2), program_stencil.ops[0].attention.seq_kv);
}

test "program stencil bounds attention patches by mask storage" {
    var attention = testAttentionOp();
    attention.attention.patch_seq_kv = true;
    attention.attention.has_mask = true;
    const ops = [_]backend_mod.DeviceOp{attention};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 2, 4, 20, 20, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(std.testing.allocator, program);
    defer program_stencil.deinit(std.testing.allocator);
    const inspection = program_stencil.inspect();
    try std.testing.expectEqual(@as(?u32, std.math.maxInt(u32)), inspection.runtime_patch_envelope.max_cache_write_pos);
    try std.testing.expectEqual(@as(?u32, 2), inspection.runtime_patch_envelope.max_attention_seq_kv);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(0, 2)));
    try std.testing.expectEqual(@as(u32, 2), program_stencil.ops[0].attention.seq_kv);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, program_stencil.patchRuntimeWindow(try backend_mod.RuntimeWindow.init(0, 3)));
    try std.testing.expectEqual(@as(u32, 2), program_stencil.ops[0].attention.seq_kv);
}

fn initProgramStencilWithFusedPayloads(alloc: std.mem.Allocator) !void {
    const steps_a = [_]backend_mod.FusedEwStep{
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 3 },
    };
    const steps_b = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = true, .secondary_buf = 4, .secondary_offset = 5 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .fused_elementwise = .{ .steps = &steps_a, .n = 4, .dst = 1, .src = 0, .dst_offset = 0, .src_offset = 0 } },
        .{ .fused_elementwise = .{ .steps = &steps_b, .n = 4, .dst = 3, .src = 1, .dst_offset = 0, .src_offset = 0 } },
    };

    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &.{ 4, 4, 4, 4, 4 }, .initial_uploads = &.{} };
    var program_stencil = try ProgramStencil.initProgram(alloc, program);
    defer program_stencil.deinit(alloc);
}

test "program stencil cleans up partial fused payload copies" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        initProgramStencilWithFusedPayloads,
        .{},
    );
}

/// Broad, model-agnostic operation families used by backend schedulers.
/// These are intentionally coarser than DeviceOp tags: a backend can reason
/// about "row kernels" or "movement kernels" without knowing which model
/// produced the program.
const KernelFamily = enum {
    elementwise,
    fused_elementwise,
    row,
    reduce,
    movement,
    matmul,
    qmatvec,
    qmatmul,
    rope,
    attention,
};

const n_kernel_families = @typeInfo(KernelFamily).@"enum".fields.len;
comptime {
    if (n_kernel_families > 64) @compileError("KernelFamilyMask stores families in u64");
}

const KernelFamilyMask = struct {
    bits: u64 = 0,

    const empty: KernelFamilyMask = .{};

    fn init(comptime families: []const KernelFamily) KernelFamilyMask {
        var mask: KernelFamilyMask = .empty;
        inline for (families) |family| {
            mask = mask.with(family);
        }
        return mask;
    }

    fn with(self: KernelFamilyMask, family: KernelFamily) KernelFamilyMask {
        return .{ .bits = self.bits | bit(family) };
    }

    fn contains(self: KernelFamilyMask, family: KernelFamily) bool {
        return (self.bits & bit(family)) != 0;
    }

    fn bit(family: KernelFamily) u64 {
        return @as(u64, 1) << @intCast(@intFromEnum(family));
    }
};

/// Whether a scheduled region is expected to run in the backend's native
/// execution path or through the backend's semantic fallback.
const ExecutionClass = enum {
    backend,
    fallback,
};

/// Native kernel families a backend can lower directly.
/// This is separate from backend.Capabilities.supportsOp(): a backend may
/// support a DeviceProgram by falling back for some ops.
const KernelSupport = struct {
    elementwise: bool = false,
    fused_elementwise: bool = false,
    row: bool = false,
    reduce: bool = false,
    movement: bool = false,
    matmul: bool = false,
    qmatvec: bool = false,
    qmatmul: bool = false,
    rope: bool = false,
    attention: bool = false,

    fn supports(self: KernelSupport, family: KernelFamily) bool {
        return switch (family) {
            .elementwise => self.elementwise,
            .fused_elementwise => self.fused_elementwise,
            .row => self.row,
            .reduce => self.reduce,
            .movement => self.movement,
            .matmul => self.matmul,
            .qmatvec => self.qmatvec,
            .qmatmul => self.qmatmul,
            .rope => self.rope,
            .attention => self.attention,
        };
    }
};

/// Pure scheduling policy. It has no backend state and no model knowledge:
/// given capabilities, native kernel availability, and dispatch thresholds,
/// the same DeviceProgram always maps to the same KernelItems.
pub const SchedulePolicy = struct {
    capabilities: backend_mod.Capabilities,
    native_kernels: KernelSupport = .{},
    fine_grained: bool = false,
    min_backend_matmul_m: u32 = 16,
    min_backend_qmatmul_m: u32 = 16,
};

/// A contiguous DeviceOp range with the same broad family and execution class.
pub const KernelItem = struct {
    family: KernelFamily,
    execution: ExecutionClass,
    start: u32,
    len: u32,
};

const KernelRegion = struct {
    start_item: u32,
    item_count: u32,
    op_start: u32,
    op_count: u32,
    anchor_count: u32,
};

const invalid_pattern_index = std.math.maxInt(u32);

const PatternRegion = struct {
    pattern_index: u32,
    region: KernelRegion,
};

const ScheduleUnitKind = enum {
    item,
    pattern_region,
};

/// A logical execution unit over a KernelItem schedule. Backends can start with
/// item units, then replace selected ranges with pattern regions as they grow
/// fused lowerings. This stays model-agnostic: a pattern is just a family
/// sequence with a backend-owned lowering.
pub const ScheduleUnit = struct {
    kind: ScheduleUnitKind,
    pattern_index: u32 = invalid_pattern_index,
    start_item: u32,
    item_count: u32,
    op_start: u32,
    op_count: u32,
};

const RegionExecutionSummary = struct {
    units: u32 = 0,
    backend_units: u32 = 0,
    fallback_units: u32 = 0,
    ops: u32 = 0,
    backend_ops: u32 = 0,
    fallback_ops: u32 = 0,
    backend_islands: u32 = 0,
    max_backend_island_units: u32 = 0,
    max_backend_island_ops: u32 = 0,
    execution_transitions: u32 = 0,
};

/// Describes reusable anchored regions over a KernelItem schedule. This is the
/// pure planning layer for future fusion: choose anchor families (for example
/// qmatvec) and families allowed to travel with them, then emit contiguous
/// candidate regions without knowing which model produced the ops.
pub const RegionPolicy = struct {
    anchor_families: KernelFamilyMask,
    member_families: KernelFamilyMask,

    pub fn qmatvecCluster() RegionPolicy {
        return .{
            .anchor_families = KernelFamilyMask.init(&.{.qmatvec}),
            .member_families = KernelFamilyMask.init(&.{
                .elementwise,
                .fused_elementwise,
                .row,
                .movement,
                .matmul,
                .qmatvec,
                .rope,
                .attention,
            }),
        };
    }

    pub fn qmatmulCluster() RegionPolicy {
        return .{
            .anchor_families = KernelFamilyMask.init(&.{.qmatmul}),
            .member_families = KernelFamilyMask.init(&.{
                .elementwise,
                .fused_elementwise,
                .row,
                .reduce,
                .movement,
                .matmul,
                .qmatmul,
                .rope,
                .attention,
            }),
        };
    }

    pub fn matmulCluster() RegionPolicy {
        return .{
            .anchor_families = KernelFamilyMask.init(&.{.matmul}),
            .member_families = KernelFamilyMask.init(&.{
                .elementwise,
                .fused_elementwise,
                .row,
                .reduce,
                .movement,
                .matmul,
                .rope,
                .attention,
            }),
        };
    }
};

/// A backend-owned lowering target over a KernelItem schedule.
///
/// StagePolicy keeps the central idea tiny: a reusable anchored region with a
/// pattern id. Model code does not know about it, and backend code can
/// progressively replace the conservative per-op walk with a fused lowering.
pub const StagePolicy = struct {
    pattern_index: u32,
    region_policy: RegionPolicy,
    anchors_per_stage: u32,
    min_items_per_stage: u32 = 1,
    min_ops_per_stage: u32 = 1,

    pub fn anchored(
        pattern_index: u32,
        region_policy: RegionPolicy,
        anchors_per_stage: u32,
    ) StagePolicy {
        return .{
            .pattern_index = pattern_index,
            .region_policy = region_policy,
            .anchors_per_stage = anchors_per_stage,
        };
    }
};

pub const ProgramCommandKind = enum {
    op,
    row_chain,
    rope_chain,
    rope_batch,
    rope_store_group,
    movement_batch,
    movement_group,
    attention_chain,
    attention_store_chain,
    attention_store_group,
    rope_attention_store_chain,
    rope_attention_store_group,
    attention_group,
    elementwise_batch,
    repeat_fused_elementwise_chain,
    projection_pair_elementwise_chain,
    projection_pair_fused_elementwise_chain,
    dense_projection_pair_fused_elementwise_chain,
    projection_row_chain,
    dense_projection_row_chain,
    dense_projection_chain,
    projection_chain,
    projection_group,
    dense_projection_cache_group,
    projection_cache_group,

    fn shape(self: ProgramCommandKind) ProgramCommandShape {
        return switch (self) {
            .op,
            .row_chain,
            .rope_chain,
            .rope_batch,
            .movement_batch,
            .repeat_fused_elementwise_chain,
            .projection_pair_fused_elementwise_chain,
            .dense_projection_pair_fused_elementwise_chain,
            => .{},

            .projection_pair_elementwise_chain,
            => .{
                .coverage = .anchor_sidecars,
                .advance = .explicit_indices,
            },

            .projection_row_chain,
            .dense_projection_row_chain,
            => .{
                .coverage = .anchor_sidecars,
                .sidecars = .flat,
                .advance = .explicit_indices,
            },

            .dense_projection_chain,
            => .{
                .coverage = .anchor_sidecars,
                .sidecars = .flat,
            },

            .projection_chain,
            .attention_chain,
            => .{ .coverage = .anchor_sidecars },

            .projection_group,
            .rope_store_group,
            .attention_store_chain,
            .attention_store_group,
            .rope_attention_store_chain,
            .rope_attention_store_group,
            => .{
                .coverage = .anchor_sidecars,
                .advance = .explicit_indices,
            },

            .dense_projection_cache_group,
            .projection_cache_group,
            => .{
                .coverage = .anchor_sidecars,
                .sidecars = .flat,
                .advance = .explicit_indices,
            },

            .movement_group,
            .attention_group,
            .elementwise_batch,
            => .{
                .coverage = .anchors_only,
                .advance = .explicit_indices,
            },
        };
    }
};

pub const n_program_command_kinds = @typeInfo(ProgramCommandKind).@"enum".fields.len;

pub const program_command_kind_names = blk: {
    var names: [n_program_command_kinds][]const u8 = undefined;
    for (@typeInfo(ProgramCommandKind).@"enum".fields, 0..) |field, i| names[i] = field.name;
    break :blk names;
};

const ProgramCommandCoverage = enum {
    contiguous,
    anchor_sidecars,
    anchors_only,
};

const ProgramCommandSidecarLayout = enum {
    anchor_aligned,
    flat,
};

const ProgramCommandAdvance = enum {
    contiguous,
    explicit_indices,
};

const ProgramCommandShape = struct {
    coverage: ProgramCommandCoverage = .contiguous,
    sidecars: ProgramCommandSidecarLayout = .anchor_aligned,
    advance: ProgramCommandAdvance = .contiguous,
};

const ProjectionGroupKind = enum {
    qmatvec,
    qmatmul,
};

/// Pure policy for projection batching. Backends still decide whether they
/// have a native kernel for the command; this only answers "is it legal to
/// batch these independent projections and carry simple side effects?"
const ProjectionGroupPolicy = struct {
    kind: ProjectionGroupKind,
    max_anchors: u32 = 4,
    carry_slice_sidecars: bool = true,

    fn decodeQMatvec(max_anchors: u32) ProjectionGroupPolicy {
        return .{ .kind = .qmatvec, .max_anchors = max_anchors, .carry_slice_sidecars = true };
    }

    fn prefillQMatmul(max_anchors: u32) ProjectionGroupPolicy {
        return .{ .kind = .qmatmul, .max_anchors = max_anchors };
    }
};

pub const CommandStreamPolicy = struct {
    row_rope_chains: bool = true,
    qmatvec_group_size: u32 = 4,
    qmatmul_group_size: u32 = 4,
    dense_matvec_group_size: u32 = 4,
    qmatmul_sidecars: bool = true,
    qmatmul_cache_sidecars_per_anchor: u32 = 8,
    max_rope_batch: u32 = 16,
    max_movement_batch: u32 = 16,
    max_attention_batch: u32 = 16,
    max_attention_store_batch: u32 = 4,
    max_rope_attention_store_batch: u32 = 16,
    max_elementwise_batch: u32 = 8,
    fuse_repeat_fused_elementwise: bool = true,
    fuse_projection_chain: bool = true,
    fuse_projection_row_chain: bool = false,
    fuse_projection_row_chain_qmatvec: bool = false,
    fuse_projection_row_chain_single_dispatch: bool = false,
    fuse_dense_projection_row_chain: bool = false,
    min_projection_row_chain_rows: u32 = 8,

    pub fn default() CommandStreamPolicy {
        return .{};
    }

    pub fn promptProjectionRowChainCandidate() CommandStreamPolicy {
        return promptProjectionRowChainSingleDispatchCandidate();
    }

    pub fn promptProjectionRowChainCommand() CommandStreamPolicy {
        var policy = CommandStreamPolicy.default();
        policy.fuse_projection_row_chain = true;
        policy.fuse_projection_row_chain_qmatvec = false;
        policy.fuse_projection_row_chain_single_dispatch = false;
        policy.min_projection_row_chain_rows = 8;
        return policy;
    }

    pub fn promptProjectionRowChainSingleDispatchCandidate() CommandStreamPolicy {
        var policy = CommandStreamPolicy.default();
        policy.fuse_projection_row_chain = true;
        policy.fuse_projection_row_chain_qmatvec = false;
        policy.fuse_projection_row_chain_single_dispatch = true;
        policy.min_projection_row_chain_rows = 8;
        return policy;
    }

    fn grouped(qmatvec_group_size: u32, qmatmul_group_size: u32) CommandStreamPolicy {
        var policy = CommandStreamPolicy.default();
        policy.qmatvec_group_size = qmatvec_group_size;
        policy.qmatmul_group_size = qmatmul_group_size;
        policy.dense_matvec_group_size = qmatvec_group_size;
        return policy;
    }

    fn projectionPolicyFor(self: CommandStreamPolicy, q: anytype) ?ProjectionGroupPolicy {
        if (q.M == 1) {
            if (self.qmatvec_group_size < 2) return null;
            return ProjectionGroupPolicy.decodeQMatvec(self.qmatvec_group_size);
        }
        if (self.qmatmul_group_size < 2) return null;
        var policy = ProjectionGroupPolicy.prefillQMatmul(self.qmatmul_group_size);
        policy.carry_slice_sidecars = self.qmatmul_sidecars;
        return policy;
    }
};

pub const ProgramCommand = struct {
    const invalid_sidecar_slot = std.math.maxInt(u8);

    kind: ProgramCommandKind,
    op_start: u32,
    op_count: u32,
    projection_kind: ProjectionGroupKind = .qmatmul,
    anchor_count: u32 = 0,
    sidecar_count: u32 = 0,
    indices: [max_projection_group_anchors]usize = [_]usize{0} ** max_projection_group_anchors,
    sidecar_indices: [max_projection_group_anchors]?usize = [_]?usize{null} ** max_projection_group_anchors,
    sidecar_slots: [max_projection_group_anchors]u8 = [_]u8{invalid_sidecar_slot} ** max_projection_group_anchors,

    fn op(start: usize) ProgramCommand {
        return .{
            .kind = .op,
            .op_start = @intCast(start),
            .op_count = 1,
        };
    }

    fn fromProjectionSelection(selection: ProjectionGroupSelection) ProgramCommand {
        var command = ProgramCommand{
            .kind = .projection_group,
            .op_start = @intCast(selection.start_op),
            .op_count = @intCast(selection.end_op - selection.start_op + 1),
            .projection_kind = selection.kind,
            .anchor_count = @intCast(selection.anchor_count),
            .sidecar_count = @intCast(selection.sidecar_count),
        };
        for (selection.anchorIndices(), 0..) |idx, slot| command.indices[slot] = idx;
        for (selection.sidecarIndices(), 0..) |idx, slot| command.sidecar_indices[slot] = idx;
        return command;
    }

    fn contiguous(kind: ProgramCommandKind, start: usize, count: usize) ProgramCommand {
        return .{
            .kind = kind,
            .op_start = @intCast(start),
            .op_count = @intCast(count),
        };
    }

    fn shape(self: ProgramCommand) ProgramCommandShape {
        return self.kind.shape();
    }

    fn coveredOpCount(self: ProgramCommand) u32 {
        return switch (self.shape().coverage) {
            .contiguous => self.op_count,
            .anchor_sidecars => self.anchor_count + self.sidecar_count,
            .anchors_only => self.anchor_count,
        };
    }

    fn advanceCount(self: ProgramCommand) u32 {
        return switch (self.shape().advance) {
            .contiguous => self.op_count,
            .explicit_indices => 1,
        };
    }

    fn hasExplicitCoverage(self: ProgramCommand) bool {
        return self.shape().coverage != .contiguous;
    }

    fn explicitIndexSet(self: *const ProgramCommand) CommandIndexSet {
        var set = CommandIndexSet{};
        for (self.anchorIndices()) |idx| {
            _ = set.append(idx);
        }
        for (self.carriedSidecarIndices()) |maybe_idx| {
            if (maybe_idx) |idx| _ = set.append(idx);
        }
        return set;
    }

    fn sortedExplicitIndexSet(self: *const ProgramCommand) CommandIndexSet {
        var set = self.explicitIndexSet();
        set.sort();
        return set;
    }

    pub fn coveredIndexIterator(self: *const ProgramCommand) CommandIndexIterator {
        return CommandIndexIterator.init(self);
    }

    pub fn anchorIndices(self: *const ProgramCommand) []const usize {
        return self.indices[0..self.anchor_count];
    }

    pub fn sidecarIndices(self: *const ProgramCommand) []const ?usize {
        return self.sidecar_indices[0..self.anchor_count];
    }

    pub fn sidecarAnchorSlot(self: *const ProgramCommand, sidecar_index: usize) ?usize {
        if (sidecar_index >= self.sidecar_count) return null;
        const slot = self.sidecar_slots[sidecar_index];
        if (slot == invalid_sidecar_slot or slot >= self.anchor_count) return null;
        return slot;
    }

    fn flatSidecarIndices(self: *const ProgramCommand) []const ?usize {
        return self.sidecar_indices[0..self.sidecar_count];
    }

    fn carriedSidecarIndices(self: *const ProgramCommand) []const ?usize {
        return switch (self.shape().sidecars) {
            .anchor_aligned => self.sidecarIndices(),
            .flat => self.flatSidecarIndices(),
        };
    }
};

pub const ProgramCommandStreamShape = struct {
    command_count: u32 = 0,
    covered_ops: u32 = 0,
    estimated_saved_dispatches: u32 = 0,
    row_chains: u32 = 0,
    projection_row_chains: u32 = 0,
    dense_projection_row_chains: u32 = 0,
    projection_chains: u32 = 0,
    dense_projection_chains: u32 = 0,
    quantized_projection_chains: u32 = 0,
    projection_chain_sidecars: u32 = 0,
    projection_chain_row_chain_frontiers: u32 = 0,
    projection_groups: u32 = 0,
    projection_anchors: u32 = 0,
    projection_sidecars: u32 = 0,
    projection_cache_groups: u32 = 0,
    projection_cache_anchors: u32 = 0,
    projection_cache_sidecars: u32 = 0,
    max_projection_span_ops: u32 = 0,
    command_kind_counts: [n_program_command_kinds]u32 = [_]u32{0} ** n_program_command_kinds,
    command_stencil_hash: u64 = 0,

    fn init(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp, policy: CommandStreamPolicy) !ProgramCommandStreamShape {
        const commands = try buildProgramCommands(alloc, ops, policy);
        defer if (commands.len > 0) alloc.free(commands);
        return fromCommands(commands);
    }

    pub fn fromCommands(commands: []const ProgramCommand) !ProgramCommandStreamShape {
        const summary = summarizeProgramCommands(commands);
        var shape = ProgramCommandStreamShape{
            .command_count = std.math.cast(u32, commands.len) orelse return error.UnsupportedDeviceOp,
            .covered_ops = summary.covered_ops,
            .estimated_saved_dispatches = summary.estimated_saved_dispatches,
            .row_chains = summary.row_chains,
            .projection_row_chains = summary.projection_row_chains,
            .dense_projection_row_chains = summary.dense_projection_row_chains,
            .projection_chains = summary.projection_chains,
            .dense_projection_chains = summary.dense_projection_chains,
            .quantized_projection_chains = summary.quantized_projection_chains,
            .projection_chain_sidecars = summary.projection_chain_sidecars,
            .projection_chain_row_chain_frontiers = summary.projection_chain_row_chain_frontiers,
            .projection_groups = summary.projection_groups,
            .projection_anchors = summary.projection_anchors,
            .projection_sidecars = summary.projection_sidecars,
            .projection_cache_groups = summary.projection_cache_groups,
            .projection_cache_anchors = summary.projection_cache_anchors,
            .projection_cache_sidecars = summary.projection_cache_sidecars,
            .max_projection_span_ops = summary.max_projection_span_ops,
            .command_stencil_hash = if (commands.len > 0) 14695981039346656037 else 0,
        };
        var h = RuntimeStencilHasher{ .state = shape.command_stencil_hash };
        for (commands) |command| {
            shape.command_kind_counts[@intFromEnum(command.kind)] =
                std.math.add(u32, shape.command_kind_counts[@intFromEnum(command.kind)], 1) catch return error.UnsupportedDeviceOp;
            addProgramCommandStencil(&h, command);
        }
        shape.command_stencil_hash = if (commands.len > 0) h.state else 0;
        return shape;
    }

    pub fn merge(self: ProgramCommandStreamShape, other: ProgramCommandStreamShape) ProgramCommandStreamShape {
        if (self.command_count == 0 and self.command_stencil_hash == 0) return other;
        if (other.command_count == 0 and other.command_stencil_hash == 0) return self;

        var merged = ProgramCommandStreamShape{
            .command_count = @max(self.command_count, other.command_count),
            .covered_ops = @max(self.covered_ops, other.covered_ops),
            .estimated_saved_dispatches = @max(self.estimated_saved_dispatches, other.estimated_saved_dispatches),
            .row_chains = @max(self.row_chains, other.row_chains),
            .projection_row_chains = @max(self.projection_row_chains, other.projection_row_chains),
            .dense_projection_row_chains = @max(self.dense_projection_row_chains, other.dense_projection_row_chains),
            .projection_chains = @max(self.projection_chains, other.projection_chains),
            .dense_projection_chains = @max(self.dense_projection_chains, other.dense_projection_chains),
            .quantized_projection_chains = @max(self.quantized_projection_chains, other.quantized_projection_chains),
            .projection_chain_sidecars = @max(self.projection_chain_sidecars, other.projection_chain_sidecars),
            .projection_chain_row_chain_frontiers = @max(self.projection_chain_row_chain_frontiers, other.projection_chain_row_chain_frontiers),
            .projection_groups = @max(self.projection_groups, other.projection_groups),
            .projection_anchors = @max(self.projection_anchors, other.projection_anchors),
            .projection_sidecars = @max(self.projection_sidecars, other.projection_sidecars),
            .projection_cache_groups = @max(self.projection_cache_groups, other.projection_cache_groups),
            .projection_cache_anchors = @max(self.projection_cache_anchors, other.projection_cache_anchors),
            .projection_cache_sidecars = @max(self.projection_cache_sidecars, other.projection_cache_sidecars),
            .max_projection_span_ops = @max(self.max_projection_span_ops, other.max_projection_span_ops),
            .command_stencil_hash = if (self.command_stencil_hash == other.command_stencil_hash) self.command_stencil_hash else 0,
        };
        for (&merged.command_kind_counts, self.command_kind_counts, other.command_kind_counts) |*dst, a, b| {
            dst.* = @max(a, b);
        }
        return merged;
    }

    pub fn categoryCounts(self: ProgramCommandStreamShape) ProgramCommandCategoryCounts {
        var counts = ProgramCommandCategoryCounts{};
        for (self.command_kind_counts, 0..) |count, index| {
            const kind: ProgramCommandKind = @enumFromInt(index);
            const value: u64 = count;
            switch (kind) {
                .op => counts.op += value,
                .row_chain => counts.row += value,
                .projection_pair_fused_elementwise_chain,
                .projection_pair_elementwise_chain,
                .dense_projection_pair_fused_elementwise_chain,
                .projection_row_chain,
                .dense_projection_row_chain,
                .dense_projection_chain,
                .projection_chain,
                .projection_group,
                .dense_projection_cache_group,
                .projection_cache_group,
                => counts.projection += value,
                .attention_chain,
                .attention_store_chain,
                .attention_store_group,
                .rope_attention_store_chain,
                .rope_attention_store_group,
                .attention_group,
                => counts.attention += value,
                .movement_batch,
                .movement_group,
                => counts.movement += value,
                .elementwise_batch,
                .repeat_fused_elementwise_chain,
                => counts.elementwise += value,
                .rope_chain,
                .rope_batch,
                .rope_store_group,
                => counts.rope += value,
            }
        }
        return counts;
    }
};

pub const ProjectionRowChainFrontierDebug = struct {
    reason: Reason,
    command_count: u32 = 0,
    first_command_kind: ProgramCommandKind = .op,
    first_projection_command_index: u32 = 0,
    first_projection_op_start: u32 = 0,
    first_projection_op_count: u32 = 0,
    first_projection_sidecars: u32 = 0,
    first_projection_op_tags: [6]DeviceOpTag = .{.elementwise} ** 6,
    first_projection_prev_kind: ProgramCommandKind = .op,
    first_projection_kind: ProgramCommandKind = .op,
    first_projection_next_kind: ProgramCommandKind = .op,
    projection_command_index: u32 = 0,
    projection_op_start: u32 = 0,
    row_op_start: u32 = 0,
    q_m: u32 = 0,
    q_n: u32 = 0,
    q_dst: u16 = 0,
    elementwise_dst: u16 = 0,
    elementwise_src0: u16 = 0,
    elementwise_src1: u16 = 0,
    rms_src: u16 = 0,
    rms_dst: u16 = 0,

    pub const Reason = enum {
        no_frontier,
        already_projection_row_chain,
        policy_disabled,
        qmatvec_policy_disabled,
        qmatmul_too_few_rows,
        malformed_frontier,
        sidecar_incompatible,
        rms_source_mismatch,
        scale_chain_mismatch,
        projection_primary_external_users,
        elementwise_external_read,
        scale_chain_external_users,
        would_fuse,
    };
};

pub fn firstProjectionRowChainFrontierDebug(
    ops: []const backend_mod.DeviceOp,
    commands: []const ProgramCommand,
    policy: CommandStreamPolicy,
) ProjectionRowChainFrontierDebug {
    var base = ProjectionRowChainFrontierDebug{
        .reason = .no_frontier,
        .command_count = @intCast(commands.len),
        .first_command_kind = if (commands.len > 0) commands[0].kind else .op,
    };
    for (commands, 0..) |command, index| {
        if (command.kind == .projection_chain or command.kind == .dense_projection_chain) {
            base.first_projection_command_index = @intCast(index);
            base.first_projection_op_start = command.op_start;
            base.first_projection_op_count = command.op_count;
            base.first_projection_sidecars = command.sidecar_count;
            for (base.first_projection_op_tags[0..], 0..) |*tag, offset| {
                const op_index = @as(usize, command.op_start) + offset;
                if (op_index < ops.len) tag.* = std.meta.activeTag(ops[op_index]);
            }
            base.first_projection_prev_kind = if (index > 0) commands[index - 1].kind else .op;
            base.first_projection_kind = command.kind;
            base.first_projection_next_kind = if (index + 1 < commands.len) commands[index + 1].kind else .op;
            break;
        }
    }
    for (commands, 0..) |command, command_index| {
        if (command_index + 1 >= commands.len) continue;
        const row = commands[command_index + 1];
        if ((command.kind != .projection_chain and command.kind != .dense_projection_chain) or row.kind != .row_chain) continue;
        if (command.op_start + command.op_count != row.op_start) continue;

        var out = base;
        out.reason = .malformed_frontier;
        out.projection_command_index = @intCast(command_index);
        out.projection_op_start = command.op_start;
        out.row_op_start = row.op_start;
        const start: usize = @intCast(command.op_start);
        if (start + 4 >= ops.len) return out;
        const e = switch (ops[start + 1]) {
            .elementwise => |e| e,
            else => return out,
        };
        const rn = switch (ops[start + 2]) {
            .rmsnorm => |rn| rn,
            else => return out,
        };
        if (command.kind == .projection_chain) {
            const q = switch (ops[start]) {
                .qmatmul => |q| q,
                else => return out,
            };
            out.q_m = q.M;
            out.q_n = q.N;
            out.q_dst = q.dst;
        } else {
            const m = switch (ops[start]) {
                .matmul => |m| m,
                else => return out,
            };
            out.q_m = @intCast(m.geom.M);
            out.q_n = @intCast(m.geom.N);
            out.q_dst = m.dst;
        }
        out.elementwise_dst = e.dst;
        out.elementwise_src0 = e.src0;
        out.elementwise_src1 = e.src1;
        out.rms_src = rn.src;
        out.rms_dst = rn.dst;

        if (!policy.fuse_projection_row_chain) {
            out.reason = .policy_disabled;
            return out;
        }
        if (command.kind == .projection_chain) {
            const q = ops[start].qmatmul;
            if (q.M == 1) {
                if (!policy.fuse_projection_row_chain_qmatvec) {
                    out.reason = .qmatvec_policy_disabled;
                    return out;
                }
                if (!qmatvecElementwiseSidecarCompatible(q, e)) {
                    out.reason = .sidecar_incompatible;
                    return out;
                }
            } else {
                if (q.M < policy.min_projection_row_chain_rows) {
                    out.reason = .qmatmul_too_few_rows;
                    return out;
                }
                if (!qmatmulElementwiseSidecarCompatible(q, e)) {
                    out.reason = .sidecar_incompatible;
                    return out;
                }
            }
        } else {
            const m = ops[start].matmul;
            if (!matmulElementwiseSidecarCompatible(m, e)) {
                out.reason = .sidecar_incompatible;
                return out;
            }
        }
        if (rn.src != e.dst or rn.src_offset != e.dst_offset) {
            out.reason = .rms_source_mismatch;
            return out;
        }
        if (!isRmsnormScaleChain(ops[start + 2], ops[start + 3], ops[start + 4])) {
            out.reason = .scale_chain_mismatch;
            return out;
        }
        const projection_sidecars = [_]?usize{start + 1};
        const has_projection_users = if (command.kind == .projection_chain)
            projectionPrimaryOutputHasExternalUsersExcept(ops, start, projection_sidecars[0..])
        else
            matmulPrimaryOutputHasExternalUsersExcept(ops, start, projection_sidecars[0..]);
        if (has_projection_users) {
            out.reason = .projection_primary_external_users;
            return out;
        }
        if (command.kind == .projection_chain and spanHasExternalReadAfter(ops, start + 1, start + 2, start + 5, bufferSpan(e.dst, e.dst_offset, e.n))) {
            out.reason = .elementwise_external_read;
            return out;
        }
        if (rmsnormScaleChainHasExternalUsers(ops, start + 2)) {
            out.reason = .scale_chain_external_users;
            return out;
        }
        out.reason = .would_fuse;
        return out;
    }

    for (commands) |command| {
        if (command.kind == .projection_row_chain or command.kind == .dense_projection_row_chain) {
            base.reason = .already_projection_row_chain;
            return base;
        }
    }

    return base;
}

pub const ProjectionElementwiseChainDebug = struct {
    reason: Reason,
    command_count: u32 = 0,
    command_index: u32 = 0,
    op_start: u32 = 0,
    op_count: u32 = 0,
    prev_kind: ProgramCommandKind = .op,
    kind: ProgramCommandKind = .op,
    next_kind: ProgramCommandKind = .op,
    q_m: u32 = 0,
    q_n: u32 = 0,
    q_dst: u16 = 0,
    q_input: u16 = 0,
    weight_idx: u32 = 0,
    elementwise_op: backend_mod.Op = .add,
    elementwise_dst: u16 = 0,
    elementwise_src0: u16 = 0,
    elementwise_src1: u16 = 0,
    elementwise_n: u32 = 0,
    primary_has_external_users: bool = false,

    pub const Reason = enum {
        no_projection_elementwise_chain,
        dense_projection_elementwise_chain,
        quantized_projection_elementwise_chain,
        malformed_chain,
    };
};

pub fn firstProjectionElementwiseChainDebug(
    ops: []const backend_mod.DeviceOp,
    commands: []const ProgramCommand,
) ProjectionElementwiseChainDebug {
    var out = ProjectionElementwiseChainDebug{
        .reason = .no_projection_elementwise_chain,
        .command_count = @intCast(commands.len),
    };
    for (commands, 0..) |command, command_index| {
        if (command.kind != .projection_chain and command.kind != .dense_projection_chain) continue;
        out.reason = .malformed_chain;
        out.command_index = @intCast(command_index);
        out.op_start = command.op_start;
        out.op_count = command.op_count;
        out.prev_kind = if (command_index > 0) commands[command_index - 1].kind else .op;
        out.kind = command.kind;
        out.next_kind = if (command_index + 1 < commands.len) commands[command_index + 1].kind else .op;
        if (command.anchor_count != 1 or command.sidecar_count != 1) return out;
        const q_idx = command.indices[0];
        const sidecar_idx = command.sidecar_indices[0] orelse return out;
        if (q_idx >= ops.len or sidecar_idx >= ops.len) return out;
        const e = switch (ops[sidecar_idx]) {
            .elementwise => |e| e,
            else => continue,
        };
        if (command.kind == .projection_chain) {
            const q = switch (ops[q_idx]) {
                .qmatmul => |q| q,
                else => return out,
            };
            out.reason = .quantized_projection_elementwise_chain;
            out.q_m = q.M;
            out.q_n = q.N;
            out.q_dst = q.dst;
            out.q_input = q.input;
            out.weight_idx = q.weight_idx;
            out.primary_has_external_users = projectionPrimaryOutputHasExternalUsers(ops, q_idx, sidecar_idx);
        } else {
            const m = switch (ops[q_idx]) {
                .matmul => |m| m,
                else => return out,
            };
            out.reason = .dense_projection_elementwise_chain;
            out.q_m = @intCast(m.geom.M);
            out.q_n = @intCast(m.geom.N);
            out.q_dst = m.dst;
            out.q_input = m.a;
            out.primary_has_external_users = matmulPrimaryOutputHasExternalUsers(ops, q_idx, sidecar_idx);
        }
        out.elementwise_op = e.op;
        out.elementwise_dst = e.dst;
        out.elementwise_src0 = e.src0;
        out.elementwise_src1 = e.src1;
        out.elementwise_n = e.n;
        return out;
    }
    return out;
}

pub const ProgramCommandCategoryCounts = struct {
    op: u64 = 0,
    row: u64 = 0,
    projection: u64 = 0,
    attention: u64 = 0,
    movement: u64 = 0,
    elementwise: u64 = 0,
    rope: u64 = 0,
};

fn addProgramCommandStencil(h: *RuntimeStencilHasher, command: ProgramCommand) void {
    h.add(@intFromEnum(command.kind));
    h.add(command.op_start);
    h.add(command.op_count);
    h.add(@intFromEnum(command.projection_kind));
    h.add(command.anchor_count);
    h.add(command.sidecar_count);
    for (command.indices[0..command.anchor_count]) |idx| h.add(idx);
    for (command.carriedSidecarIndices()) |maybe_idx| h.add(maybe_idx orelse std.math.maxInt(u64));
    for (command.sidecar_slots[0..command.sidecar_count]) |slot| h.add(slot);
}

const max_command_indices = max_projection_group_anchors * 2;

const CommandIndexIterator = struct {
    mode: enum { contiguous, explicit },
    next_index: usize = 0,
    end_index: usize = 0,
    explicit: CommandIndexSet = .{},
    explicit_pos: usize = 0,

    fn init(command: *const ProgramCommand) CommandIndexIterator {
        if (command.hasExplicitCoverage()) {
            return .{
                .mode = .explicit,
                .explicit = command.sortedExplicitIndexSet(),
            };
        }

        const start: usize = @intCast(command.op_start);
        return .{
            .mode = .contiguous,
            .next_index = start,
            .end_index = start + @as(usize, command.op_count),
        };
    }

    pub fn next(self: *CommandIndexIterator) ?usize {
        return switch (self.mode) {
            .contiguous => {
                if (self.next_index >= self.end_index) return null;
                const idx = self.next_index;
                self.next_index += 1;
                return idx;
            },
            .explicit => {
                if (self.explicit_pos >= self.explicit.count) return null;
                const idx = self.explicit.indices[self.explicit_pos];
                self.explicit_pos += 1;
                return idx;
            },
        };
    }

    fn remainingCount(self: *const CommandIndexIterator) usize {
        return switch (self.mode) {
            .contiguous => self.end_index - self.next_index,
            .explicit => self.explicit.count - self.explicit_pos,
        };
    }
};

const CommandIndexSet = struct {
    indices: [max_command_indices]usize = undefined,
    count: usize = 0,

    fn append(self: *CommandIndexSet, idx: usize) bool {
        for (self.indices[0..self.count]) |existing| {
            if (existing == idx) return true;
        }
        if (self.count >= self.indices.len) return false;
        self.indices[self.count] = idx;
        self.count += 1;
        return true;
    }

    fn sort(self: *CommandIndexSet) void {
        std.mem.sort(usize, self.indices[0..self.count], {}, std.sort.asc(usize));
    }

    fn slice(self: *const CommandIndexSet) []const usize {
        return self.indices[0..self.count];
    }
};

const ProgramCommandSummary = struct {
    commands: u32 = 0,
    covered_ops: u32 = 0,
    estimated_dispatches: u32 = 0,
    estimated_saved_dispatches: u32 = 0,
    op_commands: u32 = 0,
    row_chains: u32 = 0,
    rope_chains: u32 = 0,
    rope_batches: u32 = 0,
    rope_store_groups: u32 = 0,
    rope_store_group_ops: u32 = 0,
    rope_store_group_sidecars: u32 = 0,
    movement_batches: u32 = 0,
    movement_groups: u32 = 0,
    movement_group_ops: u32 = 0,
    attention_chains: u32 = 0,
    attention_chain_sidecars: u32 = 0,
    attention_store_chains: u32 = 0,
    attention_store_chain_sidecars: u32 = 0,
    attention_store_groups: u32 = 0,
    attention_store_group_ops: u32 = 0,
    attention_store_group_sidecars: u32 = 0,
    rope_attention_store_chains: u32 = 0,
    rope_attention_store_chain_sidecars: u32 = 0,
    rope_attention_store_groups: u32 = 0,
    rope_attention_store_group_ops: u32 = 0,
    rope_attention_store_group_sidecars: u32 = 0,
    attention_groups: u32 = 0,
    attention_group_ops: u32 = 0,
    elementwise_batches: u32 = 0,
    elementwise_ops: u32 = 0,
    repeat_fused_elementwise_chains: u32 = 0,
    projection_pair_fused_elementwise_chains: u32 = 0,
    projection_row_chains: u32 = 0,
    dense_projection_row_chains: u32 = 0,
    projection_chains: u32 = 0,
    dense_projection_chains: u32 = 0,
    quantized_projection_chains: u32 = 0,
    projection_chain_sidecars: u32 = 0,
    projection_chain_row_chain_frontiers: u32 = 0,
    projection_groups: u32 = 0,
    projection_anchors: u32 = 0,
    projection_sidecars: u32 = 0,
    projection_cache_groups: u32 = 0,
    projection_cache_anchors: u32 = 0,
    projection_cache_sidecars: u32 = 0,
    max_projection_span_ops: u32 = 0,
};

const max_projection_group_anchors = 32;

const BufferSpan = struct {
    buf: u16,
    start: u64,
    end: u64,

    fn overlaps(self: BufferSpan, other: BufferSpan) bool {
        return self.buf == other.buf and self.start < other.end and other.start < self.end;
    }
};

const max_access_spans = 16;

const OpAccessSpans = struct {
    reads: [max_access_spans]BufferSpan = undefined,
    writes: [max_access_spans]BufferSpan = undefined,
    read_count: u8 = 0,
    write_count: u8 = 0,
    read_overflow: bool = false,
    write_overflow: bool = false,

    fn addRead(self: *OpAccessSpans, span: BufferSpan) void {
        if (span.start == span.end) return;
        if (self.read_count >= max_access_spans) {
            self.read_overflow = true;
            return;
        }
        self.reads[self.read_count] = span;
        self.read_count += 1;
    }

    fn addWrite(self: *OpAccessSpans, span: BufferSpan) void {
        if (span.start == span.end) return;
        if (self.write_count >= max_access_spans) {
            self.write_overflow = true;
            return;
        }
        self.writes[self.write_count] = span;
        self.write_count += 1;
    }

    fn readSpans(self: *const OpAccessSpans) []const BufferSpan {
        return self.reads[0..self.read_count];
    }

    fn writeSpans(self: *const OpAccessSpans) []const BufferSpan {
        return self.writes[0..self.write_count];
    }
};

const ProjectionGroupSelection = struct {
    kind: ProjectionGroupKind,
    start_op: usize,
    end_op: usize,
    anchor_count: usize,
    sidecar_count: usize,
    indices: [max_projection_group_anchors]usize = undefined,
    sidecar_indices: [max_projection_group_anchors]?usize = [_]?usize{null} ** max_projection_group_anchors,

    fn anchorIndices(self: *const ProjectionGroupSelection) []const usize {
        return self.indices[0..self.anchor_count];
    }

    fn sidecarIndices(self: *const ProjectionGroupSelection) []const ?usize {
        return self.sidecar_indices[0..self.anchor_count];
    }
};

fn findProjectionGroup(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: ProjectionGroupPolicy,
    used: ?[]const bool,
) ?ProjectionGroupSelection {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_anchors = @min(policy.max_anchors, max_projection_group_anchors);
    if (max_anchors < 2) return null;

    const first = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (!projectionMatchesPolicy(first, policy)) return null;

    var selection = ProjectionGroupSelection{
        .kind = policy.kind,
        .start_op = start,
        .end_op = start,
        .anchor_count = 1,
        .sidecar_count = 0,
    };
    selection.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and selection.anchor_count < max_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const q = switch (ops[scan]) {
            .qmatmul => |q| q,
            else => continue,
        };
        if (!projectionMatchesPolicy(q, policy)) continue;
        if (!canHoistProjectionTo(ops, start, scan, q)) continue;
        if (projectionConflictsSelected(ops, selection.anchorIndices(), q)) continue;
        selection.indices[selection.anchor_count] = scan;
        selection.anchor_count += 1;
    }

    if (selection.anchor_count < 2) return null;

    for (selection.anchorIndices(), 0..) |idx, slot| {
        selection.end_op = @max(selection.end_op, idx);
        if (!policy.carry_slice_sidecars) continue;
        if (idx + 1 >= ops.len) continue;
        if (used) |used_ops| {
            if (idx + 1 >= used_ops.len or used_ops[idx + 1]) continue;
        }
        const q = ops[idx].qmatmul;
        const sidecar = ops[idx + 1];
        if (!projectionSidecarMatchesPolicy(policy, q, sidecar)) continue;

        var candidate = selection;
        candidate.sidecar_indices[slot] = idx + 1;
        candidate.sidecar_count += 1;
        candidate.end_op = @max(candidate.end_op, idx + 1);
        if (!canHoistProjectionSidecarToGroup(ops, selection.start_op, idx + 1, q, sidecar, &candidate)) continue;
        selection = candidate;
    }

    return selection;
}

fn findProjectionCacheGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    const projection_policy = policy.projectionPolicyFor(ops[start].qmatmul) orelse return null;
    if (!projection_policy.carry_slice_sidecars) return null;
    const selection = findProjectionGroup(ops, start, projection_policy, used) orelse return null;

    var command = ProgramCommand{
        .kind = .projection_cache_group,
        .op_start = @intCast(selection.start_op),
        .op_count = @intCast(selection.end_op - selection.start_op + 1),
        .projection_kind = selection.kind,
        .anchor_count = @intCast(selection.anchor_count),
    };
    var per_anchor_sidecars = [_]u32{0} ** max_projection_group_anchors;
    for (selection.anchorIndices(), 0..) |idx, slot| command.indices[slot] = idx;
    for (selection.sidecarIndices(), 0..) |maybe_idx, slot| {
        const idx = maybe_idx orelse continue;
        const sa = switch (ops[idx]) {
            .slice_assign => |sa| sa,
            else => return null,
        };
        if (!appendProjectionCacheSidecar(&command, idx, selection.start_op, slot)) return null;
        per_anchor_sidecars[slot] += 1;
        if (!projectionCacheSidecarsShareSink(ops, &command, slot, sa)) return null;
    }

    const initial_sidecars = command.sidecar_count;
    const max_sidecars_per_anchor = @max(1, policy.qmatmul_cache_sidecars_per_anchor);
    var scan = ProjectionCacheSidecarScan{
        .ops = ops,
        .used = used,
        .executed = executed,
        .group_start = selection.start_op,
        .fixed_end = selection.end_op,
        .max_sidecars_per_anchor = max_sidecars_per_anchor,
        .per_anchor_sidecars = per_anchor_sidecars,
    };
    scan.collect(.qmatmul, projectionSliceSidecarCompatible, qmatvecRopeStoreSidecarCompatiblePair, &command);
    collectProjectionElementwiseSidecars(ops, used, executed, selection.start_op, &command);

    return if (command.sidecar_count > initial_sidecars) command else null;
}

fn findDenseProjectionCacheGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_anchors = @min(max_projection_group_anchors, @as(usize, @intCast(policy.dense_matvec_group_size)));
    if (max_anchors < 2) return null;
    const first = switch (ops[start]) {
        .matmul => |m| m,
        else => return null,
    };
    if (!denseProjectionAnchor(first)) return null;

    var command = ProgramCommand{
        .kind = .dense_projection_cache_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const m = switch (ops[scan]) {
            .matmul => |m| m,
            else => continue,
        };
        if (!denseProjectionAnchor(m)) continue;
        if (!denseProjectionGroupCompatible(first, m)) continue;
        if (!canHoistOpTo(ops, start, scan, .{ .matmul = m })) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .matmul = m })) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }
    if (command.anchor_count < 2) return null;

    const initial_sidecars = command.sidecar_count;
    var sidecar_scan = ProjectionCacheSidecarScan{
        .ops = ops,
        .used = used,
        .executed = executed,
        .group_start = start,
        .max_sidecars_per_anchor = MAX_DENSE_CACHE_SIDECARS_PER_ANCHOR,
    };
    sidecar_scan.collect(.matmul, denseProjectionSliceSidecarCompatible, denseMatvecRopeStoreSidecarCompatiblePair, &command);
    collectDenseProjectionElementwiseSidecars(ops, used, executed, start, &command);

    return if (command.sidecar_count > initial_sidecars) command else null;
}

const MAX_DENSE_CACHE_SIDECARS_PER_ANCHOR = 8;

fn commandAnchorSlotHasSidecar(command: *const ProgramCommand, slot: usize) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        if (command.sidecarAnchorSlot(i) == slot) return true;
    }
    return false;
}

fn carriedSidecarConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    candidate: backend_mod.DeviceOp,
) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        if (idx >= ops.len) return true;
        if (opAccessConflicts(ops[idx], candidate)) return true;
    }
    return false;
}

fn sidecarConflictsSiblingAnchors(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    producer_slot: usize,
    candidate: backend_mod.DeviceOp,
) bool {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (slot == producer_slot) continue;
        if (idx >= ops.len or opAccessConflicts(ops[idx], candidate)) return true;
    }
    return false;
}

fn denseProjectionElementwiseAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    e: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len) return null;
        const m = switch (ops[idx]) {
            .matmul => |m| m,
            else => return null,
        };
        if (matmulElementwiseSidecarCompatible(m, e)) return slot;
    }
    return null;
}

fn projectionElementwiseAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    e: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len) return null;
        const q = switch (ops[idx]) {
            .qmatmul => |q| q,
            else => return null,
        };
        if (projectionSidecarCompatible(q, .{ .elementwise = e })) return slot;
    }
    return null;
}

fn collectProjectionElementwiseSidecars(
    ops: []const backend_mod.DeviceOp,
    used: ?[]const bool,
    executed: ?[]const bool,
    group_start: usize,
    command: *ProgramCommand,
) void {
    var scan = group_start + 1;
    scan_loop: while (scan < ops.len and command.sidecar_count < max_projection_group_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        if (commandContainsIndex(command, scan)) continue;
        const e = switch (ops[scan]) {
            .elementwise => |e| e,
            else => continue,
        };
        const slot = projectionElementwiseAnchorSlot(ops, command, e) orelse continue;
        if (commandAnchorSlotHasSidecar(command, slot)) continue;

        var candidate = command.*;
        if (!appendProjectionCacheSidecar(&candidate, scan, group_start, slot)) break :scan_loop;
        const producer = backend_mod.DeviceOp{ .qmatmul = ops[command.indices[slot]].qmatmul };
        const sidecar = backend_mod.DeviceOp{ .elementwise = e };
        if (sidecarConflictsSiblingAnchors(ops, command, slot, sidecar)) continue;
        if (!canHoistCacheSidecarToGroup(ops, group_start, scan, producer, sidecar, &candidate, executed)) continue;
        if (carriedSidecarConflictsSelected(ops, command, sidecar)) continue;

        command.* = candidate;
    }
}

fn collectDenseProjectionElementwiseSidecars(
    ops: []const backend_mod.DeviceOp,
    used: ?[]const bool,
    executed: ?[]const bool,
    group_start: usize,
    command: *ProgramCommand,
) void {
    var scan = group_start + 1;
    scan_loop: while (scan < ops.len and command.sidecar_count < max_projection_group_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        if (commandContainsIndex(command, scan)) continue;
        const e = switch (ops[scan]) {
            .elementwise => |e| e,
            else => continue,
        };
        const slot = denseProjectionElementwiseAnchorSlot(ops, command, e) orelse continue;
        if (commandAnchorSlotHasSidecar(command, slot)) continue;

        var candidate = command.*;
        if (!appendProjectionCacheSidecar(&candidate, scan, group_start, slot)) break :scan_loop;
        const producer = backend_mod.DeviceOp{ .matmul = ops[command.indices[slot]].matmul };
        const sidecar = backend_mod.DeviceOp{ .elementwise = e };
        if (sidecarConflictsSiblingAnchors(ops, command, slot, sidecar)) continue;
        if (!canHoistCacheSidecarToGroup(ops, group_start, scan, producer, sidecar, &candidate, executed)) continue;
        if (carriedSidecarConflictsSelected(ops, command, sidecar)) continue;

        command.* = candidate;
    }
}

const ProjectionCacheSidecarScan = struct {
    ops: []const backend_mod.DeviceOp,
    used: ?[]const bool,
    executed: ?[]const bool,
    group_start: usize,
    fixed_end: ?usize = null,
    max_sidecars_per_anchor: u32,
    per_anchor_sidecars: [max_projection_group_anchors]u32 = [_]u32{0} ** max_projection_group_anchors,

    fn collect(
        self: *ProjectionCacheSidecarScan,
        comptime anchor_tag: DeviceOpTag,
        comptime sliceCompatibleFn: anytype,
        comptime ropeStoreCompatibleFn: anytype,
        command: *ProgramCommand,
    ) void {
        var scan = self.group_start + 1;
        scan_loop: while (scan < self.ops.len and command.sidecar_count < max_projection_group_anchors) : (scan += 1) {
            if (self.used) |used_ops| {
                if (scan >= used_ops.len or used_ops[scan]) continue;
            }
            if (commandContainsIndex(command, scan)) continue;
            if (self.stopsAtNextAnchor(command, scan, anchor_tag)) break;
            if (self.ops[scan] == .rope and scan + 1 < self.ops.len) store: {
                if (self.used) |used_ops| {
                    if (scan + 1 >= used_ops.len or used_ops[scan + 1]) break :store;
                }
                const rr = self.ops[scan].rope;
                const sa = switch (self.ops[scan + 1]) {
                    .slice_assign => |sa| sa,
                    else => break :store,
                };
                const slot = projectionCacheAnchorSlot(self.ops, command, anchor_tag, .{ .rr = rr, .sa = sa }, ropeStoreCompatibleFn) orelse break :store;
                if (!self.slotCanAccept(command, slot, sa)) break :store;

                var candidate = command.*;
                if (!appendProjectionCacheRopeStoreSidecar(&candidate, scan, scan + 1, self.group_start, slot)) break :scan_loop;
                if (!canHoistCacheRopeStoreToGroup(self.ops, self.group_start, scan, scan + 1, self.anchorProducer(anchor_tag, command, slot), rr, sa, &candidate, self.executed)) break :store;
                if (projectionCacheSidecarConflictsSelected(self.ops, command, sa)) break :store;
                if (projectionCacheRopeOutputHasExternalUsers(self.ops, scan, scan + 1)) break :store;

                command.* = candidate;
                self.per_anchor_sidecars[slot] += 1;
                scan += 1;
                continue;
            }
            const sa = switch (self.ops[scan]) {
                .slice_assign => |sa| sa,
                else => continue,
            };
            const slot = projectionCacheAnchorSlot(self.ops, command, anchor_tag, sa, sliceCompatibleFn) orelse continue;
            if (!self.slotCanAccept(command, slot, sa)) continue;

            var candidate = command.*;
            if (!appendProjectionCacheSidecar(&candidate, scan, self.group_start, slot)) break;
            if (!canHoistCacheSidecarToGroup(self.ops, self.group_start, scan, self.anchorProducer(anchor_tag, command, slot), .{ .slice_assign = sa }, &candidate, self.executed)) continue;
            if (projectionCacheSidecarConflictsSelected(self.ops, command, sa)) continue;

            command.* = candidate;
            self.per_anchor_sidecars[slot] += 1;
        }
    }

    fn stopsAtNextAnchor(
        self: *const ProjectionCacheSidecarScan,
        command: *const ProgramCommand,
        scan: usize,
        comptime anchor_tag: DeviceOpTag,
    ) bool {
        const extent_end = self.fixed_end orelse self.group_start + @as(usize, command.op_count);
        return scan > extent_end and std.meta.activeTag(self.ops[scan]) == anchor_tag;
    }

    fn slotCanAccept(self: *const ProjectionCacheSidecarScan, command: *const ProgramCommand, slot: usize, sa: anytype) bool {
        return self.per_anchor_sidecars[slot] < self.max_sidecars_per_anchor and
            projectionCacheSidecarsShareSink(self.ops, command, slot, sa);
    }

    fn anchorProducer(
        self: *const ProjectionCacheSidecarScan,
        comptime anchor_tag: DeviceOpTag,
        command: *const ProgramCommand,
        slot: usize,
    ) backend_mod.DeviceOp {
        const anchor = self.ops[command.indices[slot]];
        return @unionInit(backend_mod.DeviceOp, @tagName(anchor_tag), @field(anchor, @tagName(anchor_tag)));
    }
};

fn appendProjectionCacheSidecar(command: *ProgramCommand, sidecar_index: usize, group_start: usize, slot: usize) bool {
    if (command.sidecar_count >= max_projection_group_anchors) return false;
    if (slot >= command.anchor_count or slot > std.math.maxInt(u8)) return false;
    const sidecar_slot = command.sidecar_count;
    command.sidecar_indices[sidecar_slot] = sidecar_index;
    command.sidecar_slots[sidecar_slot] = @intCast(slot);
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn appendProjectionCacheRopeStoreSidecar(command: *ProgramCommand, rope_index: usize, sidecar_index: usize, group_start: usize, slot: usize) bool {
    if (command.sidecar_count + 2 > max_projection_group_anchors) return false;
    if (slot >= command.anchor_count or slot > std.math.maxInt(u8)) return false;
    command.sidecar_indices[command.sidecar_count] = rope_index;
    command.sidecar_slots[command.sidecar_count] = @intCast(slot);
    command.sidecar_count += 1;
    command.sidecar_indices[command.sidecar_count] = sidecar_index;
    command.sidecar_slots[command.sidecar_count] = @intCast(slot);
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn projectionCacheAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    comptime anchor_tag: DeviceOpTag,
    candidate: anytype,
    comptime compatibleFn: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len or std.meta.activeTag(ops[idx]) != anchor_tag) return null;
        if (compatibleFn(@field(ops[idx], @tagName(anchor_tag)), candidate)) return slot;
    }
    return null;
}

fn projectionSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    return projectionSidecarCompatible(q, .{ .slice_assign = sa });
}

fn qmatvecRopeStoreSidecarCompatiblePair(q: anytype, pair: anytype) bool {
    return qmatvecRopeStoreSidecarCompatible(q, pair.rr, pair.sa);
}

fn denseMatvecRopeStoreSidecarCompatiblePair(m: anytype, pair: anytype) bool {
    return denseMatvecRopeStoreSidecarCompatible(m, pair.rr, pair.sa);
}

fn projectionCacheSidecarsShareSink(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    anchor_slot: usize,
    sa: anytype,
) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        switch (ops[idx]) {
            .slice_assign => |selected| {
                const selected_slot = command.sidecarAnchorSlot(i) orelse return false;
                if (selected_slot == anchor_slot and selected.dst != sa.dst) return false;
            },
            .rope => {
                if (i + 1 < command.sidecar_count) {
                    const sidecar_idx = command.sidecar_indices[i + 1] orelse return false;
                    const selected = switch (ops[sidecar_idx]) {
                        .slice_assign => |selected| selected,
                        else => continue,
                    };
                    const selected_slot = command.sidecarAnchorSlot(i) orelse return false;
                    if (command.sidecarAnchorSlot(i + 1) != selected_slot) return false;
                    if (selected_slot == anchor_slot and selected.dst != sa.dst) return false;
                    i += 1;
                }
            },
            else => return false,
        }
    }
    return true;
}

fn projectionCacheSidecarConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    sa: anytype,
) bool {
    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        switch (ops[idx]) {
            .slice_assign => |selected| {
                if (sliceAssignWritesMayOverlap(selected, sa)) return true;
            },
            .rope => |rr| {
                if (opAccessConflicts(.{ .rope = rr }, .{ .slice_assign = sa })) return true;
                if (i + 1 >= command.sidecar_count) return true;
                const slot = command.sidecarAnchorSlot(i) orelse return true;
                if (command.sidecarAnchorSlot(i + 1) != slot) return true;
                const sidecar_idx = command.sidecar_indices[i + 1] orelse return true;
                const selected = switch (ops[sidecar_idx]) {
                    .slice_assign => |selected| selected,
                    else => return true,
                };
                if (sliceAssignWritesMayOverlap(selected, sa)) return true;
                i += 1;
            },
            else => return true,
        }
    }
    return false;
}

fn hoistScanSkips(command: *const ProgramCommand, executed: ?[]const bool, idx: usize) bool {
    if (commandContainsIndex(command, idx)) return true;
    if (executed) |executed_ops| return idx < executed_ops.len and executed_ops[idx];
    return false;
}

fn opWriteCoversRead(op: backend_mod.DeviceOp, read: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.writeSpans()) |write| {
        if (write.buf == read.buf and write.start <= read.start and write.end >= read.end) return true;
    }
    return access.write_overflow and opWritesBuffer(op, read.buf);
}

fn canHoistCacheSidecarToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    sidecar_index: usize,
    producer: backend_mod.DeviceOp,
    sidecar: backend_mod.DeviceOp,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const sidecar_access = opAccessSpans(sidecar);
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (hoistScanSkips(command, executed, idx)) continue;
        for (sidecar_access.readSpans()) |read| {
            if (opWriteCoversRead(producer, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.read_overflow or sidecar_access.write_overflow) {
            if (opAccessConflicts(op, sidecar)) return false;
        }
    }
    return true;
}

fn canHoistCacheRopeStoreToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    rope_index: usize,
    sidecar_index: usize,
    producer: backend_mod.DeviceOp,
    rr: anytype,
    sa: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[group_start..rope_index], group_start..) |op, idx| {
        if (hoistScanSkips(command, executed, idx)) continue;
        for (rope_access.readSpans()) |read| {
            if (opWriteCoversRead(producer, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        if (rope_access.read_overflow and (opWritesBuffer(op, rr.src) or opWritesBuffer(op, rr.cos_sin))) return false;
    }

    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (hoistScanSkips(command, executed, idx)) continue;
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.write_overflow and opTouchesBuffer(op, sa.dst)) return false;
    }

    return true;
}

fn projectionCacheRopeOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    sidecar_index: usize,
) bool {
    var command = ProgramCommand{
        .kind = .rope_store_group,
        .op_start = @intCast(rope_index),
        .op_count = 1,
    };
    appendRopeStorePair(&command, rope_index, sidecar_index, rope_index);
    return ropeStoreGroupOutputsHaveExternalUsers(ops, &command);
}

pub fn buildProgramCommands(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) ![]ProgramCommand {
    var commands: std.ArrayListUnmanaged(ProgramCommand) = .empty;
    errdefer commands.deinit(alloc);
    var pending: std.ArrayListUnmanaged(PendingProgramCommand) = .empty;
    defer pending.deinit(alloc);

    const used = try alloc.alloc(bool, ops.len);
    defer alloc.free(used);
    @memset(used, false);
    const executed = try alloc.alloc(bool, ops.len);
    defer alloc.free(executed);
    @memset(executed, false);

    var i: usize = 0;
    while (i < ops.len) {
        try emitPendingProgramCommandsAt(alloc, &commands, &pending, used, executed, i);
        if (used[i]) {
            i += 1;
            continue;
        }

        if (findDelayedProgramCommand(ops, i, policy, used, executed)) |command| {
            markProgramCommandUsed(used, command);
            try pending.append(alloc, .{ .emit_at = command.op_start, .command = command });
            i += 1;
            continue;
        }

        if (findProgramCommandWithExecuted(ops, i, policy, used, executed)) |command| {
            markProgramCommandUsed(used, command);
            markProgramCommandUsed(executed, command);
            try commands.append(alloc, command);
            i += command.advanceCount();
            continue;
        }

        const command = ProgramCommand.op(i);
        markProgramCommandUsed(used, command);
        markProgramCommandUsed(executed, command);
        try commands.append(alloc, command);
        i += 1;
    }
    while (pending.items.len != 0) {
        const pending_command = pending.orderedRemove(0);
        try commands.append(alloc, pending_command.command);
    }
    mergeAdjacentProjectionRowChains(ops, policy, &commands);

    return commands.toOwnedSlice(alloc);
}

fn mergeAdjacentProjectionRowChains(
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
    commands: *std.ArrayListUnmanaged(ProgramCommand),
) void {
    if (!policy.fuse_projection_row_chain and !policy.fuse_dense_projection_row_chain) return;
    const items = commands.items;
    var read: usize = 0;
    var write: usize = 0;
    while (read < items.len) {
        if (read + 1 < items.len) {
            const current = items[read];
            const next = items[read + 1];
            if (policy.fuse_projection_row_chain and current.kind == .projection_chain and next.kind == .row_chain) {
                if (findProjectionRowChainCommand(ops, current.op_start, policy, null)) |fused| {
                    if (fused.op_start == current.op_start and fused.op_count == current.op_count + next.op_count) {
                        items[write] = fused;
                        write += 1;
                        read += 2;
                        continue;
                    }
                }
            } else if (policy.fuse_dense_projection_row_chain and current.kind == .dense_projection_chain and next.kind == .row_chain) {
                if (findDenseProjectionRowChainCommand(ops, current.op_start, policy, null)) |fused| {
                    if (fused.op_start == current.op_start and fused.op_count == current.op_count + next.op_count) {
                        items[write] = fused;
                        write += 1;
                        read += 2;
                        continue;
                    }
                }
            }
        }
        items[write] = items[read];
        write += 1;
        read += 1;
    }
    commands.items.len = write;
}

fn findDelayedProgramCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    return findProgramCommandIn(delayed_program_command_finders, ops, start, policy, used, executed);
}

const PendingProgramCommand = struct {
    emit_at: u32,
    command: ProgramCommand,
};

fn emitPendingProgramCommandsAt(
    alloc: std.mem.Allocator,
    commands: *std.ArrayListUnmanaged(ProgramCommand),
    pending: *std.ArrayListUnmanaged(PendingProgramCommand),
    used: []bool,
    executed: []bool,
    index: usize,
) !void {
    var p: usize = 0;
    while (p < pending.items.len) {
        if (pending.items[p].emit_at != index) {
            p += 1;
            continue;
        }
        const pending_command = pending.orderedRemove(p);
        markProgramCommandUsed(used, pending_command.command);
        markProgramCommandUsed(executed, pending_command.command);
        try commands.append(alloc, pending_command.command);
    }
}

fn findProgramCommandWithExecuted(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }

    return findProgramCommandIn(program_command_finders, ops, start, policy, used, executed);
}

fn findProgramCommandIn(
    comptime finders: anytype,
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    inline for (finders) |finder| {
        if (finder.run(ops, start, policy, used, executed)) |command| return command;
    }
    return null;
}

const ProgramCommandFinderFn = *const fn ([]const backend_mod.DeviceOp, usize, CommandStreamPolicy, ?[]const bool, ?[]const bool) ?ProgramCommand;

const ProgramCommandFinderFeature = enum {
    always,
    row_rope_chains,
    elementwise_batch,
    repeat_fused_elementwise,
    projection_chain,
    projection_row_chain,
    rope_batch,
    movement_batch,
    attention_group,
    rope_attention_store_batch,

    fn enabled(self: ProgramCommandFinderFeature, policy: CommandStreamPolicy) bool {
        return switch (self) {
            .always => true,
            .row_rope_chains => policy.row_rope_chains,
            .elementwise_batch => policy.max_elementwise_batch >= 2,
            .repeat_fused_elementwise => policy.fuse_repeat_fused_elementwise,
            .projection_chain => policy.fuse_projection_chain,
            .projection_row_chain => policy.fuse_projection_row_chain,
            .rope_batch => policy.max_rope_batch >= 2,
            .movement_batch => policy.max_movement_batch >= 2,
            .attention_group => policy.max_attention_batch >= 2,
            .rope_attention_store_batch => policy.max_rope_attention_store_batch >= 2,
        };
    }
};

const ProgramCommandFinder = struct {
    start_tag: ?DeviceOpTag = null,
    feature: ProgramCommandFinderFeature = .always,
    find: ProgramCommandFinderFn,

    fn run(
        self: ProgramCommandFinder,
        ops: []const backend_mod.DeviceOp,
        start: usize,
        policy: CommandStreamPolicy,
        used: ?[]const bool,
        executed: ?[]const bool,
    ) ?ProgramCommand {
        if (self.start_tag) |tag| {
            if (!opIs(ops, start, tag)) return null;
        }
        if (!self.feature.enabled(policy)) return null;
        return self.find(ops, start, policy, used, executed);
    }
};

const program_command_finders = [_]ProgramCommandFinder{
    .{ .start_tag = .qmatmul, .find = finderUsedOnly(findProjectionPairElementwiseChainCommand) },
    .{ .start_tag = .qmatmul, .find = finderUsedOnly(findProjectionPairFusedElementwiseChainCommand) },
    .{ .start_tag = .qmatmul, .feature = .projection_row_chain, .find = finderPolicyUsed(findProjectionRowChainCommand) },
    .{ .start_tag = .qmatmul, .find = findProjectionCacheGroupCommand },
    .{ .start_tag = .qmatmul, .find = findProjectionGroupAt },
    .{ .start_tag = .qmatmul, .feature = .projection_chain, .find = finderUsedOnly(findProjectionChainCommand) },
    .{ .start_tag = .matmul, .find = finderUsedOnly(findDenseProjectionPairFusedElementwiseChainCommand) },
    .{ .start_tag = .matmul, .find = findDenseProjectionCacheGroupCommand },
    .{ .start_tag = .matmul, .feature = .projection_row_chain, .find = finderPolicyUsed(findDenseProjectionRowChainCommand) },
    .{ .start_tag = .matmul, .feature = .projection_chain, .find = finderUsedOnly(findDenseProjectionChainCommand) },
    .{ .start_tag = .elementwise, .feature = .elementwise_batch, .find = finderPolicyUsed(findElementwiseBatchCommand) },
    .{ .start_tag = .repeat, .feature = .repeat_fused_elementwise, .find = finderUsedOnly(findRepeatFusedElementwiseCommand) },
    .{ .start_tag = .rope, .feature = .rope_batch, .find = findRopeStoreGroupCommand },
    .{ .feature = .row_rope_chains, .find = findRowRopeChainCommand },
    .{ .find = finderPolicyUsed(findContiguousBatchCommand) },
    .{ .start_tag = .rope, .feature = .rope_attention_store_batch, .find = findRopeAttentionStoreGroupCommand },
    .{ .start_tag = .rope, .find = finderUsedExecuted(findRopeAttentionStoreChainCommand) },
    .{ .start_tag = .slice_assign, .find = finderUsedOnly(findAttentionChainCommand) },
    .{ .start_tag = .attention, .feature = .attention_group, .find = findAttentionStoreGroupCommand },
    .{ .start_tag = .attention, .find = finderUsedOnly(findAttentionStoreChainCommand) },
    .{ .start_tag = .slice_assign, .feature = .movement_batch, .find = finderPolicyUsed(findMovementGroupCommand) },
    .{ .start_tag = .attention, .feature = .attention_group, .find = finderPolicyUsed(findAttentionGroupCommand) },
};

const delayed_program_command_finders = [_]ProgramCommandFinder{
    .{ .start_tag = .rope, .find = findDelayedRopeAttentionStoreGroupCommand },
};

fn opIs(ops: []const backend_mod.DeviceOp, start: usize, tag: DeviceOpTag) bool {
    if (start >= ops.len) return false;
    return std.meta.activeTag(ops[start]) == tag;
}

fn findProjectionGroupAt(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    _ = executed;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const projection_policy = policy.projectionPolicyFor(q) orelse return null;
    const selection = findProjectionGroup(ops, start, projection_policy, used) orelse return null;
    return ProgramCommand.fromProjectionSelection(selection);
}

fn findRowRopeChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    _ = executed;
    _ = policy;
    if (start + 3 < ops.len and isRmsnormScaleActivationChain(ops[start], ops[start + 1], ops[start + 2], ops[start + 3])) {
        if (commandRangeTouchesUsed(@intCast(start), 4, used)) return null;
        if (rmsnormScaleActivationChainHasExternalUsers(ops, start)) return null;
        return ProgramCommand.contiguous(.row_chain, start, 4);
    }
    if (start + 2 < ops.len and isRmsnormScaleChain(ops[start], ops[start + 1], ops[start + 2])) {
        if (commandRangeTouchesUsed(@intCast(start), 3, used)) return null;
        return ProgramCommand.contiguous(.row_chain, start, 3);
    }
    if (start + 1 < ops.len and isRopeSliceAssignChain(ops[start], ops[start + 1])) {
        if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
        return ProgramCommand.contiguous(.rope_chain, start, 2);
    }
    return null;
}

fn finderUsedOnly(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = policy;
            _ = executed;
            return find(ops, start, used);
        }
    }.run;
}

fn finderPolicyUsed(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = executed;
            return find(ops, start, policy, used);
        }
    }.run;
}

fn finderUsedExecuted(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = policy;
            return find(ops, start, used, executed);
        }
    }.run;
}

fn findProjectionPairFusedElementwiseChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 5 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 6, used)) return null;
    const gate = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const first = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const rp = switch (ops[start + 2]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const second = switch (ops[start + 3]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const up = switch (ops[start + 4]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const product = switch (ops[start + 5]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!projectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return null;
    if (projectionPairFusedElementwiseChainHasExternalUsers(ops, start)) return null;
    var command = ProgramCommand.contiguous(.projection_pair_fused_elementwise_chain, start, 6);
    command.projection_kind = if (gate.M == 1) .qmatvec else .qmatmul;
    return command;
}

fn findProjectionPairElementwiseChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const left = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (!qmatvecSiblingProjectionAnchor(left)) return null;

    var right_index = start + 1;
    while (right_index < ops.len) : (right_index += 1) {
        if (used) |used_ops| {
            if (right_index >= used_ops.len or used_ops[right_index]) continue;
        }
        const right = switch (ops[right_index]) {
            .qmatmul => |q| q,
            else => continue,
        };
        if (!qmatvecSiblingProjectionCompatible(left, right)) continue;
        if (!canHoistProjectionTo(ops, start, right_index, right)) continue;

        var elementwise_index = right_index + 1;
        while (elementwise_index < ops.len) : (elementwise_index += 1) {
            if (used) |used_ops| {
                if (elementwise_index >= used_ops.len or used_ops[elementwise_index]) continue;
            }
            const e = switch (ops[elementwise_index]) {
                .elementwise => |e| e,
                else => continue,
            };
            if (!qmatvecPairElementwiseCompatible(left, right, e)) continue;

            var command = ProgramCommand{
                .kind = .projection_pair_elementwise_chain,
                .op_start = @intCast(start),
                .op_count = @intCast(elementwise_index - start + 1),
                .projection_kind = .qmatvec,
                .anchor_count = 2,
                .sidecar_count = 1,
            };
            command.indices[0] = start;
            command.indices[1] = right_index;
            command.sidecar_indices[0] = elementwise_index;
            if (!canHoistSiblingElementwiseToCommand(ops, start, elementwise_index, .{ .elementwise = e }, &command)) continue;
            if (projectionPairElementwiseHasExternalUsers(ops, &command)) continue;
            return command;
        }
    }
    return null;
}

fn findDenseProjectionPairFusedElementwiseChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 5 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 6, used)) return null;
    const gate = switch (ops[start]) {
        .matmul => |m| m,
        else => return null,
    };
    const first = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const rp = switch (ops[start + 2]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const second = switch (ops[start + 3]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const up = switch (ops[start + 4]) {
        .matmul => |m| m,
        else => return null,
    };
    const product = switch (ops[start + 5]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!denseProjectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return null;
    if (denseProjectionPairFusedElementwiseChainHasExternalUsers(ops, start)) return null;
    return ProgramCommand.contiguous(.dense_projection_pair_fused_elementwise_chain, start, 6);
}

fn findRepeatFusedElementwiseCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const rp = switch (ops[start]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const fe = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    if (!repeatFusedElementwiseCompatible(rp, fe)) return null;
    if (repeatOutputHasExternalUsers(ops, start, start + 1)) return null;
    return ProgramCommand.contiguous(.repeat_fused_elementwise_chain, start, 2);
}

fn findAttentionChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const sa = switch (ops[start]) {
        .slice_assign => |sa| sa,
        else => return null,
    };
    const att = switch (ops[start + 1]) {
        .attention => |att| att,
        else => return null,
    };
    if (attentionSliceAssignOperand(sa, att) == null) return null;
    if (attentionHasFusableStoreSidecar(ops, start + 1, used)) return null;

    var command = ProgramCommand{
        .kind = .attention_chain,
        .op_start = @intCast(start),
        .op_count = 2,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start + 1;
    command.sidecar_indices[0] = start;
    return command;
}

fn attentionHasFusableStoreSidecar(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    used: ?[]const bool,
) bool {
    if (attention_index >= ops.len) return false;
    if (used) |used_ops| {
        if (attention_index >= used_ops.len or used_ops[attention_index]) return false;
    }
    const att = switch (ops[attention_index]) {
        .attention => |att| att,
        else => return false,
    };

    var scan = attention_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (sa.src != att.dst) continue;
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, attention_index, scan, sa)) continue;
        return true;
    }
    return false;
}

const RopeStorePair = struct {
    sidecar_index: usize,
};

fn findRopeStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @intCast(@min(policy.max_rope_batch, max_projection_group_anchors));
    if (max_pairs < 2) return null;

    const first_pair = findRopeStorePair(ops, start, used) orelse return null;
    const first_rope = ops[start].rope;
    const first_sa = ops[first_pair.sidecar_index].slice_assign;

    var command = ProgramCommand{
        .kind = .rope_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeStorePair(&command, start, first_pair.sidecar_index, start);

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        const pair = findRopeStorePair(ops, scan, used) orelse continue;
        const sa = ops[pair.sidecar_index].slice_assign;
        if (!ropeStoreGroupCompatible(first_rope, first_sa, rr, sa)) continue;

        var candidate = command;
        if (candidate.anchor_count + 1 > max_projection_group_anchors) break;
        appendRopeStorePair(&candidate, scan, pair.sidecar_index, start);
        if (!canHoistRopeStorePairToGroup(ops, start, scan, pair.sidecar_index, rr, sa, &candidate, executed)) continue;
        if (ropeStorePairConflictsSelected(ops, &command, scan, sa)) continue;

        command = candidate;
    }

    if (ropeStoreGroupOutputsHaveExternalUsers(ops, &command)) return null;
    return if (command.anchor_count >= 2) command else null;
}

fn findRopeStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
) ?RopeStorePair {
    if (rope_index + 1 >= ops.len) return null;
    if (used) |used_ops| {
        if (rope_index >= used_ops.len or used_ops[rope_index]) return null;
        if (rope_index + 1 >= used_ops.len or used_ops[rope_index + 1]) return null;
    }
    if (!isRopeSliceAssignChain(ops[rope_index], ops[rope_index + 1])) return null;
    return .{ .sidecar_index = rope_index + 1 };
}

fn appendRopeStorePair(
    command: *ProgramCommand,
    rope_index: usize,
    sidecar_index: usize,
    group_start: usize,
) void {
    const slot = command.anchor_count;
    command.indices[slot] = rope_index;
    command.sidecar_indices[slot] = sidecar_index;
    command.anchor_count += 1;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
}

fn canHoistRopeStorePairToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    rope_index: usize,
    sidecar_index: usize,
    rr: anytype,
    sa: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[group_start..rope_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (rope_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        if (rope_access.read_overflow and (opWritesBuffer(op, rr.src) or opWritesBuffer(op, rr.cos_sin))) return false;
    }

    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.write_overflow and opTouchesBuffer(op, sa.dst)) return false;
    }

    return true;
}

fn ropeStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rope_index: usize,
    sa: anytype,
) bool {
    const rope_op = ops[rope_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    const sidecar_access = opAccessSpans(sidecar_op);
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
        const selected_access = opAccessSpans(ops[idx]);
        for (selected_access.writeSpans()) |write| {
            if (opReadsSpan(rope_op, write)) return true;
        }
        if (selected_access.write_overflow and opReadsBuffer(rope_op, selected_sa.dst)) return true;
    }
    return false;
}

fn ropeStoreGroupOutputsHaveExternalUsers(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    for (command.anchorIndices()) |rope_index| {
        const rr = switch (ops[rope_index]) {
            .rope => |rr| rr,
            else => return true,
        };
        const rope_access = opAccessSpans(.{ .rope = rr });
        var live_writes = [_]bool{false} ** max_access_spans;
        for (rope_access.writeSpans(), 0..) |_, slot| live_writes[slot] = true;
        var overflow_live = rope_access.write_overflow;

        for (ops[rope_index + 1 ..], rope_index + 1..) |op, idx| {
            if (commandContainsIndex(command, idx)) continue;
            for (rope_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opReadsSpan(op, write)) return true;
            }
            if (overflow_live and opReadsBuffer(op, rr.dst)) return true;

            var any_live = false;
            for (rope_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opWritesCoverSpan(op, write)) {
                    live_writes[slot] = false;
                } else {
                    any_live = true;
                }
            }
            if (overflow_live and opWritesBuffer(op, rr.dst)) overflow_live = false;
            if (!any_live and !overflow_live) break;
        }
    }
    return false;
}

pub fn projectionPrimaryOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
    sidecar_index: usize,
) bool {
    const sidecars = [_]?usize{sidecar_index};
    return projectionPrimaryOutputHasExternalUsersExcept(ops, q_index, sidecars[0..]);
}

pub fn projectionPrimaryOutputHasExternalUsersExcept(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
    sidecar_indices: []const ?usize,
) bool {
    if (q_index >= ops.len) return true;
    const q = switch (ops[q_index]) {
        .qmatmul => |q| q,
        else => return true,
    };
    for (sidecar_indices) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        if (idx >= ops.len or idx <= q_index) return true;
        if (opReadsBuffer(ops[idx], q.dst) and !projectionPrimarySidecarCompatible(q, ops[idx])) return true;
    }

    const q_access = opAccessSpans(.{ .qmatmul = q });
    var live_writes = [_]bool{false} ** max_access_spans;
    for (q_access.writeSpans(), 0..) |_, slot| live_writes[slot] = true;
    var overflow_live = q_access.write_overflow;

    var scan = q_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const op = ops[scan];
        if (!optionalIndexContains(sidecar_indices, scan)) {
            for (q_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opReadsSpan(op, write)) return true;
            }
            if (overflow_live and opReadsBuffer(op, q.dst)) return true;
        }

        var any_live = false;
        for (q_access.writeSpans(), 0..) |write, slot| {
            if (!live_writes[slot]) continue;
            if (opWritesCoverSpan(op, write)) {
                live_writes[slot] = false;
            } else {
                any_live = true;
            }
        }
        if (overflow_live and opWritesBuffer(op, q.dst)) overflow_live = false;
        if (!any_live and !overflow_live) break;
    }

    return false;
}

pub fn matmulPrimaryOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    matmul_index: usize,
    sidecar_index: usize,
) bool {
    const sidecars = [_]?usize{sidecar_index};
    return matmulPrimaryOutputHasExternalUsersExcept(ops, matmul_index, sidecars[0..]);
}

pub fn matmulPrimaryOutputHasExternalUsersExcept(
    ops: []const backend_mod.DeviceOp,
    matmul_index: usize,
    sidecar_indices: []const ?usize,
) bool {
    if (matmul_index >= ops.len) return true;
    const m = switch (ops[matmul_index]) {
        .matmul => |m| m,
        else => return true,
    };
    for (sidecar_indices) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        if (idx >= ops.len or idx <= matmul_index) return true;
        if (opReadsBuffer(ops[idx], m.dst) and !matmulPrimarySidecarCompatible(m, ops[idx])) return true;
    }

    const m_access = opAccessSpans(.{ .matmul = m });
    var live_writes = [_]bool{false} ** max_access_spans;
    for (m_access.writeSpans(), 0..) |_, slot| live_writes[slot] = true;
    var overflow_live = m_access.write_overflow;

    var scan = matmul_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const op = ops[scan];
        if (!optionalIndexContains(sidecar_indices, scan)) {
            for (m_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opReadsSpan(op, write)) return true;
            }
            if (overflow_live and opReadsBuffer(op, m.dst)) return true;
        }

        var any_live = false;
        for (m_access.writeSpans(), 0..) |write, slot| {
            if (!live_writes[slot]) continue;
            if (opWritesCoverSpan(op, write)) {
                live_writes[slot] = false;
            } else {
                any_live = true;
            }
        }
        if (overflow_live and opWritesBuffer(op, m.dst)) overflow_live = false;
        if (!any_live and !overflow_live) break;
    }

    return false;
}

fn optionalIndexContains(indices: []const ?usize, candidate: usize) bool {
    for (indices) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

pub fn repeatFusedElementwiseCompatible(rp: anytype, fe: anytype) bool {
    if (rp.src == rp.dst) return false;
    if (fe.src == rp.dst) return false;

    var found_secondary = false;
    for (fe.steps) |step| {
        if (!step.op.isBinary() or step.secondary_buf != rp.dst) continue;
        if (step.secondary_offset < rp.dst_offset) return false;
        const rel = step.secondary_offset - rp.dst_offset;
        if (@as(u64, rel) + @as(u64, fe.n) > @as(u64, rp.n)) return false;
        found_secondary = true;
    }
    return found_secondary;
}

fn projectionActivationChainCompatible(q: anytype, first: anytype, rp: anytype, second: anytype) bool {
    const projection_compatible = if (q.M == 1)
        qmatvecFusedElementwiseSidecarCompatible(q, first)
    else
        qmatmulFusedElementwiseSidecarCompatible(q, first);
    if (!projection_compatible) return false;
    if (!repeatFusedElementwiseCompatible(rp, second)) return false;
    if (rp.dst == q.dst) return false;
    if (first.dst == q.dst) return false;
    if (first.dst == rp.dst) return false;
    if (second.src != first.dst or second.src_offset != first.dst_offset) return false;
    if (second.n != first.n) return false;
    if (second.dst == q.dst or second.dst == first.dst or second.dst == rp.dst) return false;

    var reads_primary = false;
    for (second.steps) |step| {
        if (!step.op.isBinary()) continue;
        if (step.secondary_buf == first.dst) return false;
        if (step.secondary_buf == q.dst) {
            if (step.secondary_offset != q.dst_offset) return false;
            reads_primary = true;
        }
    }
    return reads_primary;
}

fn denseProjectionActivationChainCompatible(m: anytype, first: anytype, rp: anytype, second: anytype) bool {
    if (!matmulFusedElementwiseSidecarCompatible(m, first)) return false;
    if (!repeatFusedElementwiseCompatible(rp, second)) return false;
    if (rp.dst == m.dst) return false;
    if (first.dst == m.dst) return false;
    if (first.dst == rp.dst) return false;
    if (second.src != first.dst or second.src_offset != first.dst_offset) return false;
    if (second.n != first.n) return false;
    if (second.dst == m.dst or second.dst == first.dst or second.dst == rp.dst) return false;

    var reads_primary = false;
    for (second.steps) |step| {
        if (!step.op.isBinary()) continue;
        if (step.secondary_buf == first.dst) return false;
        if (step.secondary_buf == m.dst) {
            if (step.secondary_offset != m.geom.dst_offset) return false;
            reads_primary = true;
        }
    }
    return reads_primary;
}

fn projectionPairGeometryCompatible(a: anytype, b: anytype) bool {
    return a.M == b.M and
        a.N == b.N and
        a.K == b.K and
        a.input == b.input and
        a.input_offset == b.input_offset and
        a.input_row_stride == b.input_row_stride and
        qmatmulDstRowStride(a) == a.N and
        qmatmulDstRowStride(b) == b.N;
}

fn denseProjectionPairGeometryCompatible(a: anytype, b: anytype) bool {
    const ag = a.geom;
    const bg = b.geom;
    return ag.M != 0 and
        ag.M == bg.M and
        ag.N == bg.N and
        ag.K == bg.K and
        a.a == b.a and
        ag.a_offset == bg.a_offset and
        ag.a_row_stride == bg.a_row_stride and
        ag.a_col_stride == bg.a_col_stride and
        ag.dst_row_stride == ag.N and
        bg.dst_row_stride == bg.N;
}

pub fn projectionPairFusedElementwiseChainCompatible(
    gate: anytype,
    first: anytype,
    rp: anytype,
    second: anytype,
    up: anytype,
    product: anytype,
) bool {
    if (!projectionActivationChainCompatible(gate, first, rp, second)) return false;
    if (!projectionPairGeometryCompatible(gate, up)) return false;
    if (product.op != .mul) return false;
    if (product.n != gate.M * gate.N or product.n != second.n) return false;
    const second_is_src0 = product.src0 == second.dst and product.src0_offset == second.dst_offset;
    const second_is_src1 = product.src1 == second.dst and product.src1_offset == second.dst_offset;
    const up_is_src0 = product.src0 == up.dst and product.src0_offset == up.dst_offset;
    const up_is_src1 = product.src1 == up.dst and product.src1_offset == up.dst_offset;
    if (!((second_is_src0 and up_is_src1) or (second_is_src1 and up_is_src0))) return false;
    if (product.dst == gate.dst or product.dst == first.dst or product.dst == second.dst or product.dst == up.dst) return false;
    return true;
}

pub fn denseProjectionPairFusedElementwiseChainCompatible(
    gate: anytype,
    first: anytype,
    rp: anytype,
    second: anytype,
    up: anytype,
    product: anytype,
) bool {
    const g = gate.geom;
    if (!denseProjectionActivationChainCompatible(gate, first, rp, second)) return false;
    if (!denseProjectionPairGeometryCompatible(gate, up)) return false;
    if (product.op != .mul) return false;
    if (product.n != g.M * g.N or product.n != second.n) return false;
    const second_is_src0 = product.src0 == second.dst and product.src0_offset == second.dst_offset;
    const second_is_src1 = product.src1 == second.dst and product.src1_offset == second.dst_offset;
    const up_is_src0 = product.src0 == up.dst and product.src0_offset == up.geom.dst_offset;
    const up_is_src1 = product.src1 == up.dst and product.src1_offset == up.geom.dst_offset;
    if (!((second_is_src0 and up_is_src1) or (second_is_src1 and up_is_src0))) return false;
    if (product.dst == gate.dst or product.dst == first.dst or product.dst == second.dst or product.dst == up.dst) return false;
    return true;
}

fn spanHasExternalReadAfter(
    ops: []const backend_mod.DeviceOp,
    producer_index: usize,
    included_start: usize,
    included_end: usize,
    span: BufferSpan,
) bool {
    var scan = producer_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const included = scan >= included_start and scan < included_end;
        if (!included and opReadsSpan(ops[scan], span)) return true;
        if (opWritesCoverSpan(ops[scan], span)) break;
    }
    return false;
}

fn projectionPairFusedElementwiseChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
) bool {
    if (q_index + 5 >= ops.len) return true;
    const gate = switch (ops[q_index]) {
        .qmatmul => |q| q,
        else => return true,
    };
    const first = switch (ops[q_index + 1]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const rp = switch (ops[q_index + 2]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const second = switch (ops[q_index + 3]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const up = switch (ops[q_index + 4]) {
        .qmatmul => |q| q,
        else => return true,
    };
    const product = switch (ops[q_index + 5]) {
        .elementwise => |e| e,
        else => return true,
    };
    if (!projectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return true;

    const included_start = q_index + 1;
    const included_end = q_index + 6;
    if (spanHasExternalReadAfter(ops, q_index, included_start, included_end, bufferSpan(gate.dst, gate.dst_offset, gate.M * gate.N))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 1, included_start, included_end, bufferSpan(first.dst, first.dst_offset, first.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 2, included_start, included_end, bufferSpan(rp.dst, rp.dst_offset, rp.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 3, included_start, included_end, bufferSpan(second.dst, second.dst_offset, second.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 4, included_start, included_end, bufferSpan(up.dst, up.dst_offset, up.M * up.N))) return true;
    return false;
}

fn denseProjectionPairFusedElementwiseChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    matmul_index: usize,
) bool {
    if (matmul_index + 5 >= ops.len) return true;
    const gate = switch (ops[matmul_index]) {
        .matmul => |m| m,
        else => return true,
    };
    const first = switch (ops[matmul_index + 1]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const rp = switch (ops[matmul_index + 2]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const second = switch (ops[matmul_index + 3]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const up = switch (ops[matmul_index + 4]) {
        .matmul => |m| m,
        else => return true,
    };
    const product = switch (ops[matmul_index + 5]) {
        .elementwise => |e| e,
        else => return true,
    };
    if (!denseProjectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return true;

    const included_start = matmul_index + 1;
    const included_end = matmul_index + 6;
    if (spanHasExternalReadAfter(ops, matmul_index, included_start, included_end, bufferSpan(gate.dst, gate.geom.dst_offset, gate.geom.M * gate.geom.N))) return true;
    if (spanHasExternalReadAfter(ops, matmul_index + 1, included_start, included_end, bufferSpan(first.dst, first.dst_offset, first.n))) return true;
    if (spanHasExternalReadAfter(ops, matmul_index + 2, included_start, included_end, bufferSpan(rp.dst, rp.dst_offset, rp.n))) return true;
    if (spanHasExternalReadAfter(ops, matmul_index + 3, included_start, included_end, bufferSpan(second.dst, second.dst_offset, second.n))) return true;
    if (spanHasExternalReadAfter(ops, matmul_index + 4, included_start, included_end, bufferSpan(up.dst, up.geom.dst_offset, up.geom.M * up.geom.N))) return true;
    return false;
}

pub fn rmsnormScaleChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    start: usize,
) bool {
    if (start + 2 >= ops.len) return true;
    const rn = switch (ops[start]) {
        .rmsnorm => |r| r,
        else => return true,
    };
    const rp = switch (ops[start + 1]) {
        .repeat => |r| r,
        else => return true,
    };
    if (!isRmsnormScaleChain(ops[start], ops[start + 1], ops[start + 2])) return true;

    const included_start = start + 1;
    const included_end = start + 3;
    if (spanHasExternalReadAfter(ops, start, included_start, included_end, bufferSpan(rn.dst, rn.dst_offset, rn.rows * rn.cols))) return true;
    if (spanHasExternalReadAfter(ops, start + 1, included_start, included_end, bufferSpan(rp.dst, rp.dst_offset, rp.n))) return true;
    return false;
}

pub fn isRmsnormScaleActivationChain(
    rms_op: backend_mod.DeviceOp,
    repeat_op: backend_mod.DeviceOp,
    scale_op: backend_mod.DeviceOp,
    activation_op: backend_mod.DeviceOp,
) bool {
    const scale = switch (scale_op) {
        .elementwise => |e| e,
        else => return false,
    };
    const activation = switch (activation_op) {
        .elementwise => |e| e,
        else => return false,
    };
    if (!isRmsnormScaleChain(rms_op, repeat_op, scale_op)) return false;
    if (activation.op != .relu and activation.op != .gelu and activation.op != .silu) return false;
    if (activation.n != scale.n) return false;
    return activation.src0 == scale.dst and
        activation.src0_offset == scale.dst_offset;
}

pub fn rmsnormScaleActivationChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    start: usize,
) bool {
    if (start + 3 >= ops.len) return true;
    const rn = switch (ops[start]) {
        .rmsnorm => |r| r,
        else => return true,
    };
    const rp = switch (ops[start + 1]) {
        .repeat => |r| r,
        else => return true,
    };
    const scale = switch (ops[start + 2]) {
        .elementwise => |e| e,
        else => return true,
    };
    if (!isRmsnormScaleActivationChain(ops[start], ops[start + 1], ops[start + 2], ops[start + 3])) return true;

    const included_start = start + 1;
    const included_end = start + 4;
    if (spanHasExternalReadAfter(ops, start, included_start, included_end, bufferSpan(rn.dst, rn.dst_offset, rn.rows * rn.cols))) return true;
    if (spanHasExternalReadAfter(ops, start + 1, included_start, included_end, bufferSpan(rp.dst, rp.dst_offset, rp.n))) return true;
    if (spanHasExternalReadAfter(ops, start + 2, included_start, included_end, bufferSpan(scale.dst, scale.dst_offset, scale.n))) return true;
    return false;
}

pub fn projectionRowChainElementwiseHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    command: ProgramCommand,
) bool {
    if (command.sidecar_count < 4) return true;
    const e_idx = command.sidecar_indices[0] orelse return true;
    const rn_idx = command.sidecar_indices[1] orelse return true;
    const rp_idx = command.sidecar_indices[2] orelse return true;
    const out_idx = command.sidecar_indices[3] orelse return true;
    if (e_idx >= ops.len or rn_idx >= ops.len or rp_idx >= ops.len or out_idx >= ops.len) return true;
    const e = switch (ops[e_idx]) {
        .elementwise => |e| e,
        else => return true,
    };
    if (rn_idx != e_idx + 1 or rp_idx != rn_idx + 1 or out_idx != rp_idx + 1) return true;
    return spanHasExternalReadAfter(ops, e_idx, rn_idx, out_idx + 1, bufferSpan(e.dst, e.dst_offset, e.n));
}

fn repeatOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    repeat_index: usize,
    fused_index: usize,
) bool {
    if (repeat_index >= ops.len or fused_index >= ops.len) return true;
    if (fused_index <= repeat_index) return true;
    const rp = switch (ops[repeat_index]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const fe = switch (ops[fused_index]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    if (!repeatFusedElementwiseCompatible(rp, fe)) return true;

    const repeat_write = bufferSpan(rp.dst, rp.dst_offset, rp.n);

    var scan = repeat_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const op = ops[scan];
        if (scan != fused_index) {
            if (opReadsSpan(op, repeat_write)) return true;
        }
        if (opWritesCoverSpan(op, repeat_write)) break;
    }

    return false;
}

fn findAttentionStoreChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const att = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    var sidecar_idx: ?usize = null;
    var scan = start + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, start, scan, sa)) continue;
        sidecar_idx = scan;
        break;
    }
    const found = sidecar_idx orelse return null;

    var command = ProgramCommand{
        .kind = .attention_store_chain,
        .op_start = @intCast(start),
        .op_count = @intCast(found - start + 1),
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = found;
    return command;
}

fn findAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_attention_store_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .attention_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    if (!appendAttentionStorePair(ops, &command, start, first, start, used)) return null;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!attentionGeometryCompatible(first, next)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, scan, next, &command, executed)) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .attention = next })) continue;
        if (!appendAttentionStorePair(ops, &command, scan, next, start, used)) continue;
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findRopeAttentionStoreChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const rr = switch (ops[start]) {
        .rope => |rr| rr,
        else => return null,
    };

    var scan = start + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;

        var command = ProgramCommand{
            .kind = .rope_attention_store_chain,
            .op_start = @intCast(start),
            .op_count = @intCast(scan - start + 1),
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = start;
        command.indices[1] = scan;
        if (ropeOutputHasExternalUsers(ops, start, scan, rr)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, scan, att, &command, executed)) continue;

        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        command.sidecar_indices[0] = sidecar_index;
        command.sidecar_count = 1;
        command.op_count = @intCast(sidecar_index - start + 1);
        return command;
    }
    return null;
}

fn findRopeAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2));
    if (max_pairs < 2) return null;

    const first_pair = findRopeAttentionStorePair(ops, start, used, executed) orelse return null;
    var command = ProgramCommand{
        .kind = .rope_attention_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeAttentionStorePair(&command, start, first_pair.attention_index, first_pair.sidecar_index, start);

    const first_att = ops[first_pair.attention_index].attention;
    const first_rope = ops[start].rope;
    var scan = start + 1;
    while (scan < ops.len and command.sidecar_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) continue;
        const pair = findRopeAttentionStorePair(ops, scan, used, executed) orelse continue;
        const att = ops[pair.attention_index].attention;
        if (!attentionGeometryCompatible(first_att, att)) continue;

        var candidate = command;
        const slot = candidate.anchor_count;
        if (slot + 2 > max_projection_group_anchors) break;
        candidate.indices[slot] = scan;
        candidate.indices[slot + 1] = pair.attention_index;
        candidate.anchor_count += 2;
        if (!canHoistOpToRopeAttentionStoreGroup(ops, start, scan, .{ .rope = rr }, &candidate, executed)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, pair.attention_index, att, &candidate, executed)) continue;
        if (ropeAttentionStorePairConflictsSelected(ops, &command, scan, pair.attention_index, ops[pair.sidecar_index].slice_assign)) continue;

        var selected = command;
        appendRopeAttentionStorePair(&selected, scan, pair.attention_index, pair.sidecar_index, start);
        if (selected.sidecar_count > policy.max_attention_store_batch and !ropeAttentionStoreCompactBatchCompatible(ops, &selected)) continue;
        command = selected;
    }

    return if (command.sidecar_count >= 2) command else null;
}

fn findDelayedRopeAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @max(1, @as(usize, @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2))));

    const first_pair = findDelayableRopeAttentionStorePair(ops, start, used, executed) orelse return null;
    var emit_at = first_pair.attention_index;
    if (emit_at <= start) return null;

    var command = ProgramCommand{
        .kind = .rope_attention_store_group,
        .op_start = @intCast(emit_at),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeAttentionStorePair(&command, start, first_pair.attention_index, first_pair.sidecar_index, emit_at);

    const first_att = ops[first_pair.attention_index].attention;
    const first_rope = ops[start].rope;
    var scan = start + 1;
    while (scan < ops.len and command.sidecar_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) continue;
        const pair = findDelayableRopeAttentionStorePair(ops, scan, used, executed) orelse continue;
        if (pair.attention_index < emit_at) continue;
        const att = ops[pair.attention_index].attention;
        if (!attentionGeometryCompatible(first_att, att)) continue;
        const sa = ops[pair.sidecar_index].slice_assign;

        var candidate = command;
        if (candidate.anchor_count + 2 > max_projection_group_anchors) break;
        appendRopeAttentionStorePair(&candidate, scan, pair.attention_index, pair.sidecar_index, emit_at);
        if (ropeAttentionStorePairConflictsSelected(ops, &command, scan, pair.attention_index, sa)) continue;
        const candidate_emit_at = @max(emit_at, pair.attention_index);
        setRopeAttentionStoreCommandEmit(&candidate, candidate_emit_at);
        if (!ropeAttentionStoreCommandLegalAt(ops, &candidate, candidate_emit_at, executed)) continue;
        if (candidate.sidecar_count > policy.max_attention_store_batch and !ropeAttentionStoreCompactBatchCompatible(ops, &candidate)) continue;

        command = candidate;
        emit_at = candidate_emit_at;
    }

    if (command.sidecar_count >= 2) return command;
    if (findRopeAttentionStoreGroupCommand(ops, start, policy, used, executed) != null) return null;
    if (findRopeAttentionStoreChainCommand(ops, start, used, executed) != null) return null;
    command.kind = .rope_attention_store_chain;
    command.op_count = @intCast(first_pair.sidecar_index - emit_at + 1);
    return command;
}

const RopeAttentionStorePair = struct {
    attention_index: usize,
    sidecar_index: usize,
};

fn findRopeAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?RopeAttentionStorePair {
    const rr = switch (ops[rope_index]) {
        .rope => |rr| rr,
        else => return null,
    };
    var scan = rope_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;
        if (ropeOutputHasExternalUsers(ops, rope_index, scan, rr)) continue;
        var command = ProgramCommand{
            .kind = .rope_attention_store_chain,
            .op_start = @intCast(rope_index),
            .op_count = @intCast(scan - rope_index + 1),
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = rope_index;
        command.indices[1] = scan;
        if (!canHoistAttentionForStoreGroup(ops, rope_index, scan, att, &command, executed)) continue;
        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        return .{ .attention_index = scan, .sidecar_index = sidecar_index };
    }
    return null;
}

fn findDelayableRopeAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?RopeAttentionStorePair {
    const rr = switch (ops[rope_index]) {
        .rope => |rr| rr,
        else => return null,
    };
    var scan = rope_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;
        if (ropeOutputHasExternalUsers(ops, rope_index, scan, rr)) continue;

        var command = ProgramCommand{
            .kind = .rope_attention_store_group,
            .op_start = @intCast(scan),
            .op_count = 1,
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = rope_index;
        command.indices[1] = scan;
        if (!canDelayOpToCommand(ops, rope_index, scan, .{ .rope = rr }, &command)) continue;

        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        command.sidecar_indices[0] = sidecar_index;
        command.sidecar_count = 1;
        const sa = ops[sidecar_index].slice_assign;
        if (!canHoistOpToRopeAttentionStoreGroup(ops, scan, sidecar_index, .{ .slice_assign = sa }, &command, executed)) continue;
        return .{ .attention_index = scan, .sidecar_index = sidecar_index };
    }
    return null;
}

fn appendRopeAttentionStorePair(
    command: *ProgramCommand,
    rope_index: usize,
    attention_index: usize,
    sidecar_index: usize,
    group_start: usize,
) void {
    const slot = command.anchor_count;
    command.indices[slot] = rope_index;
    command.indices[slot + 1] = attention_index;
    command.sidecar_indices[command.sidecar_count] = sidecar_index;
    command.anchor_count += 2;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
}

fn setRopeAttentionStoreCommandEmit(command: *ProgramCommand, emit_at: usize) void {
    command.op_start = @intCast(emit_at);
    var end = emit_at + 1;
    for (command.sidecarIndices()[0..command.sidecar_count]) |maybe_idx| {
        if (maybe_idx) |idx| end = @max(end, idx + 1);
    }
    command.op_count = @intCast(end - emit_at);
}

fn ropeAttentionStoreCommandLegalAt(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    emit_at: usize,
    executed: ?[]const bool,
) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const rope_index = command.indices[i * 2];
        const attention_index = command.indices[i * 2 + 1];
        const sidecar_index = command.sidecar_indices[i] orelse return false;
        const rr = switch (ops[rope_index]) {
            .rope => |rr| rr,
            else => return false,
        };
        const att = switch (ops[attention_index]) {
            .attention => |att| att,
            else => return false,
        };
        const sa = switch (ops[sidecar_index]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!canMoveOpToRopeAttentionStoreEmit(ops, rope_index, emit_at, .{ .rope = rr }, command, executed)) return false;
        if (!canMoveAttentionToRopeAttentionStoreEmit(ops, attention_index, emit_at, att, command, executed)) return false;
        if (!canMoveOpToRopeAttentionStoreEmit(ops, sidecar_index, emit_at, .{ .slice_assign = sa }, command, executed)) return false;
    }
    return true;
}

fn canMoveOpToRopeAttentionStoreEmit(
    ops: []const backend_mod.DeviceOp,
    candidate_index: usize,
    emit_at: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    if (candidate_index < emit_at) return canDelayOpToCommand(ops, candidate_index, emit_at, candidate, command);
    if (candidate_index > emit_at) return canHoistOpToRopeAttentionStoreGroup(ops, emit_at, candidate_index, candidate, command, executed);
    return true;
}

fn canMoveAttentionToRopeAttentionStoreEmit(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    emit_at: usize,
    att: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    if (attention_index < emit_at) return canDelayOpToCommand(ops, attention_index, emit_at, .{ .attention = att }, command);
    if (attention_index > emit_at) return canHoistAttentionForStoreGroup(ops, emit_at, attention_index, att, command, executed);
    return true;
}

fn canDelayOpToCommand(
    ops: []const backend_mod.DeviceOp,
    candidate_index: usize,
    emit_at: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    if (candidate_index > emit_at) return false;
    const candidate_access = opAccessSpans(candidate);
    for (ops[candidate_index + 1 .. emit_at + 1], candidate_index + 1..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (hoistWriteConflict(op, candidate, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

fn canHoistOpToRopeAttentionStoreGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    candidate_index: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const candidate_access = opAccessSpans(candidate);
    for (ops[group_start..candidate_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (hoistWriteConflict(op, candidate, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

fn hoistWriteConflict(op: backend_mod.DeviceOp, candidate: backend_mod.DeviceOp, write: BufferSpan) bool {
    if (!opTouchesSpan(op, write)) return false;
    const candidate_sa = switch (candidate) {
        .slice_assign => |sa| sa,
        else => return true,
    };
    const op_sa = switch (op) {
        .slice_assign => |sa| sa,
        else => return true,
    };
    return opReadsSpan(op, write) or sliceAssignWritesMayOverlap(op_sa, candidate_sa);
}

fn ropeAttentionStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rope_index: usize,
    attention_index: usize,
    sa: anytype,
) bool {
    const rope_op = ops[rope_index];
    const attention_op = ops[attention_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    const sidecar_access = opAccessSpans(sidecar_op);
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
        const selected_access = opAccessSpans(ops[idx]);
        for (selected_access.writeSpans()) |write| {
            if (opReadsSpan(rope_op, write) or opReadsSpan(attention_op, write)) return true;
        }
        if (selected_access.write_overflow and
            (opReadsBuffer(rope_op, selected_sa.dst) or opReadsBuffer(attention_op, selected_sa.dst))) return true;
    }
    return false;
}

fn ropeOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    attention_index: usize,
    rr: anytype,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[rope_index + 1 .. attention_index], rope_index + 1..) |op, idx| {
        if (idx == attention_index) continue;
        for (rope_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return true;
        }
        if (rope_access.write_overflow and opTouchesBuffer(op, rr.dst)) return true;
    }
    return false;
}

fn canHoistAttentionForStoreGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    attention_index: usize,
    att: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const candidate_access = opAccessSpans(.{ .attention = att });
    for (ops[group_start..attention_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, .{ .attention = att })) return false;
        }
    }
    return true;
}

fn commandContainsIndex(command: *const ProgramCommand, candidate: usize) bool {
    for (command.anchorIndices()) |idx| {
        if (idx == candidate) return true;
    }
    for (command.carriedSidecarIndices()) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

fn appendAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    command: *ProgramCommand,
    attention_index: usize,
    att: anytype,
    group_start: usize,
    used: ?[]const bool,
) bool {
    const sidecar_index = findAttentionStoreSidecarIndex(ops, attention_index, att, used) orelse return false;
    const sa = ops[sidecar_index].slice_assign;
    if (attentionStorePairConflictsSelected(ops, command, attention_index, sa)) return false;

    const slot = command.anchor_count;
    command.indices[slot] = attention_index;
    command.sidecar_indices[slot] = sidecar_index;
    command.anchor_count += 1;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn findAttentionStoreSidecarIndex(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    att: anytype,
    used: ?[]const bool,
) ?usize {
    var scan = attention_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, attention_index, scan, sa)) continue;
        return scan;
    }
    return null;
}

fn attentionStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    attention_index: usize,
    sa: anytype,
) bool {
    const attention_op = ops[attention_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    for (command.anchorIndices()) |idx| {
        if (opAccessConflicts(ops[idx], attention_op)) return true;
        if (opAccessConflicts(ops[idx], sidecar_op)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
    }
    return false;
}

fn findMovementGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .slice_assign => |sa| sa,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_movement_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .movement_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!sliceAssignBatchCompatible(first, next)) continue;
        if (!canHoistOpTo(ops, start, scan, .{ .slice_assign = next })) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .slice_assign = next })) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findAttentionGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_attention_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .attention_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!attentionBatchCompatible(first, next)) continue;
        if (!canHoistOpTo(ops, start, scan, .{ .attention = next })) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .attention = next })) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findProjectionChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (!projectionSidecarCompatible(q, ops[start + 1])) return null;

    var command = ProgramCommand{
        .kind = .projection_chain,
        .op_start = @intCast(start),
        .op_count = 2,
        .projection_kind = if (q.M == 1) .qmatvec else .qmatmul,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = start + 1;
    return command;
}

fn findProjectionRowChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 4 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 5, used)) return null;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (q.M == 1) {
        if (!policy.fuse_projection_row_chain_qmatvec) return null;
    } else if (q.M < policy.min_projection_row_chain_rows) return null;
    const e = switch (ops[start + 1]) {
        .elementwise => |e| e,
        else => return null,
    };
    const rn = switch (ops[start + 2]) {
        .rmsnorm => |rn| rn,
        else => return null,
    };
    if (q.M == 1) {
        if (!qmatvecElementwiseSidecarCompatible(q, e)) return null;
    } else if (!qmatmulElementwiseSidecarCompatible(q, e)) return null;
    if (rn.src != e.dst or rn.src_offset != e.dst_offset) return null;
    if (!isRmsnormScaleChain(ops[start + 2], ops[start + 3], ops[start + 4])) return null;

    const projection_sidecars = [_]?usize{start + 1};
    if (projectionPrimaryOutputHasExternalUsersExcept(ops, start, projection_sidecars[0..])) return null;
    if (rmsnormScaleChainHasExternalUsers(ops, start + 2)) return null;

    var command = ProgramCommand{
        .kind = .projection_row_chain,
        .op_start = @intCast(start),
        .op_count = 5,
        .projection_kind = if (q.M == 1) .qmatvec else .qmatmul,
        .anchor_count = 1,
        .sidecar_count = 4,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = start + 1;
    command.sidecar_indices[1] = start + 2;
    command.sidecar_indices[2] = start + 3;
    command.sidecar_indices[3] = start + 4;
    command.sidecar_slots[0] = 0;
    command.sidecar_slots[1] = 0;
    command.sidecar_slots[2] = 0;
    command.sidecar_slots[3] = 0;
    return command;
}

fn findDenseProjectionChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    const m = switch (ops[start]) {
        .matmul => |m| m,
        else => return null,
    };
    if (start + 3 < ops.len and !commandRangeTouchesUsed(@intCast(start), 4, used)) {
        if (matmulRepeatElementwiseBiasActivationCompatible(m, ops[start + 1], ops[start + 2], ops[start + 3])) {
            var command = ProgramCommand{
                .kind = .dense_projection_chain,
                .op_start = @intCast(start),
                .op_count = 4,
                .anchor_count = 1,
                .sidecar_count = 3,
            };
            command.indices[0] = start;
            command.sidecar_indices[0] = start + 1;
            command.sidecar_indices[1] = start + 2;
            command.sidecar_indices[2] = start + 3;
            return command;
        }
    }

    if (start + 2 < ops.len and !commandRangeTouchesUsed(@intCast(start), 3, used)) {
        if (matmulRepeatElementwiseBiasCompatible(m, ops[start + 1], ops[start + 2])) {
            var command = ProgramCommand{
                .kind = .dense_projection_chain,
                .op_start = @intCast(start),
                .op_count = 3,
                .anchor_count = 1,
                .sidecar_count = 2,
            };
            command.indices[0] = start;
            command.sidecar_indices[0] = start + 1;
            command.sidecar_indices[1] = start + 2;
            return command;
        }
    }

    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    switch (ops[start + 1]) {
        .elementwise => |e| if (!matmulElementwiseSidecarCompatible(m, e)) return null,
        .fused_elementwise => |fe| if (!matmulFusedElementwiseSidecarCompatible(m, fe)) return null,
        else => return null,
    }

    var command = ProgramCommand{
        .kind = .dense_projection_chain,
        .op_start = @intCast(start),
        .op_count = 2,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = start + 1;
    return command;
}

fn findDenseProjectionRowChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (!policy.fuse_dense_projection_row_chain) return null;
    if (start + 4 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 5, used)) return null;
    const m = switch (ops[start]) {
        .matmul => |m| m,
        else => return null,
    };
    const e = switch (ops[start + 1]) {
        .elementwise => |e| e,
        else => return null,
    };
    const rn = switch (ops[start + 2]) {
        .rmsnorm => |rn| rn,
        else => return null,
    };
    if (!matmulElementwiseSidecarCompatible(m, e)) return null;
    if (rn.src != e.dst or rn.src_offset != e.dst_offset) return null;
    if (!isRmsnormScaleChain(ops[start + 2], ops[start + 3], ops[start + 4])) return null;

    const projection_sidecars = [_]?usize{start + 1};
    if (matmulPrimaryOutputHasExternalUsersExcept(ops, start, projection_sidecars[0..])) return null;
    if (rmsnormScaleChainHasExternalUsers(ops, start + 2)) return null;

    var command = ProgramCommand{
        .kind = .dense_projection_row_chain,
        .op_start = @intCast(start),
        .op_count = 5,
        .anchor_count = 1,
        .sidecar_count = 4,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = start + 1;
    command.sidecar_indices[1] = start + 2;
    command.sidecar_indices[2] = start + 3;
    command.sidecar_indices[3] = start + 4;
    command.sidecar_slots[0] = 0;
    command.sidecar_slots[1] = 0;
    command.sidecar_slots[2] = 0;
    command.sidecar_slots[3] = 0;
    return command;
}

fn findElementwiseBatchCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!canBatchElementwiseOp(first)) return null;

    const max_ops: usize = @intCast(@min(policy.max_elementwise_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .elementwise_batch,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const e = switch (ops[scan]) {
            .elementwise => |e| e,
            else => continue,
        };
        if (!canBatchElementwiseOp(e)) continue;
        if (!canHoistElementwiseTo(ops, start, scan, e)) continue;
        if (elementwiseConflictsSelected(ops, command.anchorIndices(), e)) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findContiguousBatchCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (policy.max_rope_batch >= 2) {
        const n = ropeBatchRunLen(ops[start..], policy.max_rope_batch);
        if (n >= 2 and !commandRangeTouchesUsed(@intCast(start), @intCast(n), used)) {
            return ProgramCommand.contiguous(.rope_batch, start, n);
        }
    }

    if (policy.max_movement_batch >= 2) {
        const n = sliceAssignBatchRunLen(ops[start..], policy.max_movement_batch);
        if (n >= 2 and !commandRangeTouchesUsed(@intCast(start), @intCast(n), used)) {
            return ProgramCommand.contiguous(.movement_batch, start, n);
        }
    }

    return null;
}

fn summarizeProgramCommands(commands: []const ProgramCommand) ProgramCommandSummary {
    var summary = ProgramCommandSummary{ .commands = @intCast(commands.len) };
    for (commands, 0..) |command, command_index| {
        const covered = command.coveredOpCount();
        summary.covered_ops += covered;
        summary.estimated_dispatches += 1;
        if (covered > 1) {
            summary.estimated_saved_dispatches += covered - 1;
        }
        if (command_index + 1 < commands.len and
            (command.kind == .projection_chain or command.kind == .dense_projection_chain) and
            commands[command_index + 1].kind == .row_chain and
            command.op_start + command.op_count == commands[command_index + 1].op_start)
        {
            summary.projection_chain_row_chain_frontiers += 1;
        }

        switch (command.kind) {
            .op => summary.op_commands += 1,
            .row_chain => summary.row_chains += 1,
            .rope_chain => summary.rope_chains += 1,
            .rope_batch => summary.rope_batches += 1,
            .rope_store_group => {
                summary.rope_store_groups += 1;
                summary.rope_store_group_ops += command.anchor_count;
                summary.rope_store_group_sidecars += command.sidecar_count;
            },
            .movement_batch => summary.movement_batches += 1,
            .movement_group => {
                summary.movement_groups += 1;
                summary.movement_group_ops += command.anchor_count;
            },
            .attention_chain => {
                summary.attention_chains += 1;
                summary.attention_chain_sidecars += command.sidecar_count;
            },
            .attention_store_chain => {
                summary.attention_store_chains += 1;
                summary.attention_store_chain_sidecars += command.sidecar_count;
            },
            .attention_store_group => {
                summary.attention_store_groups += 1;
                summary.attention_store_group_ops += command.anchor_count;
                summary.attention_store_group_sidecars += command.sidecar_count;
            },
            .rope_attention_store_chain => {
                summary.rope_attention_store_chains += 1;
                summary.rope_attention_store_chain_sidecars += command.sidecar_count;
            },
            .rope_attention_store_group => {
                summary.rope_attention_store_groups += 1;
                summary.rope_attention_store_group_ops += command.anchor_count;
                summary.rope_attention_store_group_sidecars += command.sidecar_count;
            },
            .attention_group => {
                summary.attention_groups += 1;
                summary.attention_group_ops += command.anchor_count;
            },
            .elementwise_batch => {
                summary.elementwise_batches += 1;
                summary.elementwise_ops += command.anchor_count;
            },
            .repeat_fused_elementwise_chain => summary.repeat_fused_elementwise_chains += 1,
            .projection_pair_elementwise_chain,
            .dense_projection_pair_fused_elementwise_chain,
            .projection_pair_fused_elementwise_chain,
            => summary.projection_pair_fused_elementwise_chains += 1,
            .projection_row_chain => summary.projection_row_chains += 1,
            .dense_projection_row_chain => summary.dense_projection_row_chains += 1,
            .dense_projection_chain => {
                summary.projection_chains += 1;
                summary.dense_projection_chains += 1;
                summary.projection_chain_sidecars += command.sidecar_count;
            },
            .projection_chain => {
                summary.projection_chains += 1;
                summary.quantized_projection_chains += 1;
                summary.projection_chain_sidecars += command.sidecar_count;
            },
            .projection_group => {
                summary.projection_groups += 1;
                summary.projection_anchors += command.anchor_count;
                summary.projection_sidecars += command.sidecar_count;
                summary.max_projection_span_ops = @max(summary.max_projection_span_ops, command.op_count);
            },
            .dense_projection_cache_group,
            .projection_cache_group,
            => {
                summary.projection_cache_groups += 1;
                summary.projection_cache_anchors += command.anchor_count;
                summary.projection_cache_sidecars += command.sidecar_count;
                summary.max_projection_span_ops = @max(summary.max_projection_span_ops, command.op_count);
            },
        }
    }
    return summary;
}

pub fn markProgramCommandUsed(used: []bool, command: ProgramCommand) void {
    var indices = command.coveredIndexIterator();
    while (indices.next()) |idx| {
        if (idx < used.len) used[idx] = true;
    }
}

fn commandRangeTouchesUsed(op_start: u32, op_count: u32, used: ?[]const bool) bool {
    const used_ops = used orelse return false;
    const start: usize = @intCast(op_start);
    if (start >= used_ops.len) return true;
    const end = @min(used_ops.len, start + @as(usize, op_count));
    for (used_ops[start..end]) |slot| {
        if (slot) return true;
    }
    return false;
}

fn projectionMatchesPolicy(q: anytype, policy: ProjectionGroupPolicy) bool {
    return switch (policy.kind) {
        .qmatvec => q.M == 1,
        .qmatmul => q.M != 1,
    };
}

fn opReadsBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return switch (op) {
        .elementwise => |e| e.src0 == buf or e.src1 == buf,
        .matmul => |m| m.a == buf or m.b == buf,
        .qmatmul => |q| q.input == buf,
        .conv2d => |c| c.src == buf or c.weight == buf or c.bias == buf,
        .softmax => |s| s.src == buf,
        .logsoftmax => |s| s.src == buf,
        .layernorm => |l| l.src == buf,
        .rmsnorm => |r| r.src == buf,
        .reduce => |r| r.src == buf,
        .max_pool2d => |mp| mp.src == buf,
        .avg_pool2d => |mp| mp.src == buf,
        .repeat => |rp| rp.src == buf,
        .gather_rows => |g| g.src == buf or g.indices == buf,
        .slice_assign => |sa| sa.src == buf,
        .rope => |rr| rr.src == buf or rr.cos_sin == buf,
        .attention => |att| att.q == buf or att.k == buf or att.v == buf or att.mask == buf,
        .fused_elementwise => |fe| {
            if (fe.src == buf) return true;
            for (fe.steps) |step| {
                if (step.op.isBinary() and step.secondary_buf == buf) return true;
            }
            return false;
        },
    };
}

fn opWritesBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return switch (op) {
        .elementwise => |e| e.dst == buf,
        .matmul => |m| m.dst == buf,
        .qmatmul => |q| q.dst == buf,
        .conv2d => |c| c.dst == buf,
        .softmax => |s| s.dst == buf,
        .logsoftmax => |s| s.dst == buf,
        .layernorm => |l| l.dst == buf,
        .rmsnorm => |r| r.dst == buf,
        .reduce => |r| r.dst == buf,
        .max_pool2d => |mp| mp.dst == buf,
        .avg_pool2d => |mp| mp.dst == buf,
        .repeat => |rp| rp.dst == buf,
        .gather_rows => |g| g.dst == buf,
        .slice_assign => |sa| sa.dst == buf,
        .rope => |rr| rr.dst == buf,
        .attention => |att| att.dst == buf,
        .fused_elementwise => |fe| fe.dst == buf,
    };
}

fn opTouchesBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return opReadsBuffer(op, buf) or opWritesBuffer(op, buf);
}

fn bufferSpan(buf: u16, offset: anytype, len: anytype) BufferSpan {
    const start: u64 = @intCast(offset);
    const n: u64 = @intCast(len);
    return .{ .buf = buf, .start = start, .end = start + n };
}

fn stridedSpan(buf: u16, offset: anytype, rows: anytype, cols: anytype, row_stride: anytype, col_stride: anytype) BufferSpan {
    const start: u64 = @intCast(offset);
    const r: u64 = @intCast(rows);
    const c: u64 = @intCast(cols);
    if (r == 0 or c == 0) return .{ .buf = buf, .start = start, .end = start };
    const rs: u64 = @intCast(row_stride);
    const cs: u64 = @intCast(col_stride);
    const last = start + (r - 1) * rs + (c - 1) * cs;
    return .{ .buf = buf, .start = start, .end = last + 1 };
}

fn strided4Span(buf: u16, offset: anytype, ne: [4]u32, strides: [4]u32) BufferSpan {
    const start: u64 = @intCast(offset);
    var last = start;
    for (ne, 0..) |extent, i| {
        if (extent == 0) return .{ .buf = buf, .start = start, .end = start };
        last += @as(u64, extent - 1) * @as(u64, strides[i]);
    }
    return .{ .buf = buf, .start = start, .end = last + 1 };
}

fn opAccessSpans(op: backend_mod.DeviceOp) OpAccessSpans {
    var access = OpAccessSpans{};
    switch (op) {
        .elementwise => |e| {
            access.addRead(bufferSpan(e.src0, e.src0_offset, e.n));
            if (e.op.isBinary()) access.addRead(bufferSpan(e.src1, e.src1_offset, e.n));
            access.addWrite(bufferSpan(e.dst, e.dst_offset, e.n));
        },
        .matmul => |m| {
            const g = m.geom;
            access.addRead(stridedSpan(m.a, g.a_offset, g.M, g.K, g.a_row_stride, g.a_col_stride));
            access.addRead(stridedSpan(m.b, g.b_offset, g.K, g.N, g.b_row_stride, g.b_col_stride));
            access.addWrite(stridedSpan(m.dst, g.dst_offset, g.M, g.N, g.dst_row_stride, 1));
        },
        .qmatmul => |q| {
            const input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K;
            access.addRead(stridedSpan(q.input, q.input_offset, q.M, q.K, input_row_stride, 1));
            access.addWrite(stridedSpan(q.dst, q.dst_offset, q.M, q.N, qmatmulDstRowStride(q), 1));
        },
        .conv2d => |c| {
            access.addRead(bufferSpan(c.src, c.src_offset, c.in_w * c.in_h * c.in_channels * c.batch));
            access.addRead(bufferSpan(c.weight, c.weight_offset, c.kernel_w * c.kernel_h * c.in_channels * c.out_channels));
            if (c.bias != std.math.maxInt(u16)) access.addRead(bufferSpan(c.bias, c.bias_offset, c.out_channels));
            access.addWrite(bufferSpan(c.dst, c.dst_offset, c.out_w * c.out_h * c.out_channels * c.batch));
        },
        .softmax => |s| {
            access.addRead(bufferSpan(s.src, s.src_offset, s.rows * s.cols));
            access.addWrite(bufferSpan(s.dst, s.dst_offset, s.rows * s.cols));
        },
        .logsoftmax => |s| {
            access.addRead(bufferSpan(s.src, s.src_offset, s.rows * s.cols));
            access.addWrite(bufferSpan(s.dst, s.dst_offset, s.rows * s.cols));
        },
        .layernorm => |l| {
            access.addRead(bufferSpan(l.src, l.src_offset, l.rows * l.cols));
            access.addWrite(bufferSpan(l.dst, l.dst_offset, l.rows * l.cols));
        },
        .rmsnorm => |r| {
            access.addRead(bufferSpan(r.src, r.src_offset, r.rows * r.cols));
            access.addWrite(bufferSpan(r.dst, r.dst_offset, r.rows * r.cols));
        },
        .reduce => |r| {
            access.addRead(bufferSpan(r.src, r.src_offset, r.n_out * r.reduce_size));
            access.addWrite(bufferSpan(r.dst, r.dst_offset, r.n_out));
        },
        .max_pool2d => |mp| {
            access.addRead(bufferSpan(mp.src, mp.src_offset, mp.src_w * mp.src_h * mp.channels * mp.batch));
            access.addWrite(bufferSpan(mp.dst, mp.dst_offset, mp.out_w * mp.out_h * mp.channels * mp.batch));
        },
        .avg_pool2d => |mp| {
            access.addRead(bufferSpan(mp.src, mp.src_offset, mp.src_w * mp.src_h * mp.channels * mp.batch));
            access.addWrite(bufferSpan(mp.dst, mp.dst_offset, mp.out_w * mp.out_h * mp.channels * mp.batch));
        },
        .repeat => |rp| {
            access.addRead(strided4Span(rp.src, rp.src_offset, rp.src_ne, rp.src_strides));
            access.addWrite(strided4Span(rp.dst, rp.dst_offset, rp.dst_ne, rp.dst_strides));
        },
        .gather_rows => |g| {
            const src_row_stride = if (g.src_row_stride != 0) g.src_row_stride else g.width;
            const dst_row_stride = if (g.dst_row_stride != 0) g.dst_row_stride else g.width;
            access.addRead(stridedSpan(g.src, g.src_offset, g.width, g.src_rows, 1, src_row_stride));
            access.addRead(bufferSpan(g.indices, g.indices_offset, g.count));
            access.addWrite(stridedSpan(g.dst, g.dst_offset, g.width, g.count, 1, dst_row_stride));
        },
        .slice_assign => |sa| {
            access.addRead(stridedSpan(sa.src, sa.src_offset, sa.rows, sa.cols, sa.src_row_stride, sa.src_col_stride));
            access.addWrite(stridedSpan(sa.dst, sa.dst_offset, sa.rows, sa.cols, sa.dst_row_stride, sa.dst_col_stride));
        },
        .rope => |rr| {
            const d = rr.half_d * 2;
            access.addRead(stridedSpan(rr.src, rr.src_off, d, rr.seq_len, rr.src_rs, rr.src_cs));
            access.addRead(stridedSpan(rr.cos_sin, rr.cs_off, d, rr.seq_len, 1, rr.cs_cs));
            access.addWrite(stridedSpan(rr.dst, rr.dst_off, d, rr.seq_len, 1, d));
        },
        .attention => |att| {
            access.addRead(stridedSpan(att.q, att.q_off, att.d_head, att.seq_q, att.q_rs, att.q_cs));
            access.addRead(stridedSpan(att.k, att.k_off, att.d_head, att.seq_kv, att.k_rs, att.k_cs));
            access.addRead(stridedSpan(att.v, att.v_off, att.d_head, att.seq_kv, att.v_rs, att.v_cs));
            if (att.has_mask) access.addRead(stridedSpan(att.mask, att.mask_off, att.seq_kv, att.seq_q, att.mask_rs, att.mask_cs));
            access.addWrite(stridedSpan(att.dst, att.dst_off, att.d_head, att.seq_q, att.dst_rs, att.dst_cs));
        },
        .fused_elementwise => |fe| {
            access.addRead(bufferSpan(fe.src, fe.src_offset, fe.n));
            for (fe.steps) |step| {
                if (step.op.isBinary()) access.addRead(bufferSpan(step.secondary_buf, step.secondary_offset, fe.n));
            }
            access.addWrite(bufferSpan(fe.dst, fe.dst_offset, fe.n));
        },
    }
    return access;
}

fn opReadsSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.readSpans()) |read| {
        if (read.overlaps(target)) return true;
    }
    return access.read_overflow and opReadsBuffer(op, target.buf);
}

fn opWritesSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.writeSpans()) |write| {
        if (write.overlaps(target)) return true;
    }
    return access.write_overflow and opWritesBuffer(op, target.buf);
}

fn opWritesCoverSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.writeSpans()) |write| {
        if (write.buf == target.buf and write.start <= target.start and write.end >= target.end) return true;
    }
    return access.write_overflow and opWritesBuffer(op, target.buf);
}

fn opTouchesSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    return opReadsSpan(op, target) or opWritesSpan(op, target);
}

fn opAccessConflicts(a: backend_mod.DeviceOp, b: backend_mod.DeviceOp) bool {
    const a_access = opAccessSpans(a);
    const b_access = opAccessSpans(b);
    for (a_access.writeSpans()) |write| {
        for (b_access.writeSpans()) |other_write| {
            if (write.overlaps(other_write)) return true;
        }
        for (b_access.readSpans()) |read| {
            if (write.overlaps(read)) return true;
        }
        if ((b_access.read_overflow and opReadsBuffer(b, write.buf)) or (b_access.write_overflow and opWritesBuffer(b, write.buf))) return true;
    }
    for (a_access.readSpans()) |read| {
        for (b_access.writeSpans()) |write| {
            if (read.overlaps(write)) return true;
        }
        if (b_access.write_overflow and opWritesBuffer(b, read.buf)) return true;
    }
    if (a_access.read_overflow or a_access.write_overflow) {
        const b_may_touch_overflowed_buffer = for (b_access.readSpans()) |read| {
            if (opTouchesBuffer(a, read.buf)) break true;
        } else for (b_access.writeSpans()) |write| {
            if (opTouchesBuffer(a, write.buf)) break true;
        } else false;
        if (b_may_touch_overflowed_buffer) return true;
    }
    return false;
}

fn canHoistOpTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    candidate: backend_mod.DeviceOp,
) bool {
    const candidate_access = opAccessSpans(candidate);
    for (ops[start..candidate_index]) |op| {
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

pub fn canFuseAttentionStoreSidecar(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    sidecar_index: usize,
    sa: anytype,
) bool {
    if (attention_index >= sidecar_index or sidecar_index > ops.len) return false;
    const candidate_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[attention_index + 1 .. sidecar_index]) |op| {
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opReadsSpan(op, write)) return false;
            if (opWritesSpan(op, write)) {
                const other_sa = switch (op) {
                    .slice_assign => |other| other,
                    else => return false,
                };
                if (sliceAssignWritesMayOverlap(other_sa, sa)) return false;
            }
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, .{ .slice_assign = sa })) return false;
        }
    }
    return true;
}

fn sliceAssignWritesMayOverlap(a: anytype, b: anytype) bool {
    if (a.dst != b.dst) return false;
    if (a.dst_row_stride == 1 and
        b.dst_row_stride == 1 and
        a.dst_col_stride == b.dst_col_stride and
        a.dst_col_stride > 0)
    {
        const stride: i64 = @intCast(a.dst_col_stride);
        const diff: i64 = @as(i64, @intCast(b.dst_offset)) - @as(i64, @intCast(a.dst_offset));
        const min_delta = -@as(i64, @intCast(a.cols)) + 1;
        const max_delta = @as(i64, @intCast(b.cols)) - 1;
        var delta = min_delta;
        while (delta <= max_delta) : (delta += 1) {
            const start_delta = diff + delta * stride;
            if (start_delta < @as(i64, @intCast(a.rows)) and -start_delta < @as(i64, @intCast(b.rows))) return true;
        }
        return false;
    }
    const a_span = stridedSpan(a.dst, a.dst_offset, a.rows, a.cols, a.dst_row_stride, a.dst_col_stride);
    const b_span = stridedSpan(b.dst, b.dst_offset, b.rows, b.cols, b.dst_row_stride, b.dst_col_stride);
    return a_span.overlaps(b_span);
}

fn opConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    candidate: backend_mod.DeviceOp,
) bool {
    for (indices) |idx| {
        if (opAccessConflicts(ops[idx], candidate)) return true;
    }
    return false;
}

fn canHoistProjectionTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    q: anytype,
) bool {
    return canHoistOpTo(ops, start, candidate_index, .{ .qmatmul = q });
}

pub fn canBatchElementwiseOp(e: anytype) bool {
    return e.op.isFusible();
}

fn canHoistElementwiseTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    e: anytype,
) bool {
    return canHoistOpTo(ops, start, candidate_index, .{ .elementwise = e });
}

fn elementwiseConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    e: anytype,
) bool {
    return opConflictsSelected(ops, indices, .{ .elementwise = e });
}

fn projectionConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    q: anytype,
) bool {
    return opConflictsSelected(ops, indices, .{ .qmatmul = q });
}

fn qmatvecSiblingProjectionAnchor(q: anytype) bool {
    return q.M == 1 and q.N != 0 and q.K != 0 and (q.dst_row_stride == 0 or q.dst_row_stride == q.N);
}

fn qmatvecSiblingProjectionCompatible(left: anytype, right: anytype) bool {
    return qmatvecSiblingProjectionAnchor(left) and
        qmatvecSiblingProjectionAnchor(right) and
        left.input == right.input and
        left.N == right.N and
        left.K == right.K and
        left.input_offset == right.input_offset and
        left.input_row_stride == right.input_row_stride;
}

pub fn qmatvecPairElementwiseCompatible(left: anytype, right: anytype, e: anytype) bool {
    if (!qmatvecSiblingProjectionCompatible(left, right)) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != left.N) return false;
    const left_src0 = e.src0 == left.dst and e.src0_offset == left.dst_offset;
    const left_src1 = e.src1 == left.dst and e.src1_offset == left.dst_offset;
    const right_src0 = e.src0 == right.dst and e.src0_offset == right.dst_offset;
    const right_src1 = e.src1 == right.dst and e.src1_offset == right.dst_offset;
    return (left_src0 and right_src1) or (right_src0 and left_src1);
}

fn anchorWritesCoverRead(ops: []const backend_mod.DeviceOp, command: *const ProgramCommand, read: BufferSpan) bool {
    for (command.anchorIndices()) |idx| {
        if (idx >= ops.len) return false;
        if (opWriteCoversRead(ops[idx], read)) return true;
    }
    return false;
}

fn canHoistSiblingElementwiseToCommand(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    sidecar_index: usize,
    sidecar: backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    const sidecar_access = opAccessSpans(sidecar);
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        for (sidecar_access.readSpans()) |read| {
            if (anchorWritesCoverRead(ops, command, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.read_overflow or sidecar_access.write_overflow) {
            if (opAccessConflicts(op, sidecar)) return false;
        }
    }
    return true;
}

fn projectionPairElementwiseHasExternalUsers(ops: []const backend_mod.DeviceOp, command: *const ProgramCommand) bool {
    if (command.anchor_count != 2 or command.sidecar_count != 1) return true;
    const sidecar_index = command.sidecar_indices[0] orelse return true;
    const carried = [_]?usize{sidecar_index};
    for (command.anchorIndices()) |idx| {
        if (idx >= ops.len) return true;
        if (projectionPrimaryOutputHasExternalUsersExcept(ops, idx, &carried)) return true;
    }
    return false;
}

fn denseMatvecProjectionAnchor(m: anytype) bool {
    const g = m.geom;
    return g.M == 1 and denseProjectionAnchor(m);
}

fn denseProjectionAnchor(m: anytype) bool {
    const g = m.geom;
    return g.M != 0 and g.N != 0 and g.K != 0 and g.dst_row_stride == g.N;
}

fn denseProjectionGroupCompatible(first: anytype, next: anytype) bool {
    const a = first.geom;
    const b = next.geom;
    return denseProjectionAnchor(first) and
        denseProjectionAnchor(next) and
        a.M == b.M and
        first.a == next.a and
        a.K == b.K and
        a.a_offset == b.a_offset and
        a.a_row_stride == b.a_row_stride and
        a.a_col_stride == b.a_col_stride;
}

fn projectionSidecarCompatible(q: anytype, op: backend_mod.DeviceOp) bool {
    return switch (op) {
        .slice_assign => |sa| if (q.M == 1) qmatvecSliceSidecarCompatible(q, sa) else qmatmulSliceSidecarCompatible(q, sa),
        .elementwise => |e| if (q.M == 1) qmatvecElementwiseSidecarCompatible(q, e) else qmatmulElementwiseSidecarCompatible(q, e),
        .fused_elementwise => |fe| if (q.M == 1) qmatvecFusedElementwiseSidecarCompatible(q, fe) else qmatmulFusedElementwiseSidecarCompatible(q, fe),
        else => false,
    };
}

fn projectionSidecarMatchesPolicy(policy: ProjectionGroupPolicy, q: anytype, op: backend_mod.DeviceOp) bool {
    return switch (policy.kind) {
        .qmatvec => switch (op) {
            .slice_assign => |sa| qmatvecSliceSidecarCompatible(q, sa),
            .elementwise => |e| qmatvecElementwiseSidecarCompatible(q, e),
            else => false,
        },
        .qmatmul => switch (op) {
            .slice_assign => |sa| qmatmulSliceSidecarCompatible(q, sa),
            .elementwise => |e| qmatmulElementwiseSidecarCompatible(q, e),
            else => false,
        },
    };
}

fn projectionPrimarySidecarCompatible(q: anytype, op: backend_mod.DeviceOp) bool {
    return projectionSidecarCompatible(q, op) or switch (op) {
        .rope => |rr| projectionRopeSidecarCompatible(q, rr),
        else => false,
    };
}

fn matmulPrimarySidecarCompatible(m: anytype, op: backend_mod.DeviceOp) bool {
    return switch (op) {
        .slice_assign => |sa| denseProjectionSliceSidecarCompatible(m, sa),
        .elementwise => |e| matmulElementwiseSidecarCompatible(m, e),
        .fused_elementwise => |fe| matmulFusedElementwiseSidecarCompatible(m, fe),
        .rope => |rr| denseMatvecRopeSidecarCompatible(m, rr),
        else => false,
    };
}

fn projectionSelectionContainsIndex(selection: *const ProjectionGroupSelection, candidate: usize) bool {
    for (selection.anchorIndices()) |idx| {
        if (idx == candidate) return true;
    }
    for (selection.sidecarIndices()) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

fn canHoistProjectionSidecarToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    sidecar_index: usize,
    q: anytype,
    sidecar: backend_mod.DeviceOp,
    selection: *const ProjectionGroupSelection,
) bool {
    const sidecar_access = opAccessSpans(sidecar);
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (projectionSelectionContainsIndex(selection, idx)) continue;
        for (sidecar_access.readSpans()) |read| {
            if (opWriteCoversRead(.{ .qmatmul = q }, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.read_overflow or sidecar_access.write_overflow) {
            if (opAccessConflicts(op, sidecar)) return false;
        }
    }
    return true;
}

pub fn qmatmulElementwiseSidecarCompatible(q: anytype, e: anytype) bool {
    if (q.M == 1) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != q.M * q.N) return false;
    if (qmatmulDstRowStride(q) != q.N) return false;
    return (e.src0 == q.dst and e.src0_offset == q.dst_offset) or
        (e.src1 == q.dst and e.src1_offset == q.dst_offset);
}

pub fn matmulElementwiseSidecarCompatible(m: anytype, e: anytype) bool {
    const g = m.geom;
    if (g.M == 0 or g.N == 0 or g.dst_row_stride != g.N) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != g.M * g.N) return false;
    const src0_primary = e.src0 == m.dst and e.src0_offset == g.dst_offset;
    const src1_primary = e.src1 == m.dst and e.src1_offset == g.dst_offset;
    return src0_primary != src1_primary;
}

pub fn matmulRepeatElementwiseBiasCompatible(m: anytype, repeat_op: backend_mod.DeviceOp, elementwise_op: backend_mod.DeviceOp) bool {
    const rp = switch (repeat_op) {
        .repeat => |rp| rp,
        else => return false,
    };
    const e = switch (elementwise_op) {
        .elementwise => |e| e,
        else => return false,
    };
    const g = m.geom;
    if (g.M == 0 or g.N == 0 or g.dst_row_stride != g.N) return false;
    if (e.op != .add) return false;
    if (e.n != g.M * g.N or rp.n != e.n) return false;
    if (rp.dst != e.dst or rp.dst_offset != e.dst_offset) return false;

    const src0_primary = e.src0 == m.dst and e.src0_offset == g.dst_offset;
    const src1_primary = e.src1 == m.dst and e.src1_offset == g.dst_offset;
    const src0_repeat = e.src0 == rp.dst and e.src0_offset == rp.dst_offset;
    const src1_repeat = e.src1 == rp.dst and e.src1_offset == rp.dst_offset;
    if (src0_primary == src1_primary) return false;
    if (src0_repeat == src1_repeat) return false;
    if (src0_primary == src0_repeat) return false;

    if (rp.src_ne[0] != g.N) return false;
    if (rp.dst_ne[0] != g.N or rp.dst_ne[1] != g.M) return false;
    if (rp.src_strides[0] != 1 or rp.dst_strides[0] != 1) return false;
    if (rp.dst_ne[1] > 1 and rp.dst_strides[1] != g.N) return false;
    return true;
}

pub fn matmulRepeatElementwiseBiasActivationCompatible(
    m: anytype,
    repeat_op: backend_mod.DeviceOp,
    bias_op: backend_mod.DeviceOp,
    activation_op: backend_mod.DeviceOp,
) bool {
    if (!matmulRepeatElementwiseBiasCompatible(m, repeat_op, bias_op)) return false;
    const bias = switch (bias_op) {
        .elementwise => |e| e,
        else => return false,
    };
    const activation = switch (activation_op) {
        .elementwise => |e| e,
        else => return false,
    };
    if (activation.op != .relu and activation.op != .gelu and activation.op != .silu) return false;
    if (activation.n != bias.n) return false;
    return activation.src0 == bias.dst and activation.src0_offset == bias.dst_offset;
}

pub fn matmulFusedElementwiseSidecarCompatible(m: anytype, fe: anytype) bool {
    const g = m.geom;
    if (g.M == 0 or g.N == 0) return false;
    if (fe.n != g.M * g.N) return false;
    if (g.dst_row_stride != g.N) return false;
    return fe.src == m.dst and fe.src_offset == g.dst_offset;
}

pub fn qmatvecElementwiseSidecarCompatible(q: anytype, e: anytype) bool {
    if (q.M != 1) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != q.N) return false;
    if (q.dst_row_stride != 0 and q.dst_row_stride != q.N) return false;
    const src0_primary = e.src0 == q.dst and e.src0_offset == q.dst_offset;
    const src1_primary = e.src1 == q.dst and e.src1_offset == q.dst_offset;
    return src0_primary != src1_primary;
}

pub fn qmatvecFusedElementwiseSidecarCompatible(q: anytype, fe: anytype) bool {
    if (q.M != 1) return false;
    if (fe.n != q.N) return false;
    if (q.dst_row_stride != 0 and q.dst_row_stride != q.N) return false;
    return fe.src == q.dst and fe.src_offset == q.dst_offset;
}

pub fn qmatmulFusedElementwiseSidecarCompatible(q: anytype, fe: anytype) bool {
    if (q.M == 1) return false;
    if (fe.n != q.M * q.N) return false;
    if (qmatmulDstRowStride(q) != q.N) return false;
    return fe.src == q.dst and fe.src_offset == q.dst_offset;
}

pub fn qmatmulSliceSrcColStart(q: anytype, sa: anytype) ?u32 {
    if (sa.src_offset < q.dst_offset) return null;
    const delta = sa.src_offset - q.dst_offset;
    const dst_row_stride = qmatmulDstRowStride(q);
    if (delta >= dst_row_stride) return null;
    return delta;
}

pub fn qmatmulSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    const slice_src_col_start = qmatmulSliceSrcColStart(q, sa) orelse return false;
    return q.M != 1 and
        q.dst == sa.src and
        slice_src_col_start + sa.rows <= q.N and
        q.M == sa.cols and
        sa.src_row_stride == 1 and
        sa.src_col_stride == qmatmulDstRowStride(q);
}

pub fn qmatvecSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    const slice_src_col_start = qmatmulSliceSrcColStart(q, sa) orelse return false;
    const slice_len = @as(u64, sa.rows) * @as(u64, sa.cols);
    return q.M == 1 and
        q.dst == sa.src and
        sa.rows != 0 and
        sa.cols != 0 and
        @as(u64, slice_src_col_start) + slice_len <= @as(u64, q.N) and
        sa.src_row_stride == 1 and
        (sa.src_col_stride == sa.rows or (sa.cols == 1 and sa.src_col_stride == q.N));
}

pub fn denseMatvecSliceSrcColStart(m: anytype, sa: anytype) ?u32 {
    const g = m.geom;
    if (sa.src_offset < g.dst_offset) return null;
    const delta = sa.src_offset - g.dst_offset;
    if (delta >= g.dst_row_stride) return null;
    return @intCast(delta);
}

pub fn denseMatvecSliceSidecarCompatible(m: anytype, sa: anytype) bool {
    const slice_src_col_start = denseMatvecSliceSrcColStart(m, sa) orelse return false;
    const g = m.geom;
    const slice_len = @as(u64, sa.rows) * @as(u64, sa.cols);
    return denseMatvecProjectionAnchor(m) and
        m.dst == sa.src and
        sa.rows != 0 and
        sa.cols != 0 and
        @as(u64, slice_src_col_start) + slice_len <= @as(u64, g.N) and
        sa.src_row_stride == 1 and
        (sa.src_col_stride == sa.rows or (sa.cols == 1 and sa.src_col_stride == g.N));
}

pub fn denseProjectionSliceSidecarCompatible(m: anytype, sa: anytype) bool {
    const slice_src_col_start = denseMatvecSliceSrcColStart(m, sa) orelse return false;
    const g = m.geom;
    if (g.M == 1) return denseMatvecSliceSidecarCompatible(m, sa);
    return denseProjectionAnchor(m) and
        m.dst == sa.src and
        sa.rows != 0 and
        sa.cols == g.M and
        slice_src_col_start + sa.rows <= g.N and
        sa.src_row_stride == 1 and
        sa.src_col_stride == g.dst_row_stride;
}

pub fn qmatmulRopeSrcColStart(q: anytype, rr: anytype) ?u32 {
    if (rr.src_off < q.dst_offset) return null;
    const delta = rr.src_off - q.dst_offset;
    const dst_row_stride = qmatmulDstRowStride(q);
    if (delta >= dst_row_stride) return null;
    return delta;
}

fn projectionRopeSidecarCompatible(q: anytype, rr: anytype) bool {
    const rope_src_col_start = qmatmulRopeSrcColStart(q, rr) orelse return false;
    const d = rr.half_d * 2;
    return q.dst == rr.src and
        rope_src_col_start + d <= q.N and
        q.M == rr.seq_len and
        rr.src_rs == 1 and
        rr.src_cs == qmatmulDstRowStride(q);
}

pub fn qmatvecRopeStoreSidecarCompatible(q: anytype, rr: anytype, sa: anytype) bool {
    return q.M == 1 and projectionRopeSidecarCompatible(q, rr) and ropeSliceAssignCompatible(rr, sa);
}

pub fn denseMatvecRopeSrcColStart(m: anytype, rr: anytype) ?u32 {
    const g = m.geom;
    if (rr.src_off < g.dst_offset) return null;
    const delta = rr.src_off - g.dst_offset;
    if (delta >= g.dst_row_stride) return null;
    return @intCast(delta);
}

fn denseMatvecRopeSidecarCompatible(m: anytype, rr: anytype) bool {
    const rope_src_col_start = denseMatvecRopeSrcColStart(m, rr) orelse return false;
    const g = m.geom;
    const d = rr.half_d * 2;
    return denseMatvecProjectionAnchor(m) and
        m.dst == rr.src and
        rope_src_col_start + d <= g.N and
        rr.seq_len == 1 and
        rr.src_rs == 1 and
        rr.src_cs == g.dst_row_stride;
}

pub fn denseMatvecRopeStoreSidecarCompatible(m: anytype, rr: anytype, sa: anytype) bool {
    return denseMatvecRopeSidecarCompatible(m, rr) and ropeSliceAssignCompatible(rr, sa);
}

fn qmatmulDstRowStride(q: anytype) u32 {
    return if (q.dst_row_stride != 0) q.dst_row_stride else q.N;
}

const AttentionOperand = enum { q, k, v };

pub fn attentionSliceAssignOperand(sa: anytype, att: anytype) ?AttentionOperand {
    if (attentionSliceMatches(sa, att.q, att.q_off, att.d_head, att.seq_q, att.q_rs, att.q_cs)) return .q;
    if (attentionSliceMatches(sa, att.k, att.k_off, att.d_head, att.seq_kv, att.k_rs, att.k_cs)) return .k;
    if (attentionSliceMatches(sa, att.v, att.v_off, att.d_head, att.seq_kv, att.v_rs, att.v_cs)) return .v;
    return null;
}

pub fn attentionSliceStoreCompatible(att: anytype, sa: anytype) bool {
    return sa.src == att.dst and
        sa.src_offset == att.dst_off and
        sa.rows == att.d_head and
        sa.cols == att.seq_q and
        sa.src_row_stride == att.dst_rs and
        sa.src_col_stride == att.dst_cs;
}

pub fn ropeAttentionCompatible(rr: anytype, att: anytype) bool {
    const d = rr.half_d * 2;
    return rr.dst == att.q and
        rr.dst_off == att.q_off and
        d == att.d_head and
        rr.seq_len == att.seq_q and
        att.q_rs == 1 and
        att.q_cs == d;
}

pub fn ropeStoreBatchGeometryCompatible(first: anytype, next: anytype) bool {
    return first.half_d == next.half_d and
        first.seq_len == next.seq_len and
        first.src_rs == next.src_rs and
        first.src_cs == next.src_cs and
        first.cs_cs == next.cs_cs;
}

pub fn ropeAttentionStoreCompactBatchCompatible(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    if (command.kind != .rope_attention_store_group) return false;
    if (command.sidecar_count < 2 or command.anchor_count != command.sidecar_count * 2) return false;
    const first_sa_idx = command.sidecar_indices[0] orelse return false;
    if (command.indices[0] >= ops.len or command.indices[1] >= ops.len or first_sa_idx >= ops.len) return false;
    const first_rope = switch (ops[command.indices[0]]) {
        .rope => |rr| rr,
        else => return false,
    };
    const first_att = switch (ops[command.indices[1]]) {
        .attention => |att| att,
        else => return false,
    };
    const first_sa = switch (ops[first_sa_idx]) {
        .slice_assign => |sa| sa,
        else => return false,
    };

    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const rope_idx = command.indices[i * 2];
        const att_idx = command.indices[i * 2 + 1];
        const sa_idx = command.sidecar_indices[i] orelse return false;
        if (rope_idx >= ops.len or att_idx >= ops.len or sa_idx >= ops.len) return false;
        const rr = switch (ops[rope_idx]) {
            .rope => |rr| rr,
            else => return false,
        };
        const att = switch (ops[att_idx]) {
            .attention => |att| att,
            else => return false,
        };
        const sa = switch (ops[sa_idx]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!ropeAttentionCompatible(rr, att)) return false;
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) return false;
        if (!attentionGeometryCompatible(first_att, att)) return false;
        if (!attentionSliceStoreCompatible(att, sa)) return false;
        if (rr.src != first_rope.src or rr.cos_sin != first_rope.cos_sin or
            att.k != first_att.k or att.v != first_att.v or att.mask != first_att.mask or
            sa.dst != first_sa.dst) return false;
        if (sa.dst_row_stride != first_sa.dst_row_stride or
            sa.dst_col_stride != first_sa.dst_col_stride) return false;
    }
    return true;
}

fn attentionSliceMatches(
    sa: anytype,
    buf: u16,
    offset: u32,
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
) bool {
    return sa.dst == buf and
        sa.dst_offset == offset and
        sa.rows == rows and
        sa.cols == cols and
        sa.dst_row_stride == row_stride and
        sa.dst_col_stride == col_stride;
}

fn isRopeSliceAssignChain(
    a: backend_mod.DeviceOp,
    b: backend_mod.DeviceOp,
) bool {
    const rr = switch (a) {
        .rope => |rr| rr,
        else => return false,
    };
    const sa = switch (b) {
        .slice_assign => |sa| sa,
        else => return false,
    };
    return ropeSliceAssignCompatible(rr, sa);
}

pub fn ropeSliceAssignCompatible(rr: anytype, sa: anytype) bool {
    const d = rr.half_d * 2;
    return rr.dst == sa.src and
        rr.dst_off == sa.src_offset and
        sa.rows == d and
        sa.cols == rr.seq_len and
        sa.src_row_stride == 1 and
        sa.src_col_stride == d;
}

pub fn ropeStoreGroupCompatible(first_rope: anytype, first_sa: anytype, next_rope: anytype, next_sa: anytype) bool {
    return first_rope.cos_sin == next_rope.cos_sin and
        first_sa.dst == next_sa.dst and
        first_rope.half_d == next_rope.half_d and
        first_rope.seq_len == next_rope.seq_len and
        first_rope.src_rs == next_rope.src_rs and
        first_rope.src_cs == next_rope.src_cs and
        first_rope.cs_cs == next_rope.cs_cs and
        first_sa.dst_row_stride == next_sa.dst_row_stride and
        first_sa.dst_col_stride == next_sa.dst_col_stride;
}

fn ropeBatchCompatible(first: anytype, next: anytype) bool {
    return first.src == next.src and
        first.cos_sin == next.cos_sin and
        first.dst == next.dst and
        first.half_d == next.half_d and
        first.seq_len == next.seq_len and
        first.src_rs == next.src_rs and
        first.src_cs == next.src_cs and
        first.cs_cs == next.cs_cs;
}

pub fn ropeBatchRunLen(ops: []const backend_mod.DeviceOp, max_ops: u32) usize {
    if (ops.len < 2 or max_ops < 2) return 0;
    const first = switch (ops[0]) {
        .rope => |rr| rr,
        else => return 0,
    };
    var n: usize = 1;
    const limit = @min(ops.len, @as(usize, @intCast(max_ops)));
    while (n < limit) : (n += 1) {
        const next = switch (ops[n]) {
            .rope => |rr| rr,
            else => break,
        };
        if (!ropeBatchCompatible(first, next)) break;
    }
    return if (n >= 2) n else 0;
}

pub fn sliceAssignBatchCompatible(first: anytype, next: anytype) bool {
    return first.src == next.src and first.dst == next.dst;
}

pub fn sliceAssignBatchRunLen(ops: []const backend_mod.DeviceOp, max_ops: u32) usize {
    if (ops.len < 2 or max_ops < 2) return 0;
    const first = switch (ops[0]) {
        .slice_assign => |sa| sa,
        else => return 0,
    };
    var n: usize = 1;
    const limit = @min(ops.len, @as(usize, @intCast(max_ops)));
    while (n < limit) : (n += 1) {
        const next = switch (ops[n]) {
            .slice_assign => |sa| sa,
            else => break,
        };
        if (!sliceAssignBatchCompatible(first, next)) break;
    }
    return if (n >= 2) n else 0;
}

pub fn attentionBatchCompatible(first: anytype, next: anytype) bool {
    return first.q == next.q and
        first.k == next.k and
        first.v == next.v and
        first.mask == next.mask and
        first.dst == next.dst and
        first.has_mask == next.has_mask and
        first.d_head == next.d_head and
        first.seq_q == next.seq_q and
        first.seq_kv == next.seq_kv and
        first.scale == next.scale and
        first.q_rs == next.q_rs and
        first.q_cs == next.q_cs and
        first.k_rs == next.k_rs and
        first.k_cs == next.k_cs and
        first.v_rs == next.v_rs and
        first.v_cs == next.v_cs and
        first.mask_rs == next.mask_rs and
        first.mask_cs == next.mask_cs and
        first.dst_rs == next.dst_rs and
        first.dst_cs == next.dst_cs;
}

pub fn attentionGeometryCompatible(first: anytype, next: anytype) bool {
    return first.has_mask == next.has_mask and
        first.d_head == next.d_head and
        first.seq_q == next.seq_q and
        first.seq_kv == next.seq_kv and
        first.scale == next.scale and
        first.q_rs == next.q_rs and
        first.q_cs == next.q_cs and
        first.k_rs == next.k_rs and
        first.k_cs == next.k_cs and
        first.v_rs == next.v_rs and
        first.v_cs == next.v_cs and
        first.mask_rs == next.mask_rs and
        first.mask_cs == next.mask_cs and
        first.dst_rs == next.dst_rs and
        first.dst_cs == next.dst_cs;
}

pub fn isRmsnormScaleChain(
    a: backend_mod.DeviceOp,
    b: backend_mod.DeviceOp,
    c: backend_mod.DeviceOp,
) bool {
    const rn = switch (a) {
        .rmsnorm => |rn| rn,
        else => return false,
    };
    const rp = switch (b) {
        .repeat => |rp| rp,
        else => return false,
    };
    const e = switch (c) {
        .elementwise => |e| e,
        else => return false,
    };

    const n = rn.rows * rn.cols;
    return e.op == .mul and
        rp.n == n and
        e.n == n and
        rp.dst == (if (e.src0 == rn.dst and e.src0_offset == rn.dst_offset) e.src1 else e.src0) and
        rp.src_ne[0] == rn.cols and
        rp.src_ne[1] == 1 and
        rp.dst_ne[0] == rn.cols and
        rp.dst_ne[1] == rn.rows and
        rp.src_strides[0] == 1 and
        rp.dst_strides[0] == 1 and
        rp.dst_strides[1] == rn.cols and
        mulSourcesMatchNormAndScale(e, rn.dst, rn.dst_offset, rp.dst, rp.dst_offset);
}

fn mulSourcesMatchNormAndScale(e: anytype, norm_buf: u16, norm_offset: u32, scale_buf: u16, scale_offset: u32) bool {
    return (e.src0 == norm_buf and e.src0_offset == norm_offset and e.src1 == scale_buf and e.src1_offset == scale_offset) or
        (e.src1 == norm_buf and e.src1_offset == norm_offset and e.src0 == scale_buf and e.src0_offset == scale_offset);
}

fn kernelFamily(op: backend_mod.DeviceOp) KernelFamily {
    return switch (op) {
        .elementwise => .elementwise,
        .fused_elementwise => .fused_elementwise,
        .softmax, .logsoftmax, .layernorm, .rmsnorm => .row,
        .reduce => .reduce,
        .repeat, .conv2d, .max_pool2d, .avg_pool2d, .gather_rows, .slice_assign => .movement,
        .matmul => .matmul,
        .qmatmul => |q| if (q.M == 1) .qmatvec else .qmatmul,
        .rope => .rope,
        .attention => .attention,
    };
}

fn executionClass(op: backend_mod.DeviceOp, policy: SchedulePolicy) ExecutionClass {
    if (!policy.capabilities.supportsOp(op)) return .fallback;

    const family = kernelFamily(op);
    if (!policy.native_kernels.supports(family)) return .fallback;

    const can_use_backend = switch (op) {
        .matmul => |m| policy.fine_grained or m.geom.M >= @as(usize, policy.min_backend_matmul_m),
        .qmatmul => |q| if (q.M == 1) true else policy.fine_grained or q.M >= policy.min_backend_qmatmul_m,
        .elementwise,
        .fused_elementwise,
        .softmax,
        .logsoftmax,
        .layernorm,
        .rmsnorm,
        .reduce,
        .max_pool2d,
        .avg_pool2d,
        .conv2d,
        .repeat,
        .gather_rows,
        .slice_assign,
        .rope,
        .attention,
        => policy.fine_grained,
    };

    return if (can_use_backend) .backend else .fallback;
}

fn buildKernelSchedule(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: SchedulePolicy,
) ![]KernelItem {
    var items: std.ArrayListUnmanaged(KernelItem) = .empty;
    errdefer items.deinit(alloc);

    for (ops, 0..) |op, i| {
        const next = KernelItem{
            .family = kernelFamily(op),
            .execution = executionClass(op, policy),
            .start = @intCast(i),
            .len = 1,
        };

        if (items.items.len > 0) {
            const last = &items.items[items.items.len - 1];
            if (last.family == next.family and
                last.execution == next.execution and
                last.start + last.len == next.start)
            {
                last.len += 1;
                continue;
            }
        }

        try items.append(alloc, next);
    }

    return items.toOwnedSlice(alloc);
}

fn scheduleShapeMatches(
    ops: []const backend_mod.DeviceOp,
    items: []const KernelItem,
    policy: SchedulePolicy,
) bool {
    var item_index: usize = 0;
    var current: ?KernelItem = null;

    for (ops, 0..) |op, i| {
        const next = KernelItem{
            .family = kernelFamily(op),
            .execution = executionClass(op, policy),
            .start = @intCast(i),
            .len = 1,
        };

        if (current) |*item| {
            if (item.family == next.family and
                item.execution == next.execution and
                item.start + item.len == next.start)
            {
                item.len += 1;
                continue;
            }

            if (item_index >= items.len or !kernelItemsEqual(item.*, items[item_index])) return false;
            item_index += 1;
            current = next;
        } else {
            current = next;
        }
    }

    if (current) |item| {
        if (item_index >= items.len or !kernelItemsEqual(item, items[item_index])) return false;
        item_index += 1;
    }

    return item_index == items.len;
}

fn kernelItemsEqual(a: KernelItem, b: KernelItem) bool {
    return a.family == b.family and
        a.execution == b.execution and
        a.start == b.start and
        a.len == b.len;
}

/// Split anchored member runs into non-overlapping windows with at least
/// `anchors_per_region` anchors. Schedule items are not split, so a coalesced
/// anchor item may make a region contain more anchors than requested.
fn buildAnchorWindowRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    policy: RegionPolicy,
    anchors_per_region: u32,
) ![]KernelRegion {
    var regions: std.ArrayListUnmanaged(KernelRegion) = .empty;
    errdefer regions.deinit(alloc);

    if (anchors_per_region == 0) return regions.toOwnedSlice(alloc);

    var run_start: usize = 0;
    while (run_start < items.len) {
        while (run_start < items.len and !policy.member_families.contains(items[run_start].family)) {
            run_start += 1;
        }
        if (run_start >= items.len) break;

        var run_end = run_start;
        while (run_end < items.len and policy.member_families.contains(items[run_end].family)) : (run_end += 1) {}

        var window_start = run_start;
        while (window_start < run_end) {
            var window_end = window_start;
            var anchor_count: u32 = 0;

            while (window_end < run_end and anchor_count < anchors_per_region) : (window_end += 1) {
                if (policy.anchor_families.contains(items[window_end].family)) {
                    anchor_count += items[window_end].len;
                }
            }

            if (anchor_count < anchors_per_region) break;

            while (window_end < run_end and !policy.anchor_families.contains(items[window_end].family)) : (window_end += 1) {}

            const first = items[window_start];
            const last = items[window_end - 1];
            const op_end = last.start + last.len;
            try regions.append(alloc, .{
                .start_item = @intCast(window_start),
                .item_count = @intCast(window_end - window_start),
                .op_start = first.start,
                .op_count = op_end - first.start,
                .anchor_count = anchor_count,
            });

            window_start = window_end;
        }

        run_start = run_end;
    }

    return regions.toOwnedSlice(alloc);
}

fn buildStagePatternRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stage: StagePolicy,
) ![]PatternRegion {
    var pattern_regions: std.ArrayListUnmanaged(PatternRegion) = .empty;
    errdefer pattern_regions.deinit(alloc);

    const regions = try buildAnchorWindowRegions(
        alloc,
        items,
        stage.region_policy,
        stage.anchors_per_stage,
    );
    defer alloc.free(regions);

    for (regions) |region| {
        if (region.item_count < stage.min_items_per_stage) continue;
        if (region.op_count < stage.min_ops_per_stage) continue;
        try pattern_regions.append(alloc, .{
            .pattern_index = stage.pattern_index,
            .region = region,
        });
    }

    return pattern_regions.toOwnedSlice(alloc);
}

fn buildStagePlan(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stages: []const StagePolicy,
) ![]PatternRegion {
    var candidates: std.ArrayListUnmanaged(PatternRegion) = .empty;
    defer candidates.deinit(alloc);

    for (stages) |stage| {
        const regions = try buildStagePatternRegions(alloc, items, stage);
        defer alloc.free(regions);
        for (regions) |region| {
            try candidates.append(alloc, region);
        }
    }

    return selectPatternRegions(alloc, candidates.items);
}

fn buildStageRegionSchedule(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stages: []const StagePolicy,
) ![]ScheduleUnit {
    const plan = try buildStagePlan(alloc, items, stages);
    defer alloc.free(plan);
    return buildRegionSchedule(alloc, items, plan);
}

fn selectPatternRegions(
    alloc: std.mem.Allocator,
    candidates: []const PatternRegion,
) ![]PatternRegion {
    const sorted = try alloc.dupe(PatternRegion, candidates);
    defer alloc.free(sorted);
    sortPatternRegions(sorted);

    var selected: std.ArrayListUnmanaged(PatternRegion) = .empty;
    errdefer selected.deinit(alloc);

    var next_free_item: u32 = 0;
    for (sorted) |candidate| {
        if (candidate.region.start_item < next_free_item) continue;
        try selected.append(alloc, candidate);
        next_free_item = candidate.region.start_item + candidate.region.item_count;
    }

    return selected.toOwnedSlice(alloc);
}

fn buildRegionSchedule(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    pattern_regions: []const PatternRegion,
) ![]ScheduleUnit {
    var units: std.ArrayListUnmanaged(ScheduleUnit) = .empty;
    errdefer units.deinit(alloc);

    var item_index: usize = 0;
    for (pattern_regions) |pattern_region| {
        const region = pattern_region.region;
        const region_start: usize = @intCast(region.start_item);
        if (region_start < item_index) continue;
        while (item_index < region_start) : (item_index += 1) {
            try units.append(alloc, itemScheduleUnit(@intCast(item_index), items[item_index]));
        }

        try units.append(alloc, .{
            .kind = .pattern_region,
            .pattern_index = pattern_region.pattern_index,
            .start_item = region.start_item,
            .item_count = region.item_count,
            .op_start = region.op_start,
            .op_count = region.op_count,
        });
        item_index = @intCast(region.start_item + region.item_count);
    }

    while (item_index < items.len) : (item_index += 1) {
        try units.append(alloc, itemScheduleUnit(@intCast(item_index), items[item_index]));
    }

    return units.toOwnedSlice(alloc);
}

const RegionKernelPlan = struct {
    kernel_plan: KernelPlan = .{},

    fn deinit(self: *RegionKernelPlan, alloc: std.mem.Allocator) void {
        self.kernel_plan.deinit(alloc);
    }
};

fn deinitRegionKernelPlans(alloc: std.mem.Allocator, plans: []RegionKernelPlan) void {
    for (plans) |*plan| plan.deinit(alloc);
}

fn buildRegionKernelPlans(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    units: []const ScheduleUnit,
    kernelizer: Kernelizer,
) ![]RegionKernelPlan {
    if (units.len == 0) return &.{};

    const plans = try alloc.alloc(RegionKernelPlan, units.len);
    errdefer alloc.free(plans);
    @memset(plans, .{});
    errdefer deinitRegionKernelPlans(alloc, plans);

    for (units, 0..) |unit, i| {
        if (unit.kind != .pattern_region or unit.op_count > 256) continue;
        const start: usize = @intCast(unit.op_start);
        const end = start + @as(usize, unit.op_count);
        if (end > ops.len) continue;
        plans[i].kernel_plan = try kernelizer.kernelize(alloc, ops[start..end]);
    }

    return plans;
}

pub const ExecutionPlan = struct {
    schedule: []const KernelItem = &.{},
    regions: []const ScheduleUnit = &.{},
    region_kernel_plans: []RegionKernelPlan = &.{},

    pub fn deinit(self: ExecutionPlan, alloc: std.mem.Allocator) void {
        if (self.schedule.len > 0) alloc.free(self.schedule);
        if (self.regions.len > 0) alloc.free(self.regions);
        deinitRegionKernelPlans(alloc, self.region_kernel_plans);
        if (self.region_kernel_plans.len > 0) alloc.free(self.region_kernel_plans);
    }

    pub fn regionCommandPlan(self: ExecutionPlan, unit_index: usize) []const ProgramCommand {
        if (unit_index >= self.region_kernel_plans.len) return &.{};
        return self.region_kernel_plans[unit_index].kernel_plan.commands;
    }

    pub fn executableCommandShape(self: ExecutionPlan) !ProgramCommandStreamShape {
        var shape = ProgramCommandStreamShape{};
        var h = RuntimeStencilHasher{};
        var saw_command = false;
        for (self.region_kernel_plans) |region_plan| {
            const commands = region_plan.kernel_plan.commands;
            if (commands.len == 0) continue;
            saw_command = true;
            const summary = summarizeProgramCommands(commands);
            shape.command_count = try std.math.add(u32, shape.command_count, @intCast(commands.len));
            shape.covered_ops = try std.math.add(u32, shape.covered_ops, summary.covered_ops);
            shape.estimated_saved_dispatches = try std.math.add(u32, shape.estimated_saved_dispatches, summary.estimated_saved_dispatches);
            shape.row_chains = try std.math.add(u32, shape.row_chains, summary.row_chains);
            shape.projection_row_chains = try std.math.add(u32, shape.projection_row_chains, summary.projection_row_chains);
            shape.dense_projection_row_chains = try std.math.add(u32, shape.dense_projection_row_chains, summary.dense_projection_row_chains);
            shape.projection_chains = try std.math.add(u32, shape.projection_chains, summary.projection_chains);
            shape.dense_projection_chains = try std.math.add(u32, shape.dense_projection_chains, summary.dense_projection_chains);
            shape.quantized_projection_chains = try std.math.add(u32, shape.quantized_projection_chains, summary.quantized_projection_chains);
            shape.projection_chain_sidecars = try std.math.add(u32, shape.projection_chain_sidecars, summary.projection_chain_sidecars);
            shape.projection_chain_row_chain_frontiers = try std.math.add(u32, shape.projection_chain_row_chain_frontiers, summary.projection_chain_row_chain_frontiers);
            shape.projection_groups = try std.math.add(u32, shape.projection_groups, summary.projection_groups);
            shape.projection_anchors = try std.math.add(u32, shape.projection_anchors, summary.projection_anchors);
            shape.projection_sidecars = try std.math.add(u32, shape.projection_sidecars, summary.projection_sidecars);
            shape.projection_cache_groups = try std.math.add(u32, shape.projection_cache_groups, summary.projection_cache_groups);
            shape.projection_cache_anchors = try std.math.add(u32, shape.projection_cache_anchors, summary.projection_cache_anchors);
            shape.projection_cache_sidecars = try std.math.add(u32, shape.projection_cache_sidecars, summary.projection_cache_sidecars);
            shape.max_projection_span_ops = @max(shape.max_projection_span_ops, summary.max_projection_span_ops);
            for (commands) |command| {
                const kind_index = @intFromEnum(command.kind);
                shape.command_kind_counts[kind_index] = try std.math.add(u32, shape.command_kind_counts[kind_index], 1);
                addProgramCommandStencil(&h, command);
            }
        }
        shape.command_stencil_hash = if (saw_command) h.state else 0;
        return shape;
    }
};

fn buildExecutionPlan(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    schedule_policy: SchedulePolicy,
    stages: []const StagePolicy,
    kernelizer: Kernelizer,
) !ExecutionPlan {
    const schedule = try buildKernelSchedule(alloc, ops, schedule_policy);
    errdefer if (schedule.len > 0) alloc.free(schedule);

    const regions = try buildStageRegionSchedule(alloc, schedule, stages);
    errdefer if (regions.len > 0) alloc.free(regions);

    const region_kernel_plans = try buildRegionKernelPlans(alloc, ops, regions, kernelizer);
    errdefer {
        deinitRegionKernelPlans(alloc, region_kernel_plans);
        if (region_kernel_plans.len > 0) alloc.free(region_kernel_plans);
    }

    return .{
        .schedule = schedule,
        .regions = regions,
        .region_kernel_plans = region_kernel_plans,
    };
}

fn summarizeRegionExecution(
    units: []const ScheduleUnit,
    items: []const KernelItem,
    backend_pattern_indices: []const u32,
) RegionExecutionSummary {
    var summary = RegionExecutionSummary{ .units = @intCast(units.len) };
    var prev_execution: ?ExecutionClass = null;
    var current_backend_island_units: u32 = 0;
    var current_backend_island_ops: u32 = 0;

    for (units) |unit| {
        const execution = scheduleUnitExecution(unit, items, backend_pattern_indices);
        summary.ops += unit.op_count;
        switch (execution) {
            .backend => {
                summary.backend_units += 1;
                summary.backend_ops += unit.op_count;
                if (prev_execution == null or prev_execution.? != .backend) {
                    summary.backend_islands += 1;
                    current_backend_island_units = 0;
                    current_backend_island_ops = 0;
                }
                current_backend_island_units += 1;
                current_backend_island_ops += unit.op_count;
                summary.max_backend_island_units = @max(summary.max_backend_island_units, current_backend_island_units);
                summary.max_backend_island_ops = @max(summary.max_backend_island_ops, current_backend_island_ops);
            },
            .fallback => {
                summary.fallback_units += 1;
                summary.fallback_ops += unit.op_count;
            },
        }
        if (prev_execution) |prev| {
            if (prev != execution) summary.execution_transitions += 1;
        }
        prev_execution = execution;
    }

    return summary;
}

fn scheduleUnitExecution(
    unit: ScheduleUnit,
    items: []const KernelItem,
    backend_pattern_indices: []const u32,
) ExecutionClass {
    return switch (unit.kind) {
        .item => items[@intCast(unit.start_item)].execution,
        .pattern_region => if (containsPatternIndex(backend_pattern_indices, unit.pattern_index)) .backend else .fallback,
    };
}

fn containsPatternIndex(indices: []const u32, pattern_index: u32) bool {
    for (indices) |idx| {
        if (idx == pattern_index) return true;
    }
    return false;
}

fn itemScheduleUnit(item_index: u32, item: KernelItem) ScheduleUnit {
    return .{
        .kind = .item,
        .start_item = item_index,
        .item_count = 1,
        .op_start = item.start,
        .op_count = item.len,
    };
}

fn sortPatternRegions(regions: []PatternRegion) void {
    if (regions.len < 2) return;
    for (1..regions.len) |i| {
        const tmp = regions[i];
        var j = i;
        while (j > 0 and patternRegionLess(tmp, regions[j - 1])) : (j -= 1) {
            regions[j] = regions[j - 1];
        }
        regions[j] = tmp;
    }
}

fn patternRegionLess(lhs: PatternRegion, rhs: PatternRegion) bool {
    if (lhs.region.start_item != rhs.region.start_item) {
        return lhs.region.start_item < rhs.region.start_item;
    }
    if (lhs.region.item_count != rhs.region.item_count) {
        return lhs.region.item_count > rhs.region.item_count;
    }
    return lhs.pattern_index < rhs.pattern_index;
}

fn testElementwise(op: backend_mod.Op) backend_mod.DeviceOp {
    return .{ .elementwise = .{ .op = op, .dst = 0, .src0 = 0, .src1 = 0, .n = 1 } };
}

fn testMatmul(rows: usize) backend_mod.DeviceOp {
    return .{ .matmul = .{
        .dst = 0,
        .a = 0,
        .b = 0,
        .geom = .{
            .M = rows,
            .N = 4,
            .K = 4,
            .a_row_stride = 4,
            .a_col_stride = 1,
            .b_row_stride = 4,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 4,
        },
    } };
}

fn testMatmulWith(dst: u16, input: u16, weights: u16, rows: usize) backend_mod.DeviceOp {
    var op = testMatmul(rows);
    op.matmul.dst = dst;
    op.matmul.a = input;
    op.matmul.b = weights;
    return op;
}

fn testQMatmul(rows: u32) backend_mod.DeviceOp {
    return .{ .qmatmul = .{ .dst = 0, .input = 0, .weight_idx = 0, .M = rows, .N = 4, .K = 4 } };
}

fn testQMatmulWith(dst: u16, input: u16, rows: u32) backend_mod.DeviceOp {
    return .{ .qmatmul = .{ .dst = dst, .input = input, .weight_idx = 0, .M = rows, .N = 4, .K = 4 } };
}

fn testSliceAssign(dst_offset: u32) backend_mod.DeviceOp {
    return .{ .slice_assign = .{
        .dst = 0,
        .src = 1,
        .rows = 1,
        .cols = 4,
        .dst_base_offset = 0,
        .dst_offset = dst_offset,
        .dst_row_stride = 4,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 4,
        .src_col_stride = 1,
        .patch_stride = 4,
    } };
}

fn testQMatmulSidecar(src: u16, dst: u16) backend_mod.DeviceOp {
    return .{ .slice_assign = .{
        .dst = dst,
        .src = src,
        .rows = 4,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };
}

fn testRopeSliceAssignOps() [2]backend_mod.DeviceOp {
    return .{
        .{ .rope = .{
            .dst = 2,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 3,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        .{ .slice_assign = .{
            .dst = 3,
            .src = 2,
            .rows = 4,
            .cols = 3,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };
}

fn testAttention(offset: u32) backend_mod.DeviceOp {
    return .{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 2,
        .seq_kv = 4,
        .scale = 0.5,
        .q_off = offset,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = offset,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 4,
        .dst_rs = 1,
        .dst_cs = 4,
    } };
}

fn testRmsnormScaleOps() [3]backend_mod.DeviceOp {
    return .{
        .{ .rmsnorm = .{
            .dst = 1,
            .src = 0,
            .rows = 2,
            .cols = 4,
            .eps = 1e-5,
        } },
        .{ .repeat = .{
            .dst = 2,
            .src = 3,
            .n = 8,
            .src_ne = .{ 4, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 4, 4, 4 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 4,
            .src0 = 1,
            .src1 = 2,
            .n = 8,
        } },
    };
}

fn testRmsnormScaleActivationOps() [4]backend_mod.DeviceOp {
    const row_chain = testRmsnormScaleOps();
    return row_chain ++ [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .gelu,
        .dst = 5,
        .src0 = 4,
        .src1 = 4,
        .n = 8,
    } }};
}

fn expectKernelItem(item: KernelItem, family: KernelFamily, execution: ExecutionClass, start: u32, len: u32) !void {
    try std.testing.expectEqual(family, item.family);
    try std.testing.expectEqual(execution, item.execution);
    try std.testing.expectEqual(start, item.start);
    try std.testing.expectEqual(len, item.len);
}

test "program command index set carries flat sidecars once" {
    var command = ProgramCommand{
        .kind = .projection_cache_group,
        .op_start = 2,
        .op_count = 8,
        .anchor_count = 2,
        .sidecar_count = 3,
    };
    command.indices[0] = 5;
    command.indices[1] = 2;
    command.sidecar_indices[0] = 6;
    command.sidecar_indices[1] = 5;
    command.sidecar_indices[2] = 9;

    const indices = command.explicitIndexSet();
    const expected = [_]usize{ 5, 2, 6, 9 };
    try std.testing.expectEqualSlices(usize, &expected, indices.slice());

    var sorted = command.sortedExplicitIndexSet();
    const expected_sorted = [_]usize{ 2, 5, 6, 9 };
    try std.testing.expectEqualSlices(usize, &expected_sorted, sorted.slice());

    var iter = command.coveredIndexIterator();
    var got: [4]usize = undefined;
    var count: usize = 0;
    while (iter.next()) |idx| {
        got[count] = idx;
        count += 1;
    }
    try std.testing.expectEqualSlices(usize, &expected_sorted, got[0..count]);
}

test "program command shape drives coverage and advancement" {
    try std.testing.expectEqual(ProgramCommandCoverage.anchor_sidecars, ProgramCommandKind.projection_chain.shape().coverage);
    try std.testing.expectEqual(ProgramCommandCoverage.anchor_sidecars, ProgramCommandKind.dense_projection_chain.shape().coverage);
    try std.testing.expectEqual(ProgramCommandAdvance.contiguous, ProgramCommandKind.projection_chain.shape().advance);
    try std.testing.expectEqual(ProgramCommandSidecarLayout.flat, ProgramCommandKind.projection_cache_group.shape().sidecars);

    const chain = ProgramCommand{
        .kind = .projection_chain,
        .op_start = 3,
        .op_count = 2,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    try std.testing.expect(chain.hasExplicitCoverage());
    try std.testing.expectEqual(@as(u32, 2), chain.coveredOpCount());
    try std.testing.expectEqual(@as(u32, 2), chain.advanceCount());

    const group = ProgramCommand{
        .kind = .movement_group,
        .op_start = 3,
        .op_count = 8,
        .anchor_count = 4,
    };
    try std.testing.expect(group.hasExplicitCoverage());
    try std.testing.expectEqual(@as(u32, 4), group.coveredOpCount());
    try std.testing.expectEqual(@as(u32, 1), group.advanceCount());

    const contiguous = ProgramCommand.contiguous(.row_chain, 4, 3);
    var iter = contiguous.coveredIndexIterator();
    const expected = [_]usize{ 4, 5, 6 };
    for (expected) |idx| {
        try std.testing.expectEqual(idx, iter.next().?);
    }
    try std.testing.expectEqual(@as(?usize, null), iter.next());
    try std.testing.expectEqual(@as(usize, 0), iter.remainingCount());
}

test "kernel schedule groups contiguous fallback ops by family" {
    const ops = [_]backend_mod.DeviceOp{
        testElementwise(.add),
        testElementwise(.relu),
        .{ .softmax = .{ .dst = 0, .src = 0, .rows = 1, .cols = 4 } },
        .{ .rmsnorm = .{ .dst = 0, .src = 0, .rows = 1, .cols = 4 } },
        .{ .reduce = .{ .op = .sum, .dst = 0, .src = 0, .n_out = 1, .reduce_size = 4 } },
    };
    const policy = SchedulePolicy{ .capabilities = backend_mod.Capabilities.reference_cpu };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 3), items.len);
    try expectKernelItem(items[0], .elementwise, .fallback, 0, 2);
    try expectKernelItem(items[1], .row, .fallback, 2, 2);
    try expectKernelItem(items[2], .reduce, .fallback, 4, 1);
}

test "kernel schedule uses coarse backend thresholds for matmul families" {
    const ops = [_]backend_mod.DeviceOp{
        testMatmul(1),
        testMatmul(16),
        testMatmul(32),
        testQMatmul(1),
        testQMatmul(16),
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .matmul = true, .qmatvec = true, .qmatmul = true },
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 4), items.len);
    try expectKernelItem(items[0], .matmul, .fallback, 0, 1);
    try expectKernelItem(items[1], .matmul, .backend, 1, 2);
    try expectKernelItem(items[2], .qmatvec, .backend, 3, 1);
    try expectKernelItem(items[3], .qmatmul, .backend, 4, 1);
}

test "kernel schedule treats single-row quantized matmul as qmatvec" {
    const op = testQMatmul(1);
    var policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .qmatvec = true },
    };

    try std.testing.expectEqual(KernelFamily.qmatvec, kernelFamily(op));
    try std.testing.expectEqual(ExecutionClass.backend, executionClass(op, policy));
    policy.fine_grained = true;
    try std.testing.expectEqual(ExecutionClass.backend, executionClass(op, policy));
}

test "kernel schedule respects fused elementwise capability limits" {
    const small_steps = [_]backend_mod.FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }};
    const large_steps = [_]backend_mod.FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }} ** 9;
    const ops = [_]backend_mod.DeviceOp{
        .{ .fused_elementwise = .{ .steps = &small_steps, .n = 1, .dst = 0, .src = 0, .dst_offset = 0, .src_offset = 0 } },
        .{ .fused_elementwise = .{ .steps = &large_steps, .n = 1, .dst = 0, .src = 0, .dst_offset = 0, .src_offset = 0 } },
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .fused_elementwise = true },
        .fine_grained = true,
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 2), items.len);
    try expectKernelItem(items[0], .fused_elementwise, .backend, 0, 1);
    try expectKernelItem(items[1], .fused_elementwise, .fallback, 1, 1);
}

test "kernel schedule fine grained policy unlocks small native kernels" {
    const op = testElementwise(.add);
    var policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .elementwise = true },
    };

    try std.testing.expectEqual(ExecutionClass.fallback, executionClass(op, policy));
    policy.fine_grained = true;
    try std.testing.expectEqual(ExecutionClass.backend, executionClass(op, policy));
}

test "schedule shape match ignores dynamic offsets but catches family changes" {
    var ops = [_]backend_mod.DeviceOp{
        testSliceAssign(0),
        testQMatmul(1),
        testQMatmul(16),
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.reference_cpu,
        .native_kernels = .{ .movement = true, .qmatvec = true, .qmatmul = true },
        .fine_grained = true,
        .min_backend_qmatmul_m = 0,
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    ops[0].slice_assign.dst_offset = 128;
    try std.testing.expect(scheduleShapeMatches(&ops, items, policy));

    ops[1].qmatmul.M = 16;
    try std.testing.expect(!scheduleShapeMatches(&ops, items, policy));
}

test "anchor window regions split member runs by anchor count" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .elementwise, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 3, .len = 2 },
        .{ .family = .row, .execution = .fallback, .start = 5, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 7, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .matmul, .execution = .backend, .start = 9, .len = 1 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster(), 3);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 5), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 6), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 3), regions[0].anchor_count);
}

test "qmatmul cluster windows keep prefill attention inside layer regions" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .fallback, .start = 4, .len = 2 },
        .{ .family = .attention, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 7, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 9, .len = 3 },
        .{ .family = .matmul, .execution = .backend, .start = 12, .len = 1 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.qmatmulCluster(), 7);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 8), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 13), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 7), regions[0].anchor_count);
}

test "qmatvec cluster windows carry final dense projection tails" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .backend, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .fallback, .start = 4, .len = 2 },
        .{ .family = .attention, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .qmatvec, .execution = .backend, .start = 7, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .qmatvec, .execution = .backend, .start = 9, .len = 3 },
        .{ .family = .row, .execution = .fallback, .start = 12, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 13, .len = 1 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster(), 7);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 9), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 14), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 7), regions[0].anchor_count);
}

test "matmul cluster windows keep dense prefill attention inside layer regions" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .fallback, .start = 4, .len = 2 },
        .{ .family = .attention, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 7, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 9, .len = 3 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.matmulCluster(), 7);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 7), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 12), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 7), regions[0].anchor_count);
}

test "stage plan builds named anchored layer windows" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .backend, .start = 4, .len = 2 },
        .{ .family = .movement, .execution = .backend, .start = 6, .len = 2 },
        .{ .family = .attention, .execution = .backend, .start = 8, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 9, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .backend, .start = 10, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 11, .len = 3 },
        .{ .family = .matmul, .execution = .backend, .start = 14, .len = 1 },
        .{ .family = .qmatvec, .execution = .backend, .start = 15, .len = 1 },
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored(7, RegionPolicy.qmatmulCluster(), 7),
    };

    const plan = try buildStagePlan(std.testing.allocator, &items, &stages);
    defer std.testing.allocator.free(plan);

    try std.testing.expectEqual(@as(usize, 1), plan.len);
    try std.testing.expectEqual(@as(u32, 7), plan[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), plan[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 9), plan[0].region.item_count);
    try std.testing.expectEqual(@as(u32, 15), plan[0].region.op_count);
    try std.testing.expectEqual(@as(u32, 7), plan[0].region.anchor_count);

    const units = try buildStageRegionSchedule(std.testing.allocator, &items, &stages);
    defer std.testing.allocator.free(units);

    try std.testing.expectEqual(@as(usize, 2), units.len);
    try std.testing.expectEqual(ScheduleUnitKind.pattern_region, units[0].kind);
    try std.testing.expectEqual(@as(u32, 7), units[0].pattern_index);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[1].kind);
    try std.testing.expectEqual(@as(u32, 15), units[1].op_start);
}

test "stage plan prefers dense layer windows before dense matmul tails" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .fallback, .start = 4, .len = 2 },
        .{ .family = .attention, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 7, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 9, .len = 3 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 12, .len = 1 },
        .{ .family = .matmul, .execution = .fallback, .start = 13, .len = 1 },
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored(7, RegionPolicy.matmulCluster(), 7),
        StagePolicy.anchored(8, RegionPolicy.matmulCluster(), 1),
    };

    const plan = try buildStagePlan(std.testing.allocator, &items, &stages);
    defer std.testing.allocator.free(plan);

    try std.testing.expectEqual(@as(usize, 2), plan.len);
    try std.testing.expectEqual(@as(u32, 7), plan[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), plan[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 8), plan[0].region.item_count);
    try std.testing.expectEqual(@as(u32, 7), plan[0].region.anchor_count);
    try std.testing.expectEqual(@as(u32, 8), plan[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 8), plan[1].region.start_item);
    try std.testing.expectEqual(@as(u32, 1), plan[1].region.item_count);
    try std.testing.expectEqual(@as(u32, 1), plan[1].region.anchor_count);
}

test "program command stream emits row chains directly" {
    const row_chain = testRmsnormScaleOps();
    const commands = try buildProgramCommands(std.testing.allocator, &row_chain, CommandStreamPolicy.default());
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.row_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(@as(u32, 3), commands[0].op_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.row_chains);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
}

test "program command stream emits rmsnorm scale activation row chains directly" {
    const row_chain = testRmsnormScaleActivationOps();
    const commands = try buildProgramCommands(std.testing.allocator, &row_chain, CommandStreamPolicy.default());
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.row_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(@as(u32, 4), commands[0].op_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.row_chains);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
}

test "rmsnorm scale chain liveness distinguishes materialized intermediates" {
    const row_chain = testRmsnormScaleOps();
    try std.testing.expect(!rmsnormScaleChainHasExternalUsers(&row_chain, 0));

    const norm_read = row_chain ++ [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 6,
        .src0 = 1,
        .src1 = 1,
        .n = 1,
    } }};
    try std.testing.expect(rmsnormScaleChainHasExternalUsers(&norm_read, 0));

    const repeat_read = row_chain ++ [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 6,
        .src0 = 2,
        .src1 = 2,
        .n = 1,
    } }};
    try std.testing.expect(rmsnormScaleChainHasExternalUsers(&repeat_read, 0));

    const overwritten = row_chain ++ [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 0, .src1 = 0, .n = 8 } },
        .{ .elementwise = .{ .op = .add, .dst = 6, .src0 = 1, .src1 = 1, .n = 1 } },
    };
    try std.testing.expect(!rmsnormScaleChainHasExternalUsers(&overwritten, 0));
}

test "rmsnorm scale activation chain liveness includes activation consumer" {
    const row_chain = testRmsnormScaleActivationOps();
    try std.testing.expect(!rmsnormScaleActivationChainHasExternalUsers(&row_chain, 0));

    const scale_read = row_chain ++ [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 6,
        .src0 = 4,
        .src1 = 4,
        .n = 1,
    } }};
    try std.testing.expect(rmsnormScaleActivationChainHasExternalUsers(&scale_read, 0));
}

test "projection groups batch independent prefill qmatmuls" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        testQMatmulWith(2, 0, 16),
        testQMatmulWith(3, 0, 16),
        testQMatmulWith(4, 0, 16),
        testQMatmulWith(5, 0, 16),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, selection.kind);
    try std.testing.expectEqual(@as(usize, 0), selection.start_op);
    try std.testing.expectEqual(@as(usize, 3), selection.end_op);
    try std.testing.expectEqual(@as(usize, 4), selection.anchor_count);
    try std.testing.expectEqual(@as(usize, 0), selection.sidecar_count);

    const command = ProgramCommand.fromProjectionSelection(selection);
    try std.testing.expectEqual(@as(u32, 4), command.coveredOpCount());
}

test "projection groups carry compatible qmatmul cache-store sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulSidecar(1, 8),
        .{ .elementwise = .{ .op = .add, .dst = 9, .src0 = 9, .src1 = 9, .n = 1 } },
        testQMatmulWith(2, 0, 2),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 3), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);
    try std.testing.expectEqual(@as(usize, 0), selection.start_op);
    try std.testing.expectEqual(@as(usize, 3), selection.end_op);
    try std.testing.expectEqual(@as(usize, 2), selection.anchor_count);
    try std.testing.expectEqual(@as(usize, 1), selection.sidecar_count);

    try std.testing.expect(qmatmulSliceSidecarCompatible(ops[0].qmatmul, ops[1].slice_assign));
}

test "projection groups carry compatible qmatvec cache-store sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .slice_assign = .{
            .dst = 8,
            .src = 1,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testQMatmulWith(2, 0, 1),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.decodeQMatvec(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);
    try std.testing.expect(qmatvecSliceSidecarCompatible(ops[0].qmatmul, ops[1].slice_assign));

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
}

test "projection groups carry compatible qmatvec elementwise sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .elementwise = .{
            .op = .add,
            .dst = 8,
            .src0 = 1,
            .src1 = 9,
            .n = 4,
            .dst_offset = 0,
            .src0_offset = 0,
            .src1_offset = 0,
        } },
        testQMatmulWith(2, 0, 1),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.decodeQMatvec(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);
    try std.testing.expect(qmatvecElementwiseSidecarCompatible(ops[0].qmatmul, ops[1].elementwise));

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_sidecars);
}

test "projection groups carry compatible qmatmul elementwise sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 10, .src0 = 1, .src1 = 20, .n = 8 } },
        testQMatmulWith(2, 0, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 11, .src0 = 21, .src1 = 2, .n = 8 } },
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 3), selection.sidecar_indices[1]);

    try std.testing.expectEqual(@as(usize, 2), selection.anchor_count);
    try std.testing.expectEqual(@as(usize, 2), selection.sidecar_count);

    const command = ProgramCommand.fromProjectionSelection(selection);
    try std.testing.expectEqual(@as(u32, 4), command.coveredOpCount());

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);
    const shape = try ProgramCommandStreamShape.fromCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), shape.command_count);
    try std.testing.expectEqual(@as(u32, 4), shape.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), shape.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), shape.projection_groups);
    try std.testing.expectEqual(@as(u32, 2), shape.projection_anchors);
    try std.testing.expectEqual(@as(u32, 2), shape.projection_sidecars);
    try std.testing.expectEqual(@as(u32, 0), shape.projection_chain_row_chain_frontiers);
    try std.testing.expectEqual(@as(u32, 4), shape.max_projection_span_ops);
}

test "projection groups reject conflicting or nonhoistable projections" {
    const conflict_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        testQMatmulWith(1, 0, 16),
    };
    try std.testing.expect(findProjectionGroup(&conflict_ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null) == null);

    const blocked_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 0, .n = 1 } },
        testQMatmulWith(3, 2, 16),
    };
    try std.testing.expect(findProjectionGroup(&blocked_ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null) == null);
}

test "projection groups use access spans instead of whole-buffer conflicts" {
    var first = testQMatmulWith(1, 0, 2);
    first.qmatmul.dst_offset = 0;
    var second = testQMatmulWith(1, 0, 2);
    second.qmatmul.dst_offset = 8;
    const disjoint_ops = [_]backend_mod.DeviceOp{ first, second };

    const disjoint = findProjectionGroup(&disjoint_ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(@as(usize, 2), disjoint.anchor_count);

    second.qmatmul.dst_offset = 4;
    const overlapping_ops = [_]backend_mod.DeviceOp{ first, second };
    try std.testing.expect(findProjectionGroup(&overlapping_ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null) == null);
}

test "program command stream merges stage and projection commands" {
    const rope_chain = testRopeSliceAssignOps();
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulSidecar(1, 8),
        rope_chain[0],
        rope_chain[1],
        testQMatmulWith(4, 0, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 4), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(ProgramCommandKind.rope_chain, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_chains);
}

test "program command stream derives defaults from backend capabilities" {
    const policy = CommandStreamPolicy.default();
    try std.testing.expect(policy.row_rope_chains);
    try std.testing.expectEqual(@as(u32, 4), policy.qmatvec_group_size);
    try std.testing.expectEqual(@as(u32, 4), policy.qmatmul_group_size);
    try std.testing.expectEqual(@as(u32, 4), policy.dense_matvec_group_size);
    try std.testing.expect(policy.qmatmul_sidecars);
    try std.testing.expectEqual(@as(u32, 8), policy.qmatmul_cache_sidecars_per_anchor);
    try std.testing.expectEqual(@as(u32, 16), policy.max_rope_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_movement_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_attention_batch);
    try std.testing.expectEqual(@as(u32, 4), policy.max_attention_store_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_rope_attention_store_batch);
    try std.testing.expectEqual(@as(u32, 8), policy.max_elementwise_batch);
    try std.testing.expect(policy.fuse_repeat_fused_elementwise);
}

test "program command stream carries multiple projection cache stores" {
    var store0 = testQMatmulSidecar(3, 8);
    store0.slice_assign.dst_offset = 0;
    var store1 = testQMatmulSidecar(3, 8);
    store1.slice_assign.dst_offset = 8;
    var store2 = testQMatmulSidecar(3, 8);
    store2.slice_assign.dst_offset = 16;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulWith(2, 0, 2),
        testQMatmulWith(3, 0, 2),
        store0,
        .{ .elementwise = .{ .op = .add, .dst = 20, .src0 = 20, .src1 = 20, .n = 1 } },
        store1,
        store2,
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 5), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 6), commands[0].sidecar_indices[2]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecarAnchorSlot(1));
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecarAnchorSlot(2));
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
}

test "program command stream carries qmatvec rope cache stores" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;
    rope.rope.seq_len = 1;
    var store = rope_pair[1];
    store.slice_assign.cols = 1;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        rope,
        store,
        testQMatmulWith(4, 0, 1),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecarAnchorSlot(1));

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "program command stream carries dense matvec rope and value cache stores" {
    var rope_pair = testRopeSliceAssignOps();
    rope_pair[0].rope.src = 2;
    rope_pair[0].rope.dst = 4;
    rope_pair[0].rope.src_off = 0;
    rope_pair[0].rope.dst_off = 0;
    rope_pair[0].rope.seq_len = 1;
    rope_pair[1].slice_assign.src = 4;
    rope_pair[1].slice_assign.dst = 8;
    rope_pair[1].slice_assign.src_offset = 0;
    rope_pair[1].slice_assign.cols = 1;

    var v_store = testQMatmulSidecar(3, 9);
    v_store.slice_assign.cols = 1;

    const ops = [_]backend_mod.DeviceOp{
        testMatmulWith(1, 0, 10, 1),
        testMatmulWith(2, 0, 11, 1),
        testMatmulWith(3, 0, 12, 1),
        rope_pair[0],
        rope_pair[1],
        v_store,
        testMatmulWith(7, 14, 13, 1),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 4), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 5), commands[0].sidecar_indices[2]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecarAnchorSlot(1));
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecarAnchorSlot(2));
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
}

test "program command stream carries dense prefill value cache stores" {
    var v_store = testQMatmulSidecar(3, 9);
    v_store.slice_assign.cols = 2;
    v_store.slice_assign.src_col_stride = 4;

    const ops = [_]backend_mod.DeviceOp{
        testMatmulWith(1, 0, 10, 2),
        testMatmulWith(2, 0, 11, 2),
        testMatmulWith(3, 0, 12, 2),
        v_store,
        testMatmulWith(7, 14, 13, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "program command stream rejects live qmatvec rope cache stores" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;
    rope.rope.seq_len = 1;
    var store = rope_pair[1];
    store.slice_assign.cols = 1;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        rope,
        store,
        .{ .elementwise = .{ .op = .add, .dst = 9, .src0 = 2, .src1 = 8, .n = 4, .src0_offset = 8 } },
        testQMatmulWith(4, 0, 1),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    for (commands) |command| {
        try std.testing.expect(command.kind != .projection_cache_group);
    }
}

test "program command stream groups rope slice stores" {
    const first = testRopeSliceAssignOps();
    var second = testRopeSliceAssignOps();
    second[0].rope.src = 11;
    second[0].rope.src_off = 8;
    second[1].slice_assign.dst_offset = 16;

    const ops = [_]backend_mod.DeviceOp{
        first[0],
        first[1],
        testQMatmulWith(8, 9, 2),
        second[0],
        second[1],
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(usize, 3), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 4), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_store_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_store_group_sidecars);
}

test "program command stream keeps used noncontiguous ops single-owned" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 9, .src0 = 9, .src1 = 9, .n = 1 } },
        testQMatmulWith(4, 0, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
}

test "program command stream emits projection sidecar chains" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_chain_sidecars);

    var staged_policy = CommandStreamPolicy.grouped(4, 4);
    staged_policy.fuse_projection_chain = false;
    const staged_commands = try buildProgramCommands(std.testing.allocator, &ops, staged_policy);
    defer std.testing.allocator.free(staged_commands);

    try std.testing.expectEqual(@as(usize, 2), staged_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, staged_commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, staged_commands[1].kind);
    try std.testing.expectEqual(@as(u32, 0), staged_commands[0].op_start);
    try std.testing.expectEqual(@as(u32, 1), staged_commands[1].op_start);

    const staged_summary = summarizeProgramCommands(staged_commands);
    try std.testing.expectEqual(@as(u32, 2), staged_summary.commands);
    try std.testing.expectEqual(@as(u32, 2), staged_summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), staged_summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 0), staged_summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 0), staged_summary.projection_chains);
    try std.testing.expectEqual(@as(u32, 0), staged_summary.projection_chain_sidecars);
}

test "program command stream uses projection row-chain only for prompt-sized semantic frontiers" {
    const tiny_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 7, .n = 8 } },
        .{ .rmsnorm = .{ .dst = 3, .src = 2, .rows = 2, .cols = 4, .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 4, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 4, 4, 4 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 6, .src0 = 3, .src1 = 4, .n = 8 } },
    };

    var legacy_policy = CommandStreamPolicy.grouped(4, 4);
    legacy_policy.fuse_projection_row_chain = false;
    const default_commands = try buildProgramCommands(std.testing.allocator, &tiny_ops, legacy_policy);
    defer std.testing.allocator.free(default_commands);
    try std.testing.expectEqual(@as(usize, 2), default_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, default_commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.row_chain, default_commands[1].kind);

    var policy = CommandStreamPolicy.grouped(4, 4);
    policy.fuse_projection_row_chain = true;
    policy.min_projection_row_chain_rows = 2;
    const fused_commands = try buildProgramCommands(std.testing.allocator, &tiny_ops, policy);
    defer std.testing.allocator.free(fused_commands);

    try std.testing.expectEqual(@as(usize, 1), fused_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_row_chain, fused_commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), fused_commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 4), fused_commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 5), fused_commands[0].coveredOpCount());
    try std.testing.expectEqual(@as(usize, 0), fused_commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), fused_commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 4), fused_commands[0].sidecar_indices[3]);

    const shape = try ProgramCommandStreamShape.fromCommands(fused_commands);
    try std.testing.expectEqual(@as(u32, 1), shape.command_count);
    try std.testing.expectEqual(@as(u32, 5), shape.covered_ops);
    try std.testing.expectEqual(@as(u32, 4), shape.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 0), shape.projection_chain_row_chain_frontiers);

    const prompt_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(11, 10, 8),
        .{ .elementwise = .{ .op = .add, .dst = 12, .src0 = 11, .src1 = 12, .n = 32 } },
        .{ .rmsnorm = .{ .dst = 13, .src = 12, .rows = 8, .cols = 4, .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 15,
            .src = 14,
            .n = 32,
            .src_ne = .{ 4, 1, 1, 1 },
            .dst_ne = .{ 4, 8, 1, 1 },
            .src_strides = .{ 1, 4, 4, 4 },
            .dst_strides = .{ 1, 4, 32, 32 },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 16, .src0 = 13, .src1 = 15, .n = 32 } },
    };
    var prompt_policy = CommandStreamPolicy.grouped(4, 4);
    prompt_policy.fuse_projection_row_chain = true;
    const prompt_commands = try buildProgramCommands(std.testing.allocator, &prompt_ops, prompt_policy);
    defer std.testing.allocator.free(prompt_commands);
    try std.testing.expectEqual(@as(usize, 1), prompt_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_row_chain, prompt_commands[0].kind);

    const default_prompt_commands = try buildProgramCommands(std.testing.allocator, &prompt_ops, CommandStreamPolicy.default());
    defer std.testing.allocator.free(default_prompt_commands);
    try std.testing.expectEqual(@as(usize, 2), default_prompt_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, default_prompt_commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.row_chain, default_prompt_commands[1].kind);

    const prompt_candidate_commands = try buildProgramCommands(std.testing.allocator, &prompt_ops, CommandStreamPolicy.promptProjectionRowChainSingleDispatchCandidate());
    defer std.testing.allocator.free(prompt_candidate_commands);
    try std.testing.expectEqual(@as(usize, 1), prompt_candidate_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_row_chain, prompt_candidate_commands[0].kind);

    const decode_candidate_commands = try buildProgramCommands(std.testing.allocator, &tiny_ops, CommandStreamPolicy.promptProjectionRowChainSingleDispatchCandidate());
    defer std.testing.allocator.free(decode_candidate_commands);
    try std.testing.expectEqual(@as(usize, 2), decode_candidate_commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, decode_candidate_commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.row_chain, decode_candidate_commands[1].kind);
}

test "program command stream emits qmatvec projection sidecar chains" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
}

test "program command stream batches qmatvec elementwise sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        testQMatmulWith(2, 0, 1),
        testQMatmulWith(3, 0, 1),
        .{ .elementwise = .{ .op = .add, .dst = 4, .src0 = 1, .src1 = 9, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 5, .src0 = 2, .src1 = 8, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 4), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecarAnchorSlot(1));

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 4), summary.estimated_saved_dispatches);
}

test "program command stream fuses sibling qmatvec elementwise chain" {
    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = testQMatmulWith(@intCast(i + 1), 0, 1);
    }
    ops[7] = .{ .elementwise = .{ .op = .add, .dst = 8, .src0 = 1, .src1 = 2, .n = 4 } };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_pair_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 7), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.commands);
    try std.testing.expectEqual(@as(u32, 8), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "program command stream emits qmatvec fused-elementwise projection chains" {
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 2,
            .src = 1,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].coveredOpCount());
    try std.testing.expectEqual(@as(u32, 1), summarizeProgramCommands(commands).estimated_dispatches);
}

test "program command stream emits dense matmul elementwise sidecar chains" {
    var matmul = testMatmul(1);
    matmul.matmul.dst = 1;
    const ops = [_]backend_mod.DeviceOp{
        matmul,
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].coveredOpCount());
    try std.testing.expectEqual(@as(u32, 1), summarizeProgramCommands(commands).estimated_dispatches);
}

test "program command stream emits dense matmul fused-elementwise sidecar chains" {
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 3, .secondary_offset = 0 },
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    var matmul = testMatmul(1);
    matmul.matmul.dst = 1;
    const ops = [_]backend_mod.DeviceOp{
        matmul,
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 2,
            .src = 1,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].coveredOpCount());
    try std.testing.expectEqual(@as(u32, 1), summarizeProgramCommands(commands).estimated_dispatches);
}

test "dense matmul sidecar chains accept prefill geometry and reject mismatched spans" {
    var batched = testMatmul(2);
    batched.matmul.dst = 1;
    const prefill = [_]backend_mod.DeviceOp{
        batched,
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
    };
    const commands_prefill = try buildProgramCommands(std.testing.allocator, &prefill, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands_prefill);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands_prefill[0].kind);

    var matmul = testMatmul(1);
    matmul.matmul.dst = 1;
    const mismatched = [_]backend_mod.DeviceOp{
        matmul,
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 3 } },
    };
    const commands_mismatched = try buildProgramCommands(std.testing.allocator, &mismatched, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands_mismatched);
    try std.testing.expectEqual(ProgramCommandKind.op, commands_mismatched[0].kind);
}

test "program command stream fuses dense matmul repeated bias add" {
    var matmul = testMatmul(4);
    matmul.matmul.dst = 1;
    matmul.matmul.geom.N = 3;
    matmul.matmul.geom.dst_row_stride = 3;
    const ops = [_]backend_mod.DeviceOp{
        matmul,
        .{ .repeat = .{
            .dst = 2,
            .src = 3,
            .n = 12,
            .src_ne = .{ 3, 1, 1, 1 },
            .dst_ne = .{ 3, 4, 1, 1 },
            .src_strides = .{ 1, 3, 3, 3 },
            .dst_strides = .{ 1, 3, 12, 12 },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 2, .n = 12 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].coveredOpCount());

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.dense_projection_chains);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_chain_sidecars);
}

test "program command stream fuses dense matmul repeated bias gelu" {
    var matmul = testMatmul(4);
    matmul.matmul.dst = 1;
    matmul.matmul.geom.N = 3;
    matmul.matmul.geom.dst_row_stride = 3;
    const ops = [_]backend_mod.DeviceOp{
        matmul,
        .{ .repeat = .{
            .dst = 2,
            .src = 3,
            .n = 12,
            .src_ne = .{ 3, 1, 1, 1 },
            .dst_ne = .{ 3, 4, 1, 1 },
            .src_strides = .{ 1, 3, 3, 3 },
            .dst_strides = .{ 1, 3, 12, 12 },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 2, .n = 12 } },
        .{ .elementwise = .{ .op = .gelu, .dst = 4, .src0 = 2, .src1 = 2, .n = 12 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 4), commands[0].coveredOpCount());

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.dense_projection_chains);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_chain_sidecars);
}

test "program command stream fuses dense matmul repeated bias silu" {
    var matmul = testMatmul(4);
    matmul.matmul.dst = 1;
    matmul.matmul.geom.N = 3;
    matmul.matmul.geom.dst_row_stride = 3;
    const ops = [_]backend_mod.DeviceOp{
        matmul,
        .{ .repeat = .{
            .dst = 2,
            .src = 3,
            .n = 12,
            .src_ne = .{ 3, 1, 1, 1 },
            .dst_ne = .{ 3, 4, 1, 1 },
            .src_strides = .{ 1, 3, 3, 3 },
            .dst_strides = .{ 1, 3, 12, 12 },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 2, .n = 12 } },
        .{ .elementwise = .{ .op = .silu, .dst = 4, .src0 = 2, .src1 = 2, .n = 12 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(u32, 4), commands[0].coveredOpCount());
}

test "program command stream batches dense matvec elementwise sidecars" {
    var first = testMatmulWith(1, 0, 10, 1);
    first.matmul.geom.N = 4;
    first.matmul.geom.dst_row_stride = 4;
    var second = testMatmulWith(2, 0, 11, 1);
    second.matmul.geom.N = 4;
    second.matmul.geom.dst_row_stride = 4;
    var third = testMatmulWith(3, 0, 12, 1);
    third.matmul.geom.N = 4;
    third.matmul.geom.dst_row_stride = 4;

    const ops = [_]backend_mod.DeviceOp{
        first,
        second,
        third,
        .{ .elementwise = .{ .op = .add, .dst = 4, .src0 = 1, .src1 = 9, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 5, .src0 = 2, .src1 = 8, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 4), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecarAnchorSlot(0));
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecarAnchorSlot(1));

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 4), summary.estimated_saved_dispatches);
}

test "dense matmul primary output liveness ignores internal sidecar reads" {
    var matmul = testMatmul(1);
    matmul.matmul.dst = 1;
    const scratch_only = [_]backend_mod.DeviceOp{
        matmul,
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 4 } },
    };
    try std.testing.expect(!matmulPrimaryOutputHasExternalUsers(&scratch_only, 0, 1));

    const external_read = [_]backend_mod.DeviceOp{
        matmul,
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 1, .src1 = 5, .n = 4 } },
    };
    try std.testing.expect(matmulPrimaryOutputHasExternalUsers(&external_read, 0, 1));
}

test "dense matmul primary output liveness ignores cache store sidecars" {
    var matmul = testMatmul(1);
    matmul.matmul.dst = 1;

    var store = testQMatmulSidecar(1, 8);
    store.slice_assign.cols = 1;
    const slice_only = [_]backend_mod.DeviceOp{ matmul, store };
    try std.testing.expect(!matmulPrimaryOutputHasExternalUsers(&slice_only, 0, 1));

    var prefill = testMatmul(2);
    prefill.matmul.dst = 1;
    const prefill_slice_only = [_]backend_mod.DeviceOp{ prefill, testQMatmulSidecar(1, 8) };
    try std.testing.expect(!matmulPrimaryOutputHasExternalUsers(&prefill_slice_only, 0, 1));

    var rope_pair = testRopeSliceAssignOps();
    rope_pair[0].rope.src = 1;
    rope_pair[0].rope.src_off = 0;
    rope_pair[0].rope.seq_len = 1;
    rope_pair[1].slice_assign.cols = 1;
    const rope_only = [_]backend_mod.DeviceOp{ matmul, rope_pair[0], rope_pair[1] };
    const rope_sidecars = [_]?usize{ 1, 2 };
    try std.testing.expect(!matmulPrimaryOutputHasExternalUsersExcept(&rope_only, 0, &rope_sidecars));
}

test "projection sidecar chains reject incompatible consumers" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 7 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
}

test "projection primary output liveness ignores internal sidecar reads" {
    const scratch_only = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&scratch_only, 0, 1));

    const external_read = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 1, .src1 = 5, .n = 8 } },
    };
    try std.testing.expect(projectionPrimaryOutputHasExternalUsers(&external_read, 0, 1));

    const overwritten = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
        testQMatmulWith(1, 4, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 5, .src0 = 1, .src1 = 6, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&overwritten, 0, 1));

    const inplace_sidecar = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 1, .src1 = 3, .n = 8 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 1, .src1 = 5, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&inplace_sidecar, 0, 1));
}

test "program command stream emits contiguous batch commands" {
    var rope_pair = testRopeSliceAssignOps();
    rope_pair[1] = rope_pair[0];
    rope_pair[1].rope.src_off = 4;
    rope_pair[1].rope.dst_off = 12;

    const ops = [_]backend_mod.DeviceOp{
        rope_pair[0],
        rope_pair[1],
        testQMatmulSidecar(1, 8),
        testQMatmulSidecar(1, 8),
        testAttention(0),
        testAttention(8),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_batch, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.movement_batch, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_count);
    try std.testing.expectEqual(ProgramCommandKind.attention_group, commands[2].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[2].anchor_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.commands);
    try std.testing.expectEqual(@as(u32, 6), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_batches);
    try std.testing.expectEqual(@as(u32, 1), summary.movement_batches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_group_ops);
}

test "program command stream emits noncontiguous attention groups" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        testQMatmulWith(9, 8, 2),
        testAttention(8),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_group_ops);
}

test "program command stream emits attention producer chains" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 2,
            .src = 9,
            .rows = 4,
            .cols = 4,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testAttention(0),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_chain_sidecars);
}

test "program command stream preserves attention output store over input sidecar" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 2,
            .src = 9,
            .rows = 4,
            .cols = 4,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testAttention(0),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[1].kind);
    try std.testing.expectEqual(@as(usize, 1), commands[1].indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[1].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 0), summary.attention_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
}

test "program command stream emits attention output store chains" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chain_sidecars);
}

test "program command stream groups attention output stores" {
    const att0 = testAttention(0);
    var att1 = testAttention(8);
    att1.attention.q = 10;
    att1.attention.k = 11;
    att1.attention.v = 12;
    att1.attention.dst = 13;

    const ops = [_]backend_mod.DeviceOp{
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 13,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_store_group_sidecars);
}

test "program command stream emits rope attention output store chains" {
    var att = testAttention(0);
    att.attention.q = 4;
    att.attention.k = 10;
    att.attention.dst = 5;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
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
        } },
        att,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chain_sidecars);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream groups rope attention output stores" {
    var att0 = testAttention(0);
    var att1 = testAttention(8);
    att0.attention.q = 4;
    att0.attention.dst = 5;
    att1.attention.q = 6;
    att1.attention.dst = 7;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
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
        } },
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        .{ .rope = .{
            .dst = 6,
            .src = 10,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 8,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 7,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 4), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[1]);
    try std.testing.expectEqual(@as(usize, 3), commands[0].indices[2]);
    try std.testing.expectEqual(@as(usize, 4), commands[0].indices[3]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 5), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 4), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_attention_store_group_sidecars);
    try std.testing.expectEqual(@as(u32, 6), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream groups wide shared-offset rope attention stores" {
    var ops: [15]backend_mod.DeviceOp = undefined;
    for (0..5) |h| {
        const off: u32 = @intCast(h * 8);
        ops[h * 3] = .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = off,
            .cs_off = off,
            .dst_off = off,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } };
        ops[h * 3 + 1] = testAttention(off);
        ops[h * 3 + 1].attention.q = 4;
        ops[h * 3 + 1].attention.dst = 5;
        ops[h * 3 + 1].attention.k_off = off;
        ops[h * 3 + 1].attention.v_off = off;
        ops[h * 3 + 1].attention.mask_off = off;
        ops[h * 3 + 2] = .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = off,
            .dst_row_stride = 1,
            .dst_col_stride = 20,
            .src_offset = off,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } };
    }

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 10), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 5), commands[0].sidecar_count);
    try std.testing.expect(ropeAttentionStoreCompactBatchCompatible(&ops, &commands[0]));

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 10), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.rope_attention_store_group_sidecars);
    try std.testing.expectEqual(@as(u32, 15), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream delays rope attention groups until inputs are ready" {
    var att0 = testAttention(0);
    var att1 = testAttention(8);
    att0.attention.q = 4;
    att0.attention.dst = 5;
    att1.attention.q = 6;
    att1.attention.dst = 7;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
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
        } },
        testQMatmulWith(10, 8, 2),
        .{ .rope = .{
            .dst = 6,
            .src = 10,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 7,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 5), commands[1].op_start);
    try std.testing.expectEqual(@as(u32, 4), commands[1].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[1].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[1].indices[0]);
    try std.testing.expectEqual(@as(usize, 3), commands[1].indices[1]);
    try std.testing.expectEqual(@as(usize, 2), commands[1].indices[2]);
    try std.testing.expectEqual(@as(usize, 5), commands[1].indices[3]);
    try std.testing.expectEqual(@as(?usize, 4), commands[1].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 6), commands[1].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 4), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_attention_store_group_sidecars);
}

test "program command stream delays single rope attention store chains" {
    var att = testAttention(0);
    att.attention.q = 4;
    att.attention.k = 10;
    att.attention.dst = 5;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
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
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 10,
            .src0 = 8,
            .src1 = 8,
            .n = 16,
        } },
        att,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_chain, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);
    try std.testing.expectEqual(@as(u32, 2), commands[1].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[1].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[1].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[1].indices[1]);
    try std.testing.expectEqual(@as(?usize, 3), commands[1].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chain_sidecars);
}

test "program command stream carries delayed attention output stores" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        testQMatmulWith(8, 0, 2),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].op_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chain_sidecars);
}

test "program command stream emits noncontiguous movement groups" {
    const ops = [_]backend_mod.DeviceOp{
        testSliceAssign(0),
        testQMatmulWith(9, 8, 2),
        testSliceAssign(4),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.movement_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.movement_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.movement_group_ops);
}

test "program command stream emits noncontiguous elementwise batch commands" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
        testQMatmulWith(9, 8, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 2, .src0 = 0, .src1 = 0, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.elementwise_batch, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.elementwise_batches);
    try std.testing.expectEqual(@as(u32, 2), summary.elementwise_ops);
}

test "program command stream fuses repeat feeding fused elementwise secondary" {
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 3, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 8,
            .dst = 4,
            .src = 5,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.repeat_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].op_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.repeat_fused_elementwise_chains);
}

test "program command stream fuses paired projection activation product chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(2, 8, 2),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 8,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 8,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testQMatmulWith(2, 8, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 8 } },
        testQMatmulWith(7, 8, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_pair_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 6), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "program command stream fuses qmatvec paired projection activation product chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(2, 8, 1),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 4,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 4,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testQMatmulWith(2, 8, 1),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 4 } },
        testQMatmulWith(7, 8, 1),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_pair_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 6), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "program command stream fuses dense paired projection activation product chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testMatmulWith(2, 8, 9, 1),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 4,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 4,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testMatmulWith(2, 8, 10, 1),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 4 } },
        testMatmulWith(7, 8, 11, 1),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_pair_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 6), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "program command stream fuses dense paired projection activation product prefill chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testMatmulWith(2, 8, 9, 2),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 8,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 8,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testMatmulWith(2, 8, 10, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 8 } },
        testMatmulWith(7, 8, 11, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.dense_projection_pair_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 6), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "dense paired projection liveness covers every prefill row" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testMatmulWith(2, 8, 9, 2),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 8,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 8,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testMatmulWith(2, 8, 10, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 8 } },
        .{ .elementwise = .{ .op = .add, .dst = 12, .src0 = 2, .src1 = 2, .src0_offset = 4, .src1_offset = 4, .n = 1 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    for (commands) |command| {
        try std.testing.expect(command.kind != .dense_projection_pair_fused_elementwise_chain);
    }
}

test "program command stream keeps live repeat outputs materialized" {
    const steps = [_]backend_mod.FusedEwStep{.{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 4,
            .src = 5,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .elementwise = .{ .op = .add, .dst = 6, .src0 = 2, .src1 = 5, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[2].kind);
}

test "program command stream keeps conflicting elementwise ops separate" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.grouped(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 0), summary.elementwise_batches);
    try std.testing.expectEqual(@as(u32, 2), summary.op_commands);
}

test "select pattern regions sorts candidates and removes overlaps" {
    const candidates = [_]PatternRegion{
        .{ .pattern_index = 0, .region = .{ .start_item = 2, .item_count = 2, .op_start = 2, .op_count = 2, .anchor_count = 2 } },
        .{ .pattern_index = 1, .region = .{ .start_item = 0, .item_count = 3, .op_start = 0, .op_count = 3, .anchor_count = 3 } },
        .{ .pattern_index = 2, .region = .{ .start_item = 4, .item_count = 1, .op_start = 4, .op_count = 1, .anchor_count = 1 } },
    };

    const selected = try selectPatternRegions(std.testing.allocator, &candidates);
    defer std.testing.allocator.free(selected);

    try std.testing.expectEqual(@as(usize, 2), selected.len);
    try std.testing.expectEqual(@as(u32, 1), selected[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), selected[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 2), selected[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 4), selected[1].region.start_item);
}

test "region schedule covers items once and replaces selected patterns" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 2 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 5, .len = 1 },
    };
    const regions = [_]PatternRegion{.{
        .pattern_index = 0,
        .region = .{ .start_item = 1, .item_count = 3, .op_start = 2, .op_count = 3, .anchor_count = 3 },
    }};
    const units = try buildRegionSchedule(std.testing.allocator, &items, &regions);
    defer std.testing.allocator.free(units);

    try std.testing.expectEqual(@as(usize, 3), units.len);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[0].kind);
    try std.testing.expectEqual(@as(u32, 0), units[0].op_start);
    try std.testing.expectEqual(@as(u32, 2), units[0].op_count);
    try std.testing.expectEqual(ScheduleUnitKind.pattern_region, units[1].kind);
    try std.testing.expectEqual(@as(u32, 0), units[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 1), units[1].start_item);
    try std.testing.expectEqual(@as(u32, 3), units[1].item_count);
    try std.testing.expectEqual(@as(u32, 2), units[1].op_start);
    try std.testing.expectEqual(@as(u32, 3), units[1].op_count);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[2].kind);
    try std.testing.expectEqual(@as(u32, 5), units[2].op_start);
}

test "region command plans compile commands only for pattern regions" {
    const ops = [_]backend_mod.DeviceOp{
        testElementwise(.add),
        testElementwise(.mul),
        testMatmul(1),
        testElementwise(.relu),
    };
    const units = [_]ScheduleUnit{
        .{
            .kind = .pattern_region,
            .pattern_index = 0,
            .start_item = 0,
            .item_count = 1,
            .op_start = 0,
            .op_count = 3,
        },
        .{
            .kind = .item,
            .start_item = 1,
            .item_count = 1,
            .op_start = 3,
            .op_count = 1,
        },
    };

    const plans = try buildRegionKernelPlans(std.testing.allocator, &ops, &units, Kernelizer.init(.{}));
    defer {
        deinitRegionKernelPlans(std.testing.allocator, plans);
        std.testing.allocator.free(plans);
    }

    try std.testing.expectEqual(@as(usize, units.len), plans.len);
    try std.testing.expect(plans[0].kernel_plan.commands.len > 0);
    try std.testing.expectEqual(@as(usize, 0), plans[1].kernel_plan.commands.len);

    const summary = summarizeProgramCommands(plans[0].kernel_plan.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
}

test "execution plan bundles schedule regions and cached command plans" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmul(16),
        testElementwise(.add),
        testQMatmul(16),
    };
    const schedule_policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .qmatmul = true, .elementwise = true },
        .fine_grained = true,
        .min_backend_qmatmul_m = 0,
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored(0, RegionPolicy.qmatmulCluster(), 1),
    };
    const kernelizer = Kernelizer.init(CommandStreamPolicy.grouped(4, 4));
    const plan = try kernelizer.executionPlan(
        std.testing.allocator,
        &ops,
        schedule_policy,
        &stages,
    );
    defer plan.deinit(std.testing.allocator);

    try std.testing.expect(plan.schedule.len > 0);
    try std.testing.expect(plan.regions.len > 0);
    try std.testing.expectEqual(plan.regions.len, plan.region_kernel_plans.len);
    try std.testing.expect(scheduleShapeMatches(&ops, plan.schedule, schedule_policy));
    try std.testing.expect(plan.regionCommandPlan(0).len > 0);
}

fn buildExecutionPlanAllocationCase(alloc: std.mem.Allocator) !void {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmul(16),
        testElementwise(.add),
        testQMatmul(16),
    };
    const schedule_policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .qmatmul = true, .elementwise = true },
        .fine_grained = true,
        .min_backend_qmatmul_m = 0,
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored(0, RegionPolicy.qmatmulCluster(), 1),
    };
    const plan = try Kernelizer.init(CommandStreamPolicy.grouped(4, 4)).executionPlan(alloc, &ops, schedule_policy, &stages);
    defer plan.deinit(alloc);
}

test "execution plan propagates planner allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, buildExecutionPlanAllocationCase, .{});
}

test "region execution summary counts backend islands and transitions" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .elementwise, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 5, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 6, .len = 1 },
    };
    const units = [_]ScheduleUnit{
        .{ .kind = .item, .start_item = 0, .item_count = 1, .op_start = 0, .op_count = 1 },
        .{ .kind = .pattern_region, .pattern_index = 0, .start_item = 1, .item_count = 2, .op_start = 1, .op_count = 2 },
        .{ .kind = .item, .start_item = 3, .item_count = 1, .op_start = 3, .op_count = 1 },
        .{ .kind = .item, .start_item = 4, .item_count = 1, .op_start = 4, .op_count = 1 },
        .{ .kind = .pattern_region, .pattern_index = 0, .start_item = 5, .item_count = 2, .op_start = 5, .op_count = 2 },
    };

    const backend_patterns = [_]u32{0};
    const summary = summarizeRegionExecution(&units, &items, &backend_patterns);
    try std.testing.expectEqual(@as(u32, 5), summary.units);
    try std.testing.expectEqual(@as(u32, 2), summary.backend_units);
    try std.testing.expectEqual(@as(u32, 3), summary.fallback_units);
    try std.testing.expectEqual(@as(u32, 4), summary.backend_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.fallback_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.backend_islands);
    try std.testing.expectEqual(@as(u32, 1), summary.max_backend_island_units);
    try std.testing.expectEqual(@as(u32, 2), summary.max_backend_island_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.execution_transitions);
}
