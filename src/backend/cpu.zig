//! CPU backend: BLAS matmul override for graph execution plus a compiled
//! DeviceProgram path backed by the shared reference executor.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const reference = @import("reference.zig");
const forward = @import("../tensor/forward.zig");

pub const CpuBackend = struct {
    pub fn backend(self: *CpuBackend) backend_mod.Backend {
        _ = self;
        return .{
            .ctx = undefined,
            .vtable = &vtable,
            .name_str = "cpu",
            .device_type = .cpu,
            .capabilities = backend_mod.Capabilities.reference_cpu,
        };
    }
};

fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

const OwnedQWeight = struct {
    data: []i8,
    scales: []f32,
    t_data: []i8 = &.{},
    t_scales: []f32 = &.{},
};

const CompiledProgram = struct {
    buffer_table: reference.OwnedBufferTable,
    qweights: []reference.QWeight,
    owned_qweights: []OwnedQWeight,
    ops: []const backend_mod.DeviceOp,
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledProgram) void {
        self.buffer_table.deinit();
        for (self.owned_qweights) |qw| {
            self.alloc.free(qw.data);
            self.alloc.free(qw.scales);
            if (qw.t_data.len > 0) self.alloc.free(qw.t_data);
            if (qw.t_scales.len > 0) self.alloc.free(qw.t_scales);
        }
        if (self.owned_qweights.len > 0) self.alloc.free(self.owned_qweights);
        if (self.qweights.len > 0) self.alloc.free(self.qweights);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        self.buffer_table.upload(inputs);
        reference.executeProgram(self.buffer_table.buffers, self.qweights, self.ops);
        self.buffer_table.download(outputs);
    }
};

fn compileProgram(_: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const alloc = std.heap.page_allocator;

    var buffer_table = reference.OwnedBufferTable.init(alloc, program.buffer_sizes) catch return null;
    errdefer buffer_table.deinit();
    buffer_table.upload(program.initial_uploads);

    const qweights = alloc.alloc(reference.QWeight, program.qweights.len) catch return null;
    errdefer if (qweights.len > 0) alloc.free(qweights);
    const owned_qweights = alloc.alloc(OwnedQWeight, program.qweights.len) catch return null;
    var n_qweights: usize = 0;
    errdefer {
        for (owned_qweights[0..n_qweights]) |qw| {
            alloc.free(qw.data);
            alloc.free(qw.scales);
            if (qw.t_data.len > 0) alloc.free(qw.t_data);
            if (qw.t_scales.len > 0) alloc.free(qw.t_scales);
        }
        if (owned_qweights.len > 0) alloc.free(owned_qweights);
    }

    for (program.qweights, 0..) |qw, i| {
        const data = alloc.dupe(i8, qw.data) catch return null;
        const scales = alloc.dupe(f32, qw.scales) catch {
            alloc.free(data);
            return null;
        };
        const desc = backend_mod.QuantizedWeightUpload{
            .data = data,
            .scales = scales,
            .rows = qw.rows,
            .cols = qw.cols,
            .block_size = qw.block_size,
        };
        qweights[i] = reference.prepareTransposedQWeight(alloc, desc) catch {
            alloc.free(data);
            alloc.free(scales);
            return null;
        };
        owned_qweights[i] = .{
            .data = data,
            .scales = scales,
            .t_data = @constCast(qweights[i].t_data),
            .t_scales = @constCast(qweights[i].t_scales),
        };
        n_qweights += 1;
    }

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .buffer_table = buffer_table,
        .qweights = qweights,
        .owned_qweights = owned_qweights,
        .ops = program.ops,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn executeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn refreshProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, ops: []const backend_mod.DeviceOp) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.ops = ops;
}

fn freeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

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
        .dst = &dst,
        .a = &a,
        .b = &b,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });
    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "cpu backend compiled program matmul" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 6, 6, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&a_data), .size = 6 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&b_data), .size = 6 * 4 },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}
