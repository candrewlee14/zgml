//! CPU backend: BLAS matmul override for graph execution plus a compiled
//! DeviceProgram path backed by the shared reference executor.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const program_mod = @import("program.zig");
const profile_mod = @import("../profile.zig");
const reference = @import("reference.zig");
const forward = @import("../tensor/forward.zig");

pub const CpuBackend = struct {
    alloc: std.mem.Allocator = std.heap.page_allocator,

    pub fn init(alloc: std.mem.Allocator) CpuBackend {
        return .{ .alloc = alloc };
    }

    pub fn backend(self: *CpuBackend) backend_mod.Backend {
        return .{
            .ctx = self,
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

const PreparedQWeights = struct {
    qweights: []reference.QWeight,
    owned_qweights: []OwnedQWeight,
};

fn deinitPreparedQWeights(alloc: std.mem.Allocator, prepared: PreparedQWeights) void {
    for (prepared.owned_qweights) |qw| {
        alloc.free(qw.data);
        alloc.free(qw.scales);
        if (qw.t_data.len > 0) alloc.free(qw.t_data);
        if (qw.t_scales.len > 0) alloc.free(qw.t_scales);
    }
    if (prepared.owned_qweights.len > 0) alloc.free(prepared.owned_qweights);
    if (prepared.qweights.len > 0) alloc.free(prepared.qweights);
}

fn prepareQWeights(alloc: std.mem.Allocator, uploads: []const backend_mod.QuantizedWeightUpload) !PreparedQWeights {
    const qweights = try alloc.alloc(reference.QWeight, uploads.len);
    const owned_qweights = alloc.alloc(OwnedQWeight, uploads.len) catch |err| {
        if (qweights.len > 0) alloc.free(qweights);
        return err;
    };
    var n_qweights: usize = 0;
    errdefer {
        for (owned_qweights[0..n_qweights]) |qw| {
            alloc.free(qw.data);
            alloc.free(qw.scales);
            if (qw.t_data.len > 0) alloc.free(qw.t_data);
            if (qw.t_scales.len > 0) alloc.free(qw.t_scales);
        }
        if (owned_qweights.len > 0) alloc.free(owned_qweights);
        if (qweights.len > 0) alloc.free(qweights);
    }

    for (uploads, 0..) |qw, i| {
        const data = try alloc.dupe(i8, qw.data);
        const scales = alloc.dupe(f32, qw.scales) catch |err| {
            alloc.free(data);
            return err;
        };
        const desc = backend_mod.QuantizedWeightUpload{
            .data = data,
            .scales = scales,
            .rows = qw.rows,
            .cols = qw.cols,
            .block_size = qw.block_size,
        };
        qweights[i] = reference.prepareTransposedQWeight(alloc, desc) catch |err| {
            alloc.free(data);
            alloc.free(scales);
            return err;
        };
        owned_qweights[i] = .{
            .data = data,
            .scales = scales,
            .t_data = @constCast(qweights[i].t_data),
            .t_scales = @constCast(qweights[i].t_scales),
        };
        n_qweights += 1;
    }

    return .{ .qweights = qweights, .owned_qweights = owned_qweights };
}

const CompiledProgram = struct {
    base_buffer_table: reference.OwnedBufferTable,
    qweights: []reference.QWeight,
    owned_qweights: []OwnedQWeight,
    program_stencil: program_mod.ProgramStencil,
    execution_tape: reference.ExecutionTape,
    default_runtime: RuntimeBindings,
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledProgram) void {
        self.default_runtime.deinit();
        self.execution_tape.deinit(self.alloc);
        self.program_stencil.deinit(self.alloc);
        self.base_buffer_table.deinit();
        deinitPreparedQWeights(self.alloc, .{ .qweights = self.qweights, .owned_qweights = self.owned_qweights });
        self.alloc.destroy(self);
    }

    fn upload(self: *CompiledProgram, runtime: *RuntimeBindings, inputs: []const backend_mod.ProgramIO) void {
        _ = self;
        if (!runtime.program_stencil.ioValid(inputs, &.{})) return;
        uploadHostBindings(runtime.buffer_table, inputs);
    }

    fn execute(self: *CompiledProgram, runtime: *RuntimeBindings, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        if (!runtime.program_stencil.ioValid(inputs, outputs)) return;
        if (programIOListHasResources(runtime.configured_persistent) or
            programIOListHasResources(inputs) or
            programIOListHasResources(outputs))
        {
            std.debug.panic("cpu reference executor cannot execute external-resource bindings", .{});
        }
        uploadHostBindings(runtime.buffer_table, inputs);
        const qweights = if (runtime.qweights.len > 0) runtime.qweights else self.qweights;
        self.execution_tape.executeProfiled(runtime.buffer_table.buffers, qweights, runtime.program_stencil.ops, &runtime.runtime_profile);
        downloadHostBindings(runtime.buffer_table, outputs);
        const op_count: u64 = @intCast(runtime.program_stencil.ops.len);
        const command_count: u64 = @intCast(self.execution_tape.commandLen());
        runtime.runtime_profile.backend_op_count +%= op_count;
        runtime.runtime_profile.backend_dispatch_count +%= command_count;
        runtime.runtime_profile.call_count += 1;
    }

    fn patchRuntimeWindow(self: *CompiledProgram, runtime: *RuntimeBindings, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
        _ = self;
        const status = runtime.program_stencil.patchRuntimeWindow(window);
        runtime.runtime_profile.recordRuntimePatch(status);
        return status;
    }
};

fn programIOListHasResources(bindings: []const backend_mod.ProgramIO) bool {
    for (bindings) |io| if (!io.isHost()) return true;
    return false;
}

fn uploadHostBindings(buffer_table: reference.OwnedBufferTable, inputs: []const backend_mod.ProgramIO) void {
    for (inputs) |io| {
        if (!io.isHost()) continue;
        reference.uploadBindingToBuffers(buffer_table.buffers, io);
    }
}

fn downloadHostBindings(buffer_table: reference.OwnedBufferTable, outputs: []const backend_mod.ProgramIO) void {
    for (outputs) |io| {
        if (!io.isHost()) continue;
        reference.downloadBindingFromBuffers(buffer_table.buffers, io);
    }
}

const RuntimeBindings = struct {
    buffer_table: reference.OwnedBufferTable,
    program_stencil: program_mod.ProgramStencil,
    configured_persistent: []backend_mod.ProgramIO = &.{},
    configured_inputs: []backend_mod.ProgramIO = &.{},
    configured_outputs: []backend_mod.ProgramIO = &.{},
    qweights: []reference.QWeight = &.{},
    owned_qweights: []OwnedQWeight = &.{},
    runtime_profile: profile_mod.RuntimeProfile,
    alloc: std.mem.Allocator,

    fn init(alloc: std.mem.Allocator, base_buffer_table: reference.OwnedBufferTable, program_stencil: program_mod.ProgramStencil) !RuntimeBindings {
        var buffer_table = try base_buffer_table.clone(alloc);
        errdefer buffer_table.deinit();
        var runtime_stencil = try program_stencil.clone(alloc);
        errdefer runtime_stencil.deinit(alloc);
        const inspection = runtime_stencil.inspect();
        return .{
            .buffer_table = buffer_table,
            .program_stencil = runtime_stencil,
            .configured_persistent = &.{},
            .configured_inputs = &.{},
            .configured_outputs = &.{},
            .qweights = &.{},
            .owned_qweights = &.{},
            .runtime_profile = .{
                .runtime_patch_shape = inspection.runtime_patch_shape,
                .program_command_shape = inspection.command_shape,
            },
            .alloc = alloc,
        };
    }

    fn configure(self: *RuntimeBindings, persistent: []const backend_mod.ProgramIO, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) !void {
        if (!self.program_stencil.ioValid(persistent, &.{})) return error.InvalidProgramIO;
        if (!self.program_stencil.ioValid(inputs, outputs)) return error.InvalidProgramIO;
        const next_persistent = try self.alloc.dupe(backend_mod.ProgramIO, persistent);
        errdefer self.alloc.free(next_persistent);
        const next_inputs = try self.alloc.dupe(backend_mod.ProgramIO, inputs);
        errdefer self.alloc.free(next_inputs);
        const next_outputs = try self.alloc.dupe(backend_mod.ProgramIO, outputs);
        errdefer self.alloc.free(next_outputs);
        if (self.configured_persistent.len > 0) self.alloc.free(self.configured_persistent);
        if (self.configured_inputs.len > 0) self.alloc.free(self.configured_inputs);
        if (self.configured_outputs.len > 0) self.alloc.free(self.configured_outputs);
        self.configured_persistent = next_persistent;
        self.configured_inputs = next_inputs;
        self.configured_outputs = next_outputs;
    }

    fn uploadQWeights(self: *RuntimeBindings, qweights: []const backend_mod.QuantizedWeightUpload) !void {
        const prepared = try prepareQWeights(self.alloc, qweights);
        deinitPreparedQWeights(self.alloc, .{ .qweights = self.qweights, .owned_qweights = self.owned_qweights });
        self.qweights = prepared.qweights;
        self.owned_qweights = prepared.owned_qweights;
    }

    fn deinit(self: *RuntimeBindings) void {
        if (self.configured_persistent.len > 0) self.alloc.free(self.configured_persistent);
        if (self.configured_inputs.len > 0) self.alloc.free(self.configured_inputs);
        if (self.configured_outputs.len > 0) self.alloc.free(self.configured_outputs);
        deinitPreparedQWeights(self.alloc, .{ .qweights = self.qweights, .owned_qweights = self.owned_qweights });
        self.program_stencil.deinit(self.alloc);
        self.buffer_table.deinit();
        self.* = undefined;
    }
};

fn compileProgramInner(alloc: std.mem.Allocator, program: backend_mod.DeviceProgram) !*CompiledProgram {
    var buffer_table = try reference.OwnedBufferTable.init(alloc, program.buffer_sizes);
    errdefer buffer_table.deinit();
    buffer_table.upload(program.initial_uploads);

    const prepared_qweights = try prepareQWeights(alloc, program.qweights);
    errdefer deinitPreparedQWeights(alloc, prepared_qweights);

    const compiled = try alloc.create(CompiledProgram);
    errdefer alloc.destroy(compiled);
    var program_stencil = try program_mod.ProgramStencil.initProgram(alloc, program);
    errdefer program_stencil.deinit(alloc);
    var execution_tape = try reference.ExecutionTape.initFromCommands(alloc, program_stencil.ops, program_stencil.kernel_plan.commands);
    errdefer execution_tape.deinit(alloc);
    var default_runtime = try RuntimeBindings.init(alloc, buffer_table, program_stencil);
    errdefer default_runtime.deinit();
    compiled.* = .{
        .base_buffer_table = buffer_table,
        .qweights = prepared_qweights.qweights,
        .owned_qweights = prepared_qweights.owned_qweights,
        .program_stencil = program_stencil,
        .execution_tape = execution_tape,
        .default_runtime = default_runtime,
        .alloc = alloc,
    };
    return compiled;
}

fn cpuBackend(ctx: *anyopaque) *CpuBackend {
    return @ptrCast(@alignCast(ctx));
}

fn compileProgram(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const compiled = compileProgramInner(cpuBackend(ctx).alloc, program) catch return null;
    return @ptrCast(compiled);
}

fn bindProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?backend_mod.Backend.RuntimeHandle {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const runtime = compiled.alloc.create(RuntimeBindings) catch return null;
    runtime.* = RuntimeBindings.init(compiled.alloc, compiled.base_buffer_table, compiled.program_stencil) catch {
        compiled.alloc.destroy(runtime);
        return null;
    };
    return @ptrCast(runtime);
}

fn executeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(&compiled.default_runtime, inputs, outputs);
}

fn patchRuntimeWindow(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(&compiled.default_runtime, window);
}

fn uploadProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.upload(&compiled.default_runtime, inputs);
}

fn runtimeBindings(runtime: backend_mod.Backend.RuntimeHandle) *RuntimeBindings {
    return @ptrCast(@alignCast(runtime));
}

fn configureBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) bool {
    runtimeBindings(runtime).configure(persistent_inputs, step_inputs, step_outputs) catch return false;
    return true;
}

fn patchRuntimeBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(runtimeBindings(runtime), window);
}

fn uploadBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.upload(runtimeBindings(runtime), inputs);
}

fn uploadQWeights(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, qweights: []const backend_mod.QuantizedWeightUpload) bool {
    const bindings = runtimeBindings(runtime);
    bindings.uploadQWeights(qweights) catch return false;
    return true;
}

fn executeBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(runtimeBindings(runtime), inputs, outputs);
}

fn executeConfiguredBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, download_outputs: bool) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const bindings = runtimeBindings(runtime);
    const outputs = if (download_outputs) bindings.configured_outputs else &.{};
    compiled.execute(bindings, bindings.configured_inputs, outputs);
}

fn freeBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle) void {
    const binding = runtimeBindings(runtime);
    const alloc = binding.alloc;
    binding.deinit();
    alloc.destroy(binding);
}

fn freeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn resetRuntimeProfile(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.default_runtime.runtime_profile.reset();
}

fn addRuntimeProfileTo(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, dest: *profile_mod.RuntimeProfile) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    dest.add(compiled.default_runtime.runtime_profile);
}

fn resetRuntimeBindingsProfile(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle) void {
    runtimeBindings(runtime).runtime_profile.reset();
}

fn addRuntimeBindingsProfileTo(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, dest: *profile_mod.RuntimeProfile) void {
    dest.add(runtimeBindings(runtime).runtime_profile);
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgram,
    .bind_program = bindProgram,
    .configure_bindings = configureBindings,
    .patch_runtime_bindings = patchRuntimeBindings,
    .upload_bindings = uploadBindings,
    .upload_qweights = uploadQWeights,
    .execute_bindings = executeBindings,
    .execute_configured_bindings = executeConfiguredBindings,
    .free_bindings = freeBindings,
    .patch_runtime_window = patchRuntimeWindow,
    .upload_program = uploadProgram,
    .execute_program = executeProgram,
    .free_program = freeProgram,
    .reset_runtime_profile = resetRuntimeProfile,
    .add_runtime_profile_to = addRuntimeProfileTo,
    .reset_runtime_bindings_profile = resetRuntimeBindingsProfile,
    .add_runtime_bindings_profile_to = addRuntimeBindingsProfileTo,
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
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    try std.testing.expectEqual(@as(usize, ops.len), compiled.execution_tape.len());
    try std.testing.expectEqual(@as(usize, 1), compiled.execution_tape.commandLen());
    try std.testing.expectEqual(compiled.program_stencil.kernel_plan.commands.len, compiled.execution_tape.commandLen());
    try std.testing.expectEqual(@as(usize, @intCast(compiled.program_stencil.inspect().command_shape.command_count)), compiled.execution_tape.commandLen());

    var untouched = [_]f32{-1} ** 4;
    var bad_out = [_]backend_mod.ProgramIO{.{ .buf_idx = 3, .host_ptr = @ptrCast(&untouched), .size = untouched.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &bad_out);
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1 }, &untouched);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    try std.testing.expectEqual(@as(u32, 0), rt.call_count);

    var dst: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
    var rt_after = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt_after);
    const op_command = @intFromEnum(program_mod.ProgramCommandKind.op);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_attempt_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_dispatch_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 0), rt_after.program_command_failed_counts[op_command]);
}

test "cpu backend runtime bindings own independent buffer state" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 1, .N = 2, .K = 2, .a_row_stride = 2, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 2, 4, 2 };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &.{} };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);
    const runtime_a = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime_a);
    const runtime_b = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime_b);

    var input = [_]f32{ 2, 3 };
    var weights_a = [_]f32{ 1, 0, 0, 1 };
    var weights_b = [_]f32{ 2, 0, 0, 2 };
    const inputs = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * @sizeOf(f32) }};
    const bind_a = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&weights_a), .size = weights_a.len * @sizeOf(f32) }};
    const bind_b = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&weights_b), .size = weights_b.len * @sizeOf(f32) }};
    be.uploadBindings(handle, runtime_a, &bind_a);
    be.uploadBindings(handle, runtime_b, &bind_b);

    var out_a = [_]f32{0} ** 2;
    var out_b = [_]f32{0} ** 2;
    const output_a = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&out_a), .size = out_a.len * @sizeOf(f32) }};
    const output_b = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&out_b), .size = out_b.len * @sizeOf(f32) }};

    be.executeBindings(handle, runtime_a, &inputs, &output_a);
    be.executeBindings(handle, runtime_b, &inputs, &output_b);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &out_a);
    try std.testing.expectEqualSlices(f32, &.{ 4, 6 }, &out_b);

    out_a = .{ 0, 0 };
    be.executeBindings(handle, runtime_a, &inputs, &output_a);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &out_a);
}

test "cpu backend configure bindings owns complete session io table" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 1, .N = 2, .K = 2, .a_row_stride = 2, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 2, 4, 2 };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &.{} };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{ 1, 0, 0, 1 };
    var output = [_]f32{ 0, 0 };
    var persistent = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&weights), .size = weights.len * @sizeOf(f32) }};
    var inputs = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * @sizeOf(f32) }};
    var outputs = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&output), .size = output.len * @sizeOf(f32) }};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    const bindings = runtimeBindings(runtime);
    try std.testing.expectEqual(@as(usize, 1), bindings.configured_persistent.len);
    try std.testing.expectEqual(@as(usize, 1), bindings.configured_inputs.len);
    try std.testing.expectEqual(@as(usize, 1), bindings.configured_outputs.len);

    persistent[0].buf_idx = 0;
    inputs[0].buf_idx = 1;
    outputs[0].buf_idx = 0;
    try std.testing.expectEqual(@as(u16, 1), bindings.configured_persistent[0].buf_idx);
    try std.testing.expectEqual(@as(u16, 0), bindings.configured_inputs[0].buf_idx);
    try std.testing.expectEqual(@as(u16, 2), bindings.configured_outputs[0].buf_idx);
}

test "cpu backend runtime bindings own independent qweight state" {
    var cpu = CpuBackend{};
    const be = cpu.backend();
    try std.testing.expect(be.capabilities.runtime_qweights);

    const ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 2,
        .K = 2,
    } }};
    const buf_sizes = [_]usize{ 4, 4 };
    const qdata_identity = [_]i8{ 1, 0, 0, 1 };
    const qdata_double = [_]i8{ 2, 0, 0, 2 };
    const scales = [_]f32{ 1, 1 };
    const compile_qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata_identity,
        .scales = &scales,
        .rows = 2,
        .cols = 2,
        .block_size = 2,
    }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &buf_sizes, .initial_uploads = &.{}, .qweights = &compile_qweights };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);
    const runtime_a = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime_a);
    const runtime_b = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime_b);

    const qweights_a = [_]backend_mod.QuantizedWeightUpload{compile_qweights[0]};
    const qweights_b = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata_double,
        .scales = &scales,
        .rows = 2,
        .cols = 2,
        .block_size = 2,
    }};
    try std.testing.expect(be.uploadQWeights(handle, runtime_a, &qweights_a));
    try std.testing.expect(be.uploadQWeights(handle, runtime_b, &qweights_b));

    var input = [_]f32{ 2, 3, 4, 5 };
    const inputs = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * @sizeOf(f32) }};
    var out_a = [_]f32{0} ** 4;
    var out_b = [_]f32{0} ** 4;
    const output_a = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&out_a), .size = out_a.len * @sizeOf(f32) }};
    const output_b = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&out_b), .size = out_b.len * @sizeOf(f32) }};

    be.executeBindings(handle, runtime_a, &inputs, &output_a);
    be.executeBindings(handle, runtime_b, &inputs, &output_b);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 4, 5 }, &out_a);
    try std.testing.expectEqualSlices(f32, &.{ 4, 6, 8, 10 }, &out_b);
}

test "cpu backend exposes compiled runtime patch evidence" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    var src = [_]f32{42};
    var dst_init = [_]f32{0} ** 8;
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
    const buf_sizes = [_]usize{ 1, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = src.len * @sizeOf(f32) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&dst_init), .size = dst_init.len * @sizeOf(f32) },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var rt_before = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt_before);
    try std.testing.expectEqual(@as(u32, 1), rt_before.runtime_patch_shape.runtime_patch_holes);
    try std.testing.expectEqual(@as(u32, 1), rt_before.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(@as(u32, 0), rt_before.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);
    try std.testing.expect(rt_before.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(u32, 1), rt_before.program_command_shape.command_count);
    try std.testing.expect(rt_before.program_command_shape.command_stencil_hash != 0);
    try std.testing.expectEqual(@as(u32, 0), rt_before.call_count);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(3, 1)));
    var dst: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&dst), .size = dst.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &out);

    var rt_after = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt_after);
    try std.testing.expectEqual(rt_before.runtime_patch_shape, rt_after.runtime_patch_shape);
    try std.testing.expectEqual(@as(u64, 1), rt_after.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 1), rt_after.runtime_patch_changed_count);
    try std.testing.expectEqual(@as(u64, 1), rt_after.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt_after.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt_after.backend_dispatch_count);
    try std.testing.expectEqual(@as(u32, 1), rt_after.call_count);
    const op_command = @intFromEnum(program_mod.ProgramCommandKind.op);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_attempt_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), rt_after.program_command_dispatch_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 0), rt_after.program_command_failed_counts[op_command]);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 42, 0, 0, 0, 0 }, &dst);
}

fn compileCpuProgramWithOwnedPayloads(alloc: std.mem.Allocator) !void {
    var input = [_]f32{ 1, -2, 3, -4 };
    var qdst = [_]f32{0} ** 4;
    var out = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
            .input_offset = 0,
            .input_row_stride = 4,
            .dst_offset = 0,
            .dst_row_stride = 4,
        } },
        .{ .fused_elementwise = .{ .steps = &steps, .n = 4, .dst = 2, .src = 1, .dst_offset = 0, .src_offset = 0 } },
    };
    const buf_sizes = [_]usize{ 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * @sizeOf(f32) },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&qdst), .size = qdst.len * @sizeOf(f32) },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&out), .size = out.len * @sizeOf(f32) },
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 4,
        .cols = 4,
        .block_size = 4,
    }};
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads, .qweights = &qweights };

    const compiled = try compileProgramInner(alloc, program);
    defer compiled.deinit();
}

test "cpu backend compile cleanup survives allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        compileCpuProgramWithOwnedPayloads,
        .{},
    );
}
