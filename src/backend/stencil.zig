//! Shape-only backend Adapter for offline ProgramStencil evidence.
//!
//! It does not execute kernels or copy model weights. It only compiles the
//! same ProgramStencil runtime patch table that an executable backend is
//! expected to report.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const program_mod = @import("program.zig");
const profile_mod = @import("../profile.zig");

pub const StencilBackend = struct {
    capabilities: backend_mod.Capabilities,
    name_str: []const u8 = "stencil",
    device_type: backend_mod.Device = .cpu,
    command_policy: program_mod.CommandStreamPolicy = program_mod.CommandStreamPolicy.default(),

    pub fn init(capabilities: backend_mod.Capabilities) StencilBackend {
        return .{ .capabilities = capabilities };
    }

    pub fn webgpuCompileOnly() StencilBackend {
        return .{
            .capabilities = backend_mod.Capabilities.webgpu_compile_only,
            .name_str = "webgpu-stencil",
            .device_type = .webgpu,
            .command_policy = program_mod.CommandStreamPolicy.default(),
        };
    }

    pub fn webgpuResourceSessionProbe() StencilBackend {
        return .{
            .capabilities = backend_mod.Capabilities.webgpu_resource_plan,
            .name_str = "webgpu-resource-stencil",
            .device_type = .webgpu,
            .command_policy = program_mod.CommandStreamPolicy.default(),
        };
    }

    pub fn backend(self: *StencilBackend) backend_mod.Backend {
        var capabilities = self.capabilities;
        capabilities.executes_programs = false;
        return .{
            .ctx = self,
            .vtable = &vtable,
            .name_str = self.name_str,
            .device_type = self.device_type,
            .capabilities = capabilities,
        };
    }
};

pub fn firstProjectionRowChainFrontierDebug(handle: backend_mod.Backend.CompiledHandle) program_mod.ProjectionRowChainFrontierDebug {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    const stencil = &compiled.program_stencil;
    return program_mod.firstProjectionRowChainFrontierDebug(
        stencil.ops,
        stencil.kernel_plan.commands,
        program_mod.CommandStreamPolicy.default(),
    );
}

pub fn firstProjectionElementwiseChainDebug(handle: backend_mod.Backend.CompiledHandle) program_mod.ProjectionElementwiseChainDebug {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    const stencil = &compiled.program_stencil;
    return program_mod.firstProjectionElementwiseChainDebug(
        stencil.ops,
        stencil.kernel_plan.commands,
    );
}

const CompiledStencil = struct {
    program_stencil: program_mod.ProgramStencil,
    default_runtime: RuntimeBindings,
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledStencil) void {
        self.default_runtime.deinit();
        self.program_stencil.deinit(self.alloc);
        self.alloc.destroy(self);
    }

    fn patchRuntimeWindow(self: *CompiledStencil, runtime: *RuntimeBindings, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
        _ = self;
        const status = runtime.program_stencil.patchRuntimeWindow(window);
        runtime.runtime_profile.recordRuntimePatch(status);
        return status;
    }
};

const RuntimeBindings = struct {
    program_stencil: program_mod.ProgramStencil,
    configured_persistent: []backend_mod.ProgramIO = &.{},
    configured_inputs: []backend_mod.ProgramIO = &.{},
    configured_outputs: []backend_mod.ProgramIO = &.{},
    runtime_profile: profile_mod.RuntimeProfile,
    alloc: std.mem.Allocator,

    fn init(alloc: std.mem.Allocator, program_stencil: program_mod.ProgramStencil) !RuntimeBindings {
        var runtime_stencil = try program_stencil.clone(alloc);
        errdefer runtime_stencil.deinit(alloc);
        const inspection = runtime_stencil.inspect();
        return .{
            .program_stencil = runtime_stencil,
            .configured_persistent = &.{},
            .configured_inputs = &.{},
            .configured_outputs = &.{},
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

    fn deinit(self: *RuntimeBindings) void {
        if (self.configured_persistent.len > 0) self.alloc.free(self.configured_persistent);
        if (self.configured_inputs.len > 0) self.alloc.free(self.configured_inputs);
        if (self.configured_outputs.len > 0) self.alloc.free(self.configured_outputs);
        self.program_stencil.deinit(self.alloc);
        self.* = undefined;
    }

    fn recordShapeExecution(self: *RuntimeBindings) void {
        self.runtime_profile.recordProgramCommandShapeDispatch(self.program_stencil.kernel_plan.command_shape);
        self.runtime_profile.call_count += 1;
    }
};

fn denseMatMulF32(_: *anyopaque, _: backend_mod.DenseMatMulSpecF32) bool {
    return false;
}

fn compileProgram(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self: *StencilBackend = @ptrCast(@alignCast(ctx));
    const alloc = std.heap.page_allocator;
    const compiled = alloc.create(CompiledStencil) catch return null;
    var program_stencil = program_mod.ProgramStencil.initProgramWithKernelizer(alloc, program, program_mod.Kernelizer.init(self.command_policy)) catch {
        alloc.destroy(compiled);
        return null;
    };
    const default_runtime = RuntimeBindings.init(alloc, program_stencil) catch {
        program_stencil.deinit(alloc);
        alloc.destroy(compiled);
        return null;
    };
    compiled.* = .{
        .program_stencil = program_stencil,
        .default_runtime = default_runtime,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn runtimeBindings(runtime: backend_mod.Backend.RuntimeHandle) *RuntimeBindings {
    return @ptrCast(@alignCast(runtime));
}

fn bindProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?backend_mod.Backend.RuntimeHandle {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    const runtime = compiled.alloc.create(RuntimeBindings) catch return null;
    runtime.* = RuntimeBindings.init(compiled.alloc, compiled.program_stencil) catch {
        compiled.alloc.destroy(runtime);
        return null;
    };
    return @ptrCast(runtime);
}

fn configureBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) bool {
    runtimeBindings(runtime).configure(persistent_inputs, step_inputs, step_outputs) catch return false;
    return true;
}

fn patchRuntimeWindow(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(&compiled.default_runtime, window);
}

fn patchRuntimeBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(runtimeBindings(runtime), window);
}

fn uploadProgram(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.ProgramIO) void {}

fn uploadBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, _: []const backend_mod.ProgramIO) void {}

fn uploadQWeights(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, _: []const backend_mod.QuantizedWeightUpload) bool {
    return true;
}

fn executeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, _: []const backend_mod.ProgramIO, _: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    compiled.default_runtime.recordShapeExecution();
}

fn executeBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, _: []const backend_mod.ProgramIO, _: []const backend_mod.ProgramIO) void {
    runtimeBindings(runtime).recordShapeExecution();
}

fn executeConfiguredBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, _: bool) void {
    runtimeBindings(runtime).recordShapeExecution();
}

fn freeBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle) void {
    const binding = runtimeBindings(runtime);
    const alloc = binding.alloc;
    binding.deinit();
    alloc.destroy(binding);
}

fn freeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn resetRuntimeProfile(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
    compiled.default_runtime.runtime_profile.reset();
}

fn addRuntimeProfileTo(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, dest: *profile_mod.RuntimeProfile) void {
    const compiled: *CompiledStencil = @ptrCast(@alignCast(handle));
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

test "StencilBackend reports canonical runtime patch stencil" {
    var stencil = StencilBackend.init(backend_mod.Capabilities.metal);
    const be = stencil.backend();
    var ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
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
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var profile = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &profile);
    try std.testing.expectEqual(backend_mod.RuntimePatchShape.actual(1, 0, profile.runtime_patch_shape.runtime_patch_stencil_hash), profile.runtime_patch_shape);
    try std.testing.expect(profile.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(2, 1)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.unchanged, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(2, 1)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(4, 1)));
    profile = .{};
    be.addRuntimeProfileTo(handle, &profile);
    try std.testing.expectEqual(@as(u64, 3), profile.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 1), profile.runtime_patch_changed_count);
    try std.testing.expectEqual(@as(u64, 1), profile.runtime_patch_invalid_count);
}

test "StencilBackend can advertise a WebGPU compile-only target" {
    var stencil = StencilBackend.webgpuCompileOnly();
    const be = stencil.backend();
    try std.testing.expectEqual(backend_mod.Capabilities.webgpu_compile_only, stencil.capabilities);
    try std.testing.expectEqual(backend_mod.Device.webgpu, be.device_type);
    try std.testing.expectEqualStrings("webgpu-stencil", be.name_str);
    try std.testing.expect(!be.capabilities.executes_programs);
    try std.testing.expect(!be.capabilities.external_resources);

    var ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
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
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var profile = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &profile);
    try std.testing.expectEqual(@as(u32, 1), profile.program_command_shape.command_count);
    try std.testing.expect(profile.program_command_shape.command_stencil_hash != 0);
    try std.testing.expectEqual(backend_mod.RuntimePatchShape.actual(1, 0, profile.runtime_patch_shape.runtime_patch_stencil_hash), profile.runtime_patch_shape);
}

test "StencilBackend WebGPU resource probe accepts resource sessions without executable support" {
    var stencil = StencilBackend.webgpuResourceSessionProbe();
    const be = stencil.backend();
    try std.testing.expectEqual(backend_mod.Capabilities.webgpu_resource_plan, stencil.capabilities);
    try std.testing.expectEqual(backend_mod.Device.webgpu, be.device_type);
    try std.testing.expectEqualStrings("webgpu-resource-stencil", be.name_str);
    try std.testing.expect(!be.capabilities.executes_programs);
    try std.testing.expect(be.capabilities.external_resources);

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
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &.{ 1, 1 }, .initial_uploads = &.{} };
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const runtime = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime);

    const input_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 11,
        .byte_len = @sizeOf(f32),
        .access = .read_only,
    };
    const output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 12,
        .byte_len = @sizeOf(f32),
        .access = .write_only,
    };
    const inputs = [_]backend_mod.ProgramIO{backend_mod.ProgramIO.external(0, 0, input_resource, @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{backend_mod.ProgramIO.external(1, 0, output_resource, @sizeOf(f32))};
    try std.testing.expect(be.supportsProgramIOLists(&inputs, &outputs));

    be.uploadBindings(handle, runtime, &inputs);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(0, 1)));
    be.executeBindings(handle, runtime, &inputs, &outputs);

    var profile = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &profile);
    try std.testing.expectEqual(@as(u32, 1), profile.call_count);
    try std.testing.expectEqual(@as(u64, 1), profile.backend_dispatch_count);
}

test "StencilBackend execution records command-shaped profile evidence" {
    var stencil = StencilBackend.init(backend_mod.Capabilities.metal);
    const be = stencil.backend();
    var ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
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
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const runtime = be.bindProgram(handle) orelse return error.CompileFailed;
    defer be.freeBindings(handle, runtime);

    be.executeBindings(handle, runtime, &.{}, &.{});

    var profile = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &profile);
    const op_command = @intFromEnum(program_mod.ProgramCommandKind.op);
    try std.testing.expectEqual(@as(u32, 1), profile.call_count);
    try std.testing.expectEqual(@as(u64, 1), profile.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_attempt_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 1), profile.program_command_dispatch_counts[op_command]);
    try std.testing.expectEqual(@as(u64, 0), profile.program_command_failed_counts[op_command]);
}
