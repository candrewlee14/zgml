//! High-level LLM inference facade.
//!
//! This module keeps the public inference surface small while the backend
//! implementation grows toward llama.cpp parity: explicit model/program/session
//! types, a streaming `prefill`/`step` API, and one `load(io, path)` entry point
//! for supported model files.

const std = @import("std");
const builtin = @import("builtin");
const opts = @import("zgml_options");

const backend_mod = @import("backend.zig");
const cpu_mod = @import("backend/cpu.zig");
const profile_mod = @import("profile.zig");
const sampling = @import("llm_sampling.zig");
const program_mod = @import("backend/program.zig");
const metal_mod = if (opts.use_metal and builtin.os.tag == .macos) @import("backend/metal.zig") else struct {
    pub const MetalBackend = struct {
        pub fn init() !MetalBackend {
            return error.MetalNotAvailable;
        }

        pub fn deinit(_: *MetalBackend) void {}

        pub fn backend(_: *MetalBackend) backend_mod.Backend {
            unreachable;
        }
    };
};
const wgpu_mod = if (opts.use_wgpu) @import("backend/wgpu.zig") else struct {
    pub const WgpuBackend = struct {
        pub fn init(_: std.mem.Allocator) WgpuBackend {
            unreachable;
        }

        pub fn deinit(_: *WgpuBackend) void {}

        pub fn backend(_: *WgpuBackend) backend_mod.Backend {
            unreachable;
        }
    };

    pub fn createDeviceBuffer(_: backend_mod.Backend.CompiledHandle, _: usize, _: backend_mod.ProgramIO.ExternalResource.Access) !backend_mod.ProgramIO.ExternalResource {
        return error.WgpuUnavailable;
    }

    pub fn writeDeviceBuffer(_: backend_mod.Backend.CompiledHandle, _: backend_mod.ProgramIO.ExternalResource, _: usize, _: []const u8) !void {
        return error.WgpuUnavailable;
    }

    pub fn readDeviceBuffer(_: backend_mod.Backend.CompiledHandle, _: backend_mod.ProgramIO.ExternalResource, _: usize, _: []u8) !void {
        return error.WgpuUnavailable;
    }

    pub fn deviceHandle(_: backend_mod.Backend.CompiledHandle) !usize {
        return error.WgpuUnavailable;
    }

    pub fn importDeviceBuffer(_: backend_mod.Backend.CompiledHandle, _: usize, _: usize, _: usize, _: usize, _: backend_mod.ProgramIO.ExternalResource.Access) !backend_mod.ProgramIO.ExternalResource {
        return error.WgpuUnavailable;
    }

    pub fn releaseDeviceBuffer(_: backend_mod.Backend.CompiledHandle, _: backend_mod.ProgramIO.ExternalResource) void {}

    pub const DispatchFamilyCounts = struct {
        attention: u64 = 0,
        pub fn projectionCount(_: DispatchFamilyCounts) u64 {
            return 0;
        }
        pub fn quantizedProjectionCount(_: DispatchFamilyCounts) u64 {
            return 0;
        }
    };

    pub const DispatchPlanInspection = struct {
        supported: bool = false,
        total_op_count: u64 = 0,
        covered_op_count: u64 = 0,
        dispatch_count: u64 = 0,
        first_unsupported_op: ?u64 = null,
        family_counts: DispatchFamilyCounts = .{},
        pub fn fullCoverage(_: DispatchPlanInspection) bool {
            return false;
        }
    };

    pub fn inspectCompiledDispatchPlan(_: backend_mod.Backend.CompiledHandle) DispatchPlanInspection {
        return .{};
    }
};
const stencil_mod = @import("backend/stencil.zig");
const llama_inference = @import("llama_inference.zig");

pub const LlamaConfig = @import("models/llama.zig").LlamaConfig;

pub const LlamaBackend = enum {
    auto,
    cpu,
    metal,
    webgpu,
};

pub const LlamaCompileOptions = struct {
    context_len: ?usize = null,
    batch: usize = 1,
    backend: LlamaBackend = .auto,
};

pub const LlamaBindOptions = struct {
    pub const HostBuffer = struct {
        ptr: [*]u8,
        byte_len: usize,
    };

    pub const CacheResources = struct {
        k: []const backend_mod.ProgramIO.ExternalResource,
        v: []const backend_mod.ProgramIO.ExternalResource,
        element_count: ?usize = null,
    };

    pub const CacheBuffers = struct {
        k: []const HostBuffer,
        v: []const HostBuffer,
        element_count: ?usize = null,
    };

    output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
    cache_resources: ?CacheResources = null,
    cache_buffers: ?CacheBuffers = null,
};

pub const LlamaStepParams = struct {
    token: usize,
};

pub const LlamaSampleOptions = sampling.Options;

pub const LlamaExecuteParams = struct {
    pub const Output = enum {
        none,
        logits,
    };

    tokens: []const usize,
    output: Output = .logits,
};

pub const LlamaProgramInspection = struct {
    vocab_size: usize,
    max_seq_len: usize,
    context_len: usize,
    batch: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    semantic_stage_count: usize,
    semantic_token_count: usize,
    semantic_layer_stage_count: usize,
    semantic_terminal_stage_count: usize,
    semantic_runtime_patch_holes: usize,
    semantic_runtime_patch_cache_write_pos_holes: usize,
    semantic_runtime_patch_attention_seq_kv_holes: usize,
};

pub const LlamaExecutionMode = enum {
    executable,
    resource_probe,
    compile_only,
};

fn llamaExecutionMode(execution_supported: bool, external_resources_supported: bool) LlamaExecutionMode {
    if (execution_supported) return .executable;
    if (external_resources_supported) return .resource_probe;
    return .compile_only;
}

pub const LlamaExecutableInspection = struct {
    backend: LlamaBackend,
    execution_supported: bool,
    external_resources_supported: bool,
    execution_mode: LlamaExecutionMode,
    buffer_count: usize,
    buffer_element_count: usize,
    buffer_byte_len: usize,
    initial_upload_count: usize,
    qweight_count: usize,
    op_count: usize,
    binding_requirement_hash: u64,
    persistent_requirement_count: usize,
    step_input_requirement_count: usize,
    step_output_requirement_count: usize,
    command_count: u64,
    command_stencil_hash: u64,
    command_op_count: u64,
    command_row_count: u64,
    command_projection_count: u64,
    command_attention_count: u64,
    command_movement_count: u64,
    command_elementwise_count: u64,
    command_rope_count: u64,
    runtime_patch_holes: usize,
    runtime_patch_cache_write_pos_holes: usize,
    runtime_patch_attention_seq_kv_holes: usize,
    runtime_patch_stencil_hash: u64,
    runtime_patch_envelope: program_mod.RuntimePatchEnvelope,
    execution_plan: backend_mod.ExecutionPlanInspection = .{},
};

pub const LlamaSessionInspection = struct {
    backend: LlamaBackend,
    position: usize,
    context_len: usize,
    output_storage: u32,
    kv_cache_storage: u32,
    persistent_binding_count: usize,
    step_input_count: usize,
    step_output_count: usize,
    host_binding_count: usize,
    resource_binding_count: usize,
    binding_shape_hash: u64,
};

fn LlamaFacade(comptime T: type, comptime config: LlamaConfig) type {
    const Inner = llama_inference.LlamaInferenceSession(T, config);
    const ExecutableProgram = Inner.DeviceDecodeProgram;
    const ExecutableSession = Inner.DeviceDecodeSession;
    const native_llama_webgpu_execution_enabled = opts.experimental_llama_wgpu_execution;

    const ModelState = struct {
        alloc: std.mem.Allocator,
        inner: Inner,
    };

    const ProgramState = struct {
        alloc: std.mem.Allocator,
        model: *ModelState,
        source_model: *ModelState,
        options: LlamaCompileOptions,
        executable_inspection: LlamaExecutableInspection,
        decode_program: ?ExecutableProgram,
        prefill_program: ?ExecutableProgram,
        prefill_chunk: usize,
        cpu_backend: cpu_mod.CpuBackend,
        metal_backend: ?*metal_mod.MetalBackend,
        webgpu_backend: ?*wgpu_mod.WgpuBackend,
        webgpu_stencil: stencil_mod.StencilBackend,
    };

    const SessionState = struct {
        alloc: std.mem.Allocator,
        model: *ModelState,
        owns_model: bool,
        program: ?*ProgramState,
        decode_session: ?ExecutableSession,
        prefill_session: ?ExecutableSession,
        prefill_chunk: usize,
        logits_buf: []T,
        output_resource: ?backend_mod.ProgramIO.ExternalResource,
        cache_bindings: ?ExecutableProgram.CacheBindings,
        prefill_output_resource: ?backend_mod.ProgramIO.ExternalResource,
        prefill_cache_bindings: ?ExecutableProgram.CacheBindings,
        owns_prefill_imports: bool,
    };

    return struct {
        fn initModelStateWithContext(alloc: std.mem.Allocator, context_len: usize) !*ModelState {
            const model = try alloc.create(ModelState);
            errdefer alloc.destroy(model);
            model.* = .{
                .alloc = alloc,
                .inner = try Inner.initWithBackendAndContext(alloc, null, context_len),
            };
            return model;
        }

        fn initModelState(alloc: std.mem.Allocator) !*ModelState {
            return initModelStateWithContext(alloc, config.max_seq_len);
        }

        fn deinitModelState(model: *ModelState) void {
            const alloc = model.alloc;
            model.inner.deinit();
            alloc.destroy(model);
        }

        fn validateCompileOptions(options: LlamaCompileOptions) !void {
            if (options.batch == 0) return error.InvalidBatch;
            if (options.batch != 1) return error.UnsupportedBatch;
            if (options.context_len) |context_len| {
                if (context_len == 0 or context_len > config.max_seq_len) return error.InvalidContextLength;
            }
        }

        fn effectiveContextLen(st: *const ProgramState) usize {
            return st.options.context_len orelse config.max_seq_len;
        }

        fn executablePrefillChunk(st: *const ProgramState) usize {
            return @min(llama_inference.default_prefill_chunk, effectiveContextLen(st));
        }

        fn hasExecutablePrefill(st: *const ProgramState) bool {
            return executablePrefillChunk(st) > 1;
        }

        fn tokenFromStepArg(arg: anytype) usize {
            const Arg = @TypeOf(arg);
            return switch (@typeInfo(Arg)) {
                .comptime_int, .int => @intCast(arg),
                .@"struct" => if (@hasField(Arg, "token"))
                    arg.token
                else
                    @compileError("LLaMA step params must include a token field"),
                else => @compileError("LLaMA step expects a token id or params with a token field"),
            };
        }

        fn isTokenWindowArg(comptime Arg: type) bool {
            return switch (@typeInfo(Arg)) {
                .@"struct" => @hasField(Arg, "tokens"),
                else => false,
            };
        }

        fn executeParamsFromArg(arg: anytype) LlamaExecuteParams {
            const Arg = @TypeOf(arg);
            const output: LlamaExecuteParams.Output = if (@hasField(Arg, "output")) arg.output else .logits;
            return .{
                .tokens = arg.tokens,
                .output = output,
            };
        }

        fn evidenceBackend(choice: LlamaBackend) LlamaBackend {
            return switch (choice) {
                .auto => .cpu,
                .cpu, .metal, .webgpu => choice,
            };
        }

        fn placeholderExecutableInspection(options: LlamaCompileOptions) LlamaExecutableInspection {
            return .{
                .backend = evidenceBackend(options.backend),
                .execution_supported = false,
                .external_resources_supported = false,
                .execution_mode = .compile_only,
                .buffer_count = 0,
                .buffer_element_count = 0,
                .buffer_byte_len = 0,
                .initial_upload_count = 0,
                .qweight_count = 0,
                .op_count = 0,
                .binding_requirement_hash = 0,
                .persistent_requirement_count = 0,
                .step_input_requirement_count = 0,
                .step_output_requirement_count = 0,
                .command_count = 0,
                .command_stencil_hash = 0,
                .command_op_count = 0,
                .command_row_count = 0,
                .command_projection_count = 0,
                .command_attention_count = 0,
                .command_movement_count = 0,
                .command_elementwise_count = 0,
                .command_rope_count = 0,
                .runtime_patch_holes = 0,
                .runtime_patch_cache_write_pos_holes = 0,
                .runtime_patch_attention_seq_kv_holes = 0,
                .runtime_patch_stencil_hash = 0,
                .runtime_patch_envelope = .{},
                .execution_plan = .{},
            };
        }

        fn executableInspectionFromDevice(options: LlamaCompileOptions, inspection: anytype) LlamaExecutableInspection {
            const command_categories = inspection.command_shape.categoryCounts();
            return .{
                .backend = evidenceBackend(options.backend),
                .execution_supported = inspection.execution_supported,
                .external_resources_supported = inspection.external_resources_supported,
                .execution_mode = llamaExecutionMode(inspection.execution_supported, inspection.external_resources_supported),
                .buffer_count = inspection.buffer_count,
                .buffer_element_count = inspection.buffer_element_count,
                .buffer_byte_len = inspection.buffer_byte_len,
                .initial_upload_count = inspection.initial_upload_count,
                .qweight_count = inspection.qweight_count,
                .op_count = inspection.op_count,
                .binding_requirement_hash = inspection.binding_requirement_hash,
                .persistent_requirement_count = inspection.persistent_requirement_count,
                .step_input_requirement_count = inspection.step_input_requirement_count,
                .step_output_requirement_count = inspection.step_output_requirement_count,
                .command_count = inspection.command_shape.command_count,
                .command_stencil_hash = inspection.command_shape.command_stencil_hash,
                .command_op_count = command_categories.op,
                .command_row_count = command_categories.row,
                .command_projection_count = command_categories.projection,
                .command_attention_count = command_categories.attention,
                .command_movement_count = command_categories.movement,
                .command_elementwise_count = command_categories.elementwise,
                .command_rope_count = command_categories.rope,
                .runtime_patch_holes = inspection.runtime_patch_shape.runtime_patch_holes,
                .runtime_patch_cache_write_pos_holes = inspection.runtime_patch_shape.runtime_patch_cache_write_pos_holes,
                .runtime_patch_attention_seq_kv_holes = inspection.runtime_patch_shape.runtime_patch_attention_seq_kv_holes,
                .runtime_patch_stencil_hash = inspection.runtime_patch_shape.runtime_patch_stencil_hash,
                .runtime_patch_envelope = inspection.runtime_patch_envelope,
                .execution_plan = inspection.execution_plan,
            };
        }

        fn bindingStorageCode(storage: anytype) u32 {
            return switch (storage) {
                .none => 0,
                .host => 1,
                .external_resource => 2,
            };
        }

        fn addDeviceSessionInspection(dest: *LlamaSessionInspection, session: *const ExecutableSession) void {
            const inspection = session.device_session.inspect();
            dest.persistent_binding_count = inspection.persistent_binding_count;
            dest.step_input_count = inspection.step_input_count;
            dest.step_output_count = inspection.step_output_count;
            dest.output_storage = bindingStorageCode(inspection.output_storage);
            dest.host_binding_count += inspection.host_binding_count;
            dest.resource_binding_count += inspection.resource_binding_count;
            dest.binding_shape_hash = inspection.binding_shape_hash;
        }

        fn ensureMetalBackend(st: *ProgramState) !*metal_mod.MetalBackend {
            if (st.metal_backend) |metal| return metal;
            const metal = try st.alloc.create(metal_mod.MetalBackend);
            errdefer st.alloc.destroy(metal);
            metal.* = try metal_mod.MetalBackend.init();
            st.metal_backend = metal;
            return metal;
        }

        fn ensureWebGpuBackend(st: *ProgramState) !?*wgpu_mod.WgpuBackend {
            if (!opts.use_wgpu) return null;
            if (st.webgpu_backend) |wgpu| return wgpu;
            const wgpu = try st.alloc.create(wgpu_mod.WgpuBackend);
            errdefer st.alloc.destroy(wgpu);
            wgpu.* = wgpu_mod.WgpuBackend.init(st.alloc);
            st.webgpu_backend = wgpu;
            return wgpu;
        }

        fn clearWebGpuBackend(st: *ProgramState) void {
            if (!opts.use_wgpu) return;
            if (st.webgpu_backend) |wgpu| {
                wgpu.deinit();
                st.alloc.destroy(wgpu);
                st.webgpu_backend = null;
            }
        }

        fn selectedBackend(st: *ProgramState) !backend_mod.Backend {
            return switch (st.options.backend) {
                .auto, .cpu => st.cpu_backend.backend(),
                .metal => (try ensureMetalBackend(st)).backend(),
                .webgpu => if (st.webgpu_backend) |wgpu| wgpu.backend() else st.webgpu_stencil.backend(),
            };
        }

        fn unsupportedNativeWebGpu(err: anyerror) bool {
            return switch (err) {
                error.UnsupportedDeviceOp, error.WebGPUExecutionUnavailable, error.WebGPUUnavailable, error.WgpuUnavailable => true,
                else => false,
            };
        }

        fn clearExecutablePrograms(st: *ProgramState) void {
            if (st.decode_program) |*program| {
                program.deinit();
                st.decode_program = null;
            }
            if (st.prefill_program) |*program| {
                program.deinit();
                st.prefill_program = null;
            }
            st.prefill_chunk = 0;
        }

        fn tryNativeWebGpuPrograms(st: *ProgramState) !bool {
            if (st.options.backend != .webgpu or !opts.use_wgpu) return false;
            // The native wgpu executor can compile the current LLaMA op tape, but
            // broader LLaMA-family coverage is not yet a default support claim.
            // Keep public LLaMA WebGPU on the resource-probe path unless the
            // explicit experimental execution gate is enabled.
            if (!native_llama_webgpu_execution_enabled) return false;
            if ((try ensureWebGpuBackend(st)) == null) return false;

            const decode = ensureDecodeProgram(st) catch |err| {
                if (unsupportedNativeWebGpu(err)) {
                    clearExecutablePrograms(st);
                    clearWebGpuBackend(st);
                    return false;
                }
                return err;
            };
            st.executable_inspection = executableInspectionFromDevice(st.options, decode.inspectExecutable());
            if (!st.executable_inspection.execution_supported) {
                clearExecutablePrograms(st);
                clearWebGpuBackend(st);
                return false;
            }

            if (hasExecutablePrefill(st)) {
                _ = ensurePrefillProgram(st) catch |err| {
                    if (unsupportedNativeWebGpu(err)) {
                        clearExecutablePrograms(st);
                        clearWebGpuBackend(st);
                        st.executable_inspection = placeholderExecutableInspection(st.options);
                        return false;
                    }
                    return err;
                };
            }
            return true;
        }

        fn deinitProgramState(st: *ProgramState) void {
            const alloc = st.alloc;
            clearExecutablePrograms(st);
            if (st.metal_backend) |metal| {
                metal.deinit();
                alloc.destroy(metal);
            }
            clearWebGpuBackend(st);
            deinitModelState(st.model);
            alloc.destroy(st);
        }

        fn resetProgramRuntimeProfileState(st: *ProgramState) void {
            if (st.decode_program) |*program| program.resetRuntimeProfile();
            if (st.prefill_program) |*program| program.resetRuntimeProfile();
        }

        fn addProgramRuntimeProfileState(st: *ProgramState, dest: *profile_mod.RuntimeProfile) void {
            if (st.decode_program) |*program| program.addRuntimeProfileTo(dest);
            if (st.prefill_program) |*program| program.addRuntimeProfileTo(dest);
        }

        fn resetSessionRuntimeProfileState(st: *SessionState) void {
            if (st.decode_session) |*session| session.resetRuntimeProfile();
            if (st.prefill_session) |*session| session.resetRuntimeProfile();
        }

        fn addSessionRuntimeProfileState(st: *SessionState, dest: *profile_mod.RuntimeProfile) void {
            if (st.decode_session) |*session| session.addRuntimeProfileTo(dest);
            if (st.prefill_session) |*session| session.addRuntimeProfileTo(dest);
        }

        fn ensureDecodeProgram(st: *ProgramState) !*ExecutableProgram {
            if (st.decode_program == null) {
                const backend = try selectedBackend(st);
                st.decode_program = try st.model.inner.compileDeviceDecodeProgram(backend, st.alloc);
            }
            return &st.decode_program.?;
        }

        fn ensurePrefillProgram(st: *ProgramState) !*ExecutableProgram {
            const chunk_tokens = executablePrefillChunk(st);
            if (chunk_tokens == 0) return error.InvalidPrefillLength;
            if (st.prefill_program) |*program| {
                if (st.prefill_chunk == chunk_tokens) return program;
                program.deinit();
                st.prefill_program = null;
                st.prefill_chunk = 0;
            }
            const backend = try selectedBackend(st);
            st.prefill_program = try st.model.inner.compileDevicePrefillProgram(backend, st.alloc, chunk_tokens);
            st.prefill_chunk = chunk_tokens;
            return &st.prefill_program.?;
        }

        fn bindModelState(st: *ProgramState, source: *ModelState) !*ModelState {
            const session_model = try st.alloc.create(ModelState);
            errdefer st.alloc.destroy(session_model);
            session_model.* = .{
                .alloc = st.alloc,
                .inner = try source.inner.bindRuntimeWithContext(st.alloc, effectiveContextLen(st)),
            };
            errdefer session_model.inner.deinit();
            return session_model;
        }

        fn ensureLogitsBuf(st: *SessionState) ![]T {
            if (st.logits_buf.len == 0) {
                st.logits_buf = try st.alloc.alloc(T, config.vocab_size);
            }
            return st.logits_buf;
        }

        fn clearDecodeSession(st: *SessionState) void {
            if (st.decode_session) |*session| {
                session.deinit();
                st.decode_session = null;
            }
        }

        fn clearPrefillSession(st: *SessionState) void {
            if (st.prefill_session) |*session| {
                session.deinit();
                st.prefill_session = null;
                st.prefill_chunk = 0;
            }
        }

        fn releasePrefillResourceImports(st: *SessionState) void {
            if (!st.owns_prefill_imports) return;
            const program_state = st.program orelse return;
            if (program_state.prefill_program) |*program| {
                const handle = program.program.handle;
                if (st.prefill_output_resource) |resource| {
                    wgpu_mod.releaseDeviceBuffer(handle, resource);
                }
                if (st.prefill_cache_bindings) |bindings| switch (bindings) {
                    .resource => |resources| {
                        for (0..config.n_layers) |l| {
                            wgpu_mod.releaseDeviceBuffer(handle, resources.k[l]);
                            wgpu_mod.releaseDeviceBuffer(handle, resources.v[l]);
                        }
                    },
                    .host => {},
                };
            }
            st.prefill_output_resource = null;
            st.prefill_cache_bindings = null;
            st.owns_prefill_imports = false;
        }

        fn ensureDecodeSession(st: *SessionState) !*ExecutableSession {
            if (st.decode_session == null) {
                const program_state = st.program orelse return error.UnsupportedDeviceOp;
                const program = try ensureDecodeProgram(program_state);
                const logits_buf = try ensureLogitsBuf(st);
                st.decode_session = try program.bindWithOptions(&st.model.inner, .{
                    .logits_buf = logits_buf,
                    .output_resource = st.output_resource,
                    .cache_bindings = st.cache_bindings,
                });
            }
            return &st.decode_session.?;
        }

        fn ensurePrefillSession(st: *SessionState) !*ExecutableSession {
            const program_state = st.program orelse return error.UnsupportedDeviceOp;
            const chunk_tokens = executablePrefillChunk(program_state);
            if (chunk_tokens == 0) return error.InvalidPrefillLength;
            if (st.prefill_session) |*session| {
                if (st.prefill_chunk == chunk_tokens) return session;
                session.deinit();
                st.prefill_session = null;
                st.prefill_chunk = 0;
            }
            const program = try ensurePrefillProgram(program_state);
            const logits_buf = try ensureLogitsBuf(st);
            const bind_options = try prefillBindOptions(st, program);
            st.prefill_session = try program.bindWithOptions(&st.model.inner, .{
                .logits_buf = logits_buf,
                .output_resource = bind_options.output_resource,
                .cache_bindings = bind_options.cache_bindings,
            });
            st.prefill_chunk = chunk_tokens;
            return &st.prefill_session.?;
        }

        fn importResourceForProgram(
            program: *ExecutableProgram,
            device_handle: usize,
            resource: backend_mod.ProgramIO.ExternalResource,
            access: backend_mod.ProgramIO.ExternalResource.Access,
        ) !backend_mod.ProgramIO.ExternalResource {
            if (resource.placement != .webgpu) return error.UnsupportedResourceBinding;
            return wgpu_mod.importDeviceBuffer(program.program.handle, device_handle, resource.handle, resource.byte_offset, resource.byte_len, access);
        }

        fn ensurePrefillResourceImports(st: *SessionState, prefill_program: *ExecutableProgram) !void {
            if (st.owns_prefill_imports) return;
            const needs_output_import = st.output_resource != null;
            const needs_cache_import = if (st.cache_bindings) |bindings| switch (bindings) {
                .resource => true,
                .host => false,
            } else false;
            if (!needs_output_import and !needs_cache_import) return;

            const program_state = st.program orelse return error.UnsupportedDeviceOp;
            if (program_state.options.backend != .webgpu or !program_state.executable_inspection.execution_supported) return error.UnsupportedResourceBinding;
            const decode_program = try ensureDecodeProgram(program_state);
            const device_handle = try wgpu_mod.deviceHandle(decode_program.program.handle);

            if (st.output_resource) |resource| {
                st.prefill_output_resource = try importResourceForProgram(prefill_program, device_handle, resource, resource.access);
                st.owns_prefill_imports = true;
            }

            if (st.cache_bindings) |bindings| switch (bindings) {
                .resource => |resources| {
                    var imported: ExecutableProgram.CacheResourceBindings = undefined;
                    imported.element_count = resources.element_count;
                    for (0..config.n_layers) |l| {
                        imported.k[l] = try importResourceForProgram(prefill_program, device_handle, resources.k[l], resources.k[l].access);
                        imported.v[l] = try importResourceForProgram(prefill_program, device_handle, resources.v[l], resources.v[l].access);
                    }
                    st.prefill_cache_bindings = .{ .resource = imported };
                    st.owns_prefill_imports = true;
                },
                .host => {},
            };
        }

        fn prefillBindOptions(st: *SessionState, prefill_program: *ExecutableProgram) !ExecutableProgram.BindOptions {
            try ensurePrefillResourceImports(st, prefill_program);
            const output_resource = if (st.prefill_output_resource) |resource| @as(?backend_mod.ProgramIO.ExternalResource, resource) else st.output_resource;
            const cache_bindings = if (st.prefill_cache_bindings) |bindings| @as(?ExecutableProgram.CacheBindings, bindings) else st.cache_bindings;
            return .{
                .logits_buf = st.logits_buf,
                .output_resource = output_resource,
                .cache_bindings = cache_bindings,
            };
        }

        fn refreshDecodeSession(st: *SessionState) !void {
            if (st.decode_session) |*session| try session.refreshCacheBindings();
        }

        fn refreshResetRuntimeBindings(st: *SessionState) void {
            if (st.decode_session) |*session| {
                session.refreshCacheBindings() catch {
                    clearDecodeSession(st);
                };
            }
            if (st.prefill_session) |*session| {
                session.refreshCacheBindings() catch {
                    clearPrefillSession(st);
                };
            }
        }

        fn ensureContextCapacity(st: *SessionState, tokens: usize) !void {
            if (tokens == 0) return;
            const program_state = st.program orelse return;
            const next_position = std.math.add(usize, st.model.inner.position(), tokens) catch return error.SequenceTooLong;
            if (next_position > effectiveContextLen(program_state)) return error.SequenceTooLong;
        }

        pub const Model = opaque {
            const Self = @This();

            fn state(self: *Self) *ModelState {
                return @ptrCast(@alignCast(self));
            }

            fn constState(self: *const Self) *const ModelState {
                return @ptrCast(@alignCast(self));
            }

            pub fn init(alloc: std.mem.Allocator) !*Self {
                return @ptrCast(try initModelState(alloc));
            }

            pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !*Self {
                const model = try Self.init(alloc);
                errdefer model.deinit();
                try model.state().inner.load(io, path);
                return model;
            }

            pub fn loadSafetensorsBytes(alloc: std.mem.Allocator, bytes: []const u8) !*Self {
                const model = try Self.init(alloc);
                errdefer model.deinit();
                try model.state().inner.loadSafetensorsBytes(bytes);
                return model;
            }

            pub fn loadInto(self: *Self, io: std.Io, path: []const u8) !void {
                return self.state().inner.load(io, path);
            }

            pub fn loadSafetensorsBytesInto(self: *Self, bytes: []const u8) !void {
                return self.state().inner.loadSafetensorsBytes(bytes);
            }

            pub fn compile(self: *Self, options: LlamaCompileOptions) !*Program {
                try validateCompileOptions(options);
                const st = self.state();
                var compile_model = try st.inner.bindRuntimeWithContext(st.alloc, options.context_len orelse config.max_seq_len);
                var compile_model_owned = true;
                errdefer if (compile_model_owned) compile_model.deinit();
                const compile_model_state = try st.alloc.create(ModelState);
                var compile_model_state_owned = true;
                errdefer if (compile_model_state_owned) st.alloc.destroy(compile_model_state);
                compile_model_state.* = .{
                    .alloc = st.alloc,
                    .inner = compile_model,
                };
                const program = try st.alloc.create(ProgramState);
                program.* = .{
                    .alloc = st.alloc,
                    .model = compile_model_state,
                    .source_model = st,
                    .options = options,
                    .executable_inspection = placeholderExecutableInspection(options),
                    .decode_program = null,
                    .prefill_program = null,
                    .prefill_chunk = 0,
                    .cpu_backend = cpu_mod.CpuBackend.init(st.alloc),
                    .metal_backend = null,
                    .webgpu_backend = null,
                    .webgpu_stencil = stencil_mod.StencilBackend.webgpuResourceSessionProbe(),
                };
                compile_model_owned = false;
                compile_model_state_owned = false;
                errdefer deinitProgramState(program);
                if (try tryNativeWebGpuPrograms(program)) return @ptrCast(program);

                const decode = try ensureDecodeProgram(program);
                program.executable_inspection = executableInspectionFromDevice(options, decode.inspectExecutable());
                if (hasExecutablePrefill(program)) {
                    if (program.executable_inspection.execution_supported) {
                        _ = try ensurePrefillProgram(program);
                    }
                }
                return @ptrCast(program);
            }

            pub fn reset(self: *Self) void {
                self.state().inner.reset();
            }

            pub fn position(self: *const Self) usize {
                return self.constState().inner.position();
            }

            pub fn deinit(self: *Self) void {
                deinitModelState(self.state());
            }
        };

        pub const Program = opaque {
            const Self = @This();

            fn state(self: *Self) *ProgramState {
                return @ptrCast(@alignCast(self));
            }

            pub fn bind(self: *Self, options: LlamaBindOptions) !*Session {
                return self.bindExecutableDecode(options);
            }

            pub fn bindModel(self: *Self, model: *Model, options: LlamaBindOptions) !*Session {
                return self.bindExecutableDecodeFromModel(model, options);
            }

            pub fn bindExecutableDecode(self: *Self, options: LlamaBindOptions) !*Session {
                return self.bindExecutableDecodeFromState(self.state().source_model, options);
            }

            pub fn bindExecutableDecodeFromModel(self: *Self, model: *Model, options: LlamaBindOptions) !*Session {
                return self.bindExecutableDecodeFromState(model.state(), options);
            }

            fn bindExecutableDecodeFromState(self: *Self, source_model: *ModelState, options: LlamaBindOptions) !*Session {
                const st = self.state();
                if (!st.executable_inspection.execution_supported and !canBindNonExecutingResourceSession(st, options)) return error.WebGPUExecutionUnavailable;
                if (options.output_resource) |resource| {
                    if (!resource.canWrite()) return error.UnsupportedResourceBinding;
                }
                const cache_bindings = try cacheBindingsFromOptions(st, options);
                const session_model = try bindModelState(st, source_model);
                errdefer deinitModelState(session_model);

                const session = try st.alloc.create(SessionState);
                errdefer st.alloc.destroy(session);
                session.* = .{
                    .alloc = st.alloc,
                    .model = session_model,
                    .owns_model = true,
                    .program = st,
                    .decode_session = null,
                    .prefill_session = null,
                    .prefill_chunk = 0,
                    .logits_buf = &.{},
                    .output_resource = options.output_resource,
                    .cache_bindings = cache_bindings,
                    .prefill_output_resource = null,
                    .prefill_cache_bindings = null,
                    .owns_prefill_imports = false,
                };
                errdefer {
                    clearDecodeSession(session);
                    clearPrefillSession(session);
                    if (session.logits_buf.len > 0) st.alloc.free(session.logits_buf);
                }
                _ = try ensureDecodeSession(session);
                if (hasExecutablePrefill(st) and st.executable_inspection.execution_supported) {
                    _ = try ensurePrefillSession(session);
                }
                return @ptrCast(session);
            }

            fn canBindNonExecutingResourceSession(st: *ProgramState, options: LlamaBindOptions) bool {
                return !st.executable_inspection.execution_supported and
                    st.executable_inspection.external_resources_supported and
                    options.output_resource != null and
                    options.cache_resources != null and
                    options.cache_buffers == null;
            }

            fn cacheBindingsFromOptions(st: *ProgramState, options: LlamaBindOptions) !?ExecutableProgram.CacheBindings {
                if (options.cache_resources != null and options.cache_buffers != null) return error.ShapeMismatch;
                const element_count = if (options.cache_resources) |resources|
                    resources.element_count orelse try ExecutableProgram.cacheElementCountForContext(effectiveContextLen(st))
                else if (options.cache_buffers) |buffers|
                    buffers.element_count orelse try ExecutableProgram.cacheElementCountForContext(effectiveContextLen(st))
                else
                    return null;

                if (options.cache_resources) |resources| {
                    if (resources.k.len != config.n_layers or resources.v.len != config.n_layers) return error.ShapeMismatch;
                    var bindings: ExecutableProgram.CacheResourceBindings = undefined;
                    bindings.element_count = element_count;
                    for (0..config.n_layers) |l| {
                        if (!resources.k[l].canRead() or !resources.k[l].canWrite()) return error.UnsupportedResourceBinding;
                        if (!resources.v[l].canRead() or !resources.v[l].canWrite()) return error.UnsupportedResourceBinding;
                        bindings.k[l] = resources.k[l];
                        bindings.v[l] = resources.v[l];
                    }
                    return .{ .resource = bindings };
                }

                const buffers = options.cache_buffers.?;
                if (buffers.k.len != config.n_layers or buffers.v.len != config.n_layers) return error.ShapeMismatch;
                const byte_len = std.math.mul(usize, element_count, @sizeOf(T)) catch return error.ShapeMismatch;
                var bindings: ExecutableProgram.CacheHostBindings = undefined;
                bindings.element_count = element_count;
                for (0..config.n_layers) |l| {
                    if (buffers.k[l].byte_len < byte_len or buffers.v[l].byte_len < byte_len) return error.ShapeMismatch;
                    if (@intFromPtr(buffers.k[l].ptr) % @alignOf(T) != 0 or @intFromPtr(buffers.v[l].ptr) % @alignOf(T) != 0) return error.ShapeMismatch;
                    bindings.k[l] = @ptrCast(@alignCast(buffers.k[l].ptr));
                    bindings.v[l] = @ptrCast(@alignCast(buffers.v[l].ptr));
                }
                return .{ .host = bindings };
            }

            pub fn inspect(self: *Self) LlamaProgramInspection {
                const st = self.state();
                const shape = st.model.inner.semanticShape();
                return .{
                    .vocab_size = config.vocab_size,
                    .max_seq_len = config.max_seq_len,
                    .context_len = effectiveContextLen(st),
                    .batch = st.options.batch,
                    .d_model = config.d_model,
                    .n_layers = config.n_layers,
                    .n_heads = config.n_heads,
                    .n_kv_heads = config.n_kv_heads,
                    .semantic_stage_count = shape.semantic_stage_count,
                    .semantic_token_count = shape.semantic_token_count,
                    .semantic_layer_stage_count = shape.semantic_layer_stage_count,
                    .semantic_terminal_stage_count = shape.semantic_terminal_stage_count,
                    .semantic_runtime_patch_holes = shape.semantic_runtime_patch_holes,
                    .semantic_runtime_patch_cache_write_pos_holes = shape.semantic_runtime_patch_cache_write_pos_holes,
                    .semantic_runtime_patch_attention_seq_kv_holes = shape.semantic_runtime_patch_attention_seq_kv_holes,
                };
            }

            pub fn inspectExecutable(self: *Self) LlamaExecutableInspection {
                return self.state().executable_inspection;
            }

            pub fn resetRuntimeProfile(self: *Self) void {
                resetProgramRuntimeProfileState(self.state());
            }

            pub fn addRuntimeProfileTo(self: *Self, dest: *profile_mod.RuntimeProfile) void {
                addProgramRuntimeProfileState(self.state(), dest);
            }

            fn requireWebGpuExecutableProgram(self: *Self) !*ExecutableProgram {
                const st = self.state();
                if (st.options.backend != .webgpu or !st.executable_inspection.execution_supported or st.webgpu_backend == null) {
                    return error.UnsupportedDeviceOp;
                }
                return ensureDecodeProgram(st);
            }

            pub fn deviceHandle(self: *Self) !usize {
                const program = try self.requireWebGpuExecutableProgram();
                return wgpu_mod.deviceHandle(program.program.handle);
            }

            pub fn createDeviceBuffer(
                self: *Self,
                byte_len: usize,
                access: backend_mod.ProgramIO.ExternalResource.Access,
            ) !backend_mod.ProgramIO.ExternalResource {
                const program = try self.requireWebGpuExecutableProgram();
                return wgpu_mod.createDeviceBuffer(program.program.handle, byte_len, access);
            }

            pub fn importDeviceBuffer(
                self: *Self,
                device_handle: usize,
                buffer_handle: usize,
                byte_offset: usize,
                byte_len: usize,
                access: backend_mod.ProgramIO.ExternalResource.Access,
            ) !backend_mod.ProgramIO.ExternalResource {
                const program = try self.requireWebGpuExecutableProgram();
                return wgpu_mod.importDeviceBuffer(program.program.handle, device_handle, buffer_handle, byte_offset, byte_len, access);
            }

            pub fn writeDeviceBuffer(
                self: *Self,
                resource: backend_mod.ProgramIO.ExternalResource,
                byte_offset: usize,
                src: []const u8,
            ) !void {
                const program = try self.requireWebGpuExecutableProgram();
                try wgpu_mod.writeDeviceBuffer(program.program.handle, resource, byte_offset, src);
            }

            pub fn readDeviceBuffer(
                self: *Self,
                resource: backend_mod.ProgramIO.ExternalResource,
                byte_offset: usize,
                dst: []u8,
            ) !void {
                const program = try self.requireWebGpuExecutableProgram();
                try wgpu_mod.readDeviceBuffer(program.program.handle, resource, byte_offset, dst);
            }

            pub fn releaseDeviceBuffer(self: *Self, resource: backend_mod.ProgramIO.ExternalResource) void {
                const program = self.requireWebGpuExecutableProgram() catch return;
                wgpu_mod.releaseDeviceBuffer(program.program.handle, resource);
            }

            pub fn deinit(self: *Self) void {
                deinitProgramState(self.state());
            }
        };

        pub const Session = opaque {
            const Self = @This();
            pub const TokenScore = sampling.TokenScore(T);
            pub const GenerateResult = struct {
                tokens_generated: usize,
                last_token: usize,
                last_logit: T,
            };

            fn state(self: *Self) *SessionState {
                return @ptrCast(@alignCast(self));
            }

            fn constState(self: *const Self) *const SessionState {
                return @ptrCast(@alignCast(self));
            }

            pub fn init(alloc: std.mem.Allocator) !*Self {
                const model = try initModelState(alloc);
                errdefer deinitModelState(model);
                const session = try alloc.create(SessionState);
                errdefer alloc.destroy(session);
                session.* = .{
                    .alloc = alloc,
                    .model = model,
                    .owns_model = true,
                    .program = null,
                    .decode_session = null,
                    .prefill_session = null,
                    .prefill_chunk = 0,
                    .logits_buf = &.{},
                    .output_resource = null,
                    .cache_bindings = null,
                    .prefill_output_resource = null,
                    .prefill_cache_bindings = null,
                    .owns_prefill_imports = false,
                };
                return @ptrCast(session);
            }

            pub fn deinit(self: *Self) void {
                const st = self.state();
                const alloc = st.alloc;
                clearDecodeSession(st);
                clearPrefillSession(st);
                releasePrefillResourceImports(st);
                if (st.logits_buf.len > 0) alloc.free(st.logits_buf);
                if (st.owns_model) deinitModelState(st.model);
                alloc.destroy(st);
            }

            pub fn reset(self: *Self) void {
                const st = self.state();
                st.model.inner.reset();
                refreshResetRuntimeBindings(st);
            }

            pub fn resetRuntimeProfile(self: *Self) void {
                resetSessionRuntimeProfileState(self.state());
            }

            pub fn addRuntimeProfileTo(self: *Self, dest: *profile_mod.RuntimeProfile) void {
                addSessionRuntimeProfileState(self.state(), dest);
            }

            pub fn position(self: *const Self) usize {
                return self.constState().model.inner.position();
            }

            pub fn inspect(self: *const Self) LlamaSessionInspection {
                const st = self.constState();
                var out = LlamaSessionInspection{
                    .backend = if (st.program) |program| evidenceBackend(program.options.backend) else .cpu,
                    .position = st.model.inner.position(),
                    .context_len = if (st.program) |program| effectiveContextLen(program) else config.max_seq_len,
                    .output_storage = if (st.output_resource != null) 2 else 0,
                    .kv_cache_storage = 0,
                    .persistent_binding_count = 0,
                    .step_input_count = 0,
                    .step_output_count = 0,
                    .host_binding_count = 0,
                    .resource_binding_count = 0,
                    .binding_shape_hash = 0,
                };
                if (st.cache_bindings) |bindings| {
                    out.kv_cache_storage = switch (bindings) {
                        .host => 1,
                        .resource => 2,
                    };
                } else if (st.program != null) {
                    out.kv_cache_storage = 1;
                }
                if (st.decode_session) |*decode| {
                    addDeviceSessionInspection(&out, decode);
                }
                return out;
            }

            fn ensureGenerateCapacity(st: *SessionState, prompt_len: usize, output_len: usize) !void {
                if (prompt_len == 0) return error.InvalidPrefillLength;
                if (output_len == 0) return error.InvalidTokenWindow;
                const generated_steps = std.math.add(usize, prompt_len, output_len - 1) catch return error.SequenceTooLong;
                try ensureContextCapacity(st, generated_steps);
            }

            fn ensureExecutionSupported(st: *SessionState) !void {
                const program = st.program orelse return;
                if (!program.executable_inspection.execution_supported) return error.WebGPUExecutionUnavailable;
            }

            fn executeTokenWindowInternal(self: *Self, output: ?[]T, params: LlamaExecuteParams) ![]const T {
                if (params.tokens.len == 0) return error.InvalidPrefillLength;
                try ensureExecutionSupported(self.state());
                return switch (params.output) {
                    .none => blk: {
                        try self.executeTokenWindowNoOutput(params.tokens);
                        break :blk &.{};
                    },
                    .logits => self.executeTokenWindowLogits(output, params.tokens),
                };
            }

            fn executeDecodeWindowNoOutput(st: *SessionState, tokens: []const usize) !void {
                const decode = try ensureDecodeSession(st);
                for (tokens) |token| try decode.advance(token);
            }

            fn executeDecodeWindowLogits(st: *SessionState, output: ?[]T, tokens: []const usize) ![]const T {
                const decode = try ensureDecodeSession(st);
                for (tokens[0 .. tokens.len - 1]) |token| try decode.advance(token);
                const last = tokens[tokens.len - 1];
                return if (output) |out|
                    try decode.stepInto(out, last)
                else
                    try decode.step(last);
            }

            fn executeTokenWindowNoOutput(self: *Self, tokens: []const usize) !void {
                const st = self.state();
                if (st.program != null) {
                    try ensureContextCapacity(st, tokens.len);
                    const program_state = st.program.?;
                    const chunk = executablePrefillChunk(program_state);
                    var offset: usize = 0;
                    var used_prefill = false;
                    if (chunk > 1 and tokens.len >= chunk) {
                        const prefill_session = try ensurePrefillSession(st);
                        while (offset + chunk <= tokens.len) : (offset += chunk) {
                            try prefill_session.advanceTokens(tokens[offset..][0..chunk]);
                            used_prefill = true;
                        }
                    }
                    if (used_prefill) try refreshDecodeSession(st);
                    if (offset < tokens.len) try executeDecodeWindowNoOutput(st, tokens[offset..]);
                    return;
                }
                _ = try st.model.inner.prefill(tokens);
            }

            fn executeTokenWindowLogits(self: *Self, output: ?[]T, tokens: []const usize) ![]const T {
                if (output) |out| {
                    if (out.len < config.vocab_size) return error.OutputBufferTooSmall;
                }
                const st = self.state();
                if (st.program != null) {
                    try ensureContextCapacity(st, tokens.len);
                    const program_state = st.program.?;
                    const chunk = executablePrefillChunk(program_state);
                    var offset: usize = 0;
                    var used_prefill = false;
                    if (chunk > 1 and tokens.len >= chunk) {
                        const prefill_session = try ensurePrefillSession(st);
                        while (offset + chunk < tokens.len) : (offset += chunk) {
                            try prefill_session.advanceTokens(tokens[offset..][0..chunk]);
                            used_prefill = true;
                        }
                        if (tokens.len - offset == chunk) {
                            const logits = if (output) |out|
                                try prefill_session.prefillInto(out, tokens[offset..][0..chunk])
                            else
                                try prefill_session.prefill(tokens[offset..][0..chunk]);
                            try refreshDecodeSession(st);
                            return logits;
                        }
                    }
                    if (used_prefill) try refreshDecodeSession(st);
                    return executeDecodeWindowLogits(st, output, tokens[offset..]);
                }

                if (output) |out| {
                    const logits = try st.model.inner.prefill(tokens);
                    @memcpy(out[0..config.vocab_size], logits);
                    return out[0..config.vocab_size];
                }
                return st.model.inner.prefill(tokens);
            }

            fn executeDecodeWindowBoundOutput(st: *SessionState, tokens: []const usize) !void {
                const decode = try ensureDecodeSession(st);
                for (tokens[0 .. tokens.len - 1]) |token| try decode.advance(token);
                try decode.stepBoundOutput(tokens[tokens.len - 1]);
            }

            fn executeTokenWindowBoundOutput(self: *Self, tokens: []const usize) !void {
                if (tokens.len == 0) return error.InvalidPrefillLength;
                const st = self.state();
                if (st.output_resource == null) return error.UnsupportedResourceBinding;
                try ensureExecutionSupported(st);
                const program_state = st.program orelse return error.UnsupportedDeviceOp;
                try ensureContextCapacity(st, tokens.len);
                const chunk = executablePrefillChunk(program_state);
                var offset: usize = 0;
                var used_prefill = false;
                if (chunk > 1 and tokens.len >= chunk) {
                    const prefill_session = try ensurePrefillSession(st);
                    while (offset + chunk < tokens.len) : (offset += chunk) {
                        try prefill_session.advanceTokens(tokens[offset..][0..chunk]);
                        used_prefill = true;
                    }
                    if (tokens.len - offset == chunk) {
                        try prefill_session.prefillBoundOutput(tokens[offset..][0..chunk]);
                        try refreshDecodeSession(st);
                        return;
                    }
                }
                if (used_prefill) try refreshDecodeSession(st);
                try executeDecodeWindowBoundOutput(st, tokens[offset..]);
            }

            fn readBoundOutputLogits(st: *SessionState) ![]const T {
                const output_resource = st.output_resource orelse return error.UnsupportedResourceBinding;
                const program_state = st.program orelse return error.UnsupportedDeviceOp;
                const decode_program = try ensureDecodeProgram(program_state);
                const logits = try ensureLogitsBuf(st);
                try wgpu_mod.readDeviceBuffer(decode_program.program.handle, output_resource, 0, std.mem.sliceAsBytes(logits[0..config.vocab_size]));
                return logits[0..config.vocab_size];
            }

            fn executeTokenWindowForSelection(self: *Self, tokens: []const usize) ![]const T {
                if (self.state().output_resource) |_| {
                    try self.executeTokenWindowBoundOutput(tokens);
                    return readBoundOutputLogits(self.state());
                }
                return self.executeTokenWindow(.{ .tokens = tokens, .output = .logits });
            }

            fn stepForSelection(self: *Self, token: usize) ![]const T {
                var window = [_]usize{token};
                return self.executeTokenWindowForSelection(window[0..]);
            }

            pub fn prefill(self: *Self, tokens: []const usize) ![]const T {
                if (tokens.len == 0) return self.state().model.inner.prefill(tokens);
                return self.executeTokenWindow(.{ .tokens = tokens, .output = .logits });
            }

            pub fn prefillInto(self: *Self, output: []T, tokens: []const usize) ![]T {
                if (output.len < config.vocab_size) return error.OutputBufferTooSmall;
                if (tokens.len == 0) {
                    const logits = try self.prefill(tokens);
                    @memcpy(output[0..config.vocab_size], logits);
                    return output[0..config.vocab_size];
                }
                _ = try self.executeTokenWindowInto(output, .{ .tokens = tokens, .output = .logits });
                return output[0..config.vocab_size];
            }

            pub fn advanceTokens(self: *Self, tokens: []const usize) !void {
                _ = try self.executeTokenWindow(.{ .tokens = tokens, .output = .none });
            }

            pub fn prefillBoundOutput(self: *Self, tokens: []const usize) !void {
                try self.executeTokenWindowBoundOutput(tokens);
            }

            pub fn step(self: *Self, arg: anytype) ![]const T {
                var token = [_]usize{tokenFromStepArg(arg)};
                return self.executeTokenWindow(.{ .tokens = token[0..], .output = .logits });
            }

            pub fn stepInto(self: *Self, output: []T, arg: anytype) ![]T {
                var token = [_]usize{tokenFromStepArg(arg)};
                return self.executeTokenWindowInto(output, .{ .tokens = token[0..], .output = .logits });
            }

            pub fn stepBoundOutput(self: *Self, arg: anytype) !void {
                var token = [_]usize{tokenFromStepArg(arg)};
                try self.executeTokenWindowBoundOutput(token[0..]);
            }

            pub fn stepArgmax(self: *Self, arg: anytype) !TokenScore {
                return sampling.argmax(T, try self.step(arg));
            }

            pub fn stepSample(self: *Self, arg: anytype, options: LlamaSampleOptions) !TokenScore {
                try sampling.validateOptions(options);
                return sampling.sampleTopK(T, try self.step(arg), options);
            }

            pub fn advance(self: *Self, arg: anytype) !void {
                var token = [_]usize{tokenFromStepArg(arg)};
                _ = try self.executeTokenWindow(.{ .tokens = token[0..], .output = .none });
            }

            fn executeTokenWindow(self: *Self, params: LlamaExecuteParams) ![]const T {
                return self.executeTokenWindowInternal(null, params);
            }

            fn executeTokenWindowInto(self: *Self, output: []T, params: LlamaExecuteParams) ![]T {
                _ = try self.executeTokenWindowInternal(output, params);
                return switch (params.output) {
                    .none => output[0..0],
                    .logits => output[0..config.vocab_size],
                };
            }

            pub fn execute(self: *Self, params: anytype) ![]const T {
                if (comptime isTokenWindowArg(@TypeOf(params))) return self.executeTokenWindow(executeParamsFromArg(params));
                return self.step(params);
            }

            pub fn executeInto(self: *Self, output: []T, params: anytype) ![]T {
                if (comptime isTokenWindowArg(@TypeOf(params))) return self.executeTokenWindowInto(output, executeParamsFromArg(params));
                return self.stepInto(output, params);
            }

            pub fn generateArgmaxInto(self: *Self, output_tokens: []usize, prompt_tokens: []const usize) !GenerateResult {
                try ensureGenerateCapacity(self.state(), prompt_tokens.len, output_tokens.len);

                var selected = try sampling.argmax(T, try self.executeTokenWindowForSelection(prompt_tokens));
                output_tokens[0] = selected.token;

                var generated: usize = 1;
                while (generated < output_tokens.len) {
                    selected = try sampling.argmax(T, try self.stepForSelection(selected.token));
                    output_tokens[generated] = selected.token;
                    generated += 1;
                }

                return .{
                    .tokens_generated = generated,
                    .last_token = selected.token,
                    .last_logit = selected.logit,
                };
            }

            pub fn generateSampleInto(self: *Self, output_tokens: []usize, prompt_tokens: []const usize, options: LlamaSampleOptions) !GenerateResult {
                try ensureGenerateCapacity(self.state(), prompt_tokens.len, output_tokens.len);
                try sampling.validateOptions(options);

                var selected = try sampling.sampleTopK(T, try self.executeTokenWindowForSelection(prompt_tokens), options);
                output_tokens[0] = selected.token;

                var generated: usize = 1;
                while (generated < output_tokens.len) {
                    if (generated > std.math.maxInt(u32)) return error.InvalidTokenWindow;
                    var step_options = options;
                    step_options.seed = options.seed +% @as(u32, @intCast(generated));
                    selected = try sampling.sampleTopK(T, try self.stepForSelection(selected.token), step_options);
                    output_tokens[generated] = selected.token;
                    generated += 1;
                }

                return .{
                    .tokens_generated = generated,
                    .last_token = selected.token,
                    .last_logit = selected.logit,
                };
            }

            pub fn load(self: *Self, io: std.Io, path: []const u8) !void {
                return self.state().model.inner.load(io, path);
            }
        };
    };
}

pub fn LlamaModel(comptime T: type, comptime config: LlamaConfig) type {
    return LlamaFacade(T, config).Model;
}

pub fn LlamaProgram(comptime T: type, comptime config: LlamaConfig) type {
    return LlamaFacade(T, config).Program;
}

/// Thin typed wrapper over the existing persistent LLaMA inference session.
/// Users get `prefill(tokens)` and `step(token)` without touching graph internals.
pub fn LlamaSession(comptime T: type, comptime config: LlamaConfig) type {
    return LlamaFacade(T, config).Session;
}

const testing = std.testing;

test "llm Interface stays curated" {
    const root = @This();
    const expected = .{
        "LlamaBackend",
        "LlamaBindOptions",
        "LlamaCompileOptions",
        "LlamaConfig",
        "LlamaExecuteParams",
        "LlamaExecutionMode",
        "LlamaExecutableInspection",
        "LlamaModel",
        "LlamaProgramInspection",
        "LlamaProgram",
        "LlamaSampleOptions",
        "LlamaSession",
        "LlamaSessionInspection",
        "LlamaStepParams",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
}

test "llm facade exposes WebGPU executable or resource-probe target" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    const model = try Model.init(alloc);
    defer model.deinit();

    const program = try model.compile(.{ .backend = .webgpu, .context_len = 4, .batch = 1 });
    defer program.deinit();

    const semantic = program.inspect();
    try testing.expectEqual(@as(usize, 4), semantic.context_len);
    try testing.expect(semantic.semantic_runtime_patch_holes > 0);

    const executable = program.inspectExecutable();
    try testing.expectEqual(LlamaBackend.webgpu, executable.backend);
    try testing.expect(executable.external_resources_supported);
    try testing.expectEqual(
        llamaExecutionMode(executable.execution_supported, executable.external_resources_supported),
        executable.execution_mode,
    );
    try testing.expect(executable.command_count > 0);
    try testing.expect(executable.command_projection_count > 0 or executable.command_attention_count > 0);
    try testing.expect(executable.command_stencil_hash != 0);
    try testing.expectEqual(semantic.semantic_runtime_patch_holes, executable.runtime_patch_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_cache_write_pos_holes, executable.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(semantic.semantic_runtime_patch_attention_seq_kv_holes, executable.runtime_patch_attention_seq_kv_holes);
    if (executable.execution_supported) {
        try testing.expect(executable.execution_plan.fullCoverage());
        try testing.expectEqual(@as(u64, @intCast(executable.op_count)), executable.execution_plan.covered_op_count);
        try testing.expect(executable.execution_plan.dispatch_count > 0);
        try testing.expect(executable.execution_plan.family_counts.projectionCount() > 0);
        try testing.expect(executable.execution_plan.family_counts.attention > 0);
    } else {
        try testing.expect(!executable.execution_plan.supported);
        try testing.expectEqual(@as(u64, 0), executable.execution_plan.dispatch_count);
        try testing.expectEqual(@as(u64, 0), executable.execution_plan.covered_op_count);
        try testing.expectEqual(@as(?u64, null), executable.execution_plan.first_unsupported_op);
    }

    if (executable.execution_supported) {
        const cpu_program = try model.compile(.{ .backend = .cpu, .context_len = 4, .batch = 1 });
        defer cpu_program.deinit();
        const cpu_session = try cpu_program.bind(.{});
        defer cpu_session.deinit();
        const expected = try testing.allocator.dupe(f32, try cpu_session.step(.{ .token = 0 }));
        defer testing.allocator.free(expected);

        const session = try program.bind(.{});
        defer session.deinit();
        var logits = [_]f32{-999} ** (cfg.vocab_size + 1);
        const got = try session.stepInto(&logits, .{ .token = 0 });
        try testing.expect(got.ptr == logits[0..cfg.vocab_size].ptr);
        try testing.expectEqual(@as(f32, -999), logits[cfg.vocab_size]);
        try testing.expectEqual(@as(usize, 1), session.position());
        try testing.expectEqual(expected.len, got.len);
        for (got, expected) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        try session.advance(.{ .token = 1 });
        try testing.expectEqual(@as(usize, 2), session.position());
        try session.advance(.{ .token = 2 });
        try session.advance(.{ .token = 3 });
        try testing.expectEqual(@as(usize, 4), session.position());
        try testing.expectError(error.SequenceTooLong, session.advance(.{ .token = 4 }));
        try testing.expectEqual(@as(usize, 4), session.position());

        const cpu_prefill_session = try cpu_program.bind(.{});
        defer cpu_prefill_session.deinit();
        const webgpu_prefill_session = try program.bind(.{});
        defer webgpu_prefill_session.deinit();
        var cpu_prefill_logits = [_]f32{-808} ** (cfg.vocab_size + 1);
        var webgpu_prefill_logits = [_]f32{-909} ** (cfg.vocab_size + 1);
        const cpu_prefill = try cpu_prefill_session.prefillInto(&cpu_prefill_logits, &.{ 0, 1 });
        const webgpu_prefill = try webgpu_prefill_session.prefillInto(&webgpu_prefill_logits, &.{ 0, 1 });
        try testing.expectEqual(@as(usize, 2), cpu_prefill_session.position());
        try testing.expectEqual(@as(usize, 2), webgpu_prefill_session.position());
        try testing.expectEqual(@as(f32, -808), cpu_prefill_logits[cfg.vocab_size]);
        try testing.expectEqual(@as(f32, -909), webgpu_prefill_logits[cfg.vocab_size]);
        for (webgpu_prefill, cpu_prefill) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var cpu_after_prefill_logits = [_]f32{-707} ** (cfg.vocab_size + 1);
        var webgpu_after_prefill_logits = [_]f32{-606} ** (cfg.vocab_size + 1);
        const cpu_after_prefill = try cpu_prefill_session.stepInto(&cpu_after_prefill_logits, .{ .token = 2 });
        const webgpu_after_prefill = try webgpu_prefill_session.stepInto(&webgpu_after_prefill_logits, .{ .token = 2 });
        try testing.expectEqual(@as(usize, 3), cpu_prefill_session.position());
        try testing.expectEqual(@as(usize, 3), webgpu_prefill_session.position());
        try testing.expectEqual(@as(f32, -707), cpu_after_prefill_logits[cfg.vocab_size]);
        try testing.expectEqual(@as(f32, -606), webgpu_after_prefill_logits[cfg.vocab_size]);
        for (webgpu_after_prefill, cpu_after_prefill) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        const cpu_no_output_session = try cpu_program.bind(.{});
        defer cpu_no_output_session.deinit();
        const webgpu_no_output_session = try program.bind(.{});
        defer webgpu_no_output_session.deinit();
        webgpu_no_output_session.resetRuntimeProfile();
        try cpu_no_output_session.advanceTokens(&.{ 0, 1 });
        try webgpu_no_output_session.advanceTokens(&.{ 0, 1 });
        try testing.expectEqual(@as(usize, 2), cpu_no_output_session.position());
        try testing.expectEqual(@as(usize, 2), webgpu_no_output_session.position());
        var cpu_no_output_logits = [_]f32{-505} ** (cfg.vocab_size + 1);
        var webgpu_no_output_logits = [_]f32{-404} ** (cfg.vocab_size + 1);
        const cpu_no_output_after = try cpu_no_output_session.stepInto(&cpu_no_output_logits, .{ .token = 2 });
        const webgpu_no_output_after = try webgpu_no_output_session.stepInto(&webgpu_no_output_logits, .{ .token = 2 });
        try testing.expectEqual(@as(f32, -505), cpu_no_output_logits[cfg.vocab_size]);
        try testing.expectEqual(@as(f32, -404), webgpu_no_output_logits[cfg.vocab_size]);
        for (webgpu_no_output_after, cpu_no_output_after) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var no_output_profile = profile_mod.RuntimeProfile{};
        webgpu_no_output_session.addRuntimeProfileTo(&no_output_profile);
        try testing.expect(no_output_profile.call_count >= 2);
        try testing.expect(no_output_profile.backend_op_count > 0);
        try testing.expectEqual(@as(u64, 0), no_output_profile.fallback_op_count);

        const device_handle = try program.deviceHandle();
        try testing.expect(device_handle != 0);
        const output_byte_len = cfg.vocab_size * @sizeOf(f32);
        const cache_elements = (cfg.d_model / cfg.n_heads) * semantic.context_len * cfg.n_kv_heads;
        const cache_byte_len = cache_elements * @sizeOf(f32);
        const device_output = try program.createDeviceBuffer(output_byte_len, .write_only);
        errdefer program.releaseDeviceBuffer(device_output);
        try testing.expectEqual(backend_mod.Device.webgpu, device_output.placement);
        try testing.expectEqual(@as(u32, @intCast(output_byte_len)), device_output.byte_len);
        const imported_output = try program.importDeviceBuffer(device_handle, device_output.handle, device_output.byte_offset, device_output.byte_len, .write_only);
        defer program.releaseDeviceBuffer(imported_output);
        try testing.expectEqual(device_output.byte_len, imported_output.byte_len);
        try testing.expectError(error.UnsupportedResourceBinding, program.importDeviceBuffer(device_handle + 1, device_output.handle, device_output.byte_offset, device_output.byte_len, .write_only));

        var device_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
        var device_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
        var init_layer_count: usize = 0;
        errdefer for (0..init_layer_count) |l| {
            program.releaseDeviceBuffer(device_k[l]);
            program.releaseDeviceBuffer(device_v[l]);
        };
        for (0..cfg.n_layers) |l| {
            device_k[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
            device_v[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
            init_layer_count += 1;
        }
        defer {
            program.releaseDeviceBuffer(device_output);
            for (0..cfg.n_layers) |l| {
                program.releaseDeviceBuffer(device_k[l]);
                program.releaseDeviceBuffer(device_v[l]);
            }
        }

        var zero_cache = [_]f32{0} ** 16;
        const zero_cache_bytes = std.mem.sliceAsBytes(zero_cache[0..cache_elements]);
        for (0..cfg.n_layers) |l| {
            try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
            try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
        }

        const cpu_resource_session = try cpu_program.bind(.{});
        defer cpu_resource_session.deinit();
        const webgpu_resource_session = try program.bind(.{
            .output_resource = device_output,
            .cache_resources = .{
                .k = device_k[0..],
                .v = device_v[0..],
                .element_count = cache_elements,
            },
        });
        defer webgpu_resource_session.deinit();
        var cpu_resource_logits = [_]f32{-303} ** (cfg.vocab_size + 1);
        const cpu_resource = try cpu_resource_session.stepInto(&cpu_resource_logits, .{ .token = 0 });
        try webgpu_resource_session.stepBoundOutput(.{ .token = 0 });
        try testing.expectEqual(@as(usize, 1), webgpu_resource_session.position());
        var resource_logits = [_]f32{0} ** cfg.vocab_size;
        try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
        for (resource_logits[0..], cpu_resource) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        const resource_inspection = webgpu_resource_session.inspect();
        try testing.expectEqual(@as(u32, 2), resource_inspection.output_storage);
        try testing.expectEqual(@as(u32, 2), resource_inspection.kv_cache_storage);
        try testing.expect(resource_inspection.resource_binding_count >= 3);

        for (0..cfg.n_layers) |l| {
            try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
            try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
        }
        const cpu_resource_prefill_session = try cpu_program.bind(.{});
        defer cpu_resource_prefill_session.deinit();
        const webgpu_resource_prefill_session = try program.bind(.{
            .output_resource = device_output,
            .cache_resources = .{
                .k = device_k[0..],
                .v = device_v[0..],
                .element_count = cache_elements,
            },
        });
        defer webgpu_resource_prefill_session.deinit();
        var cpu_resource_prefill_logits = [_]f32{-202} ** (cfg.vocab_size + 1);
        const cpu_resource_prefill = try cpu_resource_prefill_session.prefillInto(&cpu_resource_prefill_logits, &.{ 0, 1 });
        try webgpu_resource_prefill_session.prefillBoundOutput(&.{ 0, 1 });
        try testing.expectEqual(@as(usize, 2), webgpu_resource_prefill_session.position());
        try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
        for (resource_logits[0..], cpu_resource_prefill) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        var cpu_resource_after_logits = [_]f32{-101} ** (cfg.vocab_size + 1);
        const cpu_resource_after = try cpu_resource_prefill_session.stepInto(&cpu_resource_after_logits, .{ .token = 2 });
        try webgpu_resource_prefill_session.stepBoundOutput(.{ .token = 2 });
        try testing.expectEqual(@as(usize, 3), webgpu_resource_prefill_session.position());
        try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
        for (resource_logits[0..], cpu_resource_after) |actual, want| {
            try testing.expectApproxEqAbs(want, actual, 1e-3);
        }
    }

    const output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 100,
        .byte_len = cfg.vocab_size * @sizeOf(f32),
        .access = .write_only,
    };
    var k_resources = [_]backend_mod.ProgramIO.ExternalResource{.{
        .placement = .webgpu,
        .handle = 101,
        .byte_len = 1024,
        .access = .read_write,
    }};
    var v_resources = [_]backend_mod.ProgramIO.ExternalResource{.{
        .placement = .webgpu,
        .handle = 102,
        .byte_len = 1024,
        .access = .read_write,
    }};
    if (executable.execution_supported) {
        if (program.bind(.{
            .output_resource = output_resource,
            .cache_resources = .{
                .k = k_resources[0..],
                .v = v_resources[0..],
            },
        })) |resource_session| {
            resource_session.deinit();
            return error.TestUnexpectedResult;
        } else |err| switch (err) {
            error.UnsupportedResourceBinding,
            error.WebGPUExecutionUnavailable,
            error.UnsupportedDeviceOp,
            => {},
            else => return err,
        }

        if (program.bindModel(model, .{
            .output_resource = output_resource,
            .cache_resources = .{
                .k = k_resources[0..],
                .v = v_resources[0..],
            },
        })) |model_resource_session| {
            model_resource_session.deinit();
            return error.TestUnexpectedResult;
        } else |err| switch (err) {
            error.UnsupportedResourceBinding,
            error.WebGPUExecutionUnavailable,
            error.UnsupportedDeviceOp,
            => {},
            else => return err,
        }
    } else {
        const resource_session = try program.bind(.{
            .output_resource = output_resource,
            .cache_resources = .{
                .k = k_resources[0..],
                .v = v_resources[0..],
            },
        });
        defer resource_session.deinit();
        const session_inspection = resource_session.inspect();
        try testing.expectEqual(LlamaBackend.webgpu, session_inspection.backend);
        try testing.expectEqual(@as(usize, 0), session_inspection.position);
        try testing.expectEqual(@as(usize, 4), session_inspection.context_len);
        try testing.expectEqual(@as(u32, 2), session_inspection.output_storage);
        try testing.expectEqual(@as(u32, 2), session_inspection.kv_cache_storage);
        try testing.expect(session_inspection.persistent_binding_count > 0);
        try testing.expect(session_inspection.step_input_count > 0);
        try testing.expectEqual(@as(usize, 1), session_inspection.step_output_count);
        try testing.expect(session_inspection.host_binding_count > 0);
        try testing.expectEqual(@as(usize, 3), session_inspection.resource_binding_count);
        try testing.expect(session_inspection.binding_shape_hash != 0);
        try testing.expectError(error.WebGPUExecutionUnavailable, resource_session.step(.{ .token = 0 }));
        try testing.expectEqual(@as(usize, 0), resource_session.position());
        var resource_profile = profile_mod.RuntimeProfile{};
        resource_session.addRuntimeProfileTo(&resource_profile);
        try testing.expectEqual(@as(u64, 0), resource_profile.call_count);
        try testing.expectEqual(@as(u64, 0), resource_profile.backend_op_count);
        try testing.expectEqual(@as(u64, 0), resource_profile.fallback_op_count);
        try testing.expectEqual(@as(u64, 0), resource_profile.sync_count);
        try testing.expectEqual(@as(u64, 0), resource_profile.runtime_patch_call_count);

        const model_resource_session = try program.bindModel(model, .{
            .output_resource = output_resource,
            .cache_resources = .{
                .k = k_resources[0..],
                .v = v_resources[0..],
            },
        });
        defer model_resource_session.deinit();
        const model_session_inspection = model_resource_session.inspect();
        try testing.expectEqual(LlamaBackend.webgpu, model_session_inspection.backend);
        try testing.expectEqual(@as(usize, 0), model_session_inspection.position);
        try testing.expectEqual(@as(usize, 4), model_session_inspection.context_len);
        try testing.expectEqual(@as(u32, 2), model_session_inspection.output_storage);
        try testing.expectEqual(@as(u32, 2), model_session_inspection.kv_cache_storage);
        try testing.expect(model_session_inspection.persistent_binding_count > 0);
        try testing.expect(model_session_inspection.step_input_count > 0);
        try testing.expectEqual(@as(usize, 1), model_session_inspection.step_output_count);
        try testing.expect(model_session_inspection.host_binding_count > 0);
        try testing.expectEqual(@as(usize, 3), model_session_inspection.resource_binding_count);
        try testing.expect(model_session_inspection.binding_shape_hash != 0);
        try testing.expectEqual(session_inspection.binding_shape_hash, model_session_inspection.binding_shape_hash);
        try testing.expectError(error.WebGPUExecutionUnavailable, model_resource_session.step(.{ .token = 0 }));
        try testing.expectEqual(@as(usize, 0), model_resource_session.position());
        var model_resource_profile = profile_mod.RuntimeProfile{};
        model_resource_session.addRuntimeProfileTo(&model_resource_profile);
        try testing.expectEqual(@as(u64, 0), model_resource_profile.call_count);
        try testing.expectEqual(@as(u64, 0), model_resource_profile.backend_op_count);
        try testing.expectEqual(@as(u64, 0), model_resource_profile.fallback_op_count);
        try testing.expectEqual(@as(u64, 0), model_resource_profile.sync_count);
        try testing.expectEqual(@as(u64, 0), model_resource_profile.runtime_patch_call_count);

        failing.fail_index = failing.alloc_index;
        const alloc_index = failing.alloc_index;
        try testing.expectError(error.WebGPUExecutionUnavailable, program.bind(.{}));
        try testing.expectEqual(alloc_index, failing.alloc_index);
    }
}

fn proveLlmFacadeWgpuResourceHandoff(comptime cfg: LlamaConfig, comptime prompt_len: usize) !void {
    return proveLlmFacadeWgpuResourceHandoffWithContext(cfg, cfg.max_seq_len, prompt_len);
}

fn proveLlmFacadeWgpuResourceHandoffWithContext(comptime cfg: LlamaConfig, comptime context_len: usize, comptime prompt_len: usize) !void {
    comptime {
        if (prompt_len < 2) @compileError("public WebGPU resource handoff proof needs at least two prompt tokens");
        if (context_len == 0 or context_len > cfg.max_seq_len) @compileError("public WebGPU resource handoff proof context must fit the model envelope");
        if (prompt_len >= context_len) @compileError("public WebGPU resource handoff proof needs room for a decode token");
    }
    const decode_token: usize = prompt_len % cfg.vocab_size;
    const cache_elements: usize = (cfg.d_model / cfg.n_heads) * context_len * cfg.n_kv_heads;
    const cache_byte_len = cache_elements * @sizeOf(f32);
    const output_byte_len = cfg.vocab_size * @sizeOf(f32);
    const Model = LlamaModel(f32, cfg);

    var prompt: [prompt_len]usize = undefined;
    for (&prompt, 0..) |*token, i| token.* = i % cfg.vocab_size;

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .backend = .webgpu, .context_len = context_len, .batch = 1 });
    defer program.deinit();

    const semantic = program.inspect();
    try testing.expectEqual(@as(usize, cfg.max_seq_len), semantic.max_seq_len);
    try testing.expectEqual(context_len, semantic.context_len);
    try testing.expectEqual(@as(usize, cfg.n_layers), semantic.n_layers);
    try testing.expectEqual(@as(usize, cfg.n_heads), semantic.n_heads);
    try testing.expectEqual(@as(usize, cfg.n_kv_heads), semantic.n_kv_heads);

    const executable = program.inspectExecutable();
    try testing.expectEqual(LlamaBackend.webgpu, executable.backend);
    try testing.expect(executable.external_resources_supported);
    try testing.expectEqual(
        llamaExecutionMode(executable.execution_supported, executable.external_resources_supported),
        executable.execution_mode,
    );
    try testing.expect(executable.command_attention_count > 0);
    if (!executable.execution_supported) {
        try testing.expect(!executable.execution_plan.supported);
        try testing.expectEqual(@as(u64, 0), executable.execution_plan.dispatch_count);
        return;
    }

    try testing.expect(executable.execution_plan.fullCoverage());
    try testing.expectEqual(@as(u64, @intCast(executable.op_count)), executable.execution_plan.covered_op_count);
    try testing.expect(executable.execution_plan.dispatch_count > 0);
    try testing.expect(executable.execution_plan.family_counts.projectionCount() > 0);
    try testing.expect(executable.execution_plan.family_counts.attention > 0);

    const cpu_program = try model.compile(.{ .backend = .cpu, .context_len = context_len, .batch = 1 });
    defer cpu_program.deinit();
    const cpu_session = try cpu_program.bind(.{});
    defer cpu_session.deinit();
    const expected_prefill = try testing.allocator.dupe(f32, try cpu_session.prefill(&prompt));
    defer testing.allocator.free(expected_prefill);
    const expected_after_prefill = try testing.allocator.dupe(f32, try cpu_session.step(.{ .token = decode_token }));
    defer testing.allocator.free(expected_after_prefill);

    const device_handle = try program.deviceHandle();
    try testing.expect(device_handle != 0);
    const device_output = try program.createDeviceBuffer(output_byte_len, .write_only);
    errdefer program.releaseDeviceBuffer(device_output);
    const imported_output = try program.importDeviceBuffer(device_handle, device_output.handle, device_output.byte_offset, device_output.byte_len, .write_only);
    defer program.releaseDeviceBuffer(imported_output);
    try testing.expectEqual(device_output.byte_len, imported_output.byte_len);

    var device_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var device_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var init_layer_count: usize = 0;
    errdefer for (0..init_layer_count) |l| {
        program.releaseDeviceBuffer(device_k[l]);
        program.releaseDeviceBuffer(device_v[l]);
    };
    for (0..cfg.n_layers) |l| {
        device_k[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        device_v[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        init_layer_count += 1;
    }
    defer {
        program.releaseDeviceBuffer(device_output);
        for (0..cfg.n_layers) |l| {
            program.releaseDeviceBuffer(device_k[l]);
            program.releaseDeviceBuffer(device_v[l]);
        }
    }

    var zero_cache = [_]f32{0} ** cache_elements;
    const zero_cache_bytes = std.mem.sliceAsBytes(zero_cache[0..]);
    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }

    const resource_session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer resource_session.deinit();

    try resource_session.prefillBoundOutput(&prompt);
    try testing.expectEqual(prompt_len, resource_session.position());
    var resource_logits = [_]f32{0} ** cfg.vocab_size;
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    for (resource_logits[0..], expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try resource_session.stepBoundOutput(.{ .token = decode_token });
    try testing.expectEqual(prompt_len + 1, resource_session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    for (resource_logits[0..], expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var resource_profile = profile_mod.RuntimeProfile{};
    resource_session.addRuntimeProfileTo(&resource_profile);
    try testing.expect(resource_profile.call_count >= 2);
    try testing.expect(resource_profile.backend_op_count > 0);
    try testing.expect(resource_profile.backend_dispatch_count >= executable.execution_plan.dispatch_count);
    try testing.expectEqual(@as(u64, 0), resource_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), resource_profile.sync_count);

    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }
    var output_sentinel = [_]f32{-4141} ** cfg.vocab_size;
    try program.writeDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(output_sentinel[0..]));

    const no_output_session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer no_output_session.deinit();

    try no_output_session.advanceTokens(&prompt);
    try testing.expectEqual(prompt_len, no_output_session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    try testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);

    try no_output_session.stepBoundOutput(.{ .token = decode_token });
    try testing.expectEqual(prompt_len + 1, no_output_session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    for (resource_logits[0..], expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var no_output_profile = profile_mod.RuntimeProfile{};
    no_output_session.addRuntimeProfileTo(&no_output_profile);
    try testing.expect(no_output_profile.call_count >= 2);
    try testing.expect(no_output_profile.backend_op_count > 0);
    try testing.expectEqual(@as(u64, 0), no_output_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), no_output_profile.sync_count);

    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }
    const cpu_generate_session = try cpu_program.bind(.{});
    defer cpu_generate_session.deinit();
    const expected_generated_0 = try sampling.argmax(f32, try cpu_generate_session.prefill(&prompt));
    const expected_generated_1 = try sampling.argmax(f32, try cpu_generate_session.step(.{ .token = expected_generated_0.token }));

    const generated_session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer generated_session.deinit();
    var generated_tokens = [_]usize{ 99, 99 };
    const generated = try generated_session.generateArgmaxInto(&generated_tokens, &prompt);
    try testing.expectEqual(@as(usize, generated_tokens.len), generated.tokens_generated);
    try testing.expectEqual(expected_generated_0.token, generated_tokens[0]);
    try testing.expectEqual(expected_generated_1.token, generated_tokens[1]);
    try testing.expectEqual(expected_generated_1.token, generated.last_token);
    try testing.expectApproxEqAbs(expected_generated_1.logit, generated.last_logit, 1e-3);
    try testing.expectEqual(prompt_len + 1, generated_session.position());

    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }
    const sampled_session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer sampled_session.deinit();
    var sampled_tokens = [_]usize{ 77, 77 };
    const sampled = try sampled_session.generateSampleInto(&sampled_tokens, &prompt, .{ .top_k = 1, .seed = 123, .temperature = 1 });
    try testing.expectEqual(@as(usize, sampled_tokens.len), sampled.tokens_generated);
    try testing.expectEqual(expected_generated_0.token, sampled_tokens[0]);
    try testing.expectEqual(expected_generated_1.token, sampled_tokens[1]);
    try testing.expectEqual(expected_generated_1.token, sampled.last_token);
    try testing.expectApproxEqAbs(expected_generated_1.logit, sampled.last_logit, 1e-3);
    try testing.expectEqual(prompt_len + 1, sampled_session.position());
}

fn expectProfileNoAdditionalWork(before: profile_mod.RuntimeProfile, after: profile_mod.RuntimeProfile) !void {
    try testing.expectEqual(before.call_count, after.call_count);
    try testing.expectEqual(before.backend_op_count, after.backend_op_count);
    try testing.expectEqual(before.fallback_op_count, after.fallback_op_count);
    try testing.expectEqual(before.backend_dispatch_count, after.backend_dispatch_count);
    try testing.expectEqual(before.sync_count, after.sync_count);
    try testing.expectEqual(before.runtime_patch_call_count, after.runtime_patch_call_count);
    try testing.expectEqual(before.runtime_patch_changed_count, after.runtime_patch_changed_count);
    try testing.expectEqual(before.runtime_patch_invalid_count, after.runtime_patch_invalid_count);
    try testing.expectEqual(before.region_command_plan_cached_count, after.region_command_plan_cached_count);
    try testing.expectEqual(before.region_command_plan_cached_command_count, after.region_command_plan_cached_command_count);
    try testing.expectEqual(before.region_command_plan_dynamic_count, after.region_command_plan_dynamic_count);
    try testing.expectEqualSlices(u64, before.program_command_counts[0..], after.program_command_counts[0..]);
    try testing.expectEqualSlices(u64, before.program_command_dispatch_counts[0..], after.program_command_dispatch_counts[0..]);
    try testing.expectEqualSlices(u64, before.program_command_attempt_counts[0..], after.program_command_attempt_counts[0..]);
    try testing.expectEqualSlices(u64, before.program_command_failed_counts[0..], after.program_command_failed_counts[0..]);
    try testing.expectEqualSlices(u64, before.program_op_command_counts[0..], after.program_op_command_counts[0..]);
}

fn proveLlmFacadeWgpuResourceOverContextRejects(comptime cfg: LlamaConfig, comptime context_len: usize) !void {
    comptime {
        if (context_len == 0 or context_len > cfg.max_seq_len) @compileError("public WebGPU over-context proof context must fit the model envelope");
    }
    const cache_elements: usize = (cfg.d_model / cfg.n_heads) * context_len * cfg.n_kv_heads;
    const cache_byte_len = cache_elements * @sizeOf(f32);
    const output_byte_len = cfg.vocab_size * @sizeOf(f32);
    const Model = LlamaModel(f32, cfg);

    var tokens: [context_len]usize = undefined;
    for (&tokens, 0..) |*token, i| token.* = i % cfg.vocab_size;

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .backend = .webgpu, .context_len = context_len, .batch = 1 });
    defer program.deinit();

    const semantic = program.inspect();
    try testing.expectEqual(@as(usize, cfg.max_seq_len), semantic.max_seq_len);
    try testing.expectEqual(context_len, semantic.context_len);

    const executable = program.inspectExecutable();
    try testing.expectEqual(LlamaBackend.webgpu, executable.backend);
    try testing.expect(executable.external_resources_supported);
    try testing.expectEqual(
        llamaExecutionMode(executable.execution_supported, executable.external_resources_supported),
        executable.execution_mode,
    );
    if (!executable.execution_supported) return;

    const device_output = try program.createDeviceBuffer(output_byte_len, .write_only);
    errdefer program.releaseDeviceBuffer(device_output);

    var device_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var device_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var init_layer_count: usize = 0;
    errdefer for (0..init_layer_count) |l| {
        program.releaseDeviceBuffer(device_k[l]);
        program.releaseDeviceBuffer(device_v[l]);
    };
    for (0..cfg.n_layers) |l| {
        device_k[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        device_v[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        init_layer_count += 1;
    }
    defer {
        program.releaseDeviceBuffer(device_output);
        for (0..cfg.n_layers) |l| {
            program.releaseDeviceBuffer(device_k[l]);
            program.releaseDeviceBuffer(device_v[l]);
        }
    }

    var zero_cache = [_]f32{0} ** cache_elements;
    const zero_cache_bytes = std.mem.sliceAsBytes(zero_cache[0..]);
    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }

    var output_sentinel = [_]f32{-8181} ** cfg.vocab_size;
    try program.writeDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(output_sentinel[0..]));

    const session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer session.deinit();

    try session.advanceTokens(tokens[0..]);
    try testing.expectEqual(context_len, session.position());

    var resource_logits = [_]f32{0} ** cfg.vocab_size;
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    try testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);

    var before_reject = profile_mod.RuntimeProfile{};
    session.addRuntimeProfileTo(&before_reject);
    try testing.expect(before_reject.call_count > 0);
    try testing.expect(before_reject.backend_op_count > 0);
    try testing.expectEqual(@as(u64, 0), before_reject.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), before_reject.sync_count);

    try testing.expectError(error.SequenceTooLong, session.stepBoundOutput(.{ .token = 0 }));
    try testing.expectEqual(context_len, session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    try testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);

    var after_reject = profile_mod.RuntimeProfile{};
    session.addRuntimeProfileTo(&after_reject);
    try expectProfileNoAdditionalWork(before_reject, after_reject);

    var argmax_tokens = [_]usize{ 1234, 5678 };
    try testing.expectError(error.SequenceTooLong, session.generateArgmaxInto(&argmax_tokens, &.{0}));
    try testing.expectEqualSlices(usize, &[_]usize{ 1234, 5678 }, argmax_tokens[0..]);
    try testing.expectEqual(context_len, session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    try testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);

    var after_argmax_reject = profile_mod.RuntimeProfile{};
    session.addRuntimeProfileTo(&after_argmax_reject);
    try expectProfileNoAdditionalWork(before_reject, after_argmax_reject);

    var sampled_tokens = [_]usize{ 8765, 4321 };
    try testing.expectError(error.SequenceTooLong, session.generateSampleInto(&sampled_tokens, &.{0}, .{ .top_k = 1, .seed = 123, .temperature = 1 }));
    try testing.expectEqualSlices(usize, &[_]usize{ 8765, 4321 }, sampled_tokens[0..]);
    try testing.expectEqual(context_len, session.position());
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    try testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);

    var after_sample_reject = profile_mod.RuntimeProfile{};
    session.addRuntimeProfileTo(&after_sample_reject);
    try expectProfileNoAdditionalWork(before_reject, after_sample_reject);
}

test "llm facade WebGPU resource handoff covers GQA multilayer shape" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade WebGPU resource handoff covers MQA multilayer shape" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 1,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade WebGPU resource handoff covers MHA multilayer shape" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 3,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade WebGPU resource handoff covers tied LM head shape" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
        .tied_lm_head = true,
    }, 8);
}

test "llm facade WebGPU resource handoff covers realistic head width" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 128,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 256,
        .n_layers = 1,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade WebGPU resource handoff covers realistic GQA multilayer shape" {
    try proveLlmFacadeWgpuResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 512,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 1024,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade WebGPU resource handoff covers quantized projections" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();
    try model.state().inner.quantize();
    try testing.expect(model.state().inner.plan.quant_weights.len > 0);

    const program = try model.compile(.{ .backend = .webgpu, .context_len = 4, .batch = 1 });
    defer program.deinit();

    const executable = program.inspectExecutable();
    try testing.expectEqual(LlamaBackend.webgpu, executable.backend);
    try testing.expect(executable.external_resources_supported);
    try testing.expectEqual(
        llamaExecutionMode(executable.execution_supported, executable.external_resources_supported),
        executable.execution_mode,
    );
    if (!executable.execution_supported) {
        try testing.expect(!executable.execution_plan.supported);
        try testing.expectEqual(@as(u64, 0), executable.execution_plan.dispatch_count);
        return;
    }

    try testing.expect(executable.qweight_count > 0);
    try testing.expect(executable.execution_plan.fullCoverage());
    try testing.expectEqual(@as(u64, @intCast(executable.op_count)), executable.execution_plan.covered_op_count);
    try testing.expect(executable.execution_plan.family_counts.projectionCount() > 0);
    try testing.expect(executable.execution_plan.family_counts.attention > 0);
    try testing.expect(executable.execution_plan.family_counts.quantizedProjectionCount() > 0);

    const cpu_program = try model.compile(.{ .backend = .cpu, .context_len = 4, .batch = 1 });
    defer cpu_program.deinit();
    const cpu_session = try cpu_program.bind(.{});
    defer cpu_session.deinit();
    const expected = try testing.allocator.dupe(f32, try cpu_session.step(.{ .token = 0 }));
    defer testing.allocator.free(expected);

    const output_byte_len = cfg.vocab_size * @sizeOf(f32);
    const cache_elements = (cfg.d_model / cfg.n_heads) * cfg.max_seq_len * cfg.n_kv_heads;
    const cache_byte_len = cache_elements * @sizeOf(f32);

    const device_output = try program.createDeviceBuffer(output_byte_len, .write_only);
    errdefer program.releaseDeviceBuffer(device_output);
    var device_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var device_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var init_layer_count: usize = 0;
    errdefer for (0..init_layer_count) |l| {
        program.releaseDeviceBuffer(device_k[l]);
        program.releaseDeviceBuffer(device_v[l]);
    };
    for (0..cfg.n_layers) |l| {
        device_k[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        device_v[l] = try program.createDeviceBuffer(cache_byte_len, .read_write);
        init_layer_count += 1;
    }
    defer {
        program.releaseDeviceBuffer(device_output);
        for (0..cfg.n_layers) |l| {
            program.releaseDeviceBuffer(device_k[l]);
            program.releaseDeviceBuffer(device_v[l]);
        }
    }

    var zero_cache = [_]f32{0} ** cache_elements;
    const zero_cache_bytes = std.mem.sliceAsBytes(zero_cache[0..]);
    for (0..cfg.n_layers) |l| {
        try program.writeDeviceBuffer(device_k[l], 0, zero_cache_bytes);
        try program.writeDeviceBuffer(device_v[l], 0, zero_cache_bytes);
    }

    const resource_session = try program.bind(.{
        .output_resource = device_output,
        .cache_resources = .{
            .k = device_k[0..],
            .v = device_v[0..],
            .element_count = cache_elements,
        },
    });
    defer resource_session.deinit();
    try resource_session.stepBoundOutput(.{ .token = 0 });
    try testing.expectEqual(@as(usize, 1), resource_session.position());

    var resource_logits = [_]f32{0} ** cfg.vocab_size;
    try program.readDeviceBuffer(device_output, 0, std.mem.sliceAsBytes(resource_logits[0..]));
    const qweight_tolerance: f32 = 5e-2;
    for (resource_logits[0..], expected) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, qweight_tolerance);
    }

    var profile = profile_mod.RuntimeProfile{};
    resource_session.addRuntimeProfileTo(&profile);
    try testing.expectEqual(@as(u64, 1), profile.call_count);
    try testing.expect(profile.backend_op_count > 0);
    try testing.expectEqual(executable.execution_plan.dispatch_count, profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), profile.sync_count);
}

test "llm facade WebGPU resource handoff uses compiled context envelope" {
    try proveLlmFacadeWgpuResourceHandoffWithContext(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8, 4);
}

test "llm facade WebGPU resource handoff rejects over-context without GPU work" {
    try proveLlmFacadeWgpuResourceOverContextRejects(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

fn wgpuF32ByteLen(element_count: usize) !usize {
    return std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
}

fn createWgpuF32Resource(
    handle: backend_mod.Backend.CompiledHandle,
    element_count: usize,
    access: backend_mod.ProgramIO.ExternalResource.Access,
) !backend_mod.ProgramIO.ExternalResource {
    return wgpu_mod.createDeviceBuffer(handle, try wgpuF32ByteLen(element_count), access);
}

fn zeroWgpuResource(
    alloc: std.mem.Allocator,
    handle: backend_mod.Backend.CompiledHandle,
    resource: backend_mod.ProgramIO.ExternalResource,
) !void {
    const zeros = try alloc.alloc(u8, @intCast(resource.byte_len));
    defer alloc.free(zeros);
    @memset(zeros, 0);
    try wgpu_mod.writeDeviceBuffer(handle, resource, 0, zeros);
}

fn writeWgpuF32Resource(
    handle: backend_mod.Backend.CompiledHandle,
    resource: backend_mod.ProgramIO.ExternalResource,
    values: []const f32,
) !void {
    try wgpu_mod.writeDeviceBuffer(handle, resource, 0, std.mem.sliceAsBytes(values));
}

fn readWgpuF32Resource(
    handle: backend_mod.Backend.CompiledHandle,
    resource: backend_mod.ProgramIO.ExternalResource,
    values: []f32,
) !void {
    try wgpu_mod.readDeviceBuffer(handle, resource, 0, std.mem.sliceAsBytes(values));
}

fn importWgpuResourceView(
    handle: backend_mod.Backend.CompiledHandle,
    device_handle: usize,
    resource: backend_mod.ProgramIO.ExternalResource,
    access: backend_mod.ProgramIO.ExternalResource.Access,
) !backend_mod.ProgramIO.ExternalResource {
    return wgpu_mod.importDeviceBuffer(handle, device_handle, resource.handle, resource.byte_offset, resource.byte_len, access);
}

fn zeroQuantizedWeightPayloads(comptime T: type, qweights: []@import("quant.zig").QuantizedWeight(T)) void {
    for (qweights) |*qw| {
        @memset(@constCast(qw.data), 0);
        if (qw.t_data) |data| @memset(@constCast(data), 0);
    }
}

fn expectFullNativeWgpuExecutionPlan(plan: anytype) !void {
    try testing.expect(plan.fullCoverage());
    try testing.expectEqual(plan.total_op_count, plan.covered_op_count);
    try testing.expect(plan.dispatch_count > 0);
    try testing.expectEqual(@as(?u64, null), plan.first_unsupported_op);
}

fn expectFullNativeWgpuDispatchPlan(handle: backend_mod.Backend.CompiledHandle) !wgpu_mod.DispatchPlanInspection {
    const plan = wgpu_mod.inspectCompiledDispatchPlan(handle);
    try expectFullNativeWgpuExecutionPlan(plan);
    return plan;
}

fn expectNativeWgpuLlamaDispatchFamilies(plan: anytype) !void {
    try testing.expect(plan.family_counts.projectionCount() > 0);
    try testing.expect(plan.family_counts.attention > 0);
}

test "native WebGPU tiny llama decode program matches CPU internally" {
    if (!opts.use_wgpu) return error.SkipZigTest;

    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);

    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_expected.deinit();
    const expected_first = try testing.allocator.dupe(f32, try cpu_expected.step(0));
    defer testing.allocator.free(expected_first);
    const expected_second = try testing.allocator.dupe(f32, try cpu_expected.step(1));
    defer testing.allocator.free(expected_second);

    var cpu_prefill_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_prefill_expected.deinit();
    const expected_prefill = try testing.allocator.dupe(f32, try cpu_prefill_expected.prefill(&.{ 0, 1 }));
    defer testing.allocator.free(expected_prefill);
    const expected_after_prefill = try testing.allocator.dupe(f32, try cpu_prefill_expected.step(2));
    defer testing.allocator.free(expected_after_prefill);

    var wgpu = wgpu_mod.WgpuBackend.init(testing.allocator);
    defer wgpu.deinit();
    var program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), testing.allocator);
    defer program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(wgpu.backend(), testing.allocator, 2);
    defer prefill_program.deinit();

    const executable = program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, executable.backend);
    try testing.expect(executable.execution_supported);
    try testing.expect(executable.command_shape.command_count > 0);
    try testing.expect(executable.command_shape.command_stencil_hash != 0);
    try expectFullNativeWgpuExecutionPlan(executable.execution_plan);
    const decode_dispatch_plan = try expectFullNativeWgpuDispatchPlan(program.program.handle);
    try testing.expectEqual(decode_dispatch_plan.dispatch_count, executable.execution_plan.dispatch_count);
    try expectNativeWgpuLlamaDispatchFamilies(decode_dispatch_plan);

    const prefill_executable = prefill_program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, prefill_executable.backend);
    try testing.expect(prefill_executable.execution_supported);
    try expectFullNativeWgpuExecutionPlan(prefill_executable.execution_plan);
    const prefill_dispatch_plan = try expectFullNativeWgpuDispatchPlan(prefill_program.program.handle);
    try testing.expectEqual(prefill_dispatch_plan.dispatch_count, prefill_executable.execution_plan.dispatch_count);
    try expectNativeWgpuLlamaDispatchFamilies(prefill_dispatch_plan);

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();
    var logits = [_]f32{-999} ** (cfg.vocab_size + 1);
    var exec = try program.bind(&runtime, logits[0..cfg.vocab_size]);
    defer exec.deinit();

    const first = try exec.stepInto(&logits, 0);
    try testing.expect(first.ptr == logits[0..cfg.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -999), logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 1), runtime.position());
    try testing.expectEqual(expected_first.len, first.len);
    for (first, expected_first) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try exec.advance(1);
    try testing.expectEqual(@as(usize, 2), runtime.position());

    var next_runtime = try model_session.bindRuntime(testing.allocator);
    defer next_runtime.deinit();
    var next_logits = [_]f32{-777} ** (cfg.vocab_size + 1);
    var next_exec = try program.bind(&next_runtime, next_logits[0..cfg.vocab_size]);
    defer next_exec.deinit();
    try next_exec.advance(0);
    const second = try next_exec.stepInto(&next_logits, 1);
    try testing.expectEqual(@as(f32, -777), next_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), next_runtime.position());
    try testing.expectEqual(expected_second.len, second.len);
    for (second, expected_second) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var handoff_runtime = try model_session.bindRuntime(testing.allocator);
    defer handoff_runtime.deinit();
    var handoff_decode_logits = [_]f32{-111} ** (cfg.vocab_size + 1);
    var handoff_decode = try program.bind(&handoff_runtime, handoff_decode_logits[0..cfg.vocab_size]);
    defer handoff_decode.deinit();
    var prefill_logits = [_]f32{-222} ** (cfg.vocab_size + 1);
    var prefill_exec = try prefill_program.bind(&handoff_runtime, prefill_logits[0..cfg.vocab_size]);
    defer prefill_exec.deinit();

    const got_prefill = try prefill_exec.prefillInto(&prefill_logits, &.{ 0, 1 });
    try testing.expectEqual(@as(f32, -222), prefill_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), handoff_runtime.position());
    try testing.expectEqual(expected_prefill.len, got_prefill.len);
    for (got_prefill, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try handoff_decode.refreshCacheBindings();
    const got_after_prefill = try handoff_decode.stepInto(&handoff_decode_logits, 2);
    try testing.expectEqual(@as(f32, -111), handoff_decode_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 3), handoff_runtime.position());
    try testing.expectEqual(expected_after_prefill.len, got_after_prefill.len);
    for (got_after_prefill, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var no_output_runtime = try model_session.bindRuntime(testing.allocator);
    defer no_output_runtime.deinit();
    var no_output_decode_logits = [_]f32{-333} ** (cfg.vocab_size + 1);
    var no_output_decode = try program.bind(&no_output_runtime, no_output_decode_logits[0..cfg.vocab_size]);
    defer no_output_decode.deinit();
    var no_output_prefill_logits = [_]f32{-444} ** (cfg.vocab_size + 1);
    var no_output_prefill = try prefill_program.bind(&no_output_runtime, no_output_prefill_logits[0..cfg.vocab_size]);
    defer no_output_prefill.deinit();

    try no_output_prefill.advanceTokens(&.{ 0, 1 });
    try testing.expectEqual(@as(f32, -444), no_output_prefill_logits[0]);
    try testing.expectEqual(@as(f32, -444), no_output_prefill_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), no_output_runtime.position());

    try no_output_decode.refreshCacheBindings();
    const got_after_no_output_prefill = try no_output_decode.stepInto(&no_output_decode_logits, 2);
    try testing.expectEqual(@as(f32, -333), no_output_decode_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 3), no_output_runtime.position());
    try testing.expectEqual(expected_after_prefill.len, got_after_no_output_prefill.len);
    for (got_after_no_output_prefill, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }
}

test "native WebGPU tiny llama quantized decode uses runtime qweights internally" {
    if (!opts.use_wgpu) return error.SkipZigTest;

    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);

    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();
    try model_session.quantize();
    try testing.expect(model_session.plan.quant_weights.len > 0);

    var cpu_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_expected.deinit();
    const expected = try testing.allocator.dupe(f32, try cpu_expected.step(0));
    defer testing.allocator.free(expected);

    var modified_cpu = try model_session.bindRuntime(testing.allocator);
    defer modified_cpu.deinit();
    zeroQuantizedWeightPayloads(f32, modified_cpu.plan.quant_weights);
    const expected_rebound = try testing.allocator.dupe(f32, try modified_cpu.step(0));
    defer testing.allocator.free(expected_rebound);
    try testing.expect(!std.mem.eql(f32, expected, expected_rebound));

    var wgpu = wgpu_mod.WgpuBackend.init(testing.allocator);
    defer wgpu.deinit();
    var program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), testing.allocator);
    defer program.deinit();

    const executable = program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, executable.backend);
    try testing.expect(executable.execution_supported);
    try testing.expect(executable.qweight_count > 0);
    try testing.expect(executable.command_shape.categoryCounts().projection > 0);
    try expectFullNativeWgpuExecutionPlan(executable.execution_plan);
    const dispatch_plan = try expectFullNativeWgpuDispatchPlan(program.program.handle);
    try testing.expectEqual(dispatch_plan.dispatch_count, executable.execution_plan.dispatch_count);
    try expectNativeWgpuLlamaDispatchFamilies(dispatch_plan);
    try testing.expect(dispatch_plan.family_counts.quantizedProjectionCount() > 0);

    const qweight_tolerance: f32 = 5e-2;

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();
    var logits = [_]f32{-8181} ** (cfg.vocab_size + 1);
    var exec = try program.bind(&runtime, logits[0..cfg.vocab_size]);
    defer exec.deinit();
    const got = try exec.stepInto(&logits, 0);
    try testing.expectEqual(@as(f32, -8181), logits[cfg.vocab_size]);
    for (got, expected) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, qweight_tolerance);
    }
    var profile = profile_mod.RuntimeProfile{};
    exec.addRuntimeProfileTo(&profile);
    try testing.expectEqual(@as(u64, 1), profile.call_count);
    try testing.expect(profile.backend_op_count > 0);
    try testing.expectEqual(dispatch_plan.dispatch_count, profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), profile.fallback_op_count);

    const cache_elements = try Session.DeviceDecodeProgram.cacheElementCountForContext(cfg.max_seq_len);
    const handle = program.program.handle;
    const resource_output = try createWgpuF32Resource(handle, cfg.vocab_size, .write_only);
    var resource_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var resource_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        resource_k[l] = try createWgpuF32Resource(handle, cache_elements, .read_write);
        resource_v[l] = try createWgpuF32Resource(handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, handle, resource_k[l]);
        try zeroWgpuResource(testing.allocator, handle, resource_v[l]);
    }

    var resource_runtime = try model_session.bindRuntime(testing.allocator);
    defer resource_runtime.deinit();
    var resource_scratch = [_]f32{-2727} ** cfg.vocab_size;
    var resource_exec = try program.bindWithOptions(&resource_runtime, .{
        .logits_buf = resource_scratch[0..],
        .output_resource = resource_output,
        .cache_bindings = .{ .resource = .{
            .k = resource_k,
            .v = resource_v,
            .element_count = cache_elements,
        } },
    });
    defer resource_exec.deinit();
    try resource_exec.stepBoundOutput(0);
    try testing.expectEqual(@as(f32, -2727), resource_scratch[0]);
    var resource_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(handle, resource_output, resource_logits[0..]);
    for (resource_logits, expected) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, qweight_tolerance);
    }
    var resource_profile = profile_mod.RuntimeProfile{};
    resource_exec.addRuntimeProfileTo(&resource_profile);
    try testing.expectEqual(@as(u64, 1), resource_profile.call_count);
    try testing.expect(resource_profile.backend_op_count > 0);
    try testing.expectEqual(dispatch_plan.dispatch_count, resource_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), resource_profile.fallback_op_count);

    var rebound_runtime = try model_session.bindRuntime(testing.allocator);
    defer rebound_runtime.deinit();
    zeroQuantizedWeightPayloads(f32, rebound_runtime.plan.quant_weights);
    var rebound_logits = [_]f32{-9191} ** (cfg.vocab_size + 1);
    var rebound_exec = try program.bind(&rebound_runtime, rebound_logits[0..cfg.vocab_size]);
    defer rebound_exec.deinit();
    const rebound = try rebound_exec.stepInto(&rebound_logits, 0);
    try testing.expectEqual(@as(f32, -9191), rebound_logits[cfg.vocab_size]);
    for (rebound, expected_rebound) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, qweight_tolerance);
    }

    const rebound_resource_output = try createWgpuF32Resource(handle, cfg.vocab_size, .write_only);
    var rebound_resource_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var rebound_resource_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        rebound_resource_k[l] = try createWgpuF32Resource(handle, cache_elements, .read_write);
        rebound_resource_v[l] = try createWgpuF32Resource(handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, handle, rebound_resource_k[l]);
        try zeroWgpuResource(testing.allocator, handle, rebound_resource_v[l]);
    }

    var rebound_resource_runtime = try model_session.bindRuntime(testing.allocator);
    defer rebound_resource_runtime.deinit();
    zeroQuantizedWeightPayloads(f32, rebound_resource_runtime.plan.quant_weights);
    var rebound_resource_scratch = [_]f32{-3737} ** cfg.vocab_size;
    var rebound_resource_exec = try program.bindWithOptions(&rebound_resource_runtime, .{
        .logits_buf = rebound_resource_scratch[0..],
        .output_resource = rebound_resource_output,
        .cache_bindings = .{ .resource = .{
            .k = rebound_resource_k,
            .v = rebound_resource_v,
            .element_count = cache_elements,
        } },
    });
    defer rebound_resource_exec.deinit();
    try rebound_resource_exec.stepBoundOutput(0);
    try testing.expectEqual(@as(f32, -3737), rebound_resource_scratch[0]);
    var rebound_resource_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(handle, rebound_resource_output, rebound_resource_logits[0..]);
    for (rebound_resource_logits, expected_rebound) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, qweight_tolerance);
    }
    var rebound_resource_profile = profile_mod.RuntimeProfile{};
    rebound_resource_exec.addRuntimeProfileTo(&rebound_resource_profile);
    try testing.expectEqual(@as(u64, 1), rebound_resource_profile.call_count);
    try testing.expect(rebound_resource_profile.backend_op_count > 0);
    try testing.expectEqual(dispatch_plan.dispatch_count, rebound_resource_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), rebound_resource_profile.fallback_op_count);
}

test "native WebGPU tiny llama resource bindings execute internally" {
    if (!opts.use_wgpu) return error.SkipZigTest;

    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);

    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_expected.deinit();
    const expected_first = try testing.allocator.dupe(f32, try cpu_expected.step(0));
    defer testing.allocator.free(expected_first);
    const expected_second = try testing.allocator.dupe(f32, try cpu_expected.step(1));
    defer testing.allocator.free(expected_second);

    var cpu_prefill_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_prefill_expected.deinit();
    const expected_prefill = try testing.allocator.dupe(f32, try cpu_prefill_expected.prefill(&.{ 0, 1 }));
    defer testing.allocator.free(expected_prefill);
    const expected_after_prefill = try testing.allocator.dupe(f32, try cpu_prefill_expected.step(2));
    defer testing.allocator.free(expected_after_prefill);

    var wgpu = wgpu_mod.WgpuBackend.init(testing.allocator);
    defer wgpu.deinit();
    var decode_program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), testing.allocator);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(wgpu.backend(), testing.allocator, 2);
    defer prefill_program.deinit();

    const cache_elements = try Session.DeviceDecodeProgram.cacheElementCountForContext(cfg.max_seq_len);

    const decode_handle = decode_program.program.handle;
    const decode_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    var decode_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var decode_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        decode_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        decode_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_k[l]);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_v[l]);
    }
    var decode_output_sentinel = [_]f32{-5151} ** cfg.vocab_size;
    try writeWgpuF32Resource(decode_handle, decode_output, decode_output_sentinel[0..]);

    var output_only_runtime = try model_session.bindRuntime(testing.allocator);
    defer output_only_runtime.deinit();
    var output_only_scratch = [_]f32{-6161} ** cfg.vocab_size;
    var output_only_exec = try decode_program.bindWithOptions(&output_only_runtime, .{
        .logits_buf = output_only_scratch[0..],
        .output_resource = decode_output,
    });
    defer output_only_exec.deinit();

    var output_only_logits = [_]f32{0} ** cfg.vocab_size;
    try output_only_exec.stepBoundOutput(0);
    try testing.expectEqual(@as(usize, 1), output_only_runtime.position());
    try testing.expectEqual(@as(f32, -6161), output_only_scratch[0]);
    try readWgpuF32Resource(decode_handle, decode_output, output_only_logits[0..]);
    for (output_only_logits, expected_first) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var unreadable_cache_runtime = try model_session.bindRuntime(testing.allocator);
    defer unreadable_cache_runtime.deinit();
    var bad_decode_k = decode_k;
    bad_decode_k[0].access = .write_only;
    var bad_decode_scratch = [_]f32{-3131} ** cfg.vocab_size;
    try testing.expectError(error.UnsupportedResourceBinding, decode_program.bindWithOptions(&unreadable_cache_runtime, .{
        .logits_buf = bad_decode_scratch[0..],
        .output_resource = decode_output,
        .cache_bindings = .{ .resource = .{
            .k = bad_decode_k,
            .v = decode_v,
            .element_count = cache_elements,
        } },
    }));
    try testing.expectEqual(@as(usize, 0), unreadable_cache_runtime.position());

    var resource_runtime = try model_session.bindRuntime(testing.allocator);
    defer resource_runtime.deinit();
    var host_scratch = [_]f32{-7171} ** cfg.vocab_size;
    var resource_exec = try decode_program.bindWithOptions(&resource_runtime, .{
        .logits_buf = host_scratch[0..],
        .output_resource = decode_output,
        .cache_bindings = .{ .resource = .{
            .k = decode_k,
            .v = decode_v,
            .element_count = cache_elements,
        } },
    });
    defer resource_exec.deinit();

    var resource_logits = [_]f32{0} ** cfg.vocab_size;
    try resource_exec.stepBoundOutput(0);
    try testing.expectEqual(@as(usize, 1), resource_runtime.position());
    try testing.expectEqual(@as(f32, -7171), host_scratch[0]);
    try readWgpuF32Resource(decode_handle, decode_output, resource_logits[0..]);
    for (resource_logits, expected_first) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    @memset(resource_logits[0..], 0);
    try resource_exec.stepBoundOutput(1);
    try testing.expectEqual(@as(usize, 2), resource_runtime.position());
    try testing.expectEqual(@as(f32, -7171), host_scratch[0]);
    try readWgpuF32Resource(decode_handle, decode_output, resource_logits[0..]);
    for (resource_logits, expected_second) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    const prefill_handle = prefill_program.program.handle;
    const prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        prefill_k[l] = try createWgpuF32Resource(prefill_handle, cache_elements, .read_write);
        prefill_v[l] = try createWgpuF32Resource(prefill_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, prefill_handle, prefill_k[l]);
        try zeroWgpuResource(testing.allocator, prefill_handle, prefill_v[l]);
    }
    var prefill_output_sentinel = [_]f32{-6262} ** cfg.vocab_size;
    try writeWgpuF32Resource(prefill_handle, prefill_output, prefill_output_sentinel[0..]);

    var unwritable_cache_runtime = try model_session.bindRuntime(testing.allocator);
    defer unwritable_cache_runtime.deinit();
    var bad_prefill_k = prefill_k;
    bad_prefill_k[0].access = .read_only;
    var bad_prefill_scratch = [_]f32{-4141} ** cfg.vocab_size;
    try testing.expectError(error.UnsupportedResourceBinding, prefill_program.bindWithOptions(&unwritable_cache_runtime, .{
        .logits_buf = bad_prefill_scratch[0..],
        .output_resource = prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = bad_prefill_k,
            .v = prefill_v,
            .element_count = cache_elements,
        } },
    }));
    try testing.expectEqual(@as(usize, 0), unwritable_cache_runtime.position());

    var prefill_runtime = try model_session.bindRuntime(testing.allocator);
    defer prefill_runtime.deinit();
    var prefill_host_scratch = [_]f32{-8181} ** cfg.vocab_size;
    var resource_prefill = try prefill_program.bindWithOptions(&prefill_runtime, .{
        .logits_buf = prefill_host_scratch[0..],
        .output_resource = prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = prefill_k,
            .v = prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer resource_prefill.deinit();

    var prefill_resource_logits = [_]f32{0} ** cfg.vocab_size;
    try resource_prefill.prefillBoundOutput(&.{ 0, 1 });
    try testing.expectEqual(@as(usize, 2), prefill_runtime.position());
    try testing.expectEqual(@as(f32, -8181), prefill_host_scratch[0]);
    try readWgpuF32Resource(prefill_handle, prefill_output, prefill_resource_logits[0..]);
    for (prefill_resource_logits, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    const decode_device = try wgpu_mod.deviceHandle(decode_handle);
    const prefill_device = try wgpu_mod.deviceHandle(prefill_handle);
    try testing.expectEqual(decode_device, prefill_device);

    const shared_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    const shared_prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var shared_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var shared_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var shared_prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var shared_prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        shared_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        shared_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, decode_handle, shared_k[l]);
        try zeroWgpuResource(testing.allocator, decode_handle, shared_v[l]);
        shared_prefill_k[l] = try importWgpuResourceView(prefill_handle, decode_device, shared_k[l], .read_write);
        shared_prefill_v[l] = try importWgpuResourceView(prefill_handle, decode_device, shared_v[l], .read_write);
        try testing.expectEqual(shared_k[l].handle, shared_prefill_k[l].handle);
        try testing.expectEqual(shared_v[l].handle, shared_prefill_v[l].handle);
    }

    var shared_runtime = try model_session.bindRuntime(testing.allocator);
    defer shared_runtime.deinit();
    var shared_decode_scratch = [_]f32{-9191} ** cfg.vocab_size;
    var shared_decode = try decode_program.bindWithOptions(&shared_runtime, .{
        .logits_buf = shared_decode_scratch[0..],
        .output_resource = shared_output,
        .cache_bindings = .{ .resource = .{
            .k = shared_k,
            .v = shared_v,
            .element_count = cache_elements,
        } },
    });
    defer shared_decode.deinit();

    var shared_prefill_scratch = [_]f32{-9292} ** cfg.vocab_size;
    var shared_prefill = try prefill_program.bindWithOptions(&shared_runtime, .{
        .logits_buf = shared_prefill_scratch[0..],
        .output_resource = shared_prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = shared_prefill_k,
            .v = shared_prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer shared_prefill.deinit();

    var shared_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try shared_prefill.prefillBoundOutput(&.{ 0, 1 });
    try testing.expectEqual(@as(usize, 2), shared_runtime.position());
    try testing.expectEqual(@as(f32, -9292), shared_prefill_scratch[0]);
    try readWgpuF32Resource(prefill_handle, shared_prefill_output, shared_prefill_logits[0..]);
    for (shared_prefill_logits, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try shared_decode.refreshCacheBindings();
    var shared_after_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try shared_decode.stepBoundOutput(2);
    try testing.expectEqual(@as(usize, 3), shared_runtime.position());
    try testing.expectEqual(@as(f32, -9191), shared_decode_scratch[0]);
    try readWgpuF32Resource(decode_handle, shared_output, shared_after_prefill_logits[0..]);
    for (shared_after_prefill_logits, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }
}

fn proveNativeWgpuTinyLlamaResourceHandoff(comptime cfg: LlamaConfig) !void {
    if (!opts.use_wgpu) return error.SkipZigTest;
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);

    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_expected.deinit();
    const expected_prefill = try testing.allocator.dupe(f32, try cpu_expected.prefill(&.{ 0, 1 }));
    defer testing.allocator.free(expected_prefill);
    const expected_after_prefill = try testing.allocator.dupe(f32, try cpu_expected.step(2));
    defer testing.allocator.free(expected_after_prefill);

    var wgpu = wgpu_mod.WgpuBackend.init(testing.allocator);
    defer wgpu.deinit();
    var decode_program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), testing.allocator);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(wgpu.backend(), testing.allocator, 2);
    defer prefill_program.deinit();

    const decode_inspection = decode_program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, decode_inspection.backend);
    try testing.expect(decode_inspection.execution_supported);
    try testing.expect(decode_inspection.command_shape.command_count > 0);
    const decode_categories = decode_inspection.command_shape.categoryCounts();
    try testing.expect(decode_categories.attention > 0);
    const decode_dispatch_plan = try expectFullNativeWgpuDispatchPlan(decode_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(decode_dispatch_plan);

    const prefill_inspection = prefill_program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, prefill_inspection.backend);
    try testing.expect(prefill_inspection.execution_supported);
    const prefill_dispatch_plan = try expectFullNativeWgpuDispatchPlan(prefill_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(prefill_dispatch_plan);

    const cache_elements = try Session.DeviceDecodeProgram.cacheElementCountForContext(cfg.max_seq_len);
    const decode_handle = decode_program.program.handle;
    const prefill_handle = prefill_program.program.handle;
    const decode_device = try wgpu_mod.deviceHandle(decode_handle);
    const prefill_device = try wgpu_mod.deviceHandle(prefill_handle);
    try testing.expectEqual(decode_device, prefill_device);

    const decode_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    const prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var decode_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var decode_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        decode_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        decode_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_k[l]);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_v[l]);
        prefill_k[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_k[l], .read_write);
        prefill_v[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_v[l], .read_write);
        try testing.expectEqual(decode_k[l].handle, prefill_k[l].handle);
        try testing.expectEqual(decode_v[l].handle, prefill_v[l].handle);
    }

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();

    var decode_scratch = [_]f32{-3030} ** cfg.vocab_size;
    var decode = try decode_program.bindWithOptions(&runtime, .{
        .logits_buf = decode_scratch[0..],
        .output_resource = decode_output,
        .cache_bindings = .{ .resource = .{
            .k = decode_k,
            .v = decode_v,
            .element_count = cache_elements,
        } },
    });
    defer decode.deinit();

    var prefill_scratch = [_]f32{-4040} ** cfg.vocab_size;
    var prefill = try prefill_program.bindWithOptions(&runtime, .{
        .logits_buf = prefill_scratch[0..],
        .output_resource = prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = prefill_k,
            .v = prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer prefill.deinit();

    try prefill.prefillBoundOutput(&.{ 0, 1 });
    try testing.expectEqual(@as(usize, 2), runtime.position());
    try testing.expectEqual(@as(f32, -4040), prefill_scratch[0]);

    var prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(prefill_handle, prefill_output, prefill_logits[0..]);
    for (prefill_logits, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try decode.refreshCacheBindings();
    try decode.stepBoundOutput(2);
    try testing.expectEqual(@as(usize, 3), runtime.position());
    try testing.expectEqual(@as(f32, -3030), decode_scratch[0]);

    var after_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(decode_handle, decode_output, after_prefill_logits[0..]);
    for (after_prefill_logits, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    const no_output_decode_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    const no_output_prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var no_output_prefill_sentinel = [_]f32{-5050} ** cfg.vocab_size;
    try writeWgpuF32Resource(prefill_handle, no_output_prefill_output, no_output_prefill_sentinel[0..]);
    var no_output_decode_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var no_output_decode_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var no_output_prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var no_output_prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        no_output_decode_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        no_output_decode_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, decode_handle, no_output_decode_k[l]);
        try zeroWgpuResource(testing.allocator, decode_handle, no_output_decode_v[l]);
        no_output_prefill_k[l] = try importWgpuResourceView(prefill_handle, decode_device, no_output_decode_k[l], .read_write);
        no_output_prefill_v[l] = try importWgpuResourceView(prefill_handle, decode_device, no_output_decode_v[l], .read_write);
    }

    var no_output_runtime = try model_session.bindRuntime(testing.allocator);
    defer no_output_runtime.deinit();

    var no_output_decode_scratch = [_]f32{-6060} ** cfg.vocab_size;
    var no_output_decode = try decode_program.bindWithOptions(&no_output_runtime, .{
        .logits_buf = no_output_decode_scratch[0..],
        .output_resource = no_output_decode_output,
        .cache_bindings = .{ .resource = .{
            .k = no_output_decode_k,
            .v = no_output_decode_v,
            .element_count = cache_elements,
        } },
    });
    defer no_output_decode.deinit();

    var no_output_prefill_scratch = [_]f32{-7070} ** cfg.vocab_size;
    var no_output_prefill = try prefill_program.bindWithOptions(&no_output_runtime, .{
        .logits_buf = no_output_prefill_scratch[0..],
        .output_resource = no_output_prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = no_output_prefill_k,
            .v = no_output_prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer no_output_prefill.deinit();

    try no_output_prefill.advanceTokens(&.{ 0, 1 });
    try testing.expectEqual(@as(usize, 2), no_output_runtime.position());
    try testing.expectEqual(@as(f32, -7070), no_output_prefill_scratch[0]);

    var untouched_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(prefill_handle, no_output_prefill_output, untouched_prefill_logits[0..]);
    try testing.expectEqualSlices(f32, no_output_prefill_sentinel[0..], untouched_prefill_logits[0..]);

    var no_output_prefill_profile = profile_mod.RuntimeProfile{};
    no_output_prefill.addRuntimeProfileTo(&no_output_prefill_profile);
    try testing.expectEqual(@as(u64, 1), no_output_prefill_profile.call_count);
    try testing.expect(no_output_prefill_profile.backend_op_count > 0);
    try testing.expectEqual(prefill_dispatch_plan.dispatch_count, no_output_prefill_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), no_output_prefill_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), no_output_prefill_profile.sync_count);

    try no_output_decode.refreshCacheBindings();
    try no_output_decode.stepBoundOutput(2);
    try testing.expectEqual(@as(usize, 3), no_output_runtime.position());
    try testing.expectEqual(@as(f32, -6060), no_output_decode_scratch[0]);

    var no_output_decode_profile = profile_mod.RuntimeProfile{};
    no_output_decode.addRuntimeProfileTo(&no_output_decode_profile);
    try testing.expectEqual(@as(u64, 1), no_output_decode_profile.call_count);
    try testing.expect(no_output_decode_profile.backend_op_count > 0);
    try testing.expectEqual(decode_dispatch_plan.dispatch_count, no_output_decode_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), no_output_decode_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), no_output_decode_profile.sync_count);

    var no_output_after_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(decode_handle, no_output_decode_output, no_output_after_prefill_logits[0..]);
    for (no_output_after_prefill_logits, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }
}

test "native WebGPU tiny llama GQA resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaResourceHandoff(.{
        .vocab_size = 8,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 4,
    });
}

test "native WebGPU tiny llama multilayer resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaResourceHandoff(.{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 4,
    });
}

test "native WebGPU tiny llama MQA multilayer resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaResourceHandoff(.{
        .vocab_size = 8,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 1,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 4,
    });
}

fn proveNativeWgpuTinyLlamaLongPromptResourceHandoff(comptime cfg: LlamaConfig, comptime prompt_len: usize) !void {
    if (!opts.use_wgpu) return error.SkipZigTest;
    comptime {
        if (prompt_len < 2) @compileError("long prompt proof needs at least two prompt tokens");
        if (prompt_len >= cfg.max_seq_len) @compileError("long prompt proof needs room for a decode token");
    }
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);

    var prompt: [prompt_len]usize = undefined;
    for (&prompt, 0..) |*token, i| token.* = i % cfg.vocab_size;
    const decode_token = prompt_len % cfg.vocab_size;

    var model_session = try Session.init(testing.allocator);
    defer model_session.deinit();

    var cpu_expected = try model_session.bindRuntime(testing.allocator);
    defer cpu_expected.deinit();
    const expected_prefill = try testing.allocator.dupe(f32, try cpu_expected.prefill(&prompt));
    defer testing.allocator.free(expected_prefill);
    const expected_after_prefill = try testing.allocator.dupe(f32, try cpu_expected.step(decode_token));
    defer testing.allocator.free(expected_after_prefill);

    var wgpu = wgpu_mod.WgpuBackend.init(testing.allocator);
    defer wgpu.deinit();
    var decode_program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), testing.allocator);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(wgpu.backend(), testing.allocator, prompt_len);
    defer prefill_program.deinit();

    const decode_inspection = decode_program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, decode_inspection.backend);
    try testing.expect(decode_inspection.execution_supported);
    try testing.expect(decode_inspection.command_shape.categoryCounts().attention > 0);
    const decode_dispatch_plan = try expectFullNativeWgpuDispatchPlan(decode_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(decode_dispatch_plan);

    const prefill_inspection = prefill_program.inspectExecutable();
    try testing.expectEqual(backend_mod.Device.webgpu, prefill_inspection.backend);
    try testing.expect(prefill_inspection.execution_supported);
    try testing.expect(prefill_inspection.command_shape.categoryCounts().attention > 0);
    try testing.expect(prefill_inspection.runtime_patch_envelope.max_attention_seq_kv != null);
    try testing.expect(prefill_inspection.runtime_patch_envelope.max_attention_seq_kv.? >= prompt_len);
    const prefill_dispatch_plan = try expectFullNativeWgpuDispatchPlan(prefill_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(prefill_dispatch_plan);

    const cache_elements = try Session.DeviceDecodeProgram.cacheElementCountForContext(cfg.max_seq_len);
    const decode_handle = decode_program.program.handle;
    const prefill_handle = prefill_program.program.handle;
    const decode_device = try wgpu_mod.deviceHandle(decode_handle);
    const prefill_device = try wgpu_mod.deviceHandle(prefill_handle);
    try testing.expectEqual(decode_device, prefill_device);

    const decode_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    const prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var decode_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var decode_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        decode_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        decode_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_k[l]);
        try zeroWgpuResource(testing.allocator, decode_handle, decode_v[l]);
        prefill_k[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_k[l], .read_write);
        prefill_v[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_v[l], .read_write);
    }

    var runtime = try model_session.bindRuntime(testing.allocator);
    defer runtime.deinit();
    var decode_scratch = [_]f32{-2121} ** cfg.vocab_size;
    var decode = try decode_program.bindWithOptions(&runtime, .{
        .logits_buf = decode_scratch[0..],
        .output_resource = decode_output,
        .cache_bindings = .{ .resource = .{
            .k = decode_k,
            .v = decode_v,
            .element_count = cache_elements,
        } },
    });
    defer decode.deinit();

    var prefill_scratch = [_]f32{-3131} ** cfg.vocab_size;
    var prefill = try prefill_program.bindWithOptions(&runtime, .{
        .logits_buf = prefill_scratch[0..],
        .output_resource = prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = prefill_k,
            .v = prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer prefill.deinit();

    try prefill.prefillBoundOutput(&prompt);
    try testing.expectEqual(@as(usize, prompt_len), runtime.position());
    try testing.expectEqual(@as(f32, -3131), prefill_scratch[0]);
    var prefill_profile = profile_mod.RuntimeProfile{};
    prefill.addRuntimeProfileTo(&prefill_profile);
    try testing.expectEqual(@as(u64, 1), prefill_profile.call_count);
    try testing.expect(prefill_profile.backend_op_count > 0);
    try testing.expectEqual(prefill_dispatch_plan.dispatch_count, prefill_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), prefill_profile.fallback_op_count);

    var prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(prefill_handle, prefill_output, prefill_logits[0..]);
    for (prefill_logits, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try decode.refreshCacheBindings();
    try decode.stepBoundOutput(decode_token);
    try testing.expectEqual(@as(usize, prompt_len + 1), runtime.position());
    try testing.expectEqual(@as(f32, -2121), decode_scratch[0]);
    var decode_profile = profile_mod.RuntimeProfile{};
    decode.addRuntimeProfileTo(&decode_profile);
    try testing.expectEqual(@as(u64, 1), decode_profile.call_count);
    try testing.expect(decode_profile.backend_op_count > 0);
    try testing.expectEqual(decode_dispatch_plan.dispatch_count, decode_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), decode_profile.fallback_op_count);

    var after_prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(decode_handle, decode_output, after_prefill_logits[0..]);
    for (after_prefill_logits, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }
}

test "native WebGPU tiny llama long prompt resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaLongPromptResourceHandoff(.{
        .vocab_size = 8,
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 1,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 16,
    }, 8);
}

test "native WebGPU tiny llama GQA multilayer long prompt resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaLongPromptResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

test "native WebGPU LLaMA realistic head width resource handoff matches CPU" {
    try proveNativeWgpuTinyLlamaLongPromptResourceHandoff(.{
        .vocab_size = 16,
        .d_model = 128,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 256,
        .n_layers = 1,
        .max_seq_len = 16,
    }, 8);
}

fn proveNativeWgpuTinyLlamaResourceHotPathNoAlloc(comptime cfg: LlamaConfig, comptime prompt_len: usize) !void {
    if (!opts.use_wgpu) return error.SkipZigTest;
    comptime {
        if (prompt_len < 2) @compileError("hot path proof needs at least two prompt tokens");
        if (prompt_len >= cfg.max_seq_len) @compileError("hot path proof needs room for a decode token");
    }
    const Session = llama_inference.LlamaInferenceSession(f32, cfg);
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    var prompt: [prompt_len]usize = undefined;
    for (&prompt, 0..) |*token, i| token.* = i % cfg.vocab_size;
    const decode_token = prompt_len % cfg.vocab_size;

    var model_session = try Session.init(alloc);
    defer model_session.deinit();

    var cpu_expected = try model_session.bindRuntime(alloc);
    defer cpu_expected.deinit();
    const expected_prefill = try alloc.dupe(f32, try cpu_expected.prefill(&prompt));
    defer alloc.free(expected_prefill);
    const expected_after_prefill = try alloc.dupe(f32, try cpu_expected.step(decode_token));
    defer alloc.free(expected_after_prefill);

    var wgpu = wgpu_mod.WgpuBackend.init(alloc);
    defer wgpu.deinit();
    var decode_program = try model_session.compileDeviceDecodeProgram(wgpu.backend(), alloc);
    defer decode_program.deinit();
    var prefill_program = try model_session.compileDevicePrefillProgram(wgpu.backend(), alloc, prompt_len);
    defer prefill_program.deinit();

    const decode_plan = try expectFullNativeWgpuDispatchPlan(decode_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(decode_plan);
    const prefill_plan = try expectFullNativeWgpuDispatchPlan(prefill_program.program.handle);
    try expectNativeWgpuLlamaDispatchFamilies(prefill_plan);

    const cache_elements = try Session.DeviceDecodeProgram.cacheElementCountForContext(cfg.max_seq_len);
    const decode_handle = decode_program.program.handle;
    const prefill_handle = prefill_program.program.handle;
    const decode_device = try wgpu_mod.deviceHandle(decode_handle);
    try testing.expectEqual(decode_device, try wgpu_mod.deviceHandle(prefill_handle));

    const decode_output = try createWgpuF32Resource(decode_handle, cfg.vocab_size, .write_only);
    const prefill_output = try createWgpuF32Resource(prefill_handle, cfg.vocab_size, .write_only);
    var decode_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var decode_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_k: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    var prefill_v: [cfg.n_layers]backend_mod.ProgramIO.ExternalResource = undefined;
    for (0..cfg.n_layers) |l| {
        decode_k[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        decode_v[l] = try createWgpuF32Resource(decode_handle, cache_elements, .read_write);
        try zeroWgpuResource(alloc, decode_handle, decode_k[l]);
        try zeroWgpuResource(alloc, decode_handle, decode_v[l]);
        prefill_k[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_k[l], .read_write);
        prefill_v[l] = try importWgpuResourceView(prefill_handle, decode_device, decode_v[l], .read_write);
    }

    var runtime = try model_session.bindRuntime(alloc);
    defer runtime.deinit();
    var decode_scratch = [_]f32{-4242} ** cfg.vocab_size;
    var decode = try decode_program.bindWithOptions(&runtime, .{
        .logits_buf = decode_scratch[0..],
        .output_resource = decode_output,
        .cache_bindings = .{ .resource = .{
            .k = decode_k,
            .v = decode_v,
            .element_count = cache_elements,
        } },
    });
    defer decode.deinit();

    var prefill_scratch = [_]f32{-5252} ** cfg.vocab_size;
    var prefill = try prefill_program.bindWithOptions(&runtime, .{
        .logits_buf = prefill_scratch[0..],
        .output_resource = prefill_output,
        .cache_bindings = .{ .resource = .{
            .k = prefill_k,
            .v = prefill_v,
            .element_count = cache_elements,
        } },
    });
    defer prefill.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    try prefill.prefillBoundOutput(&prompt);
    try testing.expectEqual(@as(usize, prompt_len), runtime.position());
    try testing.expectEqual(@as(f32, -5252), prefill_scratch[0]);
    var prefill_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(prefill_handle, prefill_output, prefill_logits[0..]);
    for (prefill_logits, expected_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    try decode.refreshCacheBindings();
    try decode.stepBoundOutput(decode_token);
    try testing.expectEqual(@as(usize, prompt_len + 1), runtime.position());
    try testing.expectEqual(@as(f32, -4242), decode_scratch[0]);
    var decode_logits = [_]f32{0} ** cfg.vocab_size;
    try readWgpuF32Resource(decode_handle, decode_output, decode_logits[0..]);
    for (decode_logits, expected_after_prefill) |actual, want| {
        try testing.expectApproxEqAbs(want, actual, 1e-3);
    }

    var prefill_profile = profile_mod.RuntimeProfile{};
    prefill.addRuntimeProfileTo(&prefill_profile);
    try testing.expectEqual(@as(u64, 1), prefill_profile.call_count);
    try testing.expectEqual(prefill_plan.dispatch_count, prefill_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), prefill_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), prefill_profile.sync_count);

    var decode_profile = profile_mod.RuntimeProfile{};
    decode.addRuntimeProfileTo(&decode_profile);
    try testing.expectEqual(@as(u64, 1), decode_profile.call_count);
    try testing.expectEqual(decode_plan.dispatch_count, decode_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), decode_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), decode_profile.sync_count);

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);
}

test "native WebGPU tiny llama resource handoff hot path does not allocate" {
    try proveNativeWgpuTinyLlamaResourceHotPathNoAlloc(.{
        .vocab_size = 16,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .max_seq_len = 16,
    }, 8);
}

test "llm facade exposes a usable session type" {
    const Session = LlamaSession(f32, .{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    });
    try testing.expect(@hasDecl(Session, "init"));
    try testing.expect(@hasDecl(Session, "load"));
    try testing.expect(@hasDecl(Session, "prefill"));
    try testing.expect(@hasDecl(Session, "prefillInto"));
    try testing.expect(@hasDecl(Session, "step"));
    try testing.expect(@hasDecl(Session, "stepInto"));
    try testing.expect(@hasDecl(Session, "stepArgmax"));
    try testing.expect(@hasDecl(Session, "stepSample"));
    try testing.expect(@hasDecl(Session, "advance"));
    try testing.expect(@hasDecl(Session, "advanceTokens"));
    try testing.expect(@hasDecl(Session, "execute"));
    try testing.expect(@hasDecl(Session, "executeInto"));
    try testing.expect(@hasDecl(Session, "generateArgmaxInto"));
    try testing.expect(@hasDecl(Session, "generateSampleInto"));
    try testing.expect(@hasDecl(Session, "inspect"));
    try testing.expect(!@hasDecl(Session, "quantizeWeights"));
    try testing.expect(!@hasDecl(Session, "quantizeKV"));
}

test "llm facade supports model program session lifecycle" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const inspection = program.inspect();
    try testing.expectEqual(@as(usize, cfg.vocab_size), inspection.vocab_size);
    try testing.expectEqual(@as(usize, cfg.max_seq_len), inspection.max_seq_len);
    try testing.expectEqual(@as(usize, 4), inspection.context_len);
    try testing.expectEqual(@as(usize, 1), inspection.batch);
    try testing.expectEqual(@as(usize, cfg.n_layers), inspection.n_layers);
    try testing.expectEqual(@as(usize, cfg.n_heads), inspection.n_heads);
    try testing.expectEqual(@as(usize, cfg.n_kv_heads), inspection.n_kv_heads);
    try testing.expectEqual(@as(usize, 1), inspection.semantic_token_count);
    try testing.expect(inspection.semantic_stage_count > 0);
    try testing.expect(inspection.semantic_runtime_patch_holes > 0);

    const executable = program.inspectExecutable();
    try testing.expectEqual(LlamaBackend.cpu, executable.backend);
    try testing.expect(executable.execution_supported);
    try testing.expectEqual(LlamaExecutionMode.executable, executable.execution_mode);
    try testing.expect(executable.command_count > 0);
    try testing.expect(executable.command_projection_count > 0 or executable.command_attention_count > 0);
    try testing.expect(executable.command_stencil_hash != 0);
    try testing.expect(executable.runtime_patch_holes > 0);
    try testing.expectEqual(inspection.semantic_runtime_patch_holes, executable.runtime_patch_holes);
    try testing.expectEqual(inspection.semantic_runtime_patch_cache_write_pos_holes, executable.runtime_patch_cache_write_pos_holes);
    try testing.expectEqual(inspection.semantic_runtime_patch_attention_seq_kv_holes, executable.runtime_patch_attention_seq_kv_holes);
    try testing.expect(executable.runtime_patch_stencil_hash != 0);

    const decode_inspection = program.state().decode_program.?.inspectExecutable();
    const decode_command_categories = decode_inspection.command_shape.categoryCounts();
    try testing.expectEqual(decode_inspection.execution_supported, executable.execution_supported);
    try testing.expectEqual(executable.command_count, decode_inspection.command_shape.command_count);
    try testing.expectEqual(executable.command_stencil_hash, decode_inspection.command_shape.command_stencil_hash);
    try testing.expectEqual(executable.command_op_count, decode_command_categories.op);
    try testing.expectEqual(executable.command_row_count, decode_command_categories.row);
    try testing.expectEqual(executable.command_projection_count, decode_command_categories.projection);
    try testing.expectEqual(executable.command_attention_count, decode_command_categories.attention);
    try testing.expectEqual(executable.command_movement_count, decode_command_categories.movement);
    try testing.expectEqual(executable.command_elementwise_count, decode_command_categories.elementwise);
    try testing.expectEqual(executable.command_rope_count, decode_command_categories.rope);
    try testing.expectEqual(executable.runtime_patch_holes, @as(usize, decode_inspection.runtime_patch_shape.runtime_patch_holes));
    try testing.expectEqual(executable.runtime_patch_cache_write_pos_holes, @as(usize, decode_inspection.runtime_patch_shape.runtime_patch_cache_write_pos_holes));
    try testing.expectEqual(executable.runtime_patch_attention_seq_kv_holes, @as(usize, decode_inspection.runtime_patch_shape.runtime_patch_attention_seq_kv_holes));
    try testing.expectEqual(executable.runtime_patch_stencil_hash, decode_inspection.runtime_patch_shape.runtime_patch_stencil_hash);

    const session = try program.bind(.{});
    defer session.deinit();

    const logits = try session.step(.{ .token = 0 });
    try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
    try testing.expectEqual(@as(usize, 1), session.position());
}

test "llm facade compiles GQA KV caches to the context envelope" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .n_layers = 1,
        .max_seq_len = 8,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const context_len: usize = 3;
    const program = try model.compile(.{ .context_len = context_len, .batch = 1 });
    defer program.deinit();

    const d_head = cfg.d_model / cfg.n_heads;
    const packed_elements = d_head * context_len * cfg.n_kv_heads;
    const max_stride_elements = d_head * ((cfg.n_kv_heads - 1) * cfg.max_seq_len + context_len);

    try testing.expectEqual(packed_elements, program.state().model.inner.k_caches[0].data.len);
    try testing.expectEqual(packed_elements, program.state().model.inner.v_caches[0].data.len);
    try testing.expect(max_stride_elements > packed_elements);

    const k_storage = try testing.allocator.alloc(f32, packed_elements);
    defer testing.allocator.free(k_storage);
    const v_storage = try testing.allocator.alloc(f32, packed_elements);
    defer testing.allocator.free(v_storage);
    @memset(k_storage, 0);
    @memset(v_storage, 0);

    var k_buffers = [_]LlamaBindOptions.HostBuffer{.{
        .ptr = @ptrCast(k_storage.ptr),
        .byte_len = packed_elements * @sizeOf(f32),
    }};
    var v_buffers = [_]LlamaBindOptions.HostBuffer{.{
        .ptr = @ptrCast(v_storage.ptr),
        .byte_len = packed_elements * @sizeOf(f32),
    }};
    const session = try program.bind(.{ .cache_buffers = .{
        .k = &k_buffers,
        .v = &v_buffers,
        .element_count = packed_elements,
    } });
    defer session.deinit();

    const inspection = session.inspect();
    try testing.expectEqual(@as(usize, context_len), inspection.context_len);
    try testing.expectEqual(@as(u32, 1), inspection.kv_cache_storage);
    _ = try session.step(.{ .token = 0 });
    try testing.expectEqual(@as(usize, 1), session.position());
}

test "llm compile options reject unsupported batch sizes" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    try testing.expectError(error.InvalidBatch, model.compile(.{ .batch = 0 }));
    try testing.expectError(error.UnsupportedBatch, model.compile(.{ .batch = 2 }));
}

test "llm executable sessions enforce compiled context length" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 1, .batch = 1 });
    defer program.deinit();

    const inspection = program.inspect();
    try testing.expectEqual(@as(usize, 1), inspection.context_len);

    const stepped = try program.bind(.{});
    defer stepped.deinit();
    _ = try stepped.step(.{ .token = 0 });
    try testing.expectEqual(@as(usize, 1), stepped.position());
    try testing.expectError(error.SequenceTooLong, stepped.step(.{ .token = 1 }));
    try testing.expectEqual(@as(usize, 1), stepped.position());

    const advanced = try program.bind(.{});
    defer advanced.deinit();
    try advanced.advance(.{ .token = 0 });
    try testing.expectError(error.SequenceTooLong, advanced.advance(.{ .token = 1 }));

    const prefilled = try program.bind(.{});
    defer prefilled.deinit();
    try testing.expectError(error.SequenceTooLong, prefilled.prefill(&.{ 0, 1 }));
}

test "llm program binds independent runtime sessions" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const a = try program.bind(.{});
    defer a.deinit();
    const b = try program.bind(.{});
    defer b.deinit();

    const a0 = try a.step(.{ .token = 0 });
    try testing.expectEqual(@as(usize, 1), a.position());
    try testing.expectEqual(@as(usize, 0), b.position());

    const b0 = try b.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, a0, b0);
    try testing.expectEqual(@as(usize, 1), a.position());
    try testing.expectEqual(@as(usize, 1), b.position());

    _ = try a.step(.{ .token = 1 });
    try testing.expectEqual(@as(usize, 2), a.position());
    try testing.expectEqual(@as(usize, 1), b.position());
}

test "llm program can bind a compatible model as session state" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const compile_model = try Model.init(testing.allocator);
    defer compile_model.deinit();
    const runtime_model = try Model.init(testing.allocator);
    defer runtime_model.deinit();

    const program = try compile_model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const default_session = try program.bind(.{});
    defer default_session.deinit();
    const default_logits = try testing.allocator.dupe(f32, try default_session.step(.{ .token = 0 }));
    defer testing.allocator.free(default_logits);

    @memset(runtime_model.state().inner.model.out_proj.data, 0);
    const rebound_session = try program.bindModel(runtime_model, .{});
    defer rebound_session.deinit();
    const rebound_logits = try rebound_session.step(.{ .token = 0 });
    try testing.expectEqual(@as(usize, cfg.vocab_size), rebound_logits.len);
    try testing.expect(!std.mem.eql(f32, default_logits, rebound_logits));
    for (rebound_logits) |logit| try testing.expectEqual(@as(f32, 0), logit);

    const compile_model_session = try program.bindModel(compile_model, .{});
    defer compile_model_session.deinit();
    const compile_model_logits = try compile_model_session.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, default_logits, compile_model_logits);
}

test "llm program binds executable decode sessions" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const host = try program.bind(.{});
    defer host.deinit();
    const expected = try host.step(.{ .token = 0 });

    const a = try program.bindExecutableDecode(.{});
    defer a.deinit();
    const b = try program.bindExecutableDecode(.{});
    defer b.deinit();

    const a0 = try a.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, expected, a0);
    try testing.expectEqual(@as(usize, 1), a.position());
    try testing.expectEqual(@as(usize, 0), b.position());

    const b0 = try b.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, a0, b0);
    try testing.expectEqual(@as(usize, 1), a.position());
    try testing.expectEqual(@as(usize, 1), b.position());

    a.reset();
    try testing.expectEqual(@as(usize, 0), a.position());
    const after_reset = try a.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, expected, after_reset);

    host.reset();
    _ = try host.step(.{ .token = 0 });
    const expected_after_advance = try host.step(.{ .token = 1 });

    const advancing = try program.bindExecutableDecode(.{});
    defer advancing.deinit();
    try advancing.advance(.{ .token = 0 });
    try testing.expectEqual(@as(usize, 1), advancing.position());
    var caller_logits = [_]f32{-321} ** (cfg.vocab_size + 1);
    const after_advance = try advancing.stepInto(&caller_logits, .{ .token = 1 });
    try testing.expect(after_advance.ptr == caller_logits[0..cfg.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -321), caller_logits[cfg.vocab_size]);
    try testing.expectEqualSlices(f32, expected_after_advance, after_advance);
    try testing.expectEqual(@as(usize, 2), advancing.position());

    const prefill_host = try program.bind(.{});
    defer prefill_host.deinit();
    const expected_prefill = try prefill_host.prefill(&.{ 0, 1 });

    const prefill_exec = try program.bindExecutableDecode(.{});
    defer prefill_exec.deinit();
    var prefill_logits = [_]f32{-654} ** (cfg.vocab_size + 1);
    const got_prefill = try prefill_exec.prefillInto(&prefill_logits, &.{ 0, 1 });
    try testing.expect(got_prefill.ptr == prefill_logits[0..cfg.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -654), prefill_logits[cfg.vocab_size]);
    try testing.expectEqualSlices(f32, expected_prefill, got_prefill);
    try testing.expectEqual(@as(usize, 2), prefill_exec.position());

    prefill_exec.reset();
    try testing.expectEqual(@as(usize, 0), prefill_exec.position());
    @memset(&prefill_logits, -456);
    const got_after_reset = try prefill_exec.prefillInto(&prefill_logits, &.{ 0, 1 });
    try testing.expectEqualSlices(f32, expected_prefill, got_after_reset);
    try testing.expectEqual(@as(f32, -456), prefill_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), prefill_exec.position());
}

test "llm execute accepts token windows and output policy" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const expected_session = try program.bind(.{});
    defer expected_session.deinit();
    const expected = try testing.allocator.dupe(f32, try expected_session.prefill(&.{ 0, 1 }));
    defer testing.allocator.free(expected);

    const executed = try program.bind(.{});
    defer executed.deinit();
    const got = try executed.execute(.{ .tokens = &.{ 0, 1 }, .output = .logits });
    try testing.expectEqualSlices(f32, expected, got);
    try testing.expectEqual(@as(usize, 2), executed.position());

    const no_output = try program.bind(.{});
    defer no_output.deinit();
    const empty = try no_output.execute(.{ .tokens = &.{ 0, 1 }, .output = .none });
    try testing.expectEqual(@as(usize, 0), empty.len);
    try testing.expectEqual(@as(usize, 2), no_output.position());

    var caller_logits = [_]f32{-777} ** (cfg.vocab_size + 1);
    const next = try no_output.executeInto(&caller_logits, .{ .tokens = &.{2}, .output = .logits });
    try testing.expect(next.ptr == caller_logits[0..cfg.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -777), caller_logits[cfg.vocab_size]);

    const expected_next_session = try program.bind(.{});
    defer expected_next_session.deinit();
    const expected_next = try expected_next_session.prefill(&.{ 0, 1, 2 });
    try testing.expectEqualSlices(f32, expected_next, next);
    try testing.expectEqual(@as(usize, 3), no_output.position());

    try testing.expectError(error.InvalidPrefillLength, no_output.execute(.{ .tokens = &.{}, .output = .none }));

    const short_program = try model.compile(.{ .context_len = 1, .batch = 1 });
    defer short_program.deinit();
    const short_session = try short_program.bind(.{});
    defer short_session.deinit();
    try testing.expectError(error.SequenceTooLong, short_session.execute(.{ .tokens = &.{ 0, 1 }, .output = .logits }));
    try testing.expectEqual(@as(usize, 0), short_session.position());
}

test "llm executable token windows do not allocate after session bind" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    const model = try Model.init(alloc);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const full_logits_session = try program.bind(.{});
    defer full_logits_session.deinit();
    const short_logits_session = try program.bind(.{});
    defer short_logits_session.deinit();
    const full_no_output_session = try program.bind(.{});
    defer full_no_output_session.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    var full_logits = [_]f32{-999} ** (cfg.vocab_size + 1);
    const full = try full_logits_session.executeInto(&full_logits, .{ .tokens = &.{ 0, 1, 2, 3 }, .output = .logits });
    try testing.expectEqual(@as(usize, cfg.vocab_size), full.len);
    try testing.expectEqual(@as(f32, -999), full_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 4), full_logits_session.position());

    var short_logits = [_]f32{-777} ** (cfg.vocab_size + 1);
    const short = try short_logits_session.executeInto(&short_logits, .{ .tokens = &.{ 0, 1 }, .output = .logits });
    try testing.expectEqual(@as(usize, cfg.vocab_size), short.len);
    try testing.expectEqual(@as(f32, -777), short_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), short_logits_session.position());
    try short_logits_session.advanceTokens(&.{ 2, 3 });
    try testing.expectEqual(@as(usize, 4), short_logits_session.position());

    try full_no_output_session.advanceTokens(&.{ 0, 1, 2, 3 });
    try testing.expectEqual(@as(usize, 4), full_no_output_session.position());

    full_logits_session.reset();
    try testing.expectEqual(@as(usize, 0), full_logits_session.position());
    @memset(&full_logits, -333);
    const replay_full = try full_logits_session.executeInto(&full_logits, .{ .tokens = &.{ 0, 1, 2, 3 }, .output = .logits });
    try testing.expectEqual(@as(usize, cfg.vocab_size), replay_full.len);
    try testing.expectEqual(@as(f32, -333), full_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 4), full_logits_session.position());

    short_logits_session.reset();
    try testing.expectEqual(@as(usize, 0), short_logits_session.position());
    @memset(&short_logits, -222);
    const replay_short = try short_logits_session.executeInto(&short_logits, .{ .tokens = &.{ 0, 1 }, .output = .logits });
    try testing.expectEqual(@as(usize, cfg.vocab_size), replay_short.len);
    try testing.expectEqual(@as(f32, -222), short_logits[cfg.vocab_size]);
    try testing.expectEqual(@as(usize, 2), short_logits_session.position());

    full_no_output_session.reset();
    try testing.expectEqual(@as(usize, 0), full_no_output_session.position());
    try full_no_output_session.advanceTokens(&.{ 0, 1, 2, 3 });
    try testing.expectEqual(@as(usize, 4), full_no_output_session.position());

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);
}

test "llm facade generates greedy tokens into caller-owned output" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const expected = try program.bind(.{});
    defer expected.deinit();
    const expected_first = try expected.stepArgmax(.{ .token = 0 });
    const expected_second = try expected.stepArgmax(.{ .token = expected_first.token });

    const generated_session = try program.bind(.{});
    defer generated_session.deinit();
    var output_tokens = [_]usize{ 999, 999 };
    const generated = try generated_session.generateArgmaxInto(&output_tokens, &.{0});
    try testing.expectEqual(@as(usize, 2), generated.tokens_generated);
    try testing.expectEqual(expected_first.token, output_tokens[0]);
    try testing.expectEqual(expected_second.token, output_tokens[1]);
    try testing.expectEqual(expected_second.token, generated.last_token);
    try testing.expectEqual(expected_second.logit, generated.last_logit);
    try testing.expectEqual(@as(usize, 2), generated_session.position());

    const empty_output = try program.bind(.{});
    defer empty_output.deinit();
    try testing.expectError(error.InvalidTokenWindow, empty_output.generateArgmaxInto(&.{}, &.{0}));
    try testing.expectEqual(@as(usize, 0), empty_output.position());

    const empty_prompt = try program.bind(.{});
    defer empty_prompt.deinit();
    var one_token = [_]usize{999};
    try testing.expectError(error.InvalidPrefillLength, empty_prompt.generateArgmaxInto(&one_token, &.{}));
    try testing.expectEqual(@as(usize, 0), empty_prompt.position());

    const short_program = try model.compile(.{ .context_len = 1, .batch = 1 });
    defer short_program.deinit();
    const short_session = try short_program.bind(.{});
    defer short_session.deinit();
    var too_many = [_]usize{ 999, 999 };
    try testing.expectError(error.SequenceTooLong, short_session.generateArgmaxInto(&too_many, &.{0}));
    try testing.expectEqual(@as(usize, 0), short_session.position());
}

test "llm facade samples tokens into caller-owned output" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Model = LlamaModel(f32, cfg);

    const model = try Model.init(testing.allocator);
    defer model.deinit();

    const program = try model.compile(.{ .context_len = 4, .batch = 1 });
    defer program.deinit();

    const expected = try program.bind(.{});
    defer expected.deinit();
    const expected_first = try expected.stepSample(.{ .token = 0 }, .{ .top_k = 1, .seed = 123, .temperature = 1.0 });
    const expected_second = try expected.stepSample(.{ .token = expected_first.token }, .{ .top_k = 1, .seed = 124, .temperature = 1.0 });

    const generated_session = try program.bind(.{});
    defer generated_session.deinit();
    var output_tokens = [_]usize{ 999, 999 };
    const generated = try generated_session.generateSampleInto(&output_tokens, &.{0}, .{ .top_k = 1, .seed = 123, .temperature = 1.0 });
    try testing.expectEqual(@as(usize, 2), generated.tokens_generated);
    try testing.expectEqual(expected_first.token, output_tokens[0]);
    try testing.expectEqual(expected_second.token, output_tokens[1]);
    try testing.expectEqual(expected_second.token, generated.last_token);
    try testing.expectEqual(expected_second.logit, generated.last_logit);
    try testing.expectEqual(@as(usize, 2), generated_session.position());

    const invalid_step = try program.bind(.{});
    defer invalid_step.deinit();
    try testing.expectError(error.InvalidArgument, invalid_step.stepSample(.{ .token = 0 }, .{ .top_k = 0 }));
    try testing.expectEqual(@as(usize, 0), invalid_step.position());

    const invalid_generate = try program.bind(.{});
    defer invalid_generate.deinit();
    var one_token = [_]usize{999};
    try testing.expectError(error.InvalidArgument, invalid_generate.generateSampleInto(&one_token, &.{0}, .{ .top_k = 1, .temperature = 0 }));
    try testing.expectEqual(@as(usize, 0), invalid_generate.position());

    const short_program = try model.compile(.{ .context_len = 1, .batch = 1 });
    defer short_program.deinit();
    const short_session = try short_program.bind(.{});
    defer short_session.deinit();
    var too_many = [_]usize{ 999, 999 };
    try testing.expectError(error.SequenceTooLong, short_session.generateSampleInto(&too_many, &.{0}, .{ .top_k = 1 }));
    try testing.expectEqual(@as(usize, 0), short_session.position());
}

test "llm session steps into caller-provided logits buffer" {
    const cfg = LlamaConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 1,
        .n_kv_heads = 1,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 4,
    };
    const Session = LlamaSession(f32, cfg);

    const session = try Session.init(testing.allocator);
    defer session.deinit();

    var too_small = [_]f32{0} ** (cfg.vocab_size - 1);
    try testing.expectError(error.OutputBufferTooSmall, session.stepInto(&too_small, .{ .token = 0 }));
    try testing.expectEqual(@as(usize, 0), session.position());

    var output = [_]f32{-999} ** (cfg.vocab_size + 2);
    const logits = try session.stepInto(&output, .{ .token = 0 });
    try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
    try testing.expectEqual(@as(usize, 1), session.position());
    try testing.expect(logits.ptr == output[0..cfg.vocab_size].ptr);
    try testing.expectEqual(@as(f32, -999), output[cfg.vocab_size]);

    session.reset();
    const expected = try session.step(.{ .token = 0 });
    try testing.expectEqualSlices(f32, expected, output[0..cfg.vocab_size]);

    session.reset();
    @memset(&output, -123);
    const prefill_logits = try session.prefillInto(&output, &.{ 0, 1 });
    try testing.expectEqual(@as(usize, cfg.vocab_size), prefill_logits.len);
    try testing.expectEqual(@as(usize, 2), session.position());
    try testing.expectEqual(@as(f32, -123), output[cfg.vocab_size]);
}
