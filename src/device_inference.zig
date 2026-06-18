//! Device-accelerated inference wrapper.
//!
//! Compiles a ComputeGraph into a backend Program, binds persistent per-step
//! I/O as a Session, then executes each token/window through StepParams.
//! Model-agnostic — works with any graph-shaped decoder path.
//!
//! ```
//! var metal = try MetalBackend.init();
//! const binding = try model_owned_plan.deviceBindingForCompiledWindow();
//! var device = try DeviceInference(f32).init(.{
//!     .graph = binding.graph,
//!     .be = metal.backend(),
//!     .alloc = alloc,
//!     .input_tensors = binding.inputs(),
//!     .output_tensor = binding.output_tensor,
//!     .output_host_buf = logits_buf.ptr,
//!     .output_len = vocab_size,
//!     .expected_runtime_patch_shape = binding.expected_runtime_patch_shape,
//! });
//! defer device.deinit();
//!
//! // Caller patches tensor data each step, then executes with StepParams.
//! try device.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(pos, 1) });
//! ```

const std = @import("std");
const backend_mod = @import("backend.zig");
const program_mod = @import("backend/program.zig");
const profile = @import("profile.zig");
const tensor_program_ir = @import("tensor_program_ir.zig");
const Op = @import("op.zig").Op;

const QuantizedWeight = @import("quant.zig").QuantizedWeight;

fn hashBindingInt(comptime T: type, hasher: *std.hash.Wyhash, value: T) void {
    var v = value;
    hasher.update(std.mem.asBytes(&v));
}

fn hashProgramIOList(hasher: *std.hash.Wyhash, section: u8, ios: []const backend_mod.ProgramIO) void {
    hashBindingInt(u8, hasher, section);
    hashBindingInt(u64, hasher, @intCast(ios.len));
    for (ios, 0..) |io, i| {
        hashBindingInt(u64, hasher, @intCast(i));
        hashBindingInt(u16, hasher, io.buf_idx);
        hashBindingInt(u32, hasher, io.offset);
        hashBindingInt(u32, hasher, io.size);
        if (io.resource) |resource| {
            hashBindingInt(u8, hasher, 2);
            hashBindingInt(u32, hasher, @intFromEnum(resource.placement));
            hashBindingInt(u64, hasher, std.math.cast(u64, resource.handle) orelse std.math.maxInt(u64));
            hashBindingInt(u32, hasher, resource.byte_offset);
            hashBindingInt(u32, hasher, resource.byte_len);
            hashBindingInt(u32, hasher, resource.access.toFlags());
        } else {
            hashBindingInt(u8, hasher, 1);
        }
    }
}

fn sessionBindingShapeHash(persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("zgml-session-binding-shape-v1");
    hashProgramIOList(&hasher, 1, persistent_inputs);
    hashProgramIOList(&hasher, 2, step_inputs);
    hashProgramIOList(&hasher, 3, step_outputs);
    return hasher.final();
}

fn programBindingRequirementHash(persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("zgml-program-binding-requirements-v1");
    hashProgramIOList(&hasher, 1, persistent_inputs);
    hashProgramIOList(&hasher, 2, step_inputs);
    hashProgramIOList(&hasher, 3, step_outputs);
    return hasher.final();
}

fn freeOpPayloads(alloc: std.mem.Allocator, ops: []const backend_mod.DeviceOp) void {
    for (ops) |op| switch (op) {
        .fused_elementwise => |fe| if (fe.steps.len > 0) alloc.free(fe.steps),
        else => {},
    };
}

pub fn DeviceInference(comptime T: type) type {
    const Tensor = @import("tensor.zig").Tensor(T);
    const ComputeGraph = @import("graph.zig").ComputeGraph(T);

    return struct {
        const Self = @This();
        const TensorProgramIr = tensor_program_ir.TensorProgramIr(T);

        program: Program,
        session: Session,

        pub const StepParams = struct {
            window: backend_mod.RuntimeWindow,
            download_outputs: bool = true,
            dynamic_io: bool = false,
        };

        pub const ProgramInspection = struct {
            backend: backend_mod.Device,
            execution_supported: bool,
            external_resources_supported: bool,
            ir: ProgramIrInspection,
            buffer_count: usize,
            buffer_element_count: usize,
            buffer_byte_len: usize,
            initial_upload_count: usize,
            qweight_count: usize,
            op_count: usize,
            runtime_patch_shape: backend_mod.RuntimePatchShape,
            runtime_patch_envelope: program_mod.RuntimePatchEnvelope,
            command_shape: program_mod.ProgramCommandStreamShape,
            execution_plan: backend_mod.ExecutionPlanInspection,
            binding_requirement_hash: u64,
            persistent_requirement_count: usize,
            step_input_requirement_count: usize,
            step_output_requirement_count: usize,
        };

        pub const ProgramIrInspection = tensor_program_ir.Inspection;

        pub const BindingStorage = enum(u8) {
            none,
            host,
            external_resource,
        };

        pub const SessionInspection = struct {
            backend: backend_mod.Device,
            output_storage: BindingStorage,
            persistent_binding_count: usize,
            step_input_count: usize,
            step_output_count: usize,
            host_binding_count: usize,
            resource_binding_count: usize,
            binding_shape_hash: u64,
        };

        pub const CompileOptions = struct {
            graph: *ComputeGraph,
            be: backend_mod.Backend,
            alloc: std.mem.Allocator,
            input_tensors: []const *const Tensor = &.{},
            persistent_tensors: []const *const Tensor = &.{},
            output_tensors: []const *const Tensor = &.{},
            quant_weights: []const QuantizedWeight(T) = &.{},
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize) = &empty_quant_map,
            expected_runtime_patch_shape: ?backend_mod.RuntimePatchShape = null,
        };

        pub const BindOptions = struct {
            input_tensors: []const *const Tensor,
            output_tensor: *const Tensor,
            output_host_buf: [*]T,
            output_len: usize,
            persistent_tensors: []const *const Tensor = &.{},
            input_bindings: []const TensorBinding = &.{},
            persistent_bindings: []const TensorBinding = &.{},
            output_binding: ?TensorBinding = null,
            quant_weights: []const QuantizedWeight(T) = &.{},
        };

        pub const TensorBinding = struct {
            tensor: *const Tensor,
            host_ptr: [*]T = undefined,
            resource: ?backend_mod.ProgramIO.ExternalResource = null,
            element_count: usize,

            pub fn host(tensor: *const Tensor, host_ptr: [*]T, element_count: usize) TensorBinding {
                return .{ .tensor = tensor, .host_ptr = host_ptr, .element_count = element_count };
            }

            pub fn externalResource(tensor: *const Tensor, resource: backend_mod.ProgramIO.ExternalResource, element_count: usize) TensorBinding {
                return .{ .tensor = tensor, .resource = resource, .element_count = element_count };
            }
        };

        const QWeightShape = struct {
            rows: usize,
            cols: usize,
            block_size: usize,
        };

        const ProgramBindingManifest = struct {
            persistent_count: usize,
            step_input_count: usize,
            step_output_count: usize,
            hash: u64,

            fn init(
                alloc: std.mem.Allocator,
                buffers: *const BufferMap,
                persistent_tensors: []const *const Tensor,
                input_tensors: []const *const Tensor,
                output_tensors: []const *const Tensor,
            ) !ProgramBindingManifest {
                const persistent = try alloc.alloc(backend_mod.ProgramIO, persistent_tensors.len);
                defer alloc.free(persistent);
                for (persistent_tensors, 0..) |t, i| {
                    persistent[i] = try tensorIO(buffers, t, t.data.ptr, t.data.len);
                }

                const inputs = try alloc.alloc(backend_mod.ProgramIO, input_tensors.len);
                defer alloc.free(inputs);
                for (input_tensors, 0..) |t, i| {
                    inputs[i] = try tensorExactIO(buffers, t, t.data.ptr, t.data.len);
                }

                const outputs = try alloc.alloc(backend_mod.ProgramIO, output_tensors.len);
                defer alloc.free(outputs);
                for (output_tensors, 0..) |t, i| {
                    outputs[i] = try tensorExactIO(buffers, t, t.data.ptr, t.nElems());
                }

                return .{
                    .persistent_count = persistent.len,
                    .step_input_count = inputs.len,
                    .step_output_count = outputs.len,
                    .hash = programBindingRequirementHash(persistent, inputs, outputs),
                };
            }
        };

        pub const Program = struct {
            alloc: std.mem.Allocator,
            be: backend_mod.Backend,
            handle: backend_mod.Backend.CompiledHandle,
            buffers: BufferMap,
            qweight_shapes: []QWeightShape,
            ir_inspection: ProgramIrInspection,
            op_count: usize,
            runtime_patch_shape: backend_mod.RuntimePatchShape,
            runtime_patch_envelope: program_mod.RuntimePatchEnvelope,
            command_shape: program_mod.ProgramCommandStreamShape,
            binding_manifest: ProgramBindingManifest,

            pub fn compile(opts: CompileOptions) !Program {
                const alloc = opts.alloc;
                var lowered = try LoweredDeviceProgram.init(.{
                    .graph = opts.graph,
                    .alloc = alloc,
                    .capabilities = opts.be.capabilities,
                    .input_tensors = opts.input_tensors,
                    .persistent_tensors = opts.persistent_tensors,
                    .output_tensors = opts.output_tensors,
                    .quant_weights = opts.quant_weights,
                    .quant_map = opts.quant_map,
                });
                defer lowered.deinit();

                const device_program = lowered.deviceProgram();
                const binding_manifest = try ProgramBindingManifest.init(alloc, &lowered.buffers, opts.persistent_tensors, opts.input_tensors, opts.output_tensors);
                const runtime_patch_envelope = try program_mod.RuntimePatchEnvelope.initProgram(device_program);
                const op_count = lowered.owned_ops.len;

                const compiled = try compileHandle(opts.be, device_program, opts.expected_runtime_patch_shape);
                errdefer opts.be.freeProgram(compiled.handle);
                return .{
                    .alloc = alloc,
                    .be = opts.be,
                    .handle = compiled.handle,
                    .buffers = lowered.takeBuffers(),
                    .qweight_shapes = lowered.takeQWeightShapes(),
                    .ir_inspection = lowered.ir_inspection,
                    .op_count = op_count,
                    .runtime_patch_shape = compiled.runtime_patch_shape,
                    .runtime_patch_envelope = runtime_patch_envelope,
                    .command_shape = compiled.command_shape,
                    .binding_manifest = binding_manifest,
                };
            }

            const CompiledHandleEvidence = struct {
                handle: backend_mod.Backend.CompiledHandle,
                runtime_patch_shape: backend_mod.RuntimePatchShape,
                command_shape: program_mod.ProgramCommandStreamShape,
            };

            fn compileHandle(be: backend_mod.Backend, program: backend_mod.DeviceProgram, expected_runtime_patch_shape: ?backend_mod.RuntimePatchShape) !CompiledHandleEvidence {
                const handle = be.compileProgram(program) orelse {
                    if (!be.supportsProgram(program)) return error.UnsupportedDeviceOp;
                    return error.CompileFailed;
                };
                errdefer be.freeProgram(handle);
                var rt = profile.RuntimeProfile{};
                be.addRuntimeProfileTo(handle, &rt);
                if (program.ops.len > 0 and rt.program_command_shape.command_count == 0) return error.CommandShapeMissing;
                if (expected_runtime_patch_shape) |expected| {
                    if (!expected.matches(rt.runtime_patch_shape)) return error.RuntimePatchShapeMismatch;
                }
                return .{
                    .handle = handle,
                    .runtime_patch_shape = rt.runtime_patch_shape,
                    .command_shape = rt.program_command_shape,
                };
            }

            pub fn bind(self: *Program, opts: BindOptions) !Session {
                const runtime_handle = self.be.bindProgram(self.handle) orelse return error.BindFailed;
                var runtime_owned = true;
                errdefer if (runtime_owned) self.be.freeBindings(self.handle, runtime_handle);

                var session = try Session.bind(self.alloc, self.be, self.handle, runtime_handle, &self.buffers, opts.persistent_tensors, opts.persistent_bindings, opts.input_tensors, opts.input_bindings, opts.output_tensor, opts.output_host_buf, opts.output_len, opts.output_binding);
                runtime_owned = false;
                errdefer session.deinit();
                try self.configureBindings(&session);
                try self.uploadQWeights(&session, opts.quant_weights);
                try session.uploadPersistent(self);
                return session;
            }

            pub fn configureBindings(self: *Program, session: *Session) !void {
                try session.bindings.validatePersistent(self);
                try session.bindings.validateStep(self, true);
                if (!session.bindings.configure(self, session.runtime_handle)) return error.UnsupportedResourceBinding;
            }

            fn validatePersistentIO(self: *const Program, persistent_inputs: []const backend_mod.ProgramIO) !void {
                if (!backend_mod.BufferBounds.programIOListsValid(self.buffers.buf_sizes.items, persistent_inputs, &.{})) return error.InvalidProgramIO;
                if (!self.be.capabilities.external_resources) {
                    if (!backend_mod.BufferBounds.programIOListHostOnly(persistent_inputs)) return error.UnsupportedResourceBinding;
                } else {
                    if (!backend_mod.BufferBounds.programIOListResourcesReadable(persistent_inputs, self.be.device_type)) return error.UnsupportedResourceBinding;
                }
            }

            fn validateStepIO(self: *const Program, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) !void {
                if (!backend_mod.BufferBounds.programIOListsValid(self.buffers.buf_sizes.items, step_inputs, step_outputs)) return error.InvalidProgramIO;
                if (!self.be.capabilities.external_resources) {
                    if (!backend_mod.BufferBounds.programIOListsHostOnly(step_inputs, step_outputs)) return error.UnsupportedResourceBinding;
                } else {
                    if (!backend_mod.BufferBounds.programIOListsResourcesUsable(step_inputs, step_outputs, self.be.device_type)) return error.UnsupportedResourceBinding;
                }
            }

            fn uploadQWeights(self: *Program, session: *Session, qweights: []const QuantizedWeight(T)) !void {
                if (qweights.len == 0) return;
                if (!self.be.capabilities.runtime_qweights) return error.UnsupportedDeviceOp;
                if (qweights.len != self.qweight_shapes.len) return error.ShapeMismatch;
                const uploads = try self.alloc.alloc(backend_mod.QuantizedWeightUpload, qweights.len);
                defer self.alloc.free(uploads);
                for (qweights, self.qweight_shapes, 0..) |qw, expected, i| {
                    if (qw.rows != expected.rows or qw.cols != expected.cols or qw.block_size != expected.block_size) return error.ShapeMismatch;
                    uploads[i] = .{ .data = qw.data, .scales = qw.scales, .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
                }
                if (!self.be.uploadQWeights(self.handle, session.runtime_handle, uploads)) return error.BindFailed;
            }

            pub fn executeStep(self: *Program, session: *Session, params: StepParams) !void {
                try session.executeStep(self, params);
            }

            pub fn uploadPersistentInputs(self: *Program, session: *Session) !void {
                try session.uploadPersistent(self);
            }

            pub fn uploadPersistentInputRange(self: *Program, session: *Session, first: usize, len: usize) !void {
                try session.uploadPersistentRange(self, first, len);
            }

            pub fn inspect(self: *const Program) ProgramInspection {
                return .{
                    .backend = self.be.device_type,
                    .execution_supported = self.be.capabilities.executes_programs,
                    .external_resources_supported = self.be.capabilities.external_resources,
                    .ir = self.ir_inspection,
                    .buffer_count = self.buffers.buf_sizes.items.len,
                    .buffer_element_count = self.totalBufferElementCount(),
                    .buffer_byte_len = self.totalBufferByteLen(),
                    .initial_upload_count = self.buffers.uploads.items.len,
                    .qweight_count = self.qweight_shapes.len,
                    .op_count = self.op_count,
                    .runtime_patch_shape = self.runtime_patch_shape,
                    .runtime_patch_envelope = self.runtime_patch_envelope,
                    .command_shape = self.command_shape,
                    .execution_plan = self.be.inspectExecutionPlan(self.handle),
                    .binding_requirement_hash = self.binding_manifest.hash,
                    .persistent_requirement_count = self.binding_manifest.persistent_count,
                    .step_input_requirement_count = self.binding_manifest.step_input_count,
                    .step_output_requirement_count = self.binding_manifest.step_output_count,
                };
            }

            fn totalBufferElementCount(self: *const Program) usize {
                var elements: usize = 0;
                for (self.buffers.buf_sizes.items) |len| {
                    elements = std.math.add(usize, elements, len) catch return std.math.maxInt(usize);
                }
                return elements;
            }

            fn totalBufferByteLen(self: *const Program) usize {
                return std.math.mul(usize, self.totalBufferElementCount(), @sizeOf(T)) catch std.math.maxInt(usize);
            }

            pub fn tensorBindingIO(self: *const Program, tensor: *const Tensor, host_ptr: [*]T, element_count: usize) !backend_mod.ProgramIO {
                return tensorIO(&self.buffers, tensor, host_ptr, element_count);
            }

            pub fn tensorResourceBindingIO(self: *const Program, tensor: *const Tensor, resource: backend_mod.ProgramIO.ExternalResource, element_count: usize) !backend_mod.ProgramIO {
                return tensorResourceIO(&self.buffers, tensor, resource, element_count);
            }

            pub fn resetRuntimeProfile(self: *const Program) void {
                self.be.resetRuntimeProfile(self.handle);
            }

            pub fn addRuntimeProfileTo(self: *const Program, dest: *profile.RuntimeProfile) void {
                self.be.addRuntimeProfileTo(self.handle, dest);
            }

            pub fn deinit(self: *Program) void {
                self.be.freeProgram(self.handle);
                self.alloc.free(self.qweight_shapes);
                self.buffers.deinit();
                self.* = undefined;
            }
        };

        const DeviceOpLowerer = struct {
            alloc: std.mem.Allocator,
            capabilities: backend_mod.Capabilities,
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize),
            buffers: *BufferMap,
            ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp),

            fn lowerIr(self: *DeviceOpLowerer, ir: TensorProgramIr) !void {
                for (ir.steps) |step| {
                    const op = ir.ops[step.op_index];
                    switch (op.info.kind) {
                        .fusion => switch (op.info.fusion_kind.?) {
                            .elementwise_chain => try self.lowerElementwiseChainIrOp(ir, op),
                            .log_softmax => try self.lowerLogSoftmaxIrOp(ir, op),
                            .layer_norm => try self.lowerLayerNormIrOp(ir, op),
                            .conv2d => try self.lowerConv2dIrOp(ir, op),
                            .max_pool2d => try self.lowerMaxPool2dIrOp(ir, op),
                            .avg_pool2d => try self.lowerAvgPool2dIrOp(ir, op),
                            else => return error.UnsupportedDeviceOp,
                        },
                        .node => try self.lowerNodeOp(ir, op),
                    }
                }
            }

            fn lowerConv2dIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const input = ir.opInputTensor(ir_op, 0) orelse return error.UnsupportedDeviceOp;
                const weight = ir.opInputTensor(ir_op, 1) orelse return error.UnsupportedDeviceOp;
                const bias = ir.opInputTensor(ir_op, 2);
                const output = ir.opOutputTensor(ir_op);
                if (!input.isDenseLayout() or !weight.isDenseLayout() or !output.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (bias) |b| if (!b.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (input.n_dims != 4 or weight.n_dims != 4 or output.n_dims != 4) return error.UnsupportedDeviceOp;
                const kernel_w = weight.ne[0];
                const kernel_h = weight.ne[1];
                const in_channels = weight.ne[2];
                const out_channels = weight.ne[3];
                if (input.ne[2] != in_channels or output.ne[2] != out_channels or input.ne[3] != output.ne[3]) return error.UnsupportedDeviceOp;
                if (output.ne[0] + kernel_w != input.ne[0] + 1 or output.ne[1] + kernel_h != input.ne[1] + 1) return error.UnsupportedDeviceOp;
                try self.ops.append(self.alloc, .{ .conv2d = .{
                    .dst = self.buffers.idx(output),
                    .src = self.buffers.idx(input),
                    .weight = self.buffers.idx(weight),
                    .bias = if (bias) |b| self.buffers.idx(b) else std.math.maxInt(u16),
                    .out_w = @intCast(output.ne[0]),
                    .out_h = @intCast(output.ne[1]),
                    .in_w = @intCast(input.ne[0]),
                    .in_h = @intCast(input.ne[1]),
                    .in_channels = @intCast(in_channels),
                    .out_channels = @intCast(out_channels),
                    .kernel_w = @intCast(kernel_w),
                    .kernel_h = @intCast(kernel_h),
                    .batch = @intCast(output.ne[3]),
                    .src_offset = @intCast(self.buffers.offset(input)),
                    .weight_offset = @intCast(self.buffers.offset(weight)),
                    .bias_offset = if (bias) |b| @intCast(self.buffers.offset(b)) else 0,
                    .dst_offset = @intCast(self.buffers.offset(output)),
                    .relu = ir_op.info.op == .relu,
                } });
            }

            fn lowerMaxPool2dIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const input = ir.opInputTensor(ir_op, 0) orelse return error.UnsupportedDeviceOp;
                const output = ir.opOutputTensor(ir_op);
                if (!input.isDenseLayout() or !output.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (input.n_dims != 4 or output.n_dims != 4) return error.UnsupportedDeviceOp;
                if (input.ne[0] != output.ne[0] * 2 or input.ne[1] != output.ne[1] * 2) return error.UnsupportedDeviceOp;
                if (input.ne[2] != output.ne[2] or input.ne[3] != output.ne[3]) return error.UnsupportedDeviceOp;
                try self.ops.append(self.alloc, .{ .max_pool2d = .{
                    .dst = self.buffers.idx(output),
                    .src = self.buffers.idx(input),
                    .out_w = @intCast(output.ne[0]),
                    .out_h = @intCast(output.ne[1]),
                    .channels = @intCast(output.ne[2]),
                    .batch = @intCast(output.ne[3]),
                    .src_w = @intCast(input.ne[0]),
                    .src_h = @intCast(input.ne[1]),
                    .src_offset = @intCast(self.buffers.offset(input)),
                    .dst_offset = @intCast(self.buffers.offset(output)),
                } });
            }

            fn lowerAvgPool2dIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const input = ir.opInputTensor(ir_op, 0) orelse return error.UnsupportedDeviceOp;
                const output = ir.opOutputTensor(ir_op);
                if (!input.isDenseLayout() or !output.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (input.n_dims != 4 or output.n_dims != 4) return error.UnsupportedDeviceOp;
                if (input.ne[0] != output.ne[0] * 2 or input.ne[1] != output.ne[1] * 2) return error.UnsupportedDeviceOp;
                if (input.ne[2] != output.ne[2] or input.ne[3] != output.ne[3]) return error.UnsupportedDeviceOp;
                try self.ops.append(self.alloc, .{ .avg_pool2d = .{
                    .dst = self.buffers.idx(output),
                    .src = self.buffers.idx(input),
                    .out_w = @intCast(output.ne[0]),
                    .out_h = @intCast(output.ne[1]),
                    .channels = @intCast(output.ne[2]),
                    .batch = @intCast(output.ne[3]),
                    .src_w = @intCast(input.ne[0]),
                    .src_h = @intCast(input.ne[1]),
                    .src_offset = @intCast(self.buffers.offset(input)),
                    .dst_offset = @intCast(self.buffers.offset(output)),
                } });
            }

            fn lowerLayerNormIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const input = ir.opInputTensor(ir_op, 0).?;
                const output = ir.opOutputTensor(ir_op);
                const attrs = switch (ir_op.attrs) {
                    .layer_norm => |ln| ln,
                    else => return error.UnsupportedDeviceOp,
                };
                if (!input.isDenseLayout() or !output.isDenseLayout()) return error.UnsupportedDeviceOp;
                try self.ops.append(self.alloc, .{ .layernorm = .{
                    .dst = self.buffers.idx(output),
                    .src = self.buffers.idx(input),
                    .rows = @intCast(input.ne[1]),
                    .cols = @intCast(input.ne[0]),
                    .eps = attrs.eps,
                    .src_offset = @intCast(self.buffers.offset(input)),
                    .dst_offset = @intCast(self.buffers.offset(output)),
                } });
            }

            fn lowerElementwiseChainIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const chain = switch (ir_op.attrs) {
                    .elementwise_chain => |chain| chain,
                    else => return error.UnsupportedDeviceOp,
                };
                const chain_steps = ir.elementwise_steps[chain.step_start..][0..chain.step_count];
                if (chain_steps.len == 0) return;
                const input = ir.valueTensor(chain.input);
                const output = ir.opOutputTensor(ir_op);
                const can_fuse = self.capabilities.fused_elementwise and !elementwiseChainNeedsBroadcastIr(ir, chain) and blk: {
                    const max = self.capabilities.max_fused_elementwise_steps orelse break :blk true;
                    break :blk chain_steps.len <= @as(usize, @intCast(max));
                };
                if (can_fuse) {
                    if (!input.isDenseLayout() or !output.isDenseLayout()) return error.UnsupportedDeviceOp;
                    const ew_steps = try self.alloc.alloc(backend_mod.FusedEwStep, chain_steps.len);
                    errdefer self.alloc.free(ew_steps);
                    for (chain_steps, 0..) |step, k| {
                        var secondary_buf: u16 = 0;
                        var secondary_offset: u32 = 0;
                        if (step.op.isBinary()) {
                            const secondary = ir.valueTensor(step.secondary.?);
                            if (!secondary.isDenseLayout()) return error.UnsupportedDeviceOp;
                            secondary_buf = self.buffers.idx(secondary);
                            secondary_offset = @intCast(self.buffers.offset(secondary));
                        }
                        ew_steps[k] = .{
                            .op = step.op,
                            .is_swapped = step.is_swapped,
                            .secondary_buf = secondary_buf,
                            .secondary_offset = secondary_offset,
                        };
                    }
                    try self.ops.append(self.alloc, .{ .fused_elementwise = .{
                        .steps = ew_steps,
                        .n = @intCast(output.nElems()),
                        .dst = self.buffers.idx(output),
                        .src = self.buffers.idx(input),
                        .dst_offset = @intCast(self.buffers.offset(output)),
                        .src_offset = @intCast(self.buffers.offset(input)),
                    } });
                    return;
                }

                var current = input;
                for (chain_steps) |step| {
                    const dst = ir.valueTensor(step.output);
                    if (step.op.isBinary()) {
                        const secondary = ir.valueTensor(step.secondary.?);
                        if (step.is_swapped) {
                            try self.appendElementwiseIrOps(step.op, dst, secondary, current);
                        } else {
                            try self.appendElementwiseIrOps(step.op, dst, current, secondary);
                        }
                    } else {
                        try self.appendElementwiseIrOps(step.op, dst, current, null);
                    }
                    current = dst;
                }
            }

            fn lowerLogSoftmaxIrOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const log_softmax = switch (ir_op.attrs) {
                    .log_softmax => |log_softmax| log_softmax,
                    else => return error.UnsupportedDeviceOp,
                };
                const inputs = ir_op.inputs(ir);
                if (inputs.len == 1) {
                    const src = ir.valueTensor(inputs[0]);
                    const dst = ir.opOutputTensor(ir_op);
                    if (self.capabilities.logsoftmax and isCanonicalRowSoftmaxInput(dst, src)) {
                        try self.ops.append(self.alloc, logsoftmaxDeviceOp(self.buffers, dst, src, self.buffers.idx(dst), self.buffers.idx(src)));
                        return;
                    }
                }
                for (ir.ops[log_softmax.op_start..][0..log_softmax.op_count]) |sub_op| {
                    try self.lowerNodeOp(ir, sub_op);
                }
            }

            fn lowerNodeOp(self: *DeviceOpLowerer, ir: TensorProgramIr, ir_op: TensorProgramIr.IrOp) !void {
                const op = ir_op.info.op;
                if (!ir_op.info.isValue()) return;

                const dst = ir.opOutputTensor(ir_op);
                const src0 = ir.opInputTensor(ir_op, 0);
                const src1 = ir.opInputTensor(ir_op, 1);
                const src2 = ir.opInputTensor(ir_op, 2);
                const src3 = ir.opInputTensor(ir_op, 3);

                if (op == .matmul) {
                    try self.appendMatmulIrOp(dst, src0.?, src1.?);
                    return;
                }

                const dst_idx = self.buffers.idx(dst);
                const src0_idx = if (src0) |s| self.buffers.idx(s) else dst_idx;
                const src1_idx = if (src1) |s| self.buffers.idx(s) else src0_idx;

                switch (op) {
                    .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu, .sqr => {
                        try self.appendElementwiseIrOps(op, dst, src0, src1);
                    },
                    .sum, .max => {
                        const src = src0.?;
                        if (!src.isDenseLayout() or !dst.isDenseLayout()) return error.UnsupportedDeviceOp;
                        try self.ops.append(self.alloc, reduceDeviceOp(self.buffers, dst, src, dst_idx, src0_idx));
                    },
                    .repeat => {
                        const src = src0.?;
                        try self.ops.append(self.alloc, repeatDeviceOp(self.buffers, dst, src, dst_idx, src0_idx));
                    },
                    .gather_rows => {
                        const src = src0.?;
                        const indices = src1.?;
                        if (!src.isMatrix() or !indices.isVector() or !dst.isDenseLayout() or !src.isDenseLayout() or !indices.isDenseLayout()) return error.UnsupportedDeviceOp;
                        try self.ops.append(self.alloc, gatherRowsDeviceOp(self.buffers, dst, src, indices, dst_idx, src0_idx, src1_idx));
                    },
                    .softmax => {
                        const src = src0.?;
                        if (!isCanonicalRowSoftmaxInput(dst, src)) return error.UnsupportedDeviceOp;
                        try self.ops.append(self.alloc, softmaxDeviceOp(self.buffers, dst, src, dst_idx, src0_idx));
                    },
                    .rmsnorm => {
                        const src = src0.?;
                        if (!src.isDenseLayout() or !dst.isDenseLayout()) return error.UnsupportedDeviceOp;
                        try self.ops.append(self.alloc, rmsnormDeviceOp(self.buffers, dst, src, dst_idx, src0_idx));
                    },
                    .slice_assign => {
                        const src = src0.?;
                        const write_dst = src1 orelse dst;
                        const rows = write_dst.ne[0];
                        const cols = if (src.n_dims >= 2) src.ne[1] else 1;
                        const base = self.buffers.offset(write_dst);
                        const dst_offset = base + dst.storage_offset * write_dst.strides[1];
                        try self.ops.append(self.alloc, sliceAssignDeviceOp(self.buffers, src, write_dst, src0_idx, rows, cols, base, dst_offset, write_dst.strides[1]));
                    },
                    .slice_assign_rows => {
                        const src = src0.?;
                        const write_dst = src1 orelse dst;
                        const base = self.buffers.offset(write_dst);
                        const dst_offset = base + dst.storage_offset * write_dst.strides[0];
                        try self.ops.append(self.alloc, sliceAssignDeviceOp(self.buffers, src, write_dst, src0_idx, src.ne[0], src.ne[1], base, dst_offset, 0));
                    },
                    .rope => {
                        const src = src0.?;
                        const cs = src1.?;
                        try self.ops.append(self.alloc, ropeDeviceOp(self.buffers, dst, src, cs, dst_idx, src0_idx, src1_idx));
                    },
                    .attention => {
                        const q = src0.?;
                        const k = src1.?;
                        const v = src2.?;
                        try self.ops.append(self.alloc, attentionDeviceOp(self.buffers, dst, q, k, v, src3, dst_idx, src0_idx, src1_idx));
                    },
                    // Structural ops are filtered by TensorProgramIr.StepInfo.
                    // Any remaining unhandled op is outside this lowerer's executable subset.
                    else => return error.UnsupportedDeviceOp,
                }
            }

            fn appendMatmulIrOp(self: *DeviceOpLowerer, dst: *Tensor, s0: *Tensor, s1: *Tensor) !void {
                const flags = dst.matmul_flags;
                const M = if (flags.trans0) s0.ne[0] else s0.ne[1];
                const N = if (flags.trans1) s1.ne[1] else s1.ne[0];
                const K = if (flags.trans0) s0.ne[1] else s0.ne[0];

                if (self.quant_map.get(dst)) |qi| {
                    const input_tensor = if (s1.isParam()) s0 else s1;
                    if (!input_tensor.isDenseLayout() or !dst.isDenseLayout()) return error.UnsupportedDeviceOp;
                    const input_is_src0 = @intFromPtr(input_tensor) == @intFromPtr(s0);
                    const input_col_stride = if (input_is_src0)
                        (if (flags.trans0) s0.strides[1] else s0.strides[0])
                    else
                        (if (flags.trans1) s1.strides[1] else s1.strides[0]);
                    if (input_col_stride != 1 or dst.strides[0] != 1) return error.UnsupportedDeviceOp;
                    const input_row_stride = if (input_is_src0)
                        (if (flags.trans0) s0.strides[0] else if (s0.n_dims >= 2) s0.strides[1] else K)
                    else
                        (if (flags.trans1) s1.strides[0] else if (s1.n_dims >= 2) s1.strides[1] else K);
                    try self.ops.append(self.alloc, .{ .qmatmul = .{
                        .dst = self.buffers.idx(dst),
                        .input = self.buffers.idx(input_tensor),
                        .weight_idx = @intCast(qi),
                        .M = @intCast(M),
                        .N = @intCast(N),
                        .K = @intCast(K),
                        .input_offset = @intCast(self.buffers.offset(input_tensor)),
                        .input_row_stride = @intCast(input_row_stride),
                        .dst_offset = @intCast(self.buffers.offset(dst)),
                        .dst_row_stride = @intCast(if (dst.n_dims >= 2) dst.strides[1] else N),
                    } });
                    return;
                }

                try self.ops.append(self.alloc, .{ .matmul = .{
                    .dst = self.buffers.idx(dst),
                    .a = self.buffers.idx(s0),
                    .b = self.buffers.idx(s1),
                    .geom = .{
                        .M = M,
                        .N = N,
                        .K = K,
                        .a_row_stride = if (flags.trans0) s0.strides[0] else s0.strides[1],
                        .a_col_stride = if (flags.trans0) s0.strides[1] else s0.strides[0],
                        .b_row_stride = if (flags.trans1) s1.strides[0] else s1.strides[1],
                        .b_col_stride = if (flags.trans1) s1.strides[1] else s1.strides[0],
                        .a_offset = self.buffers.offset(s0),
                        .b_offset = self.buffers.offset(s1),
                        .dst_offset = self.buffers.offset(dst),
                        .dst_row_stride = N,
                    },
                } });
            }

            fn appendElementwiseIrOps(self: *DeviceOpLowerer, op: Op, dst: *Tensor, maybe_src0: ?*Tensor, maybe_src1: ?*Tensor) !void {
                if (!dst.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (maybe_src0) |s| if (!s.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (maybe_src1) |s| if (!s.isDenseLayout()) return error.UnsupportedDeviceOp;

                const dst_idx = self.buffers.idx(dst);
                const src0_idx = if (maybe_src0) |s| self.buffers.idx(s) else dst_idx;
                const src1 = maybe_src1 orelse maybe_src0.?;
                const src1_idx = self.buffers.idx(src1);

                if (!op.isBinary()) {
                    try appendElementwiseRawOp(
                        self.ops,
                        self.alloc,
                        op,
                        dst_idx,
                        src0_idx,
                        src1_idx,
                        dst.nElems(),
                        self.buffers.offset(dst),
                        if (maybe_src0) |s| self.buffers.offset(s) else 0,
                        self.buffers.offset(src1),
                    );
                    return;
                }

                const lhs = maybe_src0.?;
                const rhs = maybe_src1.?;
                const lhs_exact = elementwiseSourceExact(dst, lhs);
                const rhs_exact = elementwiseSourceExact(dst, rhs);
                if (lhs_exact and rhs_exact) {
                    try appendElementwiseRawOp(
                        self.ops,
                        self.alloc,
                        op,
                        dst_idx,
                        src0_idx,
                        src1_idx,
                        dst.nElems(),
                        self.buffers.offset(dst),
                        self.buffers.offset(lhs),
                        self.buffers.offset(rhs),
                    );
                    return;
                }

                if (lhs_exact and rhs.canRepeatTo(dst)) {
                    try self.ops.append(self.alloc, repeatDeviceOp(self.buffers, dst, rhs, dst_idx, src1_idx));
                    try appendElementwiseRawOp(
                        self.ops,
                        self.alloc,
                        op,
                        dst_idx,
                        src0_idx,
                        dst_idx,
                        dst.nElems(),
                        self.buffers.offset(dst),
                        self.buffers.offset(lhs),
                        self.buffers.offset(dst),
                    );
                    return;
                }

                if (rhs_exact and lhs.canRepeatTo(dst)) {
                    try self.ops.append(self.alloc, repeatDeviceOp(self.buffers, dst, lhs, dst_idx, src0_idx));
                    try appendElementwiseRawOp(
                        self.ops,
                        self.alloc,
                        op,
                        dst_idx,
                        dst_idx,
                        src1_idx,
                        dst.nElems(),
                        self.buffers.offset(dst),
                        self.buffers.offset(dst),
                        self.buffers.offset(rhs),
                    );
                    return;
                }

                return error.UnsupportedDeviceOp;
            }
        };

        const LoweringOptions = struct {
            graph: *ComputeGraph,
            alloc: std.mem.Allocator,
            capabilities: backend_mod.Capabilities,
            input_tensors: []const *const Tensor,
            persistent_tensors: []const *const Tensor,
            output_tensors: []const *const Tensor,
            quant_weights: []const QuantizedWeight(T),
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize),
        };

        const LoweredDeviceProgram = struct {
            alloc: std.mem.Allocator,
            buffers: BufferMap,
            owned_ops: []backend_mod.DeviceOp,
            qweight_uploads: []backend_mod.QuantizedWeightUpload,
            qweight_shapes: []QWeightShape,
            ir_inspection: ProgramIrInspection,
            owns_buffers: bool = true,
            owns_qweight_shapes: bool = true,

            fn init(opts: LoweringOptions) !LoweredDeviceProgram {
                const alloc = opts.alloc;
                var ir = try TensorProgramIr.init(alloc, opts.graph);
                defer ir.deinit(alloc);
                const ir_inspection = ir.inspect();

                var buffers = BufferMap.init(alloc);
                errdefer buffers.deinit();
                try ir.collectBuffers(&buffers, opts.input_tensors, opts.persistent_tensors, opts.output_tensors);

                var ops: std.ArrayListUnmanaged(backend_mod.DeviceOp) = .empty;
                defer ops.deinit(alloc);
                var ops_payloads_transferred = false;
                errdefer if (!ops_payloads_transferred) freeOpPayloads(alloc, ops.items);
                var lowerer = DeviceOpLowerer{
                    .alloc = alloc,
                    .capabilities = opts.capabilities,
                    .quant_map = opts.quant_map,
                    .buffers = &buffers,
                    .ops = &ops,
                };
                try lowerer.lowerIr(ir);

                const qweight_uploads = try alloc.alloc(backend_mod.QuantizedWeightUpload, opts.quant_weights.len);
                errdefer alloc.free(qweight_uploads);
                const qweight_shapes = try alloc.alloc(QWeightShape, opts.quant_weights.len);
                errdefer alloc.free(qweight_shapes);
                for (opts.quant_weights, 0..) |qw, i| {
                    qweight_uploads[i] = .{ .data = qw.data, .scales = qw.scales, .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
                    qweight_shapes[i] = .{ .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
                }

                const owned_ops = try alloc.dupe(backend_mod.DeviceOp, ops.items);
                ops_payloads_transferred = true;
                return .{
                    .alloc = alloc,
                    .buffers = buffers,
                    .owned_ops = owned_ops,
                    .qweight_uploads = qweight_uploads,
                    .qweight_shapes = qweight_shapes,
                    .ir_inspection = ir_inspection,
                };
            }

            fn deviceProgram(self: *const LoweredDeviceProgram) backend_mod.DeviceProgram {
                return .{
                    .ops = self.owned_ops,
                    .n_buffers = @intCast(self.buffers.buf_sizes.items.len),
                    .buffer_sizes = self.buffers.buf_sizes.items,
                    .initial_uploads = self.buffers.uploads.items,
                    .qweights = self.qweight_uploads,
                };
            }

            fn takeBuffers(self: *LoweredDeviceProgram) BufferMap {
                self.owns_buffers = false;
                return self.buffers;
            }

            fn takeQWeightShapes(self: *LoweredDeviceProgram) []QWeightShape {
                self.owns_qweight_shapes = false;
                return self.qweight_shapes;
            }

            fn deinit(self: *LoweredDeviceProgram) void {
                freeOpPayloads(self.alloc, self.owned_ops);
                self.alloc.free(self.owned_ops);
                self.alloc.free(self.qweight_uploads);
                if (self.owns_qweight_shapes) self.alloc.free(self.qweight_shapes);
                if (self.owns_buffers) self.buffers.deinit();
                self.* = undefined;
            }
        };

        const SessionBindingLayout = struct {
            alloc: std.mem.Allocator,
            persistent_inputs: []backend_mod.ProgramIO,
            step_inputs: []backend_mod.ProgramIO,
            step_outputs: []backend_mod.ProgramIO,

            fn init(
                alloc: std.mem.Allocator,
                buffers: *const BufferMap,
                persistent_tensors: []const *const Tensor,
                persistent_bindings: []const TensorBinding,
                input_tensors: []const *const Tensor,
                input_bindings: []const TensorBinding,
                output_tensor: *const Tensor,
                output_host_buf: [*]T,
                output_len: usize,
                output_binding: ?TensorBinding,
            ) !SessionBindingLayout {
                const n_persistent = if (persistent_bindings.len > 0) persistent_bindings.len else persistent_tensors.len;
                const persistent = try alloc.alloc(backend_mod.ProgramIO, n_persistent);
                errdefer alloc.free(persistent);
                if (persistent_bindings.len > 0) {
                    for (persistent_bindings, 0..) |binding, i| {
                        persistent[i] = try tensorBindingIO(buffers, binding);
                    }
                } else {
                    for (persistent_tensors, 0..) |t, i| {
                        persistent[i] = try tensorIO(buffers, t, t.data.ptr, t.data.len);
                    }
                }

                const n_inputs = if (input_bindings.len > 0) input_bindings.len else input_tensors.len;
                const inputs = try alloc.alloc(backend_mod.ProgramIO, n_inputs);
                errdefer alloc.free(inputs);
                if (input_bindings.len > 0) {
                    for (input_bindings, 0..) |binding, i| {
                        inputs[i] = try tensorExactBindingIO(buffers, binding);
                    }
                } else {
                    for (input_tensors, 0..) |t, i| {
                        inputs[i] = try tensorExactIO(buffers, t, t.data.ptr, t.data.len);
                    }
                }

                const outputs = try alloc.alloc(backend_mod.ProgramIO, 1);
                errdefer alloc.free(outputs);
                outputs[0] = if (output_binding) |binding| blk: {
                    if (binding.tensor != output_tensor) return error.ShapeMismatch;
                    break :blk try tensorExactBindingIO(buffers, binding);
                } else try tensorExactIO(buffers, output_tensor, output_host_buf, output_len);

                return .{
                    .alloc = alloc,
                    .persistent_inputs = persistent,
                    .step_inputs = inputs,
                    .step_outputs = outputs,
                };
            }

            fn configure(self: *const SessionBindingLayout, program: *const Program, runtime_handle: backend_mod.Backend.RuntimeHandle) bool {
                return program.be.configureBindings(program.handle, runtime_handle, self.persistent_inputs, self.step_inputs, self.step_outputs);
            }

            fn validatePersistent(self: *const SessionBindingLayout, program: *const Program) !void {
                try program.validatePersistentIO(self.persistent_inputs);
            }

            fn validatePersistentRange(self: *const SessionBindingLayout, program: *const Program, first: usize, len: usize) ![]const backend_mod.ProgramIO {
                if (first > self.persistent_inputs.len or len > self.persistent_inputs.len - first) return error.InvalidProgramIO;
                const inputs = self.persistent_inputs[first..][0..len];
                if (inputs.len == 0) return inputs;
                try program.validatePersistentIO(inputs);
                return inputs;
            }

            fn stepOutputsForDownload(self: *const SessionBindingLayout, download_outputs: bool) []const backend_mod.ProgramIO {
                return if (download_outputs) self.step_outputs else &.{};
            }

            fn validateStep(self: *const SessionBindingLayout, program: *const Program, download_outputs: bool) !void {
                try program.validateStepIO(self.step_inputs, self.stepOutputsForDownload(download_outputs));
            }

            fn outputStorage(self: *const SessionBindingLayout) BindingStorage {
                return if (self.step_outputs.len > 0) storageForProgramIO(self.step_outputs[0]) else .none;
            }

            fn bindingShapeHash(self: *const SessionBindingLayout) u64 {
                return sessionBindingShapeHash(self.persistent_inputs, self.step_inputs, self.step_outputs);
            }

            fn addCounts(self: *const SessionBindingLayout, dest: *SessionInspection) void {
                addProgramIOCounts(dest, self.persistent_inputs);
                addProgramIOCounts(dest, self.step_inputs);
                addProgramIOCounts(dest, self.step_outputs);
            }

            fn inspect(self: *const SessionBindingLayout, backend: backend_mod.Device) SessionInspection {
                var out = SessionInspection{
                    .backend = backend,
                    .output_storage = self.outputStorage(),
                    .persistent_binding_count = self.persistent_inputs.len,
                    .step_input_count = self.step_inputs.len,
                    .step_output_count = self.step_outputs.len,
                    .host_binding_count = 0,
                    .resource_binding_count = 0,
                    .binding_shape_hash = self.bindingShapeHash(),
                };
                self.addCounts(&out);
                return out;
            }

            fn deinit(self: *SessionBindingLayout) void {
                self.alloc.free(self.persistent_inputs);
                self.alloc.free(self.step_inputs);
                self.alloc.free(self.step_outputs);
                self.* = undefined;
            }
        };

        pub const Session = struct {
            be: backend_mod.Backend,
            program_handle: backend_mod.Backend.CompiledHandle,
            runtime_handle: backend_mod.Backend.RuntimeHandle,
            bindings: SessionBindingLayout,

            fn bind(
                alloc: std.mem.Allocator,
                be: backend_mod.Backend,
                program_handle: backend_mod.Backend.CompiledHandle,
                runtime_handle: backend_mod.Backend.RuntimeHandle,
                buffers: *const BufferMap,
                persistent_tensors: []const *const Tensor,
                persistent_bindings: []const TensorBinding,
                input_tensors: []const *const Tensor,
                input_bindings: []const TensorBinding,
                output_tensor: *const Tensor,
                output_host_buf: [*]T,
                output_len: usize,
                output_binding: ?TensorBinding,
            ) !Session {
                const bindings = try SessionBindingLayout.init(alloc, buffers, persistent_tensors, persistent_bindings, input_tensors, input_bindings, output_tensor, output_host_buf, output_len, output_binding);

                return .{
                    .be = be,
                    .program_handle = program_handle,
                    .runtime_handle = runtime_handle,
                    .bindings = bindings,
                };
            }

            pub fn uploadPersistent(self: *Session, program: *Program) !void {
                if (self.bindings.persistent_inputs.len == 0) return;
                try self.bindings.validatePersistent(program);
                self.be.uploadBindings(self.program_handle, self.runtime_handle, self.bindings.persistent_inputs);
            }

            pub fn uploadPersistentRange(self: *Session, program: *Program, first: usize, len: usize) !void {
                const inputs = try self.bindings.validatePersistentRange(program, first, len);
                if (inputs.len == 0) return;
                self.be.uploadBindings(self.program_handle, self.runtime_handle, inputs);
            }

            pub fn executeStep(self: *Session, program: *Program, params: StepParams) !void {
                if (!self.be.capabilities.executes_programs) return error.ExecutionUnsupported;
                if (params.dynamic_io) {
                    try self.bindings.validateStep(program, params.download_outputs);
                }
                const status = self.be.patchRuntimeBindings(self.program_handle, self.runtime_handle, params.window);
                if (status == .invalid) return error.InvalidRuntimePatch;
                if (params.dynamic_io) {
                    const outputs = self.bindings.stepOutputsForDownload(params.download_outputs);
                    self.be.executeBindings(self.program_handle, self.runtime_handle, self.bindings.step_inputs, outputs);
                } else {
                    self.be.executeConfiguredBindings(self.program_handle, self.runtime_handle, params.download_outputs);
                }
            }

            pub fn resetRuntimeProfile(self: *const Session) void {
                self.be.resetRuntimeBindingsProfile(self.program_handle, self.runtime_handle);
            }

            pub fn addRuntimeProfileTo(self: *const Session, dest: *profile.RuntimeProfile) void {
                self.be.addRuntimeBindingsProfileTo(self.program_handle, self.runtime_handle, dest);
            }

            pub fn inspect(self: *const Session) SessionInspection {
                return self.bindings.inspect(self.be.device_type);
            }

            pub fn deinit(self: *Session) void {
                self.be.freeBindings(self.program_handle, self.runtime_handle);
                self.bindings.deinit();
                self.* = undefined;
            }
        };

        fn storageForProgramIO(io: backend_mod.ProgramIO) BindingStorage {
            return if (io.resource != null) .external_resource else .host;
        }

        fn addProgramIOCounts(dest: *SessionInspection, ios: []const backend_mod.ProgramIO) void {
            for (ios) |io| {
                if (io.resource != null) {
                    dest.resource_binding_count += 1;
                } else {
                    dest.host_binding_count += 1;
                }
            }
        }

        const InitOptions = struct {
            graph: *ComputeGraph,
            be: backend_mod.Backend,
            alloc: std.mem.Allocator,
            input_tensors: []const *const Tensor,
            output_tensor: *const Tensor,
            output_host_buf: [*]T,
            output_len: usize,
            persistent_tensors: []const *const Tensor = &.{},
            quant_weights: []const QuantizedWeight(T) = &.{},
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize) = &empty_quant_map,
            expected_runtime_patch_shape: ?backend_mod.RuntimePatchShape = null,
        };

        const empty_quant_map: std.AutoHashMapUnmanaged(*Tensor, usize) = .empty;
        pub fn init(opts: InitOptions) !Self {
            var program = try Program.compile(.{
                .graph = opts.graph,
                .be = opts.be,
                .alloc = opts.alloc,
                .input_tensors = opts.input_tensors,
                .persistent_tensors = opts.persistent_tensors,
                .output_tensors = &.{opts.output_tensor},
                .quant_weights = opts.quant_weights,
                .quant_map = opts.quant_map,
                .expected_runtime_patch_shape = opts.expected_runtime_patch_shape,
            });
            errdefer program.deinit();

            var session = try program.bind(.{
                .input_tensors = opts.input_tensors,
                .persistent_tensors = opts.persistent_tensors,
                .output_tensor = opts.output_tensor,
                .output_host_buf = opts.output_host_buf,
                .output_len = opts.output_len,
            });
            errdefer session.deinit();

            return .{
                .program = program,
                .session = session,
            };
        }

        pub inline fn uploadPersistentInputs(self: *Self) !void {
            try self.program.uploadPersistentInputs(&self.session);
        }

        pub inline fn executeStep(self: *Self, params: StepParams) !void {
            try self.program.executeStep(&self.session, params);
        }

        /// Execute after patching a token window into the compiled program.
        pub inline fn executeDynamic(self: *Self, window: backend_mod.RuntimeWindow) !void {
            try self.executeStep(.{ .window = window });
        }

        /// Execute a dynamic program without downloading the configured output tensor.
        /// This is for known-token advancement and diagnostics, not parity decode rows.
        pub inline fn executeDynamicNoOutputs(self: *Self, window: backend_mod.RuntimeWindow) !void {
            try self.executeStep(.{ .window = window, .download_outputs = false });
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.session.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile.RuntimeProfile) void {
            self.session.addRuntimeProfileTo(dest);
        }

        pub fn deinit(self: *Self) void {
            self.session.deinit();
            self.program.deinit();
        }

        // ── Helpers ──────────────────────────────────────────────

        const BufferRef = struct {
            idx: u16,
            offset: usize,
        };

        const StorageRef = struct {
            ptr: [*]T,
            len: usize,
            offset: usize,
            base: *const Tensor,
        };

        const BufferMap = struct {
            alloc: std.mem.Allocator,
            ptr_to_idx: std.AutoHashMap([*]T, u16),
            tensor_to_ref: std.AutoHashMap(*const Tensor, BufferRef),
            skip_upload_ptrs: std.AutoHashMap([*]T, void),
            buf_sizes: std.ArrayListUnmanaged(usize) = .empty,
            uploads: std.ArrayListUnmanaged(backend_mod.ProgramIO) = .empty,

            fn init(alloc: std.mem.Allocator) BufferMap {
                return .{
                    .alloc = alloc,
                    .ptr_to_idx = std.AutoHashMap([*]T, u16).init(alloc),
                    .tensor_to_ref = std.AutoHashMap(*const Tensor, BufferRef).init(alloc),
                    .skip_upload_ptrs = std.AutoHashMap([*]T, void).init(alloc),
                };
            }

            fn deinit(self: *BufferMap) void {
                self.ptr_to_idx.deinit();
                self.tensor_to_ref.deinit();
                self.skip_upload_ptrs.deinit();
                self.buf_sizes.deinit(self.alloc);
                self.uploads.deinit(self.alloc);
            }

            fn idx(self: *const BufferMap, tensor: *const Tensor) u16 {
                return self.tensor_to_ref.get(tensor).?.idx;
            }

            fn offset(self: *const BufferMap, tensor: *const Tensor) usize {
                return self.tensor_to_ref.get(tensor).?.offset;
            }

            pub fn skipInitialUpload(self: *BufferMap, tensor: *const Tensor) !void {
                const sr = storageRef(tensor);
                try self.skip_upload_ptrs.put(sr.ptr, {});
            }

            pub fn ensure(self: *BufferMap, tensor: *const Tensor) !void {
                if (self.tensor_to_ref.contains(tensor)) return;
                const sr = storageRef(tensor);
                const entry = try self.ptr_to_idx.getOrPut(sr.ptr);
                if (!entry.found_existing) {
                    entry.value_ptr.* = std.math.cast(u16, self.buf_sizes.items.len) orelse return error.UnsupportedDeviceOp;
                    try self.buf_sizes.append(self.alloc, @max(sr.len, 1));
                    if (sr.base.opTag() == .none and sr.base.data.len > 0 and !self.skip_upload_ptrs.contains(sr.ptr)) {
                        try self.uploads.append(self.alloc, try programIO(entry.value_ptr.*, 0, sr.base.data.ptr, sr.base.data.len));
                    }
                } else {
                    self.buf_sizes.items[entry.value_ptr.*] = @max(self.buf_sizes.items[entry.value_ptr.*], @max(sr.len, 1));
                }
                try self.tensor_to_ref.put(tensor, .{ .idx = entry.value_ptr.*, .offset = sr.offset });
            }
        };

        fn storageBase(tensor: *const Tensor) *const Tensor {
            return switch (tensor.opTag()) {
                .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to => storageBase(tensor.src0.?),
                .slice_assign, .slice_assign_rows => storageBase(tensor.src1.?),
                else => tensor,
            };
        }

        fn elementwiseSourceExact(node: *const Tensor, src: *const Tensor) bool {
            return src.nElems() == node.nElems();
        }

        fn elementwiseChainNeedsBroadcastIr(ir: anytype, chain: anytype) bool {
            var current = ir.valueTensor(chain.input);
            for (ir.elementwise_steps[chain.step_start..][0..chain.step_count]) |step| {
                const dst = ir.valueTensor(step.output);
                if (step.op.isBinary()) {
                    const secondary = if (step.secondary) |id| ir.valueTensor(id) else return true;
                    const lhs = if (step.is_swapped) secondary else current;
                    const rhs = if (step.is_swapped) current else secondary;
                    if (!elementwiseSourceExact(dst, lhs) or !elementwiseSourceExact(dst, rhs)) return true;
                }
                current = dst;
            }
            return false;
        }

        fn appendElementwiseRawOp(
            ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp),
            alloc: std.mem.Allocator,
            op: Op,
            dst_idx: u16,
            src0_idx: u16,
            src1_idx: u16,
            n: usize,
            dst_offset: usize,
            src0_offset: usize,
            src1_offset: usize,
        ) !void {
            try ops.append(alloc, .{ .elementwise = .{
                .op = op,
                .dst = dst_idx,
                .src0 = src0_idx,
                .src1 = src1_idx,
                .n = @intCast(n),
                .dst_offset = @intCast(dst_offset),
                .src0_offset = @intCast(src0_offset),
                .src1_offset = @intCast(src1_offset),
            } });
        }

        fn storageRef(tensor: *const Tensor) StorageRef {
            if (tensor.opTag() == .slice_assign or tensor.opTag() == .slice_assign_rows) {
                return storageRef(tensor.src1.?);
            }
            const base = storageBase(tensor);
            const base_addr = @intFromPtr(base.data.ptr);
            const tensor_addr = @intFromPtr(tensor.data.ptr);
            std.debug.assert(tensor_addr >= base_addr);
            const ptr_delta = (tensor_addr - base_addr) / @sizeOf(T);
            return .{
                .ptr = base.data.ptr,
                .len = base.data.len,
                .offset = ptr_delta + tensor.storage_offset,
                .base = base,
            };
        }

        fn programIO(buf_idx: u16, element_offset: usize, host_ptr: [*]T, element_count: usize) !backend_mod.ProgramIO {
            const byte_offset = std.math.mul(usize, element_offset, @sizeOf(T)) catch return error.InvalidProgramIO;
            const byte_size = std.math.mul(usize, element_count, @sizeOf(T)) catch return error.InvalidProgramIO;
            return backend_mod.ProgramIO.host(
                buf_idx,
                std.math.cast(u32, byte_offset) orelse return error.InvalidProgramIO,
                @ptrCast(host_ptr),
                std.math.cast(u32, byte_size) orelse return error.InvalidProgramIO,
            );
        }

        fn programResourceIO(buf_idx: u16, element_offset: usize, resource: backend_mod.ProgramIO.ExternalResource, element_count: usize) !backend_mod.ProgramIO {
            const byte_offset = std.math.mul(usize, element_offset, @sizeOf(T)) catch return error.InvalidProgramIO;
            const byte_size = std.math.mul(usize, element_count, @sizeOf(T)) catch return error.InvalidProgramIO;
            return backend_mod.ProgramIO.external(
                buf_idx,
                std.math.cast(u32, byte_offset) orelse return error.InvalidProgramIO,
                resource,
                std.math.cast(u32, byte_size) orelse return error.InvalidProgramIO,
            );
        }

        fn tensorIO(buffers: *const BufferMap, tensor: *const Tensor, host_ptr: [*]T, element_count: usize) !backend_mod.ProgramIO {
            return programIO(buffers.idx(tensor), buffers.offset(tensor), host_ptr, element_count);
        }

        fn tensorResourceIO(buffers: *const BufferMap, tensor: *const Tensor, resource: backend_mod.ProgramIO.ExternalResource, element_count: usize) !backend_mod.ProgramIO {
            return programResourceIO(buffers.idx(tensor), buffers.offset(tensor), resource, element_count);
        }

        fn tensorBindingIO(buffers: *const BufferMap, binding: TensorBinding) !backend_mod.ProgramIO {
            if (binding.resource) |resource| return tensorResourceIO(buffers, binding.tensor, resource, binding.element_count);
            return tensorIO(buffers, binding.tensor, binding.host_ptr, binding.element_count);
        }

        fn requireExactTensorSpan(tensor: *const Tensor, element_count: usize) !void {
            if (element_count != tensor.nElems()) return error.ShapeMismatch;
        }

        fn tensorExactIO(buffers: *const BufferMap, tensor: *const Tensor, host_ptr: [*]T, element_count: usize) !backend_mod.ProgramIO {
            try requireExactTensorSpan(tensor, element_count);
            return tensorIO(buffers, tensor, host_ptr, element_count);
        }

        fn tensorExactResourceIO(buffers: *const BufferMap, tensor: *const Tensor, resource: backend_mod.ProgramIO.ExternalResource, element_count: usize) !backend_mod.ProgramIO {
            try requireExactTensorSpan(tensor, element_count);
            return tensorResourceIO(buffers, tensor, resource, element_count);
        }

        fn tensorExactBindingIO(buffers: *const BufferMap, binding: TensorBinding) !backend_mod.ProgramIO {
            if (binding.resource) |resource| return tensorExactResourceIO(buffers, binding.tensor, resource, binding.element_count);
            return tensorExactIO(buffers, binding.tensor, binding.host_ptr, binding.element_count);
        }

        fn isCanonicalRowSoftmaxInput(node: *const Tensor, src: *const Tensor) bool {
            if (!node.isSameShape(src)) return false;
            if (!node.isDenseLayout() or !src.isDenseLayout()) return false;
            if (node.reduce_ne[0] != 1) return false;

            var dim: usize = 1;
            while (dim < src.n_dims) : (dim += 1) {
                if (node.reduce_ne[dim] != src.ne[dim]) return false;
            }
            while (dim < @import("tensor.zig").max_dims) : (dim += 1) {
                if (node.reduce_ne[dim] != 1) return false;
            }
            return true;
        }

        fn shape4(tensor: *const Tensor) [4]u32 {
            return .{
                @intCast(tensor.ne[0]),
                @intCast(tensor.ne[1]),
                @intCast(tensor.ne[2]),
                @intCast(tensor.ne[3]),
            };
        }

        fn strides4(tensor: *const Tensor) [4]u32 {
            return .{
                @intCast(tensor.strides[0]),
                @intCast(tensor.strides[1]),
                @intCast(tensor.strides[2]),
                @intCast(tensor.strides[3]),
            };
        }

        fn reduceDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .reduce = .{
                .op = node.opTag(),
                .dst = dst_idx,
                .src = src_idx,
                .n_out = @intCast(node.nElems()),
                .reduce_size = @intCast(src.ne[0]),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn repeatDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .repeat = .{
                .dst = dst_idx,
                .src = src_idx,
                .n = @intCast(node.nElems()),
                .src_ne = shape4(src),
                .dst_ne = shape4(node),
                .src_strides = strides4(src),
                .dst_strides = strides4(node),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn gatherRowsDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, indices: *const Tensor, dst_idx: u16, src_idx: u16, indices_idx: u16) backend_mod.DeviceOp {
            return .{ .gather_rows = .{
                .dst = dst_idx,
                .src = src_idx,
                .indices = indices_idx,
                .width = @intCast(src.ne[0]),
                .count = @intCast(indices.ne[0]),
                .src_rows = @intCast(src.ne[1]),
                .src_offset = @intCast(buffers.offset(src)),
                .indices_offset = @intCast(buffers.offset(indices)),
                .dst_offset = @intCast(buffers.offset(node)),
                .src_row_stride = @intCast(src.strides[1]),
                .dst_row_stride = @intCast(node.strides[1]),
            } };
        }

        fn softmaxDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .softmax = .{
                .dst = dst_idx,
                .src = src_idx,
                .rows = @intCast(src.nElems() / src.ne[0]),
                .cols = @intCast(src.ne[0]),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn logsoftmaxDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .logsoftmax = .{
                .dst = dst_idx,
                .src = src_idx,
                .rows = @intCast(src.nElems() / src.ne[0]),
                .cols = @intCast(src.ne[0]),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn rmsnormDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .rmsnorm = .{
                .dst = dst_idx,
                .src = src_idx,
                .rows = @intCast(src.ne[1]),
                .cols = @intCast(src.ne[0]),
                .eps = node.op_eps,
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn sliceAssignDeviceOp(
            buffers: *const BufferMap,
            src: *const Tensor,
            dst: *const Tensor,
            src_idx: u16,
            rows: usize,
            cols: usize,
            dst_base_offset: usize,
            dst_offset: usize,
            patch_stride: usize,
        ) backend_mod.DeviceOp {
            return .{ .slice_assign = .{
                .dst = buffers.idx(dst),
                .src = src_idx,
                .rows = @intCast(rows),
                .cols = @intCast(cols),
                .dst_base_offset = @intCast(dst_base_offset),
                .dst_offset = @intCast(dst_offset),
                .dst_row_stride = @intCast(dst.strides[0]),
                .dst_col_stride = @intCast(dst.strides[1]),
                .src_offset = @intCast(buffers.offset(src)),
                .src_row_stride = @intCast(src.strides[0]),
                .src_col_stride = @intCast(src.strides[1]),
                .patch_stride = @intCast(patch_stride),
            } };
        }

        fn ropeDeviceOp(
            buffers: *const BufferMap,
            node: *Tensor,
            src: *const Tensor,
            cs: *const Tensor,
            dst_idx: u16,
            src_idx: u16,
            cos_sin_idx: u16,
        ) backend_mod.DeviceOp {
            return .{ .rope = .{
                .dst = dst_idx,
                .src = src_idx,
                .cos_sin = cos_sin_idx,
                .half_d = @intCast(src.ne[0] / 2),
                .seq_len = @intCast(src.ne[1]),
                .src_off = @intCast(buffers.offset(src)),
                .cs_off = @intCast(buffers.offset(cs)),
                .dst_off = @intCast(buffers.offset(node)),
                .src_rs = @intCast(src.strides[0]),
                .src_cs = @intCast(src.strides[1]),
                .cs_cs = @intCast(cs.strides[1]),
            } };
        }

        fn attentionDeviceOp(
            buffers: *const BufferMap,
            node: *Tensor,
            q: *const Tensor,
            k: *const Tensor,
            v: *const Tensor,
            mask: ?*const Tensor,
            dst_idx: u16,
            q_idx: u16,
            k_idx: u16,
        ) backend_mod.DeviceOp {
            return .{ .attention = .{
                .dst = dst_idx,
                .q = q_idx,
                .k = k_idx,
                .v = buffers.idx(v),
                .mask = if (mask) |m| buffers.idx(m) else dst_idx,
                .has_mask = mask != null,
                .d_head = @intCast(q.ne[0]),
                .seq_q = @intCast(q.ne[1]),
                .seq_kv = @intCast(k.ne[1]),
                .scale = node.op_scale,
                .q_off = @intCast(buffers.offset(q)),
                .k_off = @intCast(buffers.offset(k)),
                .v_off = @intCast(buffers.offset(v)),
                .mask_off = if (mask) |m| @intCast(buffers.offset(m)) else 0,
                .dst_off = @intCast(buffers.offset(node)),
                .q_rs = @intCast(q.strides[0]),
                .q_cs = @intCast(q.strides[1]),
                .k_rs = @intCast(k.strides[0]),
                .k_cs = @intCast(k.strides[1]),
                .v_rs = @intCast(v.strides[0]),
                .v_cs = @intCast(v.strides[1]),
                .mask_rs = if (mask) |m| @intCast(m.strides[0]) else 0,
                .mask_cs = if (mask) |m| @intCast(m.strides[1]) else 0,
                .dst_rs = @intCast(node.strides[0]),
                .dst_cs = @intCast(node.strides[1]),
                .patch_seq_kv = k.opTag() == .slice_assign or v.opTag() == .slice_assign,
            } };
        }
    };
}

const testing = std.testing;
const TensorF32 = @import("tensor.zig").Tensor(f32);
const ComputeGraphF32 = @import("graph.zig").ComputeGraph(f32);

test "DeviceInference builds checked ProgramIO byte spans" {
    const DeviceF32 = DeviceInference(f32);
    var data = [_]f32{0} ** 4;

    const io = try DeviceF32.programIO(2, 1, data[0..].ptr, 2);
    try testing.expectEqual(@as(u16, 2), io.buf_idx);
    try testing.expectEqual(@as(u32, @sizeOf(f32)), io.offset);
    try testing.expectEqual(@as(u32, 2 * @sizeOf(f32)), io.size);

    const too_many_f32 = @as(usize, std.math.maxInt(u32)) / @sizeOf(f32) + 1;
    try testing.expectError(error.InvalidProgramIO, DeviceF32.programIO(0, too_many_f32, data[0..].ptr, 1));
    try testing.expectError(error.InvalidProgramIO, DeviceF32.programIO(0, 0, data[0..].ptr, too_many_f32));
}

test "Session binding shape hash ignores host addresses and includes resources" {
    var host_a = [_]f32{ 1, 2 };
    var host_b = [_]f32{ 3, 4 };
    const byte_len: u32 = @intCast(host_a.len * @sizeOf(f32));
    const host_io_a = backend_mod.ProgramIO.host(0, 0, @ptrCast(host_a[0..].ptr), byte_len);
    const host_io_b = backend_mod.ProgramIO.host(0, 0, @ptrCast(host_b[0..].ptr), byte_len);
    const host_hash_a = sessionBindingShapeHash(&.{host_io_a}, &.{}, &.{});
    const host_hash_b = sessionBindingShapeHash(&.{host_io_b}, &.{}, &.{});
    try testing.expect(host_hash_a != 0);
    try testing.expectEqual(host_hash_a, host_hash_b);

    const resource_a = backend_mod.ProgramIO.external(0, 0, .{
        .placement = .webgpu,
        .handle = 100,
        .byte_len = byte_len,
        .access = .read_only,
    }, byte_len);
    const resource_b = backend_mod.ProgramIO.external(0, 0, .{
        .placement = .webgpu,
        .handle = 101,
        .byte_len = byte_len,
        .access = .read_only,
    }, byte_len);
    try testing.expect(sessionBindingShapeHash(&.{resource_a}, &.{}, &.{}) != sessionBindingShapeHash(&.{resource_b}, &.{}, &.{}));
}

const TestBackendState = struct {
    device_type: backend_mod.Device = .cpu,
    compile_calls: usize = 0,
    bind_calls: usize = 0,
    configure_binding_calls: usize = 0,
    free_binding_calls: usize = 0,
    upload_calls: usize = 0,
    execute_calls: usize = 0,
    patch_calls: usize = 0,
    last_upload_count: usize = 0,
    last_input_count: usize = 0,
    last_output_count: usize = 0,
    configured_persistent_count: usize = 0,
    configured_input_count: usize = 0,
    configured_output_count: usize = 0,
    last_uploads_storage: [8]backend_mod.ProgramIO = undefined,
    last_inputs_storage: [8]backend_mod.ProgramIO = undefined,
    last_outputs_storage: [8]backend_mod.ProgramIO = undefined,
    configured_persistent_storage: [8]backend_mod.ProgramIO = undefined,
    configured_inputs_storage: [8]backend_mod.ProgramIO = undefined,
    configured_outputs_storage: [8]backend_mod.ProgramIO = undefined,
    last_uploads: []const backend_mod.ProgramIO = &.{},
    last_inputs: []const backend_mod.ProgramIO = &.{},
    last_outputs: []const backend_mod.ProgramIO = &.{},
    configured_persistent: []const backend_mod.ProgramIO = &.{},
    configured_inputs: []const backend_mod.ProgramIO = &.{},
    configured_outputs: []const backend_mod.ProgramIO = &.{},
    configured_execute_binding_calls: usize = 0,
    compiled_initial_upload_count: usize = 0,
    compiled_ops_storage: [64]backend_mod.DeviceOp = undefined,
    compiled_fused_steps_storage: [256]backend_mod.FusedEwStep = undefined,
    compiled_fused_step_count: usize = 0,
    compiled_ops: []const backend_mod.DeviceOp = &.{},
    compiled_stencil: program_mod.ProgramStencil = .{},
    has_compiled_stencil: bool = false,
    compiled_runtime_patch_shape: backend_mod.RuntimePatchShape = .{},
    compiled_command_shape: program_mod.ProgramCommandStreamShape = .{},
    patched_window: backend_mod.RuntimeWindow = .{ .position = 0, .len = 0 },
    patch_status: backend_mod.RuntimePatchStatus = .changed,
    runtime_profile: profile.RuntimeProfile = .{},
    binding_runtime_profile: profile.RuntimeProfile = .{},
};

fn testDenseMatMul(_: *anyopaque, _: backend_mod.DenseMatMulSpecF32) bool {
    return false;
}

fn recordProgramIOList(storage: *[8]backend_mod.ProgramIO, dest: *[]const backend_mod.ProgramIO, source: []const backend_mod.ProgramIO) void {
    const n = @min(storage.len, source.len);
    if (n > 0) @memcpy(storage[0..n], source[0..n]);
    dest.* = storage[0..n];
}

fn testCompile(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.compile_calls += 1;
    if (program.ops.len > state.compiled_ops_storage.len) return null;
    if (state.has_compiled_stencil) {
        state.compiled_stencil.deinit(testing.allocator);
        state.has_compiled_stencil = false;
    }
    @memcpy(state.compiled_ops_storage[0..program.ops.len], program.ops);
    state.compiled_fused_step_count = 0;
    for (state.compiled_ops_storage[0..program.ops.len]) |*op| {
        switch (op.*) {
            .fused_elementwise => |fe| {
                const start = state.compiled_fused_step_count;
                const end = start + fe.steps.len;
                if (end > state.compiled_fused_steps_storage.len) return null;
                if (fe.steps.len > 0) @memcpy(state.compiled_fused_steps_storage[start..end], fe.steps);
                state.compiled_fused_step_count = end;
                op.* = .{ .fused_elementwise = .{
                    .steps = state.compiled_fused_steps_storage[start..end],
                    .n = fe.n,
                    .dst = fe.dst,
                    .src = fe.src,
                    .dst_offset = fe.dst_offset,
                    .src_offset = fe.src_offset,
                } };
            },
            else => {},
        }
    }
    state.compiled_ops = state.compiled_ops_storage[0..program.ops.len];
    state.compiled_initial_upload_count = program.initial_uploads.len;

    state.compiled_stencil = program_mod.ProgramStencil.initProgramWithKernelizer(testing.allocator, program, program_mod.Kernelizer.default()) catch return null;
    state.has_compiled_stencil = true;
    const inspection = state.compiled_stencil.inspect();
    state.compiled_runtime_patch_shape = inspection.runtime_patch_shape;
    state.compiled_command_shape = inspection.command_shape;
    state.runtime_profile = .{
        .runtime_patch_shape = state.compiled_runtime_patch_shape,
        .program_command_shape = state.compiled_command_shape,
    };
    state.binding_runtime_profile = .{
        .runtime_patch_shape = state.compiled_runtime_patch_shape,
        .program_command_shape = state.compiled_command_shape,
    };
    return @ptrFromInt(1);
}

fn testBind(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle) ?backend_mod.Backend.RuntimeHandle {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.bind_calls += 1;
    state.binding_runtime_profile = .{
        .runtime_patch_shape = state.compiled_runtime_patch_shape,
        .program_command_shape = state.compiled_command_shape,
    };
    return @ptrFromInt(2);
}

fn recordTestExecution(state: *TestBackendState, runtime_profile: *profile.RuntimeProfile, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    state.execute_calls += 1;
    state.last_input_count = inputs.len;
    state.last_output_count = outputs.len;
    recordProgramIOList(&state.last_inputs_storage, &state.last_inputs, inputs);
    recordProgramIOList(&state.last_outputs_storage, &state.last_outputs, outputs);
    runtime_profile.call_count +%= 1;
    runtime_profile.backend_op_count +%= state.compiled_ops.len;
    runtime_profile.recordProgramCommandShapeDispatch(state.compiled_command_shape);
}

fn testExecute(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    recordTestExecution(state, &state.runtime_profile, inputs, outputs);
}

fn testPatchRuntimeWindow(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.patch_calls += 1;
    state.patched_window = window;
    state.runtime_profile.recordRuntimePatch(state.patch_status);
    return state.patch_status;
}

fn testUpload(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.upload_calls += 1;
    state.last_upload_count = inputs.len;
    recordProgramIOList(&state.last_uploads_storage, &state.last_uploads, inputs);
}

fn testExecuteBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    recordTestExecution(state, &state.binding_runtime_profile, inputs, outputs);
}

fn testExecuteConfiguredBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, download_outputs: bool) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.configured_execute_binding_calls += 1;
    const outputs = if (download_outputs) state.configured_outputs else &.{};
    recordTestExecution(state, &state.binding_runtime_profile, state.configured_inputs, outputs);
}

fn testPatchRuntimeBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.patch_calls += 1;
    state.patched_window = window;
    state.binding_runtime_profile.recordRuntimePatch(state.patch_status);
    return state.patch_status;
}

fn testUploadBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO) void {
    testUpload(ctx, @ptrFromInt(1), inputs);
}

fn testConfigureBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) bool {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.configure_binding_calls += 1;
    state.configured_persistent_count = persistent_inputs.len;
    state.configured_input_count = step_inputs.len;
    state.configured_output_count = step_outputs.len;
    recordProgramIOList(&state.configured_persistent_storage, &state.configured_persistent, persistent_inputs);
    recordProgramIOList(&state.configured_inputs_storage, &state.configured_inputs, step_inputs);
    recordProgramIOList(&state.configured_outputs_storage, &state.configured_outputs, step_outputs);
    return true;
}

fn testUploadQWeights(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, _: []const backend_mod.QuantizedWeightUpload) bool {
    return true;
}

fn testFreeBindings(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.free_binding_calls += 1;
}

fn testFree(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    if (state.has_compiled_stencil) {
        state.compiled_stencil.deinit(testing.allocator);
        state.has_compiled_stencil = false;
        state.compiled_runtime_patch_shape = .{};
        state.compiled_command_shape = .{};
    }
}

fn testResetRuntimeProfile(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.runtime_profile.reset();
}

fn testAddRuntimeProfileTo(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, dest: *profile.RuntimeProfile) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    dest.add(state.runtime_profile);
}

fn testResetRuntimeBindingsProfile(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.binding_runtime_profile.reset();
}

fn testAddRuntimeBindingsProfileTo(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: backend_mod.Backend.RuntimeHandle, dest: *profile.RuntimeProfile) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    dest.add(state.binding_runtime_profile);
}

const test_vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = testDenseMatMul,
    .compile_program = testCompile,
    .bind_program = testBind,
    .configure_bindings = testConfigureBindings,
    .patch_runtime_bindings = testPatchRuntimeBindings,
    .upload_bindings = testUploadBindings,
    .upload_qweights = testUploadQWeights,
    .execute_bindings = testExecuteBindings,
    .execute_configured_bindings = testExecuteConfiguredBindings,
    .free_bindings = testFreeBindings,
    .patch_runtime_window = testPatchRuntimeWindow,
    .upload_program = testUpload,
    .execute_program = testExecute,
    .free_program = testFree,
    .reset_runtime_profile = testResetRuntimeProfile,
    .add_runtime_profile_to = testAddRuntimeProfileTo,
    .reset_runtime_bindings_profile = testResetRuntimeBindingsProfile,
    .add_runtime_bindings_profile_to = testAddRuntimeBindingsProfileTo,
};

fn testBackend(state: *TestBackendState) backend_mod.Backend {
    return testBackendForDevice(state, .cpu);
}

fn testBackendForDevice(state: *TestBackendState, device_type: backend_mod.Device) backend_mod.Backend {
    state.device_type = device_type;
    const capabilities = switch (device_type) {
        .metal => backend_mod.Capabilities.metal,
        else => backend_mod.Capabilities.reference_cpu,
    };
    return .{
        .ctx = state,
        .vtable = &test_vtable,
        .name_str = "test",
        .device_type = device_type,
        .capabilities = capabilities,
    };
}

fn testResourceBackend(state: *TestBackendState) backend_mod.Backend {
    state.device_type = .webgpu;
    var capabilities = backend_mod.Capabilities.webgpu;
    capabilities.external_resources = true;
    return .{
        .ctx = state,
        .vtable = &test_vtable,
        .name_str = "test-resource",
        .device_type = .webgpu,
        .capabilities = capabilities,
    };
}

test "DeviceInference lowers canonical softmax" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 3 });
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.125;
    const y = x.softmax(&.{ 1, 3 });
    try graph.infer(y);

    var out = [_]f32{0} ** 12;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const ops = state.compiled_ops;
    try testing.expectEqual(@as(usize, 1), state.compile_calls);
    try testing.expectEqual(@as(usize, 1), ops.len);
    try testing.expectEqual(@as(u32, 3), ops[0].softmax.rows);
    try testing.expectEqual(@as(u32, 4), ops[0].softmax.cols);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(0, 0) });
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
}

test "DeviceInference inspection reports node-only TensorProgramIr shape" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(graph.forward_node_count, ir.source_node_count);
    try testing.expectEqual(ir.step_count, ir.op_count);
    try testing.expectEqual(ir.step_count, ir.node_step_count);
    try testing.expectEqual(@as(usize, 3), ir.value_count);
    try testing.expectEqual(ir.value_count, ir.dense_value_count);
    try testing.expectEqual(@as(usize, 2), ir.input_edge_count);
    try testing.expectEqual(@as(usize, 0), ir.elementwise_substep_count);
    try testing.expectEqual(@as(usize, 1), ir.value_step_count);
    try testing.expectEqual(@as(usize, 0), ir.structural_step_count);
    try testing.expectEqual(@as(usize, 0), ir.effect_step_count);
    try testing.expectEqual(@as(usize, 0), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 0), ir.unsupported_fusion_step_count);
    try testing.expectEqual(@as(usize, 2), ir.max_rank);
    try testing.expectEqual(@as(usize, 2), ir.total_output_elements);
    try testing.expect(ir.normalized_step_hash != 0);
    try testing.expect(ir.normalized_shape_hash != 0);
}

test "DeviceInference inspection reports fused TensorProgramIr shape" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4 });
    const y = x.exp().log();
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(graph.forward_node_count, ir.source_node_count);
    try testing.expectEqual(ir.step_count, ir.node_step_count + ir.fusion_step_count);
    try testing.expectEqual(ir.step_count, ir.op_count);
    try testing.expectEqual(@as(usize, 3), ir.value_count);
    try testing.expectEqual(ir.value_count, ir.dense_value_count);
    try testing.expectEqual(@as(usize, 1), ir.input_edge_count);
    try testing.expectEqual(@as(usize, 2), ir.elementwise_substep_count);
    try testing.expectEqual(@as(usize, 1), ir.value_step_count);
    try testing.expectEqual(@as(usize, 0), ir.structural_step_count);
    try testing.expectEqual(@as(usize, 0), ir.effect_step_count);
    try testing.expectEqual(@as(usize, 1), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 1), ir.elementwise_chain_step_count);
    try testing.expectEqual(@as(usize, 0), ir.unsupported_fusion_step_count);
    try testing.expectEqual(@as(usize, 1), ir.max_rank);
    try testing.expectEqual(@as(usize, 4), ir.total_output_elements);
    try testing.expect(ir.normalized_step_hash != 0);
    try testing.expect(ir.normalized_shape_hash != 0);
    try testing.expectEqual(@as(usize, 1), state.compiled_ops.len);
    try testing.expectEqual(@as(usize, 2), state.compiled_ops[0].fused_elementwise.steps.len);
}

test "DeviceInference lowers square activation chains as one fused op" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{3});
    x.setData(&.{ -2, 3, 0.5 });
    const y = x.relu().sqr().sqrt();
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(@as(usize, 3), ir.elementwise_substep_count);
    try testing.expectEqual(@as(usize, 1), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 1), ir.elementwise_chain_step_count);
    try testing.expectEqual(@as(usize, 0), ir.unsupported_fusion_step_count);
    const command_shape = program.inspect().command_shape;
    try testing.expectEqual(@as(u32, 1), command_shape.command_count);
    const command_categories = command_shape.categoryCounts();
    try testing.expectEqual(@as(u64, 1), command_categories.op);
    try testing.expectEqual(@as(usize, 1), state.compiled_ops.len);
    try testing.expectEqual(backend_mod.DeviceOp.fused_elementwise, std.meta.activeTag(state.compiled_ops[0]));
    const steps = state.compiled_ops[0].fused_elementwise.steps;
    try testing.expectEqual(@as(usize, 3), steps.len);
    try testing.expectEqual(Op.relu, steps[0].op);
    try testing.expectEqual(Op.sqr, steps[1].op);
    try testing.expectEqual(Op.sqrt, steps[2].op);
}

test "DeviceInference lowers avg pool2d fusion as one native op" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 4, 1, 1 });
    x.setData(&.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    });
    const y = x.avgPool2d();
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(@as(usize, 1), ir.step_count);
    try testing.expectEqual(@as(usize, 1), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 0), ir.unsupported_fusion_step_count);
    try testing.expectEqual(@as(usize, 1), state.compiled_ops.len);
    try testing.expectEqual(backend_mod.DeviceOp.avg_pool2d, std.meta.activeTag(state.compiled_ops[0]));
}

test "DeviceInference lowers layer norm fusion through TensorProgramIr attrs" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 2 });
    x.setData(&.{ 1, 3, 2, 4 });
    const y = x.layerNorm(&.{ 1, 2 }, 1e-3);
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(@as(usize, 1), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 1), ir.layer_norm_step_count);
    try testing.expect(ir.input_edge_count >= 1);
    try testing.expectEqual(@as(usize, 0), ir.elementwise_substep_count);
    try testing.expect(ir.normalized_step_hash != 0);
    var maybe_layernorm: ?backend_mod.DeviceOp = null;
    for (state.compiled_ops) |op| switch (op) {
        .layernorm => {
            try testing.expect(maybe_layernorm == null);
            maybe_layernorm = op;
        },
        else => {},
    };
    const op = (maybe_layernorm orelse return error.TestExpectedEqual).layernorm;
    try testing.expectEqual(@as(u32, 2), op.rows);
    try testing.expectEqual(@as(u32, 2), op.cols);
    try testing.expectApproxEqAbs(@as(f32, 1e-3), op.eps, 1e-8);
}

test "DeviceInference lowers log softmax fusion to native row op when supported" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.logSoftmax(&.{1});
    try graph.infer(y);

    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(@as(usize, 1), ir.step_count);
    try testing.expectEqual(@as(usize, 11), ir.op_count);
    try testing.expectEqual(@as(usize, 10), ir.node_step_count);
    try testing.expectEqual(@as(usize, 1), ir.fusion_step_count);
    try testing.expectEqual(@as(usize, 1), ir.log_softmax_step_count);
    try testing.expect(ir.input_edge_count > ir.step_count);
    try testing.expect(ir.normalized_step_hash != 0);
    try testing.expect(ir.normalized_shape_hash != 0);
    try testing.expectEqual(@as(usize, 1), state.compiled_ops.len);
    try testing.expectEqual(backend_mod.DeviceOp.logsoftmax, std.meta.activeTag(state.compiled_ops[0]));
    try testing.expectEqual(@as(u32, 1), state.compiled_ops[0].logsoftmax.rows);
    try testing.expectEqual(@as(u32, 3), state.compiled_ops[0].logsoftmax.cols);
}

test "DeviceInference keeps log softmax fusion sub-ops without native row op support" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const y = x.logSoftmax(&.{1});
    try graph.infer(y);

    var state = TestBackendState{};
    var be = testBackend(&state);
    be.capabilities.logsoftmax = false;
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = be,
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const ir = program.inspect().ir;
    try testing.expectEqual(@as(usize, 1), ir.log_softmax_step_count);
    try testing.expectEqual(@as(usize, 10), state.compiled_ops.len);
    try testing.expectEqual(backend_mod.DeviceOp.reduce, std.meta.activeTag(state.compiled_ops[0]));
    try testing.expectEqual(backend_mod.DeviceOp.elementwise, std.meta.activeTag(state.compiled_ops[state.compiled_ops.len - 1]));
}

test "DeviceInference uploads persistent bindings without executing" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    var session = try program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer session.deinit();

    try testing.expectEqual(@as(usize, 1), state.compile_calls);
    try testing.expectEqual(@as(usize, 1), state.compiled_initial_upload_count);
    try testing.expectEqual(@as(usize, 1), state.upload_calls);
    try testing.expectEqual(@as(usize, 1), state.last_upload_count);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);
    try testing.expectEqual(@as(usize, 0), state.patch_calls);
}

test "DeviceInference requires step bindings to cover exact tensor spans" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    var short_input = [_]f32{1};
    const short_input_bindings = [_]DeviceF32.TensorBinding{DeviceF32.TensorBinding.host(x, &short_input, short_input.len)};
    try testing.expectError(error.ShapeMismatch, program.bind(.{
        .input_tensors = &.{x},
        .input_bindings = &short_input_bindings,
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));

    var short_output = [_]f32{0};
    try testing.expectError(error.ShapeMismatch, program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &short_output,
        .output_len = short_output.len,
    }));

    const short_input_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 42,
        .byte_len = @intCast(x.data.len * @sizeOf(f32)),
    };
    const short_resource_inputs = [_]DeviceF32.TensorBinding{DeviceF32.TensorBinding.externalResource(x, short_input_resource, x.data.len - 1)};
    try testing.expectError(error.ShapeMismatch, program.bind(.{
        .input_tensors = &.{x},
        .input_bindings = &short_resource_inputs,
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));

    const short_output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 43,
        .byte_len = @intCast(y.data.len * @sizeOf(f32)),
    };
    try testing.expectError(error.ShapeMismatch, program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, short_output_resource, y.data.len - 1),
    }));
}

test "DeviceInference rejects external resource bindings on host-only backends" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const program_inspection = program.inspect();
    try testing.expectEqual(backend_mod.Device.cpu, program_inspection.backend);
    try testing.expect(program_inspection.execution_supported);
    try testing.expect(!program_inspection.external_resources_supported);

    const resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 99,
        .byte_len = @intCast(w.data.len * @sizeOf(f32)),
    };
    const bindings = [_]DeviceF32.TensorBinding{DeviceF32.TensorBinding.externalResource(w, resource, w.data.len)};
    const output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 100,
        .byte_len = @intCast(out.len * @sizeOf(f32)),
    };
    try testing.expectError(error.UnsupportedResourceBinding, program.bind(.{
        .input_tensors = &.{x},
        .persistent_bindings = &bindings,
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, output_resource, out.len),
    }));
    try testing.expectEqual(@as(usize, 1), state.bind_calls);
    try testing.expectEqual(@as(usize, 1), state.free_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.upload_calls);
}

test "DeviceInference passes external resource bindings to resource-capable backends" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testResourceBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const program_inspection = program.inspect();
    try testing.expectEqual(backend_mod.Device.webgpu, program_inspection.backend);
    try testing.expect(program_inspection.execution_supported);
    try testing.expect(program_inspection.external_resources_supported);
    try testing.expect(program_inspection.command_shape.command_count > 0);
    try testing.expect(program_inspection.command_shape.command_stencil_hash != 0);
    program.resetRuntimeProfile();
    const reset_program_inspection = program.inspect();
    try testing.expectEqual(program_inspection.runtime_patch_shape, reset_program_inspection.runtime_patch_shape);
    try testing.expectEqual(program_inspection.command_shape, reset_program_inspection.command_shape);

    const weights_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 99,
        .byte_offset = 16,
        .byte_len = 16 + @as(u32, @intCast(w.data.len * @sizeOf(f32))),
    };
    const input_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 100,
        .byte_offset = 4,
        .byte_len = 4 + @as(u32, @intCast(x.data.len * @sizeOf(f32))),
    };
    const output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 101,
        .byte_offset = 8,
        .byte_len = 8 + @as(u32, @intCast(out.len * @sizeOf(f32))),
    };

    var wrong_output_resource = output_resource;
    wrong_output_resource.placement = .metal;
    try testing.expectError(error.UnsupportedResourceBinding, program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, wrong_output_resource, out.len),
    }));
    try testing.expectEqual(@as(usize, 1), state.bind_calls);
    try testing.expectEqual(@as(usize, 1), state.free_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.configure_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.upload_calls);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);

    var write_only_weights_resource = weights_resource;
    write_only_weights_resource.access = .write_only;
    const unreadable_persistent_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(w, write_only_weights_resource, w.data.len),
    };
    try testing.expectError(error.UnsupportedResourceBinding, program.bind(.{
        .input_tensors = &.{x},
        .persistent_bindings = &unreadable_persistent_bindings,
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));
    try testing.expectEqual(@as(usize, 2), state.bind_calls);
    try testing.expectEqual(@as(usize, 2), state.free_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.configure_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.upload_calls);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);

    var write_only_input_resource = input_resource;
    write_only_input_resource.access = .write_only;
    const unreadable_input_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(x, write_only_input_resource, x.data.len),
    };
    try testing.expectError(error.UnsupportedResourceBinding, program.bind(.{
        .input_tensors = &.{x},
        .input_bindings = &unreadable_input_bindings,
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));
    try testing.expectEqual(@as(usize, 3), state.bind_calls);
    try testing.expectEqual(@as(usize, 3), state.free_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.configure_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.upload_calls);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);

    var read_only_output_resource = output_resource;
    read_only_output_resource.access = .read_only;
    try testing.expectError(error.UnsupportedResourceBinding, program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, read_only_output_resource, out.len),
    }));
    try testing.expectEqual(@as(usize, 4), state.bind_calls);
    try testing.expectEqual(@as(usize, 4), state.free_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.configure_binding_calls);
    try testing.expectEqual(@as(usize, 0), state.upload_calls);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);

    const persistent_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(w, weights_resource, w.data.len),
    };
    const input_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(x, input_resource, x.data.len),
    };
    var session = try program.bind(.{
        .input_tensors = &.{x},
        .input_bindings = &input_bindings,
        .persistent_bindings = &persistent_bindings,
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, output_resource, out.len),
    });
    defer session.deinit();

    try testing.expectEqual(@as(usize, 5), state.bind_calls);
    try testing.expectEqual(@as(usize, 1), state.configure_binding_calls);
    try testing.expectEqual(@as(usize, 1), state.configured_persistent_count);
    try testing.expectEqual(@as(usize, 1), state.configured_input_count);
    try testing.expectEqual(@as(usize, 1), state.configured_output_count);
    try testing.expectEqual(@as(usize, 1), state.configured_persistent.len);
    try testing.expectEqual(@as(usize, 1), state.configured_inputs.len);
    try testing.expectEqual(@as(usize, 1), state.configured_outputs.len);
    try testing.expect(!state.configured_persistent[0].isHost());
    try testing.expect(!state.configured_inputs[0].isHost());
    try testing.expect(!state.configured_outputs[0].isHost());
    try testing.expectEqual(@as(usize, 99), state.configured_persistent[0].resource.?.handle);
    try testing.expectEqual(@as(usize, 100), state.configured_inputs[0].resource.?.handle);
    try testing.expectEqual(@as(usize, 101), state.configured_outputs[0].resource.?.handle);
    try testing.expectEqual(@as(usize, 1), state.upload_calls);
    try testing.expectEqual(@as(usize, 1), state.last_upload_count);
    try testing.expectEqual(@as(usize, 1), state.last_uploads.len);
    try testing.expect(!state.last_uploads[0].isHost());
    try testing.expect(state.last_uploads[0].hostSlice() == null);
    const uploaded = state.last_uploads[0].resource.?;
    try testing.expectEqual(backend_mod.Device.webgpu, uploaded.placement);
    try testing.expectEqual(@as(usize, 99), uploaded.handle);
    try testing.expectEqual(@as(u32, 16), uploaded.byte_offset);
    try testing.expectEqual(weights_resource.byte_len, uploaded.byte_len);
    try testing.expectEqual(@as(u32, 0), state.last_uploads[0].offset);
    try testing.expectEqual(@as(u32, @intCast(w.data.len * @sizeOf(f32))), state.last_uploads[0].size);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);

    const session_inspection = session.inspect();
    try testing.expectEqual(backend_mod.Device.webgpu, session_inspection.backend);
    try testing.expectEqual(DeviceF32.BindingStorage.external_resource, session_inspection.output_storage);
    try testing.expectEqual(@as(usize, 1), session_inspection.persistent_binding_count);
    try testing.expectEqual(@as(usize, 1), session_inspection.step_input_count);
    try testing.expectEqual(@as(usize, 1), session_inspection.step_output_count);
    try testing.expectEqual(@as(usize, 0), session_inspection.host_binding_count);
    try testing.expectEqual(@as(usize, 3), session_inspection.resource_binding_count);
    try testing.expect(session_inspection.binding_shape_hash != 0);

    var write_only_upload_resource = weights_resource;
    write_only_upload_resource.access = .write_only;
    const original_persistent = session.bindings.persistent_inputs[0];
    session.bindings.persistent_inputs[0] = try program.tensorResourceBindingIO(w, write_only_upload_resource, w.data.len);
    try testing.expectError(error.UnsupportedResourceBinding, program.uploadPersistentInputs(&session));
    try testing.expectEqual(@as(usize, 1), state.upload_calls);
    session.bindings.persistent_inputs[0] = original_persistent;

    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(3, 1) });
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(u32, 3), state.patched_window.position);
    try testing.expectEqual(@as(u32, 1), state.patched_window.len);
    try testing.expectEqual(@as(usize, 1), state.execute_calls);
    try testing.expectEqual(@as(usize, 1), state.configured_execute_binding_calls);
    try testing.expectEqual(@as(usize, 1), state.last_input_count);
    try testing.expectEqual(@as(usize, 1), state.last_output_count);
    try testing.expectEqual(@as(usize, 1), state.last_inputs.len);
    try testing.expectEqual(@as(usize, 1), state.last_outputs.len);
    try testing.expect(!state.last_inputs[0].isHost());
    try testing.expect(!state.last_outputs[0].isHost());

    const executed_input = state.last_inputs[0].resource.?;
    try testing.expectEqual(@as(usize, 100), executed_input.handle);
    try testing.expectEqual(@as(u32, 4), executed_input.byte_offset);
    try testing.expectEqual(input_resource.byte_len, executed_input.byte_len);
    try testing.expectEqual(@as(u32, @intCast(x.data.len * @sizeOf(f32))), state.last_inputs[0].size);

    const executed_output = state.last_outputs[0].resource.?;
    try testing.expectEqual(@as(usize, 101), executed_output.handle);
    try testing.expectEqual(@as(u32, 8), executed_output.byte_offset);
    try testing.expectEqual(output_resource.byte_len, executed_output.byte_len);
    try testing.expectEqual(@as(u32, @intCast(out.len * @sizeOf(f32))), state.last_outputs[0].size);

    var session_profile = profile.RuntimeProfile{};
    session.addRuntimeProfileTo(&session_profile);
    try testing.expectEqual(@as(u32, 1), session_profile.call_count);
    try testing.expectEqual(program_inspection.command_shape, session_profile.program_command_shape);
    try testing.expectEqual(@as(u64, 1), session_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 1), session_profile.backend_op_count);
    try testing.expectEqual(@as(u64, 1), session_profile.runtime_patch_call_count);
    try testing.expectEqual(@as(u64, 1), session_profile.runtime_patch_changed_count);
    const op_command = @intFromEnum(program_mod.ProgramCommandKind.op);
    try testing.expectEqual(@as(u64, program_inspection.command_shape.command_kind_counts[op_command]), session_profile.program_command_counts[op_command]);
    try testing.expectEqual(@as(u64, program_inspection.command_shape.command_kind_counts[op_command]), session_profile.program_command_dispatch_counts[op_command]);
    try testing.expectEqual(@as(u64, program_inspection.command_shape.command_kind_counts[op_command]), session_profile.program_command_attempt_counts[op_command]);

    var program_profile = profile.RuntimeProfile{};
    program.addRuntimeProfileTo(&program_profile);
    try testing.expectEqual(program_inspection.command_shape, program_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 0), program_profile.call_count);
    try testing.expectEqual(@as(u64, 0), program_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), program_profile.runtime_patch_call_count);

    session.resetRuntimeProfile();
    const reset_session_inspection = session.inspect();
    try testing.expectEqual(session_inspection.binding_shape_hash, reset_session_inspection.binding_shape_hash);
    session_profile = .{};
    session.addRuntimeProfileTo(&session_profile);
    try testing.expectEqual(program_inspection.command_shape, session_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 0), session_profile.call_count);
    try testing.expectEqual(@as(u64, 0), session_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), session_profile.runtime_patch_call_count);

    var write_only_dynamic_input_resource = input_resource;
    write_only_dynamic_input_resource.access = .write_only;
    const original_input = session.bindings.step_inputs[0];
    session.bindings.step_inputs[0] = try program.tensorResourceBindingIO(x, write_only_dynamic_input_resource, x.data.len);
    try testing.expectError(error.UnsupportedResourceBinding, program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(4, 1), .dynamic_io = true }));
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(usize, 1), state.execute_calls);
    session.bindings.step_inputs[0] = original_input;

    var read_only_dynamic_output_resource = output_resource;
    read_only_dynamic_output_resource.access = .read_only;
    const original_output_resource = session.bindings.step_outputs[0];
    session.bindings.step_outputs[0] = try program.tensorResourceBindingIO(y, read_only_dynamic_output_resource, out.len);
    try testing.expectError(error.UnsupportedResourceBinding, program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(4, 1), .dynamic_io = true }));
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(usize, 1), state.execute_calls);
    session.bindings.step_outputs[0] = original_output_resource;

    var dynamic_out = [_]f32{0} ** 2;
    const original_output = session.bindings.step_outputs[0];
    session.bindings.step_outputs[0] = try program.tensorBindingIO(y, &dynamic_out, dynamic_out.len);
    defer session.bindings.step_outputs[0] = original_output;
    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(4, 1), .dynamic_io = true });
    try testing.expectEqual(@as(usize, 2), state.execute_calls);
    try testing.expectEqual(@as(usize, 1), state.configured_execute_binding_calls);
    try testing.expectEqual(@as(usize, 1), state.last_output_count);
    try testing.expect(state.last_outputs[0].isHost());
    try testing.expectEqual(@as(u32, @intCast(dynamic_out.len * @sizeOf(f32))), state.last_outputs[0].size);
}

test "DeviceInference refuses to execute WebGPU compile-only resource sessions" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var webgpu_plan = @import("backend/stencil.zig").StencilBackend.webgpuResourceSessionProbe();
    const be = webgpu_plan.backend();
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = be,
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    const program_inspection = program.inspect();
    try testing.expectEqual(backend_mod.Device.webgpu, program_inspection.backend);
    try testing.expect(!program_inspection.execution_supported);
    try testing.expect(program_inspection.external_resources_supported);
    try testing.expect(program_inspection.command_shape.command_count > 0);

    const weights_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 201,
        .byte_len = @intCast(w.data.len * @sizeOf(f32)),
        .access = .read_only,
    };
    const input_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 202,
        .byte_len = @intCast(x.data.len * @sizeOf(f32)),
        .access = .read_only,
    };
    const output_resource = backend_mod.ProgramIO.ExternalResource{
        .placement = .webgpu,
        .handle = 203,
        .byte_len = @intCast(y.data.len * @sizeOf(f32)),
        .access = .write_only,
    };
    const persistent_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(w, weights_resource, w.data.len),
    };
    const input_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.externalResource(x, input_resource, x.data.len),
    };
    var out = [_]f32{0} ** 2;
    var session = try program.bind(.{
        .input_tensors = &.{x},
        .input_bindings = &input_bindings,
        .persistent_bindings = &persistent_bindings,
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .output_binding = DeviceF32.TensorBinding.externalResource(y, output_resource, y.data.len),
    });
    defer session.deinit();

    const session_inspection = session.inspect();
    try testing.expectEqual(backend_mod.Device.webgpu, session_inspection.backend);
    try testing.expectEqual(DeviceF32.BindingStorage.external_resource, session_inspection.output_storage);
    try testing.expectEqual(@as(usize, 3), session_inspection.resource_binding_count);
    try testing.expect(session_inspection.binding_shape_hash != 0);

    try testing.expectError(error.ExecutionUnsupported, program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(3, 1) }));

    var session_profile = profile.RuntimeProfile{};
    session.addRuntimeProfileTo(&session_profile);
    try testing.expectEqual(@as(u32, 0), session_profile.call_count);
    try testing.expectEqual(@as(u64, 0), session_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), session_profile.runtime_patch_call_count);
}

test "DeviceInference can refresh a persistent binding range" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const cache_like = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var state = TestBackendState{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{ w, cache_like },
        .output_tensors = &.{y},
    });
    defer program.deinit();

    var session = try program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{ w, cache_like },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer session.deinit();

    try testing.expectEqual(@as(usize, 1), state.upload_calls);
    try testing.expectEqual(@as(usize, 2), state.last_upload_count);

    try program.uploadPersistentInputRange(&session, 1, 1);
    try testing.expectEqual(@as(usize, 2), state.upload_calls);
    try testing.expectEqual(@as(usize, 1), state.last_upload_count);
    try testing.expectError(error.InvalidProgramIO, program.uploadPersistentInputRange(&session, 2, 2));
}

test "DeviceInference persistent bindings can be rebound on CPU" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 1 });
    const w = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.mm(w);
    try graph.infer(y);

    var out = [_]f32{0} ** 2;
    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = cpu.backend(),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensors = &.{y},
    });
    defer program.deinit();

    x.setData(&.{ 2, 3 });
    w.setData(&.{ 1, 0, 0, 1 });
    var session = try program.bind(.{
        .input_tensors = &.{x},
        .persistent_tensors = &.{w},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer session.deinit();

    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(0, 0) });
    try testing.expectEqualSlices(f32, &.{ 2, 3 }, &out);

    w.setData(&.{ 2, 0, 0, 4 });
    try program.uploadPersistentInputs(&session);
    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(0, 0) });
    try testing.expectEqualSlices(f32, &.{ 4, 12 }, &out);
}

test "DeviceInference CPU executable hot path does not allocate" {
    const DeviceF32 = DeviceInference(f32);
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    var graph = ComputeGraphF32.init(alloc);
    defer graph.deinit();
    const a = graph.allocator();

    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src_full = try TensorF32.init(a, &.{ 12, 2 });
    const src = src_full.sliceRows(4, 8);
    const y = cache.sliceAssign(src, 3);
    try graph.infer(y);

    @memset(cache.data, 0);
    @memset(src_full.data, 7);
    var out = [_]f32{-1} ** 32;
    var cpu = @import("backend/cpu.zig").CpuBackend.init(alloc);
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = cpu.backend(),
        .alloc = alloc,
        .input_tensors = &.{src_full},
        .output_tensors = &.{y},
        .expected_runtime_patch_shape = backend_mod.RuntimePatchShape.expectedDynamic(1, 0),
    });
    defer program.deinit();

    var session = try program.bind(.{
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer session.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(5, 1) });
    @memset(src_full.data, 9);
    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(1, 1), .download_outputs = false });
    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(1, 1) });

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);

    const sevens = [_]f32{7} ** 8;
    const nines = [_]f32{9} ** 8;
    try testing.expectEqualSlices(f32, &sevens, out[20..28]);
    try testing.expectEqualSlices(f32, &nines, out[4..12]);
}

test "DeviceInference CPU runtime profiles are session local" {
    const DeviceF32 = DeviceInference(f32);
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src_full = try TensorF32.init(a, &.{ 12, 2 });
    const src = src_full.sliceRows(4, 8);
    const y = cache.sliceAssign(src, 3);
    try graph.infer(y);

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = cpu.backend(),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensors = &.{y},
        .expected_runtime_patch_shape = backend_mod.RuntimePatchShape.expectedDynamic(1, 0),
    });
    defer program.deinit();

    var out_a = [_]f32{0} ** 32;
    var session_a = try program.bind(.{
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out_a,
        .output_len = out_a.len,
    });
    defer session_a.deinit();

    var out_b = [_]f32{0} ** 32;
    var session_b = try program.bind(.{
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out_b,
        .output_len = out_b.len,
    });
    defer session_b.deinit();

    try program.executeStep(&session_a, .{ .window = try backend_mod.RuntimeWindow.init(1, 1) });
    try program.executeStep(&session_b, .{ .window = try backend_mod.RuntimeWindow.init(2, 1) });
    try program.executeStep(&session_b, .{ .window = try backend_mod.RuntimeWindow.init(3, 1) });

    var profile_a = profile.RuntimeProfile{};
    session_a.addRuntimeProfileTo(&profile_a);
    var profile_b = profile.RuntimeProfile{};
    session_b.addRuntimeProfileTo(&profile_b);
    var profile_program = profile.RuntimeProfile{};
    program.addRuntimeProfileTo(&profile_program);

    try testing.expectEqual(@as(u32, 1), profile_a.call_count);
    try testing.expectEqual(@as(u64, 1), profile_a.runtime_patch_call_count);
    try testing.expectEqual(@as(u32, 2), profile_b.call_count);
    try testing.expectEqual(@as(u64, 2), profile_b.runtime_patch_call_count);
    try testing.expectEqual(@as(u32, 0), profile_program.call_count);
    try testing.expectEqual(@as(u64, 0), profile_program.runtime_patch_call_count);
    try testing.expect(profile_a.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try testing.expectEqual(profile_a.runtime_patch_shape, profile_b.runtime_patch_shape);
    try testing.expectEqual(profile_a.runtime_patch_shape, profile_program.runtime_patch_shape);

    session_a.resetRuntimeProfile();
    profile_a = .{};
    session_a.addRuntimeProfileTo(&profile_a);
    profile_b = .{};
    session_b.addRuntimeProfileTo(&profile_b);
    try testing.expectEqual(@as(u32, 0), profile_a.call_count);
    try testing.expectEqual(@as(u64, 0), profile_a.runtime_patch_call_count);
    try testing.expectEqual(@as(u32, 2), profile_b.call_count);
    try testing.expectEqual(@as(u64, 2), profile_b.runtime_patch_call_count);
}

test "DeviceInference rejects invalid runtime patches before execution" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.relu();
    try graph.infer(y);

    var out = [_]f32{0} ** 4;
    var state = TestBackendState{ .patch_status = .invalid };
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    try testing.expectError(error.InvalidRuntimePatch, dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(4, 1) }));
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(usize, 0), state.execute_calls);
}

test "DeviceInference rejects runtime patch shape mismatch at compile seam" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 2, 2 });
    const y = x.relu();
    try graph.infer(y);

    var out = [_]f32{0} ** 4;
    var state = TestBackendState{};
    try testing.expectError(error.RuntimePatchShapeMismatch, DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .expected_runtime_patch_shape = .{ .runtime_patch_holes = 1 },
    }));
    try testing.expectEqual(@as(usize, 1), state.compile_calls);
}

test "DeviceInference requires matching stencil hash evidence for expected runtime patches" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src_full = try TensorF32.init(a, &.{ 12, 2 });
    const src = src_full.sliceRows(4, 8);
    const y = cache.sliceAssign(src, 3);
    try graph.infer(y);

    var out = [_]f32{0} ** 32;
    const expected_dynamic = backend_mod.RuntimePatchShape.expectedDynamic(1, 0);
    var shaped = TestBackendState{};
    var shaped_dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&shaped),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .expected_runtime_patch_shape = expected_dynamic,
    });
    defer shaped_dev.deinit();

    const observed = shaped_dev.program.inspect().runtime_patch_shape;
    try testing.expectEqual(@as(u64, 1), observed.runtime_patch_holes);
    try testing.expectEqual(@as(u64, 1), observed.runtime_patch_cache_write_pos_holes);
    try testing.expect(observed.runtime_patch_stencil_hash != 0);

    var wrong_hash = TestBackendState{};
    var mismatched = observed;
    mismatched.runtime_patch_stencil_hash +%= 1;
    try testing.expectError(error.RuntimePatchShapeMismatch, DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&wrong_hash),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .expected_runtime_patch_shape = mismatched,
    }));

    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = cpu.backend(),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
        .expected_runtime_patch_shape = expected_dynamic,
    });
    defer dev.deinit();
}

test "DeviceInference aliases dense views to base buffers" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 4 });
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    const view = x.sliceColumns(1, 3);
    const y = view.relu();
    try graph.infer(y);

    var out = [_]f32{0} ** 8;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const ir = dev.program.inspect().ir;
    try testing.expectEqual(@as(usize, 1), ir.structural_step_count);
    try testing.expectEqual(@as(usize, 1), ir.value_step_count);

    const ops = state.compiled_ops;
    try testing.expectEqual(@as(usize, 1), ops.len);
    try testing.expectEqual(dev.session.bindings.step_inputs[0].buf_idx, ops[0].elementwise.src0);
    try testing.expectEqual(ops[0].elementwise.src0, ops[0].elementwise.src1);
    try testing.expect(dev.session.bindings.step_inputs[0].buf_idx != ops[0].elementwise.dst);
    try testing.expectEqual(@as(u32, 4), ops[0].elementwise.src0_offset);
    try testing.expectEqual(@as(u32, 4), ops[0].elementwise.src1_offset);
    try testing.expectEqual(@as(u32, 0), ops[0].elementwise.dst_offset);
}

test "DeviceInference downloads view-only contiguous output span" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 4 });
    const y = x.sliceColumns(1, 3);
    try graph.infer(y);

    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    var out = [_]f32{0} ** 8;
    var cpu = @import("backend/cpu.zig").CpuBackend{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = cpu.backend(),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(0, 0) });
    try testing.expectEqualSlices(f32, &.{ 4, 5, 6, 7, 8, 9, 10, 11 }, &out);
}

test "DeviceInference lowers broadcasted binary add through explicit repeat" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 3, 2 });
    const bias = try TensorF32.init(a, &.{3});
    const y = x.add(bias);
    try graph.infer(y);

    var out = [_]f32{0} ** 6;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .persistent_tensors = &.{bias},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const ops = state.compiled_ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    const repeat_op = switch (ops[0]) {
        .repeat => |rp| rp,
        else => return error.TestExpectedEqual,
    };
    try testing.expectEqual(dev.session.bindings.persistent_inputs[0].buf_idx, repeat_op.src);
    try testing.expectEqual(dev.session.bindings.step_outputs[0].buf_idx, repeat_op.dst);
    try testing.expectEqual(@as(u32, 6), repeat_op.n);

    const add_op = switch (ops[1]) {
        .elementwise => |ew| ew,
        else => return error.TestExpectedEqual,
    };
    try testing.expectEqual(@import("op.zig").Op.add, add_op.op);
    try testing.expectEqual(dev.session.bindings.step_inputs[0].buf_idx, add_op.src0);
    try testing.expectEqual(repeat_op.dst, add_op.src1);
    try testing.expectEqual(repeat_op.dst, add_op.dst);
    try testing.expectEqual(@as(u32, 6), add_op.n);
}

test "DeviceInference lowers prefill slice_assign with 2D strides" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src_full = try TensorF32.init(a, &.{ 12, 2 });
    const src = src_full.sliceRows(4, 8);
    const y = cache.sliceAssign(src, 3);
    try graph.infer(y);

    var out = [_]f32{0} ** 32;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const ops = state.compiled_ops;
    try testing.expectEqual(@as(usize, 1), ops.len);
    const sa = ops[0].slice_assign;
    try testing.expectEqual(@as(u32, 4), sa.rows);
    try testing.expectEqual(@as(u32, 2), sa.cols);
    try testing.expectEqual(@as(u32, 12), sa.dst_offset);
    try testing.expectEqual(@as(u32, 1), sa.dst_row_stride);
    try testing.expectEqual(@as(u32, 4), sa.dst_col_stride);
    try testing.expectEqual(@as(u32, 4), sa.src_offset);
    try testing.expectEqual(@as(u32, 1), sa.src_row_stride);
    try testing.expectEqual(@as(u32, 12), sa.src_col_stride);
    try testing.expectEqual(@as(u32, 4), sa.patch_stride);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(5, 1) });
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(u32, 5), state.patched_window.position);
    try testing.expectEqual(@as(u32, 12), state.compiled_ops[0].slice_assign.dst_offset);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(5, 1) });
    try testing.expectEqual(@as(usize, 2), state.patch_calls);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(5, 1) });
    try testing.expectEqual(@as(usize, 3), state.patch_calls);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(5, 2) });
    try testing.expectEqual(@as(usize, 4), state.patch_calls);
    try testing.expectEqual(@as(usize, 1), state.last_output_count);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(6, 1), .download_outputs = false });
    try testing.expectEqual(@as(usize, 5), state.patch_calls);
    try testing.expectEqual(@as(usize, 0), state.last_output_count);
}

test "DeviceInference keeps slice_assign_rows static during position patch" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const dst = try TensorF32.init(a, &.{ 8, 2 });
    const src_full = try TensorF32.init(a, &.{ 8, 2 });
    const src = src_full.sliceRows(2, 6);
    const y = dst.sliceAssignRows(src, 2);
    try graph.infer(y);

    var out = [_]f32{0} ** 16;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const before = state.compiled_ops[0].slice_assign.dst_offset;
    try testing.expectEqual(@as(u32, 0), state.compiled_ops[0].slice_assign.patch_stride);
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(5, 1) });
    try testing.expectEqual(before, state.compiled_ops[0].slice_assign.dst_offset);
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(u32, 5), state.patched_window.position);
}

test "DeviceInference lowers batched attention geometry" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const q = try TensorF32.init(a, &.{ 4, 2 });
    const k_full = try TensorF32.init(a, &.{ 8, 8 });
    const v_full = try TensorF32.init(a, &.{ 8, 8 });
    const k = k_full.sliceRows(2, 6);
    const v = v_full.sliceRows(2, 6);
    const mask = try TensorF32.init(a, &.{ 8, 2 });
    const y = q.attention(k, v, mask, 0.5);
    try graph.infer(y);

    var out = [_]f32{0} ** 8;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{ q, k_full, v_full, mask },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const ops = state.compiled_ops;
    try testing.expectEqual(@as(usize, 1), ops.len);
    const att = ops[0].attention;
    try testing.expect(att.has_mask);
    try testing.expect(!att.patch_seq_kv);
    try testing.expectEqual(@as(u32, 4), att.d_head);
    try testing.expectEqual(@as(u32, 2), att.seq_q);
    try testing.expectEqual(@as(u32, 8), att.seq_kv);
    try testing.expectEqual(@as(u32, 2), att.k_off);
    try testing.expectEqual(@as(u32, 1), att.k_rs);
    try testing.expectEqual(@as(u32, 8), att.k_cs);
    try testing.expectEqual(@as(u32, 8), att.mask_cs);

    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(0, 5) });
    try testing.expectEqual(@as(usize, 1), state.patch_calls);
    try testing.expectEqual(@as(?u32, 5), state.patched_window.attentionSeqKv());
    try dev.executeStep(.{ .window = try backend_mod.RuntimeWindow.init(4, 1) });
    try testing.expectEqual(@as(u32, 8), state.compiled_ops[0].attention.seq_kv);
    try testing.expectEqual(@as(usize, 2), state.patch_calls);
}

test "DeviceInference marks cached attention length as a runtime patch hole" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const q = try TensorF32.init(a, &.{ 4, 1 });
    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src = try TensorF32.init(a, &.{ 4, 1 });
    const write = cache.sliceAssign(src, 0);
    const y = q.attention(write, write, null, 0.5);
    try graph.infer(y);

    var out = [_]f32{0} ** 4;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{ q, src },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const inspection = dev.program.inspect();
    try testing.expectEqual(@as(?u32, 7), inspection.runtime_patch_envelope.max_cache_write_pos);
    try testing.expectEqual(@as(?u32, 8), inspection.runtime_patch_envelope.max_attention_seq_kv);

    var cache_patch_found = false;
    var found_attention = false;
    var attention_patch_found = false;
    for (state.compiled_ops) |op| switch (op) {
        .slice_assign => |sa| {
            if (sa.patch_stride != 0) cache_patch_found = true;
        },
        .attention => |att| {
            found_attention = true;
            if (att.patch_seq_kv) attention_patch_found = true;
        },
        else => {},
    };
    try testing.expect(cache_patch_found);
    try testing.expect(found_attention);
    try testing.expect(attention_patch_found);
}

test "DeviceInference rejects unsupported graph ops" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 3 });
    const idx = try TensorF32.initIndexVectorCopy(a, &.{ 2, 0, 1 });
    const y = x.pickRows(idx);
    try graph.infer(y);

    var out = [_]f32{0} ** 3;
    var state = TestBackendState{};
    try testing.expectError(error.UnsupportedDeviceOp, DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{ x, idx },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));
    try testing.expectEqual(@as(usize, 0), state.compile_calls);
}
