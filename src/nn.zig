//! Native neural-network helpers that decompose into primitive tensor ops.

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
const max_dims = tensor_mod.max_dims;
const Alloc = std.mem.Allocator;
const backend_mod = @import("backend.zig");
const backend_program_mod = @import("backend/program.zig");
const CpuBackend = @import("backend/cpu.zig").CpuBackend;
const DeviceInference = @import("device_inference.zig").DeviceInference;
const loss_mod = @import("loss.zig");
const profile_mod = @import("profile.zig");
const tensor_program_ir = @import("tensor_program_ir.zig");

const linear_weight_layout = "feature-major:linear.weight[out_features,in_features]";
const linear_bias_layout = "feature-major:linear.bias[out_features]";
const embedding_weight_layout = "feature-major:embedding.weight[embedding_dim,num_embeddings]";
const layer_norm_weight_layout = "feature-major:layer_norm.weight[features]";
const layer_norm_bias_layout = "feature-major:layer_norm.bias[features]";
const rms_norm_weight_layout = "feature-major:rms_norm.weight[features]";

pub const native_helper_manifest = .{
    .kind = "zgml-native-helper",
    .domain = "nn",
    .product_frontend = false,
    .product_owner = "src/ts/**",
    .product_policy = "forbidden",
    .role = "native-helper-substrate",
    .alignment_boundary = "Program/Session/ABI contracts",
};

pub const LinearConfig = struct {
    bias: bool = true,
    seed: u64 = 0,
};

pub const EmbeddingConfig = struct {
    seed: u64 = 0,
};

pub const LayerNormConfig = struct {
    eps: f64 = 1e-5,
    affine: bool = true,
    bias: bool = true,
};

pub const RmsNormConfig = struct {
    eps: f64 = 1e-5,
    affine: bool = true,
};

pub const DropoutConfig = struct {
    seed: u64 = 0,
    training: bool = true,
};

pub const DropoutError = error{
    InvalidDropoutProbability,
};

pub fn NamedParameter(comptime T: type) type {
    return struct {
        name: []const u8,
        tensor: *Tensor(T),
        layout: ?[]const u8 = null,
    };
}

const ParameterMetadata = struct {
    name: []u8,
    layout: ?[]u8 = null,
};

pub const ModuleParameterInfo = struct {
    name: []const u8,
    layout: ?[]const u8 = null,
    n_dims: u8,
    ne: [max_dims]usize,
    len: usize,

    pub fn shape(self: *const ModuleParameterInfo) []const usize {
        return self.ne[0..@as(usize, self.n_dims)];
    }
};

pub const ModuleCompatibilityCode = enum {
    compatible,
    unsupported_module,
    native_path_mismatch,
    model_kind_mismatch,
    layer_count_mismatch,
    input_shape_mismatch,
    output_shape_mismatch,
    input_len_mismatch,
    output_len_mismatch,
    weights_len_mismatch,
    bias_len_mismatch,
    parameter_len_mismatch,
    module_analysis_failed,
    ir_mismatch,
    missing_parameter,
    unexpected_parameter,
    shape_mismatch,
    layout_mismatch,
    unsupported_tensor_layout,
};

pub const ModuleCompatibility = struct {
    compatible: bool,
    code: ModuleCompatibilityCode,
    reason: ?[]const u8 = null,
    parameter_index: ?usize = null,
    parameter_name: ?[]const u8 = null,
    expected_shape: ?[]const usize = null,
    actual_shape: ?[]const usize = null,
    expected_layout: ?[]const u8 = null,
    actual_layout: ?[]const u8 = null,
    expected_native_path: ?[]const u8 = null,
    actual_native_path: ?[]const u8 = null,
    expected_model_kind: ?[]const u8 = null,
    actual_model_kind: ?[]const u8 = null,
    expected_layer_count: ?usize = null,
    actual_layer_count: ?usize = null,
    expected_scalar_count: ?usize = null,
    actual_scalar_count: ?usize = null,
    expected_ir_step_hash: ?u64 = null,
    actual_ir_step_hash: ?u64 = null,
    expected_ir_shape_hash: ?u64 = null,
    actual_ir_shape_hash: ?u64 = null,

    pub fn canBind(self: ModuleCompatibility) bool {
        return self.compatible;
    }
};

pub const CompileSupport = struct {
    supported: bool,
    reason: ?[]const u8 = null,
    native_path: ?[]const u8 = null,
    model_kind: ?[]const u8 = null,
    layer_count: ?usize = null,
    input_n_dims: ?u8 = null,
    input_ne: [max_dims]usize = [_]usize{0} ** max_dims,
    output_n_dims: ?u8 = null,
    output_ne: [max_dims]usize = [_]usize{0} ** max_dims,
    input_len: ?usize = null,
    output_len: ?usize = null,
    weights_len: ?usize = null,
    bias_len: ?usize = null,
    parameter_len: ?usize = null,
    ir_op_count: ?usize = null,
    ir_normalized_step_hash: ?u64 = null,
    ir_normalized_shape_hash: ?u64 = null,
    shape_preserving: bool = false,
    composable: bool = false,

    pub fn canCompile(self: CompileSupport) bool {
        return self.supported;
    }

    pub fn inputShape(self: *const CompileSupport) ?[]const usize {
        const n_dims = self.input_n_dims orelse return null;
        return self.input_ne[0..@as(usize, n_dims)];
    }

    pub fn outputShape(self: *const CompileSupport) ?[]const usize {
        const n_dims = self.output_n_dims orelse return null;
        return self.output_ne[0..@as(usize, n_dims)];
    }
};

pub const CompileError = error{
    UnsupportedCompile,
    UnsupportedScalarType,
    InvalidInputShape,
    ShapeMismatch,
    UnsupportedTensorLayout,
};

pub const ModuleCompileOptions = struct {
    /// Optional exemplar shape. If omitted, modules with a leading Linear
    /// weight default to the exact vector tiny-linear shape `[in_features]`.
    input_shape: ?[]const usize = null,
    /// Optional explicit backend. If omitted, `ModuleProgram` owns a CPU backend.
    backend: ?backend_mod.Backend = null,
};

pub const LinearCompileOptions = ModuleCompileOptions;

fn validateModuleInputShape(input_shape: []const usize) CompileError!void {
    if (input_shape.len == 0 or input_shape.len > max_dims) return error.InvalidInputShape;
    for (input_shape) |dim| {
        if (dim == 0) return error.InvalidInputShape;
    }
}

fn requireDenseModuleTensor(comptime T: type, expected: *const Tensor(T), actual: *const Tensor(T)) CompileError!void {
    if (!actual.hasShape(expected.ne[0..expected.n_dims])) return error.ShapeMismatch;
    if (!actual.isDenseLayout()) return error.UnsupportedTensorLayout;
}

pub const StateLoadError = error{
    MissingParameter,
    UnexpectedParameter,
    ShapeMismatch,
    LayoutMismatch,
};

pub fn StateEntry(comptime T: type) type {
    return struct {
        name: []const u8,
        n_dims: u8,
        ne: [max_dims]usize,
        layout: ?[]const u8 = null,
        data: []T,

        pub fn shape(self: *const @This()) []const usize {
            return self.ne[0..@as(usize, self.n_dims)];
        }
    };
}

pub fn stateDict(comptime T: type, alloc: Alloc, named_params: []const NamedParameter(T)) Alloc.Error![]StateEntry(T) {
    const out = try alloc.alloc(StateEntry(T), named_params.len);
    var initialized: usize = 0;
    errdefer {
        freeStateEntryPayloads(T, alloc, out[0..initialized]);
        alloc.free(out);
    }

    for (named_params) |param| {
        const entry = blk: {
            const name = try alloc.dupe(u8, param.name);
            errdefer alloc.free(name);
            const data = try alloc.dupe(T, param.tensor.denseSliceConst());
            errdefer alloc.free(data);
            const layout = if (param.layout) |layout_name| try alloc.dupe(u8, layout_name) else null;
            errdefer if (layout) |owned_layout| alloc.free(owned_layout);
            break :blk StateEntry(T){
                .name = name,
                .n_dims = param.tensor.n_dims,
                .ne = param.tensor.ne,
                .layout = layout,
                .data = data,
            };
        };
        out[initialized] = entry;
        initialized += 1;
    }

    return out;
}

pub fn deinitStateDict(comptime T: type, alloc: Alloc, entries: []StateEntry(T)) void {
    freeStateEntryPayloads(T, alloc, entries);
    alloc.free(entries);
}

fn freeStateEntryPayloads(comptime T: type, alloc: Alloc, entries: []StateEntry(T)) void {
    for (entries) |entry| {
        alloc.free(entry.name);
        if (entry.layout) |layout| alloc.free(layout);
        alloc.free(entry.data);
    }
}

pub fn loadStateDict(comptime T: type, named_params: []const NamedParameter(T), entries: []const StateEntry(T)) StateLoadError!void {
    for (entries) |entry| {
        if (findNamedParameter(T, named_params, entry.name) == null) return error.UnexpectedParameter;
    }

    for (named_params) |param| {
        const entry = findStateEntry(T, entries, param.name) orelse return error.MissingParameter;
        if (!stateEntryMatchesTensor(T, entry, param.tensor)) return error.ShapeMismatch;
        if (!stateEntryMatchesLayout(T, entry, param)) return error.LayoutMismatch;
    }

    if (entries.len != named_params.len) return error.UnexpectedParameter;

    for (named_params) |param| {
        const entry = findStateEntry(T, entries, param.name).?;
        param.tensor.setData(entry.data);
    }
}

fn findStateEntry(comptime T: type, entries: []const StateEntry(T), name: []const u8) ?StateEntry(T) {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry;
    }
    return null;
}

fn findNamedParameter(comptime T: type, named_params: []const NamedParameter(T), name: []const u8) ?NamedParameter(T) {
    for (named_params) |param| {
        if (std.mem.eql(u8, param.name, name)) return param;
    }
    return null;
}

fn stateEntryMatchesTensor(comptime T: type, entry: StateEntry(T), tensor: *const Tensor(T)) bool {
    if (entry.n_dims != tensor.n_dims) return false;
    if (entry.data.len != tensor.nElems()) return false;
    return tensor.hasShape(entry.shape());
}

fn stateEntryMatchesLayout(comptime T: type, entry: StateEntry(T), param: NamedParameter(T)) bool {
    if (entry.layout == null or param.layout == null) return true;
    return std.mem.eql(u8, entry.layout.?, param.layout.?);
}

fn moduleValueType(comptime Module: type) type {
    return switch (@typeInfo(Module)) {
        .pointer => |ptr| ptr.child,
        else => Module,
    };
}

fn moduleParameters(comptime T: type, module: anytype) []const *Tensor(T) {
    const Module = @TypeOf(module);
    const Value = moduleValueType(Module);
    if (comptime @hasDecl(Value, "parameters")) {
        return switch (@typeInfo(Module)) {
            .pointer => module.parameters(),
            else => @compileError("nn.parameters requires pointers for trainable modules; pass &layer or use nn.Sequential to own value layers"),
        };
    }
    return &.{};
}

fn moduleNamedParameters(comptime T: type, module: anytype) []const NamedParameter(T) {
    const Module = @TypeOf(module);
    const Value = moduleValueType(Module);
    if (comptime @hasDecl(Value, "namedParameters")) {
        return switch (@typeInfo(Module)) {
            .pointer => module.namedParameters(),
            else => @compileError("nn.namedParameters requires pointers for trainable modules; pass &layer or use nn.Sequential to own value layers"),
        };
    }
    return &.{};
}

fn moduleCompileSupportValue(module: anytype) CompileSupport {
    const Module = @TypeOf(module);
    const Value = moduleValueType(Module);
    if (comptime @hasDecl(Value, "compileSupport")) {
        return switch (@typeInfo(Module)) {
            .pointer => module.compileSupport(),
            else => @compileError("nn.compileSupport requires pointers for module compatibility; pass &layer or use nn.Sequential to own value layers"),
        };
    }
    return unsupportedCompile("module does not expose compileSupport()");
}

fn findNamedParameterForTensor(comptime T: type, named_params: []const NamedParameter(T), tensor: *Tensor(T), preferred_index: usize) ?NamedParameter(T) {
    if (preferred_index < named_params.len and named_params[preferred_index].tensor == tensor) return named_params[preferred_index];
    for (named_params) |param| {
        if (param.tensor == tensor) return param;
    }
    return null;
}

fn copyParameterMetadata(comptime T: type, alloc: Alloc, tensors: []const *Tensor(T), named_params: []const NamedParameter(T)) Alloc.Error![]ParameterMetadata {
    const metadata = try alloc.alloc(ParameterMetadata, tensors.len);
    var initialized: usize = 0;
    errdefer {
        freeParameterMetadata(alloc, metadata[0..initialized]);
        alloc.free(metadata);
    }

    for (tensors, 0..) |tensor, index| {
        const entry = blk: {
            const source = findNamedParameterForTensor(T, named_params, tensor, index);
            const name = if (source) |param| try alloc.dupe(u8, param.name) else try std.fmt.allocPrint(alloc, "parameter.{d}", .{index});
            errdefer alloc.free(name);
            const layout = if (source) |param| blk_layout: {
                if (param.layout) |layout_name| break :blk_layout try alloc.dupe(u8, layout_name);
                break :blk_layout null;
            } else null;
            errdefer if (layout) |owned_layout| alloc.free(owned_layout);
            break :blk ParameterMetadata{ .name = name, .layout = layout };
        };
        metadata[initialized] = entry;
        initialized += 1;
    }

    return metadata;
}

fn freeParameterMetadata(alloc: Alloc, metadata: []ParameterMetadata) void {
    for (metadata) |entry| {
        alloc.free(entry.name);
        if (entry.layout) |layout| alloc.free(layout);
    }
}

fn buildModuleParameterInfo(comptime T: type, alloc: Alloc, metadata: []const ParameterMetadata, tensors: []const *Tensor(T)) Alloc.Error![]ModuleParameterInfo {
    std.debug.assert(metadata.len == tensors.len);
    const info = try alloc.alloc(ModuleParameterInfo, tensors.len);
    for (metadata, tensors, 0..) |entry, tensor, index| {
        info[index] = .{
            .name = entry.name,
            .layout = entry.layout,
            .n_dims = tensor.n_dims,
            .ne = tensor.ne,
            .len = tensor.nElems(),
        };
    }
    return info;
}

fn boxedForward(comptime T: type, comptime Module: type, ctx: *anyopaque, x: *Tensor(T)) *Tensor(T) {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime !@hasDecl(Module, "forward")) {
        @compileError("nn.Sequential layers must expose forward(x)");
    }
    return module.forward(x);
}

fn boxedParameters(comptime T: type, comptime Module: type, ctx: *anyopaque) []const *Tensor(T) {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime @hasDecl(Module, "parameters")) {
        return module.parameters();
    }
    return &.{};
}

fn boxedNamedParameters(comptime T: type, comptime Module: type, ctx: *anyopaque) []const NamedParameter(T) {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime @hasDecl(Module, "namedParameters")) {
        return module.namedParameters();
    }
    return &.{};
}

fn unsupportedCompile(reason: []const u8) CompileSupport {
    return .{ .supported = false, .reason = reason };
}

fn composableCompile(reason: []const u8) CompileSupport {
    return .{ .supported = false, .reason = reason, .shape_preserving = true, .composable = true };
}

fn deviceProgramCompileSupport(comptime T: type) CompileSupport {
    if (T != f32) {
        return unsupportedCompile("native Program compilation currently supports f32 tensors");
    }
    return .{
        .supported = true,
        .native_path = "device-program",
        .model_kind = "module",
    };
}

fn shapePreservingDeviceProgramCompileSupport(comptime T: type) CompileSupport {
    var support = deviceProgramCompileSupport(T);
    support.shape_preserving = support.supported;
    return support;
}

fn compileSupportShape1(len: usize) [max_dims]usize {
    var ne = [_]usize{0} ** max_dims;
    ne[0] = len;
    return ne;
}

fn copyCompileSupportShape(dest: *[max_dims]usize, shape: []const usize) void {
    std.debug.assert(shape.len <= max_dims);
    dest.* = [_]usize{0} ** max_dims;
    @memcpy(dest[0..shape.len], shape);
}

fn zigTinyLinearCompileSupport(comptime T: type, in_features: usize, out_features: usize, has_bias: bool) CompileSupport {
    if (T != f32) {
        return unsupportedCompile("nn.Linear native Program compilation currently supports f32 tensors");
    }
    return .{
        .supported = true,
        .native_path = "tiny-linear",
        .model_kind = "tiny-linear",
        .layer_count = 1,
        .input_n_dims = 1,
        .input_ne = compileSupportShape1(in_features),
        .output_n_dims = 1,
        .output_ne = compileSupportShape1(out_features),
        .input_len = in_features,
        .output_len = out_features,
        .weights_len = in_features * out_features,
        .bias_len = if (has_bias) out_features else 0,
        .parameter_len = in_features * out_features + if (has_bias) out_features else 0,
    };
}

fn isBiasParameter(comptime T: type, param: NamedParameter(T)) bool {
    if (param.layout) |layout| {
        if (std.mem.endsWith(u8, layout, ".bias[out_features]")) return true;
        if (std.mem.endsWith(u8, layout, ".bias[features]")) return true;
    }
    return std.mem.endsWith(u8, param.name, ".bias") or std.mem.eql(u8, param.name, "bias");
}

fn compileSupportParameterLengths(comptime T: type, named_params: []const NamedParameter(T)) struct {
    weights_len: usize,
    bias_len: usize,
    parameter_len: usize,
} {
    var weights_len: usize = 0;
    var bias_len: usize = 0;
    for (named_params) |param| {
        const len = param.tensor.nElems();
        if (isBiasParameter(T, param)) {
            bias_len += len;
        } else {
            weights_len += len;
        }
    }
    return .{
        .weights_len = weights_len,
        .bias_len = bias_len,
        .parameter_len = weights_len + bias_len,
    };
}

fn compiledModuleSupport(comptime T: type, base: CompileSupport, input: *const Tensor(T), output: *const Tensor(T), named_params: []const NamedParameter(T)) CompileSupport {
    var support = base;
    if (support.native_path) |path| {
        if (std.mem.eql(u8, path, "tiny-linear") and !input.hasShape(support.inputShape() orelse &.{})) {
            support.native_path = "device-program";
            support.model_kind = "module";
        }
    }
    support.input_n_dims = input.n_dims;
    support.input_ne = input.ne;
    support.output_n_dims = output.n_dims;
    support.output_ne = output.ne;
    support.input_len = input.nElems();
    support.output_len = output.nElems();
    const lengths = compileSupportParameterLengths(T, named_params);
    support.weights_len = lengths.weights_len;
    support.bias_len = lengths.bias_len;
    support.parameter_len = lengths.parameter_len;
    return support;
}

fn annotateCompileSupportWithIr(support: *CompileSupport, ir: tensor_program_ir.Inspection) void {
    support.ir_op_count = ir.op_count;
    support.ir_normalized_step_hash = ir.normalized_step_hash;
    support.ir_normalized_shape_hash = ir.normalized_shape_hash;
}

fn validateLinearCompileInputShape(shape: []const usize, in_features: usize) CompileError!void {
    try validateModuleInputShape(shape);
    if (shape.len > 2) return error.InvalidInputShape;
    if (shape[0] != in_features) return error.InvalidInputShape;
}

fn defaultCompileInputShape(module: anytype, buf: *[max_dims]usize) ?[]const usize {
    const Module = @TypeOf(module);
    const Value = moduleValueType(Module);
    if (comptime @hasDecl(Value, "namedParameters")) {
        const named = switch (@typeInfo(Module)) {
            .pointer => module.namedParameters(),
            else => return null,
        };
        for (named) |param| {
            if (param.layout) |layout| {
                if (std.mem.eql(u8, layout, linear_weight_layout) and param.tensor.n_dims >= 2) {
                    buf[0] = param.tensor.ne[1];
                    return buf[0..1];
                }
            }
        }
    }
    return null;
}

const TracedCompileSupport = struct {
    support: CompileSupport,
    ir: tensor_program_ir.Inspection,
};

fn traceCompileSupportForInputShape(comptime T: type, alloc: Alloc, module: anytype, input_shape: []const usize) !TracedCompileSupport {
    if (T != f32) return error.UnsupportedScalarType;
    try validateModuleInputShape(input_shape);

    var graph = @import("graph.zig").ComputeGraph(T).init(alloc);
    defer graph.deinit();

    const input = try Tensor(T).init(graph.allocator(), input_shape);
    _ = input.setAllScalar(0);
    const output = module.forward(input);
    try graph.infer(output);

    var support = compiledModuleSupport(T, moduleCompileSupportValue(module), input, output, moduleNamedParameters(T, module));
    var ir = try tensor_program_ir.TensorProgramIr(T).init(alloc, &graph);
    defer ir.deinit(alloc);
    const inspection = ir.inspect();
    annotateCompileSupportWithIr(&support, inspection);
    return .{ .support = support, .ir = inspection };
}

/// Run the cold module trace for an explicit input shape and return exact compile
/// support evidence without constructing a backend Program.
pub fn compileSupportForInputShape(comptime T: type, alloc: Alloc, module: anytype, input_shape: []const usize) !CompileSupport {
    return (try traceCompileSupportForInputShape(T, alloc, module, input_shape)).support;
}

pub fn ModuleProgram(comptime T: type) type {
    const TensorT = Tensor(T);
    const GraphT = @import("graph.zig").ComputeGraph(T);
    const DeviceT = DeviceInference(T);

    return struct {
        const Self = @This();

        alloc: Alloc,
        graph: GraphT,
        input: *TensorT,
        output: *TensorT,
        persistent_tensors: []*TensorT,
        parameter_metadata: []ParameterMetadata,
        parameter_info: []ModuleParameterInfo,
        compile_support: CompileSupport,
        program: DeviceT.Program,
        owned_cpu_backend: ?*CpuBackend,

        pub fn compile(alloc: Alloc, module: anytype, options: ModuleCompileOptions) !Self {
            if (T != f32) return error.UnsupportedScalarType;

            var default_shape_buf: [max_dims]usize = undefined;
            const input_shape = options.input_shape orelse
                defaultCompileInputShape(module, &default_shape_buf) orelse
                return error.InvalidInputShape;
            try validateModuleInputShape(input_shape);

            var owned_cpu_backend: ?*CpuBackend = null;
            errdefer if (owned_cpu_backend) |cpu| alloc.destroy(cpu);
            const be = if (options.backend) |backend| backend else blk: {
                const cpu = try alloc.create(CpuBackend);
                cpu.* = CpuBackend.init(alloc);
                owned_cpu_backend = cpu;
                break :blk cpu.backend();
            };

            var graph = GraphT.init(alloc);
            errdefer graph.deinit();
            const input = try TensorT.init(graph.allocator(), input_shape);
            _ = input.setAllScalar(0);
            const output = module.forward(input);
            try graph.infer(output);

            const named_params = moduleNamedParameters(T, module);
            const params = moduleParameters(T, module);
            const persistent_tensors = try alloc.dupe(*TensorT, params);
            errdefer alloc.free(persistent_tensors);

            const parameter_metadata = try copyParameterMetadata(T, alloc, persistent_tensors, named_params);
            errdefer {
                freeParameterMetadata(alloc, parameter_metadata);
                alloc.free(parameter_metadata);
            }
            const parameter_info = try buildModuleParameterInfo(T, alloc, parameter_metadata, persistent_tensors);
            errdefer alloc.free(parameter_info);
            var compile_support = compiledModuleSupport(T, moduleCompileSupportValue(module), input, output, named_params);

            var device_program = try DeviceT.Program.compile(.{
                .graph = &graph,
                .be = be,
                .alloc = alloc,
                .input_tensors = &.{input},
                .persistent_tensors = persistent_tensors,
                .output_tensors = &.{output},
            });
            errdefer device_program.deinit();
            annotateCompileSupportWithIr(&compile_support, device_program.inspect().ir);

            return .{
                .alloc = alloc,
                .graph = graph,
                .input = input,
                .output = output,
                .persistent_tensors = persistent_tensors,
                .parameter_metadata = parameter_metadata,
                .parameter_info = parameter_info,
                .compile_support = compile_support,
                .program = device_program,
                .owned_cpu_backend = owned_cpu_backend,
            };
        }

        pub fn compileSupport(self: *const Self) CompileSupport {
            return self.compile_support;
        }

        pub fn canCompile(self: *const Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn inputLen(self: *const Self) usize {
            return self.input.nElems();
        }

        pub fn inputShape(self: *const Self) []const usize {
            return self.input.ne[0..@as(usize, self.input.n_dims)];
        }

        pub fn outputLen(self: *const Self) usize {
            return self.output.nElems();
        }

        pub fn outputShape(self: *const Self) []const usize {
            return self.output.ne[0..@as(usize, self.output.n_dims)];
        }

        pub fn parameterTensorCount(self: *const Self) usize {
            return self.persistent_tensors.len;
        }

        pub fn parameterTensorLen(self: *const Self, index: usize) ?usize {
            if (index >= self.persistent_tensors.len) return null;
            return self.persistent_tensors[index].nElems();
        }

        pub fn parameterTensorName(self: *const Self, index: usize) ?[]const u8 {
            if (index >= self.parameter_metadata.len) return null;
            return self.parameter_metadata[index].name;
        }

        pub fn parameterTensorLayout(self: *const Self, index: usize) ?[]const u8 {
            if (index >= self.parameter_metadata.len) return null;
            return self.parameter_metadata[index].layout;
        }

        pub fn parameterTensorShape(self: *const Self, index: usize) ?[]const usize {
            if (index >= self.persistent_tensors.len) return null;
            const param = self.persistent_tensors[index];
            return param.ne[0..@as(usize, param.n_dims)];
        }

        pub fn parameterIndex(self: *const Self, name: []const u8) ?usize {
            return self.parameterIndexByName(name);
        }

        pub fn parameterInfo(self: *const Self, index: usize) ?ModuleParameterInfo {
            if (index >= self.parameter_info.len) return null;
            return self.parameter_info[index];
        }

        pub fn parameterInfoByName(self: *const Self, name: []const u8) ?ModuleParameterInfo {
            return self.parameterInfo(self.parameterIndex(name) orelse return null);
        }

        pub fn parameterInfos(self: *const Self) []const ModuleParameterInfo {
            return self.parameter_info;
        }

        pub fn parameterLen(self: *const Self) usize {
            var len: usize = 0;
            for (self.persistent_tensors) |param| {
                len += param.nElems();
            }
            return len;
        }

        pub fn bindingRequirementHash(self: *const Self) u64 {
            return self.inspect().binding_requirement_hash;
        }

        pub fn persistentRequirementCount(self: *const Self) usize {
            return self.inspect().persistent_requirement_count;
        }

        pub fn stepInputRequirementCount(self: *const Self) usize {
            return self.inspect().step_input_requirement_count;
        }

        pub fn stepOutputRequirementCount(self: *const Self) usize {
            return self.inspect().step_output_requirement_count;
        }

        fn parameterIndexByName(self: *const Self, name: []const u8) ?usize {
            for (self.parameter_metadata, 0..) |entry, index| {
                if (std.mem.eql(u8, entry.name, name)) return index;
            }
            return null;
        }

        fn tensorShape(tensor: *const TensorT) []const usize {
            return tensor.ne[0..@as(usize, tensor.n_dims)];
        }

        fn compatibilityOk() ModuleCompatibility {
            return .{ .compatible = true, .code = .compatible };
        }

        fn compatibilityFailure(
            code: ModuleCompatibilityCode,
            reason: []const u8,
            parameter_index: ?usize,
            parameter_name: ?[]const u8,
            expected_shape: ?[]const usize,
            actual_shape: ?[]const usize,
            expected_layout: ?[]const u8,
            actual_layout: ?[]const u8,
        ) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = code,
                .reason = reason,
                .parameter_index = parameter_index,
                .parameter_name = parameter_name,
                .expected_shape = expected_shape,
                .actual_shape = actual_shape,
                .expected_layout = expected_layout,
                .actual_layout = actual_layout,
            };
        }

        fn supportCompatibilityFailure(code: ModuleCompatibilityCode, reason: []const u8, expected: CompileSupport, actual: CompileSupport) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = code,
                .reason = reason,
                .expected_native_path = expected.native_path,
                .actual_native_path = actual.native_path,
                .expected_model_kind = expected.model_kind,
                .actual_model_kind = actual.model_kind,
                .expected_layer_count = expected.layer_count,
                .actual_layer_count = actual.layer_count,
            };
        }

        fn scalarCompatibilityFailure(code: ModuleCompatibilityCode, reason: []const u8, expected_count: ?usize, actual_count: ?usize) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = code,
                .reason = reason,
                .expected_scalar_count = expected_count,
                .actual_scalar_count = actual_count,
            };
        }

        fn irCompatibilityFailure(reason: []const u8, expected: CompileSupport, actual: CompileSupport) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = .ir_mismatch,
                .reason = reason,
                .expected_ir_step_hash = expected.ir_normalized_step_hash,
                .actual_ir_step_hash = actual.ir_normalized_step_hash,
                .expected_ir_shape_hash = expected.ir_normalized_shape_hash,
                .actual_ir_shape_hash = actual.ir_normalized_shape_hash,
            };
        }

        fn moduleAnalysisFailure(reason: []const u8) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = .module_analysis_failed,
                .reason = reason,
            };
        }

        fn layoutsCompatible(expected: ?[]const u8, actual: ?[]const u8) bool {
            if (expected == null) return true;
            const actual_layout = actual orelse return false;
            return std.mem.eql(u8, expected.?, actual_layout);
        }

        fn checkCompatibilityResult(compatibility: ModuleCompatibility) !void {
            if (compatibility.compatible) return;
            return switch (compatibility.code) {
                .compatible => {},
                .unsupported_module => error.UnsupportedCompile,
                .native_path_mismatch => error.UnsupportedCompile,
                .model_kind_mismatch => error.UnsupportedCompile,
                .layer_count_mismatch => error.UnsupportedCompile,
                .input_shape_mismatch => error.ShapeMismatch,
                .output_shape_mismatch => error.ShapeMismatch,
                .input_len_mismatch => error.ShapeMismatch,
                .output_len_mismatch => error.ShapeMismatch,
                .weights_len_mismatch => error.ShapeMismatch,
                .bias_len_mismatch => error.ShapeMismatch,
                .parameter_len_mismatch => error.ShapeMismatch,
                .module_analysis_failed => error.UnsupportedCompile,
                .ir_mismatch => error.UnsupportedCompile,
                .missing_parameter => error.MissingParameter,
                .unexpected_parameter => error.UnexpectedParameter,
                .shape_mismatch => error.ShapeMismatch,
                .layout_mismatch => error.LayoutMismatch,
                .unsupported_tensor_layout => error.UnsupportedTensorLayout,
            };
        }

        fn optionalTextMatches(expected: ?[]const u8, actual: ?[]const u8) bool {
            if (expected == null) return true;
            const actual_text = actual orelse return false;
            return std.mem.eql(u8, expected.?, actual_text);
        }

        fn optionalCountMatches(expected: ?usize, actual: ?usize) bool {
            if (expected == null) return true;
            const actual_count = actual orelse return true;
            return expected.? == actual_count;
        }

        fn optionalShapeMatches(expected: ?[]const usize, actual: ?[]const usize) bool {
            if (expected == null) return true;
            const actual_shape = actual orelse return true;
            return std.mem.eql(usize, expected.?, actual_shape);
        }

        fn supportShapeFailure(code: ModuleCompatibilityCode, reason: []const u8, expected_shape: ?[]const usize, actual_shape: ?[]const usize) ModuleCompatibility {
            return .{
                .compatible = false,
                .code = code,
                .reason = reason,
                .expected_shape = expected_shape,
                .actual_shape = actual_shape,
            };
        }

        fn supportNeedsProgramShapeTrace(self: *const Self, actual: CompileSupport) bool {
            if (self.compile_support.ir_normalized_step_hash != null and actual.ir_normalized_step_hash == null) return true;
            if (self.compile_support.ir_normalized_shape_hash != null and actual.ir_normalized_shape_hash == null) return true;
            if (actual.inputShape() == null) return true;
            if (actual.outputShape() == null) return true;
            if (actual.input_len == null) return true;
            if (actual.output_len == null) return true;
            return false;
        }

        fn compileSupportCompatibility(self: *const Self, actual: CompileSupport, compare_ir: bool) ModuleCompatibility {
            const expected = self.compile_support;
            if (!actual.supported) return supportCompatibilityFailure(.unsupported_module, actual.reason orelse "module does not report native compile support", expected, actual);
            if (!optionalTextMatches(expected.native_path, actual.native_path)) return supportCompatibilityFailure(.native_path_mismatch, "module native path does not match Program native path", expected, actual);
            if (!optionalTextMatches(expected.model_kind, actual.model_kind)) return supportCompatibilityFailure(.model_kind_mismatch, "module model kind does not match Program model kind", expected, actual);
            if (!optionalCountMatches(expected.layer_count, actual.layer_count)) return supportCompatibilityFailure(.layer_count_mismatch, "module layer count does not match Program layer count", expected, actual);
            if (!optionalShapeMatches(self.inputShape(), actual.inputShape())) return supportShapeFailure(.input_shape_mismatch, "module input shape does not match Program input shape", self.inputShape(), actual.inputShape());
            if (!optionalShapeMatches(self.outputShape(), actual.outputShape())) return supportShapeFailure(.output_shape_mismatch, "module output shape does not match Program output shape", self.outputShape(), actual.outputShape());
            if (!optionalCountMatches(self.inputLen(), actual.input_len)) return scalarCompatibilityFailure(.input_len_mismatch, "module input length does not match Program input length", self.inputLen(), actual.input_len);
            if (!optionalCountMatches(self.outputLen(), actual.output_len)) return scalarCompatibilityFailure(.output_len_mismatch, "module output length does not match Program output length", self.outputLen(), actual.output_len);
            if (!optionalCountMatches(expected.weights_len, actual.weights_len)) return scalarCompatibilityFailure(.weights_len_mismatch, "module weights length does not match Program weights length", expected.weights_len, actual.weights_len);
            if (!optionalCountMatches(expected.bias_len, actual.bias_len)) return scalarCompatibilityFailure(.bias_len_mismatch, "module bias length does not match Program bias length", expected.bias_len, actual.bias_len);
            if (!optionalCountMatches(self.parameterLen(), actual.parameter_len)) return scalarCompatibilityFailure(.parameter_len_mismatch, "module parameter length does not match Program parameter length", self.parameterLen(), actual.parameter_len);
            if (compare_ir) {
                if (expected.ir_normalized_step_hash) |expected_hash| {
                    const actual_hash = actual.ir_normalized_step_hash orelse return irCompatibilityFailure("module Tensor Program IR is missing normalized step signature", expected, actual);
                    if (expected_hash != actual_hash) return irCompatibilityFailure("module Tensor Program IR step signature does not match Program", expected, actual);
                }
                if (expected.ir_normalized_shape_hash) |expected_hash| {
                    const actual_hash = actual.ir_normalized_shape_hash orelse return irCompatibilityFailure("module Tensor Program IR is missing normalized shape signature", expected, actual);
                    if (expected_hash != actual_hash) return irCompatibilityFailure("module Tensor Program IR shape signature does not match Program", expected, actual);
                }
            }
            return compatibilityOk();
        }

        pub fn moduleCompatibility(self: *const Self, module: anytype) ModuleCompatibility {
            var module_support = moduleCompileSupportValue(module);
            const static_compatibility = self.compileSupportCompatibility(module_support, false);
            if (!static_compatibility.compatible) return static_compatibility;

            if (self.supportNeedsProgramShapeTrace(module_support)) {
                const input_shape = self.inputShape();
                module_support = (traceCompileSupportForInputShape(T, self.alloc, module, input_shape) catch |err| switch (err) {
                    error.UnsupportedScalarType => return moduleAnalysisFailure("module trace compatibility only supports f32 Programs"),
                    error.InvalidInputShape => return supportShapeFailure(.input_shape_mismatch, "module cannot trace with Program input shape", self.inputShape(), null),
                    error.OutOfMemory => return moduleAnalysisFailure("module trace compatibility ran out of memory"),
                    else => return moduleAnalysisFailure("module trace compatibility failed"),
                }).support;
            }

            const support_compatibility = self.compileSupportCompatibility(module_support, true);
            if (!support_compatibility.compatible) return support_compatibility;

            const named = moduleNamedParameters(T, module);
            for (self.parameter_metadata, self.persistent_tensors, 0..) |expected_metadata, compiled_param, index| {
                const expected_shape = tensorShape(compiled_param);
                const candidate = findNamedParameter(T, named, expected_metadata.name) orelse return compatibilityFailure(
                    .missing_parameter,
                    "module is missing a Program parameter",
                    index,
                    expected_metadata.name,
                    expected_shape,
                    null,
                    expected_metadata.layout,
                    null,
                );
                const actual_shape = tensorShape(candidate.tensor);
                if (!candidate.tensor.hasShape(expected_shape)) return compatibilityFailure(
                    .shape_mismatch,
                    "module parameter shape does not match Program parameter shape",
                    index,
                    expected_metadata.name,
                    expected_shape,
                    actual_shape,
                    expected_metadata.layout,
                    candidate.layout,
                );
                if (!candidate.tensor.isDenseLayout()) return compatibilityFailure(
                    .unsupported_tensor_layout,
                    "module parameter must be dense to bind as Program state",
                    index,
                    expected_metadata.name,
                    expected_shape,
                    actual_shape,
                    expected_metadata.layout,
                    candidate.layout,
                );
                if (!layoutsCompatible(expected_metadata.layout, candidate.layout)) return compatibilityFailure(
                    .layout_mismatch,
                    "module parameter layout does not match Program parameter layout",
                    index,
                    expected_metadata.name,
                    expected_shape,
                    actual_shape,
                    expected_metadata.layout,
                    candidate.layout,
                );
            }

            for (named) |candidate| {
                if (self.parameterIndexByName(candidate.name) == null) return compatibilityFailure(
                    .unexpected_parameter,
                    "module has a parameter that is not part of this Program",
                    null,
                    candidate.name,
                    null,
                    tensorShape(candidate.tensor),
                    null,
                    candidate.layout,
                );
            }

            return compatibilityOk();
        }

        fn fillModulePersistentBindings(self: *const Self, module: anytype, bindings: []DeviceT.TensorBinding) !void {
            if (bindings.len != self.parameterTensorCount()) return error.ShapeMismatch;
            try checkCompatibilityResult(self.moduleCompatibility(module));
            const named = moduleNamedParameters(T, module);

            for (self.parameter_metadata, self.persistent_tensors, 0..) |expected_metadata, compiled_param, index| {
                const candidate = findNamedParameter(T, named, expected_metadata.name).?;
                bindings[index] = DeviceT.TensorBinding.host(compiled_param, candidate.tensor.denseSlice().ptr, candidate.tensor.nElems());
            }
        }

        pub fn acceptsModule(self: *const Self, module: anytype) bool {
            return self.moduleCompatibility(module).compatible;
        }

        pub fn checkModuleCompatibility(self: *const Self, module: anytype) !void {
            try checkCompatibilityResult(self.moduleCompatibility(module));
        }

        pub fn inspect(self: *const Self) DeviceT.ProgramInspection {
            return self.program.inspect();
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.program.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile_mod.RuntimeProfile) void {
            self.program.addRuntimeProfileTo(dest);
        }

        pub fn runtimeProfile(self: *const Self) profile_mod.RuntimeProfile {
            var out = profile_mod.RuntimeProfile{};
            self.addRuntimeProfileTo(&out);
            return out;
        }

        fn bindWithPersistentBindings(self: *Self, input: []T, output: []T, persistent_bindings: []const DeviceT.TensorBinding) !ModuleSession(T) {
            if (input.len != self.inputLen()) return error.ShapeMismatch;
            if (output.len != self.outputLen()) return error.ShapeMismatch;

            const input_binding = DeviceT.TensorBinding.host(self.input, input.ptr, input.len);
            const use_explicit_persistent = persistent_bindings.len > 0;
            var session = try self.program.bind(.{
                .input_tensors = &.{self.input},
                .input_bindings = &.{input_binding},
                .persistent_tensors = if (use_explicit_persistent) &.{} else self.persistent_tensors,
                .persistent_bindings = persistent_bindings,
                .output_tensor = self.output,
                .output_host_buf = output.ptr,
                .output_len = output.len,
            });
            errdefer session.deinit();

            return .{
                .program = self,
                .session = session,
                .input = input,
                .output = output,
                .input_tensor = null,
                .output_tensor = null,
            };
        }

        pub fn bind(self: *Self, input: []T, output: []T) !ModuleSession(T) {
            return self.bindWithPersistentBindings(input, output, &.{});
        }

        pub fn bindModule(self: *Self, module: anytype, input: []T, output: []T) !ModuleSession(T) {
            const bindings = try self.alloc.alloc(DeviceT.TensorBinding, self.parameterTensorCount());
            defer self.alloc.free(bindings);
            try self.fillModulePersistentBindings(module, bindings);
            return self.bindWithPersistentBindings(input, output, bindings);
        }

        pub fn bindTensors(self: *Self, input: *TensorT, output: *TensorT) !ModuleSession(T) {
            try requireDenseModuleTensor(T, self.input, input);
            try requireDenseModuleTensor(T, self.output, output);
            var session = try self.bind(input.denseSlice(), output.denseSlice());
            session.input_tensor = input;
            session.output_tensor = output;
            return session;
        }

        pub fn bindModuleTensors(self: *Self, module: anytype, input: *TensorT, output: *TensorT) !ModuleSession(T) {
            try requireDenseModuleTensor(T, self.input, input);
            try requireDenseModuleTensor(T, self.output, output);
            var session = try self.bindModule(module, input.denseSlice(), output.denseSlice());
            session.input_tensor = input;
            session.output_tensor = output;
            return session;
        }

        pub fn deinit(self: *Self) void {
            self.program.deinit();
            self.graph.deinit();
            self.alloc.free(self.parameter_info);
            freeParameterMetadata(self.alloc, self.parameter_metadata);
            self.alloc.free(self.parameter_metadata);
            self.alloc.free(self.persistent_tensors);
            if (self.owned_cpu_backend) |cpu| self.alloc.destroy(cpu);
            self.* = undefined;
        }
    };
}

pub fn LinearProgram(comptime T: type) type {
    return ModuleProgram(T);
}

pub fn ModuleSession(comptime T: type) type {
    const DeviceT = DeviceInference(T);
    const TensorT = Tensor(T);

    return struct {
        const Self = @This();

        program: *ModuleProgram(T),
        session: DeviceT.Session,
        input: []T,
        output: []T,
        input_tensor: ?*TensorT = null,
        output_tensor: ?*TensorT = null,

        pub fn step(self: *Self) ![]T {
            try self.program.program.executeStep(&self.session, .{
                .window = try backend_mod.RuntimeWindow.init(0, 0),
            });
            return self.output;
        }

        pub fn execute(self: *Self, input: []const T) ![]T {
            if (input.len != self.input.len) return error.ShapeMismatch;
            @memcpy(self.input, input);
            return self.step();
        }

        pub fn compileSupport(self: *const Self) CompileSupport {
            return self.program.compileSupport();
        }

        pub fn canCompile(self: *const Self) bool {
            return self.program.canCompile();
        }

        pub fn inputShape(self: *const Self) []const usize {
            return self.program.inputShape();
        }

        pub fn outputShape(self: *const Self) []const usize {
            return self.program.outputShape();
        }

        pub fn parameterTensorCount(self: *const Self) usize {
            return self.program.parameterTensorCount();
        }

        pub fn parameterTensorLen(self: *const Self, index: usize) ?usize {
            return self.program.parameterTensorLen(index);
        }

        pub fn parameterTensorName(self: *const Self, index: usize) ?[]const u8 {
            return self.program.parameterTensorName(index);
        }

        pub fn parameterTensorLayout(self: *const Self, index: usize) ?[]const u8 {
            return self.program.parameterTensorLayout(index);
        }

        pub fn parameterTensorShape(self: *const Self, index: usize) ?[]const usize {
            return self.program.parameterTensorShape(index);
        }

        pub fn parameterIndex(self: *const Self, name: []const u8) ?usize {
            return self.program.parameterIndex(name);
        }

        pub fn parameterInfo(self: *const Self, index: usize) ?ModuleParameterInfo {
            return self.program.parameterInfo(index);
        }

        pub fn parameterInfoByName(self: *const Self, name: []const u8) ?ModuleParameterInfo {
            return self.program.parameterInfoByName(name);
        }

        pub fn parameterInfos(self: *const Self) []const ModuleParameterInfo {
            return self.program.parameterInfos();
        }

        pub fn parameterLen(self: *const Self) usize {
            return self.program.parameterLen();
        }

        pub fn bindingShapeHash(self: *const Self) u64 {
            return self.inspect().binding_shape_hash;
        }

        pub fn persistentBindingCount(self: *const Self) usize {
            return self.inspect().persistent_binding_count;
        }

        pub fn stepInputCount(self: *const Self) usize {
            return self.inspect().step_input_count;
        }

        pub fn stepOutputCount(self: *const Self) usize {
            return self.inspect().step_output_count;
        }

        pub fn hostBindingCount(self: *const Self) usize {
            return self.inspect().host_binding_count;
        }

        pub fn resourceBindingCount(self: *const Self) usize {
            return self.inspect().resource_binding_count;
        }

        pub fn outputTensor(self: *Self) !*TensorT {
            return self.output_tensor orelse error.MissingTensorBinding;
        }

        pub fn stepTensor(self: *Self) !*TensorT {
            _ = try self.step();
            return self.outputTensor();
        }

        pub fn executeTensor(self: *Self, input: *const TensorT) !*TensorT {
            try requireDenseModuleTensor(T, self.program.input, input);
            @memcpy(self.input, input.denseSliceConst());
            return self.stepTensor();
        }

        pub fn uploadParameters(self: *Self) !void {
            try self.program.program.uploadPersistentInputs(&self.session);
        }

        pub fn uploadParameter(self: *Self, index: usize) !void {
            try self.uploadParameterRange(index, 1);
        }

        pub fn uploadParameterByName(self: *Self, name: []const u8) !void {
            try self.uploadParameter(self.parameterIndex(name) orelse return error.MissingParameter);
        }

        pub fn uploadParameterRange(self: *Self, first: usize, len: usize) !void {
            try self.program.program.uploadPersistentInputRange(&self.session, first, len);
        }

        pub fn inspect(self: *const Self) DeviceT.SessionInspection {
            return self.session.inspect();
        }

        pub fn resetRuntimeProfile(self: *const Self) void {
            self.session.resetRuntimeProfile();
        }

        pub fn addRuntimeProfileTo(self: *const Self, dest: *profile_mod.RuntimeProfile) void {
            self.session.addRuntimeProfileTo(dest);
        }

        pub fn runtimeProfile(self: *const Self) profile_mod.RuntimeProfile {
            var out = profile_mod.RuntimeProfile{};
            self.addRuntimeProfileTo(&out);
            return out;
        }

        pub fn deinit(self: *Self) void {
            self.session.deinit();
            self.* = undefined;
        }
    };
}

pub fn LinearSession(comptime T: type) type {
    return ModuleSession(T);
}

fn boxedCompileSupport(comptime Module: type, ctx: *anyopaque) CompileSupport {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime @hasDecl(Module, "compileSupport")) {
        return module.compileSupport();
    }
    return unsupportedCompile("module does not expose a native Program compiler");
}

fn boxedTrain(comptime Module: type, ctx: *anyopaque, mode: bool) void {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime @hasDecl(Module, "train")) {
        _ = module.train(mode);
    }
}

fn boxedDeinit(comptime Module: type, alloc: Alloc, ctx: *anyopaque) void {
    const module: *Module = @ptrCast(@alignCast(ctx));
    if (comptime @hasDecl(Module, "deinit")) {
        module.deinit();
    }
    alloc.destroy(module);
}

fn featureReduceNe(comptime T: type, x: *const Tensor(T), features: usize) [max_dims]usize {
    std.debug.assert(x.n_dims >= 1);
    std.debug.assert(x.ne[0] == features);
    var reduce_ne = x.ne;
    reduce_ne[0] = 1;
    return reduce_ne;
}

fn sigmoidTensor(x: anytype) @TypeOf(x) {
    return x.sigmoid();
}

fn siluTensor(x: anytype) @TypeOf(x) {
    return x.silu();
}

fn tanhTensor(x: anytype) @TypeOf(x) {
    return x.tanh();
}

/// Fully-connected layer: `x @ w + b`.
///
/// This is syntax over existing tensor primitives, so it does not add a graph
/// op or backend obligation.
pub fn linear(comptime T: type, x: *Tensor(T), w: *Tensor(T), b: ?*Tensor(T)) *Tensor(T) {
    const h = x.mm(w);
    return if (b) |bias| h.add(bias) else h;
}

/// Owned fully-connected layer parameters plus a primitive-op forward pass.
pub fn Linear(comptime T: type) type {
    return struct {
        const Self = @This();

        weight: *Tensor(T),
        bias: ?*Tensor(T),
        param_buf: [2]*Tensor(T),
        named_param_buf: [2]NamedParameter(T),

        pub fn init(alloc: Alloc, in_features: usize, out_features: usize, config: LinearConfig) Alloc.Error!Self {
            const weight = try Tensor(T).init(alloc, &.{ out_features, in_features });
            errdefer weight.deinit();
            weight.setParam();
            kaimingUniform(T, weight, config.seed);

            const bias = if (config.bias) blk: {
                const b = try Tensor(T).init(alloc, &.{out_features});
                errdefer b.deinit();
                b.setParam();
                _ = b.setAllScalar(0);
                break :blk b;
            } else null;

            return .{ .weight = weight, .bias = bias, .param_buf = undefined, .named_param_buf = undefined };
        }

        pub fn deinit(self: *Self) void {
            if (self.bias) |bias| bias.deinit();
            self.weight.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor(T)) *Tensor(T) {
            return linear(T, x, self.weight, self.bias);
        }

        pub fn parameters(self: *Self) []const *Tensor(T) {
            self.param_buf[0] = self.weight;
            if (self.bias) |bias| {
                self.param_buf[1] = bias;
                return self.param_buf[0..2];
            }
            return self.param_buf[0..1];
        }

        pub fn namedParameters(self: *Self) []const NamedParameter(T) {
            self.named_param_buf[0] = .{ .name = "weight", .tensor = self.weight, .layout = linear_weight_layout };
            if (self.bias) |bias| {
                self.named_param_buf[1] = .{ .name = "bias", .tensor = bias, .layout = linear_bias_layout };
                return self.named_param_buf[0..2];
            }
            return self.named_param_buf[0..1];
        }

        pub fn compileSupport(self: *Self) CompileSupport {
            return zigTinyLinearCompileSupport(T, self.weight.ne[1], self.weight.ne[0], self.bias != null);
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(T) {
            const in_features = self.weight.ne[1];
            const default_shape = [_]usize{in_features};
            try validateLinearCompileInputShape(options.input_shape orelse default_shape[0..], in_features);
            return ModuleProgram(T).compile(alloc, self, options);
        }
    };
}

/// Owned embedding table plus a primitive-op lookup.
pub fn Embedding(comptime T: type) type {
    return struct {
        const Self = @This();

        weight: *Tensor(T),
        param_buf: [1]*Tensor(T),
        named_param_buf: [1]NamedParameter(T),

        pub fn init(alloc: Alloc, num_embeddings: usize, embedding_dim: usize, config: EmbeddingConfig) Alloc.Error!Self {
            const weight = try Tensor(T).init(alloc, &.{ embedding_dim, num_embeddings });
            weight.setParam();
            uniform(T, weight, -0.02, 0.02, config.seed);
            return .{ .weight = weight, .param_buf = undefined, .named_param_buf = undefined };
        }

        pub fn deinit(self: *Self) void {
            self.weight.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, indices: *Tensor(T)) *Tensor(T) {
            return self.weight.gatherRows(indices);
        }

        pub fn parameters(self: *Self) []const *Tensor(T) {
            self.param_buf[0] = self.weight;
            return self.param_buf[0..1];
        }

        pub fn namedParameters(self: *Self) []const NamedParameter(T) {
            self.named_param_buf[0] = .{ .name = "weight", .tensor = self.weight, .layout = embedding_weight_layout };
            return self.named_param_buf[0..1];
        }

        pub fn compileSupport(_: *Self) CompileSupport {
            return deviceProgramCompileSupport(T);
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(T) {
            return ModuleProgram(T).compile(alloc, self, options);
        }
    };
}

/// Owned feature-axis LayerNorm parameters plus a primitive-op forward pass.
///
/// zgml tensors are feature-major (`{features, batch, ...}`), so this module
/// normalizes axis 0 while preserving all following batch/window axes.
pub fn LayerNorm(comptime T: type) type {
    return struct {
        const Self = @This();

        features: usize,
        eps: T,
        weight: ?*Tensor(T),
        bias: ?*Tensor(T),
        param_buf: [2]*Tensor(T),
        named_param_buf: [2]NamedParameter(T),

        pub fn init(alloc: Alloc, features: usize, config: LayerNormConfig) Alloc.Error!Self {
            const weight = if (config.affine) blk: {
                const w = try Tensor(T).init(alloc, &.{features});
                errdefer w.deinit();
                w.setParam();
                _ = w.setAllScalar(1);
                break :blk w;
            } else null;

            const bias = if (config.affine and config.bias) blk: {
                const b = try Tensor(T).init(alloc, &.{features});
                errdefer b.deinit();
                b.setParam();
                _ = b.setAllScalar(0);
                break :blk b;
            } else null;

            return .{
                .features = features,
                .eps = @floatCast(config.eps),
                .weight = weight,
                .bias = bias,
                .param_buf = undefined,
                .named_param_buf = undefined,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.bias) |bias| bias.deinit();
            if (self.weight) |weight| weight.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor(T)) *Tensor(T) {
            const reduce_ne = featureReduceNe(T, x, self.features);
            var out = x.layerNorm(reduce_ne[0..x.n_dims], self.eps);
            if (self.weight) |weight| out = out.mul(weight);
            if (self.bias) |bias| out = out.addBias(bias);
            return out;
        }

        pub fn parameters(self: *Self) []const *Tensor(T) {
            var len: usize = 0;
            if (self.weight) |weight| {
                self.param_buf[len] = weight;
                len += 1;
            }
            if (self.bias) |bias| {
                self.param_buf[len] = bias;
                len += 1;
            }
            return self.param_buf[0..len];
        }

        pub fn namedParameters(self: *Self) []const NamedParameter(T) {
            var len: usize = 0;
            if (self.weight) |weight| {
                self.named_param_buf[len] = .{ .name = "weight", .tensor = weight, .layout = layer_norm_weight_layout };
                len += 1;
            }
            if (self.bias) |bias| {
                self.named_param_buf[len] = .{ .name = "bias", .tensor = bias, .layout = layer_norm_bias_layout };
                len += 1;
            }
            return self.named_param_buf[0..len];
        }

        pub fn compileSupport(_: *Self) CompileSupport {
            return composableCompile("nn.LayerNorm compiles as part of a native module Program");
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }
    };
}

/// Owned feature-axis RMSNorm scale plus a primitive-op forward pass.
///
/// RMSNorm is the transformer-friendly normalization used by LLaMA-family
/// models. It intentionally has no bias term.
pub fn RmsNorm(comptime T: type) type {
    return struct {
        const Self = @This();

        features: usize,
        eps: T,
        weight: ?*Tensor(T),
        param_buf: [1]*Tensor(T),
        named_param_buf: [1]NamedParameter(T),

        pub fn init(alloc: Alloc, features: usize, config: RmsNormConfig) Alloc.Error!Self {
            const weight = if (config.affine) blk: {
                const w = try Tensor(T).init(alloc, &.{features});
                errdefer w.deinit();
                w.setParam();
                _ = w.setAllScalar(1);
                break :blk w;
            } else null;

            return .{
                .features = features,
                .eps = @floatCast(config.eps),
                .weight = weight,
                .param_buf = undefined,
                .named_param_buf = undefined,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.weight) |weight| weight.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor(T)) *Tensor(T) {
            const reduce_ne = featureReduceNe(T, x, self.features);
            var out = x.rmsNorm(reduce_ne[0..x.n_dims], self.eps);
            if (self.weight) |weight| out = out.mul(weight);
            return out;
        }

        pub fn parameters(self: *Self) []const *Tensor(T) {
            if (self.weight) |weight| {
                self.param_buf[0] = weight;
                return self.param_buf[0..1];
            }
            return self.param_buf[0..0];
        }

        pub fn namedParameters(self: *Self) []const NamedParameter(T) {
            if (self.weight) |weight| {
                self.named_param_buf[0] = .{ .name = "weight", .tensor = weight, .layout = rms_norm_weight_layout };
                return self.named_param_buf[0..1];
            }
            return self.named_param_buf[0..0];
        }

        pub fn compileSupport(_: *Self) CompileSupport {
            return composableCompile("nn.RmsNorm compiles as part of a native module Program");
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }
    };
}

pub const Gelu = struct {
    pub fn forward(_: *Gelu, x: anytype) @TypeOf(x) {
        return x.gelu();
    }

    pub fn compileSupport(_: *Gelu) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Gelu) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Gelu, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Relu = struct {
    pub fn forward(_: *Relu, x: anytype) @TypeOf(x) {
        return x.relu();
    }

    pub fn compileSupport(_: *Relu) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Relu) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Relu, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Silu = struct {
    pub fn forward(_: *Silu, x: anytype) @TypeOf(x) {
        return siluTensor(x);
    }

    pub fn compileSupport(_: *Silu) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Silu) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Silu, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Sigmoid = struct {
    pub fn forward(_: *Sigmoid, x: anytype) @TypeOf(x) {
        return sigmoidTensor(x);
    }

    pub fn compileSupport(_: *Sigmoid) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Sigmoid) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Sigmoid, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Tanh = struct {
    pub fn forward(_: *Tanh, x: anytype) @TypeOf(x) {
        return tanhTensor(x);
    }

    pub fn compileSupport(_: *Tanh) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Tanh) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Tanh, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

const UnaryActivationKind = enum { exp, log, neg, recip, abs, sqrt, square, sgn, step };

fn UnaryActivation(comptime kind: UnaryActivationKind) type {
    return struct {
        const Self = @This();

        pub fn forward(_: *Self, x: anytype) @TypeOf(x) {
            return switch (kind) {
                .exp => x.exp(),
                .log => x.log(),
                .neg => x.neg(),
                .recip => x.recip(),
                .abs => x.abs(),
                .sqrt => x.sqrt(),
                .square => x.sqr(),
                .sgn => x.sgn(),
                .step => x.step(),
            };
        }

        pub fn compileSupport(_: *Self) CompileSupport {
            return shapePreservingDeviceProgramCompileSupport(f32);
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
            if (!self.canCompile()) return error.UnsupportedCompile;
            return ModuleProgram(f32).compile(alloc, self, options);
        }
    };
}

pub const Exp = UnaryActivation(.exp);
pub const Log = UnaryActivation(.log);
pub const Neg = UnaryActivation(.neg);
pub const Recip = UnaryActivation(.recip);
pub const Abs = UnaryActivation(.abs);
pub const Sqrt = UnaryActivation(.sqrt);
pub const Square = UnaryActivation(.square);
pub const Sgn = UnaryActivation(.sgn);
pub const StepActivation = UnaryActivation(.step);

pub fn Dropout(comptime T: type) type {
    return struct {
        const Self = @This();

        p: f32,
        training: bool,
        rng: std.Random.DefaultPrng,

        pub fn init(p: f64, config: DropoutConfig) DropoutError!Self {
            if (!std.math.isFinite(p) or p < 0 or p > 1) return error.InvalidDropoutProbability;
            return .{
                .p = @floatCast(p),
                .training = config.training,
                .rng = std.Random.DefaultPrng.init(config.seed),
            };
        }

        pub fn forward(self: *Self, x: *Tensor(T)) *Tensor(T) {
            if (!self.training or self.p == 0) return x.view();

            const mask = Tensor(T).init(x.alloc.?, x.ne[0..x.n_dims]) catch unreachable;
            _ = mask.markInternalAux();
            if (self.p == 1) {
                _ = mask.setAllScalar(0);
            } else {
                const scale: T = @floatCast(1.0 / (1.0 - self.p));
                var random = self.rng.random();
                for (mask.data) |*value| {
                    value.* = if (random.float(f32) >= self.p) scale else 0;
                }
            }
            return x.mul(mask);
        }

        pub fn parameters(_: *Self) []const *Tensor(T) {
            return &.{};
        }

        pub fn namedParameters(_: *Self) []const NamedParameter(T) {
            return &.{};
        }

        pub fn train(self: *Self, mode: bool) *Self {
            self.training = mode;
            return self;
        }

        pub fn eval(self: *Self) *Self {
            return self.train(false);
        }

        pub fn compileSupport(self: *Self) CompileSupport {
            if (!self.training or self.p == 0) return shapePreservingDeviceProgramCompileSupport(T);
            return unsupportedCompile("nn.Dropout native Program compilation only supports deterministic no-op Dropout (eval mode or p=0)");
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(T) {
            if (!self.canCompile()) return error.UnsupportedCompile;
            return ModuleProgram(T).compile(alloc, self, options);
        }
    };
}

pub const Softmax = struct {
    dim: usize,

    pub fn forward(self: *Softmax, x: anytype) @TypeOf(x) {
        return x.softmaxDim(self.dim);
    }

    pub fn compileSupport(_: *Softmax) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Softmax) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Softmax, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const LogSoftmax = struct {
    dim: usize,

    pub fn forward(self: *LogSoftmax, x: anytype) @TypeOf(x) {
        return x.logSoftmaxDim(self.dim);
    }

    pub fn compileSupport(_: *LogSoftmax) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *LogSoftmax) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *LogSoftmax, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

const ReductionKind = enum { sum, mean, max, min };

fn Reduction(comptime kind: ReductionKind) type {
    return struct {
        const Self = @This();

        dim: usize,

        pub fn forward(self: *Self, x: anytype) @TypeOf(x) {
            return switch (kind) {
                .sum => x.sumDim(self.dim),
                .mean => x.meanDim(self.dim),
                .max => x.maxDim(self.dim),
                .min => x.minDim(self.dim),
            };
        }

        pub fn compileSupport(_: *Self) CompileSupport {
            return deviceProgramCompileSupport(f32);
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
            if (!self.canCompile()) return error.UnsupportedCompile;
            return ModuleProgram(f32).compile(alloc, self, options);
        }
    };
}

pub const Sum = Reduction(.sum);
pub const Mean = Reduction(.mean);
pub const Max = Reduction(.max);
pub const Min = Reduction(.min);

fn copyModuleShape(shape: []const usize) struct { n_dims: u8, ne: [max_dims]usize } {
    std.debug.assert(shape.len > 0);
    std.debug.assert(shape.len <= max_dims);
    var ne = [_]usize{0} ** max_dims;
    @memcpy(ne[0..shape.len], shape);
    return .{ .n_dims = @intCast(shape.len), .ne = ne };
}

fn materializeDense(source: anytype) @TypeOf(source) {
    const TensorT = @typeInfo(@TypeOf(source)).pointer.child;
    const out = TensorT.init(source.alloc.?, source.ne[0..source.n_dims]) catch unreachable;
    out.op = .repeat;
    out.src0 = source;
    return out;
}

pub const Identity = struct {
    pub fn forward(_: *Identity, x: anytype) @TypeOf(x) {
        return x.view();
    }

    pub fn compileSupport(_: *Identity) CompileSupport {
        return shapePreservingDeviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Identity) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Identity, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Reshape = struct {
    n_dims: u8,
    ne: [max_dims]usize,

    pub fn forward(self: *Reshape, x: anytype) @TypeOf(x) {
        return x.reshape(self.ne[0..@as(usize, self.n_dims)]);
    }

    pub fn compileSupport(_: *Reshape) CompileSupport {
        return deviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Reshape) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Reshape, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const View = Reshape;

pub const Transpose = struct {
    pub fn forward(_: *Transpose, x: anytype) @TypeOf(x) {
        return materializeDense(x.transpose());
    }

    pub fn compileSupport(_: *Transpose) CompileSupport {
        return deviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Transpose) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Transpose, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const BroadcastTo = struct {
    n_dims: u8,
    ne: [max_dims]usize,

    pub fn forward(self: *BroadcastTo, x: anytype) @TypeOf(x) {
        const broadcasted = x.broadcastTo(self.ne[0..@as(usize, self.n_dims)]);
        return materializeDense(broadcasted);
    }

    pub fn compileSupport(_: *BroadcastTo) CompileSupport {
        return deviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *BroadcastTo) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *BroadcastTo, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub const Expand = BroadcastTo;

pub const Flatten = struct {
    start_dim: usize,
    end_dim: ?usize,

    pub fn forward(self: *Flatten, x: anytype) @TypeOf(x) {
        const last_dim = self.end_dim orelse @as(usize, x.n_dims - 1);
        std.debug.assert(self.start_dim <= last_dim);
        std.debug.assert(last_dim < x.n_dims);

        var ne = [_]usize{0} ** max_dims;
        var out_dim: usize = 0;
        for (0..self.start_dim) |dim| {
            ne[out_dim] = x.ne[dim];
            out_dim += 1;
        }
        var flattened: usize = 1;
        for (self.start_dim..last_dim + 1) |dim| flattened *= x.ne[dim];
        ne[out_dim] = flattened;
        out_dim += 1;
        for (last_dim + 1..x.n_dims) |dim| {
            ne[out_dim] = x.ne[dim];
            out_dim += 1;
        }
        return x.reshape(ne[0..out_dim]);
    }

    pub fn compileSupport(_: *Flatten) CompileSupport {
        return deviceProgramCompileSupport(f32);
    }

    pub fn canCompile(self: *Flatten) bool {
        return self.compileSupport().canCompile();
    }

    pub fn compile(self: *Flatten, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(f32) {
        if (!self.canCompile()) return error.UnsupportedCompile;
        return ModuleProgram(f32).compile(alloc, self, options);
    }
};

pub fn gelu() Gelu {
    return .{};
}

pub fn relu() Relu {
    return .{};
}

/// Constructor for the SiLU/Swish activation module.
///
/// `silu(T, x)` remains the functional tensor helper, so this constructor keeps
/// existing code source-compatible while making `Sequential` composition easy.
pub fn siluActivation() Silu {
    return .{};
}

pub fn sigmoidActivation() Sigmoid {
    return .{};
}

pub fn tanhActivation() Tanh {
    return .{};
}

pub fn exp() Exp {
    return .{};
}

pub fn log() Log {
    return .{};
}

pub fn neg() Neg {
    return .{};
}

pub fn recip() Recip {
    return .{};
}

pub fn abs() Abs {
    return .{};
}

pub fn sqrt() Sqrt {
    return .{};
}

pub fn square() Square {
    return .{};
}

pub fn sqr() Square {
    return .{};
}

pub fn sgn() Sgn {
    return .{};
}

pub fn sign() Sgn {
    return .{};
}

pub fn stepActivation() StepActivation {
    return .{};
}

pub fn dropout(comptime T: type, p: f64, config: DropoutConfig) DropoutError!Dropout(T) {
    return Dropout(T).init(p, config);
}

pub fn softmax(dim: usize) Softmax {
    return .{ .dim = dim };
}

pub fn logSoftmax(dim: usize) LogSoftmax {
    return .{ .dim = dim };
}

pub fn sum(dim: usize) Sum {
    return .{ .dim = dim };
}

pub fn mean(dim: usize) Mean {
    return .{ .dim = dim };
}

pub fn max(dim: usize) Max {
    return .{ .dim = dim };
}

pub fn min(dim: usize) Min {
    return .{ .dim = dim };
}

pub fn identity() Identity {
    return .{};
}

pub fn reshape(shape: []const usize) Reshape {
    const copied = copyModuleShape(shape);
    return .{ .n_dims = copied.n_dims, .ne = copied.ne };
}

pub fn view(shape: []const usize) View {
    const copied = copyModuleShape(shape);
    return .{ .n_dims = copied.n_dims, .ne = copied.ne };
}

pub fn transpose() Transpose {
    return .{};
}

pub fn broadcastTo(shape: []const usize) BroadcastTo {
    const copied = copyModuleShape(shape);
    return .{ .n_dims = copied.n_dims, .ne = copied.ne };
}

pub fn expand(shape: []const usize) Expand {
    const copied = copyModuleShape(shape);
    return .{ .n_dims = copied.n_dims, .ne = copied.ne };
}

pub fn flatten(start_dim: usize, end_dim: ?usize) Flatten {
    return .{ .start_dim = start_dim, .end_dim = end_dim };
}

pub fn flattenAll() Flatten {
    return flatten(0, null);
}

fn SequentialStep(comptime T: type) type {
    const TensorT = Tensor(T);
    return struct {
        ctx: *anyopaque,
        forward_fn: *const fn (*anyopaque, *TensorT) *TensorT,
        parameters_fn: *const fn (*anyopaque) []const *TensorT,
        named_parameters_fn: *const fn (*anyopaque) []const NamedParameter(T),
        compile_support_fn: *const fn (*anyopaque) CompileSupport,
        train_fn: *const fn (*anyopaque, bool) void,
        deinit_fn: *const fn (Alloc, *anyopaque) void,
    };
}

fn SequentialModel(comptime T: type) type {
    const TensorT = Tensor(T);
    return struct {
        const Self = @This();
        pub const Step = SequentialStep(T);

        alloc: Alloc,
        steps: []Step,
        param_buf: []*TensorT,
        named_param_buf: []NamedParameter(T),
        named_name_buf: [][]u8,
        named_param_len: usize,

        fn destroySteps(alloc: Alloc, steps: []Step) void {
            for (steps) |step| {
                step.deinit_fn(alloc, step.ctx);
            }
            alloc.free(steps);
        }

        pub fn forward(self: *Self, x: *TensorT) *TensorT {
            var out = x;
            for (self.steps) |step| {
                out = step.forward_fn(step.ctx, out);
            }
            return out;
        }

        pub fn parameters(self: *Self) []const *TensorT {
            var cursor: usize = 0;
            for (self.steps) |step| {
                const params = step.parameters_fn(step.ctx);
                @memcpy(self.param_buf[cursor..][0..params.len], params);
                cursor += params.len;
            }
            return self.param_buf[0..cursor];
        }

        pub fn namedParameters(self: *Self) []const NamedParameter(T) {
            return self.named_param_buf[0..self.named_param_len];
        }

        pub fn compileSupport(self: *Self) CompileSupport {
            if (T != f32) return unsupportedCompile("nn.Sequential native Program compilation currently supports f32 tensors");
            if (self.steps.len == 1) return self.steps[0].compile_support_fn(self.steps[0].ctx);
            var has_native_or_composable = false;
            var model_input_len: ?usize = null;
            var current_len: ?usize = null;
            var model_input_n_dims: ?u8 = null;
            var model_input_ne = [_]usize{0} ** max_dims;
            var current_n_dims: ?u8 = null;
            var current_ne = [_]usize{0} ** max_dims;
            var edge_lens_known = false;
            for (self.steps, 0..) |step, step_index| {
                const support = step.compile_support_fn(step.ctx);
                if (!support.supported and !support.composable) {
                    return .{
                        .supported = false,
                        .reason = support.reason,
                        .layer_count = self.steps.len,
                    };
                }
                if (support.input_len) |step_input_len| {
                    if (support.output_len) |step_output_len| {
                        if (!edge_lens_known) {
                            if (step_index == 0) {
                                model_input_len = step_input_len;
                                current_len = step_output_len;
                                if (support.inputShape()) |shape| {
                                    model_input_n_dims = @intCast(shape.len);
                                    copyCompileSupportShape(&model_input_ne, shape);
                                }
                                if (support.outputShape()) |shape| {
                                    current_n_dims = @intCast(shape.len);
                                    copyCompileSupportShape(&current_ne, shape);
                                }
                                edge_lens_known = true;
                            }
                        } else if (current_len) |known_len| {
                            if (known_len != step_input_len) {
                                return .{
                                    .supported = false,
                                    .reason = "nn.Sequential.compileSupport detected a layer shape mismatch",
                                    .layer_count = self.steps.len,
                                };
                            }
                            if (current_n_dims) |known_n_dims| {
                                if (support.inputShape()) |step_input_shape| {
                                    if (!std.mem.eql(usize, current_ne[0..@as(usize, known_n_dims)], step_input_shape)) {
                                        return .{
                                            .supported = false,
                                            .reason = "nn.Sequential.compileSupport detected a layer shape mismatch",
                                            .layer_count = self.steps.len,
                                        };
                                    }
                                } else {
                                    current_n_dims = null;
                                }
                            }
                            current_len = step_output_len;
                            if (support.outputShape()) |shape| {
                                current_n_dims = @intCast(shape.len);
                                copyCompileSupportShape(&current_ne, shape);
                            } else {
                                current_n_dims = null;
                            }
                        }
                    }
                } else if (edge_lens_known and !support.shape_preserving) {
                    current_len = null;
                    current_n_dims = null;
                    edge_lens_known = false;
                }
                has_native_or_composable = true;
            }
            if (has_native_or_composable) {
                const lengths = compileSupportParameterLengths(T, self.namedParameters());
                return .{
                    .supported = true,
                    .native_path = "device-program",
                    .model_kind = "module",
                    .layer_count = self.steps.len,
                    .input_n_dims = model_input_n_dims,
                    .input_ne = model_input_ne,
                    .output_n_dims = if (edge_lens_known) current_n_dims else null,
                    .output_ne = current_ne,
                    .input_len = model_input_len,
                    .output_len = if (edge_lens_known) current_len else null,
                    .weights_len = lengths.weights_len,
                    .bias_len = lengths.bias_len,
                    .parameter_len = lengths.parameter_len,
                };
            }
            return .{
                .supported = false,
                .reason = "nn.Sequential.compile requires at least one layer",
                .layer_count = self.steps.len,
            };
        }

        pub fn canCompile(self: *Self) bool {
            return self.compileSupport().canCompile();
        }

        pub fn train(self: *Self, mode: bool) *Self {
            for (self.steps) |step| step.train_fn(step.ctx, mode);
            return self;
        }

        pub fn eval(self: *Self) *Self {
            return self.train(false);
        }

        pub fn compile(self: *Self, alloc: Alloc, options: ModuleCompileOptions) !ModuleProgram(T) {
            if (!self.canCompile()) return error.UnsupportedCompile;
            return ModuleProgram(T).compile(alloc, self, options);
        }

        pub fn deinit(self: *Self) void {
            destroySteps(self.alloc, self.steps);
            for (self.named_name_buf[0..self.named_param_len]) |name| self.alloc.free(name);
            self.alloc.free(self.named_name_buf);
            self.alloc.free(self.named_param_buf);
            self.alloc.free(self.param_buf);
            self.* = undefined;
        }
    };
}

/// Owned sequence of modules that forwards through each layer in order.
///
/// The sequence is compile-time typed and decomposes to each layer's primitive
/// tensor ops; it does not add a graph op or backend obligation.
pub fn Sequential(comptime T: type) type {
    return struct {
        const Model = SequentialModel(T);

        pub fn init(alloc: Alloc, layers: anytype) Alloc.Error!Model {
            const fields = std.meta.fields(@TypeOf(layers));
            const steps = try alloc.alloc(Model.Step, fields.len);
            var initialized: usize = 0;
            errdefer Model.destroySteps(alloc, steps[0..initialized]);

            inline for (fields) |field| {
                const Module = field.type;
                if (@typeInfo(Module) == .pointer) {
                    @compileError("nn.Sequential owns modules by value; pass layer values instead of pointers");
                }
                const module = try alloc.create(Module);
                module.* = @field(layers, field.name);
                steps[initialized] = .{
                    .ctx = module,
                    .forward_fn = struct {
                        fn call(ctx: *anyopaque, x: *Tensor(T)) *Tensor(T) {
                            return boxedForward(T, Module, ctx, x);
                        }
                    }.call,
                    .parameters_fn = struct {
                        fn call(ctx: *anyopaque) []const *Tensor(T) {
                            return boxedParameters(T, Module, ctx);
                        }
                    }.call,
                    .named_parameters_fn = struct {
                        fn call(ctx: *anyopaque) []const NamedParameter(T) {
                            return boxedNamedParameters(T, Module, ctx);
                        }
                    }.call,
                    .compile_support_fn = struct {
                        fn call(ctx: *anyopaque) CompileSupport {
                            return boxedCompileSupport(Module, ctx);
                        }
                    }.call,
                    .train_fn = struct {
                        fn call(ctx: *anyopaque, mode: bool) void {
                            boxedTrain(Module, ctx, mode);
                        }
                    }.call,
                    .deinit_fn = struct {
                        fn call(a: Alloc, ctx: *anyopaque) void {
                            boxedDeinit(Module, a, ctx);
                        }
                    }.call,
                };
                initialized += 1;
            }

            var param_count: usize = 0;
            for (steps) |step| {
                param_count += step.parameters_fn(step.ctx).len;
            }
            const param_buf = try alloc.alloc(*Tensor(T), param_count);
            errdefer alloc.free(param_buf);

            const named_param_buf = try alloc.alloc(NamedParameter(T), param_count);
            errdefer alloc.free(named_param_buf);
            const named_name_buf = try alloc.alloc([]u8, param_count);
            var named_param_len: usize = 0;
            errdefer {
                for (named_name_buf[0..named_param_len]) |name| alloc.free(name);
                alloc.free(named_name_buf);
            }

            for (steps, 0..) |step, step_index| {
                const named = step.named_parameters_fn(step.ctx);
                for (named) |param| {
                    const name = try std.fmt.allocPrint(alloc, "{d}.{s}", .{ step_index, param.name });
                    named_name_buf[named_param_len] = name;
                    named_param_buf[named_param_len] = .{ .name = name, .tensor = param.tensor, .layout = param.layout };
                    named_param_len += 1;
                }
            }

            return .{
                .alloc = alloc,
                .steps = steps,
                .param_buf = param_buf,
                .named_param_buf = named_param_buf,
                .named_name_buf = named_name_buf,
                .named_param_len = named_param_len,
            };
        }
    };
}

/// Convenience constructor for `Sequential(T).init(...)`.
pub fn sequential(comptime T: type, alloc: Alloc, layers: anytype) Alloc.Error!SequentialModel(T) {
    return Sequential(T).init(alloc, layers);
}

/// Mean squared error reduced to a scalar loss.
pub fn meanSquaredError(comptime T: type, pred: *Tensor(T), target: *Tensor(T)) *Tensor(T) {
    return loss_mod.meanSquaredError(T, pred, target);
}

/// Build typed class-index targets for `crossEntropy`.
pub fn classTargets(comptime T: type, alloc: Alloc, classes: []const usize) Alloc.Error!*Tensor(T) {
    return loss_mod.classTargets(T, alloc, classes);
}

/// Collect trainable parameters from a tuple of trainable module pointers and
/// parameterless module values.
pub fn parameters(comptime T: type, alloc: Alloc, modules: anytype) Alloc.Error![]*Tensor(T) {
    const fields = std.meta.fields(@TypeOf(modules));
    var count: usize = 0;
    inline for (fields) |field| {
        count += moduleParameters(T, @field(modules, field.name)).len;
    }

    const out = try alloc.alloc(*Tensor(T), count);
    var cursor: usize = 0;
    inline for (fields) |field| {
        const params = moduleParameters(T, @field(modules, field.name));
        @memcpy(out[cursor..][0..params.len], params);
        cursor += params.len;
    }
    return out;
}

/// Cross-entropy over per-row class logits.
///
/// `logits` has shape `{n_classes, batch}` and `targets` has shape `{batch}`,
/// where each target is the class index for that sample.
pub fn crossEntropy(comptime T: type, logits: *Tensor(T), targets: *Tensor(T)) *Tensor(T) {
    return loss_mod.crossEntropy(T, logits, targets);
}

/// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
pub fn silu(comptime T: type, x: *Tensor(T)) *Tensor(T) {
    return siluTensor(x);
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
pub fn sigmoid(comptime T: type, x: *Tensor(T)) *Tensor(T) {
    return sigmoidTensor(x);
}

/// Hyperbolic tangent activation.
pub fn tanh(comptime T: type, x: *Tensor(T)) *Tensor(T) {
    return tanhTensor(x);
}

/// Argmax over rows: for each column, find the row index with the largest value.
pub fn argmax(comptime T: type, logits: *const Tensor(T), preds: []usize) void {
    const n_classes = logits.ne[0];
    const batch = logits.ne[1];
    std.debug.assert(preds.len >= batch);
    for (0..batch) |s| {
        var best_class: usize = 0;
        var best_val: T = logits.data[s * n_classes];
        for (1..n_classes) |c| {
            const val = logits.data[s * n_classes + c];
            if (val > best_val) {
                best_val = val;
                best_class = c;
            }
        }
        preds[s] = best_class;
    }
}

/// Initialize tensor weights with Kaiming uniform distribution.
///
/// Draws from U(-bound, +bound) where bound = sqrt(6 / fan_in).
/// Standard initialization for layers followed by ReLU.
///
/// For 2D weights {out_features, in_features}: fan_in = in_features.
/// For 3D+ conv kernels {kw, kh, c_in, c_out}: fan_in = kw * kh * c_in.
pub fn kaimingUniform(comptime T: type, tensor: *Tensor(T), seed: u64) void {
    const fan_in: usize = if (tensor.n_dims == 2)
        tensor.ne[1]
    else blk: {
        var fi: usize = 1;
        for (tensor.ne[0 .. tensor.n_dims - 1]) |d| fi *= d;
        break :blk fi;
    };
    const bound: f32 = @sqrt(6.0 / @as(f32, @floatFromInt(fan_in)));
    var rng = std.Random.DefaultPrng.init(seed);
    var random = rng.random();
    for (tensor.data) |*d| {
        d.* = @floatCast((random.float(f32) * 2.0 - 1.0) * bound);
    }
}

/// Initialize tensor with uniform random values in [low, high).
pub fn uniform(comptime T: type, tensor: *Tensor(T), low: T, high: T, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    var random = rng.random();
    const range: f32 = @floatCast(high - low);
    const lo: f32 = @floatCast(low);
    for (tensor.data) |*d| {
        d.* = @floatCast(random.float(f32) * range + lo);
    }
}

const testing = std.testing;
const tac = testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

fn expectApproxSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try testing.expectApproxEqAbs(want, got, tolerance);
    }
}

fn expectUnaryActivationModule(module: anytype, input_values: []const f32, expected_values: []const f32) !void {
    try testing.expectEqual(input_values.len, expected_values.len);

    var layer = module;
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expect(support.shape_preserving);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const shape = [_]usize{input_values.len};
    const x = try Tensor(f32).init(a, shape[0..]);
    x.setData(input_values);
    const y = layer.forward(x);
    try g.infer(y);
    try expectApproxSlices(expected_values, y.data, 1e-5);

    var program = try layer.compile(tac, .{ .input_shape = shape[0..] });
    defer program.deinit();
    const program_support = program.compileSupport();
    try testing.expect(program_support.supported);
    try testing.expect(program_support.shape_preserving);
    try testing.expectEqual(input_values.len, program_support.input_len);
    try testing.expectEqual(expected_values.len, program_support.output_len);

    const input = try tac.dupe(f32, input_values);
    defer tac.free(input);
    const output = try tac.alloc(f32, expected_values.len);
    defer tac.free(output);
    @memset(output, 0);

    var session = try program.bind(input, output);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(expected_values, output, 1e-5);
}

fn expectReductionModule(module: anytype, input_shape: []const usize, input_values: []const f32, expected_shape: []const usize, expected_values: []const f32) !void {
    var layer = module;
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expect(!support.shape_preserving);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const x = try Tensor(f32).init(a, input_shape);
    x.setData(input_values);
    const y = layer.forward(x);
    try g.infer(y);
    try testing.expectEqualSlices(usize, expected_shape, y.ne[0..y.n_dims]);
    try expectApproxSlices(expected_values, y.data, 1e-5);

    var program = try layer.compile(tac, .{ .input_shape = input_shape });
    defer program.deinit();
    const program_support = program.compileSupport();
    try testing.expect(program_support.supported);
    try testing.expectEqualSlices(usize, input_shape, program_support.inputShape().?);
    try testing.expectEqualSlices(usize, expected_shape, program_support.outputShape().?);
    try testing.expectEqualSlices(usize, expected_shape, program.outputShape());

    const input = try tac.dupe(f32, input_values);
    defer tac.free(input);
    const output = try tac.alloc(f32, expected_values.len);
    defer tac.free(output);
    @memset(output, 0);

    var session = try program.bind(input, output);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(expected_values, output, 1e-5);
}

fn expectShapeModule(module: anytype, input_shape: []const usize, input_values: []const f32, expected_shape: []const usize, expected_values: []const f32, shape_preserving: bool) !void {
    var layer = module;
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expectEqual(shape_preserving, support.shape_preserving);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const x = try Tensor(f32).init(a, input_shape);
    x.setData(input_values);
    const y = layer.forward(x);
    try g.infer(y);
    try testing.expectEqualSlices(usize, expected_shape, y.ne[0..y.n_dims]);
    try expectApproxSlices(expected_values, y.data, 1e-5);

    var program = try layer.compile(tac, .{ .input_shape = input_shape });
    defer program.deinit();
    const program_support = program.compileSupport();
    try testing.expect(program_support.supported);
    try testing.expectEqualSlices(usize, input_shape, program_support.inputShape().?);
    try testing.expectEqualSlices(usize, expected_shape, program_support.outputShape().?);
    try testing.expectEqualSlices(usize, expected_shape, program.outputShape());

    const input = try tac.dupe(f32, input_values);
    defer tac.free(input);
    const output = try tac.alloc(f32, expected_values.len);
    defer tac.free(output);
    @memset(output, 0);

    var session = try program.bind(input, output);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(expected_values, output, 1e-5);
}

test "nn native helper surface stays curated" {
    const root = @This();
    const expected = .{
        "native_helper_manifest",
        "LinearConfig",
        "EmbeddingConfig",
        "LayerNormConfig",
        "RmsNormConfig",
        "DropoutConfig",
        "DropoutError",
        "NamedParameter",
        "ModuleParameterInfo",
        "ModuleCompatibilityCode",
        "ModuleCompatibility",
        "CompileSupport",
        "CompileError",
        "ModuleCompileOptions",
        "LinearCompileOptions",
        "StateLoadError",
        "StateEntry",
        "stateDict",
        "deinitStateDict",
        "loadStateDict",
        "compileSupportForInputShape",
        "ModuleProgram",
        "LinearProgram",
        "ModuleSession",
        "LinearSession",
        "linear",
        "Linear",
        "Embedding",
        "LayerNorm",
        "RmsNorm",
        "Gelu",
        "Relu",
        "Silu",
        "Sigmoid",
        "Tanh",
        "Exp",
        "Log",
        "Neg",
        "Recip",
        "Abs",
        "Sqrt",
        "Square",
        "Sgn",
        "StepActivation",
        "Dropout",
        "Softmax",
        "LogSoftmax",
        "Sum",
        "Mean",
        "Max",
        "Min",
        "Identity",
        "Reshape",
        "View",
        "Transpose",
        "BroadcastTo",
        "Expand",
        "Flatten",
        "gelu",
        "relu",
        "siluActivation",
        "sigmoidActivation",
        "tanhActivation",
        "exp",
        "log",
        "neg",
        "recip",
        "abs",
        "sqrt",
        "square",
        "sqr",
        "sgn",
        "sign",
        "stepActivation",
        "dropout",
        "softmax",
        "logSoftmax",
        "sum",
        "mean",
        "max",
        "min",
        "identity",
        "reshape",
        "view",
        "transpose",
        "broadcastTo",
        "expand",
        "flatten",
        "flattenAll",
        "Sequential",
        "sequential",
        "meanSquaredError",
        "classTargets",
        "parameters",
        "crossEntropy",
        "silu",
        "sigmoid",
        "tanh",
        "argmax",
        "kaimingUniform",
        "uniform",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
    try testing.expectEqualStrings("zgml-native-helper", native_helper_manifest.kind);
    try testing.expectEqualStrings("nn", native_helper_manifest.domain);
    try testing.expectEqual(false, native_helper_manifest.product_frontend);
    try testing.expectEqualStrings("src/ts/**", native_helper_manifest.product_owner);
    try testing.expectEqualStrings("forbidden", native_helper_manifest.product_policy);
    try testing.expectEqualStrings("native-helper-substrate", native_helper_manifest.role);
    try testing.expectEqualStrings("Program/Session/ABI contracts", native_helper_manifest.alignment_boundary);
}

test "linear composes matmul plus bias" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const b = try Tensor(f32).init(a, &.{3});
    b.setData(&.{ 10, 20, 30 });

    const y = linear(f32, x, w, b);
    try g.infer(y);

    try testing.expectEqualSlices(f32, &.{ 19, 32, 45, 29, 46, 63, 39, 60, 81 }, y.data);
}

test "linear without bias is plain matmul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 2, 3, 4, 5, 6 });

    const y = linear(f32, x, w, null);
    try g.infer(y);

    try testing.expectEqualSlices(f32, &.{ 9, 12, 15, 19, 26, 33, 29, 40, 51 }, y.data);
}

test "linear backward produces parameter gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 1 });
    x.setData(&.{ 1, 2 });
    const w = try Tensor(f32).init(a, &.{ 3, 2 });
    w.setData(&.{ 1, 0, 0, 1, 1, 1 });
    w.setParam();
    const b = try Tensor(f32).init(a, &.{3});
    b.setData(&.{ 0, 0, 0 });
    b.setParam();

    const loss = linear(f32, x, w, b).sumAll();
    try g.run(loss);

    var has_nonzero_w = false;
    for (w.grad.?.data) |v| {
        if (v != 0) has_nonzero_w = true;
    }
    var has_nonzero_b = false;
    for (b.grad.?.data) |v| {
        if (v != 0) has_nonzero_b = true;
    }
    try testing.expect(has_nonzero_w);
    try testing.expect(has_nonzero_b);
}

test "Linear layer owns trainable parameters" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();

    var layer = try Linear(f32).init(g.allocator(), 2, 3, .{ .seed = 7 });
    const params = layer.parameters();
    try testing.expectEqual(@as(usize, 2), params.len);
    try testing.expect(params[0] == layer.weight);
    try testing.expect(params[1] == layer.bias.?);
    try testing.expect(params[0].grad != null);
    try testing.expect(params[1].grad != null);
    const named = layer.namedParameters();
    try testing.expectEqualStrings("weight", named[0].name);
    try testing.expect(named[0].tensor == layer.weight);
    try testing.expectEqualStrings(linear_weight_layout, named[0].layout.?);
    try testing.expectEqualStrings("bias", named[1].name);
    try testing.expect(named[1].tensor == layer.bias.?);
    try testing.expectEqualStrings(linear_bias_layout, named[1].layout.?);

    const x = try Tensor(f32).init(g.allocator(), &.{ 2, 1 });
    x.setData(&.{ 1, 2 });
    const loss = layer.forward(x).sumAll();
    try g.run(loss);

    try testing.expect(layer.weight.grad != null);
    try testing.expect(layer.bias.?.grad != null);
}

test "Linear layer supports no-bias parameter list and deinit" {
    var layer = try Linear(f32).init(tac, 2, 1, .{ .bias = false, .seed = 11 });
    defer layer.deinit();

    const params = layer.parameters();
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expect(params[0] == layer.weight);
    try testing.expect(layer.bias == null);
    const named = layer.namedParameters();
    try testing.expectEqual(@as(usize, 1), named.len);
    try testing.expectEqualStrings("weight", named[0].name);
    try testing.expect(named[0].tensor == layer.weight);
    try testing.expectEqualStrings(linear_weight_layout, named[0].layout.?);
}

test "Linear compileSupport reports native tiny-linear compiler support" {
    var layer = try Linear(f32).init(tac, 2, 3, .{ .seed = 7 });
    defer layer.deinit();

    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expect(support.reason == null);
    try testing.expectEqualStrings("tiny-linear", support.native_path.?);
    try testing.expectEqualStrings("tiny-linear", support.model_kind.?);
    try testing.expectEqual(@as(?usize, 1), support.layer_count);
    try testing.expectEqualSlices(usize, &.{2}, support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{3}, support.outputShape().?);
    try testing.expectEqual(@as(?usize, 2), support.input_len);
    try testing.expectEqual(@as(?usize, 3), support.output_len);
    try testing.expectEqual(@as(?usize, 6), support.weights_len);
    try testing.expectEqual(@as(?usize, 3), support.bias_len);
    try testing.expectEqual(@as(?usize, 9), support.parameter_len);

    var no_bias = try Linear(f32).init(tac, 4, 2, .{ .bias = false, .seed = 8 });
    defer no_bias.deinit();

    const no_bias_support = no_bias.compileSupport();
    try testing.expect(no_bias_support.supported);
    try testing.expectEqual(@as(?usize, 1), no_bias_support.layer_count);
    try testing.expectEqualSlices(usize, &.{4}, no_bias_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{2}, no_bias_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 4), no_bias_support.input_len);
    try testing.expectEqual(@as(?usize, 2), no_bias_support.output_len);
    try testing.expectEqual(@as(?usize, 8), no_bias_support.weights_len);
    try testing.expectEqual(@as(?usize, 0), no_bias_support.bias_len);
    try testing.expectEqual(@as(?usize, 8), no_bias_support.parameter_len);
}

test "compileSupportForInputShape reports exact cold module trace evidence" {
    var reducer = sum(0);
    const reduction_support = try compileSupportForInputShape(f32, tac, &reducer, &.{ 3, 2 });
    try testing.expect(reduction_support.supported);
    try testing.expectEqualStrings("device-program", reduction_support.native_path.?);
    try testing.expectEqualStrings("module", reduction_support.model_kind.?);
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, reduction_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{ 1, 2 }, reduction_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 6), reduction_support.input_len);
    try testing.expectEqual(@as(?usize, 2), reduction_support.output_len);
    try testing.expectEqual(@as(?usize, 0), reduction_support.weights_len);
    try testing.expectEqual(@as(?usize, 0), reduction_support.bias_len);
    try testing.expectEqual(@as(?usize, 0), reduction_support.parameter_len);
    try testing.expect(reduction_support.ir_op_count.? > 0);
    try testing.expect(reduction_support.ir_normalized_step_hash.? != 0);
    try testing.expect(reduction_support.ir_normalized_shape_hash.? != 0);

    var flatten_layer = flattenAll();
    const flatten_support = try compileSupportForInputShape(f32, tac, &flatten_layer, &.{ 2, 3 });
    try testing.expect(flatten_support.supported);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, flatten_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{6}, flatten_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 6), flatten_support.input_len);
    try testing.expectEqual(@as(?usize, 6), flatten_support.output_len);

    var model = try sequential(f32, tac, .{
        flattenAll(),
        try Linear(f32).init(tac, 4, 2, .{ .seed = 17 }),
    });
    defer model.deinit();

    const generic_support = model.compileSupport();
    try testing.expect(generic_support.supported);
    try testing.expect(generic_support.inputShape() == null);
    try testing.expect(generic_support.outputShape() == null);

    const shaped_support = try compileSupportForInputShape(f32, tac, &model, &.{ 2, 2 });
    try testing.expect(shaped_support.supported);
    try testing.expectEqualStrings("device-program", shaped_support.native_path.?);
    try testing.expectEqualStrings("module", shaped_support.model_kind.?);
    try testing.expectEqual(@as(?usize, 2), shaped_support.layer_count);
    try testing.expectEqualSlices(usize, &.{ 2, 2 }, shaped_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{2}, shaped_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 4), shaped_support.input_len);
    try testing.expectEqual(@as(?usize, 2), shaped_support.output_len);
    try testing.expectEqual(@as(?usize, 8), shaped_support.weights_len);
    try testing.expectEqual(@as(?usize, 2), shaped_support.bias_len);
    try testing.expectEqual(@as(?usize, 10), shaped_support.parameter_len);
    try testing.expect(shaped_support.ir_op_count.? > 0);
    try testing.expect(shaped_support.ir_normalized_step_hash.? != 0);
    try testing.expect(shaped_support.ir_normalized_shape_hash.? != 0);

    var program = try model.compile(tac, .{ .input_shape = &.{ 2, 2 } });
    defer program.deinit();
    const program_support = program.compileSupport();
    try testing.expectEqualSlices(usize, shaped_support.inputShape().?, program_support.inputShape().?);
    try testing.expectEqualSlices(usize, shaped_support.outputShape().?, program_support.outputShape().?);
    try testing.expectEqual(shaped_support.parameter_len, program_support.parameter_len);
    try testing.expectEqual(shaped_support.ir_normalized_step_hash.?, program_support.ir_normalized_step_hash.?);
    try testing.expectEqual(shaped_support.ir_normalized_shape_hash.?, program_support.ir_normalized_shape_hash.?);

    try testing.expectError(error.InvalidInputShape, compileSupportForInputShape(f32, tac, &reducer, &.{}));
}

test "Linear compiles and executes through native Program Session" {
    var layer = try Linear(f32).init(tac, 2, 3, .{ .seed = 7 });
    defer layer.deinit();
    layer.weight.setData(&.{ 1, 2, 3, 4, 5, 6 });
    layer.bias.?.setData(&.{ 10, 20, 30 });

    var program = try layer.compile(tac, .{});
    defer program.deinit();

    const info = program.inspect();
    try testing.expectEqual(@import("backend.zig").Device.cpu, info.backend);
    try testing.expect(info.execution_supported);
    try testing.expect(info.op_count >= 2);
    try testing.expectEqual(@as(usize, 2), program.inputLen());
    try testing.expectEqual(@as(usize, 3), program.outputLen());

    var input = [_]f32{ 10, 20 };
    var output = [_]f32{ 0, 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();

    _ = try session.step();
    try testing.expectEqualSlices(f32, &.{ 100, 140, 180 }, &output);

    layer.weight.setData(&.{ 2, 0, 0, 3, 4, 0 });
    layer.bias.?.setData(&.{ 1, 2, 3 });
    try session.uploadParameters();
    _ = try session.execute(&.{ 5, 7 });
    try testing.expectEqualSlices(f32, &.{ 32, 30, 3 }, &output);
}

test "Linear compiles feature-major batched input through traced Program Session" {
    var layer = try Linear(f32).init(tac, 2, 3, .{ .seed = 7 });
    defer layer.deinit();
    layer.weight.setData(&.{ 1, 2, 3, 4, 5, 6 });
    layer.bias.?.setData(&.{ 10, 20, 30 });

    var program = try layer.compile(tac, .{ .input_shape = &.{ 2, 2 } });
    defer program.deinit();

    const support = program.compileSupport();
    try testing.expect(support.supported);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);
    try testing.expectEqualSlices(usize, &.{ 2, 2 }, program.inputShape());
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, program.outputShape());
    try testing.expectEqual(@as(usize, 4), program.inputLen());
    try testing.expectEqual(@as(usize, 6), program.outputLen());
    try testing.expect(support.ir_normalized_step_hash.? != 0);

    var input = [_]f32{ 10, 20, 30, 40 };
    var output = [_]f32{0} ** 6;
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();

    _ = try session.step();
    try testing.expectEqualSlices(f32, &.{ 100, 140, 180, 200, 280, 360 }, &output);

    layer.bias.?.setData(&.{ 1, 2, 3 });
    try session.uploadParameters();
    _ = try session.execute(&.{ 5, 7, 11, 13 });
    try testing.expectEqualSlices(f32, &.{ 34, 47, 60, 64, 89, 114 }, &output);
}

test "Linear compile rejects invalid input shapes" {
    var layer = try Linear(f32).init(tac, 2, 3, .{ .seed = 7 });
    defer layer.deinit();

    try testing.expectError(error.InvalidInputShape, layer.compile(tac, .{ .input_shape = &.{ 2, 1, 1 } }));
    try testing.expectError(error.InvalidInputShape, layer.compile(tac, .{ .input_shape = &.{3} }));
    try testing.expectError(error.InvalidInputShape, layer.compile(tac, .{ .input_shape = &.{ 3, 1 } }));
}

test "LayerNorm module normalizes feature axis and owns affine parameters" {
    var layer = try LayerNorm(f32).init(tac, 2, .{ .eps = 0 });
    defer layer.deinit();

    const params = layer.parameters();
    try testing.expectEqual(@as(usize, 2), params.len);
    try testing.expect(params[0] == layer.weight.?);
    try testing.expect(params[1] == layer.bias.?);
    const named = layer.namedParameters();
    try testing.expectEqualStrings("weight", named[0].name);
    try testing.expect(named[0].tensor == layer.weight.?);
    try testing.expectEqualStrings("bias", named[1].name);
    try testing.expect(named[1].tensor == layer.bias.?);
    try testing.expectEqualSlices(f32, &.{ 1, 1 }, layer.weight.?.data);
    try testing.expectEqualSlices(f32, &.{ 0, 0 }, layer.bias.?.data);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 2 });
    x.setData(&.{ 1, 3, 2, 4 });

    const out = layer.forward(x);
    try g.infer(out);

    try testing.expectApproxEqAbs(@as(f32, -1), out.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1), out.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1), out.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1), out.data[3], 1e-5);
}

test "LayerNorm backward restores durable affine gradients" {
    var layer = try LayerNorm(f32).init(tac, 2, .{ .eps = 1e-5 });
    defer layer.deinit();

    const params = layer.parameters();
    const weight_grad_buf = params[0].paramGradOrNull().?;
    const bias_grad_buf = params[1].paramGradOrNull().?;

    {
        var g = ComputeGraph(f32).init(tac);
        defer g.deinit();
        const a = g.allocator();

        const x = try Tensor(f32).init(a, &.{ 2, 2 });
        x.setData(&.{ 1, 3, 2, 5 });
        const target = try Tensor(f32).init(a, &.{ 2, 2 });
        target.setData(&.{ 0.25, -0.25, -0.5, 0.5 });

        const loss = meanSquaredError(f32, layer.forward(x), target);
        try g.run(loss);

        try testing.expect(params[0].grad != null);
        try testing.expect(params[0].grad.? != weight_grad_buf);
        try testing.expect(params[1].grad != null);
        try testing.expect(params[1].grad.? != bias_grad_buf);
    }

    try testing.expect(params[0].grad.? == weight_grad_buf);
    try testing.expect(params[1].grad.? == bias_grad_buf);

    var saw_weight_grad = false;
    for (weight_grad_buf.data) |v| {
        if (v != 0) saw_weight_grad = true;
    }
    var saw_bias_grad = false;
    for (bias_grad_buf.data) |v| {
        if (v != 0) saw_bias_grad = true;
    }
    try testing.expect(saw_weight_grad);
    try testing.expect(saw_bias_grad);
}

test "RmsNorm module normalizes feature axis and owns scale" {
    var layer = try RmsNorm(f32).init(tac, 2, .{ .eps = 0 });
    defer layer.deinit();

    const params = layer.parameters();
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expect(params[0] == layer.weight.?);
    const named = layer.namedParameters();
    try testing.expectEqual(@as(usize, 1), named.len);
    try testing.expectEqualStrings("weight", named[0].name);
    try testing.expect(named[0].tensor == layer.weight.?);
    try testing.expectEqualSlices(f32, &.{ 1, 1 }, layer.weight.?.data);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 2 });
    x.setData(&.{ 3, 4, 0, 5 });

    const out = layer.forward(x);
    try g.infer(out);

    const inv0 = @as(f32, 1) / @sqrt(@as(f32, 12.5));
    const inv1 = @as(f32, 1) / @sqrt(@as(f32, 12.5));
    try testing.expectApproxEqAbs(@as(f32, 3) * inv0, out.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4) * inv0, out.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0) * inv1, out.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5) * inv1, out.data[3], 1e-5);
}

test "RmsNorm backward restores durable scale gradient" {
    var layer = try RmsNorm(f32).init(tac, 2, .{ .eps = 1e-5 });
    defer layer.deinit();

    const params = layer.parameters();
    const grad_buf = params[0].paramGradOrNull().?;

    {
        var g = ComputeGraph(f32).init(tac);
        defer g.deinit();
        const a = g.allocator();

        const x = try Tensor(f32).init(a, &.{ 2, 2 });
        x.setData(&.{ 1, 3, 2, 5 });
        const target = try Tensor(f32).init(a, &.{ 2, 2 });
        target.setData(&.{ 0.25, -0.25, -0.5, 0.5 });

        const loss = meanSquaredError(f32, layer.forward(x), target);
        try g.run(loss);

        try testing.expect(params[0].grad != null);
        try testing.expect(params[0].grad.? != grad_buf);
    }

    try testing.expect(params[0].grad.? == grad_buf);

    var saw_grad = false;
    for (grad_buf.data) |v| {
        if (v != 0) saw_grad = true;
    }
    try testing.expect(saw_grad);
}

test "Sequential forwards through layers and collects trainable parameters" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var model = try Sequential(f32).init(tac, .{
        try Linear(f32).init(a, 2, 3, .{ .seed = 1 }),
        gelu(),
        try Linear(f32).init(a, 3, 1, .{ .seed = 2 }),
    });
    defer model.deinit();

    const params = model.parameters();
    try testing.expectEqual(@as(usize, 4), params.len);
    for (params) |param| {
        try testing.expect(param.grad != null);
    }
    const named = model.namedParameters();
    try testing.expectEqual(@as(usize, 4), named.len);
    try testing.expectEqualStrings("0.weight", named[0].name);
    try testing.expect(named[0].tensor == params[0]);
    try testing.expectEqualStrings("0.bias", named[1].name);
    try testing.expectEqualStrings("2.weight", named[2].name);
    try testing.expectEqualStrings("2.bias", named[3].name);

    const x = try Tensor(f32).init(a, &.{ 2, 1 });
    x.setData(&.{ 1, 2 });
    const loss = model.forward(x).sumAll();
    try g.run(loss);

    var saw_grad = false;
    for (params) |param| {
        for (param.grad.?.data) |v| {
            if (v != 0) saw_grad = true;
        }
    }
    try testing.expect(saw_grad);
}

test "Sequential compileSupport and compile cover multi-layer native module programs" {
    var single = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
    });
    defer single.deinit();

    const single_support = single.compileSupport();
    try testing.expect(single_support.supported);
    try testing.expect(single.canCompile());
    try testing.expectEqualStrings("tiny-linear", single_support.native_path.?);
    try testing.expectEqualStrings("tiny-linear", single_support.model_kind.?);
    try testing.expectEqual(@as(?usize, 1), single_support.layer_count);
    try testing.expectEqualSlices(usize, &.{2}, single_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{3}, single_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 2), single_support.input_len);
    try testing.expectEqual(@as(?usize, 3), single_support.output_len);
    try testing.expectEqual(@as(?usize, 6), single_support.weights_len);
    try testing.expectEqual(@as(?usize, 3), single_support.bias_len);
    try testing.expectEqual(@as(?usize, 9), single_support.parameter_len);

    var single_program = try single.compile(tac, .{});
    defer single_program.deinit();
    try testing.expectEqual(@as(usize, 2), single_program.inputLen());
    try testing.expectEqualSlices(usize, &.{2}, single_program.inputShape());
    try testing.expectEqual(@as(usize, 3), single_program.outputLen());
    try testing.expectEqualSlices(usize, &.{3}, single_program.outputShape());
    try testing.expectEqual(@as(usize, 2), single_program.parameterTensorCount());
    try testing.expectEqual(@as(usize, 9), single_program.parameterLen());
    try testing.expectEqual(@as(?usize, 6), single_program.parameterTensorLen(0));
    try testing.expectEqual(@as(?usize, 3), single_program.parameterTensorLen(1));
    try testing.expectEqual(@as(?usize, null), single_program.parameterTensorLen(2));
    try testing.expectEqualStrings("0.weight", single_program.parameterTensorName(0).?);
    try testing.expectEqualStrings("0.bias", single_program.parameterTensorName(1).?);
    try testing.expect(single_program.parameterTensorName(2) == null);
    try testing.expectEqualStrings(linear_weight_layout, single_program.parameterTensorLayout(0).?);
    try testing.expectEqualStrings(linear_bias_layout, single_program.parameterTensorLayout(1).?);
    try testing.expect(single_program.parameterTensorLayout(2) == null);
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, single_program.parameterTensorShape(0).?);
    try testing.expectEqualSlices(usize, &.{3}, single_program.parameterTensorShape(1).?);
    try testing.expect(single_program.parameterTensorShape(2) == null);

    var multi = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        relu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer multi.deinit();

    const multi_support = multi.compileSupport();
    try testing.expect(multi_support.supported);
    try testing.expect(multi.canCompile());
    try testing.expect(multi_support.layer_count != null);
    try testing.expectEqual(@as(usize, 3), multi_support.layer_count.?);
    try testing.expectEqualStrings("device-program", multi_support.native_path.?);
    try testing.expectEqualStrings("module", multi_support.model_kind.?);
    try testing.expectEqualSlices(usize, &.{2}, multi_support.inputShape().?);
    try testing.expectEqualSlices(usize, &.{2}, multi_support.outputShape().?);
    try testing.expectEqual(@as(?usize, 2), multi_support.input_len);
    try testing.expectEqual(@as(?usize, 2), multi_support.output_len);
    try testing.expectEqual(@as(?usize, 12), multi_support.weights_len);
    try testing.expectEqual(@as(?usize, 5), multi_support.bias_len);
    try testing.expectEqual(@as(?usize, 17), multi_support.parameter_len);

    const named = multi.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0, 1, 0 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ 10, 20 });

    var multi_program = try multi.compile(tac, .{});
    defer multi_program.deinit();
    try testing.expectEqual(@as(usize, 2), multi_program.inputLen());
    try testing.expectEqualSlices(usize, &.{2}, multi_program.inputShape());
    try testing.expectEqual(@as(usize, 2), multi_program.outputLen());
    try testing.expectEqualSlices(usize, &.{2}, multi_program.outputShape());
    try testing.expectEqual(@as(usize, 4), multi_program.parameterTensorCount());
    try testing.expectEqual(@as(usize, 17), multi_program.parameterLen());
    try testing.expectEqual(@as(?usize, 6), multi_program.parameterTensorLen(0));
    try testing.expectEqual(@as(?usize, 3), multi_program.parameterTensorLen(1));
    try testing.expectEqual(@as(?usize, 6), multi_program.parameterTensorLen(2));
    try testing.expectEqual(@as(?usize, 2), multi_program.parameterTensorLen(3));
    try testing.expectEqual(@as(?usize, null), multi_program.parameterTensorLen(4));
    try testing.expectEqualStrings("0.weight", multi_program.parameterTensorName(0).?);
    try testing.expectEqualStrings("0.bias", multi_program.parameterTensorName(1).?);
    try testing.expectEqualStrings("2.weight", multi_program.parameterTensorName(2).?);
    try testing.expectEqualStrings("2.bias", multi_program.parameterTensorName(3).?);
    try testing.expect(multi_program.parameterTensorName(4) == null);
    try testing.expectEqualStrings(linear_weight_layout, multi_program.parameterTensorLayout(0).?);
    try testing.expectEqualStrings(linear_bias_layout, multi_program.parameterTensorLayout(1).?);
    try testing.expectEqualStrings(linear_weight_layout, multi_program.parameterTensorLayout(2).?);
    try testing.expectEqualStrings(linear_bias_layout, multi_program.parameterTensorLayout(3).?);
    try testing.expect(multi_program.parameterTensorLayout(4) == null);
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, multi_program.parameterTensorShape(0).?);
    try testing.expectEqualSlices(usize, &.{3}, multi_program.parameterTensorShape(1).?);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, multi_program.parameterTensorShape(2).?);
    try testing.expectEqualSlices(usize, &.{2}, multi_program.parameterTensorShape(3).?);
    try testing.expect(multi_program.parameterTensorShape(4) == null);
    const parameter_infos = multi_program.parameterInfos();
    try testing.expectEqual(@as(usize, 4), parameter_infos.len);
    try testing.expectEqualStrings("0.weight", parameter_infos[0].name);
    try testing.expectEqualStrings(linear_weight_layout, parameter_infos[0].layout.?);
    try testing.expectEqual(@as(usize, 6), parameter_infos[0].len);
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, parameter_infos[0].shape());
    try testing.expectEqualStrings("2.bias", parameter_infos[3].name);
    try testing.expectEqualStrings(linear_bias_layout, parameter_infos[3].layout.?);
    try testing.expectEqual(@as(usize, 2), parameter_infos[3].len);
    try testing.expectEqualSlices(usize, &.{2}, parameter_infos[3].shape());
    try testing.expectEqualStrings(parameter_infos[2].name, multi_program.parameterInfo(2).?.name);
    try testing.expectEqualSlices(usize, parameter_infos[2].shape(), multi_program.parameterInfo(2).?.shape());
    try testing.expect(multi_program.parameterInfo(4) == null);
    try testing.expectEqual(@as(?usize, 3), multi_program.parameterIndex("2.bias"));
    try testing.expect(multi_program.parameterIndex("missing") == null);
    try testing.expectEqualStrings("2.bias", multi_program.parameterInfoByName("2.bias").?.name);
    try testing.expectEqualSlices(usize, &.{2}, multi_program.parameterInfoByName("2.bias").?.shape());
    try testing.expect(multi_program.parameterInfoByName("missing") == null);
    const program_support = multi_program.compileSupport();
    try testing.expect(program_support.supported);
    try testing.expect(multi_program.canCompile());
    try testing.expectEqualStrings("device-program", program_support.native_path.?);
    try testing.expectEqualStrings("module", program_support.model_kind.?);
    try testing.expectEqual(@as(?usize, 3), program_support.layer_count);
    try testing.expectEqualSlices(usize, multi_program.inputShape(), program_support.inputShape().?);
    try testing.expectEqualSlices(usize, multi_program.outputShape(), program_support.outputShape().?);
    try testing.expectEqual(@as(?usize, multi_program.inputLen()), program_support.input_len);
    try testing.expectEqual(@as(?usize, multi_program.outputLen()), program_support.output_len);
    try testing.expectEqual(@as(?usize, 12), program_support.weights_len);
    try testing.expectEqual(@as(?usize, 5), program_support.bias_len);
    try testing.expectEqual(@as(?usize, multi_program.parameterLen()), program_support.parameter_len);
    try testing.expect(multi_program.bindingRequirementHash() != 0);
    try testing.expectEqual(multi_program.parameterTensorCount(), multi_program.persistentRequirementCount());
    try testing.expectEqual(@as(usize, 1), multi_program.stepInputRequirementCount());
    try testing.expectEqual(@as(usize, 1), multi_program.stepOutputRequirementCount());

    var input = [_]f32{ 2, -3 };
    var output = [_]f32{ 0, 0 };
    var session = try multi_program.bind(input[0..], output[0..]);
    defer session.deinit();
    try testing.expectEqualSlices(usize, multi_program.inputShape(), session.inputShape());
    try testing.expectEqualSlices(usize, multi_program.outputShape(), session.outputShape());
    try testing.expectEqual(multi_program.parameterTensorCount(), session.parameterTensorCount());
    try testing.expectEqual(multi_program.parameterLen(), session.parameterLen());
    try testing.expectEqual(multi_program.parameterTensorLen(2), session.parameterTensorLen(2));
    const session_support = session.compileSupport();
    try testing.expect(session_support.supported);
    try testing.expect(session.canCompile());
    try testing.expectEqualSlices(usize, program_support.inputShape().?, session_support.inputShape().?);
    try testing.expectEqualSlices(usize, program_support.outputShape().?, session_support.outputShape().?);
    try testing.expectEqual(program_support.parameter_len, session_support.parameter_len);
    try testing.expectEqual(multi_program.parameterInfos().len, session.parameterInfos().len);
    try testing.expectEqualStrings(multi_program.parameterInfo(1).?.name, session.parameterInfo(1).?.name);
    try testing.expectEqualStrings(multi_program.parameterInfo(1).?.layout.?, session.parameterInfo(1).?.layout.?);
    try testing.expectEqual(multi_program.parameterInfo(1).?.len, session.parameterInfo(1).?.len);
    try testing.expectEqualSlices(usize, multi_program.parameterInfo(1).?.shape(), session.parameterInfo(1).?.shape());
    try testing.expect(session.parameterInfo(4) == null);
    try testing.expectEqual(@as(?usize, 3), session.parameterIndex("2.bias"));
    try testing.expect(session.parameterIndex("missing") == null);
    try testing.expectEqualStrings(multi_program.parameterInfoByName("2.bias").?.name, session.parameterInfoByName("2.bias").?.name);
    try testing.expectEqualSlices(usize, multi_program.parameterInfoByName("2.bias").?.shape(), session.parameterInfoByName("2.bias").?.shape());
    try testing.expect(session.parameterInfoByName("missing") == null);
    try testing.expectEqualStrings(multi_program.parameterTensorName(2).?, session.parameterTensorName(2).?);
    try testing.expectEqualStrings(multi_program.parameterTensorLayout(2).?, session.parameterTensorLayout(2).?);
    try testing.expectEqualSlices(usize, multi_program.parameterTensorShape(2).?, session.parameterTensorShape(2).?);
    try testing.expect(session.parameterTensorName(4) == null);
    try testing.expect(session.parameterTensorLayout(4) == null);
    try testing.expect(session.parameterTensorShape(4) == null);
    const session_inspection = session.inspect();
    try testing.expect(session_inspection.binding_shape_hash != 0);
    try testing.expectEqual(session_inspection.binding_shape_hash, session.bindingShapeHash());
    try testing.expectEqual(multi_program.parameterTensorCount(), session.persistentBindingCount());
    try testing.expectEqual(@as(usize, 1), session.stepInputCount());
    try testing.expectEqual(@as(usize, 1), session.stepOutputCount());
    try testing.expectEqual(@as(usize, multi_program.parameterTensorCount() + 2), session.hostBindingCount());
    try testing.expectEqual(@as(usize, 0), session.resourceBindingCount());

    _ = try session.step();
    try testing.expectEqualSlices(f32, &.{ 22, 26 }, &output);

    named[3].tensor.setData(&.{ -1, -2 });
    _ = try session.execute(&.{ -5, 4 });
    try testing.expectEqualSlices(f32, &.{ 10, 25 }, &output);
    try session.uploadParameterByName("2.bias");
    _ = try session.execute(&.{ -5, 4 });
    try testing.expectEqualSlices(f32, &.{ -1, 3 }, &output);
    try testing.expectError(error.InvalidProgramIO, session.uploadParameter(4));
    try testing.expectError(error.InvalidProgramIO, session.uploadParameterRange(2, 3));
    try testing.expectError(error.MissingParameter, session.uploadParameterByName("missing"));
}

test "Sequential compileSupport reports layer shape mismatches before compile" {
    var bad = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        try Linear(f32).init(tac, 4, 2, .{ .seed = 2 }),
    });
    defer bad.deinit();

    const support = bad.compileSupport();
    try testing.expect(!support.supported);
    try testing.expect(!bad.canCompile());
    try testing.expectEqual(@as(?usize, 2), support.layer_count);
    try testing.expectEqualStrings("nn.Sequential.compileSupport detected a layer shape mismatch", support.reason.?);
}

test "ModuleProgram binds and executes eager Tensor buffers" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        relu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0, 1, 0 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ 10, 20 });

    var program = try model.compile(tac, .{});
    defer program.deinit();
    try testing.expectEqualSlices(usize, &.{2}, program.inputShape());
    try testing.expectEqualSlices(usize, &.{2}, program.outputShape());
    try testing.expectEqual(@as(usize, 4), program.parameterTensorCount());
    try testing.expectEqual(@as(usize, 17), program.parameterLen());
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, program.parameterTensorShape(0).?);
    try testing.expectEqualSlices(usize, &.{3}, program.parameterTensorShape(1).?);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, program.parameterTensorShape(2).?);
    try testing.expectEqualSlices(usize, &.{2}, program.parameterTensorShape(3).?);
    try testing.expectEqualStrings("0.weight", program.parameterTensorName(0).?);
    try testing.expectEqualStrings("0.bias", program.parameterTensorName(1).?);
    try testing.expectEqualStrings("2.weight", program.parameterTensorName(2).?);
    try testing.expectEqualStrings("2.bias", program.parameterTensorName(3).?);
    try testing.expectEqualStrings(linear_weight_layout, program.parameterTensorLayout(0).?);
    try testing.expectEqualStrings(linear_bias_layout, program.parameterTensorLayout(1).?);
    try testing.expectEqualStrings(linear_weight_layout, program.parameterTensorLayout(2).?);
    try testing.expectEqualStrings(linear_bias_layout, program.parameterTensorLayout(3).?);
    const program_inspection = program.inspect();
    const program_command_categories = program_inspection.command_shape.categoryCounts();
    try testing.expectEqual(@as(u32, 2), program_inspection.command_shape.command_count);
    try testing.expectEqual(@as(u64, 2), program_command_categories.projection);
    try testing.expectEqual(@as(u64, 0), program_command_categories.op);
    try testing.expectEqual(@as(u64, 0), program_command_categories.elementwise);
    try testing.expectEqual(@as(u64, 0), program_command_categories.movement);
    const program_profile = program.runtimeProfile();
    try testing.expectEqual(program_inspection.command_shape, program_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 0), program_profile.call_count);
    try testing.expectEqual(@as(u64, 0), program_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), program_profile.backend_op_count);

    const input = try Tensor(f32).fromSlice(tac, &.{2}, &.{ 2, -3 });
    defer input.deinit();
    const output = try Tensor(f32).zeros(tac, &.{2});
    defer output.deinit();

    var session = try program.bindTensors(input, output);
    defer session.deinit();
    try testing.expectEqualSlices(usize, input.ne[0..input.n_dims], session.inputShape());
    try testing.expectEqualSlices(usize, output.ne[0..output.n_dims], session.outputShape());
    try testing.expectEqualStrings(program.parameterTensorName(0).?, session.parameterTensorName(0).?);
    try testing.expectEqualStrings(program.parameterTensorLayout(0).?, session.parameterTensorLayout(0).?);
    try testing.expectEqualSlices(usize, program.parameterTensorShape(0).?, session.parameterTensorShape(0).?);
    try testing.expect(session.parameterTensorName(4) == null);
    try testing.expect(session.parameterTensorLayout(4) == null);
    try testing.expect(session.parameterTensorShape(4) == null);

    const first = try session.stepTensor();
    try testing.expect(first == output);
    try testing.expect((try session.outputTensor()) == output);
    try testing.expectEqualSlices(f32, &.{ 22, 26 }, output.data);
    var session_profile = session.runtimeProfile();
    try testing.expectEqual(program_inspection.command_shape, session_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 1), session_profile.call_count);
    try testing.expectEqual(@as(u64, @intCast(program_inspection.op_count)), session_profile.backend_op_count);
    try testing.expectEqual(@as(u64, @intCast(program_inspection.command_shape.command_count)), session_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), session_profile.fallback_op_count);
    try testing.expectEqual(@as(u64, 0), session_profile.sync_count);
    try testing.expectEqual(@as(u64, 2), session_profile.program_command_counts[@intFromEnum(backend_program_mod.ProgramCommandKind.dense_projection_chain)]);
    try testing.expectEqual(@as(u64, 2), session_profile.program_command_dispatch_counts[@intFromEnum(backend_program_mod.ProgramCommandKind.dense_projection_chain)]);
    try testing.expectEqual(@as(u64, 0), session_profile.program_command_counts[@intFromEnum(backend_program_mod.ProgramCommandKind.op)]);

    const next_input = try Tensor(f32).fromSlice(tac, &.{2}, &.{ -5, 4 });
    defer next_input.deinit();
    const second = try session.executeTensor(next_input);
    try testing.expect(second == output);
    try testing.expectEqualSlices(f32, &.{ 10, 25 }, output.data);
    session_profile = session.runtimeProfile();
    try testing.expectEqual(@as(u32, 2), session_profile.call_count);

    session.resetRuntimeProfile();
    session_profile = session.runtimeProfile();
    try testing.expectEqual(program_inspection.command_shape, session_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 0), session_profile.call_count);
    try testing.expectEqual(@as(u64, 0), session_profile.backend_dispatch_count);
    try testing.expectEqual(@as(u64, 0), session_profile.backend_op_count);

    program.resetRuntimeProfile();
    const reset_program_profile = program.runtimeProfile();
    try testing.expectEqual(program_inspection.command_shape, reset_program_profile.program_command_shape);
    try testing.expectEqual(@as(u32, 0), reset_program_profile.call_count);
}

test "ModuleProgram hot path does not allocate after Session bind" {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
    const alloc = failing.allocator();

    var model = try sequential(f32, alloc, .{
        try Linear(f32).init(alloc, 2, 3, .{ .seed = 1 }),
        relu(),
        try Linear(f32).init(alloc, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0, 1, 0 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ 10, 20 });

    var program = try model.compile(alloc, .{});
    defer program.deinit();

    const input = try Tensor(f32).fromSlice(alloc, &.{2}, &.{ 2, -3 });
    defer input.deinit();
    const output = try Tensor(f32).zeros(alloc, &.{2});
    defer output.deinit();
    const next_input = try Tensor(f32).fromSlice(alloc, &.{2}, &.{ -5, 4 });
    defer next_input.deinit();

    var session = try program.bindTensors(input, output);
    defer session.deinit();

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    const first = try session.stepTensor();
    try testing.expect(first == output);
    try testing.expectEqualSlices(f32, &.{ 22, 26 }, output.data);

    _ = try session.execute(&.{ -5, 4 });
    try testing.expectEqualSlices(f32, &.{ 10, 25 }, output.data);

    const third = try session.executeTensor(next_input);
    try testing.expect(third == output);
    try testing.expectEqualSlices(f32, &.{ 10, 25 }, output.data);

    try testing.expectEqual(alloc_index, failing.alloc_index);
    try testing.expectEqual(resize_index, failing.resize_index);
    try testing.expect(!failing.has_induced_failure);
}

test "ModuleProgram tensor binding rejects non-dense views" {
    var layer = softmax(0);
    var program = try layer.compile(tac, .{ .input_shape = &.{ 2, 2 } });
    defer program.deinit();

    const base_input = try Tensor(f32).fromSlice(tac, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer base_input.deinit();
    const transposed_input = base_input.transpose();
    defer transposed_input.deinit();
    const output = try Tensor(f32).zeros(tac, &.{ 2, 2 });
    defer output.deinit();

    try testing.expect(transposed_input.hasShape(&.{ 2, 2 }));
    try testing.expect(!transposed_input.isDenseLayout());
    try testing.expectError(error.UnsupportedTensorLayout, program.bindTensors(transposed_input, output));
}

test "ModuleProgram binds compatible module state explicitly" {
    var source = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        relu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer source.deinit();

    var program = try source.compile(tac, .{});
    defer program.deinit();

    var candidate = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 3 }),
        relu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 4 }),
    });
    defer candidate.deinit();
    const candidate_params = candidate.namedParameters();
    candidate_params[0].tensor.setData(&.{ 1, 0, 0, 1, 1, 1 });
    candidate_params[1].tensor.setData(&.{ 0, 0, 0 });
    candidate_params[2].tensor.setData(&.{ 1, 2, 3, 4, 5, 6 });
    candidate_params[3].tensor.setData(&.{ 7, 8 });

    const ok_compatibility = program.moduleCompatibility(&candidate);
    try testing.expect(ok_compatibility.compatible);
    try testing.expect(ok_compatibility.canBind());
    try testing.expectEqual(ModuleCompatibilityCode.compatible, ok_compatibility.code);
    try testing.expect(ok_compatibility.reason == null);
    try program.checkModuleCompatibility(&candidate);
    try testing.expect(program.acceptsModule(&candidate));

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).fromSlice(eager_graph.allocator(), &.{2}, &.{ 1, 2 });
    const eager_output = candidate.forward(eager_input);
    try eager_graph.infer(eager_output);

    var input = [_]f32{ 1, 2 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bindModule(&candidate, input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(eager_output.data, &output, 1e-5);

    candidate_params[3].tensor.setData(&.{ -1, -2 });
    try session.uploadParameters();

    var next_graph = ComputeGraph(f32).init(tac);
    defer next_graph.deinit();
    const next_input = try Tensor(f32).fromSlice(next_graph.allocator(), &.{2}, &.{ 1, 2 });
    const next_expected = candidate.forward(next_input);
    try next_graph.infer(next_expected);
    _ = try session.execute(&.{ 1, 2 });
    try expectApproxSlices(next_expected.data, &output, 1e-5);

    var bad_shape = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 4, .{ .seed = 5 }),
        relu(),
        try Linear(f32).init(tac, 4, 2, .{ .seed = 6 }),
    });
    defer bad_shape.deinit();
    const shape_compatibility = program.moduleCompatibility(&bad_shape);
    try testing.expect(!shape_compatibility.compatible);
    try testing.expect(!shape_compatibility.canBind());
    try testing.expectEqual(ModuleCompatibilityCode.weights_len_mismatch, shape_compatibility.code);
    try testing.expectEqual(@as(?usize, 12), shape_compatibility.expected_scalar_count);
    try testing.expectEqual(@as(?usize, 16), shape_compatibility.actual_scalar_count);
    try testing.expect(!program.acceptsModule(&bad_shape));
    try testing.expectError(error.ShapeMismatch, program.checkModuleCompatibility(&bad_shape));

    var raw_layer = try Linear(f32).init(tac, 2, 3, .{ .seed = 7 });
    defer raw_layer.deinit();
    const missing_compatibility = program.moduleCompatibility(&raw_layer);
    try testing.expect(!missing_compatibility.compatible);
    try testing.expectEqual(ModuleCompatibilityCode.native_path_mismatch, missing_compatibility.code);
    try testing.expectEqualStrings("device-program", missing_compatibility.expected_native_path.?);
    try testing.expectEqualStrings("tiny-linear", missing_compatibility.actual_native_path.?);
    try testing.expect(!program.acceptsModule(&raw_layer));
    try testing.expectError(error.UnsupportedCompile, program.checkModuleCompatibility(&raw_layer));

    var bad_trace = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 8 }),
        gelu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 9 }),
    });
    defer bad_trace.deinit();

    const trace_compatibility = program.moduleCompatibility(&bad_trace);
    try testing.expect(!trace_compatibility.compatible);
    try testing.expectEqual(ModuleCompatibilityCode.ir_mismatch, trace_compatibility.code);
    try testing.expect(trace_compatibility.expected_ir_step_hash.? != 0);
    try testing.expect(trace_compatibility.actual_ir_step_hash.? != 0);
    try testing.expect(trace_compatibility.expected_ir_step_hash.? != trace_compatibility.actual_ir_step_hash.?);
    try testing.expect(!program.acceptsModule(&bad_trace));
    try testing.expectError(error.UnsupportedCompile, program.checkModuleCompatibility(&bad_trace));
}

test "ModuleProgram compiles Sequential with LayerNorm and Gelu" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        try LayerNorm(f32).init(tac, 3, .{ .eps = 1e-5 }),
        gelu(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0.5, -0.5, 0.25 });
    named[2].tensor.setData(&.{ 1, 1.5, 0.5 });
    named[3].tensor.setData(&.{ 0, 0.1, -0.2 });
    named[4].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[5].tensor.setData(&.{ 0.25, -0.75 });

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{2});
    eager_input.setData(&.{ 2, -3 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();
    var input = [_]f32{ 2, -3 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "ModuleProgram compiles Sequential with RmsNorm and Silu" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        try RmsNorm(f32).init(tac, 3, .{ .eps = 1e-5 }),
        siluActivation(),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0.5, -0.5, 0.25 });
    named[2].tensor.setData(&.{ 1.25, 0.75, 1.5 });
    named[3].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[4].tensor.setData(&.{ -0.25, 0.5 });

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{2});
    eager_input.setData(&.{ -1, 4 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();
    var input = [_]f32{ -1, 4 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "Softmax module normalizes a chosen dimension" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var layer = softmax(0);
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    const x = try Tensor(f32).init(a, &.{ 3, 2 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = layer.forward(x);
    try g.infer(y);

    for (0..2) |col| {
        var column_sum: f32 = 0;
        for (0..3) |row| column_sum += y.data[col * 3 + row];
        try testing.expectApproxEqAbs(@as(f32, 1), column_sum, 1e-6);
    }

    var program = try layer.compile(tac, .{ .input_shape = &.{ 3, 2 } });
    defer program.deinit();
    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output = [_]f32{0} ** 6;
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(y.data, &output, 1e-6);
}

test "LogSoftmax module normalizes a chosen dimension in log space" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var layer = logSoftmax(0);
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    const x = try Tensor(f32).init(a, &.{ 3, 2 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    const y = layer.forward(x);
    try g.infer(y);

    for (0..2) |col| {
        var column_sum: f32 = 0;
        for (0..3) |row| column_sum += @exp(y.data[col * 3 + row]);
        try testing.expectApproxEqAbs(@as(f32, 1), column_sum, 1e-6);
    }

    var program = try layer.compile(tac, .{ .input_shape = &.{ 3, 2 } });
    defer program.deinit();
    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output = [_]f32{0} ** 6;
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(y.data, &output, 1e-6);
}

test "Reduction modules compile through ModuleProgram" {
    const input_shape = [_]usize{ 3, 2 };
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const output_shape = [_]usize{ 1, 2 };

    try expectReductionModule(sum(0), input_shape[0..], input_values[0..], output_shape[0..], &.{ 6, 15 });
    try expectReductionModule(mean(0), input_shape[0..], input_values[0..], output_shape[0..], &.{ 2, 5 });
    try expectReductionModule(max(0), input_shape[0..], input_values[0..], output_shape[0..], &.{ 3, 6 });
    try expectReductionModule(min(0), input_shape[0..], input_values[0..], output_shape[0..], &.{ 1, 4 });
}

test "Shape modules compile through ModuleProgram" {
    const matrix_shape = [_]usize{ 2, 2 };
    const matrix_values = [_]f32{ 1, 2, 3, 4 };
    const flat_shape = [_]usize{4};

    try expectShapeModule(identity(), matrix_shape[0..], matrix_values[0..], matrix_shape[0..], matrix_values[0..], true);
    try expectShapeModule(reshape(flat_shape[0..]), matrix_shape[0..], matrix_values[0..], flat_shape[0..], matrix_values[0..], false);
    try expectShapeModule(view(flat_shape[0..]), matrix_shape[0..], matrix_values[0..], flat_shape[0..], matrix_values[0..], false);
    try expectShapeModule(flattenAll(), matrix_shape[0..], matrix_values[0..], flat_shape[0..], matrix_values[0..], false);

    const cube_shape = [_]usize{ 2, 3, 2 };
    const cube_values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const partial_flat_shape = [_]usize{ 2, 6 };
    try expectShapeModule(flatten(1, null), cube_shape[0..], cube_values[0..], partial_flat_shape[0..], cube_values[0..], false);

    const transpose_input_shape = [_]usize{ 2, 3 };
    const transpose_output_shape = [_]usize{ 3, 2 };
    const transpose_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try expectShapeModule(transpose(), transpose_input_shape[0..], transpose_input[0..], transpose_output_shape[0..], &.{ 1, 3, 5, 2, 4, 6 }, false);

    const broadcast_input_shape = [_]usize{ 1, 2 };
    const broadcast_output_shape = [_]usize{ 3, 2 };
    const broadcast_input = [_]f32{ 7, 8 };
    try expectShapeModule(broadcastTo(broadcast_output_shape[0..]), broadcast_input_shape[0..], broadcast_input[0..], broadcast_output_shape[0..], &.{ 7, 7, 7, 8, 8, 8 }, false);
    try expectShapeModule(expand(broadcast_output_shape[0..]), broadcast_input_shape[0..], broadcast_input[0..], broadcast_output_shape[0..], &.{ 7, 7, 7, 8, 8, 8 }, false);
}

test "Dropout training mode composes eager mask and gradients" {
    var layer = try Dropout(f32).init(1.0, .{ .seed = 123 });
    const support = layer.compileSupport();
    try testing.expect(!support.supported);
    try testing.expect(!layer.canCompile());
    try testing.expectEqualStrings("nn.Dropout native Program compilation only supports deterministic no-op Dropout (eval mode or p=0)", support.reason.?);

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const x = try Tensor(f32).init(g.allocator(), &.{4});
    x.setData(&.{ 1, 2, 3, 4 });
    x.setParam();

    const y = layer.forward(x);
    const loss = y.sumAll();
    try g.run(loss);

    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, y.data);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, x.grad.?.data);
}

test "Dropout eval mode compiles as identity Program" {
    var layer = try dropout(f32, 0.5, .{ .seed = 123 });
    _ = layer.eval();
    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expect(support.shape_preserving);

    var program = try layer.compile(tac, .{ .input_shape = &.{4} });
    defer program.deinit();
    const inspection = program.inspect();
    try testing.expectEqual(@as(usize, 0), inspection.op_count);
    try testing.expectEqual(@as(u32, 0), inspection.command_shape.command_count);

    var input = [_]f32{ 1, 2, 3, 4 };
    var output = [_]f32{ 0, 0, 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try testing.expectEqualSlices(f32, &input, &output);

    const profile = session.runtimeProfile();
    try testing.expectEqual(@as(u64, 0), profile.backend_op_count);
    try testing.expectEqual(@as(u64, 0), profile.backend_dispatch_count);
}

test "Sequential train eval cascades to Dropout compile support" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        try Dropout(f32).init(0.5, .{ .seed = 2 }),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 3 }),
    });
    defer model.deinit();

    const training_support = model.compileSupport();
    try testing.expect(!training_support.supported);
    try testing.expect(!model.canCompile());
    try testing.expectEqualStrings("nn.Dropout native Program compilation only supports deterministic no-op Dropout (eval mode or p=0)", training_support.reason.?);

    _ = model.eval();
    const eval_support = model.compileSupport();
    try testing.expect(eval_support.supported);
    try testing.expect(model.canCompile());

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0, 1, 0 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ 10, 20 });

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{2});
    eager_input.setData(&.{ 2, -3 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();
    var input = [_]f32{ 2, -3 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(eager_output.data, &output, 1e-5);

    _ = model.train(true);
    try testing.expect(!model.canCompile());
}

test "ModuleProgram compiles Sequential with Softmax" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        softmax(0),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const support = model.compileSupport();
    try testing.expect(support.supported);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0.5, -0.5, 0.25 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ -0.25, 0.5 });

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{2});
    eager_input.setData(&.{ 1.5, -2 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();
    var input = [_]f32{ 1.5, -2 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "ModuleProgram compiles Sequential with LogSoftmax" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        logSoftmax(0),
        try Linear(f32).init(tac, 3, 2, .{ .seed = 2 }),
    });
    defer model.deinit();

    const support = model.compileSupport();
    try testing.expect(support.supported);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    const named = model.namedParameters();
    named[0].tensor.setData(&.{ 1, 0, 0, 1, 1, -1 });
    named[1].tensor.setData(&.{ 0.5, -0.5, 0.25 });
    named[2].tensor.setData(&.{ 1, 2, 0, 1, 4, 2 });
    named[3].tensor.setData(&.{ -0.25, 0.5 });

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{2});
    eager_input.setData(&.{ 1.5, -2 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();
    var input = [_]f32{ 1.5, -2 };
    var output = [_]f32{ 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "stateDict snapshots and loadStateDict restores named module parameters" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        relu(),
        try Linear(f32).init(tac, 3, 1, .{ .seed = 2 }),
    });
    defer model.deinit();

    const named = model.namedParameters();
    try testing.expectEqual(@as(usize, 4), named.len);
    named[0].tensor.setData(&.{ 1, 2, 3, 4, 5, 6 });
    named[1].tensor.setData(&.{ 7, 8, 9 });
    named[2].tensor.setData(&.{ 10, 11, 12 });
    named[3].tensor.setData(&.{13});

    const state = try stateDict(f32, tac, named);
    defer deinitStateDict(f32, tac, state);

    try testing.expectEqualStrings("0.weight", state[0].name);
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, state[0].shape());
    try testing.expectEqualStrings(linear_weight_layout, state[0].layout.?);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, state[0].data);
    try testing.expectEqualStrings("2.bias", state[3].name);
    try testing.expectEqualStrings(linear_bias_layout, state[3].layout.?);
    try testing.expectEqualSlices(f32, &.{13}, state[3].data);

    for (named) |param| _ = param.tensor.setAllScalar(-1);

    try loadStateDict(f32, named, state);
    for (named, state) |param, entry| {
        try testing.expectEqualStrings(param.name, entry.name);
        try testing.expectEqualSlices(f32, entry.data, param.tensor.denseSliceConst());
    }
}

test "loadStateDict rejects missing and shape-mismatched parameters" {
    var src = try Linear(f32).init(tac, 2, 3, .{ .seed = 1 });
    defer src.deinit();
    const state = try stateDict(f32, tac, src.namedParameters());
    defer deinitStateDict(f32, tac, state);

    try testing.expectError(error.MissingParameter, loadStateDict(f32, src.namedParameters(), state[0..1]));

    var dst = try Linear(f32).init(tac, 2, 2, .{ .seed = 2 });
    defer dst.deinit();
    try testing.expectError(error.ShapeMismatch, loadStateDict(f32, dst.namedParameters(), state));

    var layout_src = try Linear(f32).init(tac, 2, 3, .{ .bias = false, .seed = 3 });
    defer layout_src.deinit();
    const layout_state = try stateDict(f32, tac, layout_src.namedParameters());
    defer deinitStateDict(f32, tac, layout_state);
    var wrong_layout_entries = [_]StateEntry(f32){layout_state[0]};
    wrong_layout_entries[0].layout = "row-major:linear.weight[in_features,out_features]";
    try testing.expectError(error.LayoutMismatch, loadStateDict(f32, layout_src.namedParameters(), wrong_layout_entries[0..]));
}

test "loadStateDict does not partially mutate parameters on error" {
    var layer = try Linear(f32).init(tac, 2, 1, .{ .seed = 1 });
    defer layer.deinit();
    const named = layer.namedParameters();
    named[0].tensor.setData(&.{ 1, 2 });
    named[1].tensor.setData(&.{3});

    const state = try stateDict(f32, tac, named);
    defer deinitStateDict(f32, tac, state);

    named[0].tensor.setData(&.{ 5, 6 });
    named[1].tensor.setData(&.{7});

    var unexpected_data = [_]f32{0};
    var unexpected_ne = [_]usize{1} ** max_dims;
    unexpected_ne[0] = 1;
    const bad_entries = [_]StateEntry(f32){
        state[0],
        .{
            .name = "unexpected.weight",
            .n_dims = 1,
            .ne = unexpected_ne,
            .data = unexpected_data[0..],
        },
    };

    try testing.expectError(error.UnexpectedParameter, loadStateDict(f32, named, bad_entries[0..]));
    try testing.expectEqualSlices(f32, &.{ 5, 6 }, named[0].tensor.denseSliceConst());
    try testing.expectEqualSlices(f32, &.{7}, named[1].tensor.denseSliceConst());
}

test "Sequential collects normalization module parameters" {
    var model = try Sequential(f32).init(tac, .{
        try Linear(f32).init(tac, 2, 3, .{ .seed = 1 }),
        try LayerNorm(f32).init(tac, 3, .{}),
        relu(),
        try RmsNorm(f32).init(tac, 3, .{}),
        try Linear(f32).init(tac, 3, 1, .{ .seed = 2 }),
    });
    defer model.deinit();

    const params = model.parameters();
    try testing.expectEqual(@as(usize, 7), params.len);
    for (params) |param| {
        try testing.expect(param.grad != null);
    }
}

test "parameters skips parameterless activation modules" {
    var first = try Linear(f32).init(tac, 2, 3, .{ .seed = 1 });
    defer first.deinit();
    var second = try Linear(f32).init(tac, 3, 1, .{ .seed = 2 });
    defer second.deinit();

    const params = try parameters(f32, tac, .{ &first, gelu(), relu(), siluActivation(), &second });
    defer tac.free(params);

    try testing.expectEqual(@as(usize, 4), params.len);
    try testing.expect(params[0] == first.weight);
    try testing.expect(params[1] == first.bias.?);
    try testing.expect(params[2] == second.weight);
    try testing.expect(params[3] == second.bias.?);
}

test "sequential convenience constructor owns value layers" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 1, .{ .bias = false, .seed = 3 }),
        relu(),
    });
    defer model.deinit();

    const params = model.parameters();
    try testing.expectEqual(@as(usize, 1), params.len);
}

test "Sequential parameters can outlive a training graph" {
    var model = try sequential(f32, tac, .{
        try Linear(f32).init(tac, 2, 1, .{ .seed = 5 }),
    });
    defer model.deinit();

    const params = model.parameters();
    const weight_grad_buf = params[0].paramGradOrNull().?;
    const bias_grad_buf = params[1].paramGradOrNull().?;

    {
        var g = ComputeGraph(f32).init(tac);
        defer g.deinit();
        const a = g.allocator();

        const x = try Tensor(f32).init(a, &.{ 2, 2 });
        x.setData(&.{ 1, 2, 3, 4 });
        const loss = model.forward(x).sumAll();
        try g.run(loss);

        try testing.expect(params[0].grad != null);
        try testing.expect(params[0].grad.? != weight_grad_buf);
        try testing.expect(params[1].grad != null);
    }

    try testing.expect(params[0].grad.? == weight_grad_buf);
    try testing.expect(params[1].grad.? == bias_grad_buf);

    var first_grad_nonzero = false;
    for (weight_grad_buf.data) |v| {
        if (v != 0) first_grad_nonzero = true;
    }
    try testing.expect(first_grad_nonzero);

    {
        var g = ComputeGraph(f32).init(tac);
        defer g.deinit();
        const a = g.allocator();

        const x = try Tensor(f32).init(a, &.{ 2, 1 });
        x.setData(&.{ 2, 1 });
        const loss = model.forward(x).sumAll();
        try g.run(loss);
    }

    try testing.expect(params[0].grad.? == weight_grad_buf);
    try testing.expect(params[1].grad.? == bias_grad_buf);
}

test "Embedding layer owns table and gathers rows" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var layer = try Embedding(f32).init(a, 4, 3, .{ .seed = 11 });

    layer.weight.setData(&.{
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
    });
    const indices = try Tensor(f32).init(a, &.{3});
    indices.setData(&.{ 2, 0, 3 });

    const out = layer.forward(indices);
    out.compute();

    try testing.expectEqualSlices(f32, &.{
        30, 31, 32,
        10, 11, 12,
        40, 41, 42,
    }, out.data);
    const params = layer.parameters();
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expect(params[0] == layer.weight);
    const named = layer.namedParameters();
    try testing.expectEqual(@as(usize, 1), named.len);
    try testing.expectEqualStrings("weight", named[0].name);
    try testing.expect(named[0].tensor == layer.weight);

    const support = layer.compileSupport();
    try testing.expect(support.supported);
    try testing.expect(layer.canCompile());
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);
}

test "Embedding compiles and executes through native Program Session" {
    var layer = try Embedding(f32).init(tac, 4, 3, .{ .seed = 11 });
    defer layer.deinit();
    layer.weight.setData(&.{
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
    });

    var program = try layer.compile(tac, .{ .input_shape = &.{3} });
    defer program.deinit();
    try testing.expectEqual(@as(usize, 3), program.inputLen());
    try testing.expectEqual(@as(usize, 9), program.outputLen());

    var input = [_]f32{ 2, 0, 3 };
    var output = [_]f32{0} ** 9;
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try testing.expectEqualSlices(f32, &.{
        30, 31, 32,
        10, 11, 12,
        40, 41, 42,
    }, &output);
}

test "ModuleProgram compiles Sequential with Embedding and Softmax" {
    var model = try sequential(f32, tac, .{
        try Embedding(f32).init(tac, 4, 3, .{ .seed = 11 }),
        softmax(0),
    });
    defer model.deinit();
    const named = model.namedParameters();
    named[0].tensor.setData(&.{
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
    });

    const support = model.compileSupport();
    try testing.expect(support.supported);
    try testing.expectEqualStrings("device-program", support.native_path.?);
    try testing.expectEqualStrings("module", support.model_kind.?);

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{3});
    eager_input.setData(&.{ 2, 0, 3 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{ .input_shape = &.{3} });
    defer program.deinit();
    var input = [_]f32{ 2, 0, 3 };
    var output = [_]f32{0} ** 9;
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();

    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "Embedding backward accumulates repeated indices" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var layer = try Embedding(f32).init(a, 4, 2, .{});
    layer.weight.setData(&.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    const indices = try Tensor(f32).init(a, &.{3});
    indices.setData(&.{ 1, 3, 1 });

    const loss = layer.forward(indices).sumAll();
    try g.run(loss);

    try testing.expectEqualSlices(f32, &.{
        0, 0,
        2, 2,
        0, 0,
        1, 1,
    }, layer.weight.grad.?.data);
}

test "classTargets builds typed index tensor for crossEntropy" {
    const targets = try classTargets(f32, tac, &.{ 2, 0, 1 });
    defer targets.deinit();

    try testing.expect(targets.hasIndexBuffer());
    try testing.expectEqualSlices(usize, &.{ 2, 0, 1 }, targets.indexData().?);
    try testing.expectEqual(@as(usize, 3), targets.ne[0]);
}

test "parameters collects heterogeneous layer params" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var embed = try Embedding(f32).init(a, 4, 2, .{});
    var head = try Linear(f32).init(a, 2, 3, .{});
    const params = try parameters(f32, tac, .{ &embed, &head });
    defer tac.free(params);

    try testing.expectEqual(@as(usize, 3), params.len);
    try testing.expect(params[0] == embed.weight);
    try testing.expect(params[1] == head.weight);
    try testing.expect(params[2] == head.bias.?);
}

test "meanSquaredError forward and backward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const pred = try Tensor(f32).init(a, &.{ 1, 2 });
    pred.setData(&.{ 2, -1 });
    pred.setParam();
    const target = try Tensor(f32).init(a, &.{ 1, 2 });
    target.setData(&.{ 1, 1 });

    const loss = meanSquaredError(f32, pred, target);
    try g.run(loss);

    try testing.expectApproxEqAbs(@as(f32, 2.5), loss.data[0], 1e-6);
    try testing.expectEqualSlices(f32, &.{ 1, -2 }, pred.grad.?.data);
}

test "crossEntropy forward and backward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });
    logits.setParam();

    const targets = try classTargets(f32, a, &.{ 0, 1 });

    const ce = crossEntropy(f32, logits, targets);
    try g.run(ce);

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, ce.data[0], 1e-5);

    for (logits.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "silu composes differentiable primitives" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ -1, 0, 2 });
    x.setParam();
    const y = silu(f32, x);
    const loss = y.sumAll();
    try g.run(loss);

    for (x.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "Silu module composes inside Sequential" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var model = try Sequential(f32).init(tac, .{siluActivation()});
    defer model.deinit();
    try testing.expectEqual(@as(usize, 0), model.parameters().len);

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ -1, 0, 2 });
    const y = model.forward(x);
    try g.infer(y);

    try testing.expectApproxEqAbs(@as(f32, -0.26894143), y.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), y.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.7615942), y.data[2], 1e-6);
}

test "sigmoid composes differentiable primitives and compiles as module Program" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ -1, 0, 2 });
    x.setParam();
    const y = sigmoid(f32, x);
    const loss = y.sumAll();
    try g.run(loss);

    try testing.expectApproxEqAbs(@as(f32, 0.26894143), y.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), y.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8807971), y.data[2], 1e-6);
    for (x.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }

    var standalone = sigmoidActivation();
    const standalone_support = standalone.compileSupport();
    try testing.expect(standalone_support.supported);
    try testing.expect(standalone.canCompile());
    var standalone_program = try standalone.compile(tac, .{ .input_shape = &.{3} });
    defer standalone_program.deinit();
    var standalone_input = [_]f32{ -1, 0, 2 };
    var standalone_output = [_]f32{ 0, 0, 0 };
    var standalone_session = try standalone_program.bind(standalone_input[0..], standalone_output[0..]);
    defer standalone_session.deinit();
    _ = try standalone_session.step();
    try testing.expectApproxEqAbs(@as(f32, 0.26894143), standalone_output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), standalone_output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8807971), standalone_output[2], 1e-6);

    var model = try Sequential(f32).init(tac, .{
        try Linear(f32).init(tac, 3, 3, .{ .seed = 1, .bias = false }),
        sigmoidActivation(),
        try Linear(f32).init(tac, 3, 3, .{ .seed = 2, .bias = false }),
    });
    defer model.deinit();
    const support = model.compileSupport();
    try testing.expect(support.supported);
    try testing.expectEqualStrings("device-program", support.native_path.?);

    var eager_graph = ComputeGraph(f32).init(tac);
    defer eager_graph.deinit();
    const eager_input = try Tensor(f32).init(eager_graph.allocator(), &.{3});
    eager_input.setData(&.{ -1, 0, 2 });
    const eager_output = model.forward(eager_input);
    try eager_graph.infer(eager_output);

    var program = try model.compile(tac, .{});
    defer program.deinit();

    var input = [_]f32{ -1, 0, 2 };
    var output = [_]f32{ 0, 0, 0 };
    var session = try program.bind(input[0..], output[0..]);
    defer session.deinit();
    _ = try session.step();
    try expectApproxSlices(eager_output.data, &output, 1e-5);
}

test "unary activation modules compile through ModuleProgram" {
    try expectUnaryActivationModule(tanhActivation(), &.{ 0, 1 }, &.{ 0, 0.7615942 });
    try expectUnaryActivationModule(exp(), &.{ 0, 1 }, &.{ 1, 2.7182817 });
    try expectUnaryActivationModule(log(), &.{ 1, 2.7182817 }, &.{ 0, 1 });
    try expectUnaryActivationModule(neg(), &.{ -2, 3 }, &.{ 2, -3 });
    try expectUnaryActivationModule(recip(), &.{ 2, -4 }, &.{ 0.5, -0.25 });
    try expectUnaryActivationModule(abs(), &.{ -2, 0, 3 }, &.{ 2, 0, 3 });
    try expectUnaryActivationModule(sqrt(), &.{ 4, 9 }, &.{ 2, 3 });
    try expectUnaryActivationModule(square(), &.{ 2, -3 }, &.{ 4, 9 });
    try expectUnaryActivationModule(sqr(), &.{ 2, -3 }, &.{ 4, 9 });
    try expectUnaryActivationModule(sgn(), &.{ -2, 0, 3 }, &.{ -1, 0, 1 });
    try expectUnaryActivationModule(sign(), &.{ -2, 0, 3 }, &.{ -1, 0, 1 });
    try expectUnaryActivationModule(stepActivation(), &.{ -2, 0, 3 }, &.{ 0, 0, 1 });
}

test "argmax selects row index per column" {
    const logits = try Tensor(f32).init(tac, &.{ 3, 2 });
    defer logits.deinit();
    logits.setData(&.{
        1, 4, 2,
        5, 3, 0,
    });
    var preds: [2]usize = undefined;
    argmax(f32, logits, &preds);
    try testing.expectEqualSlices(usize, &.{ 1, 0 }, &preds);
}

test "kaimingUniform - values within expected bounds" {
    const t = try Tensor(f32).init(tac, &.{ 16, 4 });
    defer t.deinit();

    kaimingUniform(f32, t, 42);

    const bound: f32 = @sqrt(6.0 / 4.0);
    for (t.data) |v| {
        try testing.expect(v >= -bound and v <= bound);
    }
}

test "uniform - values within range" {
    const t = try Tensor(f32).init(tac, &.{100});
    defer t.deinit();

    uniform(f32, t, -0.5, 0.5, 123);

    for (t.data) |v| {
        try testing.expect(v >= -0.5 and v < 0.5);
    }
}
