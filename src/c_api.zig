const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("backend.zig");
const cpu_mod = @import("backend/cpu.zig");
const device_inference_mod = @import("device_inference.zig");
const gguf_mod = @import("gguf.zig");
const gguf_loader = @import("models/gguf_loader.zig");
const graph_mod = @import("graph.zig");
const llm_mod = @import("llm.zig");
const sampling = @import("llm_sampling.zig");
const nn = @import("nn.zig");
const program_mod = @import("backend/program.zig");
const profile_mod = @import("profile.zig");
const safetensors_mod = @import("safetensors.zig");
const stencil_mod = @import("backend/stencil.zig");
const tensor_mod = @import("tensor.zig");
const forward = @import("tensor/forward.zig");
const build_options = @import("zgml_options");
const wgpu_mod = if (build_options.use_wgpu) @import("backend/wgpu.zig") else struct {};

pub const zgml_model = opaque {};
pub const zgml_program = opaque {};
pub const zgml_session = opaque {};
pub const zgml_buffer = opaque {};

const Status = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    out_of_memory = 2,
    shape_mismatch = 3,
    compile_failed = 4,
    unsupported = 5,
};

const auto_kind: u32 = 0;
const tiny_linear_kind: u32 = 1;
const tiny_llama_kind: u32 = 2;
const smollm_135m_kind: u32 = 3;
const tiny_llama_2layer_kind: u32 = 4;
const tiny_mlp_kind: u32 = 5;
const module_kind: u32 = 6;
const abi_version: u32 = 6;
const feature_buffer_handle: u64 = 1 << 0;
const feature_model_auto: u64 = 1 << 1;
const feature_runtime_profile: u64 = 1 << 2;
const feature_webgpu_compile_only: u64 = 1 << 3;
const feature_wasm_exports: u64 = 1 << 4;
const feature_native_buffer_io: u64 = 1 << 5;
const feature_native_argmax: u64 = 1 << 6;
const feature_native_execute_argmax: u64 = 1 << 7;
const feature_native_topk_sample: u64 = 1 << 8;
const feature_program_requirements: u64 = 1 << 9;
const feature_program_output_buffer: u64 = 1 << 10;
const feature_native_generate_sample: u64 = 1 << 11;
const feature_native_generate_argmax: u64 = 1 << 12;
const feature_session_model_binding: u64 = 1 << 13;
const feature_external_buffer: u64 = 1 << 14;
const feature_external_resource_buffer: u64 = 1 << 15;
const feature_llama_kv_resource_binding: u64 = 1 << 16;
const feature_llama_kv_cache_requirements: u64 = 1 << 17;
const feature_program_buffer_factory: u64 = 1 << 18;
const feature_external_resource_access: u64 = 1 << 19;
const feature_model_inspection: u64 = 1 << 20;
const feature_program_model_compatibility: u64 = 1 << 21;
const feature_buffer_inspection: u64 = 1 << 22;
const feature_session_inspection: u64 = 1 << 23;
const feature_program_resource_inspection: u64 = 1 << 24;
const feature_program_memory_inspection: u64 = 1 << 25;
const feature_program_shape_inspection: u64 = 1 << 26;
const feature_program_patch_envelope_inspection: u64 = 1 << 27;
const feature_session_binding_shape_inspection: u64 = 1 << 28;
const feature_native_wgpu_execution: u64 = 1 << 29;
const feature_program_device_buffer: u64 = 1 << 30;
const feature_program_device_buffer_import: u64 = 1 << 31;
const feature_program_dispatch_plan_inspection: u64 = 1 << 32;
const feature_abi_struct_size: u64 = 1 << 33;
const feature_experimental_llama_wgpu_execution: u64 = 1 << 34;
const feature_model_path_probe: u64 = 1 << 35;
const feature_supported_checkpoints: u64 = 1 << 36;
const feature_safetensors_header_probe: u64 = 1 << 37;
const feature_safetensors_data_load: u64 = 1 << 38;
const feature_safetensors_data_probe: u64 = 1 << 39;
const feature_native_tiny_mlp: u64 = 1 << 40;
const feature_native_module_program: u64 = 1 << 41;
const feature_program_binding_requirements: u64 = 1 << 42;
const feature_session_persistent_upload: u64 = 1 << 43;
const feature_native_module_activation_chain: u64 = 1 << 44;
const feature_native_eager_linear: u64 = 1 << 45;
const feature_native_eager_linear_activation: u64 = 1 << 46;
const feature_native_training_step: u64 = 1 << 47;
const feature_native_eager_softmax: u64 = 1 << 48;
const feature_native_eager_matmul: u64 = 1 << 49;
const feature_native_eager_activation: u64 = 1 << 50;
const feature_native_eager_elementwise: u64 = 1 << 51;
const feature_native_eager_reduce: u64 = 1 << 52;
const feature_native_eager_conv2d: u64 = 1 << 53;
const feature_native_eager_pool2d: u64 = 1 << 54;
const feature_native_eager_dot: u64 = 1 << 55;
const backend_auto: u32 = 0;
const backend_cpu: u32 = 1;
const backend_metal: u32 = 2;
const backend_webgpu: u32 = 3;
const buffer_storage_none: u32 = 0;
const buffer_storage_host: u32 = 1;
const buffer_storage_external_resource: u32 = 2;
const execute_output_none: u32 = 0;
const execute_output_logits: u32 = 1;
const resource_access_read: u32 = 1 << 0;
const resource_access_write: u32 = 1 << 1;
const resource_access_read_write: u32 = resource_access_read | resource_access_write;
const program_buffer_weights: u32 = 1;
const program_buffer_bias: u32 = 2;
const program_buffer_input: u32 = 3;
const program_buffer_output: u32 = 4;
const program_buffer_llama_k_cache: u32 = 5;
const program_buffer_llama_v_cache: u32 = 6;
const abi_struct_runtime_info: u32 = 1;
const abi_struct_model_desc: u32 = 2;
const abi_struct_model_load_desc: u32 = 3;
const abi_struct_model_inspection: u32 = 4;
const abi_struct_program_model_compatibility: u32 = 5;
const abi_struct_session_inspection: u32 = 6;
const abi_struct_compile_desc: u32 = 7;
const abi_struct_bind_desc: u32 = 8;
const abi_struct_buffer_desc: u32 = 9;
const abi_struct_buffer_inspection: u32 = 10;
const abi_struct_external_resource_desc: u32 = 11;
const abi_struct_device_buffer_import_desc: u32 = 12;
const abi_struct_buffer_bind_desc: u32 = 13;
const abi_struct_llama_kv_cache_bind_desc: u32 = 14;
const abi_struct_llama_buffer_bind_desc: u32 = 15;
const abi_struct_step_desc: u32 = 16;
const abi_struct_step_result: u32 = 17;
const abi_struct_token_step_desc: u32 = 18;
const abi_struct_token_advance_desc: u32 = 19;
const abi_struct_token_advance_tokens_desc: u32 = 20;
const abi_struct_token_prefill_desc: u32 = 21;
const abi_struct_token_execute_desc: u32 = 22;
const abi_struct_token_argmax_desc: u32 = 23;
const abi_struct_token_argmax_result: u32 = 24;
const abi_struct_token_execute_argmax_desc: u32 = 25;
const abi_struct_token_generate_argmax_desc: u32 = 26;
const abi_struct_token_generate_argmax_result: u32 = 27;
const abi_struct_token_sample_desc: u32 = 28;
const abi_struct_token_sample_result: u32 = 29;
const abi_struct_token_execute_sample_desc: u32 = 30;
const abi_struct_token_generate_sample_desc: u32 = 31;
const abi_struct_token_generate_sample_result: u32 = 32;
const abi_struct_program_requirements: u32 = 33;
const abi_struct_llama_kv_cache_requirements: u32 = 34;
const abi_struct_program_inspection: u32 = 35;
const abi_struct_llama_program_inspection: u32 = 36;
const abi_struct_runtime_profile: u32 = 37;
const abi_struct_safetensors_header_probe_desc: u32 = 38;
const abi_struct_safetensors_data_load_desc: u32 = 39;
const abi_struct_module_op_desc: u32 = 40;
const abi_struct_module_desc: u32 = 41;
const module_op_linear: u32 = 1;
const module_op_activation: u32 = 2;
const module_op_softmax: u32 = 3;
const module_op_layer_norm: u32 = 4;
const module_op_rms_norm: u32 = 5;
const module_op_embedding: u32 = 6;
const module_op_log_softmax: u32 = 7;
const module_op_reshape: u32 = 8;
const module_op_broadcast_to: u32 = 9;
const module_op_narrow: u32 = 10;
const module_op_transpose: u32 = 11;
const module_op_reduce_sum: u32 = 12;
const module_op_reduce_mean: u32 = 13;
const module_op_reduce_max: u32 = 14;
const module_op_reduce_min: u32 = 21;
const module_op_feature_affine: u32 = 22;
const module_op_diagonal: u32 = 23;
const module_op_reduce_argmax: u32 = 24;
const module_op_reduce_argmin: u32 = 25;
const module_op_reduce_prod: u32 = 26;
const module_op_mul: u32 = 27;
const module_op_slice: u32 = 15;
const module_op_activation_chain: u32 = 16;
const module_op_max_pool2d: u32 = 17;
const module_op_avg_pool2d: u32 = 18;
const module_op_conv2d: u32 = 19;
const module_op_add: u32 = 20;
const module_activation_relu: u32 = 1;
const module_activation_gelu: u32 = 2;
const module_activation_silu: u32 = 3;
const module_activation_sigmoid: u32 = 4;
const module_activation_exp: u32 = 5;
const module_activation_log: u32 = 6;
const module_activation_neg: u32 = 7;
const module_activation_recip: u32 = 8;
const module_activation_abs: u32 = 9;
const module_activation_sqrt: u32 = 10;
const module_activation_square: u32 = 11;
const module_activation_sgn: u32 = 12;
const module_activation_step: u32 = 13;
const module_activation_tanh: u32 = 14;
const module_flag_weight: u32 = 1 << 0;
const module_flag_bias: u32 = 1 << 1;
const TensorF32 = tensor_mod.Tensor(f32);
const GraphF32 = graph_mod.ComputeGraph(f32);
const DeviceF32 = device_inference_mod.DeviceInference(f32);
const tiny_llama_config = llm_mod.LlamaConfig{
    .vocab_size = 8,
    .d_model = 4,
    .n_heads = 1,
    .n_kv_heads = 1,
    .d_ff = 8,
    .n_layers = 1,
    .max_seq_len = 8,
};
const tiny_llama_2layer_config = llm_mod.LlamaConfig{
    .vocab_size = 8,
    .d_model = 4,
    .n_heads = 1,
    .n_kv_heads = 1,
    .d_ff = 8,
    .n_layers = 2,
    .max_seq_len = 32,
};
const smollm_135m_config = llm_mod.LlamaConfig{
    .vocab_size = 49152,
    .d_model = 576,
    .n_heads = 9,
    .n_kv_heads = 3,
    .d_ff = 1536,
    .n_layers = 30,
    .max_seq_len = 2048,
    .rope_base = 10000.0,
    .rms_norm_eps = 1e-5,
    .tied_lm_head = true,
};
const TinyLlamaModel = llm_mod.LlamaModel(f32, tiny_llama_config);
const TinyLlamaProgram = llm_mod.LlamaProgram(f32, tiny_llama_config);
const TinyLlamaSession = llm_mod.LlamaSession(f32, tiny_llama_config);
const TinyLlama2LayerModel = llm_mod.LlamaModel(f32, tiny_llama_2layer_config);
const TinyLlama2LayerProgram = llm_mod.LlamaProgram(f32, tiny_llama_2layer_config);
const TinyLlama2LayerSession = llm_mod.LlamaSession(f32, tiny_llama_2layer_config);
const SmolLM135MModel = llm_mod.LlamaModel(f32, smollm_135m_config);
const SmolLM135MProgram = llm_mod.LlamaProgram(f32, smollm_135m_config);
const SmolLM135MSession = llm_mod.LlamaSession(f32, smollm_135m_config);

const LlamaFamilySpec = struct {
    kind: u32,
    config: llm_mod.LlamaConfig,
};

const compatible_llama_families = [_]LlamaFamilySpec{
    .{ .kind = tiny_llama_kind, .config = tiny_llama_config },
    .{ .kind = tiny_llama_2layer_kind, .config = tiny_llama_2layer_config },
    .{ .kind = smollm_135m_kind, .config = smollm_135m_config },
};

pub const zgml_runtime_info = extern struct {
    abi_version: u32 = 0,
    size_t_bytes: u32 = 0,
    pointer_bytes: u32 = 0,
    token_id_bytes: u32 = 0,
    feature_flags: u64 = 0,
};

pub const zgml_model_desc = extern struct {
    kind: u32,
    activation: u32 = 0,
    input_len: usize,
    output_len: usize,
    hidden_len: usize = 0,
};

pub const zgml_model_load_desc = extern struct {
    kind: u32,
    path: ?[*]const u8,
    path_len: usize,
};

pub const zgml_safetensors_header_probe_desc = extern struct {
    kind: u32 = auto_kind,
    reserved: u32 = 0,
    header: ?[*]const u8,
    header_len: usize,
};

pub const zgml_safetensors_data_load_desc = extern struct {
    kind: u32 = auto_kind,
    reserved: u32 = 0,
    data: ?[*]const u8,
    data_len: usize,
};

pub const zgml_module_op_desc = extern struct {
    kind: u32,
    activation: u32 = 0,
    flags: u32 = 0,
    reserved: u32 = 0,
    a: usize = 0,
    b: usize = 0,
    c: usize = 0,
    eps: f64 = 0,
};

pub const zgml_module_desc = extern struct {
    input_shape: ?[*]const usize,
    input_rank: usize,
    ops: ?[*]const zgml_module_op_desc,
    op_count: usize,
};

pub const zgml_model_inspection = extern struct {
    model_kind: u32 = 0,
    reserved: u32 = 0,
    input_len: u64 = 0,
    output_len: u64 = 0,
    vocab_size: u64 = 0,
    max_seq_len: u64 = 0,
    d_model: u64 = 0,
    n_layers: u64 = 0,
    n_heads: u64 = 0,
    n_kv_heads: u64 = 0,
    d_ff: u64 = 0,
    rope_base: f64 = 0,
    rms_norm_eps: f64 = 0,
    tied_lm_head: u64 = 0,
};

pub const zgml_program_model_compatibility = extern struct {
    program_model_kind: u32 = 0,
    model_kind: u32 = 0,
    compatible: u64 = 0,
};

pub const zgml_session_inspection = extern struct {
    model_kind: u32 = 0,
    backend: u32 = 0,
    output_storage: u32 = 0,
    kv_cache_storage: u32 = 0,
    position: u64 = 0,
    context_len: u64 = 0,
    persistent_binding_count: u64 = 0,
    step_input_count: u64 = 0,
    step_output_count: u64 = 0,
    host_binding_count: u64 = 0,
    resource_binding_count: u64 = 0,
    binding_shape_hash: u64 = 0,
};

pub const zgml_compile_desc = extern struct {
    backend: u32 = backend_auto,
    reserved: u32 = 0,
    context_len: usize = 0,
    batch: usize = 0,
};

pub const zgml_bind_desc = extern struct {
    weights: ?[*]const f32,
    weights_len: usize,
    bias: ?[*]const f32 = null,
    bias_len: usize = 0,
    input: ?[*]const f32 = null,
    input_len: usize = 0,
    output: ?[*]f32 = null,
    output_len: usize = 0,
};

pub const zgml_buffer_desc = extern struct {
    byte_len: usize,
};

pub const zgml_buffer_inspection = extern struct {
    storage: u32 = 0,
    placement: u32 = 0,
    access_flags: u32 = 0,
    reserved: u32 = 0,
    byte_len: u64 = 0,
    handle: u64 = 0,
    byte_offset: u64 = 0,
    resource_byte_len: u64 = 0,
};

pub const zgml_external_resource_desc = extern struct {
    placement: u32,
    access_flags: u32 = 0,
    handle: usize,
    byte_offset: usize = 0,
    byte_len: usize,
};

pub const zgml_device_buffer_import_desc = extern struct {
    placement: u32,
    reserved: u32 = 0,
    device_handle: usize,
    buffer_handle: usize,
    byte_offset: usize = 0,
    byte_len: usize = 0,
};

pub const zgml_buffer_bind_desc = extern struct {
    weights: ?*zgml_buffer = null,
    weights_len: usize = 0,
    bias: ?*zgml_buffer = null,
    bias_len: usize = 0,
    input: ?*zgml_buffer = null,
    input_len: usize = 0,
    output: ?*zgml_buffer = null,
    output_len: usize = 0,
};

pub const zgml_llama_kv_cache_bind_desc = extern struct {
    k: ?[*]?*zgml_buffer = null,
    v: ?[*]?*zgml_buffer = null,
    len: usize = 0,
};

pub const zgml_llama_buffer_bind_desc = extern struct {
    output: ?*zgml_buffer = null,
    output_len: usize = 0,
    kv_cache: ?*const zgml_llama_kv_cache_bind_desc = null,
};

pub const zgml_step_desc = extern struct {
    input: ?[*]const f32,
    input_len: usize,
    output: ?[*]f32,
    output_len: usize,
};

pub const zgml_step_result = extern struct {
    output_len: usize = 0,
};

pub const zgml_token_step_desc = extern struct {
    token: u32,
    output: ?[*]f32,
    output_len: usize,
};

pub const zgml_token_advance_desc = extern struct {
    token: u32,
};

pub const zgml_token_advance_tokens_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
};

pub const zgml_token_prefill_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    output: ?[*]f32,
    output_len: usize,
};

pub const zgml_token_execute_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    output_policy: u32,
    reserved: u32 = 0,
    output: ?[*]f32,
    output_len: usize,
};

pub const zgml_token_argmax_desc = extern struct {
    logits: ?[*]const f32 = null,
    logits_len: usize = 0,
    reserved: u32 = 0,
};

pub const zgml_token_argmax_result = extern struct {
    token: u32 = 0,
    logit: f32 = 0,
};

pub const zgml_token_execute_argmax_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    reserved: u32 = 0,
};

pub const zgml_token_generate_argmax_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    output_tokens: ?[*]u32,
    output_tokens_len: usize,
    reserved: u32 = 0,
};

pub const zgml_token_generate_argmax_result = extern struct {
    tokens_generated: usize = 0,
    last_token: u32 = 0,
    last_logit: f32 = 0,
};

pub const zgml_token_sample_desc = extern struct {
    logits: ?[*]const f32 = null,
    logits_len: usize = 0,
    top_k: u32 = 0,
    seed: u32 = 0,
    temperature: f32 = 1.0,
    reserved: u32 = 0,
};

pub const zgml_token_sample_result = extern struct {
    token: u32 = 0,
    logit: f32 = 0,
};

pub const zgml_token_execute_sample_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    top_k: u32 = 0,
    seed: u32 = 0,
    temperature: f32 = 1.0,
    reserved: u32 = 0,
};

pub const zgml_token_generate_sample_desc = extern struct {
    tokens: ?[*]const u32,
    tokens_len: usize,
    output_tokens: ?[*]u32,
    output_tokens_len: usize,
    top_k: u32 = 0,
    seed: u32 = 0,
    temperature: f32 = 1.0,
    reserved: u32 = 0,
};

pub const zgml_token_generate_sample_result = extern struct {
    tokens_generated: usize = 0,
    last_token: u32 = 0,
    last_logit: f32 = 0,
};

pub const zgml_program_requirements = extern struct {
    model_kind: u32 = 0,
    scalar_bytes: u32 = 0,
    token_id_bytes: u32 = 0,
    reserved: u32 = 0,
    input_len: usize = 0,
    input_byte_len: usize = 0,
    output_len: usize = 0,
    weights_len: usize = 0,
    weights_byte_len: usize = 0,
    bias_len: usize = 0,
    bias_byte_len: usize = 0,
    parameter_len: usize = 0,
    parameter_byte_len: usize = 0,
    logits_len: usize = 0,
    output_byte_len: usize = 0,
    context_len: usize = 0,
    batch: usize = 0,
    max_token_window: usize = 0,
};

pub const zgml_llama_kv_cache_requirements = extern struct {
    model_kind: u32 = 0,
    scalar_bytes: u32 = 0,
    n_layers: u32 = 0,
    reserved: u32 = 0,
    context_len: usize = 0,
    k_buffer_byte_len: usize = 0,
    v_buffer_byte_len: usize = 0,
    buffer_byte_len: usize = 0,
};

pub const zgml_program_inspection = extern struct {
    backend: u64 = 0,
    execution_supported: u64 = 0,
    external_resources_supported: u64 = 0,
    buffer_count: u64 = 0,
    buffer_byte_len: u64 = 0,
    initial_upload_count: u64 = 0,
    qweight_count: u64 = 0,
    command_count: u64 = 0,
    command_stencil_hash: u64 = 0,
    runtime_patch_holes: u64 = 0,
    runtime_patch_cache_write_pos_holes: u64 = 0,
    runtime_patch_attention_seq_kv_holes: u64 = 0,
    runtime_patch_stencil_hash: u64 = 0,
    command_op_count: u64 = 0,
    command_row_count: u64 = 0,
    command_projection_count: u64 = 0,
    command_attention_count: u64 = 0,
    command_movement_count: u64 = 0,
    command_elementwise_count: u64 = 0,
    command_rope_count: u64 = 0,
    op_count: u64 = 0,
    buffer_element_count: u64 = 0,
    runtime_patch_max_cache_write_pos: u64 = 0,
    runtime_patch_max_attention_seq_kv: u64 = 0,
    backend_dispatch_count: u64 = 0,
    dispatch_plan_supported: u64 = 0,
    dispatch_plan_covered_op_count: u64 = 0,
    dispatch_plan_first_unsupported_op: u64 = std.math.maxInt(u64),
    dispatch_plan_projection_count: u64 = 0,
    dispatch_plan_row_count: u64 = 0,
    dispatch_plan_attention_count: u64 = 0,
    dispatch_plan_movement_count: u64 = 0,
    dispatch_plan_elementwise_count: u64 = 0,
    dispatch_plan_rope_count: u64 = 0,
    dispatch_plan_quantized_projection_count: u64 = 0,
    binding_requirement_hash: u64 = 0,
    persistent_requirement_count: u64 = 0,
    step_input_requirement_count: u64 = 0,
    step_output_requirement_count: u64 = 0,
};

pub const zgml_llama_program_inspection = extern struct {
    vocab_size: u64 = 0,
    max_seq_len: u64 = 0,
    context_len: u64 = 0,
    batch: u64 = 0,
    d_model: u64 = 0,
    n_layers: u64 = 0,
    n_heads: u64 = 0,
    n_kv_heads: u64 = 0,
    semantic_stage_count: u64 = 0,
    semantic_token_count: u64 = 0,
    semantic_layer_stage_count: u64 = 0,
    semantic_terminal_stage_count: u64 = 0,
    semantic_runtime_patch_holes: u64 = 0,
    semantic_runtime_patch_cache_write_pos_holes: u64 = 0,
    semantic_runtime_patch_attention_seq_kv_holes: u64 = 0,
};

pub const zgml_runtime_profile = extern struct {
    call_count: u64 = 0,
    backend_op_count: u64 = 0,
    fallback_op_count: u64 = 0,
    backend_dispatch_count: u64 = 0,
    sync_count: u64 = 0,
    runtime_patch_call_count: u64 = 0,
    runtime_patch_changed_count: u64 = 0,
    runtime_patch_invalid_count: u64 = 0,
    runtime_patch_holes: u64 = 0,
    runtime_patch_cache_write_pos_holes: u64 = 0,
    runtime_patch_attention_seq_kv_holes: u64 = 0,
    runtime_patch_stencil_hash: u64 = 0,
    command_count: u64 = 0,
    command_stencil_hash: u64 = 0,
    command_op_count: u64 = 0,
    command_row_count: u64 = 0,
    command_projection_count: u64 = 0,
    command_attention_count: u64 = 0,
    command_movement_count: u64 = 0,
    command_elementwise_count: u64 = 0,
    command_rope_count: u64 = 0,
};

const BufferHandle = struct {
    data: []u8 = &.{},
    owned_data: ?[]align(16) u8 = null,
    external: ?ExternalResourceBuffer = null,
    device_owner: ?DeviceBufferOwner = null,

    const ExternalResourceBuffer = struct {
        resource: backend_mod.ProgramIO.ExternalResource,
        view_byte_len: u32,
    };

    const DeviceBufferOwner = struct {
        program: *ProgramHandle,
    };

    fn isHost(self: *const BufferHandle) bool {
        return self.external == null;
    }

    fn byteLen(self: *const BufferHandle) usize {
        if (self.external) |external| return external.view_byte_len;
        return self.data.len;
    }

    fn hostData(self: *BufferHandle) ?[]u8 {
        if (!self.isHost()) return null;
        return self.data;
    }

    fn externalResource(self: *const BufferHandle) ?backend_mod.ProgramIO.ExternalResource {
        if (self.external) |external| return external.resource;
        return null;
    }

    fn deinit(self: *BufferHandle, allocator: std.mem.Allocator) void {
        if (self.device_owner) |owner| {
            releaseDeviceBuffer(owner.program, self.external.?.resource);
            owner.program.release(allocator);
        }
        if (self.owned_data) |data| allocator.free(data);
        allocator.destroy(self);
    }
};

const TinyLinearModelHandle = struct {
    input_len: usize,
    output_len: usize,
};

const TinyMlpActivation = enum(u32) {
    relu = 1,
    gelu = 2,
    silu = 3,
    sigmoid = 4,
};

const TinyMlpModelHandle = struct {
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    activation: TinyMlpActivation,
};

const TinyLlamaModelHandle = struct {
    model: *TinyLlamaModel,

    fn deinit(self: *TinyLlamaModelHandle) void {
        self.model.deinit();
    }
};

const TinyLlama2LayerModelHandle = struct {
    model: *TinyLlama2LayerModel,

    fn deinit(self: *TinyLlama2LayerModelHandle) void {
        self.model.deinit();
    }
};

const SmolLM135MModelHandle = struct {
    model: *SmolLM135MModel,

    fn deinit(self: *SmolLM135MModelHandle) void {
        self.model.deinit();
    }
};

const ModelHandle = struct {
    ref_count: usize = 1,
    data: union(enum) {
        tiny_linear: TinyLinearModelHandle,
        tiny_mlp: TinyMlpModelHandle,
        tiny_llama: TinyLlamaModelHandle,
        tiny_llama_2layer: TinyLlama2LayerModelHandle,
        smollm_135m: SmolLM135MModelHandle,
    },

    fn retain(self: *ModelHandle) void {
        self.ref_count += 1;
    }

    fn release(self: *ModelHandle, allocator: std.mem.Allocator) void {
        std.debug.assert(self.ref_count > 0);
        self.ref_count -= 1;
        if (self.ref_count == 0) self.destroy(allocator);
    }

    fn destroy(self: *ModelHandle, allocator: std.mem.Allocator) void {
        switch (self.data) {
            .tiny_linear => {},
            .tiny_mlp => {},
            .tiny_llama => |*llama| llama.deinit(),
            .tiny_llama_2layer => |*llama| llama.deinit(),
            .smollm_135m => |*llama| llama.deinit(),
        }
        allocator.destroy(self);
    }
};

const TinyLinearProgramHandle = struct {
    input_len: usize,
    output_len: usize,
    backend: u32,
    execution_supported: bool,
    graph: GraphF32,
    program: DeviceF32.Program,
    input: *TensorF32,
    weights: *TensorF32,
    bias: *TensorF32,
    output: *TensorF32,

    fn deinit(self: *TinyLinearProgramHandle) void {
        self.program.deinit();
        self.graph.deinit();
    }
};

const TinyMlpProgramHandle = struct {
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    activation: TinyMlpActivation,
    backend: u32,
    execution_supported: bool,
    graph: GraphF32,
    program: DeviceF32.Program,
    input: *TensorF32,
    w0: *TensorF32,
    b0: *TensorF32,
    w1: *TensorF32,
    b1: *TensorF32,
    output: *TensorF32,

    fn deinit(self: *TinyMlpProgramHandle) void {
        self.program.deinit();
        self.graph.deinit();
    }
};

const ModuleParamKind = enum {
    weights,
    bias,
};

const ModuleParamBinding = struct {
    tensor: *TensorF32,
    kind: ModuleParamKind,
    offset: usize,
    len: usize,
};

const DirectLinearModule = struct {
    in_features: usize,
    out_features: usize,
    has_bias: bool,
};

const DirectRmsGeluLinearModule = struct {
    features: usize,
    out_features: usize,
    eps: f32,
};

const DirectLinearLogSoftmaxModule = struct {
    in_features: usize,
    out_features: usize,
    has_bias: bool,
};

const ModuleProgramHandle = struct {
    input_len: usize,
    output_len: usize,
    weights_len: usize,
    bias_len: usize,
    backend: u32,
    execution_supported: bool,
    direct_linear: ?DirectLinearModule = null,
    direct_rms_gelu_linear: ?DirectRmsGeluLinearModule = null,
    direct_linear_log_softmax: ?DirectLinearLogSoftmaxModule = null,
    graph: GraphF32,
    program: DeviceF32.Program,
    input: *TensorF32,
    output: *TensorF32,
    persistent_tensors: []*TensorF32,
    persistent_params: []ModuleParamBinding,

    fn deinit(self: *ModuleProgramHandle) void {
        self.program.deinit();
        self.graph.deinit();
        alloc.free(self.persistent_params);
        alloc.free(self.persistent_tensors);
    }
};

const TinyLlamaProgramHandle = struct {
    model: *ModelHandle,
    program: *TinyLlamaProgram,

    fn deinit(self: *TinyLlamaProgramHandle, allocator: std.mem.Allocator) void {
        self.program.deinit();
        self.model.release(allocator);
    }
};

const TinyLlama2LayerProgramHandle = struct {
    model: *ModelHandle,
    program: *TinyLlama2LayerProgram,

    fn deinit(self: *TinyLlama2LayerProgramHandle, allocator: std.mem.Allocator) void {
        self.program.deinit();
        self.model.release(allocator);
    }
};

const SmolLM135MProgramHandle = struct {
    model: *ModelHandle,
    program: *SmolLM135MProgram,

    fn deinit(self: *SmolLM135MProgramHandle, allocator: std.mem.Allocator) void {
        self.program.deinit();
        self.model.release(allocator);
    }
};

const ProgramHandle = struct {
    ref_count: usize = 1,
    data: union(enum) {
        tiny_linear: TinyLinearProgramHandle,
        tiny_mlp: TinyMlpProgramHandle,
        module: ModuleProgramHandle,
        tiny_llama: TinyLlamaProgramHandle,
        tiny_llama_2layer: TinyLlama2LayerProgramHandle,
        smollm_135m: SmolLM135MProgramHandle,
    },

    fn retain(self: *ProgramHandle) void {
        self.ref_count += 1;
    }

    fn release(self: *ProgramHandle, allocator: std.mem.Allocator) void {
        std.debug.assert(self.ref_count > 0);
        self.ref_count -= 1;
        if (self.ref_count == 0) self.destroy(allocator);
    }

    fn destroy(self: *ProgramHandle, allocator: std.mem.Allocator) void {
        switch (self.data) {
            .tiny_linear => |*linear| linear.deinit(),
            .tiny_mlp => |*mlp| mlp.deinit(),
            .module => |*module| module.deinit(),
            .tiny_llama => |*llama| llama.deinit(allocator),
            .tiny_llama_2layer => |*llama| llama.deinit(allocator),
            .smollm_135m => |*llama| llama.deinit(allocator),
        }
        allocator.destroy(self);
    }
};

const TinyLinearSessionHandle = struct {
    input_len: usize,
    output_len: usize,
    program: *ProgramHandle,
    session: DeviceF32.Session,
    weights_buf: []f32,
    bias_buf: []f32,
    input_buf: []f32,
    output_buf: []f32,
    scratch_buf: []f32 = &.{},
    owns_weights_buf: bool,
    owns_bias_buf: bool,
    owns_input_buf: bool,
    owns_output_buf: bool,
    owns_scratch_buf: bool = false,

    fn deinit(self: *TinyLinearSessionHandle, allocator: std.mem.Allocator) void {
        self.session.deinit();
        if (self.owns_scratch_buf) allocator.free(self.scratch_buf);
        if (self.owns_output_buf) allocator.free(self.output_buf);
        if (self.owns_input_buf) allocator.free(self.input_buf);
        if (self.owns_bias_buf) allocator.free(self.bias_buf);
        if (self.owns_weights_buf) allocator.free(self.weights_buf);
        self.program.release(allocator);
    }
};

const TinyLlamaSessionHandle = struct {
    program: *ProgramHandle,
    session: *TinyLlamaSession,
    output_buf: []f32 = &.{},
    output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
    output_is_resource: bool = false,
    owns_output_buf: bool = false,

    fn deinit(self: *TinyLlamaSessionHandle, allocator: std.mem.Allocator) void {
        self.session.deinit();
        if (self.owns_output_buf) allocator.free(self.output_buf);
        self.program.release(allocator);
    }
};

const TinyLlama2LayerSessionHandle = struct {
    program: *ProgramHandle,
    session: *TinyLlama2LayerSession,
    output_buf: []f32 = &.{},
    output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
    output_is_resource: bool = false,
    owns_output_buf: bool = false,

    fn deinit(self: *TinyLlama2LayerSessionHandle, allocator: std.mem.Allocator) void {
        self.session.deinit();
        if (self.owns_output_buf) allocator.free(self.output_buf);
        self.program.release(allocator);
    }
};

const SmolLM135MSessionHandle = struct {
    program: *ProgramHandle,
    session: *SmolLM135MSession,
    output_buf: []f32 = &.{},
    output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
    output_is_resource: bool = false,
    owns_output_buf: bool = false,

    fn deinit(self: *SmolLM135MSessionHandle, allocator: std.mem.Allocator) void {
        self.session.deinit();
        if (self.owns_output_buf) allocator.free(self.output_buf);
        self.program.release(allocator);
    }
};

const SessionHandle = struct {
    data: union(enum) {
        tiny_linear: TinyLinearSessionHandle,
        tiny_mlp: TinyLinearSessionHandle,
        module: TinyLinearSessionHandle,
        tiny_llama: TinyLlamaSessionHandle,
        tiny_llama_2layer: TinyLlama2LayerSessionHandle,
        smollm_135m: SmolLM135MSessionHandle,
    },

    fn deinit(self: *SessionHandle, allocator: std.mem.Allocator) void {
        switch (self.data) {
            .tiny_linear => |*linear| linear.deinit(allocator),
            .tiny_mlp => |*mlp| mlp.deinit(allocator),
            .module => |*module| module.deinit(allocator),
            .tiny_llama => |*llama| llama.deinit(allocator),
            .tiny_llama_2layer => |*llama| llama.deinit(allocator),
            .smollm_135m => |*llama| llama.deinit(allocator),
        }
        allocator.destroy(self);
    }
};

const alloc = std.heap.page_allocator;

fn status(s: Status) c_int {
    return @intFromEnum(s);
}

const wasm_host = if (builtin.target.cpu.arch == .wasm32) struct {
    extern "zgml_host" fn zgml_wasm_host_session_step(session: usize, desc_ptr: usize, result_ptr: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_step_no_output(session: usize, desc_ptr: usize, result_ptr: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_execute_tokens(session: usize, desc_ptr: usize, result_ptr: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_position(session: usize, out_position_ptr: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_runtime_profile(session: usize, out_profile_ptr: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_reset(session: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_reset_runtime_profile(session: usize) c_int;
    extern "zgml_host" fn zgml_wasm_host_session_free(session: usize) c_int;
} else struct {};

const wasm_host_unclaimed: c_int = -1;

fn ptrValue(ptr: anytype) usize {
    return if (ptr) |p| @intFromPtr(p) else 0;
}

fn maybeWasmHostSessionStep(session: ?*zgml_session, desc_ptr: ?*const zgml_step_desc, result_ptr: ?*zgml_step_result) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_step(ptrValue(session), ptrValue(desc_ptr), ptrValue(result_ptr));
}

fn maybeWasmHostSessionStepNoOutput(session: ?*zgml_session, desc_ptr: ?*const zgml_step_desc, result_ptr: ?*zgml_step_result) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_step_no_output(ptrValue(session), ptrValue(desc_ptr), ptrValue(result_ptr));
}

fn maybeWasmHostSessionExecuteTokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_execute_desc, result_ptr: ?*zgml_step_result) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_execute_tokens(ptrValue(session), ptrValue(desc_ptr), ptrValue(result_ptr));
}

fn maybeWasmHostSessionPosition(session: ?*zgml_session, out_position: ?*usize) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_position(ptrValue(session), ptrValue(out_position));
}

fn maybeWasmHostSessionRuntimeProfile(session: ?*zgml_session, out_profile: ?*zgml_runtime_profile) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_runtime_profile(ptrValue(session), ptrValue(out_profile));
}

fn maybeWasmHostSessionReset(session: ?*zgml_session) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_reset(ptrValue(session));
}

fn maybeWasmHostSessionResetRuntimeProfile(session: ?*zgml_session) ?c_int {
    if (comptime builtin.target.cpu.arch != .wasm32) return null;
    return wasm_host.zgml_wasm_host_session_reset_runtime_profile(ptrValue(session));
}

fn maybeWasmHostSessionFree(session: ?*zgml_session) void {
    if (comptime builtin.target.cpu.arch != .wasm32) return;
    _ = wasm_host.zgml_wasm_host_session_free(ptrValue(session));
}

fn runtimeFeatureFlags() u64 {
    return feature_buffer_handle |
        feature_model_auto |
        feature_runtime_profile |
        feature_webgpu_compile_only |
        feature_wasm_exports |
        feature_native_buffer_io |
        feature_native_argmax |
        feature_native_execute_argmax |
        feature_native_topk_sample |
        feature_program_requirements |
        feature_program_output_buffer |
        feature_native_generate_sample |
        feature_native_generate_argmax |
        feature_session_model_binding |
        feature_external_buffer |
        feature_external_resource_buffer |
        feature_llama_kv_resource_binding |
        feature_llama_kv_cache_requirements |
        feature_program_buffer_factory |
        feature_external_resource_access |
        feature_model_inspection |
        feature_program_model_compatibility |
        feature_buffer_inspection |
        feature_session_inspection |
        feature_program_resource_inspection |
        feature_program_memory_inspection |
        feature_program_shape_inspection |
        feature_program_patch_envelope_inspection |
        feature_session_binding_shape_inspection |
        feature_program_device_buffer |
        feature_program_device_buffer_import |
        feature_program_dispatch_plan_inspection |
        feature_abi_struct_size |
        feature_model_path_probe |
        feature_supported_checkpoints |
        feature_safetensors_header_probe |
        feature_safetensors_data_load |
        feature_safetensors_data_probe |
        feature_native_tiny_mlp |
        feature_native_module_program |
        feature_program_binding_requirements |
        feature_session_persistent_upload |
        feature_native_module_activation_chain |
        feature_native_eager_linear |
        feature_native_eager_linear_activation |
        feature_native_training_step |
        feature_native_eager_softmax |
        feature_native_eager_matmul |
        feature_native_eager_activation |
        feature_native_eager_elementwise |
        feature_native_eager_reduce |
        feature_native_eager_conv2d |
        feature_native_eager_pool2d |
        feature_native_eager_dot |
        (if (build_options.use_wgpu) feature_native_wgpu_execution else 0) |
        if (build_options.use_wgpu and build_options.experimental_llama_wgpu_execution) feature_experimental_llama_wgpu_execution else 0;
}

export fn zgml_abi_struct_size(kind: u32) usize {
    return switch (kind) {
        abi_struct_runtime_info => @sizeOf(zgml_runtime_info),
        abi_struct_model_desc => @sizeOf(zgml_model_desc),
        abi_struct_model_load_desc => @sizeOf(zgml_model_load_desc),
        abi_struct_model_inspection => @sizeOf(zgml_model_inspection),
        abi_struct_program_model_compatibility => @sizeOf(zgml_program_model_compatibility),
        abi_struct_session_inspection => @sizeOf(zgml_session_inspection),
        abi_struct_compile_desc => @sizeOf(zgml_compile_desc),
        abi_struct_bind_desc => @sizeOf(zgml_bind_desc),
        abi_struct_buffer_desc => @sizeOf(zgml_buffer_desc),
        abi_struct_buffer_inspection => @sizeOf(zgml_buffer_inspection),
        abi_struct_external_resource_desc => @sizeOf(zgml_external_resource_desc),
        abi_struct_device_buffer_import_desc => @sizeOf(zgml_device_buffer_import_desc),
        abi_struct_buffer_bind_desc => @sizeOf(zgml_buffer_bind_desc),
        abi_struct_llama_kv_cache_bind_desc => @sizeOf(zgml_llama_kv_cache_bind_desc),
        abi_struct_llama_buffer_bind_desc => @sizeOf(zgml_llama_buffer_bind_desc),
        abi_struct_step_desc => @sizeOf(zgml_step_desc),
        abi_struct_step_result => @sizeOf(zgml_step_result),
        abi_struct_token_step_desc => @sizeOf(zgml_token_step_desc),
        abi_struct_token_advance_desc => @sizeOf(zgml_token_advance_desc),
        abi_struct_token_advance_tokens_desc => @sizeOf(zgml_token_advance_tokens_desc),
        abi_struct_token_prefill_desc => @sizeOf(zgml_token_prefill_desc),
        abi_struct_token_execute_desc => @sizeOf(zgml_token_execute_desc),
        abi_struct_token_argmax_desc => @sizeOf(zgml_token_argmax_desc),
        abi_struct_token_argmax_result => @sizeOf(zgml_token_argmax_result),
        abi_struct_token_execute_argmax_desc => @sizeOf(zgml_token_execute_argmax_desc),
        abi_struct_token_generate_argmax_desc => @sizeOf(zgml_token_generate_argmax_desc),
        abi_struct_token_generate_argmax_result => @sizeOf(zgml_token_generate_argmax_result),
        abi_struct_token_sample_desc => @sizeOf(zgml_token_sample_desc),
        abi_struct_token_sample_result => @sizeOf(zgml_token_sample_result),
        abi_struct_token_execute_sample_desc => @sizeOf(zgml_token_execute_sample_desc),
        abi_struct_token_generate_sample_desc => @sizeOf(zgml_token_generate_sample_desc),
        abi_struct_token_generate_sample_result => @sizeOf(zgml_token_generate_sample_result),
        abi_struct_program_requirements => @sizeOf(zgml_program_requirements),
        abi_struct_llama_kv_cache_requirements => @sizeOf(zgml_llama_kv_cache_requirements),
        abi_struct_program_inspection => @sizeOf(zgml_program_inspection),
        abi_struct_llama_program_inspection => @sizeOf(zgml_llama_program_inspection),
        abi_struct_runtime_profile => @sizeOf(zgml_runtime_profile),
        abi_struct_safetensors_header_probe_desc => @sizeOf(zgml_safetensors_header_probe_desc),
        abi_struct_safetensors_data_load_desc => @sizeOf(zgml_safetensors_data_load_desc),
        abi_struct_module_op_desc => @sizeOf(zgml_module_op_desc),
        abi_struct_module_desc => @sizeOf(zgml_module_desc),
        else => 0,
    };
}

export fn zgml_get_runtime_info(out_info: ?*zgml_runtime_info) c_int {
    const out = out_info orelse return status(.invalid_argument);
    out.* = .{
        .abi_version = abi_version,
        .size_t_bytes = @sizeOf(usize),
        .pointer_bytes = @sizeOf(usize),
        .token_id_bytes = @sizeOf(u32),
        .feature_flags = runtimeFeatureFlags(),
    };
    return status(.ok);
}

export fn zgml_wasm_alloc(len: usize) usize {
    if (len == 0) return 0;
    const buf = alloc.alignedAlloc(u8, .@"16", len) catch return 0;
    return @intFromPtr(buf.ptr);
}

export fn zgml_wasm_free(ptr_value: usize, len: usize) void {
    if (ptr_value == 0 or len == 0) return;
    const ptr: [*]align(16) u8 = @ptrFromInt(ptr_value);
    alloc.free(ptr[0..len]);
}

fn createBuffer(byte_len: usize, out: *?*zgml_buffer) c_int {
    if (byte_len == 0) return status(.invalid_argument);
    const data = alloc.alignedAlloc(u8, .@"16", byte_len) catch return status(.out_of_memory);
    const handle = alloc.create(BufferHandle) catch {
        alloc.free(data);
        return status(.out_of_memory);
    };
    handle.* = .{ .data = data, .owned_data = data };
    out.* = @ptrCast(handle);
    return status(.ok);
}

fn deviceFromBackendId(placement: u32) ?backend_mod.Device {
    return switch (placement) {
        backend_cpu => .cpu,
        backend_metal => .metal,
        backend_webgpu => .webgpu,
        else => null,
    };
}

fn backendIdForDevice(device: backend_mod.Device) u32 {
    return switch (device) {
        .cpu => backend_cpu,
        .metal => backend_metal,
        .webgpu => backend_webgpu,
    };
}

fn bufferStorageForDeviceBinding(storage: DeviceF32.BindingStorage) u32 {
    return switch (storage) {
        .none => buffer_storage_none,
        .host => buffer_storage_host,
        .external_resource => buffer_storage_external_resource,
    };
}

fn fillDeviceSessionInspection(out: *zgml_session_inspection, model_kind: u32, context_len: usize, session: *const DeviceF32.Session) void {
    const inspection = session.inspect();
    out.* = .{
        .model_kind = model_kind,
        .backend = backendIdForDevice(inspection.backend),
        .output_storage = bufferStorageForDeviceBinding(inspection.output_storage),
        .kv_cache_storage = buffer_storage_none,
        .position = 0,
        .context_len = @intCast(context_len),
        .persistent_binding_count = @intCast(inspection.persistent_binding_count),
        .step_input_count = @intCast(inspection.step_input_count),
        .step_output_count = @intCast(inspection.step_output_count),
        .host_binding_count = @intCast(inspection.host_binding_count),
        .resource_binding_count = @intCast(inspection.resource_binding_count),
        .binding_shape_hash = inspection.binding_shape_hash,
    };
}

const DeviceProgramRuntime = struct {
    backend: u32,
    execution_supported: bool,
    program: *DeviceF32.Program,
};

fn deviceProgramRuntime(p: *ProgramHandle) ?DeviceProgramRuntime {
    return switch (p.data) {
        .tiny_linear => |*linear| .{
            .backend = linear.backend,
            .execution_supported = linear.execution_supported,
            .program = &linear.program,
        },
        .tiny_mlp => |*mlp| .{
            .backend = mlp.backend,
            .execution_supported = mlp.execution_supported,
            .program = &mlp.program,
        },
        .module => |*module| .{
            .backend = module.backend,
            .execution_supported = module.execution_supported,
            .program = &module.program,
        },
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => null,
    };
}

const DeviceSessionRuntime = struct {
    execution_supported: bool,
    program: *DeviceF32.Program,
    session: *DeviceF32.Session,
};

fn deviceSessionRuntime(s: *SessionHandle) ?DeviceSessionRuntime {
    return switch (s.data) {
        .tiny_linear => |*linear| blk: {
            const runtime = deviceProgramRuntime(linear.program) orelse break :blk null;
            break :blk .{
                .execution_supported = runtime.execution_supported,
                .program = runtime.program,
                .session = &linear.session,
            };
        },
        .tiny_mlp => |*mlp| blk: {
            const runtime = deviceProgramRuntime(mlp.program) orelse break :blk null;
            break :blk .{
                .execution_supported = runtime.execution_supported,
                .program = runtime.program,
                .session = &mlp.session,
            };
        },
        .module => |*module| blk: {
            const runtime = deviceProgramRuntime(module.program) orelse break :blk null;
            break :blk .{
                .execution_supported = runtime.execution_supported,
                .program = runtime.program,
                .session = &module.session,
            };
        },
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => null,
    };
}

fn resourceAccessFromFlags(flags: u32) ?backend_mod.ProgramIO.ExternalResource.Access {
    if (flags == 0) return .read_write;
    if ((flags & ~resource_access_read_write) != 0) return null;
    return backend_mod.ProgramIO.ExternalResource.Access.fromFlags(flags);
}

fn programDeviceBufferAccess(kind: u32) ?backend_mod.ProgramIO.ExternalResource.Access {
    return switch (kind) {
        program_buffer_weights, program_buffer_bias, program_buffer_input => .read_only,
        program_buffer_output => .write_only,
        program_buffer_llama_k_cache, program_buffer_llama_v_cache => .read_write,
        else => null,
    };
}

fn releaseDeviceBuffer(program: *ProgramHandle, resource: backend_mod.ProgramIO.ExternalResource) void {
    if (!build_options.use_wgpu) return;
    switch (program.data) {
        .tiny_linear => |*linear| {
            if (linear.backend == backend_webgpu and linear.execution_supported) {
                wgpu_mod.releaseDeviceBuffer(linear.program.handle, resource);
            }
        },
        .tiny_mlp => {},
        .module => |*module| {
            if (module.backend == backend_webgpu and module.execution_supported) {
                wgpu_mod.releaseDeviceBuffer(module.program.handle, resource);
            }
        },
        .tiny_llama => |*llama| llama.program.releaseDeviceBuffer(resource),
        .tiny_llama_2layer => |*llama| llama.program.releaseDeviceBuffer(resource),
        .smollm_135m => |*llama| llama.program.releaseDeviceBuffer(resource),
    }
}

fn getDeviceHandle(program: *ProgramHandle, placement: u32) !usize {
    const device = deviceFromBackendId(placement) orelse return error.InvalidArgument;
    if (device != .webgpu or !build_options.use_wgpu) return error.Unsupported;
    return switch (program.data) {
        .tiny_linear => |*linear| blk: {
            if (linear.backend != backend_webgpu or !linear.execution_supported) break :blk error.Unsupported;
            break :blk try wgpu_mod.deviceHandle(linear.program.handle);
        },
        .tiny_mlp => error.Unsupported,
        .module => |*module| blk: {
            if (module.backend != backend_webgpu or !module.execution_supported) break :blk error.Unsupported;
            break :blk try wgpu_mod.deviceHandle(module.program.handle);
        },
        .tiny_llama => |*llama| llama.program.deviceHandle(),
        .tiny_llama_2layer => |*llama| llama.program.deviceHandle(),
        .smollm_135m => |*llama| llama.program.deviceHandle(),
    };
}

fn importDeviceBuffer(program: *ProgramHandle, resource: DeviceBufferImportResource) !backend_mod.ProgramIO.ExternalResource {
    if (!build_options.use_wgpu) return error.Unsupported;
    return switch (program.data) {
        .tiny_linear => |*linear| blk: {
            if (linear.backend != backend_webgpu or !linear.execution_supported) break :blk error.Unsupported;
            break :blk try wgpu_mod.importDeviceBuffer(
                linear.program.handle,
                resource.device_handle,
                resource.buffer_handle,
                resource.byte_offset,
                resource.byte_len,
                resource.access,
            );
        },
        .tiny_mlp => error.Unsupported,
        .module => |*module| blk: {
            if (module.backend != backend_webgpu or !module.execution_supported) break :blk error.Unsupported;
            break :blk try wgpu_mod.importDeviceBuffer(
                module.program.handle,
                resource.device_handle,
                resource.buffer_handle,
                resource.byte_offset,
                resource.byte_len,
                resource.access,
            );
        },
        .tiny_llama => |*llama| llama.program.importDeviceBuffer(
            resource.device_handle,
            resource.buffer_handle,
            resource.byte_offset,
            resource.byte_len,
            resource.access,
        ),
        .tiny_llama_2layer => |*llama| llama.program.importDeviceBuffer(
            resource.device_handle,
            resource.buffer_handle,
            resource.byte_offset,
            resource.byte_len,
            resource.access,
        ),
        .smollm_135m => |*llama| llama.program.importDeviceBuffer(
            resource.device_handle,
            resource.buffer_handle,
            resource.byte_offset,
            resource.byte_len,
            resource.access,
        ),
    };
}

const DeviceBufferImportResource = struct {
    device_handle: usize,
    buffer_handle: usize,
    byte_offset: usize,
    byte_len: usize,
    access: backend_mod.ProgramIO.ExternalResource.Access,
};

fn writeDeviceBuffer(owner: BufferHandle.DeviceBufferOwner, resource: backend_mod.ProgramIO.ExternalResource, byte_offset: usize, src: []const u8) !void {
    if (!build_options.use_wgpu) return error.Unsupported;
    return switch (owner.program.data) {
        .tiny_linear => |*linear| blk: {
            if (linear.backend != backend_webgpu or !linear.execution_supported) break :blk error.Unsupported;
            try wgpu_mod.writeDeviceBuffer(linear.program.handle, resource, byte_offset, src);
        },
        .tiny_mlp => error.Unsupported,
        .module => |*module| {
            if (module.backend != backend_webgpu or !module.execution_supported) return error.Unsupported;
            try wgpu_mod.writeDeviceBuffer(module.program.handle, resource, byte_offset, src);
        },
        .tiny_llama => |*llama| try llama.program.writeDeviceBuffer(resource, byte_offset, src),
        .tiny_llama_2layer => |*llama| try llama.program.writeDeviceBuffer(resource, byte_offset, src),
        .smollm_135m => |*llama| try llama.program.writeDeviceBuffer(resource, byte_offset, src),
    };
}

fn readDeviceBuffer(owner: BufferHandle.DeviceBufferOwner, resource: backend_mod.ProgramIO.ExternalResource, byte_offset: usize, dst: []u8) !void {
    if (!build_options.use_wgpu) return error.Unsupported;
    return switch (owner.program.data) {
        .tiny_linear => |*linear| blk: {
            if (linear.backend != backend_webgpu or !linear.execution_supported) break :blk error.Unsupported;
            try wgpu_mod.readDeviceBuffer(linear.program.handle, resource, byte_offset, dst);
        },
        .tiny_mlp => error.Unsupported,
        .module => |*module| {
            if (module.backend != backend_webgpu or !module.execution_supported) return error.Unsupported;
            try wgpu_mod.readDeviceBuffer(module.program.handle, resource, byte_offset, dst);
        },
        .tiny_llama => |*llama| try llama.program.readDeviceBuffer(resource, byte_offset, dst),
        .tiny_llama_2layer => |*llama| try llama.program.readDeviceBuffer(resource, byte_offset, dst),
        .smollm_135m => |*llama| try llama.program.readDeviceBuffer(resource, byte_offset, dst),
    };
}

export fn zgml_buffer_create(desc_ptr: ?*const zgml_buffer_desc, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    return createBuffer(desc.byte_len, out);
}

export fn zgml_buffer_wrap(data_ptr: ?*anyopaque, byte_len: usize, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    if (byte_len == 0) return status(.invalid_argument);
    const ptr = data_ptr orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    const handle = alloc.create(BufferHandle) catch return status(.out_of_memory);
    const bytes: [*]u8 = @ptrCast(ptr);
    handle.* = .{ .data = bytes[0..byte_len] };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_buffer_wrap_resource(desc_ptr: ?*const zgml_external_resource_desc, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    if (desc.byte_len == 0 or desc.handle == 0) return status(.invalid_argument);
    const placement = deviceFromBackendId(desc.placement) orelse return status(.invalid_argument);
    const access = resourceAccessFromFlags(desc.access_flags) orelse return status(.invalid_argument);
    const end = std.math.add(usize, desc.byte_offset, desc.byte_len) catch return status(.shape_mismatch);
    const byte_offset = std.math.cast(u32, desc.byte_offset) orelse return status(.shape_mismatch);
    const byte_len = std.math.cast(u32, desc.byte_len) orelse return status(.shape_mismatch);
    const resource_len = std.math.cast(u32, end) orelse return status(.shape_mismatch);
    const handle = alloc.create(BufferHandle) catch return status(.out_of_memory);
    handle.* = .{
        .external = .{
            .resource = .{
                .placement = placement,
                .handle = desc.handle,
                .byte_offset = byte_offset,
                .byte_len = resource_len,
                .access = access,
            },
            .view_byte_len = byte_len,
        },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_buffer_data(buffer: ?*zgml_buffer) ?*anyopaque {
    const handle = bufferHandle(buffer) orelse return null;
    const data = handle.hostData() orelse return null;
    return @ptrCast(data.ptr);
}

export fn zgml_buffer_size(buffer: ?*zgml_buffer) usize {
    const handle = bufferHandle(buffer) orelse return 0;
    return handle.byteLen();
}

export fn zgml_buffer_inspect(buffer: ?*zgml_buffer, out_inspection: ?*zgml_buffer_inspection) c_int {
    const handle = bufferHandle(buffer) orelse return status(.invalid_argument);
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{
        .storage = buffer_storage_host,
        .byte_len = @intCast(handle.byteLen()),
    };
    if (handle.external) |external| {
        out.* = .{
            .storage = buffer_storage_external_resource,
            .placement = backendIdForDevice(external.resource.placement),
            .access_flags = external.resource.access.toFlags(),
            .byte_len = external.view_byte_len,
            .handle = @intCast(external.resource.handle),
            .byte_offset = external.resource.byte_offset,
            .resource_byte_len = external.resource.byte_len,
        };
    }
    return status(.ok);
}

export fn zgml_buffer_write(buffer: ?*zgml_buffer, byte_offset: usize, src: ?*const anyopaque, byte_len: usize) c_int {
    const handle = bufferHandle(buffer) orelse return status(.invalid_argument);
    const ptr = src orelse {
        if (byte_len == 0) return status(.ok);
        return status(.invalid_argument);
    };
    if (handle.device_owner) |owner| {
        if (byte_offset > handle.byteLen() or byte_len > handle.byteLen() - byte_offset) return status(.shape_mismatch);
        const bytes: [*]const u8 = @ptrCast(ptr);
        writeDeviceBuffer(owner, handle.external.?.resource, byte_offset, bytes[0..byte_len]) catch |err| return compileErrorStatus(err);
        return status(.ok);
    }
    const data = handle.hostData() orelse return status(.unsupported);
    if (byte_offset > data.len or byte_len > data.len - byte_offset) return status(.shape_mismatch);
    const bytes: [*]const u8 = @ptrCast(ptr);
    @memcpy(data[byte_offset..][0..byte_len], bytes[0..byte_len]);
    return status(.ok);
}

export fn zgml_buffer_read(buffer: ?*zgml_buffer, byte_offset: usize, dst: ?*anyopaque, byte_len: usize) c_int {
    const handle = bufferHandle(buffer) orelse return status(.invalid_argument);
    const ptr = dst orelse {
        if (byte_len == 0) return status(.ok);
        return status(.invalid_argument);
    };
    if (handle.device_owner) |owner| {
        if (byte_offset > handle.byteLen() or byte_len > handle.byteLen() - byte_offset) return status(.shape_mismatch);
        const bytes: [*]u8 = @ptrCast(ptr);
        readDeviceBuffer(owner, handle.external.?.resource, byte_offset, bytes[0..byte_len]) catch |err| return compileErrorStatus(err);
        return status(.ok);
    }
    const data = handle.hostData() orelse return status(.unsupported);
    if (byte_offset > data.len or byte_len > data.len - byte_offset) return status(.shape_mismatch);
    const bytes: [*]u8 = @ptrCast(ptr);
    @memcpy(bytes[0..byte_len], data[byte_offset..][0..byte_len]);
    return status(.ok);
}

export fn zgml_buffer_free(buffer: ?*zgml_buffer) void {
    const handle = bufferHandle(buffer) orelse return;
    handle.deinit(alloc);
}

fn checkedElementCount(a: usize, b: usize) ?usize {
    return std.math.mul(usize, a, b) catch null;
}

fn checkedElementCount4(a: usize, b: usize, c: usize, d: usize) ?usize {
    const ab = checkedElementCount(a, b) orelse return null;
    const abc = checkedElementCount(ab, c) orelse return null;
    return checkedElementCount(abc, d);
}

fn addBiasRowsF32(
    output: []f32,
    bias: []const f32,
    batch: usize,
    out_features: usize,
) void {
    const V = 8;
    const VecT = @Vector(V, f32);
    for (0..batch) |row| {
        const output_row = output[row * out_features ..][0..out_features];
        var col: usize = 0;
        while (col + V <= out_features) : (col += V) {
            const out_v: VecT = output_row[col..][0..V].*;
            const bias_v: VecT = bias[col..][0..V].*;
            output_row[col..][0..V].* = out_v + bias_v;
        }
        while (col < out_features) : (col += 1) {
            output_row[col] += bias[col];
        }
    }
}

fn eagerActivationF32(value: f32, activation: u32) !f32 {
    return switch (activation) {
        0 => value,
        module_activation_relu => if (value > 0) value else 0,
        module_activation_gelu => blk: {
            const inner = @as(f32, 0.7978845608028654) * value * (1.0 + @as(f32, 0.044715) * value * value);
            break :blk 0.5 * value * (1.0 + std.math.tanh(inner));
        },
        module_activation_silu => value / (1.0 + @exp(-value)),
        module_activation_sigmoid => 1.0 / (1.0 + @exp(-value)),
        module_activation_tanh => std.math.tanh(value),
        else => error.InvalidArgument,
    };
}

fn writeActivationF32(input: []const f32, output: []f32, activation: u32) !void {
    if (input.len != output.len) return error.ShapeMismatch;
    const V = 8;
    const VecT = @Vector(V, f32);
    const zero: VecT = @splat(0);
    const one: VecT = @splat(1);
    switch (activation) {
        0 => {
            if (input.ptr != output.ptr) @memcpy(output, input);
        },
        module_activation_relu => {
            var i: usize = 0;
            while (i + V <= input.len) : (i += V) {
                const x: VecT = input[i..][0..V].*;
                output[i..][0..V].* = @max(x, zero);
            }
            while (i < input.len) : (i += 1) output[i] = if (input[i] > 0) input[i] else 0;
        },
        module_activation_sigmoid => {
            var i: usize = 0;
            while (i + V <= input.len) : (i += V) {
                const x: VecT = input[i..][0..V].*;
                output[i..][0..V].* = one / (one + @exp(-x));
            }
            while (i < input.len) : (i += 1) output[i] = 1.0 / (1.0 + @exp(-input[i]));
        },
        module_activation_silu => {
            var i: usize = 0;
            while (i + V <= input.len) : (i += V) {
                const x: VecT = input[i..][0..V].*;
                output[i..][0..V].* = x * (one / (one + @exp(-x)));
            }
            while (i < input.len) : (i += 1) output[i] = input[i] / (1.0 + @exp(-input[i]));
        },
        module_activation_gelu => {
            var i: usize = 0;
            while (i + V <= input.len) : (i += V) {
                const x: VecT = input[i..][0..V].*;
                output[i..][0..V].* = geluApproxVec8(x);
            }
            while (i < input.len) : (i += 1) output[i] = try eagerActivationF32(input[i], activation);
        },
        module_activation_tanh => {
            var i: usize = 0;
            while (i + V <= input.len) : (i += V) {
                const x: VecT = input[i..][0..V].*;
                output[i..][0..V].* = tanhApproxVec8(x);
            }
            while (i < input.len) : (i += 1) output[i] = std.math.tanh(input[i]);
        },
        else => return error.InvalidArgument,
    }
}

const eager_elementwise_add: u32 = 1;
const eager_elementwise_sub: u32 = 2;
const eager_elementwise_mul: u32 = 3;
const eager_elementwise_div: u32 = 4;
const eager_elementwise_neg: u32 = 5;
const eager_elementwise_exp: u32 = 6;
const eager_elementwise_log: u32 = 7;
const eager_elementwise_sqr: u32 = 8;
const eager_elementwise_recip: u32 = 9;
const eager_elementwise_abs: u32 = 10;
const eager_elementwise_sqrt: u32 = 11;
const eager_elementwise_maximum: u32 = 12;
const eager_elementwise_minimum: u32 = 13;
const eager_elementwise_eq: u32 = 15;
const eager_elementwise_ne: u32 = 16;
const eager_elementwise_lt: u32 = 17;
const eager_elementwise_le: u32 = 18;
const eager_elementwise_gt: u32 = 19;
const eager_elementwise_ge: u32 = 20;
const eager_reduce_sum: u32 = 1;
const eager_reduce_mean: u32 = 2;
const eager_reduce_max: u32 = 3;
const eager_reduce_min: u32 = 4;
const eager_reduce_prod: u32 = 5;
const eager_pool_max: u32 = 1;
const eager_pool_avg: u32 = 2;

fn eagerElementwiseUnaryF32(value: f32, op: u32) !f32 {
    return switch (op) {
        eager_elementwise_neg => -value,
        eager_elementwise_exp => @exp(value),
        eager_elementwise_log => @log(value),
        eager_elementwise_sqr => value * value,
        eager_elementwise_recip => 1.0 / value,
        eager_elementwise_abs => @abs(value),
        eager_elementwise_sqrt => @sqrt(value),
        else => error.InvalidArgument,
    };
}

fn eagerElementwiseBinaryF32(lhs: f32, rhs: f32, op: u32) !f32 {
    return switch (op) {
        eager_elementwise_add => lhs + rhs,
        eager_elementwise_sub => lhs - rhs,
        eager_elementwise_mul => lhs * rhs,
        eager_elementwise_div => lhs / rhs,
        eager_elementwise_maximum => @max(lhs, rhs),
        eager_elementwise_minimum => @min(lhs, rhs),
        eager_elementwise_eq => if (lhs == rhs or (std.math.isNan(lhs) and std.math.isNan(rhs))) 1.0 else 0.0,
        eager_elementwise_ne => if (lhs == rhs or (std.math.isNan(lhs) and std.math.isNan(rhs))) 0.0 else 1.0,
        eager_elementwise_lt => if (lhs < rhs) 1.0 else 0.0,
        eager_elementwise_le => if (lhs <= rhs) 1.0 else 0.0,
        eager_elementwise_gt => if (lhs > rhs) 1.0 else 0.0,
        eager_elementwise_ge => if (lhs >= rhs) 1.0 else 0.0,
        else => error.InvalidArgument,
    };
}

fn eagerElementwiseBinaryVec8(lhs: @Vector(8, f32), rhs: @Vector(8, f32), op: u32) !@Vector(8, f32) {
    const VecT = @Vector(8, f32);
    const zero: VecT = @splat(0.0);
    const one: VecT = @splat(1.0);
    return switch (op) {
        eager_elementwise_add => lhs + rhs,
        eager_elementwise_sub => lhs - rhs,
        eager_elementwise_mul => lhs * rhs,
        eager_elementwise_div => lhs / rhs,
        eager_elementwise_maximum => @max(lhs, rhs),
        eager_elementwise_minimum => @min(lhs, rhs),
        eager_elementwise_eq => blk: {
            const both_nan = (lhs != lhs) & (rhs != rhs);
            break :blk @select(f32, (lhs == rhs) | both_nan, one, zero);
        },
        eager_elementwise_ne => blk: {
            const both_nan = (lhs != lhs) & (rhs != rhs);
            break :blk @select(f32, (lhs == rhs) | both_nan, zero, one);
        },
        eager_elementwise_lt => @select(f32, lhs < rhs, one, zero),
        eager_elementwise_le => @select(f32, lhs <= rhs, one, zero),
        eager_elementwise_gt => @select(f32, lhs > rhs, one, zero),
        eager_elementwise_ge => @select(f32, lhs >= rhs, one, zero),
        else => error.InvalidArgument,
    };
}

fn writeElementwiseBinaryF32(lhs: []const f32, rhs: []const f32, output: []f32, op: u32) !void {
    if (lhs.len != output.len) return error.ShapeMismatch;
    if (rhs.len != 1 and rhs.len != lhs.len) return error.ShapeMismatch;

    const V = 8;
    const VecT = @Vector(V, f32);
    var i: usize = 0;
    if (rhs.len == 1) {
        const rhs_vec: VecT = @splat(rhs[0]);
        while (i + V <= lhs.len) : (i += V) {
            const lhs_vec: VecT = lhs[i..][0..V].*;
            output[i..][0..V].* = try eagerElementwiseBinaryVec8(lhs_vec, rhs_vec, op);
        }
    } else {
        while (i + V <= lhs.len) : (i += V) {
            const lhs_vec: VecT = lhs[i..][0..V].*;
            const rhs_vec: VecT = rhs[i..][0..V].*;
            output[i..][0..V].* = try eagerElementwiseBinaryVec8(lhs_vec, rhs_vec, op);
        }
    }
    while (i < lhs.len) : (i += 1) {
        output[i] = try eagerElementwiseBinaryF32(lhs[i], rhs[if (rhs.len == 1) 0 else i], op);
    }
}

fn dotF32(lhs: []const f32, rhs: []const f32) f32 {
    const V = 8;
    const VecT = @Vector(V, f32);
    var acc_vec: VecT = @splat(0.0);
    var i: usize = 0;
    while (i + V <= lhs.len) : (i += V) {
        const lhs_vec: VecT = lhs[i..][0..V].*;
        const rhs_vec: VecT = rhs[i..][0..V].*;
        acc_vec += lhs_vec * rhs_vec;
    }
    var acc = @reduce(.Add, acc_vec);
    while (i < lhs.len) : (i += 1) acc += lhs[i] * rhs[i];
    return acc;
}

fn applyActivationF32(output: []f32, activation: u32) !void {
    if (activation == 0) return;
    writeActivationF32(output, output, activation) catch |err| switch (err) {
        error.InvalidArgument => return error.InvalidArgument,
        error.ShapeMismatch => unreachable,
    };
}

fn writeSoftmaxRowsF32(input: []const f32, output: []f32, rows: usize, cols: usize, log_softmax: bool) void {
    for (0..rows) |row| {
        const input_row = input[row * cols ..][0..cols];
        const output_row = output[row * cols ..][0..cols];
        var max_val = -std.math.inf(f32);
        for (input_row) |value| max_val = @max(max_val, value);
        var sum_exp: f32 = 0;
        for (input_row) |value| sum_exp += @exp(value - max_val);
        if (log_softmax) {
            const log_denom = max_val + @log(sum_exp);
            for (input_row, 0..) |value, col| output_row[col] = value - log_denom;
        } else {
            const inv_sum = 1.0 / sum_exp;
            for (input_row, 0..) |value, col| output_row[col] = @exp(value - max_val) * inv_sum;
        }
    }
}

fn eagerLinearF32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    transposed_weights: bool,
) c_int {
    if (input_ptr == null or
        weights_ptr == null or
        output_ptr == null or
        batch == 0 or
        in_features == 0 or
        out_features == 0) return status(.invalid_argument);
    if (checkedElementCount(batch, in_features) != input_len) return status(.shape_mismatch);
    if (checkedElementCount(in_features, out_features) != weights_len) return status(.shape_mismatch);
    if (bias_len != 0 and (bias_ptr == null or bias_len != out_features)) return status(.shape_mismatch);
    if (checkedElementCount(batch, out_features) != output_len) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const weights = weights_ptr.?[0..weights_len];
    const bias = if (bias_len == 0) null else bias_ptr.?[0..bias_len];
    const output = output_ptr.?[0..output_len];
    const weight_row_stride: usize = if (transposed_weights) 1 else out_features;
    const weight_col_stride: usize = if (transposed_weights) in_features else 1;
    forward.blasSgemm(
        output,
        input,
        weights,
        batch,
        out_features,
        in_features,
        in_features,
        1,
        weight_row_stride,
        weight_col_stride,
        0,
        0,
        0,
        out_features,
    );
    if (bias) |b| addBiasRowsF32(output, b, batch, out_features);
    return status(.ok);
}

export fn zgml_eager_linear_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
) c_int {
    return eagerLinearF32(input_ptr, input_len, weights_ptr, weights_len, bias_ptr, bias_len, output_ptr, output_len, batch, in_features, out_features, false);
}

export fn zgml_eager_linear_transposed_weights_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
) c_int {
    return eagerLinearF32(input_ptr, input_len, weights_ptr, weights_len, bias_ptr, bias_len, output_ptr, output_len, batch, in_features, out_features, true);
}

export fn zgml_eager_matmul_f32(
    lhs_ptr: ?[*]const f32,
    lhs_len: usize,
    rhs_ptr: ?[*]const f32,
    rhs_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    rows: usize,
    shared: usize,
    cols: usize,
) c_int {
    if (lhs_ptr == null or
        rhs_ptr == null or
        output_ptr == null or
        rows == 0 or
        shared == 0 or
        cols == 0) return status(.invalid_argument);
    if (checkedElementCount(rows, shared) != lhs_len) return status(.shape_mismatch);
    if (checkedElementCount(shared, cols) != rhs_len) return status(.shape_mismatch);
    if (checkedElementCount(rows, cols) != output_len) return status(.shape_mismatch);

    forward.blasSgemm(
        output_ptr.?[0..output_len],
        lhs_ptr.?[0..lhs_len],
        rhs_ptr.?[0..rhs_len],
        rows,
        cols,
        shared,
        shared,
        1,
        cols,
        1,
        0,
        0,
        0,
        cols,
    );
    return status(.ok);
}

fn eagerLinearActivationF32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    activation: u32,
    transposed_weights: bool,
) c_int {
    if (input_ptr == null or
        weights_ptr == null or
        output_ptr == null or
        batch == 0 or
        in_features == 0 or
        out_features == 0) return status(.invalid_argument);
    if (checkedElementCount(batch, in_features) != input_len) return status(.shape_mismatch);
    if (checkedElementCount(in_features, out_features) != weights_len) return status(.shape_mismatch);
    if (bias_len != 0 and (bias_ptr == null or bias_len != out_features)) return status(.shape_mismatch);
    if (checkedElementCount(batch, out_features) != output_len) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const weights = weights_ptr.?[0..weights_len];
    const bias = if (bias_len == 0) null else bias_ptr.?[0..bias_len];
    const output = output_ptr.?[0..output_len];
    const weight_row_stride: usize = if (transposed_weights) 1 else out_features;
    const weight_col_stride: usize = if (transposed_weights) in_features else 1;
    forward.blasSgemm(
        output,
        input,
        weights,
        batch,
        out_features,
        in_features,
        in_features,
        1,
        weight_row_stride,
        weight_col_stride,
        0,
        0,
        0,
        out_features,
    );
    if (bias) |b| addBiasRowsF32(output, b, batch, out_features);
    applyActivationF32(output, activation) catch |err| return switch (err) {
        error.InvalidArgument => status(.invalid_argument),
    };
    return status(.ok);
}

export fn zgml_eager_linear_activation_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    activation: u32,
) c_int {
    return eagerLinearActivationF32(input_ptr, input_len, weights_ptr, weights_len, bias_ptr, bias_len, output_ptr, output_len, batch, in_features, out_features, activation, false);
}

export fn zgml_eager_linear_activation_transposed_weights_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    activation: u32,
) c_int {
    return eagerLinearActivationF32(input_ptr, input_len, weights_ptr, weights_len, bias_ptr, bias_len, output_ptr, output_len, batch, in_features, out_features, activation, true);
}

export fn zgml_eager_activation_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    activation: u32,
) c_int {
    if (input_ptr == null or output_ptr == null or input_len == 0) return status(.invalid_argument);
    if (input_len != output_len) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const output = output_ptr.?[0..output_len];
    writeActivationF32(input, output, activation) catch |err| return switch (err) {
        error.InvalidArgument => status(.invalid_argument),
        error.ShapeMismatch => status(.shape_mismatch),
    };
    return status(.ok);
}

export fn zgml_eager_elementwise_f32(
    lhs_ptr: ?[*]const f32,
    lhs_len: usize,
    rhs_ptr: ?[*]const f32,
    rhs_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    op: u32,
) c_int {
    if (lhs_ptr == null or output_ptr == null or lhs_len == 0) return status(.invalid_argument);
    if (lhs_len != output_len) return status(.shape_mismatch);
    const lhs = lhs_ptr.?[0..lhs_len];
    const output = output_ptr.?[0..output_len];

    if (rhs_len == 0) {
        if (rhs_ptr != null) return status(.shape_mismatch);
        for (lhs, output) |value, *out| {
            out.* = eagerElementwiseUnaryF32(value, op) catch |err| return switch (err) {
                error.InvalidArgument => status(.invalid_argument),
            };
        }
        return status(.ok);
    }

    if (rhs_ptr == null) return status(.invalid_argument);
    if (rhs_len != 1 and rhs_len != lhs_len) return status(.shape_mismatch);
    const rhs = rhs_ptr.?[0..rhs_len];
    writeElementwiseBinaryF32(lhs, rhs, output, op) catch |err| return switch (err) {
        error.InvalidArgument => status(.invalid_argument),
        error.ShapeMismatch => status(.shape_mismatch),
    };
    return status(.ok);
}

export fn zgml_eager_where_f32(
    condition_ptr: ?[*]const f32,
    condition_len: usize,
    input_ptr: ?[*]const f32,
    input_len: usize,
    other_ptr: ?[*]const f32,
    other_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
) c_int {
    if (condition_ptr == null or input_ptr == null or other_ptr == null or output_ptr == null or condition_len == 0) return status(.invalid_argument);
    if (condition_len != output_len) return status(.shape_mismatch);
    if (input_len != 1 and input_len != condition_len) return status(.shape_mismatch);
    if (other_len != 1 and other_len != condition_len) return status(.shape_mismatch);

    const condition = condition_ptr.?[0..condition_len];
    const input = input_ptr.?[0..input_len];
    const other = other_ptr.?[0..other_len];
    const output = output_ptr.?[0..output_len];
    for (condition, output, 0..) |cond, *out, i| {
        out.* = if (cond != 0) input[if (input_len == 1) 0 else i] else other[if (other_len == 1) 0 else i];
    }
    return status(.ok);
}

export fn zgml_eager_clamp_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    min_value: f32,
    max_value: f32,
    has_min: u32,
    has_max: u32,
) c_int {
    if (input_ptr == null or output_ptr == null or input_len == 0) return status(.invalid_argument);
    if (input_len != output_len) return status(.shape_mismatch);
    const use_min = has_min != 0;
    const use_max = has_max != 0;
    if (!use_min and !use_max) return status(.invalid_argument);
    if (use_min and !std.math.isFinite(min_value)) return status(.invalid_argument);
    if (use_max and !std.math.isFinite(max_value)) return status(.invalid_argument);
    if (use_min and use_max and min_value > max_value) return status(.invalid_argument);

    const input = input_ptr.?[0..input_len];
    const output = output_ptr.?[0..output_len];
    for (input, output) |value, *out| {
        var result = value;
        if (use_min) result = @max(result, min_value);
        if (use_max) result = @min(result, max_value);
        out.* = result;
    }
    return status(.ok);
}

export fn zgml_eager_reduce_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    op: u32,
) c_int {
    if (input_ptr == null or output_ptr == null or input_len == 0) return status(.invalid_argument);
    if (output_len != 1) return status(.shape_mismatch);
    const input = input_ptr.?[0..input_len];
    var acc: f32 = switch (op) {
        eager_reduce_sum, eager_reduce_mean => 0,
        eager_reduce_max => -std.math.inf(f32),
        eager_reduce_min => std.math.inf(f32),
        eager_reduce_prod => 1,
        else => return status(.invalid_argument),
    };
    switch (op) {
        eager_reduce_sum, eager_reduce_mean => {
            for (input) |value| acc += value;
            if (op == eager_reduce_mean) acc /= @as(f32, @floatFromInt(input_len));
        },
        eager_reduce_max => {
            for (input) |value| acc = @max(acc, value);
        },
        eager_reduce_min => {
            for (input) |value| acc = @min(acc, value);
        },
        eager_reduce_prod => {
            for (input) |value| acc *= value;
        },
        else => unreachable,
    }
    output_ptr.?[0] = acc;
    return status(.ok);
}

export fn zgml_eager_dot_f32(
    lhs_ptr: ?[*]const f32,
    lhs_len: usize,
    rhs_ptr: ?[*]const f32,
    rhs_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
) c_int {
    if (lhs_ptr == null or rhs_ptr == null or output_ptr == null or lhs_len == 0) return status(.invalid_argument);
    if (lhs_len != rhs_len or output_len != 1) return status(.shape_mismatch);
    output_ptr.?[0] = dotF32(lhs_ptr.?[0..lhs_len], rhs_ptr.?[0..rhs_len]);
    return status(.ok);
}

export fn zgml_eager_conv2d_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    weights_ptr: ?[*]const f32,
    weights_len: usize,
    bias_ptr: ?[*]const f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
) c_int {
    if (input_ptr == null or
        weights_ptr == null or
        output_ptr == null or
        batch == 0 or
        in_channels == 0 or
        height == 0 or
        width == 0 or
        out_channels == 0 or
        kernel_h == 0 or
        kernel_w == 0 or
        stride_h == 0 or
        stride_w == 0 or
        dilation_h == 0 or
        dilation_w == 0 or
        out_h == 0 or
        out_w == 0) return status(.invalid_argument);

    if (checkedElementCount4(batch, in_channels, height, width) != input_len) return status(.shape_mismatch);
    if (checkedElementCount4(out_channels, in_channels, kernel_h, kernel_w) != weights_len) return status(.shape_mismatch);
    if (bias_len != 0 and (bias_ptr == null or bias_len != out_channels)) return status(.shape_mismatch);
    if (checkedElementCount4(batch, out_channels, out_h, out_w) != output_len) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const weights = weights_ptr.?[0..weights_len];
    const bias = if (bias_len == 0) null else bias_ptr.?[0..bias_len];
    const output = output_ptr.?[0..output_len];

    if (padding_h == 0 and padding_w == 0 and dilation_h == 1 and dilation_w == 1 and height >= kernel_h and width >= kernel_w) {
        const expected_out_h = (height - kernel_h) / stride_h + 1;
        const expected_out_w = (width - kernel_w) / stride_w + 1;
        if (out_h == expected_out_h and out_w == expected_out_w) {
            for (0..batch) |n| {
                for (0..out_channels) |oc| {
                    const output_channel_base = (n * out_channels + oc) * out_h * out_w;
                    for (0..out_h) |oh| {
                        const input_h_base = oh * stride_h;
                        const output_row_base = output_channel_base + oh * out_w;
                        for (0..out_w) |ow| {
                            var sum: f32 = if (bias) |b| b[oc] else 0;
                            const input_w_base = ow * stride_w;
                            for (0..in_channels) |ic| {
                                const input_channel_base = (n * in_channels + ic) * height * width;
                                const weight_channel_base = (oc * in_channels + ic) * kernel_h * kernel_w;
                                for (0..kernel_h) |ky| {
                                    const input_row_base = input_channel_base + (input_h_base + ky) * width + input_w_base;
                                    const weight_row_base = weight_channel_base + ky * kernel_w;
                                    for (0..kernel_w) |kx| {
                                        sum += input[input_row_base + kx] * weights[weight_row_base + kx];
                                    }
                                }
                            }
                            output[output_row_base + ow] = sum;
                        }
                    }
                }
            }
            return status(.ok);
        }
    }

    for (0..batch) |n| {
        for (0..out_channels) |oc| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    var sum: f32 = if (bias) |b| b[oc] else 0;
                    for (0..in_channels) |ic| {
                        for (0..kernel_h) |ky| {
                            const raw_h = oh * stride_h + ky * dilation_h;
                            if (raw_h < padding_h) continue;
                            const ih = raw_h - padding_h;
                            if (ih >= height) continue;
                            for (0..kernel_w) |kx| {
                                const raw_w = ow * stride_w + kx * dilation_w;
                                if (raw_w < padding_w) continue;
                                const iw = raw_w - padding_w;
                                if (iw >= width) continue;
                                const input_index = ((n * in_channels + ic) * height + ih) * width + iw;
                                const weight_index = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                                sum += input[input_index] * weights[weight_index];
                            }
                        }
                    }
                    output[((n * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    }
    return status(.ok);
}

export fn zgml_eager_pool2d_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    op: u32,
    ceil_mode: u32,
    count_include_pad: u32,
) c_int {
    if (input_ptr == null or
        output_ptr == null or
        batch == 0 or
        channels == 0 or
        height == 0 or
        width == 0 or
        kernel_h == 0 or
        kernel_w == 0 or
        stride_h == 0 or
        stride_w == 0 or
        dilation_h == 0 or
        dilation_w == 0 or
        out_h == 0 or
        out_w == 0) return status(.invalid_argument);
    if (op != eager_pool_max and op != eager_pool_avg) return status(.invalid_argument);
    if (ceil_mode != 0 and ceil_mode != 1) return status(.invalid_argument);
    if (count_include_pad != 0 and count_include_pad != 1) return status(.invalid_argument);
    if (op == eager_pool_avg and (dilation_h != 1 or dilation_w != 1)) return status(.invalid_argument);

    if (checkedElementCount4(batch, channels, height, width) != input_len) return status(.shape_mismatch);
    if (checkedElementCount4(batch, channels, out_h, out_w) != output_len) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const output = output_ptr.?[0..output_len];
    if (padding_h == 0 and padding_w == 0 and dilation_h == 1 and dilation_w == 1 and height >= kernel_h and width >= kernel_w) {
        const expected_out_h = (height - kernel_h) / stride_h + 1;
        const expected_out_w = (width - kernel_w) / stride_w + 1;
        if (out_h == expected_out_h and out_w == expected_out_w) {
            const avg_denominator: f32 = @floatFromInt(kernel_h * kernel_w);
            for (0..batch) |n| {
                for (0..channels) |c| {
                    const input_channel_base = (n * channels + c) * height * width;
                    const output_channel_base = (n * channels + c) * out_h * out_w;
                    for (0..out_h) |oh| {
                        const input_h_base = oh * stride_h;
                        const output_row_base = output_channel_base + oh * out_w;
                        for (0..out_w) |ow| {
                            var acc: f32 = if (op == eager_pool_max) -std.math.inf(f32) else 0;
                            const input_w_base = ow * stride_w;
                            for (0..kernel_h) |ky| {
                                const input_row_base = input_channel_base + (input_h_base + ky) * width + input_w_base;
                                for (0..kernel_w) |kx| {
                                    const value = input[input_row_base + kx];
                                    if (op == eager_pool_max) {
                                        acc = @max(acc, value);
                                    } else {
                                        acc += value;
                                    }
                                }
                            }
                            output[output_row_base + ow] = if (op == eager_pool_avg) acc / avg_denominator else acc;
                        }
                    }
                }
            }
            return status(.ok);
        }
    }

    for (0..batch) |n| {
        for (0..channels) |c| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    var acc: f32 = if (op == eager_pool_max) -std.math.inf(f32) else 0;
                    var count: usize = if (count_include_pad == 1) kernel_h * kernel_w else 0;
                    for (0..kernel_h) |ky| {
                        const raw_h = oh * stride_h + ky * dilation_h;
                        if (raw_h < padding_h) continue;
                        const ih = raw_h - padding_h;
                        if (ih >= height) continue;
                        for (0..kernel_w) |kx| {
                            const raw_w = ow * stride_w + kx * dilation_w;
                            if (raw_w < padding_w) continue;
                            const iw = raw_w - padding_w;
                            if (iw >= width) continue;
                            const input_index = ((n * channels + c) * height + ih) * width + iw;
                            if (op == eager_pool_max) {
                                acc = @max(acc, input[input_index]);
                            } else {
                                acc += input[input_index];
                                if (count_include_pad == 0) count += 1;
                            }
                        }
                    }
                    const out_index = ((n * channels + c) * out_h + oh) * out_w + ow;
                    output[out_index] = if (op == eager_pool_avg)
                        if (count == 0) 0 else acc / @as(f32, @floatFromInt(count))
                    else
                        acc;
                }
            }
        }
    }
    return status(.ok);
}

export fn zgml_eager_softmax_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    rows: usize,
    cols: usize,
    log_softmax: u32,
) c_int {
    if (input_ptr == null or output_ptr == null or rows == 0 or cols == 0) return status(.invalid_argument);
    if (log_softmax != 0 and log_softmax != 1) return status(.invalid_argument);
    const expected = checkedElementCount(rows, cols);
    if (expected != input_len or expected != output_len) return status(.shape_mismatch);

    writeSoftmaxRowsF32(input_ptr.?[0..input_len], output_ptr.?[0..output_len], rows, cols, log_softmax != 0);
    return status(.ok);
}

fn adamUpdateF32(
    param: []f32,
    m: []f32,
    v: []f32,
    index: usize,
    grad: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_weight_decay: bool,
) void {
    const g = if (decoupled_weight_decay) grad else grad + weight_decay * param[index];
    m[index] = beta1 * m[index] + (1.0 - beta1) * g;
    v[index] = beta2 * v[index] + (1.0 - beta2) * g * g;
    const m_hat = m[index] * bias_correction1;
    const v_hat = v[index] * bias_correction2;
    if (decoupled_weight_decay and weight_decay != 0) param[index] -= lr * weight_decay * param[index];
    param[index] -= lr * m_hat / (@sqrt(v_hat) + eps);
}

fn adamUpdateSliceF32(
    param: []f32,
    m: []f32,
    v: []f32,
    grad: []const f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_weight_decay: bool,
) void {
    const lanes = 8;
    const V = @Vector(lanes, f32);
    const lr_v: V = @splat(lr);
    const beta1_v: V = @splat(beta1);
    const one_minus_beta1_v: V = @splat(1.0 - beta1);
    const beta2_v: V = @splat(beta2);
    const one_minus_beta2_v: V = @splat(1.0 - beta2);
    const bias1_v: V = @splat(bias_correction1);
    const bias2_v: V = @splat(bias_correction2);
    const eps_v: V = @splat(eps);
    const weight_decay_v: V = @splat(weight_decay);

    var i: usize = 0;
    while (i + lanes <= param.len) : (i += lanes) {
        var p: V = param[i..][0..lanes].*;
        const grad_v: V = grad[i..][0..lanes].*;
        const g: V = if (decoupled_weight_decay) grad_v else grad_v + weight_decay_v * p;
        const next_m = beta1_v * m[i..][0..lanes].* + one_minus_beta1_v * g;
        const next_v = beta2_v * v[i..][0..lanes].* + one_minus_beta2_v * g * g;
        m[i..][0..lanes].* = next_m;
        v[i..][0..lanes].* = next_v;
        if (decoupled_weight_decay and weight_decay != 0) p -= lr_v * weight_decay_v * p;
        p -= lr_v * (next_m * bias1_v) / (@sqrt(next_v * bias2_v) + eps_v);
        param[i..][0..lanes].* = p;
    }

    while (i < param.len) : (i += 1) {
        adamUpdateF32(param, m, v, i, grad[i], lr, beta1, beta2, bias_correction1, bias_correction2, eps, weight_decay, decoupled_weight_decay);
    }
}

fn finiteAdamConfig(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) bool {
    return std.math.isFinite(lr) and
        std.math.isFinite(beta1) and
        std.math.isFinite(beta2) and
        std.math.isFinite(eps) and
        std.math.isFinite(weight_decay) and
        lr > 0 and
        beta1 >= 0 and beta1 < 1 and
        beta2 >= 0 and beta2 < 1 and
        eps > 0 and
        weight_decay >= 0;
}

fn finiteSgdConfig(lr: f32, weight_decay: f32) bool {
    return std.math.isFinite(lr) and
        std.math.isFinite(weight_decay) and
        lr > 0 and
        weight_decay >= 0;
}

export fn zgml_train_linear_mse_sgd_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    target_ptr: ?[*]const f32,
    target_len: usize,
    weight_ptr: ?[*]f32,
    weight_len: usize,
    bias_ptr: ?[*]f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    grad_weight_ptr: ?[*]f32,
    grad_weight_len: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    lr: f32,
    weight_decay: f32,
    out_loss: ?*f32,
) c_int {
    if (input_ptr == null or target_ptr == null or weight_ptr == null or bias_ptr == null or
        output_ptr == null or grad_weight_ptr == null or out_loss == null or
        batch == 0 or in_features == 0 or out_features == 0 or
        !finiteSgdConfig(lr, weight_decay)) return status(.invalid_argument);

    const input_count = checkedElementCount(batch, in_features) orelse return status(.shape_mismatch);
    const output_count = checkedElementCount(batch, out_features) orelse return status(.shape_mismatch);
    const weight_count = checkedElementCount(in_features, out_features) orelse return status(.shape_mismatch);
    if (input_len != input_count or target_len != output_count or
        weight_len != weight_count or bias_len != out_features or
        output_len != output_count or grad_weight_len != weight_count) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const target = target_ptr.?[0..target_len];
    const weight = weight_ptr.?[0..weight_len];
    const bias = bias_ptr.?[0..bias_len];
    const output = output_ptr.?[0..output_len];
    const grad_weight = grad_weight_ptr.?[0..grad_weight_len];

    forward.blasSgemm(output, input, weight, batch, out_features, in_features, in_features, 1, out_features, 1, 0, 0, 0, out_features);
    addBiasRowsF32(output, bias, batch, out_features);

    var total_loss: f32 = 0;
    const inv_count = 1.0 / @as(f32, @floatFromInt(output_count));
    const grad_scale = 2.0 * inv_count;
    for (output, 0..) |*value, i| {
        const diff = value.* - target[i];
        total_loss += diff * diff;
        value.* = diff * grad_scale;
    }
    out_loss.?.* = total_loss * inv_count;

    forward.blasSgemm(grad_weight, input, output, in_features, out_features, batch, 1, in_features, out_features, 1, 0, 0, 0, out_features);
    for (weight, 0..) |*param, i| {
        param.* -= lr * (grad_weight[i] + weight_decay * param.*);
    }
    for (0..out_features) |c| {
        var grad: f32 = 0;
        for (0..batch) |row| grad += output[row * out_features + c];
        bias[c] -= lr * (grad + weight_decay * bias[c]);
    }

    return status(.ok);
}

export fn zgml_train_linear_mse_sgd_f32_bulk(
    dataset_input_ptr: ?[*]const f32,
    dataset_input_len: usize,
    dataset_target_ptr: ?[*]const f32,
    dataset_target_len: usize,
    indices_ptr: ?[*]const u32,
    indices_len: usize,
    batch_input_ptr: ?[*]f32,
    batch_input_len: usize,
    batch_target_ptr: ?[*]f32,
    batch_target_len: usize,
    weight_ptr: ?[*]f32,
    weight_len: usize,
    bias_ptr: ?[*]f32,
    bias_len: usize,
    output_ptr: ?[*]f32,
    output_len: usize,
    grad_weight_ptr: ?[*]f32,
    grad_weight_len: usize,
    sample_count: usize,
    batch: usize,
    in_features: usize,
    out_features: usize,
    epochs: usize,
    lr: f32,
    weight_decay: f32,
    out_loss: ?*f32,
    out_steps: ?*usize,
) c_int {
    if (dataset_input_ptr == null or dataset_target_ptr == null or indices_ptr == null or
        batch_input_ptr == null or batch_target_ptr == null or out_loss == null or out_steps == null or
        sample_count == 0 or batch == 0 or in_features == 0 or out_features == 0 or epochs == 0 or
        !finiteSgdConfig(lr, weight_decay)) return status(.invalid_argument);

    const dataset_count = dataset_target_len / out_features;
    const dataset_input_count = checkedElementCount(dataset_count, in_features) orelse return status(.shape_mismatch);
    const dataset_target_count = checkedElementCount(dataset_count, out_features) orelse return status(.shape_mismatch);
    const batch_input_count = checkedElementCount(batch, in_features) orelse return status(.shape_mismatch);
    const batch_target_count = checkedElementCount(batch, out_features) orelse return status(.shape_mismatch);
    if (dataset_count == 0 or dataset_target_len != dataset_target_count or dataset_input_len != dataset_input_count or
        indices_len != sample_count or batch_input_len != batch_input_count or batch_target_len != batch_target_count or
        sample_count % batch != 0) return status(.shape_mismatch);

    const dataset_input = dataset_input_ptr.?[0..dataset_input_len];
    const dataset_target = dataset_target_ptr.?[0..dataset_target_len];
    const indices = indices_ptr.?[0..indices_len];
    const batch_input = batch_input_ptr.?[0..batch_input_len];
    const batch_target = batch_target_ptr.?[0..batch_target_len];

    var total_steps: usize = 0;
    var last_loss: f32 = 0;
    const batches_per_epoch = sample_count / batch;
    for (0..epochs) |epoch| {
        _ = epoch;
        for (0..batches_per_epoch) |batch_index| {
            const batch_base = batch_index * batch;
            for (0..batch) |row| {
                const raw_index = indices[batch_base + row];
                const sample_index: usize = @intCast(raw_index);
                if (sample_index >= dataset_count) return status(.shape_mismatch);
                @memcpy(
                    batch_input[row * in_features ..][0..in_features],
                    dataset_input[sample_index * in_features ..][0..in_features],
                );
                @memcpy(
                    batch_target[row * out_features ..][0..out_features],
                    dataset_target[sample_index * out_features ..][0..out_features],
                );
            }
            const step_status = zgml_train_linear_mse_sgd_f32(
                batch_input_ptr,
                batch_input_len,
                batch_target_ptr,
                batch_target_len,
                weight_ptr,
                weight_len,
                bias_ptr,
                bias_len,
                output_ptr,
                output_len,
                grad_weight_ptr,
                grad_weight_len,
                batch,
                in_features,
                out_features,
                lr,
                weight_decay,
                &last_loss,
            );
            if (step_status != status(.ok)) return step_status;
            total_steps += 1;
        }
    }

    out_loss.?.* = last_loss;
    out_steps.?.* = total_steps;
    return status(.ok);
}

fn trainMlpReluCrossEntropyAdamLikeF32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    target_ptr: ?[*]const u32,
    target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_weight_decay: bool,
    out_loss: ?*f32,
    out_correct: ?*usize,
) c_int {
    if (input_ptr == null or target_ptr == null or
        w1_ptr == null or b1_ptr == null or w2_ptr == null or b2_ptr == null or
        mw1_ptr == null or vw1_ptr == null or mb1_ptr == null or vb1_ptr == null or
        mw2_ptr == null or vw2_ptr == null or mb2_ptr == null or vb2_ptr == null or
        hidden_ptr == null or logits_ptr == null or grad_hidden_ptr == null or grad_w1_ptr == null or grad_w2_ptr == null or
        out_loss == null or out_correct == null or
        batch == 0 or in_features == 0 or hidden_features == 0 or classes == 0 or t == 0 or
        !finiteAdamConfig(lr, beta1, beta2, eps, weight_decay)) return status(.invalid_argument);

    const input_count = checkedElementCount(batch, in_features) orelse return status(.shape_mismatch);
    const hidden_count = checkedElementCount(batch, hidden_features) orelse return status(.shape_mismatch);
    const logits_count = checkedElementCount(batch, classes) orelse return status(.shape_mismatch);
    const w1_count = checkedElementCount(in_features, hidden_features) orelse return status(.shape_mismatch);
    const w2_count = checkedElementCount(hidden_features, classes) orelse return status(.shape_mismatch);
    if (input_len != input_count or target_len != batch or
        w1_len != w1_count or b1_len != hidden_features or
        w2_len != w2_count or b2_len != classes or
        mw1_len != w1_count or vw1_len != w1_count or
        mb1_len != hidden_features or vb1_len != hidden_features or
        mw2_len != w2_count or vw2_len != w2_count or
        mb2_len != classes or vb2_len != classes or
        hidden_len != hidden_count or logits_len != logits_count or grad_hidden_len != hidden_count or
        grad_w1_len != w1_count or grad_w2_len != w2_count) return status(.shape_mismatch);

    const input = input_ptr.?[0..input_len];
    const targets = target_ptr.?[0..target_len];
    const w1 = w1_ptr.?[0..w1_len];
    const b1 = b1_ptr.?[0..b1_len];
    const w2 = w2_ptr.?[0..w2_len];
    const b2 = b2_ptr.?[0..b2_len];
    const mw1 = mw1_ptr.?[0..mw1_len];
    const vw1 = vw1_ptr.?[0..vw1_len];
    const mb1 = mb1_ptr.?[0..mb1_len];
    const vb1 = vb1_ptr.?[0..vb1_len];
    const mw2 = mw2_ptr.?[0..mw2_len];
    const vw2 = vw2_ptr.?[0..vw2_len];
    const mb2 = mb2_ptr.?[0..mb2_len];
    const vb2 = vb2_ptr.?[0..vb2_len];
    const hidden = hidden_ptr.?[0..hidden_len];
    const logits = logits_ptr.?[0..logits_len];
    const grad_hidden = grad_hidden_ptr.?[0..grad_hidden_len];
    const grad_w1 = grad_w1_ptr.?[0..grad_w1_len];
    const grad_w2 = grad_w2_ptr.?[0..grad_w2_len];

    forward.blasSgemm(hidden, input, w1, batch, hidden_features, in_features, in_features, 1, hidden_features, 1, 0, 0, 0, hidden_features);
    addBiasRowsF32(hidden, b1, batch, hidden_features);
    applyActivationF32(hidden, module_activation_relu) catch unreachable;

    forward.blasSgemm(logits, hidden, w2, batch, classes, hidden_features, hidden_features, 1, classes, 1, 0, 0, 0, classes);
    addBiasRowsF32(logits, b2, batch, classes);

    var total_loss: f32 = 0;
    var correct: usize = 0;
    const inv_batch = 1.0 / @as(f32, @floatFromInt(batch));
    for (0..batch) |row| {
        const target = targets[row];
        if (target >= classes) return status(.shape_mismatch);
        const target_index: usize = @intCast(target);
        const logits_row = logits[row * classes ..][0..classes];
        var max_logit = logits_row[0];
        var predicted: usize = 0;
        for (1..classes) |c| {
            if (logits_row[c] > max_logit) {
                max_logit = logits_row[c];
                predicted = c;
            }
        }
        if (predicted == target_index) correct += 1;
        var denom: f32 = 0;
        for (0..classes) |c| denom += @exp(logits_row[c] - max_logit);
        total_loss += -(logits_row[target_index] - max_logit - @log(denom));
        for (0..classes) |c| {
            const prob = @exp(logits_row[c] - max_logit) / denom;
            logits_row[c] = (prob - if (c == target_index) @as(f32, 1) else @as(f32, 0)) * inv_batch;
        }
    }
    out_loss.?.* = total_loss * inv_batch;
    out_correct.?.* = correct;

    const bias_correction1 = 1.0 / (1.0 - std.math.pow(f32, beta1, @floatFromInt(t)));
    const bias_correction2 = 1.0 / (1.0 - std.math.pow(f32, beta2, @floatFromInt(t)));

    forward.blasSgemm(grad_hidden, logits, w2, batch, hidden_features, classes, classes, 1, 1, classes, 0, 0, 0, hidden_features);
    for (0..batch) |row| {
        const hidden_row = hidden[row * hidden_features ..][0..hidden_features];
        const grad_hidden_row = grad_hidden[row * hidden_features ..][0..hidden_features];
        for (0..hidden_features) |h| grad_hidden_row[h] = if (hidden_row[h] > 0) grad_hidden_row[h] else 0;
    }

    forward.blasSgemm(grad_w2, hidden, logits, hidden_features, classes, batch, 1, hidden_features, classes, 1, 0, 0, 0, classes);
    adamUpdateSliceF32(w2, mw2, vw2, grad_w2, lr, beta1, beta2, bias_correction1, bias_correction2, eps, weight_decay, decoupled_weight_decay);
    for (0..classes) |c| {
        var grad: f32 = 0;
        for (0..batch) |row| grad += logits[row * classes + c];
        adamUpdateF32(b2, mb2, vb2, c, grad, lr, beta1, beta2, bias_correction1, bias_correction2, eps, weight_decay, decoupled_weight_decay);
    }

    forward.blasSgemm(grad_w1, input, grad_hidden, in_features, hidden_features, batch, 1, in_features, hidden_features, 1, 0, 0, 0, hidden_features);
    adamUpdateSliceF32(w1, mw1, vw1, grad_w1, lr, beta1, beta2, bias_correction1, bias_correction2, eps, weight_decay, decoupled_weight_decay);
    for (0..hidden_features) |h| {
        var grad: f32 = 0;
        for (0..batch) |row| grad += grad_hidden[row * hidden_features + h];
        adamUpdateF32(b1, mb1, vb1, h, grad, lr, beta1, beta2, bias_correction1, bias_correction2, eps, weight_decay, decoupled_weight_decay);
    }

    return status(.ok);
}

export fn zgml_train_mlp_relu_cross_entropy_adam_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    target_ptr: ?[*]const u32,
    target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    out_loss: ?*f32,
    out_correct: ?*usize,
) c_int {
    return trainMlpReluCrossEntropyAdamLikeF32(
        input_ptr,
        input_len,
        target_ptr,
        target_len,
        w1_ptr,
        w1_len,
        b1_ptr,
        b1_len,
        w2_ptr,
        w2_len,
        b2_ptr,
        b2_len,
        mw1_ptr,
        mw1_len,
        vw1_ptr,
        vw1_len,
        mb1_ptr,
        mb1_len,
        vb1_ptr,
        vb1_len,
        mw2_ptr,
        mw2_len,
        vw2_ptr,
        vw2_len,
        mb2_ptr,
        mb2_len,
        vb2_ptr,
        vb2_len,
        hidden_ptr,
        hidden_len,
        logits_ptr,
        logits_len,
        grad_hidden_ptr,
        grad_hidden_len,
        grad_w1_ptr,
        grad_w1_len,
        grad_w2_ptr,
        grad_w2_len,
        batch,
        in_features,
        hidden_features,
        classes,
        t,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        false,
        out_loss,
        out_correct,
    );
}

export fn zgml_train_mlp_relu_cross_entropy_adamw_f32(
    input_ptr: ?[*]const f32,
    input_len: usize,
    target_ptr: ?[*]const u32,
    target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    out_loss: ?*f32,
    out_correct: ?*usize,
) c_int {
    return trainMlpReluCrossEntropyAdamLikeF32(
        input_ptr,
        input_len,
        target_ptr,
        target_len,
        w1_ptr,
        w1_len,
        b1_ptr,
        b1_len,
        w2_ptr,
        w2_len,
        b2_ptr,
        b2_len,
        mw1_ptr,
        mw1_len,
        vw1_ptr,
        vw1_len,
        mb1_ptr,
        mb1_len,
        vb1_ptr,
        vb1_len,
        mw2_ptr,
        mw2_len,
        vw2_ptr,
        vw2_len,
        mb2_ptr,
        mb2_len,
        vb2_ptr,
        vb2_len,
        hidden_ptr,
        hidden_len,
        logits_ptr,
        logits_len,
        grad_hidden_ptr,
        grad_hidden_len,
        grad_w1_ptr,
        grad_w1_len,
        grad_w2_ptr,
        grad_w2_len,
        batch,
        in_features,
        hidden_features,
        classes,
        t,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        true,
        out_loss,
        out_correct,
    );
}

fn trainMlpReluCrossEntropyAdamLikeBulkF32(
    dataset_input_ptr: ?[*]const f32,
    dataset_input_len: usize,
    dataset_target_ptr: ?[*]const u32,
    dataset_target_len: usize,
    indices_ptr: ?[*]const u32,
    indices_len: usize,
    batch_input_ptr: ?[*]f32,
    batch_input_len: usize,
    batch_target_ptr: ?[*]u32,
    batch_target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    sample_count: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    epochs: usize,
    start_step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_weight_decay: bool,
    out_loss: ?*f32,
    out_correct: ?*usize,
    out_steps: ?*usize,
) c_int {
    if (dataset_input_ptr == null or dataset_target_ptr == null or indices_ptr == null or
        batch_input_ptr == null or batch_target_ptr == null or out_loss == null or
        out_correct == null or out_steps == null or sample_count == 0 or batch == 0 or
        in_features == 0 or hidden_features == 0 or classes == 0 or epochs == 0 or
        !finiteAdamConfig(lr, beta1, beta2, eps, weight_decay)) return status(.invalid_argument);

    // sample_count is the number of indexed rows to train; dataset_count bounds the backing tensors.
    const dataset_count = dataset_target_len;
    const dataset_input_count = checkedElementCount(dataset_count, in_features) orelse return status(.shape_mismatch);
    const batch_input_count = checkedElementCount(batch, in_features) orelse return status(.shape_mismatch);
    if (dataset_input_len != dataset_input_count or dataset_count == 0 or
        indices_len != sample_count or batch_input_len != batch_input_count or
        batch_target_len != batch or sample_count % batch != 0) return status(.shape_mismatch);

    const dataset_input = dataset_input_ptr.?[0..dataset_input_len];
    const dataset_target = dataset_target_ptr.?[0..dataset_target_len];
    const indices = indices_ptr.?[0..indices_len];
    const batch_input = batch_input_ptr.?[0..batch_input_len];
    const batch_target = batch_target_ptr.?[0..batch_target_len];

    var total_steps: usize = 0;
    var last_loss: f32 = 0;
    var last_correct: usize = 0;
    const batches_per_epoch = sample_count / batch;
    for (0..epochs) |epoch| {
        _ = epoch;
        for (0..batches_per_epoch) |batch_index| {
            const batch_base = batch_index * batch;
            for (0..batch) |row| {
                const raw_index = indices[batch_base + row];
                const sample_index: usize = @intCast(raw_index);
                if (sample_index >= dataset_count) return status(.shape_mismatch);
                const src = dataset_input[sample_index * in_features ..][0..in_features];
                const dst = batch_input[row * in_features ..][0..in_features];
                @memcpy(dst, src);
                batch_target[row] = dataset_target[sample_index];
            }
            const step_status = trainMlpReluCrossEntropyAdamLikeF32(
                batch_input_ptr,
                batch_input_len,
                batch_target_ptr,
                batch_target_len,
                w1_ptr,
                w1_len,
                b1_ptr,
                b1_len,
                w2_ptr,
                w2_len,
                b2_ptr,
                b2_len,
                mw1_ptr,
                mw1_len,
                vw1_ptr,
                vw1_len,
                mb1_ptr,
                mb1_len,
                vb1_ptr,
                vb1_len,
                mw2_ptr,
                mw2_len,
                vw2_ptr,
                vw2_len,
                mb2_ptr,
                mb2_len,
                vb2_ptr,
                vb2_len,
                hidden_ptr,
                hidden_len,
                logits_ptr,
                logits_len,
                grad_hidden_ptr,
                grad_hidden_len,
                grad_w1_ptr,
                grad_w1_len,
                grad_w2_ptr,
                grad_w2_len,
                batch,
                in_features,
                hidden_features,
                classes,
                start_step + total_steps + 1,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                decoupled_weight_decay,
                &last_loss,
                &last_correct,
            );
            if (step_status != status(.ok)) return step_status;
            total_steps += 1;
        }
    }

    out_loss.?.* = last_loss;
    out_correct.?.* = last_correct;
    out_steps.?.* = total_steps;
    return status(.ok);
}

export fn zgml_train_mlp_relu_cross_entropy_adam_f32_bulk(
    dataset_input_ptr: ?[*]const f32,
    dataset_input_len: usize,
    dataset_target_ptr: ?[*]const u32,
    dataset_target_len: usize,
    indices_ptr: ?[*]const u32,
    indices_len: usize,
    batch_input_ptr: ?[*]f32,
    batch_input_len: usize,
    batch_target_ptr: ?[*]u32,
    batch_target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    sample_count: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    epochs: usize,
    start_step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    out_loss: ?*f32,
    out_correct: ?*usize,
    out_steps: ?*usize,
) c_int {
    return trainMlpReluCrossEntropyAdamLikeBulkF32(
        dataset_input_ptr,
        dataset_input_len,
        dataset_target_ptr,
        dataset_target_len,
        indices_ptr,
        indices_len,
        batch_input_ptr,
        batch_input_len,
        batch_target_ptr,
        batch_target_len,
        w1_ptr,
        w1_len,
        b1_ptr,
        b1_len,
        w2_ptr,
        w2_len,
        b2_ptr,
        b2_len,
        mw1_ptr,
        mw1_len,
        vw1_ptr,
        vw1_len,
        mb1_ptr,
        mb1_len,
        vb1_ptr,
        vb1_len,
        mw2_ptr,
        mw2_len,
        vw2_ptr,
        vw2_len,
        mb2_ptr,
        mb2_len,
        vb2_ptr,
        vb2_len,
        hidden_ptr,
        hidden_len,
        logits_ptr,
        logits_len,
        grad_hidden_ptr,
        grad_hidden_len,
        grad_w1_ptr,
        grad_w1_len,
        grad_w2_ptr,
        grad_w2_len,
        sample_count,
        batch,
        in_features,
        hidden_features,
        classes,
        epochs,
        start_step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        false,
        out_loss,
        out_correct,
        out_steps,
    );
}

export fn zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk(
    dataset_input_ptr: ?[*]const f32,
    dataset_input_len: usize,
    dataset_target_ptr: ?[*]const u32,
    dataset_target_len: usize,
    indices_ptr: ?[*]const u32,
    indices_len: usize,
    batch_input_ptr: ?[*]f32,
    batch_input_len: usize,
    batch_target_ptr: ?[*]u32,
    batch_target_len: usize,
    w1_ptr: ?[*]f32,
    w1_len: usize,
    b1_ptr: ?[*]f32,
    b1_len: usize,
    w2_ptr: ?[*]f32,
    w2_len: usize,
    b2_ptr: ?[*]f32,
    b2_len: usize,
    mw1_ptr: ?[*]f32,
    mw1_len: usize,
    vw1_ptr: ?[*]f32,
    vw1_len: usize,
    mb1_ptr: ?[*]f32,
    mb1_len: usize,
    vb1_ptr: ?[*]f32,
    vb1_len: usize,
    mw2_ptr: ?[*]f32,
    mw2_len: usize,
    vw2_ptr: ?[*]f32,
    vw2_len: usize,
    mb2_ptr: ?[*]f32,
    mb2_len: usize,
    vb2_ptr: ?[*]f32,
    vb2_len: usize,
    hidden_ptr: ?[*]f32,
    hidden_len: usize,
    logits_ptr: ?[*]f32,
    logits_len: usize,
    grad_hidden_ptr: ?[*]f32,
    grad_hidden_len: usize,
    grad_w1_ptr: ?[*]f32,
    grad_w1_len: usize,
    grad_w2_ptr: ?[*]f32,
    grad_w2_len: usize,
    sample_count: usize,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    classes: usize,
    epochs: usize,
    start_step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    out_loss: ?*f32,
    out_correct: ?*usize,
    out_steps: ?*usize,
) c_int {
    return trainMlpReluCrossEntropyAdamLikeBulkF32(
        dataset_input_ptr,
        dataset_input_len,
        dataset_target_ptr,
        dataset_target_len,
        indices_ptr,
        indices_len,
        batch_input_ptr,
        batch_input_len,
        batch_target_ptr,
        batch_target_len,
        w1_ptr,
        w1_len,
        b1_ptr,
        b1_len,
        w2_ptr,
        w2_len,
        b2_ptr,
        b2_len,
        mw1_ptr,
        mw1_len,
        vw1_ptr,
        vw1_len,
        mb1_ptr,
        mb1_len,
        vb1_ptr,
        vb1_len,
        mw2_ptr,
        mw2_len,
        vw2_ptr,
        vw2_len,
        mb2_ptr,
        mb2_len,
        vb2_ptr,
        vb2_len,
        hidden_ptr,
        hidden_len,
        logits_ptr,
        logits_len,
        grad_hidden_ptr,
        grad_hidden_len,
        grad_w1_ptr,
        grad_w1_len,
        grad_w2_ptr,
        grad_w2_len,
        sample_count,
        batch,
        in_features,
        hidden_features,
        classes,
        epochs,
        start_step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        true,
        out_loss,
        out_correct,
        out_steps,
    );
}

fn clearModel(out: ?*?*zgml_model) void {
    if (out) |p| p.* = null;
}

fn clearProgram(out: ?*?*zgml_program) void {
    if (out) |p| p.* = null;
}

fn clearSession(out: ?*?*zgml_session) void {
    if (out) |p| p.* = null;
}

fn clearBuffer(out: ?*?*zgml_buffer) void {
    if (out) |p| p.* = null;
}

fn modelHandle(model: ?*zgml_model) ?*ModelHandle {
    const ptr = model orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn programHandle(program: ?*zgml_program) ?*ProgramHandle {
    const ptr = program orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn sessionHandle(session: ?*zgml_session) ?*SessionHandle {
    const ptr = session orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn bufferHandle(buffer: ?*zgml_buffer) ?*BufferHandle {
    const ptr = buffer orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn modelKindForHandle(m: *const ModelHandle) u32 {
    return switch (m.data) {
        .tiny_linear => tiny_linear_kind,
        .tiny_mlp => tiny_mlp_kind,
        .tiny_llama => tiny_llama_kind,
        .tiny_llama_2layer => tiny_llama_2layer_kind,
        .smollm_135m => smollm_135m_kind,
    };
}

fn programModelKindForHandle(p: *const ProgramHandle) u32 {
    return switch (p.data) {
        .tiny_linear => tiny_linear_kind,
        .tiny_mlp => tiny_mlp_kind,
        .module => module_kind,
        .tiny_llama => tiny_llama_kind,
        .tiny_llama_2layer => tiny_llama_2layer_kind,
        .smollm_135m => smollm_135m_kind,
    };
}

fn programAcceptsModel(p: *const ProgramHandle, m: *const ModelHandle) bool {
    return switch (p.data) {
        .tiny_linear => false,
        .tiny_mlp => false,
        .module => false,
        .tiny_llama => switch (m.data) {
            .tiny_llama => true,
            .tiny_linear, .tiny_mlp, .tiny_llama_2layer, .smollm_135m => false,
        },
        .tiny_llama_2layer => switch (m.data) {
            .tiny_llama_2layer => true,
            .tiny_linear, .tiny_mlp, .tiny_llama, .smollm_135m => false,
        },
        .smollm_135m => switch (m.data) {
            .smollm_135m => true,
            .tiny_linear, .tiny_mlp, .tiny_llama, .tiny_llama_2layer => false,
        },
    };
}

fn bufferConstF32Ptr(buffer: ?*zgml_buffer, element_count: usize) !?[*]const f32 {
    const handle = bufferHandle(buffer) orelse {
        if (element_count == 0) return null;
        return error.ShapeMismatch;
    };
    const data = handle.hostData() orelse return error.UnsupportedResourceBinding;
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (byte_len > data.len) return error.ShapeMismatch;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    return @ptrCast(@alignCast(data.ptr));
}

fn bufferMutF32Ptr(buffer: ?*zgml_buffer, element_count: usize) !?[*]f32 {
    const handle = bufferHandle(buffer) orelse {
        if (element_count == 0) return null;
        return error.ShapeMismatch;
    };
    const data = handle.hostData() orelse return error.UnsupportedResourceBinding;
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (byte_len > data.len) return error.ShapeMismatch;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    return @ptrCast(@alignCast(data.ptr));
}

fn bufferF32Binding(tensor: *const TensorF32, buffer: ?*zgml_buffer, element_count: usize) !?DeviceF32.TensorBinding {
    const handle = bufferHandle(buffer) orelse {
        if (element_count == 0) return null;
        return error.ShapeMismatch;
    };
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (handle.externalResource()) |resource| {
        if (byte_len > handle.byteLen()) return error.ShapeMismatch;
        return DeviceF32.TensorBinding.externalResource(tensor, resource, element_count);
    }
    const data = handle.hostData().?;
    if (byte_len > data.len) return error.ShapeMismatch;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    return DeviceF32.TensorBinding.host(tensor, @ptrCast(@alignCast(data.ptr)), element_count);
}

fn externalResourceElementWindow(resource: backend_mod.ProgramIO.ExternalResource, element_offset: usize, element_count: usize) !backend_mod.ProgramIO.ExternalResource {
    const byte_delta = std.math.mul(usize, element_offset, @sizeOf(f32)) catch return error.ShapeMismatch;
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    const absolute_offset = std.math.add(usize, resource.byte_offset, byte_delta) catch return error.ShapeMismatch;
    if (absolute_offset > resource.byte_len or byte_len > @as(usize, resource.byte_len) - absolute_offset) return error.ShapeMismatch;
    var out = resource;
    out.byte_offset = std.math.cast(u32, absolute_offset) orelse return error.ShapeMismatch;
    return out;
}

fn aggregateBufferF32Binding(
    tensor: *const TensorF32,
    buffer: ?*zgml_buffer,
    aggregate_element_count: usize,
    element_offset: usize,
    element_count: usize,
) !DeviceF32.TensorBinding {
    if (element_offset > aggregate_element_count or element_count > aggregate_element_count - element_offset) return error.ShapeMismatch;
    const handle = bufferHandle(buffer) orelse return error.ShapeMismatch;
    const aggregate_byte_len = std.math.mul(usize, aggregate_element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (handle.externalResource()) |resource| {
        if (aggregate_byte_len > handle.byteLen()) return error.ShapeMismatch;
        return DeviceF32.TensorBinding.externalResource(
            tensor,
            try externalResourceElementWindow(resource, element_offset, element_count),
            element_count,
        );
    }

    const data = handle.hostData().?;
    if (aggregate_byte_len > data.len) return error.ShapeMismatch;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    const f32_data: [*]f32 = @ptrCast(@alignCast(data.ptr));
    return DeviceF32.TensorBinding.host(tensor, f32_data + element_offset, element_count);
}

fn bufferF32HostSlice(buffer: ?*zgml_buffer, element_count: usize) ![]f32 {
    const handle = bufferHandle(buffer) orelse {
        if (element_count == 0) return @as([]f32, &.{});
        return error.ShapeMismatch;
    };
    if (handle.externalResource() != null) return @as([]f32, &.{});
    const data = handle.hostData().?;
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (byte_len > data.len) return error.ShapeMismatch;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    const f32_data: [*]f32 = @ptrCast(@alignCast(data.ptr));
    return f32_data[0..element_count];
}

fn bufferIsExternalResource(buffer: ?*zgml_buffer) bool {
    const handle = bufferHandle(buffer) orelse return false;
    return handle.external != null;
}

fn bufferBindDescHasExternalResource(desc: *const zgml_buffer_bind_desc) bool {
    return bufferIsExternalResource(desc.weights) or
        bufferIsExternalResource(desc.bias) or
        bufferIsExternalResource(desc.input) or
        bufferIsExternalResource(desc.output);
}

fn checkedWeightLen(input_len: usize, output_len: usize) ?usize {
    if (input_len == 0 or output_len == 0) return null;
    return std.math.mul(usize, input_len, output_len) catch null;
}

fn tinyMlpActivationFromId(id: u32) ?TinyMlpActivation {
    return switch (id) {
        module_activation_relu => .relu,
        module_activation_gelu => .gelu,
        module_activation_silu => .silu,
        module_activation_sigmoid => .sigmoid,
        else => null,
    };
}

fn checkedTinyMlpWeightLens(input_len: usize, hidden_len: usize, output_len: usize) ?struct { w0: usize, w1: usize, total: usize } {
    if (input_len == 0 or hidden_len == 0 or output_len == 0) return null;
    const w0 = std.math.mul(usize, input_len, hidden_len) catch return null;
    const w1 = std.math.mul(usize, hidden_len, output_len) catch return null;
    const total = std.math.add(usize, w0, w1) catch return null;
    return .{ .w0 = w0, .w1 = w1, .total = total };
}

fn checkedTinyMlpBiasLen(hidden_len: usize, output_len: usize) ?usize {
    if (hidden_len == 0 or output_len == 0) return null;
    return std.math.add(usize, hidden_len, output_len) catch null;
}

fn tinyMlpActivate(x: *TensorF32, activation: TinyMlpActivation) *TensorF32 {
    return switch (activation) {
        .relu => x.relu(),
        .gelu => x.gelu(),
        .silu => x.silu(),
        .sigmoid => x.sigmoid(),
    };
}

fn moduleAddParam(
    graph_alloc: std.mem.Allocator,
    tensors: *std.ArrayListUnmanaged(*TensorF32),
    params: *std.ArrayListUnmanaged(ModuleParamBinding),
    kind: ModuleParamKind,
    shape: []const usize,
    offset: *usize,
) !*TensorF32 {
    const tensor = try TensorF32.init(graph_alloc, shape);
    @memset(tensor.data, 0);
    try tensors.append(alloc, tensor);
    try params.append(alloc, .{
        .tensor = tensor,
        .kind = kind,
        .offset = offset.*,
        .len = tensor.nElems(),
    });
    offset.* = try std.math.add(usize, offset.*, tensor.nElems());
    return tensor;
}

fn moduleActivate(x: *TensorF32, activation: u32) !*TensorF32 {
    return switch (activation) {
        module_activation_relu,
        module_activation_gelu,
        module_activation_silu,
        module_activation_sigmoid,
        => tinyMlpActivate(x, tinyMlpActivationFromId(activation).?),
        module_activation_exp => x.exp(),
        module_activation_log => x.log(),
        module_activation_neg => x.neg(),
        module_activation_recip => x.recip(),
        module_activation_abs => x.abs(),
        module_activation_sqrt => x.sqrt(),
        module_activation_square => x.sqr(),
        module_activation_sgn => x.sgn(),
        module_activation_step => x.step(),
        module_activation_tanh => x.tanh(),
        else => error.InvalidArgument,
    };
}

fn moduleActivationIdFromField(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) return error.InvalidArgument;
    return @intCast(value);
}

fn moduleActivateChain(x: *TensorF32, op: zgml_module_op_desc) !*TensorF32 {
    const count = op.flags;
    if (count < 1 or count > 4) return error.InvalidArgument;
    if (op.reserved != 0 or op.eps != 0) return error.InvalidArgument;
    if (count == 1 and (op.a != 0 or op.b != 0 or op.c != 0)) return error.InvalidArgument;
    if (count == 2 and (op.b != 0 or op.c != 0)) return error.InvalidArgument;
    if (count == 3 and op.c != 0) return error.InvalidArgument;

    var current = x;
    current = try moduleActivate(current, op.activation);
    if (count >= 2) current = try moduleActivate(current, try moduleActivationIdFromField(op.a));
    if (count >= 3) current = try moduleActivate(current, try moduleActivationIdFromField(op.b));
    if (count >= 4) current = try moduleActivate(current, try moduleActivationIdFromField(op.c));
    return current;
}

fn moduleFeatureNormReduceShape(x: *const TensorF32, features: usize) ![tensor_mod.max_dims]usize {
    if (features == 0 or x.n_dims == 0 or x.ne[0] != features) return error.ShapeMismatch;
    var reduce_ne = x.ne;
    reduce_ne[0] = 1;
    return reduce_ne;
}

const ModuleShapeSpec = struct {
    rank: usize,
    ne: [tensor_mod.max_dims]usize,
    feature_len: usize,
};

fn moduleShapeSpecFromRowMajor(rank: usize, first: usize, second: usize, third: usize) !ModuleShapeSpec {
    var ne = [_]usize{1} ** tensor_mod.max_dims;
    switch (rank) {
        1 => {
            if (first == 0 or second != 0 or third != 0) return error.ShapeMismatch;
            ne[0] = first;
        },
        2 => {
            if (first == 0 or second == 0 or third != 0) return error.ShapeMismatch;
            ne[0] = second;
            ne[1] = first;
        },
        3 => {
            if (first == 0 or second == 0 or third == 0) return error.ShapeMismatch;
            ne[0] = third;
            ne[1] = second;
            ne[2] = first;
        },
        else => return error.ShapeMismatch,
    }
    return .{
        .rank = rank,
        .ne = ne,
        .feature_len = ne[0],
    };
}

fn moduleRowMajorDim(tensor: *const TensorF32, rank: usize, axis: usize) !usize {
    return switch (rank) {
        1 => if (axis == 0) tensor.ne[0] else error.ShapeMismatch,
        2 => switch (axis) {
            0 => tensor.ne[1],
            1 => tensor.ne[0],
            else => error.ShapeMismatch,
        },
        3 => switch (axis) {
            0 => tensor.ne[2],
            1 => tensor.ne[1],
            2 => tensor.ne[0],
            else => error.ShapeMismatch,
        },
        else => error.ShapeMismatch,
    };
}

fn moduleNativeShapeDim(spec: ModuleShapeSpec, axis: usize) !usize {
    return switch (spec.rank) {
        1 => if (axis == 0) spec.ne[0] else error.ShapeMismatch,
        2 => switch (axis) {
            0 => spec.ne[1],
            1 => spec.ne[0],
            else => error.ShapeMismatch,
        },
        3 => switch (axis) {
            0 => spec.ne[2],
            1 => spec.ne[1],
            2 => spec.ne[0],
            else => error.ShapeMismatch,
        },
        else => error.ShapeMismatch,
    };
}

fn moduleReshapeSpec(op: zgml_module_op_desc, current: *const TensorF32) !ModuleShapeSpec {
    if (op.activation != 0 or op.flags != 0 or op.eps != 0) return error.InvalidArgument;
    const spec = try moduleShapeSpecFromRowMajor(op.a, op.b, op.c, op.reserved);
    var product: usize = 1;
    for (spec.ne[0..spec.rank]) |dim| product = try std.math.mul(usize, product, dim);
    if (product != current.nElems()) return error.ShapeMismatch;
    return spec;
}

fn moduleBroadcastSpec(op: zgml_module_op_desc, current: *const TensorF32, current_rank: usize) !ModuleShapeSpec {
    if (op.activation != 0 or op.flags != 0 or op.eps != 0) return error.InvalidArgument;
    const spec = try moduleShapeSpecFromRowMajor(op.a, op.b, op.c, op.reserved);
    if (current_rank == 0 or current_rank > 3 or spec.rank < current_rank) return error.ShapeMismatch;
    const src_offset = spec.rank - current_rank;
    for (0..spec.rank) |axis| {
        const target_dim = try moduleNativeShapeDim(spec, axis);
        const src_dim = if (axis < src_offset) 1 else try moduleRowMajorDim(current, current_rank, axis - src_offset);
        if (target_dim == 0 or target_dim % src_dim != 0) return error.ShapeMismatch;
    }
    return spec;
}

const ModuleNarrowResult = struct {
    tensor: *TensorF32,
    feature_len: usize,
};

fn moduleNarrowWithStep(op: zgml_module_op_desc, graph_alloc: std.mem.Allocator, current: *TensorF32, current_rank: usize, step: usize) !ModuleNarrowResult {
    if (op.activation != 0 or op.flags != 0 or op.eps != 0) return error.InvalidArgument;
    if (current_rank == 0 or current_rank > 3) return error.ShapeMismatch;
    const axis = op.a;
    if (axis >= current_rank or op.c == 0 or step == 0) return error.ShapeMismatch;
    const dim_len = try moduleRowMajorDim(current, current_rank, axis);
    const span = std.math.mul(usize, op.c - 1, step) catch return error.ShapeMismatch;
    const end_last = std.math.add(usize, op.b, span) catch return error.ShapeMismatch;
    const end = std.math.add(usize, end_last, 1) catch return error.ShapeMismatch;
    if (end > dim_len) return error.ShapeMismatch;

    if (current_rank == 1) {
        const ne = [_]usize{op.c};
        const stride0 = std.math.mul(usize, current.strides[0], step) catch return error.ShapeMismatch;
        const strides = [_]usize{stride0};
        const offset = std.math.mul(usize, op.b, current.strides[0]) catch return error.ShapeMismatch;
        const view = current.asStrided(ne[0..], strides[0..], offset);
        return .{
            .tensor = if (step == 1) view else try moduleMaterializeDense(graph_alloc, view),
            .feature_len = op.c,
        };
    }

    if (current_rank == 3) {
        if (axis == 0) {
            const ne = [_]usize{ current.ne[0], current.ne[1], op.c };
            const stride2 = std.math.mul(usize, current.strides[2], step) catch return error.ShapeMismatch;
            const strides = [_]usize{ current.strides[0], current.strides[1], stride2 };
            const offset = std.math.mul(usize, op.b, current.strides[2]) catch return error.ShapeMismatch;
            const view = current.asStrided(ne[0..], strides[0..], offset);
            return .{
                .tensor = try moduleMaterializeDense(graph_alloc, view),
                .feature_len = current.ne[0],
            };
        }
        if (axis == 1) {
            const ne = [_]usize{ current.ne[0], op.c, current.ne[2] };
            const stride1 = std.math.mul(usize, current.strides[1], step) catch return error.ShapeMismatch;
            const strides = [_]usize{ current.strides[0], stride1, current.strides[2] };
            const offset = std.math.mul(usize, op.b, current.strides[1]) catch return error.ShapeMismatch;
            const view = current.asStrided(ne[0..], strides[0..], offset);
            return .{
                .tensor = try moduleMaterializeDense(graph_alloc, view),
                .feature_len = current.ne[0],
            };
        }
        if (axis == 2) {
            const ne = [_]usize{ op.c, current.ne[1], current.ne[2] };
            const stride0 = std.math.mul(usize, current.strides[0], step) catch return error.ShapeMismatch;
            const strides = [_]usize{ stride0, current.strides[1], current.strides[2] };
            const offset = std.math.mul(usize, op.b, current.strides[0]) catch return error.ShapeMismatch;
            const view = current.asStrided(ne[0..], strides[0..], offset);
            return .{
                .tensor = try moduleMaterializeDense(graph_alloc, view),
                .feature_len = op.c,
            };
        }
        return error.ShapeMismatch;
    }

    if (axis == 0) {
        const ne = [_]usize{ current.ne[0], op.c };
        const stride1 = std.math.mul(usize, current.strides[1], step) catch return error.ShapeMismatch;
        const strides = [_]usize{ current.strides[0], stride1 };
        const offset = std.math.mul(usize, op.b, current.strides[1]) catch return error.ShapeMismatch;
        const view = current.asStrided(ne[0..], strides[0..], offset);
        return .{
            .tensor = if (step == 1) view else try moduleMaterializeDense(graph_alloc, view),
            .feature_len = current.ne[0],
        };
    }
    if (axis == 1) {
        const ne = [_]usize{ op.c, current.ne[1] };
        const stride0 = std.math.mul(usize, current.strides[0], step) catch return error.ShapeMismatch;
        const strides = [_]usize{ stride0, current.strides[1] };
        const offset = std.math.mul(usize, op.b, current.strides[0]) catch return error.ShapeMismatch;
        const view = current.asStrided(ne[0..], strides[0..], offset);
        return .{
            .tensor = if (step == 1 and current.ne[1] == 1) view else try moduleMaterializeDense(graph_alloc, view),
            .feature_len = op.c,
        };
    }
    return error.ShapeMismatch;
}

fn moduleNarrow(op: zgml_module_op_desc, graph_alloc: std.mem.Allocator, current: *TensorF32, current_rank: usize) !ModuleNarrowResult {
    if (op.reserved != 0) return error.InvalidArgument;
    return moduleNarrowWithStep(op, graph_alloc, current, current_rank, 1);
}

fn moduleSlice(op: zgml_module_op_desc, graph_alloc: std.mem.Allocator, current: *TensorF32, current_rank: usize) !ModuleNarrowResult {
    if (op.reserved == 0) return error.InvalidArgument;
    return moduleNarrowWithStep(op, graph_alloc, current, current_rank, op.reserved);
}

fn moduleMaterializeDense(graph_alloc: std.mem.Allocator, source: *TensorF32) !*TensorF32 {
    const out = try TensorF32.init(graph_alloc, source.ne[0..source.n_dims]);
    out.op = .repeat;
    out.src0 = source;
    return out;
}

fn moduleTranspose(op: zgml_module_op_desc, graph_alloc: std.mem.Allocator, current: *TensorF32, current_rank: usize) !ModuleNarrowResult {
    if (op.activation != 0 or op.flags != 0 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
    if (current_rank != 2 and current_rank != 3) return error.Unsupported;
    if (op.a >= current_rank or op.b >= current_rank or op.a == op.b) return error.Unsupported;
    var axes = [_]usize{0} ** tensor_mod.max_dims;
    for (0..current.n_dims) |axis| axes[axis] = axis;
    const native_axis_a = current_rank - 1 - op.a;
    const native_axis_b = current_rank - 1 - op.b;
    axes[native_axis_a] = native_axis_b;
    axes[native_axis_b] = native_axis_a;
    const transposed = current.permute(axes[0..current.n_dims]);
    return .{
        .tensor = try moduleMaterializeDense(graph_alloc, transposed),
        .feature_len = transposed.ne[0],
    };
}

fn moduleDiagonal(op: zgml_module_op_desc, graph_alloc: std.mem.Allocator, current: *TensorF32, current_rank: usize) !ModuleNarrowResult {
    if (op.activation != 0 or op.flags != 0 or op.a != 0 or op.b != 0 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
    if (current_rank != 2) return error.Unsupported;
    const diag_len = @min(current.ne[0], current.ne[1]);
    if (diag_len == 0) return error.ShapeMismatch;
    const ne = [_]usize{diag_len};
    const stride = std.math.add(usize, current.strides[0], current.strides[1]) catch return error.ShapeMismatch;
    const strides = [_]usize{stride};
    const diagonal = current.asStrided(ne[0..], strides[0..], 0);
    return .{
        .tensor = try moduleMaterializeDense(graph_alloc, diagonal),
        .feature_len = diag_len,
    };
}

fn compileModuleProgram(desc: *const zgml_module_desc, backend: llm_mod.LlamaBackend) !ModuleProgramHandle {
    if ((desc.input_rank < 1 or desc.input_rank > 4) or desc.input_shape == null) return error.InvalidArgument;
    if (desc.op_count > 0 and desc.ops == null) return error.InvalidArgument;
    const input_shape = desc.input_shape.?[0..desc.input_rank];
    var input_ne: [4]usize = undefined;
    const input_native_rank: usize = switch (desc.input_rank) {
        1 => blk: {
            if (input_shape[0] == 0) return error.ShapeMismatch;
            input_ne[0] = input_shape[0];
            break :blk 1;
        },
        2 => blk: {
            if (input_shape[0] == 0 or input_shape[1] == 0) return error.ShapeMismatch;
            input_ne[0] = input_shape[1];
            input_ne[1] = input_shape[0];
            break :blk 2;
        },
        3 => blk: {
            if (input_shape[0] == 0 or input_shape[1] == 0 or input_shape[2] == 0) return error.ShapeMismatch;
            input_ne[0] = input_shape[2];
            input_ne[1] = input_shape[1];
            input_ne[2] = input_shape[0];
            input_ne[3] = 1;
            break :blk 4;
        },
        4 => blk: {
            if (input_shape[0] == 0 or input_shape[1] == 0 or input_shape[2] == 0 or input_shape[3] == 0) return error.ShapeMismatch;
            input_ne[0] = input_shape[3];
            input_ne[1] = input_shape[2];
            input_ne[2] = input_shape[1];
            input_ne[3] = input_shape[0];
            break :blk 4;
        },
        else => unreachable,
    };
    const input_features: usize = switch (desc.input_rank) {
        2 => input_shape[1],
        4 => input_shape[1],
        else => input_shape[0],
    };

    var graph = GraphF32.init(alloc);
    errdefer graph.deinit();
    const graph_alloc = graph.allocator();

    const input = try TensorF32.init(graph_alloc, input_ne[0..input_native_rank]);
    @memset(input.data, 0);
    var current = input;
    var current_rank = desc.input_rank;
    var current_len = input_features;
    var weights_len: usize = 0;
    var bias_len: usize = 0;

    var persistent_tensors_list: std.ArrayListUnmanaged(*TensorF32) = .empty;
    defer persistent_tensors_list.deinit(alloc);
    var persistent_params_list: std.ArrayListUnmanaged(ModuleParamBinding) = .empty;
    defer persistent_params_list.deinit(alloc);

    const ops: []const zgml_module_op_desc = if (desc.op_count == 0) &.{} else desc.ops.?[0..desc.op_count];
    for (ops) |op| {
        if (op.reserved != 0 and op.kind != module_op_slice and op.kind != module_op_reshape and op.kind != module_op_broadcast_to) return error.InvalidArgument;
        switch (op.kind) {
            module_op_linear => {
                const in_features = op.a;
                const out_features = op.b;
                if (in_features == 0 or out_features == 0 or current_len != in_features) return error.ShapeMismatch;
                const weight = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .weights,
                    &.{ out_features, in_features },
                    &weights_len,
                );
                const bias = if ((op.flags & module_flag_bias) != 0)
                    try moduleAddParam(
                        graph_alloc,
                        &persistent_tensors_list,
                        &persistent_params_list,
                        .bias,
                        &.{out_features},
                        &bias_len,
                    )
                else
                    null;
                if (current_rank == 3) {
                    const restore_dim1 = current.ne[1];
                    const restore_dim2 = current.ne[2];
                    const rows = try std.math.mul(usize, restore_dim1, restore_dim2);
                    var linear_output = nn.linear(f32, current.reshape(&.{ current.ne[0], rows }), weight, bias);
                    if (op.activation != 0) {
                        linear_output = try moduleActivate(linear_output, op.activation);
                    }
                    current = linear_output.reshape(&.{ out_features, restore_dim1, restore_dim2 });
                } else {
                    current = nn.linear(f32, current, weight, bias);
                    if (op.activation != 0) {
                        current = try moduleActivate(current, op.activation);
                    }
                }
                current_len = out_features;
            },
            module_op_add => {
                const features = op.a;
                if (features == 0 or features != current_len) return error.ShapeMismatch;
                if (op.b != 0 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
                if (op.flags != module_flag_bias) return error.InvalidArgument;
                const bias = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .bias,
                    &.{features},
                    &bias_len,
                );
                current = current.add(bias.repeatLike(current));
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
            },
            module_op_mul => {
                const features = op.a;
                if (features == 0 or features != current_len) return error.ShapeMismatch;
                if (op.b != 0 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
                if (op.flags != module_flag_weight) return error.InvalidArgument;
                const weight = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .weights,
                    &.{features},
                    &weights_len,
                );
                current = current.mul(weight.repeatLike(current));
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
            },
            module_op_feature_affine => {
                const features = op.a;
                if (features == 0 or features != current_len) return error.ShapeMismatch;
                if (op.b != 0 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
                if (op.flags != (module_flag_weight | module_flag_bias)) return error.InvalidArgument;
                const weight = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .weights,
                    &.{features},
                    &weights_len,
                );
                const bias = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .bias,
                    &.{features},
                    &bias_len,
                );
                current = current.mul(weight.repeatLike(current)).add(bias.repeatLike(current));
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
            },
            module_op_activation => {
                if (op.flags != 0 or op.a != 0 or op.b != 0 or op.c != 0) return error.InvalidArgument;
                current = try moduleActivate(current, op.activation);
            },
            module_op_activation_chain => {
                current = try moduleActivateChain(current, op);
            },
            module_op_softmax => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.softmaxDim(op.a);
            },
            module_op_log_softmax => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.logSoftmaxDim(op.a);
            },
            module_op_reduce_sum => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.sumDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_prod => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.prodDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_mean => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.meanDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_max => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.maxDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_min => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.minDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_argmax => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.argmaxDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reduce_argmin => {
                if (op.activation != 0 or op.flags != 0 or op.b != 0 or op.c != 0 or op.a != 0) return error.InvalidArgument;
                if (op.a >= current_rank) return error.ShapeMismatch;
                current = current.argminDim(op.a);
                current_len = current.ne[0];
            },
            module_op_reshape => {
                const spec = try moduleReshapeSpec(op, current);
                current = current.reshape(spec.ne[0..spec.rank]);
                current_rank = spec.rank;
                current_len = spec.feature_len;
            },
            module_op_broadcast_to => {
                var spec = try moduleBroadcastSpec(op, current, current_rank);
                current = current.repeat(spec.ne[0..spec.rank]);
                current_rank = spec.rank;
                current_len = spec.feature_len;
            },
            module_op_narrow => {
                const narrowed = try moduleNarrow(op, graph_alloc, current, current_rank);
                current = narrowed.tensor;
                current_len = narrowed.feature_len;
            },
            module_op_slice => {
                const sliced = try moduleSlice(op, graph_alloc, current, current_rank);
                current = sliced.tensor;
                current_len = sliced.feature_len;
            },
            module_op_transpose => {
                const transposed = try moduleTranspose(op, graph_alloc, current, current_rank);
                current = transposed.tensor;
                current_len = transposed.feature_len;
            },
            module_op_diagonal => {
                const diagonal = try moduleDiagonal(op, graph_alloc, current, current_rank);
                current = diagonal.tensor;
                current_rank = 1;
                current_len = diagonal.feature_len;
            },
            module_op_layer_norm => {
                const features = op.a;
                if (features != current_len or op.b != 0 or op.c != 0) return error.ShapeMismatch;
                const reduce_ne = try moduleFeatureNormReduceShape(current, features);
                current = current.layerNorm(reduce_ne[0..current.n_dims], @floatCast(op.eps));
                if ((op.flags & module_flag_weight) != 0) {
                    const weight = try moduleAddParam(
                        graph_alloc,
                        &persistent_tensors_list,
                        &persistent_params_list,
                        .weights,
                        &.{features},
                        &weights_len,
                    );
                    const repeated_weight = weight.repeatLike(current);
                    current = current.mul(repeated_weight);
                }
                if ((op.flags & module_flag_bias) != 0) {
                    const bias = try moduleAddParam(
                        graph_alloc,
                        &persistent_tensors_list,
                        &persistent_params_list,
                        .bias,
                        &.{features},
                        &bias_len,
                    );
                    const repeated_bias = bias.repeatLike(current);
                    current = current.add(repeated_bias);
                }
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
            },
            module_op_rms_norm => {
                const features = op.a;
                if (features != current_len or op.b != 0 or op.c != 0) return error.ShapeMismatch;
                if ((op.flags & module_flag_bias) != 0) return error.InvalidArgument;
                const reduce_ne = try moduleFeatureNormReduceShape(current, features);
                current = current.rmsNorm(reduce_ne[0..current.n_dims], @floatCast(op.eps));
                if ((op.flags & module_flag_weight) != 0) {
                    const weight = try moduleAddParam(
                        graph_alloc,
                        &persistent_tensors_list,
                        &persistent_params_list,
                        .weights,
                        &.{features},
                        &weights_len,
                    );
                    const repeated_weight = weight.repeatLike(current);
                    current = current.mul(repeated_weight);
                }
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
            },
            module_op_embedding => {
                const num_embeddings = op.a;
                const embedding_dim = op.b;
                if (num_embeddings == 0 or embedding_dim == 0) return error.ShapeMismatch;
                if (op.activation != 0 or op.c != 0) return error.InvalidArgument;
                if (op.flags != module_flag_weight) return error.InvalidArgument;
                if (!current.isVector()) return error.ShapeMismatch;
                const weight = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .weights,
                    &.{ embedding_dim, num_embeddings },
                    &weights_len,
                );
                current = weight.gatherRows(current);
                current_rank = 2;
                current_len = embedding_dim;
            },
            module_op_conv2d => {
                const out_channels = op.a;
                const kh = op.b;
                const kw = op.c;
                if (op.eps != 0 or out_channels == 0 or kh == 0 or kw == 0) return error.InvalidArgument;
                if ((op.flags & ~module_flag_bias) != 0) return error.InvalidArgument;
                if ((current_rank != 3 and current_rank != 4) or current.n_dims != 4) return error.Unsupported;
                if (current_rank == 3 and current.ne[3] != 1) return error.Unsupported;
                if (current.ne[0] < kw or current.ne[1] < kh) return error.ShapeMismatch;
                const in_channels = current.ne[2];
                const weight = try moduleAddParam(
                    graph_alloc,
                    &persistent_tensors_list,
                    &persistent_params_list,
                    .weights,
                    &.{ kw, kh, in_channels, out_channels },
                    &weights_len,
                );
                const bias = if ((op.flags & module_flag_bias) != 0)
                    try moduleAddParam(
                        graph_alloc,
                        &persistent_tensors_list,
                        &persistent_params_list,
                        .bias,
                        &.{ 1, 1, out_channels, 1 },
                        &bias_len,
                    )
                else
                    null;
                current = current.conv2d(weight);
                if (bias) |b| {
                    const repeated_bias = b.repeatLike(current);
                    current = current.add(repeated_bias);
                }
                if (op.activation != 0) {
                    current = try moduleActivate(current, op.activation);
                }
                current_rank = if (current.ne[3] == 1 and current_rank == 3) 3 else 4;
                current_len = out_channels;
            },
            module_op_max_pool2d => {
                if (op.activation != 0 or op.flags != 0 or op.a != 2 or op.b != 2 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
                if ((current_rank != 3 and current_rank != 4) or current.n_dims != 4) return error.Unsupported;
                if (current_rank == 3 and current.ne[3] != 1) return error.Unsupported;
                if (current.ne[0] % 2 != 0 or current.ne[1] % 2 != 0) return error.ShapeMismatch;
                current = current.maxPool2d();
                current_rank = if (current.ne[3] == 1 and current_rank == 3) 3 else 4;
                current_len = current.ne[2];
            },
            module_op_avg_pool2d => {
                if (op.activation != 0 or op.flags != 0 or op.a != 2 or op.b != 2 or op.c != 0 or op.eps != 0) return error.InvalidArgument;
                if ((current_rank != 3 and current_rank != 4) or current.n_dims != 4) return error.Unsupported;
                if (current_rank == 3 and current.ne[3] != 1) return error.Unsupported;
                if (current.ne[0] % 2 != 0 or current.ne[1] % 2 != 0) return error.ShapeMismatch;
                current = current.avgPool2d();
                current_rank = if (current.ne[3] == 1 and current_rank == 3) 3 else 4;
                current_len = current.ne[2];
            },
            else => return error.Unsupported,
        }
    }
    try graph.infer(current);

    const persistent_tensors = try alloc.dupe(*TensorF32, persistent_tensors_list.items);
    errdefer alloc.free(persistent_tensors);
    const persistent_params = try alloc.dupe(ModuleParamBinding, persistent_params_list.items);
    errdefer alloc.free(persistent_params);

    var cpu = cpu_mod.CpuBackend{};
    var stencil = stencil_mod.StencilBackend.webgpuResourceSessionProbe();
    const backend_id: u32 = switch (backend) {
        .auto, .cpu => backend_cpu,
        .webgpu => backend_webgpu,
        .metal => return error.Unsupported,
    };
    var program = switch (backend_id) {
        backend_cpu => try DeviceF32.Program.compile(.{
            .graph = &graph,
            .be = cpu.backend(),
            .alloc = alloc,
            .input_tensors = &.{input},
            .persistent_tensors = persistent_tensors,
            .output_tensors = &.{current},
        }),
        backend_webgpu => blk: {
            if (comptime build_options.use_wgpu) {
                var wgpu = wgpu_mod.WgpuBackend.init(alloc);
                break :blk try DeviceF32.Program.compile(.{
                    .graph = &graph,
                    .be = wgpu.backend(),
                    .alloc = alloc,
                    .input_tensors = &.{input},
                    .persistent_tensors = persistent_tensors,
                    .output_tensors = &.{current},
                });
            }
            break :blk try DeviceF32.Program.compile(.{
                .graph = &graph,
                .be = stencil.backend(),
                .alloc = alloc,
                .input_tensors = &.{input},
                .persistent_tensors = persistent_tensors,
                .output_tensors = &.{current},
            });
        },
        else => unreachable,
    };
    errdefer program.deinit();
    const inspection = program.inspect();
    const direct_linear: ?DirectLinearModule = if (ops.len == 1 and
        ops[0].kind == module_op_linear and
        ops[0].activation == 0 and
        (ops[0].flags & ~module_flag_bias) == 0)
        .{
            .in_features = ops[0].a,
            .out_features = ops[0].b,
            .has_bias = (ops[0].flags & module_flag_bias) != 0,
        }
    else
        null;
    const direct_rms_gelu_linear: ?DirectRmsGeluLinearModule = if (ops.len == 2 and
        ops[0].kind == module_op_rms_norm and
        ops[0].activation == module_activation_gelu and
        ops[0].flags == module_flag_weight and
        ops[0].a != 0 and
        ops[0].b == 0 and
        ops[0].c == 0 and
        ops[1].kind == module_op_linear and
        ops[1].activation == 0 and
        ops[1].flags == module_flag_bias and
        ops[1].a == ops[0].a and
        ops[1].b != 0)
        .{
            .features = ops[0].a,
            .out_features = ops[1].b,
            .eps = @floatCast(ops[0].eps),
        }
    else
        null;
    const direct_linear_log_softmax: ?DirectLinearLogSoftmaxModule = if (ops.len == 2 and
        ops[0].kind == module_op_linear and
        ops[0].activation == 0 and
        (ops[0].flags & ~module_flag_bias) == 0 and
        ops[0].a != 0 and
        ops[0].b != 0 and
        ops[1].kind == module_op_log_softmax and
        ops[1].activation == 0 and
        ops[1].flags == 0 and
        ops[1].a == 0 and
        ops[1].b == 0 and
        ops[1].c == 0)
        .{
            .in_features = ops[0].a,
            .out_features = ops[0].b,
            .has_bias = (ops[0].flags & module_flag_bias) != 0,
        }
    else
        null;

    return .{
        .input_len = input.nElems(),
        .output_len = current.nElems(),
        .weights_len = weights_len,
        .bias_len = bias_len,
        .backend = backend_id,
        .execution_supported = inspection.execution_supported,
        .direct_linear = direct_linear,
        .direct_rms_gelu_linear = direct_rms_gelu_linear,
        .direct_linear_log_softmax = direct_linear_log_softmax,
        .graph = graph,
        .program = program,
        .input = input,
        .output = current,
        .persistent_tensors = persistent_tensors,
        .persistent_params = persistent_params,
    };
}

fn compileErrorStatus(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => status(.out_of_memory),
        error.MetalNotAvailable => status(.unsupported),
        error.Unsupported => status(.unsupported),
        error.UnsupportedDeviceOp => status(.unsupported),
        error.ExecutionUnsupported => status(.unsupported),
        error.UnsupportedResourceBinding => status(.unsupported),
        error.WebGPUUnavailable, error.WgpuUnavailable => status(.unsupported),
        error.WebGPUExecutionUnavailable => status(.unsupported),
        error.UnsupportedBatch => status(.unsupported),
        error.InvalidArgument, error.InvalidBatch, error.InvalidContextLength => status(.invalid_argument),
        error.ShapeMismatch, error.InvalidProgramIO, error.OutputBufferTooSmall, error.TokenIdOutOfRange, error.SequenceTooLong, error.InvalidPrefillLength, error.InvalidTokenWindow => status(.shape_mismatch),
        else => status(.compile_failed),
    };
}

fn compileBackend(desc_ptr: ?*const zgml_compile_desc) !llm_mod.LlamaBackend {
    const raw = if (desc_ptr) |desc| desc.backend else backend_auto;
    return switch (raw) {
        backend_auto => .auto,
        backend_cpu => .cpu,
        backend_metal => .metal,
        backend_webgpu => .webgpu,
        else => error.InvalidBackend,
    };
}

fn llamaCompileOptions(desc_ptr: ?*const zgml_compile_desc, backend: llm_mod.LlamaBackend, max_seq_len: usize) llm_mod.LlamaCompileOptions {
    const context_len = if (desc_ptr) |desc|
        if (desc.context_len == 0) max_seq_len else desc.context_len
    else
        max_seq_len;
    const batch = if (desc_ptr) |desc|
        if (desc.batch == 0) @as(usize, 1) else desc.batch
    else
        1;
    return .{
        .context_len = context_len,
        .batch = batch,
        .backend = backend,
    };
}

fn backendIdForLlamaBackend(backend: llm_mod.LlamaBackend) u64 {
    return switch (backend) {
        .auto => backend_auto,
        .cpu => backend_cpu,
        .metal => backend_metal,
        .webgpu => backend_webgpu,
    };
}

fn fillCommandCategoryCounts(out: *zgml_program_inspection, categories: program_mod.ProgramCommandCategoryCounts) void {
    out.command_op_count = categories.op;
    out.command_row_count = categories.row;
    out.command_projection_count = categories.projection;
    out.command_attention_count = categories.attention;
    out.command_movement_count = categories.movement;
    out.command_elementwise_count = categories.elementwise;
    out.command_rope_count = categories.rope;
}

fn commandCategoryTotal(inspection: zgml_program_inspection) u64 {
    return inspection.command_op_count +
        inspection.command_row_count +
        inspection.command_projection_count +
        inspection.command_attention_count +
        inspection.command_movement_count +
        inspection.command_elementwise_count +
        inspection.command_rope_count;
}

fn runtimePatchEnvelopeValue(value: ?u32) u64 {
    return if (value) |v| @as(u64, v) else std.math.maxInt(u64);
}

fn fillExecutionPlanInspection(out: *zgml_program_inspection, plan: backend_mod.ExecutionPlanInspection) void {
    out.backend_dispatch_count = plan.dispatch_count;
    out.dispatch_plan_supported = @intFromBool(plan.supported);
    out.dispatch_plan_covered_op_count = plan.covered_op_count;
    out.dispatch_plan_first_unsupported_op = plan.first_unsupported_op orelse std.math.maxInt(u64);
    out.dispatch_plan_projection_count = plan.family_counts.projectionCount();
    out.dispatch_plan_row_count = plan.family_counts.rowCount();
    out.dispatch_plan_attention_count = plan.family_counts.attention;
    out.dispatch_plan_movement_count = plan.family_counts.movementCount();
    out.dispatch_plan_elementwise_count = plan.family_counts.fused_elementwise + plan.family_counts.elementwise;
    out.dispatch_plan_rope_count = plan.family_counts.rope;
    out.dispatch_plan_quantized_projection_count = plan.family_counts.quantizedProjectionCount();
}

fn fillProgramInspection(out: *zgml_program_inspection, inspection: anytype) void {
    out.* = .{
        .backend = backendIdForLlamaBackend(inspection.backend),
        .execution_supported = @intFromBool(inspection.execution_supported),
        .external_resources_supported = @intFromBool(inspection.external_resources_supported),
        .buffer_count = @intCast(inspection.buffer_count),
        .buffer_element_count = @intCast(inspection.buffer_element_count),
        .buffer_byte_len = @intCast(inspection.buffer_byte_len),
        .initial_upload_count = @intCast(inspection.initial_upload_count),
        .qweight_count = @intCast(inspection.qweight_count),
        .op_count = @intCast(inspection.op_count),
        .runtime_patch_max_cache_write_pos = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_cache_write_pos),
        .runtime_patch_max_attention_seq_kv = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_attention_seq_kv),
        .command_count = @intCast(inspection.command_count),
        .command_stencil_hash = @intCast(inspection.command_stencil_hash),
        .command_op_count = @intCast(inspection.command_op_count),
        .command_row_count = @intCast(inspection.command_row_count),
        .command_projection_count = @intCast(inspection.command_projection_count),
        .command_attention_count = @intCast(inspection.command_attention_count),
        .command_movement_count = @intCast(inspection.command_movement_count),
        .command_elementwise_count = @intCast(inspection.command_elementwise_count),
        .command_rope_count = @intCast(inspection.command_rope_count),
        .runtime_patch_holes = @intCast(inspection.runtime_patch_holes),
        .runtime_patch_cache_write_pos_holes = @intCast(inspection.runtime_patch_cache_write_pos_holes),
        .runtime_patch_attention_seq_kv_holes = @intCast(inspection.runtime_patch_attention_seq_kv_holes),
        .runtime_patch_stencil_hash = @intCast(inspection.runtime_patch_stencil_hash),
        .binding_requirement_hash = inspection.binding_requirement_hash,
        .persistent_requirement_count = @intCast(inspection.persistent_requirement_count),
        .step_input_requirement_count = @intCast(inspection.step_input_requirement_count),
        .step_output_requirement_count = @intCast(inspection.step_output_requirement_count),
    };
    fillExecutionPlanInspection(out, inspection.execution_plan);
}

fn fillLlamaProgramInspection(out: *zgml_llama_program_inspection, inspection: llm_mod.LlamaProgramInspection) void {
    out.* = .{
        .vocab_size = @intCast(inspection.vocab_size),
        .max_seq_len = @intCast(inspection.max_seq_len),
        .context_len = @intCast(inspection.context_len),
        .batch = @intCast(inspection.batch),
        .d_model = @intCast(inspection.d_model),
        .n_layers = @intCast(inspection.n_layers),
        .n_heads = @intCast(inspection.n_heads),
        .n_kv_heads = @intCast(inspection.n_kv_heads),
        .semantic_stage_count = @intCast(inspection.semantic_stage_count),
        .semantic_token_count = @intCast(inspection.semantic_token_count),
        .semantic_layer_stage_count = @intCast(inspection.semantic_layer_stage_count),
        .semantic_terminal_stage_count = @intCast(inspection.semantic_terminal_stage_count),
        .semantic_runtime_patch_holes = @intCast(inspection.semantic_runtime_patch_holes),
        .semantic_runtime_patch_cache_write_pos_holes = @intCast(inspection.semantic_runtime_patch_cache_write_pos_holes),
        .semantic_runtime_patch_attention_seq_kv_holes = @intCast(inspection.semantic_runtime_patch_attention_seq_kv_holes),
    };
}

fn fillLlamaSessionInspection(out: *zgml_session_inspection, model_kind: u32, inspection: llm_mod.LlamaSessionInspection) void {
    out.* = .{
        .model_kind = model_kind,
        .backend = @intCast(backendIdForLlamaBackend(inspection.backend)),
        .output_storage = inspection.output_storage,
        .kv_cache_storage = inspection.kv_cache_storage,
        .position = @intCast(inspection.position),
        .context_len = @intCast(inspection.context_len),
        .persistent_binding_count = @intCast(inspection.persistent_binding_count),
        .step_input_count = @intCast(inspection.step_input_count),
        .step_output_count = @intCast(inspection.step_output_count),
        .host_binding_count = @intCast(inspection.host_binding_count),
        .resource_binding_count = @intCast(inspection.resource_binding_count),
        .binding_shape_hash = inspection.binding_shape_hash,
    };
}

fn fillLlamaModelInspection(out: *zgml_model_inspection, kind: u32, config: llm_mod.LlamaConfig) void {
    out.* = .{
        .model_kind = kind,
        .vocab_size = @intCast(config.vocab_size),
        .max_seq_len = @intCast(config.max_seq_len),
        .d_model = @intCast(config.d_model),
        .n_layers = @intCast(config.n_layers),
        .n_heads = @intCast(config.n_heads),
        .n_kv_heads = @intCast(config.n_kv_heads),
        .d_ff = @intCast(config.d_ff),
        .rope_base = @floatCast(config.rope_base),
        .rms_norm_eps = @floatCast(config.rms_norm_eps),
        .tied_lm_head = @intFromBool(config.tied_lm_head),
    };
}

fn fillRuntimeProfile(out: *zgml_runtime_profile, rt: profile_mod.RuntimeProfile) void {
    const categories = rt.program_command_shape.categoryCounts();
    out.* = .{
        .call_count = @intCast(rt.call_count),
        .backend_op_count = rt.backend_op_count,
        .fallback_op_count = rt.fallback_op_count,
        .backend_dispatch_count = rt.backend_dispatch_count,
        .sync_count = rt.sync_count,
        .runtime_patch_call_count = rt.runtime_patch_call_count,
        .runtime_patch_changed_count = rt.runtime_patch_changed_count,
        .runtime_patch_invalid_count = rt.runtime_patch_invalid_count,
        .runtime_patch_holes = rt.runtime_patch_shape.runtime_patch_holes,
        .runtime_patch_cache_write_pos_holes = rt.runtime_patch_shape.runtime_patch_cache_write_pos_holes,
        .runtime_patch_attention_seq_kv_holes = rt.runtime_patch_shape.runtime_patch_attention_seq_kv_holes,
        .runtime_patch_stencil_hash = rt.runtime_patch_shape.runtime_patch_stencil_hash,
        .command_count = rt.program_command_shape.command_count,
        .command_stencil_hash = rt.program_command_shape.command_stencil_hash,
        .command_op_count = categories.op,
        .command_row_count = categories.row,
        .command_projection_count = categories.projection,
        .command_attention_count = categories.attention,
        .command_movement_count = categories.movement,
        .command_elementwise_count = categories.elementwise,
        .command_rope_count = categories.rope,
    };
}

fn expectRuntimeProfilesEqual(expected: zgml_runtime_profile, actual: zgml_runtime_profile) !void {
    try std.testing.expectEqual(expected.call_count, actual.call_count);
    try std.testing.expectEqual(expected.backend_op_count, actual.backend_op_count);
    try std.testing.expectEqual(expected.fallback_op_count, actual.fallback_op_count);
    try std.testing.expectEqual(expected.backend_dispatch_count, actual.backend_dispatch_count);
    try std.testing.expectEqual(expected.sync_count, actual.sync_count);
    try std.testing.expectEqual(expected.runtime_patch_call_count, actual.runtime_patch_call_count);
    try std.testing.expectEqual(expected.runtime_patch_changed_count, actual.runtime_patch_changed_count);
    try std.testing.expectEqual(expected.runtime_patch_invalid_count, actual.runtime_patch_invalid_count);
    try std.testing.expectEqual(expected.runtime_patch_holes, actual.runtime_patch_holes);
    try std.testing.expectEqual(expected.runtime_patch_cache_write_pos_holes, actual.runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(expected.runtime_patch_attention_seq_kv_holes, actual.runtime_patch_attention_seq_kv_holes);
    try std.testing.expectEqual(expected.runtime_patch_stencil_hash, actual.runtime_patch_stencil_hash);
    try std.testing.expectEqual(expected.command_count, actual.command_count);
    try std.testing.expectEqual(expected.command_stencil_hash, actual.command_stencil_hash);
    try std.testing.expectEqual(expected.command_op_count, actual.command_op_count);
    try std.testing.expectEqual(expected.command_row_count, actual.command_row_count);
    try std.testing.expectEqual(expected.command_projection_count, actual.command_projection_count);
    try std.testing.expectEqual(expected.command_attention_count, actual.command_attention_count);
    try std.testing.expectEqual(expected.command_movement_count, actual.command_movement_count);
    try std.testing.expectEqual(expected.command_elementwise_count, actual.command_elementwise_count);
    try std.testing.expectEqual(expected.command_rope_count, actual.command_rope_count);
}

fn addProgramRuntimeProfileTo(p: *ProgramHandle, dest: *profile_mod.RuntimeProfile) void {
    switch (p.data) {
        .tiny_linear => |*linear| linear.program.addRuntimeProfileTo(dest),
        .tiny_mlp => |*mlp| mlp.program.addRuntimeProfileTo(dest),
        .module => |*module| module.program.addRuntimeProfileTo(dest),
        .tiny_llama => |*llama| llama.program.addRuntimeProfileTo(dest),
        .tiny_llama_2layer => |*llama| llama.program.addRuntimeProfileTo(dest),
        .smollm_135m => |*llama| llama.program.addRuntimeProfileTo(dest),
    }
}

fn resetProgramRuntimeProfile(p: *ProgramHandle) void {
    switch (p.data) {
        .tiny_linear => |*linear| linear.program.resetRuntimeProfile(),
        .tiny_mlp => |*mlp| mlp.program.resetRuntimeProfile(),
        .module => |*module| module.program.resetRuntimeProfile(),
        .tiny_llama => |*llama| llama.program.resetRuntimeProfile(),
        .tiny_llama_2layer => |*llama| llama.program.resetRuntimeProfile(),
        .smollm_135m => |*llama| llama.program.resetRuntimeProfile(),
    }
}

fn addSessionRuntimeProfileTo(s: *SessionHandle, dest: *profile_mod.RuntimeProfile) void {
    switch (s.data) {
        .tiny_linear => |*linear| linear.session.addRuntimeProfileTo(dest),
        .tiny_mlp => |*mlp| mlp.session.addRuntimeProfileTo(dest),
        .module => |*module| module.session.addRuntimeProfileTo(dest),
        .tiny_llama => |*llama| llama.session.addRuntimeProfileTo(dest),
        .tiny_llama_2layer => |*llama| llama.session.addRuntimeProfileTo(dest),
        .smollm_135m => |*llama| llama.session.addRuntimeProfileTo(dest),
    }
}

fn resetSessionRuntimeProfile(s: *SessionHandle) void {
    switch (s.data) {
        .tiny_linear => |*linear| linear.session.resetRuntimeProfile(),
        .tiny_mlp => |*mlp| mlp.session.resetRuntimeProfile(),
        .module => |*module| module.session.resetRuntimeProfile(),
        .tiny_llama => |*llama| llama.session.resetRuntimeProfile(),
        .tiny_llama_2layer => |*llama| llama.session.resetRuntimeProfile(),
        .smollm_135m => |*llama| llama.session.resetRuntimeProfile(),
    }
}

fn sameLlamaConfig(a: llm_mod.LlamaConfig, b: llm_mod.LlamaConfig) bool {
    return a.vocab_size == b.vocab_size and
        a.d_model == b.d_model and
        a.n_heads == b.n_heads and
        a.n_kv_heads == b.n_kv_heads and
        a.d_ff == b.d_ff and
        a.n_layers == b.n_layers and
        a.max_seq_len == b.max_seq_len and
        @abs(a.rope_base - b.rope_base) <= 0.001 and
        @abs(a.rms_norm_eps - b.rms_norm_eps) <= 0.000001 and
        a.tied_lm_head == b.tied_lm_head;
}

fn compatibleLlamaKindForConfig(config: llm_mod.LlamaConfig) ?u32 {
    for (compatible_llama_families) |family| {
        if (sameLlamaConfig(config, family.config)) return family.kind;
    }
    return null;
}

fn safetensorsTensorShapeMatches(sf: *const safetensors_mod.SafetensorsFile, name: []const u8, expected: []const usize) bool {
    const meta = sf.findTensorMeta(name) orelse return false;
    switch (meta.dtype) {
        .f32, .f16 => {},
        else => return false,
    }
    if (meta.n_dims != expected.len) return false;
    for (expected, 0..) |dim, i| {
        if (meta.shape[i] != dim) return false;
    }
    return true;
}

fn safetensorsLayerTensorShapeMatches(sf: *const safetensors_mod.SafetensorsFile, layer: usize, suffix: []const u8, expected: []const usize) bool {
    var name_buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer, suffix }) catch return false;
    return safetensorsTensorShapeMatches(sf, name, expected);
}

fn safetensorsLayerEnvelopeMatches(sf: *const safetensors_mod.SafetensorsFile, config: llm_mod.LlamaConfig, layer: usize) bool {
    if (config.n_heads == 0 or config.d_model % config.n_heads != 0) return false;
    const d_head = config.d_model / config.n_heads;
    const kv_dim = config.n_kv_heads * d_head;
    return safetensorsLayerTensorShapeMatches(sf, layer, "self_attn.q_proj.weight", &.{ config.d_model, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "self_attn.k_proj.weight", &.{ kv_dim, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "self_attn.v_proj.weight", &.{ kv_dim, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "self_attn.o_proj.weight", &.{ config.d_model, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "mlp.gate_proj.weight", &.{ config.d_ff, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "mlp.up_proj.weight", &.{ config.d_ff, config.d_model }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "mlp.down_proj.weight", &.{ config.d_model, config.d_ff }) and
        safetensorsLayerTensorShapeMatches(sf, layer, "input_layernorm.weight", &.{config.d_model}) and
        safetensorsLayerTensorShapeMatches(sf, layer, "post_attention_layernorm.weight", &.{config.d_model});
}

fn safetensorsLlamaEnvelopeMatches(sf: *const safetensors_mod.SafetensorsFile, config: llm_mod.LlamaConfig) bool {
    if (config.n_layers == 0) return false;
    if (!safetensorsTensorShapeMatches(sf, "model.embed_tokens.weight", &.{ config.vocab_size, config.d_model })) return false;
    if (!safetensorsTensorShapeMatches(sf, "model.norm.weight", &.{config.d_model})) return false;
    if (!config.tied_lm_head and !safetensorsTensorShapeMatches(sf, "lm_head.weight", &.{ config.vocab_size, config.d_model })) return false;
    if (!safetensorsLayerEnvelopeMatches(sf, config, 0)) return false;
    if (!safetensorsLayerEnvelopeMatches(sf, config, config.n_layers - 1)) return false;
    if (safetensorsLayerTensorShapeMatches(sf, config.n_layers, "self_attn.q_proj.weight", &.{ config.d_model, config.d_model })) return false;
    return true;
}

fn compatibleLlamaKindForSafetensors(sf: *const safetensors_mod.SafetensorsFile) ?u32 {
    for (compatible_llama_families) |family| {
        if (safetensorsLlamaEnvelopeMatches(sf, family.config)) return family.kind;
    }
    return null;
}

fn compatibleLlamaKindForSafetensorsHeader(header: []const u8) ?u32 {
    var raw: [1]u8 align(4) = .{0};
    const raw_slice = raw[0..0];
    const sf = safetensors_mod.SafetensorsFile{
        .alloc = alloc,
        .raw_data = raw_slice,
        .header_json = header,
        .data_start = 0,
    };
    return compatibleLlamaKindForSafetensors(&sf);
}

fn safetensorsHeaderFromData(data: []const u8) ![]const u8 {
    if (data.len < 8) return error.InvalidArgument;
    const header_len = std.mem.readInt(u64, data[0..8], .little);
    const header_len_usize: usize = std.math.cast(usize, header_len) orelse return error.InvalidArgument;
    if (header_len_usize > data.len - 8) return error.InvalidArgument;
    return data[8..][0..header_len_usize];
}

fn selectAutoModelKind(path: []const u8) !u32 {
    if (std.ascii.endsWithIgnoreCase(path, ".gguf")) {
        var gf = try gguf_mod.GGUFFile.open(alloc, std.Io.Threaded.global_single_threaded.io(), path);
        defer gf.deinit();
        return compatibleLlamaKindForConfig(gguf_loader.configFromGGUF(&gf)) orelse error.UnsupportedModel;
    }
    if (std.ascii.endsWithIgnoreCase(path, ".safetensors")) {
        var sf = try safetensors_mod.SafetensorsFile.open(alloc, path, std.Io.Threaded.global_single_threaded.io());
        defer sf.deinit();
        return compatibleLlamaKindForSafetensors(&sf) orelse error.UnsupportedModel;
    }
    return error.UnsupportedModel;
}

fn selectLoadPathModelKind(kind: u32, path: []const u8) !u32 {
    return switch (kind) {
        auto_kind => selectAutoModelKind(path),
        tiny_llama_kind => blk: {
            const selected = try selectAutoModelKind(path);
            if (selected != tiny_llama_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        tiny_llama_2layer_kind => blk: {
            const selected = try selectAutoModelKind(path);
            if (selected != tiny_llama_2layer_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        smollm_135m_kind => blk: {
            const selected = try selectAutoModelKind(path);
            if (selected != smollm_135m_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        tiny_linear_kind, tiny_mlp_kind => error.UnsupportedModel,
        else => error.InvalidArgument,
    };
}

fn selectSafetensorsHeaderModelKind(kind: u32, header: []const u8) !u32 {
    return switch (kind) {
        auto_kind => compatibleLlamaKindForSafetensorsHeader(header) orelse error.UnsupportedModel,
        tiny_llama_kind => blk: {
            const selected = compatibleLlamaKindForSafetensorsHeader(header) orelse return error.UnsupportedModel;
            if (selected != tiny_llama_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        tiny_llama_2layer_kind => blk: {
            const selected = compatibleLlamaKindForSafetensorsHeader(header) orelse return error.UnsupportedModel;
            if (selected != tiny_llama_2layer_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        smollm_135m_kind => blk: {
            const selected = compatibleLlamaKindForSafetensorsHeader(header) orelse return error.UnsupportedModel;
            if (selected != smollm_135m_kind) return error.UnsupportedModel;
            break :blk selected;
        },
        tiny_linear_kind, tiny_mlp_kind => error.UnsupportedModel,
        else => error.InvalidArgument,
    };
}

fn selectSafetensorsDataModelKind(kind: u32, data: []const u8) !u32 {
    const header = try safetensorsHeaderFromData(data);
    return selectSafetensorsHeaderModelKind(kind, header);
}

fn modelLoadPathStatus(err: anyerror) c_int {
    return switch (err) {
        error.UnsupportedModel => status(.unsupported),
        error.InvalidArgument => status(.invalid_argument),
        else => compileErrorStatus(err),
    };
}

fn compileTinyLinearProgram(m: *const TinyLinearModelHandle, backend: llm_mod.LlamaBackend) !TinyLinearProgramHandle {
    var graph = GraphF32.init(alloc);
    errdefer graph.deinit();
    const graph_alloc = graph.allocator();

    const input = try TensorF32.init(graph_alloc, &.{ m.input_len, 1 });
    const weights = try TensorF32.init(graph_alloc, &.{ m.output_len, m.input_len });
    const bias = try TensorF32.init(graph_alloc, &.{m.output_len});
    @memset(weights.data, 0);
    @memset(bias.data, 0);

    const y = nn.linear(f32, input, weights, bias);
    try graph.infer(y);

    var cpu = cpu_mod.CpuBackend{};
    var stencil = stencil_mod.StencilBackend.webgpuResourceSessionProbe();
    const backend_id: u32 = switch (backend) {
        .auto, .cpu => backend_cpu,
        .webgpu => backend_webgpu,
        .metal => return error.Unsupported,
    };
    var program = switch (backend_id) {
        backend_cpu => try DeviceF32.Program.compile(.{
            .graph = &graph,
            .be = cpu.backend(),
            .alloc = alloc,
            .input_tensors = &.{input},
            .persistent_tensors = &.{ weights, bias },
            .output_tensors = &.{y},
        }),
        backend_webgpu => blk: {
            if (comptime build_options.use_wgpu) {
                var wgpu = wgpu_mod.WgpuBackend.init(alloc);
                break :blk try DeviceF32.Program.compile(.{
                    .graph = &graph,
                    .be = wgpu.backend(),
                    .alloc = alloc,
                    .input_tensors = &.{input},
                    .persistent_tensors = &.{ weights, bias },
                    .output_tensors = &.{y},
                });
            }
            break :blk try DeviceF32.Program.compile(.{
                .graph = &graph,
                .be = stencil.backend(),
                .alloc = alloc,
                .input_tensors = &.{input},
                .persistent_tensors = &.{ weights, bias },
                .output_tensors = &.{y},
            });
        },
        else => unreachable,
    };
    errdefer program.deinit();
    const inspection = program.inspect();

    return .{
        .input_len = m.input_len,
        .output_len = m.output_len,
        .backend = backend_id,
        .execution_supported = inspection.execution_supported,
        .graph = graph,
        .program = program,
        .input = input,
        .weights = weights,
        .bias = bias,
        .output = y,
    };
}

fn compileTinyMlpProgram(m: *const TinyMlpModelHandle, backend: llm_mod.LlamaBackend) !TinyMlpProgramHandle {
    var graph = GraphF32.init(alloc);
    errdefer graph.deinit();
    const graph_alloc = graph.allocator();

    const input = try TensorF32.init(graph_alloc, &.{ m.input_len, 1 });
    const w0 = try TensorF32.init(graph_alloc, &.{ m.hidden_len, m.input_len });
    const b0 = try TensorF32.init(graph_alloc, &.{m.hidden_len});
    const w1 = try TensorF32.init(graph_alloc, &.{ m.output_len, m.hidden_len });
    const b1 = try TensorF32.init(graph_alloc, &.{m.output_len});
    @memset(w0.data, 0);
    @memset(b0.data, 0);
    @memset(w1.data, 0);
    @memset(b1.data, 0);

    const hidden = tinyMlpActivate(nn.linear(f32, input, w0, b0), m.activation);
    const y = nn.linear(f32, hidden, w1, b1);
    try graph.infer(y);

    var cpu = cpu_mod.CpuBackend{};
    var stencil = stencil_mod.StencilBackend.webgpuResourceSessionProbe();
    const backend_id: u32 = switch (backend) {
        .auto, .cpu => backend_cpu,
        .webgpu => backend_webgpu,
        .metal => return error.Unsupported,
    };
    var program = switch (backend_id) {
        backend_cpu => try DeviceF32.Program.compile(.{
            .graph = &graph,
            .be = cpu.backend(),
            .alloc = alloc,
            .input_tensors = &.{input},
            .persistent_tensors = &.{ w0, b0, w1, b1 },
            .output_tensors = &.{y},
        }),
        backend_webgpu => blk: {
            if (comptime build_options.use_wgpu) {
                var wgpu = wgpu_mod.WgpuBackend.init(alloc);
                break :blk try DeviceF32.Program.compile(.{
                    .graph = &graph,
                    .be = wgpu.backend(),
                    .alloc = alloc,
                    .input_tensors = &.{input},
                    .persistent_tensors = &.{ w0, b0, w1, b1 },
                    .output_tensors = &.{y},
                });
            }
            break :blk try DeviceF32.Program.compile(.{
                .graph = &graph,
                .be = stencil.backend(),
                .alloc = alloc,
                .input_tensors = &.{input},
                .persistent_tensors = &.{ w0, b0, w1, b1 },
                .output_tensors = &.{y},
            });
        },
        else => unreachable,
    };
    errdefer program.deinit();
    const inspection = program.inspect();

    return .{
        .input_len = m.input_len,
        .hidden_len = m.hidden_len,
        .output_len = m.output_len,
        .activation = m.activation,
        .backend = backend_id,
        .execution_supported = inspection.execution_supported,
        .graph = graph,
        .program = program,
        .input = input,
        .w0 = w0,
        .b0 = b0,
        .w1 = w1,
        .b1 = b1,
        .output = y,
    };
}

export fn zgml_status_name(code: c_int) [*:0]const u8 {
    return switch (code) {
        status(.ok) => "ok",
        status(.invalid_argument) => "invalid_argument",
        status(.out_of_memory) => "out_of_memory",
        status(.shape_mismatch) => "shape_mismatch",
        status(.compile_failed) => "compile_failed",
        status(.unsupported) => "unsupported",
        else => "unknown",
    };
}

export fn zgml_model_create(desc_ptr: ?*const zgml_model_desc, out_model: ?*?*zgml_model) c_int {
    clearModel(out_model);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_model orelse return status(.invalid_argument);

    const handle = alloc.create(ModelHandle) catch return status(.out_of_memory);
    handle.* = switch (desc.kind) {
        tiny_linear_kind => blk: {
            _ = checkedWeightLen(desc.input_len, desc.output_len) orelse {
                alloc.destroy(handle);
                return status(.shape_mismatch);
            };
            break :blk .{ .data = .{ .tiny_linear = .{
                .input_len = desc.input_len,
                .output_len = desc.output_len,
            } } };
        },
        tiny_mlp_kind => blk: {
            const activation = tinyMlpActivationFromId(desc.activation) orelse {
                alloc.destroy(handle);
                return status(.invalid_argument);
            };
            _ = checkedTinyMlpWeightLens(desc.input_len, desc.hidden_len, desc.output_len) orelse {
                alloc.destroy(handle);
                return status(.shape_mismatch);
            };
            _ = checkedTinyMlpBiasLen(desc.hidden_len, desc.output_len) orelse {
                alloc.destroy(handle);
                return status(.shape_mismatch);
            };
            break :blk .{ .data = .{ .tiny_mlp = .{
                .input_len = desc.input_len,
                .hidden_len = desc.hidden_len,
                .output_len = desc.output_len,
                .activation = activation,
            } } };
        },
        tiny_llama_kind => blk: {
            const model = TinyLlamaModel.init(alloc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama = .{ .model = model } } };
        },
        tiny_llama_2layer_kind => blk: {
            const model = TinyLlama2LayerModel.init(alloc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama_2layer = .{ .model = model } } };
        },
        smollm_135m_kind => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        else => {
            alloc.destroy(handle);
            return status(.invalid_argument);
        },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_model_load_path(desc_ptr: ?*const zgml_model_load_desc, out_model: ?*?*zgml_model) c_int {
    clearModel(out_model);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_model orelse return status(.invalid_argument);
    if (desc.path == null or desc.path_len == 0) return status(.invalid_argument);
    const path = desc.path.?[0..desc.path_len];
    const kind = selectLoadPathModelKind(desc.kind, path) catch |err| return modelLoadPathStatus(err);

    const handle = alloc.create(ModelHandle) catch return status(.out_of_memory);
    handle.* = switch (kind) {
        tiny_llama_kind => blk: {
            const model = TinyLlamaModel.load(alloc, std.Io.Threaded.global_single_threaded.io(), path) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama = .{ .model = model } } };
        },
        tiny_llama_2layer_kind => blk: {
            const model = TinyLlama2LayerModel.load(alloc, std.Io.Threaded.global_single_threaded.io(), path) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama_2layer = .{ .model = model } } };
        },
        smollm_135m_kind => blk: {
            const model = SmolLM135MModel.load(alloc, std.Io.Threaded.global_single_threaded.io(), path) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .smollm_135m = .{ .model = model } } };
        },
        tiny_linear_kind, tiny_mlp_kind => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        else => {
            alloc.destroy(handle);
            return status(.invalid_argument);
        },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_model_load_safetensors_data(desc_ptr: ?*const zgml_safetensors_data_load_desc, out_model: ?*?*zgml_model) c_int {
    clearModel(out_model);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_model orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.data == null or desc.data_len == 0) return status(.invalid_argument);
    const bytes = desc.data.?[0..desc.data_len];
    const kind = selectSafetensorsDataModelKind(desc.kind, bytes) catch |err| return modelLoadPathStatus(err);

    const handle = alloc.create(ModelHandle) catch return status(.out_of_memory);
    handle.* = switch (kind) {
        tiny_llama_kind => blk: {
            const model = TinyLlamaModel.loadSafetensorsBytes(alloc, bytes) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama = .{ .model = model } } };
        },
        tiny_llama_2layer_kind => blk: {
            const model = TinyLlama2LayerModel.loadSafetensorsBytes(alloc, bytes) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .tiny_llama_2layer = .{ .model = model } } };
        },
        smollm_135m_kind => blk: {
            const model = SmolLM135MModel.loadSafetensorsBytes(alloc, bytes) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            break :blk .{ .data = .{ .smollm_135m = .{ .model = model } } };
        },
        tiny_linear_kind, tiny_mlp_kind => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        else => {
            alloc.destroy(handle);
            return status(.invalid_argument);
        },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_model_probe_path(desc_ptr: ?*const zgml_model_load_desc, out_inspection: ?*zgml_model_inspection) c_int {
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.path == null or desc.path_len == 0) return status(.invalid_argument);
    const path = desc.path.?[0..desc.path_len];
    const kind = selectLoadPathModelKind(desc.kind, path) catch |err| return modelLoadPathStatus(err);
    switch (kind) {
        tiny_llama_kind => fillLlamaModelInspection(out, tiny_llama_kind, tiny_llama_config),
        tiny_llama_2layer_kind => fillLlamaModelInspection(out, tiny_llama_2layer_kind, tiny_llama_2layer_config),
        smollm_135m_kind => fillLlamaModelInspection(out, smollm_135m_kind, smollm_135m_config),
        else => return status(.unsupported),
    }
    return status(.ok);
}

export fn zgml_model_probe_safetensors_data(desc_ptr: ?*const zgml_safetensors_data_load_desc, out_inspection: ?*zgml_model_inspection) c_int {
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.data == null or desc.data_len == 0) return status(.invalid_argument);
    const bytes = desc.data.?[0..desc.data_len];
    const kind = selectSafetensorsDataModelKind(desc.kind, bytes) catch |err| return modelLoadPathStatus(err);
    switch (kind) {
        tiny_llama_kind => fillLlamaModelInspection(out, tiny_llama_kind, tiny_llama_config),
        tiny_llama_2layer_kind => fillLlamaModelInspection(out, tiny_llama_2layer_kind, tiny_llama_2layer_config),
        smollm_135m_kind => fillLlamaModelInspection(out, smollm_135m_kind, smollm_135m_config),
        else => return status(.unsupported),
    }
    return status(.ok);
}

export fn zgml_model_probe_safetensors_header(desc_ptr: ?*const zgml_safetensors_header_probe_desc, out_inspection: ?*zgml_model_inspection) c_int {
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.header == null or desc.header_len == 0) return status(.invalid_argument);
    const header = desc.header.?[0..desc.header_len];
    const kind = selectSafetensorsHeaderModelKind(desc.kind, header) catch |err| return modelLoadPathStatus(err);
    switch (kind) {
        tiny_llama_kind => fillLlamaModelInspection(out, tiny_llama_kind, tiny_llama_config),
        tiny_llama_2layer_kind => fillLlamaModelInspection(out, tiny_llama_2layer_kind, tiny_llama_2layer_config),
        smollm_135m_kind => fillLlamaModelInspection(out, smollm_135m_kind, smollm_135m_config),
        else => return status(.unsupported),
    }
    return status(.ok);
}

export fn zgml_supported_checkpoint_count() usize {
    return compatible_llama_families.len;
}

export fn zgml_supported_checkpoint_inspect(index: usize, out_inspection: ?*zgml_model_inspection) c_int {
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    if (index >= compatible_llama_families.len) return status(.invalid_argument);
    const family = compatible_llama_families[index];
    fillLlamaModelInspection(out, family.kind, family.config);
    return status(.ok);
}

export fn zgml_model_inspect(model: ?*zgml_model, out_inspection: ?*zgml_model_inspection) c_int {
    const m = modelHandle(model) orelse return status(.invalid_argument);
    const out = out_inspection orelse return status(.invalid_argument);
    switch (m.data) {
        .tiny_linear => |linear| {
            out.* = .{
                .model_kind = tiny_linear_kind,
                .input_len = @intCast(linear.input_len),
                .output_len = @intCast(linear.output_len),
            };
        },
        .tiny_mlp => |mlp| {
            out.* = .{
                .model_kind = tiny_mlp_kind,
                .input_len = @intCast(mlp.input_len),
                .output_len = @intCast(mlp.output_len),
                .d_model = @intCast(mlp.hidden_len),
                .n_layers = 2,
            };
        },
        .tiny_llama => fillLlamaModelInspection(out, tiny_llama_kind, tiny_llama_config),
        .tiny_llama_2layer => fillLlamaModelInspection(out, tiny_llama_2layer_kind, tiny_llama_2layer_config),
        .smollm_135m => fillLlamaModelInspection(out, smollm_135m_kind, smollm_135m_config),
    }
    return status(.ok);
}

export fn zgml_program_compile(model: ?*zgml_model, desc_ptr: ?*const zgml_compile_desc, out_program: ?*?*zgml_program) c_int {
    clearProgram(out_program);
    const m = modelHandle(model) orelse return status(.invalid_argument);
    const out = out_program orelse return status(.invalid_argument);
    const backend = compileBackend(desc_ptr) catch return status(.invalid_argument);

    const handle = alloc.create(ProgramHandle) catch return status(.out_of_memory);
    handle.* = switch (m.data) {
        .tiny_linear => |*linear| blk: {
            break :blk .{ .data = .{ .tiny_linear = compileTinyLinearProgram(linear, backend) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            } } };
        },
        .tiny_mlp => |*mlp| blk: {
            break :blk .{ .data = .{ .tiny_mlp = compileTinyMlpProgram(mlp, backend) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            } } };
        },
        .tiny_llama => |*llama| blk: {
            const program = llama.model.compile(llamaCompileOptions(desc_ptr, backend, tiny_llama_config.max_seq_len)) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            m.retain();
            break :blk .{ .data = .{ .tiny_llama = .{ .model = m, .program = program } } };
        },
        .tiny_llama_2layer => |*llama| blk: {
            const program = llama.model.compile(llamaCompileOptions(desc_ptr, backend, tiny_llama_2layer_config.max_seq_len)) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            m.retain();
            break :blk .{ .data = .{ .tiny_llama_2layer = .{ .model = m, .program = program } } };
        },
        .smollm_135m => |*llama| blk: {
            const program = llama.model.compile(llamaCompileOptions(desc_ptr, backend, smollm_135m_config.max_seq_len)) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            m.retain();
            break :blk .{ .data = .{ .smollm_135m = .{ .model = m, .program = program } } };
        },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_module_program_compile(module_desc: ?*const zgml_module_desc, compile_desc: ?*const zgml_compile_desc, out_program: ?*?*zgml_program) c_int {
    clearProgram(out_program);
    const desc = module_desc orelse return status(.invalid_argument);
    const out = out_program orelse return status(.invalid_argument);
    const backend = compileBackend(compile_desc) catch return status(.invalid_argument);

    const handle = alloc.create(ProgramHandle) catch return status(.out_of_memory);
    handle.* = .{ .data = .{ .module = compileModuleProgram(desc, backend) catch |err| {
        alloc.destroy(handle);
        return compileErrorStatus(err);
    } } };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_module_program_get_requirements(module_desc: ?*const zgml_module_desc, compile_desc: ?*const zgml_compile_desc, out_requirements: ?*zgml_program_requirements) c_int {
    const desc = module_desc orelse return status(.invalid_argument);
    const out = out_requirements orelse return status(.invalid_argument);
    out.* = .{};
    const backend = compileBackend(compile_desc) catch return status(.invalid_argument);
    var module = compileModuleProgram(desc, backend) catch |err| return compileErrorStatus(err);
    defer module.deinit();
    fillModuleRequirements(out, &module);
    return status(.ok);
}

fn fillTinyLinearRequirements(out: *zgml_program_requirements, linear: *TinyLinearProgramHandle) void {
    const weights_len = checkedWeightLen(linear.input_len, linear.output_len) orelse 0;
    const bias_len = linear.output_len;
    out.* = .{
        .model_kind = tiny_linear_kind,
        .scalar_bytes = @sizeOf(f32),
        .token_id_bytes = @sizeOf(u32),
        .input_len = linear.input_len,
        .input_byte_len = f32ByteLen(linear.input_len),
        .output_len = linear.output_len,
        .weights_len = weights_len,
        .weights_byte_len = f32ByteLen(weights_len),
        .bias_len = bias_len,
        .bias_byte_len = f32ByteLen(bias_len),
        .parameter_len = weights_len + bias_len,
        .parameter_byte_len = f32ByteLen(weights_len + bias_len),
        .logits_len = 0,
        .output_byte_len = f32ByteLen(linear.output_len),
        .context_len = 0,
        .batch = 0,
        .max_token_window = 0,
    };
}

fn fillTinyMlpRequirements(out: *zgml_program_requirements, mlp: *TinyMlpProgramHandle) void {
    const weights_len = if (checkedTinyMlpWeightLens(mlp.input_len, mlp.hidden_len, mlp.output_len)) |weights| weights.total else 0;
    const bias_len = checkedTinyMlpBiasLen(mlp.hidden_len, mlp.output_len) orelse 0;
    out.* = .{
        .model_kind = tiny_mlp_kind,
        .scalar_bytes = @sizeOf(f32),
        .token_id_bytes = @sizeOf(u32),
        .input_len = mlp.input_len,
        .input_byte_len = f32ByteLen(mlp.input_len),
        .output_len = mlp.output_len,
        .weights_len = weights_len,
        .weights_byte_len = f32ByteLen(weights_len),
        .bias_len = bias_len,
        .bias_byte_len = f32ByteLen(bias_len),
        .parameter_len = weights_len + bias_len,
        .parameter_byte_len = f32ByteLen(weights_len + bias_len),
        .logits_len = 0,
        .output_byte_len = f32ByteLen(mlp.output_len),
        .context_len = 0,
        .batch = 0,
        .max_token_window = 0,
    };
}

fn fillModuleRequirements(out: *zgml_program_requirements, module: *ModuleProgramHandle) void {
    out.* = .{
        .model_kind = module_kind,
        .scalar_bytes = @sizeOf(f32),
        .token_id_bytes = @sizeOf(u32),
        .input_len = module.input_len,
        .input_byte_len = f32ByteLen(module.input_len),
        .output_len = module.output_len,
        .weights_len = module.weights_len,
        .weights_byte_len = f32ByteLen(module.weights_len),
        .bias_len = module.bias_len,
        .bias_byte_len = f32ByteLen(module.bias_len),
        .parameter_len = module.weights_len + module.bias_len,
        .parameter_byte_len = f32ByteLen(module.weights_len + module.bias_len),
        .logits_len = 0,
        .output_byte_len = f32ByteLen(module.output_len),
        .context_len = 0,
        .batch = 0,
        .max_token_window = 0,
    };
}

fn f32ByteLen(element_count: usize) usize {
    return std.math.mul(usize, element_count, @sizeOf(f32)) catch 0;
}

fn fillLlamaRequirements(out: *zgml_program_requirements, model_kind: u32, inspection: llm_mod.LlamaProgramInspection) void {
    out.* = .{
        .model_kind = model_kind,
        .scalar_bytes = @sizeOf(f32),
        .token_id_bytes = @sizeOf(u32),
        .input_len = 0,
        .input_byte_len = 0,
        .output_len = inspection.vocab_size,
        .weights_len = 0,
        .weights_byte_len = 0,
        .bias_len = 0,
        .bias_byte_len = 0,
        .parameter_len = 0,
        .parameter_byte_len = 0,
        .logits_len = inspection.vocab_size,
        .output_byte_len = f32ByteLen(inspection.vocab_size),
        .context_len = inspection.context_len,
        .batch = inspection.batch,
        .max_token_window = inspection.context_len,
    };
}

fn fillLlamaKvCacheRequirements(
    comptime config: llm_mod.LlamaConfig,
    out: *zgml_llama_kv_cache_requirements,
    model_kind: u32,
    inspection: llm_mod.LlamaProgramInspection,
) !void {
    const element_count = try llamaKvCacheElementCount(config, inspection.context_len);
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    out.* = .{
        .model_kind = model_kind,
        .scalar_bytes = @sizeOf(f32),
        .n_layers = @intCast(config.n_layers),
        .context_len = inspection.context_len,
        .k_buffer_byte_len = byte_len,
        .v_buffer_byte_len = byte_len,
        .buffer_byte_len = byte_len,
    };
}

export fn zgml_program_get_requirements(program: ?*zgml_program, out_requirements: ?*zgml_program_requirements) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_requirements orelse return status(.invalid_argument);
    out.* = .{};
    switch (p.data) {
        .tiny_linear => |*linear| fillTinyLinearRequirements(out, linear),
        .tiny_mlp => |*mlp| fillTinyMlpRequirements(out, mlp),
        .module => |*module| fillModuleRequirements(out, module),
        .tiny_llama => |*llama| fillLlamaRequirements(out, tiny_llama_kind, llama.program.inspect()),
        .tiny_llama_2layer => |*llama| fillLlamaRequirements(out, tiny_llama_2layer_kind, llama.program.inspect()),
        .smollm_135m => |*llama| fillLlamaRequirements(out, smollm_135m_kind, llama.program.inspect()),
    }
    return status(.ok);
}

export fn zgml_program_check_model_compatibility(program: ?*zgml_program, model: ?*zgml_model, out_compatibility: ?*zgml_program_model_compatibility) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const m = modelHandle(model) orelse return status(.invalid_argument);
    const out = out_compatibility orelse return status(.invalid_argument);
    out.* = .{
        .program_model_kind = programModelKindForHandle(p),
        .model_kind = modelKindForHandle(m),
        .compatible = @intFromBool(programAcceptsModel(p, m)),
    };
    return status(.ok);
}

export fn zgml_llama_program_get_kv_cache_requirements(program: ?*zgml_program, out_requirements: ?*zgml_llama_kv_cache_requirements) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_requirements orelse return status(.invalid_argument);
    out.* = .{};
    return switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => status(.unsupported),
        .tiny_llama => |*llama| blk: {
            fillLlamaKvCacheRequirements(tiny_llama_config, out, tiny_llama_kind, llama.program.inspect()) catch |err| return compileErrorStatus(err);
            break :blk status(.ok);
        },
        .tiny_llama_2layer => |*llama| blk: {
            fillLlamaKvCacheRequirements(tiny_llama_2layer_config, out, tiny_llama_2layer_kind, llama.program.inspect()) catch |err| return compileErrorStatus(err);
            break :blk status(.ok);
        },
        .smollm_135m => |*llama| blk: {
            fillLlamaKvCacheRequirements(smollm_135m_config, out, smollm_135m_kind, llama.program.inspect()) catch |err| return compileErrorStatus(err);
            break :blk status(.ok);
        },
    };
}

fn fillProgramRequirementsForHandle(p: *ProgramHandle, out: *zgml_program_requirements) void {
    switch (p.data) {
        .tiny_linear => |*linear| fillTinyLinearRequirements(out, linear),
        .tiny_mlp => |*mlp| fillTinyMlpRequirements(out, mlp),
        .module => |*module| fillModuleRequirements(out, module),
        .tiny_llama => |*llama| fillLlamaRequirements(out, tiny_llama_kind, llama.program.inspect()),
        .tiny_llama_2layer => |*llama| fillLlamaRequirements(out, tiny_llama_2layer_kind, llama.program.inspect()),
        .smollm_135m => |*llama| fillLlamaRequirements(out, smollm_135m_kind, llama.program.inspect()),
    }
}

fn programBufferByteLen(requirements: zgml_program_requirements, kind: u32) !usize {
    return switch (kind) {
        program_buffer_weights => requirements.weights_byte_len,
        program_buffer_bias => requirements.bias_byte_len,
        program_buffer_input => requirements.input_byte_len,
        program_buffer_output => requirements.output_byte_len,
        else => error.InvalidArgument,
    };
}

fn programBufferByteLenForHandle(p: *ProgramHandle, kind: u32) !usize {
    switch (kind) {
        program_buffer_weights, program_buffer_bias, program_buffer_input, program_buffer_output => {
            var requirements = zgml_program_requirements{};
            fillProgramRequirementsForHandle(p, &requirements);
            return programBufferByteLen(requirements, kind);
        },
        program_buffer_llama_k_cache, program_buffer_llama_v_cache => return switch (p.data) {
            .tiny_linear, .tiny_mlp, .module => error.Unsupported,
            .tiny_llama => |*llama| blk: {
                var requirements = zgml_llama_kv_cache_requirements{};
                try fillLlamaKvCacheRequirements(tiny_llama_config, &requirements, tiny_llama_kind, llama.program.inspect());
                break :blk if (kind == program_buffer_llama_k_cache) requirements.k_buffer_byte_len else requirements.v_buffer_byte_len;
            },
            .tiny_llama_2layer => |*llama| blk: {
                var requirements = zgml_llama_kv_cache_requirements{};
                try fillLlamaKvCacheRequirements(tiny_llama_2layer_config, &requirements, tiny_llama_2layer_kind, llama.program.inspect());
                break :blk if (kind == program_buffer_llama_k_cache) requirements.k_buffer_byte_len else requirements.v_buffer_byte_len;
            },
            .smollm_135m => |*llama| blk: {
                var requirements = zgml_llama_kv_cache_requirements{};
                try fillLlamaKvCacheRequirements(smollm_135m_config, &requirements, smollm_135m_kind, llama.program.inspect());
                break :blk if (kind == program_buffer_llama_k_cache) requirements.k_buffer_byte_len else requirements.v_buffer_byte_len;
            },
        },
        else => return error.InvalidArgument,
    }
}

export fn zgml_program_create_buffer(program: ?*zgml_program, kind: u32, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    const byte_len = programBufferByteLenForHandle(p, kind) catch |err| return compileErrorStatus(err);
    if (byte_len == 0) return status(.invalid_argument);
    return createBuffer(byte_len, out);
}

export fn zgml_program_create_device_buffer(program: ?*zgml_program, kind: u32, placement: u32, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    const device = deviceFromBackendId(placement) orelse return status(.invalid_argument);
    if (device != .webgpu) return status(.unsupported);
    const access = programDeviceBufferAccess(kind) orelse return status(.invalid_argument);
    const byte_len = programBufferByteLenForHandle(p, kind) catch |err| return compileErrorStatus(err);
    if (byte_len == 0) return status(.invalid_argument);

    const resource = switch (p.data) {
        .tiny_linear => |*linear| blk: {
            if (!build_options.use_wgpu or linear.backend != backend_webgpu or !linear.execution_supported) return status(.unsupported);
            break :blk wgpu_mod.createDeviceBuffer(linear.program.handle, byte_len, access) catch |err| return compileErrorStatus(err);
        },
        .tiny_mlp => return status(.unsupported),
        .module => |*module| blk: {
            if (!build_options.use_wgpu or module.backend != backend_webgpu or !module.execution_supported) return status(.unsupported);
            break :blk wgpu_mod.createDeviceBuffer(module.program.handle, byte_len, access) catch |err| return compileErrorStatus(err);
        },
        .tiny_llama => |*llama| llama.program.createDeviceBuffer(byte_len, access) catch |err| return compileErrorStatus(err),
        .tiny_llama_2layer => |*llama| llama.program.createDeviceBuffer(byte_len, access) catch |err| return compileErrorStatus(err),
        .smollm_135m => |*llama| llama.program.createDeviceBuffer(byte_len, access) catch |err| return compileErrorStatus(err),
    };

    const handle = alloc.create(BufferHandle) catch {
        releaseDeviceBuffer(p, resource);
        return status(.out_of_memory);
    };
    p.retain();
    handle.* = .{
        .external = .{
            .resource = resource,
            .view_byte_len = @intCast(byte_len),
        },
        .device_owner = .{ .program = p },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_program_get_device_handle(program: ?*zgml_program, placement: u32, out_handle: ?*usize) c_int {
    if (out_handle) |out| out.* = 0;
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_handle orelse return status(.invalid_argument);
    out.* = getDeviceHandle(p, placement) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_program_import_device_buffer(program: ?*zgml_program, kind: u32, desc_ptr: ?*const zgml_device_buffer_import_desc, out_buffer: ?*?*zgml_buffer) c_int {
    clearBuffer(out_buffer);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const out = out_buffer orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    const device = deviceFromBackendId(desc.placement) orelse return status(.invalid_argument);
    if (device != .webgpu) return status(.unsupported);
    const access = programDeviceBufferAccess(kind) orelse return status(.invalid_argument);
    const byte_len = programBufferByteLenForHandle(p, kind) catch |err| return compileErrorStatus(err);
    if (byte_len == 0) return status(.invalid_argument);
    if (desc.byte_len != 0 and desc.byte_len != byte_len) return status(.shape_mismatch);

    const resource = importDeviceBuffer(p, .{
        .device_handle = desc.device_handle,
        .buffer_handle = desc.buffer_handle,
        .byte_offset = desc.byte_offset,
        .byte_len = byte_len,
        .access = access,
    }) catch |err| return compileErrorStatus(err);
    const handle = alloc.create(BufferHandle) catch {
        releaseDeviceBuffer(p, resource);
        return status(.out_of_memory);
    };
    p.retain();
    handle.* = .{
        .external = .{
            .resource = resource,
            .view_byte_len = @intCast(byte_len),
        },
        .device_owner = .{ .program = p },
    };
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_program_create_output_buffer(program: ?*zgml_program, out_buffer: ?*?*zgml_buffer) c_int {
    return zgml_program_create_buffer(program, program_buffer_output, out_buffer);
}

export fn zgml_program_inspect(program: ?*zgml_program, out_inspection: ?*zgml_program_inspection) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    return switch (p.data) {
        .tiny_linear => |*linear| blk: {
            const inspection = linear.program.inspect();
            out.* = .{
                .backend = linear.backend,
                .execution_supported = @intFromBool(inspection.execution_supported),
                .external_resources_supported = @intFromBool(inspection.external_resources_supported),
                .buffer_count = @intCast(inspection.buffer_count),
                .buffer_element_count = @intCast(inspection.buffer_element_count),
                .buffer_byte_len = @intCast(inspection.buffer_byte_len),
                .initial_upload_count = @intCast(inspection.initial_upload_count),
                .qweight_count = @intCast(inspection.qweight_count),
                .op_count = @intCast(inspection.op_count),
                .runtime_patch_max_cache_write_pos = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_cache_write_pos),
                .runtime_patch_max_attention_seq_kv = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_attention_seq_kv),
                .command_count = inspection.command_shape.command_count,
                .command_stencil_hash = inspection.command_shape.command_stencil_hash,
                .binding_requirement_hash = inspection.binding_requirement_hash,
                .persistent_requirement_count = @intCast(inspection.persistent_requirement_count),
                .step_input_requirement_count = @intCast(inspection.step_input_requirement_count),
                .step_output_requirement_count = @intCast(inspection.step_output_requirement_count),
                .runtime_patch_holes = inspection.runtime_patch_shape.runtime_patch_holes,
                .runtime_patch_cache_write_pos_holes = inspection.runtime_patch_shape.runtime_patch_cache_write_pos_holes,
                .runtime_patch_attention_seq_kv_holes = inspection.runtime_patch_shape.runtime_patch_attention_seq_kv_holes,
                .runtime_patch_stencil_hash = inspection.runtime_patch_shape.runtime_patch_stencil_hash,
            };
            fillCommandCategoryCounts(out, inspection.command_shape.categoryCounts());
            fillExecutionPlanInspection(out, inspection.execution_plan);
            break :blk status(.ok);
        },
        .tiny_mlp => |*mlp| blk: {
            const inspection = mlp.program.inspect();
            out.* = .{
                .backend = mlp.backend,
                .execution_supported = @intFromBool(inspection.execution_supported),
                .external_resources_supported = @intFromBool(false),
                .buffer_count = @intCast(inspection.buffer_count),
                .buffer_element_count = @intCast(inspection.buffer_element_count),
                .buffer_byte_len = @intCast(inspection.buffer_byte_len),
                .initial_upload_count = @intCast(inspection.initial_upload_count),
                .qweight_count = @intCast(inspection.qweight_count),
                .op_count = @intCast(inspection.op_count),
                .runtime_patch_max_cache_write_pos = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_cache_write_pos),
                .runtime_patch_max_attention_seq_kv = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_attention_seq_kv),
                .command_count = inspection.command_shape.command_count,
                .command_stencil_hash = inspection.command_shape.command_stencil_hash,
                .binding_requirement_hash = inspection.binding_requirement_hash,
                .persistent_requirement_count = @intCast(inspection.persistent_requirement_count),
                .step_input_requirement_count = @intCast(inspection.step_input_requirement_count),
                .step_output_requirement_count = @intCast(inspection.step_output_requirement_count),
                .runtime_patch_holes = inspection.runtime_patch_shape.runtime_patch_holes,
                .runtime_patch_cache_write_pos_holes = inspection.runtime_patch_shape.runtime_patch_cache_write_pos_holes,
                .runtime_patch_attention_seq_kv_holes = inspection.runtime_patch_shape.runtime_patch_attention_seq_kv_holes,
                .runtime_patch_stencil_hash = inspection.runtime_patch_shape.runtime_patch_stencil_hash,
            };
            fillCommandCategoryCounts(out, inspection.command_shape.categoryCounts());
            fillExecutionPlanInspection(out, inspection.execution_plan);
            break :blk status(.ok);
        },
        .module => |*module| blk: {
            const inspection = module.program.inspect();
            out.* = .{
                .backend = module.backend,
                .execution_supported = @intFromBool(inspection.execution_supported),
                .external_resources_supported = @intFromBool(inspection.external_resources_supported),
                .buffer_count = @intCast(inspection.buffer_count),
                .buffer_element_count = @intCast(inspection.buffer_element_count),
                .buffer_byte_len = @intCast(inspection.buffer_byte_len),
                .initial_upload_count = @intCast(inspection.initial_upload_count),
                .qweight_count = @intCast(inspection.qweight_count),
                .op_count = @intCast(inspection.op_count),
                .runtime_patch_max_cache_write_pos = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_cache_write_pos),
                .runtime_patch_max_attention_seq_kv = runtimePatchEnvelopeValue(inspection.runtime_patch_envelope.max_attention_seq_kv),
                .command_count = inspection.command_shape.command_count,
                .command_stencil_hash = inspection.command_shape.command_stencil_hash,
                .binding_requirement_hash = inspection.binding_requirement_hash,
                .persistent_requirement_count = @intCast(inspection.persistent_requirement_count),
                .step_input_requirement_count = @intCast(inspection.step_input_requirement_count),
                .step_output_requirement_count = @intCast(inspection.step_output_requirement_count),
                .runtime_patch_holes = inspection.runtime_patch_shape.runtime_patch_holes,
                .runtime_patch_cache_write_pos_holes = inspection.runtime_patch_shape.runtime_patch_cache_write_pos_holes,
                .runtime_patch_attention_seq_kv_holes = inspection.runtime_patch_shape.runtime_patch_attention_seq_kv_holes,
                .runtime_patch_stencil_hash = inspection.runtime_patch_shape.runtime_patch_stencil_hash,
            };
            fillCommandCategoryCounts(out, inspection.command_shape.categoryCounts());
            fillExecutionPlanInspection(out, inspection.execution_plan);
            break :blk status(.ok);
        },
        .tiny_llama => |*llama| blk: {
            fillProgramInspection(out, llama.program.inspectExecutable());
            break :blk status(.ok);
        },
        .tiny_llama_2layer => |*llama| blk: {
            fillProgramInspection(out, llama.program.inspectExecutable());
            break :blk status(.ok);
        },
        .smollm_135m => |*llama| blk: {
            fillProgramInspection(out, llama.program.inspectExecutable());
            break :blk status(.ok);
        },
    };
}

export fn zgml_llama_program_inspect(program: ?*zgml_program, out_inspection: ?*zgml_llama_program_inspection) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    return switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => status(.unsupported),
        .tiny_llama => |*llama| blk: {
            fillLlamaProgramInspection(out, llama.program.inspect());
            break :blk status(.ok);
        },
        .tiny_llama_2layer => |*llama| blk: {
            fillLlamaProgramInspection(out, llama.program.inspect());
            break :blk status(.ok);
        },
        .smollm_135m => |*llama| blk: {
            fillLlamaProgramInspection(out, llama.program.inspect());
            break :blk status(.ok);
        },
    };
}

export fn zgml_program_runtime_profile(program: ?*zgml_program, out_profile: ?*zgml_runtime_profile) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_profile orelse return status(.invalid_argument);
    var rt = profile_mod.RuntimeProfile{};
    addProgramRuntimeProfileTo(p, &rt);
    fillRuntimeProfile(out, rt);
    return status(.ok);
}

export fn zgml_program_reset_runtime_profile(program: ?*zgml_program) c_int {
    const p = programHandle(program) orelse return status(.invalid_argument);
    resetProgramRuntimeProfile(p);
    return status(.ok);
}

export fn zgml_session_bind(program: ?*zgml_program, desc_ptr: ?*const zgml_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const out = out_session orelse return status(.invalid_argument);
    const handle = alloc.create(SessionHandle) catch return status(.out_of_memory);

    switch (p.data) {
        .tiny_linear => |*linear| {
            const desc = desc_ptr orelse {
                alloc.destroy(handle);
                return status(.invalid_argument);
            };
            handle.* = .{ .data = .{ .tiny_linear = bindTinyLinearSession(p, linear, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            } } };
        },
        .tiny_mlp => |*mlp| {
            const desc = desc_ptr orelse {
                alloc.destroy(handle);
                return status(.invalid_argument);
            };
            handle.* = .{ .data = .{ .tiny_mlp = bindTinyMlpSession(p, mlp, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            } } };
        },
        .module => |*module| {
            const desc = desc_ptr orelse {
                alloc.destroy(handle);
                return status(.invalid_argument);
            };
            handle.* = .{ .data = .{ .module = bindModuleSession(p, module, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            } } };
        },
        .tiny_llama => |*llama| {
            const output_buf = parseLlamaBoundOutput(desc_ptr, tiny_llama_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindExecutableDecode(.{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
        .tiny_llama_2layer => |*llama| {
            const output_buf = parseLlamaBoundOutput(desc_ptr, tiny_llama_2layer_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindExecutableDecode(.{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama_2layer = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
        .smollm_135m => |*llama| {
            const output_buf = parseLlamaBoundOutput(desc_ptr, smollm_135m_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindExecutableDecode(.{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .smollm_135m = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
    }

    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_session_bind_model(program: ?*zgml_program, model: ?*zgml_model, desc_ptr: ?*const zgml_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const m = modelHandle(model) orelse return status(.invalid_argument);
    const out = out_session orelse return status(.invalid_argument);
    const handle = alloc.create(SessionHandle) catch return status(.out_of_memory);

    switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        .tiny_llama => |*llama| {
            const source = switch (m.data) {
                .tiny_llama => |*source| source.model,
                .tiny_linear, .tiny_mlp, .tiny_llama_2layer, .smollm_135m => {
                    alloc.destroy(handle);
                    return status(.shape_mismatch);
                },
            };
            const output_buf = parseLlamaBoundOutput(desc_ptr, tiny_llama_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindModel(source, .{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
        .tiny_llama_2layer => |*llama| {
            const source = switch (m.data) {
                .tiny_llama_2layer => |*source| source.model,
                .tiny_linear, .tiny_mlp, .tiny_llama, .smollm_135m => {
                    alloc.destroy(handle);
                    return status(.shape_mismatch);
                },
            };
            const output_buf = parseLlamaBoundOutput(desc_ptr, tiny_llama_2layer_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindModel(source, .{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama_2layer = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
        .smollm_135m => |*llama| {
            const source = switch (m.data) {
                .smollm_135m => |*source| source.model,
                .tiny_linear, .tiny_mlp, .tiny_llama, .tiny_llama_2layer => {
                    alloc.destroy(handle);
                    return status(.shape_mismatch);
                },
            };
            const output_buf = parseLlamaBoundOutput(desc_ptr, smollm_135m_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session = llama.program.bindModel(source, .{}) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .smollm_135m = .{ .program = p, .session = session, .output_buf = output_buf } } };
        },
    }

    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_session_bind_model_buffers(program: ?*zgml_program, model: ?*zgml_model, desc_ptr: ?*const zgml_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const m = modelHandle(model) orelse return status(.invalid_argument);
    switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => {},
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => return bindLlamaSessionBuffers(p, m, desc, out_session),
    }

    const bind_desc = zgml_bind_desc{
        .weights = bufferConstF32Ptr(desc.weights, desc.weights_len) catch |err| return compileErrorStatus(err),
        .weights_len = desc.weights_len,
        .bias = bufferConstF32Ptr(desc.bias, desc.bias_len) catch |err| return compileErrorStatus(err),
        .bias_len = desc.bias_len,
        .input = bufferConstF32Ptr(desc.input, desc.input_len) catch |err| return compileErrorStatus(err),
        .input_len = desc.input_len,
        .output = bufferMutF32Ptr(desc.output, desc.output_len) catch |err| return compileErrorStatus(err),
        .output_len = desc.output_len,
    };
    return zgml_session_bind_model(program, model, &bind_desc, out_session);
}

export fn zgml_session_bind_buffers(program: ?*zgml_program, desc_ptr: ?*const zgml_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const p = programHandle(program) orelse return status(.invalid_argument);

    switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => {},
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => return bindLlamaSessionBuffers(p, null, desc, out_session),
    }

    if (switch (p.data) {
        .tiny_mlp => bufferBindDescHasExternalResource(desc),
        .tiny_linear, .module => false,
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => unreachable,
    }) return status(.unsupported);

    const out = out_session orelse return status(.invalid_argument);
    const handle = alloc.create(SessionHandle) catch return status(.out_of_memory);

    switch (p.data) {
        .tiny_linear => |*linear| {
            const session = bindTinyLinearBufferSession(p, linear, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            handle.* = .{ .data = .{ .tiny_linear = session } };
        },
        .tiny_mlp => |*mlp| {
            const session = bindTinyMlpBufferSession(p, mlp, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            handle.* = .{ .data = .{ .tiny_mlp = session } };
        },
        .module => |*module| {
            const session = bindModuleBufferSession(p, module, desc) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            handle.* = .{ .data = .{ .module = session } };
        },
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => unreachable,
    }
    out.* = @ptrCast(handle);
    return status(.ok);
}

export fn zgml_llama_session_bind_model_buffers(program: ?*zgml_program, model: ?*zgml_model, desc_ptr: ?*const zgml_llama_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const p = programHandle(program) orelse return status(.invalid_argument);
    const m = modelHandle(model) orelse return status(.invalid_argument);
    return bindLlamaSessionLlamaBuffers(p, m, desc_ptr, out_session);
}

export fn zgml_llama_session_bind_buffers(program: ?*zgml_program, desc_ptr: ?*const zgml_llama_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    clearSession(out_session);
    const p = programHandle(program) orelse return status(.invalid_argument);
    return bindLlamaSessionLlamaBuffers(p, null, desc_ptr, out_session);
}

fn parseLlamaBoundOutput(desc_ptr: ?*const zgml_bind_desc, vocab_size: usize) ![]f32 {
    const desc = desc_ptr orelse return &.{};
    if (desc.weights != null or desc.weights_len != 0 or desc.bias != null or desc.bias_len != 0 or desc.input != null or desc.input_len != 0) {
        return error.ShapeMismatch;
    }
    if (desc.output) |output| {
        if (desc.output_len < vocab_size) return error.OutputBufferTooSmall;
        return output[0..vocab_size];
    }
    if (desc.output_len != 0) return error.ShapeMismatch;
    return &.{};
}

const LlamaBufferBoundOutput = struct {
    output_buf: []f32 = &.{},
    bind_options: llm_mod.LlamaBindOptions = .{},
};

const LlamaOutputStaging = struct {
    output_buf: []f32 = &.{},
    output_resource: ?backend_mod.ProgramIO.ExternalResource = null,
    owns_output_buf: bool = false,
};

fn createLlamaOutputStaging(allocator: std.mem.Allocator, bound: LlamaBufferBoundOutput, vocab_size: usize) !LlamaOutputStaging {
    if (bound.output_buf.len != 0) {
        return .{
            .output_buf = bound.output_buf,
        };
    }
    const output_resource = bound.bind_options.output_resource orelse return .{};
    return .{
        .output_buf = try allocator.alloc(f32, vocab_size),
        .output_resource = output_resource,
        .owns_output_buf = true,
    };
}

fn parseLlamaBufferOutput(output_buffer: ?*zgml_buffer, output_len: usize, vocab_size: usize) !LlamaBufferBoundOutput {
    const output = output_buffer orelse {
        if (output_len != 0) return error.ShapeMismatch;
        return .{};
    };
    if (output_len < vocab_size) return error.OutputBufferTooSmall;
    const handle = bufferHandle(output) orelse return error.ShapeMismatch;
    const byte_len = std.math.mul(usize, vocab_size, @sizeOf(f32)) catch return error.ShapeMismatch;
    if (byte_len > handle.byteLen()) return error.ShapeMismatch;
    if (handle.externalResource()) |resource| {
        return .{ .bind_options = .{ .output_resource = resource } };
    }
    const data = handle.hostData().?;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    return .{ .output_buf = @as([*]f32, @ptrCast(@alignCast(data.ptr)))[0..vocab_size] };
}

fn parseLlamaBufferBoundOutput(desc: *const zgml_buffer_bind_desc, vocab_size: usize) !LlamaBufferBoundOutput {
    if (desc.weights != null or desc.weights_len != 0 or desc.bias != null or desc.bias_len != 0 or desc.input != null or desc.input_len != 0) {
        return error.ShapeMismatch;
    }
    return parseLlamaBufferOutput(desc.output, desc.output_len, vocab_size);
}

fn LlamaCacheResourceArrays(comptime config: llm_mod.LlamaConfig) type {
    return struct {
        k: [config.n_layers]backend_mod.ProgramIO.ExternalResource,
        v: [config.n_layers]backend_mod.ProgramIO.ExternalResource,
        element_count: usize = 0,
    };
}

fn LlamaCacheHostArrays(comptime config: llm_mod.LlamaConfig) type {
    return struct {
        k: [config.n_layers]llm_mod.LlamaBindOptions.HostBuffer,
        v: [config.n_layers]llm_mod.LlamaBindOptions.HostBuffer,
        element_count: usize = 0,
    };
}

fn LlamaCacheBindings(comptime config: llm_mod.LlamaConfig) type {
    return struct {
        resources: ?LlamaCacheResourceArrays(config) = null,
        buffers: ?LlamaCacheHostArrays(config) = null,
    };
}

fn llamaKvCacheElementCount(comptime config: llm_mod.LlamaConfig, context_len: usize) !usize {
    if (context_len == 0 or context_len > config.max_seq_len) return error.InvalidContextLength;
    const d_head = config.d_model / config.n_heads;
    if (config.n_kv_heads == 0) return error.ShapeMismatch;
    const columns = std.math.mul(usize, context_len, config.n_kv_heads) catch return error.ShapeMismatch;
    return std.math.mul(usize, d_head, columns) catch return error.ShapeMismatch;
}

const ParsedLlamaCacheBuffer = union(enum) {
    host: llm_mod.LlamaBindOptions.HostBuffer,
    resource: backend_mod.ProgramIO.ExternalResource,
};

fn parseLlamaCacheBuffer(
    buffer: ?*zgml_buffer,
    byte_len: usize,
) !ParsedLlamaCacheBuffer {
    const handle = bufferHandle(buffer) orelse return error.ShapeMismatch;
    if (byte_len > handle.byteLen()) return error.ShapeMismatch;
    if (handle.externalResource()) |resource| return .{ .resource = resource };
    const data = handle.hostData().?;
    if (@intFromPtr(data.ptr) % @alignOf(f32) != 0) return error.ShapeMismatch;
    return .{ .host = .{ .ptr = data.ptr, .byte_len = data.len } };
}

fn parseLlamaCacheBindings(
    comptime config: llm_mod.LlamaConfig,
    context_len: usize,
    desc_ptr: ?*const zgml_llama_kv_cache_bind_desc,
) !?LlamaCacheBindings(config) {
    const desc = desc_ptr orelse return null;
    if (desc.len == 0) {
        if (desc.k != null or desc.v != null) return error.ShapeMismatch;
        return null;
    }
    if (desc.len != config.n_layers or desc.k == null or desc.v == null) return error.ShapeMismatch;

    const element_count = try llamaKvCacheElementCount(config, context_len);
    const byte_len = std.math.mul(usize, element_count, @sizeOf(f32)) catch return error.ShapeMismatch;
    var resources: LlamaCacheResourceArrays(config) = undefined;
    var buffers: LlamaCacheHostArrays(config) = undefined;
    resources.element_count = element_count;
    buffers.element_count = element_count;
    var kind: enum { unset, host, resource } = .unset;
    for (0..config.n_layers) |l| {
        const k = try parseLlamaCacheBuffer(desc.k.?[l], byte_len);
        const v = try parseLlamaCacheBuffer(desc.v.?[l], byte_len);
        switch (k) {
            .host => |host| {
                if (kind == .resource) return error.UnsupportedResourceBinding;
                kind = .host;
                buffers.k[l] = host;
            },
            .resource => |resource| {
                if (kind == .host) return error.UnsupportedResourceBinding;
                kind = .resource;
                resources.k[l] = resource;
            },
        }
        switch (v) {
            .host => |host| {
                if (kind == .resource) return error.UnsupportedResourceBinding;
                kind = .host;
                buffers.v[l] = host;
            },
            .resource => |resource| {
                if (kind == .host) return error.UnsupportedResourceBinding;
                kind = .resource;
                resources.v[l] = resource;
            },
        }
    }
    return switch (kind) {
        .unset => null,
        .host => .{ .buffers = buffers },
        .resource => .{ .resources = resources },
    };
}

fn optionsWithCacheBindings(
    comptime config: llm_mod.LlamaConfig,
    output_options: llm_mod.LlamaBindOptions,
    cache_bindings: ?*const LlamaCacheBindings(config),
) llm_mod.LlamaBindOptions {
    var options = output_options;
    if (cache_bindings) |bindings| {
        if (bindings.buffers) |*buffers| {
            options.cache_buffers = .{ .k = buffers.k[0..], .v = buffers.v[0..], .element_count = buffers.element_count };
        }
        if (bindings.resources) |*resources| {
            options.cache_resources = .{ .k = resources.k[0..], .v = resources.v[0..], .element_count = resources.element_count };
        }
    }
    return options;
}

fn LlamaParsedBufferBind(comptime config: llm_mod.LlamaConfig) type {
    return struct {
        output: LlamaBufferBoundOutput = .{},
        cache_bindings: ?LlamaCacheBindings(config) = null,
    };
}

fn parseLlamaBufferBind(
    comptime config: llm_mod.LlamaConfig,
    context_len: usize,
    desc_ptr: ?*const zgml_llama_buffer_bind_desc,
) !LlamaParsedBufferBind(config) {
    const desc = desc_ptr orelse return .{};
    return .{
        .output = try parseLlamaBufferOutput(desc.output, desc.output_len, config.vocab_size),
        .cache_bindings = try parseLlamaCacheBindings(config, context_len, desc.kv_cache),
    };
}

fn bindLlamaSessionLlamaBuffers(p: *ProgramHandle, source_model: ?*ModelHandle, desc_ptr: ?*const zgml_llama_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    const out = out_session orelse return status(.invalid_argument);
    const handle = alloc.create(SessionHandle) catch return status(.out_of_memory);

    switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        .tiny_llama => |*llama| {
            const context_len = llama.program.inspect().context_len;
            var parsed = parseLlamaBufferBind(tiny_llama_config, context_len, desc_ptr) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const cache_ptr: ?*const LlamaCacheBindings(tiny_llama_config) = if (parsed.cache_bindings) |*bindings| bindings else null;
            const options = optionsWithCacheBindings(tiny_llama_config, parsed.output.bind_options, cache_ptr);
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .tiny_llama => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama_2layer, .smollm_135m => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, options);
            } else llama.program.bindExecutableDecode(options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, parsed.output, tiny_llama_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
        .tiny_llama_2layer => |*llama| {
            const context_len = llama.program.inspect().context_len;
            var parsed = parseLlamaBufferBind(tiny_llama_2layer_config, context_len, desc_ptr) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const cache_ptr: ?*const LlamaCacheBindings(tiny_llama_2layer_config) = if (parsed.cache_bindings) |*bindings| bindings else null;
            const options = optionsWithCacheBindings(tiny_llama_2layer_config, parsed.output.bind_options, cache_ptr);
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .tiny_llama_2layer => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama, .smollm_135m => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, options);
            } else llama.program.bindExecutableDecode(options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, parsed.output, tiny_llama_2layer_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama_2layer = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
        .smollm_135m => |*llama| {
            const context_len = llama.program.inspect().context_len;
            var parsed = parseLlamaBufferBind(smollm_135m_config, context_len, desc_ptr) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const cache_ptr: ?*const LlamaCacheBindings(smollm_135m_config) = if (parsed.cache_bindings) |*bindings| bindings else null;
            const options = optionsWithCacheBindings(smollm_135m_config, parsed.output.bind_options, cache_ptr);
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .smollm_135m => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama, .tiny_llama_2layer => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, options);
            } else llama.program.bindExecutableDecode(options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, parsed.output, smollm_135m_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .smollm_135m = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
    }

    out.* = @ptrCast(handle);
    return status(.ok);
}

fn bindLlamaSessionBuffers(p: *ProgramHandle, source_model: ?*ModelHandle, desc: *const zgml_buffer_bind_desc, out_session: ?*?*zgml_session) c_int {
    const out = out_session orelse return status(.invalid_argument);
    const handle = alloc.create(SessionHandle) catch return status(.out_of_memory);

    switch (p.data) {
        .tiny_linear, .tiny_mlp, .module => {
            alloc.destroy(handle);
            return status(.unsupported);
        },
        .tiny_llama => |*llama| {
            const bound = parseLlamaBufferBoundOutput(desc, tiny_llama_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .tiny_llama => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama_2layer, .smollm_135m => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, bound.bind_options);
            } else llama.program.bindExecutableDecode(bound.bind_options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, bound, tiny_llama_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
        .tiny_llama_2layer => |*llama| {
            const bound = parseLlamaBufferBoundOutput(desc, tiny_llama_2layer_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .tiny_llama_2layer => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama, .smollm_135m => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, bound.bind_options);
            } else llama.program.bindExecutableDecode(bound.bind_options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, bound, tiny_llama_2layer_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .tiny_llama_2layer = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
        .smollm_135m => |*llama| {
            const bound = parseLlamaBufferBoundOutput(desc, smollm_135m_config.vocab_size) catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const session_result = if (source_model) |m| blk: {
                const source = switch (m.data) {
                    .smollm_135m => |*source| source.model,
                    .tiny_linear, .tiny_mlp, .tiny_llama, .tiny_llama_2layer => {
                        alloc.destroy(handle);
                        return status(.shape_mismatch);
                    },
                };
                break :blk llama.program.bindModel(source, bound.bind_options);
            } else llama.program.bindExecutableDecode(bound.bind_options);
            const session = session_result catch |err| {
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            const output_staging = createLlamaOutputStaging(alloc, bound, smollm_135m_config.vocab_size) catch |err| {
                session.deinit();
                alloc.destroy(handle);
                return compileErrorStatus(err);
            };
            p.retain();
            handle.* = .{ .data = .{ .smollm_135m = .{
                .program = p,
                .session = session,
                .output_buf = output_staging.output_buf,
                .output_resource = output_staging.output_resource,
                .output_is_resource = output_staging.output_resource != null,
                .owns_output_buf = output_staging.owns_output_buf,
            } } };
        },
    }

    out.* = @ptrCast(handle);
    return status(.ok);
}

fn bindTinyLinearSession(p: *ProgramHandle, linear: *TinyLinearProgramHandle, desc: *const zgml_bind_desc) !TinyLinearSessionHandle {
    if (!linear.execution_supported) return error.Unsupported;

    const expected_weights = checkedWeightLen(linear.input_len, linear.output_len) orelse return error.ShapeMismatch;
    if (desc.weights_len != expected_weights or desc.weights == null) return error.ShapeMismatch;
    if (desc.bias_len != 0 and (desc.bias_len != linear.output_len or desc.bias == null)) return error.ShapeMismatch;

    const weights_buf = try alloc.alloc(f32, expected_weights);
    errdefer alloc.free(weights_buf);
    @memcpy(weights_buf, desc.weights.?[0..expected_weights]);

    const bias_buf = try alloc.alloc(f32, linear.output_len);
    errdefer alloc.free(bias_buf);
    if (desc.bias_len == 0) {
        @memset(bias_buf, 0);
    } else {
        @memcpy(bias_buf, desc.bias.?[0..linear.output_len]);
    }

    const input_buf, const owns_input_buf = if (desc.input) |input| blk: {
        if (desc.input_len != linear.input_len) return error.ShapeMismatch;
        break :blk .{ @constCast(input[0..linear.input_len]), false };
    } else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, linear.input_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_input_buf) alloc.free(input_buf);

    const output_buf, const owns_output_buf = if (desc.output) |output| blk: {
        if (desc.output_len < linear.output_len) return error.ShapeMismatch;
        break :blk .{ output[0..linear.output_len], false };
    } else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, linear.output_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_output_buf) alloc.free(output_buf);

    const persistent_bindings = [_]DeviceF32.TensorBinding{
        .{ .tensor = linear.weights, .host_ptr = weights_buf.ptr, .element_count = weights_buf.len },
        .{ .tensor = linear.bias, .host_ptr = bias_buf.ptr, .element_count = bias_buf.len },
    };
    const input_bindings = [_]DeviceF32.TensorBinding{
        .{ .tensor = linear.input, .host_ptr = input_buf.ptr, .element_count = input_buf.len },
    };
    const session = try linear.program.bind(.{
        .input_tensors = &.{linear.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = &.{ linear.weights, linear.bias },
        .persistent_bindings = &persistent_bindings,
        .output_tensor = linear.output,
        .output_host_buf = output_buf.ptr,
        .output_len = linear.output_len,
    });
    p.retain();
    return .{
        .input_len = linear.input_len,
        .output_len = linear.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .owns_weights_buf = true,
        .owns_bias_buf = true,
        .owns_input_buf = owns_input_buf,
        .owns_output_buf = owns_output_buf,
    };
}

fn bindTinyMlpSession(p: *ProgramHandle, mlp: *TinyMlpProgramHandle, desc: *const zgml_bind_desc) !TinyLinearSessionHandle {
    if (!mlp.execution_supported) return error.Unsupported;

    const weights = checkedTinyMlpWeightLens(mlp.input_len, mlp.hidden_len, mlp.output_len) orelse return error.ShapeMismatch;
    const expected_bias = checkedTinyMlpBiasLen(mlp.hidden_len, mlp.output_len) orelse return error.ShapeMismatch;
    if (desc.weights_len != weights.total or desc.weights == null) return error.ShapeMismatch;
    if (desc.bias_len != 0 and (desc.bias_len != expected_bias or desc.bias == null)) return error.ShapeMismatch;

    const weights_buf = try alloc.alloc(f32, weights.total);
    errdefer alloc.free(weights_buf);
    @memcpy(weights_buf, desc.weights.?[0..weights.total]);

    const bias_buf = try alloc.alloc(f32, expected_bias);
    errdefer alloc.free(bias_buf);
    if (desc.bias_len == 0) {
        @memset(bias_buf, 0);
    } else {
        @memcpy(bias_buf, desc.bias.?[0..expected_bias]);
    }

    const input_buf, const owns_input_buf = if (desc.input) |input| blk: {
        if (desc.input_len != mlp.input_len) return error.ShapeMismatch;
        break :blk .{ @constCast(input[0..mlp.input_len]), false };
    } else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, mlp.input_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_input_buf) alloc.free(input_buf);

    const output_buf, const owns_output_buf = if (desc.output) |output| blk: {
        if (desc.output_len < mlp.output_len) return error.ShapeMismatch;
        break :blk .{ output[0..mlp.output_len], false };
    } else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, mlp.output_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_output_buf) alloc.free(output_buf);

    const w0_buf = weights_buf[0..weights.w0];
    const w1_buf = weights_buf[weights.w0..][0..weights.w1];
    const b0_buf = bias_buf[0..mlp.hidden_len];
    const b1_buf = bias_buf[mlp.hidden_len..][0..mlp.output_len];
    const persistent_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.host(mlp.w0, w0_buf.ptr, w0_buf.len),
        DeviceF32.TensorBinding.host(mlp.b0, b0_buf.ptr, b0_buf.len),
        DeviceF32.TensorBinding.host(mlp.w1, w1_buf.ptr, w1_buf.len),
        DeviceF32.TensorBinding.host(mlp.b1, b1_buf.ptr, b1_buf.len),
    };
    const input_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.host(mlp.input, input_buf.ptr, input_buf.len),
    };
    const session = try mlp.program.bind(.{
        .input_tensors = &.{mlp.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = &.{ mlp.w0, mlp.b0, mlp.w1, mlp.b1 },
        .persistent_bindings = &persistent_bindings,
        .output_tensor = mlp.output,
        .output_host_buf = output_buf.ptr,
        .output_len = mlp.output_len,
    });
    p.retain();
    return .{
        .input_len = mlp.input_len,
        .output_len = mlp.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .owns_weights_buf = true,
        .owns_bias_buf = true,
        .owns_input_buf = owns_input_buf,
        .owns_output_buf = owns_output_buf,
    };
}

fn bindTinyMlpBufferSession(p: *ProgramHandle, mlp: *TinyMlpProgramHandle, desc: *const zgml_buffer_bind_desc) !TinyLinearSessionHandle {
    if (!mlp.execution_supported) return error.Unsupported;

    const weights = checkedTinyMlpWeightLens(mlp.input_len, mlp.hidden_len, mlp.output_len) orelse return error.ShapeMismatch;
    const expected_bias = checkedTinyMlpBiasLen(mlp.hidden_len, mlp.output_len) orelse return error.ShapeMismatch;
    if (desc.weights_len != weights.total or desc.weights == null) return error.ShapeMismatch;
    if (desc.bias != null) {
        if (desc.bias_len != expected_bias) return error.ShapeMismatch;
    } else if (desc.bias_len != 0) {
        return error.ShapeMismatch;
    }

    var owned_bias: []f32 = &.{};
    var owned_input: []f32 = &.{};
    var owned_output: []f32 = &.{};
    errdefer {
        if (owned_output.len != 0) alloc.free(owned_output);
        if (owned_input.len != 0) alloc.free(owned_input);
        if (owned_bias.len != 0) alloc.free(owned_bias);
    }

    if (desc.bias == null) {
        owned_bias = try alloc.alloc(f32, expected_bias);
        @memset(owned_bias, 0);
    }

    const w0_binding = try aggregateBufferF32Binding(mlp.w0, desc.weights, weights.total, 0, weights.w0);
    const w1_binding = try aggregateBufferF32Binding(mlp.w1, desc.weights, weights.total, weights.w0, weights.w1);
    const b0_binding = if (desc.bias) |_|
        try aggregateBufferF32Binding(mlp.b0, desc.bias, expected_bias, 0, mlp.hidden_len)
    else
        DeviceF32.TensorBinding.host(mlp.b0, owned_bias[0..mlp.hidden_len].ptr, mlp.hidden_len);
    const b1_binding = if (desc.bias) |_|
        try aggregateBufferF32Binding(mlp.b1, desc.bias, expected_bias, mlp.hidden_len, mlp.output_len)
    else
        DeviceF32.TensorBinding.host(mlp.b1, owned_bias[mlp.hidden_len..][0..mlp.output_len].ptr, mlp.output_len);

    const input_binding = if (desc.input) |_|
        if (desc.input_len == mlp.input_len)
            (try bufferF32Binding(mlp.input, desc.input, mlp.input_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        owned_input = try alloc.alloc(f32, mlp.input_len);
        break :blk DeviceF32.TensorBinding.host(mlp.input, owned_input.ptr, owned_input.len);
    };
    const output_binding: ?DeviceF32.TensorBinding = if (desc.output) |_|
        if (desc.output_len >= mlp.output_len)
            (try bufferF32Binding(mlp.output, desc.output, mlp.output_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        owned_output = try alloc.alloc(f32, mlp.output_len);
        break :blk null;
    };

    const persistent_bindings = [_]DeviceF32.TensorBinding{
        w0_binding,
        b0_binding,
        w1_binding,
        b1_binding,
    };
    const input_bindings = [_]DeviceF32.TensorBinding{input_binding};
    var fallback_output = [_]f32{0};
    const output_ptr = if (owned_output.len != 0) owned_output.ptr else fallback_output[0..].ptr;
    var session = try mlp.program.bind(.{
        .input_tensors = &.{mlp.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = &.{ mlp.w0, mlp.b0, mlp.w1, mlp.b1 },
        .persistent_bindings = &persistent_bindings,
        .output_tensor = mlp.output,
        .output_host_buf = output_ptr,
        .output_len = if (owned_output.len != 0) owned_output.len else fallback_output.len,
        .output_binding = output_binding,
    });
    errdefer session.deinit();

    const output_buf: []f32 = if (owned_output.len != 0)
        owned_output
    else if (output_binding) |binding|
        if (binding.resource == null) binding.host_ptr[0..mlp.output_len] else &.{}
    else
        &.{};
    const input_buf: []f32 = if (owned_input.len != 0)
        owned_input
    else if (input_binding.resource == null)
        input_binding.host_ptr[0..mlp.input_len]
    else
        &.{};
    const bias_buf: []f32 = if (owned_bias.len != 0)
        owned_bias
    else
        bufferF32HostSlice(desc.bias, expected_bias) catch @as([]f32, &.{});
    const weights_buf = bufferF32HostSlice(desc.weights, weights.total) catch @as([]f32, &.{});

    p.retain();
    return .{
        .input_len = mlp.input_len,
        .output_len = mlp.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .owns_weights_buf = false,
        .owns_bias_buf = owned_bias.len != 0,
        .owns_input_buf = owned_input.len != 0,
        .owns_output_buf = owned_output.len != 0,
    };
}

fn bindModuleSession(p: *ProgramHandle, module: *ModuleProgramHandle, desc: *const zgml_bind_desc) !TinyLinearSessionHandle {
    if (!module.execution_supported) return error.Unsupported;
    if (desc.weights_len != module.weights_len) return error.ShapeMismatch;
    if (module.weights_len != 0 and desc.weights == null) return error.ShapeMismatch;
    if (desc.bias_len != 0 and desc.bias_len != module.bias_len) return error.ShapeMismatch;
    if (desc.bias_len != 0 and desc.bias == null) return error.ShapeMismatch;

    const weights_buf, const owns_weights_buf = if (module.weights_len == 0) blk: {
        if (desc.weights != null) return error.ShapeMismatch;
        break :blk .{ @as([]f32, &.{}), false };
    } else blk: {
        const buf = try alloc.alloc(f32, module.weights_len);
        errdefer alloc.free(buf);
        @memcpy(buf, desc.weights.?[0..module.weights_len]);
        break :blk .{ buf, true };
    };
    errdefer if (owns_weights_buf) alloc.free(weights_buf);

    const bias_buf, const owns_bias_buf = if (module.bias_len == 0) blk: {
        if (desc.bias != null or desc.bias_len != 0) return error.ShapeMismatch;
        break :blk .{ @as([]f32, &.{}), false };
    } else blk: {
        const buf = try alloc.alloc(f32, module.bias_len);
        errdefer alloc.free(buf);
        if (desc.bias_len == 0) {
            @memset(buf, 0);
        } else {
            @memcpy(buf, desc.bias.?[0..module.bias_len]);
        }
        break :blk .{ buf, true };
    };
    errdefer if (owns_bias_buf) alloc.free(bias_buf);

    const input_buf, const owns_input_buf = if (desc.input) |input| blk: {
        if (desc.input_len != module.input_len) return error.ShapeMismatch;
        break :blk .{ @constCast(input[0..module.input_len]), false };
    } else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, module.input_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_input_buf) alloc.free(input_buf);

    const output_buf, const owns_output_buf = if (desc.output) |output| blk: {
        if (desc.output_len < module.output_len) return error.ShapeMismatch;
        break :blk .{ output[0..module.output_len], false };
    } else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        const buf = try alloc.alloc(f32, module.output_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    };
    errdefer if (owns_output_buf) alloc.free(output_buf);

    const scratch_buf, const owns_scratch_buf = if (module.direct_rms_gelu_linear) |_| blk: {
        const buf = try alloc.alloc(f32, module.input_len);
        errdefer alloc.free(buf);
        break :blk .{ buf, true };
    } else .{ @as([]f32, &.{}), false };
    errdefer if (owns_scratch_buf) alloc.free(scratch_buf);

    const persistent_bindings = try alloc.alloc(DeviceF32.TensorBinding, module.persistent_params.len);
    defer alloc.free(persistent_bindings);
    for (module.persistent_params, 0..) |param, i| {
        const source = switch (param.kind) {
            .weights => weights_buf[param.offset..][0..param.len],
            .bias => bias_buf[param.offset..][0..param.len],
        };
        persistent_bindings[i] = DeviceF32.TensorBinding.host(param.tensor, source.ptr, source.len);
    }
    const input_bindings = [_]DeviceF32.TensorBinding{
        DeviceF32.TensorBinding.host(module.input, input_buf.ptr, input_buf.len),
    };
    const session = try module.program.bind(.{
        .input_tensors = &.{module.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = module.persistent_tensors,
        .persistent_bindings = persistent_bindings,
        .output_tensor = module.output,
        .output_host_buf = output_buf.ptr,
        .output_len = module.output_len,
    });
    p.retain();
    return .{
        .input_len = module.input_len,
        .output_len = module.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .scratch_buf = scratch_buf,
        .owns_weights_buf = owns_weights_buf,
        .owns_bias_buf = owns_bias_buf,
        .owns_input_buf = owns_input_buf,
        .owns_output_buf = owns_output_buf,
        .owns_scratch_buf = owns_scratch_buf,
    };
}

fn bindModuleBufferSession(p: *ProgramHandle, module: *ModuleProgramHandle, desc: *const zgml_buffer_bind_desc) !TinyLinearSessionHandle {
    if (desc.weights_len != module.weights_len) return error.ShapeMismatch;
    if (module.weights_len != 0 and desc.weights == null) return error.ShapeMismatch;
    if (module.weights_len == 0 and desc.weights != null) return error.ShapeMismatch;
    if (desc.bias != null) {
        if (desc.bias_len != module.bias_len) return error.ShapeMismatch;
    } else if (desc.bias_len != 0) {
        return error.ShapeMismatch;
    }

    var owned_bias: []f32 = &.{};
    var owned_input: []f32 = &.{};
    var owned_output: []f32 = &.{};
    var owned_scratch: []f32 = &.{};
    errdefer {
        if (owned_scratch.len != 0) alloc.free(owned_scratch);
        if (owned_output.len != 0) alloc.free(owned_output);
        if (owned_input.len != 0) alloc.free(owned_input);
        if (owned_bias.len != 0) alloc.free(owned_bias);
    }

    if (module.bias_len != 0 and desc.bias == null) {
        owned_bias = try alloc.alloc(f32, module.bias_len);
        @memset(owned_bias, 0);
    }

    const input_binding = if (desc.input) |_|
        if (desc.input_len == module.input_len)
            (try bufferF32Binding(module.input, desc.input, module.input_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        owned_input = try alloc.alloc(f32, module.input_len);
        break :blk DeviceF32.TensorBinding.host(module.input, owned_input.ptr, owned_input.len);
    };

    const output_binding: ?DeviceF32.TensorBinding = if (desc.output) |_|
        if (desc.output_len >= module.output_len)
            (try bufferF32Binding(module.output, desc.output, module.output_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        owned_output = try alloc.alloc(f32, module.output_len);
        break :blk null;
    };

    const persistent_bindings = try alloc.alloc(DeviceF32.TensorBinding, module.persistent_params.len);
    defer alloc.free(persistent_bindings);
    for (module.persistent_params, 0..) |param, i| {
        persistent_bindings[i] = switch (param.kind) {
            .weights => try aggregateBufferF32Binding(param.tensor, desc.weights, module.weights_len, param.offset, param.len),
            .bias => if (desc.bias) |_|
                try aggregateBufferF32Binding(param.tensor, desc.bias, module.bias_len, param.offset, param.len)
            else
                DeviceF32.TensorBinding.host(param.tensor, owned_bias[param.offset..][0..param.len].ptr, param.len),
        };
    }

    const input_bindings = [_]DeviceF32.TensorBinding{input_binding};
    var fallback_output = [_]f32{0};
    const output_ptr = if (owned_output.len != 0) owned_output.ptr else fallback_output[0..].ptr;
    var session = try module.program.bind(.{
        .input_tensors = &.{module.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = module.persistent_tensors,
        .persistent_bindings = persistent_bindings,
        .output_tensor = module.output,
        .output_host_buf = output_ptr,
        .output_len = if (owned_output.len != 0) owned_output.len else fallback_output.len,
        .output_binding = output_binding,
    });
    errdefer session.deinit();

    const weights_buf = bufferF32HostSlice(desc.weights, module.weights_len) catch @as([]f32, &.{});
    const bias_buf: []f32 = if (owned_bias.len != 0)
        owned_bias
    else
        bufferF32HostSlice(desc.bias, module.bias_len) catch @as([]f32, &.{});
    const input_buf: []f32 = if (owned_input.len != 0)
        owned_input
    else
        bufferF32HostSlice(desc.input, module.input_len) catch @as([]f32, &.{});
    const output_buf: []f32 = if (owned_output.len != 0)
        owned_output
    else
        bufferF32HostSlice(desc.output, module.output_len) catch @as([]f32, &.{});
    if (module.direct_rms_gelu_linear != null) {
        owned_scratch = try alloc.alloc(f32, module.input_len);
    }

    p.retain();
    return .{
        .input_len = module.input_len,
        .output_len = module.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .scratch_buf = owned_scratch,
        .owns_weights_buf = false,
        .owns_bias_buf = owned_bias.len != 0,
        .owns_input_buf = owned_input.len != 0,
        .owns_output_buf = owned_output.len != 0,
        .owns_scratch_buf = owned_scratch.len != 0,
    };
}

fn bindTinyLinearBufferSession(p: *ProgramHandle, linear: *TinyLinearProgramHandle, desc: *const zgml_buffer_bind_desc) !TinyLinearSessionHandle {
    const expected_weights = checkedWeightLen(linear.input_len, linear.output_len) orelse return error.ShapeMismatch;
    if (desc.weights_len != expected_weights or desc.weights == null) return error.ShapeMismatch;
    if (desc.bias_len != 0 and (desc.bias_len != linear.output_len or desc.bias == null)) return error.ShapeMismatch;

    var owned_bias: []f32 = &.{};
    var owned_input: []f32 = &.{};
    var owned_output: []f32 = &.{};
    errdefer {
        if (owned_output.len != 0) alloc.free(owned_output);
        if (owned_input.len != 0) alloc.free(owned_input);
        if (owned_bias.len != 0) alloc.free(owned_bias);
    }

    const weights_binding = (try bufferF32Binding(linear.weights, desc.weights, expected_weights)) orelse return error.ShapeMismatch;
    const bias_binding = if (desc.bias) |_|
        (try bufferF32Binding(linear.bias, desc.bias, linear.output_len)) orelse return error.ShapeMismatch
    else blk: {
        if (desc.bias_len != 0) return error.ShapeMismatch;
        owned_bias = try alloc.alloc(f32, linear.output_len);
        @memset(owned_bias, 0);
        break :blk DeviceF32.TensorBinding.host(linear.bias, owned_bias.ptr, owned_bias.len);
    };
    const input_binding = if (desc.input) |_|
        if (desc.input_len == linear.input_len)
            (try bufferF32Binding(linear.input, desc.input, linear.input_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.input_len != 0) return error.ShapeMismatch;
        owned_input = try alloc.alloc(f32, linear.input_len);
        break :blk DeviceF32.TensorBinding.host(linear.input, owned_input.ptr, owned_input.len);
    };
    const output_binding: ?DeviceF32.TensorBinding = if (desc.output) |_|
        if (desc.output_len >= linear.output_len)
            (try bufferF32Binding(linear.output, desc.output, linear.output_len)) orelse return error.ShapeMismatch
        else
            return error.ShapeMismatch
    else blk: {
        if (desc.output_len != 0) return error.ShapeMismatch;
        owned_output = try alloc.alloc(f32, linear.output_len);
        break :blk null;
    };

    const persistent_bindings = [_]DeviceF32.TensorBinding{ weights_binding, bias_binding };
    const input_bindings = [_]DeviceF32.TensorBinding{input_binding};
    var fallback_output = [_]f32{0};
    const output_ptr = if (owned_output.len != 0) owned_output.ptr else fallback_output[0..].ptr;
    var session = try linear.program.bind(.{
        .input_tensors = &.{linear.input},
        .input_bindings = &input_bindings,
        .persistent_tensors = &.{ linear.weights, linear.bias },
        .persistent_bindings = &persistent_bindings,
        .output_tensor = linear.output,
        .output_host_buf = output_ptr,
        .output_len = if (owned_output.len != 0) owned_output.len else fallback_output.len,
        .output_binding = output_binding,
    });
    errdefer session.deinit();

    const output_buf: []f32 = if (owned_output.len != 0)
        owned_output
    else if (output_binding) |binding|
        if (binding.resource == null) binding.host_ptr[0..linear.output_len] else &.{}
    else
        &.{};
    const input_buf: []f32 = if (owned_input.len != 0)
        owned_input
    else if (input_binding.resource == null)
        input_binding.host_ptr[0..linear.input_len]
    else
        &.{};
    const bias_buf: []f32 = if (owned_bias.len != 0)
        owned_bias
    else if (bias_binding.resource == null)
        bias_binding.host_ptr[0..linear.output_len]
    else
        &.{};
    const weights_buf: []f32 = if (weights_binding.resource == null)
        weights_binding.host_ptr[0..expected_weights]
    else
        &.{};

    p.retain();
    return .{
        .input_len = linear.input_len,
        .output_len = linear.output_len,
        .program = p,
        .session = session,
        .weights_buf = weights_buf,
        .bias_buf = bias_buf,
        .input_buf = input_buf,
        .output_buf = output_buf,
        .owns_weights_buf = false,
        .owns_bias_buf = owned_bias.len != 0,
        .owns_input_buf = owned_input.len != 0,
        .owns_output_buf = owned_output.len != 0,
    };
}

export fn zgml_session_upload_persistent(session: ?*zgml_session) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const runtime = deviceSessionRuntime(s) orelse return status(.unsupported);
    if (!runtime.execution_supported) return status(.unsupported);
    runtime.program.uploadPersistentInputs(runtime.session) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_upload_persistent_range(session: ?*zgml_session, first: usize, len: usize) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const runtime = deviceSessionRuntime(s) orelse return status(.unsupported);
    if (!runtime.execution_supported) return status(.unsupported);
    runtime.program.uploadPersistentInputRange(runtime.session, first, len) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_step(session: ?*zgml_session, desc_ptr: ?*const zgml_step_desc, result_ptr: ?*zgml_step_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const linear = switch (s.data) {
        .tiny_linear => |*linear| linear,
        .tiny_mlp => |*mlp| mlp,
        .module => |*module| module,
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => return status(.unsupported),
    };
    const p = deviceProgramRuntime(linear.program) orelse return status(.invalid_argument);
    if (p.backend == backend_webgpu) {
        if (maybeWasmHostSessionStep(session, desc_ptr, result_ptr)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    if (!p.execution_supported) return status(.unsupported);
    if (desc_ptr == null and (linear.owns_input_buf or linear.owns_output_buf)) return status(.invalid_argument);
    if (desc_ptr) |desc| {
        if (desc.input) |input| {
            if (desc.input_len != linear.input_len) return status(.shape_mismatch);
            if (input != linear.input_buf.ptr) {
                @memcpy(linear.input_buf[0..linear.input_len], input[0..linear.input_len]);
            }
        } else if (linear.owns_input_buf) {
            return status(.invalid_argument);
        }
        if (desc.output) |output| {
            if (desc.output_len < linear.output_len) return status(.shape_mismatch);
            _ = output;
        } else if (linear.owns_output_buf) {
            return status(.invalid_argument);
        }
    }

    const window = backend_mod.RuntimeWindow.init(0, 0) catch return status(.shape_mismatch);
    p.program.executeStep(&linear.session, .{ .window = window }) catch return status(.shape_mismatch);
    if (desc_ptr) |desc| {
        if (desc.output) |output| {
            if (output != linear.output_buf.ptr) {
                @memcpy(output[0..linear.output_len], linear.output_buf[0..linear.output_len]);
            }
        }
    }
    if (result_ptr) |result| {
        result.* = .{ .output_len = linear.output_len };
    }
    return status(.ok);
}

const DirectLinearStepShape = struct {
    M: usize,
    N: usize,
    K: usize,
    has_bias: bool,
};

const DirectRmsGeluLinearStepShape = struct {
    M: usize,
    N: usize,
    K: usize,
    eps: f32,
};

const DirectLinearLogSoftmaxStepShape = struct {
    M: usize,
    N: usize,
    K: usize,
    has_bias: bool,
};

fn directLinearShapeForSession(s: *SessionHandle, linear: *const TinyLinearSessionHandle) ?DirectLinearStepShape {
    return switch (s.data) {
        .tiny_linear => blk: {
            const expected_weights = std.math.mul(usize, linear.input_len, linear.output_len) catch break :blk null;
            if (!linear.owns_weights_buf or !linear.owns_bias_buf) break :blk null;
            if (linear.weights_buf.len != expected_weights) break :blk null;
            if (linear.bias_buf.len != linear.output_len) break :blk null;
            break :blk .{
                .M = 1,
                .N = linear.output_len,
                .K = linear.input_len,
                .has_bias = true,
            };
        },
        .module => blk: {
            const module_program = switch (linear.program.data) {
                .module => |*module| module,
                else => break :blk null,
            };
            const direct = module_program.direct_linear orelse break :blk null;
            if (direct.in_features == 0 or direct.out_features == 0) break :blk null;
            if (!linear.owns_weights_buf or (direct.has_bias and !linear.owns_bias_buf)) break :blk null;
            const expected_weights = std.math.mul(usize, direct.in_features, direct.out_features) catch break :blk null;
            if (linear.weights_buf.len != expected_weights) break :blk null;
            if (direct.has_bias and linear.bias_buf.len != direct.out_features) break :blk null;
            if (!direct.has_bias and linear.bias_buf.len != 0) break :blk null;
            if (linear.input_len % direct.in_features != 0) break :blk null;
            const M = linear.input_len / direct.in_features;
            const expected_output = std.math.mul(usize, M, direct.out_features) catch break :blk null;
            if (linear.output_len != expected_output) break :blk null;
            break :blk .{
                .M = M,
                .N = direct.out_features,
                .K = direct.in_features,
                .has_bias = direct.has_bias,
            };
        },
        else => null,
    };
}

fn directRmsGeluLinearShapeForSession(s: *SessionHandle, linear: *const TinyLinearSessionHandle) ?DirectRmsGeluLinearStepShape {
    return switch (s.data) {
        .module => blk: {
            const module_program = switch (linear.program.data) {
                .module => |*module| module,
                else => break :blk null,
            };
            const direct = module_program.direct_rms_gelu_linear orelse break :blk null;
            if (direct.features == 0 or direct.out_features == 0) break :blk null;
            if (linear.input_len % direct.features != 0) break :blk null;
            const M = linear.input_len / direct.features;
            const expected_output = std.math.mul(usize, M, direct.out_features) catch break :blk null;
            const expected_weights = std.math.add(usize, direct.features, std.math.mul(usize, direct.features, direct.out_features) catch break :blk null) catch break :blk null;
            if (linear.output_len != expected_output) break :blk null;
            if (linear.weights_buf.len != expected_weights) break :blk null;
            if (linear.bias_buf.len != direct.out_features) break :blk null;
            if (linear.scratch_buf.len != linear.input_len) break :blk null;
            break :blk .{
                .M = M,
                .N = direct.out_features,
                .K = direct.features,
                .eps = direct.eps,
            };
        },
        else => null,
    };
}

fn directLinearLogSoftmaxShapeForSession(s: *SessionHandle, linear: *const TinyLinearSessionHandle) ?DirectLinearLogSoftmaxStepShape {
    return switch (s.data) {
        .module => blk: {
            const module_program = switch (linear.program.data) {
                .module => |*module| module,
                else => break :blk null,
            };
            const direct = module_program.direct_linear_log_softmax orelse break :blk null;
            if (direct.in_features == 0 or direct.out_features == 0) break :blk null;
            if (!linear.owns_weights_buf or (direct.has_bias and !linear.owns_bias_buf)) break :blk null;
            const expected_weights = std.math.mul(usize, direct.in_features, direct.out_features) catch break :blk null;
            if (linear.weights_buf.len != expected_weights) break :blk null;
            if (direct.has_bias and linear.bias_buf.len != direct.out_features) break :blk null;
            if (!direct.has_bias and linear.bias_buf.len != 0) break :blk null;
            if (linear.input_len % direct.in_features != 0) break :blk null;
            const M = linear.input_len / direct.in_features;
            const expected_output = std.math.mul(usize, M, direct.out_features) catch break :blk null;
            if (linear.output_len != expected_output) break :blk null;
            break :blk .{
                .M = M,
                .N = direct.out_features,
                .K = direct.in_features,
                .has_bias = direct.has_bias,
            };
        },
        else => null,
    };
}

fn addDenseBiasRows(dst: []f32, bias: []const f32, M: usize, N: usize) void {
    const VecT = @Vector(8, f32);
    for (0..M) |row| {
        const dst_row = dst[row * N ..][0..N];
        var i: usize = 0;
        while (i + 8 <= N) : (i += 8) {
            const a: VecT = dst_row[i..][0..8].*;
            const b: VecT = bias[i..][0..8].*;
            dst_row[i..][0..8].* = a + b;
        }
        while (i < N) : (i += 1) {
            dst_row[i] += bias[i];
        }
    }
}

fn fastExpApproxVec8(x: @Vector(8, f32)) @Vector(8, f32) {
    return fastExpApproxVec(8, x);
}

fn fastExpApproxVec(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const VecT = @Vector(lanes, f32);
    const IVecT = @Vector(lanes, i32);
    const UVecT = @Vector(lanes, u32);
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

fn geluApproxVec8(x: @Vector(8, f32)) @Vector(8, f32) {
    const VecT = @Vector(8, f32);
    const k0: VecT = @splat(0.7978845608);
    const k1: VecT = @splat(0.044715);
    const half: VecT = @splat(0.5);
    const one: VecT = @splat(1.0);
    const k = k0 * (x + k1 * x * x * x);
    return half * x * (one + tanhApproxVec8(k));
}

fn tanhApproxVec8(x: @Vector(8, f32)) @Vector(8, f32) {
    const VecT = @Vector(8, f32);
    const one: VecT = @splat(1.0);
    const clamped = @min(@max(x, @as(VecT, @splat(-10.0))), @as(VecT, @splat(10.0)));
    const e2x = fastExpApproxVec8(clamped + clamped);
    return (e2x - one) / (e2x + one);
}

fn geluApproxScalar(x: f32) f32 {
    const kk = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(kk));
}

fn fastLogPositiveApprox(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const exp_bits = (bits >> 23) & 0xff;
    const exponent = @as(i32, @intCast(exp_bits)) - 127;
    const mantissa_bits = (bits & 0x7fffff) | 0x3f800000;
    const m: f32 = @bitCast(mantissa_bits);
    const y = (m - 1.0) / (m + 1.0);
    const y2 = y * y;
    const y3 = y * y2;
    const y5 = y3 * y2;
    const y7 = y5 * y2;
    const y9 = y7 * y2;
    return @as(f32, @floatFromInt(exponent)) * 0.6931471805599453 + 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0 + y9 / 9.0);
}

test "fast positive log approximation covers softmax denominators" {
    const samples = [_]f32{ 1.0, 1.125, 1.5, 2.0, 3.25, 8.0, 16.0, 31.75, 32.0 };
    for (samples) |value| {
        try std.testing.expectApproxEqAbs(@log(value), fastLogPositiveApprox(value), 4e-6);
    }
}

fn executeDirectRmsGeluLinearStep(linear: *const TinyLinearSessionHandle, shape: DirectRmsGeluLinearStepShape, input: [*]const f32, output: [*]f32) void {
    const VecT = @Vector(8, f32);
    const input_slice = input[0..linear.input_len];
    const scratch = linear.scratch_buf[0..linear.input_len];
    const rms_weight = linear.weights_buf[0..shape.K];
    const dense_weight = linear.weights_buf[shape.K..][0 .. shape.K * shape.N];
    for (0..shape.M) |row| {
        const input_row = input_slice[row * shape.K ..][0..shape.K];
        const scratch_row = scratch[row * shape.K ..][0..shape.K];
        var acc: VecT = @splat(0);
        var i: usize = 0;
        while (i + 8 <= shape.K) : (i += 8) {
            const v: VecT = input_row[i..][0..8].*;
            acc += v * v;
        }
        var ss: f32 = @reduce(.Add, acc);
        while (i < shape.K) : (i += 1) ss += input_row[i] * input_row[i];
        const inv_rms: VecT = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(shape.K)) + shape.eps));
        i = 0;
        while (i + 8 <= shape.K) : (i += 8) {
            const v: VecT = input_row[i..][0..8].*;
            const w: VecT = rms_weight[i..][0..8].*;
            scratch_row[i..][0..8].* = geluApproxVec8(v * inv_rms * w);
        }
        const inv_s = inv_rms[0];
        while (i < shape.K) : (i += 1) {
            scratch_row[i] = geluApproxScalar(input_row[i] * inv_s * rms_weight[i]);
        }
    }
    const output_slice = output[0..linear.output_len];
    forward.blasSgemm(
        output_slice,
        scratch,
        dense_weight,
        shape.M,
        shape.N,
        shape.K,
        shape.K,
        1,
        shape.N,
        1,
        0,
        0,
        0,
        shape.N,
    );
    addDenseBiasRows(output_slice, linear.bias_buf, shape.M, shape.N);
}

fn executeSmallDirectLinearBiasStepLanes(comptime lanes: usize, linear: *const TinyLinearSessionHandle, shape: DirectLinearStepShape, input: [*]const f32, output: [*]f32) void {
    const VecT = @Vector(lanes, f32);
    const input_slice = input[0..linear.input_len];
    const output_slice = output[0..linear.output_len];

    for (0..shape.M) |row| {
        const input_row = input_slice[row * shape.K ..][0..shape.K];
        const output_row = output_slice[row * shape.N ..][0..shape.N];
        var col: usize = 0;
        while (col < shape.N) : (col += lanes) {
            var acc: VecT = linear.bias_buf[col..][0..lanes].*;
            for (0..shape.K) |k| {
                const xv: VecT = @splat(input_row[k]);
                const wv: VecT = linear.weights_buf[k * shape.N + col ..][0..lanes].*;
                acc += xv * wv;
            }
            output_row[col..][0..lanes].* = acc;
        }
    }
}

fn shouldUseBlasForBatchedDirectLinear(shape: DirectLinearStepShape) bool {
    return shape.M >= 64 and shape.N <= 64 and shape.K <= 128;
}

fn executeSmallDirectLinearBiasStep(linear: *const TinyLinearSessionHandle, shape: DirectLinearStepShape, input: [*]const f32, output: [*]f32, prefer_blas_for_batched: bool) bool {
    if (!shape.has_bias) return false;
    if (shape.M > 128 or shape.N > 64 or shape.K > 128) return false;
    if (prefer_blas_for_batched and shouldUseBlasForBatchedDirectLinear(shape)) return false;

    if (shape.N % 16 == 0) {
        executeSmallDirectLinearBiasStepLanes(16, linear, shape, input, output);
        return true;
    }
    if (shape.N % 8 == 0) {
        executeSmallDirectLinearBiasStepLanes(8, linear, shape, input, output);
        return true;
    }
    return false;
}

fn executeDirectLinearStep(linear: *const TinyLinearSessionHandle, shape: DirectLinearStepShape, input: [*]const f32, output: [*]f32, prefer_blas_for_batched: bool) void {
    if (executeSmallDirectLinearBiasStep(linear, shape, input, output, prefer_blas_for_batched)) return;

    const input_slice = input[0..linear.input_len];
    const output_slice = output[0..linear.output_len];
    forward.blasSgemm(
        output_slice,
        input_slice,
        linear.weights_buf,
        shape.M,
        shape.N,
        shape.K,
        shape.K,
        1,
        shape.N,
        1,
        0,
        0,
        0,
        shape.N,
    );
    if (shape.has_bias) {
        addDenseBiasRows(output_slice, linear.bias_buf, shape.M, shape.N);
    }
}

fn logSoftmaxRowsInPlaceLanes(comptime lanes: usize, values: []f32, M: usize, N: usize) void {
    const VecT = @Vector(lanes, f32);
    for (0..M) |row| {
        const out_row = values[row * N ..][0..N];
        var max_vec: VecT = @splat(-std.math.inf(f32));
        var i: usize = 0;
        while (i + lanes <= N) : (i += lanes) {
            const v: VecT = out_row[i..][0..lanes].*;
            max_vec = @max(max_vec, v);
        }
        var max_val: f32 = @reduce(.Max, max_vec);
        while (i < N) : (i += 1) {
            if (out_row[i] > max_val) max_val = out_row[i];
        }
        const max_broadcast: VecT = @splat(max_val);
        var sum_vec: VecT = @splat(0);
        i = 0;
        while (i + lanes <= N) : (i += lanes) {
            const v: VecT = out_row[i..][0..lanes].*;
            sum_vec += fastExpApproxVec(lanes, v - max_broadcast);
        }
        var sum_exp: f32 = @reduce(.Add, sum_vec);
        while (i < N) : (i += 1) {
            sum_exp += @exp(out_row[i] - max_val);
        }
        const log_denom = max_val + @log(sum_exp);
        const log_denom_vec: VecT = @splat(log_denom);
        i = 0;
        while (i + lanes <= N) : (i += lanes) {
            const v: VecT = out_row[i..][0..lanes].*;
            out_row[i..][0..lanes].* = v - log_denom_vec;
        }
        while (i < N) : (i += 1) {
            out_row[i] -= log_denom;
        }
    }
}

fn logSoftmaxRowsInPlace32(values: []f32, M: usize) void {
    const VecT = @Vector(16, f32);
    for (0..M) |row| {
        const out_row = values[row * 32 ..][0..32];
        const v0: VecT = out_row[0..16].*;
        const v1: VecT = out_row[16..32].*;
        const max_val = @max(@reduce(.Max, v0), @reduce(.Max, v1));
        const max_broadcast: VecT = @splat(max_val);
        const e0 = fastExpApproxVec(16, v0 - max_broadcast);
        const e1 = fastExpApproxVec(16, v1 - max_broadcast);
        const sum_exp = @reduce(.Add, e0) + @reduce(.Add, e1);
        const log_denom_vec: VecT = @splat(max_val + fastLogPositiveApprox(sum_exp));
        out_row[0..16].* = v0 - log_denom_vec;
        out_row[16..32].* = v1 - log_denom_vec;
    }
}

fn logSoftmaxRowsInPlaceBias32(values: []f32, bias: []const f32, M: usize) void {
    const VecT = @Vector(16, f32);
    const b0: VecT = bias[0..16].*;
    const b1: VecT = bias[16..32].*;
    for (0..M) |row| {
        const out_row = values[row * 32 ..][0..32];
        const v0: VecT = out_row[0..16].* + b0;
        const v1: VecT = out_row[16..32].* + b1;
        const max_val = @max(@reduce(.Max, v0), @reduce(.Max, v1));
        const max_broadcast: VecT = @splat(max_val);
        const e0 = fastExpApproxVec(16, v0 - max_broadcast);
        const e1 = fastExpApproxVec(16, v1 - max_broadcast);
        const sum_exp = @reduce(.Add, e0) + @reduce(.Add, e1);
        const log_denom_vec: VecT = @splat(max_val + fastLogPositiveApprox(sum_exp));
        out_row[0..16].* = v0 - log_denom_vec;
        out_row[16..32].* = v1 - log_denom_vec;
    }
}

fn writeLogSoftmax32Row(output_row: []f32, v0: @Vector(16, f32), v1: @Vector(16, f32)) void {
    const VecT = @Vector(16, f32);
    const max_val = @max(@reduce(.Max, v0), @reduce(.Max, v1));
    const max_broadcast: VecT = @splat(max_val);
    const e0 = fastExpApproxVec(16, v0 - max_broadcast);
    const e1 = fastExpApproxVec(16, v1 - max_broadcast);
    const log_denom_vec: VecT = @splat(max_val + fastLogPositiveApprox(@reduce(.Add, e0) + @reduce(.Add, e1)));
    output_row[0..16].* = v0 - log_denom_vec;
    output_row[16..32].* = v1 - log_denom_vec;
}

fn executeSmallDirectLinearBiasLogSoftmax32Rows(
    comptime rows: usize,
    linear: *const TinyLinearSessionHandle,
    shape: DirectLinearLogSoftmaxStepShape,
    input_slice: []const f32,
    output_slice: []f32,
    row_start: usize,
    b0: @Vector(16, f32),
    b1: @Vector(16, f32),
) void {
    const VecT = @Vector(16, f32);
    var lo: [rows]VecT = undefined;
    var hi: [rows]VecT = undefined;
    inline for (0..rows) |r| {
        lo[r] = b0;
        hi[r] = b1;
    }
    for (0..shape.K) |k| {
        const w0: VecT = linear.weights_buf[k * 32 ..][0..16].*;
        const w1: VecT = linear.weights_buf[k * 32 + 16 ..][0..16].*;
        inline for (0..rows) |r| {
            const input_row = input_slice[(row_start + r) * shape.K ..][0..shape.K];
            const x: VecT = @splat(input_row[k]);
            lo[r] += x * w0;
            hi[r] += x * w1;
        }
    }
    inline for (0..rows) |r| {
        const output_row = output_slice[(row_start + r) * 32 ..][0..32];
        writeLogSoftmax32Row(output_row, lo[r], hi[r]);
    }
}

fn executeSmallDirectLinearBiasLogSoftmax32(linear: *const TinyLinearSessionHandle, shape: DirectLinearLogSoftmaxStepShape, input: [*]const f32, output: [*]f32) void {
    const VecT = @Vector(16, f32);
    const input_slice = input[0..linear.input_len];
    const output_slice = output[0..linear.output_len];
    const b0: VecT = linear.bias_buf[0..16].*;
    const b1: VecT = linear.bias_buf[16..32].*;
    var row: usize = 0;
    while (row + 7 < shape.M) : (row += 8) {
        executeSmallDirectLinearBiasLogSoftmax32Rows(8, linear, shape, input_slice, output_slice, row, b0, b1);
    }
    while (row + 3 < shape.M) : (row += 4) {
        executeSmallDirectLinearBiasLogSoftmax32Rows(4, linear, shape, input_slice, output_slice, row, b0, b1);
    }
    while (row + 1 < shape.M) : (row += 2) {
        executeSmallDirectLinearBiasLogSoftmax32Rows(2, linear, shape, input_slice, output_slice, row, b0, b1);
    }
    while (row < shape.M) : (row += 1) {
        executeSmallDirectLinearBiasLogSoftmax32Rows(1, linear, shape, input_slice, output_slice, row, b0, b1);
    }
}

fn shouldUseSmallDirectLinearBiasLogSoftmax32(shape: DirectLinearLogSoftmaxStepShape) bool {
    return shape.has_bias and shape.N == 32 and shape.K <= 128 and shape.M < 64;
}

test "direct linear logsoftmax n32 uses small fused path only for small batches" {
    try std.testing.expect(shouldUseSmallDirectLinearBiasLogSoftmax32(.{
        .M = 8,
        .N = 32,
        .K = 64,
        .has_bias = true,
    }));
    try std.testing.expect(!shouldUseSmallDirectLinearBiasLogSoftmax32(.{
        .M = 128,
        .N = 32,
        .K = 64,
        .has_bias = true,
    }));
}

fn logSoftmaxRowsInPlace(values: []f32, M: usize, N: usize) void {
    if (N == 32) return logSoftmaxRowsInPlace32(values, M);
    if (N % 16 == 0) return logSoftmaxRowsInPlaceLanes(16, values, M, N);
    return logSoftmaxRowsInPlaceLanes(8, values, M, N);
}

test "direct log softmax n32 specialization matches stable row math" {
    var values_buf: [64]f32 = undefined;
    var expected_buf: [64]f32 = undefined;
    for (&values_buf, 0..) |*value, index| {
        const centered: i32 = @as(i32, @intCast(index % 17)) - 8;
        value.* = @as(f32, @floatFromInt(centered)) / 9.0;
    }
    @memcpy(expected_buf[0..], values_buf[0..]);
    logSoftmaxRowsInPlace(values_buf[0..], 2, 32);
    for (0..2) |row| {
        const row_values = expected_buf[row * 32 ..][0..32];
        var max_val = -std.math.inf(f32);
        for (row_values) |value| max_val = @max(max_val, value);
        var sum_exp: f32 = 0;
        for (row_values) |value| sum_exp += @exp(value - max_val);
        const log_denom = max_val + @log(sum_exp);
        for (row_values, 0..) |value, col| {
            try std.testing.expectApproxEqAbs(value - log_denom, values_buf[row * 32 + col], 1e-4);
        }
    }
}

test "direct log softmax n32 bias specialization matches stable row math" {
    var values_buf: [64]f32 = undefined;
    var expected_buf: [64]f32 = undefined;
    var bias_buf: [32]f32 = undefined;
    for (&values_buf, 0..) |*value, index| {
        const centered: i32 = @as(i32, @intCast(index % 19)) - 9;
        value.* = @as(f32, @floatFromInt(centered)) / 11.0;
    }
    for (&bias_buf, 0..) |*value, index| {
        const centered: i32 = @as(i32, @intCast(index % 13)) - 6;
        value.* = @as(f32, @floatFromInt(centered)) / 17.0;
    }
    @memcpy(expected_buf[0..], values_buf[0..]);
    logSoftmaxRowsInPlaceBias32(values_buf[0..], bias_buf[0..], 2);
    for (0..2) |row| {
        const row_values = expected_buf[row * 32 ..][0..32];
        var max_val = -std.math.inf(f32);
        for (row_values, 0..) |value, col| max_val = @max(max_val, value + bias_buf[col]);
        var sum_exp: f32 = 0;
        for (row_values, 0..) |value, col| sum_exp += @exp(value + bias_buf[col] - max_val);
        const log_denom = max_val + @log(sum_exp);
        for (row_values, 0..) |value, col| {
            try std.testing.expectApproxEqAbs(value + bias_buf[col] - log_denom, values_buf[row * 32 + col], 1e-4);
        }
    }
}

fn executeDirectLinearLogSoftmaxStep(linear: *const TinyLinearSessionHandle, shape: DirectLinearLogSoftmaxStepShape, input: [*]const f32, output: [*]f32) void {
    if (shouldUseSmallDirectLinearBiasLogSoftmax32(shape)) {
        executeSmallDirectLinearBiasLogSoftmax32(linear, shape, input, output);
        return;
    }

    if (shape.has_bias and shape.N == 32) {
        executeDirectLinearStep(linear, .{
            .M = shape.M,
            .N = shape.N,
            .K = shape.K,
            .has_bias = false,
        }, input, output, true);
        logSoftmaxRowsInPlaceBias32(output[0..linear.output_len], linear.bias_buf, shape.M);
        return;
    }

    executeDirectLinearStep(linear, .{
        .M = shape.M,
        .N = shape.N,
        .K = shape.K,
        .has_bias = shape.has_bias,
    }, input, output, true);
    logSoftmaxRowsInPlace(output[0..linear.output_len], shape.M, shape.N);
}

export fn zgml_session_step_direct(session: ?*zgml_session, input_ptr: ?[*]const f32, input_len: usize, output_ptr: ?[*]f32, output_len: usize) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const linear = switch (s.data) {
        .tiny_linear => |*linear| linear,
        .tiny_mlp => |*mlp| mlp,
        .module => |*module| module,
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => return status(.unsupported),
    };
    const p = deviceProgramRuntime(linear.program) orelse return status(.invalid_argument);
    if (!p.execution_supported) return status(.unsupported);
    const input = input_ptr orelse return status(.invalid_argument);
    const output = output_ptr orelse return status(.invalid_argument);
    if (input_len != linear.input_len or output_len < linear.output_len) return status(.shape_mismatch);
    if (linear.session.bindings.step_inputs.len != 1 or linear.session.bindings.step_outputs.len != 1) return status(.unsupported);
    if (p.backend == backend_cpu) {
        if (directLinearShapeForSession(s, linear)) |shape| {
            executeDirectLinearStep(linear, shape, input, output, true);
            cpu_mod.recordDirectLinearRuntimeProfile(linear.session.program_handle, linear.session.runtime_handle);
            return status(.ok);
        }
        if (directRmsGeluLinearShapeForSession(s, linear)) |shape| {
            executeDirectRmsGeluLinearStep(linear, shape, input, output);
            cpu_mod.recordDirectLinearRuntimeProfile(linear.session.program_handle, linear.session.runtime_handle);
            return status(.ok);
        }
        if (directLinearLogSoftmaxShapeForSession(s, linear)) |shape| {
            executeDirectLinearLogSoftmaxStep(linear, shape, input, output);
            cpu_mod.recordDirectLinearRuntimeProfile(linear.session.program_handle, linear.session.runtime_handle);
            return status(.ok);
        }
    }

    const original_input = linear.session.bindings.step_inputs[0];
    const original_output = linear.session.bindings.step_outputs[0];
    defer {
        linear.session.bindings.step_inputs[0] = original_input;
        linear.session.bindings.step_outputs[0] = original_output;
    }

    const input_bytes = std.math.mul(usize, input_len, @sizeOf(f32)) catch return status(.shape_mismatch);
    const output_bytes = std.math.mul(usize, linear.output_len, @sizeOf(f32)) catch return status(.shape_mismatch);
    linear.session.bindings.step_inputs[0] = backend_mod.ProgramIO.host(
        original_input.buf_idx,
        original_input.offset,
        @ptrCast(@constCast(input)),
        std.math.cast(u32, input_bytes) orelse return status(.shape_mismatch),
    );
    linear.session.bindings.step_outputs[0] = backend_mod.ProgramIO.host(
        original_output.buf_idx,
        original_output.offset,
        @ptrCast(output),
        std.math.cast(u32, output_bytes) orelse return status(.shape_mismatch),
    );

    const window = backend_mod.RuntimeWindow.init(0, 0) catch return status(.shape_mismatch);
    p.program.executeStep(&linear.session, .{ .window = window, .dynamic_io = true }) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_step_no_output(session: ?*zgml_session, desc_ptr: ?*const zgml_step_desc, result_ptr: ?*zgml_step_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const linear = switch (s.data) {
        .tiny_linear => |*linear| linear,
        .tiny_mlp => |*mlp| mlp,
        .module => |*module| module,
        .tiny_llama, .tiny_llama_2layer, .smollm_135m => return status(.unsupported),
    };
    const p = deviceProgramRuntime(linear.program) orelse return status(.invalid_argument);
    if (p.backend == backend_webgpu) {
        if (maybeWasmHostSessionStepNoOutput(session, desc_ptr, result_ptr)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    if (!p.execution_supported) return status(.unsupported);
    if (desc_ptr == null and linear.owns_input_buf) return status(.invalid_argument);
    if (desc_ptr) |desc| {
        if (desc.input) |input| {
            if (desc.input_len != linear.input_len) return status(.shape_mismatch);
            if (input != linear.input_buf.ptr) {
                @memcpy(linear.input_buf[0..linear.input_len], input[0..linear.input_len]);
            }
        } else if (linear.owns_input_buf) {
            return status(.invalid_argument);
        }
        if (desc.output != null or desc.output_len != 0) return status(.invalid_argument);
    }

    const window = backend_mod.RuntimeWindow.init(0, 0) catch return status(.shape_mismatch);
    p.program.executeStep(&linear.session, .{ .window = window, .download_outputs = false }) catch return status(.shape_mismatch);
    if (result_ptr) |result| {
        result.* = .{ .output_len = 0 };
    }
    return status(.ok);
}

export fn zgml_session_step_token(session: ?*zgml_session, desc_ptr: ?*const zgml_token_step_desc, result_ptr: ?*zgml_step_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    var token = [_]u32{desc.token};
    const execute_desc = zgml_token_execute_desc{
        .tokens = token[0..].ptr,
        .tokens_len = token.len,
        .output_policy = execute_output_logits,
        .output = desc.output,
        .output_len = desc.output_len,
    };
    return executeTokensHandle(session, s, &execute_desc, result_ptr);
}

fn llamaStepOutput(output: ?[*]f32, output_len: usize, bound_output: []f32, vocab_size: usize) ![]f32 {
    if (output) |ptr| {
        if (output_len < vocab_size) return error.OutputBufferTooSmall;
        return ptr[0..vocab_size];
    }
    if (output_len != 0) return error.ShapeMismatch;
    if (bound_output.len < vocab_size) return error.InvalidArgument;
    return bound_output[0..vocab_size];
}

fn copyTokenWindow(tokens: []const u32, storage: []usize) ![]const usize {
    if (tokens.len == 0 or tokens.len > storage.len) return error.ShapeMismatch;
    for (tokens, 0..) |token, i| storage[i] = token;
    return storage[0..tokens.len];
}

fn executeTinyLlamaTokens(
    session: *TinyLlamaSession,
    tokens: []const usize,
    output_policy: u32,
    output: ?[*]f32,
    output_len: usize,
    bound_output: []f32,
    bound_output_is_resource: bool,
) !usize {
    return switch (output_policy) {
        execute_output_none => blk: {
            if (output != null or output_len != 0) return error.ShapeMismatch;
            if (tokens.len == 1) {
                try session.advance(.{ .token = tokens[0] });
            } else {
                try session.advanceTokens(tokens);
            }
            break :blk 0;
        },
        execute_output_logits => blk: {
            if (output == null and output_len == 0 and bound_output_is_resource) {
                if (tokens.len == 1)
                    try session.stepBoundOutput(.{ .token = tokens[0] })
                else
                    try session.prefillBoundOutput(tokens);
                break :blk tiny_llama_config.vocab_size;
            }
            const out = try llamaStepOutput(output, output_len, bound_output, tiny_llama_config.vocab_size);
            const logits = if (tokens.len == 1)
                try session.stepInto(out, .{ .token = tokens[0] })
            else
                try session.prefillInto(out, tokens);
            break :blk logits.len;
        },
        else => error.InvalidArgument,
    };
}

fn executeTinyLlama2LayerTokens(
    session: *TinyLlama2LayerSession,
    tokens: []const usize,
    output_policy: u32,
    output: ?[*]f32,
    output_len: usize,
    bound_output: []f32,
    bound_output_is_resource: bool,
) !usize {
    return switch (output_policy) {
        execute_output_none => blk: {
            if (output != null or output_len != 0) return error.ShapeMismatch;
            if (tokens.len == 1) {
                try session.advance(.{ .token = tokens[0] });
            } else {
                try session.advanceTokens(tokens);
            }
            break :blk 0;
        },
        execute_output_logits => blk: {
            if (output == null and output_len == 0 and bound_output_is_resource) {
                if (tokens.len == 1)
                    try session.stepBoundOutput(.{ .token = tokens[0] })
                else
                    try session.prefillBoundOutput(tokens);
                break :blk tiny_llama_2layer_config.vocab_size;
            }
            const out = try llamaStepOutput(output, output_len, bound_output, tiny_llama_2layer_config.vocab_size);
            const logits = if (tokens.len == 1)
                try session.stepInto(out, .{ .token = tokens[0] })
            else
                try session.prefillInto(out, tokens);
            break :blk logits.len;
        },
        else => error.InvalidArgument,
    };
}

fn executeSmolLMTokens(
    session: *SmolLM135MSession,
    tokens: []const usize,
    output_policy: u32,
    output: ?[*]f32,
    output_len: usize,
    bound_output: []f32,
    bound_output_is_resource: bool,
) !usize {
    return switch (output_policy) {
        execute_output_none => blk: {
            if (output != null or output_len != 0) return error.ShapeMismatch;
            if (tokens.len == 1) {
                try session.advance(.{ .token = tokens[0] });
            } else {
                try session.advanceTokens(tokens);
            }
            break :blk 0;
        },
        execute_output_logits => blk: {
            if (output == null and output_len == 0 and bound_output_is_resource) {
                if (tokens.len == 1)
                    try session.stepBoundOutput(.{ .token = tokens[0] })
                else
                    try session.prefillBoundOutput(tokens);
                break :blk smollm_135m_config.vocab_size;
            }
            const out = try llamaStepOutput(output, output_len, bound_output, smollm_135m_config.vocab_size);
            const logits = if (tokens.len == 1)
                try session.stepInto(out, .{ .token = tokens[0] })
            else
                try session.prefillInto(out, tokens);
            break :blk logits.len;
        },
        else => error.InvalidArgument,
    };
}

export fn zgml_session_advance_token(session: ?*zgml_session, desc_ptr: ?*const zgml_token_advance_desc) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    var token = [_]u32{desc.token};
    const execute_desc = zgml_token_execute_desc{
        .tokens = token[0..].ptr,
        .tokens_len = token.len,
        .output_policy = execute_output_none,
        .output = null,
        .output_len = 0,
    };
    return executeTokensHandle(session, s, &execute_desc, null);
}

export fn zgml_session_advance_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_advance_tokens_desc) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const execute_desc = zgml_token_execute_desc{
        .tokens = desc.tokens,
        .tokens_len = desc.tokens_len,
        .output_policy = execute_output_none,
        .output = null,
        .output_len = 0,
    };
    return executeTokensHandle(session, s, &execute_desc, null);
}

export fn zgml_session_prefill_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_prefill_desc, result_ptr: ?*zgml_step_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const execute_desc = zgml_token_execute_desc{
        .tokens = desc.tokens,
        .tokens_len = desc.tokens_len,
        .output_policy = execute_output_logits,
        .output = desc.output,
        .output_len = desc.output_len,
    };
    return executeTokensHandle(session, s, &execute_desc, result_ptr);
}

fn sessionBackend(s: *SessionHandle) u32 {
    return switch (s.data) {
        .tiny_linear => |*linear| switch (linear.program.data) {
            .tiny_linear => |*program| program.backend,
            .tiny_mlp, .module, .tiny_llama, .tiny_llama_2layer, .smollm_135m => backend_auto,
        },
        .tiny_mlp => |*mlp| switch (mlp.program.data) {
            .tiny_mlp => |*program| program.backend,
            .tiny_linear, .module, .tiny_llama, .tiny_llama_2layer, .smollm_135m => backend_auto,
        },
        .module => |*module| switch (module.program.data) {
            .module => |*program| program.backend,
            .tiny_linear, .tiny_mlp, .tiny_llama, .tiny_llama_2layer, .smollm_135m => backend_auto,
        },
        .tiny_llama => |*llama| switch (llama.program.data) {
            .tiny_llama => |*program| @intCast(backendIdForLlamaBackend(program.program.inspectExecutable().backend)),
            .tiny_linear, .tiny_mlp, .module, .tiny_llama_2layer, .smollm_135m => backend_auto,
        },
        .tiny_llama_2layer => |*llama| switch (llama.program.data) {
            .tiny_llama_2layer => |*program| @intCast(backendIdForLlamaBackend(program.program.inspectExecutable().backend)),
            .tiny_linear, .tiny_mlp, .module, .tiny_llama, .smollm_135m => backend_auto,
        },
        .smollm_135m => |*llama| switch (llama.program.data) {
            .smollm_135m => |*program| @intCast(backendIdForLlamaBackend(program.program.inspectExecutable().backend)),
            .tiny_linear, .tiny_mlp, .module, .tiny_llama, .tiny_llama_2layer => backend_auto,
        },
    };
}

fn executeTokensHandle(session: ?*zgml_session, s: *SessionHandle, desc: *const zgml_token_execute_desc, result_ptr: ?*zgml_step_result) c_int {
    if (result_ptr) |result| result.* = .{};
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.tokens == null) return status(.invalid_argument);
    if (desc.tokens_len == 0) return status(.shape_mismatch);
    if (sessionBackend(s) == backend_webgpu) {
        if (maybeWasmHostSessionExecuteTokens(session, desc, result_ptr)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    const raw_tokens = desc.tokens.?[0..desc.tokens_len];
    const output_len = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return status(.unsupported),
        .tiny_llama => |*llama| blk: {
            var token_storage: [tiny_llama_config.max_seq_len]usize = undefined;
            const tokens = copyTokenWindow(raw_tokens, &token_storage) catch |err| return compileErrorStatus(err);
            break :blk executeTinyLlamaTokens(
                llama.session,
                tokens,
                desc.output_policy,
                desc.output,
                desc.output_len,
                llama.output_buf,
                llama.output_is_resource,
            ) catch |err| return compileErrorStatus(err);
        },
        .tiny_llama_2layer => |*llama| blk: {
            var token_storage: [tiny_llama_2layer_config.max_seq_len]usize = undefined;
            const tokens = copyTokenWindow(raw_tokens, &token_storage) catch |err| return compileErrorStatus(err);
            break :blk executeTinyLlama2LayerTokens(
                llama.session,
                tokens,
                desc.output_policy,
                desc.output,
                desc.output_len,
                llama.output_buf,
                llama.output_is_resource,
            ) catch |err| return compileErrorStatus(err);
        },
        .smollm_135m => |*llama| blk: {
            var token_storage: [smollm_135m_config.max_seq_len]usize = undefined;
            const tokens = copyTokenWindow(raw_tokens, &token_storage) catch |err| return compileErrorStatus(err);
            break :blk executeSmolLMTokens(
                llama.session,
                tokens,
                desc.output_policy,
                desc.output,
                desc.output_len,
                llama.output_buf,
                llama.output_is_resource,
            ) catch |err| return compileErrorStatus(err);
        },
    };
    if (result_ptr) |result| {
        result.* = .{ .output_len = output_len };
    }
    return status(.ok);
}

export fn zgml_session_execute_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_execute_desc, result_ptr: ?*zgml_step_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    return executeTokensHandle(session, s, desc, result_ptr);
}

fn argmaxToken(logits: []const f32) zgml_token_argmax_result {
    const selected = sampling.argmax(f32, logits) catch unreachable;
    std.debug.assert(selected.token <= std.math.maxInt(u32));
    return .{ .token = @intCast(selected.token), .logit = selected.logit };
}

fn argmaxLogits(desc_ptr: ?*const zgml_token_argmax_desc, bound_output: []const f32, vocab_size: usize) ![]const f32 {
    const desc = desc_ptr orelse {
        if (bound_output.len < vocab_size) return error.InvalidArgument;
        return bound_output[0..vocab_size];
    };
    if (desc.reserved != 0) return error.InvalidArgument;
    if (desc.logits) |logits| {
        if (desc.logits_len < vocab_size) return error.OutputBufferTooSmall;
        return logits[0..vocab_size];
    }
    if (desc.logits_len != 0) return error.ShapeMismatch;
    if (bound_output.len < vocab_size) return error.InvalidArgument;
    return bound_output[0..vocab_size];
}

fn sampleLogits(desc: *const zgml_token_sample_desc, bound_output: []const f32, vocab_size: usize) ![]const f32 {
    if (desc.reserved != 0) return error.InvalidArgument;
    if (desc.logits) |logits| {
        if (desc.logits_len < vocab_size) return error.OutputBufferTooSmall;
        return logits[0..vocab_size];
    }
    if (desc.logits_len != 0) return error.ShapeMismatch;
    if (bound_output.len < vocab_size) return error.InvalidArgument;
    return bound_output[0..vocab_size];
}

fn validateSampleOptions(top_k: u32, temperature: f32) !void {
    try sampling.validateOptions(.{
        .top_k = @intCast(top_k),
        .temperature = @floatCast(temperature),
    });
}

fn sampleTopKToken(logits: []const f32, top_k: u32, seed: u32, temperature: f32) !zgml_token_sample_result {
    const selected = try sampling.sampleTopK(f32, logits, .{
        .top_k = @intCast(top_k),
        .seed = seed,
        .temperature = @floatCast(temperature),
    });
    if (selected.token > std.math.maxInt(u32)) return error.ShapeMismatch;
    return .{ .token = @intCast(selected.token), .logit = selected.logit };
}

fn refreshBoundResourceLogits(s: *SessionHandle) !void {
    switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return error.Unsupported,
        .tiny_llama => |*llama| {
            const resource = llama.output_resource orelse return;
            if (llama.output_buf.len < tiny_llama_config.vocab_size) return error.InvalidArgument;
            const bytes = std.mem.sliceAsBytes(llama.output_buf[0..tiny_llama_config.vocab_size]);
            try readDeviceBuffer(.{ .program = llama.program }, resource, 0, bytes);
        },
        .tiny_llama_2layer => |*llama| {
            const resource = llama.output_resource orelse return;
            if (llama.output_buf.len < tiny_llama_2layer_config.vocab_size) return error.InvalidArgument;
            const bytes = std.mem.sliceAsBytes(llama.output_buf[0..tiny_llama_2layer_config.vocab_size]);
            try readDeviceBuffer(.{ .program = llama.program }, resource, 0, bytes);
        },
        .smollm_135m => |*llama| {
            const resource = llama.output_resource orelse return;
            if (llama.output_buf.len < smollm_135m_config.vocab_size) return error.InvalidArgument;
            const bytes = std.mem.sliceAsBytes(llama.output_buf[0..smollm_135m_config.vocab_size]);
            try readDeviceBuffer(.{ .program = llama.program }, resource, 0, bytes);
        },
    }
}

fn sampleBoundToken(s: *SessionHandle, top_k: u32, seed: u32, temperature: f32) !zgml_token_sample_result {
    const sample_desc = zgml_token_sample_desc{
        .logits = null,
        .logits_len = 0,
        .top_k = top_k,
        .seed = seed,
        .temperature = temperature,
        .reserved = 0,
    };
    const logits = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return error.Unsupported,
        .tiny_llama => |*llama| try sampleLogits(&sample_desc, llama.output_buf, tiny_llama_config.vocab_size),
        .tiny_llama_2layer => |*llama| try sampleLogits(&sample_desc, llama.output_buf, tiny_llama_2layer_config.vocab_size),
        .smollm_135m => |*llama| try sampleLogits(&sample_desc, llama.output_buf, smollm_135m_config.vocab_size),
    };
    return sampleTopKToken(logits, top_k, seed, temperature);
}

fn argmaxBoundToken(s: *SessionHandle) !zgml_token_argmax_result {
    const logits = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return error.Unsupported,
        .tiny_llama => |*llama| try argmaxLogits(null, llama.output_buf, tiny_llama_config.vocab_size),
        .tiny_llama_2layer => |*llama| try argmaxLogits(null, llama.output_buf, tiny_llama_2layer_config.vocab_size),
        .smollm_135m => |*llama| try argmaxLogits(null, llama.output_buf, smollm_135m_config.vocab_size),
    };
    return argmaxToken(logits);
}

fn executeBoundLogits(session: ?*zgml_session, s: *SessionHandle, tokens: ?[*]const u32, tokens_len: usize) c_int {
    const execute_desc = zgml_token_execute_desc{
        .tokens = tokens,
        .tokens_len = tokens_len,
        .output_policy = execute_output_logits,
        .reserved = 0,
        .output = null,
        .output_len = 0,
    };
    return executeTokensHandle(session, s, &execute_desc, null);
}

fn validateGenerateCapacity(s: *SessionHandle, prompt_len: usize, output_len: usize) !void {
    const required_steps = std.math.add(usize, prompt_len, output_len - 1) catch return error.ShapeMismatch;
    switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return error.Unsupported,
        .tiny_llama => |*llama| {
            const context_len = switch (llama.program.data) {
                .tiny_llama => |*program| program.program.inspect().context_len,
                else => return error.InvalidArgument,
            };
            const final_position = std.math.add(usize, llama.session.position(), required_steps) catch return error.ShapeMismatch;
            if (final_position > context_len) return error.SequenceTooLong;
        },
        .tiny_llama_2layer => |*llama| {
            const context_len = switch (llama.program.data) {
                .tiny_llama_2layer => |*program| program.program.inspect().context_len,
                else => return error.InvalidArgument,
            };
            const final_position = std.math.add(usize, llama.session.position(), required_steps) catch return error.ShapeMismatch;
            if (final_position > context_len) return error.SequenceTooLong;
        },
        .smollm_135m => |*llama| {
            const context_len = switch (llama.program.data) {
                .smollm_135m => |*program| program.program.inspect().context_len,
                else => return error.InvalidArgument,
            };
            const final_position = std.math.add(usize, llama.session.position(), required_steps) catch return error.ShapeMismatch;
            if (final_position > context_len) return error.SequenceTooLong;
        },
    }
}

export fn zgml_session_argmax_token(session: ?*zgml_session, desc_ptr: ?*const zgml_token_argmax_desc, result_ptr: ?*zgml_token_argmax_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    if (desc_ptr) |desc| {
        if (desc.reserved != 0) return status(.invalid_argument);
        if (desc.logits == null and desc.logits_len == 0) {
            refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);
        }
    } else {
        refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);
    }
    const logits = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return status(.unsupported),
        .tiny_llama => |*llama| argmaxLogits(desc_ptr, llama.output_buf, tiny_llama_config.vocab_size) catch |err| return compileErrorStatus(err),
        .tiny_llama_2layer => |*llama| argmaxLogits(desc_ptr, llama.output_buf, tiny_llama_2layer_config.vocab_size) catch |err| return compileErrorStatus(err),
        .smollm_135m => |*llama| argmaxLogits(desc_ptr, llama.output_buf, smollm_135m_config.vocab_size) catch |err| return compileErrorStatus(err),
    };
    result.* = argmaxToken(logits);
    return status(.ok);
}

export fn zgml_session_sample_token(session: ?*zgml_session, desc_ptr: ?*const zgml_token_sample_desc, result_ptr: ?*zgml_token_sample_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.logits == null and desc.logits_len == 0) {
        refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);
    }
    const logits = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => return status(.unsupported),
        .tiny_llama => |*llama| sampleLogits(desc, llama.output_buf, tiny_llama_config.vocab_size) catch |err| return compileErrorStatus(err),
        .tiny_llama_2layer => |*llama| sampleLogits(desc, llama.output_buf, tiny_llama_2layer_config.vocab_size) catch |err| return compileErrorStatus(err),
        .smollm_135m => |*llama| sampleLogits(desc, llama.output_buf, smollm_135m_config.vocab_size) catch |err| return compileErrorStatus(err),
    };
    result.* = sampleTopKToken(logits, desc.top_k, desc.seed, desc.temperature) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_execute_argmax_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_execute_argmax_desc, result_ptr: ?*zgml_token_argmax_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const desc = desc_ptr orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    if (desc.reserved != 0) return status(.invalid_argument);

    const execute_status = executeBoundLogits(session, s, desc.tokens, desc.tokens_len);
    if (execute_status != status(.ok)) return execute_status;
    refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);

    result.* = argmaxBoundToken(s) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_generate_argmax_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_generate_argmax_desc, result_ptr: ?*zgml_token_generate_argmax_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.tokens == null or desc.output_tokens == null) return status(.invalid_argument);
    if (desc.tokens_len == 0 or desc.output_tokens_len == 0) return status(.shape_mismatch);
    validateGenerateCapacity(s, desc.tokens_len, desc.output_tokens_len) catch |err| return compileErrorStatus(err);

    const generated_tokens = desc.output_tokens.?[0..desc.output_tokens_len];
    const initial_status = executeBoundLogits(session, s, desc.tokens, desc.tokens_len);
    if (initial_status != status(.ok)) return initial_status;
    refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);

    var selected = argmaxBoundToken(s) catch |err| return compileErrorStatus(err);
    generated_tokens[0] = selected.token;
    var generated: usize = 1;

    while (generated < generated_tokens.len) {
        var token = [_]u32{selected.token};
        const execute_status = executeBoundLogits(session, s, token[0..].ptr, token.len);
        if (execute_status != status(.ok)) return execute_status;
        refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);
        selected = argmaxBoundToken(s) catch |err| return compileErrorStatus(err);
        generated_tokens[generated] = selected.token;
        generated += 1;
    }

    result.* = .{
        .tokens_generated = generated,
        .last_token = selected.token,
        .last_logit = selected.logit,
    };
    return status(.ok);
}

export fn zgml_session_execute_sample_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_execute_sample_desc, result_ptr: ?*zgml_token_sample_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    validateSampleOptions(desc.top_k, desc.temperature) catch |err| return compileErrorStatus(err);

    const execute_status = executeBoundLogits(session, s, desc.tokens, desc.tokens_len);
    if (execute_status != status(.ok)) return execute_status;
    refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);

    result.* = sampleBoundToken(s, desc.top_k, desc.seed, desc.temperature) catch |err| return compileErrorStatus(err);
    return status(.ok);
}

export fn zgml_session_generate_sample_tokens(session: ?*zgml_session, desc_ptr: ?*const zgml_token_generate_sample_desc, result_ptr: ?*zgml_token_generate_sample_result) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const result = result_ptr orelse return status(.invalid_argument);
    result.* = .{};
    const desc = desc_ptr orelse return status(.invalid_argument);
    if (desc.reserved != 0) return status(.invalid_argument);
    if (desc.tokens == null or desc.output_tokens == null) return status(.invalid_argument);
    if (desc.tokens_len == 0 or desc.output_tokens_len == 0) return status(.shape_mismatch);
    validateSampleOptions(desc.top_k, desc.temperature) catch |err| return compileErrorStatus(err);
    validateGenerateCapacity(s, desc.tokens_len, desc.output_tokens_len) catch |err| return compileErrorStatus(err);

    const generated_tokens = desc.output_tokens.?[0..desc.output_tokens_len];
    const initial_status = executeBoundLogits(session, s, desc.tokens, desc.tokens_len);
    if (initial_status != status(.ok)) return initial_status;
    refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);

    var sampled = sampleBoundToken(s, desc.top_k, desc.seed, desc.temperature) catch |err| return compileErrorStatus(err);
    generated_tokens[0] = sampled.token;
    var generated: usize = 1;

    while (generated < generated_tokens.len) {
        var token = [_]u32{sampled.token};
        const execute_status = executeBoundLogits(session, s, token[0..].ptr, token.len);
        if (execute_status != status(.ok)) return execute_status;
        refreshBoundResourceLogits(s) catch |err| return compileErrorStatus(err);
        if (generated > std.math.maxInt(u32)) return status(.shape_mismatch);
        const seed = desc.seed +% @as(u32, @intCast(generated));
        sampled = sampleBoundToken(s, desc.top_k, seed, desc.temperature) catch |err| return compileErrorStatus(err);
        generated_tokens[generated] = sampled.token;
        generated += 1;
    }

    result.* = .{
        .tokens_generated = generated,
        .last_token = sampled.token,
        .last_logit = sampled.logit,
    };
    return status(.ok);
}

export fn zgml_session_position(session: ?*zgml_session, out_position: ?*usize) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const out = out_position orelse return status(.invalid_argument);
    if (sessionBackend(s) == backend_webgpu) {
        if (maybeWasmHostSessionPosition(session, out_position)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    out.* = switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => 0,
        .tiny_llama => |*llama| llama.session.position(),
        .tiny_llama_2layer => |*llama| llama.session.position(),
        .smollm_135m => |*llama| llama.session.position(),
    };
    return status(.ok);
}

export fn zgml_session_inspect(session: ?*zgml_session, out_inspection: ?*zgml_session_inspection) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const out = out_inspection orelse return status(.invalid_argument);
    out.* = .{};
    switch (s.data) {
        .tiny_linear => |*linear| fillDeviceSessionInspection(out, tiny_linear_kind, 0, &linear.session),
        .tiny_mlp => |*mlp| fillDeviceSessionInspection(out, tiny_mlp_kind, 0, &mlp.session),
        .module => |*module| fillDeviceSessionInspection(out, module_kind, 0, &module.session),
        .tiny_llama => |*llama| fillLlamaSessionInspection(out, tiny_llama_kind, llama.session.inspect()),
        .tiny_llama_2layer => |*llama| fillLlamaSessionInspection(out, tiny_llama_2layer_kind, llama.session.inspect()),
        .smollm_135m => |*llama| fillLlamaSessionInspection(out, smollm_135m_kind, llama.session.inspect()),
    }
    if (sessionBackend(s) == backend_webgpu) {
        var host_position: usize = 0;
        if (maybeWasmHostSessionPosition(session, &host_position)) |host_status| {
            if (host_status != wasm_host_unclaimed) {
                if (host_status != status(.ok)) return host_status;
                out.position = @intCast(host_position);
            }
        }
    }
    return status(.ok);
}

export fn zgml_session_reset(session: ?*zgml_session) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    if (sessionBackend(s) == backend_webgpu) {
        if (maybeWasmHostSessionReset(session)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    switch (s.data) {
        .tiny_linear, .tiny_mlp, .module => {},
        .tiny_llama => |*llama| llama.session.reset(),
        .tiny_llama_2layer => |*llama| llama.session.reset(),
        .smollm_135m => |*llama| llama.session.reset(),
    }
    return status(.ok);
}

export fn zgml_session_runtime_profile(session: ?*zgml_session, out_profile: ?*zgml_runtime_profile) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    const out = out_profile orelse return status(.invalid_argument);
    if (sessionBackend(s) == backend_webgpu) {
        if (maybeWasmHostSessionRuntimeProfile(session, out_profile)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    var rt = profile_mod.RuntimeProfile{};
    addSessionRuntimeProfileTo(s, &rt);
    fillRuntimeProfile(out, rt);
    return status(.ok);
}

export fn zgml_session_reset_runtime_profile(session: ?*zgml_session) c_int {
    const s = sessionHandle(session) orelse return status(.invalid_argument);
    if (sessionBackend(s) == backend_webgpu) {
        if (maybeWasmHostSessionResetRuntimeProfile(session)) |host_status| {
            if (host_status != wasm_host_unclaimed) return host_status;
        }
    }
    resetSessionRuntimeProfile(s);
    return status(.ok);
}

export fn zgml_session_free(session: ?*zgml_session) void {
    const s = sessionHandle(session) orelse return;
    maybeWasmHostSessionFree(session);
    s.deinit(alloc);
}

export fn zgml_program_free(program: ?*zgml_program) void {
    const p = programHandle(program) orelse return;
    p.release(alloc);
}

export fn zgml_model_free(model: ?*zgml_model) void {
    const m = modelHandle(model) orelse return;
    m.release(alloc);
}

test "C ABI runtime info reports compatible handle surface" {
    var info = zgml_runtime_info{};
    try std.testing.expectEqual(status(.invalid_argument), zgml_get_runtime_info(null));
    try std.testing.expectEqual(status(.ok), zgml_get_runtime_info(&info));
    try std.testing.expectEqual(@as(u32, abi_version), info.abi_version);
    try std.testing.expectEqual(@as(u32, @sizeOf(usize)), info.size_t_bytes);
    try std.testing.expectEqual(@as(u32, @sizeOf(usize)), info.pointer_bytes);
    try std.testing.expectEqual(@as(u32, @sizeOf(u32)), info.token_id_bytes);
    try std.testing.expect((info.feature_flags & feature_buffer_handle) != 0);
    try std.testing.expect((info.feature_flags & feature_model_auto) != 0);
    try std.testing.expect((info.feature_flags & feature_runtime_profile) != 0);
    try std.testing.expect((info.feature_flags & feature_webgpu_compile_only) != 0);
    try std.testing.expect((info.feature_flags & feature_wasm_exports) != 0);
    try std.testing.expect((info.feature_flags & feature_native_buffer_io) != 0);
    try std.testing.expect((info.feature_flags & feature_native_argmax) != 0);
    try std.testing.expect((info.feature_flags & feature_native_execute_argmax) != 0);
    try std.testing.expect((info.feature_flags & feature_native_topk_sample) != 0);
    try std.testing.expect((info.feature_flags & feature_program_requirements) != 0);
    try std.testing.expect((info.feature_flags & feature_program_output_buffer) != 0);
    try std.testing.expect((info.feature_flags & feature_native_generate_sample) != 0);
    try std.testing.expect((info.feature_flags & feature_native_generate_argmax) != 0);
    try std.testing.expect((info.feature_flags & feature_session_model_binding) != 0);
    try std.testing.expect((info.feature_flags & feature_external_buffer) != 0);
    try std.testing.expect((info.feature_flags & feature_external_resource_buffer) != 0);
    try std.testing.expect((info.feature_flags & feature_llama_kv_resource_binding) != 0);
    try std.testing.expect((info.feature_flags & feature_llama_kv_cache_requirements) != 0);
    try std.testing.expect((info.feature_flags & feature_program_buffer_factory) != 0);
    try std.testing.expect((info.feature_flags & feature_external_resource_access) != 0);
    try std.testing.expect((info.feature_flags & feature_model_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_model_compatibility) != 0);
    try std.testing.expect((info.feature_flags & feature_buffer_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_session_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_resource_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_memory_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_shape_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_patch_envelope_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_session_binding_shape_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_program_device_buffer) != 0);
    try std.testing.expect((info.feature_flags & feature_program_device_buffer_import) != 0);
    try std.testing.expect((info.feature_flags & feature_program_dispatch_plan_inspection) != 0);
    try std.testing.expect((info.feature_flags & feature_abi_struct_size) != 0);
    try std.testing.expect((info.feature_flags & feature_model_path_probe) != 0);
    try std.testing.expect((info.feature_flags & feature_supported_checkpoints) != 0);
    try std.testing.expect((info.feature_flags & feature_safetensors_header_probe) != 0);
    try std.testing.expect((info.feature_flags & feature_safetensors_data_load) != 0);
    try std.testing.expect((info.feature_flags & feature_safetensors_data_probe) != 0);
    try std.testing.expect((info.feature_flags & feature_native_tiny_mlp) != 0);
    try std.testing.expect((info.feature_flags & feature_native_module_program) != 0);
    try std.testing.expect((info.feature_flags & feature_program_binding_requirements) != 0);
    try std.testing.expect((info.feature_flags & feature_session_persistent_upload) != 0);
    try std.testing.expect((info.feature_flags & feature_native_module_activation_chain) != 0);
    try std.testing.expect((info.feature_flags & feature_native_training_step) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_activation) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_elementwise) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_reduce) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_conv2d) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_pool2d) != 0);
    try std.testing.expect((info.feature_flags & feature_native_eager_dot) != 0);
    try std.testing.expectEqual(build_options.use_wgpu, (info.feature_flags & feature_native_wgpu_execution) != 0);
    try std.testing.expectEqual(build_options.use_wgpu and build_options.experimental_llama_wgpu_execution, (info.feature_flags & feature_experimental_llama_wgpu_execution) != 0);
    try std.testing.expect((info.feature_flags & feature_experimental_llama_wgpu_execution) == 0 or (info.feature_flags & feature_native_wgpu_execution) != 0);

    try std.testing.expectEqual(@sizeOf(zgml_runtime_info), zgml_abi_struct_size(abi_struct_runtime_info));
    try std.testing.expectEqual(@sizeOf(zgml_model_desc), zgml_abi_struct_size(abi_struct_model_desc));
    try std.testing.expectEqual(@sizeOf(zgml_model_load_desc), zgml_abi_struct_size(abi_struct_model_load_desc));
    try std.testing.expectEqual(@sizeOf(zgml_model_inspection), zgml_abi_struct_size(abi_struct_model_inspection));
    try std.testing.expectEqual(@sizeOf(zgml_program_model_compatibility), zgml_abi_struct_size(abi_struct_program_model_compatibility));
    try std.testing.expectEqual(@sizeOf(zgml_session_inspection), zgml_abi_struct_size(abi_struct_session_inspection));
    try std.testing.expectEqual(@sizeOf(zgml_compile_desc), zgml_abi_struct_size(abi_struct_compile_desc));
    try std.testing.expectEqual(@sizeOf(zgml_bind_desc), zgml_abi_struct_size(abi_struct_bind_desc));
    try std.testing.expectEqual(@sizeOf(zgml_buffer_desc), zgml_abi_struct_size(abi_struct_buffer_desc));
    try std.testing.expectEqual(@sizeOf(zgml_buffer_inspection), zgml_abi_struct_size(abi_struct_buffer_inspection));
    try std.testing.expectEqual(@sizeOf(zgml_external_resource_desc), zgml_abi_struct_size(abi_struct_external_resource_desc));
    try std.testing.expectEqual(@sizeOf(zgml_device_buffer_import_desc), zgml_abi_struct_size(abi_struct_device_buffer_import_desc));
    try std.testing.expectEqual(@sizeOf(zgml_buffer_bind_desc), zgml_abi_struct_size(abi_struct_buffer_bind_desc));
    try std.testing.expectEqual(@sizeOf(zgml_llama_kv_cache_bind_desc), zgml_abi_struct_size(abi_struct_llama_kv_cache_bind_desc));
    try std.testing.expectEqual(@sizeOf(zgml_llama_buffer_bind_desc), zgml_abi_struct_size(abi_struct_llama_buffer_bind_desc));
    try std.testing.expectEqual(@sizeOf(zgml_step_desc), zgml_abi_struct_size(abi_struct_step_desc));
    try std.testing.expectEqual(@sizeOf(zgml_step_result), zgml_abi_struct_size(abi_struct_step_result));
    try std.testing.expectEqual(@sizeOf(zgml_token_step_desc), zgml_abi_struct_size(abi_struct_token_step_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_advance_desc), zgml_abi_struct_size(abi_struct_token_advance_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_advance_tokens_desc), zgml_abi_struct_size(abi_struct_token_advance_tokens_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_prefill_desc), zgml_abi_struct_size(abi_struct_token_prefill_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_execute_desc), zgml_abi_struct_size(abi_struct_token_execute_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_argmax_desc), zgml_abi_struct_size(abi_struct_token_argmax_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_argmax_result), zgml_abi_struct_size(abi_struct_token_argmax_result));
    try std.testing.expectEqual(@sizeOf(zgml_token_execute_argmax_desc), zgml_abi_struct_size(abi_struct_token_execute_argmax_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_generate_argmax_desc), zgml_abi_struct_size(abi_struct_token_generate_argmax_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_generate_argmax_result), zgml_abi_struct_size(abi_struct_token_generate_argmax_result));
    try std.testing.expectEqual(@sizeOf(zgml_token_sample_desc), zgml_abi_struct_size(abi_struct_token_sample_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_sample_result), zgml_abi_struct_size(abi_struct_token_sample_result));
    try std.testing.expectEqual(@sizeOf(zgml_token_execute_sample_desc), zgml_abi_struct_size(abi_struct_token_execute_sample_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_generate_sample_desc), zgml_abi_struct_size(abi_struct_token_generate_sample_desc));
    try std.testing.expectEqual(@sizeOf(zgml_token_generate_sample_result), zgml_abi_struct_size(abi_struct_token_generate_sample_result));
    try std.testing.expectEqual(@sizeOf(zgml_program_requirements), zgml_abi_struct_size(abi_struct_program_requirements));
    try std.testing.expectEqual(@sizeOf(zgml_llama_kv_cache_requirements), zgml_abi_struct_size(abi_struct_llama_kv_cache_requirements));
    try std.testing.expectEqual(@sizeOf(zgml_program_inspection), zgml_abi_struct_size(abi_struct_program_inspection));
    try std.testing.expectEqual(@sizeOf(zgml_llama_program_inspection), zgml_abi_struct_size(abi_struct_llama_program_inspection));
    try std.testing.expectEqual(@sizeOf(zgml_runtime_profile), zgml_abi_struct_size(abi_struct_runtime_profile));
    try std.testing.expectEqual(@sizeOf(zgml_safetensors_header_probe_desc), zgml_abi_struct_size(abi_struct_safetensors_header_probe_desc));
    try std.testing.expectEqual(@sizeOf(zgml_safetensors_data_load_desc), zgml_abi_struct_size(abi_struct_safetensors_data_load_desc));
    try std.testing.expectEqual(@sizeOf(zgml_module_op_desc), zgml_abi_struct_size(abi_struct_module_op_desc));
    try std.testing.expectEqual(@sizeOf(zgml_module_desc), zgml_abi_struct_size(abi_struct_module_desc));
    try std.testing.expectEqual(@as(usize, 0), zgml_abi_struct_size(0));
    try std.testing.expectEqual(@as(usize, 0), zgml_abi_struct_size(9999));
}

test "C ABI model inspection exposes handle envelopes before compile" {
    var linear_model: ?*zgml_model = null;
    var llama_model: ?*zgml_model = null;
    defer zgml_model_free(llama_model);
    defer zgml_model_free(linear_model);

    var inspection = zgml_model_inspection{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_inspect(null, &inspection));
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_inspect(linear_model, null));
    try std.testing.expectEqual(@as(u32, 99), inspection.model_kind);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 3,
    }, &linear_model));

    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_model_inspect(linear_model, &inspection));
    try std.testing.expectEqual(tiny_linear_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, 2), inspection.input_len);
    try std.testing.expectEqual(@as(u64, 3), inspection.output_len);
    try std.testing.expectEqual(@as(u64, 0), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, 0), inspection.d_model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &llama_model));

    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_model_inspect(llama_model, &inspection));
    try std.testing.expectEqual(tiny_llama_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, 0), inspection.input_len);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.d_ff), inspection.d_ff);
    try std.testing.expectApproxEqAbs(@as(f64, tiny_llama_config.rope_base), inspection.rope_base, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, tiny_llama_config.rms_norm_eps), inspection.rms_norm_eps, 0.0);
    try std.testing.expectEqual(@as(u64, @intFromBool(tiny_llama_config.tied_lm_head)), inspection.tied_lm_head);
}

test "C ABI LLaMA KV cache sizing matches packed executable context layout" {
    const tiny_context = try llamaKvCacheElementCount(tiny_llama_config, 4);
    try std.testing.expectEqual(@as(usize, 4 * 4), tiny_context);

    const smollm_context = try llamaKvCacheElementCount(smollm_135m_config, 16);
    const smollm_d_head = smollm_135m_config.d_model / smollm_135m_config.n_heads;
    const smollm_packed = smollm_d_head * 16 * smollm_135m_config.n_kv_heads;
    try std.testing.expectEqual(smollm_packed, smollm_context);
    try std.testing.expect(smollm_context < smollm_d_head * ((smollm_135m_config.n_kv_heads - 1) * smollm_135m_config.max_seq_len + 16));
}

test "C ABI program model compatibility preflights dynamic model binding" {
    var linear_model: ?*zgml_model = null;
    var linear_program: ?*zgml_program = null;
    var llama_model: ?*zgml_model = null;
    var compatible_llama_model: ?*zgml_model = null;
    var llama_program: ?*zgml_program = null;
    defer zgml_program_free(llama_program);
    defer zgml_model_free(compatible_llama_model);
    defer zgml_model_free(llama_model);
    defer zgml_program_free(linear_program);
    defer zgml_model_free(linear_model);

    var compatibility = zgml_program_model_compatibility{ .compatible = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_program_check_model_compatibility(null, linear_model, &compatibility));
    try std.testing.expectEqual(status(.invalid_argument), zgml_program_check_model_compatibility(linear_program, null, &compatibility));
    try std.testing.expectEqual(status(.invalid_argument), zgml_program_check_model_compatibility(linear_program, linear_model, null));
    try std.testing.expectEqual(@as(u64, 99), compatibility.compatible);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &linear_model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(linear_model, &.{ .backend = backend_cpu }, &linear_program));

    compatibility = .{};
    try std.testing.expectEqual(status(.ok), zgml_program_check_model_compatibility(linear_program, linear_model, &compatibility));
    try std.testing.expectEqual(tiny_linear_kind, compatibility.program_model_kind);
    try std.testing.expectEqual(tiny_linear_kind, compatibility.model_kind);
    try std.testing.expectEqual(@as(u64, 0), compatibility.compatible);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &llama_model));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &compatible_llama_model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(llama_model, &.{
        .backend = backend_cpu,
        .context_len = 4,
        .batch = 1,
    }, &llama_program));

    compatibility = .{};
    try std.testing.expectEqual(status(.ok), zgml_program_check_model_compatibility(llama_program, compatible_llama_model, &compatibility));
    try std.testing.expectEqual(tiny_llama_kind, compatibility.program_model_kind);
    try std.testing.expectEqual(tiny_llama_kind, compatibility.model_kind);
    try std.testing.expectEqual(@as(u64, 1), compatibility.compatible);

    compatibility = .{};
    try std.testing.expectEqual(status(.ok), zgml_program_check_model_compatibility(llama_program, linear_model, &compatibility));
    try std.testing.expectEqual(tiny_llama_kind, compatibility.program_model_kind);
    try std.testing.expectEqual(tiny_linear_kind, compatibility.model_kind);
    try std.testing.expectEqual(@as(u64, 0), compatibility.compatible);
}

test "C ABI tiny linear handles compile bind step free through graph-built DeviceInference" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    var weights_buffer: ?*zgml_buffer = null;
    var bias_buffer: ?*zgml_buffer = null;
    var input_buffer: ?*zgml_buffer = null;
    var output_buffer: ?*zgml_buffer = null;
    var output_shorthand_buffer: ?*zgml_buffer = null;
    defer zgml_buffer_free(output_shorthand_buffer);
    defer zgml_buffer_free(output_buffer);
    defer zgml_buffer_free(input_buffer);
    defer zgml_buffer_free(bias_buffer);
    defer zgml_buffer_free(weights_buffer);
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 3,
    }, &model));
    try std.testing.expect(model != null);
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_cpu,
        .context_len = 4,
        .batch = 1,
    }, &program));
    try std.testing.expect(program != null);

    var inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &inspection));
    try std.testing.expect(inspection.command_count > 0);
    try std.testing.expectEqual(@as(u64, backend_cpu), inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 0), inspection.external_resources_supported);
    try std.testing.expectEqual(@as(u64, 5), inspection.buffer_count);
    try std.testing.expectEqual(@as(u64, 17), inspection.buffer_element_count);
    try std.testing.expectEqual(@as(u64, 68), inspection.buffer_byte_len);
    try std.testing.expectEqual(@as(u64, 1), inspection.initial_upload_count);
    try std.testing.expectEqual(@as(u64, 0), inspection.qweight_count);
    try std.testing.expect(inspection.op_count > 0);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u32)), inspection.runtime_patch_max_cache_write_pos);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u32)), inspection.runtime_patch_max_attention_seq_kv);
    try std.testing.expect(commandCategoryTotal(inspection) > 0);
    try std.testing.expect(inspection.command_stencil_hash != 0);
    try std.testing.expect(inspection.binding_requirement_hash != 0);
    try std.testing.expectEqual(@as(u64, 2), inspection.persistent_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_input_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_output_requirement_count);
    try std.testing.expectEqual(@as(u64, 0), inspection.runtime_patch_holes);

    var requirements = zgml_program_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(program, &requirements));
    try std.testing.expectEqual(tiny_linear_kind, requirements.model_kind);
    try std.testing.expectEqual(@as(u32, @sizeOf(f32)), requirements.scalar_bytes);
    try std.testing.expectEqual(@as(u32, @sizeOf(u32)), requirements.token_id_bytes);
    try std.testing.expectEqual(@as(usize, 2), requirements.input_len);
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(f32)), requirements.input_byte_len);
    try std.testing.expectEqual(@as(usize, 3), requirements.output_len);
    try std.testing.expectEqual(@as(usize, 6), requirements.weights_len);
    try std.testing.expectEqual(@as(usize, 6 * @sizeOf(f32)), requirements.weights_byte_len);
    try std.testing.expectEqual(@as(usize, 3), requirements.bias_len);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), requirements.bias_byte_len);
    try std.testing.expectEqual(@as(usize, 9), requirements.parameter_len);
    try std.testing.expectEqual(@as(usize, 9 * @sizeOf(f32)), requirements.parameter_byte_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.logits_len);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), requirements.output_byte_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.max_token_window);

    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_weights, &weights_buffer));
    try std.testing.expectEqual(@as(usize, 6 * @sizeOf(f32)), zgml_buffer_size(weights_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_bias, &bias_buffer));
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), zgml_buffer_size(bias_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_input, &input_buffer));
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(f32)), zgml_buffer_size(input_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_output, &output_buffer));
    try std.testing.expectEqual(requirements.output_byte_len, zgml_buffer_size(output_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_output_buffer(program, &output_shorthand_buffer));
    try std.testing.expectEqual(requirements.output_byte_len, zgml_buffer_size(output_shorthand_buffer));

    var llama_inspection = zgml_llama_program_inspection{ .vocab_size = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_llama_program_inspect(program, &llama_inspection));
    try std.testing.expectEqual(@as(u64, 0), llama_inspection.vocab_size);

    var kv_requirements = zgml_llama_kv_cache_requirements{ .context_len = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_llama_program_get_kv_cache_requirements(program, &kv_requirements));
    try std.testing.expectEqual(@as(usize, 0), kv_requirements.context_len);

    const weights = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bias = [_]f32{ 0.5, -0.5, 1.0 };
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = weights[0..].ptr,
        .weights_len = weights.len,
        .bias = bias[0..].ptr,
        .bias_len = bias.len,
    }, &session));
    try std.testing.expect(session != null);

    const input = [_]f32{ 1, 2 };
    var output = [_]f32{0} ** 3;
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
        .input = input[0..].ptr,
        .input_len = input.len,
        .output = output[0..].ptr,
        .output_len = output.len,
    }, &result));

    try std.testing.expectEqual(@as(usize, 3), result.output_len);
    try std.testing.expectEqualSlices(f32, &.{ 9.5, 11.5, 16.0 }, &output);

    var program_profile = zgml_runtime_profile{};
    try std.testing.expectEqual(status(.ok), zgml_program_runtime_profile(program, &program_profile));
    try std.testing.expectEqual(@as(u64, 0), program_profile.call_count);
    try std.testing.expectEqual(@as(u64, 0), program_profile.runtime_patch_call_count);
    try std.testing.expect(program_profile.command_count > 0);

    var runtime_profile = zgml_runtime_profile{};
    try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(session, &runtime_profile));
    try std.testing.expectEqual(@as(u64, 1), runtime_profile.call_count);
    try std.testing.expectEqual(@as(u64, 1), runtime_profile.runtime_patch_call_count);
    try std.testing.expect(runtime_profile.command_count > 0);

    try std.testing.expectEqual(status(.ok), zgml_session_reset_runtime_profile(session));
    runtime_profile = .{};
    try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(session, &runtime_profile));
    try std.testing.expectEqual(@as(u64, 0), runtime_profile.call_count);
    try std.testing.expectEqual(@as(u64, 0), runtime_profile.runtime_patch_call_count);
    try std.testing.expect(runtime_profile.command_count > 0);
}

test "C ABI tiny MLP handles compile bind step through native module graph" {
    var invalid_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_create(&.{
        .kind = tiny_mlp_kind,
        .activation = 99,
        .input_len = 2,
        .hidden_len = 3,
        .output_len = 2,
    }, &invalid_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), invalid_model);

    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_mlp_kind,
        .activation = @intFromEnum(TinyMlpActivation.relu),
        .input_len = 2,
        .hidden_len = 3,
        .output_len = 2,
    }, &model));
    try std.testing.expect(model != null);

    var model_inspection = zgml_model_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_model_inspect(model, &model_inspection));
    try std.testing.expectEqual(tiny_mlp_kind, model_inspection.model_kind);
    try std.testing.expectEqual(@as(u64, 2), model_inspection.input_len);
    try std.testing.expectEqual(@as(u64, 2), model_inspection.output_len);
    try std.testing.expectEqual(@as(u64, 3), model_inspection.d_model);
    try std.testing.expectEqual(@as(u64, 2), model_inspection.n_layers);

    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    try std.testing.expect(program != null);

    var program_inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &program_inspection));
    try std.testing.expectEqual(@as(u64, backend_cpu), program_inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), program_inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 0), program_inspection.external_resources_supported);
    try std.testing.expect(program_inspection.command_count > 0);
    try std.testing.expect(program_inspection.op_count > 0);
    try std.testing.expect(program_inspection.command_stencil_hash != 0);
    try std.testing.expect(program_inspection.binding_requirement_hash != 0);
    try std.testing.expectEqual(@as(u64, 4), program_inspection.persistent_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), program_inspection.step_input_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), program_inspection.step_output_requirement_count);

    var requirements = zgml_program_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(program, &requirements));
    try std.testing.expectEqual(tiny_mlp_kind, requirements.model_kind);
    try std.testing.expectEqual(@as(usize, 2), requirements.input_len);
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(f32)), requirements.input_byte_len);
    try std.testing.expectEqual(@as(usize, 2), requirements.output_len);
    try std.testing.expectEqual(@as(usize, 12), requirements.weights_len);
    try std.testing.expectEqual(@as(usize, 12 * @sizeOf(f32)), requirements.weights_byte_len);
    try std.testing.expectEqual(@as(usize, 5), requirements.bias_len);
    try std.testing.expectEqual(@as(usize, 5 * @sizeOf(f32)), requirements.bias_byte_len);
    try std.testing.expectEqual(@as(usize, 17), requirements.parameter_len);
    try std.testing.expectEqual(@as(usize, 17 * @sizeOf(f32)), requirements.parameter_byte_len);
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(f32)), requirements.output_byte_len);

    const weights = [_]f32{
        1, 2,    3,
        4, 5,    6,
        1, -1,   0.5,
        2, -0.5, 1,
    };
    const bias = [_]f32{ 0.5, -0.5, 1.0, 0.25, -0.75 };
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = weights[0..].ptr,
        .weights_len = weights.len,
        .bias = bias[0..].ptr,
        .bias_len = bias.len,
    }, &session));
    try std.testing.expect(session != null);

    var session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(tiny_mlp_kind, session_inspection.model_kind);
    try std.testing.expectEqual(backend_cpu, session_inspection.backend);
    try std.testing.expectEqual(@as(u64, 4), session_inspection.persistent_binding_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_input_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);

    const input = [_]f32{ 1, 2 };
    var output = [_]f32{0} ** 2;
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
        .input = input[0..].ptr,
        .input_len = input.len,
        .output = output[0..].ptr,
        .output_len = output.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    try std.testing.expectEqualSlices(f32, &.{ 7.5, 28.75 }, &output);

    var runtime_profile = zgml_runtime_profile{};
    try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(session, &runtime_profile));
    try std.testing.expectEqual(@as(u64, 1), runtime_profile.call_count);
    try std.testing.expectEqual(@as(u64, 1), runtime_profile.runtime_patch_call_count);
    try std.testing.expect(runtime_profile.command_count > 0);
}

test "C ABI module program compiles traced sequential ops" {
    {
        const noop_input_shape = [_]usize{4};
        var noop_program: ?*zgml_program = null;
        var noop_session: ?*zgml_session = null;
        defer zgml_session_free(noop_session);
        defer zgml_program_free(noop_program);

        try std.testing.expectEqual(status(.invalid_argument), zgml_module_program_compile(&.{
            .input_shape = noop_input_shape[0..].ptr,
            .input_rank = noop_input_shape.len,
            .ops = null,
            .op_count = 1,
        }, &.{ .backend = backend_cpu }, &noop_program));
        try std.testing.expectEqual(@as(?*zgml_program, null), noop_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = noop_input_shape[0..].ptr,
            .input_rank = noop_input_shape.len,
            .ops = null,
            .op_count = 0,
        }, &.{ .backend = backend_cpu }, &noop_program));
        try std.testing.expect(noop_program != null);

        var noop_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(noop_program, &noop_requirements));
        try std.testing.expectEqual(module_kind, noop_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), noop_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), noop_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), noop_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), noop_requirements.bias_len);

        var noop_inspection = zgml_program_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_program_inspect(noop_program, &noop_inspection));
        try std.testing.expectEqual(@as(u64, backend_cpu), noop_inspection.backend);
        try std.testing.expectEqual(@as(u64, 1), noop_inspection.execution_supported);
        try std.testing.expectEqual(@as(u64, 0), noop_inspection.command_count);
        try std.testing.expectEqual(@as(u64, 0), noop_inspection.command_stencil_hash);
        try std.testing.expectEqual(@as(u64, 0), noop_inspection.persistent_requirement_count);
        try std.testing.expectEqual(@as(u64, 1), noop_inspection.step_input_requirement_count);
        try std.testing.expectEqual(@as(u64, 1), noop_inspection.step_output_requirement_count);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(noop_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &noop_session));
        try std.testing.expect(noop_session != null);

        const noop_input = [_]f32{ 8, 6, 4, 2 };
        var noop_output = [_]f32{0} ** 4;
        var noop_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(noop_session, &.{
            .input = noop_input[0..].ptr,
            .input_len = noop_input.len,
            .output = noop_output[0..].ptr,
            .output_len = noop_output.len,
        }, &noop_result));
        try std.testing.expectEqual(@as(usize, noop_output.len), noop_result.output_len);
        try std.testing.expectEqualSlices(f32, noop_input[0..], noop_output[0..]);

        var noop_runtime_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(noop_session, &noop_runtime_profile));
        try std.testing.expectEqual(@as(u64, 1), noop_runtime_profile.call_count);
        try std.testing.expectEqual(@as(u64, 1), noop_runtime_profile.runtime_patch_call_count);
        try std.testing.expectEqual(@as(u64, 0), noop_runtime_profile.command_count);
        try std.testing.expectEqual(@as(u64, 0), noop_runtime_profile.backend_op_count);
        try std.testing.expectEqual(@as(u64, 0), noop_runtime_profile.backend_dispatch_count);
    }

    const input_shape = [_]usize{2};
    {
        const chain_input_shape = [_]usize{3};
        const invalid_chain_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_activation_chain,
            .activation = module_activation_relu,
            .flags = 0,
        }};
        const chain_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_activation_chain,
            .activation = module_activation_relu,
            .flags = 3,
            .a = @intCast(module_activation_square),
            .b = @intCast(module_activation_sqrt),
        }};
        var chain_program: ?*zgml_program = null;
        var chain_session: ?*zgml_session = null;
        defer zgml_session_free(chain_session);
        defer zgml_program_free(chain_program);

        try std.testing.expectEqual(status(.invalid_argument), zgml_module_program_compile(&.{
            .input_shape = chain_input_shape[0..].ptr,
            .input_rank = chain_input_shape.len,
            .ops = invalid_chain_ops[0..].ptr,
            .op_count = invalid_chain_ops.len,
        }, &.{ .backend = backend_cpu }, &chain_program));
        try std.testing.expectEqual(@as(?*zgml_program, null), chain_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = chain_input_shape[0..].ptr,
            .input_rank = chain_input_shape.len,
            .ops = chain_ops[0..].ptr,
            .op_count = chain_ops.len,
        }, &.{ .backend = backend_cpu }, &chain_program));
        try std.testing.expect(chain_program != null);

        var chain_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(chain_program, &chain_requirements));
        try std.testing.expectEqual(module_kind, chain_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 3), chain_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 3), chain_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), chain_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), chain_requirements.bias_len);

        var chain_inspection = zgml_program_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_program_inspect(chain_program, &chain_inspection));
        try std.testing.expectEqual(@as(u64, 1), chain_inspection.op_count);
        try std.testing.expectEqual(@as(u64, 1), chain_inspection.command_count);
        try std.testing.expectEqual(@as(u64, 1), chain_inspection.command_op_count);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(chain_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &chain_session));
        try std.testing.expect(chain_session != null);

        const chain_input = [_]f32{ -2, 3, 0.5 };
        var chain_output = [_]f32{0} ** 3;
        var chain_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(chain_session, &.{
            .input = chain_input[0..].ptr,
            .input_len = chain_input.len,
            .output = chain_output[0..].ptr,
            .output_len = chain_output.len,
        }, &chain_result));
        try std.testing.expectEqual(@as(usize, chain_output.len), chain_result.output_len);
        try std.testing.expectApproxEqAbs(@as(f32, 0), chain_output[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 3), chain_output[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), chain_output[2], 1e-6);
    }

    {
        const mlp_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_linear,
                .activation = module_activation_relu,
                .flags = module_flag_bias,
                .a = 2,
                .b = 3,
            },
            .{
                .kind = module_op_linear,
                .flags = module_flag_bias,
                .a = 3,
                .b = 2,
            },
        };
        var mlp_program: ?*zgml_program = null;
        var mlp_session: ?*zgml_session = null;
        defer zgml_session_free(mlp_session);
        defer zgml_program_free(mlp_program);

        var probed_mlp_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_module_program_get_requirements(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = mlp_ops[0..].ptr,
            .op_count = mlp_ops.len,
        }, &.{ .backend = backend_cpu }, &probed_mlp_requirements));
        try std.testing.expectEqual(module_kind, probed_mlp_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 2), probed_mlp_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), probed_mlp_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 12), probed_mlp_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 5), probed_mlp_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = mlp_ops[0..].ptr,
            .op_count = mlp_ops.len,
        }, &.{ .backend = backend_cpu }, &mlp_program));
        try std.testing.expect(mlp_program != null);

        var mlp_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(mlp_program, &mlp_requirements));
        try std.testing.expectEqual(module_kind, mlp_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 2), mlp_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), mlp_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 12), mlp_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 5), mlp_requirements.bias_len);
        try std.testing.expectEqual(probed_mlp_requirements.input_byte_len, mlp_requirements.input_byte_len);
        try std.testing.expectEqual(probed_mlp_requirements.output_byte_len, mlp_requirements.output_byte_len);
        try std.testing.expectEqual(probed_mlp_requirements.weights_byte_len, mlp_requirements.weights_byte_len);
        try std.testing.expectEqual(probed_mlp_requirements.bias_byte_len, mlp_requirements.bias_byte_len);
        try std.testing.expectEqual(probed_mlp_requirements.parameter_byte_len, mlp_requirements.parameter_byte_len);

        var mlp_inspection = zgml_program_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_program_inspect(mlp_program, &mlp_inspection));
        try std.testing.expectEqual(@as(u64, backend_cpu), mlp_inspection.backend);
        try std.testing.expectEqual(@as(u64, 1), mlp_inspection.execution_supported);
        try std.testing.expectEqual(@as(u64, 4), mlp_inspection.op_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_inspection.command_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_inspection.command_projection_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_inspection.command_op_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_inspection.command_elementwise_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_inspection.command_movement_count);
        try std.testing.expect(mlp_inspection.command_stencil_hash != 0);

        var mlp_program_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_program_runtime_profile(mlp_program, &mlp_program_profile));
        try std.testing.expectEqual(@as(u64, 0), mlp_program_profile.call_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_program_profile.backend_op_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_program_profile.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_program_profile.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_program_profile.command_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_program_profile.command_projection_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_program_profile.command_op_count);

        const mlp_weights = [_]f32{
            1, 2, 3, 4, 5, 6,
            1, 0, 0, 1, 0, 0,
        };
        const mlp_bias = [_]f32{ 0.5, -0.5, 1.0, 0.0, 0.0 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(mlp_program, &.{
            .weights = mlp_weights[0..].ptr,
            .weights_len = mlp_weights.len,
            .bias = mlp_bias[0..].ptr,
            .bias_len = mlp_bias.len,
        }, &mlp_session));
        try std.testing.expect(mlp_session != null);

        const mlp_input = [_]f32{ 1, 2 };
        var mlp_output = [_]f32{0} ** 2;
        var mlp_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(mlp_session, &.{
            .input = mlp_input[0..].ptr,
            .input_len = mlp_input.len,
            .output = mlp_output[0..].ptr,
            .output_len = mlp_output.len,
        }, &mlp_result));
        try std.testing.expectEqual(@as(usize, mlp_output.len), mlp_result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 9.5, 11.5 }, &mlp_output);

        var mlp_runtime_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(mlp_session, &mlp_runtime_profile));
        try std.testing.expectEqual(@as(u64, 1), mlp_runtime_profile.call_count);
        try std.testing.expectEqual(@as(u64, 1), mlp_runtime_profile.runtime_patch_call_count);
        try std.testing.expectEqual(@as(u64, 4), mlp_runtime_profile.backend_op_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_runtime_profile.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_runtime_profile.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_runtime_profile.sync_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_runtime_profile.command_count);
        try std.testing.expectEqual(@as(u64, 2), mlp_runtime_profile.command_projection_count);
        try std.testing.expectEqual(@as(u64, 0), mlp_runtime_profile.command_op_count);
    }

    {
        const add_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_add,
            .flags = module_flag_bias,
            .a = 2,
        }};
        var add_program: ?*zgml_program = null;
        var add_session: ?*zgml_session = null;
        defer zgml_session_free(add_session);
        defer zgml_program_free(add_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = add_ops[0..].ptr,
            .op_count = add_ops.len,
        }, &.{ .backend = backend_cpu }, &add_program));
        try std.testing.expect(add_program != null);

        var add_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(add_program, &add_requirements));
        try std.testing.expectEqual(module_kind, add_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 2), add_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), add_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), add_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 2), add_requirements.bias_len);

        const add_bias = [_]f32{ 10, -5 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(add_program, &.{
            .weights = null,
            .weights_len = 0,
            .bias = add_bias[0..].ptr,
            .bias_len = add_bias.len,
        }, &add_session));
        try std.testing.expect(add_session != null);

        const add_input = [_]f32{ 1.5, 7 };
        var add_output = [_]f32{0} ** 2;
        var add_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(add_session, &.{
            .input = add_input[0..].ptr,
            .input_len = add_input.len,
            .output = add_output[0..].ptr,
            .output_len = add_output.len,
        }, &add_result));
        try std.testing.expectEqual(@as(usize, add_output.len), add_result.output_len);
        try std.testing.expectApproxEqAbs(@as(f32, 11.5), add_output[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 2), add_output[1], 1e-6);
    }

    {
        const mul_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_mul,
            .flags = module_flag_weight,
            .a = 2,
        }};
        var mul_program: ?*zgml_program = null;
        var mul_session: ?*zgml_session = null;
        defer zgml_session_free(mul_session);
        defer zgml_program_free(mul_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = mul_ops[0..].ptr,
            .op_count = mul_ops.len,
        }, &.{ .backend = backend_cpu }, &mul_program));
        try std.testing.expect(mul_program != null);

        var mul_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(mul_program, &mul_requirements));
        try std.testing.expectEqual(module_kind, mul_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 2), mul_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), mul_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 2), mul_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), mul_requirements.bias_len);

        const mul_weights = [_]f32{ 3, -2 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(mul_program, &.{
            .weights = mul_weights[0..].ptr,
            .weights_len = mul_weights.len,
            .bias = null,
            .bias_len = 0,
        }, &mul_session));
        try std.testing.expect(mul_session != null);

        const mul_input = [_]f32{ 1.5, 7 };
        var mul_output = [_]f32{0} ** 2;
        var mul_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(mul_session, &.{
            .input = mul_input[0..].ptr,
            .input_len = mul_input.len,
            .output = mul_output[0..].ptr,
            .output_len = mul_output.len,
        }, &mul_result));
        try std.testing.expectEqual(@as(usize, mul_output.len), mul_result.output_len);
        try std.testing.expectApproxEqAbs(@as(f32, 4.5), mul_output[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, -14), mul_output[1], 1e-6);
    }

    const ops = [_]zgml_module_op_desc{
        .{
            .kind = module_op_linear,
            .flags = 0,
            .a = 2,
            .b = 2,
        },
        .{
            .kind = module_op_softmax,
            .a = 0,
        },
        .{
            .kind = module_op_activation,
            .activation = @intFromEnum(TinyMlpActivation.relu),
        },
    };
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);

    try std.testing.expectEqual(status(.invalid_argument), zgml_module_program_compile(null, &.{ .backend = backend_cpu }, &program));
    try std.testing.expectEqual(@as(?*zgml_program, null), program);

    try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
        .input_shape = input_shape[0..].ptr,
        .input_rank = input_shape.len,
        .ops = ops[0..].ptr,
        .op_count = ops.len,
    }, &.{ .backend = backend_cpu }, &program));
    try std.testing.expect(program != null);

    var requirements = zgml_program_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(program, &requirements));
    try std.testing.expectEqual(module_kind, requirements.model_kind);
    try std.testing.expectEqual(@as(usize, 2), requirements.input_len);
    try std.testing.expectEqual(@as(usize, 2), requirements.output_len);
    try std.testing.expectEqual(@as(usize, 4), requirements.weights_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.bias_len);

    var inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &inspection));
    try std.testing.expectEqual(@as(u64, backend_cpu), inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), inspection.execution_supported);
    try std.testing.expect(inspection.command_count > 0);
    try std.testing.expect(inspection.command_stencil_hash != 0);
    try std.testing.expect(inspection.binding_requirement_hash != 0);
    try std.testing.expectEqual(@as(u64, 1), inspection.persistent_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_input_requirement_count);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_output_requirement_count);

    const weights = [_]f32{ 1, 0, 0, 1 };
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = weights[0..].ptr,
        .weights_len = weights.len,
    }, &session));
    try std.testing.expect(session != null);

    var session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(module_kind, session_inspection.model_kind);
    try std.testing.expectEqual(backend_cpu, session_inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.persistent_binding_count);

    const input = [_]f32{ 1, 2 };
    var output = [_]f32{0} ** 2;
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
        .input = input[0..].ptr,
        .input_len = input.len,
        .output = output[0..].ptr,
        .output_len = output.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    const denom = @exp(@as(f32, 1)) + @exp(@as(f32, 2));
    try std.testing.expectApproxEqAbs(@exp(@as(f32, 1)) / denom, output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@exp(@as(f32, 2)) / denom, output[1], 1e-6);

    zgml_session_free(session);
    session = null;

    var weights_buffer: ?*zgml_buffer = null;
    var input_buffer: ?*zgml_buffer = null;
    var output_buffer: ?*zgml_buffer = null;
    defer zgml_buffer_free(output_buffer);
    defer zgml_buffer_free(input_buffer);
    defer zgml_buffer_free(weights_buffer);
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_weights, &weights_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_input, &input_buffer));
    try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(program, program_buffer_output, &output_buffer));
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(weights_buffer, 0, weights[0..].ptr, weights.len * @sizeOf(f32)));
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(input_buffer, 0, input[0..].ptr, input.len * @sizeOf(f32)));
    try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(program, &.{
        .weights = weights_buffer,
        .weights_len = weights.len,
        .input = input_buffer,
        .input_len = input.len,
        .output = output_buffer,
        .output_len = output.len,
    }, &session));
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(module_kind, session_inspection.model_kind);
    try std.testing.expectEqual(buffer_storage_host, session_inspection.output_storage);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.persistent_binding_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_input_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);
    try std.testing.expectEqual(@as(u64, 3), session_inspection.host_binding_count);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.resource_binding_count);
    try std.testing.expect(session_inspection.binding_shape_hash != 0);
    result = .{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    var native_output = [_]f32{0} ** 2;
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(output_buffer, 0, native_output[0..].ptr, native_output.len * @sizeOf(f32)));
    try std.testing.expectApproxEqAbs(output[0], native_output[0], 1e-6);
    try std.testing.expectApproxEqAbs(output[1], native_output[1], 1e-6);

    {
        const batched_input_shape = [_]usize{ 2, 2 };
        var batched_program: ?*zgml_program = null;
        var batched_session: ?*zgml_session = null;
        defer zgml_session_free(batched_session);
        defer zgml_program_free(batched_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = batched_input_shape[0..].ptr,
            .input_rank = batched_input_shape.len,
            .ops = ops[0..].ptr,
            .op_count = ops.len,
        }, &.{ .backend = backend_cpu }, &batched_program));
        try std.testing.expect(batched_program != null);

        var batched_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(batched_program, &batched_requirements));
        try std.testing.expectEqual(module_kind, batched_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), batched_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), batched_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 4), batched_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), batched_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(batched_program, &.{
            .weights = weights[0..].ptr,
            .weights_len = weights.len,
        }, &batched_session));
        try std.testing.expect(batched_session != null);

        const batched_input = [_]f32{ 1, 2, 3, 1 };
        var batched_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(batched_session, &.{
            .input = batched_input[0..].ptr,
            .input_len = batched_input.len,
            .output = batched_output[0..].ptr,
            .output_len = batched_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        const denom0 = @exp(@as(f32, 1)) + @exp(@as(f32, 2));
        const denom1 = @exp(@as(f32, 3)) + @exp(@as(f32, 1));
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 1)) / denom0, batched_output[0], 1e-6);
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 2)) / denom0, batched_output[1], 1e-6);
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 3)) / denom1, batched_output[2], 1e-6);
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 1)) / denom1, batched_output[3], 1e-6);
    }

    {
        const flatten_input_shape = [_]usize{ 2, 2 };
        const flatten_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_reshape,
                .a = 1,
                .b = 4,
            },
            .{
                .kind = module_op_activation,
                .activation = @intFromEnum(TinyMlpActivation.relu),
            },
        };
        var flatten_program: ?*zgml_program = null;
        var flatten_session: ?*zgml_session = null;
        defer zgml_session_free(flatten_session);
        defer zgml_program_free(flatten_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = flatten_input_shape[0..].ptr,
            .input_rank = flatten_input_shape.len,
            .ops = flatten_ops[0..].ptr,
            .op_count = flatten_ops.len,
        }, &.{ .backend = backend_cpu }, &flatten_program));
        try std.testing.expect(flatten_program != null);

        var flatten_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(flatten_program, &flatten_requirements));
        try std.testing.expectEqual(module_kind, flatten_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), flatten_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), flatten_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), flatten_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), flatten_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(flatten_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &flatten_session));
        try std.testing.expect(flatten_session != null);

        const flatten_input = [_]f32{ -1, 2, -3, 4 };
        var flatten_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(flatten_session, &.{
            .input = flatten_input[0..].ptr,
            .input_len = flatten_input.len,
            .output = flatten_output[0..].ptr,
            .output_len = flatten_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 4 }, &flatten_output);
    }

    {
        const transpose_input_shape = [_]usize{ 2, 3 };
        const transpose_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_transpose,
            .a = 0,
            .b = 1,
        }};
        var transpose_program: ?*zgml_program = null;
        var transpose_session: ?*zgml_session = null;
        defer zgml_session_free(transpose_session);
        defer zgml_program_free(transpose_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = transpose_input_shape[0..].ptr,
            .input_rank = transpose_input_shape.len,
            .ops = transpose_ops[0..].ptr,
            .op_count = transpose_ops.len,
        }, &.{ .backend = backend_cpu }, &transpose_program));
        try std.testing.expect(transpose_program != null);

        var transpose_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(transpose_program, &transpose_requirements));
        try std.testing.expectEqual(module_kind, transpose_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), transpose_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 6), transpose_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), transpose_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), transpose_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(transpose_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &transpose_session));
        try std.testing.expect(transpose_session != null);

        const transpose_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var transpose_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(transpose_session, &.{
            .input = transpose_input[0..].ptr,
            .input_len = transpose_input.len,
            .output = transpose_output[0..].ptr,
            .output_len = transpose_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &transpose_output);
    }

    {
        const transpose3_input_shape = [_]usize{ 1, 2, 3 };
        const transpose3_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_transpose,
            .a = 1,
            .b = 2,
        }};
        var transpose3_program: ?*zgml_program = null;
        var transpose3_session: ?*zgml_session = null;
        defer zgml_session_free(transpose3_session);
        defer zgml_program_free(transpose3_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = transpose3_input_shape[0..].ptr,
            .input_rank = transpose3_input_shape.len,
            .ops = transpose3_ops[0..].ptr,
            .op_count = transpose3_ops.len,
        }, &.{ .backend = backend_cpu }, &transpose3_program));
        try std.testing.expect(transpose3_program != null);

        var transpose3_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(transpose3_program, &transpose3_requirements));
        try std.testing.expectEqual(module_kind, transpose3_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), transpose3_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 6), transpose3_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(transpose3_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &transpose3_session));
        try std.testing.expect(transpose3_session != null);

        const transpose3_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var transpose3_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(transpose3_session, &.{
            .input = transpose3_input[0..].ptr,
            .input_len = transpose3_input.len,
            .output = transpose3_output[0..].ptr,
            .output_len = transpose3_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &transpose3_output);
    }

    {
        const batch_softmax_input_shape = [_]usize{ 2, 2 };
        const batch_softmax_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_transpose,
                .a = 0,
                .b = 1,
            },
            .{
                .kind = module_op_softmax,
                .a = 0,
            },
            .{
                .kind = module_op_transpose,
                .a = 0,
                .b = 1,
            },
        };
        var batch_softmax_program: ?*zgml_program = null;
        var batch_softmax_session: ?*zgml_session = null;
        defer zgml_session_free(batch_softmax_session);
        defer zgml_program_free(batch_softmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = batch_softmax_input_shape[0..].ptr,
            .input_rank = batch_softmax_input_shape.len,
            .ops = batch_softmax_ops[0..].ptr,
            .op_count = batch_softmax_ops.len,
        }, &.{ .backend = backend_cpu }, &batch_softmax_program));
        try std.testing.expect(batch_softmax_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(batch_softmax_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &batch_softmax_session));
        try std.testing.expect(batch_softmax_session != null);

        const batch_softmax_input = [_]f32{ 1, 2, 3, 1 };
        var batch_softmax_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(batch_softmax_session, &.{
            .input = batch_softmax_input[0..].ptr,
            .input_len = batch_softmax_input.len,
            .output = batch_softmax_output[0..].ptr,
            .output_len = batch_softmax_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        const col0_den = @exp(@as(f32, 1)) + @exp(@as(f32, 3));
        const col1_den = @exp(@as(f32, 2)) + @exp(@as(f32, 1));
        const batch_softmax_expected = [_]f32{
            @exp(@as(f32, 1)) / col0_den,
            @exp(@as(f32, 2)) / col1_den,
            @exp(@as(f32, 3)) / col0_den,
            @exp(@as(f32, 1)) / col1_den,
        };
        for (batch_softmax_output, batch_softmax_expected) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-6);
        }
    }

    {
        const direct_batch_softmax_input_shape = [_]usize{ 2, 2 };
        const direct_batch_softmax_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_softmax,
            .a = 1,
        }};
        var direct_batch_softmax_program: ?*zgml_program = null;
        var direct_batch_softmax_session: ?*zgml_session = null;
        defer zgml_session_free(direct_batch_softmax_session);
        defer zgml_program_free(direct_batch_softmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = direct_batch_softmax_input_shape[0..].ptr,
            .input_rank = direct_batch_softmax_input_shape.len,
            .ops = direct_batch_softmax_ops[0..].ptr,
            .op_count = direct_batch_softmax_ops.len,
        }, &.{ .backend = backend_cpu }, &direct_batch_softmax_program));
        try std.testing.expect(direct_batch_softmax_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(direct_batch_softmax_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &direct_batch_softmax_session));
        try std.testing.expect(direct_batch_softmax_session != null);

        const direct_batch_softmax_input = [_]f32{ 1, 2, 3, 1 };
        var direct_batch_softmax_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(direct_batch_softmax_session, &.{
            .input = direct_batch_softmax_input[0..].ptr,
            .input_len = direct_batch_softmax_input.len,
            .output = direct_batch_softmax_output[0..].ptr,
            .output_len = direct_batch_softmax_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        const col0_den = @exp(@as(f32, 1)) + @exp(@as(f32, 3));
        const col1_den = @exp(@as(f32, 2)) + @exp(@as(f32, 1));
        const direct_batch_softmax_expected = [_]f32{
            @exp(@as(f32, 1)) / col0_den,
            @exp(@as(f32, 2)) / col1_den,
            @exp(@as(f32, 3)) / col0_den,
            @exp(@as(f32, 1)) / col1_den,
        };
        for (direct_batch_softmax_output, direct_batch_softmax_expected) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-6);
        }
    }

    {
        const direct_batch_logsoftmax_input_shape = [_]usize{ 2, 2 };
        const direct_batch_logsoftmax_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_log_softmax,
            .a = 1,
        }};
        var direct_batch_logsoftmax_program: ?*zgml_program = null;
        var direct_batch_logsoftmax_session: ?*zgml_session = null;
        defer zgml_session_free(direct_batch_logsoftmax_session);
        defer zgml_program_free(direct_batch_logsoftmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = direct_batch_logsoftmax_input_shape[0..].ptr,
            .input_rank = direct_batch_logsoftmax_input_shape.len,
            .ops = direct_batch_logsoftmax_ops[0..].ptr,
            .op_count = direct_batch_logsoftmax_ops.len,
        }, &.{ .backend = backend_cpu }, &direct_batch_logsoftmax_program));
        try std.testing.expect(direct_batch_logsoftmax_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(direct_batch_logsoftmax_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &direct_batch_logsoftmax_session));
        try std.testing.expect(direct_batch_logsoftmax_session != null);

        const direct_batch_logsoftmax_input = [_]f32{ 1, 2, 3, 1 };
        var direct_batch_logsoftmax_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(direct_batch_logsoftmax_session, &.{
            .input = direct_batch_logsoftmax_input[0..].ptr,
            .input_len = direct_batch_logsoftmax_input.len,
            .output = direct_batch_logsoftmax_output[0..].ptr,
            .output_len = direct_batch_logsoftmax_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        const col0_log_den = @log(@exp(@as(f32, 1)) + @exp(@as(f32, 3)));
        const col1_log_den = @log(@exp(@as(f32, 2)) + @exp(@as(f32, 1)));
        const direct_batch_logsoftmax_expected = [_]f32{
            @as(f32, 1) - col0_log_den,
            @as(f32, 2) - col1_log_den,
            @as(f32, 3) - col0_log_den,
            @as(f32, 1) - col1_log_den,
        };
        for (direct_batch_logsoftmax_output, direct_batch_logsoftmax_expected) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-6);
        }
    }

    {
        const narrow_input_shape = [_]usize{4};
        const narrow_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_narrow,
            .a = 0,
            .b = 1,
            .c = 2,
        }};
        var narrow_program: ?*zgml_program = null;
        var narrow_session: ?*zgml_session = null;
        defer zgml_session_free(narrow_session);
        defer zgml_program_free(narrow_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = narrow_input_shape[0..].ptr,
            .input_rank = narrow_input_shape.len,
            .ops = narrow_ops[0..].ptr,
            .op_count = narrow_ops.len,
        }, &.{ .backend = backend_cpu }, &narrow_program));
        try std.testing.expect(narrow_program != null);

        var narrow_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(narrow_program, &narrow_requirements));
        try std.testing.expectEqual(module_kind, narrow_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), narrow_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), narrow_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), narrow_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), narrow_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(narrow_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &narrow_session));
        try std.testing.expect(narrow_session != null);

        const narrow_input = [_]f32{ 10, 20, 30, 40 };
        var narrow_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(narrow_session, &.{
            .input = narrow_input[0..].ptr,
            .input_len = narrow_input.len,
            .output = narrow_output[0..].ptr,
            .output_len = narrow_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 20, 30 }, &narrow_output);
    }

    {
        const batch_narrow_shape = [_]usize{ 3, 2 };
        const batch_narrow_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_narrow,
            .a = 0,
            .b = 1,
            .c = 2,
        }};
        var batch_narrow_program: ?*zgml_program = null;
        var batch_narrow_session: ?*zgml_session = null;
        defer zgml_session_free(batch_narrow_session);
        defer zgml_program_free(batch_narrow_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = batch_narrow_shape[0..].ptr,
            .input_rank = batch_narrow_shape.len,
            .ops = batch_narrow_ops[0..].ptr,
            .op_count = batch_narrow_ops.len,
        }, &.{ .backend = backend_cpu }, &batch_narrow_program));
        try std.testing.expect(batch_narrow_program != null);

        var batch_narrow_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(batch_narrow_program, &batch_narrow_requirements));
        try std.testing.expectEqual(module_kind, batch_narrow_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), batch_narrow_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), batch_narrow_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), batch_narrow_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), batch_narrow_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(batch_narrow_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &batch_narrow_session));
        try std.testing.expect(batch_narrow_session != null);

        const batch_narrow_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var batch_narrow_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(batch_narrow_session, &.{
            .input = batch_narrow_input[0..].ptr,
            .input_len = batch_narrow_input.len,
            .output = batch_narrow_output[0..].ptr,
            .output_len = batch_narrow_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5, 6 }, &batch_narrow_output);
    }

    {
        const reduce_input_shape = [_]usize{ 2, 3 };
        const reduce_sum_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_reduce_sum,
            .a = 0,
        }};
        var reduce_sum_program: ?*zgml_program = null;
        var reduce_sum_session: ?*zgml_session = null;
        defer zgml_session_free(reduce_sum_session);
        defer zgml_program_free(reduce_sum_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = reduce_sum_ops[0..].ptr,
            .op_count = reduce_sum_ops.len,
        }, &.{ .backend = backend_cpu }, &reduce_sum_program));
        try std.testing.expect(reduce_sum_program != null);

        var reduce_sum_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(reduce_sum_program, &reduce_sum_requirements));
        try std.testing.expectEqual(module_kind, reduce_sum_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), reduce_sum_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), reduce_sum_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), reduce_sum_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), reduce_sum_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(reduce_sum_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &reduce_sum_session));
        try std.testing.expect(reduce_sum_session != null);

        const reduce_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var reduce_sum_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(reduce_sum_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = reduce_sum_output[0..].ptr,
            .output_len = reduce_sum_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, &reduce_sum_output);

        const reduce3_input_shape = [_]usize{ 2, 2, 3 };
        var reduce3_sum_program: ?*zgml_program = null;
        var reduce3_sum_session: ?*zgml_session = null;
        defer zgml_session_free(reduce3_sum_session);
        defer zgml_program_free(reduce3_sum_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce3_input_shape[0..].ptr,
            .input_rank = reduce3_input_shape.len,
            .ops = reduce_sum_ops[0..].ptr,
            .op_count = reduce_sum_ops.len,
        }, &.{ .backend = backend_cpu }, &reduce3_sum_program));
        try std.testing.expect(reduce3_sum_program != null);

        var reduce3_sum_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(reduce3_sum_program, &reduce3_sum_requirements));
        try std.testing.expectEqual(module_kind, reduce3_sum_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 12), reduce3_sum_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), reduce3_sum_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(reduce3_sum_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &reduce3_sum_session));
        try std.testing.expect(reduce3_sum_session != null);

        const reduce3_input = [_]f32{ 1, 3, 2, 4, 0, -1, 10, 20, 30, 3, 2, 1 };
        var reduce3_sum_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(reduce3_sum_session, &.{
            .input = reduce3_input[0..].ptr,
            .input_len = reduce3_input.len,
            .output = reduce3_sum_output[0..].ptr,
            .output_len = reduce3_sum_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 6, 3, 60, 6 }, &reduce3_sum_output);

        const reduce_mean_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_reduce_mean,
            .a = 0,
        }};
        var reduce_mean_program: ?*zgml_program = null;
        var reduce_mean_session: ?*zgml_session = null;
        defer zgml_session_free(reduce_mean_session);
        defer zgml_program_free(reduce_mean_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = reduce_mean_ops[0..].ptr,
            .op_count = reduce_mean_ops.len,
        }, &.{ .backend = backend_cpu }, &reduce_mean_program));
        try std.testing.expect(reduce_mean_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(reduce_mean_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &reduce_mean_session));
        try std.testing.expect(reduce_mean_session != null);

        var reduce_mean_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(reduce_mean_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = reduce_mean_output[0..].ptr,
            .output_len = reduce_mean_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, &reduce_mean_output);

        const reduce_min_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_reduce_min,
            .a = 0,
        }};
        var reduce_min_program: ?*zgml_program = null;
        var reduce_min_session: ?*zgml_session = null;
        defer zgml_session_free(reduce_min_session);
        defer zgml_program_free(reduce_min_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = reduce_min_ops[0..].ptr,
            .op_count = reduce_min_ops.len,
        }, &.{ .backend = backend_cpu }, &reduce_min_program));
        try std.testing.expect(reduce_min_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(reduce_min_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &reduce_min_session));
        try std.testing.expect(reduce_min_session != null);

        var reduce_min_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(reduce_min_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = reduce_min_output[0..].ptr,
            .output_len = reduce_min_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 4 }, &reduce_min_output);

        const feature_affine_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_feature_affine,
            .flags = module_flag_weight | module_flag_bias,
            .a = 3,
        }};
        var feature_affine_program: ?*zgml_program = null;
        var feature_affine_session: ?*zgml_session = null;
        defer zgml_session_free(feature_affine_session);
        defer zgml_program_free(feature_affine_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = feature_affine_ops[0..].ptr,
            .op_count = feature_affine_ops.len,
        }, &.{ .backend = backend_cpu }, &feature_affine_program));
        try std.testing.expect(feature_affine_program != null);

        var feature_affine_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(feature_affine_program, &feature_affine_requirements));
        try std.testing.expectEqual(@as(usize, 3), feature_affine_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 3), feature_affine_requirements.bias_len);

        const feature_affine_weights = [_]f32{ 2, 3, 4 };
        const feature_affine_bias = [_]f32{ 0.5, -1, 1 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(feature_affine_program, &.{
            .weights = feature_affine_weights[0..].ptr,
            .weights_len = feature_affine_weights.len,
            .bias = feature_affine_bias[0..].ptr,
            .bias_len = feature_affine_bias.len,
        }, &feature_affine_session));
        try std.testing.expect(feature_affine_session != null);

        var feature_affine_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(feature_affine_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = feature_affine_output[0..].ptr,
            .output_len = feature_affine_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 2.5, 5, 13, 8.5, 14, 25 }, &feature_affine_output);

        const feature_affine_relu_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_feature_affine,
            .activation = module_activation_relu,
            .flags = module_flag_weight | module_flag_bias,
            .a = 3,
        }};
        var feature_affine_relu_program: ?*zgml_program = null;
        var feature_affine_relu_session: ?*zgml_session = null;
        defer zgml_session_free(feature_affine_relu_session);
        defer zgml_program_free(feature_affine_relu_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = feature_affine_relu_ops[0..].ptr,
            .op_count = feature_affine_relu_ops.len,
        }, &.{ .backend = backend_cpu }, &feature_affine_relu_program));
        try std.testing.expect(feature_affine_relu_program != null);

        const feature_affine_relu_weights = [_]f32{ 2, -1, 0.5 };
        const feature_affine_relu_bias = [_]f32{ 1, 10, -2 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(feature_affine_relu_program, &.{
            .weights = feature_affine_relu_weights[0..].ptr,
            .weights_len = feature_affine_relu_weights.len,
            .bias = feature_affine_relu_bias[0..].ptr,
            .bias_len = feature_affine_relu_bias.len,
        }, &feature_affine_relu_session));
        try std.testing.expect(feature_affine_relu_session != null);

        var feature_affine_relu_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(feature_affine_relu_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = feature_affine_relu_output[0..].ptr,
            .output_len = feature_affine_relu_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 3, 8, 0, 9, 5, 1 }, &feature_affine_relu_output);

        const add_relu_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_add,
            .activation = module_activation_relu,
            .flags = module_flag_bias,
            .a = 3,
        }};
        var add_relu_program: ?*zgml_program = null;
        var add_relu_session: ?*zgml_session = null;
        defer zgml_session_free(add_relu_session);
        defer zgml_program_free(add_relu_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = add_relu_ops[0..].ptr,
            .op_count = add_relu_ops.len,
        }, &.{ .backend = backend_cpu }, &add_relu_program));
        try std.testing.expect(add_relu_program != null);

        const add_relu_bias = [_]f32{ -2, 1, -5 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(add_relu_program, &.{
            .weights = null,
            .weights_len = 0,
            .bias = add_relu_bias[0..].ptr,
            .bias_len = add_relu_bias.len,
        }, &add_relu_session));
        try std.testing.expect(add_relu_session != null);

        var add_relu_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(add_relu_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = add_relu_output[0..].ptr,
            .output_len = add_relu_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 0, 3, 0, 2, 6, 1 }, &add_relu_output);

        const mul_relu_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_mul,
            .activation = module_activation_relu,
            .flags = module_flag_weight,
            .a = 3,
        }};
        var mul_relu_program: ?*zgml_program = null;
        var mul_relu_session: ?*zgml_session = null;
        defer zgml_session_free(mul_relu_session);
        defer zgml_program_free(mul_relu_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = mul_relu_ops[0..].ptr,
            .op_count = mul_relu_ops.len,
        }, &.{ .backend = backend_cpu }, &mul_relu_program));
        try std.testing.expect(mul_relu_program != null);

        const mul_relu_weights = [_]f32{ 2, -1, 0.5 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(mul_relu_program, &.{
            .weights = mul_relu_weights[0..].ptr,
            .weights_len = mul_relu_weights.len,
            .bias = null,
            .bias_len = 0,
        }, &mul_relu_session));
        try std.testing.expect(mul_relu_session != null);

        var mul_relu_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(mul_relu_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = mul_relu_output[0..].ptr,
            .output_len = mul_relu_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 2, 0, 1.5, 8, 0, 3 }, &mul_relu_output);

        const batch_reduce_max_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_transpose,
                .a = 0,
                .b = 1,
            },
            .{
                .kind = module_op_reduce_max,
                .a = 0,
            },
            .{
                .kind = module_op_transpose,
                .a = 0,
                .b = 1,
            },
        };
        var batch_reduce_max_program: ?*zgml_program = null;
        var batch_reduce_max_session: ?*zgml_session = null;
        defer zgml_session_free(batch_reduce_max_session);
        defer zgml_program_free(batch_reduce_max_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = reduce_input_shape[0..].ptr,
            .input_rank = reduce_input_shape.len,
            .ops = batch_reduce_max_ops[0..].ptr,
            .op_count = batch_reduce_max_ops.len,
        }, &.{ .backend = backend_cpu }, &batch_reduce_max_program));
        try std.testing.expect(batch_reduce_max_program != null);

        var batch_reduce_max_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(batch_reduce_max_program, &batch_reduce_max_requirements));
        try std.testing.expectEqual(module_kind, batch_reduce_max_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), batch_reduce_max_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 3), batch_reduce_max_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(batch_reduce_max_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &batch_reduce_max_session));
        try std.testing.expect(batch_reduce_max_session != null);

        var batch_reduce_max_output = [_]f32{0} ** 3;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(batch_reduce_max_session, &.{
            .input = reduce_input[0..].ptr,
            .input_len = reduce_input.len,
            .output = batch_reduce_max_output[0..].ptr,
            .output_len = batch_reduce_max_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 3), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, &batch_reduce_max_output);
    }

    {
        const feature_narrow_shape = [_]usize{ 1, 4 };
        const feature_narrow_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_narrow,
            .a = 1,
            .b = 1,
            .c = 2,
        }};
        var feature_narrow_program: ?*zgml_program = null;
        var feature_narrow_session: ?*zgml_session = null;
        defer zgml_session_free(feature_narrow_session);
        defer zgml_program_free(feature_narrow_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = feature_narrow_shape[0..].ptr,
            .input_rank = feature_narrow_shape.len,
            .ops = feature_narrow_ops[0..].ptr,
            .op_count = feature_narrow_ops.len,
        }, &.{ .backend = backend_cpu }, &feature_narrow_program));
        try std.testing.expect(feature_narrow_program != null);

        var feature_narrow_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(feature_narrow_program, &feature_narrow_requirements));
        try std.testing.expectEqual(module_kind, feature_narrow_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), feature_narrow_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), feature_narrow_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(feature_narrow_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &feature_narrow_session));
        try std.testing.expect(feature_narrow_session != null);

        const feature_narrow_input = [_]f32{ 10, 20, 30, 40 };
        var feature_narrow_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(feature_narrow_session, &.{
            .input = feature_narrow_input[0..].ptr,
            .input_len = feature_narrow_input.len,
            .output = feature_narrow_output[0..].ptr,
            .output_len = feature_narrow_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 20, 30 }, &feature_narrow_output);
    }

    {
        const strided_feature_shape = [_]usize{ 2, 3 };
        const strided_feature_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_narrow,
            .a = 1,
            .b = 0,
            .c = 1,
        }};
        var strided_feature_program: ?*zgml_program = null;
        var strided_feature_session: ?*zgml_session = null;
        defer zgml_session_free(strided_feature_session);
        defer zgml_program_free(strided_feature_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = strided_feature_shape[0..].ptr,
            .input_rank = strided_feature_shape.len,
            .ops = strided_feature_ops[0..].ptr,
            .op_count = strided_feature_ops.len,
        }, &.{ .backend = backend_cpu }, &strided_feature_program));
        try std.testing.expect(strided_feature_program != null);

        var strided_feature_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(strided_feature_program, &strided_feature_requirements));
        try std.testing.expectEqual(module_kind, strided_feature_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), strided_feature_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), strided_feature_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), strided_feature_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), strided_feature_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(strided_feature_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &strided_feature_session));
        try std.testing.expect(strided_feature_session != null);

        const strided_feature_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var strided_feature_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(strided_feature_session, &.{
            .input = strided_feature_input[0..].ptr,
            .input_len = strided_feature_input.len,
            .output = strided_feature_output[0..].ptr,
            .output_len = strided_feature_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 4 }, &strided_feature_output);
    }

    {
        const diagonal_shape = [_]usize{ 2, 3 };
        const diagonal_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_diagonal,
        }};
        var diagonal_program: ?*zgml_program = null;
        var diagonal_session: ?*zgml_session = null;
        defer zgml_session_free(diagonal_session);
        defer zgml_program_free(diagonal_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = diagonal_shape[0..].ptr,
            .input_rank = diagonal_shape.len,
            .ops = diagonal_ops[0..].ptr,
            .op_count = diagonal_ops.len,
        }, &.{ .backend = backend_cpu }, &diagonal_program));
        try std.testing.expect(diagonal_program != null);

        var diagonal_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(diagonal_program, &diagonal_requirements));
        try std.testing.expectEqual(module_kind, diagonal_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), diagonal_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), diagonal_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(diagonal_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &diagonal_session));
        try std.testing.expect(diagonal_session != null);

        const diagonal_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var diagonal_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(diagonal_session, &.{
            .input = diagonal_input[0..].ptr,
            .input_len = diagonal_input.len,
            .output = diagonal_output[0..].ptr,
            .output_len = diagonal_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 5 }, &diagonal_output);
    }

    {
        const stepped_slice_shape = [_]usize{5};
        const stepped_slice_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_slice,
            .reserved = 2,
            .a = 0,
            .b = 0,
            .c = 3,
        }};
        var stepped_slice_program: ?*zgml_program = null;
        var stepped_slice_session: ?*zgml_session = null;
        defer zgml_session_free(stepped_slice_session);
        defer zgml_program_free(stepped_slice_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = stepped_slice_shape[0..].ptr,
            .input_rank = stepped_slice_shape.len,
            .ops = stepped_slice_ops[0..].ptr,
            .op_count = stepped_slice_ops.len,
        }, &.{ .backend = backend_cpu }, &stepped_slice_program));
        try std.testing.expect(stepped_slice_program != null);

        var stepped_slice_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(stepped_slice_program, &stepped_slice_requirements));
        try std.testing.expectEqual(module_kind, stepped_slice_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 5), stepped_slice_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 3), stepped_slice_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(stepped_slice_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &stepped_slice_session));
        try std.testing.expect(stepped_slice_session != null);

        const stepped_slice_input = [_]f32{ 1, 2, 3, 4, 5 };
        var stepped_slice_output = [_]f32{0} ** 3;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(stepped_slice_session, &.{
            .input = stepped_slice_input[0..].ptr,
            .input_len = stepped_slice_input.len,
            .output = stepped_slice_output[0..].ptr,
            .output_len = stepped_slice_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 3), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5 }, &stepped_slice_output);
    }

    {
        const stepped_feature_slice_shape = [_]usize{ 2, 4 };
        const stepped_feature_slice_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_slice,
            .reserved = 2,
            .a = 1,
            .b = 0,
            .c = 2,
        }};
        var stepped_feature_slice_program: ?*zgml_program = null;
        var stepped_feature_slice_session: ?*zgml_session = null;
        defer zgml_session_free(stepped_feature_slice_session);
        defer zgml_program_free(stepped_feature_slice_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = stepped_feature_slice_shape[0..].ptr,
            .input_rank = stepped_feature_slice_shape.len,
            .ops = stepped_feature_slice_ops[0..].ptr,
            .op_count = stepped_feature_slice_ops.len,
        }, &.{ .backend = backend_cpu }, &stepped_feature_slice_program));
        try std.testing.expect(stepped_feature_slice_program != null);

        var stepped_feature_slice_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(stepped_feature_slice_program, &stepped_feature_slice_requirements));
        try std.testing.expectEqual(module_kind, stepped_feature_slice_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 8), stepped_feature_slice_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), stepped_feature_slice_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(stepped_feature_slice_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &stepped_feature_slice_session));
        try std.testing.expect(stepped_feature_slice_session != null);

        const stepped_feature_slice_input = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
        var stepped_feature_slice_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(stepped_feature_slice_session, &.{
            .input = stepped_feature_slice_input[0..].ptr,
            .input_len = stepped_feature_slice_input.len,
            .output = stepped_feature_slice_output[0..].ptr,
            .output_len = stepped_feature_slice_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 10, 30, 50, 70 }, &stepped_feature_slice_output);
    }

    {
        const direct_rms_shape = [_]usize{ 2, 2 };
        const direct_rms_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_rms_norm,
                .activation = module_activation_gelu,
                .flags = module_flag_weight,
                .a = 2,
                .eps = 1e-5,
            },
            .{
                .kind = module_op_linear,
                .flags = module_flag_bias,
                .a = 2,
                .b = 2,
            },
        };
        var direct_rms_program: ?*zgml_program = null;
        var direct_rms_session: ?*zgml_session = null;
        defer zgml_session_free(direct_rms_session);
        defer zgml_program_free(direct_rms_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = direct_rms_shape[0..].ptr,
            .input_rank = direct_rms_shape.len,
            .ops = direct_rms_ops[0..].ptr,
            .op_count = direct_rms_ops.len,
        }, &.{ .backend = backend_cpu }, &direct_rms_program));
        try std.testing.expect(direct_rms_program != null);

        const direct_rms_weights = [_]f32{
            1, 1.5,
            1, 0,
            0, 1,
        };
        const direct_rms_bias = [_]f32{ 0.1, -0.2 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(direct_rms_program, &.{
            .weights = direct_rms_weights[0..].ptr,
            .weights_len = direct_rms_weights.len,
            .bias = direct_rms_bias[0..].ptr,
            .bias_len = direct_rms_bias.len,
        }, &direct_rms_session));
        try std.testing.expect(direct_rms_session != null);

        const direct_rms_input = [_]f32{ 1, 2, 3, 4 };
        var direct_rms_output = [_]f32{0} ** 4;
        try std.testing.expectEqual(status(.ok), zgml_session_step_direct(
            direct_rms_session,
            direct_rms_input[0..].ptr,
            direct_rms_input.len,
            direct_rms_output[0..].ptr,
            direct_rms_output.len,
        ));

        const direct_gelu = struct {
            fn call(x: f32) f32 {
                const kk = 0.7978845608 * (x + 0.044715 * x * x * x);
                return 0.5 * x * (1.0 + std.math.tanh(kk));
            }
        }.call;
        const row0_inv = 1.0 / std.math.sqrt((@as(f32, 1 * 1 + 2 * 2) / 2.0) + 1e-5);
        const row1_inv = 1.0 / std.math.sqrt((@as(f32, 3 * 3 + 4 * 4) / 2.0) + 1e-5);
        try std.testing.expectApproxEqAbs(direct_gelu(1 * row0_inv) + direct_rms_bias[0], direct_rms_output[0], 1e-4);
        try std.testing.expectApproxEqAbs(direct_gelu(2 * row0_inv * 1.5) + direct_rms_bias[1], direct_rms_output[1], 1e-4);
        try std.testing.expectApproxEqAbs(direct_gelu(3 * row1_inv) + direct_rms_bias[0], direct_rms_output[2], 1e-4);
        try std.testing.expectApproxEqAbs(direct_gelu(4 * row1_inv * 1.5) + direct_rms_bias[1], direct_rms_output[3], 1e-4);

        var direct_rms_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(direct_rms_session, &direct_rms_profile));
        try std.testing.expectEqual(@as(u64, 1), direct_rms_profile.call_count);
        try std.testing.expectEqual(@as(u64, 1), direct_rms_profile.runtime_patch_call_count);
        try std.testing.expect(direct_rms_profile.command_count > 0);
    }

    {
        const direct_log_softmax_shape = [_]usize{ 2, 2 };
        const direct_log_softmax_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_linear,
                .flags = module_flag_bias,
                .a = 2,
                .b = 3,
            },
            .{
                .kind = module_op_log_softmax,
                .a = 0,
            },
        };
        var direct_log_softmax_program: ?*zgml_program = null;
        var direct_log_softmax_session: ?*zgml_session = null;
        defer zgml_session_free(direct_log_softmax_session);
        defer zgml_program_free(direct_log_softmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = direct_log_softmax_shape[0..].ptr,
            .input_rank = direct_log_softmax_shape.len,
            .ops = direct_log_softmax_ops[0..].ptr,
            .op_count = direct_log_softmax_ops.len,
        }, &.{ .backend = backend_cpu }, &direct_log_softmax_program));
        try std.testing.expect(direct_log_softmax_program != null);

        const direct_log_softmax_weights = [_]f32{
            1, 0, 0.5,
            0, 1, -0.5,
        };
        const direct_log_softmax_bias = [_]f32{ 0.1, -0.2, 0.3 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(direct_log_softmax_program, &.{
            .weights = direct_log_softmax_weights[0..].ptr,
            .weights_len = direct_log_softmax_weights.len,
            .bias = direct_log_softmax_bias[0..].ptr,
            .bias_len = direct_log_softmax_bias.len,
        }, &direct_log_softmax_session));
        try std.testing.expect(direct_log_softmax_session != null);

        const direct_log_softmax_input = [_]f32{ 1, 2, 3, 4 };
        var direct_log_softmax_output = [_]f32{0} ** 6;
        try std.testing.expectEqual(status(.ok), zgml_session_step_direct(
            direct_log_softmax_session,
            direct_log_softmax_input[0..].ptr,
            direct_log_softmax_input.len,
            direct_log_softmax_output[0..].ptr,
            direct_log_softmax_output.len,
        ));

        const expectLogSoftmaxRow = struct {
            fn call(logits: [3]f32, actual: []const f32) !void {
                const max_val = @max(logits[0], @max(logits[1], logits[2]));
                const log_denom = max_val + @log(@exp(logits[0] - max_val) + @exp(logits[1] - max_val) + @exp(logits[2] - max_val));
                try std.testing.expectApproxEqAbs(logits[0] - log_denom, actual[0], 1e-6);
                try std.testing.expectApproxEqAbs(logits[1] - log_denom, actual[1], 1e-6);
                try std.testing.expectApproxEqAbs(logits[2] - log_denom, actual[2], 1e-6);
            }
        }.call;
        try expectLogSoftmaxRow(.{ 1.1, 1.8, -0.2 }, direct_log_softmax_output[0..3]);
        try expectLogSoftmaxRow(.{ 3.1, 3.8, -0.2 }, direct_log_softmax_output[3..6]);
    }

    {
        const direct_log_softmax32_shape = [_]usize{ 2, 64 };
        const direct_log_softmax32_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_linear,
                .flags = module_flag_bias,
                .a = 64,
                .b = 32,
            },
            .{
                .kind = module_op_log_softmax,
                .a = 0,
            },
        };
        var direct_log_softmax32_program: ?*zgml_program = null;
        var direct_log_softmax32_session: ?*zgml_session = null;
        defer zgml_session_free(direct_log_softmax32_session);
        defer zgml_program_free(direct_log_softmax32_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = direct_log_softmax32_shape[0..].ptr,
            .input_rank = direct_log_softmax32_shape.len,
            .ops = direct_log_softmax32_ops[0..].ptr,
            .op_count = direct_log_softmax32_ops.len,
        }, &.{ .backend = backend_cpu }, &direct_log_softmax32_program));
        try std.testing.expect(direct_log_softmax32_program != null);

        var direct_log_softmax32_weights: [64 * 32]f32 = undefined;
        var direct_log_softmax32_bias: [32]f32 = undefined;
        var direct_log_softmax32_input: [2 * 64]f32 = undefined;
        for (&direct_log_softmax32_weights, 0..) |*value, index| {
            const centered: i32 = @as(i32, @intCast(index % 23)) - 11;
            value.* = @as(f32, @floatFromInt(centered)) / 97.0;
        }
        for (&direct_log_softmax32_bias, 0..) |*value, index| {
            const centered: i32 = @as(i32, @intCast(index % 13)) - 6;
            value.* = @as(f32, @floatFromInt(centered)) / 31.0;
        }
        for (&direct_log_softmax32_input, 0..) |*value, index| {
            const centered: i32 = @as(i32, @intCast(index % 17)) - 8;
            value.* = @as(f32, @floatFromInt(centered)) / 19.0;
        }

        try std.testing.expectEqual(status(.ok), zgml_session_bind(direct_log_softmax32_program, &.{
            .weights = direct_log_softmax32_weights[0..].ptr,
            .weights_len = direct_log_softmax32_weights.len,
            .bias = direct_log_softmax32_bias[0..].ptr,
            .bias_len = direct_log_softmax32_bias.len,
        }, &direct_log_softmax32_session));
        try std.testing.expect(direct_log_softmax32_session != null);

        var direct_log_softmax32_output = [_]f32{0} ** (2 * 32);
        try std.testing.expectEqual(status(.ok), zgml_session_step_direct(
            direct_log_softmax32_session,
            direct_log_softmax32_input[0..].ptr,
            direct_log_softmax32_input.len,
            direct_log_softmax32_output[0..].ptr,
            direct_log_softmax32_output.len,
        ));

        for (0..2) |row| {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*logit, col| {
                var acc = direct_log_softmax32_bias[col];
                for (0..64) |k| {
                    acc += direct_log_softmax32_input[row * 64 + k] * direct_log_softmax32_weights[k * 32 + col];
                }
                logit.* = acc;
            }
            var max_val = -std.math.inf(f32);
            for (logits) |logit| max_val = @max(max_val, logit);
            var sum_exp: f32 = 0;
            for (logits) |logit| sum_exp += @exp(logit - max_val);
            const log_denom = max_val + @log(sum_exp);
            for (logits, 0..) |logit, col| {
                try std.testing.expectApproxEqAbs(logit - log_denom, direct_log_softmax32_output[row * 32 + col], 1e-4);
            }
        }
    }

    {
        const broadcast_input_shape = [_]usize{3};
        const broadcast_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_broadcast_to,
                .a = 2,
                .b = 2,
                .c = 3,
            },
            .{
                .kind = module_op_log_softmax,
                .a = 0,
            },
        };
        var broadcast_program: ?*zgml_program = null;
        var broadcast_session: ?*zgml_session = null;
        defer zgml_session_free(broadcast_session);
        defer zgml_program_free(broadcast_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = broadcast_input_shape[0..].ptr,
            .input_rank = broadcast_input_shape.len,
            .ops = broadcast_ops[0..].ptr,
            .op_count = broadcast_ops.len,
        }, &.{ .backend = backend_cpu }, &broadcast_program));
        try std.testing.expect(broadcast_program != null);

        var broadcast_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(broadcast_program, &broadcast_requirements));
        try std.testing.expectEqual(module_kind, broadcast_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 3), broadcast_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 6), broadcast_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), broadcast_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), broadcast_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(broadcast_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &broadcast_session));
        try std.testing.expect(broadcast_session != null);

        const broadcast_input = [_]f32{ 1, 2, 3 };
        var broadcast_output = [_]f32{0} ** 6;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(broadcast_session, &.{
            .input = broadcast_input[0..].ptr,
            .input_len = broadcast_input.len,
            .output = broadcast_output[0..].ptr,
            .output_len = broadcast_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 6), result.output_len);
        const log_denom = @log(@exp(@as(f32, 1)) + @exp(@as(f32, 2)) + @exp(@as(f32, 3)));
        const expected = [_]f32{
            1 - log_denom, 2 - log_denom, 3 - log_denom,
            1 - log_denom, 2 - log_denom, 3 - log_denom,
        };
        for (broadcast_output, expected) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-6);
        }
    }

    {
        const tiled_input_shape = [_]usize{ 2, 3 };
        const tiled_ops = [_]zgml_module_op_desc{
            .{
                .kind = module_op_broadcast_to,
                .a = 2,
                .b = 4,
                .c = 9,
            },
        };
        var tiled_program: ?*zgml_program = null;
        var tiled_session: ?*zgml_session = null;
        defer zgml_session_free(tiled_session);
        defer zgml_program_free(tiled_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = tiled_input_shape[0..].ptr,
            .input_rank = tiled_input_shape.len,
            .ops = tiled_ops[0..].ptr,
            .op_count = tiled_ops.len,
        }, &.{ .backend = backend_cpu }, &tiled_program));
        try std.testing.expect(tiled_program != null);

        var tiled_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(tiled_program, &tiled_requirements));
        try std.testing.expectEqual(module_kind, tiled_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 6), tiled_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 36), tiled_requirements.output_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(tiled_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &tiled_session));
        try std.testing.expect(tiled_session != null);

        const tiled_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var tiled_output = [_]f32{0} ** 36;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(tiled_session, &.{
            .input = tiled_input[0..].ptr,
            .input_len = tiled_input.len,
            .output = tiled_output[0..].ptr,
            .output_len = tiled_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 36), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{
            1, 2, 3, 1, 2, 3, 1, 2, 3,
            4, 5, 6, 4, 5, 6, 4, 5, 6,
            1, 2, 3, 1, 2, 3, 1, 2, 3,
            4, 5, 6, 4, 5, 6, 4, 5, 6,
        }, &tiled_output);
    }

    {
        const batched_norm_shape = [_]usize{ 2, 2 };
        const norm_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_layer_norm,
            .activation = module_activation_gelu,
            .flags = module_flag_weight | module_flag_bias,
            .a = 2,
            .eps = 1e-5,
        }};
        var norm_program: ?*zgml_program = null;
        var norm_session: ?*zgml_session = null;
        defer zgml_session_free(norm_session);
        defer zgml_program_free(norm_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = batched_norm_shape[0..].ptr,
            .input_rank = batched_norm_shape.len,
            .ops = norm_ops[0..].ptr,
            .op_count = norm_ops.len,
        }, &.{ .backend = backend_cpu }, &norm_program));
        try std.testing.expect(norm_program != null);

        var norm_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(norm_program, &norm_requirements));
        try std.testing.expectEqual(module_kind, norm_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 4), norm_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), norm_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 2), norm_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 2), norm_requirements.bias_len);

        const norm_weights = [_]f32{ 1, 1.5 };
        const norm_bias = [_]f32{ 0.1, -0.2 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(norm_program, &.{
            .weights = norm_weights[0..].ptr,
            .weights_len = norm_weights.len,
            .bias = norm_bias[0..].ptr,
            .bias_len = norm_bias.len,
        }, &norm_session));
        try std.testing.expect(norm_session != null);

        const norm_input = [_]f32{ 1, 2, 3, 1 };
        var norm_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(norm_session, &.{
            .input = norm_input[0..].ptr,
            .input_len = norm_input.len,
            .output = norm_output[0..].ptr,
            .output_len = norm_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);

        const row0_inv = 1.0 / std.math.sqrt(@as(f32, 0.25) + 1e-5);
        const row1_inv = 1.0 / std.math.sqrt(@as(f32, 1.0) + 1e-5);
        const gelu_fn = struct {
            fn call(x: f32) f32 {
                const kk = 0.7978845608 * (x + 0.044715 * x * x * x);
                return 0.5 * x * (1.0 + std.math.tanh(kk));
            }
        }.call;
        try std.testing.expectApproxEqAbs(gelu_fn((@as(f32, -0.5) * row0_inv) * norm_weights[0] + norm_bias[0]), norm_output[0], 1e-5);
        try std.testing.expectApproxEqAbs(gelu_fn((@as(f32, 0.5) * row0_inv) * norm_weights[1] + norm_bias[1]), norm_output[1], 1e-5);
        try std.testing.expectApproxEqAbs(gelu_fn((@as(f32, 1.0) * row1_inv) * norm_weights[0] + norm_bias[0]), norm_output[2], 1e-5);
        try std.testing.expectApproxEqAbs(gelu_fn((@as(f32, -1.0) * row1_inv) * norm_weights[1] + norm_bias[1]), norm_output[3], 1e-5);
    }

    {
        const token_shape = [_]usize{3};
        const embedding_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_embedding,
            .flags = module_flag_weight,
            .a = 4,
            .b = 3,
        }};
        var embedding_program: ?*zgml_program = null;
        var embedding_session: ?*zgml_session = null;
        defer zgml_session_free(embedding_session);
        defer zgml_program_free(embedding_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = token_shape[0..].ptr,
            .input_rank = token_shape.len,
            .ops = embedding_ops[0..].ptr,
            .op_count = embedding_ops.len,
        }, &.{ .backend = backend_cpu }, &embedding_program));
        try std.testing.expect(embedding_program != null);

        var embedding_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(embedding_program, &embedding_requirements));
        try std.testing.expectEqual(module_kind, embedding_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 3), embedding_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 9), embedding_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 12), embedding_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), embedding_requirements.bias_len);

        const embedding_weights = [_]f32{
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
            40, 41, 42,
        };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(embedding_program, &.{
            .weights = embedding_weights[0..].ptr,
            .weights_len = embedding_weights.len,
        }, &embedding_session));
        try std.testing.expect(embedding_session != null);

        const token_input = [_]f32{ 2, 0, 3 };
        var embedding_output = [_]f32{0} ** 9;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(embedding_session, &.{
            .input = token_input[0..].ptr,
            .input_len = token_input.len,
            .output = embedding_output[0..].ptr,
            .output_len = embedding_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 9), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{
            30, 31, 32,
            10, 11, 12,
            40, 41, 42,
        }, &embedding_output);
    }

    {
        const max_pool_input_shape = [_]usize{ 1, 4, 4 };
        const max_pool_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_max_pool2d,
            .a = 2,
            .b = 2,
        }};
        var max_pool_program: ?*zgml_program = null;
        var max_pool_session: ?*zgml_session = null;
        defer zgml_session_free(max_pool_session);
        defer zgml_program_free(max_pool_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = max_pool_input_shape[0..].ptr,
            .input_rank = max_pool_input_shape.len,
            .ops = max_pool_ops[0..].ptr,
            .op_count = max_pool_ops.len,
        }, &.{ .backend = backend_cpu }, &max_pool_program));
        try std.testing.expect(max_pool_program != null);

        var max_pool_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(max_pool_program, &max_pool_requirements));
        try std.testing.expectEqual(module_kind, max_pool_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 16), max_pool_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 4), max_pool_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), max_pool_requirements.weights_len);
        try std.testing.expectEqual(@as(usize, 0), max_pool_requirements.bias_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(max_pool_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &max_pool_session));
        try std.testing.expect(max_pool_session != null);

        const max_pool_input = [_]f32{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        };
        var max_pool_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(max_pool_session, &.{
            .input = max_pool_input[0..].ptr,
            .input_len = max_pool_input.len,
            .output = max_pool_output[0..].ptr,
            .output_len = max_pool_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 6, 8, 14, 16 }, &max_pool_output);
    }

    {
        const avg_pool_input_shape = [_]usize{ 1, 4, 4 };
        const avg_pool_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_avg_pool2d,
            .a = 2,
            .b = 2,
        }};
        var avg_pool_program: ?*zgml_program = null;
        var avg_pool_session: ?*zgml_session = null;
        defer zgml_session_free(avg_pool_session);
        defer zgml_program_free(avg_pool_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = avg_pool_input_shape[0..].ptr,
            .input_rank = avg_pool_input_shape.len,
            .ops = avg_pool_ops[0..].ptr,
            .op_count = avg_pool_ops.len,
        }, &.{ .backend = backend_cpu }, &avg_pool_program));
        try std.testing.expect(avg_pool_program != null);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(avg_pool_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &avg_pool_session));
        try std.testing.expect(avg_pool_session != null);

        const avg_pool_input = [_]f32{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        };
        var avg_pool_output = [_]f32{0} ** 4;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(avg_pool_session, &.{
            .input = avg_pool_input[0..].ptr,
            .input_len = avg_pool_input.len,
            .output = avg_pool_output[0..].ptr,
            .output_len = avg_pool_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 4), result.output_len);
        try std.testing.expectEqualSlices(f32, &.{ 3.5, 5.5, 11.5, 13.5 }, &avg_pool_output);
    }

    {
        const log_softmax_ops = [_]zgml_module_op_desc{.{
            .kind = module_op_log_softmax,
            .a = 0,
        }};
        var log_softmax_program: ?*zgml_program = null;
        var log_softmax_session: ?*zgml_session = null;
        defer zgml_session_free(log_softmax_session);
        defer zgml_program_free(log_softmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = log_softmax_ops[0..].ptr,
            .op_count = log_softmax_ops.len,
        }, &.{ .backend = backend_cpu }, &log_softmax_program));
        try std.testing.expect(log_softmax_program != null);

        var log_softmax_requirements = zgml_program_requirements{};
        try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(log_softmax_program, &log_softmax_requirements));
        try std.testing.expectEqual(module_kind, log_softmax_requirements.model_kind);
        try std.testing.expectEqual(@as(usize, 2), log_softmax_requirements.input_len);
        try std.testing.expectEqual(@as(usize, 2), log_softmax_requirements.output_len);
        try std.testing.expectEqual(@as(usize, 0), log_softmax_requirements.weights_len);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(log_softmax_program, &.{
            .weights = null,
            .weights_len = 0,
        }, &log_softmax_session));
        try std.testing.expect(log_softmax_session != null);

        const log_input = [_]f32{ 1, 2 };
        var log_output = [_]f32{0} ** 2;
        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(log_softmax_session, &.{
            .input = log_input[0..].ptr,
            .input_len = log_input.len,
            .output = log_output[0..].ptr,
            .output_len = log_output.len,
        }, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        const log_denom = @log(@exp(@as(f32, 1)) + @exp(@as(f32, 2)));
        try std.testing.expectApproxEqAbs(@as(f32, 1) - log_denom, log_output[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 2) - log_denom, log_output[1], 1e-6);

        var webgpu_log_softmax_program: ?*zgml_program = null;
        var webgpu_log_softmax_session: ?*zgml_session = null;
        defer zgml_session_free(webgpu_log_softmax_session);
        defer zgml_program_free(webgpu_log_softmax_program);

        try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
            .input_shape = input_shape[0..].ptr,
            .input_rank = input_shape.len,
            .ops = log_softmax_ops[0..].ptr,
            .op_count = log_softmax_ops.len,
        }, &.{ .backend = backend_webgpu }, &webgpu_log_softmax_program));
        try std.testing.expect(webgpu_log_softmax_program != null);

        var webgpu_log_inspection = zgml_program_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_program_inspect(webgpu_log_softmax_program, &webgpu_log_inspection));
        try std.testing.expectEqual(@as(u64, backend_webgpu), webgpu_log_inspection.backend);
        try std.testing.expectEqual(@as(u64, @intFromBool(build_options.use_wgpu)), webgpu_log_inspection.execution_supported);
        try std.testing.expectEqual(@as(u64, 1), webgpu_log_inspection.external_resources_supported);
        try std.testing.expectEqual(@as(u64, 1), webgpu_log_inspection.op_count);
        try std.testing.expectEqual(@as(u64, 1), webgpu_log_inspection.command_count);
        try std.testing.expect(webgpu_log_inspection.command_stencil_hash != 0);

        if (build_options.use_wgpu) {
            try std.testing.expectEqual(status(.ok), zgml_session_bind(webgpu_log_softmax_program, &.{
                .weights = null,
                .weights_len = 0,
            }, &webgpu_log_softmax_session));
            try std.testing.expect(webgpu_log_softmax_session != null);

            const webgpu_log_input = [_]f32{ 1, 2 };
            var webgpu_log_output = [_]f32{0} ** 2;
            result = .{};
            try std.testing.expectEqual(status(.ok), zgml_session_step(webgpu_log_softmax_session, &.{
                .input = webgpu_log_input[0..].ptr,
                .input_len = webgpu_log_input.len,
                .output = webgpu_log_output[0..].ptr,
                .output_len = webgpu_log_output.len,
            }, &result));
            try std.testing.expectEqual(@as(usize, 2), result.output_len);
            try std.testing.expectApproxEqAbs(@as(f32, 1) - log_denom, webgpu_log_output[0], 1e-5);
            try std.testing.expectApproxEqAbs(@as(f32, 2) - log_denom, webgpu_log_output[1], 1e-5);

            var webgpu_log_profile = zgml_runtime_profile{};
            try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(webgpu_log_softmax_session, &webgpu_log_profile));
            try std.testing.expectEqual(@as(u64, 1), webgpu_log_profile.backend_op_count);
            try std.testing.expectEqual(@as(u64, 1), webgpu_log_profile.backend_dispatch_count);
            try std.testing.expectEqual(@as(u64, 1), webgpu_log_profile.sync_count);

            var webgpu_device_input: ?*zgml_buffer = null;
            var webgpu_device_output: ?*zgml_buffer = null;
            var webgpu_device_session: ?*zgml_session = null;
            defer zgml_session_free(webgpu_device_session);
            defer zgml_buffer_free(webgpu_device_output);
            defer zgml_buffer_free(webgpu_device_input);

            try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_log_softmax_program, program_buffer_input, backend_webgpu, &webgpu_device_input));
            try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_log_softmax_program, program_buffer_output, backend_webgpu, &webgpu_device_output));
            try std.testing.expect(webgpu_device_input != null);
            try std.testing.expect(webgpu_device_output != null);

            const device_log_input = [_]f32{ 3, 1 };
            try std.testing.expectEqual(status(.ok), zgml_buffer_write(webgpu_device_input, 0, device_log_input[0..].ptr, device_log_input.len * @sizeOf(f32)));
            try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_log_softmax_program, &.{
                .weights = null,
                .weights_len = 0,
                .input = webgpu_device_input,
                .input_len = device_log_input.len,
                .output = webgpu_device_output,
                .output_len = device_log_input.len,
            }, &webgpu_device_session));

            var webgpu_device_session_inspection = zgml_session_inspection{};
            try std.testing.expectEqual(status(.ok), zgml_session_inspect(webgpu_device_session, &webgpu_device_session_inspection));
            try std.testing.expectEqual(module_kind, webgpu_device_session_inspection.model_kind);
            try std.testing.expectEqual(backend_webgpu, webgpu_device_session_inspection.backend);
            try std.testing.expectEqual(buffer_storage_external_resource, webgpu_device_session_inspection.output_storage);
            try std.testing.expectEqual(@as(u64, 0), webgpu_device_session_inspection.host_binding_count);
            try std.testing.expectEqual(@as(u64, 2), webgpu_device_session_inspection.resource_binding_count);

            result = .{};
            try std.testing.expectEqual(status(.ok), zgml_session_step(webgpu_device_session, null, &result));
            try std.testing.expectEqual(@as(usize, 2), result.output_len);
            var device_log_output = [_]f32{0} ** 2;
            try std.testing.expectEqual(status(.ok), zgml_buffer_read(webgpu_device_output, 0, device_log_output[0..].ptr, device_log_output.len * @sizeOf(f32)));
            const device_log_denom = @log(@exp(@as(f32, 3)) + @exp(@as(f32, 1)));
            try std.testing.expectApproxEqAbs(@as(f32, 3) - device_log_denom, device_log_output[0], 1e-5);
            try std.testing.expectApproxEqAbs(@as(f32, 1) - device_log_denom, device_log_output[1], 1e-5);

            var webgpu_device_profile = zgml_runtime_profile{};
            try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(webgpu_device_session, &webgpu_device_profile));
            try std.testing.expectEqual(@as(u64, 1), webgpu_device_profile.call_count);
            try std.testing.expectEqual(@as(u64, 1), webgpu_device_profile.backend_op_count);
            try std.testing.expectEqual(@as(u64, 1), webgpu_device_profile.backend_dispatch_count);
            try std.testing.expectEqual(@as(u64, 0), webgpu_device_profile.sync_count);
        } else {
            try std.testing.expectEqual(status(.unsupported), zgml_session_bind(webgpu_log_softmax_program, &.{
                .weights = null,
                .weights_len = 0,
            }, &webgpu_log_softmax_session));
            try std.testing.expectEqual(@as(?*zgml_session, null), webgpu_log_softmax_session);
        }
    }

    var webgpu_program: ?*zgml_program = null;
    var webgpu_session: ?*zgml_session = null;
    defer zgml_session_free(webgpu_session);
    defer zgml_program_free(webgpu_program);
    try std.testing.expectEqual(status(.ok), zgml_module_program_compile(&.{
        .input_shape = input_shape[0..].ptr,
        .input_rank = input_shape.len,
        .ops = ops[0..].ptr,
        .op_count = ops.len,
    }, &.{ .backend = backend_webgpu }, &webgpu_program));
    try std.testing.expect(webgpu_program != null);

    if (build_options.use_wgpu) {
        var device_weights: ?*zgml_buffer = null;
        var device_input: ?*zgml_buffer = null;
        var device_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(device_output);
        defer zgml_buffer_free(device_input);
        defer zgml_buffer_free(device_weights);
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_program, program_buffer_weights, backend_webgpu, &device_weights));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_program, program_buffer_input, backend_webgpu, &device_input));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_program, program_buffer_output, backend_webgpu, &device_output));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_weights, 0, weights[0..].ptr, weights.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_input, 0, input[0..].ptr, input.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_program, &.{
            .weights = device_weights,
            .weights_len = weights.len,
            .input = device_input,
            .input_len = input.len,
            .output = device_output,
            .output_len = output.len,
        }, &webgpu_session));
        try std.testing.expectEqual(status(.ok), zgml_session_inspect(webgpu_session, &session_inspection));
        try std.testing.expectEqual(module_kind, session_inspection.model_kind);
        try std.testing.expectEqual(backend_webgpu, session_inspection.backend);
        try std.testing.expectEqual(buffer_storage_external_resource, session_inspection.output_storage);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.persistent_binding_count);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.step_input_count);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);
        try std.testing.expectEqual(@as(u64, 0), session_inspection.host_binding_count);
        try std.testing.expectEqual(@as(u64, 3), session_inspection.resource_binding_count);
        try std.testing.expect(session_inspection.binding_shape_hash != 0);

        result = .{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(webgpu_session, null, &result));
        try std.testing.expectEqual(@as(usize, 2), result.output_len);
        var device_output_values = [_]f32{ -1, -1 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, device_output_values[0..].ptr, device_output_values.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 2)) / (@exp(@as(f32, 2)) + @exp(@as(f32, 3))), device_output_values[0], 1e-5);
        try std.testing.expectApproxEqAbs(@exp(@as(f32, 3)) / (@exp(@as(f32, 2)) + @exp(@as(f32, 3))), device_output_values[1], 1e-5);

        var device_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(webgpu_session, &device_profile));
        try std.testing.expectEqual(@as(u64, 1), device_profile.call_count);
        try std.testing.expect(device_profile.backend_op_count > 0);
        try std.testing.expect(device_profile.backend_dispatch_count > 0);
        try std.testing.expectEqual(@as(u64, 0), device_profile.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 0), device_profile.sync_count);
    } else {
        var resource_weights: ?*zgml_buffer = null;
        var resource_input: ?*zgml_buffer = null;
        var resource_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(resource_output);
        defer zgml_buffer_free(resource_input);
        defer zgml_buffer_free(resource_weights);
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 801,
            .byte_len = weights.len * @sizeOf(f32),
        }, &resource_weights));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 802,
            .byte_len = input.len * @sizeOf(f32),
        }, &resource_input));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_write,
            .handle = 803,
            .byte_len = output.len * @sizeOf(f32),
        }, &resource_output));
        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_program, &.{
            .weights = resource_weights,
            .weights_len = weights.len,
            .input = resource_input,
            .input_len = input.len,
            .output = resource_output,
            .output_len = output.len,
        }, &webgpu_session));
        try std.testing.expectEqual(status(.ok), zgml_session_inspect(webgpu_session, &session_inspection));
        try std.testing.expectEqual(module_kind, session_inspection.model_kind);
        try std.testing.expectEqual(backend_webgpu, session_inspection.backend);
        try std.testing.expectEqual(buffer_storage_external_resource, session_inspection.output_storage);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.persistent_binding_count);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.step_input_count);
        try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);
        try std.testing.expectEqual(@as(u64, 0), session_inspection.host_binding_count);
        try std.testing.expectEqual(@as(u64, 3), session_inspection.resource_binding_count);
        try std.testing.expect(session_inspection.binding_shape_hash != 0);
        result = .{};
        try std.testing.expectEqual(status(.unsupported), zgml_session_step(webgpu_session, null, &result));
        try std.testing.expectEqual(@as(usize, 0), result.output_len);
    }
}

test "C ABI rebinds a compiled tiny linear program with new persistent weights" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    defer zgml_program_free(program);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));

    const input = [_]f32{ 2, 3 };
    var result = zgml_step_result{};

    {
        var session: ?*zgml_session = null;
        defer zgml_session_free(session);
        const weights = [_]f32{ 1, 0, 0, 1 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
            .weights = weights[0..].ptr,
            .weights_len = weights.len,
        }, &session));

        var output = [_]f32{0} ** 2;
        try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
            .input = input[0..].ptr,
            .input_len = input.len,
            .output = output[0..].ptr,
            .output_len = output.len,
        }, &result));
        try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &output);
    }

    {
        var session: ?*zgml_session = null;
        defer zgml_session_free(session);
        const weights = [_]f32{ 2, 0, 0, 2 };
        const bias = [_]f32{ 1, -1 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
            .weights = weights[0..].ptr,
            .weights_len = weights.len,
            .bias = bias[0..].ptr,
            .bias_len = bias.len,
        }, &session));

        var output = [_]f32{0} ** 2;
        try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
            .input = input[0..].ptr,
            .input_len = input.len,
            .output = output[0..].ptr,
            .output_len = output.len,
        }, &result));
        try std.testing.expectEqualSlices(f32, &.{ 5, 5 }, &output);
    }
}

test "C ABI child handles retain parents until child free" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    try std.testing.expectEqual(@as(usize, 1), programHandle(program).?.ref_count);

    const weights = [_]f32{ 1, 0, 0, 1 };
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = weights[0..].ptr,
        .weights_len = weights.len,
    }, &session));
    try std.testing.expectEqual(@as(usize, 2), programHandle(program).?.ref_count);

    zgml_model_free(model);
    model = null;
    zgml_program_free(program);
    try std.testing.expectEqual(@as(usize, 1), programHandle(program).?.ref_count);
    program = null;

    const input = [_]f32{ 4, 5 };
    var output = [_]f32{0} ** 2;
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
        .input = input[0..].ptr,
        .input_len = input.len,
        .output = output[0..].ptr,
        .output_len = output.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, &output);
}

test "C ABI LLaMA program retains model after public model free" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    defer zgml_program_free(program);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_cpu,
        .context_len = 4,
        .batch = 1,
    }, &program));
    try std.testing.expectEqual(@as(usize, 2), modelHandle(model).?.ref_count);

    zgml_model_free(model);
    model = null;

    var inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &inspection));
    try std.testing.expectEqual(@as(u64, backend_cpu), inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 0), inspection.external_resources_supported);
    try std.testing.expect(inspection.buffer_count > 0);
    try std.testing.expect(inspection.buffer_byte_len > 0);
    try std.testing.expect(inspection.command_count > 0);
}

test "C ABI buffers can back tiny linear session bindings" {
    var weights_buffer: ?*zgml_buffer = null;
    var input_buffer: ?*zgml_buffer = null;
    var output_buffer: ?*zgml_buffer = null;
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);
    defer zgml_buffer_free(output_buffer);
    defer zgml_buffer_free(input_buffer);
    defer zgml_buffer_free(weights_buffer);

    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_create(&.{ .byte_len = 0 }, &weights_buffer));
    try std.testing.expect(weights_buffer == null);

    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = 4 * @sizeOf(f32) }, &weights_buffer));
    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = 2 * @sizeOf(f32) }, &input_buffer));
    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = 3 * @sizeOf(f32) }, &output_buffer));
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), zgml_buffer_size(weights_buffer));
    try std.testing.expectEqual(@as(usize, 0), zgml_buffer_size(null));

    try std.testing.expect(zgml_buffer_data(null) == null);

    const weights = [_]f32{ 1, 0, 0, 1 };
    const input = [_]f32{ 2, 3 };
    const sentinel = [_]f32{-999} ** 3;
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(weights_buffer, 0, &weights, @sizeOf(@TypeOf(weights))));
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(input_buffer, 0, &input, @sizeOf(@TypeOf(input))));
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(output_buffer, 0, &sentinel, @sizeOf(@TypeOf(sentinel))));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_buffer_write(output_buffer, @sizeOf(@TypeOf(sentinel)), &sentinel, 1));

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_bind_buffers(program, &.{
        .weights = weights_buffer,
        .weights_len = 5,
    }, &session));
    try std.testing.expect(session == null);
    try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(program, &.{
        .weights = weights_buffer,
        .weights_len = 4,
        .input = input_buffer,
        .input_len = 2,
        .output = output_buffer,
        .output_len = 3,
    }, &session));
    var session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_inspect(null, &session_inspection));
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_inspect(session, null));
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(tiny_linear_kind, session_inspection.model_kind);
    try std.testing.expectEqual(backend_cpu, session_inspection.backend);
    try std.testing.expectEqual(buffer_storage_host, session_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_none, session_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.position);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.context_len);
    try std.testing.expectEqual(@as(u64, 2), session_inspection.persistent_binding_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_input_count);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);
    try std.testing.expectEqual(@as(u64, 4), session_inspection.host_binding_count);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.resource_binding_count);
    try std.testing.expect(session_inspection.binding_shape_hash != 0);

    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    var output = [_]f32{0} ** 3;
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(output_buffer, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, output[0..2]);
    try std.testing.expectEqual(@as(f32, -999), output[2]);

    const updated_weights = [_]f32{ 2, 0, 0, 2 };
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(weights_buffer, 0, &updated_weights, @sizeOf(@TypeOf(updated_weights))));
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(output_buffer, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, output[0..2]);
    var direct_output = [_]f32{0} ** 2;
    try std.testing.expectEqual(status(.ok), zgml_session_step_direct(session, &input, input.len, &direct_output, direct_output.len));
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &direct_output);
    try std.testing.expectEqual(status(.ok), zgml_session_upload_persistent_range(session, 0, 1));
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(output_buffer, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 4, 6 }, output[0..2]);
    direct_output = .{ 0, 0 };
    try std.testing.expectEqual(status(.ok), zgml_session_step_direct(session, &input, input.len, &direct_output, direct_output.len));
    try std.testing.expectEqualSlices(f32, &.{ 4, 6 }, &direct_output);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_upload_persistent_range(session, 2, 1));

    const identity_weights = [_]f32{ 1, 0, 0, 1 };
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(weights_buffer, 0, &identity_weights, @sizeOf(@TypeOf(identity_weights))));
    try std.testing.expectEqual(status(.ok), zgml_session_upload_persistent(session));
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(output_buffer, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, output[0..2]);
}

test "C ABI native eager linear writes caller output" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const weights = [_]f32{
        1, 10,
        2, 20,
        3, 30,
    };
    const bias = [_]f32{ 0.5, -1 };
    var output = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_linear_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 14.5, 139, 32.5, 319 }, &output);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_linear_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len - 1,
        null,
        0,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
    ));
}

test "C ABI native eager linear accepts transposed caller weights" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const weights_out_in = [_]f32{
        1,  2,  3,
        10, 20, 30,
    };
    const bias = [_]f32{ 0.5, -1 };
    var output = [_]f32{0} ** 4;
    var activated = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_linear_transposed_weights_f32(
        input[0..].ptr,
        input.len,
        weights_out_in[0..].ptr,
        weights_out_in.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 14.5, 139, 32.5, 319 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_linear_activation_transposed_weights_f32(
        input[0..].ptr,
        input.len,
        weights_out_in[0..].ptr,
        weights_out_in.len,
        bias[0..].ptr,
        bias.len,
        activated[0..].ptr,
        activated.len,
        2,
        3,
        2,
        module_activation_relu,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 14.5, 139, 32.5, 319 }, &activated);
}

test "C ABI native eager matmul writes caller output" {
    const lhs = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const rhs = [_]f32{
        1, 2,
        3, 4,
        5, 6,
    };
    var output = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_matmul_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64 }, &output);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_matmul_f32(
        lhs[0..].ptr,
        lhs.len - 1,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
    ));
}

test "C ABI native linear MSE SGD training learns caller-owned weights" {
    const input = [_]f32{
        -1, -1,
        -1, 1,
        1,  -1,
        1,  1,
    };
    const target = [_]f32{
        -0.5,
        -2.5,
        2.5,
        0.5,
    };
    var weight = [_]f32{ 0, 0 };
    var bias = [_]f32{0};
    var output = [_]f32{0} ** 4;
    var grad_weight = [_]f32{0} ** 2;
    var train_loss: f32 = 0;

    try std.testing.expectEqual(status(.shape_mismatch), zgml_train_linear_mse_sgd_f32(
        input[0..].ptr,
        input.len,
        target[0..].ptr,
        target.len,
        weight[0..].ptr,
        weight.len - 1,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        grad_weight[0..].ptr,
        grad_weight.len,
        4,
        2,
        1,
        0.04,
        0,
        &train_loss,
    ));

    for (0..80) |_| {
        try std.testing.expectEqual(status(.ok), zgml_train_linear_mse_sgd_f32(
            input[0..].ptr,
            input.len,
            target[0..].ptr,
            target.len,
            weight[0..].ptr,
            weight.len,
            bias[0..].ptr,
            bias.len,
            output[0..].ptr,
            output.len,
            grad_weight[0..].ptr,
            grad_weight.len,
            4,
            2,
            1,
            0.04,
            0,
            &train_loss,
        ));
    }

    var heldout = [_]f32{0};
    const heldout_input = [_]f32{ 1, -1 };
    try std.testing.expectEqual(status(.ok), zgml_eager_linear_f32(
        heldout_input[0..].ptr,
        heldout_input.len,
        weight[0..].ptr,
        weight.len,
        bias[0..].ptr,
        bias.len,
        heldout[0..].ptr,
        heldout.len,
        1,
        2,
        1,
    ));
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), heldout[0], 0.02);
    try std.testing.expect(train_loss < 0.001);
}

test "C ABI native linear MSE SGD bulk training consumes indexed tensor dataset" {
    const dataset_input = [_]f32{
        -1, -1,
        -1, 1,
        1,  -1,
        1,  1,
        2,  -2,
    };
    const dataset_target = [_]f32{
        -0.5,
        -2.5,
        2.5,
        0.5,
        5.0,
    };
    const indices = [_]u32{ 0, 1, 2, 3 };
    var weight = [_]f32{ 0, 0 };
    var bias = [_]f32{0};
    var batch_input = [_]f32{0} ** 4;
    var batch_target = [_]f32{0} ** 2;
    var output = [_]f32{0} ** 2;
    var grad_weight = [_]f32{0} ** 2;
    var train_loss: f32 = 0;
    var steps: usize = 0;

    try std.testing.expectEqual(status(.shape_mismatch), zgml_train_linear_mse_sgd_f32_bulk(
        dataset_input[0..].ptr,
        dataset_input.len,
        dataset_target[0..].ptr,
        dataset_target.len,
        indices[0..].ptr,
        indices.len - 1,
        batch_input[0..].ptr,
        batch_input.len,
        batch_target[0..].ptr,
        batch_target.len,
        weight[0..].ptr,
        weight.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        grad_weight[0..].ptr,
        grad_weight.len,
        indices.len,
        2,
        2,
        1,
        1,
        0.04,
        0,
        &train_loss,
        &steps,
    ));

    try std.testing.expectEqual(status(.ok), zgml_train_linear_mse_sgd_f32_bulk(
        dataset_input[0..].ptr,
        dataset_input.len,
        dataset_target[0..].ptr,
        dataset_target.len,
        indices[0..].ptr,
        indices.len,
        batch_input[0..].ptr,
        batch_input.len,
        batch_target[0..].ptr,
        batch_target.len,
        weight[0..].ptr,
        weight.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        grad_weight[0..].ptr,
        grad_weight.len,
        indices.len,
        2,
        2,
        1,
        80,
        0.04,
        0,
        &train_loss,
        &steps,
    ));

    try std.testing.expectEqual(@as(usize, 160), steps);
    var heldout = [_]f32{0};
    const heldout_input = [_]f32{ 1, -1 };
    try std.testing.expectEqual(status(.ok), zgml_eager_linear_f32(
        heldout_input[0..].ptr,
        heldout_input.len,
        weight[0..].ptr,
        weight.len,
        bias[0..].ptr,
        bias.len,
        heldout[0..].ptr,
        heldout.len,
        1,
        2,
        1,
    ));
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), heldout[0], 0.02);
    try std.testing.expect(train_loss < 0.001);
}

test "C ABI native eager linear activation writes caller output" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const weights = [_]f32{
        1,    -1,
        0,    2,
        -0.5, 0.25,
    };
    const bias = [_]f32{ 0.5, -1 };
    var output = [_]f32{0} ** 4;
    var linear = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_linear_activation_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
        module_activation_gelu,
    ));
    try std.testing.expectEqual(status(.ok), zgml_eager_linear_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        linear[0..].ptr,
        linear.len,
        2,
        3,
        2,
    ));
    for (linear, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_gelu), activated, 1e-6);
    }
    try std.testing.expectEqual(status(.ok), zgml_eager_linear_activation_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
        module_activation_sigmoid,
    ));
    for (linear, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_sigmoid), activated, 1e-6);
    }
    try std.testing.expectEqual(status(.ok), zgml_eager_linear_activation_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
        module_activation_tanh,
    ));
    for (linear, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_tanh), activated, 1e-6);
    }
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_linear_activation_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        null,
        0,
        output[0..].ptr,
        output.len,
        2,
        3,
        2,
        99,
    ));
}

test "C ABI native eager activation writes caller output" {
    const input = [_]f32{ -2, -0.5, 0, 0.5, 2 };
    var output = [_]f32{0} ** input.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        module_activation_relu,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0.5, 2 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        module_activation_tanh,
    ));
    for (input, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_tanh), activated, 1e-6);
    }

    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len - 1,
        output[0..].ptr,
        output.len,
        module_activation_relu,
    ));
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        99,
    ));
}

test "C ABI native eager activation vectorizes gelu and tanh" {
    const input = [_]f32{ -2.0, -1.036, -0.5, -0.173, 0.0, 0.173, 0.5, 2.0 };
    var output = [_]f32{0} ** input.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        module_activation_gelu,
    ));
    for (input, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_gelu), activated, 2e-6);
    }

    try std.testing.expectEqual(status(.ok), zgml_eager_activation_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        module_activation_tanh,
    ));
    for (input, output) |plain, activated| {
        try std.testing.expectApproxEqAbs(try eagerActivationF32(plain, module_activation_tanh), activated, 2e-6);
    }
}

test "C ABI native eager elementwise writes caller output" {
    const lhs = [_]f32{ -2, -0.5, 0.5, 2 };
    const rhs = [_]f32{ 2, 4, -1, 0.5 };
    const scalar = [_]f32{2};
    var output = [_]f32{0} ** lhs.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_add,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 0, 3.5, -0.5, 2.5 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        scalar[0..].ptr,
        scalar.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_mul,
    ));
    try std.testing.expectEqualSlices(f32, &.{ -4, -1, 1, 4 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        null,
        0,
        output[0..].ptr,
        output.len,
        eager_elementwise_sqr,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 4, 0.25, 0.25, 4 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_lt,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 0 }, &output);

    const nan_lhs = [_]f32{ std.math.nan(f32), -0.0, 3 };
    const nan_rhs = [_]f32{ std.math.nan(f32), 0.0, 4 };
    var compare_output = [_]f32{0} ** nan_lhs.len;
    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        nan_lhs[0..].ptr,
        nan_lhs.len,
        nan_rhs[0..].ptr,
        nan_rhs.len,
        compare_output[0..].ptr,
        compare_output.len,
        eager_elementwise_eq,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0 }, &compare_output);

    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        null,
        0,
        output[0..].ptr,
        output.len,
        eager_elementwise_add,
    ));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len - 1,
        output[0..].ptr,
        output.len,
        eager_elementwise_add,
    ));
}

test "C ABI native eager elementwise vectorizes scalar and same-shape binary ops" {
    const lhs = [_]f32{ -4, -3, -2, -1, -0.0, 0.0, 1, 2, 3, 4, std.math.nan(f32), 6, 7, 8, 9, 10 };
    const rhs = [_]f32{ -5, -2, -2, 0, 0.0, -0.0, 2, 1, 3, 5, std.math.nan(f32), 6, 8, 7, 10, 9 };
    const scalar = [_]f32{3};
    var output = [_]f32{0} ** lhs.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        scalar[0..].ptr,
        scalar.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_mul,
    ));
    for (lhs, output) |plain, scaled| {
        if (std.math.isNan(plain)) try std.testing.expect(std.math.isNan(scaled)) else try std.testing.expectEqual(plain * 3, scaled);
    }

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        scalar[0..].ptr,
        scalar.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_lt,
    ));
    for (lhs, output) |plain, compared| try std.testing.expectEqual(if (plain < 3) @as(f32, 1) else 0, compared);

    try std.testing.expectEqual(status(.ok), zgml_eager_elementwise_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
        eager_elementwise_eq,
    ));
    for (lhs, rhs, output) |left, right, compared| {
        const expected: f32 = if (left == right or (std.math.isNan(left) and std.math.isNan(right))) 1 else 0;
        try std.testing.expectEqual(expected, compared);
    }
}

test "C ABI native eager where writes caller output" {
    const condition = [_]f32{ 1, 0, -2, 0 };
    const input = [_]f32{ 10, 20, 30, 40 };
    const scalar_other = [_]f32{-1};
    var output = [_]f32{0} ** condition.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_where_f32(
        condition[0..].ptr,
        condition.len,
        input[0..].ptr,
        input.len,
        scalar_other[0..].ptr,
        scalar_other.len,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 10, -1, 30, -1 }, &output);

    const other = [_]f32{ 5, 6, 7, 8 };
    const scalar_input = [_]f32{99};
    try std.testing.expectEqual(status(.ok), zgml_eager_where_f32(
        condition[0..].ptr,
        condition.len,
        scalar_input[0..].ptr,
        scalar_input.len,
        other[0..].ptr,
        other.len,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 99, 6, 99, 8 }, &output);

    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_where_f32(
        null,
        condition.len,
        input[0..].ptr,
        input.len,
        scalar_other[0..].ptr,
        scalar_other.len,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_where_f32(
        condition[0..].ptr,
        condition.len,
        input[0..].ptr,
        input.len - 1,
        scalar_other[0..].ptr,
        scalar_other.len,
        output[0..].ptr,
        output.len,
    ));
}

test "C ABI native eager clamp writes caller output" {
    const input = [_]f32{ -2, -0.5, 0.5, 2 };
    var output = [_]f32{0} ** input.len;

    try std.testing.expectEqual(status(.ok), zgml_eager_clamp_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        -1,
        1,
        1,
        1,
    ));
    try std.testing.expectEqualSlices(f32, &.{ -1, -0.5, 0.5, 1 }, &output);

    try std.testing.expectEqual(status(.ok), zgml_eager_clamp_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        0,
        0,
        1,
        0,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0.5, 2 }, &output);

    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_clamp_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        0,
        1,
        0,
        0,
    ));
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_clamp_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        2,
        1,
        1,
        1,
    ));
}

test "C ABI native eager reduce writes scalar output" {
    const input = [_]f32{ -2, 4, 0.5, 3 };
    var output = [_]f32{0};

    try std.testing.expectEqual(status(.ok), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        eager_reduce_sum,
    ));
    try std.testing.expectEqual(@as(f32, 5.5), output[0]);

    try std.testing.expectEqual(status(.ok), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        eager_reduce_mean,
    ));
    try std.testing.expectEqual(@as(f32, 1.375), output[0]);

    try std.testing.expectEqual(status(.ok), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        eager_reduce_max,
    ));
    try std.testing.expectEqual(@as(f32, 4), output[0]);

    try std.testing.expectEqual(status(.ok), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        eager_reduce_min,
    ));
    try std.testing.expectEqual(@as(f32, -2), output[0]);

    try std.testing.expectEqual(status(.ok), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        eager_reduce_prod,
    ));
    try std.testing.expectEqual(@as(f32, -12), output[0]);

    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_reduce_f32(
        input[0..].ptr,
        0,
        output[0..].ptr,
        output.len,
        eager_reduce_sum,
    ));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        0,
        eager_reduce_sum,
    ));
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_reduce_f32(
        input[0..].ptr,
        input.len,
        output[0..].ptr,
        output.len,
        999,
    ));
}

test "C ABI native eager dot writes scalar output" {
    const lhs = [_]f32{ -2, 4, 0.5, 3, 1, -1, 2, -3, 0.25 };
    const rhs = [_]f32{ 2, -1, 4, 0.5, 3, 5, -2, -1, 8 };
    var output = [_]f32{0};

    try std.testing.expectEqual(status(.ok), zgml_eager_dot_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqual(@as(f32, -5.5), output[0]);

    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_dot_f32(
        null,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_dot_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len - 1,
        output[0..].ptr,
        output.len,
    ));
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_dot_f32(
        lhs[0..].ptr,
        lhs.len,
        rhs[0..].ptr,
        rhs.len,
        output[0..].ptr,
        0,
    ));
}

test "C ABI native eager conv2d writes caller output" {
    const input = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
    };
    const weights = [_]f32{
        1,  0,
        0,  1,
        -1, 1,
        1,  -1,
    };
    const bias = [_]f32{0.5};
    var output = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_conv2d_f32(
        input[0..].ptr,
        input.len,
        weights[0..].ptr,
        weights.len,
        bias[0..].ptr,
        bias.len,
        output[0..].ptr,
        output.len,
        1,
        2,
        3,
        3,
        1,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 6.5, 8.5, 12.5, 14.5 }, &output);

    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_conv2d_f32(
        input[0..].ptr,
        input.len - 1,
        weights[0..].ptr,
        weights.len,
        null,
        0,
        output[0..].ptr,
        output.len,
        1,
        2,
        3,
        3,
        1,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
    ));
}

test "C ABI native eager pool2d writes caller output" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    var max_output = [_]f32{0} ** 4;
    var avg_output = [_]f32{0} ** 4;

    try std.testing.expectEqual(status(.ok), zgml_eager_pool2d_f32(
        input[0..].ptr,
        input.len,
        max_output[0..].ptr,
        max_output.len,
        1,
        1,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        eager_pool_max,
        0,
        1,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 8, 9 }, &max_output);

    try std.testing.expectEqual(status(.ok), zgml_eager_pool2d_f32(
        input[0..].ptr,
        input.len,
        avg_output[0..].ptr,
        avg_output.len,
        1,
        1,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        eager_pool_avg,
        0,
        1,
    ));
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 6, 7 }, &avg_output);

    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_pool2d_f32(
        input[0..].ptr,
        input.len - 1,
        max_output[0..].ptr,
        max_output.len,
        1,
        1,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        eager_pool_max,
        0,
        1,
    ));
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_pool2d_f32(
        input[0..].ptr,
        input.len,
        max_output[0..].ptr,
        max_output.len,
        1,
        1,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        999,
        0,
        1,
    ));
}

test "C ABI native eager row softmax writes caller output" {
    const input = [_]f32{
        1, 2, 3,
        3, 1, -1,
    };
    var softmax_output = [_]f32{0} ** 6;
    var log_softmax_output = [_]f32{0} ** 6;

    try std.testing.expectEqual(status(.ok), zgml_eager_softmax_f32(
        input[0..].ptr,
        input.len,
        softmax_output[0..].ptr,
        softmax_output.len,
        2,
        3,
        0,
    ));
    try std.testing.expectEqual(status(.ok), zgml_eager_softmax_f32(
        input[0..].ptr,
        input.len,
        log_softmax_output[0..].ptr,
        log_softmax_output.len,
        2,
        3,
        1,
    ));

    for (0..2) |row| {
        const input_row = input[row * 3 ..][0..3];
        var max_val = -std.math.inf(f32);
        for (input_row) |value| max_val = @max(max_val, value);
        var sum_exp: f32 = 0;
        for (input_row) |value| sum_exp += @exp(value - max_val);
        const log_denom = max_val + @log(sum_exp);
        for (input_row, 0..) |value, col| {
            try std.testing.expectApproxEqAbs(@exp(value - max_val) / sum_exp, softmax_output[row * 3 + col], 1e-6);
            try std.testing.expectApproxEqAbs(value - log_denom, log_softmax_output[row * 3 + col], 1e-6);
        }
    }
    try std.testing.expectEqual(status(.shape_mismatch), zgml_eager_softmax_f32(
        input[0..].ptr,
        input.len - 1,
        softmax_output[0..].ptr,
        softmax_output.len,
        2,
        3,
        0,
    ));
    try std.testing.expectEqual(status(.invalid_argument), zgml_eager_softmax_f32(
        input[0..].ptr,
        input.len,
        softmax_output[0..].ptr,
        softmax_output.len,
        2,
        3,
        2,
    ));
}

test "C ABI buffers can wrap caller-owned host memory" {
    var weights = [_]f32{ 1, 0, 0, 1 };
    var input = [_]f32{ 2, 3 };
    var output = [_]f32{ -999, -999, -777 };
    var bad_buffer: ?*zgml_buffer = @ptrFromInt(1);
    var weights_buffer: ?*zgml_buffer = null;
    var input_buffer: ?*zgml_buffer = null;
    var output_buffer: ?*zgml_buffer = null;
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);
    defer zgml_buffer_free(output_buffer);
    defer zgml_buffer_free(input_buffer);
    defer zgml_buffer_free(weights_buffer);

    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap(null, @sizeOf(@TypeOf(weights)), &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);
    bad_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap(&weights, 0, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap(&weights, @sizeOf(@TypeOf(weights)), &weights_buffer));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap(&input, @sizeOf(@TypeOf(input)), &input_buffer));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap(&output, @sizeOf(@TypeOf(output)), &output_buffer));
    try std.testing.expectEqual(@as(usize, @sizeOf(@TypeOf(output))), zgml_buffer_size(output_buffer));
    try std.testing.expectEqual(@intFromPtr(&output), @intFromPtr(zgml_buffer_data(output_buffer).?));

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(program, &.{
        .weights = weights_buffer,
        .weights_len = 4,
        .input = input_buffer,
        .input_len = 2,
        .output = output_buffer,
        .output_len = 3,
    }, &session));

    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, output[0..2]);
    try std.testing.expectEqual(@as(f32, -777), output[2]);

    const new_input = [_]f32{ 4, 5 };
    try std.testing.expectEqual(status(.ok), zgml_buffer_write(input_buffer, 0, &new_input, @sizeOf(@TypeOf(new_input))));
    output = [_]f32{ -111, -111, -222 };
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, output[0..2]);
    try std.testing.expectEqual(@as(f32, -222), output[2]);
}

test "C ABI external resource buffers are explicit and rejected by host-only sessions" {
    var bad_buffer: ?*zgml_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap_resource(null, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    bad_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .handle = 0,
        .byte_len = 4 * @sizeOf(f32),
    }, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    bad_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap_resource(&.{
        .placement = 999,
        .handle = 7,
        .byte_len = 4 * @sizeOf(f32),
    }, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    bad_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = 1 << 8,
        .handle = 7,
        .byte_len = 4 * @sizeOf(f32),
    }, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    bad_buffer = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .handle = 7,
        .byte_offset = std.math.maxInt(usize),
        .byte_len = 4,
    }, &bad_buffer));
    try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_buffer);

    var weights_resource: ?*zgml_buffer = null;
    var output_resource: ?*zgml_buffer = null;
    var host_input: ?*zgml_buffer = null;
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);
    defer zgml_buffer_free(host_input);
    defer zgml_buffer_free(output_resource);
    defer zgml_buffer_free(weights_resource);

    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_read_write,
        .handle = 7,
        .byte_offset = 16,
        .byte_len = 4 * @sizeOf(f32),
    }, &weights_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_write,
        .handle = 8,
        .byte_len = 3 * @sizeOf(f32),
    }, &output_resource));
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), zgml_buffer_size(weights_resource));
    try std.testing.expect(zgml_buffer_data(weights_resource) == null);

    const weights = [_]f32{ 1, 0, 0, 1 };
    var readback = [_]f32{0} ** 4;
    try std.testing.expectEqual(status(.unsupported), zgml_buffer_write(weights_resource, 0, &weights, @sizeOf(@TypeOf(weights))));
    try std.testing.expectEqual(status(.unsupported), zgml_buffer_read(weights_resource, 0, &readback, @sizeOf(@TypeOf(readback))));

    var input = [_]f32{ 2, 3 };
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap(&input, @sizeOf(@TypeOf(input)), &host_input));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_session_bind_buffers(program, &.{
        .weights = weights_resource,
        .weights_len = 4,
        .input = host_input,
        .input_len = 2,
        .output = output_resource,
        .output_len = 3,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);
}

test "C ABI buffer inspection exposes host and external-resource shape" {
    var host_buffer: ?*zgml_buffer = null;
    var resource_buffer: ?*zgml_buffer = null;
    defer zgml_buffer_free(resource_buffer);
    defer zgml_buffer_free(host_buffer);

    var inspection = zgml_buffer_inspection{ .storage = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_inspect(null, &inspection));
    try std.testing.expectEqual(status(.invalid_argument), zgml_buffer_inspect(host_buffer, null));
    try std.testing.expectEqual(@as(u32, 99), inspection.storage);

    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = 64 }, &host_buffer));
    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(host_buffer, &inspection));
    try std.testing.expectEqual(buffer_storage_host, inspection.storage);
    try std.testing.expectEqual(@as(u32, 0), inspection.placement);
    try std.testing.expectEqual(@as(u32, 0), inspection.access_flags);
    try std.testing.expectEqual(@as(u64, 64), inspection.byte_len);
    try std.testing.expectEqual(@as(u64, 0), inspection.handle);
    try std.testing.expectEqual(@as(u64, 0), inspection.byte_offset);
    try std.testing.expectEqual(@as(u64, 0), inspection.resource_byte_len);

    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_write,
        .handle = 1234,
        .byte_offset = 16,
        .byte_len = 32,
    }, &resource_buffer));
    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(resource_buffer, &inspection));
    try std.testing.expectEqual(buffer_storage_external_resource, inspection.storage);
    try std.testing.expectEqual(backend_webgpu, inspection.placement);
    try std.testing.expectEqual(resource_access_write, inspection.access_flags);
    try std.testing.expectEqual(@as(u64, 32), inspection.byte_len);
    try std.testing.expectEqual(@as(u64, 1234), inspection.handle);
    try std.testing.expectEqual(@as(u64, 16), inspection.byte_offset);
    try std.testing.expectEqual(@as(u64, 48), inspection.resource_byte_len);
}

test "C ABI llama external resource output binding remains unsupported on host backends" {
    var model: ?*zgml_model = null;
    var compatible_model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var output_resource: ?*zgml_buffer = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_buffer_free(output_resource);
    defer zgml_program_free(program);
    defer zgml_model_free(compatible_model);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &compatible_model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .handle = 9,
        .byte_offset = 32,
        .byte_len = 32 + tiny_llama_config.vocab_size * @sizeOf(f32),
    }, &output_resource));

    try std.testing.expectEqual(status(.unsupported), zgml_session_bind_buffers(program, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);

    session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_session_bind_model_buffers(program, compatible_model, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);
}

test "C ABI llama KV resource binding remains unsupported on host backends" {
    var model: ?*zgml_model = null;
    var compatible_model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var k_resource: ?*zgml_buffer = null;
    var v_resource: ?*zgml_buffer = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_buffer_free(v_resource);
    defer zgml_buffer_free(k_resource);
    defer zgml_program_free(program);
    defer zgml_model_free(compatible_model);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &compatible_model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu, .context_len = 4 }, &program));

    const cache_bytes = (try llamaKvCacheElementCount(tiny_llama_config, 4)) * @sizeOf(f32);
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .handle = 17,
        .byte_len = cache_bytes,
    }, &k_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .handle = 18,
        .byte_len = cache_bytes,
    }, &v_resource));

    var k_buffers = [_]?*zgml_buffer{k_resource};
    var v_buffers = [_]?*zgml_buffer{v_resource};
    const kv = zgml_llama_kv_cache_bind_desc{
        .k = k_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = k_buffers.len,
    };

    session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_llama_session_bind_buffers(program, &.{
        .kv_cache = &kv,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);

    session = @ptrFromInt(1);
    const model_resource_status = zgml_llama_session_bind_model_buffers(program, compatible_model, &.{
        .kv_cache = &kv,
    }, &session);
    try std.testing.expect(model_resource_status == status(.unsupported) or model_resource_status == status(.shape_mismatch));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);

    const wrong_len = zgml_llama_kv_cache_bind_desc{
        .k = k_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = k_buffers.len + 1,
    };
    session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_llama_session_bind_buffers(program, &.{
        .kv_cache = &wrong_len,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);
}

test "C ABI tiny linear can bind caller-owned step buffers" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{ .backend = backend_cpu }, &program));

    const weights = [_]f32{ 1, 0, 0, 1 };
    var input = [_]f32{ 2, 3 };
    var output = [_]f32{-999} ** 3;
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = weights[0..].ptr,
        .weights_len = weights.len,
        .input = input[0..].ptr,
        .input_len = input.len,
        .output = output[0..].ptr,
        .output_len = output.len,
    }, &session));

    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqual(@as(usize, 2), result.output_len);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, output[0..2]);
    try std.testing.expectEqual(@as(f32, -999), output[2]);

    input = .{ 4, 5 };
    output = .{ -111, -111, -111 };
    try std.testing.expectEqual(status(.ok), zgml_session_reset(session));
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, null, &result));
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, output[0..2]);
    try std.testing.expectEqual(@as(f32, -111), output[2]);

    const override_input = [_]f32{ 7, 8 };
    var override_output = [_]f32{-222} ** 3;
    try std.testing.expectEqual(status(.ok), zgml_session_step(session, &.{
        .input = override_input[0..].ptr,
        .input_len = override_input.len,
        .output = override_output[0..].ptr,
        .output_len = override_output.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, &.{ 7, 8 }, override_output[0..2]);
    try std.testing.expectEqual(@as(f32, -222), override_output[2]);
}

test "C ABI tiny llama handles compile bind token step into caller logits" {
    var model: ?*zgml_model = null;
    var compatible_model: ?*zgml_model = null;
    var wrong_model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session_a: ?*zgml_session = null;
    var session_b: ?*zgml_session = null;
    var session_model_bound: ?*zgml_session = null;
    var session_bound_output: ?*zgml_session = null;
    var session_prefill: ?*zgml_session = null;
    var session_ref: ?*zgml_session = null;
    var session_bulk: ?*zgml_session = null;
    var session_bulk_expected: ?*zgml_session = null;
    var session_execute: ?*zgml_session = null;
    var session_program_output: ?*zgml_session = null;
    var session_host_kv: ?*zgml_session = null;
    var program_output_buffer: ?*zgml_buffer = null;
    var host_k_cache: ?*zgml_buffer = null;
    var host_v_cache: ?*zgml_buffer = null;
    defer zgml_buffer_free(host_v_cache);
    defer zgml_buffer_free(host_k_cache);
    defer zgml_buffer_free(program_output_buffer);
    defer zgml_session_free(session_host_kv);
    defer zgml_session_free(session_program_output);
    defer zgml_session_free(session_execute);
    defer zgml_session_free(session_bulk_expected);
    defer zgml_session_free(session_bulk);
    defer zgml_session_free(session_ref);
    defer zgml_session_free(session_prefill);
    defer zgml_session_free(session_bound_output);
    defer zgml_session_free(session_model_bound);
    defer zgml_session_free(session_b);
    defer zgml_session_free(session_a);
    defer zgml_program_free(program);
    defer zgml_model_free(wrong_model);
    defer zgml_model_free(compatible_model);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_cpu,
        .context_len = 4,
        .batch = 1,
    }, &program));

    var inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &inspection));
    try std.testing.expect(inspection.command_count > 0);
    try std.testing.expectEqual(@as(u64, backend_cpu), inspection.backend);
    try std.testing.expectEqual(@as(u64, 1), inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 0), inspection.external_resources_supported);
    try std.testing.expect(inspection.buffer_count > 0);
    try std.testing.expect(inspection.buffer_byte_len > 0);
    try std.testing.expect(inspection.command_projection_count > 0 or inspection.command_attention_count > 0);
    try std.testing.expect(inspection.command_stencil_hash != 0);
    try std.testing.expect(inspection.binding_requirement_hash != 0);
    try std.testing.expect(inspection.persistent_requirement_count > 0);
    try std.testing.expect(inspection.step_input_requirement_count > 0);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_output_requirement_count);
    try std.testing.expect(inspection.runtime_patch_holes > 0);
    try std.testing.expect(inspection.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(u64, 3), inspection.runtime_patch_max_cache_write_pos);
    try std.testing.expectEqual(@as(u64, 4), inspection.runtime_patch_max_attention_seq_kv);
    try std.testing.expectEqual(@as(u64, 0), inspection.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_supported);
    try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_covered_op_count);
    try std.testing.expectEqual(std.math.maxInt(u64), inspection.dispatch_plan_first_unsupported_op);
    try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_projection_count);
    try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_attention_count);

    var llama_inspection = zgml_llama_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_llama_program_inspect(program, &llama_inspection));
    try std.testing.expectEqual(@as(u64, tiny_llama_config.vocab_size), llama_inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.max_seq_len), llama_inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, 4), llama_inspection.context_len);
    try std.testing.expectEqual(@as(u64, 1), llama_inspection.batch);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_layers), llama_inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_heads), llama_inspection.n_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_kv_heads), llama_inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, 1), llama_inspection.semantic_token_count);
    try std.testing.expect(llama_inspection.semantic_stage_count > 0);
    try std.testing.expect(llama_inspection.semantic_runtime_patch_holes > 0);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_holes, inspection.runtime_patch_holes);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_cache_write_pos_holes, inspection.runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_attention_seq_kv_holes, inspection.runtime_patch_attention_seq_kv_holes);

    var requirements = zgml_program_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(program, &requirements));
    try std.testing.expectEqual(tiny_llama_kind, requirements.model_kind);
    try std.testing.expectEqual(@as(u32, @sizeOf(f32)), requirements.scalar_bytes);
    try std.testing.expectEqual(@as(u32, @sizeOf(u32)), requirements.token_id_bytes);
    try std.testing.expectEqual(@as(usize, 0), requirements.input_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.input_byte_len);
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), requirements.output_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.weights_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.weights_byte_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.bias_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.bias_byte_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.parameter_len);
    try std.testing.expectEqual(@as(usize, 0), requirements.parameter_byte_len);
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), requirements.logits_len);
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size * @sizeOf(f32)), requirements.output_byte_len);
    try std.testing.expectEqual(@as(usize, 4), requirements.context_len);
    try std.testing.expectEqual(@as(usize, 1), requirements.batch);
    try std.testing.expectEqual(@as(usize, 4), requirements.max_token_window);

    var kv_requirements = zgml_llama_kv_cache_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_llama_program_get_kv_cache_requirements(program, &kv_requirements));
    try std.testing.expectEqual(tiny_llama_kind, kv_requirements.model_kind);
    try std.testing.expectEqual(@as(u32, @sizeOf(f32)), kv_requirements.scalar_bytes);
    try std.testing.expectEqual(@as(u32, @intCast(tiny_llama_config.n_layers)), kv_requirements.n_layers);
    try std.testing.expectEqual(@as(usize, 4), kv_requirements.context_len);
    try std.testing.expectEqual(@as(usize, 64), kv_requirements.k_buffer_byte_len);
    try std.testing.expectEqual(@as(usize, 64), kv_requirements.v_buffer_byte_len);
    try std.testing.expectEqual(@as(usize, 64), kv_requirements.buffer_byte_len);

    try std.testing.expectEqual(status(.ok), zgml_program_create_output_buffer(program, &program_output_buffer));
    try std.testing.expectEqual(requirements.output_byte_len, zgml_buffer_size(program_output_buffer));

    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = kv_requirements.k_buffer_byte_len }, &host_k_cache));
    try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = kv_requirements.v_buffer_byte_len }, &host_v_cache));
    var host_k_buffers = [_]?*zgml_buffer{host_k_cache};
    var host_v_buffers = [_]?*zgml_buffer{host_v_cache};
    const host_kv = zgml_llama_kv_cache_bind_desc{
        .k = host_k_buffers[0..].ptr,
        .v = host_v_buffers[0..].ptr,
        .len = host_k_buffers.len,
    };
    try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{ .kv_cache = &host_kv }, &session_host_kv));
    var host_kv_session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session_host_kv, &host_kv_session_inspection));
    try std.testing.expectEqual(tiny_llama_kind, host_kv_session_inspection.model_kind);
    try std.testing.expectEqual(backend_cpu, host_kv_session_inspection.backend);
    try std.testing.expectEqual(buffer_storage_host, host_kv_session_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_host, host_kv_session_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), host_kv_session_inspection.position);
    try std.testing.expectEqual(@as(u64, 4), host_kv_session_inspection.context_len);
    try std.testing.expect(host_kv_session_inspection.persistent_binding_count > 0);
    try std.testing.expect(host_kv_session_inspection.step_input_count > 0);
    try std.testing.expect(host_kv_session_inspection.step_output_count > 0);
    try std.testing.expect(host_kv_session_inspection.host_binding_count > 0);
    try std.testing.expectEqual(@as(u64, 0), host_kv_session_inspection.resource_binding_count);
    try std.testing.expect(host_kv_session_inspection.binding_shape_hash != 0);
    var host_kv_logits = [_]f32{-444} ** (tiny_llama_config.vocab_size + 1);
    var host_kv_result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_host_kv, &.{
        .token = 1,
        .output = host_kv_logits[0..].ptr,
        .output_len = host_kv_logits.len,
    }, &host_kv_result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), host_kv_result.output_len);
    try std.testing.expectEqual(@as(f32, -444), host_kv_logits[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_a));
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_b));
    var session_a_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session_a, &session_a_inspection));
    try std.testing.expectEqual(tiny_llama_kind, session_a_inspection.model_kind);
    try std.testing.expectEqual(buffer_storage_host, session_a_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_host, session_a_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), session_a_inspection.position);
    try std.testing.expectEqual(@as(u64, 4), session_a_inspection.context_len);
    try std.testing.expect(session_a_inspection.host_binding_count > 0);
    try std.testing.expectEqual(@as(u64, 0), session_a_inspection.resource_binding_count);
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &compatible_model));
    try std.testing.expectEqual(status(.ok), zgml_session_bind_model(program, compatible_model, null, &session_model_bound));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 1,
        .output_len = 1,
    }, &wrong_model));
    var rejected_model_session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_bind_model(program, wrong_model, null, &rejected_model_session));
    try std.testing.expectEqual(@as(?*zgml_session, null), rejected_model_session);
    var bound_logits = [_]f32{-333} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, &.{
        .weights = null,
        .weights_len = 0,
        .output = bound_logits[0..].ptr,
        .output_len = bound_logits.len,
    }, &session_bound_output));
    try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(program, &.{
        .output = program_output_buffer,
        .output_len = tiny_llama_config.vocab_size,
    }, &session_program_output));

    var pos: usize = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_a, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    var argmax = zgml_token_argmax_result{ .token = 999, .logit = -999 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_argmax_token(session_b, null, &argmax));
    try std.testing.expectEqual(@as(u32, 0), argmax.token);
    try std.testing.expectEqual(@as(f32, 0), argmax.logit);
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_execute_argmax_tokens(session_b, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
    }, &argmax));
    try std.testing.expectEqual(@as(u32, 0), argmax.token);
    try std.testing.expectEqual(@as(f32, 0), argmax.logit);
    var sample = zgml_token_sample_result{ .token = 999, .logit = -999 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_sample_token(session_b, null, &sample));
    try std.testing.expectEqual(@as(u32, 0), sample.token);
    try std.testing.expectEqual(@as(f32, 0), sample.logit);
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_execute_sample_tokens(session_b, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &sample));
    try std.testing.expectEqual(@as(u32, 0), sample.token);
    try std.testing.expectEqual(@as(f32, 0), sample.logit);

    var too_small = [_]f32{0} ** (tiny_llama_config.vocab_size - 1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_step_token(session_a, &.{
        .token = 0,
        .output = too_small[0..].ptr,
        .output_len = too_small.len,
    }, null));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_a, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    var logits_a = [_]f32{-999} ** (tiny_llama_config.vocab_size + 1);
    var logits_b = [_]f32{-777} ** (tiny_llama_config.vocab_size + 1);
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_a, &.{
        .token = 0,
        .output = logits_a[0..].ptr,
        .output_len = logits_a.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_a, &pos));
    try std.testing.expectEqual(@as(usize, 1), pos);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_b, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    var logits_model_bound = [_]f32{-444} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_model_bound, &.{
        .token = 0,
        .output = logits_model_bound[0..].ptr,
        .output_len = logits_model_bound.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqualSlices(f32, logits_a[0..tiny_llama_config.vocab_size], logits_model_bound[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_model_bound, &pos));
    try std.testing.expectEqual(@as(usize, 1), pos);

    var caller_argmax = zgml_token_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_argmax_token(session_a, &.{
        .logits = logits_a[0..].ptr,
        .logits_len = tiny_llama_config.vocab_size,
    }, &caller_argmax));
    const caller_argmax_index: usize = caller_argmax.token;
    try std.testing.expect(caller_argmax_index < tiny_llama_config.vocab_size);
    try std.testing.expectEqual(logits_a[caller_argmax_index], caller_argmax.logit);
    for (logits_a[0..tiny_llama_config.vocab_size]) |logit| {
        try std.testing.expect(logit <= caller_argmax.logit);
    }

    var fake_logits = [_]f32{-100} ** tiny_llama_config.vocab_size;
    fake_logits[2] = 3.0;
    fake_logits[5] = 2.5;
    var caller_sample = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_sample_token(session_a, &.{
        .logits = fake_logits[0..].ptr,
        .logits_len = fake_logits.len,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &caller_sample));
    try std.testing.expectEqual(@as(u32, 2), caller_sample.token);
    try std.testing.expectEqual(@as(f32, 3.0), caller_sample.logit);
    try std.testing.expectEqual(status(.ok), zgml_session_sample_token(session_a, &.{
        .logits = fake_logits[0..].ptr,
        .logits_len = fake_logits.len,
        .top_k = 2,
        .seed = 1,
        .temperature = 0.7,
    }, &caller_sample));
    try std.testing.expect(caller_sample.token == 2 or caller_sample.token == 5);
    try std.testing.expectEqual(fake_logits[@intCast(caller_sample.token)], caller_sample.logit);
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_sample_token(session_a, &.{
        .logits = fake_logits[0..].ptr,
        .logits_len = fake_logits.len,
        .top_k = 0,
        .seed = 123,
        .temperature = 1.0,
    }, &caller_sample));
    try std.testing.expectEqual(@as(u32, 0), caller_sample.token);
    try std.testing.expectEqual(@as(f32, 0), caller_sample.logit);
    try std.testing.expectEqual(status(.invalid_argument), zgml_session_sample_token(session_a, &.{
        .logits = fake_logits[0..].ptr,
        .logits_len = fake_logits.len,
        .top_k = 1,
        .seed = 123,
        .temperature = 0,
    }, &caller_sample));
    try std.testing.expectEqual(@as(u32, 0), caller_sample.token);
    try std.testing.expectEqual(@as(f32, 0), caller_sample.logit);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_a));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_a, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    var logits_reset = [_]f32{-888} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_a, &.{
        .token = 0,
        .output = logits_reset[0..].ptr,
        .output_len = logits_reset.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, logits_a[0..tiny_llama_config.vocab_size], logits_reset[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -888), logits_reset[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_bound_output, &.{
        .token = 0,
        .output = null,
        .output_len = 0,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqualSlices(f32, logits_a[0..tiny_llama_config.vocab_size], bound_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -333), bound_logits[tiny_llama_config.vocab_size]);
    var bound_argmax = zgml_token_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_argmax_token(session_bound_output, null, &bound_argmax));
    try std.testing.expectEqual(caller_argmax.token, bound_argmax.token);
    try std.testing.expectEqual(caller_argmax.logit, bound_argmax.logit);
    var bound_sample = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_sample_token(session_bound_output, &.{
        .logits = null,
        .logits_len = 0,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &bound_sample));
    try std.testing.expectEqual(caller_argmax.token, bound_sample.token);
    try std.testing.expectEqual(caller_argmax.logit, bound_sample.logit);
    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_bound_output));
    bound_logits = [_]f32{-222} ** (tiny_llama_config.vocab_size + 1);
    var execute_argmax = zgml_token_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_argmax_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
    }, &execute_argmax));
    try std.testing.expectEqual(caller_argmax.token, execute_argmax.token);
    try std.testing.expectEqual(caller_argmax.logit, execute_argmax.logit);
    try std.testing.expectEqual(@as(f32, -222), bound_logits[tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_bound_output, &pos));
    try std.testing.expectEqual(@as(usize, 1), pos);
    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_bound_output));
    bound_logits = [_]f32{-111} ** (tiny_llama_config.vocab_size + 1);
    var execute_sample = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_sample_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &execute_sample));
    try std.testing.expectEqual(caller_argmax.token, execute_sample.token);
    try std.testing.expectEqual(caller_argmax.logit, execute_sample.logit);
    try std.testing.expectEqual(@as(f32, -111), bound_logits[tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_bound_output, &pos));
    try std.testing.expectEqual(@as(usize, 1), pos);

    var program_output_sample = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_sample_tokens(session_program_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &program_output_sample));
    try std.testing.expectEqual(caller_argmax.token, program_output_sample.token);
    try std.testing.expectEqual(caller_argmax.logit, program_output_sample.logit);
    var program_output_logits = [_]f32{0} ** tiny_llama_config.vocab_size;
    try std.testing.expectEqual(status(.ok), zgml_buffer_read(program_output_buffer, 0, &program_output_logits, @sizeOf(@TypeOf(program_output_logits))));
    try std.testing.expectEqualSlices(f32, logits_a[0..tiny_llama_config.vocab_size], program_output_logits[0..]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_program_output, &pos));
    try std.testing.expectEqual(@as(usize, 1), pos);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_program_output));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_program_output, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    var invalid_generated = [_]u32{999};
    var invalid_generate = zgml_token_generate_sample_result{ .tokens_generated = 99, .last_token = 99, .last_logit = -99 };
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_generate_sample_tokens(session_program_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .output_tokens = invalid_generated[0..].ptr,
        .output_tokens_len = 0,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &invalid_generate));
    try std.testing.expectEqual(@as(usize, 0), invalid_generate.tokens_generated);
    try std.testing.expectEqual(@as(u32, 0), invalid_generate.last_token);
    try std.testing.expectEqual(@as(f32, 0), invalid_generate.last_logit);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_program_output, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_bound_output));
    var expected_first = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_sample_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &expected_first));
    var expected_second = zgml_token_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_sample_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{expected_first.token})[0..].ptr,
        .tokens_len = 1,
        .top_k = 1,
        .seed = 124,
        .temperature = 1.0,
    }, &expected_second));
    var generated = [_]u32{ 777, 777 };
    var generate_result = zgml_token_generate_sample_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_generate_sample_tokens(session_program_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .output_tokens = generated[0..].ptr,
        .output_tokens_len = generated.len,
        .top_k = 1,
        .seed = 123,
        .temperature = 1.0,
    }, &generate_result));
    try std.testing.expectEqual(@as(usize, 2), generate_result.tokens_generated);
    try std.testing.expectEqual(expected_first.token, generated[0]);
    try std.testing.expectEqual(expected_second.token, generated[1]);
    try std.testing.expectEqual(expected_second.token, generate_result.last_token);
    try std.testing.expectEqual(expected_second.logit, generate_result.last_logit);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_program_output, &pos));
    try std.testing.expectEqual(@as(usize, 2), pos);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_bound_output));
    var expected_argmax_first = zgml_token_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_argmax_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
    }, &expected_argmax_first));
    var expected_argmax_second = zgml_token_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_execute_argmax_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{expected_argmax_first.token})[0..].ptr,
        .tokens_len = 1,
    }, &expected_argmax_second));
    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_program_output));
    var greedy_generated = [_]u32{ 777, 777 };
    var greedy_result = zgml_token_generate_argmax_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_generate_argmax_tokens(session_program_output, &.{
        .tokens = (&[_]u32{0})[0..].ptr,
        .tokens_len = 1,
        .output_tokens = greedy_generated[0..].ptr,
        .output_tokens_len = greedy_generated.len,
    }, &greedy_result));
    try std.testing.expectEqual(@as(usize, 2), greedy_result.tokens_generated);
    try std.testing.expectEqual(expected_argmax_first.token, greedy_generated[0]);
    try std.testing.expectEqual(expected_argmax_second.token, greedy_generated[1]);
    try std.testing.expectEqual(expected_argmax_second.token, greedy_result.last_token);
    try std.testing.expectEqual(expected_argmax_second.logit, greedy_result.last_logit);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_program_output, &pos));
    try std.testing.expectEqual(@as(usize, 2), pos);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_bound_output));
    bound_logits = [_]f32{-555} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(session_bound_output, &.{
        .tokens = (&[_]u32{ 0, 1 })[0..].ptr,
        .tokens_len = 2,
        .output = null,
        .output_len = 0,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqual(@as(f32, -555), bound_logits[tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_bound_output, &pos));
    try std.testing.expectEqual(@as(usize, 2), pos);

    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_b, &.{
        .token = 0,
        .output = logits_b[0..].ptr,
        .output_len = logits_b.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, logits_a[0..tiny_llama_config.vocab_size], logits_b[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -999), logits_a[tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -777), logits_b[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_prefill));
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_ref));

    const prefill_tokens = [_]u32{ 0, 1 };
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_prefill_tokens(session_prefill, &.{
        .tokens = prefill_tokens[0..].ptr,
        .tokens_len = prefill_tokens.len,
        .output = too_small[0..].ptr,
        .output_len = too_small.len,
    }, null));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_prefill, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    var prefill_logits = [_]f32{-222} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(session_prefill, &.{
        .tokens = prefill_tokens[0..].ptr,
        .tokens_len = prefill_tokens.len,
        .output = prefill_logits[0..].ptr,
        .output_len = prefill_logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_prefill, &pos));
    try std.testing.expectEqual(@as(usize, prefill_tokens.len), pos);
    try std.testing.expectEqualSlices(f32, prefill_logits[0..tiny_llama_config.vocab_size], bound_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -222), prefill_logits[tiny_llama_config.vocab_size]);

    var ref_logits = [_]f32{-444} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_advance_token(session_ref, &.{ .token = 0 }));
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_ref, &.{
        .token = 1,
        .output = ref_logits[0..].ptr,
        .output_len = ref_logits.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, ref_logits[0..tiny_llama_config.vocab_size], prefill_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -444), ref_logits[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_bulk));
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_bulk_expected));

    const too_many_tokens = [_]u32{ 0, 1, 2, 3, 4 };
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_advance_tokens(session_bulk, &.{
        .tokens = too_many_tokens[0..].ptr,
        .tokens_len = too_many_tokens.len,
    }));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_bulk, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    const bulk_prefix = [_]u32{ 0, 1 };
    try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(session_bulk, &.{
        .tokens = bulk_prefix[0..].ptr,
        .tokens_len = bulk_prefix.len,
    }));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_bulk, &pos));
    try std.testing.expectEqual(@as(usize, bulk_prefix.len), pos);

    var bulk_logits = [_]f32{-123} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session_bulk, &.{
        .token = 2,
        .output = bulk_logits[0..].ptr,
        .output_len = bulk_logits.len,
    }, &result));
    try std.testing.expectEqual(@as(f32, -123), bulk_logits[tiny_llama_config.vocab_size]);

    const expected_tokens = [_]u32{ 0, 1, 2 };
    var expected_logits = [_]f32{-321} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(session_bulk_expected, &.{
        .tokens = expected_tokens[0..].ptr,
        .tokens_len = expected_tokens.len,
        .output = expected_logits[0..].ptr,
        .output_len = expected_logits.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, expected_logits[0..tiny_llama_config.vocab_size], bulk_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -321), expected_logits[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session_execute));
    var too_many_execute_logits = [_]f32{-909} ** (tiny_llama_config.vocab_size + 1);
    result = .{ .output_len = 99 };
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_execute_tokens(session_execute, &.{
        .tokens = too_many_tokens[0..].ptr,
        .tokens_len = too_many_tokens.len,
        .output_policy = execute_output_logits,
        .output = too_many_execute_logits[0..].ptr,
        .output_len = too_many_execute_logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, 0), result.output_len);
    try std.testing.expectEqual(@as(f32, -909), too_many_execute_logits[tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_execute, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_execute_tokens(session_execute, &.{
        .tokens = prefill_tokens[0..].ptr,
        .tokens_len = prefill_tokens.len,
        .output_policy = execute_output_logits,
        .output = too_small[0..].ptr,
        .output_len = too_small.len,
    }, null));
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_execute, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);

    var execute_prefill_logits = [_]f32{-654} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_execute_tokens(session_execute, &.{
        .tokens = prefill_tokens[0..].ptr,
        .tokens_len = prefill_tokens.len,
        .output_policy = execute_output_logits,
        .output = execute_prefill_logits[0..].ptr,
        .output_len = execute_prefill_logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqualSlices(f32, prefill_logits[0..tiny_llama_config.vocab_size], execute_prefill_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -654), execute_prefill_logits[tiny_llama_config.vocab_size]);

    try std.testing.expectEqual(status(.ok), zgml_session_reset(session_execute));
    try std.testing.expectEqual(status(.ok), zgml_session_execute_tokens(session_execute, &.{
        .tokens = bulk_prefix[0..].ptr,
        .tokens_len = bulk_prefix.len,
        .output_policy = execute_output_none,
        .output = null,
        .output_len = 0,
    }, &result));
    try std.testing.expectEqual(@as(usize, 0), result.output_len);
    try std.testing.expectEqual(status(.ok), zgml_session_position(session_execute, &pos));
    try std.testing.expectEqual(@as(usize, bulk_prefix.len), pos);

    var execute_next_logits = [_]f32{-876} ** (tiny_llama_config.vocab_size + 1);
    try std.testing.expectEqual(status(.ok), zgml_session_execute_tokens(session_execute, &.{
        .tokens = (&[_]u32{2})[0..].ptr,
        .tokens_len = 1,
        .output_policy = execute_output_logits,
        .output = execute_next_logits[0..].ptr,
        .output_len = execute_next_logits.len,
    }, &result));
    try std.testing.expectEqualSlices(f32, expected_logits[0..tiny_llama_config.vocab_size], execute_next_logits[0..tiny_llama_config.vocab_size]);
    try std.testing.expectEqual(@as(f32, -876), execute_next_logits[tiny_llama_config.vocab_size]);
}

test "C ABI tiny llama can request Metal backend when available" {
    var model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    var session: ?*zgml_session = null;
    defer zgml_session_free(session);
    defer zgml_program_free(program);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));

    const compile_status = zgml_program_compile(model, &.{ .backend = backend_metal }, &program);
    if (compile_status == status(.unsupported)) return;
    try std.testing.expectEqual(status(.ok), compile_status);
    try std.testing.expect(program != null);

    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session));
    var logits = [_]f32{-999} ** (tiny_llama_config.vocab_size + 1);
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session, &.{
        .token = 0,
        .output = logits[0..].ptr,
        .output_len = logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), result.output_len);
    try std.testing.expectEqual(@as(f32, -999), logits[tiny_llama_config.vocab_size]);
}

test "C ABI tiny llama WebGPU backend follows execution gate" {
    var model: ?*zgml_model = null;
    var compatible_model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    defer zgml_program_free(program);
    defer zgml_model_free(compatible_model);
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &compatible_model));

    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_webgpu,
        .context_len = 4,
        .batch = 1,
    }, &program));
    try std.testing.expect(program != null);

    var inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(program, &inspection));
    try std.testing.expect(inspection.command_count > 0);
    try std.testing.expectEqual(@as(u64, backend_webgpu), inspection.backend);
    const expected_execution_supported = build_options.use_wgpu and build_options.experimental_llama_wgpu_execution;
    try std.testing.expectEqual(@as(u64, @intFromBool(expected_execution_supported)), inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 1), inspection.external_resources_supported);
    try std.testing.expect(inspection.buffer_count > 0);
    try std.testing.expect(inspection.buffer_byte_len > 0);
    try std.testing.expect(inspection.command_projection_count > 0 or inspection.command_attention_count > 0);
    try std.testing.expect(inspection.command_stencil_hash != 0);
    try std.testing.expect(inspection.binding_requirement_hash != 0);
    try std.testing.expect(inspection.persistent_requirement_count > 0);
    try std.testing.expect(inspection.step_input_requirement_count > 0);
    try std.testing.expectEqual(@as(u64, 1), inspection.step_output_requirement_count);
    try std.testing.expect(inspection.runtime_patch_holes > 0);
    try std.testing.expect(inspection.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(u64, 3), inspection.runtime_patch_max_cache_write_pos);
    try std.testing.expectEqual(@as(u64, 4), inspection.runtime_patch_max_attention_seq_kv);
    if (expected_execution_supported) {
        try std.testing.expect(inspection.backend_dispatch_count > 0);
        try std.testing.expectEqual(@as(u64, 1), inspection.dispatch_plan_supported);
        try std.testing.expectEqual(inspection.op_count, inspection.dispatch_plan_covered_op_count);
        try std.testing.expectEqual(std.math.maxInt(u64), inspection.dispatch_plan_first_unsupported_op);
        try std.testing.expect(inspection.dispatch_plan_projection_count > 0);
        try std.testing.expect(inspection.dispatch_plan_attention_count > 0);
    } else {
        try std.testing.expectEqual(@as(u64, 0), inspection.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_supported);
        try std.testing.expectEqual(@as(u64, 0), inspection.dispatch_plan_covered_op_count);
        try std.testing.expectEqual(std.math.maxInt(u64), inspection.dispatch_plan_first_unsupported_op);
    }

    var llama_inspection = zgml_llama_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_llama_program_inspect(program, &llama_inspection));
    try std.testing.expectEqual(@as(u64, tiny_llama_config.vocab_size), llama_inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, 4), llama_inspection.context_len);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_holes, inspection.runtime_patch_holes);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_cache_write_pos_holes, inspection.runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(llama_inspection.semantic_runtime_patch_attention_seq_kv_holes, inspection.runtime_patch_attention_seq_kv_holes);

    var plain_session: ?*zgml_session = @ptrFromInt(1);
    if (expected_execution_supported) {
        var cpu_program: ?*zgml_program = null;
        var cpu_session: ?*zgml_session = null;
        defer zgml_session_free(cpu_session);
        defer zgml_program_free(cpu_program);
        try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
            .backend = backend_cpu,
            .context_len = 4,
            .batch = 1,
        }, &cpu_program));
        try std.testing.expect(cpu_program != null);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(cpu_program, null, &cpu_session));
        try std.testing.expect(cpu_session != null);

        var cpu_logits = [_]f32{-999} ** (tiny_llama_config.vocab_size + 1);
        var cpu_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_session, &.{
            .token = 0,
            .output = cpu_logits[0..].ptr,
            .output_len = cpu_logits.len,
        }, &cpu_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), cpu_result.output_len);
        try std.testing.expectEqual(@as(f32, -999), cpu_logits[tiny_llama_config.vocab_size]);

        try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &plain_session));
        defer zgml_session_free(plain_session);
        try std.testing.expect(plain_session != null);
        var wgpu_logits = [_]f32{-777} ** (tiny_llama_config.vocab_size + 1);
        var wgpu_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(plain_session, &.{
            .token = 0,
            .output = wgpu_logits[0..].ptr,
            .output_len = wgpu_logits.len,
        }, &wgpu_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), wgpu_result.output_len);
        try std.testing.expectEqual(@as(f32, -777), wgpu_logits[tiny_llama_config.vocab_size]);
        for (wgpu_logits[0..tiny_llama_config.vocab_size], cpu_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var plain_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(plain_session, &plain_pos));
        try std.testing.expectEqual(@as(usize, 1), plain_pos);
        var plain_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(plain_session, &plain_profile));
        try std.testing.expectEqual(@as(u64, 1), plain_profile.call_count);
        try std.testing.expect(plain_profile.backend_op_count > 0);
        try std.testing.expectEqual(inspection.backend_dispatch_count, plain_profile.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), plain_profile.fallback_op_count);

        var cpu_prefill_session: ?*zgml_session = null;
        var wgpu_prefill_session: ?*zgml_session = null;
        defer zgml_session_free(wgpu_prefill_session);
        defer zgml_session_free(cpu_prefill_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(cpu_program, null, &cpu_prefill_session));
        try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &wgpu_prefill_session));
        try std.testing.expect(cpu_prefill_session != null);
        try std.testing.expect(wgpu_prefill_session != null);
        var prompt_tokens = [_]u32{ 0, 1 };
        var cpu_prefill_logits = [_]f32{-606} ** (tiny_llama_config.vocab_size + 1);
        var wgpu_prefill_logits = [_]f32{-505} ** (tiny_llama_config.vocab_size + 1);
        var cpu_prefill_result = zgml_step_result{ .output_len = 99 };
        var wgpu_prefill_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(cpu_prefill_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
            .output = cpu_prefill_logits[0..].ptr,
            .output_len = cpu_prefill_logits.len,
        }, &cpu_prefill_result));
        try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(wgpu_prefill_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
            .output = wgpu_prefill_logits[0..].ptr,
            .output_len = wgpu_prefill_logits.len,
        }, &wgpu_prefill_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), cpu_prefill_result.output_len);
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), wgpu_prefill_result.output_len);
        try std.testing.expectEqual(@as(f32, -606), cpu_prefill_logits[tiny_llama_config.vocab_size]);
        try std.testing.expectEqual(@as(f32, -505), wgpu_prefill_logits[tiny_llama_config.vocab_size]);
        for (wgpu_prefill_logits[0..tiny_llama_config.vocab_size], cpu_prefill_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var cpu_prefill_pos: usize = 99;
        var wgpu_prefill_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(cpu_prefill_session, &cpu_prefill_pos));
        try std.testing.expectEqual(status(.ok), zgml_session_position(wgpu_prefill_session, &wgpu_prefill_pos));
        try std.testing.expectEqual(@as(usize, 2), cpu_prefill_pos);
        try std.testing.expectEqual(@as(usize, 2), wgpu_prefill_pos);

        var cpu_after_prefill_logits = [_]f32{-404} ** (tiny_llama_config.vocab_size + 1);
        var wgpu_after_prefill_logits = [_]f32{-303} ** (tiny_llama_config.vocab_size + 1);
        var cpu_after_prefill_result = zgml_step_result{ .output_len = 99 };
        var wgpu_after_prefill_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_prefill_session, &.{
            .token = 2,
            .output = cpu_after_prefill_logits[0..].ptr,
            .output_len = cpu_after_prefill_logits.len,
        }, &cpu_after_prefill_result));
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(wgpu_prefill_session, &.{
            .token = 2,
            .output = wgpu_after_prefill_logits[0..].ptr,
            .output_len = wgpu_after_prefill_logits.len,
        }, &wgpu_after_prefill_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), cpu_after_prefill_result.output_len);
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), wgpu_after_prefill_result.output_len);
        try std.testing.expectEqual(@as(f32, -404), cpu_after_prefill_logits[tiny_llama_config.vocab_size]);
        try std.testing.expectEqual(@as(f32, -303), wgpu_after_prefill_logits[tiny_llama_config.vocab_size]);
        for (wgpu_after_prefill_logits[0..tiny_llama_config.vocab_size], cpu_after_prefill_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        var cpu_no_output_session: ?*zgml_session = null;
        var wgpu_no_output_session: ?*zgml_session = null;
        defer zgml_session_free(wgpu_no_output_session);
        defer zgml_session_free(cpu_no_output_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(cpu_program, null, &cpu_no_output_session));
        try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &wgpu_no_output_session));
        try std.testing.expectEqual(status(.ok), zgml_session_reset_runtime_profile(wgpu_no_output_session));
        try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(cpu_no_output_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
        }));
        try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(wgpu_no_output_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
        }));
        var cpu_no_output_pos: usize = 99;
        var wgpu_no_output_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(cpu_no_output_session, &cpu_no_output_pos));
        try std.testing.expectEqual(status(.ok), zgml_session_position(wgpu_no_output_session, &wgpu_no_output_pos));
        try std.testing.expectEqual(@as(usize, 2), cpu_no_output_pos);
        try std.testing.expectEqual(@as(usize, 2), wgpu_no_output_pos);
        var cpu_no_output_logits = [_]f32{-202} ** (tiny_llama_config.vocab_size + 1);
        var wgpu_no_output_logits = [_]f32{-101} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_no_output_session, &.{
            .token = 2,
            .output = cpu_no_output_logits[0..].ptr,
            .output_len = cpu_no_output_logits.len,
        }, &cpu_after_prefill_result));
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(wgpu_no_output_session, &.{
            .token = 2,
            .output = wgpu_no_output_logits[0..].ptr,
            .output_len = wgpu_no_output_logits.len,
        }, &wgpu_after_prefill_result));
        try std.testing.expectEqual(@as(f32, -202), cpu_no_output_logits[tiny_llama_config.vocab_size]);
        try std.testing.expectEqual(@as(f32, -101), wgpu_no_output_logits[tiny_llama_config.vocab_size]);
        for (wgpu_no_output_logits[0..tiny_llama_config.vocab_size], cpu_no_output_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var no_output_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(wgpu_no_output_session, &no_output_profile));
        try std.testing.expect(no_output_profile.call_count >= 2);
        try std.testing.expect(no_output_profile.backend_op_count > 0);
        try std.testing.expectEqual(@as(u64, 0), no_output_profile.fallback_op_count);
    } else {
        try std.testing.expectEqual(status(.unsupported), zgml_session_bind(program, null, &plain_session));
        try std.testing.expectEqual(@as(?*zgml_session, null), plain_session);
    }

    var requirements = zgml_llama_kv_cache_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_llama_program_get_kv_cache_requirements(program, &requirements));
    try std.testing.expectEqual(@as(usize, 4), requirements.context_len);

    if (expected_execution_supported) {
        var device_handle: usize = 0;
        try std.testing.expectEqual(status(.ok), zgml_program_get_device_handle(program, backend_webgpu, &device_handle));
        try std.testing.expect(device_handle != 0);

        var device_output: ?*zgml_buffer = null;
        var device_k: ?*zgml_buffer = null;
        var device_v: ?*zgml_buffer = null;
        var imported_output: ?*zgml_buffer = null;
        var imported_k: ?*zgml_buffer = null;
        var imported_v: ?*zgml_buffer = null;
        defer zgml_buffer_free(imported_v);
        defer zgml_buffer_free(imported_k);
        defer zgml_buffer_free(imported_output);
        defer zgml_buffer_free(device_v);
        defer zgml_buffer_free(device_k);
        defer zgml_buffer_free(device_output);
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(program, program_buffer_output, backend_webgpu, &device_output));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(program, program_buffer_llama_k_cache, backend_webgpu, &device_k));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(program, program_buffer_llama_v_cache, backend_webgpu, &device_v));
        try std.testing.expectEqual(tiny_llama_config.vocab_size * @sizeOf(f32), zgml_buffer_size(device_output));
        try std.testing.expectEqual(requirements.k_buffer_byte_len, zgml_buffer_size(device_k));
        try std.testing.expectEqual(requirements.v_buffer_byte_len, zgml_buffer_size(device_v));

        var output_info = zgml_buffer_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_output, &output_info));
        try std.testing.expectEqual(buffer_storage_external_resource, output_info.storage);
        try std.testing.expectEqual(backend_webgpu, output_info.placement);
        try std.testing.expect(output_info.handle != 0);
        var k_info = zgml_buffer_inspection{};
        var v_info = zgml_buffer_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_k, &k_info));
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_v, &v_info));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(program, program_buffer_output, &.{
            .placement = backend_webgpu,
            .device_handle = device_handle,
            .buffer_handle = output_info.handle,
            .byte_offset = output_info.byte_offset,
            .byte_len = output_info.byte_len,
        }, &imported_output));
        try std.testing.expectEqual(zgml_buffer_size(device_output), zgml_buffer_size(imported_output));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(program, program_buffer_llama_k_cache, &.{
            .placement = backend_webgpu,
            .device_handle = device_handle,
            .buffer_handle = k_info.handle,
            .byte_offset = k_info.byte_offset,
            .byte_len = k_info.byte_len,
        }, &imported_k));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(program, program_buffer_llama_v_cache, &.{
            .placement = backend_webgpu,
            .device_handle = device_handle,
            .buffer_handle = v_info.handle,
            .byte_offset = v_info.byte_offset,
            .byte_len = v_info.byte_len,
        }, &imported_v));
        try std.testing.expectEqual(zgml_buffer_size(device_k), zgml_buffer_size(imported_k));
        try std.testing.expectEqual(zgml_buffer_size(device_v), zgml_buffer_size(imported_v));

        var wrong_device_import: ?*zgml_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_import_device_buffer(program, program_buffer_output, &.{
            .placement = backend_webgpu,
            .device_handle = device_handle + 1,
            .buffer_handle = output_info.handle,
            .byte_offset = output_info.byte_offset,
            .byte_len = output_info.byte_len,
        }, &wrong_device_import));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), wrong_device_import);

        var zero_cache = [_]f32{0} ** 64;
        const k_zero = std.mem.sliceAsBytes(zero_cache[0 .. requirements.k_buffer_byte_len / @sizeOf(f32)]);
        const v_zero = std.mem.sliceAsBytes(zero_cache[0 .. requirements.v_buffer_byte_len / @sizeOf(f32)]);
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));

        var device_k_buffers = [_]?*zgml_buffer{device_k};
        var device_v_buffers = [_]?*zgml_buffer{device_v};
        const device_kv = zgml_llama_kv_cache_bind_desc{
            .k = device_k_buffers[0..].ptr,
            .v = device_v_buffers[0..].ptr,
            .len = device_k_buffers.len,
        };
        var imported_k_buffers = [_]?*zgml_buffer{imported_k};
        var imported_v_buffers = [_]?*zgml_buffer{imported_v};
        const imported_kv = zgml_llama_kv_cache_bind_desc{
            .k = imported_k_buffers[0..].ptr,
            .v = imported_v_buffers[0..].ptr,
            .len = imported_k_buffers.len,
        };

        var resource_cpu_program: ?*zgml_program = null;
        var resource_cpu_session: ?*zgml_session = null;
        var resource_session: ?*zgml_session = null;
        defer zgml_session_free(resource_session);
        defer zgml_session_free(resource_cpu_session);
        defer zgml_program_free(resource_cpu_program);
        try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
            .backend = backend_cpu,
            .context_len = 4,
            .batch = 1,
        }, &resource_cpu_program));
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &resource_cpu_session));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_session));

        var session_info = zgml_session_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_session_inspect(resource_session, &session_info));
        try std.testing.expectEqual(buffer_storage_external_resource, session_info.output_storage);
        try std.testing.expectEqual(buffer_storage_external_resource, session_info.kv_cache_storage);
        try std.testing.expect(session_info.resource_binding_count >= 3);

        var cpu_resource_logits = [_]f32{-909} ** (tiny_llama_config.vocab_size + 1);
        var cpu_resource_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(resource_cpu_session, &.{
            .token = 0,
            .output = cpu_resource_logits[0..].ptr,
            .output_len = cpu_resource_logits.len,
        }, &cpu_resource_result));
        var resource_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(resource_session, &.{
            .token = 0,
            .output = null,
            .output_len = 0,
        }, &resource_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), resource_result.output_len);
        var resource_logits = [_]f32{0} ** tiny_llama_config.vocab_size;
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        for (resource_logits[0..], cpu_resource_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }
        var resource_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_session, &resource_profile));
        try std.testing.expectEqual(@as(u64, 1), resource_profile.call_count);
        try std.testing.expect(resource_profile.backend_op_count > 0);
        try std.testing.expectEqual(inspection.backend_dispatch_count, resource_profile.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), resource_profile.fallback_op_count);

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var cpu_select_session: ?*zgml_session = null;
        var resource_select_session: ?*zgml_session = null;
        defer zgml_session_free(resource_select_session);
        defer zgml_session_free(cpu_select_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &cpu_select_session));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_select_session));
        var select_prompt = [_]u32{0};
        var cpu_select_logits = [_]f32{-1717} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_select_session, &.{
            .token = select_prompt[0],
            .output = cpu_select_logits[0..].ptr,
            .output_len = cpu_select_logits.len,
        }, &cpu_resource_result));
        const expected_select = argmaxToken(cpu_select_logits[0..tiny_llama_config.vocab_size]);
        var resource_argmax = zgml_token_argmax_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_execute_argmax_tokens(resource_select_session, &.{
            .tokens = select_prompt[0..].ptr,
            .tokens_len = select_prompt.len,
        }, &resource_argmax));
        try std.testing.expectEqual(expected_select.token, resource_argmax.token);
        try std.testing.expectApproxEqAbs(expected_select.logit, resource_argmax.logit, 1e-3);

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var resource_sample_session: ?*zgml_session = null;
        defer zgml_session_free(resource_sample_session);
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_sample_session));
        var resource_sample = zgml_token_sample_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_execute_sample_tokens(resource_sample_session, &.{
            .tokens = select_prompt[0..].ptr,
            .tokens_len = select_prompt.len,
            .top_k = 1,
            .seed = 123,
            .temperature = 1,
        }, &resource_sample));
        try std.testing.expectEqual(expected_select.token, resource_sample.token);
        try std.testing.expectApproxEqAbs(expected_select.logit, resource_sample.logit, 1e-3);

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var cpu_generate_session: ?*zgml_session = null;
        var resource_generate_session: ?*zgml_session = null;
        defer zgml_session_free(resource_generate_session);
        defer zgml_session_free(cpu_generate_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &cpu_generate_session));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_generate_session));
        var cpu_generate_logits = [_]f32{-1515} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_generate_session, &.{
            .token = select_prompt[0],
            .output = cpu_generate_logits[0..].ptr,
            .output_len = cpu_generate_logits.len,
        }, &cpu_resource_result));
        const expected_generated_0 = argmaxToken(cpu_generate_logits[0..tiny_llama_config.vocab_size]);
        var cpu_generate_next_logits = [_]f32{-1414} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_generate_session, &.{
            .token = expected_generated_0.token,
            .output = cpu_generate_next_logits[0..].ptr,
            .output_len = cpu_generate_next_logits.len,
        }, &cpu_resource_result));
        const expected_generated_1 = argmaxToken(cpu_generate_next_logits[0..tiny_llama_config.vocab_size]);
        var generated_tokens = [_]u32{ 99, 99 };
        var generate_result = zgml_token_generate_argmax_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_generate_argmax_tokens(resource_generate_session, &.{
            .tokens = select_prompt[0..].ptr,
            .tokens_len = select_prompt.len,
            .output_tokens = generated_tokens[0..].ptr,
            .output_tokens_len = generated_tokens.len,
        }, &generate_result));
        try std.testing.expectEqual(@as(usize, generated_tokens.len), generate_result.tokens_generated);
        try std.testing.expectEqual(expected_generated_0.token, generated_tokens[0]);
        try std.testing.expectEqual(expected_generated_1.token, generated_tokens[1]);
        try std.testing.expectEqual(expected_generated_1.token, generate_result.last_token);
        try std.testing.expectApproxEqAbs(expected_generated_1.logit, generate_result.last_logit, 1e-3);
        var resource_generate_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_generate_session, &resource_generate_pos));
        try std.testing.expectEqual(@as(usize, 2), resource_generate_pos);

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var resource_generate_sample_session: ?*zgml_session = null;
        defer zgml_session_free(resource_generate_sample_session);
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_generate_sample_session));
        var sampled_tokens = [_]u32{ 77, 77 };
        var generate_sample_result = zgml_token_generate_sample_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_generate_sample_tokens(resource_generate_sample_session, &.{
            .tokens = select_prompt[0..].ptr,
            .tokens_len = select_prompt.len,
            .output_tokens = sampled_tokens[0..].ptr,
            .output_tokens_len = sampled_tokens.len,
            .top_k = 1,
            .seed = 321,
            .temperature = 1,
        }, &generate_sample_result));
        try std.testing.expectEqual(@as(usize, sampled_tokens.len), generate_sample_result.tokens_generated);
        try std.testing.expectEqual(expected_generated_0.token, sampled_tokens[0]);
        try std.testing.expectEqual(expected_generated_1.token, sampled_tokens[1]);
        try std.testing.expectEqual(expected_generated_1.token, generate_sample_result.last_token);
        try std.testing.expectApproxEqAbs(expected_generated_1.logit, generate_sample_result.last_logit, 1e-3);

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(imported_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(imported_v, 0, v_zero.ptr, v_zero.len));
        var imported_resource_cpu_session: ?*zgml_session = null;
        var imported_resource_session: ?*zgml_session = null;
        defer zgml_session_free(imported_resource_session);
        defer zgml_session_free(imported_resource_cpu_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &imported_resource_cpu_session));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = imported_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &imported_kv,
        }, &imported_resource_session));
        var cpu_imported_logits = [_]f32{-818} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(imported_resource_cpu_session, &.{
            .token = 0,
            .output = cpu_imported_logits[0..].ptr,
            .output_len = cpu_imported_logits.len,
        }, &cpu_resource_result));
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(imported_resource_session, &.{
            .token = 0,
            .output = null,
            .output_len = 0,
        }, &resource_result));
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(imported_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        for (resource_logits[0..], cpu_imported_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        var cpu_prompt_session: ?*zgml_session = null;
        var resource_prompt_session: ?*zgml_session = null;
        defer zgml_session_free(resource_prompt_session);
        defer zgml_session_free(cpu_prompt_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &cpu_prompt_session));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_prompt_session));
        var prompt_tokens = [_]u32{ 0, 1 };
        var cpu_prompt_logits = [_]f32{-808} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(cpu_prompt_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
            .output = cpu_prompt_logits[0..].ptr,
            .output_len = cpu_prompt_logits.len,
        }, &cpu_resource_result));
        try std.testing.expectEqual(status(.ok), zgml_session_prefill_tokens(resource_prompt_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
            .output = null,
            .output_len = 0,
        }, &resource_result));
        try std.testing.expectEqual(@as(usize, tiny_llama_config.vocab_size), resource_result.output_len);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        for (resource_logits[0..], cpu_prompt_logits[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        var cpu_after_resource_prefill = [_]f32{-707} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_prompt_session, &.{
            .token = 2,
            .output = cpu_after_resource_prefill[0..].ptr,
            .output_len = cpu_after_resource_prefill.len,
        }, &cpu_resource_result));
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(resource_prompt_session, &.{
            .token = 2,
            .output = null,
            .output_len = 0,
        }, &resource_result));
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        for (resource_logits[0..], cpu_after_resource_prefill[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var resource_no_output_session: ?*zgml_session = null;
        var cpu_resource_no_output_session: ?*zgml_session = null;
        defer zgml_session_free(resource_no_output_session);
        defer zgml_session_free(cpu_resource_no_output_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind(resource_cpu_program, null, &cpu_resource_no_output_session));
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_no_output_session));
        var output_sentinel = [_]f32{-4141} ** tiny_llama_config.vocab_size;
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_output, 0, &output_sentinel, @sizeOf(@TypeOf(output_sentinel))));
        try std.testing.expectEqual(status(.ok), zgml_session_reset_runtime_profile(resource_no_output_session));
        try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(cpu_resource_no_output_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
        }));
        try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(resource_no_output_session, &.{
            .tokens = prompt_tokens[0..].ptr,
            .tokens_len = prompt_tokens.len,
        }));
        var resource_no_output_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_no_output_session, &resource_no_output_pos));
        try std.testing.expectEqual(@as(usize, prompt_tokens.len), resource_no_output_pos);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        try std.testing.expectEqualSlices(f32, output_sentinel[0..], resource_logits[0..]);
        var resource_no_output_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_no_output_session, &resource_no_output_profile));
        try std.testing.expect(resource_no_output_profile.call_count >= 1);
        try std.testing.expect(resource_no_output_profile.backend_op_count > 0);
        try std.testing.expectEqual(@as(u64, 0), resource_no_output_profile.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 0), resource_no_output_profile.sync_count);

        var cpu_after_resource_no_output = [_]f32{-616} ** (tiny_llama_config.vocab_size + 1);
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(cpu_resource_no_output_session, &.{
            .token = 2,
            .output = cpu_after_resource_no_output[0..].ptr,
            .output_len = cpu_after_resource_no_output.len,
        }, &cpu_resource_result));
        try std.testing.expectEqual(status(.ok), zgml_session_step_token(resource_no_output_session, &.{
            .token = 2,
            .output = null,
            .output_len = 0,
        }, &resource_result));
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        for (resource_logits[0..], cpu_after_resource_no_output[0..tiny_llama_config.vocab_size]) |actual, want| {
            try std.testing.expectApproxEqAbs(want, actual, 1e-3);
        }

        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_k, 0, k_zero.ptr, k_zero.len));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_v, 0, v_zero.ptr, v_zero.len));
        var resource_over_context_session: ?*zgml_session = null;
        defer zgml_session_free(resource_over_context_session);
        try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
            .output = device_output,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &device_kv,
        }, &resource_over_context_session));
        var over_context_sentinel = [_]f32{-8181} ** tiny_llama_config.vocab_size;
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_output, 0, &over_context_sentinel, @sizeOf(@TypeOf(over_context_sentinel))));
        var over_context_tokens = [_]u32{ 0, 1, 2, 3 };
        try std.testing.expectEqual(status(.ok), zgml_session_advance_tokens(resource_over_context_session, &.{
            .tokens = over_context_tokens[0..].ptr,
            .tokens_len = over_context_tokens.len,
        }));
        var resource_over_context_pos: usize = 99;
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_over_context_session, &resource_over_context_pos));
        try std.testing.expectEqual(@as(usize, 4), resource_over_context_pos);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        try std.testing.expectEqualSlices(f32, over_context_sentinel[0..], resource_logits[0..]);

        var over_context_profile_before = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_over_context_session, &over_context_profile_before));
        try std.testing.expect(over_context_profile_before.call_count > 0);
        try std.testing.expect(over_context_profile_before.backend_op_count > 0);
        try std.testing.expectEqual(@as(u64, 0), over_context_profile_before.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 0), over_context_profile_before.sync_count);

        var over_context_result = zgml_step_result{ .output_len = 99 };
        try std.testing.expectEqual(status(.shape_mismatch), zgml_session_step_token(resource_over_context_session, &.{
            .token = 0,
            .output = null,
            .output_len = 0,
        }, &over_context_result));
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_over_context_session, &resource_over_context_pos));
        try std.testing.expectEqual(@as(usize, 4), resource_over_context_pos);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        try std.testing.expectEqualSlices(f32, over_context_sentinel[0..], resource_logits[0..]);
        var over_context_profile_after = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_over_context_session, &over_context_profile_after));
        try expectRuntimeProfilesEqual(over_context_profile_before, over_context_profile_after);

        var over_context_argmax_tokens = [_]u32{ 1234, 5678 };
        var over_context_argmax = zgml_token_generate_argmax_result{
            .tokens_generated = 99,
            .last_token = 99,
            .last_logit = -99,
        };
        try std.testing.expectEqual(status(.shape_mismatch), zgml_session_generate_argmax_tokens(resource_over_context_session, &.{
            .tokens = over_context_tokens[0..1].ptr,
            .tokens_len = 1,
            .output_tokens = over_context_argmax_tokens[0..].ptr,
            .output_tokens_len = over_context_argmax_tokens.len,
        }, &over_context_argmax));
        try std.testing.expectEqual(@as(usize, 0), over_context_argmax.tokens_generated);
        try std.testing.expectEqual(@as(u32, 0), over_context_argmax.last_token);
        try std.testing.expectEqual(@as(f32, 0), over_context_argmax.last_logit);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 1234, 5678 }, over_context_argmax_tokens[0..]);
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_over_context_session, &resource_over_context_pos));
        try std.testing.expectEqual(@as(usize, 4), resource_over_context_pos);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        try std.testing.expectEqualSlices(f32, over_context_sentinel[0..], resource_logits[0..]);
        var over_context_profile_after_argmax = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_over_context_session, &over_context_profile_after_argmax));
        try expectRuntimeProfilesEqual(over_context_profile_before, over_context_profile_after_argmax);

        var over_context_sample_tokens = [_]u32{ 8765, 4321 };
        var over_context_sample = zgml_token_generate_sample_result{
            .tokens_generated = 99,
            .last_token = 99,
            .last_logit = -99,
        };
        try std.testing.expectEqual(status(.shape_mismatch), zgml_session_generate_sample_tokens(resource_over_context_session, &.{
            .tokens = over_context_tokens[0..1].ptr,
            .tokens_len = 1,
            .output_tokens = over_context_sample_tokens[0..].ptr,
            .output_tokens_len = over_context_sample_tokens.len,
            .top_k = 1,
            .seed = 123,
            .temperature = 1,
        }, &over_context_sample));
        try std.testing.expectEqual(@as(usize, 0), over_context_sample.tokens_generated);
        try std.testing.expectEqual(@as(u32, 0), over_context_sample.last_token);
        try std.testing.expectEqual(@as(f32, 0), over_context_sample.last_logit);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 8765, 4321 }, over_context_sample_tokens[0..]);
        try std.testing.expectEqual(status(.ok), zgml_session_position(resource_over_context_session, &resource_over_context_pos));
        try std.testing.expectEqual(@as(usize, 4), resource_over_context_pos);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, resource_logits[0..].ptr, @sizeOf(@TypeOf(resource_logits))));
        try std.testing.expectEqualSlices(f32, over_context_sentinel[0..], resource_logits[0..]);
        var over_context_profile_after_sample = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(resource_over_context_session, &over_context_profile_after_sample));
        try expectRuntimeProfilesEqual(over_context_profile_before, over_context_profile_after_sample);
    } else {
        var unsupported_device_buffer: ?*zgml_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_create_device_buffer(program, program_buffer_output, backend_webgpu, &unsupported_device_buffer));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), unsupported_device_buffer);

        var unsupported_device_handle: usize = 99;
        try std.testing.expectEqual(status(.unsupported), zgml_program_get_device_handle(program, backend_webgpu, &unsupported_device_handle));
        try std.testing.expectEqual(@as(usize, 0), unsupported_device_handle);

        unsupported_device_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_import_device_buffer(program, program_buffer_output, &.{
            .placement = backend_webgpu,
            .device_handle = 77,
            .buffer_handle = 88,
            .byte_len = 0,
        }, &unsupported_device_buffer));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), unsupported_device_buffer);
    }

    var output_resource: ?*zgml_buffer = null;
    var k_resource: ?*zgml_buffer = null;
    var v_resource: ?*zgml_buffer = null;
    defer zgml_buffer_free(v_resource);
    defer zgml_buffer_free(k_resource);
    defer zgml_buffer_free(output_resource);
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_write,
        .handle = 201,
        .byte_len = tiny_llama_config.vocab_size * @sizeOf(f32),
    }, &output_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_read_write,
        .handle = 202,
        .byte_len = requirements.k_buffer_byte_len,
    }, &k_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_read_write,
        .handle = 203,
        .byte_len = requirements.v_buffer_byte_len,
    }, &v_resource));

    var bad_output_resource: ?*zgml_buffer = null;
    var bad_k_write_only_resource: ?*zgml_buffer = null;
    var bad_k_read_only_resource: ?*zgml_buffer = null;
    defer zgml_buffer_free(bad_k_read_only_resource);
    defer zgml_buffer_free(bad_k_write_only_resource);
    defer zgml_buffer_free(bad_output_resource);
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_read,
        .handle = 301,
        .byte_len = tiny_llama_config.vocab_size * @sizeOf(f32),
    }, &bad_output_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_write,
        .handle = 302,
        .byte_len = requirements.k_buffer_byte_len,
    }, &bad_k_write_only_resource));
    try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
        .placement = backend_webgpu,
        .access_flags = resource_access_read,
        .handle = 303,
        .byte_len = requirements.k_buffer_byte_len,
    }, &bad_k_read_only_resource));

    var k_buffers = [_]?*zgml_buffer{k_resource};
    var v_buffers = [_]?*zgml_buffer{v_resource};
    const kv = zgml_llama_kv_cache_bind_desc{
        .k = k_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = k_buffers.len,
    };
    var bad_output_session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_llama_session_bind_buffers(program, &.{
        .output = bad_output_resource,
        .output_len = tiny_llama_config.vocab_size,
        .kv_cache = &kv,
    }, &bad_output_session));
    try std.testing.expectEqual(@as(?*zgml_session, null), bad_output_session);

    var bad_k_write_buffers = [_]?*zgml_buffer{bad_k_write_only_resource};
    const bad_k_write_kv = zgml_llama_kv_cache_bind_desc{
        .k = bad_k_write_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = bad_k_write_buffers.len,
    };
    var bad_k_write_session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_llama_session_bind_buffers(program, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
        .kv_cache = &bad_k_write_kv,
    }, &bad_k_write_session));
    try std.testing.expectEqual(@as(?*zgml_session, null), bad_k_write_session);

    var bad_k_read_buffers = [_]?*zgml_buffer{bad_k_read_only_resource};
    const bad_k_read_kv = zgml_llama_kv_cache_bind_desc{
        .k = bad_k_read_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = bad_k_read_buffers.len,
    };
    var bad_k_read_session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_llama_session_bind_buffers(program, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
        .kv_cache = &bad_k_read_kv,
    }, &bad_k_read_session));
    try std.testing.expectEqual(@as(?*zgml_session, null), bad_k_read_session);

    if (expected_execution_supported) {
        var fake_resource_session: ?*zgml_session = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_llama_session_bind_buffers(program, &.{
            .output = output_resource,
            .output_len = tiny_llama_config.vocab_size,
            .kv_cache = &kv,
        }, &fake_resource_session));
        try std.testing.expectEqual(@as(?*zgml_session, null), fake_resource_session);
        return;
    }

    var session: ?*zgml_session = null;
    try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
        .kv_cache = &kv,
    }, &session));
    defer zgml_session_free(session);
    try std.testing.expect(session != null);

    var session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(tiny_llama_kind, session_inspection.model_kind);
    try std.testing.expectEqual(backend_webgpu, session_inspection.backend);
    try std.testing.expectEqual(buffer_storage_external_resource, session_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_external_resource, session_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.position);
    try std.testing.expectEqual(@as(u64, 4), session_inspection.context_len);
    try std.testing.expect(session_inspection.persistent_binding_count > 0);
    try std.testing.expect(session_inspection.step_input_count > 0);
    try std.testing.expectEqual(@as(u64, 1), session_inspection.step_output_count);
    try std.testing.expect(session_inspection.host_binding_count > 0);
    try std.testing.expectEqual(@as(u64, 3), session_inspection.resource_binding_count);
    try std.testing.expect(session_inspection.binding_shape_hash != 0);

    var logits = [_]f32{-88} ** (tiny_llama_config.vocab_size + 1);
    var step_result = zgml_step_result{ .output_len = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_session_step_token(session, &.{
        .token = 0,
        .output = logits[0..].ptr,
        .output_len = logits.len,
    }, &step_result));
    try std.testing.expectEqual(@as(usize, 0), step_result.output_len);
    try std.testing.expectEqual(@as(f32, -88), logits[0]);
    var pos: usize = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(session, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    try std.testing.expectEqual(status(.unsupported), zgml_session_advance_token(session, &.{ .token = 0 }));
    pos = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(session, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    var resource_profile = zgml_runtime_profile{};
    try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(session, &resource_profile));
    try std.testing.expectEqual(@as(u64, 0), resource_profile.call_count);
    try std.testing.expectEqual(@as(u64, 0), resource_profile.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), resource_profile.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 0), resource_profile.sync_count);
    try std.testing.expectEqual(@as(u64, 0), resource_profile.runtime_patch_call_count);

    var model_session: ?*zgml_session = null;
    defer zgml_session_free(model_session);
    try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_model_buffers(program, compatible_model, &.{
        .output = output_resource,
        .output_len = tiny_llama_config.vocab_size,
        .kv_cache = &kv,
    }, &model_session));
    try std.testing.expect(model_session != null);

    var model_session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(model_session, &model_session_inspection));
    try std.testing.expectEqual(tiny_llama_kind, model_session_inspection.model_kind);
    try std.testing.expectEqual(backend_webgpu, model_session_inspection.backend);
    try std.testing.expectEqual(buffer_storage_external_resource, model_session_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_external_resource, model_session_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), model_session_inspection.position);
    try std.testing.expectEqual(@as(u64, 4), model_session_inspection.context_len);
    try std.testing.expect(model_session_inspection.persistent_binding_count > 0);
    try std.testing.expect(model_session_inspection.step_input_count > 0);
    try std.testing.expectEqual(@as(u64, 1), model_session_inspection.step_output_count);
    try std.testing.expect(model_session_inspection.host_binding_count > 0);
    try std.testing.expectEqual(@as(u64, 3), model_session_inspection.resource_binding_count);
    try std.testing.expect(model_session_inspection.binding_shape_hash != 0);
    try std.testing.expectEqual(session_inspection.binding_shape_hash, model_session_inspection.binding_shape_hash);

    @memset(&logits, -77);
    step_result = .{ .output_len = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_session_step_token(model_session, &.{
        .token = 0,
        .output = logits[0..].ptr,
        .output_len = logits.len,
    }, &step_result));
    try std.testing.expectEqual(@as(usize, 0), step_result.output_len);
    try std.testing.expectEqual(@as(f32, -77), logits[0]);
    pos = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(model_session, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    try std.testing.expectEqual(status(.unsupported), zgml_session_advance_token(model_session, &.{ .token = 0 }));
    pos = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(model_session, &pos));
    try std.testing.expectEqual(@as(usize, 0), pos);
    var model_resource_profile = zgml_runtime_profile{};
    try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(model_session, &model_resource_profile));
    try std.testing.expectEqual(@as(u64, 0), model_resource_profile.call_count);
    try std.testing.expectEqual(@as(u64, 0), model_resource_profile.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), model_resource_profile.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 0), model_resource_profile.sync_count);
    try std.testing.expectEqual(@as(u64, 0), model_resource_profile.runtime_patch_call_count);
}

test "C ABI validates llama compile envelope" {
    var model: ?*zgml_model = null;
    defer zgml_model_free(model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_llama_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));

    var program: ?*zgml_program = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_program_compile(model, &.{
        .context_len = tiny_llama_config.max_seq_len + 1,
    }, &program));
    try std.testing.expectEqual(@as(?*zgml_program, null), program);

    program = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_program_compile(model, &.{ .batch = 2 }, &program));
    try std.testing.expectEqual(@as(?*zgml_program, null), program);
}

test "C ABI validates handle shapes and clears failed outputs" {
    var model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 0,
        .output_len = 3,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);

    var good_model: ?*zgml_model = null;
    var program: ?*zgml_program = null;
    defer zgml_program_free(program);
    defer zgml_model_free(good_model);

    try std.testing.expectEqual(status(.ok), zgml_model_create(&.{
        .kind = tiny_linear_kind,
        .input_len = 2,
        .output_len = 2,
    }, &good_model));

    var invalid_program: ?*zgml_program = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_program_compile(good_model, &.{ .backend = 999 }, &invalid_program));
    try std.testing.expectEqual(@as(?*zgml_program, null), invalid_program);

    var webgpu_linear_program: ?*zgml_program = null;
    defer zgml_program_free(webgpu_linear_program);
    try std.testing.expectEqual(status(.ok), zgml_program_compile(good_model, &.{ .backend = backend_webgpu }, &webgpu_linear_program));
    try std.testing.expect(webgpu_linear_program != null);

    var webgpu_inspection = zgml_program_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_program_inspect(webgpu_linear_program, &webgpu_inspection));
    try std.testing.expectEqual(@as(u64, backend_webgpu), webgpu_inspection.backend);
    try std.testing.expectEqual(@as(u64, if (build_options.use_wgpu) 1 else 0), webgpu_inspection.execution_supported);
    try std.testing.expectEqual(@as(u64, 1), webgpu_inspection.external_resources_supported);
    try std.testing.expectEqual(@as(u64, 5), webgpu_inspection.buffer_count);
    try std.testing.expectEqual(@as(u64, 12), webgpu_inspection.buffer_element_count);
    try std.testing.expectEqual(@as(u64, 48), webgpu_inspection.buffer_byte_len);
    try std.testing.expectEqual(@as(u64, 1), webgpu_inspection.initial_upload_count);
    try std.testing.expectEqual(@as(u64, 0), webgpu_inspection.qweight_count);
    try std.testing.expect(webgpu_inspection.op_count > 0);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u32)), webgpu_inspection.runtime_patch_max_cache_write_pos);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u32)), webgpu_inspection.runtime_patch_max_attention_seq_kv);
    try std.testing.expect(webgpu_inspection.command_count > 0);
    try std.testing.expect(webgpu_inspection.command_stencil_hash != 0);
    try std.testing.expect(commandCategoryTotal(webgpu_inspection) > 0);
    if (build_options.use_wgpu) {
        try std.testing.expect(webgpu_inspection.backend_dispatch_count > 0);
        try std.testing.expectEqual(@as(u64, 1), webgpu_inspection.dispatch_plan_supported);
        try std.testing.expectEqual(webgpu_inspection.op_count, webgpu_inspection.dispatch_plan_covered_op_count);
        try std.testing.expectEqual(std.math.maxInt(u64), webgpu_inspection.dispatch_plan_first_unsupported_op);
        try std.testing.expect(webgpu_inspection.dispatch_plan_projection_count > 0);
    } else {
        try std.testing.expectEqual(@as(u64, 0), webgpu_inspection.backend_dispatch_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_inspection.dispatch_plan_supported);
        try std.testing.expectEqual(@as(u64, 0), webgpu_inspection.dispatch_plan_covered_op_count);
        try std.testing.expectEqual(std.math.maxInt(u64), webgpu_inspection.dispatch_plan_first_unsupported_op);
    }

    const valid_weights = [_]f32{ 1, 0, 0, 1 };
    const valid_bias = [_]f32{ 0, 0 };
    var webgpu_session: ?*zgml_session = @ptrFromInt(1);
    if (build_options.use_wgpu) {
        var webgpu_input = [_]f32{ 2, 3 };
        var webgpu_output = [_]f32{ -9, -9 };
        try std.testing.expectEqual(status(.ok), zgml_session_bind(webgpu_linear_program, &.{
            .weights = valid_weights[0..].ptr,
            .weights_len = valid_weights.len,
            .bias = valid_bias[0..].ptr,
            .bias_len = valid_bias.len,
            .input = webgpu_input[0..].ptr,
            .input_len = webgpu_input.len,
            .output = webgpu_output[0..].ptr,
            .output_len = webgpu_output.len,
        }, &webgpu_session));
        defer zgml_session_free(webgpu_session);
        try std.testing.expect(webgpu_session != null);

        var webgpu_session_inspection = zgml_session_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_session_inspect(webgpu_session, &webgpu_session_inspection));
        try std.testing.expectEqual(tiny_linear_kind, webgpu_session_inspection.model_kind);
        try std.testing.expectEqual(backend_webgpu, webgpu_session_inspection.backend);
        try std.testing.expectEqual(buffer_storage_host, webgpu_session_inspection.output_storage);
        try std.testing.expectEqual(@as(u64, 2), webgpu_session_inspection.persistent_binding_count);
        try std.testing.expectEqual(@as(u64, 1), webgpu_session_inspection.step_input_count);
        try std.testing.expectEqual(@as(u64, 1), webgpu_session_inspection.step_output_count);
        try std.testing.expectEqual(@as(u64, 4), webgpu_session_inspection.host_binding_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_session_inspection.resource_binding_count);
        try std.testing.expect(webgpu_session_inspection.binding_shape_hash != 0);

        var webgpu_step_result = zgml_step_result{};
        try std.testing.expectEqual(status(.ok), zgml_session_step(webgpu_session, null, &webgpu_step_result));
        try std.testing.expectEqual(@as(usize, 2), webgpu_step_result.output_len);
        try std.testing.expectApproxEqAbs(@as(f32, 2), webgpu_output[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 3), webgpu_output[1], 1e-5);

        var host_weights: ?*zgml_buffer = null;
        var host_bias: ?*zgml_buffer = null;
        var host_input: ?*zgml_buffer = null;
        var host_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(host_output);
        defer zgml_buffer_free(host_input);
        defer zgml_buffer_free(host_bias);
        defer zgml_buffer_free(host_weights);
        try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(webgpu_linear_program, program_buffer_weights, &host_weights));
        try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(webgpu_linear_program, program_buffer_bias, &host_bias));
        try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(webgpu_linear_program, program_buffer_input, &host_input));
        try std.testing.expectEqual(status(.ok), zgml_program_create_buffer(webgpu_linear_program, program_buffer_output, &host_output));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_weights, 0, valid_weights[0..].ptr, valid_weights.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_bias, 0, valid_bias[0..].ptr, valid_bias.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_input, 0, webgpu_input[0..].ptr, webgpu_input.len * @sizeOf(f32)));
        var host_buffer_session: ?*zgml_session = null;
        defer zgml_session_free(host_buffer_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_linear_program, &.{
            .weights = host_weights,
            .weights_len = 4,
            .bias = host_bias,
            .bias_len = 2,
            .input = host_input,
            .input_len = 2,
            .output = host_output,
            .output_len = 2,
        }, &host_buffer_session));
        try std.testing.expectEqual(status(.ok), zgml_session_step(host_buffer_session, null, &webgpu_step_result));
        var host_buffer_output = [_]f32{ -1, -1 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(host_output, 0, host_buffer_output[0..].ptr, host_buffer_output.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@as(f32, 2), host_buffer_output[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 3), host_buffer_output[1], 1e-5);

        const mutated_weights = [_]f32{ 9, 9, 9, 9 };
        const next_input = [_]f32{ 4, 5 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_weights, 0, mutated_weights[0..].ptr, mutated_weights.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_input, 0, next_input[0..].ptr, next_input.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_session_step(host_buffer_session, null, &webgpu_step_result));
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(host_output, 0, host_buffer_output[0..].ptr, host_buffer_output.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@as(f32, 4), host_buffer_output[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5), host_buffer_output[1], 1e-5);

        const no_output_input = [_]f32{ 6, 7 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(host_input, 0, no_output_input[0..].ptr, no_output_input.len * @sizeOf(f32)));
        webgpu_step_result = .{ .output_len = 99 };
        try std.testing.expectEqual(status(.ok), zgml_session_step_no_output(host_buffer_session, null, &webgpu_step_result));
        try std.testing.expectEqual(@as(usize, 0), webgpu_step_result.output_len);
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(host_output, 0, host_buffer_output[0..].ptr, host_buffer_output.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@as(f32, 4), host_buffer_output[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5), host_buffer_output[1], 1e-5);
        var host_buffer_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(host_buffer_session, &host_buffer_profile));
        try std.testing.expectEqual(@as(u64, 3), host_buffer_profile.call_count);
        try std.testing.expectEqual(@as(u64, 2), host_buffer_profile.sync_count);

        var resource_weights: ?*zgml_buffer = null;
        var resource_bias: ?*zgml_buffer = null;
        var resource_input: ?*zgml_buffer = null;
        var resource_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(resource_output);
        defer zgml_buffer_free(resource_input);
        defer zgml_buffer_free(resource_bias);
        defer zgml_buffer_free(resource_weights);
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 101,
            .byte_len = 4 * @sizeOf(f32),
        }, &resource_weights));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 102,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_bias));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 103,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_input));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_write,
            .handle = 104,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_output));

        var resource_session: ?*zgml_session = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_session_bind_buffers(webgpu_linear_program, &.{
            .weights = resource_weights,
            .weights_len = 4,
            .bias = resource_bias,
            .bias_len = 2,
            .input = resource_input,
            .input_len = 2,
            .output = resource_output,
            .output_len = 2,
        }, &resource_session));
        try std.testing.expectEqual(@as(?*zgml_session, null), resource_session);

        var device_weights: ?*zgml_buffer = null;
        var device_bias: ?*zgml_buffer = null;
        var device_input: ?*zgml_buffer = null;
        var device_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(device_output);
        defer zgml_buffer_free(device_input);
        defer zgml_buffer_free(device_bias);
        defer zgml_buffer_free(device_weights);
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_weights, backend_webgpu, &device_weights));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_bias, backend_webgpu, &device_bias));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_input, backend_webgpu, &device_input));
        try std.testing.expectEqual(status(.ok), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_output, backend_webgpu, &device_output));

        var device_output_inspection = zgml_buffer_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_output, &device_output_inspection));
        try std.testing.expectEqual(buffer_storage_external_resource, device_output_inspection.storage);
        try std.testing.expectEqual(backend_webgpu, device_output_inspection.placement);
        try std.testing.expectEqual(resource_access_write, device_output_inspection.access_flags);
        try std.testing.expect(device_output_inspection.handle != 0);

        const device_weights_values = [_]f32{ 1, 0, 0, 1 };
        const device_bias_values = [_]f32{ 10, -1 };
        const device_input_values = [_]f32{ 6, 7 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_weights, 0, device_weights_values[0..].ptr, device_weights_values.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_bias, 0, device_bias_values[0..].ptr, device_bias_values.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(device_input, 0, device_input_values[0..].ptr, device_input_values.len * @sizeOf(f32)));

        var device_session: ?*zgml_session = null;
        defer zgml_session_free(device_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_linear_program, &.{
            .weights = device_weights,
            .weights_len = 4,
            .bias = device_bias,
            .bias_len = 2,
            .input = device_input,
            .input_len = 2,
            .output = device_output,
            .output_len = 2,
        }, &device_session));
        try std.testing.expectEqual(status(.ok), zgml_session_step(device_session, null, &webgpu_step_result));
        var device_output_values = [_]f32{ -1, -1 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(device_output, 0, device_output_values[0..].ptr, device_output_values.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@as(f32, 16), device_output_values[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 6), device_output_values[1], 1e-5);

        var device_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(device_session, &device_profile));
        try std.testing.expectEqual(@as(u64, 1), device_profile.call_count);
        try std.testing.expectEqual(@as(u64, 0), device_profile.sync_count);

        var webgpu_device_handle: usize = 0;
        try std.testing.expectEqual(status(.ok), zgml_program_get_device_handle(webgpu_linear_program, backend_webgpu, &webgpu_device_handle));
        try std.testing.expect(webgpu_device_handle != 0);
        var cpu_device_handle: usize = 99;
        try std.testing.expectEqual(status(.unsupported), zgml_program_get_device_handle(webgpu_linear_program, backend_cpu, &cpu_device_handle));
        try std.testing.expectEqual(@as(usize, 0), cpu_device_handle);

        var device_weights_inspection = zgml_buffer_inspection{};
        var device_bias_inspection = zgml_buffer_inspection{};
        var device_input_inspection = zgml_buffer_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_weights, &device_weights_inspection));
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_bias, &device_bias_inspection));
        try std.testing.expectEqual(status(.ok), zgml_buffer_inspect(device_input, &device_input_inspection));

        var bad_import: ?*zgml_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_weights, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle + 1,
            .buffer_handle = @intCast(device_weights_inspection.handle),
            .byte_len = device_weights_inspection.byte_len,
        }, &bad_import));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_import);
        try std.testing.expectEqual(status(.unsupported), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_weights, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_weights_inspection.handle),
            .byte_offset = 4,
            .byte_len = device_weights_inspection.byte_len,
        }, &bad_import));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_import);
        try std.testing.expectEqual(status(.shape_mismatch), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_weights, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_weights_inspection.handle),
            .byte_len = device_weights_inspection.byte_len + 4,
        }, &bad_import));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), bad_import);

        var imported_weights: ?*zgml_buffer = null;
        var imported_bias: ?*zgml_buffer = null;
        var imported_input: ?*zgml_buffer = null;
        var imported_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(imported_output);
        defer zgml_buffer_free(imported_input);
        defer zgml_buffer_free(imported_bias);
        defer zgml_buffer_free(imported_weights);
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_weights, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_weights_inspection.handle),
            .byte_len = 0,
        }, &imported_weights));
        try std.testing.expectEqual(@as(usize, @intCast(device_weights_inspection.byte_len)), zgml_buffer_size(imported_weights));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_bias, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_bias_inspection.handle),
            .byte_len = device_bias_inspection.byte_len,
        }, &imported_bias));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_input, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_input_inspection.handle),
            .byte_len = device_input_inspection.byte_len,
        }, &imported_input));
        try std.testing.expectEqual(status(.ok), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_output, &.{
            .placement = backend_webgpu,
            .device_handle = webgpu_device_handle,
            .buffer_handle = @intCast(device_output_inspection.handle),
            .byte_len = device_output_inspection.byte_len,
        }, &imported_output));

        zgml_buffer_free(device_output);
        zgml_buffer_free(device_input);
        zgml_buffer_free(device_bias);
        zgml_buffer_free(device_weights);
        device_output = null;
        device_input = null;
        device_bias = null;
        device_weights = null;

        const imported_weights_values = [_]f32{ 2, 0, 0, 2 };
        const imported_bias_values = [_]f32{ 5, -1 };
        const imported_input_values = [_]f32{ 3, 4 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(imported_weights, 0, imported_weights_values[0..].ptr, imported_weights_values.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(imported_bias, 0, imported_bias_values[0..].ptr, imported_bias_values.len * @sizeOf(f32)));
        try std.testing.expectEqual(status(.ok), zgml_buffer_write(imported_input, 0, imported_input_values[0..].ptr, imported_input_values.len * @sizeOf(f32)));

        var imported_session: ?*zgml_session = null;
        defer zgml_session_free(imported_session);
        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_linear_program, &.{
            .weights = imported_weights,
            .weights_len = 4,
            .bias = imported_bias,
            .bias_len = 2,
            .input = imported_input,
            .input_len = 2,
            .output = imported_output,
            .output_len = 2,
        }, &imported_session));
        try std.testing.expectEqual(status(.ok), zgml_session_step(imported_session, null, &webgpu_step_result));
        var imported_output_values = [_]f32{ -1, -1 };
        try std.testing.expectEqual(status(.ok), zgml_buffer_read(imported_output, 0, imported_output_values[0..].ptr, imported_output_values.len * @sizeOf(f32)));
        try std.testing.expectApproxEqAbs(@as(f32, 11), imported_output_values[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 7), imported_output_values[1], 1e-5);

        var imported_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(imported_session, &imported_profile));
        try std.testing.expectEqual(@as(u64, 1), imported_profile.call_count);
        try std.testing.expectEqual(@as(u64, 0), imported_profile.sync_count);
    } else {
        try std.testing.expectEqual(status(.unsupported), zgml_session_bind(webgpu_linear_program, &.{
            .weights = valid_weights[0..].ptr,
            .weights_len = valid_weights.len,
        }, &webgpu_session));
        try std.testing.expectEqual(@as(?*zgml_session, null), webgpu_session);

        var unsupported_device_buffer: ?*zgml_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_weights, backend_cpu, &unsupported_device_buffer));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), unsupported_device_buffer);
        unsupported_device_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_create_device_buffer(webgpu_linear_program, program_buffer_weights, backend_webgpu, &unsupported_device_buffer));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), unsupported_device_buffer);

        var unsupported_device_handle: usize = 99;
        try std.testing.expectEqual(status(.unsupported), zgml_program_get_device_handle(webgpu_linear_program, backend_webgpu, &unsupported_device_handle));
        try std.testing.expectEqual(@as(usize, 0), unsupported_device_handle);

        unsupported_device_buffer = @ptrFromInt(1);
        try std.testing.expectEqual(status(.unsupported), zgml_program_import_device_buffer(webgpu_linear_program, program_buffer_weights, &.{
            .placement = backend_webgpu,
            .device_handle = 77,
            .buffer_handle = 88,
            .byte_len = 0,
        }, &unsupported_device_buffer));
        try std.testing.expectEqual(@as(?*zgml_buffer, null), unsupported_device_buffer);

        var resource_weights: ?*zgml_buffer = null;
        var resource_bias: ?*zgml_buffer = null;
        var resource_input: ?*zgml_buffer = null;
        var resource_output: ?*zgml_buffer = null;
        defer zgml_buffer_free(resource_output);
        defer zgml_buffer_free(resource_input);
        defer zgml_buffer_free(resource_bias);
        defer zgml_buffer_free(resource_weights);
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 101,
            .byte_len = 4 * @sizeOf(f32),
        }, &resource_weights));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 102,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_bias));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_read,
            .handle = 103,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_input));
        try std.testing.expectEqual(status(.ok), zgml_buffer_wrap_resource(&.{
            .placement = backend_webgpu,
            .access_flags = resource_access_write,
            .handle = 104,
            .byte_len = 2 * @sizeOf(f32),
        }, &resource_output));

        try std.testing.expectEqual(status(.ok), zgml_session_bind_buffers(webgpu_linear_program, &.{
            .weights = resource_weights,
            .weights_len = 4,
            .bias = resource_bias,
            .bias_len = 2,
            .input = resource_input,
            .input_len = 2,
            .output = resource_output,
            .output_len = 2,
        }, &webgpu_session));
        defer zgml_session_free(webgpu_session);
        try std.testing.expect(webgpu_session != null);

        var webgpu_session_inspection = zgml_session_inspection{};
        try std.testing.expectEqual(status(.ok), zgml_session_inspect(webgpu_session, &webgpu_session_inspection));
        try std.testing.expectEqual(tiny_linear_kind, webgpu_session_inspection.model_kind);
        try std.testing.expectEqual(backend_webgpu, webgpu_session_inspection.backend);
        try std.testing.expectEqual(buffer_storage_external_resource, webgpu_session_inspection.output_storage);
        try std.testing.expectEqual(@as(u64, 2), webgpu_session_inspection.persistent_binding_count);
        try std.testing.expectEqual(@as(u64, 1), webgpu_session_inspection.step_input_count);
        try std.testing.expectEqual(@as(u64, 1), webgpu_session_inspection.step_output_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_session_inspection.host_binding_count);
        try std.testing.expectEqual(@as(u64, 4), webgpu_session_inspection.resource_binding_count);
        try std.testing.expect(webgpu_session_inspection.binding_shape_hash != 0);

        var webgpu_step_result = zgml_step_result{};
        try std.testing.expectEqual(status(.unsupported), zgml_session_step(webgpu_session, null, &webgpu_step_result));
        try std.testing.expectEqual(@as(usize, 0), webgpu_step_result.output_len);
        webgpu_step_result = .{ .output_len = 99 };
        try std.testing.expectEqual(status(.unsupported), zgml_session_step_no_output(webgpu_session, null, &webgpu_step_result));
        try std.testing.expectEqual(@as(usize, 99), webgpu_step_result.output_len);

        var webgpu_resource_profile = zgml_runtime_profile{};
        try std.testing.expectEqual(status(.ok), zgml_session_runtime_profile(webgpu_session, &webgpu_resource_profile));
        try std.testing.expectEqual(@as(u64, 0), webgpu_resource_profile.call_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_resource_profile.backend_op_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_resource_profile.fallback_op_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_resource_profile.sync_count);
        try std.testing.expectEqual(@as(u64, 0), webgpu_resource_profile.runtime_patch_call_count);
    }

    try std.testing.expectEqual(status(.ok), zgml_program_compile(good_model, &.{}, &program));

    const bad_weights = [_]f32{ 1, 2, 3 };
    var session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_session_bind(program, &.{
        .weights = bad_weights[0..].ptr,
        .weights_len = bad_weights.len,
    }, &session));
    try std.testing.expectEqual(@as(?*zgml_session, null), session);
}

test "C ABI validates model load path handles" {
    var model: ?*zgml_model = @ptrFromInt(1);
    const path = "missing-smollm.gguf";

    try std.testing.expectEqual(status(.unsupported), zgml_model_create(&.{
        .kind = smollm_135m_kind,
        .input_len = 0,
        .output_len = 0,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);

    model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_load_path(&.{
        .kind = smollm_135m_kind,
        .path = null,
        .path_len = 0,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);

    model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.compile_failed), zgml_model_load_path(&.{
        .kind = tiny_llama_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);

    model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.compile_failed), zgml_model_load_path(&.{
        .kind = auto_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);

    model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.compile_failed), zgml_model_load_path(&.{
        .kind = smollm_135m_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &model));
    try std.testing.expectEqual(@as(?*zgml_model, null), model);
}

fn appendSafetensorsTensorHeader(list: *std.ArrayList(u8), allocator: std.mem.Allocator, first: *bool, name: []const u8, shape: []const usize) !void {
    if (first.*) {
        first.* = false;
    } else {
        try list.append(allocator, ',');
    }
    try list.append(allocator, '"');
    try list.appendSlice(allocator, name);
    try list.appendSlice(allocator, "\":{\"dtype\":\"F16\",\"shape\":[");
    for (shape, 0..) |dim, i| {
        if (i != 0) try list.append(allocator, ',');
        var dim_buf: [32]u8 = undefined;
        const dim_str = try std.fmt.bufPrint(&dim_buf, "{d}", .{dim});
        try list.appendSlice(allocator, dim_str);
    }
    try list.appendSlice(allocator, "],\"data_offsets\":[0,0]}");
}

fn appendSafetensorsF32TensorHeader(
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    first: *bool,
    name: []const u8,
    shape: []const usize,
    data_offset: *usize,
) !void {
    if (first.*) {
        first.* = false;
    } else {
        try list.append(allocator, ',');
    }
    try list.append(allocator, '"');
    try list.appendSlice(allocator, name);
    try list.appendSlice(allocator, "\":{\"dtype\":\"F32\",\"shape\":[");
    var element_count: usize = 1;
    for (shape, 0..) |dim, i| {
        if (i != 0) try list.append(allocator, ',');
        element_count *= dim;
        var dim_buf: [32]u8 = undefined;
        const dim_str = try std.fmt.bufPrint(&dim_buf, "{d}", .{dim});
        try list.appendSlice(allocator, dim_str);
    }
    const start = data_offset.*;
    const end = start + element_count * @sizeOf(f32);
    data_offset.* = end;
    var range_buf: [64]u8 = undefined;
    const range = try std.fmt.bufPrint(&range_buf, "],\"data_offsets\":[{d},{d}]}}", .{ start, end });
    try list.appendSlice(allocator, range);
}

fn appendSmolLMSafetensorsLayerHeader(list: *std.ArrayList(u8), allocator: std.mem.Allocator, first: *bool, layer: usize) !void {
    const d_head = smollm_135m_config.d_model / smollm_135m_config.n_heads;
    const kv_dim = smollm_135m_config.n_kv_heads * d_head;
    const specs = [_]struct {
        suffix: []const u8,
        shape: []const usize,
    }{
        .{ .suffix = "self_attn.q_proj.weight", .shape = &.{ smollm_135m_config.d_model, smollm_135m_config.d_model } },
        .{ .suffix = "self_attn.k_proj.weight", .shape = &.{ kv_dim, smollm_135m_config.d_model } },
        .{ .suffix = "self_attn.v_proj.weight", .shape = &.{ kv_dim, smollm_135m_config.d_model } },
        .{ .suffix = "self_attn.o_proj.weight", .shape = &.{ smollm_135m_config.d_model, smollm_135m_config.d_model } },
        .{ .suffix = "mlp.gate_proj.weight", .shape = &.{ smollm_135m_config.d_ff, smollm_135m_config.d_model } },
        .{ .suffix = "mlp.up_proj.weight", .shape = &.{ smollm_135m_config.d_ff, smollm_135m_config.d_model } },
        .{ .suffix = "mlp.down_proj.weight", .shape = &.{ smollm_135m_config.d_model, smollm_135m_config.d_ff } },
        .{ .suffix = "input_layernorm.weight", .shape = &.{smollm_135m_config.d_model} },
        .{ .suffix = "post_attention_layernorm.weight", .shape = &.{smollm_135m_config.d_model} },
    };
    for (specs) |spec| {
        var name_buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer, spec.suffix });
        try appendSafetensorsTensorHeader(list, allocator, first, name, spec.shape);
    }
}

fn buildSmolLMSafetensorsHeader(allocator: std.mem.Allocator) ![]u8 {
    var list = std.ArrayList(u8).empty;
    errdefer list.deinit(allocator);

    try list.append(allocator, '{');
    var first = true;
    try appendSafetensorsTensorHeader(&list, allocator, &first, "model.embed_tokens.weight", &.{ smollm_135m_config.vocab_size, smollm_135m_config.d_model });
    try appendSafetensorsTensorHeader(&list, allocator, &first, "model.norm.weight", &.{smollm_135m_config.d_model});
    try appendSmolLMSafetensorsLayerHeader(&list, allocator, &first, 0);
    try appendSmolLMSafetensorsLayerHeader(&list, allocator, &first, smollm_135m_config.n_layers - 1);
    try list.append(allocator, '}');
    return list.toOwnedSlice(allocator);
}

fn buildSafetensorsFileBytes(allocator: std.mem.Allocator, header: []const u8) ![]u8 {
    const bytes = try allocator.alloc(u8, 8 + header.len);
    std.mem.writeInt(u64, bytes[0..8], @intCast(header.len), .little);
    @memcpy(bytes[8..], header);
    return bytes;
}

fn appendTinyLlamaSafetensorsLayerHeaders(
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    first: *bool,
    comptime config: llm_mod.LlamaConfig,
    layer: usize,
    data_offset: *usize,
) !void {
    const d_head = config.d_model / config.n_heads;
    const kv_dim = config.n_kv_heads * d_head;
    const specs = [_]struct {
        suffix: []const u8,
        shape: []const usize,
    }{
        .{ .suffix = "self_attn.q_proj.weight", .shape = &.{ config.d_model, config.d_model } },
        .{ .suffix = "self_attn.k_proj.weight", .shape = &.{ kv_dim, config.d_model } },
        .{ .suffix = "self_attn.v_proj.weight", .shape = &.{ kv_dim, config.d_model } },
        .{ .suffix = "self_attn.o_proj.weight", .shape = &.{ config.d_model, config.d_model } },
        .{ .suffix = "mlp.gate_proj.weight", .shape = &.{ config.d_ff, config.d_model } },
        .{ .suffix = "mlp.up_proj.weight", .shape = &.{ config.d_ff, config.d_model } },
        .{ .suffix = "mlp.down_proj.weight", .shape = &.{ config.d_model, config.d_ff } },
        .{ .suffix = "input_layernorm.weight", .shape = &.{config.d_model} },
        .{ .suffix = "post_attention_layernorm.weight", .shape = &.{config.d_model} },
    };
    for (specs) |spec| {
        var name_buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer, spec.suffix });
        try appendSafetensorsF32TensorHeader(list, allocator, first, name, spec.shape, data_offset);
    }
}

fn appendTinyLlamaSafetensorsHeadersForConfig(
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    first: *bool,
    comptime config: llm_mod.LlamaConfig,
    data_offset: *usize,
) !void {
    try appendSafetensorsF32TensorHeader(list, allocator, first, "model.embed_tokens.weight", &.{ config.vocab_size, config.d_model }, data_offset);
    try appendSafetensorsF32TensorHeader(list, allocator, first, "model.norm.weight", &.{config.d_model}, data_offset);
    try appendSafetensorsF32TensorHeader(list, allocator, first, "lm_head.weight", &.{ config.vocab_size, config.d_model }, data_offset);
    for (0..config.n_layers) |layer| {
        try appendTinyLlamaSafetensorsLayerHeaders(list, allocator, first, config, layer, data_offset);
    }
}

fn appendTinyLlamaSafetensorsHeaders(list: *std.ArrayList(u8), allocator: std.mem.Allocator, first: *bool, data_offset: *usize) !void {
    try appendTinyLlamaSafetensorsHeadersForConfig(list, allocator, first, tiny_llama_config, data_offset);
}

fn buildTinyLlamaSafetensorsFileBytesForConfig(allocator: std.mem.Allocator, comptime config: llm_mod.LlamaConfig) ![]u8 {
    var list = std.ArrayList(u8).empty;
    defer list.deinit(allocator);

    try list.append(allocator, '{');
    var first = true;
    var data_byte_len: usize = 0;
    try appendTinyLlamaSafetensorsHeadersForConfig(&list, allocator, &first, config, &data_byte_len);
    try list.append(allocator, '}');
    while ((8 + list.items.len) % @alignOf(f32) != 0) {
        try list.append(allocator, ' ');
    }

    const bytes = try allocator.alloc(u8, 8 + list.items.len + data_byte_len);
    std.mem.writeInt(u64, bytes[0..8], @intCast(list.items.len), .little);
    @memcpy(bytes[8..][0..list.items.len], list.items);
    @memset(bytes[8 + list.items.len ..], 0);
    return bytes;
}

fn buildTinyLlamaSafetensorsFileBytes(allocator: std.mem.Allocator) ![]u8 {
    return buildTinyLlamaSafetensorsFileBytesForConfig(allocator, tiny_llama_config);
}

fn buildTinyLlama2LayerSafetensorsFileBytes(allocator: std.mem.Allocator) ![]u8 {
    return buildTinyLlamaSafetensorsFileBytesForConfig(allocator, tiny_llama_2layer_config);
}

fn safetensorsViewForHeader(allocator: std.mem.Allocator, header: []const u8, raw: []align(4) u8) safetensors_mod.SafetensorsFile {
    return .{
        .alloc = allocator,
        .raw_data = raw,
        .header_json = header,
        .data_start = 0,
    };
}

fn expectLoadedTinyFamilyModelExecutes(comptime config: llm_mod.LlamaConfig, model_kind: u32, model: ?*zgml_model) !void {
    var inspection = zgml_model_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_model_inspect(model, &inspection));
    try std.testing.expectEqual(model_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, 0), inspection.tied_lm_head);

    var program: ?*zgml_program = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_cpu,
        .context_len = 4,
    }, &program));
    defer if (program) |handle| zgml_program_free(handle);

    var session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_session_bind(program, null, &session));
    defer if (session) |handle| zgml_session_free(handle);

    var logits = [_]f32{-123} ** (config.vocab_size + 1);
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session, &.{
        .token = 0,
        .output = logits[0..].ptr,
        .output_len = logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, config.vocab_size), result.output_len);
    try std.testing.expectEqual(@as(f32, -123), logits[config.vocab_size]);
    for (logits[0..config.vocab_size]) |logit| {
        try std.testing.expectEqual(@as(f32, 0), logit);
    }

    var position: usize = 99;
    try std.testing.expectEqual(status(.ok), zgml_session_position(session, &position));
    try std.testing.expectEqual(@as(usize, 1), position);
}

fn expectTinyLlamaLoadedModelExecutes(model: ?*zgml_model) !void {
    try expectLoadedTinyFamilyModelExecutes(tiny_llama_config, tiny_llama_kind, model);
}

fn expectTinyLlama2LayerLoadedModelExecutes(model: ?*zgml_model) !void {
    try expectLoadedTinyFamilyModelExecutes(tiny_llama_2layer_config, tiny_llama_2layer_kind, model);
}

test "C ABI compatible checkpoint selector accepts only compiled-in LLaMA envelopes" {
    try std.testing.expectEqual(@as(?u32, smollm_135m_kind), compatibleLlamaKindForConfig(smollm_135m_config));
    try std.testing.expectEqual(@as(?u32, tiny_llama_2layer_kind), compatibleLlamaKindForConfig(tiny_llama_2layer_config));

    var changed = smollm_135m_config;
    changed.d_model += 1;
    try std.testing.expectEqual(@as(?u32, null), compatibleLlamaKindForConfig(changed));

    changed = smollm_135m_config;
    changed.tied_lm_head = false;
    try std.testing.expectEqual(@as(?u32, null), compatibleLlamaKindForConfig(changed));

    const header = try buildSmolLMSafetensorsHeader(std.testing.allocator);
    defer std.testing.allocator.free(header);
    const raw = try std.testing.allocator.alignedAlloc(u8, .@"4", 1);
    defer std.testing.allocator.free(raw);
    var sf = safetensorsViewForHeader(std.testing.allocator, header, raw);
    try std.testing.expectEqual(@as(?u32, smollm_135m_kind), compatibleLlamaKindForSafetensors(&sf));

    const two_layer_bytes = try buildTinyLlama2LayerSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(two_layer_bytes);
    const two_layer_header = try safetensorsHeaderFromData(two_layer_bytes);
    var two_layer_sf = safetensorsViewForHeader(std.testing.allocator, two_layer_header, raw);
    try std.testing.expectEqual(@as(?u32, tiny_llama_2layer_kind), compatibleLlamaKindForSafetensors(&two_layer_sf));

    const incomplete_header =
        \\{"model.embed_tokens.weight":{"dtype":"F16","shape":[49152,576],"data_offsets":[0,0]}}
    ;
    var incomplete_sf = safetensorsViewForHeader(std.testing.allocator, incomplete_header, raw);
    try std.testing.expectEqual(@as(?u32, null), compatibleLlamaKindForSafetensors(&incomplete_sf));
}

test "C ABI model path probe preflights compatible checkpoint envelope" {
    var inspection = zgml_model_inspection{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_path(null, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    const header = try buildSmolLMSafetensorsHeader(std.testing.allocator);
    defer std.testing.allocator.free(header);
    const file_bytes = try buildSafetensorsFileBytes(std.testing.allocator, header);
    defer std.testing.allocator.free(file_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/model.safetensors", .{&tmp.sub_path});

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_path(&.{
        .kind = auto_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_ff), inspection.d_ff);
    try std.testing.expectEqual(@as(u64, 1), inspection.tied_lm_head);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_path(&.{
        .kind = smollm_135m_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_model_probe_path(&.{
        .kind = tiny_llama_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);
}

test "C ABI model load path executes exact tiny LLaMA safetensors checkpoint" {
    const file_bytes = try buildTinyLlamaSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(file_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "tiny.safetensors", .data = file_bytes });

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/tiny.safetensors", .{&tmp.sub_path});

    var rejected_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_model_load_path(&.{
        .kind = smollm_135m_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &rejected_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), rejected_model);

    var fixed_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_path(&.{
        .kind = tiny_llama_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &fixed_model));
    defer if (fixed_model) |handle| zgml_model_free(handle);

    var model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_path(&.{
        .kind = auto_kind,
        .path = path.ptr,
        .path_len = path.len,
    }, &model));
    defer if (model) |handle| zgml_model_free(handle);

    try expectTinyLlamaLoadedModelExecutes(model);
}

test "C ABI model load safetensors data executes exact tiny LLaMA checkpoint" {
    const file_bytes = try buildTinyLlamaSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(file_bytes);

    var invalid_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_load_safetensors_data(null, &invalid_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), invalid_model);

    invalid_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_load_safetensors_data(&.{
        .kind = auto_kind,
        .reserved = 1,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &invalid_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), invalid_model);

    invalid_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_load_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = 7,
    }, &invalid_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), invalid_model);

    var rejected_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_model_load_safetensors_data(&.{
        .kind = smollm_135m_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &rejected_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), rejected_model);

    var fixed_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_safetensors_data(&.{
        .kind = tiny_llama_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &fixed_model));
    defer if (fixed_model) |handle| zgml_model_free(handle);

    var model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &model));
    defer if (model) |handle| zgml_model_free(handle);

    try expectTinyLlamaLoadedModelExecutes(model);
}

test "C ABI model load safetensors data executes exact tiny two-layer LLaMA checkpoint with native KV binding" {
    const one_layer_bytes = try buildTinyLlamaSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(one_layer_bytes);
    const file_bytes = try buildTinyLlama2LayerSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(file_bytes);

    var rejected_one_layer_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_model_load_safetensors_data(&.{
        .kind = tiny_llama_2layer_kind,
        .data = one_layer_bytes.ptr,
        .data_len = one_layer_bytes.len,
    }, &rejected_one_layer_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), rejected_one_layer_model);

    var rejected_two_layer_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.unsupported), zgml_model_load_safetensors_data(&.{
        .kind = tiny_llama_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &rejected_two_layer_model));
    try std.testing.expectEqual(@as(?*zgml_model, null), rejected_two_layer_model);

    var fixed_model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_safetensors_data(&.{
        .kind = tiny_llama_2layer_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &fixed_model));
    defer if (fixed_model) |handle| zgml_model_free(handle);

    var model: ?*zgml_model = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_model_load_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &model));
    defer if (model) |handle| zgml_model_free(handle);

    try expectTinyLlama2LayerLoadedModelExecutes(model);

    var program: ?*zgml_program = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_program_compile(model, &.{
        .backend = backend_cpu,
        .context_len = 4,
    }, &program));
    defer if (program) |handle| zgml_program_free(handle);

    var requirements = zgml_program_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_program_get_requirements(program, &requirements));
    try std.testing.expectEqual(tiny_llama_2layer_kind, requirements.model_kind);
    try std.testing.expectEqual(@as(usize, tiny_llama_2layer_config.vocab_size), requirements.output_len);
    try std.testing.expectEqual(@as(usize, 4), requirements.context_len);

    var kv_requirements = zgml_llama_kv_cache_requirements{};
    try std.testing.expectEqual(status(.ok), zgml_llama_program_get_kv_cache_requirements(program, &kv_requirements));
    try std.testing.expectEqual(tiny_llama_2layer_kind, kv_requirements.model_kind);
    try std.testing.expectEqual(@as(u32, @intCast(tiny_llama_2layer_config.n_layers)), kv_requirements.n_layers);
    try std.testing.expectEqual(@as(usize, 4), kv_requirements.context_len);
    try std.testing.expectEqual(@as(usize, 64), kv_requirements.k_buffer_byte_len);
    try std.testing.expectEqual(@as(usize, 64), kv_requirements.v_buffer_byte_len);

    var output_buffer: ?*zgml_buffer = null;
    defer if (output_buffer) |handle| zgml_buffer_free(handle);
    try std.testing.expectEqual(status(.ok), zgml_program_create_output_buffer(program, &output_buffer));
    try std.testing.expectEqual(requirements.output_byte_len, zgml_buffer_size(output_buffer));

    var k_buffers: [2]?*zgml_buffer = .{ null, null };
    var v_buffers: [2]?*zgml_buffer = .{ null, null };
    defer {
        for (k_buffers) |buffer| if (buffer) |handle| zgml_buffer_free(handle);
        for (v_buffers) |buffer| if (buffer) |handle| zgml_buffer_free(handle);
    }
    for (0..tiny_llama_2layer_config.n_layers) |layer| {
        try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = kv_requirements.k_buffer_byte_len }, &k_buffers[layer]));
        try std.testing.expectEqual(status(.ok), zgml_buffer_create(&.{ .byte_len = kv_requirements.v_buffer_byte_len }, &v_buffers[layer]));
    }

    var bad_k_buffers = [_]?*zgml_buffer{k_buffers[0]};
    var bad_v_buffers = [_]?*zgml_buffer{v_buffers[0]};
    const bad_kv = zgml_llama_kv_cache_bind_desc{
        .k = bad_k_buffers[0..].ptr,
        .v = bad_v_buffers[0..].ptr,
        .len = bad_k_buffers.len,
    };
    var rejected_session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.shape_mismatch), zgml_llama_session_bind_buffers(program, &.{ .kv_cache = &bad_kv }, &rejected_session));
    try std.testing.expectEqual(@as(?*zgml_session, null), rejected_session);

    const kv = zgml_llama_kv_cache_bind_desc{
        .k = k_buffers[0..].ptr,
        .v = v_buffers[0..].ptr,
        .len = k_buffers.len,
    };
    var session: ?*zgml_session = @ptrFromInt(1);
    try std.testing.expectEqual(status(.ok), zgml_llama_session_bind_buffers(program, &.{
        .output = output_buffer,
        .output_len = tiny_llama_2layer_config.vocab_size,
        .kv_cache = &kv,
    }, &session));
    defer if (session) |handle| zgml_session_free(handle);

    var session_inspection = zgml_session_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_session_inspect(session, &session_inspection));
    try std.testing.expectEqual(tiny_llama_2layer_kind, session_inspection.model_kind);
    try std.testing.expectEqual(backend_cpu, session_inspection.backend);
    try std.testing.expectEqual(buffer_storage_host, session_inspection.output_storage);
    try std.testing.expectEqual(buffer_storage_host, session_inspection.kv_cache_storage);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.position);
    try std.testing.expectEqual(@as(u64, 4), session_inspection.context_len);
    try std.testing.expect(session_inspection.host_binding_count > 0);
    try std.testing.expectEqual(@as(u64, 0), session_inspection.resource_binding_count);
    try std.testing.expect(session_inspection.binding_shape_hash != 0);

    var logits = [_]f32{-2020} ** (tiny_llama_2layer_config.vocab_size + 1);
    var result = zgml_step_result{};
    try std.testing.expectEqual(status(.ok), zgml_session_step_token(session, &.{
        .token = 1,
        .output = logits[0..].ptr,
        .output_len = logits.len,
    }, &result));
    try std.testing.expectEqual(@as(usize, tiny_llama_2layer_config.vocab_size), result.output_len);
    try std.testing.expectEqual(@as(f32, -2020), logits[tiny_llama_2layer_config.vocab_size]);
}

test "C ABI safetensors data probe preflights exact tiny LLaMA checkpoint without loading weights" {
    const file_bytes = try buildTinyLlamaSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(file_bytes);
    const two_layer_bytes = try buildTinyLlama2LayerSafetensorsFileBytes(std.testing.allocator);
    defer std.testing.allocator.free(two_layer_bytes);

    var inspection = zgml_model_inspection{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_safetensors_data(null, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_safetensors_data(&.{
        .kind = auto_kind,
        .reserved = 1,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = 7,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_model_probe_safetensors_data(&.{
        .kind = smollm_135m_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_data(&.{
        .kind = tiny_llama_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(tiny_llama_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_kv_heads), inspection.n_kv_heads);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(tiny_llama_kind, inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_data(&.{
        .kind = auto_kind,
        .data = two_layer_bytes.ptr,
        .data_len = two_layer_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(tiny_llama_2layer_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.n_kv_heads), inspection.n_kv_heads);
}

test "C ABI safetensors data probe preflights SmolLM envelope from host-owned bytes" {
    const header = try buildSmolLMSafetensorsHeader(std.testing.allocator);
    defer std.testing.allocator.free(header);
    const file_bytes = try buildSafetensorsFileBytes(std.testing.allocator, header);
    defer std.testing.allocator.free(file_bytes);

    var inspection = zgml_model_inspection{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_data(&.{
        .kind = auto_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_ff), inspection.d_ff);
    try std.testing.expectEqual(@as(u64, 1), inspection.tied_lm_head);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_data(&.{
        .kind = smollm_135m_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_model_probe_safetensors_data(&.{
        .kind = tiny_llama_kind,
        .data = file_bytes.ptr,
        .data_len = file_bytes.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);
}

test "C ABI safetensors header probe preflights compatible checkpoint envelope without a file path" {
    const header = try buildSmolLMSafetensorsHeader(std.testing.allocator);
    defer std.testing.allocator.free(header);

    var inspection = zgml_model_inspection{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_safetensors_header(null, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_model_probe_safetensors_header(&.{
        .kind = auto_kind,
        .reserved = 1,
        .header = header.ptr,
        .header_len = header.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_header(&.{
        .kind = auto_kind,
        .header = header.ptr,
        .header_len = header.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, 1), inspection.tied_lm_head);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.ok), zgml_model_probe_safetensors_header(&.{
        .kind = smollm_135m_kind,
        .header = header.ptr,
        .header_len = header.len,
    }, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);

    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_model_probe_safetensors_header(&.{
        .kind = tiny_llama_kind,
        .header = header.ptr,
        .header_len = header.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);

    const incomplete_header =
        \\{"model.embed_tokens.weight":{"dtype":"F16","shape":[49152,576],"data_offsets":[0,0]}}
    ;
    inspection = .{ .model_kind = 99 };
    try std.testing.expectEqual(status(.unsupported), zgml_model_probe_safetensors_header(&.{
        .kind = auto_kind,
        .header = incomplete_header.ptr,
        .header_len = incomplete_header.len,
    }, &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);
}

test "C ABI supported checkpoint catalog exposes compiled-in envelopes" {
    try std.testing.expectEqual(compatible_llama_families.len, zgml_supported_checkpoint_count());

    var inspection = zgml_model_inspection{};
    try std.testing.expectEqual(status(.ok), zgml_supported_checkpoint_inspect(0, &inspection));
    try std.testing.expectEqual(tiny_llama_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_config.d_ff), inspection.d_ff);
    try std.testing.expectEqual(@as(u64, 0), inspection.tied_lm_head);

    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_supported_checkpoint_inspect(1, &inspection));
    try std.testing.expectEqual(tiny_llama_2layer_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, tiny_llama_2layer_config.d_ff), inspection.d_ff);
    try std.testing.expectEqual(@as(u64, 0), inspection.tied_lm_head);

    inspection = .{};
    try std.testing.expectEqual(status(.ok), zgml_supported_checkpoint_inspect(2, &inspection));
    try std.testing.expectEqual(smollm_135m_kind, inspection.model_kind);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.vocab_size), inspection.vocab_size);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.max_seq_len), inspection.max_seq_len);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_model), inspection.d_model);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_layers), inspection.n_layers);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_heads), inspection.n_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.n_kv_heads), inspection.n_kv_heads);
    try std.testing.expectEqual(@as(u64, smollm_135m_config.d_ff), inspection.d_ff);
    try std.testing.expectEqual(@as(u64, 1), inspection.tied_lm_head);

    inspection = .{ .model_kind = 99, .vocab_size = 123 };
    try std.testing.expectEqual(status(.invalid_argument), zgml_supported_checkpoint_inspect(zgml_supported_checkpoint_count(), &inspection));
    try std.testing.expectEqual(@as(u32, 0), inspection.model_kind);
    try std.testing.expectEqual(@as(u64, 0), inspection.vocab_size);
    try std.testing.expectEqual(status(.invalid_argument), zgml_supported_checkpoint_inspect(0, null));
}
