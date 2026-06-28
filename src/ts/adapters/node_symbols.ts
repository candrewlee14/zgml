"use strict";

type NativeFunction = (...args: any[]) => any;

type NativeLibrary = {
  func(signature: string): NativeFunction;
};

export type NodeNativeSymbols = Readonly<{
  abiStructSize: NativeFunction;
  getRuntimeInfo: NativeFunction;
  statusName: NativeFunction;
  modelCreate: NativeFunction;
  modelLoadPath: NativeFunction;
  modelLoadSafetensorsData: NativeFunction;
  modelProbePath: NativeFunction;
  modelProbeSafetensorsData: NativeFunction;
  modelProbeSafetensorsHeader: NativeFunction;
  supportedCheckpointCount: NativeFunction;
  supportedCheckpointInspect: NativeFunction;
  modelInspect: NativeFunction;
  programCompile: NativeFunction;
  moduleProgramCompile: NativeFunction;
  moduleProgramGetRequirements: NativeFunction;
  programGetRequirements: NativeFunction;
  programCheckModelCompatibility: NativeFunction;
  programCreateBuffer: NativeFunction;
  programCreateDeviceBuffer: NativeFunction;
  programGetDeviceHandle: NativeFunction;
  programImportDeviceBuffer: NativeFunction;
  programCreateOutputBuffer: NativeFunction;
  programInspect: NativeFunction;
  llamaProgramInspect: NativeFunction;
  llamaProgramGetKvCacheRequirements: NativeFunction;
  programRuntimeProfile: NativeFunction;
  programResetRuntimeProfile: NativeFunction;
  bufferCreate: NativeFunction;
  bufferWrap: NativeFunction;
  bufferWrapResource: NativeFunction;
  bufferSize: NativeFunction;
  bufferInspect: NativeFunction;
  bufferWrite: NativeFunction;
  bufferRead: NativeFunction;
  bufferFree: NativeFunction;
  eagerLinearF32: NativeFunction;
  eagerLinearTransposedWeightsF32: NativeFunction;
  eagerMatmulF32: NativeFunction;
  eagerBmmF32: NativeFunction;
  eagerLinearActivationF32: NativeFunction;
  eagerLinearActivationTransposedWeightsF32: NativeFunction;
  eagerActivationF32: NativeFunction;
  eagerElementwiseF32: NativeFunction;
  eagerElementwiseBroadcastRhsF32: NativeFunction;
  eagerElementwiseBroadcastLhsF32: NativeFunction;
  eagerWhereF32: NativeFunction;
  eagerClampF32: NativeFunction;
  eagerReduceF32: NativeFunction;
  eagerReduceDimF32: NativeFunction;
  eagerArgReduceDimF32: NativeFunction;
  eagerCumsumF32: NativeFunction;
  eagerMomentF32: NativeFunction;
  eagerDotF32: NativeFunction;
  eagerFullF32: NativeFunction;
  eagerArangeF32: NativeFunction;
  eagerPermuteF32: NativeFunction;
  eagerTakeF32: NativeFunction;
  eagerConv2dF32: NativeFunction;
  eagerPool2dF32: NativeFunction;
  eagerSoftmaxF32: NativeFunction;
  trainLinearMseSgdF32: NativeFunction;
  trainLinearMseSgdBulkF32: NativeFunction;
  trainingPlanF32: NativeFunction;
  trainMlpReluCrossEntropyAdamF32: NativeFunction;
  trainMlpReluCrossEntropyAdamWF32: NativeFunction;
  trainMlpReluCrossEntropyAdamBulkF32: NativeFunction;
  trainMlpReluCrossEntropyAdamWBulkF32: NativeFunction;
  sessionBind: NativeFunction;
  sessionBindModel: NativeFunction;
  sessionBindModelBuffers: NativeFunction;
  sessionBindBuffers: NativeFunction;
  llamaSessionBindModelBuffers: NativeFunction;
  llamaSessionBindBuffers: NativeFunction;
  sessionUploadPersistent: NativeFunction;
  sessionUploadPersistentRange: NativeFunction;
  sessionStep: NativeFunction;
  sessionStepDirect: NativeFunction;
  sessionStepNoOutput: NativeFunction;
  sessionStepToken: NativeFunction;
  sessionAdvanceToken: NativeFunction;
  sessionExecuteTokens: NativeFunction;
  sessionArgmaxToken: NativeFunction;
  sessionExecuteArgmaxTokens: NativeFunction;
  sessionGenerateArgmaxTokens: NativeFunction;
  sessionSampleToken: NativeFunction;
  sessionExecuteSampleTokens: NativeFunction;
  sessionGenerateSampleTokens: NativeFunction;
  sessionPosition: NativeFunction;
  sessionInspect: NativeFunction;
  sessionReset: NativeFunction;
  sessionRuntimeProfile: NativeFunction;
  sessionResetRuntimeProfile: NativeFunction;
  sessionFree: NativeFunction;
  programFree: NativeFunction;
  modelFree: NativeFunction;
}>;

function requireNativeLibrary(value: unknown): NativeLibrary {
  if (!value || typeof value !== "object" || typeof (value as { func?: unknown }).func !== "function") {
    throw new Error("zgml Node FFI expected koffi.load() to return a native library with func(signature)");
  }
  return value as NativeLibrary;
}

export function bindNodeSymbols(nativeLibrary: unknown): NodeNativeSymbols {
  const lib = requireNativeLibrary(nativeLibrary);
  return {
    abiStructSize: lib.func("size_t zgml_abi_struct_size(uint32_t kind)"),
    getRuntimeInfo: lib.func("int zgml_get_runtime_info(_Out_ zgml_runtime_info *out_info)"),
    statusName: lib.func("const char *zgml_status_name(int code)"),
    modelCreate: lib.func("int zgml_model_create(const zgml_model_desc *desc, _Out_ void **out_model)"),
    modelLoadPath: lib.func("int zgml_model_load_path(const zgml_model_load_desc *desc, _Out_ void **out_model)"),
    modelLoadSafetensorsData: lib.func("int zgml_model_load_safetensors_data(const zgml_safetensors_data_load_desc *desc, _Out_ void **out_model)"),
    modelProbePath: lib.func("int zgml_model_probe_path(const zgml_model_load_desc *desc, _Out_ zgml_model_inspection *out_inspection)"),
    modelProbeSafetensorsData: lib.func("int zgml_model_probe_safetensors_data(const zgml_safetensors_data_load_desc *desc, _Out_ zgml_model_inspection *out_inspection)"),
    modelProbeSafetensorsHeader: lib.func("int zgml_model_probe_safetensors_header(const zgml_safetensors_header_probe_desc *desc, _Out_ zgml_model_inspection *out_inspection)"),
    supportedCheckpointCount: lib.func("size_t zgml_supported_checkpoint_count(void)"),
    supportedCheckpointInspect: lib.func("int zgml_supported_checkpoint_inspect(size_t index, _Out_ zgml_model_inspection *out_inspection)"),
    modelInspect: lib.func("int zgml_model_inspect(void *model, _Out_ zgml_model_inspection *out_inspection)"),
    programCompile: lib.func("int zgml_program_compile(void *model, const zgml_compile_desc *desc, _Out_ void **out_program)"),
    moduleProgramCompile: lib.func("int zgml_module_program_compile(const zgml_module_desc *module_desc, const zgml_compile_desc *compile_desc, _Out_ void **out_program)"),
    moduleProgramGetRequirements: lib.func("int zgml_module_program_get_requirements(const zgml_module_desc *module_desc, const zgml_compile_desc *compile_desc, _Out_ zgml_program_requirements *out_requirements)"),
    programGetRequirements: lib.func("int zgml_program_get_requirements(void *program, _Out_ zgml_program_requirements *out_requirements)"),
    programCheckModelCompatibility: lib.func("int zgml_program_check_model_compatibility(void *program, void *model, _Out_ zgml_program_model_compatibility *out_compatibility)"),
    programCreateBuffer: lib.func("int zgml_program_create_buffer(void *program, uint32_t kind, _Out_ void **out_buffer)"),
    programCreateDeviceBuffer: lib.func("int zgml_program_create_device_buffer(void *program, uint32_t kind, uint32_t placement, _Out_ void **out_buffer)"),
    programGetDeviceHandle: lib.func("int zgml_program_get_device_handle(void *program, uint32_t placement, _Out_ size_t *out_handle)"),
    programImportDeviceBuffer: lib.func("int zgml_program_import_device_buffer(void *program, uint32_t kind, const zgml_device_buffer_import_desc *desc, _Out_ void **out_buffer)"),
    programCreateOutputBuffer: lib.func("int zgml_program_create_output_buffer(void *program, _Out_ void **out_buffer)"),
    programInspect: lib.func("int zgml_program_inspect(void *program, _Out_ zgml_program_inspection *out_inspection)"),
    llamaProgramInspect: lib.func("int zgml_llama_program_inspect(void *program, _Out_ zgml_llama_program_inspection *out_inspection)"),
    llamaProgramGetKvCacheRequirements: lib.func("int zgml_llama_program_get_kv_cache_requirements(void *program, _Out_ zgml_llama_kv_cache_requirements *out_requirements)"),
    programRuntimeProfile: lib.func("int zgml_program_runtime_profile(void *program, _Out_ zgml_runtime_profile *out_profile)"),
    programResetRuntimeProfile: lib.func("int zgml_program_reset_runtime_profile(void *program)"),
    bufferCreate: lib.func("int zgml_buffer_create(const zgml_buffer_desc *desc, _Out_ void **out_buffer)"),
    bufferWrap: lib.func("int zgml_buffer_wrap(void *data, size_t byte_len, _Out_ void **out_buffer)"),
    bufferWrapResource: lib.func("int zgml_buffer_wrap_resource(const zgml_external_resource_desc *desc, _Out_ void **out_buffer)"),
    bufferSize: lib.func("size_t zgml_buffer_size(void *buffer)"),
    bufferInspect: lib.func("int zgml_buffer_inspect(void *buffer, _Out_ zgml_buffer_inspection *out_inspection)"),
    bufferWrite: lib.func("int zgml_buffer_write(void *buffer, size_t byte_offset, const void *src, size_t byte_len)"),
    bufferRead: lib.func("int zgml_buffer_read(void *buffer, size_t byte_offset, void *dst, size_t byte_len)"),
    bufferFree: lib.func("void zgml_buffer_free(void *buffer)"),
    eagerLinearF32: lib.func("int zgml_eager_linear_f32(const float *input, size_t input_len, const float *weights, size_t weights_len, const float *bias, size_t bias_len, float *output, size_t output_len, size_t batch, size_t in_features, size_t out_features)"),
    eagerLinearTransposedWeightsF32: lib.func("int zgml_eager_linear_transposed_weights_f32(const float *input, size_t input_len, const float *weights, size_t weights_len, const float *bias, size_t bias_len, float *output, size_t output_len, size_t batch, size_t in_features, size_t out_features)"),
    eagerMatmulF32: lib.func("int zgml_eager_matmul_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len, size_t rows, size_t shared, size_t cols)"),
    eagerBmmF32: lib.func("int zgml_eager_bmm_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len, size_t batch, size_t rows, size_t shared, size_t cols)"),
    eagerLinearActivationF32: lib.func("int zgml_eager_linear_activation_f32(const float *input, size_t input_len, const float *weights, size_t weights_len, const float *bias, size_t bias_len, float *output, size_t output_len, size_t batch, size_t in_features, size_t out_features, uint32_t activation)"),
    eagerLinearActivationTransposedWeightsF32: lib.func("int zgml_eager_linear_activation_transposed_weights_f32(const float *input, size_t input_len, const float *weights, size_t weights_len, const float *bias, size_t bias_len, float *output, size_t output_len, size_t batch, size_t in_features, size_t out_features, uint32_t activation)"),
    eagerActivationF32: lib.func("int zgml_eager_activation_f32(const float *input, size_t input_len, float *output, size_t output_len, uint32_t activation)"),
    eagerElementwiseF32: lib.func("int zgml_eager_elementwise_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len, uint32_t op)"),
    eagerElementwiseBroadcastRhsF32: lib.func("int zgml_eager_elementwise_broadcast_rhs_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len, size_t rows, size_t cols, uint32_t op)"),
    eagerElementwiseBroadcastLhsF32: lib.func("int zgml_eager_elementwise_broadcast_lhs_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len, size_t rows, size_t cols, uint32_t op)"),
    eagerWhereF32: lib.func("int zgml_eager_where_f32(const float *condition, size_t condition_len, const float *input, size_t input_len, const float *other, size_t other_len, float *output, size_t output_len)"),
    eagerClampF32: lib.func("int zgml_eager_clamp_f32(const float *input, size_t input_len, float *output, size_t output_len, float min, float max, uint32_t has_min, uint32_t has_max)"),
    eagerReduceF32: lib.func("int zgml_eager_reduce_f32(const float *input, size_t input_len, float *output, size_t output_len, uint32_t op)"),
    eagerReduceDimF32: lib.func("int zgml_eager_reduce_dim_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t outer, size_t reduce, size_t inner, uint32_t op)"),
    eagerArgReduceDimF32: lib.func("int zgml_eager_arg_reduce_dim_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t outer, size_t reduce, size_t inner, uint32_t op)"),
    eagerCumsumF32: lib.func("int zgml_eager_cumsum_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t outer, size_t axis_len, size_t inner, uint32_t reverse)"),
    eagerMomentF32: lib.func("int zgml_eager_moment_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t outer, size_t reduce, size_t inner, float correction, uint32_t sqrt_output)"),
    eagerDotF32: lib.func("int zgml_eager_dot_f32(const float *lhs, size_t lhs_len, const float *rhs, size_t rhs_len, float *output, size_t output_len)"),
    eagerFullF32: lib.func("int zgml_eager_full_f32(float *output, size_t output_len, float value)"),
    eagerArangeF32: lib.func("int zgml_eager_arange_f32(float *output, size_t output_len, float start, float step)"),
    eagerPermuteF32: lib.func("int zgml_eager_permute_f32(const float *input, size_t input_len, float *output, size_t output_len, const uint32_t *output_shape, const uint32_t *input_strides, const uint32_t *axes, size_t rank)"),
    eagerTakeF32: lib.func("int zgml_eager_take_f32(const float *input, size_t input_len, const uint32_t *indices, size_t indices_len, float *output, size_t output_len)"),
    eagerConv2dF32: lib.func("int zgml_eager_conv2d_f32(const float *input, size_t input_len, const float *weights, size_t weights_len, const float *bias, size_t bias_len, float *output, size_t output_len, size_t batch, size_t in_channels, size_t height, size_t width, size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w, size_t dilation_h, size_t dilation_w, size_t out_h, size_t out_w)"),
    eagerPool2dF32: lib.func("int zgml_eager_pool2d_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t batch, size_t channels, size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w, size_t dilation_h, size_t dilation_w, size_t out_h, size_t out_w, uint32_t op, uint32_t ceil_mode, uint32_t count_include_pad)"),
    eagerSoftmaxF32: lib.func("int zgml_eager_softmax_f32(const float *input, size_t input_len, float *output, size_t output_len, size_t rows, size_t cols, uint32_t log_softmax)"),
    trainLinearMseSgdF32: lib.func("int zgml_train_linear_mse_sgd_f32(const float *input, size_t input_len, const float *target, size_t target_len, float *weight, size_t weight_len, float *bias, size_t bias_len, float *output, size_t output_len, float *grad_weight, size_t grad_weight_len, size_t batch, size_t in_features, size_t out_features, float lr, float weight_decay, float *out_loss)"),
    trainLinearMseSgdBulkF32: lib.func("int zgml_train_linear_mse_sgd_f32_bulk(const float *dataset_input, size_t dataset_input_len, const float *dataset_target, size_t dataset_target_len, const uint32_t *indices, size_t indices_len, float *batch_input, size_t batch_input_len, float *batch_target, size_t batch_target_len, float *weight, size_t weight_len, float *bias, size_t bias_len, float *output, size_t output_len, float *grad_weight, size_t grad_weight_len, size_t sample_count, size_t batch, size_t in_features, size_t out_features, size_t epochs, float lr, float weight_decay, float *out_loss, size_t *out_steps)"),
    trainingPlanF32: lib.func("int zgml_training_plan_f32(const zgml_training_plan_desc *desc, _Out_ zgml_training_plan *out_plan)"),
    trainMlpReluCrossEntropyAdamF32: lib.func("int zgml_train_mlp_relu_cross_entropy_adam_f32(const float *input, size_t input_len, const uint32_t *targets, size_t target_len, float *w1, size_t w1_len, float *b1, size_t b1_len, float *w2, size_t w2_len, float *b2, size_t b2_len, float *mw1, size_t mw1_len, float *vw1, size_t vw1_len, float *mb1, size_t mb1_len, float *vb1, size_t vb1_len, float *mw2, size_t mw2_len, float *vw2, size_t vw2_len, float *mb2, size_t mb2_len, float *vb2, size_t vb2_len, float *hidden, size_t hidden_len, float *logits, size_t logits_len, float *grad_hidden, size_t grad_hidden_len, float *grad_w1, size_t grad_w1_len, float *grad_w2, size_t grad_w2_len, size_t batch, size_t in_features, size_t hidden_features, size_t classes, size_t step, float lr, float beta1, float beta2, float eps, float weight_decay, float *out_loss, size_t *out_correct)"),
    trainMlpReluCrossEntropyAdamWF32: lib.func("int zgml_train_mlp_relu_cross_entropy_adamw_f32(const float *input, size_t input_len, const uint32_t *targets, size_t target_len, float *w1, size_t w1_len, float *b1, size_t b1_len, float *w2, size_t w2_len, float *b2, size_t b2_len, float *mw1, size_t mw1_len, float *vw1, size_t vw1_len, float *mb1, size_t mb1_len, float *vb1, size_t vb1_len, float *mw2, size_t mw2_len, float *vw2, size_t vw2_len, float *mb2, size_t mb2_len, float *vb2, size_t vb2_len, float *hidden, size_t hidden_len, float *logits, size_t logits_len, float *grad_hidden, size_t grad_hidden_len, float *grad_w1, size_t grad_w1_len, float *grad_w2, size_t grad_w2_len, size_t batch, size_t in_features, size_t hidden_features, size_t classes, size_t step, float lr, float beta1, float beta2, float eps, float weight_decay, float *out_loss, size_t *out_correct)"),
    trainMlpReluCrossEntropyAdamBulkF32: lib.func("int zgml_train_mlp_relu_cross_entropy_adam_f32_bulk(const float *dataset_input, size_t dataset_input_len, const uint32_t *dataset_targets, size_t dataset_target_len, const uint32_t *indices, size_t indices_len, float *batch_input, size_t batch_input_len, uint32_t *batch_targets, size_t batch_target_len, float *w1, size_t w1_len, float *b1, size_t b1_len, float *w2, size_t w2_len, float *b2, size_t b2_len, float *mw1, size_t mw1_len, float *vw1, size_t vw1_len, float *mb1, size_t mb1_len, float *vb1, size_t vb1_len, float *mw2, size_t mw2_len, float *vw2, size_t vw2_len, float *mb2, size_t mb2_len, float *vb2, size_t vb2_len, float *hidden, size_t hidden_len, float *logits, size_t logits_len, float *grad_hidden, size_t grad_hidden_len, float *grad_w1, size_t grad_w1_len, float *grad_w2, size_t grad_w2_len, size_t sample_count, size_t batch, size_t in_features, size_t hidden_features, size_t classes, size_t epochs, size_t start_step, float lr, float beta1, float beta2, float eps, float weight_decay, float *out_loss, size_t *out_correct, size_t *out_steps)"),
    trainMlpReluCrossEntropyAdamWBulkF32: lib.func("int zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk(const float *dataset_input, size_t dataset_input_len, const uint32_t *dataset_targets, size_t dataset_target_len, const uint32_t *indices, size_t indices_len, float *batch_input, size_t batch_input_len, uint32_t *batch_targets, size_t batch_target_len, float *w1, size_t w1_len, float *b1, size_t b1_len, float *w2, size_t w2_len, float *b2, size_t b2_len, float *mw1, size_t mw1_len, float *vw1, size_t vw1_len, float *mb1, size_t mb1_len, float *vb1, size_t vb1_len, float *mw2, size_t mw2_len, float *vw2, size_t vw2_len, float *mb2, size_t mb2_len, float *vb2, size_t vb2_len, float *hidden, size_t hidden_len, float *logits, size_t logits_len, float *grad_hidden, size_t grad_hidden_len, float *grad_w1, size_t grad_w1_len, float *grad_w2, size_t grad_w2_len, size_t sample_count, size_t batch, size_t in_features, size_t hidden_features, size_t classes, size_t epochs, size_t start_step, float lr, float beta1, float beta2, float eps, float weight_decay, float *out_loss, size_t *out_correct, size_t *out_steps)"),
    sessionBind: lib.func("int zgml_session_bind(void *program, const zgml_bind_desc *desc, _Out_ void **out_session)"),
    sessionBindModel: lib.func("int zgml_session_bind_model(void *program, void *model, const zgml_bind_desc *desc, _Out_ void **out_session)"),
    sessionBindModelBuffers: lib.func("int zgml_session_bind_model_buffers(void *program, void *model, const zgml_buffer_bind_desc *desc, _Out_ void **out_session)"),
    sessionBindBuffers: lib.func("int zgml_session_bind_buffers(void *program, const zgml_buffer_bind_desc *desc, _Out_ void **out_session)"),
    llamaSessionBindModelBuffers: lib.func("int zgml_llama_session_bind_model_buffers(void *program, void *model, const zgml_llama_buffer_bind_desc *desc, _Out_ void **out_session)"),
    llamaSessionBindBuffers: lib.func("int zgml_llama_session_bind_buffers(void *program, const zgml_llama_buffer_bind_desc *desc, _Out_ void **out_session)"),
    sessionUploadPersistent: lib.func("int zgml_session_upload_persistent(void *session)"),
    sessionUploadPersistentRange: lib.func("int zgml_session_upload_persistent_range(void *session, size_t first, size_t len)"),
    sessionStep: lib.func("int zgml_session_step(void *session, const zgml_step_desc *desc, _Out_ zgml_step_result *out_result)"),
    sessionStepDirect: lib.func("int zgml_session_step_direct(void *session, const float *input, size_t input_len, float *output, size_t output_len)"),
    sessionStepNoOutput: lib.func("int zgml_session_step_no_output(void *session, const zgml_step_desc *desc, _Out_ zgml_step_result *out_result)"),
    sessionStepToken: lib.func("int zgml_session_step_token(void *session, const zgml_token_step_desc *desc, _Out_ zgml_step_result *out_result)"),
    sessionAdvanceToken: lib.func("int zgml_session_advance_token(void *session, const zgml_token_advance_desc *desc)"),
    sessionExecuteTokens: lib.func("int zgml_session_execute_tokens(void *session, const zgml_token_execute_desc *desc, _Out_ zgml_step_result *out_result)"),
    sessionArgmaxToken: lib.func("int zgml_session_argmax_token(void *session, const zgml_token_argmax_desc *desc, _Out_ zgml_token_argmax_result *out_result)"),
    sessionExecuteArgmaxTokens: lib.func("int zgml_session_execute_argmax_tokens(void *session, const zgml_token_execute_argmax_desc *desc, _Out_ zgml_token_argmax_result *out_result)"),
    sessionGenerateArgmaxTokens: lib.func("int zgml_session_generate_argmax_tokens(void *session, const zgml_token_generate_argmax_desc *desc, _Out_ zgml_token_generate_argmax_result *out_result)"),
    sessionSampleToken: lib.func("int zgml_session_sample_token(void *session, const zgml_token_sample_desc *desc, _Out_ zgml_token_sample_result *out_result)"),
    sessionExecuteSampleTokens: lib.func("int zgml_session_execute_sample_tokens(void *session, const zgml_token_execute_sample_desc *desc, _Out_ zgml_token_sample_result *out_result)"),
    sessionGenerateSampleTokens: lib.func("int zgml_session_generate_sample_tokens(void *session, const zgml_token_generate_sample_desc *desc, _Out_ zgml_token_generate_sample_result *out_result)"),
    sessionPosition: lib.func("int zgml_session_position(void *session, _Out_ size_t *out_position)"),
    sessionInspect: lib.func("int zgml_session_inspect(void *session, _Out_ zgml_session_inspection *out_inspection)"),
    sessionReset: lib.func("int zgml_session_reset(void *session)"),
    sessionRuntimeProfile: lib.func("int zgml_session_runtime_profile(void *session, _Out_ zgml_runtime_profile *out_profile)"),
    sessionResetRuntimeProfile: lib.func("int zgml_session_reset_runtime_profile(void *session)"),
    sessionFree: lib.func("void zgml_session_free(void *session)"),
    programFree: lib.func("void zgml_program_free(void *program)"),
    modelFree: lib.func("void zgml_model_free(void *model)"),
  };
}
