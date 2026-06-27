"use strict";

import type {
  BunNativeSymbols,
} from "./bun_symbols.js";

export type BunSymbolGroups = ReturnType<typeof createBunSymbolGroups>;

export function createBunSymbolGroups(symbols: BunNativeSymbols) {
  return Object.freeze({
    inspection: Object.freeze({
      modelInspect: symbols.zgml_model_inspect,
      programCompile: symbols.zgml_program_compile,
      bufferInspect: symbols.zgml_buffer_inspect,
      sessionInspect: symbols.zgml_session_inspect,
      sessionPosition: symbols.zgml_session_position,
      sessionReset: symbols.zgml_session_reset,
      sessionRuntimeProfile: symbols.zgml_session_runtime_profile,
      sessionResetRuntimeProfile: symbols.zgml_session_reset_runtime_profile,
      programInspect: symbols.zgml_program_inspect,
      programGetRequirements: symbols.zgml_program_get_requirements,
      programCheckModelCompatibility: symbols.zgml_program_check_model_compatibility,
      llamaProgramInspect: symbols.zgml_llama_program_inspect,
      programRuntimeProfile: symbols.zgml_program_runtime_profile,
    }),
    modelSource: Object.freeze({
      modelCreate: symbols.zgml_model_create,
      modelLoadPath: symbols.zgml_model_load_path,
      modelLoadSafetensorsData: symbols.zgml_model_load_safetensors_data,
      modelProbePath: symbols.zgml_model_probe_path,
      modelProbeSafetensorsData: symbols.zgml_model_probe_safetensors_data,
      modelProbeSafetensorsHeader: symbols.zgml_model_probe_safetensors_header,
      supportedCheckpointCount: symbols.zgml_supported_checkpoint_count,
      supportedCheckpointInspect: symbols.zgml_supported_checkpoint_inspect,
    }),
    llamaSessionBind: Object.freeze({
      llamaProgramGetKvCacheRequirements: symbols.zgml_llama_program_get_kv_cache_requirements,
      llamaSessionBindBuffers: symbols.zgml_llama_session_bind_buffers,
      llamaSessionBindModelBuffers: symbols.zgml_llama_session_bind_model_buffers,
      sessionBindBuffers: symbols.zgml_session_bind_buffers,
      sessionBindModelBuffers: symbols.zgml_session_bind_model_buffers,
      sessionBind: symbols.zgml_session_bind,
      sessionBindModel: symbols.zgml_session_bind_model,
    }),
    moduleProgram: Object.freeze({
      moduleProgramCompile: symbols.zgml_module_program_compile,
    }),
    llamaToken: Object.freeze({
      sessionStepToken: symbols.zgml_session_step_token,
      sessionAdvanceToken: symbols.zgml_session_advance_token,
      sessionExecuteTokens: symbols.zgml_session_execute_tokens,
      sessionArgmaxToken: symbols.zgml_session_argmax_token,
      sessionExecuteArgmaxTokens: symbols.zgml_session_execute_argmax_tokens,
      sessionGenerateArgmaxTokens: symbols.zgml_session_generate_argmax_tokens,
      sessionSampleToken: symbols.zgml_session_sample_token,
      sessionExecuteSampleTokens: symbols.zgml_session_execute_sample_tokens,
      sessionGenerateSampleTokens: symbols.zgml_session_generate_sample_tokens,
    }),
    session: Object.freeze({
      sessionUploadPersistent: symbols.zgml_session_upload_persistent,
      sessionUploadPersistentRange: symbols.zgml_session_upload_persistent_range,
      sessionStep: symbols.zgml_session_step,
      sessionStepNoOutput: symbols.zgml_session_step_no_output,
    }),
    nativeLifecycle: Object.freeze({
      zgml_session_free: symbols.zgml_session_free,
      zgml_program_reset_runtime_profile: symbols.zgml_program_reset_runtime_profile,
      zgml_program_free: symbols.zgml_program_free,
      zgml_model_free: symbols.zgml_model_free,
    }),
    nativeBuffer: Object.freeze({
      zgml_buffer_create: symbols.zgml_buffer_create,
      zgml_buffer_wrap: symbols.zgml_buffer_wrap,
      zgml_buffer_wrap_resource: symbols.zgml_buffer_wrap_resource,
      zgml_buffer_size: symbols.zgml_buffer_size,
      zgml_buffer_write: symbols.zgml_buffer_write,
      zgml_buffer_read: symbols.zgml_buffer_read,
      zgml_buffer_free: symbols.zgml_buffer_free,
    }),
    nativeEager: Object.freeze({
      eagerLinearF32: symbols.zgml_eager_linear_f32,
      eagerMatmulF32: symbols.zgml_eager_matmul_f32,
      eagerLinearActivationF32: symbols.zgml_eager_linear_activation_f32,
      eagerActivationF32: symbols.zgml_eager_activation_f32,
      eagerElementwiseF32: symbols.zgml_eager_elementwise_f32,
      eagerReduceF32: symbols.zgml_eager_reduce_f32,
      eagerSoftmaxF32: symbols.zgml_eager_softmax_f32,
    }),
    nativeTraining: Object.freeze({
      trainLinearMseSgdF32: symbols.zgml_train_linear_mse_sgd_f32,
      trainMlpReluCrossEntropyAdamF32: symbols.zgml_train_mlp_relu_cross_entropy_adam_f32,
      trainMlpReluCrossEntropyAdamWF32: symbols.zgml_train_mlp_relu_cross_entropy_adamw_f32,
    }),
    programBind: Object.freeze({
      sessionBind: symbols.zgml_session_bind,
      sessionBindBuffers: symbols.zgml_session_bind_buffers,
    }),
    programBuffer: Object.freeze({
      programCreateBuffer: symbols.zgml_program_create_buffer,
      programCreateDeviceBuffer: symbols.zgml_program_create_device_buffer,
      programGetDeviceHandle: symbols.zgml_program_get_device_handle,
      programImportDeviceBuffer: symbols.zgml_program_import_device_buffer,
      bufferSize: symbols.zgml_buffer_size,
    }),
    genericFamily: Object.freeze({
      modelCreate: symbols.zgml_model_create,
    }),
  });
}
