"use strict";

import {
  dlopen,
  FFIType,
} from "./bun_ffi_intrinsics.js";

export type BunNativeHandle = number;

type BunTrainMlpReluCrossEntropyAdamF32 = (
  input: Float32Array,
  inputLen: bigint,
  targets: Uint32Array,
  targetLen: bigint,
  w1: Float32Array,
  w1Len: bigint,
  b1: Float32Array,
  b1Len: bigint,
  w2: Float32Array,
  w2Len: bigint,
  b2: Float32Array,
  b2Len: bigint,
  mw1: Float32Array,
  mw1Len: bigint,
  vw1: Float32Array,
  vw1Len: bigint,
  mb1: Float32Array,
  mb1Len: bigint,
  vb1: Float32Array,
  vb1Len: bigint,
  mw2: Float32Array,
  mw2Len: bigint,
  vw2: Float32Array,
  vw2Len: bigint,
  mb2: Float32Array,
  mb2Len: bigint,
  vb2: Float32Array,
  vb2Len: bigint,
  hidden: Float32Array,
  hiddenLen: bigint,
  logits: Float32Array,
  logitsLen: bigint,
  gradHidden: Float32Array,
  gradHiddenLen: bigint,
  gradW1: Float32Array,
  gradW1Len: bigint,
  gradW2: Float32Array,
  gradW2Len: bigint,
  batch: bigint,
  inFeatures: bigint,
  hiddenFeatures: bigint,
  classes: bigint,
  step: bigint,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
  outLoss: Float32Array,
  outCorrect: BigUint64Array,
) => number;

type BunTrainMlpReluCrossEntropyAdamBulkF32 = (...args: any[]) => number;

type BunTrainLinearMseSgdF32 = (
  input: Float32Array,
  inputLen: bigint,
  target: Float32Array,
  targetLen: bigint,
  weight: Float32Array,
  weightLen: bigint,
  bias: Float32Array,
  biasLen: bigint,
  output: Float32Array,
  outputLen: bigint,
  gradWeight: Float32Array,
  gradWeightLen: bigint,
  batch: bigint,
  inFeatures: bigint,
  outFeatures: bigint,
  lr: number,
  weightDecay: number,
  outLoss: Float32Array,
) => number;

type BunTrainLinearMseSgdBulkF32 = (...args: any[]) => number;

export type BunNativeSymbols = Readonly<{
  zgml_abi_struct_size(kind: number): bigint;
  zgml_get_runtime_info(outInfo: BigUint64Array): number;
  zgml_status_name(code: number): string;
  zgml_model_create(desc: Uint8Array | BigUint64Array | DataView, outModel: BigUint64Array): number;
  zgml_model_load_path(desc: BigUint64Array, outModel: BigUint64Array): number;
  zgml_model_load_safetensors_data(desc: BigUint64Array, outModel: BigUint64Array): number;
  zgml_model_probe_path(desc: BigUint64Array, outInspection: BigUint64Array): number;
  zgml_model_probe_safetensors_data(desc: BigUint64Array, outInspection: BigUint64Array): number;
  zgml_model_probe_safetensors_header(desc: BigUint64Array, outInspection: BigUint64Array): number;
  zgml_supported_checkpoint_count(): bigint;
  zgml_supported_checkpoint_inspect(index: bigint, outInspection: BigUint64Array): number;
  zgml_model_inspect(model: BunNativeHandle, outInspection: BigUint64Array): number;
  zgml_program_compile(model: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, outProgram: BigUint64Array): number;
  zgml_module_program_compile(moduleDesc: BigUint64Array, compileDesc: BunNativeHandle | BigUint64Array, outProgram: BigUint64Array): number;
  zgml_module_program_get_requirements(moduleDesc: BigUint64Array, compileDesc: BunNativeHandle | BigUint64Array, outRequirements: BigUint64Array): number;
  zgml_program_get_requirements(program: BunNativeHandle, outRequirements: BigUint64Array): number;
  zgml_program_check_model_compatibility(program: BunNativeHandle, model: BunNativeHandle, outCompatibility: BigUint64Array): number;
  zgml_program_create_buffer(program: BunNativeHandle, kind: number, outBuffer: BigUint64Array): number;
  zgml_program_create_device_buffer(program: BunNativeHandle, kind: number, placement: number, outBuffer: BigUint64Array): number;
  zgml_program_get_device_handle(program: BunNativeHandle, placement: number, outHandle: BigUint64Array): number;
  zgml_program_import_device_buffer(program: BunNativeHandle, kind: number, desc: BigUint64Array, outBuffer: BigUint64Array): number;
  zgml_program_create_output_buffer(program: BunNativeHandle, outBuffer: BigUint64Array): number;
  zgml_program_inspect(program: BunNativeHandle, outInspection: BigUint64Array): number;
  zgml_llama_program_inspect(program: BunNativeHandle, outInspection: BigUint64Array): number;
  zgml_llama_program_get_kv_cache_requirements(program: BunNativeHandle, outRequirements: BigUint64Array): number;
  zgml_program_runtime_profile(program: BunNativeHandle, outProfile: BigUint64Array): number;
  zgml_program_reset_runtime_profile(program: BunNativeHandle): number;
  zgml_buffer_create(desc: BigUint64Array, outBuffer: BigUint64Array): number;
  zgml_buffer_wrap(data: BunNativeHandle, byteLen: bigint, outBuffer: BigUint64Array): number;
  zgml_buffer_wrap_resource(desc: BigUint64Array, outBuffer: BigUint64Array): number;
  zgml_buffer_size(buffer: BunNativeHandle): bigint;
  zgml_buffer_inspect(buffer: BunNativeHandle, outInspection: BigUint64Array): number;
  zgml_buffer_write(buffer: BunNativeHandle, byteOffset: bigint, src: Float32Array | Uint8Array, byteLen: bigint): number;
  zgml_buffer_read(buffer: BunNativeHandle, byteOffset: bigint, dst: Float32Array | Uint8Array, byteLen: bigint): number;
  zgml_buffer_free(buffer: BunNativeHandle): void;
  zgml_session_bind(program: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, outSession: BigUint64Array): number;
  zgml_session_bind_model(program: BunNativeHandle, model: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, outSession: BigUint64Array): number;
  zgml_session_bind_model_buffers(program: BunNativeHandle, model: BunNativeHandle, desc: BigUint64Array, outSession: BigUint64Array): number;
  zgml_session_bind_buffers(program: BunNativeHandle, desc: BigUint64Array, outSession: BigUint64Array): number;
  zgml_llama_session_bind_model_buffers(program: BunNativeHandle, model: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, outSession: BigUint64Array): number;
  zgml_llama_session_bind_buffers(program: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, outSession: BigUint64Array): number;
  zgml_session_upload_persistent(session: BunNativeHandle): number;
  zgml_session_upload_persistent_range(session: BunNativeHandle, first: bigint, len: bigint): number;
  zgml_session_step(session: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, result: BigUint64Array): number;
  zgml_session_step_no_output(session: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, result: BigUint64Array): number;
  zgml_session_step_token(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_advance_token(session: BunNativeHandle, desc: BigUint64Array): number;
  zgml_session_execute_tokens(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_argmax_token(session: BunNativeHandle, desc: BunNativeHandle | BigUint64Array, result: BigUint64Array): number;
  zgml_session_execute_argmax_tokens(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_generate_argmax_tokens(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_sample_token(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_execute_sample_tokens(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_generate_sample_tokens(session: BunNativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  zgml_session_position(session: BunNativeHandle, outPosition: BigUint64Array): number;
  zgml_session_inspect(session: BunNativeHandle, outInspection: BigUint64Array): number;
  zgml_session_reset(session: BunNativeHandle): number;
  zgml_session_runtime_profile(session: BunNativeHandle, outProfile: BigUint64Array): number;
  zgml_session_reset_runtime_profile(session: BunNativeHandle): number;
  zgml_eager_linear_f32(
    input: Float32Array,
    inputLen: bigint,
    weights: Float32Array,
    weightsLen: bigint,
    bias: Float32Array | null,
    biasLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    inFeatures: bigint,
    outFeatures: bigint,
  ): number;
  zgml_eager_linear_transposed_weights_f32(
    input: Float32Array,
    inputLen: bigint,
    weights: Float32Array,
    weightsLen: bigint,
    bias: Float32Array | null,
    biasLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    inFeatures: bigint,
    outFeatures: bigint,
  ): number;
  zgml_eager_matmul_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    rows: bigint,
    shared: bigint,
    cols: bigint,
  ): number;
  zgml_eager_bmm_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    rows: bigint,
    shared: bigint,
    cols: bigint,
  ): number;
  zgml_eager_linear_activation_f32(
    input: Float32Array,
    inputLen: bigint,
    weights: Float32Array,
    weightsLen: bigint,
    bias: Float32Array | null,
    biasLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    inFeatures: bigint,
    outFeatures: bigint,
    activation: number,
  ): number;
  zgml_eager_linear_activation_transposed_weights_f32(
    input: Float32Array,
    inputLen: bigint,
    weights: Float32Array,
    weightsLen: bigint,
    bias: Float32Array | null,
    biasLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    inFeatures: bigint,
    outFeatures: bigint,
    activation: number,
  ): number;
  zgml_eager_activation_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    activation: number,
  ): number;
  zgml_eager_elementwise_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array | null,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    op: number,
  ): number;
  zgml_eager_elementwise_broadcast_rhs_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    rows: bigint,
    cols: bigint,
    op: number,
  ): number;
  zgml_eager_elementwise_broadcast_lhs_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    rows: bigint,
    cols: bigint,
    op: number,
  ): number;
  zgml_eager_where_f32(
    condition: Float32Array,
    conditionLen: bigint,
    input: Float32Array,
    inputLen: bigint,
    other: Float32Array,
    otherLen: bigint,
    output: Float32Array,
    outputLen: bigint,
  ): number;
  zgml_eager_clamp_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    min: number,
    max: number,
    hasMin: number,
    hasMax: number,
  ): number;
  zgml_eager_reduce_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    op: number,
  ): number;
  zgml_eager_reduce_dim_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    outer: bigint,
    reduce: bigint,
    inner: bigint,
    op: number,
  ): number;
  zgml_eager_arg_reduce_dim_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    outer: bigint,
    reduce: bigint,
    inner: bigint,
    op: number,
  ): number;
  zgml_eager_dot_f32(
    lhs: Float32Array,
    lhsLen: bigint,
    rhs: Float32Array,
    rhsLen: bigint,
    output: Float32Array,
    outputLen: bigint,
  ): number;
  zgml_eager_conv2d_f32(
    input: Float32Array,
    inputLen: bigint,
    weights: Float32Array,
    weightsLen: bigint,
    bias: Float32Array | null,
    biasLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    inChannels: bigint,
    height: bigint,
    width: bigint,
    outChannels: bigint,
    kernelH: bigint,
    kernelW: bigint,
    strideH: bigint,
    strideW: bigint,
    paddingH: bigint,
    paddingW: bigint,
    dilationH: bigint,
    dilationW: bigint,
    outH: bigint,
    outW: bigint,
  ): number;
  zgml_eager_pool2d_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    batch: bigint,
    channels: bigint,
    height: bigint,
    width: bigint,
    kernelH: bigint,
    kernelW: bigint,
    strideH: bigint,
    strideW: bigint,
    paddingH: bigint,
    paddingW: bigint,
    dilationH: bigint,
    dilationW: bigint,
    outH: bigint,
    outW: bigint,
    op: number,
    ceilMode: number,
    countIncludePad: number,
  ): number;
  zgml_eager_softmax_f32(
    input: Float32Array,
    inputLen: bigint,
    output: Float32Array,
    outputLen: bigint,
    rows: bigint,
    cols: bigint,
    logSoftmax: number,
  ): number;
  zgml_training_plan_f32(desc: BigUint64Array, outPlan: BigUint64Array): number;
  zgml_train_linear_mse_sgd_f32: BunTrainLinearMseSgdF32;
  zgml_train_linear_mse_sgd_f32_bulk: BunTrainLinearMseSgdBulkF32;
  zgml_train_mlp_relu_cross_entropy_adam_f32: BunTrainMlpReluCrossEntropyAdamF32;
  zgml_train_mlp_relu_cross_entropy_adamw_f32: BunTrainMlpReluCrossEntropyAdamF32;
  zgml_train_mlp_relu_cross_entropy_adam_f32_bulk: BunTrainMlpReluCrossEntropyAdamBulkF32;
  zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk: BunTrainMlpReluCrossEntropyAdamBulkF32;
  zgml_session_free(session: BunNativeHandle): void;
  zgml_program_free(program: BunNativeHandle): void;
  zgml_model_free(model: BunNativeHandle): void;
}>;

export function bindBunSymbols(libPath: string): BunNativeSymbols {
  const ptrLen = [FFIType.ptr, FFIType.u64] as const;
  const trainMlpReluCrossEntropyAdamArgs = [
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen,
    FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64,
    FFIType.u64,
    FFIType.float, FFIType.float, FFIType.float, FFIType.float, FFIType.float,
    FFIType.ptr,
    FFIType.ptr,
  ];
  const trainMlpReluCrossEntropyAdamBulkArgs = [
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    ...ptrLen, ...ptrLen, ...ptrLen, ...ptrLen,
    FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64,
    FFIType.u64, FFIType.u64,
    FFIType.float, FFIType.float, FFIType.float, FFIType.float, FFIType.float,
    FFIType.ptr,
    FFIType.ptr,
    FFIType.ptr,
  ];
  const dylib = dlopen<BunNativeSymbols>(libPath, {
    zgml_abi_struct_size: {
      args: [FFIType.u32],
      returns: FFIType.u64,
    },
    zgml_get_runtime_info: {
      args: [FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_status_name: {
      args: [FFIType.i32],
      returns: FFIType.cstring,
    },
    zgml_model_create: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_load_path: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_load_safetensors_data: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_probe_path: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_probe_safetensors_data: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_probe_safetensors_header: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_supported_checkpoint_count: {
      args: [],
      returns: FFIType.u64,
    },
    zgml_supported_checkpoint_inspect: {
      args: [FFIType.u64, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_model_inspect: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_compile: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_module_program_compile: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_module_program_get_requirements: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_get_requirements: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_check_model_compatibility: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_create_output_buffer: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_create_buffer: {
      args: [FFIType.ptr, FFIType.u32, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_create_device_buffer: {
      args: [FFIType.ptr, FFIType.u32, FFIType.u32, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_get_device_handle: {
      args: [FFIType.ptr, FFIType.u32, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_import_device_buffer: {
      args: [FFIType.ptr, FFIType.u32, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_inspect: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_llama_program_inspect: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_llama_program_get_kv_cache_requirements: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_runtime_profile: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_program_reset_runtime_profile: {
      args: [FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_buffer_create: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_buffer_wrap: {
      args: [FFIType.ptr, FFIType.u64, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_buffer_wrap_resource: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_buffer_size: {
      args: [FFIType.ptr],
      returns: FFIType.u64,
    },
    zgml_buffer_inspect: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_buffer_write: {
      args: [FFIType.ptr, FFIType.u64, FFIType.ptr, FFIType.u64],
      returns: FFIType.i32,
    },
    zgml_buffer_read: {
      args: [FFIType.ptr, FFIType.u64, FFIType.ptr, FFIType.u64],
      returns: FFIType.i32,
    },
    zgml_buffer_free: {
      args: [FFIType.ptr],
      returns: "void",
    },
    zgml_session_bind: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_bind_model: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_bind_model_buffers: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_bind_buffers: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_llama_session_bind_model_buffers: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_llama_session_bind_buffers: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_upload_persistent: {
      args: [FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_upload_persistent_range: {
      args: [FFIType.ptr, FFIType.u64, FFIType.u64],
      returns: FFIType.i32,
    },
    zgml_session_step: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_step_no_output: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_step_token: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_advance_token: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_execute_tokens: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_argmax_token: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_execute_argmax_tokens: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_generate_argmax_tokens: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_sample_token: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_execute_sample_tokens: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_generate_sample_tokens: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_position: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_inspect: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_reset: {
      args: [FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_runtime_profile: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_session_reset_runtime_profile: {
      args: [FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_eager_linear_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_linear_transposed_weights_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_matmul_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_bmm_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_linear_activation_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_linear_activation_transposed_weights_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_activation_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_elementwise_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_elementwise_broadcast_rhs_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_elementwise_broadcast_lhs_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_where_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_clamp_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.float,
        FFIType.float,
        FFIType.u32,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_reduce_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_reduce_dim_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_arg_reduce_dim_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_dot_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_conv2d_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_pool2d_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
        FFIType.u32,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_eager_softmax_f32: {
      args: [
        FFIType.ptr,
        FFIType.u64,
        FFIType.ptr,
        FFIType.u64,
        FFIType.u64,
        FFIType.u64,
        FFIType.u32,
      ],
      returns: FFIType.i32,
    },
    zgml_train_linear_mse_sgd_f32: {
      args: [
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.u64, FFIType.u64, FFIType.u64,
        FFIType.float, FFIType.float,
        FFIType.ptr,
      ],
      returns: FFIType.i32,
    },
    zgml_train_linear_mse_sgd_f32_bulk: {
      args: [
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.ptr, FFIType.u64,
        FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64,
        FFIType.float, FFIType.float,
        FFIType.ptr,
        FFIType.ptr,
      ],
      returns: FFIType.i32,
    },
    zgml_training_plan_f32: {
      args: [FFIType.ptr, FFIType.ptr],
      returns: FFIType.i32,
    },
    zgml_train_mlp_relu_cross_entropy_adam_f32: {
      args: trainMlpReluCrossEntropyAdamArgs,
      returns: FFIType.i32,
    },
    zgml_train_mlp_relu_cross_entropy_adamw_f32: {
      args: trainMlpReluCrossEntropyAdamArgs,
      returns: FFIType.i32,
    },
    zgml_train_mlp_relu_cross_entropy_adam_f32_bulk: {
      args: trainMlpReluCrossEntropyAdamBulkArgs,
      returns: FFIType.i32,
    },
    zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk: {
      args: trainMlpReluCrossEntropyAdamBulkArgs,
      returns: FFIType.i32,
    },
    zgml_session_free: {
      args: [FFIType.ptr],
      returns: "void",
    },
    zgml_program_free: {
      args: [FFIType.ptr],
      returns: "void",
    },
    zgml_model_free: {
      args: [FFIType.ptr],
      returns: "void",
    },
  });
  return dylib.symbols;
}
