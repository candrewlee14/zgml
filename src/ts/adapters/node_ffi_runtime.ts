// @ts-nocheck

const {
  createNodeFfiBootstrap,
} = require("./node_ffi_bootstrap.js");
const {
  createNodeSymbolGroups,
} = require("./node_symbol_groups.js");
const {
  createNodeLlamaTokenOps,
} = require("./node_llama_token_ops.js");
const {
  createNodeModelSourceOps,
} = require("./node_model_source_ops.js");
const {
  createNodeModelSourceBytes,
} = require("./node_model_source_bytes.js");
const {
  createAdapterModelSourceSurface,
} = require("./model_source_surface.js");
const {
  createAdapterModelSourceFacade,
} = require("./model_source_facade.js");
const {
  createAdapterModelHandlePolicy,
} = require("./model_handle_policy.js");
const {
  createNodeInspectionOps,
} = require("./node_inspection_ops.js");
const {
  createNodeLlamaSessionBindOps,
} = require("./node_llama_session_bind_ops.js");
const {
  createNodeModuleProgramOps,
} = require("./node_module_program_ops.js");
const {
  createAdapterProgramFactory,
} = require("./program_factory_surface.js");
const {
  createAdapterModuleCompilerSurface,
} = require("./module_compiler_surface.js");
const {
  createAdapterModuleStateSurface,
} = require("./module_state_surface.js");
const {
  createAdapterProgramRuntimeSurface,
} = require("./program_runtime_surface.js");
const {
  createNodeLlamaKvCacheOps,
} = require("./node_llama_kv_cache_ops.js");
const {
  createNodeLlamaKvCacheClass,
} = require("./node_llama_kv_cache_surface.js");
const {
  createNodeLlamaKvCacheFactorySurface,
} = require("./node_llama_kv_cache_factory_surface.js");
const {
  createNodeProgramBufferOps,
} = require("./node_program_buffer_ops.js");
const {
  createAdapterProgramBufferNativeBridge,
} = require("./program_buffer_native_bridge.js");
const {
  createAdapterProgramBufferFactorySurface,
} = require("./program_buffer_factory_surface.js");
const {
  createNodeSessionOps,
} = require("./node_session_ops.js");
const {
  createNodeProgramBindOps,
} = require("./node_program_bind_ops.js");
const {
  createAdapterSessionFactory,
} = require("./session_factory_surface.js");
const {
  createAdapterSessionRuntimeSurface,
} = require("./session_runtime_surface.js");
const {
  createAdapterLlamaSessionFacadeSurface,
} = require("./llama_session_facade_surface.js");
const {
  createAdapterTensorRuntimeSurface,
} = require("./tensor_runtime_surface.js");
const {
  createAdapterLlamaFamilySurface,
} = require("./llama_family_surface.js");
const {
  createNodeGenericFamilySurface,
} = require("./node_generic_family_surface.js");
const {
  NativeBuffer: AdapterNativeBuffer,
  setAdapterNativeBufferFacade,
} = require("./native_buffer_surface.js");
const {
  Tensor,
  setAdapterTensorSurfaceHelpers,
} = require("./tensor_surface.js");
const {
  createAdapterPublicRuntimeExports,
} = require("./public_runtime_exports_surface.js");
const {
  createAdapterCompileNamespace,
  createAdapterFrontendNamespaces,
  createAdapterZgmlCheckpointIo,
  createAdapterZgmlNamespace,
} = require("./frontend_namespace_surface.js");
const {
  createAdapterFrontendModuleSurface,
} = require("./frontend_module_surface.js");
const {
  createAdapterNativeEagerSurface,
} = require("./native_eager_surface.js");
const {
  nativeEagerRoutingActivationEnabled,
  nativeEagerRoutingUnaryOpEnabled,
  nodeNativeEagerRoutingPolicy,
} = require("./native_eager_routing_policy.js");
const {
  createAdapterNativeCoreSurface,
} = require("./native_core_surface.js");
const {
  createAdapterNativeTrainingSurface,
} = require("./native_training_surface.js");
const {
  createAdapterSafetensorsFileHeaderHelpers,
} = require("./safetensors_file_header.js");
const {
  createAdapterNativeBufferInstanceSurface,
} = require("./native_buffer_instance_surface.js");
const {
  createAdapterNativeBufferFacadeSurface,
} = require("./native_buffer_facade_surface.js");
const {
  createNodeNativeBufferSyscalls,
} = require("./node_native_buffer_syscalls.js");
const {
  createNodeNativeLifecycleOps,
} = require("./node_native_lifecycle_ops.js");
const {
  requireNodeSharedFrontendRuntime,
} = require("./node_shared_frontend_runtime.js");
const {
  createAdapterSharedFrontendProjection,
} = require("./shared_frontend_projection.js");
const {
  readFileSync,
  writeFileSync,
} = require("node:fs");
const {
  createAdapterDeferredSlot,
} = require("./deferred_slot.js");
const {
  webgpuInterop,
} = require("./webgpu_interop.js");
const {
  adapterBufferStorageAliases,
  adapterModelKindAliases,
} = require("../runtime/native_abi_constants.js");
const {
  createAdapterIndexValuesSurface,
} = require("./index_values_surface.js");
const {
  createAdapterGradModeSurface,
} = require("./grad_mode_surface.js");
const {
  fs,
  symbols,
  ZgmlError,
  check,
  handleOut,
  readHandle,
  assertAlive,
  runtimeInfo,
  abiStructSize,
  abiStructSizes,
  loadedRuntimeInfo,
} = createNodeFfiBootstrap({ dirname: __dirname });
const nodeSymbolGroups = createNodeSymbolGroups(symbols);

const sharedFrontend = requireNodeSharedFrontendRuntime({
  requireModule: require,
  origin: __dirname,
});

const {
  validateTensorShape,
  rowMajorStrides,
  normalizeDim,
  geluScalar,
  siluScalar,
  isGradEnabled,
  is_grad_enabled,
  setGradEnabled,
  set_grad_enabled,
  noGrad,
  no_grad,
  inferenceMode,
  inference_mode,
  enableGrad,
  enable_grad,
  normalizeFactoryShape,
  modelKinds,
  bufferStorageIds,
  programBufferKinds,
  normalizeCompileOptions,
  tokenId,
  tokenSelectionResultFromAbiRecord,
  tokenGenerateResultFromAbiRecord,
  bindLlamaSessionHandleByPolicy,
  safetensorsHeaderBytes,
  safetensorsDataBytes,
  programTraceFromCompileEvidence,
  programTensorProgramIrFromCompileEvidence,
  programKernelPlanFromCompileEvidence,
  programShapeConstraintsFromKernelPlan,
  programParameterLayoutFromKernelPlan,
  programCompatibilityAccepted,
  programCapabilityCanExecute,
  programCapabilityCanBindExternalResources,
  programCapabilityHasFullDispatchPlan,
  programExecutionModeFromCapabilities,
  backendId,
  backendName,
  normalizeWebGpuImportSource,
  programDeviceBufferImportInfo,
  programBufferKindId,
  normalizeLoadModelKind,
  modelLoadKindId,
  packedProgramWeightsLen,
  packedProgramBiasLen,
  programInputShape,
  programOutputShape,
  sessionOutputTensorShape,
  runtimeProfileFromAbiRecord,
  programInspectionFromAbiRecord,
  sessionInspectionFromAbiRecord,
  bufferInspectionFromAbiRecord,
  modelInspectionFromAbiRecord,
  programRequirementsFromAbiRecord,
  llamaKvCacheRequirementsFromAbiRecord,
  programModelCompatibilityFromAbiRecord,
  programExecutionCapabilitiesFromInspection,
  programBufferLayoutFromRequirements,
  programBufferFactorySlotFromRequirements,
  requireProgramBufferResourceFactory,
  assertProgramBufferSlotRequired,
  assertProgramBufferResourceByteLength,
  llamaKvCacheResourceFactory,
  llamaKvCacheResourceSlot,
  llamaProgramKvCacheBufferSlot,
  assertLlamaKvCacheResourceByteLength,
  llamaKvCacheLayoutFromRequirements,
  moduleProgramCompileArtifactsFromCompiledSpec,
  moduleCompatibilityForProgramEvidence,
} = createAdapterSharedFrontendProjection(sharedFrontend);

const {
  autoKind,
  tinyLinearKind,
  tinyLlamaKind,
  smollm135mKind,
  tinyLlama2LayerKind,
  tinyMlpKind,
  moduleKind,
} = adapterModelKindAliases(modelKinds);
const {
  bufferStorageExternalResource,
} = adapterBufferStorageAliases(bufferStorageIds);

let modelHandleForBind;
let modelHandleForCompatibility;

const {
  compileDesc,
  modelInspectionFromNative,
  inspectModelHandle,
  compileModelProgramHandle,
  inspectBufferHandle,
  inspectSessionHandle,
  llamaSessionPosition,
  resetSessionHandle,
  resetSessionRuntimeProfileHandle,
  inspectExecutableProgram,
  programExecutionCapabilities,
  programRequirements,
  programModelCompatibility,
  inspectLlamaProgram,
  programRuntimeProfile,
  sessionRuntimeProfile,
} = createNodeInspectionOps({
  symbols: nodeSymbolGroups.inspection,
  check,
  handleOut,
  readHandle,
  normalizeCompileOptions,
  modelHandleForCompatibility: (model) => modelHandleForCompatibility(model),
  modelInspectionFromAbiRecord,
  bufferInspectionFromAbiRecord,
  sessionInspectionFromAbiRecord,
  programInspectionFromAbiRecord,
  programExecutionCapabilitiesFromInspection,
  programRequirementsFromAbiRecord,
  programModelCompatibilityFromAbiRecord,
  runtimeProfileFromAbiRecord,
});

const nativeLifecycleOps = createNodeNativeLifecycleOps({
  symbols: nodeSymbolGroups.nativeLifecycle,
  check,
});

const {
  rawF32,
  prepareF32,
  f32,
  byteView,
  addTensorGrad,
  requirePositiveInteger,
  zerosF32,
  f32WithLength,
  defaultedF32,
  tensorCoreHelpers,
  tensorMetadataHelpers,
  tensorGradStateHelpers,
  tensorMathSurfaceHelpers,
  sessionTensorHelpers,
  tensorHostSurface,
} = createAdapterTensorRuntimeSurface({
  sharedFrontend,
  Tensor,
  isGradEnabled,
  nativeEagerMatmulInto: (output, lhs, rhs, options) => nativeEager.matmulInto(output, lhs, rhs, options),
  nativeEagerBmmInto: (output, lhs, rhs, options) => nativeEager.bmmInto(output, lhs, rhs, options),
  nativeEagerBmmMinMultiplyAdds: nodeNativeEagerRoutingPolicy.tensorMath.bmmMinMultiplyAdds,
  nativeEagerElementwiseInto: (output, lhs, rhs, options) => nativeEager.elementwiseInto(output, lhs, rhs, options),
  nativeEagerElementwiseMinLength: nodeNativeEagerRoutingPolicy.tensorMath.elementwiseMinLength,
  nativeEagerUnaryOpEnabled: (op) => nativeEagerRoutingUnaryOpEnabled(nodeNativeEagerRoutingPolicy, op),
  nativeEagerActivationInto: (output, input, options) => nativeEager.activationInto(output, input, options),
  nativeEagerActivationMinLength: nodeNativeEagerRoutingPolicy.tensorMath.activationMinLength,
  nativeEagerActivationEnabled: (activation) => nativeEagerRoutingActivationEnabled(nodeNativeEagerRoutingPolicy, activation),
  nativeEagerWhereInto: (output, condition, input, other) => nativeEager.whereInto(output, condition, input, other),
  nativeEagerClampInto: (output, input, options) => nativeEager.clampInto(output, input, options),
  nativeEagerReduceInto: (output, input, options) => nativeEager.reduceInto(output, input, options),
  nativeEagerReduceDimInto: (output, input, options) => nativeEager.reduceDimInto(output, input, options),
  nativeEagerArgReduceDimInto: (output, input, options) => nativeEager.argReduceDimInto(output, input, options),
  nativeEagerCumsumInto: (output, input, options) => nativeEager.cumsumInto(output, input, options),
  nativeEagerMomentInto: (output, input, options) => options && options.sqrtOutput
    ? nativeEager.stdInto(output, input, options)
    : nativeEager.varianceInto(output, input, options),
  nativeEagerReduceMinLength: nodeNativeEagerRoutingPolicy.tensorMath.reduceMinLength,
  nativeEagerDotInto: (output, lhs, rhs) => nativeEager.dotInto(output, lhs, rhs),
  nativeEagerSoftmaxInto: (output, input, options) => options && options.logSoftmax
    ? nativeEager.logSoftmaxInto(output, input, options)
    : nativeEager.softmaxInto(output, input, options),
  nativeFullF32: (output, value) => {
    check(nodeSymbolGroups.nativeEager.eagerFullF32(output, output.length, value));
  },
  nativeArangeF32: (output, start, step) => {
    check(nodeSymbolGroups.nativeEager.eagerArangeF32(output, output.length, start, step));
  },
  meanSquaredError: (tensor, target) => meanSquaredError(tensor, target),
  dtype: (tensor) => tensorPlacementHelpers.dtype(tensor),
  device: (tensor) => tensorPlacementHelpers.device(tensor),
  rowMajorStrides,
  normalizeDim,
  isNativeBuffer: (value) => isNativeBuffer(value),
  nativeBufferFromFloat32: (data) => NativeBuffer.fromFloat32(data),
});
let NativeBuffer = AdapterNativeBuffer;
const { isNativeBuffer } = createAdapterNativeBufferInstanceSurface({
  getNativeBufferClass: () => NativeBuffer,
  uninitializedMessage: "Node NativeBuffer instance surface is not initialized",
});
const {
  sessionUploadPersistent,
  sessionUploadPersistentRange,
  stepSession,
  prepareStepSession,
  stepNoOutput,
} = createNodeSessionOps({
  symbols: nodeSymbolGroups.session,
  check,
});
const adapterSessionRuntimeSurface = createAdapterSessionRuntimeSurface({
  sharedFrontend,
  isNativeBuffer,
  f32,
  prepareF32,
  valueShape: (value) => value instanceof Tensor ? value.shape : null,
  sessionTensorHelpers,
  assertSessionAlive: (handle) => assertAlive(handle, "session"),
  sessionInspect: inspectSessionHandle,
  sessionUploadPersistent,
  sessionUploadPersistentRange,
  sessionReset: resetSessionHandle,
  sessionRuntimeProfile,
  sessionResetRuntimeProfile: resetSessionRuntimeProfileHandle,
  sessionFree: nativeLifecycleOps.sessionFree,
  nullSessionHandle: null,
  stepSession,
  prepareStepSession,
  stepNoOutput,
});
const genericSessionFacade = adapterSessionRuntimeSurface.genericSessionFacade;

const tensorFactoryHelpers = tensorHostSurface.tensorFactoryHelpers;
const tensorViewHelpers = tensorHostSurface.tensorViewHelpers;
const tensorViewSurfaceHelpers = tensorHostSurface.tensorViewSurfaceHelpers;
const tensorPlacementHelpers = tensorHostSurface.tensorPlacementHelpers;
const tensorInfoSurfaceHelpers = tensorHostSurface.tensorInfoSurfaceHelpers;
const tensorJoinHelpers = tensorHostSurface.tensorJoinHelpers;
const tensorIndexHelpers = tensorHostSurface.tensorIndexHelpers;
const tensorIndexSurfaceHelpers = tensorHostSurface.tensorIndexSurfaceHelpers;
const tensorFacade = tensorHostSurface.tensorFacade;
const tensorNativeSurfaceHelpers = tensorHostSurface.tensorNativeSurfaceHelpers;
const tensorStaticHelpers = tensorHostSurface.tensorStaticHelpers;
setAdapterTensorSurfaceHelpers({
  tensorFacade,
  tensorNativeSurfaceHelpers,
  tensorStaticHelpers,
  tensorInfoSurfaceHelpers,
  tensorIndexSurfaceHelpers,
  tensorViewSurfaceHelpers,
  tensorMathSurfaceHelpers,
});
const tensorRootOps = tensorHostSurface.tensorRootOps;
const {
  tensor,
  parameter,
  cat,
  concat,
  concatenate,
  stack,
  vstack,
  hstack,
  einsum,
  full,
  fullLike,
  full_like,
  empty,
  emptyLike,
  empty_like,
  zeros,
  zerosLike,
  zeros_like,
  ones,
  onesLike,
  ones_like,
  eye,
  scalar,
  rand,
  randLike,
  rand_like,
  randn,
  randnLike,
  randn_like,
  randInt,
  randint,
  randPerm,
  randperm,
  manualSeed,
  manual_seed,
  initialSeed,
  initial_seed,
  seededRng,
  linspace,
  arange,
  allclose,
  equal,
  hasShape,
  requireShape,
  to,
  cpu,
  float,
  float32,
  typeAs,
  type_as,
  clone,
  detach,
  reshape,
  view,
  broadcastTo,
  expand,
  repeat,
  tile,
  flatten,
  squeeze,
  unsqueeze,
  transpose,
  permute,
  flip,
  roll,
  select,
  narrow,
  slice,
  indexSelect,
  index_select,
  gather,
  take,
  argsort,
  sort,
  topk,
  scatterAdd,
  scatter_add,
  split,
  chunk,
  unbind,
  add,
  sub,
  mul,
  div,
  eq,
  ne,
  lt,
  le,
  gt,
  ge,
  isclose,
  pow,
  neg,
  negative,
  exp,
  expm1,
  log,
  log1p,
  sqr,
  square,
  recip,
  reciprocal,
  abs,
  sgn,
  sign,
  step,
  isnan,
  isinf,
  isfinite,
  floor,
  ceil,
  round,
  trunc,
  sqrt,
  rsqrt,
  relu,
  gelu,
  silu,
  sigmoid,
  tanh,
  sin,
  cos,
  tan,
  maximum,
  minimum,
  where,
  maskedFill,
  masked_fill,
  sum,
  prod,
  cumsum,
  mean,
  max,
  min,
  any,
  all,
  argmax,
  argmin,
  variance,
  std,
  norm,
  softmax,
  softmax_dim,
  softmaxDim,
  logSoftmax,
  log_softmax,
  log_softmax_dim,
  logSoftmaxDim,
  logsumexp,
  logSumExp,
  clamp,
  clip,
  matmul,
  mm,
  dot,
  trace,
  diagonal,
  bmm,
} = tensorRootOps;

const param = parameter;
const {
  requireNativeBuffer,
  uniqueNativeBuffers,
  bindModuleThroughProgram,
  validateProgramBindParams,
  programBindingPlan,
  prepareProgramBind,
  freeOwnedProgramBindBuffers,
  programNativeBufferBindFields,
} = adapterSessionRuntimeSurface;

const {
  sessionStepToken,
  executeLlamaTokenWindow,
  sessionAdvanceToken,
  llamaArgmaxLogits,
  executeLlamaArgmaxWindow,
  generateLlamaArgmaxWindow,
  llamaSampleLogits,
  executeLlamaSampleWindow,
  generateLlamaSampleWindow,
} = createNodeLlamaTokenOps({
  symbols: nodeSymbolGroups.llamaToken,
  check,
  tokenId,
  isNativeBuffer,
  tokenSelectionResultFromAbiRecord,
  tokenGenerateResultFromAbiRecord,
});

const programBufferFactoriesSlot = createAdapterDeferredSlot("Node Program buffer factories");

const {
  createProgramOutputBuffer,
  createProgramBuffer,
  createProgramKvCacheBuffer,
  createProgramNamedBuffer,
  createProgramLlamaKvCache,
} = createAdapterProgramBufferFactorySurface({
  getProgramBufferFactories: programBufferFactoriesSlot.peek,
  uninitializedMessage: "Node Program buffer factory surface is not initialized",
});

let llamaSessionFacade;
const {
  bindLlamaProgramSession,
  defaultSessionBufferLayout: defaultLlamaSessionBufferLayout,
  createSessionScratch: createLlamaSessionScratch,
  bufferLayout: llamaSessionBufferLayout,
  bufferSlotNames: llamaSessionBufferSlotNames,
  bufferSlot: llamaSessionBufferSlot,
  kvCacheLayout: llamaSessionKvCacheLayout,
  inputLen: llamaSessionInputLen,
  outputLen: llamaSessionOutputLen,
  inputByteLength: llamaSessionInputByteLength,
  outputByteLength: llamaSessionOutputByteLength,
  weightsLen: llamaSessionWeightsLen,
  weightsByteLength: llamaSessionWeightsByteLength,
  biasLen: llamaSessionBiasLen,
  biasByteLength: llamaSessionBiasByteLength,
  parameterLen: llamaSessionParameterLen,
  parameterByteLength: llamaSessionParameterByteLength,
  bufferSizing: llamaSessionBufferSizing,
  matchesBufferSizingSignature: llamaSessionMatchesBufferSizingSignature,
  outputShape: llamaSessionOutputShape,
  stepContract: llamaSessionStepContract,
  preflightStepParams: llamaSessionPreflightStepParams,
  stepParamsCompatibility: llamaSessionStepParamsCompatibility,
  acceptsStepParams: llamaSessionAcceptsStepParams,
  canExecuteStepParams: llamaSessionCanExecuteStepParams,
  requireCanExecuteStepParams: llamaSessionRequireCanExecuteStepParams,
  acceptsAllocationFreeStepParams: llamaSessionAcceptsAllocationFreeStepParams,
  requireAllocationFreeStepParams: llamaSessionRequireAllocationFreeStepParams,
  acceptsRuntimeOutputAllocationFreeStepParams: llamaSessionAcceptsRuntimeOutputAllocationFreeStepParams,
  requireRuntimeOutputAllocationFreeStepParams: llamaSessionRequireRuntimeOutputAllocationFreeStepParams,
  acceptsNoReadbackStepParams: llamaSessionAcceptsNoReadbackStepParams,
  requireNoReadbackStepParams: llamaSessionRequireNoReadbackStepParams,
  acceptsReadbackFreeStepParams: llamaSessionAcceptsReadbackFreeStepParams,
  requireReadbackFreeStepParams: llamaSessionRequireReadbackFreeStepParams,
  acceptsHotStepParams: llamaSessionAcceptsHotStepParams,
  requireHotStepParams: llamaSessionRequireHotStepParams,
  executionPlan: llamaSessionExecutionPlan,
  requireExecutionPlan: llamaSessionRequireExecutionPlan,
  hotPathPlan: llamaSessionHotPathPlan,
  matchesStepContractSignature: llamaSessionMatchesStepContractSignature,
  matchesStepParamsSignature: llamaSessionMatchesStepParamsSignature,
  matchesStepParamsCompatibility: llamaSessionMatchesStepParamsCompatibility,
  position: llamaSessionPositionFacade,
  inspect: inspectLlamaSessionFacade,
  reset: resetLlamaSessionFacade,
  sessionCallProfile: llamaSessionCallProfileFacade,
  matchesSessionCallProfileSignature: llamaSessionMatchesCallProfileSignature,
  resetSessionCallProfile: resetLlamaSessionCallProfileFacade,
  runtimeProfile: llamaSessionRuntimeProfileFacade,
  matchesRuntimeProfileSignature: llamaSessionMatchesRuntimeProfileSignature,
  resetRuntimeProfile: resetLlamaSessionRuntimeProfileFacade,
  free: freeLlamaSessionFacade,
  dispose: disposeLlamaSessionFacade,
  scalarTokenWindow: llamaScalarTokenWindow,
  sessionScalarTokenSampleOptions: llamaScalarTokenSampleOptions,
  sessionNoOutputTokenWindowOptions: llamaNoOutputTokenWindowOptions,
  sessionOutputTokenWindowOptions: llamaOutputTokenWindowOptions,
  sessionScalarTokenExecuteOptions: llamaScalarTokenExecuteOptions,
  sessionScalarTokenOutputOptions: llamaScalarTokenOutputOptions,
  readOutputInto: readLlamaOutputInto,
  readOutputTensor: readLlamaOutputTensor,
  executeTokens: executeLlamaTokens,
  execute: executeLlama,
  executeTensor: executeLlamaTensor,
  executeInto: executeLlamaInto,
  step: stepLlama,
  advance: advanceLlama,
  advanceTokens: advanceLlamaTokens,
  stepTensor: stepLlamaTensor,
  stepInto: stepLlamaInto,
  prefill: prefillLlama,
  prefillTensor: prefillLlamaTensor,
  prefillInto: prefillLlamaInto,
  argmaxToken: argmaxLlamaToken,
  sampleToken: sampleLlamaToken,
  executeTokensArgmax: executeLlamaTokensArgmax,
  stepArgmax: stepLlamaArgmax,
  executeTokensSample: executeLlamaTokensSample,
  stepSample: stepLlamaSample,
  generateTokensArgmax: generateLlamaTokensArgmax,
  generateTokensArgmaxInto: generateLlamaTokensArgmaxInto,
  generateTokensSample: generateLlamaTokensSample,
  generateTokensSampleInto: generateLlamaTokensSampleInto,
} = (llamaSessionFacade = createAdapterLlamaSessionFacadeSurface({
  sharedFrontend,
  isNativeBuffer,
  f32,
  createProgramOutputBuffer,
  sessionTensorHelpers,
  executeTokenWindow: executeLlamaTokenWindow,
  stepToken: sessionStepToken,
  advanceToken: sessionAdvanceToken,
  assertSessionAlive: (handle) => assertAlive(handle, "session"),
  sessionPosition: llamaSessionPosition,
  sessionInspect: inspectSessionHandle,
  sessionReset: resetSessionHandle,
  sessionRuntimeProfile,
  sessionResetRuntimeProfile: resetSessionRuntimeProfileHandle,
  sessionFree: nativeLifecycleOps.sessionFree,
  nullSessionHandle: null,
  argmaxLogits: llamaArgmaxLogits,
  sampleLogits: llamaSampleLogits,
  executeArgmaxWindow: executeLlamaArgmaxWindow,
  executeSampleWindow: executeLlamaSampleWindow,
  generateArgmaxWindow: generateLlamaArgmaxWindow,
  generateSampleWindow: generateLlamaSampleWindow,
}));

const {
  loadModelPath,
  loadSafetensorsDataHandle,
  createTinyLlamaModelHandle,
  probeModelPath,
  probeSafetensorsData,
  probeSafetensorsHeader,
  supportedCheckpointModels,
} = createNodeModelSourceOps({
  symbols: nodeSymbolGroups.modelSource,
  check,
  handleOut,
  readHandle,
  ...createNodeModelSourceBytes(),
  safetensorsDataBytes,
  safetensorsHeaderBytes,
  normalizeLoadModelKind,
  modelLoadKindId,
  modelInspectionFromAbiRecord,
  tinyLlamaKind,
});
const modelSourceFacadeSlot = createAdapterDeferredSlot("Node model-source facade");
const {
  probeModel,
  loadModel,
  loadSafetensorsData,
} = createAdapterModelSourceSurface({
  getModelSourceFacade: modelSourceFacadeSlot.peek,
  uninitializedMessage: "Node model-source surface is not initialized",
});

const {
  llamaKvCacheRequirements,
  bindLlamaSessionHandle,
} = createNodeLlamaSessionBindOps({
  symbols: nodeSymbolGroups.llamaSessionBind,
  check,
  handleOut,
  readHandle,
  requireNativeBuffer,
  isNativeBuffer,
  modelHandleForBind: (model) => modelHandleForBind(model),
  bindLlamaSessionHandleByPolicy,
  llamaKvCacheRequirementsFromAbiRecord,
});

const {
  compileModuleProgram,
  attachProgramCompileEvidence,
  bufferBindDescForTinyLinear,
} = createNodeModuleProgramOps({
  symbols: nodeSymbolGroups.moduleProgram,
  check,
  handleOut,
  readHandle,
  compileDesc,
  moduleProgramCompileArtifactsFromCompiledSpec,
  inspectExecutableProgram,
  programNativeBufferBindFields,
  programRequirementsFromAbiRecord,
  createProgram: createAdapterProgramFactory({
    getProgramClass: () => Program,
  }),
});

const {
  bindProgram,
} = createNodeProgramBindOps({
  symbols: nodeSymbolGroups.programBind,
  check,
  handleOut,
  readHandle,
  prepareProgramBind,
  freeOwnedProgramBindBuffers,
  bufferBindDescForTinyLinear,
  createSession: createAdapterSessionFactory({
    getSessionClass: () => Session,
  }),
});

let nativeBufferFacade;

const nativeBufferSyscalls = createNodeNativeBufferSyscalls({
  symbols: nodeSymbolGroups.nativeBuffer,
  check,
  handleOut,
  readHandle,
});

nativeBufferFacade = createAdapterNativeBufferFacadeSurface({
  sharedFrontend,
  NativeBuffer,
  nullHandle: null,
  f32,
  byteView,
  isLiveHandle: (handle) => Boolean(handle),
  createBuffer: nativeBufferSyscalls.createBuffer,
  wrapBytes: nativeBufferSyscalls.wrapBytes,
  wrapExternalResource: nativeBufferSyscalls.wrapExternalResource,
  bufferSize: nativeBufferSyscalls.bufferSize,
  inspectBuffer: inspectBufferHandle,
  writeBuffer: nativeBufferSyscalls.writeBuffer,
  readBuffer: nativeBufferSyscalls.readBuffer,
  freeBuffer: nativeBufferSyscalls.freeBuffer,
  programDeviceHandle: (...args) => programDeviceHandle(...args),
});
setAdapterNativeBufferFacade(nativeBufferFacade);

const LlamaKvCache = createNodeLlamaKvCacheClass();

const {
  createLlamaKvCache,
  createProgramDeviceKvCache,
  createProgramBufferFactoryKvCache,
} = createNodeLlamaKvCacheOps({
  ...createNodeLlamaKvCacheFactorySurface({
    getNativeBufferClass: () => NativeBuffer,
    getKvCacheClass: () => LlamaKvCache,
  }),
  requireNativeBuffer,
  assertLlamaKvCacheResourceByteLength,
  llamaKvCacheResourceFactory,
  llamaKvCacheResourceSlot,
  llamaKvCacheLayoutFromRequirements,
  llamaKvCacheRequirements,
});

const {
  programCreateOutputBuffer,
  programCreateBuffer,
  programCreateDeviceBuffer,
  programDeviceHandle,
  programImportDeviceBuffer,
} = createNodeProgramBufferOps({
  symbols: nodeSymbolGroups.programBuffer,
  programBufferKinds,
  webgpuImportSourceKey: webgpuInterop.importSource,
  check,
  handleOut,
  readHandle,
  programBufferKindId,
  backendId,
  normalizeWebGpuImportSource,
  programDeviceBufferImportInfo,
  webgpuInterop,
  ...createAdapterProgramBufferNativeBridge({
    getNativeBufferClass: () => NativeBuffer,
  }),
});

const adapterProgramRuntimeSurface = createAdapterProgramRuntimeSurface({
  sharedFrontend,
  assertAlive,
  programDeviceHandle,
  programCreateDeviceBuffer,
  programImportDeviceBuffer,
  isNativeBuffer,
  createProgramDeviceKvCache,
  programRequirements,
  llamaKvCacheRequirements,
  programCreateBuffer,
  programCreateOutputBuffer,
  requireNativeBuffer,
  createProgramBufferFactoryKvCache,
  programModelCompatibility,
  programExecutionCapabilities,
  programRuntimeProfile,
  programResetRuntimeProfile: nativeLifecycleOps.programResetRuntimeProfile,
  programFree: nativeLifecycleOps.programFree,
  nullProgramHandle: null,
  inspectModelHandle,
  compileModelProgramHandle,
  modelFree: nativeLifecycleOps.modelFree,
  nullModelHandle: null,
  inspectExecutableProgram,
  programInputShape,
  programOutputShape,
  createProgramOutputBuffer,
  createProgramNamedBuffer,
  createProgramBuffer,
  bindModuleThroughProgram,
  programBindingPlan,
  createTinyLlamaModelHandle,
  loadModelPath,
  loadSafetensorsDataHandle,
  probeModel,
  probeSafetensorsHeader,
  inspectLlamaProgram,
  createProgramLlamaKvCache,
  bindLlamaProgramSession,
});
const ProgramDevice = adapterProgramRuntimeSurface.ProgramDevice;
programBufferFactoriesSlot.bind(adapterProgramRuntimeSurface.programBufferFactories);
const programFacadePolicy = adapterProgramRuntimeSurface.programFacadePolicy;
const modelFacadePolicy = adapterProgramRuntimeSurface.modelFacadePolicy;
const genericProgramFacade = adapterProgramRuntimeSurface.genericProgramFacade;
const llamaModelFamilyFacade = adapterProgramRuntimeSurface.llamaModelFamilyFacade;
const llamaProgramFacade = adapterProgramRuntimeSurface.llamaProgramFacade;

const {
  TinyLlamaModel,
  TinyLlamaProgram,
  TinyLlamaSession,
  LlamaModel,
  LlamaProgram,
  LlamaSession,
  SmolLM135MModel,
  SmolLM135MProgram,
  SmolLM135MSession,
} = createAdapterLlamaFamilySurface({
  autoKind,
  tinyLlamaKind,
  smollm135mKind,
  llamaModelFamilyFacade,
  llamaProgramFacade,
  bindLlamaSessionHandle,
  defaultLlamaSessionBufferLayout,
  createLlamaSessionScratch,
  assertSessionAlive: (handle) => assertAlive(handle, "session"),
  llamaSessionFacade,
});

const {
  TinyLinearModel,
  TinyMlpModel,
  Program,
  Session,
  TinyLinearProgram,
  TinyLinearSession,
} = createNodeGenericFamilySurface({
  tinyLinearKind,
  tinyMlpKind,
  symbols: nodeSymbolGroups.genericFamily,
  check,
  handleOut,
  readHandle,
  modelFacadePolicy,
  genericProgramFacade,
  genericSessionFacade,
  bindProgram,
  packedProgramWeightsLen,
  packedProgramBiasLen,
  programBufferLayoutFromRequirements,
});

({ modelHandleForBind, modelHandleForCompatibility } = createAdapterModelHandlePolicy({
  sharedFrontend,
  TinyLinearModel,
  TinyMlpModel,
  TinyLlamaModel,
  SmolLM135MModel,
  LlamaModel,
  assertAlive,
}));

const {
  makeParameter,
  parameterView,
  moduleCompileSupport,
  composableModuleCompileSupport,
  resolveParameters,
  parameterNames,
  parameterInfos,
  parameterInfo,
  stateKeys,
  stateData,
  assertStateShape,
  assertStateLayout,
  stateDict,
  loadStateDict,
  zeroGrad,
  setRequiresGrad,
  finiteConfigNumber,
  optimizerStateEntry,
  optimizerStateDict,
  optimizerStateEntries,
  optimizerStepFromState,
  loadOptimizerTensorState,
  rejectUnexpectedOptimizerState,
} = createAdapterModuleStateSurface({
  sharedFrontend,
  Tensor,
  f32WithLength,
  defaultLayout: "row-major",
  optionalParameterFields: true,
});

const { indexValues } = createAdapterIndexValuesSurface({ Tensor });
const traceModuleCompilerSlot = createAdapterDeferredSlot("Node trace Module compiler");
const {
  analyzeSequentialProgram,
  analyzeSingleModuleProgram,
  traceSequentialProgram,
  packedSequentialProgramParameters,
} = createAdapterModuleCompilerSurface({
  getTraceModuleCompiler: traceModuleCompilerSlot.peek,
  uninitializedMessage: "Node Module compiler surface is not initialized",
});

function trustedNativeEagerData(value, label) {
  if (value instanceof Float32Array) return value;
  if (value && typeof value === "object" && value.data instanceof Float32Array) return value.data;
  return f32(value, label);
}

function nativeEagerLinearInto(output, input, weights, options = {}) {
  const batch = options.batch;
  const inFeatures = options.inFeatures;
  const outFeatures = options.outFeatures;
  if (
    !Number.isSafeInteger(batch) ||
    !Number.isSafeInteger(inFeatures) ||
    !Number.isSafeInteger(outFeatures) ||
    batch <= 0 ||
    inFeatures <= 0 ||
    outFeatures <= 0
  ) {
    return nativeEager.linearInto(output, input, weights, options);
  }
  const inputData = trustedNativeEagerData(input, "native eager linear input");
  const weightData = trustedNativeEagerData(weights, "native eager linear weights");
  const biasValue = options.bias;
  const biasData = biasValue === undefined || biasValue === null
    ? null
    : trustedNativeEagerData(biasValue, "native eager linear bias");
  const expectedOutput = batch * outFeatures;
  const linearSymbol = options.weightLayout === "out-in" || options.transposedWeights === true
    ? nodeSymbolGroups.nativeEager.eagerLinearTransposedWeightsF32
    : nodeSymbolGroups.nativeEager.eagerLinearF32;
  check(linearSymbol(
    inputData,
    inputData.length,
    weightData,
    weightData.length,
    biasData,
    biasData ? biasData.length : 0,
    output,
    expectedOutput,
    batch,
    inFeatures,
    outFeatures,
  ));
  return output;
}

function nativeEagerLinearActivationInto(output, input, weights, options) {
  return nativeEager.linearActivationInto(output, input, weights, options);
}

function nativeEagerSoftmaxInto(output, input, options) {
  return options && options.logSoftmax
    ? nativeEager.logSoftmaxInto(output, input, options)
    : nativeEager.softmaxInto(output, input, options);
}

function nativeEagerConv2dInto(output, input, weights, options) {
  return nativeEager.conv2dInto(output, input, weights, options);
}

function nativeEagerPool2dInto(output, input, options) {
  return nativeEager.pool2dInto(output, input, options);
}

let compileTrainingStepHook = null;

const adapterFrontendModuleSurface = createAdapterFrontendModuleSurface({
  sharedFrontend,
  Tensor,
  f32,
  prepareF32,
  f32WithLength,
  indexValues,
  addTensorGrad,
  isGradEnabled,
  requirePositiveInteger,
  defaultedF32,
  zerosF32,
  makeParameter,
  parameterView,
  nativeEagerLinearInto,
  nativeEagerSoftmaxInto,
  nativeEagerLinearActivationInto,
  nativeEagerConv2dInto,
  nativeEagerPool2dInto,
  parameterNames,
  parameterInfos,
  parameterInfo,
  zeroGrad,
  setRequiresGrad,
  stateKeys,
  stateDict,
  loadStateDict,
  analyzeSequentialProgram,
  analyzeSingleModuleProgram,
  moduleCompileSupport,
  TinyLinearModel,
  compileModuleProgram,
  compileTrainingStep: (...args) => {
    if (compileTrainingStepHook === null) {
      throw new Error("zgml Node FFI module native training compile hook was called before compile namespace initialization");
    }
    return compileTrainingStepHook(...args);
  },
  attachProgramCompileEvidence,
  packedSequentialProgramParameters,
  traceSequentialProgram,
});
traceModuleCompilerSlot.bind(adapterFrontendModuleSurface.traceModuleCompiler);
const ActivationModule = adapterFrontendModuleSurface.ActivationModule;
const SoftmaxModule = adapterFrontendModuleSurface.SoftmaxModule;
const LogSoftmaxModule = adapterFrontendModuleSurface.LogSoftmaxModule;
const ReductionModule = adapterFrontendModuleSurface.ReductionModule;
const DropoutModule = adapterFrontendModuleSurface.DropoutModule;
const LinearModule = adapterFrontendModuleSurface.LinearModule;
const EmbeddingModule = adapterFrontendModuleSurface.EmbeddingModule;
const Conv2dModule = adapterFrontendModuleSurface.Conv2dModule;
const AvgPool2dModule = adapterFrontendModuleSurface.AvgPool2dModule;
const MaxPool2dModule = adapterFrontendModuleSurface.MaxPool2dModule;
const FeatureNormModule = adapterFrontendModuleSurface.FeatureNormModule;
const ShapeModule = adapterFrontendModuleSurface.ShapeModule;
const SequentialModule = adapterFrontendModuleSurface.SequentialModule;

const {
  traceModule,
  compileSupportForModule,
  requireCompileSupportForModule,
  compilerSignaturesForModule,
  tensorProgramIrForModule,
  kernelPlanForModule,
  bufferLayoutForModule,
  memoryLayoutForModule,
  inputShapeForModule,
  outputShapeForModule,
  shapeConstraintsForModule,
  parameterLayoutForModule,
  explainModule,
  requireCompilePlanForModule,
  canCompileModule,
  compileModule,
  bindModuleParameters,
  placeModuleParameters,
  bindingPlanForModuleBindings,
  requireBindingPlanForModuleBindings,
} = adapterFrontendModuleSurface.moduleFacadeHelpers;

const nativeEagerSurface = createAdapterNativeEagerSurface({
  f32: (value, label) => f32(value, label),
  check,
  linearF32: (args) => (args.transposedWeights
    ? nodeSymbolGroups.nativeEager.eagerLinearTransposedWeightsF32
    : nodeSymbolGroups.nativeEager.eagerLinearF32)(
    args.inputData,
    args.inputData.length,
    args.weightData,
    args.weightData.length,
    args.biasData,
    args.biasData ? args.biasData.length : 0,
    args.output,
    args.expectedOutput,
    args.batch,
    args.inFeatures,
    args.outFeatures,
  ),
  matmulF32: (args) => nodeSymbolGroups.nativeEager.eagerMatmulF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData.length,
    args.output,
    args.expectedOutput,
    args.rows,
    args.shared,
    args.cols,
  ),
  bmmF32: (args) => nodeSymbolGroups.nativeEager.eagerBmmF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData.length,
    args.output,
    args.expectedOutput,
    args.batch,
    args.rows,
    args.shared,
    args.cols,
  ),
  linearActivationF32: (args) => (args.transposedWeights
    ? nodeSymbolGroups.nativeEager.eagerLinearActivationTransposedWeightsF32
    : nodeSymbolGroups.nativeEager.eagerLinearActivationF32)(
    args.inputData,
    args.inputData.length,
    args.weightData,
    args.weightData.length,
    args.biasData,
    args.biasData ? args.biasData.length : 0,
    args.output,
    args.expectedOutput,
    args.batch,
    args.inFeatures,
    args.outFeatures,
    args.activation,
  ),
  activationF32: (args) => nodeSymbolGroups.nativeEager.eagerActivationF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.activation,
  ),
  elementwiseF32: (args) => nodeSymbolGroups.nativeEager.eagerElementwiseF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData ? args.rhsData.length : 0,
    args.output,
    args.expectedOutput,
    args.op,
  ),
  elementwiseBroadcastRhsF32: (args) => nodeSymbolGroups.nativeEager.eagerElementwiseBroadcastRhsF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData.length,
    args.output,
    args.expectedOutput,
    args.rows,
    args.cols,
    args.op,
  ),
  elementwiseBroadcastLhsF32: (args) => nodeSymbolGroups.nativeEager.eagerElementwiseBroadcastLhsF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData.length,
    args.output,
    args.expectedOutput,
    args.rows,
    args.cols,
    args.op,
  ),
  whereF32: (args) => nodeSymbolGroups.nativeEager.eagerWhereF32(
    args.conditionData,
    args.conditionData.length,
    args.inputData,
    args.inputData.length,
    args.otherData,
    args.otherData.length,
    args.output,
    args.expectedOutput,
  ),
  clampF32: (args) => nodeSymbolGroups.nativeEager.eagerClampF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.min,
    args.max,
    args.hasMin ? 1 : 0,
    args.hasMax ? 1 : 0,
  ),
  reduceF32: (args) => nodeSymbolGroups.nativeEager.eagerReduceF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.op,
  ),
  reduceDimF32: (args) => nodeSymbolGroups.nativeEager.eagerReduceDimF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.outer,
    args.reduce,
    args.inner,
    args.op,
  ),
  argReduceDimF32: (args) => nodeSymbolGroups.nativeEager.eagerArgReduceDimF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.outer,
    args.reduce,
    args.inner,
    args.op,
  ),
  cumsumF32: (args) => nodeSymbolGroups.nativeEager.eagerCumsumF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.outer,
    args.axisLen,
    args.inner,
    args.reverse ? 1 : 0,
  ),
  momentF32: (args) => nodeSymbolGroups.nativeEager.eagerMomentF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.outer,
    args.reduce,
    args.inner,
    args.correction,
    args.sqrtOutput ? 1 : 0,
  ),
  dotF32: (args) => nodeSymbolGroups.nativeEager.eagerDotF32(
    args.lhsData,
    args.lhsData.length,
    args.rhsData,
    args.rhsData.length,
    args.output,
    args.expectedOutput,
  ),
  conv2dF32: (args) => nodeSymbolGroups.nativeEager.eagerConv2dF32(
    args.inputData,
    args.inputData.length,
    args.weightData,
    args.weightData.length,
    args.biasData,
    args.biasData ? args.biasData.length : 0,
    args.output,
    args.expectedOutput,
    args.batch,
    args.inChannels,
    args.height,
    args.width,
    args.outChannels,
    args.kernelH,
    args.kernelW,
    args.strideH,
    args.strideW,
    args.paddingH,
    args.paddingW,
    args.dilationH,
    args.dilationW,
    args.outH,
    args.outW,
  ),
  pool2dF32: (args) => nodeSymbolGroups.nativeEager.eagerPool2dF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.batch,
    args.channels,
    args.height,
    args.width,
    args.kernelH,
    args.kernelW,
    args.strideH,
    args.strideW,
    args.paddingH,
    args.paddingW,
    args.dilationH,
    args.dilationW,
    args.outH,
    args.outW,
    args.op,
    args.ceilMode ? 1 : 0,
    args.countIncludePad ? 1 : 0,
  ),
  softmaxF32: (args) => nodeSymbolGroups.nativeEager.eagerSoftmaxF32(
    args.inputData,
    args.inputData.length,
    args.output,
    args.expectedOutput,
    args.rows,
    args.cols,
    args.logSoftmax ? 1 : 0,
  ),
});
const nativeEager = Object.freeze({
  ...nativeEagerSurface.nativeEager,
  routingPolicy: nodeNativeEagerRoutingPolicy,
  routing_policy: nodeNativeEagerRoutingPolicy,
});
const native_eager = nativeEager;
const { nativeCore, native_core } = createAdapterNativeCoreSurface({
  host: "node",
  runtimeInfo,
});

function callNodeNativeMlpTraining(call, args) {
  const outLoss = new Float32Array(1);
  const outCorrect = [0];
  const statusCode = call(
    args.input,
    args.input.length,
    args.targets,
    args.targets.length,
    args.w1,
    args.w1.length,
    args.b1,
    args.b1.length,
    args.w2,
    args.w2.length,
    args.b2,
    args.b2.length,
    args.mw1,
    args.mw1.length,
    args.vw1,
    args.vw1.length,
    args.mb1,
    args.mb1.length,
    args.vb1,
    args.vb1.length,
    args.mw2,
    args.mw2.length,
    args.vw2,
    args.vw2.length,
    args.mb2,
    args.mb2.length,
    args.vb2,
    args.vb2.length,
    args.hidden,
    args.hidden.length,
    args.logits,
    args.logits.length,
    args.gradHidden,
    args.gradHidden.length,
    args.gradW1,
    args.gradW1.length,
    args.gradW2,
    args.gradW2.length,
    args.batch,
    args.inFeatures,
    args.hiddenFeatures,
    args.classes,
    args.step,
    args.lr,
    args.beta1,
    args.beta2,
    args.eps,
    args.weightDecay,
    outLoss,
    outCorrect,
  );
  return { status: statusCode, loss: outLoss[0], correct: Number(outCorrect[0] ?? 0) };
}

function callNodeNativeMlpBulkTraining(call, args) {
  const outLoss = new Float32Array(1);
  const outCorrect = [0];
  const outSteps = [0];
  const statusCode = call(
    args.datasetInput,
    args.datasetInput.length,
    args.datasetTargets,
    args.datasetTargets.length,
    args.indices,
    args.indices.length,
    args.batchInput,
    args.batchInput.length,
    args.batchTargets,
    args.batchTargets.length,
    args.w1,
    args.w1.length,
    args.b1,
    args.b1.length,
    args.w2,
    args.w2.length,
    args.b2,
    args.b2.length,
    args.mw1,
    args.mw1.length,
    args.vw1,
    args.vw1.length,
    args.mb1,
    args.mb1.length,
    args.vb1,
    args.vb1.length,
    args.mw2,
    args.mw2.length,
    args.vw2,
    args.vw2.length,
    args.mb2,
    args.mb2.length,
    args.vb2,
    args.vb2.length,
    args.hidden,
    args.hidden.length,
    args.logits,
    args.logits.length,
    args.gradHidden,
    args.gradHidden.length,
    args.gradW1,
    args.gradW1.length,
    args.gradW2,
    args.gradW2.length,
    args.sampleCount,
    args.batch,
    args.inFeatures,
    args.hiddenFeatures,
    args.classes,
    args.epochs,
    args.startStep,
    args.lr,
    args.beta1,
    args.beta2,
    args.eps,
    args.weightDecay,
    outLoss,
    outCorrect,
    outSteps,
  );
  const expectedSteps = args.epochs * Math.floor(args.sampleCount / args.batch);
  const steps = Number(outSteps[0] ?? 0);
  return { status: statusCode, loss: outLoss[0], correct: Number(outCorrect[0] ?? 0), steps: steps > 0 ? steps : expectedSteps };
}

function callNodeNativeLinearBulkTraining(call, args) {
  const outLoss = new Float32Array(1);
  const outSteps = [0];
  const statusCode = call(
    args.datasetInput,
    args.datasetInput.length,
    args.datasetTargets,
    args.datasetTargets.length,
    args.indices,
    args.indices.length,
    args.batchInput,
    args.batchInput.length,
    args.batchTarget,
    args.batchTarget.length,
    args.weight,
    args.weight.length,
    args.bias,
    args.bias.length,
    args.output,
    args.output.length,
    args.gradWeight,
    args.gradWeight.length,
    args.sampleCount,
    args.batch,
    args.inFeatures,
    args.outFeatures,
    args.epochs,
    args.lr,
    args.weightDecay,
    outLoss,
    outSteps,
  );
  const expectedSteps = args.epochs * Math.floor(args.sampleCount / args.batch);
  const steps = Number(outSteps[0] ?? 0);
  return { status: statusCode, loss: outLoss[0], correct: 0, steps: steps > 0 ? steps : expectedSteps };
}

const nativeTrainingPlanIds = Object.freeze({
  model: Object.freeze({ linear: 1, "sequential-mlp-relu": 2 }),
  optimizer: Object.freeze({ sgd: 1, adam: 2, adamw: 3 }),
  loss: Object.freeze({ mse: 1, crossEntropy: 2 }),
});

const nativeTrainingKernelNames = Object.freeze({
  1: "zgml_train_linear_mse_sgd_f32",
  2: "zgml_train_mlp_relu_cross_entropy_adam_f32",
  3: "zgml_train_mlp_relu_cross_entropy_adamw_f32",
});

function requiredPlanId(table, key, label) {
  const id = table[key];
  if (!id) throw new Error(`compile.trainingStep unsupported ${label}: ${key}`);
  return id;
}

function numberField(record, key) {
  return Number(record[key] ?? 0);
}

function nativeTrainingPlanFields(record) {
  const kernel = nativeTrainingKernelNames[numberField(record, "kernel_kind")];
  if (!kernel || numberField(record, "supported") !== 1) {
    throw new Error(`compile.trainingStep Zig planner returned unsupported training plan: ${JSON.stringify(record)}`);
  }
  const modelKindId = numberField(record, "model_kind");
  const optimizerId = numberField(record, "optimizer_kind");
  const workspace = {};
  const workspaceFields = modelKindId === nativeTrainingPlanIds.model.linear
    ? {
      output: "workspace_output",
      gradWeight: "workspace_grad_weight",
      batchInput: "workspace_batch_input",
      batchTarget: "workspace_batch_targets",
    }
    : {
      hidden: "workspace_hidden",
      logits: "workspace_logits",
      gradHidden: "workspace_grad_hidden",
      gradW1: "workspace_grad_w1",
      gradW2: "workspace_grad_w2",
      batchInput: "workspace_batch_input",
      batchTargets: "workspace_batch_targets",
    };
  for (const [name, key] of Object.entries(workspaceFields)) {
    const value = numberField(record, key);
    if (value !== 0) workspace[name] = value;
  }
  return {
    modelKind: modelKindId === nativeTrainingPlanIds.model.linear ? "linear" : "sequential-mlp-relu",
    optimizerKind: optimizerId === nativeTrainingPlanIds.optimizer.sgd ? "sgd" : optimizerId === nativeTrainingPlanIds.optimizer.adamw ? "adamw" : "adam",
    lossKind: numberField(record, "loss_kind") === nativeTrainingPlanIds.loss.mse ? "mse" : "crossEntropy",
    inputShape: [numberField(record, "batch"), numberField(record, "in_features")],
    outputShape: [numberField(record, "batch"), numberField(record, "out_features")],
    parameterCount: numberField(record, "parameter_count"),
    parameterElements: numberField(record, "parameter_elements"),
    kernels: [kernel],
    workspace,
  };
}

function compileNodeNativeTrainingPlan(request) {
  const out = {};
  const statusCode = nodeSymbolGroups.nativeTraining.trainingPlanF32({
    model_kind: requiredPlanId(nativeTrainingPlanIds.model, request.modelKind, "model kind"),
    optimizer_kind: requiredPlanId(nativeTrainingPlanIds.optimizer, request.optimizerKind, "optimizer kind"),
    loss_kind: requiredPlanId(nativeTrainingPlanIds.loss, request.lossKind, "loss kind"),
    reserved: 0,
    batch: request.batch,
    in_features: request.inFeatures,
    hidden_features: request.hiddenFeatures,
    out_features: request.outFeatures,
  }, out);
  check(statusCode);
  return nativeTrainingPlanFields(out);
}

const nativeTraining = createAdapterNativeTrainingSurface({
  f32: (value, label) => f32(value, label),
  indexValues,
  check,
  compileTrainingPlan: compileNodeNativeTrainingPlan,
  trainLinearMseSgdF32: (args) => {
    const outLoss = new Float32Array(1);
    const statusCode = nodeSymbolGroups.nativeTraining.trainLinearMseSgdF32(
      args.input,
      args.input.length,
      args.target,
      args.target.length,
      args.weight,
      args.weight.length,
      args.bias,
      args.bias.length,
      args.output,
      args.output.length,
      args.gradWeight,
      args.gradWeight.length,
      args.batch,
      args.inFeatures,
      args.outFeatures,
      args.lr,
      args.weightDecay,
      outLoss,
    );
    return { status: statusCode, loss: outLoss[0], correct: 0 };
  },
  trainLinearMseSgdBulkF32: (args) => callNodeNativeLinearBulkTraining(
    nodeSymbolGroups.nativeTraining.trainLinearMseSgdBulkF32,
    args,
  ),
  trainMlpReluCrossEntropyAdamF32: (args) => callNodeNativeMlpTraining(
    nodeSymbolGroups.nativeTraining.trainMlpReluCrossEntropyAdamF32,
    args,
  ),
  trainMlpReluCrossEntropyAdamWF32: (args) => callNodeNativeMlpTraining(
    nodeSymbolGroups.nativeTraining.trainMlpReluCrossEntropyAdamWF32,
    args,
  ),
  trainMlpReluCrossEntropyAdamBulkF32: (args) => callNodeNativeMlpBulkTraining(
    nodeSymbolGroups.nativeTraining.trainMlpReluCrossEntropyAdamBulkF32,
    args,
  ),
  trainMlpReluCrossEntropyAdamWBulkF32: (args) => callNodeNativeMlpBulkTraining(
    nodeSymbolGroups.nativeTraining.trainMlpReluCrossEntropyAdamWBulkF32,
    args,
  ),
});

const publicNamespaces = createAdapterFrontendNamespaces({
  sharedFrontend,
  Tensor,
  f32,
  f32WithLength,
  indexValues,
  addTensorGrad,
  isGradEnabled,
  makeParameter,
  zeroGrad,
  resolveParameters,
  finiteConfigNumber,
  optimizerStateEntry,
  optimizerStateDict,
  optimizerStateEntries,
  optimizerStepFromState,
  loadOptimizerTensorState,
  rejectUnexpectedOptimizerState,
  compileTrainingStep: (...args) => {
    if (compileTrainingStepHook === null) {
      throw new Error("zgml Node FFI train.fitModule native compile hook was called before compile namespace initialization");
    }
    return compileTrainingStepHook(...args);
  },
  nativeEagerLinearInto,
  LinearModule,
  EmbeddingModule,
  Conv2dModule,
  AvgPool2dModule,
  MaxPool2dModule,
  SequentialModule,
  ActivationModule,
  SoftmaxModule,
  LogSoftmaxModule,
  ReductionModule,
  DropoutModule,
  ShapeModule,
  FeatureNormModule,
  geluScalar,
  siluScalar,
  resolveParameters,
  parameterNames,
  parameterInfos,
  parameterInfo,
  zeroGrad,
  setRequiresGrad,
  stateDict,
  loadStateDict,
  traceSequentialProgram,
  traceModule,
  compileSupportForModule,
  requireCompileSupportForModule,
  compilerSignaturesForModule,
  tensorProgramIrForModule,
  kernelPlanForModule,
  bufferLayoutForModule,
  memoryLayoutForModule,
  inputShapeForModule,
  outputShapeForModule,
  shapeConstraintsForModule,
  parameterLayoutForModule,
  explainModule,
  requireCompilePlanForModule,
  canCompileModule,
  compileModule,
  bindModuleParameters,
  placeModuleParameters,
  bindingPlanForModuleBindings,
  requireBindingPlanForModuleBindings,
  tensor,
  stack,
});
const {
  meanSquaredError,
  classTargets,
  crossEntropy,
  loss,
  train,
  data,
  SgdOptimizer,
  AdamOptimizer,
  AdamWOptimizer,
  RMSpropOptimizer,
  AdagradOptimizer,
  nn,
  optim,
  checkpoint,
} = publicNamespaces;
const F = nn.F;
const compile = createAdapterCompileNamespace({
  traceSequentialProgram,
  analyzeSequentialProgram,
  SequentialModule,
  compileModuleProgram,
  trainingStep: nativeTraining.trainingStep,
});
compileTrainingStepHook = compile.compileForTraining;
sharedFrontend.lazy.setLazyTensorProgramCompiler((lazyGraph, compileOptions) => compile.compile(lazyGraph, compileOptions));
const zgmlCheckpointIo = createAdapterZgmlCheckpointIo(checkpoint, {
  readTextFile: (path) => readFileSync(path, "utf8"),
  writeTextFile: (path, text) => writeFileSync(path, text, "utf8"),
});
const simple = Object.freeze({
  Tensor,
  tensor,
  nn,
  F,
  functional: F,
  compile,
  native: compile.compileForInference,
  inference: compile.compileForInference,
  forInference: compile.compileForInference,
  for_inference: compile.compile_for_inference,
  compileInference: compile.compileForInference,
  compileForInference: compile.compileForInference,
  run: compile.run,
  infer: compile.infer,
  predict: compile.predict,
  runInto: compile.runInto,
  inferInto: compile.inferInto,
  predictInto: compile.predictInto,
  trainingStep: compile.trainingStep,
  compileForTraining: compile.compileForTraining,
  forTraining: compile.forTraining,
  for_training: compile.for_training,
  nativeCore,
  lazy: sharedFrontend.lazy,
  optim,
  data,
  loss,
  train,
  fit: train.fit,
  fitModule: train.fitModule,
  fit_module: train.fit_module,
  explainNative: train.explainNative,
  explain_native: train.explain_native,
  nativeTrainingPlan: train.nativePlan,
  native_training_plan: train.native_plan,
  canTrainNative: train.canTrainNative,
  can_train_native: train.can_train_native,
  fitNative: train.fitNative,
  fit_native: train.fit_native,
  checkpoint,
  save: zgmlCheckpointIo.save,
  load: zgmlCheckpointIo.load,
  noGrad,
  no_grad,
  inferenceMode,
  inference_mode,
}) as unknown as PublicApi.PublicSimpleNamespace;
const gradMode = createAdapterGradModeSurface({
  isGradEnabled,
  is_grad_enabled,
  setGradEnabled,
  set_grad_enabled,
  noGrad,
  no_grad,
  inferenceMode,
  inference_mode,
  enableGrad,
  enable_grad,
});
const zgml = createAdapterZgmlNamespace({
  Tensor,
  tensor,
  parameter,
  param,
  factories: {
    empty,
    emptyLike,
    empty_like,
    zeros,
    zerosLike,
    zeros_like,
    ones,
    onesLike,
    ones_like,
    full,
    fullLike,
    full_like,
    eye,
    scalar,
    rand,
    randLike,
    rand_like,
    randn,
    randnLike,
    randn_like,
    randint,
    randInt,
    randperm,
    randPerm,
    manual_seed,
    manualSeed,
    initialSeed,
    initial_seed,
    seededRng,
    arange,
    linspace,
    cat,
    concat,
    concatenate,
    stack,
    vstack,
    hstack,
    einsum,
    broadcastTo,
    expand,
    repeat,
    tile,
    scatterAdd,
    scatter_add,
    add,
    sub,
    mul,
    div,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    isclose,
    matmul,
    mm,
    dot,
    trace,
    diagonal,
    bmm,
    clone,
    detach,
  },
  shape: { hasShape, requireShape },
  grad: {
    no_grad,
    noGrad,
    inference_mode,
    inferenceMode,
    enable_grad,
    enableGrad,
    is_grad_enabled,
    isGradEnabled,
    set_grad_enabled,
    setGradEnabled,
  },
  gradMode,
  nn,
  F,
  compile,
  lazy: sharedFrontend.lazy,
  optim,
  data,
  loss,
  train,
  checkpoint,
  checkpointIo: zgmlCheckpointIo,
  Program,
  Session,
  NativeBuffer,
  nativeEager,
  nativeCore,
  native_core,
});
const torch = zgml;

const { readSafetensorsHeaderFile } = createAdapterSafetensorsFileHeaderHelpers(fs);

modelSourceFacadeSlot.bind(createAdapterModelSourceFacade({
  readSafetensorsHeaderFile,
  probeModelPath,
  probeSafetensorsData,
  probeSafetensorsHeader,
  loadModelPath,
  loadSafetensorsDataHandle,
  tinyLlama2LayerKind,
  LlamaModel,
  TinyLlamaModel,
  SmolLM135MModel,
}));

module.exports = createAdapterPublicRuntimeExports({
  Tensor,
  tensor,
  asTensor: tensor,
  as_tensor: tensor,
  asarray: tensor,
  fromNumpy: tensor,
  from_numpy: tensor,
  parameter,
  param,
  cat,
  concat,
  concatenate,
  stack,
  vstack,
  hstack,
  einsum,
  allclose,
  equal,
  to,
  cpu,
  float,
  float32,
  typeAs,
  type_as,
  clone,
  detach,
  reshape,
  view,
  broadcastTo,
  expand,
  repeat,
  tile,
  flatten,
  squeeze,
  unsqueeze,
  transpose,
  permute,
  flip,
  roll,
  select,
  narrow,
  slice,
  indexSelect,
  index_select,
  gather,
  take,
  argsort,
  sort,
  topk,
  scatterAdd,
  scatter_add,
  split,
  chunk,
  unbind,
  add,
  sub,
  mul,
  div,
  isGradEnabled,
  is_grad_enabled,
  setGradEnabled,
  set_grad_enabled,
  noGrad,
  no_grad,
  inferenceMode,
  inference_mode,
  enableGrad,
  enable_grad,
  gradMode,
  simple,
  zgml,
  torch,
  eq,
  ne,
  lt,
  le,
  gt,
  ge,
  isclose,
  pow,
  neg,
  negative,
  exp,
  expm1,
  log,
  log1p,
  sqr,
  square,
  recip,
  reciprocal,
  abs,
  sgn,
  sign,
  step,
  isnan,
  isinf,
  isfinite,
  floor,
  ceil,
  round,
  trunc,
  sqrt,
  rsqrt,
  relu,
  gelu,
  silu,
  sigmoid,
  tanh,
  sin,
  cos,
  tan,
  maximum,
  minimum,
  where,
  maskedFill,
  masked_fill,
  sum,
  prod,
  cumsum,
  mean,
  max,
  min,
  any,
  all,
  argmax,
  argmin,
  variance,
  std,
  norm,
  softmax,
  softmax_dim,
  softmaxDim,
  logSoftmax,
  log_softmax,
  log_softmax_dim,
  logSoftmaxDim,
  logsumexp,
  logSumExp,
  clamp,
  clip,
  matmul,
  mm,
  dot,
  trace,
  diagonal,
  bmm,
  full,
  fullLike,
  full_like,
  empty,
  emptyLike,
  empty_like,
  zeros,
  zerosLike,
  zeros_like,
  ones,
  onesLike,
  ones_like,
  eye,
  scalar,
  rand,
  randLike,
  rand_like,
  randn,
  randnLike,
  randn_like,
  randInt,
  randint,
  randPerm,
  randperm,
  manualSeed,
  manual_seed,
  initialSeed,
  initial_seed,
  seededRng,
  hasShape,
  requireShape,
  linspace,
  arange,
  nn,
  F,
  data,
  loss,
  optim,
  train,
  fit: train.fit,
  fitModule: train.fitModule,
  fit_module: train.fit_module,
  explainNative: train.explainNative,
  explain_native: train.explain_native,
  nativeTrainingPlan: train.nativePlan,
  native_training_plan: train.native_plan,
  canTrainNative: train.canTrainNative,
  can_train_native: train.can_train_native,
  fitNative: train.fitNative,
  fit_native: train.fit_native,
  checkpoint,
  save: zgml.save,
  load: zgml.load,
  compile,
  native: compile.compileForInference,
  inference: compile.compileForInference,
  forInference: compile.compileForInference,
  for_inference: compile.compile_for_inference,
  compileInference: compile.compileForInference,
  compile_inference: compile.compile_for_inference,
  compileForInference: compile.compileForInference,
  compile_for_inference: compile.compile_for_inference,
  run: compile.run,
  infer: compile.infer,
  predict: compile.predict,
  runInto: compile.runInto,
  run_into: compile.run_into,
  inferInto: compile.inferInto,
  infer_into: compile.infer_into,
  predictInto: compile.predictInto,
  predict_into: compile.predict_into,
  trainingStep: compile.trainingStep,
  training_step: compile.training_step,
  compileForTraining: compile.compileForTraining,
  compile_for_training: compile.compile_for_training,
  forTraining: compile.forTraining,
  for_training: compile.for_training,
  fit: train.fit,
  fitModule: train.fitModule,
  fit_module: train.fit_module,
  explainNative: train.explainNative,
  explain_native: train.explain_native,
  nativeTrainingPlan: train.nativePlan,
  native_training_plan: train.native_plan,
  canTrainNative: train.canTrainNative,
  can_train_native: train.can_train_native,
  fitNative: train.fitNative,
  fit_native: train.fit_native,
  nativeEager,
  native_eager,
  nativeCore,
  native_core,
  TinyLinear: TinyLinearModel,
  TinyMlp: TinyMlpModel,
  TinyLlama: TinyLlamaModel,
  SmolLM135M: SmolLM135MModel,
  Llama: LlamaModel,
  probeModel,
  probeSafetensorsData,
  probeSafetensorsHeader,
  supportedCheckpointModels,
  loadModel,
  loadSafetensorsData,
  TinyLinearModel,
  TinyMlpModel,
  Program,
  Session,
  TinyLinearProgram,
  TinyLinearSession,
  TinyLlamaModel,
  TinyLlamaProgram,
  TinyLlamaSession,
  SmolLM135MModel,
  SmolLM135MProgram,
  SmolLM135MSession,
  LlamaModel,
  LlamaProgram,
  LlamaSession,
  NativeBuffer,
  ProgramDevice,
  LlamaKvCache,
  ZgmlError,
  runtimeInfo,
  abiStructSize,
  abiStructSizes,
  loadedRuntimeInfo,
  webgpuInterop,
}, "Node public runtime exports");
