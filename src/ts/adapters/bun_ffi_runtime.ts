import { createRequire } from "node:module";
import { closeSync, existsSync, openSync, readSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  ptr,
  readPointer,
} from "./bun_ffi_intrinsics.js";
import {
  bindBunSymbols,
} from "./bun_symbols.js";
import {
  createBunSymbolGroups,
} from "./bun_symbol_groups.js";
import {
  createBunHostRuntime,
} from "./bun_host_runtime.js";
import {
  createBunFfiBootstrap,
} from "./bun_ffi_bootstrap.js";
import {
  createBunNativeLifecycleOps,
} from "./bun_native_lifecycle_ops.js";
import {
  createBunNativeBufferSyscalls,
} from "./bun_native_buffer_syscalls.js";
import {
  createBunInspectionOps,
} from "./bun_inspection_ops.js";
import {
  createBunSessionOps,
} from "./bun_session_ops.js";
import {
  createBunLlamaTokenOps,
} from "./bun_llama_token_ops.js";
import {
  createBunProgramBufferOps,
} from "./bun_program_buffer_ops.js";
import {
  createBunLlamaSessionBindOps,
} from "./bun_llama_session_bind_ops.js";
import {
  createBunLlamaKvCacheOps,
} from "./bun_llama_kv_cache_ops.js";
import {
  createBunLlamaKvCacheClass,
  type BunLlamaKvCache,
  type BunLlamaKvCacheConstructor,
} from "./bun_llama_kv_cache_surface.js";
import {
  createBunModelSourceOps,
} from "./bun_model_source_ops.js";
import {
  createBunModuleProgramOps,
} from "./bun_module_program_ops.js";
import type {
  SingleModuleRecord,
} from "../nn/single_module_compile.js";
import {
  createBunProgramBindOps,
} from "./bun_program_bind_ops.js";
import {
  createBunGenericFamilySurface,
} from "./bun_generic_family_surface.js";
import {
  NativeBuffer,
  setAdapterNativeBufferFacade,
} from "./native_buffer_surface.js";
import {
  requireBunSharedFrontendRuntime,
} from "./bun_shared_frontend_runtime.js";
import {
  createAdapterSharedFrontendProjection,
} from "./shared_frontend_projection.js";
import {
  createAdapterDeferredSlot,
} from "./deferred_slot.js";
import {
  Tensor,
  setAdapterTensorSurfaceHelpers,
} from "./tensor_surface.js";
import {
  createAdapterTensorRuntimeSurface,
} from "./tensor_runtime_surface.js";
import {
  createAdapterModelHandlePolicy,
} from "./model_handle_policy.js";
import {
  createAdapterModelSourceFacade,
} from "./model_source_facade.js";
import {
  createAdapterModelSourceSurface,
  type AdapterModelSourceFacade,
} from "./model_source_surface.js";
import {
  createAdapterNativeBufferInstanceSurface,
} from "./native_buffer_instance_surface.js";
import {
  createAdapterNativeBufferFacadeSurface,
} from "./native_buffer_facade_surface.js";
import {
  createAdapterIndexValuesSurface,
} from "./index_values_surface.js";
import {
  createAdapterGradModeSurface,
} from "./grad_mode_surface.js";
import {
  createAdapterModuleCompilerSurface,
} from "./module_compiler_surface.js";
import {
  createAdapterModuleStateSurface,
} from "./module_state_surface.js";
import {
  createAdapterProgramBufferNativeBridge,
} from "./program_buffer_native_bridge.js";
import {
  createAdapterProgramRuntimeSurface,
} from "./program_runtime_surface.js";
import {
  createAdapterProgramBufferFactorySurface,
} from "./program_buffer_factory_surface.js";
import {
  createAdapterSafetensorsFileHeaderHelpers,
} from "./safetensors_file_header.js";
import {
  createAdapterProgramFactory,
} from "./program_factory_surface.js";
import {
  createAdapterSessionFactory,
} from "./session_factory_surface.js";
import {
  createAdapterSessionRuntimeSurface,
} from "./session_runtime_surface.js";
import {
  createAdapterLlamaSessionFacadeSurface,
} from "./llama_session_facade_surface.js";
import {
  createAdapterCompileNamespace,
  createAdapterFrontendNamespaces,
  createAdapterTorchCheckpointIo,
  createAdapterTorchNamespace,
} from "./frontend_namespace_surface.js";
import {
  createAdapterFrontendModuleSurface,
} from "./frontend_module_surface.js";
import {
  createAdapterNativeEagerSurface,
} from "./native_eager_surface.js";
import {
  adapterBufferStorageAliases,
  adapterModelKindAliases,
} from "../runtime/native_abi_constants.js";
import {
  bunGenericModelDesc,
} from "./bun_generic_model_desc.js";
import {
  type HostTraceModuleCompiler,
} from "../runtime/host_adapter_surfaces.js";
import {
  webgpuBufferHandle,
  webgpuByteLength,
  webgpuByteOffset,
  webgpuDeviceHandle,
  webgpuImportSource,
  webgpuInterop,
  webgpuPlacement,
} from "./webgpu_interop.js";
import {
  createAdapterLlamaFamilySurface,
} from "./llama_family_surface.js";
import type {
  AllCloseOptions as PublicAllCloseOptions,
  IsCloseOptions as PublicIsCloseOptions,
  BufferInspection as PublicBufferInspection,
  CheckpointOptimizerEntry as PublicCheckpointOptimizerEntry,
  CheckpointCreateOptions as PublicCheckpointCreateOptions,
  CheckpointInspection as PublicCheckpointInspection,
  CheckpointModuleState as PublicCheckpointModuleState,
  CheckpointNamespace as PublicCheckpointNamespace,
  CheckpointOptimizerState as PublicCheckpointOptimizerState,
  CheckpointRestoreOptions as PublicCheckpointRestoreOptions,
  CheckpointSchedulerState as PublicCheckpointSchedulerState,
  CheckpointTensorInspection as PublicCheckpointTensorInspection,
  CheckpointTensorEntry as PublicCheckpointTensorEntry,
  CollatedDataLoader as PublicCollatedDataLoader,
  CompileOptions as PublicCompileOptions,
  DataBatches as PublicDataBatches,
  DataBatchOptions as PublicDataBatchOptions,
  DataCollateFn as PublicDataCollateFn,
  DataCollateOptions as PublicDataCollateOptions,
  DataLoader as PublicDataLoader,
  DataNamespace as PublicDataNamespace,
  ExecuteTokensOptions as PublicExecuteTokensOptions,
  ExternalResourceOptions as PublicExternalResourceOptions,
  LlamaBindOptions as PublicLlamaBindOptions,
  LlamaExecuteIntoParams as PublicLlamaExecuteIntoParams,
  LlamaExecuteParams as PublicLlamaExecuteParams,
  LlamaExecuteTensorParams as PublicLlamaExecuteTensorParams,
  LlamaKvCacheLayout as PublicLlamaKvCacheLayout,
  LlamaKvCacheLayoutSlot as PublicLlamaKvCacheLayoutSlot,
  LlamaKvCacheRequirements as PublicLlamaKvCacheRequirements,
  LlamaKvCacheSlot as PublicLlamaKvCacheSlot,
  LlamaLogitsTensorOptions as PublicLlamaLogitsTensorOptions,
  LlamaProgramInspection as PublicLlamaProgramInspection,
  LlamaSessionStepContract as PublicLlamaSessionStepContract,
  LlamaStepParams as PublicLlamaStepParams,
  LlamaTokenWindowTensorOptions as PublicLlamaTokenWindowTensorOptions,
  LoadModelOptions as PublicLoadModelOptions,
  LoadStateDictOptions as PublicLoadStateDictOptions,
  LRScheduler as PublicLRScheduler,
  LRSchedulerConfig as PublicLRSchedulerConfig,
  LRSchedulerStateDict as PublicLRSchedulerStateDict,
  LRSchedulerStateKind as PublicLRSchedulerStateKind,
  LRSchedulerStateSnapshot as PublicLRSchedulerStateSnapshot,
  ModelInspection as PublicModelInspection,
  ModuleCompileSupport as PublicModuleCompileSupport,
  ModuleKernelBufferLayout as PublicModuleKernelBufferLayout,
  ModuleKernelMemoryLayout as PublicModuleKernelMemoryLayout,
  ModuleKernelParameterLayout as PublicModuleKernelParameterLayout,
  ModuleKernelPlan as PublicModuleKernelPlan,
  ModuleParameterInfo as PublicModuleParameterInfo,
  ModuleProgramTrace as PublicModuleProgramTrace,
  ModuleStateDict as PublicModuleStateDict,
  ModuleStateEntry as PublicModuleStateEntry,
  ModuleStateSnapshot as PublicModuleStateSnapshot,
  ModuleStateSnapshotEntry as PublicModuleStateSnapshotEntry,
  ModuleStateValue as PublicModuleStateValue,
  ModuleTensorProgramIr as PublicModuleTensorProgramIr,
  ModuleTraversalEntry as PublicModuleTraversalEntry,
  NnBuffer as PublicNnBuffer,
  NnModule as PublicNnModule,
  NnConv2dConfig as PublicNnConv2dConfig,
  NnDropoutConfig as PublicNnDropoutConfig,
  NnEmbeddingConfig as PublicNnEmbeddingConfig,
  NnLinearConfig as PublicNnLinearConfig,
  NnNormConfig as PublicNnNormConfig,
  NnParameter as PublicNnParameter,
  AdamConfig as PublicAdamConfig,
  AdagradConfig as PublicAdagradConfig,
  RMSpropConfig as PublicRMSpropConfig,
  Optimizer as PublicOptimizer,
  OptimizerConfigSnapshot as PublicOptimizerConfigSnapshot,
  OptimizerParameterSource as PublicOptimizerParameterSource,
  OptimizerParamGroupInput as PublicOptimizerParamGroupInput,
  OptimizerParamGroupSnapshot as PublicOptimizerParamGroupSnapshot,
  OptimizerTarget as PublicOptimizerTarget,
  OptimizerStateDict as PublicOptimizerStateDict,
  OptimizerStateEntry as PublicOptimizerStateEntry,
  OptimizerStateKind as PublicOptimizerStateKind,
  OptimizerStateSnapshot as PublicOptimizerStateSnapshot,
  OptimizerStateSnapshotEntry as PublicOptimizerStateSnapshotEntry,
  ProgramBufferKind as PublicProgramBufferKind,
  ProgramBufferLayout as PublicProgramBufferLayout,
  ProgramBufferSlot as PublicProgramBufferSlot,
  ProgramBufferSizing as PublicProgramBufferSizing,
  ProgramDispatchPlanDiagnostic as PublicProgramDispatchPlanDiagnostic,
  ProgramDeviceBufferKind as PublicProgramDeviceBufferKind,
  ProgramExecutionCapabilities as PublicProgramExecutionCapabilities,
  ProgramExecutionMode as PublicProgramExecutionMode,
  ProgramExecutionPlan as PublicProgramExecutionPlan,
  ProgramExecutionUnavailableDiagnostic as PublicProgramExecutionUnavailableDiagnostic,
  ProgramInspection as PublicProgramInspection,
  ProgramCompileEvidence as PublicProgramCompileEvidence,
  ProgramModelCompatibility as PublicProgramModelCompatibility,
  ProgramModuleCompatibility as PublicProgramModuleCompatibility,
  ProgramOutputBufferSlot as PublicProgramOutputBufferSlot,
  ProgramRequirements as PublicProgramRequirements,
  ProgramRuntimeDiagnostic as PublicProgramRuntimeDiagnostic,
  RandomIntTensorOptions as PublicRandomIntTensorOptions,
  RandomTensorOptions as PublicRandomTensorOptions,
  RandomUniformTensorOptions as PublicRandomUniformTensorOptions,
  SGDConfig as PublicSGDConfig,
  StepLRConfig as PublicStepLRConfig,
  ZgmlCheckpoint as PublicZgmlCheckpoint,
  ZeroGradOptions as PublicZeroGradOptions,
  RuntimeFeatures as PublicRuntimeFeatures,
  RuntimeInfo as PublicRuntimeInfo,
  RuntimeProfile as PublicRuntimeProfile,
  RuntimeProfileExpectation,
  SessionBoundBufferKind as PublicSessionBoundBufferKind,
  SessionBufferSizing as PublicSessionBufferSizing,
  SessionCallProfile as PublicSessionCallProfile,
  SessionDefaultOutputKind as PublicSessionDefaultOutputKind,
  SessionExecuteIntoParams as PublicSessionExecuteIntoParams,
  SessionExecuteParams as PublicSessionExecuteParams,
  SessionExecuteTensorParams as PublicSessionExecuteTensorParams,
  SessionExecutionPlan as PublicSessionExecutionPlan,
  SessionInspection as PublicSessionInspection,
  SessionNoOutputEffect as PublicSessionNoOutputEffect,
  SessionReadOutputTensorOptions as PublicSessionReadOutputTensorOptions,
  SessionStepContract as PublicSessionStepContract,
  SessionStepParams as PublicSessionStepParams,
  SessionStepParamsCompatibility as PublicSessionStepParamsCompatibility,
  SessionStepParamsDiagnostic as PublicSessionStepParamsDiagnostic,
  SessionStepParamsDiagnosticCode as PublicSessionStepParamsDiagnosticCode,
  SessionStepParamsElementType as PublicSessionStepParamsElementType,
  SessionStepParamsHotPathBlocker as PublicSessionStepParamsHotPathBlocker,
  SessionStepParamsHotPathStatus as PublicSessionStepParamsHotPathStatus,
  SessionStepParamsInputOwnership as PublicSessionStepParamsInputOwnership,
  SessionStepParamsInputSource as PublicSessionStepParamsInputSource,
  SessionStepParamsOutputEffect as PublicSessionStepParamsOutputEffect,
  SessionStepParamsOutputOwnership as PublicSessionStepParamsOutputOwnership,
  SessionStepParamsOutputReturnOwnership as PublicSessionStepParamsOutputReturnOwnership,
  SessionStepParamsOutputTarget as PublicSessionStepParamsOutputTarget,
  SessionStepParamsStateEffect as PublicSessionStepParamsStateEffect,
  SessionStepTensorOptions as PublicSessionStepTensorOptions,
  TensorFromNativeBufferOptions as PublicTensorFromNativeBufferOptions,
  TensorInspection as PublicTensorInspection,
  TensorLike as PublicTensorLike,
  TensorNativeBufferOptions as PublicTensorNativeBufferOptions,
  TensorOptions as PublicTensorOptions,
  TensorShape as PublicTensorShape,
  TensorShapeOf as PublicTensorShapeOf,
  TensorShapeTuple as PublicTensorShapeTuple,
  TensorDataset as PublicTensorDataset,
  TensorDatasetBatch as PublicTensorDatasetBatch,
  TensorDatasetBatchShape as PublicTensorDatasetBatchShape,
  TensorDatasetMapper as PublicTensorDatasetMapper,
  TensorDatasetSample as PublicTensorDatasetSample,
  TensorShapeTail as PublicTensorShapeTail,
  TensorToOptions as PublicTensorToOptions,
  TensorToTarget as PublicTensorToTarget,
  TokenArgmaxResult as PublicTokenArgmaxResult,
  TokenGenerateArgmaxResult as PublicTokenGenerateArgmaxResult,
  TokenGenerateSampleResult as PublicTokenGenerateSampleResult,
  TokenIds as PublicTokenIds,
  TokenSampleResult as PublicTokenSampleResult,
  TokenSampleOptions as PublicTokenSampleOptions,
  TokenWindowOptions as PublicTokenWindowOptions,
  TrainFitContext as PublicTrainFitContext,
  TrainFitEvidence as PublicTrainFitEvidence,
  TrainFitOptions as PublicTrainFitOptions,
  TrainFitStepEvidence as PublicTrainFitStepEvidence,
  TrainLossStepOptions as PublicTrainLossStepOptions,
  TrainNamespace as PublicTrainNamespace,
  TrainStepEvidence as PublicTrainStepEvidence,
  TrainStepOptions as PublicTrainStepOptions,
} from "../public_api.js";
import type * as PublicApi from "../public_api.js";
import type {
  NormalizedTokenSampleOptions as CoreNormalizedTokenSampleOptions,
} from "../core/token.js";
import type {
  NativeBufferFacade,
} from "../runtime/native_buffer.js";

declare const __filename: string;

const moduleUrl = pathToFileURL(__filename).href;
const requireSharedFrontend = createRequire(moduleUrl);

const sharedFrontend = requireBunSharedFrontendRuntime({
  requireModule: requireSharedFrontend,
  origin: moduleUrl,
});

const {
  validateTensorShape,
  rowMajorStrides,
  normalizeDim,
  normalizeFactoryShape,
  modelKinds,
  backendIds,
  bufferStorageIds,
  programBufferKinds,
  abiStructKinds,
  backendId,
  backendName,
  bufferStorageName,
  accessFlagsToObject,
  normalizeWebGpuImportSource,
  programDeviceBufferImportInfo,
  programBufferKindId,
  modelKindName,
  normalizeLoadModelKind,
  modelLoadKindId,
  tokenId,
  normalizeCompileOptions,
  tokenSelectionResultFromAbiWords,
  tokenGenerateResultFromAbiWords,
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
  packedProgramWeightsLen,
  packedProgramBiasLen,
  programInputShape,
  programOutputShape,
  sessionOutputTensorShape,
  runtimeProfileFromAbiWords,
  programInspectionFromAbiWords,
  sessionInspectionFromAbiWords,
  bufferInspectionFromAbiWords,
  modelInspectionFromAbiWords,
  programRequirementsFromAbiWords,
  llamaKvCacheRequirementsFromAbiWords,
  programModelCompatibilityFromAbiWords,
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
  bindLlamaSessionHandleByPolicy,
  safetensorsHeaderBytes,
  safetensorsDataBytes,
  geluScalar,
  siluScalar,
  isGradEnabled: projectedIsGradEnabled,
  is_grad_enabled: projectedIsGradEnabledSnake,
  setGradEnabled: projectedSetGradEnabled,
  set_grad_enabled: projectedSetGradEnabledSnake,
  noGrad: projectedNoGrad,
  no_grad: projectedNoGradSnake,
  inferenceMode: projectedInferenceMode,
  inference_mode: projectedInferenceModeSnake,
  enableGrad: projectedEnableGrad,
  enable_grad: projectedEnableGradSnake,
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
const ok = 0;
const {
  bufferStorageExternalResource,
} = adapterBufferStorageAliases(bufferStorageIds);
export { webgpuInterop };
export type AbiStructKind = keyof typeof abiStructKinds;

type NativeHandle = number;

let modelHandleForBind: (model: unknown) => NativeHandle | null;
let modelHandleForCompatibility: (model: unknown) => NativeHandle;
type GenericProgramInstance = Readonly<{
  readonly handle: NativeHandle;
  parameterLayout(): ProgramBufferLayout;
}>;
export type Program = GenericProgramInstance;

const { packageRoot, suffix, libPath } = createBunHostRuntime({
  importMetaUrl: moduleUrl,
  dirname,
  fileURLToPath,
  joinPath: join,
  exists: existsSync,
});

const symbols = bindBunSymbols(libPath);
const bunSymbolGroups = createBunSymbolGroups(symbols);

export { ZgmlError } from "./bun_status.js";

export type TinyLinearDesc = PublicApi.TinyLinearDesc;

export type TinyMlpDesc = PublicApi.TinyMlpDesc;

type ModuleProgramOpDesc = {
  kind: number;
  activation?: number;
  flags?: number;
  reserved?: number;
  a?: number;
  b?: number;
  c?: number;
  eps?: number;
};

type ModuleProgramDesc = TinyLinearDesc & {
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
  weightsLen: number;
  biasLen: number;
  ops: readonly ModuleProgramOpDesc[];
};

export type ProgramHostBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = PublicApi.ProgramHostBinding<Shape>;
export type ProgramInputBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = PublicApi.ProgramInputBinding<Shape>;
export type ProgramOutputBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = PublicApi.ProgramOutputBinding<Shape>;
export type ProgramBindings<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> =
  PublicApi.ProgramBindings<InputShape, OutputShape>;
export type ModuleBindings<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> =
  PublicApi.ModuleBindings<InputShape, OutputShape>;
export type ProgramBindingDiagnostic = PublicApi.ProgramBindingDiagnostic;
export type ProgramBindingMode = PublicApi.ProgramBindingMode;
export type ProgramBindingPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> =
  PublicApi.ProgramBindingPlan<InputShape, OutputShape>;
export type TinyLinearWeights = PublicApi.TinyLinearWeights;
export type ModuleParameterPlacementOptions = PublicApi.ModuleParameterPlacementOptions;

export type TensorNestedArray = PublicApi.TensorNestedArray;
export type TensorLike = PublicTensorLike;
export type ByteLike = ArrayBufferView | ArrayBuffer | readonly number[];
export type TensorShapeTuple = PublicTensorShapeTuple;
export type TensorShape = PublicTensorShape;
export type TensorShapeOf<S extends TensorShape> = PublicTensorShapeOf<S>;
export type IndexLike = Tensor | Uint32Array | Int32Array | Float32Array | readonly number[];
export type AllCloseOptions = PublicAllCloseOptions;
export type IsCloseOptions = PublicIsCloseOptions;
export type RandomIntTensorOptions = PublicRandomIntTensorOptions;
export type TensorNativeBufferOptions = PublicTensorNativeBufferOptions;
export type TensorDType = "f32";
export type TensorDTypeLike = TensorDType | "float32";
export type TensorDevice = "cpu";
export type TensorDeviceLike = TensorDevice | "host" | ZgmlBackend;
export type TensorInspection = PublicTensorInspection;
export type TensorToTarget = PublicTensorToTarget;
export type TensorToOptions = PublicTensorToOptions;
export type TensorFromNativeBufferOptions = PublicTensorFromNativeBufferOptions;
export type SessionStepTensorOptions = PublicSessionStepTensorOptions;
export type SessionStepParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = PublicSessionStepParams<InputShape, OutputShape>;
export type SessionExecuteParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = PublicSessionExecuteParams<InputShape, OutputShape>;
export type SessionExecuteTensorParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = PublicSessionExecuteTensorParams<InputShape, OutputShape>;
export type SessionExecuteIntoParams<InputShape extends TensorShapeTuple = TensorShapeTuple> = PublicSessionExecuteIntoParams<InputShape>;
export type SessionReadOutputTensorOptions = PublicSessionReadOutputTensorOptions;
export type SessionCallProfile = PublicSessionCallProfile;
export type SessionBoundBufferKind = PublicSessionBoundBufferKind;
export type SessionDefaultOutputKind = PublicSessionDefaultOutputKind;
export type SessionNoOutputEffect = PublicSessionNoOutputEffect;
export type SessionStepContract = PublicSessionStepContract;
export type SessionStepParamsDiagnosticCode = PublicSessionStepParamsDiagnosticCode;
export type SessionStepParamsDiagnostic = PublicSessionStepParamsDiagnostic;
export type SessionStepParamsInputSource = PublicSessionStepParamsInputSource;
export type SessionStepParamsInputOwnership = PublicSessionStepParamsInputOwnership;
export type SessionStepParamsOutputTarget = PublicSessionStepParamsOutputTarget;
export type SessionStepParamsOutputEffect = PublicSessionStepParamsOutputEffect;
export type SessionStepParamsOutputOwnership = PublicSessionStepParamsOutputOwnership;
export type SessionStepParamsOutputReturnOwnership = PublicSessionStepParamsOutputReturnOwnership;
export type SessionStepParamsStateEffect = PublicSessionStepParamsStateEffect;
export type SessionStepParamsHotPathStatus = PublicSessionStepParamsHotPathStatus;
export type SessionStepParamsHotPathBlocker = PublicSessionStepParamsHotPathBlocker;
export type SessionStepParamsElementType = PublicSessionStepParamsElementType;
export type SessionStepParamsCompatibility = PublicSessionStepParamsCompatibility;
export type SessionExecutionPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = PublicSessionExecutionPlan<InputShape, OutputShape>;

export type NnParameter = PublicNnParameter;
export type ModuleParameterInfo = PublicModuleParameterInfo;
export type ModuleTraversalEntry = PublicModuleTraversalEntry;

export type ModuleStateEntry = PublicModuleStateEntry;
export type ModuleStateSnapshotEntry = PublicModuleStateSnapshotEntry;
export type ModuleStateSnapshot = PublicModuleStateSnapshot;
export type ModuleStateValue = PublicModuleStateValue;
export type ModuleStateDict = PublicModuleStateDict;
export type LoadStateDictOptions = PublicLoadStateDictOptions;
export type ZeroGradOptions = PublicZeroGradOptions;
export type OptimizerStateKind = PublicOptimizerStateKind;
export type OptimizerParameterSource = PublicOptimizerParameterSource;
export type OptimizerParamGroupInput = PublicOptimizerParamGroupInput;
export type OptimizerParamGroupSnapshot = PublicOptimizerParamGroupSnapshot;
export type OptimizerConfigSnapshot<Kind extends OptimizerStateKind = OptimizerStateKind> = PublicOptimizerConfigSnapshot<Kind>;
export type OptimizerStateEntry = PublicOptimizerStateEntry;
export type OptimizerStateSnapshot<Kind extends OptimizerStateKind = OptimizerStateKind> = PublicOptimizerStateSnapshot<Kind>;
export type OptimizerStateSnapshotEntry = PublicOptimizerStateSnapshotEntry;
export type OptimizerStateDict<Kind extends OptimizerStateKind = OptimizerStateKind> = PublicOptimizerStateDict<Kind>;
export type Optimizer<Kind extends OptimizerStateKind = OptimizerStateKind> = PublicOptimizer<Kind>;
export type LRSchedulerStateKind = PublicLRSchedulerStateKind;
export type LRSchedulerConfig = PublicLRSchedulerConfig;
export type StepLRConfig = PublicStepLRConfig;
export type LRSchedulerStateSnapshot<
  Kind extends LRSchedulerStateKind = LRSchedulerStateKind,
  OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null,
> = PublicLRSchedulerStateSnapshot<Kind, OptimizerKind>;
export type LRSchedulerStateDict<Kind extends LRSchedulerStateKind = LRSchedulerStateKind> = PublicLRSchedulerStateDict<Kind>;
export type LRScheduler<
  Kind extends LRSchedulerStateKind = LRSchedulerStateKind,
  OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null,
> = PublicLRScheduler<Kind, OptimizerKind>;
export type TrainStepEvidence<Kind extends OptimizerStateKind | null = OptimizerStateKind | null> = PublicTrainStepEvidence<Kind>;
export type TrainStepOptions = PublicTrainStepOptions;
export type TrainLossStepOptions = PublicTrainLossStepOptions;
export type TrainFitStepEvidence<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = PublicTrainFitStepEvidence<OptimizerKind>;
export type TrainFitContext = PublicTrainFitContext;
export type TrainFitOptions<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = PublicTrainFitOptions<OptimizerKind>;
export type TrainFitEvidence<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = PublicTrainFitEvidence<OptimizerKind>;
export type TrainNamespace = PublicTrainNamespace;
export type DataBatchOptions<TSample = TensorDatasetSample, TBatch = TensorDatasetBatch> = PublicDataBatchOptions<TSample, TBatch>;
export type DataCollateFn<TSample = TensorDatasetSample, TBatch = TensorDatasetBatch> = PublicDataCollateFn<TSample, TBatch>;
export type DataCollateOptions<TSample, TBatch> = PublicDataCollateOptions<TSample, TBatch>;
export type TensorShapeTail<Shape extends TensorShapeTuple> = PublicTensorShapeTail<Shape>;
export type TensorDatasetBatchShape<Shape extends TensorShapeTuple> = PublicTensorDatasetBatchShape<Shape>;
export type TensorDatasetSample<TInput = Tensor, TTarget = Tensor> = PublicTensorDatasetSample<TInput, TTarget>;
export type TensorDatasetBatch<TInput = Tensor, TTarget = Tensor> = PublicTensorDatasetBatch<TInput, TTarget>;
export type TensorDatasetMapper<TSampleInput = Tensor, TSampleTarget = Tensor, TMappedInput = Tensor, TMappedTarget = Tensor> =
  PublicTensorDatasetMapper<TSampleInput, TSampleTarget, TMappedInput, TMappedTarget>;
export type TensorDataset<TSampleInput = Tensor, TSampleTarget = Tensor, TBatchInput = TSampleInput, TBatchTarget = TSampleTarget> =
  PublicTensorDataset<TSampleInput, TSampleTarget, TBatchInput, TBatchTarget>;
export type DataBatches<TInput = Tensor, TTarget = Tensor, TBatch = TensorDatasetBatch<TInput, TTarget>> =
  PublicDataBatches<TInput, TTarget, TBatch>;
export type DataLoader<TInput = Tensor, TTarget = Tensor, TBatch = TensorDatasetBatch<TInput, TTarget>> =
  PublicDataLoader<TInput, TTarget, TBatch>;
export type CollatedDataLoader<TBatch> = PublicCollatedDataLoader<TBatch>;
export type DataNamespace = PublicDataNamespace;
export type CheckpointTensorEntry = PublicCheckpointTensorEntry;
export type CheckpointOptimizerEntry = PublicCheckpointOptimizerEntry;
export type CheckpointModuleState = PublicCheckpointModuleState;
export type CheckpointOptimizerState = PublicCheckpointOptimizerState;
export type CheckpointSchedulerState = PublicCheckpointSchedulerState;
export type ZgmlCheckpoint = PublicZgmlCheckpoint;
export type CheckpointTensorInspection = PublicCheckpointTensorInspection;
export type CheckpointInspection = PublicCheckpointInspection;
export type CheckpointCreateOptions = PublicCheckpointCreateOptions;
export type CheckpointRestoreOptions = PublicCheckpointRestoreOptions;
export type CheckpointNamespace = PublicCheckpointNamespace;
export type ModuleActivationKind = PublicApi.ModuleActivationKind;
export type ModuleReductionKind = PublicApi.ModuleReductionKind;
export type ModuleShapeOpKind = PublicApi.ModuleShapeOpKind;
export type ModuleTraceOpKind = PublicApi.ModuleTraceOpKind;
export type ModuleTensorProgramIrOpKind = PublicApi.ModuleTensorProgramIrOpKind;
export type ModuleKernelFusedOpKind = PublicApi.ModuleKernelFusedOpKind;
export type ModuleKernelPlanOpKind = PublicApi.ModuleKernelPlanOpKind;
export type ModuleKernelName = PublicApi.ModuleKernelName;
export type ModuleCompileDiagnosticStage = PublicApi.ModuleCompileDiagnosticStage;
export type ModuleShapeDiagnosticFields = PublicApi.ModuleShapeDiagnosticFields;
export type ModuleMissingInputShapeDiagnostic = PublicApi.ModuleMissingInputShapeDiagnostic;
export type ModuleShapeMismatchDiagnostic = PublicApi.ModuleShapeMismatchDiagnostic;
export type ModuleIrShapeDiagnostic = PublicApi.ModuleIrShapeDiagnostic;
export type ModuleKernelizerDiagnostic = PublicApi.ModuleKernelizerDiagnostic;
export type ModuleSupportDiagnostic = PublicApi.ModuleSupportDiagnostic;
export type ModuleCompileDiagnostic = PublicApi.ModuleCompileDiagnostic;
export type ModuleCompilerSignatures = PublicApi.ModuleCompilerSignatures;
export type ModuleCompleteCompilerSignatures = PublicApi.ModuleCompleteCompilerSignatures;
export type ModuleCompileSupport = PublicModuleCompileSupport;
export type ModuleTraceParameter = PublicApi.ModuleTraceParameter;
export type ModuleTraceOptions = PublicApi.ModuleTraceOptions;
export type ModuleTraceOpBase<Op extends ModuleTraceOpKind> = PublicApi.ModuleTraceOpBase<Op>;
export type ModuleLinearTraceOp = PublicApi.ModuleLinearTraceOp;
export type ModuleEmbeddingTraceOp = PublicApi.ModuleEmbeddingTraceOp;
export type ModuleActivationTraceOp = PublicApi.ModuleActivationTraceOp;
export type ModuleSoftmaxTraceOp = PublicApi.ModuleSoftmaxTraceOp;
export type ModuleReductionTraceOp = PublicApi.ModuleReductionTraceOp;
export type ModuleDropoutTraceOp = PublicApi.ModuleDropoutTraceOp;
export type ModuleIdentityTraceOp = PublicApi.ModuleIdentityTraceOp;
export type ModuleShapeTargetTraceOp = PublicApi.ModuleShapeTargetTraceOp;
export type ModuleFlattenTraceOp = PublicApi.ModuleFlattenTraceOp;
export type ModuleSqueezeTraceOp = PublicApi.ModuleSqueezeTraceOp;
export type ModuleUnsqueezeTraceOp = PublicApi.ModuleUnsqueezeTraceOp;
export type ModuleTransposeTraceOp = PublicApi.ModuleTransposeTraceOp;
export type ModulePermuteTraceOp = PublicApi.ModulePermuteTraceOp;
export type ModuleNarrowTraceOp = PublicApi.ModuleNarrowTraceOp;
export type ModuleSelectTraceOp = PublicApi.ModuleSelectTraceOp;
export type ModuleSliceTraceOp = PublicApi.ModuleSliceTraceOp;
export type ModuleFeatureNormTraceOp = PublicApi.ModuleFeatureNormTraceOp;
export type ModuleUnknownTraceOp = PublicApi.ModuleUnknownTraceOp;
export type ModuleTraceOp = PublicApi.ModuleTraceOp;
export type ModuleTensorProgramIrValueRole = PublicApi.ModuleTensorProgramIrValueRole;
export type ModuleTensorProgramIrValue = PublicApi.ModuleTensorProgramIrValue;
export type ModuleTensorProgramIrLinearAttrs = PublicApi.ModuleTensorProgramIrLinearAttrs;
export type ModuleTensorProgramIrEmbeddingAttrs = PublicApi.ModuleTensorProgramIrEmbeddingAttrs;
export type ModuleTensorProgramIrActivationAttrs = PublicApi.ModuleTensorProgramIrActivationAttrs;
export type ModuleTensorProgramIrDimAttrs = PublicApi.ModuleTensorProgramIrDimAttrs;
export type ModuleTensorProgramIrDropoutAttrs = PublicApi.ModuleTensorProgramIrDropoutAttrs;
export type ModuleTensorProgramIrShapeTargetAttrs = PublicApi.ModuleTensorProgramIrShapeTargetAttrs;
export type ModuleTensorProgramIrFlattenAttrs = PublicApi.ModuleTensorProgramIrFlattenAttrs;
export type ModuleTensorProgramIrSqueezeAttrs = PublicApi.ModuleTensorProgramIrSqueezeAttrs;
export type ModuleTensorProgramIrUnsqueezeAttrs = PublicApi.ModuleTensorProgramIrUnsqueezeAttrs;
export type ModuleTensorProgramIrNarrowAttrs = PublicApi.ModuleTensorProgramIrNarrowAttrs;
export type ModuleTensorProgramIrSelectAttrs = PublicApi.ModuleTensorProgramIrSelectAttrs;
export type ModuleTensorProgramIrSliceAttrs = PublicApi.ModuleTensorProgramIrSliceAttrs;
export type ModuleTensorProgramIrTransposeAttrs = PublicApi.ModuleTensorProgramIrTransposeAttrs;
export type ModuleTensorProgramIrPermuteAttrs = PublicApi.ModuleTensorProgramIrPermuteAttrs;
export type ModuleTensorProgramIrFeatureNormAttrs = PublicApi.ModuleTensorProgramIrFeatureNormAttrs;
export type ModuleTensorProgramIrUnknownAttrs = PublicApi.ModuleTensorProgramIrUnknownAttrs;
export type ModuleTensorProgramIrNoAttrs = PublicApi.ModuleTensorProgramIrNoAttrs;
export type ModuleTensorProgramIrOpBase<Op extends ModuleTensorProgramIrOpKind, Attrs> = PublicApi.ModuleTensorProgramIrOpBase<Op, Attrs>;
export type ModuleTensorProgramIrOp = PublicApi.ModuleTensorProgramIrOp;
export type ModuleTensorProgramIr = PublicModuleTensorProgramIr;
export type ModuleKernelPlanOpBase<Op extends ModuleKernelPlanOpKind, Kernel extends ModuleKernelName> = PublicApi.ModuleKernelPlanOpBase<Op, Kernel>;
export type ModuleKernelPlanFusedValueEdge = PublicApi.ModuleKernelPlanFusedValueEdge;
export type ModuleKernelPlanLinearOp = PublicApi.ModuleKernelPlanLinearOp;
export type ModuleKernelPlanEmbeddingOp = PublicApi.ModuleKernelPlanEmbeddingOp;
export type ModuleKernelPlanActivationOp = PublicApi.ModuleKernelPlanActivationOp;
export type ModuleKernelPlanActivationChainOp = PublicApi.ModuleKernelPlanActivationChainOp;
export type ModuleKernelPlanSoftmaxOp = PublicApi.ModuleKernelPlanSoftmaxOp;
export type ModuleKernelPlanLogSoftmaxOp = PublicApi.ModuleKernelPlanLogSoftmaxOp;
export type ModuleKernelPlanReductionOp = PublicApi.ModuleKernelPlanReductionOp;
export type ModuleKernelPlanReshapeOp = PublicApi.ModuleKernelPlanReshapeOp;
export type ModuleKernelPlanBroadcastOp = PublicApi.ModuleKernelPlanBroadcastOp;
export type ModuleKernelPlanNarrowOp = PublicApi.ModuleKernelPlanNarrowOp;
export type ModuleKernelPlanSelectOp = PublicApi.ModuleKernelPlanSelectOp;
export type ModuleKernelPlanSliceOp = PublicApi.ModuleKernelPlanSliceOp;
export type ModuleKernelPlanTransposeOp = PublicApi.ModuleKernelPlanTransposeOp;
export type ModuleKernelPlanFeatureNormOp = PublicApi.ModuleKernelPlanFeatureNormOp;
export type ModuleKernelPlanShapeChainOp = PublicApi.ModuleKernelPlanShapeChainOp;
export type ModuleKernelPlanElidedOp = PublicApi.ModuleKernelPlanElidedOp;
export type ModuleKernelPlanOp = PublicApi.ModuleKernelPlanOp;
export type ModuleKernelShapeConstraints = PublicApi.ModuleKernelShapeConstraints;
export type ModuleKernelBufferRole = PublicApi.ModuleKernelBufferRole;
export type ModuleKernelBufferLayoutSlot = PublicApi.ModuleKernelBufferLayoutSlot;
export type ModuleKernelBufferLayout = PublicModuleKernelBufferLayout;
export type ModuleKernelMemoryLayoutValue = PublicApi.ModuleKernelMemoryLayoutValue;
export type ModuleKernelMemoryLayout = PublicModuleKernelMemoryLayout;
export type ProgramBufferLayoutSlot = PublicApi.ProgramBufferLayoutSlot;
export type ProgramBufferLayout = PublicProgramBufferLayout;
export type ModuleKernelParameterBinding = PublicApi.ModuleKernelParameterBinding;
export type ModuleKernelParameterTraceOpKind = PublicApi.ModuleKernelParameterTraceOpKind;
export type ModuleKernelParameterLayoutEntry = PublicApi.ModuleKernelParameterLayoutEntry;
export type ModuleKernelParameterLayout = PublicModuleKernelParameterLayout;
export type ModuleKernelPlan = PublicModuleKernelPlan;
export type ProgramCompileEvidence = PublicProgramCompileEvidence;
export type ProgramModuleCompatibilityLengthField = PublicApi.ProgramModuleCompatibilityLengthField;
export type ProgramModuleCompatibilityDiagnosticCode = PublicApi.ProgramModuleCompatibilityDiagnosticCode;
export type ProgramModuleCompatibilitySimpleDiagnosticCode = PublicApi.ProgramModuleCompatibilitySimpleDiagnosticCode;
export type ProgramModuleCompatibilitySimpleDiagnostic = PublicApi.ProgramModuleCompatibilitySimpleDiagnostic;
export type ProgramModuleCompatibilitySignatureKind = PublicApi.ProgramModuleCompatibilitySignatureKind;
export type ProgramModuleCompatibilitySignatureDiagnostic = PublicApi.ProgramModuleCompatibilitySignatureDiagnostic;
export type ProgramModuleCompatibilityNativePathDiagnostic = PublicApi.ProgramModuleCompatibilityNativePathDiagnostic;
export type ProgramModuleCompatibilityModelKindDiagnostic = PublicApi.ProgramModuleCompatibilityModelKindDiagnostic;
export type ProgramModuleCompatibilityLayerCountDiagnostic = PublicApi.ProgramModuleCompatibilityLayerCountDiagnostic;
export type ProgramModuleCompatibilityLengthDiagnostic = PublicApi.ProgramModuleCompatibilityLengthDiagnostic;
export type ProgramModuleCompatibilityShapeDiagnostic = PublicApi.ProgramModuleCompatibilityShapeDiagnostic;
export type ProgramModuleCompatibilityDiagnostic = PublicApi.ProgramModuleCompatibilityDiagnostic;
export type ProgramModuleCompatibility = PublicProgramModuleCompatibility;
export type ModuleProgramTrace = PublicModuleProgramTrace;

export type NnLinearConfig = PublicNnLinearConfig;
export type NnEmbeddingConfig = PublicNnEmbeddingConfig;
export type NnConv2dConfig = PublicNnConv2dConfig;
export type NnNormConfig = PublicNnNormConfig;
export type NnDropoutConfig = PublicNnDropoutConfig;
export type NnBuffer = PublicNnBuffer;
export type OptimizerTarget = PublicOptimizerTarget;
export type SGDConfig = PublicSGDConfig;
export type AdamConfig = PublicAdamConfig;
export type AdagradConfig = PublicAdagradConfig;
export type RMSpropConfig = PublicRMSpropConfig;

export type LoadModelKind =
  | "auto"
  | "llama"
  | "llama-auto"
  | "tiny-llama"
  | "tinyllama"
  | "tiny-llama-2layer"
  | "tiny-llama-2-layer"
  | "tinyllama2layer"
  | "smollm"
  | "smollm135m"
  | "smollm-135m";

export type LoadModelOptions = PublicLoadModelOptions;

export type LoadModelSource = string | Uint8Array | ArrayBuffer;

export type LlamaOutputBinding = NativeBuffer | "native" | true;

export type LlamaBindOptions = PublicLlamaBindOptions;

export type LlamaKvCacheResources = PublicApi.LlamaKvCacheResources;
export type LlamaKvCache = BunLlamaKvCache<NativeBuffer>;

export type ZgmlBackend = keyof typeof backendIds;

export type ExternalResourceOptions = PublicExternalResourceOptions;

export type ExternalResourceAccessName =
  | "read"
  | "readonly"
  | "read-only"
  | "write"
  | "writeonly"
  | "write-only"
  | "readwrite"
  | "read-write"
  | "readWrite";

export type ExternalResourceAccess =
  | ExternalResourceAccessName
  | readonly ExternalResourceAccessName[]
  | { read?: boolean; write?: boolean };

export type BufferInspection = PublicBufferInspection;
export type SessionInspection = PublicSessionInspection;

export type CompileOptions = PublicCompileOptions;
export type EmbeddingCompileOptions = CompileOptions & { inputShape: readonly [number] };

export type TokenIds = PublicTokenIds;

export type TokenWindowOptions = PublicTokenWindowOptions;

export type TokenArgmaxResult = PublicTokenArgmaxResult;

export type TokenSampleOptions = PublicTokenSampleOptions;
export type NormalizedTokenSampleOptions = CoreNormalizedTokenSampleOptions;

export type ExecuteTokensOptions = PublicExecuteTokensOptions;

export type LlamaLogitsTensorOptions = PublicLlamaLogitsTensorOptions;

export type LlamaTokenWindowTensorOptions = PublicLlamaTokenWindowTensorOptions;

export type LlamaExecuteParams = PublicLlamaExecuteParams;

export type LlamaExecuteTensorParams = PublicLlamaExecuteTensorParams;

export type LlamaExecuteIntoParams = PublicLlamaExecuteIntoParams;

export type LlamaStepParams = PublicLlamaStepParams;

export type LlamaSessionStepContract = PublicLlamaSessionStepContract;

export type TokenSampleResult = PublicTokenSampleResult;

export type TokenGenerateArgmaxResult = PublicTokenGenerateArgmaxResult;

export type TokenGenerateSampleResult = PublicTokenGenerateSampleResult;

export type ProgramRequirements = PublicProgramRequirements;
export type ProgramBufferSizing = PublicProgramBufferSizing;
export type SessionBufferSizing = PublicSessionBufferSizing;

export type RuntimeInfo = PublicRuntimeInfo;
export type RuntimeFeatures = PublicRuntimeFeatures;

export type ModelInspection = PublicModelInspection;
export type ProgramModelCompatibility = PublicProgramModelCompatibility;
export type ProgramInspection = PublicProgramInspection;
export type ProgramExecutionMode = PublicProgramExecutionMode;
export type ProgramExecutionUnavailableDiagnostic = PublicProgramExecutionUnavailableDiagnostic;
export type ProgramDispatchPlanDiagnostic = PublicProgramDispatchPlanDiagnostic;
export type ProgramRuntimeDiagnostic = PublicProgramRuntimeDiagnostic;

export type ProgramExecutionCapabilities = PublicProgramExecutionCapabilities;
export type ProgramExecutionPlan = PublicProgramExecutionPlan;

export type LlamaProgramInspection = PublicLlamaProgramInspection;
export type LlamaKvCacheRequirements = PublicLlamaKvCacheRequirements;

export type LlamaKvCacheSlot = PublicLlamaKvCacheSlot;

export type LlamaKvCacheLayoutSlot = PublicLlamaKvCacheLayoutSlot;

export type LlamaKvCacheLayout = PublicLlamaKvCacheLayout;

export type LlamaKvCacheCreateOptions = PublicApi.LlamaKvCacheCreateOptions;

export type ProgramOutputBufferSlot = PublicProgramOutputBufferSlot;

export type ProgramOutputBufferCreateOptions = PublicApi.ProgramOutputBufferCreateOptions;

export type ProgramBufferKind = PublicProgramBufferKind;
export type ProgramDeviceBufferKind = PublicProgramDeviceBufferKind;

export type ProgramBufferSlot = PublicProgramBufferSlot;

export type ProgramBufferCreateOptions = PublicApi.ProgramBufferCreateOptions;

export type WebGpuInteropSymbols = Readonly<{
  deviceHandle: typeof webgpuDeviceHandle;
  bufferHandle: typeof webgpuBufferHandle;
  byteLength: typeof webgpuByteLength;
  byteOffset: typeof webgpuByteOffset;
  placement: typeof webgpuPlacement;
  importSource: typeof webgpuImportSource;
}>;

export type WebGpuInteropImportFields = {
  [webgpuPlacement]?: ZgmlBackend;
  [webgpuDeviceHandle]?: number;
  [webgpuBufferHandle]?: number;
  [webgpuByteOffset]?: number;
  [webgpuByteLength]?: number;
};

export type WebGpuInteropImportSource = Record<string | symbol, unknown> & {
  [webgpuImportSource]: () =>
    | ProgramDeviceBufferImportOptions
    | ProgramDeviceBufferImportDescriptor
    | HostGpuBufferImportSource
    | NativeBuffer;
};

export type ProgramDeviceBufferImportOptions = WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  bufferHandle?: number;
  wgpuBufferHandle?: number;
  handle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
} & (
  | { deviceHandle: number; wgpuDeviceHandle?: number }
  | { wgpuDeviceHandle: number; deviceHandle?: number }
  | { [webgpuDeviceHandle]: number; deviceHandle?: number; wgpuDeviceHandle?: number }
) & (
  | { bufferHandle: number; handle?: number; wgpuBufferHandle?: number }
  | { handle: number; bufferHandle?: number; wgpuBufferHandle?: number }
  | { wgpuBufferHandle: number; bufferHandle?: number; handle?: number }
  | { [webgpuBufferHandle]: number; bufferHandle?: number; handle?: number; wgpuBufferHandle?: number }
);

export type HostGpuBufferImportSource = Record<string | symbol, unknown> & WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend | unknown;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  bufferHandle?: number;
  wgpuBufferHandle?: number;
  handle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
  size?: number;
} & (
  | { deviceHandle: number }
  | { wgpuDeviceHandle: number }
  | { [webgpuDeviceHandle]: number }
) & (
  | { bufferHandle: number }
  | { wgpuBufferHandle: number }
  | { handle: number }
  | { [webgpuBufferHandle]: number }
);

export type ProgramDeviceBufferImportDescriptor = WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
} & (
  | { bufferHandle: number; handle?: number; wgpuBufferHandle?: number }
  | { handle: number; bufferHandle?: number; wgpuBufferHandle?: number }
  | { wgpuBufferHandle: number; bufferHandle?: number; handle?: number }
  | { [webgpuBufferHandle]: number; bufferHandle?: number; handle?: number; wgpuBufferHandle?: number }
);

export type ProgramDeviceBufferImportSource =
  | ProgramDeviceBufferImportOptions
  | HostGpuBufferImportSource
  | WebGpuInteropImportSource
  | NativeBuffer;
export type ProgramDeviceImportBufferSource =
  | NativeBuffer
  | HostGpuBufferImportSource
  | ProgramDeviceBufferImportDescriptor
  | WebGpuInteropImportSource;
export type ProgramCreateBufferOptions = PublicApi.ProgramCreateBufferOptions;

export type RuntimeProfile = PublicRuntimeProfile;

const bunFfiBootstrap = createBunFfiBootstrap({
  symbols,
  readPointer,
});
const {
  check,
  handleOut,
  readHandle,
  assertAlive,
  runtimeInfo,
  abiStructSize,
  abiStructSizes,
  loadedRuntimeInfo,
} = bunFfiBootstrap;
export {
  runtimeInfo,
  abiStructSize,
  abiStructSizes,
  loadedRuntimeInfo,
};

const nativeLifecycleOps = createBunNativeLifecycleOps({
  symbols: bunSymbolGroups.nativeLifecycle,
  check,
});

const {
  compileDesc,
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
} = createBunInspectionOps({
  symbols: bunSymbolGroups.inspection,
  check,
  handleOut,
  readHandle,
  normalizeCompileOptions,
  modelHandleForCompatibility: (model) => modelHandleForCompatibility(model),
  modelInspectionFromAbiWords,
  bufferInspectionFromAbiWords,
  sessionInspectionFromAbiWords,
  programInspectionFromAbiWords,
  programExecutionCapabilitiesFromInspection,
  programRequirementsFromAbiWords,
  programModelCompatibilityFromAbiWords,
  runtimeProfileFromAbiWords,
});

const { isNativeBuffer } = createAdapterNativeBufferInstanceSurface({
  getNativeBufferClass: () => NativeBuffer,
  uninitializedMessage: "Bun NativeBuffer instance surface is not initialized",
});

const {
  loadModelPath,
  loadSafetensorsDataHandle,
  createTinyLlamaModelHandle,
  probeModelPath: probeModelPathHandle,
  probeSafetensorsData: probeSafetensorsDataHandle,
  probeSafetensorsHeader: probeSafetensorsHeaderHandle,
  supportedCheckpointModels: supportedCheckpointModelsFromNative,
} = createBunModelSourceOps({
  symbols: bunSymbolGroups.modelSource,
  check,
  handleOut,
  readHandle,
  pointerFor: (value) => ptr(value as unknown as Float32Array),
  pathBytes: (path) => new TextEncoder().encode(path),
  safetensorsDataBytes,
  safetensorsHeaderBytes,
  normalizeLoadModelKind,
  modelLoadKindId,
  modelInspectionFromAbiWords,
  tinyLlamaKind,
});
const modelSourceFacadeSlot = createAdapterDeferredSlot<AdapterModelSourceFacade>("Bun model-source facade");
const bunModelSourceSurface = createAdapterModelSourceSurface({
  getModelSourceFacade: modelSourceFacadeSlot.peek,
  uninitializedMessage: "Bun model-source surface is not initialized",
});
export const probeModel = bunModelSourceSurface.probeModel as (
  source: LoadModelSource,
  options?: LoadModelOptions | LoadModelKind,
) => ModelInspection;
export const loadModel = bunModelSourceSurface.loadModel as (
  source: LoadModelSource,
  options?: LoadModelOptions | LoadModelKind,
) => LoadedLlamaModel;
export const loadSafetensorsData = bunModelSourceSurface.loadSafetensorsData as (
  data: Uint8Array | ArrayBuffer,
  options?: LoadModelOptions | LoadModelKind,
) => LoadedLlamaModel;

const {
  compileModuleProgram,
  attachProgramCompileEvidence,
} = createBunModuleProgramOps<any>({
  symbols: bunSymbolGroups.moduleProgram,
  check,
  handleOut,
  readHandle,
  pointerFor: (value) => ptr(value),
  compileDesc,
  moduleProgramCompileArtifactsFromCompiledSpec,
  createProgram: createAdapterProgramFactory({
    getProgramClass: () => Program,
  }),
});

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
} = createBunLlamaTokenOps({
  symbols: bunSymbolGroups.llamaToken,
  check,
  pointerFor: (value) => ptr(value),
  tokenId,
  isNativeBuffer,
  tokenSelectionResultFromAbiWords,
  tokenGenerateResultFromAbiWords,
});

export type TensorOptions = PublicTensorOptions;
export type TensorSortResult = Readonly<{ values: Tensor; indices: Tensor }>;
type TensorInitOptions = TensorOptions & {
  prev?: Tensor[];
  backward?: (grad: Float32Array | null) => void;
};

export type RandomTensorOptions = PublicRandomTensorOptions;
export type RandomUniformTensorOptions = PublicRandomUniformTensorOptions;

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
  isGradEnabled: projectedIsGradEnabled,
  meanSquaredError: (tensor: Tensor, target: unknown): Tensor => {
    const loss = meanSquaredError(tensor, target as TensorLike);
    if (!(loss instanceof Tensor)) throw new Error("adapter tensor meanSquaredError must return a Tensor loss");
    return loss;
  },
  dtype: (_tensor: Tensor): TensorDType => "f32",
  device: (_tensor: Tensor): TensorDevice => "cpu",
  rowMajorStrides,
  normalizeDim,
  isNativeBuffer,
  nativeBufferFromFloat32: (data: Float32Array): NativeBuffer => NativeBuffer.fromFloat32(data),
});
const {
  sessionUploadPersistent,
  sessionUploadPersistentRange,
  stepSession,
  prepareStepSession,
  stepNoOutput,
} = createBunSessionOps({
  symbols: bunSymbolGroups.session,
  check,
  pointerFor: (value) => ptr(value as unknown as ArrayBufferView),
});
const programBufferFactoriesSlot = createAdapterDeferredSlot<ReturnType<typeof sharedFrontend.createProgramBufferFactoryHelpers<NativeHandle, NativeBuffer, LlamaKvCache>>>("Bun Program buffer factories");
const {
  createProgramOutputBuffer,
  createProgramBuffer,
  createProgramKvCacheBuffer,
  createProgramNamedBuffer,
  createProgramLlamaKvCache,
} = createAdapterProgramBufferFactorySurface<NativeHandle, NativeBuffer, LlamaKvCache>({
  getProgramBufferFactories: programBufferFactoriesSlot.peek,
  uninitializedMessage: "Bun Program buffer factory surface is not initialized",
});
const adapterSessionRuntimeSurface = createAdapterSessionRuntimeSurface({
  sharedFrontend,
  isNativeBuffer,
  f32,
  prepareF32,
  valueShape: (value: unknown): readonly number[] | null => value instanceof Tensor ? value.shape : null,
  sessionTensorHelpers,
  assertSessionAlive: (handle: NativeHandle): void => assertAlive(handle, "session"),
  sessionInspect: (handle: NativeHandle) => inspectSessionHandle(handle) as Record<string, any>,
  sessionUploadPersistent,
  sessionUploadPersistentRange,
  sessionReset: resetSessionHandle,
  sessionRuntimeProfile: (handle: NativeHandle) => sessionRuntimeProfile(handle) as Record<string, any>,
  sessionResetRuntimeProfile: resetSessionRuntimeProfileHandle,
  sessionFree: nativeLifecycleOps.sessionFree,
  nullSessionHandle: 0,
  stepSession,
  prepareStepSession,
  stepNoOutput,
});
const genericSessionFacade = adapterSessionRuntimeSurface.genericSessionFacade;
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
const requireNativeBufferForBind = (value: unknown, label: string): NativeBuffer | null => (
  requireNativeBuffer(value, label)
);
const {
  llamaKvCacheRequirements,
  bindLlamaSessionHandle,
} = createBunLlamaSessionBindOps({
  symbols: bunSymbolGroups.llamaSessionBind,
  check,
  handleOut,
  readHandle,
  requireNativeBuffer: requireNativeBufferForBind,
  isNativeBuffer,
  modelHandleForBind: (model) => modelHandleForBind(model),
  pointerFor: (value) => ptr(value as unknown as ArrayBufferView),
  bindLlamaSessionHandleByPolicy: (programHandle, options, inspection, policy) => (
    bindLlamaSessionHandleByPolicy(programHandle, options, inspection, policy) as NativeHandle
  ),
  llamaKvCacheRequirementsFromAbiWords,
});
const { bindProgram } = createBunProgramBindOps<NativeBuffer, unknown>({
  symbols: bunSymbolGroups.programBind,
  check,
  handleOut,
  readHandle,
  pointerFor: (value) => ptr(value as unknown as ArrayBufferView),
  prepareProgramBind,
  freeOwnedProgramBindBuffers,
  programNativeBufferBindFields,
  createSession: createAdapterSessionFactory({
    getSessionClass: () => Session,
    copyOwnedBuffers: true,
  }),
});
const llamaSessionFacade = createAdapterLlamaSessionFacadeSurface({
  sharedFrontend,
  isNativeBuffer,
  f32,
  createProgramOutputBuffer,
  sessionTensorHelpers,
  executeTokenWindow: executeLlamaTokenWindow,
  stepToken: sessionStepToken,
  advanceToken: sessionAdvanceToken,
  assertSessionAlive: (handle: NativeHandle): void => assertAlive(handle, "session"),
  sessionPosition: llamaSessionPosition,
  sessionInspect: (handle: NativeHandle) => inspectSessionHandle(handle) as Record<string, any>,
  sessionReset: resetSessionHandle,
  sessionRuntimeProfile: (handle: NativeHandle) => sessionRuntimeProfile(handle) as Record<string, any>,
  sessionResetRuntimeProfile: resetSessionRuntimeProfileHandle,
  sessionFree: nativeLifecycleOps.sessionFree,
  nullSessionHandle: 0,
  argmaxLogits: llamaArgmaxLogits,
  sampleLogits: llamaSampleLogits,
  executeArgmaxWindow: executeLlamaArgmaxWindow,
  executeSampleWindow: executeLlamaSampleWindow,
  generateArgmaxWindow: generateLlamaArgmaxWindow,
  generateSampleWindow: generateLlamaSampleWindow,
});
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
} = llamaSessionFacade;

let tensorFacade: ReturnType<typeof sharedFrontend.createTensorFacadeHelpers>;
let tensorNativeSurfaceHelpers: ReturnType<typeof sharedFrontend.createTensorNativeSurfaceHelpers<Tensor>>;
let tensorStaticHelpers: ReturnType<typeof sharedFrontend.createTensorStaticSurfaceFromFacade<Tensor>>;
let tensorInfoSurfaceHelpers: ReturnType<typeof sharedFrontend.createTensorInfoSurfaceHelpers<Tensor>>;

export let tensor: (data: TensorLike, shape?: TensorShape, options?: TensorOptions) => Tensor;
export let asTensor: typeof tensor;
export let as_tensor: typeof tensor;
export let asarray: typeof tensor;
export let fromNumpy: typeof tensor;
export let from_numpy: typeof tensor;
export let parameter: (data: TensorLike, shape?: TensorShape | TensorOptions, options?: TensorOptions) => Tensor;
export let param: typeof parameter;
export let cat: (tensors: readonly Tensor[], dim?: number) => Tensor;
export let concat: (tensors: readonly Tensor[], dim?: number) => Tensor;
export let concatenate: (tensors: readonly Tensor[], dim?: number) => Tensor;
export let stack: (tensors: readonly Tensor[], dim?: number) => Tensor;
export let vstack: (tensors: readonly Tensor[]) => Tensor;
export let hstack: (tensors: readonly Tensor[]) => Tensor;
export let einsum: (equation: string, tensors: readonly Tensor[] | Tensor, ...moreTensors: readonly Tensor[]) => Tensor;
export let full: (shape: TensorShape, value: number, options?: TensorOptions) => Tensor;
export let fullLike: (input: TensorLike, value: number, options?: TensorOptions) => Tensor;
export let full_like: (input: TensorLike, value: number, options?: TensorOptions) => Tensor;
export let empty: (shape: TensorShape, options?: TensorOptions) => Tensor;
export let emptyLike: (input: TensorLike, options?: TensorOptions) => Tensor;
export let empty_like: (input: TensorLike, options?: TensorOptions) => Tensor;
export let zeros: (shape: TensorShape, options?: TensorOptions) => Tensor;
export let zerosLike: (input: TensorLike, options?: TensorOptions) => Tensor;
export let zeros_like: (input: TensorLike, options?: TensorOptions) => Tensor;
export let ones: (shape: TensorShape, options?: TensorOptions) => Tensor;
export let onesLike: (input: TensorLike, options?: TensorOptions) => Tensor;
export let ones_like: (input: TensorLike, options?: TensorOptions) => Tensor;
export let eye: (size: number, options?: TensorOptions) => Tensor;
export let scalar: (value: number, options?: TensorOptions) => Tensor;
export let rand: (shape: TensorShape, options?: RandomUniformTensorOptions) => Tensor;
export let randLike: (input: TensorLike, options?: RandomUniformTensorOptions) => Tensor;
export let rand_like: (input: TensorLike, options?: RandomUniformTensorOptions) => Tensor;
export let randn: (shape: TensorShape, options?: RandomTensorOptions) => Tensor;
export let randnLike: (input: TensorLike, options?: RandomTensorOptions) => Tensor;
export let randn_like: (input: TensorLike, options?: RandomTensorOptions) => Tensor;
export let randInt: (first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions) => Tensor;
export let randint: (first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions) => Tensor;
export let randPerm: (size: number, options?: RandomIntTensorOptions) => Tensor;
export let randperm: (size: number, options?: RandomIntTensorOptions) => Tensor;
export let hasShape: (input: TensorLike, shape: TensorShape) => boolean;
export let requireShape: (input: TensorLike, shape: TensorShape) => Tensor;
export let manualSeed: (seed: number) => number;
export let manual_seed: (seed: number) => number;
export let initialSeed: () => number | null;
export let initial_seed: () => number | null;
export let seededRng: (seed: number) => () => number;
export let linspace: {
  (shape: readonly number[], start: number, end: number, options?: TensorOptions): Tensor;
  (start: number, end: number, steps: number, options?: TensorOptions): Tensor;
};
export let arange: {
  (end: number, options?: TensorOptions): Tensor;
  (start: number, end: number, options?: TensorOptions): Tensor;
  (start: number, end: number, step: number, options?: TensorOptions): Tensor;
};
export let allclose: (actual: TensorLike, expected: TensorLike, options?: AllCloseOptions) => boolean;
export let equal: (actual: TensorLike, expected: TensorLike) => boolean;
export let isclose: (input: TensorLike, other: TensorLike, options?: IsCloseOptions) => Tensor;
export let to: (input: TensorLike, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions) => Tensor;
export let cpu: (input: TensorLike, options?: TensorToOptions) => Tensor;
export let float: (input: TensorLike, options?: TensorToOptions) => Tensor;
export let float32: (input: TensorLike, options?: TensorToOptions) => Tensor;
export let typeAs: (input: TensorLike, other: TensorLike, options?: TensorToOptions) => Tensor;
export let type_as: (input: TensorLike, other: TensorLike, options?: TensorToOptions) => Tensor;
export let clone: (input: TensorLike) => Tensor;
export let detach: (input: TensorLike) => Tensor;
export let reshape: (input: TensorLike, shape: TensorShape) => Tensor;
export let view: (input: TensorLike, shape: TensorShape) => Tensor;
export let broadcastTo: (input: TensorLike, shape: TensorShape) => Tensor;
export let expand: (input: TensorLike, shape: TensorShape) => Tensor;
export let repeat: (input: TensorLike, repeats: TensorShape) => Tensor;
export let tile: (input: TensorLike, repeats: TensorShape) => Tensor;
export let flatten: (input: TensorLike, startDim?: number, endDim?: number) => Tensor;
export let squeeze: (input: TensorLike, dim?: number | null) => Tensor;
export let unsqueeze: (input: TensorLike, dim: number) => Tensor;
export let transpose: (input: TensorLike, dim0?: number, dim1?: number) => Tensor;
export let permute: (input: TensorLike, dims: readonly number[]) => Tensor;
export let flip: (input: TensorLike, dims: readonly number[]) => Tensor;
export let roll: (input: TensorLike, shifts: number | readonly number[], dims?: number | readonly number[] | null) => Tensor;
export let select: (input: TensorLike, dim: number, index: number) => Tensor;
export let narrow: (input: TensorLike, dim: number, start: number, length: number) => Tensor;
export let slice: (input: TensorLike, dim: number, start?: number | null, end?: number | null, step?: number) => Tensor;
export let indexSelect: (input: TensorLike, dim: number, indices: TensorLike) => Tensor;
export let index_select: (input: TensorLike, dim: number, indices: TensorLike) => Tensor;
export let gather: (input: TensorLike, dim: number, index: TensorLike) => Tensor;
export let take: (input: TensorLike, index: TensorLike) => Tensor;
export let argsort: (input: TensorLike, dim?: number, descending?: boolean) => Tensor;
export let sort: (input: TensorLike, dim?: number, descending?: boolean) => TensorSortResult;
export let topk: (input: TensorLike, k: number, dim?: number, largest?: boolean, sorted?: boolean) => TensorSortResult;
export let scatterAdd: (input: TensorLike, dim: number, index: TensorLike, src: TensorLike) => Tensor;
export let scatter_add: (input: TensorLike, dim: number, index: TensorLike, src: TensorLike) => Tensor;
export let split: (input: TensorLike, splitSizeOrSections: number | readonly number[], dim?: number) => readonly Tensor[];
export let chunk: (input: TensorLike, chunks: number, dim?: number) => readonly Tensor[];
export let unbind: (input: TensorLike, dim?: number) => readonly Tensor[];
export let add: (input: TensorLike, other: TensorLike) => Tensor;
export let sub: (input: TensorLike, other: TensorLike) => Tensor;
export let mul: (input: TensorLike, other: TensorLike) => Tensor;
export let div: (input: TensorLike, other: TensorLike) => Tensor;
export let eq: (input: TensorLike, other: TensorLike) => Tensor;
export let ne: (input: TensorLike, other: TensorLike) => Tensor;
export let lt: (input: TensorLike, other: TensorLike) => Tensor;
export let le: (input: TensorLike, other: TensorLike) => Tensor;
export let gt: (input: TensorLike, other: TensorLike) => Tensor;
export let ge: (input: TensorLike, other: TensorLike) => Tensor;
export let pow: (input: TensorLike, exponent: number) => Tensor;
export let neg: (input: TensorLike) => Tensor;
export let negative: (input: TensorLike) => Tensor;
export let exp: (input: TensorLike) => Tensor;
export let expm1: (input: TensorLike) => Tensor;
export let log: (input: TensorLike) => Tensor;
export let log1p: (input: TensorLike) => Tensor;
export let sqr: (input: TensorLike) => Tensor;
export let square: (input: TensorLike) => Tensor;
export let recip: (input: TensorLike) => Tensor;
export let reciprocal: (input: TensorLike) => Tensor;
export let abs: (input: TensorLike) => Tensor;
export let sgn: (input: TensorLike) => Tensor;
export let sign: (input: TensorLike) => Tensor;
export let step: (input: TensorLike) => Tensor;
export let isnan: (input: TensorLike) => Tensor;
export let isinf: (input: TensorLike) => Tensor;
export let isfinite: (input: TensorLike) => Tensor;
export let floor: (input: TensorLike) => Tensor;
export let ceil: (input: TensorLike) => Tensor;
export let round: (input: TensorLike) => Tensor;
export let trunc: (input: TensorLike) => Tensor;
export let sqrt: (input: TensorLike) => Tensor;
export let rsqrt: (input: TensorLike) => Tensor;
export let relu: (input: TensorLike) => Tensor;
export let gelu: (input: TensorLike) => Tensor;
export let silu: (input: TensorLike) => Tensor;
export let sigmoid: (input: TensorLike) => Tensor;
export let tanh: (input: TensorLike) => Tensor;
export let sin: (input: TensorLike) => Tensor;
export let cos: (input: TensorLike) => Tensor;
export let tan: (input: TensorLike) => Tensor;
export let maximum: (input: TensorLike, other: TensorLike) => Tensor;
export let minimum: (input: TensorLike, other: TensorLike) => Tensor;
export let where: (condition: TensorLike, input: TensorLike, other: TensorLike) => Tensor;
export let maskedFill: (input: TensorLike, mask: TensorLike, value: TensorLike) => Tensor;
export let masked_fill: (input: TensorLike, mask: TensorLike, value: TensorLike) => Tensor;
export let sum: (input: TensorLike, dim?: number) => Tensor;
export let prod: (input: TensorLike, dim?: number) => Tensor;
export let cumsum: (input: TensorLike, dim?: number) => Tensor;
export let mean: (input: TensorLike, dim?: number) => Tensor;
export let max: (input: TensorLike, dim?: number) => Tensor;
export let min: (input: TensorLike, dim?: number) => Tensor;
export let any: (input: TensorLike, dim?: number) => Tensor;
export let all: (input: TensorLike, dim?: number) => Tensor;
export let argmax: (input: TensorLike, dim?: number) => Tensor;
export let argmin: (input: TensorLike, dim?: number) => Tensor;
export let variance: (input: TensorLike, dim?: number, correction?: number) => Tensor;
export let std: (input: TensorLike, dim?: number, correction?: number) => Tensor;
export let norm: (input: TensorLike, dim?: number, p?: number) => Tensor;
export let softmax: (input: TensorLike, dim?: number) => Tensor;
export let softmax_dim: (input: TensorLike, dim?: number) => Tensor;
export let softmaxDim: (input: TensorLike, dim?: number) => Tensor;
export let logSoftmax: (input: TensorLike, dim?: number) => Tensor;
export let log_softmax: (input: TensorLike, dim?: number) => Tensor;
export let log_softmax_dim: (input: TensorLike, dim?: number) => Tensor;
export let logSoftmaxDim: (input: TensorLike, dim?: number) => Tensor;
export let logsumexp: (input: TensorLike, dim?: number) => Tensor;
export let logSumExp: (input: TensorLike, dim?: number) => Tensor;
export let clamp: (input: TensorLike, min?: number | null, max?: number | null) => Tensor;
export let clip: (input: TensorLike, min?: number | null, max?: number | null) => Tensor;
export let matmul: (input: TensorLike, other: TensorLike, otherShape?: readonly number[]) => Tensor;
export let mm: (input: TensorLike, other: TensorLike, otherShape?: readonly number[]) => Tensor;
export let dot: (input: TensorLike, other: TensorLike, otherShape?: readonly number[]) => Tensor;
export let trace: (input: TensorLike) => Tensor;
export let diagonal: (input: TensorLike) => Tensor;
export let bmm: (input: TensorLike, other: TensorLike, otherShape?: readonly number[]) => Tensor;

export const isGradEnabled = projectedIsGradEnabled;
export const is_grad_enabled = projectedIsGradEnabledSnake;
export const setGradEnabled = projectedSetGradEnabled;
export const set_grad_enabled = projectedSetGradEnabledSnake;
export const noGrad = projectedNoGrad;
export const no_grad = projectedNoGradSnake;
export const inferenceMode = projectedInferenceMode;
export const inference_mode = projectedInferenceModeSnake;
export const enableGrad = projectedEnableGrad;
export const enable_grad = projectedEnableGradSnake;
export const gradMode = createAdapterGradModeSurface({
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

let tensorPlacementHelpers: ReturnType<typeof sharedFrontend.createTensorPlacementHelpers>;
const tensorFactoryHelpers = tensorHostSurface.tensorFactoryHelpers;
const tensorViewHelpers = tensorHostSurface.tensorViewHelpers;
const tensorViewSurfaceHelpers = tensorHostSurface.tensorViewSurfaceHelpers;
tensorPlacementHelpers = tensorHostSurface.tensorPlacementHelpers;
tensorInfoSurfaceHelpers = tensorHostSurface.tensorInfoSurfaceHelpers;
const tensorJoinHelpers = tensorHostSurface.tensorJoinHelpers;
const tensorIndexHelpers = tensorHostSurface.tensorIndexHelpers;
const tensorIndexSurfaceHelpers = tensorHostSurface.tensorIndexSurfaceHelpers;
tensorFacade = tensorHostSurface.tensorFacade;
tensorNativeSurfaceHelpers = tensorHostSurface.tensorNativeSurfaceHelpers;
tensorStaticHelpers = tensorHostSurface.tensorStaticHelpers;
const tensorRootOps = tensorHostSurface.tensorRootOps;
setAdapterTensorSurfaceHelpers({
  tensorFacade,
  tensorNativeSurfaceHelpers,
  tensorStaticHelpers,
  tensorInfoSurfaceHelpers,
  tensorIndexSurfaceHelpers,
  tensorViewSurfaceHelpers,
  tensorMathSurfaceHelpers,
});
({
  tensor,
  asTensor: tensor,
  as_tensor: tensor,
  asarray: tensor,
  fromNumpy: tensor,
  from_numpy: tensor,
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
} = tensorRootOps);
asTensor = tensor;
as_tensor = tensor;
asarray = tensor;
fromNumpy = tensor;
from_numpy = tensor;
param = parameter;

tensorStaticHelpers = tensorHostSurface.tensorStaticHelpers;

const {
  programCreateOutputBuffer,
  programCreateBuffer,
  programCreateDeviceBuffer,
  programDeviceHandle,
  programImportDeviceBuffer,
} = createBunProgramBufferOps({
  symbols: bunSymbolGroups.programBuffer,
  programBufferKinds,
  webgpuImportSourceKey: webgpuInterop.importSource,
  webgpuInterop,
  check,
  handleOut,
  readHandle,
  readDeviceHandle: (out) => Number(out[0]),
  pointerForHandle: (handle) => handle,
  programBufferKindId,
  backendId,
  normalizeWebGpuImportSource,
  programDeviceBufferImportInfo,
  ...createAdapterProgramBufferNativeBridge<NativeBuffer, ZgmlBackend>({
    getNativeBufferClass: () => NativeBuffer,
  }),
});

export interface ProgramDevice {
  readonly handle: NativeHandle;
  readonly placement: ZgmlBackend;
  createBuffer(kind: ProgramDeviceBufferKind): NativeBuffer;
  createOutputBuffer(): NativeBuffer;
  createWeightsBuffer(): NativeBuffer;
  createBiasBuffer(): NativeBuffer;
  createInputBuffer(): NativeBuffer;
  createKvCache(): LlamaKvCache;
  importBuffer(kind: ProgramDeviceBufferKind, source: ProgramDeviceImportBufferSource): NativeBuffer;
}

const adapterProgramRuntimeSurface = createAdapterProgramRuntimeSurface({
  sharedFrontend,
  assertAlive,
  programDeviceHandle,
  programCreateDeviceBuffer,
  programImportDeviceBuffer,
  isNativeBuffer,
  createProgramDeviceKvCache: (programHandle: NativeHandle, createBuffer: (kind: ProgramDeviceBufferKind) => NativeBuffer): LlamaKvCache => {
    const requirements = llamaKvCacheRequirements(programHandle) as unknown as LlamaKvCacheRequirements;
    return new LlamaKvCache(requirements, {}, (slot) =>
      createBuffer(slot.kind === "k" ? "kv-k" : "kv-v"));
  },
  programRequirements,
  llamaKvCacheRequirements,
  programCreateBuffer,
  programCreateOutputBuffer,
  requireNativeBuffer,
  createProgramBufferFactoryKvCache: (
    requirements,
    options,
    createBuffer: ((slot: LlamaKvCacheLayoutSlot) => NativeBuffer) | null,
  ): LlamaKvCache => new LlamaKvCache(requirements as LlamaKvCacheRequirements, options as LlamaKvCacheCreateOptions, createBuffer ?? undefined),
  programModelCompatibility,
  programExecutionCapabilities,
  programRuntimeProfile,
  programResetRuntimeProfile: nativeLifecycleOps.programResetRuntimeProfile,
  programFree: nativeLifecycleOps.programFree,
  nullProgramHandle: 0,
  inspectModelHandle,
  compileModelProgramHandle,
  modelFree: nativeLifecycleOps.modelFree,
  nullModelHandle: 0,
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
export const ProgramDevice = adapterProgramRuntimeSurface.ProgramDevice as unknown as new(handle: NativeHandle, placement?: ZgmlBackend) => ProgramDevice;
programBufferFactoriesSlot.bind(adapterProgramRuntimeSurface.programBufferFactories as ReturnType<typeof sharedFrontend.createProgramBufferFactoryHelpers<NativeHandle, NativeBuffer, LlamaKvCache>>);
const programFacadePolicy = adapterProgramRuntimeSurface.programFacadePolicy;
const modelFacadePolicy = adapterProgramRuntimeSurface.modelFacadePolicy;
const genericProgramFacade = adapterProgramRuntimeSurface.genericProgramFacade;
const llamaModelFamilyFacade = adapterProgramRuntimeSurface.llamaModelFamilyFacade;
const llamaProgramFacade = adapterProgramRuntimeSurface.llamaProgramFacade;

let nativeBufferFacade: NativeBufferFacade<NativeBuffer>;

export { NativeBuffer };

const nativeBufferSyscalls = createBunNativeBufferSyscalls({
  symbols: bunSymbolGroups.nativeBuffer,
  check,
  handleOut,
  readHandle,
});

nativeBufferFacade = createAdapterNativeBufferFacadeSurface({
  sharedFrontend,
  NativeBuffer,
  nullHandle: 0,
  f32,
  byteView,
  isLiveHandle: (handle: NativeHandle): boolean => handle !== 0,
  createBuffer: nativeBufferSyscalls.createBuffer,
  wrapBytes: nativeBufferSyscalls.wrapBytes,
  wrapExternalResource: nativeBufferSyscalls.wrapExternalResource,
  bufferSize: nativeBufferSyscalls.bufferSize,
  inspectBuffer: inspectBufferHandle,
  writeBuffer: nativeBufferSyscalls.writeBuffer,
  readBuffer: nativeBufferSyscalls.readBuffer,
  freeBuffer: nativeBufferSyscalls.freeBuffer,
  programDeviceHandle,
}) as NativeBufferFacade<NativeBuffer>;
setAdapterNativeBufferFacade(nativeBufferFacade);

const {
  createLlamaKvCacheBuffers,
  freeKvCacheBuffers,
} = createBunLlamaKvCacheOps<NativeBuffer>({
  createNativeBuffer: (byteLength) => NativeBuffer.create(byteLength),
  requireNativeBuffer: requireNativeBufferForBind,
  assertLlamaKvCacheResourceByteLength: assertLlamaKvCacheResourceByteLength as any,
  llamaKvCacheResourceFactory: llamaKvCacheResourceFactory as any,
  llamaKvCacheResourceSlot: llamaKvCacheResourceSlot as any,
  llamaKvCacheLayoutFromRequirements: llamaKvCacheLayoutFromRequirements as any,
});

export const LlamaKvCache: BunLlamaKvCacheConstructor<NativeBuffer> = createBunLlamaKvCacheClass<NativeBuffer>({
  createLlamaKvCacheBuffers: createLlamaKvCacheBuffers as any,
  freeKvCacheBuffers,
});

const genericFamilySurface = createBunGenericFamilySurface({
  symbols: bunSymbolGroups.genericFamily,
  check,
  handleOut,
  readHandle,
  modelDesc: (desc) => bunGenericModelDesc(desc, { tinyLinearKind, tinyMlpKind }),
  modelFacadePolicy,
  genericProgramFacade,
  genericSessionFacade,
  bindProgram,
  packedProgramWeightsLen,
  packedProgramBiasLen,
  programBufferLayoutFromRequirements,
});

export const TinyLinearModel = genericFamilySurface.TinyLinearModel;
export const TinyMlpModel = genericFamilySurface.TinyMlpModel;
export const Program = genericFamilySurface.Program;
export const Session = genericFamilySurface.Session;
export const TinyLinearProgram = genericFamilySurface.TinyLinearProgram;
export const TinyLinearSession = genericFamilySurface.TinyLinearSession;

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
});

const { indexValues } = createAdapterIndexValuesSurface({ Tensor });

export interface ModuleMode {
  readonly training: boolean;
  children(): readonly NnModule[];
  modules(): readonly NnModule[];
  namedChildren(prefix?: string): readonly ModuleTraversalEntry[];
  named_children(prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(name: string): NnModule | null;
  get_submodule(name: string): NnModule | null;
  apply(callback: (module: NnModule, entry: ModuleTraversalEntry) => void): this;
  named_parameters(prefix?: string): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  namedBuffers(prefix?: string): readonly unknown[];
  named_buffers(prefix?: string): readonly unknown[];
  getBuffer(name: string): unknown | null;
  get_buffer(name: string): unknown | null;
  zero_grad(options?: Record<string, unknown>): void;
  requiresGrad_(requiresGrad?: boolean): this;
  train(mode?: boolean): this;
  eval(): this;
  state_dict(prefix?: string): ModuleStateSnapshot;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
}

export interface LinearModule extends ModuleMode {
  readonly kind: "linear";
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array | null;
  forward(inputValues: TensorLike): Tensor;
  parameters(prefix?: string): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(prefix?: string): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export interface EmbeddingModule extends ModuleMode {
  readonly kind: "embedding";
  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly weight: Float32Array;
  forward(indexValuesInput: IndexLike): Tensor;
  parameters(prefix?: string): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(prefix?: string): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options: EmbeddingCompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options: EmbeddingCompileOptions): Program;
}

export interface ActivationModule extends ModuleMode {
  readonly kind: "gelu" | "relu" | "silu" | "sigmoid" | "tanh" | "exp" | "log" | "neg" | "recip" | "abs" | "sgn" | "step" | "sqrt" | "square";
  forward(inputValues: TensorLike): Float32Array | Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export interface SoftmaxModule extends ModuleMode {
  readonly kind: "softmax";
  readonly dim: number;
  forward(inputValues: TensorLike): Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export interface LogSoftmaxModule extends ModuleMode {
  readonly kind: "logSoftmax";
  readonly dim: number;
  forward(inputValues: TensorLike): Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

type SequentialProgramAnalysis = {
  supported: boolean;
  reason: string | null;
  compiled: CompiledSequentialProgramSpec | null;
  support: Omit<ModuleCompileSupport, "supported" | "reason">;
};

type TraceModuleCompilerConstructors = {
  LinearModule: typeof LinearModule;
  ActivationModule: typeof ActivationModule;
  SequentialModule?: typeof SequentialModule;
  EmbeddingModule?: typeof EmbeddingModule;
  SoftmaxModule?: typeof SoftmaxModule;
  LogSoftmaxModule?: typeof LogSoftmaxModule;
  ReductionModule?: typeof ReductionModule;
  DropoutModule?: typeof DropoutModule;
  FeatureNormModule?: typeof FeatureNormModule;
  ShapeModule?: typeof ShapeModule;
};

type TraceModuleCompiler = {
  trace(layers: readonly NnModule[], options?: ModuleTraceOptions): ModuleProgramTrace;
  analyze(layers: readonly NnModule[], options?: CompileOptions): SequentialProgramAnalysis;
  analyzeSingle(layer: SingleModuleRecord, options?: CompileOptions): SequentialProgramAnalysis;
  packParameters(spec: CompiledSequentialProgramSpec): ModuleBindings;
};

const traceModuleCompilerSlot = createAdapterDeferredSlot<TraceModuleCompiler & HostTraceModuleCompiler>("Bun trace Module compiler");
const bunModuleCompilerSurface = createAdapterModuleCompilerSurface({
  getTraceModuleCompiler: traceModuleCompilerSlot.peek,
  uninitializedMessage: "Bun Module compiler surface is not initialized",
});
const analyzeSequentialProgram = bunModuleCompilerSurface.analyzeSequentialProgram as (
  layers: readonly NnModule[],
  options?: CompileOptions,
) => SequentialProgramAnalysis;
const analyzeSingleModuleProgram = bunModuleCompilerSurface.analyzeSingleModuleProgram as (
  layer: SingleModuleRecord,
  options?: CompileOptions,
) => SequentialProgramAnalysis;
const traceSequentialProgram = bunModuleCompilerSurface.traceSequentialProgram as (
  layers: readonly NnModule[],
  options?: ModuleTraceOptions,
) => ModuleProgramTrace;
const packedSequentialProgramParameters = bunModuleCompilerSurface.packedSequentialProgramParameters as (
  spec: CompiledSequentialProgramSpec,
) => ModuleBindings;

function nativeEagerLinearInto(output: Float32Array, input: TensorLike, weights: TensorLike, options?: Record<string, unknown>) {
  return nativeEager.linearInto(output, input, weights, options);
}

function nativeEagerLinearActivationInto(output: Float32Array, input: unknown, weights: unknown, options: Record<string, unknown>) {
  return nativeEager.linearActivationInto(output, input as TensorLike, weights as TensorLike, options);
}

const adapterFrontendModuleSurface = createAdapterFrontendModuleSurface({
  sharedFrontend,
  Tensor,
  f32,
  f32WithLength,
  indexValues,
  addTensorGrad,
  isGradEnabled: projectedIsGradEnabled,
  requirePositiveInteger,
  defaultedF32,
  zerosF32,
  makeParameter,
  parameterView,
  nativeEagerLinearInto,
  nativeEagerLinearActivationInto,
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
  attachProgramCompileEvidence,
  packedSequentialProgramParameters,
  traceSequentialProgram,
});

export const ActivationModule = adapterFrontendModuleSurface.ActivationModule as unknown as new(kind: ActivationModule["kind"]) => ActivationModule;
export const SoftmaxModule = adapterFrontendModuleSurface.SoftmaxModule as new(dim?: number) => SoftmaxModule;
export const LogSoftmaxModule = adapterFrontendModuleSurface.LogSoftmaxModule as new(dim?: number) => LogSoftmaxModule;
export const ReductionModule = adapterFrontendModuleSurface.ReductionModule as new(kind: "sum" | "mean" | "prod" | "max" | "min" | "argmax" | "argmin", dim?: number) => ReductionModule;
export const DropoutModule = adapterFrontendModuleSurface.DropoutModule as unknown as new(p?: number, config?: NnDropoutConfig) => DropoutModule;
export const LinearModule = adapterFrontendModuleSurface.LinearModule as unknown as {
  new(inFeatures: number, outFeatures: number, config?: NnLinearConfig): LinearModule;
};
export const EmbeddingModule = adapterFrontendModuleSurface.EmbeddingModule as unknown as {
  new(numEmbeddings: number, embeddingDim: number, config?: NnEmbeddingConfig): EmbeddingModule;
};
export const Conv2dModule = adapterFrontendModuleSurface.Conv2dModule as unknown as {
  new(inChannels: number, outChannels: number, kernelSize: unknown, config?: Record<string, unknown>): Conv2dModule;
};

export interface Conv2dModule extends ModuleMode {
  readonly kind: "conv2d";
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: readonly [number, number];
  readonly stride: readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly groups: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array | null;
  forward(inputValues: TensorLike): Tensor;
  parameters(prefix?: string): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(prefix?: string): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
}

export interface FeatureNormModule extends ModuleMode {
  readonly kind: "layerNorm" | "rmsNorm" | "batchNorm1d";
  readonly features: number;
  readonly eps: number;
  readonly momentum: number;
  readonly trackRunningStats: boolean;
  readonly weight: Float32Array | null;
  readonly bias: Float32Array | null;
  readonly runningMean: NnBuffer | null;
  readonly runningVar: NnBuffer | null;
  readonly numBatchesTracked: NnBuffer | null;
  forward(inputValues: TensorLike): Float32Array | Tensor;
  parameters(prefix?: string): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(prefix?: string): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export interface ReductionModule extends ModuleMode {
  readonly kind: "sum" | "mean" | "prod" | "max" | "min" | "argmax" | "argmin";
  readonly dim: number;
  forward(inputValues: TensorLike): Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export interface DropoutModule extends ModuleMode {
  readonly kind: "dropout";
  readonly p: number;
  forward(inputValues: TensorLike): Float32Array | Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
}

export const FeatureNormModule = adapterFrontendModuleSurface.FeatureNormModule as unknown as new(features: number, config?: NnNormConfig) => FeatureNormModule;

export interface ShapeModule extends ModuleMode {
  readonly kind: "identity" | "reshape" | "view" | "flatten" | "squeeze" | "unsqueeze" | "transpose" | "permute" | "broadcastTo" | "expand" | "narrow" | "select" | "slice";
  readonly shape: readonly number[];
  readonly dims: readonly number[];
  readonly startDim: number;
  readonly endDim: number;
  readonly squeezeAll: boolean;
  readonly dim: number;
  readonly start: number;
  readonly length: number;
  readonly index: number;
  readonly end: number | null;
  readonly step: number;
  readonly dim0: number;
  readonly dim1: number;
  forward(inputValues: TensorLike): Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  stateDict(): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export const ShapeModule = adapterFrontendModuleSurface.ShapeModule as unknown as new(kind: ShapeModule["kind"], config?: Record<string, unknown>) => ShapeModule;

type CompiledSequentialProgramSpec = {
  kind: "tiny-linear";
  layer: LinearModule;
  nativePath: "tiny-linear";
  modelKind: "tiny-linear";
  layerCount: 1;
  trace?: ModuleProgramTrace | null;
  ir?: ModuleTensorProgramIr | null;
  kernelPlan?: (ModuleKernelPlan & { nativeOps: readonly ModuleProgramOpDesc[] }) | null;
} | {
  kind: "module";
  desc: ModuleProgramDesc;
  entries: readonly unknown[];
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr;
  kernelPlan: ModuleKernelPlan & { nativeOps: readonly ModuleProgramOpDesc[] };
  nativePath: "device-program";
  modelKind: "module";
  layerCount: number;
};
traceModuleCompilerSlot.bind(adapterFrontendModuleSurface.traceModuleCompiler as TraceModuleCompiler & HostTraceModuleCompiler);

export type NnModule = PublicNnModule;
export type BunNnModule = LinearModule | EmbeddingModule | Conv2dModule | ActivationModule | SoftmaxModule | LogSoftmaxModule | ReductionModule | DropoutModule | FeatureNormModule | ShapeModule | SequentialModule;
export type NnCompilableModule =
  | LinearModule
  | Conv2dModule
  | ActivationModule
  | SoftmaxModule
  | LogSoftmaxModule
  | ReductionModule
  | FeatureNormModule
  | ShapeModule
  | SequentialModule;

export interface SequentialModule extends ModuleMode {
  readonly layers: NnModule[];
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  at(index: number): NnModule;
  get(index: number): NnModule;
  __getitem__(index: number): NnModule;
  __setitem__(index: number, module: NnModule): this;
  __delitem__(index: number): this;
  pop(index?: number): NnModule | undefined;
  clear(): this;
  append(module: NnModule): this;
  insert(index: number, module: NnModule): this;
  extend(modules: Iterable<NnModule>): this;
  [Symbol.iterator](): IterableIterator<NnModule>;
  call(inputValues: TensorLike): Float32Array | Tensor;
  __call__(inputValues: TensorLike): Float32Array | Tensor;
  forward(inputValues: TensorLike): Float32Array | Tensor;
  parameters(): NnParameter[];
  namedParameters(): NnParameter[];
  named_parameters(): NnParameter[];
  children(): readonly NnModule[];
  modules(): readonly NnModule[];
  namedChildren(prefix?: string): readonly ModuleTraversalEntry[];
  named_children(prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(name: string): NnModule | null;
  get_submodule(name: string): NnModule | null;
  apply(callback: (module: NnModule, entry: ModuleTraversalEntry) => void): this;
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number): ModuleParameterInfo | null;
  zeroGrad(options?: Record<string, unknown>): void;
  zero_grad(options?: Record<string, unknown>): void;
  train(mode?: boolean): this;
  eval(): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  trace(options?: ModuleTraceOptions): ModuleProgramTrace;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compile(options?: CompileOptions): Program;
}

export const SequentialModule = adapterFrontendModuleSurface.SequentialModule as unknown as {
  new(): SequentialModule;
  new(first: readonly NnModule[] | Readonly<Record<string, NnModule>> | NnModule, ...layers: readonly NnModule[]): SequentialModule;
};

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

export const { nativeEager } = createAdapterNativeEagerSurface({
  f32: (value) => f32(value as TensorLike),
  check,
  linearF32: (args) => bunSymbolGroups.nativeEager.eagerLinearF32(
    args.inputData,
    BigInt(args.inputData.length),
    args.weightData,
    BigInt(args.weightData.length),
    args.biasData,
    BigInt(args.biasData ? args.biasData.length : 0),
    args.output,
    BigInt(args.expectedOutput),
    BigInt(args.batch),
    BigInt(args.inFeatures),
    BigInt(args.outFeatures),
  ),
  linearActivationF32: (args) => bunSymbolGroups.nativeEager.eagerLinearActivationF32(
    args.inputData,
    BigInt(args.inputData.length),
    args.weightData,
    BigInt(args.weightData.length),
    args.biasData,
    BigInt(args.biasData ? args.biasData.length : 0),
    args.output,
    BigInt(args.expectedOutput),
    BigInt(args.batch),
    BigInt(args.inFeatures),
    BigInt(args.outFeatures),
    args.activation,
  ),
});

const publicNamespaces = createAdapterFrontendNamespaces({
  sharedFrontend,
  Tensor,
  f32,
  f32WithLength,
  indexValues,
  addTensorGrad,
  isGradEnabled: projectedIsGradEnabled,
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
  LinearModule,
  EmbeddingModule,
  Conv2dModule,
  AvgPool2dModule: adapterFrontendModuleSurface.AvgPool2dModule,
  MaxPool2dModule: adapterFrontendModuleSurface.MaxPool2dModule,
  SequentialModule,
  ActivationModule,
  SoftmaxModule,
  LogSoftmaxModule,
  ReductionModule,
  DropoutModule,
  ShapeModule: adapterFrontendModuleSurface.ShapeModule,
  FeatureNormModule: adapterFrontendModuleSurface.FeatureNormModule,
  geluScalar,
  siluScalar,
  parameterNames,
  parameterInfos,
  parameterInfo,
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
  tensor: tensor as any,
  stack,
});
const lossTrainHelpers = publicNamespaces.lossTrainHelpers;
const {
  meanSquaredError,
  classTargets,
  crossEntropy,
} = lossTrainHelpers;
export const loss = lossTrainHelpers.loss;
export const train = lossTrainHelpers.train;
export const data = publicNamespaces.data;

export type SgdOptimizer = Optimizer<"sgd">;
export type AdamOptimizer = Optimizer<"adam">;
export type AdamWOptimizer = Optimizer<"adamw">;
export type RMSpropOptimizer = Optimizer<"rmsprop">;
export type AdagradOptimizer = Optimizer<"adagrad">;

const optimizerClasses = publicNamespaces.optimizerClasses;
export const SgdOptimizer = optimizerClasses.SgdOptimizer as unknown as {
  new(paramsOrModule: OptimizerTarget, config?: SGDConfig): SgdOptimizer;
};
export const AdamOptimizer = optimizerClasses.AdamOptimizer as unknown as {
  new(paramsOrModule: OptimizerTarget, config?: AdamConfig, decoupledWeightDecay?: boolean): AdamOptimizer;
};
export const AdamWOptimizer = optimizerClasses.AdamWOptimizer as unknown as {
  new(paramsOrModule: OptimizerTarget, config?: AdamConfig): AdamWOptimizer;
};
export const RMSpropOptimizer = optimizerClasses.RMSpropOptimizer as unknown as {
  new(paramsOrModule: OptimizerTarget, config?: RMSpropConfig): RMSpropOptimizer;
};
export const AdagradOptimizer = optimizerClasses.AdagradOptimizer as unknown as {
  new(paramsOrModule: OptimizerTarget, config?: AdagradConfig): AdagradOptimizer;
};

export const nn = publicNamespaces.nn;
export const F = publicNamespaces.nn.F;
export const compile = createAdapterCompileNamespace({
  traceSequentialProgram,
  analyzeSequentialProgram,
  compileModuleProgram,
});
sharedFrontend.lazy.setLazyTensorProgramCompiler((lazyGraph, compileOptions) => compile.compile(lazyGraph, compileOptions) as any);
export const optim = publicNamespaces.optim;
export const checkpoint: CheckpointNamespace = publicNamespaces.checkpoint as unknown as CheckpointNamespace;
const bunCheckpointFs = requireSharedFrontend("node:fs") as {
  readFileSync(path: string, encoding: "utf8"): string;
  writeFileSync(path: string, text: string, encoding: "utf8"): void;
};
const torchCheckpointIo = createAdapterTorchCheckpointIo(checkpoint, {
  readTextFile: (path) => bunCheckpointFs.readFileSync(path, "utf8"),
  writeTextFile: (path, text) => bunCheckpointFs.writeFileSync(path, text, "utf8"),
});
export const save = torchCheckpointIo.save as typeof PublicApi.save;
export const load = torchCheckpointIo.load as typeof PublicApi.load;
export const simple = Object.freeze({
  Tensor,
  tensor,
  nn,
  F,
  functional: F,
  compile,
  compileForInference: compile.compileForInference,
  lazy: sharedFrontend.lazy,
  optim,
  data,
  loss,
  train,
  checkpoint,
  save,
  load,
  noGrad,
  no_grad,
  inferenceMode,
  inference_mode,
}) as unknown as PublicApi.PublicSimpleNamespace;
export const torch = createAdapterTorchNamespace({
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
  checkpointIo: torchCheckpointIo,
  Program,
  Session,
  NativeBuffer,
  nativeEager,
});
export const zgml = torch;

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
  assertSessionAlive: (handle: unknown) => assertAlive(handle as NativeHandle, "session"),
  llamaSessionFacade,
});

export {
  Tensor,
  TinyLlamaModel,
  TinyLlamaProgram,
  TinyLlamaSession,
  LlamaModel,
  LlamaProgram,
  LlamaSession,
  SmolLM135MModel,
  SmolLM135MProgram,
  SmolLM135MSession,
};

({ modelHandleForBind, modelHandleForCompatibility } = createAdapterModelHandlePolicy({
  sharedFrontend,
  TinyLinearModel,
  TinyMlpModel,
  TinyLlamaModel,
  SmolLM135MModel,
  LlamaModel,
  assertAlive,
  nullHandle: 0,
}));

const { readSafetensorsHeaderFile } = createAdapterSafetensorsFileHeaderHelpers<number>({
  openSync,
  closeSync,
  readSync,
});

function probeModelPath(path: string, kind: LoadModelKind): ModelInspection {
  return probeModelPathHandle(path, kind) as ModelInspection;
}

export function probeSafetensorsData(data: Uint8Array | ArrayBuffer, options: LoadModelOptions | LoadModelKind = {}): ModelInspection {
  return probeSafetensorsDataHandle(data, options) as ModelInspection;
}

export function probeSafetensorsHeader(header: string | Uint8Array, options: LoadModelOptions | LoadModelKind = {}): ModelInspection {
  return probeSafetensorsHeaderHandle(header, options) as ModelInspection;
}

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

export const supportedCheckpointModels: () => readonly ModelInspection[] = (
  () => supportedCheckpointModelsFromNative() as readonly ModelInspection[]
);

type LoadedLlamaModel =
  InstanceType<typeof LlamaModel> |
  InstanceType<typeof TinyLlamaModel> |
  InstanceType<typeof SmolLM135MModel>;

export const TinyLinear = TinyLinearModel;
export const TinyMlp = TinyMlpModel;
export const TinyLlama = TinyLlamaModel;
export const SmolLM135M = SmolLM135MModel;
export const Llama = LlamaModel;
