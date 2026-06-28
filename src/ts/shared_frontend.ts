"use strict";

import {
  createShapeModuleClass as createAuthoredShapeModuleClass,
  type ShapeModuleClassHooks,
} from "./nn/shape_module.js";
import {
  createActivationModuleClass as createAuthoredActivationModuleClass,
  createSoftmaxModuleClass as createAuthoredSoftmaxModuleClass,
  createReductionModuleClass as createAuthoredReductionModuleClass,
  createDropoutModuleClass as createAuthoredDropoutModuleClass,
  type ActivationModuleClassOptions,
  type DropoutModuleClassOptions,
  type DropoutModuleClassExtras,
  type ParameterlessModuleClassHooks,
  type ReductionModuleClassOptions,
  type SoftmaxModuleClassOptions,
  type SoftmaxModuleClassExtras,
} from "./nn/parameterless_modules.js";
import {
  createLinearModuleClass as createAuthoredLinearModuleClass,
  type LinearModuleClassOptions,
} from "./nn/linear_module.js";
import {
  createEmbeddingModuleClass as createAuthoredEmbeddingModuleClass,
  type EmbeddingModuleClassOptions,
} from "./nn/embedding_module.js";
import {
  createConv2dModuleClass as createAuthoredConv2dModuleClass,
  type Conv2dModuleClassOptions,
} from "./nn/conv_module.js";
import {
  createAvgPool2dModuleClass as createAuthoredAvgPool2dModuleClass,
  createMaxPool2dModuleClass as createAuthoredMaxPool2dModuleClass,
  type PoolingModuleClassOptions,
} from "./nn/pooling_module.js";
import {
  createSequentialModuleClass as createAuthoredSequentialModuleClass,
  type SequentialModuleClassHooks,
  type SequentialModuleClassOptions,
} from "./nn/sequential_module.js";
import type {
  SequentialProgramCompileCoreHooksInput,
} from "./nn/sequential_program_compile.js";
import type {
  SingleModuleCompileCoreHooksInput,
} from "./nn/single_module_compile.js";
import {
  createFeatureNormModuleClass as createAuthoredFeatureNormModuleClass,
  type FeatureNormModuleClassOptions,
} from "./nn/feature_norm_module.js";
import {
  createNnNamespace as createAuthoredNnNamespace,
  type NnNamespaceHooks,
  type NnNamespaceOptions,
} from "./nn/namespace.js";
import {
  placeModuleParameterBindings,
} from "./runtime/module_bindings.js";
import * as lazy from "./lazy.js";
import {
  sigmoidScalar,
} from "./core/activation.js";
import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export {
  tinyMlpActivationIds,
  moduleActivationIds,
  moduleOpIds,
  moduleFlags,
  modelKinds,
  backendIds,
  backendNames,
  bufferStorageIds,
  resourceAccessIds,
  programBufferKinds,
  abiStructKinds,
  runtimeFeatureBits,
  requiredRuntimeFeatureMask,
  decodeRuntimeFeatures,
  assertCompatibleRuntimeInfo,
  createRuntimeAbiFacadeHelpers,
  normalizeCompileOptions,
  backendId,
  backendName,
  bufferStorageName,
  resourceAccessFlags,
  accessFlagsToObject,
  programBufferKindId,
  modelKindName,
  abiStructKindId,
} from "./runtime/abi.js";

export { lazy };

export {
  tokenId,
  tokenWindowLength,
  tokenWindow,
  validateMaxTokens,
  tokenOutputArray,
  normalizeTokenSampleOptions,
  tokenSelectionResultFromAbiRecord,
  tokenSelectionResultFromAbiWords,
  tokenGenerateResultFromAbiRecord,
  tokenGenerateResultFromAbiWords,
} from "./core/token.js";

export {
  safetensorsHeaderBytes,
  safetensorsDataBytes,
  isSafetensorsDataSource,
  isSafetensorsPath,
  normalizeLoadModelKind,
  modelLoadKindId,
  createSafetensorsFileHeaderHelpers,
  createSupportedCheckpointCatalogHelpers,
  createModelHandleHelpers,
  createModelSourceFacadeHelpers,
  createModelFacadePolicyHelpers,
  createLlamaModelFamilyFacadeHelpers,
} from "./runtime/model_source.js";

export {
  runtimeProfileFromAbiRecord,
  runtimeProfileFromAbiWords,
  programInspectionFromAbiRecord,
  programInspectionFromAbiWords,
  sessionInspectionFromAbiRecord,
  sessionInspectionFromAbiWords,
  bufferInspectionFromAbiRecord,
  bufferInspectionFromAbiWords,
  modelInspectionFromAbiRecord,
  modelInspectionFromAbiWords,
  programRequirementsFromAbiRecord,
  programRequirementsFromAbiWords,
  llamaKvCacheRequirementsFromAbiRecord,
  llamaKvCacheRequirementsFromAbiWords,
  programModelCompatibilityFromAbiRecord,
  programModelCompatibilityFromAbiWords,
  programRuntimeDiagnostics,
  programExecutionMode,
  programExecutionCapabilitiesFromInspection,
  programCompatibilityAccepted,
  programCapabilityCanExecute,
  programCapabilityCanBindExternalResources,
  programCapabilityHasFullDispatchPlan,
  programExecutionModeFromCapabilities,
} from "./runtime/inspection.js";

export {
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
} from "./runtime/program_buffers.js";

export {
  assert_native_buffer_byte_range_info,
  assert_native_buffer_device_import_info,
  assert_native_buffer_external_resource_info,
  assert_program_device_buffer_import_info,
  assertNativeBufferByteRangeInfo,
  assertNativeBufferDeviceImportInfo,
  assertNativeBufferExternalResourceInfo,
  assertProgramDeviceBufferImportInfo,
  isNativeBufferByteRangeInfo,
  isNativeBufferDeviceImportInfo,
  isNativeBufferExternalResourceInfo,
  isProgramDeviceBufferImportInfo,
  matches_native_buffer_byte_range_signature,
  matches_native_buffer_device_import_signature,
  matches_native_buffer_external_resource_signature,
  matches_program_device_buffer_import_signature,
  matchesNativeBufferByteRangeSignature,
  matchesNativeBufferDeviceImportSignature,
  matchesNativeBufferExternalResourceSignature,
  matchesProgramDeviceBufferImportSignature,
  nativeBufferByteRangeInfo,
  nativeBufferDeviceImportSignature,
  nativeBufferExternalResourceSignature,
  readImportField,
  readImportString,
  normalizeWebGpuImportSource,
  programDeviceBufferImportInfo,
  programDeviceBufferImportSignature,
  nativeBufferDeviceImportOptions,
  externalResourceInfo,
  nativeBufferWrapByteLength,
  validateNativeBufferWrapFloat32,
  nativeBufferCreateByteLength,
  nativeBufferWriteInfo,
  nativeBufferReadFloat32Target,
  nativeBufferReadBytesTarget,
  nativeBufferReadFloat32IntoInfo,
  nativeBufferReadBytesIntoInfo,
  requireNativeBufferByteRangeInfo,
  requireNativeBufferDeviceImportInfo,
  requireNativeBufferExternalResourceInfo,
  requireProgramDeviceBufferImportInfo,
  validateNativeBufferReadFloat32Into,
  validateNativeBufferReadBytesInto,
  createNativeBufferFacadeHelpers,
  createNativeBufferLifetimeHelpers,
} from "./runtime/native_buffer.js";

export {
  moduleProgramDescFromCompiledSpec,
  createModuleProgramDescPackerHelpers,
} from "./runtime/module_program_desc.js";

export {
  validateTensorShape,
  shapeProduct,
  normalizeFactoryShape,
  normalizeViewShape,
  rowMajorStrides,
  inferredOperandShape,
  broadcastPlan,
  normalizeDim,
  normalizeInsertDim,
  dimReductionPlan,
} from "./core/shape.js";

export {
  compileDiagnostic,
} from "./runtime/compile_diagnostics.js";

export {
  freezeSequentialTrace,
  programCompileEvidenceFromCompiledSpec,
  moduleProgramCompileArtifactsFromCompiledSpec,
  programTraceFromCompileEvidence,
  programTensorProgramIrFromCompileEvidence,
  programKernelPlanFromCompileEvidence,
  programShapeConstraintsFromKernelPlan,
  programParameterLayoutFromKernelPlan,
} from "./runtime/trace_compiler.js";

export {
  programCompilerSignaturesFromCompileEvidence,
  isProgramCompileEvidence,
  requireProgramCompileEvidence,
  assertProgramCompileEvidence,
  assert_program_compile_evidence,
  matchesProgramCompileEvidenceSignature,
  matches_program_compile_evidence_signature,
  programCompileEvidenceSignature,
} from "./runtime/compiler_signatures.js";

export {
  flattenSequentialEntries,
  flattenSequentialLayers,
  traceSequentialEntries,
  traceSequentialProgram,
  analyzeSequentialProgram,
  compiledSequentialLinearSpec,
  compiledSequentialProgramSpec,
  compiledSequentialModuleSpec,
  createTraceModuleCompiler,
  sequentialProgramSupportDetails,
} from "./runtime/module_compiler_policy.js";

export {
  packedSequentialModuleParameters,
  packedSequentialProgramParameters,
  placeModuleParameterBindings,
} from "./runtime/module_bindings.js";

export {
  createModuleFacadeHelpers,
} from "./runtime/module_facade.js";

export {
  moduleCompatibilityForProgramEvidence,
} from "./runtime/module_compatibility.js";

export {
  createProgramBufferFactoryHelpers,
} from "./runtime/program_buffer_factory.js";

export {
  createProgramFacadePolicyHelpers,
} from "./runtime/program_facade_policy.js";

export {
  createProgramModuleBindingHelpers,
} from "./runtime/program_module_binding.js";

export {
  createProgramDeviceClass,
} from "./runtime/program_device.js";

export {
  createGenericProgramFacadeHelpers,
  createLlamaProgramFacadeHelpers,
} from "./runtime/program_facade.js";

export {
  programInputShape,
  programOutputShape,
  sessionOutputTensorShape,
  createSessionTensorHelpers,
} from "./runtime/session_tensor.js";

export {
  bindLlamaSessionHandleByPolicy,
  createLlamaSessionFacadeHelpers,
  createGenericSessionFacadeHelpers,
  sessionFacadeCompositionManifest,
} from "./runtime/session_facade_composition.js";

export {
  packedProgramWeightsLen,
  packedProgramBiasLen,
  createTensorPlacementHelpers,
  createProgramBindValidationHelpers,
  createProgramBindPreparationHelpers,
  createProgramNativeBufferBindFieldHelpers,
} from "./runtime/tensor_placement.js";

export {
  geluScalar,
  geluDerivativeScalar,
  siluScalar,
  siluDerivativeScalar,
  sigmoidScalar,
  sigmoidDerivativeScalar,
} from "./core/activation.js";

export {
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
} from "./core/grad_mode.js";

export {
  createTensorDataHelpers,
} from "./core/tensor_data.js";

export {
  createTensorCoreHelpers,
  createTensorInfoSurfaceHelpers,
} from "./core/tensor_core.js";

export {
  createTensorGradStateHelpers,
} from "./core/tensor_grad_state.js";

export {
  createTensorStaticSurface,
  createTensorStaticSurfaceFromFacade,
} from "./core/tensor_static_surface.js";

export {
  createTensorHostSurface,
} from "./core/tensor_host_surface.js";

export {
  createTensorMetadataOps,
} from "./core/tensor_metadata.js";

export {
  createProgramBindingSurface,
} from "./runtime/program_binding_surface.js";

export {
  createTensorFactoryHelpers,
  initialSeed,
  initial_seed,
  manual_seed,
  manualSeed,
  seededRng,
} from "./core/tensor_factory.js";

export {
  createTensorFacadeHelpers,
  createTensorNativeSurfaceHelpers,
  normalizeShapeModuleShape,
} from "./core/tensor_facade.js";

export {
  createTensorViewHelpers,
  createTensorViewSurfaceHelpers,
} from "./core/tensor_view.js";

export {
  createTensorJoinHelpers,
} from "./core/tensor_join.js";

export {
  createTensorIndexHelpers,
  createTensorIndexSurfaceHelpers,
} from "./core/tensor_index.js";

export {
  createTensorMathHelpers,
  createTensorMathSurfaceHelpers,
} from "./core/tensor_math.js";

export {
  createModuleStateHelpers,
} from "./train/module_state.js";

export {
  createCheckpointHelpers,
  createOptimNamespace,
} from "./train/training.js";

export {
  createOptimizerClasses,
} from "./train/optimizer_classes.js";

export {
  assertLRSchedulerStateSnapshot,
  assertOptimizerConfigSnapshot,
  assertOptimizerStateSnapshot,
  assert_lr_scheduler_state_snapshot,
  assert_optimizer_config_snapshot,
  assert_optimizer_state_snapshot,
  isLRSchedulerStateSnapshot,
  isOptimizerConfigSnapshot,
  isOptimizerStateSnapshot,
  lrSchedulerStateSnapshotSignature,
  matchesLRSchedulerStateSnapshotSignature,
  matchesOptimizerConfigSnapshotSignature,
  matchesOptimizerStateSnapshotSignature,
  matches_lr_scheduler_state_snapshot_signature,
  matches_optimizer_config_snapshot_signature,
  matches_optimizer_state_snapshot_signature,
  optimizerConfigSnapshotSignature,
  optimizerStateSnapshotSignature,
  requireLRSchedulerStateSnapshot,
  requireOptimizerConfigSnapshot,
  requireOptimizerStateSnapshot,
} from "./train/optimizer_snapshot.js";

export {
  createLossTrainHelpers,
} from "./train/loss_train_helpers.js";

export {
  createDataNamespace,
} from "./data.js";

export {
  normalizeModuleTrainingMode,
  initializeModuleMode,
  setModuleTraining,
  setChildModuleTraining,
} from "./train/module_mode.js";

type AnyRecord = Record<string, unknown>;

export type SharedShapeModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<ShapeModuleClassHooks, "Tensor" | "f32">
>;

export function createShapeModuleClass(options: SharedShapeModuleClassOptions) {
  const authoredOptions: ShapeModuleClassHooks = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredShapeModuleClass(authoredOptions);
}

export type SharedParameterlessModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<ParameterlessModuleClassHooks, "Tensor" | "f32">
>;

export type SharedParameterlessModuleClassHooks = SharedParameterlessModuleClassOptions;

export type SharedActivationModuleClassOptions = Readonly<Record<string, unknown> & SharedParameterlessModuleClassHooks>;

export function createActivationModuleClass(options: SharedActivationModuleClassOptions) {
  const authoredOptions: ActivationModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredActivationModuleClass(authoredOptions);
}

export type SharedSoftmaxModuleClassOptions = Readonly<Record<string, unknown> & SharedParameterlessModuleClassHooks & SoftmaxModuleClassExtras>;

export function createSoftmaxModuleClass(options: SharedSoftmaxModuleClassOptions) {
  const authoredOptions: SoftmaxModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredSoftmaxModuleClass(authoredOptions);
}

export type SharedReductionModuleClassOptions = Readonly<Record<string, unknown> & SharedParameterlessModuleClassHooks>;

export function createReductionModuleClass(options: SharedReductionModuleClassOptions) {
  const authoredOptions: ReductionModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredReductionModuleClass(authoredOptions);
}

export type SharedDropoutModuleClassOptions = Readonly<Record<string, unknown> & SharedParameterlessModuleClassHooks & DropoutModuleClassExtras>;

export function createDropoutModuleClass(options: SharedDropoutModuleClassOptions) {
  const authoredOptions: DropoutModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredDropoutModuleClass(authoredOptions);
}

export type SharedLinearModuleClassOptions = Readonly<
  SequentialProgramCompileCoreHooksInput & Pick<
    LinearModuleClassOptions,
    "Tensor" | "prepareF32" | "f32WithLength" | "requirePositiveInteger" | "defaultedF32" | "zerosF32" | "makeParameter" | "parameterView" | "nativeEagerLinearInto" | "isGradEnabled" | "TinyLinearModel"
  >
>;

export function createLinearModuleClass(options: SharedLinearModuleClassOptions) {
  const authoredOptions: LinearModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredLinearModuleClass(authoredOptions);
}

export type SharedEmbeddingModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<
    EmbeddingModuleClassOptions,
    "Tensor" | "indexValues" | "addTensorGrad" | "requirePositiveInteger" | "defaultedF32" | "zerosF32" | "makeParameter" | "parameterView" | "traceSequentialProgram" | "freezeSequentialTrace"
  >
>;

export function createEmbeddingModuleClass(options: SharedEmbeddingModuleClassOptions) {
  const authoredOptions: EmbeddingModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredEmbeddingModuleClass(authoredOptions);
}

export type SharedConv2dModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<
    Conv2dModuleClassOptions,
    "Tensor" | "addTensorGrad" | "isGradEnabled" | "requirePositiveInteger" | "defaultedF32" | "zerosF32" | "makeParameter" | "parameterView" | "nativeEagerConv2dInto" | "moduleCompileSupport"
  > & Pick<Conv2dModuleClassOptions, "parameterNames" | "parameterInfos" | "parameterInfo" | "zeroGrad" | "setRequiresGrad" | "stateDict" | "loadStateDict">
>;

export function createConv2dModuleClass(options: SharedConv2dModuleClassOptions) {
  return createAuthoredConv2dModuleClass({ ...options, placeModuleParameterBindings });
}

export type SharedMaxPool2dModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<PoolingModuleClassOptions, "Tensor" | "addTensorGrad" | "isGradEnabled" | "nativeEagerPool2dInto">
>;

export function createMaxPool2dModuleClass(options: SharedMaxPool2dModuleClassOptions) {
  return createAuthoredMaxPool2dModuleClass({ ...options, placeModuleParameterBindings });
}

export function createAvgPool2dModuleClass(options: SharedMaxPool2dModuleClassOptions) {
  return createAuthoredAvgPool2dModuleClass({ ...options, placeModuleParameterBindings });
}

export type SharedSequentialModuleClassOptions = Readonly<
  SequentialProgramCompileCoreHooksInput & Pick<SequentialModuleClassOptions, "Tensor" | "f32" | "prepareF32" | "traceSequentialProgram" | "nativeEagerLinearActivationInto" | "isGradEnabled">
>;

export function createSequentialModuleClass(options: SharedSequentialModuleClassOptions) {
  const authoredOptions: SequentialModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredSequentialModuleClass(authoredOptions);
}

export type SharedFeatureNormModuleClassOptions = Readonly<
  SingleModuleCompileCoreHooksInput & Pick<
    FeatureNormModuleClassOptions,
    "Tensor" | "f32" | "addTensorGrad" | "requirePositiveInteger" | "defaultedF32" | "zerosF32" | "makeParameter" | "parameterView"
  >
>;

export function createFeatureNormModuleClass(options: SharedFeatureNormModuleClassOptions) {
  const authoredOptions: FeatureNormModuleClassOptions = {
    ...options,
    placeModuleParameterBindings,
  };
  return createAuthoredFeatureNormModuleClass(authoredOptions);
}

export type SharedNnNamespaceOptions = Readonly<Record<string, unknown> & Omit<NnNamespaceHooks, "sigmoidScalar">>;

export function createNnNamespace(options: SharedNnNamespaceOptions) {
  return createAuthoredNnNamespace({
    ...options,
    sigmoidScalar,
  });
}

export const sharedFrontendManifest = Object.freeze({
  kind: "zgml-shared-frontend",
  ...tsProductManifestPolicy("src/ts/shared_frontend.ts"),
  genericSessionComposition: "ts",
  llamaSessionComposition: "ts",
  legacyCjsBridge: false,
});
