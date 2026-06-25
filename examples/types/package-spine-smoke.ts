import {
  checkpoint as packageFrontendCheckpoint,
  compile as packageFrontendCompile,
  data as packageFrontendData,
  frontendManifest as packageFrontendManifest,
  lazy as packageFrontendLazy,
  loss as packageFrontendLoss,
  inspection as packageFrontendInspection,
  moduleCompilerPolicy,
  moduleFacade,
  modelSource as packageFrontendModelSource,
  nn as packageFrontendNn,
  nnLinear,
  optim as packageFrontendOptim,
  program as packageFrontendProgram,
  shape,
  shape as packageFrontendShape,
  session as packageFrontendSession,
  simple as packageFrontendSimple,
  sessionFacade as packageSessionFacade,
  sessionValues,
  stepParamsFacade as packageFrontendStepParams,
  tensor as packageFrontendTensor,
  token,
  train as packageFrontendTrain,
} from "zgml/frontend";
import {
  browserManifest as packageBrowserEntryManifest,
  compile as packageBrowserCompile,
  frontendManifest as packageBrowserManifest,
  nn as packageBrowserNn,
  program as packageBrowserProgram,
  session as packageBrowserSession,
  stepParamsFacade as packageBrowserStepParams,
  tensor as packageBrowserTensor,
} from "zgml/browser";
import {
  Tensor as PackageNodeTensor,
  F as packageNodeF,
  gradMode as packageNodeGradMode,
  nn as packageNodeNn,
  tanh as packageNodeTanh,
} from "zgml/node";
import {
  Tensor as PackageBunTensor,
  F as packageBunF,
  gradMode as packageBunGradMode,
  tanh as packageBunTanh,
} from "zgml/bun";
import {
  checkpointManifest as packageCheckpointManifest,
  createCheckpointHelpers as packageCreateCheckpointHelpers,
  type CheckpointCreateOptions as PackageCheckpointCreateOptions,
  type CheckpointInspection as PackageCheckpointInspection,
  type CheckpointModuleState as PackageCheckpointModuleState,
  type CheckpointOptimizerEntry as PackageCheckpointOptimizerEntry,
  type CheckpointNamespace as PackageCheckpointNamespace,
  type CheckpointOptimizerState as PackageCheckpointOptimizerState,
  type CheckpointRestoreOptions as PackageCheckpointRestoreOptions,
  type CheckpointSchedulerState as PackageCheckpointSchedulerState,
  type CheckpointTensorEntry as PackageCheckpointTensorEntry,
  type CheckpointTensorInspection as PackageCheckpointTensorInspection,
  type PublicCheckpointNamespace as PackagePublicCheckpointNamespace,
  type ZgmlCheckpoint as PackageZgmlCheckpoint,
} from "zgml/checkpoint";
import {
  acceptsModuleCompilePlan as packageCompileAcceptsModuleCompilePlan,
  compileManifest as packageCompileManifest,
  isCompileAnalysis as packageIsCompileAnalysis,
  isProgramCompileEvidence as packageCompileIsProgramCompileEvidence,
  matchesCompileAnalysisSignature as packageMatchesCompileAnalysisSignature,
  matchesModuleCompilePlanSignature as packageCompileMatchesModuleCompilePlanSignature,
  matchesProgramCompileEvidenceSignature as packageCompileMatchesProgramCompileEvidenceSignature,
  requireCompileAnalysis as packageRequireCompileAnalysis,
  requireModuleCompilePlan as packageCompileRequireModuleCompilePlan,
  requireProgramCompileEvidence as packageCompileRequireProgramCompileEvidence,
  traceCompilerArtifacts as packageTraceCompilerArtifacts,
  type CompileAnalysis as PackageCompileAnalysis,
  type CompileMode as PackageCompileMode,
  type CompiledInference as PackageCompiledInference,
  type CompileNamespace as PackageCompileNamespace,
  type CompileOptions as PackageCompileOptions,
  type CompileOptionsWithInputShape as PackageCompileOptionsWithInputShape,
  type PublicCompileNamespace as PackagePublicCompileNamespace,
  type ModuleCompileExplanation as PackageModuleCompileExplanation,
  type ModuleBindingPlan as PackageCompileModuleBindingPlan,
  type ModuleBindings as PackageCompileModuleBindings,
  type ModuleCompileSupport as PackageModuleCompileSupport,
  type ModuleCompilerSignatures as PackageModuleCompilerSignatures,
  type ModuleKernelBufferLayout as PackageModuleKernelBufferLayout,
  type ModuleKernelMemoryLayout as PackageModuleKernelMemoryLayout,
  type ModuleKernelParameterLayout as PackageModuleKernelParameterLayout,
  type ModuleKernelPlan as PackageModuleKernelPlan,
  type ModuleKernelShapeConstraints as PackageModuleKernelShapeConstraints,
  type ModuleParameterPlacementOptions as PackageModuleParameterPlacementOptions,
  type ModuleProgramTrace as PackageModuleProgramTrace,
  type ModuleTargetForwardShape as PackageCompileModuleTargetForwardShape,
  type ModuleTensorProgramIr as PackageModuleTensorProgramIr,
  type ProgramCompileEvidence as PackageCompileProgramCompileEvidence,
} from "zgml/compile";
import {
  input as packageLazyInput,
  lazyManifest as packageLazyManifest,
  type LazyTensor as PackageLazyTensor,
  type LazyCompileSupport as PackageLazyCompileSupport,
} from "zgml/lazy";
import {
  compile as packageSimpleCompile,
  nn as packageSimpleNn,
  simpleManifest as packageSimpleManifest,
  tensor as packageSimpleTensor,
  type PublicSimpleNamespace as PackageSimpleNamespace,
} from "zgml/simple";
import {
  createDataNamespace as packageCreateDataNamespace,
  dataManifest as packageDataManifest,
  type CollatedDataLoader as PackageCollatedDataLoader,
  type DataBatches as PackageDataBatches,
  type DataBatchOptions as PackageDataBatchOptions,
  type DataCollateFn as PackageDataCollateFn,
  type DefaultCollateOptions as PackageDefaultCollateOptions,
  type DataLoader as PackageDataLoader,
  type DataNamespace as PackageDataNamespace,
  type DataNamespaceOptions as PackageDataNamespaceOptions,
  type DataSplitOptions as PackageDataSplitOptions,
  type PublicDataNamespace as PackagePublicDataNamespace,
  type TensorDataset as PackageTensorDataset,
  type TensorDatasetBatch as PackageTensorDatasetBatch,
  type TensorDatasetBatchShape as PackageTensorDatasetBatchShape,
  type TensorDatasetMapper as PackageTensorDatasetMapper,
  type TensorDatasetSample as PackageTensorDatasetSample,
  type TensorShapeTail as PackageTensorShapeTail,
} from "zgml/data";
import {
  acceptsRuntimeProfile as packageInspectionAcceptsRuntimeProfile,
  acceptsSessionCallProfile as packageInspectionAcceptsSessionCallProfile,
  acceptsModuleBindingPlan as packageInspectionAcceptsModuleBindingPlan,
  acceptsModuleCompilePlan as packageInspectionAcceptsModuleCompilePlan,
  acceptsProgramBindingPlan as packageInspectionAcceptsProgramBindingPlan,
  acceptsProgramExecutionPlan as packageInspectionAcceptsProgramExecutionPlan,
  acceptsSessionExecutionPlan as packageInspectionAcceptsSessionExecutionPlan,
  inspectionManifest as packageInspectionManifest,
  matchesModuleCompilePlanSignature as packageInspectionMatchesModuleCompilePlanSignature,
  matchesRuntimeProfileSignature as packageInspectionMatchesRuntimeProfileSignature,
  matchesSessionCallProfileSignature as packageInspectionMatchesSessionCallProfileSignature,
  programExecutionModeFromCapabilities as packageProgramExecutionModeFromCapabilities,
  requireModuleBindingPlan as packageInspectionRequireModuleBindingPlan,
  requireModuleCompilePlan as packageInspectionRequireModuleCompilePlan,
  requireProgramBindingPlan as packageInspectionRequireProgramBindingPlan,
  requireProgramExecutionPlan as packageInspectionRequireProgramExecutionPlan,
  requireRuntimeProfile as packageInspectionRequireRuntimeProfile,
  requireSessionCallProfile as packageInspectionRequireSessionCallProfile,
  requireSessionExecutionPlan as packageInspectionRequireSessionExecutionPlan,
} from "zgml/inspection";
import {
  createLossTrainHelpers as packageCreateLossTrainHelpers,
  lossManifest as packageLossManifest,
  type BCELossOptions as PackageBCELossOptions,
  type ClassLossOptions as PackageClassLossOptions,
  type CrossEntropyLoss as PackageCrossEntropyLoss,
  type HuberLossOptions as PackageHuberLossOptions,
  type LossNamespace as PackageLossNamespace,
  type LossReduction as PackageLossReduction,
  type LossReductionOptions as PackageLossReductionOptions,
  type MSELoss as PackageMSELoss,
  type NLLLoss as PackageNLLLoss,
  type PublicLossNamespace as PackagePublicLossNamespace,
  type SmoothL1LossOptions as PackageSmoothL1LossOptions,
} from "zgml/loss";
import {
  modelSourceManifest as packageModelSourceManifest,
  normalizeLoadModelKind as packageNormalizeLoadModelKind,
} from "zgml/model_source";
import {
  createOptimNamespace as packageCreateOptimNamespace,
  isLRSchedulerStateSnapshot as packageIsLRSchedulerStateSnapshot,
  isOptimizerConfigSnapshot as packageIsOptimizerConfigSnapshot,
  isOptimizerStateSnapshot as packageIsOptimizerStateSnapshot,
  matchesLRSchedulerStateSnapshotSignature as packageMatchesLRSchedulerStateSnapshotSignature,
  matchesOptimizerConfigSnapshotSignature as packageMatchesOptimizerConfigSnapshotSignature,
  matchesOptimizerStateSnapshotSignature as packageMatchesOptimizerStateSnapshotSignature,
  optimManifest as packageOptimManifest,
  requireLRSchedulerStateSnapshot as packageRequireLRSchedulerStateSnapshot,
  requireOptimizerConfigSnapshot as packageRequireOptimizerConfigSnapshot,
  requireOptimizerStateSnapshot as packageRequireOptimizerStateSnapshot,
  type AdamConfig as PackageAdamConfig,
  type AdagradConfig as PackageAdagradConfig,
  type CosineAnnealingLRConfig as PackageCosineAnnealingLRConfig,
  type LRScheduler as PackageLRScheduler,
  type LRSchedulerConfig as PackageLRSchedulerConfig,
  type LRSchedulerNamespace as PackageLRSchedulerNamespace,
  type LRSchedulerStateDict as PackageLRSchedulerStateDict,
  type LRSchedulerStateKind as PackageLRSchedulerStateKind,
  type LRSchedulerStateSnapshot as PackageLRSchedulerStateSnapshot,
  type OptimNamespace as PackageOptimNamespace,
  type Optimizer as PackageOptimizer,
  type OptimizerConfigSnapshot as PackageOptimizerConfigSnapshot,
  type OptimizerParamGroupInput as PackageOptimizerParamGroupInput,
  type OptimizerParamGroupSnapshot as PackageOptimizerParamGroupSnapshot,
  type OptimizerParameterSource as PackageOptimizerParameterSource,
  type OptimizerTarget as PackageOptimizerTarget,
  type OptimizerStateDict as PackageOptimizerStateDict,
  type OptimizerStateKind as PackageOptimizerStateKind,
  type OptimizerStateSnapshot as PackageOptimizerStateSnapshot,
  type PublicOptimNamespace as PackagePublicOptimNamespace,
  type ReduceLROnPlateauConfig as PackageReduceLROnPlateauConfig,
  type RMSpropConfig as PackageRMSpropConfig,
  type SGDConfig as PackageSGDConfig,
  type StepLRConfig as PackageStepLRConfig,
} from "zgml/optim";
import {
  createLossTrainHelpers as packageCreateTrainHelpers,
  isTrainEvaluateEvidence as packageIsTrainEvaluateEvidence,
  isTrainEvaluateStepEvidence as packageIsTrainEvaluateStepEvidence,
  isTrainFitEvidence as packageIsTrainFitEvidence,
  isTrainFitStepEvidence as packageIsTrainFitStepEvidence,
  isTrainPredictEvidence as packageIsTrainPredictEvidence,
  isTrainPredictStepEvidence as packageIsTrainPredictStepEvidence,
  isTrainStepEvidence as packageIsTrainStepEvidence,
  matchesTrainEvaluateEvidenceSignature as packageMatchesTrainEvaluateEvidenceSignature,
  matchesTrainEvaluateStepEvidenceSignature as packageMatchesTrainEvaluateStepEvidenceSignature,
  matchesTrainFitEvidenceSignature as packageMatchesTrainFitEvidenceSignature,
  matchesTrainFitStepEvidenceSignature as packageMatchesTrainFitStepEvidenceSignature,
  matchesTrainPredictEvidenceSignature as packageMatchesTrainPredictEvidenceSignature,
  matchesTrainPredictStepEvidenceSignature as packageMatchesTrainPredictStepEvidenceSignature,
  matchesTrainStepEvidenceSignature as packageMatchesTrainStepEvidenceSignature,
  requireTrainEvaluateEvidence as packageRequireTrainEvaluateEvidence,
  requireTrainEvaluateStepEvidence as packageRequireTrainEvaluateStepEvidence,
  requireTrainFitEvidence as packageRequireTrainFitEvidence,
  requireTrainFitStepEvidence as packageRequireTrainFitStepEvidence,
  requireTrainPredictEvidence as packageRequireTrainPredictEvidence,
  requireTrainPredictStepEvidence as packageRequireTrainPredictStepEvidence,
  requireTrainStepEvidence as packageRequireTrainStepEvidence,
  trainManifest as packageTrainManifest,
  type PublicTrainNamespace as PackagePublicTrainNamespace,
  type TrainBatchInputShape as PackageTrainBatchInputShape,
  type TrainBatchTargetShape as PackageTrainBatchTargetShape,
  type TrainClassificationBatch as PackageTrainClassificationBatch,
  type TrainClassificationClassMetric as PackageTrainClassificationClassMetric,
  type TrainClassificationCriterion as PackageTrainClassificationCriterion,
  type TrainClassificationReport as PackageTrainClassificationReport,
  type TrainClassificationTargetShape as PackageTrainClassificationTargetShape,
  type TrainEarlyStoppingOptions as PackageTrainEarlyStoppingOptions,
  type TrainEvaluateContext as PackageTrainEvaluateContext,
  type TrainEvaluateEvidence as PackageTrainEvaluateEvidence,
  type TrainEvaluateOptions as PackageTrainEvaluateOptions,
  type TrainEvaluateStepEvidence as PackageTrainEvaluateStepEvidence,
  type TrainFitContext as PackageTrainFitContext,
  type TrainFitEvidence as PackageTrainFitEvidence,
  type TrainFitOptions as PackageTrainFitOptions,
  type TrainFitStepEvidence as PackageTrainFitStepEvidence,
  type TrainGradientClipOptions as PackageTrainGradientClipOptions,
  type TrainLossStepOptions as PackageTrainLossStepOptions,
  type TrainModuleOutputShape as PackageTrainModuleOutputShape,
  type TrainNamespace as PackageTrainNamespace,
  type TrainPredictContext as PackageTrainPredictContext,
  type TrainPredictEvidence as PackageTrainPredictEvidence,
  type TrainPredictOptions as PackageTrainPredictOptions,
  type TrainPredictStepEvidence as PackageTrainPredictStepEvidence,
  type TrainStepOptions as PackageTrainStepOptions,
  type TrainStepEvidence as PackageTrainStepEvidence,
  type TrainSupervisedBatch as PackageTrainSupervisedBatch,
  type TrainSupervisedCriterion as PackageTrainSupervisedCriterion,
} from "zgml/train";
import {
  acceptsModuleBindingPlan as packageAcceptsModuleBindingPlan,
  acceptsModuleCompilePlan as packageAcceptsModuleCompilePlan,
  createNnNamespace as packageCreateNnNamespace,
  matchesModuleCompilePlanSignature as packageMatchesModuleCompilePlanSignature,
  matchesModuleBindingPlanSignature as packageMatchesModuleBindingPlanSignature,
  nnManifest as packageNnManifest,
  requireModuleBindingPlan as packageRequireModuleBindingPlan,
  requireModuleCompilePlan as packageRequireModuleCompilePlan,
  type KaimingNormalOptions as PackageKaimingNormalOptions,
  type KaimingUniformOptions as PackageKaimingUniformOptions,
  type LinearModule as PackageLinearModule,
  type LoadStateDictOptions as PackageLoadStateDictOptions,
  type ModuleBufferTraversalOptions as PackageModuleBufferTraversalOptions,
  type ModuleDict as PackageModuleDict,
  type ModuleForwardShape as PackageModuleForwardShape,
  type ModuleList as PackageModuleList,
  type ModuleParameterInfo as PackageModuleParameterInfo,
  type ModuleParameterTraversalOptions as PackageModuleParameterTraversalOptions,
  type ModuleStateDict as PackageModuleStateDict,
  type ModuleStateDictOptions as PackageModuleStateDictOptions,
  type ModuleStateEntry as PackageModuleStateEntry,
  type ModuleStateSnapshot as PackageModuleStateSnapshot,
  type ModuleStateValue as PackageModuleStateValue,
  type ModuleTargetForwardShape as PackageModuleTargetForwardShape,
  type ModuleTraversalEntry as PackageModuleTraversalEntry,
  type NnBuffer as PackageNnBuffer,
  type NnBufferOptions as PackageNnBufferOptions,
  type NnDropoutConfig as PackageNnDropoutConfig,
  type NnEmbeddingConfig as PackageNnEmbeddingConfig,
  type NnFunctionalNamespace as PackageNnFunctionalNamespace,
  type NnInitNamespace as PackageNnInitNamespace,
  type NnInitTarget as PackageNnInitTarget,
  type NnLinearConfig as PackageNnLinearConfig,
  type NnLossConstructorName as PackageNnLossConstructorName,
  type NnLossConstructors as PackageNnLossConstructors,
  type NnCompilableModule as PackageNnCompilableModule,
  type NnModule as PackageNnModule,
  type NnNamespace as PackageNnNamespace,
  type NnNormConfig as PackageNnNormConfig,
  type NnParameter as PackageNnParameter,
  type NnParameterOptions as PackageNnParameterOptions,
  type ParameterDict as PackageParameterDict,
  type ParameterList as PackageParameterList,
  type PublicNnNamespace as PackagePublicNnNamespace,
  type SequentialModule as PackageSequentialModule,
  type EmbeddingForwardShape as PackageEmbeddingForwardShape,
  type ShapeModuleForwardShape as PackageShapeModuleForwardShape,
  type ZeroGradOptions as PackageZeroGradOptions,
} from "zgml/nn";
import {
  acceptsProgramBindingPlan as packageAcceptsProgramBindingPlan,
  acceptsProgramExecutionPlan as packageAcceptsProgramExecutionPlan,
  createGenericProgramFacadeHelpers as packageCreateGenericProgramFacadeHelpers,
  isProgramCompileEvidence as packageProgramIsProgramCompileEvidence,
  matchesProgramBindingPlanSignature as packageMatchesProgramBindingPlanSignature,
  matchesProgramCompileEvidenceSignature as packageProgramMatchesProgramCompileEvidenceSignature,
  matchesProgramExecutionPlanSignature as packageMatchesProgramExecutionPlanSignature,
  programManifest as packageProgramManifest,
  requireProgramBindingPlan as packageRequireProgramBindingPlan,
  requireProgramCompileEvidence as packageProgramRequireProgramCompileEvidence,
  requireProgramExecutionPlan as packageRequireProgramExecutionPlan,
  type ExternalResourceAccess as PackageProgramExternalResourceAccess,
  type ExternalResourceAccessName as PackageProgramExternalResourceAccessName,
  type ExternalResourceOptions as PackageProgramExternalResourceOptions,
  type HostGpuBufferImportSource as PackageProgramHostGpuBufferImportSource,
  type Program as PackageProgram,
  type ProgramBindingPlan as PackageProgramBindingPlan,
  type ProgramBindings as PackageProgramBindings,
  type ProgramBufferCreateOptions as PackageProgramBufferCreateOptions,
  type ProgramBufferKind as PackageProgramBufferKind,
  type ProgramBufferSlot as PackageProgramBufferSlot,
  type ProgramCreateBufferOptions as PackageProgramCreateBufferOptions,
  type ProgramDeviceBufferImportDescriptor as PackageProgramDeviceBufferImportDescriptor,
  type ProgramDeviceBufferImportOptions as PackageProgramDeviceBufferImportOptions,
  type ProgramDeviceBufferImportSource as PackageProgramDeviceBufferImportSource,
  type ProgramDeviceImportBufferSource as PackageProgramDeviceImportBufferSource,
  type ProgramInputBinding as PackageProgramInputBinding,
  type ProgramNamespace as PackageProgramNamespace,
  type ProgramOutputBufferCreateOptions as PackageProgramOutputBufferCreateOptions,
  type ProgramOutputBufferSlot as PackageProgramOutputBufferSlot,
  type ProgramOutputBinding as PackageProgramOutputBinding,
  type PublicProgramNamespace as PackagePublicProgramNamespace,
  type WebGpuInteropImportFields as PackageProgramWebGpuInteropImportFields,
  type WebGpuInteropImportSource as PackageProgramWebGpuInteropImportSource,
  type WebGpuInteropSymbols as PackageProgramWebGpuInteropSymbols,
} from "zgml/program";
import {
  acceptsSessionCallProfile as packageAcceptsSessionCallProfile,
  acceptsSessionExecutionPlan as packageAcceptsSessionExecutionPlan,
  createSessionLiveFacadeHelpers as packageCreateSessionLiveFacadeHelpers,
  matchesSessionCallProfileSignature as packageMatchesSessionCallProfileSignature,
  matchesSessionExecutionPlanSignature as packageMatchesSessionExecutionPlanSignature,
  requireSessionCallProfile as packageRequireSessionCallProfile,
  requireSessionExecutionPlan as packageRequireSessionExecutionPlan,
  sessionManifest as packageSessionManifest,
  type ProgramInputBinding as PackageSessionProgramInputBinding,
  type ProgramOutputBinding as PackageSessionProgramOutputBinding,
  type Session as PackageSession,
  type SessionExecuteIntoParams as PackageSessionExecuteIntoParams,
  type SessionExecuteParams as PackageSessionExecuteParams,
  type SessionExecuteTensorParams as PackageSessionExecuteTensorParams,
  type SessionReadOutputTensorOptions as PackageSessionReadOutputTensorOptions,
  type SessionNamespace as PackageSessionNamespace,
  type SessionStepParams as PackageSessionStepParams,
  type SessionStepTensorOptions as PackageSessionStepTensorOptions,
  type PublicSessionNamespace as PackagePublicSessionNamespace,
} from "zgml/session";
import {
  isStepParamsCompatibility as packageIsStepParamsCompatibility,
  requireStepParamsCompatibility as packageRequireStepParamsCompatibility,
  stepParamsCompatibilityResult as packageFriendlyStepParamsCompatibilityResult,
  stepParamsManifest as packageStepParamsManifest,
  type ProgramInputBinding as PackageStepParamsProgramInputBinding,
  type ProgramOutputBinding as PackageStepParamsProgramOutputBinding,
  type SessionStepParams as PackageStepParamsSessionStepParams,
} from "zgml/step_params";
import {
  isNativeBufferByteRangeInfo as packageIsNativeBufferByteRangeInfo,
  isNativeBufferDeviceImportInfo as packageIsNativeBufferDeviceImportInfo,
  isNativeBufferExternalResourceInfo as packageIsNativeBufferExternalResourceInfo,
  isProgramDeviceBufferImportInfo as packageIsProgramDeviceBufferImportInfo,
  matchesNativeBufferDeviceImportSignature as packageMatchesNativeBufferDeviceImportSignature,
  matchesNativeBufferExternalResourceSignature as packageMatchesNativeBufferExternalResourceSignature,
  matchesNativeBufferByteRangeSignature as packageMatchesNativeBufferByteRangeSignature,
  matchesProgramDeviceBufferImportSignature as packageMatchesProgramDeviceBufferImportSignature,
  type NativeBufferByteRangeInfo as PackageNativeBufferByteRangeInfo,
  type NativeBufferDeviceImportInfo as PackageNativeBufferDeviceImportInfo,
  type NativeBufferExternalResourceInfo as PackageNativeBufferExternalResourceInfo,
  type ExternalResourceAccess as PackageNativeExternalResourceAccess,
  type ExternalResourceOptions as PackageNativeExternalResourceOptions,
  type ProgramDeviceBufferImportOptions as PackageNativeProgramDeviceBufferImportOptions,
  type ProgramDeviceBufferImportSource as PackageNativeProgramDeviceBufferImportSource,
  type WebGpuInteropImportSource as PackageNativeWebGpuInteropImportSource,
  type WebGpuInteropSymbols as PackageNativeWebGpuInteropSymbols,
  nativeBufferManifest as packageNativeBufferManifest,
  nativeBufferDeviceImportOptions as packageNativeBufferDeviceImportOptions,
  externalResourceInfo as packageExternalResourceInfo,
  nativeBufferWriteInfo as packageNativeBufferWriteInfo,
  programDeviceBufferImportInfo as packageProgramDeviceBufferImportInfo,
  type NativeBuffer as PackageNativeBuffer,
  type ProgramDeviceBufferImportInfo as PackageProgramDeviceBufferImportInfo,
  requireNativeBufferDeviceImportInfo as packageRequireNativeBufferDeviceImportInfo,
  requireNativeBufferExternalResourceInfo as packageRequireNativeBufferExternalResourceInfo,
  requireNativeBufferByteRangeInfo as packageRequireNativeBufferByteRangeInfo,
  requireProgramDeviceBufferImportInfo as packageRequireProgramDeviceBufferImportInfo,
} from "zgml/native_buffer";
import {
  isProgramDeviceInfo as packageIsProgramDeviceInfo,
  matchesProgramDeviceInfoSignature as packageMatchesProgramDeviceInfoSignature,
  createProgramDeviceClass as packageCreateProgramDeviceClass,
  type ProgramDeviceBufferImportOptions as PackageDeviceProgramDeviceBufferImportOptions,
  type ProgramDeviceImportBufferSource as PackageDeviceProgramDeviceImportBufferSource,
  type ProgramDeviceInfo as PackageProgramDeviceInfo,
  type WebGpuInteropImportSource as PackageDeviceWebGpuInteropImportSource,
  type WebGpuInteropSymbols as PackageDeviceWebGpuInteropSymbols,
  programDeviceInfo as packageProgramDeviceInfo,
  programDeviceManifest as packageProgramDeviceManifest,
  requireProgramDeviceInfo as packageRequireProgramDeviceInfo,
} from "zgml/program_device";
import {
  createTensorFacadeHelpers as packageCreateTensorFacadeHelpers,
  tensorManifest as packageTensorManifest,
  type AllCloseOptions as PackageAllCloseOptions,
  type ArangeRangeShape as PackageArangeRangeShape,
  type ArangeShape as PackageArangeShape,
  type BroadcastShape as PackageBroadcastShape,
  type FlattenShape as PackageFlattenShape,
  type IndexLike as PackageIndexLike,
  type MatmulShape as PackageMatmulShape,
  type NarrowShape as PackageNarrowShape,
  type OneHotShape as PackageOneHotShape,
  type PermuteShape as PackagePermuteShape,
  type RandomIntTensorOptions as PackageRandomIntTensorOptions,
  type RandomTensorOptions as PackageRandomTensorOptions,
  type RandomUniformTensorOptions as PackageRandomUniformTensorOptions,
  type ReductionShape as PackageReductionShape,
  type ReshapeShape as PackageReshapeShape,
  type SelectShape as PackageSelectShape,
  type SliceShape as PackageSliceShape,
  type SqueezeAllShape as PackageSqueezeAllShape,
  type SqueezeShape as PackageSqueezeShape,
  type Tensor as PackageTensor,
  type TensorCatShape as PackageTensorCatShape,
  type TensorData as PackageTensorData,
  type TensorDevice as PackageTensorDevice,
  type TensorDeviceLike as PackageTensorDeviceLike,
  type TensorDType as PackageTensorDType,
  type TensorDTypeLike as PackageTensorDTypeLike,
  type TensorFromNativeBufferOptions as PackageTensorFromNativeBufferOptions,
  type TensorInspection as PackageTensorInspection,
  type TensorJSON as PackageTensorJSON,
  type TensorLike as PackageTensorLike,
  type TensorNativeBufferOptions as PackageTensorNativeBufferOptions,
  type TensorNestedArray as PackageTensorNestedArray,
  type TensorOptions as PackageTensorOptions,
  type TensorShape as PackageTensorShape,
  type TensorShapeOf as PackageTensorShapeOf,
  type TensorShapeTuple as PackageTensorShapeTuple,
  type TensorStackShape as PackageTensorStackShape,
  type TensorToOptions as PackageTensorToOptions,
  type TensorToTarget as PackageTensorToTarget,
  type TransposeShape as PackageTransposeShape,
  type UnsqueezeShape as PackageUnsqueezeShape,
  type WhereShape as PackageWhereShape,
} from "zgml/tensor";
import {
  adapterOwnsFrontendPolicy,
  nativeLibraryExtensionForPlatform,
  nativeLibraryFilename as packageNativeLibraryFilename,
  nativeLibraryMissingMessage,
  resolveNativeLibraryPath,
  type NativeAdapterEvidence as PackageNativeAdapterEvidence,
  type NativeRuntime as PackageNativeRuntime,
  type NativeLibraryPathOptions,
} from "zgml/adapters/native";
import {
  nodeAdapterEvidence as packageNodeAdapterEvidence,
  nodeAdapterManifest as packageNodeAdapterManifest,
  nodeKoffiFallbackPath,
} from "zgml/adapters/node";
import {
  bunAdapterEvidence as packageBunAdapterEvidence,
  bunAdapterManifest as packageBunAdapterManifest,
  resolveNativeLibraryPath as bunResolveNativeLibraryPath,
} from "zgml/adapters/bun";
import {
  shapeScalarCount as packageShapeScalarCount,
} from "zgml/core/shape";
import {
  inspectionManifest as packageRuntimeInspectionManifest,
} from "zgml/runtime/inspection";
import {
  modelSourceManifest as packageRuntimeModelSourceManifest,
} from "zgml/runtime/model_source";
import {
  nativeBufferManifest as packageRuntimeNativeBufferManifest,
} from "zgml/runtime/native_buffer";
import {
  programDeviceManifest as packageRuntimeProgramDeviceManifest,
} from "zgml/runtime/program_device";
import {
  stepParamsCompatibilityResult as packageStepParamsCompatibilityResult,
} from "zgml/runtime/step_params";
import {
  acceptsKernelPlan as packageTopLevelAcceptsKernelPlan,
  assertKernelPlan as packageTopLevelAssertKernelPlan,
  kernelPlanManifest as packageTopLevelKernelPlanManifest,
  kernelPlanSignature as packageTopLevelKernelPlanSignature,
  matchesKernelPlanSignature as packageTopLevelMatchesKernelPlanSignature,
  requireKernelPlan as packageTopLevelRequireKernelPlan,
  type KernelPlan as PackageTopLevelKernelPlan,
} from "zgml/kernel_plan";
import {
  acceptsKernelPlan as packageAcceptsKernelPlan,
  kernelPlanManifest as packageKernelPlanManifest,
  kernelPlanSignature as packageKernelPlanSignature,
  matchesKernelPlanSignature as packageMatchesKernelPlanSignature,
  requireKernelPlan as packageRequireKernelPlan,
} from "zgml/runtime/kernel_plan";
import {
  formatNativeApiContractSignature as packageFormatNativeApiContractSignature,
  nativeApiContractManifest as packageNativeApiContractManifest,
  nativeApiContractSignature as packageNativeApiContractSignature,
  nativeApiContractSignatureParts as packageNativeApiContractSignatureParts,
  nativePackageSpineContractManifest as packageNativePackageSpineContractManifest,
  requiredNativeApiExports as packageRequiredNativeApiExports,
  requiredNativePackageSpineExports as packageRequiredNativePackageSpineExports,
  type NativeApiContractSignatureParts as PackageNativeApiContractSignatureParts,
  type NativePackageSpineContractManifest as PackageNativePackageSpineContractManifest,
} from "zgml/runtime/native_api_contract";
import {
  createLinearModuleClass as packageCreateLinearModuleClass,
} from "zgml/nn/linear_module";
import {
  setModuleTraining as packageSetModuleTraining,
} from "zgml/train/module_mode";
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2)
    ? true
    : false;
type Expect<T extends true> = T;

type PackageFrontendSource = Expect<Equal<typeof packageFrontendManifest.source, "ts">>;
type PackageFrontendProductLanguage = Expect<Equal<typeof packageFrontendManifest.productLanguage, "typescript">>;
type PackageFrontendProductSource = Expect<Equal<typeof packageFrontendManifest.productSource, "src/ts/**">>;
type PackageFrontendProductSourceOfTruth = Expect<Equal<typeof packageFrontendManifest.productSourceOfTruth, "ts-only">>;
type PackageFrontendProductSemanticsOwner = Expect<Equal<typeof packageFrontendManifest.productSemanticsOwner, "src/ts/**">>;
type PackageFrontendFanout = Expect<Equal<typeof packageFrontendManifest.packageFanout, "tsdown">>;
type PackageFrontendNativeRole = Expect<Equal<typeof packageFrontendManifest.nativeRole, "runtime-kernel-abi-substrate">>;
type PackageFrontendNativeAlignment = Expect<Equal<typeof packageFrontendManifest.nativeAlignment, "contract-tested-substrate">>;
type PackageFrontendNativeProductPolicy = Expect<Equal<typeof packageFrontendManifest.nativeProductPolicy, "forbidden">>;
type PackageFrontendSync = Expect<Equal<typeof packageFrontendManifest.frontendSync, "none">>;
type PackageFrontendHandwrittenMirrors = Expect<Equal<typeof packageFrontendManifest.handwrittenFrontendMirrors, false>>;
type PackageFrontendNativeContractBoundary = Expect<Equal<typeof packageFrontendManifest.nativeContractBoundary, "Program/Session/ABI contracts">>;
type PackageFrontendRuntimePath = Expect<
  Equal<typeof packageFrontendManifest.runtimePath, "Program -> Session -> StepParams">
>;
type PackageBrowserSource = Expect<Equal<typeof packageBrowserManifest.source, "ts">>;
type PackageBrowserEntryOwner = Expect<Equal<typeof packageBrowserEntryManifest.policyOwner, "src/ts/browser.ts">>;
type PackageBrowserEntryLoader = Expect<Equal<typeof packageBrowserEntryManifest.nativeLoader, false>>;
type PackageBrowserEntryRole = Expect<Equal<typeof packageBrowserEntryManifest.adapterRole, "browser-safe-frontend">>;
type PackageBrowserRuntimePath = Expect<
  Equal<typeof packageBrowserManifest.runtimePath, typeof packageFrontendManifest.runtimePath>
>;
type PackageBrowserCompilePath = Expect<Equal<typeof packageBrowserCompile.compileManifest.runtimePath, "Trace -> TensorProgramIr -> KernelPlan -> Program">>;
type PackageBrowserNnOwner = Expect<Equal<typeof packageBrowserNn.nnManifest.policyOwner, "src/ts/nn.ts">>;
type PackageBrowserTensorOwner = Expect<Equal<typeof packageBrowserTensor.tensorManifest.policyOwner, "src/ts/tensor.ts">>;
type PackageBrowserProgramPath = Expect<Equal<typeof packageBrowserProgram.programManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageBrowserSessionPath = Expect<Equal<typeof packageBrowserSession.sessionManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageBrowserStepParamsPath = Expect<Equal<typeof packageBrowserStepParams.stepParamsManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageFrontendCheckpointOwner = Expect<Equal<typeof packageFrontendCheckpoint.checkpointManifest.policyOwner, "src/ts/checkpoint.ts">>;
type PackageFrontendCompilePath = Expect<Equal<typeof packageFrontendCompile.compileManifest.runtimePath, "Trace -> TensorProgramIr -> KernelPlan -> Program">>;
type PackageFrontendInspectionPath = Expect<Equal<typeof packageFrontendInspection.inspectionManifest.runtimePath, "Program -> Session -> inspection evidence">>;
type PackageFrontendModelSourcePath = Expect<Equal<typeof packageFrontendModelSource.modelSourceManifest.runtimePath, "ModelSource -> Program -> Session">>;
type PackageFrontendLossOwner = Expect<Equal<typeof packageFrontendLoss.lossManifest.policyOwner, "src/ts/loss.ts">>;
type PackageFrontendDataOwner = Expect<Equal<typeof packageFrontendData.dataManifest.policyOwner, "src/ts/data.ts">>;
type PackageFrontendOptimOwner = Expect<Equal<typeof packageFrontendOptim.optimManifest.policyOwner, "src/ts/optim.ts">>;
type PackageFrontendTrainOwner = Expect<Equal<typeof packageFrontendTrain.trainManifest.policyOwner, "src/ts/train.ts">>;
type PackageFrontendTensorOwner = Expect<Equal<typeof packageFrontendTensor.tensorManifest.policyOwner, "src/ts/tensor.ts">>;
type PackageFrontendNnOwner = Expect<Equal<typeof packageFrontendNn.nnManifest.policyOwner, "src/ts/nn.ts">>;
type PackageFrontendSimpleOwner = Expect<Equal<typeof packageFrontendSimple.simpleManifest.policyOwner, "src/ts/simple.ts">>;
type PackageFrontendTensorSourceOfTruth = Expect<Equal<typeof packageFrontendTensor.tensorManifest.productSourceOfTruth, "ts-only">>;
type PackageFrontendNnNativeProductPolicy = Expect<Equal<typeof packageFrontendNn.nnManifest.nativeProductPolicy, "forbidden">>;
type PackageFrontendLossNoMirrors = Expect<Equal<typeof packageFrontendLoss.lossManifest.handwrittenFrontendMirrors, false>>;
type PackageFrontendProgramPath = Expect<Equal<typeof packageFrontendProgram.programManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageFrontendSessionPath = Expect<Equal<typeof packageFrontendSession.sessionManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageFrontendStepParamsPath = Expect<Equal<typeof packageFrontendStepParams.stepParamsManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageCheckpointOwner = Expect<Equal<typeof packageCheckpointManifest.policyOwner, "src/ts/checkpoint.ts">>;
type PackageCheckpointNamespaceCreate = Expect<Equal<ReturnType<PackageCheckpointNamespace["create"]>, PackageZgmlCheckpoint>>;
type PackagePublicCheckpointNamespaceShape = Expect<Equal<PackagePublicCheckpointNamespace, Readonly<PackageCheckpointNamespace>>>;
type PackageCheckpointNamespaceInspect = Expect<Equal<ReturnType<PackageCheckpointNamespace["inspect"]>, PackageCheckpointInspection>>;
type PackageCheckpointCreateModel = Expect<Equal<PackageCheckpointCreateOptions["model"], PackageOptimizerTarget | undefined>>;
type PackageCheckpointRestoreModel = Expect<Equal<PackageCheckpointRestoreOptions["model"], PackageOptimizerTarget | undefined>>;
type PackageCheckpointCreateAdamScheduler = Expect<Equal<PackageCheckpointCreateOptions<"adam">["scheduler"], PackageLRScheduler<PackageLRSchedulerStateKind, "adam"> | undefined>>;
type PackageCheckpointRestoreAdamScheduler = Expect<Equal<PackageCheckpointRestoreOptions<"adam">["scheduler"], PackageLRScheduler<PackageLRSchedulerStateKind, "adam"> | undefined>>;
type PackageCheckpointModelStateEntry = Expect<Equal<PackageCheckpointModuleState[string], PackageCheckpointTensorEntry>>;
type PackageCheckpointOptimizerEntries = Expect<Equal<PackageCheckpointOptimizerState["entries"][number], PackageCheckpointOptimizerEntry>>;
type PackageCheckpointSchedulerKind = Expect<Equal<PackageCheckpointSchedulerState["kind"], "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau">>;
type PackageCheckpointInspectionSchedulerTMax = Expect<Equal<PackageCheckpointInspection["schedulerTMax"], number | null>>;
type PackageCheckpointInspectionSchedulerEtaMinAlias = Expect<Equal<PackageCheckpointInspection["schedulerEta_min"], number | null>>;
type PackageCheckpointInspectionEntry = Expect<Equal<PackageCheckpointInspection["modelParameters"][number], PackageCheckpointTensorInspection>>;
type PackageCompilePath = Expect<Equal<typeof packageCompileManifest.runtimePath, "Trace -> TensorProgramIr -> KernelPlan -> Program">>;
type PackageCompileNamespaceShape = Expect<PackageCompileNamespace extends { compile: unknown } ? true : false>;
type PackageCompileNamespaceInferenceShape = Expect<PackageCompileNamespace extends { compileForInference: unknown; compile_for_inference: unknown } ? true : false>;
type PackagePublicCompileNamespaceShape = Expect<PackagePublicCompileNamespace extends Readonly<PackageCompileNamespace> ? true : false>;
type PackagePublicCompileNamespaceCallable = Expect<PackagePublicCompileNamespace extends (target: PackageNnCompilableModule, options?: PackageCompileOptions) => PackageProgram ? true : false>;
type PackageCompiledInferenceShape = Expect<PackageCompiledInference<readonly [2], readonly [3]> extends { program: PackageProgram<readonly [2], readonly [3]>; forward(input: unknown): unknown; into(output: Float32Array, input: unknown): Float32Array } ? true : false>;
type PackageCompileModeShape = Expect<Equal<PackageCompileMode, "auto" | "tiny" | "module">>;
type PackageCompileOptionsShape = Expect<Equal<PackageCompileOptions["backend"], "auto" | "cpu" | "metal" | "webgpu" | undefined>>;
type PackageCompileOptionsInputShape = Expect<PackageCompileOptionsWithInputShape<readonly [2]> extends { inputShape: readonly [2] } ? true : false>;
type PackageCompileTargetForwardShape = Expect<Equal<PackageCompileModuleTargetForwardShape<readonly [PackageLinearModule<2, 3>], readonly [2]>, readonly [3]>>;
type PackageCompileModuleBindingsShape = Expect<Equal<PackageCompileModuleBindings<readonly [2], readonly [3]>["inputShape"], readonly [2] | undefined>>;
type PackageCompileModuleBindingPlanShape = Expect<Equal<PackageCompileModuleBindingPlan<readonly [2], readonly [3]>["parameterInfos"][number], PackageModuleParameterInfo>>;
type PackageModuleParameterPlacementOptionsShape = Expect<Equal<PackageModuleParameterPlacementOptions["weights"], PackageProgramCreateBufferOptions | undefined>>;
type PackageModuleCompileSupportShape = Expect<Equal<PackageModuleCompileSupport<readonly [2], readonly [3]>["outputShape"], readonly [3] | undefined>>;
type PackageModuleCompileExplanationShape = Expect<Equal<PackageModuleCompileExplanation<readonly [2], readonly [3]>["outputShape"], readonly [3] | null>>;
type PackageModuleCompilerSignaturesShape = Expect<Equal<PackageModuleCompilerSignatures["kernelPlan"], string | undefined>>;
type PackageModuleProgramTraceShape = Expect<Equal<PackageModuleProgramTrace["ops"][number]["parameters"][number]["shape"], readonly number[]>>;
type PackageModuleTensorProgramIrShape = Expect<Equal<PackageModuleTensorProgramIr["values"][number]["shape"], readonly number[]>>;
type PackageModuleKernelPlanShape = Expect<Equal<PackageModuleKernelPlan["dispatchCount"], number>>;
type PackageModuleKernelBufferLayoutShape = Expect<Equal<PackageModuleKernelBufferLayout["slots"][number]["byteLength"], number>>;
type PackageModuleKernelMemoryLayoutShape = Expect<Equal<PackageModuleKernelMemoryLayout["values"][number]["byteLength"], number>>;
type PackageModuleKernelParameterLayoutShape = Expect<Equal<PackageModuleKernelParameterLayout["parameters"][number]["scalarCount"], number>>;
type PackageModuleKernelShapeConstraintsShape = Expect<Equal<PackageModuleKernelShapeConstraints["inputShape"], readonly number[]>>;
type PackageCompileAnalysisShape = Expect<Equal<PackageCompileAnalysis["kind"], "zgml.compile.analysis">>;
type PackageCompileProgramEvidenceShape = Expect<Equal<PackageCompileProgramCompileEvidence["kernelPlan"]["signature"], string>>;
const packageLazyInputTensor = packageFrontendLazy.input([2] as const);
const packageLazyGraph = packageLazyInputTensor.linear(3).relu().linear(1);
const packageLazySubpathGraph = packageLazyInput([2] as const).linear(3);
const packageLazyModuleGraph = packageFrontendLazy.fromModule(packageNodeNn.sequential([packageNodeNn.linear(2, 3), packageNodeNn.relu()]), { inputShape: [2] as const });
const packageLazyModuleSupport = packageFrontendLazy.moduleCompileSupport(packageNodeNn.linear(2, 1), { inputShape: [2] as const });
type PackageLazyFrontendShape = Expect<Equal<typeof packageLazyGraph.shape, readonly [1]>>;
type PackageLazySubpathShape = Expect<Equal<typeof packageLazySubpathGraph.shape, readonly [3]>>;
type PackageLazyTensorShape = Expect<PackageLazyTensor<readonly [2]> extends { shape: readonly [2] } ? true : false>;
type PackageLazySupportShape = Expect<Equal<PackageLazyCompileSupport["nativePath"], "device-program">>;
type PackageLazyPath = Expect<Equal<typeof packageLazyManifest.runtimePath, "LazyTensor -> Trace -> TensorProgramIr -> KernelPlan -> Program">>;
type PackageSimpleOwner = Expect<Equal<typeof packageSimpleManifest.policyOwner, "src/ts/simple.ts">>;
type PackageSimpleRootRuntime = Expect<Equal<typeof packageSimpleManifest.rootRuntimeValue, "simple">>;
type PackageSimpleCompileHandle = Expect<Equal<typeof packageSimpleManifest.firstContactRuntimeHandle, "zgml.compileInference">>;
type PackageSimpleNamespaceShape = Expect<PackageSimpleNamespace extends { compile: unknown; nn: unknown; tensor: unknown } ? true : false>;
const packageSimpleTensorManifest = packageSimpleTensor.tensorManifest;
const packageSimpleNnManifest = packageSimpleNn.nnManifest;
const packageSimpleCompileManifest = packageSimpleCompile.compileManifest;
void packageLazyGraph;
void packageLazySubpathGraph;
void packageLazyModuleGraph;
void packageLazyModuleSupport;
type PackageInspectionPath = Expect<Equal<typeof packageInspectionManifest.runtimePath, "Program -> Session -> inspection evidence">>;
type PackageModelSourcePath = Expect<Equal<typeof packageModelSourceManifest.runtimePath, "ModelSource -> Program -> Session">>;
type PackageLossOwner = Expect<Equal<typeof packageLossManifest.policyOwner, "src/ts/loss.ts">>;
type PackageLossReductionShape = Expect<Equal<PackageLossReduction, "mean" | "sum">>;
type PackageLossReductionOptionsShape = Expect<Equal<PackageLossReductionOptions["reduction"], PackageLossReduction | undefined>>;
type PackageHuberLossOptionsShape = Expect<Equal<PackageHuberLossOptions["delta"], number | undefined>>;
type PackageSmoothL1LossOptionsShape = Expect<Equal<PackageSmoothL1LossOptions["beta"], number | undefined>>;
type PackageBceLossOptionsShape = Expect<Equal<PackageBCELossOptions["eps"], number | undefined>>;
type PackageClassLossOptionsShape = Expect<Equal<PackageClassLossOptions["classes"], number | undefined>>;
type PackagePublicLossNamespaceShape = Expect<Equal<PackagePublicLossNamespace, Readonly<PackageLossNamespace>>>;
type PackageDataOwner = Expect<Equal<typeof packageDataManifest.policyOwner, "src/ts/data.ts">>;
type PackageTensorDatasetRowShape = Expect<Equal<PackageTensorShapeTail<readonly [2, 2]>, readonly [2]>>;
type PackageTensorDatasetLoaderShape = Expect<Equal<PackageTensorDatasetBatchShape<readonly [2, 2]>, readonly [number, 2]>>;
type PackageDataNamespaceOptionsShape = Expect<Equal<ReturnType<PackageDataNamespaceOptions<PackageTensor>["stack"]>, PackageTensor>>;
type PackageDataBatchOptionsShape = Expect<Equal<PackageDataBatchOptions["batch_size"], number | undefined>>;
type PackageDefaultCollateOptionsShape = Expect<Equal<PackageDefaultCollateOptions["sample_indices"], readonly number[] | undefined>>;
type PackageDataSplitOptionsShape = Expect<Equal<PackageDataSplitOptions["seed"], number | undefined>>;
type PackageTensorDatasetSampleInput = Expect<Equal<ReturnType<PackageTensorDataset<PackageTensor<readonly [2]>, PackageTensor<readonly [1]>, PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>>["sample"]>["input"], PackageTensor<readonly [2]>>>;
type PackageTensorDatasetSampleTarget = Expect<Equal<PackageTensorDatasetSample<PackageTensor<readonly [2]>, PackageTensor<readonly [1]>>["target"], PackageTensor<readonly [1]> | undefined>>;
type PackageTensorDatasetBatchInput = Expect<Equal<ReturnType<PackageTensorDataset<PackageTensor<readonly [2]>, PackageTensor<readonly [1]>, PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>>["batch"]>["input"], PackageTensor<readonly [number, 2]>>>;
type PackageTensorDatasetBatchTarget = Expect<Equal<PackageTensorDatasetBatch<PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>>["target"], PackageTensor<readonly [number, 1]> | undefined>>;
type PackageTensorDatasetMapperShape = Expect<ReturnType<PackageTensorDatasetMapper<PackageTensor<readonly [2]>, PackageTensor<readonly [1]>, PackageTensor<readonly [3]>>> extends { input: PackageTensor<readonly [3]>; target?: PackageTensor } ? true : false>;
type PackageDataLoaderBatchInput = Expect<Equal<PackageDataLoader<PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>> extends Iterable<infer Batch extends { input: unknown }> ? Batch["input"] : never, PackageTensor<readonly [number, 2]>>>;
type PackageDataBatchesBatchInput = Expect<Equal<ReturnType<PackageDataBatches<PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>>["get"]>["input"], PackageTensor<readonly [number, 2]>>>;
type PackageCustomCollateBatch = Readonly<{ kind: "package-custom-collate"; input: PackageTensor<readonly [2]> }>;
type PackageCollatedDataLoaderBatch = Expect<Equal<ReturnType<PackageCollatedDataLoader<PackageCustomCollateBatch>["get"]>, PackageCustomCollateBatch>>;
type PackageDataCollateFnBatch = Expect<Equal<ReturnType<PackageDataCollateFn<PackageTensorDatasetSample<PackageTensor<readonly [2]>>, PackageCustomCollateBatch>>, PackageCustomCollateBatch>>;
type PackageDataNamespaceDefaultCollateShape = Expect<Equal<ReturnType<PackageDataNamespace["default_collate"]>, PackageTensorDatasetBatch>>;
type PackageDataNamespaceDataLoaderShape = Expect<Equal<ReturnType<PackageDataNamespace["dataLoader"]>, PackageDataLoader>>;
type PackagePublicDataNamespaceShape = Expect<Equal<PackagePublicDataNamespace, Readonly<PackageDataNamespace>>>;
type PackageSgdConfigShape = Expect<Equal<PackageSGDConfig["momentum"], number | undefined>>;
type PackageAdamConfigShape = Expect<Equal<PackageAdamConfig["beta1"], number | undefined>>;
type PackageRMSpropConfigShape = Expect<Equal<PackageRMSpropConfig["alpha"], number | undefined>>;
type PackageAdagradConfigShape = Expect<Equal<PackageAdagradConfig["lrDecay"], number | undefined>>;
type PackageOptimizerParamGroupInputShape = Expect<Equal<PackageOptimizerParamGroupInput["lr"], number | undefined>>;
type PackageOptimizerParamGroupSnapshotShape = Expect<Equal<PackageOptimizerParamGroupSnapshot["paramCount"], number>>;
type PackageOptimizerParameterSourceShape = Expect<PackageTensor extends PackageOptimizerParameterSource ? true : false>;
type PackagePublicOptimNamespaceShape = Expect<Equal<PackagePublicOptimNamespace, Readonly<PackageOptimNamespace>>>;
type PackageOptimNamespaceSgdShape = Expect<Equal<ReturnType<PackageOptimNamespace["sgd"]>["kind"], "sgd">>;
type PackageOptimNamespaceAdamWShape = Expect<Equal<ReturnType<PackageOptimNamespace["adamW"]>["kind"], "adamw">>;
type PackageOptimNamespaceRMSpropShape = Expect<Equal<ReturnType<PackageOptimNamespace["rmsprop"]>["kind"], "rmsprop">>;
type PackageOptimNamespaceAdagradShape = Expect<Equal<ReturnType<PackageOptimNamespace["adagrad"]>["kind"], "adagrad">>;
type PackageOptimizerStateKindShape = Expect<Equal<PackageOptimizerStateKind, "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad">>;
type PackageLrSchedulerStateKindShape = Expect<Equal<PackageLRSchedulerStateKind, "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau">>;
type PackageLrSchedulerConfigShape = Expect<Equal<PackageLRSchedulerConfig["gamma"], number | undefined>>;
type PackageStepLrConfigShape = Expect<Equal<PackageStepLRConfig["stepSize"], number | undefined>>;
type PackageCosineAnnealingLrConfigShape = Expect<Equal<PackageCosineAnnealingLRConfig["tMax"], number | undefined>>;
type PackageCosineAnnealingLrConfigEtaShape = Expect<Equal<PackageCosineAnnealingLRConfig["eta_min"], number | undefined>>;
type PackageReduceLrOnPlateauConfigShape = Expect<Equal<PackageReduceLROnPlateauConfig["mode"], "min" | "max" | undefined>>;
type PackageLrSchedulerNamespaceStepShape = Expect<Equal<ReturnType<ReturnType<PackageLRSchedulerNamespace["stepLR"]>["stateDict"]>["kind"], "step-lr">>;
type PackageLrSchedulerNamespaceCosineShape = Expect<Equal<ReturnType<ReturnType<PackageLRSchedulerNamespace["cosineAnnealingLR"]>["stateDict"]>["kind"], "cosine-annealing-lr">>;
type PackageLrSchedulerNamespacePlateauShape = Expect<Equal<ReturnType<ReturnType<PackageLRSchedulerNamespace["reduceLROnPlateau"]>["stateDict"]>["kind"], "reduce-lr-on-plateau">>;
type PackageTrainNamespaceShape = Expect<Equal<PackagePublicTrainNamespace, Readonly<PackageTrainNamespace>>>;
type PackageTrainStepOptionsLossShape = Expect<Equal<PackageTrainStepOptions["loss"], PackageTensor | undefined>>;
type PackageTrainLossStepOptionsGradientShape = Expect<Equal<PackageTrainLossStepOptions["gradient"], PackageTensorLike | undefined>>;
type PackageTrainGradientClipSnakeShape = Expect<Equal<PackageTrainGradientClipOptions["clip_grad_value"], number | undefined>>;
type PackageTrainEarlyStoppingSnakeShape = Expect<Equal<PackageTrainEarlyStoppingOptions["min_delta"], number | undefined>>;
type PackageTrainFitOptionsEarlyStoppingShape = Expect<Equal<PackageTrainFitOptions["earlyStopping"], boolean | PackageTrainEarlyStoppingOptions | undefined>>;
type PackageTrainFitContextShape = Expect<Equal<PackageTrainFitContext["sample_indices"], readonly number[] | null>>;
type PackageTrainFitEvidenceStopReasonShape = Expect<Equal<PackageTrainFitEvidence["stop_reason"], "max-steps" | "early-stopping" | null>>;
type PackageTrainFitEvidenceBestLossShape = Expect<Equal<PackageTrainFitEvidence["best_loss"], number | null>>;
type PackageTrainFitOptionsCallbackShape = Expect<Equal<Parameters<NonNullable<PackageTrainFitOptions<"adam">["onStep"]>>[0]["stepEvidence"], PackageTrainStepEvidence<"adam"> | null>>;
type PackageTypedTrainBatch = PackageTensorDatasetBatch<PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number, 1]>>;
type PackageTrainBatchInputShapeProof = Expect<Equal<PackageTrainBatchInputShape<PackageTypedTrainBatch>, readonly [number, 2]>>;
type PackageTrainBatchTargetShapeProof = Expect<Equal<PackageTrainBatchTargetShape<PackageTypedTrainBatch>, readonly [number, 1]>>;
type PackageTrainModuleOutputShapeProof = Expect<Equal<PackageTrainModuleOutputShape<PackageLinearModule<2, 1>, PackageTypedTrainBatch>, readonly [number, 1]>>;
type PackageTrainSupervisedBatchProof = Expect<Equal<PackageTrainSupervisedBatch<PackageLinearModule<2, 1>, PackageTypedTrainBatch>, PackageTypedTrainBatch>>;
type PackageTrainSupervisedBatchMismatchProof = Expect<Equal<PackageTrainSupervisedBatch<PackageLinearModule<2, 3>, PackageTypedTrainBatch>, never>>;
type PackageTrainSupervisedCriterionProof = Expect<Parameters<PackageTrainSupervisedCriterion<PackageLinearModule<2, 1>, PackageTypedTrainBatch>["forward"]>[0] extends PackageTensor<readonly [number, 1]> ? true : false>;
type PackageTypedClassBatch = PackageTensorDatasetBatch<PackageTensor<readonly [number, 2]>, PackageTensor<readonly [number]>>;
type PackageTrainClassificationTargetShapeProof = Expect<Equal<PackageTrainClassificationTargetShape<readonly [number, 3]>, readonly [number]>>;
type PackageTrainClassificationBatchProof = Expect<Equal<PackageTrainClassificationBatch<PackageLinearModule<2, 3>, PackageTypedClassBatch>, PackageTypedClassBatch>>;
type PackageTrainClassificationBatchMismatchProof = Expect<Equal<PackageTrainClassificationBatch<PackageLinearModule<2, 3>, PackageTypedTrainBatch>, never>>;
type PackageTrainClassificationCriterionProof = Expect<Parameters<PackageTrainClassificationCriterion<PackageLinearModule<2, 3>, PackageTypedClassBatch>["forward"]>[1] extends PackageTensor<readonly [number]> ? true : false>;
type PackageTrainEvaluateContextShape = Expect<Equal<PackageTrainEvaluateContext["batchIndex"], number>>;
type PackageTrainEvaluateOptionsCallbackShape = Expect<Equal<Parameters<NonNullable<PackageTrainEvaluateOptions["on_step"]>>[0]["kind"], "zgml.train.evaluate-step">>;
type PackageTrainNamespaceEvaluateModuleShape = Expect<Equal<ReturnType<PackageTrainNamespace["evaluateModule"]>, PackageTrainEvaluateEvidence>>;
type PackageTrainNamespacePredictModuleShape = Expect<Equal<ReturnType<PackageTrainNamespace["predictModule"]>["kind"], "zgml.train.predict">>;
type PackageTrainNamespacePredictClassifierShape = Expect<Equal<ReturnType<PackageTrainNamespace["predictClassifier"]>["kind"], "zgml.train.predict">>;
type PackageTrainPredictContextShape = Expect<Equal<PackageTrainPredictContext["sampleIndices"], readonly number[] | null>>;
type PackageTrainPredictOptionsCallbackShape = Expect<Equal<Parameters<NonNullable<PackageTrainPredictOptions<PackageTensor<readonly [2]>>["onStep"]>>[0]["output"], PackageTensor<readonly [2]>>>;
type PackageTrainClassificationClassMetricShape = Expect<Equal<PackageTrainClassificationClassMetric["class_index"], number>>;
type PackageTrainClassificationReportShape = Expect<Equal<PackageTrainClassificationReport["per_class"][number], PackageTrainClassificationClassMetric>>;
type PackageAdamWOptimizerKind = Expect<Equal<PackageOptimizer<"adamw">["kind"], "adamw">>;
type PackageAdamWOptimizerConfigKind = Expect<Equal<ReturnType<PackageOptimizer<"adamw">["config"]>["kind"], "adamw">>;
type PackageAdamWOptimizerStateDictKind = Expect<Equal<PackageOptimizerStateDict<"adamw">["kind"], "adamw" | undefined>>;
type PackageAdamWStateKind = Expect<Equal<PackageOptimizerStateSnapshot<"adamw">["kind"], "adamw">>;
type PackageAdamWConfigKind = Expect<Equal<PackageOptimizerConfigSnapshot<"adamw">["kind"], "adamw">>;
type PackageRMSpropOptimizerKind = Expect<Equal<PackageOptimizer<"rmsprop">["kind"], "rmsprop">>;
type PackageRMSpropOptimizerConfigKind = Expect<Equal<ReturnType<PackageOptimizer<"rmsprop">["config"]>["kind"], "rmsprop">>;
type PackageRMSpropOptimizerStateDictKind = Expect<Equal<PackageOptimizerStateDict<"rmsprop">["kind"], "rmsprop" | undefined>>;
type PackageRMSpropStateKind = Expect<Equal<PackageOptimizerStateSnapshot<"rmsprop">["kind"], "rmsprop">>;
type PackageRMSpropConfigKind = Expect<Equal<PackageOptimizerConfigSnapshot<"rmsprop">["kind"], "rmsprop">>;
type PackageAdagradOptimizerKind = Expect<Equal<PackageOptimizer<"adagrad">["kind"], "adagrad">>;
type PackageAdagradOptimizerConfigKind = Expect<Equal<ReturnType<PackageOptimizer<"adagrad">["config"]>["kind"], "adagrad">>;
type PackageAdagradOptimizerStateDictKind = Expect<Equal<PackageOptimizerStateDict<"adagrad">["kind"], "adagrad" | undefined>>;
type PackageAdagradStateKind = Expect<Equal<PackageOptimizerStateSnapshot<"adagrad">["kind"], "adagrad">>;
type PackageAdagradConfigKind = Expect<Equal<PackageOptimizerConfigSnapshot<"adagrad">["kind"], "adagrad">>;
type PackageStepAdamSchedulerStateKind = Expect<Equal<ReturnType<PackageLRScheduler<"step-lr", "adam">["stateDict"]>["kind"], "step-lr">>;
type PackageStepAdamSchedulerOptimizerKind = Expect<Equal<ReturnType<PackageLRScheduler<"step-lr", "adam">["stateDict"]>["optimizerKind"], "adam">>;
type PackageStepSchedulerStateDictKind = Expect<Equal<PackageLRSchedulerStateDict<"step-lr">["kind"], "step-lr">>;
type PackageExponentialAdamSchedulerStateKind = Expect<Equal<PackageLRSchedulerStateSnapshot<"exponential-lr", "adam">["kind"], "exponential-lr">>;
type PackageCosineAdamSchedulerStateKind = Expect<Equal<PackageLRSchedulerStateSnapshot<"cosine-annealing-lr", "adam">["kind"], "cosine-annealing-lr">>;
type PackagePlateauAdamSchedulerStateKind = Expect<Equal<PackageLRSchedulerStateSnapshot<"reduce-lr-on-plateau", "adam">["kind"], "reduce-lr-on-plateau">>;
type PackageAdamTrainStepEvidenceKind = Expect<Equal<PackageTrainStepEvidence<"adam">["optimizerKind"], "adam">>;
type PackageAdamTrainFitStepEvidenceKind = Expect<Equal<NonNullable<PackageTrainFitStepEvidence<"adam">["stepEvidence"]>["optimizerKind"], "adam">>;
type PackageAdamTrainFitEvidenceKind = Expect<Equal<NonNullable<PackageTrainFitEvidence<"adam">["lastStep"]>["optimizerKind"], "adam">>;
type PackageTrainEvaluateEvidenceKind = Expect<Equal<PackageTrainEvaluateEvidence["kind"], "zgml.train.evaluate">>;
type PackageTrainEvaluateStepEvidenceKind = Expect<Equal<PackageTrainEvaluateStepEvidence["kind"], "zgml.train.evaluate-step">>;
type PackageTrainPredictEvidenceOutput = Expect<Equal<PackageTrainPredictEvidence<PackageTensor<readonly [2]>>["outputs"][number], PackageTensor<readonly [2]>>>;
type PackageTrainPredictStepEvidenceOutput = Expect<Equal<PackageTrainPredictStepEvidence<PackageTensor<readonly [2]>>["output"], PackageTensor<readonly [2]>>>;
type PackageFlattenShapeCheck = Expect<Equal<PackageFlattenShape<readonly [2, 3]>, readonly [6]>>;
type PackageFlatten3dShape = Expect<Equal<PackageFlattenShape<readonly [2, 3, 4]>, readonly [24]>>;
type PackageFlatten3dMiddleShape = Expect<Equal<PackageFlattenShape<readonly [2, 3, 4], 1, -1>, readonly [2, 12]>>;
type PackageFlatten4dMiddleShape = Expect<Equal<PackageFlattenShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 12, 5]>>;
type PackageNarrowRowsShape = Expect<Equal<PackageNarrowShape<readonly [2, 3], 0, 1>, readonly [1, 3]>>;
type PackageNarrowColsShape = Expect<Equal<PackageNarrowShape<readonly [2, 3], -1, 2>, readonly [2, 2]>>;
type PackageNarrow3dMiddleShape = Expect<Equal<PackageNarrowShape<readonly [2, 3, 4], 1, 2>, readonly [2, 2, 4]>>;
type PackageSelect3dTrailingShape = Expect<Equal<PackageSelectShape<readonly [2, 3, 4], -1>, readonly [2, 3]>>;
type PackageSelect4dMiddleShape = Expect<Equal<PackageSelectShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 5]>>;
type PackagePermuteShapeCheck = Expect<Equal<PackagePermuteShape<readonly [2, 3], readonly [1, 0]>, readonly [3, 2]>>;
type PackagePermute3dShapeCheck = Expect<Equal<PackagePermuteShape<readonly [2, 3, 4], readonly [2, 0, 1]>, readonly [4, 2, 3]>>;
type PackagePermute4dShapeCheck = Expect<Equal<PackagePermuteShape<readonly [2, 3, 4, 5], readonly [0, -1, 1, -2]>, readonly [2, 5, 3, 4]>>;
type PackageReduction3dTrailingShape = Expect<Equal<PackageReductionShape<readonly [2, 3, 4], -1>, readonly [2, 3, 1]>>;
type PackageReduction4dDim2Shape = Expect<Equal<PackageReductionShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 1, 5]>>;
type PackageMatrixMatmulShape = Expect<Equal<PackageMatmulShape<readonly [2, 3], readonly [3, 2]>, readonly [2, 2]>>;
type PackageVectorMatrixMatmulShape = Expect<Equal<PackageMatmulShape<readonly [3], readonly [3, 2]>, PackageTensorShapeTuple>>;
type PackageBroadcast3dShape = Expect<Equal<PackageBroadcastShape<readonly [2, 3, 4], readonly [4]>, readonly [2, 3, 4]>>;
type PackageBroadcast4dShape = Expect<Equal<PackageBroadcastShape<readonly [2, 3, 4, 5], readonly [1, 5]>, readonly [2, 3, 4, 5]>>;
type PackageWhereBroadcastShape = Expect<Equal<PackageWhereShape<readonly [2, 3], readonly [1, 3], readonly [1]>, readonly [2, 3]>>;
type PackageReshapeInferShape = Expect<Equal<PackageReshapeShape<readonly [2, 3], readonly [3, -1]>, readonly [3, 2]>>;
type PackageReshapeInfer4dShape = Expect<Equal<PackageReshapeShape<readonly [2, 3, 4, 5], readonly [2, -1, 5]>, readonly [2, 12, 5]>>;
type PackageArangeShapeCheck = Expect<Equal<PackageArangeShape<4>, readonly [4]>>;
type PackageArangeRangeShapeCheck = Expect<Equal<PackageArangeRangeShape<2, 5>, readonly [3]>>;
type PackageSliceShapeCheck = Expect<Equal<PackageSliceShape<readonly [2, 3], 1, 1, 3>, readonly [2, 2]>>;
type PackageSqueezeShapeCheck = Expect<Equal<PackageSqueezeShape<readonly [1, 2, 3], 0>, readonly [2, 3]>>;
type PackageSqueezeAllShapeCheck = Expect<Equal<PackageSqueezeAllShape<readonly [1, 2, 1, 3]>, readonly [2, 3]>>;
type PackageSqueeze4dMiddleShape = Expect<Equal<PackageSqueezeShape<readonly [2, 1, 3, 4], 1>, readonly [2, 3, 4]>>;
type PackageTranspose3dOuterShape = Expect<Equal<PackageTransposeShape<readonly [2, 3, 4], 0, -1>, readonly [4, 3, 2]>>;
type PackageTranspose4dMiddleShape = Expect<Equal<PackageTransposeShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 4, 3, 5]>>;
type PackageUnsqueeze3dMiddleShape = Expect<Equal<PackageUnsqueezeShape<readonly [2, 3, 4], -2>, readonly [2, 3, 1, 4]>>;
type PackageUnsqueeze4dMiddleShape = Expect<Equal<PackageUnsqueezeShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 1, 4, 5]>>;
type PackageTensorShapeOfNumber = Expect<Equal<PackageTensorShapeOf<4>, readonly [4]>>;
type PackageTensorShapeUnion = Expect<Equal<PackageTensorShape, number | PackageTensorShapeTuple>>;
type PackageTensorOptionsShape = Expect<Equal<PackageTensorOptions["requiresGrad"], boolean | undefined>>;
type PackageRandomTensorOptionsShape = Expect<Equal<PackageRandomTensorOptions["seed"], number | undefined>>;
type PackageRandomUniformTensorOptionsShape = Expect<Equal<PackageRandomUniformTensorOptions["rng"], (() => number) | undefined>>;
type PackageRandomIntTensorOptionsShape = Expect<Equal<PackageRandomIntTensorOptions["seed"], number | undefined>>;
type PackageAllCloseOptionsShape = Expect<Equal<PackageAllCloseOptions["rtol"], number | undefined>>;
type PackageTensorJsonShape = Expect<Equal<PackageTensorJSON["dtype"], "f32">>;
type PackageTensorDataShape = Expect<PackageTensorNestedArray extends PackageTensorData ? true : false>;
type PackageTensorDTypeShape = Expect<Equal<PackageTensorDType, "f32">>;
type PackageTensorDTypeLikeShape = Expect<Equal<PackageTensorDTypeLike, "f32" | "float32">>;
type PackageTensorDeviceShape = Expect<Equal<PackageTensorDevice, "cpu">>;
type PackageTensorDeviceLikeShape = Expect<PackageTensorDeviceLike extends "cpu" | "host" | "auto" | "metal" | "webgpu" ? true : false>;
type PackageTensorToTargetShape = Expect<PackageTensorToTarget extends PackageTensorDTypeLike | PackageTensorDeviceLike ? true : false>;
type PackageTensorToOptionsShape = Expect<Equal<PackageTensorToOptions["copy"], boolean | undefined>>;
type PackageTensorInspectionShape = Expect<Equal<PackageTensorInspection["strides"], readonly number[]>>;
type PackageTensorNativeBufferOptionsShape = Expect<Equal<PackageTensorNativeBufferOptions["kind"], "input" | "output" | "weights" | "bias" | "kv-k" | "kv-v" | undefined>>;
type PackageTensorFromNativeBufferOptionsShape = Expect<Equal<PackageTensorFromNativeBufferOptions["byteOffset"], number | undefined>>;
type PackageIndexLikeShape = Expect<Uint32Array extends PackageIndexLike ? true : false>;
type PackageOneHotShapeCheck = Expect<Equal<PackageOneHotShape<readonly [2], 3>, readonly [2, 3]>>;
type PackageCatDim1Shape = Expect<Equal<PackageTensorCatShape<readonly [typeof packageElementwiseInputTensor, typeof packageElementwiseOtherTensor], 1>, readonly [2, 6]>>;
type PackageCat3dDim2Shape = Expect<Equal<PackageTensorCatShape<readonly [typeof packageTensor3d, typeof packageTensor3d], 2>, readonly [2, 3, 8]>>;
type PackageCat4dTrailingShape = Expect<Equal<PackageTensorCatShape<readonly [typeof packageTensor4d, typeof packageTensor4d], -1>, readonly [2, 3, 4, 10]>>;
type PackageStackDim1Shape = Expect<Equal<PackageTensorStackShape<readonly [typeof packageElementwiseInputTensor, typeof packageElementwiseOtherTensor], 1>, readonly [2, 2, 3]>>;
type PackageStack3dDim2Shape = Expect<Equal<PackageTensorStackShape<readonly [typeof packageTensor3d, typeof packageTensor3d], 2>, readonly [2, 3, 2, 4]>>;
type PackageStack4dDim2Shape = Expect<Equal<PackageTensorStackShape<readonly [typeof packageTensor4d, typeof packageTensor4d], 2>, readonly [2, 3, 2, 4, 5]>>;
type PackageEmbeddingGridShape = Expect<Equal<PackageEmbeddingForwardShape<readonly [2, 3], 4>, readonly [2, 3, 4]>>;
type PackageInferredShapeModuleForwardShape = Expect<Equal<PackageShapeModuleForwardShape<"reshape", readonly [3, -1], readonly [2, 3]>, readonly [3, 2]>>;
type PackageLinearForwardShape = Expect<Equal<PackageModuleForwardShape<PackageLinearModule<2, 3>, readonly [2]>, readonly [3]>>;
type PackageLinearBatchForwardShape = Expect<Equal<PackageModuleForwardShape<PackageLinearModule<2, 3>, readonly [5, 2]>, readonly [5, 3]>>;
type PackageSequentialForwardShape = Expect<Equal<PackageModuleForwardShape<PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>, readonly [2]>, readonly [3]>>;
type PackageSequentialPopShape = Expect<Equal<ReturnType<PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>["pop"]>, PackageNnModule | undefined>>;
type PackageSequentialClearShape = Expect<Equal<ReturnType<PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>["clear"]>, PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>>>;
type PackageSequentialNamedModulesShape = Expect<Equal<ReturnType<PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>["namedModules"]>, readonly PackageModuleTraversalEntry[]>>;
type PackageSequentialApplyShape = Expect<Equal<ReturnType<PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>["apply"]>, PackageSequentialModule<readonly [PackageLinearModule<2, 3>]>>>;
type PackageModuleTargetArrayForwardShape = Expect<Equal<PackageModuleTargetForwardShape<readonly [PackageLinearModule<2, 3>], readonly [2]>, readonly [3]>>;
type PackageNnNamespaceLinearReturn = Expect<Equal<ReturnType<PackageNnNamespace["linear"]>, PackageLinearModule<number, number>>>;
type PackagePublicNnNamespaceShape = Expect<Equal<PackagePublicNnNamespace, Readonly<PackageNnNamespace>>>;
type PackageNnNamespaceFunctionalShape = Expect<Equal<PackageNnNamespace["functional"], PackageNnFunctionalNamespace>>;
type PackageParameterTensorShape = Expect<Equal<PackageNnParameter<readonly [2]>["tensor"], PackageTensor<readonly [2]>>>;
type PackageParameterOptionsShape = Expect<Equal<PackageNnParameterOptions<readonly [2]>["shape"], readonly [2] | undefined>>;
type PackageBufferShape = Expect<Equal<PackageNnBuffer<readonly [2]>["shape"], readonly [2]>>;
type PackageBufferOptionsShape = Expect<Equal<PackageNnBufferOptions<readonly [2]>["persistent"], boolean | undefined>>;
type PackageNnInitTargetShape = Expect<PackageNnParameter<readonly [2]> extends PackageNnInitTarget ? true : false>;
type PackageNnInitNamespaceShape = Expect<PackageNnInitNamespace["zeros_"] extends <T extends PackageNnInitTarget>(target: T) => T ? true : false>;
type PackageNnInitKaimingNamespaceShape = Expect<PackageNnInitNamespace["kaiming_uniform_"] extends <T extends PackageNnInitTarget>(target: T, options?: PackageKaimingUniformOptions) => T ? true : false>;
type PackageNnInitXavierNormalNamespaceShape = Expect<PackageNnInitNamespace["xavier_normal_"] extends <T extends PackageNnInitTarget>(target: T, gain?: number) => T ? true : false>;
type PackageNnInitKaimingNormalNamespaceShape = Expect<PackageNnInitNamespace["kaiming_normal_"] extends <T extends PackageNnInitTarget>(target: T, options?: PackageKaimingNormalOptions) => T ? true : false>;
type PackageModuleListItemShape = Expect<Equal<ReturnType<PackageModuleList<readonly [PackageLinearModule<2, 3>]>["get"]>, PackageLinearModule<2, 3> | undefined>>;
type PackageModuleDictItemShape = Expect<Equal<ReturnType<PackageModuleDict<{ readonly head: PackageLinearModule<2, 3> }>["get"]>, PackageNnModule | undefined>>;
type PackageParameterListItemShape = Expect<Equal<ReturnType<PackageParameterList<readonly [PackageNnParameter<readonly [2]>]>["get"]>, PackageNnParameter<readonly [2]> | undefined>>;
type PackageParameterDictItemShape = Expect<Equal<ReturnType<PackageParameterDict<{ readonly weight: PackageNnParameter<readonly [2]> }>["get"]>, PackageNnParameter | undefined>>;
type PackageModuleParameterInfoShape = Expect<Equal<PackageModuleParameterInfo["requires_grad"], boolean>>;
type PackageModuleParameterTraversalOptionsShape = Expect<Equal<PackageModuleParameterTraversalOptions["recurse"], boolean | undefined>>;
type PackageModuleParameterTraversalOptionsPrefixShape = Expect<Equal<PackageModuleParameterTraversalOptions["prefix"], string | undefined>>;
type PackageModuleBufferTraversalOptionsShape = Expect<Equal<PackageModuleBufferTraversalOptions["recurse"], boolean | undefined>>;
type PackageModuleBufferTraversalOptionsPrefixShape = Expect<Equal<PackageModuleBufferTraversalOptions["prefix"], string | undefined>>;
type PackageModuleStateDictOptionsPrefixShape = Expect<Equal<PackageModuleStateDictOptions["prefix"], string | undefined>>;
type PackageNnParametersRecurseShape = Expect<Equal<ReturnType<PackageNnNamespace["parameters"]>, PackageNnParameter[]>>;
type PackageModuleTraversalEntryShape = Expect<Equal<PackageModuleTraversalEntry["module"], PackageNnModule>>;
type PackageModuleStateEntryShape = Expect<Equal<PackageModuleStateEntry["data"], PackageTensorLike>>;
type PackageModuleStateSnapshotEntryShape = Expect<Equal<PackageModuleStateSnapshot[string]["data"], Float32Array>>;
type PackageModuleStateValueShape = Expect<PackageModuleStateEntry extends PackageModuleStateValue ? true : false>;
type PackageModuleStateDictMapShape = Expect<PackageModuleStateDict extends Record<string, unknown> | Map<string, unknown> ? true : false>;
type PackageLoadStateDictValidateOnlyShape = Expect<Equal<PackageLoadStateDictOptions["validateOnly"], boolean | undefined>>;
type PackageNnLinearConfigBiasShape = Expect<Equal<PackageNnLinearConfig["bias"], false | true | PackageTensorLike | undefined>>;
type PackageNnEmbeddingConfigWeightShape = Expect<Equal<PackageNnEmbeddingConfig["weight"], PackageTensorLike | undefined>>;
type PackageNnNormConfigEpsShape = Expect<Equal<PackageNnNormConfig["eps"], number | undefined>>;
type PackageNnDropoutConfigTrainingShape = Expect<Equal<PackageNnDropoutConfig["training"], boolean | undefined>>;
type PackageNnLossConstructorNameShape = Expect<PackageNnLossConstructorName extends keyof PackageNnLossConstructors ? true : false>;
type PackageZeroGradSetToNoneShape = Expect<Equal<PackageZeroGradOptions["set_to_none"], boolean | undefined>>;
type PackageNnOwner = Expect<Equal<typeof packageNnManifest.policyOwner, "src/ts/nn.ts">>;
type PackageOptimOwner = Expect<Equal<typeof packageOptimManifest.policyOwner, "src/ts/optim.ts">>;
type PackageTrainOwner = Expect<Equal<typeof packageTrainManifest.policyOwner, "src/ts/train.ts">>;
type PackageNnProductSourceOfTruth = Expect<Equal<typeof packageNnManifest.productSourceOfTruth, "ts-only">>;
type PackageOptimNativeProductPolicy = Expect<Equal<typeof packageOptimManifest.nativeProductPolicy, "forbidden">>;
type PackageTrainProductOwner = Expect<Equal<typeof packageTrainManifest.productSemanticsOwner, "src/ts/**">>;
type PackageProgramPath = Expect<Equal<typeof packageProgramManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageSessionPath = Expect<Equal<typeof packageSessionManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageStepParamsPath = Expect<Equal<typeof packageStepParamsManifest.runtimePath, "Program -> Session -> StepParams">>;
type PackageProgramProductSourceOfTruth = Expect<Equal<typeof packageProgramManifest.productSourceOfTruth, "ts-only">>;
type PackageSessionNativeProductPolicy = Expect<Equal<typeof packageSessionManifest.nativeProductPolicy, "forbidden">>;
type PackageStepParamsNoMirrors = Expect<Equal<typeof packageStepParamsManifest.handwrittenFrontendMirrors, false>>;
type PackageProgramBindingsInput = Expect<Equal<NonNullable<PackageProgramBindings<readonly [2], readonly [3]>["input"]>, PackageProgramInputBinding<readonly [2]> | PackageNativeBuffer>>;
type PackageProgramBindingsInputShape = Expect<Equal<PackageProgramBindings<readonly [2], readonly [3]>["inputShape"], readonly [2] | undefined>>;
type PackageProgramBindingsOutputShape = Expect<Equal<PackageProgramBindings<readonly [2], readonly [3]>["outputShape"], readonly [3] | undefined>>;
type PackageProgramBindingsOutput = Expect<Equal<NonNullable<PackageProgramBindings<readonly [2], readonly [3]>["output"]>, PackageProgramOutputBinding<readonly [3]>>>;
type PackageProgramBindingPlanInputShape = Expect<Equal<PackageProgramBindingPlan<readonly [2], readonly [3]>["inputShape"], readonly [2] | null>>;
type PackageProgramBindingPlanOutputShape = Expect<Equal<PackageProgramBindingPlan<readonly [2], readonly [3]>["outputShape"], readonly [3] | null>>;
type PackageProgramBufferKindShape = Expect<Equal<PackageProgramBufferKind, "weights" | "bias" | "input">>;
type PackageProgramExternalResourceAccessShape = Expect<PackageProgramExternalResourceAccessName extends PackageProgramExternalResourceAccess ? true : false>;
type PackageProgramExternalResourceOptionsShape = Expect<Equal<PackageProgramExternalResourceOptions["byteLength"], number>>;
type PackageProgramBufferCreateOptionsShape = Expect<Equal<ReturnType<NonNullable<PackageProgramBufferCreateOptions["resource"]>>, PackageNativeBuffer>>;
type PackageProgramOutputBufferCreateOptionsShape = Expect<Equal<ReturnType<NonNullable<PackageProgramOutputBufferCreateOptions["resource"]>>, PackageNativeBuffer>>;
type PackageProgramCreateBufferOptionsShape = Expect<PackageProgramCreateBufferOptions extends PackageProgramBufferCreateOptions | PackageProgramOutputBufferCreateOptions ? true : false>;
type PackageProgramBufferSlotShape = Expect<Equal<PackageProgramBufferSlot["kind"], PackageProgramBufferKind>>;
type PackageProgramOutputBufferSlotShape = Expect<Equal<PackageProgramOutputBufferSlot["kind"], "output">>;
type PackageProgramDeviceBufferImportOptionsShape = Expect<Equal<PackageProgramDeviceBufferImportOptions["byteLength"], number | undefined>>;
type PackageProgramDeviceBufferImportDescriptorShape = Expect<Equal<PackageProgramDeviceBufferImportDescriptor["byteOffset"], number | undefined>>;
type PackageProgramDeviceBufferImportSourceShape = Expect<PackageNativeBuffer extends PackageProgramDeviceBufferImportSource ? true : false>;
type PackageProgramDeviceImportBufferSourceShape = Expect<PackageNativeBuffer extends PackageProgramDeviceImportBufferSource ? true : false>;
type PackageProgramHostGpuBufferImportSourceShape = Expect<Equal<PackageProgramHostGpuBufferImportSource["byteLen"], number | undefined>>;
type PackageProgramWebGpuInteropImportFieldsShape = Expect<Equal<PackageProgramWebGpuInteropImportFields[PackageProgramWebGpuInteropSymbols["byteLength"]], number | undefined>>;
type PackageProgramWebGpuInteropImportSourceShape = Expect<PackageProgramWebGpuInteropImportSource extends Record<string | symbol, unknown> ? true : false>;
type PackageProgramOutputShapeMethod = Expect<Equal<PackageProgram<readonly [2], readonly [3]>["outputShape"], () => readonly [3]>>;
type PackagePublicProgramNamespaceShape = Expect<Equal<PackagePublicProgramNamespace, Readonly<PackageProgramNamespace>>>;
type PackageSessionOutputShapeMethod = Expect<Equal<PackageSession<readonly [2], readonly [3]>["outputShape"], () => readonly [3]>>;
type PackagePublicSessionNamespaceShape = Expect<Equal<PackagePublicSessionNamespace, Readonly<PackageSessionNamespace>>>;
type PackageSessionStepParamsInput = Expect<Equal<NonNullable<PackageSessionStepParams<readonly [2], readonly [3]>["input"]>, PackageSessionProgramInputBinding<readonly [2]>>>;
type PackageSessionStepParamsOutput = Expect<Equal<NonNullable<PackageSessionStepParams<readonly [2], readonly [3]>["output"]>, PackageSessionProgramOutputBinding<readonly [3]> | false>>;
type PackageSessionExecuteParamsInput = Expect<Equal<NonNullable<PackageSessionExecuteParams<readonly [2], readonly [3]>["input"]>, PackageSessionProgramInputBinding<readonly [2]>>>;
type PackageSessionExecuteTensorParamsOutput = Expect<NonNullable<PackageSessionExecuteTensorParams<readonly [2], readonly [3]>["output"]> extends PackageTensor<readonly [3]> | Float32Array ? true : false>;
type PackageSessionStepTensorOptionsShape = Expect<Equal<PackageSessionStepTensorOptions["requiresGrad"], boolean | undefined>>;
type PackageSessionReadOutputTensorOptionsShape = Expect<Equal<PackageSessionReadOutputTensorOptions["byteOffset"], number | undefined>>;
type PackageSessionExecuteIntoParamsInput = Expect<Equal<NonNullable<PackageSessionExecuteIntoParams<readonly [2]>["input"]>, PackageSessionProgramInputBinding<readonly [2]>>>;
type PackageStepParamsSessionInput = Expect<Equal<NonNullable<PackageStepParamsSessionStepParams<readonly [2], readonly [3]>["input"]>, PackageStepParamsProgramInputBinding<readonly [2]>>>;
type PackageStepParamsSessionOutput = Expect<Equal<NonNullable<PackageStepParamsSessionStepParams<readonly [2], readonly [3]>["output"]>, PackageStepParamsProgramOutputBinding<readonly [3]> | false>>;
type PackageNativeBufferPath = Expect<Equal<typeof packageNativeBufferManifest.runtimePath, "Program -> Session -> NativeBuffer">>;
type PackageProgramDevicePath = Expect<Equal<typeof packageProgramDeviceManifest.runtimePath, "Program -> ProgramDevice -> NativeBuffer">>;
type PackageNativeBufferProductSourceOfTruth = Expect<Equal<typeof packageNativeBufferManifest.productSourceOfTruth, "ts-only">>;
type PackageProgramDeviceNativeProductPolicy = Expect<Equal<typeof packageProgramDeviceManifest.nativeProductPolicy, "forbidden">>;
type PackageNativeExternalResourceOptionsShape = Expect<Equal<PackageNativeExternalResourceOptions["access"], PackageNativeExternalResourceAccess | undefined>>;
type PackageNativeProgramDeviceBufferImportOptionsShape = Expect<Equal<PackageNativeProgramDeviceBufferImportOptions["byteLen"], number | undefined>>;
type PackageNativeProgramDeviceBufferImportSourceShape = Expect<PackageNativeBuffer extends PackageNativeProgramDeviceBufferImportSource ? true : false>;
type PackageNativeWebGpuInteropImportSourceShape = Expect<PackageNativeWebGpuInteropImportSource extends Record<string | symbol, unknown> ? true : false>;
type PackageNativeWebGpuInteropSymbolsShape = Expect<Equal<keyof PackageNativeWebGpuInteropSymbols, keyof PackageProgramWebGpuInteropSymbols>>;
type PackageDeviceProgramDeviceBufferImportOptionsShape = Expect<Equal<PackageDeviceProgramDeviceBufferImportOptions["byteLen"], number | undefined>>;
type PackageDeviceProgramDeviceImportBufferSourceShape = Expect<PackageNativeBuffer extends PackageDeviceProgramDeviceImportBufferSource ? true : false>;
type PackageDeviceWebGpuInteropImportSourceShape = Expect<PackageDeviceWebGpuInteropImportSource extends Record<string | symbol, unknown> ? true : false>;
type PackageDeviceWebGpuInteropSymbolsShape = Expect<Equal<keyof PackageDeviceWebGpuInteropSymbols, keyof PackageProgramWebGpuInteropSymbols>>;
type PackageRuntimeInspectionOwner = Expect<Equal<typeof packageRuntimeInspectionManifest.policyOwner, "src/ts/runtime/inspection.ts">>;
type PackageRuntimeModelSourceOwner = Expect<Equal<typeof packageRuntimeModelSourceManifest.policyOwner, "src/ts/runtime/model_source.ts">>;
type PackageRuntimeNativeBufferOwner = Expect<Equal<typeof packageRuntimeNativeBufferManifest.policyOwner, "src/ts/runtime/native_buffer.ts">>;
type PackageRuntimeProgramDeviceOwner = Expect<Equal<typeof packageRuntimeProgramDeviceManifest.policyOwner, "src/ts/runtime/program_device.ts">>;
type PackageTopLevelKernelPlanOwner = Expect<Equal<typeof packageTopLevelKernelPlanManifest.policyOwner, "src/ts/runtime/kernel_plan.ts">>;
type PackageTopLevelKernelPlanShape = Expect<Equal<PackageTopLevelKernelPlan, PackageModuleKernelPlan>>;
type PackageTensorOwner = Expect<Equal<typeof packageTensorManifest.policyOwner, "src/ts/tensor.ts">>;
type PackageNativeApiContractOwner = Expect<Equal<typeof packageNativeApiContractManifest.policyOwner, "src/ts/runtime/native_api_contract.ts">>;
type PackageNativeApiContractProductOwner = Expect<Equal<typeof packageNativeApiContractManifest.productSemanticsOwner, "src/ts/**">>;
type PackageNativeApiContractNativeAlignment = Expect<Equal<typeof packageNativeApiContractManifest.nativeAlignment, "contract-tested-substrate">>;
type PackageNativeApiContractProductSourceOfTruth = Expect<Equal<typeof packageNativeApiContractManifest.productSourceOfTruth, "ts-only">>;
type PackageNativeApiContractNativeProductPolicy = Expect<Equal<typeof packageNativeApiContractManifest.nativeProductPolicy, "forbidden">>;
type PackageNativeApiContractNoMirrors = Expect<Equal<typeof packageNativeApiContractManifest.handwrittenFrontendMirrors, false>>;
type PackageNativeApiContractPackageSpine = Expect<Equal<typeof packageNativeApiContractManifest.packageSpine, PackageNativePackageSpineContractManifest>>;
type PackageNativePackageSpineSource = Expect<Equal<typeof packageNativePackageSpineContractManifest.source, "generated-ts">>;
type PackageNativePackageSpineGenerator = Expect<Equal<typeof packageNativePackageSpineContractManifest.generator, "scripts/generate_native_runtime_wrappers.cjs">>;
type PackageNativePackageSpineGeneratedFrom = Expect<Equal<typeof packageNativePackageSpineContractManifest.generatedFrom, "src/ts/runtime/native_api_contract.ts">>;
type PackageNativePackageSpineNoHandwrittenRootExports = Expect<Equal<typeof packageNativePackageSpineContractManifest.handwrittenRootExportList, false>>;
type PackageNodeAdapterEvidenceShape = Expect<Equal<typeof packageNodeAdapterEvidence, PackageNativeAdapterEvidence>>;
type PackageBunAdapterEvidenceShape = Expect<Equal<typeof packageBunAdapterEvidence, PackageNativeAdapterEvidence>>;
type PackageNodeAdapterEvidenceProductOwner = Expect<Equal<typeof packageNodeAdapterEvidence.productSemanticsOwner, "src/ts/**">>;
type PackageBunAdapterEvidenceNativeAlignment = Expect<Equal<typeof packageBunAdapterEvidence.nativeAlignment, "contract-tested-substrate">>;
type PackageNodeAdapterManifestOwner = Expect<Equal<typeof packageNodeAdapterManifest.policyOwner, "src/ts/adapters/node.ts">>;
type PackageBunAdapterManifestOwner = Expect<Equal<typeof packageBunAdapterManifest.policyOwner, "src/ts/adapters/bun.ts">>;
type PackageNodeAdapterManifestRuntimeKind = Expect<Equal<typeof packageNodeAdapterManifest.concreteRuntime.runtimeLoad.runtimeKind, "node-ffi-runtime">>;
type PackageBunAdapterManifestRuntimeKind = Expect<Equal<typeof packageBunAdapterManifest.concreteRuntime.runtimeLoad.runtimeKind, "bun-ffi-runtime">>;
type PackageNodeAdapterManifestLoaderAlignment = Expect<Equal<typeof packageNodeAdapterManifest.concreteRuntime.loader.nativeAlignment, "contract-tested-substrate">>;
type PackageBunAdapterManifestLoaderProductOwner = Expect<Equal<typeof packageBunAdapterManifest.concreteRuntime.loader.productSemanticsOwner, "src/ts/**">>;
type PackageNodeAdapterManifestNoMirrors = Expect<Equal<typeof packageNodeAdapterManifest.concreteRuntime.loader.handwrittenFrontendMirrors, false>>;
type PackageNodeRuntimeExtendsNativeRuntime = Expect<typeof import("zgml/node") extends PackageNativeRuntime ? true : false>;
type PackageBunRuntimeExtendsNativeRuntime = Expect<typeof import("zgml/bun") extends PackageNativeRuntime ? true : false>;
type PackageNativeRuntimeTensorShape = Expect<Equal<ReturnType<PackageNativeRuntime["tensor"]>, PackageTensor>>;
type PackageNativeRuntimeNnShape = Expect<Equal<PackageNativeRuntime["nn"], PackagePublicNnNamespace>>;
type PackageNativeRuntimeFShape = Expect<Equal<PackageNativeRuntime["F"], PackageNnFunctionalNamespace>>;
type PackageNativeRuntimeGradModeShape = Expect<PackageNativeRuntime["gradMode"] extends { isGradEnabled: () => boolean; noGrad: <T>(fn: () => T) => T } ? true : false>;
type PackageNativeRuntimeDataShape = Expect<Equal<PackageNativeRuntime["data"], PackagePublicDataNamespace>>;
type PackageNativeRuntimeTrainShape = Expect<Equal<PackageNativeRuntime["train"], PackagePublicTrainNamespace>>;
type PackageNativeRuntimeProgramShape = Expect<PackageNativeRuntime["Program"] extends new (...args: never[]) => PackageProgram ? true : false>;

const shapeCount: number = shape.shapeScalarCount([2, 3, 4]);
const tokenValue: number = token.tokenId(42);
const ownsPolicy: false = adapterOwnsFrontendPolicy(packageNodeAdapterEvidence);
const nativeApiContractParts: PackageNativeApiContractSignatureParts = packageNativeApiContractSignatureParts();
const nativeApiContractSignature: string = packageNativeApiContractSignature();
const formattedNativeApiContractSignature: string = packageFormatNativeApiContractSignature(nativeApiContractParts);
const nativeApiContractApiCount: number = packageRequiredNativeApiExports.length;
const nativeApiContractSpineCount: number = packageRequiredNativePackageSpineExports.length;
const nativeApiContractSpineIncludesF: boolean = packageRequiredNativePackageSpineExports.includes("F");
const nativeApiContractSpineIncludesGradMode: boolean = packageRequiredNativePackageSpineExports.includes("gradMode");
const extension: string = nativeLibraryExtensionForPlatform("darwin");
const filename: string = packageNativeLibraryFilename("so");
const missingMessage: string = nativeLibraryMissingMessage("/tmp/libzgml_c.so", "so");
const koffiFallback: string = nodeKoffiFallbackPath({
  moduleDir: "/pkg/js",
  joinPath: (...parts: string[]) => parts.join("/"),
});
const pathOptions: NativeLibraryPathOptions = {
  moduleDir: "/pkg/js",
  cwd: "/repo",
  extension: "so",
  joinPath: (...parts: string[]) => parts.join("/"),
  exists: () => false,
};
const nativePath: string = resolveNativeLibraryPath(pathOptions);
const bunNativePath: string = bunResolveNativeLibraryPath(pathOptions);
const linearFactory: Function = nnLinear.createLinearModuleClass;
const genericStepCompatibility: Function = sessionValues.genericSessionStepParamsCompatibility;
const moduleFacadeFactory: Function = moduleFacade.createModuleFacadeHelpers;
const moduleCompilerFlatten: Function = moduleCompilerPolicy.flattenSequentialEntries;
const packageFrontendShapeCount: number = packageFrontendShape.shapeScalarCount([2, 3]);
const packageSessionFacadeFactory: Function = packageSessionFacade.createGenericSessionExecutionFacadeHelpers;
const packageCheckpointFactory: Function = packageCreateCheckpointHelpers;
const packageTraceCompiler: Function = packageTraceCompilerArtifacts;
const packageCompileModulePlanPredicate: Function = packageCompileAcceptsModuleCompilePlan;
const packageCompileModulePlanRequire: Function = packageCompileRequireModuleCompilePlan;
const packageCompileModulePlanSignatureMatch: Function = packageCompileMatchesModuleCompilePlanSignature;
const packageCompileAnalysisPredicate: Function = packageIsCompileAnalysis;
const packageCompileAnalysisRequire: Function = packageRequireCompileAnalysis;
const packageCompileAnalysisSignatureMatch: Function = packageMatchesCompileAnalysisSignature;
const packageCompileProgramEvidencePredicate: Function = packageCompileIsProgramCompileEvidence;
const packageCompileProgramEvidenceRequire: Function = packageCompileRequireProgramCompileEvidence;
const packageCompileProgramEvidenceSignatureMatch: Function = packageCompileMatchesProgramCompileEvidenceSignature;
const packageProgramEvidencePredicate: Function = packageProgramIsProgramCompileEvidence;
const packageProgramEvidenceRequire: Function = packageProgramRequireProgramCompileEvidence;
const packageProgramEvidenceSignatureMatch: Function = packageProgramMatchesProgramCompileEvidenceSignature;
const packageInspectionMode: string = packageProgramExecutionModeFromCapabilities(null);
const packageInspectionProgramPlanPredicate: Function = packageInspectionAcceptsProgramExecutionPlan;
const packageInspectionSessionPlanPredicate: Function = packageInspectionAcceptsSessionExecutionPlan;
const packageInspectionProgramPlanRequire: Function = packageInspectionRequireProgramExecutionPlan;
const packageInspectionSessionPlanRequire: Function = packageInspectionRequireSessionExecutionPlan;
const packageInspectionProgramBindingPlanPredicate: Function = packageInspectionAcceptsProgramBindingPlan;
const packageInspectionModuleBindingPlanPredicate: Function = packageInspectionAcceptsModuleBindingPlan;
const packageInspectionProgramBindingPlanRequire: Function = packageInspectionRequireProgramBindingPlan;
const packageInspectionModuleBindingPlanRequire: Function = packageInspectionRequireModuleBindingPlan;
const packageInspectionModuleCompilePlanPredicate: Function = packageInspectionAcceptsModuleCompilePlan;
const packageInspectionModuleCompilePlanRequire: Function = packageInspectionRequireModuleCompilePlan;
const packageInspectionModuleCompilePlanSignatureMatch: Function = packageInspectionMatchesModuleCompilePlanSignature;
const packageInspectionRuntimeProfilePredicate: Function = packageInspectionAcceptsRuntimeProfile;
const packageInspectionRuntimeProfileRequire: Function = packageInspectionRequireRuntimeProfile;
const packageInspectionRuntimeProfileSignatureMatch: Function = packageInspectionMatchesRuntimeProfileSignature;
const packageInspectionSessionCallProfilePredicate: Function = packageInspectionAcceptsSessionCallProfile;
const packageInspectionSessionCallProfileRequire: Function = packageInspectionRequireSessionCallProfile;
const packageInspectionSessionCallProfileSignatureMatch: Function = packageInspectionMatchesSessionCallProfileSignature;
const packageModelSourceKind: string = packageNormalizeLoadModelKind("tinyllama");
const packageLossFactory: Function = packageCreateLossTrainHelpers;
const packageDataFactory: Function = packageCreateDataNamespace;
const packageNnFactory: Function = packageCreateNnNamespace;
const packageNnModuleBindingPlanPredicate: Function = packageAcceptsModuleBindingPlan;
const packageNnModuleBindingPlanRequire: Function = packageRequireModuleBindingPlan;
const packageNnModuleBindingPlanSignatureMatch: Function = packageMatchesModuleBindingPlanSignature;
const packageNnModuleCompilePlanPredicate: Function = packageAcceptsModuleCompilePlan;
const packageNnModuleCompilePlanRequire: Function = packageRequireModuleCompilePlan;
const packageNnModuleCompilePlanSignatureMatch: Function = packageMatchesModuleCompilePlanSignature;
const packageOptimFactory: Function = packageCreateOptimNamespace;
const packageOptimizerConfigPredicate: Function = packageIsOptimizerConfigSnapshot;
const packageOptimizerConfigRequire: Function = packageRequireOptimizerConfigSnapshot;
const packageOptimizerConfigSignatureMatch: Function = packageMatchesOptimizerConfigSnapshotSignature;
const packageOptimizerStatePredicate: Function = packageIsOptimizerStateSnapshot;
const packageOptimizerStateRequire: Function = packageRequireOptimizerStateSnapshot;
const packageOptimizerStateSignatureMatch: Function = packageMatchesOptimizerStateSnapshotSignature;
const packageLRSchedulerStatePredicate: Function = packageIsLRSchedulerStateSnapshot;
const packageLRSchedulerStateRequire: Function = packageRequireLRSchedulerStateSnapshot;
const packageLRSchedulerStateSignatureMatch: Function = packageMatchesLRSchedulerStateSnapshotSignature;
const packageTrainFactory: Function = packageCreateTrainHelpers;
const packageTrainStepEvidencePredicate: Function = packageIsTrainStepEvidence;
const packageTrainStepEvidenceRequire: Function = packageRequireTrainStepEvidence;
const packageTrainStepEvidenceSignatureMatch: Function = packageMatchesTrainStepEvidenceSignature;
const packageTrainFitStepEvidencePredicate: Function = packageIsTrainFitStepEvidence;
const packageTrainFitStepEvidenceRequire: Function = packageRequireTrainFitStepEvidence;
const packageTrainFitStepEvidenceSignatureMatch: Function = packageMatchesTrainFitStepEvidenceSignature;
const packageTrainFitEvidencePredicate: Function = packageIsTrainFitEvidence;
const packageTrainFitEvidenceRequire: Function = packageRequireTrainFitEvidence;
const packageTrainFitEvidenceSignatureMatch: Function = packageMatchesTrainFitEvidenceSignature;
const packageTrainEvaluateStepEvidencePredicate: Function = packageIsTrainEvaluateStepEvidence;
const packageTrainEvaluateStepEvidenceRequire: Function = packageRequireTrainEvaluateStepEvidence;
const packageTrainEvaluateStepEvidenceSignatureMatch: Function = packageMatchesTrainEvaluateStepEvidenceSignature;
const packageTrainEvaluateEvidencePredicate: Function = packageIsTrainEvaluateEvidence;
const packageTrainEvaluateEvidenceRequire: Function = packageRequireTrainEvaluateEvidence;
const packageTrainEvaluateEvidenceSignatureMatch: Function = packageMatchesTrainEvaluateEvidenceSignature;
const packageTrainPredictStepEvidencePredicate: Function = packageIsTrainPredictStepEvidence;
const packageTrainPredictStepEvidenceRequire: Function = packageRequireTrainPredictStepEvidence;
const packageTrainPredictStepEvidenceSignatureMatch: Function = packageMatchesTrainPredictStepEvidenceSignature;
const packageTrainPredictEvidencePredicate: Function = packageIsTrainPredictEvidence;
const packageTrainPredictEvidenceRequire: Function = packageRequireTrainPredictEvidence;
const packageTrainPredictEvidenceSignatureMatch: Function = packageMatchesTrainPredictEvidenceSignature;
const packageProgramFactory: Function = packageCreateGenericProgramFacadeHelpers;
const packageProgramExecutionPlanPredicate: Function = packageAcceptsProgramExecutionPlan;
const packageProgramExecutionPlanRequire: Function = packageRequireProgramExecutionPlan;
const packageProgramExecutionPlanSignatureMatch: Function = packageMatchesProgramExecutionPlanSignature;
const packageProgramBindingPlanPredicate: Function = packageAcceptsProgramBindingPlan;
const packageProgramBindingPlanRequire: Function = packageRequireProgramBindingPlan;
const packageProgramBindingPlanSignatureMatch: Function = packageMatchesProgramBindingPlanSignature;
const packageTypedProgramBindings: PackageProgramBindings<readonly [2], readonly [3]> = {
  inputShape: [2] as const,
  outputShape: [3] as const,
};
const packageTypedProgramBindingsInputShape: readonly [2] | undefined = packageTypedProgramBindings.inputShape;
const packageTypedProgramBindingsOutputShape: readonly [3] | undefined = packageTypedProgramBindings.outputShape;
declare const packageTypedInputTensor: PackageTensor<readonly [2]>;
declare const packageTypedOutputTensor: PackageTensor<readonly [3]>;
declare const packageElementwiseInputTensor: PackageTensor<readonly [2, 3]>;
declare const packageElementwiseOtherTensor: PackageTensor<readonly [2, 3]>;
declare const packageTensor3d: PackageTensor<readonly [2, 3, 4]>;
declare const packageTensor4d: PackageTensor<readonly [2, 3, 4, 5]>;
const packageElementwiseAddTensor: PackageTensor<readonly [2, 3]> = packageElementwiseInputTensor.add(packageElementwiseOtherTensor);
const packageBroadcastAddTensor: PackageTensor<readonly [2, 3]> = packageElementwiseInputTensor.add(PackageNodeTensor.tensor([1, 2, 3], [1, 3] as const));
const packageBroadcastWhereTensor: PackageTensor<readonly [2, 3]> = packageElementwiseInputTensor.gt(0).where(PackageNodeTensor.tensor([1, 2, 3], [1, 3] as const), PackageNodeTensor.tensor([0], [1] as const));
const packageElementwiseEqTensor: PackageTensor<readonly [2, 3]> = packageElementwiseInputTensor.eq(packageElementwiseOtherTensor);
const packageElementwiseMaximumTensor: PackageTensor<readonly [2, 3]> = packageElementwiseInputTensor.maximum(packageElementwiseOtherTensor);
const packageFlattenTensor: PackageTensor<readonly [6]> = packageElementwiseInputTensor.flatten();
const packageInferReshapeTensor: PackageTensor<readonly [3, 2]> = packageElementwiseInputTensor.reshape([3, -1] as const);
const packageInferViewTensor: PackageTensor<readonly [1, 6]> = PackageNodeTensor.view(packageElementwiseInputTensor, [1, -1] as const);
const packageFlatten3dTensor: PackageTensor<readonly [24]> = packageTensor3d.flatten();
const packageFlatten3dMiddleTensor: PackageTensor<readonly [2, 12]> = packageTensor3d.flatten(1, -1);
const packageNarrowRowsTensor: PackageTensor<readonly [1, 3]> = packageElementwiseInputTensor.narrow(0, 0, 1);
const packageNarrowColsTensor: PackageTensor<readonly [2, 2]> = packageElementwiseInputTensor.narrow(-1, 0, 2);
const packageNarrow3dMiddleTensor: PackageTensor<readonly [2, 2, 4]> = packageTensor3d.narrow(1, 0, 2);
const packageSelect3dTrailingTensor: PackageTensor<readonly [2, 3]> = packageTensor3d.select(-1, 2);
const packageSliceColsTensor: PackageTensor<readonly [2, 2]> = packageElementwiseInputTensor.slice(1, 1, 3);
const packageStaticSlice3dTensor: PackageTensor<readonly [2, 2, 4]> = PackageNodeTensor.slice(packageTensor3d, 1, 0, 2);
const packageUnsqueeze3dMiddleTensor: PackageTensor<readonly [2, 3, 1, 4]> = packageTensor3d.unsqueeze(-2);
const packageTranspose3dOuterTensor: PackageTensor<readonly [4, 3, 2]> = packageTensor3d.transpose(0, -1);
const packageCatDim1Tensor: PackageTensor<readonly [2, 6]> = PackageNodeTensor.cat([packageElementwiseInputTensor, packageElementwiseOtherTensor] as const, 1);
const packageCat3dDim2Tensor: PackageTensor<readonly [2, 3, 8]> = PackageNodeTensor.cat([packageTensor3d, packageTensor3d] as const, 2);
const packageCat4dTrailingTensor: PackageTensor<readonly [2, 3, 4, 10]> = PackageNodeTensor.cat([packageTensor4d, packageTensor4d] as const, -1);
const packagePermuteTensor: PackageTensor<readonly [3, 2]> = packageElementwiseInputTensor.permute([1, 0] as const);
const packagePermute3dTensor: PackageTensor<readonly [4, 2, 3]> = packageTensor3d.permute([2, 0, 1] as const);
const packagePermute4dTensor: PackageTensor<readonly [2, 5, 3, 4]> = packageTensor4d.permute([0, -1, 1, -2] as const);
const packageSum3dTrailingTensor: PackageTensor<readonly [2, 3, 1]> = packageTensor3d.sumDim(-1);
const packageMean3dDim0Tensor: PackageTensor<readonly [1, 3, 4]> = PackageNodeTensor.mean(packageTensor3d, 0);
declare const packageSqueezeInputTensor: PackageTensor<readonly [1, 2, 3]>;
const packageSqueezeTensor: PackageTensor<readonly [2, 3]> = packageSqueezeInputTensor.squeeze(0);
const packageStackDim1Tensor: PackageTensor<readonly [2, 2, 3]> = PackageNodeTensor.stack([packageElementwiseInputTensor, packageElementwiseOtherTensor] as const, 1);
const packageStack3dDim2Tensor: PackageTensor<readonly [2, 3, 2, 4]> = PackageNodeTensor.stack([packageTensor3d, packageTensor3d] as const, 2);
const packageStack4dDim2Tensor: PackageTensor<readonly [2, 3, 2, 4, 5]> = PackageNodeTensor.stack([packageTensor4d, packageTensor4d] as const, 2);
const packageEinsumMatmulTensor: PackageTensor = PackageNodeTensor.einsum("ij,jk->ik", [packageElementwiseInputTensor, packageElementwiseOtherTensor]);
const packageEinsumTraceTensor: PackageTensor = PackageNodeTensor.einsum("ii->", PackageNodeTensor.tensor([1, 2, 3, 4], [2, 2] as const));
const packageEinsumEllipsisTensor: PackageTensor = PackageNodeTensor.einsum("...i->...", PackageNodeTensor.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const));
const packageNodeLinspaceTensor: PackageTensor<readonly [4]> = PackageNodeTensor.linspace(0, 1, 4);
const packageNodeArangeTensor: PackageTensor<readonly [4]> = PackageNodeTensor.arange(4);
const packageNodeArangeRangeTensor: PackageTensor<readonly [3]> = PackageNodeTensor.arange(2, 5);
const packageInferredReshapeForwardTensor: PackageTensor<readonly [3, 2]> = packageNodeNn.reshape([3, -1] as const).forward(packageElementwiseInputTensor);
declare const packageMseLoss: PackageMSELoss;
declare const packageCrossEntropyLoss: PackageCrossEntropyLoss;
declare const packageNllLoss: PackageNLLLoss;
declare const packageLossTensor: PackageTensor<readonly [2]>;
declare const packageLogitTensor: PackageTensor<readonly [1, 3]>;
declare const packageLogProbabilityTensor: PackageTensor<readonly [1, 3]>;
const packageMseLossTensor: PackageTensor<readonly [1]> = packageMseLoss.forward(packageLossTensor, packageLossTensor);
const packageCrossEntropyLossTensor: PackageTensor<readonly [1]> = packageCrossEntropyLoss.forward(packageLogitTensor, [2] as const);
const packageNllLossTensor: PackageTensor<readonly [1]> = packageNllLoss.forward(packageLogProbabilityTensor, [2] as const);
const packageTypedProgramInputBinding: PackageProgramInputBinding<readonly [2]> = packageTypedInputTensor;
const packageTypedProgramOutputBinding: PackageProgramOutputBinding<readonly [3]> = packageTypedOutputTensor;
const packageTypedSessionStepParams: PackageSessionStepParams<readonly [2], readonly [3]> = {
  input: packageTypedInputTensor,
  output: packageTypedOutputTensor,
};
const packageTypedStepParamsSubpathStepParams: PackageStepParamsSessionStepParams<readonly [2], readonly [3]> = {
  input: packageTypedInputTensor,
  output: packageTypedOutputTensor,
};
const packageSessionFactory: Function = packageCreateSessionLiveFacadeHelpers;
const packageSessionExecutionPlanPredicate: Function = packageAcceptsSessionExecutionPlan;
const packageSessionExecutionPlanRequire: Function = packageRequireSessionExecutionPlan;
const packageSessionExecutionPlanSignatureMatch: Function = packageMatchesSessionExecutionPlanSignature;
const packageSessionCallProfilePredicate: Function = packageAcceptsSessionCallProfile;
const packageSessionCallProfileRequire: Function = packageRequireSessionCallProfile;
const packageSessionCallProfileSignatureMatch: Function = packageMatchesSessionCallProfileSignature;
const packageStepParamsCompatibilityPredicate: Function = packageIsStepParamsCompatibility;
const packageStepParamsCompatibilityRequire: Function = packageRequireStepParamsCompatibility;
const packageFriendlyStepParamsAccepted: boolean = packageFriendlyStepParamsCompatibilityResult({
  kind: "friendly-step-params-type-smoke",
  signature: "friendly-step-params-type-smoke",
}, null).accepted;
const packageNativeBufferWrite: PackageNativeBufferByteRangeInfo = packageNativeBufferWriteInfo(16, 4);
const packageNativeBufferWriteCompat: { readonly byteOffset: number; readonly byteLength: number } = packageNativeBufferWrite;
const packageNativeBufferByteRangePredicate: Function = packageIsNativeBufferByteRangeInfo;
const packageNativeBufferByteRangeRequire: Function = packageRequireNativeBufferByteRangeInfo;
const packageNativeBufferByteRangeSignatureMatch: Function = packageMatchesNativeBufferByteRangeSignature;
const packageProgramDeviceBufferImport: PackageProgramDeviceBufferImportInfo = packageProgramDeviceBufferImportInfo({ deviceHandle: 1, handle: 2, byteLength: 4 }, {});
const packageProgramDeviceBufferImportPredicate: Function = packageIsProgramDeviceBufferImportInfo;
const packageProgramDeviceBufferImportRequire: Function = packageRequireProgramDeviceBufferImportInfo;
const packageProgramDeviceBufferImportSignatureMatch: Function = packageMatchesProgramDeviceBufferImportSignature;
const packageNativeBufferDeviceImport: PackageNativeBufferDeviceImportInfo = packageNativeBufferDeviceImportOptions("webgpu", {
  storage: "external-resource",
  placement: "webgpu",
  handle: 2,
  byteOffset: 0,
  byteLength: 4,
}, 1);
const packageNativeBufferDeviceImportPredicate: Function = packageIsNativeBufferDeviceImportInfo;
const packageNativeBufferDeviceImportRequire: Function = packageRequireNativeBufferDeviceImportInfo;
const packageNativeBufferDeviceImportSignatureMatch: Function = packageMatchesNativeBufferDeviceImportSignature;
const packageNativeBufferExternalResource: PackageNativeBufferExternalResourceInfo = packageExternalResourceInfo({ handle: 2, byteLength: 4 });
const packageNativeBufferExternalResourcePredicate: Function = packageIsNativeBufferExternalResourceInfo;
const packageNativeBufferExternalResourceRequire: Function = packageRequireNativeBufferExternalResourceInfo;
const packageNativeBufferExternalResourceSignatureMatch: Function = packageMatchesNativeBufferExternalResourceSignature;
const packageProgramDeviceFactory: Function = packageCreateProgramDeviceClass;
const packageProgramDeviceEvidence: PackageProgramDeviceInfo = packageProgramDeviceInfo("webgpu", 7);
const packageProgramDeviceInfoPredicate: Function = packageIsProgramDeviceInfo;
const packageProgramDeviceInfoRequire: Function = packageRequireProgramDeviceInfo;
const packageProgramDeviceInfoSignatureMatch: Function = packageMatchesProgramDeviceInfoSignature;
const packageTensorFactory: Function = packageCreateTensorFacadeHelpers;
const packageNativeFilename: string = packageNativeLibraryFilename("dylib");
const packageNodeEvidenceHost: PackageNativeAdapterEvidence["host"] = packageNodeAdapterEvidence.host;
const packageBunEvidenceHost: PackageNativeAdapterEvidence["host"] = packageBunAdapterEvidence.host;
const packageNodeAdapterRuntimePackageFirst: boolean = packageNodeAdapterManifest.concreteRuntime.runtimeLoad.packageArtifactFirst;
const packageBunAdapterRuntimeHasSourceFallback: boolean = packageBunAdapterManifest.concreteRuntime.runtimeLoad.hasSourceCheckoutFallback;
const packageSubpathShapeCount: number = packageShapeScalarCount([2, 3, 4]);
const packageSubpathCompatibilityAccepted: boolean = packageStepParamsCompatibilityResult({
  kind: "package-spine-type-smoke",
  signature: "package-spine-type-smoke",
}, null).accepted;
const packageKernelPlanOwner: "src/ts/runtime/kernel_plan.ts" = packageKernelPlanManifest.policyOwner;
const packageKernelPlanSignatureText: string = packageKernelPlanSignature(null);
const packageKernelPlanPredicate: Function = packageAcceptsKernelPlan;
const packageKernelPlanRequire: Function = packageRequireKernelPlan;
const packageKernelPlanSignatureMatch: Function = packageMatchesKernelPlanSignature;
const packageTopLevelKernelPlanOwner: "src/ts/runtime/kernel_plan.ts" = packageTopLevelKernelPlanManifest.policyOwner;
const packageTopLevelKernelPlanSignatureText: string = packageTopLevelKernelPlanSignature(null);
const packageTopLevelKernelPlanPredicate: Function = packageTopLevelAcceptsKernelPlan;
const packageTopLevelKernelPlanRequire: Function = packageTopLevelRequireKernelPlan;
const packageTopLevelKernelPlanAssert: Function = packageTopLevelAssertKernelPlan;
const packageTopLevelKernelPlanSignatureMatch: Function = packageTopLevelMatchesKernelPlanSignature;
const packageNativeApiFirstExport: "Tensor" = packageRequiredNativeApiExports[0];
const packageNativeApiTanhExport: "tanh" = packageRequiredNativeApiExports.find((name): name is "tanh" => name === "tanh")!;
const packageNodeTanhHelper: typeof packageNodeTanh = packageNodeTanh;
const packageBunTanhHelper: typeof packageBunTanh = packageBunTanh;
const packageNodeTanhTensor: PackageNodeTensor = packageNodeTanh([0, 1]);
const packageBunTanhTensor: PackageBunTensor = packageBunTanh([0, 1]);
const packageNodeFSoftmax: PackageNodeTensor = packageNodeF.softmax([1, 2], -1);
const packageBunFSoftmax: PackageBunTensor = packageBunF.softmax([1, 2], -1);
const packageNodeGradModeEnabled: boolean = packageNodeGradMode.isGradEnabled();
const packageBunGradModeEnabled: boolean = packageBunGradMode.isGradEnabled();
const packageSubpathLinearFactory: Function = packageCreateLinearModuleClass;
const packageSubpathTrainMode: Function = packageSetModuleTraining;

void [
  shapeCount,
  tokenValue,
  ownsPolicy,
  nativeApiContractSpineIncludesF,
  nativeApiContractSpineIncludesGradMode,
  extension,
  filename,
  missingMessage,
  koffiFallback,
  nativePath,
  bunNativePath,
  linearFactory,
  genericStepCompatibility,
  moduleFacadeFactory,
  moduleCompilerFlatten,
  packageNodeTanhHelper,
  packageBunTanhHelper,
  packageNodeTanhTensor,
  packageBunTanhTensor,
  packageNodeFSoftmax,
  packageBunFSoftmax,
  packageNodeGradModeEnabled,
  packageBunGradModeEnabled,
  packageSimpleTensorManifest,
  packageSimpleNnManifest,
  packageSimpleCompileManifest,
  packageFrontendShapeCount,
  packageSessionFacadeFactory,
  packageCheckpointFactory,
  packageTraceCompiler,
  packageCompileAnalysisPredicate,
  packageCompileAnalysisRequire,
  packageCompileAnalysisSignatureMatch,
  packageCompileProgramEvidencePredicate,
  packageCompileProgramEvidenceRequire,
  packageCompileProgramEvidenceSignatureMatch,
  packageProgramEvidencePredicate,
  packageProgramEvidenceRequire,
  packageProgramEvidenceSignatureMatch,
  packageInspectionMode,
  packageInspectionProgramPlanPredicate,
  packageInspectionSessionPlanPredicate,
  packageInspectionProgramPlanRequire,
  packageInspectionSessionPlanRequire,
  packageInspectionProgramBindingPlanPredicate,
  packageInspectionModuleBindingPlanPredicate,
  packageInspectionProgramBindingPlanRequire,
  packageInspectionModuleBindingPlanRequire,
  packageModelSourceKind,
  packageLossFactory,
  packageDataFactory,
  packageNnFactory,
  packageNnModuleBindingPlanPredicate,
  packageNnModuleBindingPlanRequire,
  packageNnModuleBindingPlanSignatureMatch,
  packageOptimFactory,
  packageElementwiseAddTensor,
  packageBroadcastAddTensor,
  packageBroadcastWhereTensor,
  packageElementwiseEqTensor,
  packageElementwiseMaximumTensor,
  packageFlattenTensor,
  packageInferReshapeTensor,
  packageInferViewTensor,
  packageFlatten3dTensor,
  packageFlatten3dMiddleTensor,
  packageNarrowRowsTensor,
  packageNarrowColsTensor,
  packageNarrow3dMiddleTensor,
  packageSelect3dTrailingTensor,
  packageSliceColsTensor,
  packageStaticSlice3dTensor,
  packageUnsqueeze3dMiddleTensor,
  packageTranspose3dOuterTensor,
  packageCatDim1Tensor,
  packageCat3dDim2Tensor,
  packageCat4dTrailingTensor,
  packagePermuteTensor,
  packagePermute3dTensor,
  packagePermute4dTensor,
  packageSum3dTrailingTensor,
  packageMean3dDim0Tensor,
  packageSqueezeTensor,
  packageStackDim1Tensor,
  packageStack3dDim2Tensor,
  packageStack4dDim2Tensor,
  packageEinsumMatmulTensor,
  packageEinsumTraceTensor,
  packageEinsumEllipsisTensor,
  packageNodeLinspaceTensor,
  packageNodeArangeTensor,
  packageNodeArangeRangeTensor,
  packageInferredReshapeForwardTensor,
  packageMseLossTensor,
  packageCrossEntropyLossTensor,
  packageNllLossTensor,
  packageOptimizerConfigPredicate,
  packageOptimizerConfigRequire,
  packageOptimizerConfigSignatureMatch,
  packageOptimizerStatePredicate,
  packageOptimizerStateRequire,
  packageOptimizerStateSignatureMatch,
  packageLRSchedulerStatePredicate,
  packageLRSchedulerStateRequire,
  packageLRSchedulerStateSignatureMatch,
  packageTrainFactory,
  packageTrainStepEvidencePredicate,
  packageTrainStepEvidenceRequire,
  packageTrainStepEvidenceSignatureMatch,
  packageTrainFitStepEvidencePredicate,
  packageTrainFitStepEvidenceRequire,
  packageTrainFitStepEvidenceSignatureMatch,
  packageTrainFitEvidencePredicate,
  packageTrainFitEvidenceRequire,
  packageTrainFitEvidenceSignatureMatch,
  packageTrainEvaluateStepEvidencePredicate,
  packageTrainEvaluateStepEvidenceRequire,
  packageTrainEvaluateStepEvidenceSignatureMatch,
  packageTrainEvaluateEvidencePredicate,
  packageTrainEvaluateEvidenceRequire,
  packageTrainEvaluateEvidenceSignatureMatch,
  packageTrainPredictStepEvidencePredicate,
  packageTrainPredictStepEvidenceRequire,
  packageTrainPredictStepEvidenceSignatureMatch,
  packageTrainPredictEvidencePredicate,
  packageTrainPredictEvidenceRequire,
  packageTrainPredictEvidenceSignatureMatch,
  packageProgramFactory,
  packageProgramExecutionPlanPredicate,
  packageProgramExecutionPlanRequire,
  packageProgramExecutionPlanSignatureMatch,
  packageProgramBindingPlanPredicate,
  packageProgramBindingPlanRequire,
  packageProgramBindingPlanSignatureMatch,
  packageTypedProgramBindings,
  packageTypedProgramBindingsInputShape,
  packageTypedProgramBindingsOutputShape,
  packageTypedProgramInputBinding,
  packageTypedProgramOutputBinding,
  packageTypedSessionStepParams,
  packageTypedStepParamsSubpathStepParams,
  packageSessionFactory,
  packageSessionExecutionPlanPredicate,
  packageSessionExecutionPlanRequire,
  packageSessionExecutionPlanSignatureMatch,
  packageStepParamsCompatibilityPredicate,
  packageStepParamsCompatibilityRequire,
  packageFriendlyStepParamsAccepted,
  packageNativeBufferWrite,
  packageNativeBufferWriteCompat,
  packageNativeBufferByteRangePredicate,
  packageNativeBufferByteRangeRequire,
  packageNativeBufferByteRangeSignatureMatch,
  packageProgramDeviceBufferImport,
  packageProgramDeviceBufferImportPredicate,
  packageProgramDeviceBufferImportRequire,
  packageProgramDeviceBufferImportSignatureMatch,
  packageNativeBufferDeviceImport,
  packageNativeBufferDeviceImportPredicate,
  packageNativeBufferDeviceImportRequire,
  packageNativeBufferDeviceImportSignatureMatch,
  packageNativeBufferExternalResource,
  packageNativeBufferExternalResourcePredicate,
  packageNativeBufferExternalResourceRequire,
  packageNativeBufferExternalResourceSignatureMatch,
  packageProgramDeviceFactory,
  packageProgramDeviceEvidence,
  packageProgramDeviceInfoPredicate,
  packageProgramDeviceInfoRequire,
  packageProgramDeviceInfoSignatureMatch,
  packageTensorFactory,
  packageNativeFilename,
  packageNodeEvidenceHost,
  packageBunEvidenceHost,
  packageSubpathShapeCount,
  packageSubpathCompatibilityAccepted,
  packageKernelPlanOwner,
  packageKernelPlanSignatureText,
  packageKernelPlanPredicate,
  packageKernelPlanRequire,
  packageKernelPlanSignatureMatch,
  packageTopLevelKernelPlanOwner,
  packageTopLevelKernelPlanSignatureText,
  packageTopLevelKernelPlanPredicate,
  packageTopLevelKernelPlanRequire,
  packageTopLevelKernelPlanAssert,
  packageTopLevelKernelPlanSignatureMatch,
  packageNativeApiFirstExport,
  packageSubpathLinearFactory,
  packageSubpathTrainMode,
];
