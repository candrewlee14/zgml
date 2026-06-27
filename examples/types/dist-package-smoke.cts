import { compile, frontendManifest, nn as distNnModule } from "../../dist/index.cjs";
import { nativeAdapterEvidence as distCreateNativeAdapterEvidence } from "../../dist/adapters/native.cjs";
import { nodeAdapterEvidence as distNodeAdapterEvidence, nodeAdapterManifest as distNodeAdapterManifest } from "../../dist/adapters/node.cjs";
import { bunAdapterEvidence as distBunAdapterEvidence, bunAdapterManifest as distBunAdapterManifest } from "../../dist/adapters/bun.cjs";
import { bunAdapterEvidence } from "../../dist/bun.cjs";
import { checkpointManifest as distCheckpointManifest } from "../../dist/checkpoint.cjs";
import { compileManifest as distCompileManifest } from "../../dist/compile.cjs";
import type {
  CollatedDataLoader as DistCollatedDataLoader,
  DataCollateFn as DistDataCollateFn,
  DefaultCollateOptions as DistDefaultCollateOptions,
  DataLoader as DistDataLoader,
  TensorDataset as DistTensorDataset,
  TensorDatasetSample as DistTensorDatasetSample,
  TensorDatasetBatchShape as DistTensorDatasetBatchShape,
  TensorShapeTail as DistTensorShapeTail,
} from "../../dist/data.cjs";
import { shapeScalarCount as distShapeScalarCount } from "../../dist/core/shape.cjs";
import {
  lossManifest as distLossManifest,
  type CrossEntropyLoss as DistCrossEntropyLoss,
  type MSELoss as DistMSELoss,
  type NLLLoss as DistNLLLoss,
} from "../../dist/loss.cjs";
import { createLinearModuleClass as distCreateLinearModuleClass } from "../../dist/nn/linear_module.cjs";
import {
  nnManifest as distNnManifest,
  type EmbeddingForwardShape as DistEmbeddingForwardShape,
  type ShapeModuleForwardShape as DistShapeModuleForwardShape,
} from "../../dist/nn.cjs";
import { gradMode as distNodeGradMode, nodeAdapterEvidence, Tensor as DistNodeTensor, nn as DistNodeNn } from "../../dist/node.cjs";
import { optimManifest as distOptimManifest } from "../../dist/optim.cjs";
import type {
  LRScheduler as DistLRScheduler,
  LRSchedulerStateDict as DistLRSchedulerStateDict,
  LRSchedulerStateSnapshot as DistLRSchedulerStateSnapshot,
  Optimizer as DistOptimizer,
  OptimizerConfigSnapshot as DistOptimizerConfigSnapshot,
  OptimizerStateDict as DistOptimizerStateDict,
  OptimizerStateKind as DistOptimizerStateKind,
  OptimizerStateSnapshot as DistOptimizerStateSnapshot,
} from "../../dist/optim.cjs";
import type { NativeBuffer as DistNativeBuffer } from "../../dist/native_buffer.cjs";
import {
  programManifest as distProgramManifest,
  type Program as DistProgram,
  type ProgramBindingPlan as DistProgramBindingPlan,
  type ProgramBindings as DistProgramBindings,
  type ProgramInputBinding as DistProgramInputBinding,
  type ProgramOutputBinding as DistProgramOutputBinding,
} from "../../dist/program.cjs";
import {
  acceptsKernelPlan as distTopLevelAcceptsKernelPlan,
  assertKernelPlan as distTopLevelAssertKernelPlan,
  kernelPlanManifest as distTopLevelKernelPlanManifest,
  kernelPlanSignature as distTopLevelKernelPlanSignature,
  matchesKernelPlanSignature as distTopLevelMatchesKernelPlanSignature,
  requireKernelPlan as distTopLevelRequireKernelPlan,
  type KernelPlan as DistTopLevelKernelPlan,
} from "../../dist/kernel_plan.cjs";
import {
  acceptsKernelPlan as distAcceptsKernelPlan,
  kernelPlanManifest as distKernelPlanManifest,
  matchesKernelPlanSignature as distMatchesKernelPlanSignature,
  requireKernelPlan as distRequireKernelPlan,
  type KernelPlan as DistRuntimeKernelPlan,
} from "../../dist/runtime/kernel_plan.cjs";
import {
  formatNativeApiContractSignature as distFormatNativeApiContractSignature,
  nativeApiContractManifest as distNativeApiContractManifest,
  nativeApiContractSignature as distNativeApiContractSignature,
  nativeApiContractSignatureParts as distNativeApiContractSignatureParts,
  nativePackageSpineContractManifest as distNativePackageSpineContractManifest,
  requiredNativeApiExports as distRequiredNativeApiExports,
  requiredNativePackageSpineExports as distRequiredNativePackageSpineExports,
  type NativeApiContractSignatureParts as DistNativeApiContractSignatureParts,
  type NativePackageSpineContractManifest as DistNativePackageSpineContractManifest,
} from "../../dist/runtime/native_api_contract.cjs";
import { stepParamsCompatibilityResult as distStepParamsCompatibilityResult } from "../../dist/runtime/step_params.cjs";
import {
  sessionManifest as distSessionManifest,
  type ProgramInputBinding as DistSessionProgramInputBinding,
  type ProgramOutputBinding as DistSessionProgramOutputBinding,
  type Session as DistSession,
  type SessionExecuteIntoParams as DistSessionExecuteIntoParams,
  type SessionExecuteParams as DistSessionExecuteParams,
  type SessionExecuteTensorParams as DistSessionExecuteTensorParams,
  type SessionStepParams as DistSessionStepParams,
} from "../../dist/session.cjs";
import {
  stepParamsManifest as distStepParamsManifest,
  type ProgramInputBinding as DistStepParamsProgramInputBinding,
  type ProgramOutputBinding as DistStepParamsProgramOutputBinding,
  type SessionStepParams as DistStepParamsSessionStepParams,
} from "../../dist/step_params.cjs";
import {
  tensorManifest as distTensorManifest,
  type ArangeRangeShape as DistArangeRangeShape,
  type ArangeShape as DistArangeShape,
  type BroadcastShape as DistBroadcastShape,
  type FlattenShape as DistFlattenShape,
  type MatmulShape as DistMatmulShape,
  type NarrowShape as DistNarrowShape,
  type PermuteShape as DistPermuteShape,
  type ReductionShape as DistReductionShape,
  type ReshapeShape as DistReshapeShape,
  type SelectShape as DistSelectShape,
  type SliceShape as DistSliceShape,
  type SqueezeShape as DistSqueezeShape,
  type Tensor as DistTensor,
  type TensorCatShape as DistTensorCatShape,
  type TensorShapeTuple as DistTensorShapeTuple,
  type TensorStackShape as DistTensorStackShape,
  type TransposeShape as DistTransposeShape,
  type UnsqueezeShape as DistUnsqueezeShape,
  type WhereShape as DistWhereShape,
} from "../../dist/tensor.cjs";
import { setModuleTraining as distSetModuleTraining } from "../../dist/train/module_mode.cjs";
import type {
  TrainFitEvidence as DistTrainFitEvidence,
  TrainFitStepEvidence as DistTrainFitStepEvidence,
  TrainStepEvidence as DistTrainStepEvidence,
} from "../../dist/train.cjs";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;

type DistFrontendSource = Expect<Equal<typeof frontendManifest.source, "ts">>;
type DistFrontendProductLanguage = Expect<Equal<typeof frontendManifest.productLanguage, "typescript">>;
type DistFrontendProductSource = Expect<Equal<typeof frontendManifest.productSource, "src/ts/**">>;
type DistFrontendProductSourceOfTruth = Expect<Equal<typeof frontendManifest.productSourceOfTruth, "ts-api-zig-core">>;
type DistFrontendProductSemanticsOwner = Expect<Equal<typeof frontendManifest.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistFrontendFanout = Expect<Equal<typeof frontendManifest.packageFanout, "tsdown">>;
type DistFrontendNativeRole = Expect<Equal<typeof frontendManifest.nativeRole, "core-kernel-runtime">>;
type DistFrontendNativeAlignment = Expect<Equal<typeof frontendManifest.nativeAlignment, "zig-core-contract-tested">>;
type DistFrontendNativeProductPolicy = Expect<Equal<typeof frontendManifest.nativeProductPolicy, "required-core">>;
type DistFrontendSync = Expect<Equal<typeof frontendManifest.frontendSync, "none">>;
type DistFrontendHandwrittenMirrors = Expect<Equal<typeof frontendManifest.handwrittenFrontendMirrors, false>>;
type DistFrontendNativeContractBoundary = Expect<Equal<typeof frontendManifest.nativeContractBoundary, "JS/TS API -> Zig C ABI -> Program/Session kernels">>;
type DistFrontendEagerHotPathCore = Expect<Equal<typeof frontendManifest.eagerHotPathCore, "zig-native-eager-when-profitable">>;
type DistFrontendInferenceHotPathCore = Expect<Equal<typeof frontendManifest.inferenceHotPathCore, "zig-program-session-required">>;
type DistFrontendTrainingHotPathCore = Expect<Equal<typeof frontendManifest.trainingHotPathCore, "zig-ffi-compiled-step-when-supported">>;
type DistFrontendUnsupportedHotPathPolicy = Expect<Equal<typeof frontendManifest.unsupportedHotPathPolicy, "explicit-evidence-no-silent-performance-claim">>;
type DistCompilePath = Expect<Equal<typeof compile.compileManifest.runtimePath, "Trace -> TensorProgramIr -> KernelPlan -> Program">>;
type DistNodeFrontendSource = Expect<Equal<typeof nodeAdapterEvidence.frontendSource, "ts">>;
type DistBunFrontendSource = Expect<Equal<typeof bunAdapterEvidence.frontendSource, "ts">>;
type DistNodeFrontendProductOwner = Expect<Equal<typeof nodeAdapterEvidence.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistBunFrontendNativeAlignment = Expect<Equal<typeof bunAdapterEvidence.nativeAlignment, "zig-core-contract-tested">>;
type DistCheckpointSource = Expect<Equal<typeof distCheckpointManifest.source, "ts">>;
type DistCompileSource = Expect<Equal<typeof distCompileManifest.source, "ts">>;
type DistTensorDatasetRowShape = Expect<Equal<DistTensorShapeTail<readonly [2, 2]>, readonly [2]>>;
type DistTensorDatasetLoaderShape = Expect<Equal<DistTensorDatasetBatchShape<readonly [2, 2]>, readonly [number, 2]>>;
type DistTensorDatasetSampleInput = Expect<Equal<ReturnType<DistTensorDataset<DistTensor<readonly [2]>, DistTensor<readonly [1]>, DistTensor<readonly [number, 2]>, DistTensor<readonly [number, 1]>>["sample"]>["input"], DistTensor<readonly [2]>>>;
type DistTensorDatasetBatchInput = Expect<Equal<ReturnType<DistTensorDataset<DistTensor<readonly [2]>, DistTensor<readonly [1]>, DistTensor<readonly [number, 2]>, DistTensor<readonly [number, 1]>>["batch"]>["input"], DistTensor<readonly [number, 2]>>>;
type DistDataLoaderBatchInput = Expect<Equal<DistDataLoader<DistTensor<readonly [number, 2]>, DistTensor<readonly [number, 1]>> extends Iterable<infer Batch extends { input: unknown }> ? Batch["input"] : never, DistTensor<readonly [number, 2]>>>;
type DistDefaultCollateOptionsShape = Expect<Equal<DistDefaultCollateOptions["indices"], readonly number[] | undefined>>;
type DistCustomCollateBatch = Readonly<{ kind: "dist-custom-collate"; input: DistTensor<readonly [2]> }>;
type DistCollatedDataLoaderBatch = Expect<Equal<ReturnType<DistCollatedDataLoader<DistCustomCollateBatch>["get"]>, DistCustomCollateBatch>>;
type DistDataCollateFnBatch = Expect<Equal<ReturnType<DistDataCollateFn<DistTensorDatasetSample<DistTensor<readonly [2]>>, DistCustomCollateBatch>>, DistCustomCollateBatch>>;
type DistAdamWOptimizerKind = Expect<Equal<DistOptimizer<"adamw">["kind"], "adamw">>;
type DistAdamWOptimizerConfigKind = Expect<Equal<ReturnType<DistOptimizer<"adamw">["config"]>["kind"], "adamw">>;
type DistAdamWStateKind = Expect<Equal<DistOptimizerStateSnapshot<"adamw">["kind"], "adamw">>;
type DistAdamWConfigKind = Expect<Equal<DistOptimizerConfigSnapshot<"adamw">["kind"], "adamw">>;
type DistRMSpropOptimizerKind = Expect<Equal<DistOptimizer<"rmsprop">["kind"], "rmsprop">>;
type DistRMSpropOptimizerConfigKind = Expect<Equal<ReturnType<DistOptimizer<"rmsprop">["config"]>["kind"], "rmsprop">>;
type DistRMSpropStateKind = Expect<Equal<DistOptimizerStateSnapshot<"rmsprop">["kind"], "rmsprop">>;
type DistRMSpropConfigKind = Expect<Equal<DistOptimizerConfigSnapshot<"rmsprop">["kind"], "rmsprop">>;
type DistAdagradOptimizerKind = Expect<Equal<DistOptimizer<"adagrad">["kind"], "adagrad">>;
type DistAdagradOptimizerConfigKind = Expect<Equal<ReturnType<DistOptimizer<"adagrad">["config"]>["kind"], "adagrad">>;
type DistAdagradOptimizerStateDictKind = Expect<Equal<DistOptimizerStateDict<"adagrad">["kind"], "adagrad" | undefined>>;
type DistAdagradStateKind = Expect<Equal<DistOptimizerStateSnapshot<"adagrad">["kind"], "adagrad">>;
type DistAdagradConfigKind = Expect<Equal<DistOptimizerConfigSnapshot<"adagrad">["kind"], "adagrad">>;
type DistOptimizerStateKindShape = Expect<Equal<DistOptimizerStateKind, "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad">>;
type DistStepAdamSchedulerStateKind = Expect<Equal<ReturnType<DistLRScheduler<"step-lr", "adam">["stateDict"]>["kind"], "step-lr">>;
type DistStepAdamSchedulerOptimizerKind = Expect<Equal<ReturnType<DistLRScheduler<"step-lr", "adam">["stateDict"]>["optimizerKind"], "adam">>;
type DistStepSchedulerStateDictKind = Expect<Equal<DistLRSchedulerStateDict<"step-lr">["kind"], "step-lr">>;
type DistExponentialAdamSchedulerStateKind = Expect<Equal<DistLRSchedulerStateSnapshot<"exponential-lr", "adam">["kind"], "exponential-lr">>;
type DistAdamTrainStepEvidenceKind = Expect<Equal<DistTrainStepEvidence<"adam">["optimizerKind"], "adam">>;
type DistAdamTrainFitStepEvidenceKind = Expect<Equal<NonNullable<DistTrainFitStepEvidence<"adam">["stepEvidence"]>["optimizerKind"], "adam">>;
type DistAdamTrainFitEvidenceKind = Expect<Equal<NonNullable<DistTrainFitEvidence<"adam">["lastStep"]>["optimizerKind"], "adam">>;
type DistFlattenShapeCheck = Expect<Equal<DistFlattenShape<readonly [2, 3]>, readonly [6]>>;
type DistFlatten3dShape = Expect<Equal<DistFlattenShape<readonly [2, 3, 4]>, readonly [24]>>;
type DistFlatten3dMiddleShape = Expect<Equal<DistFlattenShape<readonly [2, 3, 4], 1, -1>, readonly [2, 12]>>;
type DistFlatten4dMiddleShape = Expect<Equal<DistFlattenShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 12, 5]>>;
type DistNarrowRowsShape = Expect<Equal<DistNarrowShape<readonly [2, 3], 0, 1>, readonly [1, 3]>>;
type DistNarrowColsShape = Expect<Equal<DistNarrowShape<readonly [2, 3], -1, 2>, readonly [2, 2]>>;
type DistNarrow3dMiddleShape = Expect<Equal<DistNarrowShape<readonly [2, 3, 4], 1, 2>, readonly [2, 2, 4]>>;
type DistSelect3dTrailingShape = Expect<Equal<DistSelectShape<readonly [2, 3, 4], -1>, readonly [2, 3]>>;
type DistSelect4dMiddleShape = Expect<Equal<DistSelectShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 5]>>;
type DistPermuteShapeCheck = Expect<Equal<DistPermuteShape<readonly [2, 3], readonly [1, 0]>, readonly [3, 2]>>;
type DistPermute3dShapeCheck = Expect<Equal<DistPermuteShape<readonly [2, 3, 4], readonly [2, 0, 1]>, readonly [4, 2, 3]>>;
type DistPermute4dShapeCheck = Expect<Equal<DistPermuteShape<readonly [2, 3, 4, 5], readonly [0, -1, 1, -2]>, readonly [2, 5, 3, 4]>>;
type DistReduction3dTrailingShape = Expect<Equal<DistReductionShape<readonly [2, 3, 4], -1>, readonly [2, 3, 1]>>;
type DistReduction4dDim2Shape = Expect<Equal<DistReductionShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 1, 5]>>;
type DistMatrixMatmulShape = Expect<Equal<DistMatmulShape<readonly [2, 3], readonly [3, 2]>, readonly [2, 2]>>;
type DistVectorMatrixMatmulShape = Expect<Equal<DistMatmulShape<readonly [3], readonly [3, 2]>, DistTensorShapeTuple>>;
type DistBroadcast3dShape = Expect<Equal<DistBroadcastShape<readonly [2, 3, 4], readonly [4]>, readonly [2, 3, 4]>>;
type DistBroadcast4dShape = Expect<Equal<DistBroadcastShape<readonly [2, 3, 4, 5], readonly [1, 5]>, readonly [2, 3, 4, 5]>>;
type DistWhereBroadcastShape = Expect<Equal<DistWhereShape<readonly [2, 3], readonly [1, 3], readonly [1]>, readonly [2, 3]>>;
type DistReshapeInferShape = Expect<Equal<DistReshapeShape<readonly [2, 3], readonly [3, -1]>, readonly [3, 2]>>;
type DistReshapeInfer4dShape = Expect<Equal<DistReshapeShape<readonly [2, 3, 4, 5], readonly [2, -1, 5]>, readonly [2, 12, 5]>>;
type DistArangeShapeCheck = Expect<Equal<DistArangeShape<4>, readonly [4]>>;
type DistArangeRangeShapeCheck = Expect<Equal<DistArangeRangeShape<2, 5>, readonly [3]>>;
type DistSliceShapeCheck = Expect<Equal<DistSliceShape<readonly [2, 3], 1, 1, 3>, readonly [2, 2]>>;
type DistSqueezeShapeCheck = Expect<Equal<DistSqueezeShape<readonly [1, 2, 3], 0>, readonly [2, 3]>>;
type DistSqueeze4dMiddleShape = Expect<Equal<DistSqueezeShape<readonly [2, 1, 3, 4], 1>, readonly [2, 3, 4]>>;
type DistTranspose3dOuterShape = Expect<Equal<DistTransposeShape<readonly [2, 3, 4], 0, -1>, readonly [4, 3, 2]>>;
type DistTranspose4dMiddleShape = Expect<Equal<DistTransposeShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 4, 3, 5]>>;
type DistUnsqueeze3dMiddleShape = Expect<Equal<DistUnsqueezeShape<readonly [2, 3, 4], -2>, readonly [2, 3, 1, 4]>>;
type DistUnsqueeze4dMiddleShape = Expect<Equal<DistUnsqueezeShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 1, 4, 5]>>;
type DistCatDim1Shape = Expect<Equal<DistTensorCatShape<readonly [typeof distElementwiseInputTensor, typeof distElementwiseOtherTensor], 1>, readonly [2, 6]>>;
type DistCat3dDim2Shape = Expect<Equal<DistTensorCatShape<readonly [typeof distTensor3d, typeof distTensor3d], 2>, readonly [2, 3, 8]>>;
type DistCat4dTrailingShape = Expect<Equal<DistTensorCatShape<readonly [typeof distTensor4d, typeof distTensor4d], -1>, readonly [2, 3, 4, 10]>>;
type DistStackDim1Shape = Expect<Equal<DistTensorStackShape<readonly [typeof distElementwiseInputTensor, typeof distElementwiseOtherTensor], 1>, readonly [2, 2, 3]>>;
type DistStack3dDim2Shape = Expect<Equal<DistTensorStackShape<readonly [typeof distTensor3d, typeof distTensor3d], 2>, readonly [2, 3, 2, 4]>>;
type DistStack4dDim2Shape = Expect<Equal<DistTensorStackShape<readonly [typeof distTensor4d, typeof distTensor4d], 2>, readonly [2, 3, 2, 4, 5]>>;
type DistEmbeddingGridShape = Expect<Equal<DistEmbeddingForwardShape<readonly [2, 3], 4>, readonly [2, 3, 4]>>;
type DistInferredShapeModuleForwardShape = Expect<Equal<DistShapeModuleForwardShape<"reshape", readonly [3, -1], readonly [2, 3]>, readonly [3, 2]>>;
type DistLossSource = Expect<Equal<typeof distLossManifest.source, "ts">>;
type DistOptimSource = Expect<Equal<typeof distOptimManifest.source, "ts">>;
type DistTensorSource = Expect<Equal<typeof distTensorManifest.source, "ts">>;
type DistNnSource = Expect<Equal<typeof distNnManifest.source, "ts">>;
type DistLossProductSourceOfTruth = Expect<Equal<typeof distLossManifest.productSourceOfTruth, "ts-api-zig-core">>;
type DistOptimNativeProductPolicy = Expect<Equal<typeof distOptimManifest.nativeProductPolicy, "required-core">>;
type DistTensorNoMirrors = Expect<Equal<typeof distTensorManifest.handwrittenFrontendMirrors, false>>;
type DistNnProductOwner = Expect<Equal<typeof distNnManifest.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistProgramPath = Expect<Equal<typeof distProgramManifest.runtimePath, "Program -> Session -> StepParams">>;
type DistSessionPath = Expect<Equal<typeof distSessionManifest.runtimePath, "Program -> Session -> StepParams">>;
type DistStepParamsPath = Expect<Equal<typeof distStepParamsManifest.runtimePath, "Program -> Session -> StepParams">>;
type DistProgramProductSourceOfTruth = Expect<Equal<typeof distProgramManifest.productSourceOfTruth, "ts-api-zig-core">>;
type DistSessionNativeProductPolicy = Expect<Equal<typeof distSessionManifest.nativeProductPolicy, "required-core">>;
type DistStepParamsNoMirrors = Expect<Equal<typeof distStepParamsManifest.handwrittenFrontendMirrors, false>>;
type DistProgramBindingsInput = Expect<Equal<NonNullable<DistProgramBindings<readonly [2], readonly [3]>["input"]>, DistProgramInputBinding<readonly [2]> | DistNativeBuffer>>;
type DistProgramBindingsInputShape = Expect<Equal<DistProgramBindings<readonly [2], readonly [3]>["inputShape"], readonly [2] | undefined>>;
type DistProgramBindingsOutputShape = Expect<Equal<DistProgramBindings<readonly [2], readonly [3]>["outputShape"], readonly [3] | undefined>>;
type DistProgramBindingsOutput = Expect<Equal<NonNullable<DistProgramBindings<readonly [2], readonly [3]>["output"]>, DistProgramOutputBinding<readonly [3]>>>;
type DistProgramBindingPlanInputShape = Expect<Equal<DistProgramBindingPlan<readonly [2], readonly [3]>["inputShape"], readonly [2] | null>>;
type DistProgramBindingPlanOutputShape = Expect<Equal<DistProgramBindingPlan<readonly [2], readonly [3]>["outputShape"], readonly [3] | null>>;
type DistProgramOutputShapeMethod = Expect<Equal<DistProgram<readonly [2], readonly [3]>["outputShape"], () => readonly [3]>>;
type DistSessionOutputShapeMethod = Expect<Equal<DistSession<readonly [2], readonly [3]>["outputShape"], () => readonly [3]>>;
type DistSessionStepParamsInput = Expect<Equal<NonNullable<DistSessionStepParams<readonly [2], readonly [3]>["input"]>, DistSessionProgramInputBinding<readonly [2]>>>;
type DistSessionStepParamsOutput = Expect<Equal<NonNullable<DistSessionStepParams<readonly [2], readonly [3]>["output"]>, DistSessionProgramOutputBinding<readonly [3]> | false>>;
type DistSessionExecuteParamsInput = Expect<Equal<NonNullable<DistSessionExecuteParams<readonly [2], readonly [3]>["input"]>, DistSessionProgramInputBinding<readonly [2]>>>;
type DistSessionExecuteTensorParamsOutput = Expect<NonNullable<DistSessionExecuteTensorParams<readonly [2], readonly [3]>["output"]> extends DistTensor<readonly [3]> | Float32Array ? true : false>;
type DistSessionExecuteIntoParamsInput = Expect<Equal<NonNullable<DistSessionExecuteIntoParams<readonly [2]>["input"]>, DistSessionProgramInputBinding<readonly [2]>>>;
type DistStepParamsSessionInput = Expect<Equal<NonNullable<DistStepParamsSessionStepParams<readonly [2], readonly [3]>["input"]>, DistStepParamsProgramInputBinding<readonly [2]>>>;
type DistStepParamsSessionOutput = Expect<Equal<NonNullable<DistStepParamsSessionStepParams<readonly [2], readonly [3]>["output"]>, DistStepParamsProgramOutputBinding<readonly [3]> | false>>;
type DistNodeAdapterSource = Expect<Equal<typeof distNodeAdapterEvidence.frontendSource, "ts">>;
type DistBunAdapterSource = Expect<Equal<typeof distBunAdapterEvidence.frontendSource, "ts">>;
type DistNodeAdapterProductOwner = Expect<Equal<typeof distNodeAdapterEvidence.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistBunAdapterNativeAlignment = Expect<Equal<typeof distBunAdapterEvidence.nativeAlignment, "zig-core-contract-tested">>;
type DistNodeAdapterManifestOwner = Expect<Equal<typeof distNodeAdapterManifest.policyOwner, "src/ts/adapters/node.ts">>;
type DistBunAdapterManifestOwner = Expect<Equal<typeof distBunAdapterManifest.policyOwner, "src/ts/adapters/bun.ts">>;
type DistNodeAdapterManifestRuntimeKind = Expect<Equal<typeof distNodeAdapterManifest.concreteRuntime.runtimeLoad.runtimeKind, "node-ffi-runtime">>;
type DistBunAdapterManifestRuntimeKind = Expect<Equal<typeof distBunAdapterManifest.concreteRuntime.runtimeLoad.runtimeKind, "bun-ffi-runtime">>;
type DistNodeAdapterManifestLoaderAlignment = Expect<Equal<typeof distNodeAdapterManifest.concreteRuntime.loader.nativeAlignment, "zig-core-contract-tested">>;
type DistBunAdapterManifestLoaderProductOwner = Expect<Equal<typeof distBunAdapterManifest.concreteRuntime.loader.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistNodeAdapterManifestNoMirrors = Expect<Equal<typeof distNodeAdapterManifest.concreteRuntime.loader.handwrittenFrontendMirrors, false>>;
type DistKernelPlanOwner = Expect<Equal<typeof distKernelPlanManifest.policyOwner, "src/ts/runtime/kernel_plan.ts">>;
type DistTopLevelKernelPlanOwner = Expect<Equal<typeof distTopLevelKernelPlanManifest.policyOwner, "src/ts/runtime/kernel_plan.ts">>;
type DistTopLevelKernelPlanShape = Expect<Equal<DistTopLevelKernelPlan, DistRuntimeKernelPlan>>;
type DistNativeApiContractOwner = Expect<Equal<typeof distNativeApiContractManifest.policyOwner, "src/ts/runtime/native_api_contract.ts">>;
type DistNativeApiContractProductOwner = Expect<Equal<typeof distNativeApiContractManifest.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type DistNativeApiContractNativeAlignment = Expect<Equal<typeof distNativeApiContractManifest.nativeAlignment, "zig-core-contract-tested">>;
type DistNativeApiContractProductSourceOfTruth = Expect<Equal<typeof distNativeApiContractManifest.productSourceOfTruth, "ts-api-zig-core">>;
type DistNativeApiContractNativeProductPolicy = Expect<Equal<typeof distNativeApiContractManifest.nativeProductPolicy, "required-core">>;
type DistNativeApiContractNoMirrors = Expect<Equal<typeof distNativeApiContractManifest.handwrittenFrontendMirrors, false>>;
type DistNativeApiContractPackageSpine = Expect<Equal<typeof distNativeApiContractManifest.packageSpine, DistNativePackageSpineContractManifest>>;
type DistNativePackageSpineSource = Expect<Equal<typeof distNativePackageSpineContractManifest.source, "generated-ts">>;
type DistNativePackageSpineGenerator = Expect<Equal<typeof distNativePackageSpineContractManifest.generator, "scripts/generate_native_runtime_wrappers.cjs">>;
type DistNativePackageSpineGeneratedFrom = Expect<Equal<typeof distNativePackageSpineContractManifest.generatedFrom, "src/ts/runtime/native_api_contract.ts">>;
type DistNativePackageSpineNoHandwrittenRootExports = Expect<Equal<typeof distNativePackageSpineContractManifest.handwrittenRootExportList, false>>;

declare const distElementwiseInputTensor: DistTensor<readonly [2, 3]>;
declare const distElementwiseOtherTensor: DistTensor<readonly [2, 3]>;
declare const distTensor3d: DistTensor<readonly [2, 3, 4]>;
declare const distTensor4d: DistTensor<readonly [2, 3, 4, 5]>;
const distElementwiseAddTensor: DistTensor<readonly [2, 3]> = distElementwiseInputTensor.add(distElementwiseOtherTensor);
const distBroadcastAddTensor: DistTensor<readonly [2, 3]> = distElementwiseInputTensor.add(DistNodeTensor.tensor([1, 2, 3], [1, 3] as const));
const distBroadcastWhereTensor: DistTensor<readonly [2, 3]> = distElementwiseInputTensor.gt(0).where(DistNodeTensor.tensor([1, 2, 3], [1, 3] as const), DistNodeTensor.tensor([0], [1] as const));
const distElementwiseEqTensor: DistTensor<readonly [2, 3]> = distElementwiseInputTensor.eq(distElementwiseOtherTensor);
const distElementwiseMaximumTensor: DistTensor<readonly [2, 3]> = distElementwiseInputTensor.maximum(distElementwiseOtherTensor);
const distFlattenTensor: DistTensor<readonly [6]> = distElementwiseInputTensor.flatten();
const distInferReshapeTensor: DistTensor<readonly [3, 2]> = distElementwiseInputTensor.reshape([3, -1] as const);
const distInferViewTensor: DistTensor<readonly [1, 6]> = DistNodeTensor.view(distElementwiseInputTensor, [1, -1] as const);
const distFlatten3dTensor: DistTensor<readonly [24]> = distTensor3d.flatten();
const distFlatten3dMiddleTensor: DistTensor<readonly [2, 12]> = distTensor3d.flatten(1, -1);
const distNarrowRowsTensor: DistTensor<readonly [1, 3]> = distElementwiseInputTensor.narrow(0, 0, 1);
const distNarrowColsTensor: DistTensor<readonly [2, 2]> = distElementwiseInputTensor.narrow(-1, 0, 2);
const distNarrow3dMiddleTensor: DistTensor<readonly [2, 2, 4]> = distTensor3d.narrow(1, 0, 2);
const distSelect3dTrailingTensor: DistTensor<readonly [2, 3]> = distTensor3d.select(-1, 2);
const distSliceColsTensor: DistTensor<readonly [2, 2]> = distElementwiseInputTensor.slice(1, 1, 3);
const distStaticSlice3dTensor: DistTensor<readonly [2, 2, 4]> = DistNodeTensor.slice(distTensor3d, 1, 0, 2);
const distUnsqueeze3dMiddleTensor: DistTensor<readonly [2, 3, 1, 4]> = distTensor3d.unsqueeze(-2);
const distTranspose3dOuterTensor: DistTensor<readonly [4, 3, 2]> = distTensor3d.transpose(0, -1);
const distCatDim1Tensor: DistTensor<readonly [2, 6]> = DistNodeTensor.cat([distElementwiseInputTensor, distElementwiseOtherTensor] as const, 1);
const distCat3dDim2Tensor: DistTensor<readonly [2, 3, 8]> = DistNodeTensor.cat([distTensor3d, distTensor3d] as const, 2);
const distCat4dTrailingTensor: DistTensor<readonly [2, 3, 4, 10]> = DistNodeTensor.cat([distTensor4d, distTensor4d] as const, -1);
const distPermuteTensor: DistTensor<readonly [3, 2]> = distElementwiseInputTensor.permute([1, 0] as const);
const distPermute3dTensor: DistTensor<readonly [4, 2, 3]> = distTensor3d.permute([2, 0, 1] as const);
const distPermute4dTensor: DistTensor<readonly [2, 5, 3, 4]> = distTensor4d.permute([0, -1, 1, -2] as const);
const distSum3dTrailingTensor: DistTensor<readonly [2, 3, 1]> = distTensor3d.sumDim(-1);
const distMean3dDim0Tensor: DistTensor<readonly [1, 3, 4]> = DistNodeTensor.mean(distTensor3d, 0);
declare const distSqueezeInputTensor: DistTensor<readonly [1, 2, 3]>;
const distSqueezeTensor: DistTensor<readonly [2, 3]> = distSqueezeInputTensor.squeeze(0);
const distStackDim1Tensor: DistTensor<readonly [2, 2, 3]> = DistNodeTensor.stack([distElementwiseInputTensor, distElementwiseOtherTensor] as const, 1);
const distStack3dDim2Tensor: DistTensor<readonly [2, 3, 2, 4]> = DistNodeTensor.stack([distTensor3d, distTensor3d] as const, 2);
const distStack4dDim2Tensor: DistTensor<readonly [2, 3, 2, 4, 5]> = DistNodeTensor.stack([distTensor4d, distTensor4d] as const, 2);
const distNodeLinspaceTensor: DistTensor<readonly [4]> = DistNodeTensor.linspace(0, 1, 4);
const distNodeArangeTensor: DistTensor<readonly [4]> = DistNodeTensor.arange(4);
const distNodeArangeRangeTensor: DistTensor<readonly [3]> = DistNodeTensor.arange(2, 5);
const distInferredReshapeForwardTensor: DistTensor<readonly [3, 2]> = DistNodeNn.reshape([3, -1] as const).forward(distElementwiseInputTensor);
declare const distMseLoss: DistMSELoss;
declare const distCrossEntropyLoss: DistCrossEntropyLoss;
declare const distNllLoss: DistNLLLoss;
declare const distLossTensor: DistTensor<readonly [2]>;
declare const distLogitTensor: DistTensor<readonly [1, 3]>;
declare const distLogProbabilityTensor: DistTensor<readonly [1, 3]>;
const distMseLossTensor: DistTensor<readonly [1]> = distMseLoss.forward(distLossTensor, distLossTensor);
const distCrossEntropyLossTensor: DistTensor<readonly [1]> = distCrossEntropyLoss.forward(distLogitTensor, [2] as const);
const distNllLossTensor: DistTensor<readonly [1]> = distNllLoss.forward(distLogProbabilityTensor, [2] as const);

const sequentialClassFactory: Function = distNnModule.createSequentialModuleClass;
const distNativeAdapterEvidence = distCreateNativeAdapterEvidence("node");
const distNativeAdapterSource: "ts" = distNativeAdapterEvidence.frontendSource;
const distNativeAdapterProductOwner: "src/ts/** + src/**/*.zig" = distNativeAdapterEvidence.productSemanticsOwner;
const distNativeAdapterNativeAlignment: "zig-core-contract-tested" = distNativeAdapterEvidence.nativeAlignment;
const distLinearFactory: Function = distCreateLinearModuleClass;
const distTopLevelKernelPlanOwner: "src/ts/runtime/kernel_plan.ts" = distTopLevelKernelPlanManifest.policyOwner;
const distTopLevelKernelPlanSignatureText: string = distTopLevelKernelPlanSignature(null);
const distKernelPlanPredicate: Function = distAcceptsKernelPlan;
const distKernelPlanRequire: Function = distRequireKernelPlan;
const distKernelPlanSignatureMatch: Function = distMatchesKernelPlanSignature;
const distTopLevelKernelPlanPredicate: Function = distTopLevelAcceptsKernelPlan;
const distTopLevelKernelPlanRequire: Function = distTopLevelRequireKernelPlan;
const distTopLevelKernelPlanAssert: Function = distTopLevelAssertKernelPlan;
const distTopLevelKernelPlanSignatureMatch: Function = distTopLevelMatchesKernelPlanSignature;
const distStepParamsCompatibility: Function = distStepParamsCompatibilityResult;
const distTrainingModeSetter: Function = distSetModuleTraining;
const distShapeCount: number = distShapeScalarCount([2, 3]);
const distRequiredNativeApiFirst: "Tensor" = distRequiredNativeApiExports[0];
const distRequiredNativeApiCount: number = distRequiredNativeApiExports.length;
const distRequiredNativeSpineCount: number = distRequiredNativePackageSpineExports.length;
const distRequiredNativeSpineIncludesF: boolean = distRequiredNativePackageSpineExports.includes("F");
const distRequiredNativeSpineIncludesGradMode: boolean = distRequiredNativePackageSpineExports.includes("gradMode");
const distNativeApiContractParts: DistNativeApiContractSignatureParts = distNativeApiContractSignatureParts();
const distNativeApiContractSignatureText: string = distNativeApiContractSignature();
const distFormattedNativeApiContractSignature: string = distFormatNativeApiContractSignature(distNativeApiContractParts);
const distTypedProgramBindings: DistProgramBindings<readonly [2], readonly [3]> = {
  inputShape: [2] as const,
  outputShape: [3] as const,
};
const distTypedProgramBindingsInputShape: readonly [2] | undefined = distTypedProgramBindings.inputShape;
const distTypedProgramBindingsOutputShape: readonly [3] | undefined = distTypedProgramBindings.outputShape;
declare const distTypedInputTensor: DistTensor<readonly [2]>;
declare const distTypedOutputTensor: DistTensor<readonly [3]>;
const distTypedProgramInputBinding: DistProgramInputBinding<readonly [2]> = distTypedInputTensor;
const distTypedProgramOutputBinding: DistProgramOutputBinding<readonly [3]> = distTypedOutputTensor;
const distTypedSessionStepParams: DistSessionStepParams<readonly [2], readonly [3]> = {
  input: distTypedInputTensor,
  output: distTypedOutputTensor,
};
const distTypedStepParamsSubpathStepParams: DistStepParamsSessionStepParams<readonly [2], readonly [3]> = {
  input: distTypedInputTensor,
  output: distTypedOutputTensor,
};
const distNodeGradModeEnabled: boolean = distNodeGradMode.isGradEnabled();
const nodeHost: string = nodeAdapterEvidence.host;
const bunHost: string = bunAdapterEvidence.host;

void [sequentialClassFactory, distNativeAdapterSource, distLinearFactory, distTopLevelKernelPlanOwner, distTopLevelKernelPlanSignatureText, distKernelPlanPredicate, distKernelPlanRequire, distKernelPlanSignatureMatch, distTopLevelKernelPlanPredicate, distTopLevelKernelPlanRequire, distTopLevelKernelPlanAssert, distTopLevelKernelPlanSignatureMatch, distStepParamsCompatibility, distTrainingModeSetter, distElementwiseAddTensor, distBroadcastAddTensor, distBroadcastWhereTensor, distElementwiseEqTensor, distElementwiseMaximumTensor, distFlattenTensor, distInferReshapeTensor, distInferViewTensor, distFlatten3dTensor, distFlatten3dMiddleTensor, distNarrowRowsTensor, distNarrowColsTensor, distNarrow3dMiddleTensor, distSelect3dTrailingTensor, distSliceColsTensor, distStaticSlice3dTensor, distUnsqueeze3dMiddleTensor, distTranspose3dOuterTensor, distCatDim1Tensor, distCat3dDim2Tensor, distCat4dTrailingTensor, distPermuteTensor, distPermute3dTensor, distPermute4dTensor, distSum3dTrailingTensor, distMean3dDim0Tensor, distSqueezeTensor, distStackDim1Tensor, distStack3dDim2Tensor, distStack4dDim2Tensor, distNodeLinspaceTensor, distNodeArangeTensor, distNodeArangeRangeTensor, distInferredReshapeForwardTensor, distMseLossTensor, distCrossEntropyLossTensor, distNllLossTensor, distShapeCount, distRequiredNativeApiFirst, distRequiredNativeApiCount, distRequiredNativeSpineCount, distRequiredNativeSpineIncludesF, distRequiredNativeSpineIncludesGradMode, distNativeApiContractSignatureText, distFormattedNativeApiContractSignature, distTypedProgramBindingsInputShape, distTypedProgramBindingsOutputShape, distTypedProgramInputBinding, distTypedProgramOutputBinding, distTypedSessionStepParams, distTypedStepParamsSubpathStepParams, distNodeGradModeEnabled, nodeHost, bunHost];
