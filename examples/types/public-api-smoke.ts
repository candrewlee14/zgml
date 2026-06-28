import {
  Tensor,
  NativeBuffer,
  TinyLlama,
  add,
  all,
  any,
  argsort,
  arange,
  argmax,
  bmm,
  checkpoint,
  chunk,
  clamp,
  cat,
  ceil,
  clone,
  compile,
  concat,
  concatenate,
  clip,
  cos,
  cpu,
  cumsum,
  data,
  div,
  dot,
  diagonal,
  enable_grad,
  empty,
  einsum,
  emptyLike,
  empty_like,
  eq,
  expm1,
  eye,
  frontendManifest,
  F,
  float,
  float32,
  flip,
  floor,
  forInference,
  for_inference,
  forTraining,
  for_training,
  gradMode,
  infer,
  inferInto,
  infer_into,
  inference,
  inference_mode,
  fullLike,
  full_like,
  flatten,
  explainNative,
  explain_native,
  fit,
  fitModule,
  fit_module,
  fitNative,
  nativeTrainingPlan,
  native_training_plan,
  gather,
  hasShape,
  hstack,
  indexSelect,
  index_select,
  initialSeed,
  initial_seed,
  inspection,
  is_grad_enabled,
  isclose,
  isfinite,
  isinf,
  isnan,
  linspace,
  load,
  log1p,
  logSoftmax,
  logSumExp,
  log_softmax,
  log_softmax_dim,
  logsumexp,
  lazy,
  loss,
  manual_seed,
  manualSeed,
  matmul,
  mean,
  maximum,
  maskedFill,
  masked_fill,
  minimum,
  mm,
  mul,
  narrow,
  neg,
  negative,
  native,
  nativeCore,
  native_core,
  nativeApiContract,
  no_grad,
  nn,
  norm,
  optim,
  permute,
  pow,
  predict,
  predictInto,
  predict_into,
  prod,
  program,
  randLike,
  rand_like,
  randInt,
  randint,
  randPerm,
  randperm,
  randnLike,
  randn_like,
  reciprocal,
  requireShape,
  relu,
  repeat,
  reshape,
  roll,
  round,
  rsqrt,
  run,
  runInto,
  run_into,
  scatterAdd,
  scatter_add,
  session,
  runtimeInfo,
  set_grad_enabled,
  seededRng,
  simple,
  sin,
  slice,
  sort,
  split,
  softmax,
  softmax_dim,
  save,
  sqrt,
  squeeze,
  stack,
  stepParams,
  sub,
  sum,
  std,
  tan,
  tanh,
  take,
  tensor,
  asTensor,
  as_tensor,
  asarray,
  fromNumpy,
  from_numpy,
  tile,
  to,
  topk,
  trace,
  train,
  transpose,
  torch,
  trunc,
  typeAs,
  type_as,
  variance,
  select,
  unbind,
  unsqueeze,
  vstack,
  where,
  zeros,
  zerosLike,
  zeros_like,
  onesLike,
  ones_like,
  zgml,
  type CheckpointNamespace,
  type CheckpointCreateOptions,
  type CheckpointModuleState,
  type CheckpointOptimizerState,
  type CheckpointInspection,
  type CheckpointRestoreOptions,
  type CheckpointSchedulerState,
  type CheckpointTensorInspection,
  type CompileAnalysis,
  type CompileNamespace,
  type CompiledInference,
  type BatchSampler,
  type BCELoss,
  type BCEWithLogitsLoss,
  type CrossEntropyLoss,
  type NLLLoss,
  type CompilableActivationModule,
  type HuberLoss,
  type InspectionNamespace,
  type ArangeRangeShape,
  type ArangeShape,
  type BmmShape,
  type BroadcastShape,
  type CompileOptionsWithInputShape,
  type CustomModule,
  type Dataset,
  type DataBatches,
  type CollatedDataLoader,
  type DataCollateFn,
  type DataLoader,
  type DataNamespace,
  type DefaultCollateOptions,
  type DataSplitOptions,
  type EmbeddingCompileOptions,
  type EmbeddingForwardShape,
  type EinsumShape,
  type FlattenShape,
  type GradModeNamespace,
  type IndexLike,
  type KaimingNormalOptions,
  type KaimingUniformOptions,
  type LlamaExecuteIntoParams,
  type LlamaExecuteParams,
  type LlamaKvCache,
  type LlamaKvCacheLayout,
  type LlamaKvCacheRequirements,
  type LlamaKvCacheSlot,
  type LlamaProgramInspection,
  type LlamaSessionStepContract,
  type LlamaStepParams,
  type LossNamespace,
  type NativeEagerConv2dIntoOptions,
  type NativeEagerArgReduceDimIntoOptions,
  type NativeEagerCumsumIntoOptions,
  type NativeEagerMatmulIntoOptions,
  type NativeEagerPool2dIntoOptions,
  type NativeEagerReduceDimIntoOptions,
  type NativeEagerRoutingPolicy,
  type PublicLossNamespace,
  type NnLossConstructorName,
  type NnLossConstructors,
  type LRScheduler,
  type LRSchedulerStateDict,
  type LRSchedulerStateSnapshot,
  type L1Loss,
  type LinearForwardShape,
  type LazyCompileSupport,
  type LinearModule,
  type LossReduction,
  type MatmulShape,
  type MSELoss,
  type SmoothL1Loss,
  type TensorDataset,
  type TensorDatasetBatch,
  type TensorDatasetBatchShape,
  type TensorDatasetMapper,
  type TensorDatasetSample,
  type TensorShapeTail,
  type ModuleCompleteCompilerSignatures,
  type ModuleBindings,
  type ModuleBindingPlacementMode,
  type ModuleBindingPlan,
  type ModuleCompileExplanation,
  type ModuleCompileDiagnostic,
  type ModuleCompilerSignatures,
  type ModuleCompileSupport,
  type ModuleForwardShape,
  type ModuleTargetForwardShape,
  type ModuleProgramTrace,
  type ModuleKernelBufferLayout,
  type ModuleKernelMemoryLayout,
  type ModuleKernelParameterLayout,
  type ModuleKernelPlan,
  type ModuleKernelPlanActivationChainOp,
  type ModuleKernelPlanFusedValueEdge,
  type ModuleKernelParameterLayoutEntry,
  type ModuleKernelShapeConstraints,
  type ModuleBase,
  type ModuleBufferTraversalOptions,
  type Conv2dModule,
  type AvgPool2dModule,
  type MaxPool2dModule,
  type ModuleDict,
  type ModuleList,
  type NarrowShape,
  type PermuteShape,
  type RawSequentialLayerListCompileDiagnostic,
  type ModuleParameterInfo,
  type ModuleReductionKind,
  type ModuleParameterTraversalOptions,
  type ModuleStateDictOptions,
  type ModuleTraversalEntry,
  type ModuleStateSnapshot,
  type ModuleTensorProgramIr,
  type NativeApiContractNamespace,
  type NativeApiContractSignatureParts,
  type NativeCoreEvidence,
  type NnFunctionalNamespace,
  type NnNamespace,
  type OneHotShape,
  type NnBuffer,
  type NnConv2dConfig,
  type NnAvgPool2dConfig,
  type NnMaxPool2dConfig,
  type NnModule,
  type NnModuleConfig,
  type NnParameter,
  type NnParameterOptions,
  type OptimNamespace,
  type Optimizer,
  type OptimizerConfigSnapshot,
  type OptimizerParamGroupSnapshot,
  type OptimizerStateSnapshot,
  type Program,
  type ParameterDict,
  type ParameterList,
  type ProgramBindingDiagnostic,
  type ProgramBindingMode,
  type ProgramBindingPlan,
  type ProgramBindings,
  type ProgramBufferSlot,
  type ProgramBufferSizing,
  type ProgramCompileEvidence,
  type ProgramDeviceBufferKind,
  type ProgramNamespace,
  type ProgramOutputBufferSlot,
  type ProgramInspection,
  type PublicApiContractManifest,
  type PublicSimpleNamespace,
  type ProgramBufferLayout,
  type ProgramBufferLayoutSlot,
  type ProgramExecutionCapabilities,
  type ProgramExecutionPlan,
  type ProgramInputBinding,
  type ProgramModelCompatibility,
  type ProgramModuleCompatibility,
  type ProgramOutputBinding,
  type ProgramRequirements,
  type ProgramRuntimeDiagnostic,
  type TensorNativePlacement,
  type RuntimeProfile,
  type RuntimeProfileExpectation,
  type RuntimeFeatures,
  type RandomSampler,
  type ReductionShape,
  type ReshapeShape,
  type Sampler,
  type SelectShape,
  type SliceShape,
  type SequentialModule,
  type SequentialForwardShape,
  type SequentialSampler,
  type SqueezeAllShape,
  type SqueezeShape,
  type Session,
  type SessionBufferSizing,
  type SessionCallProfile,
  type SessionDefaultOutputKind,
  type SessionExecutionPlan,
  type SessionExecuteIntoParams,
  type SessionExecuteParams,
  type SessionExecuteTensorParams,
  type SessionInspection,
  type SessionNoOutputEffect,
  type SessionStepParamsDiagnosticCode,
  type SessionStepParamsElementType,
  type SessionStepParamsHotPathBlocker,
  type SessionStepParamsHotPathStatus,
  type SessionStepParamsOutputEffect,
  type SessionStepParamsOutputOwnership,
  type SessionStepParamsOutputReturnOwnership,
  type SessionStepParamsInputSource,
  type SessionStepParamsInputOwnership,
  type SessionStepParamsOutputTarget,
  type SessionStepParamsStateEffect,
  type SessionStepParamsCompatibility,
  type SessionStepParams,
  type SessionStepContract,
  type SessionNamespace,
  type ShapeModuleForwardShape,
  type StepParamsNamespace,
  type TensorCatShape,
  type TensorJSON,
  type TensorData,
  type TensorGatherShape,
  type TensorHStackShape,
  type TensorIndexSelectShape,
  type TensorLikeShape,
  type TensorNestedArray,
  type TensorShapeTuple,
  type TensorShapeOf,
  type TensorStackShape,
  type TensorSortResult,
  type TensorTopkResult,
  type TensorTopkShape,
  type TensorVStackShape,
  type TransposeShape,
  type UnsqueezeShape,
  type WhereShape,
  type CompiledTrainingPlan,
  type CompiledTrainingStep,
  type NativeTrainingBulkFitPlan,
  type NativeTrainingExplanation,
  type TinyLlamaProgram,
  type TinyLlamaSession,
  type TokenArgmaxResult,
  type TokenGenerateArgmaxResult,
  type TokenGenerateSampleResult,
  type TokenSampleResult,
  type PublicTrainNamespace,
  type TrainClassificationBatch,
  type TrainClassificationReport,
  type TrainClassificationTargetShape,
  type TrainBatchInputShape,
  type TrainBatchTargetShape,
  type TrainEarlyStoppingOptions,
  type TrainEvaluateContext,
  type TrainEvaluateEvidence,
  type TrainEvaluateStepEvidence,
  type TrainPredictContext,
  type TrainPredictEvidence,
  type TrainPredictStepEvidence,
  type TrainFitContext,
  type TrainFitEvidence,
  type TrainFitStepEvidence,
  type TrainModelFitOptions,
  type TrainModuleOutputShape,
  type TrainNativeLossName,
  type TrainSupervisedBatch,
  type TrainStepEvidence,
  type ZeroGradOptions,
  type ZgmlCheckpoint,
} from "zgml";
import type {
  ModuleKernelPlan as BunModuleKernelPlan,
  Program as BunProgram,
  Session as BunSession,
  Tensor as BunTensor,
} from "zgml/bun";
import type {
  Program as ProgramSubpath,
  ProgramBindingPlan as ProgramSubpathBindingPlan,
  ProgramExecutionPlan as ProgramSubpathExecutionPlan,
} from "zgml/program";
import type {
  Session as SessionSubpath,
  SessionExecutionPlan as SessionSubpathExecutionPlan,
  SessionStepParamsCompatibility as SessionSubpathStepParamsCompatibility,
} from "zgml/session";
import type {
  SessionExecutionPlan as StepParamsSubpathExecutionPlan,
  SessionStepParams as StepParamsSubpathParams,
  SessionStepParamsCompatibility as StepParamsSubpathCompatibility,
} from "zgml/step_params";
import type {
  Program as NodeProgram,
  Session as NodeSession,
  Tensor as NodeTensor,
} from "zgml/node";
import {
  shapeScalarCount as coreShapeScalarCount,
  type Shape as CoreShapeSubpath,
} from "zgml/core/shape";
import {
  acceptsKernelPlan as acceptsRuntimeKernelPlan,
  assertKernelPlan as assertRuntimeKernelPlan,
  kernelPlanManifest as runtimeKernelPlanManifest,
  matchesKernelPlanSignature as matchesRuntimeKernelPlanSignature,
  requireKernelPlan as requireRuntimeKernelPlan,
} from "zgml/runtime/kernel_plan";
import { createLinearModuleClass as createLinearModuleClassSubpath } from "zgml/nn/linear_module";
import { setModuleTraining as setModuleTrainingSubpath } from "zgml/train/module_mode";
import {
  requireTrainEvaluateEvidence as requireTrainEvaluateEvidenceSubpath,
  requireTrainEvaluateStepEvidence as requireTrainEvaluateStepEvidenceSubpath,
  requireTrainPredictEvidence as requireTrainPredictEvidenceSubpath,
  requireTrainPredictStepEvidence as requireTrainPredictStepEvidenceSubpath,
  matchesTrainPredictEvidenceSignature as matchesTrainPredictEvidenceSignatureSubpath,
  matchesTrainPredictStepEvidenceSignature as matchesTrainPredictStepEvidenceSignatureSubpath,
  isTrainStepEvidence as isTrainStepEvidenceSubpath,
  requireTrainFitEvidence as requireTrainFitEvidenceSubpath,
  requireTrainFitStepEvidence as requireTrainFitStepEvidenceSubpath,
  requireTrainStepEvidence as requireTrainStepEvidenceSubpath,
} from "zgml/train";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2)
    ? true
    : false;
type Expect<T extends true> = T;

type ShapeFromNumber = Expect<Equal<TensorShapeOf<3>, readonly [3]>>;
type ShapeFromTuple = Expect<Equal<TensorShapeOf<readonly [2, 3]>, readonly [2, 3]>>;
type BunProgramSubpath = Expect<Equal<BunProgram, Program>>;
type BunSessionSubpath = Expect<Equal<BunSession, Session>>;
type BunTensorSubpath = Expect<Equal<BunTensor, Tensor>>;
type BunKernelPlanDescriptorSignatures = Expect<Equal<
  NonNullable<BunModuleKernelPlan["ops"][number]["nativeDescriptorSignatures"]>,
  readonly string[]
>>;
type NodeProgramSubpath = Expect<Equal<NodeProgram, Program>>;
type NodeSessionSubpath = Expect<Equal<NodeSession, Session>>;
type NodeTensorSubpath = Expect<Equal<NodeTensor, Tensor>>;
type ProgramFriendlySubpath = Expect<Equal<ProgramSubpath, Program>>;
type ProgramExecutionPlanFriendlySubpath = Expect<Equal<ProgramSubpathExecutionPlan, ProgramExecutionPlan>>;
type ProgramBindingPlanFriendlySubpath = Expect<Equal<ProgramSubpathBindingPlan, ProgramBindingPlan>>;
type SessionFriendlySubpath = Expect<Equal<SessionSubpath, Session>>;
type SessionExecutionPlanFriendlySubpath = Expect<Equal<SessionSubpathExecutionPlan, SessionExecutionPlan>>;
type SessionCompatibilityFriendlySubpath = Expect<Equal<SessionSubpathStepParamsCompatibility, SessionStepParamsCompatibility>>;
type StepParamsExecutionPlanFriendlySubpath = Expect<Equal<StepParamsSubpathExecutionPlan, SessionExecutionPlan>>;
type StepParamsCompatibilityFriendlySubpath = Expect<Equal<StepParamsSubpathCompatibility, SessionStepParamsCompatibility>>;
type StepParamsFriendlySubpath = Expect<Equal<StepParamsSubpathParams, SessionStepParams>>;
type FrontendManifestSource = Expect<Equal<typeof frontendManifest.source, "ts">>;
type FrontendManifestProductLanguage = Expect<Equal<typeof frontendManifest.productLanguage, "typescript">>;
type FrontendManifestProductSource = Expect<Equal<typeof frontendManifest.productSource, "src/ts/**">>;
type FrontendManifestProductSourceOfTruth = Expect<Equal<typeof frontendManifest.productSourceOfTruth, "ts-api-zig-core">>;
type FrontendManifestProductSemanticsOwner = Expect<Equal<typeof frontendManifest.productSemanticsOwner, "src/ts/** + src/**/*.zig">>;
type FrontendManifestFanout = Expect<Equal<typeof frontendManifest.packageFanout, "tsdown">>;
type FrontendManifestNativeRole = Expect<Equal<typeof frontendManifest.nativeRole, "core-kernel-runtime">>;
type FrontendManifestNativeAlignment = Expect<Equal<typeof frontendManifest.nativeAlignment, "zig-core-contract-tested">>;
type FrontendManifestNativeProductPolicy = Expect<Equal<typeof frontendManifest.nativeProductPolicy, "required-core">>;
type FrontendManifestSync = Expect<Equal<typeof frontendManifest.frontendSync, "none">>;
type FrontendManifestHandwrittenMirrors = Expect<Equal<typeof frontendManifest.handwrittenFrontendMirrors, false>>;
type FrontendManifestNativeContractBoundary = Expect<Equal<typeof frontendManifest.nativeContractBoundary, "JS/TS API -> Zig C ABI -> Program/Session kernels">>;
type FrontendManifestModuleCompilerCore = Expect<Equal<typeof frontendManifest.moduleCompilerCore, "zig-module-program">>;
type FrontendManifestNativeCompileEvidence = Expect<Equal<typeof frontendManifest.nativeCompileEvidence, "native-program-inspection">>;
type FrontendManifestProgramInspectionCore = Expect<Equal<typeof frontendManifest.programInspectionCore, "zig-program-inspection">>;
type FrontendManifestEagerHotPathCore = Expect<Equal<typeof frontendManifest.eagerHotPathCore, "zig-native-eager-when-profitable">>;
type FrontendManifestInferenceHotPathCore = Expect<Equal<typeof frontendManifest.inferenceHotPathCore, "zig-program-session-required">>;
type FrontendManifestTrainingHotPathCore = Expect<Equal<typeof frontendManifest.trainingHotPathCore, "zig-ffi-compiled-step-when-supported">>;
type FrontendManifestUnsupportedHotPathPolicy = Expect<Equal<typeof frontendManifest.unsupportedHotPathPolicy, "explicit-evidence-no-silent-performance-claim">>;

const lazyInput = lazy.input([2] as const);
const lazyGraph = lazyInput.linear(3).relu().linear(1);
const lazyEmbeddingGraph = lazy.input([2] as const).embedding(4, 3).layerNorm(3).rmsNorm(3);
const lazyNamespaceEmbeddingGraph = lazy.rmsNorm(lazy.layerNorm(lazy.embedding(lazy.input([2] as const), 4, 3), 3), 3);
const lazySnakeNamespaceEmbeddingGraph = lazy.rms_norm(lazy.layer_norm(lazy.embedding(lazy.input([2] as const), 4, 3), 3), 3);
const lazyGridEmbeddingGraph = lazy.input([1, 2] as const).embedding(4, 3);
const lazyGridEmbeddingSupport = lazyGridEmbeddingGraph.compileSupport();
const lazyConvPoolGraph = lazy.input([1, 4, 4] as const).conv2d(2, 1).maxPool2d(2).avgPool2d(2);
const lazyNamespaceConvPoolGraph = lazy.avgPool2d(lazy.maxPool2d(lazy.conv2d(lazy.input([1, 4, 4] as const), 2, 1), 2), 2);
const lazySnakeConvPoolGraph = lazy.avg_pool2d(lazy.max_pool2d(lazy.input([1, 4, 4] as const).conv2d(2, 1), 2), 2);
const lazySigmoidGraph = lazyInput.linear(3).sigmoid().linear(1);
const lazyNamespaceSigmoidGraph = lazy.sigmoid(lazyInput.linear(3)).linear(1);
const lazyMatmulWeight = lazy.parameter([2, 3] as const, "head.weight");
const lazyMatmulBias = lazy.parameter([3] as const, "head.bias");
const lazyMatmulScale = lazy.parameter([3] as const, "head.scale");
const lazyMatmulShift = lazy.parameter([3] as const, "head.shift");
const lazyMatmulGraph = lazyInput.matmul(lazyMatmulWeight).relu();
const lazyMatmulBiasGraph = lazyInput.matmul(lazyMatmulWeight).add(lazyMatmulBias).relu();
const lazyMatmulScaleGraph = lazyInput.matmul(lazyMatmulWeight).mul(lazyMatmulScale).relu();
const lazyMatmulAffineGraph = lazyInput.matmul(lazyMatmulWeight).affine(lazyMatmulScale, lazyMatmulShift).relu();
const lazyAddReluGraph = lazy.input([2, 3] as const).add(lazyMatmulShift).relu();
const lazyMulReluGraph = lazy.input([2, 3] as const).mul(lazyMatmulScale).relu();
const lazyAffineReluGraph = lazy.input([2, 3] as const).affine(lazyMatmulScale, lazyMatmulShift).relu();
const lazyNaturalAffineGraph = lazy.input([2, 3] as const).mul(lazyMatmulScale).add(lazyMatmulShift);
const lazyMatmulNaturalAffineGraph = lazyInput.matmul(lazyMatmulWeight).mul(lazyMatmulScale).add(lazyMatmulShift).relu();
const lazyNamespaceMatmulGraph = lazy.relu(lazy.matmul(lazyInput, lazyMatmulWeight));
const lazyNamespaceMatmulBiasGraph = lazy.relu(lazy.add(lazy.matmul(lazyInput, lazyMatmulWeight), lazyMatmulBias));
const lazyNamespaceMatmulScaleGraph = lazy.relu(lazy.mul(lazy.matmul(lazyInput, lazyMatmulWeight), lazyMatmulScale));
const lazyNamespaceMatmulAffineGraph = lazy.relu(lazy.affine(lazy.matmul(lazyInput, lazyMatmulWeight), lazyMatmulScale, lazyMatmulShift));
const lazyNamespaceNaturalAffineGraph = lazy.relu(lazy.add(lazy.mul(lazy.matmul(lazyInput, lazyMatmulWeight), lazyMatmulScale), lazyMatmulShift));
const lazyMmGraph = lazy.input([1, 2] as const).mm(lazyMatmulWeight);
const lazyActivationChain = lazyInput.exp().log().neg().recip().abs().sqrt().square().sgn().step();
const lazyNamespaceActivationChain = lazy.step(lazy.sgn(lazy.square(lazy.sqrt(lazy.abs(lazy.recip(lazy.neg(lazy.log(lazy.exp(lazyInput)))))))));
const lazySnakeSoftmaxGraph = lazy.input([2, 3] as const).log_softmax(1);
const lazyBatchSoftmaxGraph = lazy.input([2, 3] as const).softmax(0);
const lazyBatchLogSoftmaxGraph = lazy.input([2, 3] as const).logSoftmax(0);
const lazyDropoutGraph = lazyInput.dropout(0.5, { training: false }).relu();
const lazyNamespaceDropoutGraph = lazy.dropout(lazyInput, 0.5, { training: false }).relu();
const lazyTrainingDropoutGraph = lazyInput.dropout(0.5, { training: true });
const lazyTrainingDropoutGraphSupport = lazyTrainingDropoutGraph.compileSupport();
const lazyReductionInput = lazy.input([2, 3] as const);
const lazyReductionChain = lazyReductionInput.sum(1).mean(0).max(0).min(0);
const lazyNamespaceReduction = lazy.min(lazy.max(lazy.mean(lazy.sum(lazyReductionInput, 1), 0), 0), 0);
const lazyShapeInput = lazy.input([1, 3] as const);
const lazyShapeChain = lazyShapeInput.squeeze(0).unsqueeze(0).broadcastTo([2, 3] as const).narrow(0, 0, 1).slice(1, 0, 2).transpose(0, 1).permute([1, 0] as const).select(0, 0);
const lazyNamespaceShapeChain = lazy.select(lazy.permute(lazy.transpose(lazy.slice(lazy.narrow(lazy.broadcastTo(lazy.unsqueeze(lazy.squeeze(lazyShapeInput, 0), 0), [2, 3] as const), 0, 0, 1), 1, 0, 2), 0, 1), [1, 0] as const), 0, 0);
const lazySnakeShapeChain = lazyShapeInput.squeeze(0).unsqueeze(0).broadcast_to([2, 3] as const);
const lazyTrace = lazyGraph.trace();
const lazyIr: ModuleTensorProgramIr | null = lazyGraph.tensorProgramIr();
const lazyIrAlias: ModuleTensorProgramIr | null = lazyGraph.tensor_program_ir();
const lazyPlan: ModuleKernelPlan | null = lazyGraph.kernelPlan();
const lazySupport = lazyGraph.compileSupport();
const lazyCanCompile: boolean = lazyGraph.canCompile();
const lazyCanCompileSnake: boolean = lazyGraph.can_compile();
const lazyRequiredSupport = lazyGraph.requireCompileSupport();
const lazyRequiredSupportSnake = lazyGraph.require_compile_support();
const lazyCompiledProgram: Program<TensorShapeTuple, readonly [1]> = compile.compile(lazyGraph, { backend: "cpu" });
const lazyCompiledProgramOutputShape: readonly [1] = lazyCompiledProgram.outputShape();
const lazyMethodCompiledProgram: Program<TensorShapeTuple, readonly [1]> = lazyGraph.compile({ backend: "cpu" });
const lazyMethodCompiledProgramOutputShape: readonly [1] = lazyMethodCompiledProgram.outputShape();
const lazyNamespaceCanCompile: boolean = lazy.canCompile(lazyGraph);
const lazyNamespaceCanCompileSnake: boolean = lazy.can_compile(lazyGraph);
const lazyNamespaceRequiredSupport = lazy.requireCompileSupport(lazyGraph);
const lazyNamespaceRequiredSupportSnake = lazy.require_compile_support(lazyGraph);
const compileLazyTrace: ModuleProgramTrace = compile.trace(lazyGraph);
const compileLazyAnalysis: CompileAnalysis = compile.analyze(lazyGraph);
const compileLazySupport: LazyCompileSupport = compile.compileSupport(lazyGraph);
const compileLazyRequiredSupport: LazyCompileSupport = compile.requireCompileSupport(lazyGraph);
const compileLazyPlan: LazyCompileSupport = compile.requireCompilePlan(lazyGraph);
const compileLazyCanCompile: boolean = compile.canCompile(lazyGraph);
const compileLazyCanCompileSnake: boolean = compile.can_compile(lazyGraph);
const compileLazyIr: ModuleTensorProgramIr | null = compile.tensorProgramIr(lazyGraph);
const compileLazyIrSnake: ModuleTensorProgramIr | null = compile.tensor_program_ir(lazyGraph);
const compileLazyKernelPlan: ModuleKernelPlan | null = compile.kernelPlan(lazyGraph);
const compileLazyKernelPlanSnake: ModuleKernelPlan | null = compile.kernel_plan(lazyGraph);
const compileLazyInputShape: readonly number[] | null = compile.inputShape(lazyGraph);
const compileLazyInputShapeSnake: readonly number[] | null = compile.input_shape(lazyGraph);
const compileLazyOutputShape: readonly [1] | null = compile.outputShape(lazyGraph);
const compileLazyOutputShapeSnake: readonly [1] | null = compile.output_shape(lazyGraph);
const compileLazyShapeConstraints: ModuleKernelShapeConstraints | null = compile.shapeConstraints(lazyGraph);
const compileLazyShapeConstraintsSnake: ModuleKernelShapeConstraints | null = compile.shape_constraints(lazyGraph);
const compileLazyParameterLayout: ModuleKernelParameterLayout | null = compile.parameterLayout(lazyGraph);
const compileLazyParameterLayoutSnake: ModuleKernelParameterLayout | null = compile.parameter_layout(lazyGraph);
const compileLazyMemoryLayout: ModuleKernelMemoryLayout | null = compile.memoryLayout(lazyGraph);
const compileLazyMemoryLayoutSnake: ModuleKernelMemoryLayout | null = compile.memory_layout(lazyGraph);
const compileLazyBufferLayout: ModuleKernelBufferLayout | null = compile.bufferLayout(lazyGraph);
const compileLazyBufferLayoutSnake: ModuleKernelBufferLayout | null = compile.buffer_layout(lazyGraph);
const lazyCompileForInferenceBindings: ProgramBindings<readonly [2], readonly [3]> = {
  weights: new Float32Array([1, 0, 0, 1, 1, 1]),
  bias: new Float32Array([0, 0, 0]),
};
const lazyCompiledInference: CompiledInference<readonly [2], readonly [3]> = compile.compileForInference(lazyMatmulBiasGraph, { backend: "cpu", inputShape: [2] as const }, lazyCompileForInferenceBindings);
const lazyCompiledInferenceAlias: CompiledInference<readonly [2], readonly [3]> = compile.compile_for_inference(lazyMatmulBiasGraph, { backend: "cpu", inputShape: [2] as const }, lazyCompileForInferenceBindings);
const lazyCompiledInferenceInto: Float32Array = lazyCompiledInference.into(new Float32Array(3), tensor([1, 2], [2] as const));
const lazyModuleGraph = lazy.fromModule(nn.sequential([nn.linear(2, 3), nn.relu(), nn.linear(3, 1)]), { inputShape: [2] as const });
const lazyShapeModuleGraph = lazy.fromModule(nn.sequential([
  nn.squeeze(0),
  nn.unsqueeze(0),
  nn.broadcastTo([2, 3] as const),
  nn.narrow(0, 0, 1),
  nn.slice(1, 0, 2),
  nn.transpose(0, 1),
  nn.permute([1, 0] as const),
  nn.select(0, 0),
]), { inputShape: [1, 3] as const });
const lazyEmbeddingModuleGraph = lazy.fromModule(nn.embedding(4, 3), { inputShape: [2] as const });
const lazyNormModuleGraph = lazy.fromModule(nn.sequential([
  nn.embedding(4, 3),
  nn.layerNorm(3),
  nn.rmsNorm(3),
]), { inputShape: [2] as const });
const lazyEvalDropoutModuleGraph = lazy.fromModule(nn.sequential([
  nn.dropout(0.5, { training: false }),
  nn.relu(),
]), { inputShape: [2] as const });
const lazyConvPoolModuleGraph = lazy.fromModule(nn.sequential([
  nn.conv2d(1, 2, 1),
  nn.max_pool2d(2),
  nn.avg_pool2d(2),
]), { inputShape: [1, 4, 4] as const });
const lazyTrainingDropoutModuleGraph = lazy.fromModule(nn.dropout(0.5, { training: true }), { inputShape: [2] as const });
const lazyTrainingDropoutSupport = lazyTrainingDropoutModuleGraph.compileSupport();
const lazyModuleTrace: ModuleProgramTrace = lazy.traceModule(nn.linear(2, 1), { inputShape: [2] as const });
const lazyModuleArtifacts = lazy.moduleArtifacts(nn.linear(2, 1), { inputShape: [2] as const });
const lazyModuleSupport = lazy.moduleCompileSupport(nn.linear(2, 1), { inputShape: [2] as const });
const simpleLinear = simple.nn.linear(2, 1);
const simpleTensor = simple.tensor([1, 2], [2] as const);
const simpleSupport = simple.compile.compileSupport(simpleLinear, { inputShape: [2] as const });
type LazyInputShape = Expect<Equal<typeof lazyInput.shape, readonly [2]>>;
type LazyGraphShape = Expect<Equal<typeof lazyGraph.shape, readonly [1]>>;
type LazyEmbeddingGraphShape = Expect<Equal<typeof lazyEmbeddingGraph.shape, readonly [2, 3]>>;
type LazyNamespaceEmbeddingGraphShape = Expect<Equal<typeof lazyNamespaceEmbeddingGraph.shape, readonly [2, 3]>>;
type LazySnakeNamespaceEmbeddingGraphShape = Expect<Equal<typeof lazySnakeNamespaceEmbeddingGraph.shape, readonly [2, 3]>>;
type LazyGridEmbeddingGraphShape = Expect<Equal<typeof lazyGridEmbeddingGraph.shape, readonly [1, 2, 3]>>;
type LazySnakeSoftmaxGraphShape = Expect<Equal<typeof lazySnakeSoftmaxGraph.shape, readonly [2, 3]>>;
type LazyBatchSoftmaxGraphShape = Expect<Equal<typeof lazyBatchSoftmaxGraph.shape, readonly [2, 3]>>;
type LazyBatchLogSoftmaxGraphShape = Expect<Equal<typeof lazyBatchLogSoftmaxGraph.shape, readonly [2, 3]>>;
type LazySigmoidGraphShape = Expect<Equal<typeof lazySigmoidGraph.shape, readonly [1]>>;
type LazyNamespaceSigmoidGraphShape = Expect<Equal<typeof lazyNamespaceSigmoidGraph.shape, readonly [1]>>;
type LazyMatmulWeightShape = Expect<Equal<typeof lazyMatmulWeight.shape, readonly [2, 3]>>;
type LazyMatmulBiasShape = Expect<Equal<typeof lazyMatmulBias.shape, readonly [3]>>;
type LazyMatmulGraphShape = Expect<Equal<typeof lazyMatmulGraph.shape, readonly [3]>>;
type LazyMatmulBiasGraphShape = Expect<Equal<typeof lazyMatmulBiasGraph.shape, readonly [3]>>;
type LazyNamespaceMatmulGraphShape = Expect<Equal<typeof lazyNamespaceMatmulGraph.shape, readonly [3]>>;
type LazyNamespaceMatmulBiasGraphShape = Expect<Equal<typeof lazyNamespaceMatmulBiasGraph.shape, readonly [3]>>;
type LazyMmGraphShape = Expect<Equal<typeof lazyMmGraph.shape, readonly [1, 3]>>;
type LazyActivationChainShape = Expect<Equal<typeof lazyActivationChain.shape, readonly [2]>>;
type LazyNamespaceActivationChainShape = Expect<Equal<typeof lazyNamespaceActivationChain.shape, readonly [2]>>;
type LazyDropoutGraphShape = Expect<Equal<typeof lazyDropoutGraph.shape, readonly [2]>>;
type LazyNamespaceDropoutGraphShape = Expect<Equal<typeof lazyNamespaceDropoutGraph.shape, readonly [2]>>;
type LazyReductionChainShape = Expect<Equal<typeof lazyReductionChain.shape, readonly [1]>>;
type LazyNamespaceReductionShape = Expect<Equal<typeof lazyNamespaceReduction.shape, readonly [1]>>;
type LazyShapeChainShape = Expect<Equal<typeof lazyShapeChain.shape, readonly [2]>>;
type LazyNamespaceShapeChainShape = Expect<Equal<typeof lazyNamespaceShapeChain.shape, readonly [2]>>;
type LazySnakeShapeChainShape = Expect<Equal<typeof lazySnakeShapeChain.shape, readonly [2, 3]>>;
type LazyShapeModuleGraphShape = Expect<Equal<typeof lazyShapeModuleGraph.shape, readonly [2]>>;
type LazyEvalDropoutModuleGraphShape = Expect<Equal<typeof lazyEvalDropoutModuleGraph.shape, readonly [2]>>;
type LazyConvPoolModuleGraphShape = Expect<Equal<typeof lazyConvPoolModuleGraph.shape, readonly [2, 1, 1]>>;
type LazySupportNativePath = Expect<Equal<typeof lazySupport.nativePath, "device-program">>;
type LazyRequiredSupportNativePath = Expect<Equal<typeof lazyRequiredSupport.nativePath, "device-program">>;
type LazyNamespaceRequiredSupportNativePath = Expect<Equal<typeof lazyNamespaceRequiredSupport.nativePath, "device-program">>;
type CompileLazyRequiredSupportNativePath = Expect<Equal<typeof compileLazyRequiredSupport.nativePath, "device-program">>;
type CompileLazyPlanNativePath = Expect<Equal<typeof compileLazyPlan.nativePath, "device-program">>;
type LazyManifestPath = Expect<Equal<typeof lazy.lazyManifest.runtimePath, "LazyTensor -> Trace -> TensorProgramIr -> KernelPlan -> Program">>;
void lazyTrace;
void lazyIr;
void lazyIrAlias;
void lazyPlan;
void lazyCanCompile;
void lazyCanCompileSnake;
void lazyRequiredSupport;
void lazyRequiredSupportSnake;
void lazyCompiledProgram;
void lazyCompiledProgramOutputShape;
void lazyMethodCompiledProgram;
void lazyMethodCompiledProgramOutputShape;
void lazyNamespaceCanCompile;
void lazyNamespaceCanCompileSnake;
void lazyNamespaceRequiredSupport;
void lazyNamespaceRequiredSupportSnake;
void compileLazyTrace;
void compileLazyAnalysis;
void compileLazySupport;
void compileLazyRequiredSupport;
void compileLazyPlan;
void compileLazyCanCompile;
void compileLazyCanCompileSnake;
void compileLazyIr;
void compileLazyIrSnake;
void compileLazyKernelPlan;
void compileLazyKernelPlanSnake;
void compileLazyInputShape;
void compileLazyInputShapeSnake;
void compileLazyOutputShape;
void compileLazyOutputShapeSnake;
void compileLazyShapeConstraints;
void compileLazyShapeConstraintsSnake;
void compileLazyParameterLayout;
void compileLazyParameterLayoutSnake;
void compileLazyMemoryLayout;
void compileLazyMemoryLayoutSnake;
void compileLazyBufferLayout;
void compileLazyBufferLayoutSnake;
void lazyCompileForInferenceBindings;
void lazyCompiledInference;
void lazyCompiledInferenceAlias;
void lazyCompiledInferenceInto;
void lazyEmbeddingGraph;
void lazyNamespaceEmbeddingGraph;
void lazySnakeNamespaceEmbeddingGraph;
void lazyGridEmbeddingGraph;
void lazyGridEmbeddingSupport;
void lazyConvPoolGraph;
void lazyNamespaceConvPoolGraph;
void lazySnakeConvPoolGraph;
void lazySigmoidGraph;
void lazyNamespaceSigmoidGraph;
void lazyMatmulWeight;
void lazyMatmulBias;
void lazyMatmulGraph;
void lazyMatmulBiasGraph;
void lazyNamespaceMatmulGraph;
void lazyNamespaceMatmulBiasGraph;
void lazyMmGraph;
void lazyActivationChain;
void lazyNamespaceActivationChain;
void lazySnakeSoftmaxGraph;
void lazyBatchSoftmaxGraph;
void lazyBatchLogSoftmaxGraph;
void lazyDropoutGraph;
void lazyNamespaceDropoutGraph;
void lazyTrainingDropoutGraph;
void lazyTrainingDropoutGraphSupport;
void lazyReductionChain;
void lazyNamespaceReduction;
void lazyModuleGraph;
void lazyShapeChain;
void lazyNamespaceShapeChain;
void lazySnakeShapeChain;
void lazyShapeModuleGraph;
void lazyEmbeddingModuleGraph;
void lazyNormModuleGraph;
void lazyEvalDropoutModuleGraph;
void lazyConvPoolModuleGraph;
void lazyTrainingDropoutModuleGraph;
void lazyTrainingDropoutSupport;
void lazyModuleTrace;
void lazyModuleArtifacts;
void lazyModuleSupport;
void simpleLinear;
void simpleTensor;
void simpleSupport;
type SimpleNamespaceShape = Expect<Equal<typeof simple, PublicSimpleNamespace>>;
type SimpleTensorShape = Expect<typeof simpleTensor.shape extends TensorShapeTuple ? true : false>;
type SimpleCompileSupportShape = Expect<Equal<typeof simpleSupport.inputShape, readonly [2] | undefined>>;
type PublicApiContractKind = Expect<Equal<PublicApiContractManifest["kind"], "zgml-public-api-contract">>;
type PublicApiContractOwner = Expect<Equal<PublicApiContractManifest["policyOwner"], "src/ts/public_api.ts">>;
type PublicApiContractProductSourceOfTruth = Expect<Equal<PublicApiContractManifest["productSourceOfTruth"], "ts-api-zig-core">>;
type PublicApiContractPackageFanout = Expect<Equal<PublicApiContractManifest["packageFanout"], "tsdown">>;
type PublicApiContractTypesRoot = Expect<Equal<PublicApiContractManifest["packageTypesRoot"], "dist/public_api.d.cts">>;
type PublicApiContractDeclarationMirror = Expect<Equal<PublicApiContractManifest["declarationMirror"], false>>;
type ShapeAssertions = [
  ShapeFromNumber,
  ShapeFromTuple,
  BunProgramSubpath,
  BunSessionSubpath,
  BunTensorSubpath,
  BunKernelPlanDescriptorSignatures,
  NodeProgramSubpath,
  NodeSessionSubpath,
  NodeTensorSubpath,
  ProgramFriendlySubpath,
  ProgramExecutionPlanFriendlySubpath,
  ProgramBindingPlanFriendlySubpath,
  SessionFriendlySubpath,
  SessionExecutionPlanFriendlySubpath,
  SessionCompatibilityFriendlySubpath,
  StepParamsExecutionPlanFriendlySubpath,
  StepParamsCompatibilityFriendlySubpath,
  StepParamsFriendlySubpath,
  FrontendManifestSource,
  FrontendManifestProductLanguage,
  FrontendManifestProductSource,
  FrontendManifestProductSemanticsOwner,
  FrontendManifestFanout,
  FrontendManifestNativeRole,
  FrontendManifestNativeAlignment,
  FrontendManifestSync,
  FrontendManifestHandwrittenMirrors,
  FrontendManifestNativeContractBoundary,
  FrontendManifestModuleCompilerCore,
  FrontendManifestNativeCompileEvidence,
  FrontendManifestProgramInspectionCore,
  FrontendManifestEagerHotPathCore,
  FrontendManifestInferenceHotPathCore,
  FrontendManifestTrainingHotPathCore,
  FrontendManifestUnsupportedHotPathPolicy,
  PublicApiContractKind,
  PublicApiContractOwner,
  PublicApiContractProductSourceOfTruth,
  PublicApiContractPackageFanout,
  PublicApiContractTypesRoot,
  PublicApiContractDeclarationMirror,
];

const currentRuntimeFeatures: RuntimeFeatures = runtimeInfo().features;
const runtimeActivationChainFeature: boolean = currentRuntimeFeatures.nativeModuleActivationChain;
const runtimeNativeEagerMatmulFeature: boolean = currentRuntimeFeatures.nativeEagerMatmul;
const runtimeNativeEagerElementwiseFeature: boolean = currentRuntimeFeatures.nativeEagerElementwise;
const runtimeNativeEagerReduceFeature: boolean = currentRuntimeFeatures.nativeEagerReduce;
const runtimeNativeEagerConv2dFeature: boolean = currentRuntimeFeatures.nativeEagerConv2d;
const runtimeNativeEagerPool2dFeature: boolean = currentRuntimeFeatures.nativeEagerPool2d;
const rootNativeCoreEvidence: NativeCoreEvidence = nativeCore();
const rootNativeCoreAliasEvidence: NativeCoreEvidence = native_core();
const namespaceNativeCoreEvidence: NativeCoreEvidence = zgml.nativeCore();
const namespaceNativeCoreAliasEvidence: NativeCoreEvidence = zgml.native_core();
const simpleNativeCoreEvidence: NativeCoreEvidence = simple.nativeCore();
const nativeCoreProgramSessionDomain: boolean = namespaceNativeCoreEvidence.domains.programSession;
const nativeCoreMatmulOp: boolean = namespaceNativeCoreEvidence.eagerOps.matmul;
const nativeCoreUserApiBoundary: "typescript" = namespaceNativeCoreEvidence.boundary.userApi;
const nativeCoreTensorRuntimeBoundary: "zig" = namespaceNativeCoreEvidence.boundary.tensorRuntime;
const nativeCoreFfiBoundary: "c-abi" = namespaceNativeCoreEvidence.boundary.ffi;
const nativeCoreModuleCompilerBoundary: "zig-module-program" = namespaceNativeCoreEvidence.boundary.moduleCompiler;
const nativeCoreCompileEvidenceBoundary: "native-program-inspection" = namespaceNativeCoreEvidence.boundary.compileEvidence;
const nativeCoreHotPathBoundary: "program-session" = namespaceNativeCoreEvidence.boundary.hotPath;
const nativeCoreTrainingBoundary: "zig-ffi-kernels" = namespaceNativeCoreEvidence.boundary.training;
const coreShapeSubpathShape: CoreShapeSubpath = [2, 3];
const coreShapeSubpathCount: number = coreShapeScalarCount(coreShapeSubpathShape);
const runtimeKernelPlanSubpathSource: "ts" = runtimeKernelPlanManifest.source;
const runtimeKernelPlanSubpathOwner: "src/ts/runtime/kernel_plan.ts" = runtimeKernelPlanManifest.policyOwner;
const createLinearModuleClassSubpathFactory: typeof createLinearModuleClassSubpath = createLinearModuleClassSubpath;
const setModuleTrainingSubpathHelper: typeof setModuleTrainingSubpath = setModuleTrainingSubpath;

const literalTensor = tensor([1, 2, 3, 4, 5, 6], [2, 3] as const);
const literalAsTensor: Tensor<readonly [2, 3]> = as_tensor(new Float32Array(6), [2, 3] as const);
const literalAsTensorCamel: Tensor<readonly [2, 3]> = asTensor([1, 2, 3, 4, 5, 6], [2, 3] as const);
const literalAsArrayTensor: Tensor<readonly [2, 3]> = asarray([1, 2, 3, 4, 5, 6], [2, 3] as const);
const literalFromNumpyTensor: Tensor<readonly [2, 3]> = from_numpy(new Float32Array(6), [2, 3] as const);
const literalFromNumpyCamelTensor: Tensor<readonly [2, 3]> = fromNumpy([1, 2, 3, 4, 5, 6], [2, 3] as const);
const literalTensor3d = tensor(new Float32Array(24), [2, 3, 4] as const);
const literalTensor4d = tensor(new Float32Array(120), [2, 3, 4, 5] as const);
const literalShape: readonly [2, 3] = literalTensor.shape;
const literalCpuTensor: Tensor<readonly [2, 3]> = literalTensor.cpu();
const literalFloatTensor: Tensor<readonly [2, 3]> = literalTensor.float();
const literalFloat32Tensor: Tensor<readonly [2, 3]> = literalTensor.float32();
const literalToTensor: Tensor<readonly [2, 3]> = literalTensor.to("cpu");
const literalRootToTensor: Tensor<readonly [2, 3]> = to(literalTensor, "cpu");
const literalRootCpuTensor: Tensor<readonly [2, 3]> = cpu(literalTensor);
const literalRootFloatTensor: Tensor<readonly [2, 3]> = float(literalTensor);
const literalRootFloat32Tensor: Tensor<readonly [2, 3]> = float32(literalTensor);
const literalStaticToTensor: Tensor<readonly [2, 3]> = Tensor.to(literalTensor, { device: "cpu", copy: true });
const literalTypeAsTensor: Tensor<readonly [2, 3]> = literalTensor.typeAs(literalFloatTensor);
const literalTypeAsSnakeTensor: Tensor<readonly [2, 3]> = literalTensor.type_as(literalFloatTensor);
const literalRootTypeAsTensor: Tensor<readonly [2, 3]> = typeAs(literalTensor, literalFloatTensor);
const literalRootTypeAsSnakeTensor: Tensor<readonly [2, 3]> = type_as(literalTensor, literalFloatTensor);
const literalStaticTypeAsTensor: Tensor<readonly [2, 3]> = Tensor.typeAs(literalTensor, literalFloatTensor);
const literalStaticTypeAsSnakeTensor: Tensor<readonly [2, 3]> = Tensor.type_as(literalTensor, literalFloatTensor);
const literalNewEmptyTensor: Tensor<readonly [4, 5]> = literalTensor.new_empty([4, 5] as const);
const literalNewEmptyCamelTensor: Tensor<readonly [4, 5]> = literalTensor.newEmpty([4, 5] as const);
const literalNewZerosTensor: Tensor<readonly [4, 5]> = literalTensor.new_zeros([4, 5] as const);
const literalNewZerosCamelTensor: Tensor<readonly [4, 5]> = literalTensor.newZeros([4, 5] as const);
const literalNewOnesTensor: Tensor<readonly [4, 5]> = literalTensor.new_ones([4, 5] as const);
const literalNewOnesCamelTensor: Tensor<readonly [4, 5]> = literalTensor.newOnes([4, 5] as const);
const literalNewFullTensor: Tensor<readonly [4, 5]> = literalTensor.new_full([4, 5] as const, 3);
const literalNewFullCamelTensor: Tensor<readonly [4, 5]> = literalTensor.newFull([4, 5] as const, 3);
const literalReshapeAsTensor: Tensor<readonly [3, 2]> = literalTensor.reshape_as(tensor(new Float32Array(6), [3, 2] as const));
const literalViewAsTensor: Tensor<readonly [3, 2]> = literalTensor.view_as(tensor(new Float32Array(6), [3, 2] as const));
const literalExpandAsTensor: Tensor<readonly [2, 3]> = tensor([1, 2, 3], [1, 3] as const).expand_as(literalTensor);
const literalRepeatTensor: Tensor<readonly [4, 9]> = literalTensor.repeat([2, 3] as const);
const literalRepeatVarargsTensor: Tensor<readonly [4, 9]> = literalTensor.repeat(2, 3);
const literalTileTensor: Tensor<readonly [2, 2, 9]> = literalTensor.tile([2, 1, 3] as const);
const literalRootRepeatTensor: Tensor<readonly [4, 9]> = repeat(literalTensor, [2, 3] as const);
const literalRootRepeatVarargsTensor: Tensor<readonly [4, 9]> = repeat(literalTensor, 2, 3);
const literalStaticTileTensor: Tensor<readonly [4, 9]> = Tensor.tile(literalTensor, 2, 3);
const literalRootTileTensor: Tensor<readonly [4, 9]> = tile(literalTensor, [2, 3] as const);
const literalRootEmptyTensor: Tensor<readonly [2, 3]> = empty([2, 3] as const);
const literalRootEmptyLikeTensor: Tensor<readonly [2, 3]> = emptyLike(literalTensor);
const literalRootEmptyLikeAliasTensor: Tensor<readonly [2, 3]> = empty_like(literalTensor);
const literalRootZerosLikeTensor: Tensor<readonly [2, 3]> = zerosLike(literalTensor);
const literalRootZerosLikeAliasTensor: Tensor<readonly [2, 3]> = zeros_like(literalTensor);
const literalRootOnesLikeTensor: Tensor<readonly [2, 3]> = onesLike(literalTensor);
const literalRootOnesLikeAliasTensor: Tensor<readonly [2, 3]> = ones_like(literalTensor);
const literalRootEyeTensor: Tensor<readonly [3, 3]> = eye(3);
const literalRootFullLikeTensor: Tensor<readonly [2, 3]> = fullLike(literalTensor, 7);
const literalRootFullLikeAliasTensor: Tensor<readonly [2, 3]> = full_like(literalTensor, 8);
const literalRootRandLikeTensor: Tensor<readonly [2, 3]> = randLike(literalTensor);
const literalRootRandLikeAliasTensor: Tensor<readonly [2, 3]> = rand_like(literalTensor);
const literalRootRandIntTensor: Tensor<readonly [2, 3]> = randInt(1, 5, [2, 3] as const, { seed: 11 });
const literalRootRandintTensor: Tensor<readonly [2, 3]> = randint(5, [2, 3] as const, { seed: 11 });
const literalRootRandPermTensor: Tensor<readonly [5]> = randPerm(5, { seed: 13 });
const literalRootRandpermTensor: Tensor<readonly [5]> = randperm(5, { seed: 13 });
const literalRootRandnLikeTensor: Tensor<readonly [2, 3]> = randnLike(literalTensor);
const literalRootRandnLikeAliasTensor: Tensor<readonly [2, 3]> = randn_like(literalTensor);
const literalStaticEmptyTensor: Tensor<readonly [2, 3]> = Tensor.empty([2, 3] as const);
const literalStaticEmptyLikeTensor: Tensor<readonly [2, 3]> = Tensor.emptyLike(literalTensor);
const literalStaticEmptyLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.empty_like(literalTensor);
const literalStaticZerosLikeTensor: Tensor<readonly [2, 3]> = Tensor.zerosLike(literalTensor);
const literalStaticZerosLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.zeros_like(literalTensor);
const literalStaticOnesLikeTensor: Tensor<readonly [2, 3]> = Tensor.onesLike(literalTensor);
const literalStaticOnesLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.ones_like(literalTensor);
const literalStaticEyeTensor: Tensor<readonly [4, 4]> = Tensor.eye(4);
const literalStaticFullLikeTensor: Tensor<readonly [2, 3]> = Tensor.fullLike(literalTensor, 9);
const literalStaticFullLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.full_like(literalTensor, 10);
const literalStaticRandLikeTensor: Tensor<readonly [2, 3]> = Tensor.randLike(literalTensor);
const literalStaticRandLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.rand_like(literalTensor);
const literalStaticRandIntTensor: Tensor<readonly [2, 3]> = Tensor.randInt(2, 6, [2, 3] as const, { seed: 12 });
const literalStaticRandintTensor: Tensor<readonly [2, 3]> = Tensor.randint(6, [2, 3] as const, { seed: 12 });
const literalStaticRandPermTensor: Tensor<readonly [5]> = Tensor.randPerm(5, { seed: 14 });
const literalStaticRandpermTensor: Tensor<readonly [5]> = Tensor.randperm(5, { seed: 14 });
const literalStaticRandnLikeTensor: Tensor<readonly [2, 3]> = Tensor.randnLike(literalTensor);
const literalStaticRandnLikeAliasTensor: Tensor<readonly [2, 3]> = Tensor.randn_like(literalTensor);
const literalCloneTensor: Tensor<readonly [2, 3]> = literalTensor.clone();
const literalFillInPlaceTensor: Tensor<readonly [2, 3]> = literalCloneTensor.fill_(3);
const literalZeroInPlaceTensor: Tensor<readonly [2, 3]> = literalCloneTensor.zero_();
const literalOnesInPlaceTensor: Tensor<readonly [2, 3]> = literalCloneTensor.ones_();
const literalCopyInPlaceTensor: Tensor<readonly [2, 3]> = literalCloneTensor.copy_(literalTensor);
const literalDetachTensor: Tensor<readonly [2, 3]> = literalTensor.detach();
const literalEqTensor: Tensor<readonly [2, 3]> = literalTensor.eq(1);
const literalNeTensor: Tensor<readonly [2, 3]> = literalTensor.ne(1);
const literalLtTensor: Tensor = literalTensor.lt(tensor([3, 3, 3], [1, 3] as const));
const literalLeTensor: Tensor<readonly [2, 3]> = literalTensor.le(3);
const literalGtTensor: Tensor<readonly [2, 3]> = literalTensor.gt(3);
const literalGeTensor: Tensor = literalTensor.ge(tensor([3, 3, 3], [1, 3] as const));
const literalSameShapeTensor = tensor([1, 1, 1, 1, 1, 1], [2, 3] as const);
const literalTensorAddSameShape: Tensor<readonly [2, 3]> = literalTensor.add(literalSameShapeTensor);
const literalTensorSubSameShape: Tensor<readonly [2, 3]> = literalTensor.sub(literalSameShapeTensor);
const literalTensorEqSameShape: Tensor<readonly [2, 3]> = literalTensor.eq(literalSameShapeTensor);
const literalTensorIsCloseSameShape: Tensor<readonly [2, 3]> = literalTensor.isclose(literalSameShapeTensor);
const literalTensorMaximumSameShape: Tensor<readonly [2, 3]> = literalTensor.maximum(literalSameShapeTensor);
const literalTensorWhereSameShape: Tensor<readonly [2, 3]> = literalTensor.gt(3).where(literalTensor, literalSameShapeTensor);
const literalTensorAddInPlace: Tensor<readonly [2, 3]> = literalCloneTensor.add_(1);
const literalTensorSubInPlace: Tensor<readonly [2, 3]> = literalCloneTensor.sub_(literalSameShapeTensor);
const literalTensorMulInPlace: Tensor<readonly [2, 3]> = literalCloneTensor.mul_(2);
const literalTensorDivInPlace: Tensor<readonly [2, 3]> = literalCloneTensor.div_(literalSameShapeTensor);
const literalTensorBroadcastAdd: Tensor<readonly [2, 3]> = literalTensor.add(tensor([1, 2, 3], [1, 3] as const));
const literalTensorBroadcastEq: Tensor<readonly [2, 3]> = literalTensor.eq(tensor([1, 2, 3], [3] as const));
const literalTensorBroadcastIsClose: Tensor<readonly [2, 3]> = literalTensor.isclose(tensor([1, 2, 3], [3] as const), { rtol: 1e-4, equal_nan: true });
const literalTensorBroadcastWhere: Tensor<readonly [2, 3]> = literalTensor.gt(3).where(tensor([1, 2, 3], [1, 3] as const), tensor([0], [1] as const));
type LiteralBroadcastShape = Expect<Equal<BroadcastShape<readonly [2, 3], readonly [1, 3]>, readonly [2, 3]>>;
type LiteralBroadcast3dShape = Expect<Equal<BroadcastShape<readonly [2, 3, 4], readonly [4]>, readonly [2, 3, 4]>>;
type LiteralBroadcast4dShape = Expect<Equal<BroadcastShape<readonly [2, 3, 4, 5], readonly [1, 5]>, readonly [2, 3, 4, 5]>>;
type LiteralWhereBroadcastShape = Expect<Equal<WhereShape<readonly [2, 3], readonly [1, 3], readonly [1]>, readonly [2, 3]>>;
const literalPowTensor: Tensor<readonly [2, 3]> = literalTensor.pow(2);
const literalRootPowTensor: Tensor<readonly [2, 3]> = pow(literalTensor, 0.5);
const literalRootAddTensor: Tensor<readonly [2, 3]> = add(literalTensor, tensor([1, 2, 3], [1, 3] as const));
const literalRootAddSameShapeTensor: Tensor<readonly [2, 3]> = add(literalTensor, literalSameShapeTensor);
const literalRootIsCloseTensor: Tensor<readonly [2, 3]> = isclose(literalTensor, literalSameShapeTensor);
const literalRootMulSameShapeTensor: Tensor<readonly [2, 3]> = mul(literalTensor, literalSameShapeTensor);
const literalRootSubTensor: Tensor<readonly [2, 3]> = sub(literalTensor, 1);
const literalStaticMulTensor: Tensor<readonly [2, 3]> = Tensor.mul(literalTensor, 2);
const literalStaticMulBroadcastTensor: Tensor<readonly [2, 3]> = Tensor.mul(literalTensor, tensor([1, 2, 3], [3] as const));
const literalStaticIsCloseBroadcastTensor: Tensor<readonly [2, 3]> = Tensor.isclose(literalTensor, tensor([1, 2, 3], [3] as const));
const literalStaticDivSameShapeTensor: Tensor<readonly [2, 3]> = Tensor.div(literalTensor, literalSameShapeTensor);
const literalStaticDivTensor: Tensor<readonly [2, 3]> = Tensor.div(literalTensor, 2);
const literalRootCloneTensor: Tensor<readonly [2, 3]> = clone(literalTensor);
const literalStaticDetachTensor: Tensor<readonly [2, 3]> = Tensor.detach(literalTensor);
const literalRootReshapeTensor: Tensor<readonly [3, 2]> = reshape(literalTensor, [3, 2] as const);
const literalRootInferReshapeTensor: Tensor<readonly [3, 2]> = reshape(literalTensor, [3, -1] as const);
const literalStaticViewTensor: Tensor<readonly [1, 6]> = Tensor.view(literalTensor, [1, 6] as const);
const literalStaticInferViewTensor: Tensor<readonly [1, 6]> = Tensor.view(literalTensor, [1, -1] as const);
const literalFlattenTensor: Tensor<readonly [6]> = literalTensor.flatten();
const literalRootFlattenTensor: Tensor<readonly [6]> = flatten(literalTensor);
const literalStaticFlattenTensor: Tensor<readonly [6]> = Tensor.flatten(literalTensor, 0, -1);
const literalFlatten3dTensor: Tensor<readonly [24]> = literalTensor3d.flatten();
const literalRootFlatten3dMiddleTensor: Tensor<readonly [2, 12]> = flatten(literalTensor3d, 1, -1);
const literalStaticFlatten3dLeadingTensor: Tensor<readonly [6, 4]> = Tensor.flatten(literalTensor3d, 0, -2);
const literalUnsqueezeTensor: Tensor<readonly [1, 2, 3]> = literalTensor.unsqueeze(0);
const literalRootUnsqueezeTensor: Tensor<readonly [2, 3, 1]> = unsqueeze(literalTensor, -1);
const literalStaticUnsqueezeTensor: Tensor<readonly [2, 1, 3]> = Tensor.unsqueeze(literalTensor, 1);
const literalUnsqueeze3dMiddleTensor: Tensor<readonly [2, 3, 1, 4]> = literalTensor3d.unsqueeze(2);
const literalTransposeTensor: Tensor<readonly [3, 2]> = literalTensor.transpose();
const literalPropertyTransposeTensor: Tensor<readonly [3, 2]> = literalTensor.T;
const literalMatrixTransposeTensor: Tensor<readonly [3, 2]> = literalTensor.mT;
const literalRootTransposeTensor: Tensor<readonly [3, 2]> = transpose(literalTensor, 0, 1);
const literalStaticTransposeTensor: Tensor<readonly [3, 2]> = Tensor.transpose(literalTensor);
const literalTranspose3dOuterTensor: Tensor<readonly [4, 3, 2]> = literalTensor3d.transpose(0, -1);
const literalMatrixTranspose3dTensor: Tensor<readonly [2, 4, 3]> = literalTensor3d.mT;
const literalSelectRowsTensor: Tensor<readonly [3]> = literalTensor.select(0, 1);
const literalRootSelectColsTensor: Tensor<readonly [2]> = select(literalTensor, -1, 2);
const literalStaticSelectColsTensor: Tensor<readonly [2]> = Tensor.select(literalTensor, 1, 2);
const literalSelect3dLeadingTensor: Tensor<readonly [3, 4]> = literalTensor3d.select(0, 1);
const literalRootSelect3dTrailingTensor: Tensor<readonly [2, 3]> = select(literalTensor3d, -1, 2);
const literalSliceColsTensor: Tensor<readonly [2, 2]> = literalTensor.slice(1, 1, 3);
const literalRootSliceRowsTensor: Tensor<readonly [1, 3]> = slice(literalTensor, 0, 0, 1);
const literalStaticSlice3dTensor: Tensor<readonly [2, 2, 4]> = Tensor.slice(literalTensor3d, 1, 0, 2);
const literalNarrowRowsTensor: Tensor<readonly [1, 3]> = literalTensor.narrow(0, 0, 1);
const literalRootNarrowColsTensor: Tensor<readonly [2, 2]> = narrow(literalTensor, -1, 0, 2);
const literalStaticNarrowColsTensor: Tensor<readonly [2, 2]> = Tensor.narrow(literalTensor, 1, 1, 2);
const literalNarrow3dMiddleTensor: Tensor<readonly [2, 2, 4]> = literalTensor3d.narrow(1, 0, 2);
const literalStaticNarrow3dTrailingTensor: Tensor<readonly [2, 3, 2]> = Tensor.narrow(literalTensor3d, -1, 1, 2);
type LiteralTransposeShape = Expect<Equal<TransposeShape<readonly [2, 3]>, readonly [3, 2]>>;
type LiteralFlattenShape = Expect<Equal<FlattenShape<readonly [2, 3]>, readonly [6]>>;
type LiteralFlatten3dShape = Expect<Equal<FlattenShape<readonly [2, 3, 4]>, readonly [24]>>;
type LiteralFlatten3dMiddleShape = Expect<Equal<FlattenShape<readonly [2, 3, 4], 1, -1>, readonly [2, 12]>>;
type LiteralFlatten4dMiddleShape = Expect<Equal<FlattenShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 12, 5]>>;
type LiteralUnsqueezeShape = Expect<Equal<UnsqueezeShape<readonly [2, 3], 1>, readonly [2, 1, 3]>>;
type LiteralUnsqueeze3dShape = Expect<Equal<UnsqueezeShape<readonly [2, 3, 4], -2>, readonly [2, 3, 1, 4]>>;
type LiteralUnsqueeze4dShape = Expect<Equal<UnsqueezeShape<readonly [2, 3, 4, 5], 2>, readonly [2, 3, 1, 4, 5]>>;
type LiteralTranspose3dOuterShape = Expect<Equal<TransposeShape<readonly [2, 3, 4], 0, -1>, readonly [4, 3, 2]>>;
type LiteralTranspose4dMiddleShape = Expect<Equal<TransposeShape<readonly [2, 3, 4, 5], 1, -2>, readonly [2, 4, 3, 5]>>;
type LiteralSqueezeAllShape = Expect<Equal<SqueezeAllShape<readonly [1, 2, 1, 3]>, readonly [2, 3]>>;
type LiteralSqueezeShape = Expect<Equal<SqueezeShape<readonly [1, 2, 3], 0>, readonly [2, 3]>>;
type LiteralSqueeze4dMiddleShape = Expect<Equal<SqueezeShape<readonly [2, 1, 3, 4], 1>, readonly [2, 3, 4]>>;
type LiteralSelectShape = Expect<Equal<SelectShape<readonly [2, 3], -1>, readonly [2]>>;
type LiteralSelect3dMiddleShape = Expect<Equal<SelectShape<readonly [2, 3, 4], 1>, readonly [2, 4]>>;
type LiteralSelect4dTrailingShape = Expect<Equal<SelectShape<readonly [2, 3, 4, 5], -1>, readonly [2, 3, 4]>>;
type LiteralSliceShape = Expect<Equal<SliceShape<readonly [2, 3], 1, 1, 3>, readonly [2, 2]>>;
type LiteralSlice3dMiddleShape = Expect<Equal<SliceShape<readonly [2, 3, 4], 1, 0, 2>, readonly [2, 2, 4]>>;
type LiteralNarrowRowsShape = Expect<Equal<NarrowShape<readonly [2, 3], 0, 1>, readonly [1, 3]>>;
type LiteralNarrowColsShape = Expect<Equal<NarrowShape<readonly [2, 3], -1, 2>, readonly [2, 2]>>;
type LiteralNarrow3dMiddleShape = Expect<Equal<NarrowShape<readonly [2, 3, 4], 1, 2>, readonly [2, 2, 4]>>;
type LiteralNarrow4dDim2Shape = Expect<Equal<NarrowShape<readonly [2, 3, 4, 5], -2, 2>, readonly [2, 3, 2, 5]>>;
type LiteralPermuteShape = Expect<Equal<PermuteShape<readonly [2, 3], readonly [1, 0]>, readonly [3, 2]>>;
type LiteralPermute3dShape = Expect<Equal<PermuteShape<readonly [2, 3, 4], readonly [2, 0, 1]>, readonly [4, 2, 3]>>;
type LiteralPermute4dShape = Expect<Equal<PermuteShape<readonly [2, 3, 4, 5], readonly [0, -1, 1, -2]>, readonly [2, 5, 3, 4]>>;
const literalSqueezeTensor: Tensor<readonly [2, 3]> = literalUnsqueezeTensor.squeeze(0);
const literalRootSqueezeTensor: Tensor<readonly [2, 3]> = squeeze(literalRootUnsqueezeTensor, -1);
const literalStaticSqueezeTensor: Tensor<readonly [2, 3]> = Tensor.squeeze(literalStaticUnsqueezeTensor, 1);
const literalPermuteTensor: Tensor<readonly [3, 2]> = literalTensor.permute([1, 0] as const);
const literalRootPermuteTensor: Tensor<readonly [3, 2]> = permute(literalTensor, [1, 0] as const);
const literalStaticPermuteTensor: Tensor<readonly [3, 2]> = Tensor.permute(literalTensor, [1, 0] as const);
const literalPermute3dTensor: Tensor<readonly [4, 2, 3]> = literalTensor3d.permute([2, 0, 1] as const);
const literalRootPermute3dTensor: Tensor<readonly [2, 4, 3]> = permute(literalTensor3d, [0, -1, -2] as const);
const literalStaticPermute3dTensor: Tensor<readonly [3, 4, 2]> = Tensor.permute(literalTensor3d, [-2, -1, -3] as const);
const literalRootPermute4dTensor: Tensor<readonly [2, 5, 3, 4]> = permute(literalTensor4d, [0, -1, 1, -2] as const);
const literalFlipTensor: Tensor<readonly [2, 3]> = literalTensor.flip([0, -1] as const);
const literalRootFlipTensor: Tensor<readonly [2, 3]> = flip(literalTensor, [1] as const);
const literalStaticFlipTensor: Tensor<readonly [2, 3]> = Tensor.flip(literalTensor, [-1] as const);
const literalRollTensor: Tensor<readonly [2, 3]> = literalTensor.roll(1, 1);
const literalRootRollTensor: Tensor<readonly [2, 3]> = roll(literalTensor, [1, -1] as const, [0, 1] as const);
const literalStaticRollTensor: Tensor<readonly [2, 3]> = Tensor.roll(literalTensor, -2);
const literalRootEqTensor: Tensor<readonly [2, 3]> = eq(literalTensor, literalTensor);
const literalSplitTensors: readonly Tensor[] = literalTensor.split(1, 0);
const literalRootSplitTensors: readonly Tensor[] = split(literalTensor, [1, 1] as const, 0);
const literalStaticSplitTensors: readonly Tensor[] = Tensor.split(literalTensor, 2, 1);
const literalChunkTensors: readonly Tensor[] = literalTensor.chunk(2, 1);
const literalRootChunkTensors: readonly Tensor[] = chunk(literalTensor, 2, 0);
const literalStaticChunkTensors: readonly Tensor[] = Tensor.chunk(literalTensor, 3, 1);
const literalUnbindTensors: readonly Tensor[] = literalTensor.unbind(0);
const literalRootUnbindTensors: readonly Tensor[] = unbind(literalTensor, 1);
const literalStaticUnbindTensors: readonly Tensor[] = Tensor.unbind(literalTensor, -1);
const literalRootEqTensorScalar: Tensor<readonly [2, 3]> = eq(literalTensor, 2);
const literalStaticGeTensor: Tensor<readonly [2, 3]> = Tensor.ge(literalTensor, 2);
const literalStaticGeSameShapeTensor: Tensor<readonly [2, 3]> = Tensor.ge(literalTensor, literalSameShapeTensor);
const literalRootNegTensor: Tensor<readonly [2, 3]> = neg(literalTensor);
const literalRootNegativeTensor: Tensor<readonly [2, 3]> = negative(literalTensor);
const literalStaticNegativeTensor: Tensor<readonly [2, 3]> = Tensor.negative(literalTensor);
const literalInstanceNegativeTensor: Tensor<readonly [2, 3]> = literalTensor.negative();
const literalRootExpm1Tensor: Tensor<readonly [2, 3]> = expm1(literalTensor);
const literalStaticLog1pTensor: Tensor<readonly [2, 3]> = Tensor.log1p(literalTensor);
const literalInstanceLog1pTensor: Tensor<readonly [2, 3]> = literalTensor.log1p();
const literalStaticSqrtTensor: Tensor<readonly [2, 3]> = Tensor.sqrt(literalTensor);
const literalRootRsqrtTensor: Tensor<readonly [2, 3]> = rsqrt(literalTensor);
const literalStaticRsqrtTensor: Tensor<readonly [2, 3]> = Tensor.rsqrt(literalTensor);
const literalRootReciprocalTensor: Tensor<readonly [2, 3]> = reciprocal(literalTensor);
const literalStaticReciprocalTensor: Tensor<readonly [2, 3]> = Tensor.reciprocal(literalTensor);
const literalInstanceReciprocalTensor: Tensor<readonly [2, 3]> = literalTensor.reciprocal();
const literalRootIsfiniteTensor: Tensor<readonly [2, 3]> = isfinite(literalTensor);
const literalStaticIsinfTensor: Tensor<readonly [2, 3]> = Tensor.isinf(literalTensor);
const literalInstanceIsnanTensor: Tensor<readonly [2, 3]> = literalTensor.isnan();
const literalRootIsnanTensor: Tensor<readonly [2, 3]> = isnan(literalTensor);
const literalRootIsinfTensor: Tensor<readonly [2, 3]> = isinf(literalTensor);
const literalRootFloorTensor: Tensor<readonly [2, 3]> = floor(literalTensor);
const literalStaticCeilTensor: Tensor<readonly [2, 3]> = Tensor.ceil(literalTensor);
const literalInstanceFloorTensor: Tensor<readonly [2, 3]> = literalTensor.floor();
const literalRootRoundTensor: Tensor<readonly [2, 3]> = round(literalTensor);
const literalStaticTruncTensor: Tensor<readonly [2, 3]> = Tensor.trunc(literalTensor);
const literalInstanceRoundTensor: Tensor<readonly [2, 3]> = literalTensor.round();
const literalRootSinTensor: Tensor<readonly [2, 3]> = sin(literalTensor);
const literalStaticCosTensor: Tensor<readonly [2, 3]> = Tensor.cos(literalTensor);
const literalInstanceSinTensor: Tensor<readonly [2, 3]> = literalTensor.sin();
const literalRootTanTensor: Tensor<readonly [2, 3]> = tan(literalTensor);
const literalStaticTanTensor: Tensor<readonly [2, 3]> = Tensor.tan(literalTensor);
const literalRootReluTensor: Tensor<readonly [2, 3]> = relu(literalTensor.sub(3));
const literalStaticSigmoidTensor: Tensor<readonly [2, 3]> = Tensor.sigmoid(literalTensor);
const literalRootTanhTensor: Tensor<readonly [2, 3]> = tanh(literalTensor);
const literalStaticTanhTensor: Tensor<readonly [2, 3]> = Tensor.tanh(literalTensor);
const literalTensorTanh: Tensor<readonly [2, 3]> = literalTensor.tanh();
const literalSumTensor: Tensor<readonly [1]> = literalTensor.sum();
const literalRootSumTensor: Tensor<readonly [2, 1]> = sum(literalTensor, 1);
const literalMeanDimTensor: Tensor<readonly [1, 3]> = literalTensor.meanDim(0);
const literalRootMeanTensor: Tensor<readonly [1, 3]> = mean(literalTensor, 0);
const literalProdTensor: Tensor<readonly [1]> = literalTensor.prod();
const literalProdDimTensor: Tensor<readonly [2, 1]> = literalTensor.prod(-1);
const literalProdDimAliasTensor: Tensor<readonly [2, 1]> = literalTensor.prodDim(-1);
const literalRootProdTensor: Tensor<readonly [1, 3]> = prod(literalTensor, 0);
const literalStaticProdTensor: Tensor<readonly [2, 1]> = Tensor.prod(literalTensor, 1);
const literalCumsumTensor: Tensor<readonly [2, 3]> = literalTensor.cumsum(-1);
const literalRootCumsumTensor: Tensor<readonly [2, 3]> = cumsum(literalTensor, 0);
const literalStaticCumsumTensor: Tensor<readonly [2, 3]> = Tensor.cumsum(literalTensor, 1);
const literalStaticMaxTensor: Tensor<readonly [2, 1]> = Tensor.max(literalTensor, 1);
const literalRootArgmaxTensor: Tensor<readonly [2, 1]> = argmax(literalTensor, 1);
const literalAnyDimTensorTyped: Tensor<readonly [2, 1]> = literalTensor.gt(3).anyDim(-1);
const literalStaticAllTensor: Tensor<readonly [1]> = Tensor.all(literalTensor);
const literalSum3dDim0Tensor: Tensor<readonly [1, 3, 4]> = literalTensor3d.sumDim(0);
const literalRootMean3dTrailingTensor: Tensor<readonly [2, 3, 1]> = mean(literalTensor3d, -1);
const literalVarianceTensor: Tensor<readonly [1]> = literalTensor.variance();
const literalVarDimTensor: Tensor<readonly [2, 1]> = literalTensor.var(-1);
const literalRootVarianceTensor: Tensor<readonly [1, 3]> = variance(literalTensor, 0);
const literalStaticStdTensor: Tensor<readonly [2, 1]> = Tensor.std(literalTensor, 1);
const literalRootStdTensor: Tensor<readonly [1]> = std(literalTensor);
const literalNormTensor: Tensor<readonly [1]> = literalTensor.norm();
const literalNormDimTensor: Tensor<readonly [2, 1]> = literalTensor.norm(-1);
const literalRootNormTensor: Tensor<readonly [1, 3]> = norm(literalTensor, 0);
const literalStaticNormTensor: Tensor<readonly [2, 1]> = Tensor.norm(literalTensor, 1);
type LiteralReductionWholeShape = Expect<Equal<ReductionShape<readonly [2, 3]>, readonly [1]>>;
type LiteralReductionRowsShape = Expect<Equal<ReductionShape<readonly [2, 3], 0>, readonly [1, 3]>>;
type LiteralReductionColsShape = Expect<Equal<ReductionShape<readonly [2, 3], -1>, readonly [2, 1]>>;
type LiteralReduction3dDim0Shape = Expect<Equal<ReductionShape<readonly [2, 3, 4], 0>, readonly [1, 3, 4]>>;
type LiteralReduction3dTrailingShape = Expect<Equal<ReductionShape<readonly [2, 3, 4], -1>, readonly [2, 3, 1]>>;
type LiteralReduction4dDim2Shape = Expect<Equal<ReductionShape<readonly [2, 3, 4, 5], -2>, readonly [2, 3, 1, 5]>>;
type LiteralVectorMatrixMatmulShape = Expect<Equal<MatmulShape<readonly [3], readonly [3, 2]>, TensorShapeTuple>>;
const literalRootSoftmaxTensor: Tensor<readonly [2, 3]> = softmax(literalTensor);
const literalRootSoftmaxDimAliasTensor: Tensor<readonly [2, 3]> = softmax(literalTensor, -1);
const literalRootSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = softmax_dim(literalTensor, -1);
const literalInstanceSoftmaxDimAliasTensor: Tensor<readonly [2, 3]> = literalTensor.softmax(-1);
const literalInstanceSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = literalTensor.softmax_dim(-1);
const literalStaticSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = Tensor.softmax_dim(literalTensor, -1);
const literalStaticSoftmaxDimTensor: Tensor<readonly [2, 3]> = Tensor.softmaxDim(literalTensor, -1);
const literalStaticLogSoftmaxTensor: Tensor<readonly [2, 3]> = Tensor.logSoftmax(literalTensor);
const literalStaticLogSoftmaxDimAliasTensor: Tensor<readonly [2, 3]> = Tensor.logSoftmax(literalTensor, -1);
const literalStaticLogSoftmaxSnakeTensor: Tensor<readonly [2, 3]> = Tensor.log_softmax(literalTensor, -1);
const literalStaticLogSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = Tensor.log_softmax_dim(literalTensor, -1);
const literalRootLogSoftmaxTensor: Tensor<readonly [2, 3]> = logSoftmax(literalTensor);
const literalRootLogSoftmaxDimAliasTensor: Tensor<readonly [2, 3]> = logSoftmax(literalTensor, -1);
const literalRootLogSoftmaxSnakeTensor: Tensor<readonly [2, 3]> = log_softmax(literalTensor, -1);
const literalRootLogSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = log_softmax_dim(literalTensor, -1);
const literalInstanceLogSoftmaxDimAliasTensor: Tensor<readonly [2, 3]> = literalTensor.logSoftmax(-1);
const literalInstanceLogSoftmaxSnakeTensor: Tensor<readonly [2, 3]> = literalTensor.log_softmax(-1);
const literalInstanceLogSoftmaxSnakeDimTensor: Tensor<readonly [2, 3]> = literalTensor.log_softmax_dim(-1);
const literalInstanceLogsumexpTensor: Tensor<readonly [1]> = literalTensor.logsumexp();
const literalInstanceLogsumexpDimTensor: Tensor<readonly [2, 1]> = literalTensor.logsumexp(-1);
const literalRootLogsumexpTensor: Tensor<readonly [1, 3]> = logsumexp(literalTensor, 0);
const literalStaticLogSumExpTensor: Tensor<readonly [2, 1]> = Tensor.logSumExp(literalTensor, 1);
const literalRootLogSumExpTensor: Tensor<readonly [1]> = logSumExp(literalTensor);
const literalRootClampTensor: Tensor<readonly [2, 3]> = clamp(literalTensor, 0, 1);
const literalStaticClipTensor: Tensor<readonly [2, 3]> = Tensor.clip(literalTensor, null, 1);
const literalMatmulTensor: Tensor<readonly [2, 2]> = literalTensor.matmul(tensor([1, 2, 3, 4, 5, 6], [3, 2] as const));
const literalRootMatmulTensor: Tensor<readonly [2, 2]> = matmul(literalTensor, tensor([1, 2, 3, 4, 5, 6], [3, 2] as const));
const literalRootMmTensor: Tensor<readonly [2, 2]> = mm(literalTensor, tensor([1, 2, 3, 4, 5, 6], [3, 2] as const));
const literalStaticMatmulTensor: Tensor<readonly [2, 2]> = Tensor.matmul(literalTensor, tensor([1, 2, 3, 4, 5, 6], [3, 2] as const));
const literalStaticMmTensor: Tensor<readonly [2, 2]> = Tensor.mm(literalTensor, tensor([1, 2, 3, 4, 5, 6], [3, 2] as const));
const literalDotTensor: Tensor<readonly [1]> = literalSelectRowsTensor.dot(literalSelectRowsTensor);
const literalRootDotTensor: Tensor<readonly [1]> = dot(literalSelectRowsTensor, literalSelectRowsTensor);
const literalStaticDotTensor: Tensor<readonly [1]> = Tensor.dot(literalSelectRowsTensor, literalSelectRowsTensor);
const literalTraceTensor: Tensor<readonly [1]> = literalTensor.trace();
const literalRootTraceTensor: Tensor<readonly [1]> = trace(literalTensor);
const literalStaticTraceTensor: Tensor<readonly [1]> = Tensor.trace(literalTensor);
const literalDiagonalTensor: Tensor<readonly [2]> = literalTensor.diagonal();
const literalRootDiagonalTensor: Tensor<readonly [2]> = diagonal(literalTensor);
const literalStaticDiagonalTensor: Tensor<readonly [2]> = Tensor.diagonal(literalTensor);
type LiteralMatrixMatmulShape = Expect<Equal<MatmulShape<readonly [2, 3], readonly [3, 2]>, readonly [2, 2]>>;
const literalBmmLhs = tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2] as const);
const literalBmmRhs = tensor([1, 0, 0, 1, 2, 0, 0, 2], [2, 2, 2] as const);
const literalBmmTensor: Tensor<readonly [2, 2, 2]> = literalBmmLhs.bmm(literalBmmRhs);
const literalRootBmmTensor: Tensor<readonly [2, 2, 2]> = bmm(literalBmmLhs, literalBmmRhs);
const literalStaticBmmTensor: Tensor<readonly [2, 2, 2]> = Tensor.bmm(literalBmmLhs, literalBmmRhs);
type LiteralBmmShape = Expect<Equal<BmmShape<readonly [2, 2, 3], readonly [2, 3, 4]>, readonly [2, 2, 4]>>;
const literalMaximumTensor: Tensor<readonly [2, 3]> = literalTensor.maximum(tensor([3, 3, 3], [1, 3] as const));
const literalMaximumScalarTensor: Tensor<readonly [2, 3]> = literalTensor.maximum(3);
const literalRootMaximumSameShapeTensor: Tensor<readonly [2, 3]> = maximum(literalTensor, literalSameShapeTensor);
const literalMinimumTensor: Tensor<readonly [2, 3]> = minimum(literalTensor, 3);
const literalStaticMinimumSameShapeTensor: Tensor<readonly [2, 3]> = Tensor.minimum(literalTensor, literalSameShapeTensor);
const literalWhereTensor: Tensor<readonly [2, 3]> = literalTensor.gt(3).where(1, 0);
const literalRootWhereTensor: Tensor<readonly [2, 3]> = where(literalTensor.lt(3), literalTensor, tensor([0, 0, 0], [1, 3] as const));
const literalRootWhereSameShapeTensor: Tensor<readonly [2, 3]> = where(literalTensor.lt(3), literalTensor, literalSameShapeTensor);
const literalInstanceMaskedFillTensor: Tensor<readonly [2, 3]> = literalTensor.maskedFill(literalTensor.gt(3), -1);
const literalInstanceMaskedFillSnakeTensor: Tensor<readonly [2, 3]> = literalTensor.masked_fill(literalTensor.gt(3), -1);
const literalRootMaskedFillTensor: Tensor<readonly [2, 3]> = maskedFill(literalTensor, literalTensor.gt(3), -1);
const literalRootMaskedFillSnakeTensor: Tensor<readonly [2, 3]> = masked_fill(literalTensor, literalTensor.gt(3), -1);
const literalStaticMaskedFillTensor: Tensor<readonly [2, 3]> = Tensor.maskedFill(literalTensor, literalTensor.gt(3), -1);
const literalStaticMaskedFillSnakeTensor: Tensor<readonly [2, 3]> = Tensor.masked_fill(literalTensor, literalTensor.gt(3), -1);
const literalAnyTensor: Tensor = literalTensor.gt(3).any();
const literalAnyDimTensor: Tensor = literalTensor.gt(3).anyDim(1);
const literalAllTensor: Tensor = all(literalTensor.gt(0));
const literalAllDimTensor: Tensor = any(literalTensor.gt(3), 0).allDim(0);
const literalRootCatDim0Tensor: Tensor<readonly [4, 3]> = cat([literalTensor, literalTensor] as const, 0);
const literalStaticCatDim1Tensor: Tensor<readonly [2, 6]> = Tensor.cat([literalTensor, literalTensor] as const, 1);
const literalRootConcatDim0Tensor: Tensor<readonly [4, 3]> = concat([literalTensor, literalTensor] as const, 0);
const literalStaticConcatDim1Tensor: Tensor<readonly [2, 6]> = Tensor.concat([literalTensor, literalTensor] as const, 1);
const literalRootConcatenateDim0Tensor: Tensor<readonly [4, 3]> = concatenate([literalTensor, literalTensor] as const, 0);
const literalStaticConcatenateDim1Tensor: Tensor<readonly [2, 6]> = Tensor.concatenate([literalTensor, literalTensor] as const, 1);
const literalRootCat3dDim2Tensor: Tensor<readonly [2, 3, 8]> = cat([literalTensor3d, literalTensor3d] as const, 2);
const literalStaticCat4dTrailingTensor: Tensor<readonly [2, 3, 4, 10]> = Tensor.cat([literalTensor4d, literalTensor4d] as const, -1);
const literalIndexSelectIndices = [2, 0] as const satisfies IndexLike;
const literalIndexSelectTensorIndices = tensor([1, 0], [2] as const);
const literalInstanceIndexSelectTensor: Tensor<readonly [2, 2]> = literalTensor.indexSelect(1, literalIndexSelectIndices);
const literalInstanceIndexSelectSnakeTensor: Tensor<readonly [2, 3]> = literalTensor.index_select(0, literalIndexSelectTensorIndices);
const literalRootIndexSelectTensor: Tensor<readonly [2, 2]> = indexSelect(literalTensor, 1, literalIndexSelectIndices);
const literalRootIndexSelectSnakeTensor: Tensor<readonly [2, 3]> = index_select(literalTensor, 0, literalIndexSelectTensorIndices);
const literalStaticIndexSelectTensor: Tensor<readonly [2, 2]> = Tensor.indexSelect(literalTensor, 1, literalIndexSelectIndices);
const literalStaticIndexSelectSnakeTensor: Tensor<readonly [2, 3]> = Tensor.index_select(literalTensor, 0, literalIndexSelectTensorIndices);
type LiteralIndexSelectDim1Shape = Expect<Equal<TensorIndexSelectShape<readonly [2, 3], 1, typeof literalIndexSelectIndices>, readonly [2, 2]>>;
type LiteralIndexSelectTensorIndexShape = Expect<Equal<TensorIndexSelectShape<readonly [2, 3], 0, typeof literalIndexSelectTensorIndices>, readonly [2, 3]>>;
const literalGatherIndex = tensor([2, 1, 0, 0], [2, 2] as const);
const literalInstanceGatherTensor: Tensor<readonly [2, 2]> = literalTensor.gather(1, literalGatherIndex);
const literalRootGatherTensor: Tensor<readonly [2, 2]> = gather(literalTensor, 1, literalGatherIndex);
const literalStaticGatherTensor: Tensor<readonly [2, 2]> = Tensor.gather(literalTensor, 1, literalGatherIndex);
type LiteralGatherShape = Expect<Equal<TensorGatherShape<typeof literalGatherIndex>, readonly [2, 2]>>;
const literalTakeIndex = tensor([5, 0, 3, 3], [2, 2] as const);
const literalInstanceTakeTensor: Tensor<readonly [2, 2]> = literalTensor.take(literalTakeIndex);
const literalRootTakeTensor: Tensor<readonly [2, 2]> = take(literalTensor, literalTakeIndex);
const literalStaticTakeTensor: Tensor<readonly [2, 2]> = Tensor.take(literalTensor, literalTakeIndex);
const literalInstanceArgsortTensor: Tensor<readonly [2, 3]> = literalTensor.argsort(-1);
const literalRootArgsortTensor: Tensor<readonly [2, 3]> = argsort(literalTensor, 1, true);
const literalStaticArgsortTensor: Tensor<readonly [2, 3]> = Tensor.argsort(literalTensor);
const literalInstanceSortResult: TensorSortResult<readonly [2, 3]> = literalTensor.sort(-1);
const literalRootSortResult: TensorSortResult<readonly [2, 3]> = sort(literalTensor, 1, true);
const literalStaticSortResult: TensorSortResult<readonly [2, 3]> = Tensor.sort(literalTensor);
const literalInstanceTopkResult: TensorTopkResult<readonly [2, 2]> = literalTensor.topk(2, -1);
const literalRootTopkResult: TensorTopkResult<readonly [1, 3]> = topk(literalTensor, 1, 0, false, false);
const literalStaticTopkResult: TensorTopkResult<readonly [2, 1]> = Tensor.topk(literalTensor, 1, 1);
type LiteralTopkShape = Expect<Equal<TensorTopkShape<readonly [2, 3], -1, 2>, readonly [2, 2]>>;
const literalInstanceScatterAddTensor: Tensor<readonly [2, 3]> = literalTensor.scatterAdd(1, literalGatherIndex, tensor([1, 1, 1, 1], [2, 2] as const));
const literalInstanceScatterAddSnakeTensor: Tensor<readonly [2, 3]> = literalTensor.scatter_add(1, literalGatherIndex, 1);
const literalRootScatterAddTensor: Tensor<readonly [2, 3]> = scatterAdd(literalTensor, 1, literalGatherIndex, tensor([1, 1, 1, 1], [2, 2] as const));
const literalRootScatterAddSnakeTensor: Tensor<readonly [2, 3]> = scatter_add(literalTensor, 1, literalGatherIndex, 1);
const literalStaticScatterAddTensor: Tensor<readonly [2, 3]> = Tensor.scatterAdd(literalTensor, 1, literalGatherIndex, tensor([1, 1, 1, 1], [2, 2] as const));
const literalStaticScatterAddSnakeTensor: Tensor<readonly [2, 3]> = Tensor.scatter_add(literalTensor, 1, literalGatherIndex, 1);
type LiteralCatDim0Shape = Expect<Equal<TensorCatShape<readonly [typeof literalTensor, typeof literalTensor], 0>, readonly [4, 3]>>;
type LiteralCatDim1Shape = Expect<Equal<TensorCatShape<readonly [typeof literalTensor, typeof literalTensor], 1>, readonly [2, 6]>>;
type LiteralCat3dDim1Shape = Expect<Equal<TensorCatShape<readonly [typeof literalTensor3d, typeof literalTensor3d], 1>, readonly [2, 6, 4]>>;
type LiteralCat4dTrailingShape = Expect<Equal<TensorCatShape<readonly [typeof literalTensor4d, typeof literalTensor4d], -1>, readonly [2, 3, 4, 10]>>;
const literalRootStackTensor: Tensor<readonly [2, 2, 3]> = stack([literalTensor, literalTensor] as const);
const literalRootStackDim0Tensor: Tensor<readonly [2, 2, 3]> = stack([literalTensor, literalTensor] as const, 0);
const literalStaticStackTensor: Tensor<readonly [2, 2, 3]> = Tensor.stack([literalTensor, literalTensor] as const);
const literalRootStackDim1Tensor: Tensor<readonly [2, 2, 3]> = stack([literalTensor, literalTensor] as const, 1);
const literalStaticStackTrailingTensor: Tensor<readonly [2, 3, 2]> = Tensor.stack([literalTensor, literalTensor] as const, -1);
const literalRootStack3dDim2Tensor: Tensor<readonly [2, 3, 2, 4]> = stack([literalTensor3d, literalTensor3d] as const, 2);
const literalStaticStack3dTrailingTensor: Tensor<readonly [2, 3, 4, 2]> = Tensor.stack([literalTensor3d, literalTensor3d] as const, -1);
const literalRootStack4dDim2Tensor: Tensor<readonly [2, 3, 2, 4, 5]> = stack([literalTensor4d, literalTensor4d] as const, 2);
const literalStaticMax4dDim2Tensor: Tensor<readonly [2, 3, 1, 2]> = Tensor.max(literalStaticStack3dTrailingTensor, -2);
const literalRowTensor = tensor([1, 2, 3], [3] as const);
const literalRootVStackRowsTensor: Tensor<readonly [2, 3]> = vstack([literalRowTensor, literalRowTensor] as const);
const literalStaticVStackMatrixTensor: Tensor<readonly [4, 3]> = Tensor.vstack([literalTensor, literalTensor] as const);
const literalRootHStackRowsTensor: Tensor<readonly [6]> = hstack([literalRowTensor, literalRowTensor] as const);
const literalStaticHStackMatrixTensor: Tensor<readonly [2, 6]> = Tensor.hstack([literalTensor, literalTensor] as const);
type LiteralStackShape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor, typeof literalTensor]>, readonly [2, 2, 3]>>;
type LiteralStackDim1Shape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor, typeof literalTensor], 1>, readonly [2, 2, 3]>>;
type LiteralStackTrailingShape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor, typeof literalTensor], -1>, readonly [2, 3, 2]>>;
type LiteralStack3dDim2Shape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor3d, typeof literalTensor3d], 2>, readonly [2, 3, 2, 4]>>;
type LiteralStack3dTrailingShape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor3d, typeof literalTensor3d], -1>, readonly [2, 3, 4, 2]>>;
type LiteralStack4dDim2Shape = Expect<Equal<TensorStackShape<readonly [typeof literalTensor4d, typeof literalTensor4d], 2>, readonly [2, 3, 2, 4, 5]>>;
type LiteralVStackRowsShape = Expect<Equal<TensorVStackShape<readonly [typeof literalRowTensor, typeof literalRowTensor]>, readonly [2, 3]>>;
type LiteralVStackMatrixShape = Expect<Equal<TensorVStackShape<readonly [typeof literalTensor, typeof literalTensor]>, readonly [4, 3]>>;
type LiteralHStackRowsShape = Expect<Equal<TensorHStackShape<readonly [typeof literalRowTensor, typeof literalRowTensor]>, readonly [6]>>;
type LiteralHStackMatrixShape = Expect<Equal<TensorHStackShape<readonly [typeof literalTensor, typeof literalTensor]>, readonly [2, 6]>>;
const reshaped = literalTensor.reshape([3, 2] as const);
const reshapedShape: readonly [3, 2] = reshaped.shape;
const inferredReshaped: Tensor<readonly [3, 2]> = literalTensor.reshape([3, -1] as const);
const inferredViewed: Tensor<readonly [1, 6]> = literalTensor.view([1, -1] as const);
type LiteralReshapeInferShape = Expect<Equal<ReshapeShape<readonly [2, 3], readonly [3, -1]>, readonly [3, 2]>>;
type LiteralReshapeInfer4dShape = Expect<Equal<ReshapeShape<readonly [2, 3, 4, 5], readonly [2, -1, 5]>, readonly [2, 12, 5]>>;
const scalarTensor = Tensor.scalar(1);
const scalarShape: readonly [1] = scalarTensor.shape;
const scalarValue: number = scalarTensor.item();
const scalarNumberValue: number = scalarTensor.toNumber();
const literalTensorDim: number = literalTensor.dim();
const literalTensorNdimension: number = literalTensor.ndimension();
const literalTensorNumel: number = literalTensor.numel();
const literalTensorSize: readonly number[] = literalTensor.size();
const literalTensorSizeDim: number = literalTensor.size(-1);
const literalTensorStride: readonly number[] = literalTensor.stride();
const literalTensorStrideDim: number = literalTensor.stride(0);
const literalTensorStorageOffset: number = literalTensor.storage_offset();
const literalTensorIsContiguous: boolean = literalTensor.isContiguous();
const literalTensorIsContiguousAlias: boolean = literalTensor.is_contiguous();
const literalTensorContiguous: typeof literalTensor = literalTensor.contiguous();
const literalTensorElementSizeAlias: number = literalTensor.element_size();
const literalTensorNbytes: number = literalTensor.nbytes();
const literalRootLinspaceTensor: Tensor<readonly [4]> = linspace(0, 1, 4);
const literalStaticLinspaceTensor: Tensor<readonly [4]> = Tensor.linspace(0, 1, 4);
const literalRootArangeTensor: Tensor<readonly [4]> = arange(4);
const literalStaticArangeRangeTensor: Tensor<readonly [3]> = Tensor.arange(2, 5);
type LiteralArangeShape = Expect<Equal<ArangeShape<4>, readonly [4]>>;
type LiteralArangeRangeShape = Expect<Equal<ArangeRangeShape<2, 5>, readonly [3]>>;
const scalarValueOf: number = scalarTensor.valueOf();
const scalarPrimitive: number = scalarTensor[Symbol.toPrimitive]();
const zerosTensor = zeros([2, 2] as const);
const zerosShape: readonly [2, 2] = zerosTensor.shape;
const zerosHasShape: boolean = hasShape(zerosTensor, [2, 2] as const);
if (hasShape(zerosTensor, [2, 2] as const)) {
  const narrowedZerosShape: readonly [2, 2] = zerosTensor.shape;
  void narrowedZerosShape;
}
const requiredZerosShapeTensor: Tensor<readonly [2, 2]> = requireShape(zerosTensor, [2, 2] as const);
const requiredZerosShapeFromStatic: Tensor<readonly [2, 2]> = Tensor.requireShape(zerosTensor, [2, 2] as const);
const staticZerosHasShape: boolean = Tensor.hasShape(zerosTensor, [2, 2] as const);
const manualSeedValue: number = manualSeed(123);
const manualSeedAliasValue: number = manual_seed(123);
const initialSeedValue: number | null = initialSeed();
const initialSeedAliasValue: number | null = initial_seed();
const seededRngValue: number = seededRng(123)();
const seededRandTensor: Tensor<readonly [2]> = Tensor.rand([2] as const, { seed: 123 });
const seededRandnTensor: Tensor<readonly [2]> = Tensor.randn([2] as const, { seed: 123, mean: 0, std: 1 });
const tensorManualSeedValue: number = Tensor.manualSeed(456);
const tensorManualSeedAliasValue: number = Tensor.manual_seed(456);
const tensorInitialSeedValue: number | null = Tensor.initialSeed();
const tensorInitialSeedAliasValue: number | null = Tensor.initial_seed();
const gradEnabledAlias: boolean = is_grad_enabled();
const previousGradEnabledAlias: boolean = set_grad_enabled(gradEnabledAlias);
const noGradAliasValue: number = no_grad(() => 1);
const inferenceModeAliasValue: number = inference_mode(() => 2);
const enableGradAliasValue: number = enable_grad(() => 3);
const rootGradModeNamespace: GradModeNamespace = gradMode;
const rootGradModeEnabled: boolean = gradMode.isGradEnabled();
const rootGradModeNoGradValue: number = gradMode.noGrad(() => 4);

const linear = nn.linear(2, 3);
const linearLiteralInFeatures: 2 = linear.inFeatures;
const linearLiteralOutFeatures: 3 = linear.outFeatures;
const linearTypedForward: Tensor<readonly [3]> = linear.forward(tensor([1, 2], [2] as const));
const linearTypedBatchForward: Tensor<readonly [4, 3]> = linear.forward(tensor(new Float32Array(8), [4, 2] as const));
const linearPlainTupleForward: Tensor<readonly [3]> = linear.forward([1, 2] as const);
const linearPlainBatchTupleForward: Tensor<readonly [2, 3]> = linear.forward([[1, 2], [3, 4]] as const);
const linearSequentialPlainBatchFactory = nn.sequential as unknown as (layers: readonly [typeof linear]) => { forward(input: readonly [readonly [1, 2], readonly [3, 4]]): Tensor<readonly [2, 3]> };
const linearSequentialPlainBatch = linearSequentialPlainBatchFactory([linear] as const);
const linearSequentialPlainBatchForward: Tensor<readonly [2, 3]> = linearSequentialPlainBatch.forward([[1, 2], [3, 4]] as const);
const linearPlainBatchTypedForward: Tensor<readonly [number, 3]> = linear.forward([[1, 2], [3, 4]] as readonly (readonly [number, number])[]);
type LinearSequentialBatchForwardShape = Expect<Equal<SequentialForwardShape<readonly [typeof linear], readonly [2, 2]>, readonly [2, 3]>>;
type PlainTupleTensorLikeShape = Expect<Equal<TensorLikeShape<readonly [1, 2]>, readonly [2]>>;
type PlainBatchTensorLikeShape = Expect<Equal<TensorLikeShape<readonly [readonly [1, 2], readonly [3, 4]]>, readonly [2, 2]>>;
type LinearVectorForwardShape = Expect<Equal<LinearForwardShape<readonly [2], 3>, readonly [3]>>;
type LinearBatchForwardShape = Expect<Equal<LinearForwardShape<readonly [4, 2], 3>, readonly [4, 3]>>;
const customChild = nn.linear(2, 1);
const customModuleConfig: NnModuleConfig<readonly [2], readonly [1]> = {
  kind: "custom-head",
  children: [customChild],
  graph: customChild,
  parameters: (prefix = "") => customChild.parameters(prefix),
  forward(input: Tensor<readonly [2]>) {
    return customChild.call(input);
  },
};
const customModule: CustomModule<readonly [2], readonly [1]> = nn.module(customModuleConfig);
const customModuleForward: Tensor<readonly [1]> = customModule.forward(tensor([1, 2], [2] as const));
const customModuleCall: Tensor<readonly [1]> = customModule.call(tensor([1, 2], [2] as const));
const customModuleDunderCall: Tensor<readonly [1]> = customModule.__call__(tensor([1, 2], [2] as const));
const customModuleNamespaceCall: Tensor<readonly [1]> = nn.call(customModule, tensor([1, 2], [2] as const));
const customModuleParameters: NnParameter[] = customModule.parameters();
const customModuleChildren: readonly NnModule[] = customModule.children();
const customModuleState: ModuleStateSnapshot = customModule.stateDict("custom");
const customModuleOptimizer: Optimizer<"sgd"> = optim.sgd(customModule, { lr: 0.01 });
const customModuleProgram: Program<readonly [2], readonly [1]> = customModule.compile({ inputShape: [2] as const });
const customModuleSession: Session<readonly [2], readonly [1]> = customModuleProgram.bindModule(customModule);
const customModuleCompiledForward: Tensor<readonly [1]> = customModuleSession.stepTensor(tensor([1, 2], [2] as const));
customModuleOptimizer.zero_grad();
class CustomClassModule extends nn.Module<readonly [2], readonly [1]> {
  readonly graph = customChild;

  forward(input: Tensor<readonly [2]>): Tensor<readonly [1]> {
    return this.graph.forward(input);
  }
}
const customClassModule: ModuleBase<readonly [2], readonly [1]> = new CustomClassModule({ kind: "custom-class-head" });
const customClassForward: Tensor<readonly [1]> = customClassModule.forward(tensor([1, 2], [2] as const));
const customClassNames: readonly string[] = customClassModule.parameterNames("customClass");
const customClassProgram: Program<readonly [2], readonly [1]> = customClassModule.compile({ inputShape: [2] as const });
const customClassSession: Session<readonly [2], readonly [1]> = customClassProgram.bindModule(customClassModule);
const customClassCompiledForward: Tensor<readonly [1]> = customClassSession.stepTensor(tensor([1, 2], [2] as const));
class FieldOwnedModule extends nn.Module<readonly [2], readonly [1]> {
  readonly head = nn.linear(2, 1);

  forward(input: Tensor<readonly [2]>): Tensor<readonly [1]> {
    return this.head.forward(input);
  }
}
const fieldOwnedModule = new FieldOwnedModule({ kind: "field-owned-head" });
const fieldOwnedChildren: readonly NnModule[] = fieldOwnedModule.children();
const fieldOwnedNamedChildren: readonly ModuleTraversalEntry[] = fieldOwnedModule.namedChildren("fieldOwned");
const fieldOwnedTraversalOptions: ModuleParameterTraversalOptions = { recurse: false };
const fieldOwnedOwnParameters: NnParameter[] = fieldOwnedModule.parameters(fieldOwnedTraversalOptions);
const fieldOwnedOwnNamedParameters: NnParameter[] = fieldOwnedModule.namedParameters("fieldOwned", fieldOwnedTraversalOptions);
const fieldOwnedNamespaceOwnParameters: NnParameter[] = nn.parameters(fieldOwnedModule, { recurse: false });
const fieldOwnedNamespaceOwnNamedParameters: NnParameter[] = nn.namedParameters(fieldOwnedModule, "fieldOwned", { recurse: false });
const fieldOwnedNames: readonly string[] = fieldOwnedModule.parameterNames("fieldOwned");
const fieldOwnedProgram: Program<readonly [2], readonly [1]> = fieldOwnedModule.compile({ inputShape: [2] as const });
const fieldOwnedSession: Session<readonly [2], readonly [1]> = fieldOwnedProgram.bindModule(fieldOwnedModule);
const fieldOwnedCompiledForward: Tensor<readonly [1]> = fieldOwnedSession.stepTensor(tensor([1, 2], [2] as const));
const customParameterOptions: NnParameterOptions<readonly [2]> = { layout: "row-major" };
const customWeightParameter: NnParameter<readonly [2]> = nn.parameter("weight", tensor([0.5, -0.5], [2] as const), customParameterOptions);
const customBiasParameter: NnParameter<readonly [1]> = nn.Parameter("bias", new Float32Array([0.1]), [1] as const);
const initWeightParameter: NnParameter<readonly [2, 2]> = nn.Parameter("initWeight", tensor([0, 0, 0, 0], [2, 2] as const));
const initTensorTarget: Tensor<readonly [2, 2]> = tensor([0, 0, 0, 0], [2, 2] as const);
const initConstantParameter: NnParameter<readonly [2, 2]> = nn.init.constant_(initWeightParameter, 0.25);
const initZerosParameter: NnParameter<readonly [2, 2]> = nn.init.zeros_(initWeightParameter);
const initOnesTensor: Tensor<readonly [2, 2]> = nn.init.ones_(initTensorTarget);
const initUniformParameter: NnParameter<readonly [2, 2]> = nn.init.uniform_(initWeightParameter, -0.1, 0.1);
const initNormalParameter: NnParameter<readonly [2, 2]> = nn.init.normal_(initWeightParameter, 0, 0.02);
const initXavierParameter: NnParameter<readonly [2, 2]> = nn.init.xavier_uniform_(initWeightParameter);
const initXavierCamelParameter: NnParameter<readonly [2, 2]> = nn.init.xavierUniform_(initWeightParameter);
const initXavierNormalParameter: NnParameter<readonly [2, 2]> = nn.init.xavier_normal_(initWeightParameter);
const initXavierNormalCamelParameter: NnParameter<readonly [2, 2]> = nn.init.xavierNormal_(initWeightParameter);
const initKaimingOptions: KaimingUniformOptions = { mode: "fan_in", nonlinearity: "relu" };
const initKaimingParameter: NnParameter<readonly [2, 2]> = nn.init.kaiming_uniform_(initWeightParameter, initKaimingOptions);
const initKaimingCamelParameter: NnParameter<readonly [2, 2]> = nn.init.kaimingUniform_(initWeightParameter, { mode: "fanOut", nonlinearity: "leaky_relu", negativeSlope: 0.2 });
const initKaimingNormalOptions: KaimingNormalOptions = { mode: "fan_in", nonlinearity: "relu" };
const initKaimingNormalParameter: NnParameter<readonly [2, 2]> = nn.init.kaiming_normal_(initWeightParameter, initKaimingNormalOptions);
const initKaimingNormalCamelParameter: NnParameter<readonly [2, 2]> = nn.init.kaimingNormal_(initWeightParameter, { mode: "fanOut", nonlinearity: "leaky_relu", negativeSlope: 0.2 });
const einsumRootTensor: Tensor = einsum("ij,jk->ik", [tensor([1, 2, 3, 4], [2, 2] as const), tensor([5, 6, 7, 8], [2, 2] as const)]);
const einsumStaticTensor: Tensor = Tensor.einsum("ii->", tensor([1, 2, 3, 4], [2, 2] as const));
const einsumZgmlTensor: Tensor = zgml.einsum("ij,jk->ik", tensor([1, 2, 3, 4], [2, 2] as const), tensor([5, 6, 7, 8], [2, 2] as const));
const einsumTorchTensor: Tensor = torch.einsum("ij,jk->ik", tensor([1, 2, 3, 4], [2, 2] as const), tensor([5, 6, 7, 8], [2, 2] as const));
const einsumEllipsisTensor: Tensor = einsum("...ij,jk->...ik", tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2] as const), tensor([1, 2, 3, 4], [2, 2] as const));
const einsumImplicitEllipsisTensor: Tensor = Tensor.einsum("...i->...", tensor([1, 2, 3, 4, 5, 6], [2, 3] as const));
const typedEinsumRootTensor: Tensor<readonly [2, 2]> = einsum("ij,jk->ik", [tensor([1, 2, 3, 4], [2, 2] as const), tensor([5, 6, 7, 8], [2, 2] as const)]);
const typedEinsumStaticTraceTensor: Tensor<readonly [1]> = Tensor.einsum("ii->", tensor([1, 2, 3, 4], [2, 2] as const));
const typedEinsumEllipsisTensor: Tensor<readonly [2, 2, 2]> = einsum("...ij,jk->...ik", tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2] as const), tensor([1, 2, 3, 4], [2, 2] as const));
const typedEinsumImplicitEllipsisTensor: Tensor<readonly [2]> = Tensor.einsum("...i->...", tensor([1, 2, 3, 4, 5, 6], [2, 3] as const));
type TypedEinsumMatmulShape = Expect<Equal<EinsumShape<"ij,jk->ik", readonly [Tensor<readonly [2, 3]>, Tensor<readonly [3, 4]>]>, readonly [2, 4]>>;
type TypedEinsumBatchBroadcastShape = Expect<Equal<EinsumShape<"bij,bjk->bik", readonly [Tensor<readonly [1, 2, 3]>, Tensor<readonly [4, 3, 5]>]>, readonly [4, 2, 5]>>;
type TypedEinsumImplicitEllipsisShape = Expect<Equal<EinsumShape<"...i->...", readonly [Tensor<readonly [2, 3, 4]>]>, readonly [2, 3]>>;
const customParameterModule: CustomModule<readonly [2], readonly [1]> = nn.module({
  kind: "custom-parameter-head",
  parameters: [customWeightParameter, customBiasParameter],
  forward(_input: Tensor<readonly [2]>) {
    return tensor([customBiasParameter.data[0] ?? 0], [1] as const);
  },
});
const customParameterState: ModuleStateSnapshot = customParameterModule.stateDict("head");
const customParameterOptimizer: Optimizer<"sgd"> = optim.sgd(customParameterModule, { lr: 0.01 });
customParameterOptimizer.zero_grad();
class FieldParameterModule extends nn.Module<readonly [1], readonly [1]> {
  readonly weight = nn.Parameter("weight", tensor([0.5], [1] as const));
  readonly bias = nn.Parameter("bias", tensor([0.1], [1] as const));

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return input.mul(this.weight.tensor).add(this.bias.tensor);
  }
}
const fieldParameterModule = new FieldParameterModule({ kind: "field-parameter-head" });
const fieldParameterNames: readonly string[] = fieldParameterModule.parameterNames("fieldParameter");
const fieldParameterParameters: NnParameter[] = fieldParameterModule.parameters();
const fieldParameterState: ModuleStateSnapshot = fieldParameterModule.stateDict("fieldParameter");
const fieldParameterOptimizer: Optimizer<"sgd"> = optim.sgd(fieldParameterModule, { lr: 0.01 });
fieldParameterOptimizer.zeroGrad();
class RegisteredModule extends nn.Module<readonly [2], readonly [1]> {
  declare head: ReturnType<typeof nn.linear<2, 1>>;
  declare scale: NnParameter<readonly [1]>;
  declare runningScale: NnBuffer<readonly [1]>;

  constructor() {
    super({ kind: "registered-head" });
    this.addModule("head", nn.linear(2, 1));
    this.registerModule("alias", nn.relu());
    this.register_module("aliasSnake", nn.sigmoid());
    this.registerParameter("scale", nn.Parameter("scale", tensor([1], [1] as const)));
    this.register_parameter("biasShift", nn.parameter("biasShift", tensor([0], [1] as const)));
    this.registerBuffer("runningScale", nn.Buffer("runningScale", tensor([1], [1] as const)));
  }

  forward(input: Tensor<readonly [2]>): Tensor<readonly [1]> {
    return this.head.forward(input).mul(this.scale.tensor);
  }
}
const registeredModule = new RegisteredModule();
const registeredModuleNames: readonly string[] = registeredModule.parameterNames("registered");
const registeredModuleBufferTraversalOptions: ModuleBufferTraversalOptions = { recurse: false };
const registeredModuleBuffers: NnBuffer[] = registeredModule.namedBuffers("registered");
const registeredModuleOwnBuffers: NnBuffer[] = registeredModule.buffers(registeredModuleBufferTraversalOptions);
const registeredModuleOwnNamedBuffers: NnBuffer[] = registeredModule.namedBuffers("registered", registeredModuleBufferTraversalOptions);
const registeredModuleBufferAliases: NnBuffer[] = registeredModule.named_buffers("registered");
const registeredModuleNamespaceBuffers: NnBuffer[] = nn.namedBuffers(registeredModule, "registered");
const registeredModuleNamespaceOwnBuffers: NnBuffer[] = nn.buffers(registeredModule, { recurse: false });
const registeredModuleNamespaceOwnNamedBuffers: NnBuffer[] = nn.namedBuffers(registeredModule, "registered", { recurse: false });
const registeredModuleNamespaceBufferAlias: NnBuffer[] = nn.named_buffers(registeredModule, "registered");
const registeredModuleHeadLookup: NnModule | null = registeredModule.getSubmodule("head");
const registeredModuleHeadLookupAlias: NnModule | null = registeredModule.get_submodule("head");
const registeredModuleNamespaceHeadLookup: NnModule | null = nn.getSubmodule(registeredModule, "head");
const registeredModuleNamespaceHeadLookupAlias: NnModule | null = nn.get_submodule(registeredModule, "head");
const registeredModuleScaleLookup: NnParameter | null = registeredModule.getParameter("scale");
const registeredModuleScaleLookupAlias: NnParameter | null = registeredModule.get_parameter("scale");
const registeredModuleNamespaceScaleLookup: NnParameter | null = nn.getParameter(registeredModule, "scale");
const registeredModuleNamespaceScaleLookupAlias: NnParameter | null = nn.get_parameter(registeredModule, "scale");
const registeredModuleBufferLookup: NnBuffer | null = registeredModule.getBuffer("runningScale");
const registeredModuleBufferLookupAlias: NnBuffer | null = registeredModule.get_buffer("runningScale");
const registeredModuleNamespaceBufferLookup: NnBuffer | null = nn.getBuffer(registeredModule, "runningScale");
const registeredModuleNamespaceBufferLookupAlias: NnBuffer | null = nn.get_buffer(registeredModule, "runningScale");
const registeredModuleChildren: readonly NnModule[] = registeredModule.children();
const registeredModuleState: ModuleStateSnapshot = registeredModule.stateDict("registered");
const registeredModuleStateDictOptions: ModuleStateDictOptions = { prefix: "registered" };
const registeredModuleOptionsState: ModuleStateSnapshot = registeredModule.stateDict(registeredModuleStateDictOptions);
const registeredModuleOptionsStateAlias: ModuleStateSnapshot = registeredModule.state_dict(registeredModuleStateDictOptions);
const registeredModuleNamespaceStateWithBuffers: ModuleStateSnapshot = nn.stateDict(registeredModule, "registered");
const registeredModuleNamespaceOptionsStateWithBuffers: ModuleStateSnapshot = nn.stateDict(registeredModule, registeredModuleStateDictOptions);
const registeredModuleNamespaceOptionsStateAliasWithBuffers: ModuleStateSnapshot = nn.state_dict(registeredModule, registeredModuleStateDictOptions);
const registeredModuleLoadedBuffers: RegisteredModule = registeredModule.loadStateDict(registeredModuleState, { prefix: "registered", validateOnly: true });
const registeredModuleNamespaceLoadedBuffers: RegisteredModule = nn.loadStateDict(registeredModule, registeredModuleNamespaceStateWithBuffers, {
  prefix: "registered",
  validateOnly: true,
});
const registeredModuleCheckpointWithBuffers: ZgmlCheckpoint = checkpoint.create({ model: registeredModule, prefix: "registered" });
const registeredModuleCheckpointInspection: CheckpointInspection = checkpoint.inspect(registeredModuleCheckpointWithBuffers);
const registeredModuleCheckpointBufferInfo = checkpoint.modelParameterInfo(registeredModuleCheckpointWithBuffers, "registered.runningScale");
const registeredModuleCheckpointLoadedBuffers = checkpoint.load(registeredModuleCheckpointWithBuffers, {
  model: registeredModule,
  prefix: "registered",
  strict: true,
});
const registeredModuleOptimizer: Optimizer<"sgd"> = optim.sgd(registeredModule, { lr: 0.01 });
const registeredModuleProgram: Program<readonly [2], readonly [1]> = registeredModule.compile({ inputShape: [2] as const });
const registeredModuleSession: Session<readonly [2], readonly [1]> = registeredModuleProgram.bindModule(registeredModule);
const registeredModuleCompiledForward: Tensor<readonly [1]> = registeredModuleSession.stepTensor(tensor([1, 2], [2] as const));
registeredModuleOptimizer.zeroGrad();
class ContainerModule extends nn.Module<readonly [1], readonly [1]> {
  readonly layers: ModuleList<readonly [LinearModule<1, 1>]>;
  readonly namedLayers: ModuleDict<{ readonly head: LinearModule<1, 1> }>;
  readonly params: ParameterList<readonly [NnParameter<readonly [1]>, NnParameter<readonly [1]>]>;
  readonly namedParams: ParameterDict<{
    readonly scale: NnParameter<readonly [1]>;
    readonly shift: NnParameter<readonly [1]>;
  }>;

  constructor() {
    super({ kind: "container-head" });
    this.layers = nn.moduleList([nn.linear(1, 1)]);
    this.namedLayers = nn.moduleDict({ head: nn.linear(1, 1) });
    this.params = new nn.ParameterList([
      nn.Parameter("scale", tensor([1], [1] as const)),
      nn.Parameter("shift", tensor([0], [1] as const)),
    ]);
    this.namedParams = new nn.ParameterDict({
      scale: nn.Parameter("scale", tensor([1], [1] as const)),
      shift: nn.Parameter("shift", tensor([0], [1] as const)),
    });
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return this.namedLayers.get("head")!.forward(input).mul(this.namedParams.get("scale")!.tensor).add(this.namedParams.get("shift")!.tensor);
  }
}
const containerModule = new ContainerModule();
const standaloneBuffer: NnBuffer<readonly [1]> = nn.buffer("temperature", tensor([1], [1] as const));
const standaloneBufferAlias: NnBuffer<readonly [1]> = nn.Buffer("temperature", tensor([1], [1] as const));
const containerModuleList: ModuleList<readonly [LinearModule<1, 1>]> = new nn.ModuleList([nn.linear(1, 1)]);
const containerModuleListFromIterable: ModuleList = new nn.ModuleList(new nn.ModuleList([nn.relu()]));
const containerModuleListFactoryFromIterable: ModuleList = nn.module_list(new nn.ModuleList([nn.relu()]));
const containerModuleDict: ModuleDict<{ readonly head: LinearModule<1, 1> }> = new nn.ModuleDict({ head: nn.linear(1, 1) });
const containerParameterList: ParameterList<readonly [NnParameter<readonly [1]>]> = nn.parameter_list([
  nn.parameter("temperature", tensor([1], [1] as const)),
]);
const containerParameterListFromIterable: ParameterList = new nn.ParameterList(
  new nn.ParameterList([nn.parameter("iterableCtor", tensor([1], [1] as const))]),
);
const containerParameterListFactoryFromIterable: ParameterList = nn.parameter_list(
  new nn.ParameterList([nn.parameter("iterableFactory", tensor([1], [1] as const))]),
);
const containerParameterDict: ParameterDict<{ readonly temperature: NnParameter<readonly [1]> }> = nn.parameter_dict({
  temperature: nn.parameter("temperature", tensor([1], [1] as const)),
});
const containerModuleDictUpdateResult: ModuleDict<{ readonly head: LinearModule<1, 1> }> = containerModuleDict.update({
  tail: nn.relu(),
});
const containerModuleDictItems: readonly (readonly [string, NnModule])[] = containerModuleDict.items();
const containerModuleDictLen: number = containerModuleDict.__len__();
const containerModuleDictSize: number = containerModuleDict.size();
const containerModuleDictDunderItem: NnModule | undefined = containerModuleDict.__getitem__("head");
const containerModuleDictSetItemResult: ModuleDict<{ readonly head: LinearModule<1, 1> }> = containerModuleDict.__setitem__("head", nn.linear(1, 1));
const containerModuleDictContainsHead: boolean = containerModuleDict.__contains__("head");
const containerModuleDictDelItemResult: ModuleDict<{ readonly head: LinearModule<1, 1> }> = containerModuleDict
  .set("drop", nn.relu())
  .__delitem__("drop");
const containerModuleDictPoppedTail: NnModule | undefined = containerModuleDict.pop("tail");
const containerModuleDictClearResult: ModuleDict<{ readonly head: LinearModule<1, 1> }> = containerModuleDict.clear();
const containerParameterDictUpdateResult: ParameterDict<{ readonly temperature: NnParameter<readonly [1]> }> = containerParameterDict.update({
  bias: nn.parameter("bias", tensor([0], [1] as const)),
});
const containerParameterDictItems: readonly (readonly [string, NnParameter])[] = containerParameterDict.items();
const containerParameterDictLen: number = containerParameterDict.__len__();
const containerParameterDictSize: number = containerParameterDict.size();
const containerParameterDictDunderItem: NnParameter | undefined = containerParameterDict.__getitem__("temperature");
const containerParameterDictSetItemResult: ParameterDict<{ readonly temperature: NnParameter<readonly [1]> }> = containerParameterDict.__setitem__(
  "temperature",
  nn.parameter("temperature", tensor([2], [1] as const)),
);
const containerParameterDictContainsTemperature: boolean = containerParameterDict.__contains__("temperature");
const containerParameterDictDelItemResult: ParameterDict<{ readonly temperature: NnParameter<readonly [1]> }> = containerParameterDict
  .set("drop", nn.parameter("drop", tensor([0], [1] as const)))
  .__delitem__("drop");
const containerParameterDictPoppedBias: NnParameter | undefined = containerParameterDict.pop("bias");
const containerParameterDictClearResult: ParameterDict<{ readonly temperature: NnParameter<readonly [1]> }> = containerParameterDict.clear();
const containerModuleListAppendResult: ModuleList<readonly [LinearModule<1, 1>]> = containerModuleList.append(nn.relu());
const containerModuleListSetItemResult: ModuleList<readonly [LinearModule<1, 1>]> = containerModuleList.__setitem__(0, nn.linear(1, 1));
const containerModuleListInsertResult: ModuleList<readonly [LinearModule<1, 1>]> = containerModuleList.insert(1, nn.tanh());
const containerModuleListDelItemResult: ModuleList = new nn.ModuleList([nn.relu(), nn.tanh()]).__delitem__(-1);
const containerModuleListPoppedLast: NnModule | undefined = new nn.ModuleList([nn.relu(), nn.tanh()]).pop();
const containerModuleListPoppedFirst: NnModule | undefined = new nn.ModuleList([nn.relu(), nn.tanh()]).pop(0);
const containerModuleListClearResult: ModuleList = new nn.ModuleList([nn.relu(), nn.tanh()]).clear();
const containerModuleListExtendResult: ModuleList<readonly [LinearModule<1, 1>]> = containerModuleList.extend([
  nn.linear(1, 1),
]);
const containerModuleListExtendIterableResult: ModuleList<readonly [LinearModule<1, 1>]> = containerModuleList.extend(
  new nn.ModuleList([nn.relu()]),
);
const containerParameterListAppendResult: ParameterList<readonly [NnParameter<readonly [1]>]> = containerParameterList.append(
  nn.parameter("gain", tensor([1], [1] as const)),
);
const containerParameterListSetItemResult: ParameterList<readonly [NnParameter<readonly [1]>]> = containerParameterList.__setitem__(
  0,
  nn.parameter("temperature", tensor([2], [1] as const)),
);
const containerParameterListInsertResult: ParameterList<readonly [NnParameter<readonly [1]>]> = containerParameterList.insert(
  1,
  nn.parameter("inserted", tensor([2], [1] as const)),
);
const containerParameterListDelItemResult: ParameterList = new nn.ParameterList([
  nn.parameter("keep", tensor([1], [1] as const)),
  nn.parameter("drop", tensor([0], [1] as const)),
]).__delitem__(-1);
const containerParameterListPoppedLast: NnParameter | undefined = new nn.ParameterList([
  nn.parameter("keep", tensor([1], [1] as const)),
  nn.parameter("drop", tensor([0], [1] as const)),
]).pop();
const containerParameterListPoppedFirst: NnParameter | undefined = new nn.ParameterList([
  nn.parameter("keep", tensor([1], [1] as const)),
  nn.parameter("drop", tensor([0], [1] as const)),
]).pop(0);
const containerParameterListClearResult: ParameterList = new nn.ParameterList([
  nn.parameter("keep", tensor([1], [1] as const)),
  nn.parameter("drop", tensor([0], [1] as const)),
]).clear();
const containerParameterListExtendResult: ParameterList<readonly [NnParameter<readonly [1]>]> = containerParameterList.extend([
  nn.parameter("offset", tensor([0], [1] as const)),
]);
const containerParameterListExtendIterableResult: ParameterList<readonly [NnParameter<readonly [1]>]> = containerParameterList.extend(
  new nn.ParameterList([nn.parameter("iterable", tensor([3], [1] as const))]),
);
const containerModuleListLen: number = containerModuleList.__len__();
const containerModuleListSize: number = containerModuleList.size();
const containerModuleListGet: NnModule | undefined = containerModuleList.get(0);
const containerModuleListDunderItem: NnModule | undefined = containerModuleList.__getitem__(-1);
const containerParameterListLen: number = containerParameterList.__len__();
const containerParameterListSize: number = containerParameterList.size();
const containerParameterListGet: NnParameter | undefined = containerParameterList.get(0);
const containerParameterListDunderItem: NnParameter | undefined = containerParameterList.__getitem__(-1);
const containerChildren: readonly NnModule[] = containerModuleList.children();
const containerDictChildren: readonly NnModule[] = containerModuleDict.children();
const containerNamedChildren: readonly ModuleTraversalEntry[] = containerModule.layers.namedChildren("container.layers");
const containerDictNamedChildren: readonly ModuleTraversalEntry[] = containerModule.namedLayers.namedChildren("container.namedLayers");
const containerLayerLookup: NnModule | null = containerModule.getSubmodule("layers.0");
const containerDictLayerLookup: NnModule | null = containerModule.namedLayers.getSubmodule("head");
const containerLayerParameterLookup: NnParameter | null = containerModule.layers.getParameter("0.weight");
const containerDictParameterLookup: NnParameter | null = containerModule.namedLayers.getParameter("head.weight");
const containerListParameterLookup: NnParameter | null = containerModule.params.getParameter("0");
const containerListParameterLookupAlias: NnParameter | null = containerModule.params.get_parameter("1");
const containerDictNamedParameterLookup: NnParameter | null = containerModule.namedParams.getParameter("scale");
const containerDictNamedParameterLookupAlias: NnParameter | null = containerModule.namedParams.get_parameter("shift");
const containerListBufferLookup: NnBuffer | null = containerModule.params.getBuffer("0");
const containerListBufferLookupAlias: NnBuffer | null = containerModule.params.get_buffer("1");
const containerDictNamedBufferLookup: NnBuffer | null = containerModule.namedParams.getBuffer("scale");
const containerDictNamedBufferLookupAlias: NnBuffer | null = containerModule.namedParams.get_buffer("shift");
const containerApplyResult: ContainerModule = containerModule.apply((_module, entry) => {
  const traversalEntry: ModuleTraversalEntry = entry;
  const traversalModule: NnModule = traversalEntry.module;
});
const containerDictApplyResult: ModuleDict<{ readonly head: LinearModule<1, 1> }> = containerModuleDict.apply((module) => {
  const appliedModule: NnModule = module;
});
const containerParameterNames: readonly string[] = containerModule.parameterNames("container");
const containerStandaloneParameterNames: readonly string[] = containerParameterList.parameterNames("containerParams");
const containerStandaloneDictParameterNames: readonly string[] = containerParameterDict.parameterNames("containerParamDict");
const containerState: ModuleStateSnapshot = containerModule.stateDict("container");
const containerOptimizer: Optimizer<"sgd"> = optim.sgd(containerModule, { lr: 0.01 });
const containerProgram: Program<readonly [1], readonly [1]> = containerModule.compile({ inputShape: [1] as const });
const containerSession: Session<readonly [1], readonly [1]> = containerProgram.bindModule(containerModule);
const containerCompiledForward: Tensor<readonly [1]> = containerSession.stepTensor(tensor([2], [1] as const));
containerOptimizer.zeroGrad();
const customAutogradWeight: NnParameter<readonly [1]> = nn.parameter("weight", tensor([0.5], [1] as const));
const customAutogradBias: NnParameter<readonly [1]> = nn.parameter("bias", tensor([0.1], [1] as const));
const customAutogradModule: CustomModule<readonly [1], readonly [1]> = nn.module({
  kind: "custom-autograd-head",
  parameters: [customAutogradWeight, customAutogradBias],
  forward(input: Tensor<readonly [1]>) {
    return input.mul(customAutogradWeight.tensor).add(customAutogradBias.tensor);
  },
});
const customAutogradOutput: Tensor<readonly [1]> = customAutogradModule.__call__(tensor([2], [1] as const));
const customAutogradOptimizer: Optimizer<"sgd"> = optim.sgd(customAutogradModule, { lr: 0.01 });
customAutogradOptimizer.zero_grad();
const customAutogradState: ModuleStateSnapshot = customAutogradModule.stateDict("trained");
const customAutogradCloneWeight: NnParameter<readonly [1]> = nn.parameter("weight", tensor([0], [1] as const));
const customAutogradCloneBias: NnParameter<readonly [1]> = nn.parameter("bias", tensor([0], [1] as const));
const customAutogradClone: CustomModule<readonly [1], readonly [1]> = nn.module({
  kind: "custom-autograd-clone",
  parameters: [customAutogradCloneWeight, customAutogradCloneBias],
  forward(input: Tensor<readonly [1]>) {
    return input.mul(customAutogradCloneWeight.tensor).add(customAutogradCloneBias.tensor);
  },
});
const customAutogradValidatedClone: typeof customAutogradClone = customAutogradClone.loadStateDict(customAutogradState, {
  strict: true,
  prefix: "trained",
  validateOnly: true,
});
const customAutogradLoadedClone: typeof customAutogradClone = customAutogradClone.load_state_dict(customAutogradState, {
  strict: true,
  prefix: "trained",
});
const reshapeModule = nn.reshape([3, 2] as const);
const reshapeModuleShape: readonly [3, 2] = reshapeModule.shape;
const reshapeModuleForward: Tensor<readonly [3, 2]> = reshapeModule.forward(literalTensor);
const typedReshapeSupport: ModuleCompileSupport<readonly [2, 3], readonly [3, 2]> = reshapeModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const typedReshapeCompilePlan: ModuleCompileExplanation<readonly [2, 3], readonly [3, 2]> = reshapeModule.compilePlan({ backend: "cpu", inputShape: [2, 3] as const });
const inferredReshapeModule = nn.reshape([3, -1] as const);
const inferredReshapeModuleShape: readonly [3, -1] = inferredReshapeModule.shape;
const inferredReshapeModuleForward: Tensor<readonly [3, 2]> = inferredReshapeModule.forward(literalTensor);
const inferredReshapeModuleOutputShape: readonly [3, 2] | null = inferredReshapeModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
const inferredReshapeSupport: ModuleCompileSupport<readonly [2, 3], readonly [3, 2]> = inferredReshapeModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const inferredReshapeCompilePlan: ModuleCompileExplanation<readonly [2, 3], readonly [3, 2]> = inferredReshapeModule.compilePlan({ backend: "cpu", inputShape: [2, 3] as const });
type InferredReshapeModuleForwardShape = Expect<Equal<ShapeModuleForwardShape<"reshape", readonly [3, -1], readonly [2, 3]>, readonly [3, 2]>>;
const viewModuleForward: Tensor<readonly [1, 6]> = nn.view([1, 6] as const).forward(literalTensor);
const inferredViewModuleForward: Tensor<readonly [1, 6]> = nn.view([1, -1] as const).forward(literalTensor);
const broadcastModuleForward: Tensor<readonly [2, 3]> = nn.broadcastTo([2, 3] as const).forward(tensor([1, 2, 3], [1, 3] as const));
const expandModuleForward: Tensor<readonly [2, 3]> = nn.expand([2, 3] as const).forward(tensor([1, 2, 3], [1, 3] as const));
const repeatModule = nn.repeat([2, 3] as const);
const repeatModuleForward: Tensor<readonly [4, 9]> = repeatModule.forward(literalTensor);
const repeatModuleOutputShape: readonly [4, 9] | null = repeatModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
const repeatModuleTrace: ModuleProgramTrace = nn.trace(repeatModule, { inputShape: [2, 3] as const });
const repeatModuleIr: ModuleTensorProgramIr | null = repeatModule.tensorProgramIr({ backend: "cpu", inputShape: [2, 3] as const });
const repeatModuleSupport: ModuleCompileSupport<readonly [2, 3], readonly [4, 9]> = repeatModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const tileModule = new nn.Tile([2, 1, 3] as const);
const tileModuleForward: Tensor<readonly [2, 2, 9]> = tileModule.forward(literalTensor);
const tileModuleOutputShape: readonly [2, 2, 9] | null = tileModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
type RepeatShapeModuleForwardShape = Expect<Equal<ShapeModuleForwardShape<"repeat", readonly [2, 3], readonly [2, 3]>, readonly [4, 9]>>;
type TileShapeModuleForwardShape = Expect<Equal<ShapeModuleForwardShape<"tile", readonly [2, 1, 3], readonly [2, 3]>, readonly [2, 2, 9]>>;
const diagonalModule = nn.diagonal();
const diagonalModuleForward: Tensor<readonly [2]> = diagonalModule.forward(literalTensor);
const diagonalModuleOutputShape: readonly [2] | null = diagonalModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
const diagonalModuleTrace: ModuleProgramTrace = nn.trace(diagonalModule, { inputShape: [2, 3] as const });
const diagonalModuleIr: ModuleTensorProgramIr | null = diagonalModule.tensorProgramIr({ backend: "cpu", inputShape: [2, 3] as const });
const diagonalModuleSupport: ModuleCompileSupport<readonly [2, 3], readonly [2]> = diagonalModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
type DiagonalShapeModuleForwardShape = Expect<Equal<ShapeModuleForwardShape<"diagonal", readonly [], readonly [2, 3]>, readonly [2]>>;
const permuteModule: NnModule = nn.permute([1, 0]);
const permuteTrace = nn.sequential(nn.permute([1, 0]), nn.flatten()).trace({ inputShape: [2, 3] as const });
const permuteTraceOutputShape: readonly number[] | null | undefined = permuteTrace.outputShape;
const linearSupport: ModuleCompileSupport = linear.compileSupport({ backend: "cpu" });
const linearSupportAlias: ModuleCompileSupport = linear.compile_support({ backend: "cpu" });
const linearRequiredSupport: ModuleCompileSupport = linear.requireCompileSupport({ backend: "cpu" });
const linearRequiredSupportAlias: ModuleCompileSupport = linear.require_compile_support({ backend: "cpu" });
const linearSupportInputShape: readonly number[] | undefined = linearSupport.inputShape;
const linearSupportOutputShape: readonly number[] | undefined = linearSupport.outputShape;
const linearTrace: ModuleProgramTrace = linear.trace({ inputShape: [2] as const });
const linearModuleCompilerSignatures: ModuleCompilerSignatures | null = linear.compilerSignatures({ backend: "cpu" });
const linearModuleCompilerSignaturesAlias: ModuleCompilerSignatures | null = linear.compiler_signatures({ backend: "cpu" });
const linearModuleIr: ModuleTensorProgramIr | null = linear.tensorProgramIr({ backend: "cpu" });
const linearModuleIrAlias: ModuleTensorProgramIr | null = linear.tensor_program_ir({ backend: "cpu" });
const linearModuleKernelPlan: ModuleKernelPlan | null = linear.kernelPlan({ backend: "cpu" });
const linearModuleKernelPlanAlias: ModuleKernelPlan | null = linear.kernel_plan({ backend: "cpu" });
const linearModuleBufferLayout: ModuleKernelBufferLayout | null = linear.bufferLayout({ backend: "cpu" });
const linearModuleBufferLayoutAlias: ModuleKernelBufferLayout | null = linear.buffer_layout({ backend: "cpu" });
const linearModuleMemoryLayout: ModuleKernelMemoryLayout | null = linear.memoryLayout({ backend: "cpu" });
const linearModuleMemoryLayoutAlias: ModuleKernelMemoryLayout | null = linear.memory_layout({ backend: "cpu" });
const linearModuleInputShape: readonly number[] | null = linear.inputShape({ backend: "cpu" });
const linearModuleInputShapeAlias: readonly number[] | null = linear.input_shape({ backend: "cpu" });
const linearModuleOutputShape: readonly number[] | null = linear.outputShape({ backend: "cpu" });
const linearModuleOutputShapeAlias: readonly number[] | null = linear.output_shape({ backend: "cpu" });
const linearTypedCompileOptions: CompileOptionsWithInputShape<readonly [2]> = { backend: "cpu", inputShape: [2] as const };
const linearBatchedTypedCompileOptions: CompileOptionsWithInputShape<readonly [4, 2]> = { backend: "cpu", inputShape: [4, 2] as const };
const typedLinearSupport: ModuleCompileSupport<readonly [2], readonly [3]> = linear.compileSupport(linearTypedCompileOptions);
const typedBatchedLinearSupport: ModuleCompileSupport<readonly [4, 2], readonly [4, 3]> = linear.compileSupport(linearBatchedTypedCompileOptions);
const typedLinearRequiredSupport: ModuleCompileSupport<readonly [2], readonly [3]> = linear.requireCompileSupport(linearTypedCompileOptions);
const typedLinearSupportInputShape: readonly [2] | undefined = typedLinearSupport.inputShape;
const typedLinearSupportOutputShape: readonly [3] | undefined = typedLinearSupport.outputShape;
const typedLinearCompilePlan: ModuleCompileExplanation<readonly [2], readonly [3]> = linear.compilePlan(linearTypedCompileOptions);
const typedLinearCompilePlanInputShape: readonly [2] | null = typedLinearCompilePlan.inputShape;
const typedLinearCompilePlanOutputShape: readonly [3] | null = typedLinearCompilePlan.outputShape;
const typedLinearNamespaceSupport: ModuleCompileSupport<readonly [2], readonly [3]> = nn.requireCompileSupport(linear, linearTypedCompileOptions);
const typedLinearNamespaceCompilePlan: ModuleCompileExplanation<readonly [2], readonly [3]> = nn.requireCompilePlan(linear, linearTypedCompileOptions);
const linearTypedMethodOutputShape: readonly [3] | null = linear.outputShape(linearTypedCompileOptions);
const linearTypedMethodOutputShapeAlias: readonly [3] | null = linear.output_shape(linearTypedCompileOptions);
const nnNamespaceTypedOutputShape: readonly [3] | null = nn.outputShape(linear, linearTypedCompileOptions);
const nnNamespaceTypedOutputShapeAlias: readonly [3] | null = nn.output_shape(linear, linearTypedCompileOptions);
const linearModuleShapeConstraints: ModuleKernelShapeConstraints | null = linear.shapeConstraints({ backend: "cpu" });
const linearModuleShapeConstraintsAlias: ModuleKernelShapeConstraints | null = linear.shape_constraints({ backend: "cpu" });
const linearModuleParameterLayout: ModuleKernelParameterLayout | null = linear.parameterLayout({ backend: "cpu" });
const linearModuleParameterLayoutAlias: ModuleKernelParameterLayout | null = linear.parameter_layout({ backend: "cpu" });
const linearModuleCompileExplanation: ModuleCompileExplanation = linear.compileExplanation({ backend: "cpu" });
const linearModuleCompileExplanationAlias: ModuleCompileExplanation = linear.compile_explanation({ backend: "cpu" });
const linearModuleCompilePlan: ModuleCompileExplanation = linear.compilePlan({ backend: "cpu" });
const linearModuleCompilePlanAlias: ModuleCompileExplanation = linear.compile_plan({ backend: "cpu" });
const linearModuleRequiredCompilePlan: ModuleCompileExplanation = linear.requireCompilePlan({ backend: "cpu" });
const linearModuleRequiredCompilePlanAlias: ModuleCompileExplanation = linear.require_compile_plan({ backend: "cpu" });
const linearModuleAssertedCompilePlan: ModuleCompileExplanation = linear.assertCompilePlan({ backend: "cpu" });
const linearModuleAssertedCompilePlanAlias: ModuleCompileExplanation = linear.assert_compile_plan({ backend: "cpu" });
const linearModulePreflight: ModuleCompileExplanation = linear.preflight({ backend: "cpu" });
const linearCanCompile: boolean = linear.canCompile({ backend: "cpu" });
const linearCanCompileAlias: boolean = linear.can_compile({ backend: "cpu" });
const compileNamespace: CompileNamespace = compile;
const compileNamespaceTrace: ModuleProgramTrace = compile.trace(linear, { inputShape: [2] as const });
const compileNamespaceAnalysis: CompileAnalysis | ModuleCompileExplanation = compile.analyze(linear, { inputShape: [2] as const });
const compileNamespaceAnalysisPredicate: boolean = compile.isCompileAnalysis(compileNamespaceAnalysis);
const compileNamespaceAnalysisRequired: CompileAnalysis = compile.requireCompileAnalysis(compileNamespaceAnalysis);
const compileNamespaceAnalysisAsserted: CompileAnalysis = compile.assert_compile_analysis(compileNamespaceAnalysis);
const compileNamespaceAnalysisSignature: string = compileNamespaceAnalysisRequired.signature;
const compileNamespaceAnalysisSignatureFromNamespace: string = compile.compileAnalysisSignature(compileNamespaceAnalysisRequired);
const compileNamespaceAnalysisSignatureMatch: boolean = compile.matchesCompileAnalysisSignature(
  compileNamespaceAnalysisRequired,
  compileNamespaceAnalysisRequired.signature,
);
const compileNamespaceSupport: ModuleCompileSupport = compile.compileSupport(linear, { backend: "cpu" });
const compileNamespaceSupportAlias: ModuleCompileSupport = compile.compile_support(linear, { backend: "cpu" });
const compileNamespaceRequiredSupport: ModuleCompileSupport = compile.requireCompileSupport(linear, { backend: "cpu" });
const compileNamespaceRequiredSupportAlias: ModuleCompileSupport = compile.require_compile_support(linear, { backend: "cpu" });
const nnNamespaceRequiredSupport: ModuleCompileSupport = nn.requireCompileSupport(linear, { backend: "cpu" });
const nnNamespaceRequiredSupportAlias: ModuleCompileSupport = nn.require_compile_support(linear, { backend: "cpu" });
const nnNamespaceRequiredCompilePlan: ModuleCompileExplanation = nn.requireCompilePlan(linear, { backend: "cpu" });
const nnNamespaceRequiredCompilePlanAlias: ModuleCompileExplanation = nn.require_compile_plan(linear, { backend: "cpu" });
const nnNamespaceAssertedCompilePlan: ModuleCompileExplanation = nn.assertCompilePlan(linear, { backend: "cpu" });
const nnNamespaceAssertedCompilePlanAlias: ModuleCompileExplanation = nn.assert_compile_plan(linear, { backend: "cpu" });
const nnNamespaceAcceptedCompilePlan: boolean = nn.acceptsModuleCompilePlan(linearModuleCompilePlan);
const nnNamespaceRequiredEvidenceCompilePlan: ModuleCompileExplanation = nn.requireModuleCompilePlan(linearModuleCompilePlan);
const nnNamespaceAssertedEvidenceCompilePlan: ModuleCompileExplanation = nn.assertModuleCompilePlan(linearModuleCompilePlan);
const nnNamespaceAssertedEvidenceCompilePlanAlias: ModuleCompileExplanation = nn.assert_module_compile_plan(linearModuleCompilePlan);
const nnNamespaceMatchedEvidenceCompilePlan: boolean = nn.matchesModuleCompilePlanSignature(linearModuleCompilePlan, linearModuleCompilePlan.signature);
const compileNamespaceAssertedSupport: ModuleCompileSupport = compile.assertCompileSupport(linear, { backend: "cpu" });
const compileNamespaceAssertedSupportAlias: ModuleCompileSupport = compile.assert_compile_support(linear, { backend: "cpu" });
const compileNamespaceCanCompile: boolean = compile.canCompile(linear, { backend: "cpu" });
const compileNamespaceCanCompileAlias: boolean = compile.can_compile(linear, { backend: "cpu" });
const compileNamespaceExplanation: ModuleCompileExplanation | ModuleCompileSupport = compile.explain(linear, { backend: "cpu" });
const compileNamespacePreflight: ModuleCompileExplanation | ModuleCompileSupport = compile.preflight(linear, { backend: "cpu" });
const compileNamespaceCompileExplanation: ModuleCompileExplanation | ModuleCompileSupport = compile.compileExplanation(linear, { backend: "cpu" });
const compileNamespaceCompileExplanationAlias: ModuleCompileExplanation | ModuleCompileSupport = compile.compile_explanation(linear, { backend: "cpu" });
const compileNamespaceCompilePlan: ModuleCompileExplanation | ModuleCompileSupport = compile.compilePlan(linear, { backend: "cpu" });
const compileNamespaceCompilePlanAlias: ModuleCompileExplanation | ModuleCompileSupport = compile.compile_plan(linear, { backend: "cpu" });
const compileNamespaceRequiredCompilePlan: ModuleCompileExplanation | ModuleCompileSupport = compile.requireCompilePlan(linear, { backend: "cpu" });
const compileNamespaceRequiredCompilePlanAlias: ModuleCompileExplanation | ModuleCompileSupport = compile.require_compile_plan(linear, { backend: "cpu" });
const compileNamespaceAssertedCompilePlan: ModuleCompileExplanation | ModuleCompileSupport = compile.assertCompilePlan(linear, { backend: "cpu" });
const compileNamespaceAssertedCompilePlanAlias: ModuleCompileExplanation | ModuleCompileSupport = compile.assert_compile_plan(linear, { backend: "cpu" });
const compileNamespaceAcceptedEvidenceCompilePlan: boolean = compile.acceptsModuleCompilePlan(linearModuleCompilePlan);
const compileNamespaceRequiredEvidenceCompilePlan: ModuleCompileExplanation = compile.requireModuleCompilePlan(linearModuleCompilePlan);
const compileNamespaceAssertedEvidenceCompilePlan: ModuleCompileExplanation = compile.assertModuleCompilePlan(linearModuleCompilePlan);
const compileNamespaceAssertedEvidenceCompilePlanAlias: ModuleCompileExplanation = compile.assert_module_compile_plan(linearModuleCompilePlan);
const compileNamespaceMatchedEvidenceCompilePlan: boolean = compile.matchesModuleCompilePlanSignature(linearModuleCompilePlan, linearModuleCompilePlan.signature);
const compileNamespaceCompilerSignatures: ModuleCompilerSignatures | null = compile.compilerSignatures(linear, { backend: "cpu" });
const compileNamespaceCompilerSignaturesAlias: ModuleCompilerSignatures | null = compile.compiler_signatures(linear, { backend: "cpu" });
const compileNamespaceIr: ModuleTensorProgramIr | null = compile.tensorProgramIr(linear, { backend: "cpu" });
const compileNamespaceIrAlias: ModuleTensorProgramIr | null = compile.tensor_program_ir(linear, { backend: "cpu" });
const compileNamespaceKernelPlan: ModuleKernelPlan | null = compile.kernelPlan(linear, { backend: "cpu" });
const compileNamespaceKernelPlanAlias: ModuleKernelPlan | null = compile.kernel_plan(linear, { backend: "cpu" });
const compileNamespaceBufferLayout: ModuleKernelBufferLayout | null = compile.bufferLayout(linear, { backend: "cpu" });
const compileNamespaceBufferLayoutAlias: ModuleKernelBufferLayout | null = compile.buffer_layout(linear, { backend: "cpu" });
const compileNamespaceMemoryLayout: ModuleKernelMemoryLayout | null = compile.memoryLayout(linear, { backend: "cpu" });
const compileNamespaceMemoryLayoutAlias: ModuleKernelMemoryLayout | null = compile.memory_layout(linear, { backend: "cpu" });
const compileNamespaceInputShape: readonly number[] | null = compile.inputShape(linear, { backend: "cpu" });
const compileNamespaceInputShapeAlias: readonly number[] | null = compile.input_shape(linear, { backend: "cpu" });
const compileNamespaceOutputShape: readonly number[] | null = compile.outputShape(linear, { backend: "cpu" });
const compileNamespaceOutputShapeAlias: readonly number[] | null = compile.output_shape(linear, { backend: "cpu" });
const compileNamespaceTypedOutputShape: readonly [3] | null = compile.outputShape(linear, linearTypedCompileOptions);
const compileNamespaceTypedOutputShapeAlias: readonly [3] | null = compile.output_shape(linear, linearTypedCompileOptions);
const compileNamespaceShapeConstraints: ModuleKernelShapeConstraints | null = compile.shapeConstraints(linear, { backend: "cpu" });
const compileNamespaceShapeConstraintsAlias: ModuleKernelShapeConstraints | null = compile.shape_constraints(linear, { backend: "cpu" });
const compileNamespaceParameterLayout: ModuleKernelParameterLayout | null = compile.parameterLayout(linear, { backend: "cpu" });
const compileNamespaceParameterLayoutAlias: ModuleKernelParameterLayout | null = compile.parameter_layout(linear, { backend: "cpu" });
const linearModuleParameterNames: readonly string[] = linear.parameterNames();
const linearModuleParameterNamesWithPrefix: readonly string[] = nn.parameterNames(linear, "model");
const linearNamespaceNamedParametersWithPrefix: NnParameter[] = nn.namedParameters(linear, "model");
const linearModuleNamedParametersAlias: NnParameter[] = linear.named_parameters("model");
const linearNamespaceNamedParametersAlias: NnParameter[] = nn.named_parameters(linear, "model");
const linearModuleGetParameter: NnParameter | null = linear.getParameter("weight");
const linearModuleGetParameterAlias: NnParameter | null = linear.get_parameter("bias");
const linearNamespaceGetParameter: NnParameter | null = nn.getParameter(linear, "weight");
const linearNamespaceGetParameterAlias: NnParameter | null = nn.get_parameter(linear, "bias");
const linearModuleParameterInfos: readonly ModuleParameterInfo[] = linear.parameterInfos();
const linearModuleParameterInfoByName: ModuleParameterInfo | null = linear.parameterInfo("weight");
const linearModuleParameterInfoByIndex: ModuleParameterInfo | null = nn.parameterInfo(linear, 0);
type LinearNamespaceForwardShape = Expect<Equal<ModuleForwardShape<typeof linear, readonly [2]>, readonly [3]>>;
const linearNamespaceForward: Tensor<readonly [3]> = nn.forward(linear, tensor([1, 2], [2] as const));
const reluNamespaceForward: Tensor<readonly [2, 3]> = nn.forward(nn.relu(), literalTensor);
const sumNamespaceForward: Tensor<readonly [2, 1]> = nn.forward(nn.sum(1), literalTensor);
const reshapeNamespaceForward: Tensor<readonly [3, 2]> = nn.forward(nn.reshape([3, 2] as const), literalTensor);
const inferredReshapeNamespaceForward: Tensor<readonly [3, 2]> = nn.forward(nn.reshape([3, -1] as const), literalTensor);
const linearParameterRequiresGrad: boolean = linearNamespaceNamedParametersAlias[0].requiresGrad;
const linearParameterRequiresGradAlias: boolean = linearNamespaceNamedParametersAlias[0].requires_grad;
const linearParameterInfoRequiresGrad: boolean | undefined = linearModuleParameterInfoByName?.requiresGrad;
const linearParameterInfoRequiresGradAlias: boolean | undefined = linearModuleParameterInfoByName?.requires_grad;
const linearModuleChildren: readonly NnModule[] = linear.children();
const linearModuleModules: readonly NnModule[] = linear.modules();
const linearModuleNamedChildren: readonly ModuleTraversalEntry[] = linear.namedChildren("model");
const linearModuleNamedChildrenAlias: readonly ModuleTraversalEntry[] = linear.named_children("model");
const linearModuleNamedModules: readonly ModuleTraversalEntry[] = linear.namedModules("model");
const linearModuleNamedModulesAlias: readonly ModuleTraversalEntry[] = linear.named_modules("model");
const linearModuleGetSubmodule: NnModule | null = linear.getSubmodule("");
const linearModuleGetSubmoduleAlias: NnModule | null = linear.get_submodule("");
const linearNamespaceGetSubmodule: NnModule | null = nn.getSubmodule(linear, "");
const linearNamespaceGetSubmoduleAlias: NnModule | null = nn.get_submodule(linear, "");
const linearModuleGetBuffer: NnBuffer | null = linear.getBuffer("missing");
const linearModuleGetBufferAlias: NnBuffer | null = linear.get_buffer("missing");
const linearNamespaceGetBuffer: NnBuffer | null = nn.getBuffer(linear, "missing");
const linearNamespaceGetBufferAlias: NnBuffer | null = nn.get_buffer(linear, "missing");
const linearModuleApply: typeof linear = linear.apply((module, entry) => {
  const appliedModule: NnModule = module;
  const appliedEntry: ModuleTraversalEntry = entry;
});
const linearNamespaceChildren: readonly NnModule[] = nn.children(linear);
const linearNamespaceModules: readonly NnModule[] = nn.modules(linear);
const linearNamespaceNamedChildren: readonly ModuleTraversalEntry[] = nn.namedChildren(linear, "model");
const linearNamespaceNamedChildrenAlias: readonly ModuleTraversalEntry[] = nn.named_children(linear, "model");
const linearNamespaceNamedModules: readonly ModuleTraversalEntry[] = nn.namedModules(linear, "model");
const linearNamespaceNamedModulesAlias: readonly ModuleTraversalEntry[] = nn.named_modules(linear, "model");
const linearNamespaceApply: typeof linear = nn.apply(linear, (module) => {
  const appliedModule: NnModule = module;
});
const linearModuleRequiresGrad: typeof linear = linear.requiresGrad_(false);
const linearModuleRequiresGradAlias: typeof linear = linear.requires_grad_(true);
const linearModuleTrain: typeof linear = linear.train();
const linearModuleEval: typeof linear = linear.eval();
const linearNamespaceTrain: typeof linear = nn.train(linear, true);
const linearNamespaceEval: typeof linear = nn.eval(linear);
const linearModuleCpu: typeof linear = linear.cpu();
const linearModuleToCpu: typeof linear = linear.to("cpu");
const linearModuleFloat: typeof linear = linear.float();
const linearModuleFloat32: typeof linear = linear.float32();
const linearNamespaceCpu: typeof linear = nn.cpu(linear);
const linearNamespaceToCpu: typeof linear = nn.to(linear, { device: "cpu", copy: false });
const linearNamespaceFloat: typeof linear = nn.float(linear);
const linearNamespaceFloat32: typeof linear = nn.float32(linear, { dtype: "float32", copy: false });
const linearNamespaceFreeze: typeof linear = nn.freeze(linear);
const linearNamespaceUnfreeze: typeof linear = nn.unfreeze(linear);
const zeroGradOptions: ZeroGradOptions = { setToNone: true, set_to_none: true };
linear.zeroGrad();
linear.zero_grad();
linear.zero_grad(zeroGradOptions);
nn.zeroGrad(linear, { setToNone: true });
nn.zero_grad(linear);
nn.zero_grad(linear, { set_to_none: true });
const nnRequiresGradTarget: typeof linear = nn.requiresGrad_(linear, true);
const nnRequiresGradAliasTarget: typeof linear = nn.requires_grad_(linear, false);
const nnStateSnapshot: ModuleStateSnapshot = nn.stateDict(linear, "head");
const nnStateSnapshotOptions: ModuleStateSnapshot = nn.stateDict(linear, { prefix: "head" });
const nnStateSnapshotAlias: ModuleStateSnapshot = nn.state_dict(linear, "head");
const nnStateSnapshotAliasOptions: ModuleStateSnapshot = nn.state_dict(linear, { prefix: "head" });
const nnLoadedStateTarget: typeof linear = nn.loadStateDict(linear, nnStateSnapshot, { strict: true, prefix: "head" });
const nnLoadedStateTargetAlias: typeof linear = nn.load_state_dict(linear, nnStateSnapshotAlias, { strict: true, prefix: "head" });
const linearStateSnapshotAlias: ModuleStateSnapshot = linear.state_dict();
const linearPrefixedStateSnapshotOptions: ModuleStateSnapshot = linear.stateDict({ prefix: "head" });
const linearPrefixedStateSnapshotAliasOptions: ModuleStateSnapshot = linear.state_dict({ prefix: "head" });
const linearLoadedStateTargetAlias: typeof linear = linear.load_state_dict(linearStateSnapshotAlias, { strict: true });
const linearProgram: Program = nn.compile(linear, { backend: "cpu" });
const compileNamespaceProgram: Program = compile.compile(linear, { backend: "cpu" });
const typedLinearProgram: Program<readonly [2], readonly [3]> = nn.compile(linear, linearTypedCompileOptions);
const typedBatchedLinearProgram: Program<readonly [4, 2], readonly [4, 3]> = linear.compile(linearBatchedTypedCompileOptions);
const typedCompileNamespaceProgram: Program<readonly [2], readonly [3]> = compile.compile(linear, linearTypedCompileOptions);
const typedLinearInstanceProgram: Program<readonly [2], readonly [3]> = linear.compile(linearTypedCompileOptions);
const typedLinearProgramInputShape: readonly [2] = typedLinearProgram.inputShape();
const typedLinearProgramOutputShape: readonly [3] = typedLinearProgram.outputShape();
const typedLinearProgramExecutionPlan: ProgramExecutionPlan<readonly [2], readonly [3]> = typedLinearProgram.executionPlan();
const typedLinearProgramExecutionPlanAlias: ProgramExecutionPlan<readonly [2], readonly [3]> = typedLinearProgram.execution_plan();
const typedLinearProgramRequiredExecutionPlan: ProgramExecutionPlan<readonly [2], readonly [3]> = typedLinearProgram.requireExecutionPlan();
const typedLinearProgramExecutionPlanInputShape: readonly [2] | null = typedLinearProgramExecutionPlan.inputShape;
const typedLinearProgramExecutionPlanOutputShape: readonly [3] = typedLinearProgramExecutionPlan.outputShape;
const typedLinearModuleBindings: ModuleBindings<readonly [2], readonly [3]> = linear.bindParameters(linearTypedCompileOptions);
const typedBatchedLinearModuleBindings: ModuleBindings<readonly [4, 2], readonly [4, 3]> = linear.bindParameters(linearBatchedTypedCompileOptions);
const typedLinearModuleBindingsAlias: ModuleBindings<readonly [2], readonly [3]> = linear.bind_parameters(linearTypedCompileOptions);
const typedLinearNamespaceBindings: ModuleBindings<readonly [2], readonly [3]> = nn.bindParameters(linear, linearTypedCompileOptions);
const typedLinearPlacedBindings: ModuleBindings<readonly [2], readonly [3]> = linear.placeParameters(typedLinearProgram);
const typedLinearNamespacePlacedBindings: ModuleBindings<readonly [2], readonly [3]> = nn.placeParameters(linear, typedLinearProgram);
const typedLinearModuleBindingPlan: ModuleBindingPlan<readonly [2], readonly [3]> = nn.bindingPlan(typedLinearModuleBindings);
const typedLinearModuleBindingPlanAlias: ModuleBindingPlan<readonly [2], readonly [3]> = nn.binding_plan(typedLinearModuleBindingsAlias);
const typedLinearModuleRequiredBindingPlan: ModuleBindingPlan<readonly [2], readonly [3]> = nn.requireBindingPlan(typedLinearNamespaceBindings);
const typedLinearModuleBindingPlanInputShape: readonly [2] | null = typedLinearModuleBindingPlan.inputShape;
const typedLinearModuleBindingPlanOutputShape: readonly [3] | null = typedLinearModuleBindingPlan.outputShape;
const typedLinearModuleBindingPlanPlacementMode: ModuleBindingPlacementMode = typedLinearModuleBindingPlan.placementMode;
const typedLinearModuleBindingPlanNativeSlots: readonly string[] = typedLinearModuleBindingPlan.nativeSlots;
const typedLinearProgramBindingPlan: ProgramBindingPlan<readonly [2], readonly [3]> =
  typedLinearProgram.bindingPlan(typedLinearModuleBindings);
const typedLinearProgramBindingPlanAlias: ProgramBindingPlan<readonly [2], readonly [3]> =
  typedLinearProgram.binding_plan(typedLinearModuleBindingsAlias);
const typedLinearProgramRequiredBindingPlan: ProgramBindingPlan<readonly [2], readonly [3]> =
  typedLinearProgram.requireBindingPlan(typedLinearNamespaceBindings);
const typedLinearProgramBindingPlanInputShape: readonly [2] | null = typedLinearProgramBindingPlan.inputShape;
const typedLinearProgramBindingPlanOutputShape: readonly [3] | null = typedLinearProgramBindingPlan.outputShape;
const typedLinearInstanceProgramOutputShape: readonly [3] = typedLinearInstanceProgram.outputShape();
const typedRawLinearProgramBindings: ProgramBindings<readonly [2], readonly [3]> = {
  weights: new Float32Array(6),
  bias: new Float32Array(3),
  inputShape: [2] as const,
  outputShape: [3] as const,
};
type TypedRawLinearProgramBindingsInput = Expect<Equal<NonNullable<ProgramBindings<readonly [2], readonly [3]>["input"]>, ProgramInputBinding<readonly [2]> | NativeBuffer>>;
type TypedRawLinearProgramBindingsOutput = Expect<Equal<NonNullable<ProgramBindings<readonly [2], readonly [3]>["output"]>, ProgramOutputBinding<readonly [3]>>>;
const typedRawLinearProgramBindingsInputShape: readonly [2] | undefined = typedRawLinearProgramBindings.inputShape;
const typedRawLinearProgramBindingsOutputShape: readonly [3] | undefined = typedRawLinearProgramBindings.outputShape;
const typedProgramInputTensor: Tensor<readonly [2]> = tensor([1, 2], [2] as const);
const typedProgramOutputTensor: Tensor<readonly [3]> = tensor([0, 0, 0], [3] as const);
const typedProgramInputBinding: ProgramInputBinding<readonly [2]> = typedProgramInputTensor;
const typedProgramOutputBinding: ProgramOutputBinding<readonly [3]> = typedProgramOutputTensor;
// @ts-expect-error shaped Tensor input bindings must preserve literal input shape evidence.
const typedProgramInputBindingMismatch: ProgramInputBinding<readonly [3]> = typedProgramInputTensor;
// @ts-expect-error shaped Tensor output bindings must preserve literal output shape evidence.
const typedProgramOutputBindingMismatch: ProgramOutputBinding<readonly [2]> = typedProgramOutputTensor;
const typedRawLinearBindingPlan: ModuleBindingPlan<readonly [2], readonly [3]> = nn.bindingPlan(typedRawLinearProgramBindings);
const typedRawLinearProgramBindingPlan: ProgramBindingPlan<readonly [2], readonly [3]> =
  typedLinearProgram.bindingPlan(typedRawLinearProgramBindings);
const typedRawLinearProgramSession: Session<readonly [2], readonly [3]> =
  typedLinearProgram.bind(typedRawLinearProgramBindings);
const programNamespace: ProgramNamespace = program;
const programNamespacePath: "Program -> Session -> StepParams" = program.programManifest.runtimePath;
const sessionNamespace: SessionNamespace = session;
const sessionNamespacePath: "Program -> Session -> StepParams" = session.sessionManifest.runtimePath;
const stepParamsNamespace: StepParamsNamespace = stepParams;
const stepParamsNamespacePath: "Program -> Session -> StepParams" = stepParams.stepParamsManifest.runtimePath;
const linearEvidence: ProgramCompileEvidence | null = linearProgram.compileEvidence();
const linearEvidencePredicate: boolean = linearEvidence !== null && program.isProgramCompileEvidence(linearEvidence);
const linearEvidenceRequired: ProgramCompileEvidence = program.requireProgramCompileEvidence(linearEvidence);
const linearEvidenceAsserted: ProgramCompileEvidence = program.assert_program_compile_evidence(linearEvidence);
const linearEvidenceSignature: string = program.programCompileEvidenceSignature(linearEvidenceRequired);
const linearEvidenceSignatureMatch: boolean = program.matchesProgramCompileEvidenceSignature(linearEvidenceRequired, linearEvidenceSignature);
const linearEvidenceCompileNamespacePredicate: boolean = compile.isProgramCompileEvidence(linearEvidenceRequired);
const linearEvidenceCompileNamespaceRequired: ProgramCompileEvidence = compile.requireProgramCompileEvidence(linearEvidenceRequired);
const linearEvidenceCompileNamespaceSignatureMatch: boolean =
  compile.matchesProgramCompileEvidenceSignature(linearEvidenceCompileNamespaceRequired, linearEvidenceCompileNamespaceRequired.signature);
const linearSignatures: ModuleCompleteCompilerSignatures | null = linearProgram.compilerSignatures();
const linearIr: ModuleTensorProgramIr | null = linearProgram.tensorProgramIr();
const linearIrSignature: string | undefined = linearIr?.signature;
const linearKernelPlan: ModuleKernelPlan | null = linearProgram.kernelPlan();
const linearKernelPlanSignature: string | undefined = linearKernelPlan?.signature;
const linearKernelPlanAccepted: boolean = acceptsRuntimeKernelPlan(linearKernelPlan);
const linearKernelPlanRequired: ModuleKernelPlan = requireRuntimeKernelPlan(linearKernelPlan);
const linearKernelPlanAsserted: ModuleKernelPlan = assertRuntimeKernelPlan(linearKernelPlanRequired);
const linearKernelPlanSignatureMatch: boolean = matchesRuntimeKernelPlanSignature(linearKernelPlanRequired, linearKernelPlanRequired.signature);
const linearKernelPlanMemoryLayoutSignature: string | undefined = linearKernelPlan?.memoryLayout.signature;
const linearKernelPlanBufferLayoutSignature: string | undefined = linearKernelPlan?.bufferLayout.signature;
const linearKernelPlanParameterLayoutSignature: string | undefined = linearKernelPlan?.parameterLayout.signature;
const linearShapeConstraints: ModuleKernelShapeConstraints | null = linearProgram.shapeConstraints();
const linearParameterLayout: ModuleKernelParameterLayout | null = linearProgram.parameterLayout();
const linearParameterLayoutSignature: string | undefined = linearParameterLayout?.signature;
const linearProgramParameterNames: readonly string[] = linearProgram.parameterNames();
const linearProgramParameterInfos: readonly ModuleKernelParameterLayoutEntry[] = linearProgram.parameterInfos();
const linearProgramParameterInfoByName: ModuleKernelParameterLayoutEntry | null = linearProgram.parameterInfo("0.weight");
const linearProgramParameterInfoByIndex: ModuleKernelParameterLayoutEntry | null = linearProgram.parameterInfo(0);
const linearCompatibility: ProgramModuleCompatibility = linearProgram.moduleCompatibility(linear);
const linearProgramModuleCompatibilityKind: "zgml.program.module-compatibility" = linearCompatibility.kind;
const linearProgramModuleCompatibilitySignature: string = linearCompatibility.signature;
const linearCompatibilityDiagnostic = linearCompatibility.diagnostics[0];
const linearCompatibilityShapeDiagnosticProgramShape: readonly number[] | undefined =
  linearCompatibilityDiagnostic?.code === "input-shape-mismatch" || linearCompatibilityDiagnostic?.code === "output-shape-mismatch"
    ? linearCompatibilityDiagnostic.programShape
    : undefined;
const linearInspection = linearProgram.inspect();
const linearProgramInspectionKind: "zgml.program.inspection" = linearInspection.kind;
const linearProgramInspectionSignature: string = linearInspection.signature;
const linearRequirements = linearProgram.requirements();
const linearRequirementsKind: "zgml.program.requirements" = linearRequirements.kind;
const linearRequirementsSignature: string = linearRequirements.signature;
const linearProgramInputLen: number = linearProgram.inputLen();
const linearProgramOutputLen: number = linearProgram.outputLen();
const linearProgramInputByteLength: number = linearProgram.inputByteLength();
const linearProgramOutputByteLength: number = linearProgram.outputByteLength();
const linearProgramWeightsLen: number = linearProgram.weightsLen();
const linearProgramWeightsByteLength: number = linearProgram.weightsByteLength();
const linearProgramBiasLen: number = linearProgram.biasLen();
const linearProgramBiasByteLength: number = linearProgram.biasByteLength();
const linearProgramParameterLen: number = linearProgram.parameterLen();
const linearProgramParameterByteLength: number = linearProgram.parameterByteLength();
const linearProgramBufferSizing: ProgramBufferSizing = linearProgram.bufferSizing();
const linearProgramBufferSizingKind: "program-buffer-sizing" = linearProgramBufferSizing.kind;
const linearProgramBufferSizingSignature: string = linearProgramBufferSizing.signature;
const linearProgramBufferSizingModelKind: ProgramRequirements["modelKind"] = linearProgramBufferSizing.modelKind;
const linearProgramMatchesBufferSizingSignature: boolean =
  linearProgram.matchesBufferSizingSignature(linearProgramBufferSizingSignature);
const linearCapabilities = linearProgram.capabilities();
const linearProgramCapabilitiesKind: "zgml.program.capabilities" = linearCapabilities.kind;
const linearProgramExecutionPlan: ProgramExecutionPlan = linearProgram.executionPlan();
const linearProgramModuleExecutionPlan: ProgramExecutionPlan = linearProgram.executionPlan(linear);
const linearProgramExecutionPlanAlias: ProgramExecutionPlan = linearProgram.execution_plan();
const linearProgramModuleExecutionPlanAlias: ProgramExecutionPlan = linearProgram.execution_plan(linear);
const linearProgramRequiredExecutionPlan: ProgramExecutionPlan = linearProgram.requireExecutionPlan();
const linearProgramRequiredModuleExecutionPlan: ProgramExecutionPlan = linearProgram.requireExecutionPlan(linear);
const linearProgramRequiredExecutionPlanAlias: ProgramExecutionPlan = linearProgram.require_execution_plan();
const linearProgramRequiredModuleExecutionPlanAlias: ProgramExecutionPlan = linearProgram.require_execution_plan(linear);
const linearProgramAcceptedExecutionPlan: boolean = program.acceptsProgramExecutionPlan(linearProgramExecutionPlan);
const linearProgramRequiredEvidencePlan: ProgramExecutionPlan = program.requireProgramExecutionPlan(linearProgramExecutionPlan);
const linearProgramAssertedEvidencePlan: ProgramExecutionPlan = program.assertProgramExecutionPlan(linearProgramExecutionPlan);
const linearProgramAssertedEvidencePlanAlias: ProgramExecutionPlan = program.assert_program_execution_plan(linearProgramExecutionPlan);
const linearProgramMatchedEvidencePlan: boolean = program.matchesProgramExecutionPlanSignature(linearProgramExecutionPlan, linearProgramExecutionPlan.signature);
const linearProgramInspectionAcceptedExecutionPlan: boolean = inspection.acceptsProgramExecutionPlan(linearProgramExecutionPlan);
const linearProgramInspectionRequiredEvidencePlan: ProgramExecutionPlan = inspection.requireProgramExecutionPlan(linearProgramExecutionPlan);
const linearProgramExecutionPlanKind: "zgml.program.execution-plan" = linearProgramExecutionPlan.kind;
const linearProgramExecutionPlanSignature: string = linearProgramExecutionPlan.signature;
const linearProgramExecutionPlanProgramKind: "generic" | "llama" = linearProgramExecutionPlan.programKind;
const linearProgramExecutionPlanRequirements: ProgramRequirements = linearProgramExecutionPlan.requirements;
const linearProgramExecutionPlanRequirementsKind: "zgml.program.requirements" = linearProgramExecutionPlanRequirements.kind;
const linearProgramExecutionPlanCapabilities: ProgramExecutionCapabilities = linearProgramExecutionPlan.capabilities;
const linearProgramExecutionPlanEvidence: ProgramCompileEvidence | null = linearProgramExecutionPlan.compileEvidence;
const linearProgramExecutionPlanBufferSizing: ProgramBufferSizing = linearProgramExecutionPlan.bufferSizing;
const linearProgramExecutionPlanBufferLayout: ProgramBufferLayout = linearProgramExecutionPlan.bufferLayout;
const linearProgramExecutionPlanInputShape: readonly number[] | null = linearProgramExecutionPlan.inputShape;
const linearProgramExecutionPlanOutputShape: readonly number[] = linearProgramExecutionPlan.outputShape;
const linearProgramExecutionPlanKernelPlan: ModuleKernelPlan | null = linearProgramExecutionPlan.kernelPlan;
const linearProgramExecutionPlanParameterNames: readonly string[] = linearProgramExecutionPlan.parameterNames;
const linearProgramExecutionPlanParameterInfos: readonly ModuleKernelParameterLayoutEntry[] = linearProgramExecutionPlan.parameterInfos;
const linearProgramExecutionPlanCompatibility: ProgramModuleCompatibility | null = linearProgramModuleExecutionPlan.moduleCompatibility;
const linearProgramExecutionPlanAcceptsModule: boolean | null = linearProgramModuleExecutionPlan.acceptsModule;
const linearProgramExecutionPlanDiagnostics: readonly ProgramRuntimeDiagnostic[] = linearProgramExecutionPlan.diagnostics;
const linearProgramBindingPlan: ProgramBindingPlan = linearProgram.bindingPlan(linear.bindParameters());
const linearProgramBindingPlanAlias: ProgramBindingPlan = linearProgram.binding_plan(linear.bind_parameters());
const linearProgramRequiredBindingPlan: ProgramBindingPlan = linearProgram.requireBindingPlan(linear.bindParameters());
const linearProgramRequiredBindingPlanAlias: ProgramBindingPlan = linearProgram.require_binding_plan(linear.bind_parameters());
const linearProgramAcceptedBindingPlan: boolean = program.acceptsProgramBindingPlan(linearProgramBindingPlan);
const linearProgramRequiredEvidenceBindingPlan: ProgramBindingPlan = program.requireProgramBindingPlan(linearProgramBindingPlan);
const linearProgramAssertedEvidenceBindingPlan: ProgramBindingPlan = program.assertProgramBindingPlan(linearProgramBindingPlan);
const linearProgramAssertedEvidenceBindingPlanAlias: ProgramBindingPlan = program.assert_program_binding_plan(linearProgramBindingPlan);
const linearProgramMatchedEvidenceBindingPlan: boolean = program.matchesProgramBindingPlanSignature(linearProgramBindingPlan, linearProgramBindingPlan.signature);
const linearProgramInspectionAcceptedBindingPlan: boolean = inspection.acceptsProgramBindingPlan(linearProgramBindingPlan);
const linearProgramInspectionRequiredEvidenceBindingPlan: ProgramBindingPlan = inspection.requireProgramBindingPlan(linearProgramBindingPlan);
const linearProgramBindingPlanKind: "zgml.program.binding-plan" = linearProgramBindingPlan.kind;
const linearProgramBindingPlanSignature: string = linearProgramBindingPlan.signature;
const linearProgramBindingPlanAccepted: boolean = linearProgramBindingPlan.accepted;
const linearProgramBindingPlanMode: ProgramBindingMode = linearProgramBindingPlan.mode;
const linearProgramBindingPlanDiagnostics: readonly ProgramBindingDiagnostic[] = linearProgramBindingPlan.diagnostics;
const linearProgramBindingPlanBufferLayout: ProgramBufferLayout | null = linearProgramBindingPlan.bufferLayout;
const linearProgramRejectedBindingPlan: ProgramBindingPlan = linearProgram.bindingPlan({ weights: new Float32Array(1) });
const linearProgramRejectedBindingPlanSignature: string = linearProgramRejectedBindingPlan.signature;
const linearProgramRejectedBindingReason: string | null = linearProgramRejectedBindingPlan.reason;
const linearCapabilitySignature: string = linearCapabilities.signature;
const linearMatchesCapabilitySignature: boolean = linearProgram.matchesCapabilitySignature(linearCapabilitySignature);
const linearMatchesCapabilitySignatureAlias: boolean = linearProgram.matches_capability_signature(linearCapabilitySignature);
const linearProgramCanExecuteAlias: boolean = linearProgram.can_execute();
const linearProgramCanBindExternalResourcesAlias: boolean = linearProgram.can_bind_external_resources();
const linearProgramHasFullDispatchPlanAlias: boolean = linearProgram.has_full_dispatch_plan();
const linearProgramExecutionModeAlias: ProgramExecutionCapabilities["mode"] = linearProgram.execution_mode();
const linearRuntimeProfile = linearProgram.runtimeProfile();
const linearRuntimeProfileAlias: RuntimeProfile = linearProgram.runtime_profile();
const linearProgramRuntimeProfileKind: "zgml.runtime.profile" = linearRuntimeProfile.kind;
const linearProgramRuntimeProfileSignature: string = linearRuntimeProfile.signature;
const linearProgramMatchesRuntimeProfileSignature: boolean =
  linearProgram.matchesRuntimeProfileSignature(linearProgramRuntimeProfileSignature);
const linearProgramMatchesRuntimeProfileSignatureAlias: boolean =
  linearProgram.matches_runtime_profile_signature(linearProgramRuntimeProfileSignature);
const inspectionNamespace: InspectionNamespace = inspection;
const linearProgramRuntimeProfileAccepted: boolean = inspection.acceptsRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileRequiredEvidence: RuntimeProfile = inspection.requireRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileAssertedEvidence: RuntimeProfile = inspection.assertRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileAssertedEvidenceAlias: RuntimeProfile = inspection.assert_runtime_profile(linearRuntimeProfile);
const linearProgramRuntimeProfileMatchedEvidence: boolean =
  inspection.matchesRuntimeProfileSignature(linearRuntimeProfile, linearRuntimeProfile.signature);
const linearProgramRuntimeProfileExpectation: RuntimeProfileExpectation = inspection.runtimeProfileExpectation(linearRuntimeProfile);
const linearProgramRuntimeProfileExpectationKind: "zgml.runtime.profile-expectation" =
  linearProgramRuntimeProfileExpectation.kind;
const linearProgramRuntimeProfileNoFallback: boolean = inspection.runtimeProfileHasNoFallback(linearRuntimeProfile);
const linearProgramRuntimeProfileNoSync: boolean = inspection.runtimeProfileHasNoSync(linearRuntimeProfile);
const linearProgramRuntimeProfileNoInvalidPatches: boolean =
  inspection.runtimeProfileHasNoInvalidRuntimePatches(linearRuntimeProfile);
const linearProgramRuntimeProfileRequired: RuntimeProfileExpectation =
  inspection.requireNoFallbackRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileNoSyncRequired: RuntimeProfileExpectation =
  inspection.requireNoSyncRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimePatchValid: RuntimeProfileExpectation =
  inspection.requireRuntimePatchValidProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileHot: RuntimeProfileExpectation =
  inspection.requireHotRuntimeProfile(linearRuntimeProfile);
const linearProgramRuntimeProfileExpectationFromProgram: RuntimeProfileExpectation =
  linearProgram.runtimeProfileExpectation();
const linearProgramRuntimeProfileExpectationAliasFromProgram: RuntimeProfileExpectation =
  linearProgram.runtime_profile_expectation();
const linearProgramRuntimeProfileNoFallbackFromProgram: boolean = linearProgram.runtimeProfileHasNoFallback();
const linearProgramRuntimeProfileNoFallbackAliasFromProgram: boolean = linearProgram.runtime_profile_has_no_fallback();
const linearProgramRuntimeProfileNoSyncFromProgram: boolean = linearProgram.runtimeProfileHasNoSync();
const linearProgramRuntimeProfileNoSyncAliasFromProgram: boolean = linearProgram.runtime_profile_has_no_sync();
const linearProgramRuntimeProfileNoInvalidPatchesFromProgram: boolean =
  linearProgram.runtimeProfileHasNoInvalidRuntimePatches();
const linearProgramRuntimeProfileNoInvalidPatchesAliasFromProgram: boolean =
  linearProgram.runtime_profile_has_no_invalid_runtime_patches();
const linearProgramRuntimeProfileRequiredFromProgram: RuntimeProfileExpectation =
  linearProgram.requireNoFallbackRuntimeProfile();
const linearProgramRuntimeProfileRequiredAliasFromProgram: RuntimeProfileExpectation =
  linearProgram.require_no_fallback_runtime_profile();
const linearProgramRuntimeProfileNoSyncRequiredFromProgram: RuntimeProfileExpectation =
  linearProgram.requireNoSyncRuntimeProfile();
const linearProgramRuntimeProfileNoSyncRequiredAliasFromProgram: RuntimeProfileExpectation =
  linearProgram.require_no_sync_runtime_profile();
const linearProgramRuntimePatchValidFromProgram: RuntimeProfileExpectation =
  linearProgram.requireRuntimePatchValidProfile();
const linearProgramRuntimePatchValidAliasFromProgram: RuntimeProfileExpectation =
  linearProgram.require_runtime_patch_valid_profile();
const linearProgramRuntimeProfileHotFromProgram: RuntimeProfileExpectation =
  linearProgram.requireHotRuntimeProfile();
const linearProgramRuntimeProfileHotAliasFromProgram: RuntimeProfileExpectation =
  linearProgram.require_hot_runtime_profile();
const linearBufferLayout = linearProgram.bufferLayout();
const linearBufferLayoutKind: "zgml.program.buffer-layout" = linearBufferLayout.kind;
const linearBufferLayoutSignature: string = linearBufferLayout.signature;
const linearBufferSlotNames: readonly string[] = linearProgram.bufferSlotNames();
const linearBufferInputSlot: ProgramBufferLayoutSlot | null = linearProgram.bufferSlot("input");
const linearAccepted: boolean = linearProgram.acceptsModule(linear);
const linearBindings: ModuleBindings = linear.bindParameters();
const linearBindingsAlias: ModuleBindings = linear.bind_parameters();
const linearBindingPlan: ModuleBindingPlan = nn.bindingPlan(linearBindings);
const linearBindingPlanAlias: ModuleBindingPlan = nn.binding_plan(linearBindingsAlias);
const linearRequiredBindingPlan: ModuleBindingPlan = nn.requireBindingPlan(linearBindings);
const linearRequiredBindingPlanAlias: ModuleBindingPlan = nn.require_binding_plan(linearBindingsAlias);
const linearAcceptedBindingPlan: boolean = nn.acceptsModuleBindingPlan(linearBindingPlan);
const linearRequiredEvidenceBindingPlan: ModuleBindingPlan = nn.requireModuleBindingPlan(linearBindingPlan);
const linearAssertedEvidenceBindingPlan: ModuleBindingPlan = nn.assertModuleBindingPlan(linearBindingPlan);
const linearAssertedEvidenceBindingPlanAlias: ModuleBindingPlan = nn.assert_module_binding_plan(linearBindingPlan);
const linearMatchedEvidenceBindingPlan: boolean = nn.matchesModuleBindingPlanSignature(linearBindingPlan, linearBindingPlan.signature);
const linearInspectionAcceptedBindingPlan: boolean = inspection.acceptsModuleBindingPlan(linearBindingPlan);
const linearInspectionRequiredEvidenceBindingPlan: ModuleBindingPlan = inspection.requireModuleBindingPlan(linearBindingPlan);
const linearInspectionAcceptedCompilePlan: boolean = inspection.acceptsModuleCompilePlan(linearModuleCompilePlan);
const linearInspectionRequiredEvidenceCompilePlan: ModuleCompileExplanation = inspection.requireModuleCompilePlan(linearModuleCompilePlan);
const linearInspectionAssertedEvidenceCompilePlan: ModuleCompileExplanation = inspection.assertModuleCompilePlan(linearModuleCompilePlan);
const linearInspectionAssertedEvidenceCompilePlanAlias: ModuleCompileExplanation = inspection.assert_module_compile_plan(linearModuleCompilePlan);
const linearInspectionMatchedEvidenceCompilePlan: boolean = inspection.matchesModuleCompilePlanSignature(linearModuleCompilePlan, linearModuleCompilePlan.signature);
const linearBindingPlanKind: "zgml.nn.module-bindings-plan" = linearBindingPlan.kind;
const linearBindingPlanSignature: string = linearBindingPlan.signature;
const linearBindingPlanIsModuleBindings: boolean = linearBindingPlan.moduleBindings;
const linearBindingPlanPlacementMode: ModuleBindingPlacementMode = linearBindingPlan.placementMode;
const linearBindingPlanUsesNativeBuffers: boolean = linearBindingPlan.usesNativeBuffers;
const linearBindingPlanNativeSlots: readonly string[] = linearBindingPlan.nativeSlots;
const linearBindingPlanHostSlots: readonly string[] = linearBindingPlan.hostSlots;
const linearBindingPlanSupport: ModuleCompileSupport | null = linearBindingPlan.support;
const linearBindingPlanParameterNames: readonly string[] = linearBindingPlan.parameterNames;
const linearBindingPlanParameterInfos: readonly ModuleParameterInfo[] = linearBindingPlan.parameterInfos;
const rawLinearProgramBindings: ProgramBindings = { weights: new Float32Array(6), bias: new Float32Array(3) };
const rawLinearBindingPlan: ModuleBindingPlan = nn.bindingPlan(rawLinearProgramBindings);
const rawLinearBindingPlanSignature: string = rawLinearBindingPlan.signature;
const rawLinearBindingPlanPlacementMode: ModuleBindingPlacementMode = rawLinearBindingPlan.placementMode;
const rawLinearProgramBindingPlan: ProgramBindingPlan = linearProgram.bindingPlan(rawLinearProgramBindings);
const rawLinearProgramBindingPlanSignature: string = rawLinearProgramBindingPlan.signature;
const rawLinearBindingPlanSupport: ModuleCompileSupport | null = rawLinearBindingPlan.support;
// @ts-expect-error raw ProgramBindings are not binding-time module evidence.
const rawLinearModuleBindings: ModuleBindings = rawLinearProgramBindings;
const spreadLinearBindings: ModuleBindings = { ...linearBindings };
const linearSession: Session = linearProgram.bind(linearBindings);
const spreadLinearSession: Session = linearProgram.bind(spreadLinearBindings);
const linearModuleSessionViaBind: Session = linearProgram.bind(linear);
const linearModuleSession: Session = linearProgram.bindModule(linear);
const typedLinearSession: Session<readonly [2], readonly [3]> = typedLinearProgram.bind(linear.bindParameters(linearTypedCompileOptions));
const typedLinearModuleSession: Session<readonly [2], readonly [3]> = typedLinearProgram.bindModule(linear);
const typedLinearClone = nn.linear(2, 3);
const typedLinearCloneState: ModuleStateSnapshot = linear.stateDict("compiled");
const typedLinearLoadedClone: typeof typedLinearClone = typedLinearClone.loadStateDict(typedLinearCloneState, {
  strict: true,
  prefix: "compiled",
});
const typedLinearCloneSession: Session<readonly [2], readonly [3]> = typedLinearProgram.bindModule(typedLinearLoadedClone);
const typedLinearSessionInputShape: readonly [2] = typedLinearSession.inputShape();
const typedLinearSessionOutputShape: readonly [3] = typedLinearSession.outputShape();
const typedLinearSessionExecutionPlan: SessionExecutionPlan<readonly [2], readonly [3]> =
  typedLinearSession.executionPlan({ input: tensor([1, 2], [2] as const), output: false });
const typedLinearSessionHotPathPlan: SessionExecutionPlan<readonly [2], readonly [3]> =
  typedLinearSession.hotPathPlan({ input: tensor([1, 2], [2] as const), output: false });
const typedLinearSessionExecutionPlanInputShape: readonly [2] = typedLinearSessionExecutionPlan.inputShape;
const typedLinearSessionExecutionPlanOutputShape: readonly [3] = typedLinearSessionExecutionPlan.outputShape;
const linearSessionInspection = linearSession.inspect();
const linearSessionInspectionKind: "zgml.session.inspection" = linearSessionInspection.kind;
const linearSessionInspectionSignature: string = linearSessionInspection.signature;
const linearSessionInputLen: number = linearSession.inputLen();
const linearSessionOutputLen: number = linearSession.outputLen();
const linearSessionInputByteLength: number = linearSession.inputByteLength();
const linearSessionOutputByteLength: number = linearSession.outputByteLength();
const linearSessionWeightsLen: number = linearSession.weightsLen();
const linearSessionWeightsByteLength: number = linearSession.weightsByteLength();
const linearSessionBiasLen: number = linearSession.biasLen();
const linearSessionBiasByteLength: number = linearSession.biasByteLength();
const linearSessionParameterLen: number = linearSession.parameterLen();
const linearSessionParameterByteLength: number = linearSession.parameterByteLength();
const linearSessionBufferSizing: SessionBufferSizing = linearSession.bufferSizing();
const linearSessionBufferSizingKind: "session-buffer-sizing" = linearSessionBufferSizing.kind;
const linearSessionBufferSizingSignature: string = linearSessionBufferSizing.signature;
const linearSessionBufferSizingModelKind: ProgramRequirements["modelKind"] = linearSessionBufferSizing.modelKind;
const linearSessionMatchesBufferSizingSignature: boolean =
  linearSession.matchesBufferSizingSignature(linearSessionBufferSizingSignature);
const linearSessionRuntimeProfile = linearSession.runtimeProfile();
const linearSessionRuntimeProfileAlias: RuntimeProfile = linearSession.runtime_profile();
const linearSessionRuntimeProfileKind: "zgml.runtime.profile" = linearSessionRuntimeProfile.kind;
const linearSessionRuntimeProfileSignature: string = linearSessionRuntimeProfile.signature;
const linearSessionMatchesRuntimeProfileSignature: boolean =
  linearSession.matchesRuntimeProfileSignature(linearSessionRuntimeProfileSignature);
const linearSessionMatchesRuntimeProfileSignatureAlias: boolean =
  linearSession.matches_runtime_profile_signature(linearSessionRuntimeProfileSignature);
const linearSessionRuntimeProfileAccepted: boolean = inspection.acceptsRuntimeProfile(linearSessionRuntimeProfile);
const linearSessionRuntimeProfileRequiredEvidence: RuntimeProfile = inspection.requireRuntimeProfile(linearSessionRuntimeProfile);
const linearSessionRuntimeProfileMatchedEvidence: boolean =
  inspection.matchesRuntimeProfileSignature(linearSessionRuntimeProfile, linearSessionRuntimeProfile.signature);
const linearSessionRuntimeProfileExpectation: RuntimeProfileExpectation =
  inspection.runtimeProfileExpectation(linearSessionRuntimeProfile);
const linearSessionRuntimeProfileNoFallback: boolean = inspection.runtimeProfileHasNoFallback(linearSessionRuntimeProfile);
const linearSessionRuntimeProfileNoInvalidPatches: boolean =
  inspection.runtimeProfileHasNoInvalidRuntimePatches(linearSessionRuntimeProfile);
const linearSessionRuntimeProfileExpectationFromSession: RuntimeProfileExpectation =
  linearSession.runtimeProfileExpectation();
const linearSessionRuntimeProfileExpectationAliasFromSession: RuntimeProfileExpectation =
  linearSession.runtime_profile_expectation();
const linearSessionRuntimeProfileNoFallbackFromSession: boolean = linearSession.runtimeProfileHasNoFallback();
const linearSessionRuntimeProfileNoFallbackAliasFromSession: boolean = linearSession.runtime_profile_has_no_fallback();
const linearSessionRuntimeProfileNoSyncFromSession: boolean = linearSession.runtimeProfileHasNoSync();
const linearSessionRuntimeProfileNoSyncAliasFromSession: boolean = linearSession.runtime_profile_has_no_sync();
const linearSessionRuntimeProfileNoInvalidPatchesFromSession: boolean =
  linearSession.runtimeProfileHasNoInvalidRuntimePatches();
const linearSessionRuntimeProfileNoInvalidPatchesAliasFromSession: boolean =
  linearSession.runtime_profile_has_no_invalid_runtime_patches();
const linearSessionRuntimeProfileRequiredFromSession: RuntimeProfileExpectation =
  linearSession.requireNoFallbackRuntimeProfile();
const linearSessionRuntimeProfileRequiredAliasFromSession: RuntimeProfileExpectation =
  linearSession.require_no_fallback_runtime_profile();
const linearSessionRuntimeProfileNoSyncRequiredFromSession: RuntimeProfileExpectation =
  linearSession.requireNoSyncRuntimeProfile();
const linearSessionRuntimeProfileNoSyncRequiredAliasFromSession: RuntimeProfileExpectation =
  linearSession.require_no_sync_runtime_profile();
const linearSessionRuntimePatchValidFromSession: RuntimeProfileExpectation =
  linearSession.requireRuntimePatchValidProfile();
const linearSessionRuntimePatchValidAliasFromSession: RuntimeProfileExpectation =
  linearSession.require_runtime_patch_valid_profile();
const linearSessionRuntimeProfileHotFromSession: RuntimeProfileExpectation =
  linearSession.requireHotRuntimeProfile();
const linearSessionRuntimeProfileHotAliasFromSession: RuntimeProfileExpectation =
  linearSession.require_hot_runtime_profile();
const linearSessionCallProfile: SessionCallProfile = linearSession.sessionCallProfile();
const linearSessionCallProfileAlias: SessionCallProfile = linearSession.session_call_profile();
const linearSessionCallProfileKind: "zgml.session.call-profile" = linearSessionCallProfile.kind;
const linearSessionCallProfileSignature: string = linearSessionCallProfile.signature;
const linearSessionMatchesCallProfileSignature: boolean =
  linearSession.matchesSessionCallProfileSignature(linearSessionCallProfileSignature);
const linearSessionMatchesCallProfileSignatureAlias: boolean =
  linearSession.matches_session_call_profile_signature(linearSessionCallProfileSignature);
const linearSessionCallProfileAccepted: boolean = session.acceptsSessionCallProfile(linearSessionCallProfile);
const linearSessionCallProfileRequiredEvidence: SessionCallProfile =
  session.requireSessionCallProfile(linearSessionCallProfile);
const linearSessionCallProfileAssertedEvidence: SessionCallProfile =
  session.assertSessionCallProfile(linearSessionCallProfile);
const linearSessionCallProfileAssertedEvidenceAlias: SessionCallProfile =
  session.assert_session_call_profile(linearSessionCallProfile);
const linearSessionCallProfileMatchedEvidence: boolean =
  session.matchesSessionCallProfileSignature(linearSessionCallProfile, linearSessionCallProfile.signature);
const linearInspectionSessionCallProfileAccepted: boolean =
  inspection.acceptsSessionCallProfile(linearSessionCallProfile);
const linearInspectionSessionCallProfileRequiredEvidence: SessionCallProfile =
  inspection.requireSessionCallProfile(linearSessionCallProfile);
const linearInspectionSessionCallProfileMatchedEvidence: boolean =
  inspection.matchesSessionCallProfileSignature(linearSessionCallProfile, linearSessionCallProfile.signature);
const linearPrefillCount: number = linearSessionCallProfile.prefillTensorCount;
linearSession.resetSessionCallProfile();
linearSession.reset_session_call_profile();
const linearSessionStepContract: SessionStepContract = linearSession.stepContract();
const linearSessionStepContractAlias: SessionStepContract = linearSession.step_contract();
const linearSessionStepContractInputShape: readonly number[] = linearSessionStepContract.inputShape;
const linearSessionStepContractSignature: string = linearSessionStepContract.signature;
const linearSessionStepContractInputByteLength: number = linearSessionStepContract.inputByteLength;
const linearSessionStepContractOutputByteLength: number = linearSessionStepContract.outputByteLength;
const linearSessionStepContractInputSlotName: "input" = linearSessionStepContract.inputSlotName;
const linearSessionStepContractOutputSlotName: "output" = linearSessionStepContract.outputSlotName;
const linearSessionStepContractInputSlotRole: "step-input" = linearSessionStepContract.inputSlotRole;
const linearSessionStepContractOutputSlotRole: "step-output" = linearSessionStepContract.outputSlotRole;
const linearSessionStepContractAcceptsNoInput: boolean = linearSessionStepContract.acceptsNoInput;
const linearSessionStepContractRequiresInput: boolean = linearSessionStepContract.requiresInput;
const linearSessionStepContractDefaultOutput: SessionDefaultOutputKind = linearSessionStepContract.defaultOutput;
const linearSessionStepContractDefaultReadbackRequired: boolean = linearSessionStepContract.defaultReadbackRequired;
const linearSessionStepContractDefaultAllocationFree: boolean = linearSessionStepContract.defaultAllocationFree;
const linearSessionStepContractNoOutputEffect: SessionNoOutputEffect = linearSessionStepContract.noOutputEffect;
const linearSessionMatchesStepContractSignature: boolean =
  linearSession.matchesStepContractSignature(linearSessionStepContractSignature);
const linearSessionMatchesStepContractSignatureAlias: boolean =
  linearSession.matches_step_contract_signature(linearSessionStepContractSignature);
const linearSessionStepParamsPreflight: SessionStepParamsCompatibility = linearSession.preflightStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionStepParamsPreflightAlias: SessionStepParamsCompatibility = linearSession.preflight_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionStepParamsCompatibility: SessionStepParamsCompatibility = linearSession.stepParamsCompatibility({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionStepParamsCompatibilityAlias: SessionStepParamsCompatibility = linearSession.step_params_compatibility({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionStepParamsKind: "zgml.step-params.compatibility" = linearSessionStepParamsCompatibility.kind;
const linearSessionStepParamsAccepted: boolean = linearSessionStepParamsCompatibility.accepted;
const linearSessionStepParamsCanExecute: boolean = linearSessionStepParamsCompatibility.canExecute;
const linearSessionStepParamsStatus: "accepted" | "rejected" = linearSessionStepParamsCompatibility.status;
const linearSessionStepParamsContractKind: SessionStepParamsCompatibility["contractKind"] = linearSessionStepParamsCompatibility.contractKind;
const linearSessionStepParamsContractSignature: string = linearSessionStepParamsCompatibility.contractSignature;
const linearSessionStepParamsContractPosition: number | null = linearSessionStepParamsCompatibility.contractPosition;
const linearSessionStepParamsSignature: string = linearSessionStepParamsCompatibility.stepParamsSignature;
const linearSessionStepParamsRejectionCode: SessionStepParamsDiagnosticCode | null = linearSessionStepParamsCompatibility.rejectionCode;
const linearSessionStepParamsStateEffect: SessionStepParamsStateEffect = linearSessionStepParamsCompatibility.stateEffect;
const linearSessionStepParamsAllocationFree: boolean = linearSessionStepParamsCompatibility.allocationFree;
const linearSessionStepParamsInputSource: SessionStepParamsInputSource = linearSessionStepParamsCompatibility.inputSource;
const linearSessionStepParamsInputOwnership: SessionStepParamsInputOwnership = linearSessionStepParamsCompatibility.inputOwnership;
const linearSessionStepParamsReadsInput: boolean = linearSessionStepParamsCompatibility.readsInput;
const linearSessionStepParamsOutputTarget: SessionStepParamsOutputTarget = linearSessionStepParamsCompatibility.outputTarget;
const linearSessionStepParamsOutputEffect: SessionStepParamsOutputEffect = linearSessionStepParamsCompatibility.outputEffect;
const linearSessionStepParamsOutputOwnership: SessionStepParamsOutputOwnership = linearSessionStepParamsCompatibility.outputOwnership;
const linearSessionStepParamsOutputReturnOwnership: SessionStepParamsOutputReturnOwnership = linearSessionStepParamsCompatibility.outputReturnOwnership;
const linearSessionStepParamsWritesOutput: boolean = linearSessionStepParamsCompatibility.writesOutput;
const linearSessionStepParamsReadbackRequired: boolean = linearSessionStepParamsCompatibility.readbackRequired;
const linearSessionStepParamsReadbackFree: boolean = linearSessionStepParamsCompatibility.readbackFree;
const linearSessionStepParamsRuntimeOutputAllocationFree: boolean = linearSessionStepParamsCompatibility.runtimeOutputAllocationFree;
const linearSessionStepParamsHotPath: boolean = linearSessionStepParamsCompatibility.hotPath;
const linearSessionStepParamsHotPathStatus: SessionStepParamsHotPathStatus = linearSessionStepParamsCompatibility.hotPathStatus;
const linearSessionStepParamsHotPathBlockers: readonly SessionStepParamsHotPathBlocker[] = linearSessionStepParamsCompatibility.hotPathBlockers;
const linearSessionStepParamsInputElementType: SessionStepParamsElementType = linearSessionStepParamsCompatibility.inputElementType;
const linearSessionStepParamsOutputElementType: SessionStepParamsElementType = linearSessionStepParamsCompatibility.outputElementType;
const linearSessionStepParamsInputElementLength: number = linearSessionStepParamsCompatibility.inputElementLength;
const linearSessionStepParamsOutputElementLength: number = linearSessionStepParamsCompatibility.outputElementLength;
const linearSessionStepParamsInputShape: readonly number[] = linearSessionStepParamsCompatibility.inputShape;
const linearSessionStepParamsOutputShape: readonly number[] = linearSessionStepParamsCompatibility.outputShape;
const linearSessionStepParamsInputShapeSignature: string = linearSessionStepParamsCompatibility.inputShapeSignature;
const linearSessionStepParamsOutputShapeSignature: string = linearSessionStepParamsCompatibility.outputShapeSignature;
const linearSessionStepParamsInputByteLength: number = linearSessionStepParamsCompatibility.inputByteLength;
const linearSessionStepParamsOutputByteLength: number = linearSessionStepParamsCompatibility.outputByteLength;
const linearSessionStepParamsDiagnosticCode: SessionStepParamsDiagnosticCode | undefined = linearSessionStepParamsCompatibility.diagnostics[0]?.code;
const linearSessionStepParamsDiagnosticExpectedLength: number | undefined = linearSessionStepParamsCompatibility.diagnostics[0]?.expectedLength;
const linearSessionStepParamsDiagnosticExpectedShape: readonly number[] | undefined = linearSessionStepParamsCompatibility.diagnostics[0]?.expectedShape;
const stepParamsNamespaceAccepts: boolean = stepParams.acceptsStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceEvidenceUnknown: unknown = linearSessionStepParamsCompatibility;
const stepParamsNamespaceIsEvidence: boolean = stepParams.isStepParamsCompatibility(stepParamsNamespaceEvidenceUnknown);
if (stepParams.isStepParamsCompatibility(stepParamsNamespaceEvidenceUnknown)) {
  const narrowedStepParamsSignature: string = stepParamsNamespaceEvidenceUnknown.stepParamsSignature;
  void narrowedStepParamsSignature;
}
const stepParamsNamespaceRequiredEvidence: SessionStepParamsCompatibility =
  stepParams.requireStepParamsCompatibility(stepParamsNamespaceEvidenceUnknown);
const stepParamsNamespaceAssertedEvidence: SessionStepParamsCompatibility =
  stepParams.assertStepParamsCompatibility(stepParamsNamespaceEvidenceUnknown);
const stepParamsNamespaceAssertedEvidenceAlias: SessionStepParamsCompatibility =
  stepParams.assert_step_params_compatibility(stepParamsNamespaceEvidenceUnknown);
const stepParamsNamespaceCanExecute: boolean = stepParams.canExecuteStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredCanExecute: SessionStepParamsCompatibility = stepParams.requireCanExecuteStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceAcceptsAllocationFree: boolean = stepParams.acceptsAllocationFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredAllocationFree: SessionStepParamsCompatibility = stepParams.requireAllocationFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceAcceptsRuntimeOutputAllocationFree: boolean =
  stepParams.acceptsRuntimeOutputAllocationFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredRuntimeOutputAllocationFree: SessionStepParamsCompatibility =
  stepParams.requireRuntimeOutputAllocationFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceAcceptsNoReadback: boolean = stepParams.acceptsNoReadbackStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredNoReadback: SessionStepParamsCompatibility = stepParams.requireNoReadbackStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceAcceptsReadbackFree: boolean = stepParams.acceptsReadbackFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredReadbackFree: SessionStepParamsCompatibility = stepParams.requireReadbackFreeStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceAcceptsHot: boolean = stepParams.acceptsHotStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceRequiredHot: SessionStepParamsCompatibility = stepParams.requireHotStepParamsCompatibility(linearSessionStepParamsCompatibility);
const stepParamsNamespaceMatchesSignature: boolean = stepParams.matchesStepParamsSignature(linearSessionStepParamsCompatibility, linearSessionStepParamsSignature);
const stepParamsNamespaceMatchesCompatibility: boolean = stepParams.matchesStepParamsCompatibility(linearSessionStepParamsCompatibility, linearSessionStepParamsCompatibility);
const linearSessionAcceptsStepParams: boolean = linearSession.acceptsStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsStepParamsAlias: boolean = linearSession.accepts_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredCanExecuteStepParams: SessionStepParamsCompatibility = linearSession.requireCanExecuteStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredCanExecuteStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_can_execute_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsAllocationFreeStepParams: boolean = linearSession.acceptsAllocationFreeStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsAllocationFreeStepParamsAlias: boolean = linearSession.accepts_allocation_free_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredAllocationFreeStepParams: SessionStepParamsCompatibility = linearSession.requireAllocationFreeStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredAllocationFreeStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_allocation_free_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredRuntimeOutputAllocationFreeStepParams: SessionStepParamsCompatibility = linearSession.requireRuntimeOutputAllocationFreeStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredRuntimeOutputAllocationFreeStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_runtime_output_allocation_free_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsNoReadbackStepParams: boolean = linearSession.acceptsNoReadbackStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsNoReadbackStepParamsAlias: boolean = linearSession.accepts_no_readback_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredNoReadbackStepParams: SessionStepParamsCompatibility = linearSession.requireNoReadbackStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredNoReadbackStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_no_readback_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredReadbackFreeStepParams: SessionStepParamsCompatibility = linearSession.requireReadbackFreeStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsReadbackFreeStepParamsAlias: boolean = linearSession.accepts_readback_free_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredReadbackFreeStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_readback_free_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsHotStepParams: boolean = linearSession.acceptsHotStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptsHotStepParamsAlias: boolean = linearSession.accepts_hot_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredHotStepParams: SessionStepParamsCompatibility = linearSession.requireHotStepParams({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredHotStepParamsAlias: SessionStepParamsCompatibility = linearSession.require_hot_step_params({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionExecutionPlan: SessionExecutionPlan = linearSession.executionPlan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionExecutionPlanAlias: SessionExecutionPlan = linearSession.execution_plan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredExecutionPlan: SessionExecutionPlan = linearSession.requireExecutionPlan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionRequiredExecutionPlanAlias: SessionExecutionPlan = linearSession.require_execution_plan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionHotPathPlan: SessionExecutionPlan = linearSession.hotPathPlan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionHotPathPlanAlias: SessionExecutionPlan = linearSession.hot_path_plan({ input: tensor([1, 2], [2] as const), output: false });
const linearSessionAcceptedExecutionPlan: boolean = session.acceptsSessionExecutionPlan(linearSessionExecutionPlan);
const linearSessionRequiredEvidencePlan: SessionExecutionPlan = session.requireSessionExecutionPlan(linearSessionExecutionPlan);
const linearSessionAssertedEvidencePlan: SessionExecutionPlan = session.assertSessionExecutionPlan(linearSessionExecutionPlan);
const linearSessionAssertedEvidencePlanAlias: SessionExecutionPlan = session.assert_session_execution_plan(linearSessionExecutionPlan);
const linearSessionMatchedEvidencePlan: boolean = session.matchesSessionExecutionPlanSignature(linearSessionExecutionPlan, linearSessionExecutionPlan.signature);
const linearSessionInspectionAcceptedExecutionPlan: boolean = inspection.acceptsSessionExecutionPlan(linearSessionExecutionPlan);
const linearSessionInspectionRequiredEvidencePlan: SessionExecutionPlan = inspection.requireSessionExecutionPlan(linearSessionExecutionPlan);
const linearSessionExecutionPlanKind: "zgml.session.execution-plan" = linearSessionExecutionPlan.kind;
const linearSessionExecutionPlanContract: SessionStepContract | LlamaSessionStepContract = linearSessionExecutionPlan.contract;
const linearSessionExecutionPlanCompatibility: SessionStepParamsCompatibility = linearSessionExecutionPlan.compatibility;
const linearSessionExecutionPlanSignature: string = linearSessionExecutionPlan.signature;
const linearSessionExecutionPlanHotPath: boolean = linearSessionExecutionPlan.hotPath;
const linearSessionExecutionPlanStatus: SessionStepParamsHotPathStatus = linearSessionExecutionPlan.hotPathStatus;
const linearSessionExecutionPlanBlockers: readonly SessionStepParamsHotPathBlocker[] = linearSessionExecutionPlan.hotPathBlockers;
const linearSessionExecutionPlanInputOwnership: SessionStepParamsInputOwnership = linearSessionExecutionPlan.inputOwnership;
const linearSessionExecutionPlanReadsInput: boolean = linearSessionExecutionPlan.readsInput;
const linearSessionExecutionPlanOutputOwnership: SessionStepParamsOutputOwnership = linearSessionExecutionPlan.outputOwnership;
const linearSessionExecutionPlanOutputReturnOwnership: SessionStepParamsOutputReturnOwnership = linearSessionExecutionPlan.outputReturnOwnership;
const linearSessionExecutionPlanWritesOutput: boolean = linearSessionExecutionPlan.writesOutput;
const linearSessionExecutionPlanInputElementType: SessionStepParamsElementType = linearSessionExecutionPlan.inputElementType;
const linearSessionExecutionPlanOutputElementType: SessionStepParamsElementType = linearSessionExecutionPlan.outputElementType;
const linearSessionExecutionPlanInputElementLength: number = linearSessionExecutionPlan.inputElementLength;
const linearSessionExecutionPlanOutputElementLength: number = linearSessionExecutionPlan.outputElementLength;
const linearSessionExecutionPlanInputShape: readonly number[] = linearSessionExecutionPlan.inputShape;
const linearSessionExecutionPlanOutputShape: readonly number[] = linearSessionExecutionPlan.outputShape;
const linearSessionExecutionPlanInputShapeSignature: string = linearSessionExecutionPlan.inputShapeSignature;
const linearSessionExecutionPlanOutputShapeSignature: string = linearSessionExecutionPlan.outputShapeSignature;
const linearSessionExecutionPlanRejectionCode: SessionStepParamsDiagnosticCode | null = linearSessionExecutionPlan.rejectionCode;
const linearSessionHotPathPlanSignature: string = linearSessionHotPathPlan.signature;
const linearSessionHotPathPlanStepParamsSignature: string = linearSessionHotPathPlan.stepParamsSignature;
const linearSessionMatchesStepParamsSignature: boolean = linearSession.matchesStepParamsSignature({ input: tensor([1, 2], [2] as const), output: false }, linearSessionStepParamsSignature);
const linearSessionMatchesStepParamsSignatureAlias: boolean = linearSession.matches_step_params_signature({ input: tensor([1, 2], [2] as const), output: false }, linearSessionStepParamsSignature);
const linearSessionMatchesStepParamsCompatibility: boolean = linearSession.matchesStepParamsCompatibility({ input: tensor([1, 2], [2] as const), output: false }, linearSessionStepParamsCompatibility);
const linearSessionMatchesStepParamsCompatibilityAlias: boolean = linearSession.matches_step_params_compatibility({ input: tensor([1, 2], [2] as const), output: false }, linearSessionStepParamsCompatibility);
const linearSessionRejectedRawStepParams: SessionStepParamsCompatibility = linearSession.stepParamsCompatibility(new Float32Array([1, 2]));
const linearSessionRejectsRawStepParams: boolean = linearSession.acceptsStepParams(new Float32Array([1, 2]));
const linearSessionBufferLayout = linearSession.bufferLayout();
const linearSessionBufferLayoutKind: "zgml.program.buffer-layout" = linearSessionBufferLayout.kind;
const linearSessionBufferLayoutSignature: string = linearSessionBufferLayout.signature;
const linearSessionBufferSlotNames: readonly string[] = linearSession.bufferSlotNames();
const linearSessionOutputSlot: ProgramBufferLayoutSlot | null = linearSession.bufferSlot("output");
const linearSessionParameterLayout: ModuleKernelParameterLayout | null = linearSession.parameterLayout();
const linearSessionParameterName: string | undefined = linearSessionParameterLayout?.parameters[0]?.name;
const linearSessionParameterNames: readonly string[] = linearSession.parameterNames();
const linearSessionParameterInfos: readonly ModuleKernelParameterLayoutEntry[] = linearSession.parameterInfos();
const linearSessionParameterInfoByName: ModuleKernelParameterLayoutEntry | null = linearSession.parameterInfo("0.weight");
const linearSessionParameterInfoByIndex: ModuleKernelParameterLayoutEntry | null = linearSession.parameterInfo(0);
const linearInput = tensor([1, 2], [2] as const);
const compiledInference: CompiledInference<readonly [2], readonly [3]> = compile.compileForInference(linear, linearTypedCompileOptions);
const compiledInferenceAlias: CompiledInference<readonly [2], readonly [3]> = compile.compile_for_inference(linear, linearTypedCompileOptions);
const compiledInferenceForAlias: CompiledInference<readonly [2], readonly [3]> = compile.forInference(linear, linearTypedCompileOptions);
const compiledInferenceForSnakeAlias: CompiledInference<readonly [2], readonly [3]> = compile.for_inference(linear, linearTypedCompileOptions);
const compiledInferenceNativeAlias: CompiledInference<readonly [2], readonly [3]> = compile.native(linear, linearTypedCompileOptions);
const compiledInferenceParameterBindingPlan: ModuleBindingPlan<readonly [2], readonly [3]> | null = compiledInference.parameterBindingPlan();
const compiledInferenceParameterBindingPlanAlias: ModuleBindingPlan<readonly [2], readonly [3]> | null = compiledInference.parameter_binding_plan();
const compiledInferenceProgramBindingPlan: ProgramBindingPlan<readonly [2], readonly [3]> = compiledInference.programBindingPlan();
const compiledInferenceProgramBindingPlanAlias: ProgramBindingPlan<readonly [2], readonly [3]> = compiledInference.program_binding_plan();
const compiledInferenceLayerList: CompiledInference<readonly [2], readonly [3]> = compile.compileForInference([linear], linearTypedCompileOptions);
const compiledInferenceLayerListAlias: CompiledInference<readonly [2], readonly [3]> = compile.native([linear], linearTypedCompileOptions);
const rootNativeInference: CompiledInference<readonly [2], readonly [3]> = native(linear, linearTypedCompileOptions);
const rootInferenceAlias: CompiledInference<readonly [2], readonly [3]> = inference(linear, linearTypedCompileOptions);
const rootForInferenceAlias: CompiledInference<readonly [2], readonly [3]> = forInference(linear, linearTypedCompileOptions);
const rootForInferenceSnakeAlias: CompiledInference<readonly [2], readonly [3]> = for_inference(linear, linearTypedCompileOptions);
const rootRunOutput: Tensor<readonly [3]> = run(linear, linearInput, linearTypedCompileOptions);
const rootInferOutput: Tensor<readonly [3]> = infer(linear, linearInput, linearTypedCompileOptions);
const rootPredictOutput: Tensor<readonly [3]> = predict(linear, linearInput, linearTypedCompileOptions);
const rootRunIntoOutput: Float32Array = runInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const rootRunIntoSnakeOutput: Float32Array = run_into(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const rootInferIntoOutput: Float32Array = inferInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const rootInferIntoSnakeOutput: Float32Array = infer_into(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const rootPredictIntoOutput: Float32Array = predictInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const rootPredictIntoSnakeOutput: Float32Array = predict_into(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const zgmlNativeInference: CompiledInference<readonly [2], readonly [3]> = zgml.native(linear, linearTypedCompileOptions);
const zgmlNativeLayerListInference: CompiledInference<readonly [2], readonly [3]> = zgml.native([linear], linearTypedCompileOptions);
const zgmlInferenceAlias: CompiledInference<readonly [2], readonly [3]> = zgml.inference(linear, linearTypedCompileOptions);
const zgmlForInferenceAlias: CompiledInference<readonly [2], readonly [3]> = zgml.forInference(linear, linearTypedCompileOptions);
const zgmlForInferenceSnakeAlias: CompiledInference<readonly [2], readonly [3]> = zgml.for_inference(linear, linearTypedCompileOptions);
const zgmlCompiledInference: CompiledInference<readonly [2], readonly [3]> = zgml.compileForInference(linear, linearTypedCompileOptions);
const zgmlCompileInferenceAlias: CompiledInference<readonly [2], readonly [3]> = zgml.compileInference(linear, linearTypedCompileOptions);
const zgmlCompiledInferenceAlias: CompiledInference<readonly [2], readonly [3]> = zgml.compile_for_inference(linear, linearTypedCompileOptions);
const zgmlInferOutput: Tensor<readonly [3]> = zgml.infer(linear, linearInput, linearTypedCompileOptions);
const zgmlPredictOutput: Tensor<readonly [3]> = zgml.predict(linear, linearInput, linearTypedCompileOptions);
const zgmlInferIntoOutput: Float32Array = zgml.inferInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const zgmlPredictIntoOutput: Float32Array = zgml.predictInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const compileInferOutput: Tensor<readonly [3]> = compile.infer(linear, linearInput, linearTypedCompileOptions);
const compilePredictOutput: Tensor<readonly [3]> = compile.predict(linear, linearInput, linearTypedCompileOptions);
const compileInferIntoOutput: Float32Array = compile.inferInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const compileInferIntoSnakeOutput: Float32Array = compile.infer_into(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const compilePredictIntoOutput: Float32Array = compile.predictInto(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const compilePredictIntoSnakeOutput: Float32Array = compile.predict_into(new Float32Array(3), linear, linearInput, linearTypedCompileOptions);
const compiledInferenceNative: true = compiledInference.native;
const compiledInferenceProgram: Program<readonly [2], readonly [3]> = compiledInference.program;
const compiledInferenceSession: Session<readonly [2], readonly [3]> = compiledInference.session;
const compiledInferenceExecutionPlan: ProgramExecutionPlan<readonly [2], readonly [3]> = compiledInference.executionPlan();
const compiledInferenceRequiredExecutionPlan: ProgramExecutionPlan<readonly [2], readonly [3]> = compiledInference.requireExecutionPlan();
const compiledInferenceForward: Tensor<readonly [3]> = compiledInference.forward(linearInput);
const compiledInferenceCall: Tensor<readonly [3]> = compiledInference.call(linearInput);
const compiledInferenceDunderCall: Tensor<readonly [3]> = compiledInference.__call__(linearInput);
const compiledInferenceStepTensor: Tensor<readonly [3]> = compiledInference.stepTensor(linearInput);
const compiledInferenceInto: Float32Array = compiledInference.into(new Float32Array(3), linearInput);
const nnNativeInference: CompiledInference<readonly [2], readonly [3]> = nn.native(linear, linearTypedCompileOptions);
const nnInferenceAlias: CompiledInference<readonly [2], readonly [3]> = nn.inference(linear, linearTypedCompileOptions);
const nnForInferenceAlias: CompiledInference<readonly [2], readonly [3]> = nn.forInference(linear, linearTypedCompileOptions);
const nnForInferenceSnakeAlias: CompiledInference<readonly [2], readonly [3]> = nn.for_inference(linear, linearTypedCompileOptions);
const nnCompileInferenceAlias: CompiledInference<readonly [2], readonly [3]> = nn.compileInference(linear, linearTypedCompileOptions);
const moduleNativeInference: CompiledInference<readonly [2], TensorShapeTuple> = linear.native(linearTypedCompileOptions);
const moduleInferenceAlias: CompiledInference<readonly [2], TensorShapeTuple> = linear.inference(linearTypedCompileOptions);
const moduleForInferenceAlias: CompiledInference<readonly [2], TensorShapeTuple> = linear.forInference(linearTypedCompileOptions);
const moduleForInferenceSnakeAlias: CompiledInference<readonly [2], TensorShapeTuple> = linear.for_inference(linearTypedCompileOptions);
const moduleCompileInferenceAlias: CompiledInference<readonly [2], TensorShapeTuple> = linear.compileInference(linearTypedCompileOptions);
const nativeEagerLinearInto: Float32Array = zgml.nativeEager.linearInto(new Float32Array(3), linearInput, tensor([1, 0, 0, 1, 1, 1], [2, 3] as const), { bias: tensor([0, 0, 0], [3] as const) });
const nativeEagerLinearIntoAlias: Float32Array = zgml.native_eager.linear_into(new Float32Array(3), linearInput, tensor([1, 0, 0, 1, 1, 1], [2, 3] as const), { bias: tensor([0, 0, 0], [3] as const) });
const nativeEagerRoutingPolicy: NativeEagerRoutingPolicy = zgml.nativeEager.routingPolicy;
const nativeEagerRoutingPolicyAlias: NativeEagerRoutingPolicy = zgml.native_eager.routing_policy;
const nativeEagerRoutingCore: "zig-c-abi" = nativeEagerRoutingPolicy.nativeCore;
const nativeEagerRoutingMatmul: "native" = nativeEagerRoutingPolicy.tensorMath.matmul;
const nativeEagerMatmulOptions: NativeEagerMatmulIntoOptions = { rows: 1, shared: 2, cols: 3 };
const nativeEagerMatmulInto: Float32Array = zgml.nativeEager.matmulInto(new Float32Array(3), linearInput, tensor([1, 0, 0, 1, 1, 1], [2, 3] as const), nativeEagerMatmulOptions);
const nativeEagerMatmulIntoAlias: Float32Array = zgml.native_eager.matmul_into(new Float32Array(3), linearInput, tensor([1, 0, 0, 1, 1, 1], [2, 3] as const), nativeEagerMatmulOptions);
const nativeEagerElementwiseInto: Float32Array = zgml.nativeEager.elementwiseInto(new Float32Array(2), linearInput, tensor([1, 1], [2] as const), { op: "add" });
const nativeEagerElementwiseIntoAlias: Float32Array = zgml.native_eager.elementwise_into(new Float32Array(2), linearInput, null, { op: "sqr" });
const nativeEagerReduceInto: Float32Array = zgml.nativeEager.reduceInto(new Float32Array(1), linearInput, { op: "sum" });
const nativeEagerReduceIntoAlias: Float32Array = zgml.native_eager.reduce_into(new Float32Array(1), linearInput, { op: "max" });
const nativeEagerReduceDimOptions: NativeEagerReduceDimIntoOptions = { op: "sum", outer: 2, reduce: 3, inner: 1 };
const nativeEagerReduceDimInto: Float32Array = zgml.nativeEager.reduceDimInto(new Float32Array(2), tensor([1, 2, 3, 4, 5, 6], [2, 3] as const), nativeEagerReduceDimOptions);
const nativeEagerReduceDimIntoAlias: Float32Array = zgml.native_eager.reduce_dim_into(new Float32Array(2), tensor([1, 2, 3, 4, 5, 6], [2, 3] as const), { op: "max", outer: 2, reduce: 3, inner: 1 });
const nativeEagerArgReduceDimOptions: NativeEagerArgReduceDimIntoOptions = { op: "argmax", outer: 2, reduce: 3, inner: 1 };
const nativeEagerArgReduceDimInto: Float32Array = zgml.nativeEager.argReduceDimInto(new Float32Array(2), tensor([1, 3, 2, 6, 4, 5], [2, 3] as const), nativeEagerArgReduceDimOptions);
const nativeEagerArgReduceDimIntoAlias: Float32Array = zgml.native_eager.arg_reduce_dim_into(new Float32Array(2), tensor([1, 3, 2, 6, 4, 5], [2, 3] as const), { op: "argmin", outer: 2, reduce: 3, inner: 1 });
const nativeEagerCumsumOptions: NativeEagerCumsumIntoOptions = { outer: 2, axis: 3, inner: 1 };
const nativeEagerCumsumInto: Float32Array = zgml.nativeEager.cumsumInto(new Float32Array(6), tensor([1, 2, 3, 4, 5, 6], [2, 3] as const), nativeEagerCumsumOptions);
const nativeEagerCumsumIntoAlias: Float32Array = zgml.native_eager.cumsum_into(new Float32Array(6), tensor([1, 2, 3, 4, 5, 6], [2, 3] as const), { outer: 2, axis: 3, inner: 1, reverse: true });
const nativeEagerConv2dOptions: NativeEagerConv2dIntoOptions = { outH: 2, outW: 2 };
const nativeEagerConv2dInto: Float32Array = zgml.nativeEager.conv2dInto(
  new Float32Array(4),
  tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3] as const),
  tensor([1, 0, 0, 1], [1, 1, 2, 2] as const),
  nativeEagerConv2dOptions,
);
const nativeEagerConv2dIntoAlias: Float32Array = zgml.native_eager.conv2d_into(
  new Float32Array(4),
  tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3] as const),
  tensor([1, 0, 0, 1], [1, 1, 2, 2] as const),
  { out_h: 2, out_w: 2 },
);
const nativeEagerPool2dOptions: NativeEagerPool2dIntoOptions = { op: "max", kernelH: 2, kernelW: 2, strideH: 1, strideW: 1, outH: 2, outW: 2 };
const nativeEagerPool2dInto: Float32Array = zgml.nativeEager.pool2dInto(
  new Float32Array(4),
  tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3] as const),
  nativeEagerPool2dOptions,
);
const nativeEagerPool2dIntoAlias: Float32Array = zgml.native_eager.pool2d_into(
  new Float32Array(4),
  tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3] as const),
  { op: "avg", kernel_h: 2, kernel_w: 2, stride_h: 1, stride_w: 1, out_h: 2, out_w: 2 },
);
const nativeEagerSoftmaxInto: Float32Array = zgml.nativeEager.softmaxInto(new Float32Array(3), linearInput, { dim: -1 });
const nativeEagerLogSoftmaxIntoAlias: Float32Array = zgml.native_eager.log_softmax_into(new Float32Array(3), linearInput, { dim: -1 });
const compiledInferencePrepared: () => Float32Array = compiledInference.prepareInto(new Float32Array(3), linearInput);
const compiledInferenceExplanation: ModuleCompileExplanation<readonly [2], readonly [3]> | ModuleCompileSupport<readonly [2], readonly [3]> = compiledInference.explain();
const compiledInferencePreflight: ModuleCompileExplanation<readonly [2], readonly [3]> | ModuleCompileSupport<readonly [2], readonly [3]> = compiledInference.preflight();
const compiledInferenceSupport: ModuleCompileSupport<readonly [2], readonly [3]> = compiledInference.compileSupport();
const compiledInferenceInputShape: readonly [2] = compiledInference.inputShape();
const compiledInferenceOutputShape: readonly [3] = compiledInference.outputShape();
const compiledInferenceKernelPlan: ModuleKernelPlan | null = compiledInference.kernelPlan();
const compiledInferenceCompilerSignatures: ModuleCompleteCompilerSignatures | null = compiledInference.compilerSignatures();
void compiledInferenceNative;
void compiledInferenceProgram;
void compiledInferenceSession;
void compiledInferenceExecutionPlan;
void compiledInferenceRequiredExecutionPlan;
void rootRunOutput;
void rootInferOutput;
void rootPredictOutput;
void rootRunIntoOutput;
void rootRunIntoSnakeOutput;
void rootInferIntoOutput;
void rootInferIntoSnakeOutput;
void rootPredictIntoOutput;
void rootPredictIntoSnakeOutput;
void zgmlInferOutput;
void zgmlPredictOutput;
void zgmlInferIntoOutput;
void zgmlPredictIntoOutput;
void compileInferOutput;
void compilePredictOutput;
void compileInferIntoOutput;
void compileInferIntoSnakeOutput;
void compilePredictIntoOutput;
void compilePredictIntoSnakeOutput;
compiledInference.dispose();
compiledInferenceAlias.free();
compiledInferenceNativeAlias.dispose();
rootNativeInference.dispose();
rootInferenceAlias.dispose();
zgmlNativeInference.dispose();
zgmlNativeLayerListInference.dispose();
zgmlInferenceAlias.dispose();
nnNativeInference.dispose();
nnInferenceAlias.dispose();
nnCompileInferenceAlias.dispose();
moduleNativeInference.dispose();
moduleInferenceAlias.dispose();
moduleCompileInferenceAlias.dispose();
zgmlCompiledInference.dispose();
zgmlCompileInferenceAlias.dispose();
zgmlCompiledInferenceAlias.free();
const placementKind: ProgramDeviceBufferKind = "input";
const linearBufferPlacedSlot: ProgramBufferLayoutSlot | null = linearProgram.bufferSlot(placementKind);
const hostNativePlacement: TensorNativePlacement = linearInput.nativePlacement();
const hostNativePlacementKind: "zgml.tensor.native-placement" = hostNativePlacement.kind;
const programNativePlacement: TensorNativePlacement = linearInput.nativePlacement({ program: linearProgram, kind: "input" });
const programNativePlacementBufferKind: ProgramDeviceBufferKind | null = programNativePlacement.bufferKind;
const programNativePlacementEvidenceSignature: string | null = programNativePlacement.programCompileEvidenceSignature;
const programNativePlacementCompiler: "zig-module-program" | null = programNativePlacement.nativeCompilerAuthority;
const programNativePlacementInspectionSignature: string | null = programNativePlacement.nativeProgramInspectionSignature;
const programNativePlacementInspectionSource: "zig-program-inspection" | null = programNativePlacement.nativeProgramInspectionSource;
const snakeNativePlacement: TensorNativePlacement = linearInput.native_placement({ program: linearProgram, kind: "input" });
const placedInput: NativeBuffer = linearInput.place(linearProgram, placementKind);
const nativeInput: NativeBuffer = linearInput.toNativeBuffer({ program: linearProgram, kind: "input" });
const fromNative = Tensor.fromNativeBuffer(nativeInput, [2] as const);
const fromNativeShape: readonly [2] = fromNative.shape;
const copyFromNative: Tensor<readonly [2]> = linearInput.copyFromNativeBuffer_(nativeInput);
const copyFromNativeSnake: Tensor<readonly [2]> = linearInput.copy_from_native_buffer_(nativeInput);
const explicitOutputTensor = linearSession.stepTensor(linearInput, { shape: [3] as const });
const explicitOutputShape: readonly [3] = explicitOutputTensor.shape;
const typedLinearSessionOutputTensor: Tensor<readonly [3]> = typedLinearSession.stepTensor(linearInput);
const typedLinearSessionExecuteTensor: Tensor<readonly [3]> = typedLinearSession.executeTensor({ input: linearInput });
const typedLinearSessionReadOutputTensor: Tensor<readonly [3]> = typedLinearSession.readOutputTensor();
type TypedLinearSessionStepParamsInput = Expect<Equal<NonNullable<SessionStepParams<readonly [2], readonly [3]>["input"]>, ProgramInputBinding<readonly [2]>>>;
type TypedLinearSessionStepParamsOutput = Expect<Equal<NonNullable<SessionStepParams<readonly [2], readonly [3]>["output"]>, ProgramOutputBinding<readonly [3]> | false>>;
type TypedLinearSessionExecuteParamsInput = Expect<Equal<NonNullable<SessionExecuteParams<readonly [2], readonly [3]>["input"]>, ProgramInputBinding<readonly [2]>>>;
type TypedLinearSessionExecuteTensorParamsOutput = Expect<NonNullable<SessionExecuteTensorParams<readonly [2], readonly [3]>["output"]> extends Tensor<readonly [3]> | Float32Array ? true : false>;
type TypedLinearSessionExecuteIntoParamsInput = Expect<Equal<NonNullable<SessionExecuteIntoParams<readonly [2]>["input"]>, ProgramInputBinding<readonly [2]>>>;
const typedLinearStepParams: SessionStepParams<readonly [2], readonly [3]> = {
  input: linearInput,
  output: typedProgramOutputTensor,
};
// @ts-expect-error typed SessionStepParams must reject the wrong shaped input Tensor.
const typedLinearStepParamsBadInput: SessionStepParams<readonly [3], readonly [3]> = { input: linearInput };
// @ts-expect-error typed SessionStepParams must reject the wrong shaped output Tensor.
const typedLinearStepParamsBadOutput: SessionStepParams<readonly [2], readonly [2]> = { input: linearInput, output: typedProgramOutputTensor };
const numericOutputTensor = linearSession.stepTensor(linearInput, { shape: 3 });
const numericOutputShape: readonly [3] = numericOutputTensor.shape;
const outputCarrier = zeros([3] as const);
const carrierOutputTensor = linearSession.stepTensor(linearInput, { output: outputCarrier });
const carrierOutputShape: readonly [3] = carrierOutputTensor.shape;
const linearStepInto: Float32Array = linearSession.stepInto(new Float32Array(3), linearInput);
const linearExecuteParams: SessionExecuteParams = { input: linearInput, output: new Float32Array(3) };
const linearExecuteIntoParams: SessionExecuteIntoParams = { input: linearInput };
const linearStepParams: SessionStepParams = { input: linearInput, output: false };
const linearExecuted: Float32Array | undefined = linearSession.execute(linearExecuteParams);
const linearNoOutput: undefined = linearSession.execute({ input: linearInput, output: false });
const linearExecuteInto: Float32Array = linearSession.executeInto(new Float32Array(3), linearExecuteIntoParams);
// @ts-expect-error Session executeInto params take the output buffer as the first argument.
const badLinearExecuteIntoParams: SessionExecuteIntoParams = { input: linearInput, output: new Float32Array(3) };
// @ts-expect-error executeInto takes the output buffer as its first argument.
linearSession.executeInto(new Float32Array(3), { input: linearInput, output: new Float32Array(3) });
const linearExecuteTensor = linearSession.executeTensor({ input: linearInput, output: new Float32Array(3), shape: [3] as const });
const linearExecuteTensorShape: readonly [3] = linearExecuteTensor.shape;
const moduleBoundTensor = linearModuleSession.stepTensor(linearInput, { shape: [3] as const });
const moduleBoundShape: readonly [3] = moduleBoundTensor.shape;
const readOutputTensor = linearSession.readOutputTensor({ shape: [3] as const });
const readOutputShape: readonly [3] = readOutputTensor.shape;
const readOutputCarrier = zeros([3] as const);
const carrierReadOutputTensor = linearSession.readOutputTensor({ output: readOutputCarrier });
const carrierReadOutputShape: readonly [3] = carrierReadOutputTensor.shape;
const readOutputArrayCarrier = new Float32Array(3);
const arrayCarrierReadOutputTensor = linearSession.readOutputTensor({ output: readOutputArrayCarrier, shape: [3] as const });
const arrayCarrierReadOutputShape: readonly [3] = arrayCarrierReadOutputTensor.shape;
const readOutputLargeArrayCarrier = new Float32Array(8);
const largeArrayCarrierReadOutputTensor = linearSession.readOutputTensor({
  output: readOutputLargeArrayCarrier,
  shape: [3] as const,
  length: 3,
});
const largeArrayCarrierReadOutputShape: readonly [3] = largeArrayCarrierReadOutputTensor.shape;
// @ts-expect-error Program inspection snapshots are immutable evidence records.
linearInspection.backend = "cpu";
// @ts-expect-error Program requirements snapshots are immutable evidence records.
linearRequirements.inputLen = 4;
// @ts-expect-error Program buffer sizing snapshots are immutable evidence records.
linearProgramBufferSizing.inputLen = 4;
// @ts-expect-error Session buffer sizing snapshots are immutable evidence records.
linearSessionBufferSizing.outputLen = 4;
// @ts-expect-error Program capability snapshots are immutable evidence records.
linearCapabilities.canExecute = false;
// @ts-expect-error Runtime profile snapshots are immutable evidence records.
linearRuntimeProfile.callCount = 1;
// @ts-expect-error Session call profile snapshots are immutable evidence records.
linearSessionCallProfile.stepCount = 1;
// @ts-expect-error Session call profile reset counter is immutable evidence too.
linearSessionCallProfile.resetCount = 1;
// @ts-expect-error Session call profile upload counters are immutable evidence too.
linearSessionCallProfile.uploadParametersCount = 1;
// @ts-expect-error Session call profile single-parameter upload counter is immutable evidence too.
linearSessionCallProfile.uploadParameterCount = 1;
// @ts-expect-error Session call profile named-parameter upload counter is immutable evidence too.
linearSessionCallProfile.uploadParameterByNameCount = 1;
linearSession.uploadParameter(0);
linearSession.uploadParameterByName("0.weight");
linearSession.uploadParameterRange(0, 1);
// @ts-expect-error StepParams hot-path blocker arrays are immutable evidence records.
linearSessionStepParamsHotPathBlockers[0] = "readback";
// @ts-expect-error StepParams input shapes are immutable evidence records.
linearSessionStepParamsInputShape[0] = 4;
// @ts-expect-error Program buffer-layout slot arrays are immutable evidence records.
linearBufferLayout.slots[0] = linearBufferLayout.output;
// @ts-expect-error Program buffer-layout slots are immutable evidence records.
linearBufferLayout.input.byteLength = 0;
// @ts-expect-error Module compatibility snapshots are immutable evidence records.
linearCompatibility.compatible = false;
// @ts-expect-error Session inspection snapshots are immutable evidence records.
linearSessionInspection.position = 1;
// @ts-expect-error Session runtime profile snapshots are immutable evidence records.
linearSessionRuntimeProfile.syncCount = 1;
// @ts-expect-error Session buffer layouts share immutable Program buffer-layout evidence.
linearSessionBufferLayout.output.elementCount = 1;
placedInput.free();
nativeInput.free();
linearModuleSession.free();
spreadLinearSession.free();
linearSession.free();
linearProgram.free();

const embedding = nn.embedding(8, 4);
const embeddingOptions: EmbeddingCompileOptions<readonly [3]> = { backend: "cpu", inputShape: [3] as const };
const embeddingProgram: Program = embedding.compile(embeddingOptions);
const typedEmbeddingProgram: Program<readonly [3], readonly [3, 4]> = embedding.compile(embeddingOptions);
const typedEmbeddingNamespaceProgram: Program<readonly [3], readonly [3, 4]> = nn.compile(embedding, embeddingOptions);
const typedEmbeddingCompileNamespaceProgram: Program<readonly [3], readonly [3, 4]> = compile.compile(embedding, embeddingOptions);
const typedEmbeddingOutputShape: readonly [3, 4] = typedEmbeddingProgram.outputShape();
const typedEmbeddingNamespaceOutputShape: readonly [3, 4] | null = nn.outputShape(embedding, embeddingOptions);
const typedEmbeddingMethodOutputShape: readonly [3, 4] | null = embedding.outputShape(embeddingOptions);
const typedEmbeddingMethodOutputShapeAlias: readonly [3, 4] | null = embedding.output_shape(embeddingOptions);
const typedEmbeddingSupport: ModuleCompileSupport<readonly [3], readonly [3, 4]> = embedding.compileSupport(embeddingOptions);
const typedEmbeddingCompilePlan: ModuleCompileExplanation<readonly [3], readonly [3, 4]> = embedding.compilePlan(embeddingOptions);
const embeddingGridOptions: EmbeddingCompileOptions<readonly [2, 3]> = { backend: "cpu", inputShape: [2, 3] as const };
const typedEmbeddingGridProgram: Program<readonly [2, 3], readonly [2, 3, 4]> = embedding.compile(embeddingGridOptions);
const typedEmbeddingGridNamespaceProgram: Program<readonly [2, 3], readonly [2, 3, 4]> = nn.compile(embedding, embeddingGridOptions);
const typedEmbeddingGridOutputShape: readonly [2, 3, 4] = typedEmbeddingGridProgram.outputShape();
const typedEmbeddingGridMethodOutputShape: readonly [2, 3, 4] | null = embedding.outputShape(embeddingGridOptions);
const typedEmbeddingGridSupport: ModuleCompileSupport<readonly [2, 3], readonly [2, 3, 4]> = embedding.compileSupport(embeddingGridOptions);
const typedEmbeddingGridCompilePlan: ModuleCompileExplanation<readonly [2, 3], readonly [2, 3, 4]> = embedding.compilePlan(embeddingGridOptions);
const embeddingBindings: ModuleBindings = nn.bindParameters(embedding, embeddingOptions);
const embeddingBindingsAlias: ModuleBindings = nn.bind_parameters(embedding, embeddingOptions);
const embeddingPlacedBindings: ModuleBindings = embedding.placeParameters(embeddingProgram);
const embeddingPlacedBindingsAlias: ModuleBindings = embedding.place_parameters(embeddingProgram);
const embeddingRootPlacedBindings: ModuleBindings = nn.placeParameters(embedding, embeddingProgram);
const embeddingRootPlacedBindingsAlias: ModuleBindings = nn.place_parameters(embedding, embeddingProgram);
const embeddingPlacedBindingPlan: ModuleBindingPlan = nn.bindingPlan(embeddingPlacedBindings);
const embeddingRootPlacedBindingPlan: ModuleBindingPlan = nn.bindingPlan(embeddingRootPlacedBindings);
const embeddingPlacedBindingPlanSignature: string = embeddingPlacedBindingPlan.signature;
const embeddingPlacedBindingPlanPlacementMode: ModuleBindingPlacementMode = embeddingPlacedBindingPlan.placementMode;
const embeddingCompatibility: ProgramModuleCompatibility = embeddingProgram.moduleCompatibility(embedding);
const embeddingCompatibilitySignature: string = embeddingCompatibility.signature;
const embeddingAccepted: boolean = embeddingProgram.acceptsModule(embedding);
const embeddingSession: Session = embeddingProgram.bind(embeddingBindings);
const embeddingTensor = new Tensor([1, 2, 3], [3] as const);
const embeddingGridTensor = new Tensor([1, 2, 3, 1, 2, 3], [2, 3] as const);
type EmbeddingTensorForwardShape = Expect<Equal<EmbeddingForwardShape<readonly [3], 4>, readonly [3, 4]>>;
type EmbeddingGridForwardShape = Expect<Equal<EmbeddingForwardShape<readonly [2, 3], 4>, readonly [2, 3, 4]>>;
type OneHotTensorShape = Expect<Equal<OneHotShape<readonly [2], 3>, readonly [2, 3]>>;
type OneHotGridShape = Expect<Equal<OneHotShape<readonly [2, 2], 3>, readonly [2, 2, 3]>>;
type EmbeddingModuleForwardShape = Expect<Equal<ModuleForwardShape<typeof embedding, readonly [3]>, readonly [3, 4]>>;
type EmbeddingGridModuleForwardShape = Expect<Equal<ModuleForwardShape<typeof embedding, readonly [2, 3]>, readonly [2, 3, 4]>>;
const embeddingForwardTensor: Tensor<readonly [3, 4]> = embedding.forward(embeddingTensor);
const embeddingGridForwardTensor: Tensor<readonly [2, 3, 4]> = embedding.forward(embeddingGridTensor);
embeddingSession.step(embeddingTensor);
embeddingSession.free();
const embeddingModuleSession: Session = embeddingProgram.bindModule(embedding);
embeddingModuleSession.free();
embeddingProgram.free();

const sequential = nn.sequential([nn.linear(2, 3), nn.relu(), nn.linear(3, 2)]);
const emptySequential = nn.sequential();
const emptyClassSequential = new nn.Sequential();
const namedSequential = nn.sequential({
  stem: nn.linear(2, 3),
  act: nn.relu(),
  head: nn.linear(3, 2),
});
const namedClassSequential = new nn.Sequential({
  stem: nn.linear(2, 3),
  head: nn.linear(3, 2),
});
type SequentialOutputShape = Expect<Equal<SequentialForwardShape<typeof sequential.layers, readonly [2]>, readonly [2]>>;
type EmptySequentialOutputShape = Expect<Equal<SequentialForwardShape<typeof emptySequential.layers, readonly [2]>, readonly [2]>>;
const sequentialForward = sequential.forward as unknown as (input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const sequentialForwardTensor: Tensor<readonly [2]> = sequentialForward(linearInput);
const emptySequentialForwardTensor: Tensor<readonly [2]> = emptySequential.forward(linearInput);
const emptyClassSequentialForwardTensor: Tensor<readonly [2]> = emptyClassSequential.__call__(linearInput);
const namedSequentialForward = namedSequential.forward as unknown as (input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const namedClassSequentialForward = namedClassSequential.forward as unknown as (input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const namedSequentialForwardTensor: Tensor<readonly [2]> = namedSequentialForward(linearInput);
const namedClassSequentialForwardTensor: Tensor<readonly [2]> = namedClassSequentialForward(linearInput);
const namedSequentialState: ModuleStateSnapshot = namedSequential.stateDict("named");
const sequentialNamespaceForward = nn.forward as unknown as (module: NnModule, input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const sequentialNamespaceForwardTensor: Tensor<readonly [2]> = sequentialNamespaceForward(sequential, linearInput);
const variadicSequential = nn.sequential(nn.linear(2, 3), nn.relu(), nn.linear(3, 2));
const variadicSequentialAsModule: NnModule = variadicSequential;
const variadicSequentialForward = variadicSequential.forward as unknown as (input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const variadicSequentialForwardTensor: Tensor<readonly [2]> = variadicSequentialForward(linearInput);
const variadicClassSequential = new nn.Sequential(nn.linear(2, 3), nn.relu(), nn.linear(3, 2));
const variadicClassSequentialAsModule: NnModule = variadicClassSequential;
const variadicClassSequentialForward = variadicClassSequential.forward as unknown as (input: Tensor<readonly [2]>) => Tensor<readonly [2]>;
const variadicClassSequentialForwardTensor: Tensor<readonly [2]> = variadicClassSequentialForward(linearInput);
const variadicClassSequentialLen: number = variadicClassSequential.__len__();
const variadicClassSequentialSize: number = variadicClassSequential.size();
const variadicClassSequentialGet: NnModule = variadicClassSequential.get(0);
const variadicClassSequentialDunderItem: NnModule = variadicClassSequential.__getitem__(1);
const variadicClassSequentialSetItem: typeof variadicClassSequential = variadicClassSequential.__setitem__(1, nn.tanh());
const variadicClassSequentialDelItem: SequentialModule = new nn.Sequential(nn.relu(), nn.tanh()).__delitem__(-1);
const variadicClassSequentialPoppedLast: NnModule | undefined = new nn.Sequential(nn.relu(), nn.tanh()).pop();
const variadicClassSequentialPoppedFirst: NnModule | undefined = new nn.Sequential(nn.relu(), nn.tanh()).pop(0);
const variadicClassSequentialClear: SequentialModule = new nn.Sequential(nn.relu(), nn.tanh()).clear();
const variadicClassSequentialAppend: typeof variadicClassSequential = variadicClassSequential.append(nn.relu());
const variadicClassSequentialInsert: typeof variadicClassSequential = variadicClassSequential.insert(1, nn.tanh());
const variadicClassSequentialExtend: typeof variadicClassSequential = variadicClassSequential.extend([nn.relu()]);
const variadicClassSequentialExtendSequential: typeof variadicClassSequential = variadicClassSequential.extend(new nn.Sequential(nn.relu()));
const singleLayerSequential = nn.sequential(nn.relu());
const singleLayerSequentialAsModule: NnModule = singleLayerSequential;
const singleLayerSequentialForwardTensor: Tensor<readonly [2, 3]> = singleLayerSequential.forward(literalTensor);
const sequentialSupport: ModuleCompileSupport = sequential.compileSupport({ backend: "cpu" });
const sequentialExplanation: ModuleCompileExplanation = nn.explain(sequential, { backend: "cpu" });
const sequentialPreflight: ModuleCompileExplanation = nn.preflight(sequential, { backend: "cpu" });
const sequentialMethodExplanation: ModuleCompileExplanation = sequential.explain({ backend: "cpu" });
const sequentialMethodPreflight: ModuleCompileExplanation = sequential.preflight({ backend: "cpu" });
const sequentialCompileExplanation: ModuleCompileExplanation = sequential.compileExplanation({ backend: "cpu" });
const sequentialNamespaceCompileExplanation: ModuleCompileExplanation = nn.compileExplanation(sequential, { backend: "cpu" });
const sequentialCompilePlan: ModuleCompileExplanation = nn.compilePlan(sequential, { backend: "cpu" });
const sequentialMethodCompilePlan: ModuleCompileExplanation = sequential.compilePlan({ backend: "cpu" });
const sequentialTrace = nn.trace(sequential, { inputShape: [2] as const });
const sequentialCompilerSignatures: ModuleCompilerSignatures | null = nn.compilerSignatures(sequential, { backend: "cpu" });
const sequentialIr: ModuleTensorProgramIr | null = nn.tensorProgramIr(sequential, { backend: "cpu" });
const sequentialIrSignature: string | undefined = sequentialIr?.signature;
const sequentialKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(sequential, { backend: "cpu" });
const sequentialKernelPlanSignature: string | undefined = sequentialKernelPlan?.signature;
const sequentialMemoryLayout: ModuleKernelMemoryLayout | null = nn.memoryLayout(sequential, { backend: "cpu" });
const sequentialMemoryLayoutSignature: string | undefined = sequentialMemoryLayout?.signature;
const sequentialBufferLayout: ModuleKernelBufferLayout | null = nn.bufferLayout(sequential, { backend: "cpu" });
const sequentialBufferLayoutSignature: string | undefined = sequentialBufferLayout?.signature;
const sequentialInputShape: readonly number[] | null = nn.inputShape(sequential, { backend: "cpu" });
const sequentialOutputShapeFromHelper: readonly number[] | null = nn.outputShape(sequential, { backend: "cpu" });
const sequentialNamespaceTypedOutputShape: readonly [2] | null = nn.outputShape(sequential, { backend: "cpu", inputShape: [2] as const });
const sequentialMethodTypedOutputShape: readonly [2] | null = sequential.outputShape({ backend: "cpu", inputShape: [2] as const });
const sequentialMethodTypedOutputShapeAlias: readonly [2] | null = sequential.output_shape({ backend: "cpu", inputShape: [2] as const });
const sequentialCompileNamespaceTypedOutputShape: readonly [2] | null = compile.outputShape(sequential, { backend: "cpu", inputShape: [2] as const });
const sequentialLayerList = [nn.linear(2, 3), nn.relu(), nn.linear(3, 2)] as const;
type SequentialLayerListOutputShape = Expect<Equal<ModuleTargetForwardShape<typeof sequentialLayerList, readonly [2]>, readonly [2]>>;
const sequentialArrayNnNamespaceParameters: NnParameter[] = nn.parameters(sequentialLayerList);
const sequentialArrayNnNamespaceNamedParameters: NnParameter[] = nn.namedParameters(sequentialLayerList, "model");
const sequentialArrayNnNamespaceParameterNames: readonly string[] = nn.parameterNames(sequentialLayerList, "model");
const sequentialArrayNnNamespaceParameterInfos: readonly ModuleParameterInfo[] = nn.parameterInfos(sequentialLayerList, "model");
const sequentialArrayNnNamespaceParameterInfo: ModuleParameterInfo | null = nn.parameterInfo(sequentialLayerList, "model.0.weight", "model");
const sequentialArrayNnNamespaceState: ModuleStateSnapshot = nn.stateDict(sequentialLayerList, "model");
const sequentialArrayNnNamespaceStateAlias: ModuleStateSnapshot = nn.state_dict(sequentialLayerList, "model");
const sequentialArrayNnNamespaceRequiresGrad: SequentialModule<typeof sequentialLayerList> = nn.requiresGrad(sequentialLayerList, true);
const sequentialArrayNnNamespaceRequiresGradAlias: SequentialModule<typeof sequentialLayerList> = nn.requiresGrad_(sequentialLayerList, true);
const sequentialArrayNnNamespaceRequiresGradSnakeAlias: SequentialModule<typeof sequentialLayerList> = nn.requires_grad_(sequentialLayerList, false);
const sequentialArrayNnNamespaceFreeze: SequentialModule<typeof sequentialLayerList> = nn.freeze(sequentialLayerList);
const sequentialArrayNnNamespaceUnfreeze: SequentialModule<typeof sequentialLayerList> = nn.unfreeze(sequentialLayerList);
const sequentialArrayNnNamespaceLoadState: SequentialModule<typeof sequentialLayerList> = nn.loadStateDict(sequentialLayerList, sequentialArrayNnNamespaceState, { strict: true, prefix: "model", validateOnly: true });
const sequentialArrayNnNamespaceLoadStateAlias: SequentialModule<typeof sequentialLayerList> = nn.load_state_dict(sequentialLayerList, sequentialArrayNnNamespaceStateAlias, { strict: true, prefix: "model", validateOnly: true });
nn.zeroGrad(sequentialLayerList);
nn.zero_grad(sequentialLayerList);
const sequentialArrayNnNamespaceForwardTensor: Tensor<readonly [2]> = nn.forward(sequentialLayerList, linearInput);
const sequentialArrayNnNamespaceChildren: readonly NnModule[] = nn.children(sequentialLayerList);
const sequentialArrayNnNamespaceModules: readonly NnModule[] = nn.modules(sequentialLayerList);
const sequentialArrayNnNamespaceNamedChildren: readonly ModuleTraversalEntry[] = nn.namedChildren(sequentialLayerList);
const sequentialArrayNnNamespaceNamedModules: readonly ModuleTraversalEntry[] = nn.namedModules(sequentialLayerList);
const sequentialArrayNnNamespaceGetSubmodule: NnModule | null = nn.getSubmodule(sequentialLayerList, "0");
const sequentialArrayNnNamespaceGetSubmoduleAlias: NnModule | null = nn.get_submodule(sequentialLayerList, "1");
const sequentialArrayNnNamespaceGetParameter: NnParameter | null = nn.getParameter(sequentialLayerList, "0.weight");
const sequentialArrayNnNamespaceGetParameterAlias: NnParameter | null = nn.get_parameter(sequentialLayerList, "2.bias");
const sequentialArrayNnNamespaceGetBuffer: NnBuffer | null = nn.getBuffer(sequentialLayerList, "missing");
const sequentialArrayNnNamespaceGetBufferAlias: NnBuffer | null = nn.get_buffer(sequentialLayerList, "missing");
const sequentialArrayNnNamespaceApply: SequentialModule<typeof sequentialLayerList> = nn.apply(sequentialLayerList, (_module, entry) => {
  const appliedEntryName: string = entry.name;
});
const sequentialArrayNnNamespaceTrainMode: SequentialModule<typeof sequentialLayerList> = nn.train(sequentialLayerList);
const sequentialArrayNnNamespaceEvalMode: SequentialModule<typeof sequentialLayerList> = nn.eval(sequentialLayerList);
const sequentialArrayCompileNamespaceTrace: ModuleProgramTrace = compile.trace(sequentialLayerList, { backend: "cpu", inputShape: [2] as const });
const sequentialArrayCompileNamespaceSupport: ModuleCompileSupport<readonly [2], readonly [2]> = compile.compileSupport(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceCanCompile: boolean = compile.canCompile(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceRequiredSupport: ModuleCompileSupport<readonly [2], readonly [2]> = compile.requireCompileSupport(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceExplanation: ModuleCompileExplanation<readonly [2], readonly [2]> | ModuleCompileSupport<readonly [2], readonly [2]> = compile.explain(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespacePlan: ModuleCompileExplanation<readonly [2], readonly [2]> | ModuleCompileSupport<readonly [2], readonly [2]> = compile.requireCompilePlan(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceTypedOutputShape: readonly [2] | null = compile.outputShape(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceCompilerSignatures: ModuleCompilerSignatures | null = compile.compilerSignatures(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceTensorProgramIr: ModuleTensorProgramIr | null = compile.tensorProgramIr(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceKernelPlan: ModuleKernelPlan | null = compile.kernelPlan(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceBufferLayout: ModuleKernelBufferLayout | null = compile.bufferLayout(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceMemoryLayout: ModuleKernelMemoryLayout | null = compile.memoryLayout(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceShapeConstraints: ModuleKernelShapeConstraints | null = compile.shapeConstraints(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceParameterLayout: ModuleKernelParameterLayout | null = compile.parameterLayout(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceTrace: ModuleProgramTrace = nn.trace(sequentialLayerList, { inputShape: [2] as const });
const sequentialArrayNnNamespaceSupport: ModuleCompileSupport<readonly [2], readonly [2]> = nn.compileSupport(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespacePlan: ModuleCompileExplanation<readonly [2], readonly [2]> = nn.requireCompilePlan(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceProgram: Program<readonly [2], readonly [2]> = nn.compile(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceBindings: ModuleBindings<readonly [2], readonly [2]> = nn.bindParameters(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceCompilerSignatures: ModuleCompilerSignatures | null = nn.compilerSignatures(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceTensorProgramIr: ModuleTensorProgramIr | null = nn.tensorProgramIr(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayNnNamespaceTypedOutputShape: readonly [2] | null = nn.outputShape(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialArrayCompileNamespaceProgramDiagnostic: RawSequentialLayerListCompileDiagnostic = compile.compile(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
// @ts-expect-error raw Sequential layer lists expose compile evidence, not Program creation.
const sequentialArrayCompileNamespaceProgram: Program<readonly [2], readonly [2]> = compile.compile(
  sequentialLayerList,
  { backend: "cpu", inputShape: [2] as const },
);
const sequentialShapeConstraints: ModuleKernelShapeConstraints | null = nn.shapeConstraints(sequential, { backend: "cpu" });
const sequentialParameterLayout: ModuleKernelParameterLayout | null = nn.parameterLayout(sequential, { backend: "cpu" });
const sequentialParameterLayoutSignature: string | undefined = sequentialParameterLayout?.signature;
const sequentialCanCompile: boolean = nn.canCompile(sequential, { backend: "cpu" });
const sequentialExplanationKind: "zgml.nn.compile-explanation" = sequentialExplanation.kind;
const sequentialMethodExplanationKind: "zgml.nn.compile-explanation" = sequentialMethodExplanation.kind;
const sequentialCompilePlanKind: "zgml.nn.compile-explanation" = sequentialCompilePlan.kind;
const sequentialMethodCompilePlanKind: "zgml.nn.compile-explanation" = sequentialMethodCompilePlan.kind;
const sequentialExplanationReason: string | null = sequentialExplanation.reason;
const sequentialExplanationNativePath: ModuleCompileSupport["nativePath"] | null = sequentialExplanation.nativePath;
const sequentialExplanationModelKind: ModuleCompileSupport["modelKind"] | null = sequentialExplanation.modelKind;
const sequentialExplanationSignature: string = sequentialExplanation.signature;
const sequentialCompilePlanSignature: string = sequentialCompilePlan.signature;
const sequentialExplanationInputShape: readonly number[] | null = sequentialExplanation.inputShape;
const sequentialExplanationParameterNames: readonly string[] = sequentialExplanation.parameterNames;
const sequentialExplanationParameterInfos: readonly ModuleParameterInfo[] = sequentialExplanation.parameterInfos;
const sequentialExplanationIr: ModuleTensorProgramIr | null = sequentialExplanation.ir;
const sequentialExplanationIrSignature: string | undefined = sequentialExplanationIr?.signature;
const sequentialExplanationTrace: ModuleProgramTrace | null = sequentialExplanation.trace;
const sequentialExplanationKernelPlan: ModuleKernelPlan | null = sequentialExplanation.kernelPlan;
const sequentialExplanationKernelPlanSignature: string | undefined = sequentialExplanationKernelPlan?.signature;
const sequentialCompileExplanationKernelPlan: ModuleKernelPlan | null = sequentialCompileExplanation.kernelPlan;
const sequentialExplanationSignatures: ModuleCompilerSignatures = sequentialExplanation.compilerSignatures;
const sequentialExplanationBufferLayout: ModuleKernelBufferLayout | null = sequentialExplanation.bufferLayout;
const sequentialExplanationMemoryLayout: ModuleKernelMemoryLayout | null = sequentialExplanation.memoryLayout;
const sequentialExplanationShapeConstraints: ModuleKernelShapeConstraints | null = sequentialExplanation.shapeConstraints;
const sequentialExplanationParameterLayout: ModuleKernelParameterLayout | null = sequentialExplanation.parameterLayout;
const sequentialExplanationOutputShape: readonly number[] | null = sequentialExplanation.outputShape;
const sequentialExplanationParameterCount: number | null = sequentialExplanation.parameterCount;
const sequentialExplanationParameterScalarCount: number | null = sequentialExplanation.parameterScalarCount;
const sequentialExplanationDiagnostics: readonly ModuleCompileDiagnostic[] = sequentialExplanation.diagnostics;
const sequentialFusedValueEdges: readonly ModuleKernelPlanFusedValueEdge[] | undefined = sequentialKernelPlan?.ops[0]?.fusedValueEdges;
const sequentialFusedInputValueIds: readonly number[] | undefined = sequentialFusedValueEdges?.[0]?.inputValueIds;
const sequentialProgram: Program = sequential.compile({ backend: "cpu" });
const typedSequentialInstanceProgram: Program<readonly [2], readonly [2]> = sequential.compile({ backend: "cpu", inputShape: [2] as const });
const typedSequentialSupport: ModuleCompileSupport<readonly [2], readonly [2]> = sequential.compileSupport({ backend: "cpu", inputShape: [2] as const });
const typedSequentialCompilePlan: ModuleCompileExplanation<readonly [2], readonly [2]> = sequential.compilePlan({ backend: "cpu", inputShape: [2] as const });
const typedBatchedSequentialProgram: Program<readonly [4, 2], readonly [4, 2]> = sequential.compile({ backend: "cpu", inputShape: [4, 2] as const });
const typedBatchedSequentialSupport: ModuleCompileSupport<readonly [4, 2], readonly [4, 2]> = sequential.compileSupport({ backend: "cpu", inputShape: [4, 2] as const });
const typedBatchedSequentialPlan: ModuleCompileExplanation<readonly [4, 2], readonly [4, 2]> = sequential.compilePlan({ backend: "cpu", inputShape: [4, 2] as const });
const sequentialEvidence: ProgramCompileEvidence | null = sequentialProgram.compileEvidence();
const sequentialNativeProgramInspection: ProgramInspection | undefined =
  sequentialEvidence?.kind === "module" ? sequentialEvidence.nativeProgramInspection : undefined;
const sequentialNativeProgramInspectionSource: "zig-program-inspection" | undefined =
  sequentialEvidence?.kind === "module" ? sequentialEvidence.nativeProgramInspectionSource : undefined;
const sequentialNativeCompilerAuthority: "zig-module-program" | undefined =
  sequentialEvidence?.kind === "module" ? sequentialEvidence.nativeCompilerAuthority : undefined;
const sequentialNativeInspectionExecutionSupported: boolean | undefined =
  sequentialEvidence?.kind === "module" ? sequentialEvidence.nativeExecutionSupported : undefined;
const sequentialNativeCommandStencilHash: bigint | undefined =
  sequentialEvidence?.kind === "module" ? sequentialEvidence.nativeCommandStencilHash : undefined;
const sequentialProgramIr: ModuleTensorProgramIr | null = sequentialProgram.tensorProgramIr();
const sequentialProgramIrSignature: string | undefined = sequentialProgramIr?.signature;
const sequentialProgramKernelPlan: ModuleKernelPlan | null = sequentialProgram.kernelPlan();
const sequentialProgramKernelPlanSignature: string | undefined = sequentialProgramKernelPlan?.signature;
const sequentialCompatibility: ProgramModuleCompatibility = sequentialProgram.moduleCompatibility(sequential);
const sequentialCompatibilitySignature: string = sequentialCompatibility.signature;
const sequentialAccepted: boolean = sequentialProgram.acceptsModule(sequential);
const sequentialSession: Session = sequentialProgram.bindModule(sequential);
const sequentialOutput = sequentialSession.stepTensor(linearInput, { shape: [2] as const });
const sequentialOutputShape: readonly [2] = sequentialOutput.shape;
// @ts-expect-error KernelPlan fused value-edge evidence is immutable.
sequentialFusedValueEdges![0].outputValueId = 0;
// @ts-expect-error KernelPlan fused input value IDs are immutable.
sequentialFusedInputValueIds![0] = 0;
// @ts-expect-error Compile explanations are immutable evidence records.
sequentialExplanation.supported = false;
// @ts-expect-error Compile explanation parameter-name arrays are immutable evidence records.
sequentialExplanationParameterNames[0] = "changed";
// @ts-expect-error Compile explanation parameter-info arrays are immutable evidence records.
sequentialExplanationParameterInfos[0] = linearModuleParameterInfos[0]!;
// @ts-expect-error Compile explanation diagnostic arrays are immutable evidence records.
sequentialExplanationDiagnostics[0] = { stage: "support", message: "changed" };
// @ts-expect-error Compile explanation parameter names are immutable evidence records.
sequentialExplanationParameterNames[0] = "changed";
// @ts-expect-error Compile explanation parameter info records are immutable evidence records.
sequentialExplanationParameterInfos[0].name = "changed";
sequentialSession.free();
sequentialProgram.free();

const steppedTensor = linearInput.step();
const steppedTensorShape: readonly [2] = steppedTensor.shape;
const stepModule = nn.step();
const stepSupport: ModuleCompileSupport = stepModule.compileSupport({ backend: "cpu", inputShape: [2] as const });
const typedStepSupport: ModuleCompileSupport<readonly [2], readonly [2]> = stepModule.compileSupport({ backend: "cpu", inputShape: [2] as const });
const typedStepCompilePlan: ModuleCompileExplanation<readonly [2], readonly [2]> = stepModule.compilePlan({ backend: "cpu", inputShape: [2] as const });
const stepIr: ModuleTensorProgramIr | null = nn.tensorProgramIr(stepModule, { backend: "cpu", inputShape: [2] as const });
const stepKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(stepModule, { backend: "cpu", inputShape: [2] as const });
const stepDescriptorSignatures: readonly string[] | undefined = stepKernelPlan?.ops[0]?.nativeDescriptorSignatures;
const stepProgram: Program = stepModule.compile({ backend: "cpu", inputShape: [2] as const });
const stepSession: Session = stepProgram.bindModule(stepModule);
const stepOutput = stepSession.stepTensor(linearInput, { shape: [2] as const });
const stepOutputShape: readonly [2] = stepOutput.shape;
stepSession.free();
stepProgram.free();

const rootNativeApiContract: NativeApiContractNamespace = nativeApiContract;
const rootNativeApiContractOwner: "src/ts/runtime/native_api_contract.ts" = rootNativeApiContract.nativeApiContractManifest.policyOwner;
const rootNativeApiContractProductOwner: "src/ts/** + src/**/*.zig" = rootNativeApiContract.nativeApiContractManifest.productSemanticsOwner;
const rootNativeApiContractNativeAlignment: "zig-core-contract-tested" = rootNativeApiContract.nativeApiContractManifest.nativeAlignment;
const rootNativeApiContractProductSourceOfTruth: "ts-api-zig-core" = rootNativeApiContract.nativeApiContractManifest.productSourceOfTruth;
const rootNativeApiContractNativeProductPolicy: "required-core" = rootNativeApiContract.nativeApiContractManifest.nativeProductPolicy;
const rootNativeApiContractNoMirrors: false = rootNativeApiContract.nativeApiContractManifest.handwrittenFrontendMirrors;
const rootNativeApiContractGeneratedSpine: "generated-ts" = rootNativeApiContract.nativePackageSpineContractManifest.source;
const rootNativeApiContractSpineOrigin: "src/ts/runtime/native_api_contract.ts" = rootNativeApiContract.nativePackageSpineContractManifest.generatedFrom;
const rootNativeApiContractSpineNoHandwrittenExports: false = rootNativeApiContract.nativePackageSpineContractManifest.handwrittenRootExportList;
const rootNativeApiContractParts: NativeApiContractSignatureParts = rootNativeApiContract.nativeApiContractSignatureParts();
const rootNativeApiContractSignature: string = rootNativeApiContract.formatNativeApiContractSignature(rootNativeApiContractParts);
const rootNativeApiContractSelfSignature: string = rootNativeApiContract.nativeApiContractSignature();
const rootNativeApiContractMissing: string[] = rootNativeApiContract.missingExports({ Tensor }, ["Tensor", "notExported"]);
const rootNativeApiTypedMissing: ("Tensor" | "notExported")[] = rootNativeApiContract.missingExports({ Tensor }, ["Tensor", "notExported"] as const);
const rootNativeApiTypedSortedKeys: ("Tensor" | "tensor")[] = rootNativeApiContract.sortedExportKeys({ Tensor, tensor });
const rootNativeApiTypedExtraKeys: "tensor"[] = rootNativeApiContract.extraExports({ Tensor, tensor }, { Tensor });
const rootNativeApiRequiredExports: readonly string[] = rootNativeApiContract.requiredNativeApiExports;
const rootNativeApiPackageSpinePrefixExports: readonly string[] = rootNativeApiContract.requiredNativePackageSpinePrefixExports;
const rootNativeApiPackageSpineSuffixExports: readonly string[] = rootNativeApiContract.requiredNativePackageSpineSuffixExports;
const rootNativeApiPackageSpineExports: readonly string[] = rootNativeApiContract.requiredNativePackageSpineExports;

const activationChain = nn.sequential([nn.relu(), nn.square(), nn.sqrt()]);
const activationChainSupport: ModuleCompileSupport = activationChain.compileSupport({ backend: "cpu", inputShape: [2] as const });
const activationChainKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(activationChain, { backend: "cpu", inputShape: [2] as const });
const firstActivationChainOp = activationChainKernelPlan?.ops[0];
const activationChainOp: ModuleKernelPlanActivationChainOp | undefined =
  firstActivationChainOp?.op === "activation-chain" ? firstActivationChainOp : undefined;
const activationChainFusedEdges: readonly ModuleKernelPlanFusedValueEdge[] | undefined = activationChainOp?.fusedValueEdges;
// @ts-expect-error Activation-chain fused op evidence is immutable.
activationChainOp!.fusedOps[0] = "activation";
const tanhModule: CompilableActivationModule = nn.tanh();
const tanhModuleClass: CompilableActivationModule = new nn.Tanh();
const tanhModuleForwardTensor: Tensor<readonly [2, 3]> = tanhModule.forward(literalTensor);
const tanhModuleClassForwardTensor: Tensor<readonly [2, 3]> = tanhModuleClass.forward(literalTensor);
const tanhModuleSupport: ModuleCompileSupport = tanhModule.compileSupport({ backend: "cpu", inputShape: [2] as const });
const typedTanhModuleSupport: ModuleCompileSupport<readonly [2, 3], readonly [2, 3]> = tanhModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const typedTanhModuleCompilePlan: ModuleCompileExplanation<readonly [2, 3], readonly [2, 3]> = tanhModule.compilePlan({ backend: "cpu", inputShape: [2, 3] as const });
const tanhModuleProgram: Program = tanhModule.compile({ backend: "cpu", inputShape: [2] as const });
const typedTanhModuleProgram: Program<readonly [2, 3], readonly [2, 3]> = tanhModule.compile({ backend: "cpu", inputShape: [2, 3] as const });
const typedTanhModuleOutputShape: readonly [2, 3] | null = tanhModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
const tanhModuleSession: Session = tanhModuleProgram.bindModule(tanhModule);
tanhModuleSession.free();
tanhModuleProgram.free();
const minReductionModule = nn.min(0);
const compiledMinReductionSupport: ModuleCompileSupport = minReductionModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const typedMinReductionSupport: ModuleCompileSupport<readonly [2, 3], readonly [1, 3]> = minReductionModule.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const typedMinReductionCompilePlan: ModuleCompileExplanation<readonly [2, 3], readonly [1, 3]> = minReductionModule.compilePlan({ backend: "cpu", inputShape: [2, 3] as const });
const compiledMinReductionKernelPlan: ModuleKernelPlan | null = minReductionModule.kernelPlan({ backend: "cpu", inputShape: [2, 3] as const });
const compiledMinReductionProgram: Program = minReductionModule.compile({ backend: "cpu", inputShape: [2, 3] as const });
const typedMinReductionProgram: Program<readonly [2, 3], readonly [1, 3]> = minReductionModule.compile({ backend: "cpu", inputShape: [2, 3] as const });
const typedMinReductionOutputShape: readonly [1, 3] | null = minReductionModule.outputShape({ backend: "cpu", inputShape: [2, 3] as const });
const compiledMinReductionSession: Session = compiledMinReductionProgram.bindModule(minReductionModule);
compiledMinReductionSession.free();
compiledMinReductionProgram.free();

const trainingDropout = nn.sequential([nn.dropout(0.5), nn.relu()]);
const dropoutModule = nn.dropout(0.25);
const dropoutModuleForwardTensor: Tensor<readonly [2, 3]> = dropoutModule.forward(literalTensor);
const trainingDropoutSupport: ModuleCompileSupport = trainingDropout.compileSupport({ backend: "cpu", inputShape: [2] as const });
const trainingDropoutExplanation: ModuleCompileExplanation = nn.explain(trainingDropout, { backend: "cpu", inputShape: [2] as const });
const trainingDropoutPreflight: ModuleCompileExplanation = nn.preflight(trainingDropout, { backend: "cpu", inputShape: [2] as const });
const trainingDropoutMethodExplanation: ModuleCompileExplanation = trainingDropout.explain({ backend: "cpu", inputShape: [2] as const });
const trainingDropoutMethodPreflight: ModuleCompileExplanation = trainingDropout.preflight({ backend: "cpu", inputShape: [2] as const });
const trainingDropoutCompileExplanation: ModuleCompileExplanation = trainingDropout.compileExplanation({ backend: "cpu", inputShape: [2] as const });
const trainingDropoutNamespaceCompileExplanation: ModuleCompileExplanation = nn.compileExplanation(trainingDropout, { backend: "cpu", inputShape: [2] as const });
const trainingDropoutCompilePlan: ModuleCompileExplanation = nn.compilePlan(trainingDropout, { backend: "cpu", inputShape: [2] as const });
const trainingDropoutMethodCompilePlan: ModuleCompileExplanation = trainingDropout.compilePlan({ backend: "cpu", inputShape: [2] as const });
const trainingDropoutDiagnostics = trainingDropoutSupport.diagnostics;
const trainingDropoutIr: ModuleTensorProgramIr | undefined = trainingDropoutSupport.ir;
const trainingDropoutExplanationKernelPlan: ModuleKernelPlan | null = trainingDropoutExplanation.kernelPlan;
const trainingDropoutExplanationDiagnostics: readonly ModuleCompileDiagnostic[] = trainingDropoutExplanation.diagnostics;
const trainingDropoutMethodExplanationDiagnostics: readonly ModuleCompileDiagnostic[] = trainingDropoutMethodExplanation.diagnostics;
const trainingDropoutCompilePlanDiagnostics: readonly ModuleCompileDiagnostic[] = trainingDropoutCompilePlan.diagnostics;
const trainingDropoutMethodCompilePlanDiagnostics: readonly ModuleCompileDiagnostic[] = trainingDropoutMethodCompilePlan.diagnostics;
const trainingDropoutExplanationParameterNames: readonly string[] = trainingDropoutExplanation.parameterNames;
const trainingDropoutExplanationParameterInfos: readonly ModuleParameterInfo[] = trainingDropoutExplanation.parameterInfos;
const trainingDropoutExplanationReason: string | null = trainingDropoutExplanation.reason;
const trainingDropoutExplanationSignature: string = trainingDropoutExplanation.signature;
const trainingDropoutCompilePlanSignature: string = trainingDropoutCompilePlan.signature;
const trainingDropoutDiagnosticStage: "trace" | "ir" | "kernelizer" | "support" | undefined = trainingDropoutDiagnostics?.[0]?.stage;
const trainingDropoutInputLen: number | undefined = trainingDropoutSupport.inputLen;
const trainingDropoutOutputLen: number | undefined = trainingDropoutSupport.outputLen;
const trainingDropoutInputShape: readonly number[] | undefined = trainingDropoutSupport.inputShape;
const trainingDropoutOutputShape: readonly number[] | undefined = trainingDropoutSupport.outputShape;
// @ts-expect-error Unsupported compiler diagnostics are immutable evidence records.
trainingDropoutDiagnostics![0].message = "changed";
// @ts-expect-error Partial unsupported IR is immutable evidence.
trainingDropoutIr!.ops[0].attrs.p = 0;
// @ts-expect-error Unsupported compile explanations are immutable evidence records.
trainingDropoutExplanationDiagnostics[0].message = "changed";
// @ts-expect-error Unsupported compile explanation parameter info arrays are immutable evidence records.
trainingDropoutExplanationParameterInfos[0] = linearModuleParameterInfos[0];

const logSoftmaxModule = nn.logSoftmax();
const logSoftmaxSnakeModule: ReturnType<typeof nn.log_softmax> = nn.log_softmax(-1);
const logSoftmaxModuleForwardTensor: Tensor<readonly [2, 3]> = logSoftmaxModule.forward(literalTensor);
const logSoftmaxSnakeModuleForwardTensor: Tensor<readonly [2, 3]> = logSoftmaxSnakeModule.forward(literalTensor);
const logSoftmaxSupport: ModuleCompileSupport = logSoftmaxModule.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const logSoftmaxSupportWeightsLen: number | undefined = logSoftmaxSupport.weightsLen;
const logSoftmaxSupportBiasLen: number | undefined = logSoftmaxSupport.biasLen;
const logSoftmaxProgram: Program = logSoftmaxModule.compile({ backend: "cpu", inputShape: [2, 2] as const });
const logSoftmaxProgramWeightsLen: number = logSoftmaxProgram.weightsLen();
const logSoftmaxProgramBiasLen: number = logSoftmaxProgram.biasLen();
const logSoftmaxBindings: ModuleBindings = logSoftmaxModule.bindParameters({ inputShape: [2, 2] as const });
const logSoftmaxRawBindings: ProgramBindings = {};
const logSoftmaxSession: Session = logSoftmaxProgram.bind(logSoftmaxBindings);
const logSoftmaxRawSession: Session = logSoftmaxProgram.bind(logSoftmaxRawBindings);
const logSoftmaxInputBuffer: NativeBuffer = logSoftmaxProgram.createInputBuffer();
const logSoftmaxOutputBuffer: NativeBuffer = logSoftmaxProgram.createOutputBuffer();
const logSoftmaxEmptyWeightsBuffer: NativeBuffer = logSoftmaxProgram.createWeightsBuffer();
const logSoftmaxEmptyBiasBuffer: NativeBuffer = logSoftmaxProgram.createBiasBuffer();
logSoftmaxEmptyBiasBuffer.free();
logSoftmaxEmptyWeightsBuffer.free();
const logSoftmaxResourceInputBuffer: NativeBuffer = logSoftmaxProgram.createInputBuffer({
  resource: (slot: ProgramBufferSlot): NativeBuffer => {
    const inputKind: ProgramBufferSlot["kind"] = slot.kind;
    if (inputKind !== "input" || slot.elementLength !== 4 || slot.layout.input.byteLength !== slot.byteLength) {
      throw new Error("unexpected Program input resource slot");
    }
    return NativeBuffer.create(slot.byteLength);
  },
});
logSoftmaxResourceInputBuffer.free();
const logSoftmaxResourceOutputBuffer: NativeBuffer = logSoftmaxProgram.createOutputBuffer({
  resource: (slot: ProgramOutputBufferSlot): NativeBuffer => {
    const outputKind: "output" = slot.kind;
    if (outputKind !== "output" || slot.elementLength !== 4 || slot.layout.output.byteLength !== slot.byteLength) {
      throw new Error("unexpected Program output resource slot");
    }
    return NativeBuffer.create(slot.byteLength);
  },
});
logSoftmaxResourceOutputBuffer.free();
const logSoftmaxBufferSession: Session = logSoftmaxProgram.bind({
  input: logSoftmaxInputBuffer,
  output: logSoftmaxOutputBuffer,
});
const logSoftmaxDefaultTensorOutput: Tensor = logSoftmaxBufferSession.stepTensor();
const logSoftmaxDefaultTensorOutputShape: readonly number[] = logSoftmaxDefaultTensorOutput.shape;
const logSoftmaxTensorOutput = logSoftmaxBufferSession.stepTensor(undefined, { shape: [2, 2] as const });
const logSoftmaxTensorOutputShape: readonly [2, 2] = logSoftmaxTensorOutput.shape;
const logSoftmaxDefaultReadback: Tensor = logSoftmaxBufferSession.readOutputTensor();
const logSoftmaxDefaultReadbackShape: readonly number[] = logSoftmaxDefaultReadback.shape;
const logSoftmaxReadback = logSoftmaxBufferSession.readOutputTensor({ shape: [2, 2] as const });
const logSoftmaxReadbackShape: readonly [2, 2] = logSoftmaxReadback.shape;
logSoftmaxBufferSession.free();
logSoftmaxOutputBuffer.free();
logSoftmaxInputBuffer.free();
logSoftmaxRawSession.free();
logSoftmaxSession.free();
logSoftmaxProgram.free();

const dim0Softmax = nn.softmax(0);
const dim0SoftmaxForwardTensor: Tensor<readonly [2, 3]> = dim0Softmax.forward(literalTensor);
const dim0SoftmaxSupport: ModuleCompileSupport = dim0Softmax.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const dim0SoftmaxKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(dim0Softmax, { backend: "cpu", inputShape: [2, 3] as const });
const dim0SoftmaxDescriptorSignatures: readonly string[] | undefined = dim0SoftmaxKernelPlan?.ops[0]?.nativeDescriptorSignatures;
const dim0SoftmaxProgram: Program = dim0Softmax.compile({ backend: "cpu", inputShape: [2, 3] as const });
const dim0SoftmaxProgramKernelPlan: ModuleKernelPlan | null = dim0SoftmaxProgram.kernelPlan();
const dim0SoftmaxProgramDescriptorSignatures: readonly string[] | undefined = dim0SoftmaxProgramKernelPlan?.ops[0]?.nativeDescriptorSignatures;
const dim0SoftmaxSession: Session = dim0SoftmaxProgram.bindModule(dim0Softmax);
const dim0SoftmaxOutput = dim0SoftmaxSession.stepTensor(tensor([1, 3, 2, 4, 0, -1], [2, 3] as const));
const dim0SoftmaxOutputShape: readonly number[] = dim0SoftmaxOutput.shape;
dim0SoftmaxSession.free();
dim0SoftmaxProgram.free();

const batchedLogSoftmaxClassifier = nn.sequential([
  nn.linear(3, 2, { weights: [[1, 0], [0, 1], [0, 1]], bias: [0.1, -0.1] }),
  nn.logSoftmax(-1),
]);
const batchedLogSoftmaxClassifierProgram: Program<readonly [4, 3], readonly [4, 2]> = batchedLogSoftmaxClassifier.compile({ backend: "cpu", inputShape: [4, 3] as const });
const batchedLogSoftmaxClassifierSupport: ModuleCompileSupport<readonly [4, 3], readonly [4, 2]> = batchedLogSoftmaxClassifier.compileSupport({ backend: "cpu", inputShape: [4, 3] as const });
batchedLogSoftmaxClassifierProgram.free();

const batchedSoftmaxClassifier = nn.sequential([
  nn.linear(3, 3, { weights: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias: [0.1, -0.2, 0.3] }),
  nn.softmax(-1),
  nn.linear(3, 2, { weights: [[1, 0], [0, 1], [1, -1]], bias: [0.05, -0.05] }),
]);
const batchedSoftmaxClassifierProgram: Program<readonly [4, 3], readonly [4, 2]> = batchedSoftmaxClassifier.compile({ backend: "cpu", inputShape: [4, 3] as const });
const batchedSoftmaxClassifierSupport: ModuleCompileSupport<readonly [4, 3], readonly [4, 2]> = batchedSoftmaxClassifier.compileSupport({ backend: "cpu", inputShape: [4, 3] as const });
batchedSoftmaxClassifierProgram.free();

const minReductionKind: ModuleReductionKind = "min";
const minReduction = nn.min(0);
const minReductionOutput: Tensor<readonly [1, 3]> = minReduction.forward(tensor([1, 3, 2, 4, 0, -1], [2, 3] as const));
const minReductionOutputShape: readonly [1, 3] = minReductionOutput.shape;
const minReductionSupport: ModuleCompileSupport = minReduction.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const minReductionIr: ModuleTensorProgramIr | undefined = minReductionSupport.ir;
const minReductionIrOp: ModuleReductionKind | undefined = minReductionIr?.ops[0]?.op as ModuleReductionKind | undefined;
const prodReductionKind: ModuleReductionKind = "prod";
const prodReduction = nn.prod(1);
const prodReductionOutput: Tensor<readonly [2, 1]> = prodReduction.forward(tensor([1, 3, 2, 4, 5, -1], [2, 3] as const));
const prodReductionOutputShape: readonly [2, 1] = prodReductionOutput.shape;
const prodReductionSupport: ModuleCompileSupport = prodReduction.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const prodReductionIr: ModuleTensorProgramIr | undefined = prodReductionSupport.ir;
const prodReductionIrOp: ModuleReductionKind | undefined = prodReductionIr?.ops[0]?.op as ModuleReductionKind | undefined;
const argmaxReductionKind: ModuleReductionKind = "argmax";
const argmaxTensorOutput: Tensor = literalTensor.argmax(1);
const argmaxTensorDimOutput: Tensor = literalTensor.argmaxDim(0);
const argmaxReduction = nn.argmax(1);
const argmaxReductionOutput: Tensor<readonly [2, 1]> = argmaxReduction.forward(tensor([1, 3, 2, 4, 0, -1], [2, 3] as const));
const argmaxReductionOutputShape: readonly [2, 1] = argmaxReductionOutput.shape;
const argmaxReductionSupport: ModuleCompileSupport = argmaxReduction.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const argmaxReductionIr: ModuleTensorProgramIr | undefined = argmaxReductionSupport.ir;
const argmaxReductionIrOp: ModuleReductionKind | undefined = argmaxReductionIr?.ops[0]?.op as ModuleReductionKind | undefined;
const argminReductionKind: ModuleReductionKind = "argmin";
const argminTensorOutput: Tensor = literalTensor.argmin(1);
const argminTensorDimOutput: Tensor = literalTensor.argminDim(0);
const argminReduction = nn.argmin(1);
const argminReductionOutput: Tensor<readonly [2, 1]> = argminReduction.forward(tensor([1, 3, 2, 4, 0, -1], [2, 3] as const));
const argminReductionOutputShape: readonly [2, 1] = argminReductionOutput.shape;
const argminReductionSupport: ModuleCompileSupport = argminReduction.compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const argminReductionIr: ModuleTensorProgramIr | undefined = argminReductionSupport.ir;
const argminReductionIrOp: ModuleReductionKind | undefined = argminReductionIr?.ops[0]?.op as ModuleReductionKind | undefined;

const elidedShapeChain = nn.sequential([
  nn.identity(),
  nn.unsqueeze(0),
  nn.squeeze(0),
  nn.view([3] as const),
]);
const elidedShapeChainSupport: ModuleCompileSupport = elidedShapeChain.compileSupport({ backend: "cpu", inputShape: [3] as const });
const elidedShapeChainKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(elidedShapeChain, { backend: "cpu", inputShape: [3] as const });
const elidedShapeChainDescriptorSignatures: readonly string[] | undefined = elidedShapeChainKernelPlan?.elidedOps[0]?.nativeDescriptorSignatures;
const elidedShapeChainProgram: Program = elidedShapeChain.compile({ backend: "cpu", inputShape: [3] as const });
const typedElidedShapeChainProgram: Program<readonly [3], readonly [3]> = elidedShapeChain.compile({ backend: "cpu", inputShape: [3] as const });
const typedElidedShapeChainOutputShape: readonly [3] | null = elidedShapeChain.outputShape({ backend: "cpu", inputShape: [3] as const });
const elidedShapeChainProgramKernelPlan: ModuleKernelPlan | null = elidedShapeChainProgram.kernelPlan();
const elidedShapeChainProgramDescriptorSignatures: readonly string[] | undefined = elidedShapeChainProgramKernelPlan?.elidedOps[0]?.nativeDescriptorSignatures;
const elidedShapeChainSession: Session = elidedShapeChainProgram.bindModule(elidedShapeChain);
const elidedShapeChainOutput = elidedShapeChainSession.stepTensor(tensor([4, 5, 6], [3] as const));
const elidedShapeChainOutputShape: readonly number[] = elidedShapeChainOutput.shape;
elidedShapeChainSession.free();
elidedShapeChainProgram.free();

const layerNorm = nn.layerNorm(2, { weight: [1, 1.5], bias: [0.1, -0.2] });
const layerNormSnake: ReturnType<typeof nn.layer_norm> = nn.layer_norm(2, { weight: [1, 1.5], bias: [0.1, -0.2] });
const layerNormModuleForwardTensor: Tensor<readonly [2, 3]> = layerNorm.forward(literalTensor);
const layerNormSnakeModuleForwardTensor: Tensor<readonly [2, 3]> = layerNormSnake.forward(literalTensor);
const layerNormSupport: ModuleCompileSupport = layerNorm.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const layerNormKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(layerNorm, { backend: "cpu", inputShape: [2, 2] as const });
const layerNormDescriptorSignatures: readonly string[] | undefined = layerNormKernelPlan?.ops[0]?.nativeDescriptorSignatures;
const layerNormProgram: Program = nn.compile(layerNorm, { backend: "cpu", inputShape: [2, 2] as const });
const layerNormPlacedBindings: ModuleBindings = nn.placeParameters(layerNorm, layerNormProgram);
const layerNormSession: Session = layerNormProgram.bind({
  ...layerNormPlacedBindings,
  input: tensor([1, 2, 3, 1], [2, 2] as const),
  output: new Float32Array(4),
});
const layerNormOutput = layerNormSession.stepTensor();
const layerNormOutputShape: readonly number[] = layerNormOutput.shape;
layerNormSession.free();
layerNormProgram.free();

const rmsNorm = nn.rmsNorm(2, { eps: 0, weights: [1, 2] });
const rmsNormSnake: ReturnType<typeof nn.rms_norm> = nn.rms_norm(2, { eps: 0, weights: [1, 2] });
const rmsNormModuleForwardTensor: Tensor<readonly [2, 3]> = rmsNorm.forward(literalTensor);
const rmsNormSnakeModuleForwardTensor: Tensor<readonly [2, 3]> = rmsNormSnake.forward(literalTensor);
const rmsNormSupport: ModuleCompileSupport = rmsNorm.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const typedRmsNormSupport: ModuleCompileSupport<readonly [2, 2], readonly [2, 2]> = rmsNorm.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const typedRmsNormCompilePlan: ModuleCompileExplanation<readonly [2, 2], readonly [2, 2]> = rmsNorm.compilePlan({ backend: "cpu", inputShape: [2, 2] as const });
const rmsNormKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(rmsNorm, { backend: "cpu", inputShape: [2, 2] as const });
const rmsNormDescriptorSignatures: readonly string[] | undefined = rmsNormKernelPlan?.ops[0]?.nativeDescriptorSignatures;
const rmsNormProgram: Program = rmsNorm.compile({ backend: "cpu", inputShape: [2, 2] as const });
const typedRmsNormProgram: Program<readonly [2, 2], readonly [2, 2]> = rmsNorm.compile({ backend: "cpu", inputShape: [2, 2] as const });
const typedRmsNormOutputShape: readonly [2, 2] | null = rmsNorm.outputShape({ backend: "cpu", inputShape: [2, 2] as const });
const rmsNormSession: Session = rmsNormProgram.bindModule(rmsNorm);
const rmsNormOutput = rmsNormSession.stepTensor(tensor([3, 4, 1, 2], [2, 2] as const));
const rmsNormOutputShape: readonly number[] = rmsNormOutput.shape;
rmsNormSession.free();
rmsNormProgram.free();

const batchNorm = new nn.BatchNorm1d(3, { momentum: 0.2, runningMean: [0, 0, 0], runningVar: [1, 1, 1] });
const batchNormFactory: ReturnType<typeof nn.batchNorm1d> = nn.batchNorm1d(3);
const batchNormSnake: ReturnType<typeof nn.batch_norm1d> = nn.batch_norm1d(3, { track_running_stats: true });
const batchNormModuleForwardTensor: Tensor<readonly [2, 3]> = batchNorm.forward(literalTensor);
const batchNormEvalForwardTensor: Tensor<readonly [2, 3]> = batchNorm.eval().forward(literalTensor);
const batchNormState: ModuleStateSnapshot = batchNorm.stateDict("bn");
const batchNormRunningMean: NnBuffer | null = batchNorm.runningMean;
const batchNormRunningVar: NnBuffer | null = batchNorm.runningVar;
const batchNormTracked: NnBuffer | null = batchNorm.numBatchesTracked;
const batchNormTrace = nn.sequential(batchNormFactory).trace({ inputShape: [2, 3] as const });
const batchNormTraceOutputShape: readonly number[] | null | undefined = batchNormTrace.outputShape;
const batchNormTracedOutputShape: readonly number[] | null = nn.outputShape(nn.sequential(batchNormFactory), { inputShape: [2, 3] as const });
const batchNormSupport: ModuleCompileSupport = nn.sequential(batchNormFactory).compileSupport({ backend: "cpu", inputShape: [2, 3] as const });
const conv2dConfig: NnConv2dConfig = { stride: [1, 1] as const, padding: 0, dilation: 1, weight: [1, 0, 0, 1], bias: false };
const conv2d: Conv2dModule<1, 1> = new nn.Conv2d(1, 1, 2, conv2dConfig);
const conv2dFactory: ReturnType<typeof nn.conv2d> = nn.conv2d(1, 1, [2, 2] as const, conv2dConfig);
const conv2dSpatialConfig: NnConv2dConfig = { stride: 2, padding: [1, 1] as const, dilation: [2, 2] as const, weight: [1, 1, 1, 1], bias: false };
const conv2dSpatial: Conv2dModule<1, 1> = nn.conv2d(1, 1, 2, conv2dSpatialConfig);
const conv2dForwardTensor: Tensor<readonly [1, number, number]> = conv2d.forward(tensor([1, 2, 3, 4], [1, 2, 2] as const));
const conv2dSpatialForwardTensor: Tensor<readonly [1, number, number]> = conv2dSpatial.forward(tensor([
  1, 2, 3, 4, 5,
  6, 7, 8, 9, 10,
  11, 12, 13, 14, 15,
  16, 17, 18, 19, 20,
  21, 22, 23, 24, 25,
], [1, 5, 5] as const));
const conv2dBatchForwardTensor: Tensor<readonly [1, 1, number, number]> = conv2d.forward(tensor([1, 2, 3, 4], [1, 1, 2, 2] as const));
const conv2dState: ModuleStateSnapshot = conv2d.stateDict("conv");
const conv2dSupport: ModuleCompileSupport = conv2d.compileSupport({ inputShape: [1, 2, 2] as const });
const conv2dTrace = nn.sequential(conv2d).trace({ inputShape: [1, 3, 3] as const });
const conv2dTraceOutputShape: readonly number[] | null | undefined = conv2dTrace.outputShape;
const conv2dTracedOutputShape: readonly number[] | null = nn.outputShape(nn.sequential(conv2d), { inputShape: [1, 3, 3] as const });
const conv2dSequentialSupport: ModuleCompileSupport = nn.sequential(conv2d).compileSupport({ backend: "cpu", inputShape: [1, 3, 3] as const });
const conv2dSequentialIrOp = conv2dSequentialSupport.ir?.ops[0];
const conv2dBatchedProgram: Program<readonly [2, 1, 3, 3], readonly [2, 1, number, number]> = conv2d.compile({ backend: "cpu", inputShape: [2, 1, 3, 3] as const });
const conv2dBatchedBindings: ModuleBindings<readonly [2, 1, 3, 3], readonly [2, 1, number, number]> = conv2d.bindParameters({ backend: "cpu", inputShape: [2, 1, 3, 3] as const });
const maxPool2dConfig: NnMaxPool2dConfig = { stride: 1, padding: [0, 0] as const, dilation: 1, ceilMode: false };
const maxPool2d: MaxPool2dModule = new nn.MaxPool2d(2, maxPool2dConfig);
const maxPool2dFactory: ReturnType<typeof nn.max_pool2d> = nn.max_pool2d([2, 2] as const, maxPool2dConfig);
const maxPool2dForwardTensor: Tensor<readonly [1, number, number]> = maxPool2d.forward(tensor([1, 2, 3, 4], [1, 2, 2] as const));
const maxPool2dTrace = nn.sequential(conv2d, maxPool2dFactory).trace({ inputShape: [1, 4, 4] as const });
const maxPool2dTraceOutputShape: readonly number[] | null | undefined = maxPool2dTrace.outputShape;
const maxPool2dTracedOutputShape: readonly number[] | null = nn.outputShape(nn.sequential(maxPool2d), { inputShape: [1, 3, 3] as const });
const maxPool2dSupport: ModuleCompileSupport = nn.sequential(maxPool2dFactory).compileSupport({ backend: "cpu", inputShape: [1, 3, 3] as const });
const maxPool2dIrOp = maxPool2dSupport.ir?.ops[0];
const directMaxPool2dSupport: ModuleCompileSupport = maxPool2d.compileSupport({ backend: "cpu", inputShape: [1, 4, 4] as const });
const directMaxPool2dProgram = maxPool2d.compile({ backend: "cpu", inputShape: [1, 4, 4] as const });
const directMaxPool2dNamespaceProgram: Program<readonly [1, 4, 4], readonly [1, number, number]> = compile.compile(maxPool2d, { backend: "cpu", inputShape: [1, 4, 4] as const });
const directMaxPool2dNnNamespaceProgram: Program<readonly [1, 4, 4], readonly [1, number, number]> = nn.compile(maxPool2d, { backend: "cpu", inputShape: [1, 4, 4] as const });
const batchedMaxPool2dProgram: Program<readonly [2, 2, 4, 4], readonly [2, 2, number, number]> = maxPool2d.compile({ backend: "cpu", inputShape: [2, 2, 4, 4] as const });
const batchedMaxPool2dBindings: ModuleBindings<readonly [2, 2, 4, 4], readonly [2, 2, number, number]> = maxPool2d.bindParameters({ backend: "cpu", inputShape: [2, 2, 4, 4] as const });
const fixedMaxPool2dSupport: ModuleCompileSupport = nn.sequential(nn.max_pool2d(2)).compileSupport({ backend: "cpu", inputShape: [1, 4, 4] as const });
const fixedMaxPool2dKernel = fixedMaxPool2dSupport.kernelPlan?.ops[0]?.kernel;
const avgPool2dConfig: NnAvgPool2dConfig = { stride: [1, 1] as const, padding: 0, ceilMode: false, countIncludePad: true };
const avgPool2d: AvgPool2dModule = new nn.AvgPool2d(2, avgPool2dConfig);
const avgPool2dFactory: ReturnType<typeof nn.avg_pool2d> = nn.avg_pool2d([2, 2] as const, avgPool2dConfig);
const avgPool2dForwardTensor: Tensor<readonly [1, number, number]> = avgPool2d.forward(tensor([1, 2, 3, 4], [1, 2, 2] as const));
const avgPool2dTrace = nn.sequential(avgPool2dFactory).trace({ inputShape: [1, 3, 3] as const });
const avgPool2dTraceOutputShape: readonly number[] | null | undefined = avgPool2dTrace.outputShape;
const avgPool2dTracedOutputShape: readonly number[] | null = nn.outputShape(nn.sequential(avgPool2d), { inputShape: [1, 3, 3] as const });
const avgPool2dSupport: ModuleCompileSupport = nn.sequential(avgPool2dFactory).compileSupport({ backend: "cpu", inputShape: [1, 3, 3] as const });
const avgPool2dIrOp = avgPool2dSupport.ir?.ops[0];
const directAvgPool2dSupport: ModuleCompileSupport = avgPool2d.compileSupport({ backend: "cpu", inputShape: [1, 4, 4] as const });
const directAvgPool2dProgram = avgPool2d.compile({ backend: "cpu", inputShape: [1, 4, 4] as const });
const directAvgPool2dNamespaceProgram: Program<readonly [1, 4, 4], readonly [1, number, number]> = compile.compile(avgPool2d, { backend: "cpu", inputShape: [1, 4, 4] as const });
const directAvgPool2dNnNamespaceProgram: Program<readonly [1, 4, 4], readonly [1, number, number]> = nn.compile(avgPool2d, { backend: "cpu", inputShape: [1, 4, 4] as const });
const batchedAvgPool2dProgram: Program<readonly [2, 2, 4, 4], readonly [2, 2, number, number]> = avgPool2d.compile({ backend: "cpu", inputShape: [2, 2, 4, 4] as const });
const batchedAvgPool2dBindings: ModuleBindings<readonly [2, 2, 4, 4], readonly [2, 2, number, number]> = avgPool2d.bindParameters({ backend: "cpu", inputShape: [2, 2, 4, 4] as const });
const fixedAvgPool2dSupport: ModuleCompileSupport = nn.sequential(nn.avg_pool2d(2)).compileSupport({ backend: "cpu", inputShape: [1, 4, 4] as const });
const fixedAvgPool2dKernel = fixedAvgPool2dSupport.kernelPlan?.ops[0]?.kernel;

const nestedTensor = tensor([[1, 2], [3, 4]]);
const nestedTensorShape: readonly number[] = nestedTensor.shape;
const nestedTensorRows: TensorNestedArray = nestedTensor.toArray();
const nestedTensorList: TensorNestedArray = nestedTensor.tolist();
const nestedTensorListAlias: TensorNestedArray = nestedTensor.to_list();
const nestedTensorNumpy: Float32Array = nestedTensor.numpy();
const nestedTensorJson: TensorJSON = nestedTensor.toJSON();
const nestedTensorJsonRequiresGrad: boolean | undefined = nestedTensorJson.requiresGrad;
const nestedTensorJsonRequiresGradAlias: boolean | undefined = nestedTensorJson.requires_grad;
const nestedTensorFromJson: Tensor = Tensor.fromJSON(nestedTensorJson);
const nestedTensorFromTrainableJson: Tensor = Tensor.fromJSON({ dtype: "f32", shape: [1] as const, data: [1], requiresGrad: true });
const nestedTensorFromTrainableJsonAlias: Tensor = Tensor.fromJSON({ dtype: "f32", shape: [1] as const, data: [1], requires_grad: true });
const nestedTensorRequiresGrad: boolean = nestedTensor.requires_grad;
nestedTensor.requires_grad = true;
const nestedTensorRequiresGradAlias: boolean = nestedTensor.requiresGrad;
const nestedTensorRequiresGradInPlace: Tensor = nestedTensor.requires_grad_();
const nestedTensorRequiresGradCamelInPlace: Tensor = nestedTensor.requiresGrad_(false);
const nestedTensorDetachedInPlace: Tensor = nestedTensor.detach_();

const composedNorm = nn.sequential([
  nn.linear(2, 2, { weights: [[1, 0], [0, 1]], bias: [0.1, -0.2] }),
  nn.layerNorm(2, { weight: [1, 1.5], bias: [0.01, -0.02] }),
  nn.gelu(),
  nn.linear(2, 1, { weights: [[1], [-0.5]], bias: [0.2] }),
]);
const composedNormSupport: ModuleCompileSupport = composedNorm.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const composedNormKernelPlan: ModuleKernelPlan | null = nn.kernelPlan(composedNorm, { backend: "cpu", inputShape: [2, 2] as const });
const composedNormDescriptorSignatures: readonly string[] | undefined = composedNormKernelPlan?.ops[1]?.nativeDescriptorSignatures;
const composedNormProgram: Program = composedNorm.compile({ backend: "cpu", inputShape: [2, 2] as const });
const composedNormTypedProgram: Program<readonly [2, 2], readonly [2, 1]> = composedNorm.compile({ backend: "cpu", inputShape: [2, 2] as const });
const composedNormTypedSupport: ModuleCompileSupport<readonly [2, 2], readonly [2, 1]> = composedNorm.compileSupport({ backend: "cpu", inputShape: [2, 2] as const });
const composedNormCompatibility: ProgramModuleCompatibility = composedNormProgram.moduleCompatibility(composedNorm);
const composedNormNestedInput: ProgramInputBinding = [[0.2, -0.4], [1.2, 0.7]];
const composedNormDirectSession: Session = composedNormProgram.bind({
  ...composedNorm.bindParameters({ inputShape: [2, 2] as const }),
  input: composedNormNestedInput,
  output: new Float32Array(2),
});
const composedNormDirectOutput = composedNormDirectSession.stepTensor(composedNormNestedInput);
const composedNormDirectOutputShape: readonly number[] = composedNormDirectOutput.shape;
composedNormDirectSession.free();
const composedNormSession: Session = composedNormProgram.bindModule(composedNorm);
const composedNormOutput = composedNormSession.stepTensor(tensor([[0.2, -0.4], [1.2, 0.7]]));
const composedNormOutputShape: readonly number[] = composedNormOutput.shape;
composedNormSession.free();
composedNormTypedProgram.free();
composedNormProgram.free();

const tinyLlamaProgram: TinyLlamaProgram = TinyLlama.create().compile({ backend: "cpu", contextLength: 8 });
const llamaProgramInspection: LlamaProgramInspection = tinyLlamaProgram.inspect();
const llamaProgramInspectionKind: "zgml.llama.program.inspection" = llamaProgramInspection.kind;
const llamaProgramInspectionSignature: string = llamaProgramInspection.signature;
const llamaProgramRequirements: ProgramRequirements = tinyLlamaProgram.requirements();
const llamaProgramRequirementsKind: "zgml.program.requirements" = llamaProgramRequirements.kind;
const llamaProgramRequirementsSignature: string = llamaProgramRequirements.signature;
const llamaProgramInputLen: number = tinyLlamaProgram.inputLen();
const llamaProgramOutputLen: number = tinyLlamaProgram.outputLen();
const llamaProgramInputByteLength: number = tinyLlamaProgram.inputByteLength();
const llamaProgramOutputByteLength: number = tinyLlamaProgram.outputByteLength();
const llamaProgramWeightsLen: number = tinyLlamaProgram.weightsLen();
const llamaProgramWeightsByteLength: number = tinyLlamaProgram.weightsByteLength();
const llamaProgramBiasLen: number = tinyLlamaProgram.biasLen();
const llamaProgramBiasByteLength: number = tinyLlamaProgram.biasByteLength();
const llamaProgramParameterLen: number = tinyLlamaProgram.parameterLen();
const llamaProgramParameterByteLength: number = tinyLlamaProgram.parameterByteLength();
const llamaProgramBufferSizing: ProgramBufferSizing = tinyLlamaProgram.bufferSizing();
const llamaProgramBufferSizingKind: "program-buffer-sizing" = llamaProgramBufferSizing.kind;
const llamaProgramBufferSizingSignature: string = llamaProgramBufferSizing.signature;
const llamaProgramBufferSizingModelKind: ProgramRequirements["modelKind"] = llamaProgramBufferSizing.modelKind;
const llamaProgramMatchesBufferSizingSignature: boolean =
  tinyLlamaProgram.matchesBufferSizingSignature(llamaProgramBufferSizingSignature);
const llamaProgramCapabilities: ProgramExecutionCapabilities = tinyLlamaProgram.capabilities();
const llamaProgramCapabilitiesKind: "zgml.program.capabilities" = llamaProgramCapabilities.kind;
const llamaProgramModelCompatibility: ProgramModelCompatibility = tinyLlamaProgram.modelCompatibility(TinyLlama.create());
const llamaProgramModelCompatibilityKind: "zgml.program.model-compatibility" = llamaProgramModelCompatibility.kind;
const llamaProgramModelCompatibilitySignature: string = llamaProgramModelCompatibility.signature;
const llamaProgramExecutionPlan: ProgramExecutionPlan = tinyLlamaProgram.executionPlan();
const llamaProgramExecutionPlanAlias: ProgramExecutionPlan = tinyLlamaProgram.execution_plan();
const llamaProgramRequiredExecutionPlan: ProgramExecutionPlan = tinyLlamaProgram.requireExecutionPlan();
const llamaProgramRequiredExecutionPlanAlias: ProgramExecutionPlan = tinyLlamaProgram.require_execution_plan();
const llamaProgramExecutionPlanKind: "zgml.program.execution-plan" = llamaProgramExecutionPlan.kind;
const llamaProgramExecutionPlanSignature: string = llamaProgramExecutionPlan.signature;
const llamaProgramExecutionPlanProgramKind: "generic" | "llama" = llamaProgramExecutionPlan.programKind;
const llamaProgramExecutionPlanInputShape: readonly number[] | null = llamaProgramExecutionPlan.inputShape;
const llamaProgramExecutionPlanOutputShape: readonly number[] = llamaProgramExecutionPlan.outputShape;
const llamaProgramExecutionPlanCompileEvidence: ProgramCompileEvidence | null = llamaProgramExecutionPlan.compileEvidence;
const llamaProgramCapabilitySignature: string = llamaProgramCapabilities.signature;
const llamaProgramMatchesCapabilitySignature: boolean =
  tinyLlamaProgram.matchesCapabilitySignature(llamaProgramCapabilitySignature);
const llamaProgramMatchesCapabilitySignatureAlias: boolean =
  tinyLlamaProgram.matches_capability_signature(llamaProgramCapabilitySignature);
const llamaProgramCanExecuteAlias: boolean = tinyLlamaProgram.can_execute();
const llamaProgramCanBindExternalResourcesAlias: boolean = tinyLlamaProgram.can_bind_external_resources();
const llamaProgramHasFullDispatchPlanAlias: boolean = tinyLlamaProgram.has_full_dispatch_plan();
const llamaProgramExecutionModeAlias: ProgramExecutionCapabilities["mode"] = tinyLlamaProgram.execution_mode();
const llamaProgramBufferLayout: ProgramBufferLayout = tinyLlamaProgram.bufferLayout();
const llamaProgramBufferLayoutKind: "zgml.program.buffer-layout" = llamaProgramBufferLayout.kind;
const llamaProgramBufferLayoutSignature: string = llamaProgramBufferLayout.signature;
const llamaProgramBufferSlotNames: readonly string[] = tinyLlamaProgram.bufferSlotNames();
const llamaProgramOutputSlot: ProgramBufferLayoutSlot | null = tinyLlamaProgram.bufferSlot("output");
const llamaKvCacheRequirements: LlamaKvCacheRequirements = tinyLlamaProgram.kvCacheRequirements();
const llamaKvCacheRequirementsKind: "zgml.llama.kv-cache.requirements" = llamaKvCacheRequirements.kind;
const llamaKvCacheRequirementsSignature: string = llamaKvCacheRequirements.signature;
const llamaKvCacheLayout: LlamaKvCacheLayout = tinyLlamaProgram.kvCacheLayout();
const llamaKvCacheLayoutKind: "zgml.llama.kv-cache.layout" = llamaKvCacheLayout.kind;
const llamaKvCacheLayoutSignature: string = llamaKvCacheLayout.signature;
const llamaKvCache: LlamaKvCache = tinyLlamaProgram.createKvCache();
llamaKvCache.free();
const llamaResourceKvCache: LlamaKvCache = tinyLlamaProgram.createKvCache({
  resource: (slot: LlamaKvCacheSlot): NativeBuffer => {
    const slotKind: "k" | "v" = slot.kind;
    const slotLayer: number = slot.layer;
    const slotLayout: LlamaKvCacheLayout = slot.layout;
    const slotRequirements: LlamaKvCacheRequirements = slot.requirements;
    if (slotLayer < 0 || slotLayout.layers !== slotRequirements.layers || (slotKind !== "k" && slotKind !== "v")) {
      throw new Error("unexpected LLaMA KV-cache resource slot");
    }
    return NativeBuffer.create(slot.byteLength);
  },
});
llamaResourceKvCache.free();
const llamaProgramOutputShape: readonly number[] = tinyLlamaProgram.outputShape();
const llamaProgramRuntimeProfile: RuntimeProfile = tinyLlamaProgram.runtimeProfile();
const llamaProgramRuntimeProfileAlias: RuntimeProfile = tinyLlamaProgram.runtime_profile();
const llamaProgramRuntimeProfileKind: "zgml.runtime.profile" = llamaProgramRuntimeProfile.kind;
const llamaProgramRuntimeProfileSignature: string = llamaProgramRuntimeProfile.signature;
const llamaProgramMatchesRuntimeProfileSignature: boolean =
  tinyLlamaProgram.matchesRuntimeProfileSignature(llamaProgramRuntimeProfileSignature);
const llamaProgramMatchesRuntimeProfileSignatureAlias: boolean =
  tinyLlamaProgram.matches_runtime_profile_signature(llamaProgramRuntimeProfileSignature);
const llamaProgramRuntimeProfileExpectationFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.runtimeProfileExpectation();
const llamaProgramRuntimeProfileExpectationAliasFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.runtime_profile_expectation();
const llamaProgramRuntimeProfileNoFallbackFromProgram: boolean =
  tinyLlamaProgram.runtimeProfileHasNoFallback();
const llamaProgramRuntimeProfileNoFallbackAliasFromProgram: boolean =
  tinyLlamaProgram.runtime_profile_has_no_fallback();
const llamaProgramRuntimeProfileNoSyncFromProgram: boolean =
  tinyLlamaProgram.runtimeProfileHasNoSync();
const llamaProgramRuntimeProfileNoSyncAliasFromProgram: boolean =
  tinyLlamaProgram.runtime_profile_has_no_sync();
const llamaProgramRuntimeProfileNoInvalidPatchesFromProgram: boolean =
  tinyLlamaProgram.runtimeProfileHasNoInvalidRuntimePatches();
const llamaProgramRuntimeProfileNoInvalidPatchesAliasFromProgram: boolean =
  tinyLlamaProgram.runtime_profile_has_no_invalid_runtime_patches();
const llamaProgramRuntimeProfileRequiredFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.requireNoFallbackRuntimeProfile();
const llamaProgramRuntimeProfileRequiredAliasFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.require_no_fallback_runtime_profile();
const llamaProgramRuntimeProfileNoSyncRequiredFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.requireNoSyncRuntimeProfile();
const llamaProgramRuntimeProfileNoSyncRequiredAliasFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.require_no_sync_runtime_profile();
const llamaProgramRuntimePatchValidFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.requireRuntimePatchValidProfile();
const llamaProgramRuntimePatchValidAliasFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.require_runtime_patch_valid_profile();
const llamaProgramRuntimeProfileHotFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.requireHotRuntimeProfile();
const llamaProgramRuntimeProfileHotAliasFromProgram: RuntimeProfileExpectation =
  tinyLlamaProgram.require_hot_runtime_profile();
tinyLlamaProgram.resetRuntimeProfile();
tinyLlamaProgram.reset_runtime_profile();
const tinyLlamaSession: TinyLlamaSession = tinyLlamaProgram.bind({ output: "native" });
const llamaSessionInspection: SessionInspection = tinyLlamaSession.inspect();
const llamaSessionInspectionKind: "zgml.session.inspection" = llamaSessionInspection.kind;
const llamaSessionInspectionSignature: string = llamaSessionInspection.signature;
const llamaSessionInputLen: number = tinyLlamaSession.inputLen();
const llamaSessionOutputLen: number = tinyLlamaSession.outputLen();
const llamaSessionInputByteLength: number = tinyLlamaSession.inputByteLength();
const llamaSessionOutputByteLength: number = tinyLlamaSession.outputByteLength();
const llamaSessionWeightsLen: number = tinyLlamaSession.weightsLen();
const llamaSessionWeightsByteLength: number = tinyLlamaSession.weightsByteLength();
const llamaSessionBiasLen: number = tinyLlamaSession.biasLen();
const llamaSessionBiasByteLength: number = tinyLlamaSession.biasByteLength();
const llamaSessionParameterLen: number = tinyLlamaSession.parameterLen();
const llamaSessionParameterByteLength: number = tinyLlamaSession.parameterByteLength();
const llamaSessionBufferSizing: SessionBufferSizing = tinyLlamaSession.bufferSizing();
const llamaSessionBufferSizingKind: "session-buffer-sizing" = llamaSessionBufferSizing.kind;
const llamaSessionBufferSizingSignature: string = llamaSessionBufferSizing.signature;
const llamaSessionBufferSizingModelKind: ProgramRequirements["modelKind"] = llamaSessionBufferSizing.modelKind;
const llamaSessionMatchesBufferSizingSignature: boolean =
  tinyLlamaSession.matchesBufferSizingSignature(llamaSessionBufferSizingSignature);
const llamaSessionBufferLayout: ProgramBufferLayout = tinyLlamaSession.bufferLayout();
const llamaSessionBufferLayoutKind: "zgml.program.buffer-layout" = llamaSessionBufferLayout.kind;
const llamaSessionBufferLayoutSignature: string = llamaSessionBufferLayout.signature;
const llamaSessionBufferSlotNames: readonly string[] = tinyLlamaSession.bufferSlotNames();
const llamaSessionOutputSlot: ProgramBufferLayoutSlot | null = tinyLlamaSession.bufferSlot("output");
const llamaSessionKvCacheLayout: LlamaKvCacheLayout = tinyLlamaSession.kvCacheLayout();
const llamaSessionKvCacheLayoutKind: "zgml.llama.kv-cache.layout" = llamaSessionKvCacheLayout.kind;
const llamaSessionKvCacheLayoutSignature: string = llamaSessionKvCacheLayout.signature;
const llamaSessionOutputShape: readonly number[] = tinyLlamaSession.outputShape();
const llamaSessionRuntimeProfile: RuntimeProfile = tinyLlamaSession.runtimeProfile();
const llamaSessionRuntimeProfileAlias: RuntimeProfile = tinyLlamaSession.runtime_profile();
const llamaSessionRuntimeProfileKind: "zgml.runtime.profile" = llamaSessionRuntimeProfile.kind;
const llamaSessionRuntimeProfileSignature: string = llamaSessionRuntimeProfile.signature;
const llamaSessionMatchesRuntimeProfileSignature: boolean =
  tinyLlamaSession.matchesRuntimeProfileSignature(llamaSessionRuntimeProfileSignature);
const llamaSessionMatchesRuntimeProfileSignatureAlias: boolean =
  tinyLlamaSession.matches_runtime_profile_signature(llamaSessionRuntimeProfileSignature);
const llamaSessionRuntimeProfileExpectationFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.runtimeProfileExpectation();
const llamaSessionRuntimeProfileExpectationAliasFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.runtime_profile_expectation();
const llamaSessionRuntimeProfileNoFallbackFromSession: boolean =
  tinyLlamaSession.runtimeProfileHasNoFallback();
const llamaSessionRuntimeProfileNoFallbackAliasFromSession: boolean =
  tinyLlamaSession.runtime_profile_has_no_fallback();
const llamaSessionRuntimeProfileNoSyncFromSession: boolean =
  tinyLlamaSession.runtimeProfileHasNoSync();
const llamaSessionRuntimeProfileNoSyncAliasFromSession: boolean =
  tinyLlamaSession.runtime_profile_has_no_sync();
const llamaSessionRuntimeProfileNoInvalidPatchesFromSession: boolean =
  tinyLlamaSession.runtimeProfileHasNoInvalidRuntimePatches();
const llamaSessionRuntimeProfileNoInvalidPatchesAliasFromSession: boolean =
  tinyLlamaSession.runtime_profile_has_no_invalid_runtime_patches();
const llamaSessionRuntimeProfileRequiredFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.requireNoFallbackRuntimeProfile();
const llamaSessionRuntimeProfileRequiredAliasFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.require_no_fallback_runtime_profile();
const llamaSessionRuntimeProfileNoSyncRequiredFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.requireNoSyncRuntimeProfile();
const llamaSessionRuntimeProfileNoSyncRequiredAliasFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.require_no_sync_runtime_profile();
const llamaSessionRuntimePatchValidFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.requireRuntimePatchValidProfile();
const llamaSessionRuntimePatchValidAliasFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.require_runtime_patch_valid_profile();
const llamaSessionRuntimeProfileHotFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.requireHotRuntimeProfile();
const llamaSessionRuntimeProfileHotAliasFromSession: RuntimeProfileExpectation =
  tinyLlamaSession.require_hot_runtime_profile();
tinyLlamaSession.resetRuntimeProfile();
const llamaSessionCallProfile: SessionCallProfile = tinyLlamaSession.sessionCallProfile();
const llamaSessionCallProfileAlias: SessionCallProfile = tinyLlamaSession.session_call_profile();
const llamaSessionCallProfileKind: "zgml.session.call-profile" = llamaSessionCallProfile.kind;
const llamaSessionCallProfileSignature: string = llamaSessionCallProfile.signature;
const llamaSessionMatchesCallProfileSignature: boolean =
  tinyLlamaSession.matchesSessionCallProfileSignature(llamaSessionCallProfileSignature);
const llamaSessionMatchesCallProfileSignatureAlias: boolean =
  tinyLlamaSession.matches_session_call_profile_signature(llamaSessionCallProfileSignature);
const llamaPrefillCount: number = llamaSessionCallProfile.prefillTensorCount;
tinyLlamaSession.resetSessionCallProfile();
tinyLlamaSession.reset_session_call_profile();
const llamaSessionStepContract: LlamaSessionStepContract = tinyLlamaSession.stepContract();
const llamaSessionStepContractAlias: LlamaSessionStepContract = tinyLlamaSession.step_contract();
const llamaSessionStepContractOutputShape: readonly number[] = llamaSessionStepContract.outputShape;
const llamaSessionStepContractSignature: string = llamaSessionStepContract.signature;
const llamaSessionStepContractOutputByteLength: number = llamaSessionStepContract.outputByteLength;
const llamaSessionStepContractOutputSlotName: "output" = llamaSessionStepContract.outputSlotName;
const llamaSessionStepContractOutputSlotRole: "step-output" = llamaSessionStepContract.outputSlotRole;
const llamaSessionStepContractRemainingContext: number = llamaSessionStepContract.remainingContext;
const llamaSessionStepContractDefaultOutput: SessionDefaultOutputKind = llamaSessionStepContract.defaultOutput;
const llamaSessionStepContractDefaultReadbackRequired: boolean = llamaSessionStepContract.defaultReadbackRequired;
const llamaSessionStepContractDefaultAllocationFree: boolean = llamaSessionStepContract.defaultAllocationFree;
const llamaSessionStepContractNoOutputEffect: SessionNoOutputEffect = llamaSessionStepContract.noOutputEffect;
const llamaSessionMatchesStepContractSignature: boolean =
  tinyLlamaSession.matchesStepContractSignature(llamaSessionStepContractSignature);
const llamaSessionMatchesStepContractSignatureAlias: boolean =
  tinyLlamaSession.matches_step_contract_signature(llamaSessionStepContractSignature);
const llamaSessionStepParamsPreflight: SessionStepParamsCompatibility = tinyLlamaSession.preflightStepParams({ token: 0, output: false });
const llamaSessionStepParamsPreflightAlias: SessionStepParamsCompatibility = tinyLlamaSession.preflight_step_params({ token: 0, output: false });
const llamaSessionStepParamsCompatibility: SessionStepParamsCompatibility = tinyLlamaSession.stepParamsCompatibility({ token: 0, output: false });
const llamaSessionStepParamsCompatibilityAlias: SessionStepParamsCompatibility = tinyLlamaSession.step_params_compatibility({ token: 0, output: false });
const llamaSessionStepParamsKind: "zgml.step-params.compatibility" = llamaSessionStepParamsCompatibility.kind;
const llamaSessionStepParamsStatus: "accepted" | "rejected" = llamaSessionStepParamsCompatibility.status;
const llamaSessionStepParamsContractKind: SessionStepParamsCompatibility["contractKind"] = llamaSessionStepParamsCompatibility.contractKind;
const llamaSessionStepParamsContractPosition: number | null = llamaSessionStepParamsCompatibility.contractPosition;
const llamaSessionStepParamsSignature: string = llamaSessionStepParamsCompatibility.stepParamsSignature;
const llamaSessionStepParamsRejectionCode: SessionStepParamsDiagnosticCode | null = llamaSessionStepParamsCompatibility.rejectionCode;
const llamaSessionStepParamsStateEffect: SessionStepParamsStateEffect = llamaSessionStepParamsCompatibility.stateEffect;
const llamaSessionStepParamsAllocationFree: boolean = llamaSessionStepParamsCompatibility.allocationFree;
const llamaSessionStepParamsInputSource: SessionStepParamsInputSource = llamaSessionStepParamsCompatibility.inputSource;
const llamaSessionStepParamsInputOwnership: SessionStepParamsInputOwnership = llamaSessionStepParamsCompatibility.inputOwnership;
const llamaSessionStepParamsReadsInput: boolean = llamaSessionStepParamsCompatibility.readsInput;
const llamaSessionStepParamsOutputTarget: SessionStepParamsOutputTarget = llamaSessionStepParamsCompatibility.outputTarget;
const llamaSessionStepParamsOutputEffect: SessionStepParamsOutputEffect = llamaSessionStepParamsCompatibility.outputEffect;
const llamaSessionStepParamsOutputOwnership: SessionStepParamsOutputOwnership = llamaSessionStepParamsCompatibility.outputOwnership;
const llamaSessionStepParamsOutputReturnOwnership: SessionStepParamsOutputReturnOwnership = llamaSessionStepParamsCompatibility.outputReturnOwnership;
const llamaSessionStepParamsWritesOutput: boolean = llamaSessionStepParamsCompatibility.writesOutput;
const llamaSessionStepParamsReadbackRequired: boolean = llamaSessionStepParamsCompatibility.readbackRequired;
const llamaSessionStepParamsHotPath: boolean = llamaSessionStepParamsCompatibility.hotPath;
const llamaSessionStepParamsHotPathStatus: SessionStepParamsHotPathStatus = llamaSessionStepParamsCompatibility.hotPathStatus;
const llamaSessionStepParamsInputElementType: SessionStepParamsElementType = llamaSessionStepParamsCompatibility.inputElementType;
const llamaSessionStepParamsOutputElementType: SessionStepParamsElementType = llamaSessionStepParamsCompatibility.outputElementType;
const llamaSessionStepParamsInputElementLength: number = llamaSessionStepParamsCompatibility.inputElementLength;
const llamaSessionStepParamsOutputElementLength: number = llamaSessionStepParamsCompatibility.outputElementLength;
const llamaSessionStepParamsInputByteLength: number = llamaSessionStepParamsCompatibility.inputByteLength;
const llamaSessionStepParamsOutputByteLength: number = llamaSessionStepParamsCompatibility.outputByteLength;
const llamaSessionStepParamsDiagnosticRemainingContext: number | undefined = llamaSessionStepParamsCompatibility.diagnostics[0]?.remainingContext;
const llamaSessionAcceptsStepParams: boolean = tinyLlamaSession.acceptsStepParams({ token: 0, output: false });
const llamaSessionAcceptsStepParamsAlias: boolean = tinyLlamaSession.accepts_step_params({ token: 0, output: false });
const llamaSessionRequiredCanExecuteStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireCanExecuteStepParams({ token: 0, output: false });
const llamaSessionRequiredCanExecuteStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_can_execute_step_params({ token: 0, output: false });
const llamaSessionAcceptsAllocationFreeStepParams: boolean = tinyLlamaSession.acceptsAllocationFreeStepParams({ token: 0, output: false });
const llamaSessionAcceptsAllocationFreeStepParamsAlias: boolean = tinyLlamaSession.accepts_allocation_free_step_params({ token: 0, output: false });
const llamaSessionRequiredAllocationFreeStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireAllocationFreeStepParams({ token: 0, output: false });
const llamaSessionRequiredAllocationFreeStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_allocation_free_step_params({ token: 0, output: false });
const llamaSessionRequiredRuntimeOutputAllocationFreeStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireRuntimeOutputAllocationFreeStepParams({ token: 0, output: false });
const llamaSessionRequiredRuntimeOutputAllocationFreeStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_runtime_output_allocation_free_step_params({ token: 0, output: false });
const llamaSessionAcceptsNoReadbackStepParams: boolean = tinyLlamaSession.acceptsNoReadbackStepParams({ token: 0, output: false });
const llamaSessionAcceptsNoReadbackStepParamsAlias: boolean = tinyLlamaSession.accepts_no_readback_step_params({ token: 0, output: false });
const llamaSessionRequiredNoReadbackStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireNoReadbackStepParams({ token: 0, output: false });
const llamaSessionRequiredNoReadbackStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_no_readback_step_params({ token: 0, output: false });
const llamaSessionRequiredReadbackFreeStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireReadbackFreeStepParams({ token: 0, output: false });
const llamaSessionAcceptsReadbackFreeStepParamsAlias: boolean = tinyLlamaSession.accepts_readback_free_step_params({ token: 0, output: false });
const llamaSessionRequiredReadbackFreeStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_readback_free_step_params({ token: 0, output: false });
const llamaSessionAcceptsHotStepParams: boolean = tinyLlamaSession.acceptsHotStepParams({ token: 0, output: false });
const llamaSessionAcceptsHotStepParamsAlias: boolean = tinyLlamaSession.accepts_hot_step_params({ token: 0, output: false });
const llamaSessionRequiredHotStepParams: SessionStepParamsCompatibility = tinyLlamaSession.requireHotStepParams({ token: 0, output: false });
const llamaSessionRequiredHotStepParamsAlias: SessionStepParamsCompatibility = tinyLlamaSession.require_hot_step_params({ token: 0, output: false });
const llamaSessionExecutionPlan: SessionExecutionPlan = tinyLlamaSession.executionPlan({ token: 0, output: false });
const llamaSessionExecutionPlanAlias: SessionExecutionPlan = tinyLlamaSession.execution_plan({ token: 0, output: false });
const llamaSessionRequiredExecutionPlan: SessionExecutionPlan = tinyLlamaSession.requireExecutionPlan({ token: 0, output: false });
const llamaSessionRequiredExecutionPlanAlias: SessionExecutionPlan = tinyLlamaSession.require_execution_plan({ token: 0, output: false });
const llamaSessionHotPathPlan: SessionExecutionPlan = tinyLlamaSession.hotPathPlan({ token: 0, output: false });
const llamaSessionHotPathPlanAlias: SessionExecutionPlan = tinyLlamaSession.hot_path_plan({ token: 0, output: false });
const llamaSessionExecutionPlanContract: SessionStepContract | LlamaSessionStepContract = llamaSessionExecutionPlan.contract;
const llamaSessionExecutionPlanCompatibility: SessionStepParamsCompatibility = llamaSessionExecutionPlan.compatibility;
const llamaSessionExecutionPlanSignature: string = llamaSessionExecutionPlan.signature;
const llamaSessionExecutionPlanOutputTarget: SessionStepParamsOutputTarget = llamaSessionExecutionPlan.outputTarget;
const llamaSessionExecutionPlanInputOwnership: SessionStepParamsInputOwnership = llamaSessionExecutionPlan.inputOwnership;
const llamaSessionExecutionPlanOutputOwnership: SessionStepParamsOutputOwnership = llamaSessionExecutionPlan.outputOwnership;
const llamaSessionExecutionPlanOutputReturnOwnership: SessionStepParamsOutputReturnOwnership = llamaSessionExecutionPlan.outputReturnOwnership;
const llamaSessionExecutionPlanInputElementType: SessionStepParamsElementType = llamaSessionExecutionPlan.inputElementType;
const llamaSessionExecutionPlanOutputElementType: SessionStepParamsElementType = llamaSessionExecutionPlan.outputElementType;
const llamaSessionExecutionPlanInputElementLength: number = llamaSessionExecutionPlan.inputElementLength;
const llamaSessionExecutionPlanOutputElementLength: number = llamaSessionExecutionPlan.outputElementLength;
const llamaSessionExecutionPlanRejectionCode: SessionStepParamsDiagnosticCode | null = llamaSessionExecutionPlan.rejectionCode;
const llamaSessionHotPathPlanHotPath: boolean = llamaSessionHotPathPlan.hotPath;
const llamaSessionHotPathPlanSignature: string = llamaSessionHotPathPlan.signature;
const llamaSessionMatchesStepParamsSignature: boolean = tinyLlamaSession.matchesStepParamsSignature({ token: 0, output: false }, llamaSessionStepParamsSignature);
const llamaSessionMatchesStepParamsSignatureAlias: boolean = tinyLlamaSession.matches_step_params_signature({ token: 0, output: false }, llamaSessionStepParamsSignature);
const llamaSessionMatchesStepParamsCompatibility: boolean = tinyLlamaSession.matchesStepParamsCompatibility({ token: 0, output: false }, llamaSessionStepParamsCompatibility);
const llamaSessionMatchesStepParamsCompatibilityAlias: boolean = tinyLlamaSession.matches_step_params_compatibility({ token: 0, output: false }, llamaSessionStepParamsCompatibility);
const llamaSessionRejectedMissingStepParams: SessionStepParamsCompatibility = tinyLlamaSession.stepParamsCompatibility();
const llamaSessionRejectedEmptyStepParams: SessionStepParamsCompatibility = tinyLlamaSession.stepParamsCompatibility({});
const llamaSessionRejectsEmptyStepParams: boolean = tinyLlamaSession.acceptsStepParams({});
// @ts-expect-error LLaMA Program inspection snapshots are immutable evidence records.
llamaProgramInspection.vocabSize = 9;
// @ts-expect-error LLaMA Program requirements snapshots are immutable evidence records.
llamaProgramRequirements.logitsLen = 9;
// @ts-expect-error LLaMA Program buffer sizing snapshots are immutable evidence records.
llamaProgramBufferSizing.outputLen = 9;
// @ts-expect-error LLaMA Session buffer sizing snapshots are immutable evidence records.
llamaSessionBufferSizing.outputLen = 9;
// @ts-expect-error LLaMA Program capability snapshots are immutable evidence records.
llamaProgramCapabilities.canExecute = false;
// @ts-expect-error LLaMA KV-cache requirements snapshots are immutable evidence records.
llamaKvCacheRequirements.layers = 0;
// @ts-expect-error LLaMA KV-cache layout slot arrays are immutable evidence records.
llamaKvCacheLayout.k[0] = llamaKvCacheLayout.v[0];
// @ts-expect-error LLaMA Session KV-cache layout slot arrays are immutable evidence records.
llamaSessionKvCacheLayout.v[0] = llamaSessionKvCacheLayout.k[0];
const llamaStepTensor = tinyLlamaSession.stepTensor(0, { shape: [8] as const });
const llamaStepTensorShape: readonly [8] = llamaStepTensor.shape;
const llamaCarrier = zeros([8] as const);
const llamaCarrierTensor = tinyLlamaSession.stepTensor(0, { output: llamaCarrier });
const llamaCarrierShape: readonly [8] = llamaCarrierTensor.shape;
const llamaExecuteTensor = tinyLlamaSession.executeTensor({ token: 0, shape: [8] as const });
const llamaExecuteTensorShape: readonly [8] = llamaExecuteTensor.shape;
const llamaExecuteTensorAlias = tinyLlamaSession.execute_tensor({ token: 0, shape: [8] as const });
const llamaExecuteTensorAliasShape: readonly [8] = llamaExecuteTensorAlias.shape;
const llamaWindowExecuteTensor = tinyLlamaSession.executeTensor({ tokens: [0, 1], tokensLen: 1, shape: [8] as const });
const llamaWindowExecuteTensorShape: readonly [8] = llamaWindowExecuteTensor.shape;
const llamaWindowExecuteTensorAlias = tinyLlamaSession.execute_tensor({ tokens: [0, 1], tokensLen: 1, shape: [8] as const });
const llamaWindowExecuteTensorAliasShape: readonly [8] = llamaWindowExecuteTensorAlias.shape;
const llamaPrefillTensor = tinyLlamaSession.prefillTensor([0, 1], { tokensLen: 1, shape: [8] as const });
const llamaPrefillTensorShape: readonly [8] = llamaPrefillTensor.shape;
const llamaPrefillTensorAlias = tinyLlamaSession.prefill_tensor([0, 1], { tokensLen: 1, shape: [8] as const });
const llamaPrefillTensorAliasShape: readonly [8] = llamaPrefillTensorAlias.shape;
const llamaReadOutputTensor = tinyLlamaSession.readOutputTensor({ shape: [8] as const });
const llamaReadOutputShape: readonly [8] = llamaReadOutputTensor.shape;
const llamaReadOutputCarrier = zeros([8] as const);
const llamaCarrierReadOutputTensor = tinyLlamaSession.readOutputTensor({ output: llamaReadOutputCarrier });
const llamaCarrierReadOutputShape: readonly [8] = llamaCarrierReadOutputTensor.shape;
const llamaReadOutputArrayCarrier = new Float32Array(8);
const llamaArrayCarrierReadOutputTensor = tinyLlamaSession.readOutputTensor({ output: llamaReadOutputArrayCarrier, shape: [8] as const });
const llamaArrayCarrierReadOutputShape: readonly [8] = llamaArrayCarrierReadOutputTensor.shape;
const llamaReadOutputLargeArrayCarrier = new Float32Array(16);
const llamaLargeArrayCarrierReadOutputTensor = tinyLlamaSession.readOutputTensor({
  output: llamaReadOutputLargeArrayCarrier,
  shape: [8] as const,
  length: 8,
});
const llamaLargeArrayCarrierReadOutputShape: readonly [8] = llamaLargeArrayCarrierReadOutputTensor.shape;
const llamaStepParams: LlamaStepParams = { token: 0 };
const llamaExecuteParamsToken: LlamaExecuteParams = { token: 0, output: false };
const llamaExecuteParamsWindow: LlamaExecuteParams = { tokens: [0, 1], tokensLen: 1 };
const llamaExecuteIntoParamsToken: LlamaExecuteIntoParams = { token: 0 };
const llamaExecuteIntoParamsWindow: LlamaExecuteIntoParams = { tokens: [0, 1], tokensLen: 1 };
const llamaExecuteInto: Float32Array = tinyLlamaSession.executeInto(new Float32Array(8), llamaExecuteIntoParamsToken);
const llamaExecuteWindowInto: Float32Array = tinyLlamaSession.executeInto(new Float32Array(8), llamaExecuteIntoParamsWindow);
const llamaExecuteIntoAlias: Float32Array = tinyLlamaSession.execute_into(new Float32Array(8), llamaExecuteIntoParamsToken);
const llamaExecuteWindowIntoAlias: Float32Array = tinyLlamaSession.execute_into(new Float32Array(8), llamaExecuteIntoParamsWindow);
const llamaExecuteTokensAlias: Float32Array | undefined = tinyLlamaSession.execute_tokens([0, 1], { tokensLen: 1 });
const llamaPrefillIntoAlias: Float32Array = tinyLlamaSession.prefill_into(new Float32Array(8), [0, 1], { tokensLen: 1 });
const llamaAdvanceTokensAlias: void = tinyLlamaSession.advance_tokens([0, 1], { tokensLen: 1 });
// @ts-expect-error LLaMA execute params must choose either token or tokens, not both.
const badLlamaExecuteParams: LlamaExecuteParams = { token: 0, tokens: [0] };
// @ts-expect-error LLaMA step params describe one scalar token, not a token window.
const badLlamaStepParams: LlamaStepParams = { tokens: [0] };
// @ts-expect-error LLaMA executeInto params take the output buffer as the first argument.
const badLlamaExecuteIntoParams: LlamaExecuteIntoParams = { token: 0, output: new Float32Array(8) };
const llamaArgmax: TokenArgmaxResult = tinyLlamaSession.stepArgmax(0);
const llamaSample: TokenSampleResult = tinyLlamaSession.stepSample(0, { topK: 1, seed: 1 });
const llamaGeneratedArgmax: TokenGenerateArgmaxResult = tinyLlamaSession.generateTokensArgmax([0], 1);
const llamaGeneratedSample: TokenGenerateSampleResult = tinyLlamaSession.generateTokensSample([0], 1, { topK: 1 });
const llamaArgmaxAlias: TokenArgmaxResult = tinyLlamaSession.step_argmax(0);
const llamaSampleAlias: TokenSampleResult = tinyLlamaSession.step_sample(0, { topK: 1, seed: 1 });
const llamaExecuteArgmaxAlias: TokenArgmaxResult = tinyLlamaSession.execute_tokens_argmax([0, 1], { tokensLen: 1 });
const llamaExecuteSampleAlias: TokenSampleResult = tinyLlamaSession.execute_tokens_sample([0, 1], { tokensLen: 1, topK: 1 });
const llamaGeneratedArgmaxAlias: TokenGenerateArgmaxResult = tinyLlamaSession.generate_tokens_argmax([0], 1);
const llamaGeneratedArgmaxIntoAlias: TokenGenerateArgmaxResult = tinyLlamaSession.generate_tokens_argmax_into([0], new Uint32Array(1));
const llamaGeneratedSampleAlias: TokenGenerateSampleResult = tinyLlamaSession.generate_tokens_sample([0], 1, { topK: 1 });
const llamaGeneratedSampleIntoAlias: TokenGenerateSampleResult = tinyLlamaSession.generate_tokens_sample_into([0], new Uint32Array(1), { topK: 1 });
const llamaArgmaxTokenAlias: TokenArgmaxResult = tinyLlamaSession.argmax_token();
const llamaSampleTokenAlias: TokenSampleResult = tinyLlamaSession.sample_token(undefined, { topK: 1 });
// @ts-expect-error Token selection results are immutable snapshots.
llamaArgmax.token = 1;
// @ts-expect-error Token sample results are immutable snapshots.
llamaSample.logit = 0;
// @ts-expect-error Token generation result metadata is immutable.
llamaGeneratedArgmax.lastToken = 1;
// @ts-expect-error Token generation result buffers are stable top-level bindings.
llamaGeneratedSample.tokens = new Uint32Array(1);
tinyLlamaSession.free();
tinyLlamaProgram.free();

const checkpointModel = nn.linear(2, 1);
const publicOptimNamespace: OptimNamespace = optim;
const checkpointOptimizer = optim.adam(checkpointModel);
const checkpointAdamW = optim.adamW(checkpointModel, { lr: 0.001, weightDecay: 0.01 });
const moduleStateSnapshot: ModuleStateSnapshot = checkpointModel.stateDict();
const moduleStateSnapshotSignature: string = moduleStateSnapshot.weight.signature;
const checkpointModelParameterNames: readonly string[] = checkpointModel.parameterNames();
const prefixedModuleStateSnapshot: ModuleStateSnapshot = checkpointModel.stateDict("head");
const prefixedModuleStateSnapshotSignature: string = prefixedModuleStateSnapshot["head.weight"].signature;
const optimizerStateSnapshot: OptimizerStateSnapshot<"adam"> = checkpointOptimizer.stateDict();
const optimizerStateSnapshotAlias: OptimizerStateSnapshot<"adam"> = checkpointOptimizer.state_dict();
const optimizerStateSnapshotSignature: string = optimizerStateSnapshot.signature;
const adamOptimizerKind: "adam" = checkpointOptimizer.kind;
const adamWOptimizerKind: "adamw" = checkpointAdamW.kind;
const adamOptimizerTyped: Optimizer<"adam"> = checkpointOptimizer;
const adamWStateSnapshot: OptimizerStateSnapshot<"adamw"> = checkpointAdamW.stateDict();
const adamWStateSnapshotSignature: string = adamWStateSnapshot.signature;
const optimizerConfigSnapshot: OptimizerConfigSnapshot = checkpointOptimizer.config();
const optimizerConfigSnapshotSignature: string = optimizerConfigSnapshot.signature;
const optimizerConfigStep: number | undefined = optimizerConfigSnapshot.step;
const adamWConfigSnapshot: OptimizerConfigSnapshot<"adamw"> = optim.config(checkpointAdamW);
const adamWConfigSnapshotSignature: string = adamWConfigSnapshot.signature;
const optimizerConfigSnapshotPredicate: boolean = optim.isOptimizerConfigSnapshot(optimizerConfigSnapshot);
const optimizerConfigSnapshotRequire: OptimizerConfigSnapshot = optim.requireOptimizerConfigSnapshot(optimizerConfigSnapshot);
const optimizerConfigSnapshotAssert: OptimizerConfigSnapshot = optim.assert_optimizer_config_snapshot(optimizerConfigSnapshot);
const optimizerConfigSnapshotSignatureFromNamespace: string = optim.optimizerConfigSnapshotSignature(optimizerConfigSnapshot);
const optimizerConfigSnapshotSignatureMatch: boolean = optim.matchesOptimizerConfigSnapshotSignature(
  optimizerConfigSnapshot,
  optimizerConfigSnapshot.signature,
);
const groupedOptimizer = optim.sgd([
  { params: [checkpointModel.parameters()[0]], lr: 0.01 },
  { params: [checkpointModel.parameters()[1]], lr: 0.001, weight_decay: 0 },
]);
const groupedOptimizerKind: "sgd" = groupedOptimizer.kind;
const groupedOptimizerConfig: OptimizerConfigSnapshot<"sgd"> = groupedOptimizer.config();
const groupedOptimizerDefaults: OptimizerConfigSnapshot<"sgd"> = groupedOptimizer.defaults;
const groupedOptimizerParamGroups: readonly OptimizerParamGroupSnapshot[] = groupedOptimizerConfig.paramGroups;
const groupedOptimizerParamGroupsConfigAlias: readonly OptimizerParamGroupSnapshot[] = groupedOptimizerConfig.param_groups;
const groupedOptimizerParamGroupsDirect: readonly OptimizerParamGroupSnapshot[] = groupedOptimizer.paramGroups;
const groupedOptimizerParamGroupsAlias: readonly OptimizerParamGroupSnapshot[] = groupedOptimizer.param_groups;
const groupedOptimizerFirstGroupLr: number = groupedOptimizerParamGroups[0].lr;
const groupedOptimizerSecondGroupWeightDecay: number = groupedOptimizerParamGroups[1].weight_decay;
const groupedOptimizerWithAddedGroup: typeof groupedOptimizer = groupedOptimizer.addParamGroup({ params: [checkpointModel.parameters()[0]], lr: 0.0001 });
const groupedOptimizerWithAddedGroupAlias: typeof groupedOptimizer = optim.add_param_group(groupedOptimizer, { params: [checkpointModel.parameters()[1]], lr: 0.0002 });
const optimizerConfigFromNamespace: OptimizerConfigSnapshot = optim.config(checkpointOptimizer);
const optimizerWithUpdatedLr: typeof checkpointOptimizer = checkpointOptimizer.setLearningRate(0.0005);
const optimizerWithSnakeLr: typeof checkpointOptimizer = optim.set_lr(checkpointOptimizer, 0.0004);
const optimizerLearningRate: number = checkpointOptimizer.getLearningRate();
const optimizerLearningRateAlias: number = checkpointOptimizer.get_lr();
const optimizerLearningRateFromNamespace: number = optim.getLearningRate(checkpointOptimizer);
const optimizerLearningRateFromNamespaceAlias: number = optim.get_lr(checkpointOptimizer);
const stepScheduler: LRScheduler<"step-lr", "adam"> = optim.stepLR(checkpointOptimizer, { stepSize: 2, gamma: 0.5 });
const stepSchedulerAlias: LRScheduler<"step-lr", "adam"> = optim.step_lr(checkpointOptimizer, { step_size: 2, gamma: 0.5 });
const stepSchedulerClass: LRScheduler<"step-lr", "adam"> = new optim.StepLR(checkpointOptimizer, { stepSize: 2, gamma: 0.5 });
const exponentialScheduler: LRScheduler<"exponential-lr", "adam"> = optim.exponentialLR(checkpointOptimizer, { gamma: 0.95 });
const exponentialSchedulerAlias: LRScheduler<"exponential-lr", "adam"> = optim.exponential_lr(checkpointOptimizer, { gamma: 0.95 });
const exponentialSchedulerClass: LRScheduler<"exponential-lr", "adam"> = new optim.ExponentialLR(checkpointOptimizer, { gamma: 0.95 });
const cosineScheduler: LRScheduler<"cosine-annealing-lr", "adam"> = optim.cosineAnnealingLR(checkpointOptimizer, { tMax: 4, etaMin: 0 });
const cosineSchedulerAlias: LRScheduler<"cosine-annealing-lr", "adam"> = optim.cosine_annealing_lr(checkpointOptimizer, { t_max: 4, eta_min: 0 });
const cosineSchedulerClass: LRScheduler<"cosine-annealing-lr", "adam"> = new optim.CosineAnnealingLR(checkpointOptimizer, { tMax: 4, etaMin: 0 });
const plateauScheduler: LRScheduler<"reduce-lr-on-plateau", "adam"> = optim.reduceLROnPlateau(checkpointOptimizer, { mode: "min", factor: 0.5, patience: 1 });
const plateauSchedulerAlias: LRScheduler<"reduce-lr-on-plateau", "adam"> = optim.reduce_lr_on_plateau(checkpointOptimizer, { threshold_mode: "abs", min_lr: 0 });
const plateauSchedulerClass: LRScheduler<"reduce-lr-on-plateau", "adam"> = new optim.ReduceLROnPlateau(checkpointOptimizer, { cooldown: 1 });
const nestedStepScheduler: LRScheduler<"step-lr", "adam"> = new optim.lr_scheduler.StepLR(checkpointOptimizer, { step_size: 2, gamma: 0.5 });
const nestedStepSchedulerFactory: LRScheduler<"step-lr", "adam"> = optim.lrScheduler.stepLR(checkpointOptimizer, { stepSize: 2, gamma: 0.5 });
const nestedExponentialScheduler: LRScheduler<"exponential-lr", "adam"> = new optim.lrScheduler.ExponentialLR(checkpointOptimizer, { gamma: 0.95 });
const nestedExponentialSchedulerFactory: LRScheduler<"exponential-lr", "adam"> = optim.lr_scheduler.exponential_lr(checkpointOptimizer, { gamma: 0.95 });
const nestedCosineScheduler: LRScheduler<"cosine-annealing-lr", "adam"> = new optim.lrScheduler.CosineAnnealingLR(checkpointOptimizer, { tMax: 4, etaMin: 0 });
const nestedCosineSchedulerFactory: LRScheduler<"cosine-annealing-lr", "adam"> = optim.lr_scheduler.cosineAnnealingLR(checkpointOptimizer, { t_max: 4, eta_min: 0 });
const nestedPlateauScheduler: LRScheduler<"reduce-lr-on-plateau", "adam"> = new optim.lrScheduler.ReduceLROnPlateau(checkpointOptimizer, { mode: "max" });
const nestedPlateauSchedulerFactory: LRScheduler<"reduce-lr-on-plateau", "adam"> = optim.lr_scheduler.reduceLROnPlateau(checkpointOptimizer, { patience: 2 });
const schedulerLr: number = stepScheduler.step();
const schedulerLastLr: number = stepScheduler.get_last_lr();
const schedulerLastLrProperty: number = stepScheduler.lastLr;
const schedulerLastLrAliasProperty: number = stepScheduler.last_lr;
const schedulerBaseLrAliasProperty: number = stepScheduler.base_lr;
const schedulerStepSizeAliasProperty: number = stepScheduler.step_size;
const schedulerStateSnapshot: LRSchedulerStateSnapshot<"step-lr", "adam"> = stepScheduler.state_dict();
const schedulerStateSnapshotSignature: string = schedulerStateSnapshot.signature;
const schedulerStateKind: "step-lr" = schedulerStateSnapshot.kind;
const schedulerStateOptimizerKind: "adam" = schedulerStateSnapshot.optimizerKind;
const schedulerStateStepSize: number | undefined = schedulerStateSnapshot.stepSize;
const schedulerStateBaseLrAlias: number = schedulerStateSnapshot.base_lr;
const schedulerStateLastLrAlias: number = schedulerStateSnapshot.last_lr;
const schedulerStateStepSizeAlias: number | undefined = schedulerStateSnapshot.step_size;
const cosineSchedulerState: LRSchedulerStateSnapshot<"cosine-annealing-lr", "adam"> = cosineScheduler.state_dict();
const cosineSchedulerTMaxAlias: number | undefined = cosineSchedulerState.t_max;
const cosineSchedulerEtaMinAlias: number | undefined = cosineSchedulerState.eta_min;
const plateauSchedulerLr: number = plateauScheduler.step(1);
const plateauSchedulerState: LRSchedulerStateSnapshot<"reduce-lr-on-plateau", "adam"> = plateauScheduler.state_dict();
const plateauSchedulerMode: "min" | "max" | undefined = plateauSchedulerState.mode;
const plateauSchedulerBadEpochsAlias: number | undefined = plateauSchedulerState.bad_epochs;
const schedulerStateSnapshotPredicate: boolean = optim.isLRSchedulerStateSnapshot(schedulerStateSnapshot);
const schedulerStateSnapshotRequire: LRSchedulerStateSnapshot = optim.requireLRSchedulerStateSnapshot(schedulerStateSnapshot);
const schedulerStateSnapshotAssert: LRSchedulerStateSnapshot = optim.assert_lr_scheduler_state_snapshot(schedulerStateSnapshot);
const schedulerStateSnapshotSignatureFromNamespace: string = optim.lrSchedulerStateSnapshotSignature(schedulerStateSnapshot);
const schedulerStateSnapshotSignatureMatch: boolean = optim.matchesLRSchedulerStateSnapshotSignature(
  schedulerStateSnapshot,
  schedulerStateSnapshot.signature,
);
const schedulerStateSnakeDict: LRSchedulerStateDict<"step-lr"> = {
  kind: schedulerStateSnapshot.kind,
  step: schedulerStateSnapshot.step,
  base_lr: schedulerStateSnapshot.base_lr,
  last_lr: schedulerStateSnapshot.last_lr,
  gamma: schedulerStateSnapshot.gamma,
  step_size: schedulerStateSnapshot.step_size,
};
stepScheduler.load_state_dict(schedulerStateSnakeDict, { strict: true });
checkpointModel.loadStateDict(moduleStateSnapshot, { strict: true });
checkpointModel.loadStateDict(prefixedModuleStateSnapshot, { strict: true, prefix: "head" });
checkpointModel.loadStateDict(prefixedModuleStateSnapshot, { strict: true, prefix: "head", validateOnly: true });
checkpointOptimizer.loadStateDict(optimizerStateSnapshot, { strict: true });
checkpointOptimizer.loadStateDict(optimizerStateSnapshot, { strict: true, validateOnly: true });
checkpointOptimizer.load_state_dict(optimizerStateSnapshotAlias, { strict: true });
checkpointOptimizer.zero_grad();
checkpointOptimizer.zero_grad({ set_to_none: true });
const optimizerStateFromNamespace: OptimizerStateSnapshot<"adam"> = optim.stateDict(checkpointOptimizer);
const optimizerStateFromNamespaceAlias: OptimizerStateSnapshot<"adam"> = optim.state_dict(checkpointOptimizer);
const optimizerStateSnapshotPredicate: boolean = optim.isOptimizerStateSnapshot(optimizerStateSnapshot);
const optimizerStateSnapshotRequire: OptimizerStateSnapshot = optim.requireOptimizerStateSnapshot(optimizerStateSnapshot);
const optimizerStateSnapshotAssert: OptimizerStateSnapshot = optim.assert_optimizer_state_snapshot(optimizerStateSnapshot);
const optimizerStateSnapshotSignatureFromNamespace: string = optim.optimizerStateSnapshotSignature(optimizerStateSnapshot);
const optimizerStateSnapshotSignatureMatch: boolean = optim.matchesOptimizerStateSnapshotSignature(
  optimizerStateSnapshot,
  optimizerStateSnapshot.signature,
);
const optimizerLoadedFromNamespace: typeof checkpointOptimizer = optim.loadStateDict(checkpointOptimizer, optimizerStateSnapshot, {
  strict: true,
  validateOnly: true,
});
const optimizerLoadedFromNamespaceAlias: typeof checkpointOptimizer = optim.load_state_dict(checkpointOptimizer, optimizerStateSnapshotAlias, {
  strict: true,
  validateOnly: true,
});
const nnValidateOnlyStateTarget: typeof checkpointModel = nn.loadStateDict(checkpointModel, prefixedModuleStateSnapshot, {
  strict: true,
  prefix: "head",
  validateOnly: true,
});
optim.zeroGrad(checkpointModel, { setToNone: true });
optim.zero_grad(checkpointModel);
optim.zero_grad(checkpointModel, { set_to_none: true });
train.zeroGrad(checkpointModel, { setToNone: true });
const publicLossNamespace: PublicLossNamespace = loss;
const rawLossInput: TensorData = [1, 2] as const;
const rawMseLoss: number = loss.mse(rawLossInput, [1, 3] as const);
const rawMseSumLoss: number = loss.mse(rawLossInput, [1, 3] as const, { reduction: "sum" });
const rawMeanSquaredErrorLoss: number = loss.meanSquaredError(rawLossInput, [1, 3] as const);
const rawMeanAbsoluteErrorLoss: number = loss.meanAbsoluteError(rawLossInput, [1, 3] as const);
const rawL1Loss: number = loss.l1(rawLossInput, [1, 3] as const);
const rawL1SumLoss: number = loss.l1(rawLossInput, [1, 3] as const, { reduction: "sum" });
const rawHuberLoss: number = loss.huber(rawLossInput, [1, 3] as const, { delta: 1 });
const rawSmoothL1Loss: number = loss.smoothL1(rawLossInput, [1, 3] as const, { beta: 1 });
const rawSmoothL1LossAlias: number = loss.smooth_l1(rawLossInput, [1, 3] as const, { beta: 1 });
const rawBceLoss: number = loss.bce([0.25, 0.75] as const, [0, 1] as const);
const rawBinaryCrossEntropyLoss: number = loss.binaryCrossEntropy([0.25, 0.75] as const, [0, 1] as const);
const rawBinaryCrossEntropyLossAlias: number = loss.binary_cross_entropy([0.25, 0.75] as const, [0, 1] as const);
const rawBceWithLogitsLoss: number = loss.bceWithLogits([0, 2] as const, [0, 1] as const);
const rawBceWithLogitsLossSnake: number = loss.bce_with_logits([0, 2] as const, [0, 1] as const);
const rawBinaryCrossEntropyWithLogitsLoss: number = loss.binaryCrossEntropyWithLogits([0, 2] as const, [0, 1] as const);
const rawBinaryCrossEntropyWithLogitsLossAlias: number = loss.binary_cross_entropy_with_logits([0, 2] as const, [0, 1] as const);
const rawBinaryCrossEntropyWithLogitsSumLoss: number = loss.binary_cross_entropy_with_logits([0, 2] as const, [0, 1] as const, { reduction: "sum" });
const rawCrossEntropyLoss: number = loss.crossEntropy([1, 2, 3] as const, [2], { classes: 3 });
const rawCrossEntropyLossSnake: number = loss.cross_entropy([1, 2, 3] as const, [2], { numClasses: 3 });
const rawCrossEntropySumLoss: number = loss.cross_entropy([1, 2, 3] as const, [2], { numClasses: 3, reduction: "sum" });
const publicNnNamespace: NnNamespace = nn;
const nnFunctional: NnFunctionalNamespace = nn.functional;
const nnFunctionalLossSubset: Omit<LossNamespace, "mse_loss" | "l1_loss" | "huber_loss" | "smooth_l1_loss"> = nn.functional;
const nnFunctionalAlias: NnFunctionalNamespace = nn.F;
const rootFunctionalAlias: NnFunctionalNamespace = F;
const nnLossConstructorName: NnLossConstructorName = "MSELoss";
const nnLossConstructors: NnLossConstructors = nn;
const functionalReluTensor: Tensor<readonly [1, 3]> = nnFunctional.relu(tensor([[1, 2, 3]], [1, 3] as const));
const functionalReluInplaceFalseTensor: Tensor<readonly [1, 3]> = nnFunctional.relu(tensor([[1, 2, 3]], [1, 3] as const), false);
const functionalGeluTensor: Tensor<readonly [1, 3]> = nnFunctional.gelu(tensor([[1, 2, 3]], [1, 3] as const));
const functionalSiluTensor: Tensor<readonly [1, 3]> = nnFunctional.silu(tensor([[1, 2, 3]], [1, 3] as const));
const functionalSigmoidTensor: Tensor<readonly [1, 3]> = nnFunctional.sigmoid(tensor([[1, 2, 3]], [1, 3] as const));
const functionalTanhTensor: Tensor<readonly [1, 3]> = nnFunctional.tanh(tensor([[1, 2, 3]], [1, 3] as const));
const functionalSoftmaxTensor: Tensor<readonly [1, 3]> = nnFunctional.softmax(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalSoftmaxDimTensor: Tensor<readonly [1, 3]> = nnFunctional.softmax_dim(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalSoftmaxCamelDimTensor: Tensor<readonly [1, 3]> = nnFunctional.softmaxDim(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalLogSoftmaxTensor: Tensor<readonly [1, 3]> = nnFunctional.log_softmax(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalLogSoftmaxDimTensor: Tensor<readonly [1, 3]> = nnFunctional.log_softmax_dim(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalLogSoftmaxCamelDimTensor: Tensor<readonly [1, 3]> = nnFunctional.logSoftmaxDim(tensor([[1, 2, 3]], [1, 3] as const), 1);
const functionalDropoutTensor: Tensor<readonly [1, 3]> = nnFunctional.dropout(tensor([[1, 2, 3]], [1, 3] as const), 0.25, { training: false });
const functionalDropoutBooleanTensor: Tensor<readonly [1, 3]> = nnFunctional.dropout(tensor([[1, 2, 3]], [1, 3] as const), 0.25, false);
const functionalDropoutInplaceFalseTensor: Tensor<readonly [1, 3]> = nnFunctional.dropout(tensor([[1, 2, 3]], [1, 3] as const), 0.25, false, false);
const functionalFlattenTensor: Tensor<readonly [3]> = nnFunctional.flatten(tensor([[1, 2, 3]], [1, 3] as const));
const functionalFlattenRangeTensor: Tensor<readonly [1, 3]> = nnFunctional.flatten(tensor([[1, 2, 3]], [1, 3] as const), 1, -1);
const functionalLinear1dTensor: Tensor<readonly [2]> = nnFunctional.linear(tensor([1, 2, 3], [3] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const), tensor([0, 0], [2] as const));
const functionalLinear2dTensor: Tensor<readonly [1, 2]> = nnFunctional.linear(tensor([[1, 2, 3]], [1, 3] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const), tensor([0, 0], [2] as const));
const functionalLinear3dTensor: Tensor<readonly [1, 2, 2]> = nnFunctional.linear(tensor([1, 2, 3, 4, 5, 6], [1, 2, 3] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const), tensor([0, 0], [2] as const));
const functionalLinear4dTensor: Tensor<readonly [1, 1, 2, 2]> = nnFunctional.linear(tensor([1, 2, 3, 4, 5, 6], [1, 1, 2, 3] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const), tensor([0, 0], [2] as const));
const functionalNormalizeTensor: Tensor<readonly [1, 3]> = nnFunctional.normalize(tensor([[1, 2, 3]], [1, 3] as const), 2, 1);
const functionalOneHotTensor: Tensor<readonly [2, 3]> = nnFunctional.one_hot(tensor([0, 2], [2] as const), 3);
const functionalOneHotAliasTensor: Tensor<readonly [number, 3]> = nnFunctional.oneHot([0, 2] as const, 3);
const functionalOneHotGridTensor: Tensor<readonly [1, 2, 3]> = nnFunctional.one_hot(tensor([0, 2], [1, 2] as const), 3);
const functionalEmbeddingTensor: Tensor<readonly [2, 3]> = nnFunctional.embedding(tensor([0, 1], [2] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const));
const functionalEmbeddingGridTensor: Tensor<readonly [1, 2, 3]> = nnFunctional.embedding(tensor([0, 1], [1, 2] as const), tensor([[1, 0, 0], [0, 1, 0]], [2, 3] as const));
const functionalLayerNormTensor: Tensor<readonly [1, 3]> = nnFunctional.layer_norm(tensor([[1, 2, 3]], [1, 3] as const), [3] as const);
const functionalLayerNormPositionalTensor: Tensor<readonly [1, 3]> = nnFunctional.layer_norm(tensor([[1, 2, 3]], [1, 3] as const), [3] as const, tensor([1, 1, 1], [3] as const), tensor([0, 0, 0], [3] as const), 1e-5);
const functionalRmsNormTensor: Tensor<readonly [1, 3]> = nnFunctional.rmsNorm(tensor([[1, 2, 3]], [1, 3] as const), 3);
const functionalRmsNormPositionalTensor: Tensor<readonly [1, 3]> = nnFunctional.rms_norm(tensor([[1, 2, 3]], [1, 3] as const), 3, tensor([1, 1, 1], [3] as const), 1e-5);
const functionalBatchNormTensor: Tensor<readonly [2, 3]> = nnFunctional.batchNorm1d(tensor([1, 2, 3, 3, 5, 9], [2, 3] as const), 3, { eps: 1e-5 });
const functionalBatchNormSnakeTensor: Tensor<readonly [2, 3]> = nnFunctional.batch_norm1d(tensor([1, 2, 3, 3, 5, 9], [2, 3] as const), [3] as const, { eps: 1e-5, momentum: 0.2 });
const rawFunctionalMseLoss: number = nnFunctional.mse(rawLossInput, [1, 3] as const);
const rawFunctionalMseLossAlias: number = nnFunctional.mse_loss(rawLossInput, [1, 3] as const);
const rawFunctionalMseSumLoss: number = nnFunctional.mse_loss(rawLossInput, [1, 3] as const, { reduction: "sum" });
const rawFunctionalL1Loss: number = nnFunctional.l1(rawLossInput, [1, 3] as const);
const rawFunctionalL1LossAlias: number = nnFunctional.l1_loss(rawLossInput, [1, 3] as const);
const rawFunctionalL1SumLoss: number = nnFunctional.l1_loss(rawLossInput, [1, 3] as const, { reduction: "sum" });
const rawFunctionalHuberLoss: number = nnFunctional.huber(rawLossInput, [1, 3] as const, { delta: 1 });
const rawFunctionalHuberLossAlias: number = nnFunctional.huber_loss(rawLossInput, [1, 3] as const, { delta: 1 });
const rawFunctionalSmoothL1Loss: number = nnFunctional.smooth_l1(rawLossInput, [1, 3] as const, { beta: 1 });
const rawFunctionalSmoothL1LossAlias: number = nnFunctional.smooth_l1_loss(rawLossInput, [1, 3] as const, { beta: 1 });
const rawFunctionalBceLoss: number = nnFunctional.binary_cross_entropy([0.25, 0.75] as const, [0, 1] as const);
const rawFunctionalBceWithLogitsLoss: number = nnFunctional.binary_cross_entropy_with_logits([0, 2] as const, [0, 1] as const);
const rawFunctionalCrossEntropyLossSnake: number = nnFunctional.cross_entropy([1, 2, 3] as const, [2], { numClasses: 3 });
const rawFunctionalCrossEntropySumLoss: number = nnFunctional.cross_entropy([1, 2, 3] as const, [2], { numClasses: 3, reduction: "sum" });
const rawFunctionalAliasCrossEntropyLoss: number = nnFunctionalAlias.crossEntropy([1, 2, 3] as const, [2], { classes: 3 });
const rootFunctionalAliasSoftmaxTensor: Tensor<readonly [1, 3]> = rootFunctionalAlias.softmax(tensor([[1, 2, 3]], [1, 3] as const), 1);
const rawNllLoss: number = loss.nllLoss([-3, -2, -0.5] as const, [2], { classes: 3 });
const rawNllLossSnake: number = loss.nll_loss([-3, -2, -0.5] as const, [2], { numClasses: 3 });
const rawNllSumLoss: number = loss.nll_loss([-3, -2, -0.5] as const, [2], { numClasses: 3, reduction: "sum" });
const classTargetBuffer: Uint32Array = loss.classTargets([2]);
const trainingLogits = tensor([[1, 2, 3]], [1, 3] as const).requires_grad_();
const tensorCrossEntropyLoss: Tensor<readonly [1]> = loss.crossEntropy(trainingLogits, classTargetBuffer);
const trainingLogProbabilities: Tensor<readonly [1, 3]> = trainingLogits.logSoftmax(1);
const functionalNllLossTensor: Tensor<readonly [1]> = nnFunctional.nll_loss(trainingLogProbabilities, classTargetBuffer, { classes: 3 });
const tensorNllLoss: Tensor<readonly [1]> = loss.negativeLogLikelihood(trainingLogProbabilities, classTargetBuffer);
const tensorNllLossSnake: Tensor<readonly [1]> = loss.negative_log_likelihood(trainingLogProbabilities, classTargetBuffer);
const classificationAccuracy: number = train.accuracy(trainingLogits, classTargetBuffer, { classes: 3 });
const classificationAccuracyCamel: number = train.classificationAccuracy([1, 2, 3] as const, [2], { numClasses: 3 });
const classificationAccuracySnake: number = train.classification_accuracy([1, 2, 3] as const, [1], { classes: 3 });
const classPredictions: Uint32Array = train.classPredictions(trainingLogits, { classes: 3 });
const classPredictionsSnake: Uint32Array = train.class_predictions([1, 2, 3] as const, { numClasses: 3 });
const predictedClasses: Uint32Array = train.predictClasses(trainingLogits, { classes: 3 });
const predictedClassesSnake: Uint32Array = train.predict_classes([1, 2, 3] as const, { numClasses: 3 });
const topKAccuracy: number = train.topKAccuracy(trainingLogits, classTargetBuffer, { classes: 3, k: 2 });
const topKAccuracySnake: number = train.top_k_accuracy([1, 2, 3] as const, [1], { numClasses: 3, top_k: 2 });
const confusionMatrix: readonly (readonly number[])[] = train.confusionMatrix(trainingLogits, classTargetBuffer, { classes: 3 });
const confusionMatrixSnake: readonly (readonly number[])[] = train.confusion_matrix([1, 2, 3] as const, [2], { numClasses: 3 });
const classificationReport: TrainClassificationReport = train.classificationReport(trainingLogits, classTargetBuffer, { classes: 3 });
const classificationReportSnake: TrainClassificationReport = train.classification_report([1, 2, 3] as const, [2], { numClasses: 3 });
const binaryAccuracy: number = train.binaryAccuracy([0.1, 0.9] as const, [0, 1] as const);
const binaryAccuracySnake: number = train.binary_accuracy(tensor([0.1, 0.9], [2] as const), tensor([0, 1], [2] as const), { threshold: 0.5 });
const binaryLogitsAccuracy: number = train.binaryLogitsAccuracy([-1, 2] as const, [0, 1] as const);
const binaryLogitsAccuracySnake: number = train.binary_logits_accuracy(tensor([-1, 2], [2] as const), tensor([0, 1], [2] as const));
const mseLossModule: MSELoss = loss.mseLoss();
const mseLossModuleAlias: MSELoss = loss.mse_loss();
const mseLossModuleClass: MSELoss = new loss.MSELoss();
const nnMseLossModuleClass: MSELoss = new nn.MSELoss();
const mseLossSumModule: MSELoss = loss.mseLoss({ reduction: "sum" });
const mseLossSumReduction: LossReduction = mseLossSumModule.reduction;
const nnMseLossSumModuleClass: MSELoss = new nn.MSELoss({ reduction: "sum" });
const nnMseLossSumReduction: LossReduction = nnMseLossSumModuleClass.reduction;
const mseLossModuleTensor: Tensor<readonly [1]> = mseLossModule.forward(checkpointModel.forward(tensor([1, 2], [2] as const)), tensor([0], [1] as const));
const mseLossModuleNumber: number = mseLossModuleAlias.call(rawLossInput, [1, 3] as const);
const mseLossModuleClassNumber: number = mseLossModuleClass.forward(rawLossInput, [1, 3] as const);
const nnMseLossModuleClassNumber: number = nnMseLossModuleClass.forward(rawLossInput, [1, 3] as const);
const mseLossModuleDunderNumber: number = mseLossModuleClass.__call__(rawLossInput, [1, 3] as const);
const mseLossSumModuleNumber: number = mseLossSumModule.forward(rawLossInput, [1, 3] as const);
const nnMseLossSumModuleClassNumber: number = nnMseLossSumModuleClass.forward(rawLossInput, [1, 3] as const);
const l1LossModule: L1Loss = loss.l1Loss();
const l1LossModuleAlias: L1Loss = loss.l1_loss();
const l1LossModuleClass: L1Loss = new loss.L1Loss();
const nnL1LossModuleClass: L1Loss = new nn.L1Loss();
const nnL1LossSumModuleClass: L1Loss = new nn.L1Loss({ reduction: "sum" });
const nnL1LossSumReduction: LossReduction = nnL1LossSumModuleClass.reduction;
const l1LossModuleTensor: Tensor<readonly [1]> = l1LossModule.forward(checkpointModel.forward(tensor([1, 2], [2] as const)), tensor([0], [1] as const));
const l1LossModuleNumber: number = l1LossModuleAlias.call(rawLossInput, [1, 3] as const);
const l1LossModuleClassNumber: number = l1LossModuleClass.forward(rawLossInput, [1, 3] as const);
const nnL1LossModuleClassNumber: number = nnL1LossModuleClass.forward(rawLossInput, [1, 3] as const);
const l1LossModuleDunderNumber: number = l1LossModuleClass.__call__(rawLossInput, [1, 3] as const);
const huberLossModule: HuberLoss = loss.huberLoss({ delta: 1 });
const huberLossModuleAlias: HuberLoss = loss.huber_loss({ delta: 1 });
const huberLossModuleClass: HuberLoss = new loss.HuberLoss({ delta: 1 });
const nnHuberLossModuleClass: HuberLoss = new nn.HuberLoss({ delta: 1 });
const nnHuberLossSumModuleClass: HuberLoss = new nn.HuberLoss({ delta: 1, reduction: "sum" });
const nnHuberLossSumReduction: LossReduction = nnHuberLossSumModuleClass.reduction;
const huberLossModuleTensor: Tensor<readonly [1]> = huberLossModule.forward(checkpointModel.forward(tensor([1, 2], [2] as const)), tensor([0], [1] as const));
const huberLossModuleNumber: number = huberLossModuleAlias.call(rawLossInput, [1, 3] as const);
const huberLossModuleClassNumber: number = huberLossModuleClass.forward(rawLossInput, [1, 3] as const);
const nnHuberLossModuleClassNumber: number = nnHuberLossModuleClass.forward(rawLossInput, [1, 3] as const);
const huberLossModuleDunderNumber: number = huberLossModuleClass.__call__(rawLossInput, [1, 3] as const);
const smoothL1LossModule: SmoothL1Loss = loss.smoothL1Loss({ beta: 1 });
const smoothL1LossModuleAlias: SmoothL1Loss = loss.smooth_l1_loss({ beta: 1 });
const smoothL1LossModuleClass: SmoothL1Loss = new loss.SmoothL1Loss({ beta: 1 });
const nnSmoothL1LossModuleClass: SmoothL1Loss = new nn.SmoothL1Loss({ beta: 1 });
const nnSmoothL1LossSumModuleClass: SmoothL1Loss = new nn.SmoothL1Loss({ beta: 1, reduction: "sum" });
const nnSmoothL1LossSumReduction: LossReduction = nnSmoothL1LossSumModuleClass.reduction;
const smoothL1LossModuleTensor: Tensor<readonly [1]> = smoothL1LossModule.forward(checkpointModel.forward(tensor([1, 2], [2] as const)), tensor([0], [1] as const));
const smoothL1LossModuleNumber: number = smoothL1LossModuleAlias.call(rawLossInput, [1, 3] as const);
const smoothL1LossModuleClassNumber: number = smoothL1LossModuleClass.forward(rawLossInput, [1, 3] as const);
const nnSmoothL1LossModuleClassNumber: number = nnSmoothL1LossModuleClass.forward(rawLossInput, [1, 3] as const);
const smoothL1LossModuleDunderNumber: number = smoothL1LossModuleClass.__call__(rawLossInput, [1, 3] as const);
const bceLossModule: BCELoss = loss.bceLoss();
const bceLossModuleAlias: BCELoss = loss.bce_loss({ eps: 1e-7 });
const bceLossModuleClass: BCELoss = new loss.BCELoss({ eps: 1e-7 });
const nnBceLossModuleClass: BCELoss = new nn.BCELoss({ eps: 1e-7 });
const nnBceLossSumModuleClass: BCELoss = new nn.BCELoss({ eps: 1e-7, reduction: "sum" });
const nnBceLossSumReduction: LossReduction = nnBceLossSumModuleClass.reduction;
const bceLossModuleTensor: Tensor<readonly [1]> = bceLossModule.forward(tensor([0.25, 0.75], [2] as const), tensor([0, 1], [2] as const));
const bceLossModuleNumber: number = bceLossModuleAlias.call([0.25, 0.75] as const, [0, 1] as const);
const bceLossModuleClassNumber: number = bceLossModuleClass.forward([0.25, 0.75] as const, [0, 1] as const);
const nnBceLossModuleClassNumber: number = nnBceLossModuleClass.forward([0.25, 0.75] as const, [0, 1] as const);
const bceLossModuleDunderNumber: number = bceLossModuleClass.__call__([0.25, 0.75] as const, [0, 1] as const);
const bceWithLogitsLossModule: BCEWithLogitsLoss = loss.bceWithLogitsLoss();
const bceWithLogitsLossModuleAlias: BCEWithLogitsLoss = loss.bce_with_logits_loss();
const bceWithLogitsLossModuleClass: BCEWithLogitsLoss = new loss.BCEWithLogitsLoss();
const nnBceWithLogitsLossModuleClass: BCEWithLogitsLoss = new nn.BCEWithLogitsLoss();
const nnBceWithLogitsLossSumModuleClass: BCEWithLogitsLoss = new nn.BCEWithLogitsLoss({ reduction: "sum" });
const nnBceWithLogitsLossSumReduction: LossReduction = nnBceWithLogitsLossSumModuleClass.reduction;
const bceWithLogitsLossModuleTensor: Tensor<readonly [1]> = bceWithLogitsLossModule.forward(tensor([0, 2], [2] as const), tensor([0, 1], [2] as const));
const bceWithLogitsLossModuleNumber: number = bceWithLogitsLossModuleAlias.call([0, 2] as const, [0, 1] as const);
const bceWithLogitsLossModuleClassNumber: number = bceWithLogitsLossModuleClass.forward([0, 2] as const, [0, 1] as const);
const nnBceWithLogitsLossModuleClassNumber: number = nnBceWithLogitsLossModuleClass.forward([0, 2] as const, [0, 1] as const);
const bceWithLogitsLossModuleDunderNumber: number = bceWithLogitsLossModuleClass.__call__([0, 2] as const, [0, 1] as const);
const crossEntropyLossModule: CrossEntropyLoss = loss.crossEntropyLoss({ classes: 3 });
const crossEntropyLossModuleAlias: CrossEntropyLoss = loss.cross_entropy_loss({ numClasses: 3 });
const crossEntropyLossModuleClass: CrossEntropyLoss = new loss.CrossEntropyLoss({ classes: 3 });
const nnCrossEntropyLossModuleClass: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 3 });
const crossEntropyLossSumModule: CrossEntropyLoss = loss.crossEntropyLoss({ classes: 3, reduction: "sum" });
const crossEntropyLossSumReduction: LossReduction = crossEntropyLossSumModule.reduction;
const nnCrossEntropyLossSumModuleClass: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 3, reduction: "sum" });
const nnCrossEntropyLossSumReduction: LossReduction = nnCrossEntropyLossSumModuleClass.reduction;
const crossEntropyLossModuleTensor: Tensor<readonly [1]> = crossEntropyLossModule.forward(trainingLogits, classTargetBuffer);
const crossEntropyLossModuleNumber: number = crossEntropyLossModuleAlias.call([1, 2, 3] as const, [2]);
const crossEntropyLossModuleClassNumber: number = crossEntropyLossModuleClass.forward([1, 2, 3] as const, [2]);
const nnCrossEntropyLossModuleClassNumber: number = nnCrossEntropyLossModuleClass.forward([1, 2, 3] as const, [2]);
const crossEntropyLossModuleDunderNumber: number = crossEntropyLossModuleClass.__call__([1, 2, 3] as const, [2]);
const nllLossModule: NLLLoss = loss.nllLossModule({ classes: 3 });
const nllLossModuleAlias: NLLLoss = loss.nll_loss_module({ numClasses: 3 });
const nllLossModuleClass: NLLLoss = new loss.NLLLoss({ classes: 3 });
const nnNllLossModuleClass: NLLLoss = new nn.NLLLoss({ classes: 3 });
const nllLossSumModule: NLLLoss = loss.nllLossModule({ classes: 3, reduction: "sum" });
const nllLossSumReduction: LossReduction = nllLossSumModule.reduction;
const nnNllLossSumModuleClass: NLLLoss = new nn.NLLLoss({ classes: 3, reduction: "sum" });
const nnNllLossSumReduction: LossReduction = nnNllLossSumModuleClass.reduction;
const nllLossModuleTensor: Tensor<readonly [1]> = nllLossModule.forward(trainingLogProbabilities, classTargetBuffer);
const nllLossModuleNumber: number = nllLossModuleAlias.call([-3, -2, -0.5] as const, [2]);
const nllLossModuleClassNumber: number = nllLossModuleClass.forward([-3, -2, -0.5] as const, [2]);
const nnNllLossModuleClassNumber: number = nnNllLossModuleClass.forward([-3, -2, -0.5] as const, [2]);
const nllLossModuleDunderNumber: number = nllLossModuleClass.__call__([-3, -2, -0.5] as const, [2]);
const trainingLoss: Tensor<readonly [1]> = checkpointModel.forward(tensor([1, 2], [2] as const)).meanSquaredError(tensor([0], [1] as const));
const namespaceTensorLoss: Tensor<readonly [1]> = loss.mse(checkpointModel.forward(tensor([1, 2], [2] as const)), tensor([0], [1] as const));
const publicTrainNamespace: PublicTrainNamespace = train;
train.backward(trainingLoss);
const observedTrainingNorm: number = train.gradNorm(checkpointModel);
const observedTrainingNormSnake: number = train.grad_norm(checkpointModel);
const clippedTrainingNorm: number = train.clipGradNorm(checkpointModel, 1.0);
const clippedTrainingNormSnake: number = train.clip_grad_norm_(checkpointModel, 1.0, { eps: 1e-6 });
const valueClippedCheckpointModel: typeof checkpointModel = train.clipGradValue(checkpointModel, 0.5);
const valueClippedCheckpointModelSnake: typeof checkpointModel = train.clip_grad_value_(checkpointModel, 0.5);
train.step(checkpointOptimizer, { loss: trainingLoss, zeroGrad: true, zeroGradOptions: { setToNone: true } });
train.step(checkpointOptimizer, {
  loss: trainingLoss,
  zero_grad: true,
  zero_grad_options: { set_to_none: true },
  clipGradNorm: 1,
  clipGradNormOptions: { eps: 1e-6 },
  clipGradValue: 0.5,
});
const inspectedTrainingStep: TrainStepEvidence | null = train.inspectStep();
const inspectedTrainingStepSignature: string | null = inspectedTrainingStep?.signature ?? null;
const inlineTrainingStepEvidence: TrainStepEvidence = train.step(checkpointOptimizer, {
  loss: trainingLoss,
  zeroGrad: true,
  zero_grad_options: { set_to_none: true },
  clip_grad_norm: 1,
  clip_grad_norm_options: { eps: 1e-6 },
  clip_grad_value: 0.5,
  inspect: true,
});
const inlineTrainingStepEvidenceSignature: string = inlineTrainingStepEvidence.signature;
const inlineTrainingStepClipNormApplied: boolean = inlineTrainingStepEvidence.clipGradNormApplied;
const inlineTrainingStepClipValueApplied: boolean = inlineTrainingStepEvidence.clipGradValueApplied;
const inlineTrainingStepGradNormBeforeClip: number | null = inlineTrainingStepEvidence.gradNormBeforeClip;
const inlineTrainingStepGradNormAfterClip: number | null = inlineTrainingStepEvidence.gradNormAfterClip;
const inlineAdamTrainingStepEvidence: TrainStepEvidence<"adam"> = train.step(checkpointOptimizer, {
  loss: trainingLoss,
  zeroGrad: true,
  clipGradNorm: 1,
  evidence: true,
});
const inlineAdamTrainingStepOptimizerKind: "adam" = inlineAdamTrainingStepEvidence.optimizerKind;
const inlineTrainingStepIsEvidence: boolean = train.isTrainStepEvidence(inlineTrainingStepEvidence);
const inlineTrainingStepRequired: TrainStepEvidence = train.requireTrainStepEvidence(inlineTrainingStepEvidence);
const inlineTrainingStepAsserted: TrainStepEvidence = train.assert_train_step_evidence(inlineTrainingStepEvidence);
const inlineTrainingStepSignatureMatch: boolean = train.matchesTrainStepEvidenceSignature(inlineTrainingStepEvidence, inlineTrainingStepEvidence.signature);
const inlineTrainingStepSubpathIsEvidence: boolean = isTrainStepEvidenceSubpath(inlineTrainingStepEvidence);
const inlineTrainingStepSubpathRequired: TrainStepEvidence = requireTrainStepEvidenceSubpath(inlineTrainingStepEvidence);
const trainingLossStep: Tensor = train.lossStep(checkpointOptimizer, () => (
  checkpointModel.forward(tensor([1, 2], [2] as const)).meanSquaredError(tensor([0], [1] as const))
));
const trainingLossStepSnake: Tensor = train.lossStep(checkpointOptimizer, () => (
  checkpointModel.forward(tensor([1, 2], [2] as const)).meanSquaredError(tensor([0], [1] as const))
), { zero_grad: true, zero_grad_options: { set_to_none: true } });
const publicDataNamespace: DataNamespace = data;
class CustomDataset extends data.Dataset<TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>, TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>> {
  readonly length = 2;
  sample(index: number): TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>> {
    return {
      kind: "zgml.data.sample",
      index,
      input: tensor(index === 0 ? [1, 0] : [0, 1], [2] as const),
      target: tensor([index], [1] as const),
    };
  }
}
const customDataset: Dataset<TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>, TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>> = new CustomDataset();
const customDatasetLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = new data.DataLoader(customDataset, { batch_size: 2 });
const customTorchDataset = new torch.utils.data.Dataset<TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>, TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>>();
const customSequentialSampler: SequentialSampler = new data.SequentialSampler(customDataset);
const customRandomSampler: RandomSampler = new torch.utils.data.RandomSampler(customDataset, { seed: 7 });
const customReplacementSampler: Sampler = new data.RandomSampler(customDataset, { replacement: true, numSamples: 3, seed: 7 });
const customBatchSampler: BatchSampler = new torch.utils.data.BatchSampler(customSequentialSampler, 1, false);
const customSamplerLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = data.dataLoader(customDataset, { sampler: customRandomSampler, batch_size: 1 });
const customBatchSamplerLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = new data.DataLoader(customDataset, { batchSampler: customBatchSampler });
const customCollateFn: DataCollateFn<
  TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>,
  TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>
> = (samples, context) => ({
  kind: "zgml.data.batch",
  batchIndex: context.batchIndex,
  indices: context.indices,
  input: tensor(samples.flatMap((sample) => Array.from(sample.input.data)), [samples.length, 2] as const),
  target: tensor(samples.flatMap((sample) => Array.from(sample.target!.data)), [samples.length, 1] as const),
});
const customCollateLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = torch.utils.data.dataLoader(customDataset, { batch_size: 2, collateFn: customCollateFn });
type CustomObjectCollateBatch = Readonly<{
  kind: "custom-object-collate";
  indices: readonly number[];
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
  firstInput: Tensor<readonly [2]>;
}>;
const customObjectCollateFn: DataCollateFn<
  TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>,
  CustomObjectCollateBatch
> = (samples, context) => ({
  kind: "custom-object-collate",
  indices: context.indices,
  sampleIndices: context.sampleIndices,
  sample_indices: context.sample_indices,
  firstInput: samples[0]!.input,
});
const customObjectCollateLoader: CollatedDataLoader<CustomObjectCollateBatch> = torch.utils.data.dataLoader(customDataset, { batch_size: 2, collateFn: customObjectCollateFn });
const customObjectCollateBatch: CustomObjectCollateBatch = customObjectCollateLoader.batch(0);
const defaultCollateOptions: DefaultCollateOptions = { batch_index: 7, sample_indices: [1, 0] };
const defaultCollateBatch: TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = data.defaultCollate(
  [customDataset.sample(1), customDataset.sample(0)],
  defaultCollateOptions,
);
const defaultCollateBatchAlias: TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = torch.utils.data.default_collate(
  [customDataset.sample(0), customDataset.sample(1)],
);
const tensorDataset: TensorDataset = data.tensorDataset(
  tensor([1, 0, 0, 1], [2, 2] as const),
  tensor([1, 1], [2, 1] as const),
);
const tensorDatasetAlias: TensorDataset = data.tensor_dataset(
  tensor([1, 0, 0, 1], [2, 2] as const),
  tensor([1, 1], [2, 1] as const),
);
const tensorDatasetSample: TensorDatasetSample = tensorDataset.sample(0);
const tensorDatasetSampleAlias: TensorDatasetSample = tensorDataset.get(1);
const tensorDatasetSampleAt: TensorDatasetSample = tensorDataset.at(0);
const tensorDatasetSampleDunder: TensorDatasetSample = tensorDataset.__getitem__(1);
const tensorDatasetIterableSample: TensorDatasetSample = Array.from(tensorDataset)[0]!;
const tensorDatasetDunderIterSample: TensorDatasetSample = tensorDataset.__iter__().next().value!;
const typedTensorDataset = data.tensorDataset(
  tensor([1, 0, 0, 1], [2, 2] as const),
  tensor([1, 1], [2, 1] as const),
);
type TypedTensorDatasetInputRowShape = Expect<Equal<TensorShapeTail<readonly [2, 2]>, readonly [2]>>;
type TypedTensorDatasetBatchShape = Expect<Equal<TensorDatasetBatchShape<readonly [2, 2]>, readonly [number, 2]>>;
const typedTensorDatasetSampleInput: Tensor<readonly [2]> = typedTensorDataset.sample(0).input;
const typedTensorDatasetSampleTarget: Tensor<readonly [1]> | undefined = typedTensorDataset.get(1).target;
const typedTensorDatasetAtInput: Tensor<readonly [2]> = typedTensorDataset.at(0).input;
const typedTensorDatasetDunderTarget: Tensor<readonly [1]> | undefined = typedTensorDataset.__getitem__(1).target;
const typedTensorDatasetIterInput: Tensor<readonly [2]> = Array.from(typedTensorDataset)[0]!.input;
const typedTensorDatasetDunderIterTarget: Tensor<readonly [1]> | undefined = typedTensorDataset.__iter__().next().value!.target;
const typedTensorDatasetBatchInput: Tensor<readonly [number, 2]> = typedTensorDataset.batch([0, 1]).input;
const typedTensorDatasetBatchTarget: Tensor<readonly [number, 1]> | undefined = typedTensorDataset.batch([0, 1]).target;
const typedTensorDatasetIterableBatchInput: Tensor<readonly [number, 2]> = typedTensorDataset.batch(new Set([0, 1])).input;
const splitOptions: DataSplitOptions = { seed: 11 };
const tensorDatasetSplitPair: readonly [TensorDataset, TensorDataset] = data.randomSplit(tensorDataset, [1, 1], splitOptions);
const tensorDatasetSplitPairAlias: readonly [TensorDataset, TensorDataset] = data.random_split(tensorDataset, [1, 1], { shuffle: false });
const tensorDatasetIterableSplit: readonly TensorDataset[] = data.randomSplit(tensorDataset, new Set([1, 1]), splitOptions);
const tensorDatasetSubset: TensorDataset = data.subset(tensorDataset, [1, 0]);
const tensorDatasetSubsetClass: TensorDataset = new data.Subset(tensorDataset, [1, 0]);
const tensorDatasetIterableSubset: TensorDataset = data.subset(tensorDataset, new Set([1, 0]));
const tensorDatasetTake: TensorDataset = data.take(tensorDataset, 1);
const tensorDatasetConcat: TensorDataset = data.concatDataset([tensorDatasetTake, tensorDatasetSubset]);
const tensorDatasetConcatClass: TensorDataset = new data.ConcatDataset([tensorDatasetTake, tensorDatasetSubset]);
const tensorDatasetIterableConcat: TensorDataset = data.concatDataset(new Set([tensorDatasetTake, tensorDatasetIterableSubset]));
const tensorDatasetConcatAlias: TensorDataset = data.concat_dataset([tensorDatasetSubset, tensorDatasetTake]);
const tensorDatasetMapper: TensorDatasetMapper = (sample) => ({ input: sample.input, target: sample.target });
const tensorDatasetMapped: TensorDataset = data.mapDataset(tensorDataset, tensorDatasetMapper);
const tensorDatasetMappedClass: TensorDataset = new data.MapDataset(tensorDataset, tensorDatasetMapper);
const tensorDatasetMappedAlias: TensorDataset = data.map_dataset(tensorDataset, (sample) => ({ input: sample.input }));
const typedTensorDatasetSplitPair: readonly [
  TensorDataset<Tensor<readonly [2]>, Tensor<readonly [1]>, Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>,
  TensorDataset<Tensor<readonly [2]>, Tensor<readonly [1]>, Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>,
] = data.randomSplit(typedTensorDataset, [1, 1], { seed: 17 });
const typedTensorDatasetIterableSplit: readonly TensorDataset<
  Tensor<readonly [2]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 2]>,
  Tensor<readonly [number, 1]>
>[] = data.randomSplit(typedTensorDataset, new Set([1, 1]), { seed: 17 });
const typedTensorDatasetSplitInput: Tensor<readonly [2]> = typedTensorDatasetSplitPair[0].sample(0).input;
const typedTensorDatasetSplitTarget: Tensor<readonly [1]> | undefined = typedTensorDatasetSplitPair[1].__iter__().next().value!.target;
const typedTensorDatasetSplitBatchInput: Tensor<readonly [number, 2]> = typedTensorDatasetSplitPair[0].batch([0]).input;
const typedTensorDatasetSplitIterableBatchInput: Tensor<readonly [number, 2]> = typedTensorDatasetSplitPair[0].batch(new Set([0])).input;
const typedTensorDatasetConcat: TensorDataset<
  Tensor<readonly [2]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 2]>,
  Tensor<readonly [number, 1]>
> = data.concatDataset(typedTensorDatasetSplitPair);
const typedTensorDatasetSubsetClass: TensorDataset<
  Tensor<readonly [2]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 2]>,
  Tensor<readonly [number, 1]>
> = new data.Subset(typedTensorDataset, [1, 0]);
const typedTensorDatasetConcatClass: TensorDataset<
  Tensor<readonly [2]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 2]>,
  Tensor<readonly [number, 1]>
> = new data.ConcatDataset(typedTensorDatasetSplitPair);
const typedTensorDatasetConcatInput: Tensor<readonly [2]> = typedTensorDatasetConcat.sample(0).input;
const typedTensorDatasetConcatBatchTarget: Tensor<readonly [number, 1]> | undefined = typedTensorDatasetConcat.batch([0, 1]).target;
const typedTensorDatasetConcatIterableBatchTarget: Tensor<readonly [number, 1]> | undefined = typedTensorDatasetConcat.batch(new Set([0, 1])).target;
const tensorDatasetLen: number = tensorDataset.len();
const tensorDatasetDunderLen: number = tensorDataset.__len__();
const tensorDatasetSize: number = tensorDataset.size();
const tensorDatasetBatch: TensorDatasetBatch = tensorDataset.batch([0, 1], 0);
const tensorDatasetBatches: DataBatches = data.batches(tensorDataset, { batchSize: 1, shuffle: true, seed: 7 });
const tensorDatasetLoader: DataLoader = data.dataLoader(tensorDatasetAlias, { batch_size: 2, drop_last: false });
const classTensorDataset: TensorDataset<
  Tensor<readonly [2]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 2]>,
  Tensor<readonly [number, 1]>
> = new data.TensorDataset(
  tensor([1, 0, 0, 1], [2, 2] as const),
  tensor([1, 0], [2, 1] as const),
);
const classDataLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> =
  new data.DataLoader(classTensorDataset, { batch_size: 2, shuffle: false });
const typedTensorDatasetLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = data.dataLoader(typedTensorDataset, { batch_size: 2 });
const typedTensorDatasetLoaderBatchInput: Tensor<readonly [number, 2]> = Array.from(typedTensorDatasetLoader)[0]!.input;
const typedTensorDatasetLoaderGetInput: Tensor<readonly [number, 2]> = typedTensorDatasetLoader.get(0).input;
const typedTensorDatasetLoaderDunderTarget: Tensor<readonly [number, 1]> | undefined = typedTensorDatasetLoader.__getitem__(0).target;
const typedTensorDatasetLoaderIterInput: Tensor<readonly [number, 2]> = typedTensorDatasetLoader.__iter__().next().value!.input;
type TypedTrainBatch = TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>;
type TypedTrainBatchInputShape = Expect<Equal<TrainBatchInputShape<TypedTrainBatch>, readonly [number, 2]>>;
type TypedTrainBatchTargetShape = Expect<Equal<TrainBatchTargetShape<TypedTrainBatch>, readonly [number, 1]>>;
type TypedTrainBatchOutputShape = Expect<Equal<TrainModuleOutputShape<typeof customChild, TypedTrainBatch>, readonly [number, 1]>>;
const typedTrainSupervisedBatch: TrainSupervisedBatch<typeof customChild, TypedTrainBatch> = typedTensorDatasetLoader.__getitem__(0);
// @ts-expect-error supervised training batches reject modules whose output shape misses the target shape.
const invalidTrainSupervisedBatch: TrainSupervisedBatch<typeof linear, TypedTrainBatch> = typedTensorDatasetLoader.__getitem__(0);
const typedClassTensorDataset = data.tensorDataset(
  tensor([1, 0, 0, 1], [2, 2] as const),
  tensor([0, 2], [2] as const),
);
const typedClassDatasetLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number]>> = data.dataLoader(typedClassTensorDataset, { batch_size: 2 });
type TypedClassBatch = TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number]>>;
type TypedClassTargetShape = Expect<Equal<TrainClassificationTargetShape<readonly [number, 3]>, readonly [number]>>;
type TypedClassBatchOutputShape = Expect<Equal<TrainModuleOutputShape<typeof linear, TypedClassBatch>, readonly [number, 3]>>;
const typedTrainClassificationBatch: TrainClassificationBatch<typeof linear, TypedClassBatch> = typedClassDatasetLoader.__getitem__(0);
// @ts-expect-error classification batches require target shape to omit the logits class axis.
const invalidTrainClassificationBatch: TrainClassificationBatch<typeof linear, TypedTrainBatch> = typedTensorDatasetLoader.__getitem__(0);
const tensorDatasetDataloader: DataLoader = data.dataloader(tensorDatasetAlias, { batch_size: 2, drop_last: false });
const tensorDatasetLoaderSampleCount: number = tensorDatasetLoader.sampleCount;
const tensorDatasetLoaderSampleCountAlias: number = tensorDatasetLoader.sample_count;
const tensorDatasetLoaderBatchSize: number = tensorDatasetLoader.batchSize;
const tensorDatasetLoaderBatchSizeAlias: number = tensorDatasetLoader.batch_size;
const tensorDatasetLoaderBatchCount: number = tensorDatasetLoader.batchCount;
const tensorDatasetLoaderBatchCountAlias: number = tensorDatasetLoader.batch_count;
const tensorDatasetLoaderLen: number = tensorDatasetLoader.len();
const tensorDatasetLoaderDunderLen: number = tensorDatasetLoader.__len__();
const tensorDatasetLoaderDropLastAlias: boolean = tensorDatasetLoader.drop_last;
const tensorDatasetLoaderSize: number = tensorDatasetLoader.size();
const tensorDatasetDataloaderDropLast: boolean = tensorDatasetDataloader.dropLast;
const fitDataEvidence: TrainFitEvidence = train.fit(checkpointOptimizer, tensorDatasetBatches, (batch: TensorDatasetBatch, context: TrainFitContext) => {
  const batchIndex: number = batch.batchIndex;
  const inputBatch: Tensor = batch.input;
  const targetBatch: Tensor | undefined = batch.target;
  const epoch: number = context.epoch;
  const contextSampleIndices: readonly number[] | null = context.sampleIndices;
  const contextSampleIndicesAlias: readonly number[] | null = context.sample_indices;
  void batchIndex;
  void inputBatch;
  void targetBatch;
  void epoch;
  void contextSampleIndices;
  void contextSampleIndicesAlias;
  return loss.mse(checkpointModel.forward(batch.input.select(0, 0)), batch.target!.select(0, 0));
}, { maxSteps: 1 });
const fitDataEvidenceTyped: TrainFitEvidence<"adam"> = train.fit(checkpointOptimizer, tensorDatasetBatches, (batch: TensorDatasetBatch, context: TrainFitContext) => {
  void context;
  return loss.mse(checkpointModel.forward(batch.input.select(0, 0)), batch.target!.select(0, 0));
}, {
  maxSteps: 1,
  onStep: (evidence: TrainFitStepEvidence<"adam">) => {
    const fitStepOptimizerKind: "adam" | null = evidence.stepEvidence?.optimizerKind ?? null;
    void fitStepOptimizerKind;
  },
});
const modelFirstFitOptions: TrainModelFitOptions<"adam", typeof checkpointModel, TensorDatasetBatch> = {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
};
const modelFirstFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointModel, tensorDatasetBatches, modelFirstFitOptions);
const rootFitEvidence: TrainFitEvidence<"adam"> = fit(checkpointModel, tensorDatasetBatches, modelFirstFitOptions);
const rootFitModuleEvidence: TrainFitEvidence<"adam"> = fitModule(checkpointOptimizer, checkpointModel, tensorDatasetBatches, new nn.MSELoss(), { maxSteps: 1 });
const rootFitModuleSnakeEvidence: TrainFitEvidence<"adam"> = fit_module(checkpointOptimizer, checkpointModel, tensorDatasetBatches, new nn.MSELoss(), { maxSteps: 1 });
const optimizerFirstModuleFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointOptimizer, checkpointModel, tensorDatasetBatches, new nn.MSELoss(), { maxSteps: 1 });
const zgmlFitEvidence: TrainFitEvidence<"adam"> = zgml.fit(checkpointModel, tensorDatasetBatches, modelFirstFitOptions);
void rootFitEvidence;
void rootFitModuleEvidence;
void rootFitModuleSnakeEvidence;
void optimizerFirstModuleFitEvidence;
void zgmlFitEvidence;
const modelFirstNativeFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  requireNative: true,
  inputShape: [1, 2] as const,
  classes: 1,
  numClasses: 1,
  num_classes: 1,
});
const modelFirstNativeFitPlan: CompiledTrainingPlan | null | undefined = modelFirstNativeFitEvidence.compiledPlan;
const nativeRegressionLossName: TrainNativeLossName = "mse";
const nativeClassificationLossName: TrainNativeLossName = "crossEntropy";
const modelFirstNativeStringLossFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: nativeRegressionLossName,
  maxSteps: 1,
  requireNative: true,
  inputShape: [1, 2] as const,
});
const optimizerFirstNativeFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointOptimizer, checkpointModel, tensorDatasetBatches, new nn.MSELoss(), {
  maxSteps: 1,
  requireNative: true,
  inputShape: [1, 2] as const,
});
const optimizerFirstNativeStringLossFitEvidence: TrainFitEvidence<"adam"> = train.fitModule(checkpointOptimizer, checkpointModel, tensorDatasetBatches, nativeRegressionLossName, {
  maxSteps: 1,
  requireNative: true,
  inputShape: [1, 2] as const,
});
const optimizerFirstNativeFitPlan: CompiledTrainingPlan | null | undefined = optimizerFirstNativeFitEvidence.compiledPlan;
const modelFirstNativeTrainingExplanation: NativeTrainingExplanation = train.explainNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const modelFirstNativeStringLossTrainingExplanation: NativeTrainingExplanation = train.explainNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  criterion: nativeRegressionLossName,
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const optimizerFirstNativeTrainingExplanation: NativeTrainingExplanation = train.explain_native(
  checkpointOptimizer,
  checkpointModel,
  tensorDatasetBatches,
  new nn.MSELoss(),
  { maxSteps: 1, inputShape: [1, 2] as const },
);
const rootNativeTrainingExplanation: NativeTrainingExplanation = explainNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const rootNativeTrainingPlanAlias: NativeTrainingExplanation = nativeTrainingPlan(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const rootNativeTrainingSnakePlanAlias: NativeTrainingExplanation = native_training_plan(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const moduleNativeTrainingExplanation: NativeTrainingExplanation = checkpointModel.explainTraining(tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const moduleNativeTrainingSnakeExplanation: NativeTrainingExplanation = checkpointModel.explain_native_training(tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const rootNativeTrainingBulkPlan: NativeTrainingBulkFitPlan | null = rootNativeTrainingExplanation.bulkPlan;
const rootNativeTrainingSnakeBulkPlan: NativeTrainingBulkFitPlan | null = rootNativeTrainingExplanation.bulk_plan;
const rootNativeTrainingBulkKernel: string | null = rootNativeTrainingExplanation.bulkKernel;
const rootNativeTrainingNativeBulk: boolean = rootNativeTrainingExplanation.nativeBulk;
void modelFirstNativeTrainingExplanation;
void modelFirstNativeStringLossTrainingExplanation;
void optimizerFirstNativeTrainingExplanation;
void rootNativeTrainingExplanation;
void rootNativeTrainingBulkPlan;
void rootNativeTrainingSnakeBulkPlan;
void rootNativeTrainingBulkKernel;
void rootNativeTrainingNativeBulk;
void rootNativeTrainingPlanAlias;
void rootNativeTrainingSnakePlanAlias;
void moduleNativeTrainingExplanation;
void moduleNativeTrainingSnakeExplanation;
void modelFirstNativeStringLossFitEvidence;
void optimizerFirstNativeStringLossFitEvidence;
void nativeClassificationLossName;
const compiledTrainingForAlias: CompiledTrainingStep = compile.forTraining(checkpointModel, checkpointOptimizer, {
  inputShape: [1, 2] as const,
  loss: "mse",
});
const compiledTrainingForSnakeAlias: CompiledTrainingStep = compile.for_training(checkpointModel, checkpointOptimizer, {
  input_shape: [1, 2] as const,
  criterion: "mse",
});
const rootForTrainingAlias: CompiledTrainingStep = forTraining(checkpointModel, checkpointOptimizer, {
  inputShape: [1, 2] as const,
  loss: "mse",
});
const rootForTrainingSnakeAlias: CompiledTrainingStep = for_training(checkpointModel, checkpointOptimizer, {
  input_shape: [1, 2] as const,
  criterion: "mse",
});
const zgmlForTrainingAlias: CompiledTrainingStep = zgml.forTraining(checkpointModel, checkpointOptimizer, {
  inputShape: [1, 2] as const,
  loss: "mse",
});
const moduleForTrainingAlias: CompiledTrainingStep = checkpointModel.forTraining(checkpointOptimizer, {
  inputShape: [1, 2] as const,
  loss: "mse",
});
const moduleForTrainingSnakeAlias: CompiledTrainingStep = checkpointModel.for_training(checkpointOptimizer, {
  input_shape: [1, 2] as const,
  criterion: "mse",
});
const modelFirstFitNativeEvidence: TrainFitEvidence<"adam"> = train.fitNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const rootFitNativeEvidence: TrainFitEvidence<"adam"> = fitNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const rootFitNativeAliasEvidence: TrainFitEvidence<"adam"> = zgml.fitNative(checkpointModel, tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
void modelFirstFitNativeEvidence;
void optimizerFirstNativeFitPlan;
void rootFitNativeEvidence;
void rootFitNativeAliasEvidence;
const moduleMethodFitEvidence: TrainFitEvidence = checkpointModel.fit(tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
});
const moduleMethodFitNativeEvidence: TrainFitEvidence = checkpointModel.fitNative(tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
const moduleMethodFitNativeSnakeEvidence: TrainFitEvidence = checkpointModel.fit_native(tensorDatasetBatches, {
  optimizer: checkpointOptimizer,
  loss: new nn.MSELoss(),
  maxSteps: 1,
  inputShape: [1, 2] as const,
});
void moduleMethodFitNativeEvidence;
void moduleMethodFitNativeSnakeEvidence;
const moduleMethodEvalEvidence: TrainEvaluateEvidence = checkpointModel.evaluate(tensorDatasetBatches, new nn.MSELoss(), {
  maxSteps: 1,
});
const moduleMethodPredictEvidence: TrainPredictEvidence = checkpointModel.predict(tensorDatasetBatches, {
  maxSteps: 1,
});
const fitDataLastStepOptimizerKind: "adam" | null = fitDataEvidenceTyped.lastStep?.optimizerKind ?? null;
const fitDataBatchCount: number | null = fitDataEvidence.batchCount;
const fitDataBatchCountAlias: number | null = fitDataEvidence.batch_count;
const fitDataSampleCount: number | null = fitDataEvidence.sampleCount;
const fitDataSampleCountAlias: number | null = fitDataEvidence.sample_count;
const fitDataStopReason: "max-steps" | "early-stopping" | null = fitDataEvidence.stopReason;
const fitDataBestLoss: number | null = fitDataEvidence.best_loss;
const fitEarlyStoppingOptions: TrainEarlyStoppingOptions = { patience: 1, min_delta: 0, mode: "min" };
const fitDataEvidenceRequired: TrainFitEvidence = train.requireTrainFitEvidence(fitDataEvidence);
const fitDataEvidenceAsserted: TrainFitEvidence = train.assert_train_fit_evidence(fitDataEvidence);
const fitDataEvidenceSignatureMatch: boolean = train.matchesTrainFitEvidenceSignature(fitDataEvidence, fitDataEvidence.signature);
const fitDataEvidenceSubpathRequired: TrainFitEvidence = requireTrainFitEvidenceSubpath(fitDataEvidence);
const fitDataEvidenceSnake: TrainFitEvidence = train.fit(checkpointOptimizer, tensorDatasetLoader, (batch: TensorDatasetBatch, context: TrainFitContext) => {
  const batchIndex: number = batch.batchIndex;
  const step: number = context.step;
  void batchIndex;
  void step;
  return loss.mse(checkpointModel.forward(batch.input.select(0, 0)), batch.target!.select(0, 0));
}, {
  max_steps: 1,
  zero_grad: true,
  early_stopping: fitEarlyStoppingOptions,
  on_step: (evidence: TrainFitStepEvidence) => {
    const fitStepLoss: number = evidence.loss;
    const fitStepSignature: string = evidence.signature;
    const fitStepSampleIndices: readonly number[] | null = evidence.sampleIndices;
    const fitStepSampleIndicesAlias: readonly number[] | null = evidence.sample_indices;
    const fitStepRequired: TrainFitStepEvidence = train.requireTrainFitStepEvidence(evidence);
    const fitStepAsserted: TrainFitStepEvidence = train.assert_train_fit_step_evidence(evidence);
    const fitStepSignatureMatch: boolean = train.matchesTrainFitStepEvidenceSignature(evidence, evidence.signature);
    const fitStepSubpathRequired: TrainFitStepEvidence = requireTrainFitStepEvidenceSubpath(evidence);
    void fitStepLoss;
    void fitStepSignature;
    void fitStepSampleIndices;
    void fitStepSampleIndicesAlias;
    void fitStepRequired;
    void fitStepAsserted;
    void fitStepSignatureMatch;
    void fitStepSubpathRequired;
  },
});
const fitBatch: readonly { input: Tensor; target: Tensor }[] = [
  { input: tensor([1, 2], [2] as const), target: tensor([0], [1] as const) },
];
const fitEvidence: TrainFitEvidence = train.fit(checkpointOptimizer, fitBatch, (batch: { input: Tensor; target: Tensor }, context: TrainFitContext) => {
  const fitEpoch: number = context.epoch;
  const fitStep: number = context.step;
  void fitEpoch;
  void fitStep;
  return loss.mse(checkpointModel.forward(batch.input), batch.target);
}, {
  epochs: 1,
  maxSteps: 1,
  onStep: (evidence: TrainFitStepEvidence) => {
    const fitStepLoss: number = evidence.loss;
    const fitStepSignature: string = evidence.signature;
    void fitStepLoss;
    void fitStepSignature;
  },
});
const fitEvidenceSignature: string = fitEvidence.signature;
const fitFinalLoss: number | null = fitEvidence.finalLoss;
const fitPlainBatchCount: number | null = fitEvidence.batchCount;
const evaluateEvidence: TrainEvaluateEvidence = train.evaluate(tensorDatasetLoader, (batch: TensorDatasetBatch, context: TrainEvaluateContext) => {
  const batchIndex: number = context.batchIndex;
  const step: number = context.step;
  const sampleIndices: readonly number[] | null = context.sampleIndices;
  void batchIndex;
  void step;
  void sampleIndices;
  return loss.mse(checkpointModel.forward(batch.input.select(0, 0)), batch.target!.select(0, 0));
}, {
  maxSteps: 1,
  onStep: (evidence: TrainEvaluateStepEvidence) => {
    const evaluateStepLoss: number = evidence.loss;
    const evaluateStepSignature: string = evidence.signature;
    const evaluateStepSampleIndices: readonly number[] | null = evidence.sampleIndices;
    const evaluateStepRequired: TrainEvaluateStepEvidence = train.requireTrainEvaluateStepEvidence(evidence);
    const evaluateStepAsserted: TrainEvaluateStepEvidence = train.assert_train_evaluate_step_evidence(evidence);
    const evaluateStepSignatureMatch: boolean = train.matchesTrainEvaluateStepEvidenceSignature(evidence, evidence.signature);
    const evaluateStepSubpathRequired: TrainEvaluateStepEvidence = requireTrainEvaluateStepEvidenceSubpath(evidence);
    void evaluateStepLoss;
    void evaluateStepSignature;
    void evaluateStepSampleIndices;
    void evaluateStepRequired;
    void evaluateStepAsserted;
    void evaluateStepSignatureMatch;
    void evaluateStepSubpathRequired;
  },
});
const evaluateEvidenceAlias: TrainEvaluateEvidence = train.evaluate_loss(tensorDatasetLoader, (batch: TensorDatasetBatch) => (
  loss.mse(checkpointModel.forward(batch.input.select(0, 0)), batch.target!.select(0, 0))
), { max_steps: 1 });
const evaluateModuleEvidence: TrainEvaluateEvidence = train.evaluateModule(checkpointModel, tensorDatasetLoader, new nn.MSELoss(), { maxSteps: 1 });
const evaluateModuleEvidenceAlias: TrainEvaluateEvidence = train.evaluate_module(checkpointModel, tensorDatasetLoader, new nn.MSELoss(), { max_steps: 1 });
const evaluateMeanLoss: number | null = evaluateEvidence.meanLoss;
const evaluateMeanLossAlias: number | null = evaluateEvidence.mean_loss;
const evaluateFinalLoss: number | null = evaluateEvidence.finalLoss;
const evaluateFinalLossAlias: number | null = evaluateEvidence.final_loss;
const evaluateEvidenceRequired: TrainEvaluateEvidence = train.requireTrainEvaluateEvidence(evaluateEvidence);
const evaluateEvidenceAsserted: TrainEvaluateEvidence = train.assert_train_evaluate_evidence(evaluateEvidence);
const evaluateEvidenceSignatureMatch: boolean = train.matchesTrainEvaluateEvidenceSignature(evaluateEvidence, evaluateEvidence.signature);
const evaluateEvidenceSubpathRequired: TrainEvaluateEvidence = requireTrainEvaluateEvidenceSubpath(evaluateEvidence);
const predictEvidence: TrainPredictEvidence<Tensor> = train.predict(tensorDatasetLoader, (batch: TensorDatasetBatch, context: TrainPredictContext) => {
  const batchIndex: number = context.batchIndex;
  const step: number = context.step;
  const sampleIndices: readonly number[] | null = context.sample_indices;
  void batchIndex;
  void step;
  void sampleIndices;
  return checkpointModel.forward(batch.input.select(0, 0));
}, {
  maxSteps: 1,
  onStep: (evidence: TrainPredictStepEvidence<Tensor>) => {
    const predictStepOutput: Tensor = evidence.output;
    const predictStepSignature: string = evidence.signature;
    const predictStepSampleIndices: readonly number[] | null = evidence.sampleIndices;
    const predictStepRequired: TrainPredictStepEvidence<Tensor> = train.requireTrainPredictStepEvidence(evidence);
    const predictStepAsserted: TrainPredictStepEvidence<Tensor> = train.assert_train_predict_step_evidence(evidence);
    const predictStepSignatureMatch: boolean = train.matchesTrainPredictStepEvidenceSignature(evidence, evidence.signature);
    const predictStepSubpathRequired: TrainPredictStepEvidence = requireTrainPredictStepEvidenceSubpath(evidence);
    const predictStepSubpathSignatureMatch: boolean = matchesTrainPredictStepEvidenceSignatureSubpath(evidence, evidence.signature);
    void predictStepOutput;
    void predictStepSignature;
    void predictStepSampleIndices;
    void predictStepRequired;
    void predictStepAsserted;
    void predictStepSignatureMatch;
    void predictStepSubpathRequired;
    void predictStepSubpathSignatureMatch;
  },
});
const predictModuleEvidence: TrainPredictEvidence<Tensor> = train.predictModule(checkpointModel, tensorDatasetLoader, { maxSteps: 1 });
const predictModuleEvidenceAlias: TrainPredictEvidence<Tensor> = train.predict_module(checkpointModel, tensorDatasetLoader, { max_steps: 1 });
const predictClassifierEvidence: TrainPredictEvidence<Tensor> = train.predictClassifier(checkpointModel, tensorDatasetLoader, { maxSteps: 1 });
const predictClassifierEvidenceAlias: TrainPredictEvidence<Tensor> = train.predict_classifier(checkpointModel, tensorDatasetLoader, { max_steps: 1 });
const predictEvidenceAlias: TrainPredictEvidence<Tensor> = train.predict_batches(tensorDatasetLoader, (batch: TensorDatasetBatch) => (
  checkpointModel.forward(batch.input.select(0, 0))
), { max_steps: 1 });
const predictOutputs: readonly Tensor[] = predictEvidence.outputs;
const predictLastOutput: Tensor | null = predictEvidence.lastOutput;
const predictLastOutputAlias: Tensor | null = predictEvidence.last_output;
const predictEvidenceValid: boolean = train.isTrainPredictEvidence(predictEvidence);
const predictEvidenceRequired: TrainPredictEvidence<Tensor> = train.requireTrainPredictEvidence(predictEvidence);
const predictEvidenceAsserted: TrainPredictEvidence<Tensor> = train.assert_train_predict_evidence(predictEvidence);
const predictEvidenceSignatureMatch: boolean = train.matchesTrainPredictEvidenceSignature(predictEvidence, predictEvidence.signature);
const predictEvidenceSubpathRequired: TrainPredictEvidence = requireTrainPredictEvidenceSubpath(predictEvidence);
const predictEvidenceSubpathSignatureMatch: boolean = matchesTrainPredictEvidenceSignatureSubpath(predictEvidence, predictEvidence.signature);
const publicCheckpointNamespace: CheckpointNamespace = checkpoint;
const createdCheckpoint: ZgmlCheckpoint = checkpoint.create({
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  scheduler: stepScheduler,
  metadata: { epoch: 1, tags: ["smoke"] },
});
const typedCheckpointCreateOptions: CheckpointCreateOptions<"adam"> = {
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  scheduler: stepScheduler,
};
const typedCheckpointRestoreOptions: CheckpointRestoreOptions<"adam"> = {
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  scheduler: stepScheduler,
  strict: true,
};
const checkpointFromTypedOptions: ZgmlCheckpoint = checkpoint.create(typedCheckpointCreateOptions);
const restoredFromTypedOptions: CheckpointRestoreOptions<"adam"> = checkpoint.restore(
  checkpointFromTypedOptions,
  typedCheckpointRestoreOptions,
);
const invalidCheckpointCreateOptions: CheckpointCreateOptions<"adam"> = {
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  // @ts-expect-error Checkpoint optimizer and scheduler targets must agree on optimizer state kind.
  scheduler: optim.stepLR(groupedOptimizer),
};
// @ts-expect-error checkpoint.create rejects mismatched optimizer/scheduler kind pairs.
checkpoint.create({ model: checkpointModel, optimizer: checkpointOptimizer, scheduler: optim.stepLR(groupedOptimizer) });
// @ts-expect-error checkpoint.restore rejects mismatched optimizer/scheduler kind pairs.
checkpoint.restore(createdCheckpoint, { model: checkpointModel, optimizer: checkpointOptimizer, scheduler: optim.stepLR(groupedOptimizer) });
const checkpointModelState: CheckpointModuleState | undefined = createdCheckpoint.model;
const checkpointOptimizerState: CheckpointOptimizerState | undefined = createdCheckpoint.optimizer;
const checkpointSchedulerState: CheckpointSchedulerState | undefined = createdCheckpoint.scheduler;
const checkpointInspection: CheckpointInspection = checkpoint.inspect(createdCheckpoint);
const checkpointInspectionSignature: string = checkpointInspection.signature;
const checkpointInspectionNames: readonly string[] = checkpointInspection.modelParameterNames;
const checkpointInspectionOptimizerStep: number | null = checkpointInspection.optimizerStep;
const checkpointInspectionHasScheduler: boolean = checkpointInspection.hasScheduler;
const checkpointInspectionSchedulerStep: number | null = checkpointInspection.schedulerStep;
const checkpointInspectionSchedulerLastLr: number | null = checkpointInspection.schedulerLastLr;
const checkpointInspectionSchedulerTMax: number | null = checkpointInspection.schedulerTMax;
const checkpointInspectionSchedulerTMaxAlias: number | null = checkpointInspection.schedulerT_max;
const checkpointInspectionSchedulerEtaMin: number | null = checkpointInspection.schedulerEtaMin;
const checkpointInspectionSchedulerEtaMinAlias: number | null = checkpointInspection.schedulerEta_min;
const cosineSchedulerCheckpoint: ZgmlCheckpoint = checkpoint.create({ scheduler: cosineScheduler });
const cosineSchedulerInspection: CheckpointInspection = checkpoint.inspect(cosineSchedulerCheckpoint);
const cosineSchedulerInspectionTMax: number | null = cosineSchedulerInspection.schedulerTMax;
const cosineSchedulerInspectionEtaMin: number | null = cosineSchedulerInspection.schedulerEtaMin;
const checkpointModelParameterInfo: CheckpointTensorInspection | undefined = checkpointInspection.modelParameters[0];
const checkpointModelParameterInfoSignature: string | undefined = checkpointModelParameterInfo?.signature;
const checkpointOptimizerEntryInfo: CheckpointTensorInspection | undefined = checkpointInspection.optimizerEntries[0];
const checkpointOptimizerEntryInfoSignature: string | undefined = checkpointOptimizerEntryInfo?.signature;
const checkpointModelParameterInfoByName: CheckpointTensorInspection | null = checkpoint.modelParameterInfo(createdCheckpoint, "weight");
const checkpointModelParameterInfoByIndex: CheckpointTensorInspection | null = checkpoint.modelParameterInfo(createdCheckpoint, 0);
const checkpointOptimizerEntryInfoByIndex: CheckpointTensorInspection | null = checkpoint.optimizerEntryInfo(createdCheckpoint, 0);
const checkpointOptimizerEntryInfoByName: CheckpointTensorInspection | null = checkpoint.optimizerEntryInfo(createdCheckpoint, "velocity.0");
const checkpointMetadata: Readonly<{ epoch: number; tags: readonly string[] }> | undefined =
  createdCheckpoint.metadata as Readonly<{ epoch: number; tags: readonly string[] }> | undefined;
const checkpointJsonSnapshot: ZgmlCheckpoint = checkpoint.toJSON(createdCheckpoint);
const checkpointFromJsonSnapshot: ZgmlCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpointJsonSnapshot)));
const checkpointSerializedSnapshot: string = checkpoint.stringify(createdCheckpoint, 2);
const checkpointParsedSnapshot: ZgmlCheckpoint = checkpoint.parse(checkpointSerializedSnapshot);
const checkpointSerializedSnapshotAlias: string = checkpoint.serialize(createdCheckpoint, 2);
const checkpointParsedSnapshotAlias: ZgmlCheckpoint = checkpoint.deserialize(checkpointSerializedSnapshotAlias);
const rootCheckpointText: string = save(createdCheckpoint, 2);
const rootCheckpointPath: string = save(createdCheckpoint, "/tmp/zgml-type-smoke.zgml", 2);
const rootCheckpointLoadedText: ZgmlCheckpoint = load(rootCheckpointText);
const rootCheckpointLoadedPath: ZgmlCheckpoint = load(rootCheckpointPath);
const torchCheckpointText: string = torch.save(createdCheckpoint, 2);
const torchCheckpointPath: string = torch.save(createdCheckpoint, "/tmp/zgml-type-smoke.torch.zgml", 2);
const torchCheckpointLoadedText: ZgmlCheckpoint = torch.load(torchCheckpointText);
const torchCheckpointLoadedPath: ZgmlCheckpoint = torch.load(torchCheckpointPath);
const restoreTargets: CheckpointRestoreOptions = checkpoint.restore(createdCheckpoint, {
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  scheduler: stepScheduler,
  strict: true,
});
const prefixedCheckpointRestoreTargets: CheckpointRestoreOptions = checkpoint.restore(createdCheckpoint, {
  model: checkpointModel,
  optimizer: checkpointOptimizer,
  scheduler: stepScheduler,
  strict: true,
  prefix: "head",
});
const rootCheckpointRestoreTargets: CheckpointRestoreOptions = load(rootCheckpointPath, restoreTargets);
const torchCheckpointRestoreTargets: CheckpointRestoreOptions = torch.load(torchCheckpointPath, restoreTargets);
checkpoint.load(createdCheckpoint, restoreTargets);
const checkpointModuleState: CheckpointModuleState = checkpoint.moduleStateDict(checkpointModel);
const checkpointModuleStateAlias: CheckpointModuleState = checkpoint.module_state_dict(checkpointModel);
const checkpointOptimizerStateFromNamespace: CheckpointOptimizerState = checkpoint.optimizerStateDict(checkpointOptimizer);
const checkpointOptimizerStateAlias: CheckpointOptimizerState = checkpoint.optimizer_state_dict(checkpointOptimizer);
const checkpointSchedulerStateFromNamespace: CheckpointSchedulerState = checkpoint.schedulerStateDict(stepScheduler);
const checkpointSchedulerStateAlias: CheckpointSchedulerState = checkpoint.scheduler_state_dict(stepScheduler);
const checkpointOptimizerStepFromNamespace: number = checkpointOptimizerStateFromNamespace.step;
const loadedOptimizer = optim.loadStateDict(checkpointOptimizer, optimizerStateSnapshot, { strict: true });

// @ts-expect-error Module state snapshots are immutable evidence records.
moduleStateSnapshot.extra = moduleStateSnapshot.weight;

// @ts-expect-error Optimizer snapshot entry names are immutable evidence records.
optimizerStateSnapshot.entries[0].name = "moved.0";

// @ts-expect-error Module binding plan parameter names are immutable evidence records.
linearBindingPlanParameterNames[0] = "moved.weight";

// @ts-expect-error Narrowed checkpoint metadata can be treated as immutable evidence.
checkpointMetadata!.tags[0] = "changed";

// @ts-expect-error Embedding programs need a static token-window shape at compile time.
embedding.compile({ backend: "cpu" });

// @ts-expect-error The root compile helper must keep the same embedding shape contract.
nn.compile(embedding, { backend: "cpu" });

void embeddingCompatibility;
void embeddingAccepted;
void typedEmbeddingProgram;
void typedEmbeddingNamespaceProgram;
void typedEmbeddingCompileNamespaceProgram;
void typedEmbeddingOutputShape;
void typedEmbeddingNamespaceOutputShape;
void typedEmbeddingMethodOutputShape;
void typedEmbeddingMethodOutputShapeAlias;
void typedEmbeddingSupport;
void typedEmbeddingCompilePlan;
void embeddingForwardTensor;
void embeddingGridForwardTensor;
void embeddingPlacedBindings;
void embeddingRootPlacedBindings;
void embeddingPlacedBindingPlan;
void embeddingPlacedBindingPlanSignature;
void embeddingRootPlacedBindingPlan;
void sequentialForwardTensor;
void sequentialNamespaceForwardTensor;
void variadicSequential;
void variadicSequentialAsModule;
void variadicSequentialForwardTensor;
void variadicClassSequential;
void variadicClassSequentialAsModule;
void variadicClassSequentialForwardTensor;
void singleLayerSequential;
void singleLayerSequentialAsModule;
void singleLayerSequentialForwardTensor;
void sequentialSupport;
void sequentialMethodExplanation;
void sequentialCompileExplanation;
void sequentialNamespaceCompileExplanation;
void sequentialCompilePlan;
void sequentialMethodCompilePlan;
void sequentialTrace;
void sequentialCompilerSignatures;
void sequentialIr;
void sequentialIrSignature;
void sequentialKernelPlan;
void sequentialKernelPlanSignature;
void sequentialMemoryLayout;
void sequentialMemoryLayoutSignature;
void sequentialBufferLayout;
void sequentialBufferLayoutSignature;
void sequentialInputShape;
void sequentialOutputShapeFromHelper;
void sequentialNamespaceTypedOutputShape;
void sequentialCompileNamespaceTypedOutputShape;
void sequentialArrayCompileNamespaceTypedOutputShape;
void sequentialShapeConstraints;
void sequentialParameterLayout;
void sequentialParameterLayoutSignature;
void sequentialCanCompile;
void sequentialMethodExplanationKind;
void sequentialCompilePlanKind;
void sequentialMethodCompilePlanKind;
void sequentialExplanationSignature;
void sequentialCompilePlanSignature;
void sequentialCompileExplanationKernelPlan;
void sequentialExplanationIrSignature;
void sequentialExplanationKernelPlanSignature;
void sequentialExplanationParameterNames;
void sequentialExplanationParameterInfos;
void sequentialEvidence;
void sequentialNativeProgramInspection;
void sequentialNativeProgramInspectionSource;
void sequentialNativeCompilerAuthority;
void sequentialNativeInspectionExecutionSupported;
void sequentialNativeCommandStencilHash;
void sequentialProgramIr;
void sequentialProgramIrSignature;
void sequentialProgramKernelPlan;
void sequentialProgramKernelPlanSignature;
void typedSequentialInstanceProgram;
void typedSequentialSupport;
void typedSequentialCompilePlan;
void typedBatchedSequentialProgram;
void typedBatchedSequentialSupport;
void typedBatchedSequentialPlan;
void sequentialCompatibility;
void sequentialAccepted;
void sequentialOutputShape;
void steppedTensorShape;
void stepSupport;
void typedStepSupport;
void typedStepCompilePlan;
void stepIr;
void stepKernelPlan;
void stepOutputShape;
void rootNativeApiContract;
void rootNativeApiContractOwner;
void rootNativeApiContractParts;
void rootNativeApiContractSignature;
void rootNativeApiContractSelfSignature;
void rootNativeApiContractMissing;
void rootNativeApiTypedMissing;
void rootNativeApiTypedSortedKeys;
void rootNativeApiTypedExtraKeys;
void rootNativeApiRequiredExports;
void rootNativeApiPackageSpineExports;
void activationChainSupport;
void activationChainKernelPlan;
void activationChainOp;
void activationChainFusedEdges;
void tanhModule;
void tanhModuleClass;
void tanhModuleForwardTensor;
void tanhModuleClassForwardTensor;
void literalRootRoundTensor;
void literalStaticTruncTensor;
void literalInstanceRoundTensor;
void literalRootExpm1Tensor;
void literalStaticLog1pTensor;
void literalInstanceLog1pTensor;
void literalRootRsqrtTensor;
void literalStaticRsqrtTensor;
void typedTanhModuleProgram;
void tanhModuleSupport;
void typedTanhModuleSupport;
void typedTanhModuleCompilePlan;
void minReductionModule;
void compiledMinReductionSupport;
void typedMinReductionSupport;
void typedMinReductionCompilePlan;
void compiledMinReductionKernelPlan;
void typedMinReductionProgram;
void dropoutModule;
void dropoutModuleForwardTensor;
void trainingDropoutSupport;
void trainingDropoutExplanation;
void trainingDropoutMethodExplanation;
void trainingDropoutCompileExplanation;
void trainingDropoutNamespaceCompileExplanation;
void trainingDropoutCompilePlan;
void trainingDropoutMethodCompilePlan;
void trainingDropoutDiagnostics;
void trainingDropoutIr;
void trainingDropoutExplanationKernelPlan;
void trainingDropoutExplanationDiagnostics;
void trainingDropoutMethodExplanationDiagnostics;
void trainingDropoutCompilePlanDiagnostics;
void trainingDropoutMethodCompilePlanDiagnostics;
void trainingDropoutExplanationParameterNames;
void trainingDropoutExplanationParameterInfos;
void trainingDropoutExplanationReason;
void trainingDropoutExplanationSignature;
void trainingDropoutCompilePlanSignature;
void trainingDropoutDiagnosticStage;
void trainingDropoutInputLen;
void trainingDropoutOutputLen;
void trainingDropoutInputShape;
void trainingDropoutOutputShape;
void currentRuntimeFeatures;
void runtimeNativeEagerMatmulFeature;
void runtimeNativeEagerElementwiseFeature;
void runtimeNativeEagerReduceFeature;
void runtimeNativeEagerConv2dFeature;
void runtimeNativeEagerPool2dFeature;
void runtimeActivationChainFeature;
void literalEqTensor;
void literalNeTensor;
void literalLtTensor;
void literalLeTensor;
void literalGtTensor;
void literalGeTensor;
void literalSameShapeTensor;
void literalTensorAddSameShape;
void literalTensorSubSameShape;
void literalTensorEqSameShape;
void literalTensorIsCloseSameShape;
void literalTensorMaximumSameShape;
void literalTensorWhereSameShape;
void literalTensorAddInPlace;
void literalTensorSubInPlace;
void literalTensorMulInPlace;
void literalTensorDivInPlace;
void literalTensorBroadcastAdd;
void literalTensorBroadcastEq;
void literalTensorBroadcastIsClose;
void literalTensorBroadcastWhere;
void literalPowTensor;
void literalRootPowTensor;
void literalRootAddTensor;
void literalRootAddSameShapeTensor;
void literalRootIsCloseTensor;
void literalRootMulSameShapeTensor;
void literalRootSubTensor;
void literalStaticMulTensor;
void literalStaticMulBroadcastTensor;
void literalStaticIsCloseBroadcastTensor;
void literalStaticDivSameShapeTensor;
void literalStaticDivTensor;
void literalRootCloneTensor;
void literalStaticDetachTensor;
void literalRootReshapeTensor;
void literalRootInferReshapeTensor;
void literalStaticViewTensor;
void literalStaticInferViewTensor;
void literalFlattenTensor;
void literalRootFlattenTensor;
void literalStaticFlattenTensor;
void literalFlatten3dTensor;
void literalRootFlatten3dMiddleTensor;
void literalStaticFlatten3dLeadingTensor;
void literalUnsqueezeTensor;
void literalRootUnsqueezeTensor;
void literalStaticUnsqueezeTensor;
void literalUnsqueeze3dMiddleTensor;
void literalTransposeTensor;
void literalRootTransposeTensor;
void literalStaticTransposeTensor;
void literalTranspose3dOuterTensor;
void literalFlipTensor;
void literalRootFlipTensor;
void literalStaticFlipTensor;
void literalSelectRowsTensor;
void literalRootSelectColsTensor;
void literalStaticSelectColsTensor;
void literalSelect3dLeadingTensor;
void literalRootSelect3dTrailingTensor;
void literalSliceColsTensor;
void literalRootSliceRowsTensor;
void literalStaticSlice3dTensor;
void literalNarrowRowsTensor;
void literalRootNarrowColsTensor;
void literalStaticNarrowColsTensor;
void literalNarrow3dMiddleTensor;
void literalStaticNarrow3dTrailingTensor;
void literalRootEqTensor;
void literalStaticGeTensor;
void literalStaticGeSameShapeTensor;
void literalRootNegTensor;
void literalRootNegativeTensor;
void literalStaticNegativeTensor;
void literalInstanceNegativeTensor;
void literalRootReciprocalTensor;
void literalStaticReciprocalTensor;
void literalInstanceReciprocalTensor;
void literalAsTensor;
void literalAsTensorCamel;
void literalAsArrayTensor;
void literalFromNumpyTensor;
void literalFromNumpyCamelTensor;
void literalStaticSqrtTensor;
void literalRootReluTensor;
void literalStaticSigmoidTensor;
void literalRootTanhTensor;
void literalStaticTanhTensor;
void literalTensorTanh;
void literalRootSumTensor;
void literalRootMeanTensor;
void literalStaticMaxTensor;
void literalRootArgmaxTensor;
void literalSum3dDim0Tensor;
void literalRootMean3dTrailingTensor;
void literalStaticMax4dDim2Tensor;
void literalTensorDim;
void literalTensorNdimension;
void literalTensorNumel;
void literalTensorSize;
void literalTensorSizeDim;
void literalTensorStride;
void literalTensorStrideDim;
void literalTensorStorageOffset;
void literalTensorIsContiguous;
void literalRootSoftmaxTensor;
void literalTensorIsContiguousAlias;
void literalTensorContiguous;
void literalTensorElementSizeAlias;
void literalTensorNbytes;
void literalRootSoftmaxDimAliasTensor;
void literalRootSoftmaxSnakeDimTensor;
void literalInstanceSoftmaxDimAliasTensor;
void literalInstanceSoftmaxSnakeDimTensor;
void literalStaticSoftmaxSnakeDimTensor;
void literalStaticSoftmaxDimTensor;
void literalStaticLogSoftmaxTensor;
void literalStaticLogSoftmaxDimAliasTensor;
void literalStaticLogSoftmaxSnakeTensor;
void literalStaticLogSoftmaxSnakeDimTensor;
void literalRootLogSoftmaxTensor;
void literalRootLogSoftmaxDimAliasTensor;
void literalRootLogSoftmaxSnakeTensor;
void literalRootLogSoftmaxSnakeDimTensor;
void literalInstanceLogSoftmaxDimAliasTensor;
void literalInstanceLogSoftmaxSnakeTensor;
void literalInstanceLogSoftmaxSnakeDimTensor;
void literalInstanceLogsumexpTensor;
void literalInstanceLogsumexpDimTensor;
void literalRootLogsumexpTensor;
void literalStaticLogSumExpTensor;
void literalRootLogSumExpTensor;
void literalProdTensor;
void literalProdDimTensor;
void literalProdDimAliasTensor;
void literalRootProdTensor;
void literalStaticProdTensor;
void literalCumsumTensor;
void literalRootCumsumTensor;
void literalStaticCumsumTensor;
void literalVarianceTensor;
void literalVarDimTensor;
void literalRootVarianceTensor;
void literalStaticStdTensor;
void literalRootStdTensor;
void literalNormTensor;
void literalNormDimTensor;
void literalRootNormTensor;
void literalStaticNormTensor;
void literalRootClampTensor;
void literalStaticClipTensor;
void literalMatmulTensor;
void literalRootMatmulTensor;
void literalRootMmTensor;
void literalStaticMatmulTensor;
void literalStaticMmTensor;
void literalDotTensor;
void literalRootDotTensor;
void literalStaticDotTensor;
void literalTraceTensor;
void literalRootTraceTensor;
void literalStaticTraceTensor;
void literalDiagonalTensor;
void literalRootDiagonalTensor;
void literalStaticDiagonalTensor;
void repeatModuleForward;
void repeatModuleOutputShape;
void repeatModuleTrace;
void repeatModuleIr;
void repeatModuleSupport;
void tileModuleForward;
void tileModuleOutputShape;
void diagonalModuleForward;
void diagonalModuleOutputShape;
void diagonalModuleTrace;
void diagonalModuleIr;
void diagonalModuleSupport;
void literalBmmTensor;
void literalRootBmmTensor;
void literalStaticBmmTensor;
void literalMaximumTensor;
void literalRootMaximumSameShapeTensor;
void literalMinimumTensor;
void literalStaticMinimumSameShapeTensor;
void literalWhereTensor;
void literalRootWhereTensor;
void literalRootWhereSameShapeTensor;
void literalAnyTensor;
void literalAnyDimTensor;
void literalAllTensor;
void literalAllDimTensor;
void logSoftmaxModuleForwardTensor;
void dim0SoftmaxForwardTensor;
void layerNormModuleForwardTensor;
void rmsNormModuleForwardTensor;
void batchNormFactory;
void batchNormSnake;
void batchNormModuleForwardTensor;
void batchNormEvalForwardTensor;
void batchNormState;
void batchNormRunningMean;
void batchNormRunningVar;
void batchNormTracked;
void batchNormTraceOutputShape;
void batchNormTracedOutputShape;
void batchNormSupport;
void conv2dConfig;
void conv2dSpatialConfig;
void conv2dFactory;
void conv2dForwardTensor;
void conv2dSpatialForwardTensor;
void conv2dBatchForwardTensor;
void conv2dState;
void conv2dSupport;
void conv2dSequentialSupport;
void conv2dSequentialIrOp;
void conv2dBatchedProgram;
void conv2dBatchedBindings;
void maxPool2dConfig;
void maxPool2dFactory;
void maxPool2dForwardTensor;
void maxPool2dTraceOutputShape;
void maxPool2dTracedOutputShape;
void maxPool2dSupport;
void maxPool2dIrOp;
void directMaxPool2dSupport;
void directMaxPool2dProgram;
void directMaxPool2dNamespaceProgram;
void directMaxPool2dNnNamespaceProgram;
void batchedMaxPool2dProgram;
void batchedMaxPool2dBindings;
void fixedMaxPool2dSupport;
void fixedMaxPool2dKernel;
void avgPool2dConfig;
void avgPool2dFactory;
void avgPool2dForwardTensor;
void avgPool2dTraceOutputShape;
void avgPool2dTracedOutputShape;
void avgPool2dSupport;
void avgPool2dIrOp;
void directAvgPool2dSupport;
void directAvgPool2dProgram;
void directAvgPool2dNamespaceProgram;
void directAvgPool2dNnNamespaceProgram;
void batchedAvgPool2dProgram;
void batchedAvgPool2dBindings;
void fixedAvgPool2dSupport;
void fixedAvgPool2dKernel;
void typedRmsNormSupport;
void typedRmsNormCompilePlan;
void typedElidedShapeChainProgram;
void typedRmsNormProgram;
void logSoftmaxSupport;
void logSoftmaxDefaultTensorOutputShape;
void logSoftmaxTensorOutputShape;
void logSoftmaxDefaultReadbackShape;
void logSoftmaxReadbackShape;
void minReductionKind;
void minReductionOutput;
void minReductionOutputShape;
void minReductionSupport;
void minReductionIr;
void minReductionIrOp;
void argmaxReductionKind;
void argmaxTensorOutput;
void argmaxTensorDimOutput;
void argmaxReduction;
void prodReductionKind;
void prodReduction;
void prodReductionOutput;
void prodReductionOutputShape;
void prodReductionSupport;
void prodReductionIr;
void prodReductionIrOp;
void argmaxReductionOutput;
void argmaxReductionOutputShape;
void argmaxReductionSupport;
void argmaxReductionIr;
void argmaxReductionIrOp;
void argminReductionKind;
void argminTensorOutput;
void argminTensorDimOutput;
void argminReduction;
void argminReductionOutput;
void argminReductionOutputShape;
void argminReductionSupport;
void argminReductionIr;
void argminReductionIrOp;
void literalRootIsfiniteTensor;
void literalStaticIsinfTensor;
void literalInstanceIsnanTensor;
void literalRootIsnanTensor;
void literalRootIsinfTensor;
void literalRootFloorTensor;
void literalStaticCeilTensor;
void literalInstanceFloorTensor;
void literalRootSinTensor;
void literalStaticCosTensor;
void literalInstanceSinTensor;
void literalRootTanTensor;
void literalStaticTanTensor;
void moduleStateSnapshot;
void moduleStateSnapshotSignature;
void checkpointAdamW;
void checkpointModelParameterNames;
void prefixedModuleStateSnapshot;
void prefixedModuleStateSnapshotSignature;
void publicOptimNamespace;
void optimizerStateSnapshot;
void optimizerStateSnapshotAlias;
void optimizerStateSnapshotSignature;
void adamOptimizerKind;
void adamWOptimizerKind;
void adamOptimizerTyped;
void adamWStateSnapshot;
void adamWStateSnapshotSignature;
void optimizerConfigSnapshot;
void optimizerConfigSnapshotSignature;
void optimizerConfigStep;
void adamWConfigSnapshot;
void adamWConfigSnapshotSignature;
void groupedOptimizer;
void groupedOptimizerKind;
void groupedOptimizerConfig;
void groupedOptimizerParamGroups;
void groupedOptimizerParamGroupsDirect;
void groupedOptimizerParamGroupsAlias;
void groupedOptimizerFirstGroupLr;
void groupedOptimizerSecondGroupWeightDecay;
void groupedOptimizerWithAddedGroup;
void groupedOptimizerWithAddedGroupAlias;
void optimizerConfigFromNamespace;
void optimizerWithUpdatedLr;
void optimizerWithSnakeLr;
void optimizerStateFromNamespace;
void optimizerStateFromNamespaceAlias;
void optimizerStateSnapshotPredicate;
void optimizerStateSnapshotRequire;
void optimizerStateSnapshotAssert;
void optimizerStateSnapshotSignatureFromNamespace;
void optimizerStateSnapshotSignatureMatch;
void optimizerLoadedFromNamespace;
void optimizerLoadedFromNamespaceAlias;
void optimizerConfigSnapshotPredicate;
void optimizerConfigSnapshotRequire;
void optimizerConfigSnapshotAssert;
void optimizerConfigSnapshotSignatureFromNamespace;
void optimizerConfigSnapshotSignatureMatch;
void stepSchedulerAlias;
void exponentialScheduler;
void exponentialSchedulerAlias;
void exponentialSchedulerClass;
void cosineScheduler;
void cosineSchedulerAlias;
void cosineSchedulerClass;
void plateauScheduler;
void plateauSchedulerAlias;
void plateauSchedulerClass;
void nestedStepScheduler;
void nestedStepSchedulerFactory;
void nestedExponentialScheduler;
void nestedExponentialSchedulerFactory;
void nestedCosineScheduler;
void nestedCosineSchedulerFactory;
void nestedPlateauScheduler;
void nestedPlateauSchedulerFactory;
void schedulerLr;
void schedulerLastLr;
void schedulerStateSnapshot;
void schedulerStateSnapshotSignature;
void schedulerStateKind;
void schedulerStateOptimizerKind;
void schedulerStateStepSize;
void schedulerStateStepSizeAlias;
void cosineSchedulerState;
void cosineSchedulerTMaxAlias;
void cosineSchedulerEtaMinAlias;
void plateauSchedulerLr;
void plateauSchedulerState;
void plateauSchedulerMode;
void plateauSchedulerBadEpochsAlias;
void checkpointInspectionSchedulerTMax;
void checkpointInspectionSchedulerTMaxAlias;
void checkpointInspectionSchedulerEtaMin;
void checkpointInspectionSchedulerEtaMinAlias;
void cosineSchedulerCheckpoint;
void cosineSchedulerInspection;
void cosineSchedulerInspectionTMax;
void cosineSchedulerInspectionEtaMin;
void schedulerStateSnapshotPredicate;
void schedulerStateSnapshotRequire;
void schedulerStateSnapshotAssert;
void schedulerStateSnapshotSignatureFromNamespace;
void schedulerStateSnapshotSignatureMatch;
void schedulerStateSnakeDict;
void publicLossNamespace;
void rawLossInput;
void rawMseLoss;
void rawMseSumLoss;
void rawMeanSquaredErrorLoss;
void rawCrossEntropyLoss;
void rawCrossEntropyLossSnake;
void rawCrossEntropySumLoss;
void publicNnNamespace;
void nnFunctional;
void nnFunctionalLossSubset;
void nnFunctionalAlias;
void nnLossConstructorName;
void nnLossConstructors;
void functionalReluTensor;
void functionalReluInplaceFalseTensor;
void functionalGeluTensor;
void functionalSiluTensor;
void functionalSigmoidTensor;
void functionalTanhTensor;
void functionalSoftmaxTensor;
void functionalSoftmaxDimTensor;
void functionalSoftmaxCamelDimTensor;
void functionalLogSoftmaxTensor;
void functionalLogSoftmaxDimTensor;
void functionalLogSoftmaxCamelDimTensor;
void functionalDropoutTensor;
void functionalDropoutBooleanTensor;
void functionalDropoutInplaceFalseTensor;
void functionalFlattenTensor;
void functionalFlattenRangeTensor;
void functionalLinear1dTensor;
void functionalLinear2dTensor;
void functionalLinear3dTensor;
void functionalLinear4dTensor;
void functionalNormalizeTensor;
void functionalOneHotTensor;
void functionalOneHotAliasTensor;
void functionalOneHotGridTensor;
void functionalEmbeddingTensor;
void functionalEmbeddingGridTensor;
void functionalLayerNormTensor;
void functionalLayerNormPositionalTensor;
void functionalRmsNormTensor;
void functionalRmsNormPositionalTensor;
void functionalBatchNormTensor;
void functionalBatchNormSnakeTensor;
void rawFunctionalMseLoss;
void rawFunctionalMseLossAlias;
void rawFunctionalMseSumLoss;
void rawFunctionalL1Loss;
void rawFunctionalL1LossAlias;
void rawFunctionalL1SumLoss;
void rawFunctionalHuberLoss;
void rawFunctionalHuberLossAlias;
void rawFunctionalSmoothL1Loss;
void rawFunctionalSmoothL1LossAlias;
void rawFunctionalBceLoss;
void rawFunctionalBceWithLogitsLoss;
void functionalNllLossTensor;
void rawFunctionalCrossEntropyLossSnake;
void rawFunctionalCrossEntropySumLoss;
void rawFunctionalAliasCrossEntropyLoss;
void rootFunctionalAliasSoftmaxTensor;
void rawNllLoss;
void rawNllLossSnake;
void rawNllSumLoss;
void rawMeanAbsoluteErrorLoss;
void rawL1Loss;
void rawL1SumLoss;
void rawHuberLoss;
void rawSmoothL1Loss;
void rawSmoothL1LossAlias;
void rawBceLoss;
void rawBinaryCrossEntropyLoss;
void rawBinaryCrossEntropyLossAlias;
void rawBceWithLogitsLoss;
void rawBceWithLogitsLossSnake;
void rawBinaryCrossEntropyWithLogitsLoss;
void rawBinaryCrossEntropyWithLogitsLossAlias;
void rawBinaryCrossEntropyWithLogitsSumLoss;
void classTargetBuffer;
void trainingLogits;
void tensorCrossEntropyLoss;
void trainingLogProbabilities;
void tensorNllLoss;
void tensorNllLossSnake;
void classificationAccuracy;
void classificationAccuracyCamel;
void classificationAccuracySnake;
void classPredictions;
void classPredictionsSnake;
void predictedClasses;
void predictedClassesSnake;
void topKAccuracy;
void topKAccuracySnake;
void confusionMatrix;
void confusionMatrixSnake;
void classificationReport;
void classificationReportSnake;
void binaryAccuracy;
void binaryAccuracySnake;
void binaryLogitsAccuracy;
void binaryLogitsAccuracySnake;
void mseLossModule;
void mseLossModuleAlias;
void mseLossModuleClass;
void nnMseLossModuleClass;
void mseLossSumModule;
void mseLossSumReduction;
void nnMseLossSumModuleClass;
void nnMseLossSumReduction;
void mseLossModuleTensor;
void mseLossModuleNumber;
void mseLossModuleClassNumber;
void nnMseLossModuleClassNumber;
void mseLossModuleDunderNumber;
void mseLossSumModuleNumber;
void nnMseLossSumModuleClassNumber;
void l1LossModule;
void l1LossModuleAlias;
void l1LossModuleClass;
void nnL1LossModuleClass;
void nnL1LossSumModuleClass;
void nnL1LossSumReduction;
void l1LossModuleTensor;
void l1LossModuleNumber;
void l1LossModuleClassNumber;
void nnL1LossModuleClassNumber;
void l1LossModuleDunderNumber;
void huberLossModule;
void huberLossModuleAlias;
void huberLossModuleClass;
void nnHuberLossModuleClass;
void nnHuberLossSumModuleClass;
void nnHuberLossSumReduction;
void huberLossModuleTensor;
void huberLossModuleNumber;
void huberLossModuleClassNumber;
void nnHuberLossModuleClassNumber;
void huberLossModuleDunderNumber;
void smoothL1LossModule;
void smoothL1LossModuleAlias;
void smoothL1LossModuleClass;
void nnSmoothL1LossModuleClass;
void nnSmoothL1LossSumModuleClass;
void nnSmoothL1LossSumReduction;
void smoothL1LossModuleTensor;
void smoothL1LossModuleNumber;
void smoothL1LossModuleClassNumber;
void nnSmoothL1LossModuleClassNumber;
void smoothL1LossModuleDunderNumber;
void bceLossModule;
void bceLossModuleAlias;
void bceLossModuleClass;
void nnBceLossModuleClass;
void nnBceLossSumModuleClass;
void nnBceLossSumReduction;
void bceLossModuleTensor;
void bceLossModuleNumber;
void bceLossModuleClassNumber;
void nnBceLossModuleClassNumber;
void bceLossModuleDunderNumber;
void bceWithLogitsLossModule;
void bceWithLogitsLossModuleAlias;
void bceWithLogitsLossModuleClass;
void nnBceWithLogitsLossModuleClass;
void nnBceWithLogitsLossSumModuleClass;
void nnBceWithLogitsLossSumReduction;
void bceWithLogitsLossModuleTensor;
void bceWithLogitsLossModuleNumber;
void bceWithLogitsLossModuleClassNumber;
void nnBceWithLogitsLossModuleClassNumber;
void bceWithLogitsLossModuleDunderNumber;
void crossEntropyLossModule;
void crossEntropyLossModuleAlias;
void crossEntropyLossModuleClass;
void nnCrossEntropyLossModuleClass;
void crossEntropyLossSumModule;
void crossEntropyLossSumReduction;
void nnCrossEntropyLossSumModuleClass;
void nnCrossEntropyLossSumReduction;
void crossEntropyLossModuleTensor;
void crossEntropyLossModuleNumber;
void crossEntropyLossModuleClassNumber;
void nnCrossEntropyLossModuleClassNumber;
void crossEntropyLossModuleDunderNumber;
void nllLossModule;
void nllLossModuleAlias;
void nllLossModuleClass;
void nnNllLossModuleClass;
void nllLossSumModule;
void nllLossSumReduction;
void nnNllLossSumModuleClass;
void nnNllLossSumReduction;
void nllLossModuleTensor;
void nllLossModuleNumber;
void nllLossModuleClassNumber;
void nnNllLossModuleClassNumber;
void nllLossModuleDunderNumber;
void trainingLoss;
void trainingLossStep;
void trainingLossStepSnake;
void namespaceTensorLoss;
void publicTrainNamespace;
void observedTrainingNorm;
void observedTrainingNormSnake;
void compiledTrainingForAlias;
void compiledTrainingForSnakeAlias;
void rootForTrainingAlias;
void rootForTrainingSnakeAlias;
void zgmlForTrainingAlias;
void moduleForTrainingAlias;
void moduleForTrainingSnakeAlias;
void createdCheckpoint;
void checkpointModelState;
void checkpointOptimizerState;
void checkpointSchedulerState;
void checkpointInspection;
void checkpointInspectionSignature;
void checkpointInspectionNames;
void checkpointInspectionOptimizerStep;
void checkpointInspectionHasScheduler;
void checkpointInspectionSchedulerStep;
void checkpointInspectionSchedulerLastLr;
void checkpointModelParameterInfo;
void checkpointModelParameterInfoSignature;
void checkpointOptimizerEntryInfo;
void checkpointOptimizerEntryInfoSignature;
void checkpointModelParameterInfoByName;
void checkpointModelParameterInfoByIndex;
void checkpointOptimizerEntryInfoByIndex;
void checkpointOptimizerEntryInfoByName;
void checkpointMetadata;
void checkpointSerializedSnapshot;
void checkpointParsedSnapshot;
void checkpointSerializedSnapshotAlias;
void checkpointParsedSnapshotAlias;
void publicDataNamespace;
void customDataset;
void customDatasetLoader;
void customTorchDataset;
void customSequentialSampler;
void customRandomSampler;
void customReplacementSampler;
void customBatchSampler;
void customSamplerLoader;
void customBatchSamplerLoader;
void customCollateFn;
void customCollateLoader;
void customObjectCollateFn;
void customObjectCollateLoader;
void customObjectCollateBatch;
void defaultCollateOptions;
void defaultCollateBatch;
void defaultCollateBatchAlias;
void tensorDatasetSample;
void tensorDatasetSampleAlias;
void tensorDatasetSampleAt;
void tensorDatasetSampleDunder;
void tensorDatasetIterableSample;
void tensorDatasetDunderIterSample;
void typedTensorDatasetSampleInput;
void typedTensorDatasetSampleTarget;
void typedTensorDatasetAtInput;
void typedTensorDatasetDunderTarget;
void typedTensorDatasetIterInput;
void typedTensorDatasetDunderIterTarget;
void typedTensorDatasetBatchInput;
void typedTensorDatasetBatchTarget;
void typedTensorDatasetIterableBatchInput;
void splitOptions;
void tensorDatasetSplitPair;
void tensorDatasetSplitPairAlias;
void tensorDatasetIterableSplit;
void tensorDatasetSubset;
void tensorDatasetTake;
void tensorDatasetConcat;
void tensorDatasetConcatAlias;
void tensorDatasetMapped;
void tensorDatasetMappedAlias;
void typedTensorDatasetSplitPair;
void typedTensorDatasetIterableSplit;
void typedTensorDatasetSplitInput;
void typedTensorDatasetSplitTarget;
void typedTensorDatasetSplitBatchInput;
void typedTensorDatasetSplitIterableBatchInput;
void typedTensorDatasetConcat;
void typedTensorDatasetConcatInput;
void typedTensorDatasetConcatBatchTarget;
void typedTensorDatasetConcatIterableBatchTarget;
void typedTensorDatasetLoader;
void typedTensorDatasetLoaderBatchInput;
void typedTensorDatasetLoaderGetInput;
void typedTensorDatasetLoaderDunderTarget;
void tensorDatasetLen;
void tensorDatasetDunderLen;
void tensorDatasetSize;
void tensorDatasetBatch;
void tensorDatasetLoader;
void classTensorDataset;
void classDataLoader;
void typedTensorDatasetLoaderIterInput;
void tensorDatasetDataloader;
void tensorDatasetLoaderSampleCount;
void tensorDatasetLoaderSampleCountAlias;
void tensorDatasetLoaderBatchSize;
void tensorDatasetLoaderBatchSizeAlias;
void tensorDatasetLoaderBatchCount;
void tensorDatasetLoaderBatchCountAlias;
void tensorDatasetLoaderLen;
void tensorDatasetLoaderDunderLen;
void tensorDatasetLoaderSize;
void tensorDatasetDataloaderDropLast;
void tensorDatasetLoaderDropLastAlias;
void fitDataEvidence;
void modelFirstFitOptions;
void modelFirstFitEvidence;
void fitDataBatchCount;
void fitDataBatchCountAlias;
void fitDataSampleCount;
void fitDataSampleCountAlias;
void fitDataStopReason;
void fitDataBestLoss;
void fitEarlyStoppingOptions;
void fitDataEvidenceSnake;
void predictEvidence;
void predictEvidenceAlias;
void predictOutputs;
void predictLastOutput;
void predictLastOutputAlias;
void publicCheckpointNamespace;
void restoreTargets;
void prefixedCheckpointRestoreTargets;
void checkpointModuleState;
void checkpointModuleStateAlias;
void checkpointOptimizerStateFromNamespace;
void checkpointOptimizerStateAlias;
void checkpointSchedulerStateFromNamespace;
void checkpointSchedulerStateAlias;
void checkpointOptimizerStepFromNamespace;
void loadedOptimizer;
void linearSupport;
void linearSupportAlias;
void linearRequiredSupport;
void linearRequiredSupportAlias;
void linearSupportInputShape;
void linearSupportOutputShape;
void linearCanCompile;
void compileNamespace;
void compileNamespaceTrace;
void compileNamespaceAnalysis;
void compileNamespaceAnalysisPredicate;
void compileNamespaceAnalysisRequired;
void compileNamespaceAnalysisAsserted;
void compileNamespaceAnalysisSignature;
void compileNamespaceAnalysisSignatureFromNamespace;
void compileNamespaceAnalysisSignatureMatch;
void compileNamespaceSupport;
void compileNamespaceSupportAlias;
void compileNamespaceRequiredSupport;
void compileNamespaceRequiredSupportAlias;
void nnNamespaceRequiredSupport;
void nnNamespaceRequiredSupportAlias;
void nnNamespaceRequiredCompilePlan;
void nnNamespaceRequiredCompilePlanAlias;
void nnNamespaceAssertedCompilePlan;
void nnNamespaceAssertedCompilePlanAlias;
void compileNamespaceAssertedSupport;
void compileNamespaceAssertedSupportAlias;
void compileNamespaceRequiredCompilePlan;
void compileNamespaceRequiredCompilePlanAlias;
void compileNamespaceAssertedCompilePlan;
void compileNamespaceAssertedCompilePlanAlias;
void compileNamespaceCanCompile;
void compileNamespaceCanCompileAlias;
void compiledInference;
void compiledInferenceAlias;
void zgmlCompiledInference;
void zgmlCompiledInferenceAlias;
void compiledInferenceForward;
void compiledInferenceStepTensor;
void compiledInferenceInto;
void compiledInferencePrepared;
void compiledInferenceExplanation;
void compiledInferencePreflight;
void compiledInferenceSupport;
void compiledInferenceInputShape;
void compiledInferenceOutputShape;
void compiledInferenceKernelPlan;
void compiledInferenceCompilerSignatures;
void compileNamespaceExplanation;
void compileNamespaceTypedOutputShape;
void compileNamespaceTypedOutputShapeAlias;
void compileNamespaceProgram;
void typedLinearProgram;
void typedBatchedLinearProgram;
void typedLinearSupport;
void typedBatchedLinearSupport;
void typedLinearRequiredSupport;
void typedLinearSupportInputShape;
void typedLinearSupportOutputShape;
void typedLinearCompilePlan;
void typedLinearCompilePlanInputShape;
void typedLinearCompilePlanOutputShape;
void typedLinearNamespaceSupport;
void typedLinearNamespaceCompilePlan;
void typedCompileNamespaceProgram;
void typedLinearInstanceProgram;
void typedLinearProgramInputShape;
void typedLinearProgramOutputShape;
void typedLinearProgramExecutionPlan;
void typedLinearProgramExecutionPlanAlias;
void typedLinearProgramRequiredExecutionPlan;
void typedLinearProgramExecutionPlanInputShape;
void typedLinearProgramExecutionPlanOutputShape;
void typedLinearModuleBindings;
void typedBatchedLinearModuleBindings;
void typedLinearModuleBindingsAlias;
void typedLinearNamespaceBindings;
void typedLinearPlacedBindings;
void typedLinearNamespacePlacedBindings;
void typedLinearModuleBindingPlan;
void typedLinearModuleBindingPlanAlias;
void typedLinearModuleRequiredBindingPlan;
void typedLinearModuleBindingPlanInputShape;
void typedLinearModuleBindingPlanOutputShape;
void typedLinearProgramBindingPlan;
void typedLinearProgramBindingPlanAlias;
void typedLinearProgramRequiredBindingPlan;
void typedLinearProgramBindingPlanInputShape;
void typedLinearProgramBindingPlanOutputShape;
void typedLinearInstanceProgramOutputShape;
void typedRawLinearProgramBindings;
void typedRawLinearProgramBindingsInputShape;
void typedRawLinearProgramBindingsOutputShape;
void typedProgramInputBinding;
void typedProgramOutputBinding;
void typedLinearStepParams;
void typedRawLinearBindingPlan;
void typedRawLinearProgramBindingPlan;
void typedRawLinearProgramSession;
void programNamespace;
void programNamespacePath;
void sessionNamespace;
void sessionNamespacePath;
void stepParamsNamespace;
void stepParamsNamespacePath;
void linearProgramInputLen;
void linearProgramOutputLen;
void linearProgramInputByteLength;
void linearProgramOutputByteLength;
void linearProgramWeightsLen;
void linearProgramWeightsByteLength;
void linearProgramBiasLen;
void linearProgramBiasByteLength;
void linearProgramParameterLen;
void linearProgramParameterByteLength;
void linearProgramInspectionKind;
void linearProgramInspectionSignature;
void linearRequirementsKind;
void linearRequirementsSignature;
void linearProgramBufferSizing;
void linearProgramBufferSizingKind;
void linearProgramBufferSizingSignature;
void linearProgramBufferSizingModelKind;
void linearProgramMatchesBufferSizingSignature;
void linearProgramCapabilitiesKind;
void linearProgramExecutionPlanRequirementsKind;
void linearProgramRequiredExecutionPlan;
void linearProgramRequiredModuleExecutionPlan;
void linearProgramRequiredExecutionPlanAlias;
void linearProgramRequiredModuleExecutionPlanAlias;
void linearProgramAcceptedExecutionPlan;
void linearProgramRequiredEvidencePlan;
void linearProgramAssertedEvidencePlan;
void linearProgramAssertedEvidencePlanAlias;
void linearProgramMatchedEvidencePlan;
void linearProgramInspectionAcceptedExecutionPlan;
void linearProgramInspectionRequiredEvidencePlan;
void linearBufferLayoutKind;
void linearBufferLayoutSignature;
void linearProgramBindingPlan;
void linearProgramRequiredBindingPlan;
void linearProgramRequiredBindingPlanAlias;
void linearProgramAcceptedBindingPlan;
void linearProgramRequiredEvidenceBindingPlan;
void linearProgramAssertedEvidenceBindingPlan;
void linearProgramAssertedEvidenceBindingPlanAlias;
void linearProgramMatchedEvidenceBindingPlan;
void linearProgramInspectionAcceptedBindingPlan;
void linearProgramInspectionRequiredEvidenceBindingPlan;
void linearProgramBindingPlanKind;
void linearProgramBindingPlanSignature;
void linearProgramBindingPlanAccepted;
void linearProgramBindingPlanMode;
void linearProgramBindingPlanDiagnostics;
void linearProgramBindingPlanBufferLayout;
void linearProgramRejectedBindingPlan;
void linearProgramRejectedBindingPlanSignature;
void linearProgramRejectedBindingReason;
void linearCapabilitySignature;
void linearMatchesCapabilitySignature;
void linearRuntimeProfileAlias;
void linearProgramRuntimeProfileKind;
void linearProgramRuntimeProfileSignature;
void linearProgramMatchesRuntimeProfileSignature;
void linearProgramMatchesRuntimeProfileSignatureAlias;
void inspectionNamespace;
void linearProgramRuntimeProfileExpectationKind;
void linearProgramRuntimeProfileNoFallback;
void linearProgramRuntimeProfileNoSync;
void linearProgramRuntimeProfileNoInvalidPatches;
void linearProgramRuntimeProfileRequired;
void linearProgramRuntimeProfileNoSyncRequired;
void linearProgramRuntimePatchValid;
void linearProgramRuntimeProfileHot;
void linearProgramRuntimeProfileExpectationFromProgram;
void linearProgramRuntimeProfileExpectationAliasFromProgram;
void linearProgramRuntimeProfileNoFallbackFromProgram;
void linearProgramRuntimeProfileNoFallbackAliasFromProgram;
void linearProgramRuntimeProfileNoSyncFromProgram;
void linearProgramRuntimeProfileNoSyncAliasFromProgram;
void linearProgramRuntimeProfileNoInvalidPatchesFromProgram;
void linearProgramRuntimeProfileNoInvalidPatchesAliasFromProgram;
void linearProgramRuntimeProfileRequiredFromProgram;
void linearProgramRuntimeProfileRequiredAliasFromProgram;
void linearProgramRuntimeProfileNoSyncRequiredFromProgram;
void linearProgramRuntimeProfileNoSyncRequiredAliasFromProgram;
void linearProgramRuntimePatchValidFromProgram;
void linearProgramRuntimePatchValidAliasFromProgram;
void linearProgramRuntimeProfileHotFromProgram;
void linearProgramRuntimeProfileHotAliasFromProgram;
void linearSessionInspectionKind;
void linearSessionInspectionSignature;
void linearModuleSessionViaBind;
void typedLinearSession;
void typedLinearModuleSession;
void typedLinearSessionInputShape;
void typedLinearSessionOutputShape;
void typedLinearSessionExecutionPlan;
void typedLinearSessionHotPathPlan;
void typedLinearSessionExecutionPlanInputShape;
void typedLinearSessionExecutionPlanOutputShape;
void linearSessionBufferLayoutKind;
void linearSessionBufferLayoutSignature;
void linearSessionInputLen;
void linearSessionOutputLen;
void linearSessionInputByteLength;
void linearSessionOutputByteLength;
void linearSessionWeightsLen;
void linearSessionWeightsByteLength;
void linearSessionBiasLen;
void linearSessionBiasByteLength;
void linearSessionParameterLen;
void linearSessionParameterByteLength;
void linearSessionBufferSizing;
void linearSessionBufferSizingKind;
void linearSessionBufferSizingSignature;
void linearSessionBufferSizingModelKind;
void linearSessionMatchesBufferSizingSignature;
void linearSessionRuntimeProfileAlias;
void linearSessionRuntimeProfileKind;
void linearSessionRuntimeProfileSignature;
void linearSessionMatchesRuntimeProfileSignature;
void linearSessionMatchesRuntimeProfileSignatureAlias;
void linearSessionRuntimeProfileExpectation;
void linearSessionRuntimeProfileNoFallback;
void linearSessionRuntimeProfileNoInvalidPatches;
void linearSessionRuntimeProfileExpectationFromSession;
void linearSessionRuntimeProfileExpectationAliasFromSession;
void linearSessionRuntimeProfileNoFallbackFromSession;
void linearSessionRuntimeProfileNoFallbackAliasFromSession;
void linearSessionRuntimeProfileNoSyncFromSession;
void linearSessionRuntimeProfileNoSyncAliasFromSession;
void linearSessionRuntimeProfileNoInvalidPatchesFromSession;
void linearSessionRuntimeProfileNoInvalidPatchesAliasFromSession;
void linearSessionRuntimeProfileRequiredFromSession;
void linearSessionRuntimeProfileRequiredAliasFromSession;
void linearSessionRuntimeProfileNoSyncRequiredFromSession;
void linearSessionRuntimeProfileNoSyncRequiredAliasFromSession;
void linearSessionRuntimePatchValidFromSession;
void linearSessionRuntimePatchValidAliasFromSession;
void linearSessionRuntimeProfileHotFromSession;
void linearSessionRuntimeProfileHotAliasFromSession;
void linearSessionCallProfileAlias;
void linearSessionCallProfileKind;
void linearSessionCallProfileSignature;
void linearSessionMatchesCallProfileSignature;
void linearSessionMatchesCallProfileSignatureAlias;
void linearSessionStepContractSignature;
void linearSessionStepContractInputByteLength;
void linearSessionStepContractOutputByteLength;
void linearSessionStepContractInputSlotName;
void linearSessionStepContractOutputSlotName;
void linearSessionStepContractInputSlotRole;
void linearSessionStepContractOutputSlotRole;
void linearSessionStepContractAcceptsNoInput;
void linearSessionStepContractRequiresInput;
void linearSessionStepContractDefaultOutput;
void linearSessionStepContractDefaultReadbackRequired;
void linearSessionStepContractDefaultAllocationFree;
void linearSessionStepContractNoOutputEffect;
void linearSessionMatchesStepContractSignature;
void linearSessionStepParamsPreflight;
void linearSessionStepParamsCompatibility;
void linearSessionStepParamsKind;
void linearSessionStepParamsStatus;
void linearSessionStepParamsContractKind;
void linearSessionStepParamsContractPosition;
void linearSessionStepParamsSignature;
void linearSessionStepParamsRejectionCode;
void linearSessionStepParamsStateEffect;
void linearSessionStepParamsAllocationFree;
void linearSessionStepParamsInputSource;
void linearSessionStepParamsInputOwnership;
void linearSessionStepParamsReadsInput;
void linearSessionStepParamsOutputTarget;
void linearSessionStepParamsOutputEffect;
void linearSessionStepParamsOutputOwnership;
void linearSessionStepParamsOutputReturnOwnership;
void linearSessionStepParamsWritesOutput;
void linearSessionStepParamsReadbackRequired;
void linearSessionStepParamsHotPath;
void linearSessionStepParamsHotPathStatus;
void linearSessionStepParamsInputElementType;
void linearSessionStepParamsOutputElementType;
void linearSessionStepParamsInputElementLength;
void linearSessionStepParamsOutputElementLength;
void linearSessionStepParamsInputByteLength;
void linearSessionStepParamsOutputByteLength;
void linearSessionStepParamsDiagnosticCode;
void linearSessionStepParamsDiagnosticExpectedLength;
void linearSessionStepParamsDiagnosticExpectedShape;
void stepParamsNamespaceAccepts;
void stepParamsNamespaceCanExecute;
void stepParamsNamespaceRequiredCanExecute;
void stepParamsNamespaceAcceptsAllocationFree;
void stepParamsNamespaceRequiredAllocationFree;
void stepParamsNamespaceAcceptsRuntimeOutputAllocationFree;
void stepParamsNamespaceRequiredRuntimeOutputAllocationFree;
void stepParamsNamespaceAcceptsNoReadback;
void stepParamsNamespaceRequiredNoReadback;
void stepParamsNamespaceAcceptsReadbackFree;
void stepParamsNamespaceRequiredReadbackFree;
void stepParamsNamespaceAcceptsHot;
void stepParamsNamespaceRequiredHot;
void stepParamsNamespaceMatchesSignature;
void stepParamsNamespaceMatchesCompatibility;
void linearSessionAcceptedExecutionPlan;
void linearSessionRequiredEvidencePlan;
void linearSessionAssertedEvidencePlan;
void linearSessionAssertedEvidencePlanAlias;
void linearSessionMatchedEvidencePlan;
void linearSessionInspectionAcceptedExecutionPlan;
void linearSessionInspectionRequiredEvidencePlan;
void linearSessionExecutionPlanInputOwnership;
void linearSessionExecutionPlanReadsInput;
void linearSessionExecutionPlanOutputOwnership;
void linearSessionExecutionPlanOutputReturnOwnership;
void linearSessionExecutionPlanWritesOutput;
void linearSessionExecutionPlanInputElementType;
void linearSessionExecutionPlanOutputElementType;
void linearSessionExecutionPlanInputElementLength;
void linearSessionExecutionPlanOutputElementLength;
void linearSessionRequiredExecutionPlan;
void linearSessionRequiredExecutionPlanAlias;
void linearSessionExecutionPlanInputShapeSignature;
void linearSessionExecutionPlanOutputShapeSignature;
void linearSessionExecutionPlanRejectionCode;
void linearSessionAcceptsStepParams;
void linearSessionRequiredCanExecuteStepParams;
void linearSessionRequiredCanExecuteStepParamsAlias;
void linearSessionAcceptsAllocationFreeStepParams;
void linearSessionRequiredAllocationFreeStepParams;
void linearSessionRequiredRuntimeOutputAllocationFreeStepParams;
void linearSessionAcceptsNoReadbackStepParams;
void linearSessionRequiredNoReadbackStepParams;
void linearSessionRequiredReadbackFreeStepParams;
void linearSessionAcceptsHotStepParams;
void linearSessionRequiredHotStepParams;
void linearSessionMatchesStepParamsSignature;
void linearSessionMatchesStepParamsCompatibility;
void linearSessionRejectedRawStepParams;
void linearSessionRejectsRawStepParams;
void linearModuleParameterNames;
void linearModuleParameterNamesWithPrefix;
void linearModuleParameterInfos;
void linearModuleParameterInfoByName;
void linearModuleParameterInfoByIndex;
void linearParameterRequiresGrad;
void linearParameterRequiresGradAlias;
void linearParameterInfoRequiresGrad;
void linearParameterInfoRequiresGradAlias;
void linearModuleChildren;
void linearModuleModules;
void linearModuleNamedChildren;
void linearModuleNamedChildrenAlias;
void linearModuleNamedModules;
void linearModuleNamedModulesAlias;
void linearNamespaceChildren;
void linearNamespaceModules;
void linearNamespaceNamedChildren;
void linearNamespaceNamedChildrenAlias;
void linearNamespaceNamedModules;
void linearNamespaceNamedModulesAlias;
void linearModuleRequiresGrad;
void linearModuleTrain;
void linearModuleEval;
void linearModuleCpu;
void linearModuleToCpu;
void linearModuleFloat;
void linearModuleFloat32;
void linearNamespaceCpu;
void linearNamespaceToCpu;
void linearNamespaceFloat;
void linearNamespaceFloat32;
void zeroGradOptions;
void nnRequiresGradTarget;
void nnStateSnapshot;
void nnLoadedStateTarget;
void linearEvidence;
void linearEvidencePredicate;
void linearEvidenceRequired;
void linearEvidenceAsserted;
void linearEvidenceSignature;
void linearEvidenceSignatureMatch;
void linearEvidenceCompileNamespacePredicate;
void linearEvidenceCompileNamespaceRequired;
void linearEvidenceCompileNamespaceSignatureMatch;
void linearSignatures;
void linearIr;
void linearIrSignature;
void linearKernelPlan;
void linearKernelPlanSignature;
void linearKernelPlanAccepted;
void linearKernelPlanRequired;
void linearKernelPlanAsserted;
void linearKernelPlanSignatureMatch;
void linearKernelPlanMemoryLayoutSignature;
void linearKernelPlanBufferLayoutSignature;
void linearKernelPlanParameterLayoutSignature;
void linearShapeConstraints;
void linearParameterLayout;
void linearParameterLayoutSignature;
void linearCompatibility;
void linearProgramModuleCompatibilityKind;
void linearCompatibilityDiagnostic;
void linearCompatibilityShapeDiagnosticProgramShape;
void linearAccepted;
void linearBindingPlan;
void linearRequiredBindingPlan;
void linearRequiredBindingPlanAlias;
void linearAcceptedBindingPlan;
void linearRequiredEvidenceBindingPlan;
void linearAssertedEvidenceBindingPlan;
void linearAssertedEvidenceBindingPlanAlias;
void linearMatchedEvidenceBindingPlan;
void linearInspectionAcceptedBindingPlan;
void linearInspectionRequiredEvidenceBindingPlan;
void linearBindingPlanKind;
void linearBindingPlanSignature;
void linearBindingPlanIsModuleBindings;
void linearBindingPlanSupport;
void linearBindingPlanParameterNames;
void linearBindingPlanParameterInfos;
void rawLinearBindingPlan;
void rawLinearBindingPlanSignature;
void rawLinearProgramBindingPlan;
void rawLinearProgramBindingPlanSignature;
void rawLinearBindingPlanSupport;
void literalShape;
void literalCpuTensor;
void literalFloatTensor;
void literalFloat32Tensor;
void literalToTensor;
void literalRootToTensor;
void literalRootCpuTensor;
void literalRootFloatTensor;
void literalRootFloat32Tensor;
void literalTypeAsTensor;
void literalTypeAsSnakeTensor;
void literalRootTypeAsTensor;
void literalRootTypeAsSnakeTensor;
void literalStaticTypeAsTensor;
void literalStaticTypeAsSnakeTensor;
void literalNewZerosTensor;
void literalNewOnesTensor;
void literalNewFullTensor;
void literalReshapeAsTensor;
void literalViewAsTensor;
void literalExpandAsTensor;
void literalRepeatTensor;
void literalRepeatVarargsTensor;
void literalTileTensor;
void literalRootRepeatTensor;
void literalRootRepeatVarargsTensor;
void literalStaticTileTensor;
void literalRootTileTensor;
void literalCloneTensor;
void literalFillInPlaceTensor;
void literalZeroInPlaceTensor;
void literalOnesInPlaceTensor;
void literalCopyInPlaceTensor;
void literalDetachTensor;
void reshapedShape;
void inferredReshaped;
void inferredViewed;
void scalarShape;
void scalarValue;
void scalarNumberValue;
void literalRootArangeTensor;
void literalStaticArangeRangeTensor;
void scalarValueOf;
void scalarPrimitive;
void zerosShape;
void gradEnabledAlias;
void previousGradEnabledAlias;
void noGradAliasValue;
void inferenceModeAliasValue;
void enableGradAliasValue;
void rootGradModeNamespace;
void rootGradModeEnabled;
void rootGradModeNoGradValue;
void hostNativePlacementKind;
void programNativePlacementBufferKind;
void programNativePlacementEvidenceSignature;
void programNativePlacementCompiler;
void programNativePlacementInspectionSignature;
void programNativePlacementInspectionSource;
void fromNativeShape;
void copyFromNative;
void copyFromNativeSnake;
void explicitOutputShape;
void typedLinearSessionOutputTensor;
void typedLinearClone;
void typedLinearCloneState;
void typedLinearLoadedClone;
void typedLinearCloneSession;
void typedLinearSessionExecuteTensor;
void typedLinearSessionReadOutputTensor;
void numericOutputShape;
void carrierOutputShape;
void moduleBoundShape;
void readOutputShape;
void carrierReadOutputShape;
void nestedTensorShape;
void nestedTensorRows;
void nestedTensorList;
void nestedTensorNumpy;
void nestedTensorJson;
void nestedTensorFromJson;
void llamaProgramInputLen;
void llamaProgramOutputLen;
void llamaProgramInputByteLength;
void llamaProgramOutputByteLength;
void llamaProgramWeightsLen;
void llamaProgramWeightsByteLength;
void llamaProgramBiasLen;
void llamaProgramBiasByteLength;
void llamaProgramParameterLen;
void llamaProgramParameterByteLength;
void llamaProgramInspectionKind;
void llamaProgramInspectionSignature;
void llamaProgramRequirementsKind;
void llamaProgramRequirementsSignature;
void llamaProgramBufferSizing;
void llamaProgramBufferSizingKind;
void llamaProgramBufferSizingSignature;
void llamaProgramBufferSizingModelKind;
void llamaProgramMatchesBufferSizingSignature;
void llamaProgramCapabilitiesKind;
void llamaProgramCapabilitySignature;
void llamaProgramMatchesCapabilitySignature;
void llamaProgramModelCompatibility;
void llamaProgramModelCompatibilityKind;
void llamaProgramModelCompatibilitySignature;
void llamaKvCacheRequirementsKind;
void llamaKvCacheRequirementsSignature;
void llamaProgramBufferLayoutKind;
void llamaProgramBufferLayoutSignature;
void llamaKvCacheLayoutKind;
void llamaKvCacheLayoutSignature;
void llamaProgramRuntimeProfileAlias;
void llamaProgramRuntimeProfileKind;
void llamaProgramRuntimeProfileSignature;
void llamaProgramMatchesRuntimeProfileSignature;
void llamaProgramMatchesRuntimeProfileSignatureAlias;
void llamaProgramRuntimeProfileExpectationFromProgram;
void llamaProgramRuntimeProfileExpectationAliasFromProgram;
void llamaProgramRuntimeProfileNoFallbackFromProgram;
void llamaProgramRuntimeProfileNoFallbackAliasFromProgram;
void llamaProgramRuntimeProfileNoSyncFromProgram;
void llamaProgramRuntimeProfileNoSyncAliasFromProgram;
void llamaProgramRuntimeProfileNoInvalidPatchesFromProgram;
void llamaProgramRuntimeProfileNoInvalidPatchesAliasFromProgram;
void llamaProgramRuntimeProfileRequiredFromProgram;
void llamaProgramRuntimeProfileRequiredAliasFromProgram;
void llamaProgramRuntimeProfileNoSyncRequiredFromProgram;
void llamaProgramRuntimeProfileNoSyncRequiredAliasFromProgram;
void llamaProgramRuntimePatchValidFromProgram;
void llamaProgramRuntimePatchValidAliasFromProgram;
void llamaProgramRuntimeProfileHotFromProgram;
void llamaProgramRuntimeProfileHotAliasFromProgram;
void llamaSessionInspectionKind;
void llamaSessionInspectionSignature;
void llamaSessionInputLen;
void llamaSessionOutputLen;
void llamaSessionInputByteLength;
void llamaSessionOutputByteLength;
void llamaSessionWeightsLen;
void llamaSessionWeightsByteLength;
void llamaSessionBiasLen;
void llamaSessionBiasByteLength;
void llamaSessionParameterLen;
void llamaSessionParameterByteLength;
void llamaSessionBufferSizing;
void llamaSessionBufferSizingKind;
void llamaSessionBufferSizingSignature;
void llamaSessionBufferSizingModelKind;
void llamaSessionMatchesBufferSizingSignature;
void llamaSessionBufferLayoutKind;
void llamaSessionBufferLayoutSignature;
void llamaSessionKvCacheLayoutKind;
void llamaSessionKvCacheLayoutSignature;
void llamaSessionRuntimeProfileAlias;
void llamaSessionRuntimeProfileKind;
void llamaSessionRuntimeProfileSignature;
void llamaSessionMatchesRuntimeProfileSignature;
void llamaSessionMatchesRuntimeProfileSignatureAlias;
void llamaSessionRuntimeProfileExpectationFromSession;
void llamaSessionRuntimeProfileExpectationAliasFromSession;
void llamaSessionRuntimeProfileNoFallbackFromSession;
void llamaSessionRuntimeProfileNoFallbackAliasFromSession;
void llamaSessionRuntimeProfileNoSyncFromSession;
void llamaSessionRuntimeProfileNoSyncAliasFromSession;
void llamaSessionRuntimeProfileNoInvalidPatchesFromSession;
void llamaSessionRuntimeProfileNoInvalidPatchesAliasFromSession;
void llamaSessionRuntimeProfileRequiredFromSession;
void llamaSessionRuntimeProfileRequiredAliasFromSession;
void llamaSessionRuntimeProfileNoSyncRequiredFromSession;
void llamaSessionRuntimeProfileNoSyncRequiredAliasFromSession;
void llamaSessionRuntimePatchValidFromSession;
void llamaSessionRuntimePatchValidAliasFromSession;
void llamaSessionRuntimeProfileHotFromSession;
void llamaSessionRuntimeProfileHotAliasFromSession;
void llamaSessionCallProfileAlias;
void llamaSessionCallProfileKind;
void llamaSessionCallProfileSignature;
void llamaSessionMatchesCallProfileSignature;
void llamaSessionMatchesCallProfileSignatureAlias;
void llamaSessionStepContractOutputShape;
void llamaSessionStepContractSignature;
void llamaSessionStepContractOutputByteLength;
void llamaSessionStepContractOutputSlotName;
void llamaSessionStepContractOutputSlotRole;
void llamaSessionStepContractRemainingContext;
void llamaSessionStepContractDefaultOutput;
void llamaSessionStepContractDefaultReadbackRequired;
void llamaSessionStepContractDefaultAllocationFree;
void llamaSessionStepContractNoOutputEffect;
void llamaSessionMatchesStepContractSignature;
void llamaSessionStepParamsPreflight;
void llamaSessionStepParamsCompatibility;
void llamaSessionStepParamsKind;
void llamaSessionStepParamsStatus;
void llamaSessionStepParamsContractKind;
void llamaSessionStepParamsContractPosition;
void llamaSessionStepParamsSignature;
void llamaSessionStepParamsRejectionCode;
void llamaSessionStepParamsStateEffect;
void llamaSessionStepParamsAllocationFree;
void llamaSessionStepParamsInputSource;
void llamaSessionStepParamsInputOwnership;
void llamaSessionStepParamsReadsInput;
void llamaSessionStepParamsOutputTarget;
void llamaSessionStepParamsOutputEffect;
void llamaSessionStepParamsOutputOwnership;
void llamaSessionStepParamsOutputReturnOwnership;
void llamaSessionStepParamsWritesOutput;
void llamaSessionStepParamsReadbackRequired;
void llamaSessionStepParamsHotPath;
void llamaSessionStepParamsHotPathStatus;
void llamaSessionStepParamsInputElementType;
void llamaSessionStepParamsOutputElementType;
void llamaSessionStepParamsInputElementLength;
void llamaSessionStepParamsOutputElementLength;
void llamaSessionStepParamsInputByteLength;
void llamaSessionStepParamsOutputByteLength;
void llamaSessionStepParamsDiagnosticRemainingContext;
void llamaSessionExecutionPlanInputOwnership;
void llamaSessionExecutionPlanOutputOwnership;
void llamaSessionExecutionPlanOutputReturnOwnership;
void llamaSessionExecutionPlanInputElementType;
void llamaSessionExecutionPlanOutputElementType;
void llamaSessionExecutionPlanInputElementLength;
void llamaSessionExecutionPlanOutputElementLength;
void llamaSessionExecutionPlanRejectionCode;
void llamaProgramRequiredExecutionPlan;
void llamaProgramRequiredExecutionPlanAlias;
void llamaSessionRequiredExecutionPlan;
void llamaSessionRequiredExecutionPlanAlias;
void llamaSessionAcceptsStepParams;
void llamaSessionRequiredCanExecuteStepParams;
void llamaSessionRequiredCanExecuteStepParamsAlias;
void llamaSessionAcceptsAllocationFreeStepParams;
void llamaSessionRequiredAllocationFreeStepParams;
void llamaSessionRequiredRuntimeOutputAllocationFreeStepParams;
void llamaSessionAcceptsNoReadbackStepParams;
void llamaSessionRequiredNoReadbackStepParams;
void llamaSessionRequiredReadbackFreeStepParams;
void llamaSessionAcceptsHotStepParams;
void llamaSessionRequiredHotStepParams;
void llamaSessionMatchesStepParamsSignature;
void llamaSessionMatchesStepParamsCompatibility;
void llamaSessionRejectedMissingStepParams;
void llamaSessionRejectedEmptyStepParams;
void llamaSessionRejectsEmptyStepParams;
void llamaCarrierReadOutputShape;
void inspectedTrainingStepSignature;
void inlineTrainingStepEvidenceSignature;
void inlineTrainingStepIsEvidence;
void inlineTrainingStepRequired;
void inlineTrainingStepAsserted;
void inlineTrainingStepSignatureMatch;
void inlineTrainingStepSubpathIsEvidence;
void inlineTrainingStepSubpathRequired;
void fitDataEvidenceRequired;
void fitDataEvidenceAsserted;
void fitDataEvidenceSignatureMatch;
void fitDataEvidenceSubpathRequired;
void fitEvidenceSignature;
void fitFinalLoss;
void fitPlainBatchCount;
void evaluateEvidence;
void evaluateEvidenceAlias;
void evaluateMeanLoss;
void evaluateMeanLossAlias;
void evaluateFinalLoss;
void evaluateFinalLossAlias;
void evaluateEvidenceRequired;
void evaluateEvidenceAsserted;
void evaluateEvidenceSignatureMatch;
void evaluateEvidenceSubpathRequired;
void manualSeedValue;
void manualSeedAliasValue;
void initialSeedValue;
void initialSeedAliasValue;
void tensorInitialSeedValue;
void tensorInitialSeedAliasValue;
void seededRngValue;
void seededRandTensor;
void seededRandnTensor;
void reshapeModule;
void reshapeModuleShape;
void reshapeModuleForward;
void typedReshapeSupport;
void typedReshapeCompilePlan;
void inferredReshapeModule;
void inferredReshapeModuleShape;
void inferredReshapeModuleForward;
void inferredReshapeModuleOutputShape;
void inferredReshapeSupport;
void inferredReshapeCompilePlan;
void customModuleConfig;
void customModule;
void customModuleForward;
void customModuleCall;
void customModuleDunderCall;
void customModuleNamespaceCall;
void customModuleParameters;
void customModuleChildren;
void customModuleState;
void customModuleOptimizer;
void customParameterOptions;
void customWeightParameter;
void customBiasParameter;
void initWeightParameter;
void initTensorTarget;
void initConstantParameter;
void initZerosParameter;
void initOnesTensor;
void initUniformParameter;
void initNormalParameter;
void initXavierParameter;
void initXavierCamelParameter;
void initXavierNormalParameter;
void initXavierNormalCamelParameter;
void initKaimingParameter;
void initKaimingCamelParameter;
void initKaimingNormalParameter;
void initKaimingNormalCamelParameter;
void einsumRootTensor;
void einsumStaticTensor;
void einsumTorchTensor;
void einsumEllipsisTensor;
void einsumImplicitEllipsisTensor;
void typedEinsumRootTensor;
void typedEinsumStaticTraceTensor;
void typedEinsumEllipsisTensor;
void typedEinsumImplicitEllipsisTensor;
void customParameterModule;
void customParameterState;
void customParameterOptimizer;
void customAutogradWeight;
void customAutogradBias;
void customAutogradModule;
void customAutogradOutput;
void customAutogradOptimizer;
void customAutogradState;
void customAutogradCloneWeight;
void customAutogradCloneBias;
void customAutogradClone;
void customAutogradValidatedClone;
void customAutogradLoadedClone;
void viewModuleForward;
void inferredViewModuleForward;
void broadcastModuleForward;
void expandModuleForward;
void permuteModule;
void permuteTrace;
void permuteTraceOutputShape;
void linearTrace;
void linearModuleCompilerSignatures;
void linearModuleIr;
void linearModuleKernelPlan;
void linearModuleBufferLayout;
void linearModuleMemoryLayout;
void linearModuleInputShape;
void linearModuleOutputShape;
void linearTypedCompileOptions;
void linearBatchedTypedCompileOptions;
void nnNamespaceTypedOutputShape;
void nnNamespaceTypedOutputShapeAlias;
void linearModuleShapeConstraints;
void linearModuleParameterLayout;
void linearModuleRequiredCompilePlan;
void linearModuleRequiredCompilePlanAlias;
void linearModuleAssertedCompilePlan;
void linearModuleAssertedCompilePlanAlias;
void linearModulePreflight;
void compileNamespacePreflight;
void nativeEagerLinearInto;
void nativeEagerLinearIntoAlias;
void nativeEagerRoutingPolicy;
void nativeEagerRoutingPolicyAlias;
void nativeEagerRoutingCore;
void nativeEagerRoutingMatmul;
void nativeEagerMatmulOptions;
void nativeEagerMatmulInto;
void nativeEagerMatmulIntoAlias;
void nativeEagerElementwiseInto;
void nativeEagerElementwiseIntoAlias;
void nativeEagerReduceInto;
void nativeEagerReduceIntoAlias;
void nativeEagerReduceDimOptions;
void nativeEagerReduceDimInto;
void nativeEagerReduceDimIntoAlias;
void nativeEagerArgReduceDimOptions;
void nativeEagerArgReduceDimInto;
void nativeEagerArgReduceDimIntoAlias;
void nativeEagerCumsumOptions;
void nativeEagerCumsumInto;
void nativeEagerCumsumIntoAlias;
void nativeEagerConv2dOptions;
void nativeEagerConv2dInto;
void nativeEagerConv2dIntoAlias;
void nativeEagerPool2dOptions;
void nativeEagerPool2dInto;
void nativeEagerPool2dIntoAlias;
void nativeEagerSoftmaxInto;
void nativeEagerLogSoftmaxIntoAlias;
void rootNativeCoreEvidence;
void rootNativeCoreAliasEvidence;
void namespaceNativeCoreEvidence;
void namespaceNativeCoreAliasEvidence;
void simpleNativeCoreEvidence;
void nativeCoreProgramSessionDomain;
void nativeCoreMatmulOp;
void nativeCoreUserApiBoundary;
void nativeCoreTensorRuntimeBoundary;
void nativeCoreFfiBoundary;
void nativeCoreModuleCompilerBoundary;
void nativeCoreCompileEvidenceBoundary;
void nativeCoreHotPathBoundary;
void nativeCoreTrainingBoundary;
void sequentialPreflight;
void sequentialMethodPreflight;
void trainingDropoutPreflight;
void trainingDropoutMethodPreflight;
void tensorManualSeedValue;
void tensorManualSeedAliasValue;
void tensorInitialSeedValue;
