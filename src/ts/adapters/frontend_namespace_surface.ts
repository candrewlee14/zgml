import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";
import { tsProductManifestPolicy } from "../internal/product_manifest.js";
import {
  assert_program_compile_evidence,
  assertProgramCompileEvidence,
  completeProgramCompileEvidence,
  compilerEvidenceSignature,
  compilerSignatureEvidence,
  compilerSignaturesFromCompilerEvidence,
  isProgramCompileEvidence,
  matches_program_compile_evidence_signature,
  matchesProgramCompileEvidenceSignature,
  programCompileEvidenceSignature,
  programCompilerSignaturesFromCompileEvidence,
  requireProgramCompileEvidence,
} from "../runtime/compiler_signatures.js";
import {
  compiledSequentialModuleSpecFromTrace,
} from "../runtime/trace_compiler.js";
import {
  compileSupportRejectionReason,
} from "../runtime/compile_support.js";
import {
  moduleBindingPlan as moduleBindingPlanForBindings,
} from "../runtime/module_bindings.js";
import {
  programModuleBindingMetadata,
} from "../runtime/program_module_binding.js";

type LossTrainHelpersOptions = Parameters<SharedFrontendRuntime["createLossTrainHelpers"]>[0];
type OptimizerClassesOptions = Parameters<SharedFrontendRuntime["createOptimizerClasses"]>[0];
type NnNamespaceOptions = Parameters<SharedFrontendRuntime["createNnNamespace"]>[0];
type OptimNamespaceOptions = Parameters<SharedFrontendRuntime["createOptimNamespace"]>[0];
type CheckpointHelpersOptions = Parameters<SharedFrontendRuntime["createCheckpointHelpers"]>[0];
type CheckpointModuleState = ReturnType<CheckpointHelpersOptions["moduleStateDict"]>;
type ZgmlCheckpointIoOptions = Readonly<{
  readTextFile?: (path: string) => string;
  writeTextFile?: (path: string, text: string) => void;
}>;
type AdapterZgmlNamespaceOptions = Readonly<{
  Tensor: LossTrainHelpersOptions["Tensor"];
  tensor: (...args: any[]) => any;
  parameter: (...args: any[]) => any;
  param: (...args: any[]) => any;
  factories: Readonly<Record<string, unknown>>;
  shape: Readonly<{
    hasShape: unknown;
    requireShape: unknown;
  }>;
  grad: Readonly<Record<string, unknown>>;
  gradMode: unknown;
  nn: unknown;
  F: unknown;
  compile: unknown;
  lazy: unknown;
  optim: unknown;
  data: Readonly<Record<string, unknown>>;
  loss: unknown;
  train: unknown;
  checkpoint: CheckpointNamespaceLike;
  checkpointIo: Readonly<{
    save: unknown;
    load: unknown;
  }>;
  Program: unknown;
  Session: unknown;
  NativeBuffer: unknown;
  nativeEager?: unknown;
  nativeCore?: unknown;
  native_core?: unknown;
}>;
type AdapterDataNamespaceOptions<TTensor> = Readonly<{
  tensor: (data: unknown, shape?: unknown, options?: unknown) => TTensor;
  stack: (tensors: readonly TTensor[], dim?: number) => TTensor;
}>;
type UnknownRecord = Record<string, unknown>;
type FunctionalTensor = Readonly<Record<string, unknown>> & {
  shape: readonly number[];
  relu(): unknown;
  gelu(): unknown;
  silu(): unknown;
  sigmoid(): unknown;
  tanh(): unknown;
  softmax(dim?: number): unknown;
  logSoftmax(dim?: number): unknown;
  reshape(shape: readonly number[]): FunctionalTensor;
  flatten(startDim?: number, endDim?: number): FunctionalTensor;
  transpose(dim0?: number, dim1?: number): FunctionalTensor;
  matmul(other: unknown, otherShape?: readonly number[]): FunctionalTensor;
  add(other: unknown): FunctionalTensor;
  abs(): FunctionalTensor;
  pow(exponent: number): FunctionalTensor;
  sum(dim?: number): FunctionalTensor;
  clamp(min?: number | null, max?: number | null): FunctionalTensor;
  div(other: unknown): unknown;
};
type NativeEagerLinearInto = (
  output: Float32Array,
  input: unknown,
  weights: unknown,
  options?: Readonly<{
    bias?: unknown;
    batch?: number;
    inFeatures?: number;
    outFeatures?: number;
    weightLayout?: "in-out" | "out-in" | string;
    weight_layout?: "in-out" | "out-in" | string;
  }>,
) => Float32Array;

type NamespaceModuleConstructors = Pick<
  NnNamespaceOptions,
  | "LinearModule"
  | "EmbeddingModule"
  | "Conv2dModule"
  | "AvgPool2dModule"
  | "MaxPool2dModule"
  | "SequentialModule"
  | "ActivationModule"
  | "SoftmaxModule"
  | "LogSoftmaxModule"
  | "ReductionModule"
  | "DropoutModule"
  | "ShapeModule"
  | "FeatureNormModule"
>;

type NamespaceModuleStateHooks = Pick<
  NnNamespaceOptions,
  | "parameterNames"
  | "parameterInfos"
  | "parameterInfo"
  | "setRequiresGrad"
  | "stateDict"
  | "loadStateDict"
>;

type NamespaceModuleFacadeHooks = Pick<
  NnNamespaceOptions,
  | "traceSequentialProgram"
  | "traceModule"
  | "compileSupportForModule"
  | "requireCompileSupportForModule"
  | "compilerSignaturesForModule"
  | "tensorProgramIrForModule"
  | "kernelPlanForModule"
  | "bufferLayoutForModule"
  | "memoryLayoutForModule"
  | "inputShapeForModule"
  | "outputShapeForModule"
  | "shapeConstraintsForModule"
  | "parameterLayoutForModule"
  | "explainModule"
  | "requireCompilePlanForModule"
  | "canCompileModule"
  | "compileModule"
  | "bindModuleParameters"
  | "placeModuleParameters"
  | "bindingPlanForModuleBindings"
  | "requireBindingPlanForModuleBindings"
>;
type CompileNamespaceTarget = Readonly<{
  trace?: unknown;
  compileSupport?: unknown;
  compile_support?: unknown;
  canCompile?: unknown;
  can_compile?: unknown;
  explain?: unknown;
  compileExplanation?: unknown;
  compile_explanation?: unknown;
  compilePlan?: unknown;
  compile_plan?: unknown;
  compilerSignatures?: unknown;
  compiler_signatures?: unknown;
  tensorProgramIr?: unknown;
  tensor_program_ir?: unknown;
  kernelPlan?: unknown;
  kernel_plan?: unknown;
  bufferLayout?: unknown;
  buffer_layout?: unknown;
  memoryLayout?: unknown;
  memory_layout?: unknown;
  inputShape?: unknown;
  input_shape?: unknown;
  outputShape?: unknown;
  output_shape?: unknown;
  shapeConstraints?: unknown;
  shape_constraints?: unknown;
  parameterLayout?: unknown;
  parameter_layout?: unknown;
  compile?: unknown;
}>;
type CompileNamespaceOptions = Readonly<Record<string, unknown>>;
type CompileNamespaceMethodName = keyof CompileNamespaceTarget;
type CompileNamespaceMethod = (options?: CompileNamespaceOptions) => unknown;
type CompileEvidenceRecord = Readonly<Record<string, any>>;
type CheckpointNamespaceLike = Readonly<{
  stringify(snapshot: unknown, space?: string | number): string;
  parse(text: unknown): unknown;
  restore(snapshot: unknown, targets?: unknown): unknown;
}>;
type TensorMethodSource = Readonly<Record<string, unknown>>;
type AdapterCompileNamespaceOptions = Readonly<{
  traceSequentialProgram: (...args: any[]) => unknown;
  analyzeSequentialProgram: (...args: any[]) => unknown;
  SequentialModule?: new (first?: readonly unknown[] | unknown, ...rest: readonly unknown[]) => unknown;
  compileModuleProgram?: (spec: unknown, compileOptions?: unknown) => unknown;
  trainingStep?: (model: unknown, optimizer: unknown, options?: UnknownRecord) => unknown;
}>;

const compileManifest = Object.freeze({
  kind: "zgml-compile",
  ...tsProductManifestPolicy("src/ts/compile.ts"),
  runtimePath: "Trace -> TensorProgramIr -> KernelPlan -> Program",
});

export type AdapterFrontendNamespaceOptions<TTensor = unknown> =
  NamespaceModuleConstructors &
  NamespaceModuleStateHooks &
  NamespaceModuleFacadeHooks &
  AdapterDataNamespaceOptions<TTensor> &
  Readonly<{
    sharedFrontend: SharedFrontendRuntime;
    Tensor: LossTrainHelpersOptions["Tensor"];
    f32: LossTrainHelpersOptions["f32"];
    f32WithLength: LossTrainHelpersOptions["f32WithLength"];
    indexValues: LossTrainHelpersOptions["indexValues"];
    addTensorGrad: LossTrainHelpersOptions["addTensorGrad"];
    isGradEnabled: NonNullable<LossTrainHelpersOptions["isGradEnabled"]>;
    makeParameter: NnNamespaceOptions["makeParameter"];
    zeroGrad: LossTrainHelpersOptions["zeroGrad"] & OptimizerClassesOptions["zeroGrad"] & OptimNamespaceOptions["zeroGrad"] & NnNamespaceOptions["zeroGrad"];
    resolveParameters: LossTrainHelpersOptions["resolveParameters"] & OptimizerClassesOptions["resolveParameters"] & NnNamespaceOptions["resolveParameters"];
    finiteConfigNumber: OptimizerClassesOptions["finiteConfigNumber"];
    optimizerStateEntry: OptimizerClassesOptions["optimizerStateEntry"];
    optimizerStateDict: OptimizerClassesOptions["optimizerStateDict"];
    optimizerStateEntries: OptimizerClassesOptions["optimizerStateEntries"];
    optimizerStepFromState: OptimizerClassesOptions["optimizerStepFromState"];
    loadOptimizerTensorState: OptimizerClassesOptions["loadOptimizerTensorState"];
    rejectUnexpectedOptimizerState: OptimizerClassesOptions["rejectUnexpectedOptimizerState"];
    compileTrainingStep?: LossTrainHelpersOptions["compileTrainingStep"];
    nativeEagerLinearInto?: NativeEagerLinearInto;
    geluScalar: NnNamespaceOptions["geluScalar"];
    siluScalar: NnNamespaceOptions["siluScalar"];
    stateDict: NnNamespaceOptions["stateDict"] & CheckpointHelpersOptions["moduleStateDict"];
    loadStateDict: NnNamespaceOptions["loadStateDict"] & CheckpointHelpersOptions["loadModuleStateDict"];
  }>;

export function createAdapterFrontendNamespaces<TTensor = unknown>(options: AdapterFrontendNamespaceOptions<TTensor>) {
  const lossTrainHelpers = options.sharedFrontend.createLossTrainHelpers({
    Tensor: options.Tensor,
    f32: options.f32,
    f32WithLength: options.f32WithLength,
    indexValues: options.indexValues,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    zeroGrad: options.zeroGrad,
    resolveParameters: options.resolveParameters,
    compileTrainingStep: options.compileTrainingStep,
  });
  const fitModuleFromTrain = lossTrainHelpers.train.fit as (target: unknown, batches: unknown, options?: unknown) => unknown;
  const fitNativeModuleFromTrain = lossTrainHelpers.train.fitNative as (target: unknown, batches: unknown, options?: unknown) => unknown;
  const evaluateModuleFromTrain = lossTrainHelpers.train.evaluateModule as (
    target: unknown,
    batches: unknown,
    criterion: unknown,
    options?: unknown,
  ) => unknown;
  const predictModuleFromTrain = lossTrainHelpers.train.predictModule as (
    target: unknown,
    batches: unknown,
    options?: unknown,
  ) => unknown;

  const optimizerClasses = options.sharedFrontend.createOptimizerClasses({
    resolveParameters: options.resolveParameters,
    finiteConfigNumber: options.finiteConfigNumber,
    zeroGrad: options.zeroGrad,
    optimizerStateEntry: options.optimizerStateEntry,
    optimizerStateDict: options.optimizerStateDict,
    optimizerStateEntries: options.optimizerStateEntries,
    optimizerStepFromState: options.optimizerStepFromState,
    loadOptimizerTensorState: options.loadOptimizerTensorState,
    rejectUnexpectedOptimizerState: options.rejectUnexpectedOptimizerState,
  });

  const nnBase = options.sharedFrontend.createNnNamespace({
    LinearModule: options.LinearModule,
    EmbeddingModule: options.EmbeddingModule,
    Conv2dModule: options.Conv2dModule,
    AvgPool2dModule: options.AvgPool2dModule,
    MaxPool2dModule: options.MaxPool2dModule,
    SequentialModule: options.SequentialModule,
    ActivationModule: options.ActivationModule,
    SoftmaxModule: options.SoftmaxModule,
    LogSoftmaxModule: options.LogSoftmaxModule,
    ReductionModule: options.ReductionModule,
    DropoutModule: options.DropoutModule,
    ShapeModule: options.ShapeModule,
    FeatureNormModule: options.FeatureNormModule,
    MSELoss: lossTrainHelpers.loss.MSELoss,
    L1Loss: lossTrainHelpers.loss.L1Loss,
    HuberLoss: lossTrainHelpers.loss.HuberLoss,
    SmoothL1Loss: lossTrainHelpers.loss.SmoothL1Loss,
    BCELoss: lossTrainHelpers.loss.BCELoss,
    BCEWithLogitsLoss: lossTrainHelpers.loss.BCEWithLogitsLoss,
    CrossEntropyLoss: lossTrainHelpers.loss.CrossEntropyLoss,
    NLLLoss: lossTrainHelpers.loss.NLLLoss,
    geluScalar: options.geluScalar,
    siluScalar: options.siluScalar,
    f32WithLength: options.f32WithLength,
    makeParameter: options.makeParameter,
    resolveParameters: options.resolveParameters,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    zeroGrad: options.zeroGrad,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    traceSequentialProgram: options.traceSequentialProgram,
    traceModule: options.traceModule,
    compileSupportForModule: options.compileSupportForModule,
    requireCompileSupportForModule: options.requireCompileSupportForModule,
    compilerSignaturesForModule: options.compilerSignaturesForModule,
    tensorProgramIrForModule: options.tensorProgramIrForModule,
    kernelPlanForModule: options.kernelPlanForModule,
    bufferLayoutForModule: options.bufferLayoutForModule,
    memoryLayoutForModule: options.memoryLayoutForModule,
    inputShapeForModule: options.inputShapeForModule,
    outputShapeForModule: options.outputShapeForModule,
    shapeConstraintsForModule: options.shapeConstraintsForModule,
    parameterLayoutForModule: options.parameterLayoutForModule,
    explainModule: options.explainModule,
    requireCompilePlanForModule: options.requireCompilePlanForModule,
    canCompileModule: options.canCompileModule,
    compileModule: options.compileModule,
    bindModuleParameters: options.bindModuleParameters,
    placeModuleParameters: options.placeModuleParameters,
    bindingPlanForModuleBindings: options.bindingPlanForModuleBindings,
    requireBindingPlanForModuleBindings: options.requireBindingPlanForModuleBindings,
    fitModule: (target: unknown, batches: unknown, fitOptions = {}) => fitModuleFromTrain(target, batches, fitOptions),
    fitNativeModule: (target: unknown, batches: unknown, fitOptions = {}) => fitNativeModuleFromTrain(target, batches, fitOptions),
    evaluateModule: (target: unknown, batches: unknown, criterion: unknown, evaluateOptions = {}) => (
      evaluateModuleFromTrain(target, batches, criterion, evaluateOptions)
    ),
    predictModule: (target: unknown, batches: unknown, predictOptions = {}) => (
      predictModuleFromTrain(target, batches, predictOptions)
    ),
  });

  const optim = options.sharedFrontend.createOptimNamespace({
    SgdOptimizer: optimizerClasses.SgdOptimizer,
    AdamOptimizer: optimizerClasses.AdamOptimizer,
    AdamWOptimizer: optimizerClasses.AdamWOptimizer,
    RMSpropOptimizer: optimizerClasses.RMSpropOptimizer,
    AdagradOptimizer: optimizerClasses.AdagradOptimizer,
    zeroGrad: options.zeroGrad,
  });

  const checkpoint = options.sharedFrontend.createCheckpointHelpers({
    moduleStateDict: (target: unknown, prefix = "") => nnBase.stateDict(target, prefix) as CheckpointModuleState,
    loadModuleStateDict: nnBase.loadStateDict,
    optimizerStateDict: (optimizer: unknown) => {
      const target = optimizer as { stateDict?: () => unknown };
      if (typeof target.stateDict !== "function") {
        throw new Error("checkpoint optimizer target requires stateDict()");
      }
      return target.stateDict() as UnknownRecord;
    },
    loadOptimizerStateDict: (
      optimizer: unknown,
      source: unknown,
      checkpointOptions: unknown = {},
    ) => {
      const target = optimizer as { loadStateDict?: (source: unknown, checkpointOptions?: unknown) => unknown };
      if (typeof target.loadStateDict !== "function") {
        throw new Error("checkpoint optimizer target requires loadStateDict()");
      }
      return target.loadStateDict(source, checkpointOptions);
    },
  });

  function functionalTensor(input: unknown) {
    return input instanceof options.Tensor ? input as unknown as FunctionalTensor : options.tensor(input) as unknown as FunctionalTensor;
  }

  function rejectFunctionalInplace(inplace: unknown, label: string) {
    if (inplace === undefined || inplace === null || inplace === false) return;
    throw new Error(`${label} does not support inplace=true; use the returned tensor instead`);
  }

  function normFeatureCount(input: FunctionalTensor, features: unknown, label: string) {
    if (Number.isSafeInteger(features) && (features as number) > 0) return features as number;
    if (Array.isArray(features) && features.length > 0) {
      return features.reduce((total, dim, index) => {
        if (!Number.isSafeInteger(dim) || (dim as number) <= 0) {
          throw new Error(`${label} normalized shape dimension ${index} must be a positive safe integer, got ${dim}`);
        }
        return total * (dim as number);
      }, 1);
    }
    if (features === undefined || features === null) {
      const last = input.shape.at(-1);
      if (Number.isSafeInteger(last) && last! > 0) return last!;
    }
    throw new Error(`${label} features must be a positive safe integer or non-empty normalized shape`);
  }

  function normForward(kind: "layerNorm" | "rmsNorm" | "batchNorm1d", input: unknown, features: unknown, config: UnknownRecord = {}) {
    const tensorInput = functionalTensor(input);
    const featureCount = normFeatureCount(tensorInput, features, `nn.functional.${kind}`);
    const module = kind === "layerNorm"
      ? (nnBase as { layerNorm(features: number, config?: UnknownRecord): { forward(input: unknown): unknown } }).layerNorm(featureCount, config)
      : kind === "rmsNorm"
        ? (nnBase as { rmsNorm(features: number, config?: UnknownRecord): { forward(input: unknown): unknown } }).rmsNorm(featureCount, config)
        : (nnBase as { batchNorm1d(features: number, config?: UnknownRecord): { forward(input: unknown): unknown } }).batchNorm1d(featureCount, config);
    return module.forward(tensorInput);
  }

  function isNormConfig(value: unknown) {
    if (!value || typeof value !== "object" || value instanceof options.Tensor || Array.isArray(value) || ArrayBuffer.isView(value)) return false;
    return [
      "eps",
      "affine",
      "weight",
      "weights",
      "bias",
      "biasValues",
      "momentum",
      "trackRunningStats",
      "track_running_stats",
      "runningMean",
      "running_mean",
      "runningVar",
      "running_var",
      "numBatchesTracked",
      "num_batches_tracked",
    ].some((key) => key in value);
  }

  function layerNormConfig(weightOrConfig: unknown = {}, bias: unknown = null, eps: unknown = undefined) {
    if (isNormConfig(weightOrConfig) && bias === null && eps === undefined) return weightOrConfig as UnknownRecord;
    const config: UnknownRecord = {};
    if (eps !== undefined && eps !== null) config.eps = eps;
    if (weightOrConfig !== undefined && weightOrConfig !== null) config.weight = weightOrConfig;
    if (bias !== undefined && bias !== null) config.bias = bias;
    else if (weightOrConfig !== undefined && weightOrConfig !== null) config.bias = false;
    else config.affine = false;
    return config;
  }

  function rmsNormConfig(weightOrConfig: unknown = {}, eps: unknown = undefined) {
    if (isNormConfig(weightOrConfig) && eps === undefined) return weightOrConfig as UnknownRecord;
    const config: UnknownRecord = {};
    if (eps !== undefined && eps !== null) config.eps = eps;
    if (weightOrConfig !== undefined && weightOrConfig !== null) config.weight = weightOrConfig;
    else config.affine = false;
    return config;
  }

  function functionalEmbedding(input: unknown, weight: unknown) {
    const weightTensor = functionalTensor(weight);
    if (weightTensor.shape.length !== 2) {
      throw new Error(`nn.functional.embedding weight must be rank-2 [num_embeddings, embedding_dim], got [${weightTensor.shape.join(",")}]`);
    }
    const [numEmbeddings, embeddingDim] = weightTensor.shape;
    const module = (nnBase as {
      embedding(numEmbeddings: number, embeddingDim: number, config?: UnknownRecord): { forward(input: unknown): unknown };
    }).embedding(numEmbeddings!, embeddingDim!, { weight: weightTensor });
    return module.forward(input);
  }

  function functionalTensorElementCount(tensor: FunctionalTensor) {
    return tensor.shape.reduce((total, dim) => total * dim, 1);
  }

  function functionalNativeLinear(
    tensorInput: FunctionalTensor,
    weightTensor: FunctionalTensor,
    bias: unknown,
    outFeatures: number,
    inFeatures: number,
  ) {
    if (typeof options.nativeEagerLinearInto !== "function" || options.isGradEnabled()) return null;
    const inputRank = tensorInput.shape.length;
    const leadingShape = inputRank === 1 ? [] : tensorInput.shape.slice(0, -1);
    const rowCount = leadingShape.length === 0
      ? 1
      : leadingShape.reduce((total, dim) => total * dim, 1);
    const nativeInput = inputRank === 1
      ? tensorInput.reshape([1, inFeatures])
      : inputRank === 2
        ? tensorInput
        : tensorInput.reshape([rowCount, inFeatures]);
    let nativeBias: unknown = null;
    if (bias !== undefined && bias !== null) {
      const biasTensor = functionalTensor(bias);
      if (functionalTensorElementCount(biasTensor) !== outFeatures) return null;
      nativeBias = biasTensor;
    }
    const outputData = new Float32Array(rowCount * outFeatures);
    options.nativeEagerLinearInto(outputData, nativeInput, weightTensor, {
      bias: nativeBias,
      batch: rowCount,
      inFeatures,
      outFeatures,
      weightLayout: "out-in",
    });
    const outputShape = inputRank === 1 ? [outFeatures] : [...leadingShape, outFeatures];
    return options.tensor(outputData, outputShape);
  }

  function functionalLinear(input: unknown, weight: unknown, bias: unknown = null) {
    const tensorInput = functionalTensor(input);
    const weightTensor = functionalTensor(weight);
    if (weightTensor.shape.length !== 2) {
      throw new Error(`nn.functional.linear weight must be rank-2 [out_features, in_features], got [${weightTensor.shape.join(",")}]`);
    }
    const [outFeatures, inFeatures] = weightTensor.shape;
    if (!Number.isSafeInteger(outFeatures) || outFeatures! <= 0 || !Number.isSafeInteger(inFeatures) || inFeatures! <= 0) {
      throw new Error(`nn.functional.linear weight shape must use positive safe integer dimensions, got [${weightTensor.shape.join(",")}]`);
    }
    if (tensorInput.shape.length === 1) {
      if (tensorInput.shape[0] !== inFeatures) {
        throw new Error(`nn.functional.linear input last dimension ${tensorInput.shape[0]} must match weight in_features ${inFeatures}`);
      }
      const nativeOutput = functionalNativeLinear(tensorInput, weightTensor, bias, outFeatures!, inFeatures!);
      if (nativeOutput !== null) return nativeOutput;
      const output = tensorInput
        .reshape([1, inFeatures!])
        .matmul(weightTensor.transpose())
        .reshape([outFeatures!]);
      return bias === undefined || bias === null ? output : output.add(bias);
    }
    if (tensorInput.shape.length < 2) {
      throw new Error(`nn.functional.linear input must be rank-1 or higher, got [${tensorInput.shape.join(",")}]`);
    }
    const lastDim = tensorInput.shape.at(-1);
    if (lastDim !== inFeatures) {
      throw new Error(`nn.functional.linear input last dimension ${lastDim} must match weight in_features ${inFeatures}`);
    }
    const nativeOutput = functionalNativeLinear(tensorInput, weightTensor, bias, outFeatures!, inFeatures!);
    if (nativeOutput !== null) return nativeOutput;
    if (tensorInput.shape.length === 2) {
      const output = tensorInput.matmul(weightTensor.transpose());
      return bias === undefined || bias === null ? output : output.add(bias);
    }
    const leadingShape = tensorInput.shape.slice(0, -1);
    const leadingCount = leadingShape.reduce((total, dim) => total * dim, 1);
    const output = tensorInput
      .reshape([leadingCount, inFeatures!])
      .matmul(weightTensor.transpose())
      .reshape([...leadingShape, outFeatures!]);
    return bias === undefined || bias === null ? output : output.add(bias);
  }

  function functionalConv2d(
    input: unknown,
    weight: unknown,
    bias: unknown = null,
    stride: unknown = 1,
    padding: unknown = 0,
    dilation: unknown = 1,
    groups: unknown = 1,
  ) {
    const weightTensor = functionalTensor(weight);
    if (weightTensor.shape.length !== 4) {
      throw new Error(`nn.functional.conv2d weight must be rank-4 [out_channels,in_channels,kernel_h,kernel_w], got [${weightTensor.shape.join(",")}]`);
    }
    const [outChannels, inChannels, kernelH, kernelW] = weightTensor.shape;
    if (
      !Number.isSafeInteger(outChannels) || outChannels! <= 0 ||
      !Number.isSafeInteger(inChannels) || inChannels! <= 0 ||
      !Number.isSafeInteger(kernelH) || kernelH! <= 0 ||
      !Number.isSafeInteger(kernelW) || kernelW! <= 0
    ) {
      throw new Error(`nn.functional.conv2d weight shape must use positive safe integer dimensions, got [${weightTensor.shape.join(",")}]`);
    }
    const module = (nnBase as {
      conv2d(inChannels: number, outChannels: number, kernelSize: readonly [number, number], config?: UnknownRecord): { forward(input: unknown): unknown };
    }).conv2d(inChannels!, outChannels!, [kernelH!, kernelW!], {
      weight: weightTensor,
      bias: bias === undefined || bias === null ? false : bias,
      stride,
      padding,
      dilation,
      groups,
    });
    return module.forward(functionalTensor(input));
  }

  function poolConfig(stride: unknown, padding: unknown, trailing: UnknownRecord = {}) {
    return Object.freeze({
      ...(stride !== undefined && stride !== null ? { stride } : {}),
      ...(padding !== undefined && padding !== null ? { padding } : {}),
      ...trailing,
    });
  }

  function functionalMaxPool2d(
    input: unknown,
    kernelSize: unknown,
    stride: unknown = null,
    padding: unknown = 0,
    dilation: unknown = 1,
    ceilMode: unknown = false,
  ) {
    const config = poolConfig(stride, padding, { dilation, ceilMode });
    const module = (nnBase as {
      max_pool2d(kernelSize: unknown, config?: UnknownRecord): { forward(input: unknown): unknown };
    }).max_pool2d(kernelSize, config);
    return module.forward(functionalTensor(input));
  }

  function functionalAvgPool2d(
    input: unknown,
    kernelSize: unknown,
    stride: unknown = null,
    padding: unknown = 0,
    ceilMode: unknown = false,
    countIncludePad: unknown = true,
  ) {
    const config = poolConfig(stride, padding, { ceilMode, countIncludePad });
    const module = (nnBase as {
      avg_pool2d(kernelSize: unknown, config?: UnknownRecord): { forward(input: unknown): unknown };
    }).avg_pool2d(kernelSize, config);
    return module.forward(functionalTensor(input));
  }

  function functionalNormalize(input: unknown, p = 2, dim = 1, eps = 1e-12) {
    if (!Number.isFinite(p) || p <= 0) throw new Error(`nn.functional.normalize p must be finite and positive, got ${p}`);
    if (!Number.isFinite(eps) || eps <= 0) throw new Error(`nn.functional.normalize eps must be finite and positive, got ${eps}`);
    const tensorInput = functionalTensor(input);
    const norm = tensorInput.abs().pow(p).sum(dim).pow(1 / p).clamp(eps, null);
    return tensorInput.div(norm);
  }

  function functionalOneHot(input: unknown, numClasses: unknown = -1) {
    const values = options.indexValues(input, "nn.functional.one_hot input");
    const inputShape = input instanceof options.Tensor ? (input as unknown as FunctionalTensor).shape : [values.length];
    let classes = Number(numClasses);
    if (numClasses === undefined || numClasses === null || classes === -1) {
      classes = 0;
      for (let i = 0; i < values.length; i += 1) classes = Math.max(classes, Number(values[i]) + 1);
    }
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`nn.functional.one_hot numClasses must be a positive safe integer or -1, got ${numClasses}`);
    }
    const data = new Float32Array(values.length * classes);
    for (let i = 0; i < values.length; i += 1) {
      const target = Number(values[i]);
      if (!Number.isSafeInteger(target) || target < 0 || target >= classes) {
        throw new Error(`nn.functional.one_hot input ${target} at position ${i} must be an integer in [0, ${classes - 1}]`);
      }
      data[i * classes + target] = 1;
    }
    return options.tensor(data, [...inputShape, classes]);
  }

  const functional = Object.freeze({
    ...lossTrainHelpers.loss,
    mse_loss: lossTrainHelpers.meanSquaredError,
    l1_loss: lossTrainHelpers.meanAbsoluteError,
    huber_loss: lossTrainHelpers.huber,
    smooth_l1_loss: lossTrainHelpers.smoothL1,
    relu: (input: unknown, inplace: unknown = false) => {
      rejectFunctionalInplace(inplace, "nn.functional.relu");
      return functionalTensor(input).relu();
    },
    gelu: (input: unknown) => functionalTensor(input).gelu(),
    silu: (input: unknown) => functionalTensor(input).silu(),
    sigmoid: (input: unknown) => functionalTensor(input).sigmoid(),
    tanh: (input: unknown) => functionalTensor(input).tanh(),
    softmax: (input: unknown, dim = -1) => functionalTensor(input).softmax(dim),
    softmaxDim: (input: unknown, dim = -1) => functionalTensor(input).softmax(dim),
    softmax_dim: (input: unknown, dim = -1) => functionalTensor(input).softmax(dim),
    logSoftmax: (input: unknown, dim = -1) => functionalTensor(input).logSoftmax(dim),
    log_softmax: (input: unknown, dim = -1) => functionalTensor(input).logSoftmax(dim),
    logSoftmaxDim: (input: unknown, dim = -1) => functionalTensor(input).logSoftmax(dim),
    log_softmax_dim: (input: unknown, dim = -1) => functionalTensor(input).logSoftmax(dim),
    dropout: (input: unknown, p = 0.5, config: UnknownRecord | boolean = {}, inplace: unknown = false) => {
      rejectFunctionalInplace(inplace, "nn.functional.dropout");
      const dropoutConfig = typeof config === "boolean" ? { training: config } : config;
      const dropoutModule = (nnBase as { dropout(p?: number, config?: UnknownRecord): { forward(input: unknown): unknown } }).dropout(p, dropoutConfig);
      return dropoutModule.forward(functionalTensor(input));
    },
    flatten: (input: unknown, startDim = 0, endDim = -1) => functionalTensor(input).flatten(startDim, endDim),
    linear: functionalLinear,
    conv2d: functionalConv2d,
    maxPool2d: functionalMaxPool2d,
    max_pool2d: functionalMaxPool2d,
    avgPool2d: functionalAvgPool2d,
    avg_pool2d: functionalAvgPool2d,
    normalize: functionalNormalize,
    oneHot: functionalOneHot,
    one_hot: functionalOneHot,
    embedding: functionalEmbedding,
    layerNorm: (input: unknown, features: unknown, weightOrConfig: unknown = {}, bias: unknown = null, eps: unknown = undefined) => {
      return normForward("layerNorm", input, features, layerNormConfig(weightOrConfig, bias, eps));
    },
    layer_norm: (input: unknown, features: unknown, weightOrConfig: unknown = {}, bias: unknown = null, eps: unknown = undefined) => {
      return normForward("layerNorm", input, features, layerNormConfig(weightOrConfig, bias, eps));
    },
    rmsNorm: (input: unknown, features: unknown, weightOrConfig: unknown = {}, eps: unknown = undefined) => {
      return normForward("rmsNorm", input, features, rmsNormConfig(weightOrConfig, eps));
    },
    rms_norm: (input: unknown, features: unknown, weightOrConfig: unknown = {}, eps: unknown = undefined) => {
      return normForward("rmsNorm", input, features, rmsNormConfig(weightOrConfig, eps));
    },
    batchNorm1d: (input: unknown, features: unknown, config: UnknownRecord = {}) => {
      return normForward("batchNorm1d", input, features, config);
    },
    batch_norm1d: (input: unknown, features: unknown, config: UnknownRecord = {}) => {
      return normForward("batchNorm1d", input, features, config);
    },
  });
  const nn = Object.freeze({
    ...nnBase,
    functional,
    F: functional,
  });

  return Object.freeze({
    lossTrainHelpers,
    meanSquaredError: lossTrainHelpers.meanSquaredError,
    meanAbsoluteError: lossTrainHelpers.meanAbsoluteError,
    huber: lossTrainHelpers.huber,
    smoothL1: lossTrainHelpers.smoothL1,
    binaryCrossEntropy: lossTrainHelpers.binaryCrossEntropy,
    binaryCrossEntropyWithLogits: lossTrainHelpers.binaryCrossEntropyWithLogits,
    classTargets: lossTrainHelpers.classTargets,
    crossEntropy: lossTrainHelpers.crossEntropy,
    loss: lossTrainHelpers.loss,
    train: lossTrainHelpers.train,
    data: options.sharedFrontend.createDataNamespace({
      tensor: options.tensor,
      stack: options.stack,
    }),
    optimizerClasses,
    SgdOptimizer: optimizerClasses.SgdOptimizer,
    AdamOptimizer: optimizerClasses.AdamOptimizer,
    AdamWOptimizer: optimizerClasses.AdamWOptimizer,
    RMSpropOptimizer: optimizerClasses.RMSpropOptimizer,
    AdagradOptimizer: optimizerClasses.AdagradOptimizer,
    nn,
    optim,
    checkpoint,
  });
}

function requireCompileTarget(target: unknown, name: string): CompileNamespaceTarget {
  if (!target || typeof target !== "object") {
    throw new Error(`compile.${name} requires an nn module or Sequential layer list`);
  }
  return target as CompileNamespaceTarget;
}

function compileTargetMethod(target: unknown, name: CompileNamespaceMethodName): CompileNamespaceMethod | null {
  const source = requireCompileTarget(target, name);
  const method = source[name];
  return typeof method === "function" ? method.bind(source) as CompileNamespaceMethod : null;
}

function compileObjectEvidence(value: unknown): CompileEvidenceRecord | null {
  return value && typeof value === "object" ? value as CompileEvidenceRecord : null;
}

function nestedCompileEvidence(value: unknown): CompileEvidenceRecord | null {
  return value && typeof value === "object" ? value as CompileEvidenceRecord : null;
}

function lazyModuleSpecFromSupport(support: unknown) {
  const evidence = compileObjectEvidence(support);
  if (
    evidence?.supported !== true ||
    !evidence.trace ||
    !evidence.ir ||
    !evidence.kernelPlan
  ) {
    return null;
  }
  return compiledSequentialModuleSpecFromTrace([], evidence.trace, {
    ir: evidence.ir,
    kernelPlan: evidence.kernelPlan,
    diagnostic: evidence.diagnostic ?? null,
  });
}

function looksLikeLazyTensor(target: unknown) {
  if (!target || typeof target !== "object") return false;
  const record = target as Readonly<Record<string, unknown>>;
  return Array.isArray(record.ops) &&
    typeof record.trace === "function" &&
    typeof record.compileSupport === "function";
}

export function createAdapterCompileNamespace(options: AdapterCompileNamespaceOptions) {
  function moduleTarget(target: unknown, name: string) {
    if (!Array.isArray(target)) return target;
    if (typeof options.SequentialModule !== "function") {
      throw new Error(`compile.${name} requires nn.Sequential for raw layer lists`);
    }
    return new options.SequentialModule(target);
  }

  function trace(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const method = compileTargetMethod(target, "trace");
    if (method) return method(compileOptions);
    if (Array.isArray(target)) return options.traceSequentialProgram(target, compileOptions);
    throw new Error("compile.trace requires a module with trace() or a Sequential layer list");
  }

  function analyze(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    if (Array.isArray(target)) return options.analyzeSequentialProgram(target, compileOptions);
    const direct = compileTargetMethod(target, "explain")
      ?? compileTargetMethod(target, "compileExplanation")
      ?? compileTargetMethod(target, "compile_explanation")
      ?? compileTargetMethod(target, "compilePlan")
      ?? compileTargetMethod(target, "compile_plan")
      ?? compileTargetMethod(target, "compileSupport")
      ?? compileTargetMethod(target, "compile_support");
    if (direct) return direct(compileOptions);
    return trace(target, compileOptions);
  }

  function compileSupport(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "compileSupport");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "compile_support");
    if (snake) return snake(compileOptions);
    const analysis = analyze(target, compileOptions);
    const evidence = compileObjectEvidence(analysis);
    if (typeof evidence?.supported === "boolean") return analysis;
    return evidence?.support ?? evidence?.artifacts?.support ?? analysis;
  }

  function requireCompileSupport(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const support = compileSupport(target, compileOptions);
    if (!support || typeof support !== "object") {
      throw new Error("compile.requireCompileSupport requires structured compileSupport evidence");
    }
    if (compileObjectEvidence(support)?.supported !== true) {
      throw new Error(`compile.requireCompileSupport rejected unsupported target: ${compileSupportRejectionReason(support)}`);
    }
    return support;
  }

  function compileEvidence(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const support = compileSupport(target, compileOptions);
    if (!support || typeof support !== "object") return {};
    const evidence = support as CompileEvidenceRecord;
    return nestedCompileEvidence(evidence.support)
      ?? nestedCompileEvidence(evidence.compiled)
      ?? evidence;
  }

  function canCompile(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "canCompile");
    if (camel) return Boolean(camel(compileOptions));
    const snake = compileTargetMethod(target, "can_compile");
    if (snake) return Boolean(snake(compileOptions));
    return Boolean(compileObjectEvidence(compileSupport(target, compileOptions))?.supported);
  }

  function explain(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const direct = compileTargetMethod(target, "explain")
      ?? compileTargetMethod(target, "compileExplanation")
      ?? compileTargetMethod(target, "compile_explanation")
      ?? compileTargetMethod(target, "compilePlan")
      ?? compileTargetMethod(target, "compile_plan");
    return direct ? direct(compileOptions) : compileSupport(target, compileOptions);
  }

  function requireCompilePlan(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const plan = explain(target, compileOptions);
    if (!plan || typeof plan !== "object") {
      throw new Error("compile.requireCompilePlan requires structured compile explanation evidence");
    }
    if (compileObjectEvidence(plan)?.supported !== true) {
      throw new Error(`compile.requireCompilePlan rejected unsupported target: ${compileSupportRejectionReason(plan)}`);
    }
    return plan;
  }

  function compilerSignatures(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "compilerSignatures");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "compiler_signatures");
    if (snake) return snake(compileOptions);
    const evidence = compileEvidence(target, compileOptions);
    return evidence.compilerSignatures ?? evidence.signatures ?? evidence.artifacts?.compilerSignatures ?? null;
  }

  function tensorProgramIr(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "tensorProgramIr");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "tensor_program_ir");
    if (snake) return snake(compileOptions);
    const evidence = compileEvidence(target, compileOptions);
    return evidence.ir ?? evidence.artifacts?.ir ?? null;
  }

  function kernelPlan(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "kernelPlan");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "kernel_plan");
    if (snake) return snake(compileOptions);
    const evidence = compileEvidence(target, compileOptions);
    return evidence.kernelPlan ?? evidence.artifacts?.kernelPlan ?? null;
  }

  function kernelPlanField(target: unknown, field: string, compileOptions: CompileNamespaceOptions = {}) {
    const plan = compileObjectEvidence(kernelPlan(target, compileOptions));
    return plan ? plan[field] ?? null : null;
  }

  function inputShape(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "inputShape");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "input_shape");
    if (snake) return snake(compileOptions);
    const evidence = compileEvidence(target, compileOptions);
    return evidence.inputShape ?? evidence.trace?.inputShape ?? kernelPlanField(target, "inputShape", compileOptions);
  }

  function outputShape(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    const camel = compileTargetMethod(target, "outputShape");
    if (camel) return camel(compileOptions);
    const snake = compileTargetMethod(target, "output_shape");
    if (snake) return snake(compileOptions);
    const evidence = compileEvidence(target, compileOptions);
    return evidence.outputShape ?? evidence.trace?.outputShape ?? kernelPlanField(target, "outputShape", compileOptions);
  }

  function compileLazyTarget(target: unknown, compileOptions: CompileNamespaceOptions, requireLazyGraph = false) {
    const support = compileSupport(target, compileOptions);
    const lazySpec = lazyModuleSpecFromSupport(support);
    if (lazySpec && options.compileModuleProgram) return options.compileModuleProgram(lazySpec, compileOptions);
    if (lazySpec) throw new Error("compile.compile requires a native module Program compiler for lazy graphs");
    if (requireLazyGraph) {
      throw new Error(`lazy graph cannot compile to native Program: ${compileSupportRejectionReason(support)}`);
    }
    return null;
  }

  function compile(target: unknown, compileOptions: CompileNamespaceOptions = {}) {
    if (Array.isArray(target)) {
      return compile(moduleTarget(target, "compile"), compileOptions);
    }
    if (looksLikeLazyTensor(target)) {
      const compiled = compileLazyTarget(target, compileOptions, true);
      if (compiled) return compiled;
    }
    const method = compileTargetMethod(target, "compile");
    if (method) return method(compileOptions);
    const compiled = compileLazyTarget(target, compileOptions);
    if (compiled) return compiled;
    throw new Error("compile.compile requires a module with compile() or a compile-capable lazy graph");
  }

  function requireNativeProgramMethod(program: Record<string, any>, method: string) {
    const fn = program[method];
    if (typeof fn !== "function") {
      throw new Error(`compile.compileForInference expected native Program.${method} proof method`);
    }
    return fn.bind(program);
  }

  function requireNativeSessionMethod(session: Record<string, any>, method: string) {
    const fn = session[method];
    if (typeof fn !== "function") {
      throw new Error(`compile.compileForInference expected native Session.${method} proof method`);
    }
    return fn.bind(session);
  }

  function compiledInferenceHandle(
    program: Record<string, any>,
    session: Record<string, any>,
    target: unknown,
    compileOptions: CompileNamespaceOptions,
    bindOptions?: unknown,
  ) {
    let disposed = false;
    const compileEvidence = requireNativeProgramMethod(program, "compileEvidence");
    const requirements = requireNativeProgramMethod(program, "requirements");
    const bufferLayout = requireNativeProgramMethod(program, "bufferLayout");
    const bufferSlotNames = requireNativeProgramMethod(program, "bufferSlotNames");
    const bufferSlot = requireNativeProgramMethod(program, "bufferSlot");
    const inputShape = requireNativeProgramMethod(program, "inputShape");
    const outputShape = requireNativeProgramMethod(program, "outputShape");
    const kernelPlan = requireNativeProgramMethod(program, "kernelPlan");
    const compilerSignatures = requireNativeProgramMethod(program, "compilerSignatures");
    const bindingPlan = requireNativeProgramMethod(program, "bindingPlan");
    const sessionBufferLayout = requireNativeSessionMethod(session, "bufferLayout");
    const sessionBufferSlotNames = requireNativeSessionMethod(session, "bufferSlotNames");
    const sessionBufferSlot = requireNativeSessionMethod(session, "bufferSlot");
    const stepContract = requireNativeSessionMethod(session, "stepContract");
    const hotPathPlan = requireNativeSessionMethod(session, "hotPathPlan");
    const dispose = () => {
      if (disposed) return;
      disposed = true;
      session.dispose();
      program.dispose();
    };
    const executionPlan = program.requireExecutionPlan();
    const moduleBindingMetadata = programModuleBindingMetadata(session);
    const inferenceBindings = moduleBindingMetadata?.bindings ?? bindOptions ?? {};
    const parameterBindingPlan = moduleBindingMetadata
      ? moduleBindingPlanForBindings(moduleBindingMetadata.bindings)
      : null;
    const programBindingPlan = bindingPlan(inferenceBindings);
    return Object.freeze({
      native: true,
      program,
      session,
      executionPlan() {
        return executionPlan;
      },
      requireExecutionPlan() {
        return executionPlan;
      },
      forward(input: unknown) {
        return session.stepTensor(input);
      },
      call(input: unknown) {
        return session.stepTensor(input);
      },
      __call__(input: unknown) {
        return session.stepTensor(input);
      },
      stepTensor(input: unknown) {
        return session.stepTensor(input);
      },
      into(output: Float32Array, input: unknown) {
        return session.executeInto(output, { input });
      },
      prepareInto(output: Float32Array, input: unknown) {
        return session.prepareExecuteInto(output, { input });
      },
      explain() {
        return explain(target, compileOptions);
      },
      preflight() {
        return explain(target, compileOptions);
      },
      compileSupport() {
        return compileSupport(target, compileOptions);
      },
      compileEvidence() {
        return compileEvidence();
      },
      parameterBindingPlan() {
        return parameterBindingPlan;
      },
      parameter_binding_plan() {
        return parameterBindingPlan;
      },
      programBindingPlan() {
        return programBindingPlan;
      },
      program_binding_plan() {
        return programBindingPlan;
      },
      requirements() {
        return requirements();
      },
      bufferLayout() {
        return bufferLayout();
      },
      bufferSlotNames() {
        return bufferSlotNames();
      },
      bufferSlot(nameOrKind: unknown) {
        return bufferSlot(nameOrKind);
      },
      sessionBufferLayout() {
        return sessionBufferLayout();
      },
      sessionBufferSlotNames() {
        return sessionBufferSlotNames();
      },
      sessionBufferSlot(nameOrKind: unknown) {
        return sessionBufferSlot(nameOrKind);
      },
      stepContract() {
        return stepContract();
      },
      hotPathPlan(params?: unknown) {
        return hotPathPlan(params);
      },
      inputShape() {
        return inputShape();
      },
      outputShape() {
        return outputShape();
      },
      kernelPlan() {
        return kernelPlan();
      },
      compilerSignatures() {
        return compilerSignatures();
      },
      dispose,
      free: dispose,
    });
  }

  function compileForInference(target: unknown, compileOptions: CompileNamespaceOptions = {}, bindOptions?: unknown) {
    const module = moduleTarget(target, "compileForInference");
    const program = compile(module, compileOptions);
    if (!program || typeof program !== "object") {
      throw new Error("compile.compileForInference expected compile() to return a Program");
    }
    if (typeof (program as Record<string, any>).requireExecutionPlan !== "function") {
      throw new Error("compile.compileForInference requires a native executable Program; use compile.explain for diagnostics-only targets");
    }
    const bindModule = (program as Record<string, any>).bindModule;
    const bind = (program as Record<string, any>).bind;
    const targetCanPlaceModuleParameters = module != null && typeof module === "object" && typeof (module as Record<string, any>).placeParameters === "function";
    const session = typeof bindModule === "function" && targetCanPlaceModuleParameters
      ? bindModule.call(program, module, bindOptions)
      : typeof bind === "function" && bindOptions != null
        ? bind.call(program, bindOptions)
        : null;
    if (!session || typeof session !== "object") {
      throw new Error("compile.compileForInference expected bindModule() or explicit Program bindings to return a Session");
    }
    return compiledInferenceHandle(
      program as Record<string, any>,
      session as Record<string, any>,
      module,
      compileOptions,
      bindOptions,
    );
  }

  function run(target: unknown, input: unknown, compileOptions: CompileNamespaceOptions = {}, bindOptions?: unknown) {
    const inference = compileForInference(target, compileOptions, bindOptions) as { forward(input: unknown): unknown; dispose(): void };
    try {
      return inference.forward(input);
    } finally {
      inference.dispose();
    }
  }

  function runInto(output: Float32Array, target: unknown, input: unknown, compileOptions: CompileNamespaceOptions = {}, bindOptions?: unknown) {
    const inference = compileForInference(target, compileOptions, bindOptions) as { into(output: Float32Array, input: unknown): Float32Array; dispose(): void };
    try {
      return inference.into(output, input);
    } finally {
      inference.dispose();
    }
  }

  function trainingStep(model: unknown, optimizer: unknown, trainingOptions: UnknownRecord = {}) {
    if (typeof options.trainingStep !== "function") {
      throw new Error("compile.trainingStep requires a native training-step runtime");
    }
    return options.trainingStep(model, optimizer, trainingOptions);
  }

  return Object.freeze(Object.assign(compile, {
    compileManifest,
    trace,
    analyze,
    compileSupport,
    compile_support: compileSupport,
    requireCompileSupport,
    require_compile_support: requireCompileSupport,
    assertCompileSupport: requireCompileSupport,
    assert_compile_support: requireCompileSupport,
    canCompile,
    can_compile: canCompile,
    explain,
    preflight: explain,
    compileExplanation: explain,
    compile_explanation: explain,
    compilePlan: explain,
    compile_plan: explain,
    requireCompilePlan,
    require_compile_plan: requireCompilePlan,
    assertCompilePlan: requireCompilePlan,
    assert_compile_plan: requireCompilePlan,
    compilerSignatures,
    compiler_signatures: compilerSignatures,
    tensorProgramIr,
    tensor_program_ir: tensorProgramIr,
    kernelPlan,
    kernel_plan: kernelPlan,
    bufferLayout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "bufferLayout", compileOptions),
    buffer_layout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "bufferLayout", compileOptions),
    memoryLayout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "memoryLayout", compileOptions),
    memory_layout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "memoryLayout", compileOptions),
    inputShape,
    input_shape: inputShape,
    outputShape,
    output_shape: outputShape,
    shapeConstraints: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "shapeConstraints", compileOptions),
    shape_constraints: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "shapeConstraints", compileOptions),
    parameterLayout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "parameterLayout", compileOptions),
    parameter_layout: (target: unknown, compileOptions?: CompileNamespaceOptions) => kernelPlanField(target, "parameterLayout", compileOptions),
    compilerSignatureEvidence,
    compilerSignaturesFromCompilerEvidence,
    compilerEvidenceSignature,
    programCompilerSignaturesFromCompileEvidence,
    completeProgramCompileEvidence,
    isProgramCompileEvidence,
    requireProgramCompileEvidence,
    assertProgramCompileEvidence,
    assert_program_compile_evidence,
    matchesProgramCompileEvidenceSignature,
    matches_program_compile_evidence_signature,
    programCompileEvidenceSignature,
    compile,
    compileForInference,
    compile_for_inference: compileForInference,
    forInference: compileForInference,
    for_inference: compileForInference,
    native: compileForInference,
    inference: compileForInference,
    run,
    infer: run,
    predict: run,
    run_into: runInto,
    runInto,
    infer_into: runInto,
    inferInto: runInto,
    predict_into: runInto,
    predictInto: runInto,
    trainingStep,
    training_step: trainingStep,
    compileForTraining: trainingStep,
    compile_for_training: trainingStep,
    forTraining: trainingStep,
    for_training: trainingStep,
  }));
}

function checkpointLooksLikePath(value: string) {
  if (value.length === 0) return false;
  if (value.includes("/") || value.includes("\\")) return true;
  return /\.(json|ckpt|checkpoint|zgml)$/i.test(value);
}

function checkpointLooksLikeJson(value: string) {
  const trimmed = value.trimStart();
  return trimmed.startsWith("{") || trimmed.startsWith("[");
}

export function createAdapterZgmlCheckpointIo(checkpoint: CheckpointNamespaceLike, options: ZgmlCheckpointIoOptions = {}) {
  function save(snapshot: unknown, pathOrSpace?: string | number, space?: string | number) {
    const hasExplicitPath = space !== undefined;
    const hasImplicitPath = typeof pathOrSpace === "string" && checkpointLooksLikePath(pathOrSpace);
    if (hasExplicitPath || hasImplicitPath) {
      if (typeof pathOrSpace !== "string" || pathOrSpace.length === 0) throw new Error("zgml.save path must be a non-empty string");
      if (typeof options.writeTextFile !== "function") throw new Error("zgml.save path output is unavailable in this runtime");
      const text = checkpoint.stringify(snapshot, space);
      options.writeTextFile(pathOrSpace, text);
      return pathOrSpace;
    }
    return checkpoint.stringify(snapshot, pathOrSpace);
  }

  function load(textOrPath: unknown, targets?: unknown) {
    let text = textOrPath;
    if (typeof textOrPath === "string" && !checkpointLooksLikeJson(textOrPath) && typeof options.readTextFile === "function" && checkpointLooksLikePath(textOrPath)) {
      text = options.readTextFile(textOrPath);
    }
    const snapshot = checkpoint.parse(text);
    return targets === undefined ? snapshot : checkpoint.restore(snapshot, targets);
  }

  return Object.freeze({
    save,
    load,
  });
}

export const createAdapterTorchCheckpointIo = createAdapterZgmlCheckpointIo;

function requireTensorMethod(input: unknown, methodName: string) {
  const source = input as TensorMethodSource | null | undefined;
  const method = source && source[methodName];
  if (typeof method !== "function") {
    throw new Error(`zgml.${methodName} requires a Tensor with ${methodName}()`);
  }
  return method.bind(source) as (...args: unknown[]) => unknown;
}

export function createAdapterZgmlTensorOps() {
  function call(input: unknown, methodName: string, ...args: unknown[]) {
    return requireTensorMethod(input, methodName)(...args);
  }

  return Object.freeze({
    relu: (input: unknown) => call(input, "relu"),
    gelu: (input: unknown) => call(input, "gelu"),
    silu: (input: unknown) => call(input, "silu"),
    sigmoid: (input: unknown) => call(input, "sigmoid"),
    tanh: (input: unknown) => call(input, "tanh"),
    softmax: (input: unknown, dim?: number) => call(input, "softmax", dim),
    softmax_dim: (input: unknown, dim?: number) => call(input, "softmax_dim", dim),
    softmaxDim: (input: unknown, dim?: number) => call(input, "softmaxDim", dim),
    logSoftmax: (input: unknown, dim?: number) => call(input, "logSoftmax", dim),
    log_softmax: (input: unknown, dim?: number) => call(input, "log_softmax", dim),
    log_softmax_dim: (input: unknown, dim?: number) => call(input, "log_softmax_dim", dim),
    logSoftmaxDim: (input: unknown, dim?: number) => call(input, "logSoftmaxDim", dim),
    to: (input: unknown, target?: unknown, options?: unknown) => call(input, "to", target, options),
    cpu: (input: unknown, options?: unknown) => call(input, "cpu", options),
    float: (input: unknown, options?: unknown) => call(input, "float", options),
    float32: (input: unknown, options?: unknown) => call(input, "float32", options),
    typeAs: (input: unknown, other: unknown, options?: unknown) => call(input, "typeAs", other, options),
    type_as: (input: unknown, other: unknown, options?: unknown) => call(input, "type_as", other, options),
    sum: (input: unknown, dim?: number) => call(input, "sum", dim),
    prod: (input: unknown, dim?: number) => call(input, "prod", dim),
    cumsum: (input: unknown, dim?: number) => call(input, "cumsum", dim),
    mean: (input: unknown, dim?: number) => call(input, "mean", dim),
    max: (input: unknown, dim?: number) => call(input, "max", dim),
    min: (input: unknown, dim?: number) => call(input, "min", dim),
    argmax: (input: unknown, dim?: number) => call(input, "argmax", dim),
    argmin: (input: unknown, dim?: number) => call(input, "argmin", dim),
    any: (input: unknown, dim?: number) => call(input, "any", dim),
    all: (input: unknown, dim?: number) => call(input, "all", dim),
    logsumexp: (input: unknown, dim?: number) => call(input, "logsumexp", dim),
    logSumExp: (input: unknown, dim?: number) => call(input, "logSumExp", dim),
    variance: (input: unknown, dim?: number, correction?: number) => call(input, "variance", dim, correction),
    var: (input: unknown, dim?: number, correction?: number) => call(input, "var", dim, correction),
    std: (input: unknown, dim?: number, correction?: number) => call(input, "std", dim, correction),
    norm: (input: unknown, dim?: number, p?: number) => call(input, "norm", dim, p),
    neg: (input: unknown) => call(input, "neg"),
    negative: (input: unknown) => call(input, "negative"),
    expm1: (input: unknown) => call(input, "expm1"),
    log1p: (input: unknown) => call(input, "log1p"),
    sqr: (input: unknown) => call(input, "sqr"),
    square: (input: unknown) => call(input, "square"),
    recip: (input: unknown) => call(input, "recip"),
    reciprocal: (input: unknown) => call(input, "reciprocal"),
    sgn: (input: unknown) => call(input, "sgn"),
    sign: (input: unknown) => call(input, "sign"),
    step: (input: unknown) => call(input, "step"),
    isnan: (input: unknown) => call(input, "isnan"),
    isinf: (input: unknown) => call(input, "isinf"),
    isfinite: (input: unknown) => call(input, "isfinite"),
    floor: (input: unknown) => call(input, "floor"),
    ceil: (input: unknown) => call(input, "ceil"),
    round: (input: unknown) => call(input, "round"),
    trunc: (input: unknown) => call(input, "trunc"),
    sin: (input: unknown) => call(input, "sin"),
    cos: (input: unknown) => call(input, "cos"),
    tan: (input: unknown) => call(input, "tan"),
    sqrt: (input: unknown) => call(input, "sqrt"),
    rsqrt: (input: unknown) => call(input, "rsqrt"),
    exp: (input: unknown) => call(input, "exp"),
    log: (input: unknown) => call(input, "log"),
    abs: (input: unknown) => call(input, "abs"),
    pow: (input: unknown, exponent: number) => call(input, "pow", exponent),
    clamp: (input: unknown, min?: number | null, max?: number | null) => call(input, "clamp", min, max),
    clip: (input: unknown, min?: number | null, max?: number | null) => call(input, "clip", min, max),
    flatten: (input: unknown, startDim?: number, endDim?: number) => call(input, "flatten", startDim, endDim),
    reshape: (input: unknown, shape: readonly number[]) => call(input, "reshape", shape),
    view: (input: unknown, shape: readonly number[]) => call(input, "view", shape),
    squeeze: (input: unknown, dim?: number | null) => call(input, "squeeze", dim),
    unsqueeze: (input: unknown, dim: number) => call(input, "unsqueeze", dim),
    transpose: (input: unknown, dim0?: number, dim1?: number) => call(input, "transpose", dim0, dim1),
    permute: (input: unknown, dims: readonly number[]) => call(input, "permute", dims),
    flip: (input: unknown, dims: readonly number[]) => call(input, "flip", dims),
    roll: (input: unknown, shifts: number | readonly number[], dims?: number | readonly number[] | null) => call(input, "roll", shifts, dims),
    select: (input: unknown, dim: number, index: number) => call(input, "select", dim, index),
    narrow: (input: unknown, dim: number, start: number, length: number) => call(input, "narrow", dim, start, length),
    slice: (input: unknown, dim: number, start?: number | null, end?: number | null, step?: number) => call(input, "slice", dim, start, end, step),
    indexSelect: (input: unknown, dim: number, indices: unknown) => call(input, "indexSelect", dim, indices),
    index_select: (input: unknown, dim: number, indices: unknown) => call(input, "index_select", dim, indices),
    gather: (input: unknown, dim: number, index: unknown) => call(input, "gather", dim, index),
    take: (input: unknown, index: unknown) => call(input, "take", index),
    unbind: (input: unknown, dim?: number) => call(input, "unbind", dim),
    maximum: (input: unknown, other: unknown) => call(input, "maximum", other),
    minimum: (input: unknown, other: unknown) => call(input, "minimum", other),
    where: (condition: unknown, input: unknown, other: unknown) => call(condition, "where", input, other),
    maskedFill: (input: unknown, mask: unknown, value: unknown) => call(input, "maskedFill", mask, value),
    masked_fill: (input: unknown, mask: unknown, value: unknown) => call(input, "masked_fill", mask, value),
    allclose: (actual: unknown, expected: unknown, options?: unknown) => call(actual, "allclose", expected, options),
    equal: (actual: unknown, expected: unknown) => call(actual, "equal", expected),
    argsort: (input: unknown, dim?: number, descending?: boolean) => call(input, "argsort", dim, descending),
    sort: (input: unknown, dim?: number, descending?: boolean) => call(input, "sort", dim, descending),
    topk: (input: unknown, k: number, dim?: number, largest?: boolean, sorted?: boolean) => call(input, "topk", k, dim, largest, sorted),
    split: (input: unknown, splitSizeOrSections: number | readonly number[], dim?: number) => call(input, "split", splitSizeOrSections, dim),
    chunk: (input: unknown, chunks: number, dim?: number) => call(input, "chunk", chunks, dim),
  });
}

export const createAdapterTorchTensorOps = createAdapterZgmlTensorOps;

export function createAdapterZgmlNamespace(options: AdapterZgmlNamespaceOptions) {
  const tensorOps = createAdapterZgmlTensorOps();
  const data = options.data;
  const compileNamespace = options.compile as Record<string, unknown>;
  return Object.freeze({
    Tensor: options.Tensor,
    tensor: options.tensor,
    asTensor: options.tensor,
    as_tensor: options.tensor,
    asarray: options.tensor,
    fromNumpy: options.tensor,
    from_numpy: options.tensor,
    parameter: options.parameter,
    param: options.param,
    ...options.factories,
    relu: tensorOps.relu,
    gelu: tensorOps.gelu,
    silu: tensorOps.silu,
    sigmoid: tensorOps.sigmoid,
    tanh: tensorOps.tanh,
    softmax: tensorOps.softmax,
    softmax_dim: tensorOps.softmax_dim,
    softmaxDim: tensorOps.softmaxDim,
    logSoftmax: tensorOps.logSoftmax,
    log_softmax: tensorOps.log_softmax,
    log_softmax_dim: tensorOps.log_softmax_dim,
    logSoftmaxDim: tensorOps.logSoftmaxDim,
    to: tensorOps.to,
    cpu: tensorOps.cpu,
    float: tensorOps.float,
    float32: tensorOps.float32,
    typeAs: tensorOps.typeAs,
    type_as: tensorOps.type_as,
    sum: tensorOps.sum,
    prod: tensorOps.prod,
    cumsum: tensorOps.cumsum,
    mean: tensorOps.mean,
    max: tensorOps.max,
    min: tensorOps.min,
    argmax: tensorOps.argmax,
    argmin: tensorOps.argmin,
    any: tensorOps.any,
    all: tensorOps.all,
    logsumexp: tensorOps.logsumexp,
    logSumExp: tensorOps.logSumExp,
    variance: tensorOps.variance,
    var: tensorOps.var,
    std: tensorOps.std,
    norm: tensorOps.norm,
    neg: tensorOps.neg,
    negative: tensorOps.negative,
    expm1: tensorOps.expm1,
    log1p: tensorOps.log1p,
    sqr: tensorOps.sqr,
    square: tensorOps.square,
    recip: tensorOps.recip,
    reciprocal: tensorOps.reciprocal,
    sgn: tensorOps.sgn,
    sign: tensorOps.sign,
    step: tensorOps.step,
    isnan: tensorOps.isnan,
    isinf: tensorOps.isinf,
    isfinite: tensorOps.isfinite,
    floor: tensorOps.floor,
    ceil: tensorOps.ceil,
    round: tensorOps.round,
    trunc: tensorOps.trunc,
    sin: tensorOps.sin,
    cos: tensorOps.cos,
    tan: tensorOps.tan,
    sqrt: tensorOps.sqrt,
    rsqrt: tensorOps.rsqrt,
    exp: tensorOps.exp,
    log: tensorOps.log,
    abs: tensorOps.abs,
    pow: tensorOps.pow,
    clamp: tensorOps.clamp,
    clip: tensorOps.clip,
    flatten: tensorOps.flatten,
    reshape: tensorOps.reshape,
    view: tensorOps.view,
    squeeze: tensorOps.squeeze,
    unsqueeze: tensorOps.unsqueeze,
    transpose: tensorOps.transpose,
    permute: tensorOps.permute,
    flip: tensorOps.flip,
    roll: tensorOps.roll,
    select: tensorOps.select,
    narrow: tensorOps.narrow,
    slice: tensorOps.slice,
    indexSelect: tensorOps.indexSelect,
    index_select: tensorOps.index_select,
    gather: tensorOps.gather,
    take: tensorOps.take,
    unbind: tensorOps.unbind,
    maximum: tensorOps.maximum,
    minimum: tensorOps.minimum,
    where: tensorOps.where,
    maskedFill: tensorOps.maskedFill,
    masked_fill: tensorOps.masked_fill,
    allclose: tensorOps.allclose,
    equal: tensorOps.equal,
    argsort: tensorOps.argsort,
    sort: tensorOps.sort,
    topk: tensorOps.topk,
    split: tensorOps.split,
    chunk: tensorOps.chunk,
    hasShape: options.shape.hasShape,
    requireShape: options.shape.requireShape,
    no_grad: options.grad.no_grad,
    noGrad: options.grad.noGrad,
    inference_mode: options.grad.inference_mode,
    inferenceMode: options.grad.inferenceMode,
    enable_grad: options.grad.enable_grad,
    enableGrad: options.grad.enableGrad,
    is_grad_enabled: options.grad.is_grad_enabled,
    isGradEnabled: options.grad.isGradEnabled,
    set_grad_enabled: options.grad.set_grad_enabled,
    setGradEnabled: options.grad.setGradEnabled,
    gradMode: options.gradMode,
    nn: options.nn,
    F: options.F,
    functional: options.F,
    compile: options.compile,
    native: compileNamespace.compileForInference,
    inference: compileNamespace.compileForInference,
    compileInference: compileNamespace.compileForInference,
    compile_inference: compileNamespace.compileForInference,
    compileForInference: compileNamespace.compileForInference,
    compile_for_inference: compileNamespace.compile_for_inference,
    forInference: compileNamespace.compileForInference,
    for_inference: compileNamespace.compile_for_inference,
    run: compileNamespace.run,
    infer: compileNamespace.infer,
    predict: compileNamespace.predict,
    runInto: compileNamespace.runInto,
    run_into: compileNamespace.run_into,
    inferInto: compileNamespace.inferInto,
    infer_into: compileNamespace.infer_into,
    predictInto: compileNamespace.predictInto,
    predict_into: compileNamespace.predict_into,
    trainingStep: compileNamespace.trainingStep,
    training_step: compileNamespace.training_step,
    compileForTraining: compileNamespace.compileForTraining,
    compile_for_training: compileNamespace.compile_for_training,
    forTraining: compileNamespace.forTraining,
    for_training: compileNamespace.for_training,
    nativeEager: options.nativeEager,
    native_eager: options.nativeEager,
    nativeCore: options.nativeCore,
    native_core: options.native_core ?? options.nativeCore,
    lazy: options.lazy,
    optim: options.optim,
    data,
    utils: Object.freeze({
      data: Object.freeze({
        Dataset: data.Dataset,
        TensorDataset: data.TensorDataset,
        DataLoader: data.DataLoader,
        Subset: data.Subset,
        ConcatDataset: data.ConcatDataset,
        MapDataset: data.MapDataset,
        SequentialSampler: data.SequentialSampler,
        RandomSampler: data.RandomSampler,
        BatchSampler: data.BatchSampler,
        tensorDataset: data.tensorDataset,
        tensor_dataset: data.tensor_dataset,
        defaultCollate: data.defaultCollate,
        default_collate: data.default_collate,
        dataLoader: data.dataLoader,
        dataloader: data.dataloader,
        randomSplit: data.randomSplit,
        random_split: data.random_split,
        subset: data.subset,
        take: data.take,
        concatDataset: data.concatDataset,
        concat_dataset: data.concat_dataset,
        mapDataset: data.mapDataset,
        map_dataset: data.map_dataset,
      }),
    }),
    loss: options.loss,
    train: options.train,
    fit: (options.train as Record<string, unknown>).fit,
    fitModule: (options.train as Record<string, unknown>).fitModule,
    fit_module: (options.train as Record<string, unknown>).fit_module,
    fitNative: (options.train as Record<string, unknown>).fitNative,
    fit_native: (options.train as Record<string, unknown>).fit_native,
    checkpoint: options.checkpoint,
    save: options.checkpointIo.save,
    load: options.checkpointIo.load,
    Program: options.Program,
    Session: options.Session,
    NativeBuffer: options.NativeBuffer,
  });
}

export const createAdapterTorchNamespace = createAdapterZgmlNamespace;
