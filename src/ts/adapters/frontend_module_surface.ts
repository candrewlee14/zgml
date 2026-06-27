import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";

type ActivationModuleOptions = Parameters<SharedFrontendRuntime["createActivationModuleClass"]>[0];
type SoftmaxModuleOptions = Parameters<SharedFrontendRuntime["createSoftmaxModuleClass"]>[0];
type ReductionModuleOptions = Parameters<SharedFrontendRuntime["createReductionModuleClass"]>[0];
type DropoutModuleOptions = Parameters<SharedFrontendRuntime["createDropoutModuleClass"]>[0];
type LinearModuleOptions = Parameters<SharedFrontendRuntime["createLinearModuleClass"]>[0];
type EmbeddingModuleOptions = Parameters<SharedFrontendRuntime["createEmbeddingModuleClass"]>[0];
type Conv2dModuleOptions = Parameters<SharedFrontendRuntime["createConv2dModuleClass"]>[0];
type AvgPool2dModuleOptions = Parameters<SharedFrontendRuntime["createAvgPool2dModuleClass"]>[0];
type MaxPool2dModuleOptions = Parameters<SharedFrontendRuntime["createMaxPool2dModuleClass"]>[0];
type FeatureNormModuleOptions = Parameters<SharedFrontendRuntime["createFeatureNormModuleClass"]>[0];
type ShapeModuleOptions = Parameters<SharedFrontendRuntime["createShapeModuleClass"]>[0];
type SequentialModuleOptions = Parameters<SharedFrontendRuntime["createSequentialModuleClass"]>[0];
type ModuleFacadeHelpersOptions = Parameters<SharedFrontendRuntime["createModuleFacadeHelpers"]>[0];

type AdapterTensorConstructor =
  ActivationModuleOptions["Tensor"] &
  LinearModuleOptions["Tensor"] &
  EmbeddingModuleOptions["Tensor"] &
  Conv2dModuleOptions["Tensor"] &
  AvgPool2dModuleOptions["Tensor"] &
  MaxPool2dModuleOptions["Tensor"] &
  FeatureNormModuleOptions["Tensor"] &
  ShapeModuleOptions["Tensor"] &
  SequentialModuleOptions["Tensor"];

type AdapterModuleCompileHooks = Readonly<{
  analyzeSequentialProgram: LinearModuleOptions["analyzeSequentialProgram"] & SequentialModuleOptions["analyzeSequentialProgram"];
  analyzeSingleModuleProgram: ActivationModuleOptions["analyzeSingleModuleProgram"] & EmbeddingModuleOptions["analyzeSingleModuleProgram"] & Conv2dModuleOptions["analyzeSingleModuleProgram"] & MaxPool2dModuleOptions["analyzeSingleModuleProgram"];
  moduleCompileSupport: ActivationModuleOptions["moduleCompileSupport"] & LinearModuleOptions["moduleCompileSupport"] & MaxPool2dModuleOptions["moduleCompileSupport"];
  compileModuleProgram: ActivationModuleOptions["compileModuleProgram"] & LinearModuleOptions["compileModuleProgram"] & Conv2dModuleOptions["compileModuleProgram"] & MaxPool2dModuleOptions["compileModuleProgram"];
  compileTrainingStep?: LinearModuleOptions["compileTrainingStep"] & SequentialModuleOptions["compileTrainingStep"];
  attachProgramCompileEvidence: NonNullable<LinearModuleOptions["attachProgramCompileEvidence"]>;
  packedSequentialProgramParameters: NonNullable<EmbeddingModuleOptions["packedSequentialProgramParameters"] & MaxPool2dModuleOptions["packedSequentialProgramParameters"]>;
  traceSequentialProgram: EmbeddingModuleOptions["traceSequentialProgram"] & SequentialModuleOptions["traceSequentialProgram"] & ModuleFacadeHelpersOptions["traceSequentialProgram"];
}>;

type AdapterModuleStateHooks = Readonly<{
  parameterNames: (paramsOrModule: unknown, prefix?: string) => readonly string[];
  parameterInfos: (paramsOrModule: unknown, prefix?: string) => readonly unknown[];
  parameterInfo: (paramsOrModule: unknown, nameOrIndex: unknown, prefix?: string) => unknown;
  zeroGrad: (paramsOrModule: unknown, options?: Readonly<Record<string, unknown>>) => unknown;
  setRequiresGrad: (paramsOrModule: unknown, requiresGrad?: boolean) => unknown;
  stateKeys: (source: unknown) => readonly string[];
  stateDict: (paramsOrModule: unknown, prefix?: string) => unknown;
  loadStateDict: (paramsOrModule: unknown, source: unknown, options?: Readonly<Record<string, unknown>>) => unknown;
}>;

export type AdapterFrontendModuleSurfaceOptions = AdapterModuleCompileHooks & AdapterModuleStateHooks & Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  Tensor: AdapterTensorConstructor;
  f32: ActivationModuleOptions["f32"] & FeatureNormModuleOptions["f32"] & ShapeModuleOptions["f32"] & SequentialModuleOptions["f32"];
  prepareF32: NonNullable<SequentialModuleOptions["prepareF32"]>;
  f32WithLength: LinearModuleOptions["f32WithLength"];
  indexValues: EmbeddingModuleOptions["indexValues"];
  addTensorGrad: DropoutModuleOptions["addTensorGrad"] & EmbeddingModuleOptions["addTensorGrad"] & Conv2dModuleOptions["addTensorGrad"] & AvgPool2dModuleOptions["addTensorGrad"] & MaxPool2dModuleOptions["addTensorGrad"] & FeatureNormModuleOptions["addTensorGrad"];
  isGradEnabled: NonNullable<DropoutModuleOptions["isGradEnabled"]>;
  requirePositiveInteger: LinearModuleOptions["requirePositiveInteger"] & EmbeddingModuleOptions["requirePositiveInteger"] & Conv2dModuleOptions["requirePositiveInteger"] & FeatureNormModuleOptions["requirePositiveInteger"];
  defaultedF32: LinearModuleOptions["defaultedF32"] & EmbeddingModuleOptions["defaultedF32"] & Conv2dModuleOptions["defaultedF32"] & FeatureNormModuleOptions["defaultedF32"];
  zerosF32: LinearModuleOptions["zerosF32"] & EmbeddingModuleOptions["zerosF32"] & Conv2dModuleOptions["zerosF32"] & FeatureNormModuleOptions["zerosF32"];
  makeParameter: LinearModuleOptions["makeParameter"] & EmbeddingModuleOptions["makeParameter"] & Conv2dModuleOptions["makeParameter"] & FeatureNormModuleOptions["makeParameter"];
  parameterView: LinearModuleOptions["parameterView"] & EmbeddingModuleOptions["parameterView"] & Conv2dModuleOptions["parameterView"] & FeatureNormModuleOptions["parameterView"];
  nativeEagerLinearInto?: LinearModuleOptions["nativeEagerLinearInto"];
  nativeEagerSoftmaxInto?: SoftmaxModuleOptions["nativeEagerSoftmaxInto"];
  nativeEagerLinearActivationInto?: SequentialModuleOptions["nativeEagerLinearActivationInto"];
  TinyLinearModel: LinearModuleOptions["TinyLinearModel"];
}>;

export function createAdapterFrontendModuleSurface(options: AdapterFrontendModuleSurfaceOptions) {
  const parameterlessModuleHooks = {
    Tensor: options.Tensor,
    f32: options.f32,
    isGradEnabled: options.isGradEnabled,
    stateKeys: options.stateKeys,
    setRequiresGrad: options.setRequiresGrad,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    compileModuleProgram: options.compileModuleProgram,
  };

  const ActivationModule = options.sharedFrontend.createActivationModuleClass(parameterlessModuleHooks);
  const SoftmaxModule = options.sharedFrontend.createSoftmaxModuleClass({
    ...parameterlessModuleHooks,
    kind: "softmax",
    tensorMethod: "softmaxDim",
    parameterLabel: "softmax",
    nativeEagerSoftmaxInto: options.nativeEagerSoftmaxInto,
    isGradEnabled: options.isGradEnabled,
  });
  const LogSoftmaxModule = options.sharedFrontend.createSoftmaxModuleClass({
    ...parameterlessModuleHooks,
    kind: "logSoftmax",
    tensorMethod: "logSoftmaxDim",
    parameterLabel: "logSoftmax",
    nativeEagerSoftmaxInto: options.nativeEagerSoftmaxInto,
    isGradEnabled: options.isGradEnabled,
  });
  const ReductionModule = options.sharedFrontend.createReductionModuleClass(parameterlessModuleHooks);
  const DropoutModule = options.sharedFrontend.createDropoutModuleClass({
    ...parameterlessModuleHooks,
    addTensorGrad: options.addTensorGrad,
  });

  const LinearModule = options.sharedFrontend.createLinearModuleClass({
    Tensor: options.Tensor,
    f32WithLength: options.f32WithLength,
    requirePositiveInteger: options.requirePositiveInteger,
    defaultedF32: options.defaultedF32,
    zerosF32: options.zerosF32,
    makeParameter: options.makeParameter,
    parameterView: options.parameterView,
    nativeEagerLinearInto: options.nativeEagerLinearInto,
    isGradEnabled: options.isGradEnabled,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    zeroGrad: options.zeroGrad,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    analyzeSequentialProgram: options.analyzeSequentialProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    TinyLinearModel: options.TinyLinearModel,
    compileModuleProgram: options.compileModuleProgram,
    compileTrainingStep: options.compileTrainingStep,
    attachProgramCompileEvidence: options.attachProgramCompileEvidence,
  });

  const EmbeddingModule = options.sharedFrontend.createEmbeddingModuleClass({
    Tensor: options.Tensor,
    indexValues: options.indexValues,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    requirePositiveInteger: options.requirePositiveInteger,
    defaultedF32: options.defaultedF32,
    zerosF32: options.zerosF32,
    makeParameter: options.makeParameter,
    parameterView: options.parameterView,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    zeroGrad: options.zeroGrad,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    traceSequentialProgram: options.traceSequentialProgram,
    freezeSequentialTrace: options.sharedFrontend.freezeSequentialTrace,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
    compileModuleProgram: options.compileModuleProgram,
    compileTrainingStep: options.compileTrainingStep,
    attachProgramCompileEvidence: options.attachProgramCompileEvidence,
  });

  const Conv2dModule = options.sharedFrontend.createConv2dModuleClass({
    Tensor: options.Tensor,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    requirePositiveInteger: options.requirePositiveInteger,
    defaultedF32: options.defaultedF32,
    zerosF32: options.zerosF32,
    makeParameter: options.makeParameter,
    parameterView: options.parameterView,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    zeroGrad: options.zeroGrad,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    moduleCompileSupport: options.moduleCompileSupport,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    compileModuleProgram: options.compileModuleProgram,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
  });

  const MaxPool2dModule = options.sharedFrontend.createMaxPool2dModuleClass({
    Tensor: options.Tensor,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
    compileModuleProgram: options.compileModuleProgram,
  });
  const AvgPool2dModule = options.sharedFrontend.createAvgPool2dModuleClass({
    Tensor: options.Tensor,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
    compileModuleProgram: options.compileModuleProgram,
  });

  const FeatureNormModule = options.sharedFrontend.createFeatureNormModuleClass({
    Tensor: options.Tensor,
    f32: options.f32,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    requirePositiveInteger: options.requirePositiveInteger,
    defaultedF32: options.defaultedF32,
    zerosF32: options.zerosF32,
    makeParameter: options.makeParameter,
    parameterView: options.parameterView,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    zeroGrad: options.zeroGrad,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    compileModuleProgram: options.compileModuleProgram,
  });

  const ShapeModule = options.sharedFrontend.createShapeModuleClass({
    Tensor: options.Tensor,
    f32: options.f32,
    stateKeys: options.stateKeys,
    setRequiresGrad: options.setRequiresGrad,
    analyzeSingleModuleProgram: options.analyzeSingleModuleProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
    compileModuleProgram: options.compileModuleProgram,
    attachProgramCompileEvidence: options.attachProgramCompileEvidence,
  });

  let SequentialModule: ReturnType<SharedFrontendRuntime["createSequentialModuleClass"]> | undefined;
  const traceModuleCompiler = options.sharedFrontend.createTraceModuleCompiler({
    getConstructors: () => ({
      LinearModule,
      ActivationModule,
      SequentialModule,
      EmbeddingModule,
      Conv2dModule,
      AvgPool2dModule,
      MaxPool2dModule,
      SoftmaxModule,
      LogSoftmaxModule,
      ReductionModule,
      DropoutModule,
      FeatureNormModule,
      ShapeModule,
    }),
  });

  SequentialModule = options.sharedFrontend.createSequentialModuleClass({
    Tensor: options.Tensor,
    f32: options.f32,
    prepareF32: options.prepareF32,
    isGradEnabled: options.isGradEnabled,
    nativeEagerLinearActivationInto: options.nativeEagerLinearActivationInto,
    zeroGrad: options.zeroGrad,
    parameterNames: options.parameterNames,
    parameterInfos: options.parameterInfos,
    parameterInfo: options.parameterInfo,
    setRequiresGrad: options.setRequiresGrad,
    stateDict: options.stateDict,
    loadStateDict: options.loadStateDict,
    traceSequentialProgram: options.traceSequentialProgram,
    analyzeSequentialProgram: options.analyzeSequentialProgram,
    moduleCompileSupport: options.moduleCompileSupport,
    packedSequentialProgramParameters: options.packedSequentialProgramParameters,
    compileModuleProgram: options.compileModuleProgram,
    attachProgramCompileEvidence: options.attachProgramCompileEvidence,
  });

  const moduleFacadeHelpers = options.sharedFrontend.createModuleFacadeHelpers({
    traceSequentialProgram: options.traceSequentialProgram,
  });

  return Object.freeze({
    traceModuleCompiler,
    ActivationModule,
    SoftmaxModule,
    LogSoftmaxModule,
    ReductionModule,
    DropoutModule,
    LinearModule,
    EmbeddingModule,
    Conv2dModule,
    AvgPool2dModule,
    MaxPool2dModule,
    FeatureNormModule,
    ShapeModule,
    SequentialModule,
    moduleFacadeHelpers,
  });
}
