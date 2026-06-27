// @ts-nocheck

import * as sharedFrontend from "./shared_frontend.js";
import type * as PublicApi from "./public_api.js";
import { Tensor, setAdapterTensorSurfaceHelpers } from "./adapters/tensor_surface.js";
import { nnManifest as nnProductManifest } from "./nn.js";
import { tensorManifest as tensorProductManifest } from "./tensor.js";
import { createAdapterTensorRuntimeSurface } from "./adapters/tensor_runtime_surface.js";
import { createAdapterFrontendModuleSurface } from "./adapters/frontend_module_surface.js";
import { createAdapterCompileNamespace, createAdapterFrontendNamespaces, createAdapterZgmlCheckpointIo, createAdapterZgmlTensorOps } from "./adapters/frontend_namespace_surface.js";
import { createAdapterGradModeSurface } from "./adapters/grad_mode_surface.js";
import { createHostAdapterIndexValuesSurface } from "./runtime/host_adapter_surfaces.js";
import { attachProgramCompileEvidence } from "./runtime/module_program_evidence.js";
import { normalizeDim, rowMajorStrides } from "./core/shape.js";

const unsupportedBrowserCompileReason = "zgml/browser provides eager TS execution; native browser Program execution requires the Wasm/WebGPU runtime adapter";

const {
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
} = sharedFrontend;

function rejectBrowserNative(label: string): never {
  throw new Error(`${label} is unavailable in zgml/browser: ${unsupportedBrowserCompileReason}`);
}

class BrowserNativeBuffer {
  static create() { return rejectBrowserNative("NativeBuffer.create"); }
  static wrapBytes() { return rejectBrowserNative("NativeBuffer.wrapBytes"); }
  static wrapFloat32() { return rejectBrowserNative("NativeBuffer.wrapFloat32"); }
  static externalResource() { return rejectBrowserNative("NativeBuffer.externalResource"); }
  static fromFloat32() { return rejectBrowserNative("NativeBuffer.fromFloat32"); }
  static fromBytes() { return rejectBrowserNative("NativeBuffer.fromBytes"); }
}

class BrowserProgram {
  constructor() { rejectBrowserNative("Program"); }
}

class BrowserSession {
  constructor() { rejectBrowserNative("Session"); }
}

class BrowserTinyLinearModel {
  static create(desc = {}) {
    return new BrowserTinyLinearModel(desc);
  }

  constructor(desc = {}) {
    this.desc = Object.freeze({ ...desc });
  }

  compile() {
    return rejectBrowserNative("TinyLinearModel.compile");
  }
}

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
  tensorMathSurfaceHelpers,
  sessionTensorHelpers,
  tensorMetadataHelpers,
  tensorGradStateHelpers,
  tensorHostSurface,
} = createAdapterTensorRuntimeSurface({
  sharedFrontend,
  Tensor,
  isGradEnabled,
  meanSquaredError: (tensor, target) => tensor.sub(target).pow(2).mean(),
  dtype: () => "f32",
  device: () => "cpu",
  rowMajorStrides,
  normalizeDim,
  isNativeBuffer: (value) => value instanceof BrowserNativeBuffer,
  nativeBufferFromFloat32: () => rejectBrowserNative("Tensor.toNativeBuffer"),
});

void rawF32;
void byteView;
void tensorCoreHelpers;
void sessionTensorHelpers;
void tensorMetadataHelpers;
void tensorGradStateHelpers;

const tensorFactoryHelpers = tensorHostSurface.tensorFactoryHelpers;
const tensorViewSurfaceHelpers = tensorHostSurface.tensorViewSurfaceHelpers;
const tensorPlacementHelpers = tensorHostSurface.tensorPlacementHelpers;
const tensorInfoSurfaceHelpers = tensorHostSurface.tensorInfoSurfaceHelpers;
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

const { indexValues } = createHostAdapterIndexValuesSurface({ Tensor });

const moduleState = sharedFrontend.createModuleStateHelpers({
  Tensor,
  f32WithLength,
  defaultLayout: "row-major",
});

let traceModuleCompiler;
function analyzeSequentialProgram(layers, options = {}) {
  return traceModuleCompiler.analyze(layers, options);
}
function analyzeSingleModuleProgram(layer, options = {}) {
  return traceModuleCompiler.analyzeSingle(layer, options);
}
function traceSequentialProgram(layers, options = {}) {
  return traceModuleCompiler.trace(layers, options);
}
function packedSequentialProgramParameters(spec) {
  return traceModuleCompiler.packParameters(spec);
}

function compileModuleProgram() {
  return rejectBrowserNative("Module.compile");
}

const browserModuleCompileSupport = (supported, reason, extra = {}) => moduleState.moduleCompileSupport(
  false,
  reason || unsupportedBrowserCompileReason,
  {
    ...extra,
    browserRuntime: true,
    nativeProgramAvailable: false,
    supported: false,
  },
);

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
  makeParameter: moduleState.makeParameter,
  parameterView: moduleState.parameterView,
  parameterNames: moduleState.parameterNames,
  parameterInfos: moduleState.parameterInfos,
  parameterInfo: moduleState.parameterInfo,
  zeroGrad: moduleState.zeroGrad,
  setRequiresGrad: moduleState.setRequiresGrad,
  stateKeys: moduleState.stateKeys,
  stateDict: moduleState.stateDict,
  loadStateDict: moduleState.loadStateDict,
  analyzeSequentialProgram,
  analyzeSingleModuleProgram,
  moduleCompileSupport: browserModuleCompileSupport,
  TinyLinearModel: BrowserTinyLinearModel,
  compileModuleProgram,
  attachProgramCompileEvidence,
  packedSequentialProgramParameters,
  traceSequentialProgram,
});
traceModuleCompiler = adapterFrontendModuleSurface.traceModuleCompiler;

const {
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
const {
  parameter: rawParameter,
  cat: rawCat,
  stack: rawStack,
  einsum: rawEinsum,
} = tensorFacade;
const TensorClass = Tensor as unknown as typeof PublicApi.Tensor;
const parameter = rawParameter as typeof PublicApi.parameter;
const param = parameter;
const cat = rawCat as typeof PublicApi.cat;
const concat = rawCat as typeof PublicApi.concat;
const concatenate = rawCat as typeof PublicApi.concatenate;
const stack = rawStack as typeof PublicApi.stack;
const vstack = tensorStaticHelpers.vstack as typeof PublicApi.vstack;
const hstack = tensorStaticHelpers.hstack as typeof PublicApi.hstack;
const einsum = rawEinsum as typeof PublicApi.einsum;
const tensor = Object.assign(tensorFacade.tensor, {
  tensorManifest: tensorProductManifest,
}) as typeof PublicApi.tensor & Readonly<{ tensorManifest: typeof tensorProductManifest }>;
export const asTensor = tensor;
export const as_tensor = tensor;
export const asarray = tensor;
export const fromNumpy = tensor;
export const from_numpy = tensor;

const publicNamespaces = createAdapterFrontendNamespaces({
  sharedFrontend,
  Tensor,
  f32,
  f32WithLength,
  indexValues,
  addTensorGrad,
  isGradEnabled,
  makeParameter: moduleState.makeParameter,
  zeroGrad: moduleState.zeroGrad,
  resolveParameters: moduleState.resolveParameters,
  finiteConfigNumber: moduleState.finiteConfigNumber,
  optimizerStateEntry: moduleState.optimizerStateEntry,
  optimizerStateDict: moduleState.optimizerStateDict,
  optimizerStateEntries: moduleState.optimizerStateEntries,
  optimizerStepFromState: moduleState.optimizerStepFromState,
  loadOptimizerTensorState: moduleState.loadOptimizerTensorState,
  rejectUnexpectedOptimizerState: moduleState.rejectUnexpectedOptimizerState,
  LinearModule: adapterFrontendModuleSurface.LinearModule,
  EmbeddingModule: adapterFrontendModuleSurface.EmbeddingModule,
  Conv2dModule: adapterFrontendModuleSurface.Conv2dModule,
  AvgPool2dModule: adapterFrontendModuleSurface.AvgPool2dModule,
  MaxPool2dModule: adapterFrontendModuleSurface.MaxPool2dModule,
  SequentialModule: adapterFrontendModuleSurface.SequentialModule,
  ActivationModule: adapterFrontendModuleSurface.ActivationModule,
  SoftmaxModule: adapterFrontendModuleSurface.SoftmaxModule,
  LogSoftmaxModule: adapterFrontendModuleSurface.LogSoftmaxModule,
  ReductionModule: adapterFrontendModuleSurface.ReductionModule,
  DropoutModule: adapterFrontendModuleSurface.DropoutModule,
  ShapeModule: adapterFrontendModuleSurface.ShapeModule,
  FeatureNormModule: adapterFrontendModuleSurface.FeatureNormModule,
  geluScalar,
  siluScalar,
  parameterNames: moduleState.parameterNames,
  parameterInfos: moduleState.parameterInfos,
  parameterInfo: moduleState.parameterInfo,
  setRequiresGrad: moduleState.setRequiresGrad,
  stateDict: moduleState.stateDict,
  loadStateDict: moduleState.loadStateDict,
  traceSequentialProgram,
  traceModule: traceModuleCompiler.trace,
  compileSupportForModule: browserModuleCompileSupport,
  requireCompileSupportForModule: (_target, _options = {}) => rejectBrowserNative("nn.requireCompileSupport"),
  compilerSignaturesForModule: () => null,
  tensorProgramIrForModule: () => null,
  kernelPlanForModule: () => null,
  bufferLayoutForModule: () => null,
  memoryLayoutForModule: () => null,
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

export {
  TensorClass as Tensor,
  BrowserNativeBuffer as NativeBuffer,
  BrowserProgram as Program,
  BrowserSession as Session,
  BrowserTinyLinearModel as TinyLinearModel,
  BrowserTinyLinearModel as TinyLinear,
  tensor,
  parameter,
  parameter as param,
  cat,
  concat,
  concatenate,
  stack,
  vstack,
  hstack,
  einsum,
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
};

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

export const full = tensorFactoryHelpers.full as typeof PublicApi.full;
export const fullLike = tensorFactoryHelpers.fullLike as typeof PublicApi.fullLike;
export const full_like = tensorFactoryHelpers.full_like as typeof PublicApi.full_like;
export const empty = tensorFactoryHelpers.empty as typeof PublicApi.empty;
export const emptyLike = tensorFactoryHelpers.emptyLike as typeof PublicApi.emptyLike;
export const empty_like = tensorFactoryHelpers.empty_like as typeof PublicApi.empty_like;
export const zeros = tensorFactoryHelpers.zeros as typeof PublicApi.zeros;
export const zerosLike = tensorFactoryHelpers.zerosLike as typeof PublicApi.zerosLike;
export const zeros_like = tensorFactoryHelpers.zeros_like as typeof PublicApi.zeros_like;
export const ones = tensorFactoryHelpers.ones as typeof PublicApi.ones;
export const onesLike = tensorFactoryHelpers.onesLike as typeof PublicApi.onesLike;
export const ones_like = tensorFactoryHelpers.ones_like as typeof PublicApi.ones_like;
export const eye = tensorFactoryHelpers.eye as typeof PublicApi.eye;
export const scalar = tensorFactoryHelpers.scalar as typeof PublicApi.scalar;
export const rand = tensorFactoryHelpers.rand as typeof PublicApi.rand;
export const randLike = tensorFactoryHelpers.randLike as typeof PublicApi.randLike;
export const rand_like = tensorFactoryHelpers.rand_like as typeof PublicApi.rand_like;
export const randn = tensorFactoryHelpers.randn as typeof PublicApi.randn;
export const randnLike = tensorFactoryHelpers.randnLike as typeof PublicApi.randnLike;
export const randn_like = tensorFactoryHelpers.randn_like as typeof PublicApi.randn_like;
export const randInt = tensorFactoryHelpers.randInt as typeof PublicApi.randInt;
export const randint = tensorFactoryHelpers.randint as typeof PublicApi.randint;
export const randPerm = tensorFactoryHelpers.randPerm as typeof PublicApi.randPerm;
export const randperm = tensorFactoryHelpers.randperm as typeof PublicApi.randperm;
export const manualSeed = tensorFactoryHelpers.manualSeed;
export const manual_seed = tensorFactoryHelpers.manual_seed;
export const initialSeed = tensorFactoryHelpers.initialSeed;
export const initial_seed = tensorFactoryHelpers.initial_seed;
export const seededRng = tensorFactoryHelpers.seededRng;
export const linspace = tensorFactoryHelpers.linspace as typeof PublicApi.linspace;
export const arange = tensorFactoryHelpers.arange as typeof PublicApi.arange;
export const hasShape = tensorHostSurface.tensorRootOps.hasShape as typeof PublicApi.hasShape;
export const requireShape = tensorHostSurface.tensorRootOps.requireShape as typeof PublicApi.requireShape;

export const allclose = tensorStaticHelpers.allclose as typeof PublicApi.allclose;
export const equal = tensorStaticHelpers.equal as typeof PublicApi.equal;
export const isclose = tensorStaticHelpers.isclose as typeof PublicApi.isclose;
export const to = tensorStaticHelpers.to as typeof PublicApi.to;
export const cpu = tensorStaticHelpers.cpu as typeof PublicApi.cpu;
export const float = tensorStaticHelpers.float as typeof PublicApi.float;
export const float32 = tensorStaticHelpers.float32 as typeof PublicApi.float32;
export const typeAs = tensorStaticHelpers.typeAs as typeof PublicApi.typeAs;
export const type_as = tensorStaticHelpers.type_as as typeof PublicApi.type_as;
export const clone = tensorStaticHelpers.clone as typeof PublicApi.clone;
export const detach = tensorStaticHelpers.detach as typeof PublicApi.detach;
export const reshape = tensorStaticHelpers.reshape as typeof PublicApi.reshape;
export const view = tensorStaticHelpers.view as typeof PublicApi.view;
export const broadcastTo = tensorStaticHelpers.broadcastTo as typeof PublicApi.broadcastTo;
export const expand = tensorStaticHelpers.expand as typeof PublicApi.expand;
export const repeat = tensorStaticHelpers.repeat as typeof PublicApi.repeat;
export const tile = tensorStaticHelpers.tile as typeof PublicApi.tile;
export const flatten = tensorStaticHelpers.flatten as typeof PublicApi.flatten;
export const squeeze = tensorStaticHelpers.squeeze as typeof PublicApi.squeeze;
export const unsqueeze = tensorStaticHelpers.unsqueeze as typeof PublicApi.unsqueeze;
export const transpose = tensorStaticHelpers.transpose as typeof PublicApi.transpose;
export const permute = tensorStaticHelpers.permute as typeof PublicApi.permute;
export const flip = tensorStaticHelpers.flip as typeof PublicApi.flip;
export const roll = tensorStaticHelpers.roll as typeof PublicApi.roll;
export const select = tensorStaticHelpers.select as typeof PublicApi.select;
export const narrow = tensorStaticHelpers.narrow as typeof PublicApi.narrow;
export const slice = tensorStaticHelpers.slice as typeof PublicApi.slice;
export const indexSelect = tensorStaticHelpers.indexSelect as typeof PublicApi.indexSelect;
export const index_select = tensorStaticHelpers.index_select as typeof PublicApi.index_select;
export const gather = tensorStaticHelpers.gather as typeof PublicApi.gather;
export const take = tensorStaticHelpers.take as typeof PublicApi.take;
export const argsort = tensorStaticHelpers.argsort as typeof PublicApi.argsort;
export const sort = tensorStaticHelpers.sort as typeof PublicApi.sort;
export const topk = tensorStaticHelpers.topk as typeof PublicApi.topk;
export const scatterAdd = tensorStaticHelpers.scatterAdd as typeof PublicApi.scatterAdd;
export const scatter_add = tensorStaticHelpers.scatter_add as typeof PublicApi.scatter_add;
export const split = tensorStaticHelpers.split as typeof PublicApi.split;
export const chunk = tensorStaticHelpers.chunk as typeof PublicApi.chunk;
export const unbind = tensorStaticHelpers.unbind as typeof PublicApi.unbind;
export const add = tensorStaticHelpers.add as typeof PublicApi.add;
export const sub = tensorStaticHelpers.sub as typeof PublicApi.sub;
export const mul = tensorStaticHelpers.mul as typeof PublicApi.mul;
export const div = tensorStaticHelpers.div as typeof PublicApi.div;
export const eq = tensorStaticHelpers.eq as typeof PublicApi.eq;
export const ne = tensorStaticHelpers.ne as typeof PublicApi.ne;
export const lt = tensorStaticHelpers.lt as typeof PublicApi.lt;
export const le = tensorStaticHelpers.le as typeof PublicApi.le;
export const gt = tensorStaticHelpers.gt as typeof PublicApi.gt;
export const ge = tensorStaticHelpers.ge as typeof PublicApi.ge;
export const pow = tensorStaticHelpers.pow as typeof PublicApi.pow;
export const neg = tensorStaticHelpers.neg as typeof PublicApi.neg;
export const negative = tensorStaticHelpers.negative as typeof PublicApi.negative;
export const exp = tensorStaticHelpers.exp as typeof PublicApi.exp;
export const expm1 = tensorStaticHelpers.expm1 as typeof PublicApi.expm1;
export const log = tensorStaticHelpers.log as typeof PublicApi.log;
export const log1p = tensorStaticHelpers.log1p as typeof PublicApi.log1p;
export const sqr = tensorStaticHelpers.sqr as typeof PublicApi.sqr;
export const square = tensorStaticHelpers.square as typeof PublicApi.square;
export const recip = tensorStaticHelpers.recip as typeof PublicApi.recip;
export const reciprocal = tensorStaticHelpers.reciprocal as typeof PublicApi.reciprocal;
export const abs = tensorStaticHelpers.abs as typeof PublicApi.abs;
export const sgn = tensorStaticHelpers.sgn as typeof PublicApi.sgn;
export const sign = tensorStaticHelpers.sign as typeof PublicApi.sign;
export const step = tensorStaticHelpers.step as typeof PublicApi.step;
export const isnan = tensorStaticHelpers.isnan as typeof PublicApi.isnan;
export const isinf = tensorStaticHelpers.isinf as typeof PublicApi.isinf;
export const isfinite = tensorStaticHelpers.isfinite as typeof PublicApi.isfinite;
export const floor = tensorStaticHelpers.floor as typeof PublicApi.floor;
export const ceil = tensorStaticHelpers.ceil as typeof PublicApi.ceil;
export const round = tensorStaticHelpers.round as typeof PublicApi.round;
export const trunc = tensorStaticHelpers.trunc as typeof PublicApi.trunc;
export const sqrt = tensorStaticHelpers.sqrt as typeof PublicApi.sqrt;
export const rsqrt = tensorStaticHelpers.rsqrt as typeof PublicApi.rsqrt;
export const relu = tensorStaticHelpers.relu as typeof PublicApi.relu;
export const gelu = tensorStaticHelpers.gelu as typeof PublicApi.gelu;
export const silu = tensorStaticHelpers.silu as typeof PublicApi.silu;
export const sigmoid = tensorStaticHelpers.sigmoid as typeof PublicApi.sigmoid;
export const tanh = tensorStaticHelpers.tanh as typeof PublicApi.tanh;
export const sin = tensorStaticHelpers.sin as typeof PublicApi.sin;
export const cos = tensorStaticHelpers.cos as typeof PublicApi.cos;
export const tan = tensorStaticHelpers.tan as typeof PublicApi.tan;
export const maximum = tensorStaticHelpers.maximum as typeof PublicApi.maximum;
export const minimum = tensorStaticHelpers.minimum as typeof PublicApi.minimum;
export const where = tensorStaticHelpers.where as typeof PublicApi.where;
export const maskedFill = tensorStaticHelpers.maskedFill as typeof PublicApi.maskedFill;
export const masked_fill = tensorStaticHelpers.masked_fill as typeof PublicApi.masked_fill;
export const sum = tensorStaticHelpers.sum as typeof PublicApi.sum;
export const prod = tensorStaticHelpers.prod as typeof PublicApi.prod;
export const cumsum = tensorStaticHelpers.cumsum as typeof PublicApi.cumsum;
export const mean = tensorStaticHelpers.mean as typeof PublicApi.mean;
export const max = tensorStaticHelpers.max as typeof PublicApi.max;
export const min = tensorStaticHelpers.min as typeof PublicApi.min;
export const any = tensorStaticHelpers.any as typeof PublicApi.any;
export const all = tensorStaticHelpers.all as typeof PublicApi.all;
export const argmax = tensorStaticHelpers.argmax as typeof PublicApi.argmax;
export const argmin = tensorStaticHelpers.argmin as typeof PublicApi.argmin;
export const variance = tensorStaticHelpers.variance as typeof PublicApi.variance;
export const std = tensorStaticHelpers.std as typeof PublicApi.std;
export const norm = tensorStaticHelpers.norm as typeof PublicApi.norm;
export const softmax = tensorStaticHelpers.softmax as typeof PublicApi.softmax;
export const softmax_dim = tensorStaticHelpers.softmax_dim as typeof PublicApi.softmax_dim;
export const softmaxDim = tensorStaticHelpers.softmaxDim as typeof PublicApi.softmaxDim;
export const logSoftmax = tensorStaticHelpers.logSoftmax as typeof PublicApi.logSoftmax;
export const log_softmax = tensorStaticHelpers.log_softmax as typeof PublicApi.log_softmax;
export const log_softmax_dim = tensorStaticHelpers.log_softmax_dim as typeof PublicApi.log_softmax_dim;
export const logSoftmaxDim = tensorStaticHelpers.logSoftmaxDim as typeof PublicApi.logSoftmaxDim;
export const logsumexp = tensorStaticHelpers.logsumexp as typeof PublicApi.logsumexp;
export const logSumExp = tensorStaticHelpers.logSumExp as typeof PublicApi.logSumExp;
export const clamp = tensorStaticHelpers.clamp as typeof PublicApi.clamp;
export const clip = tensorStaticHelpers.clip as typeof PublicApi.clip;
export const matmul = tensorStaticHelpers.matmul as typeof PublicApi.matmul;
export const mm = tensorStaticHelpers.mm as typeof PublicApi.mm;
export const dot = tensorStaticHelpers.dot as typeof PublicApi.dot;
export const trace = tensorStaticHelpers.trace as typeof PublicApi.trace;
export const diagonal = tensorStaticHelpers.diagonal as typeof PublicApi.diagonal;
export const bmm = tensorStaticHelpers.bmm as typeof PublicApi.bmm;

export type BrowserTensor<Shape extends PublicApi.TensorShapeTuple = PublicApi.TensorShapeTuple> = PublicApi.Tensor<Shape>;
export type BrowserParameter<Shape extends PublicApi.TensorShapeTuple = PublicApi.TensorShapeTuple> = PublicApi.NnParameter<Shape>;
export type BrowserCompileSupport<
  InputShape extends PublicApi.TensorShapeTuple = PublicApi.TensorShapeTuple,
  OutputShape extends PublicApi.TensorShapeTuple = PublicApi.TensorShapeTuple,
> = PublicApi.ModuleCompileSupport<InputShape, OutputShape> & Readonly<{
  supported: false;
  nativeProgramAvailable?: false;
}>;
export type BrowserLinearModule<InFeatures extends number = number, OutFeatures extends number = number> =
  Omit<PublicApi.LinearModule<InFeatures, OutFeatures>, "compile" | "compileSupport" | "requireCompileSupport" | "compilePlan" | "requireCompilePlan"> &
  Readonly<{
  forward<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: BrowserTensor<InputShape>): BrowserTensor<PublicApi.LinearForwardShape<InputShape, OutFeatures>>;
  forward(input: PublicApi.TensorLike): BrowserTensor;
  call<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: BrowserTensor<InputShape>): BrowserTensor<PublicApi.LinearForwardShape<InputShape, OutFeatures>>;
  call(input: PublicApi.TensorLike): BrowserTensor;
  __call__<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: BrowserTensor<InputShape>): BrowserTensor<PublicApi.LinearForwardShape<InputShape, OutFeatures>>;
  __call__(input: PublicApi.TensorLike): BrowserTensor;
  parameters(): readonly BrowserParameter[];
  compileSupport<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(options: PublicApi.CompileOptionsWithInputShape<InputShape>): BrowserCompileSupport<InputShape, PublicApi.LinearForwardShape<InputShape, OutFeatures>>;
  compileSupport(options?: PublicApi.CompileOptions): BrowserCompileSupport;
  requireCompileSupport(options?: PublicApi.CompileOptions): never;
  compilePlan<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(options: PublicApi.CompileOptionsWithInputShape<InputShape>): PublicApi.ModuleCompileExplanation<InputShape, PublicApi.LinearForwardShape<InputShape, OutFeatures>>;
  compilePlan(options?: PublicApi.CompileOptions): PublicApi.ModuleCompileExplanation;
  requireCompilePlan(options?: PublicApi.CompileOptions): never;
  compile(options?: PublicApi.CompileOptions): never;
}>;
export type BrowserOptimizer<Kind extends PublicApi.OptimizerStateKind = PublicApi.OptimizerStateKind> = PublicApi.Optimizer<Kind>;
export type BrowserLossModule = Readonly<{
  forward(prediction: BrowserTensor, target: PublicApi.TensorLike): BrowserTensor<readonly [1]>;
  call(prediction: BrowserTensor, target: PublicApi.TensorLike): BrowserTensor<readonly [1]>;
  __call__(prediction: BrowserTensor, target: PublicApi.TensorLike): BrowserTensor<readonly [1]>;
}>;
export type BrowserFitEvidence<Kind extends PublicApi.OptimizerStateKind | null = PublicApi.OptimizerStateKind | null> = PublicApi.TrainFitEvidence<Kind>;
type BrowserNnNamespace = Omit<typeof publicNamespaces.nn, "linear"> & Readonly<{
  linear<const InFeatures extends number, const OutFeatures extends number>(inFeatures: InFeatures, outFeatures: OutFeatures, config?: PublicApi.NnLinearConfig): BrowserLinearModule<InFeatures, OutFeatures>;
  nnManifest: typeof nnProductManifest;
}>;
type BrowserOptimNamespace = Omit<typeof publicNamespaces.optim, "sgd"> & Readonly<{
  sgd(paramsOrModule: PublicApi.OptimizerTarget, config?: PublicApi.SGDConfig): BrowserOptimizer<"sgd">;
}>;
type BrowserLossNamespace = Omit<typeof publicNamespaces.loss, "mseLoss" | "mse_loss"> & Readonly<{
  mseLoss(options?: PublicApi.LossReductionOptions): BrowserLossModule;
  mse_loss(options?: PublicApi.LossReductionOptions): BrowserLossModule;
}>;
type BrowserTrainNamespace = Omit<PublicApi.PublicTrainNamespace, "fit"> & Readonly<{
  fit<const Kind extends PublicApi.OptimizerStateKind, Batch>(
    optimizer: BrowserOptimizer<Kind>,
    data: Iterable<Batch>,
    criterion: (batch: Batch, context: PublicApi.TrainFitContext) => BrowserTensor,
    options?: PublicApi.TrainFitOptions<Kind>,
  ): BrowserFitEvidence<Kind>;
}>;

export const {
  meanSquaredError,
  classTargets,
  crossEntropy,
  SgdOptimizer,
  AdamOptimizer,
  AdamWOptimizer,
  RMSpropOptimizer,
  AdagradOptimizer,
  checkpoint,
} = publicNamespaces;

export const data = publicNamespaces.data as PublicApi.PublicDataNamespace;
export const loss = publicNamespaces.loss as BrowserLossNamespace;
export const train = publicNamespaces.train as BrowserTrainNamespace;
export const optim = publicNamespaces.optim as BrowserOptimNamespace;
export const nn = Object.freeze({
  ...publicNamespaces.nn,
  nnManifest: nnProductManifest,
}) as BrowserNnNamespace;
export const F = nn.F;
export const compile = createAdapterCompileNamespace({
  traceSequentialProgram,
  analyzeSequentialProgram,
});
const zgmlCheckpointIo = createAdapterZgmlCheckpointIo(checkpoint);
export const save = zgmlCheckpointIo.save as typeof PublicApi.save;
export const load = zgmlCheckpointIo.load as typeof PublicApi.load;
const zgmlTensorOps = createAdapterZgmlTensorOps();
export const zgml = Object.freeze({
  Tensor,
  tensor,
  asTensor: tensor,
  as_tensor: tensor,
  asarray: tensor,
  fromNumpy: tensor,
  from_numpy: tensor,
  parameter,
  param,
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
  relu: zgmlTensorOps.relu,
  gelu: zgmlTensorOps.gelu,
  silu: zgmlTensorOps.silu,
  sigmoid: zgmlTensorOps.sigmoid,
  tanh: zgmlTensorOps.tanh,
  softmax: zgmlTensorOps.softmax,
  softmax_dim: zgmlTensorOps.softmax_dim,
  softmaxDim: zgmlTensorOps.softmaxDim,
  logSoftmax: zgmlTensorOps.logSoftmax,
  log_softmax: zgmlTensorOps.log_softmax,
  log_softmax_dim: zgmlTensorOps.log_softmax_dim,
  logSoftmaxDim: zgmlTensorOps.logSoftmaxDim,
  to: zgmlTensorOps.to,
  cpu: zgmlTensorOps.cpu,
  float: zgmlTensorOps.float,
  float32: zgmlTensorOps.float32,
  typeAs: zgmlTensorOps.typeAs,
  type_as: zgmlTensorOps.type_as,
  sum: zgmlTensorOps.sum,
  prod: zgmlTensorOps.prod,
  cumsum: zgmlTensorOps.cumsum,
  mean: zgmlTensorOps.mean,
  max: zgmlTensorOps.max,
  min: zgmlTensorOps.min,
  argmax: zgmlTensorOps.argmax,
  argmin: zgmlTensorOps.argmin,
  any: zgmlTensorOps.any,
  all: zgmlTensorOps.all,
  logsumexp: zgmlTensorOps.logsumexp,
  logSumExp: zgmlTensorOps.logSumExp,
  variance: zgmlTensorOps.variance,
  var: zgmlTensorOps.var,
  std: zgmlTensorOps.std,
  norm: zgmlTensorOps.norm,
  neg: zgmlTensorOps.neg,
  negative: zgmlTensorOps.negative,
  expm1: zgmlTensorOps.expm1,
  log1p: zgmlTensorOps.log1p,
  sqr: zgmlTensorOps.sqr,
  square: zgmlTensorOps.square,
  recip: zgmlTensorOps.recip,
  reciprocal: zgmlTensorOps.reciprocal,
  sgn: zgmlTensorOps.sgn,
  sign: zgmlTensorOps.sign,
  step: zgmlTensorOps.step,
  isnan: zgmlTensorOps.isnan,
  isinf: zgmlTensorOps.isinf,
  isfinite: zgmlTensorOps.isfinite,
  floor: zgmlTensorOps.floor,
  ceil: zgmlTensorOps.ceil,
  round: zgmlTensorOps.round,
  trunc: zgmlTensorOps.trunc,
  sin: zgmlTensorOps.sin,
  cos: zgmlTensorOps.cos,
  tan: zgmlTensorOps.tan,
  sqrt: zgmlTensorOps.sqrt,
  rsqrt: zgmlTensorOps.rsqrt,
  exp: zgmlTensorOps.exp,
  log: zgmlTensorOps.log,
  abs: zgmlTensorOps.abs,
  pow: zgmlTensorOps.pow,
  clamp: zgmlTensorOps.clamp,
  clip: zgmlTensorOps.clip,
  flatten: zgmlTensorOps.flatten,
  reshape: zgmlTensorOps.reshape,
  view: zgmlTensorOps.view,
  squeeze: zgmlTensorOps.squeeze,
  unsqueeze: zgmlTensorOps.unsqueeze,
  transpose: zgmlTensorOps.transpose,
  permute: zgmlTensorOps.permute,
  flip: zgmlTensorOps.flip,
  roll: zgmlTensorOps.roll,
  select: zgmlTensorOps.select,
  narrow: zgmlTensorOps.narrow,
  slice: zgmlTensorOps.slice,
  indexSelect: zgmlTensorOps.indexSelect,
  index_select: zgmlTensorOps.index_select,
  gather: zgmlTensorOps.gather,
  take: zgmlTensorOps.take,
  unbind: zgmlTensorOps.unbind,
  maximum: zgmlTensorOps.maximum,
  minimum: zgmlTensorOps.minimum,
  where: zgmlTensorOps.where,
  maskedFill: zgmlTensorOps.maskedFill,
  masked_fill: zgmlTensorOps.masked_fill,
  allclose: zgmlTensorOps.allclose,
  equal: zgmlTensorOps.equal,
  argsort: zgmlTensorOps.argsort,
  sort: zgmlTensorOps.sort,
  topk: zgmlTensorOps.topk,
  split: zgmlTensorOps.split,
  chunk: zgmlTensorOps.chunk,
  hasShape,
  requireShape,
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
  gradMode,
  nn,
  F,
  functional: F,
  compile,
  lazy: sharedFrontend.lazy,
  optim,
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
  loss,
  train,
  checkpoint,
  save: zgmlCheckpointIo.save,
  load: zgmlCheckpointIo.load,
  Program: BrowserProgram,
  Session: BrowserSession,
  NativeBuffer: BrowserNativeBuffer,
});
export const torch = zgml;
export const browserFrontendRuntimeManifest = Object.freeze({
  kind: "zgml-browser-frontend-runtime",
  source: "ts",
  productSourceOfTruth: "ts-only",
  runtimePath: "Tensor -> nn/loss/optim/train -> honest compile support",
  nativeLoader: false,
  nativeProgramAvailable: false,
});
