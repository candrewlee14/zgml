"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";
import {
  shapeScalarCount,
} from "./core/shape.js";
import {
  freezeSequentialTrace,
  traceCompilerArtifacts,
} from "./runtime/trace_compiler.js";
import type { CompileDiagnostic } from "./runtime/compile_diagnostics.js";
import type {
  LazyReductionShape,
  NarrowShape,
  ModuleTargetForwardShape,
  NnModule,
  PermuteShape,
  SelectShape,
  SliceShape,
  SqueezeShape,
  TensorShape,
  TensorShapeOf,
  ModuleKernelPlan,
  ModuleProgramTrace,
  ModuleReductionKind,
  ModuleTensorProgramIr,
  TensorShapeTuple,
  TransposeShape,
  UnsqueezeShape,
} from "./public_api.js";

type AnyRecord = Record<string, any>;
type LazyOpKind =
  | "linear"
  | "embedding"
  | "conv2d"
  | "maxPool2d"
  | "avgPool2d"
  | "activation"
  | "dropout"
  | "softmax"
  | "logSoftmax"
  | ModuleReductionKind
  | "layerNorm"
  | "rmsNorm"
  | "batchNorm1d"
  | "reshape"
  | "view"
  | "flatten"
  | "squeeze"
  | "unsqueeze"
  | "broadcastTo"
  | "expand"
  | "narrow"
  | "select"
  | "slice"
  | "transpose"
  | "permute";
type ActivationKind = "relu" | "gelu" | "silu" | "sigmoid" | "tanh" | "step" | "exp" | "log" | "neg" | "recip" | "abs" | "sqrt" | "square" | "sgn";
type LazyTensorRole = "input" | "op-output";
type LazyTensorSource = Readonly<{
  role: LazyTensorRole;
  name: string | null;
  shape: readonly number[];
}>;
type LazyGraphOp = Readonly<{
  index: number;
  path: string;
  op: LazyOpKind;
  inputShape: readonly number[];
  outputShape: readonly number[];
  parameters: readonly LazyParameterTrace[];
  attrs: Readonly<AnyRecord>;
}>;
type LazyParameterTrace = Readonly<{
  name: string;
  shape: readonly number[];
  layout: string | null;
  scalarCount: number;
}>;

export type LazyLinearOptions = Readonly<{
  name?: string;
  bias?: boolean;
  weightLayout?: string | null;
  biasLayout?: string | null;
}>;
export type LazyFlattenOptions = Readonly<{
  startDim?: number;
  endDim?: number;
}>;
export type LazyDropoutOptions = Readonly<{
  training?: boolean;
}>;
export type LazyNormOptions = Readonly<{
  eps?: number;
  affine?: boolean;
  bias?: boolean;
}>;
export type LazyConv2dOptions = Readonly<{
  name?: string;
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  groups?: number;
  bias?: boolean;
}>;
export type LazyPool2dOptions = Readonly<{
  kernelSize?: number | readonly [number, number];
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  ceilMode?: boolean;
  countIncludePad?: boolean;
}>;
export type LazyModuleTraceOptions<Shape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  inputShape: Shape;
  name?: string;
}>;
export type LazyCompileSupport = Readonly<{
  supported: boolean;
  reason: string | null;
  nativePath: "device-program";
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  diagnostic: CompileDiagnostic | null;
}>;
export type LazyModuleTarget = NnModule | readonly NnModule[];
export type LazyCompilerArtifacts = Readonly<{
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  diagnostic: CompileDiagnostic | null;
}>;

export class LazyTensor<Shape extends TensorShapeTuple = TensorShapeTuple> {
  readonly shape: Shape;
  readonly name: string | null;
  readonly source: LazyTensorSource;
  readonly ops: readonly LazyGraphOp[];

  constructor(shape: Shape, options: Readonly<{ name?: string | null; source?: LazyTensorSource; ops?: readonly LazyGraphOp[] }> = {}) {
    this.shape = Object.freeze(shape.slice()) as Shape;
    this.name = options.name ?? null;
    this.source = options.source ?? Object.freeze({
      role: "input",
      name: this.name,
      shape: this.shape,
    });
    this.ops = Object.freeze((options.ops ?? []).slice());
    Object.freeze(this);
  }

  linear<OutFeatures extends number>(outFeatures: OutFeatures, options: LazyLinearOptions = {}): LazyTensor<LazyLinearShape<Shape, OutFeatures>> {
    return linear(this, outFeatures, options);
  }

  embedding<EmbeddingDim extends number>(numEmbeddings: number, embeddingDim: EmbeddingDim, options: Readonly<{ name?: string }> = {}): LazyTensor<LazyEmbeddingShape<Shape, EmbeddingDim>> {
    return embedding(this, numEmbeddings, embeddingDim, options);
  }

  layerNorm(features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
    return layerNorm(this, features, options);
  }

  layer_norm(features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
    return layerNorm(this, features, options);
  }

  rmsNorm(features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
    return rmsNorm(this, features, options);
  }

  rms_norm(features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
    return rmsNorm(this, features, options);
  }

  conv2d(outChannels: number, kernelSize: number | readonly [number, number], options: LazyConv2dOptions = {}): LazyTensor<TensorShapeTuple> {
    return conv2d(this, outChannels, kernelSize, options);
  }

  maxPool2d(kernelSize: number | readonly [number, number] = 2, options: LazyPool2dOptions = {}): LazyTensor<TensorShapeTuple> {
    return maxPool2d(this, kernelSize, options);
  }

  max_pool2d(kernelSize: number | readonly [number, number] = 2, options: LazyPool2dOptions = {}): LazyTensor<TensorShapeTuple> {
    return maxPool2d(this, kernelSize, options);
  }

  avgPool2d(kernelSize: number | readonly [number, number] = 2, options: LazyPool2dOptions = {}): LazyTensor<TensorShapeTuple> {
    return avgPool2d(this, kernelSize, options);
  }

  avg_pool2d(kernelSize: number | readonly [number, number] = 2, options: LazyPool2dOptions = {}): LazyTensor<TensorShapeTuple> {
    return avgPool2d(this, kernelSize, options);
  }

  relu(): LazyTensor<Shape> {
    return activation(this, "relu");
  }

  gelu(): LazyTensor<Shape> {
    return activation(this, "gelu");
  }

  silu(): LazyTensor<Shape> {
    return activation(this, "silu");
  }

  sigmoid(): LazyTensor<Shape> {
    return activation(this, "sigmoid");
  }

  tanh(): LazyTensor<Shape> {
    return activation(this, "tanh");
  }

  exp(): LazyTensor<Shape> {
    return activation(this, "exp");
  }

  log(): LazyTensor<Shape> {
    return activation(this, "log");
  }

  neg(): LazyTensor<Shape> {
    return activation(this, "neg");
  }

  recip(): LazyTensor<Shape> {
    return activation(this, "recip");
  }

  abs(): LazyTensor<Shape> {
    return activation(this, "abs");
  }

  sqrt(): LazyTensor<Shape> {
    return activation(this, "sqrt");
  }

  square(): LazyTensor<Shape> {
    return activation(this, "square");
  }

  sgn(): LazyTensor<Shape> {
    return activation(this, "sgn");
  }

  step(): LazyTensor<Shape> {
    return activation(this, "step");
  }

  dropout(p = 0.5, options: LazyDropoutOptions = {}): LazyTensor<Shape> {
    return dropout(this, p, options);
  }

  softmax(dim: number): LazyTensor<Shape> {
    return softmax(this, dim);
  }

  logSoftmax(dim: number): LazyTensor<Shape> {
    return logSoftmax(this, dim);
  }

  log_softmax(dim: number): LazyTensor<Shape> {
    return logSoftmax(this, dim);
  }

  reshape<const TargetShape extends TensorShapeTuple>(shape: TargetShape): LazyTensor<TargetShape> {
    return reshape(this, shape);
  }

  view<const TargetShape extends TensorShapeTuple>(shape: TargetShape): LazyTensor<TargetShape> {
    return view(this, shape);
  }

  flatten(options: LazyFlattenOptions = {}): LazyTensor<TensorShapeTuple> {
    return flatten(this, options);
  }

  squeeze(): LazyTensor<SqueezeShape<Shape>>;
  squeeze<const Dim extends number>(dim: Dim): LazyTensor<SqueezeShape<Shape, Dim>>;
  squeeze(dim?: number | null): LazyTensor<TensorShapeTuple> {
    return dim === null || dim === undefined ? squeeze(this) : squeeze(this, dim);
  }

  unsqueeze<const Dim extends number>(dim: Dim): LazyTensor<UnsqueezeShape<Shape, Dim>> {
    return unsqueeze(this, dim);
  }

  broadcastTo<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
    return broadcastTo(this, shape);
  }

  broadcast_to<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
    return broadcastTo(this, shape);
  }

  expand<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
    return expand(this, shape);
  }

  narrow<const Dim extends number, const Length extends number>(dim: Dim, start: number, length: Length): LazyTensor<NarrowShape<Shape, Dim, Length>>;
  narrow(dim: number, start: number, length: number): LazyTensor<TensorShapeTuple>;
  narrow(dim: number, start: number, length: number): LazyTensor<TensorShapeTuple> {
    return narrow(this, dim, start, length);
  }

  select<const Dim extends number>(dim: Dim, index: number): LazyTensor<SelectShape<Shape, Dim>>;
  select(dim: number, index: number): LazyTensor<TensorShapeTuple>;
  select(dim: number, index: number): LazyTensor<TensorShapeTuple> {
    return select(this, dim, index);
  }

  slice<const Dim extends number, const Start extends number, const End extends number>(dim: Dim, start: Start, end: End, step?: 1): LazyTensor<SliceShape<Shape, Dim, Start, End>>;
  slice(dim: number, start?: number | null, end?: number | null, step?: number): LazyTensor<TensorShapeTuple>;
  slice(dim: number, start: number | null = null, end: number | null = null, step = 1): LazyTensor<TensorShapeTuple> {
    return slice(this, dim, start, end, step);
  }

  transpose(): LazyTensor<TransposeShape<Shape>>;
  transpose<const Dim0 extends number, const Dim1 extends number>(dim0: Dim0, dim1: Dim1): LazyTensor<TransposeShape<Shape, Dim0, Dim1>>;
  transpose(dim0 = 0, dim1 = 1): LazyTensor<TensorShapeTuple> {
    return transpose(this, dim0, dim1);
  }

  permute<const Dims extends readonly number[]>(dims: Dims): LazyTensor<PermuteShape<Shape, Dims>>;
  permute(dims: readonly number[]): LazyTensor<TensorShapeTuple>;
  permute(dims: readonly number[]): LazyTensor<TensorShapeTuple> {
    return permute(this, dims);
  }

  sum<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
    return reduction(this, "sum", dim);
  }

  mean<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
    return reduction(this, "mean", dim);
  }

  max<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
    return reduction(this, "max", dim);
  }

  min<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
    return reduction(this, "min", dim);
  }

  apply<const Target extends LazyModuleTarget>(module: Target): LazyTensor<ModuleTargetForwardShape<Target, Shape>> {
    return apply(module, this);
  }

  trace(): ModuleProgramTrace {
    return trace(this);
  }

  tensorProgramIr(): ModuleTensorProgramIr | null {
    return tensorProgramIr(this);
  }

  tensor_program_ir(): ModuleTensorProgramIr | null {
    return tensorProgramIr(this);
  }

  kernelPlan(): ModuleKernelPlan | null {
    return kernelPlan(this);
  }

  kernel_plan(): ModuleKernelPlan | null {
    return kernelPlan(this);
  }

  artifacts(): LazyCompilerArtifacts {
    return artifacts(this);
  }

  compileSupport(): LazyCompileSupport {
    return compileSupport(this);
  }

  compile_support(): LazyCompileSupport {
    return compileSupport(this);
  }

  canCompile(): boolean {
    return canCompile(this);
  }

  can_compile(): boolean {
    return canCompile(this);
  }

  requireCompileSupport(): LazyCompileSupport {
    return requireCompileSupport(this);
  }

  require_compile_support(): LazyCompileSupport {
    return requireCompileSupport(this);
  }
}

type LazyLinearShape<InputShape extends TensorShapeTuple, OutFeatures extends number> =
  InputShape extends readonly [number]
    ? readonly [OutFeatures]
    : InputShape extends readonly [infer Batch extends number, number]
      ? readonly [Batch, OutFeatures]
      : TensorShapeTuple;
type LazyEmbeddingShape<InputShape extends TensorShapeTuple, EmbeddingDim extends number> =
  readonly [...InputShape, EmbeddingDim];

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value as number;
}

function normalizeShape(shape: readonly unknown[], label: string): number[] {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`${label} must be a non-empty shape array`);
  }
  return shape.map((dim, index) => positiveInteger(dim, `${label}[${index}]`));
}

function normalizeShapeArg(shape: TensorShape, label: string): number[] {
  return typeof shape === "number"
    ? [positiveInteger(shape, label)]
    : normalizeShape(shape, label);
}

function normalizePair(value: unknown, label: string, defaultValue: readonly [number, number]): readonly [number, number] {
  if (value === undefined || value === null) return defaultValue;
  if (Number.isSafeInteger(value) && (value as number) >= 0) return [value as number, value as number] as const;
  if (Array.isArray(value) && value.length === 2 && value.every((dim) => Number.isSafeInteger(dim) && dim >= 0)) {
    return [value[0] as number, value[1] as number] as const;
  }
  throw new Error(`${label} must be a non-negative integer or [height, width] pair`);
}

function normalizePositivePair(value: unknown, label: string, defaultValue: readonly [number, number]): readonly [number, number] {
  const result = normalizePair(value, label, defaultValue);
  if (result[0] <= 0 || result[1] <= 0) throw new Error(`${label} dimensions must be positive`);
  return result;
}

function normalizeDim(dim: number, rank: number, label: string): number {
  if (!Number.isSafeInteger(dim)) throw new Error(`${label} must be an integer`);
  const axis = dim < 0 ? rank + dim : dim;
  if (axis < 0 || axis >= rank) throw new Error(`${label} ${dim} is out of bounds for rank ${rank}`);
  return axis;
}

function normalizeInsertDim(dim: number, rank: number, label: string): number {
  if (!Number.isSafeInteger(dim)) throw new Error(`${label} must be an integer`);
  const axis = dim < 0 ? rank + dim + 1 : dim;
  if (axis < 0 || axis > rank) throw new Error(`${label} ${dim} is out of bounds for rank ${rank + 1}`);
  return axis;
}

function normalizeIndex(index: number, dim: number, label: string): number {
  if (!Number.isSafeInteger(index)) throw new Error(`${label} must be an integer`);
  const normalized = index < 0 ? dim + index : index;
  if (normalized < 0 || normalized >= dim) throw new Error(`${label} ${index} is out of bounds for dimension length ${dim}`);
  return normalized;
}

function normalizeSliceBound(bound: number | null | undefined, dim: number, fallback: number, label: string): number {
  if (bound === null || bound === undefined) return fallback;
  if (!Number.isSafeInteger(bound)) throw new Error(`${label} must be an integer`);
  const normalized = bound < 0 ? dim + bound : bound;
  if (normalized < 0 || normalized > dim) throw new Error(`${label} ${bound} is out of bounds for dimension length ${dim}`);
  return normalized;
}

function normalizePermutation(dims: readonly number[], rank: number, label: string): number[] {
  if (!Array.isArray(dims) || dims.length !== rank) throw new Error(`${label} length must match rank ${rank}`);
  const seen = new Set<number>();
  return dims.map((dim, index) => {
    const axis = normalizeDim(dim, rank, `${label}[${index}]`);
    if (seen.has(axis)) throw new Error(`${label} must be a permutation of axes; duplicate axis ${axis}`);
    seen.add(axis);
    return axis;
  });
}

function assertBroadcastable(inputShape: readonly number[], targetShape: readonly number[], label: string) {
  for (let i = 0; i < targetShape.length; i += 1) {
    const sourceIndex = inputShape.length - targetShape.length + i;
    const sourceDim = sourceIndex < 0 ? 1 : inputShape[sourceIndex];
    const targetDim = targetShape[i];
    if (sourceDim !== 1 && sourceDim !== targetDim) {
      throw new Error(`${label} cannot broadcast [${inputShape.join(",")}] to [${targetShape.join(",")}]`);
    }
  }
}

function nextPath(tensor: LazyTensor) {
  return String(tensor.ops.length);
}

function lazyParameter(name: string, shape: readonly number[], layout: string | null = "row-major"): LazyParameterTrace {
  const normalizedShape = normalizeShape(shape, `${name} shape`);
  return Object.freeze({
    name,
    shape: Object.freeze(normalizedShape),
    layout,
    scalarCount: shapeScalarCount(normalizedShape),
  });
}

function appendOp<const S extends TensorShapeTuple>(inputTensor: LazyTensor, op: Omit<LazyGraphOp, "index" | "path" | "inputShape">, outputShape: S): LazyTensor<S> {
  const index = inputTensor.ops.length;
  const graphOp = Object.freeze({
    ...op,
    index,
    path: nextPath(inputTensor),
    inputShape: Object.freeze(inputTensor.shape.slice()),
    outputShape: Object.freeze(outputShape.slice()),
    parameters: Object.freeze((op.parameters ?? []).slice()),
    attrs: Object.freeze({ ...(op.attrs ?? {}) }),
  });
  return new LazyTensor(outputShape, {
    source: Object.freeze({ role: "op-output", name: null, shape: graphOp.outputShape }),
    ops: Object.freeze([...inputTensor.ops, graphOp]),
  });
}

function traceOp(op: LazyGraphOp) {
  const base: AnyRecord = {
    index: op.index,
    path: op.path,
    op: op.op,
    parameters: op.parameters,
    inputShape: op.inputShape,
    outputShape: op.outputShape,
    ...op.attrs,
  };
  return base;
}

function moduleRecord(module: unknown, label: string): AnyRecord {
  if (!module || typeof module !== "object") throw new Error(`${label} must be an nn module`);
  return module as AnyRecord;
}

function moduleLayers(target: LazyModuleTarget): AnyRecord[] {
  if (Array.isArray(target)) return target.map((layer, index) => moduleRecord(layer, `lazy module layer ${index}`));
  const module = moduleRecord(target, "lazy module");
  if (Array.isArray(module.layers)) return module.layers.map((layer: unknown, index: number) => moduleRecord(layer, `lazy module layer ${index}`));
  return [module];
}

function moduleKind(module: AnyRecord) {
  return typeof module.kind === "string" ? module.kind : "";
}

function moduleName(fallback: number, module: AnyRecord) {
  if (typeof module.name === "string" && module.name.length > 0) return module.name;
  return String(fallback);
}

function scalarCountFromParameter(param: AnyRecord, fallbackShape: readonly number[]) {
  if (param.data && typeof param.data.length === "number") return param.data.length;
  if (param.tensor && Array.isArray(param.tensor.shape)) return shapeScalarCount(param.tensor.shape);
  return shapeScalarCount(fallbackShape);
}

function parameterShape(param: AnyRecord, fallbackShape: readonly number[]) {
  if (param.tensor && Array.isArray(param.tensor.shape)) return param.tensor.shape.slice();
  if (Array.isArray(param.shape)) return param.shape.slice();
  return fallbackShape.slice();
}

function moduleParameter(module: AnyRecord, paramName: string, fallbackShape: readonly number[], fallbackLayout: string | null) {
  const getter = module && typeof module.parameters === "function" ? module.parameters.bind(module) : null;
  const params = getter ? getter("") : [];
  const found = Array.isArray(params) ? params.find((param: AnyRecord) => param && (param.name === paramName || String(param.name).endsWith(`.${paramName}`))) : null;
  if (!found) return lazyParameter(paramName, fallbackShape, fallbackLayout);
  const shape = parameterShape(found, fallbackShape);
  const layout = typeof found.layout === "string" ? found.layout : fallbackLayout;
  return Object.freeze({
    name: paramName,
    shape: Object.freeze(normalizeShape(shape, `${paramName} shape`)),
    layout,
    scalarCount: scalarCountFromParameter(found, shape),
  });
}

function moduleLinear<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord, name: string) {
  const outFeatures = positiveInteger(module.outFeatures, "lazy nn.Linear outFeatures");
  const inputFeatures = positiveInteger(module.inFeatures, "lazy nn.Linear inFeatures");
  if (tensor.shape[tensor.shape.length - 1] !== inputFeatures) {
    throw new Error(`lazy nn.Linear expected input last dimension ${inputFeatures}, got ${tensor.shape[tensor.shape.length - 1]}`);
  }
  const bias = module.bias !== null && module.bias !== false && module.bias !== undefined;
  return linear(tensor, outFeatures, {
    name,
    bias,
    weightLayout: typeof module.weightParam?.layout === "string" ? module.weightParam.layout : "row-major:linear.weight[in_features,out_features]",
    biasLayout: typeof module.biasParam?.layout === "string" ? module.biasParam.layout : "row-major:linear.bias[out_features]",
  });
}

function moduleEmbedding<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord, name: string) {
  const numEmbeddings = positiveInteger(module.numEmbeddings, "lazy nn.Embedding numEmbeddings");
  const embeddingDim = positiveInteger(module.embeddingDim, "lazy nn.Embedding embeddingDim");
  const weightLayout = typeof module.weightParam?.layout === "string"
    ? module.weightParam.layout
    : "row-major:embedding.weight[num_embeddings,embedding_dim]";
  return embedding(tensor, numEmbeddings, embeddingDim, { name, weightLayout, parameter: moduleParameter(module, "weight", [numEmbeddings, embeddingDim], weightLayout) });
}

function moduleConv2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord) {
  const inChannels = positiveInteger(module.inChannels, "lazy nn.Conv2d inChannels");
  const outChannels = positiveInteger(module.outChannels, "lazy nn.Conv2d outChannels");
  const rank = tensor.shape.length;
  const batched = rank === 4;
  if (!batched && rank !== 3) {
    throw new Error(`lazy nn.Conv2d expected input shape [channels,height,width] or [batch,channels,height,width], got [${tensor.shape.join(",")}]`);
  }
  const channels = batched ? tensor.shape[1] : tensor.shape[0];
  const height = batched ? tensor.shape[2] : tensor.shape[1];
  const width = batched ? tensor.shape[3] : tensor.shape[2];
  if (channels !== inChannels) throw new Error(`lazy nn.Conv2d expected input channels ${inChannels}, got ${channels}`);
  const kernelSize = normalizePositivePair(module.kernelSize, "lazy nn.Conv2d kernelSize", [1, 1]);
  const stride = normalizePositivePair(module.stride, "lazy nn.Conv2d stride", [1, 1]);
  const padding = normalizePair(module.padding, "lazy nn.Conv2d padding", [0, 0]);
  const dilation = normalizePositivePair(module.dilation, "lazy nn.Conv2d dilation", [1, 1]);
  const groups = positiveInteger(module.groups ?? 1, "lazy nn.Conv2d groups");
  const outH = Math.floor((height + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] + 1);
  const outW = Math.floor((width + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`lazy nn.Conv2d output spatial shape must be positive, got [${outH},${outW}]`);
  }
  const outputShape = batched ? [tensor.shape[0], outChannels, outH, outW] : [outChannels, outH, outW];
  const parameters = [
    moduleParameter(
      module,
      "weight",
      [outChannels, inChannels, kernelSize[0], kernelSize[1]],
      typeof module.weightParam?.layout === "string"
        ? module.weightParam.layout
        : "row-major:conv2d.weight[out_channels,in_channels,kernel_h,kernel_w]",
    ),
  ];
  if (module.bias !== null && module.bias !== false && module.bias !== undefined) {
    parameters.push(moduleParameter(
      module,
      "bias",
      [outChannels],
      typeof module.biasParam?.layout === "string" ? module.biasParam.layout : "row-major:conv2d.bias[out_channels]",
    ));
  }
  return appendOp(tensor, {
    op: "conv2d",
    outputShape,
    parameters,
    attrs: {
      inChannels,
      outChannels,
      kernelSize,
      stride,
      padding,
      dilation,
      groups,
    },
  }, outputShape);
}

function appendPool2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  kind: "maxPool2d" | "avgPool2d",
  kernelSizeValue: unknown,
  options: LazyPool2dOptions,
): LazyTensor<TensorShapeTuple> {
  const rank = tensor.shape.length;
  const batched = rank === 4;
  if (!batched && rank !== 3) {
    throw new Error(`lazy ${kind} expected input shape [channels,height,width] or [batch,channels,height,width], got [${tensor.shape.join(",")}]`);
  }
  const channels = batched ? tensor.shape[1] : tensor.shape[0];
  const height = batched ? tensor.shape[2] : tensor.shape[1];
  const width = batched ? tensor.shape[3] : tensor.shape[2];
  const kernelSize = normalizePositivePair(kernelSizeValue, `lazy ${kind} kernelSize`, [1, 1]);
  const stride = normalizePositivePair(options.stride, `lazy ${kind} stride`, kernelSize);
  const padding = normalizePair(options.padding, `lazy ${kind} padding`, [0, 0]);
  const ceilMode = Boolean(options.ceilMode);
  const round = ceilMode ? Math.ceil : Math.floor;
  const dilation = kind === "maxPool2d"
    ? normalizePositivePair(options.dilation, "lazy maxPool2d dilation", [1, 1])
    : [1, 1] as const;
  const outH = kind === "maxPool2d"
    ? round((height + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] + 1)
    : round((height + 2 * padding[0] - kernelSize[0]) / stride[0] + 1);
  const outW = kind === "maxPool2d"
    ? round((width + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] + 1)
    : round((width + 2 * padding[1] - kernelSize[1]) / stride[1] + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`lazy ${kind} output spatial shape must be positive, got [${outH},${outW}]`);
  }
  const outputShape = batched ? [tensor.shape[0], channels, outH, outW] : [channels, outH, outW];
  const attrs: AnyRecord = {
    kernelSize,
    stride,
    padding,
    ceilMode,
  };
  if (kind === "maxPool2d") attrs.dilation = dilation;
  if (kind === "avgPool2d") attrs.countIncludePad = options.countIncludePad !== false;
  return appendOp(tensor, {
    op: kind,
    outputShape,
    parameters: [],
    attrs,
  }, outputShape);
}

function modulePool2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord, kind: "maxPool2d" | "avgPool2d") {
  return appendPool2d(tensor, kind, module.kernelSize, {
    stride: module.stride,
    padding: module.padding,
    dilation: module.dilation,
    ceilMode: Boolean(module.ceilMode),
    countIncludePad: module.countIncludePad,
  });
}

function moduleDropout<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord) {
  const p = module.p === undefined || module.p === null ? 0.5 : Number(module.p);
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error(`lazy nn.Dropout probability must be between 0 and 1, got ${module.p}`);
  return appendOp(tensor, {
    op: "dropout",
    outputShape: tensor.shape,
    parameters: [],
    attrs: {
      p,
      training: Boolean(module.training),
    },
  }, tensor.shape);
}

function moduleFeatureNorm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, module: AnyRecord) {
  const kind = moduleKind(module);
  const features = positiveInteger(module.features, `lazy nn.${kind} features`);
  const eps = Number.isFinite(module.eps) ? Number(module.eps) : 1e-5;
  const parameters: LazyParameterTrace[] = [];
  const hasWeight = module.weight !== null && module.weight !== false && module.weight !== undefined;
  const hasBias = module.bias !== null && module.bias !== false && module.bias !== undefined;
  if (hasWeight) {
    const weightLayout = typeof module.weightParam?.layout === "string"
      ? module.weightParam.layout
      : `row-major:${kind}.weight[features]`;
    parameters.push(moduleParameter(module, "weight", [features], weightLayout));
  }
  if (kind !== "rmsNorm" && hasBias) {
    const biasLayout = typeof module.biasParam?.layout === "string"
      ? module.biasParam.layout
      : `row-major:${kind}.bias[features]`;
    parameters.push(moduleParameter(module, "bias", [features], biasLayout));
  }
  return appendOp(tensor, {
    op: kind as "layerNorm" | "rmsNorm" | "batchNorm1d",
    outputShape: tensor.shape,
    parameters,
    attrs: {
      features,
      eps,
      momentum: Number.isFinite(module.momentum) ? Number(module.momentum) : 0.1,
      trackRunningStats: Boolean(module.trackRunningStats),
      training: Boolean(module.training),
      affine: hasWeight,
      bias: hasBias,
    },
  }, tensor.shape);
}

export function input<const Shape extends TensorShapeTuple>(shape: Shape, name = "input"): LazyTensor<Shape> {
  const normalizedShape = normalizeShape(shape, "lazy input shape") as unknown as Shape;
  return new LazyTensor(normalizedShape, { name });
}

export function linear<const Shape extends TensorShapeTuple, const OutFeatures extends number>(
  tensor: LazyTensor<Shape>,
  outFeatures: OutFeatures,
  options: LazyLinearOptions = {},
): LazyTensor<LazyLinearShape<Shape, OutFeatures>> {
  const inputShape = tensor.shape;
  if (inputShape.length !== 1 && inputShape.length !== 2) {
    throw new Error(`lazy linear expects rank-1 or rank-2 input, got rank ${inputShape.length}`);
  }
  const inFeatures = positiveInteger(inputShape[inputShape.length - 1], "lazy linear inFeatures");
  const outputFeatures = positiveInteger(outFeatures, "lazy linear outFeatures") as OutFeatures;
  const outputShape = (inputShape.length === 1 ? [outputFeatures] : [inputShape[0], outputFeatures]) as unknown as LazyLinearShape<Shape, OutFeatures>;
  const prefix = options.name ?? nextPath(tensor);
  const parameters = [
    lazyParameter(`${prefix}.weight`, [inFeatures, outputFeatures], options.weightLayout ?? "row-major"),
  ];
  if (options.bias !== false) {
    parameters.push(lazyParameter(`${prefix}.bias`, [outputFeatures], options.biasLayout ?? "row-major"));
  }
  return appendOp(tensor, {
    op: "linear",
    outputShape,
    parameters,
    attrs: { inFeatures, outFeatures: outputFeatures, bias: options.bias !== false },
  }, outputShape);
}

export function embedding<const Shape extends TensorShapeTuple, const EmbeddingDim extends number>(
  tensor: LazyTensor<Shape>,
  numEmbeddings: number,
  embeddingDim: EmbeddingDim,
  options: Readonly<{ name?: string; weightLayout?: string | null; parameter?: LazyParameterTrace }> = {},
): LazyTensor<LazyEmbeddingShape<Shape, EmbeddingDim>> {
  const normalizedNumEmbeddings = positiveInteger(numEmbeddings, "lazy embedding numEmbeddings");
  const normalizedEmbeddingDim = positiveInteger(embeddingDim, "lazy embedding embeddingDim") as EmbeddingDim;
  const outputShape = [...tensor.shape, normalizedEmbeddingDim] as unknown as LazyEmbeddingShape<Shape, EmbeddingDim>;
  const prefix = options.name ?? nextPath(tensor);
  return appendOp(tensor, {
    op: "embedding",
    outputShape,
    parameters: [
      options.parameter ?? lazyParameter(`${prefix}.weight`, [normalizedNumEmbeddings, normalizedEmbeddingDim], options.weightLayout ?? "row-major:embedding.weight[num_embeddings,embedding_dim]"),
    ],
    attrs: { numEmbeddings: normalizedNumEmbeddings, embeddingDim: normalizedEmbeddingDim },
  }, outputShape);
}

function normOp<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kind: "layerNorm" | "rmsNorm", features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
  const normalizedFeatures = positiveInteger(features, `lazy ${kind} features`);
  const eps = Number.isFinite(options.eps) ? Number(options.eps) : 1e-5;
  const hasWeight = options.affine !== false;
  const hasBias = kind === "layerNorm" && options.bias !== false && hasWeight;
  const parameters: LazyParameterTrace[] = [];
  if (hasWeight) parameters.push(lazyParameter(`${nextPath(tensor)}.weight`, [normalizedFeatures], `row-major:${kind}.weight[features]`));
  if (hasBias) parameters.push(lazyParameter(`${nextPath(tensor)}.bias`, [normalizedFeatures], `row-major:${kind}.bias[features]`));
  return appendOp(tensor, {
    op: kind,
    outputShape: tensor.shape,
    parameters,
    attrs: {
      features: normalizedFeatures,
      eps,
      momentum: 0.1,
      trackRunningStats: false,
      training: false,
      affine: hasWeight,
      bias: hasBias,
    },
  }, tensor.shape);
}

export function layerNorm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
  return normOp(tensor, "layerNorm", features, options);
}

export function layer_norm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
  return layerNorm(tensor, features, options);
}

export function rmsNorm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
  return normOp(tensor, "rmsNorm", features, options);
}

export function rms_norm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options: LazyNormOptions = {}): LazyTensor<Shape> {
  return rmsNorm(tensor, features, options);
}

export function conv2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  outChannels: number,
  kernelSizeValue: number | readonly [number, number],
  options: LazyConv2dOptions = {},
): LazyTensor<TensorShapeTuple> {
  const rank = tensor.shape.length;
  const batched = rank === 4;
  if (!batched && rank !== 3) {
    throw new Error(`lazy conv2d expected input shape [channels,height,width] or [batch,channels,height,width], got [${tensor.shape.join(",")}]`);
  }
  const inChannels = positiveInteger(batched ? tensor.shape[1] : tensor.shape[0], "lazy conv2d inChannels");
  const normalizedOutChannels = positiveInteger(outChannels, "lazy conv2d outChannels");
  const height = batched ? tensor.shape[2] : tensor.shape[1];
  const width = batched ? tensor.shape[3] : tensor.shape[2];
  const kernelSize = normalizePositivePair(kernelSizeValue, "lazy conv2d kernelSize", [1, 1]);
  const stride = normalizePositivePair(options.stride, "lazy conv2d stride", [1, 1]);
  const padding = normalizePair(options.padding, "lazy conv2d padding", [0, 0]);
  const dilation = normalizePositivePair(options.dilation, "lazy conv2d dilation", [1, 1]);
  const groups = positiveInteger(options.groups ?? 1, "lazy conv2d groups");
  const outH = Math.floor((height + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] + 1);
  const outW = Math.floor((width + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`lazy conv2d output spatial shape must be positive, got [${outH},${outW}]`);
  }
  const outputShape = batched ? [tensor.shape[0], normalizedOutChannels, outH, outW] : [normalizedOutChannels, outH, outW];
  const prefix = options.name ?? nextPath(tensor);
  const parameters = [
    lazyParameter(`${prefix}.weight`, [normalizedOutChannels, inChannels, kernelSize[0], kernelSize[1]], "row-major:conv2d.weight[out_channels,in_channels,kernel_h,kernel_w]"),
  ];
  if (options.bias !== false) {
    parameters.push(lazyParameter(`${prefix}.bias`, [normalizedOutChannels], "row-major:conv2d.bias[out_channels]"));
  }
  return appendOp(tensor, {
    op: "conv2d",
    outputShape,
    parameters,
    attrs: {
      inChannels,
      outChannels: normalizedOutChannels,
      kernelSize,
      stride,
      padding,
      dilation,
      groups,
    },
  }, outputShape);
}

export function maxPool2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  kernelSize: number | readonly [number, number] = 2,
  options: LazyPool2dOptions = {},
): LazyTensor<TensorShapeTuple> {
  return appendPool2d(tensor, "maxPool2d", kernelSize, options);
}

export function max_pool2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  kernelSize: number | readonly [number, number] = 2,
  options: LazyPool2dOptions = {},
): LazyTensor<TensorShapeTuple> {
  return maxPool2d(tensor, kernelSize, options);
}

export function avgPool2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  kernelSize: number | readonly [number, number] = 2,
  options: LazyPool2dOptions = {},
): LazyTensor<TensorShapeTuple> {
  return appendPool2d(tensor, "avgPool2d", kernelSize, options);
}

export function avg_pool2d<const Shape extends TensorShapeTuple>(
  tensor: LazyTensor<Shape>,
  kernelSize: number | readonly [number, number] = 2,
  options: LazyPool2dOptions = {},
): LazyTensor<TensorShapeTuple> {
  return avgPool2d(tensor, kernelSize, options);
}

export function activation<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kind: ActivationKind): LazyTensor<Shape> {
  return appendOp(tensor, {
    op: "activation",
    outputShape: tensor.shape,
    parameters: [],
    attrs: { activation: kind },
  }, tensor.shape);
}

export function relu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "relu");
}

export function gelu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "gelu");
}

export function silu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "silu");
}

export function sigmoid<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "sigmoid");
}

export function tanh<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "tanh");
}

export function exp<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "exp");
}

export function log<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "log");
}

export function neg<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "neg");
}

export function recip<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "recip");
}

export function abs<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "abs");
}

export function sqrt<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "sqrt");
}

export function square<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "square");
}

export function sgn<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "sgn");
}

export function step<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape> {
  return activation(tensor, "step");
}

export function dropout<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, p = 0.5, options: LazyDropoutOptions = {}): LazyTensor<Shape> {
  const probability = Number(p);
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error(`lazy dropout probability must be between 0 and 1, got ${p}`);
  }
  return appendOp(tensor, {
    op: "dropout",
    outputShape: tensor.shape,
    parameters: [],
    attrs: { p: probability, training: Boolean(options.training) },
  }, tensor.shape);
}

export function softmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape> {
  return appendOp(tensor, {
    op: "softmax",
    outputShape: tensor.shape,
    parameters: [],
    attrs: { dim: normalizeDim(dim, tensor.shape.length, "lazy softmax dim") },
  }, tensor.shape);
}

export function logSoftmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape> {
  return appendOp(tensor, {
    op: "logSoftmax",
    outputShape: tensor.shape,
    parameters: [],
    attrs: { dim: normalizeDim(dim, tensor.shape.length, "lazy logSoftmax dim") },
  }, tensor.shape);
}

export function log_softmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape> {
  return logSoftmax(tensor, dim);
}

export function reshape<const Shape extends TensorShapeTuple, const TargetShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TargetShape> {
  const targetShape = normalizeShape(shape, "lazy reshape target shape") as unknown as TargetShape;
  if (shapeScalarCount(tensor.shape) !== shapeScalarCount(targetShape)) {
    throw new Error(`lazy reshape cannot change scalar count from ${shapeScalarCount(tensor.shape)} to ${shapeScalarCount(targetShape)}`);
  }
  return appendOp(tensor, {
    op: "reshape",
    outputShape: targetShape,
    parameters: [],
    attrs: { shape: targetShape },
  }, targetShape);
}

export function view<const Shape extends TensorShapeTuple, const TargetShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TargetShape> {
  const viewed = reshape(tensor, shape);
  const lastOp = viewed.ops[viewed.ops.length - 1];
  const replacement: LazyGraphOp = Object.freeze({
    index: lastOp.index,
    path: lastOp.path,
    op: "view",
    inputShape: lastOp.inputShape,
    outputShape: lastOp.outputShape,
    parameters: lastOp.parameters,
    attrs: Object.freeze({ shape: viewed.shape }),
  });
  return new LazyTensor(viewed.shape, { source: viewed.source, ops: Object.freeze([...viewed.ops.slice(0, -1), replacement]) });
}

export function flatten<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, options: LazyFlattenOptions = {}): LazyTensor<TensorShapeTuple> {
  const rank = tensor.shape.length;
  const startDim = normalizeDim(options.startDim ?? 0, rank, "lazy flatten startDim");
  const endDim = normalizeDim(options.endDim ?? -1, rank, "lazy flatten endDim");
  if (startDim > endDim) throw new Error("lazy flatten startDim must be <= endDim");
  const outputShape = [
    ...tensor.shape.slice(0, startDim),
    shapeScalarCount(tensor.shape.slice(startDim, endDim + 1)),
    ...tensor.shape.slice(endDim + 1),
  ];
  return appendOp(tensor, {
    op: "flatten",
    outputShape,
    parameters: [],
    attrs: { startDim, endDim },
  }, outputShape);
}

export function squeeze<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<SqueezeShape<Shape>>;
export function squeeze<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<SqueezeShape<Shape, Dim>>;
export function squeeze<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim?: number | null): LazyTensor<TensorShapeTuple> {
  let outputShape: number[];
  let attrs: AnyRecord;
  if (dim === null || dim === undefined) {
    outputShape = tensor.shape.filter((size) => size !== 1);
    if (outputShape.length === 0) outputShape.push(1);
    attrs = { squeezeAll: true };
  } else {
    const axis = normalizeDim(dim, tensor.shape.length, "lazy squeeze dim");
    outputShape = tensor.shape[axis] === 1
      ? tensor.shape.filter((_, index) => index !== axis)
      : tensor.shape.slice();
    if (outputShape.length === 0) outputShape.push(1);
    attrs = { squeezeAll: false, dim: axis };
  }
  return appendOp(tensor, {
    op: "squeeze",
    outputShape,
    parameters: [],
    attrs,
  }, outputShape);
}

export function unsqueeze<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<UnsqueezeShape<Shape, Dim>> {
  const axis = normalizeInsertDim(dim, tensor.shape.length, "lazy unsqueeze dim");
  const outputShape = tensor.shape.slice();
  outputShape.splice(axis, 0, 1);
  return appendOp(tensor, {
    op: "unsqueeze",
    outputShape,
    parameters: [],
    attrs: { dim: axis },
  }, outputShape) as unknown as LazyTensor<UnsqueezeShape<Shape, Dim>>;
}

export function broadcastTo<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
  const outputShape = normalizeShapeArg(shape, "lazy broadcastTo shape");
  assertBroadcastable(tensor.shape, outputShape, "lazy broadcastTo");
  return appendOp(tensor, {
    op: "broadcastTo",
    outputShape,
    parameters: [],
    attrs: { shape: outputShape },
  }, outputShape as unknown as TensorShapeOf<TargetShape>);
}

export function broadcast_to<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
  return broadcastTo(tensor, shape);
}

export function expand<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>> {
  const outputShape = normalizeShapeArg(shape, "lazy expand shape");
  assertBroadcastable(tensor.shape, outputShape, "lazy expand");
  return appendOp(tensor, {
    op: "expand",
    outputShape,
    parameters: [],
    attrs: { shape: outputShape },
  }, outputShape as unknown as TensorShapeOf<TargetShape>);
}

export function narrow<const Shape extends TensorShapeTuple, const Dim extends number, const Length extends number>(tensor: LazyTensor<Shape>, dim: Dim, start: number, length: Length): LazyTensor<NarrowShape<Shape, Dim, Length>>;
export function narrow<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start: number, length: number): LazyTensor<TensorShapeTuple>;
export function narrow<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start: number, length: number): LazyTensor<TensorShapeTuple> {
  const axis = normalizeDim(dim, tensor.shape.length, "lazy narrow dim");
  const normalizedStart = normalizeIndex(start, tensor.shape[axis], "lazy narrow start");
  const normalizedLength = positiveInteger(length, "lazy narrow length");
  if (normalizedStart + normalizedLength > tensor.shape[axis]) throw new Error("lazy narrow range is out of bounds");
  const outputShape = tensor.shape.slice();
  outputShape[axis] = normalizedLength;
  return appendOp(tensor, {
    op: "narrow",
    outputShape,
    parameters: [],
    attrs: { dim: axis, start: normalizedStart, length: normalizedLength },
  }, outputShape);
}

export function select<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim, index: number): LazyTensor<SelectShape<Shape, Dim>>;
export function select<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, index: number): LazyTensor<TensorShapeTuple>;
export function select<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, index: number): LazyTensor<TensorShapeTuple> {
  const axis = normalizeDim(dim, tensor.shape.length, "lazy select dim");
  const normalizedIndex = normalizeIndex(index, tensor.shape[axis], "lazy select index");
  const outputShape = tensor.shape.filter((_, shapeAxis) => shapeAxis !== axis);
  if (outputShape.length === 0) outputShape.push(1);
  return appendOp(tensor, {
    op: "select",
    outputShape,
    parameters: [],
    attrs: { dim: axis, selectIndex: normalizedIndex },
  }, outputShape);
}

export function slice<const Shape extends TensorShapeTuple, const Dim extends number, const Start extends number, const End extends number>(tensor: LazyTensor<Shape>, dim: Dim, start: Start, end: End, step?: 1): LazyTensor<SliceShape<Shape, Dim, Start, End>>;
export function slice<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start?: number | null, end?: number | null, step?: number): LazyTensor<TensorShapeTuple>;
export function slice<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start: number | null = null, end: number | null = null, step = 1): LazyTensor<TensorShapeTuple> {
  const axis = normalizeDim(dim, tensor.shape.length, "lazy slice dim");
  const normalizedStep = positiveInteger(step, "lazy slice step");
  const normalizedStart = normalizeSliceBound(start, tensor.shape[axis], 0, "lazy slice start");
  const normalizedEnd = normalizeSliceBound(end, tensor.shape[axis], tensor.shape[axis], "lazy slice end");
  const length = Math.ceil((normalizedEnd - normalizedStart) / normalizedStep);
  if (length <= 0) throw new Error("lazy slice would produce an empty tensor");
  const outputShape = tensor.shape.slice();
  outputShape[axis] = length;
  return appendOp(tensor, {
    op: "slice",
    outputShape,
    parameters: [],
    attrs: { dim: axis, start: normalizedStart, end: normalizedEnd, step: normalizedStep },
  }, outputShape);
}

export function transpose<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<TransposeShape<Shape>>;
export function transpose<const Shape extends TensorShapeTuple, const Dim0 extends number, const Dim1 extends number>(tensor: LazyTensor<Shape>, dim0: Dim0, dim1: Dim1): LazyTensor<TransposeShape<Shape, Dim0, Dim1>>;
export function transpose<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim0 = 0, dim1 = 1): LazyTensor<TensorShapeTuple> {
  if (tensor.shape.length < 2) throw new Error("lazy transpose requires rank >= 2");
  const axis0 = normalizeDim(dim0, tensor.shape.length, "lazy transpose dim0");
  const axis1 = normalizeDim(dim1, tensor.shape.length, "lazy transpose dim1");
  const outputShape = tensor.shape.slice();
  const tmp = outputShape[axis0];
  outputShape[axis0] = outputShape[axis1];
  outputShape[axis1] = tmp;
  return appendOp(tensor, {
    op: "transpose",
    outputShape,
    parameters: [],
    attrs: { dim0: axis0, dim1: axis1 },
  }, outputShape);
}

export function permute<const Shape extends TensorShapeTuple, const Dims extends readonly number[]>(tensor: LazyTensor<Shape>, dims: Dims): LazyTensor<PermuteShape<Shape, Dims>>;
export function permute<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dims: readonly number[]): LazyTensor<TensorShapeTuple>;
export function permute<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dims: readonly number[]): LazyTensor<TensorShapeTuple> {
  const axes = normalizePermutation(dims, tensor.shape.length, "lazy permute dims");
  const outputShape = axes.map((axis) => tensor.shape[axis]);
  return appendOp(tensor, {
    op: "permute",
    outputShape,
    parameters: [],
    attrs: { dims: axes },
  }, outputShape);
}

export function reduction<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, kind: ModuleReductionKind, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
  const axis = normalizeDim(dim, tensor.shape.length, `lazy ${kind} dim`);
  const outputShape = tensor.shape.filter((_, index) => index !== axis);
  if (outputShape.length === 0) outputShape.push(1);
  return appendOp(tensor, {
    op: kind,
    outputShape,
    parameters: [],
    attrs: { dim: axis },
  }, outputShape) as unknown as LazyTensor<LazyReductionShape<Shape, Dim>>;
}

export function sum<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
  return reduction(tensor, "sum", dim);
}

export function mean<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
  return reduction(tensor, "mean", dim);
}

export function max<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
  return reduction(tensor, "max", dim);
}

export function min<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>> {
  return reduction(tensor, "min", dim);
}

function applyActivationModule(tensor: LazyTensor, module: AnyRecord) {
  const kind = moduleKind(module);
  if (kind === "sigmoid" || kind === "tanh" || kind === "step" || kind === "exp" || kind === "log" || kind === "neg" || kind === "recip" || kind === "abs" || kind === "sqrt" || kind === "square" || kind === "sgn" || kind === "relu" || kind === "gelu" || kind === "silu") {
    return activation(tensor, kind);
  }
  return null;
}

function applyShapeModule(tensor: LazyTensor, module: AnyRecord) {
  switch (moduleKind(module)) {
    case "identity":
      return tensor;
    case "reshape":
      return reshape(tensor, normalizeShape(module.shape ?? [], "lazy nn.reshape shape"));
    case "view":
      return view(tensor, normalizeShape(module.shape ?? [], "lazy nn.view shape"));
    case "flatten":
      return flatten(tensor, {
        startDim: Number.isSafeInteger(module.startDim) ? module.startDim : 0,
        endDim: Number.isSafeInteger(module.endDim) ? module.endDim : -1,
      });
    case "squeeze":
      return module.squeezeAll ? squeeze(tensor) : squeeze(tensor, Number(module.dim ?? 0));
    case "unsqueeze":
      return unsqueeze(tensor, Number(module.dim ?? 0));
    case "broadcastTo":
      return broadcastTo(tensor, normalizeShape(module.shape ?? [], "lazy nn.broadcastTo shape"));
    case "expand":
      return expand(tensor, normalizeShape(module.shape ?? [], "lazy nn.expand shape"));
    case "narrow":
      return narrow(tensor, Number(module.dim ?? 0), Number(module.start ?? 0), Number(module.length ?? 1));
    case "select":
      return select(tensor, Number(module.dim ?? 0), Number(module.index ?? 0));
    case "slice":
      return slice(tensor, Number(module.dim ?? 0), module.start ?? null, module.end ?? null, Number(module.step ?? 1));
    case "transpose":
      return transpose(tensor, Number(module.dim0 ?? 0), Number(module.dim1 ?? 1));
    case "permute":
      return permute(tensor, Array.isArray(module.dims) ? module.dims : []);
    default:
      return null;
  }
}

export function apply<const Target extends LazyModuleTarget, const Shape extends TensorShapeTuple>(target: Target, tensor: LazyTensor<Shape>): LazyTensor<ModuleTargetForwardShape<Target, Shape>> {
  let out: LazyTensor = tensor;
  const layers = moduleLayers(target);
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const kind = moduleKind(layer);
    if (kind === "linear") {
      out = moduleLinear(out, layer, moduleName(index, layer));
    } else if (kind === "embedding") {
      out = moduleEmbedding(out, layer, moduleName(index, layer));
    } else if (kind === "conv2d") {
      out = moduleConv2d(out, layer);
    } else if (kind === "maxPool2d" || kind === "avgPool2d") {
      out = modulePool2d(out, layer, kind);
    } else if (kind === "dropout") {
      out = moduleDropout(out, layer);
    } else if (kind === "layerNorm" || kind === "rmsNorm" || kind === "batchNorm1d") {
      out = moduleFeatureNorm(out, layer);
    } else if (kind === "softmax") {
      out = softmax(out, Number(layer.dim ?? -1));
    } else if (kind === "logSoftmax") {
      out = logSoftmax(out, Number(layer.dim ?? -1));
    } else if (kind === "sum" || kind === "mean" || kind === "max" || kind === "min" || kind === "argmax" || kind === "argmin") {
      out = reduction(out, kind, Number(layer.dim ?? -1));
    } else {
      const activationResult = applyActivationModule(out, layer);
      if (activationResult) {
        out = activationResult;
        continue;
      }
      const shapeResult = applyShapeModule(out, layer);
      if (shapeResult) {
        out = shapeResult;
        continue;
      }
      throw new Error(`lazy.apply does not support nn module kind: ${kind || "unknown"}`);
    }
  }
  return out as unknown as LazyTensor<ModuleTargetForwardShape<Target, Shape>>;
}

export function fromModule<const Target extends LazyModuleTarget, const Shape extends TensorShapeTuple>(target: Target, options: LazyModuleTraceOptions<Shape>): LazyTensor<ModuleTargetForwardShape<Target, Shape>> {
  return apply(target, input(options.inputShape, options.name ?? "input"));
}

export function traceModule<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): ModuleProgramTrace {
  return trace(fromModule(target, options));
}

export function moduleArtifacts<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): LazyCompilerArtifacts {
  return artifacts(fromModule(target, options));
}

export function moduleCompileSupport<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): LazyCompileSupport {
  return compileSupport(fromModule(target, options));
}

export function trace(tensor: LazyTensor): ModuleProgramTrace {
  let parameterCount = 0;
  let parameterScalarCount = 0;
  for (const op of tensor.ops) {
    parameterCount += op.parameters.length;
    for (const param of op.parameters) parameterScalarCount += param.scalarCount;
  }
  return freezeSequentialTrace({
    kind: "sequential",
    normalized: true,
    layerCount: tensor.ops.length,
    opCount: tensor.ops.length,
    parameterCount,
    parameterScalarCount,
    shapeKnown: true,
    inputShape: tensor.ops[0]?.inputShape ?? tensor.shape,
    outputShape: tensor.shape,
    ops: tensor.ops.map(traceOp),
  }) as ModuleProgramTrace;
}

export function artifacts(tensor: LazyTensor): LazyCompilerArtifacts {
  const traced = trace(tensor);
  const compilerArtifacts = traceCompilerArtifacts(traced);
  return Object.freeze({
    trace: traced,
    ir: compilerArtifacts.ir,
    kernelPlan: compilerArtifacts.kernelPlan,
    diagnostic: compilerArtifacts.diagnostic,
  });
}

export function tensorProgramIr(tensor: LazyTensor): ModuleTensorProgramIr | null {
  return artifacts(tensor).ir;
}

export function kernelPlan(tensor: LazyTensor): ModuleKernelPlan | null {
  return artifacts(tensor).kernelPlan;
}

export function compileSupport(tensor: LazyTensor): LazyCompileSupport {
  const result = artifacts(tensor);
  return Object.freeze({
    supported: Boolean(result.ir && result.kernelPlan && !result.diagnostic),
    reason: result.diagnostic ? result.diagnostic.message : null,
    nativePath: "device-program",
    trace: result.trace,
    ir: result.ir,
    kernelPlan: result.kernelPlan,
    diagnostic: result.diagnostic,
  });
}

export function canCompile(tensor: LazyTensor): boolean {
  return compileSupport(tensor).supported;
}

export function can_compile(tensor: LazyTensor): boolean {
  return canCompile(tensor);
}

export function requireCompileSupport(tensor: LazyTensor): LazyCompileSupport {
  const support = compileSupport(tensor);
  if (support.supported) return support;
  throw new Error(`lazy graph cannot compile to native Program: ${support.reason ?? "unsupported graph"}`);
}

export function require_compile_support(tensor: LazyTensor): LazyCompileSupport {
  return requireCompileSupport(tensor);
}

export const lazyManifest = Object.freeze({
  kind: "zgml-lazy-tensor-frontend",
  ...tsProductManifestPolicy("src/ts/lazy.ts"),
  runtimePath: "LazyTensor -> Trace -> TensorProgramIr -> KernelPlan -> Program",
});
