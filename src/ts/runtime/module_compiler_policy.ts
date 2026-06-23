"use strict";

import {
  shapeScalarCount,
  normalizeFactoryShape,
  normalizeViewShape,
  normalizeDim,
  normalizeInsertDim,
  broadcastPlan,
  traceIndexForDim,
  traceSliceBound,
} from "../core/shape.js";
import { compileSupportRejectionReason } from "./compile_support.js";
import {
  compileDiagnostic,
} from "./compile_diagnostics.js";
import { packedSequentialProgramParameters } from "./module_bindings.js";
import {
  freezeSequentialTrace,
  traceCompilerArtifacts,
  compiledSequentialModuleSpecFromTrace,
  supportDetailsWithTrace,
  sequentialUnsupportedSupportDetails,
  shapeMismatchAnalysis,
} from "./trace_compiler.js";
import type {
  CompiledSequentialModuleSpec,
  TraceCompilerArtifacts,
  TraceCompilerSupportDetails,
} from "./trace_compiler.js";
import type { CompileDiagnostic } from "./compile_diagnostics.js";
import type {
  CompileOptions,
  ModuleProgramTrace,
  ModuleTraceOptions,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type CompilerOptionsExtra = Readonly<CompileOptions & {
  skipLayerSupportFallback?: boolean;
}>;

export type SequentialCompilerConstructors = Readonly<{
  LinearModule: Function;
  ActivationModule: Function;
  SequentialModule?: Function;
  EmbeddingModule?: Function;
  Conv2dModule?: Function;
  AvgPool2dModule?: Function;
  MaxPool2dModule?: Function;
  SoftmaxModule?: Function;
  LogSoftmaxModule?: Function;
  ReductionModule?: Function;
  DropoutModule?: Function;
  FeatureNormModule?: Function;
  ShapeModule?: Function;
}>;

export type SequentialTraceEntry = Readonly<{
  layer: AnyRecord;
  path: readonly number[];
}>;

export type SequentialTinyLinearLayerSpec = Readonly<{
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly weight?: ArrayLike<number> | null;
  readonly bias?: ArrayLike<number> | null;
  readonly bindParameters?: () => {
    readonly weights?: unknown;
    readonly bias?: unknown;
  };
}>;

export type SequentialTinyLinearCompiledProgramSpec = Readonly<{
  readonly kind: "tiny-linear";
  readonly layer: SequentialTinyLinearLayerSpec;
  readonly nativePath: "tiny-linear";
  readonly modelKind: "tiny-linear";
  readonly layerCount: 1;
  readonly trace?: ModuleProgramTrace | null;
  readonly ir?: TraceCompilerArtifacts["ir"];
  readonly kernelPlan?: TraceCompilerArtifacts["kernelPlan"];
}>;

export type SequentialCompiledProgramSpec =
  | SequentialTinyLinearCompiledProgramSpec
  | CompiledSequentialModuleSpec;

export type SequentialDiagnosticSupportEvidence = Readonly<{
  readonly composable: true;
  readonly nativePath: "device-program";
  readonly modelKind: "module";
  readonly layerCount: number;
  readonly diagnostics: readonly CompileDiagnostic[];
}>;

export type SequentialSupportEvidence =
  | TraceCompilerSupportDetails
  | SequentialDiagnosticSupportEvidence;

export type SequentialProgramAnalysis = Readonly<{
  supported: boolean;
  reason: string | null;
  compiled: SequentialCompiledProgramSpec | null;
  normalizedLayers: readonly unknown[];
  trace: ModuleProgramTrace | null;
  support: SequentialSupportEvidence;
}>;

export type TraceModuleCompiler = Readonly<{
  trace(layers: readonly AnyRecord[], traceOptions?: ModuleTraceOptions): ModuleProgramTrace;
  analyze(layers: readonly AnyRecord[], compileOptions?: CompilerOptionsExtra): SequentialProgramAnalysis;
  analyzeSingle(layer: AnyRecord, compileOptions?: CompilerOptionsExtra): SequentialProgramAnalysis;
  packParameters: typeof packedSequentialProgramParameters;
}>;

export type SequentialTraceOptions = Readonly<{
  inputShape?: readonly number[] | null;
}>;

export function requireSequentialCompilerOptions(options: AnyRecord): SequentialCompilerConstructors {
  const LinearModule = options && options.LinearModule;
  const ActivationModule = options && options.ActivationModule;
  if (typeof LinearModule !== "function" || typeof ActivationModule !== "function") {
    throw new Error("Sequential compiler helpers require LinearModule and ActivationModule constructors");
  }
  const SequentialModule = options && options.SequentialModule;
  const EmbeddingModule = options && options.EmbeddingModule;
  const Conv2dModule = options && options.Conv2dModule;
  const AvgPool2dModule = options && options.AvgPool2dModule;
  const MaxPool2dModule = options && options.MaxPool2dModule;
  const SoftmaxModule = options && options.SoftmaxModule;
  const LogSoftmaxModule = options && options.LogSoftmaxModule;
  const ReductionModule = options && options.ReductionModule;
  const DropoutModule = options && options.DropoutModule;
  const FeatureNormModule = options && options.FeatureNormModule;
  const ShapeModule = options && options.ShapeModule;
  return { LinearModule, ActivationModule, SequentialModule, EmbeddingModule, Conv2dModule, AvgPool2dModule, MaxPool2dModule, SoftmaxModule, LogSoftmaxModule, ReductionModule, DropoutModule, FeatureNormModule, ShapeModule };
}

function sequentialCompilerOptions(options: AnyRecord): Partial<SequentialCompilerConstructors> {
  return {
    LinearModule: options && typeof options.LinearModule === "function" ? options.LinearModule : undefined,
    ActivationModule: options && typeof options.ActivationModule === "function" ? options.ActivationModule : undefined,
    SequentialModule: options && typeof options.SequentialModule === "function" ? options.SequentialModule : undefined,
    EmbeddingModule: options && typeof options.EmbeddingModule === "function" ? options.EmbeddingModule : undefined,
    Conv2dModule: options && typeof options.Conv2dModule === "function" ? options.Conv2dModule : undefined,
    AvgPool2dModule: options && typeof options.AvgPool2dModule === "function" ? options.AvgPool2dModule : undefined,
    MaxPool2dModule: options && typeof options.MaxPool2dModule === "function" ? options.MaxPool2dModule : undefined,
    SoftmaxModule: options && typeof options.SoftmaxModule === "function" ? options.SoftmaxModule : undefined,
    LogSoftmaxModule: options && typeof options.LogSoftmaxModule === "function" ? options.LogSoftmaxModule : undefined,
    ReductionModule: options && typeof options.ReductionModule === "function" ? options.ReductionModule : undefined,
    DropoutModule: options && typeof options.DropoutModule === "function" ? options.DropoutModule : undefined,
    FeatureNormModule: options && typeof options.FeatureNormModule === "function" ? options.FeatureNormModule : undefined,
    ShapeModule: options && typeof options.ShapeModule === "function" ? options.ShapeModule : undefined,
  };
}

function hasModuleShape(layer: AnyRecord, kind: string, fields: readonly string[] = []): boolean {
  return Boolean(layer && layer.kind === kind && fields.every((field) => field in layer));
}

function isModuleLike(layer: AnyRecord, ctor: Function | undefined, kind: string, fields: readonly string[] = []): boolean {
  return Boolean((typeof ctor === "function" && layer instanceof ctor) || hasModuleShape(layer, kind, fields));
}

function isSequentialLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return Boolean((typeof ctor === "function" && layer instanceof ctor) || (layer && Array.isArray(layer.layers) && typeof layer.forward === "function"));
}

function isLinearLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "linear", ["inFeatures", "outFeatures", "weight"]);
}

function isEmbeddingLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "embedding", ["numEmbeddings", "embeddingDim"]);
}

function isConv2dLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "conv2d", ["inChannels", "outChannels", "kernelSize", "weight"]);
}

function isMaxPool2dLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "maxPool2d", ["kernelSize", "stride"]);
}

function isAvgPool2dLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "avgPool2d", ["kernelSize", "stride"]);
}

function isActivationLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  if (typeof ctor === "function" && layer instanceof ctor) return true;
  return typeof layer?.kind === "string" && [
    "gelu",
    "relu",
    "silu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "neg",
    "recip",
    "abs",
    "sgn",
    "step",
    "sqrt",
    "square",
  ].includes(layer.kind);
}

function isShapeLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  if (typeof ctor === "function" && layer instanceof ctor) return true;
  return typeof layer?.kind === "string" && [
    "identity",
    "reshape",
    "view",
    "flatten",
    "squeeze",
    "unsqueeze",
    "broadcastTo",
    "expand",
    "narrow",
    "select",
    "slice",
    "transpose",
    "permute",
  ].includes(layer.kind);
}

function isSoftmaxLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "softmax", ["dim"]);
}

function isLogSoftmaxLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "logSoftmax", ["dim"]);
}

function isReductionLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  if (typeof ctor === "function" && layer instanceof ctor) return true;
  return typeof layer?.kind === "string" && ["sum", "mean", "prod", "max", "min", "argmax", "argmin"].includes(layer.kind) && "dim" in layer;
}

function isDropoutLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return isModuleLike(layer, ctor, "dropout", ["p"]);
}

function isFeatureNormLayer(layer: AnyRecord, ctor: Function | undefined): boolean {
  return typeof layer?.kind === "string" && ["layerNorm", "rmsNorm", "batchNorm1d"].includes(layer.kind) && "features" in layer;
}

export function flattenSequentialEntries(
  layers: readonly AnyRecord[],
  options: AnyRecord,
  seen?: Set<AnyRecord>,
  prefix?: readonly number[],
): SequentialTraceEntry[] {
  const { SequentialModule } = sequentialCompilerOptions(options);

  const stack = seen || new Set<AnyRecord>();
  const base = prefix || [];
  const entries: SequentialTraceEntry[] = [];
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const path = [...base, index];
    if (isSequentialLayer(layer, SequentialModule)) {
      if (stack.has(layer)) {
        throw new Error("nested nn.Sequential graph contains a cycle");
      }
      stack.add(layer);
      entries.push(...flattenSequentialEntries(layer.layers, options, stack, path));
      stack.delete(layer);
    } else {
      entries.push({ layer, path });
    }
  }
  return entries;
}

export function flattenSequentialLayers(layers: readonly AnyRecord[], options: AnyRecord, seen?: Set<AnyRecord>): AnyRecord[] {
  return flattenSequentialEntries(layers, options, seen).map((entry) => entry.layer);
}

export function compiledSequentialLinearSpec(layers: readonly AnyRecord[], options: AnyRecord): SequentialCompiledProgramSpec | null {
  const { LinearModule } = sequentialCompilerOptions(options);
  if (layers.length !== 1) return null;
  const layer = layers[0];
  if (!isLinearLayer(layer, LinearModule)) return null;
  const linearLayer = layer as SequentialTinyLinearLayerSpec;
  return {
    kind: "tiny-linear",
    layer: linearLayer,
    nativePath: "tiny-linear",
    modelKind: "tiny-linear",
    layerCount: 1,
  };
}

export function compiledSequentialProgramSpec(layers: readonly AnyRecord[], options: AnyRecord): SequentialCompiledProgramSpec | null {
  const normalizedLayers = flattenSequentialLayers(layers, options);
  return compiledSequentialProgramSpecForNormalizedLayers(normalizedLayers, options);
}

export function compiledSequentialProgramSpecForNormalizedLayers(layers: readonly AnyRecord[], options: AnyRecord): SequentialCompiledProgramSpec | null {
  const compatibility = compiledSequentialLinearSpec(layers, options);
  if (compatibility && !(options && options.inputShape)) return compatibility;

  const entries = layers.map((layer, index) => ({ layer, path: [index] }));
  return compiledSequentialModuleSpec(entries, options) ?? compatibility;
}

export function normalizeTraceShape(shape: unknown, name: string): number[] | null {
  if (shape == null) return null;
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`${name} must be a non-empty shape array`);
  }
  return shape.map((dim) => {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`${name} dimensions must be positive integers`);
    }
    return dim;
  });
}

export function defaultTraceInputShape(entries: readonly SequentialTraceEntry[], options: AnyRecord): number[] | null {
  if (options && options.inputShape) {
    return normalizeTraceShape(options.inputShape, "compile inputShape");
  }
  const { LinearModule } = sequentialCompilerOptions(options);
  for (const entry of entries) {
    if (isLinearLayer(entry.layer, LinearModule)) return [entry.layer.inFeatures];
    break;
  }
  return null;
}

export function sequentialProgramSupportDetails(spec: AnyRecord): AnyRecord {
  const details: AnyRecord = {
    nativePath: spec.nativePath,
    modelKind: spec.modelKind,
    layerCount: spec.layerCount,
  };
  if (spec.desc) {
    details.inputLen = spec.desc.inputLen;
    details.outputLen = spec.desc.outputLen;
    details.weightsLen = spec.desc.weightsLen;
    details.biasLen = spec.desc.biasLen;
  } else if (spec.kind === "tiny-linear" && spec.layer) {
    details.inputLen = spec.layer.inFeatures;
    details.outputLen = spec.layer.outFeatures;
    details.weightsLen = spec.layer.weight ? spec.layer.weight.length : 0;
    details.biasLen = spec.layer.bias ? spec.layer.bias.length : 0;
  }
  return details;
}

export function traceCompilerUnsupportedReason(artifacts: AnyRecord | null | undefined, fallback: string): string {
  return compileSupportRejectionReason(artifacts, { fallbackReason: fallback });
}

export function parameterTrace(param: AnyRecord): AnyRecord {
  return {
    name: param.name,
    shape: param.tensor && Array.isArray(param.tensor.shape) ? param.tensor.shape.slice() : [],
    layout: param.layout ?? null,
    scalarCount: param.data ? param.data.length : 0,
  };
}

export function layerParametersTrace(layer: AnyRecord, path: readonly number[]): AnyRecord[] {
  if (typeof layer.parameters !== "function") return [];
  return layer.parameters(path.join(".")).map(parameterTrace);
}

export function copyShape(shape: readonly number[] | null | undefined): number[] | null {
  return shape ? shape.slice() : null;
}

export function withTraceShapes<T extends AnyRecord>(
  op: T,
  inputShape: readonly number[] | null | undefined,
  outputShape: readonly number[] | null | undefined,
): T & { inputShape?: number[]; outputShape?: number[] | null } {
  if (!inputShape) return op;
  return {
    ...op,
    inputShape: inputShape.slice(),
    outputShape: outputShape ? outputShape.slice() : outputShape,
  };
}

export function singleLinearCompatibilityAllowed(
  normalizedLayers: readonly AnyRecord[],
  options: AnyRecord,
  inputShape: readonly number[] | null,
): boolean {
  if (!(options && options.inputShape)) return true;
  const { LinearModule } = sequentialCompilerOptions(options);
  if (normalizedLayers.length !== 1 || !isLinearLayer(normalizedLayers[0], LinearModule)) return false;
  return Boolean(inputShape && inputShape.length === 1 && inputShape[0] === normalizedLayers[0].inFeatures);
}

export function inferLinearShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  if (inputShape.length === 1 && inputShape[0] === layer.inFeatures) {
    return [layer.outFeatures];
  }
  if (inputShape.length === 2 && inputShape[1] === layer.inFeatures) {
    return [inputShape[0], layer.outFeatures];
  }
  if (inputShape.length === 3 && inputShape[2] === layer.inFeatures) {
    return [inputShape[0], inputShape[1], layer.outFeatures];
  }
  throw new Error(`linear trace input shape must end with ${layer.inFeatures}, got [${inputShape.join(",")}]`);
}

export function inferEmbeddingShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  return [...inputShape, layer.embeddingDim];
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

export function inferConv2dShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  const batched = inputShape.length === 4;
  if (!batched && inputShape.length !== 3) {
    throw new Error(`conv2d trace input shape must be [channels,height,width] or [batch,channels,height,width], got [${inputShape.join(",")}]`);
  }
  const channels = batched ? inputShape[1] : inputShape[0];
  const height = batched ? inputShape[2] : inputShape[1];
  const width = batched ? inputShape[3] : inputShape[2];
  if (channels !== layer.inChannels) {
    throw new Error(`conv2d trace input channels must be ${layer.inChannels}, got ${channels}`);
  }
  const [kh, kw] = normalizePositivePair(layer.kernelSize, "conv2d trace kernelSize", [1, 1]);
  const [sh, sw] = normalizePositivePair(layer.stride, "conv2d trace stride", [1, 1]);
  const [ph, pw] = normalizePair(layer.padding, "conv2d trace padding", [0, 0]);
  const [dh, dw] = normalizePositivePair(layer.dilation, "conv2d trace dilation", [1, 1]);
  const outH = Math.floor((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
  const outW = Math.floor((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`conv2d trace output spatial shape must be positive, got [${outH},${outW}]`);
  }
  return batched
    ? [inputShape[0], layer.outChannels, outH, outW]
    : [layer.outChannels, outH, outW];
}

export function inferMaxPool2dShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  const batched = inputShape.length === 4;
  if (!batched && inputShape.length !== 3) {
    throw new Error(`maxPool2d trace input shape must be [channels,height,width] or [batch,channels,height,width], got [${inputShape.join(",")}]`);
  }
  const channels = batched ? inputShape[1] : inputShape[0];
  const height = batched ? inputShape[2] : inputShape[1];
  const width = batched ? inputShape[3] : inputShape[2];
  const [kh, kw] = normalizePositivePair(layer.kernelSize, "maxPool2d trace kernelSize", [1, 1]);
  const [sh, sw] = normalizePositivePair(layer.stride, "maxPool2d trace stride", [kh, kw]);
  const [ph, pw] = normalizePair(layer.padding, "maxPool2d trace padding", [0, 0]);
  const [dh, dw] = normalizePositivePair(layer.dilation, "maxPool2d trace dilation", [1, 1]);
  const round = layer.ceilMode ? Math.ceil : Math.floor;
  const outH = round((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
  const outW = round((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`maxPool2d trace output spatial shape must be positive, got [${outH},${outW}]`);
  }
  return batched
    ? [inputShape[0], channels, outH, outW]
    : [channels, outH, outW];
}

export function inferAvgPool2dShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  const batched = inputShape.length === 4;
  if (!batched && inputShape.length !== 3) {
    throw new Error(`avgPool2d trace input shape must be [channels,height,width] or [batch,channels,height,width], got [${inputShape.join(",")}]`);
  }
  const channels = batched ? inputShape[1] : inputShape[0];
  const height = batched ? inputShape[2] : inputShape[1];
  const width = batched ? inputShape[3] : inputShape[2];
  const [kh, kw] = normalizePositivePair(layer.kernelSize, "avgPool2d trace kernelSize", [1, 1]);
  const [sh, sw] = normalizePositivePair(layer.stride, "avgPool2d trace stride", [kh, kw]);
  const [ph, pw] = normalizePair(layer.padding, "avgPool2d trace padding", [0, 0]);
  const round = layer.ceilMode ? Math.ceil : Math.floor;
  const outH = round((height + 2 * ph - kh) / sh + 1);
  const outW = round((width + 2 * pw - kw) / sw + 1);
  if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
    throw new Error(`avgPool2d trace output spatial shape must be positive, got [${outH},${outW}]`);
  }
  return batched
    ? [inputShape[0], channels, outH, outW]
    : [channels, outH, outW];
}

export function inferSoftmaxShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  const rank = inputShape.length;
  const axis = layer.dim < 0 ? rank + layer.dim : layer.dim;
  if (axis < 0 || axis >= rank) {
    throw new Error(`softmax trace dim ${layer.dim} is out of range for rank ${rank}`);
  }
  return inputShape.slice();
}

export function inferReductionShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  const rank = inputShape.length;
  const axis = layer.dim < 0 ? rank + layer.dim : layer.dim;
  if (axis < 0 || axis >= rank) {
    throw new Error(`${layer.kind} trace dim ${layer.dim} is out of range for rank ${rank}`);
  }
  const outputShape = inputShape.slice();
  outputShape[axis] = 1;
  return outputShape;
}

export function inferFeatureNormShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  if (!inputShape) return null;
  if (shapeScalarCount(inputShape) % layer.features !== 0) {
    throw new Error(`${layer.kind} trace input shape [${inputShape.join(",")}] must have scalar count divisible by features ${layer.features}`);
  }
  return inputShape.slice();
}

export function inferShapeModuleShape(layer: AnyRecord, inputShape: readonly number[] | null): number[] | null {
  switch (layer.kind) {
    case "identity":
      return inputShape ? inputShape.slice() : null;
    case "diagonal":
      if (!inputShape) return null;
      if (inputShape.length !== 2) throw new Error("diagonal trace requires rank-2 input");
      return [Math.min(inputShape[0], inputShape[1])];
    case "reshape":
    case "view":
      return inputShape ? normalizeViewShape(layer.shape, shapeScalarCount(inputShape), `${layer.kind} trace`) : null;
    case "flatten":
      if (!inputShape) return null;
      {
        const rank = inputShape.length;
        const start = normalizeDim(layer.startDim, rank, "flatten trace");
        const end = normalizeDim(layer.endDim, rank, "flatten trace");
        if (end < start) throw new Error(`flatten trace end dim ${layer.endDim} must be >= start dim ${layer.startDim}`);
        let flattened = 1;
        for (let i = start; i <= end; i += 1) flattened *= inputShape[i];
        return [
          ...inputShape.slice(0, start),
          flattened,
          ...inputShape.slice(end + 1),
        ];
      }
    case "squeeze": {
      if (!inputShape) return null;
      if (layer.squeezeAll) {
        const outputShape = inputShape.filter((dim) => dim !== 1);
        if (outputShape.length === 0) outputShape.push(1);
        return outputShape;
      }
      const axis = normalizeDim(layer.dim, inputShape.length, "squeeze trace");
      if (inputShape[axis] !== 1) return inputShape.slice();
      const outputShape = inputShape.filter((_, shapeAxis) => shapeAxis !== axis);
      if (outputShape.length === 0) outputShape.push(1);
      return outputShape;
    }
    case "unsqueeze": {
      if (!inputShape) return null;
      const axis = normalizeInsertDim(layer.dim, inputShape.length, "unsqueeze trace");
      const outputShape = inputShape.slice();
      outputShape.splice(axis, 0, 1);
      return outputShape;
    }
    case "broadcastTo":
    case "expand": {
      const target = normalizeFactoryShape(layer.shape, `${layer.kind} trace shape`).shape;
      if (!inputShape) return target;
      const plan = broadcastPlan(inputShape, target, `${layer.kind} trace`);
      if (plan.shape.length !== target.length || !plan.shape.every((dim, index) => dim === target[index])) {
        throw new Error(`${layer.kind} trace shape mismatch: [${inputShape.join(",")}] cannot broadcast to [${target.join(",")}]`);
      }
      return target;
    }
    case "repeat":
    case "tile": {
      if (!inputShape) return null;
      const repeats = normalizeFactoryShape(layer.shape, `${layer.kind} trace repeats`).shape;
      if (repeats.length < inputShape.length) {
        throw new Error(`${layer.kind} trace repeats length ${repeats.length} must be >= input rank ${inputShape.length}`);
      }
      const paddedShape = [
        ...Array.from({ length: repeats.length - inputShape.length }, () => 1),
        ...inputShape,
      ];
      return repeats.map((repeatDim, axis) => paddedShape[axis] * repeatDim);
    }
    case "narrow": {
      if (!inputShape) return null;
      const axis = normalizeDim(layer.dim, inputShape.length, "narrow trace");
      const start = traceIndexForDim(layer.start, inputShape[axis], "narrow trace");
      if (!Number.isSafeInteger(layer.length) || layer.length <= 0) {
        throw new Error(`narrow trace length must be a positive safe integer, got ${layer.length}`);
      }
      if (start + layer.length > inputShape[axis]) {
        throw new Error(`narrow trace range ${layer.start}..${layer.start + layer.length} is out of range for dimension length ${inputShape[axis]}`);
      }
      const outputShape = inputShape.slice();
      outputShape[axis] = layer.length;
      return outputShape;
    }
    case "select": {
      if (!inputShape) return null;
      const axis = normalizeDim(layer.dim, inputShape.length, "select trace");
      traceIndexForDim(layer.index, inputShape[axis], "select trace");
      const outputShape = inputShape.filter((_, shapeAxis) => shapeAxis !== axis);
      if (outputShape.length === 0) outputShape.push(1);
      return outputShape;
    }
    case "slice": {
      if (!inputShape) return null;
      const axis = normalizeDim(layer.dim, inputShape.length, "slice trace");
      const step = layer.step === undefined ? 1 : layer.step;
      if (!Number.isSafeInteger(step) || step <= 0) {
        throw new Error(`slice trace step must be a positive safe integer, got ${step}`);
      }
      const start = traceSliceBound(layer.start, inputShape[axis], 0, "slice trace start");
      const end = traceSliceBound(layer.end, inputShape[axis], inputShape[axis], "slice trace end");
      const length = Math.ceil((end - start) / step);
      if (length <= 0) throw new Error("slice trace would produce an empty tensor");
      const outputShape = inputShape.slice();
      outputShape[axis] = length;
      return outputShape;
    }
    case "transpose": {
      if (!inputShape) return null;
      if (inputShape.length < 2) throw new Error("transpose trace requires rank >= 2");
      const axis0 = normalizeDim(layer.dim0, inputShape.length, "transpose trace dim0");
      const axis1 = normalizeDim(layer.dim1, inputShape.length, "transpose trace dim1");
      const outputShape = inputShape.slice();
      const tmp = outputShape[axis0];
      outputShape[axis0] = outputShape[axis1];
      outputShape[axis1] = tmp;
      return outputShape;
    }
    case "permute": {
      if (!inputShape) return null;
      if (!Array.isArray(layer.dims) || layer.dims.length !== inputShape.length) {
        throw new Error(`permute trace dims length ${Array.isArray(layer.dims) ? layer.dims.length : "<non-array>"} must match input rank ${inputShape.length}`);
      }
      const seen = new Set<number>();
      const axes = layer.dims.map((dim: number) => {
        const axis = normalizeDim(dim, inputShape.length, "permute trace");
        if (seen.has(axis)) throw new Error(`permute trace dims must be a permutation of input axes; duplicate axis ${axis}`);
        seen.add(axis);
        return axis;
      });
      return axes.map((axis: number) => inputShape[axis]);
    }
    default:
      return inputShape ? inputShape.slice() : null;
  }
}

export function traceOpForEntry(
  entry: SequentialTraceEntry,
  index: number,
  options: AnyRecord,
  inputShape: readonly number[] | null,
): AnyRecord {
  const { LinearModule, ActivationModule, EmbeddingModule, Conv2dModule, AvgPool2dModule, MaxPool2dModule, SoftmaxModule, LogSoftmaxModule, ReductionModule, DropoutModule, FeatureNormModule, ShapeModule } = sequentialCompilerOptions(options);
  const layer = entry.layer;
  const path = entry.path.join(".");
  const parameters = layerParametersTrace(layer, entry.path);

  if (isLinearLayer(layer, LinearModule)) {
    const outputShape = inferLinearShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "linear",
      inFeatures: layer.inFeatures,
      outFeatures: layer.outFeatures,
      bias: Boolean(layer.bias),
      parameters,
    }, inputShape, outputShape);
  }
  if (isEmbeddingLayer(layer, EmbeddingModule)) {
    const outputShape = inferEmbeddingShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "embedding",
      numEmbeddings: layer.numEmbeddings,
      embeddingDim: layer.embeddingDim,
      parameters,
    }, inputShape, outputShape);
  }
  if (isConv2dLayer(layer, Conv2dModule)) {
    const outputShape = inferConv2dShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "conv2d",
      inChannels: layer.inChannels,
      outChannels: layer.outChannels,
      kernelSize: Array.isArray(layer.kernelSize) ? layer.kernelSize.slice() : layer.kernelSize,
      stride: Array.isArray(layer.stride) ? layer.stride.slice() : layer.stride,
      padding: Array.isArray(layer.padding) ? layer.padding.slice() : layer.padding,
      dilation: Array.isArray(layer.dilation) ? layer.dilation.slice() : layer.dilation,
      groups: layer.groups,
      parameters,
    }, inputShape, outputShape);
  }
  if (isMaxPool2dLayer(layer, MaxPool2dModule)) {
    const outputShape = inferMaxPool2dShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "maxPool2d",
      kernelSize: Array.isArray(layer.kernelSize) ? layer.kernelSize.slice() : layer.kernelSize,
      stride: Array.isArray(layer.stride) ? layer.stride.slice() : layer.stride,
      padding: Array.isArray(layer.padding) ? layer.padding.slice() : layer.padding,
      dilation: Array.isArray(layer.dilation) ? layer.dilation.slice() : layer.dilation,
      ceilMode: Boolean(layer.ceilMode),
      parameters,
    }, inputShape, outputShape);
  }
  if (isAvgPool2dLayer(layer, AvgPool2dModule)) {
    const outputShape = inferAvgPool2dShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "avgPool2d",
      kernelSize: Array.isArray(layer.kernelSize) ? layer.kernelSize.slice() : layer.kernelSize,
      stride: Array.isArray(layer.stride) ? layer.stride.slice() : layer.stride,
      padding: Array.isArray(layer.padding) ? layer.padding.slice() : layer.padding,
      ceilMode: Boolean(layer.ceilMode),
      countIncludePad: Boolean(layer.countIncludePad),
      parameters,
    }, inputShape, outputShape);
  }
  if (isActivationLayer(layer, ActivationModule)) {
    const outputShape = copyShape(inputShape);
    return withTraceShapes({
      index,
      path,
      op: "activation",
      activation: layer.kind,
      parameters,
    }, inputShape, outputShape);
  }
  if (isShapeLayer(layer, ShapeModule)) {
    const outputShape = inferShapeModuleShape(layer, inputShape);
    const op: AnyRecord = {
      index,
      path,
      op: layer.kind,
      parameters,
    };
    if (Array.isArray(layer.shape)) op.shape = layer.shape.slice();
    if (layer.kind === "flatten") {
      op.startDim = layer.startDim;
      op.endDim = layer.endDim;
    } else if (layer.kind === "squeeze") {
      op.squeezeAll = Boolean(layer.squeezeAll);
      if (!layer.squeezeAll) op.dim = layer.dim;
    } else if (layer.kind === "unsqueeze") {
      op.dim = layer.dim;
    } else if (layer.kind === "narrow") {
      op.dim = layer.dim;
      op.start = layer.start;
      op.length = layer.length;
    } else if (layer.kind === "select") {
      op.dim = layer.dim;
      op.selectIndex = layer.index;
    } else if (layer.kind === "slice") {
      op.dim = layer.dim;
      op.start = layer.start;
      op.end = layer.end;
      op.step = layer.step;
    } else if (layer.kind === "transpose") {
      op.dim0 = layer.dim0;
      op.dim1 = layer.dim1;
    } else if (layer.kind === "permute") {
      op.dims = layer.dims.slice();
    }
    return withTraceShapes(op, inputShape, outputShape);
  }
  if (isSoftmaxLayer(layer, SoftmaxModule)) {
    const outputShape = inferSoftmaxShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "softmax",
      dim: layer.dim,
      parameters,
    }, inputShape, outputShape);
  }
  if (isLogSoftmaxLayer(layer, LogSoftmaxModule)) {
    const outputShape = inferSoftmaxShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: "logSoftmax",
      dim: layer.dim,
      parameters,
    }, inputShape, outputShape);
  }
  if (isReductionLayer(layer, ReductionModule)) {
    const outputShape = inferReductionShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: layer.kind,
      dim: layer.dim,
      parameters,
    }, inputShape, outputShape);
  }
  if (isDropoutLayer(layer, DropoutModule)) {
    const outputShape = copyShape(inputShape);
    return withTraceShapes({
      index,
      path,
      op: "dropout",
      p: layer.p,
      training: Boolean(layer.training),
      parameters,
    }, inputShape, outputShape);
  }
  if (isFeatureNormLayer(layer, FeatureNormModule)) {
    const outputShape = inferFeatureNormShape(layer, inputShape);
    return withTraceShapes({
      index,
      path,
      op: layer.kind,
      features: layer.features,
      eps: layer.eps,
      momentum: layer.momentum,
      trackRunningStats: Boolean(layer.trackRunningStats),
      training: Boolean(layer.training),
      affine: Boolean(layer.weight),
      bias: Boolean(layer.bias),
      parameters,
    }, inputShape, outputShape);
  }

  return withTraceShapes({
    index,
    path,
    op: "unknown",
    moduleKind: typeof layer.kind === "string" ? layer.kind : (layer.constructor && layer.constructor.name) || "unknown",
    parameters,
  }, inputShape, inputShape);
}

export function traceSequentialEntries(
  entries: readonly SequentialTraceEntry[],
  options: AnyRecord,
  traceOptions?: SequentialTraceOptions,
): ModuleProgramTrace {
  const inputShape = normalizeTraceShape(traceOptions && traceOptions.inputShape, "trace inputShape");
  const ops: AnyRecord[] = [];
  let currentShape = inputShape;
  for (let index = 0; index < entries.length; index += 1) {
    const op = traceOpForEntry(entries[index], index, options, currentShape);
    ops.push(op);
    currentShape = op.outputShape ? op.outputShape.slice() : null;
  }
  let parameterCount = 0;
  let parameterScalarCount = 0;
  for (const op of ops) {
    parameterCount += op.parameters.length;
    for (const param of op.parameters) parameterScalarCount += param.scalarCount;
  }
  const trace: AnyRecord = {
    kind: "sequential",
    normalized: true,
    layerCount: entries.length,
    opCount: ops.length,
    parameterCount,
    parameterScalarCount,
    shapeKnown: Boolean(inputShape),
    ops,
  };
  if (inputShape) {
    trace.inputShape = inputShape.slice();
    trace.outputShape = currentShape ? currentShape.slice() : null;
  }
  return trace as ModuleProgramTrace;
}

export function compiledSequentialModuleSpec(entries: readonly SequentialTraceEntry[], options: AnyRecord): SequentialCompiledProgramSpec | null {
  const inputShape = defaultTraceInputShape(entries, options);
  if (!inputShape) return null;

  const trace = freezeSequentialTrace(traceSequentialEntries(entries, options, { inputShape }));
  return compiledSequentialModuleSpecFromTrace(entries, trace) as SequentialCompiledProgramSpec | null;
}

export function traceSequentialProgram(layers: readonly AnyRecord[], options: AnyRecord, traceOptions?: SequentialTraceOptions): ModuleProgramTrace {
  return freezeSequentialTrace(traceSequentialEntries(flattenSequentialEntries(layers, options), options, traceOptions)) as ModuleProgramTrace;
}

export function analyzeSequentialProgram(layers: readonly AnyRecord[], options: AnyRecord): SequentialProgramAnalysis {
  const entries = flattenSequentialEntries(layers, options);
  const normalizedLayers = entries.map((entry) => entry.layer);
  let inputShape = null;
  try {
    inputShape = options && options.inputShape ? normalizeTraceShape(options.inputShape, "compile inputShape") : null;
  } catch (err) {
    return shapeMismatchAnalysis(normalizedLayers, null, err);
  }
  const inputShapeRequiresModule = inputShape && inputShape.length !== 1;
  const traceInputShape = defaultTraceInputShape(entries, options);
  let tracedModule: SequentialCompiledProgramSpec | null = null;
  let trace: ModuleProgramTrace | null = null;
  let traceArtifacts: TraceCompilerArtifacts | null = null;
  let compatibility: SequentialCompiledProgramSpec | null = null;
  try {
    if (singleLinearCompatibilityAllowed(normalizedLayers, options, inputShape)) {
      compatibility = compiledSequentialProgramSpecForNormalizedLayers(normalizedLayers, options);
    }
    trace = freezeSequentialTrace(traceSequentialEntries(entries, options, traceInputShape ? { inputShape: traceInputShape } : undefined));
    traceArtifacts = traceCompilerArtifacts(trace);
    if (compatibility && compatibility.kind === "tiny-linear") {
      compatibility = {
        ...compatibility,
        trace,
        ir: traceArtifacts.ir,
        kernelPlan: traceArtifacts.kernelPlan,
      };
    }
    tracedModule = traceInputShape ? compiledSequentialModuleSpecFromTrace(entries, trace, traceArtifacts) : null;
  } catch (err) {
    return shapeMismatchAnalysis(normalizedLayers, inputShape, err);
  }
  const compiled = inputShapeRequiresModule
    ? tracedModule
    : normalizedLayers.length === 1
      ? (compatibility || tracedModule)
      : tracedModule;
  if (compiled) {
    return {
      supported: true,
      reason: null,
      compiled,
      normalizedLayers,
      trace,
      support: supportDetailsWithTrace(
        sequentialProgramSupportDetails(compiled),
        compiled.trace ?? trace,
        compiled.ir ?? null,
        compiled.kernelPlan ?? null,
      ),
    };
  }

  if (options && options.skipLayerSupportFallback) {
    return {
      supported: false,
      reason: traceCompilerUnsupportedReason(traceArtifacts, "Module graph is outside the native module Program compiler subset"),
      compiled: null,
      normalizedLayers,
      trace,
      support: sequentialUnsupportedSupportDetails(normalizedLayers, trace, null, traceArtifacts),
    };
  }

  for (const layer of normalizedLayers) {
    if (typeof layer.compileSupport !== "function") {
      const diagnostic = compileDiagnostic(
        "layer-missing-compiler",
        "module does not expose a native Program compiler",
        {
          stage: "support",
          layerIndex: normalizedLayers.indexOf(layer),
          moduleKind: typeof layer.kind === "string" ? layer.kind : (layer.constructor && layer.constructor.name) || "unknown",
        },
      );
      return {
        supported: false,
        reason: "module does not expose a native Program compiler",
        compiled: null,
        normalizedLayers,
        trace,
        support: sequentialUnsupportedSupportDetails(normalizedLayers, trace, diagnostic, traceArtifacts),
      };
    }
    const support = layer.compileSupport(options);
    if (!support.supported && !support.composable) {
      const diagnostic = compileDiagnostic(
        "layer-unsupported",
        support.reason || "module layer is unsupported by the native Program compiler",
        {
          stage: "support",
          layerIndex: normalizedLayers.indexOf(layer),
          moduleKind: typeof layer.kind === "string" ? layer.kind : (layer.constructor && layer.constructor.name) || "unknown",
        },
      );
      return {
        supported: false,
        reason: support.reason,
        compiled: null,
        normalizedLayers,
        trace,
        support: sequentialUnsupportedSupportDetails(normalizedLayers, trace, diagnostic, traceArtifacts),
      };
    }
  }

  return {
    supported: false,
    reason: traceCompilerUnsupportedReason(traceArtifacts, "Sequential graph is outside the native module Program compiler subset"),
    compiled: null,
    normalizedLayers,
    trace,
    support: sequentialUnsupportedSupportDetails(normalizedLayers, trace, null, traceArtifacts),
  };
}

export type TraceModuleCompilerOptions = Readonly<{
  getConstructors: () => SequentialCompilerConstructors;
}>;

export function createTraceModuleCompiler(options: TraceModuleCompilerOptions): TraceModuleCompiler {
  const getConstructors = options.getConstructors;
  if (typeof getConstructors !== "function") {
    throw new Error("TraceModuleCompiler requires getConstructors");
  }

  function compilerOptions(extra: CompilerOptionsExtra = {}): AnyRecord {
    const constructors = getConstructors();
    return {
      ...constructors,
      inputShape: extra.inputShape,
      skipLayerSupportFallback: extra.skipLayerSupportFallback,
    };
  }

  function trace(layers: readonly AnyRecord[], traceOptions: ModuleTraceOptions = {}): ModuleProgramTrace {
    return traceSequentialProgram(layers, compilerOptions(), traceOptions);
  }

  function analyze(layers: readonly AnyRecord[], compileOptions: CompilerOptionsExtra = {}): SequentialProgramAnalysis {
    return analyzeSequentialProgram(layers, compilerOptions(compileOptions));
  }

  function analyzeSingle(layer: AnyRecord, compileOptions: CompilerOptionsExtra = {}): SequentialProgramAnalysis {
    return analyze([layer], { ...compileOptions, skipLayerSupportFallback: true });
  }

  return Object.freeze({
    trace,
    analyze,
    analyzeSingle,
    packParameters: packedSequentialProgramParameters,
  });
}
