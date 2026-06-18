"use strict";

import {
  shapeScalarCount,
  rowMajorStrides,
} from "../core/shape.js";
import type {
  ModuleProgramTrace,
  ModuleTensorProgramIr,
  ModuleTensorProgramIrOp,
  ModuleTensorProgramIrValue,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
export type TensorProgramIr = ModuleTensorProgramIr;
export type TensorProgramIrOp = ModuleTensorProgramIrOp;
export type TensorProgramIrValue = ModuleTensorProgramIrValue;
export type TensorProgramTraceInput = ModuleProgramTrace | AnyRecord;
export type TensorProgramIrInput = AnyRecord & {
  inputShape: readonly number[];
  outputShape: readonly number[];
  values: readonly AnyRecord[];
  ops: readonly AnyRecord[];
};

function freezeTensorProgramIrOp(op: AnyRecord): TensorProgramIrOp {
  const attrs: AnyRecord = {};
  for (const [key, value] of Object.entries(op.attrs ?? {})) {
    attrs[key] = Array.isArray(value) ? Object.freeze(value.slice()) : value;
  }
  return Object.freeze({
    index: op.index,
    path: op.path,
    op: op.op,
    inputShape: Object.freeze((op.inputShape ?? []).slice()),
    outputShape: Object.freeze((op.outputShape ?? []).slice()),
    inputLen: op.inputLen,
    outputLen: op.outputLen,
    scalarType: op.scalarType ?? "f32",
    scalarBytes: op.scalarBytes ?? 4,
    inputValueIds: Object.freeze((op.inputValueIds ?? []).slice()),
    outputValueId: op.outputValueId,
    parameterValueIds: Object.freeze((op.parameterValueIds ?? []).slice()),
    parameterScalarCount: op.parameterScalarCount ?? 0,
    attrs: Object.freeze(attrs),
  }) as TensorProgramIrOp;
}

function freezeTensorProgramIrValue(value: AnyRecord): TensorProgramIrValue {
  const scalarCount = value.scalarCount ?? shapeScalarCount(value.shape ?? []);
  const scalarBytes = value.scalarBytes ?? 4;
  return Object.freeze({
    ...value,
    shape: Object.freeze((value.shape ?? []).slice()),
    strides: Object.freeze((value.strides ?? []).slice()),
    scalarCount,
    scalarBytes,
    byteLength: value.byteLength ?? scalarCount * scalarBytes,
  }) as TensorProgramIrValue;
}

export function freezeTensorProgramIr(ir: TensorProgramIrInput): TensorProgramIr {
  const frozen = {
    ...ir,
    inputShape: Object.freeze(ir.inputShape.slice()),
    outputShape: Object.freeze(ir.outputShape.slice()),
    values: Object.freeze((ir.values ?? []).map(freezeTensorProgramIrValue)),
    ops: Object.freeze(ir.ops.map(freezeTensorProgramIrOp)),
  };
  return Object.freeze({
    ...frozen,
    signature: tensorProgramIrSignature(frozen),
  }) as TensorProgramIr;
}

function parameterListHasName(parameters: readonly AnyRecord[], name: string) {
  return parameters.some((param) => param.name.endsWith(`.${name}`) || param.name === name);
}

function traceParameterIsAddBias(op: AnyRecord, param: AnyRecord) {
  if (op.op !== "add" || !Array.isArray(param.shape) || param.shape.length !== 1) return false;
  const features = Number(op.features ?? op.outputShape?.[op.outputShape.length - 1]);
  return Number.isSafeInteger(features) && param.shape[0] === features;
}

function parameterBindingForTraceParameter(param: AnyRecord, op: AnyRecord) {
  if (traceParameterIsAddBias(op, param)) return "bias";
  return param.name.endsWith(".bias") || param.name === "bias" ? "bias" : "weights";
}

function tensorProgramIrValue(fields: AnyRecord) {
  const shape = fields.shape.slice();
  const scalarCount = fields.scalarCount ?? shapeScalarCount(shape);
  const scalarBytes = fields.scalarBytes ?? 4;
  return {
    ...fields,
    shape,
    dtype: fields.dtype ?? "f32",
    rank: shape.length,
    scalarCount,
    scalarBytes,
    byteLength: fields.byteLength ?? scalarCount * scalarBytes,
    storageLayout: "row-major",
    strides: rowMajorStrides(shape),
    storageOffset: 0,
    dense: true,
  };
}

function normalizedTraceOpAttrs(op: AnyRecord) {
  switch (op.op) {
    case "linear":
      return {
        inFeatures: op.inFeatures,
        outFeatures: op.outFeatures,
        bias: Boolean(op.bias),
      };
    case "matmul":
      return {
        inFeatures: op.inFeatures,
        outFeatures: op.outFeatures,
        bias: false,
      };
    case "add":
      return {
        features: op.features,
        hasBias: parameterListHasName(op.parameters ?? [], "bias") ||
          (op.parameters ?? []).some((param: AnyRecord) => traceParameterIsAddBias(op, param)),
      };
    case "embedding":
      return {
        numEmbeddings: op.numEmbeddings,
        embeddingDim: op.embeddingDim,
        hasWeight: parameterListHasName(op.parameters ?? [], "weight"),
      };
    case "conv2d":
      return {
        inChannels: op.inChannels,
        outChannels: op.outChannels,
        kernelSize: Array.isArray(op.kernelSize) ? op.kernelSize.slice() : op.kernelSize,
        stride: Array.isArray(op.stride) ? op.stride.slice() : op.stride,
        padding: Array.isArray(op.padding) ? op.padding.slice() : op.padding,
        dilation: Array.isArray(op.dilation) ? op.dilation.slice() : op.dilation,
        groups: op.groups,
        hasWeight: parameterListHasName(op.parameters ?? [], "weight"),
        hasBias: parameterListHasName(op.parameters ?? [], "bias"),
      };
    case "avgPool2d":
      return {
        kernelSize: Array.isArray(op.kernelSize) ? op.kernelSize.slice() : op.kernelSize,
        stride: Array.isArray(op.stride) ? op.stride.slice() : op.stride,
        padding: Array.isArray(op.padding) ? op.padding.slice() : op.padding,
        ceilMode: Boolean(op.ceilMode),
        countIncludePad: Boolean(op.countIncludePad),
      };
    case "maxPool2d":
      return {
        kernelSize: Array.isArray(op.kernelSize) ? op.kernelSize.slice() : op.kernelSize,
        stride: Array.isArray(op.stride) ? op.stride.slice() : op.stride,
        padding: Array.isArray(op.padding) ? op.padding.slice() : op.padding,
        dilation: Array.isArray(op.dilation) ? op.dilation.slice() : op.dilation,
        ceilMode: Boolean(op.ceilMode),
      };
    case "activation":
      return { activation: op.activation };
    case "softmax":
    case "logSoftmax":
    case "sum":
    case "mean":
    case "max":
    case "min":
    case "argmax":
    case "argmin":
      return { dim: op.dim };
    case "dropout":
      return {
        p: op.p,
        training: Boolean(op.training),
      };
    case "diagonal":
      return {};
    case "reshape":
    case "view":
    case "broadcastTo":
    case "expand":
    case "repeat":
    case "tile":
      return { shape: Array.isArray(op.shape) ? op.shape.slice() : [] };
    case "flatten":
      return {
        startDim: op.startDim,
        endDim: op.endDim,
      };
    case "squeeze":
      return {
        squeezeAll: Boolean(op.squeezeAll),
        dim: op.dim,
      };
    case "unsqueeze":
      return { dim: op.dim };
    case "narrow":
      return {
        dim: op.dim,
        start: op.start,
        length: op.length,
      };
    case "select":
      return {
        dim: op.dim,
        selectIndex: op.selectIndex,
      };
    case "slice":
      return {
        dim: op.dim,
        start: op.start,
        end: op.end,
        step: op.step,
      };
    case "transpose":
      return {
        dim0: op.dim0,
        dim1: op.dim1,
      };
    case "permute":
      return {
        dims: Array.isArray(op.dims) ? op.dims.slice() : [],
      };
    case "layerNorm":
      return {
        features: op.features,
        eps: op.eps,
        affine: Boolean(op.affine),
        bias: Boolean(op.bias),
        hasWeight: parameterListHasName(op.parameters ?? [], "weight"),
        hasBias: parameterListHasName(op.parameters ?? [], "bias"),
      };
    case "rmsNorm":
      return {
        features: op.features,
        eps: op.eps,
        affine: Boolean(op.affine),
        bias: Boolean(op.bias),
        hasWeight: parameterListHasName(op.parameters ?? [], "weight"),
      };
    case "batchNorm1d":
      return {
        features: op.features,
        eps: op.eps,
        momentum: op.momentum,
        trackRunningStats: Boolean(op.trackRunningStats),
        training: Boolean(op.training),
        affine: Boolean(op.affine),
        bias: Boolean(op.bias),
        hasWeight: parameterListHasName(op.parameters ?? [], "weight"),
        hasBias: parameterListHasName(op.parameters ?? [], "bias"),
      };
    case "unknown":
      return { moduleKind: op.moduleKind };
    default:
      return {};
  }
}

export function buildTensorProgramIrForTrace(trace: TensorProgramTraceInput): TensorProgramIr {
  const values = [];
  const inputValueId = 0;
  values.push(tensorProgramIrValue({
    id: inputValueId,
    role: "input",
    shape: trace.inputShape.slice(),
  }));
  let currentValueId = inputValueId;
  const ops = trace.ops.map((op: AnyRecord) => {
    const inputValueIds = [currentValueId];
    const parameterValueIds = [];
    let parameterScalarCount = 0;
    for (const param of op.parameters ?? []) {
      const id = values.length;
      const binding = parameterBindingForTraceParameter(param, op);
      parameterScalarCount += param.scalarCount;
      values.push(tensorProgramIrValue({
        id,
        role: "parameter",
        name: param.name,
        binding,
        layout: param.layout ?? null,
        shape: param.shape.slice(),
        scalarCount: param.scalarCount,
        opIndex: op.index,
        op: op.op,
        path: op.path,
      }));
      inputValueIds.push(id);
      parameterValueIds.push(id);
    }
    const outputValueId = values.length;
    values.push(tensorProgramIrValue({
      id: outputValueId,
      role: "op-output",
      shape: op.outputShape.slice(),
      opIndex: op.index,
      op: op.op,
      path: op.path,
    }));
    currentValueId = outputValueId;
    return {
      index: op.index,
      path: op.path,
      op: op.op,
      inputShape: op.inputShape.slice(),
      outputShape: op.outputShape.slice(),
      inputLen: shapeScalarCount(op.inputShape),
      outputLen: shapeScalarCount(op.outputShape),
      scalarType: "f32",
      scalarBytes: 4,
      inputValueIds,
      outputValueId,
      parameterValueIds,
      parameterScalarCount,
      attrs: normalizedTraceOpAttrs(op),
    };
  });
  const outputValueId = currentValueId;
  return freezeTensorProgramIr({
    kind: "tensor-program-ir",
    version: 1,
    inputShape: trace.inputShape.slice(),
    outputShape: trace.outputShape.slice(),
    inputLen: shapeScalarCount(trace.inputShape),
    outputLen: shapeScalarCount(trace.outputShape),
    inputValueId,
    outputValueId,
    valueCount: values.length,
    opCount: trace.ops.length,
    parameterCount: trace.parameterCount,
    parameterScalarCount: trace.parameterScalarCount,
    values,
    ops,
  });
}

function signatureArray(values: unknown) {
  return Array.isArray(values) ? values.join("x") : "";
}

function signatureScalar(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "";
  if (Array.isArray(value)) return `[${value.map(signatureScalar).join(",")}]`;
  return String(value);
}

function signatureAttrs(attrs: unknown) {
  if (!attrs || typeof attrs !== "object") return "";
  return Object.keys(attrs)
    .sort()
    .map((key) => `${key}=${signatureScalar((attrs as AnyRecord)[key])}`)
    .join(",");
}

function tensorProgramIrValueSignature(value: TensorProgramIrValue | AnyRecord) {
  if (!value) return "";
  return [
    value.id,
    value.role,
    signatureArray(value.shape),
    value.dtype ?? "",
    value.rank ?? "",
    value.scalarCount ?? "",
    value.scalarBytes ?? "",
    value.byteLength ?? "",
    value.storageLayout ?? "",
    signatureArray(value.strides),
    value.storageOffset ?? "",
    value.dense === true ? "dense" : value.dense === false ? "strided" : "",
    value.name ?? "",
    value.binding ?? "",
    value.layout ?? "",
    value.opIndex ?? "",
    value.op ?? "",
    value.path ?? "",
  ].join(":");
}

function tensorProgramIrOpSignature(op: TensorProgramIrOp | AnyRecord) {
  if (!op) return "";
  return [
    op.index,
    op.path,
    op.op,
    signatureArray(op.inputShape),
    signatureArray(op.outputShape),
    op.inputLen,
    op.outputLen,
    signatureArray(op.inputValueIds),
    op.outputValueId,
    signatureArray(op.parameterValueIds),
    op.parameterScalarCount,
    signatureAttrs(op.attrs),
  ].join(":");
}

export function tensorProgramIrSignature(ir: TensorProgramIr | AnyRecord | null | undefined): string {
  if (!ir || !Array.isArray(ir.values) || !Array.isArray(ir.ops)) return "";
  const header = [
    ir.kind,
    ir.version,
    signatureArray(ir.inputShape),
    signatureArray(ir.outputShape),
    ir.inputLen,
    ir.outputLen,
    ir.inputValueId,
    ir.outputValueId,
    ir.valueCount,
    ir.opCount,
    ir.parameterCount,
    ir.parameterScalarCount,
  ].join(":");
  return [
    header,
    ir.values.map(tensorProgramIrValueSignature).join("|"),
    ir.ops.map(tensorProgramIrOpSignature).join("|"),
  ].join("::");
}
