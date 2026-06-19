import {
  moduleActivationIds,
  moduleOpIds,
  moduleFlags,
} from "./abi.js";
import {
  kernelNameForNativeModuleDesc,
  kernelNamesForNativeModuleDescs,
  nativeModuleDescSignature,
  nativeModuleDescSignatures,
  type NativeModuleOpDesc,
} from "./native_kernel_contract.js";
import {
  freezeKernelBufferLayout,
  kernelBufferLayout,
} from "./program_buffers.js";
import {
  diagnosticForKernelizerOp,
} from "./compile_diagnostics.js";
import {
  shapeScalarCount,
  traceIndexForDim,
  traceSliceBound,
} from "../core/shape.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type {
  ModuleKernelBufferLayout,
  ModuleKernelMemoryLayout,
  ModuleKernelParameterLayout,
  ModuleKernelPlan,
  ModuleKernelPlanOp,
  ModuleKernelShapeConstraints,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
export type KernelPlan = ModuleKernelPlan;
export type KernelPlanOp = ModuleKernelPlanOp;
export type KernelShapeConstraints = ModuleKernelShapeConstraints;
export type KernelMemoryLayout = ModuleKernelMemoryLayout;
export type KernelParameterLayout = ModuleKernelParameterLayout;
export type KernelBufferLayout = ModuleKernelBufferLayout;
export type InternalKernelPlan = ModuleKernelPlan & Readonly<{
  nativeOps: readonly Readonly<Record<string, unknown>>[];
}>;
type KernelPlanEvidenceFreezeOptions = Readonly<{
  publicOnly?: boolean;
  elided?: boolean;
}>;

const maxActivationChainLength = 4;

function freezeKernelShapeConstraints(constraints: AnyRecord): KernelShapeConstraints {
  return Object.freeze({
    specialization: constraints.specialization,
    inputRank: constraints.inputRank,
    outputRank: constraints.outputRank,
    inputShape: Object.freeze(constraints.inputShape.slice()),
    outputShape: Object.freeze(constraints.outputShape.slice()),
    inputLen: constraints.inputLen,
    outputLen: constraints.outputLen,
  }) as KernelShapeConstraints;
}

function freezeKernelParameterLayout(layout: AnyRecord): KernelParameterLayout {
  const frozen = {
    weightsLen: layout.weightsLen,
    biasLen: layout.biasLen,
    parameters: Object.freeze(layout.parameters.map((param: any) => Object.freeze({
      name: param.name,
      binding: param.binding,
      offset: param.offset,
      scalarCount: param.scalarCount,
      shape: Object.freeze(param.shape.slice()),
      layout: param.layout ?? null,
      opIndex: param.opIndex,
      op: param.op,
      path: param.path,
    }))),
  };
  return Object.freeze({
    ...frozen,
    signature: parameterLayoutSignature(frozen),
  }) as KernelParameterLayout;
}

function freezeKernelMemoryLayout(layout: AnyRecord): KernelMemoryLayout {
  let nextScalarOffset = 0;
  let nextByteOffset = 0;
  let totalScalarCount = 0;
  let totalByteLength = 0;
  let scratchScalarCount = 0;
  let scratchByteLength = 0;
  const values = layout.values.map((value: any) => {
    const scalarBytes = value.scalarBytes ?? layout.scalarBytes ?? 4;
    const scalarCount = value.scalarCount ?? shapeScalarCount(value.shape ?? []);
    const byteLength = value.byteLength ?? scalarCount * scalarBytes;
    const scalarOffset = value.scalarOffset ?? nextScalarOffset;
    const byteOffset = value.byteOffset ?? nextByteOffset;
    const storageClass = value.storageClass ?? "scratch";
    const buffer = value.buffer ?? storageClass;
    const bufferScalarOffset = value.bufferScalarOffset ?? scalarOffset;
    const bufferByteOffset = value.bufferByteOffset ?? byteOffset;
    const consumerOpIndices = Object.freeze((value.consumerOpIndices ?? []).slice());
    const producerOpIndex = value.producerOpIndex ?? (value.role === "op-output" ? value.opIndex ?? null : null);
    const firstUseOpIndex = value.firstUseOpIndex ?? (consumerOpIndices.length > 0 ? consumerOpIndices[0] : null);
    const lastUseOpIndex = value.lastUseOpIndex ?? (consumerOpIndices.length > 0 ? consumerOpIndices[consumerOpIndices.length - 1] : null);
    const liveStartOpIndex = value.liveStartOpIndex ?? (producerOpIndex ?? firstUseOpIndex);
    const liveEndOpIndex = value.liveEndOpIndex ?? (lastUseOpIndex ?? producerOpIndex);
    nextScalarOffset = scalarOffset + scalarCount;
    nextByteOffset = byteOffset + byteLength;
    totalScalarCount = Math.max(totalScalarCount, nextScalarOffset);
    totalByteLength = Math.max(totalByteLength, nextByteOffset);
    if (storageClass === "scratch") {
      scratchScalarCount = Math.max(scratchScalarCount, bufferScalarOffset + scalarCount);
      scratchByteLength = Math.max(scratchByteLength, bufferByteOffset + byteLength);
    }
    return Object.freeze({
      id: value.id,
      role: value.role,
      storageClass,
      buffer,
      producerOpIndex,
      consumerOpIndices,
      firstUseOpIndex,
      lastUseOpIndex,
      liveStartOpIndex,
      liveEndOpIndex,
      scalarType: value.scalarType,
      scalarBytes,
      shape: Object.freeze(value.shape.slice()),
      rank: value.rank,
      scalarCount,
      scalarOffset,
      bufferScalarOffset,
      byteOffset,
      bufferByteOffset,
      byteLength,
      storageLayout: value.storageLayout,
      strides: Object.freeze(value.strides.slice()),
      storageOffset: value.storageOffset,
      dense: value.dense,
      name: value.name,
      binding: value.binding,
      layout: value.layout ?? null,
      opIndex: value.opIndex,
      op: value.op,
      path: value.path,
    });
  });
  const frozen = {
    scalarType: layout.scalarType,
    scalarBytes: layout.scalarBytes,
    valueCount: layout.valueCount,
    totalScalarCount: layout.totalScalarCount ?? totalScalarCount,
    totalByteLength: layout.totalByteLength ?? totalByteLength,
    scratchScalarCount: layout.scratchScalarCount ?? scratchScalarCount,
    scratchByteLength: layout.scratchByteLength ?? scratchByteLength,
    values: Object.freeze(values),
  };
  return Object.freeze({
    ...frozen,
    signature: memoryLayoutSignature(frozen),
  }) as KernelMemoryLayout;
}

function freezeKernelPlanEvidenceOp(op: AnyRecord, options: KernelPlanEvidenceFreezeOptions = {}): KernelPlanOp {
  const publicOnly = options.publicOnly === true;
  const nativeDispatchCount = Number.isSafeInteger(op.nativeDispatchCount) && op.nativeDispatchCount >= 0
    ? op.nativeDispatchCount
    : options.elided === true ? 0 : 1;
  const nativeDescriptorCount = Number.isSafeInteger(op.nativeDescriptorCount) && op.nativeDescriptorCount >= 0
    ? op.nativeDescriptorCount
    : options.elided === true ? 0 : nativeDispatchCount;
  const nativeKernels = Array.isArray(op.nativeKernels)
    ? op.nativeKernels.slice()
    : nativeDispatchCount === 0 ? [] : [op.kernel];
  const nativeDescriptorSignatures = Array.isArray(op.nativeDescriptorSignatures)
    ? op.nativeDescriptorSignatures.slice()
    : nativeDescriptorCount === 0 ? [] : op.desc ? [nativeModuleDescSignature(op.desc)] : [];
  const frozen: any = publicOnly
    ? {
        index: op.index,
        path: op.path,
        op: op.op,
        kernel: op.kernel,
        inputShape: Object.freeze(op.inputShape.slice()),
        outputShape: Object.freeze(op.outputShape.slice()),
        inputLen: op.inputLen,
        outputLen: op.outputLen,
        scalarType: op.scalarType,
        scalarBytes: op.scalarBytes,
        inputValueIds: Object.freeze((op.inputValueIds ?? []).slice()),
        outputValueId: op.outputValueId,
        parameterScalarCount: op.parameterScalarCount,
        nativeDispatchCount,
        nativeDescriptorCount,
        nativeKernels: Object.freeze(nativeKernels),
        nativeDescriptorSignatures: Object.freeze(nativeDescriptorSignatures),
      }
    : {
        ...op,
        nativeDispatchCount,
        nativeDescriptorCount,
        nativeKernels: Object.freeze(nativeKernels),
        nativeDescriptorSignatures: Object.freeze(nativeDescriptorSignatures),
        inputShape: Object.freeze(op.inputShape.slice()),
        outputShape: Object.freeze(op.outputShape.slice()),
        inputValueIds: Object.freeze((op.inputValueIds ?? []).slice()),
        desc: Object.freeze({ ...op.desc }),
      };
  if (publicOnly && op.fusedOpCount !== undefined) frozen.fusedOpCount = op.fusedOpCount;
  if (Array.isArray(op.fusedOps)) frozen.fusedOps = Object.freeze(op.fusedOps.slice());
  if (Array.isArray(op.fusedIndices)) frozen.fusedIndices = Object.freeze(op.fusedIndices.slice());
  if (Array.isArray(op.fusedValueEdges)) {
    frozen.fusedValueEdges = Object.freeze(op.fusedValueEdges.map((edge: any) => Object.freeze({
      opIndex: edge.opIndex,
      op: edge.op,
      path: edge.path,
      inputValueIds: Object.freeze((edge.inputValueIds ?? []).slice()),
      outputValueId: edge.outputValueId,
    })));
  }
  return Object.freeze(frozen) as KernelPlanOp;
}

function freezeKernelPlanEvidenceOps(ops: readonly AnyRecord[] | null | undefined, options: KernelPlanEvidenceFreezeOptions = {}): readonly KernelPlanOp[] {
  return Object.freeze((ops ?? []).map((op: any) => freezeKernelPlanEvidenceOp(op, options)));
}

function freezeKernelPlan(plan: AnyRecord): InternalKernelPlan {
  const frozen = {
    ...plan,
    inputShape: Object.freeze(plan.inputShape.slice()),
    outputShape: Object.freeze(plan.outputShape.slice()),
    shapeConstraints: freezeKernelShapeConstraints(plan.shapeConstraints),
    memoryLayout: freezeKernelMemoryLayout(plan.memoryLayout),
    bufferLayout: freezeKernelBufferLayout(plan.bufferLayout),
    parameterLayout: freezeKernelParameterLayout(plan.parameterLayout),
    ops: freezeKernelPlanEvidenceOps(plan.ops),
    elidedOps: freezeKernelPlanEvidenceOps(plan.elidedOps, { elided: true }),
    nativeOps: Object.freeze(plan.nativeOps.map((op: any) => Object.freeze({ ...op }))),
  };
  return Object.freeze({
    ...frozen,
    signature: kernelPlanSignature(frozen),
  }) as InternalKernelPlan;
}

function freezePublicKernelPlan(plan: AnyRecord): KernelPlan {
  const frozen = {
    kind: plan.kind,
    version: plan.version,
    nativePath: plan.nativePath,
    signature: plan.signature ?? kernelPlanSignature(plan),
    inputShape: Object.freeze(plan.inputShape.slice()),
    outputShape: Object.freeze(plan.outputShape.slice()),
    inputLen: plan.inputLen,
    outputLen: plan.outputLen,
    shapeConstraints: freezeKernelShapeConstraints(plan.shapeConstraints),
    memoryLayout: freezeKernelMemoryLayout(plan.memoryLayout),
    bufferLayout: freezeKernelBufferLayout(plan.bufferLayout),
    weightsLen: plan.weightsLen,
    biasLen: plan.biasLen,
    parameterLayout: freezeKernelParameterLayout(plan.parameterLayout),
    opCount: plan.opCount,
    dispatchCount: plan.dispatchCount,
    descriptorCount: plan.descriptorCount,
    elidedOpCount: plan.elidedOpCount ?? 0,
    ops: freezeKernelPlanEvidenceOps(plan.ops, { publicOnly: true }),
    elidedOps: freezeKernelPlanEvidenceOps(plan.elidedOps, { publicOnly: true, elided: true }),
  };
  return Object.freeze(frozen) as KernelPlan;
}

function frontendAxisForRank(dim: any, rank: any) {
  return dim < 0 ? rank + dim : dim;
}

function nativeAxisForFrontendAxis(axis: any, rank: any) {
  return rank - 1 - axis;
}

function narrowIrShape(op: any) {
  if (!Array.isArray(op.inputShape)) return null;
  const rank = op.inputShape.length;
  const attrs = op.attrs ?? {};
  const axis = frontendAxisForRank(attrs.dim, rank);
  if (axis < 0 || axis >= rank) return null;
  const dim = op.inputShape[axis];
  const start = traceIndexForDim(attrs.start, dim, "narrow ir");
  if (!Number.isSafeInteger(attrs.length) || attrs.length <= 0 || start + attrs.length > dim) return null;
  return { rank, axis, start, length: attrs.length };
}

function narrowIrCanLower(op: any) {
  const shape = narrowIrShape(op);
  if (!shape) return false;
  if (shape.rank === 1) return shape.axis === 0;
  if (shape.rank === 3 && op.inputShape[0] !== 1) return false;
  return (shape.rank === 2 || shape.rank === 3) && shape.axis >= 0 && shape.axis < shape.rank;
}

function selectIrInfo(op: any) {
  if (!Array.isArray(op.inputShape)) return null;
  const rank = op.inputShape.length;
  const attrs = op.attrs ?? {};
  if (!Number.isSafeInteger(attrs.dim) || !Number.isSafeInteger(attrs.selectIndex)) return null;
  const axis = frontendAxisForRank(attrs.dim, rank);
  if (axis < 0 || axis >= rank) return null;
  const dim = op.inputShape[axis];
  const index = traceIndexForDim(attrs.selectIndex, dim, "select ir");
  const narrowedShape = op.inputShape.slice();
  narrowedShape[axis] = 1;
  const outputShape = op.inputShape.filter((_: any, shapeAxis: any) => shapeAxis !== axis);
  if (outputShape.length === 0) outputShape.push(1);
  return { rank, axis, index, narrowedShape, outputShape };
}

function selectIrCanLower(op: any) {
  const info = selectIrInfo(op);
  if (!info) return false;
  if (info.rank === 1) return info.axis === 0;
  if (info.rank === 3 && op.inputShape[0] !== 1) return false;
  return (info.rank === 2 || info.rank === 3) && info.axis >= 0 && info.axis < info.rank;
}

function sliceIrInfo(op: any) {
  if (!Array.isArray(op.inputShape)) return null;
  const rank = op.inputShape.length;
  const attrs = op.attrs ?? {};
  if (!Number.isSafeInteger(attrs.dim)) return null;
  const axis = frontendAxisForRank(attrs.dim, rank);
  if (axis < 0 || axis >= rank) return null;
  const step = attrs.step === undefined ? 1 : attrs.step;
  if (!Number.isSafeInteger(step) || step <= 0) return null;
  const dim = op.inputShape[axis];
  const start = traceSliceBound(attrs.start, dim, 0, "slice ir start");
  const end = traceSliceBound(attrs.end, dim, dim, "slice ir end");
  const length = Math.ceil((end - start) / step);
  if (length <= 0) return null;
  const outputShape = op.inputShape.slice();
  outputShape[axis] = length;
  return { rank, axis, start, end, step, length, outputShape };
}

function sliceIrCanLower(op: any) {
  const info = sliceIrInfo(op);
  if (!info) return false;
  if (info.rank === 1) return info.axis === 0;
  if (info.rank === 3 && op.inputShape[0] !== 1) return false;
  return (info.rank === 2 || info.rank === 3) && info.axis >= 0 && info.axis < info.rank;
}

function reshapeDescForShape(outputShape: any) {
  if (!outputShape || outputShape.length < 1 || outputShape.length > 3) return null;
  if (outputShape.length === 3 && outputShape[2] > 0xffffffff) return null;
  return {
    kind: moduleOpIds.reshape,
    activation: 0,
    flags: 0,
    reserved: outputShape.length === 3 ? outputShape[2] : 0,
    a: outputShape.length,
    b: outputShape[0],
    c: outputShape.length >= 2 ? outputShape[1] : 0,
    eps: 0,
  };
}

function permuteDescForIrOp(op: any): NativeModuleOpDesc | null {
  const attrs = op.attrs ?? {};
  const rank = op.inputShape ? op.inputShape.length : 0;
  if (rank !== 2 || !Array.isArray(attrs.dims) || attrs.dims.length !== 2) return null;
  const axis0 = frontendAxisForRank(attrs.dims[0], rank);
  const axis1 = frontendAxisForRank(attrs.dims[1], rank);
  if (axis0 === 0 && axis1 === 1) return reshapeDescForShape(op.outputShape);
  if (!(axis0 === 1 && axis1 === 0)) return null;
  return {
    kind: moduleOpIds.transpose,
    activation: 0,
    flags: 0,
    a: 0,
    b: 1,
    c: 0,
    eps: 0,
  };
}

function isNoopDropoutIrOp(op: any) {
  const attrs = op.attrs ?? {};
  return op.op === "dropout" && (attrs.training === false || attrs.p === 0);
}

function normalizedPair(value: any, fallback: readonly [number, number]): readonly [number, number] | null {
  const pair = Array.isArray(value) ? value : value === undefined ? fallback : [value, value];
  if (pair.length !== 2 || !Number.isSafeInteger(pair[0]) || !Number.isSafeInteger(pair[1])) return null;
  return [pair[0], pair[1]];
}

function maxPool2dDescForIrOp(op: any): NativeModuleOpDesc | null {
  const attrs = op.attrs ?? {};
  const inputShape = op.inputShape;
  if (!Array.isArray(inputShape) || (inputShape.length !== 3 && inputShape.length !== 4)) return null;
  const heightDim = inputShape.length === 4 ? inputShape[2] : inputShape[1];
  const widthDim = inputShape.length === 4 ? inputShape[3] : inputShape[2];
  const kernel = normalizedPair(attrs.kernelSize, [2, 2]);
  const stride = normalizedPair(attrs.stride, kernel ?? [2, 2]);
  const padding = normalizedPair(attrs.padding, [0, 0]);
  const dilation = normalizedPair(attrs.dilation, [1, 1]);
  if (!kernel || !stride || !padding || !dilation) return null;
  if (
    kernel[0] !== 2 || kernel[1] !== 2 ||
    stride[0] !== 2 || stride[1] !== 2 ||
    padding[0] !== 0 || padding[1] !== 0 ||
    dilation[0] !== 1 || dilation[1] !== 1 ||
    attrs.ceilMode === true
  ) {
    return null;
  }
  if (heightDim % 2 !== 0 || widthDim % 2 !== 0) return null;
  return { kind: moduleOpIds.maxPool2d, activation: 0, flags: 0, a: 2, b: 2, c: 0, eps: 0 };
}

function avgPool2dDescForIrOp(op: any): NativeModuleOpDesc | null {
  const attrs = op.attrs ?? {};
  const inputShape = op.inputShape;
  if (!Array.isArray(inputShape) || (inputShape.length !== 3 && inputShape.length !== 4)) return null;
  const heightDim = inputShape.length === 4 ? inputShape[2] : inputShape[1];
  const widthDim = inputShape.length === 4 ? inputShape[3] : inputShape[2];
  const kernel = normalizedPair(attrs.kernelSize, [2, 2]);
  const stride = normalizedPair(attrs.stride, kernel ?? [2, 2]);
  const padding = normalizedPair(attrs.padding, [0, 0]);
  if (!kernel || !stride || !padding) return null;
  if (
    kernel[0] !== 2 || kernel[1] !== 2 ||
    stride[0] !== 2 || stride[1] !== 2 ||
    padding[0] !== 0 || padding[1] !== 0 ||
    attrs.ceilMode === true ||
    attrs.countIncludePad === false
  ) {
    return null;
  }
  if (heightDim % 2 !== 0 || widthDim % 2 !== 0) return null;
  return { kind: moduleOpIds.avgPool2d, activation: 0, flags: 0, a: 2, b: 2, c: 0, eps: 0 };
}

function conv2dDescForIrOp(op: any): NativeModuleOpDesc | null {
  const attrs = op.attrs ?? {};
  const inputShape = op.inputShape;
  if (!Array.isArray(inputShape) || (inputShape.length !== 3 && inputShape.length !== 4)) return null;
  const channelDim = inputShape.length === 4 ? inputShape[1] : inputShape[0];
  const heightDim = inputShape.length === 4 ? inputShape[2] : inputShape[1];
  const widthDim = inputShape.length === 4 ? inputShape[3] : inputShape[2];
  const kernel = normalizedPair(attrs.kernelSize, [1, 1]);
  const stride = normalizedPair(attrs.stride, [1, 1]);
  const padding = normalizedPair(attrs.padding, [0, 0]);
  const dilation = normalizedPair(attrs.dilation, [1, 1]);
  if (!kernel || !stride || !padding || !dilation) return null;
  if (
    attrs.groups !== 1 ||
    attrs.hasWeight !== true ||
    stride[0] !== 1 || stride[1] !== 1 ||
    padding[0] !== 0 || padding[1] !== 0 ||
    dilation[0] !== 1 || dilation[1] !== 1
  ) {
    return null;
  }
  if (channelDim !== attrs.inChannels || kernel[0] > heightDim || kernel[1] > widthDim) return null;
  return {
    kind: moduleOpIds.conv2d,
    activation: 0,
    flags: attrs.hasBias ? moduleFlags.bias : 0,
    a: attrs.outChannels,
    b: kernel[0],
    c: kernel[1],
    eps: 0,
  };
}

function moduleOpDescForIrOp(op: any): NativeModuleOpDesc | null {
  const attrs = op.attrs ?? {};
  switch (op.op) {
    case "linear":
      return {
        kind: moduleOpIds.linear,
        activation: 0,
        flags: attrs.bias ? moduleFlags.bias : 0,
        a: attrs.inFeatures,
        b: attrs.outFeatures,
        c: 0,
        eps: 0,
      };
    case "matmul":
      return {
        kind: moduleOpIds.linear,
        activation: 0,
        flags: 0,
        a: attrs.inFeatures,
        b: attrs.outFeatures,
        c: 0,
        eps: 0,
      };
    case "add":
      return {
        kind: moduleOpIds.add,
        activation: 0,
        flags: moduleFlags.bias,
        a: attrs.features,
        b: 0,
        c: 0,
        eps: 0,
      };
    case "activation": {
      const activation = moduleActivationIds[attrs.activation];
      if (!activation) return null;
      return { kind: moduleOpIds.activation, activation, flags: 0, a: 0, b: 0, c: 0, eps: 0 };
    }
    case "softmax": {
      const rank = op.inputShape ? op.inputShape.length : 0;
      const axis = frontendAxisForRank(attrs.dim, rank);
      const nativeAxis = nativeAxisForFrontendAxis(axis, rank);
      if (axis < 0 || axis >= rank || nativeAxis !== 0) return null;
      return { kind: moduleOpIds.softmax, activation: 0, flags: 0, a: nativeAxis, b: 0, c: 0, eps: 0 };
    }
    case "logSoftmax": {
      const rank = op.inputShape ? op.inputShape.length : 0;
      const axis = frontendAxisForRank(attrs.dim, rank);
      const nativeAxis = nativeAxisForFrontendAxis(axis, rank);
      if (axis < 0 || axis >= rank || nativeAxis !== 0) return null;
      return { kind: moduleOpIds.logSoftmax, activation: 0, flags: 0, a: nativeAxis, b: 0, c: 0, eps: 0 };
    }
    case "sum":
    case "mean":
    case "prod":
    case "max":
    case "min":
    case "argmax":
    case "argmin": {
      const rank = op.inputShape ? op.inputShape.length : 0;
      const axis = frontendAxisForRank(attrs.dim, rank);
      const nativeAxis = nativeAxisForFrontendAxis(axis, rank);
      if (rank === 3 && op.inputShape[0] !== 1) return null;
      if (axis < 0 || axis >= rank || nativeAxis !== 0) return null;
      const kind = op.op === "sum"
        ? moduleOpIds.reduceSum
        : op.op === "mean"
          ? moduleOpIds.reduceMean
          : op.op === "prod"
            ? moduleOpIds.reduceProd
            : op.op === "max"
              ? moduleOpIds.reduceMax
              : op.op === "min"
                ? moduleOpIds.reduceMin
                : op.op === "argmax"
                  ? moduleOpIds.reduceArgmax
                  : moduleOpIds.reduceArgmin;
      return { kind, activation: 0, flags: 0, a: nativeAxis, b: 0, c: 0, eps: 0 };
    }
    case "identity":
    case "reshape":
    case "view":
    case "flatten":
    case "squeeze":
    case "unsqueeze":
      return reshapeDescForShape(op.outputShape);
    case "dropout":
      return isNoopDropoutIrOp(op) ? reshapeDescForShape(op.outputShape) : null;
    case "broadcastTo":
    case "expand":
      if (!op.outputShape || op.outputShape.length < 1 || op.outputShape.length > 3) return null;
      if (op.outputShape.length === 3 && op.outputShape[2] > 0xffffffff) return null;
      return {
        kind: moduleOpIds.broadcastTo,
        activation: 0,
        flags: 0,
        reserved: op.outputShape.length === 3 ? op.outputShape[2] : 0,
        a: op.outputShape.length,
        b: op.outputShape[0],
        c: op.outputShape.length >= 2 ? op.outputShape[1] : 0,
        eps: 0,
      };
    case "repeat":
    case "tile":
      if (!op.outputShape || op.outputShape.length < 1 || op.outputShape.length > 2) return null;
      return {
        kind: moduleOpIds.broadcastTo,
        activation: 0,
        flags: 0,
        a: op.outputShape.length,
        b: op.outputShape[0],
        c: op.outputShape.length === 2 ? op.outputShape[1] : 0,
        eps: 0,
      };
    case "narrow": {
      const shape = narrowIrShape(op);
      if (!shape || !narrowIrCanLower(op)) return null;
      return {
        kind: moduleOpIds.narrow,
        activation: 0,
        flags: 0,
        a: shape.axis,
        b: shape.start,
        c: shape.length,
        eps: 0,
      };
    }
    case "slice": {
      const shape = sliceIrInfo(op);
      if (!shape || !sliceIrCanLower(op)) return null;
      return {
        kind: moduleOpIds.slice,
        activation: 0,
        flags: 0,
        reserved: shape.step,
        a: shape.axis,
        b: shape.start,
        c: shape.length,
        eps: 0,
      };
    }
    case "transpose": {
      const rank = op.inputShape ? op.inputShape.length : 0;
      if (rank !== 2) return null;
      const axis0 = frontendAxisForRank(attrs.dim0, rank);
      const axis1 = frontendAxisForRank(attrs.dim1, rank);
      if (!((axis0 === 0 && axis1 === 1) || (axis0 === 1 && axis1 === 0))) return null;
      return {
        kind: moduleOpIds.transpose,
        activation: 0,
        flags: 0,
        a: axis0,
        b: axis1,
        c: 0,
        eps: 0,
      };
    }
    case "diagonal":
      if (!op.inputShape || op.inputShape.length !== 2) return null;
      return {
        kind: moduleOpIds.diagonal,
        activation: 0,
        flags: 0,
        a: 0,
        b: 0,
        c: 0,
        eps: 0,
      };
    case "permute":
      return permuteDescForIrOp(op);
    case "layerNorm":
      if (!op.inputShape || op.inputShape[op.inputShape.length - 1] !== attrs.features) return null;
      return {
        kind: moduleOpIds.layerNorm,
        activation: 0,
        flags: (attrs.hasWeight ? moduleFlags.weight : 0) |
          (attrs.hasBias ? moduleFlags.bias : 0),
        a: attrs.features,
        b: 0,
        c: 0,
        eps: attrs.eps,
      };
    case "rmsNorm":
      if (!op.inputShape || op.inputShape[op.inputShape.length - 1] !== attrs.features) return null;
      return {
        kind: moduleOpIds.rmsNorm,
        activation: 0,
        flags: attrs.hasWeight ? moduleFlags.weight : 0,
        a: attrs.features,
        b: 0,
        c: 0,
        eps: attrs.eps,
      };
    case "batchNorm1d":
      if (
        attrs.training === true ||
        attrs.trackRunningStats !== true ||
        attrs.hasWeight !== true ||
        attrs.hasBias !== true ||
        !op.inputShape ||
        op.inputShape[op.inputShape.length - 1] !== attrs.features
      ) {
        return null;
      }
      return {
        kind: moduleOpIds.featureAffine,
        activation: 0,
        flags: moduleFlags.weight | moduleFlags.bias,
        a: attrs.features,
        b: 0,
        c: 0,
        eps: 0,
      };
    case "embedding":
      if (!op.inputShape || op.inputShape.length !== 1) return null;
      if (!attrs.hasWeight) return null;
      return {
        kind: moduleOpIds.embedding,
        activation: 0,
        flags: moduleFlags.weight,
        a: attrs.numEmbeddings,
        b: attrs.embeddingDim,
        c: 0,
        eps: 0,
      };
    case "conv2d":
      return conv2dDescForIrOp(op);
    case "maxPool2d":
      return maxPool2dDescForIrOp(op);
    case "avgPool2d":
      return avgPool2dDescForIrOp(op);
    default:
      return null;
  }
}

function moduleActivationIdForIrOp(op: any) {
  return op && op.op === "activation" ? moduleActivationIds[op.attrs?.activation] : undefined;
}

function canAppendActivationChainIrOp(chain: any, op: any) {
  if (!op || op.op !== "activation") return false;
  if (!moduleActivationIdForIrOp(op)) return false;
  if (op.parameterValueIds.length !== 0 || op.inputValueIds.length !== 1) return false;
  if (!traceShapesEqual(op.inputShape, op.outputShape)) return false;
  if (chain.length === 0) return true;
  const prev = chain[chain.length - 1];
  return op.inputValueIds[0] === prev.outputValueId &&
    traceShapesEqual(prev.outputShape, op.inputShape);
}

function activationChainIrOpsAt(ops: any, startIndex: any) {
  const chain: any[] = [];
  for (let index = startIndex; index < ops.length && chain.length < maxActivationChainLength; index += 1) {
    const op = ops[index];
    if (!canAppendActivationChainIrOp(chain, op)) break;
    chain.push(op);
  }
  return chain.length > 1 ? chain : null;
}

function activationChainDescForIrOps(chain: any): NativeModuleOpDesc {
  const ids = chain.map(moduleActivationIdForIrOp);
  return {
    kind: moduleOpIds.activationChain,
    activation: ids[0],
    flags: ids.length,
    a: ids[1] ?? 0,
    b: ids[2] ?? 0,
    c: ids[3] ?? 0,
    eps: 0,
  };
}

function transposeModuleOpDesc(axis0: any = 0, axis1: any = 1): NativeModuleOpDesc {
  return {
    kind: moduleOpIds.transpose,
    activation: 0,
    flags: 0,
    a: axis0,
    b: axis1,
    c: 0,
    eps: 0,
  };
}

function activationModuleOpDesc(activation: any): NativeModuleOpDesc | null {
  const activationId = moduleActivationIds[activation];
  if (!activationId) return null;
  return {
    kind: moduleOpIds.activation,
    activation: activationId,
    flags: 0,
    a: 0,
    b: 0,
    c: 0,
    eps: 0,
  };
}

function reduceDimModuleOpDesc(kind: number): NativeModuleOpDesc {
  return {
    kind,
    activation: 0,
    flags: 0,
    a: 0,
    b: 0,
    c: 0,
    eps: 0,
  };
}

function reduceModuleOpKind(op: string): number | null {
  switch (op) {
    case "sum": return moduleOpIds.reduceSum;
    case "mean": return moduleOpIds.reduceMean;
    case "prod": return moduleOpIds.reduceProd;
    case "max": return moduleOpIds.reduceMax;
    case "min": return moduleOpIds.reduceMin;
    case "argmax": return moduleOpIds.reduceArgmax;
    case "argmin": return moduleOpIds.reduceArgmin;
    default: return null;
  }
}

function reduceDimModuleOpDescs(op: any, attrs: any): readonly NativeModuleOpDesc[] | null {
  const kind = reduceModuleOpKind(op.op);
  if (!kind) return null;
  const rank = op.inputShape ? op.inputShape.length : 0;
  const axis = frontendAxisForRank(attrs.dim, rank);
  const nativeAxis = nativeAxisForFrontendAxis(axis, rank);
  if (rank === 3 && op.inputShape[0] !== 1) return null;
  if (axis < 0 || axis >= rank) return null;
  const reduce = reduceDimModuleOpDesc(kind);
  if (nativeAxis === 0) return [reduce];
  if (rank === 2 && nativeAxis === 1) {
    return [transposeModuleOpDesc(), reduce, transposeModuleOpDesc()];
  }
  return null;
}

function moduleOpDescsForIrOp(op: any): readonly NativeModuleOpDesc[] | null {
  const attrs = op.attrs ?? {};
  if (op.op === "sum" || op.op === "mean" || op.op === "prod" || op.op === "max" || op.op === "min" || op.op === "argmax" || op.op === "argmin") {
    const desc = moduleOpDescForIrOp(op);
    if (desc) return [desc];
    return reduceDimModuleOpDescs(op, attrs);
  }

  if (op.op === "softmax" || op.op === "logSoftmax") {
    const desc = moduleOpDescForIrOp(op);
    if (desc) return [desc];
    const rank = op.inputShape ? op.inputShape.length : 0;
    const axis = frontendAxisForRank(attrs.dim, rank);
    const nativeAxis = nativeAxisForFrontendAxis(axis, rank);
    if (rank === 2 && nativeAxis === 1) {
      const middle = {
        kind: op.op === "softmax" ? moduleOpIds.softmax : moduleOpIds.logSoftmax,
        activation: 0,
        flags: 0,
        a: 0,
        b: 0,
        c: 0,
        eps: 0,
      };
      return [transposeModuleOpDesc(), middle, transposeModuleOpDesc()];
    }
    return null;
  }

  if (op.op !== "select") {
    const desc = moduleOpDescForIrOp(op);
    return desc ? [desc] : null;
  }

  const info = selectIrInfo(op);
  if (!info || !selectIrCanLower(op)) return null;
  return [{
    kind: moduleOpIds.narrow,
    activation: 0,
    flags: 0,
    a: info.axis,
    b: info.index,
    c: 1,
    eps: 0,
  }];
}

function kernelNameForIrOp(op: any) {
  const attrs = op.attrs ?? {};
  switch (op.op) {
    case "linear": return "linear";
    case "matmul": return "matmul";
    case "add": return "add";
    case "activation": return attrs.activation;
    case "softmax": return "softmax";
    case "logSoftmax": return "log-softmax";
    case "sum": return "sum";
    case "mean": return "mean";
    case "prod": return "prod";
    case "max": return "max";
    case "min": return "min";
    case "argmax": return "argmax";
    case "argmin": return "argmin";
    case "identity":
    case "reshape":
    case "view":
    case "flatten":
    case "squeeze":
    case "unsqueeze":
    case "dropout": return "reshape";
    case "broadcastTo":
    case "expand":
    case "repeat":
    case "tile": return "broadcast";
    case "narrow": return "narrow";
    case "select": return "select";
    case "slice": return "slice";
    case "transpose": return "transpose";
    case "permute": return "transpose";
    case "diagonal": return "diagonal";
    case "layerNorm": return "layer-norm";
    case "rmsNorm": return "rms-norm";
    case "batchNorm1d": return "affine";
    case "embedding": return "embedding";
    case "conv2d": return "conv2d";
    case "maxPool2d": return "max-pool2d";
    case "avgPool2d": return "avg-pool2d";
    default: return op.op;
  }
}

function parameterLayoutForOps(ops: any, values: any) {
  const parameters = [];
  let weightsLen = 0;
  let biasLen = 0;
  for (const op of ops) {
    for (const valueId of op.parameterValueIds) {
      const param = values[valueId];
      if (!param || param.role !== "parameter") continue;
      const binding = param.binding;
      const offset = binding === "bias" ? biasLen : weightsLen;
      parameters.push({
        name: param.name,
        binding,
        offset,
        scalarCount: param.scalarCount,
        shape: param.shape.slice(),
        layout: param.layout ?? null,
        opIndex: op.index,
        op: op.op,
        path: op.path,
      });
      if (binding === "bias") biasLen += param.scalarCount;
      else weightsLen += param.scalarCount;
    }
  }
  return { weightsLen, biasLen, parameters };
}

function memoryLayoutStorageForValue(value: any, outputValueId: any) {
  if (value.role === "input") return { storageClass: "step-input", buffer: "input" };
  if (value.role === "parameter") return { storageClass: "persistent", buffer: value.binding ?? "weights" };
  if (value.id === outputValueId) return { storageClass: "step-output", buffer: "output" };
  return { storageClass: "scratch", buffer: "scratch" };
}

function memoryLayoutLifetimesForValues(values: any, ops: any) {
  const consumersByValueId = new Map<any, any[]>(values.map((value: any) => [value.id, []]));
  for (const op of ops) {
    for (const valueId of op.inputValueIds ?? []) {
      const consumers = consumersByValueId.get(valueId);
      if (consumers) consumers.push(op.index);
    }
  }
  return new Map(values.map((value: any) => {
    const consumerOpIndices = (consumersByValueId.get(value.id) ?? []).slice();
    const producerOpIndex = value.role === "op-output" ? value.opIndex ?? null : null;
    const firstUseOpIndex = consumerOpIndices.length > 0 ? consumerOpIndices[0] : null;
    const lastUseOpIndex = consumerOpIndices.length > 0 ? consumerOpIndices[consumerOpIndices.length - 1] : null;
    return [value.id, {
      producerOpIndex,
      consumerOpIndices,
      firstUseOpIndex,
      lastUseOpIndex,
      liveStartOpIndex: producerOpIndex ?? firstUseOpIndex,
      liveEndOpIndex: lastUseOpIndex ?? producerOpIndex,
    }];
  }));
}

function allocateScratchValue(scratchBlocks: any, nextScratch: any, scalarCount: any, byteLength: any, lifetime: any) {
  const liveStart = lifetime.liveStartOpIndex;
  const liveEnd = lifetime.liveEndOpIndex;
  if (liveStart !== null && liveStart !== undefined) {
    for (const block of scratchBlocks) {
      if (
        block.scalarCount >= scalarCount &&
        block.byteLength >= byteLength &&
        block.liveEndOpIndex !== null &&
        block.liveEndOpIndex !== undefined &&
        block.liveEndOpIndex < liveStart
      ) {
        block.liveEndOpIndex = liveEnd;
        return {
          bufferScalarOffset: block.scalarOffset,
          bufferByteOffset: block.byteOffset,
          nextScratch,
        };
      }
    }
  }
  const allocated = {
    scalarOffset: nextScratch.scalarOffset,
    byteOffset: nextScratch.byteOffset,
    scalarCount,
    byteLength,
    liveEndOpIndex: liveEnd,
  };
  scratchBlocks.push(allocated);
  return {
    bufferScalarOffset: allocated.scalarOffset,
    bufferByteOffset: allocated.byteOffset,
    nextScratch: {
      scalarOffset: nextScratch.scalarOffset + scalarCount,
      byteOffset: nextScratch.byteOffset + byteLength,
    },
  };
}

function memoryLayoutForValues(values: any, outputValueId: any, ops: any) {
  let scalarOffset = 0;
  let byteOffset = 0;
  let nextScratch = { scalarOffset: 0, byteOffset: 0 };
  const scratchBlocks: any[] = [];
  let weightsScalarOffset = 0;
  let weightsByteOffset = 0;
  let biasScalarOffset = 0;
  let biasByteOffset = 0;
  const lifetimes: Map<any, any> = memoryLayoutLifetimesForValues(values, ops);
  const layoutValues = values.map((value: any) => {
    const scalarBytes = value.scalarBytes ?? 4;
    const scalarCount = value.scalarCount;
    const byteLength = value.byteLength ?? scalarCount * scalarBytes;
    const { storageClass, buffer } = memoryLayoutStorageForValue(value, outputValueId);
    const lifetime = lifetimes.get(value.id) ?? {};
    let bufferScalarOffset = 0;
    let bufferByteOffset = 0;
    if (buffer === "scratch") {
      const allocation = allocateScratchValue(scratchBlocks, nextScratch, scalarCount, byteLength, lifetime);
      bufferScalarOffset = allocation.bufferScalarOffset;
      bufferByteOffset = allocation.bufferByteOffset;
      nextScratch = allocation.nextScratch;
    } else if (buffer === "weights") {
      bufferScalarOffset = weightsScalarOffset;
      bufferByteOffset = weightsByteOffset;
      weightsScalarOffset += scalarCount;
      weightsByteOffset += byteLength;
    } else if (buffer === "bias") {
      bufferScalarOffset = biasScalarOffset;
      bufferByteOffset = biasByteOffset;
      biasScalarOffset += scalarCount;
      biasByteOffset += byteLength;
    }
    const entry = {
      id: value.id,
      role: value.role,
      storageClass,
      buffer,
      producerOpIndex: lifetime.producerOpIndex ?? null,
      consumerOpIndices: (lifetime.consumerOpIndices ?? []).slice(),
      firstUseOpIndex: lifetime.firstUseOpIndex ?? null,
      lastUseOpIndex: lifetime.lastUseOpIndex ?? null,
      liveStartOpIndex: lifetime.liveStartOpIndex ?? null,
      liveEndOpIndex: lifetime.liveEndOpIndex ?? null,
      scalarType: value.dtype ?? "f32",
      scalarBytes,
      shape: value.shape.slice(),
      rank: value.rank,
      scalarCount,
      scalarOffset,
      bufferScalarOffset,
      byteOffset,
      bufferByteOffset,
      byteLength,
      storageLayout: value.storageLayout,
      strides: value.strides.slice(),
      storageOffset: value.storageOffset,
      dense: value.dense,
      name: value.name,
      binding: value.binding,
      layout: value.layout ?? null,
      opIndex: value.opIndex,
      op: value.op,
      path: value.path,
    };
    scalarOffset += scalarCount;
    byteOffset += byteLength;
    return entry;
  });
  return {
    scalarType: "f32",
    scalarBytes: 4,
    valueCount: values.length,
    totalScalarCount: scalarOffset,
    totalByteLength: byteOffset,
    scratchScalarCount: nextScratch.scalarOffset,
    scratchByteLength: nextScratch.byteOffset,
    values: layoutValues,
  };
}

function kernelShapeConstraints(inputShape: any, outputShape: any) {
  return {
    specialization: "exact",
    inputRank: inputShape.length,
    outputRank: outputShape.length,
    inputShape: inputShape.slice(),
    outputShape: outputShape.slice(),
    inputLen: shapeScalarCount(inputShape),
    outputLen: shapeScalarCount(outputShape),
  };
}

function isReshapeKernelIrOp(op: any) {
  return op.op === "identity" ||
    op.op === "reshape" ||
    op.op === "view" ||
    op.op === "flatten" ||
    op.op === "squeeze" ||
    op.op === "unsqueeze" ||
    isNoopDropoutIrOp(op);
}

function traceShapesEqual(a: any, b: any) {
  return Array.isArray(a) &&
    Array.isArray(b) &&
    a.length === b.length &&
    a.every((dim, index) => dim === b[index]);
}

function canElideShapeKernelIrOps(shapeOps: any, hasFollowingOp: any) {
  if (shapeOps.length === 0) return false;
  const first = shapeOps[0];
  const last = shapeOps[shapeOps.length - 1];
  if (hasFollowingOp) return traceShapesEqual(first.inputShape, last.outputShape);
  return first.inputLen === last.outputLen;
}

function scalarEvidenceForIrOp(op: any, values: any) {
  const output = values && values[op.outputValueId];
  return {
    scalarType: output && output.dtype ? output.dtype : "f32",
    scalarBytes: output && output.scalarBytes ? output.scalarBytes : 4,
  };
}

function kernelPlanOpForIrOp(op: any, descs: readonly NativeModuleOpDesc[], values: any, nativeDispatchCount: any = descs.length, nativeDescriptorCount: any = descs.length) {
  return {
    index: op.index,
    path: op.path,
    op: op.op,
    kernel: kernelNameForIrOp(op),
    inputShape: op.inputShape.slice(),
    outputShape: op.outputShape.slice(),
    inputLen: op.inputLen,
    outputLen: op.outputLen,
    ...scalarEvidenceForIrOp(op, values),
    inputValueIds: op.inputValueIds.slice(),
    outputValueId: op.outputValueId,
    parameterScalarCount: op.parameterScalarCount,
    nativeDispatchCount,
    nativeDescriptorCount,
    nativeKernels: kernelNamesForNativeModuleDescs(descs),
    nativeDescriptorSignatures: nativeModuleDescSignatures(descs),
    desc: descs[0],
  };
}

function fusedValueEdgeForIrOp(op: any) {
  return Object.freeze({
    opIndex: op.index,
    op: op.op,
    path: op.path,
    inputValueIds: Object.freeze((op.inputValueIds ?? []).slice()),
    outputValueId: op.outputValueId,
  });
}

function fusedValueEdgesForIrOps(ops: any) {
  return Object.freeze(ops.map(fusedValueEdgeForIrOp));
}

function canFuseLinearActivationIrOps(linearOp: any, activationOp: any) {
  if (!linearOp || !activationOp) return false;
  if (linearOp.op !== "linear" || activationOp.op !== "activation") return false;
  const activation = moduleActivationIds[activationOp.attrs?.activation];
  if (!activation) return false;
  return activationOp.inputValueIds.length === 1 &&
    activationOp.parameterValueIds.length === 0 &&
    activationOp.inputValueIds[0] === linearOp.outputValueId &&
    traceShapesEqual(linearOp.outputShape, activationOp.inputShape) &&
    traceShapesEqual(activationOp.inputShape, activationOp.outputShape);
}

function fusedLinearActivationKernelPlanOp(linearOp: any, activationOp: any, desc: any, values: any) {
  return {
    index: linearOp.index,
    path: `${linearOp.path}..${activationOp.path}`,
    op: "linear",
    kernel: "linear",
    inputShape: linearOp.inputShape.slice(),
    outputShape: activationOp.outputShape.slice(),
    inputLen: linearOp.inputLen,
    outputLen: activationOp.outputLen,
    ...scalarEvidenceForIrOp(activationOp, values),
    inputValueIds: linearOp.inputValueIds.slice(),
    outputValueId: activationOp.outputValueId,
    parameterScalarCount: linearOp.parameterScalarCount,
    nativeDispatchCount: 1,
    nativeDescriptorCount: 1,
    nativeKernels: ["linear", kernelNameForIrOp(activationOp)],
    nativeDescriptorSignatures: [nativeModuleDescSignature(desc)],
    fusedOpCount: 2,
    fusedOps: [linearOp.op, activationOp.op],
    fusedIndices: [linearOp.index, activationOp.index],
    fusedValueEdges: fusedValueEdgesForIrOps([linearOp, activationOp]),
    desc,
  };
}

function biasParameterValueForAddIrOp(addOp: any, values: any) {
  if (!addOp || addOp.op !== "add" || addOp.parameterValueIds.length !== 1) return null;
  const parameterValueId = addOp.parameterValueIds[0];
  const parameter = values && values[parameterValueId];
  if (!parameter || parameter.role !== "parameter" || parameter.binding !== "bias") return null;
  return parameter;
}

function canFuseMatmulAddActivationIrOps(matmulOp: any, addOp: any, activationOp: any, values: any) {
  if (!matmulOp || !addOp) return false;
  if (matmulOp.op !== "matmul" || addOp.op !== "add") return false;
  const bias = biasParameterValueForAddIrOp(addOp, values);
  if (!bias) return false;
  if (
    addOp.inputValueIds.length !== 2 ||
    addOp.inputValueIds[0] !== matmulOp.outputValueId ||
    !traceShapesEqual(matmulOp.outputShape, addOp.inputShape) ||
    !traceShapesEqual(matmulOp.outputShape, addOp.outputShape)
  ) {
    return false;
  }
  const outFeatures = matmulOp.attrs?.outFeatures;
  if (!Number.isSafeInteger(outFeatures) || bias.scalarCount !== outFeatures) return false;
  if (Array.isArray(bias.shape) && bias.shape.length !== 1) return false;
  if (!activationOp) return true;
  const activation = moduleActivationIds[activationOp.attrs?.activation];
  if (!activation) return false;
  return activationOp.op === "activation" &&
    activationOp.inputValueIds.length === 1 &&
    activationOp.parameterValueIds.length === 0 &&
    activationOp.inputValueIds[0] === addOp.outputValueId &&
    traceShapesEqual(addOp.outputShape, activationOp.inputShape) &&
    traceShapesEqual(activationOp.inputShape, activationOp.outputShape);
}

function fusedMatmulAddActivationKernelPlanOp(matmulOp: any, addOp: any, activationOp: any | null, desc: any, values: any) {
  const fusedOps = activationOp ? [matmulOp, addOp, activationOp] : [matmulOp, addOp];
  const last = fusedOps[fusedOps.length - 1];
  return {
    index: matmulOp.index,
    path: `${matmulOp.path}..${last.path}`,
    op: "matmul",
    kernel: "linear",
    inputShape: matmulOp.inputShape.slice(),
    outputShape: last.outputShape.slice(),
    inputLen: matmulOp.inputLen,
    outputLen: last.outputLen,
    ...scalarEvidenceForIrOp(last, values),
    inputValueIds: Object.freeze([
      ...matmulOp.inputValueIds,
      addOp.parameterValueIds[0],
    ]),
    outputValueId: last.outputValueId,
    parameterScalarCount: matmulOp.parameterScalarCount + addOp.parameterScalarCount,
    nativeDispatchCount: 1,
    nativeDescriptorCount: 1,
    nativeKernels: activationOp ? ["linear", "add", kernelNameForIrOp(activationOp)] : ["linear", "add"],
    nativeDescriptorSignatures: [nativeModuleDescSignature(desc)],
    fusedOpCount: fusedOps.length,
    fusedOps: fusedOps.map((op: any) => op.op),
    fusedIndices: fusedOps.map((op: any) => op.index),
    fusedValueEdges: fusedValueEdgesForIrOps(fusedOps),
    desc,
  };
}

function canFuseConv2dActivationIrOps(convOp: any, activationOp: any) {
  if (!convOp || !activationOp) return false;
  if (convOp.op !== "conv2d" || activationOp.op !== "activation") return false;
  const activation = moduleActivationIds[activationOp.attrs?.activation];
  if (!activation) return false;
  return activationOp.inputValueIds.length === 1 &&
    activationOp.parameterValueIds.length === 0 &&
    activationOp.inputValueIds[0] === convOp.outputValueId &&
    traceShapesEqual(convOp.outputShape, activationOp.inputShape) &&
    traceShapesEqual(activationOp.inputShape, activationOp.outputShape);
}

function fusedConv2dActivationKernelPlanOp(convOp: any, activationOp: any, desc: any, values: any) {
  return {
    index: convOp.index,
    path: `${convOp.path}..${activationOp.path}`,
    op: "conv2d",
    kernel: "conv2d",
    inputShape: convOp.inputShape.slice(),
    outputShape: activationOp.outputShape.slice(),
    inputLen: convOp.inputLen,
    outputLen: activationOp.outputLen,
    ...scalarEvidenceForIrOp(activationOp, values),
    inputValueIds: convOp.inputValueIds.slice(),
    outputValueId: activationOp.outputValueId,
    parameterScalarCount: convOp.parameterScalarCount,
    nativeDispatchCount: 1,
    nativeDescriptorCount: 1,
    nativeKernels: ["conv2d", kernelNameForIrOp(activationOp)],
    nativeDescriptorSignatures: [nativeModuleDescSignature(desc)],
    fusedOpCount: 2,
    fusedOps: [convOp.op, activationOp.op],
    fusedIndices: [convOp.index, activationOp.index],
    fusedValueEdges: fusedValueEdgesForIrOps([convOp, activationOp]),
    desc,
  };
}

function canFuseFeatureNormActivationIrOps(normOp: any, activationOp: any) {
  if (!normOp || !activationOp) return false;
  if (normOp.op !== "layerNorm" && normOp.op !== "rmsNorm") return false;
  const activation = moduleActivationIds[activationOp.attrs?.activation];
  if (!activation) return false;
  return activationOp.inputValueIds.length === 1 &&
    activationOp.parameterValueIds.length === 0 &&
    activationOp.inputValueIds[0] === normOp.outputValueId &&
    traceShapesEqual(normOp.outputShape, activationOp.inputShape) &&
    traceShapesEqual(activationOp.inputShape, activationOp.outputShape);
}

function fusedFeatureNormActivationKernelPlanOp(normOp: any, activationOp: any, desc: any, values: any) {
  return {
    index: normOp.index,
    path: `${normOp.path}..${activationOp.path}`,
    op: normOp.op,
    kernel: kernelNameForIrOp(normOp),
    inputShape: normOp.inputShape.slice(),
    outputShape: activationOp.outputShape.slice(),
    inputLen: normOp.inputLen,
    outputLen: activationOp.outputLen,
    ...scalarEvidenceForIrOp(activationOp, values),
    inputValueIds: normOp.inputValueIds.slice(),
    outputValueId: activationOp.outputValueId,
    parameterScalarCount: normOp.parameterScalarCount,
    nativeDispatchCount: 1,
    nativeDescriptorCount: 1,
    nativeKernels: [kernelNameForIrOp(normOp), kernelNameForIrOp(activationOp)],
    nativeDescriptorSignatures: [nativeModuleDescSignature(desc)],
    fusedOpCount: 2,
    fusedOps: [normOp.op, activationOp.op],
    fusedIndices: [normOp.index, activationOp.index],
    fusedValueEdges: fusedValueEdgesForIrOps([normOp, activationOp]),
    desc,
  };
}

function fusedActivationChainKernelPlanOp(chain: any, desc: any, values: any) {
  const first = chain[0];
  const last = chain[chain.length - 1];
  return {
    index: first.index,
    path: `${first.path}..${last.path}`,
    op: "activation-chain",
    kernel: "activation-chain",
    inputShape: first.inputShape.slice(),
    outputShape: last.outputShape.slice(),
    inputLen: first.inputLen,
    outputLen: last.outputLen,
    ...scalarEvidenceForIrOp(last, values),
    inputValueIds: first.inputValueIds.slice(),
    outputValueId: last.outputValueId,
    parameterScalarCount: 0,
    nativeDispatchCount: 1,
    nativeDescriptorCount: 1,
    nativeKernels: chain.map(kernelNameForIrOp),
    nativeDescriptorSignatures: [nativeModuleDescSignature(desc)],
    fusedOpCount: chain.length,
    fusedOps: chain.map((op: any) => op.op),
    fusedIndices: chain.map((op: any) => op.index),
    fusedValueEdges: fusedValueEdgesForIrOps(chain),
    desc,
  };
}

function coalescedShapeKernelPlanOp(shapeOps: any, desc: any, values: any, nativeDispatchCount: any = 1) {
  const first = shapeOps[0];
  const last = shapeOps[shapeOps.length - 1];
  if (shapeOps.length === 1) return kernelPlanOpForIrOp(first, [desc], values, nativeDispatchCount, nativeDispatchCount === 0 ? 0 : 1);
  return {
    index: first.index,
    path: `${first.path}..${last.path}`,
    op: "shape-chain",
    kernel: "reshape",
    inputShape: first.inputShape.slice(),
    outputShape: last.outputShape.slice(),
    inputLen: first.inputLen,
    outputLen: last.outputLen,
    ...scalarEvidenceForIrOp(last, values),
    inputValueIds: first.inputValueIds.slice(),
    outputValueId: last.outputValueId,
    parameterScalarCount: 0,
    nativeDispatchCount,
    nativeDescriptorCount: nativeDispatchCount === 0 ? 0 : 1,
    nativeKernels: nativeDispatchCount === 0 ? [] : [kernelNameForNativeModuleDesc(desc)],
    nativeDescriptorSignatures: nativeDispatchCount === 0 ? [] : [nativeModuleDescSignature(desc)],
    fusedOpCount: shapeOps.length,
    fusedOps: shapeOps.map((op: any) => op.op),
    fusedIndices: shapeOps.map((op: any) => op.index),
    fusedValueEdges: fusedValueEdgesForIrOps(shapeOps),
    desc,
  };
}

function kernelizerDiagnosticForIrOp(op: any) {
  if (op.op === "dropout" && !isNoopDropoutIrOp(op)) {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-op",
      "native module Program compiler can only lower deterministic no-op nn.Dropout",
    );
  }
  if (op.op === "narrow" && !narrowIrCanLower(op)) {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-view",
      "native module Program narrow currently supports rank-1/rank-2 and singleton-envelope rank-3 materialized output views",
    );
  }
  if (op.op === "select" && !selectIrCanLower(op)) {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-view",
      "native module Program select currently supports rank-1/rank-2 and singleton-envelope rank-3 materialized output views",
    );
  }
  if (op.op === "slice" && !sliceIrCanLower(op)) {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-view",
      "native module Program slice currently supports rank-1/rank-2 and singleton-envelope rank-3 materialized output views",
    );
  }
  if (op.op === "transpose") {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-view",
      "native module Program transpose currently supports rank-2 axis swaps only",
    );
  }
  if (op.op === "permute") {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-view",
      "native module Program permute currently supports rank-2 identity or axis swaps only",
    );
  }
  if (op.op === "maxPool2d") {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-op",
      "native module Program MaxPool2d currently supports rank-3 [channels,height,width] or rank-4 [batch,channels,height,width] 2x2 stride-2 pooling without padding, dilation, or ceil mode",
    );
  }
  if (op.op === "conv2d") {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-op",
      "native module Program Conv2d currently supports rank-3 [channels,height,width] or rank-4 [batch,channels,height,width], groups=1, stride=1, padding=0, dilation=1, and dense f32 weights with optional bias",
    );
  }
  if (op.op === "avgPool2d") {
    return diagnosticForKernelizerOp(
      op,
      "unsupported-op",
      "native module Program AvgPool2d currently supports rank-3 [channels,height,width] or rank-4 [batch,channels,height,width] 2x2 stride-2 pooling with countIncludePad=true and without padding or ceil mode",
    );
  }
  return diagnosticForKernelizerOp(op, "unsupported-op", `native module Program compiler cannot lower ${op.op}`);
}

function kernelizeTensorProgramIr(ir: any) {
  const nativeOps = [];
  const ops = [];
  const elidedOps = [];
  for (let index = 0; index < ir.ops.length; index += 1) {
    const op = ir.ops[index];
    if (isReshapeKernelIrOp(op)) {
      const shapeOps = [op];
      while (index + 1 < ir.ops.length && isReshapeKernelIrOp(ir.ops[index + 1])) {
        index += 1;
        shapeOps.push(ir.ops[index]);
      }
      const finalShapeOp = shapeOps[shapeOps.length - 1];
      const desc = moduleOpDescForIrOp(finalShapeOp);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(finalShapeOp),
        };
      }
      if (canElideShapeKernelIrOps(shapeOps, index + 1 < ir.ops.length)) {
        elidedOps.push(coalescedShapeKernelPlanOp(shapeOps, desc, ir.values, 0));
        continue;
      }
      nativeOps.push(desc);
      ops.push(coalescedShapeKernelPlanOp(shapeOps, desc, ir.values));
      continue;
    }

    const activationChain = activationChainIrOpsAt(ir.ops, index);
    if (activationChain) {
      const desc = activationChainDescForIrOps(activationChain);
      nativeOps.push(desc);
      ops.push(fusedActivationChainKernelPlanOp(activationChain, desc, ir.values));
      index += activationChain.length - 1;
      continue;
    }

    if (ir.ops[index + 2] && canFuseMatmulAddActivationIrOps(op, ir.ops[index + 1], ir.ops[index + 2], ir.values)) {
      const addOp = ir.ops[index + 1];
      const activationOp = ir.ops[index + 2];
      const desc = moduleOpDescForIrOp(op);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(op),
        };
      }
      const fusedDesc = Object.freeze({
        ...desc,
        flags: (desc.flags ?? 0) | moduleFlags.bias,
        activation: moduleActivationIds[activationOp.attrs.activation],
      });
      nativeOps.push(fusedDesc);
      ops.push(fusedMatmulAddActivationKernelPlanOp(op, addOp, activationOp, fusedDesc, ir.values));
      index += 2;
      continue;
    }

    if (canFuseMatmulAddActivationIrOps(op, ir.ops[index + 1], null, ir.values)) {
      const addOp = ir.ops[index + 1];
      const desc = moduleOpDescForIrOp(op);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(op),
        };
      }
      const fusedDesc = Object.freeze({
        ...desc,
        flags: (desc.flags ?? 0) | moduleFlags.bias,
      });
      nativeOps.push(fusedDesc);
      ops.push(fusedMatmulAddActivationKernelPlanOp(op, addOp, null, fusedDesc, ir.values));
      index += 1;
      continue;
    }

    if (canFuseLinearActivationIrOps(op, ir.ops[index + 1])) {
      const activationOp = ir.ops[index + 1];
      const desc = moduleOpDescForIrOp(op);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(op),
        };
      }
      const fusedDesc = Object.freeze({
        ...desc,
        activation: moduleActivationIds[activationOp.attrs.activation],
      });
      nativeOps.push(fusedDesc);
      ops.push(fusedLinearActivationKernelPlanOp(op, activationOp, fusedDesc, ir.values));
      index += 1;
      continue;
    }

    if (canFuseConv2dActivationIrOps(op, ir.ops[index + 1])) {
      const activationOp = ir.ops[index + 1];
      const desc = moduleOpDescForIrOp(op);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(op),
        };
      }
      const fusedDesc = Object.freeze({
        ...desc,
        activation: moduleActivationIds[activationOp.attrs.activation],
      });
      nativeOps.push(fusedDesc);
      ops.push(fusedConv2dActivationKernelPlanOp(op, activationOp, fusedDesc, ir.values));
      index += 1;
      continue;
    }

    if (canFuseFeatureNormActivationIrOps(op, ir.ops[index + 1])) {
      const activationOp = ir.ops[index + 1];
      const desc = moduleOpDescForIrOp(op);
      if (!desc) {
        return {
          kernelPlan: null,
          diagnostic: kernelizerDiagnosticForIrOp(op),
        };
      }
      const fusedDesc = Object.freeze({
        ...desc,
        activation: moduleActivationIds[activationOp.attrs.activation],
      });
      nativeOps.push(fusedDesc);
      ops.push(fusedFeatureNormActivationKernelPlanOp(op, activationOp, fusedDesc, ir.values));
      index += 1;
      continue;
    }

    const descs = moduleOpDescsForIrOp(op);
    if (!descs) {
      return {
        kernelPlan: null,
        diagnostic: kernelizerDiagnosticForIrOp(op),
      };
    }
    for (const desc of descs) nativeOps.push(desc);
    ops.push(kernelPlanOpForIrOp(op, descs, ir.values));
  }
  const parameterLayout = parameterLayoutForOps(ir.ops, ir.values);
  const { weightsLen, biasLen } = parameterLayout;
  const shapeConstraints = kernelShapeConstraints(ir.inputShape, ir.outputShape);
  const memoryLayout = memoryLayoutForValues(ir.values, ir.outputValueId, ir.ops);
  const bufferLayout = kernelBufferLayout(ir.inputLen, ir.outputLen, weightsLen, biasLen);
  const descriptorCount = nativeOps.length;
  const dispatchCount = ops.reduce((total, op) => total + (Number.isSafeInteger(op.nativeDispatchCount) ? op.nativeDispatchCount : 1), 0);
  return {
    kernelPlan: freezeKernelPlan({
      kind: "native-module-kernel-plan",
      version: 1,
      nativePath: "device-program",
      inputShape: ir.inputShape.slice(),
      outputShape: ir.outputShape.slice(),
      inputLen: ir.inputLen,
      outputLen: ir.outputLen,
      shapeConstraints,
      memoryLayout,
      bufferLayout,
      weightsLen,
      biasLen,
      parameterLayout,
      opCount: ir.opCount,
      dispatchCount,
      descriptorCount,
      elidedOpCount: elidedOps.length,
      ops,
      elidedOps,
      nativeOps,
    }),
    diagnostic: null,
  };
}

function programShapeConstraintsFromKernelPlan(plan: KernelPlan | AnyRecord | null | undefined): KernelShapeConstraints | null {
  return plan ? plan.shapeConstraints : null;
}

function kernelPlanInputShape(plan: KernelPlan | AnyRecord | null | undefined) {
  const constraints = programShapeConstraintsFromKernelPlan(plan);
  if (constraints && Array.isArray(constraints.inputShape)) return constraints.inputShape;
  return plan && Array.isArray(plan.inputShape) ? plan.inputShape : null;
}

function kernelPlanOutputShape(plan: KernelPlan | AnyRecord | null | undefined) {
  const constraints = programShapeConstraintsFromKernelPlan(plan);
  if (constraints && Array.isArray(constraints.outputShape)) return constraints.outputShape;
  return plan && Array.isArray(plan.outputShape) ? plan.outputShape : null;
}

function kernelPlanInputLen(plan: KernelPlan | AnyRecord | null | undefined) {
  const constraints = programShapeConstraintsFromKernelPlan(plan);
  if (constraints && Number.isSafeInteger(constraints.inputLen)) return constraints.inputLen;
  return plan && Number.isSafeInteger(plan.inputLen) ? plan.inputLen : undefined;
}

function kernelPlanOutputLen(plan: KernelPlan | AnyRecord | null | undefined) {
  const constraints = programShapeConstraintsFromKernelPlan(plan);
  if (constraints && Number.isSafeInteger(constraints.outputLen)) return constraints.outputLen;
  return plan && Number.isSafeInteger(plan.outputLen) ? plan.outputLen : undefined;
}

function programMemoryLayoutFromKernelPlan(plan: KernelPlan | AnyRecord | null | undefined): KernelMemoryLayout | null {
  return plan ? plan.memoryLayout : null;
}

function programParameterLayoutFromKernelPlan(plan: KernelPlan | AnyRecord | null | undefined): KernelParameterLayout | null {
  return plan ? plan.parameterLayout : null;
}

function kernelPlanSignature(plan: KernelPlan | AnyRecord | null | undefined): string {
  if (!plan || !Array.isArray(plan.ops)) return "";
  const valueEdge = (op: any) => `${(op.inputValueIds ?? []).join("+")}=>${op.outputValueId ?? ""}`;
  const fusedValueEdges = (op: any) => Array.isArray(op.fusedValueEdges)
    ? op.fusedValueEdges.map((edge: any) => `${edge.opIndex}:${edge.op}:${(edge.inputValueIds ?? []).join("+")}=>${edge.outputValueId ?? ""}`).join("+")
    : "";
  const descriptorEvidence = (op: any) => Array.isArray(op.nativeDescriptorSignatures)
    ? op.nativeDescriptorSignatures.join("+")
    : nativeModuleDescSignature(op.desc);
  const scheduled = plan.ops.map((op: any) => {
    const fused = Array.isArray(op.fusedOps) ? op.fusedOps.join("+") : "";
    const indices = Array.isArray(op.fusedIndices) ? op.fusedIndices.join("+") : "";
    return `${op.op}:${op.kernel}:${op.inputShape.join("x")}:${op.outputShape.join("x")}:${op.scalarType ?? ""}:${op.scalarBytes ?? ""}:${op.parameterScalarCount}:${op.nativeDispatchCount ?? ""}:${op.nativeDescriptorCount ?? ""}:${(op.nativeKernels ?? []).join("+")}:${valueEdge(op)}:${op.fusedOpCount ?? 1}:${fused}:${indices}:${fusedValueEdges(op)}:${descriptorEvidence(op)}`;
  }).join("|");
  const elided = Array.isArray(plan.elidedOps)
    ? plan.elidedOps.map((op: any) => {
      const fused = Array.isArray(op.fusedOps) ? op.fusedOps.join("+") : "";
      const indices = Array.isArray(op.fusedIndices) ? op.fusedIndices.join("+") : "";
      return `${op.op}:${op.kernel}:${op.inputShape.join("x")}:${op.outputShape.join("x")}:${op.scalarType ?? ""}:${op.scalarBytes ?? ""}:${op.parameterScalarCount}:${op.nativeDispatchCount ?? ""}:${op.nativeDescriptorCount ?? ""}:${(op.nativeKernels ?? []).join("+")}:${valueEdge(op)}:${op.fusedOpCount ?? 1}:${fused}:${indices}:${fusedValueEdges(op)}:${descriptorEvidence(op)}`;
    }).join("|")
    : "";
  return `dispatch=${plan.dispatchCount ?? ""}:descriptor=${plan.descriptorCount ?? ""}::${scheduled}::elided::${elided}`;
}

function parameterLayoutSignature(layout: KernelParameterLayout | AnyRecord | null | undefined): string {
  return layout && Array.isArray(layout.parameters)
    ? layout.parameters.map((param: any) => `${param.name}:${param.binding}:${param.offset}:${param.scalarCount}:${param.shape.join("x")}:${param.layout ?? ""}`).join("|")
    : "";
}

function memoryLayoutSignature(layout: KernelMemoryLayout | AnyRecord | null | undefined): string {
  if (!layout || !Array.isArray(layout.values)) return "";
  let nextScalarOffset = 0;
  let nextByteOffset = 0;
  let totalScalarCount = 0;
  let totalByteLength = 0;
  let scratchScalarCount = 0;
  let scratchByteLength = 0;
  const valueSignatures = layout.values.map((value: any) => {
    const scalarBytes = value.scalarBytes ?? layout.scalarBytes ?? 4;
    const scalarCount = value.scalarCount ?? shapeScalarCount(value.shape ?? []);
    const byteLength = value.byteLength ?? scalarCount * scalarBytes;
    const scalarOffset = value.scalarOffset ?? nextScalarOffset;
    const byteOffset = value.byteOffset ?? nextByteOffset;
    const storageClass = value.storageClass ?? "scratch";
    const buffer = value.buffer ?? storageClass;
    const bufferScalarOffset = value.bufferScalarOffset ?? scalarOffset;
    const bufferByteOffset = value.bufferByteOffset ?? byteOffset;
    const consumerOpIndices = Array.isArray(value.consumerOpIndices) ? value.consumerOpIndices : [];
    const producerOpIndex = value.producerOpIndex ?? (value.role === "op-output" ? value.opIndex ?? null : null);
    const firstUseOpIndex = value.firstUseOpIndex ?? (consumerOpIndices.length > 0 ? consumerOpIndices[0] : null);
    const lastUseOpIndex = value.lastUseOpIndex ?? (consumerOpIndices.length > 0 ? consumerOpIndices[consumerOpIndices.length - 1] : null);
    const liveStartOpIndex = value.liveStartOpIndex ?? (producerOpIndex ?? firstUseOpIndex);
    const liveEndOpIndex = value.liveEndOpIndex ?? (lastUseOpIndex ?? producerOpIndex);
    nextScalarOffset = scalarOffset + scalarCount;
    nextByteOffset = byteOffset + byteLength;
    totalScalarCount = Math.max(totalScalarCount, nextScalarOffset);
    totalByteLength = Math.max(totalByteLength, nextByteOffset);
    if (storageClass === "scratch") {
      scratchScalarCount = Math.max(scratchScalarCount, bufferScalarOffset + scalarCount);
      scratchByteLength = Math.max(scratchByteLength, bufferByteOffset + byteLength);
    }
    return [
      value.id,
      value.role,
      storageClass,
      buffer,
      producerOpIndex ?? "",
      consumerOpIndices.join("+"),
      firstUseOpIndex ?? "",
      lastUseOpIndex ?? "",
      liveStartOpIndex ?? "",
      liveEndOpIndex ?? "",
      value.scalarType,
      scalarBytes,
      value.shape.join("x"),
      value.rank,
      scalarCount,
      scalarOffset,
      bufferScalarOffset,
      byteOffset,
      bufferByteOffset,
      byteLength,
      value.storageLayout,
      value.strides.join("x"),
      value.storageOffset,
      value.dense === true ? "dense" : value.dense === false ? "strided" : "",
      value.name ?? "",
      value.binding ?? "",
      value.layout ?? "",
      value.opIndex ?? "",
      value.op ?? "",
      value.path ?? "",
    ].join(":");
  }).join("|");
  return [
    layout.scalarType ?? "",
    layout.scalarBytes ?? "",
    layout.valueCount ?? layout.values.length,
    layout.totalScalarCount ?? totalScalarCount,
    layout.totalByteLength ?? totalByteLength,
    layout.scratchScalarCount ?? scratchScalarCount,
    layout.scratchByteLength ?? scratchByteLength,
    valueSignatures,
  ].join("::");
}

function bufferLayoutSignature(layout: KernelBufferLayout | AnyRecord | null | undefined): string {
  return layout && Array.isArray(layout.slots)
    ? layout.slots.map((slot: any) => `${slot.name}:${slot.role}:${slot.scalarType}:${slot.scalarBytes}:${slot.elementOffset}:${slot.elementCount}:${slot.byteOffset}:${slot.byteLength}`).join("|")
    : "";
}

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

function isShapeArray(value: unknown): value is readonly number[] {
  return Array.isArray(value) && value.length > 0 && value.every((dim) => Number.isSafeInteger(dim) && dim > 0);
}

function kernelPlanOpHasEvidence(op: unknown): op is AnyRecord {
  return isRecord(op) &&
    Number.isSafeInteger(op.index) &&
    typeof op.path === "string" &&
    typeof op.op === "string" &&
    typeof op.kernel === "string" &&
    isShapeArray(op.inputShape) &&
    isShapeArray(op.outputShape) &&
    Number.isSafeInteger(op.inputLen) &&
    Number.isSafeInteger(op.outputLen) &&
    Array.isArray(op.inputValueIds) &&
    Number.isSafeInteger(op.outputValueId) &&
    Number.isSafeInteger(op.nativeDispatchCount) &&
    Number.isSafeInteger(op.nativeDescriptorCount) &&
    Array.isArray(op.nativeKernels) &&
    Array.isArray(op.nativeDescriptorSignatures);
}

function kernelPlanHasLayoutEvidence(plan: AnyRecord) {
  return isRecord(plan.shapeConstraints) &&
    isRecord(plan.memoryLayout) &&
    isRecord(plan.parameterLayout) &&
    isRecord(plan.bufferLayout) &&
    typeof plan.memoryLayout.signature === "string" &&
    typeof plan.parameterLayout.signature === "string" &&
    typeof plan.bufferLayout.signature === "string";
}

function kernelPlanRejectionReason(plan: unknown) {
  if (!isRecord(plan)) return "value is not a KernelPlan";
  if (plan.kind !== "native-module-kernel-plan") return `kind is ${String(plan.kind || "missing")}`;
  if (plan.version !== 1) return `version is ${String(plan.version || "missing")}`;
  if (plan.nativePath !== "device-program") return `nativePath is ${String(plan.nativePath || "missing")}`;
  if (typeof plan.signature !== "string" || plan.signature.length === 0) return "signature is missing";
  if (!isShapeArray(plan.inputShape)) return "inputShape is missing or invalid";
  if (!isShapeArray(plan.outputShape)) return "outputShape is missing or invalid";
  if (!Number.isSafeInteger(plan.inputLen) || plan.inputLen <= 0) return "inputLen is missing or invalid";
  if (!Number.isSafeInteger(plan.outputLen) || plan.outputLen <= 0) return "outputLen is missing or invalid";
  if (!kernelPlanHasLayoutEvidence(plan)) return "layout evidence is missing";
  if (!Array.isArray(plan.ops) || plan.ops.length === 0) return "ops are missing";
  if (!plan.ops.every(kernelPlanOpHasEvidence)) return "one or more ops are missing KernelPlan evidence";
  if (!Number.isSafeInteger(plan.dispatchCount) || plan.dispatchCount < 0) return "dispatchCount is missing or invalid";
  if (!Number.isSafeInteger(plan.descriptorCount) || plan.descriptorCount < 0) return "descriptorCount is missing or invalid";
  const expected = kernelPlanSignature(plan);
  if (plan.signature !== expected) return "signature does not match KernelPlan schedule evidence";
  return null;
}

function acceptsKernelPlan(plan: unknown): plan is KernelPlan {
  return kernelPlanRejectionReason(plan) === null;
}

function requireKernelPlan(plan: unknown): KernelPlan {
  if (acceptsKernelPlan(plan)) return plan;
  throw new Error(`KernelPlan is not executable evidence: ${kernelPlanRejectionReason(plan)}`);
}

const assertKernelPlan = requireKernelPlan;
const assert_kernel_plan = requireKernelPlan;

function matchesKernelPlanSignature(plan: unknown, signature: unknown): boolean {
  return acceptsKernelPlan(plan) &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}

export {
  acceptsKernelPlan,
  assert_kernel_plan,
  assertKernelPlan,
  freezeKernelShapeConstraints,
  freezeKernelParameterLayout,
  freezeKernelMemoryLayout,
  freezeKernelPlan,
  freezePublicKernelPlan,
  kernelizeTensorProgramIr,
  programShapeConstraintsFromKernelPlan,
  kernelPlanInputShape,
  kernelPlanOutputShape,
  kernelPlanInputLen,
  kernelPlanOutputLen,
  programMemoryLayoutFromKernelPlan,
  programParameterLayoutFromKernelPlan,
  kernelPlanSignature,
  matchesKernelPlanSignature,
  memoryLayoutSignature,
  parameterLayoutSignature,
  requireKernelPlan,
  bufferLayoutSignature,
};

export const kernelPlanManifest = Object.freeze({
  kind: "zgml-kernel-plan",
  ...tsRuntimeManifestPolicy("src/ts/runtime/kernel_plan.ts", "Lazy Tensor IR -> KernelPlan -> Program"),
});
