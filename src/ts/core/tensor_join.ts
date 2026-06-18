"use strict";

import {
  normalizeDim,
  rowMajorStrides,
  shapeProduct,
} from "./shape.js";
import {
  isGradEnabled,
} from "./grad_mode.js";

type TensorJoinConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  prev?: readonly TensorJoinTensor[];
}>;
type TensorJoinTensor = {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  requiresGrad: boolean;
  _backward?: (grad: Float32Array | null) => void;
};
type TensorConstructor = new (
  data: Float32Array,
  shape: readonly number[],
  options?: TensorJoinConstructOptions,
) => TensorJoinTensor;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type AddTensorGradCallback = BivariantCallback<[tensor: TensorJoinTensor, grad: Float32Array], void>;

export type TensorJoinHelpersOptions = Readonly<{
  Tensor: TensorConstructor;
  addTensorGrad: AddTensorGradCallback;
  isGradEnabled?: () => boolean;
}>;

export function createTensorJoinHelpers(options: TensorJoinHelpersOptions) {
  const TensorClass = options.Tensor;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : isGradEnabled;
  if (typeof TensorClass !== "function" || typeof addTensorGrad !== "function") {
    throw new Error("tensor join helpers require Tensor and addTensorGrad");
  }
  const TensorCtor = TensorClass as TensorConstructor;

  function requireTensorList(tensors: unknown, label: string) {
    if (!Array.isArray(tensors) || tensors.length === 0) {
      throw new Error(`${label} requires a non-empty tensor array`);
    }
    for (let i = 0; i < tensors.length; i += 1) {
      if (!(tensors[i] instanceof TensorCtor)) throw new Error(`${label} input ${i} must be a Tensor`);
    }
    return tensors as TensorJoinTensor[];
  }

  function sameShape(a: readonly number[], b: readonly number[]) {
    return a.length === b.length && a.every((dim, index) => dim === b[index]);
  }

  function shapeText(shape: readonly number[]) {
    return `[${shape.join(",")}]`;
  }

  function flatToOffset(flat: number, inShape: readonly number[], outShape: readonly number[], axis: number, axisOffset: number) {
    const inStrides = rowMajorStrides(inShape);
    const outStrides = rowMajorStrides(outShape);
    let outIndex = 0;
    for (let dim = 0; dim < inShape.length; dim += 1) {
      const coord = Math.floor(flat / inStrides[dim]) % inShape[dim];
      outIndex += (dim === axis ? coord + axisOffset : coord) * outStrides[dim];
    }
    return outIndex;
  }

  function cat(tensorValues: unknown, dim = 0) {
    const tensors = requireTensorList(tensorValues, "cat");
    const rank = tensors[0].shape.length;
    const axis = normalizeDim(dim, rank, "cat");
    const outShape = tensors[0].shape.slice();
    outShape[axis] = 0;
    for (let i = 0; i < tensors.length; i += 1) {
      const shape = tensors[i].shape;
      if (shape.length !== rank) throw new Error(`cat input ${i} rank must be ${rank}, got ${shape.length}`);
      for (let d = 0; d < rank; d += 1) {
        if (d !== axis && shape[d] !== outShape[d]) {
          throw new Error(`cat input ${i} shape ${shapeText(shape)} is incompatible with ${shapeText(tensors[0].shape)} along dim ${dim}`);
        }
      }
      outShape[axis] += shape[axis];
    }

    const outData = new Float32Array(shapeProduct(outShape));
    const maps: Uint32Array[] = [];
    let axisOffset = 0;
    for (const tensor of tensors) {
      const map = new Uint32Array(tensor.length);
      for (let flat = 0; flat < tensor.length; flat += 1) {
        const outIndex = flatToOffset(flat, tensor.shape, outShape, axis, axisOffset);
        map[flat] = outIndex;
        outData[outIndex] = tensor.data[flat];
      }
      maps.push(map);
      axisOffset += tensor.shape[axis];
    }

    const prev = gradModeEnabled() ? tensors.filter((tensor) => tensor.requiresGrad) : [];
    const out = new TensorCtor(outData, outShape, {
      requiresGrad: prev.length > 0,
      prev,
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      for (let t = 0; t < tensors.length; t += 1) {
        const tensor = tensors[t];
        if (!tensor.requiresGrad) continue;
        const inGrad = new Float32Array(tensor.length);
        const map = maps[t];
        for (let i = 0; i < tensor.length; i += 1) inGrad[i] += grad[map[i]];
        addTensorGrad(tensor, inGrad);
      }
    };
    return out;
  }

  function normalizeStackDim(dim: number, rank: number) {
    if (!Number.isSafeInteger(dim)) throw new Error(`stack dim must be an integer, got ${dim}`);
    const outRank = rank + 1;
    const axis = dim < 0 ? outRank + dim : dim;
    if (axis < 0 || axis >= outRank) throw new Error(`stack dim ${dim} is out of range for output rank ${outRank}`);
    return axis;
  }

  function stack(tensorValues: unknown, dim = 0) {
    const tensors = requireTensorList(tensorValues, "stack");
    const baseShape = tensors[0].shape;
    for (let i = 1; i < tensors.length; i += 1) {
      if (!sameShape(tensors[i].shape, baseShape)) {
        throw new Error(`stack input ${i} shape ${shapeText(tensors[i].shape)} must match ${shapeText(baseShape)}`);
      }
    }
    const axis = normalizeStackDim(dim, baseShape.length);
    const outShape = [...baseShape.slice(0, axis), tensors.length, ...baseShape.slice(axis)];
    const outStrides = rowMajorStrides(outShape);
    const inStrides = rowMajorStrides(baseShape);
    const outData = new Float32Array(shapeProduct(outShape));
    const maps: Uint32Array[] = [];
    for (let t = 0; t < tensors.length; t += 1) {
      const tensor = tensors[t];
      const map = new Uint32Array(tensor.length);
      for (let flat = 0; flat < tensor.length; flat += 1) {
        let outIndex = t * outStrides[axis];
        for (let dimIndex = 0; dimIndex < baseShape.length; dimIndex += 1) {
          const coord = Math.floor(flat / inStrides[dimIndex]) % baseShape[dimIndex];
          const outDim = dimIndex < axis ? dimIndex : dimIndex + 1;
          outIndex += coord * outStrides[outDim];
        }
        map[flat] = outIndex;
        outData[outIndex] = tensor.data[flat];
      }
      maps.push(map);
    }

    const prev = gradModeEnabled() ? tensors.filter((tensor) => tensor.requiresGrad) : [];
    const out = new TensorCtor(outData, outShape, {
      requiresGrad: prev.length > 0,
      prev,
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      for (let t = 0; t < tensors.length; t += 1) {
        const tensor = tensors[t];
        if (!tensor.requiresGrad) continue;
        const inGrad = new Float32Array(tensor.length);
        const map = maps[t];
        for (let i = 0; i < tensor.length; i += 1) inGrad[i] += grad[map[i]];
        addTensorGrad(tensor, inGrad);
      }
    };
    return out;
  }

  return Object.freeze({ cat, stack });
}
