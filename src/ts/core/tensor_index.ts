"use strict";

import {
  rowMajorStrides,
} from "./shape.js";
import type {
  TensorNestedArray,
} from "../public_api.js";

type TensorLike = {
  data: ArrayLike<number> & { [index: number]: number };
  shape: readonly number[];
};

type MutableTensorLike = TensorLike & {
  data: TensorLike["data"] & { [index: number]: number };
};

export type TensorIndexSurfaceHelpersOptions<TTensor> = Readonly<{
  tensorIndexHelpers: {
    get: (tensor: TTensor, indices: number | readonly number[]) => number;
    setFlexible: (tensor: TTensor, indicesOrFirst: number | readonly number[], rest: readonly number[]) => TTensor;
    toArray: (tensor: TTensor) => TensorNestedArray;
    tolist: (tensor: TTensor) => TensorNestedArray;
    to_list: (tensor: TTensor) => TensorNestedArray;
  };
}>;

export function createTensorIndexHelpers() {
  function normalizeCoordinate(index: number, dim: number, label: string) {
    if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be a safe integer, got ${index}`);
    const resolved = index < 0 ? dim + index : index;
    if (resolved < 0 || resolved >= dim) {
      throw new Error(`${label} index ${index} is out of range for dimension length ${dim}`);
    }
    return resolved;
  }

  function normalizeCoordinates(indices: number | readonly number[], rank: number, label: string) {
    const coords = Array.isArray(indices) ? indices : [indices];
    if (coords.length !== rank) {
      throw new Error(`${label} expects ${rank} index${rank === 1 ? "" : "es"}, got ${coords.length}`);
    }
    return coords;
  }

  function flatIndex(tensor: TensorLike, indices: number | readonly number[], label: string) {
    const coords = normalizeCoordinates(indices, tensor.shape.length, label);
    const strides = rowMajorStrides(tensor.shape);
    let offset = 0;
    for (let dim = 0; dim < tensor.shape.length; dim += 1) {
      offset += normalizeCoordinate(coords[dim], tensor.shape[dim], `${label} dim ${dim}`) * strides[dim];
    }
    return offset;
  }

  function get(tensor: TensorLike, indices: number | readonly number[]) {
    return tensor.data[flatIndex(tensor, indices, "Tensor.get")];
  }

  function set(tensor: MutableTensorLike, indices: number | readonly number[], value: number) {
    if (!Number.isFinite(value)) throw new Error(`Tensor.set value must be finite, got ${value}`);
    tensor.data[flatIndex(tensor, indices, "Tensor.set")] = value;
    return tensor;
  }

  function setFlexible(tensor: MutableTensorLike, indicesOrFirst: number | readonly number[], rest: readonly number[]) {
    let indices: number | readonly number[] = indicesOrFirst;
    let value: number = rest[0];
    if (typeof indicesOrFirst === "number") {
      if (rest.length === 0) throw new Error("Tensor.set requires indices and a value");
      value = rest[rest.length - 1];
      indices = [indicesOrFirst, ...rest.slice(0, -1)];
    } else if (rest.length !== 1) {
      throw new Error("Tensor.set with an index array requires exactly one value");
    }
    return set(tensor, indices, value);
  }

  function toArray(tensor: TensorLike): TensorNestedArray {
    const strides = rowMajorStrides(tensor.shape);
    function build(dim: number, offset: number): TensorNestedArray {
      const len = tensor.shape[dim];
      const out: Array<number | TensorNestedArray> = new Array(len);
      if (dim === tensor.shape.length - 1) {
        for (let i = 0; i < len; i += 1) out[i] = tensor.data[offset + i * strides[dim]];
        return out;
      }
      for (let i = 0; i < len; i += 1) out[i] = build(dim + 1, offset + i * strides[dim]);
      return out;
    }
    return build(0, 0);
  }

  function tolist(tensor: TensorLike): TensorNestedArray {
    return toArray(tensor);
  }

  function to_list(tensor: TensorLike): TensorNestedArray {
    return toArray(tensor);
  }

  return {
    flatIndex,
    get,
    set,
    setFlexible,
    toArray,
    tolist,
    to_list,
  };
}

export function createTensorIndexSurfaceHelpers<TTensor>(options: TensorIndexSurfaceHelpersOptions<TTensor>) {
  const { tensorIndexHelpers } = options;

  return Object.freeze({
    get: (tensor: TTensor, indices: readonly (number | readonly number[])[]) => {
      const normalized = indices.length === 1 ? indices[0] : indices as readonly number[];
      return tensorIndexHelpers.get(tensor, normalized);
    },
    set: (tensor: TTensor, indicesOrFirst: number | readonly number[], rest: readonly number[]) => (
      tensorIndexHelpers.setFlexible(tensor, indicesOrFirst, rest)
    ),
    toArray: (tensor: TTensor) => tensorIndexHelpers.toArray(tensor),
    tolist: (tensor: TTensor) => tensorIndexHelpers.tolist(tensor),
    to_list: (tensor: TTensor) => tensorIndexHelpers.to_list(tensor),
  });
}
