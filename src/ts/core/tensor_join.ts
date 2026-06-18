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

  function parseEinsumEquation(equation: unknown, operandCount: number) {
    if (typeof equation !== "string" || equation.trim().length === 0) {
      throw new Error("einsum equation must be a non-empty string");
    }
    const compact = equation.replace(/\s+/g, "");
    if (compact.includes("...")) {
      throw new Error("einsum ellipsis is not supported yet");
    }
    const parts = compact.split("->");
    if (parts.length > 2) throw new Error(`invalid einsum equation: ${equation}`);
    const inputSpecs = parts[0].split(",");
    if (inputSpecs.length !== operandCount) {
      throw new Error(`einsum equation expects ${inputSpecs.length} operand(s), got ${operandCount}`);
    }
    for (const spec of inputSpecs) {
      if (!/^[A-Za-z]*$/.test(spec)) throw new Error(`invalid einsum input subscript: ${spec}`);
    }
    let outputSpec = parts[1];
    if (outputSpec !== undefined) {
      if (!/^[A-Za-z]*$/.test(outputSpec)) throw new Error(`invalid einsum output subscript: ${outputSpec}`);
      const seen = new Set();
      for (const label of outputSpec) {
        if (seen.has(label)) throw new Error(`einsum output label ${label} appears more than once`);
        seen.add(label);
      }
    } else {
      const counts = new Map<string, number>();
      for (const spec of inputSpecs) {
        for (const label of spec) counts.set(label, (counts.get(label) ?? 0) + 1);
      }
      outputSpec = [...counts.entries()]
        .filter((entry) => entry[1] === 1)
        .map((entry) => entry[0])
        .sort()
        .join("");
    }
    return { inputSpecs, outputSpec };
  }

  function coordsForFlat(flat: number, shape: readonly number[], strides: readonly number[]) {
    const coords = new Array(shape.length);
    for (let dim = 0; dim < shape.length; dim += 1) coords[dim] = Math.floor(flat / strides[dim]) % shape[dim];
    return coords;
  }

  function einsum(equation: unknown, tensorValues: unknown, ...moreTensorValues: unknown[]) {
    const operands = Array.isArray(tensorValues) && moreTensorValues.length === 0
      ? requireTensorList(tensorValues, "einsum")
      : requireTensorList([tensorValues, ...moreTensorValues], "einsum");
    const { inputSpecs, outputSpec } = parseEinsumEquation(equation, operands.length);
    const labelSizes = new Map<string, number>();
    const labelOrder: string[] = [];
    const operandStrides = operands.map((operand) => rowMajorStrides(operand.shape));
    for (let operandIndex = 0; operandIndex < operands.length; operandIndex += 1) {
      const operand = operands[operandIndex];
      const spec = inputSpecs[operandIndex];
      if (spec.length !== operand.shape.length) {
        throw new Error(`einsum operand ${operandIndex} rank ${operand.shape.length} does not match subscript ${spec}`);
      }
      for (let dim = 0; dim < spec.length; dim += 1) {
        const label = spec[dim];
        const size = operand.shape[dim];
        if (!labelSizes.has(label)) {
          labelSizes.set(label, size);
          labelOrder.push(label);
        } else if (labelSizes.get(label) !== size) {
          throw new Error(`einsum label ${label} has inconsistent dimensions ${labelSizes.get(label)} and ${size}`);
        }
      }
    }
    for (const label of outputSpec) {
      if (!labelSizes.has(label)) throw new Error(`einsum output label ${label} does not appear in any input`);
    }
    const outputShape = outputSpec.length === 0 ? [1] : [...outputSpec].map((label) => labelSizes.get(label)!);
    const outputStrides = rowMajorStrides(outputShape);
    const outData = new Float32Array(shapeProduct(outputShape));
    const assignment = new Map<string, number>();

    function operandFlatIndex(operandIndex: number) {
      const spec = inputSpecs[operandIndex];
      const strides = operandStrides[operandIndex];
      let index = 0;
      for (let dim = 0; dim < spec.length; dim += 1) index += assignment.get(spec[dim])! * strides[dim];
      return index;
    }

    function outputFlatIndex() {
      if (outputSpec.length === 0) return 0;
      let index = 0;
      for (let dim = 0; dim < outputSpec.length; dim += 1) index += assignment.get(outputSpec[dim])! * outputStrides[dim];
      return index;
    }

    function visit(labelIndex: number, callback: () => void) {
      if (labelIndex === labelOrder.length) {
        callback();
        return;
      }
      const label = labelOrder[labelIndex];
      const size = labelSizes.get(label)!;
      for (let value = 0; value < size; value += 1) {
        assignment.set(label, value);
        visit(labelIndex + 1, callback);
      }
    }

    visit(0, () => {
      let product = 1;
      for (let operandIndex = 0; operandIndex < operands.length; operandIndex += 1) {
        product *= operands[operandIndex].data[operandFlatIndex(operandIndex)];
      }
      outData[outputFlatIndex()] += product;
    });

    const prev = gradModeEnabled() ? operands.filter((tensor) => tensor.requiresGrad) : [];
    const out = new TensorCtor(outData, outputShape, {
      requiresGrad: prev.length > 0,
      prev,
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const operandGrads = operands.map((operand) => operand.requiresGrad ? new Float32Array(operand.length) : null);
      visit(0, () => {
        const outGrad = grad[outputFlatIndex()];
        if (outGrad === 0) return;
        for (let target = 0; target < operands.length; target += 1) {
          const targetGrad = operandGrads[target];
          if (!targetGrad) continue;
          let product = outGrad;
          for (let operandIndex = 0; operandIndex < operands.length; operandIndex += 1) {
            if (operandIndex !== target) product *= operands[operandIndex].data[operandFlatIndex(operandIndex)];
          }
          targetGrad[operandFlatIndex(target)] += product;
        }
      });
      for (let operandIndex = 0; operandIndex < operands.length; operandIndex += 1) {
        const targetGrad = operandGrads[operandIndex];
        if (targetGrad) addTensorGrad(operands[operandIndex], targetGrad);
      }
    };
    return out;
  }

  return Object.freeze({ cat, stack, einsum });
}
