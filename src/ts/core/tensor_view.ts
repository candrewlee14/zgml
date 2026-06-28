"use strict";

import {
  broadcastPlan,
  normalizeDim,
  normalizeFactoryShape,
  normalizeInsertDim,
  normalizeViewShape,
  rowMajorStrides,
  shapeProduct,
} from "./shape.js";
import {
  isGradEnabled,
} from "./grad_mode.js";

type AnyRecord = Record<string, any>;
type TensorViewConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  prev?: readonly TensorViewTensor[];
}>;
type TensorViewTensor = AnyRecord & {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  requiresGrad: boolean;
  _backward?: (grad: Float32Array | null) => void;
};
type TensorConstructor = new (
  data: Float32Array,
  shape: readonly number[],
  options?: TensorViewConstructOptions,
) => TensorViewTensor;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type AddTensorGradCallback = BivariantCallback<[tensor: TensorViewTensor, grad: Float32Array], void>;
type NativePermuteInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{
    outputShape: Uint32Array;
    inputStrides: Uint32Array;
    axes: Uint32Array;
  }>,
) => Float32Array;
type NativeTakeInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
) => Float32Array;
type NativeIndexSelectInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
  options: Readonly<{ outer: number; axisLen: number; inner: number }>,
) => Float32Array;
type NativeGatherInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
  options: Readonly<{
    outputShape: Uint32Array;
    inputStrides: Uint32Array;
    axis: number;
    axisLen: number;
  }>,
) => Float32Array;
type NativeFlipInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{
    shape: Uint32Array;
    strides: Uint32Array;
    axes: Uint32Array;
  }>,
) => Float32Array;

export type TensorViewHelpersOptions = Readonly<{
  Tensor: TensorConstructor;
  addTensorGrad: AddTensorGradCallback;
  isGradEnabled?: () => boolean;
  nativePermuteInto?: NativePermuteInto;
  nativeTakeInto?: NativeTakeInto;
  nativeIndexSelectInto?: NativeIndexSelectInto;
  nativeGatherInto?: NativeGatherInto;
  nativeFlipInto?: NativeFlipInto;
}>;

export type TensorViewSurfaceHelpersOptions = Readonly<{
  tensorViewHelpers: AnyRecord;
}>;

export function createTensorViewHelpers(options: TensorViewHelpersOptions) {
  const TensorClass = options.Tensor;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : isGradEnabled;
  const nativePermuteInto = typeof options.nativePermuteInto === "function" ? options.nativePermuteInto : null;
  const nativeTakeInto = typeof options.nativeTakeInto === "function" ? options.nativeTakeInto : null;
  const nativeIndexSelectInto = typeof options.nativeIndexSelectInto === "function" ? options.nativeIndexSelectInto : null;
  const nativeGatherInto = typeof options.nativeGatherInto === "function" ? options.nativeGatherInto : null;
  const nativeFlipInto = typeof options.nativeFlipInto === "function" ? options.nativeFlipInto : null;
  if (typeof TensorClass !== "function" || typeof addTensorGrad !== "function") {
    throw new Error("tensor view helpers require Tensor and addTensorGrad");
  }
  const TensorCtor = TensorClass as TensorConstructor;

  function u32Array(values: readonly number[], label: string) {
    const out = new Uint32Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
        throw new Error(`${label} entry ${i} must be a uint32, got ${value}`);
      }
      out[i] = value;
    }
    return out;
  }

  function nativePermuteData(
    output: Float32Array,
    tensor: TensorViewTensor,
    outputShape: readonly number[],
    inputStrides: readonly number[],
    axes: readonly number[],
  ) {
    if (nativePermuteInto === null) return false;
    if (outputShape.length !== inputStrides.length || outputShape.length !== axes.length) return false;
    nativePermuteInto(output, tensor, {
      outputShape: u32Array(outputShape, "native permute outputShape"),
      inputStrides: u32Array(inputStrides, "native permute inputStrides"),
      axes: u32Array(axes, "native permute axes"),
    });
    return true;
  }

  function normalizeAxisIndex(index: number, dim: number, label: string) {
    if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be a safe integer, got ${index}`);
    const resolved = index < 0 ? dim + index : index;
    if (resolved < 0 || resolved >= dim) {
      throw new Error(`${label} index ${index} is out of range for dimension length ${dim}`);
    }
    return resolved;
  }

  function normalizeSliceBound(value: number | null | undefined, dim: number, fallback: number, label: string) {
    if (value === undefined || value === null) return fallback;
    if (!Number.isSafeInteger(value)) throw new Error(`${label} must be a safe integer, got ${value}`);
    const resolved = value < 0 ? dim + value : value;
    return Math.max(0, Math.min(dim, resolved));
  }

  function gatherAxis(tensor: TensorViewTensor, axis: number, positions: number[], keepAxis: boolean, label: string) {
    if (positions.length === 0) throw new Error(`${label} would produce an empty tensor`);
    const rank = tensor.shape.length;
    const outShape = keepAxis
      ? tensor.shape.map((dim: number, index: number) => index === axis ? positions.length : dim)
      : tensor.shape.filter((_: number, index: number) => index !== axis);
    if (outShape.length === 0) outShape.push(1);

    const inStrides = rowMajorStrides(tensor.shape);
    const outStrides = rowMajorStrides(outShape);
    const outLen = shapeProduct(outShape);
    const outData = new Float32Array(outLen);
    const outToIn = new Uint32Array(outLen);

    for (let flat = 0; flat < outLen; flat += 1) {
      let inputIndex = 0;
      let outDim = 0;
      for (let dim = 0; dim < rank; dim += 1) {
        if (dim === axis) {
          const positionIndex = keepAxis ? Math.floor(flat / outStrides[outDim]) % outShape[outDim] : 0;
          inputIndex += positions[positionIndex] * inStrides[dim];
          if (keepAxis) outDim += 1;
        } else {
          const coord = Math.floor(flat / outStrides[outDim]) % outShape[outDim];
          inputIndex += coord * inStrides[dim];
          outDim += 1;
        }
      }
      outToIn[flat] = inputIndex;
      outData[flat] = tensor.data[inputIndex];
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function clone(tensor: TensorViewTensor) {
    const out = new TensorCtor(new Float32Array(tensor.data), tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      addTensorGrad(tensor, grad);
    };
    return out;
  }

  function detach(tensor: TensorViewTensor) {
    return new TensorCtor(new Float32Array(tensor.data), tensor.shape);
  }

  function reshape(tensor: TensorViewTensor, shape: unknown, name = "reshape") {
    const outShape = normalizeViewShape(shape as number | readonly number[], tensor.length, name);
    const out = new TensorCtor(tensor.data, outShape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      addTensorGrad(tensor, grad);
    };
    return out;
  }

  function broadcastTo(tensor: TensorViewTensor, shape: unknown) {
    const target = normalizeFactoryShape(shape as number | readonly number[], "broadcastTo shape").shape;
    const plan = broadcastPlan(tensor.shape, target, "broadcastTo");
    if (plan.shape.length !== target.length || !plan.shape.every((dim, index) => dim === target[index])) {
      throw new Error(`tensor broadcastTo shape mismatch: [${tensor.shape.join(",")}] cannot broadcast to [${target.join(",")}]`);
    }

    const outData = new Float32Array(plan.lhsIndex.length);
    for (let i = 0; i < outData.length; i += 1) outData[i] = tensor.data[plan.lhsIndex[i]];
    const out = new TensorCtor(outData, target, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[plan.lhsIndex[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function expand(tensor: TensorViewTensor, shape: unknown) {
    return broadcastTo(tensor, shape);
  }

  function normalizeRepeats(repeats: unknown, rank: number, label: string) {
    const dims = Number.isSafeInteger(repeats) ? [repeats as number] : repeats;
    if (!Array.isArray(dims) || dims.length === 0) {
      throw new Error(`${label} repeats must be a positive integer or non-empty array`);
    }
    if (dims.length < rank) {
      throw new Error(`${label} repeats length ${dims.length} must be >= tensor rank ${rank}`);
    }
    return dims.map((dim) => {
      if (!Number.isSafeInteger(dim) || dim <= 0) {
        throw new Error(`${label} repeats must be positive safe integers, got ${dim}`);
      }
      return dim;
    });
  }

  function repeat(tensor: TensorViewTensor, repeats: unknown) {
    const repeatShape = normalizeRepeats(repeats, tensor.shape.length, "repeat");
    const paddedShape = [
      ...Array.from({ length: repeatShape.length - tensor.shape.length }, () => 1),
      ...tensor.shape,
    ];
    const outShape = repeatShape.map((repeatDim, index) => paddedShape[index] * repeatDim);
    const inStrides = rowMajorStrides(paddedShape);
    const outStrides = rowMajorStrides(outShape);
    const outLen = shapeProduct(outShape);
    const outData = new Float32Array(outLen);
    const outToIn = new Uint32Array(outLen);

    for (let flat = 0; flat < outLen; flat += 1) {
      let inIndex = 0;
      for (let dim = 0; dim < outShape.length; dim += 1) {
        const coord = Math.floor(flat / outStrides[dim]) % outShape[dim];
        inIndex += (coord % paddedShape[dim]) * inStrides[dim];
      }
      outToIn[flat] = inIndex;
      outData[flat] = tensor.data[inIndex];
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function tile(tensor: TensorViewTensor, repeats: unknown) {
    return repeat(tensor, repeats);
  }

  function flatten(tensor: TensorViewTensor, startDim = 0, endDim = -1) {
    const rank = tensor.shape.length;
    const start = normalizeDim(startDim, rank, "flatten");
    const end = normalizeDim(endDim, rank, "flatten");
    if (end < start) throw new Error(`flatten end dim ${endDim} must be >= start dim ${startDim}`);
    let flattened = 1;
    for (let i = start; i <= end; i += 1) flattened *= tensor.shape[i];
    return reshape(tensor, [
      ...tensor.shape.slice(0, start),
      flattened,
      ...tensor.shape.slice(end + 1),
    ], "flatten");
  }

  function squeeze(tensor: TensorViewTensor, dim: number | null = null) {
    if (dim === undefined || dim === null) {
      const outShape = tensor.shape.filter((size: number) => size !== 1);
      if (outShape.length === 0) outShape.push(1);
      return reshape(tensor, outShape, "squeeze");
    }
    const axis = normalizeDim(dim, tensor.shape.length, "squeeze");
    if (tensor.shape[axis] !== 1) return reshape(tensor, tensor.shape, "squeeze");
    const outShape = tensor.shape.filter((_: number, index: number) => index !== axis);
    if (outShape.length === 0) outShape.push(1);
    return reshape(tensor, outShape, "squeeze");
  }

  function unsqueeze(tensor: TensorViewTensor, dim: number) {
    const axis = normalizeInsertDim(dim, tensor.shape.length, "unsqueeze");
    const outShape = tensor.shape.slice();
    outShape.splice(axis, 0, 1);
    return reshape(tensor, outShape, "unsqueeze");
  }

  function transpose(tensor: TensorViewTensor, dim0 = 0, dim1 = 1) {
    const rank = tensor.shape.length;
    if (rank < 2) throw new Error("transpose requires a tensor with rank >= 2");
    const axis0 = normalizeDim(dim0, rank, "transpose");
    const axis1 = normalizeDim(dim1, rank, "transpose");
    if (axis0 === axis1) return reshape(tensor, tensor.shape, "transpose");

    const outShape = tensor.shape.slice();
    const tmp = outShape[axis0];
    outShape[axis0] = outShape[axis1];
    outShape[axis1] = tmp;

    const inStrides = rowMajorStrides(tensor.shape);
    const outStrides = rowMajorStrides(outShape);
    const outData = new Float32Array(tensor.length);
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    const outToIn = needsGrad ? new Uint32Array(tensor.length) : null;
    const axes = Array.from({ length: rank }, (_value, dim) => dim === axis0 ? axis1 : dim === axis1 ? axis0 : dim);
    const wroteNative = nativePermuteData(outData, tensor, outShape, inStrides, axes);
    if (!wroteNative || outToIn !== null) {
      for (let flat = 0; flat < outData.length; flat += 1) {
        let inIndex = 0;
        for (let dim = 0; dim < rank; dim += 1) {
          const coord = Math.floor(flat / outStrides[dim]) % outShape[dim];
          inIndex += coord * inStrides[axes[dim]];
        }
        if (outToIn !== null) outToIn[flat] = inIndex;
        if (!wroteNative) outData[flat] = tensor.data[inIndex];
      }
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad || outToIn === null) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function permute(tensor: TensorViewTensor, dims: readonly number[]) {
    const rank = tensor.shape.length;
    if (!Array.isArray(dims) || dims.length !== rank) {
      throw new Error(`permute dims length ${Array.isArray(dims) ? dims.length : "<non-array>"} must match tensor rank ${rank}`);
    }
    const seen = new Set<number>();
    const axes = dims.map((dim) => {
      const axis = normalizeDim(dim, rank, "permute");
      if (seen.has(axis)) throw new Error(`permute dims must be a permutation of tensor axes; duplicate axis ${axis}`);
      seen.add(axis);
      return axis;
    });
    const outShape = axes.map((axis) => tensor.shape[axis]);
    const inStrides = rowMajorStrides(tensor.shape);
    const outStrides = rowMajorStrides(outShape);
    const outData = new Float32Array(tensor.length);
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    const outToIn = needsGrad ? new Uint32Array(tensor.length) : null;
    const wroteNative = nativePermuteData(outData, tensor, outShape, inStrides, axes);

    if (!wroteNative || outToIn !== null) {
      for (let flat = 0; flat < outData.length; flat += 1) {
        let inIndex = 0;
        for (let dim = 0; dim < rank; dim += 1) {
          const coord = Math.floor(flat / outStrides[dim]) % outShape[dim];
          inIndex += coord * inStrides[axes[dim]];
        }
        if (outToIn !== null) outToIn[flat] = inIndex;
        if (!wroteNative) outData[flat] = tensor.data[inIndex];
      }
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad || outToIn === null) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function flip(tensor: TensorViewTensor, dims: readonly number[]) {
    const rank = tensor.shape.length;
    if (!Array.isArray(dims) || dims.length === 0) {
      throw new Error("flip dims must be a non-empty axis array");
    }
    const axes = new Set<number>();
    for (const dim of dims) {
      const axis = normalizeDim(dim, rank, "flip");
      if (axes.has(axis)) throw new Error(`flip dims must be unique; duplicate axis ${axis}`);
      axes.add(axis);
    }
    const inStrides = rowMajorStrides(tensor.shape);
    const outData = new Float32Array(tensor.length);
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    const outToIn = needsGrad ? new Uint32Array(tensor.length) : null;

    if (!needsGrad && nativeFlipInto !== null) {
      nativeFlipInto(outData, tensor, {
        shape: u32Array(tensor.shape, "flip shape"),
        strides: u32Array(inStrides, "flip strides"),
        axes: u32Array(Array.from(axes), "flip axes"),
      });
    } else {
      for (let flat = 0; flat < outData.length; flat += 1) {
        let inIndex = 0;
        for (let dim = 0; dim < rank; dim += 1) {
          const coord = Math.floor(flat / inStrides[dim]) % tensor.shape[dim];
          const inputCoord = axes.has(dim) ? tensor.shape[dim] - 1 - coord : coord;
          inIndex += inputCoord * inStrides[dim];
        }
        if (outToIn !== null) outToIn[flat] = inIndex;
        outData[flat] = tensor.data[inIndex];
      }
    }

    const out = new TensorCtor(outData, tensor.shape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad || outToIn === null) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function normalizeRollArray(value: unknown, label: string) {
    const values = Number.isSafeInteger(value) ? [value as number] : value;
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`${label} must be a safe integer or non-empty integer array`);
    }
    return values.map((item) => {
      if (!Number.isSafeInteger(item)) throw new Error(`${label} entries must be safe integers, got ${item}`);
      return item;
    });
  }

  function roll(tensor: TensorViewTensor, shifts: unknown, dims: unknown = null) {
    const rank = tensor.shape.length;
    const shiftValues = normalizeRollArray(shifts, "roll shifts");
    const dimValues = dims === undefined || dims === null ? null : normalizeRollArray(dims, "roll dims");
    const axes = dimValues === null ? null : dimValues.map((dim) => normalizeDim(dim, rank, "roll"));
    if (axes === null && shiftValues.length !== 1) {
      throw new Error("roll shifts must have length 1 when dims is omitted");
    }
    if (axes !== null && shiftValues.length !== 1 && shiftValues.length !== axes.length) {
      throw new Error(`roll shifts length ${shiftValues.length} must be 1 or match dims length ${axes.length}`);
    }
    const axisShifts = new Map<number, number>();
    if (axes === null) {
      axisShifts.set(-1, shiftValues[0]);
    } else {
      for (let index = 0; index < axes.length; index += 1) {
        const axis = axes[index];
        if (axisShifts.has(axis)) throw new Error(`roll dims must be unique; duplicate axis ${axis}`);
        axisShifts.set(axis, shiftValues.length === 1 ? shiftValues[0] : shiftValues[index]);
      }
    }

    const strides = rowMajorStrides(tensor.shape);
    const outData = new Float32Array(tensor.length);
    const outToIn = new Uint32Array(tensor.length);

    for (let flat = 0; flat < outData.length; flat += 1) {
      let inIndex = 0;
      if (axes === null) {
        const shift = ((axisShifts.get(-1) || 0) % tensor.length + tensor.length) % tensor.length;
        inIndex = (flat - shift + tensor.length) % tensor.length;
      } else {
        for (let dim = 0; dim < rank; dim += 1) {
          const coord = Math.floor(flat / strides[dim]) % tensor.shape[dim];
          const dimLen = tensor.shape[dim];
          const rawShift = axisShifts.get(dim) || 0;
          const shift = ((rawShift % dimLen) + dimLen) % dimLen;
          const inputCoord = axisShifts.has(dim) ? (coord - shift + dimLen) % dimLen : coord;
          inIndex += inputCoord * strides[dim];
        }
      }
      outToIn[flat] = inIndex;
      outData[flat] = tensor.data[inIndex];
    }

    const out = new TensorCtor(outData, tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function select(tensor: TensorViewTensor, dim: number, index: number) {
    const axis = normalizeDim(dim, tensor.shape.length, "select");
    const resolved = normalizeAxisIndex(index, tensor.shape[axis], "select");
    return gatherAxis(tensor, axis, [resolved], false, "select");
  }

  function narrow(tensor: TensorViewTensor, dim: number, start: number, length: number) {
    const axis = normalizeDim(dim, tensor.shape.length, "narrow");
    if (!Number.isSafeInteger(start)) throw new Error(`narrow start must be a safe integer, got ${start}`);
    if (!Number.isSafeInteger(length) || length <= 0) throw new Error(`narrow length must be a positive safe integer, got ${length}`);
    const resolvedStart = normalizeAxisIndex(start, tensor.shape[axis], "narrow");
    if (resolvedStart + length > tensor.shape[axis]) {
      throw new Error(`narrow range ${start}..${start + length} is out of range for dimension length ${tensor.shape[axis]}`);
    }
    return slice(tensor, axis, resolvedStart, resolvedStart + length, 1);
  }

  function slice(tensor: TensorViewTensor, dim: number, start = 0, end: number | null = null, step = 1) {
    const axis = normalizeDim(dim, tensor.shape.length, "slice");
    if (!Number.isSafeInteger(step) || step <= 0) throw new Error(`slice step must be a positive safe integer, got ${step}`);
    const dimLen = tensor.shape[axis];
    const begin = normalizeSliceBound(start, dimLen, 0, "slice start");
    const finish = normalizeSliceBound(end, dimLen, dimLen, "slice end");
    const positions = [];
    for (let index = begin; index < finish; index += step) positions.push(index);
    return gatherAxis(tensor, axis, positions, true, "slice");
  }

  function indexSelect(tensor: TensorViewTensor, dim: number, indices: unknown) {
    const axis = normalizeDim(dim, tensor.shape.length, "indexSelect");
    const raw = indices && typeof indices === "object" && "data" in indices
      ? Array.from((indices as { data: ArrayLike<number> }).data, Number)
      : Array.isArray(indices) || ArrayBuffer.isView(indices)
        ? Array.from(indices as ArrayLike<number>, Number)
        : null;
    if (!raw || raw.length === 0) throw new Error("indexSelect indices must be a non-empty 1D index array or Tensor");
    const positions = raw.map((value, index) => normalizeAxisIndex(value, tensor.shape[axis], `indexSelect index ${index}`));
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    if (!needsGrad && nativeIndexSelectInto !== null) {
      const outShape = tensor.shape.map((size: number, index: number) => index === axis ? positions.length : size);
      const outData = new Float32Array(shapeProduct(outShape));
      const outer = axis === 0 ? 1 : shapeProduct(tensor.shape.slice(0, axis));
      const inner = axis === tensor.shape.length - 1 ? 1 : shapeProduct(tensor.shape.slice(axis + 1));
      nativeIndexSelectInto(outData, tensor, u32Array(positions, "indexSelect index"), {
        outer,
        axisLen: tensor.shape[axis],
        inner,
      });
      return new TensorCtor(outData, outShape);
    }
    return gatherAxis(tensor, axis, positions, true, "indexSelect");
  }

  function index_select(tensor: TensorViewTensor, dim: number, indices: unknown) {
    return indexSelect(tensor, dim, indices);
  }

  function normalizeGatherIndex(index: unknown) {
    if (index && typeof index === "object" && "data" in index) {
      const tensorIndex = index as { data: ArrayLike<number>; shape?: readonly number[] };
      if (!Array.isArray(tensorIndex.shape) || tensorIndex.shape.length === 0) {
        throw new Error("gather index Tensor must carry a non-empty shape");
      }
      return {
        data: Array.from(tensorIndex.data, Number),
        shape: tensorIndex.shape.slice(),
      };
    }
    if (Array.isArray(index) || ArrayBuffer.isView(index)) {
      const data = Array.from(index as ArrayLike<number>, Number);
      return { data, shape: [data.length] };
    }
    throw new Error("gather index must be a Tensor or 1D index array");
  }

  function gather(tensor: TensorViewTensor, dim: number, index: unknown) {
    const axis = normalizeDim(dim, tensor.shape.length, "gather");
    const normalized = normalizeGatherIndex(index);
    if (normalized.data.length === 0) throw new Error("gather index must be non-empty");
    if (normalized.shape.length !== tensor.shape.length) {
      throw new Error(`gather index rank ${normalized.shape.length} must match input rank ${tensor.shape.length}`);
    }
    const outShape = normalized.shape.slice();
    const outLen = shapeProduct(outShape);
    if (outLen !== normalized.data.length) {
      throw new Error(`gather index data length ${normalized.data.length} does not match shape [${outShape.join(",")}]`);
    }
    for (let dimIndex = 0; dimIndex < outShape.length; dimIndex += 1) {
      if (dimIndex !== axis && outShape[dimIndex] > tensor.shape[dimIndex]) {
        throw new Error(`gather index shape at dim ${dimIndex} must be <= input shape, got ${outShape[dimIndex]} > ${tensor.shape[dimIndex]}`);
      }
    }

    const inStrides = rowMajorStrides(tensor.shape);
    const outStrides = rowMajorStrides(outShape);
    const outData = new Float32Array(outLen);
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    const outToIn = needsGrad ? new Uint32Array(outLen) : null;

    if (!needsGrad && nativeGatherInto !== null) {
      for (let flat = 0; flat < outLen; flat += 1) normalizeAxisIndex(normalized.data[flat], tensor.shape[axis], `gather index ${flat}`);
      nativeGatherInto(outData, tensor, u32Array(normalized.data, "gather index"), {
        outputShape: u32Array(outShape, "gather outputShape"),
        inputStrides: u32Array(inStrides, "gather inputStrides"),
        axis,
        axisLen: tensor.shape[axis],
      });
    } else {
      for (let flat = 0; flat < outLen; flat += 1) {
        let inputIndex = 0;
        for (let dimIndex = 0; dimIndex < outShape.length; dimIndex += 1) {
          const coord = Math.floor(flat / outStrides[dimIndex]) % outShape[dimIndex];
          const inputCoord = dimIndex === axis
            ? normalizeAxisIndex(normalized.data[flat], tensor.shape[axis], `gather index ${flat}`)
            : coord;
          inputIndex += inputCoord * inStrides[dimIndex];
        }
        if (outToIn !== null) outToIn[flat] = inputIndex;
        outData[flat] = tensor.data[inputIndex];
      }
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad || outToIn === null) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function take(tensor: TensorViewTensor, index: unknown) {
    const normalized = normalizeGatherIndex(index);
    if (normalized.data.length === 0) throw new Error("take index must be non-empty");
    const outShape = normalized.shape.slice();
    const outLen = shapeProduct(outShape);
    if (outLen !== normalized.data.length) {
      throw new Error(`take index data length ${normalized.data.length} does not match shape [${outShape.join(",")}]`);
    }

    const outData = new Float32Array(outLen);
    const needsGrad = gradModeEnabled() && tensor.requiresGrad;
    const outToIn = needsGrad ? new Uint32Array(outLen) : null;
    if (!needsGrad && nativeTakeInto !== null) {
      for (let flat = 0; flat < outLen; flat += 1) normalizeAxisIndex(normalized.data[flat], tensor.length, `take index ${flat}`);
      nativeTakeInto(outData, tensor, u32Array(normalized.data, "take index"));
    } else {
      for (let flat = 0; flat < outLen; flat += 1) {
        const inputIndex = normalizeAxisIndex(normalized.data[flat], tensor.length, `take index ${flat}`);
        if (outToIn !== null) outToIn[flat] = inputIndex;
        outData[flat] = tensor.data[inputIndex];
      }
    }

    const out = new TensorCtor(outData, outShape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor] : [],
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad || outToIn === null) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return out;
  }

  function sortedAxis(tensor: TensorViewTensor, dim: number, descending: boolean, label: string) {
    const axis = normalizeDim(dim, tensor.shape.length, label);
    if (typeof descending !== "boolean") throw new Error(`${label} descending must be a boolean, got ${descending}`);
    const strides = rowMajorStrides(tensor.shape);
    const axisLen = tensor.shape[axis];
    const axisStride = strides[axis];
    const outerShape = tensor.shape.filter((_: number, index: number) => index !== axis);
    const outerLen = outerShape.length === 0 ? 1 : shapeProduct(outerShape);
    const outerStrides = outerShape.length === 0 ? [] : rowMajorStrides(outerShape);
    const indexData = new Float32Array(tensor.length);
    const valueData = new Float32Array(tensor.length);
    const outToIn = new Uint32Array(tensor.length);

    for (let outerFlat = 0; outerFlat < outerLen; outerFlat += 1) {
      let base = 0;
      let outerDim = 0;
      for (let dimIndex = 0; dimIndex < tensor.shape.length; dimIndex += 1) {
        if (dimIndex === axis) continue;
        const coord = outerShape.length === 0 ? 0 : Math.floor(outerFlat / outerStrides[outerDim]) % outerShape[outerDim];
        base += coord * strides[dimIndex];
        outerDim += 1;
      }
      const positions = Array.from({ length: axisLen }, (_value, index) => index);
      positions.sort((left, right) => {
        const leftValue = tensor.data[base + left * axisStride];
        const rightValue = tensor.data[base + right * axisStride];
        const order = leftValue < rightValue ? -1 : leftValue > rightValue ? 1 : left - right;
        return descending ? -order : order;
      });
      for (let outIndex = 0; outIndex < axisLen; outIndex += 1) {
        const outputFlat = base + outIndex * axisStride;
        const inputFlat = base + positions[outIndex] * axisStride;
        indexData[outputFlat] = positions[outIndex];
        valueData[outputFlat] = tensor.data[inputFlat];
        outToIn[outputFlat] = inputFlat;
      }
    }

    return { indexData, valueData, outToIn };
  }

  function argsort(tensor: TensorViewTensor, dim = -1, descending = false) {
    return new TensorCtor(sortedAxis(tensor, dim, descending, "argsort").indexData, tensor.shape);
  }

  function sort(tensor: TensorViewTensor, dim = -1, descending = false) {
    const sorted = sortedAxis(tensor, dim, descending, "sort");
    const values = new TensorCtor(sorted.valueData, tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    values._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[sorted.outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return Object.freeze({
      values,
      indices: new TensorCtor(sorted.indexData, tensor.shape),
    });
  }

  function topk(tensor: TensorViewTensor, k: number, dim = -1, largest = true, sorted = true) {
    const axis = normalizeDim(dim, tensor.shape.length, "topk");
    if (!Number.isSafeInteger(k) || k <= 0) throw new Error(`topk k must be a positive safe integer, got ${k}`);
    if (typeof largest !== "boolean") throw new Error(`topk largest must be a boolean, got ${largest}`);
    if (typeof sorted !== "boolean") throw new Error(`topk sorted must be a boolean, got ${sorted}`);
    const axisLen = tensor.shape[axis];
    if (k > axisLen) throw new Error(`topk k ${k} must be <= dimension length ${axisLen}`);

    const inStrides = rowMajorStrides(tensor.shape);
    const outShape = tensor.shape.map((dimSize: number, index: number) => index === axis ? k : dimSize);
    const outStrides = rowMajorStrides(outShape);
    const axisStride = inStrides[axis];
    const outAxisStride = outStrides[axis];
    const outerShape = tensor.shape.filter((_: number, index: number) => index !== axis);
    const outerLen = outerShape.length === 0 ? 1 : shapeProduct(outerShape);
    const outerStrides = outerShape.length === 0 ? [] : rowMajorStrides(outerShape);
    const outLen = shapeProduct(outShape);
    const valueData = new Float32Array(outLen);
    const indexData = new Float32Array(outLen);
    const outToIn = new Uint32Array(outLen);

    for (let outerFlat = 0; outerFlat < outerLen; outerFlat += 1) {
      let inputBase = 0;
      let outputBase = 0;
      let outerDim = 0;
      for (let dimIndex = 0; dimIndex < tensor.shape.length; dimIndex += 1) {
        if (dimIndex === axis) continue;
        const coord = outerShape.length === 0 ? 0 : Math.floor(outerFlat / outerStrides[outerDim]) % outerShape[outerDim];
        inputBase += coord * inStrides[dimIndex];
        outputBase += coord * outStrides[dimIndex];
        outerDim += 1;
      }

      const positions = Array.from({ length: axisLen }, (_value, index) => index);
      positions.sort((left, right) => {
        const leftValue = tensor.data[inputBase + left * axisStride];
        const rightValue = tensor.data[inputBase + right * axisStride];
        const order = leftValue < rightValue ? -1 : leftValue > rightValue ? 1 : left - right;
        return largest ? -order : order;
      });
      const selected = positions.slice(0, k);
      if (!sorted) selected.sort((left, right) => left - right);
      for (let outIndex = 0; outIndex < k; outIndex += 1) {
        const inputAxisIndex = selected[outIndex];
        const outputFlat = outputBase + outIndex * outAxisStride;
        const inputFlat = inputBase + inputAxisIndex * axisStride;
        indexData[outputFlat] = inputAxisIndex;
        valueData[outputFlat] = tensor.data[inputFlat];
        outToIn[outputFlat] = inputFlat;
      }
    }

    const values = new TensorCtor(valueData, outShape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    values._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < grad.length; i += 1) inGrad[outToIn[i]] += grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return Object.freeze({
      values,
      indices: new TensorCtor(indexData, outShape),
    });
  }

  function normalizeScatterSource(src: unknown, count: number) {
    if (typeof src === "number") {
      if (!Number.isFinite(src)) throw new Error(`scatterAdd source scalar must be finite, got ${src}`);
      return { data: null, scalar: src, tensor: null as TensorViewTensor | null };
    }
    if (src && typeof src === "object" && "data" in src) {
      const tensorSrc = src as TensorViewTensor;
      if (tensorSrc.data.length !== count) {
        throw new Error(`scatterAdd source Tensor length ${tensorSrc.data.length} must match index length ${count}`);
      }
      return { data: tensorSrc.data, scalar: null, tensor: tensorSrc };
    }
    if (Array.isArray(src) || ArrayBuffer.isView(src)) {
      const data = Float32Array.from(src as ArrayLike<number>, Number);
      if (data.length !== count) {
        throw new Error(`scatterAdd source length ${data.length} must match index length ${count}`);
      }
      return { data, scalar: null, tensor: null as TensorViewTensor | null };
    }
    throw new Error("scatterAdd source must be a finite number, Tensor, or array");
  }

  function scatterAdd(tensor: TensorViewTensor, dim: number, index: unknown, src: unknown) {
    const axis = normalizeDim(dim, tensor.shape.length, "scatterAdd");
    const normalized = normalizeGatherIndex(index);
    if (normalized.data.length === 0) throw new Error("scatterAdd index must be non-empty");
    if (normalized.shape.length !== tensor.shape.length) {
      throw new Error(`scatterAdd index rank ${normalized.shape.length} must match input rank ${tensor.shape.length}`);
    }
    const indexLen = shapeProduct(normalized.shape);
    if (indexLen !== normalized.data.length) {
      throw new Error(`scatterAdd index data length ${normalized.data.length} does not match shape [${normalized.shape.join(",")}]`);
    }
    for (let dimIndex = 0; dimIndex < normalized.shape.length; dimIndex += 1) {
      if (dimIndex !== axis && normalized.shape[dimIndex] > tensor.shape[dimIndex]) {
        throw new Error(`scatterAdd index shape at dim ${dimIndex} must be <= input shape, got ${normalized.shape[dimIndex]} > ${tensor.shape[dimIndex]}`);
      }
    }

    const source = normalizeScatterSource(src, normalized.data.length);
    const inStrides = rowMajorStrides(tensor.shape);
    const indexStrides = rowMajorStrides(normalized.shape);
    const outData = new Float32Array(tensor.data);
    const indexToOut = new Uint32Array(normalized.data.length);

    for (let flat = 0; flat < normalized.data.length; flat += 1) {
      let outputIndex = 0;
      for (let dimIndex = 0; dimIndex < normalized.shape.length; dimIndex += 1) {
        const coord = Math.floor(flat / indexStrides[dimIndex]) % normalized.shape[dimIndex];
        const outputCoord = dimIndex === axis
          ? normalizeAxisIndex(normalized.data[flat], tensor.shape[axis], `scatterAdd index ${flat}`)
          : coord;
        outputIndex += outputCoord * inStrides[dimIndex];
      }
      indexToOut[flat] = outputIndex;
      outData[outputIndex] += source.scalar === null ? source.data![flat] : source.scalar;
    }

    const prev = gradModeEnabled()
      ? source.tensor && source.tensor.requiresGrad && tensor.requiresGrad
        ? [tensor, source.tensor]
        : tensor.requiresGrad
          ? [tensor]
          : source.tensor && source.tensor.requiresGrad
            ? [source.tensor]
            : []
      : [];
    const out = new TensorCtor(outData, tensor.shape, {
      requiresGrad: prev.length > 0,
      prev,
    });
    out._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      if (tensor.requiresGrad) addTensorGrad(tensor, grad);
      if (source.tensor && source.tensor.requiresGrad) {
        const srcGrad = new Float32Array(source.tensor.length);
        for (let i = 0; i < indexToOut.length; i += 1) srcGrad[i] += grad[indexToOut[i]];
        addTensorGrad(source.tensor, srcGrad);
      }
    };
    return out;
  }

  function scatter_add(tensor: TensorViewTensor, dim: number, index: unknown, src: unknown) {
    return scatterAdd(tensor, dim, index, src);
  }

  function split(tensor: TensorViewTensor, splitSizeOrSections: unknown, dim = 0) {
    const axis = normalizeDim(dim, tensor.shape.length, "split");
    const dimLen = tensor.shape[axis];
    const sections = Number.isSafeInteger(splitSizeOrSections)
      ? (() => {
        const splitSize = splitSizeOrSections as number;
        if (splitSize <= 0) throw new Error(`split size must be a positive safe integer, got ${splitSize}`);
        const out: number[] = [];
        for (let remaining = dimLen; remaining > 0; remaining -= splitSize) out.push(Math.min(splitSize, remaining));
        return out;
      })()
      : splitSizeOrSections;
    if (!Array.isArray(sections) || sections.length === 0) {
      throw new Error("split sections must be a positive integer or non-empty integer array");
    }
    let total = 0;
    const sizes = sections.map((section) => {
      if (!Number.isSafeInteger(section) || section <= 0) {
        throw new Error(`split sections must be positive safe integers, got ${section}`);
      }
      total += section;
      return section;
    });
    if (total !== dimLen) throw new Error(`split sections must sum to dimension length ${dimLen}, got ${total}`);
    let start = 0;
    return sizes.map((size) => {
      const out = slice(tensor, axis, start, start + size, 1);
      start += size;
      return out;
    });
  }

  function chunk(tensor: TensorViewTensor, chunks: number, dim = 0) {
    const axis = normalizeDim(dim, tensor.shape.length, "chunk");
    if (!Number.isSafeInteger(chunks) || chunks <= 0) {
      throw new Error(`chunk chunks must be a positive safe integer, got ${chunks}`);
    }
    const dimLen = tensor.shape[axis];
    const size = Math.ceil(dimLen / chunks);
    const out: TensorViewTensor[] = [];
    for (let start = 0; start < dimLen; start += size) out.push(slice(tensor, axis, start, Math.min(start + size, dimLen), 1));
    return out;
  }

  function unbind(tensor: TensorViewTensor, dim = 0) {
    const axis = normalizeDim(dim, tensor.shape.length, "unbind");
    return Array.from({ length: tensor.shape[axis] }, (_value, index) => select(tensor, axis, index));
  }

  return {
    clone,
    detach,
    reshape,
    broadcastTo,
    expand,
    repeat,
    tile,
    flatten,
    squeeze,
    unsqueeze,
    transpose,
    permute,
    flip,
    roll,
    select,
    narrow,
    slice,
    indexSelect,
    index_select,
    gather,
    take,
    argsort,
    sort,
    topk,
    scatterAdd,
    scatter_add,
    split,
    chunk,
    unbind,
  };
}

export function createTensorViewSurfaceHelpers<TTensor>(options: TensorViewSurfaceHelpersOptions) {
  const helpers = options.tensorViewHelpers;
  type TensorViewSortResult = Readonly<{ values: TTensor; indices: TTensor }>;
  return Object.freeze({
    clone: (tensor: TTensor) => helpers.clone(tensor) as TTensor,
    detach: (tensor: TTensor) => helpers.detach(tensor) as TTensor,
    reshape: (tensor: TTensor, shape: unknown) => helpers.reshape(tensor, shape) as TTensor,
    view: (tensor: TTensor, shape: unknown) => helpers.reshape(tensor, shape, "view") as TTensor,
    broadcastTo: (tensor: TTensor, shape: unknown) => helpers.broadcastTo(tensor, shape) as TTensor,
    expand: (tensor: TTensor, shape: unknown) => helpers.expand(tensor, shape) as TTensor,
    repeat: (tensor: TTensor, repeats: unknown) => helpers.repeat(tensor, repeats) as TTensor,
    tile: (tensor: TTensor, repeats: unknown) => helpers.tile(tensor, repeats) as TTensor,
    flatten: (tensor: TTensor, startDim = 0, endDim = -1) => helpers.flatten(tensor, startDim, endDim) as TTensor,
    squeeze: (tensor: TTensor, dim: number | null = null) => helpers.squeeze(tensor, dim) as TTensor,
    unsqueeze: (tensor: TTensor, dim: number) => helpers.unsqueeze(tensor, dim) as TTensor,
    transpose: (tensor: TTensor, dim0 = 0, dim1 = 1) => helpers.transpose(tensor, dim0, dim1) as TTensor,
    permute: (tensor: TTensor, dims: readonly number[]) => helpers.permute(tensor, dims) as TTensor,
    flip: (tensor: TTensor, dims: readonly number[]) => helpers.flip(tensor, dims) as TTensor,
    roll: (tensor: TTensor, shifts: unknown, dims?: unknown) => helpers.roll(tensor, shifts, dims) as TTensor,
    select: (tensor: TTensor, dim: number, index: number) => helpers.select(tensor, dim, index) as TTensor,
    narrow: (tensor: TTensor, dim: number, start: number, length: number) => helpers.narrow(tensor, dim, start, length) as TTensor,
    slice: (tensor: TTensor, dim: number, start?: number | null, end?: number | null, step?: number) => (
      helpers.slice(tensor, dim, start, end, step) as TTensor
    ),
    indexSelect: (tensor: TTensor, dim: number, indices: unknown) => helpers.indexSelect(tensor, dim, indices) as TTensor,
    index_select: (tensor: TTensor, dim: number, indices: unknown) => helpers.index_select(tensor, dim, indices) as TTensor,
    gather: (tensor: TTensor, dim: number, index: unknown) => helpers.gather(tensor, dim, index) as TTensor,
    take: (tensor: TTensor, index: unknown) => helpers.take(tensor, index) as TTensor,
    argsort: (tensor: TTensor, dim = -1, descending = false) => helpers.argsort(tensor, dim, descending) as TTensor,
    sort: (tensor: TTensor, dim = -1, descending = false) => helpers.sort(tensor, dim, descending) as TensorViewSortResult,
    topk: (tensor: TTensor, k: number, dim = -1, largest = true, sorted = true) => helpers.topk(tensor, k, dim, largest, sorted) as TensorViewSortResult,
    scatterAdd: (tensor: TTensor, dim: number, index: unknown, src: unknown) => helpers.scatterAdd(tensor, dim, index, src) as TTensor,
    scatter_add: (tensor: TTensor, dim: number, index: unknown, src: unknown) => helpers.scatter_add(tensor, dim, index, src) as TTensor,
    split: (tensor: TTensor, splitSizeOrSections: unknown, dim = 0) => helpers.split(tensor, splitSizeOrSections, dim) as readonly TTensor[],
    chunk: (tensor: TTensor, chunks: number, dim = 0) => helpers.chunk(tensor, chunks, dim) as readonly TTensor[],
    unbind: (tensor: TTensor, dim = 0) => helpers.unbind(tensor, dim) as readonly TTensor[],
    T: (tensor: TTensor) => helpers.transpose(tensor) as TTensor,
    mT: (tensor: TTensor) => helpers.transpose(tensor, -2, -1) as TTensor,
  });
}
