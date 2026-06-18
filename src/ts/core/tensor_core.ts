"use strict";

import {
  normalizeDim,
} from "./shape.js";
import type {
  AllCloseOptions,
  TensorToOptions,
  TensorToTarget,
  ZeroGradOptions,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type TensorCoreConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  requires_grad?: boolean;
  prev?: readonly TensorCoreTensor[];
}>;
type TensorCoreTensor = AnyRecord & {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  requiresGrad: boolean;
  grad: Float32Array | null;
  _prev?: readonly TensorCoreTensor[];
  _backward?: (grad: Float32Array | null) => void;
};
type TensorConstructor = new (
  data: unknown,
  shape?: readonly number[],
  options?: TensorCoreConstructOptions,
) => TensorCoreTensor;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type PreparedF32 = { data: Float32Array; shape: readonly number[] };
type F32WithLengthCallback = (data: unknown, length: number, name: string, expectedShape?: readonly number[]) => Float32Array;
type PrepareF32Callback = (data: unknown) => PreparedF32;
type AddTensorGradCallback = BivariantCallback<[tensor: TensorCoreTensor, grad: Float32Array], void>;

export type TensorCoreHelpersOptions = Readonly<{
  Tensor?: TensorConstructor;
  getTensorClass?: () => TensorConstructor | undefined;
  f32WithLength: F32WithLengthCallback;
  prepareF32: PrepareF32Callback;
  addTensorGrad: AddTensorGradCallback;
}>;

export type TensorInfoSurfaceHelpersOptions = Readonly<{
  tensorCoreHelpers: AnyRecord;
  tensorMetadataHelpers: AnyRecord;
  tensorGradStateHelpers: AnyRecord;
  tensorPlacementHelpers: AnyRecord;
}>;

export function createTensorCoreHelpers(options: TensorCoreHelpersOptions) {
  const getTensorClass = typeof options.getTensorClass === "function"
    ? options.getTensorClass
    : () => options.Tensor;
  const f32WithLength = options.f32WithLength;
  const prepareF32 = options.prepareF32;
  const addTensorGrad = options.addTensorGrad;
  if (typeof f32WithLength !== "function" || typeof prepareF32 !== "function" || typeof addTensorGrad !== "function") {
    throw new Error("tensor core helpers require f32WithLength, prepareF32, and addTensorGrad hooks");
  }

  function tensorClass() {
    const TensorClass = getTensorClass();
    if (typeof TensorClass !== "function") {
      throw new Error("tensor core helpers require a Tensor constructor");
    }
    return TensorClass as TensorConstructor;
  }

  function toFloat32Array(tensor: TensorCoreTensor) {
    return tensor.data;
  }

  function length(tensor: TensorCoreTensor) {
    return tensor.data.length;
  }

  function rank(tensor: TensorCoreTensor) {
    return tensor.shape.length;
  }

  function numpy(tensor: TensorCoreTensor) {
    return new Float32Array(tensor.data);
  }

  function iterator(tensor: TensorCoreTensor) {
    return tensor.data[Symbol.iterator]();
  }

  function item(tensor: TensorCoreTensor) {
    if (tensor.length !== 1) throw new Error(`Tensor.item() requires a scalar tensor, got length ${tensor.length}`);
    return tensor.data[0];
  }

  function ndim(tensor: TensorCoreTensor) {
    return tensor.shape.length;
  }

  function ndimension(tensor: TensorCoreTensor) {
    return dim(tensor);
  }

  function dim(tensor: TensorCoreTensor) {
    return tensor.shape.length;
  }

  function numel(tensor: TensorCoreTensor) {
    return tensor.data.length;
  }

  function nelement(tensor: TensorCoreTensor) {
    return numel(tensor);
  }

  function size(tensor: TensorCoreTensor, axis?: number) {
    if (axis === undefined) return Object.freeze(tensor.shape.slice());
    return tensor.shape[normalizeDim(axis, tensor.shape.length, "Tensor.size")];
  }

  function allclose(tensor: TensorCoreTensor, other: unknown, options: AllCloseOptions = {}) {
    const { rtol = 1e-5, atol = 1e-8 } = options || {};
    if (!Number.isFinite(rtol) || rtol < 0) throw new Error(`Tensor.allclose rtol must be a finite non-negative number, got ${rtol}`);
    if (!Number.isFinite(atol) || atol < 0) throw new Error(`Tensor.allclose atol must be a finite non-negative number, got ${atol}`);

    const TensorClass = tensorClass();
    let otherShape = null;
    let otherData = null;
    if (other instanceof TensorClass) {
      otherShape = other.shape;
      otherData = other.data;
    } else {
      const prepared = prepareF32(other);
      otherShape = prepared.shape;
      otherData = prepared.data;
    }
    if (otherData.length !== tensor.length || otherShape.length !== tensor.shape.length) return false;
    for (let i = 0; i < tensor.shape.length; i += 1) {
      if (otherShape[i] !== tensor.shape[i]) return false;
    }
    for (let i = 0; i < tensor.length; i += 1) {
      const actual = tensor.data[i];
      const expected = otherData[i];
      if (Math.abs(actual - expected) > atol + rtol * Math.abs(expected)) return false;
    }
    return true;
  }

  function equal(tensor: TensorCoreTensor, other: unknown) {
    const TensorClass = tensorClass();
    let otherShape = null;
    let otherData = null;
    if (other instanceof TensorClass) {
      otherShape = other.shape;
      otherData = other.data;
    } else {
      const prepared = prepareF32(other);
      otherShape = prepared.shape;
      otherData = prepared.data;
    }
    if (otherData.length !== tensor.length || otherShape.length !== tensor.shape.length) return false;
    for (let i = 0; i < tensor.shape.length; i += 1) {
      if (otherShape[i] !== tensor.shape[i]) return false;
    }
    for (let i = 0; i < tensor.length; i += 1) {
      if (!Object.is(tensor.data[i], otherData[i])) return false;
    }
    return true;
  }

  function zeroGrad(tensor: TensorCoreTensor, options: ZeroGradOptions = {}) {
    if (options.setToNone === true || options.set_to_none === true) {
      tensor.grad = null;
      return;
    }
    if (tensor.grad) tensor.grad.fill(0);
  }

  function zero_grad(tensor: TensorCoreTensor, options: ZeroGradOptions = {}) {
    return zeroGrad(tensor, options);
  }

  function fill_(tensor: TensorCoreTensor, value: unknown) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`Tensor.fill_ value must be a finite number, got ${value}`);
    }
    tensor.data.fill(value);
    return tensor;
  }

  function zero_(tensor: TensorCoreTensor) {
    tensor.data.fill(0);
    return tensor;
  }

  function ones_(tensor: TensorCoreTensor) {
    tensor.data.fill(1);
    return tensor;
  }

  function copy_(tensor: TensorCoreTensor, source: unknown) {
    const data = source instanceof Float32Array
      ? source
      : source && typeof source === "object" && (source as AnyRecord).data instanceof Float32Array
        ? (source as AnyRecord).data
        : f32WithLength(source, tensor.length, "Tensor.copy_ source", tensor.shape);
    if (data.length !== tensor.length) {
      throw new Error(`Tensor.copy_ source length ${data.length} does not match tensor length ${tensor.length}`);
    }
    tensor.data.set(data);
    return tensor;
  }

  function backward(tensor: TensorCoreTensor, gradient?: unknown) {
    const seed = gradient === undefined
      ? (tensor.length === 1 ? Float32Array.of(1) : null)
      : f32WithLength(gradient, tensor.length, "backward gradient");
    if (!seed) {
      throw new Error("Tensor.backward() on a non-scalar tensor requires an explicit gradient");
    }
    const topo: TensorCoreTensor[] = [];
    const seen = new Set();
    const visit = (next: TensorCoreTensor) => {
      if (seen.has(next)) return;
      seen.add(next);
      for (const parent of next._prev ?? []) visit(parent);
      topo.push(next);
    };
    visit(tensor);
    tensor.requiresGrad = true;
    addTensorGrad(tensor, seed);
    for (let i = topo.length - 1; i >= 0; i -= 1) {
      if ((topo[i]._prev?.length ?? 0) > 0) {
        const backwardNode = topo[i];
        const backwardCallback = backwardNode._backward;
        if (typeof backwardCallback !== "function") throw new Error("Tensor.backward() graph node is missing a backward callback");
        backwardCallback(backwardNode.grad);
      }
    }
  }

  function toJSON(tensor: TensorCoreTensor) {
    return {
      dtype: "f32",
      shape: tensor.shape.slice(),
      data: Array.from(tensor.data),
      requiresGrad: Boolean(tensor.requiresGrad),
    };
  }

  function fromJSON(value: AnyRecord) {
    const candidate = value;
    if (
      !candidate ||
      candidate.dtype !== "f32" ||
      !Array.isArray(candidate.shape) ||
      !Array.isArray(candidate.data)
    ) {
      throw new Error('Tensor.fromJSON expects { dtype: "f32", shape, data }');
    }
    const hasRequiresGrad = Object.prototype.hasOwnProperty.call(candidate, "requiresGrad");
    const hasRequires_grad = Object.prototype.hasOwnProperty.call(candidate, "requires_grad");
    const camelRequiresGrad = candidate.requiresGrad;
    const snakeRequiresGrad = candidate.requires_grad;
    if ((hasRequiresGrad && typeof camelRequiresGrad !== "boolean") || (hasRequires_grad && typeof snakeRequiresGrad !== "boolean")) {
      throw new Error("Tensor.fromJSON requiresGrad must be a boolean when provided");
    }
    if (hasRequiresGrad && hasRequires_grad && camelRequiresGrad !== snakeRequiresGrad) {
      throw new Error("Tensor.fromJSON requiresGrad and requires_grad must match when both are provided");
    }
    const requiresGrad = hasRequiresGrad ? camelRequiresGrad : hasRequires_grad ? snakeRequiresGrad : false;
    return new (tensorClass())(candidate.data, candidate.shape, { requiresGrad });
  }

  return Object.freeze({
    toFloat32Array,
    length,
    rank,
    numpy,
    iterator,
    item,
    toNumber: item,
    ndim,
    ndimension,
    dim,
    numel,
    nelement,
    size,
    allclose,
    equal,
    valueOf: item,
    toPrimitive: item,
    zeroGrad,
    zero_grad,
    fill_,
    zero_,
    ones_,
    copy_,
    backward,
    toJSON,
    fromJSON,
  });
}

export function createTensorInfoSurfaceHelpers<TTensor>(options: TensorInfoSurfaceHelpersOptions) {
  const {
    tensorCoreHelpers,
    tensorMetadataHelpers,
    tensorGradStateHelpers,
    tensorPlacementHelpers,
  } = options;

  return Object.freeze({
    length: (tensor: TTensor) => tensorCoreHelpers.length(tensor) as number,
    rank: (tensor: TTensor) => tensorCoreHelpers.rank(tensor) as number,
    ndim: (tensor: TTensor) => tensorCoreHelpers.ndim(tensor) as number,
    dtype: (tensor: TTensor) => tensorPlacementHelpers.dtype(tensor),
    device: (tensor: TTensor) => tensorPlacementHelpers.device(tensor),
    isCpu: (tensor: TTensor) => tensorMetadataHelpers.isCpu(tensor) as boolean,
    is_cpu: (tensor: TTensor) => tensorMetadataHelpers.is_cpu(tensor) as boolean,
    isFloatingPoint: (tensor: TTensor) => tensorMetadataHelpers.isFloatingPoint(tensor) as boolean,
    is_floating_point: (tensor: TTensor) => tensorMetadataHelpers.is_floating_point(tensor) as boolean,
    isLeaf: (tensor: TTensor) => tensorMetadataHelpers.isLeaf(tensor) as boolean,
    is_leaf: (tensor: TTensor) => tensorMetadataHelpers.is_leaf(tensor) as boolean,
    requires_grad: (tensor: TTensor) => tensorGradStateHelpers.requiresGrad(tensor) as boolean,
    setRequires_grad: (tensor: TTensor, value: unknown) => tensorGradStateHelpers.setRequiresGrad(tensor, value),
    requires_grad_: (tensor: TTensor, value = true) => tensorGradStateHelpers.requiresGrad_(tensor, value) as TTensor,
    requiresGrad_: (tensor: TTensor, value = true) => tensorGradStateHelpers.requiresGrad_(tensor, value) as TTensor,
    toFloat32Array: (tensor: TTensor) => tensorCoreHelpers.toFloat32Array(tensor) as Float32Array,
    numpy: (tensor: TTensor) => tensorCoreHelpers.numpy(tensor) as Float32Array,
    to: (tensor: TTensor, target?: TensorToTarget | TensorToOptions, toOptions?: TensorToOptions) => tensorPlacementHelpers.to(tensor, target, toOptions) as TTensor,
    cpu: (tensor: TTensor, cpuOptions?: TensorToOptions) => tensorPlacementHelpers.cpu(tensor, cpuOptions) as TTensor,
    float: (tensor: TTensor, floatOptions?: TensorToOptions) => tensorPlacementHelpers.float(tensor, floatOptions) as TTensor,
    float32: (tensor: TTensor, floatOptions?: TensorToOptions) => tensorPlacementHelpers.float32(tensor, floatOptions) as TTensor,
    typeAs: (tensor: TTensor, other: unknown, typeAsOptions?: TensorToOptions) => tensorPlacementHelpers.typeAs(tensor, other, typeAsOptions) as TTensor,
    type_as: (tensor: TTensor, other: unknown, typeAsOptions?: TensorToOptions) => tensorPlacementHelpers.type_as(tensor, other, typeAsOptions) as TTensor,
    inspect: (tensor: TTensor) => tensorMetadataHelpers.inspect(tensor),
    item: (tensor: TTensor) => tensorCoreHelpers.item(tensor) as number,
    toNumber: (tensor: TTensor) => tensorCoreHelpers.toNumber(tensor) as number,
    dim: (tensor: TTensor) => tensorCoreHelpers.dim(tensor) as number,
    ndimension: (tensor: TTensor) => tensorCoreHelpers.ndimension(tensor) as number,
    numel: (tensor: TTensor) => tensorCoreHelpers.numel(tensor) as number,
    nelement: (tensor: TTensor) => tensorCoreHelpers.nelement(tensor) as number,
    elementSize: (tensor: TTensor) => tensorMetadataHelpers.elementSize(tensor) as number,
    element_size: (tensor: TTensor) => tensorMetadataHelpers.element_size(tensor) as number,
    nbytes: (tensor: TTensor) => tensorMetadataHelpers.nbytes(tensor) as number,
    size: (tensor: TTensor, dim?: number) => tensorCoreHelpers.size(tensor, dim),
    allclose: (tensor: TTensor, other: unknown, allcloseOptions?: AllCloseOptions) => (
      tensorCoreHelpers.allclose(tensor, other, allcloseOptions) as boolean
    ),
    equal: (tensor: TTensor, other: unknown) => tensorCoreHelpers.equal(tensor, other) as boolean,
    stride: (tensor: TTensor, dim?: number) => tensorMetadataHelpers.stride(tensor, dim),
    strides: (tensor: TTensor) => tensorMetadataHelpers.strides(tensor),
    storageOffset: (tensor: TTensor) => tensorMetadataHelpers.storageOffset(tensor) as number,
    storage_offset: (tensor: TTensor) => tensorMetadataHelpers.storage_offset(tensor) as number,
    isContiguous: (tensor: TTensor) => tensorMetadataHelpers.isContiguous(tensor) as boolean,
    is_contiguous: (tensor: TTensor) => tensorMetadataHelpers.is_contiguous(tensor) as boolean,
    contiguous: (tensor: TTensor) => tensorMetadataHelpers.contiguous(tensor) as TTensor,
    valueOf: (tensor: TTensor) => tensorCoreHelpers["valueOf"](tensor) as number,
    toPrimitive: (tensor: TTensor) => tensorCoreHelpers["toPrimitive"](tensor) as number,
    zeroGrad: (tensor: TTensor, options?: ZeroGradOptions) => tensorCoreHelpers.zeroGrad(tensor, options),
    zero_grad: (tensor: TTensor, options?: ZeroGradOptions) => tensorCoreHelpers.zero_grad(tensor, options),
    fill_: (tensor: TTensor, value: unknown) => tensorCoreHelpers.fill_(tensor, value) as TTensor,
    zero_: (tensor: TTensor) => tensorCoreHelpers.zero_(tensor) as TTensor,
    ones_: (tensor: TTensor) => tensorCoreHelpers.ones_(tensor) as TTensor,
    copy_: (tensor: TTensor, source: unknown) => tensorCoreHelpers.copy_(tensor, source) as TTensor,
    backward: (tensor: TTensor, gradient?: unknown) => tensorCoreHelpers.backward(tensor, gradient),
    detach_: (tensor: TTensor) => tensorGradStateHelpers.detach_(tensor) as TTensor,
    toJSON: (tensor: TTensor) => tensorCoreHelpers.toJSON(tensor),
    iterator: (tensor: TTensor) => tensorCoreHelpers.iterator(tensor) as IterableIterator<number>,
  });
}
