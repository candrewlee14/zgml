"use strict";

import {
  normalizeFactoryShape,
  validateTensorShape,
  type Shape,
} from "./shape.js";
import type {
  RandomIntTensorOptions,
  RandomTensorOptions,
  RandomUniformTensorOptions,
  TensorFromNativeBufferOptions,
  TensorNativeBufferOptions,
  TensorNativePlacement,
  TensorOptions,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type TensorInitOptions = Readonly<TensorOptions & {
  prev?: readonly unknown[];
  backward?: (grad: Float32Array | null) => void;
}>;
type TensorFacadeTensor = AnyRecord & {
  data: Float32Array;
  shape: readonly number[];
  requiresGrad: boolean;
  grad: Float32Array | null;
  _prev?: readonly unknown[];
  _backward?: (grad: Float32Array | null) => void;
};
type TensorConstructor = new (
  data: unknown,
  shape?: unknown,
  options?: TensorInitOptions,
) => TensorFacadeTensor;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type PreparedF32 = { data: Float32Array; shape: readonly number[] };
type TensorFactoryFacade = Readonly<{
  full: BivariantCallback<[shape: unknown, value: number, options?: TensorOptions], AnyRecord>;
  empty: BivariantCallback<[shape: unknown, options?: TensorOptions], AnyRecord>;
  zeros: BivariantCallback<[shape: unknown, options?: TensorOptions], AnyRecord>;
  ones: BivariantCallback<[shape: unknown, options?: TensorOptions], AnyRecord>;
  eye: BivariantCallback<[size: unknown, options?: TensorOptions], AnyRecord>;
  scalar: BivariantCallback<[value: number, options?: TensorOptions], AnyRecord>;
  rand: BivariantCallback<[shape: unknown, options?: RandomUniformTensorOptions], AnyRecord>;
  randn: BivariantCallback<[shape: unknown, options?: RandomTensorOptions], AnyRecord>;
  randInt: BivariantCallback<[first: unknown, second: unknown, third?: unknown, fourth?: RandomIntTensorOptions], AnyRecord>;
  randint: BivariantCallback<[first: unknown, second: unknown, third?: unknown, fourth?: RandomIntTensorOptions], AnyRecord>;
  randPerm: BivariantCallback<[size: unknown, options?: RandomIntTensorOptions], AnyRecord>;
  randperm: BivariantCallback<[size: unknown, options?: RandomIntTensorOptions], AnyRecord>;
  manualSeed: BivariantCallback<[seed: unknown], unknown>;
  manual_seed: BivariantCallback<[seed: unknown], unknown>;
  initialSeed: () => unknown;
  initial_seed: () => unknown;
  seededRng: BivariantCallback<[seed: unknown], unknown>;
  linspace: BivariantCallback<[first: unknown, second: unknown, third: unknown, fourth?: TensorOptions], AnyRecord>;
  arange: BivariantCallback<[start: unknown, endOrOptions?: unknown, stepOrOptions?: unknown, maybeOptions?: unknown], AnyRecord>;
}>;
type TensorPlacementFacade = Readonly<{
  validateProgramPlacement: BivariantCallback<[tensor: AnyRecord, program: AnyRecord, kind?: string], void>;
}>;
type TensorJoinFacade = Readonly<{
  cat: BivariantCallback<[tensors: unknown, dim?: number], AnyRecord>;
  stack: BivariantCallback<[tensors: unknown, dim?: number], AnyRecord>;
  einsum: BivariantCallback<[equation: unknown, tensors: unknown, ...moreTensors: unknown[]], AnyRecord>;
}>;

export type TensorFacadeHelpersOptions = Readonly<{
  Tensor: TensorConstructor;
  rawF32: (data: unknown) => Float32Array;
  prepareF32: (data: unknown) => PreparedF32;
  isTensor: (value: unknown) => boolean;
  isNativeBuffer: (value: unknown) => boolean;
  nativeBufferFromFloat32: (data: Float32Array) => unknown;
  tensorFactory: TensorFactoryFacade;
  tensorPlacement: TensorPlacementFacade;
  tensorJoin: TensorJoinFacade;
}>;

export type TensorNativeSurfaceHelpersOptions = Readonly<{
  tensorFacade: AnyRecord;
}>;

function isOptionsObject(value: unknown) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function createTensorFacadeHelpers(options: TensorFacadeHelpersOptions) {
  const TensorClass = options.Tensor;
  const rawF32 = options.rawF32;
  const prepareF32 = options.prepareF32;
  const isTensor = options.isTensor;
  const isNativeBuffer = options.isNativeBuffer;
  const nativeBufferFromFloat32 = options.nativeBufferFromFloat32;
  const tensorFactory = options.tensorFactory;
  const tensorPlacement = options.tensorPlacement;
  const tensorJoin = options.tensorJoin;

  if (typeof TensorClass !== "function") {
    throw new Error("createTensorFacadeHelpers requires Tensor");
  }
  if (typeof rawF32 !== "function") {
    throw new Error("createTensorFacadeHelpers requires rawF32");
  }
  if (typeof prepareF32 !== "function") {
    throw new Error("createTensorFacadeHelpers requires prepareF32");
  }
  if (typeof isTensor !== "function") {
    throw new Error("createTensorFacadeHelpers requires isTensor");
  }
  if (typeof isNativeBuffer !== "function") {
    throw new Error("createTensorFacadeHelpers requires isNativeBuffer");
  }
  if (typeof nativeBufferFromFloat32 !== "function") {
    throw new Error("createTensorFacadeHelpers requires nativeBufferFromFloat32");
  }
  if (
    !tensorFactory ||
    typeof tensorFactory.full !== "function" ||
    typeof tensorFactory.empty !== "function" ||
    typeof tensorFactory.zeros !== "function" ||
    typeof tensorFactory.ones !== "function" ||
    typeof tensorFactory.eye !== "function" ||
    typeof tensorFactory.scalar !== "function" ||
    typeof tensorFactory.rand !== "function" ||
    typeof tensorFactory.randn !== "function" ||
    typeof tensorFactory.randInt !== "function" ||
    typeof tensorFactory.randint !== "function" ||
    typeof tensorFactory.randPerm !== "function" ||
    typeof tensorFactory.randperm !== "function" ||
    typeof tensorFactory.manualSeed !== "function" ||
    typeof tensorFactory.manual_seed !== "function" ||
    typeof tensorFactory.initialSeed !== "function" ||
    typeof tensorFactory.initial_seed !== "function" ||
    typeof tensorFactory.seededRng !== "function" ||
    typeof tensorFactory.linspace !== "function" ||
    typeof tensorFactory.arange !== "function"
  ) {
    throw new Error("createTensorFacadeHelpers requires tensorFactory");
  }
  if (!tensorPlacement || typeof tensorPlacement.validateProgramPlacement !== "function") {
    throw new Error("createTensorFacadeHelpers requires tensorPlacement");
  }
  if (!tensorJoin || typeof tensorJoin.cat !== "function" || typeof tensorJoin.stack !== "function" || typeof tensorJoin.einsum !== "function") {
    throw new Error("createTensorFacadeHelpers requires tensorJoin");
  }

  const TensorCtor = TensorClass as TensorConstructor;

  function initialize(tensor: TensorFacadeTensor, data: unknown, shape: Shape | undefined, initOptions: TensorInitOptions = {}) {
    const tensorOptions = initOptions || {};
    if (isTensor(data)) {
      const source = data as TensorFacadeTensor;
      tensor.data = source.data;
      tensor.shape = validateTensorShape(shape ?? source.shape, source.data.length);
      tensor.requiresGrad = Boolean(tensorOptions.requiresGrad ?? source.requiresGrad);
    } else {
      const prepared = prepareF32(data);
      tensor.data = prepared.data;
      tensor.shape = validateTensorShape(shape ?? prepared.shape, tensor.data.length);
      tensor.requiresGrad = Boolean(tensorOptions.requiresGrad);
    }
    tensor.grad = tensorOptions.grad ?? (tensor.requiresGrad ? new Float32Array(tensor.data.length) : null);
    tensor._prev = tensorOptions.prev ?? [];
    tensor._backward = tensorOptions.backward ?? (() => {});
  }

  function tensor(data: unknown, shape?: unknown, tensorOptions?: TensorInitOptions) {
    return new TensorCtor(data, shape, tensorOptions);
  }

  function parameter(data: unknown, shape?: unknown, options: TensorOptions = {}) {
    let resolvedShape = shape;
    let resolvedOptions = options ?? {};
    if (isOptionsObject(resolvedShape)) {
      resolvedOptions = resolvedShape as TensorOptions;
      resolvedShape = undefined;
    }
    return new TensorCtor(data, resolvedShape, { ...resolvedOptions, requiresGrad: true });
  }

  function toNativeBuffer(tensorValue: AnyRecord, bufferOptions: TensorNativeBufferOptions = {}) {
    const opts = bufferOptions || {};
    if (opts.program !== undefined) {
      return place(tensorValue, opts.program, opts.kind ?? "input", opts);
    }
    return nativeBufferFromFloat32(tensorValue.data);
  }

  function nativePlacement(tensorValue: AnyRecord, bufferOptions: TensorNativeBufferOptions = {}): TensorNativePlacement {
    const opts = bufferOptions || {};
    const kind = opts.kind ?? "input";
    const placement = opts.placement ?? opts.backend ?? opts.device ?? null;
    if (opts.program !== undefined) {
      tensorPlacement.validateProgramPlacement(tensorValue, opts.program, kind);
    }
    const shape = Array.from(tensorValue.shape ?? [], Number);
    return Object.freeze({
      kind: "zgml.tensor.native-placement",
      storage: opts.program === undefined ? "host" : "program",
      bufferKind: opts.program === undefined ? null : kind,
      dtype: "f32",
      device: placement === null ? (opts.program === undefined ? "cpu" : "program") : String(placement),
      shape: Object.freeze(shape),
      length: tensorValue.data.length,
      byteLength: tensorValue.data.length * Float32Array.BYTES_PER_ELEMENT,
      signature: [
        "tensor-native-placement",
        `storage=${opts.program === undefined ? "host" : "program"}`,
        `buffer=${opts.program === undefined ? "none" : kind}`,
        `device=${placement === null ? (opts.program === undefined ? "cpu" : "program") : String(placement)}`,
        `shape=${shape.join("x")}`,
        `length=${tensorValue.data.length}`,
        `bytes=${tensorValue.data.length * Float32Array.BYTES_PER_ELEMENT}`,
      ].join("|"),
    });
  }

  function place(tensorValue: AnyRecord, program: AnyRecord, kind = "input", options: TensorNativeBufferOptions = {}) {
    if (!program || typeof program.createBuffer !== "function") {
      throw new Error("Tensor.place requires a compiled zgml Program");
    }
    tensorPlacement.validateProgramPlacement(tensorValue, program, kind);
    const { program: _program, kind: _kind, ...bufferOptions } = options || {};
    const buffer = program.createBuffer(kind, bufferOptions);
    buffer.writeFloat32(tensorValue.data);
    return buffer;
  }

  function fromNativeBuffer(buffer: AnyRecord, shape?: unknown, options: TensorFromNativeBufferOptions = {}) {
    if (!isNativeBuffer(buffer)) {
      throw new Error("Tensor.fromNativeBuffer requires a zgml NativeBuffer");
    }
    let resolvedShape = shape;
    let resolvedOptions = options || {};
    if (isOptionsObject(resolvedShape)) {
      resolvedOptions = resolvedShape as TensorFromNativeBufferOptions;
      resolvedShape = undefined;
    }
    const { length: requestedLength, byteOffset = 0, ...tensorOptions } = resolvedOptions;
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`invalid Tensor.fromNativeBuffer byteOffset: ${byteOffset}`);
    }

    let length = requestedLength;
    let tensorShape = null;
    if (resolvedShape !== undefined) {
      const spec = normalizeFactoryShape(resolvedShape as number | Shape, "Tensor.fromNativeBuffer shape");
      tensorShape = spec.shape;
      if (length === undefined) {
        length = spec.length;
      } else if (length !== spec.length) {
        throw new Error(`Tensor.fromNativeBuffer length ${length} does not match shape length ${spec.length}`);
      }
    } else if (length === undefined) {
      const byteLength = buffer.byteLength - byteOffset;
      if (byteLength < 0 || byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
        throw new Error(`invalid Tensor.fromNativeBuffer byte range: ${byteOffset}..${buffer.byteLength}`);
      }
      length = byteLength / Float32Array.BYTES_PER_ELEMENT;
      tensorShape = [length];
    } else {
      tensorShape = [length];
    }
    if (!Number.isSafeInteger(length) || length <= 0) {
      throw new Error(`invalid Tensor.fromNativeBuffer length: ${length}`);
    }
    return new TensorCtor(buffer.readFloat32(length, byteOffset), tensorShape, tensorOptions);
  }

  return Object.freeze({
    initialize,
    tensor,
    parameter,
    param: parameter,
    toNativeBuffer,
    nativePlacement,
    place,
    fromNativeBuffer,
    cat: (tensors: unknown, dim = 0) => tensorJoin.cat(tensors, dim),
    stack: (tensors: unknown, dim = 0) => tensorJoin.stack(tensors, dim),
    einsum: (equation: unknown, tensors: unknown, ...moreTensors: unknown[]) => tensorJoin.einsum(equation, tensors, ...moreTensors),
    full: (shape: unknown, value: number, options: TensorOptions = {}) => tensorFactory.full(shape, value, options),
    empty: (shape: unknown, options: TensorOptions = {}) => tensorFactory.empty(shape, options),
    zeros: (shape: unknown, options: TensorOptions = {}) => tensorFactory.zeros(shape, options),
    ones: (shape: unknown, options: TensorOptions = {}) => tensorFactory.ones(shape, options),
    eye: (size: unknown, options: TensorOptions = {}) => tensorFactory.eye(size, options),
    scalar: (value: number, options: TensorOptions = {}) => tensorFactory.scalar(value, options),
    rand: (shape: unknown, options: RandomUniformTensorOptions = {}) => tensorFactory.rand(shape, options),
    randn: (shape: unknown, options: RandomTensorOptions = {}) => tensorFactory.randn(shape, options),
    randInt: (first: unknown, second: unknown, third?: unknown, fourth?: RandomIntTensorOptions) => tensorFactory.randInt(first, second, third, fourth),
    randint: (first: unknown, second: unknown, third?: unknown, fourth?: RandomIntTensorOptions) => tensorFactory.randint(first, second, third, fourth),
    randPerm: (size: unknown, options: RandomIntTensorOptions = {}) => tensorFactory.randPerm(size, options),
    randperm: (size: unknown, options: RandomIntTensorOptions = {}) => tensorFactory.randperm(size, options),
    manualSeed: (seed: unknown) => tensorFactory.manualSeed(seed),
    manual_seed: (seed: unknown) => tensorFactory.manual_seed(seed),
    initialSeed: () => tensorFactory.initialSeed(),
    initial_seed: () => tensorFactory.initial_seed(),
    seededRng: (seed: unknown) => tensorFactory.seededRng(seed),
    linspace: (first: unknown, second: unknown, third: unknown, fourth?: TensorOptions) => tensorFactory.linspace(first, second, third, fourth),
    arange: (start: number, endOrOptions?: unknown, stepOrOptions?: unknown, maybeOptions?: unknown) => (
      tensorFactory.arange(start, endOrOptions, stepOrOptions, maybeOptions)
    ),
  });
}

export function createTensorNativeSurfaceHelpers<TTensor>(options: TensorNativeSurfaceHelpersOptions) {
  const { tensorFacade } = options;

  return Object.freeze({
    toNativeBuffer: (tensor: TTensor, nativeOptions?: unknown) => (
      tensorFacade.toNativeBuffer(tensor, nativeOptions)
    ),
    nativePlacement: (tensor: TTensor, nativeOptions?: unknown) => (
      tensorFacade.nativePlacement(tensor, nativeOptions)
    ),
    place: (tensor: TTensor, program: unknown, kind = "input", placeOptions?: unknown) => (
      tensorFacade.place(tensor, program, kind, placeOptions)
    ),
  });
}

export function normalizeShapeModuleShape(shape: unknown, name: string, allowInfer = false) {
  const dims = Number.isSafeInteger(shape) ? [shape] : shape;
  if (!Array.isArray(dims) || dims.length === 0) {
    throw new Error(`${name} must be an integer or non-empty shape array`);
  }
  let inferCount = 0;
  return dims.map((dim) => {
    if (dim === -1 && allowInfer) {
      inferCount += 1;
      if (inferCount > 1) throw new Error(`${name} can infer at most one dimension`);
      return dim;
    }
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`${name} dimensions must be positive safe integers${allowInfer ? " or one -1" : ""}, got ${dim}`);
    }
    return dim;
  });
}
