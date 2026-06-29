import { normalizeShapeModuleShape } from "../core/tensor_facade.js";
import { initializeModuleMode } from "../train/module_mode.js";
import {
  createParameterlessStateHooks,
  installParameterlessStateMethods,
} from "./parameterless_state.js";
import {
  createSingleModuleCompileHooks,
  installSingleModuleCompileMethods,
  packedSingleModuleBindings,
  type SingleModuleCompileHooksInput,
} from "./single_module_compile.js";

export type ShapeTensorConstructOptions = Readonly<Record<string, unknown>>;
export type ShapeTensor = {
  readonly data?: Float32Array;
  readonly shape?: readonly number[];
  reshape(shape: readonly number[]): ShapeTensor;
  view(shape: readonly number[]): ShapeTensor;
  flatten(startDim?: number, endDim?: number): ShapeTensor;
  squeeze(dim?: number): ShapeTensor;
  unsqueeze(dim: number): ShapeTensor;
  broadcastTo(shape: readonly number[]): ShapeTensor;
  expand(shape: readonly number[]): ShapeTensor;
  repeat(repeats: readonly number[]): ShapeTensor;
  tile(repeats: readonly number[]): ShapeTensor;
  narrow(dim: number, start: number, length: number): ShapeTensor;
  select(dim: number, index: number): ShapeTensor;
  slice(dim: number, start: number, end: number | null, step: number): ShapeTensor;
  transpose(dim0: number, dim1: number): ShapeTensor;
  permute(dims: readonly number[]): ShapeTensor;
  diagonal(): ShapeTensor;
};
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: ShapeTensorConstructOptions) => ShapeTensor;
type F32 = (values: unknown) => Float32Array;

export type ShapeModuleClassHooks = SingleModuleCompileHooksInput & {
  Tensor: TensorConstructor;
  f32: F32;
};

export type ShapeModuleClassOptions = Readonly<Record<string, unknown> & ShapeModuleClassHooks>;

function normalizePermuteDims(dims: unknown) {
  if (!Array.isArray(dims) || dims.length === 0) {
    throw new Error("permute dims must be a non-empty axis array");
  }
  return dims.map((dim) => {
    const axis = Number(dim);
    if (!Number.isSafeInteger(axis)) throw new Error(`permute dims must be safe integers, got ${dim}`);
    return axis;
  });
}

export function createShapeModuleClass(options: ShapeModuleClassOptions) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const stateHooks = createParameterlessStateHooks(options, "ShapeModule");
  const compileHooks = createSingleModuleCompileHooks(options, "ShapeModule");
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function"
  ) {
    throw new Error("ShapeModule factory requires Tensor, f32, state, analysis, packing, placement, and compile hooks");
  }

  class ShapeModule {
    kind: string;
    shape: readonly number[];
    startDim: number;
    endDim: number;
    dim: number;
    start: number;
    length: number;
    end: number | null;
    step: number;
    index: number;
    dim0?: number;
    dim1?: number;
    dims: readonly number[];
    squeezeAll?: boolean;
    training?: boolean;

    constructor(kind: string, shapeOrStartDim?: unknown, maybeEndDim?: unknown, maybeLength?: unknown, maybeStep?: unknown) {
      initializeModuleMode(this);
      this.kind = kind;
      this.dims = [];
      if (kind === "flatten") {
        this.shape = [];
        this.startDim = shapeOrStartDim === undefined ? 0 : Number(shapeOrStartDim);
        this.endDim = maybeEndDim === undefined ? -1 : Number(maybeEndDim);
        this.dim = 0;
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        this.index = 0;
        if (!Number.isSafeInteger(this.startDim)) throw new Error(`flatten startDim must be a safe integer, got ${this.startDim}`);
        if (!Number.isSafeInteger(this.endDim)) throw new Error(`flatten endDim must be a safe integer, got ${this.endDim}`);
      } else if (kind === "squeeze") {
        this.shape = [];
        this.startDim = 0;
        this.endDim = -1;
        this.squeezeAll = shapeOrStartDim === undefined || shapeOrStartDim === null;
        this.dim = this.squeezeAll ? 0 : Number(shapeOrStartDim);
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        this.index = 0;
        if (!this.squeezeAll && !Number.isSafeInteger(this.dim)) throw new Error(`squeeze dim must be a safe integer, got ${this.dim}`);
      } else if (kind === "unsqueeze") {
        this.shape = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = Number(shapeOrStartDim);
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        this.index = 0;
        if (!Number.isSafeInteger(this.dim)) throw new Error(`unsqueeze dim must be a safe integer, got ${this.dim}`);
      } else if (kind === "narrow") {
        this.shape = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = Number(shapeOrStartDim ?? 0);
        this.start = Number(maybeEndDim ?? 0);
        this.length = Number(maybeLength);
        this.end = null;
        this.step = 1;
        this.index = 0;
        if (!Number.isSafeInteger(this.dim)) throw new Error(`narrow dim must be a safe integer, got ${this.dim}`);
        if (!Number.isSafeInteger(this.start)) throw new Error(`narrow start must be a safe integer, got ${this.start}`);
        if (!Number.isSafeInteger(this.length) || this.length <= 0) throw new Error(`narrow length must be a positive safe integer, got ${this.length}`);
      } else if (kind === "select") {
        this.shape = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = Number(shapeOrStartDim ?? 0);
        this.index = Number(maybeEndDim ?? 0);
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        if (!Number.isSafeInteger(this.dim)) throw new Error(`select dim must be a safe integer, got ${this.dim}`);
        if (!Number.isSafeInteger(this.index)) throw new Error(`select index must be a safe integer, got ${this.index}`);
      } else if (kind === "slice") {
        this.shape = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = Number(shapeOrStartDim ?? 0);
        this.start = Number(maybeEndDim ?? 0);
        this.end = maybeLength === undefined || maybeLength === null ? null : Number(maybeLength);
        this.step = maybeStep === undefined ? 1 : Number(maybeStep);
        this.length = 0;
        this.index = 0;
        if (!Number.isSafeInteger(this.dim)) throw new Error(`slice dim must be a safe integer, got ${this.dim}`);
        if (!Number.isSafeInteger(this.start)) throw new Error(`slice start must be a safe integer, got ${this.start}`);
        if (this.end !== null && !Number.isSafeInteger(this.end)) throw new Error(`slice end must be a safe integer or null, got ${this.end}`);
        if (!Number.isSafeInteger(this.step) || this.step <= 0) throw new Error(`slice step must be a positive safe integer, got ${this.step}`);
      } else if (kind === "transpose") {
        this.shape = [];
        this.dims = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = 0;
        this.start = 0;
        this.length = 0;
        this.index = 0;
        this.end = null;
        this.step = 1;
        this.dim0 = shapeOrStartDim === undefined ? 0 : Number(shapeOrStartDim);
        this.dim1 = maybeEndDim === undefined ? 1 : Number(maybeEndDim);
        if (!Number.isSafeInteger(this.dim0)) throw new Error(`transpose dim0 must be a safe integer, got ${this.dim0}`);
        if (!Number.isSafeInteger(this.dim1)) throw new Error(`transpose dim1 must be a safe integer, got ${this.dim1}`);
      } else if (kind === "permute") {
        this.shape = [];
        this.dims = normalizePermuteDims(shapeOrStartDim);
        this.startDim = 0;
        this.endDim = -1;
        this.dim = 0;
        this.start = 0;
        this.length = 0;
        this.index = 0;
        this.end = null;
        this.step = 1;
      } else if (kind === "identity" || kind === "diagonal") {
        this.shape = [];
        this.dims = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = 0;
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        this.index = 0;
      } else if (kind === "reshape" || kind === "view" || kind === "broadcastTo" || kind === "expand" || kind === "repeat" || kind === "tile") {
        this.shape = normalizeShapeModuleShape(shapeOrStartDim, `${kind} shape`, kind === "reshape" || kind === "view");
        this.dims = [];
        this.startDim = 0;
        this.endDim = -1;
        this.dim = 0;
        this.start = 0;
        this.length = 0;
        this.end = null;
        this.step = 1;
        this.index = 0;
      } else {
        throw new Error(`unsupported shape module kind: ${kind}`);
      }
      if (this.dim0 === undefined) this.dim0 = 0;
      if (this.dim1 === undefined) this.dim1 = 1;
      if (this.squeezeAll === undefined) this.squeezeAll = false;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof TensorClass ? inputValues : new TensorClass(f32(inputValues));
      switch (this.kind) {
        case "identity": return input;
        case "reshape": return input.reshape(this.shape);
        case "view": return input.view(this.shape);
        case "flatten": return input.flatten(this.startDim, this.endDim);
        case "squeeze": return this.squeezeAll ? input.squeeze() : input.squeeze(this.dim);
        case "unsqueeze": return input.unsqueeze(this.dim);
        case "broadcastTo": return input.broadcastTo(this.shape);
        case "expand": return input.expand(this.shape);
        case "repeat": return input.repeat(this.shape);
        case "tile": return input.tile(this.shape);
        case "narrow": return input.narrow(this.dim, this.start, this.length);
        case "select": return input.select(this.dim, this.index);
        case "slice": return input.slice(this.dim, this.start, this.end, this.step);
        case "transpose": return input.transpose(this.dim0 ?? 0, this.dim1 ?? 1);
        case "permute": return input.permute(this.dims);
        case "diagonal": return input.diagonal();
        default: throw new Error(`unsupported shape module kind: ${this.kind}`);
      }
    }
  }

  installParameterlessStateMethods(ShapeModule.prototype, stateHooks, (module) => module.kind);
  installSingleModuleCompileMethods(ShapeModule.prototype, compileHooks, {
    bindParameters: packedSingleModuleBindings,
  });
  return ShapeModule;
}
