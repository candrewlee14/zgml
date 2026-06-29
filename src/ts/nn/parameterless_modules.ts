import { initializeModuleMode, normalizeModuleTrainingMode } from "../train/module_mode.js";
import type { CompileOptions } from "../public_api.js";
import {
  createParameterlessStateHooks,
  installParameterlessStateMethods,
  type ParameterlessStateHooks,
  type ParameterlessStateModule,
  type ParameterlessStatePrototype,
} from "./parameterless_state.js";
import {
  createSingleModuleCompileHooks,
  emptySingleModuleBindings,
  installSingleModuleCompileMethods,
  singleModuleCompile,
  singleModuleCompiledSpec,
  type SingleModuleCompileHooks,
  type SingleModuleCompileHooksInput,
  type SingleModuleCompilePrototype,
  type SingleModuleRecord,
} from "./single_module_compile.js";

type UnknownRecord = Record<string, unknown>;
type TensorConstructOptions = Readonly<Record<string, unknown>>;
type ParameterlessTensor = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly length: number;
  readonly requiresGrad: boolean;
  readonly grad: Float32Array | null;
  [method: string]: unknown;
};
type DropoutBackwardTensor = ParameterlessTensor & {
  _backward?: (grad: Float32Array | null | undefined) => void;
};
type ActivationTensor = ParameterlessTensor & Readonly<{
  gelu: () => unknown;
  relu: () => unknown;
  silu: () => unknown;
  sigmoid: () => unknown;
  tanh: () => unknown;
  exp: () => unknown;
  log: () => unknown;
  neg: () => unknown;
  recip: () => unknown;
  abs: () => unknown;
  sgn: () => unknown;
  step: () => unknown;
  sqrt: () => unknown;
  square: () => unknown;
}>;
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: TensorConstructOptions) => ParameterlessTensor;
type F32 = (values: unknown) => Float32Array;
type TensorGradAdder = (tensor: ParameterlessTensor, grad: Float32Array) => void;
type GradModeEnabled = () => boolean;
type NativeEagerSoftmaxInto = (
  output: Float32Array,
  input: ParameterlessTensor,
  options: Readonly<{
    rows?: number;
    cols?: number;
    dim?: number;
    logSoftmax?: boolean;
  }>,
) => Float32Array;
type ParameterlessMethodHooks = ParameterlessStateHooks & SingleModuleCompileHooks;
type ParameterlessMethodPrototype = ParameterlessStatePrototype & SingleModuleCompilePrototype;
type ParameterlessParameterLabel = (module: ParameterlessStateModule & SingleModuleRecord) => unknown;
type TrainableParameterlessModule = ParameterlessStateModule & {
  train?: (mode?: boolean) => unknown;
};
type DropoutModulePrototype = SingleModuleCompilePrototype & {
  bindParameters?: (this: SingleModuleRecord, options?: CompileOptions) => unknown;
  compile?: (this: SingleModuleRecord, options?: CompileOptions) => unknown;
};

export type ParameterlessModuleClassHooks = SingleModuleCompileHooksInput & {
  Tensor: TensorConstructor;
  f32: F32;
};

export type SoftmaxModuleClassExtras = {
  kind: "softmax" | "logSoftmax";
  tensorMethod: string;
  parameterLabel?: string;
};

export type DropoutModuleClassExtras = {
  addTensorGrad: TensorGradAdder;
  isGradEnabled?: GradModeEnabled;
};

export type ActivationModuleClassOptions = Readonly<Record<string, unknown> & ParameterlessModuleClassHooks>;

export type SoftmaxModuleClassOptions = Readonly<Record<string, unknown> & ParameterlessModuleClassHooks & SoftmaxModuleClassExtras & {
  nativeEagerSoftmaxInto?: NativeEagerSoftmaxInto;
  isGradEnabled?: GradModeEnabled;
}>;

export type ReductionModuleClassOptions = Readonly<Record<string, unknown> & ParameterlessModuleClassHooks>;

export type DropoutModuleClassOptions = Readonly<Record<string, unknown> & ParameterlessModuleClassHooks & DropoutModuleClassExtras>;

function parameterlessModuleHooks(options: ParameterlessModuleClassHooks, label: string) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const stateHooks = createParameterlessStateHooks(options, label);
  const compileHooks = createSingleModuleCompileHooks(options, label, { requirePackParameters: false });
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function"
  ) {
    throw new Error(`${label} factory requires Tensor, f32, state, analysis, placement, and compile hooks`);
  }
  return { TensorClass, f32, ...stateHooks, ...compileHooks };
}

function installParameterlessMethods(proto: ParameterlessMethodPrototype, hooks: ParameterlessMethodHooks, parameterLabel: ParameterlessParameterLabel) {
  installParameterlessStateMethods(proto, hooks, parameterLabel);
  installSingleModuleCompileMethods(proto, hooks, {
    bindParameters: emptySingleModuleBindings,
  });
}

export function createActivationModuleClass(options: ActivationModuleClassOptions) {
  const hooks = parameterlessModuleHooks(options, "ActivationModule");
  class ActivationModule {
    kind: string;
    fn: (value: number) => number;
    training?: boolean;

    constructor(kind: string, fn: (value: number) => number) {
      initializeModuleMode(this);
      this.kind = kind;
      this.fn = fn;
    }

    forward(inputValues: unknown) {
      if (inputValues instanceof hooks.TensorClass) {
        const tensor = inputValues as ActivationTensor;
        switch (this.kind) {
          case "gelu": return tensor.gelu();
          case "relu": return tensor.relu();
          case "silu": return tensor.silu();
          case "sigmoid": return tensor.sigmoid();
          case "tanh": return tensor.tanh();
          case "exp": return tensor.exp();
          case "log": return tensor.log();
          case "neg": return tensor.neg();
          case "recip": return tensor.recip();
          case "abs": return tensor.abs();
          case "sgn": return tensor.sgn();
          case "step": return tensor.step();
          case "sqrt": return tensor.sqrt();
          case "square": return tensor.square();
          default: throw new Error(`unsupported activation module kind: ${this.kind}`);
        }
      }
      const input = hooks.f32(inputValues);
      const out = new Float32Array(input.length);
      for (let i = 0; i < input.length; i += 1) out[i] = this.fn(input[i]);
      return out;
    }
  }
  installParameterlessMethods(ActivationModule.prototype, hooks, (module) => module.kind);
  return ActivationModule;
}

export function createSoftmaxModuleClass(options: SoftmaxModuleClassOptions) {
  const hooks = parameterlessModuleHooks(options, "SoftmaxModule");
  const kind = options.kind;
  const tensorMethod = options.tensorMethod;
  const parameterLabel = options.parameterLabel || kind;
  const nativeEagerSoftmaxInto = options.nativeEagerSoftmaxInto;
  const isGradEnabled = typeof options.isGradEnabled === "function" ? options.isGradEnabled : () => true;
  if ((kind !== "softmax" && kind !== "logSoftmax") || typeof tensorMethod !== "string") {
    throw new Error("SoftmaxModule factory requires kind and tensorMethod");
  }

  function tryNativeEagerSoftmax(input: ParameterlessTensor, dimValue: unknown) {
    if (typeof nativeEagerSoftmaxInto !== "function") return null;
    if (isGradEnabled()) return null;
    const shape = Array.isArray(input.shape) ? input.shape : null;
    if (!shape || shape.length === 0) return null;
    const dim = Number(dimValue ?? -1);
    const rank = shape.length;
    const normalizedDim = dim < 0 ? rank + dim : dim;
    if (!Number.isSafeInteger(dim) || normalizedDim !== rank - 1) return null;
    const cols = Number(shape[rank - 1]);
    if (!Number.isSafeInteger(cols) || cols <= 0) return null;
    const rows = shape.slice(0, -1).reduce((acc, value) => acc * Number(value), 1) || 1;
    if (!Number.isSafeInteger(rows) || rows <= 0) return null;
    const output = new Float32Array(rows * cols);
    nativeEagerSoftmaxInto(output, input, {
      rows,
      cols,
      dim,
      logSoftmax: kind === "logSoftmax",
    });
    return new hooks.TensorClass(output, input.shape);
  }

  class SoftmaxLikeModule {
    kind: string;
    dim: unknown;
    training?: boolean;

    constructor(dim: unknown = -1) {
      initializeModuleMode(this);
      this.kind = kind;
      this.dim = dim;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof hooks.TensorClass ? inputValues : new hooks.TensorClass(hooks.f32(inputValues));
      const native = tryNativeEagerSoftmax(input, this.dim);
      if (native !== null) return native;
      const method = input[tensorMethod];
      if (typeof method !== "function") throw new Error(`tensor method ${tensorMethod} is unavailable`);
      return method.call(input, this.dim);
    }
  }
  installParameterlessMethods(SoftmaxLikeModule.prototype, hooks, () => parameterLabel);
  return SoftmaxLikeModule;
}

export function createReductionModuleClass(options: ReductionModuleClassOptions) {
  const hooks = parameterlessModuleHooks(options, "ReductionModule");
  const reductions = Object.freeze({
    sum: "sumDim",
    mean: "meanDim",
    prod: "prodDim",
    max: "maxDim",
    min: "minDim",
    argmax: "argmaxDim",
    argmin: "argminDim",
  } as Record<string, string>);

  class ReductionModule {
    kind: string;
    dim: unknown;
    training?: boolean;

    constructor(kind: string, dim: unknown = -1) {
      initializeModuleMode(this);
      this.kind = kind;
      this.dim = dim;
      if (!reductions[kind]) throw new Error(`unsupported reduction module kind: ${kind}`);
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof hooks.TensorClass ? inputValues : new hooks.TensorClass(hooks.f32(inputValues));
      const tensorMethod = reductions[this.kind];
      const method = input[tensorMethod];
      if (typeof method !== "function") throw new Error(`tensor method ${tensorMethod} is unavailable`);
      return method.call(input, this.dim);
    }
  }
  installParameterlessMethods(ReductionModule.prototype, hooks, (module) => module.kind);
  return ReductionModule;
}

export function createDropoutModuleClass(options: DropoutModuleClassOptions) {
  const hooks = parameterlessModuleHooks(options, "DropoutModule");
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : () => true;
  if (typeof addTensorGrad !== "function") {
    throw new Error("DropoutModule factory requires addTensorGrad");
  }

  function dropoutProbability(value: unknown) {
    const p = value === undefined ? 0.5 : Number(value);
    if (!Number.isFinite(p) || p < 0 || p > 1) {
      throw new Error(`dropout probability must be between 0 and 1, got ${value}`);
    }
    return p;
  }

  function dropoutRng(config: UnknownRecord) {
    if (!config || config.rng === undefined) return Math.random;
    if (typeof config.rng !== "function") throw new Error("dropout rng must be a function");
    return config.rng as () => number;
  }

  class DropoutModule {
    kind: "dropout";
    p: number;
    rng: () => number;
    training?: boolean;

    constructor(p: unknown = 0.5, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "dropout";
      this.p = dropoutProbability(p);
      this.rng = dropoutRng(config);
      if (config.training !== undefined) {
        (this as TrainableParameterlessModule).train?.(normalizeModuleTrainingMode(config.training as boolean | undefined));
      }
    }

    forward(inputValues: unknown) {
      if (!(inputValues instanceof hooks.TensorClass)) {
        const input = hooks.f32(inputValues);
        if (!this.training || this.p === 0) return new Float32Array(input);
        const out = new Float32Array(input.length);
        const scale = this.p === 1 ? 0 : 1 / (1 - this.p);
        for (let i = 0; i < input.length; i += 1) {
          const keep = this.rng() >= this.p;
          out[i] = input[i] * (keep ? scale : 0);
        }
        return out;
      }

      const input = inputValues as ParameterlessTensor;
      if (!this.training || this.p === 0) return input;

      const source = input.data;
      const out = new Float32Array(source.length);
      const mask = new Float32Array(source.length);
      const scale = this.p === 1 ? 0 : 1 / (1 - this.p);
      for (let i = 0; i < source.length; i += 1) {
        const keep = this.rng() >= this.p;
        mask[i] = keep ? scale : 0;
        out[i] = source[i] * mask[i];
      }
      const needsGrad = gradModeEnabled() && input.requiresGrad;
      const result = new hooks.TensorClass(out, input.shape, {
        requiresGrad: needsGrad,
        prev: needsGrad ? [input] : [],
      });
      (result as DropoutBackwardTensor)._backward = (grad: Float32Array | null | undefined) => {
        if (!grad || !input.requiresGrad) return;
        const inGrad = new Float32Array(input.length);
        for (let i = 0; i < inGrad.length; i += 1) inGrad[i] = grad[i] * mask[i];
        addTensorGrad(input, inGrad);
      };
      return result;
    }
  }
  installParameterlessMethods(DropoutModule.prototype, hooks, () => "dropout");
  const dropoutProto = DropoutModule.prototype as DropoutModulePrototype;
  dropoutProto.bindParameters = function bindParameters(this: SingleModuleRecord, options: CompileOptions = {}) {
    const compiled = singleModuleCompiledSpec(this, hooks, options);
    if (compiled && compiled.kind === "module") {
      return emptySingleModuleBindings(this, options);
    }
    const support = this.compileSupport?.(options);
    throw new Error(support && typeof support.reason === "string" ? support.reason : "nn.Dropout does not have a native Program path");
  };
  dropoutProto.compile = function compile(this: SingleModuleRecord, options: CompileOptions = {}) {
    return singleModuleCompile(this, hooks, options, "nn.Dropout does not have a native Program path");
  };
  return DropoutModule;
}
