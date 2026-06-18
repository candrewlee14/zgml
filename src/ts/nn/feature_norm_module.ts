import { initializeModuleMode } from "../train/module_mode.js";
import type { CompileOptions } from "../public_api.js";
import {
  leafChildren,
  leafNamedChildren,
  leafNamedModules,
  moduleTraversalOptions,
  moduleListSelf,
} from "./module_tree.js";
import {
  createStatefulModuleStateHooks,
  installStatefulModuleStateMethods,
} from "./stateful_module_state.js";
import {
  createSingleModuleCompileHooks,
  emptySingleModuleBindings,
  installSingleModuleCompileMethods,
  singleModuleCompileSupport,
  type SingleModuleCompileHooks,
  type SingleModuleCompileHooksInput,
  type SingleModuleRecord,
} from "./single_module_compile.js";
import { annotateModuleBindings } from "../runtime/module_bindings.js";

type UnknownRecord = Record<string, unknown>;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type ModuleCompileOptions = Readonly<CompileOptions & Record<string, unknown>>;
export type FeatureNormTensorConstructOptions = Readonly<Record<string, unknown>>;
export type FeatureNormTensor = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly length: number;
  readonly grad?: Float32Array | null;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
  zeroGrad(options?: Readonly<Record<string, unknown>>): void;
};
export type FeatureNormGradTensor = FeatureNormTensor & {
  readonly grad: Float32Array | null;
  readonly requiresGrad: boolean;
};
export type FeatureNormParameter = {
  readonly name: string;
  readonly data: Float32Array;
  readonly grad?: Float32Array | null;
  readonly layout?: string;
  tensor: FeatureNormTensor;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
};
type FeatureNormBindableModule = Readonly<{
  kind?: unknown;
  features?: unknown;
  eps?: unknown;
  training?: unknown;
  weight: Float32Array | null;
  bias: Float32Array | null;
  runningMean?: FeatureNormBuffer | null;
  runningVar?: FeatureNormBuffer | null;
}> & SingleModuleRecord;
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: FeatureNormTensorConstructOptions) => FeatureNormTensor;
type F32 = (values: unknown) => Float32Array;
type TensorGradAdder = (tensor: FeatureNormGradTensor, grad: Float32Array) => void;
type RequirePositiveInteger = BivariantCallback<[value: unknown, label: string], number>;
type DefaultedF32 = (values: unknown, length: number, label: string, fallback: (length: number) => Float32Array, shape?: readonly number[]) => Float32Array;
type ParameterFactory = (name: string, values: Float32Array, shape: readonly number[], layout: string) => FeatureNormParameter;
type ParameterView = BivariantCallback<[prefix: string, parameter: FeatureNormParameter], FeatureNormParameter>;
type FeatureNormKind = "layerNorm" | "rmsNorm" | "batchNorm1d";
type FeatureNormBuffer = Readonly<{
  kind: "buffer";
  name: string;
  data: Float32Array;
  shape: readonly number[];
  layout: string;
  persistent: boolean;
}>;
type StateDictLike = Record<string, unknown> | Map<string, unknown>;

function featureNormGradTensor(tensor: FeatureNormTensor): FeatureNormGradTensor {
  return tensor as FeatureNormGradTensor;
}

function featureNormBindableModule(module: SingleModuleRecord): FeatureNormBindableModule {
  return module as FeatureNormBindableModule;
}

function batchNormEvalBindings(module: FeatureNormBindableModule) {
  if (
    module.kind !== "batchNorm1d" ||
    module.training === true ||
    !module.runningMean ||
    !module.runningVar ||
    !module.weight ||
    !module.bias
  ) {
    return null;
  }
  const features = Number(module.features);
  if (!Number.isSafeInteger(features) || features <= 0) return null;
  const eps = Number(module.eps ?? 1e-5);
  const weights = new Float32Array(features);
  const bias = new Float32Array(features);
  for (let i = 0; i < features; i += 1) {
    const scale = module.weight[i] / Math.sqrt(module.runningVar.data[i] + eps);
    weights[i] = scale;
    bias[i] = module.bias[i] - module.runningMean.data[i] * scale;
  }
  return { weights, bias };
}

function onesF32(length: number) {
  const values = new Float32Array(length);
  values.fill(1);
  return values;
}

function statePrefix(prefixOrOptions: unknown = "") {
  if (typeof prefixOrOptions === "string") return prefixOrOptions;
  const record = prefixOrOptions as UnknownRecord | null | undefined;
  return record && typeof record.prefix === "string" ? record.prefix : "";
}

function prefixedName(prefix: string, name: string) {
  return prefix ? `${prefix}.${name}` : name;
}

function stateKeys(source: unknown) {
  if (source instanceof Map) return Array.from(source.keys(), String);
  if (source && typeof source === "object") return Object.keys(source);
  throw new Error("state dict must be an object or Map");
}

function stateValue(source: unknown, name: string) {
  if (source instanceof Map) return source.get(name);
  if (source && typeof source === "object" && Object.prototype.hasOwnProperty.call(source, name)) {
    return (source as UnknownRecord)[name];
  }
  return undefined;
}

function stateData(value: unknown) {
  const record = value as UnknownRecord | null | undefined;
  return record && record.data !== undefined ? record.data : value;
}

function stateShape(value: unknown) {
  const record = value as UnknownRecord | null | undefined;
  if (record && Array.isArray(record.shape)) return Array.from(record.shape, Number);
  return null;
}

function stateF32(value: unknown, length: number, label: string) {
  const raw = stateData(value);
  if (raw instanceof Float32Array) {
    if (raw.length !== length) throw new Error(`${label} length must be ${length}, got ${raw.length}`);
    return raw;
  }
  if (Array.isArray(raw)) {
    if (raw.length !== length) throw new Error(`${label} length must be ${length}, got ${raw.length}`);
    return Float32Array.from(raw, Number);
  }
  throw new Error(`${label} must contain Float32Array or numeric array data`);
}

function stateEntryForBuffer(targetBuffer: FeatureNormBuffer) {
  const shape = Object.freeze(Array.from(targetBuffer.shape));
  const entry = {
    shape,
    layout: targetBuffer.layout,
    data: new Float32Array(targetBuffer.data),
  } as UnknownRecord;
  entry.signature = [
    "module-buffer-state-entry",
    `shape=${shape.join("x")}`,
    `layout=${entry.layout}`,
    `len=${targetBuffer.data.length}`,
  ].join("|");
  return Object.freeze(entry);
}

export type FeatureNormModuleClassOptions = Readonly<Record<string, unknown> & SingleModuleCompileHooksInput & {
  Tensor: TensorConstructor;
  f32: F32;
  addTensorGrad: TensorGradAdder;
  requirePositiveInteger: RequirePositiveInteger;
  defaultedF32: DefaultedF32;
  zerosF32: (length: number) => Float32Array;
  makeParameter: ParameterFactory;
  parameterView: ParameterView;
}>;

export function createFeatureNormModuleClass(options: FeatureNormModuleClassOptions) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : () => true;
  const requirePositiveInteger = options.requirePositiveInteger;
  const defaultedF32 = options.defaultedF32;
  const zerosF32 = options.zerosF32;
  const makeParameter = options.makeParameter;
  const parameterView = options.parameterView;
  const stateHooks = createStatefulModuleStateHooks(options, "FeatureNormModule");
  const compileHooks = createSingleModuleCompileHooks(options, "FeatureNormModule", { requirePackParameters: false });
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function" ||
    typeof addTensorGrad !== "function" ||
    typeof requirePositiveInteger !== "function" ||
    typeof defaultedF32 !== "function" ||
    typeof zerosF32 !== "function" ||
    typeof makeParameter !== "function" ||
    typeof parameterView !== "function"
  ) {
    throw new Error("FeatureNormModule factory requires tensor, state, analysis, placement, and compile hooks");
  }

  class FeatureNormModule {
    kind: FeatureNormKind;
    features: number;
    eps: number;
    momentum: number;
    trackRunningStats: boolean;
    weight: Float32Array | null;
    bias: Float32Array | null;
    weightParam: FeatureNormParameter | null;
    biasParam: FeatureNormParameter | null;
    runningMean: FeatureNormBuffer | null;
    runningVar: FeatureNormBuffer | null;
    numBatchesTracked: FeatureNormBuffer | null;
    training?: boolean;

    constructor(kind: FeatureNormKind, features: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = kind;
      this.features = requirePositiveInteger(features, `${kind} features`);
      this.eps = (config.eps as number | undefined) ?? 1e-5;
      this.momentum = Number(config.momentum ?? 0.1);
      if (!Number.isFinite(this.momentum) || this.momentum < 0 || this.momentum > 1) {
        throw new Error(`${kind} momentum must be between 0 and 1, got ${config.momentum}`);
      }
      this.trackRunningStats = kind === "batchNorm1d" && config.trackRunningStats !== false && config.track_running_stats !== false;
      const affine = config.affine !== false;
      this.weight = affine
        ? defaultedF32(config.weight ?? config.weights, this.features, `${kind} weight`, onesF32, [this.features])
        : null;
      this.bias = kind !== "rmsNorm" && affine && config.bias !== false
        ? defaultedF32(config.biasValues ?? (config.bias === undefined || config.bias === true ? undefined : config.bias), this.features, `${kind} bias`, zerosF32, [this.features])
        : null;
      this.weightParam = this.weight ? makeParameter("weight", this.weight, [this.features], `row-major:${this.kind}.weight[features]`) : null;
      this.biasParam = this.bias ? makeParameter("bias", this.bias, [this.features], `row-major:${this.kind}.bias[features]`) : null;
      this.runningMean = this.trackRunningStats
        ? this.makeBuffer("runningMean", defaultedF32(config.runningMean ?? config.running_mean, this.features, `${kind} runningMean`, zerosF32, [this.features]))
        : null;
      this.runningVar = this.trackRunningStats
        ? this.makeBuffer("runningVar", defaultedF32(config.runningVar ?? config.running_var, this.features, `${kind} runningVar`, onesF32, [this.features]))
        : null;
      this.numBatchesTracked = this.trackRunningStats
        ? this.makeBuffer("numBatchesTracked", Float32Array.of(Number(config.numBatchesTracked ?? config.num_batches_tracked ?? 0)), [1])
        : null;
    }

    makeBuffer(name: string, data: Float32Array, shape: readonly number[] = [this.features]): FeatureNormBuffer {
      return Object.freeze({
        kind: "buffer",
        name,
        data,
        shape: Object.freeze(Array.from(shape)),
        layout: `row-major:${this.kind}.${name}[features]`,
        persistent: true,
      });
    }

    batchNormStats(input: Float32Array, batch: number) {
      const mean = new Float32Array(this.features);
      const variance = new Float32Array(this.features);
      for (let b = 0; b < batch; b += 1) {
        const base = b * this.features;
        for (let f = 0; f < this.features; f += 1) mean[f] += input[base + f];
      }
      for (let f = 0; f < this.features; f += 1) mean[f] /= batch;
      for (let b = 0; b < batch; b += 1) {
        const base = b * this.features;
        for (let f = 0; f < this.features; f += 1) {
          const centered = input[base + f] - mean[f];
          variance[f] += centered * centered;
        }
      }
      for (let f = 0; f < this.features; f += 1) variance[f] /= batch;
      return { mean, variance };
    }

    updateRunningStats(mean: Float32Array, variance: Float32Array, batch: number) {
      if (!this.runningMean || !this.runningVar || !this.numBatchesTracked) return;
      this.numBatchesTracked.data[0] += 1;
      const unbiasedScale = batch > 1 ? batch / (batch - 1) : 1;
      for (let f = 0; f < this.features; f += 1) {
        this.runningMean.data[f] = (1 - this.momentum) * this.runningMean.data[f] + this.momentum * mean[f];
        this.runningVar.data[f] = (1 - this.momentum) * this.runningVar.data[f] + this.momentum * variance[f] * unbiasedScale;
      }
    }

    forwardBatchNorm(input: Float32Array, inputValues: unknown) {
      const batch = input.length / this.features;
      const useBatchStats = this.training || !this.runningMean || !this.runningVar;
      const stats = useBatchStats ? this.batchNormStats(input, batch) : null;
      const mean = stats?.mean ?? this.runningMean!.data;
      const variance = stats?.variance ?? this.runningVar!.data;
      if (this.training && stats) this.updateRunningStats(stats.mean, stats.variance, batch);

      const out = new Float32Array(input.length);
      for (let b = 0; b < batch; b += 1) {
        const base = b * this.features;
        for (let f = 0; f < this.features; f += 1) {
          let value = (input[base + f] - mean[f]) / Math.sqrt(variance[f] + this.eps);
          if (this.weight) value *= this.weight[f];
          if (this.bias) value += this.bias[f];
          out[base + f] = value;
        }
      }
      if (!(inputValues instanceof TensorClass)) return out;

      const inputTensor = inputValues as FeatureNormTensor;
      const needsGrad = gradModeEnabled() && (
        inputTensor.requiresGrad ||
        Boolean(this.weightParam?.tensor.requiresGrad) ||
        Boolean(this.biasParam?.tensor.requiresGrad)
      );
      const prev = [];
      if (needsGrad && inputTensor.requiresGrad) prev.push(inputTensor);
      if (needsGrad && this.weightParam?.tensor.requiresGrad) prev.push(this.weightParam.tensor);
      if (needsGrad && this.biasParam?.tensor.requiresGrad) prev.push(this.biasParam.tensor);
      return new TensorClass(out, inputTensor.shape, {
        requiresGrad: needsGrad,
        prev,
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad) return;
          const inputGrad = inputTensor.requiresGrad ? new Float32Array(input.length) : null;
          const weightGrad = this.weightParam?.tensor.requiresGrad ? new Float32Array(this.features) : null;
          const biasGrad = this.biasParam?.tensor.requiresGrad ? new Float32Array(this.features) : null;
          for (let f = 0; f < this.features; f += 1) {
            const inv = 1 / Math.sqrt(variance[f] + this.eps);
            const gamma = this.weight ? this.weight[f] : 1;
            let sumDy = 0;
            let sumDyNorm = 0;
            for (let b = 0; b < batch; b += 1) {
              const offset = b * this.features + f;
              const norm = (input[offset] - mean[f]) * inv;
              const dy = grad[offset];
              if (weightGrad) weightGrad[f] += dy * norm;
              if (biasGrad) biasGrad[f] += dy;
              const dyNorm = dy * gamma;
              sumDy += dyNorm;
              sumDyNorm += dyNorm * norm;
            }
            if (inputGrad) {
              for (let b = 0; b < batch; b += 1) {
                const offset = b * this.features + f;
                const norm = (input[offset] - mean[f]) * inv;
                inputGrad[offset] += useBatchStats
                  ? (inv / batch) * (batch * grad[offset] * gamma - sumDy - norm * sumDyNorm)
                  : grad[offset] * gamma * inv;
              }
            }
          }
          if (inputGrad) addTensorGrad(featureNormGradTensor(inputTensor), inputGrad);
          if (weightGrad) addTensorGrad(featureNormGradTensor(this.weightParam!.tensor), weightGrad);
          if (biasGrad) addTensorGrad(featureNormGradTensor(this.biasParam!.tensor), biasGrad);
        },
      });
    }

    forward(inputValues: unknown) {
      const input = f32(inputValues);
      if (input.length % this.features !== 0) {
        throw new Error(`${this.kind} input length ${input.length} must be divisible by features ${this.features}`);
      }
      if (this.kind === "batchNorm1d") return this.forwardBatchNorm(input, inputValues);
      const out = new Float32Array(input.length);
      const batch = input.length / this.features;
      for (let b = 0; b < batch; b += 1) {
        const base = b * this.features;
        let mean = 0;
        if (this.kind === "layerNorm") {
          for (let f = 0; f < this.features; f += 1) mean += input[base + f];
          mean /= this.features;
        }
        let second = 0;
        for (let f = 0; f < this.features; f += 1) {
          const centered = input[base + f] - mean;
          second += centered * centered;
        }
        const scale = 1 / Math.sqrt(second / this.features + this.eps);
        for (let f = 0; f < this.features; f += 1) {
          let value = (input[base + f] - mean) * scale;
          if (this.weight) value *= this.weight[f];
          if (this.bias) value += this.bias[f];
          out[base + f] = value;
        }
      }
      if (!(inputValues instanceof TensorClass)) return out;

      const inputTensor = inputValues as FeatureNormTensor;
      const needsGrad = gradModeEnabled() && (
        inputTensor.requiresGrad ||
        Boolean(this.weightParam?.tensor.requiresGrad) ||
        Boolean(this.biasParam?.tensor.requiresGrad)
      );
      const prev = [];
      if (needsGrad && inputTensor.requiresGrad) prev.push(inputTensor);
      if (needsGrad && this.weightParam?.tensor.requiresGrad) prev.push(this.weightParam.tensor);
      if (needsGrad && this.biasParam?.tensor.requiresGrad) prev.push(this.biasParam.tensor);
      return new TensorClass(out, inputTensor.shape, {
        requiresGrad: needsGrad,
        prev,
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad) return;
          const inputGrad = inputTensor.requiresGrad ? new Float32Array(input.length) : null;
          const weightGrad = this.weightParam?.tensor.requiresGrad ? new Float32Array(this.features) : null;
          const biasGrad = this.biasParam?.tensor.requiresGrad ? new Float32Array(this.features) : null;
          for (let b = 0; b < batch; b += 1) {
            const base = b * this.features;
            let mean = 0;
            if (this.kind === "layerNorm") {
              for (let f = 0; f < this.features; f += 1) mean += input[base + f];
              mean /= this.features;
            }
            let second = 0;
            for (let f = 0; f < this.features; f += 1) {
              const centered = input[base + f] - mean;
              second += centered * centered;
            }
            const inv = 1 / Math.sqrt(second / this.features + this.eps);
            let sumDyNorm = 0;
            let sumDyNormTimesNorm = 0;
            for (let f = 0; f < this.features; f += 1) {
              const centered = input[base + f] - mean;
              const norm = centered * inv;
              const dy = grad[base + f];
              if (weightGrad) weightGrad[f] += dy * norm;
              if (biasGrad) biasGrad[f] += dy;
              const gamma = this.weight ? this.weight[f] : 1;
              const dyNorm = dy * gamma;
              sumDyNorm += dyNorm;
              sumDyNormTimesNorm += dyNorm * norm;
            }
            if (inputGrad) {
              for (let f = 0; f < this.features; f += 1) {
                const centered = input[base + f] - mean;
                const norm = centered * inv;
                const gamma = this.weight ? this.weight[f] : 1;
                const dyNorm = grad[base + f] * gamma;
                inputGrad[base + f] += this.kind === "layerNorm"
                  ? (inv / this.features) * (this.features * dyNorm - sumDyNorm - norm * sumDyNormTimesNorm)
                  : inv * (dyNorm - norm * (sumDyNormTimesNorm / this.features));
              }
            }
          }
          if (inputGrad) addTensorGrad(featureNormGradTensor(inputTensor), inputGrad);
          if (weightGrad) addTensorGrad(featureNormGradTensor(this.weightParam!.tensor), weightGrad);
          if (biasGrad) addTensorGrad(featureNormGradTensor(this.biasParam!.tensor), biasGrad);
        },
      });
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      const params = [];
      if (this.weightParam) params.push(parameterView(prefix, this.weightParam));
      if (this.biasParam) params.push(parameterView(prefix, this.biasParam));
      return params;
    }

    children() {
      return leafChildren();
    }

    modules() {
      return moduleListSelf(this);
    }

    namedChildren() {
      return leafNamedChildren();
    }

    namedModules(prefix = "") {
      return leafNamedModules(this, prefix);
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    runningBuffers() {
      return [this.runningMean, this.runningVar, this.numBatchesTracked]
        .filter((targetBuffer): targetBuffer is FeatureNormBuffer => Boolean(targetBuffer));
    }

    namedBuffers(prefixOrOptions: unknown = "") {
      const prefix = statePrefix(prefixOrOptions);
      return Object.freeze(this.runningBuffers().map((targetBuffer) => ({
        ...targetBuffer,
        name: prefixedName(prefix, targetBuffer.name),
      })));
    }

    buffers(prefixOrOptions: unknown = "") {
      return this.namedBuffers(prefixOrOptions);
    }

    getBuffer(name: unknown) {
      if (typeof name !== "string") throw new Error("module getBuffer requires a buffer name");
      return this.runningBuffers().find((targetBuffer) => targetBuffer.name === name) ?? null;
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    compileSupport(options: ModuleCompileOptions = {}) {
      return singleModuleCompileSupport(this, compileHooks, options);
    }
  }

  installStatefulModuleStateMethods(FeatureNormModule.prototype, stateHooks);
  const featureNormPrototype = FeatureNormModule.prototype as typeof FeatureNormModule.prototype & {
    stateDict(prefixOrOptions?: unknown): unknown;
    loadStateDict(source: unknown, options?: UnknownRecord): unknown;
    load_state_dict(source: unknown, options?: UnknownRecord): unknown;
  };
  featureNormPrototype.stateDict = function stateDict(this: InstanceType<typeof FeatureNormModule> & { parameterNames(prefix?: string): readonly string[] }, prefixOrOptions: unknown = "") {
    const prefix = statePrefix(prefixOrOptions);
    const out = { ...(stateHooks.stateDict(this, prefix) as UnknownRecord) };
    for (const targetBuffer of this.namedBuffers(prefix) as readonly FeatureNormBuffer[]) {
      out[targetBuffer.name] = stateEntryForBuffer(targetBuffer);
    }
    return Object.freeze(out);
  };
  featureNormPrototype.loadStateDict = function loadStateDict(
    this: InstanceType<typeof FeatureNormModule> & { parameterNames(prefix?: string): readonly string[] },
    source: unknown,
    options: UnknownRecord = {},
  ) {
    const strict = options.strict !== false;
    const validateOnly = options.validateOnly === true;
    const prefix = typeof options.prefix === "string" ? options.prefix : "";
    const paramNames = this.parameterNames(prefix) as readonly string[];
    const bufferNames = (this.namedBuffers(prefix) as readonly FeatureNormBuffer[]).map((targetBuffer) => targetBuffer.name);
    const expected = new Set([...paramNames, ...bufferNames]);
    if (strict) {
      for (const key of stateKeys(source)) {
        if (!expected.has(key)) throw new Error(`state dict has unexpected entry ${key}`);
      }
      for (const name of expected) {
        if (stateValue(source, name) === undefined) throw new Error(`state dict is missing entry ${name}`);
      }
    }

    const paramSource = new Map<string, unknown>();
    for (const name of paramNames) {
      const value = stateValue(source, name);
      if (value !== undefined) paramSource.set(name, value);
    }
    stateHooks.loadStateDict(this, paramSource, { ...options, strict });

    const pendingBuffers: Array<{ targetBuffer: FeatureNormBuffer; data: Float32Array }> = [];
    for (const targetBuffer of this.namedBuffers(prefix) as readonly FeatureNormBuffer[]) {
      const value = stateValue(source, targetBuffer.name);
      if (value === undefined) continue;
      const shape = stateShape(value);
      if (shape && (shape.length !== targetBuffer.shape.length || shape.some((dim, index) => dim !== targetBuffer.shape[index]))) {
        throw new Error(`state dict entry ${targetBuffer.name} shape [${shape.join(",")}] does not match [${targetBuffer.shape.join(",")}]`);
      }
      pendingBuffers.push({ targetBuffer, data: stateF32(value, targetBuffer.data.length, `state dict entry ${targetBuffer.name}`) });
    }
    if (!validateOnly) {
      for (const pending of pendingBuffers) pending.targetBuffer.data.set(pending.data);
    }
    return this;
  };
  featureNormPrototype.load_state_dict = function load_state_dict(
    this: InstanceType<typeof FeatureNormModule> & { loadStateDict(source: unknown, options?: UnknownRecord): unknown },
    source: unknown,
    options: UnknownRecord = {},
  ) {
    return this.loadStateDict(source, options);
  };
  installSingleModuleCompileMethods(FeatureNormModule.prototype, compileHooks, {
    compileSupport: false,
    bindParameters(module: SingleModuleRecord, _hooks: SingleModuleCompileHooks, options: ModuleCompileOptions = {}) {
      const featureNormModule = featureNormBindableModule(module);
      const batchNormBindings = batchNormEvalBindings(featureNormModule);
      if (batchNormBindings) return annotateModuleBindings(batchNormBindings, featureNormModule, options);
      if (featureNormModule.bias) return annotateModuleBindings({ weights: featureNormModule.weight, bias: featureNormModule.bias }, featureNormModule, options);
      if (featureNormModule.weight) return annotateModuleBindings({ weights: featureNormModule.weight }, featureNormModule, options);
      return emptySingleModuleBindings(featureNormModule, options);
    },
  });
  return FeatureNormModule;
}
