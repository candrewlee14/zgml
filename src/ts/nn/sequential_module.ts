import { annotateModuleBindings } from "../runtime/module_bindings.js";
import { initializeModuleMode, setChildModuleTraining } from "../train/module_mode.js";
import type { CompileOptions, ModuleTraceOptions, NnModule } from "../public_api.js";
import { moduleTraversalEntry, moduleTraversalOptions } from "./module_tree.js";
import {
  createStatefulModuleStateHooks,
  installStatefulModuleStateMethods,
} from "./stateful_module_state.js";
import {
  attachTinyLinearEvidence,
  createSequentialProgramCompileHooks,
  installSequentialProgramCompileMethods,
  sequentialProgramAnalysis,
  sequentialProgramUnsupportedReason,
  type SequentialCompiledSpec,
  type SequentialProgramAnalysis,
  type SequentialProgramCompileHooks,
  type SequentialProgramCompileHooksInput,
  type SequentialModuleRecord,
} from "./sequential_program_compile.js";

export type SequentialTensorConstructOptions = Readonly<Record<string, unknown>>;
export type SequentialTensor = {
  readonly data?: Float32Array;
  readonly shape?: readonly number[];
  readonly rank?: number;
  readonly length?: number;
};
export type SequentialLayer = NnModule & {
  forward(input: unknown): unknown;
};
type ModuleCompileOptions = Readonly<CompileOptions & Record<string, unknown>>;
type NamedSequentialLayers = Readonly<Record<string, NnModule>>;
type SequentialLayerInput = readonly NnModule[] | NamedSequentialLayers | NnModule;
type SequentialBindingAnnotationTarget = SequentialModuleRecord & {
  readonly compileSupport?: (options: ModuleCompileOptions) => unknown;
};
type SequentialNativeForwardProgram = {
  bind?: (bindings: unknown) => SequentialNativeForwardSession;
  dispose?: () => void;
};
type SequentialNativeForwardSession = {
  stepTensor(input: unknown): unknown;
  uploadParameters?: () => unknown;
  dispose?: () => void;
};
type SequentialNativeForwardCache = {
  readonly key: string;
  readonly program: SequentialNativeForwardProgram;
  session: SequentialNativeForwardSession;
  bindings: Record<string, unknown>;
  readonly disabled?: false;
};
type SequentialNativeForwardDisabledCache = {
  readonly key: string;
  readonly disabled: true;
};
type SequentialNativeForwardCacheState = SequentialNativeForwardCache | SequentialNativeForwardDisabledCache | null;
type SequentialNativeForwardEvidence = Readonly<{
  kind: "zgml.sequential-native-forward";
  native: boolean;
  engine: "zig" | null;
  path: "program-session" | "native-eager-linear-activation" | "fallback";
  inputShape: readonly number[] | null;
  outputShape: readonly number[] | null;
  cache: "hit" | "miss" | "disabled" | "none";
  reason?: string;
}>;
type SequentialLayerNameRecord = SequentialModuleRecord & {
  readonly layerNames?: readonly string[];
};
type TinyLinearCompiledSpec = SequentialCompiledSpec & Readonly<{
  readonly kind: "tiny-linear";
  readonly layer: { compile(options?: CompileOptions): unknown };
}>;
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: SequentialTensorConstructOptions) => SequentialTensor;
type F32 = (values: unknown) => Float32Array;
type PrepareF32 = (values: unknown) => { readonly data: Float32Array; readonly shape: readonly number[] };
type FactoryCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type NativeEagerLinearActivationInto = (
  output: Float32Array,
  input: SequentialTensor,
  weights: unknown,
  options: Readonly<{
    bias?: unknown;
    batch?: number;
    inFeatures?: number;
    outFeatures?: number;
    activation: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
  }>,
) => Float32Array;
type SequentialLinearLayer = SequentialLayer & Readonly<{
  kind: "linear";
  inFeatures: number;
  outFeatures: number;
  weightParam: { readonly tensor: unknown };
  biasParam?: { readonly tensor: unknown } | null;
}>;
type SequentialActivationLayer = SequentialLayer & Readonly<{
  kind: string;
}>;

function sequentialChildren(module: SequentialModuleRecord) {
  return Object.freeze((module.layers ?? []).slice());
}

function sequentialModules(module: SequentialModuleRecord) {
  const out = [module];
  for (const layer of module.layers ?? []) {
    if (layer && typeof layer.modules === "function") {
      out.push(...layer.modules());
    } else {
      out.push(layer);
    }
  }
  return Object.freeze(out);
}

function joinModuleName(prefix: string, name: unknown) {
  const childName = String(name);
  return prefix ? `${prefix}.${childName}` : childName;
}

function sequentialLayerName(module: SequentialModuleRecord, index: number) {
  const name = (module as SequentialLayerNameRecord).layerNames?.[index];
  return name ?? String(index);
}

function sequentialNamedChildren(module: SequentialModuleRecord, prefix = "") {
  return Object.freeze((module.layers ?? []).map((layer: unknown, index: number) => moduleTraversalEntry(joinModuleName(String(prefix), sequentialLayerName(module, index)), layer)));
}

function sequentialNamedModules(module: SequentialModuleRecord, prefix = "") {
  const rootName = String(prefix);
  const out = [moduleTraversalEntry(rootName, module)];
  for (const entry of sequentialNamedChildren(module, rootName)) {
    const child = entry.module as Partial<NnModule> | null | undefined;
    if (child && typeof child.namedModules === "function") {
      const named = child.namedModules(entry.name);
      out.push(...named);
    } else {
      out.push(entry);
    }
  }
  return Object.freeze(out);
}

function sequentialIndex(index: number, length: number) {
  if (!Number.isSafeInteger(index)) throw new Error(`Sequential index must be a safe integer, got ${index}`);
  const resolved = index < 0 ? length + index : index;
  if (resolved < 0 || resolved >= length) throw new Error(`Sequential index ${index} is out of range for length ${length}`);
  return resolved;
}

function sequentialInsertIndex(index: number, length: number) {
  if (!Number.isSafeInteger(index)) throw new Error(`Sequential insert index must be a safe integer, got ${index}`);
  if (index < 0) return Math.max(0, length + index);
  return Math.min(index, length);
}

function sequentialLayer(layer: NnModule, label: string) {
  if (!layer || typeof layer.forward !== "function") throw new Error(`${label} requires an nn module`);
  return layer as SequentialLayer;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isSequentialLayerRecord(value: unknown) {
  if (!isRecord(value) || typeof value.forward === "function") return false;
  const entries = Object.entries(value);
  return entries.every(([, layer]) => layer && typeof (layer as Partial<NnModule>).forward === "function");
}

function normalizeSequentialLayers(first: SequentialLayerInput | undefined, rest: readonly NnModule[]) {
  if (first === undefined && rest.length === 0) return { layers: [] as NnModule[], names: [] as string[] };
  if (rest.length === 0 && Array.isArray(first)) return { layers: first.slice(), names: first.map((_, index) => String(index)) };
  if (rest.length === 0 && isSequentialLayerRecord(first)) {
    const entries = Object.entries(first as NamedSequentialLayers);
    return {
      layers: entries.map(([, layer]) => layer as NnModule),
      names: entries.map(([name]) => name),
    };
  }
  return {
    layers: [first, ...rest] as NnModule[],
    names: Array.from({ length: rest.length + 1 }, (_, index) => String(index)),
  };
}

function tinyLinearCompiledSpec(compiled: SequentialCompiledSpec): TinyLinearCompiledSpec {
  return compiled as TinyLinearCompiledSpec;
}

function sequentialActivationKind(layer: SequentialLayer): "relu" | "gelu" | "silu" | "sigmoid" | "tanh" | null {
  const kind = String((layer as SequentialActivationLayer).kind ?? "");
  switch (kind) {
    case "relu":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
      return kind;
    default:
      return null;
  }
}

function isSequentialLinearLayer(layer: unknown): layer is SequentialLinearLayer {
  if (!layer || typeof layer !== "object") return false;
  const candidate = layer as Partial<SequentialLinearLayer>;
  return (
    candidate.kind === "linear" &&
    Number.isSafeInteger(candidate.inFeatures) &&
    Number.isSafeInteger(candidate.outFeatures) &&
    candidate.weightParam !== null &&
    typeof candidate.weightParam === "object"
  );
}

function tensorShapeSignature(input: SequentialTensor) {
  const shape = input.shape;
  if (!Array.isArray(shape) || shape.length === 0) return null;
  if (!shape.every((dim) => Number.isSafeInteger(dim) && dim > 0)) return null;
  return shape.join("x");
}

function tensorShape(input: SequentialTensor | null | undefined) {
  const shape = input?.shape;
  return Array.isArray(shape) ? Object.freeze(shape.slice()) : null;
}

function nativeForwardEvidence(
  input: SequentialTensor | null,
  output: SequentialTensor | null,
  fields: Readonly<{
    native: boolean;
    engine?: "zig" | null;
    path?: SequentialNativeForwardEvidence["path"];
    cache?: SequentialNativeForwardEvidence["cache"];
    reason?: string;
  }>,
): SequentialNativeForwardEvidence {
  return Object.freeze({
    kind: "zgml.sequential-native-forward",
    native: fields.native,
    engine: fields.engine === undefined ? (fields.native ? "zig" : null) : fields.engine,
    path: fields.path ?? (fields.native ? "program-session" : "fallback"),
    inputShape: tensorShape(input),
    outputShape: tensorShape(output),
    cache: fields.cache ?? "none",
    ...(fields.reason ? { reason: fields.reason } : {}),
  });
}

function samePackedBinding(left: unknown, right: unknown) {
  if (left === undefined || right === undefined) return left === right;
  if (!(left instanceof Float32Array) || !(right instanceof Float32Array)) return left === right;
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return false;
  }
  return true;
}

function samePackedBindings(left: Record<string, unknown>, right: Record<string, unknown>) {
  return samePackedBinding(left.weights, right.weights) && samePackedBinding(left.bias, right.bias);
}

export type SequentialModuleClassHooks = SequentialProgramCompileHooksInput & {
  Tensor: TensorConstructor;
  f32: F32;
  prepareF32?: PrepareF32;
  traceSequentialProgram: FactoryCallback<[layers: readonly NnModule[], options: ModuleTraceOptions], unknown>;
  nativeEagerLinearActivationInto?: NativeEagerLinearActivationInto;
  isGradEnabled?: () => boolean;
};

export type SequentialModuleClassOptions = Readonly<Record<string, unknown> & SequentialModuleClassHooks>;

export function createSequentialModuleClass(options: SequentialModuleClassOptions) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const prepareF32 = typeof options.prepareF32 === "function" ? options.prepareF32 : null;
  const stateHooks = createStatefulModuleStateHooks(options, "SequentialModule", {
    trainModule: setChildModuleTraining,
  });
  const traceSequentialProgram = options.traceSequentialProgram;
  const compileHooks = createSequentialProgramCompileHooks(options, "SequentialModule");
  const nativeEagerLinearActivationInto = options.nativeEagerLinearActivationInto;
  const isGradEnabled = typeof options.isGradEnabled === "function" ? options.isGradEnabled : () => true;
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function" ||
    typeof traceSequentialProgram !== "function"
  ) {
    throw new Error("SequentialModule factory requires tensor, state, trace, analysis, packing, placement, and compile hooks");
  }

  function canUseNativeEagerLinearActivation() {
    if (typeof nativeEagerLinearActivationInto !== "function") return false;
    if (isGradEnabled()) return false;
    return true;
  }

  function tryNativeEagerLinearActivation(input: unknown, linear: SequentialLayer, activationLayer: SequentialLayer) {
    if (!canUseNativeEagerLinearActivation()) return null;
    if (!(input instanceof TensorClass)) return null;
    if (!isSequentialLinearLayer(linear)) return null;
    const activation = sequentialActivationKind(activationLayer);
    if (activation === null) return null;
    const tensor = input as SequentialTensor;
    const shape = Array.isArray(tensor.shape) ? tensor.shape : null;
    const rank = Number.isSafeInteger(tensor.rank) ? tensor.rank as number : shape ? shape.length : null;
    if (rank === null || rank < 1 || !shape) return null;
    if (shape[shape.length - 1] !== linear.inFeatures) return null;

    const leadingShape = shape.slice(0, -1);
    const rowCount = leadingShape.length === 0 ? 1 : leadingShape.reduce((acc, dim) => acc * dim, 1);
    const output = new Float32Array(rowCount * linear.outFeatures);
    nativeEagerLinearActivationInto!(output, tensor, linear.weightParam.tensor, {
      bias: linear.biasParam?.tensor ?? null,
      batch: rowCount,
      inFeatures: linear.inFeatures,
      outFeatures: linear.outFeatures,
      activation,
    });
    const outputShape = rank === 1 ? [linear.outFeatures] : [...leadingShape, linear.outFeatures];
    return new TensorClass(output, outputShape);
  }

  class SequentialModule {
    layers: SequentialLayer[];
    layerNames: string[];
    nativeForwardCache: SequentialNativeForwardCacheState;
    nativeForwardProof: SequentialNativeForwardEvidence | null;
    training?: boolean;

    constructor(first?: SequentialLayerInput, ...rest: readonly NnModule[]) {
      const { layers, names } = normalizeSequentialLayers(first, rest);
      initializeModuleMode(this);
      this.layers = layers.map((layer) => sequentialLayer(layer, "nn.Sequential")) as SequentialLayer[];
      this.layerNames = names.slice();
      this.nativeForwardCache = null;
      this.nativeForwardProof = null;
    }

    get length() {
      return this.layers.length;
    }

    at(index: number) {
      return this.layers[sequentialIndex(index, this.layers.length)];
    }

    get(index: number) {
      return this.at(index);
    }

    __getitem__(index: number) {
      return this.at(index);
    }

    __setitem__(index: number, layer: NnModule) {
      this.disposeNativeForwardCache();
      this.layers[sequentialIndex(index, this.layers.length)] = sequentialLayer(layer, "nn.Sequential.__setitem__");
      return this;
    }

    __delitem__(index: number) {
      this.disposeNativeForwardCache();
      const resolved = sequentialIndex(index, this.layers.length);
      this.layers.splice(resolved, 1);
      this.layerNames.splice(resolved, 1);
      return this;
    }

    pop(index = -1) {
      this.disposeNativeForwardCache();
      const resolved = sequentialIndex(index, this.layers.length);
      const [layer] = this.layers.splice(resolved, 1);
      this.layerNames.splice(resolved, 1);
      return layer;
    }

    clear() {
      this.disposeNativeForwardCache();
      this.layers.length = 0;
      this.layerNames.length = 0;
      return this;
    }

    append(layer: NnModule) {
      this.disposeNativeForwardCache();
      this.layers.push(sequentialLayer(layer, "nn.Sequential.append"));
      this.layerNames.push(String(this.layerNames.length));
      return this;
    }

    insert(index: number, layer: NnModule) {
      this.disposeNativeForwardCache();
      const resolved = sequentialInsertIndex(index, this.layers.length);
      this.layers.splice(resolved, 0, sequentialLayer(layer, "nn.Sequential.insert"));
      this.layerNames.splice(resolved, 0, String(resolved));
      return this;
    }

    extend(layers: Iterable<NnModule>) {
      if (!layers || typeof layers[Symbol.iterator] !== "function") {
        throw new Error("nn.Sequential.extend requires an iterable of nn modules");
      }
      for (const layer of layers) this.append(layer);
      return this;
    }

    len() {
      return this.layers.length;
    }

    __len__() {
      return this.len();
    }

    size() {
      return this.len();
    }

    [Symbol.iterator]() {
      return this.children()[Symbol.iterator]();
    }

    disposeNativeForwardCache() {
      const cache = this.nativeForwardCache;
      this.nativeForwardCache = null;
      if (!cache || cache.disabled) return;
      if (typeof cache.session.dispose === "function") cache.session.dispose();
      if (typeof cache.program.dispose === "function") cache.program.dispose();
    }

    refreshNativeForwardBindings(cache: SequentialNativeForwardCache, key: string) {
      const bindParameters = (this as SequentialModuleRecord).bindParameters;
      if (typeof bindParameters !== "function") return false;
      const latest = bindParameters.call(this, { inputShape: key.split("x").map(Number) }) as Record<string, unknown>;
      if (samePackedBindings(cache.bindings, latest)) return true;
      if (typeof cache.session.dispose === "function") cache.session.dispose();
      if (typeof cache.program.bind !== "function") return false;
      const session = cache.program.bind(latest);
      if (!session || typeof session.stepTensor !== "function") return false;
      cache.bindings = latest;
      cache.session = session;
      return true;
    }

    nativeForward(input: SequentialTensor) {
      if (isGradEnabled()) {
        this.nativeForwardProof = nativeForwardEvidence(input, null, {
          native: false,
          cache: "none",
          reason: "grad-enabled",
        });
        return null;
      }
      const key = tensorShapeSignature(input);
      if (key === null) {
        this.nativeForwardProof = nativeForwardEvidence(input, null, {
          native: false,
          cache: "none",
          reason: "unsupported-input-shape",
        });
        return null;
      }
      const cached = this.nativeForwardCache;
      if (cached && cached.key === key) {
        if (cached.disabled) {
          this.nativeForwardProof = nativeForwardEvidence(input, null, {
            native: false,
            cache: "disabled",
            reason: "compile-disabled-for-shape",
          });
          return null;
        }
        try {
          if (!this.refreshNativeForwardBindings(cached, key)) {
            this.nativeForwardProof = nativeForwardEvidence(input, null, {
              native: false,
              cache: "hit",
              reason: "parameter-refresh-failed",
            });
            return null;
          }
          const output = cached.session.stepTensor(input) as SequentialTensor;
          this.nativeForwardProof = nativeForwardEvidence(input, output, {
            native: true,
            path: "program-session",
            cache: "hit",
          });
          return output;
        } catch (error) {
          this.disposeNativeForwardCache();
          this.nativeForwardCache = { key, disabled: true };
          this.nativeForwardProof = nativeForwardEvidence(input, null, {
            native: false,
            cache: "disabled",
            reason: error instanceof Error ? error.message : "native-forward-cache-failed",
          });
          return null;
        }
      }
      this.disposeNativeForwardCache();
      const compileSelf = this as SequentialModuleRecord & {
        compile?: (options?: ModuleCompileOptions) => SequentialNativeForwardProgram;
        bindParameters?: (options?: ModuleCompileOptions) => Record<string, unknown>;
      };
      if (typeof compileSelf.compile !== "function" || typeof compileSelf.bindParameters !== "function") {
        this.nativeForwardCache = { key, disabled: true };
        this.nativeForwardProof = nativeForwardEvidence(input, null, {
          native: false,
          cache: "disabled",
          reason: "compile-methods-missing",
        });
        return null;
      }
      let program: SequentialNativeForwardProgram | null = null;
      try {
        const compileOptions = { inputShape: key.split("x").map(Number) };
        program = compileSelf.compile(compileOptions);
        if (!program || typeof program.bind !== "function") {
          if (program && typeof program.dispose === "function") program.dispose();
          this.nativeForwardCache = { key, disabled: true };
          this.nativeForwardProof = nativeForwardEvidence(input, null, {
            native: false,
            cache: "disabled",
            reason: "compile-did-not-return-bindable-program",
          });
          return null;
        }
        const bindings = compileSelf.bindParameters(compileOptions);
        const session = program.bind(bindings);
        if (!session || typeof session.stepTensor !== "function") {
          if (typeof program.dispose === "function") program.dispose();
          this.nativeForwardCache = { key, disabled: true };
          this.nativeForwardProof = nativeForwardEvidence(input, null, {
            native: false,
            cache: "disabled",
            reason: "bind-did-not-return-step-session",
          });
          return null;
        }
        const nextCache = { key, program, session, bindings } as SequentialNativeForwardCache;
        this.nativeForwardCache = nextCache;
        const output = session.stepTensor(input) as SequentialTensor;
        this.nativeForwardProof = nativeForwardEvidence(input, output, {
          native: true,
          path: "program-session",
          cache: "miss",
        });
        return output;
      } catch (error) {
        if (program && typeof program.dispose === "function") program.dispose();
        this.nativeForwardCache = { key, disabled: true };
        this.nativeForwardProof = nativeForwardEvidence(input, null, {
          native: false,
          cache: "disabled",
          reason: error instanceof Error ? error.message : "native-forward-compile-failed",
        });
        return null;
      }
    }

    forward(inputValues: unknown) {
      const inputIsTensor = inputValues instanceof TensorClass;
      let out: unknown = inputValues;
      if (inputIsTensor) {
        if (this.layers.length === 2) {
          const fused = tryNativeEagerLinearActivation(out, this.layers[0], this.layers[1]);
          if (fused !== null) {
            this.nativeForwardProof = nativeForwardEvidence(out as SequentialTensor, fused, {
              native: true,
              path: "native-eager-linear-activation",
              cache: "none",
            });
            return fused;
          }
        }
        const nativeOut = this.nativeForward(out as SequentialTensor);
        if (nativeOut !== null) return nativeOut;
        if (this.layers.length === 1) return this.layers[0]!.forward(inputValues);
      } else {
        const firstLayer = this.layers[0];
        if (isSequentialLinearLayer(firstLayer)) {
          const preparedInput = prepareF32 ? prepareF32(inputValues) : null;
          const data = preparedInput ? preparedInput.data : f32(inputValues);
          out = data;
          const prepared = preparedInput ?? { data, shape: [firstLayer.inFeatures] };
          if (
            prepared.data instanceof Float32Array &&
            prepared.data.length % firstLayer.inFeatures === 0 &&
            (
              (prepared.shape.length === 1 && prepared.shape[0] === firstLayer.inFeatures) ||
              (prepared.shape.length >= 2 && prepared.shape[prepared.shape.length - 1] === firstLayer.inFeatures)
            )
          ) {
            const nativeInput = new TensorClass(prepared.data, prepared.shape);
            const nativeOut = this.nativeForward(nativeInput);
            if (nativeOut !== null) return nativeOut;
            if (this.layers.length === 1) return this.layers[0]!.forward(inputValues);
          }
          out = prepared.data;
        } else {
          out = f32(inputValues);
          if (this.layers.length === 1) return this.layers[0]!.forward(out);
        }
      }
      for (let index = 0; index < this.layers.length; index += 1) {
        const layer = this.layers[index];
        const nextLayer = this.layers[index + 1];
        if (nextLayer) {
          const fused = tryNativeEagerLinearActivation(out, layer, nextLayer);
          if (fused !== null) {
            this.nativeForwardProof = nativeForwardEvidence(out as SequentialTensor, fused, {
              native: true,
              path: "native-eager-linear-activation",
              cache: "none",
            });
            out = fused;
            index += 1;
            continue;
          }
        }
        out = layer.forward(out);
      }
      return out;
    }

    nativeForwardEvidence() {
      return this.nativeForwardProof;
    }

    native_forward_evidence() {
      return this.nativeForwardEvidence();
    }

    lastNativeForwardEvidence() {
      return this.nativeForwardEvidence();
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
      if (!recurse) return [];
      return this.layers.flatMap((layer, index) => layer.parameters(joinModuleName(prefix, sequentialLayerName(this, index))));
    }

    children() {
      return sequentialChildren(this);
    }

    modules() {
      return sequentialModules(this);
    }

    namedChildren(prefix = "") {
      return sequentialNamedChildren(this, prefix);
    }

    namedModules(prefix = "") {
      return sequentialNamedModules(this, prefix);
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    trace(traceOptions: ModuleTraceOptions = {}) {
      return traceSequentialProgram(this.layers, traceOptions);
    }

  }

  installStatefulModuleStateMethods(SequentialModule.prototype, stateHooks);
  installSequentialProgramCompileMethods(SequentialModule.prototype, compileHooks, {
    layersForModule: (module: SequentialModuleRecord) => module.layers ?? [],
    fallbackReason: "nn.Sequential.compile is unsupported",
    bindParameters(
      module: SequentialModuleRecord,
      hooks: SequentialProgramCompileHooks,
      bindOptions: ModuleCompileOptions = {},
      layersForModule: (module: SequentialModuleRecord) => readonly SequentialModuleRecord[],
    ) {
      const compiled = sequentialProgramAnalysis(module, hooks, bindOptions, layersForModule).compiled;
      if (compiled && typeof hooks.packParameters === "function") return annotateModuleBindings(hooks.packParameters(compiled), module as SequentialBindingAnnotationTarget, bindOptions);
      throw new Error(sequentialProgramUnsupportedReason(module, bindOptions, "nn.Sequential.compile is unsupported"));
    },
    compileTinyLinear(compiled: SequentialCompiledSpec, _analysis: SequentialProgramAnalysis, compileOptions: ModuleCompileOptions, hooks: SequentialProgramCompileHooks) {
      const program = tinyLinearCompiledSpec(compiled).layer.compile(compileOptions);
      return attachTinyLinearEvidence(program, hooks, compiled);
    },
  });
  return SequentialModule;
}
