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
type SequentialLayerNameRecord = SequentialModuleRecord & {
  readonly layerNames?: readonly string[];
};
type TinyLinearCompiledSpec = SequentialCompiledSpec & Readonly<{
  readonly kind: "tiny-linear";
  readonly layer: { compile(options?: CompileOptions): unknown };
}>;
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: SequentialTensorConstructOptions) => SequentialTensor;
type F32 = (values: unknown) => Float32Array;
type FactoryCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

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

export type SequentialModuleClassHooks = SequentialProgramCompileHooksInput & {
  Tensor: TensorConstructor;
  f32: F32;
  traceSequentialProgram: FactoryCallback<[layers: readonly NnModule[], options: ModuleTraceOptions], unknown>;
};

export type SequentialModuleClassOptions = Readonly<Record<string, unknown> & SequentialModuleClassHooks>;

export function createSequentialModuleClass(options: SequentialModuleClassOptions) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const stateHooks = createStatefulModuleStateHooks(options, "SequentialModule", {
    trainModule: setChildModuleTraining,
  });
  const traceSequentialProgram = options.traceSequentialProgram;
  const compileHooks = createSequentialProgramCompileHooks(options, "SequentialModule");
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function" ||
    typeof traceSequentialProgram !== "function"
  ) {
    throw new Error("SequentialModule factory requires tensor, state, trace, analysis, packing, placement, and compile hooks");
  }

  class SequentialModule {
    layers: SequentialLayer[];
    layerNames: string[];
    training?: boolean;

    constructor(first?: SequentialLayerInput, ...rest: readonly NnModule[]) {
      const { layers, names } = normalizeSequentialLayers(first, rest);
      initializeModuleMode(this);
      this.layers = layers.map((layer) => sequentialLayer(layer, "nn.Sequential")) as SequentialLayer[];
      this.layerNames = names.slice();
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
      this.layers[sequentialIndex(index, this.layers.length)] = sequentialLayer(layer, "nn.Sequential.__setitem__");
      return this;
    }

    __delitem__(index: number) {
      const resolved = sequentialIndex(index, this.layers.length);
      this.layers.splice(resolved, 1);
      this.layerNames.splice(resolved, 1);
      return this;
    }

    pop(index = -1) {
      const resolved = sequentialIndex(index, this.layers.length);
      const [layer] = this.layers.splice(resolved, 1);
      this.layerNames.splice(resolved, 1);
      return layer;
    }

    clear() {
      this.layers.length = 0;
      this.layerNames.length = 0;
      return this;
    }

    append(layer: NnModule) {
      this.layers.push(sequentialLayer(layer, "nn.Sequential.append"));
      this.layerNames.push(String(this.layerNames.length));
      return this;
    }

    insert(index: number, layer: NnModule) {
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

    forward(inputValues: unknown) {
      let out: unknown = inputValues instanceof TensorClass ? inputValues : f32(inputValues);
      for (const layer of this.layers) out = layer.forward(out);
      return out;
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
