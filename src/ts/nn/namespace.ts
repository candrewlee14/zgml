import {
  installNnModulePrototypeMethods,
  type NnModulePrototypeConstructor,
} from "./namespace_methods.js";
import { moduleTraversalOptions } from "./module_tree.js";
import {
  acceptsModuleBindingPlan,
  acceptsModuleCompilePlan,
  assert_module_binding_plan,
  assert_module_compile_plan,
  assertModuleBindingPlan,
  assertModuleCompilePlan,
  matchesModuleBindingPlanSignature,
  matchesModuleCompilePlanSignature,
  requireModuleBindingPlan,
  requireModuleCompilePlan,
} from "../runtime/execution_plan.js";
import type { ModuleFacadeTarget } from "../runtime/module_facade.js";
import type {
  ActivationKind,
  CompileOptions,
  LoadStateDictOptions,
  ModuleReductionKind,
  ModuleTraceOptions,
  NnModule,
  NnDropoutConfig,
  NnEmbeddingConfig,
  NnLinearConfig,
  NnNormConfig,
  ShapeModuleKind,
  TensorToOptions,
  TensorToTarget,
  TensorShape,
  ZeroGradOptions,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type UnknownRecord = Record<string, unknown>;

type NnNamespaceHookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

type NnScalarFunction = (value: number) => number;

type NnParameter = AnyRecord & {
  name?: string;
  data?: Float32Array;
  grad?: Float32Array | null;
};

type NnBuffer = AnyRecord & {
  kind: "buffer";
  name: string;
  data: Float32Array;
  shape: readonly number[];
  layout?: string;
  persistent: boolean;
  tensor?: unknown;
};

type NnModuleTarget = Omit<ModuleFacadeTarget, "forward"> & Readonly<{
  kind?: unknown;
  graph?: unknown;
  forward?: unknown;
  parameters?: unknown;
  namedParameters?: unknown;
  buffers?: unknown;
  namedBuffers?: unknown;
  parameterNames?: unknown;
  parameterInfos?: unknown;
  parameterInfo?: unknown;
  children?: unknown;
  modules?: unknown;
  namedChildren?: unknown;
  namedModules?: unknown;
  apply?: unknown;
}>;
type NativeInferenceProgram = Record<string, unknown> & {
  bindModule?: (target: unknown, options?: unknown) => unknown;
  bind?: (bindings?: unknown) => unknown;
  dispose?: () => void;
  inputShape?: () => unknown;
  outputShape?: () => unknown;
  kernelPlan?: () => unknown;
  compilerSignatures?: () => unknown;
};
type NativeInferenceSession = Record<string, unknown> & {
  stepTensor(input: unknown): unknown;
  executeInto(output: Float32Array, bindings: unknown): Float32Array;
  prepareExecuteInto(output: Float32Array, bindings: unknown): unknown;
  dispose?: () => void;
};

type LinearModuleConstructor = NnModulePrototypeConstructor & (new (inFeatures: number, outFeatures: number, config?: NnLinearConfig) => NnModuleTarget);
type EmbeddingModuleConstructor = NnModulePrototypeConstructor & (new (numEmbeddings: number, embeddingDim: number, config?: NnEmbeddingConfig) => NnModuleTarget);
type Conv2dModuleConstructor = NnModulePrototypeConstructor & (new (inChannels: number, outChannels: number, kernelSize: unknown, config?: UnknownRecord) => NnModuleTarget);
type AvgPool2dModuleConstructor = NnModulePrototypeConstructor & (new(kernelSize: unknown, config?: UnknownRecord) => NnModuleTarget);
type SequentialModuleConstructor = NnModulePrototypeConstructor & (new (first?: readonly NnModule[] | Readonly<Record<string, NnModule>> | NnModule, ...rest: readonly NnModule[]) => NnModuleTarget);
type ActivationModuleConstructor = NnModulePrototypeConstructor & (new (kind: ActivationKind, fn: NnScalarFunction) => NnModuleTarget);
type SoftmaxModuleConstructor = NnModulePrototypeConstructor & (new (dim?: number) => NnModuleTarget);
type ReductionModuleConstructor = NnModulePrototypeConstructor & (new (kind: ModuleReductionKind, dim?: number) => NnModuleTarget);
type DropoutModuleConstructor = NnModulePrototypeConstructor & (new (p?: number, config?: NnDropoutConfig) => NnModuleTarget);
type ShapeModuleConstructor = new (
  kind: ShapeModuleKind,
  shapeOrStartDim?: unknown,
  endDim?: unknown,
  length?: unknown,
  step?: unknown,
) => NnModuleTarget;
type ShapeModulePrototypeConstructor = NnModulePrototypeConstructor & ShapeModuleConstructor;
type FeatureNormKind = "layerNorm" | "rmsNorm" | "batchNorm1d";
type FeatureNormModuleConstructor = NnModulePrototypeConstructor & (new (kind: FeatureNormKind, features: unknown, config?: UnknownRecord) => NnModuleTarget);
type MSELossConstructor = new (options?: UnknownRecord) => unknown;
type L1LossConstructor = new (options?: UnknownRecord) => unknown;
type HuberLossConstructor = new (options?: UnknownRecord) => unknown;
type SmoothL1LossConstructor = new (options?: UnknownRecord) => unknown;
type BCELossConstructor = new (options?: UnknownRecord) => unknown;
type BCEWithLogitsLossConstructor = new (options?: UnknownRecord) => unknown;
type CrossEntropyLossConstructor = new (options?: UnknownRecord) => unknown;
type NLLLossConstructor = new (options?: UnknownRecord) => unknown;
type NnPrototypeConstructor =
  | LinearModuleConstructor
  | EmbeddingModuleConstructor
  | Conv2dModuleConstructor
  | AvgPool2dModuleConstructor
  | SequentialModuleConstructor
  | ActivationModuleConstructor
  | SoftmaxModuleConstructor
  | ReductionModuleConstructor
  | DropoutModuleConstructor
  | ShapeModulePrototypeConstructor
  | FeatureNormModuleConstructor;

export type NnNamespaceHooks = {
  LinearModule: LinearModuleConstructor;
  EmbeddingModule: EmbeddingModuleConstructor;
  Conv2dModule: Conv2dModuleConstructor;
  AvgPool2dModule: AvgPool2dModuleConstructor;
  SequentialModule: SequentialModuleConstructor;
  ActivationModule: ActivationModuleConstructor;
  SoftmaxModule: SoftmaxModuleConstructor;
  LogSoftmaxModule: SoftmaxModuleConstructor;
  ReductionModule: ReductionModuleConstructor;
  DropoutModule: DropoutModuleConstructor;
  ShapeModule: ShapeModulePrototypeConstructor;
  FeatureNormModule: FeatureNormModuleConstructor;
  MSELoss: MSELossConstructor;
  L1Loss: L1LossConstructor;
  HuberLoss: HuberLossConstructor;
  SmoothL1Loss: SmoothL1LossConstructor;
  BCELoss: BCELossConstructor;
  BCEWithLogitsLoss: BCEWithLogitsLossConstructor;
  CrossEntropyLoss: CrossEntropyLossConstructor;
  NLLLoss: NLLLossConstructor;
  geluScalar: NnScalarFunction;
  siluScalar: NnScalarFunction;
  sigmoidScalar: NnScalarFunction;
  f32WithLength: NnNamespaceHookCallback<[value: unknown, length: number, label: string], Float32Array>;
  makeParameter: NnNamespaceHookCallback<[name: string, data: Float32Array, shape?: readonly number[], layout?: string], NnParameter>;
  resolveParameters: NnNamespaceHookCallback<[target: unknown], NnParameter[]>;
  parameterNames: NnNamespaceHookCallback<[target: unknown, prefix?: string], readonly string[]>;
  parameterInfos: NnNamespaceHookCallback<[target: unknown, prefix?: string], readonly AnyRecord[]>;
  parameterInfo: NnNamespaceHookCallback<[target: unknown, nameOrIndex: unknown, prefix?: string], unknown>;
  zeroGrad: NnNamespaceHookCallback<[target: unknown, options?: ZeroGradOptions], unknown>;
  setRequiresGrad: NnNamespaceHookCallback<[target: unknown, requiresGrad?: boolean], unknown>;
  stateDict: NnNamespaceHookCallback<[target: unknown, prefix?: string], unknown>;
  loadStateDict: NnNamespaceHookCallback<[target: unknown, source: unknown, options?: LoadStateDictOptions], unknown>;
  traceSequentialProgram: NnNamespaceHookCallback<[target: unknown, options?: ModuleTraceOptions], unknown>;
  traceModule: NnNamespaceHookCallback<[target: unknown, options?: ModuleTraceOptions], unknown>;
  compileSupportForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  requireCompileSupportForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  compilerSignaturesForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  tensorProgramIrForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  kernelPlanForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  bufferLayoutForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  memoryLayoutForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  inputShapeForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  outputShapeForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  shapeConstraintsForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  parameterLayoutForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  explainModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  requireCompilePlanForModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  canCompileModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], boolean>;
  compileModule: NnNamespaceHookCallback<[target: unknown, options?: CompileOptions], unknown>;
  bindModuleParameters: NnNamespaceHookCallback<[target: unknown, bindings?: unknown], unknown>;
  placeModuleParameters: NnNamespaceHookCallback<[target: unknown, program: unknown, placement?: unknown], unknown>;
  bindingPlanForModuleBindings: NnNamespaceHookCallback<[target: unknown, bindings?: unknown], unknown>;
  requireBindingPlanForModuleBindings: NnNamespaceHookCallback<[target: unknown, bindings?: unknown], unknown>;
};

export type NnNamespaceOptions = Readonly<Record<string, unknown> & NnNamespaceHooks>;

export function createNnNamespace(options: NnNamespaceOptions) {
  const LinearModule = options.LinearModule;
  const EmbeddingModule = options.EmbeddingModule;
  const Conv2dModule = options.Conv2dModule;
  const AvgPool2dModule = options.AvgPool2dModule;
  const MaxPool2dModule = options.MaxPool2dModule as new(kernelSize: unknown, config?: UnknownRecord) => unknown;
  const SequentialModule = options.SequentialModule;
  const ActivationModule = options.ActivationModule;
  const SoftmaxModule = options.SoftmaxModule;
  const LogSoftmaxModule = options.LogSoftmaxModule;
  const ReductionModule = options.ReductionModule;
  const DropoutModule = options.DropoutModule;
  const ShapeModule = options.ShapeModule;
  const FeatureNormModule = options.FeatureNormModule;
  const MSELoss = options.MSELoss;
  const L1Loss = options.L1Loss;
  const HuberLoss = options.HuberLoss;
  const SmoothL1Loss = options.SmoothL1Loss;
  const BCELoss = options.BCELoss;
  const BCEWithLogitsLoss = options.BCEWithLogitsLoss;
  const CrossEntropyLoss = options.CrossEntropyLoss;
  const NLLLoss = options.NLLLoss;
  const geluScalar = options.geluScalar;
  const siluScalar = options.siluScalar;
  const sigmoidScalar = options.sigmoidScalar;
  const f32WithLength = options.f32WithLength;
  const makeParameter = options.makeParameter;
  const resolveParameters = options.resolveParameters;
  const parameterNames = options.parameterNames;
  const parameterInfos = options.parameterInfos;
  const parameterInfo = options.parameterInfo;
  const zeroGrad = options.zeroGrad;
  const setRequiresGrad = options.setRequiresGrad;
  const stateDict = options.stateDict;
  const loadStateDict = options.loadStateDict;
  const traceSequentialProgram = options.traceSequentialProgram;
  const traceModule = options.traceModule;
  const compileSupportForModule = options.compileSupportForModule;
  const requireCompileSupportForModule = options.requireCompileSupportForModule;
  const compilerSignaturesForModule = options.compilerSignaturesForModule;
  const tensorProgramIrForModule = options.tensorProgramIrForModule;
  const kernelPlanForModule = options.kernelPlanForModule;
  const bufferLayoutForModule = options.bufferLayoutForModule;
  const memoryLayoutForModule = options.memoryLayoutForModule;
  const inputShapeForModule = options.inputShapeForModule;
  const outputShapeForModule = options.outputShapeForModule;
  const shapeConstraintsForModule = options.shapeConstraintsForModule;
  const parameterLayoutForModule = options.parameterLayoutForModule;
  const explainModule = options.explainModule;
  const requireCompilePlanForModule = options.requireCompilePlanForModule;
  const canCompileModule = options.canCompileModule;
  const compileModule = options.compileModule;
  const bindModuleParameters = options.bindModuleParameters;
  const placeModuleParameters = options.placeModuleParameters;
  const bindingPlanForModuleBindings = options.bindingPlanForModuleBindings;
  const requireBindingPlanForModuleBindings = options.requireBindingPlanForModuleBindings;
  if (
    typeof LinearModule !== "function" ||
    typeof EmbeddingModule !== "function" ||
    typeof Conv2dModule !== "function" ||
    typeof AvgPool2dModule !== "function" ||
    typeof SequentialModule !== "function" ||
    typeof ActivationModule !== "function" ||
    typeof SoftmaxModule !== "function" ||
    typeof LogSoftmaxModule !== "function" ||
    typeof ReductionModule !== "function" ||
    typeof DropoutModule !== "function" ||
    typeof ShapeModule !== "function" ||
    typeof FeatureNormModule !== "function" ||
    typeof MSELoss !== "function" ||
    typeof L1Loss !== "function" ||
    typeof HuberLoss !== "function" ||
    typeof SmoothL1Loss !== "function" ||
    typeof BCELoss !== "function" ||
    typeof BCEWithLogitsLoss !== "function" ||
    typeof CrossEntropyLoss !== "function" ||
    typeof NLLLoss !== "function" ||
    typeof geluScalar !== "function" ||
    typeof siluScalar !== "function" ||
    typeof sigmoidScalar !== "function" ||
    typeof f32WithLength !== "function" ||
    typeof makeParameter !== "function" ||
    typeof resolveParameters !== "function" ||
    typeof parameterNames !== "function" ||
    typeof parameterInfos !== "function" ||
    typeof parameterInfo !== "function" ||
    typeof zeroGrad !== "function" ||
    typeof setRequiresGrad !== "function" ||
    typeof stateDict !== "function" ||
    typeof loadStateDict !== "function" ||
    typeof traceSequentialProgram !== "function" ||
    typeof traceModule !== "function" ||
    typeof compileSupportForModule !== "function" ||
    typeof requireCompileSupportForModule !== "function" ||
    typeof compilerSignaturesForModule !== "function" ||
    typeof tensorProgramIrForModule !== "function" ||
    typeof kernelPlanForModule !== "function" ||
    typeof bufferLayoutForModule !== "function" ||
    typeof memoryLayoutForModule !== "function" ||
    typeof inputShapeForModule !== "function" ||
    typeof outputShapeForModule !== "function" ||
    typeof shapeConstraintsForModule !== "function" ||
    typeof parameterLayoutForModule !== "function" ||
    typeof explainModule !== "function" ||
    typeof requireCompilePlanForModule !== "function" ||
    typeof canCompileModule !== "function" ||
    typeof compileModule !== "function" ||
    typeof bindModuleParameters !== "function" ||
    typeof placeModuleParameters !== "function" ||
    typeof bindingPlanForModuleBindings !== "function" ||
    typeof requireBindingPlanForModuleBindings !== "function"
  ) {
    throw new Error("nn namespace factory requires module constructors, loss constructors, state helpers, scalar activations, and compile helpers");
  }

  function shapeProduct(shape: readonly number[]) {
    let total = 1;
    for (const dim of shape) {
      if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`nn.parameter shape dimensions must be positive safe integers, got ${dim}`);
      total *= dim;
    }
    return total;
  }

  function isShape(value: unknown): value is readonly number[] {
    return Array.isArray(value) && value.every((dim) => Number.isSafeInteger(dim) && dim > 0);
  }

  function isOptionsRecord(value: unknown): value is UnknownRecord {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof Float32Array);
  }

  function normalizeModulePlacement(target?: TensorToTarget | TensorToOptions | unknown, options: TensorToOptions = {}) {
    const targetRecord = isOptionsRecord(target) ? target : null;
    const device = targetRecord
      ? targetRecord.device ?? targetRecord.backend ?? targetRecord.placement
      : target;
    const dtype = targetRecord ? targetRecord.dtype : undefined;
    const placement = device ?? options.device ?? options.backend ?? options.placement ?? dtype ?? options.dtype ?? "cpu";
    if (
      placement === "cpu" ||
      placement === "host" ||
      placement === "f32" ||
      placement === "float" ||
      placement === "float32" ||
      placement === undefined ||
      placement === null
    ) return;
    throw new Error(`nn.Module.to currently supports only cpu/f32 placement, got ${String(placement)}`);
  }

  function moduleTo<T>(target: T, placement?: TensorToTarget | TensorToOptions, options: TensorToOptions = {}): T {
    normalizeModulePlacement(placement, options);
    return target;
  }

  function registrationName(value: unknown, label: string) {
    if (typeof value !== "string" || value.length === 0) throw new Error(`${label} name must be a non-empty string`);
    if (value.includes(".")) throw new Error(`${label} name must not contain "."`);
    if (value === "kind" || value === "customModule" || value === "training" || value === "graph") {
      throw new Error(`${label} name is reserved: ${value}`);
    }
    return value;
  }

  function parameterData(value: unknown) {
    if (value instanceof Float32Array) return value;
    const record = value as UnknownRecord | null | undefined;
    return record && record.data instanceof Float32Array ? record.data : value;
  }

  function parameterShape(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return record && isShape(record.shape) ? record.shape : null;
  }

  function parameter(name: unknown, data: unknown, shapeOrOptions?: unknown, parameterOptions?: unknown) {
    if (typeof name !== "string" || name.length === 0) throw new Error("nn.parameter requires a non-empty parameter name");
    const optionsRecord = isOptionsRecord(shapeOrOptions) && !isShape(shapeOrOptions)
      ? shapeOrOptions
      : isOptionsRecord(parameterOptions)
        ? parameterOptions
        : {};
    const source = parameterData(data);
    const shape = isShape(shapeOrOptions)
      ? shapeOrOptions
      : isShape(optionsRecord.shape)
        ? optionsRecord.shape
        : parameterShape(data) ?? (source instanceof Float32Array ? [source.length] : null);
    if (!shape) throw new Error("nn.parameter requires a shape when data is not a Tensor or Float32Array");
    const length = shapeProduct(shape);
    const values = source instanceof Float32Array && source.length === length
      ? source
      : f32WithLength(source, length, `nn.parameter ${name}`);
    const layout = typeof optionsRecord.layout === "string" && optionsRecord.layout.length > 0 ? optionsRecord.layout : "row-major";
    return makeParameter(name, values, shape, layout);
  }

  function bufferData(value: unknown) {
    if (value instanceof Float32Array) return value;
    const record = value as UnknownRecord | null | undefined;
    return record && record.data instanceof Float32Array ? record.data : value;
  }

  function bufferShape(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return record && isShape(record.shape) ? record.shape : null;
  }

  function buffer(name: unknown, data: unknown, shapeOrOptions?: unknown, bufferOptions?: unknown) {
    if (typeof name !== "string" || name.length === 0) throw new Error("nn.buffer requires a non-empty buffer name");
    const optionsRecord = isOptionsRecord(shapeOrOptions) && !isShape(shapeOrOptions)
      ? shapeOrOptions
      : isOptionsRecord(bufferOptions)
        ? bufferOptions
        : {};
    const source = bufferData(data);
    const shape = isShape(shapeOrOptions)
      ? shapeOrOptions
      : isShape(optionsRecord.shape)
        ? optionsRecord.shape
        : bufferShape(data) ?? (source instanceof Float32Array ? [source.length] : null);
    if (!shape) throw new Error("nn.buffer requires a shape when data is not a Tensor or Float32Array");
    const length = shapeProduct(shape);
    const values = source instanceof Float32Array && source.length === length
      ? source
      : f32WithLength(source, length, `nn.buffer ${name}`);
    const layout = typeof optionsRecord.layout === "string" && optionsRecord.layout.length > 0 ? optionsRecord.layout : "row-major";
    const persistent = optionsRecord.persistent !== false;
    return Object.freeze({
      kind: "buffer",
      name,
      data: values,
      shape: Object.freeze(Array.from(shape)),
      layout,
      persistent,
      tensor: data && typeof data === "object" && data !== source ? data : undefined,
    }) as NnBuffer;
  }

  const LayerNormModule = class LayerNormModule extends FeatureNormModule {
    constructor(features: unknown, config: UnknownRecord = {}) {
      super("layerNorm", features, config);
    }
  };
  const RMSNormModule = class RMSNormModule extends FeatureNormModule {
    constructor(features: unknown, config: UnknownRecord = {}) {
      super("rmsNorm", features, config);
    }
  };
  const BatchNorm1dModule = class BatchNorm1dModule extends FeatureNormModule {
    constructor(features: unknown, config: UnknownRecord = {}) {
      super("batchNorm1d", features, config);
    }
  };
  const IdentityModule = class IdentityModule extends ShapeModule {
    constructor() {
      super("identity");
    }
  };
  const DiagonalModule = class DiagonalModule extends ShapeModule {
    constructor() {
      super("diagonal");
    }
  };
  const RepeatModule = class RepeatModule extends ShapeModule {
    constructor(repeats: unknown) {
      super("repeat", repeats);
    }
  };
  const TileModule = class TileModule extends ShapeModule {
    constructor(repeats: unknown) {
      super("tile", repeats);
    }
  };
  const FlattenModule = class FlattenModule extends ShapeModule {
    constructor(startDim = 0, endDim = -1) {
      super("flatten", startDim, endDim);
    }
  };
  const ReLUModule = class ReLUModule extends ActivationModule {
    constructor() {
      super("relu", (x: number) => Math.max(0, x));
    }
  };
  const GELUModule = class GELUModule extends ActivationModule {
    constructor() {
      super("gelu", geluScalar);
    }
  };
  const SiLUModule = class SiLUModule extends ActivationModule {
    constructor() {
      super("silu", siluScalar);
    }
  };
  const SigmoidModule = class SigmoidModule extends ActivationModule {
    constructor() {
      super("sigmoid", sigmoidScalar);
    }
  };
  const TanhModule = class TanhModule extends ActivationModule {
    constructor() {
      super("tanh", Math.tanh);
    }
  };

  function moduleMethod(target: NnModuleTarget, name: keyof NnModule | string) {
    const method = (target as Readonly<Record<string, unknown>>)[String(name)];
    if (!target || typeof method !== "function") throw new Error(`nn.${name} requires an nn module`);
    return method.bind(target);
  }

  function prefixedParameter(prefix: string, param: AnyRecord) {
    if (!prefix || !param || typeof param !== "object") return param;
    if (typeof param.name === "string" && (param.name === prefix || param.name.startsWith(`${prefix}.`))) return param;
    const name = typeof param.name === "string" ? `${prefix}.${param.name}` : String(prefix);
    return { ...param, name };
  }

  function namedParametersForTarget(target: NnModuleTarget, prefixOrOptions: unknown = "", options = {}) {
    const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
    if (target && typeof target.namedParameters === "function") return target.namedParameters(prefixOrOptions, options);
    const params = resolveParameters(target);
    if (!prefix) return params;
    return params.map((param: AnyRecord) => prefixedParameter(String(prefix), param));
  }

  function namedBuffersForTarget(target: NnModuleTarget, prefixOrOptions: unknown = "", options = {}) {
    if (target && typeof target.namedBuffers === "function") return target.namedBuffers(prefixOrOptions, options);
    return defaultBuffersForTarget(target as AnyRecord, prefixOrOptions, "nn.Module", options);
  }

  function parameterNamesForTarget(target: NnModuleTarget, prefix = "") {
    if (target && typeof target.parameterNames === "function") return target.parameterNames(prefix);
    return parameterNames(target, prefix);
  }

  function parameterInfosForTarget(target: NnModuleTarget, prefix = "") {
    if (target && typeof target.parameterInfos === "function") return target.parameterInfos(prefix);
    return parameterInfos(target, prefix);
  }

  function parameterInfoForTarget(target: NnModuleTarget, nameOrIndex: unknown, prefix = "") {
    if (target && typeof target.parameterInfo === "function") return target.parameterInfo(nameOrIndex, prefix);
    return parameterInfo(target, nameOrIndex, prefix);
  }

  function sequentialLayers(first: unknown, rest: readonly unknown[]): readonly NnModule[] {
    if (rest.length === 0 && Array.isArray(first)) return first;
    return first === undefined && rest.length === 0 ? [] : [first, ...rest] as readonly NnModule[];
  }

  function moduleTarget(target: unknown): NnModuleTarget {
    return Array.isArray(target) ? new SequentialModule(target as readonly NnModule[]) : target as NnModuleTarget;
  }

  function isModuleLikeTarget(target: unknown) {
    const moduleLike = target as Partial<Record<keyof NnModule | "parameters" | "children" | "trace" | "compile", unknown>>;
    return Boolean(target && typeof target === "object" && (
      typeof moduleLike.forward === "function" ||
      typeof moduleLike.parameters === "function" ||
      typeof moduleLike.children === "function" ||
      typeof moduleLike.trace === "function" ||
      typeof moduleLike.compile === "function"
    ));
  }

  function stateTarget(target: unknown) {
    return Array.isArray(target) && target.length > 0 && target.every(isModuleLikeTarget)
      ? new SequentialModule(target)
      : target;
  }

  function moduleNativeInference(target: unknown, compileOptions: CompileOptions = {}, bindOptions?: unknown) {
    const module = moduleTarget(target);
    const program = compileModule(module, compileOptions) as NativeInferenceProgram;
    if (!program || typeof program !== "object") {
      throw new Error("nn.native expected compile() to return a Program");
    }
    const session = typeof program.bindModule === "function"
      ? program.bindModule(module, bindOptions)
      : typeof program.bind === "function" && bindOptions !== undefined
        ? program.bind(bindOptions)
        : null;
    if (!session || typeof session !== "object" || typeof (session as Partial<NativeInferenceSession>).stepTensor !== "function") {
      throw new Error("nn.native expected bindModule() or explicit Program bindings to return a Session");
    }
    const nativeSession = session as NativeInferenceSession;
    let disposed = false;
    const dispose = () => {
      if (disposed) return;
      disposed = true;
      if (typeof nativeSession.dispose === "function") nativeSession.dispose();
      if (typeof program.dispose === "function") program.dispose();
    };
    return Object.freeze({
      program,
      session: nativeSession,
      forward(input: unknown) {
        return nativeSession.stepTensor(input);
      },
      call(input: unknown) {
        return nativeSession.stepTensor(input);
      },
      __call__(input: unknown) {
        return nativeSession.stepTensor(input);
      },
      stepTensor(input: unknown) {
        return nativeSession.stepTensor(input);
      },
      into(output: Float32Array, input: unknown) {
        return nativeSession.executeInto(output, { input });
      },
      prepareInto(output: Float32Array, input: unknown) {
        return nativeSession.prepareExecuteInto(output, { input });
      },
      explain() {
        return explainModule(module, compileOptions);
      },
      preflight() {
        return explainModule(module, compileOptions);
      },
      compileSupport() {
        return compileSupportForModule(module, compileOptions);
      },
      inputShape() {
        return typeof program.inputShape === "function" ? program.inputShape() : inputShapeForModule(module, compileOptions);
      },
      outputShape() {
        return typeof program.outputShape === "function" ? program.outputShape() : outputShapeForModule(module, compileOptions);
      },
      kernelPlan() {
        return typeof program.kernelPlan === "function" ? program.kernelPlan() : kernelPlanForModule(module, compileOptions);
      },
      compilerSignatures() {
        return typeof program.compilerSignatures === "function" ? program.compilerSignatures() : compilerSignaturesForModule(module, compileOptions);
      },
      dispose,
      free: dispose,
    });
  }

  function stateDictPrefix(prefixOrOptions: unknown = "") {
    if (prefixOrOptions && typeof prefixOrOptions === "object" && !Array.isArray(prefixOrOptions)) {
      const prefix = (prefixOrOptions as AnyRecord).prefix;
      return prefix === undefined || prefix === null ? "" : String(prefix);
    }
    return prefixOrOptions === undefined || prefixOrOptions === null ? "" : String(prefixOrOptions);
  }

  function stateDictForTarget(target: unknown, prefixOrOptions: unknown = "") {
    const prefix = stateDictPrefix(prefixOrOptions);
    const targetForState = stateTarget(target);
    return isModuleLikeTarget(targetForState)
      ? stateDictWithBuffers(targetForState as AnyRecord, prefix)
      : stateDict(targetForState, prefix);
  }

  function loadStateDictForTarget(target: unknown, source: unknown, options?: LoadStateDictOptions) {
    const targetForState = stateTarget(target);
    const result = isModuleLikeTarget(targetForState)
      ? loadStateDictWithBuffers(targetForState as AnyRecord, source, options)
      : loadStateDict(targetForState, source, options);
    return result ?? targetForState;
  }

  function graphSourceForTarget(target: AnyRecord) {
    return target.graph;
  }

  function isModuleChild(value: unknown, owner: AnyRecord) {
    return value !== owner &&
      !isModuleListLike(value) &&
      !isModuleDictLike(value) &&
      !isParameterListLike(value) &&
      !isParameterDictLike(value) &&
      !isBufferLike(value) &&
      isModuleLikeTarget(value);
  }

  function isParameterLike(value: unknown) {
    const param = value as Partial<NnParameter> | null | undefined;
    return Boolean(param && typeof param === "object" && param.data instanceof Float32Array);
  }

  function initTargetData(target: unknown, label: string) {
    const record = target as (Partial<NnParameter> & { tensor?: Partial<{ data: Float32Array }> }) | null | undefined;
    const data = record && record.data instanceof Float32Array
      ? record.data
      : record && record.tensor && record.tensor.data instanceof Float32Array
        ? record.tensor.data
        : null;
    if (!data) throw new Error(`${label} requires an nn.Parameter or Tensor with Float32Array data`);
    return data;
  }

  function initTargetShape(target: unknown, data: Float32Array) {
    const record = target as (Partial<NnParameter> & { tensor?: Partial<{ shape: readonly number[] }> }) | null | undefined;
    if (record && record.tensor && isShape(record.tensor.shape)) return record.tensor.shape;
    const shape = (record as Partial<{ shape: readonly number[] }> | null | undefined)?.shape;
    return isShape(shape) ? shape : [data.length];
  }

  function normalSample(mean: number, std: number) {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function fanInOut(shape: readonly number[], label: string) {
    if (shape.length < 2) throw new Error(`${label} requires a tensor with at least 2 dimensions`);
    const receptiveField = shape.slice(2).reduce((total, dim) => total * dim, 1);
    return {
      fanIn: shape[1] * receptiveField,
      fanOut: shape[0] * receptiveField,
    };
  }

  function constantInit<T>(target: T, value: unknown, label = "nn.init.constant_") {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} value must be a finite number`);
    initTargetData(target, label).fill(value);
    return target;
  }

  function uniformInit<T>(target: T, min = 0, max = 1, label = "nn.init.uniform_") {
    if (typeof min !== "number" || !Number.isFinite(min)) throw new Error(`${label} min must be a finite number`);
    if (typeof max !== "number" || !Number.isFinite(max)) throw new Error(`${label} max must be a finite number`);
    if (max < min) throw new Error(`${label} max must be >= min`);
    const data = initTargetData(target, label);
    const span = max - min;
    for (let i = 0; i < data.length; i += 1) data[i] = min + Math.random() * span;
    return target;
  }

  function normalInit<T>(target: T, mean = 0, std = 1, label = "nn.init.normal_") {
    if (typeof mean !== "number" || !Number.isFinite(mean)) throw new Error(`${label} mean must be a finite number`);
    if (typeof std !== "number" || !Number.isFinite(std) || std < 0) throw new Error(`${label} std must be a non-negative finite number`);
    const data = initTargetData(target, label);
    for (let i = 0; i < data.length; i += 1) data[i] = normalSample(mean, std);
    return target;
  }

  function xavierUniformInit<T>(target: T, gain = 1, label = "nn.init.xavier_uniform_") {
    if (typeof gain !== "number" || !Number.isFinite(gain)) throw new Error(`${label} gain must be a finite number`);
    const data = initTargetData(target, label);
    const shape = initTargetShape(target, data);
    const { fanIn, fanOut } = fanInOut(shape, label);
    const bound = gain * Math.sqrt(6 / (fanIn + fanOut));
    return uniformInit(target, -bound, bound, label);
  }

  function xavierNormalInit<T>(target: T, gain = 1, label = "nn.init.xavier_normal_") {
    if (typeof gain !== "number" || !Number.isFinite(gain)) throw new Error(`${label} gain must be a finite number`);
    const data = initTargetData(target, label);
    const shape = initTargetShape(target, data);
    const { fanIn, fanOut } = fanInOut(shape, label);
    const std = gain * Math.sqrt(2 / (fanIn + fanOut));
    return normalInit(target, 0, std, label);
  }

  function kaimingGain(options: UnknownRecord, label: string) {
    const nonlinearity = options.nonlinearity ?? "leaky_relu";
    const negativeSlope = Number(options.a ?? options.negativeSlope ?? options.negative_slope ?? 0);
    if (!Number.isFinite(negativeSlope)) throw new Error(`${label} negative slope must be a finite number`);
    if (nonlinearity === "linear" || nonlinearity === "sigmoid") return 1;
    if (nonlinearity === "tanh") return 5 / 3;
    if (nonlinearity === "relu") return Math.sqrt(2);
    if (nonlinearity === "leaky_relu") return Math.sqrt(2 / (1 + negativeSlope * negativeSlope));
    throw new Error(`${label} nonlinearity must be "linear", "sigmoid", "tanh", "relu", or "leaky_relu"`);
  }

  function kaimingUniformInit<T>(target: T, options: UnknownRecord = {}, label = "nn.init.kaiming_uniform_") {
    const data = initTargetData(target, label);
    const shape = initTargetShape(target, data);
    const { fanIn, fanOut } = fanInOut(shape, label);
    const mode = options.mode ?? "fan_in";
    const fan = mode === "fan_in" || mode === "fanIn"
      ? fanIn
      : mode === "fan_out" || mode === "fanOut"
        ? fanOut
        : null;
    if (fan === null) throw new Error(`${label} mode must be "fan_in" or "fan_out"`);
    const bound = Math.sqrt(3) * kaimingGain(options, label) / Math.sqrt(fan);
    return uniformInit(target, -bound, bound, label);
  }

  function kaimingNormalInit<T>(target: T, options: UnknownRecord = {}, label = "nn.init.kaiming_normal_") {
    const data = initTargetData(target, label);
    const shape = initTargetShape(target, data);
    const { fanIn, fanOut } = fanInOut(shape, label);
    const mode = options.mode ?? "fan_in";
    const fan = mode === "fan_in" || mode === "fanIn"
      ? fanIn
      : mode === "fan_out" || mode === "fanOut"
        ? fanOut
        : null;
    if (fan === null) throw new Error(`${label} mode must be "fan_in" or "fan_out"`);
    const std = kaimingGain(options, label) / Math.sqrt(fan);
    return normalInit(target, 0, std, label);
  }

  function isBufferLike(value: unknown) {
    const record = value as Partial<NnBuffer> | null | undefined;
    return Boolean(record && typeof record === "object" && record.kind === "buffer" && record.data instanceof Float32Array);
  }

  function isModuleListLike(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return Boolean(record && record.kind === "moduleList" && Array.isArray(record.layers));
  }

  function isParameterListLike(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return Boolean(record && record.kind === "parameterList" && Array.isArray(record.params));
  }

  function isModuleDictLike(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return Boolean(record && record.kind === "moduleDict" && record.layers && typeof record.layers === "object" && !Array.isArray(record.layers));
  }

  function isParameterDictLike(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return Boolean(record && record.kind === "parameterDict" && record.params && typeof record.params === "object" && !Array.isArray(record.params));
  }

  function moduleChildEntriesFromOwnProperties(target: AnyRecord) {
    const entries: { name: string; module: unknown }[] = [];
    for (const name of Object.keys(target)) {
      if (name === "kind" || name === "customModule" || name === "training" || name === "graph") continue;
      const value = target[name];
      if (isModuleListLike(value)) {
        (value as AnyRecord).layers.forEach((item: unknown, index: number) => {
          if (isModuleChild(item, target)) entries.push({ name: `${name}.${index}`, module: item });
        });
        continue;
      }
      if (isModuleDictLike(value)) {
        for (const [key, item] of Object.entries((value as AnyRecord).layers as AnyRecord)) {
          if (isModuleChild(item, target)) entries.push({ name: `${name}.${key}`, module: item });
        }
        continue;
      }
      if (isModuleChild(value, target)) {
        entries.push({ name, module: value });
        continue;
      }
      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          if (isModuleChild(item, target)) entries.push({ name: `${name}.${index}`, module: item });
        });
      }
    }
    return entries;
  }

  function parameterEntriesFromOwnProperties(target: AnyRecord) {
    const entries: { name: string; parameter: NnParameter }[] = [];
    for (const name of Object.keys(target)) {
      if (name === "kind" || name === "customModule" || name === "training" || name === "graph") continue;
      const value = target[name];
      if (isModuleListLike(value)) continue;
      if (isModuleDictLike(value)) continue;
      if (isParameterListLike(value)) {
        for (const param of (value as AnyRecord).parameters(name) as readonly AnyRecord[]) {
          if (isParameterLike(param) && typeof param.name === "string") entries.push({ name: param.name, parameter: param as NnParameter });
        }
        continue;
      }
      if (isParameterDictLike(value)) {
        for (const param of (value as AnyRecord).parameters(name) as readonly AnyRecord[]) {
          if (isParameterLike(param) && typeof param.name === "string") entries.push({ name: param.name, parameter: param as NnParameter });
        }
        continue;
      }
      if (isModuleChild(value, target)) continue;
      if (isBufferLike(value)) continue;
      if (isParameterLike(value)) {
        entries.push({ name, parameter: value as NnParameter });
        continue;
      }
      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          if (isModuleChild(item, target)) return;
          if (isParameterLike(item)) entries.push({ name: `${name}.${index}`, parameter: item as NnParameter });
        });
      }
    }
    return entries;
  }

  function bufferEntriesFromOwnProperties(target: AnyRecord) {
    const entries: { name: string; buffer: NnBuffer }[] = [];
    for (const name of Object.keys(target)) {
      if (name === "kind" || name === "customModule" || name === "training" || name === "graph") continue;
      const value = target[name];
      if (isBufferLike(value)) {
        entries.push({ name, buffer: value as NnBuffer });
      }
    }
    return entries;
  }

  function prefixedBuffer(prefix: string, entryName: string, targetBuffer: NnBuffer) {
    if (!prefix && targetBuffer.name === entryName) return targetBuffer;
    return {
      ...targetBuffer,
      name: prefix ? `${prefix}.${entryName}` : entryName,
    };
  }

  function defaultBuffersForTarget(target: AnyRecord, prefixOrOptions: unknown = "", label = "nn.Module", options = {}) {
    const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
    const buffers = bufferEntriesFromOwnProperties(target)
      .filter((entry) => entry.buffer.persistent !== false)
      .map((entry) => prefixedBuffer(prefix, entry.name, entry.buffer));
    if (!recurse) return Object.freeze(buffers);
    const entries = typeof target.namedChildren === "function" ? target.namedChildren(prefix) : namedChildrenForTarget(target, prefix, label);
    if (!Array.isArray(entries)) throw new Error(`${label} namedChildren must return an array`);
    buffers.push(...entries.flatMap((entry: AnyRecord) => {
      const child = entry && entry.module;
      if (!child || typeof child.namedBuffers !== "function") return [];
      return child.namedBuffers(String(entry.name));
    }));
    return Object.freeze(buffers);
  }

  function stateEntryForBuffer(targetBuffer: NnBuffer) {
    const shape = Object.freeze(Array.from(targetBuffer.shape));
    const entry = {
      shape,
      layout: targetBuffer.layout ?? "row-major",
      data: new Float32Array(targetBuffer.data),
    } as AnyRecord;
    entry.signature = [
      "module-buffer-state-entry",
      `shape=${shape.join("x")}`,
      `layout=${entry.layout}`,
      `len=${entry.data.length}`,
    ].join("|");
    return Object.freeze(entry);
  }

  function stateShape(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    if (record && Array.isArray(record.shape)) return Array.from(record.shape, Number);
    return null;
  }

  function stateLayout(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return record && typeof record.layout === "string" ? record.layout : null;
  }

  function stateData(value: unknown) {
    const record = value as UnknownRecord | null | undefined;
    return record && record.data !== undefined ? record.data : value;
  }

  function sameShape(a: readonly number[], b: readonly number[]) {
    return a.length === b.length && a.every((dim, index) => dim === b[index]);
  }

  function stateKeys(source: unknown) {
    if (source instanceof Map) return Array.from(source.keys());
    if (source && typeof source === "object") return Object.keys(source);
    throw new Error("state dict must be an object or Map");
  }

  function stateValue(source: unknown, name: string) {
    if (source instanceof Map) return source.get(name);
    if (source && typeof source === "object") {
      return Object.prototype.hasOwnProperty.call(source, name) ? (source as UnknownRecord)[name] : undefined;
    }
    throw new Error("state dict must be an object or Map");
  }

  function stateDictWithBuffers(target: AnyRecord, prefixOrOptions: unknown = "") {
    const prefix = stateDictPrefix(prefixOrOptions);
    const out = { ...(stateDict(target, prefix) as AnyRecord) };
    for (const targetBuffer of defaultBuffersForTarget(target, prefix)) {
      out[targetBuffer.name] = stateEntryForBuffer(targetBuffer as NnBuffer);
    }
    return Object.freeze(out);
  }

  function loadStateDictWithBuffers(target: AnyRecord, source: unknown, options: LoadStateDictOptions = {}) {
    const strict = options.strict !== false;
    const validateOnly = options.validateOnly === true;
    const prefix = typeof options.prefix === "string" ? options.prefix : "";
    const params = namedParametersForTarget(target, prefix) as readonly (NnParameter & { name: string; data: Float32Array })[];
    const buffers = defaultBuffersForTarget(target, prefix) as readonly NnBuffer[];
    const expected = new Set([...params.map((param) => String(param.name)), ...buffers.map((targetBuffer) => String(targetBuffer.name))]);
    const pendingParams: Array<{ param: NnParameter & { data: Float32Array }; data: Float32Array }> = [];
    const pendingBuffers: Array<{ targetBuffer: NnBuffer; data: Float32Array }> = [];
    if (strict) {
      for (const key of stateKeys(source)) {
        if (!expected.has(String(key))) throw new Error(`state dict has unexpected entry ${String(key)}`);
      }
    }
    for (const param of params) {
      const name = String(param.name);
      const value = stateValue(source, name);
      if (value === undefined) {
        if (strict) throw new Error(`state dict is missing parameter ${name}`);
        continue;
      }
      const shape = stateShape(value);
      if (shape && param.tensor?.shape && !sameShape(shape, param.tensor.shape)) {
        throw new Error(`state dict ${name} shape must be [${param.tensor.shape.join(",")}], got [${shape.join(",")}]`);
      }
      const layout = stateLayout(value);
      if (layout && param.layout && layout !== param.layout) throw new Error(`state dict ${name} layout must be ${param.layout}, got ${layout}`);
      pendingParams.push({ param, data: f32WithLength(stateData(value), param.data.length, `state dict ${name}`) });
    }
    for (const targetBuffer of buffers) {
      const name = String(targetBuffer.name);
      const value = stateValue(source, name);
      if (value === undefined) {
        if (strict) throw new Error(`state dict is missing buffer ${name}`);
        continue;
      }
      const shape = stateShape(value);
      if (shape && !sameShape(shape, targetBuffer.shape)) {
        throw new Error(`state dict ${name} shape must be [${targetBuffer.shape.join(",")}], got [${shape.join(",")}]`);
      }
      const layout = stateLayout(value);
      if (layout && targetBuffer.layout && layout !== targetBuffer.layout) throw new Error(`state dict ${name} layout must be ${targetBuffer.layout}, got ${layout}`);
      pendingBuffers.push({ targetBuffer, data: f32WithLength(stateData(value), targetBuffer.data.length, `state dict ${name}`) });
    }
    if (!validateOnly) {
      for (const { param, data } of pendingParams) param.data.set(data);
      for (const { targetBuffer, data } of pendingBuffers) targetBuffer.data.set(data);
    }
    return target;
  }

  function defaultChildEntriesForTarget(target: AnyRecord, label = "nn.Module") {
    if (graphSourceForTarget(target) !== undefined) {
      return Object.freeze([{ name: "graph", module: graphModuleForTarget(target, label) }]);
    }
    return Object.freeze(moduleChildEntriesFromOwnProperties(target));
  }

  function graphModuleForTarget(target: AnyRecord, label = "nn.Module") {
    const graphSource = graphSourceForTarget(target);
    const graph = typeof graphSource === "function" ? graphSource.call(target) : graphSource;
    if (graph && typeof graph === "object") return graph as AnyRecord;
    const childEntries = moduleChildEntriesFromOwnProperties(target);
    if (childEntries.length > 0) {
      return new SequentialModule(childEntries.map((entry) => entry.module as NnModule)) as AnyRecord;
    }
    if (graphSource !== undefined) {
      throw new Error(`${label} graph must resolve to an nn module`);
    } else {
      throw new Error(`${label} compile requires a declared graph module`);
    }
  }

  function graphMethodForTarget(target: AnyRecord, name: string, label = "nn.Module") {
    const graph = graphModuleForTarget(target, label);
    const method = graph[name];
    if (typeof method !== "function") throw new Error(`${label} graph must support ${name}()`);
    return method.bind(graph);
  }

  function listInsertIndex(index: unknown, label: string) {
    if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be an integer`);
    return index as number;
  }

  function listSetIndex(index: unknown, length: number, label: string) {
    if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be an integer`);
    const numeric = index as number;
    const normalized = numeric < 0 ? length + numeric : numeric;
    if (normalized < 0 || normalized >= length) throw new Error(`${label} index out of range`);
    return normalized;
  }

  function iterableValues(value: unknown, label: string) {
    if (!value || typeof (value as Iterable<unknown>)[Symbol.iterator] !== "function") {
      throw new Error(`${label} requires an iterable`);
    }
    return Array.from(value as Iterable<unknown>);
  }

  class ParameterList {
    readonly kind = "parameterList";
    readonly params: NnParameter[];

    constructor(params: Iterable<unknown> = []) {
      const values = iterableValues(params, "nn.ParameterList");
      this.params = values.map((param, index) => {
        if (!isParameterLike(param)) throw new Error(`nn.ParameterList item ${index} must be an nn.Parameter`);
        return param as NnParameter;
      });
    }

    get length() {
      return this.params.length;
    }

    len() {
      return this.params.length;
    }

    __len__() {
      return this.len();
    }

    size() {
      return this.len();
    }

    at(index: number) {
      return this.params.at(index);
    }

    get(index: number) {
      return this.at(index);
    }

    __getitem__(index: number) {
      return this.at(index);
    }

    __setitem__(index: unknown, parameterValue: unknown) {
      if (!isParameterLike(parameterValue)) throw new Error("nn.ParameterList.__setitem__ requires an nn.Parameter");
      this.params[listSetIndex(index, this.params.length, "nn.ParameterList.__setitem__")] = parameterValue as NnParameter;
      return this;
    }

    __delitem__(index: unknown) {
      this.params.splice(listSetIndex(index, this.params.length, "nn.ParameterList.__delitem__"), 1);
      return this;
    }

    pop(index: unknown = -1) {
      const [value] = this.params.splice(listSetIndex(index, this.params.length, "nn.ParameterList.pop"), 1);
      return value;
    }

    clear() {
      this.params.length = 0;
      return this;
    }

    push(parameterValue: unknown) {
      if (!isParameterLike(parameterValue)) throw new Error("nn.ParameterList.push requires an nn.Parameter");
      this.params.push(parameterValue as NnParameter);
      return this;
    }

    append(parameterValue: unknown) {
      return this.push(parameterValue);
    }

    insert(index: unknown, parameterValue: unknown) {
      if (!isParameterLike(parameterValue)) throw new Error("nn.ParameterList.insert requires an nn.Parameter");
      this.params.splice(listInsertIndex(index, "nn.ParameterList.insert"), 0, parameterValue as NnParameter);
      return this;
    }

    extend(parameterValues: Iterable<unknown>) {
      if (!parameterValues || typeof parameterValues[Symbol.iterator] !== "function") {
        throw new Error("nn.ParameterList.extend requires an iterable of nn.Parameter values");
      }
      for (const parameterValue of parameterValues) this.push(parameterValue);
      return this;
    }

    [Symbol.iterator]() {
      return this.params[Symbol.iterator]();
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      return this.params.map((param, index) => ({ ...param, name: prefix ? `${prefix}.${index}` : String(index) }));
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    named_parameters(prefixOrOptions: unknown = "", options = {}) {
      return this.namedParameters(prefixOrOptions, options);
    }

    getParameter(name: unknown) {
      const targetName = lookupName(name, "nn.ParameterList.getParameter");
      if (!/^(0|[1-9]\d*)$/.test(targetName)) return null;
      return this.params[Number(targetName)] ?? null;
    }

    get_parameter(name: unknown) {
      return this.getParameter(name);
    }

    buffers(prefixOrOptions: unknown = "", options = {}) {
      return defaultBuffersForTarget(this, prefixOrOptions, "nn.ParameterList", options);
    }

    namedBuffers(prefixOrOptions: unknown = "", options = {}) {
      return this.buffers(prefixOrOptions, options);
    }

    named_buffers(prefixOrOptions: unknown = "", options = {}) {
      return this.namedBuffers(prefixOrOptions, options);
    }

    getBuffer(name: unknown) {
      lookupName(name, "nn.ParameterList.getBuffer");
      return null;
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    parameterNames(prefix = "") {
      return parameterNames(this, prefix);
    }

    parameterInfos(prefix = "") {
      return parameterInfos(this, prefix);
    }

    parameterInfo(nameOrIndex: unknown, prefix = "") {
      return parameterInfo(this, nameOrIndex, prefix);
    }

    zeroGrad(options?: ZeroGradOptions) {
      zeroGrad(this, options);
    }

    zero_grad(options?: ZeroGradOptions) {
      return this.zeroGrad(options);
    }

    requiresGrad_(requiresGrad = true) {
      setRequiresGrad(this, requiresGrad);
      return this;
    }

    requires_grad_(requiresGrad = true) {
      return this.requiresGrad_(requiresGrad);
    }

    stateDict(prefixOrOptions: unknown = "") {
      return stateDictWithBuffers(this, prefixOrOptions);
    }

    state_dict(prefixOrOptions: unknown = "") {
      return this.stateDict(prefixOrOptions);
    }

    loadStateDict(source: unknown, options?: LoadStateDictOptions) {
      loadStateDictWithBuffers(this, source, options);
      return this;
    }

    load_state_dict(source: unknown, options?: LoadStateDictOptions) {
      return this.loadStateDict(source, options);
    }
  }

  class ParameterDict {
    readonly kind = "parameterDict";
    readonly params: Record<string, NnParameter>;

    constructor(params: Readonly<Record<string, unknown>> = {}) {
      if (!isOptionsRecord(params)) throw new Error("nn.ParameterDict requires a record of nn.Parameter values");
      this.params = {};
      for (const [key, param] of Object.entries(params)) {
        this.set(key, param);
      }
    }

    get length() {
      return Object.keys(this.params).length;
    }

    len() {
      return this.length;
    }

    __len__() {
      return this.len();
    }

    size() {
      return this.len();
    }

    get(name: unknown) {
      return this.params[registrationName(name, "nn.ParameterDict.get")];
    }

    __getitem__(name: unknown) {
      return this.get(name);
    }

    __setitem__(name: unknown, parameterValue: unknown) {
      return this.set(name, parameterValue);
    }

    set(name: unknown, parameterValue: unknown) {
      const key = registrationName(name, "nn.ParameterDict.set");
      if (!isParameterLike(parameterValue)) throw new Error(`nn.ParameterDict.set requires an nn.Parameter for ${key}`);
      this.params[key] = parameterValue as NnParameter;
      return this;
    }

    update(params: Readonly<Record<string, unknown>>) {
      if (!isOptionsRecord(params)) throw new Error("nn.ParameterDict.update requires a record of nn.Parameter values");
      for (const [key, param] of Object.entries(params)) this.set(key, param);
      return this;
    }

    has(name: unknown) {
      return Object.prototype.hasOwnProperty.call(this.params, registrationName(name, "nn.ParameterDict.has"));
    }

    __contains__(name: unknown) {
      return this.has(name);
    }

    pop(name: unknown) {
      const key = registrationName(name, "nn.ParameterDict.pop");
      const value = this.params[key];
      delete this.params[key];
      return value;
    }

    __delitem__(name: unknown) {
      this.pop(name);
      return this;
    }

    clear() {
      for (const key of Object.keys(this.params)) delete this.params[key];
      return this;
    }

    keys() {
      return Object.freeze(Object.keys(this.params));
    }

    values() {
      return Object.freeze(Object.values(this.params));
    }

    entries() {
      return Object.freeze(Object.entries(this.params));
    }

    items() {
      return this.entries();
    }

    [Symbol.iterator]() {
      return this.entries()[Symbol.iterator]();
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      return Object.entries(this.params).map(([key, param]) => ({ ...param, name: prefix ? `${prefix}.${key}` : key }));
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    named_parameters(prefixOrOptions: unknown = "", options = {}) {
      return this.namedParameters(prefixOrOptions, options);
    }

    getParameter(name: unknown) {
      return this.params[lookupName(name, "nn.ParameterDict.getParameter")] ?? null;
    }

    get_parameter(name: unknown) {
      return this.getParameter(name);
    }

    buffers(prefixOrOptions: unknown = "", options = {}) {
      return defaultBuffersForTarget(this, prefixOrOptions, "nn.ParameterDict", options);
    }

    namedBuffers(prefixOrOptions: unknown = "", options = {}) {
      return this.buffers(prefixOrOptions, options);
    }

    named_buffers(prefixOrOptions: unknown = "", options = {}) {
      return this.namedBuffers(prefixOrOptions, options);
    }

    getBuffer(name: unknown) {
      lookupName(name, "nn.ParameterDict.getBuffer");
      return null;
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    parameterNames(prefix = "") {
      return parameterNames(this, prefix);
    }

    parameterInfos(prefix = "") {
      return parameterInfos(this, prefix);
    }

    parameterInfo(nameOrIndex: unknown, prefix = "") {
      return parameterInfo(this, nameOrIndex, prefix);
    }

    zeroGrad(options?: ZeroGradOptions) {
      zeroGrad(this, options);
    }

    zero_grad(options?: ZeroGradOptions) {
      return this.zeroGrad(options);
    }

    requiresGrad_(requiresGrad = true) {
      setRequiresGrad(this, requiresGrad);
      return this;
    }

    requires_grad_(requiresGrad = true) {
      return this.requiresGrad_(requiresGrad);
    }

    stateDict(prefixOrOptions: unknown = "") {
      return stateDictWithBuffers(this, prefixOrOptions);
    }

    state_dict(prefixOrOptions: unknown = "") {
      return this.stateDict(prefixOrOptions);
    }

    loadStateDict(source: unknown, options?: LoadStateDictOptions) {
      loadStateDictWithBuffers(this, source, options);
      return this;
    }

    load_state_dict(source: unknown, options?: LoadStateDictOptions) {
      return this.loadStateDict(source, options);
    }
  }

  class ModuleList {
    readonly kind = "moduleList";
    readonly layers: NnModule[];

    constructor(layers: Iterable<unknown> = []) {
      const values = iterableValues(layers, "nn.ModuleList");
      this.layers = values.map((layer, index) => {
        if (!isModuleLikeTarget(layer)) throw new Error(`nn.ModuleList item ${index} must be an nn module`);
        return layer as NnModule;
      });
    }

    get length() {
      return this.layers.length;
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

    at(index: number) {
      return this.layers.at(index);
    }

    get(index: number) {
      return this.at(index);
    }

    __getitem__(index: number) {
      return this.at(index);
    }

    __setitem__(index: unknown, moduleValue: unknown) {
      if (!isModuleLikeTarget(moduleValue)) throw new Error("nn.ModuleList.__setitem__ requires an nn module");
      this.layers[listSetIndex(index, this.layers.length, "nn.ModuleList.__setitem__")] = moduleValue as NnModule;
      return this;
    }

    __delitem__(index: unknown) {
      this.layers.splice(listSetIndex(index, this.layers.length, "nn.ModuleList.__delitem__"), 1);
      return this;
    }

    pop(index: unknown = -1) {
      const [value] = this.layers.splice(listSetIndex(index, this.layers.length, "nn.ModuleList.pop"), 1);
      return value;
    }

    clear() {
      this.layers.length = 0;
      return this;
    }

    push(moduleValue: unknown) {
      if (!isModuleLikeTarget(moduleValue)) throw new Error("nn.ModuleList.push requires an nn module");
      this.layers.push(moduleValue as NnModule);
      return this;
    }

    append(moduleValue: unknown) {
      return this.push(moduleValue);
    }

    insert(index: unknown, moduleValue: unknown) {
      if (!isModuleLikeTarget(moduleValue)) throw new Error("nn.ModuleList.insert requires an nn module");
      this.layers.splice(listInsertIndex(index, "nn.ModuleList.insert"), 0, moduleValue as NnModule);
      return this;
    }

    extend(moduleValues: Iterable<unknown>) {
      if (!moduleValues || typeof moduleValues[Symbol.iterator] !== "function") {
        throw new Error("nn.ModuleList.extend requires an iterable of nn modules");
      }
      for (const moduleValue of moduleValues) this.push(moduleValue);
      return this;
    }

    [Symbol.iterator]() {
      return this.layers[Symbol.iterator]();
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
      if (!recurse) return [];
      return this.layers.flatMap((layer, index) => (
        typeof layer.parameters === "function" ? layer.parameters(prefix ? `${prefix}.${index}` : String(index)) : []
      ));
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    named_parameters(prefixOrOptions: unknown = "", options = {}) {
      return this.namedParameters(prefixOrOptions, options);
    }

    getParameter(name: unknown) {
      return getParameterForTarget(this, name, "nn.ModuleList.getParameter");
    }

    get_parameter(name: unknown) {
      return this.getParameter(name);
    }

    buffers(prefixOrOptions: unknown = "", options = {}) {
      return defaultBuffersForTarget(this, prefixOrOptions, "nn.ModuleList", options);
    }

    namedBuffers(prefixOrOptions: unknown = "", options = {}) {
      return this.buffers(prefixOrOptions, options);
    }

    named_buffers(prefixOrOptions: unknown = "", options = {}) {
      return this.namedBuffers(prefixOrOptions, options);
    }

    getBuffer(name: unknown) {
      return getBufferForTarget(this, name, "nn.ModuleList.getBuffer");
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    parameterNames(prefix = "") {
      return parameterNames(this, prefix);
    }

    parameterInfos(prefix = "") {
      return parameterInfos(this, prefix);
    }

    parameterInfo(nameOrIndex: unknown, prefix = "") {
      return parameterInfo(this, nameOrIndex, prefix);
    }

    children() {
      return Object.freeze(this.layers.slice());
    }

    modules() {
      return modulesForTarget(this);
    }

    namedChildren(prefix = "") {
      return Object.freeze(this.layers.map((layer, index) => ({
        name: prefix ? `${prefix}.${index}` : String(index),
        module: layer,
      })));
    }

    named_children(prefix = "") {
      return this.namedChildren(prefix);
    }

    namedModules(prefix = "") {
      return namedModulesForTarget(this, prefix);
    }

    named_modules(prefix = "") {
      return this.namedModules(prefix);
    }

    getSubmodule(name: unknown) {
      return getSubmoduleForTarget(this, name, "nn.ModuleList.getSubmodule");
    }

    get_submodule(name: unknown) {
      return this.getSubmodule(name);
    }

    apply(callback: unknown) {
      return applyToModules(this, callback);
    }

    zeroGrad(options?: ZeroGradOptions) {
      zeroGrad(this, options);
    }

    zero_grad(options?: ZeroGradOptions) {
      return this.zeroGrad(options);
    }

    requiresGrad_(requiresGrad = true) {
      setRequiresGrad(this, requiresGrad);
      return this;
    }

    requires_grad_(requiresGrad = true) {
      return this.requiresGrad_(requiresGrad);
    }

    train(mode = true) {
      for (const layer of this.layers) {
        if (layer && typeof layer.train === "function") layer.train(mode);
      }
      return this;
    }

    eval() {
      return this.train(false);
    }

    stateDict(prefixOrOptions: unknown = "") {
      return stateDictWithBuffers(this, prefixOrOptions);
    }

    state_dict(prefixOrOptions: unknown = "") {
      return this.stateDict(prefixOrOptions);
    }

    loadStateDict(source: unknown, options?: LoadStateDictOptions) {
      loadStateDictWithBuffers(this, source, options);
      return this;
    }

    load_state_dict(source: unknown, options?: LoadStateDictOptions) {
      return this.loadStateDict(source, options);
    }
  }

  class ModuleDict {
    readonly kind = "moduleDict";
    readonly layers: Record<string, NnModule>;

    constructor(layers: Readonly<Record<string, unknown>> = {}) {
      if (!isOptionsRecord(layers)) throw new Error("nn.ModuleDict requires a record of nn modules");
      this.layers = {};
      for (const [key, layer] of Object.entries(layers)) {
        this.set(key, layer);
      }
    }

    get length() {
      return Object.keys(this.layers).length;
    }

    len() {
      return this.length;
    }

    __len__() {
      return this.len();
    }

    size() {
      return this.len();
    }

    get(name: unknown) {
      return this.layers[registrationName(name, "nn.ModuleDict.get")];
    }

    __getitem__(name: unknown) {
      return this.get(name);
    }

    __setitem__(name: unknown, moduleValue: unknown) {
      return this.set(name, moduleValue);
    }

    set(name: unknown, moduleValue: unknown) {
      const key = registrationName(name, "nn.ModuleDict.set");
      if (!isModuleLikeTarget(moduleValue)) throw new Error(`nn.ModuleDict.set requires an nn module for ${key}`);
      this.layers[key] = moduleValue as NnModule;
      return this;
    }

    update(layers: Readonly<Record<string, unknown>>) {
      if (!isOptionsRecord(layers)) throw new Error("nn.ModuleDict.update requires a record of nn modules");
      for (const [key, layer] of Object.entries(layers)) this.set(key, layer);
      return this;
    }

    has(name: unknown) {
      return Object.prototype.hasOwnProperty.call(this.layers, registrationName(name, "nn.ModuleDict.has"));
    }

    __contains__(name: unknown) {
      return this.has(name);
    }

    pop(name: unknown) {
      const key = registrationName(name, "nn.ModuleDict.pop");
      const value = this.layers[key];
      delete this.layers[key];
      return value;
    }

    __delitem__(name: unknown) {
      this.pop(name);
      return this;
    }

    clear() {
      for (const key of Object.keys(this.layers)) delete this.layers[key];
      return this;
    }

    keys() {
      return Object.freeze(Object.keys(this.layers));
    }

    values() {
      return Object.freeze(Object.values(this.layers));
    }

    entries() {
      return Object.freeze(Object.entries(this.layers));
    }

    items() {
      return this.entries();
    }

    [Symbol.iterator]() {
      return this.entries()[Symbol.iterator]();
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
      if (!recurse) return [];
      return Object.entries(this.layers).flatMap(([key, layer]) => (
        typeof layer.parameters === "function" ? layer.parameters(prefix ? `${prefix}.${key}` : key) : []
      ));
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    named_parameters(prefixOrOptions: unknown = "", options = {}) {
      return this.namedParameters(prefixOrOptions, options);
    }

    getParameter(name: unknown) {
      return getParameterForTarget(this, name, "nn.ModuleDict.getParameter");
    }

    get_parameter(name: unknown) {
      return this.getParameter(name);
    }

    buffers(prefixOrOptions: unknown = "", options = {}) {
      return defaultBuffersForTarget(this, prefixOrOptions, "nn.ModuleDict", options);
    }

    namedBuffers(prefixOrOptions: unknown = "", options = {}) {
      return this.buffers(prefixOrOptions, options);
    }

    named_buffers(prefixOrOptions: unknown = "", options = {}) {
      return this.namedBuffers(prefixOrOptions, options);
    }

    getBuffer(name: unknown) {
      return getBufferForTarget(this, name, "nn.ModuleDict.getBuffer");
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    parameterNames(prefix = "") {
      return parameterNames(this, prefix);
    }

    parameterInfos(prefix = "") {
      return parameterInfos(this, prefix);
    }

    parameterInfo(nameOrIndex: unknown, prefix = "") {
      return parameterInfo(this, nameOrIndex, prefix);
    }

    children() {
      return Object.freeze(Object.values(this.layers));
    }

    modules() {
      return modulesForTarget(this);
    }

    namedChildren(prefix = "") {
      return Object.freeze(Object.entries(this.layers).map(([key, module]) => ({
        name: prefix ? `${prefix}.${key}` : key,
        module,
      })));
    }

    named_children(prefix = "") {
      return this.namedChildren(prefix);
    }

    namedModules(prefix = "") {
      return namedModulesForTarget(this, prefix);
    }

    named_modules(prefix = "") {
      return this.namedModules(prefix);
    }

    getSubmodule(name: unknown) {
      return getSubmoduleForTarget(this, name, "nn.ModuleDict.getSubmodule");
    }

    get_submodule(name: unknown) {
      return this.getSubmodule(name);
    }

    apply(callback: unknown) {
      return applyToModules(this, callback);
    }

    zeroGrad(options?: ZeroGradOptions) {
      zeroGrad(this, options);
    }

    zero_grad(options?: ZeroGradOptions) {
      return this.zeroGrad(options);
    }

    requiresGrad_(requiresGrad = true) {
      setRequiresGrad(this, requiresGrad);
      return this;
    }

    requires_grad_(requiresGrad = true) {
      return this.requiresGrad_(requiresGrad);
    }

    train(mode = true) {
      for (const layer of Object.values(this.layers)) {
        if (layer && typeof layer.train === "function") layer.train(mode);
      }
      return this;
    }

    eval() {
      return this.train(false);
    }

    stateDict(prefixOrOptions: unknown = "") {
      return stateDictWithBuffers(this, prefixOrOptions);
    }

    state_dict(prefixOrOptions: unknown = "") {
      return this.stateDict(prefixOrOptions);
    }

    loadStateDict(source: unknown, options?: LoadStateDictOptions) {
      loadStateDictWithBuffers(this, source, options);
      return this;
    }

    load_state_dict(source: unknown, options?: LoadStateDictOptions) {
      return this.loadStateDict(source, options);
    }
  }

  function defaultChildrenForTarget(target: AnyRecord, label = "nn.Module") {
    return Object.freeze(defaultChildEntriesForTarget(target, label).map((entry) => entry.module as NnModule));
  }

  function defaultParametersForTarget(target: AnyRecord, prefixOrOptions: unknown = "", label = "nn.Module", options = {}) {
    const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
    const params = parameterEntriesFromOwnProperties(target).map((entry) => {
      if (!prefix) return prefixedParameter(entry.name, entry.parameter);
      return entry.parameter.name === entry.name
        ? prefixedParameter(prefix, entry.parameter)
        : prefixedParameter(`${prefix}.${entry.name}`, entry.parameter);
    });
    if (!recurse) return params;
    const entries = typeof target.namedChildren === "function" ? target.namedChildren(prefix) : namedChildrenForTarget(target, prefix, label);
    if (!Array.isArray(entries)) throw new Error(`${label} namedChildren must return an array`);
    params.push(...entries.flatMap((entry: AnyRecord) => {
      const child = entry && entry.module;
      if (!child || typeof child.parameters !== "function") return [];
      return child.parameters(String(entry.name));
    }));
    return params;
  }

  function modulesForTarget(target: AnyRecord, label = "nn.Module") {
    const out = [target as unknown as NnModule];
    const children = typeof target.children === "function" ? target.children() : defaultChildrenForTarget(target, label);
    if (!Array.isArray(children)) throw new Error(`${label} children must return an array`);
    for (const child of children as readonly AnyRecord[]) {
      if (child && typeof child.modules === "function") out.push(...child.modules());
      else out.push(child as unknown as NnModule);
    }
    return Object.freeze(out);
  }

  function namedChildrenForTarget(target: AnyRecord, prefix = "", label = "nn.Module") {
    const entries = defaultChildEntriesForTarget(target, label);
    return Object.freeze(entries.map((entry) => ({
      name: prefix ? `${prefix}.${entry.name}` : entry.name,
      module: entry.module,
    })));
  }

  function namedModulesForTarget(target: AnyRecord, prefix = "", label = "nn.Module") {
    const rootName = String(prefix);
    const out: AnyRecord[] = [{ name: rootName, module: target }];
    for (const entry of namedChildrenForTarget(target, rootName, label) as readonly AnyRecord[]) {
      const child = entry.module as AnyRecord | null | undefined;
      if (child && typeof child.namedModules === "function") out.push(...child.namedModules(entry.name));
      else out.push(entry);
    }
    return Object.freeze(out);
  }

  function applyToModules<T extends AnyRecord>(target: T, callback: unknown, label = "nn.apply") {
    if (typeof callback !== "function") throw new Error(`${label} requires a function`);
    const entries = typeof target.namedModules === "function" ? target.namedModules("") : namedModulesForTarget(target, "", label);
    if (!Array.isArray(entries)) throw new Error(`${label} requires namedModules to return an array`);
    for (const entry of entries as readonly AnyRecord[]) {
      const module = entry && entry.module !== undefined ? entry.module : entry;
      callback(module, { name: String(entry?.name ?? ""), module });
    }
    return target;
  }

  function lookupName(name: unknown, label: string) {
    if (typeof name !== "string") throw new Error(`${label} requires a string name`);
    return name;
  }

  function getParameterForTarget(target: NnModuleTarget, name: unknown, label = "nn.getParameter") {
    const targetName = lookupName(name, label);
    const entries = namedParametersForTarget(target, "") as readonly NnParameter[];
    return entries.find((param) => param.name === targetName) ?? null;
  }

  function getBufferForTarget(target: NnModuleTarget, name: unknown, label = "nn.getBuffer") {
    const targetName = lookupName(name, label);
    const entries = namedBuffersForTarget(target, "") as readonly NnBuffer[];
    return entries.find((targetBuffer) => targetBuffer.name === targetName) ?? null;
  }

  function getSubmoduleForTarget(target: NnModuleTarget, name: unknown, label = "nn.getSubmodule") {
    const targetName = lookupName(name, label);
    const entries = typeof (target as AnyRecord).namedModules === "function"
      ? (target as AnyRecord).namedModules("")
      : namedModulesForTarget(target as AnyRecord, "", label);
    if (!Array.isArray(entries)) throw new Error(`${label} requires namedModules to return an array`);
    const entry = (entries as readonly AnyRecord[]).find((candidate) => candidate && candidate.name === targetName);
    return (entry && entry.module !== undefined ? entry.module : null) as NnModule | null;
  }

  function customModule(config: unknown): NnModuleTarget {
    const spec = config as UnknownRecord & {
      kind?: unknown;
      forward?: unknown;
      parameters?: unknown;
      children?: unknown;
      graph?: unknown;
    };
    if (!spec || typeof spec !== "object" || typeof spec.forward !== "function") {
      throw new Error("nn.module requires { forward(input), ... }");
    }
    const moduleKind = typeof spec.kind === "string" && spec.kind.length > 0 ? spec.kind : "module";
    const forward = spec.forward;
    const parameterSource = spec.parameters;
    const childSource = spec.children;
    const graphSource = spec.graph;
    function resolveChildren(target: AnyRecord) {
      const children = typeof childSource === "function"
        ? childSource.call(target)
        : Array.isArray(childSource)
          ? childSource
          : [];
      if (!Array.isArray(children)) throw new Error("nn.module children must return an array");
      return Object.freeze(children.slice());
    }
    function resolveGraph(target: AnyRecord) {
      const graph = typeof graphSource === "function" ? graphSource.call(target) : graphSource;
      if (!graph || typeof graph !== "object") {
        throw new Error("nn.module compile requires a declared graph module");
      }
      return graph as AnyRecord;
    }
    function graphMethod(target: AnyRecord, name: string) {
      const graph = resolveGraph(target);
      const method = graph[name];
      if (typeof method !== "function") throw new Error(`nn.module graph must support ${name}()`);
      return method.bind(graph);
    }
    const target = {
      kind: moduleKind,
      customModule: true,
      training: true,
      forward(input: unknown) {
        return forward.call(target, input);
      },
      call(input: unknown) {
        return target.forward(input);
      },
      __call__(input: unknown) {
        return target.forward(input);
      },
      parameters(prefixOrOptions: unknown = "", options = {}) {
        const { prefix, recurse } = moduleTraversalOptions(prefixOrOptions, options);
        const fromChildren = parameterSource === undefined;
        const params = typeof parameterSource === "function"
          ? parameterSource.call(target, prefix)
          : Array.isArray(parameterSource)
            ? parameterSource
            : recurse
              ? (target.children() as readonly AnyRecord[]).flatMap((child, index) => {
              if (!child || typeof child.parameters !== "function") return [];
              return child.parameters(prefix ? `${prefix}.${index}` : String(index));
            })
              : [];
        if (!Array.isArray(params)) throw new Error("nn.module parameters must return an array");
        return prefix && !fromChildren
          ? params.map((param: AnyRecord) => prefixedParameter(prefix, param))
          : params;
      },
      namedParameters(prefixOrOptions: unknown = "", options = {}) {
        return target.parameters(prefixOrOptions, options);
      },
      named_parameters(prefixOrOptions: unknown = "", options = {}) {
        return target.namedParameters(prefixOrOptions, options);
      },
      getParameter(name: unknown) {
        return getParameterForTarget(target, name, "nn.module.getParameter");
      },
      get_parameter(name: unknown) {
        return target.getParameter(name);
      },
      buffers(prefixOrOptions: unknown = "", options = {}) {
        return defaultBuffersForTarget(target, prefixOrOptions, "nn.module", options);
      },
      namedBuffers(prefixOrOptions: unknown = "", options = {}) {
        return target.buffers(prefixOrOptions, options);
      },
      named_buffers(prefixOrOptions: unknown = "", options = {}) {
        return target.namedBuffers(prefixOrOptions, options);
      },
      getBuffer(name: unknown) {
        return getBufferForTarget(target, name, "nn.module.getBuffer");
      },
      get_buffer(name: unknown) {
        return target.getBuffer(name);
      },
      parameterNames(prefix = "") {
        return parameterNames(target, prefix);
      },
      parameterInfos(prefix = "") {
        return parameterInfos(target, prefix);
      },
      parameterInfo(nameOrIndex: unknown, prefix = "") {
        return parameterInfo(target, nameOrIndex, prefix);
      },
      children() {
        return resolveChildren(target);
      },
      modules() {
        const out = [target as unknown as NnModule];
        for (const child of target.children() as readonly AnyRecord[]) {
          if (child && typeof child.modules === "function") out.push(...child.modules());
          else out.push(child as unknown as NnModule);
        }
        return Object.freeze(out);
      },
      namedChildren(prefix = "") {
        return Object.freeze((target.children() as readonly unknown[]).map((child, index) => ({
          name: prefix ? `${prefix}.${index}` : String(index),
          module: child,
        })));
      },
      named_children(prefix = "") {
        return target.namedChildren(prefix);
      },
      namedModules(prefix = "") {
        const rootName = String(prefix);
        const out: AnyRecord[] = [{ name: rootName, module: target }];
        for (const entry of target.namedChildren(rootName) as readonly AnyRecord[]) {
          const child = entry.module as AnyRecord | null | undefined;
          if (child && typeof child.namedModules === "function") out.push(...child.namedModules(entry.name));
          else out.push(entry);
        }
        return Object.freeze(out);
      },
      named_modules(prefix = "") {
        return target.namedModules(prefix);
      },
      getSubmodule(name: unknown) {
        return getSubmoduleForTarget(target, name, "nn.module.getSubmodule");
      },
      get_submodule(name: unknown) {
        return target.getSubmodule(name);
      },
      apply(callback: unknown) {
        return applyToModules(target, callback);
      },
      zeroGrad(options?: ZeroGradOptions) {
        zeroGrad(target, options);
      },
      zero_grad(options?: ZeroGradOptions) {
        return target.zeroGrad(options);
      },
      requiresGrad_(requiresGrad = true) {
        setRequiresGrad(target, requiresGrad);
        return target;
      },
      requires_grad_(requiresGrad = true) {
        return target.requiresGrad_(requiresGrad);
      },
      train(mode = true) {
        target.training = Boolean(mode);
        for (const child of target.children() as readonly AnyRecord[]) {
          if (child && typeof child.train === "function") child.train(mode);
        }
        return target;
      },
      eval() {
        return target.train(false);
      },
      to(placement?: TensorToTarget | TensorToOptions, toOptions: TensorToOptions = {}) {
        return moduleTo(target, placement, toOptions);
      },
      cpu(toOptions: TensorToOptions = {}) {
        return target.to("cpu", toOptions);
      },
      float(toOptions: TensorToOptions = {}) {
        return target.to("float32", toOptions);
      },
      float32(toOptions: TensorToOptions = {}) {
        return target.to("float32", toOptions);
      },
      stateDict(prefixOrOptions: unknown = "") {
        return stateDictWithBuffers(target, prefixOrOptions);
      },
      state_dict(prefixOrOptions: unknown = "") {
        return target.stateDict(prefixOrOptions);
      },
      loadStateDict(source: unknown, loadOptions?: LoadStateDictOptions) {
        loadStateDictWithBuffers(target, source, loadOptions);
        return target;
      },
      load_state_dict(source: unknown, loadOptions?: LoadStateDictOptions) {
        return target.loadStateDict(source, loadOptions);
      },
      trace(traceOptions: ModuleTraceOptions = {}) {
        if (graphSource !== undefined) return graphMethod(target, "trace")(traceOptions);
        return traceSequentialProgram([target], traceOptions);
      },
      compileSupport(compileOptions: CompileOptions = {}) {
        return graphMethod(target, "compileSupport")(compileOptions);
      },
      compile_support(compileOptions: CompileOptions = {}) {
        return target.compileSupport(compileOptions);
      },
      requireCompileSupport(compileOptions: CompileOptions = {}) {
        const graph = resolveGraph(target);
        const method = graph.requireCompileSupport;
        return typeof method === "function"
          ? method.call(graph, compileOptions)
          : requireCompileSupportForModule(graph, compileOptions);
      },
      require_compile_support(compileOptions: CompileOptions = {}) {
        return target.requireCompileSupport(compileOptions);
      },
      compilerSignatures(compileOptions: CompileOptions = {}) {
        return compilerSignaturesForModule(resolveGraph(target), compileOptions);
      },
      compiler_signatures(compileOptions: CompileOptions = {}) {
        return target.compilerSignatures(compileOptions);
      },
      tensorProgramIr(compileOptions: CompileOptions = {}) {
        return tensorProgramIrForModule(resolveGraph(target), compileOptions);
      },
      tensor_program_ir(compileOptions: CompileOptions = {}) {
        return target.tensorProgramIr(compileOptions);
      },
      kernelPlan(compileOptions: CompileOptions = {}) {
        return kernelPlanForModule(resolveGraph(target), compileOptions);
      },
      kernel_plan(compileOptions: CompileOptions = {}) {
        return target.kernelPlan(compileOptions);
      },
      bufferLayout(compileOptions: CompileOptions = {}) {
        return bufferLayoutForModule(resolveGraph(target), compileOptions);
      },
      buffer_layout(compileOptions: CompileOptions = {}) {
        return target.bufferLayout(compileOptions);
      },
      memoryLayout(compileOptions: CompileOptions = {}) {
        return memoryLayoutForModule(resolveGraph(target), compileOptions);
      },
      memory_layout(compileOptions: CompileOptions = {}) {
        return target.memoryLayout(compileOptions);
      },
      inputShape(compileOptions: CompileOptions = {}) {
        return inputShapeForModule(resolveGraph(target), compileOptions);
      },
      input_shape(compileOptions: CompileOptions = {}) {
        return target.inputShape(compileOptions);
      },
      outputShape(compileOptions: CompileOptions = {}) {
        return outputShapeForModule(resolveGraph(target), compileOptions);
      },
      output_shape(compileOptions: CompileOptions = {}) {
        return target.outputShape(compileOptions);
      },
      shapeConstraints(compileOptions: CompileOptions = {}) {
        return shapeConstraintsForModule(resolveGraph(target), compileOptions);
      },
      shape_constraints(compileOptions: CompileOptions = {}) {
        return target.shapeConstraints(compileOptions);
      },
      parameterLayout(compileOptions: CompileOptions = {}) {
        return parameterLayoutForModule(resolveGraph(target), compileOptions);
      },
      parameter_layout(compileOptions: CompileOptions = {}) {
        return target.parameterLayout(compileOptions);
      },
      explain(compileOptions: CompileOptions = {}) {
        return explainModule(resolveGraph(target), compileOptions);
      },
      preflight(compileOptions: CompileOptions = {}) {
        return target.explain(compileOptions);
      },
      compileExplanation(compileOptions: CompileOptions = {}) {
        return target.explain(compileOptions);
      },
      compile_explanation(compileOptions: CompileOptions = {}) {
        return target.explain(compileOptions);
      },
      compilePlan(compileOptions: CompileOptions = {}) {
        return target.explain(compileOptions);
      },
      compile_plan(compileOptions: CompileOptions = {}) {
        return target.explain(compileOptions);
      },
      requireCompilePlan(compileOptions: CompileOptions = {}) {
        return requireCompilePlanForModule(resolveGraph(target), compileOptions);
      },
      require_compile_plan(compileOptions: CompileOptions = {}) {
        return target.requireCompilePlan(compileOptions);
      },
      assertCompilePlan(compileOptions: CompileOptions = {}) {
        return target.requireCompilePlan(compileOptions);
      },
      assert_compile_plan(compileOptions: CompileOptions = {}) {
        return target.requireCompilePlan(compileOptions);
      },
      canCompile(compileOptions: CompileOptions = {}) {
        return canCompileModule(resolveGraph(target), compileOptions);
      },
      can_compile(compileOptions: CompileOptions = {}) {
        return target.canCompile(compileOptions);
      },
      compile(compileOptions: CompileOptions = {}) {
        return graphMethod(target, "compile")(compileOptions);
      },
      bindParameters(bindOptions: CompileOptions = {}) {
        return graphMethod(target, "bindParameters")(bindOptions);
      },
      bind_parameters(bindOptions: CompileOptions = {}) {
        return target.bindParameters(bindOptions);
      },
      placeParameters(program: unknown, placementOptions: unknown = {}) {
        return graphMethod(target, "placeParameters")(program, placementOptions);
      },
      place_parameters(program: unknown, placementOptions: unknown = {}) {
        return target.placeParameters(program, placementOptions);
      },
    };
    return target as NnModuleTarget;
  }

  class ModuleBase {
    kind: string;
    readonly customModule = true;
    training = true;
    graph?: unknown;

    constructor(config: UnknownRecord = {}) {
      this.kind = typeof config.kind === "string" && config.kind.length > 0 ? config.kind : "module";
      if (config.graph !== undefined) this.graph = config.graph;
    }

    forward(_input: unknown): unknown {
      throw new Error("nn.Module subclasses must implement forward(input)");
    }

    call(input: unknown) {
      return this.forward(input);
    }

    __call__(input: unknown) {
      return this.forward(input);
    }

    addModule(name: unknown, module: unknown) {
      const key = registrationName(name, "nn.Module.addModule");
      if (!isModuleChild(module, this)) throw new Error(`nn.Module.addModule expected an nn module for ${key}`);
      (this as AnyRecord)[key] = module;
      return this;
    }

    add_module(name: unknown, module: unknown) {
      return this.addModule(name, module);
    }

    registerModule(name: unknown, module: unknown) {
      return this.addModule(name, module);
    }

    register_module(name: unknown, module: unknown) {
      return this.addModule(name, module);
    }

    registerParameter(name: unknown, parameter: unknown) {
      const key = registrationName(name, "nn.Module.registerParameter");
      if (!isParameterLike(parameter)) throw new Error(`nn.Module.registerParameter expected an nn.Parameter for ${key}`);
      (this as AnyRecord)[key] = parameter;
      return this;
    }

    register_parameter(name: unknown, parameter: unknown) {
      return this.registerParameter(name, parameter);
    }

    registerBuffer(name: unknown, bufferValue: unknown) {
      const key = registrationName(name, "nn.Module.registerBuffer");
      const value = isBufferLike(bufferValue)
        ? bufferValue
        : buffer(key, bufferValue, bufferShape(bufferValue) ?? parameterShape(bufferValue) ?? undefined);
      (this as AnyRecord)[key] = value;
      return this;
    }

    register_buffer(name: unknown, bufferValue: unknown) {
      return this.registerBuffer(name, bufferValue);
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      return defaultParametersForTarget(this, prefixOrOptions, "nn.Module", options);
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    named_parameters(prefixOrOptions: unknown = "", options = {}) {
      return this.namedParameters(prefixOrOptions, options);
    }

    getParameter(name: unknown) {
      return getParameterForTarget(this, name, "nn.Module.getParameter");
    }

    get_parameter(name: unknown) {
      return this.getParameter(name);
    }

    buffers(prefixOrOptions: unknown = "", options = {}) {
      return defaultBuffersForTarget(this, prefixOrOptions, "nn.Module", options);
    }

    namedBuffers(prefixOrOptions: unknown = "", options = {}) {
      return this.buffers(prefixOrOptions, options);
    }

    named_buffers(prefixOrOptions: unknown = "", options = {}) {
      return this.namedBuffers(prefixOrOptions, options);
    }

    getBuffer(name: unknown) {
      return getBufferForTarget(this, name, "nn.Module.getBuffer");
    }

    get_buffer(name: unknown) {
      return this.getBuffer(name);
    }

    parameterNames(prefix = "") {
      return parameterNames(this, prefix);
    }

    parameterInfos(prefix = "") {
      return parameterInfos(this, prefix);
    }

    parameterInfo(nameOrIndex: unknown, prefix = "") {
      return parameterInfo(this, nameOrIndex, prefix);
    }

    children() {
      return defaultChildrenForTarget(this);
    }

    modules() {
      return modulesForTarget(this);
    }

    namedChildren(prefix = "") {
      return namedChildrenForTarget(this, prefix);
    }

    named_children(prefix = "") {
      return this.namedChildren(prefix);
    }

    namedModules(prefix = "") {
      return namedModulesForTarget(this, prefix);
    }

    named_modules(prefix = "") {
      return this.namedModules(prefix);
    }

    getSubmodule(name: unknown) {
      return getSubmoduleForTarget(this, name, "nn.Module.getSubmodule");
    }

    get_submodule(name: unknown) {
      return this.getSubmodule(name);
    }

    apply(callback: unknown) {
      return applyToModules(this, callback);
    }

    zeroGrad(options?: ZeroGradOptions) {
      zeroGrad(this, options);
    }

    zero_grad(options?: ZeroGradOptions) {
      return this.zeroGrad(options);
    }

    requiresGrad_(requiresGrad = true) {
      setRequiresGrad(this, requiresGrad);
      return this;
    }

    requires_grad_(requiresGrad = true) {
      return this.requiresGrad_(requiresGrad);
    }

    train(mode = true) {
      this.training = Boolean(mode);
      for (const child of this.children() as readonly AnyRecord[]) {
        if (child && typeof child.train === "function") child.train(mode);
      }
      return this;
    }

    eval() {
      return this.train(false);
    }

    to(placement?: TensorToTarget | TensorToOptions, toOptions: TensorToOptions = {}) {
      return moduleTo(this, placement, toOptions);
    }

    cpu(toOptions: TensorToOptions = {}) {
      return this.to("cpu", toOptions);
    }

    float(toOptions: TensorToOptions = {}) {
      return this.to("float32", toOptions);
    }

    float32(toOptions: TensorToOptions = {}) {
      return this.to("float32", toOptions);
    }

    stateDict(prefixOrOptions: unknown = "") {
      return stateDictWithBuffers(this, prefixOrOptions);
    }

    state_dict(prefixOrOptions: unknown = "") {
      return this.stateDict(prefixOrOptions);
    }

    loadStateDict(source: unknown, loadOptions?: LoadStateDictOptions) {
      loadStateDictWithBuffers(this, source, loadOptions);
      return this;
    }

    load_state_dict(source: unknown, loadOptions?: LoadStateDictOptions) {
      return this.loadStateDict(source, loadOptions);
    }

    trace(traceOptions: ModuleTraceOptions = {}) {
      return graphMethodForTarget(this, "trace")(traceOptions);
    }

    compileSupport(compileOptions: CompileOptions = {}) {
      return graphMethodForTarget(this, "compileSupport")(compileOptions);
    }

    compile_support(compileOptions: CompileOptions = {}) {
      return this.compileSupport(compileOptions);
    }

    requireCompileSupport(compileOptions: CompileOptions = {}) {
      const graph = graphModuleForTarget(this);
      const method = graph.requireCompileSupport;
      return typeof method === "function"
        ? method.call(graph, compileOptions)
        : requireCompileSupportForModule(graph, compileOptions);
    }

    require_compile_support(compileOptions: CompileOptions = {}) {
      return this.requireCompileSupport(compileOptions);
    }

    compilerSignatures(compileOptions: CompileOptions = {}) {
      return compilerSignaturesForModule(graphModuleForTarget(this), compileOptions);
    }

    compiler_signatures(compileOptions: CompileOptions = {}) {
      return this.compilerSignatures(compileOptions);
    }

    tensorProgramIr(compileOptions: CompileOptions = {}) {
      return tensorProgramIrForModule(graphModuleForTarget(this), compileOptions);
    }

    tensor_program_ir(compileOptions: CompileOptions = {}) {
      return this.tensorProgramIr(compileOptions);
    }

    kernelPlan(compileOptions: CompileOptions = {}) {
      return kernelPlanForModule(graphModuleForTarget(this), compileOptions);
    }

    kernel_plan(compileOptions: CompileOptions = {}) {
      return this.kernelPlan(compileOptions);
    }

    bufferLayout(compileOptions: CompileOptions = {}) {
      return bufferLayoutForModule(graphModuleForTarget(this), compileOptions);
    }

    buffer_layout(compileOptions: CompileOptions = {}) {
      return this.bufferLayout(compileOptions);
    }

    memoryLayout(compileOptions: CompileOptions = {}) {
      return memoryLayoutForModule(graphModuleForTarget(this), compileOptions);
    }

    memory_layout(compileOptions: CompileOptions = {}) {
      return this.memoryLayout(compileOptions);
    }

    inputShape(compileOptions: CompileOptions = {}) {
      return inputShapeForModule(graphModuleForTarget(this), compileOptions);
    }

    input_shape(compileOptions: CompileOptions = {}) {
      return this.inputShape(compileOptions);
    }

    outputShape(compileOptions: CompileOptions = {}) {
      return outputShapeForModule(graphModuleForTarget(this), compileOptions);
    }

    output_shape(compileOptions: CompileOptions = {}) {
      return this.outputShape(compileOptions);
    }

    shapeConstraints(compileOptions: CompileOptions = {}) {
      return shapeConstraintsForModule(graphModuleForTarget(this), compileOptions);
    }

    shape_constraints(compileOptions: CompileOptions = {}) {
      return this.shapeConstraints(compileOptions);
    }

    parameterLayout(compileOptions: CompileOptions = {}) {
      return parameterLayoutForModule(graphModuleForTarget(this), compileOptions);
    }

    parameter_layout(compileOptions: CompileOptions = {}) {
      return this.parameterLayout(compileOptions);
    }

    explain(compileOptions: CompileOptions = {}) {
      return explainModule(graphModuleForTarget(this), compileOptions);
    }

    preflight(compileOptions: CompileOptions = {}) {
      return this.explain(compileOptions);
    }

    compileExplanation(compileOptions: CompileOptions = {}) {
      return this.explain(compileOptions);
    }

    compile_explanation(compileOptions: CompileOptions = {}) {
      return this.explain(compileOptions);
    }

    compilePlan(compileOptions: CompileOptions = {}) {
      return this.explain(compileOptions);
    }

    compile_plan(compileOptions: CompileOptions = {}) {
      return this.explain(compileOptions);
    }

    requireCompilePlan(compileOptions: CompileOptions = {}) {
      return requireCompilePlanForModule(graphModuleForTarget(this), compileOptions);
    }

    require_compile_plan(compileOptions: CompileOptions = {}) {
      return this.requireCompilePlan(compileOptions);
    }

    assertCompilePlan(compileOptions: CompileOptions = {}) {
      return this.requireCompilePlan(compileOptions);
    }

    assert_compile_plan(compileOptions: CompileOptions = {}) {
      return this.requireCompilePlan(compileOptions);
    }

    canCompile(compileOptions: CompileOptions = {}) {
      return canCompileModule(graphModuleForTarget(this), compileOptions);
    }

    can_compile(compileOptions: CompileOptions = {}) {
      return this.canCompile(compileOptions);
    }

    compile(compileOptions: CompileOptions = {}) {
      return graphMethodForTarget(this, "compile")(compileOptions);
    }

    bindParameters(bindOptions: CompileOptions = {}) {
      return graphMethodForTarget(this, "bindParameters")(bindOptions);
    }

    bind_parameters(bindOptions: CompileOptions = {}) {
      return this.bindParameters(bindOptions);
    }

    placeParameters(program: unknown, placementOptions: unknown = {}) {
      return graphMethodForTarget(this, "placeParameters")(program, placementOptions);
    }

    place_parameters(program: unknown, placementOptions: unknown = {}) {
      return this.placeParameters(program, placementOptions);
    }
  }

  const moduleConstructors: readonly NnPrototypeConstructor[] = [
    LinearModule,
    EmbeddingModule,
    Conv2dModule,
    AvgPool2dModule,
    MaxPool2dModule as unknown as NnPrototypeConstructor,
    SequentialModule,
    ActivationModule,
    SoftmaxModule,
    LogSoftmaxModule,
    ReductionModule,
    DropoutModule,
    ShapeModule,
    FeatureNormModule,
  ];
  installNnModulePrototypeMethods(moduleConstructors, {
    traceSequentialProgram,
    compileSupportForModule,
    requireCompileSupportForModule,
    compilerSignaturesForModule,
    tensorProgramIrForModule,
    kernelPlanForModule,
    bufferLayoutForModule,
    memoryLayoutForModule,
    inputShapeForModule,
    outputShapeForModule,
    shapeConstraintsForModule,
    parameterLayoutForModule,
    explainModule,
    requireCompilePlanForModule,
    canCompileModule,
    nativeInferenceForModule: moduleNativeInference,
  });

  const initNamespace = Object.freeze({
    constant_: <T>(target: T, value: unknown) => constantInit(target, value),
    zeros_: <T>(target: T) => constantInit(target, 0, "nn.init.zeros_"),
    ones_: <T>(target: T) => constantInit(target, 1, "nn.init.ones_"),
    uniform_: <T>(target: T, min = 0, max = 1) => uniformInit(target, min, max),
    normal_: <T>(target: T, mean = 0, std = 1) => normalInit(target, mean, std),
    xavierUniform_: <T>(target: T, gain = 1) => xavierUniformInit(target, gain),
    xavier_uniform_: <T>(target: T, gain = 1) => xavierUniformInit(target, gain),
    xavierNormal_: <T>(target: T, gain = 1) => xavierNormalInit(target, gain),
    xavier_normal_: <T>(target: T, gain = 1) => xavierNormalInit(target, gain),
    kaimingUniform_: <T>(target: T, options: UnknownRecord = {}) => kaimingUniformInit(target, options),
    kaiming_uniform_: <T>(target: T, options: UnknownRecord = {}) => kaimingUniformInit(target, options),
    kaimingNormal_: <T>(target: T, options: UnknownRecord = {}) => kaimingNormalInit(target, options),
    kaiming_normal_: <T>(target: T, options: UnknownRecord = {}) => kaimingNormalInit(target, options),
  });

  return Object.freeze({
    Module: ModuleBase,
    ModuleList,
    ModuleDict,
    ParameterList,
    ParameterDict,
    Linear: LinearModule,
    Embedding: EmbeddingModule,
    Conv2d: Conv2dModule,
    AvgPool2d: AvgPool2dModule,
    MaxPool2d: MaxPool2dModule,
    Sequential: SequentialModule,
    Softmax: SoftmaxModule,
    LogSoftmax: LogSoftmaxModule,
    Reduction: ReductionModule,
    Dropout: DropoutModule,
    Shape: ShapeModule,
    LayerNorm: LayerNormModule,
    RMSNorm: RMSNormModule,
    BatchNorm1d: BatchNorm1dModule,
    Identity: IdentityModule,
    Diagonal: DiagonalModule,
    Repeat: RepeatModule,
    Tile: TileModule,
    Flatten: FlattenModule,
    ReLU: ReLUModule,
    GELU: GELUModule,
    SiLU: SiLUModule,
    Sigmoid: SigmoidModule,
    Tanh: TanhModule,
    MSELoss,
    L1Loss,
    HuberLoss,
    SmoothL1Loss,
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    NLLLoss,
    linear: (inFeatures: number, outFeatures: number, config: NnLinearConfig = {}) => new LinearModule(inFeatures, outFeatures, config),
    conv2d: (inChannels: number, outChannels: number, kernelSize: unknown, config: UnknownRecord = {}) => new Conv2dModule(inChannels, outChannels, kernelSize, config),
    avgPool2d: (kernelSize: unknown, config: UnknownRecord = {}) => new AvgPool2dModule(kernelSize, config),
    avg_pool2d: (kernelSize: unknown, config: UnknownRecord = {}) => new AvgPool2dModule(kernelSize, config),
    maxPool2d: (kernelSize: unknown, config: UnknownRecord = {}) => new MaxPool2dModule(kernelSize, config),
    max_pool2d: (kernelSize: unknown, config: UnknownRecord = {}) => new MaxPool2dModule(kernelSize, config),
    Buffer: buffer,
    buffer,
    Parameter: parameter,
    parameter,
    init: initNamespace,
    moduleList: (layers: Iterable<unknown> = []) => new ModuleList(layers),
    module_list: (layers: Iterable<unknown> = []) => new ModuleList(layers),
    moduleDict: (layers: Readonly<Record<string, unknown>> = {}) => new ModuleDict(layers),
    module_dict: (layers: Readonly<Record<string, unknown>> = {}) => new ModuleDict(layers),
    parameterList: (params: Iterable<unknown> = []) => new ParameterList(params),
    parameter_list: (params: Iterable<unknown> = []) => new ParameterList(params),
    parameterDict: (params: Readonly<Record<string, unknown>> = {}) => new ParameterDict(params),
    parameter_dict: (params: Readonly<Record<string, unknown>> = {}) => new ParameterDict(params),
    module: customModule,
    embedding: (numEmbeddings: number, embeddingDim: number, config: NnEmbeddingConfig = {}) => new EmbeddingModule(numEmbeddings, embeddingDim, config),
    sequential: (first?: unknown, ...rest: unknown[]) => rest.length === 0
      ? new SequentialModule(first as readonly NnModule[] | Readonly<Record<string, NnModule>> | NnModule | undefined)
      : new SequentialModule(first as NnModule, ...rest as readonly NnModule[]),
    gelu: () => new GELUModule(),
    relu: () => new ReLUModule(),
    silu: () => new SiLUModule(),
    sigmoid: () => new SigmoidModule(),
    tanh: () => new TanhModule(),
    exp: () => new ActivationModule("exp", Math.exp),
    log: () => new ActivationModule("log", Math.log),
    neg: () => new ActivationModule("neg", (x: number) => -x),
    recip: () => new ActivationModule("recip", (x: number) => 1 / x),
    abs: () => new ActivationModule("abs", Math.abs),
    sgn: () => new ActivationModule("sgn", (x: number) => x < 0 ? -1 : x > 0 ? 1 : 0),
    sign: () => new ActivationModule("sgn", (x: number) => x < 0 ? -1 : x > 0 ? 1 : 0),
    step: () => new ActivationModule("step", (x: number) => x > 0 ? 1 : 0),
    sqrt: () => new ActivationModule("sqrt", Math.sqrt),
    square: () => new ActivationModule("square", (x: number) => x * x),
    sqr: () => new ActivationModule("square", (x: number) => x * x),
    softmax: (dim = -1) => new SoftmaxModule(dim),
    logSoftmax: (dim = -1) => new LogSoftmaxModule(dim),
    log_softmax: (dim = -1) => new LogSoftmaxModule(dim),
    sum: (dim = -1) => new ReductionModule("sum", dim),
    mean: (dim = -1) => new ReductionModule("mean", dim),
    prod: (dim = -1) => new ReductionModule("prod", dim),
    max: (dim = -1) => new ReductionModule("max", dim),
    min: (dim = -1) => new ReductionModule("min", dim),
    argmax: (dim = -1) => new ReductionModule("argmax", dim),
    argmin: (dim = -1) => new ReductionModule("argmin", dim),
    dropout: (p = 0.5, config: NnDropoutConfig = {}) => new DropoutModule(p, config),
    identity: () => new IdentityModule(),
    diagonal: () => new DiagonalModule(),
    repeat: (repeats: unknown) => new RepeatModule(repeats),
    tile: (repeats: unknown) => new TileModule(repeats),
    reshape: (shape: unknown) => new ShapeModule("reshape", shape),
    view: (shape: unknown) => new ShapeModule("view", shape),
    flatten: (startDim = 0, endDim = -1) => new FlattenModule(startDim, endDim),
    squeeze: (dim: unknown = null) => new ShapeModule("squeeze", dim),
    unsqueeze: (dim: unknown) => new ShapeModule("unsqueeze", dim),
    transpose: (dim0 = 0, dim1 = 1) => new ShapeModule("transpose", dim0, dim1),
    permute: (dims: unknown) => new ShapeModule("permute", dims),
    broadcastTo: (shape: unknown) => new ShapeModule("broadcastTo", shape),
    expand: (shape: unknown) => new ShapeModule("expand", shape),
    narrow: (dim: unknown, start: unknown, length: unknown) => new ShapeModule("narrow", dim, start, length),
    select: (dim: unknown, index: unknown) => new ShapeModule("select", dim, index),
    slice: (dim: unknown, start = 0, end: unknown = null, step = 1) => new ShapeModule("slice", dim, start, end, step),
    layerNorm: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("layerNorm", features, config),
    layer_norm: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("layerNorm", features, config),
    rmsNorm: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("rmsNorm", features, config),
    rms_norm: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("rmsNorm", features, config),
    batchNorm1d: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("batchNorm1d", features, config),
    batch_norm1d: (features: number, config: NnNormConfig = {}) => new FeatureNormModule("batchNorm1d", features, config),
    parameters: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedParametersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    namedParameters: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedParametersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    named_parameters: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedParametersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    getParameter: (target: unknown, name: unknown) => getParameterForTarget(moduleTarget(stateTarget(target)), name),
    get_parameter: (target: unknown, name: unknown) => getParameterForTarget(moduleTarget(stateTarget(target)), name),
    buffers: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedBuffersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    namedBuffers: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedBuffersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    named_buffers: (target: unknown, prefixOrOptions: unknown = "", options = {}) => namedBuffersForTarget(moduleTarget(stateTarget(target)), prefixOrOptions, options),
    getBuffer: (target: unknown, name: unknown) => getBufferForTarget(moduleTarget(stateTarget(target)), name),
    get_buffer: (target: unknown, name: unknown) => getBufferForTarget(moduleTarget(stateTarget(target)), name),
    parameterNames: (target: unknown, prefix = "") => parameterNamesForTarget(moduleTarget(stateTarget(target)), prefix),
    parameterInfos: (target: unknown, prefix = "") => parameterInfosForTarget(moduleTarget(stateTarget(target)), prefix),
    parameterInfo: (target: unknown, nameOrIndex: unknown, prefix = "") => parameterInfoForTarget(moduleTarget(stateTarget(target)), nameOrIndex, prefix),
    call: (target: unknown, input: unknown) => moduleMethod(moduleTarget(target), "forward")(input),
    __call__: (target: unknown, input: unknown) => moduleMethod(moduleTarget(target), "forward")(input),
    forward: (target: unknown, input: unknown) => moduleMethod(moduleTarget(target), "forward")(input),
    children: (target: unknown) => moduleMethod(moduleTarget(target), "children")(),
    modules: (target: unknown) => moduleMethod(moduleTarget(target), "modules")(),
    namedChildren: (target: unknown, prefix = "") => moduleMethod(moduleTarget(target), "namedChildren")(prefix),
    named_children: (target: unknown, prefix = "") => moduleMethod(moduleTarget(target), "namedChildren")(prefix),
    namedModules: (target: unknown, prefix = "") => moduleMethod(moduleTarget(target), "namedModules")(prefix),
    named_modules: (target: unknown, prefix = "") => moduleMethod(moduleTarget(target), "namedModules")(prefix),
    getSubmodule: (target: unknown, name: unknown) => getSubmoduleForTarget(moduleTarget(target), name),
    get_submodule: (target: unknown, name: unknown) => getSubmoduleForTarget(moduleTarget(target), name),
    apply: (target: unknown, callback: unknown) => moduleMethod(moduleTarget(target), "apply")(callback),
    train: (target: unknown, mode = true) => moduleMethod(moduleTarget(target), "train")(mode),
    eval: (target: unknown) => moduleMethod(moduleTarget(target), "eval")(),
    to: (target: unknown, placement?: TensorToTarget | TensorToOptions, toOptions: TensorToOptions = {}) => (
      moduleTo(moduleTarget(target), placement, toOptions)
    ),
    cpu: (target: unknown, toOptions: TensorToOptions = {}) => moduleTo(moduleTarget(target), "cpu", toOptions),
    float: (target: unknown, toOptions: TensorToOptions = {}) => moduleTo(moduleTarget(target), "float32", toOptions),
    float32: (target: unknown, toOptions: TensorToOptions = {}) => moduleTo(moduleTarget(target), "float32", toOptions),
    zeroGrad: (target: unknown, options?: ZeroGradOptions) => zeroGrad(stateTarget(target), options),
    zero_grad: (target: unknown, options?: ZeroGradOptions) => zeroGrad(stateTarget(target), options),
    requiresGrad: (target: unknown, requiresGrad?: boolean) => setRequiresGrad(stateTarget(target), requiresGrad),
    requiresGrad_: (target: unknown, requiresGrad?: boolean) => setRequiresGrad(stateTarget(target), requiresGrad),
    requires_grad_: (target: unknown, requiresGrad?: boolean) => setRequiresGrad(stateTarget(target), requiresGrad),
    freeze: (target: unknown) => setRequiresGrad(stateTarget(target), false),
    unfreeze: (target: unknown) => setRequiresGrad(stateTarget(target), true),
    stateDict: (target: unknown, prefixOrOptions: unknown = "") => stateDictForTarget(target, prefixOrOptions),
    state_dict: (target: unknown, prefixOrOptions: unknown = "") => stateDictForTarget(target, prefixOrOptions),
    loadStateDict: loadStateDictForTarget,
    load_state_dict: loadStateDictForTarget,
    trace: (target: unknown, options?: ModuleTraceOptions) => traceModule(moduleTarget(target), options),
    compileSupport: (target: unknown, options?: CompileOptions) => compileSupportForModule(moduleTarget(target), options),
    compile_support: (target: unknown, options?: CompileOptions) => compileSupportForModule(moduleTarget(target), options),
    requireCompileSupport: (target: unknown, options?: CompileOptions) => requireCompileSupportForModule(moduleTarget(target), options),
    require_compile_support: (target: unknown, options?: CompileOptions) => requireCompileSupportForModule(moduleTarget(target), options),
    compilerSignatures: (target: unknown, options?: CompileOptions) => compilerSignaturesForModule(moduleTarget(target), options),
    compiler_signatures: (target: unknown, options?: CompileOptions) => compilerSignaturesForModule(moduleTarget(target), options),
    tensorProgramIr: (target: unknown, options?: CompileOptions) => tensorProgramIrForModule(moduleTarget(target), options),
    tensor_program_ir: (target: unknown, options?: CompileOptions) => tensorProgramIrForModule(moduleTarget(target), options),
    kernelPlan: (target: unknown, options?: CompileOptions) => kernelPlanForModule(moduleTarget(target), options),
    kernel_plan: (target: unknown, options?: CompileOptions) => kernelPlanForModule(moduleTarget(target), options),
    bufferLayout: (target: unknown, options?: CompileOptions) => bufferLayoutForModule(moduleTarget(target), options),
    buffer_layout: (target: unknown, options?: CompileOptions) => bufferLayoutForModule(moduleTarget(target), options),
    memoryLayout: (target: unknown, options?: CompileOptions) => memoryLayoutForModule(moduleTarget(target), options),
    memory_layout: (target: unknown, options?: CompileOptions) => memoryLayoutForModule(moduleTarget(target), options),
    inputShape: (target: unknown, options?: CompileOptions) => inputShapeForModule(moduleTarget(target), options),
    input_shape: (target: unknown, options?: CompileOptions) => inputShapeForModule(moduleTarget(target), options),
    outputShape: (target: unknown, options?: CompileOptions) => outputShapeForModule(moduleTarget(target), options),
    output_shape: (target: unknown, options?: CompileOptions) => outputShapeForModule(moduleTarget(target), options),
    shapeConstraints: (target: unknown, options?: CompileOptions) => shapeConstraintsForModule(moduleTarget(target), options),
    shape_constraints: (target: unknown, options?: CompileOptions) => shapeConstraintsForModule(moduleTarget(target), options),
    parameterLayout: (target: unknown, options?: CompileOptions) => parameterLayoutForModule(moduleTarget(target), options),
    parameter_layout: (target: unknown, options?: CompileOptions) => parameterLayoutForModule(moduleTarget(target), options),
    explain: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    preflight: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    compileExplanation: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    compile_explanation: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    compilePlan: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    compile_plan: (target: unknown, options?: CompileOptions) => explainModule(moduleTarget(target), options),
    requireCompilePlan: (target: unknown, options?: CompileOptions) => requireCompilePlanForModule(moduleTarget(target), options),
    require_compile_plan: (target: unknown, options?: CompileOptions) => requireCompilePlanForModule(moduleTarget(target), options),
    assertCompilePlan: (target: unknown, options?: CompileOptions) => requireCompilePlanForModule(moduleTarget(target), options),
    assert_compile_plan: (target: unknown, options?: CompileOptions) => requireCompilePlanForModule(moduleTarget(target), options),
    canCompile: (target: unknown, options?: CompileOptions) => canCompileModule(moduleTarget(target), options),
    can_compile: (target: unknown, options?: CompileOptions) => canCompileModule(moduleTarget(target), options),
    compile: (target: unknown, options?: CompileOptions) => compileModule(moduleTarget(target), options),
    native: (target: unknown, options?: CompileOptions, bindOptions?: unknown) => moduleNativeInference(target, options, bindOptions),
    inference: (target: unknown, options?: CompileOptions, bindOptions?: unknown) => moduleNativeInference(target, options, bindOptions),
    compileInference: (target: unknown, options?: CompileOptions, bindOptions?: unknown) => moduleNativeInference(target, options, bindOptions),
    compile_inference: (target: unknown, options?: CompileOptions, bindOptions?: unknown) => moduleNativeInference(target, options, bindOptions),
    bindParameters: (target: unknown, options?: CompileOptions) => bindModuleParameters(moduleTarget(target), options),
    bind_parameters: (target: unknown, options?: CompileOptions) => bindModuleParameters(moduleTarget(target), options),
    placeParameters: (target: unknown, program: unknown, options?: unknown) => placeModuleParameters(moduleTarget(target), program, options),
    place_parameters: (target: unknown, program: unknown, options?: unknown) => placeModuleParameters(moduleTarget(target), program, options),
    bindingPlan: bindingPlanForModuleBindings,
    binding_plan: bindingPlanForModuleBindings,
    requireBindingPlan: requireBindingPlanForModuleBindings,
    require_binding_plan: requireBindingPlanForModuleBindings,
    acceptsModuleCompilePlan,
    requireModuleCompilePlan,
    assertModuleCompilePlan,
    assert_module_compile_plan,
    matchesModuleCompilePlanSignature,
    acceptsModuleBindingPlan,
    requireModuleBindingPlan,
    assertModuleBindingPlan,
    assert_module_binding_plan,
    matchesModuleBindingPlanSignature,
  });
}
