import { annotateModuleBindings } from "../runtime/module_bindings.js";
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
  attachTinyLinearEvidence,
  createSequentialProgramCompileHooks,
  installSequentialProgramCompileMethods,
  type SequentialCompiledSpec,
  type SequentialProgramAnalysis,
  type SequentialProgramCompileHooks,
  type SequentialProgramCompileHooksInput,
  type SequentialModuleRecord,
} from "./sequential_program_compile.js";

type UnknownRecord = Record<string, unknown>;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type ModuleCompileOptions = Readonly<CompileOptions & Record<string, unknown>>;
export type LinearTensorConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  requires_grad?: boolean;
  prev?: readonly unknown[];
  backward?: (grad: Float32Array | null) => void;
}>;
export type LinearTensor = {
  data: Float32Array;
  shape: readonly number[];
  length: number;
  rank: number;
  reshape(shape: readonly number[]): LinearTensor;
  matmul(other: unknown, otherShape: unknown): LinearTensor;
  add(other: unknown): LinearTensor;
};
export type LinearParameter = Readonly<{
  name: string;
  tensor: unknown;
  data: Float32Array;
  grad?: Float32Array | null;
  layout?: string;
  requiresGrad?: boolean;
  requires_grad?: boolean;
}>;
type LinearSequentialModuleRecord = SequentialModuleRecord & Readonly<{
  readonly compileSupport?: (options: ModuleCompileOptions) => unknown;
  readonly weight: Float32Array;
  readonly bias?: Float32Array | null;
  readonly inFeatures: number;
  readonly outFeatures: number;
}>;
type TensorConstructor = new (
  values: Float32Array,
  shape?: readonly number[],
  options?: LinearTensorConstructOptions,
) => LinearTensor;
type F32WithLength = (values: unknown, length: number, label: string) => Float32Array;
type RequirePositiveInteger = BivariantCallback<[value: unknown, label: string], number>;
type DefaultedF32 = (
  values: unknown,
  length: number,
  label: string,
  fallback: (length: number) => Float32Array,
  shape?: readonly number[],
) => Float32Array;
type ParameterFactory = (name: string, values: Float32Array, shape: readonly number[], layout: string) => LinearParameter;
type ParameterView = BivariantCallback<[prefix: string, parameter: LinearParameter], UnknownRecord>;
type TinyLinearModelConstructor = {
  create(desc: { readonly inputLen: number; readonly outputLen: number }): { compile(options?: CompileOptions): unknown };
};

function linearSequentialModule(module: SequentialModuleRecord): LinearSequentialModuleRecord {
  return module as LinearSequentialModuleRecord;
}

export type LinearModuleClassOptions = Readonly<Record<string, unknown> & SequentialProgramCompileHooksInput & {
  Tensor: TensorConstructor;
  f32WithLength: F32WithLength;
  requirePositiveInteger: RequirePositiveInteger;
  defaultedF32: DefaultedF32;
  zerosF32: (length: number) => Float32Array;
  makeParameter: ParameterFactory;
  parameterView: ParameterView;
  TinyLinearModel: TinyLinearModelConstructor;
}>;

export function createLinearModuleClass(options: LinearModuleClassOptions) {
  const TensorClass = options.Tensor;
  const f32WithLength = options.f32WithLength;
  const requirePositiveInteger = options.requirePositiveInteger;
  const defaultedF32 = options.defaultedF32;
  const zerosF32 = options.zerosF32;
  const makeParameter = options.makeParameter;
  const parameterView = options.parameterView;
  const stateHooks = createStatefulModuleStateHooks(options, "LinearModule");
  const TinyLinearModel = options.TinyLinearModel;
  const compileHooks = createSequentialProgramCompileHooks(options, "LinearModule", { requirePackParameters: false });
  if (
    typeof TensorClass !== "function" ||
    typeof f32WithLength !== "function" ||
    typeof requirePositiveInteger !== "function" ||
    typeof defaultedF32 !== "function" ||
    typeof zerosF32 !== "function" ||
    typeof makeParameter !== "function" ||
    typeof parameterView !== "function" ||
    !TinyLinearModel ||
    typeof TinyLinearModel.create !== "function"
  ) {
    throw new Error("LinearModule factory requires tensor, state, analysis, tiny-linear, placement, and compile hooks");
  }

  function linearBias(config: UnknownRecord, outFeatures: number) {
    if (config.bias === false) return null;
    const biasValues = config.biasValues ?? (config.bias === undefined || config.bias === true ? undefined : config.bias);
    return defaultedF32(biasValues, outFeatures, "linear bias", zerosF32, [outFeatures]);
  }

  class LinearModule {
    kind: "linear";
    inFeatures: number;
    outFeatures: number;
    weight: Float32Array;
    bias: Float32Array | null;
    weightParam: LinearParameter;
    biasParam: LinearParameter | null;
    training?: boolean;

    constructor(inFeatures: unknown, outFeatures: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "linear";
      this.inFeatures = requirePositiveInteger(inFeatures, "linear inFeatures");
      this.outFeatures = requirePositiveInteger(outFeatures, "linear outFeatures");
      this.weight = defaultedF32(config.weight ?? config.weights, this.inFeatures * this.outFeatures, "linear weight", zerosF32, [this.inFeatures, this.outFeatures]);
      this.bias = linearBias(config, this.outFeatures);
      this.weightParam = makeParameter("weight", this.weight, [this.inFeatures, this.outFeatures], "row-major:linear.weight[in_features,out_features]");
      this.biasParam = this.bias ? makeParameter("bias", this.bias, [this.outFeatures], "row-major:linear.bias[out_features]") : null;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof TensorClass
        ? inputValues as LinearTensor
        : new TensorClass(f32WithLength(inputValues, this.inFeatures, "linear input"), [this.inFeatures]);
      if (input.rank === 1) {
        if (input.length !== this.inFeatures) {
          throw new Error(`linear input length must be ${this.inFeatures}, got ${input.length}`);
        }
        let out = input.reshape([1, this.inFeatures]).matmul(this.weightParam.tensor as LinearTensor, undefined).reshape([this.outFeatures]);
        if (this.biasParam) out = out.add(this.biasParam.tensor);
        return out;
      }
      if (input.rank < 2 || input.shape[input.shape.length - 1] !== this.inFeatures) {
        throw new Error(`linear input shape must end with ${this.inFeatures}, got [${input.shape.join(",")}]`);
      }
      const leadingShape = input.shape.slice(0, -1);
      const rowCount = leadingShape.reduce((acc: number, dim: number) => acc * dim, 1);
      let out = input.reshape([rowCount, this.inFeatures]).matmul(this.weightParam.tensor as LinearTensor, undefined);
      if (this.biasParam) out = out.add(this.biasParam.tensor);
      return out.reshape([...leadingShape, this.outFeatures]);
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      const params = [parameterView(prefix, this.weightParam)];
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
  }

  installStatefulModuleStateMethods(LinearModule.prototype, stateHooks);
  installSequentialProgramCompileMethods(LinearModule.prototype, compileHooks, {
    layersForModule: (module: SequentialModuleRecord) => [module],
    bindParameters(module: SequentialModuleRecord, _hooks: SequentialProgramCompileHooks, options: ModuleCompileOptions = {}) {
      const linearModule = linearSequentialModule(module);
      return annotateModuleBindings(
        linearModule.bias ? { weights: linearModule.weight, bias: linearModule.bias } : { weights: linearModule.weight },
        linearModule,
        options,
      );
    },
    compileTinyLinear(compiled: SequentialCompiledSpec, analysis: SequentialProgramAnalysis, compileOptions: ModuleCompileOptions, hooks: SequentialProgramCompileHooks, module: SequentialModuleRecord) {
      const linearModule = linearSequentialModule(module);
      const program = TinyLinearModel.create({ inputLen: linearModule.inFeatures, outputLen: linearModule.outFeatures }).compile(compileOptions);
      return attachTinyLinearEvidence(program, hooks, compiled, analysis);
    },
  });
  return LinearModule;
}
