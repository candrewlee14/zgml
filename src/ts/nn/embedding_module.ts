import { compileDiagnostic } from "../runtime/compile_diagnostics.js";
import { initializeModuleMode } from "../train/module_mode.js";
import type { NnModule } from "../public_api.js";
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
  installSingleModuleCompileMethods,
  packedSingleModuleBindings,
  singleModuleCompileSupport,
  type SingleModuleCompileHooksInput,
} from "./single_module_compile.js";

type UnknownRecord = Record<string, unknown>;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
export type EmbeddingTensorConstructOptions = Readonly<Record<string, unknown>>;
export type EmbeddingTensor = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly length: number;
  readonly grad?: Float32Array | null;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
  zeroGrad(options?: Readonly<Record<string, unknown>>): void;
};
export type EmbeddingGradTensor = EmbeddingTensor & {
  readonly grad: Float32Array | null;
  readonly requiresGrad: boolean;
};
export type EmbeddingParameter = {
  readonly name: string;
  readonly data: Float32Array;
  readonly grad?: Float32Array | null;
  readonly layout?: string;
  tensor: EmbeddingTensor;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
};
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: EmbeddingTensorConstructOptions) => EmbeddingTensor;
type IndexValues = (values: unknown, label: string) => ArrayLike<number>;
type TensorGradAdder = (tensor: EmbeddingGradTensor, grad: Float32Array) => void;
type RequirePositiveInteger = BivariantCallback<[value: unknown, label: string], number>;
type DefaultedF32 = (values: unknown, length: number, label: string, fallback: (length: number) => Float32Array, shape?: readonly number[]) => Float32Array;
type ParameterFactory = (name: string, values: Float32Array, shape: readonly number[], layout: string) => EmbeddingParameter;
type ParameterView = BivariantCallback<[prefix: string, parameter: EmbeddingParameter], EmbeddingParameter>;
type SequentialTraceFn = (layers: readonly NnModule[], options?: UnknownRecord) => UnknownRecord;
type FreezeSequentialTraceFn = (trace: UnknownRecord) => unknown;
type NativeEagerIndexSelectInto = (output: Float32Array, input: unknown, index: Uint32Array, options: Readonly<{ outer: number; axisLen: number; inner: number }>) => Float32Array;

function embeddingGradTensor(tensor: EmbeddingTensor): EmbeddingGradTensor {
  return tensor as EmbeddingGradTensor;
}

export type EmbeddingModuleClassOptions = Readonly<Record<string, unknown> & SingleModuleCompileHooksInput & {
  Tensor: TensorConstructor;
  indexValues: IndexValues;
  addTensorGrad: TensorGradAdder;
  requirePositiveInteger: RequirePositiveInteger;
  defaultedF32: DefaultedF32;
  zerosF32: (length: number) => Float32Array;
  makeParameter: ParameterFactory;
  parameterView: ParameterView;
  traceSequentialProgram: SequentialTraceFn;
  freezeSequentialTrace: FreezeSequentialTraceFn;
  nativeEagerIndexSelectInto?: NativeEagerIndexSelectInto;
}>;

export function createEmbeddingModuleClass(options: EmbeddingModuleClassOptions) {
  const TensorClass = options.Tensor;
  const indexValues = options.indexValues;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : () => true;
  const requirePositiveInteger = options.requirePositiveInteger;
  const defaultedF32 = options.defaultedF32;
  const zerosF32 = options.zerosF32;
  const makeParameter = options.makeParameter;
  const parameterView = options.parameterView;
  const nativeEagerIndexSelectInto = options.nativeEagerIndexSelectInto;
  const stateHooks = createStatefulModuleStateHooks(options, "EmbeddingModule");
  const traceSequentialProgram = options.traceSequentialProgram;
  const freezeSequentialTrace = options.freezeSequentialTrace;
  const compileHooks = createSingleModuleCompileHooks(options, "EmbeddingModule");
  if (
    typeof TensorClass !== "function" ||
    typeof indexValues !== "function" ||
    typeof addTensorGrad !== "function" ||
    typeof requirePositiveInteger !== "function" ||
    typeof defaultedF32 !== "function" ||
    typeof zerosF32 !== "function" ||
    typeof makeParameter !== "function" ||
    typeof parameterView !== "function" ||
    typeof traceSequentialProgram !== "function" ||
    typeof freezeSequentialTrace !== "function"
  ) {
    throw new Error("EmbeddingModule factory requires tensor, state, trace, packing, placement, and compile hooks");
  }

  class EmbeddingModule {
    kind: "embedding";
    numEmbeddings: number;
    embeddingDim: number;
    weight: Float32Array;
    weightParam: EmbeddingParameter;
    training?: boolean;

    constructor(numEmbeddings: unknown, embeddingDim: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "embedding";
      this.numEmbeddings = requirePositiveInteger(numEmbeddings, "embedding numEmbeddings");
      this.embeddingDim = requirePositiveInteger(embeddingDim, "embedding embeddingDim");
      this.weight = defaultedF32(config.weight ?? config.weights, this.numEmbeddings * this.embeddingDim, "embedding weight", zerosF32, [this.numEmbeddings, this.embeddingDim]);
      this.weightParam = makeParameter("weight", this.weight, [this.numEmbeddings, this.embeddingDim], "row-major:embedding.weight[num_embeddings,embedding_dim]");
    }

    forward(indexValuesInput: unknown) {
      const inputShape = indexValuesInput instanceof TensorClass ? indexValuesInput.shape : null;
      const indices = indexValues(indexValuesInput, "embedding indices");
      const nativeIndices = new Uint32Array(indices.length);
      for (let row = 0; row < indices.length; row += 1) {
        const index = indices[row];
        if (!Number.isSafeInteger(index) || index < 0 || index >= this.numEmbeddings) {
          throw new Error(`embedding index ${index} at position ${row} is out of range 0..${this.numEmbeddings - 1}`);
        }
        nativeIndices[row] = index;
      }
      const out = new Float32Array(indices.length * this.embeddingDim);
      if (typeof nativeEagerIndexSelectInto === "function" && indices.length > 0) {
        nativeEagerIndexSelectInto(out, this.weightParam.tensor, nativeIndices, {
          outer: 1,
          axisLen: this.numEmbeddings,
          inner: this.embeddingDim,
        });
      } else {
        for (let row = 0; row < indices.length; row += 1) {
          const index = nativeIndices[row];
          const source = index * this.embeddingDim;
          const dest = row * this.embeddingDim;
          out.set(this.weight.subarray(source, source + this.embeddingDim), dest);
        }
      }
      const needsGrad = gradModeEnabled() && this.weightParam.tensor.requiresGrad;
      return new TensorClass(out, [...(inputShape ?? [indices.length]), this.embeddingDim], {
        requiresGrad: needsGrad,
        prev: needsGrad ? [this.weightParam.tensor] : [],
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad || !this.weightParam.tensor.requiresGrad) return;
          const weightGrad = new Float32Array(this.weight.length);
          for (let row = 0; row < indices.length; row += 1) {
            const index = indices[row];
            const source = index * this.embeddingDim;
            const dest = row * this.embeddingDim;
            for (let f = 0; f < this.embeddingDim; f += 1) {
              weightGrad[source + f] += grad[dest + f];
            }
          }
          addTensorGrad(embeddingGradTensor(this.weightParam.tensor), weightGrad);
        },
      });
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      return [parameterView(prefix, this.weightParam)];
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

    compileSupport(options: UnknownRecord = {}) {
      if (!options.inputShape) {
        return compileHooks.moduleCompileSupport(false, "nn.Embedding.compile requires inputShape because output length depends on index count", {
          diagnostics: [compileDiagnostic(
            "missing-input-shape",
            "nn.Embedding.compile requires inputShape because output length depends on index count",
            { stage: "ir" },
          )],
          trace: freezeSequentialTrace(traceSequentialProgram([this as unknown as NnModule], options)),
        });
      }
      return singleModuleCompileSupport(this, compileHooks, options);
    }
  }

  installStatefulModuleStateMethods(EmbeddingModule.prototype, stateHooks);
  installSingleModuleCompileMethods(EmbeddingModule.prototype, compileHooks, {
    compileSupport: false,
    bindParameters: packedSingleModuleBindings,
  });
  return EmbeddingModule;
}
