"use strict";

declare const require: (path: string) => unknown;

import type {
  CompileOptions,
  ModuleBindingPlan,
  ModuleCompileDiagnostic,
  ModuleCompileSupport,
  ModuleParameterPlacementOptions,
  ModuleParameterInfo,
  ProgramBufferLayout,
  ProgramBufferLayoutSlot,
  ProgramBindings,
  ProgramModuleCompatibility,
} from "../public_api.js";
import { moduleOptionsWithEvidenceShape } from "./module_compatibility.js";

type AnyRecord = Record<string, any>;

type ModuleBindingOptions = Readonly<CompileOptions>;
type ModulePlacementOptions = Readonly<ModuleParameterPlacementOptions>;

type CompileSupportModule = {
  compileSupport?: (options: Readonly<ModuleBindingOptions>) => unknown;
};

type ModuleBindingMetadataSource = CompileSupportModule & Readonly<{
  parameterNames?: unknown;
  parameterInfos?: unknown;
}>;

type ModuleBindingMetadata = Readonly<{
  module: ModuleBindingMetadataSource;
  options: Readonly<ModuleBindingOptions>;
  support: unknown;
}>;

type ProgramWithModuleCompatibility = {
  compileEvidence?: () => unknown;
  moduleCompatibility?: (module: CompileSupportModule, options: Readonly<ModuleBindingOptions>) => ProgramModuleCompatibility | null | undefined;
};

type ProgramBufferLike = {
  writeFloat32: (values: Float32Array) => unknown;
};

type ProgramWithParameterPlacement = ProgramWithModuleCompatibility & {
  createBuffer?: (name: string, options?: unknown) => ProgramBufferLike;
  bufferLayout?: () => ProgramBufferLayout | null | undefined;
  inputShape?: () => readonly number[] | null | undefined;
};

type ModuleWithParameterBinding = CompileSupportModule & {
  bindParameters?: (options: ModuleBindingOptions) => {
    weights?: unknown;
    bias?: unknown;
  };
};

type SequentialParameterEntry = {
  readonly layer: {
    readonly weight?: Float32Array | ArrayLike<number> | null;
    readonly bias?: Float32Array | ArrayLike<number> | null;
  };
};

type SequentialModuleParameterSpec = {
  readonly desc: {
    readonly weightsLen: number;
    readonly biasLen: number;
  };
  readonly entries: readonly SequentialParameterEntry[];
};

type SequentialTinyLinearParameterSpec = {
  readonly kind: "tiny-linear";
  readonly layer: {
    bindParameters: () => {
      weights?: unknown;
      bias?: unknown;
    };
  };
};

type SequentialCompiledParameterSpec =
  | (SequentialModuleParameterSpec & { readonly kind: "module" })
  | SequentialTinyLinearParameterSpec
  | Readonly<Record<string, unknown> & { kind: string }>;

type ModuleCompatibilityHelpers = {
  moduleCompatibilityForSupportEvidence: (evidence: unknown, support: unknown) => ProgramModuleCompatibility | null | undefined;
  moduleCompatibilityForProgramEvidence: (evidence: unknown, module: ModuleBindingMetadataSource | null | undefined, options?: Readonly<ModuleBindingOptions>) => ProgramModuleCompatibility | null | undefined;
};

const {
  moduleCompatibilityForSupportEvidence,
  moduleCompatibilityForProgramEvidence,
} = require("./module_compatibility.js") as ModuleCompatibilityHelpers;

export const moduleBindingsSymbol = Symbol.for("zgml.moduleBindings");

function moduleBindingOptions(options: ModuleBindingOptions = {}): Readonly<ModuleBindingOptions> {
  if (!options || typeof options !== "object") return Object.freeze({});
  const copy: Record<string, unknown> = { ...options };
  if (Array.isArray(copy.inputShape)) copy.inputShape = Object.freeze(copy.inputShape.slice());
  if (Array.isArray(copy.outputShape)) copy.outputShape = Object.freeze(copy.outputShape.slice());
  return Object.freeze(copy) as Readonly<ModuleBindingOptions>;
}

function moduleBindingSupport(module: CompileSupportModule, options: Readonly<ModuleBindingOptions>): unknown {
  if (!module || typeof module.compileSupport !== "function") return null;
  return module.compileSupport(options);
}

export function annotateModuleBindings<T>(bindings: T, module: ModuleBindingMetadataSource, options: ModuleBindingOptions = {}): T {
  if (!bindings || typeof bindings !== "object") return bindings;
  const current = Object.getOwnPropertyDescriptor(bindings, moduleBindingsSymbol);
  if (current && current.configurable === false) return bindings;
  const normalizedOptions = moduleBindingOptions(options);
  Object.defineProperty(bindings, moduleBindingsSymbol, {
    value: Object.freeze({
      module,
      options: normalizedOptions,
      support: moduleBindingSupport(module, normalizedOptions),
    }),
    enumerable: true,
    configurable: false,
    writable: false,
  });
  return bindings;
}

export function moduleBindingMetadata(bindings: unknown): ModuleBindingMetadata | null {
  if (!bindings || typeof bindings !== "object") return null;
  const metadata = (bindings as Record<symbol, unknown>)[moduleBindingsSymbol];
  return metadata && typeof metadata === "object" ? metadata as ModuleBindingMetadata : null;
}

function bindingParameterNames(module: ModuleBindingMetadataSource): readonly string[] {
  const parameterNames = module && module.parameterNames;
  if (typeof parameterNames !== "function") return Object.freeze([]);
  return Object.freeze(Array.from((parameterNames as (this: ModuleBindingMetadataSource, prefix?: string) => readonly unknown[]).call(module), String));
}

function bindingParameterInfos(module: ModuleBindingMetadataSource): readonly ModuleParameterInfo[] {
  const parameterInfos = module && module.parameterInfos;
  if (typeof parameterInfos !== "function") return Object.freeze([]);
  return Object.freeze(Array.from((parameterInfos as (this: ModuleBindingMetadataSource, prefix?: string) => readonly unknown[]).call(module), (info) => {
    const record = info && typeof info === "object" ? info as Partial<ModuleParameterInfo> & { requires_grad?: unknown } : {};
    return Object.freeze({
      name: String(record.name),
      index: Number(record.index),
      scalarCount: Number(record.scalarCount),
      shape: Object.freeze(Array.from(record.shape ?? [], Number)),
      layout: String(record.layout ?? ""),
      requiresGrad: Boolean(record.requiresGrad ?? record.requires_grad ?? true),
      requires_grad: Boolean(record.requires_grad ?? record.requiresGrad ?? true),
    });
  })) as readonly ModuleParameterInfo[];
}

function bindingShape(value: unknown): readonly number[] | null {
  return Array.isArray(value) ? Object.freeze(Array.from(value, Number)) : null;
}

function shapeSignature(shape: readonly number[] | null) {
  return shape ? shape.join("x") : "null";
}

function moduleBindingPlanSignature(fields: {
  readonly moduleBindings: boolean;
  readonly supported: boolean | null;
  readonly inputShape: readonly number[] | null;
  readonly outputShape: readonly number[] | null;
  readonly parameterNames: readonly string[];
  readonly parameterInfos: readonly ModuleParameterInfo[];
  readonly diagnostics: readonly ModuleCompileDiagnostic[];
}) {
  const parameterSignature = fields.parameterInfos
    .map((info) => `${info.index}:${info.name}:${info.scalarCount}:${Array.isArray(info.shape) ? info.shape.join("x") : ""}`)
    .join(",");
  return [
    "module-bindings-plan",
    `moduleBindings=${fields.moduleBindings ? 1 : 0}`,
    `supported=${fields.supported === null ? "null" : fields.supported ? 1 : 0}`,
    `inputShape=${shapeSignature(fields.inputShape)}`,
    `outputShape=${shapeSignature(fields.outputShape)}`,
    `parameters=${fields.parameterNames.join(",")}`,
    `parameterInfos=${parameterSignature}`,
    `diagnostics=${fields.diagnostics.map((diagnostic) => diagnostic.code).join(",")}`,
  ].join("|");
}

export function moduleBindingPlan(bindings: unknown): ModuleBindingPlan {
  const metadata = moduleBindingMetadata(bindings);
  if (!metadata) {
    const diagnostics = Object.freeze([]);
    const parameterNames = Object.freeze([]);
    const parameterInfos = Object.freeze([]);
    return Object.freeze({
      kind: "zgml.nn.module-bindings-plan",
      signature: moduleBindingPlanSignature({
        moduleBindings: false,
        supported: null,
        inputShape: null,
        outputShape: null,
        parameterNames,
        parameterInfos,
        diagnostics,
      }),
      moduleBindings: false,
      options: Object.freeze({}),
      support: null,
      supported: null,
      reason: null,
      inputShape: null,
      outputShape: null,
      parameterNames,
      parameterInfos,
      diagnostics,
    });
  }
  const support = metadata.support as ModuleCompileSupport | null;
  const trace = support && support.trace && typeof support.trace === "object" ? support.trace as AnyRecord : null;
  const supported = support && typeof support.supported === "boolean" ? support.supported : null;
  const inputShape = bindingShape(support?.inputShape) ?? bindingShape(trace?.inputShape);
  const outputShape = bindingShape(support?.outputShape) ?? bindingShape(trace?.outputShape);
  const parameterNames = bindingParameterNames(metadata.module);
  const parameterInfos = bindingParameterInfos(metadata.module);
  const diagnostics = Object.freeze(Array.from(support?.diagnostics ?? [], (diagnostic) => diagnostic as ModuleCompileDiagnostic));
  return Object.freeze({
    kind: "zgml.nn.module-bindings-plan",
    signature: moduleBindingPlanSignature({
      moduleBindings: true,
      supported,
      inputShape,
      outputShape,
      parameterNames,
      parameterInfos,
      diagnostics,
    }),
    moduleBindings: true,
    options: metadata.options,
    support,
    supported,
    reason: support && typeof support.reason === "string" ? support.reason : null,
    inputShape,
    outputShape,
    parameterNames,
    parameterInfos,
    diagnostics,
  });
}

export function validateProgramBindModuleCompatibility(program: ProgramWithModuleCompatibility, bindings: unknown): void {
  const metadata = moduleBindingMetadata(bindings);
  if (!metadata) return;
  let compatibility: ProgramModuleCompatibility | null | undefined = null;
  if (metadata.module && program && typeof program.compileEvidence === "function") {
    compatibility = moduleCompatibilityForProgramEvidence(program.compileEvidence(), metadata.module, metadata.options || {});
  } else if (metadata.support && program && typeof program.compileEvidence === "function") {
    compatibility = moduleCompatibilityForSupportEvidence(program.compileEvidence(), metadata.support);
  } else if (metadata.module && program && typeof program.moduleCompatibility === "function") {
    compatibility = program.moduleCompatibility(metadata.module, metadata.options || {});
  } else {
    return;
  }
  if (!compatibility || compatibility.compatible !== true) {
    const reason = compatibility && compatibility.reason
      ? compatibility.reason
      : "module bindings are incompatible with this Program";
    throw new Error(`Program.bind ModuleBindings are not compatible with this Program: ${reason}`);
  }
}

function packedBindingF32(data: unknown, name: string): Float32Array {
  if (data === undefined || data === null) return new Float32Array(0);
  if (data instanceof Float32Array) return data;
  if (Array.isArray(data) || ArrayBuffer.isView(data)) return Float32Array.from(data as ArrayLike<number>);
  throw new Error(`${name} must be packed Float32Array-compatible parameter data`);
}

export function packedSequentialModuleParameters(spec: SequentialModuleParameterSpec): Record<string, Float32Array> {
  const weights = new Float32Array(spec.desc.weightsLen);
  const bias = new Float32Array(spec.desc.biasLen);
  let weightsOffset = 0;
  let biasOffset = 0;

  for (const entry of spec.entries) {
    const layer = entry.layer;
    if (layer.weight) {
      weights.set(layer.weight, weightsOffset);
      weightsOffset += layer.weight.length;
    }
    if (layer.bias) {
      bias.set(layer.bias, biasOffset);
      biasOffset += layer.bias.length;
    }
  }

  return spec.desc.biasLen === 0 ? { weights } : { weights, bias };
}

export function packedSequentialProgramParameters(spec: SequentialCompiledParameterSpec): Record<string, unknown> {
  if (spec.kind === "tiny-linear") {
    const bindings = (spec as SequentialTinyLinearParameterSpec).layer.bindParameters();
    return bindings.bias === undefined ? { weights: bindings.weights } : { weights: bindings.weights, bias: bindings.bias };
  }
  if (spec.kind === "module") return packedSequentialModuleParameters(spec as SequentialModuleParameterSpec);
  throw new Error(`unsupported compiled Sequential program kind: ${spec.kind}`);
}

function validateModuleParameterPlacement(program: ProgramWithParameterPlacement, weights: Float32Array, bias: Float32Array): void {
  if (!program || typeof program.bufferLayout !== "function") return;
  const layout = program.bufferLayout();
  const checkSlot = (name: "weights" | "bias", values: Float32Array): void => {
    const slot = layout && layout[name];
    if (!slot || typeof slot !== "object") return;
    if (slot.scalarType !== "f32") {
      throw new Error(`nn.placeParameters ${name} slot dtype ${slot.scalarType} is unsupported; eager zgml module parameters store f32 values`);
    }
    if (values.length !== slot.elementCount) {
      throw new Error(`nn.placeParameters ${name} length ${values.length} does not match Program ${name} slot length ${slot.elementCount}`);
    }
  };
  checkSlot("weights", weights);
  checkSlot("bias", bias);
}

function modulePlacementOptionsWithProgramShape(
  program: ProgramWithParameterPlacement,
  options: ModulePlacementOptions = {},
): ModulePlacementOptions {
  if (program && typeof program.compileEvidence === "function") {
    return moduleOptionsWithEvidenceShape(program.compileEvidence() as AnyRecord | null | undefined, options);
  }
  if (options && Object.prototype.hasOwnProperty.call(options, "inputShape") && options.inputShape !== undefined) {
    return options;
  }
  if (!program || typeof program.inputShape !== "function") return options;
  const inputShape = program.inputShape();
  if (!Array.isArray(inputShape) || inputShape.length === 0) return options;
  return {
    ...options,
    inputShape: inputShape.slice(),
  };
}

function validateModuleCompatibilityForPlacement(
  module: ModuleWithParameterBinding,
  program: ProgramWithParameterPlacement,
  options: ModulePlacementOptions,
): void {
  if (!program || typeof program.moduleCompatibility !== "function") return;
  const compatibility = program.moduleCompatibility(module, options);
  if (!compatibility || compatibility.compatible !== true) {
    const reason = compatibility && compatibility.reason
      ? compatibility.reason
      : "module is incompatible with this Program";
    throw new Error(`nn.placeParameters module is not compatible with this Program: ${reason}`);
  }
}

export function placeModuleParameterBindings(
  module: ModuleWithParameterBinding,
  program: ProgramWithParameterPlacement,
  options: ModulePlacementOptions = {},
): ProgramBindings {
  if (!module || typeof module.bindParameters !== "function") {
    throw new Error("nn.placeParameters requires a module with native Program support");
  }
  if (!program || typeof program.createBuffer !== "function") {
    throw new Error("nn.placeParameters requires a compiled zgml Program");
  }
  const placementOptions = modulePlacementOptionsWithProgramShape(program, options || {});
  validateModuleCompatibilityForPlacement(module, program, placementOptions);
  const bindings = module.bindParameters(placementOptions);
  const out: Record<string, unknown> = {};
  const weights = packedBindingF32(bindings.weights, "weights");
  const bias = packedBindingF32(bindings.bias, "bias");
  validateModuleParameterPlacement(program, weights, bias);
  if (weights.length > 0) {
    const weightsBuffer = program.createBuffer("weights", (placementOptions.weights ?? placementOptions.weight) ?? placementOptions);
    weightsBuffer.writeFloat32(weights);
    out.weights = weightsBuffer;
  } else {
    out.weights = weights;
  }
  if (bindings.bias !== undefined) {
    if (bias.length > 0) {
      const biasBuffer = program.createBuffer("bias", placementOptions.bias ?? placementOptions);
      biasBuffer.writeFloat32(bias);
      out.bias = biasBuffer;
    } else {
      out.bias = bias;
    }
  }
  return annotateModuleBindings(out, module, placementOptions);
}
