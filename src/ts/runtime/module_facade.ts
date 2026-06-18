"use strict";

import { compileSupportRejectionReason } from "./compile_support.js";
import { compilerSignaturesFromCompilerEvidence } from "./compiler_signatures.js";
import { moduleBindingPlanRejectionReason } from "./execution_plan.js";
import { moduleBindingPlan, placeModuleParameterBindings } from "./module_bindings.js";
import {
  kernelPlanInputShape,
  kernelPlanOutputShape,
  programParameterLayoutFromKernelPlan,
  programShapeConstraintsFromKernelPlan,
} from "./trace_compiler.js";
import type {
  CompileOptions,
  ModuleCompileDiagnostic,
  ModuleCompileExplanation,
  ModuleCompileSupport,
  ModuleCompilerSignatures,
  ModuleKernelBufferLayout,
  ModuleKernelMemoryLayout,
  ModuleKernelParameterLayout,
  ModuleKernelPlan,
  ModuleKernelShapeConstraints,
  ModuleParameterPlacementOptions,
  ProgramBufferLayout,
  ModuleProgramTrace,
  ModuleTraceOptions,
  ModuleTensorProgramIr,
  ProgramModuleCompatibility,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type CompileSupportEvidence = ModuleCompileSupport & AnyRecord;
type ModuleFacadeTraceTarget = Readonly<{
  trace?: (options?: ModuleTraceOptions) => unknown;
}>;
type ModuleFacadeCompileSupportTarget = Readonly<{
  compileSupport?: (options?: CompileOptions) => unknown;
}>;
type ModuleFacadeCompileTarget = Readonly<{
  compile?: (options?: CompileOptions) => unknown;
}>;
type ModuleFacadeParameterNamesTarget = Readonly<{
  parameterNames?: (...args: readonly unknown[]) => readonly unknown[];
}>;
type ModuleFacadeParameterInfosTarget = Readonly<{
  parameterInfos?: (...args: readonly unknown[]) => readonly unknown[];
}>;
type ModuleFacadeParameterBindingTarget = Readonly<{
  bindParameters?: (options?: CompileOptions) => unknown;
}>;
export type ModuleFacadeTarget = Readonly<{
  readonly forward?: unknown;
  readonly trace?: unknown;
  readonly compileSupport?: unknown;
  readonly parameterNames?: unknown;
  readonly parameterInfos?: unknown;
  readonly parameters?: unknown;
  readonly namedParameters?: unknown;
  readonly parameterInfo?: unknown;
  readonly children?: unknown;
  readonly modules?: unknown;
  readonly namedChildren?: unknown;
  readonly namedModules?: unknown;
  readonly compile?: unknown;
  readonly bindParameters?: unknown;
}>;
type ModuleFacadeProgramBuffer = {
  writeFloat32: (values: Float32Array) => unknown;
};
type ModuleFacadeProgram = Readonly<{
  compileEvidence?: () => unknown;
  moduleCompatibility?: (module: ModuleFacadePlacementTarget, options: Readonly<ModuleParameterPlacementOptions>) => ProgramModuleCompatibility | null | undefined;
  createBuffer?: (name: string, options?: unknown) => ModuleFacadeProgramBuffer;
  bufferLayout?: () => ProgramBufferLayout | null | undefined;
  inputShape?: () => readonly number[] | null | undefined;
}>;
type ModuleFacadePlacementTarget = Readonly<{
  compileSupport?: (options: Readonly<CompileOptions>) => unknown;
  bindParameters?: (options: Readonly<CompileOptions>) => {
    weights?: unknown;
    bias?: unknown;
  };
}>;
type TraceSequentialProgramCallback = {
  bivarianceHack(layers: readonly any[], options?: ModuleTraceOptions): unknown;
}["bivarianceHack"];

export type ModuleFacadeHelpersOptions = Readonly<{
  traceSequentialProgram: TraceSequentialProgramCallback;
}>;

export function createModuleFacadeHelpers(options: ModuleFacadeHelpersOptions) {
  const traceSequentialProgram = options.traceSequentialProgram;
  if (typeof traceSequentialProgram !== "function") {
    throw new Error("module facade helpers require traceSequentialProgram");
  }

	  function traceModule(module: ModuleFacadeTarget, traceOptions: ModuleTraceOptions = {}) {
	    if (!module || typeof module.forward !== "function") {
	      throw new Error("nn.trace requires an nn module");
	    }
	    if (typeof module.trace === "function") return (module as ModuleFacadeTraceTarget).trace!(traceOptions);
	    return traceSequentialProgram([module], traceOptions);
	  }

	  function compileSupportForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): CompileSupportEvidence {
	    if (!module || typeof module.compileSupport !== "function") {
	      throw new Error("nn.compileSupport requires an nn module");
	    }
	    return (module as ModuleFacadeCompileSupportTarget).compileSupport!(compileOptions) as CompileSupportEvidence;
	  }

  function requireCompileSupportForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): CompileSupportEvidence {
    const support = compileSupportForModule(module, compileOptions);
    if (!support || typeof support !== "object") {
      throw new Error("nn.requireCompileSupport requires structured compileSupport evidence");
    }
    if ((support as AnyRecord).supported !== true) {
      throw new Error(`nn.requireCompileSupport rejected unsupported module: ${compileSupportRejectionReason(support, { fallbackReason: "module is unsupported by the native Program compiler" })}`);
    }
    return support;
  }

  function compilerSignaturesForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleCompilerSignatures | null {
    const support = compileSupportForModule(module, compileOptions);
    return compilerSignaturesFromCompilerEvidence(support) as ModuleCompilerSignatures | null;
  }

  function tensorProgramIrForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleTensorProgramIr | null {
    const support = compileSupportForModule(module, compileOptions);
    return support && support.ir ? support.ir : null;
  }

  function kernelPlanForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleKernelPlan | null {
    const support = compileSupportForModule(module, compileOptions);
    return support && support.kernelPlan ? support.kernelPlan : null;
  }

  function bufferLayoutForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleKernelBufferLayout | null {
    const kernelPlan = kernelPlanForModule(module, compileOptions);
    return kernelPlan ? kernelPlan.bufferLayout : null;
  }

  function memoryLayoutForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleKernelMemoryLayout | null {
    const kernelPlan = kernelPlanForModule(module, compileOptions);
    return kernelPlan ? kernelPlan.memoryLayout : null;
  }

  function inputShapeForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    const support = compileSupportForModule(module, compileOptions);
    const inputShape = kernelPlanInputShape(support && support.kernelPlan);
    if (inputShape) return inputShape;
    if (support && support.trace && Array.isArray(support.trace.inputShape)) return support.trace.inputShape;
    return null;
  }

  function outputShapeForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    const support = compileSupportForModule(module, compileOptions);
    const outputShape = kernelPlanOutputShape(support && support.kernelPlan);
    if (outputShape) return outputShape;
    if (support && support.trace && Array.isArray(support.trace.outputShape)) return support.trace.outputShape;
    return null;
  }

  function shapeConstraintsForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleKernelShapeConstraints | null {
    return programShapeConstraintsFromKernelPlan(kernelPlanForModule(module, compileOptions));
  }

  function parameterLayoutForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleKernelParameterLayout | null {
    return programParameterLayoutFromKernelPlan(kernelPlanForModule(module, compileOptions));
  }

  function moduleParameterNames(module: ModuleFacadeTarget) {
    if (!module || typeof module.parameterNames !== "function") return Object.freeze([]);
    return Object.freeze(Array.from((module as ModuleFacadeParameterNamesTarget).parameterNames!(), String));
  }

  function moduleParameterInfos(module: ModuleFacadeTarget) {
    if (!module || typeof module.parameterInfos !== "function") return Object.freeze([]);
    return Object.freeze(Array.from((module as ModuleFacadeParameterInfosTarget).parameterInfos!(), (info: unknown) => {
      const record = info && typeof info === "object" ? info as AnyRecord : {};
      return Object.freeze({
        name: String(record.name),
        index: Number(record.index),
        scalarCount: Number(record.scalarCount),
        shape: Object.freeze(Array.from(record.shape ?? [], Number)),
        layout: String(record.layout ?? ""),
        requiresGrad: Boolean(record.requiresGrad ?? record.requires_grad ?? true),
        requires_grad: Boolean(record.requires_grad ?? record.requiresGrad ?? true),
      });
    }));
  }

  function shapeSignature(shape: readonly unknown[] | null | undefined) {
    return Array.isArray(shape) ? shape.map(Number).join("x") : "null";
  }

  function parameterInfoSignature(info: AnyRecord) {
    return [
      Number(info.index),
      String(info.name),
      Number(info.scalarCount),
      shapeSignature(info.shape),
      String(info.layout ?? ""),
    ].join(":");
  }

  function diagnosticSignature(diagnostic: AnyRecord) {
    return [
      String(diagnostic.stage ?? ""),
      String(diagnostic.code ?? ""),
      String(diagnostic.op ?? ""),
      String(diagnostic.path ?? ""),
    ].join(":");
  }

  function compilerSignaturesSignature(signatures: AnyRecord | null | undefined) {
    if (!signatures) return "null";
    return [
      `ir=${signatures.ir ?? ""}`,
      `kernel=${signatures.kernelPlan ?? ""}`,
      `memory=${signatures.memoryLayout ?? ""}`,
      `parameters=${signatures.parameterLayout ?? ""}`,
      `buffers=${signatures.bufferLayout ?? ""}`,
    ].join(",");
  }

  function moduleCompileExplanationSignature(fields: AnyRecord) {
    return [
      "nn-compile-explanation",
      `supported=${fields.supported ? 1 : 0}`,
      `native=${fields.nativePath ?? "null"}`,
      `model=${fields.modelKind ?? "null"}`,
      `inputShape=${shapeSignature(fields.inputShape)}`,
      `outputShape=${shapeSignature(fields.outputShape)}`,
      `parameters=${fields.parameterNames.join(",")}`,
      `parameterInfos=${fields.parameterInfos.map(parameterInfoSignature).join(",")}`,
      `parameterCount=${fields.parameterCount ?? "null"}`,
      `parameterScalars=${fields.parameterScalarCount ?? "null"}`,
      `traceOps=${fields.trace && typeof fields.trace.opCount === "number" ? fields.trace.opCount : "null"}`,
      `irOps=${fields.ir && typeof fields.ir.opCount === "number" ? fields.ir.opCount : "null"}`,
      `kernelOps=${fields.kernelPlan && typeof fields.kernelPlan.opCount === "number" ? fields.kernelPlan.opCount : "null"}`,
      `compiler=${compilerSignaturesSignature(fields.compilerSignatures)}`,
      `diagnostics=${fields.diagnostics.map(diagnosticSignature).join(",")}`,
    ].join("|");
  }

  function explainModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}): ModuleCompileExplanation {
    const support = compileSupportForModule(module, compileOptions);
    const kernelPlan: ModuleKernelPlan | null = support && support.kernelPlan ? support.kernelPlan : null;
    const trace: ModuleProgramTrace | null = support && support.trace ? support.trace : null;
    const ir: ModuleTensorProgramIr | null = support && support.ir ? support.ir : null;
    const inputShape = kernelPlanInputShape(kernelPlan) ?? (trace && Array.isArray(trace.inputShape) ? trace.inputShape : null);
    const outputShape = kernelPlanOutputShape(kernelPlan) ?? (trace && Array.isArray(trace.outputShape) ? trace.outputShape : null);
    const parameterNames = moduleParameterNames(module);
    const parameterInfos = moduleParameterInfos(module);
    const parameterCount = support.parameterCount ?? (ir && ir.parameterCount) ?? (trace && trace.parameterCount) ?? null;
    const parameterScalarCount = support.parameterScalarCount ?? (ir && ir.parameterScalarCount) ?? (trace && trace.parameterScalarCount) ?? null;
    const compilerSignatures = compilerSignaturesFromCompilerEvidence(support) as ModuleCompilerSignatures;
    const diagnostics = Object.freeze(Array.from(support.diagnostics ?? [])) as readonly ModuleCompileDiagnostic[];
    return Object.freeze({
      kind: "zgml.nn.compile-explanation",
      supported: Boolean(support.supported),
      reason: support.reason ?? null,
      nativePath: support.nativePath ?? null,
      modelKind: support.modelKind ?? null,
      signature: moduleCompileExplanationSignature({
        supported: Boolean(support.supported),
        nativePath: support.nativePath ?? null,
        modelKind: support.modelKind ?? null,
        inputShape,
        outputShape,
        parameterNames,
        parameterInfos,
        parameterCount,
        parameterScalarCount,
        trace,
        ir,
        kernelPlan,
        compilerSignatures,
        diagnostics,
      }),
      inputShape,
      outputShape,
      parameterNames,
      parameterInfos,
      parameterCount,
      parameterScalarCount,
      trace,
      ir,
      kernelPlan,
      compilerSignatures,
      bufferLayout: kernelPlan ? kernelPlan.bufferLayout : null,
      memoryLayout: kernelPlan ? kernelPlan.memoryLayout : null,
      shapeConstraints: programShapeConstraintsFromKernelPlan(kernelPlan),
      parameterLayout: programParameterLayoutFromKernelPlan(kernelPlan),
      diagnostics,
    });
  }

  function requireCompilePlanForModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    const plan = explainModule(module, compileOptions);
    if (plan.supported === true) return plan;
    throw new Error(`nn.requireCompilePlan rejected unsupported module: ${compileSupportRejectionReason(plan, { fallbackReason: "module is unsupported by the native Program compiler" })}`);
  }

  function canCompileModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    return compileSupportForModule(module, compileOptions).supported;
  }

  function compileModule(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    if (!module || typeof module.compile !== "function") {
      throw new Error("nn.compile requires a module with native Program support");
    }
    return (module as ModuleFacadeCompileTarget).compile!(compileOptions);
  }

  function bindModuleParameters(module: ModuleFacadeTarget, compileOptions: CompileOptions = {}) {
    if (!module || typeof module.bindParameters !== "function") {
      throw new Error("nn.bindParameters requires a module with native Program support");
    }
    return (module as ModuleFacadeParameterBindingTarget).bindParameters!(compileOptions);
  }

  function placeModuleParameters(module: ModuleFacadeTarget, program: ModuleFacadeProgram, placementOptions: ModuleParameterPlacementOptions = {}) {
    return placeModuleParameterBindings(module as ModuleFacadePlacementTarget, program, placementOptions);
  }

  function bindingPlanForModuleBindings(bindings: unknown) {
    return moduleBindingPlan(bindings);
  }

  function requireBindingPlanForModuleBindings(bindings: unknown) {
    const plan = bindingPlanForModuleBindings(bindings);
    if (plan.moduleBindings === true && plan.supported !== false) return plan;
    throw new Error(`nn.requireBindingPlan rejected bindings: ${moduleBindingPlanRejectionReason(plan)}`);
  }

  return Object.freeze({
    traceModule,
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
    compileModule,
    bindModuleParameters,
    placeModuleParameters,
    bindingPlanForModuleBindings,
    requireBindingPlanForModuleBindings,
  });
}
