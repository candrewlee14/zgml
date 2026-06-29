"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";
import {
  compilerSignaturesFromCompilerEvidence,
  type CompileEvidence,
} from "./runtime/compiler_signatures.js";
import {
  assert_compile_analysis,
  assertCompileAnalysis,
  compileAnalysisEvidence,
  compileAnalysisSignature,
  isCompileAnalysis,
  matches_compile_analysis_signature,
  matchesCompileAnalysisSignature,
  requireCompileAnalysis,
} from "./runtime/compile_analysis.js";
import {
  traceCompilerArtifacts as traceCompilerArtifactsForNamespace,
} from "./runtime/trace_compiler.js";
import {
  compileSupportRejectionReason,
  rawSequentialLayerListCompileDiagnostic,
  type RawSequentialLayerListCompileDiagnostic,
} from "./runtime/compile_support.js";
import {
  moduleBindingPlan as moduleBindingPlanForBindings,
} from "./runtime/module_bindings.js";
import {
  programModuleBindingMetadata,
} from "./runtime/program_module_binding.js";
import {
  kernelPlanInputShape,
  kernelPlanOutputShape,
  programParameterLayoutFromKernelPlan,
  programShapeConstraintsFromKernelPlan,
} from "./runtime/trace_compiler.js";
import {
  analyzeSequentialProgram as analyzeSequentialProgramForNamespace,
  traceSequentialProgram as traceSequentialProgramForNamespace,
} from "./runtime/module_compiler_policy.js";
import type {
  CompileAnalysis,
  CompileMode,
  CompiledInference,
  CompiledTrainingPlan,
  CompiledTrainingStep,
  CompileOptions,
  CompileOptionsWithInputShape,
  EmbeddingCompileOptions,
  EmbeddingModule,
  LazyCompileSupport,
  LazyTensor,
  ModuleBindingPlan,
  ModuleBindings,
  ModuleCompileExplanation,
  ModuleCompileSupport,
  ModuleCompilerSignatures,
  ModuleForwardShape,
  ModuleKernelBufferLayout,
  ModuleKernelMemoryLayout,
  ModuleKernelParameterLayout,
  ModuleKernelPlan,
  ModuleKernelShapeConstraints,
  ModuleParameterPlacementOptions,
  ModuleProgramTrace,
  ModuleTargetForwardShape,
  ModuleTensorProgramIr,
  NnCompilableModule,
  NnModule,
  Program,
  ProgramBindings,
  ProgramBindingPlan,
  ProgramBufferLayout,
  ProgramBufferLayoutSlot,
  ProgramCompileEvidence,
  ProgramDeviceBufferKind,
  ProgramExecutionPlan,
  ProgramInputBinding,
  ProgramRequirements,
  Session,
  SessionExecutionPlan,
  SessionStepContract,
  Tensor,
  TensorShapeTuple,
} from "./public_api.js";

type UnknownRecord = Record<string, unknown>;
type CompileNamespaceMethod = (options?: CompileNamespaceOptions) => unknown;
type CompileNamespaceTarget = Readonly<{
  trace?: unknown;
  compileSupport?: unknown;
  compile_support?: unknown;
  canCompile?: unknown;
  can_compile?: unknown;
  explain?: unknown;
  compileExplanation?: unknown;
  compile_explanation?: unknown;
  compilePlan?: unknown;
  compile_plan?: unknown;
  compilerSignatures?: unknown;
  compiler_signatures?: unknown;
  tensorProgramIr?: unknown;
  tensor_program_ir?: unknown;
  kernelPlan?: unknown;
  kernel_plan?: unknown;
  bufferLayout?: unknown;
  buffer_layout?: unknown;
  memoryLayout?: unknown;
  memory_layout?: unknown;
  inputShape?: unknown;
  input_shape?: unknown;
  outputShape?: unknown;
  output_shape?: unknown;
  shapeConstraints?: unknown;
  shape_constraints?: unknown;
  parameterLayout?: unknown;
  parameter_layout?: unknown;
  compile?: unknown;
}>;
type CompileNamespaceMethodName = keyof CompileNamespaceTarget;
type CompileNamespaceOptions = Readonly<CompileOptions & {
  skipLayerSupportFallback?: boolean;
}>;
type CompileEvidenceRecord = Readonly<CompileEvidence & {
  supported?: unknown;
  support?: unknown;
  compiled?: unknown;
  artifacts?: Readonly<UnknownRecord & {
    support?: unknown;
    ir?: ModuleTensorProgramIr | null;
    kernelPlan?: ModuleKernelPlan | null;
  }>;
  ir?: ModuleTensorProgramIr;
  kernelPlan?: ModuleKernelPlan;
  trace?: Readonly<UnknownRecord & {
    inputShape?: unknown;
    outputShape?: unknown;
  }>;
}>;

export type { RawSequentialLayerListCompileDiagnostic };

export type {
  CompileAnalysis,
  CompileMode,
  CompiledInference,
  CompiledTrainingPlan,
  CompiledTrainingStep,
  CompileNamespace,
  CompileOptions,
  CompileOptionsWithInputShape,
  CompileTrainingOptions,
  EmbeddingCompileOptions,
  ModuleBindingPlan,
  ModuleBindings,
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
  ModuleProgramTrace,
  ModuleTargetForwardShape,
  ModuleTensorProgramIr,
  ProgramCompileEvidence,
  PublicCompileNamespace,
} from "./public_api.js";

export {
  acceptsModuleCompilePlan,
  assert_module_compile_plan,
  assertModuleCompilePlan,
  matchesModuleCompilePlanSignature,
  requireModuleCompilePlan,
} from "./runtime/execution_plan.js";
export {
  compilerSignatureEvidence,
  compilerSignaturesFromCompilerEvidence,
  compilerEvidenceSignature,
  programCompilerSignaturesFromCompileEvidence,
  completeProgramCompileEvidence,
  isProgramCompileEvidence,
  requireProgramCompileEvidence,
  assertProgramCompileEvidence,
  assert_program_compile_evidence,
  matchesProgramCompileEvidenceSignature,
  matches_program_compile_evidence_signature,
  programCompileEvidenceSignature,
} from "./runtime/compiler_signatures.js";
export {
  freezeTensorProgramIr,
  buildTensorProgramIrForTrace,
  tensorProgramIrSignature,
} from "./runtime/tensor_program_ir.js";
export {
  compileDiagnostic,
  freezeCompileDiagnostics,
} from "./runtime/compile_diagnostics.js";
export {
  freezeSequentialTrace,
  tensorProgramIrForTrace,
  traceCompilerArtifacts,
  compiledSequentialModuleSpecFromTrace,
  supportDetailsWithTrace,
  programCompileEvidenceFromCompiledSpec,
  moduleProgramCompileArtifactsFromCompiledSpec,
  programTraceFromCompileEvidence,
  programTensorProgramIrFromCompileEvidence,
  programKernelPlanFromCompileEvidence,
  sequentialUnsupportedSupportDetails,
  shapeMismatchAnalysis,
} from "./runtime/trace_compiler.js";
export {
  requireSequentialCompilerOptions,
  flattenSequentialEntries,
  flattenSequentialLayers,
  compiledSequentialLinearSpec,
  compiledSequentialProgramSpec,
  compiledSequentialProgramSpecForNormalizedLayers,
  normalizeTraceShape,
  defaultTraceInputShape,
  sequentialProgramSupportDetails,
  traceCompilerUnsupportedReason,
  traceSequentialProgram,
  analyzeSequentialProgram,
  createTraceModuleCompiler,
} from "./runtime/module_compiler_policy.js";

export const compileManifest = Object.freeze({
  kind: "zgml-compile",
  ...tsProductManifestPolicy("src/ts/compile.ts"),
  runtimePath: "Trace -> TensorProgramIr -> KernelPlan -> Program",
});

export {
  assert_compile_analysis,
  assertCompileAnalysis,
  compileAnalysisSignature,
  isCompileAnalysis,
  matches_compile_analysis_signature,
  matchesCompileAnalysisSignature,
  requireCompileAnalysis,
};

function requireObjectTarget(target: unknown, name: string): CompileNamespaceTarget {
  if (!target || typeof target !== "object") {
    throw new Error(`compile.${name} requires a compile-capable module, lazy graph, or Sequential layer list`);
  }
  return target as CompileNamespaceTarget;
}

function targetMethod(target: unknown, name: CompileNamespaceMethodName): CompileNamespaceMethod | null {
  const source = requireObjectTarget(target, name);
  const method = source[name];
  return typeof method === "function" ? method.bind(source) as CompileNamespaceMethod : null;
}

export function trace(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleProgramTrace;
export function trace(target: LazyTensor, options?: CompileOptions): ModuleProgramTrace;
export function trace(target: unknown, options?: CompileNamespaceOptions): unknown;
export function trace(target: unknown, options: CompileNamespaceOptions = {}) {
  const method = targetMethod(target, "trace");
  if (method) return method(options);
  if (Array.isArray(target)) return traceSequentialProgramForNamespace(target, options);
  throw new Error("compile.trace requires a compile-capable module, lazy graph, or Sequential layer list");
}

export function analyze<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>> | CompileAnalysis;
export function analyze(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation | CompileAnalysis;
export function analyze(target: LazyTensor, options?: CompileOptions): CompileAnalysis;
export function analyze(target: unknown, options?: CompileNamespaceOptions): unknown;
export function analyze(target: unknown, options: CompileNamespaceOptions = {}): unknown {
  if (Array.isArray(target)) return analyzeSequentialProgramForNamespace(target, options);
  const traced = trace(target, options) as ModuleProgramTrace;
  const artifacts = Object.freeze(traceCompilerArtifactsForNamespace(traced));
  return compileAnalysisEvidence(traced, artifacts);
}

export function compileSupport<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
export function compileSupport(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
export function compileSupport(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
export function compileSupport(target: unknown, options?: CompileNamespaceOptions): unknown;
export function compileSupport(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "compileSupport");
  if (camel) return camel(options);
  const snake = targetMethod(target, "compile_support");
  if (snake) return snake(options);
  const analysis = analyze(target, options);
  const evidence = objectEvidence(analysis);
  if (typeof evidence?.supported === "boolean") {
    return analysis;
  }
  return evidence?.support ?? evidence?.artifacts?.support ?? analysis;
}

export const compile_support = compileSupport;

export function requireCompileSupport<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
export function requireCompileSupport(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
export function requireCompileSupport(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
export function requireCompileSupport(target: unknown, options?: CompileNamespaceOptions): unknown;
export function requireCompileSupport(target: unknown, options: CompileNamespaceOptions = {}) {
  const support = compileSupport(target, options);
  if (!support || typeof support !== "object") {
    throw new Error("compile.requireCompileSupport requires structured compileSupport evidence");
  }
  if (objectEvidence(support)?.supported !== true) {
    throw new Error(`compile.requireCompileSupport rejected unsupported target: ${compileSupportRejectionReason(support)}`);
  }
  return support;
}

export const require_compile_support = requireCompileSupport;
export const assertCompileSupport = requireCompileSupport;
export const assert_compile_support = requireCompileSupport;

function objectEvidence(value: unknown): CompileEvidenceRecord | null {
  return value && typeof value === "object" ? value as CompileEvidenceRecord : null;
}

function nestedObjectEvidence(value: unknown): CompileEvidenceRecord | null {
  return value && typeof value === "object" ? value as CompileEvidenceRecord : null;
}

function compileEvidence(target: unknown, options: CompileNamespaceOptions = {}): CompileEvidenceRecord {
  const support = compileSupport(target, options);
  if (!support || typeof support !== "object") return {};
  const evidence = support as CompileEvidenceRecord;
  const supportEvidence = nestedObjectEvidence(evidence.support);
  if (supportEvidence) return supportEvidence;
  const compiledEvidence = nestedObjectEvidence(evidence.compiled);
  if (compiledEvidence) return compiledEvidence;
  return evidence;
}

export function canCompile(target: NnModule | readonly NnModule[], options?: CompileOptions): boolean;
export function canCompile(target: LazyTensor, options?: CompileOptions): boolean;
export function canCompile(target: unknown, options?: CompileNamespaceOptions): boolean;
export function canCompile(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "canCompile");
  if (camel) return Boolean(camel(options));
  const snake = targetMethod(target, "can_compile");
  if (snake) return Boolean(snake(options));
  return Boolean(objectEvidence(compileSupport(target, options))?.supported);
}

export const can_compile = canCompile;

export function explain<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>> | ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
export function explain(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation | ModuleCompileSupport;
export function explain(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
export function explain(target: unknown, options?: CompileNamespaceOptions): unknown;
export function explain(target: unknown, options: CompileNamespaceOptions = {}) {
  const direct = targetMethod(target, "explain");
  if (direct) return direct(options);
  const camel = targetMethod(target, "compileExplanation");
  if (camel) return camel(options);
  const snake = targetMethod(target, "compile_explanation");
  if (snake) return snake(options);
  const plan = targetMethod(target, "compilePlan") ?? targetMethod(target, "compile_plan");
  if (plan) return plan(options);
  return compileSupport(target, options);
}

export const preflight = explain;
export const compileExplanation = explain;
export const compile_explanation = explain;
export const compilePlan = explain;
export const compile_plan = explain;

export function requireCompilePlan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>> | ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
export function requireCompilePlan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation | ModuleCompileSupport;
export function requireCompilePlan(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
export function requireCompilePlan(target: unknown, options?: CompileNamespaceOptions): unknown;
export function requireCompilePlan(target: unknown, options: CompileNamespaceOptions = {}) {
  const plan = explain(target, options);
  if (!plan || typeof plan !== "object") {
    throw new Error("compile.requireCompilePlan requires structured compile explanation evidence");
  }
  if (objectEvidence(plan)?.supported !== true) {
    throw new Error(`compile.requireCompilePlan rejected unsupported target: ${compileSupportRejectionReason(plan)}`);
  }
  return plan;
}

export const require_compile_plan = requireCompilePlan;
export const assertCompilePlan = requireCompilePlan;
export const assert_compile_plan = requireCompilePlan;

export function compilerSignatures(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleCompilerSignatures | null;
export function compilerSignatures(target: unknown, options?: CompileNamespaceOptions): ModuleCompilerSignatures | null;
export function compilerSignatures(target: unknown, options: CompileNamespaceOptions = {}): ModuleCompilerSignatures | null {
  const camel = targetMethod(target, "compilerSignatures");
  if (camel) return camel(options) as ModuleCompilerSignatures | null;
  const snake = targetMethod(target, "compiler_signatures");
  if (snake) return snake(options) as ModuleCompilerSignatures | null;
  return compilerSignaturesFromCompilerEvidence(compileEvidence(target, options)) as ModuleCompilerSignatures | null;
}

export const compiler_signatures = compilerSignatures;

export function tensorProgramIr(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleTensorProgramIr | null;
export function tensorProgramIr(target: unknown, options?: CompileNamespaceOptions): ModuleTensorProgramIr | null;
export function tensorProgramIr(target: unknown, options: CompileNamespaceOptions = {}): ModuleTensorProgramIr | null {
  const camel = targetMethod(target, "tensorProgramIr");
  if (camel) return camel(options) as ModuleTensorProgramIr | null;
  const snake = targetMethod(target, "tensor_program_ir");
  if (snake) return snake(options) as ModuleTensorProgramIr | null;
  const evidence = compileEvidence(target, options);
  return evidence.ir ?? evidence.artifacts?.ir ?? null;
}

export const tensor_program_ir = tensorProgramIr;

export function kernelPlan(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelPlan | null;
export function kernelPlan(target: unknown, options?: CompileNamespaceOptions): ModuleKernelPlan | null;
export function kernelPlan(target: unknown, options: CompileNamespaceOptions = {}): ModuleKernelPlan | null {
  const camel = targetMethod(target, "kernelPlan");
  if (camel) return camel(options) as ModuleKernelPlan | null;
  const snake = targetMethod(target, "kernel_plan");
  if (snake) return snake(options) as ModuleKernelPlan | null;
  const evidence = compileEvidence(target, options);
  return evidence.kernelPlan ?? evidence.artifacts?.kernelPlan ?? null;
}

export const kernel_plan = kernelPlan;

export function bufferLayout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelBufferLayout | null;
export function bufferLayout(target: unknown, options?: CompileNamespaceOptions): ModuleKernelBufferLayout | null;
export function bufferLayout(target: unknown, options: CompileNamespaceOptions = {}): ModuleKernelBufferLayout | null {
  const camel = targetMethod(target, "bufferLayout");
  if (camel) return camel(options) as ModuleKernelBufferLayout | null;
  const snake = targetMethod(target, "buffer_layout");
  if (snake) return snake(options) as ModuleKernelBufferLayout | null;
  const plan = kernelPlan(target, options);
  return plan ? plan.bufferLayout : null;
}

export const buffer_layout = bufferLayout;

export function memoryLayout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelMemoryLayout | null;
export function memoryLayout(target: unknown, options?: CompileNamespaceOptions): ModuleKernelMemoryLayout | null;
export function memoryLayout(target: unknown, options: CompileNamespaceOptions = {}): ModuleKernelMemoryLayout | null {
  const camel = targetMethod(target, "memoryLayout");
  if (camel) return camel(options) as ModuleKernelMemoryLayout | null;
  const snake = targetMethod(target, "memory_layout");
  if (snake) return snake(options) as ModuleKernelMemoryLayout | null;
  const plan = kernelPlan(target, options);
  return plan ? plan.memoryLayout : null;
}

export const memory_layout = memoryLayout;

export function inputShape(target: NnModule | readonly NnModule[], options?: CompileOptions): readonly number[] | null;
export function inputShape(target: LazyTensor, options?: CompileOptions): readonly number[] | null;
export function inputShape(target: unknown, options?: CompileNamespaceOptions): unknown;
export function inputShape(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "inputShape");
  if (camel) return camel(options);
  const snake = targetMethod(target, "input_shape");
  if (snake) return snake(options);
  const evidence = compileEvidence(target, options);
  const inputShapeFromKernelPlan = kernelPlanInputShape(evidence.kernelPlan ?? evidence.artifacts?.kernelPlan);
  if (inputShapeFromKernelPlan) return inputShapeFromKernelPlan;
  if (Array.isArray(evidence.trace?.inputShape)) return evidence.trace.inputShape;
  return null;
}

export const input_shape = inputShape;

export function outputShape<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleTargetForwardShape<Target, S> | null;
export function outputShape(target: NnModule | readonly NnModule[], options?: CompileOptions): readonly number[] | null;
export function outputShape<const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Shape | null;
export function outputShape(target: unknown, options?: CompileNamespaceOptions): unknown;
export function outputShape(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "outputShape");
  if (camel) return camel(options);
  const snake = targetMethod(target, "output_shape");
  if (snake) return snake(options);
  const evidence = compileEvidence(target, options);
  const outputShapeFromKernelPlan = kernelPlanOutputShape(evidence.kernelPlan ?? evidence.artifacts?.kernelPlan);
  if (outputShapeFromKernelPlan) return outputShapeFromKernelPlan;
  if (Array.isArray(evidence.trace?.outputShape)) return evidence.trace.outputShape;
  return null;
}

export const output_shape = outputShape;

export function shapeConstraints(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleKernelShapeConstraints | null;
export function shapeConstraints(target: LazyTensor, options?: CompileOptions): ModuleKernelShapeConstraints | null;
export function shapeConstraints(target: unknown, options?: CompileNamespaceOptions): unknown;
export function shapeConstraints(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "shapeConstraints");
  if (camel) return camel(options);
  const snake = targetMethod(target, "shape_constraints");
  if (snake) return snake(options);
  return programShapeConstraintsFromKernelPlan(kernelPlan(target, options));
}

export const shape_constraints = shapeConstraints;

export function parameterLayout(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleKernelParameterLayout | null;
export function parameterLayout(target: LazyTensor, options?: CompileOptions): ModuleKernelParameterLayout | null;
export function parameterLayout(target: unknown, options?: CompileNamespaceOptions): unknown;
export function parameterLayout(target: unknown, options: CompileNamespaceOptions = {}) {
  const camel = targetMethod(target, "parameterLayout");
  if (camel) return camel(options);
  const snake = targetMethod(target, "parameter_layout");
  if (snake) return snake(options);
  return programParameterLayoutFromKernelPlan(kernelPlan(target, options));
}

export const parameter_layout = parameterLayout;

export function compile<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): Program<S, ModuleForwardShape<Target, S>>;
export function compile<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): Program<S, ModuleForwardShape<Target, S>>;
export function compile<const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Program<TensorShapeTuple, Shape>;
export function compile(target: NnCompilableModule, options?: CompileOptions): Program;
export function compile(target: readonly NnModule[], options?: CompileOptions): RawSequentialLayerListCompileDiagnostic;
export function compile(target: unknown, options?: CompileNamespaceOptions): unknown;
export function compile(target: unknown, options: CompileNamespaceOptions = {}) {
  if (Array.isArray(target)) {
    return rawSequentialLayerListCompileDiagnostic();
  }
  const method = targetMethod(target, "compile");
  if (!method) throw new Error("compile.compile requires a module with compile()");
  return method(options);
}

type CompiledInferenceHandle<InputShape extends TensorShapeTuple, OutputShape extends TensorShapeTuple> = CompiledInference<InputShape, OutputShape>;

function compiledInferenceHandle<InputShape extends TensorShapeTuple, OutputShape extends TensorShapeTuple>(
  program: Program<InputShape, OutputShape>,
  session: Session<InputShape, OutputShape>,
  target: unknown,
  options: CompileNamespaceOptions,
  bindOptions?: ModuleParameterPlacementOptions | ProgramBindings,
): CompiledInferenceHandle<InputShape, OutputShape> {
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    session.dispose();
    program.dispose();
  };
  const executionPlan = program.requireExecutionPlan() as ProgramExecutionPlan<InputShape, OutputShape>;
  const moduleBindingMetadata = programModuleBindingMetadata(session);
  const inferenceBindings = moduleBindingMetadata?.bindings ?? bindOptions ?? {};
  const parameterBindingPlan = moduleBindingMetadata
    ? moduleBindingPlanForBindings(moduleBindingMetadata.bindings) as ModuleBindingPlan<InputShape, OutputShape>
    : null;
  const programBindingPlan = program.bindingPlan(inferenceBindings as ProgramBindings<InputShape, OutputShape>) as ProgramBindingPlan<InputShape, OutputShape>;
  const handle = ((input: ProgramInputBinding<InputShape>) => session.stepTensor(input)) as unknown as CompiledInferenceHandle<InputShape, OutputShape>;
  return Object.freeze(Object.assign(handle, {
    native: true,
    engine: "zig",
    runtime: "native",
    program,
    session,
    executionPlan() {
      return executionPlan;
    },
    requireExecutionPlan() {
      return executionPlan;
    },
    explain() {
      return explain(target, options) as ModuleCompileExplanation<InputShape, OutputShape> | ModuleCompileSupport<InputShape, OutputShape>;
    },
    preflight() {
      return explain(target, options) as ModuleCompileExplanation<InputShape, OutputShape> | ModuleCompileSupport<InputShape, OutputShape>;
    },
    compileSupport() {
      return compileSupport(target, options) as ModuleCompileSupport<InputShape, OutputShape>;
    },
    compileEvidence() {
      return program.compileEvidence() as ProgramCompileEvidence | null;
    },
    parameterBindingPlan() {
      return parameterBindingPlan;
    },
    parameter_binding_plan() {
      return parameterBindingPlan;
    },
    programBindingPlan() {
      return programBindingPlan;
    },
    program_binding_plan() {
      return programBindingPlan;
    },
    requirements() {
      return program.requirements() as ProgramRequirements;
    },
    bufferLayout() {
      return program.bufferLayout() as ProgramBufferLayout;
    },
    bufferSlotNames() {
      return program.bufferSlotNames();
    },
    bufferSlot(nameOrKind: ProgramDeviceBufferKind | string) {
      return program.bufferSlot(nameOrKind) as ProgramBufferLayoutSlot | null;
    },
    sessionBufferLayout() {
      return session.bufferLayout() as ProgramBufferLayout;
    },
    sessionBufferSlotNames() {
      return session.bufferSlotNames();
    },
    sessionBufferSlot(nameOrKind: ProgramDeviceBufferKind | string) {
      return session.bufferSlot(nameOrKind) as ProgramBufferLayoutSlot | null;
    },
    stepContract() {
      return session.stepContract() as SessionStepContract;
    },
    hotPathPlan(params?: unknown) {
      return session.hotPathPlan(params) as SessionExecutionPlan<InputShape, OutputShape>;
    },
    inputShape() {
      return program.inputShape();
    },
    outputShape() {
      return program.outputShape();
    },
    kernelPlan() {
      return program.kernelPlan();
    },
    compilerSignatures() {
      return program.compilerSignatures();
    },
    forward(input: ProgramInputBinding<InputShape>) {
      return session.stepTensor(input);
    },
    call(input: ProgramInputBinding<InputShape>) {
      return session.stepTensor(input);
    },
    __call__(input: ProgramInputBinding<InputShape>) {
      return session.stepTensor(input);
    },
    stepTensor(input: ProgramInputBinding<InputShape>) {
      return session.stepTensor(input);
    },
    into(output: Float32Array, input: ProgramInputBinding<InputShape>) {
      return session.executeInto(output, { input });
    },
    prepareInto(output: Float32Array, input: ProgramInputBinding<InputShape>) {
      return session.prepareExecuteInto(output, { input });
    },
    dispose,
    free: dispose,
  }));
}

export function compileForInference<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(
  target: Target,
  options: EmbeddingCompileOptions<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): CompiledInference<S, ModuleForwardShape<Target, S>>;
export function compileForInference<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(
  target: Target,
  options: CompileOptionsWithInputShape<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): CompiledInference<S, ModuleForwardShape<Target, S>>;
export function compileForInference<const Shape extends TensorShapeTuple, const S extends TensorShapeTuple>(
  target: LazyTensor<Shape>,
  options: CompileOptionsWithInputShape<S>,
  bindOptions: ProgramBindings<S, Shape>,
): CompiledInference<S, Shape>;
export function compileForInference<const Shape extends TensorShapeTuple>(
  target: LazyTensor<Shape>,
  options: CompileOptions,
  bindOptions: ProgramBindings<TensorShapeTuple, Shape>,
): CompiledInference<TensorShapeTuple, Shape>;
export function compileForInference(target: NnCompilableModule, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
export function compileForInference(target: unknown, options: CompileNamespaceOptions = {}, bindOptions?: ModuleParameterPlacementOptions | ProgramBindings): CompiledInference {
  if (Array.isArray(target)) {
    throw new Error("compile.compileForInference requires a module; wrap layer lists in nn.Sequential");
  }
  const program = compile(target, options);
  if (!program || typeof program !== "object") {
    throw new Error("compile.compileForInference expected compile() to return a Program");
  }
  if (typeof (program as { requireExecutionPlan?: unknown }).requireExecutionPlan !== "function") {
    throw new Error("compile.compileForInference requires a native executable Program; use compile.explain for diagnostics-only targets");
  }
  const bindModule = (program as { bindModule?: unknown }).bindModule;
  const bind = (program as { bind?: unknown }).bind;
  const targetCanPlaceModuleParameters = target != null && typeof target === "object" && typeof (target as { placeParameters?: unknown }).placeParameters === "function";
  const session = typeof bindModule === "function" && targetCanPlaceModuleParameters
    ? bindModule.call(program, target, bindOptions)
    : typeof bind === "function" && bindOptions != null
      ? bind.call(program, bindOptions)
      : null;
  if (!session || typeof session !== "object") {
    throw new Error("compile.compileForInference expected bindModule() or explicit Program bindings to return a Session");
  }
  return compiledInferenceHandle(program as Program, session as Session, target, options, bindOptions);
}

export const compile_for_inference = compileForInference;
export const forInference = compileForInference;
export const for_inference = compileForInference;
export const native = compileForInference;
export const inference = compileForInference;

export function run<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(
  target: Target,
  input: ProgramInputBinding<S>,
  options: EmbeddingCompileOptions<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): Tensor<ModuleForwardShape<Target, S>>;
export function run<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(
  target: Target,
  input: ProgramInputBinding<S>,
  options: CompileOptionsWithInputShape<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): Tensor<ModuleForwardShape<Target, S>>;
export function run(target: NnCompilableModule, input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Tensor;
export function run(target: unknown, input: unknown, options: CompileNamespaceOptions = {}, bindOptions?: ModuleParameterPlacementOptions | ProgramBindings) {
  const inferenceHandle = (compileForInference as (
    target: unknown,
    options?: CompileNamespaceOptions,
    bindOptions?: ModuleParameterPlacementOptions | ProgramBindings,
  ) => CompiledInference)(target, options, bindOptions);
  try {
    return inferenceHandle.forward(input as ProgramInputBinding);
  } finally {
    inferenceHandle.dispose();
  }
}

export const infer = run;
export const predict = run;

export function runInto<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(
  output: Float32Array,
  target: Target,
  input: ProgramInputBinding<S>,
  options: EmbeddingCompileOptions<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): Float32Array;
export function runInto<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(
  output: Float32Array,
  target: Target,
  input: ProgramInputBinding<S>,
  options: CompileOptionsWithInputShape<S>,
  bindOptions?: ModuleParameterPlacementOptions,
): Float32Array;
export function runInto(output: Float32Array, target: NnCompilableModule, input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
export function runInto(output: Float32Array, target: unknown, input: unknown, options: CompileNamespaceOptions = {}, bindOptions?: ModuleParameterPlacementOptions | ProgramBindings) {
  const inferenceHandle = (compileForInference as (
    target: unknown,
    options?: CompileNamespaceOptions,
    bindOptions?: ModuleParameterPlacementOptions | ProgramBindings,
  ) => CompiledInference)(target, options, bindOptions);
  try {
    return inferenceHandle.into(output, input as ProgramInputBinding);
  } finally {
    inferenceHandle.dispose();
  }
}

export const run_into = runInto;
export const inferInto = runInto;
export const infer_into = runInto;
export const predictInto = runInto;
export const predict_into = runInto;

export function trainingStep(_model: unknown, _optimizer: unknown, _options: Record<string, unknown> = {}): CompiledTrainingStep {
  throw new Error("compile.trainingStep requires a native Node or Bun runtime");
}

export const training_step = trainingStep;
export const compileForTraining: typeof trainingStep = trainingStep;
export const compile_for_training: typeof trainingStep = trainingStep;
export const forTraining: typeof trainingStep = trainingStep;
export const for_training: typeof trainingStep = trainingStep;
