"use strict";

import {
  moduleProgramDescFromKernelPlan,
  moduleProgramDescFromCompiledSpec,
  packModuleProgramDesc,
  type ModuleProgramDesc,
} from "./module_program_desc.js";

import {
  moduleCompatibilityForProgramEvidence,
} from "./module_compatibility.js";

import {
  compilerSignatureEvidence,
  compilerSignaturesFromCompilerEvidence,
  programCompilerSignaturesFromCompileEvidence,
  completeProgramCompileEvidence,
  isProgramCompileEvidence,
  requireProgramCompileEvidence,
  assertProgramCompileEvidence,
  assert_program_compile_evidence,
  matchesProgramCompileEvidenceSignature,
  matches_program_compile_evidence_signature,
  programCompileEvidenceSignature,
  type KernelPlan as CompilerSignatureKernelPlan,
} from "./compiler_signatures.js";
import {
  compileDiagnostic as compileDiagnosticValue,
  diagnosticForCompilerOp as diagnosticForCompilerOpValue,
  freezeCompileDiagnostics as freezeCompileDiagnosticsValue,
} from "./compile_diagnostics.js";

import {
  freezeTensorProgramIr,
  buildTensorProgramIrForTrace,
  tensorProgramIrSignature,
  type TensorProgramIr,
  type TensorProgramIrInput,
} from "./tensor_program_ir.js";
import {
  freezeKernelShapeConstraints,
  freezeKernelParameterLayout,
  freezeKernelPlan,
  freezePublicKernelPlan,
  kernelizeTensorProgramIr,
  programShapeConstraintsFromKernelPlan,
  kernelPlanInputShape,
  kernelPlanOutputShape,
  programMemoryLayoutFromKernelPlan,
  programParameterLayoutFromKernelPlan,
  type KernelPlan,
  type InternalKernelPlan,
} from "./kernel_plan.js";
import type {
  ModuleKernelPlan,
  ModuleProgramTrace,
  ModuleTensorProgramIr,
  ProgramCompileEvidence,
} from "../public_api.js";
import type { CompileDiagnostic } from "./compile_diagnostics.js";

type AnyRecord = Record<string, any>;
type UnknownRecord = Record<string, unknown>;
type TraceCompilerTraceInput = ModuleProgramTrace | AnyRecord;
export type CompiledTinyLinearProgramSpec = Readonly<{
  readonly kind: "tiny-linear";
  readonly layer: Readonly<{
    readonly inFeatures: number;
    readonly outFeatures: number;
    readonly weight?: ArrayLike<number> | null;
    readonly bias?: ArrayLike<number> | null;
  }>;
  readonly nativePath: "tiny-linear";
  readonly modelKind: "tiny-linear";
  readonly layerCount: 1;
  readonly trace?: ModuleProgramTrace | null;
  readonly ir?: TensorProgramIr | TensorProgramIrInput | null;
  readonly kernelPlan?: InternalKernelPlan | KernelPlan | null;
}>;
export type TraceCompilerIrResult = Readonly<{
  ir: TensorProgramIr | null;
  diagnostic: CompileDiagnostic | null;
}>;
export type TraceCompilerArtifacts = Readonly<{
  ir: TensorProgramIr | null;
  kernelPlan: InternalKernelPlan | null;
  diagnostic: CompileDiagnostic | null;
}>;
export type CompiledSequentialModuleSpec = Readonly<{
  readonly kind: "module";
  readonly nativePath: "device-program";
  readonly modelKind: "module";
  readonly layerCount: number;
  readonly entries: readonly unknown[];
  readonly trace: ModuleProgramTrace;
  readonly ir: TensorProgramIr;
  readonly kernelPlan: InternalKernelPlan;
  readonly desc: Readonly<ModuleProgramDesc>;
}>;
export type CompiledSequentialProgramSpec =
  | CompiledTinyLinearProgramSpec
  | CompiledSequentialModuleSpec;
export type TraceCompilerSupportDetails = AnyRecord & {
  trace: ModuleProgramTrace;
};
export type ProgramCompileArtifactEvidence = Readonly<{
  trace?: ModuleProgramTrace | null;
  ir?: ModuleTensorProgramIr | null;
  kernelPlan?: ModuleKernelPlan | null;
}>;
type ProgramCompileEvidenceInput = ProgramCompileEvidence | ProgramCompileArtifactEvidence;

function record(value: unknown): UnknownRecord | null {
  return value && typeof value === "object" ? value as UnknownRecord : null;
}

function compiledSequentialProgramSpec(value: unknown): CompiledSequentialProgramSpec | null {
  const spec = record(value);
  if (!spec) return null;
  if (spec.kind === "tiny-linear") {
    const layer = record(spec.layer);
    if (!layer) return null;
    return spec as CompiledTinyLinearProgramSpec;
  }
  if (spec.kind === "module" && spec.trace && spec.ir && spec.kernelPlan) {
    return spec as CompiledSequentialModuleSpec;
  }
  return null;
}

export const compileDiagnostic = compileDiagnosticValue;
export const diagnosticForCompilerOp = diagnosticForCompilerOpValue;
export const freezeCompileDiagnostics = freezeCompileDiagnosticsValue;

export function freezeTraceParameter(param: AnyRecord) {
  return Object.freeze({
    ...param,
    shape: Object.freeze((param.shape ?? []).slice()),
  });
}

export function freezeTraceOp(op: AnyRecord) {
  const copy: AnyRecord = {
    ...op,
    parameters: Object.freeze((op.parameters ?? []).map(freezeTraceParameter)),
  };
  if (Array.isArray(op.inputShape)) copy.inputShape = Object.freeze(op.inputShape.slice());
  if (Array.isArray(op.outputShape)) copy.outputShape = Object.freeze(op.outputShape.slice());
  if (Array.isArray(op.shape)) copy.shape = Object.freeze(op.shape.slice());
  return Object.freeze(copy);
}

export function freezeSequentialTrace(trace: TraceCompilerTraceInput): ModuleProgramTrace {
  const copy: AnyRecord = {
    ...trace,
    ops: Object.freeze((trace.ops ?? []).map(freezeTraceOp)),
  };
  if (Array.isArray(trace.inputShape)) copy.inputShape = Object.freeze(trace.inputShape.slice());
  if (Array.isArray(trace.outputShape)) copy.outputShape = Object.freeze(trace.outputShape.slice());
  return Object.freeze(copy) as ModuleProgramTrace;
}

function freezeShapeEvidence(shape: readonly unknown[]) {
  return Object.freeze(shape.slice());
}

const kernelizerRankCheckedOps = new Set([
  "reshape",
  "view",
  "flatten",
  "squeeze",
  "unsqueeze",
  "broadcastTo",
  "expand",
  "narrow",
  "select",
  "slice",
  "transpose",
  "permute",
  "activation",
  "conv2d",
  "avgPool2d",
  "maxPool2d",
  "sum",
  "mean",
  "max",
  "min",
  "argmax",
  "argmin",
]);

function tensorProgramIrUnsupportedDiagnostic(trace: TraceCompilerTraceInput): CompileDiagnostic | null {
  if (!trace.shapeKnown || !Array.isArray(trace.inputShape)) {
    return compileDiagnostic(
      "missing-input-shape",
      "native module Program compilation requires inputShape unless a leading Linear module provides a default shape",
      { stage: "ir" },
    );
  }
  if (!Array.isArray(trace.outputShape)) {
    return compileDiagnostic("shape-unknown", "native module Program compilation requires a statically known output shape", {
      stage: "ir",
    });
  }
  if (trace.inputShape.length < 1) {
    return compileDiagnostic("unsupported-rank", "native module Program compilation currently supports rank-1/rank-2 inputs", {
      stage: "ir",
      inputShape: trace.inputShape,
    });
  }
  if (trace.outputShape.length < 1) {
    return compileDiagnostic("unsupported-rank", "native module Program compilation currently supports rank-1/rank-2 outputs", {
      stage: "ir",
      outputShape: trace.outputShape,
    });
  }
  for (const op of trace.ops) {
    if (!Array.isArray(op.inputShape) || !Array.isArray(op.outputShape)) {
      return diagnosticForCompilerOp(op, "shape-unknown", "native module Program op requires known input/output shapes");
    }
    const rankCheckedByKernelizer = kernelizerRankCheckedOps.has(op.op);
    if (op.inputShape.length < 1 || op.outputShape.length < 1 || (!rankCheckedByKernelizer && (op.inputShape.length > 2 || op.outputShape.length > 2))) {
      return diagnosticForCompilerOp(op, "unsupported-rank", "native module Program op currently supports rank-1/rank-2 tensors");
    }
  }
  return null;
}

export function tensorProgramIrForTrace(trace: TraceCompilerTraceInput): TraceCompilerIrResult {
  const diagnostic = tensorProgramIrUnsupportedDiagnostic(trace);
  if (diagnostic) return { ir: null, diagnostic };
  return { ir: buildTensorProgramIrForTrace(trace), diagnostic: null };
}

export function traceCompilerArtifacts(trace: TraceCompilerTraceInput): TraceCompilerArtifacts {
  const { ir, diagnostic: irDiagnostic } = tensorProgramIrForTrace(trace);
  if (irDiagnostic) {
    return { ir: null, kernelPlan: null, diagnostic: irDiagnostic };
  }
  if (!ir) {
    return {
      ir: null,
      kernelPlan: null,
      diagnostic: compileDiagnostic("unsupported-graph", "Sequential graph is outside the native module Program compiler subset", {
        stage: "kernelizer",
      }),
    };
  }
  const { kernelPlan, diagnostic: kernelDiagnostic } = kernelizeTensorProgramIr(ir);
  return {
    ir,
    kernelPlan: kernelPlan ?? null,
    diagnostic: kernelDiagnostic ?? null,
  };
}

export function compiledSequentialModuleSpecFromTrace(entries: readonly unknown[], trace: TraceCompilerTraceInput, artifacts: TraceCompilerArtifacts | null = null): CompiledSequentialModuleSpec | null {
  const compilerArtifacts = artifacts ?? traceCompilerArtifacts(trace);
  const { ir, kernelPlan } = compilerArtifacts;
  if (!ir || !kernelPlan) return null;

  return {
    kind: "module",
    nativePath: "device-program",
    modelKind: "module",
    layerCount: entries.length,
    entries,
    trace: freezeSequentialTrace(trace),
    ir,
    kernelPlan,
    desc: moduleProgramDescFromKernelPlan(kernelPlan),
  };
}

export function supportDetailsWithTrace(details: AnyRecord, trace: TraceCompilerTraceInput, ir: TensorProgramIr | TensorProgramIrInput | null, kernelPlan: unknown): TraceCompilerSupportDetails {
  const support: AnyRecord = {
    ...details,
    ...compilerSignatureEvidence(ir, kernelPlan as CompilerSignatureKernelPlan | null | undefined),
    trace: freezeSequentialTrace(trace),
  };
  if (ir) {
    support.inputShape = freezeShapeEvidence(ir.inputShape);
    support.outputShape = freezeShapeEvidence(ir.outputShape);
  } else if (trace && Array.isArray(trace.inputShape) && Array.isArray(trace.outputShape)) {
    support.inputShape = freezeShapeEvidence(trace.inputShape);
    support.outputShape = freezeShapeEvidence(trace.outputShape);
  }
  if (ir) support.ir = freezeTensorProgramIr(ir);
  if (kernelPlan) support.kernelPlan = freezePublicKernelPlan(kernelPlan);
  return support as TraceCompilerSupportDetails;
}

export function programCompileEvidenceFromCompiledSpec(spec: CompiledSequentialProgramSpec | null | undefined) {
  if (!spec) return null;
  if (spec.kind === "tiny-linear" && spec.layer && spec.trace) {
    const evidence: AnyRecord = {
      kind: "tiny-linear",
      nativePath: spec.nativePath,
      modelKind: spec.modelKind,
      layerCount: spec.layerCount,
      inputLen: spec.layer.inFeatures,
      outputLen: spec.layer.outFeatures,
      weightsLen: spec.layer.weight ? spec.layer.weight.length : 0,
      biasLen: spec.layer.bias ? spec.layer.bias.length : 0,
      ...compilerSignatureEvidence(spec.ir, spec.kernelPlan),
      trace: freezeSequentialTrace(spec.trace),
    };
    if (spec.ir) evidence.ir = freezeTensorProgramIr(spec.ir);
    if (spec.kernelPlan) evidence.kernelPlan = freezePublicKernelPlan(spec.kernelPlan);
    return completeProgramCompileEvidence(Object.freeze(evidence));
  }
  if (spec.kind !== "module" || !spec.trace || !spec.ir || !spec.kernelPlan) return null;
  return completeProgramCompileEvidence(Object.freeze({
    kind: "module",
    nativePath: spec.nativePath,
    modelKind: spec.modelKind,
    layerCount: spec.layerCount,
    ...compilerSignatureEvidence(spec.ir, spec.kernelPlan),
    trace: freezeSequentialTrace(spec.trace),
    ir: freezeTensorProgramIr(spec.ir),
    kernelPlan: freezePublicKernelPlan(spec.kernelPlan),
  }));
}

export function moduleProgramCompileArtifactsFromCompiledSpec(spec: unknown) {
  const compiledSpec = compiledSequentialProgramSpec(spec);
  if (!compiledSpec || compiledSpec.kind !== "module") {
    throw new Error("module Program compile requires a compiled module spec with a kernelPlan");
  }
  const desc = moduleProgramDescFromCompiledSpec(compiledSpec);
  return Object.freeze({
    desc,
    packed: packModuleProgramDesc(desc),
    evidence: programCompileEvidenceFromCompiledSpec(compiledSpec),
  });
}

export function programTraceFromCompileEvidence(evidence: ProgramCompileEvidenceInput | null | undefined): ModuleProgramTrace | null {
  return evidence && evidence.trace ? evidence.trace : null;
}

export function programTensorProgramIrFromCompileEvidence(evidence: ProgramCompileEvidenceInput | null | undefined): ModuleTensorProgramIr | null {
  return evidence && evidence.ir ? evidence.ir : null;
}

export function programKernelPlanFromCompileEvidence(evidence: ProgramCompileEvidenceInput | null | undefined): ModuleKernelPlan | null {
  return evidence && evidence.kernelPlan ? evidence.kernelPlan : null;
}

function traceCompilerFallbackDiagnostic(artifacts: TraceCompilerArtifacts | null | undefined): CompileDiagnostic {
  return (artifacts && artifacts.diagnostic) ??
    compileDiagnostic("unsupported-graph", "Sequential graph is outside the native module Program compiler subset", {
      stage: "kernelizer",
    });
}

export function sequentialUnsupportedSupportDetails(normalizedLayers: readonly unknown[], trace: TraceCompilerTraceInput, diagnostic: CompileDiagnostic | null, artifacts: TraceCompilerArtifacts | null = null): TraceCompilerSupportDetails {
  const compilerArtifacts = artifacts ?? { ir: null, kernelPlan: null, diagnostic: null };
  const support: AnyRecord = {
    composable: true,
    nativePath: "device-program",
    modelKind: "module",
    layerCount: normalizedLayers.length,
    diagnostics: freezeCompileDiagnostics([diagnostic ?? traceCompilerFallbackDiagnostic(compilerArtifacts)]),
    trace: freezeSequentialTrace(trace),
    ...compilerSignatureEvidence(compilerArtifacts.ir, compilerArtifacts.kernelPlan),
  };
  if (compilerArtifacts.ir) {
    support.inputShape = freezeShapeEvidence(compilerArtifacts.ir.inputShape);
    support.outputShape = freezeShapeEvidence(compilerArtifacts.ir.outputShape);
    support.inputLen = compilerArtifacts.ir.inputLen;
    support.outputLen = compilerArtifacts.ir.outputLen;
    support.ir = freezeTensorProgramIr(compilerArtifacts.ir);
  } else if (trace && Array.isArray(trace.inputShape) && Array.isArray(trace.outputShape)) {
    support.inputShape = freezeShapeEvidence(trace.inputShape);
    support.outputShape = freezeShapeEvidence(trace.outputShape);
  }
  if (compilerArtifacts.kernelPlan) support.kernelPlan = freezePublicKernelPlan(compilerArtifacts.kernelPlan);
  return support as TraceCompilerSupportDetails;
}

export function shapeMismatchAnalysis(normalizedLayers: readonly unknown[], inputShape: unknown, err: unknown) {
  const message = err && (err as Error).message ? String((err as Error).message) : String(err);
  const extra: AnyRecord = { stage: "trace" };
  if (inputShape) extra.inputShape = inputShape;
  const diagnostic = compileDiagnostic("shape-mismatch", message, extra);
  return {
    supported: false,
    reason: message,
    compiled: null,
    normalizedLayers,
    trace: null,
    support: {
      composable: true as const,
      nativePath: "device-program" as const,
      modelKind: "module" as const,
      layerCount: normalizedLayers.length,
      diagnostics: freezeCompileDiagnostics([diagnostic]) as readonly CompileDiagnostic[],
    },
  };
}

export {
  freezeTensorProgramIr,
  freezeKernelShapeConstraints,
  freezeKernelParameterLayout,
  freezeKernelPlan,
  freezePublicKernelPlan,
  kernelizeTensorProgramIr,
  programCompilerSignaturesFromCompileEvidence,
  compilerSignaturesFromCompilerEvidence,
  completeProgramCompileEvidence,
  isProgramCompileEvidence,
  requireProgramCompileEvidence,
  assertProgramCompileEvidence,
  assert_program_compile_evidence,
  matchesProgramCompileEvidenceSignature,
  matches_program_compile_evidence_signature,
  programCompileEvidenceSignature,
  programShapeConstraintsFromKernelPlan,
  kernelPlanInputShape,
  kernelPlanOutputShape,
  programMemoryLayoutFromKernelPlan,
  programParameterLayoutFromKernelPlan,
  tensorProgramIrSignature,
  moduleCompatibilityForProgramEvidence,
};
