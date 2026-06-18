"use strict";

import {
  kernelPlanSignature as kernelPlanSignatureForPlan,
  memoryLayoutSignature,
  parameterLayoutSignature,
  bufferLayoutSignature,
} from "./kernel_plan.js";
import {
  tensorProgramIrSignature,
} from "./tensor_program_ir.js";

export type CompilerSignatures = Readonly<{
  ir: string;
  kernelPlan: string;
  memoryLayout: string;
  parameterLayout: string;
  bufferLayout: string;
}>;

export type PartialCompilerSignatures = Partial<CompilerSignatures>;

type MutablePartialCompilerSignatures = {
  ir?: string;
  kernelPlan?: string;
  memoryLayout?: string;
  parameterLayout?: string;
  bufferLayout?: string;
};

type AnyRecord = Record<string, any>;

export type KernelPlan = {
  readonly memoryLayout?: AnyRecord;
  readonly parameterLayout?: AnyRecord;
  readonly bufferLayout?: AnyRecord;
  readonly [key: string]: unknown;
};

export type CompileEvidence = {
  readonly kind?: unknown;
  readonly nativePath?: unknown;
  readonly modelKind?: unknown;
  readonly layerCount?: unknown;
  readonly inputLen?: unknown;
  readonly outputLen?: unknown;
  readonly weightsLen?: unknown;
  readonly biasLen?: unknown;
  readonly signature?: unknown;
  readonly ir?: unknown;
  readonly kernelPlan?: KernelPlan;
  readonly compilerSignatures?: PartialCompilerSignatures | null;
  readonly irSignature?: string;
  readonly kernelPlanSignature?: string;
  readonly memoryLayoutSignature?: string;
  readonly parameterLayoutSignature?: string;
  readonly bufferLayoutSignature?: string;
  readonly [key: string]: unknown;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

export function compilerSignatureEvidence(ir: unknown, kernelPlan: KernelPlan | null | undefined) {
  const evidence: Record<string, unknown> = {};
  const compilerSignatures: MutablePartialCompilerSignatures = {};
  if (ir) {
    compilerSignatures.ir = tensorProgramIrSignature(ir);
    evidence.irSignature = compilerSignatures.ir;
  }
  if (kernelPlan) {
    compilerSignatures.kernelPlan = kernelPlanSignatureForPlan(kernelPlan);
    compilerSignatures.memoryLayout = memoryLayoutSignature(kernelPlan.memoryLayout);
    compilerSignatures.parameterLayout = parameterLayoutSignature(kernelPlan.parameterLayout);
    compilerSignatures.bufferLayout = bufferLayoutSignature(kernelPlan.bufferLayout);
    evidence.kernelPlanSignature = compilerSignatures.kernelPlan;
    evidence.memoryLayoutSignature = compilerSignatures.memoryLayout;
    evidence.parameterLayoutSignature = compilerSignatures.parameterLayout;
    evidence.bufferLayoutSignature = compilerSignatures.bufferLayout;
  }
  if (Object.keys(compilerSignatures).length > 0) {
    evidence.compilerSignatures = Object.freeze(compilerSignatures);
  }
  return evidence;
}

function isCompleteCompilerSignatures(signatures: PartialCompilerSignatures | null | undefined): signatures is CompilerSignatures {
  return !!(
    signatures &&
    typeof signatures.ir === "string" &&
    signatures.ir.length > 0 &&
    typeof signatures.kernelPlan === "string" &&
    signatures.kernelPlan.length > 0 &&
    typeof signatures.memoryLayout === "string" &&
    signatures.memoryLayout.length > 0 &&
    typeof signatures.parameterLayout === "string" &&
    typeof signatures.bufferLayout === "string" &&
    signatures.bufferLayout.length > 0
  );
}

function freezeCompilerSignatures(signatures: CompilerSignatures): CompilerSignatures {
  return Object.freeze({
    ir: signatures.ir,
    kernelPlan: signatures.kernelPlan,
    memoryLayout: signatures.memoryLayout,
    parameterLayout: signatures.parameterLayout,
    bufferLayout: signatures.bufferLayout,
  });
}

export function compilerSignaturesFromCompilerEvidence(evidence: CompileEvidence | null | undefined): Readonly<PartialCompilerSignatures> | null {
  if (!evidence) return null;
  const existing = evidence.compilerSignatures || {};
  if (isCompleteCompilerSignatures(existing)) {
    return Object.isFrozen(existing) ? existing : freezeCompilerSignatures(existing);
  }
  const kernelPlan = evidence.kernelPlan;
  const signatures: MutablePartialCompilerSignatures = {};
  const irSignature = existing.ir ?? evidence.irSignature ?? (evidence.ir ? tensorProgramIrSignature(evidence.ir) : undefined);
  const kernelSignature = existing.kernelPlan ?? evidence.kernelPlanSignature ?? (kernelPlan ? kernelPlanSignatureForPlan(kernelPlan) : undefined);
  const memorySignature = existing.memoryLayout ?? evidence.memoryLayoutSignature ?? (kernelPlan ? memoryLayoutSignature(kernelPlan.memoryLayout) : undefined);
  const parameterSignature = existing.parameterLayout ?? evidence.parameterLayoutSignature ?? (kernelPlan ? parameterLayoutSignature(kernelPlan.parameterLayout) : undefined);
  const bufferSignature = existing.bufferLayout ?? evidence.bufferLayoutSignature ?? (kernelPlan ? bufferLayoutSignature(kernelPlan.bufferLayout) : undefined);
  if (irSignature !== undefined) signatures.ir = irSignature;
  if (kernelSignature !== undefined) signatures.kernelPlan = kernelSignature;
  if (memorySignature !== undefined) signatures.memoryLayout = memorySignature;
  if (parameterSignature !== undefined) signatures.parameterLayout = parameterSignature;
  if (bufferSignature !== undefined) signatures.bufferLayout = bufferSignature;
  return Object.keys(signatures).length > 0 ? Object.freeze(signatures) : null;
}

export function compilerEvidenceSignature(source: CompileEvidence | null | undefined, signatureField: string): string {
  const signatures = compilerSignaturesFromCompilerEvidence(source);
  const value = signatures ? (signatures as Record<string, unknown>)[signatureField] : undefined;
  return typeof value === "string" ? value : "";
}

export function programCompilerSignaturesFromCompileEvidence(evidence: CompileEvidence | null | undefined): CompilerSignatures | null {
  if (!evidence) return null;
  const signatures = compilerSignaturesFromCompilerEvidence(evidence);
  if (!isCompleteCompilerSignatures(signatures)) return null;
  return signatures;
}

export function programCompileEvidenceSignature(evidence: CompileEvidence | null | undefined): string {
  const signatures = programCompilerSignaturesFromCompileEvidence(evidence);
  return [
    "program-compile-evidence",
    `kind=${evidence?.kind ?? "null"}`,
    `nativePath=${evidence?.nativePath ?? "null"}`,
    `modelKind=${evidence?.modelKind ?? "null"}`,
    `layers=${isNonNegativeSafeInteger(evidence?.layerCount) ? evidence.layerCount : "null"}`,
    `input=${isNonNegativeSafeInteger(evidence?.inputLen) ? evidence.inputLen : "null"}`,
    `output=${isNonNegativeSafeInteger(evidence?.outputLen) ? evidence.outputLen : "null"}`,
    `weights=${isNonNegativeSafeInteger(evidence?.weightsLen) ? evidence.weightsLen : "null"}`,
    `bias=${isNonNegativeSafeInteger(evidence?.biasLen) ? evidence.biasLen : "null"}`,
    `ir=${signatures?.ir ?? "null"}`,
    `kernel=${signatures?.kernelPlan ?? "null"}`,
    `memory=${signatures?.memoryLayout ?? "null"}`,
    `params=${signatures?.parameterLayout ?? "null"}`,
    `buffers=${signatures?.bufferLayout ?? "null"}`,
  ].join("|");
}

function isProgramCompileEvidenceShape(evidence: CompileEvidence | null | undefined) {
  if (!evidence || !isRecord(evidence)) return false;
  if (evidence.kind === "tiny-linear") {
    return (
      evidence.nativePath === "tiny-linear" &&
      evidence.modelKind === "tiny-linear" &&
      isNonNegativeSafeInteger(evidence.layerCount) &&
      isNonNegativeSafeInteger(evidence.inputLen) &&
      isNonNegativeSafeInteger(evidence.outputLen) &&
      isNonNegativeSafeInteger(evidence.weightsLen) &&
      isNonNegativeSafeInteger(evidence.biasLen)
    );
  }
  if (evidence.kind === "module") {
    return (
      evidence.nativePath === "device-program" &&
      evidence.modelKind === "module" &&
      isNonNegativeSafeInteger(evidence.layerCount)
    );
  }
  return false;
}

export function isProgramCompileEvidence(evidence: unknown): evidence is import("../public_api.js").ProgramCompileEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  const record = evidence as CompileEvidence;
  if (!isProgramCompileEvidenceShape(record)) return false;
  const signatures = programCompilerSignaturesFromCompileEvidence(record);
  if (!signatures || !Object.isFrozen(signatures)) return false;
  if (record.compilerSignatures !== signatures) return false;
  if (record.irSignature !== signatures.ir) return false;
  if (record.kernelPlanSignature !== signatures.kernelPlan) return false;
  if (record.memoryLayoutSignature !== signatures.memoryLayout) return false;
  if (record.parameterLayoutSignature !== signatures.parameterLayout) return false;
  if (record.bufferLayoutSignature !== signatures.bufferLayout) return false;
  if (typeof record.signature !== "string") return false;
  if (!isRecord(record.trace) || !Object.isFrozen(record.trace)) return false;
  if (!isRecord(record.ir) || !Object.isFrozen(record.ir)) return false;
  if (!isRecord(record.kernelPlan) || !Object.isFrozen(record.kernelPlan)) return false;
  return record.signature === programCompileEvidenceSignature(record);
}

export function requireProgramCompileEvidence(evidence: unknown): import("../public_api.js").ProgramCompileEvidence {
  if (isProgramCompileEvidence(evidence)) return evidence;
  throw new Error("expected frozen ProgramCompileEvidence");
}

export const assertProgramCompileEvidence = requireProgramCompileEvidence;
export const assert_program_compile_evidence = requireProgramCompileEvidence;

export function matchesProgramCompileEvidenceSignature(evidence: unknown, signature: unknown) {
  return isProgramCompileEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_program_compile_evidence_signature = matchesProgramCompileEvidenceSignature;

export function completeProgramCompileEvidence(evidence: CompileEvidence | null | undefined): Readonly<CompileEvidence> | null {
  if (!evidence) return evidence ?? null;
  const signatures = programCompilerSignaturesFromCompileEvidence(evidence);
  if (!signatures) return evidence;
  const signature = isProgramCompileEvidenceShape(evidence) ? programCompileEvidenceSignature(evidence) : undefined;
  const isCanonical =
    Object.isFrozen(evidence) &&
    evidence.compilerSignatures === signatures &&
    evidence.irSignature === signatures.ir &&
    evidence.kernelPlanSignature === signatures.kernelPlan &&
    evidence.memoryLayoutSignature === signatures.memoryLayout &&
    evidence.parameterLayoutSignature === signatures.parameterLayout &&
    evidence.bufferLayoutSignature === signatures.bufferLayout &&
    (signature === undefined || evidence.signature === signature);
  if (isCanonical) return evidence;
  const completed: Record<string, unknown> = {
    ...evidence,
    irSignature: signatures.ir,
    kernelPlanSignature: signatures.kernelPlan,
    memoryLayoutSignature: signatures.memoryLayout,
    parameterLayoutSignature: signatures.parameterLayout,
    bufferLayoutSignature: signatures.bufferLayout,
    compilerSignatures: signatures,
  };
  if (signature !== undefined) completed.signature = signature;
  return Object.freeze(completed);
}
