"use strict";

import type {
  CompileAnalysis,
  ModuleKernelPlan,
  ModuleProgramTrace,
  ModuleTensorProgramIr,
} from "../public_api.js";
import type { CompileDiagnostic } from "./compile_diagnostics.js";

type UnknownRecord = Record<string, unknown>;

type CompileAnalysisTraceLike = Readonly<UnknownRecord & {
  kind?: unknown;
  layerCount?: unknown;
  opCount?: unknown;
  parameterCount?: unknown;
  parameterScalarCount?: unknown;
  shapeKnown?: unknown;
  inputShape?: unknown;
  outputShape?: unknown;
}>;

type CompileAnalysisArtifactsLike = Readonly<UnknownRecord & {
  ir?: { signature?: unknown } | null;
  kernelPlan?: { signature?: unknown } | null;
  diagnostic?: { code?: unknown } | null;
}>;

type CompileAnalysisLike = Readonly<UnknownRecord & {
  trace?: unknown;
  artifacts?: unknown;
}>;

export type CompileAnalysisArtifacts = Readonly<{
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  diagnostic: CompileDiagnostic | null;
}>;

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function traceRecord(value: unknown): CompileAnalysisTraceLike {
  return isRecord(value) ? value as CompileAnalysisTraceLike : {};
}

function artifactsRecord(value: unknown): CompileAnalysisArtifactsLike {
  return isRecord(value) ? value as CompileAnalysisArtifactsLike : {};
}

export function compileAnalysisSignature(analysis: CompileAnalysisLike) {
  const trace = traceRecord(analysis.trace);
  const artifacts = artifactsRecord(analysis.artifacts);
  return [
    "compile-analysis",
    `traceKind=${trace.kind ?? "null"}`,
    `layers=${Number.isSafeInteger(trace.layerCount) ? trace.layerCount : "null"}`,
    `ops=${Number.isSafeInteger(trace.opCount) ? trace.opCount : "null"}`,
    `params=${Number.isSafeInteger(trace.parameterCount) ? trace.parameterCount : "null"}`,
    `paramScalars=${Number.isSafeInteger(trace.parameterScalarCount) ? trace.parameterScalarCount : "null"}`,
    `shapeKnown=${trace.shapeKnown === true ? 1 : 0}`,
    `input=${Array.isArray(trace.inputShape) ? trace.inputShape.join("x") : "null"}`,
    `output=${Array.isArray(trace.outputShape) ? trace.outputShape.join("x") : trace.outputShape === null ? "null" : "unknown"}`,
    `ir=${artifacts.ir?.signature ?? "null"}`,
    `kernel=${artifacts.kernelPlan?.signature ?? "null"}`,
    `diagnostic=${artifacts.diagnostic?.code ?? "null"}`,
  ].join("|");
}

export function compileAnalysisEvidence(
  trace: ModuleProgramTrace,
  artifacts: CompileAnalysisArtifacts,
): CompileAnalysis {
  const publicArtifacts = Object.freeze({
    ir: artifacts.ir,
    kernelPlan: artifacts.kernelPlan,
    diagnostic: artifacts.diagnostic,
  });
  const analysis = {
    kind: "zgml.compile.analysis" as const,
    signature: "",
    trace,
    artifacts: publicArtifacts,
  };
  return Object.freeze({
    kind: analysis.kind,
    signature: compileAnalysisSignature(analysis),
    trace: analysis.trace,
    artifacts: analysis.artifacts,
  });
}

export function isCompileAnalysis(analysis: unknown): analysis is CompileAnalysis {
  if (!isRecord(analysis) || !Object.isFrozen(analysis)) return false;
  if (analysis.kind !== "zgml.compile.analysis" || typeof analysis.signature !== "string") return false;
  if (!isRecord(analysis.trace) || !Object.isFrozen(analysis.trace)) return false;
  if (!isRecord(analysis.artifacts) || !Object.isFrozen(analysis.artifacts)) return false;
  const artifacts = analysis.artifacts;
  if (artifacts.ir !== null && !isRecord(artifacts.ir)) return false;
  if (artifacts.kernelPlan !== null && !isRecord(artifacts.kernelPlan)) return false;
  if (artifacts.diagnostic !== null && !isRecord(artifacts.diagnostic)) return false;
  return analysis.signature === compileAnalysisSignature(analysis);
}

export function requireCompileAnalysis(analysis: unknown): CompileAnalysis {
  if (isCompileAnalysis(analysis)) return analysis;
  throw new Error("expected frozen CompileAnalysis evidence");
}

export const assertCompileAnalysis = requireCompileAnalysis;
export const assert_compile_analysis = requireCompileAnalysis;

export function matchesCompileAnalysisSignature(analysis: unknown, signature: unknown) {
  return isCompileAnalysis(analysis) && typeof signature === "string" && analysis.signature === signature;
}

export const matches_compile_analysis_signature = matchesCompileAnalysisSignature;
