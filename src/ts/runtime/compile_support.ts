"use strict";

type UnknownRecord = Record<string, unknown>;

type CompileSupportDiagnosticLike = Readonly<{
  message?: unknown;
}>;

type CompileSupportEvidenceLike = Readonly<UnknownRecord & {
  reason?: unknown;
  diagnostic?: CompileSupportDiagnosticLike | null;
  diagnostics?: readonly CompileSupportDiagnosticLike[];
}>;

export type CompileSupportRejectionOptions = Readonly<{
  fallbackReason?: string;
}>;

export type RawSequentialLayerListCompileDiagnostic = Readonly<{
  kind: "zgml.compile.raw-sequential-layer-list";
  supported: false;
  reason: "raw Sequential layer lists support compile evidence only";
  evidencePath: "compile.trace|compile.compileSupport|compile.requireCompileSupport|compile.explain|compile.requireCompilePlan";
  programPath: "nn.compile(layers)";
}>;

export function rawSequentialLayerListCompileDiagnostic(): RawSequentialLayerListCompileDiagnostic {
  return Object.freeze({
    kind: "zgml.compile.raw-sequential-layer-list",
    supported: false,
    reason: "raw Sequential layer lists support compile evidence only",
    evidencePath: "compile.trace|compile.compileSupport|compile.requireCompileSupport|compile.explain|compile.requireCompilePlan",
    programPath: "nn.compile(layers)",
  });
}

export function compileSupportRejectionReason(
  support: unknown,
  options: CompileSupportRejectionOptions = {},
) {
  if (!support || typeof support !== "object") {
    return options.fallbackReason ?? "compileSupport did not return structured evidence";
  }
  const record = support as CompileSupportEvidenceLike;
  if (typeof record.reason === "string" && record.reason.length > 0) return record.reason;
  if (record.diagnostic && typeof record.diagnostic.message === "string" && record.diagnostic.message.length > 0) {
    return record.diagnostic.message;
  }
  if (Array.isArray(record.diagnostics)) {
    const diagnostic = record.diagnostics.find((entry) => entry && typeof entry.message === "string" && entry.message.length > 0);
    if (diagnostic) return diagnostic.message;
  }
  return options.fallbackReason ?? "compile target is unsupported";
}
