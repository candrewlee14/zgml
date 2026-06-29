"use strict";

type UnknownRecord = Record<string, unknown>;

export type CompileDiagnosticExtra = Readonly<UnknownRecord & {
  stage?: string;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
}>;

export type CompileDiagnostic = Readonly<UnknownRecord & {
  code: string;
  message: string;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
}>;

export type CompileDiagnosticOp = Readonly<UnknownRecord & {
  index?: unknown;
  op?: unknown;
  path?: unknown;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
}>;

type MutableCompileDiagnostic = UnknownRecord & {
  code: string;
  message: string;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
};

function diagnosticRecord(value: unknown): UnknownRecord {
  return value && typeof value === "object" ? value as UnknownRecord : {};
}

export function compileDiagnostic(code: string, message: string, extra: CompileDiagnosticExtra = {}): CompileDiagnostic {
  const diagnostic: MutableCompileDiagnostic = {
    code,
    message,
    ...extra,
  };
  if (Array.isArray(diagnostic.inputShape)) diagnostic.inputShape = Object.freeze(diagnostic.inputShape.slice());
  if (Array.isArray(diagnostic.outputShape)) diagnostic.outputShape = Object.freeze(diagnostic.outputShape.slice());
  return Object.freeze(diagnostic) as CompileDiagnostic;
}

export function freezeCompileDiagnostics(diagnostics: unknown) {
  if (!Array.isArray(diagnostics)) return diagnostics;
  return Object.freeze(diagnostics.map((diagnostic: unknown) => {
    if (Object.isFrozen(diagnostic)) return diagnostic;
    const record = diagnosticRecord(diagnostic);
    return compileDiagnostic(String(record.code ?? ""), String(record.message ?? ""), record);
  }));
}

export function diagnosticForCompilerOp(op: CompileDiagnosticOp, code: string, message: string, extra: CompileDiagnosticExtra = {}) {
  return compileDiagnostic(code, message, {
    stage: extra.stage ?? "ir",
    opIndex: op.index,
    op: op.op,
    path: op.path,
    inputShape: op.inputShape,
    outputShape: op.outputShape,
    ...extra,
  });
}

export function diagnosticForKernelizerOp(op: CompileDiagnosticOp, code: string, message: string, extra: CompileDiagnosticExtra = {}) {
  return compileDiagnostic(code, message, {
    stage: extra.stage ?? "kernelizer",
    opIndex: op.index,
    op: op.op,
    path: op.path,
    inputShape: op.inputShape,
    outputShape: op.outputShape,
    ...extra,
  });
}
