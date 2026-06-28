import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  completeProgramCompileEvidence,
  type CompileEvidence,
} from "./compiler_signatures.js";
import type {
  ProgramRequirements,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ModuleProgramRequirementsSource = "zig-module-program";

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function safeInteger(value: unknown): number | null {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : null;
}

function assertNativeLengthAgreement(label: string, expected: unknown, actual: unknown) {
  const expectedLen = safeInteger(expected);
  const actualLen = safeInteger(actual);
  if (expectedLen === null || actualLen === null || expectedLen === actualLen) return;
  throw new Error(`module Program native requirements drift: ${label} TS=${expectedLen} Zig=${actualLen}`);
}

export function attachProgramCompileEvidence<T>(program: T, evidence: unknown): T {
  const maybeProgram = program as unknown as { _withCompileEvidence?: unknown };
  if (program && typeof maybeProgram._withCompileEvidence === "function") {
    return (maybeProgram as { _withCompileEvidence(evidence: unknown): T })._withCompileEvidence(evidence);
  }
  return program;
}

export function moduleProgramEvidenceWithNativeRequirements(
  evidence: CompileEvidence | Readonly<UnknownRecord> | null | undefined,
  requirements: ProgramRequirements,
  desc?: Readonly<UnknownRecord> | null,
) {
  if (!evidence || !isRecord(evidence) || evidence.kind !== "module") return evidence ?? null;
  const kernelPlan = isRecord(evidence.kernelPlan) ? evidence.kernelPlan : null;
  assertNativeLengthAgreement("inputLen", desc?.inputLen ?? kernelPlan?.inputLen, requirements.inputLen);
  assertNativeLengthAgreement("outputLen", desc?.outputLen ?? kernelPlan?.outputLen, requirements.outputLen);
  assertNativeLengthAgreement("weightsLen", desc?.weightsLen ?? kernelPlan?.weightsLen, requirements.weightsLen);
  assertNativeLengthAgreement("biasLen", desc?.biasLen ?? kernelPlan?.biasLen, requirements.biasLen);
  return completeProgramCompileEvidence(Object.freeze({
    ...evidence,
    inputLen: requirements.inputLen,
    outputLen: requirements.outputLen,
    weightsLen: requirements.weightsLen,
    biasLen: requirements.biasLen,
    nativeRequirements: requirements,
    nativeRequirementsSignature: requirements.signature,
    nativeRequirementsSource: "zig-module-program" satisfies ModuleProgramRequirementsSource,
  }));
}

export const moduleProgramEvidenceManifest = Object.freeze({
  kind: "zgml-module-program-evidence",
  ...tsRuntimeManifestPolicy("src/ts/runtime/module_program_evidence.ts", "Module compile evidence -> Program"),
});
