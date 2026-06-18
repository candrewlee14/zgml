import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export function attachProgramCompileEvidence<T>(program: T, evidence: unknown): T {
  const maybeProgram = program as unknown as { _withCompileEvidence?: unknown };
  if (program && typeof maybeProgram._withCompileEvidence === "function") {
    return (maybeProgram as { _withCompileEvidence(evidence: unknown): T })._withCompileEvidence(evidence);
  }
  return program;
}

export const moduleProgramEvidenceManifest = Object.freeze({
  kind: "zgml-module-program-evidence",
  ...tsRuntimeManifestPolicy("src/ts/runtime/module_program_evidence.ts", "Module compile evidence -> Program"),
});
