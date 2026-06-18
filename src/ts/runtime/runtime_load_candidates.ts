import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type RuntimeLoadCandidateKind = "node-ffi-runtime" | "bun-ffi-runtime" | "shared-frontend-runtime";
function runtimeLoadCandidateManifestPolicy() {
  return tsRuntimeManifestPolicy("src/ts/runtime/runtime_load_candidates.ts", "TS package artifact -> concrete runtime load");
}
export type RuntimeLoadCandidateManifestPolicy = ReturnType<typeof runtimeLoadCandidateManifestPolicy>;
export type RuntimeLoadCandidateEvidence<Kind extends RuntimeLoadCandidateKind = RuntimeLoadCandidateKind> =
  RuntimeLoadCandidateManifestPolicy & Readonly<{
  kind: "zgml-runtime-load-candidate-evidence";
  runtimeKind: Kind;
  candidates: readonly string[];
  packageCandidates: readonly string[];
  sourceCheckoutFallbacks: readonly string[];
  packageArtifactFirst: boolean;
  hasSourceCheckoutFallback: boolean;
}>;

const nodeFfiRuntimePackageCandidates = Object.freeze([
  "./adapters/node_ffi_runtime.cjs",
  "./node_ffi_runtime.cjs",
  "./node_ffi_runtime.js",
  "../../dist/adapters/node_ffi_runtime.cjs",
  "../../../dist/adapters/node_ffi_runtime.cjs",
]);

const bunFfiRuntimePackageCandidates = Object.freeze([
  "./adapters/bun_ffi_runtime.cjs",
]);

const bunFfiRuntimeSourceCheckoutFallbacks = Object.freeze([
  "./bun_ffi_runtime.cjs",
  "./bun_ffi_runtime.ts",
  "../../src/ts/adapters/bun_ffi_runtime.ts",
  "../../../src/ts/adapters/bun_ffi_runtime.ts",
]);

const sharedFrontendRuntimePackageCandidates = Object.freeze([
  "../shared_frontend.cjs",
  "../shared_frontend.js",
  "../../dist/shared_frontend.cjs",
  "../../../dist/shared_frontend.cjs",
]);

const runtimeLoadPackageCandidatesByKind = Object.freeze({
  "node-ffi-runtime": nodeFfiRuntimePackageCandidates,
  "bun-ffi-runtime": bunFfiRuntimePackageCandidates,
  "shared-frontend-runtime": sharedFrontendRuntimePackageCandidates,
} satisfies Record<RuntimeLoadCandidateKind, readonly string[]>);

const runtimeLoadSourceCheckoutFallbacksByKind = Object.freeze({
  "node-ffi-runtime": Object.freeze([]),
  "bun-ffi-runtime": bunFfiRuntimeSourceCheckoutFallbacks,
  "shared-frontend-runtime": Object.freeze([]),
} satisfies Record<RuntimeLoadCandidateKind, readonly string[]>);

const runtimeLoadCandidatesByKind = Object.freeze(Object.fromEntries(
  (Object.keys(runtimeLoadPackageCandidatesByKind) as RuntimeLoadCandidateKind[]).map((kind) => [
    kind,
    Object.freeze([
      ...runtimeLoadPackageCandidatesByKind[kind],
      ...runtimeLoadSourceCheckoutFallbacksByKind[kind],
    ]),
  ]),
) as Record<RuntimeLoadCandidateKind, readonly string[]>);

export function runtimeLoadCandidates(kind: RuntimeLoadCandidateKind): readonly string[] {
  return runtimeLoadCandidatesByKind[kind];
}

export function runtimeLoadCandidateEvidence<const Kind extends RuntimeLoadCandidateKind>(kind: Kind): RuntimeLoadCandidateEvidence<Kind> {
  const candidates = runtimeLoadCandidates(kind);
  const packageCandidates = runtimeLoadPackageCandidatesByKind[kind];
  const sourceCheckoutFallbacks = runtimeLoadSourceCheckoutFallbacksByKind[kind];
  return Object.freeze({
    kind: "zgml-runtime-load-candidate-evidence",
    runtimeKind: kind,
    ...runtimeLoadCandidateManifestPolicy(),
    candidates,
    packageCandidates,
    sourceCheckoutFallbacks,
    packageArtifactFirst: packageCandidates.length > 0 && candidates[0] === packageCandidates[0],
    hasSourceCheckoutFallback: sourceCheckoutFallbacks.length > 0,
  });
}

export const runtimeLoadCandidatesManifest = Object.freeze({
  kind: "zgml-runtime-load-candidates",
  ...tsRuntimeManifestPolicy("src/ts/runtime/runtime_load_candidates.ts", "TS package artifact -> concrete runtime load"),
  packageArtifactFirst: true,
  runtimeKinds: Object.freeze(Object.keys(runtimeLoadCandidatesByKind) as RuntimeLoadCandidateKind[]),
  sourceCheckoutFallbackKinds: Object.freeze(
    (Object.keys(runtimeLoadSourceCheckoutFallbacksByKind) as RuntimeLoadCandidateKind[])
      .filter((kind) => runtimeLoadSourceCheckoutFallbacksByKind[kind].length > 0),
  ),
});
