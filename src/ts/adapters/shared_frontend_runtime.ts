import type * as SharedFrontendModule from "../shared_frontend.js";
import { nativeAdapterManifestPolicy } from "./native.js";
import { runtimeLoadCandidateEvidence, runtimeLoadCandidates } from "../runtime/runtime_load_candidates.js";

type AnyRecord = Record<string, unknown>;

export type SharedFrontendRequire = (id: string) => unknown;
export type SharedFrontendRuntime = typeof SharedFrontendModule;

export type SharedFrontendRuntimeLoadPolicy = Readonly<{
  requireModule: SharedFrontendRequire;
  origin: string;
  bridge: string;
  candidates: readonly string[];
}>;

export type SharedFrontendRuntimeHostPolicy<Runtime extends "Node" | "Bun" = "Node" | "Bun"> = Readonly<{
  runtime: Runtime;
  bridge: `zgml ${Runtime} FFI runtime`;
  policyOwner: `src/ts/adapters/${string}_shared_frontend_runtime.ts`;
}>;

export type SharedFrontendRuntimeHostOptions = Readonly<{
  requireModule: SharedFrontendRequire;
  origin: string;
}>;

export function sharedFrontendRuntimeLoadCandidates(): readonly string[] {
  return runtimeLoadCandidates("shared-frontend-runtime");
}

export const sharedFrontendRuntimeContract = Object.freeze({
  manifestKind: "zgml-shared-frontend",
  source: "ts",
  policyOwner: "src/ts/shared_frontend.ts",
  legacyCjsBridge: false,
});

export function assertSharedFrontendRuntimeContract(runtime: unknown, bridge: string): asserts runtime is SharedFrontendRuntime {
  if (runtime === null || typeof runtime !== "object") {
    throw new Error(`${bridge} loaded a shared frontend runtime that is not an object`);
  }
  const manifest = (runtime as AnyRecord).sharedFrontendManifest as AnyRecord | undefined;
  if (
    manifest?.kind !== sharedFrontendRuntimeContract.manifestKind ||
    manifest?.source !== sharedFrontendRuntimeContract.source ||
    manifest?.policyOwner !== sharedFrontendRuntimeContract.policyOwner ||
    manifest?.legacyCjsBridge !== sharedFrontendRuntimeContract.legacyCjsBridge
  ) {
    throw new Error(`${bridge} loaded a shared frontend runtime that does not satisfy the TS shared frontend manifest contract`);
  }
}

export function requireSharedFrontendRuntimeByPolicy(policy: SharedFrontendRuntimeLoadPolicy): SharedFrontendRuntime {
  const failures: string[] = [];
  for (const candidate of policy.candidates) {
    try {
      const runtime = policy.requireModule(candidate);
      assertSharedFrontendRuntimeContract(runtime, policy.bridge);
      return runtime;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`${candidate}: ${message}`);
    }
  }
  throw new Error(
    `${policy.bridge} could not load shared frontend from ${policy.origin}: ${failures.join("; ")}`,
  );
}

export function requireSharedFrontendRuntimeHostByPolicy<const Runtime extends "Node" | "Bun">(
  host: SharedFrontendRuntimeHostPolicy<Runtime>,
  options: SharedFrontendRuntimeHostOptions,
): SharedFrontendRuntime {
  return requireSharedFrontendRuntimeByPolicy({
    requireModule: options.requireModule,
    origin: options.origin,
    bridge: host.bridge,
    candidates: sharedFrontendRuntimeLoadCandidates(),
  });
}

export const sharedFrontendRuntimeManifest = Object.freeze({
  kind: "zgml-shared-frontend-runtime-loader",
  ...nativeAdapterManifestPolicy("src/ts/adapters/shared_frontend_runtime.ts"),
  runtimeLoad: runtimeLoadCandidateEvidence("shared-frontend-runtime"),
  checkedRuntimeManifest: sharedFrontendRuntimeContract,
  legacyCjsFallback: false,
});
