"use strict";

import {
  missingExports,
  nativeApiContractSignature,
  requiredNativeApiExports,
  type RequiredNativeApiExport,
} from "../runtime/native_api_contract.js";
import { frontendManifest } from "../frontend_manifest.js";
import type * as PublicApi from "../public_api.js";
import {
  runtimeLoadCandidateEvidence,
  runtimeLoadCandidates,
  type RuntimeLoadCandidateEvidence,
  type RuntimeLoadCandidateKind,
} from "../runtime/runtime_load_candidates.js";
import {
  nativeAdapterManifestPolicy,
  type NativeAdapterManifestPolicy,
} from "./native.js";

export type ConcreteRuntimeRequire = (id: string) => unknown;
export type NativeRuntime = Pick<typeof PublicApi, RequiredNativeApiExport>;

export type ConcreteRuntimeLoadPolicy = Readonly<{
  runtime: string;
  bridge: string;
  origin: string;
  requireModule: ConcreteRuntimeRequire;
  candidates: readonly string[];
}>;

export type ConcreteRuntimeHostPolicy<Kind extends RuntimeLoadCandidateKind = RuntimeLoadCandidateKind> = Readonly<{
  runtime: string;
  bridge: string;
  runtimeKind: Kind;
  manifestKind: `zgml-${string}-concrete-runtime-loader`;
  policyOwner: `src/ts/adapters/${string}_concrete_runtime.ts`;
}>;

export type ConcreteRuntimeHostOptions = Readonly<{
  requireModule: ConcreteRuntimeRequire;
  origin: string;
}>;

export type ConcreteRuntimeLoaderEvidence = NativeAdapterManifestPolicy<"src/ts/adapters/concrete_runtime_loader.ts"> & Readonly<{
  kind: "zgml-concrete-runtime-loader-evidence";
  productSemanticsOwner: typeof frontendManifest.productSemanticsOwner;
  productLanguage: typeof frontendManifest.productLanguage;
  nativeAlignment: typeof frontendManifest.nativeAlignment;
  handwrittenFrontendMirrors: typeof frontendManifest.handwrittenFrontendMirrors;
  checkedContract: string;
  runtimeType: "NativeRuntime";
}>;

export type ConcreteRuntimeHostManifest<Kind extends RuntimeLoadCandidateKind = RuntimeLoadCandidateKind> =
  NativeAdapterManifestPolicy<ConcreteRuntimeHostPolicy<Kind>["policyOwner"]> & Readonly<{
  kind: ConcreteRuntimeHostPolicy<Kind>["manifestKind"];
  runtimeLoad: RuntimeLoadCandidateEvidence<Kind>;
  loader: ConcreteRuntimeLoaderEvidence;
  checkedContract: string;
}>;

export type ConcreteRuntimeHostSurface<Kind extends RuntimeLoadCandidateKind = RuntimeLoadCandidateKind> = Readonly<{
  policy: ConcreteRuntimeHostPolicy<Kind>;
  loadCandidates(): readonly string[];
  requireByPolicy(options: ConcreteRuntimeHostOptions): NativeRuntime;
  assertContract(runtime: unknown): asserts runtime is NativeRuntime;
  manifest: ConcreteRuntimeHostManifest<Kind>;
}>;

export function assertConcreteRuntimeContract(runtime: unknown, bridge: string): asserts runtime is NativeRuntime {
  if (runtime === null || typeof runtime !== "object") {
    throw new Error(`${bridge} loaded a native runtime that is not an object for ${nativeApiContractSignature()}`);
  }
  const missingNativeExports = missingExports(runtime as NativeRuntime, requiredNativeApiExports);
  if (missingNativeExports.length !== 0) {
    throw new Error(
      `${bridge} loaded a native runtime that does not satisfy ${nativeApiContractSignature()}: ${missingNativeExports.join(", ")}`,
    );
  }
}

export function concreteRuntimeLoadCandidates(kind: RuntimeLoadCandidateKind): readonly string[] {
  return runtimeLoadCandidates(kind);
}

export function requireConcreteRuntimeByPolicy(policy: ConcreteRuntimeLoadPolicy): NativeRuntime {
  const failures: string[] = [];
  for (const candidate of policy.candidates) {
    try {
      const runtime = policy.requireModule(candidate);
      assertConcreteRuntimeContract(runtime, policy.bridge);
      return runtime;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`${candidate}: ${message}`);
    }
  }
  throw new Error(`${policy.bridge} could not load concrete native runtime from ${policy.origin}: ${failures.join("; ")}`);
}

export function requireConcreteRuntimeHostByPolicy<const Kind extends RuntimeLoadCandidateKind>(
  host: ConcreteRuntimeHostPolicy<Kind>,
  options: ConcreteRuntimeHostOptions,
): NativeRuntime {
  return requireConcreteRuntimeByPolicy({
    runtime: host.runtime,
    bridge: host.bridge,
    origin: options.origin,
    requireModule: options.requireModule,
    candidates: concreteRuntimeLoadCandidates(host.runtimeKind),
  });
}

export function assertConcreteRuntimeHostContract<const Kind extends RuntimeLoadCandidateKind>(
  host: ConcreteRuntimeHostPolicy<Kind>,
  runtime: unknown,
): asserts runtime is NativeRuntime {
  assertConcreteRuntimeContract(runtime, host.bridge);
}

export function concreteRuntimeLoaderEvidence(): ConcreteRuntimeLoaderEvidence {
  return Object.freeze({
    kind: "zgml-concrete-runtime-loader-evidence",
    ...nativeAdapterManifestPolicy("src/ts/adapters/concrete_runtime_loader.ts"),
    productSemanticsOwner: frontendManifest.productSemanticsOwner,
    productLanguage: frontendManifest.productLanguage,
    nativeAlignment: frontendManifest.nativeAlignment,
    handwrittenFrontendMirrors: frontendManifest.handwrittenFrontendMirrors,
    checkedContract: nativeApiContractSignature(),
    runtimeType: "NativeRuntime",
  });
}

export function concreteRuntimeHostManifest<const Kind extends RuntimeLoadCandidateKind>(
  host: ConcreteRuntimeHostPolicy<Kind>,
): ConcreteRuntimeHostManifest<Kind> {
  return Object.freeze({
    kind: host.manifestKind,
    ...nativeAdapterManifestPolicy(host.policyOwner),
    runtimeLoad: runtimeLoadCandidateEvidence(host.runtimeKind),
    loader: concreteRuntimeLoaderEvidence(),
    checkedContract: nativeApiContractSignature(),
  });
}

export function createConcreteRuntimeHostSurface<const Kind extends RuntimeLoadCandidateKind>(
  host: ConcreteRuntimeHostPolicy<Kind>,
): ConcreteRuntimeHostSurface<Kind> {
  return Object.freeze({
    policy: host,
    loadCandidates(): readonly string[] {
      return concreteRuntimeLoadCandidates(host.runtimeKind);
    },
    requireByPolicy(options: ConcreteRuntimeHostOptions): NativeRuntime {
      return requireConcreteRuntimeHostByPolicy(host, options);
    },
    assertContract(runtime: unknown): asserts runtime is NativeRuntime {
      assertConcreteRuntimeHostContract(host, runtime);
    },
    manifest: concreteRuntimeHostManifest(host),
  });
}

export const concreteRuntimeLoaderManifest = Object.freeze({
  kind: "zgml-concrete-runtime-loader",
  ...nativeAdapterManifestPolicy("src/ts/adapters/concrete_runtime_loader.ts"),
  evidence: concreteRuntimeLoaderEvidence(),
  checkedContract: nativeApiContractSignature(),
});
