import {
  type ConcreteRuntimeHostManifest,
  type ConcreteRuntimeRequire,
  type ConcreteRuntimeHostOptions,
  type ConcreteRuntimeHostPolicy,
  type ConcreteRuntimeHostSurface,
  type NativeRuntime,
  createConcreteRuntimeHostSurface,
} from "./concrete_runtime_loader.js";
export type { NativeRuntime } from "./concrete_runtime_loader.js";

export type BunConcreteRuntimeRequire = ConcreteRuntimeRequire;
export type BunConcreteRuntimeManifest = ConcreteRuntimeHostManifest<"bun-ffi-runtime">;

export type BunConcreteRuntimeOptions = ConcreteRuntimeHostOptions & Readonly<{
  requireModule: BunConcreteRuntimeRequire;
}>;

export const bunConcreteRuntimePolicy = Object.freeze({
  runtime: "Bun",
  bridge: "zgml TS Bun bridge",
  runtimeKind: "bun-ffi-runtime",
  manifestKind: "zgml-bun-concrete-runtime-loader",
  policyOwner: "src/ts/adapters/bun_concrete_runtime.ts",
} satisfies ConcreteRuntimeHostPolicy<"bun-ffi-runtime">);

const bunConcreteRuntimeSurface: ConcreteRuntimeHostSurface<"bun-ffi-runtime"> =
  createConcreteRuntimeHostSurface(bunConcreteRuntimePolicy);

export function bunConcreteRuntimeLoadCandidates(): readonly string[] {
  return bunConcreteRuntimeSurface.loadCandidates();
}

export function requireBunConcreteRuntimeByPolicy(options: BunConcreteRuntimeOptions): NativeRuntime {
  return bunConcreteRuntimeSurface.requireByPolicy(options);
}

export function assertBunNativeRuntimeContract(runtime: unknown): asserts runtime is NativeRuntime {
  bunConcreteRuntimeSurface.assertContract(runtime);
}

export const bunConcreteRuntimeManifest: BunConcreteRuntimeManifest = bunConcreteRuntimeSurface.manifest;
