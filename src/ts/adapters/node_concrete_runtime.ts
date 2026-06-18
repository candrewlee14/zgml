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

export type NodeConcreteRuntimeRequire = ConcreteRuntimeRequire;
export type NodeConcreteRuntimeManifest = ConcreteRuntimeHostManifest<"node-ffi-runtime">;

export type NodeConcreteRuntimeOptions = ConcreteRuntimeHostOptions & Readonly<{
  requireModule: NodeConcreteRuntimeRequire;
}>;

export const nodeConcreteRuntimePolicy = Object.freeze({
  runtime: "Node",
  bridge: "zgml TS Node bridge",
  runtimeKind: "node-ffi-runtime",
  manifestKind: "zgml-node-concrete-runtime-loader",
  policyOwner: "src/ts/adapters/node_concrete_runtime.ts",
} satisfies ConcreteRuntimeHostPolicy<"node-ffi-runtime">);

const nodeConcreteRuntimeSurface: ConcreteRuntimeHostSurface<"node-ffi-runtime"> =
  createConcreteRuntimeHostSurface(nodeConcreteRuntimePolicy);

export function nodeConcreteRuntimeLoadCandidates(): readonly string[] {
  return nodeConcreteRuntimeSurface.loadCandidates();
}

export function requireNodeConcreteRuntimeByPolicy(options: NodeConcreteRuntimeOptions): NativeRuntime {
  return nodeConcreteRuntimeSurface.requireByPolicy(options);
}

export function assertNativeRuntimeContract(runtime: unknown): asserts runtime is NativeRuntime {
  nodeConcreteRuntimeSurface.assertContract(runtime);
}

export const nodeConcreteRuntimeManifest: NodeConcreteRuntimeManifest = nodeConcreteRuntimeSurface.manifest;
