import {
  requireSharedFrontendRuntimeHostByPolicy,
  type SharedFrontendRequire,
  type SharedFrontendRuntime,
  type SharedFrontendRuntimeHostOptions,
  type SharedFrontendRuntimeHostPolicy,
} from "./shared_frontend_runtime.js";

export type NodeSharedFrontendRuntimeOptions = SharedFrontendRuntimeHostOptions & Readonly<{
  requireModule: SharedFrontendRequire;
}>;

export const nodeSharedFrontendRuntimePolicy = Object.freeze({
  runtime: "Node",
  bridge: "zgml Node FFI runtime",
  policyOwner: "src/ts/adapters/node_shared_frontend_runtime.ts",
} satisfies SharedFrontendRuntimeHostPolicy<"Node">);

export function requireNodeSharedFrontendRuntime(options: NodeSharedFrontendRuntimeOptions): SharedFrontendRuntime {
  return requireSharedFrontendRuntimeHostByPolicy(nodeSharedFrontendRuntimePolicy, options);
}
