import {
  requireSharedFrontendRuntimeHostByPolicy,
  type SharedFrontendRequire,
  type SharedFrontendRuntime,
  type SharedFrontendRuntimeHostOptions,
  type SharedFrontendRuntimeHostPolicy,
} from "./shared_frontend_runtime.js";

export type BunSharedFrontendRuntimeOptions = SharedFrontendRuntimeHostOptions & Readonly<{
  requireModule: SharedFrontendRequire;
}>;

export const bunSharedFrontendRuntimePolicy = Object.freeze({
  runtime: "Bun",
  bridge: "zgml Bun FFI runtime",
  policyOwner: "src/ts/adapters/bun_shared_frontend_runtime.ts",
} satisfies SharedFrontendRuntimeHostPolicy<"Bun">);

export function requireBunSharedFrontendRuntime(options: BunSharedFrontendRuntimeOptions): SharedFrontendRuntime {
  return requireSharedFrontendRuntimeHostByPolicy(bunSharedFrontendRuntimePolicy, options);
}
