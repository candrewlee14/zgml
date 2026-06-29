export {
  adapterOwnsFrontendPolicy,
  nativeAdapterCapabilities,
  nativeAdapterEvidence,
  nativeLibraryExtensionForPlatform,
  nativeLibraryFilename,
  nativeLibraryMissingMessage,
  resolveNativeLibraryPath,
} from "./native.js";
export {
  bunConcreteRuntimeManifest,
  type BunConcreteRuntimeManifest,
} from "./bun_concrete_runtime.js";

import {
  nativeAdapterEvidence,
  nativeAdapterManifestPolicy,
  type NativeAdapterManifestPolicy,
} from "./native.js";
import {
  bunConcreteRuntimeManifest,
  type BunConcreteRuntimeManifest,
} from "./bun_concrete_runtime.js";

export const bunAdapterEvidence = nativeAdapterEvidence("bun");
export type BunAdapterManifest = NativeAdapterManifestPolicy<"src/ts/adapters/bun.ts"> & Readonly<{
  kind: "zgml-bun-adapter";
  adapterEvidence: typeof bunAdapterEvidence;
  concreteRuntime: BunConcreteRuntimeManifest;
}>;

export const bunAdapterManifest: BunAdapterManifest = Object.freeze({
  kind: "zgml-bun-adapter",
  ...nativeAdapterManifestPolicy("src/ts/adapters/bun.ts"),
  adapterEvidence: bunAdapterEvidence,
  concreteRuntime: bunConcreteRuntimeManifest,
});
