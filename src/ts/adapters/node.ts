export {
  adapterOwnsFrontendPolicy,
  nativeAdapterCapabilities,
  nativeAdapterEvidence,
  nodeKoffiFallbackPath,
  nativeLibraryExtensionForPlatform,
  nativeLibraryFilename,
  nativeLibraryMissingMessage,
  resolveNativeLibraryPath,
} from "./native.js";
export {
  nodeConcreteRuntimeManifest,
  type NodeConcreteRuntimeManifest,
} from "./node_concrete_runtime.js";

import { nativeAdapterEvidence } from "./native.js";
import {
  nativeAdapterManifestPolicy,
  type NativeAdapterManifestPolicy,
} from "./native.js";
import {
  nodeConcreteRuntimeManifest,
  type NodeConcreteRuntimeManifest,
} from "./node_concrete_runtime.js";

export const nodeAdapterEvidence = nativeAdapterEvidence("node");
export type NodeAdapterManifest = NativeAdapterManifestPolicy<"src/ts/adapters/node.ts"> & Readonly<{
  kind: "zgml-node-adapter";
  adapterEvidence: typeof nodeAdapterEvidence;
  concreteRuntime: NodeConcreteRuntimeManifest;
}>;

export const nodeAdapterManifest: NodeAdapterManifest = Object.freeze({
  kind: "zgml-node-adapter",
  ...nativeAdapterManifestPolicy("src/ts/adapters/node.ts"),
  adapterEvidence: nodeAdapterEvidence,
  concreteRuntime: nodeConcreteRuntimeManifest,
});
