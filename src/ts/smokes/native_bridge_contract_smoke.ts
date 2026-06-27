"use strict";

declare const console: { log: (message?: unknown, ...optionalParams: unknown[]) => void };
declare const require: {
  (id: string): any;
  resolve(id: string): string;
};
declare const process: { exit: (code?: number) => never };

const root = require(["zgml"].join("/"));
const distNode = require(["..", "node.cjs"].join("/"));
const distConcreteNode = require(["..", "adapters", "node_ffi_runtime.cjs"].join("/"));
const packageNode = require(["zgml", "node"].join("/"));
const {
  expectNativeApiContractManifest,
} = require([".", "smoke_contracts.cjs"].join("/"));
const {
  extraExports,
  missingExports,
  nativeApiContractManifest,
  nativeApiContractSignature,
  nativePackageSpineContractManifest,
  requiredNativeApiExports,
  requiredNativePackageSpineExports,
} = require(["..", "runtime", "native_api_contract.cjs"].join("/"));

expectNativeApiContractManifest(nativeApiContractManifest, "native bridge contract smoke");

if (
  nativePackageSpineContractManifest.source !== "generated-ts" ||
  nativePackageSpineContractManifest.generator !== "scripts/generate_native_runtime_wrappers.cjs" ||
  nativePackageSpineContractManifest.generatedFrom !== "src/ts/runtime/native_api_contract.ts" ||
  nativePackageSpineContractManifest.handwrittenRootExportList !== false ||
  nativeApiContractManifest.packageSpine !== nativePackageSpineContractManifest
) {
  throw new Error("native bridge contract smoke must consume the generated native package spine contract");
}

if (!require.resolve(["zgml"].join("/")).endsWith("/dist/node.cjs")) {
  throw new Error("zgml package root must resolve runtime imports to the tsdown-emitted dist/node.cjs artifact");
}

const rootOnly = extraExports(root, distNode);
const distOnly = extraExports(distNode, root);
if (rootOnly.length !== 0 || distOnly.length !== 0) {
  throw new Error(`zgml root and dist/node exports diverged: ${JSON.stringify({ rootOnly, distOnly })}`);
}

const missingNative = missingExports(root, requiredNativeApiExports);
if (missingNative.length !== 0) {
  throw new Error(`zgml root is missing native zgml exports: ${missingNative.join(", ")}`);
}

if (root.zgml !== distConcreteNode.zgml) {
  throw new Error("zgml root namespace must be the concrete Node native runtime zgml namespace");
}

if (root.torch !== root.zgml || distConcreteNode.torch !== distConcreteNode.zgml) {
  throw new Error("torch must remain an explicit compatibility alias of zgml, not the primary namespace");
}

for (const key of requiredNativeApiExports) {
  if (root[key] !== distConcreteNode[key]) {
    throw new Error(`zgml root export ${key} must bridge to the concrete Node native runtime`);
  }
}

if (typeof root.runtimeInfo !== "function" || typeof root.loadedRuntimeInfo !== "object") {
  throw new Error("zgml root must expose native runtimeInfo and loadedRuntimeInfo");
}

const missingDistSpine = missingExports(distNode, requiredNativePackageSpineExports);
if (missingDistSpine.length !== 0) {
  throw new Error(`dist/node.cjs is missing TS package-spine exports: ${missingDistSpine.join(", ")}`);
}

if (
  distNode.frontendManifest.source !== "ts" ||
  distNode.nodeAdapterEvidence.frontendSource !== "ts" ||
  distNode.nodeAdapterEvidence.productSemanticsOwner !== distNode.frontendManifest.productSemanticsOwner ||
  distNode.nativeApiContract.nativeApiContractManifest.productSourceOfTruth !== distNode.frontendManifest.productSourceOfTruth ||
  distNode.nativeApiContract.nativeApiContractManifest.nativeAlignment !== distNode.frontendManifest.nativeAlignment ||
  distNode.nativeApiContract.nativeApiContractManifest.nativeProductPolicy !== distNode.frontendManifest.nativeProductPolicy
) {
  throw new Error("dist/node.cjs must remain the TS-authored Node package spine");
}

if (distNode.nativeNodeRuntime !== distConcreteNode) {
  throw new Error("dist/node.cjs must load the tsdown-emitted concrete Node runtime without repo-only fallbacks");
}

const distStillMissingNative = missingExports(distNode, requiredNativeApiExports);
if (distStillMissingNative.length !== 0) {
  console.log(`zgml native bridge contract smoke ok: ${nativeApiContractSignature()}, ${distStillMissingNative.length} still outside dist/node`);
  process.exit(0);
}

for (const key of requiredNativeApiExports) {
  if (distNode[key] !== distConcreteNode[key]) {
    throw new Error(`dist/node.cjs export ${key} must bridge to the concrete Node native runtime`);
  }
  if (packageNode[key] !== distConcreteNode[key]) {
    throw new Error(`zgml/node export ${key} must resolve through the TS-built Node native bridge`);
  }
}

console.log(`zgml native bridge contract smoke ok: ${nativeApiContractSignature()}, dist/node satisfies the native bridge contract`);

export {};
