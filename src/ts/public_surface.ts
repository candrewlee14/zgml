"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export const firstContactRootNamespaces = Object.freeze([
  "tensor",
  "nn",
  "loss",
  "optim",
  "train",
  "data",
  "checkpoint",
  "lazy",
  "compile",
] as const);

export const firstContactRootValues = Object.freeze([
  "simple",
  "zgml",
  "F",
] as const);

export const stableRootNamespaces = Object.freeze([
  "simple",
  "tensor",
  "nn",
  "loss",
  "optim",
  "train",
  "data",
  "checkpoint",
  "lazy",
  "compile",
  "program",
  "session",
  "stepParams",
  "nativeBuffer",
  "programDevice",
  "modelSource",
  "inspection",
] as const);

export const stableRootValues = Object.freeze([
  "simple",
  "zgml",
  "F",
] as const);

export const advancedRootNamespaces = Object.freeze([
  "abi",
  "compilerSignatures",
  "executionPlan",
  "kernelPlan",
  "moduleBindings",
  "moduleCompilerPolicy",
  "moduleCompatibility",
  "moduleFacade",
  "moduleProgramDesc",
  "nativeApiContract",
  "programBuffers",
  "programBufferFactory",
  "programFacade",
  "programFacadePolicy",
  "programLayout",
  "programModuleBinding",
  "programParameters",
  "programPolicyAccessors",
  "programResources",
  "programShapes",
  "programSizing",
  "sessionBinding",
  "sessionContract",
  "sessionFacade",
  "sessionLayout",
  "sessionLifecycle",
  "sessionParameters",
  "sessionProfile",
  "sessionTensor",
  "sessionValues",
  "stepParamsFacade",
  "tensorPlacement",
  "tensorProgramIr",
  "traceCompiler",
] as const);

export const legacyCompatibleRootNamespaces = Object.freeze([
  "activation",
  "gradMode",
  "shape",
  "tensorCore",
  "tensorData",
  "tensorFacade",
  "tensorFactory",
  "tensorGradState",
  "tensorIndex",
  "tensorJoin",
  "tensorMath",
  "tensorStaticSurface",
  "tensorView",
  "token",
  "nnEmbedding",
  "nnFeatureNorm",
  "nnLinear",
  "nnNamespace",
  "nnParameterless",
  "nnSequential",
  "nnShape",
  "trainMode",
  "trainState",
] as const);

export const legacyCompatibleRootValues = Object.freeze([
  "torch",
] as const);

export const rootSurfacePolicy = Object.freeze({
  stableFirst: true,
  canonicalFriendlyNamespace: "zgml",
  simpleFriendlyNamespace: "simple",
  compatibilityFriendlyNamespace: "torch",
  firstContactSurfaceIsSmall: true,
  firstContactRuntimeHandle: "zgml.native",
  compatibilityExportsRemainPublic: true,
  internalPackagePolicyOnly: true,
  classificationCoversRootNamespaceExports: true,
  firstContactIsSubsetOfStableSurface: true,
  valueClassificationCoversFriendlyRootExports: true,
  simpleSurfaceSubpath: "zgml/simple",
  newProductSurfaceGoesThroughStableNamespaces: true,
  runtimeEvidenceStaysInspectable: true,
});

export const publicSurfaceManifest = Object.freeze({
  kind: "zgml-public-surface",
  ...tsProductManifestPolicy("src/ts/public_surface.ts"),
  rootEntry: "src/ts/index.ts",
  firstContactRootNamespaces,
  firstContactRootValues,
  stableRootNamespaces,
  stableRootValues,
  advancedRootNamespaces,
  legacyCompatibleRootNamespaces,
  legacyCompatibleRootValues,
  rootSurfacePolicy,
});
