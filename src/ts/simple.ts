"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export * as tensor from "./tensor.js";
export * as nn from "./nn.js";
export * as loss from "./loss.js";
export * as optim from "./optim.js";
export * as train from "./train.js";
export * as data from "./data.js";
export * as checkpoint from "./checkpoint.js";
export * as lazy from "./lazy.js";
export * as compile from "./compile.js";

export type {
  PublicCheckpointNamespace,
  PublicCompileNamespace,
  PublicDataNamespace,
  PublicLazyNamespace,
  PublicLossNamespace,
  PublicNnNamespace,
  PublicOptimNamespace,
  PublicSimpleNamespace,
  PublicTrainNamespace,
  Tensor,
  TensorLike,
  TensorShapeTuple,
} from "./public_api.js";

export const simpleFirstContactNamespaces = Object.freeze([
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

export const simpleRuntimeValues = Object.freeze([
  "simple",
  "zgml",
  "F",
] as const);

export const simpleManifest = Object.freeze({
  kind: "zgml-simple-surface",
  ...tsProductManifestPolicy("src/ts/simple.ts"),
  rootRuntimeValue: "simple",
  canonicalFriendlyNamespace: "zgml",
  functionalNamespace: "F",
  firstContactRuntimeHandle: "zgml.native",
  namespaces: simpleFirstContactNamespaces,
  runtimeValues: simpleRuntimeValues,
  advancedRuntimeSurface: false,
});
