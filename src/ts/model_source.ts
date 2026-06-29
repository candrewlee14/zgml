"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export const modelSourceManifest = Object.freeze({
  kind: "zgml-model-source",
  ...tsProductManifestPolicy("src/ts/model_source.ts"),
  runtimePath: "ModelSource -> Program -> Session",
});

export {
  createLlamaModelFamilyFacadeHelpers,
  createModelFacadePolicyHelpers,
  createModelHandleHelpers,
  createModelSourceFacadeHelpers,
  createSafetensorsFileHeaderHelpers,
  createSupportedCheckpointCatalogHelpers,
  isSafetensorsDataSource,
  isSafetensorsPath,
  modelKindName,
  modelLoadKindId,
  normalizeLoadModelKind,
  safetensorsDataBytes,
  safetensorsHeaderBytes,
} from "./runtime/model_source.js";
