"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export {
  createCheckpointHelpers,
} from "./train/training.js";

export type {
  CheckpointCreateOptions,
  CheckpointInspection,
  CheckpointModuleState,
  CheckpointOptimizerEntry,
  CheckpointNamespace,
  CheckpointOptimizerState,
  CheckpointRestoreOptions,
  CheckpointSchedulerState,
  CheckpointTensorEntry,
  CheckpointTensorInspection,
  PublicCheckpointNamespace,
  ZgmlCheckpoint,
} from "./public_api.js";

export const checkpointManifest = Object.freeze({
  kind: "zgml-checkpoint",
  ...tsProductManifestPolicy("src/ts/checkpoint.ts"),
  factory: "createCheckpointHelpers",
});
