"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export {
  createOptimizerClasses,
} from "./train/optimizer_classes.js";

export {
  assertLRSchedulerStateSnapshot,
  assertOptimizerConfigSnapshot,
  assertOptimizerStateSnapshot,
  assert_lr_scheduler_state_snapshot,
  assert_optimizer_config_snapshot,
  assert_optimizer_state_snapshot,
  isLRSchedulerStateSnapshot,
  isOptimizerConfigSnapshot,
  isOptimizerStateSnapshot,
  lrSchedulerStateSnapshotSignature,
  matchesLRSchedulerStateSnapshotSignature,
  matchesOptimizerConfigSnapshotSignature,
  matchesOptimizerStateSnapshotSignature,
  matches_lr_scheduler_state_snapshot_signature,
  matches_optimizer_config_snapshot_signature,
  matches_optimizer_state_snapshot_signature,
  optimizerConfigSnapshotSignature,
  optimizerStateSnapshotSignature,
  requireLRSchedulerStateSnapshot,
  requireOptimizerConfigSnapshot,
  requireOptimizerStateSnapshot,
  type LRScheduler,
  type LRSchedulerStateDict,
  type LRSchedulerStateKind,
  type LRSchedulerStateSnapshot,
  type Optimizer,
  type OptimizerConfigSnapshot,
  type OptimizerStateDict,
  type OptimizerStateKind,
  type OptimizerStateSnapshot,
} from "./train/optimizer_snapshot.js";

export {
  createOptimNamespace,
} from "./train/training.js";

export type {
  AdamConfig,
  AdagradConfig,
  CosineAnnealingLRConfig,
  LRSchedulerConfig,
  LRSchedulerNamespace,
  OptimNamespace,
  OptimizerParamGroupInput,
  OptimizerParamGroupSnapshot,
  OptimizerParameterSource,
  OptimizerTarget,
  ReduceLROnPlateauConfig,
  PublicOptimNamespace,
  RMSpropConfig,
  SGDConfig,
  StepLRConfig,
} from "./public_api.js";

export const optimManifest = Object.freeze({
  kind: "zgml-optim",
  ...tsProductManifestPolicy("src/ts/optim.ts"),
  factories: Object.freeze(["createOptimizerClasses", "createOptimNamespace"]),
});
