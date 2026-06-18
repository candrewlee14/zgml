"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export {
  createLossTrainHelpers,
} from "./train/loss_train_helpers.js";

export type {
  BCELoss,
  BCELossOptions,
  BCEWithLogitsLoss,
  ClassLossOptions,
  CrossEntropyLoss,
  HuberLoss,
  HuberLossOptions,
  L1Loss,
  LossNamespace,
  LossReduction,
  LossReductionOptions,
  MSELoss,
  NLLLoss,
  PublicLossNamespace,
  SmoothL1Loss,
  SmoothL1LossOptions,
} from "./public_api.js";

export const lossManifest = Object.freeze({
  kind: "zgml-loss",
  ...tsProductManifestPolicy("src/ts/loss.ts"),
  factory: "createLossTrainHelpers",
});
