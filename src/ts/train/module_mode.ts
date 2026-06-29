"use strict";

export type ModuleTrainingChild = {
  training?: boolean;
  train?: (mode?: boolean) => unknown;
};

export type ModuleTrainingTarget = ModuleTrainingChild & {
  layers?: readonly ModuleTrainingChild[];
};

export function normalizeModuleTrainingMode(mode = true): boolean {
  if (typeof mode !== "boolean") {
    throw new Error(`module train mode must be boolean, got ${typeof mode}`);
  }
  return mode;
}

function hasTrainingSlot(module: ModuleTrainingChild): module is ModuleTrainingChild & { training: boolean | undefined } {
  return "training" in module;
}

export function initializeModuleMode(module: ModuleTrainingTarget): void {
  module.training = true;
}

export function setModuleTraining<T extends ModuleTrainingTarget>(module: T, mode = true): T {
  module.training = normalizeModuleTrainingMode(mode);
  return module;
}

export function setChildModuleTraining<T extends ModuleTrainingTarget>(module: T, mode = true): T {
  const training = normalizeModuleTrainingMode(mode);
  module.training = training;
  for (const layer of module.layers ?? []) {
    if (layer && typeof layer.train === "function") {
      layer.train(training);
    } else if (layer && hasTrainingSlot(layer)) {
      layer.training = training;
    }
  }
  return module;
}
