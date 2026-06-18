"use strict";

import { lrSchedulerStateSnapshotSignature } from "./optimizer_snapshot.js";

type AnyRecord = Record<string, any>;

export type SchedulerOptimizer = Readonly<{
  config(): Readonly<{
    kind?: unknown;
    lr?: unknown;
  }>;
}> & {
  setLearningRate(lr: unknown): unknown;
};

export type SchedulerStateTarget = {
  t: number;
  baseLr: number;
  lastLr: number;
  gamma: number;
  stepSize?: number;
  tMax?: number;
  t_max?: number;
  etaMin?: number;
  eta_min?: number;
  mode?: string;
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: string;
  threshold_mode?: string;
  cooldown?: number;
  cooldownCounter?: number;
  cooldown_counter?: number;
  minLr?: number;
  min_lr?: number;
  eps?: number;
  best?: number;
  badEpochs?: number;
  bad_epochs?: number;
  optimizer: SchedulerOptimizer;
  optimizerKind(): unknown;
};

type SchedulerStateSnapshot = AnyRecord & {
  kind?: unknown;
  step?: unknown;
  baseLr?: unknown;
  base_lr?: unknown;
  lastLr?: unknown;
  last_lr?: unknown;
  gamma?: unknown;
  stepSize?: unknown;
  step_size?: unknown;
  tMax?: unknown;
  t_max?: unknown;
  etaMin?: unknown;
  eta_min?: unknown;
  mode?: unknown;
  factor?: unknown;
  patience?: unknown;
  threshold?: unknown;
  thresholdMode?: unknown;
  threshold_mode?: unknown;
  cooldown?: unknown;
  cooldownCounter?: unknown;
  cooldown_counter?: unknown;
  minLr?: unknown;
  min_lr?: unknown;
  eps?: unknown;
  best?: unknown;
  badEpochs?: unknown;
  bad_epochs?: unknown;
};

export function createSchedulerStateHelpers() {
  function optimizerLearningRate(optimizer: SchedulerOptimizer) {
    const config = optimizer.config();
    const lr = Number(config?.lr);
    if (!Number.isFinite(lr) || lr < 0) throw new Error("optim scheduler requires an optimizer with a non-negative finite lr");
    return lr;
  }

  function schedulerState(kind: string, scheduler: SchedulerStateTarget) {
    const snapshot: AnyRecord = {
      kind,
      step: scheduler.t,
      baseLr: scheduler.baseLr,
      base_lr: scheduler.baseLr,
      lastLr: scheduler.lastLr,
      last_lr: scheduler.lastLr,
      gamma: scheduler.gamma,
      stepSize: scheduler.stepSize,
      step_size: scheduler.stepSize,
      tMax: scheduler.tMax,
      t_max: scheduler.tMax,
      etaMin: scheduler.etaMin,
      eta_min: scheduler.etaMin,
      mode: scheduler.mode,
      factor: scheduler.factor,
      patience: scheduler.patience,
      threshold: scheduler.threshold,
      thresholdMode: scheduler.thresholdMode,
      threshold_mode: scheduler.thresholdMode,
      cooldown: scheduler.cooldown,
      cooldownCounter: scheduler.cooldownCounter,
      cooldown_counter: scheduler.cooldownCounter,
      minLr: scheduler.minLr,
      min_lr: scheduler.minLr,
      eps: scheduler.eps,
      best: scheduler.best,
      badEpochs: scheduler.badEpochs,
      bad_epochs: scheduler.badEpochs,
      optimizerKind: scheduler.optimizerKind(),
    };
    snapshot.signature = lrSchedulerStateSnapshotSignature(snapshot);
    return Object.freeze(snapshot);
  }

  function loadSchedulerState(scheduler: SchedulerStateTarget, source: unknown, kind: string, strict = true, validateOnly = false) {
    if (!source || typeof source !== "object") throw new Error("optim scheduler state must be an object");
    const state = source as SchedulerStateSnapshot;
    if (strict && state.kind !== kind) throw new Error(`optim scheduler state kind must be ${kind}, got ${state.kind}`);
    const step = Number(state.step);
    if (!Number.isSafeInteger(step) || step < 0) throw new Error(`optim scheduler state step must be a non-negative safe integer, got ${state.step}`);
    const baseLr = Number(state.baseLr ?? state.base_lr);
    const lastLr = Number(state.lastLr ?? state.last_lr);
    if (!Number.isFinite(baseLr) || baseLr < 0) {
      throw new Error(`optim scheduler state baseLr must be a non-negative finite number, got ${baseLr}`);
    }
    for (const [name, value] of [["gamma", Number(state.gamma)]] as const) {
      if (!Number.isFinite(value) || value <= 0) throw new Error(`optim scheduler state ${name} must be a positive finite number`);
    }
    if (!Number.isFinite(lastLr) || lastLr < 0) {
      throw new Error(`optim scheduler state lastLr must be a non-negative finite number, got ${lastLr}`);
    }
    const stepSizeValue = state.stepSize ?? state.step_size;
    const stepSize = stepSizeValue === undefined ? undefined : Number(stepSizeValue);
    if (stepSize !== undefined && (!Number.isSafeInteger(stepSize) || stepSize <= 0)) {
      throw new Error(`optim scheduler state stepSize must be a positive safe integer, got ${stepSize}`);
    }
    const tMaxValue = state.tMax ?? state.t_max;
    const tMax = tMaxValue === undefined ? undefined : Number(tMaxValue);
    if (tMax !== undefined && (!Number.isSafeInteger(tMax) || tMax <= 0)) {
      throw new Error(`optim scheduler state tMax must be a positive safe integer, got ${tMax}`);
    }
    const etaMinValue = state.etaMin ?? state.eta_min;
    const etaMin = etaMinValue === undefined ? undefined : Number(etaMinValue);
    if (etaMin !== undefined && (!Number.isFinite(etaMin) || etaMin < 0)) {
      throw new Error(`optim scheduler state etaMin must be a non-negative finite number, got ${etaMin}`);
    }
    const mode = state.mode === undefined ? undefined : String(state.mode);
    if (mode !== undefined && mode !== "min" && mode !== "max") throw new Error(`optim scheduler state mode must be min or max, got ${mode}`);
    const thresholdModeValue = state.thresholdMode ?? state.threshold_mode;
    const thresholdMode = thresholdModeValue === undefined ? undefined : String(thresholdModeValue);
    if (thresholdMode !== undefined && thresholdMode !== "rel" && thresholdMode !== "abs") {
      throw new Error(`optim scheduler state thresholdMode must be rel or abs, got ${thresholdMode}`);
    }
    const plateauNumbers = [
      ["factor", state.factor, (value: number) => value > 0 && value < 1],
      ["threshold", state.threshold, (value: number) => value >= 0],
      ["minLr", state.minLr ?? state.min_lr, (value: number) => value >= 0],
      ["eps", state.eps, (value: number) => value >= 0],
      ["best", state.best, (_value: number) => true],
    ] as const;
    const plateauNumberValues = new Map<string, number>();
    for (const [name, sourceValue, valid] of plateauNumbers) {
      if (sourceValue === undefined) continue;
      const value = Number(sourceValue);
      if (!Number.isFinite(value) || !valid(value)) throw new Error(`optim scheduler state ${name} must be finite and valid, got ${sourceValue}`);
      plateauNumberValues.set(name, value);
    }
    const plateauIntegers = [
      ["patience", state.patience],
      ["cooldown", state.cooldown],
      ["cooldownCounter", state.cooldownCounter ?? state.cooldown_counter],
      ["badEpochs", state.badEpochs ?? state.bad_epochs],
    ] as const;
    const plateauIntegerValues = new Map<string, number>();
    for (const [name, sourceValue] of plateauIntegers) {
      if (sourceValue === undefined) continue;
      const value = Number(sourceValue);
      if (!Number.isSafeInteger(value) || value < 0) throw new Error(`optim scheduler state ${name} must be a non-negative safe integer, got ${sourceValue}`);
      plateauIntegerValues.set(name, value);
    }
    if (!validateOnly) {
      scheduler.t = step;
      scheduler.baseLr = baseLr;
      scheduler.lastLr = lastLr;
      scheduler.gamma = Number(state.gamma);
      if (stepSize !== undefined) scheduler.stepSize = stepSize;
      if (tMax !== undefined) scheduler.tMax = tMax;
      if (etaMin !== undefined) scheduler.etaMin = etaMin;
      if (mode !== undefined) scheduler.mode = mode;
      if (thresholdMode !== undefined) scheduler.thresholdMode = thresholdMode;
      for (const [name, value] of plateauNumberValues) {
        (scheduler as AnyRecord)[name] = value;
      }
      for (const [name, value] of plateauIntegerValues) {
        (scheduler as AnyRecord)[name] = value;
      }
      scheduler.optimizer.setLearningRate(scheduler.lastLr);
    }
    return scheduler;
  }

  return Object.freeze({
    optimizerLearningRate,
    schedulerState,
    loadSchedulerState,
  });
}
