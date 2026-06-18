"use strict";

import type {
  LRScheduler,
  LRSchedulerStateDict,
  LRSchedulerStateKind,
  LRSchedulerStateSnapshot,
  OptimizerConfigSnapshot,
  OptimizerStateKind,
  OptimizerStateSnapshot,
} from "../public_api.js";

export type {
  LRScheduler,
  LRSchedulerStateDict,
  LRSchedulerStateKind,
  LRSchedulerStateSnapshot,
  Optimizer,
  OptimizerConfigSnapshot,
  OptimizerStateDict,
  OptimizerStateKind,
  OptimizerStateSnapshot,
} from "../public_api.js";

type AnyRecord = Record<string, any>;

const optimizerKinds = new Set(["sgd", "adam", "adamw", "rmsprop", "adagrad"]);
const schedulerKinds = new Set(["step-lr", "exponential-lr", "cosine-annealing-lr", "reduce-lr-on-plateau"]);

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

function isNonNegativeSafeInteger(value: unknown) {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isPositiveSafeInteger(value: unknown) {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isPositiveFiniteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isNonNegativeFiniteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isOptimizerKind(value: unknown): value is OptimizerStateKind {
  return typeof value === "string" && optimizerKinds.has(value);
}

function optimizerParamGroupSignature(group: AnyRecord) {
  return `${group.index}:${group.start}-${group.end}:${group.lr}:${group.weightDecay}`;
}

export function optimizerConfigSnapshotSignature(snapshot: AnyRecord) {
  const fields = [
    "optimizer-config",
    `kind=${snapshot.kind}`,
    `lr=${snapshot.lr}`,
  ];
  if (snapshot.kind === "sgd") {
    fields.push(
      `momentum=${snapshot.momentum}`,
      `weightDecay=${snapshot.weightDecay}`,
    );
  } else if (snapshot.kind === "rmsprop") {
    fields.push(
      `alpha=${snapshot.alpha}`,
      `eps=${snapshot.eps}`,
      `momentum=${snapshot.momentum}`,
      `weightDecay=${snapshot.weightDecay}`,
    );
  } else if (snapshot.kind === "adagrad") {
    fields.push(
      `lrDecay=${snapshot.lrDecay}`,
      `eps=${snapshot.eps}`,
      `weightDecay=${snapshot.weightDecay}`,
    );
  } else {
    fields.push(
      `beta1=${snapshot.beta1}`,
      `beta2=${snapshot.beta2}`,
      `eps=${snapshot.eps}`,
      `weightDecay=${snapshot.weightDecay}`,
      `decoupled=${snapshot.decoupledWeightDecay ? 1 : 0}`,
    );
  }
  fields.push(
    `step=${snapshot.step}`,
    `params=${snapshot.paramCount}`,
    `groups=${Array.isArray(snapshot.paramGroups) ? snapshot.paramGroups.map(optimizerParamGroupSignature).join(",") : ""}`,
  );
  return fields.join("|");
}

export function optimizerStateSnapshotSignature(snapshot: AnyRecord) {
  return [
    "optimizer-state",
    `kind=${snapshot.kind}`,
    `step=${snapshot.step}`,
    `params=${snapshot.paramCount}`,
    `entries=${Array.isArray(snapshot.entries) ? snapshot.entries.map((entry: AnyRecord) => entry.signature ?? entry.name ?? "").join(",") : ""}`,
  ].join("|");
}

export function lrSchedulerStateSnapshotSignature(snapshot: AnyRecord) {
  return [
    "lr-scheduler-state",
    `kind=${snapshot.kind}`,
    `step=${snapshot.step}`,
    `baseLr=${snapshot.baseLr}`,
    `lastLr=${snapshot.lastLr}`,
    `gamma=${snapshot.gamma}`,
    `stepSize=${snapshot.stepSize ?? "null"}`,
    `tMax=${snapshot.tMax ?? "null"}`,
    `etaMin=${snapshot.etaMin ?? "null"}`,
    `mode=${snapshot.mode ?? "null"}`,
    `factor=${snapshot.factor ?? "null"}`,
    `patience=${snapshot.patience ?? "null"}`,
    `threshold=${snapshot.threshold ?? "null"}`,
    `thresholdMode=${snapshot.thresholdMode ?? "null"}`,
    `cooldown=${snapshot.cooldown ?? "null"}`,
    `cooldownCounter=${snapshot.cooldownCounter ?? "null"}`,
    `minLr=${snapshot.minLr ?? "null"}`,
    `eps=${snapshot.eps ?? "null"}`,
    `best=${snapshot.best ?? "null"}`,
    `badEpochs=${snapshot.badEpochs ?? "null"}`,
    `optimizer=${snapshot.optimizerKind ?? "null"}`,
  ].join("|");
}

export function isOptimizerConfigSnapshot(snapshot: unknown): snapshot is OptimizerConfigSnapshot {
  if (!isRecord(snapshot) || !Object.isFrozen(snapshot)) return false;
  if (!isOptimizerKind(snapshot.kind) || typeof snapshot.signature !== "string") return false;
  if (!isNonNegativeFiniteNumber(snapshot.lr)) return false;
  if (!isNonNegativeFiniteNumber(snapshot.weightDecay) || snapshot.weight_decay !== snapshot.weightDecay) return false;
  if (!isNonNegativeSafeInteger(snapshot.step)) return false;
  if (!isNonNegativeSafeInteger(snapshot.paramCount)) return false;
  if (!Array.isArray(snapshot.paramGroups) || !Object.isFrozen(snapshot.paramGroups)) return false;
  if (snapshot.param_groups !== snapshot.paramGroups) return false;
  for (const group of snapshot.paramGroups) {
    if (!isRecord(group) || !Object.isFrozen(group)) return false;
    if (!isNonNegativeSafeInteger(group.index)) return false;
    if (!isPositiveFiniteNumber(group.lr)) return false;
    if (!isNonNegativeFiniteNumber(group.weightDecay) || group.weight_decay !== group.weightDecay) return false;
    if (!isNonNegativeSafeInteger(group.paramCount)) return false;
    if (!isNonNegativeSafeInteger(group.start) || !isNonNegativeSafeInteger(group.end) || group.end < group.start) return false;
    if (group.end - group.start !== group.paramCount) return false;
  }
  if (snapshot.kind === "sgd") {
    if (!isNonNegativeFiniteNumber(snapshot.momentum) || snapshot.momentum >= 1) return false;
  } else if (snapshot.kind === "rmsprop") {
    if (!isNonNegativeFiniteNumber(snapshot.alpha) || snapshot.alpha >= 1) return false;
    if (!isPositiveFiniteNumber(snapshot.eps)) return false;
    if (!isNonNegativeFiniteNumber(snapshot.momentum) || snapshot.momentum >= 1) return false;
  } else if (snapshot.kind === "adagrad") {
    if (!isNonNegativeFiniteNumber(snapshot.lrDecay) || snapshot.lr_decay !== snapshot.lrDecay) return false;
    if (!isPositiveFiniteNumber(snapshot.eps)) return false;
  } else {
    if (!isNonNegativeFiniteNumber(snapshot.beta1) || snapshot.beta1 >= 1) return false;
    if (!isNonNegativeFiniteNumber(snapshot.beta2) || snapshot.beta2 >= 1) return false;
    if (!isPositiveFiniteNumber(snapshot.eps)) return false;
    if (typeof snapshot.decoupledWeightDecay !== "boolean") return false;
    if (snapshot.kind === "adam" && snapshot.decoupledWeightDecay !== false) return false;
    if (snapshot.kind === "adamw" && snapshot.decoupledWeightDecay !== true) return false;
  }
  return snapshot.signature === optimizerConfigSnapshotSignature(snapshot);
}

export function requireOptimizerConfigSnapshot(snapshot: unknown): OptimizerConfigSnapshot {
  if (isOptimizerConfigSnapshot(snapshot)) return snapshot;
  throw new Error("expected frozen OptimizerConfigSnapshot");
}

export const assertOptimizerConfigSnapshot = requireOptimizerConfigSnapshot;
export const assert_optimizer_config_snapshot = requireOptimizerConfigSnapshot;

export function matchesOptimizerConfigSnapshotSignature(snapshot: unknown, signature: unknown) {
  return isOptimizerConfigSnapshot(snapshot) && typeof signature === "string" && snapshot.signature === signature;
}

export const matches_optimizer_config_snapshot_signature = matchesOptimizerConfigSnapshotSignature;

export function isOptimizerStateSnapshot(snapshot: unknown): snapshot is OptimizerStateSnapshot {
  if (!isRecord(snapshot) || !Object.isFrozen(snapshot)) return false;
  if (!isOptimizerKind(snapshot.kind) || typeof snapshot.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(snapshot.step)) return false;
  if (!isNonNegativeSafeInteger(snapshot.paramCount)) return false;
  if (!Array.isArray(snapshot.entries) || !Object.isFrozen(snapshot.entries)) return false;
  for (const entry of snapshot.entries) {
    if (!isRecord(entry) || !Object.isFrozen(entry)) return false;
    if (typeof entry.name !== "string" || entry.name.length === 0) return false;
    if (typeof entry.signature !== "string" || !entry.signature.startsWith("optimizer-state-entry|")) return false;
    if (!Array.isArray(entry.shape) || !Object.isFrozen(entry.shape)) return false;
    if (typeof entry.layout !== "string" || !(entry.data instanceof Float32Array)) return false;
  }
  return snapshot.signature === optimizerStateSnapshotSignature(snapshot);
}

export function requireOptimizerStateSnapshot(snapshot: unknown): OptimizerStateSnapshot {
  if (isOptimizerStateSnapshot(snapshot)) return snapshot;
  throw new Error("expected frozen OptimizerStateSnapshot");
}

export const assertOptimizerStateSnapshot = requireOptimizerStateSnapshot;
export const assert_optimizer_state_snapshot = requireOptimizerStateSnapshot;

export function matchesOptimizerStateSnapshotSignature(snapshot: unknown, signature: unknown) {
  return isOptimizerStateSnapshot(snapshot) && typeof signature === "string" && snapshot.signature === signature;
}

export const matches_optimizer_state_snapshot_signature = matchesOptimizerStateSnapshotSignature;

export function isLRSchedulerStateSnapshot(snapshot: unknown): snapshot is LRSchedulerStateSnapshot {
  if (!isRecord(snapshot) || !Object.isFrozen(snapshot)) return false;
  if (typeof snapshot.kind !== "string" || !schedulerKinds.has(snapshot.kind)) return false;
  if (typeof snapshot.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(snapshot.step)) return false;
  if (!isNonNegativeFiniteNumber(snapshot.baseLr)) return false;
  if (snapshot.base_lr !== snapshot.baseLr) return false;
  if (!isNonNegativeFiniteNumber(snapshot.lastLr)) return false;
  if (snapshot.last_lr !== snapshot.lastLr) return false;
  if (!isPositiveFiniteNumber(snapshot.gamma)) return false;
  if (snapshot.stepSize !== undefined && !isPositiveSafeInteger(snapshot.stepSize)) return false;
  if (snapshot.step_size !== snapshot.stepSize) return false;
  if (snapshot.tMax !== undefined && !isPositiveSafeInteger(snapshot.tMax)) return false;
  if (snapshot.t_max !== snapshot.tMax) return false;
  if (snapshot.etaMin !== undefined && !isNonNegativeFiniteNumber(snapshot.etaMin)) return false;
  if (snapshot.eta_min !== snapshot.etaMin) return false;
  if (snapshot.mode !== undefined && snapshot.mode !== "min" && snapshot.mode !== "max") return false;
  if (snapshot.factor !== undefined && (!isPositiveFiniteNumber(snapshot.factor) || snapshot.factor >= 1)) return false;
  if (snapshot.patience !== undefined && !isNonNegativeSafeInteger(snapshot.patience)) return false;
  if (snapshot.threshold !== undefined && !isNonNegativeFiniteNumber(snapshot.threshold)) return false;
  if (snapshot.thresholdMode !== undefined && snapshot.thresholdMode !== "rel" && snapshot.thresholdMode !== "abs") return false;
  if (snapshot.threshold_mode !== snapshot.thresholdMode) return false;
  if (snapshot.cooldown !== undefined && !isNonNegativeSafeInteger(snapshot.cooldown)) return false;
  if (snapshot.cooldownCounter !== undefined && !isNonNegativeSafeInteger(snapshot.cooldownCounter)) return false;
  if (snapshot.cooldown_counter !== snapshot.cooldownCounter) return false;
  if (snapshot.minLr !== undefined && !isNonNegativeFiniteNumber(snapshot.minLr)) return false;
  if (snapshot.min_lr !== snapshot.minLr) return false;
  if (snapshot.eps !== undefined && !isNonNegativeFiniteNumber(snapshot.eps)) return false;
  if (snapshot.best !== undefined && (typeof snapshot.best !== "number" || !Number.isFinite(snapshot.best))) return false;
  if (snapshot.badEpochs !== undefined && !isNonNegativeSafeInteger(snapshot.badEpochs)) return false;
  if (snapshot.bad_epochs !== snapshot.badEpochs) return false;
  if (snapshot.optimizerKind !== null && !isOptimizerKind(snapshot.optimizerKind)) return false;
  return snapshot.signature === lrSchedulerStateSnapshotSignature(snapshot);
}

export function requireLRSchedulerStateSnapshot(snapshot: unknown): LRSchedulerStateSnapshot {
  if (isLRSchedulerStateSnapshot(snapshot)) return snapshot;
  throw new Error("expected frozen LRSchedulerStateSnapshot");
}

export const assertLRSchedulerStateSnapshot = requireLRSchedulerStateSnapshot;
export const assert_lr_scheduler_state_snapshot = requireLRSchedulerStateSnapshot;

export function matchesLRSchedulerStateSnapshotSignature(snapshot: unknown, signature: unknown) {
  return isLRSchedulerStateSnapshot(snapshot) && typeof signature === "string" && snapshot.signature === signature;
}

export const matches_lr_scheduler_state_snapshot_signature = matchesLRSchedulerStateSnapshotSignature;
