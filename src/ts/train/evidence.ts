"use strict";

import type {
  TrainEvaluateEvidence,
  TrainEvaluateStepEvidence,
  TrainPredictEvidence,
  TrainPredictStepEvidence,
  TrainFitEvidence,
  TrainFitStepEvidence,
  TrainStepEvidence,
} from "../public_api.js";

type AnyRecord = Record<string, any>;

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

function scalarSignature(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "null";
}

function isNonNegativeSafeInteger(value: unknown) {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function nullableNonNegativeSafeInteger(value: unknown) {
  return value === null || isNonNegativeSafeInteger(value);
}

function nullableFiniteNumber(value: unknown) {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

export function trainStepEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-step",
    `optimizer=${evidence.optimizerKind ?? "null"}`,
    `parameters=${evidence.parameterCount ?? "null"}`,
    `before=${evidence.beforeStep ?? "null"}`,
    `after=${evidence.afterStep ?? "null"}`,
    `advanced=${evidence.stepAdvanced ?? "null"}`,
    `loss=${evidence.hadLoss ? 1 : 0}`,
    `lossScalar=${scalarSignature(evidence.lossScalar)}`,
    `gradient=${evidence.gradientProvided ? 1 : 0}`,
    `clipNorm=${evidence.clipGradNormApplied ? 1 : 0}`,
    `clipValue=${evidence.clipGradValueApplied ? 1 : 0}`,
    `gradNormBeforeClip=${scalarSignature(evidence.gradNormBeforeClip)}`,
    `gradNormAfterClip=${scalarSignature(evidence.gradNormAfterClip)}`,
    `zeroGrad=${evidence.zeroGradApplied ? 1 : 0}`,
    `cleared=${evidence.gradientsCleared === null ? "null" : evidence.gradientsCleared ? 1 : 0}`,
  ].join("|");
}

export function trainFitStepEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-fit-step",
    `epoch=${evidence.epoch}`,
    `batch=${evidence.batchIndex}`,
    `step=${evidence.step}`,
    `samples=${Array.isArray(evidence.sampleIndices) ? evidence.sampleIndices.join(",") : "null"}`,
    `loss=${scalarSignature(evidence.loss)}`,
    `stepSignature=${evidence.stepEvidence?.signature ?? "null"}`,
  ].join("|");
}

export function trainFitEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-fit",
    `epochs=${evidence.epochs}`,
    `steps=${evidence.steps}`,
    `stopped=${evidence.stoppedEarly ? 1 : 0}`,
    `stopReason=${evidence.stopReason ?? "null"}`,
    `batchCount=${evidence.batchCount ?? "null"}`,
    `sampleCount=${evidence.sampleCount ?? "null"}`,
    `losses=${Array.isArray(evidence.losses) ? evidence.losses.map(scalarSignature).join(",") : ""}`,
    `bestLoss=${scalarSignature(evidence.bestLoss)}`,
    `bestStep=${evidence.bestStep ?? "null"}`,
    `finalLoss=${scalarSignature(evidence.finalLoss)}`,
    `lastStep=${evidence.lastStep?.signature ?? "null"}`,
    `lastLoss=${evidence.lastLoss ? 1 : 0}`,
  ].join("|");
}

export function trainEvaluateStepEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-evaluate-step",
    `batch=${evidence.batchIndex}`,
    `step=${evidence.step}`,
    `samples=${Array.isArray(evidence.sampleIndices) ? evidence.sampleIndices.join(",") : "null"}`,
    `loss=${scalarSignature(evidence.loss)}`,
  ].join("|");
}

export function trainEvaluateEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-evaluate",
    `steps=${evidence.steps}`,
    `stopped=${evidence.stoppedEarly ? 1 : 0}`,
    `batchCount=${evidence.batchCount ?? "null"}`,
    `sampleCount=${evidence.sampleCount ?? "null"}`,
    `losses=${Array.isArray(evidence.losses) ? evidence.losses.map(scalarSignature).join(",") : ""}`,
    `meanLoss=${scalarSignature(evidence.meanLoss)}`,
    `finalLoss=${scalarSignature(evidence.finalLoss)}`,
    `lastLoss=${evidence.lastLoss ? 1 : 0}`,
  ].join("|");
}

export function trainPredictStepEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-predict-step",
    `batch=${evidence.batchIndex}`,
    `step=${evidence.step}`,
    `samples=${Array.isArray(evidence.sampleIndices) ? evidence.sampleIndices.join(",") : "null"}`,
    `output=${evidence.output ? 1 : 0}`,
  ].join("|");
}

export function trainPredictEvidenceSignature(evidence: AnyRecord) {
  return [
    "train-predict",
    `steps=${evidence.steps}`,
    `stopped=${evidence.stoppedEarly ? 1 : 0}`,
    `batchCount=${evidence.batchCount ?? "null"}`,
    `sampleCount=${evidence.sampleCount ?? "null"}`,
    `outputs=${Array.isArray(evidence.outputs) ? evidence.outputs.length : 0}`,
    `lastOutput=${evidence.lastOutput ? 1 : 0}`,
  ].join("|");
}

export function trainStepEvidence(fields: Omit<TrainStepEvidence, "signature">): TrainStepEvidence {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainStepEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainStepEvidence;
}

export function trainFitStepEvidence(fields: Omit<TrainFitStepEvidence, "signature">): TrainFitStepEvidence {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainFitStepEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainFitStepEvidence;
}

export function trainFitEvidence(fields: Omit<TrainFitEvidence, "signature" | "lastLoss"> & { lastLoss: unknown }): TrainFitEvidence {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainFitEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainFitEvidence;
}

export function trainEvaluateStepEvidence(fields: Omit<TrainEvaluateStepEvidence, "signature">): TrainEvaluateStepEvidence {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainEvaluateStepEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainEvaluateStepEvidence;
}

export function trainEvaluateEvidence(fields: Omit<TrainEvaluateEvidence, "signature" | "lastLoss"> & { lastLoss: unknown }): TrainEvaluateEvidence {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainEvaluateEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainEvaluateEvidence;
}

export function trainPredictStepEvidence(fields: Omit<TrainPredictStepEvidence<any>, "signature">): TrainPredictStepEvidence<any> {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainPredictStepEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainPredictStepEvidence;
}

export function trainPredictEvidence(fields: Omit<TrainPredictEvidence<any>, "signature" | "outputs" | "lastOutput" | "last_output"> & {
  outputs: readonly unknown[];
  lastOutput: unknown;
  last_output: unknown;
}): TrainPredictEvidence<any> {
  const evidence = {
    ...fields,
    signature: "",
  };
  evidence.signature = trainPredictEvidenceSignature(evidence);
  return Object.freeze(evidence) as TrainPredictEvidence;
}

export function isTrainStepEvidence(evidence: unknown): evidence is TrainStepEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.step" || typeof evidence.signature !== "string") return false;
  if (evidence.optimizerKind !== null && !["sgd", "adam", "adamw", "rmsprop", "adagrad"].includes(evidence.optimizerKind)) return false;
  if (!nullableNonNegativeSafeInteger(evidence.parameterCount)) return false;
  if (!nullableNonNegativeSafeInteger(evidence.beforeStep)) return false;
  if (!nullableNonNegativeSafeInteger(evidence.afterStep)) return false;
  if (evidence.stepAdvanced !== null && !Number.isSafeInteger(evidence.stepAdvanced)) return false;
  if (typeof evidence.hadLoss !== "boolean") return false;
  if (!nullableFiniteNumber(evidence.lossScalar)) return false;
  if (typeof evidence.gradientProvided !== "boolean") return false;
  if (typeof evidence.clipGradNormApplied !== "boolean") return false;
  if (typeof evidence.clipGradValueApplied !== "boolean") return false;
  if (!nullableFiniteNumber(evidence.gradNormBeforeClip)) return false;
  if (!nullableFiniteNumber(evidence.gradNormAfterClip)) return false;
  if (typeof evidence.zeroGradApplied !== "boolean") return false;
  if (evidence.gradientsCleared !== null && typeof evidence.gradientsCleared !== "boolean") return false;
  return evidence.signature === trainStepEvidenceSignature(evidence);
}

export function requireTrainStepEvidence(evidence: unknown): TrainStepEvidence {
  if (isTrainStepEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainStepEvidence");
}

export const assertTrainStepEvidence = requireTrainStepEvidence;
export const assert_train_step_evidence = requireTrainStepEvidence;

export function matchesTrainStepEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainStepEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_step_evidence_signature = matchesTrainStepEvidenceSignature;

export function isTrainFitStepEvidence(evidence: unknown): evidence is TrainFitStepEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.fit-step" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.epoch)) return false;
  if (!isNonNegativeSafeInteger(evidence.batchIndex)) return false;
  if (!isNonNegativeSafeInteger(evidence.step)) return false;
  if (evidence.sampleIndices !== null) {
    if (!Array.isArray(evidence.sampleIndices) || !Object.isFrozen(evidence.sampleIndices)) return false;
    for (const index of evidence.sampleIndices) {
      if (!isNonNegativeSafeInteger(index)) return false;
    }
  }
  if (evidence.sample_indices !== evidence.sampleIndices) return false;
  if (typeof evidence.loss !== "number" || !Number.isFinite(evidence.loss)) return false;
  if (evidence.stepEvidence !== null && !isTrainStepEvidence(evidence.stepEvidence)) return false;
  return evidence.signature === trainFitStepEvidenceSignature(evidence);
}

export function requireTrainFitStepEvidence(evidence: unknown): TrainFitStepEvidence {
  if (isTrainFitStepEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainFitStepEvidence");
}

export const assertTrainFitStepEvidence = requireTrainFitStepEvidence;
export const assert_train_fit_step_evidence = requireTrainFitStepEvidence;

export function matchesTrainFitStepEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainFitStepEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_fit_step_evidence_signature = matchesTrainFitStepEvidenceSignature;

export function isTrainFitEvidence(evidence: unknown): evidence is TrainFitEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.fit" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.epochs) || !isNonNegativeSafeInteger(evidence.steps)) return false;
  if (typeof evidence.stoppedEarly !== "boolean") return false;
  if (evidence.stopReason !== null && evidence.stopReason !== "max-steps" && evidence.stopReason !== "early-stopping") return false;
  if (evidence.stop_reason !== evidence.stopReason) return false;
  if (!nullableNonNegativeSafeInteger(evidence.batchCount) || evidence.batch_count !== evidence.batchCount) return false;
  if (!nullableNonNegativeSafeInteger(evidence.sampleCount) || evidence.sample_count !== evidence.sampleCount) return false;
  if (!Array.isArray(evidence.losses) || !Object.isFrozen(evidence.losses)) return false;
  for (const loss of evidence.losses) {
    if (typeof loss !== "number" || !Number.isFinite(loss)) return false;
  }
  if (!nullableFiniteNumber(evidence.bestLoss) || evidence.best_loss !== evidence.bestLoss) return false;
  if (!nullableNonNegativeSafeInteger(evidence.bestStep) || evidence.best_step !== evidence.bestStep) return false;
  if (!nullableFiniteNumber(evidence.finalLoss)) return false;
  if (evidence.lastStep !== null && !isTrainStepEvidence(evidence.lastStep)) return false;
  return evidence.signature === trainFitEvidenceSignature(evidence);
}

export function requireTrainFitEvidence(evidence: unknown): TrainFitEvidence {
  if (isTrainFitEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainFitEvidence");
}

export const assertTrainFitEvidence = requireTrainFitEvidence;
export const assert_train_fit_evidence = requireTrainFitEvidence;

export function matchesTrainFitEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainFitEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_fit_evidence_signature = matchesTrainFitEvidenceSignature;

export function isTrainEvaluateStepEvidence(evidence: unknown): evidence is TrainEvaluateStepEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.evaluate-step" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.batchIndex)) return false;
  if (!isNonNegativeSafeInteger(evidence.step)) return false;
  if (evidence.sampleIndices !== null) {
    if (!Array.isArray(evidence.sampleIndices) || !Object.isFrozen(evidence.sampleIndices)) return false;
    for (const index of evidence.sampleIndices) {
      if (!isNonNegativeSafeInteger(index)) return false;
    }
  }
  if (evidence.sample_indices !== evidence.sampleIndices) return false;
  if (typeof evidence.loss !== "number" || !Number.isFinite(evidence.loss)) return false;
  return evidence.signature === trainEvaluateStepEvidenceSignature(evidence);
}

export function requireTrainEvaluateStepEvidence(evidence: unknown): TrainEvaluateStepEvidence {
  if (isTrainEvaluateStepEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainEvaluateStepEvidence");
}

export const assertTrainEvaluateStepEvidence = requireTrainEvaluateStepEvidence;
export const assert_train_evaluate_step_evidence = requireTrainEvaluateStepEvidence;

export function matchesTrainEvaluateStepEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainEvaluateStepEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_evaluate_step_evidence_signature = matchesTrainEvaluateStepEvidenceSignature;

export function isTrainEvaluateEvidence(evidence: unknown): evidence is TrainEvaluateEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.evaluate" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.steps)) return false;
  if (typeof evidence.stoppedEarly !== "boolean") return false;
  if (!nullableNonNegativeSafeInteger(evidence.batchCount) || evidence.batch_count !== evidence.batchCount) return false;
  if (!nullableNonNegativeSafeInteger(evidence.sampleCount) || evidence.sample_count !== evidence.sampleCount) return false;
  if (!Array.isArray(evidence.losses) || !Object.isFrozen(evidence.losses)) return false;
  for (const loss of evidence.losses) {
    if (typeof loss !== "number" || !Number.isFinite(loss)) return false;
  }
  if (!nullableFiniteNumber(evidence.meanLoss) || evidence.mean_loss !== evidence.meanLoss) return false;
  if (!nullableFiniteNumber(evidence.finalLoss) || evidence.final_loss !== evidence.finalLoss) return false;
  return evidence.signature === trainEvaluateEvidenceSignature(evidence);
}

export function requireTrainEvaluateEvidence(evidence: unknown): TrainEvaluateEvidence {
  if (isTrainEvaluateEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainEvaluateEvidence");
}

export const assertTrainEvaluateEvidence = requireTrainEvaluateEvidence;
export const assert_train_evaluate_evidence = requireTrainEvaluateEvidence;

export function matchesTrainEvaluateEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainEvaluateEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_evaluate_evidence_signature = matchesTrainEvaluateEvidenceSignature;

export function isTrainPredictStepEvidence(evidence: unknown): evidence is TrainPredictStepEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.predict-step" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.batchIndex)) return false;
  if (!isNonNegativeSafeInteger(evidence.step)) return false;
  if (evidence.sampleIndices !== null) {
    if (!Array.isArray(evidence.sampleIndices) || !Object.isFrozen(evidence.sampleIndices)) return false;
    for (const index of evidence.sampleIndices) {
      if (!isNonNegativeSafeInteger(index)) return false;
    }
  }
  if (evidence.sample_indices !== evidence.sampleIndices) return false;
  if (!isRecord(evidence.output)) return false;
  return evidence.signature === trainPredictStepEvidenceSignature(evidence);
}

export function requireTrainPredictStepEvidence(evidence: unknown): TrainPredictStepEvidence {
  if (isTrainPredictStepEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainPredictStepEvidence");
}

export const assertTrainPredictStepEvidence = requireTrainPredictStepEvidence;
export const assert_train_predict_step_evidence = requireTrainPredictStepEvidence;

export function matchesTrainPredictStepEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainPredictStepEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_predict_step_evidence_signature = matchesTrainPredictStepEvidenceSignature;

export function isTrainPredictEvidence(evidence: unknown): evidence is TrainPredictEvidence {
  if (!isRecord(evidence) || !Object.isFrozen(evidence)) return false;
  if (evidence.kind !== "zgml.train.predict" || typeof evidence.signature !== "string") return false;
  if (!isNonNegativeSafeInteger(evidence.steps)) return false;
  if (typeof evidence.stoppedEarly !== "boolean") return false;
  if (!nullableNonNegativeSafeInteger(evidence.batchCount) || evidence.batch_count !== evidence.batchCount) return false;
  if (!nullableNonNegativeSafeInteger(evidence.sampleCount) || evidence.sample_count !== evidence.sampleCount) return false;
  if (!Array.isArray(evidence.outputs) || !Object.isFrozen(evidence.outputs)) return false;
  for (const output of evidence.outputs) {
    if (!isRecord(output)) return false;
  }
  if (evidence.lastOutput !== null && !isRecord(evidence.lastOutput)) return false;
  if (evidence.last_output !== evidence.lastOutput) return false;
  return evidence.signature === trainPredictEvidenceSignature(evidence);
}

export function requireTrainPredictEvidence(evidence: unknown): TrainPredictEvidence {
  if (isTrainPredictEvidence(evidence)) return evidence;
  throw new Error("expected frozen TrainPredictEvidence");
}

export const assertTrainPredictEvidence = requireTrainPredictEvidence;
export const assert_train_predict_evidence = requireTrainPredictEvidence;

export function matchesTrainPredictEvidenceSignature(evidence: unknown, signature: unknown) {
  return isTrainPredictEvidence(evidence) && typeof signature === "string" && evidence.signature === signature;
}

export const matches_train_predict_evidence_signature = matchesTrainPredictEvidenceSignature;
