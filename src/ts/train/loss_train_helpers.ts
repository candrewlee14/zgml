"use strict";

import type {
  OptimizerStateKind,
  CompiledTrainingPlan,
  TrainEvaluateContext,
  TrainEvaluateOptions,
  TrainPredictContext,
  TrainPredictEvidence,
  TrainPredictOptions,
  TrainPredictStepEvidence,
  TrainFitContext,
  TrainFitOptions,
  TrainLossStepOptions,
  TrainStepOptions,
  TrainStepEvidence,
  ZeroGradOptions,
} from "../public_api.js";
import {
  assert_train_evaluate_evidence,
  assert_train_evaluate_step_evidence,
  assert_train_fit_evidence,
  assert_train_fit_step_evidence,
  assert_train_predict_evidence,
  assert_train_predict_step_evidence,
  assert_train_step_evidence,
  assertTrainEvaluateEvidence,
  assertTrainEvaluateStepEvidence,
  assertTrainFitEvidence,
  assertTrainFitStepEvidence,
  assertTrainPredictEvidence,
  assertTrainPredictStepEvidence,
  assertTrainStepEvidence,
  isTrainEvaluateEvidence,
  isTrainEvaluateStepEvidence,
  isTrainFitEvidence,
  isTrainFitStepEvidence,
  isTrainPredictEvidence,
  isTrainPredictStepEvidence,
  isTrainStepEvidence,
  matches_train_evaluate_evidence_signature,
  matches_train_evaluate_step_evidence_signature,
  matches_train_fit_evidence_signature,
  matches_train_fit_step_evidence_signature,
  matches_train_predict_evidence_signature,
  matches_train_predict_step_evidence_signature,
  matches_train_step_evidence_signature,
  matchesTrainEvaluateEvidenceSignature,
  matchesTrainEvaluateStepEvidenceSignature,
  matchesTrainFitEvidenceSignature,
  matchesTrainFitStepEvidenceSignature,
  matchesTrainPredictEvidenceSignature,
  matchesTrainPredictStepEvidenceSignature,
  matchesTrainStepEvidenceSignature,
  requireTrainEvaluateEvidence,
  requireTrainEvaluateStepEvidence,
  requireTrainFitEvidence,
  requireTrainFitStepEvidence,
  requireTrainPredictEvidence,
  requireTrainPredictStepEvidence,
  requireTrainStepEvidence,
  trainEvaluateEvidence,
  trainEvaluateStepEvidence,
  trainFitEvidence,
  trainFitStepEvidence,
  trainPredictEvidence,
  trainPredictStepEvidence,
  trainStepEvidence,
} from "./evidence.js";

export type {
  TrainEvaluateContext,
  TrainEvaluateEvidence,
  TrainEvaluateOptions,
  TrainEvaluateStepEvidence,
  TrainPredictContext,
  TrainPredictEvidence,
  TrainPredictOptions,
  TrainPredictStepEvidence,
  TrainFitContext,
  TrainFitEvidence,
  TrainFitOptions,
  TrainFitStepEvidence,
  TrainLossStepOptions,
  TrainStepOptions,
  TrainStepEvidence,
  ZeroGradOptions,
} from "../public_api.js";
export {
  assert_train_evaluate_evidence,
  assert_train_evaluate_step_evidence,
  assert_train_fit_evidence,
  assert_train_fit_step_evidence,
  assert_train_predict_evidence,
  assert_train_predict_step_evidence,
  assert_train_step_evidence,
  assertTrainEvaluateEvidence,
  assertTrainEvaluateStepEvidence,
  assertTrainFitEvidence,
  assertTrainFitStepEvidence,
  assertTrainPredictEvidence,
  assertTrainPredictStepEvidence,
  assertTrainStepEvidence,
  isTrainEvaluateEvidence,
  isTrainEvaluateStepEvidence,
  isTrainFitEvidence,
  isTrainFitStepEvidence,
  isTrainPredictEvidence,
  isTrainPredictStepEvidence,
  isTrainStepEvidence,
  matches_train_evaluate_evidence_signature,
  matches_train_evaluate_step_evidence_signature,
  matches_train_fit_evidence_signature,
  matches_train_fit_step_evidence_signature,
  matches_train_predict_evidence_signature,
  matches_train_predict_step_evidence_signature,
  matches_train_step_evidence_signature,
  matchesTrainEvaluateEvidenceSignature,
  matchesTrainEvaluateStepEvidenceSignature,
  matchesTrainFitEvidenceSignature,
  matchesTrainFitStepEvidenceSignature,
  matchesTrainPredictEvidenceSignature,
  matchesTrainPredictStepEvidenceSignature,
  matchesTrainStepEvidenceSignature,
  requireTrainEvaluateEvidence,
  requireTrainEvaluateStepEvidence,
  requireTrainFitEvidence,
  requireTrainFitStepEvidence,
  requireTrainPredictEvidence,
  requireTrainPredictStepEvidence,
  requireTrainStepEvidence,
  trainEvaluateEvidenceSignature,
  trainEvaluateStepEvidenceSignature,
  trainFitEvidenceSignature,
  trainFitStepEvidenceSignature,
  trainPredictEvidenceSignature,
  trainPredictStepEvidenceSignature,
  trainStepEvidenceSignature,
} from "./evidence.js";

type AnyRecord = Record<string, any>;

type LossTrainHookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

type LossTrainOptionsRecord = Readonly<Record<string, unknown>>;
type LossTrainTensorConstructOptions = Readonly<Record<string, unknown>>;
type LossReduction = "mean" | "sum";

type LossTrainTensor = AnyRecord & {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  requiresGrad: boolean;
  requires_grad?: boolean;
  grad: Float32Array | null;
  add(other: unknown): LossTrainTensor;
  sub(other: unknown): LossTrainTensor;
  mul(other: unknown): LossTrainTensor;
  div(other: unknown): LossTrainTensor;
  maximum(other: unknown): LossTrainTensor;
  minimum(other: unknown): LossTrainTensor;
  abs(): LossTrainTensor;
  sqr(): LossTrainTensor;
  log(): LossTrainTensor;
  clamp(min?: number | null, max?: number | null): LossTrainTensor;
  sum(): LossTrainTensor;
  mean(): LossTrainTensor;
  backward(gradient?: unknown): void;
  item?(): number;
};

type LossTrainTensorConstructor = new (
  data: unknown,
  shape?: readonly number[],
  options?: LossTrainTensorConstructOptions,
) => LossTrainTensor;

type LossTrainParameterTensor = {
  grad?: Float32Array | null;
  requiresGrad?: boolean;
  requires_grad?: boolean;
};

type LossTrainParameter = AnyRecord & {
  name?: string;
  data: Float32Array;
  grad?: Float32Array | null;
  tensor?: LossTrainParameterTensor;
  requiresGrad?: boolean;
  requires_grad?: boolean;
};

type LossTrainOptimizerSnapshot = Readonly<Record<string, unknown> & {
  kind?: unknown;
  step?: unknown;
  paramCount?: unknown;
}>;

type LossTrainOptimizer = {
  params?: readonly LossTrainParameter[];
  step(): void;
  zeroGrad?(options?: ZeroGradOptions): void;
  stateDict?(): LossTrainOptimizerSnapshot;
};

type CompiledTrainingStep = AnyRecord & {
  step(input: unknown, target: unknown): unknown;
  inputShape?: () => readonly number[];
  outputShape?: () => readonly number[];
  plan?: () => CompiledTrainingPlan;
  compileEvidence?: () => CompiledTrainingPlan;
  compile_evidence?: () => CompiledTrainingPlan;
};

type CompileTrainingStepHook = LossTrainHookCallback<[
  model: unknown,
  optimizer: unknown,
  options?: LossTrainOptionsRecord,
], unknown>;

type LossTrainStepOptions = Omit<TrainStepOptions, "loss"> & {
  loss?: LossTrainTensor | null;
};

type GradientClipEvidence = Readonly<{
  clipGradNormApplied: boolean;
  clipGradValueApplied: boolean;
  gradNormBeforeClip: number | null;
  gradNormAfterClip: number | null;
}>;

type LossTrainOptimizerEvidence = Readonly<{
  kind: OptimizerStateKind | null;
  step: number | null;
  paramCount: number | null;
}>;

type LossTrainShapeSource = Readonly<{
  shape?: readonly number[];
}>;

export type LossTrainHelpersOptions = Readonly<{
  Tensor: LossTrainTensorConstructor;
  f32: LossTrainHookCallback<[value: unknown], Float32Array>;
  f32WithLength: LossTrainHookCallback<[value: unknown, length: number, label: string], Float32Array>;
  indexValues: LossTrainHookCallback<[value: unknown, label: string], ArrayLike<number>>;
  addTensorGrad: LossTrainHookCallback<[tensor: LossTrainTensor, grad: Float32Array], unknown>;
  zeroGrad: LossTrainHookCallback<[paramsOrModule: unknown, options?: ZeroGradOptions], unknown>;
  resolveParameters: LossTrainHookCallback<[paramsOrModule: unknown], LossTrainParameter[]>;
  isGradEnabled?: LossTrainHookCallback<[], boolean>;
  compileTrainingStep?: CompileTrainingStepHook;
}>;

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

export function createLossTrainHelpers(options: LossTrainHelpersOptions) {
  const TensorClass = options.Tensor;
  const f32 = options.f32;
  const f32WithLength = options.f32WithLength;
  const indexValues = options.indexValues;
  const addTensorGrad = options.addTensorGrad;
  const zeroGrad = options.zeroGrad;
  const resolveParameters = options.resolveParameters;
  const compileTrainingStep = typeof options.compileTrainingStep === "function"
    ? options.compileTrainingStep
    : null;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : () => true;
  if (
    typeof TensorClass !== "function" ||
    typeof f32 !== "function" ||
    typeof f32WithLength !== "function" ||
    typeof indexValues !== "function" ||
    typeof addTensorGrad !== "function" ||
    typeof zeroGrad !== "function" ||
    typeof resolveParameters !== "function"
  ) {
    throw new Error("loss/train helpers require Tensor, f32, f32WithLength, indexValues, addTensorGrad, zeroGrad, and resolveParameters");
  }

  function isTensor(value: unknown): value is LossTrainTensor {
    return value instanceof TensorClass;
  }

  function tensorShape(value: unknown) {
    if (!isRecord(value)) return undefined;
    const shape = (value as LossTrainShapeSource).shape;
    return Array.isArray(shape) ? shape : undefined;
  }

  function isOptimizerStateKind(value: unknown): value is OptimizerStateKind {
    return value === "sgd" || value === "adam" || value === "adamw" || value === "rmsprop" || value === "adagrad";
  }

  function parameterRequiresGrad(param: LossTrainParameter) {
    return param?.tensor ? param.tensor.requiresGrad !== false : param?.requiresGrad !== false;
  }

  function lossReduction(lossOptions: LossTrainOptionsRecord = {}, label: string): LossReduction {
    const reduction = lossOptions.reduction ?? "mean";
    if (reduction === "mean" || reduction === "sum") return reduction;
    throw new Error(`${label} reduction must be "mean" or "sum", got ${reduction}`);
  }

  function reduceTensorLoss(lossTensor: LossTrainTensor, reduction: LossReduction) {
    return reduction === "sum" ? lossTensor.sum() : lossTensor.mean();
  }

  function reduceScalarLoss(total: number, count: number, reduction: LossReduction) {
    return reduction === "sum" ? total : total / count;
  }

  function meanSquaredError(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const reduction = lossReduction(lossOptions, "MSELoss");
    if (isTensor(predValues) || isTensor(targetValues)) {
      const pred = isTensor(predValues) ? predValues : new TensorClass(predValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, pred.length, "mse target"), pred.shape);
      return reduceTensorLoss(pred.sub(target).sqr(), reduction);
    }
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "mse target");
    let acc = 0;
    for (let i = 0; i < pred.length; i += 1) {
      const d = pred[i] - target[i];
      acc += d * d;
    }
    return reduceScalarLoss(acc, pred.length, reduction);
  }

  function meanAbsoluteError(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const reduction = lossReduction(lossOptions, "L1Loss");
    if (isTensor(predValues) || isTensor(targetValues)) {
      const pred = isTensor(predValues) ? predValues : new TensorClass(predValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, pred.length, "l1 target"), pred.shape);
      return reduceTensorLoss(pred.sub(target).abs(), reduction);
    }
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "l1 target");
    let acc = 0;
    for (let i = 0; i < pred.length; i += 1) acc += Math.abs(pred[i] - target[i]);
    return reduceScalarLoss(acc, pred.length, reduction);
  }

  function finitePositiveLossParameter(value: unknown, label: string) {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue) || numberValue <= 0) {
      throw new Error(`${label} must be a positive finite number, got ${value}`);
    }
    return numberValue;
  }

  function huber(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const delta = finitePositiveLossParameter(lossOptions.delta ?? 1, "HuberLoss delta");
    const reduction = lossReduction(lossOptions, "HuberLoss");
    if (isTensor(predValues) || isTensor(targetValues)) {
      const pred = isTensor(predValues) ? predValues : new TensorClass(predValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, pred.length, "huber target"), pred.shape);
      const absError = pred.sub(target).abs();
      const quadratic = absError.minimum(delta);
      const linear = absError.sub(quadratic);
      return reduceTensorLoss(quadratic.sqr().mul(0.5).add(linear.mul(delta)), reduction);
    }
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "huber target");
    let acc = 0;
    for (let i = 0; i < pred.length; i += 1) {
      const absError = Math.abs(pred[i] - target[i]);
      const quadratic = Math.min(absError, delta);
      const linear = absError - quadratic;
      acc += 0.5 * quadratic * quadratic + delta * linear;
    }
    return reduceScalarLoss(acc, pred.length, reduction);
  }

  function smoothL1(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const beta = finitePositiveLossParameter(lossOptions.beta ?? 1, "SmoothL1Loss beta");
    const reduction = lossReduction(lossOptions, "SmoothL1Loss");
    if (isTensor(predValues) || isTensor(targetValues)) {
      const pred = isTensor(predValues) ? predValues : new TensorClass(predValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, pred.length, "smooth_l1 target"), pred.shape);
      const absError = pred.sub(target).abs();
      const quadratic = absError.minimum(beta);
      const linear = absError.sub(quadratic);
      return reduceTensorLoss(quadratic.sqr().mul(0.5).div(beta).add(linear), reduction);
    }
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "smooth_l1 target");
    let acc = 0;
    for (let i = 0; i < pred.length; i += 1) {
      const absError = Math.abs(pred[i] - target[i]);
      const quadratic = Math.min(absError, beta);
      const linear = absError - quadratic;
      acc += (0.5 * quadratic * quadratic) / beta + linear;
    }
    return reduceScalarLoss(acc, pred.length, reduction);
  }

  function binaryCrossEntropy(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const eps = lossOptions.eps === undefined ? 1e-7 : Number(lossOptions.eps);
    const reduction = lossReduction(lossOptions, "BCELoss");
    if (!Number.isFinite(eps) || eps <= 0 || eps >= 0.5) {
      throw new Error(`BCELoss eps must be a finite number in (0, 0.5), got ${lossOptions.eps}`);
    }
    if (isTensor(predValues) || isTensor(targetValues)) {
      const pred = isTensor(predValues) ? predValues : new TensorClass(predValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, pred.length, "bce target"), pred.shape);
      const clipped = pred.clamp(eps, 1 - eps);
      const positive = target.mul(clipped.log());
      const negative = target.mul(-1).add(1).mul(clipped.mul(-1).add(1).log());
      return reduceTensorLoss(positive.add(negative).mul(-1), reduction);
    }
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "bce target");
    let acc = 0;
    for (let i = 0; i < pred.length; i += 1) {
      const y = target[i];
      if (y < 0 || y > 1) throw new Error(`BCELoss target ${y} at position ${i} must be in [0, 1]`);
      const p = Math.min(1 - eps, Math.max(eps, pred[i]));
      acc += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    }
    return reduceScalarLoss(acc, pred.length, reduction);
  }

  function binaryCrossEntropyWithLogits(logitValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
    const reduction = lossReduction(lossOptions, "BCEWithLogitsLoss");
    if (isTensor(logitValues) || isTensor(targetValues)) {
      const logits = isTensor(logitValues) ? logitValues : new TensorClass(logitValues, tensorShape(targetValues));
      const target = isTensor(targetValues)
        ? targetValues
        : new TensorClass(f32WithLength(targetValues, logits.length, "bce-with-logits target"), logits.shape);
      const maxLogits = logits.maximum(0);
      const linear = logits.mul(target);
      const softplus = logits.abs().mul(-1).exp().add(1).log();
      return reduceTensorLoss(maxLogits.sub(linear).add(softplus), reduction);
    }
    const logits = f32(logitValues);
    const target = f32WithLength(targetValues, logits.length, "bce-with-logits target");
    let acc = 0;
    for (let i = 0; i < logits.length; i += 1) {
      const y = target[i];
      if (y < 0 || y > 1) throw new Error(`BCEWithLogitsLoss target ${y} at position ${i} must be in [0, 1]`);
      const x = logits[i];
      acc += Math.max(x, 0) - x * y + Math.log1p(Math.exp(-Math.abs(x)));
    }
    return reduceScalarLoss(acc, logits.length, reduction);
  }

  function classTargets(classes: unknown) {
    const values = indexValues(classes, "class targets");
    const out = new Uint32Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new Error(`class target ${value} at position ${i} must be a non-negative integer`);
      }
      out[i] = value;
    }
    return out;
  }

  function crossEntropy(logitValues: unknown, targetValues: unknown, entropyOptions: LossTrainOptionsRecord = {}) {
    const reduction = lossReduction(entropyOptions, "CrossEntropyLoss");
    const logitsTensor = isTensor(logitValues) ? logitValues : null;
    const logits = f32(logitValues);
    const targets = classTargets(targetValues);
    const batch = targets.length;
    const classesOption = entropyOptions.classes ?? entropyOptions.numClasses;
    const classes = classesOption === undefined ? (batch === 0 ? 0 : logits.length / batch) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`crossEntropy classes must be a positive safe integer, got ${classes}`);
    }
    if (batch === 0 || logits.length !== batch * classes) {
      throw new Error(`crossEntropy logits length ${logits.length} must equal targets length ${batch} * classes ${classes}`);
    }

    let total = 0;
    const needsGrad = gradModeEnabled() && Boolean(logitsTensor?.requiresGrad);
    const logitsGrad = needsGrad ? new Float32Array(logits.length) : null;
    for (let b = 0; b < batch; b += 1) {
      const target = targets[b];
      if (target >= classes) {
        throw new Error(`class target ${target} at position ${b} is out of range 0..${classes - 1}`);
      }
      const base = b * classes;
      let maxLogit = logits[base];
      for (let c = 1; c < classes; c += 1) maxLogit = Math.max(maxLogit, logits[base + c]);
      let denom = 0;
      for (let c = 0; c < classes; c += 1) denom += Math.exp(logits[base + c] - maxLogit);
      total += -(logits[base + target] - maxLogit - Math.log(denom));
      if (logitsGrad) {
        const gradientScale = reduction === "mean" ? 1 / batch : 1;
        for (let c = 0; c < classes; c += 1) {
          const prob = Math.exp(logits[base + c] - maxLogit) / denom;
          logitsGrad[base + c] = (prob - (c === target ? 1 : 0)) * gradientScale;
        }
      }
    }
    const value = reduceScalarLoss(total, batch, reduction);
    if (!logitsTensor) return value;
    return new TensorClass(Float32Array.of(value), [1], {
      requiresGrad: needsGrad,
      prev: needsGrad ? [logitsTensor] : [],
      backward: (grad: Float32Array | null) => {
        if (!grad || !logitsGrad) return;
        const scaled = new Float32Array(logitsGrad.length);
        for (let i = 0; i < scaled.length; i += 1) scaled[i] = logitsGrad[i] * grad[0];
        addTensorGrad(logitsTensor, scaled);
      },
    });
  }

  function negativeLogLikelihood(logProbabilityValues: unknown, targetValues: unknown, nllOptions: LossTrainOptionsRecord = {}) {
    const reduction = lossReduction(nllOptions, "NLLLoss");
    const logProbTensor = isTensor(logProbabilityValues) ? logProbabilityValues : null;
    const logProbabilities = f32(logProbabilityValues);
    const targets = classTargets(targetValues);
    const batch = targets.length;
    const classesOption = nllOptions.classes ?? nllOptions.numClasses;
    const classes = classesOption === undefined ? (batch === 0 ? 0 : logProbabilities.length / batch) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`nllLoss classes must be a positive safe integer, got ${classes}`);
    }
    if (batch === 0 || logProbabilities.length !== batch * classes) {
      throw new Error(`nllLoss log-probabilities length ${logProbabilities.length} must equal targets length ${batch} * classes ${classes}`);
    }

    let total = 0;
    const needsGrad = gradModeEnabled() && Boolean(logProbTensor?.requiresGrad);
    const logProbGrad = needsGrad ? new Float32Array(logProbabilities.length) : null;
    for (let b = 0; b < batch; b += 1) {
      const target = targets[b];
      if (target >= classes) {
        throw new Error(`class target ${target} at position ${b} is out of range 0..${classes - 1}`);
      }
      const index = b * classes + target;
      total += -logProbabilities[index];
      if (logProbGrad) logProbGrad[index] = reduction === "mean" ? -1 / batch : -1;
    }
    const value = reduceScalarLoss(total, batch, reduction);
    if (!logProbTensor) return value;
    return new TensorClass(Float32Array.of(value), [1], {
      requiresGrad: needsGrad,
      prev: needsGrad ? [logProbTensor] : [],
      backward: (grad: Float32Array | null) => {
        if (!grad || !logProbGrad) return;
        const scaled = new Float32Array(logProbGrad.length);
        for (let i = 0; i < scaled.length; i += 1) scaled[i] = logProbGrad[i] * grad[0];
        addTensorGrad(logProbTensor, scaled);
      },
    });
  }

  function classificationAccuracy(logitValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const logits = f32(logitValues);
    const targets = classTargets(targetValues);
    const batch = targets.length;
    const classesOption = metricOptions.classes ?? metricOptions.numClasses;
    const classes = classesOption === undefined ? (batch === 0 ? 0 : logits.length / batch) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`train.accuracy classes must be a positive safe integer, got ${classes}`);
    }
    if (batch === 0 || logits.length !== batch * classes) {
      throw new Error(`train.accuracy logits length ${logits.length} must equal targets length ${batch} * classes ${classes}`);
    }

    let correct = 0;
    for (let b = 0; b < batch; b += 1) {
      const target = targets[b];
      if (target >= classes) {
        throw new Error(`class target ${target} at position ${b} is out of range 0..${classes - 1}`);
      }
      const base = b * classes;
      let prediction = 0;
      let maxLogit = logits[base];
      for (let c = 1; c < classes; c += 1) {
        const value = logits[base + c];
        if (value > maxLogit) {
          maxLogit = value;
          prediction = c;
        }
      }
      if (prediction === target) correct += 1;
    }
    return correct / batch;
  }

  function classPredictions(logitValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const logits = f32(logitValues);
    const classesOption = metricOptions.classes ?? metricOptions.numClasses;
    const classes = classesOption === undefined ? (logits.length === 0 ? 0 : logits.length) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`train.classPredictions classes must be a positive safe integer, got ${classesOption ?? classes}`);
    }
    if (logits.length === 0 || logits.length % classes !== 0) {
      throw new Error(`train.classPredictions logits length ${logits.length} must be divisible by classes ${classes}`);
    }
    const batch = logits.length / classes;
    const out = new Uint32Array(batch);
    for (let b = 0; b < batch; b += 1) {
      const base = b * classes;
      let prediction = 0;
      let maxLogit = logits[base];
      for (let c = 1; c < classes; c += 1) {
        const value = logits[base + c];
        if (value > maxLogit) {
          maxLogit = value;
          prediction = c;
        }
      }
      out[b] = prediction;
    }
    return out;
  }

  function topKAccuracy(logitValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const logits = f32(logitValues);
    const targets = classTargets(targetValues);
    const batch = targets.length;
    const classesOption = metricOptions.classes ?? metricOptions.numClasses;
    const classes = classesOption === undefined ? (batch === 0 ? 0 : logits.length / batch) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`train.topKAccuracy classes must be a positive safe integer, got ${classes}`);
    }
    if (batch === 0 || logits.length !== batch * classes) {
      throw new Error(`train.topKAccuracy logits length ${logits.length} must equal targets length ${batch} * classes ${classes}`);
    }
    const kOption = metricOptions.k ?? metricOptions.topK ?? metricOptions.top_k;
    const k = kOption === undefined ? 1 : Number(kOption);
    if (!Number.isSafeInteger(k) || k <= 0 || k > classes) {
      throw new Error(`train.topKAccuracy k must be a positive safe integer <= classes ${classes}, got ${kOption}`);
    }

    let correct = 0;
    for (let b = 0; b < batch; b += 1) {
      const target = targets[b];
      if (target >= classes) {
        throw new Error(`class target ${target} at position ${b} is out of range 0..${classes - 1}`);
      }
      const base = b * classes;
      let rank = 1;
      const targetLogit = logits[base + target];
      for (let c = 0; c < classes; c += 1) {
        if (c !== target && logits[base + c] > targetLogit) rank += 1;
      }
      if (rank <= k) correct += 1;
    }
    return correct / batch;
  }

  function confusionMatrix(logitValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const logits = f32(logitValues);
    const targets = classTargets(targetValues);
    const batch = targets.length;
    const classesOption = metricOptions.classes ?? metricOptions.numClasses;
    const classes = classesOption === undefined ? (batch === 0 ? 0 : logits.length / batch) : Number(classesOption);
    if (!Number.isSafeInteger(classes) || classes <= 0) {
      throw new Error(`train.confusionMatrix classes must be a positive safe integer, got ${classes}`);
    }
    if (batch === 0 || logits.length !== batch * classes) {
      throw new Error(`train.confusionMatrix logits length ${logits.length} must equal targets length ${batch} * classes ${classes}`);
    }
    const matrix = Array.from({ length: classes }, () => Array.from({ length: classes }, () => 0));
    for (let b = 0; b < batch; b += 1) {
      const target = targets[b];
      if (target >= classes) {
        throw new Error(`class target ${target} at position ${b} is out of range 0..${classes - 1}`);
      }
      const base = b * classes;
      let prediction = 0;
      let maxLogit = logits[base];
      for (let c = 1; c < classes; c += 1) {
        const value = logits[base + c];
        if (value > maxLogit) {
          maxLogit = value;
          prediction = c;
        }
      }
      matrix[target][prediction] += 1;
    }
    return Object.freeze(matrix.map((row) => Object.freeze(row)));
  }

  function classificationReport(logitValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const matrix = confusionMatrix(logitValues, targetValues, metricOptions);
    const classes = matrix.length;
    let total = 0;
    let correct = 0;
    let macroPrecision = 0;
    let macroRecall = 0;
    let macroF1 = 0;
    let weightedPrecision = 0;
    let weightedRecall = 0;
    let weightedF1 = 0;
    const perClass = [];
    for (let klass = 0; klass < classes; klass += 1) {
      let truePositive = matrix[klass][klass];
      let falsePositive = 0;
      let falseNegative = 0;
      let support = 0;
      for (let other = 0; other < classes; other += 1) {
        support += matrix[klass][other];
        if (other !== klass) {
          falseNegative += matrix[klass][other];
          falsePositive += matrix[other][klass];
        }
      }
      total += support;
      correct += truePositive;
      const precisionDenom = truePositive + falsePositive;
      const recallDenom = truePositive + falseNegative;
      const precision = precisionDenom === 0 ? 0 : truePositive / precisionDenom;
      const recall = recallDenom === 0 ? 0 : truePositive / recallDenom;
      const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      macroPrecision += precision;
      macroRecall += recall;
      macroF1 += f1;
      weightedPrecision += precision * support;
      weightedRecall += recall * support;
      weightedF1 += f1 * support;
      perClass.push(Object.freeze({
        classIndex: klass,
        class_index: klass,
        precision,
        recall,
        f1,
        support,
      }));
    }
    const divisor = classes === 0 ? 1 : classes;
    const sampleDivisor = total === 0 ? 1 : total;
    return Object.freeze({
      accuracy: total === 0 ? 0 : correct / total,
      macroPrecision: macroPrecision / divisor,
      macro_precision: macroPrecision / divisor,
      macroRecall: macroRecall / divisor,
      macro_recall: macroRecall / divisor,
      macroF1: macroF1 / divisor,
      macro_f1: macroF1 / divisor,
      weightedPrecision: weightedPrecision / sampleDivisor,
      weighted_precision: weightedPrecision / sampleDivisor,
      weightedRecall: weightedRecall / sampleDivisor,
      weighted_recall: weightedRecall / sampleDivisor,
      weightedF1: weightedF1 / sampleDivisor,
      weighted_f1: weightedF1 / sampleDivisor,
      support: total,
      classes,
      confusionMatrix: matrix,
      confusion_matrix: matrix,
      perClass: Object.freeze(perClass),
      per_class: Object.freeze(perClass),
    });
  }

  function finiteThresholdOption(value: unknown, fallback: number, label: string) {
    const threshold = value === undefined ? fallback : Number(value);
    if (!Number.isFinite(threshold)) throw new Error(`${label} threshold must be finite, got ${value}`);
    return threshold;
  }

  function binaryAccuracy(predValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const pred = f32(predValues);
    const target = f32WithLength(targetValues, pred.length, "binary accuracy target");
    if (pred.length === 0) throw new Error("train.binaryAccuracy requires at least one prediction");
    const threshold = finiteThresholdOption(metricOptions.threshold, 0.5, "train.binaryAccuracy");
    let correct = 0;
    for (let i = 0; i < pred.length; i += 1) {
      const y = target[i];
      if (y !== 0 && y !== 1) throw new Error(`binary accuracy target ${y} at position ${i} must be 0 or 1`);
      const predicted = pred[i] >= threshold ? 1 : 0;
      if (predicted === y) correct += 1;
    }
    return correct / pred.length;
  }

  function binaryLogitsAccuracy(logitValues: unknown, targetValues: unknown, metricOptions: LossTrainOptionsRecord = {}) {
    const logits = f32(logitValues);
    const target = f32WithLength(targetValues, logits.length, "binary logits accuracy target");
    if (logits.length === 0) throw new Error("train.binaryLogitsAccuracy requires at least one logit");
    const threshold = finiteThresholdOption(metricOptions.threshold, 0, "train.binaryLogitsAccuracy");
    let correct = 0;
    for (let i = 0; i < logits.length; i += 1) {
      const y = target[i];
      if (y !== 0 && y !== 1) throw new Error(`binary logits accuracy target ${y} at position ${i} must be 0 or 1`);
      const predicted = logits[i] >= threshold ? 1 : 0;
      if (predicted === y) correct += 1;
    }
    return correct / logits.length;
  }

  class MSELoss {
    readonly kind = "mse-loss";
    readonly reduction: LossReduction;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      this.reduction = lossReduction(lossOptions, "MSELoss");
    }

    forward(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return meanSquaredError(predValues, targetValues, { ...lossOptions, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }

    __call__(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }
  }

  class L1Loss {
    readonly kind = "l1-loss";
    readonly reduction: LossReduction;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      this.reduction = lossReduction(lossOptions, "L1Loss");
    }

    forward(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return meanAbsoluteError(predValues, targetValues, { ...lossOptions, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }

    __call__(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }
  }

  class HuberLoss {
    readonly kind = "huber-loss";
    readonly reduction: LossReduction;
    readonly delta: number;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      this.delta = finitePositiveLossParameter(lossOptions.delta ?? 1, "HuberLoss delta");
      this.reduction = lossReduction(lossOptions, "HuberLoss");
    }

    forward(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return huber(predValues, targetValues, { ...lossOptions, delta: lossOptions.delta ?? this.delta, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }

    __call__(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }
  }

  class SmoothL1Loss {
    readonly kind = "smooth-l1-loss";
    readonly reduction: LossReduction;
    readonly beta: number;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      this.beta = finitePositiveLossParameter(lossOptions.beta ?? 1, "SmoothL1Loss beta");
      this.reduction = lossReduction(lossOptions, "SmoothL1Loss");
    }

    forward(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return smoothL1(predValues, targetValues, { ...lossOptions, beta: lossOptions.beta ?? this.beta, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }

    __call__(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }
  }

  class BCELoss {
    readonly kind = "bce-loss";
    readonly reduction: LossReduction;
    readonly eps: number;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      const eps = lossOptions.eps === undefined ? 1e-7 : Number(lossOptions.eps);
      if (!Number.isFinite(eps) || eps <= 0 || eps >= 0.5) {
        throw new Error(`BCELoss eps must be a finite number in (0, 0.5), got ${lossOptions.eps}`);
      }
      this.eps = eps;
      this.reduction = lossReduction(lossOptions, "BCELoss");
    }

    forward(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return binaryCrossEntropy(predValues, targetValues, { ...lossOptions, eps: lossOptions.eps ?? this.eps, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }

    __call__(predValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(predValues, targetValues, lossOptions);
    }
  }

  class BCEWithLogitsLoss {
    readonly kind = "bce-with-logits-loss";
    readonly reduction: LossReduction;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      this.reduction = lossReduction(lossOptions, "BCEWithLogitsLoss");
    }

    forward(logitValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return binaryCrossEntropyWithLogits(logitValues, targetValues, { ...lossOptions, reduction: lossOptions.reduction ?? this.reduction });
    }

    call(logitValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logitValues, targetValues, lossOptions);
    }

    __call__(logitValues: unknown, targetValues: unknown, lossOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logitValues, targetValues, lossOptions);
    }
  }

  class CrossEntropyLoss {
    readonly kind = "cross-entropy-loss";
    readonly classes: number | null;
    readonly numClasses: number | null;
    readonly reduction: LossReduction;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      const classesOption = lossOptions.classes ?? lossOptions.numClasses;
      const classes = classesOption === undefined || classesOption === null ? null : Number(classesOption);
      if (classes !== null && (!Number.isSafeInteger(classes) || classes <= 0)) {
        throw new Error(`CrossEntropyLoss classes must be a positive safe integer, got ${classes}`);
      }
      this.classes = classes;
      this.numClasses = classes;
      this.reduction = lossReduction(lossOptions, "CrossEntropyLoss");
    }

    forward(logitValues: unknown, targetValues: unknown, entropyOptions: LossTrainOptionsRecord = {}) {
      const classes = entropyOptions.classes ?? entropyOptions.numClasses ?? this.classes ?? undefined;
      return crossEntropy(logitValues, targetValues, { ...entropyOptions, classes, reduction: entropyOptions.reduction ?? this.reduction });
    }

    call(logitValues: unknown, targetValues: unknown, entropyOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logitValues, targetValues, entropyOptions);
    }

    __call__(logitValues: unknown, targetValues: unknown, entropyOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logitValues, targetValues, entropyOptions);
    }
  }

  class NLLLoss {
    readonly kind = "nll-loss";
    readonly classes: number | null;
    readonly numClasses: number | null;
    readonly reduction: LossReduction;

    constructor(lossOptions: LossTrainOptionsRecord = {}) {
      const classesOption = lossOptions.classes ?? lossOptions.numClasses;
      const classes = classesOption === undefined || classesOption === null ? null : Number(classesOption);
      if (classes !== null && (!Number.isSafeInteger(classes) || classes <= 0)) {
        throw new Error(`NLLLoss classes must be a positive safe integer, got ${classes}`);
      }
      this.classes = classes;
      this.numClasses = classes;
      this.reduction = lossReduction(lossOptions, "NLLLoss");
    }

    forward(logProbabilityValues: unknown, targetValues: unknown, nllOptions: LossTrainOptionsRecord = {}) {
      const classes = nllOptions.classes ?? nllOptions.numClasses ?? this.classes ?? undefined;
      return negativeLogLikelihood(logProbabilityValues, targetValues, { ...nllOptions, classes, reduction: nllOptions.reduction ?? this.reduction });
    }

    call(logProbabilityValues: unknown, targetValues: unknown, nllOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logProbabilityValues, targetValues, nllOptions);
    }

    __call__(logProbabilityValues: unknown, targetValues: unknown, nllOptions: LossTrainOptionsRecord = {}) {
      return this.forward(logProbabilityValues, targetValues, nllOptions);
    }
  }

  const loss = Object.freeze({
    meanSquaredError,
    mse: meanSquaredError,
    mseLoss: (lossOptions: LossTrainOptionsRecord = {}) => new MSELoss(lossOptions),
    mse_loss: (lossOptions: LossTrainOptionsRecord = {}) => new MSELoss(lossOptions),
    MSELoss,
    meanAbsoluteError,
    l1: meanAbsoluteError,
    l1Loss: (lossOptions: LossTrainOptionsRecord = {}) => new L1Loss(lossOptions),
    l1_loss: (lossOptions: LossTrainOptionsRecord = {}) => new L1Loss(lossOptions),
    L1Loss,
    huber,
    huberLoss: (lossOptions: LossTrainOptionsRecord = {}) => new HuberLoss(lossOptions),
    huber_loss: (lossOptions: LossTrainOptionsRecord = {}) => new HuberLoss(lossOptions),
    HuberLoss,
    smoothL1,
    smooth_l1: smoothL1,
    smoothL1Loss: (lossOptions: LossTrainOptionsRecord = {}) => new SmoothL1Loss(lossOptions),
    smooth_l1_loss: (lossOptions: LossTrainOptionsRecord = {}) => new SmoothL1Loss(lossOptions),
    SmoothL1Loss,
    binaryCrossEntropy,
    binary_cross_entropy: binaryCrossEntropy,
    bce: binaryCrossEntropy,
    bceLoss: (lossOptions: LossTrainOptionsRecord = {}) => new BCELoss(lossOptions),
    bce_loss: (lossOptions: LossTrainOptionsRecord = {}) => new BCELoss(lossOptions),
    BCELoss,
    binaryCrossEntropyWithLogits,
    binary_cross_entropy_with_logits: binaryCrossEntropyWithLogits,
    bceWithLogits: binaryCrossEntropyWithLogits,
    bce_with_logits: binaryCrossEntropyWithLogits,
    bceWithLogitsLoss: (lossOptions: LossTrainOptionsRecord = {}) => new BCEWithLogitsLoss(lossOptions),
    bce_with_logits_loss: (lossOptions: LossTrainOptionsRecord = {}) => new BCEWithLogitsLoss(lossOptions),
    BCEWithLogitsLoss,
    crossEntropy,
    cross_entropy: crossEntropy,
    crossEntropyLoss: (lossOptions: LossTrainOptionsRecord = {}) => new CrossEntropyLoss(lossOptions),
    cross_entropy_loss: (lossOptions: LossTrainOptionsRecord = {}) => new CrossEntropyLoss(lossOptions),
    CrossEntropyLoss,
    negativeLogLikelihood,
    negative_log_likelihood: negativeLogLikelihood,
    nllLoss: negativeLogLikelihood,
    nll_loss: negativeLogLikelihood,
    nllLossModule: (lossOptions: LossTrainOptionsRecord = {}) => new NLLLoss(lossOptions),
    nll_loss_module: (lossOptions: LossTrainOptionsRecord = {}) => new NLLLoss(lossOptions),
    NLLLoss,
    classTargets,
  });

  function gradNorm(paramsOrModule: unknown) {
    let totalSquared = 0;
    const params = resolveParameters(paramsOrModule);
    for (const param of params) {
      if (!parameterRequiresGrad(param)) continue;
      const grad = param.grad ?? param.tensor?.grad;
      if (!grad) continue;
      for (let i = 0; i < grad.length; i += 1) totalSquared += grad[i] * grad[i];
    }
    return Math.sqrt(totalSquared);
  }

  function clipGradNorm(paramsOrModule: unknown, maxNorm: unknown, clipOptions: LossTrainOptionsRecord = {}) {
    const limit = Number(maxNorm);
    if (!Number.isFinite(limit) || limit < 0) throw new Error(`train.clipGradNorm maxNorm must be a non-negative finite number, got ${maxNorm}`);
    const eps = clipOptions.eps === undefined ? 1e-6 : Number(clipOptions.eps);
    if (!Number.isFinite(eps) || eps < 0) throw new Error(`train.clipGradNorm eps must be a non-negative finite number, got ${clipOptions.eps}`);
    const params = resolveParameters(paramsOrModule);
    const totalNorm = gradNorm(params);
    if (totalNorm > limit) {
      const scale = limit / (totalNorm + eps);
      for (const param of params) {
        if (!parameterRequiresGrad(param)) continue;
        const grad = param.grad ?? param.tensor?.grad;
        if (!grad) continue;
        for (let i = 0; i < grad.length; i += 1) grad[i] *= scale;
      }
    }
    return totalNorm;
  }

  function clipGradValue(paramsOrModule: unknown, clipValue: unknown) {
    const limit = Number(clipValue);
    if (!Number.isFinite(limit) || limit < 0) throw new Error(`train.clipGradValue clipValue must be a non-negative finite number, got ${clipValue}`);
    const params = resolveParameters(paramsOrModule);
    for (const param of params) {
      if (!parameterRequiresGrad(param)) continue;
      const grad = param.grad ?? param.tensor?.grad;
      if (!grad) continue;
      for (let i = 0; i < grad.length; i += 1) {
        if (grad[i] > limit) grad[i] = limit;
        else if (grad[i] < -limit) grad[i] = -limit;
      }
    }
    return paramsOrModule;
  }

  function gradientClipTarget(optimizer: LossTrainOptimizer) {
    return Array.isArray(optimizer.params) ? optimizer.params : optimizer;
  }

  function applyGradientClipping(optimizer: LossTrainOptimizer, stepOptions: LossTrainStepOptions): GradientClipEvidence {
    const maxNorm = stepOptions.clipGradNorm ?? stepOptions.clip_grad_norm;
    const clipValue = stepOptions.clipGradValue ?? stepOptions.clip_grad_value;
    const clipGradNormApplied = maxNorm !== undefined;
    const clipGradValueApplied = clipValue !== undefined;
    if (!clipGradNormApplied && !clipGradValueApplied) {
      return {
        clipGradNormApplied: false,
        clipGradValueApplied: false,
        gradNormBeforeClip: null,
        gradNormAfterClip: null,
      };
    }
    const target = gradientClipTarget(optimizer);
    const gradNormBeforeClip = gradNorm(target);
    if (clipGradNormApplied) {
      const clipOptions = stepOptions.clipGradNormOptions ?? stepOptions.clip_grad_norm_options;
      clipGradNorm(target, maxNorm, isRecord(clipOptions) ? clipOptions : undefined);
    }
    if (clipGradValueApplied) clipGradValue(target, clipValue);
    return {
      clipGradNormApplied,
      clipGradValueApplied,
      gradNormBeforeClip,
      gradNormAfterClip: gradNorm(target),
    };
  }

  let lastStepEvidence: TrainStepEvidence | null = null;

  function optimizerSnapshot(optimizer: LossTrainOptimizer): LossTrainOptimizerEvidence {
    if (optimizer && typeof optimizer.stateDict === "function") {
      try {
        const state = optimizer.stateDict();
        const step = state?.step;
        const paramCount = state?.paramCount;
        return {
          kind: isOptimizerStateKind(state?.kind) ? state.kind : null,
          step: Number.isSafeInteger(step) ? Number(step) : null,
          paramCount: Number.isSafeInteger(paramCount) ? Number(paramCount) : null,
        };
      } catch {
        return { kind: null, step: null, paramCount: null };
      }
    }
    return { kind: null, step: null, paramCount: null };
  }

  function optimizerParamCount(optimizer: LossTrainOptimizer, snapshot: LossTrainOptimizerEvidence) {
    if (Number.isSafeInteger(snapshot.paramCount)) return snapshot.paramCount;
    if (Array.isArray(optimizer.params)) return optimizer.params.length;
    return null;
  }

  function optimizerGradientsCleared(optimizer: LossTrainOptimizer) {
    const params = Array.isArray(optimizer.params) ? optimizer.params : [];
    if (params.length === 0) return null;
    for (const param of params) {
      const grad = param?.grad ?? param?.tensor?.grad;
      if (!grad) continue;
      for (let i = 0; i < grad.length; i += 1) {
        if (grad[i] !== 0) return false;
      }
    }
    return true;
  }

  function tensorScalar(lossValue: LossTrainTensor | null) {
    if (!lossValue) return null;
    if (typeof lossValue.item === "function") {
      const value = lossValue.item();
      return Number.isFinite(value) ? value : null;
    }
    const data = lossValue.data;
    if (data && data.length === 1 && Number.isFinite(data[0])) return Number(data[0]);
    return null;
  }

  function stepWithEvidence(optimizer: LossTrainOptimizer, stepOptions: LossTrainStepOptions = {}) {
    if (!optimizer || typeof optimizer.step !== "function") throw new Error("train.step requires an optimizer");
    const before = optimizerSnapshot(optimizer);
    const lossValue = stepOptions.loss ?? null;
    if (lossValue) {
      if (!isTensor(lossValue)) throw new Error("train.step loss option must be a Tensor");
      lossValue.backward(stepOptions.gradient);
    }
    const clipEvidence = applyGradientClipping(optimizer, stepOptions);
    optimizer.step();
    const zeroGradOption = stepOptions.zeroGrad ?? stepOptions.zero_grad;
    const zeroGradApplied = zeroGradOption !== false && typeof optimizer.zeroGrad === "function";
    if (zeroGradApplied && typeof optimizer.zeroGrad === "function") {
      const zeroGradOptions = stepOptions.zeroGradOptions ?? stepOptions.zero_grad_options;
      optimizer.zeroGrad(isRecord(zeroGradOptions) ? zeroGradOptions : undefined);
    }
    const after = optimizerSnapshot(optimizer);
    const evidence = trainStepEvidence({
      kind: "zgml.train.step",
      optimizerKind: after.kind ?? before.kind,
      parameterCount: optimizerParamCount(optimizer, after),
      beforeStep: before.step,
      afterStep: after.step,
      stepAdvanced: before.step !== null && after.step !== null ? after.step - before.step : null,
      hadLoss: Boolean(lossValue),
      lossScalar: tensorScalar(lossValue),
      gradientProvided: stepOptions.gradient !== undefined,
      ...clipEvidence,
      zeroGradApplied,
      gradientsCleared: zeroGradApplied ? optimizerGradientsCleared(optimizer) : false,
    });
    lastStepEvidence = evidence;
    return lastStepEvidence;
  }

  function positiveIntegerOption(value: unknown, name: string, fallback: number, owner = "train.fit") {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isSafeInteger(out) || out <= 0) throw new Error(`${owner} ${name} must be a positive safe integer, got ${value}`);
    return out;
  }

  function nonNegativeIntegerOption(value: unknown, name: string, fallback: number, owner = "train.fit") {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isSafeInteger(out) || out < 0) throw new Error(`${owner} ${name} must be a non-negative safe integer, got ${value}`);
    return out;
  }

  function nonNegativeFiniteOption(value: unknown, name: string, fallback: number, owner = "train.fit") {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isFinite(out) || out < 0) throw new Error(`${owner} ${name} must be a non-negative finite number, got ${value}`);
    return out;
  }

  function fitEarlyStoppingOptions(fitOptions: TrainFitOptions) {
    const source = fitOptions.earlyStopping ?? fitOptions.early_stopping;
    if (source === false) return null;
    const record = isRecord(source) ? source : {};
    const explicitlyEnabled = source === true ||
      isRecord(source) ||
      fitOptions.earlyStoppingPatience !== undefined ||
      fitOptions.early_stopping_patience !== undefined ||
      fitOptions.earlyStoppingMinDelta !== undefined ||
      fitOptions.early_stopping_min_delta !== undefined ||
      fitOptions.earlyStoppingMode !== undefined ||
      fitOptions.early_stopping_mode !== undefined;
    if (!explicitlyEnabled) return null;
    const modeValue = fitOptions.earlyStoppingMode ?? fitOptions.early_stopping_mode ?? record.mode ?? "min";
    const mode = String(modeValue);
    if (mode !== "min" && mode !== "max") throw new Error(`train.fit earlyStopping mode must be min or max, got ${modeValue}`);
    return Object.freeze({
      patience: nonNegativeIntegerOption(fitOptions.earlyStoppingPatience ?? fitOptions.early_stopping_patience ?? record.patience, "earlyStoppingPatience", 0),
      minDelta: nonNegativeFiniteOption(fitOptions.earlyStoppingMinDelta ?? fitOptions.early_stopping_min_delta ?? record.minDelta ?? record.min_delta, "earlyStoppingMinDelta", 0),
      mode,
    });
  }

  function earlyStoppingImproved(loss: number, bestLoss: number | null, mode: "min" | "max", minDelta: number) {
    if (bestLoss === null) return true;
    return mode === "min" ? loss < bestLoss - minDelta : loss > bestLoss + minDelta;
  }

  function fitBatchCountEvidence(batches: unknown, owner = "train.fit") {
    const source = batches as AnyRecord;
    const value = source?.batchCount ?? source?.batch_count;
    if (value === undefined || value === null) return null;
    const count = Number(value);
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`${owner} batchCount must be a non-negative safe integer, got ${value}`);
    }
    return count;
  }

  function fitSampleCountEvidence(batches: unknown, owner = "train.fit") {
    const source = batches as AnyRecord;
    const value = source?.sampleCount ?? source?.sample_count;
    if (value === undefined || value === null) return null;
    const count = Number(value);
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`${owner} sampleCount must be a non-negative safe integer, got ${value}`);
    }
    return count;
  }

  function fitBatchSampleIndicesEvidence(batch: unknown, owner = "train.fit") {
    const source = batch as AnyRecord;
    const value = source?.indices ?? source?.sampleIndices ?? source?.sample_indices;
    if (value === undefined || value === null) return null;
    if (!Array.isArray(value)) throw new Error(`${owner} batch sample indices must be an array when provided`);
    const indices = Object.freeze(value.map((index, offset) => {
      if (!Number.isSafeInteger(index) || index < 0) {
        throw new Error(`${owner} batch sample indices[${offset}] must be a non-negative safe integer, got ${index}`);
      }
      return Number(index);
    }));
    return indices;
  }

  function isCompiledTrainingStep(value: unknown): value is CompiledTrainingStep {
    if (!isRecord(value) || typeof value.step !== "function") return false;
    return value.kind === "zgml.compiled-training-step" || value.native === true;
  }

  function compiledTrainingBatch(batch: unknown) {
    const record = batch as AnyRecord;
    if (!record || record.input === undefined || record.target === undefined) {
      throw new Error("train.fit compiled training batch requires input and target");
    }
    return record;
  }

  function compiledTrainingLoss(result: unknown) {
    const value = Number((result as AnyRecord | null)?.loss);
    if (!Number.isFinite(value)) {
      throw new Error(`train.fit compiled training step must return a finite loss, got ${value}`);
    }
    return value;
  }

  function compiledTrainingPlan(compiled: CompiledTrainingStep): CompiledTrainingPlan | null {
    const plan = typeof compiled.plan === "function"
      ? compiled.plan()
      : typeof compiled.compileEvidence === "function"
        ? compiled.compileEvidence()
        : typeof compiled.compile_evidence === "function"
          ? compiled.compile_evidence()
          : null;
    return plan && typeof plan === "object" ? plan : null;
  }

  function modelInputFeatureCount(module: unknown) {
    const target = module as AnyRecord | null;
    if (!target) return null;
    if (target.kind === "linear" && Number.isSafeInteger(target.inFeatures) && target.inFeatures > 0) {
      return Number(target.inFeatures);
    }
    const layers = target.layers;
    const first = Array.isArray(layers) ? layers[0] as AnyRecord | undefined : undefined;
    if (first?.kind === "linear" && Number.isSafeInteger(first.inFeatures) && first.inFeatures > 0) {
      return Number(first.inFeatures);
    }
    return null;
  }

  function nativeLossKind(criterion: unknown) {
    const loss = criterion as AnyRecord | null;
    if (!loss || loss.reduction === "sum") return null;
    if (loss.kind === "mse-loss") return "mse";
    if (loss.kind === "cross-entropy-loss") return "crossEntropy";
    return null;
  }

  function stableNativeBatchSize(batches: unknown, fitOptions: TrainFitOptions) {
    const source = batches as AnyRecord | null;
    const config = fitOptions as AnyRecord;
    const batchSizeValue = config.batchSize ?? config.batch_size ?? source?.batchSize ?? source?.batch_size;
    if (batchSizeValue !== undefined && batchSizeValue !== null) {
      const batchSize = Number(batchSizeValue);
      if (!Number.isSafeInteger(batchSize) || batchSize <= 0) return null;
      const maxSteps = config.maxSteps ?? config.max_steps;
      if (maxSteps !== undefined && Number(maxSteps) === 1) return batchSize;
      const dropLast = source?.dropLast ?? source?.drop_last ?? config.dropLast ?? config.drop_last;
      if (dropLast === true) return batchSize;
      const sampleCount = fitSampleCountEvidence(batches);
      if (sampleCount !== null && sampleCount % batchSize === 0) return batchSize;
    }
    if (Array.isArray(batches)) {
      const maxStepsValue = config.maxSteps ?? config.max_steps;
      const inspected = maxStepsValue === undefined
        ? batches.length
        : Math.min(batches.length, positiveIntegerOption(maxStepsValue, "maxSteps", Number.MAX_SAFE_INTEGER));
      let batchSize: number | null = null;
      let features: number | null = null;
      for (let i = 0; i < inspected; i += 1) {
        const input = (batches[i] as AnyRecord | null)?.input;
        if (!isTensor(input) || !Array.isArray(input.shape) || input.shape.length !== 2) return null;
        const currentBatch = Number(input.shape[0]);
        const currentFeatures = Number(input.shape[1]);
        if (
          !Number.isSafeInteger(currentBatch) ||
          currentBatch <= 0 ||
          !Number.isSafeInteger(currentFeatures) ||
          currentFeatures <= 0
        ) return null;
        if (batchSize === null) {
          batchSize = currentBatch;
          features = currentFeatures;
        } else if (currentBatch !== batchSize || currentFeatures !== features) {
          return null;
        }
      }
      if (batchSize !== null) return batchSize;
    }
    return null;
  }

  function nativeTrainingCompileOptions(module: unknown, batches: unknown, criterion: unknown, fitOptions: TrainFitOptions) {
    const config = fitOptions as AnyRecord;
    if (config.native === false || config.autoNative === false || config.auto_native === false || config.compile === false) return null;
    if (
      config.gradient !== undefined ||
      config.clipGradNorm !== undefined ||
      config.clip_grad_norm !== undefined ||
      config.clipGradValue !== undefined ||
      config.clip_grad_value !== undefined
    ) {
      return null;
    }
    const loss = nativeLossKind(criterion);
    if (loss === null) return null;
    const inputShapeOption = config.inputShape ?? config.input_shape;
    if (Array.isArray(inputShapeOption)) {
      if (inputShapeOption.length !== 2) return null;
      const batch = Number(inputShapeOption[0]);
      const features = Number(inputShapeOption[1]);
      if (!Number.isSafeInteger(batch) || batch <= 0 || !Number.isSafeInteger(features) || features <= 0) return null;
      return Object.freeze({ inputShape: Object.freeze([batch, features]), loss });
    }
    const batch = stableNativeBatchSize(batches, fitOptions);
    const features = modelInputFeatureCount(module);
    if (batch === null || features === null) return null;
    const out: AnyRecord = { inputShape: Object.freeze([batch, features]), loss };
    const lossRecord = criterion as AnyRecord | null;
    const classes = config.classes ?? config.numClasses ?? config.num_classes ?? lossRecord?.classes ?? lossRecord?.numClasses;
    if (classes !== undefined && classes !== null) out.classes = classes;
    return Object.freeze(out);
  }

  function maybeCompileNativeTrainingStep(optimizer: LossTrainOptimizer, module: unknown, batches: unknown, criterion: unknown, fitOptions: TrainFitOptions) {
    const config = fitOptions as AnyRecord;
    if (compileTrainingStep === null) {
      if (config.requireNative === true || config.require_native === true) {
        throw new Error("train.fitModule requireNative requested a native training compiler, but this frontend has no native compile hook");
      }
      return null;
    }
    const compileOptions = nativeTrainingCompileOptions(module, batches, criterion, fitOptions);
    if (compileOptions === null) {
      if (config.requireNative === true || config.require_native === true) {
        throw new Error("train.fitModule requireNative could not derive a supported fixed-shape native training plan");
      }
      return null;
    }
    try {
      const compiled = compileTrainingStep(module, optimizer, compileOptions);
      return isCompiledTrainingStep(compiled) ? compiled : null;
    } catch (error) {
      if (config.requireNative === true || config.require_native === true) throw error;
      return null;
    }
  }

  function finiteLossScalar(lossValue: LossTrainTensor, label: string) {
    const value = tensorScalar(lossValue);
    if (value === null) throw new Error(`${label} loss must be scalar-like`);
    return value;
  }

  function fitLoop(optimizer: LossTrainOptimizer, batches: unknown, lossFn: unknown, fitOptions: TrainFitOptions = {}) {
    if (!optimizer || typeof optimizer.step !== "function") throw new Error("train.fit requires an optimizer");
    if (!batches || typeof (batches as Iterable<unknown>)[Symbol.iterator] !== "function") {
      throw new Error("train.fit requires an iterable of batches");
    }
    if (typeof lossFn !== "function") throw new Error("train.fit requires a loss function");
    const epochs = positiveIntegerOption(fitOptions.epochs, "epochs", 1);
    const maxStepsOption = fitOptions.maxSteps ?? fitOptions.max_steps;
    const maxSteps = maxStepsOption === undefined ? null : positiveIntegerOption(maxStepsOption, "maxSteps", Number.MAX_SAFE_INTEGER);
    const zeroGradOption = fitOptions.zeroGrad ?? fitOptions.zero_grad;
    const onStep = typeof fitOptions.onStep === "function" ? fitOptions.onStep : fitOptions.on_step;
    const earlyStopping = fitEarlyStoppingOptions(fitOptions);
    const batchCount = fitBatchCountEvidence(batches);
    const sampleCount = fitSampleCountEvidence(batches);
    const losses = [];
    let bestLoss: number | null = null;
    let bestStep: number | null = null;
    let badSteps = 0;
    let steps = 0;
    let lastLoss: LossTrainTensor | null = null;
    let stopped = false;
    let stopReason: "max-steps" | "early-stopping" | null = null;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let batchIndex = 0;
      for (const batch of batches as Iterable<unknown>) {
        if (maxSteps !== null && steps >= maxSteps) {
          stopped = true;
          stopReason = "max-steps";
          break;
        }
        const sampleIndices = fitBatchSampleIndicesEvidence(batch);
        const context: TrainFitContext = Object.freeze({ epoch, batchIndex, step: steps, sampleIndices, sample_indices: sampleIndices });
        const lossValue = (lossFn as (batch: unknown, context: TrainFitContext) => unknown)(batch, context);
        if (!isTensor(lossValue)) throw new Error("train.fit loss function must return a Tensor loss");
        lastLoss = lossValue;
        losses.push(finiteLossScalar(lossValue, "train.fit"));
        stepWithEvidence(optimizer, {
          loss: lossValue,
          gradient: fitOptions.gradient,
          clipGradNorm: fitOptions.clipGradNorm,
          clip_grad_norm: fitOptions.clip_grad_norm,
          clipGradNormOptions: fitOptions.clipGradNormOptions,
          clip_grad_norm_options: fitOptions.clip_grad_norm_options,
          clipGradValue: fitOptions.clipGradValue,
          clip_grad_value: fitOptions.clip_grad_value,
          zeroGrad: zeroGradOption,
          zeroGradOptions: fitOptions.zeroGradOptions ?? fitOptions.zero_grad_options,
        });
        steps += 1;
        if (typeof onStep === "function") {
          const fitStepEvidence = trainFitStepEvidence({
            kind: "zgml.train.fit-step",
            epoch,
            batchIndex,
            step: steps,
            sampleIndices,
            sample_indices: sampleIndices,
            loss: losses[losses.length - 1],
            stepEvidence: lastStepEvidence,
          });
          onStep(fitStepEvidence);
        }
        const latestLoss = losses[losses.length - 1];
        if (earlyStopping) {
          if (earlyStoppingImproved(latestLoss, bestLoss, earlyStopping.mode, earlyStopping.minDelta)) {
            bestLoss = latestLoss;
            bestStep = steps;
            badSteps = 0;
          } else {
            badSteps += 1;
            if (badSteps > earlyStopping.patience) {
              stopped = true;
              stopReason = "early-stopping";
              break;
            }
          }
        } else if (bestLoss === null || latestLoss < bestLoss) {
          bestLoss = latestLoss;
          bestStep = steps;
        }
        batchIndex += 1;
      }
      if (stopped) break;
    }
    return trainFitEvidence({
      kind: "zgml.train.fit",
      epochs,
      steps,
      stoppedEarly: stopped,
      stopReason,
      stop_reason: stopReason,
      batchCount,
      batch_count: batchCount,
      sampleCount,
      sample_count: sampleCount,
      losses: Object.freeze(losses),
      bestLoss,
      best_loss: bestLoss,
      bestStep,
      best_step: bestStep,
      finalLoss: losses.length === 0 ? null : losses[losses.length - 1],
      lastStep: lastStepEvidence,
      lastLoss,
    });
  }

  function fitCompiledTrainingStep(compiled: CompiledTrainingStep, batches: unknown, fitOptions: TrainFitOptions = {}) {
    if (!batches || typeof (batches as Iterable<unknown>)[Symbol.iterator] !== "function") {
      throw new Error("train.fit compiled training requires an iterable of batches");
    }
    const epochs = positiveIntegerOption(fitOptions.epochs, "epochs", 1);
    const maxStepsOption = fitOptions.maxSteps ?? fitOptions.max_steps;
    const maxSteps = maxStepsOption === undefined ? null : positiveIntegerOption(maxStepsOption, "maxSteps", Number.MAX_SAFE_INTEGER);
    const onStep = typeof fitOptions.onStep === "function" ? fitOptions.onStep : fitOptions.on_step;
    const earlyStopping = fitEarlyStoppingOptions(fitOptions);
    const batchCount = fitBatchCountEvidence(batches);
    const sampleCount = fitSampleCountEvidence(batches);
    const plan = compiledTrainingPlan(compiled);
    const losses = [];
    let bestLoss: number | null = null;
    let bestStep: number | null = null;
    let badSteps = 0;
    let steps = 0;
    let stopped = false;
    let stopReason: "max-steps" | "early-stopping" | null = null;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let batchIndex = 0;
      for (const batch of batches as Iterable<unknown>) {
        if (maxSteps !== null && steps >= maxSteps) {
          stopped = true;
          stopReason = "max-steps";
          break;
        }
        const record = compiledTrainingBatch(batch);
        const sampleIndices = fitBatchSampleIndicesEvidence(batch);
        const result = compiled.step(record.input, record.target);
        const latestLoss = compiledTrainingLoss(result);
        losses.push(latestLoss);
        steps += 1;
        if (typeof onStep === "function") {
          onStep(trainFitStepEvidence({
            kind: "zgml.train.fit-step",
            epoch,
            batchIndex,
            step: steps,
            sampleIndices,
            sample_indices: sampleIndices,
            loss: latestLoss,
            stepEvidence: null,
          }));
        }
        if (earlyStopping) {
          if (earlyStoppingImproved(latestLoss, bestLoss, earlyStopping.mode, earlyStopping.minDelta)) {
            bestLoss = latestLoss;
            bestStep = steps;
            badSteps = 0;
          } else {
            badSteps += 1;
            if (badSteps > earlyStopping.patience) {
              stopped = true;
              stopReason = "early-stopping";
              break;
            }
          }
        } else if (bestLoss === null || latestLoss < bestLoss) {
          bestLoss = latestLoss;
          bestStep = steps;
        }
        batchIndex += 1;
      }
      if (stopped) break;
    }
    return trainFitEvidence({
      kind: "zgml.train.fit",
      epochs,
      steps,
      stoppedEarly: stopped,
      stopReason,
      stop_reason: stopReason,
      batchCount,
      batch_count: batchCount,
      sampleCount,
      sample_count: sampleCount,
      losses: Object.freeze(losses),
      bestLoss,
      best_loss: bestLoss,
      bestStep,
      best_step: bestStep,
      finalLoss: losses.length === 0 ? null : losses[losses.length - 1],
      lastStep: null,
      lastLoss: null,
      native: true,
      backend: compiled.backend ?? "native",
      compiledPlan: plan,
      compiled_plan: plan,
    });
  }

  function fit(targetOrOptimizer: unknown, batches: unknown, lossFnOrOptions: unknown, fitOptions: TrainFitOptions = {}) {
    const maybeOptions = isRecord(lossFnOrOptions) ? lossFnOrOptions : null;
    if (isCompiledTrainingStep(targetOrOptimizer)) {
      const options = maybeOptions ? maybeOptions as TrainFitOptions : fitOptions;
      return fitCompiledTrainingStep(targetOrOptimizer, batches, options);
    }
    if (
      maybeOptions &&
      targetOrOptimizer &&
      typeof (targetOrOptimizer as AnyRecord).forward === "function" &&
      maybeOptions.optimizer &&
      (maybeOptions.loss || maybeOptions.criterion)
    ) {
      return fitModule(
        maybeOptions.optimizer as LossTrainOptimizer,
        targetOrOptimizer,
        batches,
        maybeOptions.loss ?? maybeOptions.criterion,
        maybeOptions as TrainFitOptions,
      );
    }
    return fitLoop(targetOrOptimizer as LossTrainOptimizer, batches, lossFnOrOptions, fitOptions);
  }

  function requireNativeFitOptions(fitOptions: TrainFitOptions = {}) {
    return Object.freeze({
      ...fitOptions,
      native: true,
      requireNative: true,
      compile: true,
    });
  }

  function fitNative(
    targetOrOptimizer: unknown,
    moduleOrBatches: unknown,
    batchesOrOptions: unknown,
    criterionOrOptions: unknown = {},
    maybeFitOptions: TrainFitOptions = {},
  ) {
    if (isCompiledTrainingStep(targetOrOptimizer)) {
      const options = isRecord(batchesOrOptions) ? batchesOrOptions as TrainFitOptions : {};
      return fitCompiledTrainingStep(targetOrOptimizer, moduleOrBatches, requireNativeFitOptions(options));
    }
    if (
      targetOrOptimizer &&
      typeof (targetOrOptimizer as AnyRecord).forward === "function" &&
      isRecord(batchesOrOptions) &&
      (batchesOrOptions as AnyRecord).optimizer &&
      ((batchesOrOptions as AnyRecord).loss || (batchesOrOptions as AnyRecord).criterion)
    ) {
      const options = batchesOrOptions as TrainFitOptions & AnyRecord;
      return fitModule(
        options.optimizer as LossTrainOptimizer,
        targetOrOptimizer,
        moduleOrBatches,
        options.loss ?? options.criterion,
        requireNativeFitOptions(options),
      );
    }
    if (
      targetOrOptimizer &&
      typeof (targetOrOptimizer as LossTrainOptimizer).step === "function" &&
      moduleOrBatches &&
      typeof (moduleOrBatches as AnyRecord).forward === "function"
    ) {
      return fitModule(
        targetOrOptimizer as LossTrainOptimizer,
        moduleOrBatches,
        batchesOrOptions,
        criterionOrOptions,
        requireNativeFitOptions(maybeFitOptions),
      );
    }
    throw new Error("train.fitNative requires a compiled native training step or a module with { optimizer, loss } options");
  }

  function fitModule(optimizer: LossTrainOptimizer, module: unknown, batches: unknown, criterion: unknown, fitOptions: TrainFitOptions = {}) {
    const target = module as AnyRecord;
    const loss = criterion as AnyRecord;
    if (!target || typeof target.forward !== "function") throw new Error("train.fitModule requires a module with forward(input)");
    if (!loss || typeof loss.forward !== "function") throw new Error("train.fitModule requires a criterion with forward(prediction, target)");
    const compiled = maybeCompileNativeTrainingStep(optimizer, module, batches, criterion, fitOptions);
    if (compiled !== null) return fitCompiledTrainingStep(compiled, batches, fitOptions);
    return fitLoop(optimizer, batches, (batch: unknown, context: TrainFitContext) => {
      const record = batch as AnyRecord;
      if (!isTensor(record?.input)) throw new Error("train.fitModule batch.input must be a Tensor");
      if (!isTensor(record?.target)) throw new Error("train.fitModule batch.target must be a Tensor");
      const prediction = target.forward(record.input);
      if (!isTensor(prediction)) throw new Error("train.fitModule module.forward must return a Tensor");
      const lossValue = loss.forward(prediction, record.target);
      if (!isTensor(lossValue)) throw new Error("train.fitModule criterion.forward must return a Tensor loss");
      return lossValue;
    }, fitOptions);
  }

  function evaluate(batches: unknown, lossFn: unknown, evaluateOptions: TrainEvaluateOptions = {}) {
    if (!batches || typeof (batches as Iterable<unknown>)[Symbol.iterator] !== "function") {
      throw new Error("train.evaluate requires an iterable of batches");
    }
    if (typeof lossFn !== "function") throw new Error("train.evaluate requires a loss function");
    const maxStepsOption = evaluateOptions.maxSteps ?? evaluateOptions.max_steps;
    const maxSteps = maxStepsOption === undefined ? null : positiveIntegerOption(maxStepsOption, "maxSteps", Number.MAX_SAFE_INTEGER, "train.evaluate");
    const onStep = typeof evaluateOptions.onStep === "function" ? evaluateOptions.onStep : evaluateOptions.on_step;
    const batchCount = fitBatchCountEvidence(batches, "train.evaluate");
    const sampleCount = fitSampleCountEvidence(batches, "train.evaluate");
    const losses = [];
    let steps = 0;
    let lastLoss: LossTrainTensor | null = null;
    let stopped = false;
    for (const batch of batches as Iterable<unknown>) {
      if (maxSteps !== null && steps >= maxSteps) {
        stopped = true;
        break;
      }
      const sampleIndices = fitBatchSampleIndicesEvidence(batch, "train.evaluate");
      const context: TrainEvaluateContext = Object.freeze({ batchIndex: steps, step: steps, sampleIndices, sample_indices: sampleIndices });
      const lossValue = (lossFn as (batch: unknown, context: TrainEvaluateContext) => unknown)(batch, context);
      if (!isTensor(lossValue)) throw new Error("train.evaluate loss function must return a Tensor loss");
      lastLoss = lossValue;
      const loss = finiteLossScalar(lossValue, "train.evaluate");
      losses.push(loss);
      steps += 1;
      if (typeof onStep === "function") {
        const evaluateStepEvidence = trainEvaluateStepEvidence({
          kind: "zgml.train.evaluate-step",
          batchIndex: context.batchIndex,
          step: steps,
          sampleIndices,
          sample_indices: sampleIndices,
          loss,
        });
        onStep(evaluateStepEvidence);
      }
    }
    const finalLoss = losses.length === 0 ? null : losses[losses.length - 1];
    const meanLoss = losses.length === 0 ? null : losses.reduce((total, loss) => total + loss, 0) / losses.length;
    return trainEvaluateEvidence({
      kind: "zgml.train.evaluate",
      steps,
      stoppedEarly: stopped,
      batchCount,
      batch_count: batchCount,
      sampleCount,
      sample_count: sampleCount,
      losses: Object.freeze(losses),
      meanLoss,
      mean_loss: meanLoss,
      finalLoss,
      final_loss: finalLoss,
      lastLoss,
    });
  }

  function evaluateModule(module: unknown, batches: unknown, criterion: unknown, evaluateOptions: TrainEvaluateOptions = {}) {
    const target = module as AnyRecord;
    const loss = criterion as AnyRecord;
    if (!target || typeof target.forward !== "function") throw new Error("train.evaluateModule requires a module with forward(input)");
    if (!loss || typeof loss.forward !== "function") throw new Error("train.evaluateModule requires a criterion with forward(prediction, target)");
    return evaluate(batches, (batch: unknown, context: TrainEvaluateContext) => {
      const record = batch as AnyRecord;
      if (!isTensor(record?.input)) throw new Error("train.evaluateModule batch.input must be a Tensor");
      if (!isTensor(record?.target)) throw new Error("train.evaluateModule batch.target must be a Tensor");
      const prediction = target.forward(record.input);
      if (!isTensor(prediction)) throw new Error("train.evaluateModule module.forward must return a Tensor");
      const lossValue = loss.forward(prediction, record.target);
      if (!isTensor(lossValue)) throw new Error("train.evaluateModule criterion.forward must return a Tensor loss");
      return lossValue;
    }, evaluateOptions);
  }

  function predict(batches: unknown, predictFn: unknown, predictOptions: TrainPredictOptions = {}) {
    if (!batches || typeof (batches as Iterable<unknown>)[Symbol.iterator] !== "function") {
      throw new Error("train.predict requires an iterable of batches");
    }
    if (typeof predictFn !== "function") throw new Error("train.predict requires a prediction function");
    const maxStepsOption = predictOptions.maxSteps ?? predictOptions.max_steps;
    const maxSteps = maxStepsOption === undefined ? null : positiveIntegerOption(maxStepsOption, "maxSteps", Number.MAX_SAFE_INTEGER, "train.predict");
    const onStep = typeof predictOptions.onStep === "function" ? predictOptions.onStep : predictOptions.on_step;
    const batchCount = fitBatchCountEvidence(batches, "train.predict");
    const sampleCount = fitSampleCountEvidence(batches, "train.predict");
    const outputs: LossTrainTensor[] = [];
    let steps = 0;
    let stopped = false;
    for (const batch of batches as Iterable<unknown>) {
      if (maxSteps !== null && steps >= maxSteps) {
        stopped = true;
        break;
      }
      const sampleIndices = fitBatchSampleIndicesEvidence(batch, "train.predict");
      const context: TrainPredictContext = Object.freeze({ batchIndex: steps, step: steps, sampleIndices, sample_indices: sampleIndices });
      const output = (predictFn as (batch: unknown, context: TrainPredictContext) => unknown)(batch, context);
      if (!isTensor(output)) throw new Error("train.predict prediction function must return a Tensor output");
      outputs.push(output);
      steps += 1;
      if (typeof onStep === "function") {
        onStep(trainPredictStepEvidence({
          kind: "zgml.train.predict-step",
          batchIndex: context.batchIndex,
          step: steps,
          sampleIndices,
          sample_indices: sampleIndices,
          output,
        }));
      }
    }
    const frozenOutputs = Object.freeze(outputs.slice());
    const lastOutput = frozenOutputs.length === 0 ? null : frozenOutputs[frozenOutputs.length - 1];
    return trainPredictEvidence({
      kind: "zgml.train.predict",
      steps,
      stoppedEarly: stopped,
      batchCount,
      batch_count: batchCount,
      sampleCount,
      sample_count: sampleCount,
      outputs: frozenOutputs,
      lastOutput,
      last_output: lastOutput,
    });
  }

  function predictModule(module: unknown, batches: unknown, predictOptions: TrainPredictOptions = {}) {
    const target = module as AnyRecord;
    if (!target || typeof target.forward !== "function") throw new Error("train.predictModule requires a module with forward(input)");
    return predict(batches, (batch: unknown) => {
      const record = batch as AnyRecord;
      if (!isTensor(record?.input)) throw new Error("train.predictModule batch.input must be a Tensor");
      const output = target.forward(record.input);
      if (!isTensor(output)) throw new Error("train.predictModule module.forward must return a Tensor output");
      return output;
    }, predictOptions);
  }

  const train = Object.freeze({
    zeroGrad,
    gradNorm,
    grad_norm: gradNorm,
    clipGradNorm,
    clip_grad_norm_: clipGradNorm,
    clipGradValue,
    clip_grad_value_: clipGradValue,
    accuracy: classificationAccuracy,
    classificationAccuracy,
    classification_accuracy: classificationAccuracy,
    classPredictions,
    class_predictions: classPredictions,
    predictClasses: classPredictions,
    predict_classes: classPredictions,
    topKAccuracy,
    top_k_accuracy: topKAccuracy,
    confusionMatrix,
    confusion_matrix: confusionMatrix,
    classificationReport,
    classification_report: classificationReport,
    binaryAccuracy,
    binary_accuracy: binaryAccuracy,
    binaryLogitsAccuracy,
    binary_logits_accuracy: binaryLogitsAccuracy,
    backward: (lossValue: unknown, gradient: unknown) => {
      if (!isTensor(lossValue)) throw new Error("train.backward requires a Tensor loss");
      lossValue.backward(gradient);
    },
    step: (optimizer: LossTrainOptimizer, stepOptions: TrainStepOptions = {}) => {
      const evidence = stepWithEvidence(optimizer, stepOptions as LossTrainStepOptions);
      if (stepOptions.inspect === true || stepOptions.evidence === true) return evidence;
      return undefined;
    },
    lossStep: (optimizer: LossTrainOptimizer, lossFn: unknown, stepOptions: TrainLossStepOptions = {}) => {
      if (typeof lossFn !== "function") throw new Error("train.lossStep requires a loss function");
      const lossValue = (lossFn as () => unknown)();
      if (!isTensor(lossValue)) throw new Error("train.lossStep loss function must return a Tensor loss");
      stepWithEvidence(optimizer, { ...stepOptions, loss: lossValue });
      return lossValue;
    },
    fit,
    fitNative,
    fit_native: fitNative,
    fitModule,
    fit_module: fitModule,
    fitClassifier: fitModule,
    fit_classifier: fitModule,
    evaluate,
    evaluate_loss: evaluate,
    evaluateModule,
    evaluate_module: evaluateModule,
    evaluateClassifier: evaluateModule,
    evaluate_classifier: evaluateModule,
    predict,
    predict_batches: predict,
    predictModule,
    predict_module: predictModule,
    predictClassifier: predictModule,
    predict_classifier: predictModule,
    inspectStep: () => lastStepEvidence,
    isTrainStepEvidence,
    requireTrainStepEvidence,
    assertTrainStepEvidence,
    assert_train_step_evidence,
    matchesTrainStepEvidenceSignature,
    matches_train_step_evidence_signature,
    isTrainFitStepEvidence,
    requireTrainFitStepEvidence,
    assertTrainFitStepEvidence,
    assert_train_fit_step_evidence,
    matchesTrainFitStepEvidenceSignature,
    matches_train_fit_step_evidence_signature,
    isTrainFitEvidence,
    requireTrainFitEvidence,
    assertTrainFitEvidence,
    assert_train_fit_evidence,
    matchesTrainFitEvidenceSignature,
    matches_train_fit_evidence_signature,
    isTrainEvaluateStepEvidence,
    requireTrainEvaluateStepEvidence,
    assertTrainEvaluateStepEvidence,
    assert_train_evaluate_step_evidence,
    matchesTrainEvaluateStepEvidenceSignature,
    matches_train_evaluate_step_evidence_signature,
    isTrainEvaluateEvidence,
    requireTrainEvaluateEvidence,
    assertTrainEvaluateEvidence,
    assert_train_evaluate_evidence,
    matchesTrainEvaluateEvidenceSignature,
    matches_train_evaluate_evidence_signature,
    isTrainPredictStepEvidence,
    requireTrainPredictStepEvidence,
    assertTrainPredictStepEvidence,
    assert_train_predict_step_evidence,
    matchesTrainPredictStepEvidenceSignature,
    matches_train_predict_step_evidence_signature,
    isTrainPredictEvidence,
    requireTrainPredictEvidence,
    assertTrainPredictEvidence,
    assert_train_predict_evidence,
    matchesTrainPredictEvidenceSignature,
    matches_train_predict_evidence_signature,
  });

  return {
    meanSquaredError,
    meanAbsoluteError,
    huber,
    smoothL1,
    binaryCrossEntropy,
    binaryCrossEntropyWithLogits,
    classTargets,
    crossEntropy,
    cross_entropy: crossEntropy,
    negativeLogLikelihood,
    negative_log_likelihood: negativeLogLikelihood,
    nllLoss: negativeLogLikelihood,
    nll_loss: negativeLogLikelihood,
    classificationAccuracy,
    loss,
    train,
  };
}
