"use strict";

import { optimizerConfigSnapshotSignature } from "./optimizer_snapshot.js";

type AnyRecord = Record<string, any>;

type OptimizerHookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

type OptimizerConfig = Readonly<Record<string, unknown>>;

type OptimizerTensorState = {
  requiresGrad?: boolean;
  grad?: Float32Array | null;
};

type OptimizerParameter = AnyRecord & {
  readonly name?: string;
  readonly data: Float32Array;
  grad?: Float32Array | null;
  tensor?: OptimizerTensorState;
  requiresGrad?: boolean;
  requires_grad?: boolean;
};

type OptimizerParamGroupInput = Readonly<Record<string, unknown> & {
  params: unknown;
  lr?: unknown;
  weightDecay?: unknown;
  weight_decay?: unknown;
}>;

type OptimizerParamGroupEvidence = Readonly<{
  index: number;
  lr: number;
  weightDecay: number;
  weight_decay: number;
  paramCount: number;
  start: number;
  end: number;
}>;

type OptimizerParamGroupDefaults = Readonly<{
  lr: number;
  weightDecay: number;
}>;

type OptimizerGroupedParameters = Readonly<{
  params: OptimizerParameter[];
  groups: readonly OptimizerParamGroupEvidence[];
}>;

type OptimizerStateEntry = AnyRecord & {
  readonly name?: string;
};

type OptimizerConfigSnapshot = AnyRecord & {
  kind: "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad";
  lr: number;
  weightDecay: number;
  weight_decay: number;
  step: number;
  paramCount: number;
  paramGroups: readonly OptimizerParamGroupEvidence[];
  signature?: string;
};

type OptimizerStateEntriesResult = Readonly<{
  byName: Map<string, OptimizerStateEntry>;
  entries: readonly OptimizerStateEntry[];
  strict: boolean;
}>;

type OptimizerTensorStateUpdate = Readonly<{
  target: Float32Array;
  data: Float32Array;
}>;

export type OptimizerClassesOptions = Readonly<{
  resolveParameters: OptimizerHookCallback<[paramsOrModule: unknown], OptimizerParameter[]>;
  finiteConfigNumber: OptimizerHookCallback<[
    config: OptimizerConfig,
    name: string,
    fallback: number,
    check: (value: number) => boolean,
  ], number>;
  zeroGrad: OptimizerHookCallback<[paramsOrModule: unknown, options?: OptimizerConfig], unknown>;
  optimizerStateEntry: OptimizerHookCallback<[
    name: string,
    param: OptimizerParameter,
    data: Float32Array,
  ], OptimizerStateEntry>;
  optimizerStateDict: OptimizerHookCallback<[
    kind: string,
    params: readonly OptimizerParameter[],
    entries: readonly OptimizerStateEntry[],
    step?: number,
  ], unknown>;
  optimizerStateEntries: OptimizerHookCallback<[
    source: unknown,
    expectedKind: string,
    expectedParamCount: number,
    options?: OptimizerConfig,
  ], OptimizerStateEntriesResult>;
  optimizerStepFromState: OptimizerHookCallback<[source: unknown, fallback: number, strict: boolean], number>;
  loadOptimizerTensorState: OptimizerHookCallback<[
    byName: Map<string, OptimizerStateEntry>,
    expectedNames: Set<string>,
    name: string,
    param: OptimizerParameter,
    target: Float32Array,
    strict: boolean,
  ], OptimizerTensorStateUpdate | null>;
  rejectUnexpectedOptimizerState: OptimizerHookCallback<[
    entries: readonly OptimizerStateEntry[],
    expectedNames: Set<string>,
  ], void>;
}>;

export function createOptimizerClasses(options: OptimizerClassesOptions) {
  const resolveParameters = options.resolveParameters;
  const finiteConfigNumber = options.finiteConfigNumber;
  const zeroGrad = options.zeroGrad;
  const optimizerStateEntry = options.optimizerStateEntry;
  const optimizerStateDict = options.optimizerStateDict;
  const optimizerStateEntries = options.optimizerStateEntries;
  const optimizerStepFromState = options.optimizerStepFromState;
  const loadOptimizerTensorState = options.loadOptimizerTensorState;
  const rejectUnexpectedOptimizerState = options.rejectUnexpectedOptimizerState;
  if (
    typeof resolveParameters !== "function" ||
    typeof finiteConfigNumber !== "function" ||
    typeof zeroGrad !== "function" ||
    typeof optimizerStateEntry !== "function" ||
    typeof optimizerStateDict !== "function" ||
    typeof optimizerStateEntries !== "function" ||
    typeof optimizerStepFromState !== "function" ||
    typeof loadOptimizerTensorState !== "function" ||
    typeof rejectUnexpectedOptimizerState !== "function"
  ) {
    throw new Error("optimizer factory requires parameter, config, state, and zeroGrad hooks");
  }

  function weightDecay(config: OptimizerConfig) {
    return finiteConfigNumber({ weightDecay: config.weightDecay ?? config.weight_decay }, "weightDecay", 0, (x: number) => x >= 0);
  }

  function isParamGroup(value: unknown): value is OptimizerParamGroupInput {
    return Boolean(value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, "params"));
  }

  function parameterRequiresGrad(param: OptimizerParameter) {
    return param?.tensor ? param.tensor.requiresGrad !== false : param?.requiresGrad !== false;
  }

  function validateUniqueOptimizerParams(params: readonly OptimizerParameter[], existing: ReadonlySet<OptimizerParameter> = new Set()) {
    const seen = new Set(existing);
    for (const param of params) {
      if (seen.has(param)) {
        throw new Error("optimizer parameter groups must not contain duplicate parameters");
      }
      seen.add(param);
    }
  }

  function optimizerParamGroups(paramsOrModule: unknown, defaults: OptimizerParamGroupDefaults): OptimizerGroupedParameters {
    const sourceGroups = Array.isArray(paramsOrModule) && paramsOrModule.every(isParamGroup)
      ? paramsOrModule
      : [{ params: paramsOrModule }];
    const groups: OptimizerParamGroupEvidence[] = [];
    const params: OptimizerParameter[] = [];
    for (let i = 0; i < sourceGroups.length; i += 1) {
      const group = sourceGroups[i];
      const groupParams = resolveParameters(group.params);
      validateUniqueOptimizerParams(groupParams, new Set(params));
      const lr = finiteConfigNumber({ lr: group.lr ?? defaults.lr }, "lr", defaults.lr, (x: number) => x > 0);
      const decay = weightDecay({ weightDecay: group.weightDecay ?? group.weight_decay ?? defaults.weightDecay });
      const start = params.length;
      params.push(...groupParams);
      groups.push(Object.freeze({
        index: i,
        lr,
        weightDecay: decay,
        weight_decay: decay,
        paramCount: groupParams.length,
        start,
        end: params.length,
      }));
    }
    return { params, groups };
  }

  function groupForIndex(groups: readonly OptimizerParamGroupEvidence[], index: number) {
    return groups.find((group) => index >= group.start && index < group.end) ?? groups[0];
  }

  function optimizerParamGroup(group: OptimizerParamGroupInput, index: number, start: number, defaults: OptimizerParamGroupDefaults) {
    const groupParams = resolveParameters(group.params);
    const lr = finiteConfigNumber({ lr: group.lr ?? defaults.lr }, "lr", defaults.lr, (x: number) => x > 0);
    const decay = weightDecay({ weightDecay: group.weightDecay ?? group.weight_decay ?? defaults.weightDecay });
    return {
      params: groupParams,
      evidence: Object.freeze({
        index,
        lr,
        weightDecay: decay,
        weight_decay: decay,
        paramCount: groupParams.length,
        start,
        end: start + groupParams.length,
      }),
    };
  }

  function setLearningRate<TOptimizer extends { lr: number; paramGroups: readonly OptimizerParamGroupEvidence[] }>(optimizer: TOptimizer, lr: unknown) {
    optimizer.lr = finiteConfigNumber({ lr }, "lr", optimizer.lr, (x: number) => x >= 0);
    optimizer.paramGroups = Object.freeze((optimizer.paramGroups ?? []).map((group) => Object.freeze({
      ...group,
      lr: optimizer.lr,
    }))) as TOptimizer["paramGroups"];
    return optimizer;
  }

  class SgdOptimizer {
    readonly kind = "sgd";
    params: OptimizerParameter[];
    lr: number;
    momentum: number;
    weightDecay: number;
    paramGroups: readonly OptimizerParamGroupEvidence[];
    t: number;
    velocity: Float32Array[];

    constructor(paramsOrModule: unknown, config: OptimizerConfig = {}) {
      this.lr = finiteConfigNumber(config, "lr", 0.01, (x: number) => x > 0);
      this.momentum = finiteConfigNumber(config, "momentum", 0, (x: number) => x >= 0 && x < 1);
      this.weightDecay = weightDecay(config);
      const grouped = optimizerParamGroups(paramsOrModule, { lr: this.lr, weightDecay: this.weightDecay });
      this.params = grouped.params;
      this.paramGroups = Object.freeze(grouped.groups);
      this.t = 0;
      this.velocity = this.momentum === 0 ? [] : this.params.map((param) => new Float32Array(param.data.length));
    }

    step() {
      this.t += 1;
      for (let p = 0; p < this.params.length; p += 1) {
        const param = this.params[p];
        const grad = param.grad;
        if (!grad || !parameterRequiresGrad(param)) continue;
        const velocity = this.velocity[p];
        const group = groupForIndex(this.paramGroups, p);
        for (let i = 0; i < param.data.length; i += 1) {
          const g = grad[i] + group.weightDecay * param.data[i];
          if (this.momentum === 0) {
            param.data[i] -= group.lr * g;
          } else {
            velocity[i] = this.momentum * velocity[i] + g;
            param.data[i] -= group.lr * velocity[i];
          }
        }
      }
    }

    zeroGrad(options: OptimizerConfig = {}) {
      zeroGrad(this.params, options);
    }

    zero_grad(options: OptimizerConfig = {}) {
      return this.zeroGrad(options);
    }

    setLearningRate(lr: unknown) {
      return setLearningRate(this, lr);
    }

    set_lr(lr: unknown) {
      return this.setLearningRate(lr);
    }

    getLearningRate() {
      return this.lr;
    }

    get_lr() {
      return this.getLearningRate();
    }

    addParamGroup(group: unknown) {
      if (!isParamGroup(group)) throw new Error("optimizer.addParamGroup requires { params, ...options }");
      const next = optimizerParamGroup(group, this.paramGroups.length, this.params.length, { lr: this.lr, weightDecay: this.weightDecay });
      validateUniqueOptimizerParams(next.params, new Set(this.params));
      this.params.push(...next.params);
      this.paramGroups = Object.freeze([...this.paramGroups, next.evidence]);
      if (this.momentum !== 0) {
        this.velocity.push(...next.params.map((param) => new Float32Array(param.data.length)));
      }
      return this;
    }

    add_param_group(group: unknown) {
      return this.addParamGroup(group);
    }

    get param_groups() {
      return this.paramGroups;
    }

    get defaults() {
      return this.config();
    }

    config() {
      const snapshot: OptimizerConfigSnapshot & {
        momentum: number;
      } = {
        kind: "sgd",
        lr: this.lr,
        momentum: this.momentum,
        weightDecay: this.weightDecay,
        weight_decay: this.weightDecay,
        step: this.t,
        paramCount: this.params.length,
        paramGroups: this.paramGroups,
        param_groups: this.paramGroups,
      };
      snapshot.signature = optimizerConfigSnapshotSignature(snapshot);
      return Object.freeze(snapshot);
    }

    stateDict() {
      return optimizerStateDict(
        "sgd",
        this.params,
        this.velocity.map((velocity, index) => optimizerStateEntry(`velocity.${index}`, this.params[index], velocity)),
        this.t,
      );
    }

    state_dict() {
      return this.stateDict();
    }

    loadStateDict(source: unknown, loadOptions: OptimizerConfig = {}) {
      const { byName, entries, strict } = optimizerStateEntries(source, "sgd", this.params.length, loadOptions);
      const step = optimizerStepFromState(source, this.t, strict);
      const validateOnly = loadOptions.validateOnly === true;
      const expected = new Set<string>();
      const pending: OptimizerTensorStateUpdate[] = [];
      for (let i = 0; i < this.velocity.length; i += 1) {
        const update = loadOptimizerTensorState(byName, expected, `velocity.${i}`, this.params[i], this.velocity[i], strict);
        if (update) pending.push(update);
      }
      if (strict) rejectUnexpectedOptimizerState(entries, expected);
      for (const { target, data } of pending) {
        if (!validateOnly) target.set(data);
      }
      if (!validateOnly) this.t = step;
      return this;
    }

    load_state_dict(source: unknown, loadOptions: OptimizerConfig = {}) {
      return this.loadStateDict(source, loadOptions);
    }
  }

  class AdamOptimizer {
    readonly kind: "adam" | "adamw";
    params: OptimizerParameter[];
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    decoupledWeightDecay: boolean;
    paramGroups: readonly OptimizerParamGroupEvidence[];
    t: number;
    m: Float32Array[];
    v: Float32Array[];

    constructor(paramsOrModule: unknown, config: OptimizerConfig = {}, decoupledWeightDecay = false) {
      this.kind = decoupledWeightDecay ? "adamw" : "adam";
      this.lr = finiteConfigNumber(config, "lr", 1e-3, (x: number) => x > 0);
      this.beta1 = finiteConfigNumber(config, "beta1", 0.9, (x: number) => x >= 0 && x < 1);
      this.beta2 = finiteConfigNumber(config, "beta2", 0.999, (x: number) => x >= 0 && x < 1);
      this.eps = finiteConfigNumber(config, "eps", 1e-8, (x: number) => x > 0);
      this.weightDecay = weightDecay(config);
      this.decoupledWeightDecay = decoupledWeightDecay;
      const grouped = optimizerParamGroups(paramsOrModule, { lr: this.lr, weightDecay: this.weightDecay });
      this.params = grouped.params;
      this.paramGroups = Object.freeze(grouped.groups);
      this.t = 0;
      this.m = this.params.map((param) => new Float32Array(param.data.length));
      this.v = this.params.map((param) => new Float32Array(param.data.length));
    }

    step() {
      this.t += 1;
      const biasCorrection1 = 1 / (1 - Math.pow(this.beta1, this.t));
      const biasCorrection2 = 1 / (1 - Math.pow(this.beta2, this.t));
      for (let p = 0; p < this.params.length; p += 1) {
        const param = this.params[p];
        const grad = param.grad;
        if (!grad || !parameterRequiresGrad(param)) continue;
        const m = this.m[p];
        const v = this.v[p];
        const group = groupForIndex(this.paramGroups, p);
        for (let i = 0; i < param.data.length; i += 1) {
          const g = this.decoupledWeightDecay ? grad[i] : grad[i] + group.weightDecay * param.data[i];
          m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
          v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
          if (this.decoupledWeightDecay && group.weightDecay !== 0) {
            param.data[i] -= group.lr * group.weightDecay * param.data[i];
          }
          const mHat = m[i] * biasCorrection1;
          const vHat = v[i] * biasCorrection2;
          param.data[i] -= group.lr * mHat / (Math.sqrt(vHat) + this.eps);
        }
      }
    }

    zeroGrad(options: OptimizerConfig = {}) {
      zeroGrad(this.params, options);
    }

    zero_grad(options: OptimizerConfig = {}) {
      return this.zeroGrad(options);
    }

    setLearningRate(lr: unknown) {
      return setLearningRate(this, lr);
    }

    set_lr(lr: unknown) {
      return this.setLearningRate(lr);
    }

    getLearningRate() {
      return this.lr;
    }

    get_lr() {
      return this.getLearningRate();
    }

    addParamGroup(group: unknown) {
      if (!isParamGroup(group)) throw new Error("optimizer.addParamGroup requires { params, ...options }");
      const next = optimizerParamGroup(group, this.paramGroups.length, this.params.length, { lr: this.lr, weightDecay: this.weightDecay });
      validateUniqueOptimizerParams(next.params, new Set(this.params));
      this.params.push(...next.params);
      this.paramGroups = Object.freeze([...this.paramGroups, next.evidence]);
      this.m.push(...next.params.map((param) => new Float32Array(param.data.length)));
      this.v.push(...next.params.map((param) => new Float32Array(param.data.length)));
      return this;
    }

    add_param_group(group: unknown) {
      return this.addParamGroup(group);
    }

    get param_groups() {
      return this.paramGroups;
    }

    get defaults() {
      return this.config();
    }

    config() {
      const snapshot: OptimizerConfigSnapshot & {
        beta1: number;
        beta2: number;
        eps: number;
        decoupledWeightDecay: boolean;
      } = {
        kind: this.kind,
        lr: this.lr,
        beta1: this.beta1,
        beta2: this.beta2,
        eps: this.eps,
        weightDecay: this.weightDecay,
        weight_decay: this.weightDecay,
        decoupledWeightDecay: this.decoupledWeightDecay,
        step: this.t,
        paramCount: this.params.length,
        paramGroups: this.paramGroups,
        param_groups: this.paramGroups,
      };
      snapshot.signature = optimizerConfigSnapshotSignature(snapshot);
      return Object.freeze(snapshot);
    }

    stateDict() {
      const entries: OptimizerStateEntry[] = [];
      for (let i = 0; i < this.params.length; i += 1) {
        entries.push(optimizerStateEntry(`m.${i}`, this.params[i], this.m[i]));
        entries.push(optimizerStateEntry(`v.${i}`, this.params[i], this.v[i]));
      }
      return optimizerStateDict(this.kind, this.params, entries, this.t);
    }

    state_dict() {
      return this.stateDict();
    }

    loadStateDict(source: unknown, loadOptions: OptimizerConfig = {}) {
      const { byName, entries, strict } = optimizerStateEntries(source, this.kind, this.params.length, loadOptions);
      const validateOnly = loadOptions.validateOnly === true;
      const step = optimizerStepFromState(source, this.t, strict);
      const expected = new Set<string>();
      const pending: OptimizerTensorStateUpdate[] = [];
      for (let i = 0; i < this.params.length; i += 1) {
        const mUpdate = loadOptimizerTensorState(byName, expected, `m.${i}`, this.params[i], this.m[i], strict);
        const vUpdate = loadOptimizerTensorState(byName, expected, `v.${i}`, this.params[i], this.v[i], strict);
        if (mUpdate) pending.push(mUpdate);
        if (vUpdate) pending.push(vUpdate);
      }
      if (strict) rejectUnexpectedOptimizerState(entries, expected);
      for (const { target, data } of pending) {
        if (!validateOnly) target.set(data);
      }
      if (!validateOnly) this.t = step;
      return this;
    }

    load_state_dict(source: unknown, loadOptions: OptimizerConfig = {}) {
      return this.loadStateDict(source, loadOptions);
    }
  }

  class AdamWOptimizer extends AdamOptimizer {
    constructor(paramsOrModule: unknown, config: OptimizerConfig = {}) {
      super(paramsOrModule, config, true);
    }
  }

  class RMSpropOptimizer {
    readonly kind = "rmsprop";
    params: OptimizerParameter[];
    lr: number;
    alpha: number;
    eps: number;
    momentum: number;
    weightDecay: number;
    paramGroups: readonly OptimizerParamGroupEvidence[];
    t: number;
    squareAvg: Float32Array[];
    momentumBuffer: Float32Array[];

    constructor(paramsOrModule: unknown, config: OptimizerConfig = {}) {
      this.lr = finiteConfigNumber(config, "lr", 0.01, (x: number) => x > 0);
      this.alpha = finiteConfigNumber(config, "alpha", 0.99, (x: number) => x >= 0 && x < 1);
      this.eps = finiteConfigNumber(config, "eps", 1e-8, (x: number) => x > 0);
      this.momentum = finiteConfigNumber(config, "momentum", 0, (x: number) => x >= 0 && x < 1);
      this.weightDecay = weightDecay(config);
      const grouped = optimizerParamGroups(paramsOrModule, { lr: this.lr, weightDecay: this.weightDecay });
      this.params = grouped.params;
      this.paramGroups = Object.freeze(grouped.groups);
      this.t = 0;
      this.squareAvg = this.params.map((param) => new Float32Array(param.data.length));
      this.momentumBuffer = this.momentum === 0 ? [] : this.params.map((param) => new Float32Array(param.data.length));
    }

    step() {
      this.t += 1;
      for (let p = 0; p < this.params.length; p += 1) {
        const param = this.params[p];
        const grad = param.grad;
        if (!grad || !parameterRequiresGrad(param)) continue;
        const squareAvg = this.squareAvg[p];
        const momentumBuffer = this.momentumBuffer[p];
        const group = groupForIndex(this.paramGroups, p);
        for (let i = 0; i < param.data.length; i += 1) {
          const g = grad[i] + group.weightDecay * param.data[i];
          squareAvg[i] = this.alpha * squareAvg[i] + (1 - this.alpha) * g * g;
          const scaled = g / (Math.sqrt(squareAvg[i]) + this.eps);
          if (this.momentum === 0) {
            param.data[i] -= group.lr * scaled;
          } else {
            momentumBuffer[i] = this.momentum * momentumBuffer[i] + scaled;
            param.data[i] -= group.lr * momentumBuffer[i];
          }
        }
      }
    }

    zeroGrad(options: OptimizerConfig = {}) {
      zeroGrad(this.params, options);
    }

    zero_grad(options: OptimizerConfig = {}) {
      return this.zeroGrad(options);
    }

    setLearningRate(lr: unknown) {
      return setLearningRate(this, lr);
    }

    set_lr(lr: unknown) {
      return this.setLearningRate(lr);
    }

    getLearningRate() {
      return this.lr;
    }

    get_lr() {
      return this.getLearningRate();
    }

    addParamGroup(group: unknown) {
      if (!isParamGroup(group)) throw new Error("optimizer.addParamGroup requires { params, ...options }");
      const next = optimizerParamGroup(group, this.paramGroups.length, this.params.length, { lr: this.lr, weightDecay: this.weightDecay });
      validateUniqueOptimizerParams(next.params, new Set(this.params));
      this.params.push(...next.params);
      this.paramGroups = Object.freeze([...this.paramGroups, next.evidence]);
      this.squareAvg.push(...next.params.map((param) => new Float32Array(param.data.length)));
      if (this.momentum !== 0) {
        this.momentumBuffer.push(...next.params.map((param) => new Float32Array(param.data.length)));
      }
      return this;
    }

    add_param_group(group: unknown) {
      return this.addParamGroup(group);
    }

    get param_groups() {
      return this.paramGroups;
    }

    get defaults() {
      return this.config();
    }

    config() {
      const snapshot: OptimizerConfigSnapshot & {
        alpha: number;
        eps: number;
        momentum: number;
      } = {
        kind: "rmsprop",
        lr: this.lr,
        alpha: this.alpha,
        eps: this.eps,
        momentum: this.momentum,
        weightDecay: this.weightDecay,
        weight_decay: this.weightDecay,
        step: this.t,
        paramCount: this.params.length,
        paramGroups: this.paramGroups,
        param_groups: this.paramGroups,
      };
      snapshot.signature = optimizerConfigSnapshotSignature(snapshot);
      return Object.freeze(snapshot);
    }

    stateDict() {
      const entries: OptimizerStateEntry[] = [];
      for (let i = 0; i < this.params.length; i += 1) {
        entries.push(optimizerStateEntry(`square_avg.${i}`, this.params[i], this.squareAvg[i]));
        if (this.momentum !== 0) {
          entries.push(optimizerStateEntry(`momentum_buffer.${i}`, this.params[i], this.momentumBuffer[i]));
        }
      }
      return optimizerStateDict("rmsprop", this.params, entries, this.t);
    }

    state_dict() {
      return this.stateDict();
    }

    loadStateDict(source: unknown, loadOptions: OptimizerConfig = {}) {
      const { byName, entries, strict } = optimizerStateEntries(source, "rmsprop", this.params.length, loadOptions);
      const validateOnly = loadOptions.validateOnly === true;
      const step = optimizerStepFromState(source, this.t, strict);
      const expected = new Set<string>();
      const pending: OptimizerTensorStateUpdate[] = [];
      for (let i = 0; i < this.params.length; i += 1) {
        const squareAvgUpdate = loadOptimizerTensorState(byName, expected, `square_avg.${i}`, this.params[i], this.squareAvg[i], strict);
        if (squareAvgUpdate) pending.push(squareAvgUpdate);
        if (this.momentum !== 0) {
          const momentumUpdate = loadOptimizerTensorState(byName, expected, `momentum_buffer.${i}`, this.params[i], this.momentumBuffer[i], strict);
          if (momentumUpdate) pending.push(momentumUpdate);
        }
      }
      if (strict) rejectUnexpectedOptimizerState(entries, expected);
      for (const { target, data } of pending) {
        if (!validateOnly) target.set(data);
      }
      if (!validateOnly) this.t = step;
      return this;
    }

    load_state_dict(source: unknown, loadOptions: OptimizerConfig = {}) {
      return this.loadStateDict(source, loadOptions);
    }
  }

  class AdagradOptimizer {
    readonly kind = "adagrad";
    params: OptimizerParameter[];
    lr: number;
    lrDecay: number;
    eps: number;
    weightDecay: number;
    paramGroups: readonly OptimizerParamGroupEvidence[];
    t: number;
    sum: Float32Array[];

    constructor(paramsOrModule: unknown, config: OptimizerConfig = {}) {
      this.lr = finiteConfigNumber(config, "lr", 0.01, (x: number) => x > 0);
      this.lrDecay = finiteConfigNumber({ lrDecay: config.lrDecay ?? config.lr_decay }, "lrDecay", 0, (x: number) => x >= 0);
      this.eps = finiteConfigNumber(config, "eps", 1e-10, (x: number) => x > 0);
      this.weightDecay = weightDecay(config);
      const grouped = optimizerParamGroups(paramsOrModule, { lr: this.lr, weightDecay: this.weightDecay });
      this.params = grouped.params;
      this.paramGroups = Object.freeze(grouped.groups);
      this.t = 0;
      this.sum = this.params.map((param) => new Float32Array(param.data.length));
    }

    step() {
      this.t += 1;
      for (let p = 0; p < this.params.length; p += 1) {
        const param = this.params[p];
        const grad = param.grad;
        if (!grad || !parameterRequiresGrad(param)) continue;
        const sum = this.sum[p];
        const group = groupForIndex(this.paramGroups, p);
        const clr = group.lr / (1 + (this.t - 1) * this.lrDecay);
        for (let i = 0; i < param.data.length; i += 1) {
          const g = grad[i] + group.weightDecay * param.data[i];
          sum[i] += g * g;
          param.data[i] -= clr * g / (Math.sqrt(sum[i]) + this.eps);
        }
      }
    }

    zeroGrad(options: OptimizerConfig = {}) {
      zeroGrad(this.params, options);
    }

    zero_grad(options: OptimizerConfig = {}) {
      return this.zeroGrad(options);
    }

    setLearningRate(lr: unknown) {
      return setLearningRate(this, lr);
    }

    set_lr(lr: unknown) {
      return this.setLearningRate(lr);
    }

    getLearningRate() {
      return this.lr;
    }

    get_lr() {
      return this.getLearningRate();
    }

    addParamGroup(group: unknown) {
      if (!isParamGroup(group)) throw new Error("optimizer.addParamGroup requires { params, ...options }");
      const next = optimizerParamGroup(group, this.paramGroups.length, this.params.length, { lr: this.lr, weightDecay: this.weightDecay });
      validateUniqueOptimizerParams(next.params, new Set(this.params));
      this.params.push(...next.params);
      this.paramGroups = Object.freeze([...this.paramGroups, next.evidence]);
      this.sum.push(...next.params.map((param) => new Float32Array(param.data.length)));
      return this;
    }

    add_param_group(group: unknown) {
      return this.addParamGroup(group);
    }

    get param_groups() {
      return this.paramGroups;
    }

    get defaults() {
      return this.config();
    }

    config() {
      const snapshot: OptimizerConfigSnapshot & {
        lrDecay: number;
        lr_decay: number;
        eps: number;
      } = {
        kind: "adagrad",
        lr: this.lr,
        lrDecay: this.lrDecay,
        lr_decay: this.lrDecay,
        eps: this.eps,
        weightDecay: this.weightDecay,
        weight_decay: this.weightDecay,
        step: this.t,
        paramCount: this.params.length,
        paramGroups: this.paramGroups,
        param_groups: this.paramGroups,
      };
      snapshot.signature = optimizerConfigSnapshotSignature(snapshot);
      return Object.freeze(snapshot);
    }

    stateDict() {
      return optimizerStateDict(
        "adagrad",
        this.params,
        this.sum.map((sum, index) => optimizerStateEntry(`sum.${index}`, this.params[index], sum)),
        this.t,
      );
    }

    state_dict() {
      return this.stateDict();
    }

    loadStateDict(source: unknown, loadOptions: OptimizerConfig = {}) {
      const { byName, entries, strict } = optimizerStateEntries(source, "adagrad", this.params.length, loadOptions);
      const validateOnly = loadOptions.validateOnly === true;
      const step = optimizerStepFromState(source, this.t, strict);
      const expected = new Set<string>();
      const pending: OptimizerTensorStateUpdate[] = [];
      for (let i = 0; i < this.params.length; i += 1) {
        const update = loadOptimizerTensorState(byName, expected, `sum.${i}`, this.params[i], this.sum[i], strict);
        if (update) pending.push(update);
      }
      if (strict) rejectUnexpectedOptimizerState(entries, expected);
      for (const { target, data } of pending) {
        if (!validateOnly) target.set(data);
      }
      if (!validateOnly) this.t = step;
      return this;
    }

    load_state_dict(source: unknown, loadOptions: OptimizerConfig = {}) {
      return this.loadStateDict(source, loadOptions);
    }
  }

  return Object.freeze({ SgdOptimizer, AdamOptimizer, AdamWOptimizer, RMSpropOptimizer, AdagradOptimizer });
}
