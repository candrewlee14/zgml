"use strict";

import { createCheckpointSerializationHelpers } from "./checkpoint_serialization.js";
export { createLossTrainHelpers } from "./loss_train_helpers.js";
export { createOptimizerClasses } from "./optimizer_classes.js";
import type {
  CheckpointCreateOptions,
  CheckpointRestoreOptions,
  LoadStateDictOptions,
} from "../public_api.js";
import {
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
} from "./optimizer_snapshot.js";
import {
  createSchedulerStateHelpers,
  type SchedulerOptimizer,
} from "./scheduler_state.js";

type AnyRecord = Record<string, any>;

type CheckpointHookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

type CheckpointModuleState = Readonly<Record<string, AnyRecord>>;

type CheckpointOptimizerStateEntry = AnyRecord & {
  readonly name: string;
};

type CheckpointOptimizerState = AnyRecord & {
  readonly kind?: string;
  readonly step?: number;
  readonly paramCount?: number;
  readonly entries?: readonly CheckpointOptimizerStateEntry[];
};

type CheckpointSchedulerTarget = AnyRecord & {
  loadStateDict(source: unknown, options?: LoadStateDictOptions): unknown;
};

type CheckpointRestoreTargets = CheckpointRestoreOptions & { scheduler?: CheckpointSchedulerTarget };

export type CheckpointHelpersOptions = Readonly<{
  moduleStateDict: CheckpointHookCallback<[target: unknown, prefix?: string], CheckpointModuleState>;
  loadModuleStateDict: CheckpointHookCallback<[target: unknown, source: unknown, options?: LoadStateDictOptions], unknown>;
  optimizerStateDict: CheckpointHookCallback<[optimizer: unknown], CheckpointOptimizerState>;
  loadOptimizerStateDict: CheckpointHookCallback<[optimizer: unknown, source: unknown, options?: LoadStateDictOptions], unknown>;
}>;

export function createCheckpointHelpers(options: CheckpointHelpersOptions) {
  const moduleStateDict = options.moduleStateDict;
  const loadModuleStateDict = options.loadModuleStateDict;
  const optimizerStateDict = options.optimizerStateDict;
  const loadOptimizerStateDict = options.loadOptimizerStateDict;
  if (
    typeof moduleStateDict !== "function" ||
    typeof loadModuleStateDict !== "function" ||
    typeof optimizerStateDict !== "function" ||
    typeof loadOptimizerStateDict !== "function"
  ) {
    throw new Error("checkpoint helpers require module and optimizer state functions");
  }

  const {
    freezeArray,
    freezeRecord,
    plainRecord,
    jsonMetadata,
    checkpointTensorEntry,
    checkpointModuleStateFromJson,
    checkpointOptimizerStateFromJson,
    checkpointSchedulerStateFromJson,
    checkpointTensorInspection,
    checkpointInfo,
    tensorScalarCount,
  } = createCheckpointSerializationHelpers();

  function moduleState(target: unknown, prefix = "") {
    const source = moduleStateDict(target, prefix);
    const out: AnyRecord = {};
    for (const name of Object.keys(source)) {
      out[name] = checkpointTensorEntry(source[name], `checkpoint model ${name}`);
    }
    return freezeRecord(out);
  }

  function optimizerState(optimizer: unknown) {
    const source = optimizerStateDict(optimizer);
    const entries = Array.isArray(source.entries) ? source.entries : [];
    return freezeRecord({
      kind: source.kind,
      step: source.step,
      paramCount: source.paramCount,
      entries: freezeArray(entries.map((entry: AnyRecord, index: number) => {
        if (!entry || typeof entry.name !== "string") throw new Error(`checkpoint optimizer entry ${index} must have a name`);
        return checkpointTensorEntry(entry, `checkpoint optimizer ${entry.name}`, entry.name);
      })),
    });
  }

  function schedulerState(scheduler: unknown) {
    if (!scheduler || typeof (scheduler as AnyRecord).stateDict !== "function") {
      throw new Error("checkpoint scheduler requires stateDict()");
    }
    const source = (scheduler as AnyRecord).stateDict();
    return checkpointSchedulerStateFromJson(source);
  }

  function create(createOptions: CheckpointCreateOptions = {}) {
    const out: AnyRecord = {
      format: "zgml.checkpoint",
      version: 1,
    };
    if (createOptions.metadata !== undefined) out.metadata = jsonMetadata(createOptions.metadata);
    const prefix = createOptions.prefix === undefined ? "" : String(createOptions.prefix);
    if (createOptions.model !== undefined) out.model = moduleState(createOptions.model, prefix);
    if (createOptions.optimizer !== undefined) out.optimizer = optimizerState(createOptions.optimizer);
    if (createOptions.scheduler !== undefined) out.scheduler = schedulerState(createOptions.scheduler);
    if (!out.model && !out.optimizer && !out.scheduler) throw new Error("checkpoint.create requires a model, optimizer, scheduler, or a combination");
    return freezeRecord(out);
  }

  function requireCheckpoint(snapshot: unknown) {
    const source = plainRecord(snapshot, "checkpoint");
    if (source.format !== "zgml.checkpoint") throw new Error(`checkpoint format must be zgml.checkpoint, got ${source.format}`);
    if (source.version !== 1) throw new Error(`checkpoint version must be 1, got ${source.version}`);
    return source;
  }

  function fromJSON(value: unknown) {
    const source = requireCheckpoint(value);
    const out: AnyRecord = {
      format: "zgml.checkpoint",
      version: 1,
    };
    if (source.metadata !== undefined) out.metadata = jsonMetadata(source.metadata);
    const model = checkpointModuleStateFromJson(source.model);
    const optimizer = checkpointOptimizerStateFromJson(source.optimizer);
    const scheduler = checkpointSchedulerStateFromJson(source.scheduler);
    if (model !== undefined) out.model = model;
    if (optimizer !== undefined) out.optimizer = optimizer;
    if (scheduler !== undefined) out.scheduler = scheduler;
    if (!out.model && !out.optimizer && !out.scheduler) throw new Error("checkpoint JSON requires model, optimizer, scheduler, or a combination");
    return freezeRecord(out);
  }

  function toJSON(snapshot: unknown) {
    return fromJSON(snapshot);
  }

  function stringify(snapshot: unknown, space?: string | number) {
    return JSON.stringify(toJSON(snapshot), null, space);
  }

  function parse(text: unknown) {
    if (typeof text !== "string") throw new Error("checkpoint.parse requires a JSON string");
    return fromJSON(JSON.parse(text));
  }

  function restore(snapshot: unknown, targets: CheckpointRestoreTargets = {}) {
    const source = fromJSON(snapshot);
    const strict = targets.strict !== false;
    const prefix = targets.prefix ?? "";
    const loadOptions = prefix ? { strict, prefix } : { strict };
    const validateOptions = prefix ? { strict, prefix, validateOnly: true } : { strict, validateOnly: true };
    if (source.model !== undefined) {
      if (targets.model === undefined) throw new Error("checkpoint.restore requires a model target for checkpoint model state");
    }
    if (source.optimizer !== undefined) {
      if (targets.optimizer === undefined) throw new Error("checkpoint.restore requires an optimizer target for checkpoint optimizer state");
    }
    const schedulerTarget = targets.scheduler;
    if (source.scheduler !== undefined) {
      if (schedulerTarget === undefined) throw new Error("checkpoint.restore requires a scheduler target for checkpoint scheduler state");
      if (typeof schedulerTarget.loadStateDict !== "function") throw new Error("checkpoint.restore scheduler target requires loadStateDict()");
    }
    if (source.model !== undefined) {
      loadModuleStateDict(targets.model, source.model, validateOptions);
    }
    if (source.optimizer !== undefined) {
      loadOptimizerStateDict(targets.optimizer, source.optimizer, validateOptions);
    }
    if (source.scheduler !== undefined) {
      schedulerTarget!.loadStateDict(source.scheduler, { strict, validateOnly: true });
    }
    if (source.model !== undefined) {
      loadModuleStateDict(targets.model, source.model, loadOptions);
    }
    if (source.optimizer !== undefined) {
      loadOptimizerStateDict(targets.optimizer, source.optimizer, loadOptions);
    }
    if (source.scheduler !== undefined) {
      schedulerTarget!.loadStateDict(source.scheduler, { strict });
    }
    return targets;
  }

  function inspect(snapshot: unknown) {
    const source = fromJSON(snapshot);
    const modelNames = source.model ? Object.keys(source.model) : [];
    const modelParameters = modelNames.map((name, index) => checkpointTensorInspection(name, source.model[name], index));
    const modelScalarCount = modelNames.reduce((acc, name) => acc + tensorScalarCount(source.model[name]), 0);
    const optimizerEntries = source.optimizer && Array.isArray(source.optimizer.entries) ? source.optimizer.entries : [];
    const optimizerNames = optimizerEntries.map((entry: AnyRecord) => entry.name);
    const optimizerEntryInfos = optimizerEntries.map((entry: AnyRecord, index: number) => checkpointTensorInspection(entry.name, entry, index));
    const optimizerScalarCount = optimizerEntries.reduce((acc: number, entry: AnyRecord) => acc + tensorScalarCount(entry), 0);
    const inspection: AnyRecord = {
      format: source.format,
      version: source.version,
      hasMetadata: source.metadata !== undefined,
      hasModel: source.model !== undefined,
      hasOptimizer: source.optimizer !== undefined,
      hasScheduler: source.scheduler !== undefined,
      modelParameterCount: modelNames.length,
      modelScalarCount,
      modelParameterNames: freezeArray(modelNames),
      modelParameters: freezeArray(modelParameters),
      optimizerKind: source.optimizer?.kind ?? null,
      optimizerStep: source.optimizer?.step ?? null,
      optimizerParamCount: source.optimizer?.paramCount ?? 0,
      optimizerEntryCount: optimizerEntries.length,
      optimizerScalarCount,
      optimizerEntryNames: freezeArray(optimizerNames),
      optimizerEntries: freezeArray(optimizerEntryInfos),
      schedulerKind: source.scheduler?.kind ?? null,
      schedulerStep: source.scheduler?.step ?? null,
      schedulerBaseLr: source.scheduler?.baseLr ?? null,
      schedulerLastLr: source.scheduler?.lastLr ?? null,
      schedulerGamma: source.scheduler?.gamma ?? null,
      schedulerStepSize: source.scheduler?.stepSize ?? null,
      schedulerTMax: source.scheduler?.tMax ?? null,
      schedulerT_max: source.scheduler?.t_max ?? source.scheduler?.tMax ?? null,
      schedulerEtaMin: source.scheduler?.etaMin ?? null,
      schedulerEta_min: source.scheduler?.eta_min ?? source.scheduler?.etaMin ?? null,
      schedulerOptimizerKind: source.scheduler?.optimizerKind ?? null,
    };
    inspection.signature = [
      "checkpoint-inspection",
      `format=${inspection.format}`,
      `version=${inspection.version}`,
      `metadata=${inspection.hasMetadata ? 1 : 0}`,
      `model=${inspection.hasModel ? 1 : 0}`,
      `optimizer=${inspection.hasOptimizer ? 1 : 0}`,
      `scheduler=${inspection.hasScheduler ? 1 : 0}`,
      `modelScalars=${inspection.modelScalarCount}`,
      `modelEntries=${modelParameters.map((entry: AnyRecord) => entry.signature).join(",")}`,
      `optimizerKind=${inspection.optimizerKind ?? "null"}`,
      `optimizerStep=${inspection.optimizerStep ?? "null"}`,
      `optimizerScalars=${inspection.optimizerScalarCount}`,
      `optimizerEntries=${optimizerEntryInfos.map((entry: AnyRecord) => entry.signature).join(",")}`,
      `schedulerKind=${inspection.schedulerKind ?? "null"}`,
      `schedulerStep=${inspection.schedulerStep ?? "null"}`,
      `schedulerLastLr=${inspection.schedulerLastLr ?? "null"}`,
      `schedulerTMax=${inspection.schedulerTMax ?? "null"}`,
      `schedulerEtaMin=${inspection.schedulerEtaMin ?? "null"}`,
    ].join("|");
    return freezeRecord(inspection);
  }

  function modelParameterInfo(snapshot: unknown, nameOrIndex: unknown) {
    return checkpointInfo(inspect(snapshot).modelParameters as readonly AnyRecord[], nameOrIndex, "model parameter");
  }

  function optimizerEntryInfo(snapshot: unknown, nameOrIndex: unknown) {
    return checkpointInfo(inspect(snapshot).optimizerEntries as readonly AnyRecord[], nameOrIndex, "optimizer entry");
  }

  return Object.freeze({
    create,
    inspect,
    modelParameterInfo,
    optimizerEntryInfo,
    toJSON,
    fromJSON,
    stringify,
    serialize: stringify,
    parse,
    deserialize: parse,
    restore,
    load: restore,
    moduleStateDict: moduleState,
    module_state_dict: moduleState,
    optimizerStateDict: optimizerState,
    optimizer_state_dict: optimizerState,
    schedulerStateDict: schedulerState,
    scheduler_state_dict: schedulerState,
  });
}

type OptimNamespaceConfig = Readonly<Record<string, unknown>>;

type OptimizerInstance = SchedulerOptimizer & {
  config(): AnyRecord;
  setLearningRate(lr: unknown): unknown;
  set_lr(lr: unknown): unknown;
  getLearningRate(): unknown;
  get_lr(): unknown;
  addParamGroup(group: unknown): unknown;
  add_param_group(group: unknown): unknown;
  stateDict(): unknown;
  state_dict(): unknown;
  loadStateDict(source: unknown, loadOptions?: OptimNamespaceConfig): unknown;
  load_state_dict(source: unknown, loadOptions?: OptimNamespaceConfig): unknown;
};

type OptimizerConstructor = new (paramsOrModule: any, config?: any) => AnyRecord;

type OptimNamespaceZeroGrad = (paramsOrModule: unknown, options?: OptimNamespaceConfig) => unknown;

export type OptimNamespaceOptions = Readonly<{
  SgdOptimizer: OptimizerConstructor;
  AdamOptimizer: OptimizerConstructor;
  AdamWOptimizer: OptimizerConstructor;
  RMSpropOptimizer: OptimizerConstructor;
  AdagradOptimizer: OptimizerConstructor;
  zeroGrad: OptimNamespaceZeroGrad;
}>;

export function createOptimNamespace(options: OptimNamespaceOptions) {
  const SgdOptimizer = options.SgdOptimizer;
  const AdamOptimizer = options.AdamOptimizer;
  const AdamWOptimizer = options.AdamWOptimizer;
  const RMSpropOptimizer = options.RMSpropOptimizer;
  const AdagradOptimizer = options.AdagradOptimizer;
  const zeroGrad = options.zeroGrad;
  if (
    typeof SgdOptimizer !== "function" ||
    typeof AdamOptimizer !== "function" ||
    typeof AdamWOptimizer !== "function" ||
    typeof RMSpropOptimizer !== "function" ||
    typeof AdagradOptimizer !== "function" ||
    typeof zeroGrad !== "function"
  ) {
    throw new Error("optim namespace factory requires optimizer constructors and zeroGrad");
  }

  function finiteSchedulerNumber(value: unknown, name: string, fallback: number, valid: (value: number) => boolean) {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isFinite(out) || !valid(out)) throw new Error(`optim scheduler ${name} must be valid, got ${value}`);
    return out;
  }

  function positiveSchedulerInteger(value: unknown, name: string, fallback: number) {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isSafeInteger(out) || out <= 0) throw new Error(`optim scheduler ${name} must be a positive safe integer, got ${value}`);
    return out;
  }

  function nonNegativeSchedulerInteger(value: unknown, name: string, fallback: number) {
    const out = value === undefined ? fallback : Number(value);
    if (!Number.isSafeInteger(out) || out < 0) throw new Error(`optim scheduler ${name} must be a non-negative safe integer, got ${value}`);
    return out;
  }

  const { optimizerLearningRate, schedulerState, loadSchedulerState } = createSchedulerStateHelpers();

  class StepLRScheduler {
    optimizer: SchedulerOptimizer;
    stepSize: number;
    gamma: number;
    baseLr: number;
    lastLr: number;
    t: number;

    constructor(optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) {
      if (!optimizer || typeof optimizer.config !== "function" || typeof optimizer.setLearningRate !== "function") {
        throw new Error("optim.stepLR requires an optimizer with config() and setLearningRate()");
      }
      this.optimizer = optimizer;
      this.stepSize = positiveSchedulerInteger(config.stepSize ?? config.step_size, "stepSize", 1);
      this.gamma = finiteSchedulerNumber(config.gamma, "gamma", 0.1, (value) => value > 0);
      this.baseLr = optimizerLearningRate(optimizer);
      this.lastLr = this.baseLr;
      this.t = 0;
    }

    optimizerKind() {
      return this.optimizer.config().kind ?? null;
    }

    get base_lr() {
      return this.baseLr;
    }

    get last_lr() {
      return this.lastLr;
    }

    get step_size() {
      return this.stepSize;
    }

    step(_metric?: unknown) {
      this.t += 1;
      if (this.t % this.stepSize === 0) {
        this.lastLr *= this.gamma;
        this.optimizer.setLearningRate(this.lastLr);
      } else {
        this.lastLr = optimizerLearningRate(this.optimizer);
      }
      return this.lastLr;
    }

    getLastLr() {
      return this.lastLr;
    }

    get_last_lr() {
      return this.getLastLr();
    }

    config() {
      return schedulerState("step-lr", this);
    }

    stateDict() {
      return this.config();
    }

    state_dict() {
      return this.stateDict();
    }

    loadStateDict(source: unknown, loadOptions: OptimNamespaceConfig = {}) {
      return loadSchedulerState(this, source, "step-lr", loadOptions.strict !== false, loadOptions.validateOnly === true);
    }

    load_state_dict(source: unknown, loadOptions: OptimNamespaceConfig = {}) {
      return this.loadStateDict(source, loadOptions);
    }
  }

  class ExponentialLRScheduler extends StepLRScheduler {
    constructor(optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) {
      super(optimizer, { ...config, stepSize: 1 });
      this.gamma = finiteSchedulerNumber(config.gamma, "gamma", 0.95, (value) => value > 0);
      this.stepSize = 1;
    }

    config() {
      return schedulerState("exponential-lr", this);
    }

    stateDict() {
      return this.config();
    }

    loadStateDict(source: unknown, loadOptions: OptimNamespaceConfig = {}) {
      return loadSchedulerState(this, source, "exponential-lr", loadOptions.strict !== false, loadOptions.validateOnly === true);
    }
  }

  class CosineAnnealingLRScheduler extends StepLRScheduler {
    tMax: number;
    etaMin: number;

    constructor(optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) {
      const tMax = positiveSchedulerInteger(config.tMax ?? config.t_max, "tMax", 10);
      super(optimizer, { ...config, stepSize: tMax, gamma: 1 });
      this.tMax = tMax;
      this.etaMin = finiteSchedulerNumber(config.etaMin ?? config.eta_min, "etaMin", 0, (value) => value >= 0);
      this.gamma = 1;
      this.stepSize = tMax;
    }

    get t_max() {
      return this.tMax;
    }

    get eta_min() {
      return this.etaMin;
    }

    step(_metric?: unknown) {
      this.t += 1;
      const phase = Math.min(this.t, this.tMax);
      this.lastLr = this.etaMin + ((this.baseLr - this.etaMin) * (1 + Math.cos((Math.PI * phase) / this.tMax))) / 2;
      this.optimizer.setLearningRate(this.lastLr);
      return this.lastLr;
    }

    config() {
      return schedulerState("cosine-annealing-lr", this);
    }

    stateDict() {
      return this.config();
    }

    loadStateDict(source: unknown, loadOptions: OptimNamespaceConfig = {}) {
      return loadSchedulerState(this, source, "cosine-annealing-lr", loadOptions.strict !== false, loadOptions.validateOnly === true);
    }
  }

  class ReduceLROnPlateauScheduler extends StepLRScheduler {
    mode: "min" | "max";
    factor: number;
    patience: number;
    threshold: number;
    thresholdMode: "rel" | "abs";
    cooldown: number;
    cooldownCounter: number;
    minLr: number;
    eps: number;
    best: number | undefined;
    badEpochs: number;

    constructor(optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) {
      super(optimizer, { ...config, stepSize: 1, gamma: 1 });
      const mode = config.mode === undefined ? "min" : String(config.mode);
      if (mode !== "min" && mode !== "max") throw new Error(`optim scheduler mode must be min or max, got ${config.mode}`);
      const thresholdMode = config.thresholdMode ?? config.threshold_mode;
      const normalizedThresholdMode = thresholdMode === undefined ? "rel" : String(thresholdMode);
      if (normalizedThresholdMode !== "rel" && normalizedThresholdMode !== "abs") {
        throw new Error(`optim scheduler thresholdMode must be rel or abs, got ${thresholdMode}`);
      }
      this.mode = mode;
      this.factor = finiteSchedulerNumber(config.factor, "factor", 0.1, (value) => value > 0 && value < 1);
      this.patience = nonNegativeSchedulerInteger(config.patience, "patience", 10);
      this.threshold = finiteSchedulerNumber(config.threshold, "threshold", 1e-4, (value) => value >= 0);
      this.thresholdMode = normalizedThresholdMode;
      this.cooldown = nonNegativeSchedulerInteger(config.cooldown, "cooldown", 0);
      this.cooldownCounter = 0;
      this.minLr = finiteSchedulerNumber(config.minLr ?? config.min_lr, "minLr", 0, (value) => value >= 0);
      this.eps = finiteSchedulerNumber(config.eps, "eps", 1e-8, (value) => value >= 0);
      this.best = undefined;
      this.badEpochs = 0;
      this.gamma = 1;
      this.stepSize = 1;
    }

    get threshold_mode() {
      return this.thresholdMode;
    }

    get cooldown_counter() {
      return this.cooldownCounter;
    }

    get min_lr() {
      return this.minLr;
    }

    get bad_epochs() {
      return this.badEpochs;
    }

    isBetter(metric: number) {
      if (this.best === undefined) return true;
      if (this.mode === "min") {
        return this.thresholdMode === "rel"
          ? metric < this.best * (1 - this.threshold)
          : metric < this.best - this.threshold;
      }
      return this.thresholdMode === "rel"
        ? metric > this.best * (1 + this.threshold)
        : metric > this.best + this.threshold;
    }

    step(metric?: unknown) {
      const value = Number(metric);
      if (!Number.isFinite(value)) throw new Error(`optim.reduceLROnPlateau step metric must be finite, got ${metric}`);
      this.t += 1;
      if (this.isBetter(value)) {
        this.best = value;
        this.badEpochs = 0;
      } else {
        this.badEpochs += 1;
      }
      if (this.cooldownCounter > 0) {
        this.cooldownCounter -= 1;
        this.badEpochs = 0;
      }
      if (this.badEpochs > this.patience) {
        const nextLr = Math.max(this.lastLr * this.factor, this.minLr);
        if (this.lastLr - nextLr > this.eps) {
          this.lastLr = nextLr;
          this.optimizer.setLearningRate(this.lastLr);
        }
        this.cooldownCounter = this.cooldown;
        this.badEpochs = 0;
      }
      return this.lastLr;
    }

    config() {
      return schedulerState("reduce-lr-on-plateau", this);
    }

    stateDict() {
      return this.config();
    }

    loadStateDict(source: unknown, loadOptions: OptimNamespaceConfig = {}) {
      return loadSchedulerState(this, source, "reduce-lr-on-plateau", loadOptions.strict !== false, loadOptions.validateOnly === true);
    }
  }

  const lrSchedulerNamespace = Object.freeze({
    stepLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new StepLRScheduler(optimizer, config),
    step_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new StepLRScheduler(optimizer, config),
    StepLR: StepLRScheduler,
    exponentialLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ExponentialLRScheduler(optimizer, config),
    exponential_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ExponentialLRScheduler(optimizer, config),
    ExponentialLR: ExponentialLRScheduler,
    cosineAnnealingLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new CosineAnnealingLRScheduler(optimizer, config),
    cosine_annealing_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new CosineAnnealingLRScheduler(optimizer, config),
    CosineAnnealingLR: CosineAnnealingLRScheduler,
    reduceLROnPlateau: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ReduceLROnPlateauScheduler(optimizer, config),
    reduce_lr_on_plateau: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ReduceLROnPlateauScheduler(optimizer, config),
    ReduceLROnPlateau: ReduceLROnPlateauScheduler,
  });

  return Object.freeze({
    sgd: (paramsOrModule: unknown, config: OptimNamespaceConfig = {}) => new SgdOptimizer(paramsOrModule, config),
    adam: (paramsOrModule: unknown, config: OptimNamespaceConfig = {}) => new AdamOptimizer(paramsOrModule, config),
    adamW: (paramsOrModule: unknown, config: OptimNamespaceConfig = {}) => new AdamWOptimizer(paramsOrModule, config),
    rmsprop: (paramsOrModule: unknown, config: OptimNamespaceConfig = {}) => new RMSpropOptimizer(paramsOrModule, config),
    adagrad: (paramsOrModule: unknown, config: OptimNamespaceConfig = {}) => new AdagradOptimizer(paramsOrModule, config),
    SGD: SgdOptimizer,
    Adam: AdamOptimizer,
    AdamW: AdamWOptimizer,
    RMSprop: RMSpropOptimizer,
    Adagrad: AdagradOptimizer,
    zeroGrad,
    zero_grad: zeroGrad,
    config: (optimizer: OptimizerInstance) => optimizer.config(),
    isOptimizerConfigSnapshot,
    requireOptimizerConfigSnapshot,
    assertOptimizerConfigSnapshot,
    assert_optimizer_config_snapshot,
    matchesOptimizerConfigSnapshotSignature,
    matches_optimizer_config_snapshot_signature,
    optimizerConfigSnapshotSignature,
    setLearningRate: (optimizer: OptimizerInstance, lr: unknown) => optimizer.setLearningRate(lr),
    set_lr: (optimizer: OptimizerInstance, lr: unknown) => optimizer.set_lr(lr),
    getLearningRate: (optimizer: OptimizerInstance) => optimizer.getLearningRate(),
    get_lr: (optimizer: OptimizerInstance) => optimizer.get_lr(),
    addParamGroup: (optimizer: OptimizerInstance, group: unknown) => optimizer.addParamGroup(group),
    add_param_group: (optimizer: OptimizerInstance, group: unknown) => optimizer.add_param_group(group),
    stateDict: (optimizer: OptimizerInstance) => optimizer.stateDict(),
    state_dict: (optimizer: OptimizerInstance) => optimizer.state_dict(),
    isOptimizerStateSnapshot,
    requireOptimizerStateSnapshot,
    assertOptimizerStateSnapshot,
    assert_optimizer_state_snapshot,
    matchesOptimizerStateSnapshotSignature,
    matches_optimizer_state_snapshot_signature,
    optimizerStateSnapshotSignature,
    loadStateDict: (optimizer: OptimizerInstance, source: unknown, loadOptions: OptimNamespaceConfig = {}) => optimizer.loadStateDict(source, loadOptions),
    load_state_dict: (optimizer: OptimizerInstance, source: unknown, loadOptions: OptimNamespaceConfig = {}) => optimizer.load_state_dict(source, loadOptions),
    stepLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new StepLRScheduler(optimizer, config),
    step_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new StepLRScheduler(optimizer, config),
    StepLR: StepLRScheduler,
    exponentialLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ExponentialLRScheduler(optimizer, config),
    exponential_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ExponentialLRScheduler(optimizer, config),
    ExponentialLR: ExponentialLRScheduler,
    cosineAnnealingLR: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new CosineAnnealingLRScheduler(optimizer, config),
    cosine_annealing_lr: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new CosineAnnealingLRScheduler(optimizer, config),
    CosineAnnealingLR: CosineAnnealingLRScheduler,
    reduceLROnPlateau: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ReduceLROnPlateauScheduler(optimizer, config),
    reduce_lr_on_plateau: (optimizer: OptimizerInstance, config: OptimNamespaceConfig = {}) => new ReduceLROnPlateauScheduler(optimizer, config),
    ReduceLROnPlateau: ReduceLROnPlateauScheduler,
    lrScheduler: lrSchedulerNamespace,
    lr_scheduler: lrSchedulerNamespace,
    isLRSchedulerStateSnapshot,
    requireLRSchedulerStateSnapshot,
    assertLRSchedulerStateSnapshot,
    assert_lr_scheduler_state_snapshot,
    matchesLRSchedulerStateSnapshotSignature,
    matches_lr_scheduler_state_snapshot_signature,
    lrSchedulerStateSnapshotSignature,
  });
}
