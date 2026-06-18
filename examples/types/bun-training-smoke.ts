import {
  checkpoint,
  clone,
  data,
  detach,
  enable_grad,
  F as bunRootF,
  gradMode as bunGradMode,
  inference_mode,
  is_grad_enabled,
  loss,
  nn,
  no_grad,
  optim,
  set_grad_enabled,
  tensor,
  train,
  type DataLoader,
  type CheckpointInspection,
  type BCELoss,
  type BCEWithLogitsLoss,
  type Conv2dModule,
  type CrossEntropyLoss,
  type GradModeNamespace,
  type HuberLoss,
  type L1Loss,
  type LRScheduler,
  type LRSchedulerStateSnapshot,
  type LossReduction,
  type MSELoss,
  type NLLLoss,
  type SmoothL1Loss,
  type ModuleCompileSupport,
  type NnBuffer,
  type ModuleParameterInfo,
  type ModuleTraversalEntry,
  type ModuleStateSnapshot,
  type NnFunctionalNamespace,
  type NnModule,
  type NnParameter,
  type Optimizer,
  type OptimizerConfigSnapshot,
  type OptimizerParamGroupInput,
  type OptimizerParamGroupSnapshot,
  type OptimizerStateSnapshot,
  type Program,
  type SequentialModule,
  type Session,
  type Tensor,
  type TensorDataset,
  type TensorDatasetBatch,
  type TensorDatasetMapper,
  type TrainClassificationCriterion,
  type TrainClassificationReport,
  type TrainEvaluateEvidence,
  type TrainEvaluateStepEvidence,
  type TrainPredictEvidence,
  type TrainPredictStepEvidence,
  type TrainFitContext,
  type TrainFitEvidence,
  type TrainFitStepEvidence,
  type TrainSupervisedCriterion,
  type TrainStepEvidence,
  type ZgmlCheckpoint,
} from "zgml/bun";

const bunModel = nn.sequential([
  nn.linear(2, 3),
  nn.gelu(),
  nn.linear(3, 1),
]);
const bunClassStyleModel = new nn.Sequential([
  new nn.Linear(2, 3),
  new nn.GELU(),
  new nn.Linear(3, 1),
]);
const bunVariadicClassStyleModel = new nn.Sequential(
  new nn.Linear(2, 3),
  new nn.GELU(),
  new nn.Linear(3, 1),
);
const bunClassStyleOptimizer: Optimizer<"adamw"> = new optim.AdamW(bunClassStyleModel, { lr: 0.01, weightDecay: 0.001 });
const bunRmspropModel = new nn.Sequential(new nn.Linear(2, 1));
const bunRmspropOptimizer: Optimizer<"rmsprop"> = optim.rmsprop(bunRmspropModel, { lr: 0.01, alpha: 0.9, momentum: 0.1, weightDecay: 0.001 });
const bunRmspropClassOptimizer: Optimizer<"rmsprop"> = new optim.RMSprop(bunRmspropModel, { lr: 0.01, eps: 1e-8 });
const bunRmspropConfig: OptimizerConfigSnapshot<"rmsprop"> = bunRmspropOptimizer.config();
const bunRmspropState: OptimizerStateSnapshot<"rmsprop"> = bunRmspropOptimizer.stateDict();
const bunAdagradModel = new nn.Sequential(new nn.Linear(2, 1));
const bunAdagradOptimizer: Optimizer<"adagrad"> = optim.adagrad(bunAdagradModel, { lr: 0.01, lr_decay: 0.001, weightDecay: 0.001 });
const bunAdagradClassOptimizer: Optimizer<"adagrad"> = new optim.Adagrad(bunAdagradModel, { lr: 0.01, eps: 1e-10 });
const bunAdagradConfig: OptimizerConfigSnapshot<"adagrad"> = bunAdagradOptimizer.config();
const bunAdagradState: OptimizerStateSnapshot<"adagrad"> = bunAdagradOptimizer.stateDict();
const bunClassStyleCriterion: MSELoss = new loss.MSELoss();
const bunClassStyleSumCriterion: MSELoss = new loss.MSELoss({ reduction: "sum" });
const bunClassStyleSumReduction: LossReduction = bunClassStyleSumCriterion.reduction;
const bunNnClassStyleCriterion: MSELoss = new nn.MSELoss();
const bunNnClassStyleSumCriterion: MSELoss = new nn.MSELoss({ reduction: "sum" });
const bunNnClassStyleSumReduction: LossReduction = bunNnClassStyleSumCriterion.reduction;
const bunClassStylePrediction: Tensor<readonly [1]> = bunClassStyleModel.forward(tensor([1, 0], [2] as const));
const bunVariadicClassStylePrediction: Tensor<readonly [1]> = bunVariadicClassStyleModel.forward(tensor([1, 0], [2] as const));
const bunSequentialLen: number = bunVariadicClassStyleModel.__len__();
const bunSequentialSize: number = bunVariadicClassStyleModel.size();
const bunSequentialItem: NnModule = bunVariadicClassStyleModel.__getitem__(1);
const bunSequentialChildren: readonly NnModule[] = bunVariadicClassStyleModel.children();
const bunSequentialModules: readonly NnModule[] = bunVariadicClassStyleModel.modules();
const bunSequentialNamedChildren: readonly ModuleTraversalEntry[] = bunVariadicClassStyleModel.named_children("bunSequential");
const bunSequentialNamedModules: readonly ModuleTraversalEntry[] = bunVariadicClassStyleModel.named_modules("bunSequential");
const bunSequentialApply: SequentialModule = bunVariadicClassStyleModel.apply((_module, _entry) => {});
const bunSequentialTrain: SequentialModule = bunVariadicClassStyleModel.train();
const bunSequentialEval: SequentialModule = bunVariadicClassStyleModel.eval();
const bunSequentialPoppedLayer: NnModule | undefined = new nn.Sequential(nn.relu(), nn.tanh()).pop();
const bunSequentialCleared: SequentialModule = new nn.Sequential(nn.relu(), nn.tanh()).clear();
const bunClassStyleObjective: Tensor<readonly [1]> = bunClassStyleCriterion.forward(bunClassStylePrediction, tensor([1], [1] as const));
const bunClassStyleSumObjective: Tensor<readonly [1]> = bunClassStyleSumCriterion.forward(bunClassStylePrediction, tensor([1], [1] as const));
const bunNnClassStyleObjective: Tensor<readonly [1]> = bunNnClassStyleCriterion.forward(bunClassStylePrediction, tensor([1], [1] as const));
const bunNnClassStyleSumObjective: Tensor<readonly [1]> = bunNnClassStyleSumCriterion.forward(bunClassStylePrediction, tensor([1], [1] as const));
bunClassStyleOptimizer.zero_grad();
bunClassStyleObjective.backward();
bunClassStyleOptimizer.step();
const bunRootFunctionalAlias: NnFunctionalNamespace = bunRootF;
const bunRootFunctionalSoftmax: Tensor<readonly [1]> = bunRootFunctionalAlias.softmax(tensor([1], [1] as const));

const bunOptimizer: Optimizer<"sgd"> = optim.sgd(bunModel, { lr: 0.04, momentum: 0.8 });
const bunScheduler: LRScheduler<"step-lr", "sgd"> = optim.stepLR(bunOptimizer, { stepSize: 2, gamma: 0.5 });
const bunExponentialScheduler: LRScheduler<"exponential-lr", "sgd"> = optim.exponentialLR(bunOptimizer, { gamma: 0.95 });
const bunCosineScheduler: LRScheduler<"cosine-annealing-lr", "sgd"> = optim.cosineAnnealingLR(bunOptimizer, { tMax: 4, etaMin: 0 });
const bunPlateauScheduler: LRScheduler<"reduce-lr-on-plateau", "sgd"> = optim.reduceLROnPlateau(bunOptimizer, { mode: "min", factor: 0.5, patience: 1 });
const bunSchedulerNamespace: LRScheduler<"step-lr", "sgd"> = new optim.lr_scheduler.StepLR(bunOptimizer, { step_size: 2, gamma: 0.5 });
const bunExponentialSchedulerNamespace: LRScheduler<"exponential-lr", "sgd"> = new optim.lrScheduler.ExponentialLR(bunOptimizer, { gamma: 0.95 });
const bunCosineSchedulerNamespace: LRScheduler<"cosine-annealing-lr", "sgd"> = new optim.lrScheduler.CosineAnnealingLR(bunOptimizer, { t_max: 4, eta_min: 0 });
const bunPlateauSchedulerNamespace: LRScheduler<"reduce-lr-on-plateau", "sgd"> = new optim.lrScheduler.ReduceLROnPlateau(bunOptimizer, { threshold_mode: "abs", min_lr: 0 });
const bunSchedulerBaseLrAlias: number = bunScheduler.base_lr;
const bunSchedulerLastLrAlias: number = bunScheduler.last_lr;
const bunSchedulerStepSizeAlias: number = bunScheduler.step_size;
const bunDataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const bunLoader = data.dataLoader(bunDataset, { batchSize: 2, shuffle: true, seed: 23 });
const bunClassDataset: TensorDataset = new data.TensorDataset(
  tensor([0, 1, 1, 0], [2, 2] as const),
  tensor([1, 0], [2, 1] as const),
);
const bunClassLoader: DataLoader = new data.DataLoader(bunClassDataset, { batch_size: 2, shuffle: false });
const bunDatasetLen: number = bunDataset.__len__();
const bunDatasetSampleAt: TensorDatasetBatch["input"] = bunLoader.at(0).input;
const bunDatasetSampleDunder: TensorDataset["sample"] = bunDataset.__getitem__;
const bunDatasetIterableInput: Tensor = Array.from(bunDataset)[0]!.input;
const bunDatasetDunderIterInput: Tensor = bunDataset.__iter__().next().value!.input;
const bunDatasetSplitPair: readonly [TensorDataset, TensorDataset] = data.randomSplit(bunDataset, [3, 1], { seed: 29 });
const bunDatasetSplitPairAlias: readonly [TensorDataset, TensorDataset] = data.random_split(bunDataset, [2, 2], { shuffle: false });
const bunDatasetIterableSplit: readonly TensorDataset[] = data.randomSplit(bunDataset, new Set([3, 1]), { seed: 29 });
const bunDatasetSubset: TensorDataset = data.subset(bunDataset, [1, 0]);
const bunDatasetIterableSubset: TensorDataset = data.subset(bunDataset, new Set([1, 0]));
const bunDatasetTake: TensorDataset = data.take(bunDataset, 2);
const bunDatasetConcat: TensorDataset = data.concatDataset([bunDatasetTake, bunDatasetSubset]);
const bunDatasetIterableConcat: TensorDataset = data.concatDataset(new Set([bunDatasetTake, bunDatasetIterableSubset]));
const bunDatasetConcatAlias: TensorDataset = data.concat_dataset([bunDatasetSubset, bunDatasetTake]);
const bunDatasetMapper: TensorDatasetMapper = (sample) => ({ input: sample.input, target: sample.target });
const bunDatasetMapped: TensorDataset = data.mapDataset(bunDataset, bunDatasetMapper);
const bunDatasetMappedAlias: TensorDataset = data.map_dataset(bunDataset, (sample) => ({ input: sample.input }));
const bunDatasetSplitLoader = data.dataLoader(bunDatasetSplitPair[0], { batch_size: 2 });
const bunLoaderBatchSizeAlias: number = bunLoader.batch_size;
const bunLoaderDropLastAlias: boolean = bunLoader.drop_last;

const bunFit: TrainFitEvidence<"sgd"> = train.fit(bunOptimizer, bunLoader, (batch: TensorDatasetBatch, context: TrainFitContext) => {
  if (!batch.target) throw new Error("training batch requires a target tensor");
  const prediction: Tensor = bunModel.forward(batch.input);
  const objective: Tensor<readonly [1]> = loss.mse(prediction, batch.target);
  const step: number = context.step;
  void step;
  return objective;
}, {
  epochs: 2,
  zeroGrad: true,
  clipGradNorm: 1,
  clipGradValue: 0.5,
  early_stopping: { patience: 2, min_delta: 0, mode: "min" },
  onStep(step: TrainFitStepEvidence<"sgd">) {
    const optimizerKind: "sgd" | null = step.stepEvidence?.optimizerKind ?? null;
    const clipped: boolean = step.stepEvidence?.clipGradNormApplied ?? false;
    const schedulerLr: number = bunScheduler.step();
    void optimizerKind;
    void clipped;
    void schedulerLr;
  },
});
const bunFitStopReason: "max-steps" | "early-stopping" | null = bunFit.stopReason;
const bunFitBestLoss: number | null = bunFit.bestLoss;
const bunModuleFitDataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const bunModuleFitLoader = data.dataLoader(bunModuleFitDataset, { batchSize: 2, shuffle: false });
type BunModuleFitBatch = ReturnType<typeof bunModuleFitLoader.__getitem__>;
const bunModuleFitCriterion: TrainSupervisedCriterion<typeof bunModel, BunModuleFitBatch> = new nn.MSELoss();
const bunModuleFitEvidence: TrainFitEvidence<"sgd"> = train.fitModule(bunOptimizer, bunModel, bunModuleFitLoader, bunModuleFitCriterion, {
  epochs: 1,
  zeroGrad: true,
});
const bunModuleFitEvidenceSnake: TrainFitEvidence<"sgd"> = train.fit_module(bunOptimizer, bunModel, bunModuleFitLoader, bunModuleFitCriterion, {
  max_steps: 1,
  zero_grad: true,
});
const bunEvaluation: TrainEvaluateEvidence = train.evaluate(bunLoader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("evaluation batch requires a target tensor");
  return loss.mse(bunModel.forward(batch.input), batch.target);
}, {
  maxSteps: 1,
  onStep(step: TrainEvaluateStepEvidence) {
    const lossValue: number = step.loss;
    const sampleIndices: readonly number[] | null = step.sampleIndices;
    void lossValue;
    void sampleIndices;
  },
});
const bunEvaluationAlias: TrainEvaluateEvidence = train.evaluate_loss(bunDatasetSplitLoader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("evaluation batch requires a target tensor");
  return loss.mse(bunModel.forward(batch.input), batch.target);
}, { max_steps: 1 });
const bunModuleEvaluation: TrainEvaluateEvidence = train.evaluateModule(bunModel, bunModuleFitLoader, bunModuleFitCriterion, { maxSteps: 1 });
const bunModuleEvaluationSnake: TrainEvaluateEvidence = train.evaluate_module(bunModel, bunModuleFitLoader, bunModuleFitCriterion, { max_steps: 1 });
const bunPrediction: TrainPredictEvidence<Tensor> = train.predict(bunLoader, (batch: TensorDatasetBatch) => bunModel.forward(batch.input), {
  maxSteps: 1,
  onStep(step: TrainPredictStepEvidence<Tensor>) {
    const sampleIndices: readonly number[] | null = step.sample_indices;
    const output: Tensor = step.output;
    const requiredStep: TrainPredictStepEvidence<Tensor> = train.requireTrainPredictStepEvidence(step);
    const stepSignatureMatches: boolean = train.matchesTrainPredictStepEvidenceSignature(step, step.signature);
    void sampleIndices;
    void output;
    void requiredStep;
    void stepSignatureMatches;
  },
});
const bunPredictionValid: boolean = train.isTrainPredictEvidence(bunPrediction);
const bunRequiredPrediction: TrainPredictEvidence<Tensor> = train.requireTrainPredictEvidence(bunPrediction);
const bunPredictionSignatureMatches: boolean = train.matchesTrainPredictEvidenceSignature(bunPrediction, bunPrediction.signature);
const bunPredictionAlias: TrainPredictEvidence<Tensor> = train.predict_batches(bunDatasetSplitLoader, (batch: TensorDatasetBatch) => bunModel.forward(batch.input), { max_steps: 1 });
const bunModulePrediction: TrainPredictEvidence<Tensor> = train.predictModule(bunModel, bunModuleFitLoader, { maxSteps: 1 });
const bunModulePredictionSnake: TrainPredictEvidence<Tensor> = train.predict_module(bunModel, bunModuleFitLoader, { max_steps: 1 });
const bunClassifierModulePrediction: TrainPredictEvidence<Tensor> = train.predictClassifier(bunModel, bunModuleFitLoader, { maxSteps: 1 });
const bunClassifierModulePredictionSnake: TrainPredictEvidence<Tensor> = train.predict_classifier(bunModel, bunModuleFitLoader, { max_steps: 1 });

const bunModuleState: ModuleStateSnapshot = bunModel.stateDict("bun");
const bunOptimizerState: OptimizerStateSnapshot<"sgd"> = bunOptimizer.stateDict();
const bunSchedulerState: LRSchedulerStateSnapshot<"step-lr", "sgd"> = bunScheduler.stateDict();
const bunSnapshot = checkpoint.create({ model: bunModel, optimizer: bunOptimizer, scheduler: bunScheduler, prefix: "bun", metadata: { run: "bun", epochs: bunFit.epochs } });
const bunInspection: CheckpointInspection = checkpoint.inspect(bunSnapshot);
const bunSchedulerInspectionKind: "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau" | null = bunInspection.schedulerKind;
const bunCosineCheckpoint: ZgmlCheckpoint = checkpoint.create({ scheduler: bunCosineScheduler });
const bunCosineInspection: CheckpointInspection = checkpoint.inspect(bunCosineCheckpoint);
const bunCosineInspectionTMax: number | null = bunCosineInspection.schedulerT_max;
const bunCosineInspectionEtaMin: number | null = bunCosineInspection.schedulerEtaMin;
const bunJsonSnapshot: ZgmlCheckpoint = checkpoint.toJSON(bunSnapshot);
const bunLoadedSnapshot: ZgmlCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(bunJsonSnapshot)));
const bunSerializedSnapshot: string = checkpoint.stringify(bunSnapshot, 2);
const bunParsedSnapshot: ZgmlCheckpoint = checkpoint.parse(bunSerializedSnapshot);
const bunSerializedSnapshotAlias: string = checkpoint.serialize(bunSnapshot, 2);
const bunParsedSnapshotAlias: ZgmlCheckpoint = checkpoint.deserialize(bunSerializedSnapshotAlias);
const bunLoadedInspection: CheckpointInspection = checkpoint.inspect(bunLoadedSnapshot);

bunModel.loadStateDict(bunModuleState, { prefix: "bun", strict: true, validateOnly: true });
bunOptimizer.loadStateDict(bunOptimizerState, { strict: true, validateOnly: true });
bunScheduler.loadStateDict(bunSchedulerState, { strict: true, validateOnly: true });
checkpoint.load(bunLoadedSnapshot, { model: bunModel, optimizer: bunOptimizer, scheduler: bunScheduler, prefix: "bun", strict: true });

const bunRestoredModel = nn.sequential([
  nn.linear(2, 3),
  nn.gelu(),
  nn.linear(3, 1),
]);
const bunRestoredOptimizer: Optimizer<"sgd"> = optim.sgd(bunRestoredModel, { lr: 0.04, momentum: 0.8 });
const bunRestoredScheduler: LRScheduler<"step-lr", "sgd"> = optim.stepLR(bunRestoredOptimizer, { stepSize: 2, gamma: 0.5 });
checkpoint.restore(bunLoadedSnapshot, {
  model: bunRestoredModel,
  optimizer: bunRestoredOptimizer,
  scheduler: bunRestoredScheduler,
  prefix: "bun",
  strict: true,
});
const bunRestoredProgram: Program<readonly [2], readonly [1]> = bunRestoredModel.compile({ backend: "cpu", inputShape: [2] as const });
const bunRestoredSession: Session<readonly [2], readonly [1]> = bunRestoredProgram.bindModule(bunRestoredModel);
const bunRestoredCompiledPrediction: Tensor<readonly [1]> = bunRestoredSession.stepTensor(tensor([1, 0], [2] as const));

const bunCompileSupport: ModuleCompileSupport = bunModel.compileSupport({ inputShape: [2, 2] as const });
const bunCanCompile: boolean = bunCompileSupport.supported;
const bunSample = tensor([1, 0], [2] as const);
const bunRegularized = nn.sequential([
  nn.linear(2, 2),
  nn.dropout(0.25),
  nn.relu(),
]);
const bunRegularizedParameterNames: readonly string[] = bunRegularized.parameterNames("bunRegularized");
const bunRegularizedParameterInfos: readonly ModuleParameterInfo[] = bunRegularized.parameterInfos("bunRegularized");
const bunRegularizedNamedModules: readonly ModuleTraversalEntry[] = bunRegularized.namedModules("bunRegularized");
const bunRegularizedNamespaceNamedModules: readonly ModuleTraversalEntry[] = nn.namedModules(bunRegularized, "bunRegularized");
const bunRegularizedTrain: typeof bunRegularized = bunRegularized.train();
const bunRegularizedEval: typeof bunRegularized = bunRegularized.eval();
const bunRegularizedNamespaceTrain: typeof bunRegularized = nn.train(bunRegularized, true);
const bunRegularizedNamespaceEval: typeof bunRegularized = nn.eval(bunRegularized);
const bunRegularizedCpu: typeof bunRegularized = bunRegularized.cpu();
const bunRegularizedToCpu: typeof bunRegularized = bunRegularized.to("cpu");
const bunRegularizedFloat: typeof bunRegularized = bunRegularized.float();
const bunRegularizedFloat32: typeof bunRegularized = bunRegularized.float32();
const bunRegularizedNamespaceCpu: typeof bunRegularized = nn.cpu(bunRegularized);
const bunRegularizedNamespaceToCpu: typeof bunRegularized = nn.to(bunRegularized, { device: "cpu", copy: false });
const bunRegularizedNamespaceFloat: typeof bunRegularized = nn.float(bunRegularized);
const bunRegularizedNamespaceFloat32: typeof bunRegularized = nn.float32(bunRegularized, { dtype: "float32", copy: false });
bunRegularizedEval.eval();
const bunRegularizedEvalPrediction: Tensor<readonly [2]> = bunRegularizedEval.forward(bunSample);
const bunRegularizedEvalProgram: Program<readonly [2], readonly [2]> = bunRegularizedEval.compile({ backend: "cpu", inputShape: [2] as const });
const bunRegularizedEvalSession: Session<readonly [2], readonly [2]> = bunRegularizedEvalProgram.bindModule(bunRegularizedEval);
const bunRegularizedEvalCompiledPrediction: Tensor<readonly [2]> = bunRegularizedEvalSession.stepTensor(bunSample);
const bunGradModeWasEnabled: boolean = is_grad_enabled();
const bunPreviousGradMode: boolean = set_grad_enabled(true);
const bunRootGradModeNamespace: GradModeNamespace = bunGradMode;
const bunRootGradModeEnabled: boolean = bunGradMode.isGradEnabled();
const bunRootGradModeNoGradValue: number = bunGradMode.noGrad(() => 4);
const bunGradTrackedSample: Tensor<readonly [2]> = bunSample.clone().requires_grad_();
const bunGradTrackedSampleAlias: Tensor<readonly [2]> = bunGradTrackedSample.requiresGrad_(true);
const bunClonedSample: Tensor<readonly [2]> = clone(bunGradTrackedSampleAlias);
const bunDetachedSample: Tensor<readonly [2]> = detach(bunGradTrackedSampleAlias);
const bunDetachedMethodSample: Tensor<readonly [2]> = bunGradTrackedSampleAlias.detach();
const bunDetachedInPlaceSample: Tensor<readonly [2]> = bunGradTrackedSampleAlias.detach_();
const bunNoGradPrediction: Tensor<readonly [1]> = no_grad(() => bunModel.forward(bunSample));
const bunInferenceModePrediction: Tensor<readonly [1]> = inference_mode(() => bunModel.forward(bunSample));
const bunEnableGradPrediction: Tensor<readonly [1]> = enable_grad(() => bunModel.forward(bunSample));
set_grad_enabled(bunPreviousGradMode);
const bunManualOptimizer: Optimizer<"sgd"> = optim.sgd(bunModel, { lr: 0.01, momentum: 0.1 });
const bunManualLoss: Tensor<readonly [1]> = loss.mse(bunModel.forward(bunSample), tensor([1], [1] as const));
bunManualOptimizer.zeroGrad();
bunManualLoss.backward();
bunManualOptimizer.step();
bunManualOptimizer.zero_grad();
const bunManualStepEvidence: TrainStepEvidence<"sgd"> = train.step(bunManualOptimizer, {
  loss: loss.mse(bunModel.forward(bunSample), tensor([1], [1] as const)),
  evidence: true,
  clipGradNorm: 1,
});
const bunManualGradNorm: number = train.gradNorm(bunModel);
const bunManualStepValid: boolean = train.isTrainStepEvidence(bunManualStepEvidence);
const bunHelperLoss: Tensor<readonly [1]> = loss.mse(bunModel.forward(bunSample), tensor([1], [1] as const));
train.backward(bunHelperLoss);
const bunHelperClippedNorm: number = train.clipGradNorm(bunModel, 1, { eps: 1e-6 });
const bunHelperClippedModel: typeof bunModel = train.clipGradValue(bunModel, 0.5);
const bunHelperLossStep: Tensor = train.lossStep(bunManualOptimizer, () => loss.mse(bunModel.forward(bunSample), tensor([1], [1] as const)), {
  zeroGrad: true,
  zero_grad_options: { set_to_none: true },
  clipGradNorm: 1,
  clipGradValue: 0.5,
});
const bunHelperLossStepSnake: Tensor = train.lossStep(bunManualOptimizer, () => loss.mse(bunModel.forward(bunSample), tensor([1], [1] as const)), {
  zero_grad: true,
  zero_grad_options: { set_to_none: true },
});
const bunManualFrozenParam = bunModel.namedParameters().find((param) => param.name.endsWith("bias"));
if (bunManualFrozenParam) bunManualFrozenParam.requiresGrad = false;
const bunGroupedParams: NnParameter[] = bunModel.namedParameters("bunGrouped");
const bunGroupedOptimizer: Optimizer<"sgd"> = optim.sgd([
  { params: [bunGroupedParams[0]], lr: 0.02, weightDecay: 0.001 },
  { params: [bunGroupedParams[1]], lr: 0.005, weight_decay: 0 },
] satisfies readonly OptimizerParamGroupInput[], { lr: 0.01, momentum: 0.2 });
const bunGroupedConfig: OptimizerConfigSnapshot<"sgd"> = bunGroupedOptimizer.config();
const bunGroupedDefaults: OptimizerConfigSnapshot<"sgd"> = bunGroupedOptimizer.defaults;
const bunGroupedParamGroup: OptimizerParamGroupSnapshot = bunGroupedConfig.paramGroups[0];
const bunGroupedParamGroupsConfigAlias: readonly OptimizerParamGroupSnapshot[] = bunGroupedConfig.param_groups;
const bunGroupedParamGroups: readonly OptimizerParamGroupSnapshot[] = bunGroupedOptimizer.paramGroups;
const bunGroupedParamGroupsAlias: readonly OptimizerParamGroupSnapshot[] = bunGroupedOptimizer.param_groups;
bunGroupedOptimizer.addParamGroup({ params: [bunGroupedParams[2]], lr: 0.001 });
bunGroupedOptimizer.add_param_group({ params: [bunGroupedParams[3]], lr: 0.0005 });
const bunInitializedGroupedWeight: NnParameter = nn.init.xavier_uniform_(bunGroupedParams[0]);
const bunNormalInitializedGroupedWeight: NnParameter = nn.init.xavier_normal_(bunGroupedParams[0]);
const bunKaimingInitializedGroupedWeight: NnParameter = nn.init.kaiming_uniform_(bunGroupedParams[0], { nonlinearity: "relu" });
const bunKaimingNormalInitializedGroupedWeight: NnParameter = nn.init.kaiming_normal_(bunGroupedParams[0], { nonlinearity: "relu" });
const bunInitializedGroupedBias: NnParameter = nn.init.zeros_(bunGroupedParams[1]);
const bunProgram: Program<readonly [2], readonly [1]> = bunModel.compile({ backend: "cpu", inputShape: [2] as const });
const bunSession: Session<readonly [2], readonly [1]> = bunProgram.bindModule(bunModel);
const bunCompiledPrediction: Tensor<readonly [1]> = bunSession.stepTensor(bunSample);
const bunForwardPrediction: Tensor<readonly [1]> = bunModel.forward(bunSample);
const bunCallPrediction: Tensor<readonly [1]> = bunModel.call(bunSample);
const bunDunderPrediction: Tensor<readonly [1]> = bunModel.__call__(bunSample);
const bunNamespaceCallPrediction: Tensor<readonly [1]> = nn.call(bunModel, bunSample);
const bunNamespaceDunderPrediction: Tensor<readonly [1]> = nn.__call__(bunModel, bunSample);
const bunCriterion: MSELoss = loss.mseLoss();
const bunCriterionForward: Tensor<readonly [1]> = bunCriterion.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunCriterionCall: Tensor<readonly [1]> = bunCriterion.call(bunCallPrediction, tensor([1], [1] as const));
const bunCriterionDunderCall: Tensor<readonly [1]> = bunCriterion.__call__(bunDunderPrediction, tensor([1], [1] as const));
const bunL1Criterion: L1Loss = loss.l1Loss();
const bunL1CriterionAlias: L1Loss = loss.l1_loss();
const bunL1CriterionClass: L1Loss = new loss.L1Loss();
const bunNnL1CriterionClass: L1Loss = new nn.L1Loss();
const bunL1Objective: Tensor<readonly [1]> = loss.l1(bunForwardPrediction, tensor([1], [1] as const));
const bunL1Forward: Tensor<readonly [1]> = bunL1Criterion.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunL1Call: Tensor<readonly [1]> = bunL1CriterionAlias.call(bunCallPrediction, tensor([1], [1] as const));
const bunL1DunderCall: Tensor<readonly [1]> = bunL1CriterionClass.__call__(bunDunderPrediction, tensor([1], [1] as const));
const bunNnL1Forward: Tensor<readonly [1]> = bunNnL1CriterionClass.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunHuberCriterion: HuberLoss = loss.huberLoss({ delta: 1 });
const bunHuberCriterionAlias: HuberLoss = loss.huber_loss({ delta: 1 });
const bunHuberCriterionClass: HuberLoss = new loss.HuberLoss({ delta: 1 });
const bunNnHuberCriterionClass: HuberLoss = new nn.HuberLoss({ delta: 1 });
const bunHuberObjective: Tensor<readonly [1]> = loss.huber(bunForwardPrediction, tensor([1], [1] as const), { delta: 1 });
const bunHuberForward: Tensor<readonly [1]> = bunHuberCriterion.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunHuberCall: Tensor<readonly [1]> = bunHuberCriterionAlias.call(bunCallPrediction, tensor([1], [1] as const));
const bunHuberDunderCall: Tensor<readonly [1]> = bunHuberCriterionClass.__call__(bunDunderPrediction, tensor([1], [1] as const));
const bunNnHuberForward: Tensor<readonly [1]> = bunNnHuberCriterionClass.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunSmoothL1Criterion: SmoothL1Loss = loss.smoothL1Loss({ beta: 1 });
const bunSmoothL1CriterionAlias: SmoothL1Loss = loss.smooth_l1_loss({ beta: 1 });
const bunSmoothL1CriterionClass: SmoothL1Loss = new loss.SmoothL1Loss({ beta: 1 });
const bunNnSmoothL1CriterionClass: SmoothL1Loss = new nn.SmoothL1Loss({ beta: 1 });
const bunSmoothL1Objective: Tensor<readonly [1]> = loss.smooth_l1(bunForwardPrediction, tensor([1], [1] as const), { beta: 1 });
const bunSmoothL1Forward: Tensor<readonly [1]> = bunSmoothL1Criterion.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunSmoothL1Call: Tensor<readonly [1]> = bunSmoothL1CriterionAlias.call(bunCallPrediction, tensor([1], [1] as const));
const bunSmoothL1DunderCall: Tensor<readonly [1]> = bunSmoothL1CriterionClass.__call__(bunDunderPrediction, tensor([1], [1] as const));
const bunNnSmoothL1Forward: Tensor<readonly [1]> = bunNnSmoothL1CriterionClass.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunBceCriterion: BCELoss = loss.bceLoss();
const bunBceCriterionAlias: BCELoss = loss.bce_loss({ eps: 1e-7 });
const bunBceCriterionClass: BCELoss = new loss.BCELoss({ eps: 1e-7 });
const bunNnBceCriterionClass: BCELoss = new nn.BCELoss({ eps: 1e-7 });
const bunBinaryPrediction: Tensor<readonly [1]> = bunForwardPrediction.sigmoid();
const bunBceObjective: Tensor<readonly [1]> = loss.bce(bunBinaryPrediction, tensor([1], [1] as const));
const bunBinaryCrossEntropyObjective: Tensor<readonly [1]> = loss.binary_cross_entropy(bunBinaryPrediction, tensor([1], [1] as const));
const bunBinaryAccuracy: number = train.binaryAccuracy(bunBinaryPrediction, tensor([1], [1] as const));
const bunBinaryAccuracyAlias: number = train.binary_accuracy([0.1, 0.9] as const, [0, 1] as const);
const bunBceForward: Tensor<readonly [1]> = bunBceCriterion.forward(bunBinaryPrediction, tensor([1], [1] as const));
const bunBceCall: Tensor<readonly [1]> = bunBceCriterionAlias.call(bunBinaryPrediction, tensor([1], [1] as const));
const bunBceDunderCall: Tensor<readonly [1]> = bunBceCriterionClass.__call__(bunBinaryPrediction, tensor([1], [1] as const));
const bunNnBceForward: Tensor<readonly [1]> = bunNnBceCriterionClass.forward(bunBinaryPrediction, tensor([1], [1] as const));
const bunBceWithLogitsCriterion: BCEWithLogitsLoss = loss.bceWithLogitsLoss();
const bunBceWithLogitsCriterionAlias: BCEWithLogitsLoss = loss.bce_with_logits_loss();
const bunBceWithLogitsCriterionClass: BCEWithLogitsLoss = new loss.BCEWithLogitsLoss();
const bunNnBceWithLogitsCriterionClass: BCEWithLogitsLoss = new nn.BCEWithLogitsLoss();
const bunBceWithLogitsObjective: Tensor<readonly [1]> = loss.bceWithLogits(bunForwardPrediction, tensor([1], [1] as const));
const bunBinaryCrossEntropyWithLogitsObjective: Tensor<readonly [1]> = loss.binary_cross_entropy_with_logits(bunForwardPrediction, tensor([1], [1] as const));
const bunBinaryLogitsAccuracy: number = train.binaryLogitsAccuracy(bunForwardPrediction, tensor([1], [1] as const));
const bunBinaryLogitsAccuracyAlias: number = train.binary_logits_accuracy([-1, 2] as const, [0, 1] as const);
const bunBceWithLogitsForward: Tensor<readonly [1]> = bunBceWithLogitsCriterion.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunBceWithLogitsCall: Tensor<readonly [1]> = bunBceWithLogitsCriterionAlias.call(bunForwardPrediction, tensor([1], [1] as const));
const bunBceWithLogitsDunderCall: Tensor<readonly [1]> = bunBceWithLogitsCriterionClass.__call__(bunForwardPrediction, tensor([1], [1] as const));
const bunNnBceWithLogitsForward: Tensor<readonly [1]> = bunNnBceWithLogitsCriterionClass.forward(bunForwardPrediction, tensor([1], [1] as const));
const bunClassTargets: Uint32Array = loss.classTargets([1]);
const bunClassLogits = tensor([0.2, 1.4], [1, 2] as const);
const bunClassObjective: Tensor<readonly [1]> = loss.crossEntropy(bunClassLogits, bunClassTargets, { classes: 2 });
const bunClassObjectiveSnake: Tensor<readonly [1]> = loss.cross_entropy(bunClassLogits, bunClassTargets, { numClasses: 2 });
function createBunTypedClassifierGraph() {
  return nn.sequential([
    nn.linear(2, 4),
    nn.relu(),
    nn.linear(4, 2),
  ]);
}
class BunTypedClassifier extends nn.Module<readonly [2], readonly [2]> {
  readonly graph: ReturnType<typeof createBunTypedClassifierGraph>;

  constructor() {
    super({ kind: "bun-typed-classifier" });
    this.graph = createBunTypedClassifierGraph();
  }

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]>;
  forward(input: Tensor): Tensor;
  forward(input: Tensor): Tensor {
    return this.graph.forward(input);
  }
}
const bunTypedClassifier = new BunTypedClassifier();
const bunTypedClassifierTarget: Uint32Array = loss.classTargets([1]);
const bunTypedClassifierCriterion: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 2 });
const bunTypedClassifierObjective: Tensor<readonly [1]> = bunTypedClassifierCriterion.forward(bunTypedClassifier.forward(bunSample), bunTypedClassifierTarget);
const bunTypedClassifierDataset = data.tensorDataset(
  tensor([-1, -1, 1, -1], [2, 2] as const),
  tensor([0, 1], [2] as const),
);
const bunTypedClassifierLoader = data.dataLoader(bunTypedClassifierDataset, { batch_size: 1, shuffle: false });
const bunTypedClassifierOptimizer: Optimizer<"adamw"> = optim.adamW(bunTypedClassifier, { lr: 0.01 });
const bunTypedClassifierFit: TrainFitEvidence<"adamw"> = train.fit(bunTypedClassifierOptimizer, bunTypedClassifierLoader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("typed classifier batch requires targets");
  return bunTypedClassifierCriterion.forward(bunTypedClassifier.graph.forward(batch.input), batch.target);
}, { epochs: 1, zero_grad: true });
type BunTypedClassifierBatch = ReturnType<typeof bunTypedClassifierLoader.__getitem__>;
const bunTypedClassifierFitCriterion: TrainClassificationCriterion<typeof bunTypedClassifier, BunTypedClassifierBatch> = bunTypedClassifierCriterion;
const bunTypedClassifierFitModule: TrainFitEvidence<"adamw"> = train.fitClassifier(
  bunTypedClassifierOptimizer,
  bunTypedClassifier,
  bunTypedClassifierLoader,
  bunTypedClassifierFitCriterion,
  { epochs: 1, zero_grad: true },
);
const bunTypedClassifierFitModuleSnake: TrainFitEvidence<"adamw"> = train.fit_classifier(
  bunTypedClassifierOptimizer,
  bunTypedClassifier,
  bunTypedClassifierLoader,
  bunTypedClassifierFitCriterion,
  { max_steps: 1, zero_grad: true },
);
const bunTypedClassifierPrediction: TrainPredictEvidence<Tensor<readonly [number, 2]>> = train.predictClassifier(bunTypedClassifier, bunTypedClassifierLoader, { maxSteps: 1 });
const bunTypedClassifierPredictionSnake: TrainPredictEvidence<Tensor<readonly [number, 2]>> = train.predict_classifier(bunTypedClassifier, bunTypedClassifierLoader, { max_steps: 1 });
// @ts-expect-error fitModule rejects classifiers trained with class-index targets.
train.fitModule(bunTypedClassifierOptimizer, bunTypedClassifier, bunTypedClassifierLoader, bunModuleFitCriterion);
// @ts-expect-error fitClassifier rejects regression-style targets that still include a class axis.
train.fitClassifier(bunTypedClassifierOptimizer, bunTypedClassifier, bunModuleFitLoader, bunTypedClassifierFitCriterion);
const bunTypedClassifierState: ModuleStateSnapshot = bunTypedClassifier.stateDict("typedClassifier");
const bunTypedClassifierCheckpoint: ZgmlCheckpoint = checkpoint.create({ model: bunTypedClassifier, optimizer: bunTypedClassifierOptimizer, prefix: "typedClassifier" });
const bunTypedClassifierInspection: CheckpointInspection = checkpoint.inspect(bunTypedClassifierCheckpoint);
const bunTypedClassifierProgram: Program<readonly [2], readonly [2]> = bunTypedClassifier.compile({ backend: "cpu", inputShape: [2] as const });
const bunTypedClassifierSession: Session<readonly [2], readonly [2]> = bunTypedClassifierProgram.bindModule(bunTypedClassifier);
const bunTypedClassifierCompiledLogits: Tensor<readonly [2]> = bunTypedClassifierSession.stepTensor(bunSample);
const bunF = nn.functional;
const bunFunctionalMseObjective: Tensor<readonly [1]> = bunF.mse(bunForwardPrediction, tensor([1], [1] as const));
const bunFunctionalMseLossObjective: Tensor<readonly [1]> = bunF.mse_loss(bunForwardPrediction, tensor([1], [1] as const));
const bunFunctionalMseLossSumObjective: Tensor<readonly [1]> = bunF.mse_loss(bunForwardPrediction, tensor([1], [1] as const), { reduction: "sum" });
const bunFunctionalL1Objective: Tensor<readonly [1]> = bunF.l1(bunForwardPrediction, tensor([1], [1] as const));
const bunFunctionalL1LossObjective: Tensor<readonly [1]> = bunF.l1_loss(bunForwardPrediction, tensor([1], [1] as const));
const bunFunctionalL1LossSumObjective: Tensor<readonly [1]> = bunF.l1_loss(bunForwardPrediction, tensor([1], [1] as const), { reduction: "sum" });
const bunFunctionalHuberObjective: Tensor<readonly [1]> = bunF.huber(bunForwardPrediction, tensor([1], [1] as const), { delta: 1 });
const bunFunctionalHuberLossObjective: Tensor<readonly [1]> = bunF.huber_loss(bunForwardPrediction, tensor([1], [1] as const), { delta: 1 });
const bunFunctionalSmoothL1Objective: Tensor<readonly [1]> = bunF.smooth_l1(bunForwardPrediction, tensor([1], [1] as const), { beta: 1 });
const bunFunctionalSmoothL1LossObjective: Tensor<readonly [1]> = bunF.smooth_l1_loss(bunForwardPrediction, tensor([1], [1] as const), { beta: 1 });
const bunFunctionalBceObjective: Tensor<readonly [1]> = bunF.binary_cross_entropy(bunBinaryPrediction, tensor([1], [1] as const));
const bunFunctionalBceWithLogitsObjective: Tensor<readonly [1]> = bunF.binary_cross_entropy_with_logits(bunForwardPrediction, tensor([1], [1] as const));
const bunFunctionalActivation: Tensor<readonly [1, 2]> = bunF.relu(bunClassLogits);
const bunFunctionalActivationInplaceFalse: Tensor<readonly [1, 2]> = bunF.relu(bunClassLogits, false);
const bunFunctionalGelu: Tensor<readonly [1, 2]> = bunF.gelu(bunClassLogits);
const bunFunctionalSilu: Tensor<readonly [1, 2]> = bunF.silu(bunClassLogits);
const bunFunctionalSigmoid: Tensor<readonly [1, 2]> = bunF.sigmoid(bunClassLogits);
const bunFunctionalTanh: Tensor<readonly [1, 2]> = bunF.tanh(bunClassLogits);
const bunFunctionalSoftmax: Tensor<readonly [1, 2]> = bunF.softmax(bunClassLogits, 1);
const bunFunctionalSoftmaxDim: Tensor<readonly [1, 2]> = bunF.softmax_dim(bunClassLogits, 1);
const bunFunctionalLogSoftmax: Tensor<readonly [1, 2]> = bunF.log_softmax(bunClassLogits, 1);
const bunFunctionalLogSoftmaxDim: Tensor<readonly [1, 2]> = bunF.log_softmax_dim(bunClassLogits, 1);
const bunFunctionalDropout: Tensor<readonly [1, 2]> = bunF.dropout(bunClassLogits, 0.25, { training: false });
const bunFunctionalDropoutBoolean: Tensor<readonly [1, 2]> = bunF.dropout(bunClassLogits, 0.25, false);
const bunFunctionalDropoutInplaceFalse: Tensor<readonly [1, 2]> = bunF.dropout(bunClassLogits, 0.25, false, false);
const bunFunctionalFlatten: Tensor<readonly [2]> = bunF.flatten(bunClassLogits);
const bunFunctionalFlattenRange: Tensor<readonly [1, 2]> = bunF.flatten(bunClassLogits, 1, -1);
const bunFunctionalLinear1d: Tensor<readonly [1]> = bunF.linear(bunSample, tensor([[0.5, -0.5]], [1, 2] as const), tensor([0], [1] as const));
const bunFunctionalLinear2d: Tensor<readonly [1, 2]> = bunF.linear(bunClassLogits, tensor([[1, 0], [0, 1]], [2, 2] as const), tensor([0, 0], [2] as const));
const bunFunctionalLinear3d: Tensor<readonly [1, 2, 2]> = bunF.linear(tensor([1, 2, 3, 4], [1, 2, 2] as const), tensor([[1, 0], [0, 1]], [2, 2] as const), tensor([0, 0], [2] as const));
const bunFunctionalNormalize: Tensor<readonly [1, 2]> = bunF.normalize(bunClassLogits, 2, 1);
const bunFunctionalOneHot: Tensor<readonly [1, 2]> = bunF.one_hot(tensor([1], [1] as const), 2);
const bunFunctionalOneHotAlias: Tensor<readonly [1, 2]> = bunF.oneHot(tensor([1], [1] as const), 2);
const bunFunctionalOneHotGrid: Tensor<readonly [1, 2, 3]> = bunF.one_hot(tensor([0, 2], [1, 2] as const), 3);
const bunFunctionalEmbedding: Tensor<readonly [2, 2]> = bunF.embedding(tensor([0, 1], [2] as const), tensor([[1, 0], [0, 1], [1, 1]], [3, 2] as const));
const bunFunctionalEmbeddingGrid: Tensor<readonly [1, 2, 2]> = bunF.embedding(tensor([0, 1], [1, 2] as const), tensor([[1, 0], [0, 1], [1, 1]], [3, 2] as const));
const bunFunctionalLayerNorm: Tensor<readonly [1, 2]> = bunF.layer_norm(bunClassLogits, [2] as const);
const bunFunctionalLayerNormPositional: Tensor<readonly [1, 2]> = bunF.layer_norm(bunClassLogits, [2] as const, tensor([1, 1], [2] as const), tensor([0, 0], [2] as const), 1e-5);
const bunFunctionalRmsNorm: Tensor<readonly [1, 2]> = bunF.rmsNorm(bunClassLogits, 2);
const bunFunctionalRmsNormPositional: Tensor<readonly [1, 2]> = bunF.rms_norm(bunClassLogits, 2, tensor([1, 1], [2] as const), 1e-5);
const bunBatchNorm = new nn.BatchNorm1d(2, { momentum: 0.5, runningMean: [0, 0], runningVar: [1, 1] });
const bunBatchNormForward: Tensor<readonly [2, 2]> = bunBatchNorm.forward(tensor([1, 3, 3, 7], [2, 2] as const));
const bunBatchNormState: ModuleStateSnapshot = bunBatchNorm.stateDict("bn");
const bunBatchNormRunningMean: NnBuffer | null = bunBatchNorm.runningMean;
const bunConv2d: Conv2dModule<1, 1> = new nn.Conv2d(1, 1, 2, { weight: [1, 0, 0, 1], bias: false });
const bunConv2dFactory: ReturnType<typeof nn.conv2d> = nn.conv2d(1, 1, [2, 2] as const, { stride: 1, padding: 0 });
const bunConv2dForward: Tensor<readonly [1, number, number]> = bunConv2d.forward(tensor([1, 2, 3, 4], [1, 2, 2] as const));
const bunConv2dState: ModuleStateSnapshot = bunConv2d.stateDict("conv");
const bunConv2dSupport: ModuleCompileSupport = bunConv2d.compileSupport({ inputShape: [1, 2, 2] as const });
const bunFunctionalBatchNorm: Tensor<readonly [2, 2]> = bunF.batchNorm1d(tensor([1, 3, 3, 7], [2, 2] as const), 2, { eps: 1e-5 });
const bunFunctionalBatchNormSnake: Tensor<readonly [2, 2]> = bunF.batch_norm1d(tensor([1, 3, 3, 7], [2, 2] as const), 2, { eps: 1e-5, momentum: 0.5 });
const bunFunctionalClassObjective: Tensor<readonly [1]> = bunF.cross_entropy(bunClassLogits, bunClassTargets, { numClasses: 2 });
const bunFunctionalClassSumObjective: Tensor<readonly [1]> = bunF.cross_entropy(bunClassLogits, bunClassTargets, { numClasses: 2, reduction: "sum" });
const bunFunctionalAliasClassObjective: Tensor<readonly [1]> = nn.F.crossEntropy(bunClassLogits, bunClassTargets, { classes: 2 });
const bunClassAccuracy: number = train.accuracy(bunClassLogits, bunClassTargets, { classes: 2 });
const bunClassAccuracyAlias: number = train.classification_accuracy([0.2, 1.4] as const, [1], { numClasses: 2 });
const bunClassPredictions: Uint32Array = train.classPredictions(bunClassLogits, { classes: 2 });
const bunClassPredictionsAlias: Uint32Array = train.class_predictions([0.2, 1.4] as const, { numClasses: 2 });
const bunPredictedClasses: Uint32Array = train.predictClasses(bunClassLogits, { classes: 2 });
const bunPredictedClassesAlias: Uint32Array = train.predict_classes([0.2, 1.4] as const, { numClasses: 2 });
const bunClassTopKAccuracy: number = train.topKAccuracy(bunClassLogits, bunClassTargets, { classes: 2, k: 1 });
const bunClassTopKAccuracyAlias: number = train.top_k_accuracy([0.2, 1.4] as const, [1], { numClasses: 2, top_k: 1 });
const bunClassConfusionMatrix: readonly (readonly number[])[] = train.confusionMatrix(bunClassLogits, bunClassTargets, { classes: 2 });
const bunClassConfusionMatrixAlias: readonly (readonly number[])[] = train.confusion_matrix([0.2, 1.4] as const, [1], { numClasses: 2 });
const bunClassReport: TrainClassificationReport = train.classificationReport(bunClassLogits, bunClassTargets, { classes: 2 });
const bunClassReportAlias: TrainClassificationReport = train.classification_report([0.2, 1.4] as const, [1], { numClasses: 2 });
const bunCrossEntropyCriterion: CrossEntropyLoss = loss.crossEntropyLoss({ classes: 2 });
const bunCrossEntropyCriterionAlias: CrossEntropyLoss = loss.cross_entropy_loss({ numClasses: 2 });
const bunCrossEntropyCriterionClass: CrossEntropyLoss = new loss.CrossEntropyLoss({ classes: 2 });
const bunNnCrossEntropyCriterionClass: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 2 });
const bunCrossEntropyForward: Tensor<readonly [1]> = bunCrossEntropyCriterion.forward(bunClassLogits, bunClassTargets);
const bunCrossEntropyCall: Tensor<readonly [1]> = bunCrossEntropyCriterionAlias.call(bunClassLogits, bunClassTargets);
const bunCrossEntropyDunderCall: Tensor<readonly [1]> = bunCrossEntropyCriterionClass.__call__(bunClassLogits, bunClassTargets);
const bunNnCrossEntropyForward: Tensor<readonly [1]> = bunNnCrossEntropyCriterionClass.forward(bunClassLogits, bunClassTargets);
const bunClassLogProbabilities: Tensor<readonly [1, 2]> = bunClassLogits.logSoftmax(1);
const bunFunctionalNllObjective: Tensor<readonly [1]> = bunF.nll_loss(bunClassLogProbabilities, bunClassTargets, { classes: 2 });
const bunFunctionalNllSumObjective: Tensor<readonly [1]> = bunF.nll_loss(bunClassLogProbabilities, bunClassTargets, { classes: 2, reduction: "sum" });
const bunNegativeLogLikelihoodObjective: Tensor<readonly [1]> = loss.negative_log_likelihood(bunClassLogProbabilities, bunClassTargets, { classes: 2 });
const bunNllObjective: Tensor<readonly [1]> = loss.nllLoss(bunClassLogProbabilities, bunClassTargets, { classes: 2 });
const bunNllObjectiveAlias: Tensor<readonly [1]> = loss.nll_loss(bunClassLogProbabilities, bunClassTargets, { numClasses: 2 });
const bunNllCriterion: NLLLoss = loss.nllLossModule({ classes: 2 });
const bunNllCriterionAlias: NLLLoss = loss.nll_loss_module({ numClasses: 2 });
const bunNllCriterionClass: NLLLoss = new loss.NLLLoss({ classes: 2 });
const bunNnNllCriterionClass: NLLLoss = new nn.NLLLoss({ classes: 2 });
const bunNllForward: Tensor<readonly [1]> = bunNllCriterion.forward(bunClassLogProbabilities, bunClassTargets);
const bunNllCall: Tensor<readonly [1]> = bunNllCriterionAlias.call(bunClassLogProbabilities, bunClassTargets);
const bunNllDunderCall: Tensor<readonly [1]> = bunNllCriterionClass.__call__(bunClassLogProbabilities, bunClassTargets);
const bunNnNllForward: Tensor<readonly [1]> = bunNnNllCriterionClass.forward(bunClassLogProbabilities, bunClassTargets);

void bunFit;
void bunClassStyleOptimizer;
void bunVariadicClassStyleModel;
void bunVariadicClassStylePrediction;
void bunRmspropClassOptimizer;
void bunRmspropConfig;
void bunRmspropState;
void bunAdagradClassOptimizer;
void bunAdagradConfig;
void bunAdagradState;
void bunClassStyleObjective;
void bunRootFunctionalSoftmax;
void bunNnClassStyleCriterion;
void bunNnClassStyleObjective;
void bunInspection;
void bunLoadedInspection;
void bunSerializedSnapshot;
void bunParsedSnapshot;
void bunSerializedSnapshotAlias;
void bunParsedSnapshotAlias;
void bunSchedulerInspectionKind;
void bunFitStopReason;
void bunFitBestLoss;
void bunCosineCheckpoint;
void bunCosineInspection;
void bunCosineInspectionTMax;
void bunCosineInspectionEtaMin;
void bunSchedulerState;
void bunExponentialScheduler;
void bunCosineScheduler;
void bunPlateauScheduler;
void bunSchedulerNamespace;
void bunExponentialSchedulerNamespace;
void bunCosineSchedulerNamespace;
void bunPlateauSchedulerNamespace;
void bunDatasetLen;
void bunDatasetSampleAt;
void bunDatasetSampleDunder;
void bunDatasetIterableInput;
void bunDatasetDunderIterInput;
void bunClassDataset;
void bunClassLoader;
void bunDatasetSplitPair;
void bunDatasetSplitPairAlias;
void bunDatasetIterableSplit;
void bunDatasetSubset;
void bunDatasetIterableSubset;
void bunDatasetTake;
void bunDatasetConcat;
void bunDatasetIterableConcat;
void bunDatasetConcatAlias;
void bunDatasetMapped;
void bunDatasetMappedAlias;
void bunDatasetSplitLoader;
void bunLoaderBatchSizeAlias;
void bunLoaderDropLastAlias;
void bunRestoredCompiledPrediction;
void bunCanCompile;
void bunRegularizedParameterNames;
void bunRegularizedParameterInfos;
void bunRegularizedNamedModules;
void bunRegularizedNamespaceNamedModules;
void bunRegularizedTrain;
void bunRegularizedNamespaceTrain;
void bunRegularizedNamespaceEval;
void bunRegularizedEvalPrediction;
void bunRegularizedEvalCompiledPrediction;
void bunGradModeWasEnabled;
void bunRootGradModeNamespace;
void bunRootGradModeEnabled;
void bunRootGradModeNoGradValue;
void bunClonedSample;
void bunDetachedSample;
void bunDetachedMethodSample;
void bunDetachedInPlaceSample;
void bunNoGradPrediction;
void bunInferenceModePrediction;
void bunEnableGradPrediction;
void bunManualStepEvidence;
void bunManualGradNorm;
void bunManualStepValid;
void bunHelperClippedNorm;
void bunHelperClippedModel;
void bunHelperLossStep;
void bunHelperLossStepSnake;
void bunEvaluation;
void bunEvaluationAlias;
void bunPrediction;
void bunPredictionAlias;
void bunManualFrozenParam;
void bunGroupedConfig;
void bunGroupedParamGroup;
void bunGroupedParamGroups;
void bunGroupedParamGroupsAlias;
void bunInitializedGroupedWeight;
void bunNormalInitializedGroupedWeight;
void bunKaimingInitializedGroupedWeight;
void bunKaimingNormalInitializedGroupedWeight;
void bunInitializedGroupedBias;
void bunRegularizedCpu;
void bunRegularizedToCpu;
void bunRegularizedFloat;
void bunRegularizedFloat32;
void bunRegularizedNamespaceCpu;
void bunRegularizedNamespaceToCpu;
void bunRegularizedNamespaceFloat;
void bunRegularizedNamespaceFloat32;
void bunCompiledPrediction;
void bunForwardPrediction;
void bunCallPrediction;
void bunDunderPrediction;
void bunNamespaceCallPrediction;
void bunNamespaceDunderPrediction;
void bunCriterionForward;
void bunCriterionCall;
void bunCriterionDunderCall;
void bunClassStyleSumReduction;
void bunClassStyleSumObjective;
void bunNnClassStyleSumReduction;
void bunNnClassStyleSumObjective;
void bunFunctionalMseObjective;
void bunFunctionalMseLossObjective;
void bunFunctionalMseLossSumObjective;
void bunFunctionalL1Objective;
void bunFunctionalL1LossObjective;
void bunFunctionalL1LossSumObjective;
void bunFunctionalHuberObjective;
void bunFunctionalHuberLossObjective;
void bunFunctionalSmoothL1Objective;
void bunFunctionalSmoothL1LossObjective;
void bunFunctionalBceObjective;
void bunFunctionalBceWithLogitsObjective;
void bunL1Objective;
void bunL1Forward;
void bunL1Call;
void bunL1DunderCall;
void bunNnL1Forward;
void bunHuberObjective;
void bunHuberForward;
void bunHuberCall;
void bunHuberDunderCall;
void bunNnHuberForward;
void bunSmoothL1Objective;
void bunSmoothL1Forward;
void bunSmoothL1Call;
void bunSmoothL1DunderCall;
void bunNnSmoothL1Forward;
void bunBinaryPrediction;
void bunBceObjective;
void bunBinaryCrossEntropyObjective;
void bunBinaryAccuracy;
void bunBinaryAccuracyAlias;
void bunBceForward;
void bunBceCall;
void bunBceDunderCall;
void bunNnBceForward;
void bunBceWithLogitsObjective;
void bunBinaryCrossEntropyWithLogitsObjective;
void bunBinaryLogitsAccuracy;
void bunBinaryLogitsAccuracyAlias;
void bunBceWithLogitsForward;
void bunBceWithLogitsCall;
void bunBceWithLogitsDunderCall;
void bunNnBceWithLogitsForward;
void bunClassObjective;
void bunClassAccuracy;
void bunClassAccuracyAlias;
void bunClassPredictions;
void bunClassPredictionsAlias;
void bunPredictedClasses;
void bunPredictedClassesAlias;
void bunClassTopKAccuracy;
void bunClassTopKAccuracyAlias;
void bunClassConfusionMatrix;
void bunClassConfusionMatrixAlias;
void bunClassReport;
void bunClassReportAlias;
void bunCrossEntropyForward;
void bunCrossEntropyCall;
void bunCrossEntropyDunderCall;
void bunNnCrossEntropyCriterionClass;
void bunNnCrossEntropyForward;
void bunClassObjectiveSnake;
void bunTypedClassifierObjective;
void bunTypedClassifierFit;
void bunTypedClassifierFitModule;
void bunTypedClassifierFitModuleSnake;
void bunTypedClassifierPrediction;
void bunTypedClassifierPredictionSnake;
void bunTypedClassifierState;
void bunTypedClassifierInspection;
void bunTypedClassifierCompiledLogits;
void bunFunctionalActivation;
void bunFunctionalActivationInplaceFalse;
void bunFunctionalGelu;
void bunFunctionalSilu;
void bunFunctionalSigmoid;
void bunFunctionalTanh;
void bunFunctionalSoftmax;
void bunFunctionalSoftmaxDim;
void bunFunctionalLogSoftmax;
void bunFunctionalLogSoftmaxDim;
void bunFunctionalDropout;
void bunFunctionalDropoutBoolean;
void bunFunctionalDropoutInplaceFalse;
void bunFunctionalFlatten;
void bunFunctionalFlattenRange;
void bunFunctionalLinear1d;
void bunFunctionalLinear2d;
void bunFunctionalLinear3d;
void bunFunctionalNormalize;
void bunFunctionalOneHot;
void bunFunctionalOneHotAlias;
void bunFunctionalOneHotGrid;
void bunFunctionalEmbedding;
void bunFunctionalEmbeddingGrid;
void bunFunctionalLayerNorm;
void bunFunctionalLayerNormPositional;
void bunFunctionalRmsNorm;
void bunFunctionalRmsNormPositional;
void bunBatchNormForward;
void bunBatchNormState;
void bunBatchNormRunningMean;
void bunConv2dFactory;
void bunConv2dForward;
void bunConv2dState;
void bunConv2dSupport;
void bunFunctionalBatchNorm;
void bunFunctionalBatchNormSnake;
void bunFunctionalClassObjective;
void bunFunctionalClassSumObjective;
void bunFunctionalAliasClassObjective;
void bunFunctionalNllObjective;
void bunFunctionalNllSumObjective;
void bunNegativeLogLikelihoodObjective;
void bunNllObjective;
void bunNllObjectiveAlias;
void bunNllForward;
void bunNllCall;
void bunNllDunderCall;
void bunNnNllCriterionClass;
void bunNnNllForward;
void bunModuleFitEvidence;
void bunModuleFitEvidenceSnake;
