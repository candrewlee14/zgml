import {
  F,
  checkpoint,
  compile,
  data,
  gradMode,
  load,
  nn,
  optim,
  save,
  tensor,
  torch,
  train,
  zgml,
  type CompiledInference,
  type Program,
  type SessionExecutionPlan,
  type SessionStepParamsCompatibility,
  type Session,
  type Tensor,
} from "zgml";
import {
  F as bunF,
  checkpoint as bunCheckpoint,
  data as bunData,
  gradMode as bunGradMode,
  nn as bunNn,
  optim as bunOptim,
  tensor as bunTensor,
  train as bunTrain,
  type SessionExecutionPlan as BunSessionExecutionPlan,
  type SessionStepParamsCompatibility as BunSessionStepParamsCompatibility,
  type Tensor as BunTensor,
} from "zgml/bun";

const zgmlModel = new zgml.nn.Sequential(
  new zgml.nn.Linear(2, 4),
  new zgml.nn.ReLU(),
  new zgml.nn.Linear(4, 1),
);
const zgmlOptimizer = new zgml.optim.AdamW(zgmlModel, { lr: 3e-2, weight_decay: 1e-3 });
const zgmlScheduler = new zgml.optim.lr_scheduler.StepLR(zgmlOptimizer, { step_size: 20, gamma: 0.5 });
const zgmlDataset = new zgml.utils.data.TensorDataset(
  zgml.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  zgml.tensor([0, 1, 1, 0], [4, 1] as const),
);
const zgmlLoader = new zgml.utils.data.DataLoader(zgmlDataset, { batch_size: 2, shuffle: true });
const zgmlCriterion = new zgml.nn.MSELoss();
const zgmlFit = zgml.train.fit(zgmlModel, zgmlLoader, {
  optimizer: zgmlOptimizer,
  loss: zgmlCriterion,
  epochs: 1,
  zero_grad: true,
  clip_grad_norm: 1,
  onStep() {
    zgmlScheduler.step();
  },
});
const zgmlSnapshot = zgml.checkpoint.create({ model: zgmlModel, optimizer: zgmlOptimizer, scheduler: zgmlScheduler, prefix: "zgml" });
const zgmlFast = zgml.native(zgmlModel, { inputShape: [2] as const });
const zgmlFastAlias = zgml.compileInference(zgmlModel, { inputShape: [2] as const });
const zgmlFastTyped: CompiledInference<readonly [2], readonly [1]> = zgmlFast;
const zgmlFastProgram: Program<readonly [2], readonly [1]> = zgmlFast.program;
const zgmlFastSession: Session<readonly [2], readonly [1]> = zgmlFast.session;
const zgmlFastProof = zgmlFast.explain();
const zgmlFastSupport = zgmlFast.compileSupport();
const zgmlFastForward: Tensor<readonly [1]> = zgmlFast.forward(zgml.tensor([1, 0], [2] as const));
const zgmlFastOut = zgmlFast.into(new Float32Array(1), zgml.tensor([1, 0], [2] as const));
zgmlFast.dispose();
zgmlFastAlias.dispose();
void zgmlFit;
void zgmlSnapshot;
void zgmlFastProof;
void zgmlFastSupport;
void zgmlFastTyped;
void zgmlFastProgram;
void zgmlFastSession;
void zgmlFastForward;
void zgmlFastOut;

const torchModel = new torch.nn.Sequential(
  new torch.nn.Linear(2, 4),
  new torch.nn.ReLU(),
  new torch.nn.Linear(4, 1),
);

const torchOptimizer = new torch.optim.AdamW(torchModel, { lr: 3e-2, weight_decay: 1e-3 });
const torchScheduler = new torch.optim.lr_scheduler.StepLR(torchOptimizer, { step_size: 20, gamma: 0.5 });
const torchRegressionCriterion = new torch.nn.MSELoss();
const torchDataset = new torch.utils.data.TensorDataset(
  torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  torch.tensor([0, 1, 1, 0], [4, 1] as const),
);
const torchLoader = new torch.utils.data.DataLoader(torchDataset, { batch_size: 2, shuffle: true });

const torchFit = torch.train.fit(torchModel, torchLoader, {
  optimizer: torchOptimizer,
  loss: torchRegressionCriterion,
  epochs: 8,
  zero_grad: true,
  clip_grad_norm: 1,
  onStep() {
    torchScheduler.step();
  },
});

const torchSnapshot = torch.checkpoint.create({ model: torchModel, optimizer: torchOptimizer, scheduler: torchScheduler, prefix: "xor" });
const torchText = torch.checkpoint.stringify(torchSnapshot, 2);
const torchLoaded = torch.checkpoint.parse(torchText);
torch.checkpoint.restore(torchLoaded, { model: torchModel, optimizer: torchOptimizer, scheduler: torchScheduler, prefix: "xor", strict: true });
torch.checkpoint.restore(torchSnapshot, { model: torchModel, optimizer: torchOptimizer, scheduler: torchScheduler, prefix: "xor", strict: true });
const torchReadmeProgram = torch.compile(torchModel, { inputShape: [2] as const });

class TorchClassifier extends torch.nn.Module<readonly [2], readonly [2]> {
  readonly graph = new torch.nn.Sequential(
    new torch.nn.Linear(2, 4),
    new torch.nn.ReLU(),
    new torch.nn.Linear(4, 2),
  );

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]>;
  forward(input: Tensor): Tensor;
  forward(input: Tensor): Tensor {
    return this.graph.forward(input);
  }
}

const torchClassifier = new TorchClassifier();
const torchCriterion = new torch.nn.CrossEntropyLoss({ classes: 2 });
const torchClassifierOptimizer = new torch.optim.AdamW(torchClassifier, { lr: 1e-2 });
const torchClassifierDataset = new torch.utils.data.TensorDataset(
  torch.tensor([-1, -1, 1, -1], [2, 2] as const),
  torch.tensor([0, 1], [2] as const),
);
const torchClassifierLoader = new torch.utils.data.DataLoader(torchClassifierDataset, { batch_size: 1, shuffle: false });

const torchClassifierFit = torch.train.fitClassifier(torchClassifierOptimizer, torchClassifier, torchClassifierLoader, torchCriterion, {
  epochs: 4,
  zero_grad: true,
});

torchClassifier.eval();
const torchClassifierEvaluation = torch.inference_mode(() => torch.train.evaluate(torchClassifierLoader, (batch) => {
  if (!batch.target) throw new Error("torch classifier eval batch requires targets");
  return torchCriterion.forward(torchClassifier.forward(batch.input), batch.target);
}));
const torchClassifierPrediction = torch.no_grad(() => torch.train.predictClassifier(torchClassifier, torchClassifierLoader));

const torchProgram = torch.compile.compile(torchClassifier, { inputShape: [2] as const });
const torchSession = torchProgram.bindModule(torchClassifier);
const torchInferenceInput = torch.tensor([1, -1], [2] as const);
const torchLogits: Tensor<readonly [2]> = torch.inference_mode(() => torchSession.stepTensor(torchInferenceInput));
const torchLogitsOut = new Float32Array(2);
const torchHotParams = { input: torchInferenceInput, output: torchLogitsOut };
const torchHotCompatibility: SessionStepParamsCompatibility = torchSession.requireHotStepParams(torchHotParams);
const torchHotPlan: SessionExecutionPlan<readonly [2], readonly [2]> = torchSession.hotPathPlan(torchHotParams);
const torchLogitsInto: Float32Array = torch.inference_mode(() => torchSession.executeInto(torchLogitsOut, { input: torchInferenceInput }));
const torchProbabilities: Tensor<readonly [2]> = torch.F.softmax(torchLogits, -1);
const torchPredictedClasses: Uint32Array = torch.train.predictClasses(torchLogits, { classes: 2 });

const model = nn.sequential([
  nn.linear(2, 4),
  nn.relu(),
  nn.linear(4, 1),
]);

const optimizer = optim.adamW(model, { lr: 3e-2, weightDecay: 1e-3 });
const scheduler = optim.stepLR(optimizer, { stepSize: 20, gamma: 0.5 });
const regressionCriterion = new nn.MSELoss();
const dataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const loader = data.dataLoader(dataset, { batchSize: 2, shuffle: true });

const fit = train.fit(model, loader, {
  optimizer,
  loss: regressionCriterion,
  epochs: 8,
  zeroGrad: true,
  clipGradNorm: 1,
  onStep() {
    scheduler.step();
  },
});

const snapshot = checkpoint.create({ model, optimizer, scheduler, prefix: "xor" });
const text = save(snapshot, 2);
const loaded = load(text);
load(text, { model, optimizer, scheduler, prefix: "xor", strict: true });
checkpoint.restore(snapshot, { model, optimizer, scheduler, prefix: "xor", strict: true });
const readmeFast = compile.compileForInference(model, { inputShape: [2] as const });
const readmeFastProof = readmeFast.explain();
const readmeFastSupport = readmeFast.compileSupport();
const readmeFastOut = readmeFast.into(new Float32Array(1), tensor([1, 0], [2] as const));
readmeFast.dispose();

class Classifier extends nn.Module<readonly [2], readonly [2]> {
  readonly graph = nn.sequential([
    nn.linear(2, 4),
    nn.relu(),
    nn.linear(4, 2),
  ]);

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]>;
  forward(input: Tensor): Tensor;
  forward(input: Tensor): Tensor {
    return this.graph.forward(input);
  }
}

const classifier = new Classifier();
const criterion = new nn.CrossEntropyLoss({ classes: 2 });
const classifierOptimizer = optim.adamW(classifier, { lr: 1e-2 });
const classifierDataset = data.tensorDataset(
  tensor([-1, -1, 1, -1], [2, 2] as const),
  tensor([0, 1], [2] as const),
);
const classifierLoader = data.dataLoader(classifierDataset, { batchSize: 1, shuffle: false });

const classifierFit = train.fitClassifier(classifierOptimizer, classifier, classifierLoader, criterion, {
  epochs: 4,
  zero_grad: true,
});

classifier.eval();
const classifierEvaluation = gradMode.inferenceMode(() => train.evaluate(classifierLoader, (batch) => {
  if (!batch.target) throw new Error("classifier eval batch requires targets");
  return criterion.forward(classifier.forward(batch.input), batch.target);
}, { max_steps: 2 }));
const classifierPrediction = gradMode.noGrad(() => train.predictClassifier(classifier, classifierLoader, { max_steps: 2 }));
const classifierPredictionSnake = gradMode.noGrad(() => train.predict_classifier(classifier, classifierLoader, { maxSteps: 2 }));

const program = classifier.compile({ inputShape: [2] as const });
const session = program.bindModule(classifier);
const inferenceInput = tensor([1, -1], [2] as const);
const logits: Tensor<readonly [2]> = gradMode.inferenceMode(() => session.stepTensor(inferenceInput));
const logitsOut = new Float32Array(2);
const hotParams = { input: inferenceInput, output: logitsOut };
const hotCompatibility: SessionStepParamsCompatibility = session.requireHotStepParams(hotParams);
const hotPlan: SessionExecutionPlan<readonly [2], readonly [2]> = session.hotPathPlan(hotParams);
const logitsInto: Float32Array = gradMode.inferenceMode(() => session.executeInto(logitsOut, { input: inferenceInput }));
const probabilities: Tensor<readonly [2]> = F.softmax(logits, -1);
const predictedClasses: Uint32Array = train.predictClasses(logits, { classes: 2 });
const predictedClassesInto: Uint32Array = train.predict_classes(logitsInto, { numClasses: 2 });

class BunClassifier extends bunNn.Module<readonly [2], readonly [2]> {
  readonly graph = bunNn.sequential([
    bunNn.linear(2, 4),
    bunNn.relu(),
    bunNn.linear(4, 2),
  ]);

  forward(input: BunTensor<readonly [2]>): BunTensor<readonly [2]>;
  forward(input: BunTensor): BunTensor;
  forward(input: BunTensor): BunTensor {
    return this.graph.forward(input);
  }
}

const bunClassifier = new BunClassifier();
const bunCriterion = new bunNn.CrossEntropyLoss({ classes: 2 });
const bunClassifierOptimizer = bunOptim.adamW(bunClassifier, { lr: 1e-2 });
const bunClassifierDataset = bunData.tensorDataset(
  bunTensor([-1, -1, 1, -1], [2, 2] as const),
  bunTensor([0, 1], [2] as const),
);
const bunClassifierLoader = bunData.dataLoader(bunClassifierDataset, { batchSize: 1, shuffle: false });
const bunClassifierFit = bunTrain.fitClassifier(bunClassifierOptimizer, bunClassifier, bunClassifierLoader, bunCriterion, {
  epochs: 4,
  zero_grad: true,
});
const bunClassifierEvaluation = bunGradMode.inferenceMode(() => bunTrain.evaluate(bunClassifierLoader, (batch) => {
  if (!batch.target) throw new Error("bun classifier eval batch requires targets");
  return bunCriterion.forward(bunClassifier.forward(batch.input), batch.target);
}, { max_steps: 2 }));
const bunClassifierPrediction = bunGradMode.noGrad(() => bunTrain.predictClassifier(bunClassifier, bunClassifierLoader, { max_steps: 2 }));
const bunClassifierPredictionSnake = bunGradMode.noGrad(() => bunTrain.predict_classifier(bunClassifier, bunClassifierLoader, { maxSteps: 2 }));
const bunSnapshot = bunCheckpoint.create({ model: bunClassifier, optimizer: bunClassifierOptimizer, prefix: "bunClassifier" });
const bunProgram = bunClassifier.compile({ inputShape: [2] as const });
const bunSession = bunProgram.bindModule(bunClassifier);
const bunInferenceInput = bunTensor([1, -1], [2] as const);
const bunLogits: BunTensor<readonly [2]> = bunGradMode.inferenceMode(() => bunSession.stepTensor(bunInferenceInput));
const bunLogitsOut = new Float32Array(2);
const bunHotParams = { input: bunInferenceInput, output: bunLogitsOut };
const bunHotCompatibility: BunSessionStepParamsCompatibility = bunSession.requireHotStepParams(bunHotParams);
const bunHotPlan: BunSessionExecutionPlan<readonly [2], readonly [2]> = bunSession.hotPathPlan(bunHotParams);
const bunLogitsInto: Float32Array = bunGradMode.inferenceMode(() => bunSession.executeInto(bunLogitsOut, { input: bunInferenceInput }));
const bunProbabilities: BunTensor<readonly [2]> = bunF.softmax(bunLogits, -1);
const bunPredictedClasses: Uint32Array = bunTrain.predictClasses(bunLogits, { classes: 2 });
const bunPredictedClassesInto: Uint32Array = bunTrain.predict_classes(bunLogitsInto, { numClasses: 2 });

void fit;
void torchFit;
void torchText;
void torchLoaded;
void torchReadmeProgram;
void torchClassifierFit;
void torchClassifierEvaluation;
void torchClassifierPrediction;
void torchLogits;
void torchHotCompatibility;
void torchHotPlan;
void torchLogitsInto;
void torchProbabilities;
void torchPredictedClasses;
void text;
void loaded;
void readmeFastProof;
void readmeFastSupport;
void readmeFastOut;
void classifierFit;
void classifierEvaluation;
void classifierPrediction;
void classifierPredictionSnake;
void logits;
void hotCompatibility;
void hotPlan;
void logitsInto;
void probabilities;
void predictedClasses;
void predictedClassesInto;
void bunClassifierFit;
void bunClassifierEvaluation;
void bunClassifierPrediction;
void bunClassifierPredictionSnake;
void bunSnapshot;
void bunLogits;
void bunHotCompatibility;
void bunHotPlan;
void bunLogitsInto;
void bunProbabilities;
void bunPredictedClasses;
void bunPredictedClassesInto;
