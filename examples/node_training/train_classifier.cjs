"use strict";

const {
  checkpoint,
  compile,
  data,
  gradMode,
  loss,
  nn,
  optim,
  tensor,
  train,
} = require("../..");

function scalar(t) {
  return typeof t.item === "function" ? t.item() : t.data[0];
}

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

function assertHotRuntimeProfile(target, label) {
  const expectation = target.requireHotRuntimeProfile();
  const profile = target.runtimeProfile();
  if (
    !expectation.noFallback ||
    !expectation.noSync ||
    !expectation.noRuntimePatchInvalid ||
    profile.fallbackOpCount !== 0 ||
    profile.syncCount !== 0 ||
    profile.runtimePatchInvalidCount !== 0
  ) {
    throw new Error(`${label} runtime profile is not hot-path clean: ${JSON.stringify({ expectation, profile })}`);
  }
}

function createClassifierGraph() {
  return new nn.Sequential([
    nn.linear(2, 4, {
      weights: [
        0.3, -0.2, 0.1, 0.2,
        -0.1, 0.25, 0.2, -0.3,
      ],
      bias: [0, 0, 0, 0],
    }),
    nn.relu(),
    nn.linear(4, 2, {
      weights: [
        0.2, -0.1,
        0.05, 0.15,
        -0.15, 0.2,
        0.1, -0.05,
      ],
      bias: [0, 0],
    }),
  ]);
}

class TinyClassifier extends nn.Module {
  constructor() {
    super({ kind: "tiny-classifier" });
    this.graph = createClassifierGraph();
  }

  forward(input) {
    return this.graph.forward(input);
  }
}

function createClassifier() {
  return new TinyClassifier();
}

const model = createClassifier();
const moduleNames = model.namedModules("classifier").map((entry) => entry.name).join("|");
const parameterNames = model.parameterNames("classifier").join("|");
if (
  moduleNames !== "classifier|classifier.graph|classifier.graph.0|classifier.graph.1|classifier.graph.2" ||
  parameterNames !== "classifier.graph.0.weight|classifier.graph.0.bias|classifier.graph.2.weight|classifier.graph.2.bias"
) {
  throw new Error(`unexpected custom classifier module tree: modules=${moduleNames} parameters=${parameterNames}`);
}
const samples = data.tensorDataset(
  tensor([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
    0.5, -0.5,
    -0.5, 0.5,
  ], [6, 2]),
  tensor([0, 1, 1, 0, 1, 1], [6]),
);

const loader = data.dataLoader(samples, { batchSize: 2, shuffle: true, seed: 17 });
const optimizer = optim.adamW(model, { lr: 0.05, weightDecay: 0.0001 });
const probeInput = tensor([1, -1], [2]);
const probeTarget = loss.classTargets([1]);
const before = scalar(loss.crossEntropy(model.forward(probeInput), probeTarget, { classes: 2 }));

const criterion = loss.crossEntropyLoss({ classes: 2 });
const fit = train.fit(optimizer, loader, (batch) => {
  if (!batch.target) throw new Error("classifier training batch requires targets");
  return criterion.call(model.forward(batch.input), batch.target);
}, {
  epochs: 80,
  zeroGrad: true,
});

const eagerLogits = model.forward(probeInput);
const after = scalar(loss.crossEntropy(eagerLogits, probeTarget, { classes: 2 }));
const eagerClass = train.predictClasses(eagerLogits, { classes: 2 })[0];
if (!train.isTrainFitEvidence(fit) || fit.steps !== 240 || fit.losses.length !== 240) {
  throw new Error("train.fit must return AdamW fit evidence for each classifier optimizer step");
}
if (!(after < before * 0.01) || eagerClass !== 1) {
  throw new Error(`expected classifier training to learn class 1; before=${before}, after=${after}, logits=${Array.from(eagerLogits.data)}`);
}

const highLevelModel = createClassifier();
const highLevelOptimizer = optim.adamW(highLevelModel, { lr: 0.05, weightDecay: 0.0001 });
const highLevelLoader = data.dataLoader(samples, { batchSize: 2, shuffle: true, seed: 17 });
const highLevelBefore = scalar(loss.crossEntropy(highLevelModel.forward(probeInput), probeTarget, { classes: 2 }));
const highLevelFit = train.fitClassifier(highLevelOptimizer, highLevelModel, highLevelLoader, criterion, {
  epochs: 80,
  zeroGrad: true,
});
const highLevelLogits = highLevelModel.forward(probeInput);
const highLevelAfter = scalar(loss.crossEntropy(highLevelLogits, probeTarget, { classes: 2 }));
const highLevelClass = train.classPredictions(highLevelLogits, { classes: 2 })[0];
if (!train.isTrainFitEvidence(highLevelFit) || highLevelFit.steps !== 240 || highLevelFit.losses.length !== 240) {
  throw new Error("train.fitClassifier must return AdamW fit evidence for each classifier optimizer step");
}
if (!(highLevelAfter < highLevelBefore * 0.01) || highLevelClass !== 1) {
  throw new Error(`expected train.fitClassifier to learn class 1; before=${highLevelBefore}, after=${highLevelAfter}, logits=${Array.from(highLevelLogits.data)}`);
}

const ergonomicNativeModel = createClassifierGraph();
const ergonomicNativeOptimizer = optim.adamW(ergonomicNativeModel, { lr: 0.05, weightDecay: 0.0001 });
const ergonomicNativeLoader = data.dataLoader(samples, { batchSize: 2, shuffle: true, seed: 17 });
const ergonomicNativeBefore = scalar(loss.crossEntropy(ergonomicNativeModel.forward(probeInput), probeTarget, { classes: 2 }));
const ergonomicNativeFit = ergonomicNativeModel.fit(ergonomicNativeLoader, {
  optimizer: ergonomicNativeOptimizer,
  loss: criterion,
  epochs: 80,
  requireNative: true,
});
const ergonomicNativePlan = ergonomicNativeFit.compiledPlan;
const ergonomicNativeLogits = ergonomicNativeModel.forward(probeInput);
const ergonomicNativeAfter = scalar(loss.crossEntropy(ergonomicNativeLogits, probeTarget, { classes: 2 }));
const ergonomicNativeClass = train.predictClasses(ergonomicNativeLogits, { classes: 2 })[0];
if (
  !train.isTrainFitEvidence(ergonomicNativeFit) ||
  ergonomicNativeFit.native !== true ||
  ergonomicNativeFit.backend !== "cpu" ||
  ergonomicNativePlan?.loweredBy !== "zig-ffi" ||
  ergonomicNativePlan?.runtimePath !== "JS/TS module API -> Zig native training kernel" ||
  ergonomicNativePlan?.kernels[0] !== "zgml_train_mlp_relu_cross_entropy_adamw_f32" ||
  ergonomicNativeFit.steps !== 240 ||
  ergonomicNativeFit.losses.length !== 1 ||
  ergonomicNativeFit.nativeBulk !== true ||
  ergonomicNativeFit.bulkResult?.kernel !== "zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk"
) {
  throw new Error(`ergonomic nn.Module.fit native path must prove Zig-backed training: ${JSON.stringify(ergonomicNativeFit)}`);
}
if (!(ergonomicNativeAfter < ergonomicNativeBefore * 0.01) || ergonomicNativeClass !== 1) {
  throw new Error(`expected ergonomic native trainer to learn class 1; before=${ergonomicNativeBefore}, after=${ergonomicNativeAfter}, logits=${Array.from(ergonomicNativeLogits.data)}`);
}

const nativeModel = createClassifierGraph();
const nativeOptimizer = optim.adamW(nativeModel, { lr: 0.05, weightDecay: 0.0001 });
const nativeLoader = data.dataLoader(samples, { batchSize: 2, shuffle: true, seed: 17 });
const nativeBefore = scalar(loss.crossEntropy(nativeModel.forward(probeInput), probeTarget, { classes: 2 }));
const nativeTrainer = compile.compileForTraining(nativeModel, nativeOptimizer, {
  inputShape: [2, 2],
  classes: 2,
  loss: "crossEntropy",
});
const nativePlan = nativeTrainer.plan();
if (
  nativePlan.loweredBy !== "zig-ffi" ||
  nativePlan.kernels[0] !== "zgml_train_mlp_relu_cross_entropy_adamw_f32" ||
  nativePlan.inputShape.join("x") !== "2x2" ||
  nativePlan.outputShape.join("x") !== "2x2" ||
  nativeTrainer.compileEvidence() !== nativePlan
) {
  throw new Error(`compiled native classifier trainer must expose its Zig plan: ${JSON.stringify(nativePlan)}`);
}
const nativeFit = train.fit(nativeTrainer, nativeLoader, {
  epochs: 80,
});
const nativeLogits = nativeModel.forward(probeInput);
const nativeAfter = scalar(loss.crossEntropy(nativeLogits, probeTarget, { classes: 2 }));
const nativeClass = train.predictClasses(nativeLogits, { classes: 2 })[0];
if (
  !train.isTrainFitEvidence(nativeFit) ||
  nativeFit.native !== true ||
  nativeFit.backend !== "cpu" ||
  nativeFit.compiledPlan !== nativePlan ||
  nativeFit.compiled_plan !== nativePlan ||
  nativeFit.steps !== 240 ||
  nativeFit.losses.length !== 1 ||
  nativeFit.nativeBulk !== true ||
  nativeFit.bulkResult?.kernel !== "zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk"
) {
  throw new Error(`train.fit compiled native trainer must return native fit evidence: ${JSON.stringify(nativeFit)}`);
}
if (!(nativeAfter < nativeBefore * 0.01) || nativeClass !== 1) {
  throw new Error(`expected compiled native trainer to learn class 1; before=${nativeBefore}, after=${nativeAfter}, logits=${Array.from(nativeLogits.data)}`);
}
nativeTrainer.free();

model.eval();
const evalLoader = data.dataLoader(samples, { batchSize: 2, shuffle: false });
if (!gradMode.isGradEnabled()) {
  throw new Error("grad mode should be enabled before inference helpers enter inferenceMode");
}
const evaluation = gradMode.inferenceMode(() => train.evaluate(evalLoader, (batch) => {
  if (!batch.target) throw new Error("classifier eval batch requires targets");
  return criterion.call(model.forward(batch.input), batch.target);
}));
const prediction = gradMode.inferenceMode(() => train.predict(data.dataLoader(samples, { batchSize: 2, shuffle: false }), (batch) => model.forward(batch.input)));
const inferenceLogits = gradMode.noGrad(() => model.forward(probeInput));
if (!gradMode.isGradEnabled()) {
  throw new Error("inferenceMode/noGrad must restore the previous grad mode");
}
if (!train.isTrainEvaluateEvidence(evaluation) || evaluation.steps !== 3 || !(evaluation.meanLoss < before)) {
  throw new Error(`expected classifier eval evidence after training, got ${JSON.stringify(evaluation)}`);
}
if (!train.isTrainPredictEvidence(prediction) || prediction.steps !== 3 || prediction.outputs.length !== 3) {
  throw new Error(`expected classifier prediction evidence after training, got steps=${prediction.steps} outputs=${prediction.outputs.length}`);
}
assertClose(inferenceLogits.data[0], eagerLogits.data[0], 1e-5, "noGrad classifier logit 0");
assertClose(inferenceLogits.data[1], eagerLogits.data[1], 1e-5, "noGrad classifier logit 1");

const snapshot = checkpoint.create({ model, optimizer, prefix: "classifier" });
const inspection = checkpoint.inspect(snapshot);
if (!inspection.hasModel || !inspection.hasOptimizer || inspection.modelParameterCount !== 4) {
  throw new Error(`unexpected classifier checkpoint inspection: ${JSON.stringify(inspection)}`);
}

const restored = createClassifier();
const restoredOptimizer = optim.adamW(restored, { lr: 0.05, weightDecay: 0.0001 });
checkpoint.restore(snapshot, { model: restored, optimizer: restoredOptimizer, prefix: "classifier", strict: true });
const restoredLogits = restored.forward(probeInput);
assertClose(restoredLogits.data[0], eagerLogits.data[0], 1e-5, "restored classifier logit 0");
assertClose(restoredLogits.data[1], eagerLogits.data[1], 1e-5, "restored classifier logit 1");

const support = model.compileSupport({ inputShape: [2] });
if (!support || support.supported !== true || support.outputShape?.join("x") !== "2") {
  throw new Error(`expected custom classifier graph compile support evidence, got ${JSON.stringify(support)}`);
}
const program = model.compile({ backend: "cpu", inputShape: [2] });
assertHotRuntimeProfile(program, "trained classifier Program");
const session = program.bindModule(model);
const compiledLogits = gradMode.inferenceMode(() => session.stepTensor(probeInput));
const compiledLogitsOut = new Float32Array(2);
const hotParams = { input: probeInput, output: compiledLogitsOut };
const hotCompatibility = session.requireHotStepParams(hotParams);
const hotPlan = session.hotPathPlan(hotParams);
const compiledLogitsInto = gradMode.inferenceMode(() => session.executeInto(compiledLogitsOut, { input: probeInput }));
const compiledClass = train.predictClasses(compiledLogits, { classes: 2 })[0];
const compiledIntoClass = train.predict_classes(compiledLogitsInto, { numClasses: 2 })[0];
assertHotRuntimeProfile(session, "trained classifier Session");
if (
  compiledLogitsInto !== compiledLogitsOut ||
  compiledClass !== 1 ||
  compiledIntoClass !== 1 ||
  hotCompatibility.hotPath !== true ||
  hotCompatibility.runtimeOutputAllocationFree !== true ||
  hotCompatibility.readbackFree !== true ||
  hotPlan.hotPath !== true ||
  hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature
) {
  throw new Error(`expected trained classifier executeInto to prove allocation-free hot path: ${JSON.stringify({ hotCompatibility, hotPlan })}`);
}
assertClose(compiledLogits.data[0], eagerLogits.data[0], 1e-5, "compiled classifier logit 0");
assertClose(compiledLogits.data[1], eagerLogits.data[1], 1e-5, "compiled classifier logit 1");
assertClose(compiledLogitsInto[0], eagerLogits.data[0], 1e-5, "executeInto classifier logit 0");
assertClose(compiledLogitsInto[1], eagerLogits.data[1], 1e-5, "executeInto classifier logit 1");
if (program.inputShape().join("x") !== "2" || program.outputShape().join("x") !== "2") {
  throw new Error(`unexpected classifier Program shape: input=${program.inputShape()} output=${program.outputShape()}`);
}
session.free();
program.free();

const restoredProgram = restored.compile({ backend: "cpu", inputShape: [2] });
assertHotRuntimeProfile(restoredProgram, "restored classifier Program");
const restoredSession = restoredProgram.bindModule(restored);
const restoredCompiledLogits = restoredSession.stepTensor(probeInput);
const restoredCompiledLogitsOut = new Float32Array(2);
const restoredHotParams = { input: probeInput, output: restoredCompiledLogitsOut };
const restoredHotCompatibility = restoredSession.requireHotStepParams(restoredHotParams);
const restoredHotPlan = restoredSession.hotPathPlan(restoredHotParams);
const restoredCompiledLogitsInto = restoredSession.executeInto(restoredCompiledLogitsOut, { input: probeInput });
const restoredCompiledClass = train.classPredictions(restoredCompiledLogits, { classes: 2 })[0];
const restoredCompiledIntoClass = train.class_predictions(restoredCompiledLogitsInto, { numClasses: 2 })[0];
assertHotRuntimeProfile(restoredSession, "restored classifier Session");
if (
  restoredCompiledLogitsInto !== restoredCompiledLogitsOut ||
  restoredCompiledClass !== 1 ||
  restoredCompiledIntoClass !== 1 ||
  restoredHotCompatibility.hotPath !== true ||
  restoredHotCompatibility.runtimeOutputAllocationFree !== true ||
  restoredHotCompatibility.readbackFree !== true ||
  restoredHotPlan.hotPath !== true ||
  restoredHotPlan.stepParamsSignature !== restoredHotCompatibility.stepParamsSignature
) {
  throw new Error(`expected restored classifier executeInto to prove allocation-free hot path: ${JSON.stringify({ restoredHotCompatibility, restoredHotPlan })}`);
}
assertClose(restoredCompiledLogits.data[0], eagerLogits.data[0], 1e-5, "compiled restored classifier logit 0");
assertClose(restoredCompiledLogits.data[1], eagerLogits.data[1], 1e-5, "compiled restored classifier logit 1");
assertClose(restoredCompiledLogitsInto[0], eagerLogits.data[0], 1e-5, "executeInto restored classifier logit 0");
assertClose(restoredCompiledLogitsInto[1], eagerLogits.data[1], 1e-5, "executeInto restored classifier logit 1");
restoredSession.free();
restoredProgram.free();

console.log(`zgml classifier training smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} steps=${fit.steps} ergonomicNativeClass=${ergonomicNativeClass} nativeClass=${nativeClass} compiledClass=${compiledClass} restoredClass=${restoredCompiledClass}`);
