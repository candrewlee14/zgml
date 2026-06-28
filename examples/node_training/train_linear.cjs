"use strict";

const {
  cat,
  checkpoint,
  compile,
  data,
  fit: zgmlFit,
  initial_seed,
  loss,
  manualSeed,
  manual_seed,
  nn,
  optim,
  randn,
  stack,
  tensor,
  train,
} = require("../..");

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

function scalar(t) {
  return typeof t.item === "function" ? t.item() : t.data[0];
}

const model = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
const optimizer = optim.sgd(model, { lr: 0.04 });

manualSeed(123);
manual_seed(123);
if (initial_seed() !== 123) {
  throw new Error("initial_seed should report the last manual seed");
}
const seededNoise = randn([2], { seed: 123 });
if (!seededNoise.allclose(randn([2], { seed: 123 }))) {
  throw new Error("seeded randn should be reproducible");
}

const featureMatrix = tensor([
  -1, -1,
  -1, 1,
  1, -1,
  1, 1,
], [4, 2]);
const targetMatrix = tensor([
  -0.5,
  -2.5,
  2.5,
  0.5,
], [4, 1]);
const featureRows = featureMatrix.reshape([2, 4]).view([4, 2]).transpose().transpose();
if (!featureRows.equal(featureMatrix) || JSON.stringify(featureRows.tolist()) !== JSON.stringify(featureMatrix.tolist())) {
  throw new Error("reshape/view/transpose should preserve feature data");
}
const featureColumns = featureMatrix.mT;
if (featureColumns.shape.join("x") !== "2x4") {
  throw new Error(`mT should transpose feature matrix columns, got ${featureColumns.shape}`);
}
const featureView = featureRows.numpy();
if (!(featureView instanceof Float32Array) || featureView.length !== featureRows.length) {
  throw new Error("tensor.numpy should return a Float32Array view of tensor data");
}
const joinedFeature = cat([featureMatrix.select(0, 0), featureMatrix.select(0, 1)], 0);
const stackedFeature = stack([featureMatrix.select(0, 0), featureMatrix.select(0, 1)], 0);
if (joinedFeature.shape.join("x") !== "4" || stackedFeature.shape.join("x") !== "2x2") {
  throw new Error(`cat/stack produced unexpected shapes: cat=${joinedFeature.shape} stack=${stackedFeature.shape}`);
}

const samples = data.tensor_dataset(featureMatrix, targetMatrix);
const ergonomicBatches = data.batches(samples, { batch_size: 1, drop_last: false, shuffle: true, seed: 17 });
const aliasLoader = data.dataloader(samples, { batch_size: 2, drop_last: false });
if (ergonomicBatches.batch_count !== 4 || aliasLoader.batchSize !== 2 || aliasLoader.dropLast !== false) {
  throw new Error("data alias batching should expose PyTorch-style batch_size/drop_last evidence");
}
const batches = data.dataLoader(samples, { batchSize: 1, shuffle: true, seed: 17 });
const scheduler = optim.stepLR(optimizer, { stepSize: 40, gamma: 0.5 });

const before = scalar(loss.mse(model.forward(tensor([1, -1], [2])), tensor([2.5], [1])));
const fit = train.fit(optimizer, batches, (batch) => {
  return loss.mse(model.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
}, {
  epochs: 20,
  zeroGrad: true,
  clipGradNorm: 1000,
  onStep() {
    scheduler.step();
  },
});
const after = scalar(loss.mse(model.forward(tensor([1, -1], [2])), tensor([2.5], [1])));

if (!train.isTrainFitEvidence(fit) || fit.steps !== 80 || fit.losses.length !== 80) {
  throw new Error("train.fit must return signed fit evidence for every optimizer step");
}
if (!(after < before * 0.02)) {
  throw new Error(`expected training to reduce held-out loss sharply; before=${before}, after=${after}`);
}

const highLevelModel = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
const highLevelOptimizer = optim.sgd(highLevelModel, { lr: 0.04 });
const highLevelBatches = data.dataLoader(samples, { batchSize: 1, shuffle: true, seed: 17 });
const highLevelCriterion = loss.mseLoss();
const highLevelBefore = scalar(loss.mse(highLevelModel.forward(tensor([1, -1], [2])), tensor([2.5], [1])));
const highLevelFit = train.fitModule(highLevelOptimizer, highLevelModel, highLevelBatches, highLevelCriterion, {
  epochs: 20,
  zeroGrad: true,
  clipGradNorm: 1000,
});
const highLevelAfter = scalar(loss.mse(highLevelModel.forward(tensor([1, -1], [2])), tensor([2.5], [1])));
if (!train.isTrainFitEvidence(highLevelFit) || highLevelFit.steps !== 80 || highLevelFit.losses.length !== 80) {
  throw new Error("train.fitModule must return signed fit evidence for every optimizer step");
}
if (!(highLevelAfter < highLevelBefore * 0.02)) {
  throw new Error(`expected train.fitModule to reduce held-out loss sharply; before=${highLevelBefore}, after=${highLevelAfter}`);
}

const nativeModel = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
const nativeOptimizer = optim.sgd(nativeModel, { lr: 0.04 });
const nativeBatches = data.dataLoader(samples, { batchSize: 4, shuffle: false });
const nativeBefore = scalar(loss.mse(nativeModel.forward(tensor([1, -1], [1, 2])), tensor([2.5], [1, 1])));
const nativeTrainer = compile.compileForTraining(nativeModel, nativeOptimizer, {
  inputShape: [4, 2],
  loss: "mse",
});
const nativePlan = nativeTrainer.plan();
if (
  nativePlan.loweredBy !== "zig-ffi" ||
  nativePlan.kernels[0] !== "zgml_train_linear_mse_sgd_f32" ||
  nativePlan.inputShape.join("x") !== "4x2" ||
  nativeTrainer.compileEvidence() !== nativePlan
) {
  throw new Error(`compiled native linear trainer must expose its Zig plan: ${JSON.stringify(nativePlan)}`);
}
const nativeFit = train.fit(nativeTrainer, nativeBatches, { epochs: 80 });
const nativeAfter = scalar(loss.mse(nativeModel.forward(tensor([1, -1], [1, 2])), tensor([2.5], [1, 1])));
if (nativeFit.native !== true || nativeTrainer.native !== true || nativeTrainer.backend !== "cpu") {
  throw new Error("train.fit compiled native linear trainer must return native fit evidence");
}
if (
  nativeFit.compiledPlan !== nativePlan ||
  nativeFit.compiled_plan !== nativePlan ||
  nativeFit.compiledPlan.kernels[0] !== "zgml_train_linear_mse_sgd_f32"
) {
  throw new Error(`compiled native linear fit evidence must retain the Zig plan: ${JSON.stringify(nativeFit.compiledPlan)}`);
}
if (
  !train.isTrainFitEvidence(nativeFit) ||
  nativeFit.steps !== 80 ||
  nativeFit.losses.length !== 1 ||
  nativeFit.nativeBulk !== true ||
  nativeFit.bulkResult?.kernel !== "zgml_train_linear_mse_sgd_f32_bulk"
) {
  throw new Error("compiled native linear trainer must return native bulk fit evidence");
}
if (!(nativeAfter < nativeBefore * 0.02)) {
  throw new Error(`expected compiled native linear trainer to reduce held-out loss sharply; before=${nativeBefore}, after=${nativeAfter}`);
}
nativeTrainer.free();

const ergonomicNativeModel = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
const ergonomicNativeOptimizer = optim.sgd(ergonomicNativeModel, { lr: 0.04 });
const ergonomicNativeBatches = data.dataLoader(samples, { batchSize: 4, shuffle: false });
const ergonomicNativeCriterion = loss.mseLoss();
const ergonomicNativeBefore = scalar(loss.mse(ergonomicNativeModel.forward(tensor([1, -1], [1, 2])), tensor([2.5], [1, 1])));
const ergonomicNativeFit = zgmlFit(
  ergonomicNativeModel,
  ergonomicNativeBatches,
  {
    optimizer: ergonomicNativeOptimizer,
    loss: ergonomicNativeCriterion,
    epochs: 80,
    requireNative: true,
  },
);
const ergonomicNativeAfter = scalar(loss.mse(ergonomicNativeModel.forward(tensor([1, -1], [1, 2])), tensor([2.5], [1, 1])));
if (ergonomicNativeFit.native !== true || ergonomicNativeFit.backend !== "cpu") {
  throw new Error("root fit should automatically compile supported linear MSE training through the native Zig path");
}
if (
  ergonomicNativeFit.compiledPlan?.loweredBy !== "zig-ffi" ||
  ergonomicNativeFit.compiledPlan.kernels[0] !== "zgml_train_linear_mse_sgd_f32" ||
  ergonomicNativeFit.compiled_plan !== ergonomicNativeFit.compiledPlan
) {
  throw new Error(`ergonomic native fit evidence must expose its Zig plan: ${JSON.stringify(ergonomicNativeFit.compiledPlan)}`);
}
if (
  !train.isTrainFitEvidence(ergonomicNativeFit) ||
  ergonomicNativeFit.steps !== 80 ||
  ergonomicNativeFit.losses.length !== 1 ||
  ergonomicNativeFit.nativeBulk !== true ||
  ergonomicNativeFit.bulkResult?.kernel !== "zgml_train_linear_mse_sgd_f32_bulk"
) {
  throw new Error("native root fit must return native bulk fit evidence");
}
if (!(ergonomicNativeAfter < ergonomicNativeBefore * 0.02)) {
  throw new Error(`expected native train.fitModule to reduce held-out loss sharply; before=${ergonomicNativeBefore}, after=${ergonomicNativeAfter}`);
}

const schedulerState = scheduler.stateDict();
if (schedulerState.step !== fit.steps || scheduler.getLastLr() !== optimizer.config().lr) {
  throw new Error(`unexpected scheduler evidence: ${JSON.stringify(schedulerState)} optimizer=${optimizer.config().lr}`);
}

const snapshot = checkpoint.create({
  model,
  optimizer,
  scheduler,
  prefix: "linear",
  metadata: { example: "train_linear", steps: fit.steps },
});
const inspection = checkpoint.inspect(snapshot);
if (!inspection.hasModel || !inspection.hasOptimizer || !inspection.hasScheduler || inspection.modelParameterCount !== 2) {
  throw new Error(`unexpected checkpoint inspection: ${JSON.stringify(inspection)}`);
}
const jsonSnapshot = checkpoint.toJSON(snapshot);
const loadedSnapshot = checkpoint.fromJSON(JSON.parse(JSON.stringify(jsonSnapshot)));
const loadedInspection = checkpoint.inspect(loadedSnapshot);
if (loadedInspection.schedulerStep !== schedulerState.step || loadedInspection.modelParameterNames.join("|") !== "linear.weight|linear.bias") {
  throw new Error(`checkpoint JSON round-trip lost training state: ${JSON.stringify(loadedInspection)}`);
}

const restored = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
const restoredOptimizer = optim.sgd(restored, { lr: 0.04 });
const restoredScheduler = optim.stepLR(restoredOptimizer, { stepSize: 40, gamma: 0.5 });
checkpoint.load(loadedSnapshot, { model: restored, optimizer: restoredOptimizer, scheduler: restoredScheduler, prefix: "linear", strict: true });
if (restoredScheduler.stateDict().step !== schedulerState.step || restoredOptimizer.config().lr !== schedulerState.lastLr) {
  throw new Error(`scheduler restore failed: ${JSON.stringify(restoredScheduler.stateDict())}`);
}

assertClose(scalar(restored.forward(tensor([1, -1], [2]))), 2.5, 0.03, "restored prediction");

console.log(`zgml training smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} fitModuleAfter=${highLevelAfter.toFixed(6)} nativeAfter=${nativeAfter.toFixed(6)} steps=${fit.steps}`);
