"use strict";

const { execFileSync, spawnSync } = require("node:child_process");
const { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { join, resolve } = require("node:path");
const { tmpdir } = require("node:os");

const root = resolve(__dirname, "..");
const tempRoot = mkdtempSync(join(tmpdir(), "zgml-packed-install-"));
const tscBin = require.resolve("typescript/bin/tsc");

function run(command, args, options = {}) {
  execFileSync(command, args, {
    cwd: options.cwd ?? root,
    encoding: "utf8",
    stdio: options.stdio ?? "inherit",
    env: {
      ...process.env,
      npm_config_audit: "false",
      npm_config_fund: "false",
    },
  });
}

function commandExists(command) {
  const result = spawnSync(command, ["--version"], {
    cwd: root,
    encoding: "utf8",
    stdio: "ignore",
  });
  return result.status === 0;
}

function assertPackedPackageScripts(packageRoot) {
  const packageJson = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8"));
  const scripts = packageJson.scripts ?? {};
  const expected = {
    "smoke:bun": "npm run build:package && npm run smoke:bun:dist",
    "smoke:adapters": "npm run build:package && npm run smoke:node && npm run smoke:bun:dist",
  };
  for (const [name, command] of Object.entries(expected)) {
    if (scripts[name] !== command) {
      throw new Error(`packed package ${name} script drifted: expected ${JSON.stringify(command)}, got ${JSON.stringify(scripts[name])}`);
    }
  }
  const bunDist = String(scripts["smoke:bun:dist"] ?? "");
  for (const needle of [
    "bun ./dist/smokes/bun_package_smoke.cjs",
    "bun examples/bun_training/train_linear.ts",
    "bun examples/bun_training/train_mlp.ts",
    "bun examples/bun_program_session/run_linear.ts",
    "bun build ./dist/bun_native.cjs --target=bun",
  ]) {
    if (!bunDist.includes(needle)) {
      throw new Error(`packed package smoke:bun:dist must preserve dist-only Bun adapter proof: missing ${needle}`);
    }
  }
}

const consumerNodeSmoke = `"use strict";

const { checkpoint, data, gradMode, loss, nn, optim, tensor, train } = require("zgml");

function scalar(value) {
  return typeof value.item === "function" ? value.item() : value.data[0];
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
    throw new Error(\`\${label} runtime profile is not hot-path clean: \${JSON.stringify({ expectation, profile })}\`);
  }
}

const model = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const optimizer = optim.adam(model, { lr: 0.03 });
const dataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
  tensor([0, 1, 1, 0], [4, 1]),
);
const before = scalar(loss.mse(model.forward(tensor([1, 0], [2])), tensor([1], [1])));
const fit = train.fit(optimizer, data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 7 }), (batch) => {
  if (!batch.target) throw new Error("consumer batch requires targets");
  return loss.mseLoss().__call__(model.call(batch.input), batch.target);
}, { epochs: 8, zeroGrad: true, clipGradNorm: 1000 });
const after = scalar(loss.mse(model.forward(tensor([1, 0], [2])), tensor([1], [1])));
if (!train.isTrainFitEvidence(fit) || fit.steps !== 16 || !(after < before) || !fit.lastStep?.clipGradNormApplied) {
  throw new Error(\`consumer Node training smoke failed: before=\${before} after=\${after} steps=\${fit.steps}\`);
}
const helperModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const helperOptimizer = optim.adam(helperModel, { lr: 0.03 });
const helperCriterion = loss.mseLoss();
const helperBefore = scalar(loss.mse(helperModel.forward(tensor([1, 0], [2])), tensor([1], [1])));
const helperFit = train.fitModule(helperOptimizer, helperModel, data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 7 }), helperCriterion, {
  epochs: 8,
  zeroGrad: true,
  clipGradNorm: 1000,
});
const helperAfter = scalar(loss.mse(helperModel.forward(tensor([1, 0], [2])), tensor([1], [1])));
if (!train.isTrainFitEvidence(helperFit) || helperFit.steps !== 16 || !(helperAfter < helperBefore) || !helperFit.lastStep?.clipGradNormApplied) {
  throw new Error(\`consumer train.fitModule smoke failed: before=\${helperBefore} after=\${helperAfter} steps=\${helperFit.steps}\`);
}
const classifierModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 2),
]);
const classifierOptimizer = optim.adamW(classifierModel, { lr: 0.05 });
const classifierDataset = data.tensorDataset(
  tensor([-1, -1, -1, 1, 1, -1, 1, 1], [4, 2]),
  tensor([0, 1, 1, 0], [4]),
);
const classifierCriterion = loss.crossEntropyLoss({ classes: 2 });
const classifierBefore = scalar(loss.crossEntropy(classifierModel.forward(tensor([1, -1], [2])), loss.classTargets([1]), { classes: 2 }));
const classifierFit = train.fitClassifier(classifierOptimizer, classifierModel, data.dataLoader(classifierDataset, { batchSize: 2, shuffle: true, seed: 11 }), classifierCriterion, {
  epochs: 20,
  zeroGrad: true,
});
const classifierLogits = classifierModel.forward(tensor([1, -1], [2]));
const classifierAfter = scalar(loss.crossEntropy(classifierLogits, loss.classTargets([1]), { classes: 2 }));
if (!train.isTrainFitEvidence(classifierFit) || classifierFit.steps !== 40 || !(classifierAfter < classifierBefore) || classifierLogits.data[1] <= classifierLogits.data[0]) {
  throw new Error(\`consumer train.fitClassifier smoke failed: before=\${classifierBefore} after=\${classifierAfter} logits=\${Array.from(classifierLogits.data)}\`);
}
classifierModel.eval();
if (!gradMode.isGradEnabled()) {
  throw new Error("consumer gradMode should be enabled before inference helpers");
}
const classifierEvaluation = gradMode.inferenceMode(() => train.evaluate(
  data.dataLoader(classifierDataset, { batchSize: 2, shuffle: false }),
  (batch) => {
    if (!batch.target) throw new Error("consumer classifier eval batch requires targets");
    return classifierCriterion.call(classifierModel.forward(batch.input), batch.target);
  },
));
const classifierPrediction = gradMode.noGrad(() => train.predict(
  data.dataLoader(classifierDataset, { batchSize: 2, shuffle: false }),
  (batch) => classifierModel.forward(batch.input),
));
const classifierInferenceLogits = gradMode.inferenceMode(() => classifierModel.forward(tensor([1, -1], [2])));
if (
  !gradMode.isGradEnabled() ||
  !train.isTrainEvaluateEvidence(classifierEvaluation) ||
  classifierEvaluation.steps !== 2 ||
  !train.isTrainPredictEvidence(classifierPrediction) ||
  classifierPrediction.outputs.length !== 2 ||
  Math.abs(classifierInferenceLogits.data[1] - classifierLogits.data[1]) > 1e-6
) {
  throw new Error(\`consumer gradMode eval/predict smoke failed: enabled=\${gradMode.isGradEnabled()} evaluation=\${JSON.stringify(classifierEvaluation)} predictionSteps=\${classifierPrediction.steps}\`);
}
const stepClipModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const stepClipOptimizer = optim.sgd(stepClipModel, { lr: 0.001 });
const stepClipLoss = loss.mseLoss().__call__(stepClipModel.call(tensor([10], [1])), tensor([0], [1]));
const stepClipEvidence = train.step(stepClipOptimizer, { loss: stepClipLoss, clipGradNorm: 0.25, zeroGrad: false, inspect: true });
if (!stepClipEvidence.clipGradNormApplied || !(stepClipEvidence.gradNormBeforeClip > stepClipEvidence.gradNormAfterClip) || stepClipEvidence.gradNormAfterClip > 0.25001) {
  throw new Error(\`consumer train.step clipGradNorm evidence failed: \${JSON.stringify(stepClipEvidence)}\`);
}
const valueClipModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const valueClipOptimizer = optim.sgd(valueClipModel, { lr: 0.001 });
const valueClipLoss = loss.mseLoss().__call__(valueClipModel.call(tensor([10], [1])), tensor([0], [1]));
const valueClipEvidence = train.step(valueClipOptimizer, { loss: valueClipLoss, clipGradValue: 0.125, zeroGrad: false, inspect: true });
const valueClipMaxGrad = Math.max(...valueClipModel.parameters().flatMap((param) => Array.from(param.grad ?? []).map(Math.abs)));
if (!valueClipEvidence.clipGradValueApplied || valueClipMaxGrad > 0.12501) {
  throw new Error(\`consumer train.step clipGradValue evidence failed: max=\${valueClipMaxGrad} evidence=\${JSON.stringify(valueClipEvidence)}\`);
}
const scheduler = optim.stepLR(optimizer, { stepSize: 2, gamma: 0.5 });
const schedulerBaseLr = optimizer.config().lr;
scheduler.step();
scheduler.step();
const schedulerState = scheduler.stateDict();
if (schedulerState.step !== 2 || schedulerState.lastLr !== schedulerBaseLr * 0.5 || optimizer.config().lr !== schedulerState.lastLr) {
  throw new Error(\`consumer scheduler evidence failed: \${JSON.stringify(schedulerState)} lr=\${optimizer.config().lr}\`);
}
const snapshot = checkpoint.create({ model, optimizer, scheduler, prefix: "consumer" });
const inspection = checkpoint.inspect(snapshot);
if (!inspection.hasModel || !inspection.hasOptimizer || !inspection.hasScheduler || inspection.schedulerKind !== "step-lr" || inspection.schedulerStep !== 2 || inspection.schedulerOptimizerKind !== "adam") {
  throw new Error(\`consumer checkpoint inspection failed: \${JSON.stringify(inspection)}\`);
}
const checkpointJson = checkpoint.toJSON(snapshot);
const parsedCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpointJson)));
const restoredModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const restoredOptimizer = optim.adam(restoredModel, { lr: 0.03 });
const restoredScheduler = optim.stepLR(restoredOptimizer, { stepSize: 2, gamma: 0.5 });
checkpoint.restore(parsedCheckpoint, { model: restoredModel, optimizer: restoredOptimizer, scheduler: restoredScheduler, strict: true, prefix: "consumer" });
const restoredAfter = scalar(loss.mse(restoredModel.forward(tensor([1, 0], [2])), tensor([1], [1])));
if (Math.abs(restoredAfter - after) > 1e-6 || restoredOptimizer.stateDict().step !== optimizer.stateDict().step || restoredScheduler.stateDict().step !== schedulerState.step || restoredOptimizer.config().lr !== schedulerState.lastLr) {
  throw new Error(\`consumer checkpoint scheduler JSON restore failed: after=\${after} restored=\${restoredAfter} step=\${restoredOptimizer.stateDict().step} scheduler=\${JSON.stringify(restoredScheduler.stateDict())}\`);
}
const freezeModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const freezeOptimizer = optim.sgd(freezeModel, { lr: 0.1 });
const freezeWeightBefore = freezeModel.weight[0];
nn.freeze(freezeModel);
if (freezeModel.parameterInfos().some((info) => info.requiresGrad || info.requires_grad)) {
  throw new Error("consumer nn.freeze failed to disable parameter gradients");
}
for (let i = 0; i < 8; i++) {
  freezeOptimizer.zeroGrad();
  const frozenLoss = loss.mseLoss().__call__(freezeModel.call(tensor([2], [1])), tensor([3], [1]));
  frozenLoss.backward();
  freezeOptimizer.step();
}
if (freezeModel.weight[0] !== freezeWeightBefore) {
  throw new Error(\`consumer nn.freeze allowed optimizer update: before=\${freezeWeightBefore} after=\${freezeModel.weight[0]}\`);
}
nn.unfreeze(freezeModel);
if (freezeModel.parameterInfos().some((info) => !info.requiresGrad || !info.requires_grad)) {
  throw new Error("consumer nn.unfreeze failed to enable parameter gradients");
}
for (let i = 0; i < 8; i++) {
  freezeOptimizer.zeroGrad();
  const unfrozenLoss = loss.mseLoss().__call__(freezeModel.call(tensor([2], [1])), tensor([3], [1]));
  unfrozenLoss.backward();
  freezeOptimizer.step();
}
if (freezeModel.weight[0] === freezeWeightBefore) {
  throw new Error("consumer nn.unfreeze failed to restore optimizer updates");
}
const checkpointCompiledModel = nn.linear(2, 1, { weights: [0.25, -0.75], bias: [0.5] });
const checkpointCompiledEager = checkpointCompiledModel.call(tensor([1, 2], [2]));
const checkpointCompiledSnapshot = checkpoint.create({ model: checkpointCompiledModel, prefix: "compiled" });
const checkpointCompiledParsed = checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpoint.toJSON(checkpointCompiledSnapshot))));
const checkpointCompiledRestored = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
checkpoint.restore(checkpointCompiledParsed, { model: checkpointCompiledRestored, strict: true, prefix: "compiled" });
const checkpointCompiledProgram = checkpointCompiledRestored.compile({ backend: "cpu" });
assertHotRuntimeProfile(checkpointCompiledProgram, "consumer checkpoint restored Program");
const checkpointCompiledSession = checkpointCompiledProgram.bindModule(checkpointCompiledRestored);
const checkpointCompiledOutput = gradMode.inferenceMode(() => checkpointCompiledSession.stepTensor(tensor([1, 2], [2])));
assertHotRuntimeProfile(checkpointCompiledSession, "consumer checkpoint restored Session");
if (Math.abs(checkpointCompiledOutput.data[0] - checkpointCompiledEager.data[0]) > 1e-6) {
  throw new Error(\`consumer checkpoint restored Program/Session failed: eager=\${checkpointCompiledEager.data[0]} compiled=\${checkpointCompiledOutput.data[0]}\`);
}
checkpointCompiledSession.free();
checkpointCompiledProgram.free();

const compiledModel = nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const eager = compiledModel.forward(tensor([1, 2], [2]));
const callEager = compiledModel.call(tensor([1, 2], [2]));
const dunderEager = compiledModel.__call__(tensor([1, 2], [2]));
const namespaceCall = nn.call(compiledModel, tensor([1, 2], [2]));
const namespaceDunderCall = nn.__call__(compiledModel, tensor([1, 2], [2]));
const customModel = nn.module({
  kind: "consumer-custom",
  children: [compiledModel],
  parameters: (prefix = "") => compiledModel.parameters(prefix),
  forward(input) {
    return compiledModel.call(input);
  },
});
const customCall = customModel.__call__(tensor([1, 2], [2]));
const customState = customModel.stateDict("custom");
const customOptimizer = optim.sgd(customModel, { lr: 0.01 });
customOptimizer.zero_grad();
if (!customState["custom.weight"] || customModel.children().length !== 1 || customModel.parameters().length !== compiledModel.parameters().length) {
  throw new Error("consumer custom module state/traversal failed");
}
const customWeight = nn.parameter("weight", tensor([0.5, -0.5], [2]));
const customBias = nn.Parameter("bias", new Float32Array([0.1]), [1]);
const customParameterModel = nn.module({
  kind: "consumer-custom-parameters",
  parameters: [customWeight, customBias],
  forward() {
    return tensor([customBias.data[0]], [1]);
  },
});
const customParameterState = customParameterModel.stateDict("head");
const customParameterOptimizer = optim.sgd(customParameterModel, { lr: 0.01 });
customParameterOptimizer.zeroGrad();
if (!customParameterState["head.weight"] || !customParameterState["head.bias"] || customParameterModel.parameters().length !== 2) {
  throw new Error("consumer custom parameter module state failed");
}
const customAutogradWeight = nn.parameter("weight", tensor([0], [1]));
const customAutogradBias = nn.parameter("bias", tensor([0], [1]));
const customAutogradModel = nn.module({
  kind: "consumer-custom-autograd",
  parameters: [customAutogradWeight, customAutogradBias],
  forward(input) {
    return input.mul(customAutogradWeight.tensor).add(customAutogradBias.tensor);
  },
});
const customAutogradOptimizer = optim.sgd(customAutogradModel, { lr: 0.1 });
const customAutogradBefore = scalar(customAutogradModel.call(tensor([2], [1])));
for (let i = 0; i < 40; i++) {
  customAutogradOptimizer.zeroGrad();
  const pred = customAutogradModel.call(tensor([2], [1]));
  const trainLoss = loss.mseLoss().__call__(pred, tensor([5], [1]));
  trainLoss.backward();
  customAutogradOptimizer.step();
}
const customAutogradAfter = scalar(customAutogradModel.call(tensor([2], [1])));
if (Math.abs(customAutogradAfter - 5) > Math.abs(customAutogradBefore - 5) || Math.abs(customAutogradAfter - 5) > 0.05) {
  throw new Error(\`consumer custom autograd module failed: before=\${customAutogradBefore} after=\${customAutogradAfter}\`);
}
const customAutogradState = customAutogradModel.stateDict("trained");
const customAutogradCloneWeight = nn.parameter("weight", tensor([0], [1]));
const customAutogradCloneBias = nn.parameter("bias", tensor([0], [1]));
const customAutogradClone = nn.module({
  kind: "consumer-custom-autograd-clone",
  parameters: [customAutogradCloneWeight, customAutogradCloneBias],
  forward(input) {
    return input.mul(customAutogradCloneWeight.tensor).add(customAutogradCloneBias.tensor);
  },
});
customAutogradClone.loadStateDict(customAutogradState, { strict: true, prefix: "trained", validateOnly: true });
customAutogradClone.load_state_dict(customAutogradState, { strict: true, prefix: "trained" });
const customAutogradCloneAfter = scalar(customAutogradClone.call(tensor([2], [1])));
if (Math.abs(customAutogradCloneAfter - customAutogradAfter) > 1e-6) {
  throw new Error(\`consumer custom autograd state load failed: source=\${customAutogradAfter} clone=\${customAutogradCloneAfter}\`);
}
const program = compiledModel.compile({ backend: "cpu" });
assertHotRuntimeProfile(program, "consumer compiled Program");
const capabilities = program.capabilities();
const requirements = program.requirements();
if (!capabilities.canExecute || requirements.inputLen !== 2 || requirements.outputLen !== 1) {
  throw new Error(\`consumer Program evidence failed: \${JSON.stringify({ capabilities, requirements })}\`);
}
const compiledModelState = compiledModel.stateDict("compiled");
const compiledClone = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
compiledClone.loadStateDict(compiledModelState, { strict: true, prefix: "compiled", validateOnly: true });
compiledClone.load_state_dict(compiledModelState, { strict: true, prefix: "compiled" });
const session = program.bindModule(compiledModel);
const compiled = gradMode.inferenceMode(() => session.stepTensor(tensor([1, 2], [2])));
const compiledInto = new Float32Array(1);
const hotParams = { input: tensor([1, 2], [2]), output: compiledInto };
const hotCompatibility = session.requireHotStepParams(hotParams);
const hotPlan = session.hotPathPlan(hotParams);
const compiledIntoOutput = gradMode.inferenceMode(() => session.executeInto(compiledInto, { input: tensor([1, 2], [2]) }));
if (
  compiledIntoOutput !== compiledInto ||
  !hotCompatibility.hotPath ||
  !hotCompatibility.runtimeOutputAllocationFree ||
  !hotCompatibility.readbackFree ||
  !hotPlan.hotPath ||
  hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature ||
  Math.abs(compiledIntoOutput[0] - eager.data[0]) > 1e-6
) {
  throw new Error(\`consumer executeInto hot path failed: eager=\${eager.data[0]} compiled=\${compiledIntoOutput[0]} compatibility=\${JSON.stringify(hotCompatibility)} plan=\${JSON.stringify(hotPlan)}\`);
}
assertHotRuntimeProfile(session, "consumer compiled Session");
const clonedSession = program.bindModule(compiledClone);
const compiledCloneOutput = gradMode.noGrad(() => clonedSession.stepTensor(tensor([1, 2], [2])));
const clonedInto = new Float32Array(1);
const clonedHotParams = { input: tensor([1, 2], [2]), output: clonedInto };
const clonedHotCompatibility = clonedSession.requireHotStepParams(clonedHotParams);
const clonedHotPlan = clonedSession.hotPathPlan(clonedHotParams);
const clonedIntoOutput = gradMode.noGrad(() => clonedSession.executeInto(clonedInto, { input: tensor([1, 2], [2]) }));
if (
  clonedIntoOutput !== clonedInto ||
  !clonedHotCompatibility.hotPath ||
  !clonedHotCompatibility.runtimeOutputAllocationFree ||
  !clonedHotCompatibility.readbackFree ||
  !clonedHotPlan.hotPath ||
  clonedHotPlan.stepParamsSignature !== clonedHotCompatibility.stepParamsSignature ||
  Math.abs(clonedIntoOutput[0] - eager.data[0]) > 1e-6
) {
  throw new Error(\`consumer state-loaded executeInto hot path failed: eager=\${eager.data[0]} clone=\${clonedIntoOutput[0]} compatibility=\${JSON.stringify(clonedHotCompatibility)} plan=\${JSON.stringify(clonedHotPlan)}\`);
}
assertHotRuntimeProfile(clonedSession, "consumer state-loaded clone Session");
const contract = session.stepContract();
if (program.inputShape().join("x") !== "2" || program.outputShape().join("x") !== "1") {
  throw new Error(\`consumer Program shape evidence failed: \${program.inputShape()} -> \${program.outputShape()}\`);
}
if (contract.inputShape.join("x") !== "2" || contract.outputShape.join("x") !== "1") {
  throw new Error(\`consumer Session contract failed: \${JSON.stringify(contract)}\`);
}
if (Math.abs(compiledCloneOutput.data[0] - eager.data[0]) > 1e-6) {
  throw new Error(\`consumer loaded clone Program/Session mismatch: eager=\${eager.data[0]} clone=\${compiledCloneOutput.data[0]}\`);
}
for (const [label, observed] of [["compiled", compiled], ["call", callEager], ["__call__", dunderEager], ["nn.call", namespaceCall], ["nn.__call__", namespaceDunderCall], ["custom", customCall]]) {
  if (Math.abs(observed.data[0] - eager.data[0]) > 1e-6) {
    throw new Error(\`consumer \${label} output mismatch: eager=\${eager.data[0]} observed=\${observed.data[0]}\`);
  }
}
clonedSession.free();
session.free();
program.free();

console.log(\`zgml consumer Node smoke ok: before=\${before.toFixed(6)} after=\${after.toFixed(6)} steps=\${fit.steps} compiled=\${compiled.data[0].toFixed(6)}\`);
`;

const consumerBunSmoke = `import { checkpoint, data, gradMode, loss, nn, optim, tensor, train } from "zgml/bun";

function scalar(value: { item?: () => number; data: ArrayLike<number> }): number {
  return typeof value.item === "function" ? value.item() : value.data[0] ?? Number.NaN;
}

function assertHotRuntimeProfile(target: { requireHotRuntimeProfile: () => { noFallback: boolean; noSync: boolean; noRuntimePatchInvalid: boolean }; runtimeProfile: () => { fallbackOpCount: number; syncCount: number; runtimePatchInvalidCount: number } }, label: string): void {
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
    throw new Error(\`\${label} runtime profile is not hot-path clean: \${JSON.stringify({ expectation, profile })}\`);
  }
}

const model = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const optimizer = optim.adam(model, { lr: 0.03 });
const dataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const before = scalar(loss.mse(model.forward(tensor([1, 0], [2] as const)), tensor([1], [1] as const)));
const fit = train.fit(optimizer, data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 7 }), (batch) => {
  if (!batch.target) throw new Error("consumer batch requires targets");
  return loss.mseLoss().__call__(model.call(batch.input), batch.target);
}, { epochs: 8, zeroGrad: true, clipGradNorm: 1000 });
const after = scalar(loss.mse(model.forward(tensor([1, 0], [2] as const)), tensor([1], [1] as const)));
if (!train.isTrainFitEvidence(fit) || fit.steps !== 16 || !(after < before) || !fit.lastStep?.clipGradNormApplied) {
  throw new Error(\`consumer Bun training smoke failed: before=\${before} after=\${after} steps=\${fit.steps}\`);
}
const helperModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const helperOptimizer = optim.adam(helperModel, { lr: 0.03 });
const helperCriterion = loss.mseLoss();
const helperBefore = scalar(loss.mse(helperModel.forward(tensor([1, 0], [2] as const)), tensor([1], [1] as const)));
const helperFit = train.fitModule(helperOptimizer, helperModel, data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 7 }), helperCriterion, {
  epochs: 8,
  zeroGrad: true,
  clipGradNorm: 1000,
});
const helperAfter = scalar(loss.mse(helperModel.forward(tensor([1, 0], [2] as const)), tensor([1], [1] as const)));
if (!train.isTrainFitEvidence(helperFit) || helperFit.steps !== 16 || !(helperAfter < helperBefore) || !helperFit.lastStep?.clipGradNormApplied) {
  throw new Error(\`consumer Bun train.fitModule smoke failed: before=\${helperBefore} after=\${helperAfter} steps=\${helperFit.steps}\`);
}
const classifierModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 2),
]);
const classifierOptimizer = optim.adamW(classifierModel, { lr: 0.05 });
const classifierDataset = data.tensorDataset(
  tensor([-1, -1, -1, 1, 1, -1, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4] as const),
);
const classifierCriterion = loss.crossEntropyLoss({ classes: 2 });
const classifierBefore = scalar(loss.crossEntropy(classifierModel.forward(tensor([1, -1], [2] as const)), loss.classTargets([1]), { classes: 2 }));
const classifierFit = train.fitClassifier(classifierOptimizer, classifierModel, data.dataLoader(classifierDataset, { batchSize: 2, shuffle: true, seed: 11 }), classifierCriterion, {
  epochs: 20,
  zeroGrad: true,
});
const classifierLogits = classifierModel.forward(tensor([1, -1], [2] as const));
const classifierAfter = scalar(loss.crossEntropy(classifierLogits, loss.classTargets([1]), { classes: 2 }));
if (!train.isTrainFitEvidence(classifierFit) || classifierFit.steps !== 40 || !(classifierAfter < classifierBefore) || (classifierLogits.data[1] ?? Number.NEGATIVE_INFINITY) <= (classifierLogits.data[0] ?? Number.POSITIVE_INFINITY)) {
  throw new Error(\`consumer Bun train.fitClassifier smoke failed: before=\${classifierBefore} after=\${classifierAfter} logits=\${Array.from(classifierLogits.data)}\`);
}
classifierModel.eval();
if (!gradMode.isGradEnabled()) {
  throw new Error("consumer Bun gradMode should be enabled before inference helpers");
}
const classifierEvaluation = gradMode.inferenceMode(() => train.evaluate(
  data.dataLoader(classifierDataset, { batchSize: 2, shuffle: false }),
  (batch) => {
    if (!batch.target) throw new Error("consumer Bun classifier eval batch requires targets");
    return classifierCriterion.call(classifierModel.forward(batch.input), batch.target);
  },
));
const classifierPrediction = gradMode.noGrad(() => train.predict(
  data.dataLoader(classifierDataset, { batchSize: 2, shuffle: false }),
  (batch) => classifierModel.forward(batch.input),
));
const classifierInferenceLogits = gradMode.inferenceMode(() => classifierModel.forward(tensor([1, -1], [2] as const)));
if (
  !gradMode.isGradEnabled() ||
  !train.isTrainEvaluateEvidence(classifierEvaluation) ||
  classifierEvaluation.steps !== 2 ||
  !train.isTrainPredictEvidence(classifierPrediction) ||
  classifierPrediction.outputs.length !== 2 ||
  Math.abs((classifierInferenceLogits.data[1] ?? Number.NaN) - (classifierLogits.data[1] ?? Number.NaN)) > 1e-6
) {
  throw new Error(\`consumer Bun gradMode eval/predict smoke failed: enabled=\${gradMode.isGradEnabled()} evaluation=\${JSON.stringify(classifierEvaluation)} predictionSteps=\${classifierPrediction.steps}\`);
}
const stepClipModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const stepClipOptimizer = optim.sgd(stepClipModel, { lr: 0.001 });
const stepClipLoss = loss.mseLoss().__call__(stepClipModel.call(tensor([10], [1] as const)), tensor([0], [1] as const));
const stepClipEvidence = train.step(stepClipOptimizer, { loss: stepClipLoss, clipGradNorm: 0.25, zeroGrad: false, inspect: true });
if (!stepClipEvidence.clipGradNormApplied || !((stepClipEvidence.gradNormBeforeClip ?? 0) > (stepClipEvidence.gradNormAfterClip ?? 0)) || (stepClipEvidence.gradNormAfterClip ?? Infinity) > 0.25001) {
  throw new Error(\`consumer Bun train.step clipGradNorm evidence failed: \${JSON.stringify(stepClipEvidence)}\`);
}
const valueClipModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const valueClipOptimizer = optim.sgd(valueClipModel, { lr: 0.001 });
const valueClipLoss = loss.mseLoss().__call__(valueClipModel.call(tensor([10], [1] as const)), tensor([0], [1] as const));
const valueClipEvidence = train.step(valueClipOptimizer, { loss: valueClipLoss, clipGradValue: 0.125, zeroGrad: false, inspect: true });
const valueClipMaxGrad = Math.max(...valueClipModel.parameters().flatMap((param) => Array.from(param.grad ?? []).map(Math.abs)));
if (!valueClipEvidence.clipGradValueApplied || valueClipMaxGrad > 0.12501) {
  throw new Error(\`consumer Bun train.step clipGradValue evidence failed: max=\${valueClipMaxGrad} evidence=\${JSON.stringify(valueClipEvidence)}\`);
}
const scheduler = optim.stepLR(optimizer, { stepSize: 2, gamma: 0.5 });
const schedulerBaseLr = optimizer.config().lr;
scheduler.step();
scheduler.step();
const schedulerState = scheduler.stateDict();
if (schedulerState.step !== 2 || schedulerState.lastLr !== schedulerBaseLr * 0.5 || optimizer.config().lr !== schedulerState.lastLr) {
  throw new Error(\`consumer Bun scheduler evidence failed: \${JSON.stringify(schedulerState)} lr=\${optimizer.config().lr}\`);
}
const snapshot = checkpoint.create({ model, optimizer, scheduler, prefix: "consumer" });
const inspection = checkpoint.inspect(snapshot);
if (!inspection.hasModel || !inspection.hasOptimizer || !inspection.hasScheduler || inspection.schedulerKind !== "step-lr" || inspection.schedulerStep !== 2 || inspection.schedulerOptimizerKind !== "adam") {
  throw new Error(\`consumer checkpoint inspection failed: \${JSON.stringify(inspection)}\`);
}
const checkpointJson = checkpoint.toJSON(snapshot);
const parsedCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpointJson)));
const restoredModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const restoredOptimizer = optim.adam(restoredModel, { lr: 0.03 });
const restoredScheduler = optim.stepLR(restoredOptimizer, { stepSize: 2, gamma: 0.5 });
checkpoint.restore(parsedCheckpoint, { model: restoredModel, optimizer: restoredOptimizer, scheduler: restoredScheduler, strict: true, prefix: "consumer" });
const restoredAfter = scalar(loss.mse(restoredModel.forward(tensor([1, 0], [2] as const)), tensor([1], [1] as const)));
if (Math.abs(restoredAfter - after) > 1e-6 || restoredOptimizer.stateDict().step !== optimizer.stateDict().step || restoredScheduler.stateDict().step !== schedulerState.step || restoredOptimizer.config().lr !== schedulerState.lastLr) {
  throw new Error(\`consumer Bun checkpoint scheduler JSON restore failed: after=\${after} restored=\${restoredAfter} step=\${restoredOptimizer.stateDict().step} scheduler=\${JSON.stringify(restoredScheduler.stateDict())}\`);
}
const freezeModel = nn.linear(1, 1, { weights: [1], bias: [0] });
const freezeOptimizer = optim.sgd(freezeModel, { lr: 0.1 });
const freezeWeightBefore = freezeModel.weight[0] ?? Number.NaN;
nn.freeze(freezeModel);
if (freezeModel.parameterInfos().some((info) => info.requiresGrad || info.requires_grad)) {
  throw new Error("consumer Bun nn.freeze failed to disable parameter gradients");
}
for (let i = 0; i < 8; i++) {
  freezeOptimizer.zeroGrad();
  const frozenLoss = loss.mseLoss().__call__(freezeModel.call(tensor([2], [1] as const)), tensor([3], [1] as const));
  frozenLoss.backward();
  freezeOptimizer.step();
}
if ((freezeModel.weight[0] ?? Number.NaN) !== freezeWeightBefore) {
  throw new Error(\`consumer Bun nn.freeze allowed optimizer update: before=\${freezeWeightBefore} after=\${freezeModel.weight[0]}\`);
}
nn.unfreeze(freezeModel);
if (freezeModel.parameterInfos().some((info) => !info.requiresGrad || !info.requires_grad)) {
  throw new Error("consumer Bun nn.unfreeze failed to enable parameter gradients");
}
for (let i = 0; i < 8; i++) {
  freezeOptimizer.zeroGrad();
  const unfrozenLoss = loss.mseLoss().__call__(freezeModel.call(tensor([2], [1] as const)), tensor([3], [1] as const));
  unfrozenLoss.backward();
  freezeOptimizer.step();
}
if ((freezeModel.weight[0] ?? Number.NaN) === freezeWeightBefore) {
  throw new Error("consumer Bun nn.unfreeze failed to restore optimizer updates");
}
const checkpointCompiledModel = nn.linear(2, 1, { weights: [0.25, -0.75], bias: [0.5] });
const checkpointCompiledEager = checkpointCompiledModel.call(tensor([1, 2], [2] as const));
const checkpointCompiledSnapshot = checkpoint.create({ model: checkpointCompiledModel, prefix: "compiled" });
const checkpointCompiledParsed = checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpoint.toJSON(checkpointCompiledSnapshot))));
const checkpointCompiledRestored = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
checkpoint.restore(checkpointCompiledParsed, { model: checkpointCompiledRestored, strict: true, prefix: "compiled" });
const checkpointCompiledProgram = checkpointCompiledRestored.compile({ backend: "cpu" });
assertHotRuntimeProfile(checkpointCompiledProgram, "consumer Bun checkpoint restored Program");
const checkpointCompiledSession = checkpointCompiledProgram.bindModule(checkpointCompiledRestored);
const checkpointCompiledOutput = gradMode.inferenceMode(() => checkpointCompiledSession.stepTensor(tensor([1, 2], [2] as const)));
assertHotRuntimeProfile(checkpointCompiledSession, "consumer Bun checkpoint restored Session");
if (Math.abs((checkpointCompiledOutput.data[0] ?? Number.NaN) - (checkpointCompiledEager.data[0] ?? Number.NaN)) > 1e-6) {
  throw new Error(\`consumer Bun checkpoint restored Program/Session failed: eager=\${checkpointCompiledEager.data[0]} compiled=\${checkpointCompiledOutput.data[0]}\`);
}
checkpointCompiledSession.free();
checkpointCompiledProgram.free();

const compiledModel = nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const eager = compiledModel.forward(tensor([1, 2], [2] as const));
const callEager = compiledModel.call(tensor([1, 2], [2] as const));
const dunderEager = compiledModel.__call__(tensor([1, 2], [2] as const));
const namespaceCall = nn.call(compiledModel, tensor([1, 2], [2] as const));
const namespaceDunderCall = nn.__call__(compiledModel, tensor([1, 2], [2] as const));
const customModel = nn.module({
  kind: "consumer-custom",
  children: [compiledModel],
  parameters: (prefix = "") => compiledModel.parameters(prefix),
  forward(input: ReturnType<typeof tensor>) {
    return compiledModel.call(input);
  },
});
const customCall = customModel.__call__(tensor([1, 2], [2] as const));
const customState = customModel.stateDict("custom");
const customOptimizer = optim.sgd(customModel, { lr: 0.01 });
customOptimizer.zero_grad();
if (!customState["custom.weight"] || customModel.children().length !== 1 || customModel.parameters().length !== compiledModel.parameters().length) {
  throw new Error("consumer Bun custom module state/traversal failed");
}
const customWeight = nn.parameter("weight", tensor([0.5, -0.5], [2] as const));
const customBias = nn.Parameter("bias", new Float32Array([0.1]), [1] as const);
const customParameterModel = nn.module({
  kind: "consumer-custom-parameters",
  parameters: [customWeight, customBias],
  forward() {
    return tensor([customBias.data[0] ?? 0], [1] as const);
  },
});
const customParameterState = customParameterModel.stateDict("head");
const customParameterOptimizer = optim.sgd(customParameterModel, { lr: 0.01 });
customParameterOptimizer.zeroGrad();
if (!customParameterState["head.weight"] || !customParameterState["head.bias"] || customParameterModel.parameters().length !== 2) {
  throw new Error("consumer Bun custom parameter module state failed");
}
const customAutogradWeight = nn.parameter("weight", tensor([0], [1] as const));
const customAutogradBias = nn.parameter("bias", tensor([0], [1] as const));
const customAutogradModel = nn.module({
  kind: "consumer-custom-autograd",
  parameters: [customAutogradWeight, customAutogradBias],
  forward(input: ReturnType<typeof tensor>) {
    return input.mul(customAutogradWeight.tensor).add(customAutogradBias.tensor);
  },
});
const customAutogradOptimizer = optim.sgd(customAutogradModel, { lr: 0.1 });
const customAutogradBefore = scalar(customAutogradModel.call(tensor([2], [1] as const)));
for (let i = 0; i < 40; i++) {
  customAutogradOptimizer.zeroGrad();
  const pred = customAutogradModel.call(tensor([2], [1] as const));
  const trainLoss = loss.mseLoss().__call__(pred, tensor([5], [1] as const));
  trainLoss.backward();
  customAutogradOptimizer.step();
}
const customAutogradAfter = scalar(customAutogradModel.call(tensor([2], [1] as const)));
if (Math.abs(customAutogradAfter - 5) > Math.abs(customAutogradBefore - 5) || Math.abs(customAutogradAfter - 5) > 0.05) {
  throw new Error(\`consumer Bun custom autograd module failed: before=\${customAutogradBefore} after=\${customAutogradAfter}\`);
}
const customAutogradState = customAutogradModel.stateDict("trained");
const customAutogradCloneWeight = nn.parameter("weight", tensor([0], [1] as const));
const customAutogradCloneBias = nn.parameter("bias", tensor([0], [1] as const));
const customAutogradClone = nn.module({
  kind: "consumer-custom-autograd-clone",
  parameters: [customAutogradCloneWeight, customAutogradCloneBias],
  forward(input: ReturnType<typeof tensor>) {
    return input.mul(customAutogradCloneWeight.tensor).add(customAutogradCloneBias.tensor);
  },
});
customAutogradClone.loadStateDict(customAutogradState, { strict: true, prefix: "trained", validateOnly: true });
customAutogradClone.load_state_dict(customAutogradState, { strict: true, prefix: "trained" });
const customAutogradCloneAfter = scalar(customAutogradClone.call(tensor([2], [1] as const)));
if (Math.abs(customAutogradCloneAfter - customAutogradAfter) > 1e-6) {
  throw new Error(\`consumer Bun custom autograd state load failed: source=\${customAutogradAfter} clone=\${customAutogradCloneAfter}\`);
}
const program = compiledModel.compile({ backend: "cpu" });
assertHotRuntimeProfile(program, "consumer Bun compiled Program");
const capabilities = program.capabilities();
const requirements = program.requirements();
if (!capabilities.canExecute || requirements.inputLen !== 2 || requirements.outputLen !== 1) {
  throw new Error(\`consumer Bun Program evidence failed: \${JSON.stringify({ capabilities, requirements })}\`);
}
const compiledModelState = compiledModel.stateDict("compiled");
const compiledClone = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
compiledClone.loadStateDict(compiledModelState, { strict: true, prefix: "compiled", validateOnly: true });
compiledClone.load_state_dict(compiledModelState, { strict: true, prefix: "compiled" });
const session = program.bindModule(compiledModel);
const compiled = gradMode.inferenceMode(() => session.stepTensor(tensor([1, 2], [2] as const)));
const compiledInto = new Float32Array(1);
const hotParams = { input: tensor([1, 2], [2] as const), output: compiledInto };
const hotCompatibility = session.requireHotStepParams(hotParams);
const hotPlan = session.hotPathPlan(hotParams);
const compiledIntoOutput = gradMode.inferenceMode(() => session.executeInto(compiledInto, { input: tensor([1, 2], [2] as const) }));
if (
  compiledIntoOutput !== compiledInto ||
  !hotCompatibility.hotPath ||
  !hotCompatibility.runtimeOutputAllocationFree ||
  !hotCompatibility.readbackFree ||
  !hotPlan.hotPath ||
  hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature ||
  Math.abs((compiledIntoOutput[0] ?? Number.NaN) - (eager.data[0] ?? Number.NaN)) > 1e-6
) {
  throw new Error(\`consumer Bun executeInto hot path failed: eager=\${eager.data[0]} compiled=\${compiledIntoOutput[0]} compatibility=\${JSON.stringify(hotCompatibility)} plan=\${JSON.stringify(hotPlan)}\`);
}
assertHotRuntimeProfile(session, "consumer Bun compiled Session");
const clonedSession = program.bindModule(compiledClone);
const compiledCloneOutput = gradMode.noGrad(() => clonedSession.stepTensor(tensor([1, 2], [2] as const)));
const clonedInto = new Float32Array(1);
const clonedHotParams = { input: tensor([1, 2], [2] as const), output: clonedInto };
const clonedHotCompatibility = clonedSession.requireHotStepParams(clonedHotParams);
const clonedHotPlan = clonedSession.hotPathPlan(clonedHotParams);
const clonedIntoOutput = gradMode.noGrad(() => clonedSession.executeInto(clonedInto, { input: tensor([1, 2], [2] as const) }));
if (
  clonedIntoOutput !== clonedInto ||
  !clonedHotCompatibility.hotPath ||
  !clonedHotCompatibility.runtimeOutputAllocationFree ||
  !clonedHotCompatibility.readbackFree ||
  !clonedHotPlan.hotPath ||
  clonedHotPlan.stepParamsSignature !== clonedHotCompatibility.stepParamsSignature ||
  Math.abs((clonedIntoOutput[0] ?? Number.NaN) - (eager.data[0] ?? Number.NaN)) > 1e-6
) {
  throw new Error(\`consumer Bun state-loaded executeInto hot path failed: eager=\${eager.data[0]} clone=\${clonedIntoOutput[0]} compatibility=\${JSON.stringify(clonedHotCompatibility)} plan=\${JSON.stringify(clonedHotPlan)}\`);
}
assertHotRuntimeProfile(clonedSession, "consumer Bun state-loaded clone Session");
const contract = session.stepContract();
if (program.inputShape().join("x") !== "2" || program.outputShape().join("x") !== "1") {
  throw new Error(\`consumer Bun Program shape evidence failed: \${program.inputShape()} -> \${program.outputShape()}\`);
}
if (contract.inputShape.join("x") !== "2" || contract.outputShape.join("x") !== "1") {
  throw new Error(\`consumer Bun Session contract failed: \${JSON.stringify(contract)}\`);
}
if (Math.abs((compiledCloneOutput.data[0] ?? Number.NaN) - (eager.data[0] ?? Number.NaN)) > 1e-6) {
  throw new Error(\`consumer Bun loaded clone Program/Session mismatch: eager=\${eager.data[0]} clone=\${compiledCloneOutput.data[0]}\`);
}
for (const [label, observed] of [["compiled", compiled], ["call", callEager], ["__call__", dunderEager], ["nn.call", namespaceCall], ["nn.__call__", namespaceDunderCall], ["custom", customCall]] as const) {
  if (Math.abs((observed.data[0] ?? Number.NaN) - (eager.data[0] ?? Number.NaN)) > 1e-6) {
    throw new Error(\`consumer Bun \${label} output mismatch: eager=\${eager.data[0]} observed=\${observed.data[0]}\`);
  }
}
clonedSession.free();
session.free();
program.free();

console.log(\`zgml consumer Bun smoke ok: before=\${before.toFixed(6)} after=\${after.toFixed(6)} steps=\${fit.steps} compiled=\${(compiled.data[0] ?? Number.NaN).toFixed(6)}\`);
`;

const consumerBrowserSmoke = `"use strict";

const browser = require("zgml/browser");

if (
  browser.frontendManifest?.source !== "ts" ||
  browser.frontendManifest?.productSourceOfTruth !== "ts-only" ||
  browser.frontendManifest?.runtimePath !== "Program -> Session -> StepParams"
) {
  throw new Error(\`consumer browser frontend manifest drifted: \${JSON.stringify(browser.frontendManifest)}\`);
}

if (
  browser.browserManifest?.kind !== "zgml-browser-frontend" ||
  browser.browserManifest?.policyOwner !== "src/ts/browser.ts" ||
  browser.browserManifest?.nativeLoader !== false ||
  browser.browserManifest?.adapterRole !== "browser-safe-frontend" ||
  browser.browserManifest?.packageFanout !== "tsdown"
) {
  throw new Error(\`consumer browser entry manifest drifted: \${JSON.stringify(browser.browserManifest)}\`);
}

if (typeof browser.nativeApiContract !== "object" || browser.nativeApiContract === null) {
  throw new Error("consumer browser subpath must expose the native API contract namespace without loading a native adapter");
}

const browserInput = browser.tensor([1, 2], [2]);
const browserModel = browser.nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const browserOutput = browserModel.call(browserInput);
if (!(browserOutput instanceof browser.Tensor) || Math.abs(browserOutput.data[0] + 0.5) > 1e-6) {
  throw new Error(\`consumer browser eager nn.linear failed: \${browserOutput.data[0]}\`);
}
const browserLoss = browser.loss.mseLoss().__call__(browserOutput, browser.tensor([0], [1]));
browserLoss.backward();
const browserOptimizer = browser.optim.sgd(browserModel, { lr: 0.01 });
const browserWeightBefore = browserModel.weight[0];
browserOptimizer.step();
if (browserModel.weight[0] === browserWeightBefore) {
  throw new Error("consumer browser optimizer failed to update eager parameters");
}
const browserFitModel = browser.nn.linear(1, 1, { weights: [0], bias: [0] });
const browserFitOptimizer = browser.optim.sgd(browserFitModel, { lr: 0.05 });
const browserFit = browser.train.fit(
  browserFitOptimizer,
  [{ input: browser.tensor([2], [1]), target: browser.tensor([4], [1]) }],
  (batch) => browser.loss.mseLoss().__call__(browserFitModel.call(batch.input), batch.target),
  { epochs: 4, zeroGrad: true },
);
if (!browser.train.isTrainFitEvidence(browserFit) || browserFit.steps !== 4) {
  throw new Error(\`consumer browser train.fit evidence failed: \${JSON.stringify(browserFit)}\`);
}
const browserLoaderModel = browser.nn.linear(2, 1, { weights: [0.25, -0.25], bias: [0] });
const browserLoaderOptimizer = browser.optim.sgd(browserLoaderModel, { lr: 0.02 });
const browserCriterion = browser.loss.mseLoss();
const browserDataset = browser.data.tensorDataset(
  browser.tensor([1, 2, 2, 3, 3, 4, 4, 5], [4, 2]),
  browser.tensor([-0.5, -0.5, -0.5, -0.5], [4, 1]),
);
const browserLoader = browser.data.dataLoader(browserDataset, { batchSize: 2, shuffle: true, seed: 5 });
const browserLoaderBatch = browserLoader.get(0);
if (
  !Object.isFrozen(browserDataset) ||
  browserDataset.length !== 4 ||
  !Object.isFrozen(browserLoader) ||
  browserLoader.batchCount !== 2 ||
  browserLoaderBatch.input.shape.join("x") !== "2x2" ||
  browserLoaderBatch.target.shape.join("x") !== "2x1"
) {
  throw new Error("consumer browser data loader failed to batch eager tensors");
}
const browserFitSteps = [];
const browserLoaderFit = browser.train.fit(
  browserLoaderOptimizer,
  browserLoader,
  (batch, context) => {
    if (!Object.isFrozen(context) || !batch.target) {
      throw new Error("consumer browser train.fit did not provide frozen context and labelled batch");
    }
    return browserCriterion.forward(browserLoaderModel.forward(batch.input), batch.target);
  },
  { epochs: 1, zeroGrad: true, onStep: (evidence) => browserFitSteps.push(evidence) },
);
const browserEvaluation = browser.gradMode.inferenceMode(() =>
  browser.train.evaluate(browserLoader, (batch) => {
    if (!batch.target) {
      throw new Error("consumer browser train.evaluate received unlabelled batch");
    }
    return browserCriterion.forward(browserLoaderModel.forward(batch.input), batch.target);
  }),
);
const browserPrediction = browser.gradMode.noGrad(() =>
  browser.train.predict(browserLoader, (batch) => browserLoaderModel.forward(batch.input)),
);
if (
  !browser.train.isTrainFitEvidence(browserLoaderFit) ||
  browserLoaderFit.steps !== 2 ||
  browserFitSteps.length !== 2 ||
  !browser.train.isTrainFitStepEvidence(browserFitSteps[0]) ||
  !browser.train.isTrainEvaluateEvidence(browserEvaluation) ||
  browserEvaluation.steps !== 2 ||
  !browser.train.isTrainPredictEvidence(browserPrediction) ||
  browserPrediction.outputs.length !== 2
) {
  throw new Error("consumer browser data loader train/evaluate/predict evidence failed");
}
let browserCompileRejected = false;
try {
  browserModel.compile({ backend: "webgpu" });
} catch (error) {
  browserCompileRejected = /zgml\\/browser|native browser Program/.test(String(error && error.message));
}
if (!browserCompileRejected) {
  throw new Error("consumer browser compile must reject with an honest native Program message");
}

console.log(\`zgml consumer browser subpath smoke ok: eager=\${browserOutput.data[0].toFixed(6)} fitSteps=\${browserFit.steps}\`);
`;

const consumerTypeSmoke = `import {
  checkpoint,
  data,
  gradMode,
  loss,
  nn,
  optim,
  tensor,
  train,
  type Optimizer,
  type LRScheduler,
  type Program,
  type Session,
  type SessionExecutionPlan,
  type SessionStepParamsCompatibility,
  type Tensor,
  type TrainFitEvidence,
  type TrainClassificationCriterion,
  type TrainSupervisedCriterion,
  type TrainStepEvidence,
  type CustomModule,
  type ModuleStateSnapshot,
  type NnParameter,
} from "zgml";
import {
  gradMode as bunGradMode,
  nn as bunNn,
  tensor as bunTensor,
  type Program as BunProgram,
  type Session as BunSession,
  type SessionExecutionPlan as BunSessionExecutionPlan,
  type SessionStepParamsCompatibility as BunSessionStepParamsCompatibility,
} from "zgml/bun";
import {
  browserManifest as browserEntryManifest,
  frontendManifest as browserFrontendManifest,
  nativeApiContract as browserNativeApiContract,
  gradMode as browserGradMode,
  loss as browserLoss,
  nn as browserNn,
  optim as browserOptim,
  tensor as browserTensor,
  train as browserTrain,
} from "zgml/browser";

const model = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const optimizer: Optimizer<"adam"> = optim.adam(model, { lr: 0.01 });
const dataset = data.tensorDataset(
  tensor([0, 0, 1, 1], [2, 2] as const),
  tensor([0, 1], [2, 1] as const),
);
const fit: TrainFitEvidence<"adam"> = train.fit(
  optimizer,
  data.dataLoader(dataset, { batchSize: 1 }),
  (batch) => loss.mseLoss().__call__(model.call(batch.input), batch.target as Tensor),
  { epochs: 1, zeroGrad: true, clipGradNorm: 1, clipGradValue: 0.5 },
);
const helperLoader = data.dataLoader(dataset, { batchSize: 1 });
type HelperBatch = ReturnType<typeof helperLoader.__getitem__>;
const helperCriterion: TrainSupervisedCriterion<typeof model, HelperBatch> = loss.mseLoss();
const helperFit: TrainFitEvidence<"adam"> = train.fitModule(optimizer, model, helperLoader, helperCriterion, {
  epochs: 1,
  zeroGrad: true,
});
const classifier = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 2),
]);
const classifierOptimizer: Optimizer<"adamw"> = optim.adamW(classifier, { lr: 0.01 });
const classifierDataset = data.tensorDataset(
  tensor([0, 0, 1, 1], [2, 2] as const),
  tensor([0, 1], [2] as const),
);
const classifierLoader = data.dataLoader(classifierDataset, { batchSize: 1 });
type ClassifierBatch = ReturnType<typeof classifierLoader.__getitem__>;
const classifierCriterion: TrainClassificationCriterion<typeof classifier, ClassifierBatch> = loss.crossEntropyLoss({ classes: 2 });
const classifierFit: TrainFitEvidence<"adamw"> = train.fitClassifier(classifierOptimizer, classifier, classifierLoader, classifierCriterion, {
  epochs: 1,
  zeroGrad: true,
});
classifier.eval();
const typedClassifierEvaluation = gradMode.inferenceMode(() => train.evaluate(classifierLoader, (batch) => classifierCriterion.forward(classifier.forward(batch.input), batch.target as Tensor<readonly [number]>)));
const typedClassifierPrediction = gradMode.noGrad(() => train.predict(classifierLoader, (batch) => classifier.forward(batch.input)));
const typedStepEvidence: TrainStepEvidence<"adam"> = train.step(optimizer, {
  loss: loss.mseLoss().__call__(model.call(tensor([1, 1], [2] as const)), tensor([1], [1] as const)),
  clipGradNorm: 1,
  clipGradNormOptions: { eps: 1e-6 },
  clipGradValue: 0.5,
  zeroGrad: false,
  inspect: true,
});
const typedStepClipNormApplied: boolean = typedStepEvidence.clipGradNormApplied;
const typedStepClipValueApplied: boolean = typedStepEvidence.clipGradValueApplied;
const typedStepGradNormBeforeClip: number | null = typedStepEvidence.gradNormBeforeClip;
const typedStepGradNormAfterClip: number | null = typedStepEvidence.gradNormAfterClip;
const typedScheduler: LRScheduler<"step-lr", "adam"> = optim.stepLR(optimizer, { stepSize: 2, gamma: 0.5 });
typedScheduler.step();
const typedSchedulerState = typedScheduler.stateDict();
const typedCheckpoint = checkpoint.create({ model, optimizer, scheduler: typedScheduler, prefix: "consumer" });
const typedCheckpointJson = checkpoint.toJSON(typedCheckpoint);
const typedParsedCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(typedCheckpointJson)));
const typedRestoredModel = nn.sequential([
  nn.linear(2, 3),
  nn.relu(),
  nn.linear(3, 1),
]);
const typedRestoredOptimizer: Optimizer<"adam"> = optim.adam(typedRestoredModel, { lr: 0.01 });
const typedRestoredScheduler: LRScheduler<"step-lr", "adam"> = optim.stepLR(typedRestoredOptimizer, { stepSize: 2, gamma: 0.5 });
const typedCheckpointRestoreTargets = checkpoint.restore(typedParsedCheckpoint, {
  model: typedRestoredModel,
  optimizer: typedRestoredOptimizer,
  scheduler: typedRestoredScheduler,
  strict: true,
  prefix: "consumer",
});
const typedFreezeTarget: typeof typedRestoredModel = nn.freeze(typedRestoredModel);
const typedUnfreezeTarget: typeof typedRestoredModel = nn.unfreeze(typedRestoredModel);
const typedCheckpointCompiledModel = nn.linear(2, 1);
const typedCompiledCheckpoint = checkpoint.create({ model: typedCheckpointCompiledModel, prefix: "compiled" });
const typedCompiledCheckpointRestored = nn.linear(2, 1);
checkpoint.restore(checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpoint.toJSON(typedCompiledCheckpoint)))), {
  model: typedCompiledCheckpointRestored,
  strict: true,
  prefix: "compiled",
});
const typedRestoredCompiledProgram: Program<readonly [2], readonly [1]> = typedCompiledCheckpointRestored.compile({ backend: "cpu" });
const typedRestoredCompiledSession: Session<readonly [2], readonly [1]> = typedRestoredCompiledProgram.bindModule(typedCompiledCheckpointRestored);
const typedRestoredCompiledOutput: Tensor<readonly [1]> = gradMode.inferenceMode(() => typedRestoredCompiledSession.stepTensor(tensor([1, 2], [2] as const)));

const linear = nn.linear(2, 1);
const customChild = nn.linear(2, 1);
const customModule: CustomModule<readonly [2], readonly [1]> = nn.module({
  kind: "consumer-custom",
  children: [customChild],
  parameters: (prefix = "") => customChild.parameters(prefix),
  forward(input: Tensor<readonly [2]>) {
    return customChild.call(input);
  },
});
const customModuleOutput: Tensor<readonly [1]> = customModule.__call__(tensor([1, 2], [2] as const));
const customModuleOptimizer: Optimizer<"sgd"> = optim.sgd(customModule, { lr: 0.01 });
customModuleOptimizer.zero_grad();
const customWeightParameter: NnParameter<readonly [2]> = nn.parameter("weight", tensor([0.5, -0.5], [2] as const));
const customBiasParameter: NnParameter<readonly [1]> = nn.Parameter("bias", new Float32Array([0.1]), [1] as const);
const customParameterModule: CustomModule<readonly [2], readonly [1]> = nn.module({
  kind: "consumer-custom-parameters",
  parameters: [customWeightParameter, customBiasParameter],
  forward(_input: Tensor<readonly [2]>) {
    return tensor([customBiasParameter.data[0] ?? 0], [1] as const);
  },
});
const customParameterState = customParameterModule.stateDict("head");
const customParameterOptimizer: Optimizer<"sgd"> = optim.sgd(customParameterModule, { lr: 0.01 });
customParameterOptimizer.zero_grad();
const customAutogradWeight: NnParameter<readonly [1]> = nn.parameter("weight", tensor([0.5], [1] as const));
const customAutogradBias: NnParameter<readonly [1]> = nn.parameter("bias", tensor([0.1], [1] as const));
const customAutogradModule: CustomModule<readonly [1], readonly [1]> = nn.module({
  kind: "consumer-custom-autograd",
  parameters: [customAutogradWeight, customAutogradBias],
  forward(input: Tensor<readonly [1]>) {
    return input.mul(customAutogradWeight.tensor).add(customAutogradBias.tensor);
  },
});
const customAutogradOutput: Tensor<readonly [1]> = customAutogradModule.call(tensor([2], [1] as const));
const customAutogradOptimizer: Optimizer<"sgd"> = optim.sgd(customAutogradModule, { lr: 0.01 });
customAutogradOptimizer.zero_grad();
const customAutogradState: ModuleStateSnapshot = customAutogradModule.stateDict("trained");
const customAutogradCloneWeight: NnParameter<readonly [1]> = nn.parameter("weight", tensor([0], [1] as const));
const customAutogradCloneBias: NnParameter<readonly [1]> = nn.parameter("bias", tensor([0], [1] as const));
const customAutogradClone: CustomModule<readonly [1], readonly [1]> = nn.module({
  kind: "consumer-custom-autograd-clone",
  parameters: [customAutogradCloneWeight, customAutogradCloneBias],
  forward(input: Tensor<readonly [1]>) {
    return input.mul(customAutogradCloneWeight.tensor).add(customAutogradCloneBias.tensor);
  },
});
const customAutogradValidatedClone: typeof customAutogradClone = customAutogradClone.loadStateDict(customAutogradState, {
  strict: true,
  prefix: "trained",
  validateOnly: true,
});
const customAutogradLoadedClone: typeof customAutogradClone = customAutogradClone.load_state_dict(customAutogradState, {
  strict: true,
  prefix: "trained",
});
const callOutput: Tensor<readonly [1]> = linear.call(tensor([1, 2], [2] as const));
const dunderCallOutput: Tensor<readonly [1]> = linear.__call__(tensor([1, 2], [2] as const));
const namespaceCallOutput: Tensor<readonly [1]> = nn.call(linear, tensor([1, 2], [2] as const));
const namespaceDunderCallOutput: Tensor<readonly [1]> = nn.__call__(linear, tensor([1, 2], [2] as const));
const program: Program<readonly [2], readonly [1]> = linear.compile({ backend: "cpu" });
const session: Session<readonly [2], readonly [1]> = program.bindModule(linear);
const output: Tensor<readonly [1]> = gradMode.inferenceMode(() => session.stepTensor(tensor([1, 2], [2] as const)));
const executeIntoOutput = new Float32Array(1);
const executeIntoHotParams = { input: tensor([1, 2], [2] as const), output: executeIntoOutput };
const executeIntoHotCompatibility: SessionStepParamsCompatibility = session.requireHotStepParams(executeIntoHotParams);
const executeIntoHotPlan: SessionExecutionPlan<readonly [2], readonly [1]> = session.hotPathPlan(executeIntoHotParams);
const executeIntoResult: Float32Array = gradMode.inferenceMode(() => session.executeInto(executeIntoOutput, { input: tensor([1, 2], [2] as const) }));
const linearClone = nn.linear(2, 1);
const linearCloneState: ModuleStateSnapshot = linear.stateDict("compiled");
const linearLoadedClone: typeof linearClone = linearClone.loadStateDict(linearCloneState, {
  strict: true,
  prefix: "compiled",
});
const linearCloneSession: Session<readonly [2], readonly [1]> = program.bindModule(linearLoadedClone);
const linearCloneOutput: Tensor<readonly [1]> = gradMode.noGrad(() => linearCloneSession.stepTensor(tensor([1, 2], [2] as const)));

const bunLinear = bunNn.linear(2, 1);
const bunProgram: BunProgram<readonly [2], readonly [1]> = bunLinear.compile({ backend: "cpu" });
const bunSession: BunSession<readonly [2], readonly [1]> = bunProgram.bindModule(bunLinear);
const bunOutput = bunGradMode.inferenceMode(() => bunSession.stepTensor(bunTensor([1, 2], [2] as const)));
const bunExecuteIntoOutput = new Float32Array(1);
const bunExecuteIntoHotParams = { input: bunTensor([1, 2], [2] as const), output: bunExecuteIntoOutput };
const bunExecuteIntoHotCompatibility: BunSessionStepParamsCompatibility = bunSession.requireHotStepParams(bunExecuteIntoHotParams);
const bunExecuteIntoHotPlan: BunSessionExecutionPlan<readonly [2], readonly [1]> = bunSession.hotPathPlan(bunExecuteIntoHotParams);
const bunExecuteIntoResult: Float32Array = bunGradMode.inferenceMode(() => bunSession.executeInto(bunExecuteIntoOutput, { input: bunTensor([1, 2], [2] as const) }));
const typedBrowserFrontendSource: "ts" = browserFrontendManifest.source;
const typedBrowserProductSourceOfTruth: "ts-only" = browserFrontendManifest.productSourceOfTruth;
const typedBrowserNativeProductPolicy: "forbidden" = browserEntryManifest.nativeProductPolicy;
const typedBrowserNativeLoader: false = browserEntryManifest.nativeLoader;
const typedBrowserAdapterRole: "browser-safe-frontend" = browserEntryManifest.adapterRole;
const typedBrowserNativeApiContract: typeof browserNativeApiContract = browserNativeApiContract;
const typedBrowserTensorValue = browserTensor([1, 2], [2] as const);
const typedBrowserLinearValue = browserNn.linear(2, 1);
const typedBrowserLossModule = browserLoss.mseLoss();
const typedBrowserGradModeResult = browserGradMode.noGrad(() => typedBrowserTensorValue);
const typedBrowserOptimNamespace: typeof browserOptim = browserOptim;
const typedBrowserTrainNamespace: typeof browserTrain = browserTrain;

void fit;
void helperFit;
void classifierFit;
void typedClassifierEvaluation;
void typedClassifierPrediction;
void typedStepEvidence;
void typedStepClipNormApplied;
void typedStepClipValueApplied;
void typedStepGradNormBeforeClip;
void typedStepGradNormAfterClip;
void typedSchedulerState;
void typedCheckpointJson;
void typedParsedCheckpoint;
void typedCheckpointRestoreTargets;
void typedFreezeTarget;
void typedUnfreezeTarget;
void typedCheckpointCompiledModel;
void typedCompiledCheckpoint;
void typedCompiledCheckpointRestored;
void typedRestoredCompiledProgram;
void typedRestoredCompiledSession;
void typedRestoredCompiledOutput;
void customModuleOutput;
void customParameterState;
void customParameterOptimizer;
void customAutogradOutput;
void customAutogradOptimizer;
void customAutogradValidatedClone;
void customAutogradLoadedClone;
void callOutput;
void dunderCallOutput;
void namespaceCallOutput;
void namespaceDunderCallOutput;
void linearClone;
void linearCloneState;
void linearLoadedClone;
void linearCloneSession;
void linearCloneOutput;
void output;
void executeIntoHotCompatibility;
void executeIntoHotPlan;
void executeIntoResult;
void bunOutput;
void bunExecuteIntoHotCompatibility;
void bunExecuteIntoHotPlan;
void bunExecuteIntoResult;
void typedBrowserFrontendSource;
void typedBrowserProductSourceOfTruth;
void typedBrowserNativeProductPolicy;
void typedBrowserNativeLoader;
void typedBrowserAdapterRole;
void typedBrowserNativeApiContract;
void typedBrowserTensorValue;
void typedBrowserLinearValue;
void typedBrowserLossModule;
void typedBrowserGradModeResult;
void typedBrowserOptimNamespace;
void typedBrowserTrainNamespace;
`;

const consumerTsconfig = {
  compilerOptions: {
    noEmit: true,
    strict: true,
    target: "ES2022",
    module: "NodeNext",
    moduleResolution: "NodeNext",
    lib: ["ES2022"],
    types: [],
  },
  files: ["consumer_types.ts"],
};

try {
  run("npm", ["run", "build:package"]);
  run("npm", ["run", "check:package-artifact"]);

  const packOutput = execFileSync("npm", ["pack", "--json", "--pack-destination", tempRoot], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  const [pack] = JSON.parse(packOutput);
  const tarball = join(tempRoot, pack.filename);
  const project = join(tempRoot, "consumer");

  mkdirSync(project, { recursive: true });
  writeFileSync(join(project, "package.json"), JSON.stringify({
    private: true,
    type: "commonjs",
    dependencies: {
      zgml: tarball,
    },
  }, null, 2));
  writeFileSync(join(project, "consumer_node.cjs"), consumerNodeSmoke);
  writeFileSync(join(project, "consumer_bun.ts"), consumerBunSmoke);
  writeFileSync(join(project, "consumer_browser.cjs"), consumerBrowserSmoke);
  writeFileSync(join(project, "consumer_types.ts"), consumerTypeSmoke);
  writeFileSync(join(project, "tsconfig.consumer.json"), JSON.stringify(consumerTsconfig, null, 2));

  run("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund"], { cwd: project });
  const installedPackageRoot = join(project, "node_modules", "zgml");
  assertPackedPackageScripts(installedPackageRoot);
  run("npm", ["run", "--prefix", "node_modules/zgml", "build:native"], { cwd: project });
  run("node", ["consumer_node.cjs"], { cwd: project });
  run("node", ["consumer_browser.cjs"], { cwd: project });
  run("npm", ["run", "--prefix", "node_modules/zgml", "smoke:node"], { cwd: project });
  if (commandExists("bun")) {
    run("bun", ["consumer_bun.ts"], { cwd: project });
    run("npm", ["run", "--prefix", "node_modules/zgml", "smoke:bun:dist"], { cwd: project });
  } else {
    console.log("zgml packed install bun checks skipped: bun not found");
  }
  run(process.execPath, [tscBin, "-p", "tsconfig.consumer.json"], { cwd: project });
  run(process.execPath, [tscBin, "-p", "node_modules/zgml/tsconfig.types.json"], { cwd: project });
  run(process.execPath, [tscBin, "-p", "node_modules/zgml/tsconfig.dist.json"], { cwd: project });

  console.log(`zgml packed install ok: ${pack.filename}`);
} finally {
  if (process.env.ZGML_KEEP_PACKED_INSTALL_TMP !== "1") {
    rmSync(tempRoot, { recursive: true, force: true });
  } else {
    console.log(`zgml packed install temp kept: ${tempRoot}`);
  }
}
