"use strict";

const { mkdtempSync, readFileSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { torch } = require("../..");

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

function assertThrows(fn, expected, label) {
  try {
    fn();
  } catch (error) {
    if (String(error && error.message ? error.message : error).includes(expected)) return;
    throw new Error(`${label}: expected error containing ${expected}, got ${error && error.message ? error.message : error}`);
  }
  throw new Error(`${label}: expected function to throw`);
}

function createModel() {
  return new torch.nn.Sequential(
    new torch.nn.Linear(2, 4, {
      weights: [
        0.25, -0.2, 0.1, 0.05,
        -0.1, 0.15, 0.2, -0.25,
      ],
      bias: [0.02, -0.01, 0.03, 0],
    }),
    new torch.nn.ReLU(),
    new torch.nn.Linear(4, 1, {
      weights: [0.1, -0.15, 0.05, 0.2],
      bias: [0],
    }),
  );
}

torch.manual_seed(123);

const model = createModel();
const dataset = new torch.utils.data.TensorDataset(
  torch.tensor([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
    0.5, -0.5,
    -0.5, 0.5,
  ], [6, 2]),
  torch.tensor([
    0,
    -2,
    2,
    0,
    1,
    -1,
  ], [6, 1]),
);
const loader = new torch.utils.data.DataLoader(dataset, { batch_size: 2, shuffle: true, seed: 23 });
const optimizer = new torch.optim.AdamW(model, { lr: 0.04, weight_decay: 0.0001 });
const criterion = new torch.nn.MSELoss();
const probe = torch.tensor([1, -1], [2]);
const target = torch.scalar(2);
const before = scalar(criterion.call(model.forward(probe), target));

const fit = torch.train.fitModule(optimizer, model, loader, criterion, {
  epochs: 80,
  zero_grad: true,
});
const after = scalar(criterion.call(model.forward(probe), target));
if (!torch.train.isTrainFitEvidence(fit) || fit.steps !== 240 || fit.losses.length !== 240) {
  throw new Error("torch.train.fitModule must return fit evidence for each optimizer step");
}
if (!(after < before * 0.02)) {
  throw new Error(`expected torch quickstart training to reduce loss sharply; before=${before}, after=${after}`);
}
const inferencePrediction = torch.no_grad(() => model.forward(probe));
assertClose(scalar(inferencePrediction), scalar(model.forward(probe)), 1e-5, "torch.no_grad quickstart prediction");

const snapshotText = torch.save(torch.checkpoint.create({ model, optimizer, prefix: "quickstart" }), 2);
const loadedSnapshot = torch.load(snapshotText);
const restored = createModel();
const restoredOptimizer = new torch.optim.AdamW(restored, { lr: 0.04, weight_decay: 0.0001 });
torch.checkpoint.restore(loadedSnapshot, {
  model: restored,
  optimizer: restoredOptimizer,
  prefix: "quickstart",
  strict: true,
});
assertClose(scalar(restored.forward(probe)), scalar(model.forward(probe)), 1e-5, "restored torch quickstart prediction");
const aliasRestored = createModel();
const aliasRestoredOptimizer = new torch.optim.AdamW(aliasRestored, { lr: 0.04, weight_decay: 0.0001 });
const aliasTargets = torch.load(snapshotText, {
  model: aliasRestored,
  optimizer: aliasRestoredOptimizer,
  prefix: "quickstart",
  strict: true,
});
if (aliasTargets.model !== aliasRestored || aliasTargets.optimizer !== aliasRestoredOptimizer) {
  throw new Error("torch.load should return checkpoint restore targets when targets are supplied");
}
assertClose(scalar(aliasRestored.forward(probe)), scalar(model.forward(probe)), 1e-5, "torch.load restored quickstart prediction");

const checkpointDir = mkdtempSync(join(tmpdir(), "zgml-quickstart-"));
try {
  const checkpointPath = join(checkpointDir, "quickstart.zgml");
  const savedPath = torch.save(loadedSnapshot, checkpointPath, 2);
  if (savedPath !== checkpointPath || !readFileSync(checkpointPath, "utf8").includes("\"format\": \"zgml.checkpoint\"")) {
    throw new Error("torch.save(snapshot, path) should write a zgml checkpoint file and return the path");
  }
  const pathLoadedSnapshot = torch.load(checkpointPath);
  if (torch.checkpoint.inspect(pathLoadedSnapshot).signature !== torch.checkpoint.inspect(loadedSnapshot).signature) {
    throw new Error("torch.load(path) should load the checkpoint file written by torch.save");
  }
  const pathRestored = createModel();
  const pathRestoredOptimizer = new torch.optim.AdamW(pathRestored, { lr: 0.04, weight_decay: 0.0001 });
  const pathTargets = torch.load(checkpointPath, {
    model: pathRestored,
    optimizer: pathRestoredOptimizer,
    prefix: "quickstart",
    strict: true,
  });
  if (pathTargets.model !== pathRestored || pathTargets.optimizer !== pathRestoredOptimizer) {
    throw new Error("torch.load(path, targets) should restore checkpoint file state and return targets");
  }
  assertClose(scalar(pathRestored.forward(probe)), scalar(model.forward(probe)), 1e-5, "torch.load path restored quickstart prediction");
} finally {
  rmSync(checkpointDir, { recursive: true, force: true });
}

const lazy = torch.lazy.input([2]).linear(4).relu().linear(1);
const lazyEmbeddingGraph = torch.lazy.input([2]).embedding(4, 3).layerNorm(3).rmsNorm(3);
const lazyNamespaceEmbeddingGraph = torch.lazy.rmsNorm(torch.lazy.layerNorm(torch.lazy.embedding(torch.lazy.input([2]), 4, 3), 3), 3);
const lazySnakeNamespaceEmbeddingGraph = torch.lazy.rms_norm(torch.lazy.layer_norm(torch.lazy.embedding(torch.lazy.input([2]), 4, 3), 3), 3);
const lazyGridEmbeddingGraph = torch.lazy.input([1, 2]).embedding(4, 3);
const lazyConvPoolGraph = torch.lazy.input([1, 4, 4]).conv2d(2, 1).maxPool2d(2).avgPool2d(2);
const lazyNamespaceConvPoolGraph = torch.lazy.avgPool2d(torch.lazy.maxPool2d(torch.lazy.conv2d(torch.lazy.input([1, 4, 4]), 2, 1), 2), 2);
const lazySnakeConvPoolGraph = torch.lazy.avg_pool2d(torch.lazy.max_pool2d(torch.lazy.input([1, 4, 4]).conv2d(2, 1), 2), 2);
const lazySigmoid = torch.lazy.input([2]).linear(3).sigmoid().linear(1);
const lazyNamespaceSigmoid = torch.lazy.sigmoid(torch.lazy.input([2]).linear(3)).linear(1);
const lazyActivationChain = torch.lazy.input([2]).exp().log().neg().recip().abs().sqrt().square().sgn().step();
const lazySnakeSoftmaxGraph = torch.lazy.input([2, 3]).log_softmax(1);
const lazyDropoutGraph = torch.lazy.input([2]).dropout(0.5, { training: false }).relu();
const lazyNamespaceDropoutGraph = torch.lazy.dropout(torch.lazy.input([2]), 0.5, { training: false }).relu();
const lazyTrainingDropoutGraph = torch.lazy.input([2]).dropout(0.5, { training: true });
const lazyReductionInput = torch.lazy.input([2, 3]);
const lazyReductionChain = lazyReductionInput.sum(1).mean(0).max(0).min(0);
const lazyShapeInput = torch.lazy.input([1, 3]);
const lazyShapeChain = lazyShapeInput.squeeze(0).unsqueeze(0).broadcastTo([2, 3]).narrow(0, 0, 1).slice(1, 0, 2).transpose(0, 1).permute([1, 0]).select(0, 0);
const lazySnakeShapeGraph = lazyShapeInput.squeeze(0).unsqueeze(0).broadcast_to([2, 3]);
const lazyEmbeddingModuleGraph = torch.lazy.fromModule(torch.nn.embedding(4, 3), { inputShape: [2] });
const lazyNormModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
  torch.nn.embedding(4, 3),
  torch.nn.layerNorm(3),
  torch.nn.rmsNorm(3),
]), { inputShape: [2] });
const lazyEvalDropoutModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
  torch.nn.dropout(0.5, { training: false }),
  torch.nn.relu(),
]), { inputShape: [2] });
const lazyConvPoolModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
  torch.nn.conv2d(1, 2, 1),
  torch.nn.max_pool2d(2),
  torch.nn.avg_pool2d(2),
]), { inputShape: [1, 4, 4] });
const lazyTrainingDropoutModuleGraph = torch.lazy.fromModule(torch.nn.dropout(0.5, { training: true }), { inputShape: [2] });
if (
  lazy.compileSupport().supported !== true ||
  lazyEmbeddingGraph.compileSupport().supported !== true ||
  lazyNamespaceEmbeddingGraph.compileSupport().supported !== true ||
  lazySnakeNamespaceEmbeddingGraph.compileSupport().supported !== true ||
  lazyGridEmbeddingGraph.compileSupport().supported !== false ||
  lazyConvPoolGraph.compileSupport().supported !== true ||
  lazyNamespaceConvPoolGraph.compileSupport().supported !== true ||
  lazySnakeConvPoolGraph.compileSupport().supported !== true ||
  lazySigmoid.compileSupport().supported !== true ||
  lazyNamespaceSigmoid.compileSupport().supported !== true ||
  lazyActivationChain.compileSupport().supported !== true ||
  lazySnakeSoftmaxGraph.compileSupport().supported !== true ||
  lazyDropoutGraph.compileSupport().supported !== true ||
  lazyNamespaceDropoutGraph.compileSupport().supported !== true ||
  lazyTrainingDropoutGraph.compileSupport().supported !== false ||
  torch.lazy.canCompile(lazy) !== true ||
  torch.lazy.can_compile(lazy) !== true ||
  torch.lazy.requireCompileSupport(lazy).supported !== true ||
  torch.lazy.require_compile_support(lazy).supported !== true ||
  torch.lazy.canCompile(lazyTrainingDropoutGraph) !== false ||
  torch.compile.canCompile(lazy) !== true ||
  torch.compile.can_compile(lazy) !== true ||
  torch.compile.compileSupport(lazy).supported !== true ||
  torch.compile.requireCompileSupport(lazy).supported !== true ||
  torch.compile.requireCompilePlan(lazy).supported !== true ||
  torch.compile.tensorProgramIr(lazy)?.kind !== "tensor-program-ir" ||
  torch.compile.kernelPlan(lazy)?.kind !== "native-module-kernel-plan" ||
  torch.compile.inputShape(lazy).join("x") !== "2" ||
  torch.compile.outputShape(lazy).join("x") !== "1" ||
  torch.compile.parameterLayout(lazy).parameters.length !== 4 ||
  lazyTrainingDropoutGraph.canCompile() !== false ||
  lazyTrainingDropoutGraph.can_compile() !== false ||
  lazyReductionChain.compileSupport().supported !== true ||
  lazyShapeChain.compileSupport().supported !== true ||
  lazySnakeShapeGraph.compileSupport().supported !== true ||
  lazyEmbeddingModuleGraph.compileSupport().supported !== true ||
  lazyNormModuleGraph.compileSupport().supported !== true ||
  lazyEvalDropoutModuleGraph.compileSupport().supported !== true ||
  lazyConvPoolModuleGraph.compileSupport().supported !== true ||
  lazyTrainingDropoutModuleGraph.compileSupport().supported !== false ||
  !String(lazyTrainingDropoutModuleGraph.compileSupport().reason).includes("deterministic no-op nn.Dropout") ||
  lazy.kernelPlan().kind !== "native-module-kernel-plan"
) {
  throw new Error("torch.lazy should expose compile-capable quickstart graph evidence");
}
assertThrows(() => lazyTrainingDropoutGraph.requireCompileSupport(), "lazy graph cannot compile", "lazy requireCompileSupport unsupported graph");
assertThrows(() => torch.lazy.require_compile_support(lazyTrainingDropoutGraph), "lazy graph cannot compile", "lazy namespace require_compile_support unsupported graph");
assertThrows(() => torch.compile.requireCompileSupport(lazyTrainingDropoutGraph), "compile.requireCompileSupport rejected unsupported target", "compile namespace lazy require unsupported graph");

const support = torch.compile.compileSupport(restored, { inputShape: [2], backend: "cpu" });
const plan = torch.compile.requireCompilePlan(restored, { inputShape: [2], backend: "cpu" });
if (support.supported !== true || plan.supported !== true || !torch.compile.canCompile(restored, { inputShape: [2], backend: "cpu" })) {
  throw new Error(`expected restored torch quickstart model to compile, got ${JSON.stringify({ support, plan })}`);
}

const program = torch.compile.compile(restored, { inputShape: [2], backend: "cpu" });
assertHotRuntimeProfile(program, "torch quickstart Program");
const directProgram = torch.compile(restored, { inputShape: [2], backend: "cpu" });
assertHotRuntimeProfile(directProgram, "torch quickstart direct Program");
directProgram.dispose();
const session = program.bindModule(restored);
try {
  const output = new Float32Array(1);
  const hotCompatibility = session.requireHotStepParams({ input: probe, output });
  const hotPlan = session.hotPathPlan({ input: probe, output });
  const compiledInto = session.executeInto(output, { input: probe });
  assertHotRuntimeProfile(session, "torch quickstart Session");
  if (
    compiledInto !== output ||
    hotCompatibility.hotPath !== true ||
    hotCompatibility.runtimeOutputAllocationFree !== true ||
    hotCompatibility.readbackFree !== true ||
    hotPlan.hotPath !== true ||
    hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature
  ) {
    throw new Error(`expected torch quickstart executeInto to prove allocation-free hot path: ${JSON.stringify({ hotCompatibility, hotPlan })}`);
  }
  assertClose(output[0], scalar(restored.forward(probe)), 1e-5, "compiled torch quickstart prediction");
} finally {
  session.dispose();
  program.dispose();
}

console.log(`zgml torch quickstart smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} steps=${fit.steps}`);
