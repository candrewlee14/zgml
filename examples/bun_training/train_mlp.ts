import {
  checkpoint,
  data,
  loss,
  nn,
  optim,
  tensor,
  train,
  type Tensor,
} from "zgml/bun";

function scalar(t: { item?: () => number; data: Float32Array }): number {
  return typeof t.item === "function" ? t.item() : t.data[0] ?? Number.NaN;
}

function assertClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

function assertHotRuntimeProfile(target: Pick<ReturnType<ReturnType<typeof createModel>["compile"]>, "requireHotRuntimeProfile" | "runtimeProfile">, label: string): void {
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

function createModel() {
  class TinyMlp extends nn.Module<readonly [2], readonly [1]> {
    readonly fc1 = new nn.Linear(2, 4, {
      weights: [
        0.25, -0.2, 0.1, 0.05,
        -0.1, 0.15, 0.2, -0.25,
      ],
      bias: [0.02, -0.01, 0.03, 0],
    });
    readonly act = new nn.ReLU();
    readonly fc2 = new nn.Linear(4, 1, {
      weights: [0.1, -0.15, 0.05, 0.2],
      bias: [0],
    });

    constructor() {
      super({ kind: "tiny-mlp" });
    }

    forward(input: Tensor<readonly [2]>): Tensor<readonly [1]> {
      return this.fc2.forward(this.act.forward(this.fc1.forward(input)));
    }
  }
  return new TinyMlp();
}

const model = createModel();
const moduleNames = model.namedModules("mlp").map((entry) => entry.name).join("|");
const childNames = model.namedChildren("mlp").map((entry) => entry.name).join("|");
if (moduleNames !== "mlp|mlp.fc1|mlp.act|mlp.fc2" || childNames !== "mlp.fc1|mlp.act|mlp.fc2") {
  throw new Error(`unexpected MLP module traversal: modules=${moduleNames} children=${childNames}`);
}
if (model.children().length !== 3 || model.act.kind !== "relu") {
  throw new Error("subclassed nn.Module should expose assigned layers as PyTorch-like children");
}

const samples = data.tensorDataset(
  tensor([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
    0.5, -0.5,
    -0.5, 0.5,
  ], [6, 2] as const),
  tensor([
    0,
    -2,
    2,
    0,
    1,
    -1,
  ], [6, 1] as const),
);

const loader = data.dataLoader(samples, { batchSize: 2, shuffle: true, seed: 23 });
const optimizer = optim.adamW(model, { lr: 0.04, weightDecay: 0.0001 });
const probeInput = tensor([1, -1], [2] as const);
const probeTarget = tensor([2], [1] as const);
const before = scalar(loss.mse(model.forward(probeInput), probeTarget));

const fit = train.fit(optimizer, loader, (batch) => {
  if (!batch.target) throw new Error("MLP training batch requires targets");
  return loss.mse(model.forward(batch.input), batch.target);
}, {
  epochs: 80,
  zeroGrad: true,
});

const after = scalar(loss.mse(model.forward(probeInput), probeTarget));
if (!train.isTrainFitEvidence(fit) || fit.steps !== 240 || fit.losses.length !== 240) {
  throw new Error("train.fit must return AdamW fit evidence for each MLP optimizer step");
}
if (!(after < before * 0.02)) {
  throw new Error(`expected MLP training to reduce probe loss sharply; before=${before}, after=${after}`);
}

const state = model.stateDict("mlp");
model.loadStateDict(state, { prefix: "mlp", strict: true, validateOnly: true });
const optimizerState = optimizer.stateDict();
optimizer.loadStateDict(optimizerState, { strict: true, validateOnly: true });

const snapshot = checkpoint.create({ model, optimizer, prefix: "mlp" });
const inspection = checkpoint.inspect(snapshot);
if (!inspection.hasModel || !inspection.hasOptimizer || inspection.modelParameterCount !== 4) {
  throw new Error(`unexpected MLP checkpoint inspection: ${JSON.stringify(inspection)}`);
}

const restored = createModel();
const restoredOptimizer = optim.adamW(restored, { lr: 0.04, weightDecay: 0.0001 });
checkpoint.restore(snapshot, { model: restored, optimizer: restoredOptimizer, prefix: "mlp", strict: true });
assertClose(scalar(restored.forward(probeInput)), scalar(model.forward(probeInput)), 1e-5, "restored MLP eager prediction");

const support = model.compileSupport({ inputShape: [2, 2] as const });
if (!support || support.supported !== true) {
  throw new Error(`expected Sequential MLP compile support evidence, got ${JSON.stringify(support)}`);
}

const scalarSupport = model.compileSupport({ inputShape: [2] as const });
if (!scalarSupport || scalarSupport.supported !== true) {
  throw new Error(`expected scalar-input Sequential MLP compile support evidence, got ${JSON.stringify(scalarSupport)}`);
}
const program = model.compile({ backend: "cpu", inputShape: [2] as const });
assertHotRuntimeProfile(program, "trained MLP Program");
const session = program.bindModule(model);
const compiled = session.stepTensor(probeInput);
const compiledOut = new Float32Array(1);
const hotParams = { input: probeInput, output: compiledOut };
const hotCompatibility = session.requireHotStepParams(hotParams);
const hotPlan = session.hotPathPlan(hotParams);
const compiledInto = session.executeInto(compiledOut, { input: probeInput });
assertHotRuntimeProfile(session, "trained MLP Session");
if (
  compiledInto !== compiledOut ||
  hotCompatibility.hotPath !== true ||
  hotCompatibility.runtimeOutputAllocationFree !== true ||
  hotCompatibility.readbackFree !== true ||
  hotPlan.hotPath !== true ||
  hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature
) {
  throw new Error(`expected trained MLP executeInto to prove allocation-free hot path: ${JSON.stringify({ hotCompatibility, hotPlan })}`);
}
assertClose(scalar(compiled), scalar(model.forward(probeInput)), 1e-5, "compiled trained MLP prediction");
assertClose(compiledInto[0] ?? Number.NaN, scalar(model.forward(probeInput)), 1e-5, "executeInto trained MLP prediction");
if (program.inputShape().join("x") !== "2" || program.outputShape().join("x") !== "1") {
  throw new Error(`unexpected trained MLP Program shape: input=${program.inputShape()} output=${program.outputShape()}`);
}
session.free();
program.free();

const restoredProgram = restored.compile({ backend: "cpu", inputShape: [2] as const });
assertHotRuntimeProfile(restoredProgram, "restored MLP Program");
const restoredSession = restoredProgram.bindModule(restored);
const restoredCompiled = restoredSession.stepTensor(probeInput);
const restoredCompiledOut = new Float32Array(1);
const restoredHotParams = { input: probeInput, output: restoredCompiledOut };
const restoredHotCompatibility = restoredSession.requireHotStepParams(restoredHotParams);
const restoredHotPlan = restoredSession.hotPathPlan(restoredHotParams);
const restoredCompiledInto = restoredSession.executeInto(restoredCompiledOut, { input: probeInput });
assertHotRuntimeProfile(restoredSession, "restored MLP Session");
if (
  restoredCompiledInto !== restoredCompiledOut ||
  restoredHotCompatibility.hotPath !== true ||
  restoredHotCompatibility.runtimeOutputAllocationFree !== true ||
  restoredHotCompatibility.readbackFree !== true ||
  restoredHotPlan.hotPath !== true ||
  restoredHotPlan.stepParamsSignature !== restoredHotCompatibility.stepParamsSignature
) {
  throw new Error(`expected restored MLP executeInto to prove allocation-free hot path: ${JSON.stringify({ restoredHotCompatibility, restoredHotPlan })}`);
}
assertClose(scalar(restoredCompiled), scalar(model.forward(probeInput)), 1e-5, "compiled restored MLP prediction");
assertClose(restoredCompiledInto[0] ?? Number.NaN, scalar(model.forward(probeInput)), 1e-5, "executeInto restored MLP prediction");
restoredSession.free();
restoredProgram.free();

console.log(`zgml bun MLP training smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} steps=${fit.steps} compiled=${scalar(compiled).toFixed(6)} restoredCompiled=${scalar(restoredCompiled).toFixed(6)}`);
