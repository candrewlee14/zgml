"use strict";

const { simple } = require("../..");

const {
  checkpoint,
  compile,
  data,
  loss,
  nn,
  optim,
  tensor,
  train,
} = simple;

function scalar(t) {
  return typeof t.item === "function" ? t.item() : t.data[0];
}

function assertArrayClose(actual, expected, tolerance, label) {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: expected length ${expected.length}, got ${actual.length}`);
  }
  for (let i = 0; i < expected.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > tolerance) {
      throw new Error(`${label}[${i}]: expected ${expected[i]} +/- ${tolerance}, got ${actual[i]}`);
    }
  }
}

function createModel() {
  return new nn.Sequential([
    nn.embedding(4, 3, {
      weight: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
      ],
    }),
    nn.linear(3, 2, {
      weights: [
        0.2, -0.1,
        -0.1, 0.2,
        0.1, 0.1,
      ],
      bias: [0, 0],
    }),
    nn.logSoftmax(-1),
  ]);
}

const model = createModel();
const probeTokens = tensor([0, 1], [2]);
const probeTargets = [0, 1];
const criterion = loss.nllLossModule({ classes: 2 });
const before = scalar(criterion.call(model.forward(probeTokens), probeTargets));
const dataset = data.tensorDataset(
  tensor([0, 1, 2, 3], [4]),
  tensor([0, 1, 1, 0], [4]),
);
const loader = data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 5 });
const optimizer = optim.adamW(model, { lr: 0.05, weightDecay: 0 });

const fit = train.fit(optimizer, loader, (batch) => {
  if (!batch.target) throw new Error("token classifier batch requires class targets");
  return criterion.call(model.forward(batch.input), Array.from(batch.target.data));
}, {
  epochs: 80,
  zeroGrad: true,
});

const eager = model.forward(probeTokens);
const after = scalar(criterion.call(eager, probeTargets));
const classes = Array.from(train.classPredictions(eager, { classes: 2 }));
if (!train.isTrainFitEvidence(fit) || fit.steps !== 160 || fit.losses.length !== 160) {
  throw new Error("token classifier example expected two optimizer steps per epoch");
}
if (!(after < before * 0.001) || classes.join(",") !== "0,1") {
  throw new Error(`token classifier example expected learned classes 0,1; before=${before}, after=${after}, classes=${classes.join(",")}`);
}

const snapshot = checkpoint.create({ model, optimizer, prefix: "token" });
const restored = createModel();
const restoredOptimizer = optim.adamW(restored, { lr: 0.05, weightDecay: 0 });
checkpoint.restore(snapshot, { model: restored, optimizer: restoredOptimizer, prefix: "token", strict: true });
assertArrayClose(restored.forward(probeTokens).data, eager.data, 1e-5, "restored token classifier logits");

const support = model.compileSupport({ inputShape: [2], backend: "cpu" });
if (
  !support ||
  support.supported !== true ||
  support.outputShape?.join("x") !== "2x2" ||
  support.kernelPlan?.ops?.map((op) => op.op).join("|") !== "embedding|linear|logSoftmax"
) {
  throw new Error(`token classifier example expected Embedding/Linear/LogSoftmax native support, got ${JSON.stringify(support)}`);
}

const fast = compile.compileForInference(model, { inputShape: [2], backend: "cpu" });
const compiled = fast.into(new Float32Array(4), probeTokens);
assertArrayClose(compiled, eager.data, 1e-5, "compiled token classifier logits");
fast.dispose();

console.log(`zgml token classifier smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} steps=${fit.steps} classes=${classes.join(",")}`);
