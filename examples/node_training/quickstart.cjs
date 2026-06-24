"use strict";

const { zgml } = require("../..");

function scalar(tensor) {
  return typeof tensor.item === "function" ? tensor.item() : tensor.data[0];
}

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

const model = new zgml.nn.Sequential(
  new zgml.nn.Linear(2, 4, {
    weights: [
      0.25, -0.2, 0.1, 0.05,
      -0.1, 0.15, 0.2, -0.25,
    ],
    bias: [0.02, -0.01, 0.03, 0],
  }),
  new zgml.nn.ReLU(),
  new zgml.nn.Linear(4, 1, {
    weights: [0.1, -0.15, 0.05, 0.2],
    bias: [0],
  }),
);

const dataset = new zgml.utils.data.TensorDataset(
  zgml.tensor([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
    0.5, -0.5,
    -0.5, 0.5,
  ], [6, 2]),
  zgml.tensor([
    0,
    -2,
    2,
    0,
    1,
    -1,
  ], [6, 1]),
);

const loader = new zgml.utils.data.DataLoader(dataset, {
  batch_size: 2,
  shuffle: true,
  seed: 23,
});
const optimizer = new zgml.optim.AdamW(model, { lr: 0.04, weight_decay: 0.0001 });
const criterion = new zgml.nn.MSELoss();
const probe = zgml.tensor([1, -1], [2]);
const target = zgml.tensor([2], [1]);
const before = scalar(criterion.forward(model.forward(probe), target));

const fit = zgml.train.fitModule(optimizer, model, loader, criterion, {
  epochs: 80,
  zero_grad: true,
});
const after = scalar(criterion.forward(model.forward(probe), target));
if (!zgml.train.isTrainFitEvidence(fit) || !(after < before * 0.02)) {
  throw new Error(`expected training to reduce loss sharply; before=${before}, after=${after}`);
}

const snapshot = zgml.checkpoint.create({ model, optimizer, prefix: "quickstart" });
const text = zgml.checkpoint.stringify(snapshot, 2);
const restored = new zgml.nn.Sequential(
  new zgml.nn.Linear(2, 4),
  new zgml.nn.ReLU(),
  new zgml.nn.Linear(4, 1),
);
const restoredOptimizer = new zgml.optim.AdamW(restored, { lr: 0.04, weight_decay: 0.0001 });
zgml.checkpoint.restore(zgml.checkpoint.parse(text), {
  model: restored,
  optimizer: restoredOptimizer,
  prefix: "quickstart",
  strict: true,
});
assertClose(scalar(restored.forward(probe)), scalar(model.forward(probe)), 1e-5, "restored prediction");

const fast = zgml.compileForInference(restored, { backend: "cpu", inputShape: [2] });
const compiled = fast.forward(probe);
const output = new Float32Array(1);
const hotParams = { input: probe, output };
const compatibility = fast.session.requireHotStepParams(hotParams);
const compiledInto = fast.into(output, probe);
if (
  compiledInto !== output ||
  compatibility.hotPath !== true ||
  compatibility.runtimeOutputAllocationFree !== true ||
  compatibility.readbackFree !== true
) {
  throw new Error(`expected quickstart executeInto to be allocation-free: ${JSON.stringify(compatibility)}`);
}
assertClose(scalar(compiled), scalar(model.forward(probe)), 1e-5, "compiled prediction");
assertClose(output[0], scalar(model.forward(probe)), 1e-5, "executeInto prediction");

fast.dispose();

console.log(`zgml quickstart ok: before=${before.toFixed(6)} after=${after.toFixed(6)} compiled=${output[0].toFixed(6)}`);
