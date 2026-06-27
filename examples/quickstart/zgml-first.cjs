"use strict";

const { zgml } = require("../..");

function scalar(tensor) {
  return typeof tensor.item === "function" ? tensor.item() : tensor.data[0];
}

function createModel(initialized = false) {
  return new zgml.nn.Sequential(
    new zgml.nn.Linear(2, 4, initialized ? {
      weights: [
        0.25, -0.2, 0.1, 0.05,
        -0.1, 0.15, 0.2, -0.25,
      ],
      bias: [0.02, -0.01, 0.03, 0],
    } : undefined),
    new zgml.nn.ReLU(),
    new zgml.nn.Linear(4, 1, initialized ? {
      weights: [0.1, -0.15, 0.05, 0.2],
      bias: [0],
    } : undefined),
  );
}

const model = createModel(true);
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
const loss = new zgml.nn.MSELoss();
const probe = zgml.tensor([1, -1], [2]);
const target = zgml.tensor([2], [1]);

const before = scalar(loss.forward(model.forward(probe), target));
model.fit(loader, {
  optimizer,
  loss,
  epochs: 80,
  zero_grad: true,
});
const after = scalar(loss.forward(model.forward(probe), target));
if (!(after < before * 0.02)) {
  throw new Error(`expected training to reduce loss; before=${before}, after=${after}`);
}

const snapshot = zgml.checkpoint.create({ model, optimizer, prefix: "quickstart" });
const restored = createModel();
const restoredOptimizer = new zgml.optim.AdamW(restored, { lr: 0.04, weight_decay: 0.0001 });
zgml.checkpoint.restore(snapshot, {
  model: restored,
  optimizer: restoredOptimizer,
  prefix: "quickstart",
  strict: true,
});

const fast = zgml.native(restored, { backend: "cpu", inputShape: [2] });
const output = new Float32Array(1);
const fastTensor = fast.forward(probe);
fast.into(output, probe);
const proof = fast.explain();
const executionPlan = fast.requireExecutionPlan();
const contract = fast.session.stepContract();
const inspection = fast.program.inspect();
const eagerPrediction = scalar(restored.forward(probe));
fast.dispose();
if (
  fast.native !== true ||
  proof.supported !== true ||
  proof.nativePath !== "device-program" ||
  executionPlan.canExecute !== true ||
  executionPlan.executionMode !== "executable" ||
  inspection.executionSupported !== true ||
  inspection.backend !== "cpu" ||
  contract.inputShape.join("x") !== "2" ||
  contract.outputShape.join("x") !== "1"
) {
  throw new Error(`zgml.native did not produce a native Program/Session path: ${JSON.stringify({ proof, executionPlan, inspection, contract })}`);
}
if (Math.abs(output[0] - eagerPrediction) > 1e-5 || Math.abs(scalar(fastTensor) - eagerPrediction) > 1e-5) {
  throw new Error(`compiled prediction drifted from eager output; eager=${eagerPrediction}, compiled=${output[0]}`);
}

console.log(
  `zgml quickstart: before=${before.toFixed(6)} after=${after.toFixed(6)} ` +
  `compiled=${proof.supported} input=${proof.inputShape.join("x")} output=${proof.outputShape.join("x")} ` +
  `prediction=${output[0].toFixed(6)}`,
);
