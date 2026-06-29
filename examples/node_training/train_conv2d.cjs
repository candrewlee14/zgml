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

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

function assertArrayClose(actual, expected, tolerance, label) {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: expected length ${expected.length}, got ${actual.length}`);
  }
  for (let i = 0; i < expected.length; i += 1) {
    assertClose(actual[i], expected[i], tolerance, `${label}[${i}]`);
  }
}

function createModel() {
  return new nn.Sequential([
    nn.conv2d(1, 1, 2, {
      weight: [0.2, 0.1, 0.1, 0.2],
      bias: [0],
    }),
    nn.relu(),
  ]);
}

const model = createModel();
const image = tensor([
  1, 0, 1,
  1, 0, 1,
  1, 1, 0,
], [1, 3, 3]);
const target = tensor([1, 1, 1, 1], [1, 2, 2]);
const dataset = data.tensorDataset(
  tensor([
    1, 0, 1,
    1, 0, 1,
    1, 1, 0,
  ], [1, 1, 3, 3]),
  tensor([1, 1, 1, 1], [1, 1, 2, 2]),
);
const loader = data.dataLoader(dataset, { batchSize: 1, shuffle: false });
const optimizer = optim.sgd(model, { lr: 0.05 });
const before = scalar(loss.mse(model.forward(image), target));

const fit = train.fit(optimizer, loader, (batch) => {
  if (!batch.target) throw new Error("Conv2d training batch requires feature targets");
  return loss.mse(model.forward(batch.input), batch.target);
}, {
  epochs: 50,
  zeroGrad: true,
});

const eager = model.forward(image);
const after = scalar(loss.mse(eager, target));
if (!train.isTrainFitEvidence(fit) || fit.steps !== 50 || fit.losses.length !== 50) {
  throw new Error("Conv2d example expected one optimizer step per epoch");
}
if (!(after < before * 0.03)) {
  throw new Error(`Conv2d example expected training to reduce feature loss; before=${before}, after=${after}`);
}

const state = model.stateDict("conv");
model.loadStateDict(state, { prefix: "conv", strict: true, validateOnly: true });
const snapshot = checkpoint.create({ model, optimizer, prefix: "conv" });
const restored = createModel();
const restoredOptimizer = optim.sgd(restored, { lr: 0.05 });
checkpoint.restore(snapshot, { model: restored, optimizer: restoredOptimizer, prefix: "conv", strict: true });
assertArrayClose(restored.forward(image).data, eager.data, 1e-5, "restored Conv2d eager output");

const support = model.compileSupport({ inputShape: [1, 3, 3], backend: "cpu" });
if (
  !support ||
  support.supported !== true ||
  support.outputShape?.join("x") !== "1x2x2" ||
  support.kernelPlan?.ops?.[0]?.op !== "conv2d"
) {
  throw new Error(`Conv2d example expected native compile support, got ${JSON.stringify(support)}`);
}

const fast = compile.compileForInference(model, { inputShape: [1, 3, 3], backend: "cpu" });
const compiled = fast.into(new Float32Array(4), image);
assertArrayClose(compiled, eager.data, 1e-5, "compiled Conv2d output");
fast.dispose();

console.log(`zgml Conv2d training smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} steps=${fit.steps} compiled=${Array.from(compiled).map((v) => v.toFixed(4)).join(",")}`);
