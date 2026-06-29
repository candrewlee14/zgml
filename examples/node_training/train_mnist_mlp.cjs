"use strict";

const { createWriteStream, existsSync, mkdirSync, readFileSync } = require("node:fs");
const { get } = require("node:https");
const { join, resolve } = require("node:path");
const { gunzipSync } = require("node:zlib");

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

const cacheDir = resolve(__dirname, "..", "..", ".cache", "zgml-mnist");
const mnistBaseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist";
const files = {
  trainImages: "train-images-idx3-ubyte.gz",
  trainLabels: "train-labels-idx1-ubyte.gz",
  testImages: "t10k-images-idx3-ubyte.gz",
  testLabels: "t10k-labels-idx1-ubyte.gz",
};

const trainLimit = Number(process.env.ZGML_MNIST_TRAIN_LIMIT || "2048");
const testLimit = Number(process.env.ZGML_MNIST_TEST_LIMIT || "512");
const epochs = Number(process.env.ZGML_MNIST_EPOCHS || "3");
const batchSize = Number(process.env.ZGML_MNIST_BATCH_SIZE || "64");
const accuracyFloor = Number(process.env.ZGML_MNIST_ACCURACY_FLOOR || "0.70");

function requirePositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer, got ${value}`);
  }
}

requirePositiveInteger(trainLimit, "ZGML_MNIST_TRAIN_LIMIT");
requirePositiveInteger(testLimit, "ZGML_MNIST_TEST_LIMIT");
requirePositiveInteger(epochs, "ZGML_MNIST_EPOCHS");
requirePositiveInteger(batchSize, "ZGML_MNIST_BATCH_SIZE");

function download(url, destination) {
  return new Promise((resolvePromise, reject) => {
    const request = get(url, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        download(response.headers.location, destination).then(resolvePromise, reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`failed to download ${url}: HTTP ${response.statusCode}`));
        response.resume();
        return;
      }
      const output = createWriteStream(destination);
      response.pipe(output);
      output.on("finish", () => output.close(resolvePromise));
      output.on("error", reject);
    });
    request.on("error", reject);
  });
}

async function ensureMnist() {
  mkdirSync(cacheDir, { recursive: true });
  for (const name of Object.values(files)) {
    const path = join(cacheDir, name);
    if (!existsSync(path)) {
      await download(`${mnistBaseUrl}/${name}`, path);
    }
  }
}

function readU32BE(buffer, offset) {
  return buffer.readUInt32BE(offset);
}

function loadImages(path, limit) {
  const raw = gunzipSync(readFileSync(path));
  const magic = readU32BE(raw, 0);
  const count = readU32BE(raw, 4);
  const rows = readU32BE(raw, 8);
  const cols = readU32BE(raw, 12);
  if (magic !== 2051 || rows !== 28 || cols !== 28) {
    throw new Error(`unexpected MNIST image file header for ${path}`);
  }
  const n = Math.min(limit, count);
  const pixels = rows * cols;
  const values = new Float32Array(n * pixels);
  let source = 16;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < pixels; j += 1) {
      values[i * pixels + j] = raw[source + j] / 255;
    }
    source += pixels;
  }
  return tensor(values, [n, pixels]);
}

function loadLabels(path, limit) {
  const raw = gunzipSync(readFileSync(path));
  const magic = readU32BE(raw, 0);
  const count = readU32BE(raw, 4);
  if (magic !== 2049) throw new Error(`unexpected MNIST label file header for ${path}`);
  const n = Math.min(limit, count);
  return tensor(Array.from(raw.subarray(8, 8 + n)), [n]);
}

function seededRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function xavierWeights(inFeatures, outFeatures, seed) {
  const rng = seededRng(seed);
  const limit = Math.sqrt(6 / (inFeatures + outFeatures));
  return Array.from({ length: inFeatures * outFeatures }, () => (rng() * 2 - 1) * limit);
}

function createModel() {
  return new nn.Sequential([
    nn.linear(784, 128, {
      weights: xavierWeights(784, 128, 1),
    }),
    nn.relu(),
    nn.linear(128, 10, {
      weights: xavierWeights(128, 10, 2),
    }),
  ]);
}

function scalar(value) {
  return typeof value.item === "function" ? value.item() : value.data[0];
}

function assertCloseArray(actual, expected, tolerance, label) {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: expected length ${expected.length}, got ${actual.length}`);
  }
  for (let i = 0; i < actual.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > tolerance) {
      throw new Error(`${label}[${i}]: expected ${expected[i]} +/- ${tolerance}, got ${actual[i]}`);
    }
  }
}

function evaluate(model, loader) {
  return gradMode.inferenceMode(() => {
    let correct = 0;
    let total = 0;
    let lossSum = 0;
    for (const batch of loader) {
      const logits = model.forward(batch.input);
      const targets = Array.from(batch.target.data);
      const classes = train.predictClasses(logits, { classes: 10 });
      for (let i = 0; i < targets.length; i += 1) {
        if (classes[i] === targets[i]) correct += 1;
      }
      total += targets.length;
      lossSum += scalar(loss.crossEntropy(logits, batch.target, { classes: 10 })) * targets.length;
    }
    return { accuracy: correct / total, meanLoss: lossSum / total, total };
  });
}

async function main() {
  await ensureMnist();

  const trainImages = loadImages(join(cacheDir, files.trainImages), trainLimit);
  const trainLabels = loadLabels(join(cacheDir, files.trainLabels), trainLimit);
  const testImages = loadImages(join(cacheDir, files.testImages), testLimit);
  const testLabels = loadLabels(join(cacheDir, files.testLabels), testLimit);

  const trainDataset = data.tensorDataset(trainImages, trainLabels);
  const testDataset = data.tensorDataset(testImages, testLabels);
  const trainLoader = data.dataLoader(trainDataset, { batchSize, shuffle: true, seed: 20260627 });
  const testLoader = data.dataLoader(testDataset, { batchSize, shuffle: false });

  const model = createModel();
  const optimizer = optim.adam(model, { lr: 0.001 });
  const criterion = loss.crossEntropyLoss({ classes: 10 });
  const before = evaluate(model, testLoader);

  const fit = model.fit(trainLoader, {
    optimizer,
    loss: criterion,
    epochs,
  });

  const after = evaluate(model, testLoader);
  if (!train.isTrainFitEvidence(fit) || fit.steps !== Math.ceil(trainLimit / batchSize) * epochs) {
    throw new Error(`MNIST fit returned unexpected evidence: ${JSON.stringify(fit)}`);
  }
  if (fit.native !== true || fit.compiledPlan?.loweredBy !== "zig-ffi") {
    throw new Error(`MNIST expected model.fit to auto-select native Zig training, got ${JSON.stringify(fit.compiledPlan)}`);
  }
  if (!(after.meanLoss < before.meanLoss)) {
    throw new Error(`MNIST expected held-out loss to improve; before=${before.meanLoss}, after=${after.meanLoss}`);
  }
  if (!(after.accuracy >= accuracyFloor)) {
    throw new Error(`MNIST expected accuracy >= ${accuracyFloor}; got ${after.accuracy}`);
  }

  const snapshot = checkpoint.create({ model, optimizer, prefix: "mnist" });
  const restored = createModel();
  const restoredOptimizer = optim.adam(restored, { lr: 0.001 });
  checkpoint.restore(snapshot, { model: restored, optimizer: restoredOptimizer, prefix: "mnist", strict: true });

  const sample = testImages.select(0, 0);
  const eager = gradMode.inferenceMode(() => model.forward(sample));
  const restoredEager = gradMode.inferenceMode(() => restored.forward(sample));
  assertCloseArray(restoredEager.data, eager.data, 1e-5, "restored MNIST logits");

  const program = compile.compileForInference(model, { inputShape: [784], backend: "cpu" });
  const compiled = program.into(new Float32Array(10), sample);
  assertCloseArray(compiled, eager.data, 1e-4, "compiled MNIST logits");
  const compiledClass = train.predictClasses(tensor(compiled, [10]), { classes: 10 })[0];
  const eagerClass = train.predictClasses(eager, { classes: 10 })[0];
  if (compiledClass !== eagerClass) {
    throw new Error(`compiled MNIST class mismatch: eager=${eagerClass} compiled=${compiledClass}`);
  }
  program.dispose();

  console.log([
    "zgml MNIST MLP smoke ok:",
    `train=${trainLimit}`,
    `test=${testLimit}`,
    `epochs=${epochs}`,
    `steps=${fit.steps}`,
    `native=${fit.native === true}`,
    `lowered=${fit.compiledPlan?.loweredBy ?? "none"}`,
    `loss=${before.meanLoss.toFixed(4)}->${after.meanLoss.toFixed(4)}`,
    `accuracy=${(after.accuracy * 100).toFixed(2)}%`,
    `compiledClass=${compiledClass}`,
  ].join(" "));
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
