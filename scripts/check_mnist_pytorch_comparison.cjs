"use strict";

const { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } = require("node:fs");
const { get } = require("node:https");
const { spawnSync } = require("node:child_process");
const { join, resolve } = require("node:path");
const { gunzipSync } = require("node:zlib");
const os = require("node:os");
const { performance } = require("node:perf_hooks");

const root = resolve(__dirname, "..");
const nodeEntry = join(root, "dist", "node.cjs");
const cacheDir = join(root, ".cache", "zgml-mnist");
const artifactDir = process.env.BENCH_MNIST_ARTIFACT_DIR || join(root, "bench-results", "mnist-pytorch");
const venvPython = join(root, ".venv", "bin", "python");
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : "python3");
const installTorch = process.env.BENCH_MNIST_PYTORCH_INSTALL === "1";
const requireTrainingParity = process.env.BENCH_MNIST_REQUIRE_TRAINING_PARITY === "1";
const writeArtifact = process.env.BENCH_MNIST_WRITE_ARTIFACT !== "0";
const numericMode = process.env.BENCH_MNIST_NUMERIC_MODE || "strict";
if (!["strict", "convergence"].includes(numericMode)) {
  throw new Error(`BENCH_MNIST_NUMERIC_MODE must be strict or convergence, got ${numericMode}`);
}
const trainLimit = positiveInt(process.env.BENCH_MNIST_TRAIN_LIMIT || "2048", "BENCH_MNIST_TRAIN_LIMIT");
const testLimit = positiveInt(process.env.BENCH_MNIST_TEST_LIMIT || "512", "BENCH_MNIST_TEST_LIMIT");
const epochs = positiveInt(process.env.BENCH_MNIST_EPOCHS || "3", "BENCH_MNIST_EPOCHS");
const batchSize = positiveInt(process.env.BENCH_MNIST_BATCH_SIZE || "64", "BENCH_MNIST_BATCH_SIZE");
const trainParityFloor = Number(process.env.BENCH_MNIST_TRAINING_MIN_RATIO || "1.0");
const convergenceLossDeltaCeil = Number(process.env.BENCH_MNIST_CONVERGENCE_LOSS_DELTA_CEIL || "0.01");
const convergenceAccuracyDeltaCeil = Number(process.env.BENCH_MNIST_CONVERGENCE_ACCURACY_DELTA_CEIL || "0.005");
const mnistBaseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist";
const mnistFiles = Object.freeze([
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
]);

function positiveInt(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive safe integer, got ${value}`);
  }
  return parsed;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  if (result.status !== 0) {
    const error = new Error(`${command} ${args.join(" ")} failed with status ${result.status ?? 1}`);
    error.output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
    throw error;
  }
  return result.stdout ?? "";
}

function hasPythonTorch() {
  const result = spawnSync(python, ["-c", "import torch; print(torch.__version__)"], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return result.status === 0 ? result.stdout.trim() : null;
}

function installPythonTorchWithUv() {
  const uv = process.env.UV || "uv";
  if (!existsSync(venvPython)) run(uv, ["venv", ".venv"]);
  run(uv, ["pip", "install", "torch", "--python", venvPython], { stdio: "inherit" });
}

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
  for (const file of mnistFiles) {
    const path = join(cacheDir, file);
    if (!existsSync(path)) await download(`${mnistBaseUrl}/${file}`, path);
  }
}

function readU32BE(buffer, offset) {
  return buffer.readUInt32BE(offset);
}

function loadImages(path, limit, tensor) {
  const raw = gunzipSync(readFileSync(path));
  const magic = readU32BE(raw, 0);
  const count = readU32BE(raw, 4);
  const rows = readU32BE(raw, 8);
  const cols = readU32BE(raw, 12);
  if (magic !== 2051 || rows !== 28 || cols !== 28) throw new Error(`unexpected MNIST image header for ${path}`);
  const n = Math.min(limit, count);
  const pixels = rows * cols;
  const values = new Float32Array(n * pixels);
  let source = 16;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < pixels; j += 1) values[i * pixels + j] = raw[source + j] / 255;
    source += pixels;
  }
  return tensor(values, [n, pixels]);
}

function loadLabels(path, limit, tensor) {
  const raw = gunzipSync(readFileSync(path));
  const magic = readU32BE(raw, 0);
  const count = readU32BE(raw, 4);
  if (magic !== 2049) throw new Error(`unexpected MNIST label header for ${path}`);
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

function scalar(value) {
  return typeof value.item === "function" ? value.item() : value.data[0];
}

function bench(fn, iterations = 2000, minMs = 150) {
  for (let i = 0; i < 100; i += 1) fn();
  const start = performance.now();
  let total = 0;
  while (true) {
    for (let i = 0; i < iterations; i += 1) fn();
    total += iterations;
    const elapsed = performance.now() - start;
    if (elapsed >= minMs) return elapsed / total;
  }
}

function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`diff length mismatch: ${a.length} != ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i += 1) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

function round(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : null;
}

function timestampForArtifact(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function runZgml() {
  if (!existsSync(nodeEntry)) throw new Error("MNIST PyTorch comparison requires dist/node.cjs; run npm run build:package first");
  const { compile, data, gradMode, loss, nn, optim, tensor, train } = require(nodeEntry);
  const trainImages = loadImages(join(cacheDir, "train-images-idx3-ubyte.gz"), trainLimit, tensor);
  const trainLabels = loadLabels(join(cacheDir, "train-labels-idx1-ubyte.gz"), trainLimit, tensor);
  const testImages = loadImages(join(cacheDir, "t10k-images-idx3-ubyte.gz"), testLimit, tensor);
  const testLabels = loadLabels(join(cacheDir, "t10k-labels-idx1-ubyte.gz"), testLimit, tensor);

  function createModel() {
    return new nn.Sequential([
      nn.linear(784, 128, { weights: xavierWeights(784, 128, 1) }),
      nn.relu(),
      nn.linear(128, 10, { weights: xavierWeights(128, 10, 2) }),
    ]);
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
      return { meanLoss: lossSum / total, accuracy: correct / total, total };
    });
  }

  const model = createModel();
  const optimizer = optim.adam(model, { lr: 0.001 });
  const criterion = loss.crossEntropyLoss({ classes: 10 });
  const trainLoader = data.dataLoader(data.tensorDataset(trainImages, trainLabels), { batchSize, shuffle: true, seed: 20260627 });
  const testLoader = data.dataLoader(data.tensorDataset(testImages, testLabels), { batchSize, shuffle: false });
  const before = evaluate(model, testLoader);
  const trainStart = performance.now();
  const fit = model.fit(trainLoader, {
    optimizer,
    loss: criterion,
    epochs,
  });
  const trainMs = performance.now() - trainStart;
  if (fit.native !== true || fit.compiledPlan?.loweredBy !== "zig-ffi") {
    throw new Error(`MNIST PyTorch comparison expected ergonomic model.fit to lower through Zig FFI, got ${JSON.stringify(fit.compiledPlan)}`);
  }
  if (fit.nativeBulk !== true) {
    throw new Error(`MNIST PyTorch comparison expected ergonomic model.fit to use Zig-owned bulk training, got ${JSON.stringify({ nativeBulk: fit.nativeBulk, bulkResult: fit.bulkResult })}`);
  }
  const after = evaluate(model, testLoader);
  const sample = testImages.select(0, 0);
  const eager = gradMode.inferenceMode(() => model.forward(sample));
  const program = compile.compileForInference(model, { inputShape: [784], backend: "cpu" });
  const compiledBuffer = new Float32Array(10);
  const compiled = program.into(compiledBuffer, sample);
  const eagerSingleForwardMs = bench(() => gradMode.inferenceMode(() => model.forward(sample)));
  const compiledSingleForwardMs = bench(() => program.into(compiledBuffer, sample));
  program.dispose();

  const nativeModel = createModel();
  const nativeOptimizer = optim.adam(nativeModel, { lr: 0.001 });
  const nativeStep = compile.trainingStep(nativeModel, nativeOptimizer, {
    inputShape: [batchSize, 784],
    loss: "crossEntropy",
    classes: 10,
  });
  const nativeBefore = evaluate(nativeModel, testLoader);
  const nativeStart = performance.now();
  let nativeSteps = 0;
  let nativeLastLoss = 0;
  let nativeLastAccuracy = 0;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    for (const batch of trainLoader) {
      const step = nativeStep.step(batch.input, batch.target);
      nativeLastLoss = step.loss;
      nativeLastAccuracy = step.accuracy;
      nativeSteps += 1;
    }
  }
  const nativeTrainMs = performance.now() - nativeStart;
  const nativeAfter = evaluate(nativeModel, testLoader);
  nativeStep.dispose();

  return {
    before,
    after,
    steps: fit.steps,
    trainMs,
    trainMsPerStep: trainMs / fit.steps,
    modelFitNative: fit.native === true,
    modelFitBulk: fit.nativeBulk === true,
    modelFitBulkKernel: fit.bulkResult?.kernel ?? null,
    modelFitLoweredBy: fit.compiledPlan?.loweredBy ?? null,
    eagerSingleForwardMs,
    compiledSingleForwardMs,
    sample0Pred: train.predictClasses(eager, { classes: 10 })[0],
    compiledSample0Pred: train.predictClasses(tensor(compiled, [10]), { classes: 10 })[0],
    sample0Logits: Array.from(eager.data),
    compiledSample0Logits: Array.from(compiled),
    compiledMaxAbsDiff: maxAbsDiff(eager.data, compiled),
    crossEntropyProbe: loss.crossEntropy([1, 2, 3, 2, 0, -1], [2, 0], { classes: 3 }),
    nativeTraining: {
      before: nativeBefore,
      after: nativeAfter,
      steps: nativeSteps,
      trainMs: nativeTrainMs,
      trainMsPerStep: nativeTrainMs / nativeSteps,
      lastBatchLoss: nativeLastLoss,
      lastBatchAccuracy: nativeLastAccuracy,
      speedupVsJs: trainMs / nativeTrainMs,
    },
  };
}

const pytorchCode = String.raw`
import gzip, json, math, os, struct, time
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path(os.environ["ZGML_ROOT"])
cache = root / ".cache" / "zgml-mnist"
train_limit = int(os.environ["BENCH_MNIST_TRAIN_LIMIT"])
test_limit = int(os.environ["BENCH_MNIST_TEST_LIMIT"])
epochs = int(os.environ["BENCH_MNIST_EPOCHS"])
batch_size = int(os.environ["BENCH_MNIST_BATCH_SIZE"])
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def load_images(name, limit):
    raw = gzip.decompress((cache / name).read_bytes())
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051 and rows == 28 and cols == 28
    n = min(limit, count)
    return torch.tensor(list(raw[16:16+n*rows*cols]), dtype=torch.float32).reshape(n, rows*cols) / 255.0

def load_labels(name, limit):
    raw = gzip.decompress((cache / name).read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    assert magic == 2049
    n = min(limit, count)
    return torch.tensor(list(raw[8:8+n]), dtype=torch.long)

def seeded_rng(seed):
    state = seed & 0xffffffff
    def rng():
        nonlocal state
        state = ((1664525 * state + 1013904223) & 0xffffffff)
        return state / 2**32
    return rng

def shuffled_indices(length, seed):
    rng = seeded_rng(seed)
    indices = list(range(length))
    for i in range(len(indices) - 1, 0, -1):
        j = math.floor(rng() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    return indices

def xavier_weights(in_features, out_features, seed):
    rng = seeded_rng(seed)
    limit = math.sqrt(6 / (in_features + out_features))
    values = [(rng() * 2 - 1) * limit for _ in range(in_features * out_features)]
    return torch.tensor(values, dtype=torch.float32).reshape(in_features, out_features).T.contiguous()

class MnistMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        with torch.no_grad():
            self.fc1.weight.copy_(xavier_weights(784, 128, 1))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(xavier_weights(128, 10, 2))
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def evaluate(model, x, y):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = model(xb)
            objective = F.cross_entropy(logits, yb, reduction="mean")
            total_loss += float(objective) * len(yb)
            correct += int((logits.argmax(dim=-1) == yb).sum())
            total += len(yb)
    return {"meanLoss": total_loss / total, "accuracy": correct / total, "total": total}

def bench(fn, iterations=2000, min_ms=150):
    with torch.inference_mode():
        for _ in range(100):
            fn()
        start = time.perf_counter()
        total = 0
        while True:
            for _ in range(iterations):
                fn()
            total += iterations
            elapsed = (time.perf_counter() - start) * 1000.0
            if elapsed >= min_ms:
                return elapsed / total

train_x = load_images("train-images-idx3-ubyte.gz", train_limit)
train_y = load_labels("train-labels-idx1-ubyte.gz", train_limit)
test_x = load_images("t10k-images-idx3-ubyte.gz", test_limit)
test_y = load_labels("t10k-labels-idx1-ubyte.gz", test_limit)
model = MnistMlp()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
order = shuffled_indices(train_limit, 20260627)
batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
before = evaluate(model, test_x, test_y)
train_start = time.perf_counter()
steps = 0
for _epoch in range(epochs):
    model.train()
    for batch in batches:
        idx = torch.tensor(batch, dtype=torch.long)
        objective = F.cross_entropy(model(train_x.index_select(0, idx)), train_y.index_select(0, idx), reduction="mean")
        optimizer.zero_grad(set_to_none=False)
        objective.backward()
        optimizer.step()
        steps += 1
train_ms = (time.perf_counter() - train_start) * 1000.0
after = evaluate(model, test_x, test_y)
sample = test_x[0]
with torch.inference_mode():
    logits_tensor = model(sample)
    logits = logits_tensor.detach().numpy().tolist()
    pred = int(logits_tensor.argmax())
eager_ms = bench(lambda: model(sample))
compiled = {}
if hasattr(torch, "compile"):
    for mode in [None, "reduce-overhead", "max-autotune"]:
        label = "default" if mode is None else mode
        try:
            kwargs = {} if mode is None else {"mode": mode}
            compiled_model = torch.compile(model, **kwargs)
            with torch.inference_mode():
                for _ in range(5):
                    compiled_out = compiled_model(sample)
            compiled[label] = {
                "ms": bench(lambda cm=compiled_model: cm(sample)),
                "maxAbsDiff": float((compiled_out - logits_tensor).abs().max()),
            }
        except Exception as exc:
            compiled[label] = {"error": repr(exc)}
ce_logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=torch.float32)
ce_targets = torch.tensor([2, 0], dtype=torch.long)
print(json.dumps({
    "torchVersion": torch.__version__,
    "threads": torch.get_num_threads(),
    "before": before,
    "after": after,
    "steps": steps,
    "trainMs": train_ms,
    "trainMsPerStep": train_ms / steps,
    "eagerSingleForwardMs": eager_ms,
    "compiledSingleForward": compiled,
    "sample0Pred": pred,
    "sample0Logits": logits,
    "crossEntropyProbe": float(F.cross_entropy(ce_logits, ce_targets, reduction="mean")),
}))
`;

async function main() {
  await ensureMnist();
  let pytorchVersion = hasPythonTorch();
  if (!pytorchVersion && installTorch) {
    installPythonTorchWithUv();
    pytorchVersion = hasPythonTorch();
  }
  if (!pytorchVersion) {
    console.log(`mnist pytorch comparison skipped: ${python} import torch failed; set BENCH_MNIST_PYTORCH_INSTALL=1 to bootstrap .venv with uv`);
    return;
  }
  const zgml = runZgml();
  const pytorch = JSON.parse(run(python, ["-c", pytorchCode], {
    env: {
      ...process.env,
      ZGML_ROOT: root,
      BENCH_MNIST_TRAIN_LIMIT: String(trainLimit),
      BENCH_MNIST_TEST_LIMIT: String(testLimit),
      BENCH_MNIST_EPOCHS: String(epochs),
      BENCH_MNIST_BATCH_SIZE: String(batchSize),
    },
  }));
  const numeric = {
    beforeLossDelta: Math.abs(zgml.before.meanLoss - pytorch.before.meanLoss),
    afterLossDelta: Math.abs(zgml.after.meanLoss - pytorch.after.meanLoss),
    beforeAccuracyDelta: Math.abs(zgml.before.accuracy - pytorch.before.accuracy),
    afterAccuracyDelta: Math.abs(zgml.after.accuracy - pytorch.after.accuracy),
    crossEntropyDelta: Math.abs(zgml.crossEntropyProbe - pytorch.crossEntropyProbe),
    sample0LogitsMaxAbsDiff: maxAbsDiff(zgml.sample0Logits, pytorch.sample0Logits),
    compiledLogitsMaxAbsDiff: zgml.compiledMaxAbsDiff,
    sample0PredMatch: zgml.sample0Pred === pytorch.sample0Pred && zgml.compiledSample0Pred === zgml.sample0Pred,
    stepsMatch: zgml.steps === pytorch.steps,
  };
  const strictNumericReady = numeric.beforeLossDelta <= 1e-6 &&
    numeric.afterLossDelta <= 1e-5 &&
    numeric.beforeAccuracyDelta === 0 &&
    numeric.afterAccuracyDelta === 0 &&
    numeric.crossEntropyDelta <= 1e-6 &&
    numeric.sample0LogitsMaxAbsDiff <= 1e-5 &&
    numeric.compiledLogitsMaxAbsDiff <= 1e-6 &&
    numeric.sample0PredMatch &&
    numeric.stepsMatch;
  const convergenceNumericReady = numeric.beforeLossDelta <= 1e-6 &&
    numeric.afterLossDelta <= convergenceLossDeltaCeil &&
    numeric.beforeAccuracyDelta <= convergenceAccuracyDeltaCeil &&
    numeric.afterAccuracyDelta <= convergenceAccuracyDeltaCeil &&
    numeric.crossEntropyDelta <= 1e-6 &&
    numeric.compiledLogitsMaxAbsDiff <= 1e-6 &&
    numeric.sample0PredMatch &&
    numeric.stepsMatch;
  const numericReady = numericMode === "strict" ? strictNumericReady : convergenceNumericReady;
  const ratios = {
    trainZgmlVsPytorch: pytorch.trainMs / zgml.trainMs,
    trainStepZgmlVsPytorch: pytorch.trainMsPerStep / zgml.trainMsPerStep,
    trainManualNativeZgmlVsPytorch: pytorch.trainMs / zgml.nativeTraining.trainMs,
    trainStepManualNativeZgmlVsPytorch: pytorch.trainMsPerStep / zgml.nativeTraining.trainMsPerStep,
    trainManualNativeZgmlVsModelFitZgml: zgml.trainMs / zgml.nativeTraining.trainMs,
    eagerInferenceZgmlVsPytorch: pytorch.eagerSingleForwardMs / zgml.eagerSingleForwardMs,
    compiledInferenceZgmlVsPytorchEager: pytorch.eagerSingleForwardMs / zgml.compiledSingleForwardMs,
  };
  const trainingParityReady = ratios.trainZgmlVsPytorch >= trainParityFloor;
  const artifact = {
    schema: "zgml.mnist-pytorch-comparison.v1",
    createdAt: new Date().toISOString(),
    command: "scripts/check_mnist_pytorch_comparison.cjs",
    nodeVersion: process.version,
    python,
    pytorchVersion: pytorch.torchVersion ?? pytorchVersion,
    platform: {
      type: os.type(),
      platform: os.platform(),
      arch: os.arch(),
      release: os.release(),
      cpus: os.cpus().length,
    },
    config: {
      trainLimit,
      testLimit,
      epochs,
      batchSize,
      trainParityFloor,
      requireTrainingParity,
      numericMode,
      convergenceLossDeltaCeil,
      convergenceAccuracyDeltaCeil,
    },
    status: {
      numericReady,
      strictNumericReady,
      convergenceNumericReady,
      trainingParityReady,
      comparisonReady: numericReady && (!requireTrainingParity || trainingParityReady),
    },
    numeric,
    ratios,
    zgml,
    pytorch,
  };
  let artifactPath = null;
  if (writeArtifact) {
    mkdirSync(artifactDir, { recursive: true });
    artifactPath = join(artifactDir, `mnist-pytorch-${timestampForArtifact()}-${process.pid}.json`);
    writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  }
  const parts = [
    `mnist pytorch comparison: ${artifact.status.comparisonReady ? "pass" : "miss"}`,
    `numeric_mode=${numericMode}`,
    `numeric=${numericReady ? "pass" : "miss"}`,
    `strict_numeric=${strictNumericReady ? "pass" : "miss"}`,
    `convergence_numeric=${convergenceNumericReady ? "pass" : "miss"}`,
    `training_parity=${trainingParityReady ? "pass" : "miss"}`,
    `torch=${artifact.pytorchVersion}`,
    `train=${trainLimit}`,
    `test=${testLimit}`,
    `epochs=${epochs}`,
    `zgml_train=${round(zgml.trainMs)}ms`,
    `zgml_model_fit_native=${zgml.modelFitNative ? "yes" : "no"}:${zgml.modelFitLoweredBy ?? "none"}`,
    `zgml_model_fit_bulk=${zgml.modelFitBulk ? "yes" : "no"}:${zgml.modelFitBulkKernel ?? "none"}`,
    `zgml_manual_native_train=${round(zgml.nativeTraining.trainMs)}ms`,
    `pytorch_train=${round(pytorch.trainMs)}ms`,
    `zgml_vs_pytorch_train=${ratios.trainZgmlVsPytorch.toFixed(3)}x`,
    `zgml_manual_native_vs_pytorch_train=${ratios.trainManualNativeZgmlVsPytorch.toFixed(3)}x`,
    `manual_native_vs_model_fit=${ratios.trainManualNativeZgmlVsModelFitZgml.toFixed(3)}x`,
    `zgml_acc=${(zgml.after.accuracy * 100).toFixed(2)}%`,
    `zgml_native_acc=${(zgml.nativeTraining.after.accuracy * 100).toFixed(2)}%`,
    `pytorch_acc=${(pytorch.after.accuracy * 100).toFixed(2)}%`,
    `loss_delta=${numeric.afterLossDelta.toExponential(2)}`,
    `logits_max_abs=${numeric.sample0LogitsMaxAbsDiff.toExponential(2)}`,
    `zgml_infer=${round(zgml.eagerSingleForwardMs)}ms`,
    `zgml_compiled_infer=${round(zgml.compiledSingleForwardMs)}ms`,
    `pytorch_infer=${round(pytorch.eagerSingleForwardMs)}ms`,
    `artifact=${artifactPath ? artifactPath.replace(`${root}/`, "") : "disabled"}`,
  ];
  console.log(parts.join(" "));
  if (!numericReady) throw new Error(`MNIST PyTorch ${numericMode} numeric parity failed: ${JSON.stringify(numeric)}`);
  if (requireTrainingParity && !trainingParityReady) {
    throw new Error(`MNIST training parity failed: zgml_vs_pytorch_train=${ratios.trainZgmlVsPytorch.toFixed(3)}x floor=${trainParityFloor.toFixed(3)}x`);
  }
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  if (error && error.output) console.error(error.output);
  process.exit(1);
});
