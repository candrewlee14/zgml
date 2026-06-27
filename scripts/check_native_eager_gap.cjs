"use strict";

const { existsSync, mkdirSync, writeFileSync } = require("node:fs");
const { join, resolve } = require("node:path");
const { verifyFreshNativeLibrary } = require("./native_freshness.cjs");

const root = resolve(__dirname, "..");
const artifactDir = process.env.BENCH_NATIVE_EAGER_ARTIFACT_DIR || join("bench-results", "native-eager");
const writeArtifact = process.env.BENCH_NATIVE_EAGER_WRITE_ARTIFACT !== "0";

function timestampForArtifact(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function hostRuntime() {
  const requested = String(process.env.BENCH_NATIVE_EAGER_RUNTIME || "").trim().toLowerCase();
  if (requested.length > 0) {
    if (requested !== "node" && requested !== "bun") {
      throw new Error(`BENCH_NATIVE_EAGER_RUNTIME must be node or bun, got ${requested}`);
    }
    return requested;
  }
  return process.versions.bun ? "bun" : "node";
}

const runtime = hostRuntime();
const runtimeEntry = join(root, "dist", runtime === "bun" ? "bun_native.cjs" : "node.cjs");
if (!existsSync(runtimeEntry)) {
  throw new Error(`native eager gap check requires ${runtimeEntry}; run npm run build:package first`);
}
const nativeFreshness = verifyFreshNativeLibrary({
  root,
  label: `native eager gap check (${runtime})`,
  allowStaleEnv: "BENCH_NATIVE_EAGER_ALLOW_STALE_NATIVE",
});

const runtimeExports = require(runtimeEntry);
const zgml = runtimeExports.zgml ?? runtimeExports.torch ?? runtimeExports;
const minTimingMs = Number(process.env.BENCH_NATIVE_EAGER_MIN_TIMING_MS || "20");
if (!Number.isFinite(minTimingMs) || minTimingMs <= 0) {
  throw new Error(`BENCH_NATIVE_EAGER_MIN_TIMING_MS must be positive, got ${process.env.BENCH_NATIVE_EAGER_MIN_TIMING_MS}`);
}
const minNativeEagerSpeedup = Number(process.env.BENCH_NATIVE_EAGER_MIN_SPEEDUP || "1.0");
if (!Number.isFinite(minNativeEagerSpeedup) || minNativeEagerSpeedup <= 0) {
  throw new Error(`BENCH_NATIVE_EAGER_MIN_SPEEDUP must be positive, got ${process.env.BENCH_NATIVE_EAGER_MIN_SPEEDUP}`);
}

function values(length, scale) {
  return Array.from({ length }, (_, index) => ((index % 17) - 8) / scale);
}

function msNow() {
  return Number(process.hrtime.bigint()) / 1e6;
}

function bench(fn, iterations) {
  const warmupIterations = Math.min(100, iterations);
  for (let index = 0; index < warmupIterations; index += 1) fn();
  let elapsed = 0;
  let totalIterations = 0;
  const start = msNow();
  do {
    for (let index = 0; index < iterations; index += 1) fn();
    totalIterations += iterations;
    elapsed = msNow() - start;
  } while (elapsed < minTimingMs);
  return elapsed / totalIterations;
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let index = 0; index < a.length; index += 1) {
    max = Math.max(max, Math.abs(a[index] - b[index]));
  }
  return max;
}

function round(value) {
  return Number(value.toFixed(6));
}

function geluScalar(value) {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value)));
}

function activationScalar(value, activation) {
  switch (activation) {
    case "gelu": return geluScalar(value);
    case "relu": return value > 0 ? value : 0;
    case "silu": return value / (1 + Math.exp(-value));
    case "sigmoid": return 1 / (1 + Math.exp(-value));
    case "tanh": return Math.tanh(value);
    default: throw new Error(`unsupported native eager reference activation ${activation}`);
  }
}

function lazyMatmulAddActivationEager(input, weightValues, biasValues, batch, inFeatures, outFeatures, activation) {
  const out = new Float32Array(batch * outFeatures);
  for (let row = 0; row < batch; row += 1) {
    for (let col = 0; col < outFeatures; col += 1) {
      let sum = biasValues[col];
      for (let feature = 0; feature < inFeatures; feature += 1) {
        sum += input.data[row * inFeatures + feature] * weightValues[feature * outFeatures + col];
      }
      out[row * outFeatures + col] = activationScalar(sum, activation);
    }
  }
  return out;
}

function requireCompiledHotPath(key, session, input, output) {
  const compatibility = session.requireHotStepParams({ input, output });
  const plan = session.hotPathPlan({ input, output });
  if (
    compatibility.hotPath !== true ||
    compatibility.runtimeOutputAllocationFree !== true ||
    compatibility.readbackFree !== true ||
    plan.hotPath !== true
  ) {
    throw new Error(`${key} expected compiled hot path: ${JSON.stringify({ compatibility, plan })}`);
  }
}

function benchGap(spec) {
  const input = spec.input();
  const compiled = typeof spec.compiled === "function" ? spec.compiled() : null;
  try {
    const output = new Float32Array(spec.outputLen);
    const nativeEagerOutput = new Float32Array(spec.outputLen);
    if (compiled) {
      requireCompiledHotPath(spec.key, compiled.session, input, output);
    }
    const eagerOutput = spec.eager(input);
    const eagerData = eagerOutput.data ?? eagerOutput;
    const nativeEagerResult = typeof spec.nativeEager === "function" ? spec.nativeEager(nativeEagerOutput, input) : null;
    if (nativeEagerResult && nativeEagerResult !== nativeEagerOutput) {
      throw new Error(`${spec.key} expected native eager output to reuse caller output`);
    }
    const nativeEagerModuleOutput = typeof spec.nativeEagerModule === "function" ? spec.nativeEagerModule(input) : null;
    const nativeEagerModuleData = nativeEagerModuleOutput ? (nativeEagerModuleOutput.data ?? nativeEagerModuleOutput) : null;
    const nativeEagerDiff = nativeEagerResult ? maxAbsDiff(eagerData, nativeEagerOutput) : null;
    if (nativeEagerDiff !== null && nativeEagerDiff > spec.tolerance) {
      throw new Error(`${spec.key} native eager parity failed: max_abs_diff=${nativeEagerDiff}`);
    }
    const nativeEagerModuleDiff = nativeEagerModuleData ? maxAbsDiff(eagerData, nativeEagerModuleData) : null;
    if (nativeEagerModuleDiff !== null && nativeEagerModuleDiff > spec.tolerance) {
      throw new Error(`${spec.key} native eager module parity failed: max_abs_diff=${nativeEagerModuleDiff}`);
    }
    let diff = null;
    if (compiled) {
      const compiledOutput = compiled.into(output, input);
      if (compiledOutput !== output) {
        throw new Error(`${spec.key} expected compiled output to reuse caller output`);
      }
      diff = maxAbsDiff(eagerData, output);
      if (diff > spec.tolerance) {
        throw new Error(`${spec.key} compiled parity failed: max_abs_diff=${diff}`);
      }
    }

    const eagerMs = bench(() => {
      spec.eager(input);
    }, spec.eagerIterations);
    const preparedMs = compiled
      ? bench(() => {
          compiled.into(output, input);
        }, spec.compiledIterations)
      : null;
    const nativeEagerMs = typeof spec.nativeEager === "function"
      ? bench(() => {
          spec.nativeEager(nativeEagerOutput, input);
        }, spec.nativeEagerIterations ?? spec.compiledIterations)
      : null;
    const nativeEagerModuleMs = typeof spec.nativeEagerModule === "function"
      ? bench(() => {
          spec.nativeEagerModule(input);
        }, spec.nativeEagerModuleIterations ?? spec.eagerIterations)
      : null;
    const nativeEagerSpeedupFloor = Number.isFinite(spec.minNativeEagerSpeedup)
      ? Number(spec.minNativeEagerSpeedup)
      : minNativeEagerSpeedup;
    const nativeEagerModuleSpeedupFloor = Number.isFinite(spec.minNativeEagerModuleSpeedup)
      ? Number(spec.minNativeEagerModuleSpeedup)
      : nativeEagerSpeedupFloor;
    if (nativeEagerMs !== null && eagerMs / nativeEagerMs < nativeEagerSpeedupFloor) {
      throw new Error(`${spec.key} native eager speedup ${eagerMs / nativeEagerMs}x below ${nativeEagerSpeedupFloor}x`);
    }
    if (nativeEagerModuleMs !== null && eagerMs / nativeEagerModuleMs < nativeEagerModuleSpeedupFloor) {
      throw new Error(`${spec.key} native eager module speedup ${eagerMs / nativeEagerModuleMs}x below ${nativeEagerModuleSpeedupFloor}x`);
    }
    const speedup = preparedMs === null ? null : eagerMs / preparedMs;
    const row = {
      schema: "zgml.native-eager-gap.v1",
      key: spec.key,
      shape: Object.freeze(spec.shape),
      compiledHotPath: compiled !== null,
      eagerMs: round(eagerMs),
      nativeEagerIntoMs: nativeEagerMs === null ? null : round(nativeEagerMs),
      nativeEagerSpeedup: nativeEagerMs === null ? null : round(eagerMs / nativeEagerMs),
      nativeEagerSpeedupFloor: nativeEagerMs === null ? null : round(nativeEagerSpeedupFloor),
      nativeEagerModuleForwardMs: nativeEagerModuleMs === null ? null : round(nativeEagerModuleMs),
      nativeEagerModuleSpeedup: nativeEagerModuleMs === null ? null : round(eagerMs / nativeEagerModuleMs),
      nativeEagerModuleSpeedupFloor: nativeEagerModuleMs === null ? null : round(nativeEagerModuleSpeedupFloor),
      preparedExecuteIntoMs: preparedMs === null ? null : round(preparedMs),
      nativeProgramSpeedup: speedup === null ? null : round(speedup),
      nativeEagerMaxAbsDiff: nativeEagerDiff === null ? null : round(nativeEagerDiff),
      nativeEagerModuleMaxAbsDiff: nativeEagerModuleDiff === null ? null : round(nativeEagerModuleDiff),
      maxAbsDiff: diff === null ? null : round(diff),
      status: compiled ? "gap-measured" : "native-eager-measured",
      next: spec.next,
    };
    return Object.freeze(row);
  } finally {
    if (compiled) compiled.dispose();
  }
}

function compiledInferenceHandle(model, inputShape) {
  const fast = zgml.native(model, { backend: "cpu", inputShape });
  return Object.freeze({
    session: fast.session,
    into(output, input) {
      return fast.into(output, input);
    },
    dispose() {
      fast.dispose();
    },
  });
}

function compiledLazyHandle(graph, bindings, inputShape) {
  const fast = zgml.native(graph, { backend: "cpu", inputShape }, bindings);
  return Object.freeze({
    session: fast.session,
    into(output, input) {
      return fast.into(output, input);
    },
    dispose() {
      fast.dispose();
    },
  });
}

function linearBatchedModel() {
  return new zgml.nn.Sequential(
    new zgml.nn.Linear(64, 32, {
      weights: linearWeights,
      bias: linearBias,
    }),
  );
}

function linearActivationBatchedModel(weights, bias, ActivationModule) {
  return new zgml.nn.Sequential(
    new zgml.nn.Linear(64, 64, {
      weights,
      bias,
    }),
    new ActivationModule(),
  );
}

function linearGeluBatchedModel() {
  return linearActivationBatchedModel(geluWeights, geluBias, zgml.nn.GELU);
}

function linearReluBatchedModel() {
  return linearActivationBatchedModel(reluWeights, reluBias, zgml.nn.ReLU);
}

function linearSiluBatchedModel() {
  return linearActivationBatchedModel(siluWeights, siluBias, zgml.nn.SiLU);
}

function linearSigmoidBatchedModel() {
  return linearActivationBatchedModel(sigmoidWeights, sigmoidBias, zgml.nn.Sigmoid);
}

function linearTanhBatchedModel() {
  return linearActivationBatchedModel(tanhWeights, tanhBias, zgml.nn.Tanh);
}

function softmaxBatchedModel() {
  return new zgml.nn.Softmax(-1);
}

function logSoftmaxBatchedModel() {
  return new zgml.nn.LogSoftmax(-1);
}

function conv2dBatchedModel() {
  return new zgml.nn.Sequential(
    new zgml.nn.Conv2d(1, 1, 3, {
      weight: conv2dWeights,
      bias: conv2dBias,
    }),
  );
}

function maxPool2dBatchedModel() {
  return new zgml.nn.Sequential(new zgml.nn.MaxPool2d(2));
}

function avgPool2dBatchedModel() {
  return new zgml.nn.Sequential(new zgml.nn.AvgPool2d(2));
}

const linearWeights = values(64 * 32, 64);
const linearBias = values(32, 32);
const linearWeightTensor = zgml.tensor(linearWeights, [64, 32]);
const linearBiasTensor = zgml.tensor(linearBias, [32]);
const linearModel = linearBatchedModel();
const matmulWeights = values(64 * 64, 36);
const matmulWeightTensor = zgml.tensor(matmulWeights, [64, 64]);
const geluWeights = values(64 * 64, 32);
const geluBias = values(64, 64);
const geluWeightTensor = zgml.tensor(geluWeights, [64, 64]);
const geluBiasTensor = zgml.tensor(geluBias, [64]);
const linearGeluModel = linearGeluBatchedModel();
const reluWeights = values(64 * 64, 48);
const reluBias = values(64, 96);
const reluWeightTensor = zgml.tensor(reluWeights, [64, 64]);
const reluBiasTensor = zgml.tensor(reluBias, [64]);
const linearReluModel = linearReluBatchedModel();
const siluWeights = values(64 * 64, 40);
const siluBias = values(64, 80);
const siluWeightTensor = zgml.tensor(siluWeights, [64, 64]);
const siluBiasTensor = zgml.tensor(siluBias, [64]);
const linearSiluModel = linearSiluBatchedModel();
const sigmoidWeights = values(64 * 64, 56);
const sigmoidBias = values(64, 112);
const sigmoidWeightTensor = zgml.tensor(sigmoidWeights, [64, 64]);
const sigmoidBiasTensor = zgml.tensor(sigmoidBias, [64]);
const linearSigmoidModel = linearSigmoidBatchedModel();
const tanhWeights = values(64 * 64, 72);
const tanhBias = values(64, 144);
const tanhWeightTensor = zgml.tensor(tanhWeights, [64, 64]);
const tanhBiasTensor = zgml.tensor(tanhBias, [64]);
const linearTanhModel = linearTanhBatchedModel();
const softmaxModel = softmaxBatchedModel();
const logSoftmaxModel = logSoftmaxBatchedModel();
const conv2dWeights = values(3 * 3, 24);
const conv2dBias = values(1, 32);
const conv2dWeightTensor = zgml.tensor(conv2dWeights, [1, 1, 3, 3]);
const conv2dBiasTensor = zgml.tensor(conv2dBias, [1]);
const conv2dModel = conv2dBatchedModel();
const maxPool2dModel = maxPool2dBatchedModel();
const avgPool2dModel = avgPool2dBatchedModel();

const gapSpecs = Object.freeze([
  Object.freeze({
    key: "linear_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 32 }),
    outputLen: 128 * 32,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => linearModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.linearInto(output, input, linearWeightTensor, {
      bias: linearBiasTensor,
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearModel.forward(input)),
    compiled: () => compiledInferenceHandle(linearBatchedModel(), [128, 64]),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_linear_or_matmul_storage_slice",
  }),
  Object.freeze({
    key: "matmul_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64 }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => input.matmul(matmulWeightTensor),
    nativeEager: (output, input) => zgml.nativeEager.matmulInto(output, input, matmulWeightTensor),
    nativeEagerModule: (input) => zgml.noGrad(() => input.matmul(matmulWeightTensor)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64]).matmul(zgml.lazy.parameter([64, 64], "w")),
      {
        weights: new Float32Array(matmulWeights),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_linear_or_matmul_storage_slice",
  }),
  Object.freeze({
    key: "elementwise_mul_batched",
    shape: Object.freeze({ batch: 128, features: 64, op: "mul_scalar" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => input.mul(2),
    nativeEager: (output, input) => zgml.nativeEager.elementwiseInto(output, input, new Float32Array([2]), { op: "mul" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.mul(2)),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerModuleSpeedup: runtime === "bun" ? 0.85 : 1,
    tolerance: 1e-5,
    next: "native_eager_elementwise_storage_slice",
  }),
  Object.freeze({
    key: "activation_relu_batched",
    shape: Object.freeze({ batch: 512, features: 256, activation: "relu" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 13), [512, 256]),
    eager: (input) => input.relu(),
    nativeEager: (output, input) => zgml.nativeEager.activationInto(output, input, { activation: "relu" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.relu()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: 2,
    minNativeEagerModuleSpeedup: 2,
    tolerance: 1e-6,
    next: "native_eager_activation_silu_tensor_route",
  }),
  Object.freeze({
    key: "activation_sigmoid_batched",
    shape: Object.freeze({ batch: 512, features: 256, activation: "sigmoid" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 13), [512, 256]),
    eager: (input) => input.sigmoid(),
    nativeEager: (output, input) => zgml.nativeEager.activationInto(output, input, { activation: "sigmoid" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.sigmoid()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: 1.5,
    minNativeEagerModuleSpeedup: 1.5,
    tolerance: 1e-6,
    next: "native_eager_activation_silu_tensor_route",
  }),
  Object.freeze({
    key: "activation_gelu_batched",
    shape: Object.freeze({ batch: 512, features: 256, activation: "gelu" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 19), [512, 256]),
    eager: (input) => input.gelu(),
    nativeEager: (output, input) => zgml.nativeEager.activationInto(output, input, { activation: "gelu" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.gelu()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: 1.5,
    minNativeEagerModuleSpeedup: 1.5,
    tolerance: 2e-6,
    next: "native_eager_activation_tanh_tensor_route",
  }),
  Object.freeze({
    key: "activation_silu_batched",
    shape: Object.freeze({ batch: 512, features: 256, activation: "silu" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 17), [512, 256]),
    eager: (input) => input.silu(),
    nativeEager: (output, input) => zgml.nativeEager.activationInto(output, input, { activation: "silu" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.silu()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: 1.5,
    minNativeEagerModuleSpeedup: 1.5,
    tolerance: 2e-6,
    next: "native_eager_activation_tanh_or_gelu_tensor_route",
  }),
  Object.freeze({
    key: "activation_tanh_batched",
    shape: Object.freeze({ batch: 512, features: 256, activation: "tanh" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 23), [512, 256]),
    eager: (input) => input.tanh(),
    nativeEager: (output, input) => zgml.nativeEager.activationInto(output, input, { activation: "tanh" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.tanh()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: 1.5,
    minNativeEagerModuleSpeedup: 1.5,
    tolerance: 1e-6,
    next: "native_eager_activation_vectorized_gelu_tanh",
  }),
  Object.freeze({
    key: "reduce_sum_scalar_batched",
    shape: Object.freeze({ batch: 128, features: 64, op: "sum" }),
    outputLen: 1,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => input.sum(),
    nativeEager: (output, input) => zgml.nativeEager.reduceInto(output, input, { op: "sum" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.sum()),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-4,
    next: "native_eager_reduce_storage_slice",
  }),
  Object.freeze({
    key: "elementwise_lt_batched",
    shape: Object.freeze({ batch: 512, features: 256, op: "lt_scalar" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 13), [512, 256]),
    eager: (input) => input.lt(0.125),
    nativeEager: (output, input) => zgml.nativeEager.elementwiseInto(output, input, new Float32Array([0.125]), { op: "lt" }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.lt(0.125)),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerModuleSpeedup: 0.9,
    tolerance: 0,
    next: "native_eager_comparison_storage_slice",
  }),
  Object.freeze({
    key: "clamp_batched",
    shape: Object.freeze({ batch: 512, features: 256, op: "clamp" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 13), [512, 256]),
    eager: (input) => input.clamp(-0.25, 0.25),
    nativeEager: (output, input) => zgml.nativeEager.clampInto(output, input, { min: -0.25, max: 0.25 }),
    nativeEagerModule: (input) => zgml.noGrad(() => input.clamp(-0.25, 0.25)),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerSpeedup: runtime === "bun" ? 0.25 : 1,
    minNativeEagerModuleSpeedup: 0.95,
    tolerance: 1e-6,
    next: "native_eager_clamp_storage_slice",
  }),
  Object.freeze({
    key: "where_batched",
    shape: Object.freeze({ batch: 512, features: 256, op: "where_scalar" }),
    outputLen: 512 * 256,
    input: () => zgml.tensor(values(512 * 256, 13), [512, 256]),
    eager: (input) => input.lt(0).where(input, 0),
    nativeEager: (output, input) => {
      const condition = input.lt(0);
      return zgml.nativeEager.whereInto(output, condition, input, new Float32Array([0]));
    },
    nativeEagerModule: (input) => zgml.noGrad(() => input.lt(0).where(input, 0)),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    minNativeEagerModuleSpeedup: 0.95,
    tolerance: 1e-6,
    next: "native_eager_where_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_gelu_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_gelu" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddActivationEager(input, geluWeights, geluBias, 128, 64, 64, "gelu"),
    nativeEager: (output, input) => zgml.nativeEager.linearActivationInto(output, input, geluWeightTensor, {
      bias: geluBiasTensor,
      activation: "gelu",
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearGeluModel.forward(input)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .gelu(),
      {
        weights: new Float32Array(geluWeights),
        bias: new Float32Array(geluBias),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-4,
    next: "native_eager_fused_matmul_add_gelu_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_relu_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_relu" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddActivationEager(input, reluWeights, reluBias, 128, 64, 64, "relu"),
    nativeEager: (output, input) => zgml.nativeEager.linearActivationInto(output, input, reluWeightTensor, {
      bias: reluBiasTensor,
      activation: "relu",
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearReluModel.forward(input)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .relu(),
      {
        weights: new Float32Array(reluWeights),
        bias: new Float32Array(reluBias),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_fused_matmul_add_relu_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_silu_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_silu" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddActivationEager(input, siluWeights, siluBias, 128, 64, 64, "silu"),
    nativeEager: (output, input) => zgml.nativeEager.linearActivationInto(output, input, siluWeightTensor, {
      bias: siluBiasTensor,
      activation: "silu",
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearSiluModel.forward(input)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .silu(),
      {
        weights: new Float32Array(siluWeights),
        bias: new Float32Array(siluBias),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_fused_matmul_add_silu_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_sigmoid_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_sigmoid" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddActivationEager(input, sigmoidWeights, sigmoidBias, 128, 64, 64, "sigmoid"),
    nativeEager: (output, input) => zgml.nativeEager.linearActivationInto(output, input, sigmoidWeightTensor, {
      bias: sigmoidBiasTensor,
      activation: "sigmoid",
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearSigmoidModel.forward(input)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .sigmoid(),
      {
        weights: new Float32Array(sigmoidWeights),
        bias: new Float32Array(sigmoidBias),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_fused_matmul_add_sigmoid_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_tanh_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_tanh" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddActivationEager(input, tanhWeights, tanhBias, 128, 64, 64, "tanh"),
    nativeEager: (output, input) => zgml.nativeEager.linearActivationInto(output, input, tanhWeightTensor, {
      bias: tanhBiasTensor,
      activation: "tanh",
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => linearTanhModel.forward(input)),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .tanh(),
      {
        weights: new Float32Array(tanhWeights),
        bias: new Float32Array(tanhBias),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_fused_matmul_add_tanh_storage_slice",
  }),
  Object.freeze({
    key: "softmax_batched",
    shape: Object.freeze({ batch: 128, features: 64, op: "softmax" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 19), [128, 64]),
    eager: (input) => softmaxModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.softmaxInto(output, input, { dim: -1 }),
    nativeEagerModule: (input) => zgml.noGrad(() => softmaxModel.forward(input)),
    compiled: () => compiledInferenceHandle(softmaxBatchedModel(), [128, 64]),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_softmax_storage_slice",
  }),
  Object.freeze({
    key: "log_softmax_batched",
    shape: Object.freeze({ batch: 128, features: 64, op: "logSoftmax" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 19), [128, 64]),
    eager: (input) => logSoftmaxModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.logSoftmaxInto(output, input, { dim: -1 }),
    nativeEagerModule: (input) => zgml.noGrad(() => logSoftmaxModel.forward(input)),
    compiled: () => compiledInferenceHandle(logSoftmaxBatchedModel(), [128, 64]),
    eagerIterations: 100,
    nativeEagerIterations: 1000,
    nativeEagerModuleIterations: 1000,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_log_softmax_storage_slice",
  }),
  Object.freeze({
    key: "conv2d_batched",
    shape: Object.freeze({ batch: 2, inChannels: 1, height: 64, width: 64, outChannels: 1, kernel: 3 }),
    outputLen: 2 * 1 * 62 * 62,
    input: () => zgml.tensor(values(2 * 1 * 64 * 64, 11), [2, 1, 64, 64]),
    eager: (input) => conv2dModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.conv2dInto(output, input, conv2dWeightTensor, {
      bias: conv2dBiasTensor,
      outH: 62,
      outW: 62,
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => conv2dModel.forward(input)),
    compiled: () => compiledInferenceHandle(conv2dBatchedModel(), [2, 1, 64, 64]),
    eagerIterations: 50,
    nativeEagerIterations: 200,
    nativeEagerModuleIterations: 200,
    compiledIterations: 200,
    minNativeEagerSpeedup: runtime === "bun" ? 0.8 : 0.85,
    tolerance: 1e-4,
    next: "native_eager_conv2d_storage_slice",
  }),
  Object.freeze({
    key: "max_pool2d_batched",
    shape: Object.freeze({ batch: 2, channels: 2, height: 256, width: 256, kernel: 2 }),
    outputLen: 2 * 2 * 128 * 128,
    input: () => zgml.tensor(values(2 * 2 * 256 * 256, 13), [2, 2, 256, 256]),
    eager: (input) => maxPool2dModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.pool2dInto(output, input, {
      op: "max",
      kernelH: 2,
      kernelW: 2,
      strideH: 2,
      strideW: 2,
      outH: 128,
      outW: 128,
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => maxPool2dModel.forward(input)),
    compiled: () => compiledInferenceHandle(maxPool2dBatchedModel(), [2, 2, 256, 256]),
    eagerIterations: 50,
    nativeEagerIterations: 200,
    nativeEagerModuleIterations: 200,
    compiledIterations: 200,
    minNativeEagerSpeedup: 0.9,
    tolerance: 1e-5,
    next: "native_eager_max_pool2d_storage_slice",
  }),
  Object.freeze({
    key: "avg_pool2d_batched",
    shape: Object.freeze({ batch: 2, channels: 2, height: 256, width: 256, kernel: 2 }),
    outputLen: 2 * 2 * 128 * 128,
    input: () => zgml.tensor(values(2 * 2 * 256 * 256, 17), [2, 2, 256, 256]),
    eager: (input) => avgPool2dModel.forward(input),
    nativeEager: (output, input) => zgml.nativeEager.pool2dInto(output, input, {
      op: "avg",
      kernelH: 2,
      kernelW: 2,
      strideH: 2,
      strideW: 2,
      outH: 128,
      outW: 128,
      countIncludePad: true,
    }),
    nativeEagerModule: (input) => zgml.noGrad(() => avgPool2dModel.forward(input)),
    compiled: () => compiledInferenceHandle(avgPool2dBatchedModel(), [2, 2, 256, 256]),
    eagerIterations: 50,
    nativeEagerIterations: 200,
    nativeEagerModuleIterations: 200,
    compiledIterations: 200,
    minNativeEagerSpeedup: 0.4,
    minNativeEagerModuleSpeedup: 0.95,
    tolerance: 1e-5,
    next: "native_eager_avg_pool2d_storage_slice",
  }),
]);

const rows = Object.freeze(gapSpecs.map(benchGap));
const result = Object.freeze({
  schema: "zgml.native-eager-gap.v1",
  runtime,
  entry: runtimeEntry,
  nativeFreshness,
  config: Object.freeze({
    minTimingMs,
    minNativeEagerSpeedup,
    rows: gapSpecs.map((spec) => spec.key),
  }),
  status: "gap-measured",
  rows,
  next: "native_eager_linear_or_matmul_storage_slice",
});
let artifactPath = null;
if (writeArtifact) {
  const resolvedArtifactDir = resolve(root, artifactDir);
  mkdirSync(resolvedArtifactDir, { recursive: true });
  artifactPath = join(resolvedArtifactDir, `native-eager-${timestampForArtifact()}-${process.pid}.json`);
  writeFileSync(artifactPath, `${JSON.stringify(result, null, 2)}\n`);
}
process.stdout.write(`NATIVE_EAGER_GAP_JSON ${JSON.stringify(result)}\n`);
for (const row of rows) {
  const nativeEager = row.nativeEagerIntoMs === null
    ? "native_eager_into=n/a"
    : `native_eager_into=${row.nativeEagerIntoMs}ms native_eager_speedup=${row.nativeEagerSpeedup}x`;
  const nativeEagerModule = row.nativeEagerModuleForwardMs === null
    ? "native_eager_module=n/a"
    : `native_eager_module=${row.nativeEagerModuleForwardMs}ms native_eager_module_speedup=${row.nativeEagerModuleSpeedup}x`;
  const compiled = row.compiledHotPath
    ? `prepared_execute_into=${row.preparedExecuteIntoMs}ms speedup=${row.nativeProgramSpeedup}x`
    : "prepared_execute_into=n/a speedup=n/a";
  process.stdout.write(`native eager gap: runtime=${runtime} ${row.key} eager=${row.eagerMs}ms ${nativeEager} ${nativeEagerModule} ${compiled} next=${row.next}\n`);
}
if (artifactPath) {
  process.stdout.write(`native eager artifact: ${artifactPath}\n`);
}
