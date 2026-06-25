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
const minTimingMs = Number(process.env.BENCH_NATIVE_EAGER_MIN_TIMING_MS || "8");
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
  const compiled = spec.compiled();
  try {
    const output = new Float32Array(spec.outputLen);
    const nativeEagerOutput = new Float32Array(spec.outputLen);
    requireCompiledHotPath(spec.key, compiled.session, input, output);
    const eagerOutput = spec.eager(input);
    const eagerData = eagerOutput.data ?? eagerOutput;
    const nativeEagerResult = typeof spec.nativeEager === "function" ? spec.nativeEager(nativeEagerOutput, input) : null;
    if (nativeEagerResult && nativeEagerResult !== nativeEagerOutput) {
      throw new Error(`${spec.key} expected native eager output to reuse caller output`);
    }
    const nativeEagerModuleOutput = typeof spec.nativeEagerModule === "function" ? spec.nativeEagerModule(input) : null;
    const nativeEagerModuleData = nativeEagerModuleOutput ? (nativeEagerModuleOutput.data ?? nativeEagerModuleOutput) : null;
    const compiledOutput = compiled.into(output, input);
    if (compiledOutput !== output) {
      throw new Error(`${spec.key} expected compiled output to reuse caller output`);
    }
    const nativeEagerDiff = nativeEagerResult ? maxAbsDiff(eagerData, nativeEagerOutput) : null;
    if (nativeEagerDiff !== null && nativeEagerDiff > spec.tolerance) {
      throw new Error(`${spec.key} native eager parity failed: max_abs_diff=${nativeEagerDiff}`);
    }
    const nativeEagerModuleDiff = nativeEagerModuleData ? maxAbsDiff(eagerData, nativeEagerModuleData) : null;
    if (nativeEagerModuleDiff !== null && nativeEagerModuleDiff > spec.tolerance) {
      throw new Error(`${spec.key} native eager module parity failed: max_abs_diff=${nativeEagerModuleDiff}`);
    }
    const diff = maxAbsDiff(eagerData, output);
    if (diff > spec.tolerance) {
      throw new Error(`${spec.key} compiled parity failed: max_abs_diff=${diff}`);
    }

    const eagerMs = bench(() => {
      spec.eager(input);
    }, spec.eagerIterations);
    const preparedMs = bench(() => {
      compiled.into(output, input);
    }, spec.compiledIterations);
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
    if (nativeEagerMs !== null && eagerMs / nativeEagerMs < minNativeEagerSpeedup) {
      throw new Error(`${spec.key} native eager speedup ${eagerMs / nativeEagerMs}x below ${minNativeEagerSpeedup}x`);
    }
    if (nativeEagerModuleMs !== null && eagerMs / nativeEagerModuleMs < minNativeEagerSpeedup) {
      throw new Error(`${spec.key} native eager module speedup ${eagerMs / nativeEagerModuleMs}x below ${minNativeEagerSpeedup}x`);
    }
    const speedup = eagerMs / preparedMs;
    const row = {
      schema: "zgml.native-eager-gap.v1",
      key: spec.key,
      shape: Object.freeze(spec.shape),
      eagerMs: round(eagerMs),
      nativeEagerIntoMs: nativeEagerMs === null ? null : round(nativeEagerMs),
      nativeEagerSpeedup: nativeEagerMs === null ? null : round(eagerMs / nativeEagerMs),
      nativeEagerModuleForwardMs: nativeEagerModuleMs === null ? null : round(nativeEagerModuleMs),
      nativeEagerModuleSpeedup: nativeEagerModuleMs === null ? null : round(eagerMs / nativeEagerModuleMs),
      preparedExecuteIntoMs: round(preparedMs),
      nativeProgramSpeedup: round(speedup),
      nativeEagerMaxAbsDiff: nativeEagerDiff === null ? null : round(nativeEagerDiff),
      nativeEagerModuleMaxAbsDiff: nativeEagerModuleDiff === null ? null : round(nativeEagerModuleDiff),
      maxAbsDiff: round(diff),
      status: "gap-measured",
      next: spec.next,
    };
    return Object.freeze(row);
  } finally {
    compiled.dispose();
  }
}

function compiledInferenceHandle(model, inputShape) {
  const fast = zgml.compileForInference(model, { backend: "cpu", inputShape });
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
  const fast = zgml.compileForInference(graph, { backend: "cpu", inputShape }, bindings);
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

const linearWeights = values(64 * 32, 64);
const linearBias = values(32, 32);
const linearWeightTensor = zgml.tensor(linearWeights, [64, 32]);
const linearBiasTensor = zgml.tensor(linearBias, [32]);
const linearModel = linearBatchedModel();
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
  process.stdout.write(`native eager gap: runtime=${runtime} ${row.key} eager=${row.eagerMs}ms ${nativeEager} ${nativeEagerModule} prepared_execute_into=${row.preparedExecuteIntoMs}ms speedup=${row.nativeProgramSpeedup}x next=${row.next}\n`);
}
if (artifactPath) {
  process.stdout.write(`native eager artifact: ${artifactPath}\n`);
}
