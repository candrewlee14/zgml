"use strict";

const { existsSync } = require("node:fs");
const { join, resolve } = require("node:path");
const { verifyFreshNativeLibrary } = require("./native_freshness.cjs");

const root = resolve(__dirname, "..");
const nodeEntry = join(root, "dist", "node.cjs");
if (!existsSync(nodeEntry)) {
  throw new Error("native eager gap check requires dist/node.cjs; run npm run build:package first");
}
const nativeFreshness = verifyFreshNativeLibrary({
  root,
  label: "native eager gap check",
  allowStaleEnv: "BENCH_NATIVE_EAGER_ALLOW_STALE_NATIVE",
});

const { zgml } = require(nodeEntry);
const minTimingMs = Number(process.env.BENCH_NATIVE_EAGER_MIN_TIMING_MS || "8");
if (!Number.isFinite(minTimingMs) || minTimingMs <= 0) {
  throw new Error(`BENCH_NATIVE_EAGER_MIN_TIMING_MS must be positive, got ${process.env.BENCH_NATIVE_EAGER_MIN_TIMING_MS}`);
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

function lazyMatmulAddGeluEager(input, weightValues, biasValues, batch, inFeatures, outFeatures) {
  const out = new Float32Array(batch * outFeatures);
  for (let row = 0; row < batch; row += 1) {
    for (let col = 0; col < outFeatures; col += 1) {
      let sum = biasValues[col];
      for (let feature = 0; feature < inFeatures; feature += 1) {
        sum += input.data[row * inFeatures + feature] * weightValues[feature * outFeatures + col];
      }
      out[row * outFeatures + col] = geluScalar(sum);
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
    requireCompiledHotPath(spec.key, compiled.session, input, output);
    const eagerOutput = spec.eager(input);
    const eagerData = eagerOutput.data ?? eagerOutput;
    const compiledOutput = compiled.into(output, input);
    if (compiledOutput !== output) {
      throw new Error(`${spec.key} expected compiled output to reuse caller output`);
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
    const speedup = eagerMs / preparedMs;
    return Object.freeze({
      schema: "zgml.native-eager-gap.v1",
      key: spec.key,
      shape: Object.freeze(spec.shape),
      eagerMs: round(eagerMs),
      preparedExecuteIntoMs: round(preparedMs),
      nativeProgramSpeedup: round(speedup),
      maxAbsDiff: round(diff),
      status: "gap-measured",
      next: spec.next,
    });
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
  const program = graph.compile({ backend: "cpu", inputShape });
  const session = program.bind(bindings);
  return Object.freeze({
    session,
    into(output, input) {
      return session.executeInto(output, { input });
    },
    dispose() {
      session.dispose();
      program.dispose();
    },
  });
}

function linearBatchedModel() {
  return new zgml.nn.Sequential(
    new zgml.nn.Linear(64, 32, {
      weights: values(64 * 32, 64),
      bias: values(32, 32),
    }),
  );
}

const linearModel = linearBatchedModel();

const gapSpecs = Object.freeze([
  Object.freeze({
    key: "linear_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 32 }),
    outputLen: 128 * 32,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => linearModel.forward(input),
    compiled: () => compiledInferenceHandle(linearBatchedModel(), [128, 64]),
    eagerIterations: 100,
    compiledIterations: 1000,
    tolerance: 1e-5,
    next: "native_eager_linear_or_matmul_storage_slice",
  }),
  Object.freeze({
    key: "lazy_matmul_add_gelu_batched",
    shape: Object.freeze({ batch: 128, inFeatures: 64, outFeatures: 64, fusedOps: "matmul_add_gelu" }),
    outputLen: 128 * 64,
    input: () => zgml.tensor(values(128 * 64, 13), [128, 64]),
    eager: (input) => lazyMatmulAddGeluEager(input, values(64 * 64, 32), values(64, 64), 128, 64, 64),
    compiled: () => compiledLazyHandle(
      zgml.lazy.input([128, 64])
        .matmul(zgml.lazy.parameter([64, 64], "w"))
        .add(zgml.lazy.parameter([64], "b"))
        .gelu(),
      {
        weights: new Float32Array(values(64 * 64, 32)),
        bias: new Float32Array(values(64, 64)),
      },
      [128, 64],
    ),
    eagerIterations: 100,
    compiledIterations: 1000,
    tolerance: 1e-4,
    next: "native_eager_fused_matmul_add_gelu_storage_slice",
  }),
]);

const rows = Object.freeze(gapSpecs.map(benchGap));
const result = Object.freeze({
  schema: "zgml.native-eager-gap.v1",
  nativeFreshness,
  status: "gap-measured",
  rows,
  next: "native_eager_linear_or_matmul_storage_slice",
});
process.stdout.write(`NATIVE_EAGER_GAP_JSON ${JSON.stringify(result)}\n`);
for (const row of rows) {
  process.stdout.write(`native eager gap: ${row.key} eager=${row.eagerMs}ms prepared_execute_into=${row.preparedExecuteIntoMs}ms speedup=${row.nativeProgramSpeedup}x next=${row.next}\n`);
}
