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

const batch = 128;
const inFeatures = 64;
const outFeatures = 32;
const input = zgml.tensor(values(batch * inFeatures, 13), [batch, inFeatures]);
const model = new zgml.nn.Sequential(
  new zgml.nn.Linear(inFeatures, outFeatures, {
    weights: values(inFeatures * outFeatures, 64),
    bias: values(outFeatures, 32),
  }),
);

const eager = model.forward(input);
const fast = zgml.compileForInference(model, { backend: "cpu", inputShape: [batch, inFeatures] });
try {
  const output = new Float32Array(batch * outFeatures);
  const compatibility = fast.session.requireHotStepParams({ input, output });
  const plan = fast.session.hotPathPlan({ input, output });
  fast.into(output, input);
  const diff = maxAbsDiff(eager.data, output);
  if (diff > 1e-5) {
    throw new Error(`native eager gap compiled parity failed: max_abs_diff=${diff}`);
  }
  if (
    compatibility.hotPath !== true ||
    compatibility.runtimeOutputAllocationFree !== true ||
    compatibility.readbackFree !== true ||
    plan.hotPath !== true
  ) {
    throw new Error(`native eager gap expected compiled hot path: ${JSON.stringify({ compatibility, plan })}`);
  }

  const eagerMs = bench(() => {
    model.forward(input);
  }, 100);
  const preparedMs = bench(() => {
    fast.into(output, input);
  }, 1000);
  const speedup = eagerMs / preparedMs;
  const row = Object.freeze({
    schema: "zgml.native-eager-gap.v1",
    key: "linear_batched",
    shape: Object.freeze({ batch, inFeatures, outFeatures }),
    eagerMs: round(eagerMs),
    preparedExecuteIntoMs: round(preparedMs),
    nativeProgramSpeedup: round(speedup),
    maxAbsDiff: round(diff),
    nativeFreshness,
    status: "gap-measured",
    next: "native_eager_linear_or_matmul_storage_slice",
  });
  process.stdout.write(`NATIVE_EAGER_GAP_JSON ${JSON.stringify(row)}\n`);
  process.stdout.write(`native eager gap: ${row.key} eager=${row.eagerMs}ms prepared_execute_into=${row.preparedExecuteIntoMs}ms speedup=${row.nativeProgramSpeedup}x next=${row.next}\n`);
} finally {
  fast.dispose();
}
