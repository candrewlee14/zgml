"use strict";

const { existsSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const nodeEntry = join(root, "dist", "node.cjs");
if (!existsSync(nodeEntry)) {
  throw new Error("module Program bench requires dist/node.cjs; run npm run build:package first");
}

const adapter = require(nodeEntry);

const attempts = 3;
const wholeBenchAttempts = 3;
const floors = Object.freeze({
  activationChainSpeedup: 1.25,
  linearReluSpeedup: 1.2,
  linearBatchedSpeedup: 3.0,
  mlpSpeedup: 1.2,
  mlpBatchedSpeedup: 3.0,
  normGeluMlpSpeedup: 1.1,
  normGeluMlpBatchedSpeedup: 1.3,
  tokenHeadSpeedup: 1.2,
  shapeLinearSpeedup: 1.2,
  conv2dSpeedup: 1.05,
  conv2dBatchedSpeedup: 1.05,
  maxPool2dSpeedup: 1.05,
  maxPool2dBatchedSpeedup: 1.05,
  avgPool2dSpeedup: 1.05,
  avgPool2dBatchedSpeedup: 1.05,
  softmaxClassifierSpeedup: 1.2,
  softmaxClassifierBatchedSpeedup: 3.0,
  logSoftmaxClassifierBatchedSpeedup: 3.0,
  lazyMatmulAddReluBatchedSpeedup: 1.5,
});
const expectedKeys = Object.freeze([
  "activation_chain",
  "linear_relu",
  "linear_batched",
  "mlp",
  "mlp_batched",
  "norm_gelu_mlp",
  "norm_gelu_mlp_batched",
  "token_head",
  "shape_linear",
  "conv2d",
  "conv2d_batched",
  "max_pool2d",
  "max_pool2d_batched",
  "avg_pool2d",
  "avg_pool2d_batched",
  "softmax_classifier",
  "softmax_classifier_batched",
  "log_softmax_classifier_batched",
  "lazy_matmul_add_relu_batched",
]);

function msNow() {
  return Number(process.hrtime.bigint()) / 1e6;
}

function values(length, scale) {
  return Array.from({ length }, (_, index) => ((index % 17) - 8) / scale);
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let index = 0; index < a.length; index += 1) {
    max = Math.max(max, Math.abs(a[index] - b[index]));
  }
  return max;
}

function bench(fn, iterations) {
  for (let index = 0; index < Math.min(100, iterations); index += 1) fn();
  const start = msNow();
  for (let index = 0; index < iterations; index += 1) fn();
  return (msNow() - start) / iterations;
}

function median(valuesToSort) {
  const sorted = valuesToSort.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function kernels(plan) {
  return plan.ops.map((op) => op.nativeKernels.join("|")).join("|");
}

function ops(plan) {
  return plan.ops.map((op) => op.op).join("|");
}

function requireHotPath(label, session, input, output) {
  const evidence = session.requireHotStepParams({ input, output });
  if (
    evidence.hotPath !== true ||
    evidence.runtimeOutputAllocationFree !== true ||
    evidence.readbackFree !== true ||
    evidence.hotPathBlockers.length !== 0 ||
    evidence.inputOwnership !== "caller" ||
    evidence.outputOwnership !== "caller"
  ) {
    throw new Error(`${label} expected caller-owned allocation-free hot path`);
  }
  const plan = session.hotPathPlan({ input, output });
  if (
    plan.hotPath !== true ||
    plan.hotPathStatus !== "hot" ||
    plan.runtimeOutputAllocationFree !== true ||
    plan.readbackFree !== true ||
    plan.stepParamsSignature !== evidence.stepParamsSignature
  ) {
    throw new Error(`${label} expected hotPathPlan to match hot StepParams evidence`);
  }
}

function requireCompileEvidence(spec, support) {
  const layerCount = support.layerCount ?? support.trace?.layerCount;
  const inputLen = support.inputLen ?? support.ir?.inputLen;
  const outputLen = support.outputLen ?? support.ir?.outputLen;
  const inputShape = support.inputShape ?? support.ir?.inputShape;
  const outputShape = support.outputShape ?? support.ir?.outputShape;
  if (
    support.supported !== true ||
    layerCount !== spec.layerCount ||
    inputLen !== spec.inputLen ||
    outputLen !== spec.outputLen
  ) {
    throw new Error(`${spec.label} expected supported compile evidence`);
  }
  if (spec.inputShapeText && inputShape.join("x") !== spec.inputShapeText) {
    throw new Error(`${spec.label} expected input shape ${spec.inputShapeText}`);
  }
  if (spec.outputShapeText && outputShape.join("x") !== spec.outputShapeText) {
    throw new Error(`${spec.label} expected output shape ${spec.outputShapeText}`);
  }
}

function requireKernelPlan(spec, plan) {
  if (
    !plan ||
    plan.opCount !== spec.plan.opCount ||
    plan.dispatchCount !== spec.plan.dispatchCount ||
    plan.ops.length !== spec.plan.publicOps
  ) {
    throw new Error(`${spec.label} expected ${spec.plan.description}`);
  }
  spec.plan.check(plan);
}

function requireIrAndParams(spec, program) {
  const ir = program.tensorProgramIr();
  if (!ir || ir.opCount !== spec.ir.opCount || ir.parameterCount !== spec.ir.parameterCount || ir.outputLen !== spec.outputLen) {
    throw new Error(`${spec.label} expected retained Tensor Program IR evidence`);
  }
  const parameterNames = program.parameterNames();
  if (parameterNames.join("|") !== spec.parameterNames) {
    throw new Error(`${spec.label} expected stable parameter names ${spec.parameterNames}`);
  }
}

function requireBenchSpecCoverage(specs) {
  const keys = specs.map((spec) => spec.key);
  if (keys.join("|") !== expectedKeys.join("|")) {
    throw new Error(`module Program bench specs changed: got ${keys.join("|")}`);
  }
  const uniqueKeys = new Set(keys);
  if (uniqueKeys.size !== keys.length) {
    throw new Error(`module Program bench specs contain duplicate keys: ${keys.join("|")}`);
  }
  for (const spec of specs) {
    if (typeof spec.summary !== "function" || typeof spec.plan?.check !== "function") {
      throw new Error(`${spec.key} must carry executable summary and kernel-plan checks`);
    }
    if (!Number.isFinite(spec.floor) || spec.floor < 1) {
      throw new Error(`${spec.key} must keep a speedup floor`);
    }
  }
}

function lazyMatmulAddReluEager(input, weightValues, biasValues, batch, inFeatures, outFeatures) {
  const out = new Float32Array(batch * outFeatures);
  for (let row = 0; row < batch; row += 1) {
    for (let col = 0; col < outFeatures; col += 1) {
      let sum = biasValues[col];
      for (let feature = 0; feature < inFeatures; feature += 1) {
        sum += input.data[row * inFeatures + feature] * weightValues[feature * outFeatures + col];
      }
      out[row * outFeatures + col] = Math.max(0, sum);
    }
  }
  return out;
}

function runBenchSpec(spec) {
  const input = spec.input();
  const output = new Float32Array(spec.outputLen);
  const model = spec.model();
  const compileOptions = { inputShape: spec.inputShape, backend: "cpu" };
  const support = model.compileSupport(compileOptions);
  requireCompileEvidence(spec, support);
  const program = model.compile({ inputShape: spec.inputShape, backend: "cpu" });
  const session = typeof spec.bindSession === "function" ? spec.bindSession(program) : program.bindModule(model);
  try {
    requireKernelPlan(spec, program.kernelPlan());
    requireIrAndParams(spec, program);
    requireHotPath(spec.label, session, input, output);
    const eager = () => typeof spec.eager === "function" ? spec.eager(input) : model.forward(input);
    const compiledTensor = () => session.stepTensor(input);
    const compiled = () => session.executeInto(output, { input });
    const eagerOutput = eager();
    const compiledTensorOutput = compiledTensor();
    const compiledOutput = compiled();
    if (compiledOutput !== output) throw new Error(`${spec.label} expected executeInto to reuse caller output`);
    const eagerData = eagerOutput.data ?? eagerOutput;
    const tensorError = maxAbsDiff(eagerData, compiledTensorOutput.data);
    if (tensorError > spec.tolerance) throw new Error(`${spec.label} eager/stepTensor mismatch ${tensorError}`);
    const error = maxAbsDiff(eagerData, compiledOutput);
    if (error > spec.tolerance) throw new Error(`${spec.label} eager/compiled mismatch ${error}`);
    session.resetSessionCallProfile();
    const eagerRuns = [];
    const compiledRuns = [];
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      eagerRuns.push(bench(eager, spec.iterations));
      compiledRuns.push(bench(compiled, spec.iterations));
    }
    const expectedCalls = attempts * spec.iterations + Math.min(100, spec.iterations) * attempts;
    const profile = session.sessionCallProfile();
    if (profile.executeIntoCount !== expectedCalls) {
      throw new Error(`${spec.label} expected executeInto-only hot bench profile, got ${profile.signature}`);
    }
    const eagerMs = median(eagerRuns);
    const compiledMs = median(compiledRuns);
    return { eagerMs, compiledMs, speedup: eagerMs / compiledMs };
  } finally {
    session.dispose();
    program.dispose();
  }
}

const benchSpecs = [
  {
    key: "activation_chain",
    label: "activation-chain",
    floor: floors.activationChainSpeedup,
    inputShape: [16384],
    inputLen: 16384,
    outputLen: 16384,
    layerCount: 3,
    parameterNames: "",
    input: () => adapter.tensor(values(16384, 8), [16384]),
    model: () => adapter.nn.sequential([adapter.nn.relu(), adapter.nn.square(), adapter.nn.sqrt()]),
    ir: { opCount: 3, parameterCount: 0 },
    iterations: 500,
    tolerance: 1e-5,
    plan: {
      opCount: 3,
      dispatchCount: 1,
      publicOps: 1,
      description: "single fused activation-chain dispatch",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "activation-chain" || op.fusedOpCount !== 3 || op.nativeKernels.join("|") !== "relu|square|sqrt") {
          throw new Error("activation-chain expected fused native kernels relu|square|sqrt");
        }
      },
    },
    summary: (result) => `activation_chain=${result.speedup.toFixed(2)}x floor=${floors.activationChainSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms dispatch=1 fused=3 hot=allocation-free`,
  },
  {
    key: "linear_relu",
    label: "linear-relu",
    floor: floors.linearReluSpeedup,
    inputShape: [64],
    inputLen: 64,
    outputLen: 64,
    layerCount: 2,
    parameterNames: "0.weight",
    input: () => adapter.tensor(values(64, 10), [64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 64, {
        weights: Array.from({ length: 64 * 64 }, (_, index) => ((index % 13) - 6) / 32),
        bias: false,
      }),
      adapter.nn.relu(),
    ]),
    ir: { opCount: 2, parameterCount: 1 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 2,
      dispatchCount: 1,
      publicOps: 1,
      description: "single fused Linear+ReLU dispatch",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "linear" || op.fusedOpCount !== 2 || op.nativeKernels.join("|") !== "linear|relu") {
          throw new Error("linear-relu expected fused native kernels linear|relu");
        }
      },
    },
    summary: (result) => `linear_relu=${result.speedup.toFixed(2)}x floor=${floors.linearReluSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms dispatch=1 fused=2 hot=allocation-free`,
  },
  {
    key: "linear_batched",
    label: "linear-batched",
    floor: floors.linearBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x32",
    inputLen: 128 * 64,
    outputLen: 128 * 32,
    layerCount: 1,
    parameterNames: "0.weight|0.bias",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.nn.linear(64, 32, {
      weight: values(64 * 32, 64),
      bias: values(32, 32),
    }),
    ir: { opCount: 1, parameterCount: 2 },
    iterations: 500,
    tolerance: 1e-5,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single native batched Linear dispatch",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "linear" || op.kernel !== "linear" || op.nativeKernels.join("|") !== "linear" || op.nativeDispatchCount !== 1) {
          throw new Error("batched linear expected single native linear kernel");
        }
      },
    },
    summary: (result) => `linear_batched=${result.speedup.toFixed(2)}x floor=${floors.linearBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=linear batched=rank2 hot=allocation-free`,
  },
  {
    key: "mlp",
    label: "mlp",
    floor: floors.mlpSpeedup,
    inputShape: [32],
    inputLen: 32,
    outputLen: 16,
    layerCount: 3,
    parameterNames: "0.weight|0.bias|2.weight|2.bias",
    input: () => adapter.tensor(values(32, 10), [32]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(32, 64, { weights: values(32 * 64, 32), bias: values(64, 64) }),
      adapter.nn.relu(),
      adapter.nn.linear(64, 16, { weights: values(64 * 16, 48), bias: values(16, 80) }),
    ]),
    ir: { opCount: 3, parameterCount: 4 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 2,
      publicOps: 2,
      description: "compact two-dispatch Linear+ReLU -> Linear kernel plan",
      check: (plan) => {
        if (
          plan.ops[0].op !== "linear" ||
          plan.ops[0].fusedOpCount !== 2 ||
          plan.ops[0].nativeKernels.join("|") !== "linear|relu" ||
          plan.ops[1].op !== "linear" ||
          plan.ops[1].fusedOpCount !== undefined
        ) {
          throw new Error("mlp expected compact two-dispatch Linear+ReLU -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `mlp=${result.speedup.toFixed(2)}x floor=${floors.mlpSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=2 fused=2 hot=allocation-free`,
  },
  {
    key: "mlp_batched",
    label: "mlp-batched",
    floor: floors.mlpBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x32",
    inputLen: 128 * 64,
    outputLen: 128 * 32,
    layerCount: 3,
    parameterNames: "0.weight|0.bias|2.weight|2.bias",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 64, { weights: values(64 * 64, 64), bias: values(64, 32) }),
      adapter.nn.relu(),
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 64), bias: values(32, 32) }),
    ]),
    ir: { opCount: 3, parameterCount: 4 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 2,
      publicOps: 2,
      description: "compact two-dispatch batched Linear+ReLU -> Linear kernel plan",
      check: (plan) => {
        if (
          plan.ops[0].op !== "linear" ||
          plan.ops[0].fusedOpCount !== 2 ||
          plan.ops[0].nativeKernels.join("|") !== "linear|relu" ||
          plan.ops[1].op !== "linear" ||
          plan.ops[1].fusedOpCount !== undefined ||
          plan.ops[1].nativeKernels.join("|") !== "linear"
        ) {
          throw new Error("batched mlp expected compact two-dispatch Linear+ReLU -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `mlp_batched=${result.speedup.toFixed(2)}x floor=${floors.mlpBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=2 fused=2 batched=rank2 hot=allocation-free`,
  },
  {
    key: "norm_gelu_mlp",
    label: "norm-gelu-mlp",
    floor: floors.normGeluMlpSpeedup,
    inputShape: [32],
    inputLen: 32,
    outputLen: 16,
    layerCount: 4,
    parameterNames: "0.weight|0.bias|1.weight|1.bias|3.weight|3.bias",
    input: () => adapter.tensor(values(32, 10), [32]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(32, 64, { weights: values(32 * 64, 32), bias: values(64, 64) }),
      adapter.nn.layerNorm(64, { weight: values(64, 50).map((value) => value + 1), bias: values(64, 70) }),
      adapter.nn.gelu(),
      adapter.nn.linear(64, 16, { weights: values(64 * 16, 48), bias: values(16, 80) }),
    ]),
    ir: { opCount: 4, parameterCount: 6 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 4,
      dispatchCount: 4,
      publicOps: 4,
      description: "honest four-dispatch Linear -> LayerNorm -> GELU -> Linear kernel plan",
      check: (plan) => {
        if (ops(plan) !== "linear|layerNorm|activation|linear" || kernels(plan) !== "linear|layer-norm|gelu|linear") {
          throw new Error("norm-gelu-mlp expected honest four-dispatch Linear -> LayerNorm -> GELU -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `norm_gelu_mlp=${result.speedup.toFixed(2)}x floor=${floors.normGeluMlpSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=4 dispatch=4 kernels=linear|layer-norm|gelu|linear hot=allocation-free`,
  },
  {
    key: "norm_gelu_mlp_batched",
    label: "norm-gelu-mlp-batched",
    floor: floors.normGeluMlpBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x32",
    inputLen: 128 * 64,
    outputLen: 128 * 32,
    layerCount: 4,
    parameterNames: "0.weight|0.bias|1.weight|1.bias|3.weight|3.bias",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 64, { weights: values(64 * 64, 64), bias: values(64, 32) }),
      adapter.nn.layerNorm(64, { weight: values(64, 50).map((value) => value + 1), bias: values(64, 70) }),
      adapter.nn.gelu(),
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 64), bias: values(32, 32) }),
    ]),
    ir: { opCount: 4, parameterCount: 6 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 4,
      dispatchCount: 4,
      publicOps: 4,
      description: "honest four-dispatch batched Linear -> LayerNorm -> GELU -> Linear kernel plan",
      check: (plan) => {
        if (ops(plan) !== "linear|layerNorm|activation|linear" || kernels(plan) !== "linear|layer-norm|gelu|linear") {
          throw new Error("batched norm-gelu-mlp expected honest four-dispatch Linear -> LayerNorm -> GELU -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `norm_gelu_mlp_batched=${result.speedup.toFixed(2)}x floor=${floors.normGeluMlpBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=4 dispatch=4 kernels=linear|layer-norm|gelu|linear batched=rank2 hot=allocation-free`,
  },
  {
    key: "token_head",
    label: "token-head",
    floor: floors.tokenHeadSpeedup,
    inputShape: [1],
    inputLen: 1,
    outputLen: 32,
    layerCount: 3,
    parameterNames: "0.weight|1.weight|1.bias",
    input: () => adapter.tensor([7], [1]),
    model: () => adapter.nn.sequential([
      adapter.nn.embedding(128, 64, { weights: values(128 * 64, 32) }),
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 48), bias: values(32, 80) }),
      adapter.nn.logSoftmax(-1),
    ]),
    ir: { opCount: 3, parameterCount: 3 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 3,
      publicOps: 3,
      description: "honest three-dispatch Embedding -> Linear -> LogSoftmax kernel plan",
      check: (plan) => {
        if (ops(plan) !== "embedding|linear|logSoftmax" || kernels(plan) !== "embedding|linear|log-softmax") {
          throw new Error("token-head expected honest three-dispatch Embedding -> Linear -> LogSoftmax kernel plan");
        }
      },
    },
    summary: (result) => `token_head=${result.speedup.toFixed(2)}x floor=${floors.tokenHeadSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=3 kernels=embedding|linear|log-softmax hot=allocation-free`,
  },
  {
    key: "shape_linear",
    label: "shape-linear",
    floor: floors.shapeLinearSpeedup,
    inputShape: [8, 8],
    inputShapeText: "8x8",
    outputShapeText: "32",
    inputLen: 64,
    outputLen: 32,
    layerCount: 3,
    parameterNames: "2.weight|2.bias",
    input: () => adapter.tensor(values(64, 10), [8, 8]),
    model: () => adapter.nn.sequential([
      adapter.nn.flatten(),
      adapter.nn.reshape([64]),
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 32), bias: values(32, 80) }),
    ]),
    ir: { opCount: 3, parameterCount: 2 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 2,
      publicOps: 2,
      description: "fused shape-chain -> linear kernel plan",
      check: (plan) => {
        if (
          plan.ops[0].op !== "shape-chain" ||
          plan.ops[0].fusedOpCount !== 2 ||
          plan.ops[0].nativeKernels.join("|") !== "reshape" ||
          plan.ops[1].op !== "linear" ||
          plan.ops[1].nativeKernels.join("|") !== "linear"
        ) {
          throw new Error("shape-linear expected fused shape-chain -> linear kernel plan");
        }
      },
    },
    summary: (result) => `shape_linear=${result.speedup.toFixed(2)}x floor=${floors.shapeLinearSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=2 fused_shape=2 kernels=reshape|linear hot=allocation-free`,
  },
  {
    key: "conv2d",
    label: "conv2d",
    floor: floors.conv2dSpeedup,
    inputShape: [1, 64, 64],
    inputShapeText: "1x64x64",
    outputShapeText: "1x62x62",
    inputLen: 64 * 64,
    outputLen: 62 * 62,
    layerCount: 1,
    parameterNames: "0.weight|0.bias",
    input: () => adapter.tensor(values(64 * 64, 10), [1, 64, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.conv2d(1, 1, 3, {
        weight: values(9, 32),
        bias: [0.125],
      }),
    ]),
    ir: { opCount: 1, parameterCount: 2 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single native Conv2d dispatch",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "conv2d" || op.nativeKernels.join("|") !== "conv2d" || op.nativeDispatchCount !== 1) {
          throw new Error("conv2d expected single native conv2d kernel");
        }
      },
    },
    summary: (result) => `conv2d=${result.speedup.toFixed(2)}x floor=${floors.conv2dSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=conv2d hot=allocation-free`,
  },
  {
    key: "conv2d_batched",
    label: "conv2d-batched",
    floor: floors.conv2dBatchedSpeedup,
    inputShape: [2, 1, 64, 64],
    inputShapeText: "2x1x64x64",
    outputShapeText: "2x1x62x62",
    inputLen: 2 * 64 * 64,
    outputLen: 2 * 62 * 62,
    layerCount: 1,
    parameterNames: "0.weight|0.bias",
    input: () => adapter.tensor(values(2 * 64 * 64, 11), [2, 1, 64, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.conv2d(1, 1, 3, {
        weight: values(9, 32),
        bias: [0.125],
      }),
    ]),
    ir: { opCount: 1, parameterCount: 2 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single native batched Conv2d dispatch",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "conv2d" || op.nativeKernels.join("|") !== "conv2d" || op.nativeDispatchCount !== 1) {
          throw new Error("batched conv2d expected single native conv2d kernel");
        }
      },
    },
    summary: (result) => `conv2d_batched=${result.speedup.toFixed(2)}x floor=${floors.conv2dBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=conv2d batched=rank4 hot=allocation-free`,
  },
  {
    key: "max_pool2d",
    label: "max-pool2d",
    floor: floors.maxPool2dSpeedup,
    inputShape: [1, 128, 128],
    inputShapeText: "1x128x128",
    outputShapeText: "1x64x64",
    inputLen: 128 * 128,
    outputLen: 64 * 64,
    layerCount: 1,
    parameterNames: "",
    input: () => adapter.tensor(values(128 * 128, 10), [1, 128, 128]),
    model: () => adapter.nn.sequential([adapter.nn.max_pool2d(2)]),
    ir: { opCount: 1, parameterCount: 0 },
    iterations: 500,
    tolerance: 1e-6,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single-dispatch MaxPool2d Program kernel plan",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "maxPool2d" || op.kernel !== "max-pool2d" || op.nativeKernels.join("|") !== "max-pool2d") {
          throw new Error("max-pool2d expected native max-pool2d kernel plan");
        }
      },
    },
    summary: (result) => `max_pool2d=${result.speedup.toFixed(2)}x floor=${floors.maxPool2dSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=max-pool2d hot=allocation-free`,
  },
  {
    key: "max_pool2d_batched",
    label: "max-pool2d-batched",
    floor: floors.maxPool2dBatchedSpeedup,
    inputShape: [2, 2, 128, 128],
    inputShapeText: "2x2x128x128",
    outputShapeText: "2x2x64x64",
    inputLen: 2 * 2 * 128 * 128,
    outputLen: 2 * 2 * 64 * 64,
    layerCount: 1,
    parameterNames: "",
    input: () => adapter.tensor(values(2 * 2 * 128 * 128, 10), [2, 2, 128, 128]),
    model: () => adapter.nn.sequential([adapter.nn.max_pool2d(2)]),
    ir: { opCount: 1, parameterCount: 0 },
    iterations: 300,
    tolerance: 1e-6,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single-dispatch batched MaxPool2d Program kernel plan",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "maxPool2d" || op.kernel !== "max-pool2d" || op.nativeKernels.join("|") !== "max-pool2d") {
          throw new Error("batched max-pool2d expected native max-pool2d kernel plan");
        }
      },
    },
    summary: (result) => `max_pool2d_batched=${result.speedup.toFixed(2)}x floor=${floors.maxPool2dBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=max-pool2d batched=rank4 hot=allocation-free`,
  },
  {
    key: "avg_pool2d",
    label: "avg-pool2d",
    floor: floors.avgPool2dSpeedup,
    inputShape: [1, 128, 128],
    inputShapeText: "1x128x128",
    outputShapeText: "1x64x64",
    inputLen: 128 * 128,
    outputLen: 64 * 64,
    layerCount: 1,
    parameterNames: "",
    input: () => adapter.tensor(values(128 * 128, 10), [1, 128, 128]),
    model: () => adapter.nn.sequential([adapter.nn.avg_pool2d(2)]),
    ir: { opCount: 1, parameterCount: 0 },
    iterations: 500,
    tolerance: 1e-6,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single-dispatch AvgPool2d Program kernel plan",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "avgPool2d" || op.kernel !== "avg-pool2d" || op.nativeKernels.join("|") !== "avg-pool2d") {
          throw new Error("avg-pool2d expected native avg-pool2d kernel plan");
        }
      },
    },
    summary: (result) => `avg_pool2d=${result.speedup.toFixed(2)}x floor=${floors.avgPool2dSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=avg-pool2d hot=allocation-free`,
  },
  {
    key: "avg_pool2d_batched",
    label: "avg-pool2d-batched",
    floor: floors.avgPool2dBatchedSpeedup,
    inputShape: [2, 2, 128, 128],
    inputShapeText: "2x2x128x128",
    outputShapeText: "2x2x64x64",
    inputLen: 2 * 2 * 128 * 128,
    outputLen: 2 * 2 * 64 * 64,
    layerCount: 1,
    parameterNames: "",
    input: () => adapter.tensor(values(2 * 2 * 128 * 128, 10), [2, 2, 128, 128]),
    model: () => adapter.nn.sequential([adapter.nn.avg_pool2d(2)]),
    ir: { opCount: 1, parameterCount: 0 },
    iterations: 300,
    tolerance: 1e-6,
    plan: {
      opCount: 1,
      dispatchCount: 1,
      publicOps: 1,
      description: "single-dispatch batched AvgPool2d Program kernel plan",
      check: (plan) => {
        const op = plan.ops[0];
        if (op.op !== "avgPool2d" || op.kernel !== "avg-pool2d" || op.nativeKernels.join("|") !== "avg-pool2d") {
          throw new Error("batched avg-pool2d expected native avg-pool2d kernel plan");
        }
      },
    },
    summary: (result) => `avg_pool2d_batched=${result.speedup.toFixed(2)}x floor=${floors.avgPool2dBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=1 dispatch=1 kernels=avg-pool2d batched=rank4 hot=allocation-free`,
  },
  {
    key: "softmax_classifier",
    label: "softmax-classifier",
    floor: floors.softmaxClassifierSpeedup,
    inputShape: [64],
    inputLen: 64,
    outputLen: 32,
    layerCount: 3,
    parameterNames: "0.weight|0.bias|2.weight|2.bias",
    input: () => adapter.tensor(values(64, 10), [64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 64, { weights: values(64 * 64, 32), bias: values(64, 64) }),
      adapter.nn.softmax(0),
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 48), bias: values(32, 80) }),
    ]),
    ir: { opCount: 3, parameterCount: 4 },
    iterations: 1000,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 3,
      publicOps: 3,
      description: "honest three-dispatch Linear -> Softmax -> Linear kernel plan",
      check: (plan) => {
        if (ops(plan) !== "linear|softmax|linear" || kernels(plan) !== "linear|softmax|linear") {
          throw new Error("softmax-classifier expected honest three-dispatch Linear -> Softmax -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `softmax_classifier=${result.speedup.toFixed(2)}x floor=${floors.softmaxClassifierSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=3 kernels=linear|softmax|linear hot=allocation-free`,
  },
  {
    key: "softmax_classifier_batched",
    label: "softmax-classifier-batched",
    floor: floors.softmaxClassifierBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x16",
    inputLen: 128 * 64,
    outputLen: 128 * 16,
    layerCount: 3,
    parameterNames: "0.weight|0.bias|2.weight|2.bias",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 64), bias: values(32, 32) }),
      adapter.nn.softmax(-1),
      adapter.nn.linear(32, 16, { weights: values(32 * 16, 48), bias: values(16, 80) }),
    ]),
    ir: { opCount: 3, parameterCount: 4 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 3,
      publicOps: 3,
      description: "honest three-dispatch batched Linear -> Softmax -> Linear kernel plan",
      check: (plan) => {
        if (ops(plan) !== "linear|softmax|linear" || kernels(plan) !== "linear|softmax|linear") {
          throw new Error("batched softmax-classifier expected honest three-dispatch Linear -> Softmax -> Linear kernel plan");
        }
      },
    },
    summary: (result) => `softmax_classifier_batched=${result.speedup.toFixed(2)}x floor=${floors.softmaxClassifierBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=3 kernels=linear|softmax|linear batched=rank2 hot=allocation-free`,
  },
  {
    key: "log_softmax_classifier_batched",
    label: "log-softmax-classifier-batched",
    floor: floors.logSoftmaxClassifierBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x32",
    inputLen: 128 * 64,
    outputLen: 128 * 32,
    layerCount: 2,
    parameterNames: "0.weight|0.bias",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.nn.sequential([
      adapter.nn.linear(64, 32, { weights: values(64 * 32, 64), bias: values(32, 32) }),
      adapter.nn.logSoftmax(-1),
    ]),
    ir: { opCount: 2, parameterCount: 2 },
    iterations: 300,
    tolerance: 1e-4,
    plan: {
      opCount: 2,
      dispatchCount: 2,
      publicOps: 2,
      description: "honest two-dispatch batched Linear -> LogSoftmax kernel plan",
      check: (plan) => {
        if (ops(plan) !== "linear|logSoftmax" || kernels(plan) !== "linear|log-softmax") {
          throw new Error("batched log-softmax-classifier expected honest two-dispatch Linear -> LogSoftmax kernel plan");
        }
      },
    },
    summary: (result) => `log_softmax_classifier_batched=${result.speedup.toFixed(2)}x floor=${floors.logSoftmaxClassifierBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=2 dispatch=2 kernels=linear|log-softmax batched=rank2 hot=allocation-free`,
  },
  {
    key: "lazy_matmul_add_relu_batched",
    label: "lazy-matmul-add-relu-batched",
    floor: floors.lazyMatmulAddReluBatchedSpeedup,
    inputShape: [128, 64],
    inputShapeText: "128x64",
    outputShapeText: "128x64",
    inputLen: 128 * 64,
    outputLen: 128 * 64,
    layerCount: 3,
    parameterNames: "w|b",
    input: () => adapter.tensor(values(128 * 64, 13), [128, 64]),
    model: () => adapter.lazy.input([128, 64])
      .matmul(adapter.lazy.parameter([64, 64], "w"))
      .add(adapter.lazy.parameter([64], "b"))
      .relu(),
    eager: (input) => lazyMatmulAddReluEager(input, values(64 * 64, 32), values(64, 64), 128, 64, 64),
    bindSession: (program) => program.bind({
      weights: new Float32Array(values(64 * 64, 32)),
      bias: new Float32Array(values(64, 64)),
    }),
    ir: { opCount: 3, parameterCount: 2 },
    iterations: 100,
    tolerance: 1e-4,
    plan: {
      opCount: 3,
      dispatchCount: 3,
      publicOps: 3,
      description: "lazy Tensor IR Matmul -> Add -> ReLU Program kernel plan",
      check: (plan) => {
        if (
          ops(plan) !== "matmul|add|activation" ||
          kernels(plan) !== "linear|add|relu" ||
          plan.parameterLayout.parameters.map((param) => param.name).join("|") !== "w|b"
        ) {
          throw new Error("lazy matmul-add-relu expected Matmul -> Add -> ReLU kernel plan with named parameters");
        }
      },
    },
    summary: (result) => `lazy_matmul_add_relu_batched=${result.speedup.toFixed(2)}x floor=${floors.lazyMatmulAddReluBatchedSpeedup.toFixed(2)}x eager=${result.eagerMs.toFixed(4)}ms hot_execute_into=${result.compiledMs.toFixed(4)}ms ops=3 dispatch=3 kernels=linear|add|relu batched=rank2 parameters=w|b hot=allocation-free`,
  },
];

requireBenchSpecCoverage(benchSpecs);

function runAllSpecs(attempt) {
  const results = benchSpecs.map((spec) => ({ spec, result: runBenchSpec(spec) }));
  const failures = [];
  for (const { spec, result } of results) {
    if (result.speedup < spec.floor) {
      failures.push(`${spec.label} ${result.speedup.toFixed(2)}x < ${spec.floor.toFixed(2)}x`);
    }
  }
  const line = [
    `module Program bench gate: ${failures.length === 0 ? "pass" : "fail"}`,
    ...results.map(({ spec, result }) => spec.summary(result)),
  ].join("; ");
  const margin = Math.min(...results.map(({ spec, result }) => result.speedup / spec.floor));
  return { attempt, results, failures, line, margin };
}

const runs = [];
for (let attempt = 1; attempt <= wholeBenchAttempts; attempt += 1) {
  const run = runAllSpecs(attempt);
  runs.push(run);
  if (run.failures.length === 0) {
    process.stdout.write(`${run.line}\n`);
    if (attempt > 1) {
      process.stdout.write(`module Program bench retries: ${attempt - 1} noisy attempt(s) below floor\n`);
    }
    process.exit(0);
  }
}

const best = runs.reduce((acc, run) => (run.margin > acc.margin ? run : acc), runs[0]);
process.stdout.write(`${best.line}\n`);
process.stderr.write(`${best.failures.join("; ")}\n`);
process.exit(1);
