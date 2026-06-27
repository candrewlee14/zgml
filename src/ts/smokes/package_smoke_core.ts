"use strict";

declare const console: { log: (message?: unknown, ...optionalParams: unknown[]) => void };
declare const require: (id: string) => any;

const {
  missingExports,
  nativeApiContractManifest,
  nativeApiContractSignature,
  requiredNativeApiExports,
} = require(["..", "runtime", "native_api_contract.cjs"].join("/"));
const kernelPlanPolicy = require(["..", "kernel_plan.cjs"].join("/"));
const inspection = require(["..", "inspection.cjs"].join("/"));
const {
  expectNativeApiContractManifest,
  expectStepParamsNamespace,
  expectTsProductManifest,
} = require([".", "smoke_contracts.cjs"].join("/"));

function expectClose(actual: ArrayLike<number>, expected: readonly number[], label: string) {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch: ${actual.length} !== ${expected.length}`);
  for (let i = 0; i < expected.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > 1e-5) throw new Error(`${label}[${i}] expected ${expected[i]}, got ${actual[i]}`);
  }
}

function gradNorm(params: readonly { grad: ArrayLike<number> | null }[]) {
  let total = 0;
  for (const param of params) {
    if (!param.grad) continue;
    for (let i = 0; i < param.grad.length; i += 1) total += param.grad[i] * param.grad[i];
  }
  return Math.sqrt(total);
}

function gradMaxAbs(params: readonly { grad: ArrayLike<number> | null }[]) {
  let max = 0;
  for (const param of params) {
    if (!param.grad) continue;
    for (let i = 0; i < param.grad.length; i += 1) max = Math.max(max, Math.abs(param.grad[i]));
  }
  return max;
}

function expectThrowIncludes(fn: () => void, message: string, label: string) {
  try {
    fn();
  } catch (error) {
    if (String((error as Error).message).includes(message)) return;
    throw error;
  }
  throw new Error(`${label} expected error containing ${message}`);
}

function numericSnapshot(values: ArrayLike<number>): number[] {
  return Array.from(values, Number);
}

function expectNumericalGradient(
  adapter: Record<string, any>,
  values: readonly number[],
  shape: readonly number[],
  buildLoss: (input: any) => any,
  label: string,
  epsilon = 1e-3,
  tolerance = 5e-3,
) {
  const analyticInput = adapter.tensor(values.slice(), shape).requiresGrad_();
  const analyticLoss = buildLoss(analyticInput);
  if (!analyticLoss || typeof analyticLoss.backward !== "function") {
    throw new Error(`${label} expected differentiable scalar Tensor loss`);
  }
  if (!analyticLoss.data || analyticLoss.data.length !== 1) {
    throw new Error(`${label} expected scalar Tensor loss, got length ${analyticLoss.data?.length ?? "missing"}`);
  }
  analyticLoss.backward();
  const analyticGrad = numericSnapshot(analyticInput.grad ?? []);
  if (analyticGrad.length !== values.length) {
    throw new Error(`${label} analytic gradient length mismatch: ${analyticGrad.length} !== ${values.length}`);
  }
  for (let i = 0; i < values.length; i += 1) {
    const plus = values.slice();
    const minus = values.slice();
    plus[i] += epsilon;
    minus[i] -= epsilon;
    const plusLoss = buildLoss(adapter.tensor(plus, shape));
    const minusLoss = buildLoss(adapter.tensor(minus, shape));
    if (!plusLoss?.data || !minusLoss?.data || plusLoss.data.length !== 1 || minusLoss.data.length !== 1) {
      throw new Error(`${label} finite-difference probe must return scalar Tensor losses`);
    }
    const numericalGrad = (Number(plusLoss.data[0]) - Number(minusLoss.data[0])) / (2 * epsilon);
    if (Math.abs(analyticGrad[i] - numericalGrad) > tolerance) {
      throw new Error(`${label}[${i}] expected numerical grad ${numericalGrad}, got analytic grad ${analyticGrad[i]}`);
    }
  }
}

type PackageSmokeTensor = Readonly<{
  data: ArrayLike<number>;
  shape: readonly number[];
  mul(value: number): PackageSmokeTensor;
  select(dim: number, index: number): PackageSmokeTensor;
}>;

type PackageSmokeDatasetSample = Readonly<{
  kind: "zgml.data.sample";
  index: number;
  input: PackageSmokeTensor;
  target: PackageSmokeTensor;
}>;

type PackageSmokeCollateContext = Readonly<{
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
  batchIndex: number;
}>;

type PackageSmokeCustomCollateBatch = Readonly<{
  kind: "custom-collate";
  first: number;
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
  batchIndex: number;
}>;

type PackageSmokeDatasetBatch = Readonly<{
  kind: "zgml.data.batch";
  batchIndex: number;
  indices: readonly number[];
  input: PackageSmokeTensor;
  target: PackageSmokeTensor;
}>;

type PackageSmokeMappedSample = PackageSmokeDatasetSample & Readonly<{
  sourceIndex?: number;
}>;

type PackageSmokeTrainBatch = PackageSmokeDatasetBatch;

type PackageSmokeSampleAliasBatch = Readonly<{
  input: PackageSmokeTensor;
  target: PackageSmokeTensor;
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
}>;

type PackageSmokeUnindexedBatch = Readonly<{
  input: PackageSmokeTensor;
  target: PackageSmokeTensor;
}>;

type PackageSmokeTrainContext = Readonly<{
  epoch: number;
  step: number;
  batchIndex: number;
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
}>;

type PackageSmokeNullableTrainContext = Readonly<{
  epoch: number;
  step: number;
  batchIndex: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
}>;

type PackageSmokeTrainStepEvidence = Readonly<{
  kind: string;
  signature: string;
  step: number;
  afterStep?: number;
  loss?: number;
  output?: PackageSmokeTensor;
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
  stepEvidence?: Readonly<{
    kind: string;
    zeroGradApplied?: boolean;
    gradientsCleared?: boolean;
  }>;
}>;

type PackageSmokeNullableTrainStepEvidence =
  Omit<PackageSmokeTrainStepEvidence, "sampleIndices" | "sample_indices"> &
  Readonly<{
    sampleIndices: readonly number[] | null;
    sample_indices: readonly number[] | null;
  }>;

type PackageSmokeNodeFs = Readonly<{
  mkdtempSync(prefix: string): string;
  rmSync(path: string, options: Readonly<{ recursive: boolean; force: boolean }>): void;
  existsSync(path: string): boolean;
  readFileSync(path: string): Uint8Array;
  readFileSync(path: string, encoding: "utf8"): string;
  statSync(path: string): Readonly<{ size: number }>;
}>;

function withTempCheckpointPath<T>(name: string, fn: (path: string, fs: PackageSmokeNodeFs) => T): T {
  const fs = require("node:fs");
  const os = require("node:os");
  const path = require("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "zgml-package-smoke-"));
  try {
    return fn(path.join(dir, name), fs);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function expectPackageSelfReferenceExports(adapter: Record<string, any>, label: string) {
  const packageName = ["zg", "ml"].join("");
  const hostExport = require(label.includes("bun") ? [packageName, "bun"].join("/") : packageName);
  const program = require([packageName, "program"].join("/"));
  const session = require([packageName, "session"].join("/"));
  const stepParams = require([packageName, "step_params"].join("/"));
  const compile = require([packageName, "compile"].join("/"));
  const inspection = require([packageName, "inspection"].join("/"));
  const checkpoint = require([packageName, "checkpoint"].join("/"));
  const data = require([packageName, "data"].join("/"));
  const loss = require([packageName, "loss"].join("/"));
  const nn = require([packageName, "nn"].join("/"));
  const optim = require([packageName, "optim"].join("/"));
  const tensor = require([packageName, "tensor"].join("/"));
  const train = require([packageName, "train"].join("/"));
  if (hostExport.nativeApiContract?.nativeApiContractSignature?.() !== adapter.nativeApiContract?.nativeApiContractSignature?.()) {
    throw new Error(`${label} package self-reference must resolve to the active host runtime`);
  }
  if (
    hostExport.Tensor !== adapter.Tensor ||
    hostExport.tensor !== adapter.tensor ||
    hostExport.nn !== adapter.nn ||
    hostExport.data !== adapter.data ||
    hostExport.loss !== adapter.loss ||
    hostExport.optim !== adapter.optim ||
    hostExport.train !== adapter.train ||
    hostExport.checkpoint !== adapter.checkpoint ||
    hostExport.save !== adapter.save ||
    hostExport.load !== adapter.load
  ) {
    throw new Error(`${label} package self-reference must expose active host PyTorch-like namespaces`);
  }
  if (hostExport.tensor([1], [1]).data[0] !== 1 || hostExport.nn.linear(1, 1).parameterNames().join("|") !== "weight|bias") {
    throw new Error(`${label} package self-reference must execute through active host Tensor and nn helpers`);
  }
  if (tensor.tensorManifest?.source !== "ts" || typeof tensor.createTensorFacadeHelpers !== "function") {
    throw new Error(`${label} zgml/tensor must resolve through the TS-authored tensor package export`);
  }
  expectTsProductManifest(tensor.tensorManifest, "src/ts/tensor.ts", `${label} zgml/tensor`);
  if (nn.nnManifest?.source !== "ts" || typeof nn.createNnNamespace !== "function") {
    throw new Error(`${label} zgml/nn must resolve through the TS-authored nn package export`);
  }
  expectTsProductManifest(nn.nnManifest, "src/ts/nn.ts", `${label} zgml/nn`);
  if (data.dataManifest?.source !== "ts" || typeof data.createDataNamespace !== "function") {
    throw new Error(`${label} zgml/data must resolve through the TS-authored data package export`);
  }
  expectTsProductManifest(data.dataManifest, "src/ts/data.ts", `${label} zgml/data`);
  if (loss.lossManifest?.source !== "ts" || typeof loss.createLossTrainHelpers !== "function") {
    throw new Error(`${label} zgml/loss must resolve through the TS-authored loss package export`);
  }
  expectTsProductManifest(loss.lossManifest, "src/ts/loss.ts", `${label} zgml/loss`);
  if (
    optim.optimManifest?.source !== "ts" ||
    typeof optim.createOptimNamespace !== "function" ||
    typeof optim.isOptimizerStateSnapshot !== "function"
  ) {
    throw new Error(`${label} zgml/optim must resolve through the TS-authored optim package export`);
  }
  expectTsProductManifest(optim.optimManifest, "src/ts/optim.ts", `${label} zgml/optim`);
  if (
    train.trainManifest?.source !== "ts" ||
    typeof train.createLossTrainHelpers !== "function" ||
    typeof train.isTrainFitEvidence !== "function"
  ) {
    throw new Error(`${label} zgml/train must resolve through the TS-authored train package export`);
  }
  expectTsProductManifest(train.trainManifest, "src/ts/train.ts", `${label} zgml/train`);
  if (checkpoint.checkpointManifest?.source !== "ts" || typeof checkpoint.createCheckpointHelpers !== "function") {
    throw new Error(`${label} zgml/checkpoint must resolve through the TS-authored checkpoint package export`);
  }
  expectTsProductManifest(checkpoint.checkpointManifest, "src/ts/checkpoint.ts", `${label} zgml/checkpoint`);
  if (program.programManifest?.source !== "ts" || typeof program.createGenericProgramFacadeHelpers !== "function") {
    throw new Error(`${label} zgml/program must resolve through the TS-authored Program package export`);
  }
  expectTsProductManifest(program.programManifest, "src/ts/program.ts", `${label} zgml/program`);
  if (session.sessionManifest?.source !== "ts" || typeof session.createSessionLiveFacadeHelpers !== "function") {
    throw new Error(`${label} zgml/session must resolve through the TS-authored Session package export`);
  }
  expectTsProductManifest(session.sessionManifest, "src/ts/session.ts", `${label} zgml/session`);
  expectTsProductManifest(stepParams.stepParamsManifest, "src/ts/step_params.ts", `${label} zgml/step_params`);
  expectStepParamsNamespace(stepParams, `${label} zgml/step_params`, "package");
  expectStepParamsNamespace(hostExport.stepParams, `${label} root stepParams`, "package");
  if (
    compile.compileManifest?.source !== "ts" ||
    typeof compile.trace !== "function" ||
    typeof compile.canCompile !== "function" ||
    typeof compile.can_compile !== "function" ||
    typeof compile.requireCompileSupport !== "function" ||
    typeof compile.require_compile_support !== "function" ||
    typeof compile.requireCompilePlan !== "function" ||
    typeof compile.require_compile_plan !== "function" ||
    typeof compile.compile_support !== "function" ||
    typeof compile.compile_plan !== "function" ||
    typeof compile.compiler_signatures !== "function" ||
    typeof compile.kernelPlan !== "function" ||
    typeof compile.memoryLayout !== "function" ||
    typeof compile.parameterLayout !== "function"
  ) {
    throw new Error(`${label} zgml/compile must resolve through the TS-authored compile package export`);
  }
  expectTsProductManifest(compile.compileManifest, "src/ts/compile.ts", `${label} zgml/compile`);
  if (
    inspection.inspectionManifest?.source !== "ts" ||
    typeof inspection.runtimeProfileExpectation !== "function" ||
    typeof inspection.runtimeProfileHasNoFallback !== "function" ||
    typeof inspection.runtimeProfileHasNoSync !== "function" ||
    typeof inspection.runtimeProfileHasNoInvalidRuntimePatches !== "function" ||
    typeof inspection.requireNoFallbackRuntimeProfile !== "function" ||
    typeof inspection.requireNoSyncRuntimeProfile !== "function" ||
    typeof inspection.requireRuntimePatchValidProfile !== "function" ||
    typeof inspection.requireHotRuntimeProfile !== "function"
  ) {
    throw new Error(`${label} zgml/inspection must resolve through the TS-authored inspection package export`);
  }
  expectTsProductManifest(inspection.inspectionManifest, "src/ts/inspection.ts", `${label} zgml/inspection`);
}

function expectNativeEagerLinearEvidence(adapter: Record<string, any>, label: string) {
  const nativeEager = adapter.nativeEager ?? adapter.zgml?.nativeEager ?? adapter.torch?.nativeEager;
  const nativeEagerAlias = adapter.native_eager ?? adapter.zgml?.native_eager ?? adapter.torch?.native_eager;
  if (!nativeEager || typeof nativeEager.linearInto !== "function") {
    throw new Error(`${label} expected nativeEager.linearInto`);
  }
  if (typeof nativeEager.linearActivationInto !== "function") {
    throw new Error(`${label} expected nativeEager.linearActivationInto`);
  }
  if (typeof nativeEager.activationInto !== "function") {
    throw new Error(`${label} expected nativeEager.activationInto`);
  }
  if (typeof nativeEager.matmulInto !== "function") {
    throw new Error(`${label} expected nativeEager.matmulInto`);
  }
  if (typeof nativeEager.elementwiseInto !== "function") {
    throw new Error(`${label} expected nativeEager.elementwiseInto`);
  }
  if (typeof nativeEager.reduceInto !== "function") {
    throw new Error(`${label} expected nativeEager.reduceInto`);
  }
  if (typeof nativeEager.conv2dInto !== "function") {
    throw new Error(`${label} expected nativeEager.conv2dInto`);
  }
  if (typeof nativeEager.pool2dInto !== "function") {
    throw new Error(`${label} expected nativeEager.pool2dInto`);
  }
  if (typeof nativeEager.softmaxInto !== "function" || typeof nativeEager.logSoftmaxInto !== "function") {
    throw new Error(`${label} expected nativeEager softmax/logSoftmax into helpers`);
  }
  if (!nativeEagerAlias || typeof nativeEagerAlias.linear_into !== "function") {
    throw new Error(`${label} expected native_eager.linear_into alias`);
  }
  if (typeof nativeEagerAlias.linear_activation_into !== "function") {
    throw new Error(`${label} expected native_eager.linear_activation_into alias`);
  }
  if (typeof nativeEagerAlias.activation_into !== "function") {
    throw new Error(`${label} expected native_eager.activation_into alias`);
  }
  if (typeof nativeEagerAlias.matmul_into !== "function") {
    throw new Error(`${label} expected native_eager.matmul_into alias`);
  }
  if (typeof nativeEagerAlias.elementwise_into !== "function") {
    throw new Error(`${label} expected native_eager.elementwise_into alias`);
  }
  if (typeof nativeEagerAlias.reduce_into !== "function") {
    throw new Error(`${label} expected native_eager.reduce_into alias`);
  }
  if (typeof nativeEagerAlias.conv2d_into !== "function") {
    throw new Error(`${label} expected native_eager.conv2d_into alias`);
  }
  if (typeof nativeEagerAlias.pool2d_into !== "function") {
    throw new Error(`${label} expected native_eager.pool2d_into alias`);
  }
  if (typeof nativeEagerAlias.softmax_into !== "function" || typeof nativeEagerAlias.log_softmax_into !== "function") {
    throw new Error(`${label} expected native_eager softmax/log_softmax aliases`);
  }
  const input = adapter.tensor([1, 2, 3, 4], [2, 2]);
  const weights = adapter.tensor([1, 0, 0.5, 0, 1, -0.5], [2, 3]);
  const bias = adapter.tensor([0.25, -0.25, 0.5], [3]);
  const directOutput = new Float32Array(6);
  const directResult = nativeEager.linearInto(directOutput, input, weights, { bias });
  if (directResult !== directOutput) {
    throw new Error(`${label} expected nativeEager.linearInto to reuse caller output`);
  }
  expectClose(directOutput, [1.25, 1.75, 0, 3.25, 3.75, 0], `${label} nativeEager.linearInto output`);
  const aliasOutput = new Float32Array(6);
  const aliasResult = nativeEagerAlias.linear_into(aliasOutput, input, weights, { bias });
  if (aliasResult !== aliasOutput) {
    throw new Error(`${label} expected native_eager.linear_into to reuse caller output`);
  }
  expectClose(aliasOutput, Array.from(directOutput), `${label} native_eager.linear_into output`);
  const matmulOutput = new Float32Array(6);
  const matmulResult = nativeEager.matmulInto(matmulOutput, input, weights);
  if (matmulResult !== matmulOutput) {
    throw new Error(`${label} expected nativeEager.matmulInto to reuse caller output`);
  }
  expectClose(matmulOutput, Array.from(directOutput, (value, index) => value - bias.data[index % 3]), `${label} nativeEager.matmulInto output`);
  const matmulAliasOutput = new Float32Array(6);
  const matmulAliasResult = nativeEagerAlias.matmul_into(matmulAliasOutput, input, weights);
  if (matmulAliasResult !== matmulAliasOutput) {
    throw new Error(`${label} expected native_eager.matmul_into to reuse caller output`);
  }
  expectClose(matmulAliasOutput, Array.from(matmulOutput), `${label} native_eager.matmul_into output`);
  const elementwiseOutput = new Float32Array(6);
  const elementwiseResult = nativeEager.elementwiseInto(elementwiseOutput, directOutput, new Float32Array([2]), { op: "mul" });
  if (elementwiseResult !== elementwiseOutput) {
    throw new Error(`${label} expected nativeEager.elementwiseInto to reuse caller output`);
  }
  expectClose(elementwiseOutput, Array.from(directOutput, (value) => value * 2), `${label} nativeEager.elementwiseInto output`);
  const elementwiseAliasOutput = new Float32Array(6);
  const elementwiseAliasResult = nativeEagerAlias.elementwise_into(elementwiseAliasOutput, directOutput, null, { op: "sqr" });
  if (elementwiseAliasResult !== elementwiseAliasOutput) {
    throw new Error(`${label} expected native_eager.elementwise_into to reuse caller output`);
  }
  expectClose(elementwiseAliasOutput, Array.from(directOutput, (value) => value * value), `${label} native_eager.elementwise_into output`);
  const reduceOutput = new Float32Array(1);
  const reduceResult = nativeEager.reduceInto(reduceOutput, directOutput, { op: "sum" });
  if (reduceResult !== reduceOutput) {
    throw new Error(`${label} expected nativeEager.reduceInto to reuse caller output`);
  }
  expectClose(reduceOutput, [10], `${label} nativeEager.reduceInto output`);
  const reduceAliasOutput = new Float32Array(1);
  const reduceAliasResult = nativeEagerAlias.reduce_into(reduceAliasOutput, directOutput, { op: "max" });
  if (reduceAliasResult !== reduceAliasOutput) {
    throw new Error(`${label} expected native_eager.reduce_into to reuse caller output`);
  }
  expectClose(reduceAliasOutput, [3.75], `${label} native_eager.reduce_into output`);
  const conv2dInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]);
  const conv2dWeights = adapter.tensor([1, 0, 0, 1], [1, 1, 2, 2]);
  const conv2dBias = adapter.tensor([0.5], [1]);
  const conv2dOutput = new Float32Array(4);
  const conv2dResult = nativeEager.conv2dInto(conv2dOutput, conv2dInput, conv2dWeights, { bias: conv2dBias, outH: 2, outW: 2 });
  if (conv2dResult !== conv2dOutput) {
    throw new Error(`${label} expected nativeEager.conv2dInto to reuse caller output`);
  }
  expectClose(conv2dOutput, [6.5, 8.5, 12.5, 14.5], `${label} nativeEager.conv2dInto output`);
  const conv2dAliasOutput = new Float32Array(4);
  const conv2dAliasResult = nativeEagerAlias.conv2d_into(conv2dAliasOutput, conv2dInput, conv2dWeights, { bias: conv2dBias, out_h: 2, out_w: 2 });
  if (conv2dAliasResult !== conv2dAliasOutput) {
    throw new Error(`${label} expected native_eager.conv2d_into to reuse caller output`);
  }
  expectClose(conv2dAliasOutput, Array.from(conv2dOutput), `${label} native_eager.conv2d_into output`);
  const pool2dInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]);
  const maxPool2dOutput = new Float32Array(4);
  const maxPool2dResult = nativeEager.pool2dInto(maxPool2dOutput, pool2dInput, { op: "max", kernelH: 2, kernelW: 2, strideH: 1, strideW: 1, outH: 2, outW: 2 });
  if (maxPool2dResult !== maxPool2dOutput) {
    throw new Error(`${label} expected nativeEager.pool2dInto to reuse caller output`);
  }
  expectClose(maxPool2dOutput, [5, 6, 8, 9], `${label} nativeEager.pool2dInto max output`);
  const avgPool2dOutput = new Float32Array(4);
  const avgPool2dAliasResult = nativeEagerAlias.pool2d_into(avgPool2dOutput, pool2dInput, { op: "avg", kernel_h: 2, kernel_w: 2, stride_h: 1, stride_w: 1, out_h: 2, out_w: 2 });
  if (avgPool2dAliasResult !== avgPool2dOutput) {
    throw new Error(`${label} expected native_eager.pool2d_into to reuse caller output`);
  }
  expectClose(avgPool2dOutput, [3, 4, 6, 7], `${label} native_eager.pool2d_into avg output`);
  const tensorElementwiseLength = 1024;
  const tensorElementwiseInputData = Array.from({ length: tensorElementwiseLength }, (_value, index) => (index % 5) - 2);
  const tensorElementwiseBiasData = Array.from({ length: tensorElementwiseLength }, (_value, index) => (index % 7) + 0.5);
  const tensorElementwiseInput = adapter.tensor(tensorElementwiseInputData, [tensorElementwiseLength]);
  const tensorElementwiseBias = adapter.tensor(tensorElementwiseBiasData, [tensorElementwiseLength]);
  const tensorElementwise = adapter.noGrad(() => tensorElementwiseInput.mul(2).add(tensorElementwiseBias).sqr().sqrt());
  expectClose(tensorElementwise.data, tensorElementwiseInputData.map((value, index) => Math.abs(value * 2 + tensorElementwiseBiasData[index])), `${label} noGrad Tensor elementwise native eager output`);
  const tensorReduce = adapter.noGrad(() => tensorElementwiseInput.sum().add(tensorElementwiseInput.mean()).add(tensorElementwiseInput.max()).add(tensorElementwiseInput.min()));
  const tensorReduceExpected = tensorElementwiseInputData.reduce((acc, value) => acc + value, 0)
    + tensorElementwiseInputData.reduce((acc, value) => acc + value, 0) / tensorElementwiseLength
    + Math.max(...tensorElementwiseInputData)
    + Math.min(...tensorElementwiseInputData);
  expectClose(tensorReduce.data, [tensorReduceExpected], `${label} noGrad Tensor reduce native eager output`);
  const geluOutput = new Float32Array(6);
  const geluResult = nativeEager.linearActivationInto(geluOutput, input, weights, { bias, activation: "gelu" });
  if (geluResult !== geluOutput) {
    throw new Error(`${label} expected nativeEager.linearActivationInto to reuse caller output`);
  }
  expectClose(geluOutput, Array.from(directOutput, (value) => 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value)))), `${label} nativeEager.linearActivationInto output`);
  const geluAliasOutput = new Float32Array(6);
  const geluAliasResult = nativeEagerAlias.linear_activation_into(geluAliasOutput, input, weights, { bias, activation: "gelu" });
  if (geluAliasResult !== geluAliasOutput) {
    throw new Error(`${label} expected native_eager.linear_activation_into to reuse caller output`);
  }
  expectClose(geluAliasOutput, Array.from(geluOutput), `${label} native_eager.linear_activation_into output`);
  const reluOutput = new Float32Array(6);
  const reluResult = nativeEager.linearActivationInto(reluOutput, input, weights, { bias, activation: "relu" });
  if (reluResult !== reluOutput) {
    throw new Error(`${label} expected nativeEager.linearActivationInto ReLU to reuse caller output`);
  }
  expectClose(reluOutput, Array.from(directOutput, (value) => value > 0 ? value : 0), `${label} nativeEager.linearActivationInto ReLU output`);
  const siluOutput = new Float32Array(6);
  const siluResult = nativeEager.linearActivationInto(siluOutput, input, weights, { bias, activation: "silu" });
  if (siluResult !== siluOutput) {
    throw new Error(`${label} expected nativeEager.linearActivationInto SiLU to reuse caller output`);
  }
  expectClose(siluOutput, Array.from(directOutput, (value) => value / (1 + Math.exp(-value))), `${label} nativeEager.linearActivationInto SiLU output`);
  const sigmoidOutput = new Float32Array(6);
  const sigmoidResult = nativeEager.linearActivationInto(sigmoidOutput, input, weights, { bias, activation: "sigmoid" });
  if (sigmoidResult !== sigmoidOutput) {
    throw new Error(`${label} expected nativeEager.linearActivationInto Sigmoid to reuse caller output`);
  }
  expectClose(sigmoidOutput, Array.from(directOutput, (value) => 1 / (1 + Math.exp(-value))), `${label} nativeEager.linearActivationInto Sigmoid output`);
  const tanhOutput = new Float32Array(6);
  const tanhResult = nativeEager.linearActivationInto(tanhOutput, input, weights, { bias, activation: "tanh" });
  if (tanhResult !== tanhOutput) {
    throw new Error(`${label} expected nativeEager.linearActivationInto Tanh to reuse caller output`);
  }
  expectClose(tanhOutput, Array.from(directOutput, (value) => Math.tanh(value)), `${label} nativeEager.linearActivationInto Tanh output`);
  const activationInput = adapter.tensor([-2, -0.5, 0, 0.5, 2], [5]);
  const activationOutput = new Float32Array(5);
  const activationResult = nativeEager.activationInto(activationOutput, activationInput, { activation: "relu" });
  if (activationResult !== activationOutput) {
    throw new Error(`${label} expected nativeEager.activationInto to reuse caller output`);
  }
  expectClose(activationOutput, [0, 0, 0, 0.5, 2], `${label} nativeEager.activationInto output`);
  const activationAliasOutput = new Float32Array(5);
  const activationAliasResult = nativeEagerAlias.activation_into(activationAliasOutput, activationInput, { activation: "tanh" });
  if (activationAliasResult !== activationAliasOutput) {
    throw new Error(`${label} expected native_eager.activation_into to reuse caller output`);
  }
  expectClose(activationAliasOutput, Array.from(activationInput.data as ArrayLike<number>, (value) => Math.tanh(value)), `${label} native_eager.activation_into output`);
  const softmaxOutput = new Float32Array(6);
  const softmaxResult = nativeEager.softmaxInto(softmaxOutput, directOutput, { rows: 2, cols: 3 });
  if (softmaxResult !== softmaxOutput) {
    throw new Error(`${label} expected nativeEager.softmaxInto to reuse caller output`);
  }
  expectClose(softmaxOutput, adapter.tensor(Array.from(directOutput), [2, 3]).softmaxDim(-1).data, `${label} nativeEager.softmaxInto output`);
  const logSoftmaxOutput = new Float32Array(6);
  const logSoftmaxResult = nativeEager.logSoftmaxInto(logSoftmaxOutput, adapter.tensor(Array.from(directOutput), [2, 3]), { dim: -1 });
  if (logSoftmaxResult !== logSoftmaxOutput) {
    throw new Error(`${label} expected nativeEager.logSoftmaxInto to reuse caller output`);
  }
  expectClose(logSoftmaxOutput, adapter.tensor(Array.from(directOutput), [2, 3]).logSoftmaxDim(-1).data, `${label} nativeEager.logSoftmaxInto output`);
  const logSoftmaxAliasOutput = new Float32Array(6);
  const logSoftmaxAliasResult = nativeEagerAlias.log_softmax_into(logSoftmaxAliasOutput, adapter.tensor(Array.from(directOutput), [2, 3]), { dim: -1 });
  if (logSoftmaxAliasResult !== logSoftmaxAliasOutput) {
    throw new Error(`${label} expected native_eager.log_softmax_into to reuse caller output`);
  }
  expectClose(logSoftmaxAliasOutput, Array.from(logSoftmaxOutput), `${label} native_eager.log_softmax_into output`);

  const linear = adapter.nn.linear(2, 3, {
    weight: [1, 0, 0.5, 0, 1, -0.5],
    bias: [0.25, -0.25, 0.5],
  });
  const eager = linear.forward(input);
  const nativeModule = adapter.noGrad(() => linear.forward(input));
  expectClose(nativeModule.data, eager.data, `${label} noGrad nn.Linear native eager module output`);
  const originalMatmul = adapter.Tensor.prototype.matmul;
  try {
    adapter.Tensor.prototype.matmul = () => {
      throw new Error(`${label} poisoned Tensor.matmul fallback`);
    };
    const nativeNestedModule = adapter.noGrad(() => linear.forward([[1, 2], [3, 4]]));
    expectClose(nativeNestedModule.data, Array.from(directOutput), `${label} noGrad nn.Linear nested array native eager module output`);
    if (nativeNestedModule.shape.join("x") !== "2x3") {
      throw new Error(`${label} expected noGrad nn.Linear nested array shape [2,3], got [${nativeNestedModule.shape.join(",")}]`);
    }
  } finally {
    adapter.Tensor.prototype.matmul = originalMatmul;
  }
  const conv2d = adapter.nn.conv2d(1, 1, [2, 2], {
    weight: [1, 0, 0, 1],
    bias: [0.5],
  });
  const eagerConv2d = conv2d.forward(conv2dInput);
  const nativeConv2d = adapter.noGrad(() => conv2d.forward(conv2dInput));
  expectClose(nativeConv2d.data, eagerConv2d.data, `${label} noGrad nn.Conv2d native eager module output`);
  expectClose(nativeConv2d.data, Array.from(conv2dOutput), `${label} noGrad nn.Conv2d native eager module direct output parity`);
  const maxPool2d = adapter.nn.maxPool2d([2, 2], { stride: [1, 1] });
  const eagerMaxPool2d = maxPool2d.forward(pool2dInput);
  const nativeMaxPool2d = adapter.noGrad(() => maxPool2d.forward(pool2dInput));
  expectClose(nativeMaxPool2d.data, eagerMaxPool2d.data, `${label} noGrad nn.MaxPool2d native eager module output`);
  expectClose(nativeMaxPool2d.data, Array.from(maxPool2dOutput), `${label} noGrad nn.MaxPool2d native eager module direct output parity`);
  const avgPool2d = adapter.nn.avgPool2d([2, 2], { stride: [1, 1] });
  const eagerAvgPool2d = avgPool2d.forward(pool2dInput);
  const nativeAvgPool2d = adapter.noGrad(() => avgPool2d.forward(pool2dInput));
  expectClose(nativeAvgPool2d.data, eagerAvgPool2d.data, `${label} noGrad nn.AvgPool2d native eager module output`);
  expectClose(nativeAvgPool2d.data, Array.from(avgPool2dOutput), `${label} noGrad nn.AvgPool2d native eager module direct output parity`);

  const fusedSequential = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weight: [1, 0, 0.5, 0, 1, -0.5],
      bias: [0.25, -0.25, 0.5],
    }),
    new adapter.nn.GELU(),
  );
  const eagerFusedSequential = fusedSequential.forward(input);
  const nativeFusedSequential = adapter.noGrad(() => fusedSequential.forward(input));
  expectClose(nativeFusedSequential.data, eagerFusedSequential.data, `${label} noGrad nn.Sequential Linear+GELU native eager module output`);

  const reluSequential = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weight: [1, 0, 0.5, 0, 1, -0.5],
      bias: [0.25, -0.25, 0.5],
    }),
    new adapter.nn.ReLU(),
  );
  const eagerReluSequential = reluSequential.forward(input);
  const nativeReluSequential = adapter.noGrad(() => reluSequential.forward(input));
  expectClose(nativeReluSequential.data, eagerReluSequential.data, `${label} noGrad nn.Sequential Linear+ReLU native eager module output`);

  const siluSequential = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weight: [1, 0, 0.5, 0, 1, -0.5],
      bias: [0.25, -0.25, 0.5],
    }),
    new adapter.nn.SiLU(),
  );
  const eagerSiluSequential = siluSequential.forward(input);
  const nativeSiluSequential = adapter.noGrad(() => siluSequential.forward(input));
  expectClose(nativeSiluSequential.data, eagerSiluSequential.data, `${label} noGrad nn.Sequential Linear+SiLU native eager module output`);

  const sigmoidSequential = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weight: [1, 0, 0.5, 0, 1, -0.5],
      bias: [0.25, -0.25, 0.5],
    }),
    new adapter.nn.Sigmoid(),
  );
  const eagerSigmoidSequential = sigmoidSequential.forward(input);
  const nativeSigmoidSequential = adapter.noGrad(() => sigmoidSequential.forward(input));
  expectClose(nativeSigmoidSequential.data, eagerSigmoidSequential.data, `${label} noGrad nn.Sequential Linear+Sigmoid native eager module output`);

  const tanhSequential = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weight: [1, 0, 0.5, 0, 1, -0.5],
      bias: [0.25, -0.25, 0.5],
    }),
    new adapter.nn.Tanh(),
  );
  const eagerTanhSequential = tanhSequential.forward(input);
  const nativeTanhSequential = adapter.noGrad(() => tanhSequential.forward(input));
  expectClose(nativeTanhSequential.data, eagerTanhSequential.data, `${label} noGrad nn.Sequential Linear+Tanh native eager module output`);

  const logSoftmaxModule = adapter.nn.logSoftmax(-1);
  const eagerLogSoftmax = logSoftmaxModule.forward(input);
  const nativeLogSoftmax = adapter.noGrad(() => logSoftmaxModule.forward(input));
  expectClose(nativeLogSoftmax.data, eagerLogSoftmax.data, `${label} noGrad nn.LogSoftmax native eager module output`);

  const softmaxModule = adapter.nn.softmax(-1);
  const eagerSoftmax = softmaxModule.forward(input);
  const nativeSoftmax = adapter.noGrad(() => softmaxModule.forward(input));
  expectClose(nativeSoftmax.data, eagerSoftmax.data, `${label} noGrad nn.Softmax native eager module output`);
}

function expectLossAndAdamWEvidence(adapter: Record<string, any>, label: string) {
  if (
    adapter.manualSeed(123) !== 123 ||
    adapter.initialSeed() !== 123 ||
    adapter.initial_seed() !== 123 ||
    adapter.manual_seed(123) !== 123 ||
    adapter.Tensor.initialSeed() !== 123 ||
    adapter.Tensor.initial_seed() !== 123
  ) {
    throw new Error(`${label} expected root and Tensor manualSeed helpers`);
  }
  expectClose(adapter.rand([3], { seed: 99 }).data, Array.from(adapter.rand([3], { seed: 99 }).data), `${label} rand seed option`);
  adapter.manualSeed(321);
  const seededA = adapter.rand([3]);
  adapter.Tensor.manual_seed(321);
  const seededB = adapter.Tensor.rand([3]);
  expectClose(seededA.data, Array.from(seededB.data), `${label} manualSeed default RNG reset`);
  const rngA = adapter.seededRng(77);
  const rngB = adapter.Tensor.seededRng(77);
  expectClose([rngA(), rngA()], [rngB(), rngB()], `${label} seededRng deterministic sequence`);

  const rawCrossEntropy = adapter.loss.crossEntropy([1, 2, 3], [2], { classes: 3 });
  if (typeof rawCrossEntropy !== "number" || Math.abs(rawCrossEntropy - 0.4076059644443804) > 1e-6) {
    throw new Error(`${label} expected raw crossEntropy to return numeric host loss`);
  }
  if (Math.abs(adapter.loss.cross_entropy([1, 2, 3], [2], { numClasses: 3 }) - rawCrossEntropy) > 1e-6) {
    throw new Error(`${label} expected raw cross_entropy alias to match crossEntropy`);
  }
  if (
    adapter.nn.F !== adapter.nn.functional ||
    adapter.F !== adapter.nn.functional ||
    !Object.isFrozen(adapter.nn.functional) ||
    adapter.nn.functional.crossEntropy !== adapter.loss.crossEntropy ||
    adapter.nn.functional.cross_entropy !== adapter.loss.cross_entropy ||
    adapter.nn.functional.nll_loss !== adapter.loss.nll_loss ||
    adapter.nn.functional.mse !== adapter.loss.mse ||
    adapter.nn.functional.mse_loss !== adapter.loss.mse ||
    adapter.nn.functional.l1 !== adapter.loss.l1 ||
    adapter.nn.functional.l1_loss !== adapter.loss.l1 ||
    adapter.nn.functional.huber !== adapter.loss.huber ||
    adapter.nn.functional.huber_loss !== adapter.loss.huber ||
    adapter.nn.functional.smooth_l1 !== adapter.loss.smooth_l1 ||
    adapter.nn.functional.smooth_l1_loss !== adapter.loss.smooth_l1 ||
    adapter.nn.functional.binary_cross_entropy !== adapter.loss.binary_cross_entropy ||
    adapter.nn.functional.binary_cross_entropy_with_logits !== adapter.loss.binary_cross_entropy_with_logits ||
    Math.abs(adapter.nn.functional.cross_entropy([1, 2, 3], [2], { numClasses: 3 }) - rawCrossEntropy) > 1e-6 ||
    Math.abs(adapter.nn.functional.nll_loss([-3, -2, -0.5], [2], { classes: 3 }) - 0.5) > 1e-6 ||
    Math.abs(adapter.nn.functional.mse([1, 3], [1, 1]) - 2) > 1e-6 ||
    Math.abs(adapter.nn.functional.mse_loss([1, 3], [1, 1]) - 2) > 1e-6 ||
    Math.abs(adapter.nn.functional.mse_loss([1, 3], [1, 1], { reduction: "sum" }) - 4) > 1e-6 ||
    Math.abs(adapter.nn.functional.l1([1, 3], [1, 1]) - 1) > 1e-6 ||
    Math.abs(adapter.nn.functional.l1_loss([1, 3], [1, 1]) - 1) > 1e-6 ||
    Math.abs(adapter.nn.functional.l1_loss([1, 3], [1, 1], { reduction: "sum" }) - 2) > 1e-6 ||
    Math.abs(adapter.nn.functional.huber([1, 3], [1, 1], { delta: 1 }) - 0.75) > 1e-6 ||
    Math.abs(adapter.nn.functional.huber_loss([1, 3], [1, 1], { delta: 1 }) - 0.75) > 1e-6 ||
    Math.abs(adapter.nn.functional.smooth_l1([1, 3], [1, 1], { beta: 1 }) - 0.75) > 1e-6 ||
    Math.abs(adapter.nn.functional.smooth_l1_loss([1, 3], [1, 1], { beta: 1 }) - 0.75) > 1e-6 ||
    Math.abs(adapter.nn.functional.binary_cross_entropy([0.25, 0.75], [0, 1]) - 0.287682) > 1e-6 ||
    Math.abs(adapter.nn.functional.binary_cross_entropy_with_logits([0, 2], [0, 1]) - 0.410038) > 1e-6 ||
    Math.abs(adapter.nn.functional.binary_cross_entropy_with_logits([0, 2], [0, 1], { reduction: "sum" }) - 0.820075) > 1e-6 ||
    Math.abs(adapter.nn.F.crossEntropy([1, 2, 3], [2], { classes: 3 }) - rawCrossEntropy) > 1e-6 ||
    adapter.F.softmax([1, 2, 3], -1).shape.join("x") !== "3" ||
    adapter.nn.functional.relu(adapter.tensor([-1, 2], [2])).data.join(",") !== "0,2" ||
    adapter.nn.functional.relu(adapter.tensor([-1, 2], [2]), false).data.join(",") !== "0,2" ||
    adapter.nn.functional.gelu(adapter.tensor([0, 1], [2])).shape.join("x") !== "2" ||
    adapter.nn.functional.silu(adapter.tensor([0, 1], [2])).shape.join("x") !== "2" ||
    adapter.nn.functional.sigmoid(adapter.tensor([0, 1], [2])).shape.join("x") !== "2" ||
    adapter.nn.functional.tanh(adapter.tensor([0, 1], [2])).shape.join("x") !== "2" ||
    adapter.nn.functional.softmax([1, 2, 3], -1).shape.join("x") !== "3" ||
    adapter.nn.functional.softmax_dim([1, 2, 3], -1).shape.join("x") !== "3" ||
    adapter.nn.functional.softmaxDim([1, 2, 3], -1).shape.join("x") !== "3" ||
    adapter.nn.functional.log_softmax(adapter.tensor([[1, 2, 3]], [1, 3]), 1).shape.join("x") !== "1x3" ||
    adapter.nn.functional.log_softmax_dim(adapter.tensor([[1, 2, 3]], [1, 3]), 1).shape.join("x") !== "1x3" ||
    adapter.nn.functional.logSoftmaxDim(adapter.tensor([[1, 2, 3]], [1, 3]), 1).shape.join("x") !== "1x3" ||
    adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, { training: false }).data.join(",") !== "2,4" ||
    adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, false).data.join(",") !== "2,4" ||
    adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, false, false).data.join(",") !== "2,4" ||
    adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, true).shape.join("x") !== "2" ||
    adapter.nn.functional.flatten(adapter.tensor([[1, 2, 3]], [1, 3])).shape.join("x") !== "3" ||
    adapter.nn.functional.flatten(adapter.tensor([[[1, 2], [3, 4]]], [1, 2, 2]), 1, -1).shape.join("x") !== "1x4" ||
    adapter.nn.functional.linear(adapter.tensor([2, 4], [2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).shape.join("x") !== "2" ||
    adapter.nn.functional.linear(adapter.tensor([[2, 4]], [1, 2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).shape.join("x") !== "1x2" ||
    adapter.nn.functional.linear(adapter.tensor([2, 4, 6, 8], [1, 2, 2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).shape.join("x") !== "1x2x2" ||
    adapter.nn.functional.normalize(adapter.tensor([[3, 4]], [1, 2]), 2, 1).shape.join("x") !== "1x2" ||
    adapter.nn.functional.one_hot(adapter.tensor([0, 2], [2]), 3).shape.join("x") !== "2x3" ||
    adapter.nn.functional.one_hot(adapter.tensor([0, 2], [1, 2]), 3).shape.join("x") !== "1x2x3" ||
    adapter.nn.functional.oneHot([0, 2], 3).shape.join("x") !== "2x3" ||
    adapter.nn.functional.embedding(adapter.tensor([0, 2], [2]), adapter.tensor([[1, 0], [0, 1], [1, 1]], [3, 2])).shape.join("x") !== "2x2" ||
    adapter.nn.functional.embedding(adapter.tensor([0, 2], [1, 2]), adapter.tensor([[1, 0], [0, 1], [1, 1]], [3, 2])).shape.join("x") !== "1x2x2" ||
    adapter.nn.functional.layer_norm(adapter.tensor([[1, 3]], [1, 2]), [2], { eps: 0 }).shape.join("x") !== "1x2" ||
    adapter.nn.functional.layer_norm(adapter.tensor([[1, 3]], [1, 2]), [2], adapter.tensor([2, 3], [2]), adapter.tensor([10, 20], [2]), 0).shape.join("x") !== "1x2" ||
    adapter.nn.functional.rmsNorm(adapter.tensor([[3, 4]], [1, 2]), 2, { eps: 0 }).shape.join("x") !== "1x2" ||
    adapter.nn.functional.batch_norm1d(adapter.tensor([1, 3, 3, 7], [2, 2]), 2, { eps: 0 }).shape.join("x") !== "2x2" ||
    adapter.nn.functional.rms_norm(adapter.tensor([[3, 4]], [1, 2]), 2, adapter.tensor([2, 3], [2]), 0).shape.join("x") !== "1x2"
  ) {
    throw new Error(`${label} expected nn.functional/nn.F to expose frozen loss and tensor functional helpers`);
  }
  expectClose(adapter.nn.functional.gelu(adapter.tensor([0, 1], [2])).data, [0, 0.841192], `${label} nn.functional.gelu values`);
  expectClose(adapter.nn.functional.silu(adapter.tensor([0, 1], [2])).data, [0, 0.731059], `${label} nn.functional.silu values`);
  expectClose(adapter.nn.functional.sigmoid(adapter.tensor([0, 1], [2])).data, [0.5, 0.731059], `${label} nn.functional.sigmoid values`);
  expectClose(adapter.nn.functional.tanh(adapter.tensor([0, 1], [2])).data, [0, 0.761594], `${label} nn.functional.tanh values`);
  expectClose(adapter.nn.functional.softmax_dim(adapter.tensor([1, 2], [2]), -1).data, [0.268941, 0.731059], `${label} nn.functional.softmax_dim values`);
  expectClose(adapter.nn.functional.softmaxDim(adapter.tensor([1, 2], [2]), -1).data, [0.268941, 0.731059], `${label} nn.functional.softmaxDim values`);
  expectClose(adapter.nn.functional.log_softmax_dim(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} nn.functional.log_softmax_dim values`);
  expectClose(adapter.nn.functional.logSoftmaxDim(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} nn.functional.logSoftmaxDim values`);
  expectClose(adapter.nn.functional.flatten(adapter.tensor([[1, 2, 3]], [1, 3])).data, [1, 2, 3], `${label} nn.functional.flatten values`);
  expectClose(adapter.nn.functional.flatten(adapter.tensor([[[1, 2], [3, 4]]], [1, 2, 2]), 1, -1).data, [1, 2, 3, 4], `${label} nn.functional.flatten range values`);
  expectClose(adapter.nn.functional.linear(adapter.tensor([2, 4], [2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).data, [3, 3], `${label} nn.functional.linear rank-1 values`);
  expectClose(adapter.nn.functional.linear(adapter.tensor([[2, 4]], [1, 2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).data, [3, 3], `${label} nn.functional.linear rank-2 values`);
  expectClose(adapter.nn.functional.linear(adapter.tensor([2, 4, 6, 8], [1, 2, 2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).data, [3, 3, 7, 7], `${label} nn.functional.linear rank-3 values`);
  expectClose(adapter.nn.functional.linear(adapter.tensor([2, 4, 6, 8], [1, 1, 2, 2]), adapter.tensor([[1, 0], [0, 1]], [2, 2]), adapter.tensor([1, -1], [2])).data, [3, 3, 7, 7], `${label} nn.functional.linear rank-4 values`);
  expectClose(adapter.nn.functional.linear(adapter.tensor([[2, 4]], [1, 2]), adapter.tensor([[0.5, -0.5]], [1, 2])).data, [-1], `${label} nn.functional.linear PyTorch weight layout`);
  expectClose(adapter.nn.functional.normalize(adapter.tensor([[3, 4]], [1, 2]), 2, 1).data, [0.6, 0.8], `${label} nn.functional.normalize values`);
  expectClose(adapter.nn.functional.one_hot(adapter.tensor([0, 2], [2]), 3).data, [1, 0, 0, 0, 0, 1], `${label} nn.functional.one_hot values`);
  expectClose(adapter.nn.functional.one_hot(adapter.tensor([0, 2], [1, 2]), 3).data, [1, 0, 0, 0, 0, 1], `${label} nn.functional.one_hot grid values`);
  expectClose(adapter.nn.functional.one_hot([0, 2]).data, [1, 0, 0, 0, 0, 1], `${label} nn.functional.one_hot inferred classes`);
  expectClose(adapter.nn.functional.embedding(adapter.tensor([0, 2], [2]), adapter.tensor([[1, 0], [0, 1], [1, 1]], [3, 2])).data, [1, 0, 1, 1], `${label} nn.functional.embedding values`);
  expectClose(adapter.nn.functional.embedding(adapter.tensor([0, 2], [1, 2]), adapter.tensor([[1, 0], [0, 1], [1, 1]], [3, 2])).data, [1, 0, 1, 1], `${label} nn.functional.embedding grid values`);
  expectClose(adapter.nn.functional.layer_norm(adapter.tensor([[1, 3]], [1, 2]), [2], { eps: 0 }).data, [-1, 1], `${label} nn.functional.layer_norm values`);
  expectClose(adapter.nn.functional.layer_norm(adapter.tensor([[1, 3]], [1, 2]), [2], adapter.tensor([2, 3], [2]), adapter.tensor([10, 20], [2]), 0).data, [8, 23], `${label} nn.functional.layer_norm positional values`);
  expectClose(adapter.nn.functional.rmsNorm(adapter.tensor([[3, 4]], [1, 2]), 2, { eps: 0 }).data, [0.848528, 1.131371], `${label} nn.functional.rmsNorm values`);
  expectClose(adapter.nn.functional.batch_norm1d(adapter.tensor([1, 3, 3, 7], [2, 2]), 2, { eps: 0 }).data, [-1, -1, 1, 1], `${label} nn.functional.batch_norm1d values`);
  expectClose(adapter.nn.functional.rms_norm(adapter.tensor([[3, 4]], [1, 2]), 2, adapter.tensor([2, 3], [2]), 0).data, [1.697056, 3.394113], `${label} nn.functional.rms_norm positional values`);
  let dropoutRngCalls = 0;
  const functionalDropout = adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, {
    training: true,
    rng: () => dropoutRngCalls++ === 0 ? 0.25 : 0.75,
  });
  if (functionalDropout.data.join(",") !== "0,8") {
    throw new Error(`${label} expected nn.functional.dropout to reuse training-mode dropout semantics`);
  }
  expectThrowIncludes(
    () => adapter.nn.functional.one_hot([3], 3),
    "must be an integer in [0, 2]",
    `${label} nn.functional.one_hot rejects out-of-range targets`,
  );
  expectThrowIncludes(
    () => adapter.nn.functional.relu(adapter.tensor([-1, 2], [2]), true),
    "does not support inplace=true",
    `${label} nn.functional.relu rejects inplace=true`,
  );
  expectThrowIncludes(
    () => adapter.nn.functional.dropout(adapter.tensor([2, 4], [2]), 0.5, false, true),
    "does not support inplace=true",
    `${label} nn.functional.dropout rejects inplace=true`,
  );
  const rawNllLoss = adapter.loss.nllLoss([-3, -2, -0.5], [2], { classes: 3 });
  if (typeof rawNllLoss !== "number" || Math.abs(rawNllLoss - 0.5) > 1e-6) {
    throw new Error(`${label} expected raw nllLoss to return numeric host loss`);
  }
  if (Math.abs(adapter.loss.negative_log_likelihood([-3, -2, -0.5], [2], { numClasses: 3 }) - rawNllLoss) > 1e-6) {
    throw new Error(`${label} expected raw negative_log_likelihood alias to match nllLoss`);
  }
  const classTargets = adapter.loss.classTargets([2]);
  if (!(classTargets instanceof Uint32Array) || classTargets.length !== 1 || classTargets[0] !== 2) {
    throw new Error(`${label} expected classTargets to return a Uint32Array`);
  }
  const logits = adapter.tensor([[1, 2, 3]], [1, 3]).requiresGrad_();
  const tensorCrossEntropy = adapter.loss.crossEntropy(logits, classTargets);
  if (!(tensorCrossEntropy instanceof adapter.Tensor)) {
    throw new Error(`${label} expected tensor crossEntropy to return a Tensor`);
  }
  expectClose(tensorCrossEntropy.data, [rawCrossEntropy], `${label} tensor crossEntropy value`);
  tensorCrossEntropy.backward();
  expectClose(logits.grad, [0.09003057, 0.24472848, -0.33475906], `${label} tensor crossEntropy gradient`);
  const logProbabilities = adapter.tensor([[-3, -2, -0.5]], [1, 3]).requiresGrad_();
  const tensorNllLoss = adapter.loss.negativeLogLikelihood(logProbabilities, classTargets);
  if (!(tensorNllLoss instanceof adapter.Tensor)) {
    throw new Error(`${label} expected tensor nllLoss to return a Tensor`);
  }
  expectClose(tensorNllLoss.data, [rawNllLoss], `${label} tensor nllLoss value`);
  tensorNllLoss.backward();
  expectClose(logProbabilities.grad, [0, 0, -1], `${label} tensor nllLoss gradient`);
  const sumLogProbabilities = adapter.tensor([-3, -2, -0.5, -1, -0.25, -2], [2, 3]).requiresGrad_();
  const tensorNllSumLoss = adapter.loss.nll_loss(sumLogProbabilities, [2, 1], { classes: 3, reduction: "sum" });
  expectClose(tensorNllSumLoss.data, [0.75], `${label} tensor nllLoss sum value`);
  tensorNllSumLoss.backward();
  expectClose(sumLogProbabilities.grad, [0, 0, -1, 0, -1, 0], `${label} tensor nllLoss sum gradient`);
  if (
    adapter.train.accuracy([1, 4, 0, 5, 2, 1], [1, 0], { classes: 3 }) !== 1 ||
    adapter.train.classificationAccuracy(adapter.tensor([[1, 4, 0], [5, 2, 1]], [2, 3]), [1, 0]) !== 1 ||
    adapter.train.classification_accuracy([1, 4, 0, 5, 2, 1], [1, 2], { numClasses: 3 }) !== 0.5
  ) {
    throw new Error(`${label} expected train.accuracy helpers to score class logits`);
  }
  if (
    adapter.train.topKAccuracy([3, 2, 1, 1, 3, 2], [1, 2], { classes: 3, k: 2 }) !== 1 ||
    adapter.train.top_k_accuracy(adapter.tensor([[3, 2, 1], [1, 3, 2]], [2, 3]), [1, 2], { top_k: 1 }) !== 0
  ) {
    throw new Error(`${label} expected train top-k accuracy helpers to score class logits`);
  }
  const confusion = adapter.train.confusionMatrix([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 });
  const confusionAlias = adapter.train.confusion_matrix(adapter.tensor([[5, 1], [1, 5], [5, 1]], [3, 2]), [0, 0, 1]);
  if (
    !Object.isFrozen(confusion) ||
    !Object.isFrozen(confusion[0]) ||
    confusion.length !== 2 ||
    confusion[0].join(",") !== "1,1" ||
    confusion[1].join(",") !== "1,0" ||
    confusionAlias[0].join(",") !== "1,1" ||
    confusionAlias[1].join(",") !== "1,0"
  ) {
    throw new Error(`${label} expected train confusion matrix helpers to count true rows and predicted columns`);
  }
  const classReport = adapter.train.classificationReport([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 });
  const classReportAlias = adapter.train.classification_report(adapter.tensor([[5, 1], [1, 5], [5, 1]], [3, 2]), [0, 0, 1]);
  if (
    !Object.isFrozen(classReport) ||
    !Object.isFrozen(classReport.perClass) ||
    !Object.isFrozen(classReport.perClass[0]) ||
    classReport.classes !== 2 ||
    classReport.support !== 3 ||
    Math.abs(classReport.accuracy - 1 / 3) > 1e-12 ||
    Math.abs(classReport.macroF1 - 0.25) > 1e-12 ||
    Math.abs(classReport.weighted_f1 - 1 / 3) > 1e-12 ||
    classReport.per_class[0].support !== 2 ||
    classReport.perClass[1].precision !== 0 ||
    classReportAlias.confusion_matrix[0].join(",") !== "1,1" ||
    Math.abs(classReportAlias.macro_f1 - 0.25) > 1e-12
  ) {
    throw new Error(`${label} expected train classification report helpers to summarize confusion matrix metrics`);
  }
  const classPredictions = adapter.train.classPredictions([0.1, 0.9, 2, 1], { classes: 2 });
  const classPredictionsAlias = adapter.train.predict_classes(adapter.tensor([[0.1, 0.9], [2, 1]], [2, 2]), { numClasses: 2 });
  if (
    !(classPredictions instanceof Uint32Array) ||
    classPredictions.length !== 2 ||
    classPredictions[0] !== 1 ||
    classPredictions[1] !== 0 ||
    classPredictionsAlias[0] !== 1 ||
    classPredictionsAlias[1] !== 0
  ) {
    throw new Error(`${label} expected train class prediction helpers to return class-id tensors`);
  }
  if (
    adapter.train.binaryAccuracy([0.1, 0.9, 0.7, 0.2], [0, 1, 1, 0]) !== 1 ||
    adapter.train.binary_accuracy(adapter.tensor([0.1, 0.9, 0.4, 0.2], [4]), [0, 1, 1, 0]) !== 0.75 ||
    adapter.train.binaryLogitsAccuracy([-2, 3, 1, -1], [0, 1, 1, 0]) !== 1 ||
    adapter.train.binary_logits_accuracy([-2, 3, -1, -1], [0, 1, 1, 0]) !== 0.75
  ) {
    throw new Error(`${label} expected train binary accuracy helpers to score probabilities and logits`);
  }
  const mseLoss = adapter.loss.mseLoss();
  const mseLossAlias = adapter.loss.mse_loss();
  const mseLossClass = new adapter.loss.MSELoss();
  const nnMseLossClass = new adapter.nn.MSELoss();
  const mseLossSum = adapter.loss.mseLoss({ reduction: "sum" });
  const nnMseLossSum = new adapter.nn.MSELoss({ reduction: "sum" });
  if (
    mseLoss.kind !== "mse-loss" ||
    mseLoss.reduction !== "mean" ||
    mseLossSum.reduction !== "sum" ||
    nnMseLossSum.reduction !== "sum" ||
    !(mseLoss.forward(adapter.tensor([1, 3], [2]), adapter.tensor([1, 1], [2])) instanceof adapter.Tensor) ||
    mseLoss.call([1, 3], [1, 1]) !== 2 ||
    mseLossAlias.call([1, 3], [1, 1]) !== 2 ||
    mseLossSum.forward([1, 3], [1, 1]) !== 4 ||
    mseLossClass.forward([1, 3], [1, 1]) !== 2 ||
    nnMseLossClass.forward([1, 3], [1, 1]) !== 2 ||
    nnMseLossSum.forward([1, 3], [1, 1]) !== 4 ||
    mseLossClass.__call__([1, 3], [1, 1]) !== 2
  ) {
    throw new Error(`${label} expected PyTorch-style MSELoss helpers to wrap loss.mse from loss and nn`);
  }
  expectThrowIncludes(
    () => adapter.loss.mse([1], [1], { reduction: "none" }),
    'MSELoss reduction must be "mean" or "sum"',
    `${label} unsupported loss reduction`
  );
  const rawL1 = adapter.loss.l1([1, 3], [1, 1]);
  const rawMeanAbsoluteError = adapter.loss.meanAbsoluteError([1, 3], [1, 1]);
  const l1Loss = adapter.loss.l1Loss();
  const l1LossAlias = adapter.loss.l1_loss();
  const l1LossClass = new adapter.loss.L1Loss();
  const nnL1LossClass = new adapter.nn.L1Loss();
  if (
    rawL1 !== 1 ||
    rawMeanAbsoluteError !== 1 ||
    l1Loss.kind !== "l1-loss" ||
    l1Loss.reduction !== "mean" ||
    !(l1Loss.forward(adapter.tensor([1, 3], [2]), adapter.tensor([1, 1], [2])) instanceof adapter.Tensor) ||
    l1Loss.call([1, 3], [1, 1]) !== 1 ||
    l1LossAlias.call([1, 3], [1, 1]) !== 1 ||
    l1LossClass.forward([1, 3], [1, 1]) !== 1 ||
    nnL1LossClass.forward([1, 3], [1, 1]) !== 1 ||
    l1LossClass.__call__([1, 3], [1, 1]) !== 1
  ) {
    throw new Error(`${label} expected PyTorch-style L1Loss helpers to wrap loss.l1 from loss and nn`);
  }
  const rawHuber = adapter.loss.huber([1, 3], [1, 1]);
  const huberLoss = adapter.loss.huberLoss();
  const huberLossAlias = adapter.loss.huber_loss({ delta: 1 });
  const huberLossClass = new adapter.loss.HuberLoss({ delta: 1 });
  const nnHuberLossClass = new adapter.nn.HuberLoss({ delta: 1 });
  const rawSmoothL1 = adapter.loss.smoothL1([1, 3], [1, 1]);
  const rawSmoothL1Alias = adapter.loss.smooth_l1([1, 3], [1, 1]);
  const smoothL1Loss = adapter.loss.smoothL1Loss();
  const smoothL1LossAlias = adapter.loss.smooth_l1_loss({ beta: 1 });
  const smoothL1LossClass = new adapter.loss.SmoothL1Loss({ beta: 1 });
  const nnSmoothL1LossClass = new adapter.nn.SmoothL1Loss({ beta: 1 });
  if (
    rawHuber !== 0.75 ||
    huberLoss.kind !== "huber-loss" ||
    huberLoss.reduction !== "mean" ||
    huberLoss.delta !== 1 ||
    !(huberLoss.forward(adapter.tensor([1, 3], [2]), adapter.tensor([1, 1], [2])) instanceof adapter.Tensor) ||
    huberLoss.call([1, 3], [1, 1]) !== 0.75 ||
    huberLossAlias.call([1, 3], [1, 1]) !== 0.75 ||
    huberLossClass.forward([1, 3], [1, 1]) !== 0.75 ||
    nnHuberLossClass.forward([1, 3], [1, 1]) !== 0.75 ||
    huberLossClass.__call__([1, 3], [1, 1]) !== 0.75
  ) {
    throw new Error(`${label} expected PyTorch-style HuberLoss helpers to wrap loss.huber from loss and nn`);
  }
  if (
    rawSmoothL1 !== 0.75 ||
    rawSmoothL1Alias !== 0.75 ||
    smoothL1Loss.kind !== "smooth-l1-loss" ||
    smoothL1Loss.reduction !== "mean" ||
    smoothL1Loss.beta !== 1 ||
    !(smoothL1Loss.forward(adapter.tensor([1, 3], [2]), adapter.tensor([1, 1], [2])) instanceof adapter.Tensor) ||
    smoothL1Loss.call([1, 3], [1, 1]) !== 0.75 ||
    smoothL1LossAlias.call([1, 3], [1, 1]) !== 0.75 ||
    smoothL1LossClass.forward([1, 3], [1, 1]) !== 0.75 ||
    nnSmoothL1LossClass.forward([1, 3], [1, 1]) !== 0.75 ||
    smoothL1LossClass.__call__([1, 3], [1, 1]) !== 0.75
  ) {
    throw new Error(`${label} expected PyTorch-style SmoothL1Loss helpers to wrap loss.smoothL1 from loss and nn`);
  }
  const rawBce = adapter.loss.bce([0.25, 0.75], [0, 1]);
  const rawBinaryCrossEntropy = adapter.loss.binaryCrossEntropy([0.25, 0.75], [0, 1]);
  const rawBinaryCrossEntropyAlias = adapter.loss.binary_cross_entropy([0.25, 0.75], [0, 1]);
  const bceLoss = adapter.loss.bceLoss();
  const bceLossAlias = adapter.loss.bce_loss();
  const bceLossClass = new adapter.loss.BCELoss();
  const nnBceLossClass = new adapter.nn.BCELoss();
  if (
    Math.abs(rawBce - 0.2876820724517809) > 1e-9 ||
    Math.abs(rawBinaryCrossEntropy - 0.2876820724517809) > 1e-9 ||
    Math.abs(rawBinaryCrossEntropyAlias - 0.2876820724517809) > 1e-9 ||
    bceLoss.kind !== "bce-loss" ||
    bceLoss.reduction !== "mean" ||
    bceLoss.eps !== 1e-7 ||
    !(bceLoss.forward(adapter.tensor([0.25, 0.75], [2]), adapter.tensor([0, 1], [2])) instanceof adapter.Tensor) ||
    Math.abs(bceLoss.call([0.25, 0.75], [0, 1]) - 0.2876820724517809) > 1e-9 ||
    Math.abs(bceLossAlias.call([0.25, 0.75], [0, 1]) - 0.2876820724517809) > 1e-9 ||
    Math.abs(bceLossClass.forward([0.25, 0.75], [0, 1]) - 0.2876820724517809) > 1e-9 ||
    Math.abs(nnBceLossClass.forward([0.25, 0.75], [0, 1]) - 0.2876820724517809) > 1e-9 ||
    Math.abs(bceLossClass.__call__([0.25, 0.75], [0, 1]) - 0.2876820724517809) > 1e-9
  ) {
    throw new Error(`${label} expected PyTorch-style BCELoss helpers to wrap loss.bce from loss and nn`);
  }
  const rawBceWithLogits = adapter.loss.bceWithLogits([0, 2], [0, 1]);
  const rawBceWithLogitsAlias = adapter.loss.binary_cross_entropy_with_logits([0, 2], [0, 1]);
  const bceWithLogitsLoss = adapter.loss.bceWithLogitsLoss();
  const bceWithLogitsLossAlias = adapter.loss.bce_with_logits_loss();
  const bceWithLogitsLossClass = new adapter.loss.BCEWithLogitsLoss();
  const nnBceWithLogitsLossClass = new adapter.nn.BCEWithLogitsLoss();
  const nnBceWithLogitsLossSum = new adapter.nn.BCEWithLogitsLoss({ reduction: "sum" });
  if (
    Math.abs(rawBceWithLogits - 0.41003759580145885) > 1e-9 ||
    Math.abs(rawBceWithLogitsAlias - 0.41003759580145885) > 1e-9 ||
    bceWithLogitsLoss.kind !== "bce-with-logits-loss" ||
    bceWithLogitsLoss.reduction !== "mean" ||
    nnBceWithLogitsLossSum.reduction !== "sum" ||
    !(bceWithLogitsLoss.forward(adapter.tensor([0, 2], [2]), adapter.tensor([0, 1], [2])) instanceof adapter.Tensor) ||
    Math.abs(bceWithLogitsLoss.call([0, 2], [0, 1]) - 0.41003759580145885) > 1e-9 ||
    Math.abs(bceWithLogitsLossAlias.call([0, 2], [0, 1]) - 0.41003759580145885) > 1e-9 ||
    Math.abs(bceWithLogitsLossClass.forward([0, 2], [0, 1]) - 0.41003759580145885) > 1e-9 ||
    Math.abs(nnBceWithLogitsLossClass.forward([0, 2], [0, 1]) - 0.41003759580145885) > 1e-9 ||
    Math.abs(nnBceWithLogitsLossSum.forward([0, 2], [0, 1]) - 0.8200751916029177) > 1e-9 ||
    Math.abs(bceWithLogitsLossClass.__call__([0, 2], [0, 1]) - 0.41003759580145885) > 1e-9
  ) {
    throw new Error(`${label} expected PyTorch-style BCEWithLogitsLoss helpers to wrap logits BCE from loss and nn`);
  }
  const crossEntropyLoss = adapter.loss.crossEntropyLoss({ classes: 3 });
  const crossEntropyLossAlias = adapter.loss.cross_entropy_loss({ numClasses: 3 });
  const crossEntropyLossClass = new adapter.loss.CrossEntropyLoss({ classes: 3 });
  const nnCrossEntropyLossClass = new adapter.nn.CrossEntropyLoss({ classes: 3 });
  const crossEntropyLossSum = adapter.loss.crossEntropyLoss({ classes: 3, reduction: "sum" });
  const nnCrossEntropyLossSum = new adapter.nn.CrossEntropyLoss({ classes: 3, reduction: "sum" });
  if (
    crossEntropyLoss.kind !== "cross-entropy-loss" ||
    crossEntropyLoss.classes !== 3 ||
    crossEntropyLoss.numClasses !== 3 ||
    crossEntropyLossSum.reduction !== "sum" ||
    nnCrossEntropyLossSum.reduction !== "sum" ||
    Math.abs(crossEntropyLoss.forward([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(crossEntropyLossSum.forward([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(nnCrossEntropyLossSum.forward([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(crossEntropyLossAlias.call([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(crossEntropyLossClass.forward([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(nnCrossEntropyLossClass.forward([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6 ||
    Math.abs(crossEntropyLossClass.__call__([1, 2, 3], [2]) - rawCrossEntropy) > 1e-6
  ) {
    throw new Error(`${label} expected PyTorch-style CrossEntropyLoss helpers to wrap loss.crossEntropy from loss and nn`);
  }
  const nllLoss = adapter.loss.nllLossModule({ classes: 3 });
  const nllLossAlias = adapter.loss.nll_loss_module({ numClasses: 3 });
  const nllLossClass = new adapter.loss.NLLLoss({ classes: 3 });
  const nnNllLossClass = new adapter.nn.NLLLoss({ classes: 3 });
  const nllLossSum = adapter.loss.nllLossModule({ classes: 3, reduction: "sum" });
  const nnNllLossSum = new adapter.nn.NLLLoss({ classes: 3, reduction: "sum" });
  if (
    nllLoss.kind !== "nll-loss" ||
    nllLoss.classes !== 3 ||
    nllLoss.numClasses !== 3 ||
    nllLossSum.reduction !== "sum" ||
    nnNllLossSum.reduction !== "sum" ||
    Math.abs(adapter.loss.nll_loss([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6 ||
    Math.abs(nllLossSum.forward([-3, -2, -0.5, -1, -0.25, -2], [2, 1]) - 0.75) > 1e-6 ||
    Math.abs(nnNllLossSum.forward([-3, -2, -0.5, -1, -0.25, -2], [2, 1]) - 0.75) > 1e-6 ||
    Math.abs(nllLoss.forward([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6 ||
    Math.abs(nllLossAlias.call([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6 ||
    Math.abs(nllLossClass.forward([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6 ||
    Math.abs(nnNllLossClass.forward([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6 ||
    Math.abs(nllLossClass.__call__([-3, -2, -0.5], [2]) - rawNllLoss) > 1e-6
  ) {
    throw new Error(`${label} expected PyTorch-style NLLLoss helpers to wrap loss.nllLoss from loss and nn`);
  }

  const adamWModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] });
  const adamW = adapter.optim.adamW(adamWModel, { lr: 0.01, weightDecay: 0.1 });
  const config = adamW.config();
  if (
    !Object.isFrozen(config) ||
    config.kind !== "adamw" ||
    typeof config.signature !== "string" ||
    !config.signature.startsWith("optimizer-config|kind=adamw|") ||
    config.decoupledWeightDecay !== true ||
    config.weightDecay !== 0.1 ||
    config.weight_decay !== 0.1
  ) {
    throw new Error(`${label} expected frozen AdamW config evidence`);
  }
  if (
    !adapter.optim.isOptimizerConfigSnapshot(config) ||
    adapter.optim.requireOptimizerConfigSnapshot(config).kind !== "adamw" ||
    adapter.optim.assertOptimizerConfigSnapshot(config).decoupledWeightDecay !== true ||
    adapter.optim.assert_optimizer_config_snapshot(config).signature !== config.signature ||
    adapter.optim.optimizerConfigSnapshotSignature(config) !== config.signature ||
    !adapter.optim.matchesOptimizerConfigSnapshotSignature(config, config.signature) ||
    adapter.optim.matches_optimizer_config_snapshot_signature(config, "wrong") ||
    adapter.optim.isOptimizerConfigSnapshot({ ...config })
  ) {
    throw new Error(`${label} expected optimizer config snapshot validators to accept only frozen signed evidence`);
  }
  const adamWStepEvidence = adapter.train.step(adamW, {
    loss: adapter.loss.mse(adamWModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])),
    inspect: true,
  });
  if (
    !Object.isFrozen(adamWStepEvidence) ||
    adamWStepEvidence.kind !== "zgml.train.step" ||
    typeof adamWStepEvidence.signature !== "string" ||
    !adamWStepEvidence.signature.startsWith("train-step|optimizer=adamw|parameters=2|before=0|after=1|advanced=1|loss=1|lossScalar=1|") ||
    adamWStepEvidence.optimizerKind !== "adamw" ||
    adamWStepEvidence.beforeStep !== 0 ||
    adamWStepEvidence.afterStep !== 1 ||
    adamWStepEvidence.stepAdvanced !== 1 ||
    adamWStepEvidence.lossScalar !== 1 ||
    adamWStepEvidence.zeroGradApplied !== true ||
    adamWStepEvidence.gradientsCleared !== true
  ) {
    throw new Error(`${label} expected AdamW TrainStepEvidence to pin step counter, loss, and cleared gradients`);
  }
  const state = adamW.stateDict();
  if (
    !Object.isFrozen(state) ||
    state.kind !== "adamw" ||
    typeof state.signature !== "string" ||
    !state.signature.startsWith("optimizer-state|kind=adamw|step=1|") ||
    state.step !== 1 ||
    state.entries.length !== 4
  ) {
    throw new Error(`${label} expected AdamW step and optimizer state evidence`);
  }
  if (
    !adapter.optim.isOptimizerStateSnapshot(state) ||
    adapter.optim.requireOptimizerStateSnapshot(state).step !== 1 ||
    adapter.optim.assertOptimizerStateSnapshot(state).entries.length !== 4 ||
    adapter.optim.assert_optimizer_state_snapshot(state).signature !== state.signature ||
    adapter.optim.optimizerStateSnapshotSignature(state) !== state.signature ||
    !adapter.optim.matchesOptimizerStateSnapshotSignature(state, state.signature) ||
    adapter.optim.matches_optimizer_state_snapshot_signature(state, "wrong") ||
    adapter.optim.isOptimizerStateSnapshot({ ...state })
  ) {
    throw new Error(`${label} expected optimizer state snapshot validators to accept only frozen signed evidence`);
  }
  const snakeState = adamW.state_dict();
  if (!Object.isFrozen(snakeState) || snakeState.kind !== "adamw" || snakeState.step !== state.step || snakeState.entries.length !== state.entries.length) {
    throw new Error(`${label} expected optimizer state_dict alias`);
  }
  const restored = new adapter.optim.AdamW(adamWModel.parameters(), { lr: 0.01, weightDecay: 0.1 });
  adapter.optim.loadStateDict(restored, state, { strict: true });
  if (restored.stateDict().step !== 1) {
    throw new Error(`${label} expected AdamW stateDict restore`);
  }
  const snakeRestored = new adapter.optim.AdamW(adamWModel.parameters(), { lr: 0.01, weightDecay: 0.1 });
  if (snakeRestored.load_state_dict(snakeState, { strict: true }) !== snakeRestored) {
    throw new Error(`${label} expected optimizer load_state_dict alias to return target`);
  }
  if (adapter.optim.load_state_dict(snakeRestored, snakeState, { strict: true }) !== snakeRestored || snakeRestored.state_dict().step !== 1) {
    throw new Error(`${label} expected namespace optimizer load_state_dict alias`);
  }
  adapter.loss.mse(adamWModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])).backward();
  if (gradNorm(adamWModel.parameters()) <= 0) {
    throw new Error(`${label} expected AdamW gradients before optimizer zero_grad alias`);
  }
  adamW.zero_grad();
  if (gradNorm(adamWModel.parameters()) !== 0) {
    throw new Error(`${label} expected optimizer zero_grad alias to clear gradients`);
  }
  adapter.loss.mse(adamWModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])).backward();
  adapter.optim.zero_grad(adamWModel);
  if (gradNorm(adamWModel.parameters()) !== 0) {
    throw new Error(`${label} expected namespace optimizer zero_grad alias to clear gradients`);
  }
  adapter.loss.mse(adamWModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])).backward();
  adamW.zeroGrad({ setToNone: true });
  if (adamWModel.parameters()[0].grad !== null || adamWModel.parameters()[1].grad !== null) {
    throw new Error(`${label} expected optimizer zeroGrad setToNone to clear parameter grad refs`);
  }
  adapter.loss.mse(adamWModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])).backward();
  adapter.optim.zero_grad(adamWModel, { set_to_none: true });
  if (adamWModel.parameters()[0].grad !== null || adamWModel.parameters()[1].grad !== null) {
    throw new Error(`${label} expected namespace optimizer zero_grad set_to_none to clear parameter grad refs`);
  }

  const rmspropModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] });
  const rmsprop = adapter.optim.rmsprop(rmspropModel, { lr: 0.01, alpha: 0.9, momentum: 0.1, weightDecay: 0.01, eps: 1e-8 });
  const rmspropConfig = rmsprop.config();
  if (
    !Object.isFrozen(rmspropConfig) ||
    rmspropConfig.kind !== "rmsprop" ||
    typeof rmspropConfig.signature !== "string" ||
    !rmspropConfig.signature.startsWith("optimizer-config|kind=rmsprop|") ||
    rmspropConfig.alpha !== 0.9 ||
    rmspropConfig.momentum !== 0.1 ||
    rmspropConfig.weightDecay !== 0.01 ||
    rmspropConfig.weight_decay !== 0.01
  ) {
    throw new Error(`${label} expected frozen RMSprop config evidence`);
  }
  if (
    !adapter.optim.isOptimizerConfigSnapshot(rmspropConfig) ||
    adapter.optim.requireOptimizerConfigSnapshot(rmspropConfig).kind !== "rmsprop" ||
    adapter.optim.optimizerConfigSnapshotSignature(rmspropConfig) !== rmspropConfig.signature
  ) {
    throw new Error(`${label} expected RMSprop config validators to accept signed evidence`);
  }
  const rmspropStepEvidence = adapter.train.step(rmsprop, {
    loss: adapter.loss.mse(rmspropModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])),
    inspect: true,
  });
  if (
    !Object.isFrozen(rmspropStepEvidence) ||
    rmspropStepEvidence.kind !== "zgml.train.step" ||
    typeof rmspropStepEvidence.signature !== "string" ||
    !rmspropStepEvidence.signature.startsWith("train-step|optimizer=rmsprop|parameters=2|before=0|after=1|advanced=1|loss=1|lossScalar=1|") ||
    rmspropStepEvidence.optimizerKind !== "rmsprop" ||
    rmspropStepEvidence.afterStep !== 1 ||
    rmspropStepEvidence.gradientsCleared !== true
  ) {
    throw new Error(`${label} expected RMSprop TrainStepEvidence to pin optimizer kind, step, loss, and cleared gradients`);
  }
  const rmspropState = rmsprop.stateDict();
  if (
    !Object.isFrozen(rmspropState) ||
    rmspropState.kind !== "rmsprop" ||
    typeof rmspropState.signature !== "string" ||
    !rmspropState.signature.startsWith("optimizer-state|kind=rmsprop|step=1|") ||
    rmspropState.step !== 1 ||
    rmspropState.entries.length !== 4
  ) {
    throw new Error(`${label} expected RMSprop optimizer state evidence`);
  }
  if (
    !adapter.optim.isOptimizerStateSnapshot(rmspropState) ||
    adapter.optim.requireOptimizerStateSnapshot(rmspropState).step !== 1 ||
    adapter.optim.optimizerStateSnapshotSignature(rmspropState) !== rmspropState.signature
  ) {
    throw new Error(`${label} expected RMSprop state validators to accept signed evidence`);
  }
  const restoredRmsprop = new adapter.optim.RMSprop(rmspropModel.parameters(), { lr: 0.01, alpha: 0.9, momentum: 0.1, weightDecay: 0.01 });
  if (adapter.optim.load_state_dict(restoredRmsprop, rmspropState, { strict: true }) !== restoredRmsprop || restoredRmsprop.state_dict().step !== 1) {
    throw new Error(`${label} expected RMSprop state_dict restore`);
  }

  const adagradModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] });
  const adagrad = adapter.optim.adagrad(adagradModel, { lr: 0.01, lrDecay: 0.001, weightDecay: 0.01, eps: 1e-10 });
  const adagradConfig = adagrad.config();
  if (
    !Object.isFrozen(adagradConfig) ||
    adagradConfig.kind !== "adagrad" ||
    typeof adagradConfig.signature !== "string" ||
    !adagradConfig.signature.startsWith("optimizer-config|kind=adagrad|") ||
    adagradConfig.lrDecay !== 0.001 ||
    adagradConfig.lr_decay !== 0.001 ||
    adagradConfig.weightDecay !== 0.01 ||
    adagradConfig.weight_decay !== 0.01
  ) {
    throw new Error(`${label} expected frozen Adagrad config evidence`);
  }
  if (
    !adapter.optim.isOptimizerConfigSnapshot(adagradConfig) ||
    adapter.optim.requireOptimizerConfigSnapshot(adagradConfig).kind !== "adagrad" ||
    adapter.optim.optimizerConfigSnapshotSignature(adagradConfig) !== adagradConfig.signature
  ) {
    throw new Error(`${label} expected Adagrad config validators to accept signed evidence`);
  }
  const adagradStepEvidence = adapter.train.step(adagrad, {
    loss: adapter.loss.mse(adagradModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0], [1])),
    inspect: true,
  });
  if (
    !Object.isFrozen(adagradStepEvidence) ||
    adagradStepEvidence.kind !== "zgml.train.step" ||
    typeof adagradStepEvidence.signature !== "string" ||
    !adagradStepEvidence.signature.startsWith("train-step|optimizer=adagrad|parameters=2|before=0|after=1|advanced=1|loss=1|lossScalar=1|") ||
    adagradStepEvidence.optimizerKind !== "adagrad" ||
    adagradStepEvidence.afterStep !== 1 ||
    adagradStepEvidence.gradientsCleared !== true
  ) {
    throw new Error(`${label} expected Adagrad TrainStepEvidence to pin optimizer kind, step, loss, and cleared gradients`);
  }
  const adagradState = adagrad.stateDict();
  if (
    !Object.isFrozen(adagradState) ||
    adagradState.kind !== "adagrad" ||
    typeof adagradState.signature !== "string" ||
    !adagradState.signature.startsWith("optimizer-state|kind=adagrad|step=1|") ||
    adagradState.step !== 1 ||
    adagradState.entries.length !== 2
  ) {
    throw new Error(`${label} expected Adagrad optimizer state evidence`);
  }
  if (
    !adapter.optim.isOptimizerStateSnapshot(adagradState) ||
    adapter.optim.requireOptimizerStateSnapshot(adagradState).step !== 1 ||
    adapter.optim.optimizerStateSnapshotSignature(adagradState) !== adagradState.signature
  ) {
    throw new Error(`${label} expected Adagrad state validators to accept signed evidence`);
  }
  const restoredAdagrad = new adapter.optim.Adagrad(adagradModel.parameters(), { lr: 0.01, lrDecay: 0.001, weightDecay: 0.01 });
  if (adapter.optim.load_state_dict(restoredAdagrad, adagradState, { strict: true }) !== restoredAdagrad || restoredAdagrad.state_dict().step !== 1) {
    throw new Error(`${label} expected Adagrad state_dict restore`);
  }
}

function expectStepParamsEvidence(compiledSession: Record<string, any>, hotParams: unknown, adapter: Record<string, any>, label: string) {
  for (const method of [
    "requireCanExecuteStepParams",
    "requireAllocationFreeStepParams",
    "requireRuntimeOutputAllocationFreeStepParams",
    "requireNoReadbackStepParams",
    "requireReadbackFreeStepParams",
    "requireHotStepParams",
  ]) {
    const evidence = compiledSession[method](hotParams);
    if (!Object.isFrozen(evidence) || !compiledSession.matchesStepParamsCompatibility(hotParams, evidence)) {
      throw new Error(`${label} expected ${method} evidence to be frozen and matchable`);
    }
    if (method === "requireCanExecuteStepParams" && evidence.canExecute !== true) {
      throw new Error(`${label} expected requireCanExecuteStepParams to return executable evidence`);
    }
    if (method === "requireHotStepParams" && (evidence.hotPath !== true || evidence.status !== "accepted")) {
      throw new Error(`${label} expected requireHotStepParams to return accepted hot-path evidence`);
    }
  }
  for (const [method, alias] of [
    ["requireCanExecuteStepParams", "require_can_execute_step_params"],
    ["requireAllocationFreeStepParams", "require_allocation_free_step_params"],
    ["requireRuntimeOutputAllocationFreeStepParams", "require_runtime_output_allocation_free_step_params"],
    ["requireNoReadbackStepParams", "require_no_readback_step_params"],
    ["requireReadbackFreeStepParams", "require_readback_free_step_params"],
    ["requireHotStepParams", "require_hot_step_params"],
  ]) {
    if (compiledSession[method](hotParams).stepParamsSignature !== compiledSession[alias](hotParams).stepParamsSignature) {
      throw new Error(`${label} expected ${alias} to match ${method} evidence`);
    }
  }
  const accepted = compiledSession.stepParamsCompatibility(hotParams);
  const executionPlan = compiledSession.executionPlan(hotParams);
  const requiredExecutionPlan = compiledSession.requireExecutionPlan(hotParams);
  const hotPathPlan = compiledSession.hotPathPlan(hotParams);
  const acceptedAlias = compiledSession.step_params_compatibility(hotParams);
  const executionPlanAlias = compiledSession.execution_plan(hotParams);
  const requiredExecutionPlanAlias = compiledSession.require_execution_plan(hotParams);
  const hotPathPlanAlias = compiledSession.hot_path_plan(hotParams);
  if (
    !Object.isFrozen(accepted) ||
    accepted.kind !== "zgml.step-params.compatibility" ||
    accepted.status !== "accepted" ||
    !Object.isFrozen(accepted.diagnostics) ||
    compiledSession.preflightStepParams(hotParams).stepParamsSignature !== accepted.stepParamsSignature ||
    compiledSession.preflight_step_params(hotParams).stepParamsSignature !== accepted.stepParamsSignature ||
    acceptedAlias.stepParamsSignature !== accepted.stepParamsSignature ||
    compiledSession.step_contract().signature !== compiledSession.stepContract().signature ||
    !compiledSession.matches_step_contract_signature(compiledSession.stepContract().signature) ||
    !compiledSession.matchesStepParamsSignature(hotParams, accepted.stepParamsSignature) ||
    !compiledSession.matches_step_params_signature(hotParams, accepted.stepParamsSignature) ||
    !compiledSession.matchesStepParamsCompatibility(hotParams, accepted)
    || !compiledSession.matches_step_params_compatibility(hotParams, accepted)
  ) {
    throw new Error(`${label} expected accepted StepParams compatibility evidence to be frozen and matchable`);
  }
  if (
    !Object.isFrozen(executionPlan) ||
    !Object.isFrozen(hotPathPlan) ||
    executionPlan.kind !== "zgml.session.execution-plan" ||
    hotPathPlan.kind !== "zgml.session.execution-plan" ||
    typeof executionPlan.signature !== "string" ||
    !executionPlan.signature.startsWith("session-execution-plan|") ||
    hotPathPlan.signature !== executionPlan.signature ||
    executionPlanAlias.signature !== executionPlan.signature ||
    requiredExecutionPlan.signature !== executionPlan.signature ||
    requiredExecutionPlanAlias.signature !== executionPlan.signature ||
    hotPathPlanAlias.signature !== hotPathPlan.signature ||
    !Object.isFrozen(executionPlan.compatibility) ||
    executionPlan.compatibility.stepParamsSignature !== accepted.stepParamsSignature ||
    executionPlan.contractSignature !== compiledSession.stepContract().signature ||
    executionPlan.stepParamsSignature !== accepted.stepParamsSignature ||
    executionPlan.hotPath !== true ||
    executionPlan.hotPathStatus !== "hot" ||
    executionPlan.defaultHotPath !== false ||
    executionPlan.runtimeOutputAllocationFree !== true ||
    executionPlan.readbackFree !== true ||
    executionPlan.inputOwnership !== accepted.inputOwnership ||
    executionPlan.readsInput !== accepted.readsInput ||
    executionPlan.outputOwnership !== accepted.outputOwnership ||
    executionPlan.outputReturnOwnership !== accepted.outputReturnOwnership ||
    executionPlan.writesOutput !== accepted.writesOutput ||
    executionPlan.inputElementType !== accepted.inputElementType ||
    executionPlan.outputElementType !== accepted.outputElementType ||
    executionPlan.inputElementLength !== accepted.inputElementLength ||
    executionPlan.outputElementLength !== accepted.outputElementLength ||
    executionPlan.inputShapeSignature !== accepted.inputShapeSignature ||
    executionPlan.outputShapeSignature !== accepted.outputShapeSignature ||
    executionPlan.rejectionCode !== null ||
    hotPathPlan.stepParamsSignature !== accepted.stepParamsSignature
  ) {
    throw new Error(`${label} expected executionPlan/hotPathPlan to summarize accepted StepParams evidence`);
  }
  const rejectedParams = { input: adapter.tensor([1, 2, 3], [3]), output: false };
  const rejected = compiledSession.stepParamsCompatibility(rejectedParams);
  const rejectedPlan = compiledSession.executionPlan(rejectedParams);
  if (
    !Object.isFrozen(rejected) ||
    rejected.kind !== "zgml.step-params.compatibility" ||
    !Object.isFrozen(rejected.diagnostics) ||
    !Object.isFrozen(rejected.diagnostics[0]) ||
    rejected.status !== "rejected" ||
    rejected.rejectionCode !== "invalid-input" ||
    rejected.diagnostics[0].expectedLength !== 2 ||
    rejected.diagnostics[0].actualLength !== 3 ||
    compiledSession.acceptsStepParams(rejectedParams) !== false ||
    compiledSession.accepts_step_params(rejectedParams) !== false ||
    compiledSession.can_execute_step_params(rejectedParams) !== false ||
    compiledSession.acceptsHotStepParams(rejectedParams) !== false ||
    compiledSession.accepts_hot_step_params(rejectedParams) !== false ||
    compiledSession.accepts_allocation_free_step_params(rejectedParams) !== false ||
    compiledSession.accepts_runtime_output_allocation_free_step_params(rejectedParams) !== false ||
    compiledSession.accepts_no_readback_step_params(rejectedParams) !== false ||
    compiledSession.accepts_readback_free_step_params(rejectedParams) !== false ||
    !compiledSession.matchesStepParamsSignature(rejectedParams, rejected.stepParamsSignature) ||
    !compiledSession.matches_step_params_signature(rejectedParams, rejected.stepParamsSignature) ||
    !compiledSession.matchesStepParamsCompatibility(rejectedParams, rejected)
    || !compiledSession.matches_step_params_compatibility(rejectedParams, rejected)
  ) {
    throw new Error(`${label} expected rejected StepParams compatibility evidence to be frozen and matchable`);
  }
  if (
    !Object.isFrozen(rejectedPlan) ||
    rejectedPlan.accepted !== false ||
    typeof rejectedPlan.signature !== "string" ||
    !rejectedPlan.signature.startsWith("session-execution-plan|") ||
    rejectedPlan.signature === executionPlan.signature ||
    rejectedPlan.hotPathStatus !== "rejected" ||
    rejectedPlan.hotPathBlockers.join("|") !== "rejected" ||
    rejectedPlan.compatibility.stepParamsSignature !== rejected.stepParamsSignature ||
    rejectedPlan.inputOwnership !== "rejected" ||
    rejectedPlan.outputOwnership !== "rejected" ||
    rejectedPlan.rejectionCode !== rejected.rejectionCode ||
    rejectedPlan.diagnostics[0].code !== rejected.diagnostics[0].code
  ) {
    throw new Error(`${label} expected executionPlan to summarize rejected StepParams evidence without inventing fields`);
  }
  expectThrowIncludes(
    () => compiledSession.requireExecutionPlan(rejectedParams),
    "StepParams are not executable: status=rejected",
    `${label} rejected StepParams requireExecutionPlan`,
  );
  expectThrowIncludes(
    () => compiledSession.require_execution_plan(rejectedParams),
    "StepParams are not executable: status=rejected",
    `${label} rejected StepParams require_execution_plan`,
  );
  expectThrowIncludes(
    () => compiledSession.requireHotStepParams(rejectedParams),
    "StepParams are not hot-path compatible: status=rejected blockers=rejected rejection=invalid-input",
    `${label} rejected StepParams requireHot`,
  );
}

function expectStateAndCheckpointRoundTrip(adapter: Record<string, any>, label: string) {
  const checkpointModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0.25] });
  const checkpointOptimizer = adapter.optim.sgd(checkpointModel, { lr: 0.1, momentum: 0.5 });
  adapter.loss.mse(checkpointModel.forward(adapter.tensor([1, 2], [2])), adapter.tensor([0.5], [1])).backward();
  checkpointOptimizer.step();
  const modelState = checkpointModel.stateDict("head");
  const optimizerState = checkpointOptimizer.stateDict();
  if (
    !Object.isFrozen(modelState) ||
    !Object.isFrozen(modelState["head.weight"]) ||
    typeof modelState["head.weight"].signature !== "string" ||
    !modelState["head.weight"].signature.startsWith("module-state-entry|") ||
    !Object.isFrozen(optimizerState) ||
    typeof optimizerState.signature !== "string" ||
    !optimizerState.signature.startsWith("optimizer-state|kind=sgd|step=1|") ||
    !Object.isFrozen(optimizerState.entries) ||
    typeof optimizerState.entries[0].signature !== "string" ||
    !optimizerState.entries[0].signature.startsWith("optimizer-state-entry|name=velocity.0|") ||
    optimizerState.step !== 1
  ) {
    throw new Error(`${label} expected frozen module and optimizer state evidence`);
  }
  expectThrowIncludes(
    () => checkpointModel.loadStateDict({ weight: modelState["head.weight"] }, { strict: true, prefix: "head" }),
    "state dict has unexpected parameter weight",
    `${label} strict prefixed module load`,
  );
  expectThrowIncludes(
    () => checkpointModel.loadStateDict(
      Object.create({
        "head.weight": modelState["head.weight"],
        "head.bias": modelState["head.bias"],
      }),
      { strict: true, prefix: "head" },
    ),
    "state dict is missing parameter head.weight",
    `${label} module loadStateDict rejects inherited state`,
  );
  const savedWeight = numericSnapshot(checkpointModel.weight);
  const savedBias = numericSnapshot(checkpointModel.bias);
  checkpointModel.weight.fill(9);
  checkpointModel.bias.fill(9);
  checkpointOptimizer.velocity[0].fill(9);
  checkpointOptimizer.velocity[1].fill(9);
  checkpointModel.loadStateDict(modelState, { strict: true, prefix: "head", validateOnly: true });
  expectClose(checkpointModel.weight, [9, 9], `${label} module validateOnly keeps weight`);
  checkpointModel.loadStateDict(modelState, { strict: true, prefix: "head" });
  expectClose(checkpointModel.weight, savedWeight, `${label} module loadStateDict restores weight`);
  expectClose(checkpointModel.bias, savedBias, `${label} module loadStateDict restores bias`);
  const freshOptimizer = adapter.optim.sgd(checkpointModel, { lr: 0.1, momentum: 0.5 });
  freshOptimizer.velocity[0].fill(5);
  freshOptimizer.velocity[1].fill(5);
  adapter.optim.loadStateDict(freshOptimizer, optimizerState, { strict: true, validateOnly: true });
  expectClose(freshOptimizer.velocity[0], [5, 5], `${label} optimizer validateOnly keeps velocity 0`);
  expectClose(freshOptimizer.velocity[1], [5], `${label} optimizer validateOnly keeps velocity 1`);
  if (freshOptimizer.stateDict().step !== 0) {
    throw new Error(`${label} optimizer validateOnly must not update step`);
  }
  adapter.optim.loadStateDict(freshOptimizer, optimizerState, { strict: true });
  if (freshOptimizer.stateDict().entries.length !== optimizerState.entries.length || freshOptimizer.stateDict().step !== 1) {
    throw new Error(`${label} optimizer loadStateDict expected matching entry count and step`);
  }
  freshOptimizer.velocity[0].fill(8);
  freshOptimizer.velocity[1].fill(8);
  adapter.optim.load_state_dict(freshOptimizer, optimizerState, { strict: true, validateOnly: true });
  expectClose(freshOptimizer.velocity[0], [8, 8], `${label} optimizer load_state_dict validateOnly keeps velocity 0`);
  if (adapter.optim.state_dict(freshOptimizer).step !== 1) {
    throw new Error(`${label} optimizer state_dict alias expected restored step`);
  }
  adapter.optim.load_state_dict(freshOptimizer, optimizerState, { strict: true });
  const checkpointScheduler = adapter.optim.stepLR(freshOptimizer, { stepSize: 2, gamma: 0.5 });
  checkpointScheduler.step();
  checkpointScheduler.step();
  const checkpointSchedulerState = checkpointScheduler.stateDict();
  if (
    typeof checkpointSchedulerState.signature !== "string" ||
    !checkpointSchedulerState.signature.startsWith("lr-scheduler-state|kind=step-lr|step=2|") ||
    checkpointSchedulerState.base_lr !== checkpointSchedulerState.baseLr ||
    checkpointSchedulerState.last_lr !== checkpointSchedulerState.lastLr ||
    checkpointSchedulerState.stepSize !== 2 ||
    checkpointSchedulerState.step_size !== 2
  ) {
    throw new Error(`${label} expected scheduler state signature evidence`);
  }
  const checkpoint = adapter.checkpoint.create({
    metadata: { epoch: 1 },
    model: checkpointModel,
    optimizer: freshOptimizer,
    scheduler: checkpointScheduler,
    prefix: "head",
  });
  const checkpointJson = adapter.checkpoint.toJSON(checkpoint);
  const parsedCheckpoint = adapter.checkpoint.fromJSON(JSON.parse(JSON.stringify(checkpointJson)));
  const checkpointText = adapter.checkpoint.stringify(checkpoint, 2);
  const parsedCheckpointText = adapter.checkpoint.parse(checkpointText);
  const checkpointTextAlias = adapter.checkpoint.serialize(checkpoint, 2);
  const parsedCheckpointTextAlias = adapter.checkpoint.deserialize(checkpointTextAlias);
  if (
    !Object.isFrozen(checkpointJson) ||
    !Object.isFrozen(parsedCheckpoint) ||
    !Object.isFrozen(parsedCheckpointText) ||
    !Object.isFrozen(parsedCheckpointTextAlias) ||
    !Object.isFrozen(parsedCheckpoint.model) ||
    !Object.isFrozen(parsedCheckpointText.model) ||
    !Object.isFrozen(parsedCheckpointTextAlias.model) ||
    !Object.isFrozen(parsedCheckpoint.optimizer?.entries) ||
    !Object.isFrozen(parsedCheckpointText.optimizer?.entries) ||
    !Object.isFrozen(parsedCheckpointTextAlias.optimizer?.entries) ||
    !Object.isFrozen(parsedCheckpoint.scheduler) ||
    !Object.isFrozen(parsedCheckpointText.scheduler) ||
    !Object.isFrozen(parsedCheckpointTextAlias.scheduler) ||
    typeof parsedCheckpoint.model["head.weight"].signature !== "string" ||
    typeof parsedCheckpointText.model["head.weight"].signature !== "string" ||
    typeof parsedCheckpointTextAlias.model["head.weight"].signature !== "string" ||
    !parsedCheckpoint.model["head.weight"].signature.startsWith("checkpoint-tensor|") ||
    !parsedCheckpointText.model["head.weight"].signature.startsWith("checkpoint-tensor|") ||
    !parsedCheckpointTextAlias.model["head.weight"].signature.startsWith("checkpoint-tensor|") ||
    typeof parsedCheckpoint.optimizer.entries[0].signature !== "string" ||
    typeof parsedCheckpointText.optimizer.entries[0].signature !== "string" ||
    typeof parsedCheckpointTextAlias.optimizer.entries[0].signature !== "string" ||
    !parsedCheckpoint.optimizer.entries[0].signature.startsWith("checkpoint-tensor|name=velocity.0|") ||
    !parsedCheckpointText.optimizer.entries[0].signature.startsWith("checkpoint-tensor|name=velocity.0|") ||
    !parsedCheckpointTextAlias.optimizer.entries[0].signature.startsWith("checkpoint-tensor|name=velocity.0|") ||
    typeof parsedCheckpoint.scheduler.signature !== "string" ||
    typeof parsedCheckpointText.scheduler.signature !== "string" ||
    typeof parsedCheckpointTextAlias.scheduler.signature !== "string" ||
    !parsedCheckpoint.scheduler.signature.startsWith("lr-scheduler-state|kind=step-lr|") ||
    !parsedCheckpointText.scheduler.signature.startsWith("lr-scheduler-state|kind=step-lr|") ||
    !parsedCheckpointTextAlias.scheduler.signature.startsWith("lr-scheduler-state|kind=step-lr|") ||
    parsedCheckpoint.scheduler.stepSize !== 2 ||
    parsedCheckpointText.scheduler.stepSize !== 2 ||
    parsedCheckpointTextAlias.scheduler.stepSize !== 2 ||
    parsedCheckpoint.scheduler.base_lr !== parsedCheckpoint.scheduler.baseLr ||
    parsedCheckpointText.scheduler.base_lr !== parsedCheckpointText.scheduler.baseLr ||
    parsedCheckpointTextAlias.scheduler.base_lr !== parsedCheckpointTextAlias.scheduler.baseLr ||
    parsedCheckpoint.scheduler.last_lr !== parsedCheckpoint.scheduler.lastLr ||
    parsedCheckpointText.scheduler.last_lr !== parsedCheckpointText.scheduler.lastLr ||
    parsedCheckpointTextAlias.scheduler.last_lr !== parsedCheckpointTextAlias.scheduler.lastLr ||
    parsedCheckpoint.scheduler.step_size !== 2 ||
    parsedCheckpointText.scheduler.step_size !== 2 ||
    parsedCheckpointTextAlias.scheduler.step_size !== 2 ||
    parsedCheckpoint.format !== "zgml.checkpoint" ||
    parsedCheckpointText.format !== "zgml.checkpoint" ||
    parsedCheckpointTextAlias.format !== "zgml.checkpoint" ||
    parsedCheckpoint.version !== 1 ||
    parsedCheckpointText.version !== 1 ||
    parsedCheckpointTextAlias.version !== 1 ||
    checkpointTextAlias !== checkpointText ||
    !checkpointText.includes("\"format\": \"zgml.checkpoint\"")
  ) {
    throw new Error(`${label} expected checkpoint JSON helpers to normalize frozen checkpoint evidence; string helpers must match`);
  }
  if (
    adapter.checkpoint.serialize !== adapter.checkpoint.stringify ||
    adapter.checkpoint.deserialize !== adapter.checkpoint.parse ||
    adapter.checkpoint.module_state_dict !== adapter.checkpoint.moduleStateDict ||
    adapter.checkpoint.optimizer_state_dict !== adapter.checkpoint.optimizerStateDict ||
    adapter.checkpoint.scheduler_state_dict !== adapter.checkpoint.schedulerStateDict ||
    adapter.checkpoint.module_state_dict(checkpointModel).weight.signature !== adapter.checkpoint.moduleStateDict(checkpointModel).weight.signature ||
    adapter.checkpoint.optimizer_state_dict(freshOptimizer).entries.length !== adapter.checkpoint.optimizerStateDict(freshOptimizer).entries.length ||
    adapter.checkpoint.scheduler_state_dict(checkpointScheduler).signature !== adapter.checkpoint.schedulerStateDict(checkpointScheduler).signature
  ) {
    throw new Error(`${label} expected checkpoint serialize/deserialize and *_state_dict aliases to share checkpoint implementations`);
  }
  const snakeCheckpoint = adapter.checkpoint.fromJSON({
    format: "zgml.checkpoint",
    version: 1,
    scheduler: {
      kind: "step-lr",
      step: checkpointSchedulerState.step,
      base_lr: checkpointSchedulerState.base_lr,
      last_lr: checkpointSchedulerState.last_lr,
      gamma: checkpointSchedulerState.gamma,
      step_size: checkpointSchedulerState.step_size,
      optimizerKind: checkpointSchedulerState.optimizerKind,
    },
  });
  if (
    snakeCheckpoint.scheduler.stepSize !== 2 ||
    snakeCheckpoint.scheduler.step_size !== 2 ||
    snakeCheckpoint.scheduler.baseLr !== checkpointSchedulerState.baseLr ||
    snakeCheckpoint.scheduler.base_lr !== checkpointSchedulerState.baseLr ||
    snakeCheckpoint.scheduler.lastLr !== checkpointSchedulerState.lastLr ||
    snakeCheckpoint.scheduler.last_lr !== checkpointSchedulerState.lastLr
  ) {
    throw new Error(`${label} expected checkpoint scheduler JSON to accept snake_case aliases`);
  }
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({ format: "zgml.checkpoint", version: 1 }),
    "checkpoint JSON requires model, optimizer, scheduler, or a combination",
    `${label} checkpoint.fromJSON rejects empty checkpoint`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON(Object.create({
      format: "zgml.checkpoint",
      version: 1,
      model: {
        weight: { dtype: "f32", shape: [1], layout: "row-major", data: [1] },
      },
    })),
    "checkpoint must be a plain object",
    `${label} checkpoint.fromJSON rejects non-plain checkpoint object`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.inspect(Object.create({
      format: "zgml.checkpoint",
      version: 1,
      model: {
        weight: { dtype: "f32", shape: [1], layout: "row-major", data: [1] },
      },
    })),
    "checkpoint must be a plain object",
    `${label} checkpoint.inspect rejects non-plain checkpoint object`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      model: Object.create({
        weight: { dtype: "f32", shape: [1], layout: "row-major", data: [1] },
      }),
    }),
    "checkpoint model state must be a plain object",
    `${label} checkpoint.fromJSON rejects inherited model state`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.restore({
      format: "zgml.checkpoint",
      version: 1,
      model: Object.create({
        "head.weight": { dtype: "f32", shape: [2], layout: "row-major:linear.weight[in_features,out_features]", data: [1, -1] },
        "head.bias": { dtype: "f32", shape: [1], layout: "row-major:linear.bias[out_features]", data: [0.25] },
      }),
    }, { model: checkpointModel, strict: true, prefix: "head" }),
    "checkpoint model state must be a plain object",
    `${label} checkpoint.restore rejects inherited model state`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      model: {
        weight: Object.create({ dtype: "f32", shape: [1], layout: "row-major", data: [1] }),
      },
    }),
    "checkpoint model weight must be a plain object",
    `${label} checkpoint.fromJSON rejects inherited tensor entry`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.create({
      metadata: { epoch: 1, loss: Number.NaN },
      model: checkpointModel,
    }),
    "checkpoint metadata.loss number must be finite, got NaN",
    `${label} checkpoint.create rejects non-finite metadata`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.create({
      metadata: { epoch: 1, hook: () => null },
      model: checkpointModel,
    }),
    "checkpoint metadata.hook must be JSON-safe plain data",
    `${label} checkpoint.create rejects lossy function metadata`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      metadata: { createdAt: new Date(0) },
      model: {
        weight: { dtype: "f32", shape: [1], layout: "row-major", data: [1] },
      },
    }),
    "checkpoint metadata.createdAt must be JSON-safe plain data",
    `${label} checkpoint.fromJSON rejects non-plain metadata`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      model: {
        weight: { dtype: "i32", shape: [1], layout: "row-major", data: [1] },
      },
    }),
    "checkpoint model weight dtype must be f32, got i32",
    `${label} checkpoint.fromJSON rejects non-f32 model dtype`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      optimizer: {
        kind: "sgd",
        step: 1,
        paramCount: 1,
        entries: [
          { name: "velocity.0", dtype: "i32", shape: [1], layout: "row-major", data: [1] },
        ],
      },
    }),
    "checkpoint optimizer velocity.0 dtype must be f32, got i32",
    `${label} checkpoint.fromJSON rejects non-f32 optimizer dtype`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      optimizer: {
        kind: "sgd",
        step: 1,
        paramCount: 1,
        entries: [
          { name: "velocity.0", dtype: "f32", shape: [1], layout: "row-major", data: [1] },
          { name: "velocity.0", dtype: "f32", shape: [1], layout: "row-major", data: [2] },
        ],
      },
    }),
    "checkpoint optimizer has duplicate entry velocity.0",
    `${label} checkpoint.fromJSON rejects duplicate optimizer entries`,
  );
  expectThrowIncludes(
    () => adapter.checkpoint.fromJSON({
      format: "zgml.checkpoint",
      version: 1,
      scheduler: { kind: "step-lr", step: 1, baseLr: 0.1, lastLr: 0.05, gamma: Number.NaN, stepSize: 2 },
    }),
    "checkpoint scheduler gamma must be a positive finite number",
    `${label} checkpoint.fromJSON rejects non-finite scheduler state`,
  );
  const checkpointInfo = adapter.checkpoint.inspect(checkpoint);
  if (
    !Object.isFrozen(checkpoint) ||
    !Object.isFrozen(checkpointInfo) ||
    typeof checkpointInfo.signature !== "string" ||
    !checkpointInfo.signature.startsWith("checkpoint-inspection|format=zgml.checkpoint|version=1|") ||
    checkpointInfo.modelParameterNames.join("|") !== "head.weight|head.bias" ||
    checkpointInfo.optimizerEntryNames.join("|") !== "velocity.0|velocity.1" ||
    checkpointInfo.optimizerStep !== 1 ||
    checkpointInfo.schedulerKind !== "step-lr" ||
    checkpointInfo.schedulerStep !== 2 ||
    checkpointInfo.schedulerLastLr !== checkpointSchedulerState.lastLr ||
    checkpointInfo.schedulerOptimizerKind !== "sgd"
  ) {
    throw new Error(`${label} expected checkpoint inspect evidence for model, optimizer, and scheduler state`);
  }
  const checkpointWeightInfo = adapter.checkpoint.modelParameterInfo(checkpoint, "head.weight");
  const checkpointBiasInfo = adapter.checkpoint.modelParameterInfo(checkpoint, 1);
  const checkpointVelocityInfo = adapter.checkpoint.optimizerEntryInfo(checkpoint, 0);
  const checkpointVelocityByNameInfo = adapter.checkpoint.optimizerEntryInfo(checkpoint, "velocity.1");
  if (
    !Object.isFrozen(checkpointWeightInfo) ||
    typeof checkpointWeightInfo.signature !== "string" ||
    !checkpointWeightInfo.signature.startsWith("checkpoint-tensor-inspection|name=head.weight|") ||
    checkpointWeightInfo.name !== "head.weight" ||
    checkpointWeightInfo.scalarCount !== 2 ||
    !Object.isFrozen(checkpointBiasInfo) ||
    checkpointBiasInfo.name !== "head.bias" ||
    checkpointBiasInfo.scalarCount !== 1 ||
    !Object.isFrozen(checkpointVelocityInfo) ||
    typeof checkpointVelocityInfo.signature !== "string" ||
    !checkpointVelocityInfo.signature.startsWith("checkpoint-tensor-inspection|name=velocity.0|") ||
    checkpointVelocityInfo.name !== "velocity.0" ||
    checkpointVelocityInfo.scalarCount !== 2 ||
    !Object.isFrozen(checkpointVelocityByNameInfo) ||
    checkpointVelocityByNameInfo.name !== "velocity.1" ||
    checkpointVelocityByNameInfo.scalarCount !== 1
  ) {
    throw new Error(`${label} expected checkpoint name/index lookup evidence for model parameters and optimizer entries in both directions`);
  }
  checkpointModel.weight.fill(-7);
  checkpointModel.bias.fill(-7);
  freshOptimizer.velocity[0].fill(-7);
  freshOptimizer.velocity[1].fill(-7);
  checkpointScheduler.loadStateDict({ ...checkpointSchedulerState, lastLr: checkpointSchedulerState.lastLr * 4 }, { strict: true });
  checkpointScheduler.loadStateDict(checkpointSchedulerState, { strict: true, validateOnly: true });
  if (checkpointScheduler.getLastLr() !== checkpointSchedulerState.lastLr * 4) {
    throw new Error(`${label} scheduler validateOnly must not mutate lr state`);
  }
  adapter.checkpoint.restore(parsedCheckpoint, { model: checkpointModel, optimizer: freshOptimizer, scheduler: checkpointScheduler, strict: true, prefix: "head" });
  expectClose(checkpointModel.weight, savedWeight, `${label} checkpoint restore weight`);
  expectClose(checkpointModel.bias, savedBias, `${label} checkpoint restore bias`);
  expectClose(freshOptimizer.velocity[0], numericSnapshot(optimizerState.entries[0].data), `${label} checkpoint restore optimizer velocity 0`);
  if (checkpointScheduler.getLastLr() !== checkpointSchedulerState.lastLr || adapter.optim.config(freshOptimizer).lr !== checkpointSchedulerState.lastLr) {
    throw new Error(`${label} checkpoint restore expected scheduler and optimizer lr state`);
  }
  const checkpointSchedulerOnly = adapter.checkpoint.create({ scheduler: checkpointScheduler });
  if (adapter.checkpoint.inspect(checkpointSchedulerOnly).hasScheduler !== true) {
    throw new Error(`${label} expected scheduler-only checkpoint support`);
  }
}

function expectProgramCapabilityEvidence(compiledProgram: Record<string, any>, label: string) {
  const capabilities = compiledProgram.capabilities();
  if (
    !Object.isFrozen(capabilities) ||
    capabilities.kind !== "zgml.program.capabilities" ||
    typeof capabilities.signature !== "string" ||
    !capabilities.signature.startsWith("program-capabilities|") ||
    !compiledProgram.matchesCapabilitySignature(capabilities.signature)
  ) {
    throw new Error(`${label} expected compiled Program capability evidence to be frozen and matchable`);
  }
  if (Object.isFrozen(capabilities.diagnostics) !== true) {
    throw new Error(`${label} expected compiled Program capability diagnostics to be frozen`);
  }
  if (
    compiledProgram.canExecute() !== capabilities.canExecute ||
    compiledProgram.can_execute() !== capabilities.canExecute ||
    compiledProgram.canBindExternalResources() !== capabilities.canBindExternalResources ||
    compiledProgram.can_bind_external_resources() !== capabilities.canBindExternalResources ||
    compiledProgram.hasFullDispatchPlan() !== capabilities.hasFullDispatchPlan ||
    compiledProgram.has_full_dispatch_plan() !== capabilities.hasFullDispatchPlan ||
    compiledProgram.executionMode() !== capabilities.mode ||
    compiledProgram.execution_mode() !== capabilities.mode ||
    compiledProgram.matchesCapabilitySignature(`${capabilities.signature}|wrong`) !== false
    || !compiledProgram.matches_capability_signature(capabilities.signature)
  ) {
    throw new Error(`${label} expected compiled Program capability helpers to agree with capability evidence`);
  }
}

function expectProgramExecutionPlanEvidence(compiledProgram: Record<string, any>, module: Record<string, any>, label: string) {
  const capabilities = compiledProgram.capabilities();
  const plan = compiledProgram.executionPlan();
  const modulePlan = compiledProgram.executionPlan(module);
  const planAlias = compiledProgram.execution_plan();
  const modulePlanAlias = compiledProgram.execution_plan(module);
  const requiredPlan = compiledProgram.requireExecutionPlan();
  const requiredModulePlan = compiledProgram.requireExecutionPlan(module);
  const requiredPlanAlias = compiledProgram.require_execution_plan();
  const requiredModulePlanAlias = compiledProgram.require_execution_plan(module);
  if (
    !Object.isFrozen(plan) ||
    !Object.isFrozen(modulePlan) ||
    !Object.isFrozen(requiredPlan) ||
    !Object.isFrozen(requiredModulePlan) ||
    plan.kind !== "zgml.program.execution-plan" ||
    plan.programKind !== "generic" ||
    typeof plan.signature !== "string" ||
    !plan.signature.startsWith("program-execution-plan|") ||
    modulePlan.signature === plan.signature ||
    !modulePlan.signature.includes("module=program-module-compatibility|compatible=1") ||
    plan.capabilitySignature !== capabilities.signature ||
    plan.canExecute !== capabilities.canExecute ||
    plan.executionMode !== capabilities.mode ||
    plan.bufferSizing.signature !== compiledProgram.bufferSizing().signature ||
    plan.inputShape.join("x") !== compiledProgram.inputShape().join("x") ||
    plan.outputShape.join("x") !== compiledProgram.outputShape().join("x") ||
    plan.parameterNames.join("|") !== compiledProgram.parameterNames().join("|") ||
    plan.kernelPlan.opCount !== compiledProgram.kernelPlan().opCount ||
    !Object.isFrozen(plan.diagnostics) ||
    planAlias.signature !== plan.signature ||
    requiredPlan.signature !== plan.signature ||
    requiredPlanAlias.signature !== plan.signature ||
    modulePlanAlias.signature !== modulePlan.signature ||
    requiredModulePlan.signature !== modulePlan.signature ||
    requiredModulePlanAlias.signature !== modulePlan.signature ||
    modulePlan.moduleCompatibility.compatible !== true ||
    modulePlan.acceptsModule !== true
  ) {
    throw new Error(`${label} expected Program.executionPlan to summarize compile, capability, layout, and module compatibility evidence`);
  }
}

function expectProgramModuleCompatibilityEvidence(compiledProgram: Record<string, any>, module: Record<string, any>, adapter: Record<string, any>, label: string) {
  const compatibility = compiledProgram.moduleCompatibility(module);
  const bindings = module.bindParameters({ inputShape: compiledProgram.inputShape(), backend: "cpu" });
  const bindingsAlias = module.bind_parameters({ inputShape: compiledProgram.inputShape(), backend: "cpu" });
  const bindingPlan = adapter.nn.bindingPlan(bindings);
  const bindingPlanAlias = adapter.nn.binding_plan(bindingsAlias);
  const requiredBindingPlan = adapter.nn.requireBindingPlan(bindings);
  const requiredBindingPlanAlias = adapter.nn.require_binding_plan(bindingsAlias);
  const rawBindingPlan = adapter.nn.bindingPlan({ weights: new Float32Array(compiledProgram.weightsLen()), bias: new Float32Array(compiledProgram.biasLen()) });
  const placedBindings = module.placeParameters(compiledProgram);
  const placedBindingsAlias = module.place_parameters(compiledProgram);
  const placedBindingPlan = adapter.nn.bindingPlan(placedBindings);
  const placedBindingPlanAlias = adapter.nn.binding_plan(placedBindingsAlias);
  const programBindingPlan = compiledProgram.bindingPlan(bindings);
  const programBindingPlanAlias = compiledProgram.binding_plan(bindingsAlias);
  const requiredProgramBindingPlan = compiledProgram.requireBindingPlan(bindings);
  const requiredProgramBindingPlanAlias = compiledProgram.require_binding_plan(bindingsAlias);
  const rawProgramBindingPlan = compiledProgram.bindingPlan({
    weights: new Float32Array(compiledProgram.weightsLen()),
    bias: new Float32Array(compiledProgram.biasLen()),
  });
  const placedProgramBindingPlan = compiledProgram.bindingPlan(placedBindings);
  const rejectedProgramBindingPlan = compiledProgram.bindingPlan({
    weights: new Float32Array(Math.max(1, compiledProgram.weightsLen() - 1)),
    bias: new Float32Array(compiledProgram.biasLen()),
  });
  if (
    !Object.isFrozen(compatibility) ||
    !Object.isFrozen(compatibility.diagnostics) ||
    compatibility.kind !== "zgml.program.module-compatibility" ||
    typeof compatibility.signature !== "string" ||
    !compatibility.signature.startsWith("program-module-compatibility|compatible=1") ||
    compatibility.compatible !== true ||
    compatibility.reason !== null ||
    compiledProgram.acceptsModule(module) !== true ||
    !Object.isFrozen(bindingPlan) ||
    !Object.isFrozen(bindingPlan.options) ||
    !Object.isFrozen(bindingPlan.parameterNames) ||
    !Object.isFrozen(bindingPlan.parameterInfos) ||
    !Object.isFrozen(bindingPlan.parameterInfos[0]) ||
    !Object.isFrozen(bindingPlan.diagnostics) ||
    bindingPlan.kind !== "zgml.nn.module-bindings-plan" ||
    typeof bindingPlan.signature !== "string" ||
    !bindingPlan.signature.startsWith("module-bindings-plan|moduleBindings=1|supported=1|") ||
    bindingPlanAlias.signature !== bindingPlan.signature ||
    requiredBindingPlan.signature !== bindingPlan.signature ||
    requiredBindingPlanAlias.signature !== bindingPlan.signature ||
    bindingPlan.moduleBindings !== true ||
    bindingPlan.supported !== true ||
    bindingPlan.inputShape.join("x") !== compiledProgram.inputShape().join("x") ||
    bindingPlan.outputShape.join("x") !== compiledProgram.outputShape().join("x") ||
    bindingPlan.parameterNames.join("|") !== module.parameterNames().join("|") ||
    bindingPlan.parameterInfos.map((info: Record<string, any>) => `${info.index}:${info.name}:${info.scalarCount}:${info.shape.join("x")}`).join("|") !== module.parameterInfos().map((info: Record<string, any>) => `${info.index}:${info.name}:${info.scalarCount}:${info.shape.join("x")}`).join("|") ||
    !bindingPlan.support ||
    rawBindingPlan.kind !== "zgml.nn.module-bindings-plan" ||
    typeof rawBindingPlan.signature !== "string" ||
    !rawBindingPlan.signature.startsWith("module-bindings-plan|moduleBindings=0|supported=null|") ||
    rawBindingPlan.moduleBindings !== false ||
    rawBindingPlan.support !== null ||
    rawBindingPlan.supported !== null ||
    rawBindingPlan.parameterNames.length !== 0 ||
    placedBindingPlan.moduleBindings !== true ||
    placedBindingPlanAlias.signature !== placedBindingPlan.signature ||
    placedBindingPlan.supported !== true ||
    !placedBindingPlan.signature.startsWith("module-bindings-plan|moduleBindings=1|supported=1|") ||
    placedBindingPlan.parameterNames.join("|") !== bindingPlan.parameterNames.join("|") ||
    !Object.isFrozen(programBindingPlan) ||
    !Object.isFrozen(programBindingPlan.diagnostics) ||
    programBindingPlan.kind !== "zgml.program.binding-plan" ||
    typeof programBindingPlan.signature !== "string" ||
    !programBindingPlan.signature.startsWith("program-binding-plan|accepted=1|mode=host|") ||
    programBindingPlanAlias.signature !== programBindingPlan.signature ||
    requiredProgramBindingPlan.signature !== programBindingPlan.signature ||
    requiredProgramBindingPlanAlias.signature !== programBindingPlan.signature ||
    programBindingPlan.accepted !== true ||
    programBindingPlan.canBind !== true ||
    programBindingPlan.mode !== "host" ||
    programBindingPlan.reason !== null ||
    programBindingPlan.weightsLen !== compiledProgram.weightsLen() ||
    programBindingPlan.biasLen !== compiledProgram.biasLen() ||
    programBindingPlan.inputShape.join("x") !== compiledProgram.inputShape().join("x") ||
    programBindingPlan.outputShape.join("x") !== compiledProgram.outputShape().join("x") ||
    rawProgramBindingPlan.accepted !== true ||
    rawProgramBindingPlan.mode !== "host" ||
    placedProgramBindingPlan.accepted !== true ||
    placedProgramBindingPlan.mode !== "native" ||
    !placedProgramBindingPlan.signature.startsWith("program-binding-plan|accepted=1|mode=native|") ||
    placedProgramBindingPlan.usesNativeBuffers !== true ||
    rejectedProgramBindingPlan.kind !== "zgml.program.binding-plan" ||
    typeof rejectedProgramBindingPlan.signature !== "string" ||
    !rejectedProgramBindingPlan.signature.startsWith("program-binding-plan|accepted=0|mode=invalid|") ||
    rejectedProgramBindingPlan.signature === programBindingPlan.signature ||
    rejectedProgramBindingPlan.accepted !== false ||
    rejectedProgramBindingPlan.canBind !== false ||
    rejectedProgramBindingPlan.mode !== "invalid" ||
    !Object.isFrozen(rejectedProgramBindingPlan.diagnostics) ||
    !Object.isFrozen(rejectedProgramBindingPlan.diagnostics[0]) ||
    rejectedProgramBindingPlan.diagnostics[0].code !== "binding-invalid"
  ) {
    throw new Error(`${label} expected compatible module, ModuleBindings plan, and Program.bindingPlan evidence to be frozen and accepted`);
  }
  expectThrowIncludes(
    () => adapter.nn.requireBindingPlan({ weights: new Float32Array(compiledProgram.weightsLen()), bias: new Float32Array(compiledProgram.biasLen()) }),
    "nn.requireBindingPlan rejected bindings:",
    `${label} nn.requireBindingPlan rejects raw ProgramBindings`,
  );
  expectThrowIncludes(
    () => compiledProgram.requireBindingPlan({
      weights: new Float32Array(Math.max(1, compiledProgram.weightsLen() - 1)),
      bias: new Float32Array(compiledProgram.biasLen()),
    }),
    "Program.requireBindingPlan rejected bindings:",
    `${label} Program.requireBindingPlan rejects invalid bindings`,
  );
  try {
    if (placedBindings.weights && typeof placedBindings.weights.free === "function") placedBindings.weights.free();
    if (placedBindings.bias && typeof placedBindings.bias.free === "function") placedBindings.bias.free();
    if (placedBindingsAlias.weights && typeof placedBindingsAlias.weights.free === "function") placedBindingsAlias.weights.free();
    if (placedBindingsAlias.bias && typeof placedBindingsAlias.bias.free === "function") placedBindingsAlias.bias.free();
  } catch {
    // Smoke cleanup should not mask the binding-plan evidence failure above.
  }
  const incompatibleModule = adapter.nn.linear(2, 2);
  const mismatch = compiledProgram.moduleCompatibility(incompatibleModule);
  if (
    !Object.isFrozen(mismatch) ||
    !Object.isFrozen(mismatch.diagnostics) ||
    !Object.isFrozen(mismatch.diagnostics[0]) ||
    mismatch.kind !== "zgml.program.module-compatibility" ||
    typeof mismatch.signature !== "string" ||
    !mismatch.signature.startsWith("program-module-compatibility|compatible=0") ||
    mismatch.signature === compatibility.signature ||
    mismatch.compatible !== false ||
    !mismatch.diagnostics.some((diagnostic: Record<string, unknown>) => diagnostic.code === "outputLen-mismatch") ||
    compiledProgram.acceptsModule(incompatibleModule) !== false
  ) {
    throw new Error(`${label} expected incompatible module evidence to be frozen and rejected`);
  }
  expectThrowIncludes(
    () => compiledProgram.bindModule(incompatibleModule),
    String(mismatch.reason),
    `${label} incompatible bindModule`,
  );
  expectThrowIncludes(
    () => compiledProgram.bind(incompatibleModule),
    String(mismatch.reason),
    `${label} incompatible bind(module)`,
  );
}

function expectProgramProfileEvidence(compiledProgram: Record<string, any>, label: string) {
  const profile = compiledProgram.runtimeProfile();
  if (
    !Object.isFrozen(profile) ||
    profile.kind !== "zgml.runtime.profile" ||
    typeof profile.signature !== "string" ||
    !profile.signature.startsWith("runtime-profile|") ||
    compiledProgram.runtime_profile().signature !== profile.signature ||
    !compiledProgram.matchesRuntimeProfileSignature(profile.signature) ||
    !compiledProgram.matches_runtime_profile_signature(profile.signature)
  ) {
    throw new Error(`${label} expected compiled Program runtimeProfile evidence to be frozen and matchable`);
  }
  const expectation = inspection.runtimeProfileExpectation(profile);
  if (
    !Object.isFrozen(expectation) ||
    expectation.kind !== "zgml.runtime.profile-expectation" ||
    typeof expectation.signature !== "string" ||
    !expectation.signature.startsWith("runtime-profile-expectation|") ||
    expectation.noFallback !== true ||
    expectation.noSync !== true ||
    expectation.noRuntimePatchInvalid !== true ||
    inspection.runtimeProfileHasNoFallback(profile) !== true ||
    inspection.runtimeProfileHasNoSync(profile) !== true ||
    inspection.runtimeProfileHasNoInvalidRuntimePatches(profile) !== true ||
    inspection.requireNoFallbackRuntimeProfile(profile).kind !== "zgml.runtime.profile-expectation" ||
    inspection.requireNoSyncRuntimeProfile(profile).kind !== "zgml.runtime.profile-expectation" ||
    inspection.requireRuntimePatchValidProfile(profile).kind !== "zgml.runtime.profile-expectation" ||
    inspection.requireHotRuntimeProfile(profile).kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.runtimeProfileExpectation().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.runtime_profile_expectation().signature !== compiledProgram.runtimeProfileExpectation().signature ||
    compiledProgram.runtimeProfileHasNoFallback() !== true ||
    compiledProgram.runtime_profile_has_no_fallback() !== true ||
    compiledProgram.runtimeProfileHasNoSync() !== true ||
    compiledProgram.runtime_profile_has_no_sync() !== true ||
    compiledProgram.runtimeProfileHasNoInvalidRuntimePatches() !== true ||
    compiledProgram.runtime_profile_has_no_invalid_runtime_patches() !== true ||
    compiledProgram.requireNoFallbackRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.require_no_fallback_runtime_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.requireNoSyncRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.require_no_sync_runtime_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.requireRuntimePatchValidProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.require_runtime_patch_valid_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.requireHotRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledProgram.require_hot_runtime_profile().kind !== "zgml.runtime.profile-expectation"
  ) {
    throw new Error(`${label} expected Program runtime profile no-fallback expectation evidence`);
  }
  expectThrowIncludes(
    () => inspection.requireNoFallbackRuntimeProfile({ ...profile, fallbackOpCount: 1 }),
    "expected no fallback ops",
    `${label} runtime profile fallback expectation`,
  );
  expectThrowIncludes(
    () => inspection.requireNoSyncRuntimeProfile({ ...profile, syncCount: 1 }),
    "expected no syncs",
    `${label} runtime profile no-sync expectation`,
  );
  expectThrowIncludes(
    () => inspection.requireRuntimePatchValidProfile({ ...profile, runtimePatchInvalidCount: 1 }),
    "expected no invalid runtime patches",
    `${label} runtime profile patch-valid expectation`,
  );
  expectThrowIncludes(
    () => inspection.requireHotRuntimeProfile({ ...profile, fallbackOpCount: 1, syncCount: 1, runtimePatchInvalidCount: 1 }),
    "expected hot path evidence",
    `${label} runtime profile hot-path expectation`,
  );
  compiledProgram.resetRuntimeProfile();
  compiledProgram.reset_runtime_profile();
  const resetProfile = compiledProgram.runtimeProfile();
  if (!compiledProgram.matchesRuntimeProfileSignature(resetProfile.signature) || !compiledProgram.matches_runtime_profile_signature(resetProfile.signature)) {
    throw new Error(`${label} expected compiled Program runtimeProfile signature to match after reset`);
  }
}

function expectSessionProfileEvidence(compiledSession: Record<string, any>, label: string) {
  const callProfile = compiledSession.sessionCallProfile();
  if (
    !Object.isFrozen(callProfile) ||
    callProfile.kind !== "zgml.session.call-profile" ||
    typeof callProfile.signature !== "string" ||
    !callProfile.signature.startsWith("session-call-profile|") ||
    compiledSession.session_call_profile().signature !== callProfile.signature ||
    !compiledSession.matchesSessionCallProfileSignature(callProfile.signature) ||
    !compiledSession.matches_session_call_profile_signature(callProfile.signature)
  ) {
    throw new Error(`${label} expected compiled Session call profile evidence to be frozen and matchable`);
  }
  compiledSession.resetSessionCallProfile();
  compiledSession.reset_session_call_profile();
  const resetCallProfile = compiledSession.sessionCallProfile();
  if (
    resetCallProfile.stepCount !== 0 ||
    resetCallProfile.executeCount !== 0 ||
    compiledSession.session_call_profile().signature !== resetCallProfile.signature ||
    !compiledSession.matchesSessionCallProfileSignature(resetCallProfile.signature) ||
    !compiledSession.matches_session_call_profile_signature(resetCallProfile.signature)
  ) {
    throw new Error(`${label} expected compiled Session call profile reset evidence`);
  }
  const runtimeProfile = compiledSession.runtimeProfile();
  if (
    !Object.isFrozen(runtimeProfile) ||
    runtimeProfile.kind !== "zgml.runtime.profile" ||
    typeof runtimeProfile.signature !== "string" ||
    !runtimeProfile.signature.startsWith("runtime-profile|") ||
    compiledSession.runtime_profile().signature !== runtimeProfile.signature ||
    !compiledSession.matchesRuntimeProfileSignature(runtimeProfile.signature) ||
    !compiledSession.matches_runtime_profile_signature(runtimeProfile.signature)
  ) {
    throw new Error(`${label} expected compiled Session runtimeProfile evidence to be frozen and matchable`);
  }
  if (
    compiledSession.runtimeProfileExpectation().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.runtime_profile_expectation().signature !== compiledSession.runtimeProfileExpectation().signature ||
    compiledSession.runtimeProfileHasNoFallback() !== true ||
    compiledSession.runtime_profile_has_no_fallback() !== true ||
    compiledSession.runtimeProfileHasNoSync() !== true ||
    compiledSession.runtime_profile_has_no_sync() !== true ||
    compiledSession.runtimeProfileHasNoInvalidRuntimePatches() !== true ||
    compiledSession.runtime_profile_has_no_invalid_runtime_patches() !== true ||
    compiledSession.requireNoFallbackRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.require_no_fallback_runtime_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.requireNoSyncRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.require_no_sync_runtime_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.requireRuntimePatchValidProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.require_runtime_patch_valid_profile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.requireHotRuntimeProfile().kind !== "zgml.runtime.profile-expectation" ||
    compiledSession.require_hot_runtime_profile().kind !== "zgml.runtime.profile-expectation"
  ) {
    throw new Error(`${label} expected Session runtime profile no-fallback expectation evidence`);
  }
  compiledSession.resetRuntimeProfile();
  compiledSession.reset_runtime_profile();
  const resetRuntimeProfile = compiledSession.runtimeProfile();
  if (!compiledSession.matchesRuntimeProfileSignature(resetRuntimeProfile.signature) || !compiledSession.matches_runtime_profile_signature(resetRuntimeProfile.signature)) {
    throw new Error(`${label} expected compiled Session runtimeProfile signature to match after reset`);
  }
}

function expectInspectionEvidence(compiledProgram: Record<string, any>, compiledSession: Record<string, any>, label: string) {
  const programInspection = compiledProgram.inspect();
  const sessionInspection = compiledSession.inspect();
  const requirements = compiledProgram.requirements();
  const programBufferLayout = compiledProgram.bufferLayout();
  const sessionBufferLayout = compiledSession.bufferLayout();
  if (
    !Object.isFrozen(programInspection) ||
    programInspection.kind !== "zgml.program.inspection" ||
    typeof programInspection.signature !== "string" ||
    !programInspection.signature.startsWith("program-inspection|backend=cpu|") ||
    !Object.isFrozen(programInspection.diagnostics) ||
    programInspection.backend !== "cpu" ||
    programInspection.bufferCount < 1
  ) {
    throw new Error(`${label} expected discriminated frozen Program.inspect evidence`);
  }
  if (
    !Object.isFrozen(sessionInspection) ||
    sessionInspection.kind !== "zgml.session.inspection" ||
    typeof sessionInspection.signature !== "string" ||
    !sessionInspection.signature.startsWith("session-inspection|") ||
    sessionInspection.backend !== "cpu" ||
    sessionInspection.stepOutputCount < 1
  ) {
    throw new Error(`${label} expected discriminated frozen Session.inspect evidence`);
  }
  if (
    !Object.isFrozen(requirements) ||
    requirements.kind !== "zgml.program.requirements" ||
    typeof requirements.signature !== "string" ||
    !requirements.signature.startsWith(`program-requirements|modelKind=${requirements.modelKind}|`) ||
    requirements.modelKind !== sessionInspection.modelKind ||
    requirements.inputLen !== compiledProgram.inputLen() ||
    requirements.outputLen !== compiledProgram.outputLen()
  ) {
    throw new Error(`${label} expected discriminated frozen Program.requirements evidence`);
  }
  if (
    !Object.isFrozen(programBufferLayout) ||
    !Object.isFrozen(sessionBufferLayout) ||
    programBufferLayout.kind !== "zgml.program.buffer-layout" ||
    sessionBufferLayout.kind !== "zgml.program.buffer-layout" ||
    typeof programBufferLayout.signature !== "string" ||
    !programBufferLayout.signature.startsWith("zgml.program.buffer-layout|") ||
    sessionBufferLayout.signature !== programBufferLayout.signature ||
    programBufferLayout.output.byteLength !== sessionBufferLayout.output.byteLength
  ) {
    throw new Error(`${label} expected discriminated frozen Program/Session buffer layout evidence`);
  }
}

function expectSessionExecutionEvidence(
  compiledProgram: Record<string, any>,
  compiledSession: Record<string, any>,
  adapter: Record<string, any>,
  label: string,
) {
  compiledSession.resetSessionCallProfile();
  const input = adapter.tensor([1, 2], [2]);
  const stepTensor = compiledSession.stepTensor(input);
  if (!(stepTensor instanceof adapter.Tensor)) throw new Error(`${label} expected stepTensor to return a Tensor`);
  expectClose(stepTensor.data, [-0.5], `${label} stepTensor output`);
  const stepTensorCarrier = new Float32Array(1);
  const carrierStepTensor = compiledSession.stepTensor(input, { output: stepTensorCarrier });
  if (carrierStepTensor.data.buffer !== stepTensorCarrier.buffer) throw new Error(`${label} expected stepTensor to reuse caller output carrier`);
  expectClose(stepTensorCarrier, [-0.5], `${label} stepTensor carrier output`);
  const stepIntoCarrier = new Float32Array(1);
  if (compiledSession.stepInto(stepIntoCarrier, input) !== stepIntoCarrier) throw new Error(`${label} expected stepInto to return caller output carrier`);
  expectClose(stepIntoCarrier, [-0.5], `${label} stepInto output`);
  if (compiledSession.execute({ input, output: false }) !== undefined) throw new Error(`${label} expected execute no-output to return undefined`);
  const executeCarrier = new Float32Array(1);
  if (compiledSession.execute({ input, output: executeCarrier }) !== executeCarrier) throw new Error(`${label} expected execute to return caller output carrier`);
  expectClose(executeCarrier, [-0.5], `${label} execute output`);
  const executeIntoCarrier = new Float32Array(1);
  if (compiledSession.executeInto(executeIntoCarrier, { input }) !== executeIntoCarrier) throw new Error(`${label} expected executeInto to return caller output carrier`);
  expectClose(executeIntoCarrier, [-0.5], `${label} executeInto output`);
  const executeTensorCarrier = new Float32Array(1);
  const executeTensor = compiledSession.executeTensor({ input, output: executeTensorCarrier });
  if (!(executeTensor instanceof adapter.Tensor) || executeTensor.data.buffer !== executeTensorCarrier.buffer) {
    throw new Error(`${label} expected executeTensor to return a Tensor view over caller output carrier`);
  }
  expectClose(executeTensor.data, [-0.5], `${label} executeTensor output`);
  expectThrowIncludes(
    () => compiledSession.readOutputTensor(),
    "Session.readOutputInto requires a bound output buffer",
    `${label} unbound readOutputTensor`,
  );
  const callProfile = compiledSession.sessionCallProfile();
  if (
    callProfile.stepTensorCount !== 2 ||
    callProfile.stepIntoCount !== 1 ||
    callProfile.executeCount !== 2 ||
    callProfile.executeIntoCount !== 1 ||
    callProfile.executeTensorCount !== 1 ||
    callProfile.readOutputTensorCount !== 0 ||
    !compiledSession.matchesSessionCallProfileSignature(callProfile.signature)
  ) {
    throw new Error(`${label} expected execution call profile counters to track public Session helpers`);
  }

  const boundOutput = new Float32Array(1);
  const readbackSession = compiledProgram.bind({
    weights: new Float32Array([0.5, -0.5]),
    bias: new Float32Array([0]),
    output: boundOutput,
  });
  readbackSession.resetSessionCallProfile();
  readbackSession.step(input);
  expectClose(boundOutput, [-0.5], `${label} bound output after step`);
  const readIntoCarrier = new Float32Array(1);
  if (readbackSession.readOutputInto(readIntoCarrier) !== readIntoCarrier) throw new Error(`${label} expected readOutputInto to return caller output carrier`);
  expectClose(readIntoCarrier, [-0.5], `${label} readOutputInto output`);
  const readTensor = readbackSession.readOutputTensor();
  if (!(readTensor instanceof adapter.Tensor)) throw new Error(`${label} expected readOutputTensor to return a Tensor`);
  expectClose(readTensor.data, [-0.5], `${label} readOutputTensor output`);
  const readTensorCarrier = new Float32Array(2);
  readTensorCarrier[1] = 99;
  const carrierReadTensor = readbackSession.readOutputTensor({ output: readTensorCarrier, shape: [1] });
  if (carrierReadTensor.data.buffer !== readTensorCarrier.buffer) {
    throw new Error(`${label} expected readOutputTensor to reuse caller output carrier`);
  }
  expectClose(carrierReadTensor.data, [-0.5], `${label} readOutputTensor carrier output`);
  if (readTensorCarrier[1] !== 99) throw new Error(`${label} expected readOutputTensor to leave inactive carrier capacity untouched`);
  const readbackProfile = readbackSession.sessionCallProfile();
  if (
    readbackProfile.stepCount !== 1 ||
    readbackProfile.readOutputIntoCount !== 1 ||
    readbackProfile.readOutputTensorCount !== 2 ||
    !readbackSession.matchesSessionCallProfileSignature(readbackProfile.signature)
  ) {
    throw new Error(`${label} expected readback call profile counters to track bound-output helpers`);
  }
  readbackSession.dispose();
}

function expectSessionParameterUploadEvidence(compiledProgram: Record<string, any>, adapter: Record<string, any>, label: string) {
  const weights = compiledProgram.createWeightsBuffer();
  const bias = compiledProgram.createBiasBuffer();
  weights.writeFloat32([0.5, -0.5]);
  bias.writeFloat32([0]);
  const uploadSession = compiledProgram.bind({ weights, bias });
  uploadSession.resetSessionCallProfile();
  const names = uploadSession.parameterNames();
  const infos = uploadSession.parameterInfos();
  const weightInfo = uploadSession.parameterInfo("0.weight");
  const biasInfo = uploadSession.parameterInfo(1);
  if (
    !Object.isFrozen(names) ||
    !Object.isFrozen(infos) ||
    names.join("|") !== "0.weight|0.bias" ||
    !weightInfo ||
    weightInfo.binding !== "weights" ||
    weightInfo.scalarCount !== 2 ||
    !biasInfo ||
    biasInfo.name !== "0.bias" ||
    biasInfo.binding !== "bias"
  ) {
    throw new Error(`${label} expected Session parameter inspection to expose frozen named persistent slots`);
  }
  expectThrowIncludes(
    () => uploadSession.uploadParameterByName("weight"),
    "Session parameter weight was not found",
    `${label} uploadParameterByName requires fully qualified names`,
  );
  const input = adapter.tensor([1, 2], [2]);
  expectClose(uploadSession.stepTensor(input).data, [-0.5], `${label} NativeBuffer upload initial output`);
  weights.writeFloat32([2, 0]);
  bias.writeFloat32([1]);
  expectClose(uploadSession.stepTensor(input).data, [-0.5], `${label} NativeBuffer writes require explicit upload`);
  uploadSession.uploadParameterByName("0.weight");
  expectClose(uploadSession.stepTensor(input).data, [2], `${label} uploadParameterByName weight refresh`);
  uploadSession.uploadParameterByName("0.bias");
  expectClose(uploadSession.stepTensor(input).data, [3], `${label} uploadParameterByName bias refresh`);
  weights.writeFloat32([0, 1]);
  bias.writeFloat32([0]);
  uploadSession.uploadParameterRange(0, 2);
  expectClose(uploadSession.stepTensor(input).data, [2], `${label} uploadParameterRange refresh`);
  weights.writeFloat32([1, 1]);
  bias.writeFloat32([1]);
  uploadSession.uploadParameters();
  expectClose(uploadSession.stepTensor(input).data, [4], `${label} uploadParameters refresh`);
  weights.writeFloat32([2, -1]);
  uploadSession.uploadParameter(0);
  expectClose(uploadSession.stepTensor(input).data, [1], `${label} uploadParameter index refresh`);
  const callProfile = uploadSession.sessionCallProfile();
  if (
    callProfile.stepTensorCount !== 7 ||
    callProfile.uploadParametersCount !== 1 ||
    callProfile.uploadParameterCount !== 1 ||
    callProfile.uploadParameterByNameCount !== 2 ||
    callProfile.uploadParameterRangeCount !== 1 ||
    uploadSession.session_call_profile().signature !== callProfile.signature ||
    !uploadSession.matchesSessionCallProfileSignature(callProfile.signature) ||
    !uploadSession.matches_session_call_profile_signature(callProfile.signature)
  ) {
    throw new Error(`${label} expected parameter upload call profile counters to track persistent refresh helpers`);
  }
  uploadSession.dispose();
  weights.dispose();
  bias.dispose();
}

function expectProgramBufferEvidence(compiledProgram: Record<string, any>, label: string) {
  const weights = compiledProgram.createWeightsBuffer();
  const bias = compiledProgram.createBiasBuffer();
  const input = compiledProgram.createInputBuffer();
  const output = compiledProgram.createOutputBuffer();
  const genericInput = compiledProgram.createBuffer("input");
  try {
    const buffers = [
      ["weights", weights, 8],
      ["bias", bias, 4],
      ["input", input, 8],
      ["output", output, 4],
      ["generic input", genericInput, 8],
    ] as const;
    for (const [name, buffer, byteLength] of buffers) {
      if (buffer.byteLength !== byteLength || buffer.size() !== byteLength) {
        throw new Error(`${label} expected ${name} NativeBuffer byte length ${byteLength}`);
      }
      const inspection = buffer.inspect();
      if (
        !Object.isFrozen(inspection) ||
        inspection.kind !== "zgml.buffer.inspection" ||
        inspection.storage !== "host" ||
        inspection.byteLength !== byteLength ||
        inspection.placement !== null ||
        inspection.access !== null
      ) {
        throw new Error(`${label} expected ${name} NativeBuffer host inspection evidence`);
      }
    }
  } finally {
    weights.dispose();
    bias.dispose();
    input.dispose();
    output.dispose();
    genericInput.dispose();
  }
}

function expectTensorNativeBufferEvidence(compiledProgram: Record<string, any>, adapter: Record<string, any>, label: string) {
  const tensorInput = adapter.tensor([3, 4], [2]);
  const hostPlacement = tensorInput.nativePlacement();
  if (
    !Object.isFrozen(hostPlacement) ||
    hostPlacement.kind !== "zgml.tensor.native-placement" ||
    hostPlacement.storage !== "host" ||
    hostPlacement.bufferKind !== null ||
    hostPlacement.device !== "cpu" ||
    hostPlacement.shape.join("x") !== "2" ||
    hostPlacement.length !== 2 ||
    hostPlacement.byteLength !== 8 ||
    hostPlacement.signature !== "tensor-native-placement|storage=host|buffer=none|device=cpu|shape=2|length=2|bytes=8" ||
    tensorInput.native_placement().signature !== hostPlacement.signature
  ) {
    throw new Error(`${label} expected Tensor.nativePlacement host evidence`);
  }
  const nativeBuffer = tensorInput.toNativeBuffer();
  try {
    if (nativeBuffer.byteLength !== 8 || nativeBuffer.size() !== 8) {
      throw new Error(`${label} expected Tensor.toNativeBuffer byte length`);
    }
    expectClose(nativeBuffer.readFloat32(2), [3, 4], `${label} Tensor.toNativeBuffer readback`);
    const nativeInspection = nativeBuffer.inspect();
    if (!Object.isFrozen(nativeInspection) || nativeInspection.storage !== "host" || nativeInspection.byteLength !== 8) {
      throw new Error(`${label} expected Tensor.toNativeBuffer host NativeBuffer inspection`);
    }
    const restored = adapter.Tensor.fromNativeBuffer(nativeBuffer, [2]);
    if (!(restored instanceof adapter.Tensor) || restored.shape.join("x") !== "2") {
      throw new Error(`${label} expected Tensor.fromNativeBuffer shaped tensor`);
    }
    expectClose(restored.data, [3, 4], `${label} Tensor.fromNativeBuffer readback`);
    const reused = adapter.tensor([0, 0], [2]);
    if (reused.copyFromNativeBuffer_(nativeBuffer) !== reused) {
      throw new Error(`${label} expected Tensor.copyFromNativeBuffer_ to return this`);
    }
    expectClose(reused.data, [3, 4], `${label} Tensor.copyFromNativeBuffer_ readback`);
    reused.zero_();
    if (reused.copy_from_native_buffer_(nativeBuffer) !== reused) {
      throw new Error(`${label} expected Tensor.copy_from_native_buffer_ to return this`);
    }
    expectClose(reused.data, [3, 4], `${label} Tensor.copy_from_native_buffer_ readback`);
    expectThrowIncludes(
      () => reused.copyFromNativeBuffer_(nativeBuffer, { length: 1 }),
      "Tensor.copyFromNativeBuffer_ length 1 does not match tensor length 2",
      `${label} Tensor.copyFromNativeBuffer_ rejects wrong length`,
    );
  } finally {
    nativeBuffer.dispose();
  }

  const placedInput = tensorInput.place(compiledProgram, "input");
  const optionPlacedInput = tensorInput.toNativeBuffer({ program: compiledProgram, kind: "input" });
  const programPlacement = tensorInput.nativePlacement({ program: compiledProgram, kind: "input" });
  try {
    if (
      !Object.isFrozen(programPlacement) ||
      programPlacement.storage !== "program" ||
      programPlacement.bufferKind !== "input" ||
      programPlacement.device !== "program" ||
      programPlacement.shape.join("x") !== "2" ||
      programPlacement.length !== 2 ||
      programPlacement.byteLength !== 8 ||
      programPlacement.signature !== "tensor-native-placement|storage=program|buffer=input|device=program|shape=2|length=2|bytes=8"
    ) {
      throw new Error(`${label} expected Tensor.nativePlacement Program input evidence`);
    }
    expectClose(placedInput.readFloat32(2), [3, 4], `${label} Tensor.place input readback`);
    expectClose(optionPlacedInput.readFloat32(2), [3, 4], `${label} Tensor.toNativeBuffer program input readback`);
  } finally {
    placedInput.dispose();
    optionPlacedInput.dispose();
  }
  expectThrowIncludes(
    () => adapter.tensor([1, 2, 3], [3]).place(compiledProgram, "input"),
    "Tensor.place input length 3 does not match Program input slot length 2",
    `${label} Tensor.place shape mismatch`,
  );
  expectThrowIncludes(
    () => tensorInput.place(compiledProgram, "missing-slot" as any),
    "Tensor.place missing-slot is not a Program buffer slot",
    `${label} Tensor.place missing slot`,
  );
}

function expectNativeBufferBoundSessionEvidence(compiledProgram: Record<string, any>, adapter: Record<string, any>, label: string) {
  const weights = compiledProgram.createWeightsBuffer();
  const bias = compiledProgram.createBiasBuffer();
  const input = compiledProgram.createInputBuffer();
  const output = compiledProgram.createOutputBuffer();
  try {
    weights.writeFloat32([0.5, -0.5]);
    bias.writeFloat32([0]);
    input.writeFloat32([1, 2]);
    const session = compiledProgram.bind({ weights, bias, input, output });
    session.resetSessionCallProfile();
    const result = session.step();
    if (!(result instanceof Float32Array)) throw new Error(`${label} expected NativeBuffer-bound step to return a Float32Array readback`);
    expectClose(result, [-0.5], `${label} NativeBuffer-bound step output`);
    expectClose(output.readFloat32(1), [-0.5], `${label} NativeBuffer output readFloat32`);
    const readIntoCarrier = new Float32Array(2);
    readIntoCarrier[1] = 99;
    if (output.readFloat32Into(readIntoCarrier, 1) !== readIntoCarrier) throw new Error(`${label} expected NativeBuffer.readFloat32Into to return caller carrier`);
    expectClose(readIntoCarrier.subarray(0, 1), [-0.5], `${label} NativeBuffer output readFloat32Into`);
    if (readIntoCarrier[1] !== 99) throw new Error(`${label} expected NativeBuffer.readFloat32Into to leave inactive carrier capacity untouched`);
    const readTensor = session.readOutputTensor();
    if (!(readTensor instanceof adapter.Tensor)) throw new Error(`${label} expected NativeBuffer-bound readOutputTensor to return a Tensor`);
    expectClose(readTensor.data, [-0.5], `${label} NativeBuffer-bound readOutputTensor`);
    const callProfile = session.sessionCallProfile();
    if (
      callProfile.stepCount !== 1 ||
      callProfile.readOutputTensorCount !== 1 ||
      !session.matchesSessionCallProfileSignature(callProfile.signature)
    ) {
      throw new Error(`${label} expected NativeBuffer-bound call profile counters`);
    }
    session.dispose();
  } finally {
    weights.dispose();
    bias.dispose();
    input.dispose();
    output.dispose();
  }
}

function expectSequentialProgramEvidence(adapter: Record<string, any>, label: string) {
  const sequential = adapter.nn.sequential([
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  ]);
  const variadicSequential = adapter.nn.sequential(
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  );
  if (variadicSequential.length !== 3 || variadicSequential.parameterNames().join("|") !== "0.weight|0.bias|2.weight|2.bias") {
    throw new Error(`${label} expected nn.sequential(layer1, layer2, ...) to behave like nn.sequential([layers])`);
  }
  const variadicClassSequential = new adapter.nn.Sequential(
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  );
  if (variadicClassSequential.length !== 3 || variadicClassSequential.parameterNames().join("|") !== "0.weight|0.bias|2.weight|2.bias") {
    throw new Error(`${label} expected new nn.Sequential(layer1, layer2, ...) to behave like new nn.Sequential([layers])`);
  }
  if (
    variadicClassSequential.__len__() !== 3 ||
    variadicClassSequential.len() !== 3 ||
    variadicClassSequential.size() !== 3 ||
    variadicClassSequential.get(1) !== variadicClassSequential.at(1) ||
    variadicClassSequential.__getitem__(-1) !== variadicClassSequential.at(-1)
  ) {
    throw new Error(`${label} expected nn.Sequential to expose PyTorch-style len/get/item/setitem aliases`);
  }
  expectClose(variadicClassSequential.forward(adapter.tensor([2, -3], [2])).data, [2], `${label} variadic class Sequential forward`);
  const nativeForwardSequential = new adapter.nn.Sequential(
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  );
  const nativeForwardFirstLayer = nativeForwardSequential.at(0);
  const nativeForwardReluLayer = nativeForwardSequential.at(1);
  const nativeForwardLastLayer = nativeForwardSequential.at(2);
  nativeForwardFirstLayer.forward = () => {
    throw new Error("poisoned first JS forward");
  };
  nativeForwardReluLayer.forward = () => {
    throw new Error("poisoned relu JS forward");
  };
  nativeForwardLastLayer.forward = () => {
    throw new Error("poisoned last JS forward");
  };
  expectClose(
    adapter.noGrad(() => nativeForwardSequential.forward(adapter.tensor([2, -3], [2]))).data,
    [2],
    `${label} noGrad nn.Sequential auto native Program forward`,
  );
  const nativeForwardArraySequential = new adapter.nn.Sequential(
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  );
  nativeForwardArraySequential.at(0).forward = () => {
    throw new Error("poisoned array first JS forward");
  };
  nativeForwardArraySequential.at(1).forward = () => {
    throw new Error("poisoned array relu JS forward");
  };
  nativeForwardArraySequential.at(2).forward = () => {
    throw new Error("poisoned array last JS forward");
  };
  expectClose(
    adapter.noGrad(() => nativeForwardArraySequential.forward([2, -3])).data,
    [2],
    `${label} noGrad nn.Sequential auto native Program array forward`,
  );
  const nativeForwardBatchArraySequential = new adapter.nn.Sequential(
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.relu(),
    adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] }),
  );
  nativeForwardBatchArraySequential.at(0).forward = () => {
    throw new Error("poisoned batch array first JS forward");
  };
  nativeForwardBatchArraySequential.at(1).forward = () => {
    throw new Error("poisoned batch array relu JS forward");
  };
  nativeForwardBatchArraySequential.at(2).forward = () => {
    throw new Error("poisoned batch array last JS forward");
  };
  const nativeBatchArray = adapter.noGrad(() => nativeForwardBatchArraySequential.forward([[2, -3], [4, 1]]));
  expectClose(
    nativeBatchArray.data,
    [2, 3],
    `${label} noGrad nn.Sequential auto native Program batch array forward`,
  );
  if (nativeBatchArray.shape.join("x") !== "2x1") {
    throw new Error(`${label} expected noGrad nn.Sequential auto native Program batch array shape [2,1], got [${nativeBatchArray.shape.join(",")}]`);
  }
  nativeForwardLastLayer.weight[0] = 2;
  expectClose(
    adapter.noGrad(() => nativeForwardSequential.forward(adapter.tensor([2, -3], [2]))).data,
    [4],
    `${label} noGrad nn.Sequential auto native Program refreshes packed parameters`,
  );
  if (
    variadicClassSequential.__setitem__(1, adapter.nn.tanh()) !== variadicClassSequential ||
    variadicClassSequential.__getitem__(1).kind !== "tanh"
  ) {
    throw new Error(`${label} expected nn.Sequential __setitem__ to replace layers in place`);
  }
  const sequentialDelete = new adapter.nn.Sequential(adapter.nn.relu(), adapter.nn.tanh());
  if (sequentialDelete.__delitem__(-1) !== sequentialDelete || sequentialDelete.length !== 1 || sequentialDelete.__getitem__(-1)?.kind !== "relu") {
    throw new Error(`${label} expected nn.Sequential __delitem__ to delete layers in place with Python-style indexing`);
  }
  const sequentialPop = new adapter.nn.Sequential(adapter.nn.relu(), adapter.nn.tanh());
  if (sequentialPop.pop()?.kind !== "tanh" || sequentialPop.pop(0)?.kind !== "relu" || sequentialPop.length !== 0) {
    throw new Error(`${label} expected nn.Sequential pop to remove and return indexed layers`);
  }
  const sequentialClear = new adapter.nn.Sequential({ stem: adapter.nn.relu(), tail: adapter.nn.tanh() });
  if (sequentialClear.clear() !== sequentialClear || sequentialClear.length !== 0 || sequentialClear.namedChildren().length !== 0) {
    throw new Error(`${label} expected nn.Sequential clear to remove named layers in place`);
  }
  const sequentialMutate = new adapter.nn.Sequential(adapter.nn.relu());
  if (
    sequentialMutate.append(adapter.nn.tanh()) !== sequentialMutate ||
    sequentialMutate.insert(1, adapter.nn.relu()) !== sequentialMutate ||
    sequentialMutate.extend([adapter.nn.tanh()]) !== sequentialMutate ||
    sequentialMutate.extend(new adapter.nn.Sequential(adapter.nn.relu())) !== sequentialMutate ||
    sequentialMutate.length !== 5 ||
    sequentialMutate.__getitem__(1)?.kind !== "relu" ||
    sequentialMutate.__getitem__(-2)?.kind !== "tanh" ||
    sequentialMutate.__getitem__(-1)?.kind !== "relu"
  ) {
    throw new Error(`${label} expected nn.Sequential append/insert/extend to mutate layers in place and accept Sequential iterables`);
  }
  const moduleList = new adapter.nn.ModuleList([adapter.nn.linear(1, 1, { weights: [1], bias: [0] })]);
  const moduleListFromIterable = new adapter.nn.ModuleList(new adapter.nn.ModuleList([adapter.nn.relu()]));
  const moduleListFactoryFromIterable = adapter.nn.module_list(new adapter.nn.ModuleList([adapter.nn.tanh()]));
  if (
    moduleListFromIterable.length !== 1 ||
    moduleListFromIterable.__getitem__(0).kind !== "relu" ||
    moduleListFactoryFromIterable.length !== 1 ||
    moduleListFactoryFromIterable.__getitem__(0).kind !== "tanh" ||
    moduleList.append(adapter.nn.relu()) !== moduleList ||
    moduleList.insert(1, adapter.nn.tanh()) !== moduleList ||
    moduleList.extend([adapter.nn.linear(1, 1, { weights: [2], bias: [0] })]) !== moduleList ||
    moduleList.extend(new adapter.nn.ModuleList([adapter.nn.relu()])) !== moduleList ||
    moduleList.length !== 5 ||
    moduleList.__len__() !== 5 ||
    moduleList.len() !== 5 ||
    moduleList.size() !== 5 ||
    moduleList.get(0) !== moduleList.at(0) ||
    moduleList.__getitem__(-1) !== moduleList.at(-1) ||
    moduleList.__getitem__(1).kind !== "tanh" ||
    moduleList.__setitem__(-2, adapter.nn.linear(1, 1, { weights: [5], bias: [0] })) !== moduleList ||
    moduleList.__getitem__(-2).parameters()[0].data[0] !== 5 ||
    moduleList.__getitem__(-1).kind !== "relu" ||
    moduleList.parameterNames("layers").join("|") !== "layers.0.weight|layers.0.bias|layers.3.weight|layers.3.bias"
  ) {
    throw new Error(`${label} expected nn.ModuleList append/insert/extend and len/get/item aliases to preserve PyTorch-style module registration and accept iterable containers`);
  }
  const moduleListDelete = new adapter.nn.ModuleList([adapter.nn.relu(), adapter.nn.tanh()]);
  if (moduleListDelete.__delitem__(-1) !== moduleListDelete || moduleListDelete.length !== 1 || moduleListDelete.__getitem__(-1)?.kind !== "relu") {
    throw new Error(`${label} expected nn.ModuleList __delitem__ to delete layers in place with Python-style indexing`);
  }
  const moduleListPop = new adapter.nn.ModuleList([adapter.nn.relu(), adapter.nn.tanh()]);
  if (moduleListPop.pop()?.kind !== "tanh" || moduleListPop.pop(0)?.kind !== "relu" || moduleListPop.length !== 0) {
    throw new Error(`${label} expected nn.ModuleList pop to remove and return indexed layers`);
  }
  const moduleListClear = new adapter.nn.ModuleList([adapter.nn.relu(), adapter.nn.tanh()]);
  if (moduleListClear.clear() !== moduleListClear || moduleListClear.length !== 0) {
    throw new Error(`${label} expected nn.ModuleList clear to remove layers in place`);
  }
  const parameterList = adapter.nn.parameterList([adapter.nn.Parameter("scale", [1], [1])]);
  const parameterListFromIterable = new adapter.nn.ParameterList(new adapter.nn.ParameterList([adapter.nn.Parameter("ctor", [1], [1])]));
  const parameterListFactoryFromIterable = adapter.nn.parameter_list(new adapter.nn.ParameterList([adapter.nn.Parameter("factory", [1], [1])]));
  if (
    parameterListFromIterable.length !== 1 ||
    parameterListFromIterable.__getitem__(0).name !== "ctor" ||
    parameterListFactoryFromIterable.length !== 1 ||
    parameterListFactoryFromIterable.__getitem__(0).name !== "factory" ||
    parameterList.append(adapter.nn.Parameter("shift", [0], [1])) !== parameterList ||
    parameterList.insert(1, adapter.nn.Parameter("inserted", [3], [1])) !== parameterList ||
    parameterList.extend([adapter.nn.Parameter("gain", [2], [1])]) !== parameterList ||
    parameterList.extend(new adapter.nn.ParameterList([adapter.nn.Parameter("iterable", [7], [1])])) !== parameterList ||
    parameterList.length !== 5 ||
    parameterList.__len__() !== 5 ||
    parameterList.len() !== 5 ||
    parameterList.size() !== 5 ||
    parameterList.get(0) !== parameterList.at(0) ||
    parameterList.__getitem__(-1) !== parameterList.at(-1) ||
    parameterList.__getitem__(1).name !== "inserted" ||
    parameterList.__setitem__(-2, adapter.nn.Parameter("tail", [5], [1])) !== parameterList ||
    parameterList.__getitem__(-2).name !== "tail" ||
    parameterList.__getitem__(-1).name !== "iterable" ||
    parameterList.parameterNames("params").join("|") !== "params.0|params.1|params.2|params.3|params.4"
  ) {
    throw new Error(`${label} expected nn.ParameterList append/insert/extend and len/get/item aliases to preserve PyTorch-style parameter registration and accept iterable containers`);
  }
  const parameterListDelete = new adapter.nn.ParameterList([adapter.nn.Parameter("keep", [1], [1]), adapter.nn.Parameter("drop", [0], [1])]);
  if (parameterListDelete.__delitem__(-1) !== parameterListDelete || parameterListDelete.length !== 1 || parameterListDelete.__getitem__(-1)?.name !== "keep") {
    throw new Error(`${label} expected nn.ParameterList __delitem__ to delete parameters in place with Python-style indexing`);
  }
  const parameterListPop = new adapter.nn.ParameterList([adapter.nn.Parameter("keep", [1], [1]), adapter.nn.Parameter("drop", [0], [1])]);
  if (parameterListPop.pop()?.name !== "drop" || parameterListPop.pop(0)?.name !== "keep" || parameterListPop.length !== 0) {
    throw new Error(`${label} expected nn.ParameterList pop to remove and return indexed parameters`);
  }
  const parameterListClear = new adapter.nn.ParameterList([adapter.nn.Parameter("keep", [1], [1]), adapter.nn.Parameter("drop", [0], [1])]);
  if (parameterListClear.clear() !== parameterListClear || parameterListClear.length !== 0) {
    throw new Error(`${label} expected nn.ParameterList clear to remove parameters in place`);
  }
  const moduleDict = new adapter.nn.ModuleDict({ head: adapter.nn.linear(1, 1, { weights: [1], bias: [0] }) });
  if (
    moduleDict.update({ act: adapter.nn.relu(), tail: adapter.nn.linear(1, 1, { weights: [2], bias: [0] }) }) !== moduleDict ||
    moduleDict.length !== 3 ||
    moduleDict.__len__() !== 3 ||
    moduleDict.len() !== 3 ||
    moduleDict.size() !== 3 ||
    moduleDict.__getitem__("head") !== moduleDict.get("head") ||
    moduleDict.__setitem__("head", adapter.nn.linear(1, 1, { weights: [4], bias: [0] })) !== moduleDict ||
    moduleDict.get("head")?.parameters()[0]?.data?.[0] !== 4 ||
    moduleDict.items().map((entry: readonly [string, unknown]) => entry[0]).join("|") !== "head|act|tail" ||
    moduleDict.parameterNames("layers").join("|") !== "layers.head.weight|layers.head.bias|layers.tail.weight|layers.tail.bias"
  ) {
    throw new Error(`${label} expected nn.ModuleDict update/items and len/item aliases to preserve PyTorch-style module registration`);
  }
  if (moduleDict.set("drop", adapter.nn.relu()).__delitem__("drop") !== moduleDict || moduleDict.has("drop")) {
    throw new Error(`${label} expected nn.ModuleDict __delitem__ to delete modules in place`);
  }
  const poppedModule = moduleDict.pop("act");
  if (
    poppedModule?.kind !== "relu" ||
    moduleDict.__contains__("act") !== false ||
    moduleDict.__contains__("head") !== true ||
    moduleDict.clear() !== moduleDict ||
    moduleDict.length !== 0 ||
    moduleDict.items().length !== 0
  ) {
    throw new Error(`${label} expected nn.ModuleDict contains/pop/clear aliases to mutate PyTorch-style module registration`);
  }
  const parameterDict = new adapter.nn.ParameterDict({ scale: adapter.nn.Parameter("scale", [1], [1]) });
  if (
    parameterDict.update({ shift: adapter.nn.Parameter("shift", [0], [1]), gain: adapter.nn.Parameter("gain", [2], [1]) }) !== parameterDict ||
    parameterDict.length !== 3 ||
    parameterDict.__len__() !== 3 ||
    parameterDict.len() !== 3 ||
    parameterDict.size() !== 3 ||
    parameterDict.__getitem__("scale") !== parameterDict.get("scale") ||
    parameterDict.__setitem__("scale", adapter.nn.Parameter("scale", [4], [1])) !== parameterDict ||
    parameterDict.get("scale")?.data?.[0] !== 4 ||
    parameterDict.items().map((entry: readonly [string, unknown]) => entry[0]).join("|") !== "scale|shift|gain" ||
    parameterDict.parameterNames("params").join("|") !== "params.scale|params.shift|params.gain"
  ) {
    throw new Error(`${label} expected nn.ParameterDict update/items and len/item aliases to preserve PyTorch-style parameter registration`);
  }
  if (parameterDict.set("drop", adapter.nn.Parameter("drop", [0], [1])).__delitem__("drop") !== parameterDict || parameterDict.has("drop")) {
    throw new Error(`${label} expected nn.ParameterDict __delitem__ to delete parameters in place`);
  }
  const poppedParameter = parameterDict.pop("shift");
  if (
    poppedParameter?.name !== "shift" ||
    parameterDict.__contains__("shift") !== false ||
    parameterDict.__contains__("scale") !== true ||
    parameterDict.clear() !== parameterDict ||
    parameterDict.length !== 0 ||
    parameterDict.items().length !== 0
  ) {
    throw new Error(`${label} expected nn.ParameterDict contains/pop/clear aliases to mutate PyTorch-style parameter registration`);
  }
  class PackagedBufferedHead extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-buffered-head" });
      this.offset = adapter.nn.Buffer("offset", [1], [1]);
      this.registerBuffer("scale", adapter.nn.Buffer("scale", [2], [1]));
    }

    forward(input: Record<string, any>) {
      return input.mul(adapter.tensor(this.scale.data, [1])).add(adapter.tensor(this.offset.data, [1]));
    }
  }
  class PackagedBufferedParent extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-buffered-parent" });
      this.registerBuffer("root", adapter.nn.Buffer("root", [3], [1]));
      this.addModule("child", new PackagedBufferedHead());
    }

    forward(input: Record<string, any>) {
      return this.child.forward(input).add(adapter.tensor(this.root.data, [1]));
    }
  }
  class PackagedModuleListHead extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-module-list-head" });
      this.layers = adapter.nn.moduleList([
        adapter.nn.linear(1, 1, { weights: [2], bias: [0] }),
      ]);
    }

    forward(input: Record<string, any>) {
      return this.layers.at(0).forward(input);
    }
  }
  class PackagedParameterListHead extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-parameter-list-head" });
      this.params = new adapter.nn.ParameterList([
        adapter.nn.Parameter("weight", [0], [1]),
        adapter.nn.Parameter("bias", [0], [1]),
      ]);
    }

    forward(input: Record<string, any>) {
      return input.mul(this.params.at(0).tensor).add(this.params.at(1).tensor);
    }
  }
  class PackagedModuleDictHead extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-module-dict-head" });
      this.layers = adapter.nn.moduleDict({
        head: adapter.nn.linear(1, 1, { weights: [2], bias: [0] }),
      });
    }

    forward(input: Record<string, any>) {
      return this.layers.get("head").forward(input);
    }
  }
  class PackagedParameterDictHead extends adapter.nn.Module {
    constructor() {
      super({ kind: "packaged-parameter-dict-head" });
      this.params = new adapter.nn.ParameterDict({
        weight: adapter.nn.Parameter("weight", [0], [1]),
        bias: adapter.nn.Parameter("bias", [0], [1]),
      });
    }

    forward(input: Record<string, any>) {
      return input.mul(this.params.get("weight").tensor).add(this.params.get("bias").tensor);
    }
  }
  const packagedBuffered = new PackagedBufferedHead();
  const packagedBufferedParent = new PackagedBufferedParent();
  const packagedBufferedState = packagedBuffered.stateDict("buffered");
  const packagedBufferedOptionsState = packagedBuffered.stateDict({ prefix: "buffered" });
  const packagedBufferedOptionsStateAlias = packagedBuffered.state_dict({ prefix: "buffered" });
  const packagedBufferedNamespaceOptionsState = adapter.nn.stateDict(packagedBuffered, { prefix: "buffered" });
  const packagedBufferedNamespaceOptionsStateAlias = adapter.nn.state_dict(packagedBuffered, { prefix: "buffered" });
  if (
    packagedBuffered.namedBuffers("buffered").map((entry: Record<string, unknown>) => entry.name).join("|") !== "buffered.offset|buffered.scale" ||
    adapter.nn.namedBuffers(packagedBuffered, "buffered").map((entry: Record<string, unknown>) => entry.name).join("|") !== "buffered.offset|buffered.scale" ||
    packagedBufferedParent.namedBuffers("parent").map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root|parent.child.offset|parent.child.scale" ||
    packagedBufferedParent.namedBuffers("parent", { recurse: false }).map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root" ||
    packagedBufferedParent.named_buffers({ prefix: "parent", recurse: false }).map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root" ||
    adapter.nn.buffers(packagedBufferedParent, { prefix: "parent", recurse: false }).map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root" ||
    adapter.nn.namedBuffers(packagedBufferedParent, "parent", { recurse: false }).map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root" ||
    adapter.nn.named_buffers(packagedBufferedParent, "parent", { recurse: false }).map((entry: Record<string, unknown>) => entry.name).join("|") !== "parent.root" ||
    !packagedBufferedState["buffered.offset"] ||
    !packagedBufferedState["buffered.scale"] ||
    !packagedBufferedOptionsState["buffered.offset"] ||
    !packagedBufferedOptionsStateAlias["buffered.scale"] ||
    !packagedBufferedNamespaceOptionsState["buffered.offset"] ||
    !packagedBufferedNamespaceOptionsStateAlias["buffered.scale"] ||
    packagedBuffered.parameters().length !== 0
  ) {
    throw new Error(`${label} expected subclassed nn.Buffer fields and registered buffers to traverse and serialize through the package root`);
  }
  const packagedContainerInput = adapter.tensor([3], [1]);
  const packagedModuleListHead = new PackagedModuleListHead();
  const packagedModuleListProgram = packagedModuleListHead.compile({ backend: "cpu", inputShape: [1] });
  const packagedModuleListSession = packagedModuleListProgram.bindModule(packagedModuleListHead);
  try {
    if (
      packagedModuleListHead.parameterNames("moduleList").join("|") !== "moduleList.layers.0.weight|moduleList.layers.0.bias" ||
      packagedModuleListHead.namedModules("moduleList").map((entry: Record<string, unknown>) => entry.name).join("|") !== "moduleList|moduleList.layers.0" ||
      adapter.nn.getSubmodule(packagedModuleListHead, "layers.0") !== packagedModuleListHead.layers.at(0)
    ) {
      throw new Error(`${label} expected subclassed nn.ModuleList to participate in module traversal`);
    }
    expectClose(packagedModuleListSession.stepTensor(packagedContainerInput).data, packagedModuleListHead.forward(packagedContainerInput).data, `${label} compiled subclassed nn.ModuleList`);
  } finally {
    packagedModuleListSession.free();
    packagedModuleListProgram.free();
  }
  const packagedModuleDictHead = new PackagedModuleDictHead();
  const packagedModuleDictProgram = packagedModuleDictHead.compile({ backend: "cpu", inputShape: [1] });
  const packagedModuleDictSession = packagedModuleDictProgram.bindModule(packagedModuleDictHead);
  try {
    if (
      packagedModuleDictHead.parameterNames("moduleDict").join("|") !== "moduleDict.layers.head.weight|moduleDict.layers.head.bias" ||
      packagedModuleDictHead.namedChildren("moduleDict").map((entry: Record<string, unknown>) => entry.name).join("|") !== "moduleDict.layers.head" ||
      adapter.nn.getSubmodule(packagedModuleDictHead, "layers.head") !== packagedModuleDictHead.layers.get("head")
    ) {
      throw new Error(`${label} expected subclassed nn.ModuleDict to participate in module traversal`);
    }
    expectClose(packagedModuleDictSession.stepTensor(packagedContainerInput).data, packagedModuleDictHead.forward(packagedContainerInput).data, `${label} compiled subclassed nn.ModuleDict`);
  } finally {
    packagedModuleDictSession.free();
    packagedModuleDictProgram.free();
  }
  const packagedParameterListHead = new PackagedParameterListHead();
  const packagedParameterDictHead = new PackagedParameterDictHead();
  for (const [kind, target] of [
    ["ParameterList", packagedParameterListHead],
    ["ParameterDict", packagedParameterDictHead],
  ] as const) {
    const names = target.parameterNames(kind);
    const state = adapter.nn.stateDict(target, kind);
    const optimizer = adapter.optim.sgd(target, { lr: 0.01 });
    adapter.loss.mse(target.forward(packagedContainerInput), adapter.tensor([1], [1])).backward();
    if (
      names.length !== 2 ||
      !state[names[0]] ||
      !state[names[1]] ||
      gradNorm(target.parameters()) <= 0
    ) {
      throw new Error(`${label} expected subclassed nn.${kind} parameters to traverse, serialize, and receive gradients`);
    }
    optimizer.zeroGrad();
    if (gradNorm(target.parameters()) !== 0) {
      throw new Error(`${label} expected subclassed nn.${kind} optimizer zeroGrad to see container parameters`);
    }
  }
  const singleLayerSequential = adapter.nn.sequential(adapter.nn.relu());
  if (singleLayerSequential.length !== 1 || singleLayerSequential.forward(adapter.tensor([-1, 2], [2])).data.join("|") !== "0|2") {
    throw new Error(`${label} expected nn.sequential(layer) to create a one-layer Sequential`);
  }
  const input = adapter.tensor([2, -3], [2]);
  const eager = sequential.forward(input);
  expectClose(eager.data, [2], `${label} sequential eager output`);
  expectClose(adapter.nn.forward(sequential, input).data, eager.data, `${label} namespace forward matches module forward`);
  if (sequential.parameterNames().join("|") !== "0.weight|0.bias|2.weight|2.bias") {
    throw new Error(`${label} expected sequential fully qualified parameter names`);
  }
  if (sequential.parameterNames("model").join("|") !== "model.0.weight|model.0.bias|model.2.weight|model.2.bias") {
    throw new Error(`${label} expected sequential prefixed parameter names`);
  }
  if (adapter.nn.namedParameters(sequential, "model").map((param: Record<string, unknown>) => param.name).join("|") !== "model.0.weight|model.0.bias|model.2.weight|model.2.bias") {
    throw new Error(`${label} expected namespace prefixed named parameters`);
  }
  if (
    sequential.parameters({ recurse: false }).length !== 0 ||
    sequential.namedParameters("model", { recurse: false }).length !== 0 ||
    adapter.nn.parameters(sequential, { recurse: false }).length !== 0 ||
    adapter.nn.namedParameters(sequential, "model", { recurse: false }).length !== 0
  ) {
    throw new Error(`${label} expected PyTorch-style recurse:false parameter traversal to exclude child parameters`);
  }
  if (sequential.named_parameters("model").map((param: Record<string, unknown>) => param.name).join("|") !== "model.0.weight|model.0.bias|model.2.weight|model.2.bias") {
    throw new Error(`${label} expected module named_parameters alias`);
  }
  if (adapter.nn.named_parameters(sequential, "model").map((param: Record<string, unknown>) => param.name).join("|") !== "model.0.weight|model.0.bias|model.2.weight|model.2.bias") {
    throw new Error(`${label} expected namespace named_parameters alias`);
  }
  if (sequential.named_children("model").map((entry: Record<string, unknown>) => entry.name).join("|") !== "model.0|model.1|model.2") {
    throw new Error(`${label} expected module named_children alias`);
  }
  if (adapter.nn.named_children(sequential, "model").map((entry: Record<string, unknown>) => entry.name).join("|") !== "model.0|model.1|model.2") {
    throw new Error(`${label} expected namespace named_children alias`);
  }
  if (sequential.named_modules("model").map((entry: Record<string, unknown>) => entry.name).join("|") !== "model|model.0|model.1|model.2") {
    throw new Error(`${label} expected module named_modules alias`);
  }
  if (adapter.nn.named_modules(sequential, "model").map((entry: Record<string, unknown>) => entry.name).join("|") !== "model|model.0|model.1|model.2") {
    throw new Error(`${label} expected namespace named_modules alias`);
  }
  if (sequential.canCompile({ inputShape: [2], backend: "cpu" }) !== true) {
    throw new Error(`${label} expected sequential canCompile`);
  }
  const linearModule = adapter.nn.linear(2, 1, {
    weight: [0.5, -0.25],
    bias: [0],
  });
  const linearTrace = linearModule.trace({ inputShape: [2], backend: "cpu" });
  const linearIr = linearModule.tensorProgramIr({ inputShape: [2], backend: "cpu" });
  const linearKernelPlan = linearModule.kernelPlan({ inputShape: [2], backend: "cpu" });
  const linearKernelPlanSignature = linearKernelPlan.signature;
  const linearCompileExplanation = linearModule.compileExplanation({ inputShape: [2], backend: "cpu" });
  const linearCompilePlan = linearModule.compilePlan({ inputShape: [2], backend: "cpu" });
  const linearPreflight = linearModule.preflight({ inputShape: [2], backend: "cpu" });
  const linearCompileExplanationAlias = linearModule.compile_explanation({ inputShape: [2], backend: "cpu" });
  const linearCompilePlanAlias = linearModule.compile_plan({ inputShape: [2], backend: "cpu" });
  const linearRequiredSupport = linearModule.requireCompileSupport({ inputShape: [2], backend: "cpu" });
  const linearRequiredSupportAlias = linearModule.require_compile_support({ inputShape: [2], backend: "cpu" });
  const linearRequiredCompilePlan = linearModule.requireCompilePlan({ inputShape: [2], backend: "cpu" });
  const linearRequiredCompilePlanAlias = linearModule.require_compile_plan({ inputShape: [2], backend: "cpu" });
  const linearAssertedCompilePlan = linearModule.assertCompilePlan({ inputShape: [2], backend: "cpu" });
  const linearAssertedCompilePlanAlias = linearModule.assert_compile_plan({ inputShape: [2], backend: "cpu" });
  if (
    !Object.isFrozen(linearTrace) ||
    !Object.isFrozen(linearIr) ||
    !Object.isFrozen(linearKernelPlan) ||
    !Object.isFrozen(linearCompileExplanation) ||
    !Object.isFrozen(linearCompilePlan) ||
    !Object.isFrozen(linearPreflight) ||
    !Object.isFrozen(linearCompileExplanationAlias) ||
    !Object.isFrozen(linearCompilePlanAlias) ||
    !Object.isFrozen(linearRequiredCompilePlan) ||
    !Object.isFrozen(linearRequiredCompilePlanAlias) ||
    !Object.isFrozen(linearAssertedCompilePlan) ||
    !Object.isFrozen(linearAssertedCompilePlanAlias) ||
    linearModule.compile_support({ inputShape: [2], backend: "cpu" }).signature !== linearModule.compileSupport({ inputShape: [2], backend: "cpu" }).signature ||
    linearModule.inputShape({ inputShape: [2], backend: "cpu" }).join("x") !== "2" ||
    linearModule.input_shape({ inputShape: [2], backend: "cpu" }).join("x") !== "2" ||
    linearModule.outputShape({ inputShape: [2], backend: "cpu" }).join("x") !== "1" ||
    linearModule.output_shape({ inputShape: [2], backend: "cpu" }).join("x") !== "1" ||
    linearModule.compilerSignatures({ inputShape: [2], backend: "cpu" }).kernelPlan !== linearKernelPlan.signature ||
    linearModule.compiler_signatures({ inputShape: [2], backend: "cpu" }).kernelPlan !== linearKernelPlan.signature ||
    linearModule.tensor_program_ir({ inputShape: [2], backend: "cpu" }).signature !== linearIr.signature ||
    linearModule.kernel_plan({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.signature ||
    linearModule.bufferLayout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.bufferLayout.signature ||
    linearModule.buffer_layout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.bufferLayout.signature ||
    linearModule.memoryLayout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.memoryLayout.signature ||
    linearModule.memory_layout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.memoryLayout.signature ||
    linearModule.shapeConstraints({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.shapeConstraints.signature ||
    linearModule.shape_constraints({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.shapeConstraints.signature ||
    linearModule.parameterLayout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.parameterLayout.signature ||
    linearModule.parameter_layout({ inputShape: [2], backend: "cpu" }).signature !== linearKernelPlan.parameterLayout.signature ||
    kernelPlanPolicy.acceptsKernelPlan(linearKernelPlan) !== true ||
    kernelPlanPolicy.requireKernelPlan(linearKernelPlan).signature !== linearKernelPlanSignature ||
    kernelPlanPolicy.assert_kernel_plan(linearKernelPlan).signature !== linearKernelPlanSignature ||
    kernelPlanPolicy.matchesKernelPlanSignature(linearKernelPlan, linearKernelPlanSignature) !== true ||
    inspection.acceptsKernelPlan(linearKernelPlan) !== true ||
    inspection.requireKernelPlan(linearKernelPlan).signature !== linearKernelPlanSignature ||
    inspection.matchesKernelPlanSignature(linearKernelPlan, linearKernelPlanSignature) !== true ||
    linearCompileExplanation.kind !== "zgml.nn.compile-explanation" ||
    linearPreflight.kind !== "zgml.nn.compile-explanation" ||
    linearCompileExplanationAlias.signature !== linearCompileExplanation.signature ||
    linearCompilePlan.signature !== linearCompileExplanation.signature ||
    linearPreflight.signature !== linearCompileExplanation.signature ||
    linearCompilePlanAlias.signature !== linearCompileExplanation.signature ||
    linearRequiredCompilePlan.signature !== linearCompileExplanation.signature ||
    linearRequiredCompilePlanAlias.signature !== linearCompileExplanation.signature ||
    linearAssertedCompilePlan.signature !== linearCompileExplanation.signature ||
    linearAssertedCompilePlanAlias.signature !== linearCompileExplanation.signature ||
    linearRequiredSupport.signature !== linearModule.compileSupport({ inputShape: [2], backend: "cpu" }).signature ||
    linearRequiredSupportAlias.signature !== linearRequiredSupport.signature ||
    linearModule.can_compile({ inputShape: [2], backend: "cpu" }) !== true ||
    linearCompileExplanation.kernelPlan.signature !== linearKernelPlan.signature
  ) {
    throw new Error(`${label} expected plain module instance compile-evidence helpers`);
  }
  expectThrowIncludes(
    () => kernelPlanPolicy.requireKernelPlan({ ...linearKernelPlan, signature: `${linearKernelPlanSignature}:tampered` }),
    "signature does not match KernelPlan schedule evidence",
    `${label} KernelPlan require rejects tampered evidence`,
  );
  const batchedLinearModule = adapter.nn.linear(2, 3, {
    weight: [1, 0, 0.5, 0, 1, -0.5],
    bias: [0.25, -0.25, 0.5],
  });
  const batchedLinearSupport = batchedLinearModule.compileSupport({ inputShape: [2, 2], backend: "cpu" });
  if (
    batchedLinearSupport.supported !== true ||
    batchedLinearSupport.outputShape?.join("x") !== "2x3" ||
    batchedLinearSupport.kernelPlan?.ops[0]?.op !== "linear" ||
    batchedLinearSupport.kernelPlan?.ops[0]?.kernel !== "linear"
  ) {
    throw new Error(`${label} expected batched nn.Linear to compile to native Program`);
  }
  const batchedLinearProgram = batchedLinearModule.compile({ inputShape: [2, 2], backend: "cpu" });
  const batchedLinearSession = batchedLinearProgram.bindModule(batchedLinearModule);
  try {
    const batchedLinearInput = adapter.tensor([1, 2, 3, 4], [2, 2]);
    expectClose(
      batchedLinearSession.executeInto(new Float32Array(6), { input: batchedLinearInput }),
      batchedLinearModule.forward(batchedLinearInput).data,
      `${label} compiled nn.Linear batched output`,
    );
  } finally {
    batchedLinearSession.dispose();
    batchedLinearProgram.dispose();
  }
  expectNativeEagerLinearEvidence(adapter, label);
  const rank3LinearSupport = batchedLinearModule.compileSupport({ inputShape: [2, 2, 2], backend: "cpu" });
  if (
    rank3LinearSupport.supported !== true ||
    rank3LinearSupport.outputShape?.join("x") !== "2x2x3" ||
    rank3LinearSupport.kernelPlan?.ops[0]?.op !== "linear" ||
    rank3LinearSupport.kernelPlan?.ops[0]?.kernel !== "linear"
  ) {
    throw new Error(`${label} expected rank-3 nn.Linear to compile to native Program`);
  }
  const rank3LinearProgram = batchedLinearModule.compile({ inputShape: [2, 2, 2], backend: "cpu" });
  const rank3LinearSession = rank3LinearProgram.bindModule(batchedLinearModule);
  try {
    const rank3LinearInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    expectClose(
      rank3LinearSession.executeInto(new Float32Array(12), { input: rank3LinearInput }),
      batchedLinearModule.forward(rank3LinearInput).data,
      `${label} compiled nn.Linear rank-3 output`,
    );
  } finally {
    rank3LinearSession.dispose();
    rank3LinearProgram.dispose();
  }
  const namespaceTrace = adapter.compile.trace(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceSupport = adapter.compile.compileSupport(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceSupportAlias = adapter.compile.compile_support(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceRequiredSupport = adapter.nn.requireCompileSupport(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceRequiredSupportAlias = adapter.nn.require_compile_support(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceRequiredCompilePlan = adapter.nn.requireCompilePlan(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceRequiredCompilePlanAlias = adapter.nn.require_compile_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceAssertedCompilePlan = adapter.nn.assertCompilePlan(linearModule, { inputShape: [2], backend: "cpu" });
  const nnNamespaceAssertedCompilePlanAlias = adapter.nn.assert_compile_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceExplanation = adapter.compile.explain(linearModule, { inputShape: [2], backend: "cpu" });
  const namespacePreflight = adapter.compile.preflight(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompileExplanation = adapter.compile.compileExplanation(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompileExplanationAlias = adapter.compile.compile_explanation(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompilePlan = adapter.compile.compilePlan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompilePlanAlias = adapter.compile.compile_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceRequiredCompilePlan = adapter.compile.requireCompilePlan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceRequiredCompilePlanAlias = adapter.compile.require_compile_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceAssertedCompilePlan = adapter.compile.assertCompilePlan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceAssertedCompilePlanAlias = adapter.compile.assert_compile_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompilerSignatures = adapter.compile.compilerSignatures(linearModule, { inputShape: [2], backend: "cpu" });
  const compileNamespaceCompilerSignaturesAlias = adapter.compile.compiler_signatures(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceIr = adapter.compile.tensorProgramIr(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceIrAlias = adapter.compile.tensor_program_ir(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceKernelPlan = adapter.compile.kernelPlan(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceKernelPlanAlias = adapter.compile.kernel_plan(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceBufferLayout = adapter.compile.bufferLayout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceBufferLayoutAlias = adapter.compile.buffer_layout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceMemoryLayout = adapter.compile.memoryLayout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceMemoryLayoutAlias = adapter.compile.memory_layout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceInputShape = adapter.compile.inputShape(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceInputShapeAlias = adapter.compile.input_shape(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceOutputShape = adapter.compile.outputShape(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceOutputShapeAlias = adapter.compile.output_shape(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceShapeConstraints = adapter.compile.shapeConstraints(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceShapeConstraintsAlias = adapter.compile.shape_constraints(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceParameterLayout = adapter.compile.parameterLayout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceParameterLayoutAlias = adapter.compile.parameter_layout(linearModule, { inputShape: [2], backend: "cpu" });
  const namespaceProgram = adapter.compile.compile(linearModule, { backend: "cpu" });
  const namespaceDirectProgram = adapter.compile(linearModule, { backend: "cpu" });
  const namespaceProgramEvidence = namespaceProgram.compileEvidence();
  const namespaceDirectProgramEvidence = namespaceDirectProgram.compileEvidence();
  try {
    if (
      !Object.isFrozen(namespaceTrace) ||
      !Object.isFrozen(namespaceSupport) ||
      !Object.isFrozen(namespaceExplanation) ||
      !Object.isFrozen(namespacePreflight) ||
      namespaceTrace.signature !== linearTrace.signature ||
      namespaceSupport.signature !== linearModule.compileSupport({ inputShape: [2], backend: "cpu" }).signature ||
      namespaceSupportAlias.signature !== namespaceSupport.signature ||
      nnNamespaceRequiredSupport.signature !== namespaceSupport.signature ||
      nnNamespaceRequiredSupportAlias.signature !== namespaceSupport.signature ||
      nnNamespaceRequiredCompilePlan.signature !== linearCompileExplanation.signature ||
      nnNamespaceRequiredCompilePlanAlias.signature !== linearCompileExplanation.signature ||
      nnNamespaceAssertedCompilePlan.signature !== linearCompileExplanation.signature ||
      nnNamespaceAssertedCompilePlanAlias.signature !== linearCompileExplanation.signature ||
      namespaceExplanation.signature !== linearCompileExplanation.signature ||
      namespacePreflight.signature !== linearCompileExplanation.signature ||
      compileNamespaceCompileExplanation.signature !== linearCompileExplanation.signature ||
      compileNamespaceCompileExplanationAlias.signature !== linearCompileExplanation.signature ||
      compileNamespaceCompilePlan.signature !== linearCompileExplanation.signature ||
      compileNamespaceCompilePlanAlias.signature !== linearCompileExplanation.signature ||
      compileNamespaceRequiredCompilePlan.signature !== linearCompileExplanation.signature ||
      compileNamespaceRequiredCompilePlanAlias.signature !== linearCompileExplanation.signature ||
      compileNamespaceAssertedCompilePlan.signature !== linearCompileExplanation.signature ||
      compileNamespaceAssertedCompilePlanAlias.signature !== linearCompileExplanation.signature ||
      compileNamespaceCompilerSignatures.kernelPlan !== linearKernelPlan.signature ||
      compileNamespaceCompilerSignaturesAlias.kernelPlan !== linearKernelPlan.signature ||
      namespaceIr.signature !== linearIr.signature ||
      namespaceIrAlias.signature !== linearIr.signature ||
      namespaceKernelPlan.signature !== linearKernelPlan.signature ||
      namespaceKernelPlanAlias.signature !== linearKernelPlan.signature ||
      namespaceBufferLayout.signature !== linearKernelPlan.bufferLayout.signature ||
      namespaceBufferLayoutAlias.signature !== linearKernelPlan.bufferLayout.signature ||
      namespaceMemoryLayout.signature !== linearKernelPlan.memoryLayout.signature ||
      namespaceMemoryLayoutAlias.signature !== linearKernelPlan.memoryLayout.signature ||
      namespaceInputShape.join("x") !== "2" ||
      namespaceInputShapeAlias.join("x") !== "2" ||
      namespaceOutputShape.join("x") !== "1" ||
      namespaceOutputShapeAlias.join("x") !== "1" ||
      namespaceShapeConstraints.signature !== linearKernelPlan.shapeConstraints.signature ||
      namespaceShapeConstraintsAlias.signature !== linearKernelPlan.shapeConstraints.signature ||
      namespaceParameterLayout.signature !== linearKernelPlan.parameterLayout.signature ||
      namespaceParameterLayoutAlias.signature !== linearKernelPlan.parameterLayout.signature ||
      adapter.compile.canCompile(linearModule, { inputShape: [2], backend: "cpu" }) !== true ||
      adapter.compile.can_compile(linearModule, { inputShape: [2], backend: "cpu" }) !== true ||
      namespaceProgramEvidence.compilerSignatures.kernelPlan !== linearKernelPlan.signature ||
      adapter.program.isProgramCompileEvidence(namespaceProgramEvidence) !== true ||
      adapter.program.requireProgramCompileEvidence(namespaceProgramEvidence).signature !== namespaceProgramEvidence.signature ||
      adapter.program.assert_program_compile_evidence(namespaceProgramEvidence).kernelPlan.signature !== linearKernelPlan.signature ||
      adapter.program.programCompileEvidenceSignature(namespaceProgramEvidence) !== namespaceProgramEvidence.signature ||
      adapter.program.matchesProgramCompileEvidenceSignature(namespaceProgramEvidence, namespaceProgramEvidence.signature) !== true ||
      adapter.program.matches_program_compile_evidence_signature(namespaceProgramEvidence, "wrong") !== false ||
      adapter.compile.isProgramCompileEvidence(namespaceProgramEvidence) !== true ||
      adapter.compile.requireProgramCompileEvidence(namespaceProgramEvidence).signature !== namespaceProgramEvidence.signature ||
      adapter.compile.matchesProgramCompileEvidenceSignature(namespaceProgramEvidence, namespaceProgramEvidence.signature) !== true ||
      namespaceDirectProgramEvidence.signature !== namespaceProgramEvidence.signature
    ) {
      throw new Error(`${label} expected root compile namespace to delegate full module compile evidence`);
    }
  } finally {
    namespaceDirectProgram.dispose();
    namespaceProgram.dispose();
  }
  expectThrowIncludes(
    () => adapter.nn.requireCompileSupport({ compileSupport: () => ({ supported: false, reason: "unsupported package smoke module" }) }, { backend: "cpu" }),
    "nn.requireCompileSupport rejected unsupported module: unsupported package smoke module",
    `${label} nn.requireCompileSupport rejects unsupported evidence`,
  );
  expectThrowIncludes(
    () => adapter.nn.requireCompilePlan({ compileSupport: () => ({ supported: false, reason: "unsupported package smoke module" }) }, { backend: "cpu" }),
    "nn.requireCompilePlan rejected unsupported module: unsupported package smoke module",
    `${label} nn.requireCompilePlan rejects unsupported evidence`,
  );
  expectThrowIncludes(
    () => adapter.compile.requireCompilePlan({ compileSupport: () => ({ supported: false, reason: "unsupported package smoke module" }) }, { backend: "cpu" }),
    "compile.requireCompilePlan rejected unsupported target: unsupported package smoke module",
    `${label} compile.requireCompilePlan rejects unsupported evidence`,
  );
  const support = sequential.compileSupport({ inputShape: [2], backend: "cpu" });
  const trace = sequential.trace({ inputShape: [2], backend: "cpu" });
  const ir = adapter.nn.tensorProgramIr(sequential, { inputShape: [2], backend: "cpu" });
  const kernelPlan = adapter.nn.kernelPlan(sequential, { inputShape: [2], backend: "cpu" });
  const explanation = adapter.nn.explain(sequential, { inputShape: [2], backend: "cpu" });
  const preflight = adapter.nn.preflight(sequential, { inputShape: [2], backend: "cpu" });
  const methodExplanation = sequential.explain({ inputShape: [2], backend: "cpu" });
  const methodPreflight = sequential.preflight({ inputShape: [2], backend: "cpu" });
  const compileExplanation = sequential.compileExplanation({ inputShape: [2], backend: "cpu" });
  const namespaceCompileExplanation = adapter.nn.compileExplanation(sequential, { inputShape: [2], backend: "cpu" });
  const namespaceCompileExplanationAlias = adapter.nn.compile_explanation(sequential, { inputShape: [2], backend: "cpu" });
  const compilePlan = adapter.nn.compilePlan(sequential, { inputShape: [2], backend: "cpu" });
  const compilePlanAlias = adapter.nn.compile_plan(sequential, { inputShape: [2], backend: "cpu" });
  const methodCompilePlan = sequential.compilePlan({ inputShape: [2], backend: "cpu" });
  const requiredCompilePlan = adapter.nn.requireCompilePlan(sequential, { inputShape: [2], backend: "cpu" });
  const requiredCompilePlanAlias = adapter.nn.require_compile_plan(sequential, { inputShape: [2], backend: "cpu" });
  const assertedCompilePlan = adapter.nn.assertCompilePlan(sequential, { inputShape: [2], backend: "cpu" });
  const assertedCompilePlanAlias = adapter.nn.assert_compile_plan(sequential, { inputShape: [2], backend: "cpu" });
  const methodRequiredCompilePlan = sequential.requireCompilePlan({ inputShape: [2], backend: "cpu" });
  if (
    !Object.isFrozen(support) ||
    !Object.isFrozen(trace) ||
    !Object.isFrozen(ir) ||
    !Object.isFrozen(kernelPlan) ||
    !Object.isFrozen(explanation) ||
    !Object.isFrozen(preflight) ||
    !Object.isFrozen(methodExplanation) ||
    !Object.isFrozen(methodPreflight) ||
    !Object.isFrozen(compileExplanation) ||
    !Object.isFrozen(namespaceCompileExplanation) ||
    !Object.isFrozen(namespaceCompileExplanationAlias) ||
    !Object.isFrozen(compilePlan) ||
    !Object.isFrozen(compilePlanAlias) ||
    !Object.isFrozen(methodCompilePlan) ||
    !Object.isFrozen(requiredCompilePlan) ||
    !Object.isFrozen(requiredCompilePlanAlias) ||
    !Object.isFrozen(assertedCompilePlan) ||
    !Object.isFrozen(assertedCompilePlanAlias) ||
    !Object.isFrozen(methodRequiredCompilePlan) ||
    !Object.isFrozen(explanation.diagnostics) ||
    !Object.isFrozen(explanation.parameterNames) ||
    !Object.isFrozen(explanation.parameterInfos) ||
    !Object.isFrozen(explanation.parameterInfos[0]) ||
    support.nativePath !== "device-program" ||
    support.layerCount !== 3 ||
    support.outputLen !== 1 ||
    explanation.kind !== "zgml.nn.compile-explanation" ||
    preflight.kind !== "zgml.nn.compile-explanation" ||
    methodExplanation.kind !== "zgml.nn.compile-explanation" ||
    methodPreflight.kind !== "zgml.nn.compile-explanation" ||
    compileExplanation.kind !== "zgml.nn.compile-explanation" ||
    namespaceCompileExplanation.kind !== "zgml.nn.compile-explanation" ||
    namespaceCompileExplanationAlias.kind !== "zgml.nn.compile-explanation" ||
    compilePlan.kind !== "zgml.nn.compile-explanation" ||
    compilePlanAlias.kind !== "zgml.nn.compile-explanation" ||
    methodCompilePlan.kind !== "zgml.nn.compile-explanation" ||
    requiredCompilePlan.kind !== "zgml.nn.compile-explanation" ||
    requiredCompilePlanAlias.kind !== "zgml.nn.compile-explanation" ||
    assertedCompilePlan.kind !== "zgml.nn.compile-explanation" ||
    assertedCompilePlanAlias.kind !== "zgml.nn.compile-explanation" ||
    methodRequiredCompilePlan.kind !== "zgml.nn.compile-explanation" ||
    explanation.supported !== true ||
    preflight.supported !== true ||
    methodExplanation.supported !== true ||
    methodPreflight.supported !== true ||
    compileExplanation.supported !== true ||
    namespaceCompileExplanation.supported !== true ||
    namespaceCompileExplanationAlias.supported !== true ||
    compilePlan.supported !== true ||
    compilePlanAlias.supported !== true ||
    methodCompilePlan.supported !== true ||
    requiredCompilePlan.supported !== true ||
    requiredCompilePlanAlias.supported !== true ||
    assertedCompilePlan.supported !== true ||
    assertedCompilePlanAlias.supported !== true ||
    methodRequiredCompilePlan.supported !== true ||
    typeof explanation.signature !== "string" ||
    !explanation.signature.startsWith("nn-compile-explanation|supported=1|native=device-program|model=module|inputShape=2|outputShape=1|") ||
    preflight.signature !== explanation.signature ||
    namespaceCompileExplanation.signature !== explanation.signature ||
    namespaceCompileExplanationAlias.signature !== explanation.signature ||
    compilePlan.signature !== explanation.signature ||
    compilePlanAlias.signature !== explanation.signature ||
    requiredCompilePlan.signature !== explanation.signature ||
    requiredCompilePlanAlias.signature !== explanation.signature ||
    assertedCompilePlan.signature !== explanation.signature ||
    assertedCompilePlanAlias.signature !== explanation.signature ||
    methodRequiredCompilePlan.signature !== explanation.signature ||
    methodCompilePlan.signature !== explanation.signature ||
    methodPreflight.signature !== explanation.signature ||
    explanation.nativePath !== "device-program" ||
    explanation.modelKind !== "module" ||
    explanation.inputShape.join("x") !== "2" ||
    explanation.outputShape.join("x") !== "1" ||
    explanation.parameterNames.join("|") !== "0.weight|0.bias|2.weight|2.bias" ||
    explanation.parameterInfos.map((info: Record<string, any>) => `${info.index}:${info.name}:${info.scalarCount}:${info.shape.join("x")}`).join("|") !== "0:0.weight:4:2x2|1:0.bias:2:2|2:2.weight:2:2x1|3:2.bias:1:1" ||
    explanation.parameterCount !== 4 ||
    explanation.parameterScalarCount !== ir.parameterScalarCount ||
    explanation.trace.opCount !== trace.opCount ||
    explanation.ir.opCount !== ir.opCount ||
    typeof ir.signature !== "string" ||
    ir.signature !== support.irSignature ||
    explanation.kernelPlan.opCount !== kernelPlan.opCount ||
    typeof kernelPlan.signature !== "string" ||
    kernelPlan.signature !== support.kernelPlanSignature ||
    methodExplanation.kernelPlan.opCount !== kernelPlan.opCount ||
    compileExplanation.kernelPlan.opCount !== kernelPlan.opCount ||
    compilePlan.kernelPlan.opCount !== kernelPlan.opCount ||
    methodCompilePlan.kernelPlan.opCount !== kernelPlan.opCount ||
    explanation.kernelPlan.dispatchCount !== kernelPlan.dispatchCount ||
    compilePlan.kernelPlan.dispatchCount !== kernelPlan.dispatchCount ||
    explanation.compilerSignatures.kernelPlan !== support.kernelPlanSignature ||
    explanation.ir.signature !== support.irSignature ||
    explanation.kernelPlan.signature !== support.kernelPlanSignature ||
    explanation.memoryLayout.signature !== support.memoryLayoutSignature ||
    explanation.parameterLayout.signature !== support.parameterLayoutSignature ||
    explanation.bufferLayout.signature.startsWith("zgml.program.buffer-layout|") !== true ||
    explanation.compilerSignatures.bufferLayout !== support.bufferLayoutSignature ||
    explanation.bufferLayout.output.elementCount !== kernelPlan.bufferLayout.output.elementCount ||
    explanation.memoryLayout.totalByteLength !== kernelPlan.memoryLayout.totalByteLength ||
    explanation.shapeConstraints.inputShape.join("x") !== "2" ||
    explanation.shapeConstraints.outputShape.join("x") !== "1" ||
    explanation.parameterLayout.weightsLen !== kernelPlan.parameterLayout.weightsLen ||
    explanation.parameterLayout.biasLen !== kernelPlan.parameterLayout.biasLen ||
    trace.kind !== "sequential" ||
    trace.opCount !== 3 ||
    ir.opCount !== 3 ||
    ir.parameterScalarCount !== 9 ||
    kernelPlan.opCount !== 3 ||
    kernelPlan.dispatchCount !== 2 ||
    kernelPlan.descriptorCount !== 2 ||
    kernelPlan.ops[0].path !== "0..1" ||
    kernelPlan.ops[0].fusedOps.join("|") !== "linear|activation" ||
    kernelPlan.ops[1].path !== "2"
  ) {
    throw new Error(`${label} expected sequential Trace -> Tensor Program IR -> fused KernelPlan evidence`);
  }
  const program = sequential.compile({ inputShape: [2], backend: "cpu" });
  const programPlan = program.kernelPlan();
  const programIr = program.tensorProgramIr();
  if (
    program.inputShape().join("x") !== "2" ||
    program.outputShape().join("x") !== "1" ||
    !Object.isFrozen(programPlan) ||
    !Object.isFrozen(programIr) ||
    programPlan.signature !== kernelPlan.signature ||
    programIr.signature !== ir.signature ||
    programPlan.dispatchCount !== 2 ||
    programIr.opCount !== 3 ||
    !program.acceptsModule(sequential)
  ) {
    throw new Error(`${label} expected compiled sequential Program evidence to match module support`);
  }
  const session = program.bindModule(sequential);
  try {
    const compiled = session.stepTensor(input);
    if (!(compiled instanceof adapter.Tensor)) throw new Error(`${label} expected sequential Session stepTensor to return a Tensor`);
    expectClose(compiled.data, eager.data, `${label} sequential eager/compiled parity`);
    if (session.parameterNames().join("|") !== "0.weight|0.bias|2.weight|2.bias") {
      throw new Error(`${label} expected sequential Session parameter names`);
    }
  } finally {
    session.dispose();
  }
  const batchedInput = adapter.tensor([2, -3, 1, 4, -2, 5, 0, 3], [4, 2]);
  const batchedEager = sequential.forward(batchedInput);
  const batchedSupport = sequential.compileSupport({ inputShape: [4, 2], backend: "cpu" });
  const batchedPlan = adapter.nn.kernelPlan(sequential, { inputShape: [4, 2], backend: "cpu" });
  if (
    batchedSupport.supported !== true ||
    batchedSupport.inputShape.join("x") !== "4x2" ||
    batchedSupport.outputShape.join("x") !== "4x1" ||
    batchedPlan?.dispatchCount !== 2 ||
    batchedPlan.ops[0].fusedOps.join("|") !== "linear|activation"
  ) {
    throw new Error(`${label} expected batched sequential compile evidence`);
  }
  const batchedProgram = sequential.compile({ inputShape: [4, 2], backend: "cpu" });
  const batchedSession = batchedProgram.bindModule(sequential);
  try {
    const batchedOutput = new Float32Array(4);
    const batchedCompiled = batchedSession.executeInto(batchedOutput, { input: batchedInput });
    if (batchedCompiled !== batchedOutput) throw new Error(`${label} expected batched sequential executeInto to reuse caller output`);
    expectClose(batchedCompiled, batchedEager.data, `${label} batched sequential eager/compiled parity`);
    const hotPath = batchedSession.requireHotStepParams({ input: batchedInput, output: batchedOutput });
    if (hotPath.hotPath !== true || hotPath.runtimeOutputAllocationFree !== true) {
      throw new Error(`${label} expected batched sequential allocation-free hot path`);
    }
  } finally {
    batchedSession.dispose();
    batchedProgram.dispose();
  }
}

function expectNormGeluSequentialProgramEvidence(adapter: Record<string, any>, label: string) {
  const sequential = adapter.nn.sequential([
    adapter.nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: [0, 0] }),
    adapter.nn.layerNorm(2, { eps: 0, weight: [1, 1], bias: [0, 0] }),
    adapter.nn.gelu(),
    adapter.nn.linear(2, 1, { weights: [1, 1], bias: [0] }),
  ]);
  const input = adapter.tensor([[1, 3]], [1, 2]);
  const eager = sequential.forward(input);
  const support = sequential.compileSupport({ inputShape: [1, 2], backend: "cpu" });
  const kernelPlan = adapter.nn.kernelPlan(sequential, { inputShape: [1, 2], backend: "cpu" });
  if (
    sequential.parameterNames().join("|") !== "0.weight|0.bias|1.weight|1.bias|3.weight|3.bias" ||
    !Object.isFrozen(support) ||
    !Object.isFrozen(kernelPlan) ||
    support.supported !== true ||
    support.outputShape.join("x") !== "1x1" ||
    support.weightsLen !== 8 ||
    support.biasLen !== 5 ||
    support.trace.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear|layerNorm|activation|linear" ||
    kernelPlan.opCount !== 4 ||
    kernelPlan.dispatchCount !== 3 ||
    kernelPlan.descriptorCount !== 3 ||
    kernelPlan.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|layer-norm|gelu|linear"
  ) {
    throw new Error(`${label} expected Linear/LayerNorm/GELU/Linear compile evidence`);
  }
  const program = sequential.compile({ inputShape: [1, 2], backend: "cpu" });
  const session = program.bindModule(sequential);
  try {
    const compiled = session.stepTensor(input);
    expectClose(compiled.data, eager.data, `${label} norm gelu sequential eager/compiled parity`);
    if (compiled.shape.join("x") !== "1x1" || session.parameterNames().join("|") !== "0.weight|0.bias|1.weight|1.bias|3.weight|3.bias") {
      throw new Error(`${label} expected norm gelu sequential output shape and parameter names`);
    }
  } finally {
    session.dispose();
  }
  const batchedInput = adapter.tensor([[1, 3], [3, 1], [-1, 2], [4, -2]], [4, 2]);
  const batchedEager = sequential.forward(batchedInput);
  const batchedSupport = sequential.compileSupport({ inputShape: [4, 2], backend: "cpu" });
  const batchedKernelPlan = adapter.nn.kernelPlan(sequential, { inputShape: [4, 2], backend: "cpu" });
  if (
    batchedSupport.supported !== true ||
    batchedSupport.outputShape.join("x") !== "4x1" ||
    batchedKernelPlan?.dispatchCount !== 3 ||
    batchedKernelPlan.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|layer-norm|gelu|linear"
  ) {
    throw new Error(`${label} expected batched Linear/LayerNorm/GELU/Linear compile evidence`);
  }
  const batchedProgram = sequential.compile({ inputShape: [4, 2], backend: "cpu" });
  const batchedSession = batchedProgram.bindModule(sequential);
  try {
    const batchedOutput = new Float32Array(4);
    const batchedCompiled = batchedSession.executeInto(batchedOutput, { input: batchedInput });
    if (batchedCompiled !== batchedOutput) throw new Error(`${label} expected batched norm gelu sequential executeInto to reuse caller output`);
    expectClose(batchedCompiled, batchedEager.data, `${label} batched norm gelu sequential eager/compiled parity`);
  } finally {
    batchedSession.dispose();
    batchedProgram.dispose();
  }
}

function expectZeroParameterProgramEvidence(adapter: Record<string, any>, label: string) {
  const logSoftmax = adapter.nn.logSoftmax(-1);
  const input = adapter.tensor([[1, 2, 3]], [1, 3]);
  const eager = logSoftmax.forward(input);
  const support = logSoftmax.compileSupport({ inputShape: [1, 3], backend: "cpu" });
  if (
    !Object.isFrozen(support) ||
    support.supported !== true ||
    support.weightsLen !== 0 ||
    support.biasLen !== 0 ||
    support.outputShape.join("x") !== "1x3" ||
    support.trace.ops[0].op !== "logSoftmax"
  ) {
    throw new Error(`${label} expected zero-parameter logSoftmax compile support`);
  }
  const program = logSoftmax.compile({ inputShape: [1, 3], backend: "cpu" });
  const layout = program.bufferLayout();
  const kernelPlan = program.kernelPlan();
  if (
    !Object.isFrozen(layout) ||
    layout.kind !== "zgml.program.buffer-layout" ||
    typeof layout.signature !== "string" ||
    !layout.signature.startsWith("zgml.program.buffer-layout|") ||
    !Object.isFrozen(kernelPlan) ||
    program.parameterNames().length !== 0 ||
    layout.weights.elementCount !== 0 ||
    layout.bias.elementCount !== 0 ||
    layout.input.elementCount !== 3 ||
    layout.output.elementCount !== 3 ||
    kernelPlan.opCount !== 1 ||
    kernelPlan.dispatchCount !== 1 ||
    kernelPlan.ops[0].op !== "logSoftmax"
  ) {
    throw new Error(`${label} expected zero-parameter Program layout and kernel evidence`);
  }
  const tensorSession = program.bind({});
  try {
    const compiled = tensorSession.stepTensor(input);
    expectClose(compiled.data, eager.data, `${label} zero-parameter tensor Session output`);
  } finally {
    tensorSession.dispose();
  }

  const nativeInput = program.createInputBuffer();
  const nativeOutput = program.createOutputBuffer();
  try {
    nativeInput.writeFloat32(input.data);
    const nativeSession = program.bind({ input: nativeInput, output: nativeOutput });
    try {
      nativeSession.step();
      expectClose(nativeOutput.readFloat32(3), eager.data, `${label} zero-parameter NativeBuffer Session output`);
    } finally {
      nativeSession.dispose();
    }
  } finally {
    nativeInput.dispose();
    nativeOutput.dispose();
  }
  expectThrowIncludes(
    () => program.bind({ weights: new Float32Array([1]) }),
    "weights length 1 does not match Program weights slot length 0",
    `${label} zero-parameter rejects non-empty weights`,
  );
  expectThrowIncludes(
    () => program.bind({ bias: new Float32Array([1]) }),
    "bias length 1 does not match Program bias slot length 0",
    `${label} zero-parameter rejects non-empty bias`,
  );

  const permute = adapter.nn.permute([1, 0]);
  const permuteInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const permuteEager = permute.forward(permuteInput);
  const permuteSupport = permute.compileSupport({ inputShape: [2, 3], backend: "cpu" });
  const permuteKernelPlan = adapter.nn.kernelPlan(permute, { inputShape: [2, 3], backend: "cpu" });
  if (
    !Object.isFrozen(permuteSupport) ||
    !Object.isFrozen(permuteKernelPlan) ||
    permuteSupport.supported !== true ||
    permuteSupport.outputShape.join("x") !== "3x2" ||
    permuteSupport.weightsLen !== 0 ||
    permuteSupport.biasLen !== 0 ||
    permuteKernelPlan.ops.length !== 1 ||
    permuteKernelPlan.ops[0].op !== "permute" ||
    permuteKernelPlan.ops[0].nativeKernels.join("|") !== "transpose"
  ) {
    throw new Error(`${label} expected rank-2 permute to lower through transpose Program evidence`);
  }
  const permuteProgram = permute.compile({ inputShape: [2, 3], backend: "cpu" });
  const permuteSession = permuteProgram.bind({});
  try {
    const compiledPermute = permuteSession.stepTensor(permuteInput);
    if (compiledPermute.shape.join("x") !== "3x2") {
      throw new Error(`${label} expected compiled permute output shape`);
    }
    expectClose(compiledPermute.data, permuteEager.data, `${label} rank-2 permute eager/compiled parity`);
  } finally {
    permuteSession.dispose();
    permuteProgram.dispose();
  }

  const rank3BatchedTranspose = adapter.nn.transpose(1, 2);
  const rank3BatchedInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
  const rank3BatchedTransposeEager = rank3BatchedTranspose.forward(rank3BatchedInput);
  const rank3BatchedTransposeSupport = rank3BatchedTranspose.compileSupport({ inputShape: [2, 2, 3], backend: "cpu" });
  const rank3BatchedTransposeKernelPlan = adapter.nn.kernelPlan(rank3BatchedTranspose, { inputShape: [2, 2, 3], backend: "cpu" });
  if (
    rank3BatchedTransposeSupport.supported !== true ||
    rank3BatchedTransposeSupport.outputShape.join("x") !== "2x3x2" ||
    rank3BatchedTransposeKernelPlan.ops.length !== 1 ||
    rank3BatchedTransposeKernelPlan.ops[0].op !== "transpose" ||
    rank3BatchedTransposeKernelPlan.ops[0].nativeKernels.join("|") !== "transpose"
  ) {
    throw new Error(`${label} expected batched rank-3 transpose to lower through transpose Program evidence`);
  }
  const rank3BatchedTransposeProgram = rank3BatchedTranspose.compile({ inputShape: [2, 2, 3], backend: "cpu" });
  const rank3BatchedTransposeSession = rank3BatchedTransposeProgram.bind({});
  try {
    const compiledRank3BatchedTranspose = rank3BatchedTransposeSession.stepTensor(rank3BatchedInput);
    if (compiledRank3BatchedTranspose.shape.join("x") !== "2x3x2") {
      throw new Error(`${label} expected compiled batched rank-3 transpose output shape`);
    }
    expectClose(compiledRank3BatchedTranspose.data, rank3BatchedTransposeEager.data, `${label} batched rank-3 transpose eager/compiled parity`);
  } finally {
    rank3BatchedTransposeSession.dispose();
    rank3BatchedTransposeProgram.dispose();
  }

  const rank3Permute = adapter.nn.permute([0, 2, 1]);
  const rank3PermuteInput = adapter.tensor([1, 2, 3, 4, 5, 6], [1, 2, 3]);
  const rank3PermuteEager = rank3Permute.forward(rank3PermuteInput);
  const rank3PermuteSupport = rank3Permute.compileSupport({ inputShape: [1, 2, 3], backend: "cpu" });
  const rank3PermuteKernelPlan = adapter.nn.kernelPlan(rank3Permute, { inputShape: [1, 2, 3], backend: "cpu" });
  if (
    rank3PermuteSupport.supported !== true ||
    rank3PermuteSupport.outputShape.join("x") !== "1x3x2" ||
    rank3PermuteKernelPlan.ops.length !== 1 ||
    rank3PermuteKernelPlan.ops[0].op !== "permute" ||
    rank3PermuteKernelPlan.ops[0].nativeKernels.join("|") !== "transpose"
  ) {
    throw new Error(`${label} expected rank-3 single-axis permute to lower through transpose Program evidence`);
  }
  const rank3PermuteProgram = rank3Permute.compile({ inputShape: [1, 2, 3], backend: "cpu" });
  const rank3PermuteSession = rank3PermuteProgram.bind({});
  try {
    const compiledRank3Permute = rank3PermuteSession.stepTensor(rank3PermuteInput);
    if (compiledRank3Permute.shape.join("x") !== "1x3x2") {
      throw new Error(`${label} expected compiled rank-3 permute output shape`);
    }
    expectClose(compiledRank3Permute.data, rank3PermuteEager.data, `${label} rank-3 permute eager/compiled parity`);
  } finally {
    rank3PermuteSession.dispose();
    rank3PermuteProgram.dispose();
  }

  const rank3CyclePermute = adapter.nn.permute([1, 2, 0]);
  const rank3CyclePermuteEager = rank3CyclePermute.forward(rank3PermuteInput);
  const rank3CyclePermuteSupport = rank3CyclePermute.compileSupport({ inputShape: [1, 2, 3], backend: "cpu" });
  const rank3CyclePermuteKernelPlan = adapter.nn.kernelPlan(rank3CyclePermute, { inputShape: [1, 2, 3], backend: "cpu" });
  if (
    rank3CyclePermuteSupport.supported !== true ||
    rank3CyclePermuteSupport.outputShape.join("x") !== "2x3x1" ||
    rank3CyclePermuteKernelPlan.ops.length !== 1 ||
    rank3CyclePermuteKernelPlan.ops[0].op !== "permute" ||
    rank3CyclePermuteKernelPlan.ops[0].nativeKernels.join("|") !== "transpose|transpose"
  ) {
    throw new Error(`${label} expected rank-3 cycle permute to lower through transpose-chain Program evidence`);
  }
  const rank3CyclePermuteProgram = rank3CyclePermute.compile({ inputShape: [1, 2, 3], backend: "cpu" });
  const rank3CyclePermuteSession = rank3CyclePermuteProgram.bind({});
  try {
    const compiledRank3CyclePermute = rank3CyclePermuteSession.stepTensor(rank3PermuteInput);
    if (compiledRank3CyclePermute.shape.join("x") !== "2x3x1") {
      throw new Error(`${label} expected compiled rank-3 cycle permute output shape`);
    }
    expectClose(compiledRank3CyclePermute.data, rank3CyclePermuteEager.data, `${label} rank-3 cycle permute eager/compiled parity`);
  } finally {
    rank3CyclePermuteSession.dispose();
    rank3CyclePermuteProgram.dispose();
  }

  const rank3BatchedPermute = adapter.nn.permute([0, 2, 1]);
  const rank3BatchedPermuteEager = rank3BatchedPermute.forward(rank3BatchedInput);
  const rank3BatchedPermuteSupport = rank3BatchedPermute.compileSupport({ inputShape: [2, 2, 3], backend: "cpu" });
  const rank3BatchedPermuteKernelPlan = adapter.nn.kernelPlan(rank3BatchedPermute, { inputShape: [2, 2, 3], backend: "cpu" });
  if (
    rank3BatchedPermuteSupport.supported !== true ||
    rank3BatchedPermuteSupport.outputShape.join("x") !== "2x3x2" ||
    rank3BatchedPermuteKernelPlan.ops.length !== 1 ||
    rank3BatchedPermuteKernelPlan.ops[0].op !== "permute" ||
    rank3BatchedPermuteKernelPlan.ops[0].nativeKernels.join("|") !== "transpose"
  ) {
    throw new Error(`${label} expected batched rank-3 permute to lower through transpose Program evidence`);
  }
  const rank3BatchedPermuteProgram = rank3BatchedPermute.compile({ inputShape: [2, 2, 3], backend: "cpu" });
  const rank3BatchedPermuteSession = rank3BatchedPermuteProgram.bind({});
  try {
    const compiledRank3BatchedPermute = rank3BatchedPermuteSession.stepTensor(rank3BatchedInput);
    if (compiledRank3BatchedPermute.shape.join("x") !== "2x3x2") {
      throw new Error(`${label} expected compiled batched rank-3 permute output shape`);
    }
    expectClose(compiledRank3BatchedPermute.data, rank3BatchedPermuteEager.data, `${label} batched rank-3 permute eager/compiled parity`);
  } finally {
    rank3BatchedPermuteSession.dispose();
    rank3BatchedPermuteProgram.dispose();
  }
}

function expectReductionProgramEvidence(adapter: Record<string, any>, label: string) {
  const input = adapter.tensor([1, 3, 2, 4, 0, -1], [2, 3]);
  const batchedCube = adapter.tensor([1, 3, 2, 4, 0, -1, 10, 20, 30, 3, 2, 1], [2, 2, 3]);
  const compiledCases = [
    { name: "sum-last", module: adapter.nn.sum(-1), expectedShape: "2x1", expectedKernels: "sum" },
    { name: "mean-batch", module: adapter.nn.mean(0), expectedShape: "1x3", expectedKernels: "transpose|mean|transpose" },
    { name: "prod-last", module: adapter.nn.prod(-1), expectedShape: "2x1", expectedKernels: "prod" },
    { name: "max-last", module: adapter.nn.max(-1), expectedShape: "2x1", expectedKernels: "max" },
    { name: "min-last", module: adapter.nn.min(-1), expectedShape: "2x1", expectedKernels: "min" },
    { name: "min-batch", module: adapter.nn.min(0), expectedShape: "1x3", expectedKernels: "transpose|min|transpose" },
    { name: "argmax-last", module: adapter.nn.argmax(-1), expectedShape: "2x1", expectedKernels: "argmax" },
    { name: "argmin-last", module: adapter.nn.argmin(-1), expectedShape: "2x1", expectedKernels: "argmin" },
  ];
  const rank3CompiledCases = [
    { name: "sum-rank3-batched-last", module: adapter.nn.sum(-1), expectedShape: "2x2x1", expectedKernels: "sum" },
    { name: "mean-rank3-batched-last", module: adapter.nn.mean(-1), expectedShape: "2x2x1", expectedKernels: "mean" },
    { name: "prod-rank3-batched-last", module: adapter.nn.prod(-1), expectedShape: "2x2x1", expectedKernels: "prod" },
    { name: "max-rank3-batched-last", module: adapter.nn.max(-1), expectedShape: "2x2x1", expectedKernels: "max" },
    { name: "min-rank3-batched-last", module: adapter.nn.min(-1), expectedShape: "2x2x1", expectedKernels: "min" },
    { name: "argmax-rank3-batched-last", module: adapter.nn.argmax(-1), expectedShape: "2x2x1", expectedKernels: "argmax" },
    { name: "argmin-rank3-batched-last", module: adapter.nn.argmin(-1), expectedShape: "2x2x1", expectedKernels: "argmin" },
  ];
  for (const testCase of compiledCases) {
    const eager = testCase.module.forward(input);
    const support = testCase.module.compileSupport({ inputShape: [2, 3], backend: "cpu" });
    const kernelPlan = adapter.nn.kernelPlan(testCase.module, { inputShape: [2, 3], backend: "cpu" });
    if (
      !Object.isFrozen(support) ||
      !Object.isFrozen(kernelPlan) ||
      support.supported !== true ||
      support.outputShape.join("x") !== testCase.expectedShape ||
      support.weightsLen !== 0 ||
      support.biasLen !== 0 ||
      kernelPlan.ops.length !== 1 ||
      kernelPlan.ops[0].op !== testCase.module.kind ||
      kernelPlan.ops[0].nativeKernels.join("|") !== testCase.expectedKernels
    ) {
      throw new Error(`${label} expected ${testCase.name} reduction compile evidence`);
    }
    const program = testCase.module.compile({ inputShape: [2, 3], backend: "cpu" });
    const session = program.bind({});
    try {
      const compiled = session.stepTensor(input);
      if (compiled.shape.join("x") !== testCase.expectedShape) {
        throw new Error(`${label} expected ${testCase.name} compiled reduction shape`);
      }
      expectClose(compiled.data, eager.data, `${label} ${testCase.name} eager/compiled parity`);
    } finally {
      session.dispose();
      program.dispose();
    }
  }
  for (const testCase of rank3CompiledCases) {
    const eager = testCase.module.forward(batchedCube);
    const support = testCase.module.compileSupport({ inputShape: [2, 2, 3], backend: "cpu" });
    const kernelPlan = adapter.nn.kernelPlan(testCase.module, { inputShape: [2, 2, 3], backend: "cpu" });
    if (
      !Object.isFrozen(support) ||
      !Object.isFrozen(kernelPlan) ||
      support.supported !== true ||
      support.outputShape.join("x") !== testCase.expectedShape ||
      support.weightsLen !== 0 ||
      support.biasLen !== 0 ||
      kernelPlan.ops.length !== 1 ||
      kernelPlan.ops[0].op !== testCase.module.kind ||
      kernelPlan.ops[0].nativeKernels.join("|") !== testCase.expectedKernels
    ) {
      throw new Error(`${label} expected ${testCase.name} reduction compile evidence`);
    }
    const program = testCase.module.compile({ inputShape: [2, 2, 3], backend: "cpu" });
    const session = program.bind({});
    try {
      const compiled = session.stepTensor(batchedCube);
      if (compiled.shape.join("x") !== testCase.expectedShape) {
        throw new Error(`${label} expected ${testCase.name} compiled reduction shape`);
      }
      expectClose(compiled.data, eager.data, `${label} ${testCase.name} eager/compiled parity`);
    } finally {
      session.dispose();
      program.dispose();
    }
  }

}

function expectDiagonalModuleEvidence(adapter: Record<string, any>, label: string) {
  const diagonal = adapter.nn.diagonal();
  const input = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  expectClose(diagonal.forward(input).data, [1, 5], `${label} nn.diagonal eager forward`);
  const trace = diagonal.trace({ inputShape: [2, 3], backend: "cpu" });
  const support = diagonal.compileSupport({ inputShape: [2, 3], backend: "cpu" });
  const ir = diagonal.tensorProgramIr({ inputShape: [2, 3], backend: "cpu" });
  const kernelPlan = adapter.nn.kernelPlan(diagonal, { inputShape: [2, 3], backend: "cpu" });
  if (
    trace.outputShape.join("x") !== "2" ||
    !ir ||
    ir.ops[0]?.op !== "diagonal" ||
    support.supported !== true ||
    support.outputShape.join("x") !== "2" ||
    kernelPlan.dispatchCount !== 1 ||
    kernelPlan.ops.length !== 1 ||
    kernelPlan.ops[0]?.op !== "diagonal" ||
    kernelPlan.ops[0]?.nativeKernels.join("|") !== "diagonal"
  ) {
    throw new Error(`${label} expected nn.diagonal native Program evidence`);
  }
  const program = diagonal.compile({ inputShape: [2, 3], backend: "cpu" });
  const session = program.bind({});
  try {
    const compiled = session.stepTensor(input);
    expectClose(compiled.data, [1, 5], `${label} compiled nn.diagonal output`);
  } finally {
    session.dispose();
    program.dispose();
  }
}

function expectShapeMovementProgramEvidence(adapter: Record<string, any>, label: string) {
  const matrix = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const row = adapter.tensor([1, 2, 3], [1, 3]);
  const vector = adapter.tensor([1, 2, 3], [3]);
  const cube = adapter.tensor([1, 2, 3, 4, 5, 6], [1, 2, 3]);
  const batchedCube = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
  const compiledRepeatCases = [
    { name: "repeat", module: adapter.nn.repeat([2]), evidence: "expected nn.repeat rank-1 repeat/tile native Program evidence", outputEvidence: "compiled nn.repeat rank-1 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3] },
    { name: "tile", module: new adapter.nn.Tile([2]), evidence: "expected nn.tile rank-1 repeat/tile native Program evidence", outputEvidence: "compiled nn.tile rank-1 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3] },
    { name: "repeat", module: adapter.nn.repeat([2, 3]), input: matrix, inputShape: [2, 3], expectedShape: "4x9", evidence: "expected nn.repeat rank-2 repeat/tile native Program evidence", outputEvidence: "compiled nn.repeat rank-2 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6] },
    { name: "tile", module: new adapter.nn.Tile([2, 3]), input: matrix, inputShape: [2, 3], expectedShape: "4x9", evidence: "expected nn.tile rank-2 repeat/tile native Program evidence", outputEvidence: "compiled nn.tile rank-2 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6] },
    { name: "repeat", module: adapter.nn.repeat([2, 1, 2]), input: cube, inputShape: [1, 2, 3], expectedShape: "2x2x6", evidence: "expected nn.repeat rank-3 repeat/tile native Program evidence", outputEvidence: "compiled nn.repeat rank-3 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6] },
    { name: "tile", module: new adapter.nn.Tile([2, 1, 2]), input: cube, inputShape: [1, 2, 3], expectedShape: "2x2x6", evidence: "expected nn.tile rank-3 repeat/tile native Program evidence", outputEvidence: "compiled nn.tile rank-3 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6] },
    { name: "repeat", module: adapter.nn.repeat([1, 1, 2]), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x2x6", evidence: "expected nn.repeat batched rank-3 repeat/tile native Program evidence", outputEvidence: "compiled nn.repeat batched rank-3 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 10, 11, 12] },
    { name: "tile", module: new adapter.nn.Tile([1, 1, 2]), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x2x6", evidence: "expected nn.tile batched rank-3 repeat/tile native Program evidence", outputEvidence: "compiled nn.tile batched rank-3 repeat/tile output", expectedData: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 10, 11, 12] },
  ];
  for (const testCase of compiledRepeatCases) {
    const input = testCase.input ?? vector;
    const inputShape = testCase.inputShape ?? [3];
    const expectedShape = testCase.expectedShape ?? "6";
    const support = testCase.module.compileSupport({ inputShape, backend: "cpu" });
    const kernelPlan = adapter.nn.kernelPlan(testCase.module, { inputShape, backend: "cpu" });
    if (
      support.supported !== true ||
      support.outputShape.join("x") !== expectedShape ||
      kernelPlan.dispatchCount !== 1 ||
      kernelPlan.ops.length !== 1 ||
      kernelPlan.ops[0]?.op !== testCase.name ||
      kernelPlan.ops[0]?.nativeKernels.join("|") !== "broadcast"
    ) {
      throw new Error(`${label} ${testCase.evidence}`);
    }
    const program = testCase.module.compile({ inputShape, backend: "cpu" });
    const session = program.bind({});
    expectClose(session.stepTensor(input).data, testCase.expectedData, `${label} ${testCase.outputEvidence}`);
  }
  const compiledCases = [
    { name: "broadcastTo", module: adapter.nn.broadcastTo([2, 3]), input: row, inputShape: [1, 3], expectedShape: "2x3", expectedKernels: "broadcast", expectedDispatches: 1, expectedElided: 0 },
    { name: "expand", module: adapter.nn.expand([2, 3]), input: row, inputShape: [1, 3], expectedShape: "2x3", expectedKernels: "broadcast", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-row", module: adapter.nn.narrow(0, 1, 1), input: matrix, inputShape: [2, 3], expectedShape: "1x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-col", module: adapter.nn.narrow(1, 1, 2), input: matrix, inputShape: [2, 3], expectedShape: "2x2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-row", module: adapter.nn.select(0, 1), input: matrix, inputShape: [2, 3], expectedShape: "3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-col", module: adapter.nn.select(1, 1), input: matrix, inputShape: [2, 3], expectedShape: "2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "slice-row", module: adapter.nn.slice(0, 0, 2, 1), input: matrix, inputShape: [2, 3], expectedShape: "2x3", expectedKernels: "slice", expectedDispatches: 1, expectedElided: 0 },
    { name: "slice-col-step", module: adapter.nn.slice(1, 0, null, 2), input: matrix, inputShape: [2, 3], expectedShape: "2x2", expectedKernels: "slice", expectedDispatches: 1, expectedElided: 0 },
    { name: "flatten", module: adapter.nn.flatten(0, -1), input: matrix, inputShape: [2, 3], expectedShape: "6", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "squeeze", module: adapter.nn.squeeze(0), input: row, inputShape: [1, 3], expectedShape: "3", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "unsqueeze", module: adapter.nn.unsqueeze(0), input: vector, inputShape: [3], expectedShape: "1x3", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "reshape-rank3", module: adapter.nn.reshape([1, 2, 3]), input: matrix, inputShape: [2, 3], expectedShape: "1x2x3", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "flatten-rank3-range", module: adapter.nn.flatten(1, -1), input: cube, inputShape: [1, 2, 3], expectedShape: "1x6", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "squeeze-rank3", module: adapter.nn.squeeze(0), input: cube, inputShape: [1, 2, 3], expectedShape: "2x3", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "unsqueeze-rank3", module: adapter.nn.unsqueeze(0), input: matrix, inputShape: [2, 3], expectedShape: "1x2x3", expectedKernels: "", expectedDispatches: 0, expectedElided: 1 },
    { name: "broadcastTo-rank3", module: adapter.nn.broadcastTo([2, 2, 3]), input: cube, inputShape: [1, 2, 3], expectedShape: "2x2x3", expectedKernels: "broadcast", expectedDispatches: 1, expectedElided: 0 },
    { name: "expand-rank3", module: adapter.nn.expand([2, 2, 3]), input: cube, inputShape: [1, 2, 3], expectedShape: "2x2x3", expectedKernels: "broadcast", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-batch", module: adapter.nn.narrow(0, 0, 1), input: cube, inputShape: [1, 2, 3], expectedShape: "1x2x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-middle", module: adapter.nn.narrow(1, 0, 1), input: cube, inputShape: [1, 2, 3], expectedShape: "1x1x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-feature", module: adapter.nn.narrow(2, 1, 2), input: cube, inputShape: [1, 2, 3], expectedShape: "1x2x2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-rank3-batch", module: adapter.nn.select(0, 0), input: cube, inputShape: [1, 2, 3], expectedShape: "2x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-rank3-middle", module: adapter.nn.select(1, 0), input: cube, inputShape: [1, 2, 3], expectedShape: "1x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-rank3-feature", module: adapter.nn.select(2, 1), input: cube, inputShape: [1, 2, 3], expectedShape: "1x2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "slice-rank3-step", module: adapter.nn.slice(2, 0, null, 2), input: cube, inputShape: [1, 2, 3], expectedShape: "1x2x2", expectedKernels: "slice", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-batched-batch", module: adapter.nn.narrow(0, 1, 1), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "1x2x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-batched-middle", module: adapter.nn.narrow(1, 0, 1), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x1x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "narrow-rank3-batched-feature", module: adapter.nn.narrow(2, 1, 2), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x2x2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-rank3-batched-middle", module: adapter.nn.select(1, 0), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x3", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "select-rank3-batched-feature", module: adapter.nn.select(2, 1), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x2", expectedKernels: "narrow", expectedDispatches: 1, expectedElided: 0 },
    { name: "slice-rank3-batched-feature-step", module: adapter.nn.slice(2, 0, null, 2), input: batchedCube, inputShape: [2, 2, 3], expectedShape: "2x2x2", expectedKernels: "slice", expectedDispatches: 1, expectedElided: 0 },
  ];
  for (const testCase of compiledCases) {
    const eager = testCase.module.forward(testCase.input);
    const support = testCase.module.compileSupport({ inputShape: testCase.inputShape, backend: "cpu" });
    const kernelPlan = adapter.nn.kernelPlan(testCase.module, { inputShape: testCase.inputShape, backend: "cpu" });
    const evidenceOps = testCase.expectedDispatches === 0 ? kernelPlan.elidedOps : kernelPlan.ops;
    const evidenceOp = evidenceOps[0];
    if (
      !Object.isFrozen(support) ||
      !Object.isFrozen(kernelPlan) ||
      support.supported !== true ||
      support.outputShape.join("x") !== testCase.expectedShape ||
      support.weightsLen !== 0 ||
      support.biasLen !== 0 ||
      evidenceOps.length !== 1 ||
      evidenceOp.op !== testCase.module.kind ||
      (testCase.expectedDispatches > 0 && evidenceOp.nativeKernels.join("|") !== testCase.expectedKernels) ||
      kernelPlan.dispatchCount !== testCase.expectedDispatches ||
      kernelPlan.elidedOpCount !== testCase.expectedElided
    ) {
      throw new Error(`${label} expected ${testCase.name} shape/view compile evidence with rank-3 materialized output views`);
    }
    const program = testCase.module.compile({ inputShape: testCase.inputShape, backend: "cpu" });
    const session = program.bind({});
    try {
      const compiled = session.stepTensor(testCase.input);
      if (compiled.shape.join("x") !== testCase.expectedShape) {
        throw new Error(`${label} expected ${testCase.name} compiled shape`);
      }
      expectClose(compiled.data, eager.data, `${label} ${testCase.name} shape/view eager/compiled parity`);
    } finally {
      session.dispose();
      program.dispose();
    }
  }
}

function expectUnsupportedCompileExplanationEvidence(adapter: Record<string, any>, label: string) {
  const trainingDropout = adapter.nn.sequential([
    adapter.nn.dropout(0.5, { rng: () => 0.75 }),
    adapter.nn.relu(),
  ]);
  const support = trainingDropout.compileSupport({ inputShape: [2], backend: "cpu" });
  const explanation = adapter.nn.explain(trainingDropout, { inputShape: [2], backend: "cpu" });
  const preflight = adapter.nn.preflight(trainingDropout, { inputShape: [2], backend: "cpu" });
  const methodExplanation = trainingDropout.explain({ inputShape: [2], backend: "cpu" });
  const methodPreflight = trainingDropout.preflight({ inputShape: [2], backend: "cpu" });
  const compileExplanation = trainingDropout.compileExplanation({ inputShape: [2], backend: "cpu" });
  const namespaceCompileExplanation = adapter.nn.compileExplanation(trainingDropout, { inputShape: [2], backend: "cpu" });
  const compilePlan = adapter.nn.compilePlan(trainingDropout, { inputShape: [2], backend: "cpu" });
  const methodCompilePlan = trainingDropout.compilePlan({ inputShape: [2], backend: "cpu" });
  expectThrowIncludes(
    () => adapter.nn.requireCompilePlan(trainingDropout, { inputShape: [2], backend: "cpu" }),
    "nn.requireCompilePlan rejected unsupported module",
    `${label} nn.requireCompilePlan rejects unsupported dropout`,
  );
  expectThrowIncludes(
    () => trainingDropout.requireCompilePlan({ inputShape: [2], backend: "cpu" }),
    "nn.requireCompilePlan rejected unsupported module",
    `${label} module requireCompilePlan rejects unsupported dropout`,
  );
  const diagnostic = explanation.diagnostics[0];
  if (
    !Object.isFrozen(explanation) ||
    !Object.isFrozen(preflight) ||
    !Object.isFrozen(methodExplanation) ||
    !Object.isFrozen(methodPreflight) ||
    !Object.isFrozen(compileExplanation) ||
    !Object.isFrozen(namespaceCompileExplanation) ||
    !Object.isFrozen(compilePlan) ||
    !Object.isFrozen(methodCompilePlan) ||
    !Object.isFrozen(explanation.diagnostics) ||
    !Object.isFrozen(diagnostic) ||
    !Object.isFrozen(explanation.parameterNames) ||
    !Object.isFrozen(explanation.parameterInfos) ||
    !Object.isFrozen(explanation.trace) ||
    !Object.isFrozen(explanation.ir) ||
    !Object.isFrozen(explanation.ir.ops[0].attrs) ||
    explanation.kind !== "zgml.nn.compile-explanation" ||
    preflight.kind !== "zgml.nn.compile-explanation" ||
    methodExplanation.kind !== "zgml.nn.compile-explanation" ||
    methodPreflight.kind !== "zgml.nn.compile-explanation" ||
    compileExplanation.kind !== "zgml.nn.compile-explanation" ||
    namespaceCompileExplanation.kind !== "zgml.nn.compile-explanation" ||
    compilePlan.kind !== "zgml.nn.compile-explanation" ||
    methodCompilePlan.kind !== "zgml.nn.compile-explanation" ||
    explanation.supported !== false ||
    preflight.supported !== false ||
    methodExplanation.supported !== false ||
    methodPreflight.supported !== false ||
    compileExplanation.supported !== false ||
    namespaceCompileExplanation.supported !== false ||
    compilePlan.supported !== false ||
    methodCompilePlan.supported !== false ||
    typeof explanation.signature !== "string" ||
    !explanation.signature.startsWith("nn-compile-explanation|supported=0|native=device-program|model=module|inputShape=2|outputShape=2|") ||
    !explanation.signature.includes("diagnostics=kernelizer:unsupported-op:dropout:0") ||
    preflight.signature !== explanation.signature ||
    namespaceCompileExplanation.signature !== explanation.signature ||
    compilePlan.signature !== explanation.signature ||
    methodCompilePlan.signature !== explanation.signature ||
    methodPreflight.signature !== explanation.signature ||
    explanation.reason !== support.reason ||
    preflight.reason !== support.reason ||
    methodExplanation.reason !== support.reason ||
    methodPreflight.reason !== support.reason ||
    compileExplanation.reason !== support.reason ||
    namespaceCompileExplanation.reason !== support.reason ||
    compilePlan.reason !== support.reason ||
    methodCompilePlan.reason !== support.reason ||
    explanation.nativePath !== "device-program" ||
    explanation.modelKind !== "module" ||
    explanation.inputShape.join("x") !== "2" ||
    explanation.outputShape.join("x") !== "2" ||
    explanation.parameterNames.length !== 0 ||
    explanation.parameterInfos.length !== 0 ||
    explanation.parameterCount !== 0 ||
    explanation.parameterScalarCount !== 0 ||
    explanation.trace.opCount !== 2 ||
    explanation.ir.opCount !== 2 ||
    explanation.ir.ops[0].op !== "dropout" ||
    preflight.ir.ops[0].op !== "dropout" ||
    methodExplanation.ir.ops[0].op !== "dropout" ||
    methodPreflight.ir.ops[0].op !== "dropout" ||
    compileExplanation.ir.ops[0].op !== "dropout" ||
    compilePlan.ir.ops[0].op !== "dropout" ||
    methodCompilePlan.ir.ops[0].op !== "dropout" ||
    explanation.ir.ops[0].attrs.training !== true ||
    explanation.kernelPlan !== null ||
    preflight.kernelPlan !== null ||
    methodExplanation.kernelPlan !== null ||
    methodPreflight.kernelPlan !== null ||
    compileExplanation.kernelPlan !== null ||
    compilePlan.kernelPlan !== null ||
    methodCompilePlan.kernelPlan !== null ||
    explanation.bufferLayout !== null ||
    explanation.memoryLayout !== null ||
    explanation.shapeConstraints !== null ||
    explanation.parameterLayout !== null ||
    explanation.compilerSignatures.ir !== support.irSignature ||
    diagnostic.stage !== "kernelizer" ||
    diagnostic.code !== "unsupported-op" ||
    diagnostic.op !== "dropout" ||
    diagnostic.path !== "0"
  ) {
    throw new Error(`${label} expected unsupported dropout nn.explain evidence with partial IR and no KernelPlan; expected unsupported compile evidence`);
  }
  expectThrowIncludes(
    () => trainingDropout.compile({ inputShape: [2], backend: "cpu" }),
    "native module Program compiler can only lower deterministic no-op nn.Dropout",
    `${label} training dropout compile rejection`,
  );
}

function expectClassifierAndTokenHeadProgramEvidence(adapter: Record<string, any>, label: string) {
  const classifier = adapter.nn.sequential([
    adapter.nn.linear(3, 2, { weights: [1, 0, 0, 0, 1, 1], bias: [0, 0] }),
    adapter.nn.logSoftmax(-1),
  ]);
  const classifierInput = adapter.tensor([[1, 2, 3]], [1, 3]);
  const classifierEager = classifier.forward(classifierInput);
  const classifierSupport = classifier.compileSupport({ inputShape: [1, 3], backend: "cpu" });
  const classifierPlan = adapter.nn.kernelPlan(classifier, { inputShape: [1, 3], backend: "cpu" });
  if (
    !Object.isFrozen(classifierSupport) ||
    !Object.isFrozen(classifierPlan) ||
    classifierSupport.supported !== true ||
    classifierSupport.outputShape.join("x") !== "1x2" ||
    classifierSupport.trace.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear|logSoftmax" ||
    classifierPlan.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear|logSoftmax" ||
    classifierPlan.dispatchCount !== 2
  ) {
    throw new Error(`${label} expected Linear/LogSoftmax classifier Program evidence`);
  }
  const classifierProgram = classifier.compile({ inputShape: [1, 3], backend: "cpu" });
  const classifierSession = classifierProgram.bindModule(classifier);
  try {
    const compiled = classifierSession.stepTensor(classifierInput);
    if (compiled.shape.join("x") !== "1x2") throw new Error(`${label} expected classifier compiled output shape`);
    expectClose(compiled.data, classifierEager.data, `${label} classifier eager/compiled parity`);
  } finally {
    classifierSession.dispose();
    classifierProgram.dispose();
  }

  const batchedLogSoftmaxClassifier = adapter.nn.sequential([
    adapter.nn.linear(3, 2, { weights: [1, 0, 0, 0, 1, 1], bias: [0.1, -0.1] }),
    adapter.nn.logSoftmax(-1),
  ]);
  const batchedLogSoftmaxInput = adapter.tensor([[1, 2, 3], [2, 1, 0], [-1, 0, 1], [3, -2, 1]], [4, 3]);
  const batchedLogSoftmaxEager = batchedLogSoftmaxClassifier.forward(batchedLogSoftmaxInput);
  const batchedLogSoftmaxSupport = batchedLogSoftmaxClassifier.compileSupport({ inputShape: [4, 3], backend: "cpu" });
  const batchedLogSoftmaxPlan = adapter.nn.kernelPlan(batchedLogSoftmaxClassifier, { inputShape: [4, 3], backend: "cpu" });
  if (
    batchedLogSoftmaxSupport.supported !== true ||
    batchedLogSoftmaxSupport.outputShape.join("x") !== "4x2" ||
    batchedLogSoftmaxSupport.trace.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear|logSoftmax" ||
    batchedLogSoftmaxPlan?.dispatchCount !== 2 ||
    batchedLogSoftmaxPlan.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|log-softmax"
  ) {
    throw new Error(`${label} expected batched Linear/LogSoftmax classifier Program evidence`);
  }
  const batchedLogSoftmaxProgram = batchedLogSoftmaxClassifier.compile({ inputShape: [4, 3], backend: "cpu" });
  const batchedLogSoftmaxSession = batchedLogSoftmaxProgram.bindModule(batchedLogSoftmaxClassifier);
  try {
    const batchedLogSoftmaxOutput = new Float32Array(8);
    const batchedLogSoftmaxCompiled = batchedLogSoftmaxSession.executeInto(batchedLogSoftmaxOutput, { input: batchedLogSoftmaxInput });
    if (batchedLogSoftmaxCompiled !== batchedLogSoftmaxOutput) throw new Error(`${label} expected batched log-softmax classifier executeInto to reuse caller output`);
    expectClose(batchedLogSoftmaxCompiled, batchedLogSoftmaxEager.data, `${label} batched log-softmax classifier eager/compiled parity`);
  } finally {
    batchedLogSoftmaxSession.dispose();
    batchedLogSoftmaxProgram.dispose();
  }

  const batchedSoftmaxClassifier = adapter.nn.sequential([
    adapter.nn.linear(3, 3, { weights: [1, 0, 0, 0, 1, 0, 0, 0, 1], bias: [0.1, -0.2, 0.3] }),
    adapter.nn.softmax(-1),
    adapter.nn.linear(3, 2, { weights: [1, 0, 0, 1, 1, -1], bias: [0.05, -0.05] }),
  ]);
  const batchedClassifierInput = adapter.tensor([[1, 2, 3], [2, 1, 0], [-1, 0, 1], [3, -2, 1]], [4, 3]);
  const batchedClassifierEager = batchedSoftmaxClassifier.forward(batchedClassifierInput);
  const batchedClassifierSupport = batchedSoftmaxClassifier.compileSupport({ inputShape: [4, 3], backend: "cpu" });
  const batchedClassifierPlan = adapter.nn.kernelPlan(batchedSoftmaxClassifier, { inputShape: [4, 3], backend: "cpu" });
  if (
    batchedClassifierSupport.supported !== true ||
    batchedClassifierSupport.outputShape.join("x") !== "4x2" ||
    batchedClassifierSupport.trace.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear|softmax|linear" ||
    batchedClassifierPlan?.dispatchCount !== 3 ||
    batchedClassifierPlan.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|softmax|linear"
  ) {
    throw new Error(`${label} expected batched Linear/Softmax/Linear classifier Program evidence`);
  }
  const batchedClassifierProgram = batchedSoftmaxClassifier.compile({ inputShape: [4, 3], backend: "cpu" });
  const batchedClassifierSession = batchedClassifierProgram.bindModule(batchedSoftmaxClassifier);
  try {
    const batchedClassifierOutput = new Float32Array(8);
    const batchedClassifierCompiled = batchedClassifierSession.executeInto(batchedClassifierOutput, { input: batchedClassifierInput });
    if (batchedClassifierCompiled !== batchedClassifierOutput) throw new Error(`${label} expected batched softmax classifier executeInto to reuse caller output`);
    expectClose(batchedClassifierCompiled, batchedClassifierEager.data, `${label} batched softmax classifier eager/compiled parity`);
  } finally {
    batchedClassifierSession.dispose();
    batchedClassifierProgram.dispose();
  }

  const tokenHead = adapter.nn.sequential([
    adapter.nn.embedding(4, 3, { weight: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]] }),
    adapter.nn.linear(3, 2, { weights: [1, 0, 0, 0, 1, 1], bias: [0, 0] }),
    adapter.nn.logSoftmax(-1),
  ]);
  const tokenInput = adapter.tensor([1, 2], [2]);
  const tokenEager = tokenHead.forward(tokenInput);
  const tokenSupport = tokenHead.compileSupport({ inputShape: [2], backend: "cpu" });
  const tokenPlan = adapter.nn.kernelPlan(tokenHead, { inputShape: [2], backend: "cpu" });
  if (
    !Object.isFrozen(tokenSupport) ||
    !Object.isFrozen(tokenPlan) ||
    tokenSupport.supported !== true ||
    tokenSupport.outputShape.join("x") !== "2x2" ||
    tokenSupport.trace.ops.map((op: Record<string, any>) => op.op).join("|") !== "embedding|linear|logSoftmax" ||
    tokenPlan.ops.map((op: Record<string, any>) => op.op).join("|") !== "embedding|linear|logSoftmax" ||
    tokenPlan.dispatchCount !== 3 ||
    tokenHead.parameterNames().join("|") !== "0.weight|1.weight|1.bias"
  ) {
    throw new Error(`${label} expected Embedding/Linear/LogSoftmax token-head Program evidence`);
  }
  const tokenProgram = tokenHead.compile({ inputShape: [2], backend: "cpu" });
  const tokenSession = tokenProgram.bindModule(tokenHead);
  try {
    const compiled = tokenSession.stepTensor(tokenInput);
    if (compiled.shape.join("x") !== "2x2") throw new Error(`${label} expected token-head compiled output shape`);
    expectClose(compiled.data, tokenEager.data, `${label} token-head eager/compiled parity`);
  } finally {
    tokenSession.dispose();
    tokenProgram.dispose();
  }
}

function expectTorchNamespaceEndToEndEvidence(adapter: Record<string, any>, label: string) {
  const torch = adapter.torch;
  if (adapter.zgml !== torch) {
    throw new Error(`${label} expected zgml to be the canonical alias for the torch-compatible namespace`);
  }
  if (
    !torch ||
    typeof torch.tensor !== "function" ||
    typeof torch.as_tensor !== "function" ||
    typeof torch.asTensor !== "function" ||
    typeof torch.asarray !== "function" ||
    typeof torch.from_numpy !== "function" ||
    typeof torch.fromNumpy !== "function" ||
    typeof torch.train?.fit !== "function" ||
    typeof torch.checkpoint?.create !== "function" ||
    typeof torch.save !== "function" ||
    typeof torch.load !== "function" ||
    torch.functional !== torch.F ||
    torch.functional !== torch.nn.functional ||
    typeof torch.functional?.mse !== "function" ||
    typeof torch.compile !== "function" ||
    typeof torch.compile?.compile !== "function" ||
    typeof torch.native !== "function" ||
    torch.native !== torch.compile.compileForInference ||
    typeof torch.inference !== "function" ||
    torch.inference !== torch.compile.compileForInference ||
    typeof torch.compileInference !== "function" ||
    torch.compileInference !== torch.compile.compileForInference ||
    typeof torch.compile_inference !== "function" ||
    torch.compile_inference !== torch.compile.compileForInference ||
    typeof torch.compileForInference !== "function" ||
    torch.compileForInference !== torch.compile.compileForInference ||
    typeof torch.compile_for_inference !== "function" ||
    torch.compile_for_inference !== torch.compile.compile_for_inference ||
    typeof torch.lazy?.input !== "function" ||
    typeof torch.manual_seed !== "function" ||
    typeof torch.initial_seed !== "function" ||
    typeof torch.initialSeed !== "function" ||
    typeof torch.empty !== "function" ||
    typeof torch.empty_like !== "function" ||
    typeof torch.emptyLike !== "function" ||
    typeof torch.zeros_like !== "function" ||
    typeof torch.ones_like !== "function" ||
    typeof torch.full_like !== "function" ||
    typeof torch.rand_like !== "function" ||
    typeof torch.randn_like !== "function" ||
    typeof torch.clone !== "function" ||
    typeof torch.detach !== "function" ||
    typeof torch.relu !== "function" ||
    typeof torch.softmax !== "function" ||
    typeof torch.to !== "function" ||
    typeof torch.cpu !== "function" ||
    typeof torch.float !== "function" ||
    typeof torch.float32 !== "function" ||
    typeof torch.typeAs !== "function" ||
    typeof torch.type_as !== "function" ||
    typeof torch.sum !== "function" ||
    typeof torch.prod !== "function" ||
    typeof torch.cumsum !== "function" ||
    typeof torch.any !== "function" ||
    typeof torch.all !== "function" ||
    typeof torch.logsumexp !== "function" ||
    typeof torch.logSumExp !== "function" ||
    typeof torch.variance !== "function" ||
    typeof torch.var !== "function" ||
    typeof torch.std !== "function" ||
    typeof torch.norm !== "function" ||
    typeof torch.neg !== "function" ||
    typeof torch.negative !== "function" ||
    typeof torch.expm1 !== "function" ||
    typeof torch.log1p !== "function" ||
    typeof torch.sqr !== "function" ||
    typeof torch.square !== "function" ||
    typeof torch.recip !== "function" ||
    typeof torch.reciprocal !== "function" ||
    typeof torch.sgn !== "function" ||
    typeof torch.sign !== "function" ||
    typeof torch.step !== "function" ||
    typeof torch.isnan !== "function" ||
    typeof torch.isinf !== "function" ||
    typeof torch.isfinite !== "function" ||
    typeof torch.floor !== "function" ||
    typeof torch.ceil !== "function" ||
    typeof torch.round !== "function" ||
    typeof torch.trunc !== "function" ||
    typeof torch.sin !== "function" ||
    typeof torch.cos !== "function" ||
    typeof torch.tan !== "function" ||
    typeof torch.softmax_dim !== "function" ||
    typeof torch.softmaxDim !== "function" ||
    typeof torch.logSoftmax !== "function" ||
    typeof torch.log_softmax_dim !== "function" ||
    typeof torch.logSoftmaxDim !== "function" ||
    typeof torch.rsqrt !== "function" ||
    typeof torch.clip !== "function" ||
    typeof torch.scalar !== "function" ||
    typeof torch.add !== "function" ||
    typeof torch.matmul !== "function" ||
    typeof torch.cat !== "function" ||
    typeof torch.concat !== "function" ||
    typeof torch.concatenate !== "function" ||
    typeof torch.stack !== "function" ||
    typeof torch.vstack !== "function" ||
    typeof torch.broadcastTo !== "function" ||
    typeof torch.scatter_add !== "function" ||
    typeof torch.reshape !== "function" ||
    typeof torch.flip !== "function" ||
    typeof torch.roll !== "function" ||
    typeof torch.select !== "function" ||
    typeof torch.narrow !== "function" ||
    typeof torch.slice !== "function" ||
    typeof torch.index_select !== "function" ||
    typeof torch.gather !== "function" ||
    typeof torch.take !== "function" ||
    typeof torch.unbind !== "function" ||
    typeof torch.where !== "function" ||
    typeof torch.maskedFill !== "function" ||
    typeof torch.masked_fill !== "function" ||
    typeof torch.allclose !== "function" ||
    typeof torch.topk !== "function" ||
    typeof torch.hasShape !== "function" ||
    typeof torch.requireShape !== "function" ||
    typeof torch.utils?.data?.Dataset !== "function" ||
    typeof torch.utils?.data?.TensorDataset !== "function" ||
    typeof torch.utils?.data?.DataLoader !== "function" ||
    typeof torch.utils?.data?.Subset !== "function" ||
    typeof torch.utils?.data?.ConcatDataset !== "function" ||
    typeof torch.utils?.data?.MapDataset !== "function" ||
    typeof torch.utils?.data?.SequentialSampler !== "function" ||
    typeof torch.utils?.data?.RandomSampler !== "function" ||
    typeof torch.utils?.data?.BatchSampler !== "function" ||
    typeof torch.utils?.data?.randomSplit !== "function" ||
    typeof torch.utils?.data?.random_split !== "function"
  ) {
    throw new Error(`${label} expected torch namespace to expose tensor, train, checkpoint, save/load, compile, lazy, data, tensor convenience helpers, and root tensor ops`);
  }

  const base = torch.tensor([1, 2], [2]);
  torch.manual_seed(515);
  if (torch.initial_seed() !== 515 || torch.initialSeed() !== 515) {
    throw new Error(`${label} expected torch manual_seed/initial_seed aliases to share the root RNG state`);
  }
  expectClose(torch.as_tensor(new Float32Array([1, 2]), [2]).data, [1, 2], `${label} torch.as_tensor tensor factory alias`);
  expectClose(torch.asTensor([1, 2], [2]).data, [1, 2], `${label} torch.asTensor tensor factory alias`);
  expectClose(torch.asarray([1, 2], [2]).data, [1, 2], `${label} torch.asarray tensor factory alias`);
  expectClose(torch.from_numpy(new Float32Array([1, 2]), [2]).data, [1, 2], `${label} torch.from_numpy tensor factory alias`);
  expectClose(torch.fromNumpy([1, 2], [2]).data, [1, 2], `${label} torch.fromNumpy tensor factory alias`);
  expectClose(torch.scalar(7).data, [7], `${label} torch.scalar root factory`);
  expectClose(torch.empty([2]).data, [0, 0], `${label} torch.empty root factory zero-backed JS allocation`);
  expectClose(torch.functional.mse(base, torch.clone(base)).data, [0], `${label} torch.functional root alias`);
  if (
    torch.empty_like(base).data.join(",") !== "0,0" ||
    torch.emptyLike(base).shape.join("x") !== "2" ||
    torch.zeros_like(base).data.join(",") !== "0,0" ||
    torch.zerosLike(base).data.join(",") !== "0,0" ||
    torch.ones_like(base).data.join(",") !== "1,1" ||
    torch.onesLike(base).data.join(",") !== "1,1" ||
    torch.full_like(base, 3).data.join(",") !== "3,3" ||
    torch.fullLike(base, 4).data.join(",") !== "4,4" ||
    torch.rand_like(base, { seed: 7 }).shape.join("x") !== "2" ||
    torch.randLike(base, { seed: 7 }).shape.join("x") !== "2" ||
    torch.randn_like(base, { seed: 7 }).shape.join("x") !== "2" ||
    torch.randnLike(base, { seed: 7 }).shape.join("x") !== "2" ||
    torch.clone(base).data.join(",") !== "1,2" ||
    torch.detach(base).data.join(",") !== "1,2" ||
    torch.hasShape(base, [2]) !== true ||
    torch.requireShape(base, [2]) !== base
  ) {
    throw new Error(`${label} expected torch tensor convenience helpers to preserve familiar Tensor shape behavior`);
  }
  const signed = torch.tensor([-1, 2], [2]);
  const matrix = torch.tensor([1, 2, 3, 4], [2, 2]);
  expectClose(torch.relu(signed).data, signed.relu().data, `${label} torch.relu root op`);
  expectClose(torch.gelu(signed).data, signed.gelu().data, `${label} torch.gelu root op`);
  expectClose(torch.silu(signed).data, signed.silu().data, `${label} torch.silu root op`);
  expectClose(torch.sigmoid(signed).data, signed.sigmoid().data, `${label} torch.sigmoid root op`);
  expectClose(torch.tanh(signed).data, signed.tanh().data, `${label} torch.tanh root op`);
  expectClose(torch.softmax(signed, 0).data, signed.softmax(0).data, `${label} torch.softmax root op`);
  expectClose(torch.softmax_dim(signed, 0).data, signed.softmax_dim(0).data, `${label} torch.softmax_dim root op`);
  expectClose(torch.softmaxDim(signed, 0).data, signed.softmaxDim(0).data, `${label} torch.softmaxDim root op`);
  expectClose(torch.logSoftmax(signed, 0).data, signed.logSoftmax(0).data, `${label} torch.logSoftmax root op`);
  expectClose(torch.log_softmax(signed, 0).data, signed.log_softmax(0).data, `${label} torch.log_softmax root op`);
  expectClose(torch.log_softmax_dim(signed, 0).data, signed.log_softmax_dim(0).data, `${label} torch.log_softmax_dim root op`);
  expectClose(torch.logSoftmaxDim(signed, 0).data, signed.logSoftmaxDim(0).data, `${label} torch.logSoftmaxDim root op`);
  if (
    torch.to(base, "cpu") !== base ||
    torch.cpu(base) !== base ||
    torch.float(base) !== base ||
    torch.float32(base) !== base ||
    torch.typeAs(base, base) !== base ||
    torch.type_as(base, base) !== base
  ) {
    throw new Error(`${label} expected torch placement/dtype root helpers to preserve eager cpu f32 tensor identity`);
  }
  expectClose(torch.to(base, { device: "cpu", copy: true }).data, base.data, `${label} torch.to copy root op`);
  expectClose(torch.sum(matrix, 0).data, matrix.sum(0).data, `${label} torch.sum root op`);
  expectClose(torch.prod(matrix, 1).data, matrix.prod(1).data, `${label} torch.prod root op`);
  expectClose(torch.cumsum(matrix, 1).data, matrix.cumsum(1).data, `${label} torch.cumsum root op`);
  expectClose(torch.mean(matrix, 1).data, matrix.mean(1).data, `${label} torch.mean root op`);
  expectClose(torch.any(matrix.gt(2)).data, matrix.gt(2).any().data, `${label} torch.any root op`);
  expectClose(torch.all(matrix.gt(0)).data, matrix.gt(0).all().data, `${label} torch.all root op`);
  expectClose(torch.logsumexp(base).data, base.logsumexp().data, `${label} torch.logsumexp root op`);
  expectClose(torch.logSumExp(base).data, base.logSumExp().data, `${label} torch.logSumExp root op`);
  expectClose(torch.max(matrix, 0).data, matrix.max(0).data, `${label} torch.max root op`);
  expectClose(torch.min(matrix, 1).data, matrix.min(1).data, `${label} torch.min root op`);
  expectClose(torch.argmax(matrix, 1).data, matrix.argmax(1).data, `${label} torch.argmax root op`);
  expectClose(torch.argmin(matrix, 0).data, matrix.argmin(0).data, `${label} torch.argmin root op`);
  expectClose(torch.variance(matrix, 1).data, matrix.variance(1).data, `${label} torch.variance root op`);
  expectClose(torch.var(matrix, 1).data, matrix.var(1).data, `${label} torch.var root op`);
  expectClose(torch.std(matrix, 1).data, matrix.std(1).data, `${label} torch.std root op`);
  expectClose(torch.norm(matrix, 1).data, matrix.norm(1).data, `${label} torch.norm root op`);
  expectClose(torch.neg(signed).data, signed.neg().data, `${label} torch.neg root op`);
  expectClose(torch.negative(signed).data, signed.negative().data, `${label} torch.negative root op`);
  expectClose(torch.expm1(signed).data, signed.expm1().data, `${label} torch.expm1 root op`);
  expectClose(torch.log1p(base).data, base.log1p().data, `${label} torch.log1p root op`);
  expectClose(torch.sqr(signed).data, signed.sqr().data, `${label} torch.sqr root op`);
  expectClose(torch.square(signed).data, signed.square().data, `${label} torch.square root op`);
  expectClose(torch.recip(base).data, base.recip().data, `${label} torch.recip root op`);
  expectClose(torch.reciprocal(base).data, base.reciprocal().data, `${label} torch.reciprocal root op`);
  expectClose(torch.sgn(signed).data, signed.sgn().data, `${label} torch.sgn root op`);
  expectClose(torch.sign(signed).data, signed.sign().data, `${label} torch.sign root op`);
  expectClose(torch.step(signed).data, signed.step().data, `${label} torch.step root op`);
  expectClose(torch.isnan(torch.tensor([Number.NaN, 1], [2])).data, torch.tensor([Number.NaN, 1], [2]).isnan().data, `${label} torch.isnan root op`);
  expectClose(torch.isinf(torch.tensor([Infinity, 1], [2])).data, torch.tensor([Infinity, 1], [2]).isinf().data, `${label} torch.isinf root op`);
  expectClose(torch.isfinite(torch.tensor([Infinity, 1], [2])).data, torch.tensor([Infinity, 1], [2]).isfinite().data, `${label} torch.isfinite root op`);
  expectClose(torch.floor(torch.tensor([-1.5, 1.5], [2])).data, torch.tensor([-1.5, 1.5], [2]).floor().data, `${label} torch.floor root op`);
  expectClose(torch.ceil(torch.tensor([-1.5, 1.5], [2])).data, torch.tensor([-1.5, 1.5], [2]).ceil().data, `${label} torch.ceil root op`);
  expectClose(torch.round(torch.tensor([-1.4, 1.6], [2])).data, torch.tensor([-1.4, 1.6], [2]).round().data, `${label} torch.round root op`);
  expectClose(torch.trunc(torch.tensor([-1.5, 1.5], [2])).data, torch.tensor([-1.5, 1.5], [2]).trunc().data, `${label} torch.trunc root op`);
  expectClose(torch.sin(signed).data, signed.sin().data, `${label} torch.sin root op`);
  expectClose(torch.cos(signed).data, signed.cos().data, `${label} torch.cos root op`);
  expectClose(torch.tan(signed).data, signed.tan().data, `${label} torch.tan root op`);
  expectClose(torch.sqrt(base).data, base.sqrt().data, `${label} torch.sqrt root op`);
  expectClose(torch.rsqrt(base).data, base.rsqrt().data, `${label} torch.rsqrt root op`);
  expectClose(torch.exp(signed).data, signed.exp().data, `${label} torch.exp root op`);
  expectClose(torch.log(base).data, base.log().data, `${label} torch.log root op`);
  expectClose(torch.abs(signed).data, signed.abs().data, `${label} torch.abs root op`);
  expectClose(torch.pow(base, 2).data, base.pow(2).data, `${label} torch.pow root op`);
  expectClose(torch.clamp(signed, 0, 1).data, signed.clamp(0, 1).data, `${label} torch.clamp root op`);
  expectClose(torch.clip(signed, 0, 1).data, signed.clip(0, 1).data, `${label} torch.clip root op`);
  expectClose(torch.flatten(matrix).data, matrix.flatten().data, `${label} torch.flatten root op`);
  expectClose(torch.reshape(signed, [2, 1]).data, signed.reshape([2, 1]).data, `${label} torch.reshape root op`);
  expectClose(torch.view(signed, [2, 1]).data, signed.view([2, 1]).data, `${label} torch.view root op`);
  expectClose(torch.flip(matrix, [1]).data, matrix.flip([1]).data, `${label} torch.flip root op`);
  expectClose(torch.roll(matrix, 1, 0).data, matrix.roll(1, 0).data, `${label} torch.roll root op`);
  expectClose(torch.select(matrix, 0, 1).data, matrix.select(0, 1).data, `${label} torch.select root op`);
  expectClose(torch.narrow(matrix, 1, 0, 1).data, matrix.narrow(1, 0, 1).data, `${label} torch.narrow root op`);
  expectClose(torch.slice(matrix, 1, 0, 1).data, matrix.slice(1, 0, 1).data, `${label} torch.slice root op`);
  expectClose(torch.index_select(matrix, 0, torch.tensor([1, 0], [2])).data, matrix.index_select(0, torch.tensor([1, 0], [2])).data, `${label} torch.index_select root op`);
  expectClose(torch.gather(matrix, 1, torch.tensor([0, 1, 1, 0], [2, 2])).data, matrix.gather(1, torch.tensor([0, 1, 1, 0], [2, 2])).data, `${label} torch.gather root op`);
  expectClose(torch.take(matrix, torch.tensor([0, 3], [2])).data, matrix.take(torch.tensor([0, 3], [2])).data, `${label} torch.take root op`);
  const unbound = torch.unbind(matrix, 0);
  const expectedUnbound = matrix.unbind(0);
  if (unbound.length !== expectedUnbound.length) {
    throw new Error(`${label} expected torch.unbind root op to preserve part count`);
  }
  for (let index = 0; index < unbound.length; index += 1) {
    expectClose(unbound[index].data, expectedUnbound[index].data, `${label} torch.unbind root op ${index}`);
  }
  expectClose(torch.broadcastTo(torch.reshape(signed, [2, 1]), [2, 2]).data, torch.reshape(signed, [2, 1]).broadcastTo([2, 2]).data, `${label} torch.broadcastTo root op`);
  expectClose(torch.expand(torch.reshape(signed, [2, 1]), [2, 2]).data, torch.reshape(signed, [2, 1]).expand([2, 2]).data, `${label} torch.expand root op`);
  expectClose(torch.repeat(signed, [2]).data, signed.repeat([2]).data, `${label} torch.repeat root op`);
  expectClose(torch.tile(signed, [2]).data, signed.tile([2]).data, `${label} torch.tile root op`);
  expectClose(torch.squeeze(torch.reshape(signed, [2, 1]), 1).data, torch.reshape(signed, [2, 1]).squeeze(1).data, `${label} torch.squeeze root op`);
  expectClose(torch.unsqueeze(signed, 0).data, signed.unsqueeze(0).data, `${label} torch.unsqueeze root op`);
  expectClose(torch.transpose(matrix).data, matrix.transpose().data, `${label} torch.transpose root op`);
  expectClose(torch.permute(matrix, [1, 0]).data, matrix.permute([1, 0]).data, `${label} torch.permute root op`);
  expectClose(torch.add(signed, 1).data, signed.add(1).data, `${label} torch.add root op`);
  expectClose(torch.sub(signed, 1).data, signed.sub(1).data, `${label} torch.sub root op`);
  expectClose(torch.mul(signed, 2).data, signed.mul(2).data, `${label} torch.mul root op`);
  expectClose(torch.div(signed, 2).data, signed.div(2).data, `${label} torch.div root op`);
  expectClose(torch.eq(signed, torch.clone(signed)).data, signed.eq(torch.clone(signed)).data, `${label} torch.eq root op`);
  expectClose(torch.ne(signed, torch.zeros_like(signed)).data, signed.ne(torch.zeros_like(signed)).data, `${label} torch.ne root op`);
  expectClose(torch.lt(signed, 1).data, signed.lt(1).data, `${label} torch.lt root op`);
  expectClose(torch.le(signed, 2).data, signed.le(2).data, `${label} torch.le root op`);
  expectClose(torch.gt(signed, 0).data, signed.gt(0).data, `${label} torch.gt root op`);
  expectClose(torch.ge(signed, -1).data, signed.ge(-1).data, `${label} torch.ge root op`);
  expectClose(torch.isclose(signed, torch.clone(signed)).data, signed.isclose(torch.clone(signed)).data, `${label} torch.isclose root op`);
  expectClose(torch.maximum(signed, 1).data, signed.maximum(1).data, `${label} torch.maximum root op`);
  expectClose(torch.minimum(signed, 1).data, signed.minimum(1).data, `${label} torch.minimum root op`);
  expectClose(torch.where(signed, base, torch.zeros_like(base)).data, signed.where(base, torch.zeros_like(base)).data, `${label} torch.where root op`);
  expectClose(torch.maskedFill(signed, signed.gt(0), 0).data, signed.maskedFill(signed.gt(0), 0).data, `${label} torch.maskedFill root op`);
  expectClose(torch.masked_fill(signed, signed.gt(0), 0).data, signed.masked_fill(signed.gt(0), 0).data, `${label} torch.masked_fill root op`);
  if (torch.allclose(base, torch.clone(base)) !== true || torch.equal(base, torch.clone(base)) !== true) {
    throw new Error(`${label} expected torch.allclose/equal root ops to compare tensors`);
  }
  expectClose(torch.argsort(signed).data, signed.argsort().data, `${label} torch.argsort root op`);
  expectClose(torch.sort(signed).values.data, signed.sort().values.data, `${label} torch.sort root op`);
  expectClose(torch.topk(base, 1).values.data, base.topk(1).values.data, `${label} torch.topk root op`);
  if (torch.split(base, 1).length !== 2 || torch.chunk(base, 2).length !== 2) {
    throw new Error(`${label} expected torch.split/chunk root ops to return tensor arrays`);
  }
  expectClose(torch.cat([base, base]).data, [1, 2, 1, 2], `${label} torch.cat root op`);
  expectClose(torch.concat([base, base]).data, [1, 2, 1, 2], `${label} torch.concat root op`);
  expectClose(torch.concatenate([base, base]).data, [1, 2, 1, 2], `${label} torch.concatenate root op`);
  expectClose(torch.stack([base, base]).data, [1, 2, 1, 2], `${label} torch.stack root op`);
  if (torch.stack([base, base]).shape.join("x") !== "2x2") throw new Error(`${label} expected torch.stack root op to add a batch axis`);
  expectClose(torch.vstack([base, base]).data, [1, 2, 1, 2], `${label} torch.vstack root op`);
  expectClose(torch.hstack([base, base]).data, [1, 2, 1, 2], `${label} torch.hstack root op`);
  expectClose(torch.matmul(matrix, torch.eye(2)).data, matrix.data, `${label} torch.matmul root op`);
  expectClose(torch.mm(matrix, torch.eye(2)).data, matrix.data, `${label} torch.mm root op`);
  expectClose(torch.dot(base, base).data, [5], `${label} torch.dot root op`);
  expectClose(torch.trace(matrix).data, [5], `${label} torch.trace root op`);
  expectClose(torch.diagonal(matrix).data, [1, 4], `${label} torch.diagonal root op`);
  expectClose(torch.scatter_add(torch.zeros_like(base), 0, torch.tensor([0, 1], [2]), torch.ones_like(base)).data, [1, 1], `${label} torch.scatter_add root op`);

  const model = new torch.nn.Sequential(
    new torch.nn.Linear(2, 3, {
      weights: [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
      bias: [0, 0, 0],
    }),
    new torch.nn.ReLU(),
    new torch.nn.Linear(3, 1, {
      weights: [0.2, -0.1, 0.05],
      bias: [0],
    }),
  );
  const dataset = new torch.utils.data.TensorDataset(
    torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
    torch.tensor([0, 1, 1, 0], [4, 1]),
  );
  const loader = new torch.utils.data.DataLoader(dataset, { batch_size: 2, shuffle: false });
  const optimizer = new torch.optim.AdamW(model, { lr: 0.05, weight_decay: 0.001 });
  const input = torch.tensor([1, 0], [2]);
  const target = torch.tensor([1], [1]);
  const before = torch.F.mse(model.forward(input), target).item();
  const fit = torch.train.fit(optimizer, loader, (batch: Record<string, any>) => (
    torch.F.mse(model.forward(batch.input), batch.target)
  ), {
    epochs: 20,
    zero_grad: true,
  });
  const after = torch.F.mse(model.forward(input), target).item();
  if (
    fit.kind !== "zgml.train.fit" ||
    fit.steps !== 40 ||
    fit.batch_count !== 2 ||
    fit.sample_count !== 4 ||
    !torch.train.isTrainFitEvidence(fit) ||
    !(after < before)
  ) {
    throw new Error(`${label} expected torch.train.fit to train through the PyTorch-like namespace`);
  }

  const snapshot = torch.checkpoint.create({ model, optimizer });
  const checkpointText = torch.save(snapshot, 2);
  const loadedSnapshot = torch.load(checkpointText);
  const rootCheckpointText = adapter.save(snapshot, 2);
  const rootLoadedSnapshot = adapter.load(rootCheckpointText);
  const inspection = torch.checkpoint.inspect(snapshot);
  if (
    inspection.modelParameterCount !== 4 ||
    inspection.optimizerKind !== "adamw" ||
    inspection.optimizerStep !== fit.steps ||
    torch.checkpoint.inspect(loadedSnapshot).signature !== inspection.signature ||
    adapter.checkpoint.inspect(rootLoadedSnapshot).signature !== inspection.signature
  ) {
    throw new Error(`${label} expected root/torch checkpoint helpers to snapshot trained model and optimizer state`);
  }
  const restoredForLoad = new torch.nn.Sequential(
    new torch.nn.Linear(2, 3, {
      weights: [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
      bias: [0, 0, 0],
    }),
    new torch.nn.ReLU(),
    new torch.nn.Linear(3, 1, {
      weights: [0.2, -0.1, 0.05],
      bias: [0],
    }),
  );
  const restoredOptimizerForLoad = new torch.optim.AdamW(restoredForLoad, { lr: 0.05, weight_decay: 0.001 });
  const restoredTargets = torch.load(checkpointText, { model: restoredForLoad, optimizer: restoredOptimizerForLoad, strict: true });
  if (restoredTargets.model !== restoredForLoad || restoredTargets.optimizer !== restoredOptimizerForLoad) {
    throw new Error(`${label} expected torch.load(text, targets) to restore checkpoint state and return targets`);
  }
  expectClose(restoredForLoad.forward(input).data, model.forward(input).data, `${label} torch.load restored/eager parity`);
  withTempCheckpointPath("torch-checkpoint.zgml", (checkpointPath, fs) => {
    const savedPath = torch.save(loadedSnapshot, checkpointPath, 2);
    if (savedPath !== checkpointPath || !fs.readFileSync(checkpointPath, "utf8").includes("\"format\": \"zgml.checkpoint\"")) {
      throw new Error(`${label} expected torch.save(snapshot, path) to write checkpoint JSON and return the path`);
    }
    const loadedFromPath = torch.load(checkpointPath);
    if (torch.checkpoint.inspect(loadedFromPath).signature !== inspection.signature) {
      throw new Error(`${label} expected torch.load(path) to read checkpoint state from disk`);
    }
    const restoredFromPath = new torch.nn.Sequential(
      new torch.nn.Linear(2, 3, {
        weights: [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
        bias: [0, 0, 0],
      }),
      new torch.nn.ReLU(),
      new torch.nn.Linear(3, 1, {
        weights: [0.2, -0.1, 0.05],
        bias: [0],
      }),
    );
    const restoredOptimizerFromPath = new torch.optim.AdamW(restoredFromPath, { lr: 0.05, weight_decay: 0.001 });
    const pathTargets = torch.load(checkpointPath, { model: restoredFromPath, optimizer: restoredOptimizerFromPath, strict: true });
    if (pathTargets.model !== restoredFromPath || pathTargets.optimizer !== restoredOptimizerFromPath) {
      throw new Error(`${label} expected torch.load(path, targets) to restore checkpoint state and return targets`);
    }
    expectClose(restoredFromPath.forward(input).data, model.forward(input).data, `${label} torch.load path restored/eager parity`);
  });
  const rootRestoredForLoad = new adapter.nn.Sequential(
    new adapter.nn.Linear(2, 3, {
      weights: [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
      bias: [0, 0, 0],
    }),
    new adapter.nn.ReLU(),
    new adapter.nn.Linear(3, 1, {
      weights: [0.2, -0.1, 0.05],
      bias: [0],
    }),
  );
  const rootRestoredOptimizerForLoad = new adapter.optim.AdamW(rootRestoredForLoad, { lr: 0.05, weight_decay: 0.001 });
  const rootRestoredTargets = adapter.load(rootCheckpointText, { model: rootRestoredForLoad, optimizer: rootRestoredOptimizerForLoad, strict: true });
  if (rootRestoredTargets.model !== rootRestoredForLoad || rootRestoredTargets.optimizer !== rootRestoredOptimizerForLoad) {
    throw new Error(`${label} expected root load(text, targets) to restore checkpoint state and return targets`);
  }
  expectClose(rootRestoredForLoad.forward(input).data, model.forward(input).data, `${label} root load restored/eager parity`);
  withTempCheckpointPath("root-checkpoint.zgml", (checkpointPath, fs) => {
    const savedPath = adapter.save(rootLoadedSnapshot, checkpointPath, 2);
    if (savedPath !== checkpointPath || !fs.readFileSync(checkpointPath, "utf8").includes("\"format\": \"zgml.checkpoint\"")) {
      throw new Error(`${label} expected root save(snapshot, path) to write checkpoint JSON and return the path`);
    }
    const loadedFromPath = adapter.load(checkpointPath);
    if (adapter.checkpoint.inspect(loadedFromPath).signature !== inspection.signature) {
      throw new Error(`${label} expected root load(path) to read checkpoint state from disk`);
    }
    const restoredFromPath = new adapter.nn.Sequential(
      new adapter.nn.Linear(2, 3, {
        weights: [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
        bias: [0, 0, 0],
      }),
      new adapter.nn.ReLU(),
      new adapter.nn.Linear(3, 1, {
        weights: [0.2, -0.1, 0.05],
        bias: [0],
      }),
    );
    const restoredOptimizerFromPath = new adapter.optim.AdamW(restoredFromPath, { lr: 0.05, weight_decay: 0.001 });
    const pathTargets = adapter.load(checkpointPath, { model: restoredFromPath, optimizer: restoredOptimizerFromPath, strict: true });
    if (pathTargets.model !== restoredFromPath || pathTargets.optimizer !== restoredOptimizerFromPath) {
      throw new Error(`${label} expected root load(path, targets) to restore checkpoint state and return targets`);
    }
    expectClose(restoredFromPath.forward(input).data, model.forward(input).data, `${label} root load path restored/eager parity`);
  });

  const lazy = torch.lazy.input([2]).linear(3).relu().linear(1);
  const lazyEmbeddingGraph = torch.lazy.input([2]).embedding(4, 3).layerNorm(3).rmsNorm(3);
  const lazyNamespaceEmbeddingGraph = torch.lazy.rmsNorm(torch.lazy.layerNorm(torch.lazy.embedding(torch.lazy.input([2]), 4, 3), 3), 3);
  const lazySnakeNamespaceEmbeddingGraph = torch.lazy.rms_norm(torch.lazy.layer_norm(torch.lazy.embedding(torch.lazy.input([2]), 4, 3), 3), 3);
  const lazyGridEmbeddingGraph = torch.lazy.input([1, 2]).embedding(4, 3);
  const lazyConvPoolGraph = torch.lazy.input([1, 4, 4]).conv2d(2, 1).maxPool2d(2).avgPool2d(2);
  const lazyNamespaceConvPoolGraph = torch.lazy.avgPool2d(torch.lazy.maxPool2d(torch.lazy.conv2d(torch.lazy.input([1, 4, 4]), 2, 1), 2), 2);
  const lazySnakeConvPoolGraph = torch.lazy.avg_pool2d(torch.lazy.max_pool2d(torch.lazy.input([1, 4, 4]).conv2d(2, 1), 2), 2);
  const lazyMultiChannelConvReluGraph = torch.lazy.input([2, 1, 4, 4]).conv2d(2, 3, { name: "conv" }).relu();
  const lazySigmoid = torch.lazy.input([2]).linear(3).sigmoid().linear(1);
  const lazyNamespaceSigmoid = torch.lazy.sigmoid(torch.lazy.input([2]).linear(3)).linear(1);
  const lazyMatmulWeight = torch.lazy.parameter([2, 3], "head.weight", "row-major:matmul.weight[in_features,out_features]");
  const lazyMatmulBias = torch.lazy.parameter([3], "head.bias", "row-major:add.bias[features]");
  const lazyMatmulScale = torch.lazy.parameter([3], "head.scale", "row-major:mul.scale[features]");
  const lazyMatmulShift = torch.lazy.parameter([3], "head.shift", "row-major:affine.bias[features]");
  const lazyMatmulGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).relu();
  const lazyMatmulBiasOnlyGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).add(lazyMatmulBias);
  const lazyMatmulBiasGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).add(lazyMatmulBias).relu();
  const lazyMatmulScaleOnlyGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).mul(lazyMatmulScale);
  const lazyMatmulScaleGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).mul(lazyMatmulScale).relu();
  const lazyAffineGraph = torch.lazy.input([2, 3]).affine(lazyMatmulScale, lazyMatmulShift);
  const lazyMatmulAffineGraph = torch.lazy.input([2]).matmul(lazyMatmulWeight).affine(lazyMatmulScale, lazyMatmulShift).relu();
  const lazyNamespaceMatmulGraph = torch.lazy.relu(torch.lazy.matmul(torch.lazy.input([2]), lazyMatmulWeight));
  const lazyNamespaceMatmulBiasGraph = torch.lazy.relu(torch.lazy.add(torch.lazy.matmul(torch.lazy.input([2]), lazyMatmulWeight), lazyMatmulBias));
  const lazyNamespaceMatmulScaleGraph = torch.lazy.relu(torch.lazy.mul(torch.lazy.matmul(torch.lazy.input([2]), lazyMatmulWeight), lazyMatmulScale));
  const lazyNamespaceMatmulAffineGraph = torch.lazy.relu(torch.lazy.affine(torch.lazy.matmul(torch.lazy.input([2]), lazyMatmulWeight), lazyMatmulScale, lazyMatmulShift));
  const lazyMmGraph = torch.lazy.input([1, 2]).mm(lazyMatmulWeight);
  const lazyActivationChain = torch.lazy.input([2]).exp().log().neg().recip().abs().sqrt().square().sgn().step();
  const lazyNamespaceActivationChain = torch.lazy.step(torch.lazy.sgn(torch.lazy.square(torch.lazy.sqrt(torch.lazy.abs(torch.lazy.recip(torch.lazy.neg(torch.lazy.log(torch.lazy.exp(torch.lazy.input([2]))))))))));
  const lazySnakeSoftmaxGraph = torch.lazy.input([2, 3]).log_softmax(1);
  const lazyDropoutGraph = torch.lazy.input([2]).dropout(0.5, { training: false }).relu();
  const lazyNamespaceDropoutGraph = torch.lazy.dropout(torch.lazy.input([2]), 0.5, { training: false }).relu();
  const lazyTrainingDropoutGraph = torch.lazy.input([2]).dropout(0.5, { training: true });
  const lazyReductionInput = torch.lazy.input([2, 3]);
  const lazyReductionChain = lazyReductionInput.sum(1).mean(0).max(0).min(0);
  const lazyNamespaceReduction = torch.lazy.min(torch.lazy.max(torch.lazy.mean(torch.lazy.sum(lazyReductionInput, 1), 0), 0), 0);
  const lazyShapeInput = torch.lazy.input([1, 3]);
  const lazyShapeChain = lazyShapeInput.squeeze(0).unsqueeze(0).broadcastTo([2, 3]).narrow(0, 0, 1).slice(1, 0, 2).transpose(0, 1).permute([1, 0]).select(0, 0);
  const lazyNamespaceShapeChain = torch.lazy.select(torch.lazy.permute(torch.lazy.transpose(torch.lazy.slice(torch.lazy.narrow(torch.lazy.broadcastTo(torch.lazy.unsqueeze(torch.lazy.squeeze(lazyShapeInput, 0), 0), [2, 3]), 0, 0, 1), 1, 0, 2), 0, 1), [1, 0]), 0, 0);
  const lazySnakeShapeGraph = lazyShapeInput.squeeze(0).unsqueeze(0).broadcast_to([2, 3]);
  const lazyShapeModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
    torch.nn.squeeze(0),
    torch.nn.unsqueeze(0),
    torch.nn.broadcastTo([2, 3]),
    torch.nn.narrow(0, 0, 1),
    torch.nn.slice(1, 0, 2),
    torch.nn.transpose(0, 1),
    torch.nn.permute([1, 0]),
    torch.nn.select(0, 0),
  ]), { inputShape: [1, 3] });
  const lazyEmbeddingModuleGraph = torch.lazy.fromModule(torch.nn.embedding(4, 3), { inputShape: [2] });
  const lazyNormModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
    torch.nn.embedding(4, 3),
    torch.nn.layerNorm(3),
    torch.nn.rmsNorm(3),
  ]), { inputShape: [2] });
  const lazyEvalDropoutModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
    torch.nn.dropout(0.5, { training: false }),
    torch.nn.relu(),
  ]), { inputShape: [2] });
  const lazyConvPoolModuleGraph = torch.lazy.fromModule(torch.nn.sequential([
    torch.nn.conv2d(1, 2, 1),
    torch.nn.max_pool2d(2),
    torch.nn.avg_pool2d(2),
  ]), { inputShape: [1, 4, 4] });
  const lazyTrainingDropoutModuleGraph = torch.lazy.fromModule(torch.nn.dropout(0.5, { training: true }), { inputShape: [2] });
  const lazyCompiledProgram = torch.compile.compile(lazy, { backend: "cpu" });
  const lazyMethodCompiledProgram = lazy.compile({ backend: "cpu" });
  const lazyMatmulBiasOnlyCompiledProgram = torch.compile.compile(lazyMatmulBiasOnlyGraph, { backend: "cpu" });
  const lazyMatmulBiasCompiledProgram = torch.compile.compile(lazyMatmulBiasGraph, { backend: "cpu" });
  const lazyMatmulScaleOnlyCompiledProgram = torch.compile.compile(lazyMatmulScaleOnlyGraph, { backend: "cpu" });
  const lazyMatmulScaleCompiledProgram = torch.compile.compile(lazyMatmulScaleGraph, { backend: "cpu" });
  const lazyAffineCompiledProgram = torch.compile.compile(lazyAffineGraph, { backend: "cpu" });
  const lazyMatmulAffineCompiledProgram = torch.compile.compile(lazyMatmulAffineGraph, { backend: "cpu" });
  const lazyMultiChannelConvReluProgram = lazyMultiChannelConvReluGraph.compile({ backend: "cpu" });
  if (
    lazy.compileSupport().supported !== true ||
    lazyEmbeddingGraph.compileSupport().supported !== true ||
    lazyNamespaceEmbeddingGraph.compileSupport().supported !== true ||
    lazySnakeNamespaceEmbeddingGraph.compileSupport().supported !== true ||
    lazyGridEmbeddingGraph.compileSupport().supported !== false ||
    lazyConvPoolGraph.compileSupport().supported !== true ||
    lazyNamespaceConvPoolGraph.compileSupport().supported !== true ||
    lazySnakeConvPoolGraph.compileSupport().supported !== true ||
    lazyMultiChannelConvReluGraph.compileSupport().supported !== true ||
    lazyMultiChannelConvReluGraph.kernelPlan()?.ops[0]?.nativeKernels.join("|") !== "conv2d|relu" ||
    lazyMultiChannelConvReluGraph.kernelPlan()?.ops[0]?.fusedOpCount !== 2 ||
    lazySigmoid.compileSupport().supported !== true ||
    lazyNamespaceSigmoid.compileSupport().supported !== true ||
    lazyMatmulGraph.compileSupport().supported !== true ||
    lazyMatmulBiasOnlyGraph.compileSupport().supported !== true ||
    lazyMatmulBiasGraph.compileSupport().supported !== true ||
    lazyMatmulScaleOnlyGraph.compileSupport().supported !== true ||
    lazyMatmulScaleGraph.compileSupport().supported !== true ||
    lazyAffineGraph.compileSupport().supported !== true ||
    lazyMatmulAffineGraph.compileSupport().supported !== true ||
    lazyNamespaceMatmulGraph.compileSupport().supported !== true ||
    lazyNamespaceMatmulBiasGraph.compileSupport().supported !== true ||
    lazyNamespaceMatmulScaleGraph.compileSupport().supported !== true ||
    lazyNamespaceMatmulAffineGraph.compileSupport().supported !== true ||
    lazyMmGraph.compileSupport().supported !== true ||
    lazyMatmulGraph.trace().ops[0]?.op !== "matmul" ||
    lazyMatmulGraph.tensorProgramIr()?.ops[0]?.op !== "matmul" ||
    lazyMatmulGraph.kernelPlan()?.ops[0]?.kernel !== "matmul" ||
    lazyMatmulGraph.kernelPlan()?.ops[0]?.nativeKernels.join("|") !== "linear" ||
    lazyMatmulGraph.kernelPlan()?.parameterLayout.parameters[0]?.name !== "head.weight" ||
    lazyMatmulBiasOnlyGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul|add" ||
    lazyMatmulBiasOnlyGraph.tensorProgramIr()?.ops[1]?.op !== "add" ||
    lazyMatmulBiasOnlyGraph.kernelPlan()?.ops[0]?.fusedOpCount !== 2 ||
    lazyMatmulBiasOnlyGraph.kernelPlan()?.ops[0]?.nativeKernels.join("|") !== "linear|add" ||
    lazyMatmulBiasOnlyGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.weight:weights|head.bias:bias" ||
    lazyMatmulBiasGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul|add|activation" ||
    lazyMatmulBiasGraph.tensorProgramIr()?.ops[1]?.op !== "add" ||
    lazyMatmulBiasGraph.kernelPlan()?.ops[0]?.fusedOpCount !== 3 ||
    lazyMatmulBiasGraph.kernelPlan()?.ops[0]?.nativeKernels.join("|") !== "linear|add|relu" ||
    lazyMatmulBiasGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.weight:weights|head.bias:bias" ||
    lazyMatmulScaleOnlyGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul|mul" ||
    lazyMatmulScaleOnlyGraph.tensorProgramIr()?.ops[1]?.op !== "mul" ||
    lazyMatmulScaleOnlyGraph.kernelPlan()?.opCount !== 2 ||
    lazyMatmulScaleOnlyGraph.kernelPlan()?.dispatchCount !== 2 ||
    lazyMatmulScaleOnlyGraph.kernelPlan()?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|mul" ||
    lazyMatmulScaleOnlyGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.weight:weights|head.scale:weights" ||
    lazyMatmulScaleGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul|mul|activation" ||
    lazyMatmulScaleGraph.tensorProgramIr()?.ops[1]?.op !== "mul" ||
    lazyMatmulScaleGraph.kernelPlan()?.opCount !== 3 ||
    lazyMatmulScaleGraph.kernelPlan()?.dispatchCount !== 3 ||
    lazyMatmulScaleGraph.kernelPlan()?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|mul|relu" ||
    lazyMatmulScaleGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.weight:weights|head.scale:weights" ||
    lazyAffineGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "affine" ||
    lazyAffineGraph.tensorProgramIr()?.ops[0]?.op !== "affine" ||
    lazyAffineGraph.kernelPlan()?.opCount !== 1 ||
    lazyAffineGraph.kernelPlan()?.dispatchCount !== 1 ||
    lazyAffineGraph.kernelPlan()?.ops[0]?.nativeKernels.join("|") !== "affine" ||
    lazyAffineGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.scale:weights|head.shift:bias" ||
    lazyMatmulAffineGraph.trace().ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul|affine|activation" ||
    lazyMatmulAffineGraph.tensorProgramIr()?.ops[1]?.op !== "affine" ||
    lazyMatmulAffineGraph.kernelPlan()?.opCount !== 3 ||
    lazyMatmulAffineGraph.kernelPlan()?.dispatchCount !== 3 ||
    lazyMatmulAffineGraph.kernelPlan()?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|affine|relu" ||
    lazyMatmulAffineGraph.kernelPlan()?.parameterLayout.parameters.map((param: Record<string, any>) => `${param.name}:${param.binding}`).join("|") !== "head.weight:weights|head.scale:weights|head.shift:bias" ||
    lazyActivationChain.compileSupport().supported !== true ||
    lazyNamespaceActivationChain.compileSupport().supported !== true ||
    lazySnakeSoftmaxGraph.compileSupport().supported !== true ||
    lazyDropoutGraph.compileSupport().supported !== true ||
    lazyNamespaceDropoutGraph.compileSupport().supported !== true ||
    lazyTrainingDropoutGraph.compileSupport().supported !== false ||
    lazy.canCompile() !== true ||
    lazy.can_compile() !== true ||
    lazy.requireCompileSupport().supported !== true ||
    lazy.require_compile_support().supported !== true ||
    torch.lazy.canCompile(lazy) !== true ||
    torch.lazy.can_compile(lazy) !== true ||
    torch.lazy.requireCompileSupport(lazy).supported !== true ||
    torch.lazy.require_compile_support(lazy).supported !== true ||
    lazyTrainingDropoutGraph.canCompile() !== false ||
    lazyTrainingDropoutGraph.can_compile() !== false ||
    torch.lazy.canCompile(lazyTrainingDropoutGraph) !== false ||
    torch.compile.canCompile(lazy) !== true ||
    torch.compile.can_compile(lazy) !== true ||
    torch.compile.compileSupport(lazy).supported !== true ||
    torch.compile.requireCompileSupport(lazy).supported !== true ||
    torch.compile.requireCompilePlan(lazy).supported !== true ||
    torch.compile.tensorProgramIr(lazy)?.kind !== "tensor-program-ir" ||
    torch.compile.kernelPlan(lazy)?.kind !== "native-module-kernel-plan" ||
    torch.compile.inputShape(lazy).join("x") !== "2" ||
    torch.compile.outputShape(lazy).join("x") !== "1" ||
    torch.compile.parameterLayout(lazy).parameters.length !== 4 ||
    lazyCompiledProgram.outputShape().join("x") !== "1" ||
    lazyCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 3 ||
    lazyMethodCompiledProgram.outputShape().join("x") !== "1" ||
    lazyMethodCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 3 ||
    lazyMatmulBiasOnlyCompiledProgram.outputShape().join("x") !== "3" ||
    lazyMatmulBiasOnlyCompiledProgram.compileEvidence()?.kernelPlan?.ops[0]?.fusedOpCount !== 2 ||
    lazyMatmulBiasOnlyCompiledProgram.compileEvidence()?.kernelPlan?.ops[0]?.nativeKernels.join("|") !== "linear|add" ||
    lazyMatmulBiasCompiledProgram.outputShape().join("x") !== "3" ||
    lazyMatmulBiasCompiledProgram.compileEvidence()?.kernelPlan?.ops[0]?.fusedOpCount !== 3 ||
    lazyMatmulBiasCompiledProgram.compileEvidence()?.kernelPlan?.ops[0]?.nativeKernels.join("|") !== "linear|add|relu" ||
    lazyMatmulScaleOnlyCompiledProgram.outputShape().join("x") !== "3" ||
    lazyMatmulScaleOnlyCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 2 ||
    lazyMatmulScaleOnlyCompiledProgram.compileEvidence()?.kernelPlan?.dispatchCount !== 2 ||
    lazyMatmulScaleOnlyCompiledProgram.compileEvidence()?.kernelPlan?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|mul" ||
    lazyMatmulScaleCompiledProgram.outputShape().join("x") !== "3" ||
    lazyMatmulScaleCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 3 ||
    lazyMatmulScaleCompiledProgram.compileEvidence()?.kernelPlan?.dispatchCount !== 3 ||
    lazyMatmulScaleCompiledProgram.compileEvidence()?.kernelPlan?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|mul|relu" ||
    lazyAffineCompiledProgram.outputShape().join("x") !== "2x3" ||
    lazyAffineCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 1 ||
    lazyAffineCompiledProgram.compileEvidence()?.kernelPlan?.ops[0]?.nativeKernels.join("|") !== "affine" ||
    lazyMatmulAffineCompiledProgram.outputShape().join("x") !== "3" ||
    lazyMatmulAffineCompiledProgram.compileEvidence()?.kernelPlan?.opCount !== 3 ||
    lazyMatmulAffineCompiledProgram.compileEvidence()?.kernelPlan?.ops.map((op: Record<string, any>) => op.nativeKernels.join("|")).join("|") !== "linear|affine|relu" ||
    lazyMultiChannelConvReluProgram.outputShape().join("x") !== "2x2x2x2" ||
    lazyMultiChannelConvReluProgram.compileEvidence()?.kernelPlan?.ops[0]?.nativeKernels.join("|") !== "conv2d|relu" ||
    lazyReductionChain.compileSupport().supported !== true ||
    lazyNamespaceReduction.compileSupport().supported !== true ||
    lazyShapeChain.compileSupport().supported !== true ||
    lazyNamespaceShapeChain.compileSupport().supported !== true ||
    lazySnakeShapeGraph.compileSupport().supported !== true ||
    lazyShapeModuleGraph.compileSupport().supported !== true ||
    lazyEmbeddingModuleGraph.compileSupport().supported !== true ||
    lazyNormModuleGraph.compileSupport().supported !== true ||
    lazyEvalDropoutModuleGraph.compileSupport().supported !== true ||
    lazyConvPoolModuleGraph.compileSupport().supported !== true ||
    lazyTrainingDropoutModuleGraph.compileSupport().supported !== false ||
    !String(lazyTrainingDropoutModuleGraph.compileSupport().reason).includes("deterministic no-op nn.Dropout") ||
    lazy.tensorProgramIr()?.kind !== "tensor-program-ir" ||
    lazy.kernelPlan()?.kind !== "native-module-kernel-plan"
  ) {
    throw new Error(`${label} expected torch.lazy to expose compile-capable typed graph evidence`);
  }
  const lazyMultiChannelConvReluSession = lazyMultiChannelConvReluProgram.bind({
    weights: new Float32Array([
      1, 0, -1,
      0, 1, 0,
      -1, 0, 1,

      -1, 0.5, 0.25,
      0, -0.5, 0,
      0.25, 0.5, -1,
    ]),
    bias: new Float32Array([-1.5, 0.75]),
  });
  try {
    const convInput = torch.tensor([
      1, -2, 3, 4,
      5, 6, -7, 8,
      9, 10, 11, -12,
      13, 14, 15, 16,

      -1, -2, -3, -4,
      5, 6, 7, 8,
      -9, 10, -11, 12,
      13, -14, 15, -16,
    ], [2, 1, 4, 4]);
    expectClose(
      lazyMultiChannelConvReluSession.stepTensor(convInput).data,
      [
        4.5, 0, 22.5, 9.5,
        0, 28.75, 0, 0,
        4.5, 9.5, 8.5, 0,
        10.75, 0, 0, 25.75,
      ],
      `${label} lazy multi-channel Conv2d+ReLU compiled output`,
    );
  } finally {
    lazyMultiChannelConvReluSession.dispose();
  }
  const lazyMatmulScaleSession = lazyMatmulScaleOnlyCompiledProgram.bind({
    weights: new Float32Array([
      1, 0, 0,
      0, 1, 1,
      2, -1, 0.5,
    ]),
  });
  try {
    expectClose(
      lazyMatmulScaleSession.stepTensor(torch.tensor([1, 2], [2])).data,
      [2, -2, 1],
      `${label} lazy matmul scale compiled output`,
    );
  } finally {
    lazyMatmulScaleSession.dispose();
  }
  const lazyAffineSession = lazyAffineCompiledProgram.bind({
    weights: new Float32Array([2, -1, 0.5]),
    bias: new Float32Array([1, 10, -2]),
  });
  try {
    expectClose(
      lazyAffineSession.stepTensor(torch.tensor([1, 2, 3, 4, 5, 6], [2, 3])).data,
      [3, 8, -0.5, 9, 5, 1],
      `${label} lazy affine compiled output`,
    );
  } finally {
    lazyAffineSession.dispose();
  }
  lazyCompiledProgram.free();
  lazyMethodCompiledProgram.free();
  lazyMatmulBiasOnlyCompiledProgram.free();
  lazyMatmulBiasCompiledProgram.free();
  lazyMatmulScaleOnlyCompiledProgram.free();
  lazyMatmulScaleCompiledProgram.free();
  lazyAffineCompiledProgram.free();
  lazyMatmulAffineCompiledProgram.free();
  lazyMultiChannelConvReluProgram.free();
  expectThrowIncludes(() => lazyTrainingDropoutGraph.requireCompileSupport(), "lazy graph cannot compile", `${label} lazy requireCompileSupport unsupported graph`);
  expectThrowIncludes(() => lazyTrainingDropoutGraph.compile({ backend: "cpu" }), "lazy graph cannot compile", `${label} lazy compile rejects unsupported graph`);
  expectThrowIncludes(() => torch.lazy.require_compile_support(lazyTrainingDropoutGraph), "lazy graph cannot compile", `${label} lazy namespace require_compile_support unsupported graph`);
  expectThrowIncludes(() => torch.compile.requireCompileSupport(lazyTrainingDropoutGraph), "compile.requireCompileSupport rejected unsupported target", `${label} compile namespace lazy require unsupported graph`);

  const support = torch.compile.compileSupport(model, { inputShape: [2], backend: "cpu" });
  const plan = torch.compile.requireCompilePlan(model, { inputShape: [2], backend: "cpu" });
  if (
    support.supported !== true ||
    plan.supported !== true ||
    support.outputShape.join("x") !== "1" ||
    torch.compile.canCompile(model, { inputShape: [2], backend: "cpu" }) !== true
  ) {
    throw new Error(`${label} expected torch.compile to prove trained module compile support`);
  }

  const directProgram = torch.compile(model, { inputShape: [2], backend: "cpu" });
  directProgram.dispose();
  const program = torch.compile.compile(model, { inputShape: [2], backend: "cpu" });
  const session = program.bindModule(model);
  try {
    const output = new Float32Array(1);
    const compiled = session.executeInto(output, { input });
    const hotPath = session.requireHotStepParams({ input, output });
    if (
      !(program instanceof torch.Program) ||
      !(session instanceof torch.Session) ||
      compiled !== output ||
      hotPath.hotPath !== true ||
      hotPath.runtimeOutputAllocationFree !== true ||
      hotPath.readbackFree !== true
    ) {
      throw new Error(`${label} expected torch.compile Program/Session to execute allocation-free into caller output`);
    }
    expectClose(output, model.forward(input).data, `${label} torch namespace compiled/eager parity`);
  } finally {
    session.dispose();
    program.dispose();
  }
}

export function smokePackage(adapter: Record<string, any>, label: string) {
  expectPackageSelfReferenceExports(adapter, label);
  expectNativeApiContractManifest(nativeApiContractManifest, `${label} native API contract`);
  if (
    nativeApiContractManifest.packageSpine?.source !== "generated-ts" ||
    nativeApiContractManifest.packageSpine?.generatedFrom !== "src/ts/runtime/native_api_contract.ts" ||
    nativeApiContractManifest.packageSpine?.handwrittenRootExportList !== false
  ) {
    throw new Error(`${label} must consume the TS-authored native API contract`);
  }
  const missingNativeExports = missingExports(adapter, requiredNativeApiExports);
  if (missingNativeExports.length !== 0) {
    throw new Error(`${label} adapter is missing native zgml exports: ${missingNativeExports.join(", ")}`);
  }
  if (
    adapter.compile?.compileManifest?.policyOwner !== "src/ts/compile.ts" ||
    typeof adapter.compile !== "function" ||
    typeof adapter.compile.trace !== "function" ||
    typeof adapter.compile.compile !== "function" ||
    typeof adapter.compile.native !== "function" ||
    adapter.compile.native !== adapter.compile.compileForInference ||
    typeof adapter.compile.compileForInference !== "function" ||
    typeof adapter.compile.compile_for_inference !== "function" ||
    typeof adapter.native !== "function" ||
    adapter.native !== adapter.compile.compileForInference ||
    typeof adapter.zgml?.native !== "function" ||
    adapter.zgml.native !== adapter.compile.compileForInference ||
    typeof adapter.compileInference !== "function" ||
    adapter.compileInference !== adapter.compile.compileForInference
  ) {
    throw new Error(`${label} adapter root must expose the TS-authored compile namespace`);
  }
  const inferenceModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0.5] });
  const inferenceInput = adapter.tensor([1, 2], [2]);
  const inference = adapter.compile.compileForInference(inferenceModel, { backend: "cpu", inputShape: [2] });
  try {
    if (!Object.isFrozen(inference) || inference.program.inputLen() !== 2 || inference.session.outputLen() !== 1) {
      throw new Error(`${label} expected frozen compiled inference handle over Program/Session`);
    }
    const inferenceSupport = inference.compileSupport();
    const inferenceExplanation = inference.explain();
    const inferencePreflight = inference.preflight();
    const inferenceExecutionPlan = inference.requireExecutionPlan();
    if (
      inference.native !== true ||
      inferenceExecutionPlan.canExecute !== true ||
      inferenceExecutionPlan.executionMode !== "executable" ||
      inferenceSupport.supported !== true ||
      inferenceExplanation.supported !== true ||
      inferencePreflight.supported !== true ||
      inferenceExplanation.signature !== inferencePreflight.signature ||
      inference.inputShape().join("x") !== "2" ||
      inference.outputShape().join("x") !== "1" ||
      inference.kernelPlan()?.ops.map((op: Record<string, any>) => op.op).join("|") !== "linear" ||
      inference.compilerSignatures()?.kernelPlan !== inference.kernelPlan()?.signature
    ) {
      throw new Error(`${label} expected compileForInference handle to expose compile evidence`);
    }
    expectClose(inference.forward(inferenceInput).data, [-0.5], `${label} compileForInference forward`);
    expectClose(inference.call(inferenceInput).data, [-0.5], `${label} compileForInference call`);
    expectClose(inference.__call__(inferenceInput).data, [-0.5], `${label} compileForInference __call__`);
    expectClose(inference.stepTensor(inferenceInput).data, [-0.5], `${label} compileForInference stepTensor alias`);
    const inferenceCarrier = new Float32Array(1);
    if (inference.into(inferenceCarrier, inferenceInput) !== inferenceCarrier) {
      throw new Error(`${label} expected compileForInference into to reuse caller output`);
    }
    expectClose(inferenceCarrier, [-0.5], `${label} compileForInference into`);
    const preparedCarrier = new Float32Array(1);
    const preparedInference = inference.prepareInto(preparedCarrier, inferenceInput);
    if (preparedInference() !== preparedCarrier) {
      throw new Error(`${label} expected compileForInference prepareInto to reuse caller output`);
    }
    expectClose(preparedCarrier, [-0.5], `${label} compileForInference prepareInto`);
  } finally {
    inference.dispose();
  }
  const zgmlNativeRun = adapter.zgml.native(inferenceModel, { backend: "cpu", inputShape: [2] });
  try {
    const plan = zgmlNativeRun.requireExecutionPlan();
    if (
      zgmlNativeRun.native !== true ||
      plan.canExecute !== true ||
      plan.executionMode !== "executable" ||
      zgmlNativeRun.program.inputLen() !== 2 ||
      zgmlNativeRun.session.outputLen() !== 1
    ) {
      throw new Error(`${label} zgml.native expected native executable Program/Session handle`);
    }
    expectClose(zgmlNativeRun.forward(inferenceInput).data, [-0.5], `${label} zgml.native forward`);
    const zgmlNativeCarrier = new Float32Array(1);
    if (zgmlNativeRun.into(zgmlNativeCarrier, inferenceInput) !== zgmlNativeCarrier) {
      throw new Error(`${label} zgml.native expected into to reuse caller output`);
    }
    expectClose(zgmlNativeCarrier, [-0.5], `${label} zgml.native into`);
  } finally {
    zgmlNativeRun.dispose();
  }
  const rawLayerNativeRun = adapter.zgml.native([inferenceModel], { backend: "cpu", inputShape: [2] });
  try {
    const plan = rawLayerNativeRun.requireExecutionPlan();
    if (
      rawLayerNativeRun.native !== true ||
      plan.canExecute !== true ||
      plan.executionMode !== "executable" ||
      rawLayerNativeRun.program.inputLen() !== 2 ||
      rawLayerNativeRun.session.outputLen() !== 1
    ) {
      throw new Error(`${label} zgml.native raw layer list expected native executable Program/Session handle`);
    }
    expectClose(rawLayerNativeRun.forward(inferenceInput).data, [-0.5], `${label} zgml.native raw layer list forward`);
  } finally {
    rawLayerNativeRun.dispose();
  }
  const nativeInference = adapter.nn.native(inferenceModel, { backend: "cpu", inputShape: [2] });
  try {
    expectClose(nativeInference.forward(inferenceInput).data, [-0.5], `${label} nn.native forward`);
    expectClose(nativeInference.call(inferenceInput).data, [-0.5], `${label} nn.native call`);
    expectClose(nativeInference.__call__(inferenceInput).data, [-0.5], `${label} nn.native __call__`);
    const nativeInferenceAlias = inferenceModel.native({ backend: "cpu", inputShape: [2] });
    try {
      expectClose(nativeInferenceAlias.forward(inferenceInput).data, [-0.5], `${label} module.native forward`);
    } finally {
      nativeInferenceAlias.dispose();
    }
  } finally {
    nativeInference.dispose();
  }
  const lazyInferenceGraph = adapter.lazy.input([2])
    .matmul(adapter.lazy.parameter([2, 1], "w"))
    .add(adapter.lazy.parameter([1], "b"));
  const lazyInference = adapter.compile.compileForInference(lazyInferenceGraph, { backend: "cpu", inputShape: [2] }, {
    weights: new Float32Array([1, -1]),
    bias: new Float32Array([0.5]),
  });
  try {
    if (!Object.isFrozen(lazyInference) || lazyInference.program.inputLen() !== 2 || lazyInference.session.outputLen() !== 1) {
      throw new Error(`${label} expected lazy compileForInference handle over Program/Session`);
    }
    if (
      lazyInference.compileSupport().supported !== true ||
      lazyInference.kernelPlan()?.ops.map((op: Record<string, any>) => op.op).join("|") !== "matmul" ||
      lazyInference.inputShape().join("x") !== "2" ||
      lazyInference.outputShape().join("x") !== "1"
    ) {
      throw new Error(`${label} expected lazy compileForInference handle to expose compile evidence`);
    }
    const lazyCarrier = new Float32Array(1);
    if (lazyInference.into(lazyCarrier, inferenceInput) !== lazyCarrier) {
      throw new Error(`${label} expected lazy compileForInference into to reuse caller output`);
    }
    expectClose(lazyCarrier, [-0.5], `${label} lazy compileForInference into`);
  } finally {
    lazyInference.dispose();
  }
  const inferenceAlias = adapter.compile.compile_for_inference(inferenceModel, { backend: "cpu", inputShape: [2] });
  inferenceAlias.free();
  const nested = adapter.tensor([[1, 2], [3, 4]]);
  if (nested.shape.join("x") !== "2x2") throw new Error(`${label} nested tensor shape mismatch`);
  expectClose(nested.data, [1, 2, 3, 4], `${label} nested tensor data`);
  if (JSON.stringify(nested.to_list()) !== JSON.stringify(nested.tolist())) {
    throw new Error(`${label} expected Tensor.to_list alias to match Tensor.tolist`);
  }
  const scalarValue = adapter.Tensor.scalar(7);
  if (scalarValue.item() !== 7 || scalarValue.toNumber() !== 7 || scalarValue.valueOf() !== 7 || Number(scalarValue) !== 7) {
    throw new Error(`${label} expected scalar Tensor extraction helpers to agree`);
  }
  expectThrowIncludes(
    () => nested.toNumber(),
    "Tensor.item() requires a scalar tensor",
    `${label} Tensor.toNumber rejects non-scalar tensors`,
  );
  expectClose(nested.eq(adapter.tensor([[1, 0], [3, 5]])).data, [1, 0, 1, 0], `${label} Tensor.eq`);
  expectClose(nested.ne(2).data, [1, 0, 1, 1], `${label} Tensor.ne scalar`);
  expectClose(nested.lt(adapter.tensor([2, 4], [1, 2])).data, [1, 1, 0, 0], `${label} Tensor.lt broadcast`);
  expectClose(nested.le(3).data, [1, 1, 1, 0], `${label} Tensor.le scalar`);
  expectClose(nested.gt(2).data, [0, 0, 1, 1], `${label} Tensor.gt scalar`);
  expectClose(nested.ge(adapter.tensor([2, 4], [1, 2])).data, [0, 0, 1, 1], `${label} Tensor.ge broadcast`);
  const nestedJson = nested.toJSON();
  if (
    nestedJson.dtype !== "f32" ||
    nestedJson.shape.join("x") !== "2x2" ||
    nestedJson.data.join("|") !== "1|2|3|4" ||
    nestedJson.requiresGrad !== false
  ) {
    throw new Error(`${label} expected Tensor.toJSON to preserve dtype, shape, data, and requiresGrad`);
  }
  if (
    nested.dim() !== 2 ||
    nested.ndimension() !== 2 ||
    nested.numel() !== 4 ||
    nested.nelement() !== 4 ||
    nested.size().join("x") !== "2x2" ||
    nested.size(-1) !== 2 ||
    nested.stride().join("x") !== "2x1" ||
    nested.stride(0) !== 2 ||
    nested.storage_offset() !== 0 ||
    nested.element_size() !== Float32Array.BYTES_PER_ELEMENT ||
    nested.nbytes() !== 4 * Float32Array.BYTES_PER_ELEMENT ||
    nested.isContiguous() !== true ||
    nested.is_contiguous() !== true ||
    nested.contiguous() !== nested
  ) {
    throw new Error(`${label} expected Tensor metadata and is_contiguous aliases to match dense row-major storage`);
  }
  const trainableJson = adapter.tensor([1, 2], [2], { requiresGrad: true }).toJSON();
  if (trainableJson.requiresGrad !== true) {
    throw new Error(`${label} expected Tensor.toJSON to preserve trainable requiresGrad`);
  }
  if (adapter.is_grad_enabled() !== adapter.isGradEnabled()) {
    throw new Error(`${label} expected is_grad_enabled alias to match isGradEnabled`);
  }
  const previousGradMode = adapter.set_grad_enabled(true);
  adapter.setGradEnabled(previousGradMode);
  const gradModeInput = adapter.tensor([2], [1], { requiresGrad: true });
  const noGradOutput = adapter.no_grad(() => gradModeInput.mul(3));
  if (noGradOutput.requiresGrad !== false || noGradOutput.requires_grad !== false) {
    throw new Error(`${label} expected no_grad alias to suppress autograd graph creation`);
  }
  const inferenceModeOutput = adapter.inference_mode(() => gradModeInput.add(1));
  if (inferenceModeOutput.requiresGrad !== false || inferenceModeOutput.requires_grad !== false) {
    throw new Error(`${label} expected inference_mode alias to suppress autograd graph creation`);
  }
  adapter.set_grad_enabled(false);
  const enableGradOutput = adapter.enable_grad(() => gradModeInput.mul(4));
  if (enableGradOutput.requiresGrad !== true || adapter.is_grad_enabled() !== false) {
    throw new Error(`${label} expected enable_grad alias to enable callback then restore disabled grad mode`);
  }
  let restoredAfterThrow = false;
  try {
    adapter.no_grad(() => {
      throw new Error("intentional no_grad alias failure");
    });
  } catch (_err) {
    restoredAfterThrow = adapter.is_grad_enabled() === false;
  }
  if (!restoredAfterThrow) {
    throw new Error(`${label} expected no_grad alias to restore grad mode after callback error`);
  }
  adapter.set_grad_enabled(true);
  const nestedJsonRoundTrip = adapter.Tensor.fromJSON(nestedJson);
  if (!(nestedJsonRoundTrip instanceof adapter.Tensor)) {
    throw new Error(`${label} expected Tensor.fromJSON to return a Tensor`);
  }
  expectClose(nestedJsonRoundTrip.data, [1, 2, 3, 4], `${label} Tensor.fromJSON data`);
  if (nestedJsonRoundTrip.shape.join("x") !== "2x2") {
    throw new Error(`${label} expected Tensor.fromJSON shape round-trip`);
  }
  const trainableRoundTrip = adapter.Tensor.fromJSON(trainableJson);
  if (trainableRoundTrip.requiresGrad !== true || trainableRoundTrip.requires_grad !== true || !(trainableRoundTrip.grad instanceof Float32Array)) {
    throw new Error(`${label} expected Tensor.fromJSON to preserve trainable requiresGrad`);
  }
  const snakeGradRoundTrip = adapter.Tensor.fromJSON({ dtype: "f32", shape: [1], data: [9], requires_grad: true });
  if (snakeGradRoundTrip.requiresGrad !== true || snakeGradRoundTrip.requires_grad !== true) {
    throw new Error(`${label} expected Tensor.fromJSON to accept requires_grad alias`);
  }
  expectThrowIncludes(
    () => adapter.Tensor.fromJSON({ dtype: "f32", shape: [2, 2], data: [1, 2, 3] }),
    "tensor shape product 4 must match tensor length 3",
    `${label} Tensor.fromJSON rejects shape/data mismatch`,
  );
  expectThrowIncludes(
    () => adapter.Tensor.fromJSON({ dtype: "f32", shape: [1], data: [1], requiresGrad: "yes" }),
    "Tensor.fromJSON requiresGrad must be a boolean when provided",
    `${label} Tensor.fromJSON rejects non-boolean requiresGrad`,
  );
  expectThrowIncludes(
    () => adapter.Tensor.fromJSON({ dtype: "f32", shape: [1], data: [1], requiresGrad: true, requires_grad: false }),
    "Tensor.fromJSON requiresGrad and requires_grad must match when both are provided",
    `${label} Tensor.fromJSON rejects conflicting requiresGrad aliases`,
  );
  expectThrowIncludes(
    () => adapter.Tensor.fromJSON({ dtype: "i32", shape: [1], data: [1] }),
    'Tensor.fromJSON expects { dtype: "f32", shape, data }',
    `${label} Tensor.fromJSON rejects non-f32 payloads`,
  );
  expectClose(adapter.add(nested, adapter.tensor([10, 20], [1, 2])).data, [11, 22, 13, 24], `${label} add root helper`);
  expectClose(adapter.Tensor.sub(nested, 1).data, [0, 1, 2, 3], `${label} Tensor.sub static helper`);
  expectClose(adapter.mul(nested, 2).data, [2, 4, 6, 8], `${label} mul root helper`);
  expectClose(adapter.Tensor.div(nested, 2).data, [0.5, 1, 1.5, 2], `${label} Tensor.div static helper`);
  expectClose(adapter.to(nested, "cpu").data, [1, 2, 3, 4], `${label} to root helper`);
  expectClose(adapter.cpu(nested).data, [1, 2, 3, 4], `${label} cpu root helper`);
  expectClose(adapter.float(nested).data, [1, 2, 3, 4], `${label} float root helper`);
  expectClose(adapter.float32(nested).data, [1, 2, 3, 4], `${label} float32 root helper`);
  expectClose(adapter.Tensor.to(nested, { device: "cpu", copy: true }).data, [1, 2, 3, 4], `${label} Tensor.to static helper`);
  if (adapter.typeAs(nested, nested) !== nested) throw new Error(`${label} expected typeAs root helper to preserve eager cpu f32 tensor identity`);
  if (nested.type_as(nested) !== nested) throw new Error(`${label} expected Tensor.type_as instance helper to preserve eager cpu f32 tensor identity`);
  if (
    typeof adapter.as_tensor !== "function" ||
    typeof adapter.asTensor !== "function" ||
    typeof adapter.asarray !== "function" ||
    typeof adapter.from_numpy !== "function" ||
    typeof adapter.fromNumpy !== "function"
  ) {
    throw new Error(`${label} expected root tensor factory aliases to be runtime exports`);
  }
  expectClose(adapter.as_tensor(new Float32Array([1, 2]), [2]).data, [1, 2], `${label} as_tensor root helper`);
  expectClose(adapter.asTensor([1, 2], [2]).data, [1, 2], `${label} asTensor root helper`);
  expectClose(adapter.asarray([1, 2], [2]).data, [1, 2], `${label} asarray root helper`);
  expectClose(adapter.from_numpy(new Float32Array([1, 2]), [2]).data, [1, 2], `${label} from_numpy root helper`);
  expectClose(adapter.fromNumpy([1, 2], [2]).data, [1, 2], `${label} fromNumpy root helper`);
  const copiedTypeAs = adapter.Tensor.type_as(nested, nested, { copy: true });
  if (copiedTypeAs === nested) throw new Error(`${label} expected Tensor.type_as copy option to clone eager cpu f32 tensor`);
  expectClose(copiedTypeAs.data, [1, 2, 3, 4], `${label} Tensor.type_as static helper`);
  expectClose(nested.new_empty([2, 3]).data, [0, 0, 0, 0, 0, 0], `${label} Tensor.new_empty helper zero-backed JS allocation`);
  expectClose(nested.newEmpty([2, 3]).data, [0, 0, 0, 0, 0, 0], `${label} Tensor.newEmpty helper`);
  expectClose(nested.new_zeros([2, 3]).data, [0, 0, 0, 0, 0, 0], `${label} Tensor.new_zeros helper`);
  expectClose(nested.newZeros([2, 3]).data, [0, 0, 0, 0, 0, 0], `${label} Tensor.newZeros helper`);
  expectClose(nested.new_ones([2, 3]).data, [1, 1, 1, 1, 1, 1], `${label} Tensor.new_ones helper`);
  expectClose(nested.newOnes([2, 3]).data, [1, 1, 1, 1, 1, 1], `${label} Tensor.newOnes helper`);
  expectClose(nested.new_full([2, 3], 5).data, [5, 5, 5, 5, 5, 5], `${label} Tensor.new_full helper`);
  expectClose(nested.newFull([2, 3], 5).data, [5, 5, 5, 5, 5, 5], `${label} Tensor.newFull helper`);
  if (nested.new_empty([2, 3]).shape.join("x") !== "2x3") throw new Error(`${label} expected Tensor.new_empty to preserve requested shape`);
  if (nested.new_zeros([2, 3]).shape.join("x") !== "2x3") throw new Error(`${label} expected Tensor.new_zeros to preserve requested shape`);
  if (nested.new_full([2, 3], 5, { requiresGrad: true }).requiresGrad !== true) throw new Error(`${label} expected Tensor.new_full to forward tensor options`);
  expectClose(adapter.empty([2, 2]).data, [0, 0, 0, 0], `${label} empty root helper zero-backed JS allocation`);
  expectClose(adapter.emptyLike(nested).data, [0, 0, 0, 0], `${label} emptyLike root helper`);
  expectClose(adapter.empty_like(nested).data, [0, 0, 0, 0], `${label} empty_like root helper`);
  expectClose(adapter.zerosLike(nested).data, [0, 0, 0, 0], `${label} zerosLike root helper`);
  expectClose(adapter.zeros_like(nested).data, [0, 0, 0, 0], `${label} zeros_like root helper`);
  expectClose(adapter.onesLike(nested).data, [1, 1, 1, 1], `${label} onesLike root helper`);
  expectClose(adapter.ones_like(nested).data, [1, 1, 1, 1], `${label} ones_like root helper`);
  expectClose(adapter.eye(3).data, [1, 0, 0, 0, 1, 0, 0, 0, 1], `${label} eye root helper`);
  expectClose(adapter.fullLike(nested, 7).data, [7, 7, 7, 7], `${label} fullLike root helper`);
  expectClose(adapter.full_like(nested, 8).data, [8, 8, 8, 8], `${label} full_like root helper`);
  const randLikeTensor = adapter.randLike(nested);
  const randLikeAliasTensor = adapter.rand_like(nested);
  const randnLikeTensor = adapter.randnLike(nested);
  const randnLikeAliasTensor = adapter.randn_like(nested);
  const randIntTensor = adapter.randInt(2, 6, [2, 2], { seed: 19 });
  const randintTensor = adapter.randint(6, [2, 2], { seed: 19 });
  const randPermTensor = adapter.randPerm(5, { seed: 19 });
  const randpermTensor = adapter.randperm(5, { seed: 19 });
  if (
    randLikeTensor.shape.join("x") !== "2x2" ||
    randLikeAliasTensor.shape.join("x") !== "2x2" ||
    randnLikeTensor.shape.join("x") !== "2x2" ||
    randnLikeAliasTensor.shape.join("x") !== "2x2" ||
    randIntTensor.shape.join("x") !== "2x2" ||
    randintTensor.shape.join("x") !== "2x2"
  ) {
    throw new Error(`${label} expected random factory root helpers to preserve shape`);
  }
  const sortedRandPerm = Array.from(randPermTensor.data, Number).sort((a, b) => a - b);
  const sortedRandperm = Array.from(randpermTensor.data, Number).sort((a, b) => a - b);
  if (
    randPermTensor.shape.join("x") !== "5" ||
    randpermTensor.shape.join("x") !== "5" ||
    sortedRandPerm.join(",") !== "0,1,2,3,4" ||
    sortedRandperm.join(",") !== "0,1,2,3,4"
  ) {
    throw new Error(`${label} expected randPerm/randperm helpers to produce a full integer permutation`);
  }
  if (
    Array.from(randIntTensor.data).some((value) => {
      const numeric = Number(value);
      return !Number.isInteger(numeric) || numeric < 2 || numeric >= 6;
    }) ||
    Array.from(randintTensor.data).some((value) => {
      const numeric = Number(value);
      return !Number.isInteger(numeric) || numeric < 0 || numeric >= 6;
    })
  ) {
    throw new Error(`${label} expected randInt/randint helpers to produce integer values inside the requested half-open range`);
  }
  expectClose(adapter.Tensor.zerosLike(nested).data, [0, 0, 0, 0], `${label} Tensor.zerosLike static helper`);
  expectClose(adapter.Tensor.empty([2, 2]).data, [0, 0, 0, 0], `${label} Tensor.empty static helper zero-backed JS allocation`);
  expectClose(adapter.Tensor.emptyLike(nested).data, [0, 0, 0, 0], `${label} Tensor.emptyLike static helper`);
  expectClose(adapter.Tensor.empty_like(nested).data, [0, 0, 0, 0], `${label} Tensor.empty_like static helper`);
  expectClose(adapter.Tensor.zeros_like(nested).data, [0, 0, 0, 0], `${label} Tensor.zeros_like static helper`);
  expectClose(adapter.Tensor.onesLike(nested).data, [1, 1, 1, 1], `${label} Tensor.onesLike static helper`);
  expectClose(adapter.Tensor.ones_like(nested).data, [1, 1, 1, 1], `${label} Tensor.ones_like static helper`);
  expectClose(adapter.Tensor.eye(2).data, [1, 0, 0, 1], `${label} Tensor.eye static helper`);
  expectClose(adapter.Tensor.fullLike(nested, 9).data, [9, 9, 9, 9], `${label} Tensor.fullLike static helper`);
  expectClose(adapter.Tensor.full_like(nested, 10).data, [10, 10, 10, 10], `${label} Tensor.full_like static helper`);
  if (
    adapter.Tensor.randLike(nested).shape.join("x") !== "2x2" ||
    adapter.Tensor.rand_like(nested).shape.join("x") !== "2x2" ||
    adapter.Tensor.randInt(2, 6, [2, 2], { seed: 23 }).shape.join("x") !== "2x2" ||
    adapter.Tensor.randint(6, [2, 2], { seed: 23 }).shape.join("x") !== "2x2" ||
    adapter.Tensor.randPerm(5, { seed: 23 }).shape.join("x") !== "5" ||
    adapter.Tensor.randperm(5, { seed: 23 }).shape.join("x") !== "5" ||
    adapter.Tensor.randnLike(nested).shape.join("x") !== "2x2" ||
    adapter.Tensor.randn_like(nested).shape.join("x") !== "2x2"
  ) {
    throw new Error(`${label} expected random factory static helpers to preserve shape`);
  }
  if (
    adapter.zerosLike(nested).shape.join("x") !== "2x2" ||
    adapter.emptyLike(nested).shape.join("x") !== "2x2" ||
    adapter.Tensor.full_like(nested, 3).shape.join("x") !== "2x2" ||
    adapter.eye(3).shape.join("x") !== "3x3"
  ) {
    throw new Error(`${label} expected like-factory helpers to preserve source shape`);
  }
  if (
    adapter.zerosLike(nested, { requiresGrad: true }).requiresGrad !== true ||
    adapter.emptyLike(nested, { requiresGrad: true }).requiresGrad !== true ||
    adapter.randnLike(nested, { requiresGrad: true }).requiresGrad !== true ||
    adapter.randPerm(5, { requiresGrad: true }).requiresGrad !== true ||
    adapter.eye(2, { requiresGrad: true }).requiresGrad !== true
  ) {
    throw new Error(`${label} expected like-factory helpers to honor TensorOptions`);
  }
  expectThrowIncludes(
    () => adapter.eye(0),
    "eye size must be a positive safe integer",
    `${label} eye rejects invalid sizes`,
  );
  expectClose(adapter.reshape(nested, [4]).data, [1, 2, 3, 4], `${label} reshape root helper`);
  expectClose(adapter.concat([nested, nested], 0).data, [1, 2, 3, 4, 1, 2, 3, 4], `${label} concat root helper`);
  expectClose(adapter.concatenate([nested, nested], 1).data, [1, 2, 1, 2, 3, 4, 3, 4], `${label} concatenate root helper`);
  expectClose(adapter.Tensor.concat([nested, nested], 0).data, [1, 2, 3, 4, 1, 2, 3, 4], `${label} Tensor.concat static helper`);
  expectClose(adapter.Tensor.concatenate([nested, nested], 1).data, [1, 2, 1, 2, 3, 4, 3, 4], `${label} Tensor.concatenate static helper`);
  const rowA = adapter.tensor([1, 2, 3], [3]);
  const rowB = adapter.tensor([4, 5, 6], [3]);
  const vstackInput = rowA.requiresGrad_();
  const vstackOutput = adapter.vstack([vstackInput, rowB]);
  if (vstackOutput.shape.join("x") !== "2x3") throw new Error(`${label} expected vstack to turn 1D tensors into rows`);
  expectClose(vstackOutput.data, [1, 2, 3, 4, 5, 6], `${label} vstack root helper`);
  vstackOutput.sum().backward();
  expectClose(vstackInput.grad, [1, 1, 1], `${label} vstack autograd backward`);
  expectClose(adapter.Tensor.vstack([nested, nested]).data, [1, 2, 3, 4, 1, 2, 3, 4], `${label} Tensor.vstack 2D helper`);
  expectClose(adapter.hstack([rowA, rowB]).data, [1, 2, 3, 4, 5, 6], `${label} hstack root 1D helper`);
  expectClose(adapter.Tensor.hstack([nested, nested]).data, [1, 2, 1, 2, 3, 4, 3, 4], `${label} Tensor.hstack 2D helper`);
  const einsumLhs = adapter.tensor([1, 2, 3, 4], [2, 2]).requiresGrad_();
  const einsumRhs = adapter.tensor([5, 6, 7, 8], [2, 2]).requiresGrad_();
  const einsumMatmul = adapter.einsum("ij,jk->ik", [einsumLhs, einsumRhs]);
  if (einsumMatmul.shape.join("x") !== "2x2") throw new Error(`${label} expected einsum matrix product shape`);
  expectClose(einsumMatmul.data, [19, 22, 43, 50], `${label} einsum root matrix product`);
  einsumMatmul.sum().backward();
  expectClose(einsumLhs.grad, [11, 15, 11, 15], `${label} einsum lhs autograd backward`);
  expectClose(einsumRhs.grad, [4, 4, 6, 6], `${label} einsum rhs autograd backward`);
  expectClose(adapter.Tensor.einsum("ii->", nested).data, [5], `${label} Tensor.einsum trace scalar`);
  expectClose(adapter.torch.einsum("ij,jk->ik", einsumLhs.detach(), einsumRhs.detach()).data, [19, 22, 43, 50], `${label} torch.einsum variadic matrix product`);
  const einsumBatchLhs = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]).requiresGrad_();
  const einsumBatchRhs = adapter.tensor([1, 2, 3, 4], [2, 2]).requiresGrad_();
  const einsumEllipsis = adapter.einsum("...ij,jk->...ik", [einsumBatchLhs, einsumBatchRhs]);
  if (einsumEllipsis.shape.join("x") !== "2x2x2") throw new Error(`${label} expected einsum ellipsis batch matmul shape`);
  expectClose(einsumEllipsis.data, [7, 10, 15, 22, 23, 34, 31, 46], `${label} einsum ellipsis batch matmul`);
  einsumEllipsis.sum().backward();
  expectClose(einsumBatchLhs.grad, [3, 7, 3, 7, 3, 7, 3, 7], `${label} einsum ellipsis backward lhs`);
  expectClose(einsumBatchRhs.grad, [16, 16, 20, 20], `${label} einsum ellipsis backward rhs`);
  expectClose(adapter.Tensor.einsum("...i->...", adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3])).data, [6, 15], `${label} Tensor.einsum implicit ellipsis reduction`);
  const indexSelectInput = adapter.tensor([1, 2, 3, 4, 5, 6], [3, 2]).requiresGrad_();
  const indexSelectOutput = indexSelectInput.indexSelect(0, [2, 0, 2]);
  if (indexSelectOutput.shape.join("x") !== "3x2") throw new Error(`${label} expected indexSelect to replace the selected axis length`);
  expectClose(indexSelectOutput.data, [5, 6, 1, 2, 5, 6], `${label} Tensor.indexSelect repeated rows`);
  indexSelectOutput.sum().backward();
  expectClose(indexSelectInput.grad, [1, 1, 0, 0, 2, 2], `${label} Tensor.indexSelect repeated-index autograd backward`);
  expectClose(adapter.indexSelect(nested, 1, [1, 0]).data, [2, 1, 4, 3], `${label} indexSelect root helper`);
  expectClose(adapter.index_select(nested, 0, adapter.tensor([1, 0], [2])).data, [3, 4, 1, 2], `${label} index_select Tensor indices root helper`);
  expectClose(adapter.Tensor.indexSelect(nested, 1, new Uint32Array([1, 0])).data, [2, 1, 4, 3], `${label} Tensor.indexSelect typed-array indices`);
  expectThrowIncludes(
    () => adapter.indexSelect(nested, 0, [2]),
    "out of range",
    `${label} indexSelect rejects out-of-range indices`,
  );
  const gatherInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const gatherIndex = adapter.tensor([2, 1, 0, 0], [2, 2]);
  const gatherOutput = gatherInput.gather(1, gatherIndex);
  if (gatherOutput.shape.join("x") !== "2x2") throw new Error(`${label} expected gather output shape to match index shape`);
  expectClose(gatherOutput.data, [3, 2, 4, 4], `${label} Tensor.gather per-row index output`);
  gatherOutput.sum().backward();
  expectClose(gatherInput.grad, [0, 1, 1, 2, 0, 0], `${label} Tensor.gather repeated-index autograd backward`);
  expectClose(adapter.gather(nested, 1, adapter.tensor([1, 0, 0, 1], [2, 2])).data, [2, 1, 3, 4], `${label} gather root helper`);
  expectClose(adapter.Tensor.gather(nested, 0, adapter.tensor([1, 1, 0, 0], [2, 2])).data, [3, 4, 1, 2], `${label} Tensor.gather static helper`);
  expectThrowIncludes(
    () => adapter.gather(nested, 1, adapter.tensor([2], [1])),
    "rank",
    `${label} gather rejects rank-mismatched index`,
  );
  const takeInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const takeIndex = adapter.tensor([5, 0, 3, 3], [2, 2]);
  const takeOutput = takeInput.take(takeIndex);
  if (takeOutput.shape.join("x") !== "2x2") throw new Error(`${label} expected take output shape to match index shape`);
  expectClose(takeOutput.data, [6, 1, 4, 4], `${label} Tensor.take flattened index output`);
  takeOutput.sum().backward();
  expectClose(takeInput.grad, [1, 0, 0, 2, 0, 1], `${label} Tensor.take repeated-index autograd backward`);
  expectClose(adapter.take(nested, [3, 0]).data, [4, 1], `${label} take root helper`);
  expectClose(adapter.Tensor.take(nested, new Uint32Array([2, 1])).data, [3, 2], `${label} Tensor.take static typed-array helper`);
  expectClose(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]).argsort(-1).data, [1, 2, 0, 0, 2, 1], `${label} Tensor.argsort stable last-dim helper`);
  expectClose(adapter.argsort(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]), 1, true).data, [0, 2, 1, 1, 2, 0], `${label} argsort root descending helper`);
  expectClose(adapter.Tensor.argsort(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]), 0).data, [1, 0, 0, 0, 1, 1], `${label} Tensor.argsort static dim0 helper`);
  expectThrowIncludes(
    () => adapter.argsort(nested, 1, 1 as unknown as boolean),
    "descending",
    `${label} argsort rejects non-boolean descending`,
  );
  const sortOutput = adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]).sort(-1);
  expectClose(sortOutput.values.data, [1, 2, 3, 2, 7, 9], `${label} Tensor.sort stable last-dim values`);
  expectClose(sortOutput.indices.data, [1, 2, 0, 0, 2, 1], `${label} Tensor.sort stable last-dim indices`);
  const sortDesc = adapter.sort(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]), 1, true);
  expectClose(sortDesc.values.data, [3, 2, 1, 9, 7, 2], `${label} sort root descending values`);
  expectClose(sortDesc.indices.data, [0, 2, 1, 1, 2, 0], `${label} sort root descending indices`);
  expectClose(adapter.Tensor.sort(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]), 0).values.data, [2, 1, 2, 3, 9, 7], `${label} Tensor.sort static dim0 values`);
  const sortGradInput = adapter.tensor([3, 1, 2], [3]).requiresGrad_();
  sortGradInput.sort().values.mul(adapter.tensor([10, 20, 30], [3])).sum().backward();
  expectClose(sortGradInput.grad, [30, 10, 20], `${label} Tensor.sort values autograd backward`);
  expectThrowIncludes(
    () => adapter.sort(nested, 1, 1 as unknown as boolean),
    "descending",
    `${label} sort rejects non-boolean descending`,
  );
  const topkOutput = adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]).topk(2, -1);
  if (topkOutput.values.shape.join("x") !== "2x2") throw new Error(`${label} expected topk to replace selected axis length`);
  expectClose(topkOutput.values.data, [3, 2, 9, 7], `${label} Tensor.topk largest sorted values`);
  expectClose(topkOutput.indices.data, [0, 2, 1, 2], `${label} Tensor.topk largest sorted indices`);
  const topkSmallest = adapter.topk(adapter.tensor([3, 1, 2, 2, 9, 7], [2, 3]), 1, 1, false);
  expectClose(topkSmallest.values.data, [1, 2], `${label} topk root smallest values`);
  expectClose(topkSmallest.indices.data, [1, 0], `${label} topk root smallest indices`);
  const topkUnsorted = adapter.Tensor.topk(adapter.tensor([2, 3, 1, 7, 9, 2], [2, 3]), 2, 1, true, false);
  expectClose(topkUnsorted.values.data, [2, 3, 7, 9], `${label} Tensor.topk static unsorted values preserve original order`);
  expectClose(topkUnsorted.indices.data, [0, 1, 0, 1], `${label} Tensor.topk static unsorted indices preserve original order`);
  const topkGradInput = adapter.tensor([3, 1, 2], [3]).requiresGrad_();
  topkGradInput.topk(2).values.mul(adapter.tensor([10, 20], [2])).sum().backward();
  expectClose(topkGradInput.grad, [10, 0, 20], `${label} Tensor.topk values autograd backward`);
  expectThrowIncludes(
    () => adapter.topk(nested, 4, 1),
    "must be <=",
    `${label} topk rejects k larger than selected dim`,
  );
  expectThrowIncludes(
    () => adapter.topk(nested, 1, 1, "yes" as unknown as boolean),
    "largest",
    `${label} topk rejects non-boolean largest`,
  );
  expectThrowIncludes(
    () => adapter.take(nested, [4]),
    "out of range",
    `${label} take rejects out-of-range indices`,
  );
  const scatterBase = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const scatterSrc = adapter.tensor([10, 20, 30, 40], [2, 2]).requiresGrad_();
  const scatterIndex = adapter.tensor([2, 1, 0, 0], [2, 2]);
  const scatterOutput = scatterBase.scatterAdd(1, scatterIndex, scatterSrc);
  if (scatterOutput.shape.join("x") !== "2x3") throw new Error(`${label} expected scatterAdd output shape to match input shape`);
  expectClose(scatterOutput.data, [1, 22, 13, 74, 5, 6], `${label} Tensor.scatterAdd per-row additions`);
  scatterOutput.sum().backward();
  expectClose(scatterBase.grad, [1, 1, 1, 1, 1, 1], `${label} Tensor.scatterAdd input autograd backward`);
  expectClose(scatterSrc.grad, [1, 1, 1, 1], `${label} Tensor.scatterAdd source autograd backward`);
  expectClose(adapter.scatterAdd(nested, 1, adapter.tensor([1, 0, 0, 1], [2, 2]), 10).data, [11, 12, 13, 14], `${label} scatterAdd root scalar helper`);
  expectClose(adapter.scatter_add(nested, 0, adapter.tensor([1, 1, 0, 0], [2, 2]), [1, 2, 3, 4]).data, [4, 6, 4, 6], `${label} scatter_add root array helper`);
  expectClose(adapter.Tensor.scatterAdd(nested, 1, adapter.tensor([1, 0, 0, 1], [2, 2]), new Float32Array([1, 2, 3, 4])).data, [3, 3, 6, 8], `${label} Tensor.scatterAdd typed-array source`);
  expectThrowIncludes(
    () => adapter.scatterAdd(nested, 1, adapter.tensor([0, 1], [1, 2]), [1]),
    "source length",
    `${label} scatterAdd rejects source length mismatch`,
  );
  if (adapter.Tensor.view(nested, [1, 4]).shape.join("x") !== "1x4") {
    throw new Error(`${label} expected Tensor.view static helper to preserve requested shape`);
  }
  const shapeSource = adapter.tensor(new Float32Array(6), [3, 2]);
  const reshapeAsInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const reshapeAsOutput = reshapeAsInput.reshape_as(shapeSource);
  if (reshapeAsOutput.shape.join("x") !== "3x2") throw new Error(`${label} expected Tensor.reshape_as to copy target shape`);
  expectClose(reshapeAsOutput.data, [1, 2, 3, 4, 5, 6], `${label} Tensor.reshape_as helper`);
  const viewAsOutput = reshapeAsInput.view_as(shapeSource);
  if (viewAsOutput.shape.join("x") !== "3x2") throw new Error(`${label} expected Tensor.view_as to copy target shape`);
  const expandAsInput = adapter.tensor([1, 2, 3], [1, 3]).requiresGrad_();
  const expandAsOutput = expandAsInput.expand_as(adapter.tensor(new Float32Array(6), [2, 3]));
  expectClose(expandAsOutput.data, [1, 2, 3, 1, 2, 3], `${label} Tensor.expand_as helper`);
  expandAsOutput.sum().backward();
  expectClose(expandAsInput.grad, [2, 2, 2], `${label} Tensor.expand_as autograd backward`);
  const repeatInput = adapter.tensor([1, 2, 3, 4], [2, 2]).requiresGrad_();
  const repeatOutput = repeatInput.repeat(2, 3);
  if (repeatOutput.shape.join("x") !== "4x6") throw new Error(`${label} expected Tensor.repeat to multiply shape dimensions`);
  expectClose(repeatOutput.data, [
    1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4,
    1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4,
  ], `${label} Tensor.repeat helper`);
  repeatOutput.sum().backward();
  expectClose(repeatInput.grad, [6, 6, 6, 6], `${label} Tensor.repeat autograd backward`);
  expectClose(adapter.tile(adapter.tensor([5, 6], [2]), [2, 2]).data, [5, 6, 5, 6, 5, 6, 5, 6], `${label} tile root helper`);
  expectClose(adapter.Tensor.repeat(adapter.tensor([7, 8], [2]), 3).data, [7, 8, 7, 8, 7, 8], `${label} Tensor.repeat static helper`);
  expectClose(nested.mT.data, [1, 3, 2, 4], `${label} Tensor.mT matrix transpose`);
  if (nested.mT.shape.join("x") !== "2x2") throw new Error(`${label} expected Tensor.mT to preserve 2D transposed shape`);
  const batchedMatrix = adapter.tensor([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
  ], [2, 2, 3]);
  if (batchedMatrix.mT.shape.join("x") !== "2x3x2") {
    throw new Error(`${label} expected Tensor.mT to transpose trailing matrix dimensions`);
  }
  expectClose(batchedMatrix.mT.data, [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12], `${label} Tensor.mT batched matrix data`);
  const flipInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const flipOutput = flipInput.flip([0, -1]);
  if (flipOutput.shape.join("x") !== "2x3") throw new Error(`${label} expected Tensor.flip to preserve shape`);
  expectClose(flipOutput.data, [6, 5, 4, 3, 2, 1], `${label} Tensor.flip helper`);
  flipOutput.sum().backward();
  expectClose(flipInput.grad, [1, 1, 1, 1, 1, 1], `${label} Tensor.flip autograd backward`);
  expectClose(adapter.flip(adapter.tensor([1, 2, 3, 4], [2, 2]), [1]).data, [2, 1, 4, 3], `${label} flip root helper`);
  expectClose(adapter.Tensor.flip(adapter.tensor([1, 2, 3, 4], [2, 2]), [0]).data, [3, 4, 1, 2], `${label} Tensor.flip static helper`);
  const rollInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const rollOutput = rollInput.roll(1, 1);
  if (rollOutput.shape.join("x") !== "2x3") throw new Error(`${label} expected Tensor.roll to preserve shape`);
  expectClose(rollOutput.data, [3, 1, 2, 6, 4, 5], `${label} Tensor.roll helper`);
  rollOutput.sum().backward();
  expectClose(rollInput.grad, [1, 1, 1, 1, 1, 1], `${label} Tensor.roll autograd backward`);
  expectClose(adapter.roll(adapter.tensor([1, 2, 3, 4], [2, 2]), [1, -1], [0, 1]).data, [4, 3, 2, 1], `${label} roll root helper`);
  expectClose(adapter.Tensor.roll(adapter.tensor([1, 2, 3, 4], [2, 2]), 1).data, [4, 1, 2, 3], `${label} Tensor.roll static flattened helper`);
  const splitInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  const splitParts = splitInput.split([1, 2], 1);
  if (splitParts.length !== 2 || splitParts[0].shape.join("x") !== "2x1" || splitParts[1].shape.join("x") !== "2x2") {
    throw new Error(`${label} expected Tensor.split to preserve section shapes`);
  }
  expectClose(splitParts[0].data, [1, 4], `${label} Tensor.split first section`);
  expectClose(splitParts[1].data, [2, 3, 5, 6], `${label} Tensor.split second section`);
  splitParts[1].sum().backward();
  expectClose(splitInput.grad, [0, 1, 1, 0, 1, 1], `${label} Tensor.split autograd backward`);
  const chunkParts = adapter.chunk(adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]), 2, 1);
  expectClose(chunkParts[0].data, [1, 2, 4, 5], `${label} chunk root first part`);
  expectClose(chunkParts[1].data, [3, 6], `${label} chunk root second part`);
  const unboundRows = adapter.Tensor.unbind(adapter.tensor([1, 2, 3, 4], [2, 2]), 0);
  expectClose(unboundRows[0].data, [1, 2], `${label} Tensor.unbind first row`);
  expectClose(unboundRows[1].data, [3, 4], `${label} Tensor.unbind second row`);
  expectClose(adapter.split(adapter.tensor([1, 2, 3, 4], [4]), 2, 0)[1].data, [3, 4], `${label} split root fixed size`);
  expectClose(adapter.clone(nested).data, [1, 2, 3, 4], `${label} clone root helper`);
  const inplaceTensor = adapter.clone(nested);
  if (inplaceTensor.fill_(5) !== inplaceTensor) throw new Error(`${label} expected Tensor.fill_ to return this`);
  expectClose(inplaceTensor.data, [5, 5, 5, 5], `${label} Tensor.fill_ mutates data`);
  if (inplaceTensor.zero_() !== inplaceTensor) throw new Error(`${label} expected Tensor.zero_ to return this`);
  expectClose(inplaceTensor.data, [0, 0, 0, 0], `${label} Tensor.zero_ mutates data`);
  if (inplaceTensor.ones_() !== inplaceTensor) throw new Error(`${label} expected Tensor.ones_ to return this`);
  expectClose(inplaceTensor.data, [1, 1, 1, 1], `${label} Tensor.ones_ mutates data`);
  if (inplaceTensor.copy_(nested) !== inplaceTensor) throw new Error(`${label} expected Tensor.copy_ to return this`);
  expectClose(inplaceTensor.data, [1, 2, 3, 4], `${label} Tensor.copy_ copies tensor data`);
  inplaceTensor.copy_(new Float32Array([4, 3, 2, 1]));
  expectClose(inplaceTensor.data, [4, 3, 2, 1], `${label} Tensor.copy_ copies Float32Array data`);
  if (inplaceTensor.add_(1) !== inplaceTensor) throw new Error(`${label} expected Tensor.add_ to return this`);
  expectClose(inplaceTensor.data, [5, 4, 3, 2], `${label} Tensor.add_ mutates data`);
  if (inplaceTensor.sub_(adapter.tensor([1, 1], [1, 2])) !== inplaceTensor) throw new Error(`${label} expected Tensor.sub_ to return this`);
  expectClose(inplaceTensor.data, [4, 3, 2, 1], `${label} Tensor.sub_ mutates data with same-output broadcast`);
  if (inplaceTensor.mul_(2) !== inplaceTensor) throw new Error(`${label} expected Tensor.mul_ to return this`);
  expectClose(inplaceTensor.data, [8, 6, 4, 2], `${label} Tensor.mul_ mutates data`);
  if (inplaceTensor.div_(adapter.tensor([2, 2], [1, 2])) !== inplaceTensor) throw new Error(`${label} expected Tensor.div_ to return this`);
  expectClose(inplaceTensor.data, [4, 3, 2, 1], `${label} Tensor.div_ mutates data with same-output broadcast`);
  expectThrowIncludes(() => adapter.tensor([1, 2, 3], [3]).add_(adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3])), "Tensor.add_ result shape [2,3] must match tensor shape [3]", `${label} Tensor.add_ rejects broadcast shape expansion`);
  expectThrowIncludes(() => inplaceTensor.fill_(Number.NaN), "Tensor.fill_ value must be a finite number", `${label} Tensor.fill_ rejects NaN`);
  expectThrowIncludes(() => inplaceTensor.copy_([1, 2]), "Tensor.copy_ source length must be 4, got 2", `${label} Tensor.copy_ rejects wrong length`);
  expectClose(adapter.Tensor.detach(nested).data, [1, 2, 3, 4], `${label} Tensor.detach static helper`);
  expectClose(adapter.eq(nested, nested).data, [1, 1, 1, 1], `${label} eq root helper`);
  expectClose(adapter.Tensor.ge(nested, 3).data, [0, 0, 1, 1], `${label} Tensor.ge static helper`);
  expectClose(adapter.neg(nested).data, [-1, -2, -3, -4], `${label} neg root helper`);
  expectClose(adapter.negative(nested).data, [-1, -2, -3, -4], `${label} negative root helper`);
  const negativeInput = adapter.tensor([2], [1]).requiresGrad_();
  const negativeOutput = negativeInput.negative();
  negativeOutput.backward();
  expectClose(negativeInput.grad, [-1], `${label} Tensor.negative autograd backward`);
  expectClose(adapter.expm1(adapter.tensor([0, 1], [2])).data, [0, Math.expm1(1)], `${label} expm1 root helper`);
  expectClose(adapter.Tensor.log1p(adapter.tensor([0, 1], [2])).data, [0, Math.log1p(1)], `${label} Tensor.log1p static helper`);
  const log1pInput = adapter.tensor([0], [1]).requiresGrad_();
  const log1pOutput = log1pInput.log1p();
  log1pOutput.backward();
  expectClose(log1pInput.grad, [1], `${label} Tensor.log1p autograd backward`);
  const expm1Input = adapter.tensor([0], [1]).requiresGrad_();
  const expm1Output = expm1Input.expm1();
  expm1Output.backward();
  expectClose(expm1Input.grad, [1], `${label} Tensor.expm1 autograd backward`);
  expectClose(adapter.Tensor.sqrt(nested).data, [1, Math.sqrt(2), Math.sqrt(3), 2], `${label} Tensor.sqrt static helper`);
  expectClose(adapter.rsqrt(adapter.tensor([1, 4], [2])).data, [1, 0.5], `${label} rsqrt root helper`);
  expectClose(adapter.Tensor.rsqrt(adapter.tensor([1, 4], [2])).data, [1, 0.5], `${label} Tensor.rsqrt static helper`);
  expectClose(adapter.Tensor.reciprocal(adapter.tensor([2, 4], [2])).data, [0.5, 0.25], `${label} Tensor.reciprocal static helper`);
  const reciprocalInput = adapter.tensor([2], [1]).requiresGrad_();
  const reciprocalOutput = reciprocalInput.reciprocal();
  reciprocalOutput.backward();
  expectClose(reciprocalInput.grad, [-0.25], `${label} Tensor.reciprocal autograd backward`);
  const rsqrtInput = adapter.tensor([4], [1]).requiresGrad_();
  const rsqrtOutput = rsqrtInput.rsqrt();
  rsqrtOutput.backward();
  expectClose(rsqrtInput.grad, [-0.0625], `${label} Tensor.rsqrt autograd backward`);
  const finiteProbe = adapter.tensor([Number.NaN, Infinity, -Infinity, 3], [2, 2]);
  expectClose(adapter.isnan(finiteProbe).data, [1, 0, 0, 0], `${label} isnan root helper`);
  expectClose(adapter.Tensor.isinf(finiteProbe).data, [0, 1, 1, 0], `${label} Tensor.isinf static helper`);
  expectClose(finiteProbe.isfinite().data, [0, 0, 0, 1], `${label} Tensor.isfinite instance helper`);
  const integralProbe = adapter.tensor([-1.2, -1, 0.2, 1.8], [2, 2]).requiresGrad_();
  expectClose(adapter.floor(integralProbe).data, [-2, -1, 0, 1], `${label} floor root helper`);
  expectClose(adapter.Tensor.ceil(integralProbe).data, [-1, -1, 1, 2], `${label} Tensor.ceil static helper`);
  expectClose(adapter.round(integralProbe).data, [-1, -1, 0, 2], `${label} round root helper`);
  expectClose(adapter.Tensor.trunc(integralProbe).data, [-1, -1, 0, 1], `${label} Tensor.trunc static helper`);
  const floorLoss = integralProbe.floor().sum();
  floorLoss.backward();
  expectClose(integralProbe.grad, [0, 0, 0, 0], `${label} Tensor.floor zero gradient`);
  const roundProbe = adapter.tensor([-1.2, 1.8], [2]).requiresGrad_();
  const roundLoss = roundProbe.round().sum();
  roundLoss.backward();
  expectClose(roundProbe.grad, [0, 0], `${label} Tensor.round zero gradient`);
  expectClose(adapter.relu(adapter.tensor([-1, 0, 2], [3])).data, [0, 0, 2], `${label} relu root helper`);
  expectClose(adapter.Tensor.sigmoid(adapter.tensor([0], [1])).data, [0.5], `${label} Tensor.sigmoid static helper`);
  expectClose(adapter.tanh(adapter.tensor([0, 1], [2])).data, [0, Math.tanh(1)], `${label} tanh root helper`);
  expectClose(adapter.Tensor.tanh(adapter.tensor([0, 1], [2])).data, [0, Math.tanh(1)], `${label} Tensor.tanh static helper`);
  expectClose(adapter.sin(adapter.tensor([0, Math.PI / 2], [2])).data, [0, 1], `${label} sin root helper`);
  expectClose(adapter.Tensor.cos(adapter.tensor([0, Math.PI], [2])).data, [1, -1], `${label} Tensor.cos static helper`);
  expectClose(adapter.tan(adapter.tensor([0, Math.PI / 4], [2])).data, [0, 1], `${label} tan root helper`);
  const trigInput = adapter.tensor([0], [1]).requiresGrad_();
  const trigOutput = trigInput.sin();
  expectClose(trigOutput.data, [0], `${label} Tensor.sin autograd forward`);
  trigOutput.backward();
  expectClose(trigInput.grad, [1], `${label} Tensor.sin autograd backward`);
  expectNumericalGradient(
    adapter,
    [0.25, -0.75],
    [2],
    (input) => input.sin().sum(),
    `${label} Tensor.sin numerical gradient`,
  );
  const tanInput = adapter.tensor([0], [1]).requiresGrad_();
  const tanOutput = tanInput.tan();
  tanOutput.backward();
  expectClose(tanInput.grad, [1], `${label} Tensor.tan autograd backward`);
  const tanhInput = adapter.tensor([0], [1]).requiresGrad_();
  const tanhOutput = tanhInput.tanh();
  expectClose(tanhOutput.data, [0], `${label} Tensor.tanh autograd forward`);
  tanhOutput.backward();
  expectClose(tanhInput.grad, [1], `${label} Tensor.tanh autograd backward`);
  expectNumericalGradient(
    adapter,
    [-0.5, 0.75],
    [2],
    (input) => input.tanh().sum(),
    `${label} Tensor.tanh numerical gradient`,
  );
  expectClose(adapter.sum(nested, 1).data, [3, 7], `${label} sum root helper`);
  expectClose(adapter.prod(nested, 1).data, [2, 12], `${label} prod root helper`);
  expectClose(adapter.cumsum(nested, 1).data, [1, 3, 3, 7], `${label} cumsum root helper`);
  expectClose(adapter.Tensor.mean(nested, 0).data, [2, 3], `${label} Tensor.mean static helper`);
  expectClose(adapter.Tensor.prod(nested, 0).data, [3, 8], `${label} Tensor.prod static helper`);
  expectClose(adapter.Tensor.cumsum(nested, 0).data, [1, 2, 4, 6], `${label} Tensor.cumsum static helper`);
  expectClose(adapter.max(nested, 1).data, [2, 4], `${label} max root helper`);
  expectClose(adapter.argmax(nested, 1).data, [1, 1], `${label} argmax root helper`);
  expectClose(adapter.variance(adapter.tensor([1, 2, 3, 4], [4])).data, [1.25], `${label} variance root helper`);
  expectClose(adapter.std(adapter.tensor([1, 2, 3, 4], [4])).data, [Math.sqrt(1.25)], `${label} std root helper`);
  expectClose(adapter.norm(adapter.tensor([3, 4], [2])).data, [5], `${label} norm root helper`);
  const varianceDim = nested.variance(1);
  if (varianceDim.shape.join("x") !== "2x1") throw new Error(`${label} Tensor.variance dim shape mismatch`);
  expectClose(varianceDim.data, [0.25, 0.25], `${label} Tensor.variance dim values`);
  expectClose(adapter.Tensor.var(nested, 1, 1).data, [0.5, 0.5], `${label} Tensor.var correction values`);
  expectClose(adapter.Tensor.std(nested, 1, 1).data, [Math.SQRT1_2, Math.SQRT1_2], `${label} Tensor.std correction values`);
  const normDim = nested.norm(1);
  if (normDim.shape.join("x") !== "2x1") throw new Error(`${label} Tensor.norm dim shape mismatch`);
  expectClose(normDim.data, [Math.sqrt(5), 5], `${label} Tensor.norm dim values`);
  expectClose(adapter.Tensor.norm(nested, 0).data, [Math.sqrt(10), Math.sqrt(20)], `${label} Tensor.norm static dim values`);
  expectClose(adapter.tensor([1, -2, 3], [3]).norm(undefined, 1).data, [6], `${label} Tensor.norm p=1 values`);
  const varianceGradInput = adapter.tensor([1, 2, 3], [3]).requiresGrad_();
  varianceGradInput.variance().backward();
  expectClose(varianceGradInput.grad, [-0.666667, 0, 0.666667], `${label} Tensor.variance autograd backward`);
  const stdGradInput = adapter.tensor([1, 2, 3], [3]).requiresGrad_();
  stdGradInput.std().backward();
  expectClose(stdGradInput.grad, [-0.408248, 0, 0.408248], `${label} Tensor.std autograd backward`);
  const normGradInput = adapter.tensor([3, 4], [2]).requiresGrad_();
  normGradInput.norm().backward();
  expectClose(normGradInput.grad, [0.6, 0.8], `${label} Tensor.norm autograd backward`);
  const prodGradInput = adapter.tensor([2, 3, 4], [3]).requiresGrad_();
  prodGradInput.prod().backward();
  expectClose(prodGradInput.grad, [12, 8, 6], `${label} Tensor.prod autograd backward`);
  const prodZeroGradInput = adapter.tensor([2, 0, 4], [3]).requiresGrad_();
  prodZeroGradInput.prod().backward();
  expectClose(prodZeroGradInput.grad, [0, 8, 0], `${label} Tensor.prod zero autograd backward`);
  const cumsumGradInput = adapter.tensor([1, 2, 3], [3]).requiresGrad_();
  cumsumGradInput.cumsum(0).sum().backward();
  expectClose(cumsumGradInput.grad, [3, 2, 1], `${label} Tensor.cumsum autograd backward`);
  expectClose(adapter.softmax(adapter.tensor([1, 2], [2])).data, [0.268941, 0.731059], `${label} softmax root helper`);
  expectClose(adapter.softmax(adapter.tensor([1, 2], [2]), -1).data, [0.268941, 0.731059], `${label} softmax root dim helper`);
  expectClose(adapter.softmax_dim(adapter.tensor([1, 2], [2]), -1).data, [0.268941, 0.731059], `${label} softmax_dim root alias`);
  expectClose(adapter.tensor([1, 2], [2]).softmax(-1).data, [0.268941, 0.731059], `${label} Tensor.softmax dim alias`);
  expectClose(adapter.tensor([1, 2], [2]).softmax_dim(-1).data, [0.268941, 0.731059], `${label} Tensor.softmax_dim instance alias`);
  expectClose(adapter.Tensor.logSoftmax(adapter.tensor([1, 2], [2])).data, [-1.313262, -0.313262], `${label} Tensor.logSoftmax static helper`);
  expectClose(adapter.Tensor.logSoftmax(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} Tensor.logSoftmax static dim helper`);
  expectClose(adapter.Tensor.log_softmax(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} Tensor.log_softmax static alias`);
  expectClose(adapter.Tensor.log_softmax_dim(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} Tensor.log_softmax_dim static alias`);
  expectClose(adapter.log_softmax(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} log_softmax root alias`);
  expectClose(adapter.log_softmax_dim(adapter.tensor([1, 2], [2]), -1).data, [-1.313262, -0.313262], `${label} log_softmax_dim root alias`);
  expectClose(adapter.tensor([1, 2], [2]).logSoftmax(-1).data, [-1.313262, -0.313262], `${label} Tensor.logSoftmax dim alias`);
  expectClose(adapter.tensor([1, 2], [2]).log_softmax(-1).data, [-1.313262, -0.313262], `${label} Tensor.log_softmax instance alias`);
  expectClose(adapter.tensor([1, 2], [2]).log_softmax_dim(-1).data, [-1.313262, -0.313262], `${label} Tensor.log_softmax_dim instance alias`);
  expectClose(adapter.logsumexp(adapter.tensor([1, 2], [2])).data, [2.313262], `${label} logsumexp root helper`);
  expectClose(adapter.tensor([1, 2], [2]).logSumExp().data, [2.313262], `${label} Tensor.logSumExp instance alias`);
  const logsumexpDim = adapter.tensor([[1, 2], [3, 4]], [2, 2]).logsumexp(1);
  if (logsumexpDim.shape.join("x") !== "2x1") throw new Error(`${label} Tensor.logsumexp dim shape mismatch`);
  expectClose(logsumexpDim.data, [2.313262, 4.313262], `${label} Tensor.logsumexp dim values`);
  const logSumExpDim0 = adapter.Tensor.logSumExp(adapter.tensor([[1, 2], [3, 4]], [2, 2]), 0);
  if (logSumExpDim0.shape.join("x") !== "1x2") throw new Error(`${label} Tensor.logSumExp dim0 shape mismatch`);
  expectClose(logSumExpDim0.data, [3.126928, 4.126928], `${label} Tensor.logSumExp dim0 values`);
  const logsumexpGradInput = adapter.tensor([1, 2], [2]).requiresGrad_();
  logsumexpGradInput.logsumexp().backward();
  expectClose(logsumexpGradInput.grad, [0.268941, 0.731059], `${label} Tensor.logsumexp autograd backward`);
  expectNumericalGradient(
    adapter,
    [-0.5, 0.25, 1.25],
    [3],
    (input) => input.logsumexp(),
    `${label} Tensor.logsumexp numerical gradient`,
  );
  expectClose(adapter.matmul(nested, adapter.tensor([1, 2, 3, 4], [2, 2])).data, [7, 10, 15, 22], `${label} matmul root helper`);
  expectClose(adapter.mm(nested, adapter.tensor([1, 2, 3, 4], [2, 2])).data, [7, 10, 15, 22], `${label} mm root helper`);
  expectClose(adapter.dot(adapter.tensor([1, 2, 3], [3]), adapter.tensor([4, 5, 6], [3])).data, [32], `${label} dot root helper`);
  expectClose(adapter.trace(adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3])).data, [6], `${label} trace root helper`);
  expectClose(adapter.diagonal(adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3])).data, [1, 5], `${label} diagonal root helper`);
  expectClose(adapter.Tensor.matmul(nested, adapter.tensor([1, 2, 3, 4], [2, 2])).data, [7, 10, 15, 22], `${label} Tensor.matmul static helper`);
  expectClose(adapter.Tensor.mm(nested, adapter.tensor([1, 2, 3, 4], [2, 2])).data, [7, 10, 15, 22], `${label} Tensor.mm static helper`);
  expectClose(adapter.Tensor.dot(adapter.tensor([1, 2, 3], [3]), adapter.tensor([4, 5, 6], [3])).data, [32], `${label} Tensor.dot static helper`);
  expectClose(adapter.Tensor.trace(nested).data, [5], `${label} Tensor.trace static helper`);
  expectClose(adapter.Tensor.diagonal(nested).data, [1, 4], `${label} Tensor.diagonal static helper`);
  const dotGradLhs = adapter.tensor([1, 2, 3], [3]).requiresGrad_();
  const dotGradRhs = adapter.tensor([4, 5, 6], [3]).requiresGrad_();
  dotGradLhs.dot(dotGradRhs).backward();
  expectClose(dotGradLhs.grad, [4, 5, 6], `${label} Tensor.dot lhs autograd backward`);
  expectClose(dotGradRhs.grad, [1, 2, 3], `${label} Tensor.dot rhs autograd backward`);
  const traceGradInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  traceGradInput.trace().backward();
  expectClose(traceGradInput.grad, [1, 0, 0, 0, 1, 0], `${label} Tensor.trace autograd backward`);
  const diagonalGradInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]).requiresGrad_();
  diagonalGradInput.diagonal().sum().backward();
  expectClose(diagonalGradInput.grad, [1, 0, 0, 0, 1, 0], `${label} Tensor.diagonal autograd backward`);
  const bmmLhs = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
  const bmmRhs = adapter.tensor([1, 0, 0, 1, 2, 0, 0, 2], [2, 2, 2]);
  expectClose(bmmLhs.bmm(bmmRhs).data, [1, 2, 3, 4, 10, 12, 14, 16], `${label} Tensor.bmm instance helper`);
  expectClose(adapter.bmm(bmmLhs, bmmRhs).data, [1, 2, 3, 4, 10, 12, 14, 16], `${label} bmm root helper`);
  expectClose(adapter.Tensor.bmm(bmmLhs, bmmRhs).data, [1, 2, 3, 4, 10, 12, 14, 16], `${label} Tensor.bmm static helper`);
  const bmmGradLhs = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]).requiresGrad_();
  const bmmGradRhs = adapter.tensor([1, 0, 0, 1, 2, 0, 0, 2], [2, 2, 2]).requiresGrad_();
  bmmGradLhs.bmm(bmmGradRhs).sum().backward();
  expectClose(bmmGradLhs.grad, [1, 1, 1, 1, 2, 2, 2, 2], `${label} Tensor.bmm lhs autograd backward`);
  expectClose(bmmGradRhs.grad, [4, 4, 6, 6, 12, 12, 14, 14], `${label} Tensor.bmm rhs autograd backward`);
  expectClose(nested.pow(2).data, [1, 4, 9, 16], `${label} Tensor.pow scalar exponent`);
  expectClose(adapter.pow(nested, 0.5).data, [1, Math.sqrt(2), Math.sqrt(3), 2], `${label} pow root helper`);
  expectClose(nested.maximum(adapter.tensor([2, 3], [1, 2])).data, [2, 3, 3, 4], `${label} Tensor.maximum broadcast`);
  expectClose(adapter.minimum(nested, 2.5).data, [1, 2, 2.5, 2.5], `${label} minimum scalar helper`);
  expectClose(nested.isclose(adapter.tensor([1, 2.00001], [1, 2]), { rtol: 1e-4 }).data, [1, 1, 0, 0], `${label} Tensor.isclose broadcast helper`);
  expectClose(adapter.isclose(adapter.tensor([Number.NaN, 1], [2]), adapter.tensor([Number.NaN, 1.1], [2]), { equal_nan: true, atol: 0.2 }).data, [1, 1], `${label} isclose root helper`);
  expectClose(adapter.Tensor.isclose(nested, adapter.tensor([1, 4], [1, 2]), { atol: 0 }).data, [1, 0, 0, 1], `${label} Tensor.isclose static helper`);
  expectClose(nested.gt(2).where(nested, -1).data, [-1, -1, 3, 4], `${label} Tensor.where mask`);
  expectClose(adapter.where(nested.lt(adapter.tensor([2, 4], [1, 2])), nested, 0).data, [1, 2, 0, 0], `${label} where broadcast helper`);
  expectClose(nested.maskedFill(nested.gt(2), -1).data, [1, 2, -1, -1], `${label} Tensor.maskedFill scalar helper`);
  expectClose(adapter.masked_fill(nested, adapter.tensor([0, 1], [1, 2]), 9).data, [1, 9, 3, 9], `${label} masked_fill broadcast helper`);
  expectClose(adapter.Tensor.maskedFill(nested, nested.lt(3), adapter.tensor([10, 20], [1, 2])).data, [10, 20, 3, 4], `${label} Tensor.maskedFill static tensor value helper`);
  const maskedFillGradInput = adapter.tensor([1, 2, 3, 4], [2, 2]).requiresGrad_();
  const maskedFillValue = adapter.tensor([10, 20], [1, 2]).requiresGrad_();
  maskedFillGradInput.maskedFill(adapter.tensor([1, 0, 0, 1], [2, 2]), maskedFillValue).sum().backward();
  expectClose(maskedFillGradInput.grad, [0, 1, 1, 0], `${label} Tensor.maskedFill input autograd backward`);
  expectClose(maskedFillValue.grad, [1, 1], `${label} Tensor.maskedFill value autograd backward`);
  expectClose(nested.gt(2).any().data, [1], `${label} Tensor.any scalar`);
  expectClose(nested.gt(2).any(0).data, [1, 1], `${label} Tensor.any dim0`);
  expectClose(nested.gt(2).all(1).data, [0, 1], `${label} Tensor.all dim1`);
  expectClose(adapter.all(nested.gt(0)).data, [1], `${label} all helper`);
  const minReduction = adapter.nn.min(0).forward(adapter.tensor([1, 3, 2, 4, 0, -1], [2, 3]));
  if (!(minReduction instanceof adapter.Tensor)) throw new Error(`${label} expected nn.min to return a Tensor`);
  expectClose(minReduction.data, [1, 0, -1], `${label} nn.min dim0`);
  const prodReduction = adapter.nn.prod(1).forward(adapter.tensor([1, 3, 2, 4, 0, -1], [2, 3]));
  if (!(prodReduction instanceof adapter.Tensor)) throw new Error(`${label} expected nn.prod to return a Tensor`);
  expectClose(prodReduction.data, [6, 0], `${label} nn.prod dim1`);
  const argmaxReduction = adapter.nn.argmax(1).forward(adapter.tensor([1, 3, 2, 4, 0, -1], [2, 3]));
  if (!(argmaxReduction instanceof adapter.Tensor)) throw new Error(`${label} expected nn.argmax to return a Tensor`);
  expectClose(argmaxReduction.data, [1, 0], `${label} nn.argmax dim1`);
  expectClose(nested.argmax(0).data, [1, 1], `${label} Tensor.argmax dim0`);
  const argminReduction = adapter.nn.argmin(1).forward(adapter.tensor([1, 3, 2, 4, 0, -1], [2, 3]));
  if (!(argminReduction instanceof adapter.Tensor)) throw new Error(`${label} expected nn.argmin to return a Tensor`);
  expectClose(argminReduction.data, [0, 2], `${label} nn.argmin dim1`);
  expectClose(nested.argmin(0).data, [0, 0], `${label} Tensor.argmin dim0`);
  const batchNorm = new adapter.nn.BatchNorm1d(2, { eps: 0, momentum: 0.5, weight: [1, 1], bias: [0, 0] });
  expectClose(batchNorm.forward(adapter.tensor([1, 3, 3, 7], [2, 2])).data, [-1, -1, 1, 1], `${label} nn.BatchNorm1d train forward`);
  expectClose(batchNorm.runningMean.data, [1, 2.5], `${label} nn.BatchNorm1d running mean`);
  expectClose(batchNorm.runningVar.data, [1.5, 4.5], `${label} nn.BatchNorm1d running var`);
  const batchNormState = batchNorm.stateDict("bn");
  expectClose(batchNormState["bn.runningMean"].data, [1, 2.5], `${label} nn.BatchNorm1d stateDict running mean`);
  batchNorm.loadStateDict({
    weight: { shape: [2], data: [2, 3] },
    bias: { shape: [2], data: [0.5, -0.5] },
    runningMean: { shape: [2], data: [1, 1] },
    runningVar: { shape: [2], data: [4, 9] },
    numBatchesTracked: { shape: [1], data: [8] },
  });
  batchNorm.eval();
  expectClose(batchNorm.forward(adapter.tensor([3, 4], [1, 2])).data, [2.5, 2.5], `${label} nn.BatchNorm1d eval forward`);
  const batchNormTrace = adapter.nn.sequential(batchNorm).trace({ inputShape: [1, 2] });
  if (batchNormTrace.outputShape?.join("x") !== "1x2" || batchNormTrace.ops[0]?.op !== "batchNorm1d") {
    throw new Error(`${label} expected nn.BatchNorm1d trace output shape evidence, got ${batchNormTrace.outputShape}`);
  }
  const batchNormSupport = adapter.nn.sequential(batchNorm).compileSupport({ inputShape: [1, 2], backend: "cpu" });
  if (
    batchNormSupport.supported !== true ||
    batchNormSupport.ir?.ops[0]?.op !== "batchNorm1d" ||
    batchNormSupport.kernelPlan?.ops[0]?.op !== "batchNorm1d" ||
    batchNormSupport.kernelPlan?.ops[0]?.kernel !== "affine"
  ) {
    throw new Error(`${label} expected eval nn.BatchNorm1d compileSupport to lower as native affine`);
  }
  const batchNormProgram = adapter.nn.sequential(batchNorm).compile({ inputShape: [1, 2], backend: "cpu" });
  const batchNormSession = batchNormProgram.bindModule(adapter.nn.sequential(batchNorm));
  const batchNormCompiledOutput = batchNormSession.executeInto(new Float32Array(2), {
    input: adapter.tensor([3, 4], [1, 2]),
  });
  expectClose(batchNormCompiledOutput, [2.5, 2.5], `${label} nn.BatchNorm1d eval compiled output`);
  batchNormSession.dispose();
  batchNormProgram.dispose();
  batchNorm.train();
  const batchNormTrainSupport = adapter.nn.sequential(batchNorm).compileSupport({ inputShape: [1, 2], backend: "cpu" });
  if (
    batchNormTrainSupport.supported !== false ||
    batchNormTrainSupport.ir?.ops[0]?.op !== "batchNorm1d" ||
    batchNormTrainSupport.diagnostics?.[0]?.stage !== "kernelizer" ||
    batchNormTrainSupport.diagnostics?.[0]?.code !== "unsupported-op"
  ) {
    throw new Error(`${label} expected training nn.BatchNorm1d compileSupport to remain honest unsupported with partial IR`);
  }
  const conv2d = new adapter.nn.Conv2d(1, 1, 2, { weight: [1, 0, 0, 1], bias: false });
  expectClose(conv2d.forward(adapter.tensor([1, 2, 3, 4], [1, 2, 2])).data, [5], `${label} nn.Conv2d eager forward`);
  expectClose(conv2d.forward(adapter.tensor([1, 2, 3, 4], [1, 1, 2, 2])).data, [5], `${label} nn.Conv2d batched eager forward`);
  const conv2dSpatial = new adapter.nn.Conv2d(1, 1, 2, {
    weight: [1, 1, 1, 1],
    bias: false,
    stride: 2,
    padding: 1,
    dilation: 2,
  });
  expectClose(
    conv2dSpatial.forward(adapter.tensor([
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20,
      21, 22, 23, 24, 25,
    ], [1, 5, 5])).data,
    [7, 16, 9, 24, 52, 28, 17, 36, 19],
    `${label} nn.Conv2d stride padding dilation eager forward`,
  );
  const conv2dGradInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]).requiresGrad_();
  const conv2dGradOutput = conv2d.forward(conv2dGradInput);
  conv2dGradOutput.backward([1, 1, 1, 1]);
  expectClose(conv2dGradInput.grad, [1, 1, 0, 1, 2, 1, 0, 1, 1], `${label} nn.Conv2d input grad`);
  expectClose(conv2d.parameters()[0].grad, [12, 16, 24, 28], `${label} nn.Conv2d weight grad`);
  const conv2dBias = new adapter.nn.Conv2d(1, 1, 2, { weight: [1, 0, 0, 1], bias: [0.5] });
  const conv2dBiasOutput = conv2dBias.forward(adapter.tensor([1, 2, 3, 4], [1, 2, 2]));
  expectClose(conv2dBiasOutput.data, [5.5], `${label} nn.Conv2d bias eager forward`);
  conv2dBiasOutput.backward();
  expectClose(conv2dBias.parameters()[1].grad, [1], `${label} nn.Conv2d bias grad`);
  const conv2dBiasState = conv2dBias.stateDict("conv");
  expectClose(conv2dBiasState["conv.bias"].data, [0.5], `${label} nn.Conv2d stateDict bias`);
  const conv2dState = conv2d.stateDict("conv");
  expectClose(conv2dState["conv.weight"].data, [1, 0, 0, 1], `${label} nn.Conv2d stateDict weight`);
  const conv2dTrace = adapter.nn.sequential(conv2d).trace({ inputShape: [1, 3, 3] });
  if (conv2dTrace.outputShape?.join("x") !== "1x2x2") {
    throw new Error(`${label} expected nn.Conv2d trace output shape evidence, got ${conv2dTrace.outputShape}`);
  }
  const conv2dSequentialSupport = adapter.nn.sequential(conv2d).compileSupport({ inputShape: [1, 3, 3], backend: "cpu" });
  if (
    conv2dSequentialSupport.supported !== true ||
    conv2dSequentialSupport.kernelPlan?.ops[0]?.op !== "conv2d" ||
    conv2dSequentialSupport.kernelPlan?.ops[0]?.kernel !== "conv2d"
  ) {
    throw new Error(`${label} expected nn.Conv2d compileSupport to lower fixed valid rank-3 Conv2d`);
  }
  const batchedConv2dSupport = conv2d.compileSupport({ inputShape: [2, 1, 3, 3], backend: "cpu" });
  if (
    batchedConv2dSupport.supported !== true ||
    batchedConv2dSupport.kernelPlan?.inputShape?.join("x") !== "2x1x3x3" ||
    batchedConv2dSupport.kernelPlan?.outputShape?.join("x") !== "2x1x2x2" ||
    batchedConv2dSupport.kernelPlan?.ops[0]?.op !== "conv2d"
  ) {
    throw new Error(`${label} expected nn.Conv2d compileSupport to lower fixed valid rank-4 batched Conv2d`);
  }
  const directConv2dProgram = conv2d.compile({ inputShape: [1, 3, 3], backend: "cpu" });
  const directConv2dSession = directConv2dProgram.bindModule(conv2d);
  const directConv2dOutput = directConv2dSession.executeInto(new Float32Array(4), {
    input: adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]),
  });
  expectClose(directConv2dOutput, [6, 8, 12, 14], `${label} nn.Conv2d direct compiled output`);
  directConv2dSession.dispose();
  directConv2dProgram.dispose();
  const namespaceConv2dProgram = adapter.compile.compile(conv2d, { inputShape: [1, 3, 3], backend: "cpu" });
  const namespaceConv2dSession = namespaceConv2dProgram.bindModule(conv2d);
  const namespaceConv2dOutput = namespaceConv2dSession.executeInto(new Float32Array(4), {
    input: adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]),
  });
  expectClose(namespaceConv2dOutput, [6, 8, 12, 14], `${label} nn.Conv2d namespace compiled output`);
  namespaceConv2dSession.dispose();
  namespaceConv2dProgram.dispose();
  const batchedConv2dProgram = conv2d.compile({ inputShape: [2, 1, 3, 3], backend: "cpu" });
  const batchedConv2dSession = batchedConv2dProgram.bindModule(conv2d);
  const batchedConv2dOutput = batchedConv2dSession.executeInto(new Float32Array(8), {
    input: adapter.tensor([
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17, 18,
    ], [2, 1, 3, 3]),
  });
  expectClose(batchedConv2dOutput, [6, 8, 12, 14, 24, 26, 30, 32], `${label} nn.Conv2d batched compiled output`);
  batchedConv2dSession.dispose();
  batchedConv2dProgram.dispose();
  const unsupportedConv2dSupport = conv2dSpatial.compileSupport({ inputShape: [1, 5, 5], backend: "cpu" });
  if (
    unsupportedConv2dSupport.supported !== false ||
    unsupportedConv2dSupport.diagnostics?.[0]?.stage !== "kernelizer" ||
    unsupportedConv2dSupport.diagnostics?.[0]?.code !== "unsupported-op"
  ) {
    throw new Error(`${label} expected spatial nn.Conv2d compileSupport to stay honestly unsupported outside fixed native subset`);
  }
  const maxPool2d = new adapter.nn.MaxPool2d(2);
  expectClose(maxPool2d.forward(adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3])).data, [5], `${label} nn.MaxPool2d eager forward`);
  const directFixedMaxPool2d = new adapter.nn.MaxPool2d(2);
  const directMaxPool2dSupport = directFixedMaxPool2d.compileSupport({ inputShape: [1, 4, 4], backend: "cpu" });
  if (
    directMaxPool2dSupport.supported !== true ||
    directMaxPool2dSupport.kernelPlan?.ops[0]?.op !== "maxPool2d" ||
    directMaxPool2dSupport.kernelPlan?.ops[0]?.kernel !== "max-pool2d"
  ) {
    throw new Error(`${label} expected direct nn.MaxPool2d to compile fixed single-channel 2x2 stride2 shape`);
  }
  const directMaxPool2dProgram = directFixedMaxPool2d.compile({ inputShape: [1, 4, 4], backend: "cpu" });
  const directMaxPool2dSession = directMaxPool2dProgram.bind({});
  const directMaxPool2dInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
  expectClose(directMaxPool2dSession.stepTensor(directMaxPool2dInput).data, directFixedMaxPool2d.forward(directMaxPool2dInput).data, `${label} compiled direct nn.MaxPool2d fixed 2x2 stride2 output`);
  const namespaceMaxPool2dProgram = adapter.compile.compile(directFixedMaxPool2d, { inputShape: [1, 4, 4], backend: "cpu" });
  const namespaceMaxPool2dSession = namespaceMaxPool2dProgram.bind({});
  try {
    expectClose(namespaceMaxPool2dSession.stepTensor(directMaxPool2dInput).data, [6, 8, 14, 16], `${label} compile namespace nn.MaxPool2d fixed 2x2 stride2 output`);
  } finally {
    namespaceMaxPool2dSession.dispose();
    namespaceMaxPool2dProgram.dispose();
  }
  const fixedMaxPool2d = adapter.nn.sequential(adapter.nn.max_pool2d(2));
  const fixedMaxPool2dSupport = fixedMaxPool2d.compileSupport({ inputShape: [1, 4, 4], backend: "cpu" });
  if (
    fixedMaxPool2dSupport.supported !== true ||
    fixedMaxPool2dSupport.kernelPlan?.ops[0]?.op !== "maxPool2d" ||
    fixedMaxPool2dSupport.kernelPlan?.ops[0]?.kernel !== "max-pool2d" ||
    fixedMaxPool2dSupport.kernelPlan?.ops[0]?.nativeKernels?.[0] !== "max-pool2d"
  ) {
    throw new Error(`${label} expected fixed single-channel nn.MaxPool2d to compile to native Program`);
  }
  const fixedMaxPool2dProgram = fixedMaxPool2d.compile({ inputShape: [1, 4, 4], backend: "cpu" });
  const fixedMaxPool2dSession = fixedMaxPool2dProgram.bind({});
  const fixedMaxPool2dInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
  expectClose(fixedMaxPool2dSession.stepTensor(fixedMaxPool2dInput).data, fixedMaxPool2d.forward(fixedMaxPool2dInput).data, `${label} compiled nn.MaxPool2d fixed 2x2 stride2 output`);
  const batchedMaxPool2dInput = adapter.tensor(Array.from({ length: 64 }, (_, index) => index + 1), [2, 2, 4, 4]);
  const batchedMaxPool2dSupport = fixedMaxPool2d.compileSupport({ inputShape: [2, 2, 4, 4], backend: "cpu" });
  if (
    batchedMaxPool2dSupport.supported !== true ||
    batchedMaxPool2dSupport.kernelPlan?.inputShape?.join("x") !== "2x2x4x4" ||
    batchedMaxPool2dSupport.kernelPlan?.outputShape?.join("x") !== "2x2x2x2" ||
    batchedMaxPool2dSupport.kernelPlan?.ops[0]?.kernel !== "max-pool2d"
  ) {
    throw new Error(`${label} expected nn.MaxPool2d to compile fixed rank-4 batched channel shape`);
  }
  const batchedMaxPool2dProgram = fixedMaxPool2d.compile({ inputShape: [2, 2, 4, 4], backend: "cpu" });
  const batchedMaxPool2dSession = batchedMaxPool2dProgram.bind({});
  try {
    expectClose(batchedMaxPool2dSession.stepTensor(batchedMaxPool2dInput).data, fixedMaxPool2d.forward(batchedMaxPool2dInput).data, `${label} compiled nn.MaxPool2d batched channel output`);
  } finally {
    batchedMaxPool2dSession.dispose();
    batchedMaxPool2dProgram.dispose();
  }
  const maxPool2dFactory = adapter.nn.max_pool2d(2, { stride: 1 });
  expectClose(maxPool2dFactory.forward(adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3])).data, [5, 6, 8, 9], `${label} nn.max_pool2d eager forward`);
  const maxPool2dGradInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]).requiresGrad_();
  const maxPool2dGradOutput = maxPool2dFactory.forward(maxPool2dGradInput);
  maxPool2dGradOutput.backward([1, 1, 1, 1]);
  expectClose(maxPool2dGradInput.grad, [0, 0, 0, 0, 1, 1, 0, 1, 1], `${label} nn.MaxPool2d input grad`);
  const maxPool2dTrace = adapter.nn.sequential(maxPool2dFactory).trace({ inputShape: [1, 3, 3] });
  if (maxPool2dTrace.outputShape?.join("x") !== "1x2x2") {
    throw new Error(`${label} expected nn.MaxPool2d trace output shape evidence, got ${maxPool2dTrace.outputShape}`);
  }
  const maxPool2dSupport = adapter.nn.sequential(maxPool2dFactory).compileSupport({ inputShape: [1, 3, 3], backend: "cpu" });
  if (
    maxPool2dSupport.supported !== false ||
    maxPool2dSupport.ir?.ops[0]?.op !== "maxPool2d" ||
    maxPool2dSupport.diagnostics?.[0]?.stage !== "kernelizer" ||
    maxPool2dSupport.diagnostics?.[0]?.code !== "unsupported-op"
  ) {
    throw new Error(`${label} expected nn.MaxPool2d compileSupport to be honest unsupported with partial IR`);
  }
  if (maxPool2d.compileSupport({ inputShape: [1, 3, 3] }).supported !== false) {
    throw new Error(`${label} expected nn.MaxPool2d compileSupport to reject unsupported shape outside the native subset`);
  }
  const avgPool2d = new adapter.nn.AvgPool2d(2);
  expectClose(avgPool2d.forward(adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3])).data, [3], `${label} nn.AvgPool2d eager forward`);
  const directFixedAvgPool2d = new adapter.nn.AvgPool2d(2);
  const directAvgPool2dSupport = directFixedAvgPool2d.compileSupport({ inputShape: [1, 4, 4], backend: "cpu" });
  if (
    directAvgPool2dSupport.supported !== true ||
    directAvgPool2dSupport.kernelPlan?.ops[0]?.op !== "avgPool2d" ||
    directAvgPool2dSupport.kernelPlan?.ops[0]?.kernel !== "avg-pool2d"
  ) {
    throw new Error(`${label} expected direct nn.AvgPool2d to compile fixed single-channel 2x2 stride2 shape`);
  }
  const directAvgPool2dProgram = directFixedAvgPool2d.compile({ inputShape: [1, 4, 4], backend: "cpu" });
  const directAvgPool2dSession = directAvgPool2dProgram.bind({});
  const directAvgPool2dInput = adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
  expectClose(directAvgPool2dSession.stepTensor(directAvgPool2dInput).data, directFixedAvgPool2d.forward(directAvgPool2dInput).data, `${label} compiled direct nn.AvgPool2d fixed 2x2 stride2 output`);
  const namespaceAvgPool2dProgram = adapter.compile.compile(directFixedAvgPool2d, { inputShape: [1, 4, 4], backend: "cpu" });
  const namespaceAvgPool2dSession = namespaceAvgPool2dProgram.bind({});
  try {
    expectClose(namespaceAvgPool2dSession.stepTensor(directAvgPool2dInput).data, [3.5, 5.5, 11.5, 13.5], `${label} compile namespace nn.AvgPool2d fixed 2x2 stride2 output`);
  } finally {
    namespaceAvgPool2dSession.dispose();
    namespaceAvgPool2dProgram.dispose();
    directAvgPool2dSession.dispose();
    directAvgPool2dProgram.dispose();
  }
  const fixedAvgPool2d = adapter.nn.sequential(adapter.nn.avg_pool2d(2));
  const fixedAvgPool2dSupport = fixedAvgPool2d.compileSupport({ inputShape: [1, 4, 4], backend: "cpu" });
  if (
    fixedAvgPool2dSupport.supported !== true ||
    fixedAvgPool2dSupport.kernelPlan?.ops[0]?.op !== "avgPool2d" ||
    fixedAvgPool2dSupport.kernelPlan?.ops[0]?.kernel !== "avg-pool2d" ||
    fixedAvgPool2dSupport.kernelPlan?.ops[0]?.nativeKernels?.[0] !== "avg-pool2d"
  ) {
    throw new Error(`${label} expected fixed single-channel nn.AvgPool2d to compile to native Program`);
  }
  const fixedAvgPool2dProgram = fixedAvgPool2d.compile({ inputShape: [1, 4, 4], backend: "cpu" });
  const fixedAvgPool2dSession = fixedAvgPool2dProgram.bind({});
  try {
    expectClose(fixedAvgPool2dSession.stepTensor(directAvgPool2dInput).data, fixedAvgPool2d.forward(directAvgPool2dInput).data, `${label} compiled nn.AvgPool2d fixed 2x2 stride2 output`);
  } finally {
    fixedAvgPool2dSession.dispose();
    fixedAvgPool2dProgram.dispose();
  }
  const batchedAvgPool2dInput = adapter.tensor(Array.from({ length: 64 }, (_, index) => index + 1), [2, 2, 4, 4]);
  const batchedAvgPool2dSupport = fixedAvgPool2d.compileSupport({ inputShape: [2, 2, 4, 4], backend: "cpu" });
  if (
    batchedAvgPool2dSupport.supported !== true ||
    batchedAvgPool2dSupport.kernelPlan?.inputShape?.join("x") !== "2x2x4x4" ||
    batchedAvgPool2dSupport.kernelPlan?.outputShape?.join("x") !== "2x2x2x2" ||
    batchedAvgPool2dSupport.kernelPlan?.ops[0]?.kernel !== "avg-pool2d"
  ) {
    throw new Error(`${label} expected nn.AvgPool2d to compile fixed rank-4 batched channel shape`);
  }
  const batchedAvgPool2dProgram = fixedAvgPool2d.compile({ inputShape: [2, 2, 4, 4], backend: "cpu" });
  const batchedAvgPool2dSession = batchedAvgPool2dProgram.bind({});
  try {
    expectClose(batchedAvgPool2dSession.stepTensor(batchedAvgPool2dInput).data, fixedAvgPool2d.forward(batchedAvgPool2dInput).data, `${label} compiled nn.AvgPool2d batched channel output`);
  } finally {
    batchedAvgPool2dSession.dispose();
    batchedAvgPool2dProgram.dispose();
  }
  const avgPool2dFactory = adapter.nn.avg_pool2d(2, { stride: 1 });
  expectClose(avgPool2dFactory.forward(adapter.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3])).data, [3, 4, 6, 7], `${label} nn.avg_pool2d eager forward`);
  const avgPool2dTrace = adapter.nn.sequential(avgPool2dFactory).trace({ inputShape: [1, 3, 3] });
  if (avgPool2dTrace.outputShape?.join("x") !== "1x2x2" || avgPool2dTrace.ops[0]?.op !== "avgPool2d") {
    throw new Error(`${label} expected nn.AvgPool2d trace output shape evidence, got ${avgPool2dTrace.outputShape}`);
  }
  const avgPool2dSupport = adapter.nn.sequential(avgPool2dFactory).compileSupport({ inputShape: [1, 3, 3], backend: "cpu" });
  if (
    avgPool2dSupport.supported !== false ||
    avgPool2dSupport.diagnostics?.[0]?.op !== "avgPool2d" ||
    avgPool2dSupport.diagnostics?.[0]?.stage !== "kernelizer" ||
    avgPool2dSupport.diagnostics?.[0]?.code !== "unsupported-op" ||
    avgPool2dSupport.ir?.ops[0]?.op !== "avgPool2d"
  ) {
    throw new Error(`${label} expected nn.AvgPool2d compileSupport to be honest unsupported with partial IR`);
  }
  const tanhModule = adapter.nn.tanh();
  const tanhEager = tanhModule.forward(adapter.tensor([0, 1], [2]));
  expectClose(tanhEager.data, [0, Math.tanh(1)], `${label} nn.tanh eager forward`);
  if (new adapter.nn.Tanh().kind !== "tanh") {
    throw new Error(`${label} expected nn.Tanh class alias`);
  }
  const tanhSupport = tanhModule.compileSupport({ inputShape: [2], backend: "cpu" });
  if (
    !Object.isFrozen(tanhSupport) ||
    tanhSupport.supported !== true ||
    tanhSupport.outputShape.join("x") !== "2" ||
    tanhSupport.kernelPlan?.ops[0]?.nativeKernels?.join("|") !== "tanh"
  ) {
    throw new Error(`${label} expected nn.tanh compiled activation evidence`);
  }
  const tanhProgram = tanhModule.compile({ inputShape: [2], backend: "cpu" });
  const tanhSession = tanhProgram.bindModule(tanhModule);
  try {
    const tanhCompiled = tanhSession.stepTensor(adapter.tensor([0, 1], [2]));
    expectClose(tanhCompiled.data, tanhEager.data, `${label} nn.tanh eager/compiled parity`);
  } finally {
    tanhSession.dispose();
    tanhProgram.dispose();
  }
  const activationChain = adapter.nn.sequential([adapter.nn.relu(), adapter.nn.square(), adapter.nn.sqrt()]);
  const activationChainInput = adapter.tensor([-4, 9], [2]);
  const activationChainEager = activationChain.forward(activationChainInput);
  const activationChainSupport = activationChain.compileSupport({ inputShape: [2], backend: "cpu" });
  const activationChainPlan = adapter.nn.kernelPlan(activationChain, { inputShape: [2], backend: "cpu" });
  if (
    !Object.isFrozen(activationChainSupport) ||
    !Object.isFrozen(activationChainPlan) ||
    activationChainSupport.supported !== true ||
    activationChainSupport.trace.opCount !== 3 ||
    activationChainPlan.opCount !== 3 ||
    activationChainPlan.ops.length !== 1 ||
    activationChainPlan.dispatchCount !== 1 ||
    activationChainPlan.ops[0]?.op !== "activation-chain" ||
    activationChainPlan.ops[0]?.fusedOpCount !== 3 ||
    activationChainPlan.ops[0]?.nativeKernels?.join("|") !== "relu|square|sqrt"
  ) {
    throw new Error(`${label} expected activation-chain fusion compile evidence`);
  }
  const activationChainProgram = activationChain.compile({ inputShape: [2], backend: "cpu" });
  const activationChainSession = activationChainProgram.bindModule(activationChain);
  try {
    const activationChainCompiled = activationChainSession.stepTensor(activationChainInput);
    expectClose(activationChainCompiled.data, activationChainEager.data, `${label} activation-chain eager/compiled parity`);
  } finally {
    activationChainSession.dispose();
    activationChainProgram.dispose();
  }
  const shapeChain = adapter.nn.sequential([adapter.nn.flatten(), adapter.nn.reshape([2, 3])]);
  const shapeChainInput = adapter.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const shapeChainSupport = shapeChain.compileSupport({ inputShape: [2, 3], backend: "cpu" });
  const shapeChainPlan = adapter.nn.kernelPlan(shapeChain, { inputShape: [2, 3], backend: "cpu" });
  if (
    !Object.isFrozen(shapeChainSupport) ||
    !Object.isFrozen(shapeChainPlan) ||
    shapeChainSupport.supported !== true ||
    shapeChainSupport.trace.opCount !== 2 ||
    shapeChainPlan.opCount !== 2 ||
    shapeChainPlan.ops.length !== 0 ||
    shapeChainPlan.dispatchCount !== 0 ||
    shapeChainPlan.elidedOpCount !== 1 ||
    shapeChainPlan.elidedOps[0]?.op !== "shape-chain" ||
    shapeChainPlan.elidedOps[0]?.fusedOpCount !== 2
  ) {
    throw new Error(`${label} expected shape-chain elision compile evidence`);
  }
  const shapeChainProgram = shapeChain.compile({ inputShape: [2, 3], backend: "cpu" });
  const shapeChainSession = shapeChainProgram.bindModule(shapeChain);
  try {
    const shapeChainCompiled = shapeChainSession.stepTensor(shapeChainInput);
    if (shapeChainCompiled.shape.join("x") !== "2x3") throw new Error(`${label} expected shape-chain compiled output shape`);
    expectClose(shapeChainCompiled.data, shapeChainInput.data, `${label} shape-chain eager/compiled parity`);
  } finally {
    shapeChainSession.dispose();
    shapeChainProgram.dispose();
  }
  const gradAliasTensor = adapter.tensor([1, 2], [2]);
  if (gradAliasTensor.requires_grad !== false || gradAliasTensor.requiresGrad !== false) {
    throw new Error(`${label} expected fresh tensor grad aliases to start disabled`);
  }
  if (gradAliasTensor.requires_grad_() !== gradAliasTensor) {
    throw new Error(`${label} expected Tensor.requires_grad_ to return this`);
  }
  if (gradAliasTensor.requires_grad !== true || gradAliasTensor.requiresGrad !== true || !(gradAliasTensor.grad instanceof Float32Array)) {
    throw new Error(`${label} expected Tensor.requires_grad_ to enable both grad aliases and allocate grad`);
  }
  gradAliasTensor.zero_grad({ set_to_none: true });
  if (gradAliasTensor.grad !== null) {
    throw new Error(`${label} expected Tensor.zero_grad set_to_none to clear grad buffer`);
  }
  gradAliasTensor.zeroGrad();
  if (gradAliasTensor.grad !== null) {
    throw new Error(`${label} expected Tensor.zeroGrad without grad buffer to stay allocation-free`);
  }
  gradAliasTensor.requires_grad = false;
  if (gradAliasTensor.requires_grad !== false || gradAliasTensor.requiresGrad !== false) {
    throw new Error(`${label} expected Tensor.requires_grad setter to disable both grad aliases`);
  }
  gradAliasTensor.requiresGrad_(true);
  gradAliasTensor.detach_();
  if (gradAliasTensor.requires_grad !== false || gradAliasTensor.requiresGrad !== false || gradAliasTensor.grad !== null) {
    throw new Error(`${label} expected Tensor.detach_ to clear grad aliases and grad buffer`);
  }

  const zeroGradAliasModel = adapter.nn.linear(2, 1, { weights: [1, -1], bias: [0] });
  adapter.loss.mse(zeroGradAliasModel.forward(adapter.tensor([2, 1], [2])), adapter.tensor([0], [1])).backward();
  if (gradNorm(zeroGradAliasModel.parameters()) <= 0) {
    throw new Error(`${label} expected module gradients before zero_grad alias`);
  }
  zeroGradAliasModel.zero_grad();
  expectClose(zeroGradAliasModel.parameters()[0].grad, [0, 0], `${label} module zero_grad alias clears weight grad`);
  expectClose(zeroGradAliasModel.parameters()[1].grad, [0], `${label} module zero_grad alias clears bias grad`);
  adapter.loss.mse(zeroGradAliasModel.forward(adapter.tensor([2, 1], [2])), adapter.tensor([0], [1])).backward();
  adapter.nn.zero_grad(zeroGradAliasModel);
  expectClose(zeroGradAliasModel.parameters()[0].grad, [0, 0], `${label} nn.zero_grad alias clears weight grad`);
  expectClose(zeroGradAliasModel.parameters()[1].grad, [0], `${label} nn.zero_grad alias clears bias grad`);
  if (adapter.nn.eval(zeroGradAliasModel) !== zeroGradAliasModel || zeroGradAliasModel.training !== false) {
    throw new Error(`${label} expected nn.eval to return target and switch module to eval mode`);
  }
  if (adapter.nn.train(zeroGradAliasModel) !== zeroGradAliasModel || zeroGradAliasModel.training !== true) {
    throw new Error(`${label} expected nn.train to return target and switch module to training mode`);
  }
  if (adapter.nn.train(zeroGradAliasModel, false) !== zeroGradAliasModel || zeroGradAliasModel.training !== false) {
    throw new Error(`${label} expected nn.train(false) to return target and switch module to eval mode`);
  }
  if (
    zeroGradAliasModel.cpu() !== zeroGradAliasModel ||
    zeroGradAliasModel.to("cpu") !== zeroGradAliasModel ||
    zeroGradAliasModel.float() !== zeroGradAliasModel ||
    zeroGradAliasModel.float32() !== zeroGradAliasModel ||
    adapter.nn.cpu(zeroGradAliasModel) !== zeroGradAliasModel ||
    adapter.nn.to(zeroGradAliasModel, { device: "cpu", copy: false }) !== zeroGradAliasModel ||
    adapter.nn.float(zeroGradAliasModel) !== zeroGradAliasModel ||
    adapter.nn.float32(zeroGradAliasModel, { dtype: "float32", copy: false }) !== zeroGradAliasModel
  ) {
    throw new Error(`${label} expected module cpu/to/float helpers to return target`);
  }
  expectThrowIncludes(
    () => zeroGradAliasModel.to("metal"),
    "nn.Module.to currently supports only cpu/f32 placement",
    `${label} module to rejects unsupported placement`,
  );
  adapter.loss.mse(zeroGradAliasModel.forward(adapter.tensor([2, 1], [2])), adapter.tensor([0], [1])).backward();
  zeroGradAliasModel.zeroGrad({ setToNone: true });
  if (zeroGradAliasModel.parameters()[0].grad !== null || zeroGradAliasModel.parameters()[0].tensor.grad !== null) {
    throw new Error(`${label} expected module zeroGrad setToNone to clear weight grad references`);
  }
  adapter.loss.mse(zeroGradAliasModel.forward(adapter.tensor([2, 1], [2])), adapter.tensor([0], [1])).backward();
  adapter.nn.zero_grad(zeroGradAliasModel, { set_to_none: true });
  if (zeroGradAliasModel.parameters()[1].grad !== null || zeroGradAliasModel.parameters()[1].tensor.grad !== null) {
    throw new Error(`${label} expected nn.zero_grad set_to_none to clear bias grad references`);
  }
  const initParam = adapter.nn.Parameter("initWeight", adapter.tensor([0, 0, 0, 0], [2, 2]));
  const initTensor = adapter.tensor([0, 0, 0, 0], [2, 2]);
  if (adapter.nn.init.constant_(initParam, 0.25) !== initParam) {
    throw new Error(`${label} expected nn.init.constant_ to return the initialized parameter`);
  }
  expectClose(initParam.data, [0.25, 0.25, 0.25, 0.25], `${label} nn.init.constant_ parameter`);
  if (adapter.nn.init.zeros_(initParam) !== initParam) {
    throw new Error(`${label} expected nn.init.zeros_ to return the initialized parameter`);
  }
  expectClose(initParam.data, [0, 0, 0, 0], `${label} nn.init.zeros_ parameter`);
  if (adapter.nn.init.ones_(initTensor) !== initTensor) {
    throw new Error(`${label} expected nn.init.ones_ to return the initialized tensor`);
  }
  expectClose(initTensor.data, [1, 1, 1, 1], `${label} nn.init.ones_ tensor`);
  adapter.nn.init.uniform_(initParam, -0.1, 0.1);
  if (Array.from(initParam.data, Number).some((value) => value < -0.1 || value > 0.1)) {
    throw new Error(`${label} expected nn.init.uniform_ to keep values inside bounds`);
  }
  adapter.nn.init.normal_(initParam, 0, 0);
  expectClose(initParam.data, [0, 0, 0, 0], `${label} nn.init.normal_ zero std`);
  adapter.nn.init.xavier_uniform_(initParam);
  if (Array.from(initParam.data, Number).some((value) => value < -Math.sqrt(6 / 4) || value > Math.sqrt(6 / 4))) {
    throw new Error(`${label} expected nn.init.xavier_uniform_ to keep values inside Xavier bounds`);
  }
  adapter.nn.init.xavierUniform_(initParam, 0.5);
  adapter.nn.init.xavier_normal_(initParam, 0);
  expectClose(initParam.data, [0, 0, 0, 0], `${label} nn.init.xavier_normal_ zero gain`);
  adapter.nn.init.xavierNormal_(initParam, 0);
  expectClose(initParam.data, [0, 0, 0, 0], `${label} nn.init.xavierNormal_ zero gain`);
  adapter.nn.init.kaiming_uniform_(initParam, { mode: "fan_in", nonlinearity: "relu" });
  if (Array.from(initParam.data, Number).some((value) => value < -Math.sqrt(6 / 2) || value > Math.sqrt(6 / 2))) {
    throw new Error(`${label} expected nn.init.kaiming_uniform_ to keep values inside Kaiming fan_in ReLU bounds`);
  }
  adapter.nn.init.kaimingUniform_(initParam, { mode: "fanOut", nonlinearity: "leaky_relu", negativeSlope: 0.2 });
  adapter.nn.init.kaiming_normal_(initParam, { mode: "fan_in", nonlinearity: "relu" });
  adapter.nn.init.kaimingNormal_(initParam, { mode: "fanOut", nonlinearity: "leaky_relu", negativeSlope: 0.2 });
  expectThrowIncludes(() => adapter.nn.init.xavier_uniform_(adapter.nn.Parameter("initBias", [0], [1])), "at least 2 dimensions", `${label} nn.init.xavier_uniform_ rank check`);
  expectThrowIncludes(() => adapter.nn.init.xavier_normal_(adapter.nn.Parameter("initBias", [0], [1])), "at least 2 dimensions", `${label} nn.init.xavier_normal_ rank check`);
  expectThrowIncludes(() => adapter.nn.init.kaiming_uniform_(adapter.nn.Parameter("initBias", [0], [1])), "at least 2 dimensions", `${label} nn.init.kaiming_uniform_ rank check`);
  expectThrowIncludes(() => adapter.nn.init.kaiming_normal_(adapter.nn.Parameter("initBias", [0], [1])), "at least 2 dimensions", `${label} nn.init.kaiming_normal_ rank check`);

  const frozenModel = adapter.nn.linear(2, 1, { weights: [1, 1], bias: false });
  const frozenParam = frozenModel.parameters()[0];
  if (frozenParam.requiresGrad !== true || frozenParam.requires_grad !== true) {
    throw new Error(`${label} expected nn parameters to expose requiresGrad aliases`);
  }
  adapter.loss.mse(frozenModel.forward(adapter.tensor([1, 1], [2])), adapter.tensor([0], [1])).backward();
  if (gradNorm(frozenModel.parameters()) <= 0) {
    throw new Error(`${label} expected freeze smoke to start with stale gradients`);
  }
  frozenModel.requiresGrad_(false);
  if (
    frozenParam.requiresGrad !== false ||
    frozenParam.requires_grad !== false ||
    frozenModel.parameterInfos()[0].requiresGrad !== false ||
    frozenModel.parameterInfos()[0].requires_grad !== false
  ) {
    throw new Error(`${label} expected requiresGrad_ to freeze parameter aliases and info evidence`);
  }
  if (frozenModel.requires_grad_(true) !== frozenModel || frozenParam.requiresGrad !== true || frozenParam.requires_grad !== true) {
    throw new Error(`${label} expected module requires_grad_ alias to unfreeze parameters`);
  }
  if (adapter.nn.requires_grad_(frozenModel, false) !== frozenModel || frozenParam.requiresGrad !== false || frozenParam.requires_grad !== false) {
    throw new Error(`${label} expected nn.requires_grad_ alias to freeze parameters`);
  }
  const frozenOptimizer = adapter.optim.sgd(frozenModel, { lr: 1 });
  frozenOptimizer.step();
  expectClose(frozenModel.weight, [1, 1], `${label} optimizer must skip frozen parameters with stale gradients`);
  const frozenNorm = adapter.train.clipGradNorm(frozenModel, 0.1);
  if (frozenNorm !== 0) {
    throw new Error(`${label} expected clipGradNorm to ignore frozen parameter gradients`);
  }
  adapter.train.clipGradValue(frozenModel, 0.1);
  if (gradNorm(frozenModel.parameters()) <= 0.1) {
    throw new Error(`${label} expected clipGradValue to leave frozen stale gradients untouched`);
  }
  frozenParam.requires_grad = true;
  if (frozenModel.parameterInfos()[0].requiresGrad !== true) {
    throw new Error(`${label} expected parameter requires_grad setter to unfreeze backing tensor`);
  }

  const groupedModel = adapter.nn.linear(1, 1, { weights: [1], bias: [1] });
  const groupedParams = groupedModel.parameters();
  groupedParams[0].grad[0] = 1;
  groupedParams[1].grad[0] = 1;
  const groupedOptimizer = adapter.optim.sgd([
    { params: [groupedParams[0]], lr: 0.1 },
    { params: [groupedParams[1]], lr: 0.01, weight_decay: 0 },
  ], { lr: 1, momentum: 0 });
  expectThrowIncludes(
    () => adapter.optim.sgd([
      { params: [groupedParams[0]], lr: 0.1 },
      { params: [groupedParams[0]], lr: 0.01 },
    ]),
    "optimizer parameter groups must not contain duplicate parameters",
    `${label} rejects duplicate construction-time optimizer parameters`,
  );
  const groupedConfig = groupedOptimizer.config();
  const groupedDefaults = groupedOptimizer.defaults;
  if (
    !Object.isFrozen(groupedConfig.paramGroups) ||
    !adapter.optim.isOptimizerConfigSnapshot(groupedDefaults) ||
    groupedDefaults.kind !== "sgd" ||
    groupedDefaults.signature !== groupedConfig.signature ||
    groupedOptimizer.paramGroups !== groupedConfig.paramGroups ||
    groupedOptimizer.param_groups !== groupedConfig.paramGroups ||
    groupedConfig.param_groups !== groupedConfig.paramGroups ||
    groupedConfig.paramGroups.length !== 2 ||
    groupedConfig.paramGroups[0].lr !== 0.1 ||
    groupedConfig.paramGroups[1].lr !== 0.01 ||
    !groupedConfig.signature.includes("groups=0:0-1:0.1:0,1:1-2:0.01:0")
  ) {
    throw new Error(`${label} expected optimizer param-group config evidence`);
  }
  groupedOptimizer.step();
  expectClose(groupedModel.weight, [0.8999999761581421], `${label} grouped optimizer weight lr`);
  expectClose(groupedModel.bias, [0.9900000095367432], `${label} grouped optimizer bias lr`);
  adapter.optim.setLearningRate(groupedOptimizer, 0.02);
  if (
    groupedOptimizer.config().paramGroups.some((group: Readonly<{ lr: number }>) => group.lr !== 0.02) ||
    groupedOptimizer.getLearningRate() !== 0.02 ||
    groupedOptimizer.get_lr() !== 0.02 ||
    adapter.optim.getLearningRate(groupedOptimizer) !== 0.02 ||
    adapter.optim.get_lr(groupedOptimizer) !== 0.02
  ) {
    throw new Error(`${label} expected setLearningRate to update all optimizer parameter groups`);
  }
  const appendedGroupModel = adapter.nn.linear(1, 1, { weights: [1], bias: [1] });
  const appendedGroupParams = appendedGroupModel.parameters();
  const appendedGroupOptimizer = adapter.optim.sgd([{ params: [appendedGroupParams[0]], lr: 0.1 }], { lr: 1, momentum: 0.5 });
  if (appendedGroupOptimizer.addParamGroup({ params: [appendedGroupParams[1]], lr: 0.01 }) !== appendedGroupOptimizer) {
    throw new Error(`${label} expected optimizer.addParamGroup to return this`);
  }
  if (
    appendedGroupOptimizer.config().paramGroups.length !== 2 ||
    appendedGroupOptimizer.config().paramCount !== 2 ||
    appendedGroupOptimizer.stateDict().paramCount !== 2 ||
    appendedGroupOptimizer.stateDict().entries.length !== 2
  ) {
    throw new Error(`${label} expected appended optimizer parameter group to extend config and state`);
  }
  appendedGroupParams[0].grad[0] = 1;
  appendedGroupParams[1].grad[0] = 1;
  appendedGroupOptimizer.step();
  expectClose(appendedGroupModel.weight, [0.8999999761581421], `${label} appended group optimizer weight lr`);
  expectClose(appendedGroupModel.bias, [0.9900000095367432], `${label} appended group optimizer bias lr`);
  const namespaceGroupModel = adapter.nn.linear(1, 1, { weights: [1], bias: [1] });
  const namespaceGroupParams = namespaceGroupModel.parameters();
  const namespaceGroupOptimizer = adapter.optim.sgd([{ params: [namespaceGroupParams[0]], lr: 0.1 }], { lr: 1, momentum: 0 });
  if (adapter.optim.add_param_group(namespaceGroupOptimizer, { params: [namespaceGroupParams[1]], lr: 0.001 }) !== namespaceGroupOptimizer) {
    throw new Error(`${label} expected optimizer add_param_group namespace alias`);
  }
  if (namespaceGroupOptimizer.config().paramGroups.length !== 2) {
    throw new Error(`${label} expected optimizer add_param_group namespace alias to append a unique group`);
  }
  expectThrowIncludes(
    () => adapter.optim.add_param_group(appendedGroupOptimizer, { params: [appendedGroupParams[1]], lr: 0.001 }),
    "optimizer parameter groups must not contain duplicate parameters",
    `${label} rejects duplicate appended optimizer parameters`,
  );

  const model = adapter.nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
  if (adapter.nn.children(model).length !== 0) throw new Error(`${label} expected linear namespace children traversal to be empty`);
  if (adapter.nn.modules(model)[0] !== model) throw new Error(`${label} expected linear namespace modules traversal to include self`);
  if (adapter.nn.namedModules(model, "model")[0].name !== "model") throw new Error(`${label} expected namespace namedModules prefix`);
  expectLossAndAdamWEvidence(adapter, label);
  const input = adapter.tensor([2, -1], [2]);
  const target = adapter.tensor([1], [1]);
  const before = adapter.loss.mse(model.forward(input), target);
  if (!(before instanceof adapter.Tensor)) throw new Error(`${label} expected tensor MSE loss before training`);
  expectClose(before.data, [0.25], `${label} training loss before`);
  adapter.train.step(adapter.optim.sgd(model, { lr: 0.1 }), { loss: before });
  const inspectedStep = adapter.train.inspectStep();
  if (
    !Object.isFrozen(inspectedStep) ||
    inspectedStep.kind !== "zgml.train.step" ||
    typeof inspectedStep.signature !== "string" ||
    !inspectedStep.signature.startsWith("train-step|optimizer=sgd|parameters=2|before=0|after=1|advanced=1|loss=1|lossScalar=0.25|") ||
    inspectedStep.optimizerKind !== "sgd" ||
    inspectedStep.parameterCount !== 2 ||
    inspectedStep.beforeStep !== 0 ||
    inspectedStep.afterStep !== 1 ||
    inspectedStep.stepAdvanced !== 1 ||
    inspectedStep.lossScalar !== 0.25 ||
    inspectedStep.hadLoss !== true ||
    inspectedStep.zeroGradApplied !== true ||
    inspectedStep.gradientsCleared !== true
  ) {
    throw new Error(`${label} expected frozen SGD TrainStepEvidence from train.inspectStep after train.step`);
  }
  if (
    !adapter.train.isTrainStepEvidence(inspectedStep) ||
    adapter.train.requireTrainStepEvidence(inspectedStep).afterStep !== 1 ||
    adapter.train.assertTrainStepEvidence(inspectedStep).optimizerKind !== "sgd" ||
    adapter.train.assert_train_step_evidence(inspectedStep).signature !== inspectedStep.signature ||
    !adapter.train.matchesTrainStepEvidenceSignature(inspectedStep, inspectedStep.signature) ||
    adapter.train.matches_train_step_evidence_signature(inspectedStep, "wrong") ||
    adapter.train.isTrainStepEvidence({ ...inspectedStep })
  ) {
    throw new Error(`${label} expected train namespace TrainStepEvidence validators to accept only frozen signed evidence`);
  }
  expectClose(model.weight, [0.3, -0.4], `${label} trained weight`);
  expectClose(model.bias, [-0.1], `${label} trained bias`);
  expectClose(model.parameters()[0].grad, [0, 0], `${label} cleared weight grad`);
  expectClose(model.parameters()[1].grad, [0], `${label} cleared bias grad`);
  const noneGradModel = adapter.nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
  const noneGradOptimizer = adapter.optim.sgd(noneGradModel, { lr: 0.1 });
  const noneGradLoss = adapter.loss.mse(noneGradModel.forward(input), target);
  const noneGradEvidence = adapter.train.step(noneGradOptimizer, {
    loss: noneGradLoss,
    zeroGradOptions: { setToNone: true },
    inspect: true,
  });
  if (
    noneGradEvidence.zeroGradApplied !== true ||
    noneGradEvidence.gradientsCleared !== true ||
    noneGradModel.parameters()[0].grad !== null ||
    noneGradModel.parameters()[1].grad !== null
  ) {
    throw new Error(`${label} expected train.step zeroGradOptions setToNone to clear gradients`);
  }
  const inlineStepEvidence = adapter.train.step(adapter.optim.sgd(model, { lr: 0.01 }), {
    loss: adapter.loss.mse(model.forward(input), target),
    inspect: true,
  });
  if (
    !Object.isFrozen(inlineStepEvidence) ||
    inlineStepEvidence.kind !== "zgml.train.step" ||
    typeof inlineStepEvidence.signature !== "string" ||
    !inlineStepEvidence.signature.startsWith("train-step|optimizer=sgd|parameters=2|") ||
    inlineStepEvidence.hadLoss !== true
  ) {
    throw new Error(`${label} expected train.step inspect:true to return frozen step evidence`);
  }
  const closureLoss = adapter.train.lossStep(adapter.optim.sgd(model, { lr: 0.05 }), () => adapter.loss.mse(model.forward(input), target));
  if (!(closureLoss instanceof adapter.Tensor)) throw new Error(`${label} expected train.lossStep to return the Tensor loss`);
  expectClose(model.parameters()[0].grad, [0, 0], `${label} lossStep cleared weight grad`);
  expectClose(model.parameters()[1].grad, [0], `${label} lossStep cleared bias grad`);
  const dataset = adapter.data.tensorDataset(
    adapter.tensor([1, 0, 0, 1], [2, 2]),
    adapter.tensor([1, 1], [2, 1]),
  );
  class CustomDataset extends adapter.data.Dataset {
    get length() {
      return 2;
    }
    sample(index: number) {
      return Object.freeze({
        kind: "zgml.data.sample",
        index,
        input: adapter.tensor(index === 0 ? [1, 0] : [0, 1], [2]),
        target: adapter.tensor([index], [1]),
      });
    }
  }
  const customDataset = new CustomDataset();
  const customDatasetLoader = new adapter.data.DataLoader(customDataset, { batch_size: 2, shuffle: false });
  const torchCustomDataset = new adapter.torch.utils.data.Dataset();
  const sequentialSampler = new adapter.data.SequentialSampler(customDataset);
  const randomSampler = new adapter.data.RandomSampler(customDataset, { seed: 7 });
  const replacementSampler = new adapter.torch.utils.data.RandomSampler(customDataset, { replacement: true, num_samples: 3, seed: 7 });
  const batchSampler = new adapter.data.BatchSampler(sequentialSampler, 1, false);
  const torchBatchSampler = new adapter.torch.utils.data.BatchSampler(sequentialSampler, 2, false);
  const customCollateFn = (
    samples: readonly PackageSmokeDatasetSample[],
    context: PackageSmokeCollateContext,
  ): PackageSmokeCustomCollateBatch => Object.freeze({
    kind: "custom-collate",
    first: samples[0].target.data[0],
    sampleIndices: context.sampleIndices,
    sample_indices: context.sample_indices,
    batchIndex: context.batchIndex,
  });
  const collatedLoaderBatch = Array.from(new adapter.data.DataLoader(customDataset, {
    sampler: [1, 0],
    batch_size: 2,
    collate_fn: customCollateFn,
  }))[0] as PackageSmokeCustomCollateBatch;
  const defaultCollatedBatch = adapter.torch.utils.data.default_collate(
    [customDataset.sample(1), customDataset.sample(0)],
    { batch_index: 4 },
  );
  const defaultCollateLoaderBatch = Array.from(new adapter.data.DataLoader(customDataset, {
    sampler: [1, 0],
    batch_size: 2,
    collate_fn: adapter.data.defaultCollate,
  }))[0] as PackageSmokeDatasetBatch;
  const samplerLoaderIndices = Array.from(
    new adapter.data.DataLoader(customDataset, { sampler: [1, 0], batch_size: 1 }),
    (batch) => (batch as PackageSmokeDatasetBatch).indices,
  );
  const batchSamplerLoaderIndices = Array.from(
    new adapter.torch.utils.data.DataLoader(customDataset, { batch_sampler: batchSampler }),
    (batch) => (batch as PackageSmokeDatasetBatch).indices,
  );
  if (
    customDataset.kind !== "zgml.data.dataset" ||
    customDataset.__len__() !== 2 ||
    customDataset.__getitem__(1).target.data.join(",") !== "1" ||
    customDataset.batch([0, 1]).input.shape.join("x") !== "2x2" ||
    customDatasetLoader.__getitem__(0).target.shape.join("x") !== "2x1" ||
    typeof torchCustomDataset.__len__ !== "function" ||
    sequentialSampler.kind !== "zgml.data.sequential-sampler" ||
    sequentialSampler.__len__() !== 2 ||
    Array.from(sequentialSampler).join(",") !== "0,1" ||
    randomSampler.kind !== "zgml.data.random-sampler" ||
    Array.from(randomSampler).join(",") !== "1,0" ||
    replacementSampler.size() !== 3 ||
    Array.from(replacementSampler).join(",") !== "0,1,1" ||
    batchSampler.kind !== "zgml.data.batch-sampler" ||
    batchSampler.__len__() !== 2 ||
    JSON.stringify(Array.from(batchSampler)) !== "[[0],[1]]" ||
    JSON.stringify(Array.from(torchBatchSampler)) !== "[[0,1]]" ||
    JSON.stringify(samplerLoaderIndices) !== "[[1],[0]]" ||
    JSON.stringify(batchSamplerLoaderIndices) !== "[[0],[1]]" ||
    defaultCollatedBatch.kind !== "zgml.data.batch" ||
    defaultCollatedBatch.batchIndex !== 4 ||
    defaultCollatedBatch.indices.join(",") !== "1,0" ||
    defaultCollatedBatch.input.data.join(",") !== "0,1,1,0" ||
    defaultCollatedBatch.target.data.join(",") !== "1,0" ||
    defaultCollateLoaderBatch.input.shape.join("x") !== "2x2" ||
    defaultCollateLoaderBatch.indices.join(",") !== "1,0" ||
    collatedLoaderBatch.kind !== "custom-collate" ||
    collatedLoaderBatch.first !== 1 ||
    collatedLoaderBatch.sampleIndices.join(",") !== "1,0" ||
    collatedLoaderBatch.sampleIndices !== collatedLoaderBatch.sample_indices ||
    collatedLoaderBatch.batchIndex !== 0
  ) {
    throw new Error(`${label} expected data.Dataset subclasses and PyTorch-style samplers to compose with DataLoader and torch.utils.data`);
  }
  if (!Object.isFrozen(dataset) || dataset.kind !== "zgml.data.tensor-dataset" || dataset.length !== 2 || dataset.len() !== 2 || dataset.size() !== 2) {
    throw new Error(`${label} expected data.tensorDataset to return frozen dataset evidence`);
  }
  if (dataset.__len__() !== 2 || dataset.at(0).index !== 0 || dataset.__getitem__(1).index !== 1) {
    throw new Error(`${label} expected data.tensorDataset to expose PyTorch-style dataset accessors`);
  }
  if (
    typeof dataset.__iter__ !== "function" ||
    Array.from(dataset as Iterable<PackageSmokeDatasetSample>, (entry) => entry.index).join(",") !== "0,1" ||
    Array.from(dataset.__iter__() as Iterable<PackageSmokeDatasetSample>, (entry) => entry.index).join(",") !== "0,1"
  ) {
    throw new Error(`${label} expected data.tensorDataset to expose iterable sample accessors`);
  }
  const sample = dataset.sample(1);
  if (!Object.isFrozen(sample) || sample.kind !== "zgml.data.sample" || sample.index !== 1 || !(sample.input instanceof adapter.Tensor)) {
    throw new Error(`${label} expected data.tensorDataset sample evidence`);
  }
  expectClose(sample.input.data, [0, 1], `${label} data.tensorDataset sample input`);
  expectClose(sample.target.data, [1], `${label} data.tensorDataset sample target`);
  const manualBatch = dataset.batch([0, 1], 3);
  const manualIterableBatch = dataset.batch(new Set([0, 1]), 4);
  if (!Object.isFrozen(manualBatch) || !Object.isFrozen(manualBatch.indices) || manualBatch.batchIndex !== 3) {
    throw new Error(`${label} expected data.tensorDataset batch evidence`);
  }
  if (!Object.isFrozen(manualIterableBatch) || manualIterableBatch.batchIndex !== 4 || manualIterableBatch.indices.join(",") !== "0,1") {
    throw new Error(`${label} expected data.tensorDataset batch to accept iterable indices`);
  }
  if (manualBatch.input.shape.join("x") !== "2x2" || manualBatch.target.shape.join("x") !== "2x1") {
    throw new Error(`${label} expected data.tensorDataset batch to stack leading tensor rows`);
  }
  expectClose(manualBatch.input.data, [1, 0, 0, 1], `${label} data.tensorDataset batch input`);
  const splitDataset = adapter.data.tensorDataset(
    adapter.tensor([1, 0, 0, 1, 1, 1, 0, 0], [4, 2]),
    adapter.tensor([1, 1, 0, 0], [4, 1]),
  );
  const splitPair = adapter.data.randomSplit(splitDataset, [3, 1], { seed: 7 });
  const iterableSplit = adapter.data.randomSplit(splitDataset, new Set([3, 1]), { seed: 7 });
  const splitPairAlias = adapter.data.random_split(splitDataset, [2, 2], { shuffle: false });
  if (
    !Object.isFrozen(splitPair) ||
    splitPair.length !== 2 ||
    splitPair[0].length !== 3 ||
    splitPair[1].__len__() !== 1 ||
    iterableSplit.length !== 2 ||
    iterableSplit[0].length !== 3 ||
    iterableSplit[1].__len__() !== 1 ||
    !Object.isFrozen(iterableSplit[0].indices) ||
    splitPair[0].kind !== "zgml.data.tensor-dataset" ||
    splitPair[0].splitIndex !== 0 ||
    splitPair[1].split_index !== 1 ||
    !Object.isFrozen(splitPair[0].indices) ||
    splitPair[0].indices.length !== 3 ||
    splitPairAlias[0].indices.join(",") !== "0,1" ||
    splitPairAlias[1].indices.join(",") !== "2,3" ||
    Array.from(splitPair[0]).length !== 3 ||
    splitPair[0].__iter__().next().value?.kind !== "zgml.data.sample" ||
    splitPair[0].batch([0, 1]).indices.length !== 2 ||
    adapter.data.dataLoader(splitPair[0], { batch_size: 2 }).batchCount !== 2
  ) {
    throw new Error(`${label} expected data.randomSplit/random_split to return deterministic iterable subset datasets`);
  }
  const subsetDataset = adapter.data.subset(splitDataset, [2, 0]);
  const subsetIterableDataset = adapter.data.subset(splitDataset, new Set([2, 0]));
  const subsetClassDataset = new adapter.data.Subset(splitDataset, [2, 0]);
  const torchSubsetClassDataset = new adapter.torch.utils.data.Subset(splitDataset, [2, 0]);
  const takenDataset = adapter.data.take(splitDataset, 2);
  const concatDataset = adapter.data.concatDataset([takenDataset, subsetDataset]);
  const concatIterableDataset = adapter.data.concatDataset(new Set([takenDataset, subsetIterableDataset]));
  const concatAliasDataset = adapter.data.concat_dataset([subsetDataset, takenDataset]);
  const concatClassDataset = new adapter.data.ConcatDataset([takenDataset, subsetDataset]);
  const torchConcatClassDataset = new adapter.torch.utils.data.ConcatDataset([takenDataset, subsetDataset]);
  const mappedDataset = adapter.data.mapDataset(dataset, (entry: PackageSmokeDatasetSample): PackageSmokeMappedSample => ({
    ...entry,
    input: entry.input.mul(2),
    target: entry.target,
    sourceIndex: entry.index,
  }));
  const mappedAliasDataset = adapter.data.map_dataset(dataset, (entry: PackageSmokeDatasetSample): PackageSmokeDatasetSample => entry);
  const mappedClassDataset = new adapter.data.MapDataset(dataset, (entry: PackageSmokeDatasetSample): PackageSmokeDatasetSample => entry);
  const torchMappedClassDataset = new adapter.torch.utils.data.MapDataset(dataset, (entry: PackageSmokeDatasetSample): PackageSmokeDatasetSample => entry);
  const torchSplitPair = adapter.torch.utils.data.random_split(splitDataset, [2, 2], { shuffle: false });
  if (
    !Object.isFrozen(subsetDataset) ||
    subsetDataset.indices.join(",") !== "2,0" ||
    subsetClassDataset.indices.join(",") !== "2,0" ||
    torchSubsetClassDataset.indices.join(",") !== "2,0" ||
    subsetIterableDataset.indices.join(",") !== "2,0" ||
    subsetDataset.sample(0).index !== 2 ||
    subsetDataset.batch([0, 1]).indices.join(",") !== "2,0" ||
    subsetIterableDataset.batch(new Set([0, 1])).indices.join(",") !== "2,0" ||
    takenDataset.indices.join(",") !== "0,1" ||
    takenDataset.__getitem__(1).index !== 1 ||
    !Object.isFrozen(concatDataset) ||
    concatDataset.kind !== "zgml.data.concat-dataset" ||
    concatClassDataset.length !== 4 ||
    torchConcatClassDataset.length !== 4 ||
    concatDataset.offsets.join(",") !== "0,2" ||
    concatDataset.sample(2).source_index !== 1 ||
    concatDataset.sample(2).local_index !== 0 ||
    concatDataset.batch([0, 2]).input.shape.join("x") !== "2x2" ||
    concatDataset.batch([0, 2]).source_indices.join(",") !== "0,1" ||
    concatIterableDataset.batch(new Set([0, 2])).source_indices.join(",") !== "0,1" ||
    concatAliasDataset.sample(0).sourceIndex !== 0 ||
    !Object.isFrozen(mappedDataset) ||
    mappedDataset.kind !== "zgml.data.mapped-dataset" ||
    mappedClassDataset.length !== dataset.length ||
    torchMappedClassDataset.length !== dataset.length ||
    mappedDataset.__len__() !== 2 ||
    mappedDataset.at(1).sourceIndex !== 1 ||
    mappedDataset.sample(1).input.data.join(",") !== "0,2" ||
    mappedDataset.batch([0, 1]).input.shape.join("x") !== "2x2" ||
    mappedDataset.batch([0, 1]).sourceIndex.join(",") !== "0,1" ||
    mappedAliasDataset.sample(0).input.data.join(",") !== "1,0" ||
    torchSplitPair[0].indices.join(",") !== "0,1" ||
    torchSplitPair[1].indices.join(",") !== "2,3"
  ) {
    throw new Error(`${label} expected data subset/take/concatDataset/mapDataset helpers and constructors to compose with tensor datasets and loaders`);
  }
  const shuffledBatches = adapter.data.batches(dataset, { batchSize: 1, shuffle: true, seed: 7 });
  if (!Object.isFrozen(shuffledBatches) || shuffledBatches.kind !== "zgml.data.batches" || shuffledBatches.batchSize !== 1) {
    throw new Error(`${label} expected data.batches to return frozen iterable evidence`);
  }
  const dataLoader = adapter.data.dataLoader(dataset, { batch_size: 2, drop_last: false });
  const dataloader = adapter.data.dataloader(dataset, { batch_size: 2, drop_last: false });
  const classDataset = new adapter.data.TensorDataset(
    adapter.tensor([1, 0, 0, 1], [2, 2]),
    adapter.tensor([1, 0], [2, 1]),
  );
  const classDataLoader = new adapter.data.DataLoader(classDataset, { batch_size: 2, drop_last: false });
  if (
    !Object.isFrozen(dataLoader) ||
    !Object.isFrozen(dataloader) ||
    !Object.isFrozen(classDataset) ||
    !Object.isFrozen(classDataLoader) ||
    dataLoader.kind !== "zgml.data.batches" ||
    dataloader.kind !== "zgml.data.batches" ||
    classDataset.kind !== "zgml.data.tensor-dataset" ||
    classDataLoader.kind !== "zgml.data.batches" ||
    classDataset.__getitem__(1).target.data.join(",") !== "0" ||
    classDataLoader.__getitem__(0).input.shape.join("x") !== "2x2" ||
    dataLoader.sampleCount !== 2 ||
    dataLoader.sample_count !== 2 ||
    dataLoader.batchSize !== 2 ||
    dataLoader.batch_size !== 2 ||
    dataloader.batchSize !== 2 ||
    dataloader.batch_size !== 2 ||
    dataLoader.batchCount !== 1 ||
    dataLoader.batch_count !== 1 ||
    dataLoader.len() !== 1 ||
    dataLoader.__len__() !== 1 ||
    dataLoader.size() !== 1 ||
    dataloader.batchCount !== 1 ||
    dataloader.batch_count !== 1 ||
    dataLoader.dropLast !== false ||
    dataLoader.drop_last !== false ||
    dataloader.dropLast !== false ||
    dataloader.drop_last !== false ||
    dataLoader.get(0).batchIndex !== 0 ||
    dataLoader.at(0).indices.length !== 2 ||
    dataLoader.__getitem__(0).kind !== "zgml.data.batch" ||
    typeof dataLoader.__iter__ !== "function" ||
    dataLoader.__iter__().next().value?.kind !== "zgml.data.batch" ||
    Array.from(dataLoader).length !== 1 ||
    Array.from(dataLoader.__iter__()).length !== 1 ||
    Array.from(dataloader).length !== 1
  ) {
    throw new Error(`${label} expected dataLoader/dataloader to expose DataLoader iterable evidence`);
  }
  const droppedLoader = adapter.data.dataLoader(dataset, { batch_size: 2, drop_last: true });
  const overfullDroppedLoader = adapter.data.dataLoader(dataset, { batch_size: 3, drop_last: true });
  if (
    droppedLoader.batchCount !== 1 ||
    droppedLoader.batch_count !== 1 ||
    Array.from(droppedLoader).length !== droppedLoader.batchCount ||
    overfullDroppedLoader.batchCount !== 0 ||
    overfullDroppedLoader.batch_count !== 0 ||
    Array.from(overfullDroppedLoader).length !== 0
  ) {
    throw new Error(`${label} expected DataLoader batchCount/batch_count to match emitted batches`);
  }
  const shuffledBatchList = Array.from(shuffledBatches) as PackageSmokeTrainBatch[];
  if (shuffledBatchList.length !== 2 || shuffledBatchList[0].indices.length !== 1 || shuffledBatchList[1].indices.length !== 1) {
    throw new Error(`${label} expected data.batches to yield one-row batches`);
  }
  const fitModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
  const fitOptimizer = adapter.optim.sgd(fitModel, { lr: 0.05 });
  const fitSteps: PackageSmokeTrainStepEvidence[] = [];
  const fitEvidence = adapter.train.fit(fitOptimizer, shuffledBatches, (batch: PackageSmokeTrainBatch, context: PackageSmokeTrainContext) => {
    if (!Object.isFrozen(context) || context.epoch !== 0 || context.step !== context.batchIndex) {
      throw new Error(`${label} expected train.fit to pass frozen epoch/batch/step context`);
    }
    if (
      !Object.isFrozen(context.sampleIndices) ||
      context.sampleIndices !== context.sample_indices ||
      context.sampleIndices.length !== 1 ||
      context.sampleIndices[0] !== batch.indices[0]
    ) {
      throw new Error(`${label} expected train.fit context to preserve DataLoader sample indices`);
    }
    return adapter.loss.mse(fitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
  }, {
    epochs: 2,
    maxSteps: 2,
    onStep: (evidence: PackageSmokeTrainStepEvidence) => fitSteps.push(evidence),
  });
  if (
    !Object.isFrozen(fitEvidence) ||
    fitEvidence.kind !== "zgml.train.fit" ||
    typeof fitEvidence.signature !== "string" ||
    !fitEvidence.signature.startsWith("train-fit|epochs=2|steps=2|stopped=1|stopReason=max-steps|batchCount=2|sampleCount=2|losses=") ||
    fitEvidence.epochs !== 2 ||
    fitEvidence.steps !== 2 ||
    fitEvidence.stoppedEarly !== true ||
    fitEvidence.stopReason !== "max-steps" ||
    fitEvidence.stop_reason !== "max-steps" ||
    fitEvidence.batchCount !== 2 ||
    fitEvidence.batch_count !== 2 ||
    fitEvidence.sampleCount !== 2 ||
    fitEvidence.sample_count !== 2 ||
    fitEvidence.bestLoss !== Math.min(...fitEvidence.losses) ||
    fitEvidence.best_loss !== fitEvidence.bestLoss ||
    fitEvidence.bestStep !== fitEvidence.losses.indexOf(fitEvidence.bestLoss) + 1 ||
    fitEvidence.best_step !== fitEvidence.bestStep ||
    fitEvidence.finalLoss !== fitEvidence.losses[1] ||
    !(fitEvidence.lastLoss instanceof adapter.Tensor) ||
    !Object.isFrozen(fitEvidence.losses) ||
    !Object.isFrozen(fitEvidence.lastStep) ||
    fitEvidence.lastStep.afterStep !== 2 ||
    fitSteps.length !== 2 ||
    !Object.isFrozen(fitSteps[0]) ||
    fitSteps[0].kind !== "zgml.train.fit-step" ||
    typeof fitSteps[0].signature !== "string" ||
    !fitSteps[0].signature.startsWith(`train-fit-step|epoch=0|batch=0|step=1|samples=${fitSteps[0].sampleIndices.join(",")}|loss=`) ||
    !Object.isFrozen(fitSteps[0].sampleIndices) ||
    fitSteps[0].sampleIndices !== fitSteps[0].sample_indices ||
    fitSteps[0].sampleIndices.length !== 1 ||
    fitSteps[1].step !== 2
  ) {
    throw new Error(`${label} expected train.fit to return frozen fit evidence and per-step callbacks`);
  }
  if (
    !adapter.train.isTrainFitEvidence(fitEvidence) ||
    adapter.train.requireTrainFitEvidence(fitEvidence).steps !== 2 ||
    adapter.train.assertTrainFitEvidence(fitEvidence).lastStep?.afterStep !== 2 ||
    adapter.train.assert_train_fit_evidence(fitEvidence).signature !== fitEvidence.signature ||
    !adapter.train.matchesTrainFitEvidenceSignature(fitEvidence, fitEvidence.signature) ||
    adapter.train.matches_train_fit_evidence_signature(fitEvidence, "wrong") ||
    adapter.train.isTrainFitEvidence({ ...fitEvidence })
  ) {
    throw new Error(`${label} expected train namespace TrainFitEvidence validators to accept only frozen signed evidence`);
  }
  const sampleAliasFitSteps: PackageSmokeTrainStepEvidence[] = [];
  const sampleAliasFitEvidence = adapter.train.fit(adapter.optim.sgd(fitModel, { lr: 0.01 }), shuffledBatchList.map((batch): PackageSmokeSampleAliasBatch => Object.freeze({
    input: batch.input,
    target: batch.target,
    sampleIndices: batch.indices,
    sample_indices: batch.indices,
  })), (batch: PackageSmokeSampleAliasBatch, context: PackageSmokeTrainContext) => {
    if (
      !Object.isFrozen(context.sampleIndices) ||
      context.sampleIndices !== context.sample_indices ||
      context.sampleIndices[0] !== batch.sampleIndices[0]
    ) {
      throw new Error(`${label} expected train.fit context to accept sampleIndices/sample_indices aliases`);
    }
    return adapter.loss.mse(fitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
  }, { maxSteps: 1, onStep: (evidence: PackageSmokeTrainStepEvidence) => sampleAliasFitSteps.push(evidence) });
  if (
    sampleAliasFitEvidence.steps !== 1 ||
    sampleAliasFitSteps[0]?.sampleIndices?.[0] !== shuffledBatchList[0].indices[0]
  ) {
    throw new Error(`${label} expected train.fit evidence to preserve sampleIndices alias batches`);
  }
  const earlyStopModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
  const earlyStopOptimizer = adapter.optim.sgd(earlyStopModel, { lr: 0.01 });
  const earlyStopEvidence = adapter.train.fit(earlyStopOptimizer, shuffledBatchList, () => (
    adapter.tensor([1], [1])
  ), {
    epochs: 3,
    earlyStopping: { patience: 0, minDelta: 0, mode: "min" },
  });
  if (
    earlyStopEvidence.steps !== 2 ||
    earlyStopEvidence.stoppedEarly !== true ||
    earlyStopEvidence.stopReason !== "early-stopping" ||
    earlyStopEvidence.stop_reason !== "early-stopping" ||
    earlyStopEvidence.bestLoss !== 1 ||
    earlyStopEvidence.best_loss !== 1 ||
    earlyStopEvidence.bestStep !== 1 ||
    earlyStopEvidence.best_step !== 1 ||
    !adapter.train.isTrainFitEvidence(earlyStopEvidence)
  ) {
    throw new Error(`${label} expected train.fit earlyStopping to stop on non-improving loss with signed evidence`);
  }
  const fitModuleModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
  const fitModuleOptimizer = adapter.optim.sgd(fitModuleModel, { lr: 0.05 });
  const fitModuleCriterion = new adapter.nn.MSELoss();
  const fitModuleEvidence = adapter.train.fitModule(fitModuleOptimizer, fitModuleModel, shuffledBatches, fitModuleCriterion, {
    maxSteps: 1,
    zeroGrad: true,
  });
  const fitModelFirstModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
  const fitModelFirstOptimizer = adapter.optim.sgd(fitModelFirstModel, { lr: 0.05 });
  const fitModelFirstEvidence = adapter.train.fit(fitModelFirstModel, shuffledBatches, {
    optimizer: fitModelFirstOptimizer,
    loss: fitModuleCriterion,
    maxSteps: 1,
    zeroGrad: true,
  });
  const fitModuleSnakeEvidence = adapter.train.fit_module(fitModuleOptimizer, fitModuleModel, shuffledBatches, fitModuleCriterion, {
    max_steps: 1,
    zero_grad: true,
  });
  const fitModuleStepOk = fitModuleEvidence.lastStep?.gradientsCleared === true || fitModuleEvidence.native === true;
  const fitModelFirstStepOk = fitModelFirstEvidence.lastStep?.gradientsCleared === true || fitModelFirstEvidence.native === true;
  const fitModuleSnakeStepOk = fitModuleSnakeEvidence.lastStep?.gradientsCleared === true || fitModuleSnakeEvidence.native === true;
  if (
    fitModuleEvidence.kind !== "zgml.train.fit" ||
    fitModuleEvidence.steps !== 1 ||
    !fitModuleStepOk ||
    fitModelFirstEvidence.kind !== "zgml.train.fit" ||
    fitModelFirstEvidence.steps !== 1 ||
    !fitModelFirstStepOk ||
    fitModuleSnakeEvidence.kind !== "zgml.train.fit" ||
    fitModuleSnakeEvidence.steps !== 1 ||
    !fitModuleSnakeStepOk
  ) {
    throw new Error(`${label} expected train.fit model-first and fitModule/fit_module to train module+criterion batches`);
  }
  const hasNativeTrainingSurface = Boolean(adapter.nativeEager && (adapter.compileForTraining || adapter.compile?.compileForTraining));
  if (hasNativeTrainingSurface) {
    const moduleCompileModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
    const moduleCompileOptimizer = adapter.optim.sgd(moduleCompileModel, { lr: 0.05 });
    const moduleCompiledTrainer = moduleCompileModel.compileForTraining(moduleCompileOptimizer, {
      inputShape: [1, 2],
      loss: "mse",
    });
    const moduleCompileBatch = shuffledBatches[0];
    const moduleCompileStep = moduleCompiledTrainer.step(moduleCompileBatch.input, moduleCompileBatch.target);
    const moduleCompiledTrainerSnake = moduleCompileModel.compile_for_training(moduleCompileOptimizer, {
      input_shape: [1, 2],
      criterion: "mse",
    });
    if (
      moduleCompiledTrainer.kind !== "zgml.compiled-training-step" ||
      moduleCompiledTrainer.native !== true ||
      moduleCompiledTrainer.backend !== "cpu" ||
      moduleCompiledTrainer.modelKind !== "linear" ||
      moduleCompiledTrainer.lossKind !== "mse" ||
      moduleCompileStep.kind !== "zgml.native-training-step" ||
      moduleCompileStep.native !== true ||
      moduleCompiledTrainerSnake.kind !== "zgml.compiled-training-step"
    ) {
      throw new Error(`${label} expected module.compileForTraining to produce a native Zig training handle`);
    }
    const nativeFitModuleModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
    const nativeFitModuleOptimizer = adapter.optim.sgd(nativeFitModuleModel, { lr: 0.05 });
    const nativeFitModuleEvidence = adapter.train.fitModule(nativeFitModuleOptimizer, nativeFitModuleModel, shuffledBatches, fitModuleCriterion, {
      maxSteps: 1,
      requireNative: true,
    });
    if (
      nativeFitModuleEvidence.kind !== "zgml.train.fit" ||
      nativeFitModuleEvidence.steps !== 1 ||
      nativeFitModuleEvidence.native !== true ||
      nativeFitModuleEvidence.backend !== "cpu"
    ) {
      throw new Error(`${label} expected train.fitModule requireNative to route supported Linear+MSE training through native Zig`);
    }
  }
  if (
    !adapter.train.isTrainFitStepEvidence(fitSteps[0]) ||
    adapter.train.requireTrainFitStepEvidence(fitSteps[0]).step !== 1 ||
    adapter.train.assertTrainFitStepEvidence(fitSteps[0]).stepEvidence?.kind !== "zgml.train.step" ||
    adapter.train.assert_train_fit_step_evidence(fitSteps[0]).signature !== fitSteps[0].signature ||
    !adapter.train.matchesTrainFitStepEvidenceSignature(fitSteps[0], fitSteps[0].signature) ||
    adapter.train.matches_train_fit_step_evidence_signature(fitSteps[0], "wrong") ||
    adapter.train.isTrainFitStepEvidence({ ...fitSteps[0] })
  ) {
    throw new Error(`${label} expected train namespace TrainFitStepEvidence validators to accept only frozen signed evidence`);
  }
  if (!(fitEvidence.losses[1] < fitEvidence.losses[0])) throw new Error(`${label} expected train.fit loss to improve over simple batches`);
  const evaluateSteps: PackageSmokeTrainStepEvidence[] = [];
  const evaluateEvidence = adapter.train.evaluate(shuffledBatches, (batch: PackageSmokeTrainBatch, context: PackageSmokeTrainContext) => {
    if (
      !Object.isFrozen(context) ||
      context.batchIndex !== context.step ||
      !Object.isFrozen(context.sampleIndices) ||
      context.sampleIndices !== context.sample_indices ||
      context.sampleIndices[0] !== batch.indices[0]
    ) {
      throw new Error(`${label} expected train.evaluate to pass frozen batch context with sample indices`);
    }
    return adapter.loss.mse(fitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
  }, {
    maxSteps: 1,
    onStep: (evidence: PackageSmokeTrainStepEvidence) => evaluateSteps.push(evidence),
  });
  if (
    !Object.isFrozen(evaluateEvidence) ||
    evaluateEvidence.kind !== "zgml.train.evaluate" ||
    typeof evaluateEvidence.signature !== "string" ||
    !evaluateEvidence.signature.startsWith("train-evaluate|steps=1|stopped=1|batchCount=2|sampleCount=2|losses=") ||
    evaluateEvidence.steps !== 1 ||
    evaluateEvidence.stoppedEarly !== true ||
    evaluateEvidence.batchCount !== 2 ||
    evaluateEvidence.batch_count !== 2 ||
    evaluateEvidence.sampleCount !== 2 ||
    evaluateEvidence.sample_count !== 2 ||
    evaluateEvidence.meanLoss !== evaluateEvidence.losses[0] ||
    evaluateEvidence.mean_loss !== evaluateEvidence.meanLoss ||
    evaluateEvidence.finalLoss !== evaluateEvidence.losses[0] ||
    evaluateEvidence.final_loss !== evaluateEvidence.finalLoss ||
    !(evaluateEvidence.lastLoss instanceof adapter.Tensor) ||
    !Object.isFrozen(evaluateEvidence.losses) ||
    evaluateSteps.length !== 1 ||
    !Object.isFrozen(evaluateSteps[0]) ||
    evaluateSteps[0].kind !== "zgml.train.evaluate-step" ||
    typeof evaluateSteps[0].signature !== "string" ||
    !evaluateSteps[0].signature.startsWith(`train-evaluate-step|batch=0|step=1|samples=${evaluateSteps[0].sampleIndices.join(",")}|loss=`) ||
    !Object.isFrozen(evaluateSteps[0].sampleIndices) ||
    evaluateSteps[0].sampleIndices !== evaluateSteps[0].sample_indices
  ) {
    throw new Error(`${label} expected train.evaluate to return frozen evaluation evidence and per-step callbacks`);
  }
  if (
    !adapter.train.isTrainEvaluateEvidence(evaluateEvidence) ||
    adapter.train.requireTrainEvaluateEvidence(evaluateEvidence).steps !== 1 ||
    adapter.train.assertTrainEvaluateEvidence(evaluateEvidence).meanLoss !== evaluateEvidence.meanLoss ||
    adapter.train.assert_train_evaluate_evidence(evaluateEvidence).signature !== evaluateEvidence.signature ||
    !adapter.train.matchesTrainEvaluateEvidenceSignature(evaluateEvidence, evaluateEvidence.signature) ||
    adapter.train.matches_train_evaluate_evidence_signature(evaluateEvidence, "wrong") ||
    adapter.train.isTrainEvaluateEvidence({ ...evaluateEvidence })
  ) {
    throw new Error(`${label} expected train namespace TrainEvaluateEvidence validators to accept only frozen signed evidence`);
  }
  if (
    !adapter.train.isTrainEvaluateStepEvidence(evaluateSteps[0]) ||
    adapter.train.requireTrainEvaluateStepEvidence(evaluateSteps[0]).step !== 1 ||
    adapter.train.assertTrainEvaluateStepEvidence(evaluateSteps[0]).signature !== evaluateSteps[0].signature ||
    adapter.train.assert_train_evaluate_step_evidence(evaluateSteps[0]).loss !== evaluateSteps[0].loss ||
    !adapter.train.matchesTrainEvaluateStepEvidenceSignature(evaluateSteps[0], evaluateSteps[0].signature) ||
    adapter.train.matches_train_evaluate_step_evidence_signature(evaluateSteps[0], "wrong") ||
    adapter.train.isTrainEvaluateStepEvidence({ ...evaluateSteps[0] })
  ) {
    throw new Error(`${label} expected train namespace TrainEvaluateStepEvidence validators to accept only frozen signed evidence`);
  }
  const evaluateAliasEvidence = adapter.train.evaluate_loss(shuffledBatchList, (batch: PackageSmokeTrainBatch) => (
    adapter.loss.mse(fitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0))
  ), { max_steps: 1 });
  const evaluateModuleEvidence = adapter.train.evaluateModule(fitModel, shuffledBatches, new adapter.nn.MSELoss(), { maxSteps: 1 });
  const evaluateModuleSnakeEvidence = adapter.train.evaluate_module(fitModel, shuffledBatches, new adapter.nn.MSELoss(), { max_steps: 1 });
  if (
    evaluateAliasEvidence.steps !== 1 ||
    evaluateAliasEvidence.stoppedEarly !== true ||
    evaluateAliasEvidence.batchCount !== null ||
    evaluateAliasEvidence.batch_count !== null ||
    evaluateAliasEvidence.sampleCount !== null ||
    evaluateAliasEvidence.sample_count !== null ||
    evaluateModuleEvidence.kind !== "zgml.train.evaluate" ||
    evaluateModuleEvidence.steps !== 1 ||
    evaluateModuleSnakeEvidence.kind !== "zgml.train.evaluate" ||
    evaluateModuleSnakeEvidence.steps !== 1 ||
    !adapter.train.isTrainEvaluateEvidence(evaluateModuleEvidence)
  ) {
    throw new Error(`${label} expected train.evaluate_loss/evaluateModule aliases to preserve signed evaluation evidence`);
  }
  const predictSteps: PackageSmokeTrainStepEvidence[] = [];
  const predictEvidence = adapter.train.predict(shuffledBatches, (batch: PackageSmokeTrainBatch, context: PackageSmokeTrainContext) => {
    if (
      !Object.isFrozen(context) ||
      context.batchIndex !== context.step ||
      !Object.isFrozen(context.sampleIndices) ||
      context.sampleIndices !== context.sample_indices ||
      context.sampleIndices[0] !== batch.indices[0]
    ) {
      throw new Error(`${label} expected train.predict to pass frozen batch context with sample indices`);
    }
    return fitModel.forward(batch.input.select(0, 0));
  }, {
    maxSteps: 1,
    onStep: (evidence: PackageSmokeTrainStepEvidence) => predictSteps.push(evidence),
  });
  const predictAliasEvidence = adapter.train.predict_batches(shuffledBatchList, (batch: PackageSmokeTrainBatch) => (
    fitModel.forward(batch.input.select(0, 0))
  ), { max_steps: 1 });
  const predictModuleEvidence = adapter.train.predictModule(fitModel, shuffledBatches, { maxSteps: 1 });
  const predictModuleSnakeEvidence = adapter.train.predict_module(fitModel, shuffledBatches, { max_steps: 1 });
  const predictClassifierEvidence = adapter.train.predictClassifier(fitModel, shuffledBatches, { maxSteps: 1 });
  const predictClassifierSnakeEvidence = adapter.train.predict_classifier(fitModel, shuffledBatches, { max_steps: 1 });
  if (
    !Object.isFrozen(predictEvidence) ||
    predictEvidence.kind !== "zgml.train.predict" ||
    typeof predictEvidence.signature !== "string" ||
    !predictEvidence.signature.startsWith("train-predict|steps=1|stopped=1|batchCount=2|sampleCount=2|outputs=1|lastOutput=1") ||
    predictEvidence.steps !== 1 ||
    predictEvidence.stoppedEarly !== true ||
    predictEvidence.batchCount !== 2 ||
    predictEvidence.batch_count !== 2 ||
    predictEvidence.sampleCount !== 2 ||
    predictEvidence.sample_count !== 2 ||
    !Object.isFrozen(predictEvidence.outputs) ||
    predictEvidence.outputs.length !== 1 ||
    !(predictEvidence.outputs[0] instanceof adapter.Tensor) ||
    predictEvidence.lastOutput !== predictEvidence.outputs[0] ||
    predictEvidence.last_output !== predictEvidence.lastOutput ||
    predictSteps.length !== 1 ||
    !Object.isFrozen(predictSteps[0]) ||
    predictSteps[0].kind !== "zgml.train.predict-step" ||
    !predictSteps[0].signature.startsWith(`train-predict-step|batch=0|step=1|samples=${predictSteps[0].sampleIndices.join(",")}|output=1`) ||
    predictSteps[0].sampleIndices !== predictSteps[0].sample_indices ||
    predictSteps[0].output !== predictEvidence.outputs[0] ||
    predictAliasEvidence.steps !== 1 ||
    predictAliasEvidence.batchCount !== null ||
    predictAliasEvidence.outputs.length !== 1 ||
    predictModuleEvidence.kind !== "zgml.train.predict" ||
    predictModuleEvidence.steps !== 1 ||
    predictModuleEvidence.outputs.length !== 1 ||
    predictModuleSnakeEvidence.kind !== "zgml.train.predict" ||
    predictModuleSnakeEvidence.steps !== 1 ||
    predictClassifierEvidence.kind !== "zgml.train.predict" ||
    predictClassifierEvidence.steps !== 1 ||
    predictClassifierSnakeEvidence.kind !== "zgml.train.predict" ||
    predictClassifierSnakeEvidence.steps !== 1 ||
    !adapter.train.isTrainPredictEvidence(predictModuleEvidence)
  ) {
    throw new Error(`${label} expected train.predict/predictModule/predictClassifier aliases to return frozen prediction evidence`);
  }
  if (
    !adapter.train.isTrainPredictEvidence(predictEvidence) ||
    adapter.train.requireTrainPredictEvidence(predictEvidence).outputs.length !== 1 ||
    adapter.train.assertTrainPredictEvidence(predictEvidence).lastOutput !== predictEvidence.lastOutput ||
    adapter.train.assert_train_predict_evidence(predictEvidence).signature !== predictEvidence.signature ||
    !adapter.train.matchesTrainPredictEvidenceSignature(predictEvidence, predictEvidence.signature) ||
    adapter.train.matches_train_predict_evidence_signature(predictEvidence, "wrong") ||
    adapter.train.isTrainPredictEvidence({ ...predictEvidence })
  ) {
    throw new Error(`${label} expected train namespace TrainPredictEvidence validators to accept only frozen signed evidence`);
  }
  if (
    !adapter.train.isTrainPredictStepEvidence(predictSteps[0]) ||
    adapter.train.requireTrainPredictStepEvidence(predictSteps[0]).step !== 1 ||
    adapter.train.assertTrainPredictStepEvidence(predictSteps[0]).signature !== predictSteps[0].signature ||
    adapter.train.assert_train_predict_step_evidence(predictSteps[0]).output !== predictSteps[0].output ||
    !adapter.train.matchesTrainPredictStepEvidenceSignature(predictSteps[0], predictSteps[0].signature) ||
    adapter.train.matches_train_predict_step_evidence_signature(predictSteps[0], "wrong") ||
    adapter.train.isTrainPredictStepEvidence({ ...predictSteps[0] })
  ) {
    throw new Error(`${label} expected train namespace TrainPredictStepEvidence validators to accept only frozen signed evidence`);
  }
  const snakeFitModel = adapter.nn.linear(2, 1, { weights: [0, 0], bias: [0] });
  const snakeFitOptimizer = adapter.optim.sgd(snakeFitModel, { lr: 0.05 });
  const snakeFitSteps: PackageSmokeTrainStepEvidence[] = [];
  const snakeFitEvidence = adapter.train.fit(snakeFitOptimizer, shuffledBatches, (batch: PackageSmokeTrainBatch) => (
    adapter.loss.mse(snakeFitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0))
  ), {
    max_steps: 1,
    zero_grad: true,
    on_step: (evidence: PackageSmokeTrainStepEvidence) => snakeFitSteps.push(evidence),
  });
  if (
    snakeFitEvidence.steps !== 1 ||
    snakeFitEvidence.stoppedEarly !== true ||
    snakeFitEvidence.batchCount !== 2 ||
    snakeFitEvidence.batch_count !== 2 ||
    snakeFitEvidence.sampleCount !== 2 ||
    snakeFitEvidence.sample_count !== 2 ||
    snakeFitSteps.length !== 1 ||
    !Object.isFrozen(snakeFitSteps[0].sampleIndices) ||
    snakeFitSteps[0].sampleIndices !== snakeFitSteps[0].sample_indices ||
    snakeFitSteps[0].stepEvidence?.zeroGradApplied !== true ||
    snakeFitSteps[0].stepEvidence?.gradientsCleared !== true
  ) {
    throw new Error(`${label} expected train.fit max_steps/on_step/zero_grad aliases to drive fit evidence`);
  }
  const plainFitEvidence = adapter.train.fit(snakeFitOptimizer, shuffledBatchList, (batch: PackageSmokeTrainBatch) => (
    adapter.loss.mse(snakeFitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0))
  ), { max_steps: 1 });
  if (
    plainFitEvidence.batchCount !== null ||
    plainFitEvidence.batch_count !== null ||
    plainFitEvidence.sampleCount !== null ||
    plainFitEvidence.sample_count !== null ||
    plainFitEvidence.lastStep === null
  ) {
    throw new Error(`${label} expected train.fit plain iterable count evidence to stay nullable`);
  }
  const plainFitSteps: PackageSmokeTrainStepEvidence[] = [];
  adapter.train.fit(snakeFitOptimizer, shuffledBatchList, (batch: PackageSmokeTrainBatch, context: PackageSmokeTrainContext) => {
    if (
      !Object.isFrozen(context.sampleIndices) ||
      context.sampleIndices !== context.sample_indices ||
      context.sampleIndices.length !== 1 ||
      context.sampleIndices[0] !== batch.indices[0]
    ) {
      throw new Error(`${label} expected train.fit plain iterable to preserve batch-provided sample indices`);
    }
    return adapter.loss.mse(snakeFitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
  }, { max_steps: 1, onStep: (evidence: PackageSmokeTrainStepEvidence) => plainFitSteps.push(evidence) });
  if (
    plainFitSteps.length !== 1 ||
    !Object.isFrozen(plainFitSteps[0].sampleIndices) ||
    plainFitSteps[0].sampleIndices !== plainFitSteps[0].sample_indices
  ) {
    throw new Error(`${label} expected train.fit plain iterable step to preserve batch-provided sample indices`);
  }
  const unindexedFitSteps: PackageSmokeNullableTrainStepEvidence[] = [];
  adapter.train.fit(snakeFitOptimizer, [{ input: shuffledBatchList[0].input, target: shuffledBatchList[0].target }], (batch: PackageSmokeUnindexedBatch, context: PackageSmokeNullableTrainContext) => {
    if (context.sampleIndices !== null || context.sample_indices !== null) {
      throw new Error(`${label} expected train.fit unindexed batch context sample indices to stay nullable`);
    }
    return adapter.loss.mse(snakeFitModel.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
  }, { max_steps: 1, onStep: (evidence: PackageSmokeNullableTrainStepEvidence) => unindexedFitSteps.push(evidence) });
  if (unindexedFitSteps.length !== 1 || unindexedFitSteps[0].sampleIndices !== null || unindexedFitSteps[0].sample_indices !== null) {
    throw new Error(`${label} expected train.fit unindexed batch step sample indices to stay nullable`);
  }
  const scheduledFitModel = adapter.nn.sequential([
    adapter.nn.linear(2, 3),
    adapter.nn.relu(),
    adapter.nn.linear(3, 1),
  ]);
  const scheduledFitOptimizer = adapter.optim.adamW(scheduledFitModel, { lr: 0.02, weightDecay: 0.001 });
  const scheduledFitScheduler = adapter.optim.stepLR(scheduledFitOptimizer, { stepSize: 2, gamma: 0.5 });
  const scheduledFitDataset = adapter.data.tensorDataset(
    adapter.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
    adapter.tensor([0, 1, 1, 0], [4, 1]),
  );
  const scheduledFitEvidence = adapter.train.fit(
    scheduledFitOptimizer,
    adapter.data.dataLoader(scheduledFitDataset, { batchSize: 2, shuffle: true, seed: 19 }),
    (batch: PackageSmokeTrainBatch) => {
      if (!batch.target) throw new Error(`${label} expected scheduled fit batch targets`);
      return adapter.loss.mseLoss().__call__(scheduledFitModel.call(batch.input), batch.target);
    },
    {
      epochs: 2,
      zeroGrad: true,
      clipGradNorm: 1,
      onStep: () => {
        scheduledFitScheduler.step();
      },
    },
  );
  const scheduledFitSchedulerState = scheduledFitScheduler.stateDict();
  if (
    scheduledFitEvidence.steps !== 4 ||
    scheduledFitSchedulerState.step !== scheduledFitEvidence.steps ||
    scheduledFitOptimizer.config().lr !== scheduledFitSchedulerState.lastLr ||
    scheduledFitEvidence.lastStep?.clipGradNormApplied !== true
  ) {
    throw new Error(`${label} expected train.fit onStep scheduler and clipping evidence`);
  }
  const scheduledFitSnapshot = adapter.checkpoint.create({
    model: scheduledFitModel,
    optimizer: scheduledFitOptimizer,
    scheduler: scheduledFitScheduler,
    prefix: "scheduled",
  });
  const scheduledFitInspection = adapter.checkpoint.inspect(scheduledFitSnapshot);
  if (
    scheduledFitInspection.hasModel !== true ||
    scheduledFitInspection.hasOptimizer !== true ||
    scheduledFitInspection.hasScheduler !== true ||
    scheduledFitInspection.schedulerKind !== "step-lr" ||
    scheduledFitInspection.schedulerStep !== scheduledFitEvidence.steps ||
    scheduledFitInspection.schedulerOptimizerKind !== "adamw"
  ) {
    throw new Error(`${label} expected train.fit scheduler checkpoint inspection evidence`);
  }
  const scheduledFitRestored = adapter.nn.sequential([
    adapter.nn.linear(2, 3),
    adapter.nn.relu(),
    adapter.nn.linear(3, 1),
  ]);
  const scheduledFitRestoredOptimizer = adapter.optim.adamW(scheduledFitRestored, { lr: 0.02, weightDecay: 0.001 });
  const scheduledFitRestoredScheduler = adapter.optim.stepLR(scheduledFitRestoredOptimizer, { stepSize: 2, gamma: 0.5 });
  adapter.checkpoint.restore(scheduledFitSnapshot, {
    model: scheduledFitRestored,
    optimizer: scheduledFitRestoredOptimizer,
    scheduler: scheduledFitRestoredScheduler,
    strict: true,
    prefix: "scheduled",
  });
  if (
    scheduledFitRestoredOptimizer.stateDict().step !== scheduledFitOptimizer.stateDict().step ||
    scheduledFitRestoredScheduler.stateDict().step !== scheduledFitSchedulerState.step ||
    scheduledFitRestoredOptimizer.config().lr !== scheduledFitSchedulerState.lastLr
  ) {
    throw new Error(`${label} expected train.fit checkpoint restore to preserve optimizer and scheduler state`);
  }
  const clipModel = adapter.nn.linear(2, 1, { weights: [1, 1], bias: [0] });
  adapter.loss.mse(clipModel.forward(adapter.tensor([1, 1], [2])), adapter.tensor([0], [1])).backward();
  const observedGradNorm = adapter.train.gradNorm(clipModel);
  if (!(observedGradNorm > 6) || observedGradNorm !== adapter.train.grad_norm(clipModel)) {
    throw new Error(`${label} expected train.gradNorm/grad_norm to report unclipped gradients`);
  }
  const unclippedNorm = adapter.train.clipGradNorm(clipModel, 0.5);
  if (!(unclippedNorm > 6)) throw new Error(`${label} expected clipGradNorm to return the unclipped norm`);
  if (unclippedNorm !== observedGradNorm) throw new Error(`${label} expected clipGradNorm to share gradNorm evidence`);
  if (gradNorm(clipModel.parameters()) > 0.50001) throw new Error(`${label} expected clipGradNorm to scale gradients in place`);
  adapter.train.clip_grad_norm_(clipModel, 0.5);
  clipModel.requiresGrad_(false);
  if (adapter.train.gradNorm(clipModel) !== 0) throw new Error(`${label} expected train.gradNorm to ignore frozen stale gradients`);
  const valueClipModel = adapter.nn.linear(2, 1, { weights: [1, 1], bias: [0] });
  adapter.loss.mse(valueClipModel.forward(adapter.tensor([1, 1], [2])), adapter.tensor([0], [1])).backward();
  if (adapter.train.clipGradValue(valueClipModel, 0.25) !== valueClipModel) throw new Error(`${label} expected clipGradValue to return its target`);
  if (gradMaxAbs(valueClipModel.parameters()) > 0.25001) throw new Error(`${label} expected clipGradValue to clamp gradients in place`);
  adapter.train.clip_grad_value_(valueClipModel, 0.25);
  const adam = adapter.optim.adam(model, { lr: 0.01, weightDecay: 0.001 });
  const adamConfig = adam.config();
  if (!Object.isFrozen(adamConfig) || adamConfig.kind !== "adam" || adamConfig.lr !== 0.01 || adamConfig.weightDecay !== 0.001) {
    throw new Error(`${label} expected frozen Adam optimizer config evidence`);
  }
  adapter.optim.setLearningRate(adam, 0.002);
  if (adam.config().lr !== 0.002) throw new Error(`${label} expected optim.setLearningRate to update Adam lr`);
  adam.set_lr(0.003);
  if (
    adapter.optim.config(adam).lr !== 0.003 ||
    adam.getLearningRate() !== 0.003 ||
    adam.get_lr() !== 0.003 ||
    adapter.optim.getLearningRate(adam) !== 0.003 ||
    adapter.optim.get_lr(adam) !== 0.003
  ) throw new Error(`${label} expected optimizer.set_lr and get_lr aliases to update/read Adam lr`);
  const stepScheduler = adapter.optim.stepLR(adam, { stepSize: 2, gamma: 0.5 });
  if (stepScheduler.base_lr !== 0.003 || stepScheduler.last_lr !== 0.003 || stepScheduler.step_size !== 2) {
    throw new Error(`${label} expected StepLR snake_case runtime properties`);
  }
  if (stepScheduler.step() !== 0.003 || adapter.optim.config(adam).lr !== 0.003) {
    throw new Error(`${label} expected StepLR to preserve lr before stepSize`);
  }
  if (Math.abs(stepScheduler.step() - 0.0015) > 1e-12 || adapter.optim.config(adam).lr !== 0.0015) {
    throw new Error(`${label} expected StepLR to decay optimizer lr at stepSize`);
  }
  if (stepScheduler.lastLr !== 0.0015 || stepScheduler.last_lr !== 0.0015) {
    throw new Error(`${label} expected StepLR last_lr property to track scheduler step`);
  }
  const schedulerState = stepScheduler.state_dict();
  if (
    !Object.isFrozen(schedulerState) ||
    schedulerState.kind !== "step-lr" ||
    schedulerState.step !== 2 ||
    schedulerState.baseLr !== 0.003 ||
    schedulerState.base_lr !== 0.003 ||
    schedulerState.lastLr !== 0.0015 ||
    schedulerState.last_lr !== 0.0015 ||
    schedulerState.gamma !== 0.5 ||
    schedulerState.stepSize !== 2 ||
    schedulerState.step_size !== 2 ||
    schedulerState.optimizerKind !== "adam"
  ) {
    throw new Error(`${label} expected frozen StepLR state evidence`);
  }
  if (
    !adapter.optim.isLRSchedulerStateSnapshot(schedulerState) ||
    adapter.optim.requireLRSchedulerStateSnapshot(schedulerState).step !== 2 ||
    adapter.optim.assertLRSchedulerStateSnapshot(schedulerState).optimizerKind !== "adam" ||
    adapter.optim.assert_lr_scheduler_state_snapshot(schedulerState).signature !== schedulerState.signature ||
    adapter.optim.lrSchedulerStateSnapshotSignature(schedulerState) !== schedulerState.signature ||
    !adapter.optim.matchesLRSchedulerStateSnapshotSignature(schedulerState, schedulerState.signature) ||
    adapter.optim.matches_lr_scheduler_state_snapshot_signature(schedulerState, "wrong") ||
    adapter.optim.isLRSchedulerStateSnapshot({ ...schedulerState })
  ) {
    throw new Error(`${label} expected LR scheduler state snapshot validators to accept only frozen signed evidence`);
  }
  const restoredScheduler = adapter.optim.step_lr(adam, { step_size: 4, gamma: 0.25 });
  const classScheduler = new adapter.optim.StepLR(adam, { stepSize: 2, gamma: 0.5 });
  if (classScheduler.config().kind !== "step-lr" || classScheduler.getLastLr() !== adapter.optim.config(adam).lr) {
    throw new Error(`${label} expected optim.StepLR constructor to expose scheduler state evidence`);
  }
  if (adapter.optim.lrScheduler !== adapter.optim.lr_scheduler) {
    throw new Error(`${label} expected optim lrScheduler and lr_scheduler namespaces to share one surface`);
  }
  const nestedClassScheduler = new adapter.optim.lr_scheduler.StepLR(adam, { step_size: 2, gamma: 0.5 });
  const nestedFactoryScheduler = adapter.optim.lrScheduler.stepLR(adam, { stepSize: 2, gamma: 0.5 });
  if (nestedClassScheduler.config().kind !== "step-lr" || nestedFactoryScheduler.config().kind !== "step-lr") {
    throw new Error(`${label} expected nested lr_scheduler StepLR helpers`);
  }
  const snakeSchedulerState = {
    kind: schedulerState.kind,
    step: schedulerState.step,
    baseLr: schedulerState.baseLr,
    lastLr: schedulerState.lastLr,
    gamma: schedulerState.gamma,
    step_size: schedulerState.step_size,
  };
  restoredScheduler.load_state_dict(snakeSchedulerState, { strict: true });
  if (restoredScheduler.get_last_lr() !== 0.0015 || adapter.optim.config(adam).lr !== 0.0015) {
    throw new Error(`${label} expected StepLR load_state_dict alias to restore lr`);
  }
  const exponential = adapter.optim.exponentialLR(adam, { gamma: 0.1 });
  const beforeExponential = adapter.optim.config(adam).lr;
  if (Math.abs(exponential.step() - beforeExponential * 0.1) > 1e-12) {
    throw new Error(`${label} expected exponentialLR to decay every step`);
  }
  const exponentialAlias = adapter.optim.exponential_lr(adam, { gamma: 0.5 });
  if (exponentialAlias.config().kind !== "exponential-lr") {
    throw new Error(`${label} expected exponential_lr alias to return scheduler evidence`);
  }
  const exponentialClass = new adapter.optim.ExponentialLR(adam, { gamma: 0.5 });
  const beforeExponentialClass = adapter.optim.config(adam).lr;
  if (exponentialClass.config().kind !== "exponential-lr" || Math.abs(exponentialClass.step() - beforeExponentialClass * 0.5) > 1e-12) {
    throw new Error(`${label} expected optim.ExponentialLR constructor to decay optimizer lr`);
  }
  if (
    adapter.optim.lr_scheduler.exponential_lr(adam, { gamma: 0.5 }).config().kind !== "exponential-lr" ||
    new adapter.optim.lrScheduler.ExponentialLR(adam, { gamma: 0.5 }).config().kind !== "exponential-lr"
  ) {
    throw new Error(`${label} expected nested lr_scheduler ExponentialLR helpers`);
  }
  adapter.optim.set_lr(adam, 0.003);
  const cosine = adapter.optim.cosineAnnealingLR(adam, { tMax: 4, etaMin: 0 });
  const cosineFirst = cosine.step();
  const cosineExpected = 0.003 * (1 + Math.cos(Math.PI / 4)) / 2;
  if (
    cosine.config().kind !== "cosine-annealing-lr" ||
    cosine.t_max !== 4 ||
    cosine.eta_min !== 0 ||
    Math.abs(cosineFirst - cosineExpected) > 1e-12 ||
    Math.abs(adapter.optim.config(adam).lr - cosineExpected) > 1e-12
  ) {
    throw new Error(`${label} expected cosineAnnealingLR to update optimizer lr with typed state aliases`);
  }
  const cosineState = cosine.stateDict();
  if (
    !adapter.optim.isLRSchedulerStateSnapshot(cosineState) ||
    cosineState.kind !== "cosine-annealing-lr" ||
    cosineState.tMax !== 4 ||
    cosineState.t_max !== 4 ||
    cosineState.etaMin !== 0 ||
    cosineState.eta_min !== 0
  ) {
    throw new Error(`${label} expected cosineAnnealingLR state evidence`);
  }
  const cosineAlias = adapter.optim.cosine_annealing_lr(adam, { t_max: 4, eta_min: 0 });
  const cosineClass = new adapter.optim.lr_scheduler.CosineAnnealingLR(adam, { tMax: 4, etaMin: 0 });
  if (cosineAlias.config().kind !== "cosine-annealing-lr" || cosineClass.config().kind !== "cosine-annealing-lr") {
    throw new Error(`${label} expected cosine scheduler aliases and nested constructor`);
  }
  const cosineCheckpointInspection = adapter.checkpoint.inspect(adapter.checkpoint.create({ scheduler: cosine }));
  if (
    cosineCheckpointInspection.schedulerKind !== "cosine-annealing-lr" ||
    cosineCheckpointInspection.schedulerTMax !== 4 ||
    cosineCheckpointInspection.schedulerT_max !== 4 ||
    cosineCheckpointInspection.schedulerEtaMin !== 0 ||
    cosineCheckpointInspection.schedulerEta_min !== 0
  ) {
    throw new Error(`${label} expected checkpoint inspection to expose cosine scheduler state`);
  }
  adapter.optim.set_lr(adam, 0.01);
  const plateau = adapter.optim.reduceLROnPlateau(adam, { mode: "min", factor: 0.5, patience: 0, minLr: 0, thresholdMode: "abs", threshold: 0 });
  if (plateau.step(1) !== 0.01 || adapter.optim.config(adam).lr !== 0.01) {
    throw new Error(`${label} expected reduceLROnPlateau first metric to establish best without reducing lr`);
  }
  if (Math.abs(plateau.step(1.1) - 0.005) > 1e-12 || Math.abs(adapter.optim.config(adam).lr - 0.005) > 1e-12) {
    throw new Error(`${label} expected reduceLROnPlateau to reduce lr after plateau`);
  }
  const plateauState = plateau.state_dict();
  if (
    !adapter.optim.isLRSchedulerStateSnapshot(plateauState) ||
    plateauState.kind !== "reduce-lr-on-plateau" ||
    plateauState.mode !== "min" ||
    plateauState.factor !== 0.5 ||
    plateauState.patience !== 0 ||
    plateauState.thresholdMode !== "abs" ||
    plateauState.threshold_mode !== "abs" ||
    plateauState.best !== 1 ||
    plateauState.badEpochs !== 0 ||
    plateauState.bad_epochs !== 0
  ) {
    throw new Error(`${label} expected reduceLROnPlateau state evidence`);
  }
  const plateauAlias = adapter.optim.reduce_lr_on_plateau(adam, { mode: "max", factor: 0.5 });
  const plateauClass = new adapter.optim.lr_scheduler.ReduceLROnPlateau(adam, { mode: "min" });
  if (plateauAlias.config().kind !== "reduce-lr-on-plateau" || plateauClass.config().kind !== "reduce-lr-on-plateau") {
    throw new Error(`${label} expected reduceLROnPlateau aliases and nested constructor`);
  }
  const plateauCheckpointInspection = adapter.checkpoint.inspect(adapter.checkpoint.create({ scheduler: plateau }));
  if (
    plateauCheckpointInspection.schedulerKind !== "reduce-lr-on-plateau" ||
    plateauCheckpointInspection.schedulerLastLr !== 0.005
  ) {
    throw new Error(`${label} expected checkpoint inspection to expose plateau scheduler state`);
  }
  const after = adapter.loss.mse(model.forward(input), target);
  if (!(after instanceof adapter.Tensor)) throw new Error(`${label} expected tensor MSE loss after training`);
  if (!(after.item() < before.item())) throw new Error(`${label} training loss expected ${after.item()} < ${before.item()}`);
  const state = model.stateDict();
  const snakeState = model.state_dict();
  if (
    !Object.isFrozen(state) ||
    Object.keys(state).join("|") !== "weight|bias" ||
    state.weight.layout !== "row-major:linear.weight[in_features,out_features]" ||
    state.bias.layout !== "row-major:linear.bias[out_features]"
  ) {
    throw new Error(`${label} expected frozen named stateDict evidence`);
  }
  if (Object.keys(snakeState).join("|") !== "weight|bias" || snakeState.weight.data[0] !== state.weight.data[0]) {
    throw new Error(`${label} expected module state_dict alias`);
  }
  const restoredViaSnake = adapter.nn.linear(2, 1);
  if (adapter.nn.load_state_dict(restoredViaSnake, state, { strict: true }) !== restoredViaSnake) {
    throw new Error(`${label} expected namespace load_state_dict alias to return target`);
  }
  expectClose(restoredViaSnake.weight, Array.from(state.weight.data), `${label} namespace load_state_dict alias restores weight`);
  restoredViaSnake.weight.fill(0);
  if (restoredViaSnake.load_state_dict(snakeState, { strict: true }) !== restoredViaSnake) {
    throw new Error(`${label} expected module load_state_dict alias to return target`);
  }
  expectClose(restoredViaSnake.weight, Array.from(snakeState.weight.data), `${label} module load_state_dict alias restores weight`);
  expectStateAndCheckpointRoundTrip(adapter, label);
  expectSequentialProgramEvidence(adapter, label);
  expectNormGeluSequentialProgramEvidence(adapter, label);
  expectZeroParameterProgramEvidence(adapter, label);
  expectReductionProgramEvidence(adapter, label);
  expectDiagonalModuleEvidence(adapter, label);
  expectShapeMovementProgramEvidence(adapter, label);
  expectUnsupportedCompileExplanationEvidence(adapter, label);
  expectClassifierAndTokenHeadProgramEvidence(adapter, label);
  expectTorchNamespaceEndToEndEvidence(adapter, label);
  const compiledLinear = adapter.nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
  const compiledProgram = compiledLinear.compile({ backend: "cpu" });
  expectProgramCapabilityEvidence(compiledProgram, label);
  expectProgramExecutionPlanEvidence(compiledProgram, compiledLinear, label);
  expectProgramModuleCompatibilityEvidence(compiledProgram, compiledLinear, adapter, label);
  expectProgramProfileEvidence(compiledProgram, label);
  expectProgramBufferEvidence(compiledProgram, label);
  expectTensorNativeBufferEvidence(compiledProgram, adapter, label);
  const compiledSession = compiledProgram.bindModule(compiledLinear);
  const compiledSessionViaBind = compiledProgram.bind(compiledLinear);
  expectClose(
    compiledSessionViaBind.step(new Float32Array([1, 2])),
    [-0.5],
    `${label} Program.bind(module) output`,
  );
  expectInspectionEvidence(compiledProgram, compiledSession, label);
  expectSessionProfileEvidence(compiledSession, label);
  expectSessionExecutionEvidence(compiledProgram, compiledSession, adapter, label);
  expectSessionParameterUploadEvidence(compiledProgram, adapter, label);
  expectNativeBufferBoundSessionEvidence(compiledProgram, adapter, label);
  const hotParams = { input: adapter.tensor([1, 2], [2]), output: false };
  expectStepParamsEvidence(compiledSession, hotParams, adapter, label);
  try {
    compiledSession.requireHotStepParams(new Float32Array([1, 2]));
    throw new Error(`${label} expected raw StepParams to reject`);
  } catch (error) {
    if (!String((error as Error).message).includes("StepParams are not hot-path compatible")) throw error;
  }
  console.log(`${label} ok: ${nativeApiContractSignature()}`);
}
