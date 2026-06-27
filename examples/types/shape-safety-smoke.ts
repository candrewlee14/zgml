import {
  Tensor,
  cat,
  compile,
  data,
  matmul,
  nn,
  stack,
  tensor,
  type BroadcastShape,
  type DataLoader,
  type LinearForwardShape,
  type MatmulShape,
  type ModuleCompileExplanation,
  type ModuleCompileSupport,
  type ModuleTargetForwardShape,
  type Program,
  type ReshapeShape,
  type SequentialForwardShape,
  type Session,
  type TensorCatShape,
  type TensorDataset,
  type TensorDatasetBatch,
  type TensorDatasetBatchShape,
  type TensorShapeTail,
  type TensorStackShape,
} from "zgml";
import {
  compile as bunCompile,
  data as bunData,
  nn as bunNn,
  tensor as bunTensor,
  type DataLoader as BunDataLoader,
  type Program as BunProgram,
  type Session as BunSession,
  type Tensor as BunTensor,
} from "zgml/bun";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends
  (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;

const features = tensor([1, 2, 3, 4, 5, 6], [2, 3] as const);
const bias = tensor([10, 20, 30], [1, 3] as const);
const broadcasted: Tensor<readonly [2, 3]> = features.add(bias);
const reshaped: Tensor<readonly [3, 2]> = features.reshape([3, -1] as const);
const weights = tensor([1, 2, 3, 4, 5, 6], [3, 2] as const);
const projected: Tensor<readonly [2, 2]> = matmul(features, weights);
const concatenated: Tensor<readonly [4, 3]> = cat([features, features] as const, 0);
const stacked: Tensor<readonly [2, 2, 3]> = stack([features, features] as const);

type BroadcastedShape = Expect<Equal<BroadcastShape<readonly [2, 3], readonly [1, 3]>, readonly [2, 3]>>;
type ReshapedShape = Expect<Equal<ReshapeShape<readonly [2, 3], readonly [3, -1]>, readonly [3, 2]>>;
type ProjectedShape = Expect<Equal<MatmulShape<readonly [2, 3], readonly [3, 2]>, readonly [2, 2]>>;
type ConcatenatedShape = Expect<Equal<TensorCatShape<readonly [typeof features, typeof features], 0>, readonly [4, 3]>>;
type StackedShape = Expect<Equal<TensorStackShape<readonly [typeof features, typeof features]>, readonly [2, 2, 3]>>;

const dataset: TensorDataset<
  Tensor<readonly [3]>,
  Tensor<readonly [1]>,
  Tensor<readonly [number, 3]>,
  Tensor<readonly [number, 1]>
> = data.tensorDataset(features, tensor([0, 1], [2, 1] as const));
const loader: DataLoader<Tensor<readonly [number, 3]>, Tensor<readonly [number, 1]>> = data.dataLoader(dataset, { batchSize: 2 });
const firstBatch: TensorDatasetBatch<Tensor<readonly [number, 3]>, Tensor<readonly [number, 1]>> = dataset.batch([0, 1]);
type DatasetSampleShape = Expect<Equal<TensorShapeTail<readonly [2, 3]>, readonly [3]>>;
type DatasetBatchShape = Expect<Equal<TensorDatasetBatchShape<readonly [2, 3]>, readonly [number, 3]>>;

const model = nn.sequential([
  nn.linear(3, 4),
  nn.gelu(),
  nn.linear(4, 2),
] as const);
const modelForward = model.forward as unknown as {
  (input: Tensor<readonly [3]>): Tensor<readonly [2]>;
  (input: Tensor<readonly [2, 3]>): Tensor<readonly [2, 2]>;
};
const prediction: Tensor<readonly [2]> = modelForward(tensor([1, 2, 3], [3] as const));
const batchPrediction: Tensor<readonly [2, 2]> = modelForward(features);
const support: ModuleCompileSupport<readonly [3], readonly [2]> = model.compileSupport({ backend: "cpu", inputShape: [3] as const });
const plan: ModuleCompileExplanation<readonly [3], readonly [2]> = model.compilePlan({ backend: "cpu", inputShape: [3] as const });
const program: Program<readonly [3], readonly [2]> = compile.compile(model, { backend: "cpu", inputShape: [3] as const });
const session: Session<readonly [3], readonly [2]> = program.bindModule(model);
const compiledPrediction: Tensor<readonly [2]> = session.stepTensor(tensor([1, 2, 3], [3] as const));

type LinearVectorShape = Expect<Equal<LinearForwardShape<readonly [3], 4>, readonly [4]>>;
type SequentialVectorShape = Expect<Equal<SequentialForwardShape<typeof model.layers, readonly [3]>, readonly [2]>>;
type SequentialBatchShape = Expect<Equal<SequentialForwardShape<typeof model.layers, readonly [2, 3]>, readonly [2, 2]>>;
type ModuleTargetShape = Expect<Equal<ModuleTargetForwardShape<typeof model, readonly [3]>, readonly [2]>>;

const incompatibleWeights = tensor([1, 2, 3, 4, 5, 6, 7, 8], [4, 2] as const);
// @ts-expect-error incompatible matmul shapes must not narrow to the happy-path result.
const invalidProjected: Tensor<readonly [2, 2]> = matmul(features, incompatibleWeights);
// @ts-expect-error LinearModule only preserves exact output shape for feature-compatible inputs.
const invalidModelPrediction: Tensor<readonly [2]> = model.forward(tensor([1, 2], [2] as const));
// @ts-expect-error compiled Session input shape is part of the Program contract.
const invalidCompiledPrediction: Tensor<readonly [2]> = session.stepTensor(tensor([1, 2], [2] as const));

const bunInput = bunTensor([1, 2, 3], [3] as const);
const bunModel = bunNn.sequential([
  bunNn.linear(3, 2),
] as const);
const bunDataset = bunData.tensorDataset(bunTensor([1, 2, 3, 4, 5, 6], [2, 3] as const));
const bunLoader: BunDataLoader<BunTensor<readonly [number, 3]>> = bunData.dataLoader(bunDataset, { batchSize: 2 });
const bunPrediction: BunTensor<readonly [2]> = bunModel.forward(bunInput);
const bunProgram: BunProgram<readonly [3], readonly [2]> = bunCompile.compile(bunModel, { backend: "cpu", inputShape: [3] as const });
const bunSession: BunSession<readonly [3], readonly [2]> = bunProgram.bindModule(bunModel);
const bunCompiledPrediction: BunTensor<readonly [2]> = bunSession.stepTensor(bunInput);
// @ts-expect-error Bun Program/Session types reject the same wrong input shape.
const invalidBunCompiledPrediction: BunTensor<readonly [2]> = bunSession.stepTensor(bunTensor([1, 2], [2] as const));

void broadcasted;
void reshaped;
void projected;
void concatenated;
void stacked;
void loader;
void firstBatch;
void prediction;
void batchPrediction;
void support;
void plan;
void compiledPrediction;
void invalidProjected;
void invalidModelPrediction;
void invalidCompiledPrediction;
void bunLoader;
void bunPrediction;
void bunCompiledPrediction;
void invalidBunCompiledPrediction;
