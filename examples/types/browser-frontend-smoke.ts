import {
  Tensor,
  matmul,
  browserFrontendRuntimeManifest,
  browserManifest,
  data,
  frontendManifest,
  gradMode,
  loss,
  nn,
  optim,
  reshape,
  tensor,
  train,
  zeros,
  type BrowserFitEvidence,
  type BrowserLinearModule,
  type BrowserTensor,
  type DataLoader,
  type TensorDataset,
  type TrainEvaluateEvidence,
  type TrainFitStepEvidence,
  type TrainPredictEvidence,
} from "zgml/browser";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends
  (<T>() => T extends B ? 1 : 2) ? true : false;
type Extends<A, B> = A extends B ? true : false;
type Expect<T extends true> = T;

type BrowserFrontendSource = Expect<Equal<typeof frontendManifest.source, "ts">>;
type BrowserNativeLoader = Expect<Equal<typeof browserManifest.nativeLoader, false>>;
type BrowserAdapterRole = Expect<Equal<typeof browserManifest.adapterRole, "browser-safe-frontend">>;
type BrowserRuntimeNativeLoader = Expect<Equal<typeof browserFrontendRuntimeManifest.nativeLoader, false>>;
type BrowserRuntimeNativeProgram = Expect<Equal<typeof browserFrontendRuntimeManifest.nativeProgramAvailable, false>>;

const input = tensor([1, 2], [2] as const);
const target = tensor([0], [1] as const);
const constructed = new Tensor([1, 2], [2] as const);
const zerosMatrix = zeros([2, 3] as const);
const reshaped = reshape(zerosMatrix, [3, 2] as const);
const multiplied = matmul(reshaped, tensor([1, 2], [2, 1] as const));
const model = nn.linear(2, 1, {
  weights: [0.5, -0.5],
  bias: [0],
});
const prediction = model.forward(input);
const batchInput = tensor([1, 2, 3, 4, 5, 6], [3, 2] as const);
const batchPrediction = model.forward(batchInput);
const batchTarget = tensor([0, 1, 0], [3, 1] as const);
const dataset = data.tensorDataset(batchInput, batchTarget);
const datasetBatch = dataset.batch([0, 1]);
const loader = data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 7 });
const loaderBatch = loader.get(0);
const criterion = loss.mseLoss();
const objective = criterion.forward(prediction, target);
objective.backward();

const optimizer = optim.sgd(model.parameters(), { lr: 0.01 });
optimizer.step();
optimizer.zeroGrad();

function requireBatchTarget<T extends BrowserTensor>(targetValue: T | undefined): T {
  if (targetValue === undefined) throw new Error("expected labelled batch target");
  return targetValue;
}

const fit = train.fit(optimizer, [{ input, target }], (batch) => criterion.forward(model.forward(batch.input), batch.target), { epochs: 1 });
const loaderFit = train.fit(optimizer, loader, (batch, context) => {
  const step: number = context.step;
  const labelledTarget = requireBatchTarget(batch.target);
  void step;
  return criterion.forward(model.forward(batch.input), labelledTarget);
}, {
  epochs: 1,
  onStep: (evidence) => {
    const typedEvidence: TrainFitStepEvidence<"sgd"> = evidence;
    const afterStep: number | null = evidence.stepEvidence?.afterStep ?? null;
    void typedEvidence;
    void afterStep;
  },
});
const evaluated = train.evaluate(loader, (batch) => criterion.forward(model.forward(batch.input), requireBatchTarget(batch.target)));
const predicted = train.predict(loader, (batch) => model.forward(batch.input));
const noGradPrediction = gradMode.noGrad(() => model.forward(input));
const inferencePrediction = gradMode.inferenceMode(() => model.forward(input));
const compileSupport = model.compileSupport({ backend: "webgpu", inputShape: [2] as const });

type InputShape = Expect<Equal<typeof input, BrowserTensor<readonly [2]>>>;
type ConstructedShape = Expect<Equal<typeof constructed, BrowserTensor<readonly [2]>>>;
type ZerosMatrixShape = Expect<Equal<typeof zerosMatrix, BrowserTensor<readonly [2, 3]>>>;
type ReshapedShape = Expect<Equal<typeof reshaped, BrowserTensor<readonly [3, 2]>>>;
type MatmulShape = Expect<Equal<typeof multiplied, BrowserTensor<readonly [3, 1]>>>;
type ModelShape = Expect<Equal<typeof model, BrowserLinearModule<2, 1>>>;
type PredictionShape = Expect<Equal<typeof prediction, BrowserTensor<readonly [1]>>>;
type BatchPredictionShape = Expect<Equal<typeof batchPrediction, BrowserTensor<readonly [3, 1]>>>;
type DatasetShape = Expect<Extends<typeof dataset, TensorDataset<BrowserTensor<readonly [2]>, BrowserTensor<readonly [1]>, BrowserTensor<readonly [number, 2]>, BrowserTensor<readonly [number, 1]>>>>;
type DatasetBatchShape = Expect<Extends<typeof datasetBatch.input, BrowserTensor<readonly [number, 2]>>>;
type DataLoaderShape = Expect<Extends<typeof loader, DataLoader<BrowserTensor<readonly [number, 2]>, BrowserTensor<readonly [number, 1]>>>>;
type LoaderBatchShape = Expect<Extends<NonNullable<typeof loaderBatch.target>, BrowserTensor<readonly [number, 1]>>>;
type ObjectiveShape = Expect<Equal<typeof objective, BrowserTensor<readonly [1]>>>;
type FitShape = Expect<Equal<typeof fit, BrowserFitEvidence<"sgd">>>;
type LoaderFitShape = Expect<Equal<typeof loaderFit, BrowserFitEvidence<"sgd">>>;
type EvaluateShape = Expect<Equal<typeof evaluated, TrainEvaluateEvidence>>;
type PredictShape = Expect<Equal<typeof predicted, TrainPredictEvidence<BrowserTensor<readonly [number, 1]>>>>;
type BrowserCompileSupportRejected = Expect<Equal<typeof compileSupport.supported, false>>;
type BrowserCompileReturn = Expect<Equal<ReturnType<typeof model.compile>, never>>;

void input;
void target;
void constructed;
void zerosMatrix;
void reshaped;
void multiplied;
void prediction;
void batchInput;
void batchPrediction;
void batchTarget;
void dataset;
void datasetBatch;
void loader;
void loaderBatch;
void objective;
void fit;
void loaderFit;
void evaluated;
void predicted;
void noGradPrediction;
void inferencePrediction;
void compileSupport;
