import {
  checkpoint,
  clone,
  compile,
  data,
  detach,
  enable_grad,
  gradMode,
  inference_mode,
  is_grad_enabled,
  loss,
  load,
  nn,
  no_grad,
  optim,
  save,
  set_grad_enabled,
  tensor,
  torch,
  train,
  type BatchSampler,
  type CollatedDataLoader,
  type DataCollateFn,
  type DataLoader,
  type DefaultCollateOptions,
  type Dataset,
  type CheckpointInspection,
  type BCELoss,
  type BCEWithLogitsLoss,
  type Conv2dModule,
  type CrossEntropyLoss,
  type HuberLoss,
  type GradModeNamespace,
  type L1Loss,
  type LRScheduler,
  type LRSchedulerStateSnapshot,
  type LossReduction,
  type LinearModule,
  type MSELoss,
  type NLLLoss,
  type SmoothL1Loss,
  type ModuleCompileExplanation,
  type ModuleCompileSupport,
  type ModuleDict,
  type ModuleList,
  type ModuleParameterInfo,
  type ModuleTraversalEntry,
  type ModuleStateSnapshot,
  type NnBuffer,
  type NnFunctionalNamespace,
  type NnModule,
  type NnParameter,
  type Optimizer,
  type OptimizerConfigSnapshot,
  type OptimizerParamGroupInput,
  type OptimizerParamGroupSnapshot,
  type OptimizerStateSnapshot,
  type ParameterDict,
  type ParameterList,
  type Program,
  type RuntimeProfileExpectation,
  type RandomSampler,
  type Sampler,
  type SequentialModule,
  type SequentialSampler,
  type Session,
  type SessionExecutionPlan,
  type SessionStepParamsCompatibility,
  type Tensor,
  type TensorDataset,
  type TensorDatasetBatch,
  type TensorDatasetMapper,
  type TensorDatasetSample,
  type CompileAnalysis,
  type TrainClassificationReport,
  type TrainEvaluateEvidence,
  type TrainEvaluateStepEvidence,
  type TrainPredictEvidence,
  type TrainPredictStepEvidence,
  type TrainClassificationCriterion,
  type TrainStepEvidence,
  type TrainFitContext,
  type TrainFitEvidence,
  type TrainFitStepEvidence,
  type TrainModelFitOptions,
  type TrainSupervisedCriterion,
  type ZgmlCheckpoint,
} from "zgml";

const model = nn.sequential([
  nn.linear(2, 4),
  nn.relu(),
  nn.linear(4, 1),
]);
const torchModel = new torch.nn.Sequential(
  new torch.nn.Linear(2, 4),
  new torch.nn.ReLU(),
  new torch.nn.Linear(4, 1),
);
const torchSample: Tensor<readonly [2]> = torch.tensor([1, 0], [2] as const);
const torchPrediction: Tensor<readonly [1]> = torchModel.forward(torchSample);
const torchObjective: Tensor<readonly [1]> = torch.F.mse(torchPrediction, torch.tensor([1], [1] as const));
const torchOptimizer: Optimizer<"sgd"> = new torch.optim.SGD(torchModel, { lr: 0.01 });
const torchSnapshot: ZgmlCheckpoint = torch.checkpoint.create({ model: torchModel, optimizer: torchOptimizer });
const torchSavedText: string = torch.save(torchSnapshot, 2);
const torchLoadedSnapshot: ZgmlCheckpoint = torch.load(torchSavedText);
const torchRestoreTargets = torch.load(torchSavedText, { model: torchModel, optimizer: torchOptimizer, strict: true });
const torchSavedPath: string = torch.save(torchSnapshot, "/tmp/zgml-torch-training-smoke.zgml", 2);
const torchLoadedPathSnapshot: ZgmlCheckpoint = torch.load(torchSavedPath);
const torchPathRestoreTargets = torch.load(torchSavedPath, { model: torchModel, optimizer: torchOptimizer, strict: true });
const rootSavedText: string = save(torchSnapshot, 2);
const rootLoadedSnapshot: ZgmlCheckpoint = load(rootSavedText);
const rootRestoreTargets = load(rootSavedText, { model: torchModel, optimizer: torchOptimizer, strict: true });
const rootSavedPath: string = save(torchSnapshot, "/tmp/zgml-root-training-smoke.zgml", 2);
const rootLoadedPathSnapshot: ZgmlCheckpoint = load(rootSavedPath);
const rootPathRestoreTargets = load(rootSavedPath, { model: torchModel, optimizer: torchOptimizer, strict: true });
const torchCompiledDirect: Program<readonly [2], readonly [1]> = torch.compile(torchModel, { inputShape: [2] as const });
const rootCompiledDirect: Program<readonly [2], readonly [1]> = compile(torchModel, { inputShape: [2] as const });
const torchFunctional: NnFunctionalNamespace = torch.functional;
const torchFunctionalObjective: Tensor<readonly [1]> = torch.functional.mse(torchPrediction, torch.tensor([1], [1] as const));
const torchNoGradPrediction: Tensor<readonly [1]> = torch.no_grad(() => torchModel.forward(torchSample));
const torchAsTensor: Tensor<readonly [2]> = torch.as_tensor(new Float32Array([1, 2]), [2] as const);
const torchAsTensorCamel: Tensor<readonly [2]> = torch.asTensor([1, 2], [2] as const);
const torchAsArray: Tensor<readonly [2]> = torch.asarray([1, 2], [2] as const);
const torchFromNumpy: Tensor<readonly [2]> = torch.from_numpy(new Float32Array([1, 2]), [2] as const);
const torchFromNumpyCamel: Tensor<readonly [2]> = torch.fromNumpy([1, 2], [2] as const);
const torchRandn: Tensor<readonly [2, 2]> = torch.randn([2, 2] as const);
const torchScalar: Tensor<readonly [1]> = torch.scalar(2);
const torchInitWeight: NnParameter<readonly [2, 2]> = torch.nn.Parameter("torchInitWeight", torch.zeros([2, 2] as const));
const torchInitBias: NnParameter<readonly [2]> = torch.nn.Parameter("torchInitBias", torch.zeros([2] as const));
const torchInitializedWeight: NnParameter<readonly [2, 2]> = torch.nn.init.kaiming_uniform_(torchInitWeight, { nonlinearity: "relu" });
const torchInitializedBias: NnParameter<readonly [2]> = torch.nn.init.zeros_(torchInitBias);
const torchZerosLike: Tensor<readonly [2]> = torch.zeros_like(torchSample);
const torchZerosLikeCamel: Tensor<readonly [2]> = torch.zerosLike(torchSample);
const torchEmpty: Tensor<readonly [2]> = torch.empty([2] as const);
const torchEmptyLike: Tensor<readonly [2]> = torch.empty_like(torchSample);
const torchEmptyLikeCamel: Tensor<readonly [2]> = torch.emptyLike(torchSample);
const torchOnesLike: Tensor<readonly [2]> = torch.ones_like(torchSample);
const torchFullLike: Tensor<readonly [2]> = torch.full_like(torchSample, 3);
const torchRandLike: Tensor<readonly [2]> = torch.rand_like(torchSample);
const torchRandnLike: Tensor<readonly [2]> = torch.randn_like(torchSample);
const torchClone: Tensor<readonly [2]> = torch.clone(torchSample);
const torchDetached: Tensor<readonly [2]> = torch.detach(torchSample);
const torchRelu: Tensor<readonly [2]> = torch.relu(torchSample);
const torchSigmoid: Tensor<readonly [2]> = torch.sigmoid(torchSample);
const torchTanh: Tensor<readonly [2]> = torch.tanh(torchSample);
const torchSoftmax: Tensor = torch.softmax(torchSample, 0);
const torchSoftmaxDim: Tensor = torch.softmax_dim(torchSample, 0);
const torchSoftmaxDimCamel: Tensor = torch.softmaxDim(torchSample, 0);
const torchLogSoftmaxCamel: Tensor = torch.logSoftmax(torchSample, 0);
const torchLogSoftmax: Tensor = torch.log_softmax(torchSample, 0);
const torchLogSoftmaxDim: Tensor = torch.log_softmax_dim(torchSample, 0);
const torchLogSoftmaxDimCamel: Tensor = torch.logSoftmaxDim(torchSample, 0);
const torchToCpu: Tensor<readonly [2]> = torch.to(torchSample, "cpu");
const torchCpu: Tensor<readonly [2]> = torch.cpu(torchSample);
const torchFloat: Tensor<readonly [2]> = torch.float(torchSample);
const torchFloat32: Tensor<readonly [2]> = torch.float32(torchSample);
const torchTypeAs: Tensor<readonly [2]> = torch.typeAs(torchSample, torchSample);
const torchTypeAsSnake: Tensor<readonly [2]> = torch.type_as(torchSample, torchSample);
const torchSum: Tensor = torch.sum(torchSample, 0);
const torchProd: Tensor = torch.prod(torchSample);
const torchCumsum: Tensor = torch.cumsum(torchSample, 0);
const torchMean: Tensor = torch.mean(torchSample);
const torchAny: Tensor = torch.any(torchSample);
const torchAll: Tensor = torch.all(torchSample);
const torchLogsumexp: Tensor = torch.logsumexp(torchSample);
const torchLogSumExp: Tensor = torch.logSumExp(torchSample);
const torchVariance: Tensor = torch.variance(torchSample);
const torchVar: Tensor = torch.var(torchSample);
const torchStd: Tensor = torch.std(torchSample);
const torchNorm: Tensor = torch.norm(torchSample);
const torchNeg: Tensor<readonly [2]> = torch.neg(torchSample);
const torchNegative: Tensor<readonly [2]> = torch.negative(torchSample);
const torchExpm1: Tensor<readonly [2]> = torch.expm1(torchSample);
const torchLog1p: Tensor<readonly [2]> = torch.log1p(torch.ones_like(torchSample));
const torchSqr: Tensor<readonly [2]> = torch.sqr(torchSample);
const torchSquare: Tensor<readonly [2]> = torch.square(torchSample);
const torchRecip: Tensor<readonly [2]> = torch.recip(torch.ones_like(torchSample));
const torchReciprocal: Tensor<readonly [2]> = torch.reciprocal(torch.ones_like(torchSample));
const torchSgn: Tensor<readonly [2]> = torch.sgn(torchSample);
const torchSign: Tensor<readonly [2]> = torch.sign(torchSample);
const torchStep: Tensor<readonly [2]> = torch.step(torchSample);
const torchIsnan: Tensor<readonly [2]> = torch.isnan(torchSample);
const torchIsinf: Tensor<readonly [2]> = torch.isinf(torchSample);
const torchIsfinite: Tensor<readonly [2]> = torch.isfinite(torchSample);
const torchFloor: Tensor<readonly [2]> = torch.floor(torchSample);
const torchCeil: Tensor<readonly [2]> = torch.ceil(torchSample);
const torchRound: Tensor<readonly [2]> = torch.round(torchSample);
const torchTrunc: Tensor<readonly [2]> = torch.trunc(torchSample);
const torchSin: Tensor<readonly [2]> = torch.sin(torchSample);
const torchCos: Tensor<readonly [2]> = torch.cos(torchSample);
const torchTan: Tensor<readonly [2]> = torch.tan(torchSample);
const torchArgmax: Tensor = torch.argmax(torchSample);
const torchSqrt: Tensor<readonly [2]> = torch.sqrt(torch.ones_like(torchSample));
const torchRsqrt: Tensor<readonly [2]> = torch.rsqrt(torch.ones_like(torchSample));
const torchPow: Tensor<readonly [2]> = torch.pow(torchSample, 2);
const torchClamped: Tensor<readonly [2]> = torch.clamp(torchSample, 0, 1);
const torchClipped: Tensor<readonly [2]> = torch.clip(torchSample, 0, 1);
const torchFlattened: Tensor = torch.flatten(torch.tensor([1, 2, 3, 4], [2, 2] as const));
const torchReshaped: Tensor<readonly [2, 1]> = torch.reshape(torchSample, [2, 1] as const);
const torchViewed: Tensor<readonly [2, 1]> = torch.view(torchSample, [2, 1] as const);
const torchBroadcasted: Tensor<readonly [2, 2]> = torch.broadcastTo(torchReshaped, [2, 2] as const);
const torchExpanded: Tensor<readonly [2, 2]> = torch.expand(torchReshaped, [2, 2] as const);
const torchRepeated: Tensor<readonly [4]> = torch.repeat(torchSample, [2] as const);
const torchTiled: Tensor<readonly [4]> = torch.tile(torchSample, [2] as const);
const torchSqueezed: Tensor<readonly [2]> = torch.squeeze(torchViewed, 1);
const torchUnsqueezed: Tensor<readonly [1, 2]> = torch.unsqueeze(torchSample, 0);
const torchTransposed: Tensor<readonly [2, 2]> = torch.transpose(torch.tensor([1, 2, 3, 4], [2, 2] as const));
const torchPermuted: Tensor<readonly [2, 2]> = torch.permute(torch.tensor([1, 2, 3, 4], [2, 2] as const), [1, 0] as const);
const torchFlipped: Tensor<readonly [2, 2]> = torch.flip(torch.tensor([1, 2, 3, 4], [2, 2] as const), [1] as const);
const torchRolled: Tensor<readonly [2, 2]> = torch.roll(torch.tensor([1, 2, 3, 4], [2, 2] as const), 1, 0);
const torchSelected: Tensor<readonly [2]> = torch.select(torch.tensor([1, 2, 3, 4], [2, 2] as const), 0, 1);
const torchNarrowed: Tensor<readonly [2, 1]> = torch.narrow(torch.tensor([1, 2, 3, 4], [2, 2] as const), 1, 0, 1);
const torchSliced: Tensor<readonly [2, 1]> = torch.slice(torch.tensor([1, 2, 3, 4], [2, 2] as const), 1, 0, 1);
const torchIndexSelected: Tensor<readonly [2, 2]> = torch.index_select(torch.tensor([1, 2, 3, 4], [2, 2] as const), 0, torch.tensor([1, 0], [2] as const));
const torchGathered: Tensor<readonly [2, 2]> = torch.gather(torch.tensor([1, 2, 3, 4], [2, 2] as const), 1, torch.tensor([0, 1, 1, 0], [2, 2] as const));
const torchTaken: Tensor<readonly [2]> = torch.take(torch.tensor([1, 2, 3, 4], [2, 2] as const), torch.tensor([0, 3], [2] as const));
const torchUnbound: readonly Tensor[] = torch.unbind(torch.tensor([1, 2, 3, 4], [2, 2] as const), 0);
const torchAdded: Tensor<readonly [2]> = torch.add(torchSample, 1);
const torchSubbed: Tensor<readonly [2]> = torch.sub(torchSample, 1);
const torchMultiplied: Tensor<readonly [2]> = torch.mul(torchSample, 2);
const torchDivided: Tensor<readonly [2]> = torch.div(torchSample, 2);
const torchEq: Tensor<readonly [2]> = torch.eq(torchSample, torch.clone(torchSample));
const torchNe: Tensor<readonly [2]> = torch.ne(torchSample, torch.zeros_like(torchSample));
const torchLt: Tensor<readonly [2]> = torch.lt(torchSample, 2);
const torchLe: Tensor<readonly [2]> = torch.le(torchSample, 1);
const torchGt: Tensor<readonly [2]> = torch.gt(torchSample, 0);
const torchGe: Tensor<readonly [2]> = torch.ge(torchSample, 0);
const torchIsclose: Tensor<readonly [2]> = torch.isclose(torchSample, torch.clone(torchSample));
const torchMaximum: Tensor<readonly [2]> = torch.maximum(torchSample, 1);
const torchMinimum: Tensor<readonly [2]> = torch.minimum(torchSample, 1);
const torchWhere: Tensor<readonly [2]> = torch.where(torchSample, torch.ones_like(torchSample), torch.zeros_like(torchSample));
const torchMaskedFill: Tensor<readonly [2]> = torch.maskedFill(torchSample, torch.gt(torchSample, 0), 0);
const torchMaskedFillSnake: Tensor<readonly [2]> = torch.masked_fill(torchSample, torch.gt(torchSample, 0), 0);
const torchAllclose: boolean = torch.allclose(torchSample, torch.clone(torchSample));
const torchEqual: boolean = torch.equal(torchSample, torch.clone(torchSample));
const torchArgsort: Tensor<readonly [2]> = torch.argsort(torchSample);
const torchSort = torch.sort(torchSample);
const torchTopk = torch.topk(torchSample, 1);
const torchTensorSplit: readonly Tensor[] = torch.split(torchSample, 1);
const torchTensorChunk: readonly Tensor[] = torch.chunk(torchSample, 2);
const torchCat: Tensor<readonly [4]> = torch.cat([torchSample, torchSample]);
const torchConcat: Tensor<readonly [4]> = torch.concat([torchSample, torchSample]);
const torchConcatenated: Tensor<readonly [4]> = torch.concatenate([torchSample, torchSample]);
const torchVstacked: Tensor<readonly [2, 2]> = torch.vstack([torchSample, torchSample]);
const torchHstacked: Tensor<readonly [4]> = torch.hstack([torchSample, torchSample]);
const torchMatrixProduct: Tensor<readonly [2, 2]> = torch.matmul(torch.tensor([1, 2, 3, 4], [2, 2] as const), torch.tensor([1, 0, 0, 1], [2, 2] as const));
const torchMmProduct: Tensor<readonly [2, 2]> = torch.mm(torch.tensor([1, 2, 3, 4], [2, 2] as const), torch.tensor([1, 0, 0, 1], [2, 2] as const));
const torchDot: Tensor<readonly [1]> = torch.dot(torchSample, torchSample);
const torchTrace: Tensor<readonly [1]> = torch.trace(torch.tensor([1, 2, 3, 4], [2, 2] as const));
const torchDiagonal: Tensor<readonly [2]> = torch.diagonal(torch.tensor([1, 2, 3, 4], [2, 2] as const));
const torchScatterAdd: Tensor<readonly [2]> = torch.scatter_add(torch.zeros_like(torchSample), 0, torch.tensor([0, 1], [2] as const), torch.ones_like(torchSample));
const torchSeed: number = torch.manual_seed(123);
const torchInitialSeed: number | null = torch.initialSeed();
const torchInitialSeedAlias: number | null = torch.initial_seed();
const torchHasShape: boolean = torch.hasShape(torchSample, [2] as const);
const torchRequiredShape: Tensor<readonly [2]> = torch.requireShape(torchSample, [2] as const);
const torchStacked: Tensor<readonly [2, 2]> = torch.stack([torchSample, torchSample]);
const torchDataset: TensorDataset = new torch.utils.data.TensorDataset(
  torch.tensor([0, 0, 1, 1], [2, 2] as const),
  torch.tensor([0, 1], [2, 1] as const),
);
const torchLoader: DataLoader = new torch.utils.data.DataLoader(torchDataset, { batch_size: 1, shuffle: false });
const torchSplit: readonly [TensorDataset, TensorDataset] = torch.utils.data.random_split(torchDataset, [1, 1]);
const torchCompileSupport: ModuleCompileSupport = torch.nn.compileSupport(torchModel, { inputShape: [2] as const });
const torchCompileAnalysis: ModuleCompileExplanation | CompileAnalysis = torch.compile.analyze(torchModel, { inputShape: [2] as const });
const torchCompilePlan: ModuleCompileExplanation | ModuleCompileSupport = torch.compile.requireCompilePlan(torchModel, { inputShape: [2] as const });
const torchLazyInput = torch.lazy.input([2] as const);
const torchLazyGraph = torchLazyInput.linear(4).relu().linear(1);
const torchLazySupport = torchLazyGraph.compileSupport();
const torchProgramClass: typeof Program = torch.Program;
const torchSessionClass: typeof Session = torch.Session;
const torchNativeBufferClass = torch.NativeBuffer;
void torchObjective;
void torchOptimizer;
void torchSnapshot;
void torchSavedText;
void torchLoadedSnapshot;
void torchRestoreTargets;
void torchSavedPath;
void torchLoadedPathSnapshot;
void torchPathRestoreTargets;
void rootSavedText;
void rootLoadedSnapshot;
void rootRestoreTargets;
void rootSavedPath;
void rootLoadedPathSnapshot;
void rootPathRestoreTargets;
void torchCompiledDirect;
void rootCompiledDirect;
void torchFunctional;
void torchFunctionalObjective;
void torchNoGradPrediction;
void torchRandn;
void torchScalar;
void torchZerosLike;
void torchEmpty;
void torchEmptyLike;
void torchEmptyLikeCamel;
void torchZerosLikeCamel;
void torchOnesLike;
void torchFullLike;
void torchRandLike;
void torchRandnLike;
void torchClone;
void torchDetached;
void torchRelu;
void torchSigmoid;
void torchTanh;
void torchSoftmax;
void torchSoftmaxDim;
void torchSoftmaxDimCamel;
void torchLogSoftmaxCamel;
void torchLogSoftmax;
void torchLogSoftmaxDim;
void torchLogSoftmaxDimCamel;
void torchToCpu;
void torchCpu;
void torchFloat;
void torchFloat32;
void torchTypeAs;
void torchTypeAsSnake;
void torchSum;
void torchProd;
void torchCumsum;
void torchMean;
void torchAny;
void torchAll;
void torchLogsumexp;
void torchLogSumExp;
void torchVariance;
void torchVar;
void torchStd;
void torchNorm;
void torchNeg;
void torchNegative;
void torchExpm1;
void torchLog1p;
void torchSqr;
void torchSquare;
void torchRecip;
void torchReciprocal;
void torchSgn;
void torchSign;
void torchStep;
void torchIsnan;
void torchIsinf;
void torchIsfinite;
void torchFloor;
void torchCeil;
void torchRound;
void torchTrunc;
void torchSin;
void torchCos;
void torchTan;
void torchArgmax;
void torchSqrt;
void torchRsqrt;
void torchPow;
void torchClamped;
void torchClipped;
void torchFlattened;
void torchReshaped;
void torchViewed;
void torchBroadcasted;
void torchExpanded;
void torchRepeated;
void torchTiled;
void torchSqueezed;
void torchUnsqueezed;
void torchTransposed;
void torchPermuted;
void torchFlipped;
void torchRolled;
void torchSelected;
void torchNarrowed;
void torchSliced;
void torchIndexSelected;
void torchGathered;
void torchTaken;
void torchUnbound;
void torchAdded;
void torchSubbed;
void torchMultiplied;
void torchDivided;
void torchEq;
void torchNe;
void torchLt;
void torchLe;
void torchGt;
void torchGe;
void torchIsclose;
void torchMaximum;
void torchMinimum;
void torchWhere;
void torchMaskedFill;
void torchMaskedFillSnake;
void torchAllclose;
void torchEqual;
void torchArgsort;
void torchSort;
void torchTopk;
void torchTensorSplit;
void torchTensorChunk;
void torchConcatenated;
void torchVstacked;
void torchHstacked;
void torchMatrixProduct;
void torchMmProduct;
void torchDot;
void torchTrace;
void torchDiagonal;
void torchScatterAdd;
void torchSeed;
void torchInitialSeed;
void torchInitialSeedAlias;
void torchHasShape;
void torchRequiredShape;
void torchStacked;
void torchLoader;
void torchSplit;
void torchCompileSupport;
void torchCompileAnalysis;
void torchCompilePlan;
void torchLazySupport;
void torchProgramClass;
void torchSessionClass;
void torchNativeBufferClass;
const classStyleModel = new nn.Sequential([
  new nn.Linear(2, 4),
  new nn.ReLU(),
  new nn.Linear(4, 1),
]);
const variadicClassStyleModel = new nn.Sequential(
  new nn.Linear(2, 4),
  new nn.ReLU(),
  new nn.Linear(4, 1),
);
const emptySequential = nn.sequential();
const emptyClassSequential = new nn.Sequential();
const emptySequentialInput = tensor([1, 2], [2] as const);
const emptySequentialOutput: Tensor<readonly [2]> = emptySequential.forward(emptySequentialInput);
const emptyClassSequentialOutput: Tensor<readonly [2]> = emptyClassSequential.__call__(emptySequentialInput);
const namedSequential = nn.sequential({
  stem: nn.linear(2, 4),
  act: nn.relu(),
  head: nn.linear(4, 1),
});
const namedClassSequential = new nn.Sequential({
  stem: new nn.Linear(2, 4),
  head: new nn.Linear(4, 1),
});
const namedSequentialOutput: Tensor = namedSequential.forward(emptySequentialInput);
const namedClassSequentialState: ModuleStateSnapshot = namedClassSequential.stateDict("named");
const sequentialPoppedLayer: NnModule | undefined = new nn.Sequential(nn.relu(), nn.tanh()).pop();
const sequentialPoppedFirstLayer: NnModule | undefined = new nn.Sequential(nn.relu(), nn.tanh()).pop(0);
const sequentialCleared: SequentialModule = new nn.Sequential(nn.relu(), nn.tanh()).clear();
const classStyleOptimizer: Optimizer<"adamw"> = new optim.AdamW(classStyleModel, { lr: 0.01, weightDecay: 0.001 });
const rmspropModel = new nn.Sequential(new nn.Linear(2, 1));
const rmspropOptimizer: Optimizer<"rmsprop"> = optim.rmsprop(rmspropModel, { lr: 0.01, alpha: 0.9, momentum: 0.1, weightDecay: 0.001 });
const rmspropClassOptimizer: Optimizer<"rmsprop"> = new optim.RMSprop(rmspropModel, { lr: 0.01, eps: 1e-8 });
const rmspropConfig: OptimizerConfigSnapshot<"rmsprop"> = rmspropOptimizer.config();
const rmspropState: OptimizerStateSnapshot<"rmsprop"> = rmspropOptimizer.stateDict();
const adagradModel = new nn.Sequential(new nn.Linear(2, 1));
const adagradOptimizer: Optimizer<"adagrad"> = optim.adagrad(adagradModel, { lr: 0.01, lrDecay: 0.001, weightDecay: 0.001 });
const adagradClassOptimizer: Optimizer<"adagrad"> = new optim.Adagrad(adagradModel, { lr: 0.01, eps: 1e-10 });
const adagradConfig: OptimizerConfigSnapshot<"adagrad"> = adagradOptimizer.config();
const adagradState: OptimizerStateSnapshot<"adagrad"> = adagradOptimizer.stateDict();
const classStyleCriterion: MSELoss = new loss.MSELoss();
const classStyleSumCriterion: MSELoss = new loss.MSELoss({ reduction: "sum" });
const classStyleSumReduction: LossReduction = classStyleSumCriterion.reduction;
const nnClassStyleCriterion: MSELoss = new nn.MSELoss();
const nnClassStyleSumCriterion: MSELoss = new nn.MSELoss({ reduction: "sum" });
const nnClassStyleSumReduction: LossReduction = nnClassStyleSumCriterion.reduction;
const classStylePrediction: Tensor<readonly [1]> = classStyleModel.forward(tensor([1, 0], [2] as const));
const variadicClassStylePrediction: Tensor<readonly [1]> = variadicClassStyleModel.forward(tensor([1, 0], [2] as const));
const classStyleObjective: Tensor<readonly [1]> = classStyleCriterion.forward(classStylePrediction, tensor([1], [1] as const));
const classStyleSumObjective: Tensor<readonly [1]> = classStyleSumCriterion.forward(classStylePrediction, tensor([1], [1] as const));
const nnClassStyleObjective: Tensor<readonly [1]> = nnClassStyleCriterion.forward(classStylePrediction, tensor([1], [1] as const));
const nnClassStyleSumObjective: Tensor<readonly [1]> = nnClassStyleSumCriterion.forward(classStylePrediction, tensor([1], [1] as const));
classStyleOptimizer.zero_grad();
classStyleObjective.backward();
classStyleOptimizer.step();

const optimizer: Optimizer<"adamw"> = optim.adamW(model, { lr: 0.03, weightDecay: 0.001 });
const scheduler: LRScheduler<"step-lr", "adamw"> = optim.stepLR(optimizer, { stepSize: 2, gamma: 0.5 });
const exponentialScheduler: LRScheduler<"exponential-lr", "adamw"> = optim.exponentialLR(optimizer, { gamma: 0.95 });
const cosineScheduler: LRScheduler<"cosine-annealing-lr", "adamw"> = optim.cosineAnnealingLR(optimizer, { tMax: 4, etaMin: 0 });
const plateauScheduler: LRScheduler<"reduce-lr-on-plateau", "adamw"> = optim.reduceLROnPlateau(optimizer, { mode: "min", factor: 0.5, patience: 1 });
const schedulerNamespace: LRScheduler<"step-lr", "adamw"> = new optim.lr_scheduler.StepLR(optimizer, { step_size: 2, gamma: 0.5 });
const exponentialSchedulerNamespace: LRScheduler<"exponential-lr", "adamw"> = new optim.lrScheduler.ExponentialLR(optimizer, { gamma: 0.95 });
const cosineSchedulerNamespace: LRScheduler<"cosine-annealing-lr", "adamw"> = new optim.lr_scheduler.CosineAnnealingLR(optimizer, { t_max: 4, eta_min: 0 });
const plateauSchedulerNamespace: LRScheduler<"reduce-lr-on-plateau", "adamw"> = new optim.lr_scheduler.ReduceLROnPlateau(optimizer, { threshold_mode: "abs", min_lr: 0 });
const schedulerBaseLrAlias: number = scheduler.base_lr;
const schedulerLastLrAlias: number = scheduler.last_lr;
const schedulerStepSizeAlias: number = scheduler.step_size;
class CustomDataset extends data.Dataset<TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>, TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>> {
  readonly length = 2;
  sample(index: number): TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>> {
    return {
      kind: "zgml.data.sample",
      index,
      input: tensor(index === 0 ? [1, 0] : [0, 1], [2] as const),
      target: tensor([index], [1] as const),
    };
  }
}
const customDataset: Dataset<TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>, TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>> = new CustomDataset();
const customDatasetLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = new data.DataLoader(customDataset, { batch_size: 2 });
const sequentialSampler: SequentialSampler = new data.SequentialSampler(customDataset);
const randomSampler: RandomSampler = new data.RandomSampler(customDataset, { seed: 7 });
const replacementSampler: Sampler = new torch.utils.data.RandomSampler(customDataset, { replacement: true, num_samples: 3, seed: 7 });
const batchSampler: BatchSampler = new data.BatchSampler(sequentialSampler, 1, false);
const samplerLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = data.dataLoader(customDataset, { sampler: randomSampler, batch_size: 1 });
const batchSamplerLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = new torch.utils.data.DataLoader(customDataset, { batch_sampler: batchSampler });
const collateFn: DataCollateFn<
  TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>,
  TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>>
> = (samples, context) => ({
  kind: "zgml.data.batch",
  batchIndex: context.batchIndex,
  indices: context.indices,
  input: tensor(samples.flatMap((sample) => Array.from(sample.input.data)), [samples.length, 2] as const),
  target: tensor(samples.flatMap((sample) => Array.from(sample.target!.data)), [samples.length, 1] as const),
});
const collateLoader: DataLoader<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = data.dataLoader(customDataset, { batch_size: 2, collate_fn: collateFn });
type CustomCollateBatch = Readonly<{
  kind: "custom-collate";
  batchIndex: number;
  indices: readonly number[];
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
  firstInput: Tensor<readonly [2]>;
  sampleCount: number;
}>;
const objectCollateFn: DataCollateFn<
  TensorDatasetSample<Tensor<readonly [2]>, Tensor<readonly [1]>>,
  CustomCollateBatch
> = (samples, context) => ({
  kind: "custom-collate",
  batchIndex: context.batchIndex,
  indices: context.indices,
  sampleIndices: context.sampleIndices,
  sample_indices: context.sample_indices,
  firstInput: samples[0]!.input,
  sampleCount: samples.length,
});
const objectCollateLoader: CollatedDataLoader<CustomCollateBatch> = data.dataLoader(customDataset, { batch_size: 2, collate_fn: objectCollateFn });
const objectCollateBatch: CustomCollateBatch = objectCollateLoader.get(0);
const defaultCollateOptions: DefaultCollateOptions = { batchIndex: 5, indices: [1, 0] };
const torchDefaultCollateBatch: TensorDatasetBatch<Tensor<readonly [number, 2]>, Tensor<readonly [number, 1]>> = torch.utils.data.default_collate(
  [customDataset.sample(1), customDataset.sample(0)],
  defaultCollateOptions,
);
const dataset: TensorDataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const loader = data.dataLoader(dataset, { batchSize: 2, shuffle: true, seed: 11 });
const classDataset: TensorDataset = new data.TensorDataset(
  tensor([0, 1, 1, 0], [2, 2] as const),
  tensor([1, 0], [2, 1] as const),
);
const classLoader: DataLoader = new data.DataLoader(classDataset, { batch_size: 2, shuffle: false });
const datasetLen: number = dataset.__len__();
const datasetSampleAt: TensorDatasetBatch["input"] = loader.at(0).input;
const datasetSampleDunder: TensorDataset["sample"] = dataset.__getitem__;
const datasetIterableInput: Tensor = Array.from(dataset)[0]!.input;
const datasetDunderIterInput: Tensor = dataset.__iter__().next().value!.input;
const datasetSplitPair: readonly [TensorDataset, TensorDataset] = data.randomSplit(dataset, [3, 1], { seed: 13 });
const datasetSplitPairAlias: readonly [TensorDataset, TensorDataset] = data.random_split(dataset, [2, 2], { shuffle: false });
const datasetIterableSplit: readonly TensorDataset[] = data.randomSplit(dataset, new Set([3, 1]), { seed: 13 });
const datasetSubset: TensorDataset = data.subset(dataset, [1, 0]);
const datasetIterableSubset: TensorDataset = data.subset(dataset, new Set([1, 0]));
const datasetTake: TensorDataset = data.take(dataset, 2);
const datasetConcat: TensorDataset = data.concatDataset([datasetTake, datasetSubset]);
const datasetIterableConcat: TensorDataset = data.concatDataset(new Set([datasetTake, datasetIterableSubset]));
const datasetConcatAlias: TensorDataset = data.concat_dataset([datasetSubset, datasetTake]);
const datasetMapper: TensorDatasetMapper = (sample) => ({ input: sample.input, target: sample.target });
const datasetMapped: TensorDataset = data.mapDataset(dataset, datasetMapper);
const datasetMappedAlias: TensorDataset = data.map_dataset(dataset, (sample) => ({ input: sample.input }));
const datasetSplitLoader = data.dataLoader(datasetSplitPair[0], { batch_size: 2 });
const loaderBatchSizeAlias: number = loader.batch_size;
const loaderDropLastAlias: boolean = loader.drop_last;

const fit: TrainFitEvidence<"adamw"> = train.fit(optimizer, loader, (batch: TensorDatasetBatch, context: TrainFitContext) => {
  if (!batch.target) throw new Error("training batch requires a target tensor");
  const prediction: Tensor = model.call(batch.input);
  const objective: Tensor<readonly [1]> = loss.mse(prediction, batch.target);
  const epoch: number = context.epoch;
  void epoch;
  return objective;
}, {
  epochs: 2,
  zeroGrad: true,
  clipGradNorm: 1,
  clipGradValue: 0.5,
  earlyStopping: { patience: 2, minDelta: 0, mode: "min" },
  onStep(step: TrainFitStepEvidence<"adamw">) {
    const optimizerKind: "adamw" | null = step.stepEvidence?.optimizerKind ?? null;
    const clipped: boolean = step.stepEvidence?.clipGradNormApplied ?? false;
    const schedulerLr: number = scheduler.step();
    void optimizerKind;
    void clipped;
    void schedulerLr;
  },
});
const fitStopReason: "max-steps" | "early-stopping" | null = fit.stop_reason;
const fitBestLoss: number | null = fit.best_loss;
const moduleFitDataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2] as const),
  tensor([0, 1, 1, 0], [4, 1] as const),
);
const moduleFitLoader = data.dataLoader(moduleFitDataset, { batchSize: 2, shuffle: false });
type ModuleFitBatch = ReturnType<typeof moduleFitLoader.__getitem__>;
const moduleFitCriterion: TrainSupervisedCriterion<typeof model, ModuleFitBatch> = new nn.MSELoss();
const moduleFitEvidence: TrainFitEvidence<"adamw"> = train.fitModule(optimizer, model, moduleFitLoader, moduleFitCriterion, {
  epochs: 1,
  zeroGrad: true,
});
const moduleFitEvidenceSnake: TrainFitEvidence<"adamw"> = train.fit_module(optimizer, model, moduleFitLoader, moduleFitCriterion, {
  max_steps: 1,
  zero_grad: true,
});
const checkpointModel = nn.sequential([
  nn.linear(2, 4),
  nn.relu(),
  nn.linear(4, 1),
]);
const checkpointOptimizer: Optimizer<"adam"> = optim.adam(checkpointModel, { lr: 0.01 });
const tensorDatasetBatches = data.dataLoader(data.tensorDataset(
  tensor([0, 0, 1, 1], [2, 2] as const),
  tensor([0, 1], [2, 1] as const),
), { batchSize: 1 });
type ModelFirstFitBatch = ReturnType<typeof tensorDatasetBatches.__getitem__>;
const modelFirstCriterion: TrainSupervisedCriterion<typeof checkpointModel, ModelFirstFitBatch> = new nn.MSELoss();
const modelFirstFitOptions: TrainModelFitOptions<"adam", typeof checkpointModel, ModelFirstFitBatch> = {
  optimizer: checkpointOptimizer,
  loss: modelFirstCriterion,
  maxSteps: 1,
  zeroGrad: true,
};
const modelFirstFitEvidence: TrainFitEvidence<"adam"> = train.fit(checkpointModel, tensorDatasetBatches, modelFirstFitOptions);
const evaluation: TrainEvaluateEvidence = train.evaluate(loader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("evaluation batch requires a target tensor");
  return loss.mse(model.forward(batch.input), batch.target);
}, {
  maxSteps: 1,
  onStep(step: TrainEvaluateStepEvidence) {
    const lossValue: number = step.loss;
    const sampleIndices: readonly number[] | null = step.sampleIndices;
    void lossValue;
    void sampleIndices;
  },
});
const evaluationAlias: TrainEvaluateEvidence = train.evaluate_loss(datasetSplitLoader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("evaluation batch requires a target tensor");
  return loss.mse(model.forward(batch.input), batch.target);
}, { max_steps: 1 });
const moduleEvaluation: TrainEvaluateEvidence = train.evaluateModule(model, moduleFitLoader, moduleFitCriterion, { maxSteps: 1 });
const moduleEvaluationSnake: TrainEvaluateEvidence = train.evaluate_module(model, moduleFitLoader, moduleFitCriterion, { max_steps: 1 });
const prediction: TrainPredictEvidence<Tensor> = train.predict(loader, (batch: TensorDatasetBatch) => model.forward(batch.input), {
  maxSteps: 1,
  onStep(step: TrainPredictStepEvidence<Tensor>) {
    const sampleIndices: readonly number[] | null = step.sample_indices;
    const output: Tensor = step.output;
    const requiredStep: TrainPredictStepEvidence<Tensor> = train.requireTrainPredictStepEvidence(step);
    const stepSignatureMatches: boolean = train.matchesTrainPredictStepEvidenceSignature(step, step.signature);
    void sampleIndices;
    void output;
    void requiredStep;
    void stepSignatureMatches;
  },
});
const predictionValid: boolean = train.isTrainPredictEvidence(prediction);
const requiredPrediction: TrainPredictEvidence<Tensor> = train.requireTrainPredictEvidence(prediction);
const predictionSignatureMatches: boolean = train.matchesTrainPredictEvidenceSignature(prediction, prediction.signature);
const predictionAlias: TrainPredictEvidence<Tensor> = train.predict_batches(datasetSplitLoader, (batch: TensorDatasetBatch) => model.forward(batch.input), { max_steps: 1 });
const modulePrediction: TrainPredictEvidence<Tensor> = train.predictModule(model, moduleFitLoader, { maxSteps: 1 });
const modulePredictionSnake: TrainPredictEvidence<Tensor> = train.predict_module(model, moduleFitLoader, { max_steps: 1 });
const classifierModulePrediction: TrainPredictEvidence<Tensor> = train.predictClassifier(model, moduleFitLoader, { maxSteps: 1 });
const classifierModulePredictionSnake: TrainPredictEvidence<Tensor> = train.predict_classifier(model, moduleFitLoader, { max_steps: 1 });

const moduleState: ModuleStateSnapshot = model.stateDict("xor");
const optimizerState: OptimizerStateSnapshot<"adamw"> = optimizer.stateDict();
const schedulerState: LRSchedulerStateSnapshot<"step-lr", "adamw"> = scheduler.stateDict();
const snapshot = checkpoint.create({ model, optimizer, scheduler, prefix: "xor", metadata: { run: "xor", epochs: fit.epochs } });
const inspection: CheckpointInspection = checkpoint.inspect(snapshot);
const schedulerInspectionKind: "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau" | null = inspection.schedulerKind;
const cosineCheckpoint: ZgmlCheckpoint = checkpoint.create({ scheduler: cosineScheduler });
const cosineInspection: CheckpointInspection = checkpoint.inspect(cosineCheckpoint);
const cosineInspectionTMax: number | null = cosineInspection.schedulerTMax;
const cosineInspectionEtaMin: number | null = cosineInspection.schedulerEta_min;
const jsonSnapshot: ZgmlCheckpoint = checkpoint.toJSON(snapshot);
const loadedSnapshot: ZgmlCheckpoint = checkpoint.fromJSON(JSON.parse(JSON.stringify(jsonSnapshot)));
const serializedSnapshot: string = checkpoint.stringify(snapshot, 2);
const parsedSnapshot: ZgmlCheckpoint = checkpoint.parse(serializedSnapshot);
const serializedSnapshotAlias: string = checkpoint.serialize(snapshot, 2);
const parsedSnapshotAlias: ZgmlCheckpoint = checkpoint.deserialize(serializedSnapshotAlias);
const loadedInspection: CheckpointInspection = checkpoint.inspect(loadedSnapshot);

model.loadStateDict(moduleState, { prefix: "xor", strict: true, validateOnly: true });
optimizer.loadStateDict(optimizerState, { strict: true, validateOnly: true });
scheduler.loadStateDict(schedulerState, { strict: true, validateOnly: true });
checkpoint.load(loadedSnapshot, { model, optimizer, scheduler, prefix: "xor", strict: true });

const restoredModel = nn.sequential([
  nn.linear(2, 4),
  nn.relu(),
  nn.linear(4, 1),
]);
const restoredOptimizer: Optimizer<"adamw"> = optim.adamW(restoredModel, { lr: 0.03, weightDecay: 0.001 });
const restoredScheduler: LRScheduler<"step-lr", "adamw"> = optim.stepLR(restoredOptimizer, { stepSize: 2, gamma: 0.5 });
checkpoint.restore(loadedSnapshot, {
  model: restoredModel,
  optimizer: restoredOptimizer,
  scheduler: restoredScheduler,
  prefix: "xor",
  strict: true,
});
const restoredProgram: Program<readonly [2], readonly [1]> = restoredModel.compile({ backend: "cpu", inputShape: [2] as const });
const restoredSession: Session<readonly [2], readonly [1]> = restoredProgram.bindModule(restoredModel);
const restoredCompiledPrediction: Tensor<readonly [1]> = restoredSession.stepTensor(tensor([1, 0], [2] as const));
const restoredHotProfile: RuntimeProfileExpectation = restoredSession.requireHotRuntimeProfile();
const restoredCompiledOut = new Float32Array(1);
const restoredHotParams = { input: tensor([1, 0], [2] as const), output: restoredCompiledOut };
const restoredHotCompatibility: SessionStepParamsCompatibility = restoredSession.requireHotStepParams(restoredHotParams);
const restoredHotPlan: SessionExecutionPlan<readonly [2], readonly [1]> = restoredSession.hotPathPlan(restoredHotParams);
const restoredCompiledInto: Float32Array = restoredSession.executeInto(restoredCompiledOut, { input: tensor([1, 0], [2] as const) });

const compileSupport: ModuleCompileSupport = model.compileSupport({ inputShape: [2, 2] as const });
const canCompile: boolean = compileSupport.supported;
const sample = tensor([1, 0], [2] as const);
const regularized = nn.sequential([
  nn.linear(2, 2),
  nn.batchNorm1d(2),
  nn.dropout(0.25),
  nn.relu(),
]);
const classStyleBatchNorm: ReturnType<typeof nn.batch_norm1d> = new nn.BatchNorm1d(2, {
  momentum: 0.2,
  runningMean: [0, 0],
  runningVar: [1, 1],
});
const batchNormFactory: ReturnType<typeof nn.batchNorm1d> = nn.batchNorm1d(2, { track_running_stats: true });
const batchNormSnakeFactory: ReturnType<typeof nn.batch_norm1d> = nn.batch_norm1d(2, { trackRunningStats: true });
const batchNormForward: Tensor<readonly [2, 2]> = classStyleBatchNorm.forward(tensor([1, 3, 3, 7], [2, 2] as const));
const batchNormEval: typeof classStyleBatchNorm = classStyleBatchNorm.eval();
const batchNormEvalForward: Tensor<readonly [2, 2]> = batchNormEval.forward(tensor([1, 3, 3, 7], [2, 2] as const));
const batchNormRunningMean: NnBuffer | null = classStyleBatchNorm.runningMean;
const batchNormRunningVar: NnBuffer | null = classStyleBatchNorm.runningVar;
const batchNormState: ModuleStateSnapshot = classStyleBatchNorm.stateDict("bn");
const conv2d: Conv2dModule<1, 1> = new nn.Conv2d(1, 1, 2, { weight: [1, 0, 0, 1], bias: false });
const conv2dFactory: ReturnType<typeof nn.conv2d> = nn.conv2d(1, 1, [2, 2] as const, { stride: 1, padding: 0 });
const conv2dForward: Tensor<readonly [1, number, number]> = conv2d.forward(tensor([1, 2, 3, 4], [1, 2, 2] as const));
const conv2dBatchForward: Tensor<readonly [1, 1, number, number]> = conv2d.forward(tensor([1, 2, 3, 4], [1, 1, 2, 2] as const));
const conv2dState: ModuleStateSnapshot = conv2d.stateDict("conv");
const conv2dSupport: ModuleCompileSupport = conv2d.compileSupport({ inputShape: [1, 2, 2] as const });
const regularizedParameterNames: readonly string[] = regularized.parameterNames("regularized");
const regularizedParameterInfos: readonly ModuleParameterInfo[] = regularized.parameterInfos("regularized");
const regularizedNamedModules: readonly ModuleTraversalEntry[] = regularized.namedModules("regularized");
const regularizedNamespaceNamedModules: readonly ModuleTraversalEntry[] = nn.namedModules(regularized, "regularized");
const regularizedTrain: typeof regularized = regularized.train();
const regularizedEval: typeof regularized = regularized.eval();
const regularizedNamespaceTrain: typeof regularized = nn.train(regularized, true);
const regularizedNamespaceEval: typeof regularized = nn.eval(regularized);
const regularizedCpu: typeof regularized = regularized.cpu();
const regularizedToCpu: typeof regularized = regularized.to("cpu");
const regularizedFloat: typeof regularized = regularized.float();
const regularizedFloat32: typeof regularized = regularized.float32();
const regularizedNamespaceCpu: typeof regularized = nn.cpu(regularized);
const regularizedNamespaceToCpu: typeof regularized = nn.to(regularized, { device: "cpu", copy: false });
const regularizedNamespaceFloat: typeof regularized = nn.float(regularized);
const regularizedNamespaceFloat32: typeof regularized = nn.float32(regularized, { dtype: "float32", copy: false });
regularizedEval.eval();
const regularizedEvalPrediction: Tensor<readonly [2]> = regularizedEval.forward(sample);
const regularizedEvalProgram: Program<readonly [2], readonly [2]> = regularizedEval.compile({ backend: "cpu", inputShape: [2] as const });
const regularizedEvalSession: Session<readonly [2], readonly [2]> = regularizedEvalProgram.bindModule(regularizedEval);
const regularizedEvalCompiledPrediction: Tensor<readonly [2]> = regularizedEvalSession.stepTensor(sample);

class BufferedHead extends nn.Module<readonly [1], readonly [1]> {
  readonly offset = nn.Buffer("offset", [1], [1] as const);
  declare scale: NnBuffer<readonly [1]>;

  constructor() {
    super({ kind: "typed-buffered-head" });
    this.registerBuffer("scale", nn.Buffer("scale", [2], [1] as const));
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return input.mul(tensor(this.scale.data, [1] as const)).add(tensor(this.offset.data, [1] as const));
  }
}

class ModuleListHead extends nn.Module<readonly [1], readonly [1]> {
  declare layers: ModuleList<readonly [LinearModule<1, 1>]>;

  constructor() {
    super({ kind: "typed-module-list-head" });
    this.layers = nn.moduleList([
      nn.linear(1, 1, { weights: [2], bias: [0] }),
    ]);
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return this.layers.at(0)!.forward(input);
  }
}

class ParameterListHead extends nn.Module<readonly [1], readonly [1]> {
  declare params: ParameterList<readonly [NnParameter<readonly [1]>, NnParameter<readonly [1]>]>;

  constructor() {
    super({ kind: "typed-parameter-list-head" });
    this.params = new nn.ParameterList([
      nn.Parameter("weight", [0], [1] as const),
      nn.Parameter("bias", [0], [1] as const),
    ]);
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return input.mul(this.params.at(0)!.tensor).add(this.params.at(1)!.tensor);
  }
}

class ModuleDictHead extends nn.Module<readonly [1], readonly [1]> {
  declare layers: ModuleDict<{ readonly head: LinearModule<1, 1> }>;

  constructor() {
    super({ kind: "typed-module-dict-head" });
    this.layers = nn.moduleDict({
      head: nn.linear(1, 1, { weights: [2], bias: [0] }),
    });
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return this.layers.get("head")!.forward(input);
  }
}

class ParameterDictHead extends nn.Module<readonly [1], readonly [1]> {
  declare params: ParameterDict<{
    readonly weight: NnParameter<readonly [1]>;
    readonly bias: NnParameter<readonly [1]>;
  }>;

  constructor() {
    super({ kind: "typed-parameter-dict-head" });
    this.params = new nn.ParameterDict({
      weight: nn.Parameter("weight", [0], [1] as const),
      bias: nn.Parameter("bias", [0], [1] as const),
    });
  }

  forward(input: Tensor<readonly [1]>): Tensor<readonly [1]> {
    return input.mul(this.params.get("weight")!.tensor).add(this.params.get("bias")!.tensor);
  }
}

const containerInput = tensor([3], [1] as const);
const bufferedHead = new BufferedHead();
const bufferedHeadState: ModuleStateSnapshot = bufferedHead.stateDict("buffered");
const bufferedHeadBuffers: readonly NnBuffer[] = nn.namedBuffers(bufferedHead, "buffered");
const bufferedHeadForward: Tensor<readonly [1]> = bufferedHead.forward(containerInput);
const moduleListHead = new ModuleListHead();
const moduleListNamedModules: readonly ModuleTraversalEntry[] = moduleListHead.namedModules("moduleList");
const moduleListPoppedLayer: NnModule | undefined = nn.moduleList([nn.relu(), nn.tanh()]).pop();
const moduleListPoppedFirstLayer: NnModule | undefined = nn.moduleList([nn.relu(), nn.tanh()]).pop(0);
const moduleListCleared: ModuleList = nn.moduleList([nn.relu(), nn.tanh()]).clear();
const moduleListProgram: Program<readonly [1], readonly [1]> = moduleListHead.compile({ backend: "cpu", inputShape: [1] as const });
const moduleListSession: Session<readonly [1], readonly [1]> = moduleListProgram.bindModule(moduleListHead);
const moduleListCompiled: Tensor<readonly [1]> = moduleListSession.stepTensor(containerInput);
const parameterListHead = new ParameterListHead();
const parameterListParameters: readonly NnParameter[] = parameterListHead.namedParameters("parameterList");
const parameterListState: ModuleStateSnapshot = parameterListHead.stateDict("parameterList");
const parameterListPopped: NnParameter | undefined = nn.parameterList([
  nn.Parameter("weight", [0], [1] as const),
  nn.Parameter("bias", [0], [1] as const),
]).pop();
const parameterListCleared: ParameterList = nn.parameterList([
  nn.Parameter("weight", [0], [1] as const),
  nn.Parameter("bias", [0], [1] as const),
]).clear();
const parameterListOptimizer: Optimizer<"sgd"> = optim.sgd(parameterListHead, { lr: 0.01 });
const moduleDictHead = new ModuleDictHead();
const moduleDictNamedChildren: readonly ModuleTraversalEntry[] = nn.namedChildren(moduleDictHead, "moduleDict");
const moduleDictProgram: Program<readonly [1], readonly [1]> = moduleDictHead.compile({ backend: "cpu", inputShape: [1] as const });
const moduleDictSession: Session<readonly [1], readonly [1]> = moduleDictProgram.bindModule(moduleDictHead);
const moduleDictCompiled: Tensor<readonly [1]> = moduleDictSession.stepTensor(containerInput);
const parameterDictHead = new ParameterDictHead();
const parameterDictParameters: readonly NnParameter[] = parameterDictHead.namedParameters("parameterDict");
const parameterDictState: ModuleStateSnapshot = nn.stateDict(parameterDictHead, "parameterDict");
const parameterDictOptimizer: Optimizer<"sgd"> = optim.sgd(parameterDictHead, { lr: 0.01 });
const gradModeWasEnabled: boolean = is_grad_enabled();
const previousGradMode: boolean = set_grad_enabled(true);
const rootGradModeNamespace: GradModeNamespace = gradMode;
const rootGradModeEnabled: boolean = gradMode.isGradEnabled();
const rootGradModeNoGradValue: number = gradMode.noGrad(() => 4);
const gradTrackedSample: Tensor<readonly [2]> = sample.clone().requires_grad_();
const gradTrackedSampleAlias: Tensor<readonly [2]> = gradTrackedSample.requiresGrad_(true);
const clonedSample: Tensor<readonly [2]> = clone(gradTrackedSampleAlias);
const detachedSample: Tensor<readonly [2]> = detach(gradTrackedSampleAlias);
const detachedMethodSample: Tensor<readonly [2]> = gradTrackedSampleAlias.detach();
const detachedInPlaceSample: Tensor<readonly [2]> = gradTrackedSampleAlias.detach_();
const noGradPrediction: Tensor<readonly [1]> = no_grad(() => model.forward(sample));
const inferenceModePrediction: Tensor<readonly [1]> = inference_mode(() => model.forward(sample));
const enableGradPrediction: Tensor<readonly [1]> = enable_grad(() => model.forward(sample));
set_grad_enabled(previousGradMode);
const manualOptimizer: Optimizer<"sgd"> = optim.sgd(model, { lr: 0.01, momentum: 0.1 });
const manualLoss: Tensor<readonly [1]> = loss.mse(model.forward(sample), tensor([1], [1] as const));
manualOptimizer.zeroGrad();
manualLoss.backward();
manualOptimizer.step();
manualOptimizer.zero_grad();
const manualStepEvidence: TrainStepEvidence<"sgd"> = train.step(manualOptimizer, {
  loss: loss.mse(model.forward(sample), tensor([1], [1] as const)),
  evidence: true,
  clipGradNorm: 1,
});
const manualGradNorm: number = train.gradNorm(model);
const manualStepValid: boolean = train.isTrainStepEvidence(manualStepEvidence);
const helperLoss: Tensor<readonly [1]> = loss.mse(model.forward(sample), tensor([1], [1] as const));
train.backward(helperLoss);
const helperClippedNorm: number = train.clipGradNorm(model, 1, { eps: 1e-6 });
const helperClippedModel: typeof model = train.clipGradValue(model, 0.5);
const helperLossStep: Tensor = train.lossStep(manualOptimizer, () => loss.mse(model.forward(sample), tensor([1], [1] as const)), {
  zeroGrad: true,
  zero_grad_options: { set_to_none: true },
  clipGradNorm: 1,
  clipGradValue: 0.5,
});
const helperLossStepSnake: Tensor = train.lossStep(manualOptimizer, () => loss.mse(model.forward(sample), tensor([1], [1] as const)), {
  zero_grad: true,
  zero_grad_options: { set_to_none: true },
});
const manualFrozenParam = model.namedParameters().find((param) => param.name.endsWith("bias"));
if (manualFrozenParam) manualFrozenParam.requiresGrad = false;
const groupedParams: NnParameter[] = model.namedParameters("grouped");
const groupedOptimizer: Optimizer<"sgd"> = optim.sgd([
  { params: [groupedParams[0]], lr: 0.02, weightDecay: 0.001 },
  { params: [groupedParams[1]], lr: 0.005, weight_decay: 0 },
] satisfies readonly OptimizerParamGroupInput[], { lr: 0.01, momentum: 0.2 });
const groupedConfig: OptimizerConfigSnapshot<"sgd"> = groupedOptimizer.config();
const groupedDefaults: OptimizerConfigSnapshot<"sgd"> = groupedOptimizer.defaults;
const groupedParamGroup: OptimizerParamGroupSnapshot = groupedConfig.paramGroups[0];
const groupedParamGroupsConfigAlias: readonly OptimizerParamGroupSnapshot[] = groupedConfig.param_groups;
const groupedParamGroups: readonly OptimizerParamGroupSnapshot[] = groupedOptimizer.paramGroups;
const groupedParamGroupsAlias: readonly OptimizerParamGroupSnapshot[] = groupedOptimizer.param_groups;
groupedOptimizer.addParamGroup({ params: [groupedParams[2]], lr: 0.001 });
groupedOptimizer.add_param_group({ params: [groupedParams[3]], lr: 0.0005 });
const initializedGroupedWeight: NnParameter = nn.init.xavier_uniform_(groupedParams[0]);
const normalInitializedGroupedWeight: NnParameter = nn.init.xavier_normal_(groupedParams[0]);
const kaimingInitializedGroupedWeight: NnParameter = nn.init.kaiming_uniform_(groupedParams[0], { nonlinearity: "relu" });
const kaimingNormalInitializedGroupedWeight: NnParameter = nn.init.kaiming_normal_(groupedParams[0], { nonlinearity: "relu" });
const initializedGroupedBias: NnParameter = nn.init.zeros_(groupedParams[1]);
const program: Program<readonly [2], readonly [1]> = model.compile({ backend: "cpu", inputShape: [2] as const });
const session: Session<readonly [2], readonly [1]> = program.bindModule(model);
const compiledPrediction: Tensor<readonly [1]> = session.stepTensor(sample);
const forwardPrediction: Tensor<readonly [1]> = model.forward(sample);
const callPrediction: Tensor<readonly [1]> = model.call(sample);
const dunderPrediction: Tensor<readonly [1]> = model.__call__(sample);
const namespaceCallPrediction: Tensor<readonly [1]> = nn.call(model, sample);
const namespaceDunderPrediction: Tensor<readonly [1]> = nn.__call__(model, sample);
const criterion: MSELoss = loss.mseLoss();
const criterionForward: Tensor<readonly [1]> = criterion.forward(forwardPrediction, tensor([1], [1] as const));
const criterionCall: Tensor<readonly [1]> = criterion.call(callPrediction, tensor([1], [1] as const));
const criterionDunderCall: Tensor<readonly [1]> = criterion.__call__(dunderPrediction, tensor([1], [1] as const));
const l1Criterion: L1Loss = loss.l1Loss();
const l1CriterionAlias: L1Loss = loss.l1_loss();
const l1CriterionClass: L1Loss = new loss.L1Loss();
const nnL1CriterionClass: L1Loss = new nn.L1Loss();
const l1Objective: Tensor<readonly [1]> = loss.l1(forwardPrediction, tensor([1], [1] as const));
const l1Forward: Tensor<readonly [1]> = l1Criterion.forward(forwardPrediction, tensor([1], [1] as const));
const l1Call: Tensor<readonly [1]> = l1CriterionAlias.call(callPrediction, tensor([1], [1] as const));
const l1DunderCall: Tensor<readonly [1]> = l1CriterionClass.__call__(dunderPrediction, tensor([1], [1] as const));
const nnL1Forward: Tensor<readonly [1]> = nnL1CriterionClass.forward(forwardPrediction, tensor([1], [1] as const));
const huberCriterion: HuberLoss = loss.huberLoss({ delta: 1 });
const huberCriterionAlias: HuberLoss = loss.huber_loss({ delta: 1 });
const huberCriterionClass: HuberLoss = new loss.HuberLoss({ delta: 1 });
const nnHuberCriterionClass: HuberLoss = new nn.HuberLoss({ delta: 1 });
const huberObjective: Tensor<readonly [1]> = loss.huber(forwardPrediction, tensor([1], [1] as const), { delta: 1 });
const huberForward: Tensor<readonly [1]> = huberCriterion.forward(forwardPrediction, tensor([1], [1] as const));
const huberCall: Tensor<readonly [1]> = huberCriterionAlias.call(callPrediction, tensor([1], [1] as const));
const huberDunderCall: Tensor<readonly [1]> = huberCriterionClass.__call__(dunderPrediction, tensor([1], [1] as const));
const nnHuberForward: Tensor<readonly [1]> = nnHuberCriterionClass.forward(forwardPrediction, tensor([1], [1] as const));
const smoothL1Criterion: SmoothL1Loss = loss.smoothL1Loss({ beta: 1 });
const smoothL1CriterionAlias: SmoothL1Loss = loss.smooth_l1_loss({ beta: 1 });
const smoothL1CriterionClass: SmoothL1Loss = new loss.SmoothL1Loss({ beta: 1 });
const nnSmoothL1CriterionClass: SmoothL1Loss = new nn.SmoothL1Loss({ beta: 1 });
const smoothL1Objective: Tensor<readonly [1]> = loss.smooth_l1(forwardPrediction, tensor([1], [1] as const), { beta: 1 });
const smoothL1Forward: Tensor<readonly [1]> = smoothL1Criterion.forward(forwardPrediction, tensor([1], [1] as const));
const smoothL1Call: Tensor<readonly [1]> = smoothL1CriterionAlias.call(callPrediction, tensor([1], [1] as const));
const smoothL1DunderCall: Tensor<readonly [1]> = smoothL1CriterionClass.__call__(dunderPrediction, tensor([1], [1] as const));
const nnSmoothL1Forward: Tensor<readonly [1]> = nnSmoothL1CriterionClass.forward(forwardPrediction, tensor([1], [1] as const));
const bceCriterion: BCELoss = loss.bceLoss();
const bceCriterionAlias: BCELoss = loss.bce_loss({ eps: 1e-7 });
const bceCriterionClass: BCELoss = new loss.BCELoss({ eps: 1e-7 });
const nnBceCriterionClass: BCELoss = new nn.BCELoss({ eps: 1e-7 });
const binaryPrediction: Tensor<readonly [1]> = forwardPrediction.sigmoid();
const bceObjective: Tensor<readonly [1]> = loss.bce(binaryPrediction, tensor([1], [1] as const));
const binaryCrossEntropyObjective: Tensor<readonly [1]> = loss.binary_cross_entropy(binaryPrediction, tensor([1], [1] as const));
const binaryAccuracy: number = train.binaryAccuracy(binaryPrediction, tensor([1], [1] as const));
const binaryAccuracyAlias: number = train.binary_accuracy([0.1, 0.9] as const, [0, 1] as const);
const bceForward: Tensor<readonly [1]> = bceCriterion.forward(binaryPrediction, tensor([1], [1] as const));
const bceCall: Tensor<readonly [1]> = bceCriterionAlias.call(binaryPrediction, tensor([1], [1] as const));
const bceDunderCall: Tensor<readonly [1]> = bceCriterionClass.__call__(binaryPrediction, tensor([1], [1] as const));
const nnBceForward: Tensor<readonly [1]> = nnBceCriterionClass.forward(binaryPrediction, tensor([1], [1] as const));
const bceWithLogitsCriterion: BCEWithLogitsLoss = loss.bceWithLogitsLoss();
const bceWithLogitsCriterionAlias: BCEWithLogitsLoss = loss.bce_with_logits_loss();
const bceWithLogitsCriterionClass: BCEWithLogitsLoss = new loss.BCEWithLogitsLoss();
const nnBceWithLogitsCriterionClass: BCEWithLogitsLoss = new nn.BCEWithLogitsLoss();
const bceWithLogitsObjective: Tensor<readonly [1]> = loss.bceWithLogits(forwardPrediction, tensor([1], [1] as const));
const binaryCrossEntropyWithLogitsObjective: Tensor<readonly [1]> = loss.binary_cross_entropy_with_logits(forwardPrediction, tensor([1], [1] as const));
const binaryLogitsAccuracy: number = train.binaryLogitsAccuracy(forwardPrediction, tensor([1], [1] as const));
const binaryLogitsAccuracyAlias: number = train.binary_logits_accuracy([-1, 2] as const, [0, 1] as const);
const bceWithLogitsForward: Tensor<readonly [1]> = bceWithLogitsCriterion.forward(forwardPrediction, tensor([1], [1] as const));
const bceWithLogitsCall: Tensor<readonly [1]> = bceWithLogitsCriterionAlias.call(forwardPrediction, tensor([1], [1] as const));
const bceWithLogitsDunderCall: Tensor<readonly [1]> = bceWithLogitsCriterionClass.__call__(forwardPrediction, tensor([1], [1] as const));
const nnBceWithLogitsForward: Tensor<readonly [1]> = nnBceWithLogitsCriterionClass.forward(forwardPrediction, tensor([1], [1] as const));
const classTargets: Uint32Array = loss.classTargets([1]);
const classLogits = tensor([0.2, 1.4], [1, 2] as const);
const classObjective: Tensor<readonly [1]> = loss.crossEntropy(classLogits, classTargets, { classes: 2 });
const classObjectiveSnake: Tensor<readonly [1]> = loss.cross_entropy(classLogits, classTargets, { numClasses: 2 });
function createTypedClassifierGraph() {
  return nn.sequential([
    nn.linear(2, 4),
    nn.relu(),
    nn.linear(4, 2),
  ]);
}
class TypedClassifier extends nn.Module<readonly [2], readonly [2]> {
  readonly graph: ReturnType<typeof createTypedClassifierGraph>;

  constructor() {
    super({ kind: "typed-classifier" });
    this.graph = createTypedClassifierGraph();
  }

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]>;
  forward(input: Tensor): Tensor;
  forward(input: Tensor): Tensor {
    return this.graph.forward(input);
  }
}
const typedClassifier = new TypedClassifier();
const typedClassifierTarget: Uint32Array = loss.classTargets([1]);
const typedClassifierCriterion: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 2 });
const typedClassifierObjective: Tensor<readonly [1]> = typedClassifierCriterion.forward(typedClassifier.forward(sample), typedClassifierTarget);
const typedClassifierDataset = data.tensorDataset(
  tensor([-1, -1, 1, -1], [2, 2] as const),
  tensor([0, 1], [2] as const),
);
const typedClassifierLoader = data.dataLoader(typedClassifierDataset, { batch_size: 1, shuffle: false });
const typedClassifierOptimizer: Optimizer<"adamw"> = optim.adamW(typedClassifier, { lr: 0.01 });
const typedClassifierFit: TrainFitEvidence<"adamw"> = train.fit(typedClassifierOptimizer, typedClassifierLoader, (batch: TensorDatasetBatch) => {
  if (!batch.target) throw new Error("typed classifier batch requires targets");
  return typedClassifierCriterion.forward(typedClassifier.graph.forward(batch.input), batch.target);
}, { epochs: 1, zero_grad: true });
type TypedClassifierBatch = ReturnType<typeof typedClassifierLoader.__getitem__>;
const typedClassifierFitCriterion: TrainClassificationCriterion<typeof typedClassifier, TypedClassifierBatch> = typedClassifierCriterion;
const typedClassifierFitModule: TrainFitEvidence<"adamw"> = train.fitClassifier(
  typedClassifierOptimizer,
  typedClassifier,
  typedClassifierLoader,
  typedClassifierFitCriterion,
  { epochs: 1, zero_grad: true },
);
const typedClassifierFitModuleSnake: TrainFitEvidence<"adamw"> = train.fit_classifier(
  typedClassifierOptimizer,
  typedClassifier,
  typedClassifierLoader,
  typedClassifierFitCriterion,
  { max_steps: 1, zero_grad: true },
);
const typedClassifierPrediction: TrainPredictEvidence<Tensor<readonly [number, 2]>> = train.predictClassifier(typedClassifier, typedClassifierLoader, { maxSteps: 1 });
const typedClassifierPredictionSnake: TrainPredictEvidence<Tensor<readonly [number, 2]>> = train.predict_classifier(typedClassifier, typedClassifierLoader, { max_steps: 1 });
// @ts-expect-error fitModule rejects classifiers trained with class-index targets.
train.fitModule(typedClassifierOptimizer, typedClassifier, typedClassifierLoader, moduleFitCriterion);
// @ts-expect-error fitClassifier rejects regression-style targets that still include a class axis.
train.fitClassifier(typedClassifierOptimizer, typedClassifier, moduleFitLoader, typedClassifierFitCriterion);
const typedClassifierState: ModuleStateSnapshot = typedClassifier.stateDict("typedClassifier");
const typedClassifierCheckpoint: ZgmlCheckpoint = checkpoint.create({ model: typedClassifier, optimizer: typedClassifierOptimizer, prefix: "typedClassifier" });
const typedClassifierInspection: CheckpointInspection = checkpoint.inspect(typedClassifierCheckpoint);
const typedClassifierProgram: Program<readonly [2], readonly [2]> = typedClassifier.compile({ backend: "cpu", inputShape: [2] as const });
const typedClassifierSession: Session<readonly [2], readonly [2]> = typedClassifierProgram.bindModule(typedClassifier);
const typedClassifierCompiledLogits: Tensor<readonly [2]> = typedClassifierSession.stepTensor(sample);
const F = nn.functional;
const functionalMseObjective: Tensor<readonly [1]> = F.mse(forwardPrediction, tensor([1], [1] as const));
const functionalMseLossObjective: Tensor<readonly [1]> = F.mse_loss(forwardPrediction, tensor([1], [1] as const));
const functionalMseLossSumObjective: Tensor<readonly [1]> = F.mse_loss(forwardPrediction, tensor([1], [1] as const), { reduction: "sum" });
const functionalL1Objective: Tensor<readonly [1]> = F.l1(forwardPrediction, tensor([1], [1] as const));
const functionalL1LossObjective: Tensor<readonly [1]> = F.l1_loss(forwardPrediction, tensor([1], [1] as const));
const functionalL1LossSumObjective: Tensor<readonly [1]> = F.l1_loss(forwardPrediction, tensor([1], [1] as const), { reduction: "sum" });
const functionalHuberObjective: Tensor<readonly [1]> = F.huber(forwardPrediction, tensor([1], [1] as const), { delta: 1 });
const functionalHuberLossObjective: Tensor<readonly [1]> = F.huber_loss(forwardPrediction, tensor([1], [1] as const), { delta: 1 });
const functionalSmoothL1Objective: Tensor<readonly [1]> = F.smooth_l1(forwardPrediction, tensor([1], [1] as const), { beta: 1 });
const functionalSmoothL1LossObjective: Tensor<readonly [1]> = F.smooth_l1_loss(forwardPrediction, tensor([1], [1] as const), { beta: 1 });
const functionalBceObjective: Tensor<readonly [1]> = F.binary_cross_entropy(binaryPrediction, tensor([1], [1] as const));
const functionalBceWithLogitsObjective: Tensor<readonly [1]> = F.binary_cross_entropy_with_logits(forwardPrediction, tensor([1], [1] as const));
const functionalActivation: Tensor<readonly [1, 2]> = F.relu(classLogits);
const functionalActivationInplaceFalse: Tensor<readonly [1, 2]> = F.relu(classLogits, false);
const functionalGelu: Tensor<readonly [1, 2]> = F.gelu(classLogits);
const functionalSilu: Tensor<readonly [1, 2]> = F.silu(classLogits);
const functionalSigmoid: Tensor<readonly [1, 2]> = F.sigmoid(classLogits);
const functionalTanh: Tensor<readonly [1, 2]> = F.tanh(classLogits);
const functionalSoftmax: Tensor<readonly [1, 2]> = F.softmax(classLogits, 1);
const functionalSoftmaxDim: Tensor<readonly [1, 2]> = F.softmax_dim(classLogits, 1);
const functionalLogSoftmax: Tensor<readonly [1, 2]> = F.log_softmax(classLogits, 1);
const functionalLogSoftmaxDim: Tensor<readonly [1, 2]> = F.log_softmax_dim(classLogits, 1);
const functionalDropout: Tensor<readonly [1, 2]> = F.dropout(classLogits, 0.25, { training: false });
const functionalDropoutBoolean: Tensor<readonly [1, 2]> = F.dropout(classLogits, 0.25, false);
const functionalDropoutInplaceFalse: Tensor<readonly [1, 2]> = F.dropout(classLogits, 0.25, false, false);
const functionalFlatten: Tensor<readonly [2]> = F.flatten(classLogits);
const functionalFlattenRange: Tensor<readonly [1, 2]> = F.flatten(classLogits, 1, -1);
const functionalLinear1d: Tensor<readonly [1]> = F.linear(sample, tensor([[0.5, -0.5]], [1, 2] as const), tensor([0], [1] as const));
const functionalLinear2d: Tensor<readonly [1, 2]> = F.linear(classLogits, tensor([[1, 0], [0, 1]], [2, 2] as const), tensor([0, 0], [2] as const));
const functionalLinear3d: Tensor<readonly [1, 2, 2]> = F.linear(tensor([1, 2, 3, 4], [1, 2, 2] as const), tensor([[1, 0], [0, 1]], [2, 2] as const), tensor([0, 0], [2] as const));
const functionalConv2d: Tensor<readonly [1, number, number]> = F.conv2d(
  tensor([1, 2, 3, 4], [1, 2, 2] as const),
  tensor([1, 0, 0, 1], [1, 1, 2, 2] as const),
  tensor([0], [1] as const),
);
const functionalMaxPool2d: Tensor<readonly [1, 2, 2]> = F.max_pool2d(tensor(Array.from({ length: 16 }, (_value, index) => index), [1, 4, 4] as const), 2, 2);
const functionalAvgPool2d: Tensor<readonly [1, 2, 2]> = F.avgPool2d(tensor(Array.from({ length: 16 }, (_value, index) => index), [1, 4, 4] as const), 2, 2);
const functionalNormalize: Tensor<readonly [1, 2]> = F.normalize(classLogits, 2, 1);
const functionalOneHot: Tensor<readonly [1, 2]> = F.one_hot(tensor([1], [1] as const), 2);
const functionalOneHotAlias: Tensor<readonly [1, 2]> = F.oneHot(tensor([1], [1] as const), 2);
const functionalOneHotGrid: Tensor<readonly [1, 2, 3]> = F.one_hot(tensor([0, 2], [1, 2] as const), 3);
const functionalEmbedding: Tensor<readonly [2, 2]> = F.embedding(tensor([0, 1], [2] as const), tensor([[1, 0], [0, 1], [1, 1]], [3, 2] as const));
const functionalEmbeddingGrid: Tensor<readonly [1, 2, 2]> = F.embedding(tensor([0, 1], [1, 2] as const), tensor([[1, 0], [0, 1], [1, 1]], [3, 2] as const));
const functionalLayerNorm: Tensor<readonly [1, 2]> = F.layer_norm(classLogits, [2] as const);
const functionalLayerNormPositional: Tensor<readonly [1, 2]> = F.layer_norm(classLogits, [2] as const, tensor([1, 1], [2] as const), tensor([0, 0], [2] as const), 1e-5);
const functionalRmsNorm: Tensor<readonly [1, 2]> = F.rmsNorm(classLogits, 2);
const functionalRmsNormPositional: Tensor<readonly [1, 2]> = F.rms_norm(classLogits, 2, tensor([1, 1], [2] as const), 1e-5);
const functionalBatchNorm: Tensor<readonly [2, 2]> = F.batchNorm1d(tensor([1, 3, 3, 7], [2, 2] as const), 2, { eps: 1e-5 });
const functionalBatchNormSnake: Tensor<readonly [2, 2]> = F.batch_norm1d(tensor([1, 3, 3, 7], [2, 2] as const), 2, { eps: 1e-5, momentum: 0.5 });
const functionalClassObjective: Tensor<readonly [1]> = F.cross_entropy(classLogits, classTargets, { numClasses: 2 });
const functionalClassSumObjective: Tensor<readonly [1]> = F.cross_entropy(classLogits, classTargets, { numClasses: 2, reduction: "sum" });
const functionalAliasClassObjective: Tensor<readonly [1]> = nn.F.crossEntropy(classLogits, classTargets, { classes: 2 });
const classAccuracy: number = train.accuracy(classLogits, classTargets, { classes: 2 });
const classAccuracyAlias: number = train.classification_accuracy([0.2, 1.4] as const, [1], { numClasses: 2 });
const classPredictions: Uint32Array = train.classPredictions(classLogits, { classes: 2 });
const classPredictionsAlias: Uint32Array = train.class_predictions([0.2, 1.4] as const, { numClasses: 2 });
const predictedClasses: Uint32Array = train.predictClasses(classLogits, { classes: 2 });
const predictedClassesAlias: Uint32Array = train.predict_classes([0.2, 1.4] as const, { numClasses: 2 });
const classTopKAccuracy: number = train.topKAccuracy(classLogits, classTargets, { classes: 2, k: 1 });
const classTopKAccuracyAlias: number = train.top_k_accuracy([0.2, 1.4] as const, [1], { numClasses: 2, top_k: 1 });
const classConfusionMatrix: readonly (readonly number[])[] = train.confusionMatrix(classLogits, classTargets, { classes: 2 });
const classConfusionMatrixAlias: readonly (readonly number[])[] = train.confusion_matrix([0.2, 1.4] as const, [1], { numClasses: 2 });
const classReport: TrainClassificationReport = train.classificationReport(classLogits, classTargets, { classes: 2 });
const classReportAlias: TrainClassificationReport = train.classification_report([0.2, 1.4] as const, [1], { numClasses: 2 });
const crossEntropyCriterion: CrossEntropyLoss = loss.crossEntropyLoss({ classes: 2 });
const crossEntropyCriterionAlias: CrossEntropyLoss = loss.cross_entropy_loss({ numClasses: 2 });
const crossEntropyCriterionClass: CrossEntropyLoss = new loss.CrossEntropyLoss({ classes: 2 });
const nnCrossEntropyCriterionClass: CrossEntropyLoss = new nn.CrossEntropyLoss({ classes: 2 });
const crossEntropyForward: Tensor<readonly [1]> = crossEntropyCriterion.forward(classLogits, classTargets);
const crossEntropyCall: Tensor<readonly [1]> = crossEntropyCriterionAlias.call(classLogits, classTargets);
const crossEntropyDunderCall: Tensor<readonly [1]> = crossEntropyCriterionClass.__call__(classLogits, classTargets);
const nnCrossEntropyForward: Tensor<readonly [1]> = nnCrossEntropyCriterionClass.forward(classLogits, classTargets);
const classLogProbabilities: Tensor<readonly [1, 2]> = classLogits.logSoftmax(1);
const functionalNllObjective: Tensor<readonly [1]> = F.nll_loss(classLogProbabilities, classTargets, { classes: 2 });
const functionalNllSumObjective: Tensor<readonly [1]> = F.nll_loss(classLogProbabilities, classTargets, { classes: 2, reduction: "sum" });
const negativeLogLikelihoodObjective: Tensor<readonly [1]> = loss.negative_log_likelihood(classLogProbabilities, classTargets, { classes: 2 });
const nllObjective: Tensor<readonly [1]> = loss.nllLoss(classLogProbabilities, classTargets, { classes: 2 });
const nllObjectiveAlias: Tensor<readonly [1]> = loss.nll_loss(classLogProbabilities, classTargets, { numClasses: 2 });
const nllCriterion: NLLLoss = loss.nllLossModule({ classes: 2 });
const nllCriterionAlias: NLLLoss = loss.nll_loss_module({ numClasses: 2 });
const nllCriterionClass: NLLLoss = new loss.NLLLoss({ classes: 2 });
const nnNllCriterionClass: NLLLoss = new nn.NLLLoss({ classes: 2 });
const nllForward: Tensor<readonly [1]> = nllCriterion.forward(classLogProbabilities, classTargets);
const nllCall: Tensor<readonly [1]> = nllCriterionAlias.call(classLogProbabilities, classTargets);
const nllDunderCall: Tensor<readonly [1]> = nllCriterionClass.__call__(classLogProbabilities, classTargets);
const nnNllForward: Tensor<readonly [1]> = nnNllCriterionClass.forward(classLogProbabilities, classTargets);

void fit;
void fitStopReason;
void fitBestLoss;
void classStyleOptimizer;
void variadicClassStyleModel;
void variadicClassStylePrediction;
void rmspropClassOptimizer;
void rmspropConfig;
void rmspropState;
void adagradClassOptimizer;
void adagradConfig;
void adagradState;
void classStyleObjective;
void nnClassStyleCriterion;
void nnClassStyleObjective;
void inspection;
void loadedInspection;
void serializedSnapshot;
void parsedSnapshot;
void serializedSnapshotAlias;
void parsedSnapshotAlias;
void schedulerInspectionKind;
void cosineCheckpoint;
void cosineInspection;
void cosineInspectionTMax;
void cosineInspectionEtaMin;
void schedulerState;
void exponentialScheduler;
void cosineScheduler;
void plateauScheduler;
void schedulerNamespace;
void exponentialSchedulerNamespace;
void cosineSchedulerNamespace;
void plateauSchedulerNamespace;
void batchNormFactory;
void batchNormSnakeFactory;
void batchNormForward;
void batchNormEvalForward;
void batchNormRunningMean;
void batchNormRunningVar;
void batchNormState;
void conv2dFactory;
void conv2dForward;
void conv2dBatchForward;
void conv2dState;
void conv2dSupport;
void datasetLen;
void datasetSampleAt;
void datasetSampleDunder;
void datasetIterableInput;
void datasetDunderIterInput;
void classDataset;
void classLoader;
void datasetSplitPair;
void datasetSplitPairAlias;
void datasetIterableSplit;
void datasetSubset;
void datasetIterableSubset;
void datasetTake;
void datasetConcat;
void datasetIterableConcat;
void datasetConcatAlias;
void datasetMapped;
void datasetMappedAlias;
void datasetSplitLoader;
void loaderBatchSizeAlias;
void loaderDropLastAlias;
void restoredCompiledPrediction;
void canCompile;
void regularizedParameterNames;
void regularizedParameterInfos;
void regularizedNamedModules;
void regularizedNamespaceNamedModules;
void regularizedTrain;
void regularizedNamespaceTrain;
void regularizedNamespaceEval;
void regularizedEvalPrediction;
void regularizedEvalCompiledPrediction;
void gradModeWasEnabled;
void rootGradModeNamespace;
void rootGradModeEnabled;
void rootGradModeNoGradValue;
void clonedSample;
void detachedSample;
void detachedMethodSample;
void detachedInPlaceSample;
void noGradPrediction;
void inferenceModePrediction;
void enableGradPrediction;
void manualStepEvidence;
void manualGradNorm;
void manualStepValid;
void helperClippedNorm;
void helperClippedModel;
void helperLossStep;
void helperLossStepSnake;
void evaluation;
void evaluationAlias;
void prediction;
void predictionAlias;
void manualFrozenParam;
void groupedConfig;
void groupedParamGroup;
void groupedParamGroups;
void groupedParamGroupsAlias;
void initializedGroupedWeight;
void normalInitializedGroupedWeight;
void kaimingInitializedGroupedWeight;
void kaimingNormalInitializedGroupedWeight;
void initializedGroupedBias;
void regularizedCpu;
void regularizedToCpu;
void regularizedFloat;
void regularizedFloat32;
void regularizedNamespaceCpu;
void regularizedNamespaceToCpu;
void regularizedNamespaceFloat;
void regularizedNamespaceFloat32;
void bufferedHeadState;
void bufferedHeadBuffers;
void bufferedHeadForward;
void moduleListNamedModules;
void moduleListCompiled;
void parameterListParameters;
void parameterListState;
void parameterListOptimizer;
void moduleDictNamedChildren;
void moduleDictCompiled;
void parameterDictParameters;
void parameterDictState;
void parameterDictOptimizer;
void compiledPrediction;
void forwardPrediction;
void callPrediction;
void dunderPrediction;
void namespaceCallPrediction;
void namespaceDunderPrediction;
void criterionForward;
void criterionCall;
void criterionDunderCall;
void classStyleSumReduction;
void classStyleSumObjective;
void nnClassStyleSumReduction;
void nnClassStyleSumObjective;
void functionalMseObjective;
void functionalMseLossObjective;
void functionalMseLossSumObjective;
void functionalL1Objective;
void functionalL1LossObjective;
void functionalL1LossSumObjective;
void functionalHuberObjective;
void functionalHuberLossObjective;
void functionalSmoothL1Objective;
void functionalSmoothL1LossObjective;
void functionalBceObjective;
void functionalBceWithLogitsObjective;
void l1Objective;
void l1Forward;
void l1Call;
void l1DunderCall;
void nnL1Forward;
void huberObjective;
void huberForward;
void huberCall;
void huberDunderCall;
void nnHuberForward;
void smoothL1Objective;
void smoothL1Forward;
void smoothL1Call;
void smoothL1DunderCall;
void nnSmoothL1Forward;
void binaryPrediction;
void bceObjective;
void binaryCrossEntropyObjective;
void binaryAccuracy;
void binaryAccuracyAlias;
void bceForward;
void bceCall;
void bceDunderCall;
void nnBceForward;
void bceWithLogitsObjective;
void binaryCrossEntropyWithLogitsObjective;
void customDataset;
void customDatasetLoader;
void sequentialSampler;
void randomSampler;
void replacementSampler;
void batchSampler;
void samplerLoader;
void batchSamplerLoader;
void collateFn;
void collateLoader;
void objectCollateFn;
void objectCollateLoader;
void objectCollateBatch;
void defaultCollateOptions;
void torchDefaultCollateBatch;
void binaryLogitsAccuracy;
void binaryLogitsAccuracyAlias;
void bceWithLogitsForward;
void bceWithLogitsCall;
void bceWithLogitsDunderCall;
void nnBceWithLogitsForward;
void nnCrossEntropyCriterionClass;
void nnCrossEntropyForward;
void nnNllCriterionClass;
void nnNllForward;
void classObjective;
void classObjectiveSnake;
void typedClassifierObjective;
void typedClassifierFit;
void typedClassifierPrediction;
void typedClassifierPredictionSnake;
void typedClassifierState;
void typedClassifierInspection;
void typedClassifierCompiledLogits;
void restoredHotProfile;
void restoredHotCompatibility;
void restoredHotPlan;
void restoredCompiledInto;
void functionalActivation;
void functionalActivationInplaceFalse;
void functionalGelu;
void functionalSilu;
void functionalSigmoid;
void functionalTanh;
void functionalSoftmax;
void functionalSoftmaxDim;
void functionalLogSoftmax;
void functionalLogSoftmaxDim;
void functionalDropout;
void functionalDropoutBoolean;
void functionalDropoutInplaceFalse;
void functionalFlatten;
void functionalFlattenRange;
void functionalLinear1d;
void functionalLinear2d;
void functionalLinear3d;
void functionalConv2d;
void functionalMaxPool2d;
void functionalAvgPool2d;
void functionalNormalize;
void functionalOneHot;
void functionalOneHotAlias;
void functionalOneHotGrid;
void functionalEmbedding;
void functionalEmbeddingGrid;
void functionalLayerNorm;
void functionalLayerNormPositional;
void functionalRmsNorm;
void functionalRmsNormPositional;
void functionalBatchNorm;
void functionalBatchNormSnake;
void functionalClassObjective;
void functionalClassSumObjective;
void functionalAliasClassObjective;
void functionalNllObjective;
void functionalNllSumObjective;
void nllObjective;
void nllObjectiveAlias;
void negativeLogLikelihoodObjective;
void classAccuracy;
void classAccuracyAlias;
void classPredictions;
void classPredictionsAlias;
void predictedClasses;
void predictedClassesAlias;
void classTopKAccuracy;
void classTopKAccuracyAlias;
void classConfusionMatrix;
void classConfusionMatrixAlias;
void classReport;
void classReportAlias;
void crossEntropyForward;
void crossEntropyCall;
void crossEntropyDunderCall;
void nllForward;
void nllCall;
void nllDunderCall;
