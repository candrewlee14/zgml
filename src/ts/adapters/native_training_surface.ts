type NativeTrainingTensorFactory = (value: unknown, label: string) => Float32Array;
type NativeTrainingIndexValues = (value: unknown, label: string) => ArrayLike<number>;
type NativeTrainingCheck = (status: number) => void;

type AnyRecord = Record<string, any>;

type NativeTrainingCallArgs = {
  input: Float32Array;
  targets: Uint32Array;
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
  mw1: Float32Array;
  vw1: Float32Array;
  mb1: Float32Array;
  vb1: Float32Array;
  mw2: Float32Array;
  vw2: Float32Array;
  mb2: Float32Array;
  vb2: Float32Array;
  hidden: Float32Array;
  logits: Float32Array;
  gradHidden: Float32Array;
  gradW1: Float32Array;
  gradW2: Float32Array;
  batch: number;
  inFeatures: number;
  hiddenFeatures: number;
  classes: number;
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

type NativeTrainingBulkMlpAdamCallArgs = NativeTrainingCallArgs & {
  datasetInput: Float32Array;
  datasetTargets: Uint32Array;
  indices: Uint32Array;
  batchInput: Float32Array;
  batchTargets: Uint32Array;
  sampleCount: number;
  epochs: number;
  startStep: number;
};

type NativeTrainingLinearSgdCallArgs = {
  input: Float32Array;
  target: Float32Array;
  weight: Float32Array;
  bias: Float32Array;
  output: Float32Array;
  gradWeight: Float32Array;
  batch: number;
  inFeatures: number;
  outFeatures: number;
  lr: number;
  weightDecay: number;
};

type NativeTrainingBulkLinearSgdCallArgs = NativeTrainingLinearSgdCallArgs & {
  datasetInput: Float32Array;
  datasetTargets: Float32Array;
  indices: Uint32Array;
  batchInput: Float32Array;
  batchTarget: Float32Array;
  sampleCount: number;
  epochs: number;
};

type NativeTrainingCallResult = {
  status: number;
  loss: number;
  correct: number;
};

type NativeTrainingPlan = Readonly<{
  kind: "zgml.native-training-plan";
  native: true;
  loweredBy: "zig-ffi";
  runtimePath: "JS/TS module API -> Zig native training kernel";
  backend: "cpu";
  modelKind: string;
  optimizerKind: string;
  lossKind: string;
  inputShape: readonly [number, number];
  outputShape: readonly [number, number];
  parameterCount: number;
  parameterElements: number;
  kernels: readonly string[];
  workspace: Readonly<Record<string, number>>;
}>;

type NativeTrainingMlpAdamCall = (args: NativeTrainingCallArgs) => NativeTrainingCallResult;
type NativeTrainingBulkMlpAdamCall = (args: NativeTrainingBulkMlpAdamCallArgs) => NativeTrainingCallResult & { steps: number };
type NativeTrainingLinearSgdCall = (args: NativeTrainingLinearSgdCallArgs) => NativeTrainingCallResult;
type NativeTrainingBulkLinearSgdCall = (args: NativeTrainingBulkLinearSgdCallArgs) => NativeTrainingCallResult & { steps: number };

export type NativeTrainingSurfaceOptions = {
  f32: NativeTrainingTensorFactory;
  indexValues: NativeTrainingIndexValues;
  check: NativeTrainingCheck;
  trainMlpReluCrossEntropyAdamF32: NativeTrainingMlpAdamCall;
  trainMlpReluCrossEntropyAdamWF32: NativeTrainingMlpAdamCall;
  trainMlpReluCrossEntropyAdamBulkF32?: NativeTrainingBulkMlpAdamCall;
  trainMlpReluCrossEntropyAdamWBulkF32?: NativeTrainingBulkMlpAdamCall;
  trainLinearMseSgdF32: NativeTrainingLinearSgdCall;
  trainLinearMseSgdBulkF32?: NativeTrainingBulkLinearSgdCall;
};

function positiveInteger(value: unknown, label: string) {
  const out = Number(value);
  if (!Number.isSafeInteger(out) || out <= 0) throw new Error(`${label} must be a positive safe integer, got ${value}`);
  return out;
}

function finiteNumber(value: unknown, label: string) {
  const out = Number(value);
  if (!Number.isFinite(out)) throw new Error(`${label} must be finite, got ${value}`);
  return out;
}

function tensorData(value: unknown, label: string, f32: NativeTrainingTensorFactory) {
  if (value instanceof Float32Array) return value;
  if (value && typeof value === "object" && (value as { data?: unknown }).data instanceof Float32Array) {
    return (value as { data: Float32Array }).data;
  }
  return f32(value, label);
}

function tensorShape(value: unknown): readonly number[] | null {
  if (value && typeof value === "object" && Array.isArray((value as { shape?: unknown }).shape)) {
    return (value as { shape: readonly number[] }).shape;
  }
  return null;
}

function nativeTrainingPlan(fields: Omit<NativeTrainingPlan, "kind" | "native" | "loweredBy" | "runtimePath" | "backend">): NativeTrainingPlan {
  return Object.freeze({
    kind: "zgml.native-training-plan",
    native: true,
    loweredBy: "zig-ffi",
    runtimePath: "JS/TS module API -> Zig native training kernel",
    backend: "cpu",
    ...fields,
    inputShape: Object.freeze(fields.inputShape.slice()) as readonly [number, number],
    outputShape: Object.freeze(fields.outputShape.slice()) as readonly [number, number],
    kernels: Object.freeze(fields.kernels.slice()),
    workspace: Object.freeze({ ...fields.workspace }),
  });
}

function requireLayer(model: unknown, index: number, kind: string) {
  const layers = (model as AnyRecord | null)?.layers;
  if (!Array.isArray(layers)) throw new Error("compile.trainingStep currently requires nn.Sequential");
  const layer = layers[index] as AnyRecord | undefined;
  if (!layer || layer.kind !== kind) {
    throw new Error(`compile.trainingStep expected layer ${index} to be ${kind}`);
  }
  return layer;
}

function requireLinearLayer(model: unknown, index: number) {
  const layer = requireLayer(model, index, "linear");
  const inFeatures = positiveInteger(layer.inFeatures, `compile.trainingStep layer ${index} inFeatures`);
  const outFeatures = positiveInteger(layer.outFeatures, `compile.trainingStep layer ${index} outFeatures`);
  if (!(layer.weight instanceof Float32Array)) throw new Error(`compile.trainingStep layer ${index} weight must be Float32Array-backed`);
  if (!(layer.bias instanceof Float32Array)) throw new Error(`compile.trainingStep layer ${index} requires bias=true for the native MLP step`);
  return { layer, inFeatures, outFeatures, weight: layer.weight as Float32Array, bias: layer.bias as Float32Array };
}

function requireLinearModel(model: unknown) {
  const layer = model as AnyRecord | null;
  if (!layer || layer.kind !== "linear") {
    throw new Error("compile.trainingStep MSE native path currently requires nn.Linear");
  }
  const inFeatures = positiveInteger(layer.inFeatures, "compile.trainingStep linear inFeatures");
  const outFeatures = positiveInteger(layer.outFeatures, "compile.trainingStep linear outFeatures");
  if (!(layer.weight instanceof Float32Array)) throw new Error("compile.trainingStep linear weight must be Float32Array-backed");
  if (!(layer.bias instanceof Float32Array)) throw new Error("compile.trainingStep linear requires bias=true for the native MSE step");
  return { layer, inFeatures, outFeatures, weight: layer.weight as Float32Array, bias: layer.bias as Float32Array };
}

function modelLayers(model: unknown) {
  const layers = (model as AnyRecord | null)?.layers;
  if (!Array.isArray(layers) || layers.length !== 3) {
    throw new Error("compile.trainingStep currently supports nn.Sequential([linear, relu, linear])");
  }
  const first = requireLinearLayer(model, 0);
  requireLayer(model, 1, "relu");
  const second = requireLinearLayer(model, 2);
  if (first.outFeatures !== second.inFeatures) {
    throw new Error("compile.trainingStep MLP hidden size mismatch between first and second linear layers");
  }
  return Object.freeze({ first, second });
}

function requireAdamLikeOptimizer(optimizer: unknown, params: readonly Float32Array[]) {
  const opt = optimizer as AnyRecord | null;
  if (!opt || (opt.kind !== "adam" && opt.kind !== "adamw")) {
    throw new Error("compile.trainingStep native path currently requires optim.adam or optim.adamW");
  }
  if (!Array.isArray(opt.params) || !Array.isArray(opt.m) || !Array.isArray(opt.v)) {
    throw new Error("compile.trainingStep requires an Adam/AdamW optimizer with live parameter state");
  }
  if (opt.params.length !== params.length || opt.m.length !== params.length || opt.v.length !== params.length) {
    throw new Error("compile.trainingStep Adam/AdamW state does not match the model parameter count");
  }
  for (let i = 0; i < params.length; i += 1) {
    if (opt.params[i]?.data !== params[i]) {
      throw new Error("compile.trainingStep Adam/AdamW optimizer must be created from the same model instance");
    }
    if (!(opt.m[i] instanceof Float32Array) || opt.m[i].length !== params[i].length) {
      throw new Error(`compile.trainingStep Adam/AdamW m.${i} state shape mismatch`);
    }
    if (!(opt.v[i] instanceof Float32Array) || opt.v[i].length !== params[i].length) {
      throw new Error(`compile.trainingStep Adam/AdamW v.${i} state shape mismatch`);
    }
  }
  return opt;
}

function requirePlainSgdOptimizer(optimizer: unknown, params: readonly Float32Array[]) {
  const opt = optimizer as AnyRecord | null;
  if (!opt || opt.kind !== "sgd") {
    throw new Error("compile.trainingStep MSE native path currently requires optim.sgd");
  }
  if (finiteNumber(opt.momentum, "compile.trainingStep SGD momentum") !== 0) {
    throw new Error("compile.trainingStep MSE native path currently supports SGD momentum=0");
  }
  if (!Array.isArray(opt.params)) {
    throw new Error("compile.trainingStep requires an SGD optimizer with live parameter state");
  }
  if (opt.params.length !== params.length) {
    throw new Error("compile.trainingStep SGD state does not match the model parameter count");
  }
  for (let i = 0; i < params.length; i += 1) {
    if (opt.params[i]?.data !== params[i]) {
      throw new Error("compile.trainingStep SGD optimizer must be created from the same model instance");
    }
  }
  return opt;
}

function inputBatchShape(input: unknown, inputData: Float32Array, inputShape: readonly number[]) {
  const shape = tensorShape(input) ?? inputShape;
  if (shape.length !== 2) throw new Error(`compile.trainingStep input must be rank-2 [batch, features], got [${shape.join(",")}]`);
  const batch = positiveInteger(shape[0], "compile.trainingStep input batch");
  const inFeatures = positiveInteger(shape[1], "compile.trainingStep input features");
  if (inputData.length !== batch * inFeatures) {
    throw new Error(`compile.trainingStep input length ${inputData.length} does not match ${batch}x${inFeatures}`);
  }
  return { batch, inFeatures };
}

function targetBatchShape(target: unknown, targetData: Float32Array, batch: number, outFeatures: number) {
  const shape = tensorShape(target) ?? [batch, outFeatures];
  if (shape.length !== 2) throw new Error(`compile.trainingStep target must be rank-2 [batch, features], got [${shape.join(",")}]`);
  const targetBatch = positiveInteger(shape[0], "compile.trainingStep target batch");
  const targetFeatures = positiveInteger(shape[1], "compile.trainingStep target features");
  if (targetBatch !== batch || targetFeatures !== outFeatures || targetData.length !== batch * outFeatures) {
    throw new Error(`compile.trainingStep target shape [${targetBatch},${targetFeatures}] does not match [${batch},${outFeatures}]`);
  }
}

function classTargets(value: unknown, indexValues: NativeTrainingIndexValues) {
  const raw = indexValues(value, "compile.trainingStep target");
  const out = new Uint32Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    const target = Number(raw[i]);
    if (!Number.isSafeInteger(target) || target < 0) {
      throw new Error(`compile.trainingStep target ${target} at ${i} must be a non-negative integer`);
    }
    out[i] = target;
  }
  return out;
}

function tensorDatasetBulkSource(batches: unknown, batch: number, inFeatures: number, indexValues: NativeTrainingIndexValues) {
  const loader = batches as AnyRecord | null;
  const dataset = loader?.dataset as AnyRecord | null;
  if (
    !loader ||
    loader.kind !== "zgml.data.batches" ||
    !dataset ||
    dataset.kind !== "zgml.data.tensor-dataset" ||
    loader.collateFn !== null ||
    loader.collate_fn !== null ||
    loader.batchSize !== batch ||
    typeof loader.batchRows !== "function"
  ) return null;
  const input = dataset.input as AnyRecord | null;
  const target = dataset.target as unknown;
  if (!input || !(input.data instanceof Float32Array) || !Array.isArray(input.shape) || input.shape.length !== 2) return null;
  const sampleCount = positiveInteger(input.shape[0], "native bulk training sample count");
  const features = positiveInteger(input.shape[1], "native bulk training input features");
  if (features !== inFeatures || input.data.length !== sampleCount * inFeatures) return null;
  const targetData = classTargets(target, indexValues);
  if (targetData.length !== sampleCount) return null;
  const rows = loader.batchRows() as unknown;
  if (!Array.isArray(rows)) return null;
  const flat: number[] = [];
  for (const rowBatch of rows) {
    if (!Array.isArray(rowBatch) || rowBatch.length !== batch) return null;
    for (const row of rowBatch) {
      if (!Number.isSafeInteger(row) || row < 0 || row >= sampleCount) return null;
      flat.push(row);
    }
  }
  if (flat.length === 0 || flat.length % batch !== 0) return null;
  if (loader.dropLast !== true && flat.length !== sampleCount) return null;
  return Object.freeze({
    datasetInput: input.data as Float32Array,
    datasetTargets: targetData,
    indices: new Uint32Array(flat),
    sampleCount: flat.length,
    datasetSampleCount: sampleCount,
  });
}

function tensorDatasetRegressionBulkSource(batches: unknown, batch: number, inFeatures: number, outFeatures: number) {
  const loader = batches as AnyRecord | null;
  const dataset = loader?.dataset as AnyRecord | null;
  if (
    !loader ||
    loader.kind !== "zgml.data.batches" ||
    !dataset ||
    dataset.kind !== "zgml.data.tensor-dataset" ||
    loader.collateFn !== null ||
    loader.collate_fn !== null ||
    loader.batchSize !== batch ||
    typeof loader.batchRows !== "function"
  ) return null;
  const input = dataset.input as AnyRecord | null;
  const target = dataset.target as AnyRecord | null;
  if (!input || !(input.data instanceof Float32Array) || !Array.isArray(input.shape) || input.shape.length !== 2) return null;
  if (!target || !(target.data instanceof Float32Array) || !Array.isArray(target.shape) || target.shape.length !== 2) return null;
  const sampleCount = positiveInteger(input.shape[0], "native bulk linear training sample count");
  const features = positiveInteger(input.shape[1], "native bulk linear training input features");
  const targetSamples = positiveInteger(target.shape[0], "native bulk linear training target sample count");
  const targetFeatures = positiveInteger(target.shape[1], "native bulk linear training target features");
  if (
    features !== inFeatures ||
    targetSamples !== sampleCount ||
    targetFeatures !== outFeatures ||
    input.data.length !== sampleCount * inFeatures ||
    target.data.length !== sampleCount * outFeatures
  ) return null;
  const rows = loader.batchRows() as unknown;
  if (!Array.isArray(rows)) return null;
  const flat: number[] = [];
  for (const rowBatch of rows) {
    if (!Array.isArray(rowBatch) || rowBatch.length !== batch) return null;
    for (const row of rowBatch) {
      if (!Number.isSafeInteger(row) || row < 0 || row >= sampleCount) return null;
      flat.push(row);
    }
  }
  if (flat.length === 0 || flat.length % batch !== 0) return null;
  if (loader.dropLast !== true && flat.length !== sampleCount) return null;
  return Object.freeze({
    datasetInput: input.data as Float32Array,
    datasetTargets: target.data as Float32Array,
    indices: new Uint32Array(flat),
    sampleCount: flat.length,
    datasetSampleCount: sampleCount,
  });
}

function boundedBulkIndices(source: { indices: Uint32Array; sampleCount: number }, batch: number, epochs: number, maxSteps: number | null) {
  const batchesPerEpoch = source.sampleCount / batch;
  const plannedSteps = epochs * batchesPerEpoch;
  const steps = maxSteps === null ? plannedSteps : Math.min(maxSteps, plannedSteps);
  if (steps === plannedSteps) {
    return Object.freeze({
      indices: source.indices,
      sampleCount: source.sampleCount,
      epochs,
      steps,
      stoppedEarly: false,
      stopReason: null,
    });
  }
  const indices = new Uint32Array(steps * batch);
  for (let stepIndex = 0; stepIndex < steps; stepIndex += 1) {
    const batchIndex = stepIndex % batchesPerEpoch;
    const batchStart = batchIndex * batch;
    indices.set(source.indices.subarray(batchStart, batchStart + batch), stepIndex * batch);
  }
  return Object.freeze({
    indices,
    sampleCount: indices.length,
    epochs: 1,
    steps,
    stoppedEarly: true,
    stopReason: "max-steps" as const,
  });
}

export function createAdapterNativeTrainingSurface(options: NativeTrainingSurfaceOptions) {
  function trainingStep(model: unknown, optimizer: unknown, config: AnyRecord = {}) {
    const loss = config.loss ?? config.criterion ?? "crossEntropy";
    if (loss === "mse" || loss === "meanSquaredError" || loss === "mean_squared_error") {
      return linearMseSgdTrainingStep(model, optimizer, config);
    }
    if (loss !== "crossEntropy" && loss !== "cross_entropy") {
      throw new Error(`compile.trainingStep native path currently supports loss: "crossEntropy" or "mse", got ${loss}`);
    }
    const inputShape = Array.isArray(config.inputShape ?? config.input_shape)
      ? (config.inputShape ?? config.input_shape) as readonly number[]
      : null;
    if (!inputShape || inputShape.length !== 2) {
      throw new Error("compile.trainingStep requires inputShape: [batch, features]");
    }
    const fixedInputShape = inputShape;
    const batch = positiveInteger(fixedInputShape[0], "compile.trainingStep inputShape batch");
    const inFeatures = positiveInteger(fixedInputShape[1], "compile.trainingStep inputShape features");
    const layers = modelLayers(model);
    if (layers.first.inFeatures !== inFeatures) {
      throw new Error(`compile.trainingStep input features ${inFeatures} must match model input ${layers.first.inFeatures}`);
    }
    const classes = positiveInteger(config.classes ?? config.numClasses ?? layers.second.outFeatures, "compile.trainingStep classes");
    if (classes !== layers.second.outFeatures) {
      throw new Error(`compile.trainingStep classes ${classes} must match model output ${layers.second.outFeatures}`);
    }
    const params = [layers.first.weight, layers.first.bias, layers.second.weight, layers.second.bias];
    const adam = requireAdamLikeOptimizer(optimizer, params);
    const trainMlpReluCrossEntropy = adam.kind === "adamw"
      ? options.trainMlpReluCrossEntropyAdamWF32
      : options.trainMlpReluCrossEntropyAdamF32;
    const hidden = new Float32Array(batch * layers.first.outFeatures);
    const logits = new Float32Array(batch * classes);
    const gradHidden = new Float32Array(batch * layers.first.outFeatures);
    const gradW1 = new Float32Array(layers.first.weight.length);
    const gradW2 = new Float32Array(layers.second.weight.length);
    const batchInput = new Float32Array(batch * inFeatures);
    const batchTargets = new Uint32Array(batch);
    const plan = nativeTrainingPlan({
      modelKind: "sequential-mlp-relu",
      optimizerKind: adam.kind,
      lossKind: "crossEntropy",
      inputShape: [batch, inFeatures],
      outputShape: [batch, classes],
      parameterCount: params.length,
      parameterElements: params.reduce((sum, param) => sum + param.length, 0),
      kernels: [adam.kind === "adamw" ? "zgml_train_mlp_relu_cross_entropy_adamw_f32" : "zgml_train_mlp_relu_cross_entropy_adam_f32"],
      workspace: {
        hidden: hidden.length,
        logits: logits.length,
        gradHidden: gradHidden.length,
        gradW1: gradW1.length,
        gradW2: gradW2.length,
        batchInput: batchInput.length,
        batchTargets: batchTargets.length,
      },
    });
    const bulkKernel = adam.kind === "adamw"
      ? options.trainMlpReluCrossEntropyAdamWBulkF32
      : options.trainMlpReluCrossEntropyAdamBulkF32;
    const bulkKernelName = adam.kind === "adamw"
      ? "zgml_train_mlp_relu_cross_entropy_adamw_f32_bulk"
      : "zgml_train_mlp_relu_cross_entropy_adam_f32_bulk";

    function step(input: unknown, target: unknown) {
      const inputData = tensorData(input, "compile.trainingStep input", options.f32);
      const shape = inputBatchShape(input, inputData, fixedInputShape);
      if (shape.batch !== batch || shape.inFeatures !== inFeatures) {
        throw new Error(`compile.trainingStep expected input shape [${batch},${inFeatures}], got [${shape.batch},${shape.inFeatures}]`);
      }
      const targets = classTargets(target, options.indexValues);
      if (targets.length !== batch) {
        throw new Error(`compile.trainingStep target length ${targets.length} must equal batch ${batch}`);
      }
      const result = trainMlpReluCrossEntropy({
        input: inputData,
        targets,
        w1: layers.first.weight,
        b1: layers.first.bias,
        w2: layers.second.weight,
        b2: layers.second.bias,
        mw1: adam.m[0],
        vw1: adam.v[0],
        mb1: adam.m[1],
        vb1: adam.v[1],
        mw2: adam.m[2],
        vw2: adam.v[2],
        mb2: adam.m[3],
        vb2: adam.v[3],
        hidden,
        logits,
        gradHidden,
        gradW1,
        gradW2,
        batch,
        inFeatures,
        hiddenFeatures: layers.first.outFeatures,
        classes,
        step: Number(adam.t) + 1,
        lr: finiteNumber(adam.lr, "compile.trainingStep Adam/AdamW lr"),
        beta1: finiteNumber(adam.beta1, "compile.trainingStep Adam/AdamW beta1"),
        beta2: finiteNumber(adam.beta2, "compile.trainingStep Adam/AdamW beta2"),
        eps: finiteNumber(adam.eps, "compile.trainingStep Adam/AdamW eps"),
        weightDecay: finiteNumber(adam.weightDecay, "compile.trainingStep Adam/AdamW weightDecay"),
      });
      options.check(result.status);
      adam.t = Number(adam.t) + 1;
      return Object.freeze({
        kind: "zgml.native-training-step",
        native: true,
        backend: "cpu",
        loss: result.loss,
        correct: result.correct,
        accuracy: result.correct / batch,
        batch,
        optimizerStep: adam.t,
      });
    }

    return Object.freeze({
      kind: "zgml.compiled-training-step",
      native: true,
      backend: "cpu",
      modelKind: "sequential-mlp-relu",
      optimizerKind: adam.kind,
      lossKind: "crossEntropy",
      inputShape: () => Object.freeze([batch, inFeatures] as const),
      outputShape: () => Object.freeze([batch, classes] as const),
      plan: () => plan,
      compileEvidence: () => plan,
      compile_evidence: () => plan,
      step,
      forward: step,
      fit(batches: unknown, fitOptions: AnyRecord = {}) {
        if (typeof bulkKernel !== "function") return null;
        if (
          fitOptions.onStep !== undefined ||
          fitOptions.on_step !== undefined ||
          fitOptions.earlyStopping !== undefined ||
          fitOptions.early_stopping !== undefined
        ) return null;
        const epochs = positiveInteger(fitOptions.epochs ?? 1, "native bulk training epochs");
        const maxStepsOption = fitOptions.maxSteps ?? fitOptions.max_steps;
        const maxSteps = maxStepsOption === undefined
          ? null
          : positiveInteger(maxStepsOption, "native bulk training maxSteps");
        const source = tensorDatasetBulkSource(batches, batch, inFeatures, options.indexValues);
        if (source === null) return null;
        const bounded = boundedBulkIndices(source, batch, epochs, maxSteps);
        const startStep = Number(adam.t);
        const result = bulkKernel({
          datasetInput: source.datasetInput,
          datasetTargets: source.datasetTargets,
          indices: bounded.indices,
          batchInput,
          batchTargets,
          input: batchInput,
          targets: batchTargets,
          w1: layers.first.weight,
          b1: layers.first.bias,
          w2: layers.second.weight,
          b2: layers.second.bias,
          mw1: adam.m[0],
          vw1: adam.v[0],
          mb1: adam.m[1],
          vb1: adam.v[1],
          mw2: adam.m[2],
          vw2: adam.v[2],
          mb2: adam.m[3],
          vb2: adam.v[3],
          hidden,
          logits,
          gradHidden,
          gradW1,
          gradW2,
          batch,
          inFeatures,
          hiddenFeatures: layers.first.outFeatures,
          classes,
          sampleCount: bounded.sampleCount,
          epochs: bounded.epochs,
          startStep,
          step: startStep + 1,
          lr: finiteNumber(adam.lr, "compile.trainingStep Adam/AdamW lr"),
          beta1: finiteNumber(adam.beta1, "compile.trainingStep Adam/AdamW beta1"),
          beta2: finiteNumber(adam.beta2, "compile.trainingStep Adam/AdamW beta2"),
          eps: finiteNumber(adam.eps, "compile.trainingStep Adam/AdamW eps"),
          weightDecay: finiteNumber(adam.weightDecay, "compile.trainingStep Adam/AdamW weightDecay"),
        });
        options.check(result.status);
        adam.t = startStep + result.steps;
        return Object.freeze({
          kind: "zgml.native-training-bulk-fit",
          native: true,
          nativeBulk: true,
          native_bulk: true,
          backend: "cpu",
          loss: result.loss,
          correct: result.correct,
          accuracy: result.correct / batch,
          batch,
          sampleCount: source.sampleCount,
          sample_count: source.sampleCount,
          trainedSampleCount: bounded.sampleCount,
          trained_sample_count: bounded.sampleCount,
          datasetSampleCount: source.datasetSampleCount,
          dataset_sample_count: source.datasetSampleCount,
          epochs,
          steps: result.steps,
          plannedSteps: bounded.steps,
          planned_steps: bounded.steps,
          stoppedEarly: bounded.stoppedEarly,
          stopped_early: bounded.stoppedEarly,
          stopReason: bounded.stopReason,
          stop_reason: bounded.stopReason,
          optimizerStep: adam.t,
          optimizer_step: adam.t,
          kernel: bulkKernelName,
        });
      },
      dispose() {},
      free() {},
    });
  }

  function linearMseSgdTrainingStep(model: unknown, optimizer: unknown, config: AnyRecord = {}) {
    const inputShape = Array.isArray(config.inputShape ?? config.input_shape)
      ? (config.inputShape ?? config.input_shape) as readonly number[]
      : null;
    if (!inputShape || inputShape.length !== 2) {
      throw new Error("compile.trainingStep requires inputShape: [batch, features]");
    }
    const fixedInputShape = inputShape;
    const batch = positiveInteger(fixedInputShape[0], "compile.trainingStep inputShape batch");
    const inFeatures = positiveInteger(fixedInputShape[1], "compile.trainingStep inputShape features");
    const layer = requireLinearModel(model);
    if (layer.inFeatures !== inFeatures) {
      throw new Error(`compile.trainingStep input features ${inFeatures} must match model input ${layer.inFeatures}`);
    }
    const params = [layer.weight, layer.bias];
    const sgd = requirePlainSgdOptimizer(optimizer, params);
    const output = new Float32Array(batch * layer.outFeatures);
    const gradWeight = new Float32Array(layer.weight.length);
    const batchInput = new Float32Array(batch * inFeatures);
    const batchTarget = new Float32Array(batch * layer.outFeatures);
    const plan = nativeTrainingPlan({
      modelKind: "linear",
      optimizerKind: "sgd",
      lossKind: "mse",
      inputShape: [batch, inFeatures],
      outputShape: [batch, layer.outFeatures],
      parameterCount: params.length,
      parameterElements: params.reduce((sum, param) => sum + param.length, 0),
      kernels: ["zgml_train_linear_mse_sgd_f32"],
      workspace: {
        output: output.length,
        gradWeight: gradWeight.length,
        batchInput: batchInput.length,
        batchTarget: batchTarget.length,
      },
    });
    const bulkKernel = options.trainLinearMseSgdBulkF32;

    function step(input: unknown, target: unknown) {
      const inputData = tensorData(input, "compile.trainingStep input", options.f32);
      const shape = inputBatchShape(input, inputData, fixedInputShape);
      if (shape.batch !== batch || shape.inFeatures !== inFeatures) {
        throw new Error(`compile.trainingStep expected input shape [${batch},${inFeatures}], got [${shape.batch},${shape.inFeatures}]`);
      }
      const targetData = tensorData(target, "compile.trainingStep target", options.f32);
      targetBatchShape(target, targetData, batch, layer.outFeatures);
      const result = options.trainLinearMseSgdF32({
        input: inputData,
        target: targetData,
        weight: layer.weight,
        bias: layer.bias,
        output,
        gradWeight,
        batch,
        inFeatures,
        outFeatures: layer.outFeatures,
        lr: finiteNumber(sgd.lr, "compile.trainingStep SGD lr"),
        weightDecay: finiteNumber(sgd.weightDecay, "compile.trainingStep SGD weightDecay"),
      });
      options.check(result.status);
      sgd.t = Number(sgd.t) + 1;
      return Object.freeze({
        kind: "zgml.native-training-step",
        native: true,
        backend: "cpu",
        loss: result.loss,
        correct: 0,
        accuracy: 0,
        batch,
        optimizerStep: sgd.t,
      });
    }

    return Object.freeze({
      kind: "zgml.compiled-training-step",
      native: true,
      backend: "cpu",
      modelKind: "linear",
      optimizerKind: "sgd",
      lossKind: "mse",
      inputShape: () => Object.freeze([batch, inFeatures] as const),
      outputShape: () => Object.freeze([batch, layer.outFeatures] as const),
      plan: () => plan,
      compileEvidence: () => plan,
      compile_evidence: () => plan,
      step,
      forward: step,
      fit(batches: unknown, fitOptions: AnyRecord = {}) {
        if (typeof bulkKernel !== "function") return null;
        if (
          fitOptions.onStep !== undefined ||
          fitOptions.on_step !== undefined ||
          fitOptions.earlyStopping !== undefined ||
          fitOptions.early_stopping !== undefined
        ) return null;
        const epochs = positiveInteger(fitOptions.epochs ?? 1, "native bulk linear training epochs");
        const maxStepsOption = fitOptions.maxSteps ?? fitOptions.max_steps;
        const maxSteps = maxStepsOption === undefined
          ? null
          : positiveInteger(maxStepsOption, "native bulk linear training maxSteps");
        const source = tensorDatasetRegressionBulkSource(batches, batch, inFeatures, layer.outFeatures);
        if (source === null) return null;
        const bounded = boundedBulkIndices(source, batch, epochs, maxSteps);
        const result = bulkKernel({
          datasetInput: source.datasetInput,
          datasetTargets: source.datasetTargets,
          indices: bounded.indices,
          batchInput,
          batchTarget,
          input: batchInput,
          target: batchTarget,
          weight: layer.weight,
          bias: layer.bias,
          output,
          gradWeight,
          batch,
          inFeatures,
          outFeatures: layer.outFeatures,
          sampleCount: bounded.sampleCount,
          epochs: bounded.epochs,
          lr: finiteNumber(sgd.lr, "compile.trainingStep SGD lr"),
          weightDecay: finiteNumber(sgd.weightDecay, "compile.trainingStep SGD weightDecay"),
        });
        options.check(result.status);
        sgd.t = Number(sgd.t) + result.steps;
        return Object.freeze({
          kind: "zgml.native-training-bulk-fit",
          native: true,
          nativeBulk: true,
          native_bulk: true,
          backend: "cpu",
          loss: result.loss,
          correct: 0,
          accuracy: 0,
          batch,
          sampleCount: source.sampleCount,
          sample_count: source.sampleCount,
          trainedSampleCount: bounded.sampleCount,
          trained_sample_count: bounded.sampleCount,
          datasetSampleCount: source.datasetSampleCount,
          dataset_sample_count: source.datasetSampleCount,
          epochs,
          steps: result.steps,
          plannedSteps: bounded.steps,
          planned_steps: bounded.steps,
          stoppedEarly: bounded.stoppedEarly,
          stopped_early: bounded.stoppedEarly,
          stopReason: bounded.stopReason,
          stop_reason: bounded.stopReason,
          optimizerStep: sgd.t,
          optimizer_step: sgd.t,
          kernel: "zgml_train_linear_mse_sgd_f32_bulk",
        });
      },
      dispose() {},
      free() {},
    });
  }

  return Object.freeze({
    trainingStep,
    compileForTraining: trainingStep,
    compile_for_training: trainingStep,
  });
}
