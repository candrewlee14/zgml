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

type NativeTrainingCallResult = {
  status: number;
  loss: number;
  correct: number;
};

type NativeTrainingMlpAdamCall = (args: NativeTrainingCallArgs) => NativeTrainingCallResult;

export type NativeTrainingSurfaceOptions = {
  f32: NativeTrainingTensorFactory;
  indexValues: NativeTrainingIndexValues;
  check: NativeTrainingCheck;
  trainMlpReluCrossEntropyAdamF32: NativeTrainingMlpAdamCall;
  trainMlpReluCrossEntropyAdamWF32: NativeTrainingMlpAdamCall;
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

export function createAdapterNativeTrainingSurface(options: NativeTrainingSurfaceOptions) {
  function trainingStep(model: unknown, optimizer: unknown, config: AnyRecord = {}) {
    const loss = config.loss ?? config.criterion ?? "crossEntropy";
    if (loss !== "crossEntropy" && loss !== "cross_entropy") {
      throw new Error(`compile.trainingStep native MLP currently supports loss: "crossEntropy", got ${loss}`);
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
      step,
      forward: step,
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
