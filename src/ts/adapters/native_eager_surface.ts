type NativeEagerTensorFactory = (value: unknown, label: string) => Float32Array;

type NativeEagerCheck = (status: number) => void;

type NativeEagerLinearCall = (args: {
  inputData: Float32Array;
  weightData: Float32Array;
  biasData: Float32Array | null;
  output: Float32Array;
  expectedOutput: number;
  batch: number;
  inFeatures: number;
  outFeatures: number;
  transposedWeights: boolean;
}) => number;

type NativeEagerLinearActivationCall = (args: {
  inputData: Float32Array;
  weightData: Float32Array;
  biasData: Float32Array | null;
  output: Float32Array;
  expectedOutput: number;
  batch: number;
  inFeatures: number;
  outFeatures: number;
  activation: number;
  transposedWeights: boolean;
}) => number;

type NativeEagerMatmulCall = (args: {
  lhsData: Float32Array;
  rhsData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  rows: number;
  shared: number;
  cols: number;
}) => number;

type NativeEagerBmmCall = (args: {
  lhsData: Float32Array;
  rhsData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  batch: number;
  rows: number;
  shared: number;
  cols: number;
}) => number;

type NativeEagerSoftmaxCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  rows: number;
  cols: number;
  logSoftmax: boolean;
}) => number;

type NativeEagerActivationCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  activation: number;
}) => number;

type NativeEagerElementwiseCall = (args: {
  lhsData: Float32Array;
  rhsData: Float32Array | null;
  output: Float32Array;
  expectedOutput: number;
  op: number;
}) => number;
type NativeEagerElementwiseBroadcastRhsCall = (args: {
  lhsData: Float32Array;
  rhsData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  rows: number;
  cols: number;
  op: number;
}) => number;
type NativeEagerElementwiseBroadcastLhsCall = NativeEagerElementwiseBroadcastRhsCall;

type NativeEagerWhereCall = (args: {
  conditionData: Float32Array;
  inputData: Float32Array;
  otherData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
}) => number;

type NativeEagerClampCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  min: number;
  max: number;
  hasMin: boolean;
  hasMax: boolean;
}) => number;

type NativeEagerReduceCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  op: number;
}) => number;

type NativeEagerReduceDimCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  outer: number;
  reduce: number;
  inner: number;
  op: number;
}) => number;
type NativeEagerArgReduceDimCall = NativeEagerReduceDimCall;

type NativeEagerCumsumCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  outer: number;
  axisLen: number;
  inner: number;
  reverse: boolean;
}) => number;

type NativeEagerMomentCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  outer: number;
  reduce: number;
  inner: number;
  correction: number;
  sqrtOutput: boolean;
}) => number;

type NativeEagerDotCall = (args: {
  lhsData: Float32Array;
  rhsData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
}) => number;

type NativeEagerPermuteCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  outputShape: Uint32Array;
  inputStrides: Uint32Array;
  axes: Uint32Array;
  rank: number;
}) => number;

type NativeEagerConv2dCall = (args: {
  inputData: Float32Array;
  weightData: Float32Array;
  biasData: Float32Array | null;
  output: Float32Array;
  expectedOutput: number;
  batch: number;
  inChannels: number;
  height: number;
  width: number;
  outChannels: number;
  kernelH: number;
  kernelW: number;
  strideH: number;
  strideW: number;
  paddingH: number;
  paddingW: number;
  dilationH: number;
  dilationW: number;
  outH: number;
  outW: number;
}) => number;

type NativeEagerPool2dCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  batch: number;
  channels: number;
  height: number;
  width: number;
  kernelH: number;
  kernelW: number;
  strideH: number;
  strideW: number;
  paddingH: number;
  paddingW: number;
  dilationH: number;
  dilationW: number;
  outH: number;
  outW: number;
  op: number;
  ceilMode: boolean;
  countIncludePad: boolean;
}) => number;

type NativeEagerSurfaceOptions = {
  f32: NativeEagerTensorFactory;
  check: NativeEagerCheck;
  linearF32: NativeEagerLinearCall;
  linearActivationF32: NativeEagerLinearActivationCall;
  activationF32?: NativeEagerActivationCall;
  elementwiseF32?: NativeEagerElementwiseCall;
  elementwiseBroadcastRhsF32?: NativeEagerElementwiseBroadcastRhsCall;
  elementwiseBroadcastLhsF32?: NativeEagerElementwiseBroadcastLhsCall;
  whereF32?: NativeEagerWhereCall;
  clampF32?: NativeEagerClampCall;
  reduceF32?: NativeEagerReduceCall;
  reduceDimF32?: NativeEagerReduceDimCall;
  argReduceDimF32?: NativeEagerArgReduceDimCall;
  cumsumF32?: NativeEagerCumsumCall;
  momentF32?: NativeEagerMomentCall;
  dotF32?: NativeEagerDotCall;
  permuteF32?: NativeEagerPermuteCall;
  conv2dF32?: NativeEagerConv2dCall;
  pool2dF32?: NativeEagerPool2dCall;
  matmulF32?: NativeEagerMatmulCall;
  bmmF32?: NativeEagerBmmCall;
  softmaxF32: NativeEagerSoftmaxCall;
};

function nativeEagerTensorData(value: unknown, label: string, f32: NativeEagerTensorFactory): Float32Array {
  if (value instanceof Float32Array) return value;
  if (value && typeof value === "object" && (value as { data?: unknown }).data instanceof Float32Array) {
    return (value as { data: Float32Array }).data;
  }
  return f32(value, label);
}

function nativeEagerShape(value: unknown): readonly number[] | null {
  if (value && typeof value === "object" && Array.isArray((value as { shape?: unknown }).shape)) {
    return (value as { shape: readonly number[] }).shape;
  }
  return null;
}

function nativeEagerPositiveInteger(value: unknown, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function nativeEagerNonNegativeInteger(value: unknown, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer, got ${value}`);
  }
  return parsed;
}

function nativeEagerBoolean(value: unknown, label: string, defaultValue: boolean): boolean {
  if (value === undefined || value === null) return defaultValue;
  if (typeof value === "boolean") return value;
  throw new Error(`${label} must be a boolean`);
}

function nativeEagerLinearShape(input: unknown, weights: unknown, options: Record<string, unknown> = {}) {
  const inputShape = nativeEagerShape(input);
  const weightShape = nativeEagerShape(weights);
  const transposedWeights = nativeEagerLinearTransposedWeights(options);
  const inferredInFeatures = weightShape && weightShape.length === 2 ? weightShape[transposedWeights ? 1 : 0] : null;
  const inferredOutFeatures = weightShape && weightShape.length === 2 ? weightShape[transposedWeights ? 0 : 1] : null;
  const inFeatures = options.inFeatures ?? options.in_features ?? inferredInFeatures;
  const outFeatures = options.outFeatures ?? options.out_features ?? inferredOutFeatures;
  const batch = options.batch ?? (
    inputShape && inputShape.length === 2
      ? inputShape[0]
      : inputShape && inputShape.length === 1
        ? 1
        : null
  );
  return Object.freeze({
    batch: nativeEagerPositiveInteger(batch, "nativeEager.linearInto batch"),
    inFeatures: nativeEagerPositiveInteger(inFeatures, "nativeEager.linearInto inFeatures"),
    outFeatures: nativeEagerPositiveInteger(outFeatures, "nativeEager.linearInto outFeatures"),
    transposedWeights,
  });
}

function nativeEagerLinearTransposedWeights(options: Record<string, unknown>) {
  const layout = options.weightLayout ?? options.weight_layout ?? options.weightsLayout ?? options.weights_layout ?? options.layout;
  if (layout === undefined || layout === null || layout === false || layout === "in-out" || layout === "in_out" || layout === "io" || layout === "row-major:linear.weight[in_features,out_features]") {
    return false;
  }
  if (layout === true || layout === "out-in" || layout === "out_in" || layout === "oi" || layout === "pytorch" || layout === "torch" || layout === "row-major:linear.weight[out_features,in_features]") {
    return true;
  }
  throw new Error(`nativeEager.linearInto weightLayout must be in-out or out-in, got ${layout}`);
}

function nativeEagerActivationId(value: unknown, label: string): number {
  const normalized = String(value ?? "").trim();
  switch (normalized) {
    case "relu": return 1;
    case "gelu": return 2;
    case "silu": return 3;
    case "sigmoid": return 4;
    case "tanh": return 14;
    default: throw new Error(`${label} activation must be relu, gelu, silu, sigmoid, or tanh, got ${value}`);
  }
}

function nativeEagerElementwiseOpId(value: unknown, label: string): number {
  const normalized = String(value ?? "").trim();
  switch (normalized) {
    case "add": return 1;
    case "sub": return 2;
    case "mul": return 3;
    case "div": return 4;
    case "neg":
    case "negative": return 5;
    case "exp": return 6;
    case "log": return 7;
    case "expm1":
    case "expMinusOne": return 30;
    case "log1p":
    case "logOnePlus": return 31;
    case "sqr":
    case "square": return 8;
    case "recip":
    case "reciprocal": return 9;
    case "abs": return 10;
    case "sqrt": return 11;
    case "maximum":
    case "max": return 12;
    case "minimum":
    case "min": return 13;
    case "rsqrt":
    case "reciprocal_sqrt":
    case "reciprocalSqrt": return 14;
    case "eq":
    case "equal": return 15;
    case "ne":
    case "not_equal":
    case "notEqual": return 16;
    case "lt":
    case "less": return 17;
    case "le":
    case "less_equal":
    case "lessEqual": return 18;
    case "gt":
    case "greater": return 19;
    case "ge":
    case "greater_equal":
    case "greaterEqual": return 20;
    case "sign":
    case "sgn": return 21;
    case "step": return 22;
    case "floor": return 23;
    case "ceil": return 24;
    case "round": return 25;
    case "trunc":
    case "truncate": return 26;
    case "isnan":
    case "isNaN": return 27;
    case "isinf":
    case "isInf":
    case "is_infinite":
    case "isInfinite": return 28;
    case "isfinite":
    case "isFinite": return 29;
    default: throw new Error(`${label} op must be add, sub, mul, div, neg, exp, log, expm1, log1p, sqr, recip, abs, sqrt, rsqrt, maximum, minimum, eq, ne, lt, le, gt, ge, sign, step, floor, ceil, round, trunc, isnan, isinf, or isfinite, got ${value}`);
  }
}

function nativeEagerReduceOpId(value: unknown, label: string): number {
  const normalized = String(value ?? "").trim();
  switch (normalized) {
    case "sum": return 1;
    case "mean": return 2;
    case "max": return 3;
    case "min": return 4;
    case "prod": return 5;
    default: throw new Error(`${label} op must be sum, mean, max, min, or prod, got ${value}`);
  }
}

function nativeEagerArgReduceOpId(value: unknown, label: string): number {
  const normalized = String(value ?? "").trim();
  switch (normalized) {
    case "argmax": return 6;
    case "argmin": return 7;
    default: throw new Error(`${label} op must be argmax or argmin, got ${value}`);
  }
}

function nativeEagerPool2dOpId(value: unknown, label: string): number {
  const normalized = String(value ?? "").trim();
  switch (normalized) {
    case "max": return 1;
    case "avg":
    case "average": return 2;
    default: throw new Error(`${label} op must be max, avg, or average, got ${value}`);
  }
}

function nativeEagerLinearInputs(
  label: "nativeEager.linearInto" | "nativeEager.linearActivationInto",
  output: Float32Array,
  input: unknown,
  weights: unknown,
  options: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  const weightData = nativeEagerTensorData(weights, `${label} weights`, f32);
  const biasValue = options.bias ?? null;
  const biasData = biasValue == null ? null : nativeEagerTensorData(biasValue, `${label} bias`, f32);
  const shape = nativeEagerLinearShape(input, weights, options);
  const expectedInput = shape.batch * shape.inFeatures;
  const expectedWeights = shape.inFeatures * shape.outFeatures;
  const expectedOutput = shape.batch * shape.outFeatures;
  if (inputData.length !== expectedInput) {
    throw new Error(`${label} input length ${inputData.length} does not match ${shape.batch}x${shape.inFeatures}`);
  }
  if (weightData.length !== expectedWeights) {
    throw new Error(`${label} weights length ${weightData.length} does not match ${shape.inFeatures}x${shape.outFeatures}`);
  }
  if (biasData && biasData.length !== shape.outFeatures) {
    throw new Error(`${label} bias length ${biasData.length} does not match outFeatures ${shape.outFeatures}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    weightData,
    biasData,
    output,
    expectedOutput,
    batch: shape.batch,
    inFeatures: shape.inFeatures,
    outFeatures: shape.outFeatures,
    transposedWeights: shape.transposedWeights,
  };
}

function nativeEagerSoftmaxShape(input: unknown, options: Record<string, unknown> = {}) {
  const inputShape = nativeEagerShape(input);
  if (options.rows !== undefined || options.cols !== undefined) {
    return Object.freeze({
      rows: nativeEagerPositiveInteger(options.rows, "nativeEager.softmaxInto rows"),
      cols: nativeEagerPositiveInteger(options.cols, "nativeEager.softmaxInto cols"),
    });
  }
  if (!inputShape || inputShape.length === 0) {
    throw new Error("nativeEager.softmaxInto requires tensor shape or explicit rows/cols");
  }
  const dimValue = options.dim ?? -1;
  const dim = Number(dimValue);
  const rank = inputShape.length;
  const normalizedDim = dim < 0 ? rank + dim : dim;
  if (!Number.isSafeInteger(dim) || normalizedDim !== rank - 1) {
    throw new Error(`nativeEager.softmaxInto currently supports the last dimension only, got dim ${dimValue}`);
  }
  const cols = nativeEagerPositiveInteger(inputShape[rank - 1], "nativeEager.softmaxInto cols");
  const leading = inputShape.slice(0, -1);
  const rows = leading.length === 0
    ? 1
    : leading.reduce((acc, value) => acc * nativeEagerPositiveInteger(value, "nativeEager.softmaxInto row shape"), 1);
  return Object.freeze({ rows, cols });
}

function nativeEagerSoftmaxInputs(
  label: "nativeEager.softmaxInto" | "nativeEager.logSoftmaxInto",
  output: Float32Array,
  input: unknown,
  options: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  const shape = nativeEagerSoftmaxShape(input, options);
  const expectedOutput = shape.rows * shape.cols;
  if (inputData.length !== expectedOutput) {
    throw new Error(`${label} input length ${inputData.length} does not match ${shape.rows}x${shape.cols}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    rows: shape.rows,
    cols: shape.cols,
  };
}

function nativeEagerActivationInputs(
  output: Float32Array,
  input: unknown,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.activationInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  if (output.length < inputData.length) {
    throw new Error(`${label} output length ${output.length} is smaller than ${inputData.length}`);
  }
  return {
    inputData,
    output,
    expectedOutput: inputData.length,
  };
}

function nativeEagerElementwiseInputs(
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.elementwiseInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const lhsData = nativeEagerTensorData(lhs, `${label} lhs`, f32);
  if (lhsData.length === 0) {
    throw new Error(`${label} lhs must be non-empty`);
  }
  const rhsData = rhs == null ? null : nativeEagerTensorData(rhs, `${label} rhs`, f32);
  const lhsShape = nativeEagerShape(lhs);
  const rhsShape = rhs == null ? null : nativeEagerShape(rhs);
  const explicitBroadcast = callOptions.broadcast ?? callOptions.broadcastSide ?? callOptions.broadcast_side;
  const explicitBroadcastSide = explicitBroadcast === "lhs" || explicitBroadcast === "left"
    ? "lhs"
    : explicitBroadcast === "rhs" || explicitBroadcast === "right"
      ? "rhs"
      : null;
  const inferredRhsCols = lhsShape && lhsShape.length >= 2 ? lhsShape[lhsShape.length - 1] : null;
  const inferredRhsRows = inferredRhsCols && inferredRhsCols > 0 && lhsData.length % inferredRhsCols === 0
    ? lhsData.length / inferredRhsCols
    : null;
  const rhsCanUseInferredCols = rhsData !== null &&
    inferredRhsRows !== null &&
    inferredRhsCols !== null &&
    rhsData.length === inferredRhsCols &&
    (
      !rhsShape ||
      (rhsShape.length === 1 && rhsShape[0] === inferredRhsCols)
    );
  const inferredLhsCols = rhsShape && rhsShape.length >= 2 ? rhsShape[rhsShape.length - 1] : null;
  const inferredLhsRows = inferredLhsCols && inferredLhsCols > 0 && rhsData !== null && rhsData.length % inferredLhsCols === 0
    ? rhsData.length / inferredLhsCols
    : null;
  const lhsCanUseInferredCols = rhsData !== null &&
    inferredLhsRows !== null &&
    inferredLhsCols !== null &&
    lhsData.length === inferredLhsCols &&
    (
      !lhsShape ||
      (lhsShape.length === 1 && lhsShape[0] === inferredLhsCols)
    );
  const inferredBroadcastSide = rhsCanUseInferredCols
    ? "rhs"
    : lhsCanUseInferredCols
      ? "lhs"
      : null;
  const broadcastSide = explicitBroadcastSide ?? inferredBroadcastSide;
  const rowsValue = callOptions.rows ?? callOptions.batch ?? (
    broadcastSide === "rhs" ? inferredRhsRows : broadcastSide === "lhs" ? inferredLhsRows : null
  );
  const colsValue = callOptions.cols ?? callOptions.features ?? (
    broadcastSide === "rhs" ? inferredRhsCols : broadcastSide === "lhs" ? inferredLhsCols : null
  );
  const canBroadcast = rhsData !== null &&
    rhsData.length !== 1 &&
    rhsData.length !== lhsData.length &&
    broadcastSide !== null &&
    rowsValue != null &&
    colsValue != null;
  const broadcastShape = canBroadcast
    ? Object.freeze({
      side: broadcastSide,
      rows: nativeEagerPositiveInteger(rowsValue, `${label} rows`),
      cols: nativeEagerPositiveInteger(colsValue, `${label} cols`),
    })
    : null;
  const expectedOutput = broadcastShape?.side === "lhs" && rhsData !== null
    ? rhsData.length
    : lhsData.length;
  if (rhsData !== null && rhsData.length !== 1 && rhsData.length !== lhsData.length) {
    if (
      !broadcastShape ||
      (
        broadcastShape.side === "rhs" &&
        (broadcastShape.rows * broadcastShape.cols !== lhsData.length || rhsData.length !== broadcastShape.cols)
      ) ||
      (
        broadcastShape.side === "lhs" &&
        (broadcastShape.rows * broadcastShape.cols !== rhsData.length || lhsData.length !== broadcastShape.cols)
      )
    ) {
      throw new Error(`${label} operand lengths must be scalar, same-shape, or last-dimension row-broadcast compatible`);
    }
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    lhsData,
    rhsData,
    output,
    expectedOutput,
    broadcastShape,
  };
}

function nativeEagerWhereInputs(
  output: Float32Array,
  condition: unknown,
  input: unknown,
  other: unknown,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.whereInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const conditionData = nativeEagerTensorData(condition, `${label} condition`, f32);
  if (conditionData.length === 0) {
    throw new Error(`${label} condition must be non-empty`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  const otherData = nativeEagerTensorData(other, `${label} other`, f32);
  if (inputData.length !== 1 && inputData.length !== conditionData.length) {
    throw new Error(`${label} input length ${inputData.length} must be 1 or match condition length ${conditionData.length}`);
  }
  if (otherData.length !== 1 && otherData.length !== conditionData.length) {
    throw new Error(`${label} other length ${otherData.length} must be 1 or match condition length ${conditionData.length}`);
  }
  if (output.length < conditionData.length) {
    throw new Error(`${label} output length ${output.length} is smaller than ${conditionData.length}`);
  }
  return {
    conditionData,
    inputData,
    otherData,
    output,
    expectedOutput: conditionData.length,
  };
}

function nativeEagerClampInputs(
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.clampInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  if (output.length < inputData.length) {
    throw new Error(`${label} output length ${output.length} is smaller than ${inputData.length}`);
  }
  const rawMin = callOptions.min ?? callOptions.minimum;
  const rawMax = callOptions.max ?? callOptions.maximum;
  const hasMin = rawMin !== undefined && rawMin !== null;
  const hasMax = rawMax !== undefined && rawMax !== null;
  if (!hasMin && !hasMax) {
    throw new Error(`${label} requires min, max, or both`);
  }
  const min = hasMin ? Number(rawMin) : 0;
  const max = hasMax ? Number(rawMax) : 0;
  if (hasMin && !Number.isFinite(min)) throw new Error(`${label} min must be finite, got ${rawMin}`);
  if (hasMax && !Number.isFinite(max)) throw new Error(`${label} max must be finite, got ${rawMax}`);
  if (hasMin && hasMax && min > max) throw new Error(`${label} min ${min} must be <= max ${max}`);
  return {
    inputData,
    output,
    expectedOutput: inputData.length,
    min,
    max,
    hasMin,
    hasMax,
  };
}

function nativeEagerReduceInputs(
  output: Float32Array,
  input: unknown,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.reduceInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  if (output.length < 1) {
    throw new Error(`${label} output length ${output.length} is smaller than 1`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  return {
    inputData,
    output,
    expectedOutput: 1,
  };
}

function nativeEagerReduceDimInputs(
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.reduceDimInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  const outer = nativeEagerPositiveInteger(callOptions.outer, `${label} outer`);
  const reduce = nativeEagerPositiveInteger(callOptions.reduce, `${label} reduce`);
  const inner = nativeEagerPositiveInteger(callOptions.inner, `${label} inner`);
  const expectedInput = outer * reduce * inner;
  const expectedOutput = outer * inner;
  if (inputData.length !== expectedInput) {
    throw new Error(`${label} input length ${inputData.length} must equal outer*reduce*inner ${expectedInput}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    outer,
    reduce,
    inner,
  };
}

function nativeEagerCumsumInputs(
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.cumsumInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  const outer = nativeEagerPositiveInteger(callOptions.outer, `${label} outer`);
  const axisLen = nativeEagerPositiveInteger(callOptions.axis ?? callOptions.axisLen ?? callOptions.axis_len, `${label} axis`);
  const inner = nativeEagerPositiveInteger(callOptions.inner, `${label} inner`);
  const expectedOutput = outer * axisLen * inner;
  if (inputData.length !== expectedOutput) {
    throw new Error(`${label} input length ${inputData.length} must equal outer*axis*inner ${expectedOutput}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    outer,
    axisLen,
    inner,
    reverse: nativeEagerBoolean(callOptions.reverse, `${label} reverse`, false),
  };
}

function nativeEagerMomentInputs(
  label: string,
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
  sqrtOutput: boolean,
) {
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  const outer = nativeEagerPositiveInteger(callOptions.outer, `${label} outer`);
  const reduce = nativeEagerPositiveInteger(callOptions.reduce, `${label} reduce`);
  const inner = nativeEagerPositiveInteger(callOptions.inner, `${label} inner`);
  const correction = Number(callOptions.correction ?? 0);
  if (!Number.isFinite(correction) || correction < 0) {
    throw new Error(`${label} correction must be finite and >= 0, got ${callOptions.correction}`);
  }
  if (reduce - correction <= 0) {
    throw new Error(`${label} correction ${correction} must be less than reduce ${reduce}`);
  }
  const expectedInput = outer * reduce * inner;
  const expectedOutput = outer * inner;
  if (inputData.length !== expectedInput) {
    throw new Error(`${label} input length ${inputData.length} must equal outer*reduce*inner ${expectedInput}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    outer,
    reduce,
    inner,
    correction,
    sqrtOutput,
  };
}

function nativeEagerDotInputs(
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.dotInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  if (output.length < 1) {
    throw new Error(`${label} output length ${output.length} is smaller than 1`);
  }
  const lhsData = nativeEagerTensorData(lhs, `${label} lhs`, f32);
  const rhsData = nativeEagerTensorData(rhs, `${label} rhs`, f32);
  if (lhsData.length === 0) {
    throw new Error(`${label} lhs must be non-empty`);
  }
  if (lhsData.length !== rhsData.length) {
    throw new Error(`${label} rhs length ${rhsData.length} must match lhs length ${lhsData.length}`);
  }
  return {
    lhsData,
    rhsData,
    output,
    expectedOutput: 1,
  };
}

function nativeEagerPermuteInputs(
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.permuteInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  if (inputData.length === 0) {
    throw new Error(`${label} input must be non-empty`);
  }
  const outputShape = callOptions.outputShape ?? callOptions.output_shape;
  const inputStrides = callOptions.inputStrides ?? callOptions.input_strides;
  const axes = callOptions.axes;
  if (!(outputShape instanceof Uint32Array) || !(inputStrides instanceof Uint32Array) || !(axes instanceof Uint32Array)) {
    throw new Error(`${label} requires Uint32Array outputShape, inputStrides, and axes`);
  }
  const rank = outputShape.length;
  if (rank === 0 || inputStrides.length !== rank || axes.length !== rank) {
    throw new Error(`${label} outputShape, inputStrides, and axes must have the same non-empty rank`);
  }
  let expectedOutput = 1;
  for (let i = 0; i < outputShape.length; i += 1) {
    const dim = outputShape[i];
    if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`${label} outputShape entries must be positive uint32 values`);
    expectedOutput *= dim;
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    outputShape,
    inputStrides,
    axes,
    rank,
  };
}

function nativeEagerConv2dInputs(
  output: Float32Array,
  input: unknown,
  weights: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.conv2dInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  const weightData = nativeEagerTensorData(weights, `${label} weights`, f32);
  const biasValue = callOptions.bias ?? null;
  const biasData = biasValue == null ? null : nativeEagerTensorData(biasValue, `${label} bias`, f32);
  const inputShape = nativeEagerShape(input);
  const weightShape = nativeEagerShape(weights);
  const batch = callOptions.batch ?? (inputShape && inputShape.length === 4 ? inputShape[0] : inputShape && inputShape.length === 3 ? 1 : null);
  const inChannels = callOptions.inChannels ?? callOptions.in_channels ?? (
    inputShape && inputShape.length === 4 ? inputShape[1] : inputShape && inputShape.length === 3 ? inputShape[0] : null
  );
  const height = callOptions.height ?? (inputShape && inputShape.length === 4 ? inputShape[2] : inputShape && inputShape.length === 3 ? inputShape[1] : null);
  const width = callOptions.width ?? (inputShape && inputShape.length === 4 ? inputShape[3] : inputShape && inputShape.length === 3 ? inputShape[2] : null);
  const outChannels = callOptions.outChannels ?? callOptions.out_channels ?? (weightShape && weightShape.length === 4 ? weightShape[0] : null);
  const kernelH = callOptions.kernelH ?? callOptions.kernel_h ?? (weightShape && weightShape.length === 4 ? weightShape[2] : null);
  const kernelW = callOptions.kernelW ?? callOptions.kernel_w ?? (weightShape && weightShape.length === 4 ? weightShape[3] : null);
  const shape = Object.freeze({
    batch: nativeEagerPositiveInteger(batch, `${label} batch`),
    inChannels: nativeEagerPositiveInteger(inChannels, `${label} inChannels`),
    height: nativeEagerPositiveInteger(height, `${label} height`),
    width: nativeEagerPositiveInteger(width, `${label} width`),
    outChannels: nativeEagerPositiveInteger(outChannels, `${label} outChannels`),
    kernelH: nativeEagerPositiveInteger(kernelH, `${label} kernelH`),
    kernelW: nativeEagerPositiveInteger(kernelW, `${label} kernelW`),
    strideH: nativeEagerPositiveInteger(callOptions.strideH ?? callOptions.stride_h ?? 1, `${label} strideH`),
    strideW: nativeEagerPositiveInteger(callOptions.strideW ?? callOptions.stride_w ?? 1, `${label} strideW`),
    paddingH: nativeEagerNonNegativeInteger(callOptions.paddingH ?? callOptions.padding_h ?? 0, `${label} paddingH`),
    paddingW: nativeEagerNonNegativeInteger(callOptions.paddingW ?? callOptions.padding_w ?? 0, `${label} paddingW`),
    dilationH: nativeEagerPositiveInteger(callOptions.dilationH ?? callOptions.dilation_h ?? 1, `${label} dilationH`),
    dilationW: nativeEagerPositiveInteger(callOptions.dilationW ?? callOptions.dilation_w ?? 1, `${label} dilationW`),
    outH: nativeEagerPositiveInteger(callOptions.outH ?? callOptions.out_h, `${label} outH`),
    outW: nativeEagerPositiveInteger(callOptions.outW ?? callOptions.out_w, `${label} outW`),
  });
  const expectedInput = shape.batch * shape.inChannels * shape.height * shape.width;
  const expectedWeights = shape.outChannels * shape.inChannels * shape.kernelH * shape.kernelW;
  const expectedOutput = shape.batch * shape.outChannels * shape.outH * shape.outW;
  if (inputData.length !== expectedInput) {
    throw new Error(`${label} input length ${inputData.length} does not match ${shape.batch}x${shape.inChannels}x${shape.height}x${shape.width}`);
  }
  if (weightData.length !== expectedWeights) {
    throw new Error(`${label} weights length ${weightData.length} does not match ${shape.outChannels}x${shape.inChannels}x${shape.kernelH}x${shape.kernelW}`);
  }
  if (biasData && biasData.length !== shape.outChannels) {
    throw new Error(`${label} bias length ${biasData.length} does not match outChannels ${shape.outChannels}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return { inputData, weightData, biasData, output, expectedOutput, ...shape };
}

function nativeEagerPool2dInputs(
  output: Float32Array,
  input: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.pool2dInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const inputData = nativeEagerTensorData(input, `${label} input`, f32);
  const inputShape = nativeEagerShape(input);
  const batch = callOptions.batch ?? (inputShape && inputShape.length === 4 ? inputShape[0] : inputShape && inputShape.length === 3 ? 1 : null);
  const channels = callOptions.channels ?? (inputShape && inputShape.length === 4 ? inputShape[1] : inputShape && inputShape.length === 3 ? inputShape[0] : null);
  const height = callOptions.height ?? (inputShape && inputShape.length === 4 ? inputShape[2] : inputShape && inputShape.length === 3 ? inputShape[1] : null);
  const width = callOptions.width ?? (inputShape && inputShape.length === 4 ? inputShape[3] : inputShape && inputShape.length === 3 ? inputShape[2] : null);
  const shape = Object.freeze({
    batch: nativeEagerPositiveInteger(batch, `${label} batch`),
    channels: nativeEagerPositiveInteger(channels, `${label} channels`),
    height: nativeEagerPositiveInteger(height, `${label} height`),
    width: nativeEagerPositiveInteger(width, `${label} width`),
    kernelH: nativeEagerPositiveInteger(callOptions.kernelH ?? callOptions.kernel_h, `${label} kernelH`),
    kernelW: nativeEagerPositiveInteger(callOptions.kernelW ?? callOptions.kernel_w, `${label} kernelW`),
    strideH: nativeEagerPositiveInteger(callOptions.strideH ?? callOptions.stride_h ?? 1, `${label} strideH`),
    strideW: nativeEagerPositiveInteger(callOptions.strideW ?? callOptions.stride_w ?? 1, `${label} strideW`),
    paddingH: nativeEagerNonNegativeInteger(callOptions.paddingH ?? callOptions.padding_h ?? 0, `${label} paddingH`),
    paddingW: nativeEagerNonNegativeInteger(callOptions.paddingW ?? callOptions.padding_w ?? 0, `${label} paddingW`),
    dilationH: nativeEagerPositiveInteger(callOptions.dilationH ?? callOptions.dilation_h ?? 1, `${label} dilationH`),
    dilationW: nativeEagerPositiveInteger(callOptions.dilationW ?? callOptions.dilation_w ?? 1, `${label} dilationW`),
    outH: nativeEagerPositiveInteger(callOptions.outH ?? callOptions.out_h, `${label} outH`),
    outW: nativeEagerPositiveInteger(callOptions.outW ?? callOptions.out_w, `${label} outW`),
  });
  const expectedInput = shape.batch * shape.channels * shape.height * shape.width;
  const expectedOutput = shape.batch * shape.channels * shape.outH * shape.outW;
  if (inputData.length !== expectedInput) {
    throw new Error(`${label} input length ${inputData.length} does not match ${shape.batch}x${shape.channels}x${shape.height}x${shape.width}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return {
    inputData,
    output,
    expectedOutput,
    ...shape,
    op: nativeEagerPool2dOpId(callOptions.op, label),
    ceilMode: nativeEagerBoolean(callOptions.ceilMode ?? callOptions.ceil_mode, `${label} ceilMode`, false),
    countIncludePad: nativeEagerBoolean(callOptions.countIncludePad ?? callOptions.count_include_pad, `${label} countIncludePad`, true),
  };
}

function nativeEagerMatmulInputs(
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.matmulInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const lhsData = nativeEagerTensorData(lhs, `${label} lhs`, f32);
  const rhsData = nativeEagerTensorData(rhs, `${label} rhs`, f32);
  const lhsShape = nativeEagerShape(lhs);
  const rhsShape = nativeEagerShape(rhs);
  const rows = callOptions.rows ?? callOptions.lhsRows ?? callOptions.lhs_rows ?? (lhsShape && lhsShape.length === 2 ? lhsShape[0] : null);
  const shared = callOptions.shared ?? callOptions.lhsCols ?? callOptions.lhs_cols ?? callOptions.rhsRows ?? callOptions.rhs_rows ?? (
    lhsShape && lhsShape.length === 2
      ? lhsShape[1]
      : rhsShape && rhsShape.length === 2
        ? rhsShape[0]
        : null
  );
  const cols = callOptions.cols ?? callOptions.rhsCols ?? callOptions.rhs_cols ?? (rhsShape && rhsShape.length === 2 ? rhsShape[1] : null);
  const shape = Object.freeze({
    rows: nativeEagerPositiveInteger(rows, `${label} rows`),
    shared: nativeEagerPositiveInteger(shared, `${label} shared`),
    cols: nativeEagerPositiveInteger(cols, `${label} cols`),
  });
  const expectedLhs = shape.rows * shape.shared;
  const expectedRhs = shape.shared * shape.cols;
  const expectedOutput = shape.rows * shape.cols;
  if (lhsData.length !== expectedLhs) {
    throw new Error(`${label} lhs length ${lhsData.length} does not match ${shape.rows}x${shape.shared}`);
  }
  if (rhsData.length !== expectedRhs) {
    throw new Error(`${label} rhs length ${rhsData.length} does not match ${shape.shared}x${shape.cols}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return { lhsData, rhsData, output, expectedOutput, ...shape };
}

function nativeEagerBmmInputs(
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  callOptions: Record<string, unknown>,
  f32: NativeEagerTensorFactory,
) {
  const label = "nativeEager.bmmInto";
  if (!(output instanceof Float32Array)) {
    throw new Error(`${label} output must be a Float32Array`);
  }
  const lhsData = nativeEagerTensorData(lhs, `${label} lhs`, f32);
  const rhsData = nativeEagerTensorData(rhs, `${label} rhs`, f32);
  const lhsShape = nativeEagerShape(lhs);
  const rhsShape = nativeEagerShape(rhs);
  const batch = callOptions.batch ?? callOptions.batches ?? (lhsShape && lhsShape.length === 3
    ? lhsShape[0]
    : rhsShape && rhsShape.length === 3
      ? rhsShape[0]
      : null);
  const rows = callOptions.rows ?? callOptions.lhsRows ?? callOptions.lhs_rows ?? (lhsShape && lhsShape.length === 3 ? lhsShape[1] : null);
  const shared = callOptions.shared ?? callOptions.lhsCols ?? callOptions.lhs_cols ?? callOptions.rhsRows ?? callOptions.rhs_rows ?? (
    lhsShape && lhsShape.length === 3
      ? lhsShape[2]
      : rhsShape && rhsShape.length === 3
        ? rhsShape[1]
        : null
  );
  const cols = callOptions.cols ?? callOptions.rhsCols ?? callOptions.rhs_cols ?? (rhsShape && rhsShape.length === 3 ? rhsShape[2] : null);
  const shape = Object.freeze({
    batch: nativeEagerPositiveInteger(batch, `${label} batch`),
    rows: nativeEagerPositiveInteger(rows, `${label} rows`),
    shared: nativeEagerPositiveInteger(shared, `${label} shared`),
    cols: nativeEagerPositiveInteger(cols, `${label} cols`),
  });
  const expectedLhs = shape.batch * shape.rows * shape.shared;
  const expectedRhs = shape.batch * shape.shared * shape.cols;
  const expectedOutput = shape.batch * shape.rows * shape.cols;
  if (lhsData.length !== expectedLhs) {
    throw new Error(`${label} lhs length ${lhsData.length} does not match ${shape.batch}x${shape.rows}x${shape.shared}`);
  }
  if (rhsData.length !== expectedRhs) {
    throw new Error(`${label} rhs length ${rhsData.length} does not match ${shape.batch}x${shape.shared}x${shape.cols}`);
  }
  if (output.length < expectedOutput) {
    throw new Error(`${label} output length ${output.length} is smaller than ${expectedOutput}`);
  }
  return { lhsData, rhsData, output, expectedOutput, ...shape };
}

export function createAdapterNativeEagerSurface(options: NativeEagerSurfaceOptions) {
  const nativeEager = Object.freeze({
    linearInto(output: Float32Array, input: unknown, weights: unknown, callOptions: Record<string, unknown> = {}) {
      const args = nativeEagerLinearInputs("nativeEager.linearInto", output, input, weights, callOptions, options.f32);
      options.check(options.linearF32(args));
      return output;
    },
    linear_into(output: Float32Array, input: unknown, weights: unknown, callOptions?: Record<string, unknown>) {
      return this.linearInto(output, input, weights, callOptions);
    },
    linearActivationInto(output: Float32Array, input: unknown, weights: unknown, callOptions: Record<string, unknown> = {}) {
      const args = nativeEagerLinearInputs("nativeEager.linearActivationInto", output, input, weights, callOptions, options.f32);
      options.check(options.linearActivationF32({
        ...args,
        activation: nativeEagerActivationId(callOptions.activation, "nativeEager.linearActivationInto"),
      }));
      return output;
    },
    linear_activation_into(output: Float32Array, input: unknown, weights: unknown, callOptions?: Record<string, unknown>) {
      return this.linearActivationInto(output, input, weights, callOptions);
    },
    activationInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.activationF32 !== "function") {
        throw new Error("nativeEager.activationInto is unavailable in this runtime");
      }
      const args = nativeEagerActivationInputs(output, input, options.f32);
      options.check(options.activationF32({
        ...args,
        activation: nativeEagerActivationId(callOptions.activation, "nativeEager.activationInto"),
      }));
      return output;
    },
    activation_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.activationInto(output, input, callOptions);
    },
    elementwiseInto(output: Float32Array, lhs: unknown, rhs: unknown = null, callOptions: Record<string, unknown> = {}) {
      if (typeof options.elementwiseF32 !== "function") {
        throw new Error("nativeEager.elementwiseInto is unavailable in this runtime");
      }
      const args = nativeEagerElementwiseInputs(output, lhs, rhs, callOptions, options.f32);
      const op = nativeEagerElementwiseOpId(callOptions.op, "nativeEager.elementwiseInto");
      if (args.broadcastShape) {
        if (args.rhsData == null) {
          throw new Error("nativeEager.elementwiseInto row broadcast is unavailable in this runtime");
        }
        const broadcastCall = args.broadcastShape.side === "lhs"
          ? options.elementwiseBroadcastLhsF32
          : options.elementwiseBroadcastRhsF32;
        if (typeof broadcastCall !== "function") {
          throw new Error("nativeEager.elementwiseInto row broadcast is unavailable in this runtime");
        }
        options.check(broadcastCall({
          lhsData: args.lhsData,
          rhsData: args.rhsData,
          output: args.output,
          expectedOutput: args.expectedOutput,
          rows: args.broadcastShape.rows,
          cols: args.broadcastShape.cols,
          op,
        }));
        return output;
      }
      options.check(options.elementwiseF32({ ...args, op }));
      return output;
    },
    elementwise_into(output: Float32Array, lhs: unknown, rhs?: unknown, callOptions?: Record<string, unknown>) {
      return this.elementwiseInto(output, lhs, rhs, callOptions);
    },
    whereInto(output: Float32Array, condition: unknown, input: unknown, other: unknown) {
      if (typeof options.whereF32 !== "function") {
        throw new Error("nativeEager.whereInto is unavailable in this runtime");
      }
      const args = nativeEagerWhereInputs(output, condition, input, other, options.f32);
      options.check(options.whereF32(args));
      return output;
    },
    where_into(output: Float32Array, condition: unknown, input: unknown, other: unknown) {
      return this.whereInto(output, condition, input, other);
    },
    clampInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.clampF32 !== "function") {
        throw new Error("nativeEager.clampInto is unavailable in this runtime");
      }
      const args = nativeEagerClampInputs(output, input, callOptions, options.f32);
      options.check(options.clampF32(args));
      return output;
    },
    clamp_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.clampInto(output, input, callOptions);
    },
    reduceInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.reduceF32 !== "function") {
        throw new Error("nativeEager.reduceInto is unavailable in this runtime");
      }
      const args = nativeEagerReduceInputs(output, input, options.f32);
      options.check(options.reduceF32({
        ...args,
        op: nativeEagerReduceOpId(callOptions.op, "nativeEager.reduceInto"),
      }));
      return output;
    },
    reduce_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.reduceInto(output, input, callOptions);
    },
    reduceDimInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.reduceDimF32 !== "function") {
        throw new Error("nativeEager.reduceDimInto is unavailable in this runtime");
      }
      const args = nativeEagerReduceDimInputs(output, input, callOptions, options.f32);
      options.check(options.reduceDimF32({
        ...args,
        op: nativeEagerReduceOpId(callOptions.op, "nativeEager.reduceDimInto"),
      }));
      return output;
    },
    reduce_dim_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.reduceDimInto(output, input, callOptions);
    },
    argReduceDimInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.argReduceDimF32 !== "function") {
        throw new Error("nativeEager.argReduceDimInto is unavailable in this runtime");
      }
      const args = nativeEagerReduceDimInputs(output, input, callOptions, options.f32);
      options.check(options.argReduceDimF32({
        ...args,
        op: nativeEagerArgReduceOpId(callOptions.op, "nativeEager.argReduceDimInto"),
      }));
      return output;
    },
    arg_reduce_dim_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.argReduceDimInto(output, input, callOptions);
    },
    cumsumInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.cumsumF32 !== "function") {
        throw new Error("nativeEager.cumsumInto is unavailable in this runtime");
      }
      const args = nativeEagerCumsumInputs(output, input, callOptions, options.f32);
      options.check(options.cumsumF32(args));
      return output;
    },
    cumsum_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.cumsumInto(output, input, callOptions);
    },
    varianceInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.momentF32 !== "function") {
        throw new Error("nativeEager.varianceInto is unavailable in this runtime");
      }
      const args = nativeEagerMomentInputs("nativeEager.varianceInto", output, input, callOptions, options.f32, false);
      options.check(options.momentF32(args));
      return output;
    },
    variance_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.varianceInto(output, input, callOptions);
    },
    stdInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.momentF32 !== "function") {
        throw new Error("nativeEager.stdInto is unavailable in this runtime");
      }
      const args = nativeEagerMomentInputs("nativeEager.stdInto", output, input, callOptions, options.f32, true);
      options.check(options.momentF32(args));
      return output;
    },
    std_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.stdInto(output, input, callOptions);
    },
    dotInto(output: Float32Array, lhs: unknown, rhs: unknown) {
      if (typeof options.dotF32 !== "function") {
        throw new Error("nativeEager.dotInto is unavailable in this runtime");
      }
      const args = nativeEagerDotInputs(output, lhs, rhs, options.f32);
      options.check(options.dotF32(args));
      return output;
    },
    dot_into(output: Float32Array, lhs: unknown, rhs: unknown) {
      return this.dotInto(output, lhs, rhs);
    },
    permuteInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.permuteF32 !== "function") {
        throw new Error("nativeEager.permuteInto is unavailable in this runtime");
      }
      const args = nativeEagerPermuteInputs(output, input, callOptions, options.f32);
      options.check(options.permuteF32(args));
      return output;
    },
    permute_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.permuteInto(output, input, callOptions);
    },
    conv2dInto(output: Float32Array, input: unknown, weights: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.conv2dF32 !== "function") {
        throw new Error("nativeEager.conv2dInto is unavailable in this runtime");
      }
      const args = nativeEagerConv2dInputs(output, input, weights, callOptions, options.f32);
      options.check(options.conv2dF32(args));
      return output;
    },
    conv2d_into(output: Float32Array, input: unknown, weights: unknown, callOptions?: Record<string, unknown>) {
      return this.conv2dInto(output, input, weights, callOptions);
    },
    pool2dInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.pool2dF32 !== "function") {
        throw new Error("nativeEager.pool2dInto is unavailable in this runtime");
      }
      const args = nativeEagerPool2dInputs(output, input, callOptions, options.f32);
      options.check(options.pool2dF32(args));
      return output;
    },
    pool2d_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.pool2dInto(output, input, callOptions);
    },
    matmulInto(output: Float32Array, lhs: unknown, rhs: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.matmulF32 !== "function") {
        throw new Error("nativeEager.matmulInto is unavailable in this runtime");
      }
      const args = nativeEagerMatmulInputs(output, lhs, rhs, callOptions, options.f32);
      options.check(options.matmulF32(args));
      return output;
    },
    matmul_into(output: Float32Array, lhs: unknown, rhs: unknown, callOptions?: Record<string, unknown>) {
      return this.matmulInto(output, lhs, rhs, callOptions);
    },
    bmmInto(output: Float32Array, lhs: unknown, rhs: unknown, callOptions: Record<string, unknown> = {}) {
      if (typeof options.bmmF32 !== "function") {
        throw new Error("nativeEager.bmmInto is unavailable in this runtime");
      }
      const args = nativeEagerBmmInputs(output, lhs, rhs, callOptions, options.f32);
      options.check(options.bmmF32(args));
      return output;
    },
    bmm_into(output: Float32Array, lhs: unknown, rhs: unknown, callOptions?: Record<string, unknown>) {
      return this.bmmInto(output, lhs, rhs, callOptions);
    },
    softmaxInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      const args = nativeEagerSoftmaxInputs("nativeEager.softmaxInto", output, input, callOptions, options.f32);
      options.check(options.softmaxF32({ ...args, logSoftmax: false }));
      return output;
    },
    softmax_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.softmaxInto(output, input, callOptions);
    },
    logSoftmaxInto(output: Float32Array, input: unknown, callOptions: Record<string, unknown> = {}) {
      const args = nativeEagerSoftmaxInputs("nativeEager.logSoftmaxInto", output, input, callOptions, options.f32);
      options.check(options.softmaxF32({ ...args, logSoftmax: true }));
      return output;
    },
    log_softmax_into(output: Float32Array, input: unknown, callOptions?: Record<string, unknown>) {
      return this.logSoftmaxInto(output, input, callOptions);
    },
  });
  return { nativeEager };
}
