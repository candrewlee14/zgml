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

type NativeEagerSoftmaxCall = (args: {
  inputData: Float32Array;
  output: Float32Array;
  expectedOutput: number;
  rows: number;
  cols: number;
  logSoftmax: boolean;
}) => number;

type NativeEagerSurfaceOptions = {
  f32: NativeEagerTensorFactory;
  check: NativeEagerCheck;
  linearF32: NativeEagerLinearCall;
  linearActivationF32: NativeEagerLinearActivationCall;
  matmulF32?: NativeEagerMatmulCall;
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

function nativeEagerLinearShape(input: unknown, weights: unknown, options: Record<string, unknown> = {}) {
  const inputShape = nativeEagerShape(input);
  const weightShape = nativeEagerShape(weights);
  const inFeatures = options.inFeatures ?? options.in_features ?? (weightShape && weightShape.length === 2 ? weightShape[0] : null);
  const outFeatures = options.outFeatures ?? options.out_features ?? (weightShape && weightShape.length === 2 ? weightShape[1] : null);
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
  });
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
