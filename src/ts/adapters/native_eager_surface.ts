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

type NativeEagerSurfaceOptions = {
  f32: NativeEagerTensorFactory;
  check: NativeEagerCheck;
  linearF32: NativeEagerLinearCall;
  linearActivationF32: NativeEagerLinearActivationCall;
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
  });
  return { nativeEager };
}
