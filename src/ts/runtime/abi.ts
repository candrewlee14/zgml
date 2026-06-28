"use strict";

type NumericMap = Readonly<Record<string, number>>;
type StringMap = Readonly<Record<string, string>>;
type BigintMap = Readonly<Record<string, bigint>>;

export const tinyMlpActivationIds: NumericMap = Object.freeze({
  relu: 1,
  gelu: 2,
  silu: 3,
  sigmoid: 4,
});

export const moduleActivationIds: NumericMap = Object.freeze({
  ...tinyMlpActivationIds,
  exp: 5,
  log: 6,
  neg: 7,
  recip: 8,
  abs: 9,
  sqrt: 10,
  square: 11,
  sgn: 12,
  step: 13,
  tanh: 14,
});

export const moduleOpIds: NumericMap = Object.freeze({
  linear: 1,
  activation: 2,
  softmax: 3,
  layerNorm: 4,
  rmsNorm: 5,
  embedding: 6,
  logSoftmax: 7,
  reshape: 8,
  broadcastTo: 9,
  narrow: 10,
  transpose: 11,
  reduceSum: 12,
  reduceMean: 13,
  reduceMax: 14,
  reduceMin: 21,
  featureAffine: 22,
  diagonal: 23,
  reduceArgmax: 24,
  reduceArgmin: 25,
  reduceProd: 26,
  mul: 27,
  slice: 15,
  activationChain: 16,
  maxPool2d: 17,
  avgPool2d: 18,
  conv2d: 19,
  add: 20,
});

export const moduleFlags: NumericMap = Object.freeze({
  weight: 1 << 0,
  bias: 1 << 1,
});

export const modelKinds: NumericMap = Object.freeze({
  auto: 0,
  tinyLinear: 1,
  tinyLlama: 2,
  smollm135m: 3,
  tinyLlama2Layer: 4,
  tinyMlp: 5,
  module: 6,
});

export function modelKindName(kind: unknown): string {
  const numeric = Number(kind);
  switch (numeric) {
    case modelKinds.tinyLinear: return "tiny-linear";
    case modelKinds.tinyMlp: return "tiny-mlp";
    case modelKinds.tinyLlama: return "tiny-llama";
    case modelKinds.tinyLlama2Layer: return "tiny-llama-2layer";
    case modelKinds.smollm135m: return "smollm-135m";
    case modelKinds.module: return "module";
    default: return `unknown:${numeric}`;
  }
}

export const backendIds: NumericMap = Object.freeze({
  auto: 0,
  cpu: 1,
  metal: 2,
  webgpu: 3,
});

export const backendNames: StringMap = Object.freeze({
  0: "auto",
  1: "cpu",
  2: "metal",
  3: "webgpu",
});

const backendNamesById: Readonly<Record<number, string>> = Object.freeze({
  0: "auto",
  1: "cpu",
  2: "metal",
  3: "webgpu",
});

export const bufferStorageIds: NumericMap = Object.freeze({
  none: 0,
  host: 1,
  externalResource: 2,
});

export const resourceAccessIds: NumericMap = Object.freeze({
  read: 1,
  write: 2,
  readwrite: 1 | 2,
});

export const programBufferKinds: NumericMap = Object.freeze({
  weights: 1,
  bias: 2,
  input: 3,
  output: 4,
  "kv-k": 5,
  "kv-v": 6,
});

export const abiStructKinds: NumericMap = Object.freeze({
  runtimeInfo: 1,
  modelDesc: 2,
  modelLoadDesc: 3,
  modelInspection: 4,
  programModelCompatibility: 5,
  sessionInspection: 6,
  compileDesc: 7,
  bindDesc: 8,
  bufferDesc: 9,
  bufferInspection: 10,
  externalResourceDesc: 11,
  deviceBufferImportDesc: 12,
  bufferBindDesc: 13,
  llamaKvCacheBindDesc: 14,
  llamaBufferBindDesc: 15,
  stepDesc: 16,
  stepResult: 17,
  tokenStepDesc: 18,
  tokenAdvanceDesc: 19,
  tokenAdvanceTokensDesc: 20,
  tokenPrefillDesc: 21,
  tokenExecuteDesc: 22,
  tokenArgmaxDesc: 23,
  tokenArgmaxResult: 24,
  tokenExecuteArgmaxDesc: 25,
  tokenGenerateArgmaxDesc: 26,
  tokenGenerateArgmaxResult: 27,
  tokenSampleDesc: 28,
  tokenSampleResult: 29,
  tokenExecuteSampleDesc: 30,
  tokenGenerateSampleDesc: 31,
  tokenGenerateSampleResult: 32,
  programRequirements: 33,
  llamaKvCacheRequirements: 34,
  programInspection: 35,
  llamaProgramInspection: 36,
  runtimeProfile: 37,
  safetensorsHeaderProbeDesc: 38,
  safetensorsDataLoadDesc: 39,
  moduleOpDesc: 40,
  moduleDesc: 41,
});

export const runtimeFeatureBits = Object.freeze({
  bufferHandle: 1n << 0n,
  modelAuto: 1n << 1n,
  runtimeProfile: 1n << 2n,
  webGpuCompileOnly: 1n << 3n,
  wasmExports: 1n << 4n,
  nativeBufferIo: 1n << 5n,
  nativeArgmax: 1n << 6n,
  nativeExecuteArgmax: 1n << 7n,
  nativeTopKSample: 1n << 8n,
  programRequirements: 1n << 9n,
  programOutputBuffer: 1n << 10n,
  nativeGenerateSample: 1n << 11n,
  nativeGenerateArgmax: 1n << 12n,
  sessionModelBinding: 1n << 13n,
  externalBuffer: 1n << 14n,
  externalResourceBuffer: 1n << 15n,
  llamaKvResourceBinding: 1n << 16n,
  llamaKvCacheRequirements: 1n << 17n,
  programBufferFactory: 1n << 18n,
  externalResourceAccess: 1n << 19n,
  modelInspection: 1n << 20n,
  programModelCompatibility: 1n << 21n,
  bufferInspection: 1n << 22n,
  sessionInspection: 1n << 23n,
  programResourceInspection: 1n << 24n,
  programMemoryInspection: 1n << 25n,
  programShapeInspection: 1n << 26n,
  programPatchEnvelopeInspection: 1n << 27n,
  sessionBindingShapeInspection: 1n << 28n,
  nativeWgpuExecution: 1n << 29n,
  programDeviceBuffer: 1n << 30n,
  programDeviceBufferImport: 1n << 31n,
  programDispatchPlanInspection: 1n << 32n,
  abiStructSize: 1n << 33n,
  experimentalLlamaWgpuExecution: 1n << 34n,
  modelPathProbe: 1n << 35n,
  supportedCheckpoints: 1n << 36n,
  safetensorsHeaderProbe: 1n << 37n,
  safetensorsDataLoad: 1n << 38n,
  safetensorsDataProbe: 1n << 39n,
  nativeTinyMlp: 1n << 40n,
  nativeModuleProgram: 1n << 41n,
  programBindingRequirements: 1n << 42n,
  sessionPersistentUpload: 1n << 43n,
  nativeModuleActivationChain: 1n << 44n,
  nativeEagerLinear: 1n << 45n,
  nativeEagerLinearActivation: 1n << 46n,
  nativeTrainingStep: 1n << 47n,
  nativeEagerSoftmax: 1n << 48n,
  nativeEagerMatmul: 1n << 49n,
  nativeEagerActivation: 1n << 50n,
  nativeEagerElementwise: 1n << 51n,
  nativeEagerReduce: 1n << 52n,
  nativeEagerConv2d: 1n << 53n,
  nativeEagerPool2d: 1n << 54n,
  nativeEagerDot: 1n << 55n,
});
export type RuntimeFeatureName = keyof typeof runtimeFeatureBits;
export type RuntimeFeatureMap = Readonly<Record<RuntimeFeatureName, boolean>>;

export const requiredRuntimeFeatureNames = Object.freeze([
  "bufferHandle",
  "modelAuto",
  "runtimeProfile",
  "webGpuCompileOnly",
  "wasmExports",
  "nativeBufferIo",
  "nativeArgmax",
  "nativeExecuteArgmax",
  "nativeTopKSample",
  "programRequirements",
  "programOutputBuffer",
  "nativeGenerateSample",
  "nativeGenerateArgmax",
  "sessionModelBinding",
  "externalBuffer",
  "externalResourceBuffer",
  "llamaKvResourceBinding",
  "llamaKvCacheRequirements",
  "programBufferFactory",
  "externalResourceAccess",
  "modelInspection",
  "programModelCompatibility",
  "bufferInspection",
  "sessionInspection",
  "programResourceInspection",
  "programMemoryInspection",
  "programShapeInspection",
  "programPatchEnvelopeInspection",
  "sessionBindingShapeInspection",
  "programDeviceBuffer",
  "programDeviceBufferImport",
  "programDispatchPlanInspection",
  "abiStructSize",
  "modelPathProbe",
  "supportedCheckpoints",
  "safetensorsHeaderProbe",
  "safetensorsDataLoad",
  "safetensorsDataProbe",
  "nativeTinyMlp",
  "nativeModuleProgram",
  "programBindingRequirements",
  "sessionPersistentUpload",
  "nativeModuleActivationChain",
  "nativeEagerLinear",
  "nativeEagerLinearActivation",
  "nativeTrainingStep",
  "nativeEagerSoftmax",
  "nativeEagerMatmul",
  "nativeEagerActivation",
  "nativeEagerElementwise",
  "nativeEagerReduce",
  "nativeEagerConv2d",
  "nativeEagerPool2d",
  "nativeEagerDot",
]);

export const requiredRuntimeFeatureMask = requiredRuntimeFeatureNames.reduce(
  (mask, name) => mask | runtimeFeatureBits[name as RuntimeFeatureName],
  0n,
);

export const expectedRuntimeAbiVersion = 6;
export const expectedRuntimeTokenIdBytes = 4;

export function decodeRuntimeFeatures(featureFlags: bigint): RuntimeFeatureMap {
  const features: Record<RuntimeFeatureName, boolean> = {} as Record<RuntimeFeatureName, boolean>;
  for (const [name, bit] of Object.entries(runtimeFeatureBits)) {
    features[name as RuntimeFeatureName] = (featureFlags & bit) !== 0n;
  }
  return Object.freeze(features);
}

export function assertCompatibleRuntimeInfo<T extends { abiVersion: number; tokenIdBytes: number; featureFlags: bigint }>(info: T): T {
  if (info.abiVersion !== expectedRuntimeAbiVersion) {
    throw new Error(`unsupported zgml C ABI version ${info.abiVersion}; expected ${expectedRuntimeAbiVersion}`);
  }
  if (info.tokenIdBytes !== expectedRuntimeTokenIdBytes) {
    throw new Error(`unsupported zgml token id width ${info.tokenIdBytes}; expected ${expectedRuntimeTokenIdBytes}`);
  }
  if ((info.featureFlags & requiredRuntimeFeatureMask) !== requiredRuntimeFeatureMask) {
    throw new Error(`zgml C ABI is missing required feature flags: ${info.featureFlags}`);
  }
  return info;
}

export type RuntimeAbiInfoInput = Readonly<{
  abiVersion: unknown;
  sizeTBytes: unknown;
  pointerBytes: unknown;
  tokenIdBytes: unknown;
  featureFlags: unknown;
}>;

export type RuntimeAbiFacadeHelpersOptions = Readonly<{
  readRuntimeInfo?: () => RuntimeAbiInfoInput;
  readAbiStructSize?: (kind: number) => unknown;
}>;

export function createRuntimeAbiFacadeHelpers(options: RuntimeAbiFacadeHelpersOptions) {
  const readRuntimeInfo = options && options.readRuntimeInfo;
  const readAbiStructSize = options && options.readAbiStructSize;
  if (typeof readRuntimeInfo !== "function") {
    throw new Error("createRuntimeAbiFacadeHelpers requires readRuntimeInfo");
  }
  if (typeof readAbiStructSize !== "function") {
    throw new Error("createRuntimeAbiFacadeHelpers requires readAbiStructSize");
  }
  const readRuntimeInfoFn = readRuntimeInfo;
  const readAbiStructSizeFn = readAbiStructSize;

  function runtimeInfo() {
    const info = readRuntimeInfoFn();
    const featureFlags = BigInt(info.featureFlags as any);
    return Object.freeze({
      abiVersion: Number(info.abiVersion),
      sizeTBytes: Number(info.sizeTBytes),
      pointerBytes: Number(info.pointerBytes),
      tokenIdBytes: Number(info.tokenIdBytes),
      featureFlags,
      features: decodeRuntimeFeatures(featureFlags),
    });
  }

  function abiStructSize(kind: string | number): number {
    const size = Number(readAbiStructSizeFn(abiStructKindId(kind)));
    if (size === 0) {
      throw new Error(`unknown zgml ABI struct kind: ${String(kind)}`);
    }
    return size;
  }

  function abiStructSizes() {
    const out: Record<string, number> = {};
    for (const [name, id] of Object.entries(abiStructKinds)) {
      out[name] = Number(readAbiStructSizeFn(id));
    }
    return Object.freeze(out);
  }

  function assertCompatibleRuntime() {
    return assertCompatibleRuntimeInfo(runtimeInfo());
  }

  return Object.freeze({
    runtimeInfo,
    abiStructSize,
    abiStructSizes,
    assertCompatibleRuntime,
  });
}

export function backendId(name: string): number {
  const id = backendIds[name];
  if (id === undefined) throw new Error(`unknown zgml backend: ${name}`);
  return id;
}

export function backendName(id: unknown): string {
  const numeric = Number(id);
  return backendNamesById[numeric] || `unknown:${numeric}`;
}

export function bufferStorageName(storage: unknown): string {
  const numeric = Number(storage);
  switch (numeric) {
    case bufferStorageIds.none: return "none";
    case bufferStorageIds.host: return "host";
    case bufferStorageIds.externalResource: return "external-resource";
    default: return `unknown:${numeric}`;
  }
}

export function resourceAccessFlags(access: any = "readwrite"): number {
  if (access === "read" || access === "readonly" || access === "read-only") return resourceAccessIds.read;
  if (access === "write" || access === "writeonly" || access === "write-only") return resourceAccessIds.write;
  if (access === "readwrite" || access === "read-write" || access === "readWrite") return resourceAccessIds.readwrite;
  if (Array.isArray(access)) {
    let flags = 0;
    for (const part of access) flags |= resourceAccessFlags(part);
    if (flags === 0 || (flags & ~resourceAccessIds.readwrite) !== 0) {
      throw new Error(`invalid external resource access: ${access}`);
    }
    return flags;
  }
  if (access && typeof access === "object") {
    let flags = 0;
    if (access.read) flags |= resourceAccessIds.read;
    if (access.write) flags |= resourceAccessIds.write;
    if (flags === 0) throw new Error("external resource access must include read or write");
    return flags;
  }
  throw new Error(`invalid external resource access: ${access}`);
}

export function accessFlagsToObject(flags: unknown) {
  const numeric = Number(flags);
  return Object.freeze({
    read: (numeric & resourceAccessIds.read) !== 0,
    write: (numeric & resourceAccessIds.write) !== 0,
  });
}

export function programBufferKindId(kind: string): number {
  const id = programBufferKinds[kind];
  if (id === undefined) throw new Error(`unknown Program buffer kind: ${kind}`);
  return id;
}

export function abiStructKindId(kind: string | number): number {
  const id = typeof kind === "string" ? abiStructKinds[kind] : kind;
  if (!Number.isInteger(id) || id <= 0) {
    throw new Error(`unknown zgml ABI struct kind: ${String(kind)}`);
  }
  return id;
}

function compileEnvelopeValue(name: string, value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative safe integer, got ${value}`);
  }
  return value as number;
}

export function normalizeCompileOptions(options: { backend?: string; contextLength?: unknown; batch?: unknown } = {}) {
  const backend = options.backend || "auto";
  return Object.freeze({
    backend,
    backendId: backendId(backend),
    contextLength: compileEnvelopeValue("contextLength", options.contextLength ?? 0),
    batch: compileEnvelopeValue("batch", options.batch ?? 0),
  });
}

export type NormalizedCompileOptions = Readonly<{
  backend: string;
  backendId: number;
  contextLength: number;
  batch: number;
}>;

export function isDefaultCompileEnvelope(options: NormalizedCompileOptions): boolean {
  return options.backendId === backendIds.auto && options.contextLength === 0 && options.batch === 0;
}

export type CompileDescFields = Readonly<{
  backend: number;
  reserved: 0;
  contextLen: number;
  batch: number;
}>;

export type CompileDescRecord = Readonly<{
  backend: number;
  reserved: 0;
  context_len: number;
  batch: number;
}>;

export function compileDescFieldsFromOptions(options: NormalizedCompileOptions): CompileDescFields {
  return Object.freeze({
    backend: options.backendId,
    reserved: 0,
    contextLen: options.contextLength,
    batch: options.batch,
  });
}

export function compileDescRecordFromFields(fields: CompileDescFields): CompileDescRecord {
  return Object.freeze({
    backend: fields.backend,
    reserved: fields.reserved,
    context_len: fields.contextLen,
    batch: fields.batch,
  });
}

export function compileDescRecordFromOptions(options: NormalizedCompileOptions): CompileDescRecord {
  return compileDescRecordFromFields(compileDescFieldsFromOptions(options));
}
