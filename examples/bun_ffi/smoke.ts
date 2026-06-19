import { NativeBuffer, Program, Session, SmolLM135M, Tensor, TinyLinear, TinyLinearProgram, TinyLinearSession, TinyLlama, abiStructSize, abiStructSizes, arange, cat, checkpoint, full, linspace, loadedRuntimeInfo, loadModel, loadSafetensorsData, loss, nn, ones, optim, param, parameter, probeModel, probeSafetensorsData, probeSafetensorsHeader, rand, randn, runtimeInfo, scalar, stack, supportedCheckpointModels, tensor, train, webgpuInterop, zeros, type ProgramExecutionCapabilities, type ProgramInspection } from "zgml/bun";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TinyLlamaSessionVocabPlusSentinel = 9;
const SmolLM135MVocabPlusSentinel = 49153;
const featureBufferHandle = 1n << 0n;
const featureModelAuto = 1n << 1n;
const featureRuntimeProfile = 1n << 2n;
const featureWebGpuCompileOnly = 1n << 3n;
const featureWasmExports = 1n << 4n;
const featureNativeBufferIo = 1n << 5n;
const featureNativeArgmax = 1n << 6n;
const featureNativeExecuteArgmax = 1n << 7n;
const featureNativeTopKSample = 1n << 8n;
const featureProgramRequirements = 1n << 9n;
const featureProgramOutputBuffer = 1n << 10n;
const featureNativeGenerateSample = 1n << 11n;
const featureNativeGenerateArgmax = 1n << 12n;
const featureSessionModelBinding = 1n << 13n;
const featureExternalBuffer = 1n << 14n;
const featureExternalResourceBuffer = 1n << 15n;
const featureLlamaKvResourceBinding = 1n << 16n;
const featureLlamaKvCacheRequirements = 1n << 17n;
const featureProgramBufferFactory = 1n << 18n;
const featureExternalResourceAccess = 1n << 19n;
const featureModelInspection = 1n << 20n;
const featureProgramModelCompatibility = 1n << 21n;
const featureBufferInspection = 1n << 22n;
const featureSessionInspection = 1n << 23n;
const featureProgramResourceInspection = 1n << 24n;
const featureProgramMemoryInspection = 1n << 25n;
const featureProgramShapeInspection = 1n << 26n;
const featureProgramPatchEnvelopeInspection = 1n << 27n;
const featureSessionBindingShapeInspection = 1n << 28n;
const featureNativeWgpuExecution = 1n << 29n;
const featureProgramDeviceBuffer = 1n << 30n;
const featureProgramDeviceBufferImport = 1n << 31n;
const featureProgramDispatchPlanInspection = 1n << 32n;
const featureAbiStructSize = 1n << 33n;
const featureExperimentalLlamaWgpuExecution = 1n << 34n;
const featureModelPathProbe = 1n << 35n;
const featureSupportedCheckpoints = 1n << 36n;
const featureSafetensorsHeaderProbe = 1n << 37n;
const featureSafetensorsDataLoad = 1n << 38n;
const featureSafetensorsDataProbe = 1n << 39n;
const featureNativeTinyMlp = 1n << 40n;
const featureNativeModuleProgram = 1n << 41n;
const featureProgramBindingRequirements = 1n << 42n;
const featureSessionPersistentUpload = 1n << 43n;
const u32Max = 0xffffffff;

function expectClose(actual: Float32Array, expected: readonly number[]): void {
  if (actual.length !== expected.length) {
    throw new Error(`expected ${expected.length} outputs, got ${actual.length}`);
  }
  for (let i = 0; i < actual.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > 1e-5) {
      throw new Error(`output[${i}] expected ${expected[i]}, got ${actual[i]}`);
    }
  }
}

function expectCloseWithin(actual: Float32Array, expected: readonly number[], tolerance: number): void {
  if (actual.length !== expected.length) {
    throw new Error(`expected ${expected.length} outputs, got ${actual.length}`);
  }
  for (let i = 0; i < actual.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > tolerance) {
      throw new Error(`output[${i}] expected ${expected[i]}, got ${actual[i]}`);
    }
  }
}

function json(value: unknown): string {
  return JSON.stringify(value, (_, v) => (typeof v === "bigint" ? v.toString() : v));
}

function commandCategoryTotal(inspection: ProgramInspection): number {
  return (
    inspection.commandOpCount +
    inspection.commandRowCount +
    inspection.commandProjectionCount +
    inspection.commandAttentionCount +
    inspection.commandMovementCount +
    inspection.commandElementwiseCount +
    inspection.commandRopeCount
  );
}

function expectCapabilityDispatchPlan(capabilities: ProgramExecutionCapabilities, inspection: ProgramInspection, label: string): void {
  const expectedFull =
    inspection.dispatchPlanSupported &&
    inspection.dispatchPlanCoveredOpCount === inspection.opCount &&
    inspection.dispatchPlanFirstUnsupportedOp === null;
  const expectedMode = capabilities.canExecute
    ? "executable"
    : (capabilities.canBindExternalResources ? "resource-probe" : "compile-only");
  if (
    capabilities.mode !== expectedMode ||
    capabilities.hasFullDispatchPlan !== expectedFull ||
    capabilities.backendDispatchCount !== inspection.backendDispatchCount ||
    capabilities.dispatchPlanSupported !== inspection.dispatchPlanSupported ||
    capabilities.dispatchPlanCoveredOpCount !== inspection.dispatchPlanCoveredOpCount ||
    capabilities.dispatchPlanFirstUnsupportedOp !== inspection.dispatchPlanFirstUnsupportedOp ||
    capabilities.dispatchPlanProjectionCount !== inspection.dispatchPlanProjectionCount ||
    capabilities.dispatchPlanRowCount !== inspection.dispatchPlanRowCount ||
    capabilities.dispatchPlanAttentionCount !== inspection.dispatchPlanAttentionCount ||
    capabilities.dispatchPlanMovementCount !== inspection.dispatchPlanMovementCount ||
    capabilities.dispatchPlanElementwiseCount !== inspection.dispatchPlanElementwiseCount ||
    capabilities.dispatchPlanRopeCount !== inspection.dispatchPlanRopeCount ||
    capabilities.dispatchPlanQuantizedProjectionCount !== inspection.dispatchPlanQuantizedProjectionCount ||
    !Object.isFrozen(inspection) ||
    !Object.isFrozen(capabilities) ||
    !Array.isArray(inspection.diagnostics) ||
    !Array.isArray(capabilities.diagnostics) ||
    !Object.isFrozen(inspection.diagnostics) ||
    !Object.isFrozen(capabilities.diagnostics) ||
    inspection.diagnostics.some((diagnostic) => !Object.isFrozen(diagnostic)) ||
    capabilities.diagnostics.some((diagnostic) => !Object.isFrozen(diagnostic)) ||
    capabilities.diagnostics.map((diagnostic) => diagnostic.code).join("|") !== inspection.diagnostics.map((diagnostic) => diagnostic.code).join("|") ||
    ((!capabilities.canExecute || !expectedFull) && capabilities.diagnostics.length === 0)
  ) {
    throw new Error(`unexpected ${label} capability dispatch plan: ${json({ capabilities, inspection })}`);
  }
}

function smollmSafetensorsHeader(): string {
  const tensors: Record<string, { dtype: "F16"; shape: number[]; data_offsets: [number, number] }> = {};
  const add = (name: string, shape: number[]): void => {
    tensors[name] = { dtype: "F16", shape, data_offsets: [0, 0] };
  };
  add("model.embed_tokens.weight", [49152, 576]);
  add("model.norm.weight", [576]);
  for (const layer of [0, 29]) {
    add(`model.layers.${layer}.self_attn.q_proj.weight`, [576, 576]);
    add(`model.layers.${layer}.self_attn.k_proj.weight`, [192, 576]);
    add(`model.layers.${layer}.self_attn.v_proj.weight`, [192, 576]);
    add(`model.layers.${layer}.self_attn.o_proj.weight`, [576, 576]);
    add(`model.layers.${layer}.mlp.gate_proj.weight`, [1536, 576]);
    add(`model.layers.${layer}.mlp.up_proj.weight`, [1536, 576]);
    add(`model.layers.${layer}.mlp.down_proj.weight`, [576, 1536]);
    add(`model.layers.${layer}.input_layernorm.weight`, [576]);
    add(`model.layers.${layer}.post_attention_layernorm.weight`, [576]);
  }
  return JSON.stringify(tensors);
}

const tinyLlamaTensorSpecs: Array<[string, number[]]> = [
  ["model.embed_tokens.weight", [8, 4]],
  ["model.norm.weight", [4]],
  ["lm_head.weight", [8, 4]],
  ["model.layers.0.self_attn.q_proj.weight", [4, 4]],
  ["model.layers.0.self_attn.k_proj.weight", [4, 4]],
  ["model.layers.0.self_attn.v_proj.weight", [4, 4]],
  ["model.layers.0.self_attn.o_proj.weight", [4, 4]],
  ["model.layers.0.mlp.gate_proj.weight", [8, 4]],
  ["model.layers.0.mlp.up_proj.weight", [8, 4]],
  ["model.layers.0.mlp.down_proj.weight", [4, 8]],
  ["model.layers.0.input_layernorm.weight", [4]],
  ["model.layers.0.post_attention_layernorm.weight", [4]],
];

function elementCount(shape: readonly number[]): number {
  return shape.reduce((n, dim) => n * dim, 1);
}

function tinyLlamaSafetensorsHeader(): string {
  const tensors: Record<string, { dtype: "F16"; shape: number[]; data_offsets: [number, number] }> = {};
  for (const [name, shape] of tinyLlamaTensorSpecs) {
    tensors[name] = { dtype: "F16", shape, data_offsets: [0, 0] };
  }
  return JSON.stringify(tensors);
}

function writeTinySafetensorsFile(path: string, header: string): void {
  const headerBytes = new TextEncoder().encode(header);
  const len = new Uint8Array(8);
  new DataView(len.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  const bytes = new Uint8Array(len.length + headerBytes.length);
  bytes.set(len, 0);
  bytes.set(headerBytes, len.length);
  writeFileSync(path, bytes);
}

function safetensorsHeaderFileBytes(header: string): Uint8Array {
  const headerBytes = new TextEncoder().encode(header);
  const bytes = new Uint8Array(8 + headerBytes.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  bytes.set(headerBytes, 8);
  return bytes;
}

function tinyLlamaSafetensorsFileBytes(): Uint8Array {
  const tensors: Record<string, { dtype: "F32"; shape: number[]; data_offsets: [number, number] }> = {};
  let offset = 0;
  for (const [name, shape] of tinyLlamaTensorSpecs) {
    const byteLength = elementCount(shape) * Float32Array.BYTES_PER_ELEMENT;
    tensors[name] = { dtype: "F32", shape, data_offsets: [offset, offset + byteLength] };
    offset += byteLength;
  }
  let header = JSON.stringify(tensors);
  while ((8 + new TextEncoder().encode(header).length) % Float32Array.BYTES_PER_ELEMENT !== 0) {
    header += " ";
  }
  const headerBytes = new TextEncoder().encode(header);
  const len = new Uint8Array(8);
  new DataView(len.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  const bytes = new Uint8Array(len.length + headerBytes.length + offset);
  bytes.set(len, 0);
  bytes.set(headerBytes, len.length);
  return bytes;
}

function writeTinyLlamaSafetensorsFile(path: string): void {
  writeFileSync(path, tinyLlamaSafetensorsFileBytes());
}

type TinyCheckpointModel = {
  inspect(): {
    modelKind: string;
    vocabSize: number;
    dModel: number;
    nLayers: number;
    nKvHeads: number;
    tiedLmHead: boolean;
  };
  compile(options?: { backend?: "cpu"; contextLength?: number }): {
    bind(): {
      step(token: number, output?: Float32Array): Float32Array;
      free(): void;
    };
    free(): void;
  };
  free(): void;
};

function expectTinyLlamaCheckpointInspection(model: TinyCheckpointModel, label: string): void {
  const inspection = model.inspect();
  if (
    !Object.isFrozen(inspection) ||
    inspection.modelKind !== "tiny-llama" ||
    inspection.vocabSize !== 8 ||
    inspection.dModel !== 4 ||
    inspection.nLayers !== 1 ||
    inspection.nKvHeads !== 1 ||
    inspection.tiedLmHead
  ) {
    throw new Error(`unexpected ${label} model inspection: ${json(inspection)}`);
  }
}

function expectTinyLlamaCheckpointStep(model: TinyCheckpointModel, label: string): void {
  let program: ReturnType<TinyCheckpointModel["compile"]> | null = null;
  let session: ReturnType<ReturnType<TinyCheckpointModel["compile"]>["bind"]> | null = null;
  try {
    program = model.compile({ backend: "cpu", contextLength: 4 });
    session = program.bind();
    const out = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-123);
    const logits = session.step(0, out);
    if (logits.length !== TinyLlamaSessionVocabPlusSentinel - 1 || out[TinyLlamaSessionVocabPlusSentinel - 1] !== -123) {
      throw new Error(`unexpected ${label} checkpoint step length or sentinel`);
    }
    for (const value of logits) {
      if (Math.abs(value) > 1e-6) {
        throw new Error(`expected zero ${label} checkpoint logits, got ${json(Array.from(logits))}`);
      }
    }
  } finally {
    session?.free();
    program?.free();
  }
}

function isShapeMismatchError(err: unknown): boolean {
  if (typeof err === "object" && err !== null) {
    const maybe = err as { status?: unknown; code?: unknown; message?: unknown };
    if (maybe.status === "shape_mismatch" || maybe.code === 3 || String(maybe.message ?? "").includes("shape_mismatch")) {
      return true;
    }
  }
  return String(err).includes("shape_mismatch");
}

function expectRuntimeProfileStable(before: Record<string, unknown>, after: Record<string, unknown>, label: string): void {
  const fields = [
    "callCount",
    "backendOpCount",
    "fallbackOpCount",
    "backendDispatchCount",
    "syncCount",
    "runtimePatchCallCount",
    "runtimePatchChangedCount",
    "runtimePatchInvalidCount",
    "runtimePatchHoles",
    "runtimePatchCacheWritePosHoles",
    "runtimePatchAttentionSeqKvHoles",
    "runtimePatchStencilHash",
    "commandCount",
    "commandStencilHash",
    "commandOpCount",
    "commandRowCount",
    "commandProjectionCount",
    "commandAttentionCount",
    "commandMovementCount",
    "commandElementwiseCount",
    "commandRopeCount",
  ] as const;
  for (const field of fields) {
    if (before[field] !== after[field]) {
      throw new Error(`unexpected ${label} profile delta for ${field}: ${json({ before, after })}`);
    }
  }
}

function argmax(values: Float32Array): { token: number; logit: number } {
  let token = 0;
  let logit = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > logit) {
      token = i;
      logit = values[i];
    }
  }
  return { token, logit };
}

function isUnsupportedError(err: unknown): boolean {
  if (typeof err === "object" && err !== null) {
    const maybe = err as { status?: unknown; code?: unknown };
    if (maybe.status === "unsupported" || maybe.code === 5) return true;
  }
  return String(err).includes("unsupported");
}

function isImportRejectedError(err: unknown): boolean {
  if (typeof err === "object" && err !== null) {
    const maybe = err as { status?: unknown; code?: unknown };
    if (maybe.status === "shape_mismatch" || maybe.status === "unsupported" || maybe.code === 3 || maybe.code === 5) {
      return true;
    }
  }
  const message = String(err);
  return message.includes("shape_mismatch") || message.includes("unsupported");
}

const info = runtimeInfo();
if (
  info.abiVersion !== 6 ||
  info.tokenIdBytes !== 4 ||
  !Object.isFrozen(info) ||
  !Object.isFrozen(info.features) ||
  !Object.isFrozen(loadedRuntimeInfo) ||
  !Object.isFrozen(loadedRuntimeInfo.features) ||
  (info.featureFlags & featureBufferHandle) === 0n ||
  (info.featureFlags & featureModelAuto) === 0n ||
  (info.featureFlags & featureRuntimeProfile) === 0n ||
  (info.featureFlags & featureWebGpuCompileOnly) === 0n ||
  (info.featureFlags & featureWasmExports) === 0n ||
  (info.featureFlags & featureNativeBufferIo) === 0n ||
  (info.featureFlags & featureNativeArgmax) === 0n ||
  (info.featureFlags & featureNativeExecuteArgmax) === 0n ||
  (info.featureFlags & featureNativeTopKSample) === 0n ||
  (info.featureFlags & featureProgramRequirements) === 0n ||
  (info.featureFlags & featureProgramOutputBuffer) === 0n ||
  (info.featureFlags & featureNativeGenerateSample) === 0n ||
  (info.featureFlags & featureNativeGenerateArgmax) === 0n ||
  (info.featureFlags & featureSessionModelBinding) === 0n ||
  (info.featureFlags & featureExternalBuffer) === 0n ||
  (info.featureFlags & featureExternalResourceBuffer) === 0n ||
  (info.featureFlags & featureLlamaKvResourceBinding) === 0n ||
  (info.featureFlags & featureLlamaKvCacheRequirements) === 0n ||
  (info.featureFlags & featureProgramBufferFactory) === 0n ||
  (info.featureFlags & featureExternalResourceAccess) === 0n ||
  (info.featureFlags & featureModelInspection) === 0n ||
  (info.featureFlags & featureProgramModelCompatibility) === 0n ||
  (info.featureFlags & featureBufferInspection) === 0n ||
  (info.featureFlags & featureSessionInspection) === 0n ||
  (info.featureFlags & featureProgramResourceInspection) === 0n ||
  (info.featureFlags & featureProgramMemoryInspection) === 0n ||
  (info.featureFlags & featureProgramShapeInspection) === 0n ||
  (info.featureFlags & featureProgramPatchEnvelopeInspection) === 0n ||
  (info.featureFlags & featureSessionBindingShapeInspection) === 0n ||
  (info.featureFlags & featureProgramDeviceBuffer) === 0n ||
  (info.featureFlags & featureProgramDeviceBufferImport) === 0n ||
  (info.featureFlags & featureProgramDispatchPlanInspection) === 0n ||
  (info.featureFlags & featureAbiStructSize) === 0n ||
  (info.featureFlags & featureModelPathProbe) === 0n ||
  (info.featureFlags & featureSupportedCheckpoints) === 0n ||
  (info.featureFlags & featureSafetensorsHeaderProbe) === 0n ||
  (info.featureFlags & featureSafetensorsDataLoad) === 0n ||
  (info.featureFlags & featureSafetensorsDataProbe) === 0n ||
  (info.featureFlags & featureNativeTinyMlp) === 0n ||
  (info.featureFlags & featureNativeModuleProgram) === 0n ||
  (info.featureFlags & featureProgramBindingRequirements) === 0n ||
  (info.featureFlags & featureSessionPersistentUpload) === 0n ||
  !info.features.modelAuto ||
  !info.features.webGpuCompileOnly ||
  !info.features.wasmExports ||
  !info.features.programDispatchPlanInspection ||
  !info.features.abiStructSize ||
  !info.features.modelPathProbe ||
  !info.features.supportedCheckpoints ||
  !info.features.safetensorsHeaderProbe ||
  !info.features.safetensorsDataLoad ||
  !info.features.safetensorsDataProbe ||
  !info.features.nativeTinyMlp ||
  !info.features.nativeModuleProgram ||
  !info.features.programBindingRequirements ||
  !info.features.sessionPersistentUpload ||
  !loadedRuntimeInfo.features.modelAuto ||
  !loadedRuntimeInfo.features.webGpuCompileOnly ||
  !loadedRuntimeInfo.features.wasmExports ||
  !loadedRuntimeInfo.features.programDispatchPlanInspection ||
  !loadedRuntimeInfo.features.abiStructSize ||
  !loadedRuntimeInfo.features.modelPathProbe ||
  !loadedRuntimeInfo.features.supportedCheckpoints ||
  !loadedRuntimeInfo.features.safetensorsHeaderProbe ||
  !loadedRuntimeInfo.features.safetensorsDataLoad ||
  !loadedRuntimeInfo.features.safetensorsDataProbe ||
  !loadedRuntimeInfo.features.nativeTinyMlp ||
  !loadedRuntimeInfo.features.nativeModuleProgram ||
  !loadedRuntimeInfo.features.programBindingRequirements ||
  !loadedRuntimeInfo.features.sessionPersistentUpload ||
  info.features.nativeWgpuExecution !== ((info.featureFlags & featureNativeWgpuExecution) !== 0n) ||
  info.features.experimentalLlamaWgpuExecution !== ((info.featureFlags & featureExperimentalLlamaWgpuExecution) !== 0n) ||
  info.features.modelPathProbe !== ((info.featureFlags & featureModelPathProbe) !== 0n) ||
  info.features.supportedCheckpoints !== ((info.featureFlags & featureSupportedCheckpoints) !== 0n) ||
  info.features.safetensorsHeaderProbe !== ((info.featureFlags & featureSafetensorsHeaderProbe) !== 0n) ||
  info.features.safetensorsDataLoad !== ((info.featureFlags & featureSafetensorsDataLoad) !== 0n) ||
  info.features.safetensorsDataProbe !== ((info.featureFlags & featureSafetensorsDataProbe) !== 0n) ||
  info.features.nativeTinyMlp !== ((info.featureFlags & featureNativeTinyMlp) !== 0n) ||
  info.features.nativeModuleProgram !== ((info.featureFlags & featureNativeModuleProgram) !== 0n) ||
  info.features.programBindingRequirements !== ((info.featureFlags & featureProgramBindingRequirements) !== 0n) ||
  info.features.sessionPersistentUpload !== ((info.featureFlags & featureSessionPersistentUpload) !== 0n) ||
  (info.features.experimentalLlamaWgpuExecution && !info.features.nativeWgpuExecution) ||
  loadedRuntimeInfo.features.nativeWgpuExecution !== info.features.nativeWgpuExecution ||
  loadedRuntimeInfo.features.experimentalLlamaWgpuExecution !== info.features.experimentalLlamaWgpuExecution ||
  loadedRuntimeInfo.features.modelPathProbe !== info.features.modelPathProbe ||
  loadedRuntimeInfo.features.supportedCheckpoints !== info.features.supportedCheckpoints ||
  loadedRuntimeInfo.features.safetensorsHeaderProbe !== info.features.safetensorsHeaderProbe ||
  loadedRuntimeInfo.features.safetensorsDataLoad !== info.features.safetensorsDataLoad ||
  loadedRuntimeInfo.features.safetensorsDataProbe !== info.features.safetensorsDataProbe ||
  loadedRuntimeInfo.features.nativeTinyMlp !== info.features.nativeTinyMlp ||
  loadedRuntimeInfo.features.nativeModuleProgram !== info.features.nativeModuleProgram ||
  loadedRuntimeInfo.features.programBindingRequirements !== info.features.programBindingRequirements ||
  loadedRuntimeInfo.features.sessionPersistentUpload !== info.features.sessionPersistentUpload ||
  loadedRuntimeInfo.abiVersion !== info.abiVersion
) {
  throw new Error(`unexpected zgml runtime info: ${json(info)}`);
}
const abiSizes = abiStructSizes();
if (
  !Object.isFrozen(abiSizes) ||
  abiStructSize("runtimeInfo") !== 24 ||
  abiSizes.modelDesc !== 32 ||
  abiSizes.bufferBindDesc !== 64 ||
  abiSizes.externalResourceDesc !== 32 ||
  abiSizes.deviceBufferImportDesc !== 40 ||
  abiSizes.programInspection !== 312 ||
  abiSizes.runtimeProfile !== 168 ||
  abiSizes.safetensorsDataLoadDesc !== 24 ||
  abiSizes.moduleOpDesc !== 48 ||
  abiSizes.moduleDesc !== 32
) {
  throw new Error(`unexpected zgml ABI struct sizes: ${json(abiSizes)}`);
}
let unknownAbiStructRejected = false;
try {
  abiStructSize("missingStruct" as never);
} catch {
  unknownAbiStructRejected = true;
}
if (!unknownAbiStructRejected) throw new Error("expected unknown ABI struct kind to reject");
const runtimeReportsNativeWgpu = info.features.nativeWgpuExecution;
const runtimeReportsExperimentalLlamaWgpu = info.features.experimentalLlamaWgpuExecution;
const supportedCheckpoints = supportedCheckpointModels();
const supportedTinyLlama = supportedCheckpoints.find((model) => model.modelKind === "tiny-llama");
const supportedTinyLlama2Layer = supportedCheckpoints.find((model) => model.modelKind === "tiny-llama-2layer");
const supportedSmolLM = supportedCheckpoints.find((model) => model.modelKind === "smollm-135m");
if (
  !Object.isFrozen(supportedCheckpoints) ||
  supportedCheckpoints.some((model) => !Object.isFrozen(model)) ||
  supportedCheckpoints.length !== 3 ||
  !supportedTinyLlama ||
  supportedTinyLlama.vocabSize !== 8 ||
  supportedTinyLlama.dModel !== 4 ||
  supportedTinyLlama.nLayers !== 1 ||
  supportedTinyLlama.nKvHeads !== 1 ||
  supportedTinyLlama.tiedLmHead ||
  !supportedTinyLlama2Layer ||
  supportedTinyLlama2Layer.vocabSize !== 8 ||
  supportedTinyLlama2Layer.dModel !== 4 ||
  supportedTinyLlama2Layer.nLayers !== 2 ||
  supportedTinyLlama2Layer.nKvHeads !== 1 ||
  supportedTinyLlama2Layer.tiedLmHead ||
  !supportedSmolLM ||
  supportedSmolLM.vocabSize !== 49152 ||
  supportedSmolLM.dModel !== 576 ||
  supportedSmolLM.nLayers !== 30 ||
  supportedSmolLM.nKvHeads !== 3 ||
  !supportedSmolLM.tiedLmHead
) {
  throw new Error(`unexpected supported checkpoint catalog: ${json(supportedCheckpoints)}`);
}
const headerProbe = probeSafetensorsHeader(smollmSafetensorsHeader());
const fixedHeaderProbe = SmolLM135M.probeSafetensorsHeader(new TextEncoder().encode(smollmSafetensorsHeader()));
const tinyHeaderProbe = probeSafetensorsHeader(tinyLlamaSafetensorsHeader());
const fixedTinyHeaderProbe = TinyLlama.probeSafetensorsHeader(new TextEncoder().encode(tinyLlamaSafetensorsHeader()));
const smollmHeaderBytes = safetensorsHeaderFileBytes(smollmSafetensorsHeader());
const smollmByteProbe = probeModel(smollmHeaderBytes);
const explicitSmollmByteProbe = probeSafetensorsData(smollmHeaderBytes);
const fixedSmollmByteProbe = SmolLM135M.probe(smollmHeaderBytes);
const tempSafetensorsDir = mkdtempSync(join(tmpdir(), "zgml-safetensors-probe-"));
let fileHeaderProbe = headerProbe;
let tinyFileHeaderProbe = tinyHeaderProbe;
let tinyByteHeaderProbe = tinyHeaderProbe;
let explicitTinyByteHeaderProbe = tinyHeaderProbe;
let fixedTinyByteHeaderProbe = fixedTinyHeaderProbe;
try {
  const tempSafetensorsPath = join(tempSafetensorsDir, "model.safetensors");
  writeTinySafetensorsFile(tempSafetensorsPath, smollmSafetensorsHeader());
  fileHeaderProbe = probeModel(tempSafetensorsPath);
  const tinySafetensorsPath = join(tempSafetensorsDir, "tiny-model.safetensors");
  writeTinySafetensorsFile(tinySafetensorsPath, tinyLlamaSafetensorsHeader());
  tinyFileHeaderProbe = probeModel(tinySafetensorsPath);
  const tinyLoadPath = join(tempSafetensorsDir, "tiny-load.safetensors");
  writeTinyLlamaSafetensorsFile(tinyLoadPath);
  const tinyLoadBytes = tinyLlamaSafetensorsFileBytes();
  tinyByteHeaderProbe = probeModel(tinyLoadBytes);
  explicitTinyByteHeaderProbe = probeSafetensorsData(tinyLoadBytes);
  fixedTinyByteHeaderProbe = TinyLlama.probe(tinyLoadBytes);
  let smollmTinyProbeRejected = false;
  try {
    SmolLM135M.probe(tinyLoadBytes);
  } catch (err) {
    smollmTinyProbeRejected = String((err as { message?: unknown }).message ?? err).includes("unsupported");
  }
  if (!smollmTinyProbeRejected) {
    throw new Error("expected SmolLM135M.probe to reject tiny LLaMA safetensors data");
  }
  const autoTinyModel = loadModel(tinyLoadPath);
  try {
    expectTinyLlamaCheckpointInspection(autoTinyModel, "auto tiny LLaMA safetensors");
    expectTinyLlamaCheckpointStep(autoTinyModel, "auto tiny LLaMA safetensors");
  } finally {
    autoTinyModel.free();
  }
  const fixedTinyModel = loadModel(tinyLoadPath, { kind: "tiny-llama" });
  try {
    expectTinyLlamaCheckpointInspection(fixedTinyModel, "fixed tiny LLaMA safetensors");
  } finally {
    fixedTinyModel.free();
  }
  const classTinyModel = TinyLlama.load(tinyLoadPath);
  try {
    expectTinyLlamaCheckpointInspection(classTinyModel, "TinyLlama.load safetensors");
  } finally {
    classTinyModel.free();
  }
  const autoTinyBytesModel = loadSafetensorsData(tinyLoadBytes);
  try {
    expectTinyLlamaCheckpointInspection(autoTinyBytesModel, "auto tiny LLaMA safetensors data");
    expectTinyLlamaCheckpointStep(autoTinyBytesModel, "auto tiny LLaMA safetensors data");
  } finally {
    autoTinyBytesModel.free();
  }
  const rootTinyBytesModel = loadModel(tinyLoadBytes);
  try {
    expectTinyLlamaCheckpointInspection(rootTinyBytesModel, "root loadModel tiny LLaMA safetensors data");
    expectTinyLlamaCheckpointStep(rootTinyBytesModel, "root loadModel tiny LLaMA safetensors data");
  } finally {
    rootTinyBytesModel.free();
  }
  const fixedTinyBytesModel = TinyLlama.loadSafetensorsData(tinyLoadBytes);
  try {
    expectTinyLlamaCheckpointInspection(fixedTinyBytesModel, "TinyLlama.loadSafetensorsData");
  } finally {
    fixedTinyBytesModel.free();
  }
  let smollmTinyRejected = false;
  try {
    SmolLM135M.load(tinyLoadPath);
  } catch (err) {
    smollmTinyRejected = String((err as { message?: unknown }).message ?? err).includes("unsupported");
  }
  if (!smollmTinyRejected) {
    throw new Error("expected SmolLM135M.load to reject tiny LLaMA safetensors");
  }
  let smollmTinyBytesRejected = false;
  try {
    SmolLM135M.loadSafetensorsData(tinyLoadBytes);
  } catch (err) {
    smollmTinyBytesRejected = String((err as { message?: unknown }).message ?? err).includes("unsupported");
  }
  if (!smollmTinyBytesRejected) {
    throw new Error("expected SmolLM135M.loadSafetensorsData to reject tiny LLaMA safetensors");
  }
} finally {
  rmSync(tempSafetensorsDir, { recursive: true, force: true });
}
if (
  headerProbe.modelKind !== "smollm-135m" ||
  headerProbe.vocabSize !== 49152 ||
  headerProbe.dModel !== 576 ||
  headerProbe.nLayers !== 30 ||
  headerProbe.nKvHeads !== 3 ||
  !headerProbe.tiedLmHead ||
  fixedHeaderProbe.modelKind !== "smollm-135m" ||
  fileHeaderProbe.modelKind !== "smollm-135m" ||
  smollmByteProbe.modelKind !== "smollm-135m" ||
  explicitSmollmByteProbe.modelKind !== "smollm-135m" ||
  fixedSmollmByteProbe.modelKind !== "smollm-135m" ||
  tinyHeaderProbe.modelKind !== "tiny-llama" ||
  tinyHeaderProbe.vocabSize !== 8 ||
  tinyHeaderProbe.dModel !== 4 ||
  tinyHeaderProbe.nLayers !== 1 ||
  tinyHeaderProbe.nKvHeads !== 1 ||
  tinyHeaderProbe.tiedLmHead ||
  fixedTinyHeaderProbe.modelKind !== "tiny-llama" ||
  tinyFileHeaderProbe.modelKind !== "tiny-llama" ||
  tinyByteHeaderProbe.modelKind !== "tiny-llama" ||
  explicitTinyByteHeaderProbe.modelKind !== "tiny-llama" ||
  fixedTinyByteHeaderProbe.modelKind !== "tiny-llama"
) {
  throw new Error(`unexpected safetensors header probe: ${json({ headerProbe, fixedHeaderProbe, fileHeaderProbe, tinyHeaderProbe, fixedTinyHeaderProbe, tinyFileHeaderProbe })}`);
}
let unknownLoadModelKindRejected = false;
try {
  loadModel("does-not-need-to-exist.gguf", { kind: "missing-family" as never });
} catch (err) {
  unknownLoadModelKindRejected = String((err as { message?: unknown }).message ?? err).includes("unknown zgml model load kind");
}
if (!unknownLoadModelKindRejected) {
  throw new Error("expected unknown loadModel kind to reject before native load");
}
let unknownProbeModelKindRejected = false;
try {
  probeModel("does-not-need-to-exist.gguf", { kind: "missing-family" as never });
} catch (err) {
  unknownProbeModelKindRejected = String((err as { message?: unknown }).message ?? err).includes("unknown zgml model load kind");
}
if (!unknownProbeModelKindRejected) {
  throw new Error("expected unknown probeModel kind to reject before native probe");
}

const model = TinyLinear.create({ inputLen: 2, outputLen: 3 });
try {
  const modelInspection = model.inspect();
  if (
    !Object.isFrozen(modelInspection) ||
    modelInspection.modelKind !== "tiny-linear" ||
    modelInspection.inputLen !== 2 ||
    modelInspection.outputLen !== 3 ||
    modelInspection.vocabSize !== 0
  ) {
    throw new Error(`unexpected tiny linear model inspection: ${json(modelInspection)}`);
  }

    const program = model.compile();
    try {
      if (!(program instanceof Program) || !(program instanceof TinyLinearProgram)) {
        throw new Error("model.compile must return a Program with TinyLinearProgram compatibility");
      }
      const programInputShape = program.inputShape();
      const programOutputShape = program.outputShape();
      if (
        programInputShape.join(",") !== "2" ||
        programOutputShape.join(",") !== "3" ||
        !Object.isFrozen(programInputShape) ||
        !Object.isFrozen(programOutputShape)
      ) {
        throw new Error(`unexpected tiny linear Program shapes: ${programInputShape} -> ${programOutputShape}`);
      }
      if (program.trace() !== null) {
        throw new Error("raw tiny-linear Program should not expose retained trace evidence");
      }
      if (program.compilerSignatures() !== null) {
        throw new Error("raw tiny-linear Program should not expose module compilerSignatures");
      }
      if (program.tensorProgramIr() !== null) {
        throw new Error("tiny-linear Program should not expose module Tensor Program IR");
      }
      if (program.memoryLayout() !== null) {
        throw new Error("tiny-linear Program should not expose module memoryLayout");
      }
      if (program.parameterLayout() !== null) {
        throw new Error("tiny-linear Program should not expose module parameterLayout");
      }
      if (program.kernelPlan() !== null) {
        throw new Error("tiny-linear Program should not expose module kernelPlan");
      }
      if (program.shapeConstraints() !== null) {
        throw new Error("tiny-linear Program should not expose module shapeConstraints");
      }
      const inspection = program.inspect();
    if (
      inspection.backend !== "cpu" ||
      !inspection.executionSupported ||
      inspection.externalResourcesSupported ||
      inspection.bufferCount !== 5 ||
      inspection.bufferElementCount !== 17 ||
      inspection.bufferByteLength !== 68 ||
      inspection.initialUploadCount !== 1 ||
      inspection.qweightCount !== 0 ||
      inspection.opCount === 0 ||
      inspection.runtimePatchMaxCacheWritePos !== u32Max ||
      inspection.runtimePatchMaxAttentionSeqKv !== u32Max ||
      inspection.commandCount === 0 ||
      inspection.commandStencilHash === 0n ||
      inspection.bindingRequirementHash === 0n ||
      inspection.persistentRequirementCount !== 2 ||
      inspection.stepInputRequirementCount !== 1 ||
      inspection.stepOutputRequirementCount !== 1 ||
      commandCategoryTotal(inspection) === 0
    ) {
      throw new Error("expected compiled program shape evidence");
    }
    const capabilities = program.capabilities();
    if (
      capabilities.backend !== "cpu" ||
      capabilities.mode !== "executable" ||
      !capabilities.canExecute ||
      program.canExecute() !== true ||
      program.executionMode() !== "executable" ||
      capabilities.canBindExternalResources ||
      program.canBindExternalResources() !== false ||
      capabilities.hasFullDispatchPlan ||
      program.hasFullDispatchPlan() !== false
    ) {
      throw new Error(`unexpected tiny linear capabilities: ${json(capabilities)}`);
    }
    expectCapabilityDispatchPlan(capabilities, inspection, "tiny linear");
    const requirements = program.requirements();
    if (
      requirements.modelKind !== "tiny-linear" ||
      requirements.inputLen !== 2 ||
      requirements.outputLen !== 3 ||
      requirements.weightsLen !== 6 ||
      requirements.biasLen !== 3 ||
      requirements.outputByteLength !== 12 ||
      requirements.maxTokenWindow !== 0 ||
      !Object.isFrozen(requirements)
    ) {
      throw new Error(`unexpected tiny linear requirements: ${json(requirements)}`);
    }
    const bufferLayout = program.bufferLayout();
    if (
      !Object.isFrozen(bufferLayout) ||
      !Object.isFrozen(bufferLayout.slots) ||
      bufferLayout.scalarType !== "f32" ||
      bufferLayout.scalarBytes !== requirements.scalarBytes ||
      bufferLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== "input:step-input:2:8|output:step-output:3:12|weights:persistent:6:24|bias:persistent:3:12" ||
      bufferLayout.output.byteLength !== requirements.outputByteLength ||
      bufferLayout.weights.elementCount !== requirements.weightsLen
    ) {
      throw new Error(`unexpected tiny linear buffer layout: ${json(bufferLayout)}`);
    }
    const compatibility = program.modelCompatibility(model);
    if (
      compatibility.programModelKind !== "tiny-linear" ||
      compatibility.modelKind !== "tiny-linear" ||
      compatibility.compatible !== false ||
      !Object.isFrozen(compatibility) ||
      program.acceptsModel(model) !== false
    ) {
      throw new Error(`unexpected tiny linear model compatibility: ${json(compatibility)}`);
    }

    let badBindRejected = false;
    try {
      const unexpected = program.bind({
        weights: [1],
        bias: [0.5, -0.5, 1.0],
      });
      unexpected.free();
    } catch (err) {
      badBindRejected = String((err as { message?: unknown }).message ?? err).includes("Program weights slot length 6");
    }
    if (!badBindRejected) throw new Error("expected Program.bind to reject mismatched weights before native bind");

    const session = program.bind({
      weights: [1, 2, 3, 4, 5, 6],
      bias: [0.5, -0.5, 1.0],
    });
    try {
      if (!(session instanceof Session) || !(session instanceof TinyLinearSession)) {
        throw new Error("program.bind must return a Session with TinyLinearSession compatibility");
      }
      const sessionInputShape = session.inputShape();
      const sessionOutputShape = session.outputShape();
      if (
        sessionInputShape.join(",") !== "2" ||
        sessionOutputShape.join(",") !== "3" ||
        !Object.isFrozen(sessionInputShape) ||
        !Object.isFrozen(sessionOutputShape)
      ) {
        throw new Error(`unexpected tiny linear Session shapes: ${sessionInputShape} -> ${sessionOutputShape}`);
      }
      const output = session.step([1, 2]);
      expectClose(output, [9.5, 11.5, 16.0]);
      const programProfile = program.runtimeProfile();
      if (!Object.isFrozen(programProfile) || programProfile.callCount !== 0 || programProfile.runtimePatchCallCount !== 0) {
        throw new Error(`expected cold program profile, got ${json(programProfile)}`);
      }
      const sessionProfile = session.runtimeProfile();
      if (
        !Object.isFrozen(sessionProfile) ||
        sessionProfile.callCount !== 1 ||
        sessionProfile.runtimePatchCallCount !== 1 ||
        sessionProfile.commandCount === 0
      ) {
        throw new Error(`unexpected session profile: ${json(sessionProfile)}`);
      }
      session.resetRuntimeProfile();
      const resetProfile = session.runtimeProfile();
      if (!Object.isFrozen(resetProfile) || resetProfile.callCount !== 0 || resetProfile.runtimePatchCallCount !== 0 || resetProfile.commandCount === 0) {
        throw new Error(`unexpected reset session profile: ${json(resetProfile)}`);
      }
      console.log(`zgml bun ffi smoke ok: [${Array.from(output).join(", ")}]`);
    } finally {
      session.free();
    }

    const frontend = nn.sequential([
      nn.linear(2, 3, {
        weights: [1, 2, 3, 4, 5, 6],
        bias: [0.5, -0.5, 1.0],
      }),
    ]);
    const frontendInput = tensor([1, 2], [2]);
    if (!(frontendInput instanceof Tensor) || frontendInput.rank !== 1 || frontendInput.length !== 2) {
      throw new Error("unexpected TS Tensor frontend input shape");
    }
    const frontendInputSize = frontendInput.size();
    if (
      !Object.isFrozen(frontendInput.shape) ||
      !Object.isFrozen(frontendInputSize) ||
      frontendInputSize.join(",") !== "2" ||
      frontendInput.size(0) !== 2 ||
      !Object.isFrozen(frontendInput.mean().shape)
    ) {
      throw new Error("expected TS Tensor shape evidence to be frozen");
    }
    if (frontendInput.dtype !== "f32" || frontendInput.device !== "cpu" || frontendInput.to("cpu") !== frontendInput || frontendInput.to("float32") !== frontendInput || frontendInput.cpu() !== frontendInput || frontendInput.float() !== frontendInput || frontendInput.float32() !== frontendInput) {
      throw new Error("unexpected TS Tensor dtype/device movement surface");
    }
    const movedCopy = frontendInput.to({ device: "cpu", dtype: "f32", copy: true });
    if (movedCopy === frontendInput || movedCopy.dtype !== "f32" || movedCopy.device !== "cpu") {
      throw new Error("expected TS Tensor.to copy to return a distinct f32 cpu tensor");
    }
    expectClose(movedCopy.toFloat32Array(), [1, 2]);
    const movedGradInput = tensor([1, 2], [2], { requiresGrad: true });
    train.backward(movedGradInput.to({ copy: true }).sum());
    expectClose(movedGradInput.grad!, [1, 1]);
    let implicitDeviceRejected = false;
    try {
      frontendInput.to("webgpu");
    } catch (err) {
      implicitDeviceRejected = String((err as { message?: unknown }).message ?? err).includes("Tensor.place(program, kind)");
    }
    if (!implicitDeviceRejected) throw new Error("expected TS Tensor.to webgpu to reject without a compiled Program");
    const dropoutDraws = [0.25, 0.75, 0.1, 0.9];
    const dropout = nn.dropout(0.5, { rng: () => dropoutDraws.shift() ?? 1 });
    if (!(dropout instanceof nn.Dropout) || dropout.kind !== "dropout" || dropout.p !== 0.5 || !dropout.training) {
      throw new Error("unexpected TS nn.Dropout/dropout surface");
    }
    const dropoutInput = tensor([1, 2, 3, 4], [4], { requiresGrad: true });
    const dropoutOut = dropout.forward(dropoutInput);
    if (!(dropoutOut instanceof Tensor)) throw new Error("expected TS dropout Tensor input to return a Tensor");
    expectClose(dropoutOut.toFloat32Array(), [0, 4, 0, 8]);
    train.backward(dropoutOut.sum());
    expectClose(dropoutInput.grad!, [0, 2, 0, 2]);
    const dropoutSupport = dropout.compileSupport({ inputShape: [4] });
    if (
      dropout.canCompile({ inputShape: [4] }) ||
      dropoutSupport.supported ||
      !Object.isFrozen(dropoutSupport.trace) ||
      dropoutSupport.trace!.ops[0].op !== "dropout" ||
      dropoutSupport.trace!.ops[0].training !== true ||
      dropoutSupport.trace!.ops[0].p !== 0.5 ||
      !Object.isFrozen(dropoutSupport.ir) ||
      dropoutSupport.ir!.ops[0].op !== "dropout" ||
      !dropoutSupport.diagnostics.some((diagnostic) => diagnostic.code === "unsupported-op" && diagnostic.stage === "kernelizer")
    ) {
      throw new Error(`unexpected TS dropout compile support: ${json(dropoutSupport)}`);
    }
    const zeroDropout = nn.dropout(0).train();
    const zeroDropoutSupport = zeroDropout.compileSupport({ inputShape: [4] });
    if (
      !zeroDropout.training ||
      !zeroDropout.canCompile({ inputShape: [4] }) ||
      !zeroDropoutSupport.supported ||
      zeroDropoutSupport.trace!.ops[0].op !== "dropout" ||
      zeroDropoutSupport.trace!.ops[0].training !== true ||
      zeroDropoutSupport.trace!.ops[0].p !== 0 ||
      zeroDropoutSupport.ir!.ops[0].op !== "dropout" ||
      zeroDropoutSupport.kernelPlan!.dispatchCount !== 0 ||
      zeroDropoutSupport.kernelPlan!.ops.length !== 0 ||
      zeroDropoutSupport.kernelPlan!.elidedOps.length !== 1 ||
      zeroDropoutSupport.kernelPlan!.elidedOps[0].op !== "dropout" ||
      zeroDropoutSupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS zero-dropout compile support: ${json(zeroDropoutSupport)}`);
    }
    const zeroDropoutProgram = zeroDropout.compile({ backend: "cpu", inputShape: [4] });
    try {
      const zeroDropoutSession = zeroDropoutProgram.bind(zeroDropout.bindParameters({ inputShape: [4] }));
      try {
        expectClose(zeroDropoutSession.step(tensor([8, 6, 4, 2], [4])), [8, 6, 4, 2]);
      } finally {
        zeroDropoutSession.free();
      }
    } finally {
      zeroDropoutProgram.free();
    }
    if (runtimeReportsNativeWgpu) {
      const zeroDropoutWebgpuProgram = zeroDropout.compile({ backend: "webgpu", inputShape: [4] });
      try {
        const zeroDropoutWebgpuInspection = zeroDropoutWebgpuProgram.inspect();
        const zeroDropoutWebgpuCapabilities = zeroDropoutWebgpuProgram.capabilities();
        if (
          zeroDropoutWebgpuInspection.backend !== "webgpu" ||
          !zeroDropoutWebgpuInspection.executionSupported ||
          !zeroDropoutWebgpuInspection.externalResourcesSupported ||
          zeroDropoutWebgpuInspection.opCount !== 0 ||
          zeroDropoutWebgpuInspection.commandCount !== 0 ||
          zeroDropoutWebgpuInspection.commandStencilHash !== 0n ||
          zeroDropoutWebgpuInspection.backendDispatchCount !== 0 ||
          !zeroDropoutWebgpuInspection.dispatchPlanSupported ||
          zeroDropoutWebgpuInspection.dispatchPlanCoveredOpCount !== 0 ||
          zeroDropoutWebgpuInspection.dispatchPlanFirstUnsupportedOp !== null ||
          zeroDropoutWebgpuCapabilities.mode !== "executable" ||
          !zeroDropoutWebgpuCapabilities.canExecute ||
          !zeroDropoutWebgpuProgram.canExecute() ||
          zeroDropoutWebgpuProgram.executionMode() !== "executable" ||
          !zeroDropoutWebgpuCapabilities.canBindExternalResources ||
          !zeroDropoutWebgpuProgram.canBindExternalResources() ||
          !zeroDropoutWebgpuCapabilities.hasFullDispatchPlan ||
          !zeroDropoutWebgpuProgram.hasFullDispatchPlan()
        ) {
          throw new Error(`unexpected TS zero-dropout WebGPU inspection: ${json({ zeroDropoutWebgpuInspection, zeroDropoutWebgpuCapabilities })}`);
        }
        expectCapabilityDispatchPlan(zeroDropoutWebgpuCapabilities, zeroDropoutWebgpuInspection, "WebGPU zero-dropout");

        const zeroDropoutWebgpuSession = zeroDropoutWebgpuProgram.bind(zeroDropout.bindParameters({ inputShape: [4] }));
        try {
          expectClose(zeroDropoutWebgpuSession.step(tensor([8, 6, 4, 2], [4])), [8, 6, 4, 2]);
          const profile = zeroDropoutWebgpuSession.runtimeProfile();
          if (
            profile.callCount !== 1 ||
            profile.commandCount !== 0 ||
            profile.backendOpCount !== 0 ||
            profile.backendDispatchCount !== 0 ||
            profile.syncCount !== 1
          ) {
            throw new Error(`unexpected TS zero-dropout WebGPU host profile: ${json(profile)}`);
          }
        } finally {
          zeroDropoutWebgpuSession.free();
        }

        const zeroDropoutWebgpuDevice = zeroDropoutWebgpuProgram.device("webgpu");
        const zeroDropoutDeviceInput = zeroDropoutWebgpuDevice.createInputBuffer();
        const zeroDropoutDeviceOutput = zeroDropoutWebgpuDevice.createOutputBuffer();
        try {
          zeroDropoutDeviceInput.writeFloat32([9, 7, 5, 3]);
          const zeroDropoutDeviceSession = zeroDropoutWebgpuProgram.bind({
            weights: new Float32Array(0),
            input: zeroDropoutDeviceInput,
            output: zeroDropoutDeviceOutput,
          });
          try {
            const deviceSessionInfo = zeroDropoutDeviceSession.inspect();
            if (
              deviceSessionInfo.modelKind !== "module" ||
              deviceSessionInfo.backend !== "webgpu" ||
              deviceSessionInfo.outputStorage !== "external-resource" ||
              deviceSessionInfo.persistentBindingCount !== 0 ||
              deviceSessionInfo.stepInputCount !== 1 ||
              deviceSessionInfo.stepOutputCount !== 1 ||
              deviceSessionInfo.hostBindingCount !== 0 ||
              deviceSessionInfo.resourceBindingCount !== 2 ||
              deviceSessionInfo.bindingShapeHash === 0n
            ) {
              throw new Error(`unexpected TS zero-dropout WebGPU device-buffer session inspection: ${json(deviceSessionInfo)}`);
            }
            expectClose(zeroDropoutDeviceSession.step(), [9, 7, 5, 3]);
            const deviceProfile = zeroDropoutDeviceSession.runtimeProfile();
            if (
              deviceProfile.callCount !== 1 ||
              deviceProfile.commandCount !== 0 ||
              deviceProfile.backendOpCount !== 0 ||
              deviceProfile.backendDispatchCount !== 0 ||
              deviceProfile.syncCount !== 0
            ) {
              throw new Error(`unexpected TS zero-dropout WebGPU device-buffer profile: ${json(deviceProfile)}`);
            }
          } finally {
            zeroDropoutDeviceSession.free();
          }
        } finally {
          zeroDropoutDeviceOutput.free();
          zeroDropoutDeviceInput.free();
        }
      } finally {
        zeroDropoutWebgpuProgram.free();
      }
    }
    const dropoutEvalInput = tensor([5, 6], [2], { requiresGrad: true });
    if (dropout.eval() !== dropout || dropout.training || dropout.forward(dropoutEvalInput) !== dropoutEvalInput) {
      throw new Error("expected TS dropout eval mode to be an identity Tensor path");
    }
    dropout.train();
    const dropoutSeq = nn.sequential([dropout]);
    if (!dropoutSeq.training || !dropout.training) throw new Error("expected TS dropout Sequential to start in train mode");
    dropoutSeq.eval();
    if (dropoutSeq.training || dropout.training) throw new Error("expected TS Sequential.eval to cascade to dropout");
    dropoutSeq.train();
    if (!dropoutSeq.training || !dropout.training) throw new Error("expected TS Sequential.train to cascade to dropout");
    const evalDropout = nn.dropout(0.25).eval();
    const evalDropoutSupport = evalDropout.compileSupport({ inputShape: [4] });
    if (
      !evalDropout.canCompile({ inputShape: [4] }) ||
      !evalDropoutSupport.supported ||
      evalDropoutSupport.trace!.ops[0].op !== "dropout" ||
      evalDropoutSupport.trace!.ops[0].training !== false ||
      evalDropoutSupport.trace!.ops[0].p !== 0.25 ||
      evalDropoutSupport.ir!.ops[0].op !== "dropout" ||
      evalDropoutSupport.kernelPlan!.dispatchCount !== 0 ||
      evalDropoutSupport.kernelPlan!.ops.length !== 0 ||
      evalDropoutSupport.kernelPlan!.elidedOps.length !== 1 ||
      evalDropoutSupport.kernelPlan!.elidedOps[0].op !== "dropout" ||
      evalDropoutSupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS eval dropout compile support: ${json(evalDropoutSupport)}`);
    }
    const evalDropoutProgram = evalDropout.compile({ backend: "cpu", inputShape: [4] });
    try {
      const evalDropoutInspection = evalDropoutProgram.inspect();
      const evalDropoutEvidence = evalDropoutProgram.compileEvidence();
      if (
        evalDropoutInspection.commandCount !== 0 ||
        !evalDropoutEvidence ||
        evalDropoutEvidence.kind !== "module" ||
        evalDropoutEvidence.trace.ops[0].op !== "dropout" ||
        evalDropoutEvidence.kernelPlan.dispatchCount !== 0 ||
        evalDropoutEvidence.kernelPlan.ops.length !== 0 ||
        evalDropoutEvidence.kernelPlan.elidedOps[0].kernel !== "reshape"
      ) {
        throw new Error(`unexpected TS eval dropout compile evidence: ${json(evalDropoutEvidence)}`);
      }
      const evalDropoutSession = evalDropoutProgram.bind(evalDropout.bindParameters({ inputShape: [4] }));
      try {
        expectClose(evalDropoutSession.step(tensor([1, 2, 3, 4], [4])), [1, 2, 3, 4]);
      } finally {
        evalDropoutSession.free();
      }
    } finally {
      evalDropoutProgram.free();
    }
    const evalDropoutSeq = nn.sequential([nn.dropout(0.25), nn.identity()]).eval();
    const evalDropoutSeqSupport = evalDropoutSeq.compileSupport({ inputShape: [4] });
    if (
      !evalDropoutSeqSupport.supported ||
      evalDropoutSeqSupport.trace!.ops.map((op) => op.op).join("|") !== "dropout|identity" ||
      evalDropoutSeqSupport.trace!.ops[0].training !== false ||
      evalDropoutSeqSupport.kernelPlan!.dispatchCount !== 0 ||
      evalDropoutSeqSupport.kernelPlan!.ops.length !== 0 ||
      evalDropoutSeqSupport.kernelPlan!.elidedOps[0].op !== "shape-chain" ||
      evalDropoutSeqSupport.kernelPlan!.elidedOps[0].fusedOps!.join("|") !== "dropout|identity"
    ) {
      throw new Error(`unexpected TS eval dropout Sequential compile support: ${json(evalDropoutSeqSupport)}`);
    }
    const evalDropoutSeqProgram = evalDropoutSeq.compile({ backend: "cpu", inputShape: [4] });
    try {
      const evalDropoutSeqSession = evalDropoutSeqProgram.bind(evalDropoutSeq.bindParameters({ inputShape: [4] }));
      try {
        expectClose(evalDropoutSeqSession.step(tensor([4, 3, 2, 1], [4])), [4, 3, 2, 1]);
      } finally {
        evalDropoutSeqSession.free();
      }
    } finally {
      evalDropoutSeqProgram.free();
    }
    const elidedDropoutLinear = nn.sequential([
      nn.dropout(0.25).eval(),
      nn.identity(),
      nn.linear(4, 2, { weights: [1, 2, 3, 4, 5, 6, 7, 8], bias: false }),
    ]);
    const elidedDropoutLinearSupport = elidedDropoutLinear.compileSupport({ inputShape: [4] });
    if (
      !elidedDropoutLinearSupport.supported ||
      elidedDropoutLinearSupport.trace!.ops.map((op) => op.op).join("|") !== "dropout|identity|linear" ||
      elidedDropoutLinearSupport.ir!.ops.map((op) => op.op).join("|") !== "dropout|identity|linear" ||
      elidedDropoutLinearSupport.kernelPlan!.dispatchCount !== 1 ||
      elidedDropoutLinearSupport.kernelPlan!.ops.length !== 1 ||
      elidedDropoutLinearSupport.kernelPlan!.ops[0].op !== "linear" ||
      elidedDropoutLinearSupport.kernelPlan!.elidedOpCount !== 1 ||
      elidedDropoutLinearSupport.kernelPlan!.elidedOps.length !== 1 ||
      elidedDropoutLinearSupport.kernelPlan!.elidedOps[0].op !== "shape-chain" ||
      elidedDropoutLinearSupport.kernelPlan!.elidedOps[0].fusedOps!.join("|") !== "dropout|identity"
    ) {
      throw new Error(`unexpected TS elided dropout-linear compile support: ${json(elidedDropoutLinearSupport)}`);
    }
    const elidedDropoutLinearProgram = elidedDropoutLinear.compile({ backend: "cpu", inputShape: [4] });
    try {
      const elidedInspection = elidedDropoutLinearProgram.inspect();
      const elidedEvidence = elidedDropoutLinearProgram.compileEvidence();
      if (
        elidedInspection.commandCount !== 1 ||
        !elidedEvidence ||
        elidedEvidence.kernelPlan.dispatchCount !== 1 ||
        elidedEvidence.kernelPlan.ops[0].op !== "linear" ||
        elidedEvidence.kernelPlan.elidedOps[0].fusedOps.join("|") !== "dropout|identity"
      ) {
        throw new Error(`unexpected TS elided dropout-linear compile evidence: ${json({ elidedInspection, elidedEvidence })}`);
      }
      const elidedCompatibility = elidedDropoutLinearProgram.moduleCompatibility(elidedDropoutLinear, { inputShape: [4] });
      if (!elidedCompatibility.compatible || elidedDropoutLinearProgram.acceptsModule(elidedDropoutLinear, { inputShape: [4] }) !== true) {
        throw new Error(`unexpected TS elided dropout-linear compatibility: ${json(elidedCompatibility)}`);
      }
      const identityIdentityLinear = nn.sequential([
        nn.identity(),
        nn.identity(),
        nn.linear(4, 2, { weights: [1, 2, 3, 4, 5, 6, 7, 8], bias: false }),
      ]);
      const identityIdentityCompatibility = elidedDropoutLinearProgram.moduleCompatibility(identityIdentityLinear, { inputShape: [4] });
      if (
        identityIdentityCompatibility.compatible ||
        identityIdentityCompatibility.diagnostics[0]?.code !== "ir-mismatch" ||
        !identityIdentityCompatibility.diagnostics.some((diagnostic) => diagnostic.code === "kernel-plan-mismatch") ||
        elidedDropoutLinearProgram.acceptsModule(identityIdentityLinear, { inputShape: [4] }) !== false
      ) {
        throw new Error(`expected TS elided IR compatibility mismatch: ${json(identityIdentityCompatibility)}`);
      }
      let identityIdentityBindRejected = false;
      try {
        elidedDropoutLinearProgram.bindModule(identityIdentityLinear, { inputShape: [4] });
      } catch (err) {
        identityIdentityBindRejected = String((err as { message?: unknown }).message ?? err).includes("Tensor Program IR");
      }
      if (!identityIdentityBindRejected) throw new Error("expected TS bindModule to reject mismatched elided Tensor Program IR");
      const elidedSession = elidedDropoutLinearProgram.bind(elidedDropoutLinear.bindParameters({ inputShape: [4] }));
      try {
        expectClose(elidedSession.step(tensor([1, 2, 3, 4], [4])), [50, 60]);
      } finally {
        elidedSession.free();
      }
    } finally {
      elidedDropoutLinearProgram.free();
    }
    const explicitBroadcastBias = tensor([10, 20, 30], [3], { requiresGrad: true });
    const explicitBroadcastedBias = explicitBroadcastBias.broadcastTo([2, 3]);
    if (explicitBroadcastedBias.shape.join(",") !== "2,3") throw new Error(`unexpected TS broadcastTo shape: ${explicitBroadcastedBias.shape.join(",")}`);
    expectClose(explicitBroadcastedBias.toFloat32Array(), [10, 20, 30, 10, 20, 30]);
    train.backward(explicitBroadcastedBias.sum());
    expectClose(explicitBroadcastBias.grad!, [2, 2, 2]);
    const explicitExpandBase = tensor([1, 2], [2, 1], { requiresGrad: true });
    const explicitExpanded = explicitExpandBase.expand([2, 3]);
    if (explicitExpanded.shape.join(",") !== "2,3") throw new Error(`unexpected TS expand shape: ${explicitExpanded.shape.join(",")}`);
    expectClose(explicitExpanded.toFloat32Array(), [1, 1, 1, 2, 2, 2]);
    train.backward(explicitExpanded.sum());
    expectClose(explicitExpandBase.grad!, [3, 3]);
    let explicitBroadcastRejected = false;
    try {
      tensor([1, 2], [2]).broadcastTo([3]);
    } catch (err) {
      explicitBroadcastRejected = String((err as { message?: unknown }).message ?? err).includes("cannot broadcast");
    }
    if (!explicitBroadcastRejected) throw new Error("expected TS broadcastTo incompatible shape to reject");
    const broadcastModuleInput = tensor([10, 20, 30], [3], { requiresGrad: true });
    const broadcastModule = nn.broadcastTo([2, 3]);
    if (!(broadcastModule instanceof nn.Shape)) throw new Error("expected TS nn.broadcastTo to create a Shape module");
    const broadcastModuleOut = broadcastModule.forward(broadcastModuleInput);
    if (broadcastModuleOut.shape.join(",") !== "2,3") throw new Error(`unexpected TS nn.broadcastTo shape: ${broadcastModuleOut.shape.join(",")}`);
    expectClose(broadcastModuleOut.toFloat32Array(), [10, 20, 30, 10, 20, 30]);
    train.backward(broadcastModuleOut.sum());
    expectClose(broadcastModuleInput.grad!, [2, 2, 2]);
    const flattenedByModule = nn.flatten().forward(tensor([1, 2, 3, 4], [2, 2]));
    if (flattenedByModule.shape.join(",") !== "4") throw new Error(`unexpected TS nn.flatten shape: ${flattenedByModule.shape.join(",")}`);
    const flattenCtorModule = new nn.Flatten();
    if (!(flattenCtorModule instanceof nn.Shape) || flattenCtorModule.kind !== "flatten") throw new Error("expected TS nn.Flatten to create a Shape flatten module");
    const flattenedByCtor = flattenCtorModule.forward(tensor([1, 2, 3, 4], [2, 2]));
    if (flattenedByCtor.shape.join(",") !== "4") throw new Error(`unexpected TS nn.Flatten shape: ${flattenedByCtor.shape.join(",")}`);
    expectClose(nn.reshape([2, 2]).forward(flattenedByModule).toFloat32Array(), [1, 2, 3, 4]);
    const identityModule = nn.identity();
    const identityCtorModule = new nn.Identity();
    if (
      !(identityModule instanceof nn.Shape) ||
      !(identityCtorModule instanceof nn.Shape) ||
      identityModule.kind !== "identity" ||
      identityCtorModule.kind !== "identity"
    ) {
      throw new Error("expected TS nn.identity/Identity to create Shape identity modules");
    }
    const identityInput = tensor([1, 2, 3, 4], [2, 2]);
    if (identityModule.forward(identityInput) !== identityInput) throw new Error("expected TS nn.identity to return the input Tensor");
    const identitySupport = identityModule.compileSupport({ inputShape: [2, 2] });
    if (
      !identityModule.canCompile({ inputShape: [2, 2] }) ||
      !identitySupport.supported ||
      identitySupport.trace!.ops[0].op !== "identity" ||
      identitySupport.trace!.ops[0].outputShape!.join(",") !== "2,2" ||
      identitySupport.ir!.ops[0].op !== "identity" ||
      identitySupport.kernelPlan!.dispatchCount !== 0 ||
      identitySupport.kernelPlan!.ops.length !== 0 ||
      identitySupport.kernelPlan!.elidedOps.length !== 1 ||
      identitySupport.kernelPlan!.elidedOps[0].op !== "identity" ||
      identitySupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS nn.identity compile support: ${json(identitySupport)}`);
    }
    const identityProgram = identityModule.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const identityInspection = identityProgram.inspect();
      const identityEvidence = identityProgram.compileEvidence();
      if (
        identityInspection.commandCount !== 0 ||
        !identityEvidence ||
        identityEvidence.kind !== "module" ||
        identityEvidence.trace.ops[0].op !== "identity" ||
        identityEvidence.kernelPlan.dispatchCount !== 0 ||
        identityEvidence.kernelPlan.ops.length !== 0 ||
        identityEvidence.kernelPlan.elidedOps[0].kernel !== "reshape"
      ) {
        throw new Error(`unexpected TS nn.identity compile evidence: ${json(identityEvidence)}`);
      }
      const identitySession = identityProgram.bind(identityModule.bindParameters({ inputShape: [2, 2] }));
      try {
        expectClose(identitySession.step(identityInput), [1, 2, 3, 4]);
      } finally {
        identitySession.free();
      }
    } finally {
      identityProgram.free();
    }
    const viewModule = nn.view([2, 2]);
    if (!(viewModule instanceof nn.Shape) || viewModule.kind !== "view") throw new Error("expected TS nn.view to create a Shape module");
    const viewedByModule = viewModule.forward(tensor([1, 2, 3, 4], [4]));
    if (viewedByModule.shape.join(",") !== "2,2") throw new Error(`unexpected TS nn.view shape: ${viewedByModule.shape.join(",")}`);
    expectClose(viewedByModule.toFloat32Array(), [1, 2, 3, 4]);
    const viewSupport = viewModule.compileSupport({ inputShape: [4] });
    if (
      !viewSupport.supported ||
      viewSupport.trace!.ops[0].op !== "view" ||
      viewSupport.ir!.ops[0].op !== "view" ||
      viewSupport.kernelPlan!.dispatchCount !== 0 ||
      viewSupport.kernelPlan!.ops.length !== 0 ||
      viewSupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS nn.view compile support: ${json(viewSupport)}`);
    }
    const viewProgram = viewModule.compile({ backend: "cpu", inputShape: [4] });
    try {
      const viewInspection = viewProgram.inspect();
      const viewEvidence = viewProgram.compileEvidence();
      if (
        viewInspection.commandCount !== 0 ||
        !viewEvidence ||
        viewEvidence.kind !== "module" ||
        viewEvidence.trace.ops[0].op !== "view" ||
        viewEvidence.kernelPlan.dispatchCount !== 0 ||
        viewEvidence.kernelPlan.ops.length !== 0 ||
        viewEvidence.kernelPlan.elidedOps[0].kernel !== "reshape"
      ) {
        throw new Error(`unexpected TS nn.view compile evidence: ${json(viewEvidence)}`);
      }
      const viewSession = viewProgram.bind(viewModule.bindParameters({ inputShape: [4] }));
      try {
        expectClose(viewSession.step(tensor([1, 2, 3, 4], [4])), [1, 2, 3, 4]);
      } finally {
        viewSession.free();
      }
    } finally {
      viewProgram.free();
    }
    const unsqueezeModule = nn.unsqueeze(0);
    if (!(unsqueezeModule instanceof nn.Shape) || unsqueezeModule.kind !== "unsqueeze") throw new Error("expected TS nn.unsqueeze to create a Shape module");
    const unsqueezedByModule = unsqueezeModule.forward(tensor([1, 2, 3], [3]));
    if (unsqueezedByModule.shape.join(",") !== "1,3") throw new Error(`unexpected TS nn.unsqueeze shape: ${unsqueezedByModule.shape.join(",")}`);
    expectClose(unsqueezedByModule.toFloat32Array(), [1, 2, 3]);
    const unsqueezeSupport = unsqueezeModule.compileSupport({ inputShape: [3] });
    if (
      !unsqueezeSupport.supported ||
      unsqueezeSupport.trace!.ops[0].op !== "unsqueeze" ||
      unsqueezeSupport.trace!.ops[0].outputShape!.join(",") !== "1,3" ||
      unsqueezeSupport.ir!.ops[0].op !== "unsqueeze" ||
      unsqueezeSupport.kernelPlan!.dispatchCount !== 0 ||
      unsqueezeSupport.kernelPlan!.ops.length !== 0 ||
      unsqueezeSupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS nn.unsqueeze compile support: ${json(unsqueezeSupport)}`);
    }
    const unsqueezeProgram = unsqueezeModule.compile({ backend: "cpu", inputShape: [3] });
    try {
      const unsqueezeSession = unsqueezeProgram.bind(unsqueezeModule.bindParameters({ inputShape: [3] }));
      try {
        expectClose(unsqueezeSession.step(tensor([1, 2, 3], [3])), [1, 2, 3]);
      } finally {
        unsqueezeSession.free();
      }
    } finally {
      unsqueezeProgram.free();
    }
    const squeezeModule = nn.squeeze(0);
    if (!(squeezeModule instanceof nn.Shape) || squeezeModule.kind !== "squeeze" || squeezeModule.squeezeAll) throw new Error("expected TS nn.squeeze(dim) to create a Shape module");
    const squeezedByModule = squeezeModule.forward(tensor([1, 2, 3], [1, 3]));
    if (squeezedByModule.shape.join(",") !== "3") throw new Error(`unexpected TS nn.squeeze shape: ${squeezedByModule.shape.join(",")}`);
    expectClose(squeezedByModule.toFloat32Array(), [1, 2, 3]);
    const squeezeSupport = squeezeModule.compileSupport({ inputShape: [1, 3] });
    if (
      !squeezeSupport.supported ||
      squeezeSupport.trace!.ops[0].op !== "squeeze" ||
      squeezeSupport.trace!.ops[0].outputShape!.join(",") !== "3" ||
      squeezeSupport.ir!.ops[0].op !== "squeeze" ||
      squeezeSupport.kernelPlan!.dispatchCount !== 0 ||
      squeezeSupport.kernelPlan!.ops.length !== 0 ||
      squeezeSupport.kernelPlan!.elidedOps[0].kernel !== "reshape"
    ) {
      throw new Error(`unexpected TS nn.squeeze compile support: ${json(squeezeSupport)}`);
    }
    const squeezeProgram = squeezeModule.compile({ backend: "cpu", inputShape: [1, 3] });
    try {
      const squeezeSession = squeezeProgram.bind(squeezeModule.bindParameters({ inputShape: [1, 3] }));
      try {
        expectClose(squeezeSession.step(tensor([1, 2, 3], [1, 3])), [1, 2, 3]);
      } finally {
        squeezeSession.free();
      }
    } finally {
      squeezeProgram.free();
    }
    const squeezeAllModule = nn.squeeze();
    if (!squeezeAllModule.squeezeAll || squeezeAllModule.forward(tensor([7], [1, 1])).shape.join(",") !== "1") {
      throw new Error("unexpected TS nn.squeeze() all-dim semantics");
    }
    const transposeModule = nn.transpose();
    if (!(transposeModule instanceof nn.Shape) || transposeModule.kind !== "transpose") throw new Error("expected TS nn.transpose to create a Shape module");
    const transposedByModule = transposeModule.forward(tensor([1, 2, 3, 4, 5, 6], [2, 3]));
    if (transposedByModule.shape.join(",") !== "3,2") throw new Error(`unexpected TS nn.transpose shape: ${transposedByModule.shape.join(",")}`);
    expectClose(transposedByModule.toFloat32Array(), [1, 4, 2, 5, 3, 6]);
    const transposeSupport = transposeModule.compileSupport({ inputShape: [2, 3] });
    if (
      !transposeModule.canCompile({ inputShape: [2, 3] }) ||
      !transposeSupport.supported ||
      !transposeSupport.trace ||
      transposeSupport.trace.ops[0].op !== "transpose" ||
      transposeSupport.trace.ops[0].outputShape!.join(",") !== "3,2" ||
      !transposeSupport.ir ||
      !Object.isFrozen(transposeSupport.ir) ||
      transposeSupport.ir.ops[0].op !== "transpose" ||
      transposeSupport.ir.ops[0].outputShape.join(",") !== "3,2" ||
      !transposeSupport.kernelPlan ||
      transposeSupport.kernelPlan.ops[0].op !== "transpose" ||
      transposeSupport.kernelPlan.ops[0].kernel !== "transpose" ||
      transposeSupport.kernelPlan.ops[0].nativeDispatchCount !== 1 ||
      transposeSupport.kernelPlan.ops[0].nativeKernels.join(",") !== "transpose" ||
      transposeSupport.kernelPlan.outputShape.join(",") !== "3,2"
    ) {
      throw new Error(`unexpected TS nn.transpose compile support: ${json(transposeSupport)}`);
    }
    const transposeProgram = transposeModule.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const transposeSession = transposeProgram.bind(transposeModule.bindParameters({ inputShape: [2, 3] }));
      try {
        expectClose(transposeSession.step(tensor([1, 2, 3, 4, 5, 6], [2, 3])), [1, 4, 2, 5, 3, 6]);
      } finally {
        transposeSession.free();
      }
    } finally {
      transposeProgram.free();
    }
    const shapeSeq = nn.sequential([nn.broadcastTo([2, 3]), nn.logSoftmax(-1)]);
    const shapeTrace = shapeSeq.trace({ inputShape: [3] });
    const shapeTraceOp = shapeTrace.ops[0];
    if (
      !Object.isFrozen(shapeTrace) ||
      !Object.isFrozen(shapeTrace.ops) ||
      !Object.isFrozen(shapeTraceOp) ||
      shapeTrace.outputShape!.join(",") !== "2,3" ||
      shapeTraceOp.op !== "broadcastTo" ||
      shapeTraceOp.outputShape!.join(",") !== "2,3"
    ) {
      throw new Error("unexpected TS shape-module trace");
    }
    if (
      !Object.isFrozen(shapeTraceOp.shape) ||
      shapeTraceOp.shape.join(",") !== "2,3"
    ) {
      throw new Error("unexpected TS shape-module trace");
    }
    const shapeSupport = shapeSeq.compileSupport({ inputShape: [3] });
    if (
      !shapeSeq.canCompile({ inputShape: [3] }) ||
      !shapeSupport.supported ||
      shapeSupport.modelKind !== "module" ||
      shapeSupport.inputLen !== 3 ||
      shapeSupport.outputLen !== 6 ||
      shapeSupport.weightsLen !== 0 ||
      shapeSupport.biasLen !== 0 ||
      !shapeSupport.trace ||
      shapeSupport.trace.ops.map((op) => op.op).join("|") !== "broadcastTo|logSoftmax" ||
      shapeSupport.kernelPlan!.ops.map((op) => op.nativeDispatchCount).join(",") !== "1,1" ||
      shapeSupport.kernelPlan!.ops.map((op) => op.nativeKernels.join("+")).join("|") !== "broadcast|log-softmax"
    ) {
      throw new Error(`unexpected TS shape-module compile support: ${json(shapeSupport)}`);
    }
    const shapeProgram = shapeSeq.compile({ backend: "cpu", inputShape: [3] });
    try {
      const shapeSession = shapeProgram.bind(shapeSeq.bindParameters({ inputShape: [3] }));
      try {
        expectCloseWithin(
          shapeSession.step(tensor([10, 20, 30], [3])),
          Array.from(shapeSeq.forward(tensor([10, 20, 30], [3])).toFloat32Array()),
          1e-6,
        );
      } finally {
        shapeSession.free();
      }
    } finally {
      shapeProgram.free();
    }
    const shapeChain = nn.sequential([nn.identity(), nn.unsqueeze(0), nn.squeeze(0), nn.view([3])]);
    const shapeChainSupport = shapeChain.compileSupport({ inputShape: [3] });
    if (
      !shapeChain.canCompile({ inputShape: [3] }) ||
      !shapeChainSupport.supported ||
      shapeChainSupport.trace!.ops.map((op) => op.op).join("|") !== "identity|unsqueeze|squeeze|view" ||
      shapeChainSupport.ir!.opCount !== 4 ||
      shapeChainSupport.kernelPlan!.opCount !== 4 ||
      shapeChainSupport.kernelPlan!.dispatchCount !== 0 ||
      shapeChainSupport.kernelPlan!.ops.length !== 0 ||
      shapeChainSupport.kernelPlan!.elidedOps.length !== 1 ||
      shapeChainSupport.kernelPlan!.elidedOps[0].op !== "shape-chain" ||
      shapeChainSupport.kernelPlan!.elidedOps[0].kernel !== "reshape" ||
      shapeChainSupport.kernelPlan!.elidedOps[0].nativeDispatchCount !== 0 ||
      shapeChainSupport.kernelPlan!.elidedOps[0].nativeKernels.length !== 0 ||
      shapeChainSupport.kernelPlan!.elidedOps[0].fusedOpCount !== 4 ||
      shapeChainSupport.kernelPlan!.elidedOps[0].fusedOps!.join("|") !== "identity|unsqueeze|squeeze|view" ||
      shapeChainSupport.kernelPlan!.elidedOps[0].outputShape.join(",") !== "3" ||
      shapeChainSupport.kernelPlan!.elidedOps[0].scalarType !== "f32" ||
      shapeChainSupport.kernelPlan!.elidedOps[0].scalarBytes !== 4 ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps[0]) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps[0].inputShape) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps[0].outputShape) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps[0].inputValueIds) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.elidedOps[0].fusedOps) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.shapeConstraints) ||
      shapeChainSupport.kernelPlan!.shapeConstraints.specialization !== "exact" ||
      shapeChainSupport.kernelPlan!.shapeConstraints.inputRank !== 1 ||
      shapeChainSupport.kernelPlan!.shapeConstraints.outputRank !== 1 ||
      shapeChainSupport.kernelPlan!.shapeConstraints.inputShape.join(",") !== "3" ||
      shapeChainSupport.kernelPlan!.shapeConstraints.outputShape.join(",") !== "3" ||
      shapeChainSupport.kernelPlan!.shapeConstraints.inputLen !== 3 ||
      shapeChainSupport.kernelPlan!.shapeConstraints.outputLen !== 3 ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.memoryLayout) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.memoryLayout.values) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.memoryLayout.values[1].consumerOpIndices) ||
      shapeChainSupport.kernelPlan!.memoryLayout.valueCount !== 5 ||
      shapeChainSupport.kernelPlan!.memoryLayout.totalScalarCount !== 15 ||
      shapeChainSupport.kernelPlan!.memoryLayout.totalByteLength !== 60 ||
      shapeChainSupport.kernelPlan!.memoryLayout.scratchScalarCount !== 6 ||
      shapeChainSupport.kernelPlan!.memoryLayout.scratchByteLength !== 24 ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => `${value.id}:${value.role}:${value.scalarType}:${value.scalarBytes}:${value.shape.join("x")}:${value.strides.join("x")}:${value.op ?? "input"}:${value.path ?? ""}`).join("|") !== "0:input:f32:4:3:1:input:|1:op-output:f32:4:3:1:identity:0|2:op-output:f32:4:1x3:3x1:unsqueeze:1|3:op-output:f32:4:3:1:squeeze:2|4:op-output:f32:4:3:1:view:3" ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => `${value.storageClass}:${value.buffer}`).join("|") !== "step-input:input|scratch:scratch|scratch:scratch|scratch:scratch|step-output:output" ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => `${value.producerOpIndex ?? "ext"}>${value.consumerOpIndices.join("+") || "-"}:${value.firstUseOpIndex ?? "-"}:${value.lastUseOpIndex ?? "-"}:${value.liveStartOpIndex ?? "-"}:${value.liveEndOpIndex ?? "-"}`).join("|") !== "ext>0:0:0:0:0|0>1:1:1:0:1|1>2:2:2:1:2|2>3:3:3:2:3|3>-:-:-:3:3" ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => `${value.scalarOffset}:${value.byteOffset}`).join("|") !== "0:0|3:12|6:24|9:36|12:48" ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => `${value.bufferScalarOffset}:${value.bufferByteOffset}`).join("|") !== "0:0|0:0|3:12|0:0|0:0" ||
      shapeChainSupport.kernelPlan!.memoryLayout.values.map((value) => value.byteLength).join(",") !== "12,12,12,12,12" ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.bufferLayout) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.bufferLayout.slots) ||
      shapeChainSupport.kernelPlan!.bufferLayout.scalarType !== "f32" ||
      shapeChainSupport.kernelPlan!.bufferLayout.scalarBytes !== 4 ||
      shapeChainSupport.kernelPlan!.bufferLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== "input:step-input:3:12|output:step-output:3:12|weights:persistent:0:0|bias:persistent:0:0" ||
      shapeChainSupport.kernelPlan!.bufferLayout.input.elementCount !== 3 ||
      shapeChainSupport.kernelPlan!.bufferLayout.output.elementCount !== 3 ||
      shapeChainSupport.kernelPlan!.bufferLayout.weights.elementCount !== 0 ||
      shapeChainSupport.kernelPlan!.bufferLayout.bias.elementCount !== 0 ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.parameterLayout) ||
      !Object.isFrozen(shapeChainSupport.kernelPlan!.parameterLayout.parameters) ||
      shapeChainSupport.kernelPlan!.parameterLayout.weightsLen !== 0 ||
      shapeChainSupport.kernelPlan!.parameterLayout.biasLen !== 0 ||
      shapeChainSupport.kernelPlan!.parameterLayout.parameters.length !== 0
    ) {
      throw new Error(`unexpected TS shape-chain compile support: ${json(shapeChainSupport)}`);
    }
    const shapeChainProgram = shapeChain.compile({ backend: "cpu", inputShape: [3] });
    try {
      const evidence = shapeChainProgram.compileEvidence();
      const memoryLayout = shapeChainProgram.memoryLayout();
      if (
        !evidence ||
        evidence.kind !== "module" ||
        evidence.kernelPlan.dispatchCount !== 0 ||
        evidence.kernelPlan.ops.length !== 0 ||
        evidence.kernelPlan.elidedOps[0].op !== "shape-chain" ||
        evidence.kernelPlan.shapeConstraints.specialization !== "exact" ||
        evidence.kernelPlan.shapeConstraints.outputShape.join(",") !== "3" ||
        memoryLayout !== evidence.kernelPlan.memoryLayout ||
        memoryLayout.valueCount !== 5 ||
        memoryLayout.values[2].shape.join(",") !== "1,3" ||
        memoryLayout.values[2].byteLength !== 12 ||
        memoryLayout.values[4].op !== "view" ||
        evidence.kernelPlan.bufferLayout.output.byteLength !== 12 ||
        !Object.isFrozen(evidence.kernelPlan.bufferLayout.slots) ||
        "nativeOps" in evidence.kernelPlan ||
        evidence.kernelPlan.parameterLayout.parameters.length !== 0
      ) {
        throw new Error(`unexpected TS shape-chain compile evidence: ${json(evidence)}`);
      }
      const shapeChainSession = shapeChainProgram.bind(shapeChain.bindParameters({ inputShape: [3] }));
      try {
        expectClose(shapeChainSession.step(tensor([1, 2, 3], [3])), [1, 2, 3]);
      } finally {
        shapeChainSession.free();
      }
    } finally {
      shapeChainProgram.free();
    }
    const rankNarrow = nn.narrow(0, 1, 2);
    if (!(rankNarrow instanceof nn.Shape) || rankNarrow.kind !== "narrow") throw new Error("expected TS nn.narrow to create a Shape module");
    const rankNarrowOut = rankNarrow.forward(tensor([10, 20, 30, 40], [4]));
    if (rankNarrowOut.shape.join(",") !== "2") throw new Error(`unexpected TS nn.narrow rank-1 shape: ${rankNarrowOut.shape.join(",")}`);
    expectClose(rankNarrowOut.toFloat32Array(), [20, 30]);
    const rankNarrowSupport = rankNarrow.compileSupport({ inputShape: [4] });
    if (
      !rankNarrow.canCompile({ inputShape: [4] }) ||
      !rankNarrowSupport.supported ||
      rankNarrowSupport.inputLen !== 4 ||
      rankNarrowSupport.outputLen !== 2 ||
      rankNarrowSupport.weightsLen !== 0 ||
      rankNarrowSupport.biasLen !== 0 ||
      rankNarrowSupport.trace!.ops[0].op !== "narrow" ||
      rankNarrowSupport.ir!.ops[0].op !== "narrow" ||
      rankNarrowSupport.kernelPlan!.ops[0].kernel !== "narrow"
    ) {
      throw new Error(`unexpected TS rank-1 narrow compile support: ${json(rankNarrowSupport)}`);
    }
    const rankNarrowProgram = rankNarrow.compile({ backend: "cpu", inputShape: [4] });
    try {
      const rankNarrowEvidence = rankNarrowProgram.compileEvidence();
      if (!rankNarrowEvidence || rankNarrowEvidence.kind !== "module" || rankNarrowEvidence.kernelPlan.ops[0].kernel !== "narrow") {
        throw new Error(`unexpected TS rank-1 narrow evidence: ${json(rankNarrowEvidence)}`);
      }
      const rankNarrowSession = rankNarrowProgram.bind(rankNarrow.bindParameters({ inputShape: [4] }));
      try {
        expectClose(rankNarrowSession.step(tensor([10, 20, 30, 40], [4])), [20, 30]);
      } finally {
        rankNarrowSession.free();
      }
    } finally {
      rankNarrowProgram.free();
    }
    const batchNarrow = nn.narrow(0, 1, 2);
    const batchNarrowSupport = batchNarrow.compileSupport({ inputShape: [3, 2] });
    if (
      !batchNarrow.canCompile({ inputShape: [3, 2] }) ||
      !batchNarrowSupport.supported ||
      batchNarrowSupport.trace!.ops[0].outputShape!.join(",") !== "2,2" ||
      batchNarrowSupport.outputLen !== 4 ||
      batchNarrowSupport.kernelPlan!.ops[0].kernel !== "narrow"
    ) {
      throw new Error(`unexpected TS batch-axis narrow compile support: ${json(batchNarrowSupport)}`);
    }
    const batchNarrowProgram = batchNarrow.compile({ backend: "cpu", inputShape: [3, 2] });
    try {
      const batchNarrowSession = batchNarrowProgram.bind(batchNarrow.bindParameters({ inputShape: [3, 2] }));
      try {
        expectClose(batchNarrowSession.step(tensor([1, 2, 3, 4, 5, 6], [3, 2])), [3, 4, 5, 6]);
      } finally {
        batchNarrowSession.free();
      }
    } finally {
      batchNarrowProgram.free();
    }
    const singleBatchFeatureNarrow = nn.narrow(1, 1, 2);
    const singleBatchFeatureSupport = singleBatchFeatureNarrow.compileSupport({ inputShape: [1, 4] });
    if (
      !singleBatchFeatureNarrow.canCompile({ inputShape: [1, 4] }) ||
      !singleBatchFeatureSupport.supported ||
      singleBatchFeatureSupport.trace!.ops[0].outputShape!.join(",") !== "1,2" ||
      singleBatchFeatureSupport.outputLen !== 2 ||
      singleBatchFeatureSupport.kernelPlan!.ops[0].kernel !== "narrow"
    ) {
      throw new Error(`unexpected TS single-batch feature narrow compile support: ${json(singleBatchFeatureSupport)}`);
    }
    const singleBatchFeatureProgram = singleBatchFeatureNarrow.compile({ backend: "cpu", inputShape: [1, 4] });
    try {
      const singleBatchFeatureSession = singleBatchFeatureProgram.bind(singleBatchFeatureNarrow.bindParameters({ inputShape: [1, 4] }));
      try {
        expectClose(singleBatchFeatureSession.step(tensor([10, 20, 30, 40], [1, 4])), [20, 30]);
      } finally {
        singleBatchFeatureSession.free();
      }
    } finally {
      singleBatchFeatureProgram.free();
    }
    const stridedFeatureNarrow = nn.narrow(1, 0, 1);
    const stridedFeatureSupport = stridedFeatureNarrow.compileSupport({ inputShape: [2, 3] });
    if (
      !stridedFeatureNarrow.canCompile({ inputShape: [2, 3] }) ||
      !stridedFeatureSupport.supported ||
      stridedFeatureSupport.diagnostics ||
      !stridedFeatureSupport.ir ||
      stridedFeatureSupport.ir.ops[0].op !== "narrow" ||
      !stridedFeatureSupport.kernelPlan ||
      stridedFeatureSupport.kernelPlan.dispatchCount !== 1 ||
      stridedFeatureSupport.kernelPlan.ops[0].kernel !== "narrow" ||
      stridedFeatureSupport.outputLen !== 2 ||
      stridedFeatureSupport.trace!.ops[0].outputShape!.join(",") !== "2,1"
    ) {
      throw new Error(`unexpected TS strided feature narrow compile support: ${json(stridedFeatureSupport)}`);
    }
    const stridedFeatureProgram = stridedFeatureNarrow.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const stridedFeatureSession = stridedFeatureProgram.bind(stridedFeatureNarrow.bindParameters({ inputShape: [2, 3] }));
      try {
        expectClose(stridedFeatureSession.step(tensor([1, 2, 3, 4, 5, 6], [2, 3])), [1, 4]);
      } finally {
        stridedFeatureSession.free();
      }
    } finally {
      stridedFeatureProgram.free();
    }
    const scalarSelect = nn.select(0, -2);
    if (!(scalarSelect instanceof nn.Shape) || scalarSelect.kind !== "select") throw new Error("expected TS nn.select to create a Shape module");
    const scalarSelectOut = scalarSelect.forward(tensor([10, 20, 30], [3]));
    if (scalarSelectOut.shape.join(",") !== "1") throw new Error(`unexpected TS rank-1 select shape: ${scalarSelectOut.shape.join(",")}`);
    expectClose(scalarSelectOut.toFloat32Array(), [20]);
    const scalarSelectSupport = scalarSelect.compileSupport({ inputShape: [3] });
    if (
      !scalarSelect.canCompile({ inputShape: [3] }) ||
      !scalarSelectSupport.supported ||
      scalarSelectSupport.outputLen !== 1 ||
      scalarSelectSupport.trace!.ops[0].op !== "select" ||
      scalarSelectSupport.ir!.ops[0].op !== "select" ||
      scalarSelectSupport.kernelPlan!.dispatchCount !== 1 ||
      scalarSelectSupport.kernelPlan!.ops[0].kernel !== "select"
    ) {
      throw new Error(`unexpected TS rank-1 select compile support: ${json(scalarSelectSupport)}`);
    }
    const scalarSelectProgram = scalarSelect.compile({ backend: "cpu", inputShape: [3] });
    try {
      const scalarSelectSession = scalarSelectProgram.bind(scalarSelect.bindParameters({ inputShape: [3] }));
      try {
        expectClose(scalarSelectSession.step(tensor([10, 20, 30], [3])), [20]);
      } finally {
        scalarSelectSession.free();
      }
    } finally {
      scalarSelectProgram.free();
    }
    const rowSelect = nn.select(0, -1);
    const rowSelectOut = rowSelect.forward(tensor([1, 2, 3, 4, 5, 6], [2, 3]));
    if (rowSelectOut.shape.join(",") !== "3") throw new Error(`unexpected TS row select shape: ${rowSelectOut.shape.join(",")}`);
    expectClose(rowSelectOut.toFloat32Array(), [4, 5, 6]);
    const rowSelectSupport = rowSelect.compileSupport({ inputShape: [2, 3] });
    if (
      !rowSelect.canCompile({ inputShape: [2, 3] }) ||
      !rowSelectSupport.supported ||
      rowSelectSupport.outputLen !== 3 ||
      rowSelectSupport.trace!.ops[0].outputShape!.join(",") !== "3" ||
      rowSelectSupport.kernelPlan!.dispatchCount !== 1 ||
      rowSelectSupport.kernelPlan!.ops[0].kernel !== "select"
    ) {
      throw new Error(`unexpected TS row select compile support: ${json(rowSelectSupport)}`);
    }
    const rowSelectProgram = rowSelect.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const rowSelectSession = rowSelectProgram.bind(rowSelect.bindParameters({ inputShape: [2, 3] }));
      try {
        expectClose(rowSelectSession.step(tensor([1, 2, 3, 4, 5, 6], [2, 3])), [4, 5, 6]);
      } finally {
        rowSelectSession.free();
      }
    } finally {
      rowSelectProgram.free();
    }
    const singleBatchFeatureSelect = nn.select(1, 2);
    const singleBatchFeatureSelectSupport = singleBatchFeatureSelect.compileSupport({ inputShape: [1, 4] });
    if (
      !singleBatchFeatureSelect.canCompile({ inputShape: [1, 4] }) ||
      !singleBatchFeatureSelectSupport.supported ||
      singleBatchFeatureSelectSupport.trace!.ops[0].outputShape!.join(",") !== "1" ||
      singleBatchFeatureSelectSupport.kernelPlan!.dispatchCount !== 1 ||
      singleBatchFeatureSelectSupport.kernelPlan!.ops[0].kernel !== "select"
    ) {
      throw new Error(`unexpected TS single-batch feature select compile support: ${json(singleBatchFeatureSelectSupport)}`);
    }
    const singleBatchFeatureSelectProgram = singleBatchFeatureSelect.compile({ backend: "cpu", inputShape: [1, 4] });
    try {
      const singleBatchFeatureSelectSession = singleBatchFeatureSelectProgram.bind(singleBatchFeatureSelect.bindParameters({ inputShape: [1, 4] }));
      try {
        expectClose(singleBatchFeatureSelectSession.step(tensor([10, 20, 30, 40], [1, 4])), [30]);
      } finally {
        singleBatchFeatureSelectSession.free();
      }
    } finally {
      singleBatchFeatureSelectProgram.free();
    }
    const stridedFeatureSelect = nn.select(1, 1);
    const stridedFeatureSelectSupport = stridedFeatureSelect.compileSupport({ inputShape: [2, 3] });
    if (
      !stridedFeatureSelect.canCompile({ inputShape: [2, 3] }) ||
      !stridedFeatureSelectSupport.supported ||
      stridedFeatureSelectSupport.diagnostics ||
      !stridedFeatureSelectSupport.ir ||
      stridedFeatureSelectSupport.ir.ops[0].op !== "select" ||
      !stridedFeatureSelectSupport.kernelPlan ||
      stridedFeatureSelectSupport.kernelPlan.dispatchCount !== 1 ||
      stridedFeatureSelectSupport.kernelPlan.ops[0].kernel !== "select" ||
      stridedFeatureSelectSupport.outputLen !== 2 ||
      stridedFeatureSelectSupport.trace!.ops[0].outputShape!.join(",") !== "2"
    ) {
      throw new Error(`unexpected TS strided feature select compile support: ${json(stridedFeatureSelectSupport)}`);
    }
    const stridedFeatureSelectProgram = stridedFeatureSelect.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const stridedFeatureSelectSession = stridedFeatureSelectProgram.bind(stridedFeatureSelect.bindParameters({ inputShape: [2, 3] }));
      try {
        expectClose(stridedFeatureSelectSession.step(tensor([1, 2, 3, 4, 5, 6], [2, 3])), [2, 5]);
      } finally {
        stridedFeatureSelectSession.free();
      }
    } finally {
      stridedFeatureSelectProgram.free();
    }
    const rankSlice = nn.slice(0, 1, 3);
    if (!(rankSlice instanceof nn.Shape) || rankSlice.kind !== "slice") throw new Error("expected TS nn.slice to create a Shape module");
    const rankSliceOut = rankSlice.forward(tensor([10, 20, 30, 40], [4]));
    if (rankSliceOut.shape.join(",") !== "2") throw new Error(`unexpected TS rank-1 slice shape: ${rankSliceOut.shape.join(",")}`);
    expectClose(rankSliceOut.toFloat32Array(), [20, 30]);
    const rankSliceSupport = rankSlice.compileSupport({ inputShape: [4] });
    if (
      !rankSlice.canCompile({ inputShape: [4] }) ||
      !rankSliceSupport.supported ||
      rankSliceSupport.outputLen !== 2 ||
      rankSliceSupport.trace!.ops[0].op !== "slice" ||
      rankSliceSupport.ir!.ops[0].op !== "slice" ||
      rankSliceSupport.kernelPlan!.dispatchCount !== 1 ||
      rankSliceSupport.kernelPlan!.ops[0].kernel !== "slice"
    ) {
      throw new Error(`unexpected TS rank-1 slice compile support: ${json(rankSliceSupport)}`);
    }
    const rankSliceProgram = rankSlice.compile({ backend: "cpu", inputShape: [4] });
    try {
      const rankSliceSession = rankSliceProgram.bind(rankSlice.bindParameters({ inputShape: [4] }));
      try {
        expectClose(rankSliceSession.step(tensor([10, 20, 30, 40], [4])), [20, 30]);
      } finally {
        rankSliceSession.free();
      }
    } finally {
      rankSliceProgram.free();
    }
    const rowSlice = nn.slice(0, 1, null);
    const rowSliceSupport = rowSlice.compileSupport({ inputShape: [3, 2] });
    if (
      !rowSlice.canCompile({ inputShape: [3, 2] }) ||
      !rowSliceSupport.supported ||
      rowSliceSupport.trace!.ops[0].outputShape!.join(",") !== "2,2" ||
      rowSliceSupport.outputLen !== 4 ||
      rowSliceSupport.kernelPlan!.ops[0].kernel !== "slice"
    ) {
      throw new Error(`unexpected TS row slice compile support: ${json(rowSliceSupport)}`);
    }
    const rowSliceProgram = rowSlice.compile({ backend: "cpu", inputShape: [3, 2] });
    try {
      const rowSliceSession = rowSliceProgram.bind(rowSlice.bindParameters({ inputShape: [3, 2] }));
      try {
        expectClose(rowSliceSession.step(tensor([1, 2, 3, 4, 5, 6], [3, 2])), [3, 4, 5, 6]);
      } finally {
        rowSliceSession.free();
      }
    } finally {
      rowSliceProgram.free();
    }
    const singleBatchFeatureSlice = nn.slice(1, 1, 3);
    const singleBatchFeatureSliceSupport = singleBatchFeatureSlice.compileSupport({ inputShape: [1, 4] });
    if (
      !singleBatchFeatureSlice.canCompile({ inputShape: [1, 4] }) ||
      !singleBatchFeatureSliceSupport.supported ||
      singleBatchFeatureSliceSupport.trace!.ops[0].outputShape!.join(",") !== "1,2" ||
      singleBatchFeatureSliceSupport.outputLen !== 2 ||
      singleBatchFeatureSliceSupport.kernelPlan!.ops[0].kernel !== "slice"
    ) {
      throw new Error(`unexpected TS single-batch feature slice compile support: ${json(singleBatchFeatureSliceSupport)}`);
    }
    const singleBatchFeatureSliceProgram = singleBatchFeatureSlice.compile({ backend: "cpu", inputShape: [1, 4] });
    try {
      const singleBatchFeatureSliceSession = singleBatchFeatureSliceProgram.bind(singleBatchFeatureSlice.bindParameters({ inputShape: [1, 4] }));
      try {
        expectClose(singleBatchFeatureSliceSession.step(tensor([10, 20, 30, 40], [1, 4])), [20, 30]);
      } finally {
        singleBatchFeatureSliceSession.free();
      }
    } finally {
      singleBatchFeatureSliceProgram.free();
    }
    const batchedFeatureSlice = nn.slice(1, 1, 3);
    const batchedFeatureSliceSupport = batchedFeatureSlice.compileSupport({ inputShape: [2, 4] });
    if (
      !batchedFeatureSlice.canCompile({ inputShape: [2, 4] }) ||
      !batchedFeatureSliceSupport.supported ||
      batchedFeatureSliceSupport.diagnostics ||
      batchedFeatureSliceSupport.trace!.ops[0].outputShape!.join(",") !== "2,2" ||
      batchedFeatureSliceSupport.outputLen !== 4 ||
      !batchedFeatureSliceSupport.kernelPlan ||
      batchedFeatureSliceSupport.kernelPlan.dispatchCount !== 1 ||
      batchedFeatureSliceSupport.kernelPlan.ops[0].kernel !== "slice"
    ) {
      throw new Error(`unexpected TS batched feature slice compile support: ${json(batchedFeatureSliceSupport)}`);
    }
    const batchedFeatureSliceProgram = batchedFeatureSlice.compile({ backend: "cpu", inputShape: [2, 4] });
    try {
      const batchedFeatureSliceSession = batchedFeatureSliceProgram.bind(batchedFeatureSlice.bindParameters({ inputShape: [2, 4] }));
      try {
        expectClose(batchedFeatureSliceSession.step(tensor([10, 20, 30, 40, 50, 60, 70, 80], [2, 4])), [20, 30, 60, 70]);
      } finally {
        batchedFeatureSliceSession.free();
      }
    } finally {
      batchedFeatureSliceProgram.free();
    }
    const steppedSlice = nn.slice(0, 0, 4, 2);
    const steppedSliceOut = steppedSlice.forward(tensor([10, 20, 30, 40], [4]));
    expectClose(steppedSliceOut.toFloat32Array(), [10, 30]);
    const steppedSliceSupport = steppedSlice.compileSupport({ inputShape: [4] });
    if (
      !steppedSlice.canCompile({ inputShape: [4] }) ||
      !steppedSliceSupport.supported ||
      steppedSliceSupport.diagnostics ||
      !steppedSliceSupport.ir ||
      !Object.isFrozen(steppedSliceSupport.ir) ||
      !Object.isFrozen(steppedSliceSupport.ir.ops[0].inputValueIds) ||
      steppedSliceSupport.ir.ops[0].op !== "slice" ||
      !steppedSliceSupport.kernelPlan ||
      steppedSliceSupport.kernelPlan.dispatchCount !== 1 ||
      steppedSliceSupport.kernelPlan.ops[0].kernel !== "slice" ||
      !Object.isFrozen(steppedSliceSupport.trace!.ops[0].outputShape) ||
      steppedSliceSupport.trace!.ops[0].outputShape!.join(",") !== "2"
    ) {
      throw new Error(`unexpected TS stepped slice compile support: ${json(steppedSliceSupport)}`);
    }
    const steppedSliceProgram = steppedSlice.compile({ backend: "cpu", inputShape: [4] });
    try {
      const steppedSliceSession = steppedSliceProgram.bind(steppedSlice.bindParameters({ inputShape: [4] }));
      try {
        expectClose(steppedSliceSession.step(tensor([10, 20, 30, 40], [4])), [10, 30]);
      } finally {
        steppedSliceSession.free();
      }
    } finally {
      steppedSliceProgram.free();
    }
    const steppedFeatureSlice = nn.slice(1, 0, 4, 2);
    const steppedFeatureSliceSupport = steppedFeatureSlice.compileSupport({ inputShape: [2, 4] });
    if (
      !steppedFeatureSlice.canCompile({ inputShape: [2, 4] }) ||
      !steppedFeatureSliceSupport.supported ||
      steppedFeatureSliceSupport.diagnostics ||
      !steppedFeatureSliceSupport.kernelPlan ||
      steppedFeatureSliceSupport.kernelPlan.ops[0].kernel !== "slice" ||
      steppedFeatureSliceSupport.trace!.ops[0].outputShape!.join(",") !== "2,2"
    ) {
      throw new Error(`unexpected TS stepped feature slice compile support: ${json(steppedFeatureSliceSupport)}`);
    }
    const steppedFeatureSliceProgram = steppedFeatureSlice.compile({ backend: "cpu", inputShape: [2, 4] });
    try {
      const steppedFeatureSliceSession = steppedFeatureSliceProgram.bind(steppedFeatureSlice.bindParameters({ inputShape: [2, 4] }));
      try {
        expectClose(steppedFeatureSliceSession.step(tensor([10, 20, 30, 40, 50, 60, 70, 80], [2, 4])), [10, 30, 50, 70]);
      } finally {
        steppedFeatureSliceSession.free();
      }
    } finally {
      steppedFeatureSliceProgram.free();
    }
    const catA = tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const catB = tensor([5, 6], [1, 2], { requiresGrad: true });
    const catRows = cat([catA, catB], 0);
    if (catRows.shape.join(",") !== "3,2") throw new Error(`unexpected TS cat shape: ${catRows.shape.join(",")}`);
    expectClose(catRows.toFloat32Array(), [1, 2, 3, 4, 5, 6]);
    train.backward(catRows.sum());
    expectClose(catA.grad!, [1, 1, 1, 1]);
    expectClose(catB.grad!, [1, 1]);
    const catCols = Tensor.cat([tensor([1, 2, 3, 4], [2, 2]), tensor([5, 6, 7, 8], [2, 2])], -1);
    if (catCols.shape.join(",") !== "2,4") throw new Error(`unexpected TS cat dim -1 shape: ${catCols.shape.join(",")}`);
    expectClose(catCols.toFloat32Array(), [1, 2, 5, 6, 3, 4, 7, 8]);
    const stackA = tensor([1, 2], [2], { requiresGrad: true });
    const stackB = tensor([3, 4], [2], { requiresGrad: true });
    const stackedRows = stack([stackA, stackB], 0);
    if (stackedRows.shape.join(",") !== "2,2") throw new Error(`unexpected TS stack shape: ${stackedRows.shape.join(",")}`);
    expectClose(stackedRows.toFloat32Array(), [1, 2, 3, 4]);
    train.backward(stackedRows.sum());
    expectClose(stackA.grad!, [1, 1]);
    expectClose(stackB.grad!, [1, 1]);
    const stackedLast = Tensor.stack([tensor([1, 2], [2]), tensor([3, 4], [2])], -1);
    if (stackedLast.shape.join(",") !== "2,2") throw new Error(`unexpected TS stack dim -1 shape: ${stackedLast.shape.join(",")}`);
    expectClose(stackedLast.toFloat32Array(), [1, 3, 2, 4]);
    expectClose(zeros([2, 2], { requiresGrad: true }).toFloat32Array(), [0, 0, 0, 0]);
    if (zeros([2, 2]).shape.join(",") !== "2,2" || !zeros([2, 2], { requiresGrad: true }).requiresGrad) {
      throw new Error("unexpected TS zeros factory metadata");
    }
    expectClose(ones(3).toFloat32Array(), [1, 1, 1]);
    expectClose(full([2], 7).toFloat32Array(), [7, 7]);
    expectClose(scalar(4).toFloat32Array(), [4]);
    if (scalar(4, { requiresGrad: true }).shape.join(",") !== "1" || !scalar(4, { requiresGrad: true }).requiresGrad) {
      throw new Error("unexpected TS scalar factory metadata");
    }
    expectClose(rand([3], { rng: () => 0.25 }).toFloat32Array(), [0.25, 0.25, 0.25]);
    expectClose(linspace([4], 0, 8).toFloat32Array(), [0, 2, 4, 6]);
    expectClose(linspace(0, 8, 4).toFloat32Array(), [0, 2, 4, 6]);
    expectClose(arange(1, 6, 2).toFloat32Array(), [1, 3, 5]);
    const deterministicRandn = randn([2], { rng: () => 0.5 });
    if (deterministicRandn.shape.join(",") !== "2" || !Array.from(deterministicRandn.toFloat32Array()).every(Number.isFinite)) {
      throw new Error("unexpected TS randn factory output");
    }
    expectClose(Tensor.ones([2]).toFloat32Array(), [1, 1]);
    expectClose(Tensor.scalar(5).toFloat32Array(), [5]);
    expectClose(Tensor.rand([2], { rng: () => 0.75 }).toFloat32Array(), [0.75, 0.75]);
    expectClose(Tensor.linspace(0, 3, 3).toFloat32Array(), [0, 1, 2]);
    const standaloneParam = parameter([2], [1]);
    const standaloneParamAlias = param([3], [1]);
    if (!standaloneParamAlias.requiresGrad) {
      throw new Error("unexpected TS param alias metadata");
    }
    train.backward(standaloneParam.mul([3]).sum());
    expectClose(standaloneParam.grad!, [3]);
    train.step(optim.sgd(standaloneParam, { lr: 0.1 }));
    expectCloseWithin(standaloneParam.toFloat32Array(), [1.7], 1e-6);
    expectClose(standaloneParam.grad!, [0]);
    expectClose(frontendInput.add([3, 4]).toFloat32Array(), [4, 6]);
    expectClose(frontendInput.mul([2, 3]).toFloat32Array(), [2, 6]);
    expectClose(frontendInput.sub([0.5, 1.5]).toFloat32Array(), [0.5, 0.5]);
    expectClose(frontendInput.div([1, 4]).toFloat32Array(), [1, 0.5]);
    const indexedTensor = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    if (indexedTensor.get([1, 2]) !== 6 || indexedTensor.get(1, -1) !== 6) {
      throw new Error("unexpected TS Tensor get result");
    }
    indexedTensor.set([0, 1], 20).set(1, 0, 40);
    expectClose(indexedTensor.toFloat32Array(), [1, 20, 3, 40, 5, 6]);
    if (JSON.stringify(indexedTensor.toArray()) !== JSON.stringify([[1, 20, 3], [40, 5, 6]]) || JSON.stringify(indexedTensor.tolist()) !== JSON.stringify(indexedTensor.toArray())) {
      throw new Error("unexpected TS Tensor nested array conversion");
    }
    const selectedTensor = indexedTensor.select(0, -1);
    if (selectedTensor.shape.join(",") !== "3") throw new Error(`unexpected TS Tensor select shape: ${selectedTensor.shape.join(",")}`);
    expectClose(selectedTensor.toFloat32Array(), [40, 5, 6]);
    const narrowedTensor = indexedTensor.narrow(1, 1, 2);
    if (narrowedTensor.shape.join(",") !== "2,2") throw new Error(`unexpected TS Tensor narrow shape: ${narrowedTensor.shape.join(",")}`);
    expectClose(narrowedTensor.toFloat32Array(), [20, 3, 5, 6]);
    const slicedTensor = indexedTensor.slice(1, 0, null, 2);
    if (slicedTensor.shape.join(",") !== "2,2") throw new Error(`unexpected TS Tensor slice shape: ${slicedTensor.shape.join(",")}`);
    expectClose(slicedTensor.toFloat32Array(), [1, 3, 40, 6]);
    const sliceGradInput = tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    train.backward(sliceGradInput.slice(1, 0, 3, 2).sum());
    expectClose(sliceGradInput.grad!, [1, 0, 1, 1, 0, 1]);
    sliceGradInput.zeroGrad();
    train.backward(sliceGradInput.select(0, 1).sum());
    expectClose(sliceGradInput.grad!, [0, 0, 0, 1, 1, 1]);
    sliceGradInput.zeroGrad();
    train.backward(sliceGradInput.narrow(1, -2, 2).sum());
    expectClose(sliceGradInput.grad!, [0, 1, 1, 0, 1, 1]);
    expectClose(tensor([1, 2, 3, 4, 5, 6], [2, 3]).add(tensor([10, 20, 30], [3])).toFloat32Array(), [11, 22, 33, 14, 25, 36]);
    expectClose(tensor([10, 20, 30], [3]).add(tensor([1, 2, 3, 4, 5, 6], [2, 3])).toFloat32Array(), [11, 22, 33, 14, 25, 36]);
    expectClose(tensor([1, 2, 3], [3]).mul([2]).toFloat32Array(), [2, 4, 6]);
    const broadcastBase = tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const broadcastBias = tensor([10, 20, 30], [3], { requiresGrad: true });
    train.backward(broadcastBase.add(broadcastBias).sum());
    expectClose(broadcastBase.grad!, [1, 1, 1, 1, 1, 1]);
    expectClose(broadcastBias.grad!, [2, 2, 2]);
    const broadcastLeft = tensor([10, 20, 30], [3], { requiresGrad: true });
    const broadcastRight = tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    train.backward(broadcastLeft.mul(broadcastRight).sum());
    expectClose(broadcastLeft.grad!, [5, 7, 9]);
    expectClose(broadcastRight.grad!, [10, 20, 30, 10, 20, 30]);
    if (frontendInput.sum().item() !== 3 || frontendInput.mean().item() !== 1.5) {
      throw new Error("unexpected TS Tensor reduction result");
    }
    const dimReduce = tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const dimSum0 = dimReduce.sumDim(0);
    if (dimSum0.shape.join(",") !== "1,3") {
      throw new Error(`unexpected TS sumDim(0) shape: ${dimSum0.shape.join(",")}`);
    }
    expectClose(dimSum0.toFloat32Array(), [5, 7, 9]);
    expectClose(dimReduce.sumDim(1).toFloat32Array(), [6, 15]);
    expectClose(dimReduce.meanDim(0).toFloat32Array(), [2.5, 3.5, 4.5]);
    expectClose(dimReduce.maxDim(1).toFloat32Array(), [3, 6]);
    expectClose(dimReduce.sum(0).toFloat32Array(), [5, 7, 9]);
    expectClose(dimReduce.sum(1).toFloat32Array(), [6, 15]);
    expectClose(dimReduce.mean(0).toFloat32Array(), [2.5, 3.5, 4.5]);
    expectClose(dimReduce.max(1).toFloat32Array(), [3, 6]);
    if (dimReduce.max().item() !== 6) {
      throw new Error("unexpected TS Tensor max reduction result");
    }
    train.backward(dimSum0.sum());
    expectClose(dimReduce.grad!, [1, 1, 1, 1, 1, 1]);
    const dimMaxGrad = tensor([1, 5, 5, 4, 2, 6], [2, 3], { requiresGrad: true });
    train.backward(dimMaxGrad.maxDim(1).sum());
    expectClose(dimMaxGrad.grad!, [0, 0.5, 0.5, 0, 0, 1]);
    expectClose(tensor([1, 2, 3, 4], [2, 2]).mm(tensor([5, 6, 7, 8], [2, 2])).toFloat32Array(), [19, 22, 43, 50]);
    expectClose(tensor([-1, 0, 2], [3]).relu().toFloat32Array(), [0, 0, 2]);
    expectCloseWithin(tensor([-1, 0, 2], [3]).silu().toFloat32Array(), [-1 / (1 + Math.exp(1)), 0, 2 / (1 + Math.exp(-2))], 1e-6);
    expectClose(tensor([1, Math.E], [2]).log().toFloat32Array(), [0, 1]);
    expectClose(tensor([1, -2], [2]).neg().toFloat32Array(), [-1, 2]);
    const softmaxDenom = Math.exp(1) + Math.exp(2) + Math.exp(3);
    const softmaxExpected = [Math.exp(1) / softmaxDenom, Math.exp(2) / softmaxDenom, Math.exp(3) / softmaxDenom];
    expectCloseWithin(tensor([1, 2, 3], [3]).softmax().toFloat32Array(), softmaxExpected, 1e-6);
    expectCloseWithin(tensor([1, 2, 3, 1, 2, 3], [2, 3]).softmax().toFloat32Array(), [...softmaxExpected, ...softmaxExpected], 1e-6);
    const colSoftmax = tensor([1, 2, 3, 4, 5, 6], [2, 3]).softmaxDim(0);
    expectCloseWithin(colSoftmax.toFloat32Array(), [
      Math.exp(1) / (Math.exp(1) + Math.exp(4)),
      Math.exp(2) / (Math.exp(2) + Math.exp(5)),
      Math.exp(3) / (Math.exp(3) + Math.exp(6)),
      Math.exp(4) / (Math.exp(1) + Math.exp(4)),
      Math.exp(5) / (Math.exp(2) + Math.exp(5)),
      Math.exp(6) / (Math.exp(3) + Math.exp(6)),
    ], 1e-6);
    const softmaxInput = tensor([1, 2, 3], [3], { requiresGrad: true });
    softmaxInput.softmax().backward([1, 0, 0]);
    expectCloseWithin(softmaxInput.grad!, [
      softmaxExpected[0] * (1 - softmaxExpected[0]),
      -softmaxExpected[0] * softmaxExpected[1],
      -softmaxExpected[0] * softmaxExpected[2],
    ], 1e-6);
    const logSoftmaxExpected = [1 - 3 - Math.log(softmaxDenom / Math.exp(3)), 2 - 3 - Math.log(softmaxDenom / Math.exp(3)), 3 - 3 - Math.log(softmaxDenom / Math.exp(3))];
    expectCloseWithin(tensor([1, 2, 3], [3]).logSoftmax().toFloat32Array(), logSoftmaxExpected, 1e-6);
    const colLogSoftmax = tensor([1, 4], [2, 1]).logSoftmaxDim(0);
    expectCloseWithin(colLogSoftmax.toFloat32Array(), [
      1 - Math.log(Math.exp(1) + Math.exp(4)),
      4 - Math.log(Math.exp(1) + Math.exp(4)),
    ], 1e-6);
    const logSoftmaxInput = tensor([1, 2, 3], [3], { requiresGrad: true });
    logSoftmaxInput.logSoftmax().backward([1, 0, 0]);
    expectCloseWithin(logSoftmaxInput.grad!, [
      1 - softmaxExpected[0],
      -softmaxExpected[1],
      -softmaxExpected[2],
    ], 1e-6);
    const restoredTensor = Tensor.fromJSON(frontendInput.toJSON());
    expectClose(restoredTensor.toFloat32Array(), [1, 2]);
    if (restoredTensor.shape.join(",") !== "2") {
      throw new Error("unexpected TS Tensor JSON shape");
    }
    const clonedTensor = frontendInput.clone();
    expectClose(clonedTensor.toFloat32Array(), [1, 2]);
    if (clonedTensor.toFloat32Array() === frontendInput.toFloat32Array()) {
      throw new Error("unexpected TS Tensor clone alias");
    }
    const cloneSource = tensor([1, 2], [2], { requiresGrad: true });
    cloneSource.clone().sum().backward();
    expectClose(cloneSource.grad!, [1, 1]);
    const detachedTensor = cloneSource.detach();
    if (detachedTensor.requiresGrad || detachedTensor.toFloat32Array() === cloneSource.toFloat32Array()) {
      throw new Error("unexpected TS Tensor detach semantics");
    }
    if (frontendInput.reshape([1, 2]).shape.join(",") !== "1,2" || frontendInput.view([-1, 1]).shape.join(",") !== "2,1") {
      throw new Error("unexpected TS Tensor reshape shape");
    }
    if (
      frontendInput.unsqueeze(0).shape.join(",") !== "1,2" ||
      frontendInput.unsqueeze(-1).shape.join(",") !== "2,1" ||
      tensor([1, 2], [1, 2]).squeeze(0).shape.join(",") !== "2" ||
      tensor([1, 2], [1, 2]).squeeze(1).shape.join(",") !== "1,2" ||
      tensor([1, 2], [1, 2]).squeeze().shape.join(",") !== "2"
    ) {
      throw new Error("unexpected TS Tensor squeeze/unsqueeze shape");
    }
    const squeezeGradSource = tensor([1, 2], [1, 2], { requiresGrad: true });
    squeezeGradSource.squeeze(0).sum().backward();
    expectClose(squeezeGradSource.grad!, [1, 1]);
    if (tensor([1, 2, 3, 4], [2, 2]).flatten().shape.join(",") !== "4") {
      throw new Error("unexpected TS Tensor flatten shape");
    }
    const transposeInput = tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const transposeOutput = transposeInput.transpose();
    if (transposeOutput.shape.join(",") !== "3,2") {
      throw new Error("unexpected TS Tensor transpose shape");
    }
    expectClose(transposeOutput.toFloat32Array(), [1, 4, 2, 5, 3, 6]);
    expectClose(transposeInput.T.toFloat32Array(), [1, 4, 2, 5, 3, 6]);
    transposeOutput.mul([1, 2, 3, 4, 5, 6]).sum().backward();
    expectClose(transposeInput.grad!, [1, 3, 5, 2, 4, 6]);
    const frontendHostOutput = frontend.forward(frontendInput);
    if (!(frontendHostOutput instanceof Tensor)) {
      throw new Error("expected TS nn.linear host output to be a Tensor");
    }
    expectClose(frontendHostOutput.toFloat32Array(), [9.5, 11.5, 16.0]);
    const frontendMse = loss.meanSquaredError(new Tensor(frontendHostOutput, [3]), [9.5, 11.5, 16.0]);
    const frontendSelfMse = loss.mse(frontendHostOutput, frontendHostOutput);
    if (!(frontendMse instanceof Tensor) || !(frontendSelfMse instanceof Tensor) || frontendMse.item() !== 0 || frontendSelfMse.item() !== 0) {
      throw new Error("unexpected TS nn/loss frontend host result");
    }
    const batchedLinear = nn.linear(2, 3, {
      weights: [1, 2, 3, 4, 5, 6],
      bias: [0.5, -0.5, 1.0],
    });
    const batchedInput = tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const batchedOutput = batchedLinear.forward(batchedInput);
    if (!(batchedOutput instanceof Tensor) || batchedOutput.shape.join(",") !== "2,3") {
      throw new Error("expected TS batched linear output to be a [batch, outFeatures] Tensor");
    }
    expectClose(batchedOutput.toFloat32Array(), [9.5, 11.5, 16.0, 19.5, 25.5, 34.0]);
    train.backward(batchedOutput.sum());
    const batchedParams = batchedLinear.parameters();
    expectClose(batchedInput.grad!, [6, 15, 6, 15]);
    expectClose(batchedParams[0].grad, [4, 4, 4, 6, 6, 6]);
    expectClose(batchedParams[1].grad, [2, 2, 2]);
    const frontendParams = frontend.parameters();
    if (
      frontendParams.length !== 2 ||
      frontendParams[0].name !== "0.weight" ||
      frontendParams[1].name !== "0.bias"
    ) {
      throw new Error(`unexpected TS nn parameter list: ${json(frontendParams.map((param) => param.name))}`);
    }
    const frontendNamed = frontend.namedParameters();
    if (frontendNamed.length !== 2 || frontendNamed[0].name !== "0.weight" || nn.namedParameters(frontend)[1].name !== "0.bias") {
      throw new Error(`unexpected TS nn named parameter list: ${json(frontendNamed.map((param) => param.name))}`);
    }
    frontendParams[0].grad.set([1, 2, 3, 4, 5, 6]);
    frontend.zeroGrad();
    expectClose(frontendParams[0].grad, [0, 0, 0, 0, 0, 0]);
    const frontendState = frontend.stateDict();
    if (frontendState["0.weight"].shape.join(",") !== "2,3") {
      throw new Error(`unexpected TS stateDict weight shape: ${frontendState["0.weight"].shape.join(",")}`);
    }
    if (frontendState["0.weight"].layout !== "row-major:linear.weight[in_features,out_features]") {
      throw new Error(`unexpected TS stateDict weight layout: ${frontendState["0.weight"].layout}`);
    }
    if (frontendState["0.bias"].layout !== "row-major:linear.bias[out_features]") {
      throw new Error(`unexpected TS stateDict bias layout: ${frontendState["0.bias"].layout}`);
    }
    expectClose(frontendState["0.weight"].data, [1, 2, 3, 4, 5, 6]);
    frontendState["0.weight"].data[0] = -999;
    expectClose(frontend.parameters()[0].data, [1, 2, 3, 4, 5, 6]);
    frontend.loadStateDict({
      "0.weight": [6, 5, 4, 3, 2, 1],
      "0.bias": [1, 2, 3],
    });
    expectClose(frontend.parameters()[0].data, [6, 5, 4, 3, 2, 1]);
    expectClose(nn.stateDict(frontend)["0.bias"].data, [1, 2, 3]);
    let shapeStateRejected = false;
    try {
      frontend.loadStateDict({
        "0.weight": { shape: [3, 2], data: [1, 2, 3, 4, 5, 6] },
        "0.bias": { shape: [3], data: [0, 0, 0] },
      });
    } catch (err) {
      shapeStateRejected = String((err as { message?: unknown }).message ?? err).includes("shape must be [2,3]");
    }
    if (!shapeStateRejected) throw new Error("expected TS stateDict shape rejection");
    let layoutStateRejected = false;
    try {
      frontend.loadStateDict({
        "0.weight": {
          shape: [2, 3],
          layout: "feature-major:linear.weight[out_features,in_features]",
          data: [1, 2, 3, 4, 5, 6],
        },
        "0.bias": { shape: [3], layout: "row-major:linear.bias[out_features]", data: [0, 0, 0] },
      });
    } catch (err) {
      layoutStateRejected = String((err as { message?: unknown }).message ?? err).includes("layout must be row-major:linear.weight[in_features,out_features]");
    }
    if (!layoutStateRejected) throw new Error("expected TS stateDict layout rejection");
    let strictStateRejected = false;
    try {
      frontend.loadStateDict({
        "0.weight": [1, 2, 3, 4, 5, 6],
        "0.bias": [0, 0, 0],
        extra: [1],
      });
    } catch (err) {
      strictStateRejected = String((err as { message?: unknown }).message ?? err).includes("unexpected parameter extra");
    }
    if (!strictStateRejected) throw new Error("expected TS stateDict strict extra key rejection");
    frontend.loadStateDict({
      "0.weight": [1, 2, 3, 4, 5, 6],
      "0.bias": [0.5, -0.5, 1.0],
    });
    const checkpointModel = nn.sequential([
      nn.linear(2, 2, { weights: [1, 2, 3, 4], bias: [0.25, -0.25] }),
    ]);
    const checkpointOptimizer = optim.sgd(checkpointModel, { lr: 0.1, momentum: 0.5 });
    train.step(checkpointOptimizer, { loss: checkpointModel.forward(tensor([1, -1], [2])).sum(), zeroGrad: false });
    const checkpointSnapshot = checkpoint.create({
      model: checkpointModel,
      optimizer: checkpointOptimizer,
      metadata: { epoch: 1, tag: "bun-smoke" },
    });
    const checkpointRoundTrip = JSON.parse(JSON.stringify(checkpointSnapshot));
    if (
      checkpointRoundTrip.format !== "zgml.checkpoint" ||
      checkpointRoundTrip.version !== 1 ||
      checkpointRoundTrip.metadata.tag !== "bun-smoke" ||
      !Array.isArray(checkpointRoundTrip.model["0.weight"].data) ||
      checkpointRoundTrip.model["0.weight"].dtype !== "f32" ||
      checkpointRoundTrip.optimizer.kind !== "sgd" ||
      checkpointRoundTrip.optimizer.entries.length !== 2
    ) {
      throw new Error(`unexpected TS checkpoint snapshot: ${json(checkpointRoundTrip)}`);
    }
    const restoredCheckpointModel = nn.sequential([
      nn.linear(2, 2, { weights: [0, 0, 0, 0], bias: [0, 0] }),
    ]);
    const restoredCheckpointOptimizer = optim.sgd(restoredCheckpointModel, { lr: 0.1, momentum: 0.5 });
    checkpoint.restore(checkpointRoundTrip, { model: restoredCheckpointModel, optimizer: restoredCheckpointOptimizer });
    expectClose(restoredCheckpointModel.stateDict()["0.weight"].data, checkpointSnapshot.model!["0.weight"].data);
    expectClose(restoredCheckpointModel.stateDict()["0.bias"].data, checkpointSnapshot.model!["0.bias"].data);
    expectClose(restoredCheckpointOptimizer.stateDict().entries[0].data, checkpointSnapshot.optimizer!.entries[0].data);
    if (!Array.isArray(checkpoint.moduleStateDict(restoredCheckpointModel)["0.weight"].data)) {
      throw new Error("expected TS checkpoint module state to be JSON-safe");
    }
    const aliasLinear = new nn.Linear(2, 1, { weights: [1, 2], bias: false });
    expectClose(aliasLinear.forward([3, 4]).toFloat32Array(), [11]);
    const frontendSupport = frontend.compileSupport();
    if (
      !frontend.canCompile() ||
      !frontendSupport.supported ||
      frontendSupport.nativePath !== "tiny-linear" ||
      frontendSupport.modelKind !== "tiny-linear" ||
      frontendSupport.layerCount !== 1 ||
      frontendSupport.inputLen !== 2 ||
      frontendSupport.outputLen !== 3 ||
      frontendSupport.weightsLen !== 6 ||
      frontendSupport.biasLen !== 3
    ) {
      throw new Error(`unexpected TS single-linear compile support: ${json(frontendSupport)}`);
    }
    const rootFrontendSupport = nn.compileSupport(frontend);
    const rootFrontendCompilerSignatures = nn.compilerSignatures(frontend);
    const rootFrontendIr = nn.tensorProgramIr(frontend);
    const rootFrontendKernelPlan = nn.kernelPlan(frontend);
    const rootFrontendBufferLayout = nn.bufferLayout(frontend);
    const rootFrontendMemoryLayout = nn.memoryLayout(frontend);
    const rootFrontendInputShape = nn.inputShape(frontend);
    const rootFrontendOutputShape = nn.outputShape(frontend);
    const rootFrontendShapeConstraints = nn.shapeConstraints(frontend);
    const rootFrontendParameterLayout = nn.parameterLayout(frontend);
    const rootFrontendTrace = nn.trace(frontend);
    if (
      !nn.canCompile(frontend) ||
      rootFrontendSupport.supported !== frontendSupport.supported ||
      rootFrontendSupport.nativePath !== frontendSupport.nativePath ||
      rootFrontendSupport.modelKind !== frontendSupport.modelKind ||
      !rootFrontendCompilerSignatures ||
      rootFrontendCompilerSignatures.ir !== rootFrontendSupport.irSignature ||
      rootFrontendCompilerSignatures.kernelPlan !== rootFrontendSupport.kernelPlanSignature ||
      !rootFrontendIr ||
      rootFrontendIr.opCount !== 1 ||
      rootFrontendIr.inputLen !== 2 ||
      rootFrontendIr.outputLen !== 3 ||
      !rootFrontendKernelPlan ||
      rootFrontendKernelPlan.opCount !== 1 ||
      rootFrontendKernelPlan.inputLen !== 2 ||
      rootFrontendKernelPlan.outputLen !== 3 ||
      !rootFrontendMemoryLayout ||
      !rootFrontendBufferLayout ||
      rootFrontendBufferLayout.input.elementCount !== 2 ||
      rootFrontendBufferLayout.output.elementCount !== 3 ||
      rootFrontendBufferLayout.weights.elementCount !== 6 ||
      rootFrontendBufferLayout.bias.elementCount !== 3 ||
      !Object.isFrozen(rootFrontendInputShape) ||
      !Object.isFrozen(rootFrontendOutputShape) ||
      rootFrontendInputShape.join(",") !== "2" ||
      rootFrontendOutputShape.join(",") !== "3" ||
      !rootFrontendShapeConstraints ||
      rootFrontendShapeConstraints.inputShape.join(",") !== "2" ||
      rootFrontendShapeConstraints.outputShape.join(",") !== "3" ||
      !rootFrontendParameterLayout ||
      rootFrontendParameterLayout.weightsLen !== 6 ||
      rootFrontendParameterLayout.biasLen !== 3 ||
      rootFrontendTrace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0:linear" ||
      rootFrontendTrace.ops[0].parameters[0].name !== "0.weight"
    ) {
      throw new Error(`unexpected TS nn root compile helpers: ${json({ rootFrontendSupport, rootFrontendCompilerSignatures, rootFrontendIr, rootFrontendKernelPlan, rootFrontendMemoryLayout, rootFrontendBufferLayout, rootFrontendInputShape, rootFrontendOutputShape, rootFrontendShapeConstraints, rootFrontendParameterLayout, rootFrontendTrace })}`);
    }
    const rootFrontendProgram = nn.compile(frontend, { backend: "cpu" });
    try {
      const rootFrontendEvidence = rootFrontendProgram.compileEvidence();
      const rootFrontendProgramTrace = rootFrontendProgram.trace();
      if (
        !rootFrontendEvidence ||
        !Object.isFrozen(rootFrontendEvidence) ||
        rootFrontendEvidence.kind !== "tiny-linear" ||
        rootFrontendProgramTrace !== rootFrontendEvidence.trace ||
        rootFrontendProgram.compilerSignatures() !== rootFrontendEvidence.compilerSignatures ||
        rootFrontendProgram.tensorProgramIr() !== rootFrontendEvidence.ir ||
        rootFrontendProgram.kernelPlan() !== rootFrontendEvidence.kernelPlan ||
        rootFrontendProgram.memoryLayout() === null ||
        rootFrontendProgram.shapeConstraints() === null ||
        rootFrontendProgram.parameterLayout() === null ||
        rootFrontendEvidence.nativePath !== "tiny-linear" ||
        rootFrontendEvidence.inputLen !== frontendSupport.inputLen ||
        rootFrontendEvidence.outputLen !== frontendSupport.outputLen ||
        rootFrontendEvidence.weightsLen !== frontendSupport.weightsLen ||
        rootFrontendEvidence.biasLen !== frontendSupport.biasLen ||
        rootFrontendEvidence.trace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0:linear" ||
        !rootFrontendEvidence.ir ||
        !rootFrontendEvidence.kernelPlan
      ) {
        throw new Error(`unexpected TS root single-linear compile evidence: ${json(rootFrontendEvidence)}`);
      }
      const rootFrontendSession = rootFrontendProgram.bind(nn.bindParameters(frontend));
      try {
        expectClose(rootFrontendSession.step(frontendInput), Array.from(frontendHostOutput.toFloat32Array()));
        const rootFrontendTensor = rootFrontendSession.stepTensor(frontendInput);
        if (!(rootFrontendTensor instanceof Tensor) || rootFrontendTensor.shape.join(",") !== "3") {
          throw new Error(`unexpected TS root single-linear stepTensor shape: ${rootFrontendTensor.shape}`);
        }
        expectClose(rootFrontendTensor.toFloat32Array(), Array.from(frontendHostOutput.toFloat32Array()));
      } finally {
        rootFrontendSession.free();
      }
    } finally {
      rootFrontendProgram.free();
    }
    const frontendProgram = frontend.compile({ backend: "cpu" });
    try {
      const frontendEvidence = frontendProgram.compileEvidence();
      if (!frontendEvidence || frontendEvidence.kind !== "tiny-linear" || frontendEvidence.trace.ops[0].op !== "linear") {
        throw new Error(`unexpected TS single-linear compile evidence: ${json(frontendEvidence)}`);
      }
      const frontendSession = frontendProgram.bind(frontend.bindParameters());
      try {
        expectClose(frontendSession.step(frontendInput), Array.from(frontendHostOutput.toFloat32Array()));
      } finally {
        frontendSession.free();
      }
    } finally {
      frontendProgram.free();
    }

    const mismatchedLinearSupport = aliasLinear.compileSupport({ inputShape: [3] });
    const rootMismatchedLinearSupport = nn.compileSupport(aliasLinear, { inputShape: [3] });
    if (
      aliasLinear.canCompile({ inputShape: [3] }) ||
      nn.canCompile(aliasLinear, { inputShape: [3] }) ||
      mismatchedLinearSupport.supported ||
      rootMismatchedLinearSupport.supported ||
      !mismatchedLinearSupport.diagnostics ||
      mismatchedLinearSupport.diagnostics[0].stage !== "trace" ||
      mismatchedLinearSupport.diagnostics[0].code !== "shape-mismatch" ||
      !rootMismatchedLinearSupport.diagnostics ||
      rootMismatchedLinearSupport.diagnostics[0].stage !== "trace" ||
      rootMismatchedLinearSupport.diagnostics[0].code !== "shape-mismatch" ||
      !String(mismatchedLinearSupport.reason).includes("linear trace input shape") ||
      !String(rootMismatchedLinearSupport.reason).includes("linear trace input shape")
    ) {
      throw new Error(`unexpected TS mismatched linear compile support: ${json({ mismatchedLinearSupport, rootMismatchedLinearSupport })}`);
    }
    let mismatchedLinearCompileRejected = false;
    try {
      aliasLinear.compile({ backend: "cpu", inputShape: [3] });
    } catch (err) {
      mismatchedLinearCompileRejected = String((err as Error)?.message || err).includes("linear trace input shape");
    }
    if (!mismatchedLinearCompileRejected) throw new Error("expected TS mismatched linear compile to reject");

    const batchedLinearSupport = aliasLinear.compileSupport({ inputShape: [2, 2] });
    if (
      !aliasLinear.canCompile({ inputShape: [2, 2] }) ||
      !batchedLinearSupport.supported ||
      batchedLinearSupport.reason !== null ||
      batchedLinearSupport.nativePath !== "device-program" ||
      batchedLinearSupport.modelKind !== "module" ||
      batchedLinearSupport.inputLen !== 4 ||
      batchedLinearSupport.outputLen !== 2 ||
      batchedLinearSupport.weightsLen !== 2 ||
      batchedLinearSupport.biasLen !== 0
    ) {
      throw new Error(`unexpected TS batched linear compile support: ${json(batchedLinearSupport)}`);
    }
    const batchedLinearInput = tensor([3, 4, 5, 6], [2, 2]);
    const batchedLinearProgram = aliasLinear.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const requirements = batchedLinearProgram.requirements();
      if (
        requirements.modelKind !== "module" ||
        requirements.inputLen !== 4 ||
        requirements.outputLen !== 2 ||
        requirements.weightsLen !== 2 ||
        requirements.biasLen !== 0
      ) {
        throw new Error(`unexpected TS batched linear requirements: ${json(requirements)}`);
      }
      const batchedLinearSession = batchedLinearProgram.bind(
        aliasLinear.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const eager = aliasLinear.forward(batchedLinearInput);
        expectCloseWithin(
          batchedLinearSession.step(batchedLinearInput),
          Array.from(eager.toFloat32Array()),
          1e-6,
        );
      } finally {
        batchedLinearSession.free();
      }
    } finally {
      batchedLinearProgram.free();
    }

    const flattenLinear = nn.sequential([
      new nn.Flatten(),
      new nn.Linear(4, 2, {
        weights: [
          1, 10,
          2, 20,
          3, 30,
          4, 40,
        ],
        bias: false,
      }),
    ]);
    const flattenLinearSupport = flattenLinear.compileSupport({ inputShape: [2, 2] });
    if (
      !flattenLinear.canCompile({ inputShape: [2, 2] }) ||
      !flattenLinearSupport.supported ||
      flattenLinearSupport.modelKind !== "module" ||
      flattenLinearSupport.inputLen !== 4 ||
      flattenLinearSupport.outputLen !== 2 ||
      flattenLinearSupport.weightsLen !== 8 ||
      flattenLinearSupport.biasLen !== 0 ||
      flattenLinearSupport.trace!.ops.map((op) => op.op).join("|") !== "flatten|linear" ||
      flattenLinearSupport.trace!.ops[1].parameters[0].name !== "1.weight"
    ) {
      throw new Error(`unexpected TS flatten-linear compile support: ${json(flattenLinearSupport)}`);
    }
    const flattenLinearInput = tensor([1, 2, 3, 4], [2, 2]);
    const flattenLinearProgram = flattenLinear.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const flattenLinearSession = flattenLinearProgram.bind(
        flattenLinear.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const eager = flattenLinear.forward(flattenLinearInput);
        expectCloseWithin(
          flattenLinearSession.step(flattenLinearInput),
          Array.from(eager.toFloat32Array()),
          1e-6,
        );
      } finally {
        flattenLinearSession.free();
      }
    } finally {
      flattenLinearProgram.free();
    }

    const reluCtor = new nn.ReLU();
    const geluCtor = new nn.GELU();
    const siluCtor = new nn.SiLU();
    const sigmoidCtor = new nn.Sigmoid();
    if (
      reluCtor.kind !== "relu" ||
      geluCtor.kind !== "gelu" ||
      siluCtor.kind !== "silu" ||
      sigmoidCtor.kind !== "sigmoid"
    ) {
      throw new Error("unexpected TS activation constructor aliases");
    }
    if (nn.relu().kind !== "relu") throw new Error("unexpected TS nn.relu factory alias");
    expectClose(reluCtor.forward(tensor([-1, 2], [2])).toFloat32Array(), [0, 2]);
    expectClose(sigmoidCtor.forward(tensor([0], [1])).toFloat32Array(), [0.5]);
    const standaloneRelu = reluCtor;
    const standaloneReluSupport = standaloneRelu.compileSupport({ inputShape: [2, 2] });
    if (
      !standaloneRelu.canCompile({ inputShape: [2, 2] }) ||
      !standaloneReluSupport.supported ||
      standaloneReluSupport.modelKind !== "module" ||
      standaloneReluSupport.inputLen !== 4 ||
      standaloneReluSupport.outputLen !== 4 ||
      standaloneReluSupport.weightsLen !== 0 ||
      standaloneReluSupport.biasLen !== 0
    ) {
      throw new Error(`unexpected TS standalone relu compile support: ${json(standaloneReluSupport)}`);
    }
    const standaloneReluProgram = standaloneRelu.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const standaloneReluSession = standaloneReluProgram.bind(
        standaloneRelu.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([-1, 2, -3, 4], [2, 2]);
        const eager = standaloneRelu.forward(input);
        expectCloseWithin(standaloneReluSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        standaloneReluSession.free();
      }
    } finally {
      standaloneReluProgram.free();
    }

    const standaloneSigmoid = nn.sigmoid();
    const standaloneSigmoidSupport = standaloneSigmoid.compileSupport({ inputShape: [2, 2] });
    if (
      !standaloneSigmoid.canCompile({ inputShape: [2, 2] }) ||
      !standaloneSigmoidSupport.supported ||
      standaloneSigmoidSupport.modelKind !== "module" ||
      standaloneSigmoidSupport.inputLen !== 4 ||
      standaloneSigmoidSupport.outputLen !== 4 ||
      standaloneSigmoidSupport.weightsLen !== 0 ||
      standaloneSigmoidSupport.biasLen !== 0 ||
      standaloneSigmoidSupport.trace?.ops[0].activation !== "sigmoid"
    ) {
      throw new Error(`unexpected TS standalone sigmoid compile support: ${json(standaloneSigmoidSupport)}`);
    }
    const standaloneSigmoidProgram = standaloneSigmoid.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const standaloneSigmoidSession = standaloneSigmoidProgram.bind(
        standaloneSigmoid.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([-1, 0, 2, 4], [2, 2]);
        const eager = standaloneSigmoid.forward(input);
        expectCloseWithin(standaloneSigmoidSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        standaloneSigmoidSession.free();
      }
    } finally {
      standaloneSigmoidProgram.free();
    }

    const unaryChain = nn.sequential([
      nn.neg(),
      nn.exp(),
      nn.log(),
      nn.abs(),
      nn.sqrt(),
      nn.square(),
      nn.sign(),
      nn.step(),
      nn.recip(),
    ]);
    const unaryChainOps = "neg|exp|log|abs|sqrt|square|sgn|step|recip";
    const unaryChainPlanOps = "activation-chain:neg+exp+log+abs:4|activation-chain:sqrt+square+sgn+step:4|activation:recip:1";
    const summarizeUnaryChainPlan = (plan) => plan.ops
      .map((op) => `${op.op}:${op.nativeKernels.join("+")}:${op.fusedOpCount ?? 1}`)
      .join("|");
    const unaryChainSupport = unaryChain.compileSupport({ inputShape: [2, 2] });
    if (
      !unaryChain.canCompile({ inputShape: [2, 2] }) ||
      !unaryChainSupport.supported ||
      unaryChainSupport.modelKind !== "module" ||
      unaryChainSupport.inputLen !== 4 ||
      unaryChainSupport.outputLen !== 4 ||
      unaryChainSupport.weightsLen !== 0 ||
      unaryChainSupport.biasLen !== 0 ||
      unaryChainSupport.trace?.ops.map((op) => op.activation).join("|") !== unaryChainOps ||
      !unaryChainSupport.ir ||
      unaryChainSupport.ir.ops.map((op) => op.attrs.activation).join("|") !== unaryChainOps ||
      unaryChainSupport.ir.ops.some((op) => "activation" in op || "parameters" in op) ||
      !unaryChainSupport.kernelPlan ||
      unaryChainSupport.kernelPlan.dispatchCount !== 3 ||
      unaryChainSupport.kernelPlan.descriptorCount !== 3 ||
      summarizeUnaryChainPlan(unaryChainSupport.kernelPlan) !== unaryChainPlanOps ||
      unaryChainSupport.kernelPlan.ops[0].fusedValueEdges.length !== 4 ||
      unaryChainSupport.kernelPlan.ops[1].fusedValueEdges.length !== 4
    ) {
      throw new Error(`unexpected TS unary-chain compile support: ${json(unaryChainSupport)}`);
    }
    const unaryChainProgram = unaryChain.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const unaryEvidence = unaryChainProgram.compileEvidence();
      if (
        !unaryEvidence ||
        unaryEvidence.kind !== "module" ||
        unaryEvidence.trace.ops.map((op) => op.activation).join("|") !== unaryChainOps ||
        unaryEvidence.ir.ops.map((op) => op.attrs.activation).join("|") !== unaryChainOps ||
        unaryEvidence.ir.ops.some((op) => "activation" in op || "parameters" in op) ||
        unaryEvidence.kernelPlan.dispatchCount !== 3 ||
        unaryEvidence.kernelPlan.descriptorCount !== 3 ||
        summarizeUnaryChainPlan(unaryEvidence.kernelPlan) !== unaryChainPlanOps
      ) {
        throw new Error(`unexpected TS unary-chain compile evidence: ${json(unaryEvidence)}`);
      }
      const unaryChainSession = unaryChainProgram.bind(unaryChain.bindParameters({ inputShape: [2, 2] }));
      try {
        const input = tensor([-1, -2, -4, -0.5], [2, 2]);
        const eager = unaryChain.forward(input);
        if (!(eager instanceof Tensor)) {
          throw new Error("expected TS unary-chain eager output to be a Tensor");
        }
        expectCloseWithin(unaryChainSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        unaryChainSession.free();
      }
    } finally {
      unaryChainProgram.free();
    }

    const standaloneSoftmax = nn.softmax();
    const standaloneSoftmaxProgram = standaloneSoftmax.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const standaloneSoftmaxSession = standaloneSoftmaxProgram.bind(
        standaloneSoftmax.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([1, 2, 3, 1], [2, 2]);
        const eager = standaloneSoftmax.forward(input);
        expectCloseWithin(standaloneSoftmaxSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        standaloneSoftmaxSession.free();
      }
    } finally {
      standaloneSoftmaxProgram.free();
    }

    const batchSoftmax = nn.softmax(0);
    const batchSoftmaxSupport = batchSoftmax.compileSupport({ inputShape: [2, 2] });
    if (
      !batchSoftmax.canCompile({ inputShape: [2, 2] }) ||
      !batchSoftmaxSupport.supported ||
      batchSoftmaxSupport.trace?.ops[0].op !== "softmax" ||
      !batchSoftmaxSupport.ir ||
      batchSoftmaxSupport.ir.ops[0].op !== "softmax" ||
      !batchSoftmaxSupport.kernelPlan ||
      batchSoftmaxSupport.kernelPlan.ops[0].op !== "softmax" ||
      batchSoftmaxSupport.kernelPlan.ops[0].kernel !== "softmax" ||
      batchSoftmaxSupport.kernelPlan.dispatchCount !== 3
    ) {
      throw new Error(`unexpected TS batch-axis softmax compiler evidence: ${json(batchSoftmaxSupport)}`);
    }
    const batchSoftmaxProgram = batchSoftmax.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const batchSoftmaxSession = batchSoftmaxProgram.bind(batchSoftmax.bindParameters({ inputShape: [2, 2] }));
      try {
        const input = tensor([1, 2, 3, 1], [2, 2]);
        const eager = batchSoftmax.forward(input);
        expectCloseWithin(batchSoftmaxSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        batchSoftmaxSession.free();
      }
    } finally {
      batchSoftmaxProgram.free();
    }

    const standaloneLogSoftmax = nn.logSoftmax();
    const standaloneLogSoftmaxSupport = standaloneLogSoftmax.compileSupport({ inputShape: [2, 2] });
    if (
      !standaloneLogSoftmax.canCompile({ inputShape: [2, 2] }) ||
      !standaloneLogSoftmaxSupport.supported ||
      standaloneLogSoftmaxSupport.modelKind !== "module" ||
      standaloneLogSoftmaxSupport.inputLen !== 4 ||
      standaloneLogSoftmaxSupport.outputLen !== 4 ||
      standaloneLogSoftmaxSupport.weightsLen !== 0 ||
      standaloneLogSoftmaxSupport.biasLen !== 0 ||
      standaloneLogSoftmaxSupport.trace?.ops[0].op !== "logSoftmax"
    ) {
      throw new Error(`unexpected TS standalone logSoftmax compile support: ${json(standaloneLogSoftmaxSupport)}`);
    }
    const standaloneLogSoftmaxProgram = standaloneLogSoftmax.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const standaloneLogSoftmaxSession = standaloneLogSoftmaxProgram.bind(
        standaloneLogSoftmax.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([1, 2, 3, 1], [2, 2]);
        const eager = standaloneLogSoftmax.forward(input);
        expectCloseWithin(standaloneLogSoftmaxSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        standaloneLogSoftmaxSession.free();
      }
    } finally {
      standaloneLogSoftmaxProgram.free();
    }

    const batchLogSoftmax = nn.logSoftmax(0);
    const batchLogSoftmaxSupport = batchLogSoftmax.compileSupport({ inputShape: [2, 2] });
    if (
      !batchLogSoftmax.canCompile({ inputShape: [2, 2] }) ||
      !batchLogSoftmaxSupport.supported ||
      batchLogSoftmaxSupport.trace?.ops[0].op !== "logSoftmax" ||
      !batchLogSoftmaxSupport.ir ||
      batchLogSoftmaxSupport.ir.ops[0].op !== "logSoftmax" ||
      !batchLogSoftmaxSupport.kernelPlan ||
      batchLogSoftmaxSupport.kernelPlan.ops[0].op !== "logSoftmax" ||
      batchLogSoftmaxSupport.kernelPlan.ops[0].kernel !== "log-softmax" ||
      batchLogSoftmaxSupport.kernelPlan.dispatchCount !== 3
    ) {
      throw new Error(`unexpected TS batch-axis logSoftmax compiler evidence: ${json(batchLogSoftmaxSupport)}`);
    }
    const batchLogSoftmaxProgram = batchLogSoftmax.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const batchLogSoftmaxSession = batchLogSoftmaxProgram.bind(batchLogSoftmax.bindParameters({ inputShape: [2, 2] }));
      try {
        const input = tensor([1, 2, 3, 1], [2, 2]);
        const eager = batchLogSoftmax.forward(input);
        expectCloseWithin(batchLogSoftmaxSession.step(input), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        batchLogSoftmaxSession.free();
      }
    } finally {
      batchLogSoftmaxProgram.free();
    }

    const featureSum = nn.sum();
    const featureSumSupport = featureSum.compileSupport({ inputShape: [2, 3] });
    if (
      !featureSum.canCompile({ inputShape: [2, 3] }) ||
      !featureSumSupport.supported ||
      featureSumSupport.trace?.ops[0].op !== "sum" ||
      !featureSumSupport.ir ||
      featureSumSupport.ir.ops[0].op !== "sum" ||
      !featureSumSupport.kernelPlan ||
      featureSumSupport.kernelPlan.ops[0].op !== "sum" ||
      featureSumSupport.kernelPlan.ops[0].kernel !== "sum" ||
      featureSumSupport.kernelPlan.ops[0].nativeDispatchCount !== 1 ||
      featureSumSupport.kernelPlan.ops[0].nativeKernels.join(",") !== "sum" ||
      featureSumSupport.kernelPlan.dispatchCount !== 1 ||
      featureSumSupport.kernelPlan.outputShape.join(",") !== "2,1" ||
      featureSumSupport.outputLen !== 2
    ) {
      throw new Error(`unexpected TS feature-axis sum compiler evidence: ${json(featureSumSupport)}`);
    }
    const featureSumProgram = featureSum.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const featureSumSession = featureSumProgram.bind(featureSum.bindParameters({ inputShape: [2, 3] }));
      try {
        const input = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        expectCloseWithin(featureSumSession.step(input), [6, 15], 1e-6);
      } finally {
        featureSumSession.free();
      }
    } finally {
      featureSumProgram.free();
    }

    const batchMean = nn.mean(0);
    const batchMeanSupport = batchMean.compileSupport({ inputShape: [2, 3] });
    if (
      !batchMean.canCompile({ inputShape: [2, 3] }) ||
      !batchMeanSupport.supported ||
      batchMeanSupport.trace?.ops[0].op !== "mean" ||
      !batchMeanSupport.ir ||
      batchMeanSupport.ir.ops[0].op !== "mean" ||
      !batchMeanSupport.kernelPlan ||
      batchMeanSupport.kernelPlan.ops[0].op !== "mean" ||
      batchMeanSupport.kernelPlan.ops[0].kernel !== "mean" ||
      batchMeanSupport.kernelPlan.ops[0].nativeDispatchCount !== 3 ||
      batchMeanSupport.kernelPlan.ops[0].nativeKernels.join(",") !== "transpose,mean,transpose" ||
      batchMeanSupport.kernelPlan.dispatchCount !== 3 ||
      batchMeanSupport.kernelPlan.outputShape.join(",") !== "1,3" ||
      batchMeanSupport.outputLen !== 3
    ) {
      throw new Error(`unexpected TS batch-axis mean compiler evidence: ${json(batchMeanSupport)}`);
    }
    const batchMeanProgram = batchMean.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const batchMeanSession = batchMeanProgram.bind(batchMean.bindParameters({ inputShape: [2, 3] }));
      try {
        const input = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        expectCloseWithin(batchMeanSession.step(input), [2.5, 3.5, 4.5], 1e-6);
      } finally {
        batchMeanSession.free();
      }
    } finally {
      batchMeanProgram.free();
    }

    const batchMax = nn.max(0);
    const batchMaxSupport = batchMax.compileSupport({ inputShape: [2, 3] });
    if (
      !batchMax.canCompile({ inputShape: [2, 3] }) ||
      !batchMaxSupport.supported ||
      batchMaxSupport.trace?.ops[0].op !== "max" ||
      !batchMaxSupport.ir ||
      batchMaxSupport.ir.ops[0].op !== "max" ||
      !batchMaxSupport.kernelPlan ||
      batchMaxSupport.kernelPlan.ops[0].op !== "max" ||
      batchMaxSupport.kernelPlan.ops[0].kernel !== "max" ||
      batchMaxSupport.kernelPlan.ops[0].nativeDispatchCount !== 3 ||
      batchMaxSupport.kernelPlan.ops[0].nativeKernels.join(",") !== "transpose,max,transpose" ||
      batchMaxSupport.kernelPlan.dispatchCount !== 3 ||
      batchMaxSupport.kernelPlan.outputShape.join(",") !== "1,3" ||
      batchMaxSupport.outputLen !== 3
    ) {
      throw new Error(`unexpected TS batch-axis max compiler evidence: ${json(batchMaxSupport)}`);
    }
    const batchMaxProgram = batchMax.compile({ backend: "cpu", inputShape: [2, 3] });
    try {
      const batchMaxSession = batchMaxProgram.bind(batchMax.bindParameters({ inputShape: [2, 3] }));
      try {
        const input = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        expectCloseWithin(batchMaxSession.step(input), [4, 5, 6], 1e-6);
      } finally {
        batchMaxSession.free();
      }
    } finally {
      batchMaxProgram.free();
    }

    const webgpuLogSoftmaxProgram = standaloneLogSoftmax.compile({ backend: "webgpu", inputShape: [2, 2] });
    try {
      const webgpuLogSoftmaxInspection = webgpuLogSoftmaxProgram.inspect();
      const webgpuLogSoftmaxCapabilities = webgpuLogSoftmaxProgram.capabilities();
      if (
        webgpuLogSoftmaxInspection.backend !== "webgpu" ||
        webgpuLogSoftmaxInspection.executionSupported !== runtimeReportsNativeWgpu ||
        !webgpuLogSoftmaxInspection.externalResourcesSupported ||
        webgpuLogSoftmaxInspection.opCount !== 1 ||
        webgpuLogSoftmaxInspection.commandCount !== 1 ||
        webgpuLogSoftmaxInspection.commandStencilHash === 0n ||
        webgpuLogSoftmaxInspection.dispatchPlanRowCount !== 1 ||
        commandCategoryTotal(webgpuLogSoftmaxInspection) !== 1
      ) {
        throw new Error(`unexpected TS standalone logSoftmax WebGPU inspection: ${json(webgpuLogSoftmaxInspection)}`);
      }
      if (
        webgpuLogSoftmaxCapabilities.backend !== "webgpu" ||
        webgpuLogSoftmaxCapabilities.mode !== (runtimeReportsNativeWgpu ? "executable" : "resource-probe") ||
        webgpuLogSoftmaxCapabilities.canExecute !== runtimeReportsNativeWgpu ||
        webgpuLogSoftmaxProgram.canExecute() !== runtimeReportsNativeWgpu ||
        webgpuLogSoftmaxProgram.executionMode() !== (runtimeReportsNativeWgpu ? "executable" : "resource-probe") ||
        !webgpuLogSoftmaxCapabilities.canBindExternalResources ||
        !webgpuLogSoftmaxProgram.canBindExternalResources() ||
        webgpuLogSoftmaxCapabilities.hasFullDispatchPlan !== runtimeReportsNativeWgpu ||
        webgpuLogSoftmaxProgram.hasFullDispatchPlan() !== runtimeReportsNativeWgpu
      ) {
        throw new Error(`unexpected TS standalone logSoftmax WebGPU capabilities: ${json(webgpuLogSoftmaxCapabilities)}`);
      }
      expectCapabilityDispatchPlan(webgpuLogSoftmaxCapabilities, webgpuLogSoftmaxInspection, "WebGPU standalone logSoftmax");

      if (webgpuLogSoftmaxInspection.executionSupported) {
        const webgpuLogSoftmaxSession = webgpuLogSoftmaxProgram.bind(
          standaloneLogSoftmax.bindParameters({ inputShape: [2, 2] }),
        );
        try {
          const input = tensor([1, 2, 3, 1], [2, 2]);
          const eager = standaloneLogSoftmax.forward(input);
          expectCloseWithin(webgpuLogSoftmaxSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
          const sessionInfo = webgpuLogSoftmaxSession.inspect();
          if (
            sessionInfo.modelKind !== "module" ||
            sessionInfo.backend !== "webgpu" ||
            sessionInfo.outputStorage !== "host" ||
            sessionInfo.persistentBindingCount !== 0 ||
            sessionInfo.stepInputCount !== 1 ||
            sessionInfo.stepOutputCount !== 1 ||
            sessionInfo.hostBindingCount !== 2 ||
            sessionInfo.resourceBindingCount !== 0 ||
            sessionInfo.bindingShapeHash === 0n
          ) {
            throw new Error(`unexpected executable standalone logSoftmax WebGPU session inspection: ${json(sessionInfo)}`);
          }
          const profile = webgpuLogSoftmaxSession.runtimeProfile();
          if (profile.callCount !== 1 || profile.backendOpCount !== 1 || profile.backendDispatchCount !== 1 || profile.syncCount !== 1) {
            throw new Error(`unexpected executable standalone logSoftmax WebGPU profile: ${json(profile)}`);
          }
        } finally {
          webgpuLogSoftmaxSession.free();
        }

        const webgpuLogSoftmaxDevice = webgpuLogSoftmaxProgram.device("webgpu");
        const webgpuLogSoftmaxDeviceInput = webgpuLogSoftmaxDevice.createInputBuffer();
        const webgpuLogSoftmaxDeviceOutput = webgpuLogSoftmaxDevice.createOutputBuffer();
        try {
          const deviceInput = tensor([3, 1, 2, 0], [2, 2]);
          const deviceExpected = Array.from(standaloneLogSoftmax.forward(deviceInput).toFloat32Array());
          webgpuLogSoftmaxDeviceInput.writeFloat32(deviceInput.toFloat32Array());
          const webgpuLogSoftmaxDeviceSession = webgpuLogSoftmaxProgram.bind({
            weights: new Float32Array(0),
            input: webgpuLogSoftmaxDeviceInput,
            output: webgpuLogSoftmaxDeviceOutput,
          });
          try {
            const deviceSessionInfo = webgpuLogSoftmaxDeviceSession.inspect();
            if (
              deviceSessionInfo.modelKind !== "module" ||
              deviceSessionInfo.backend !== "webgpu" ||
              deviceSessionInfo.outputStorage !== "external-resource" ||
              deviceSessionInfo.persistentBindingCount !== 0 ||
              deviceSessionInfo.stepInputCount !== 1 ||
              deviceSessionInfo.stepOutputCount !== 1 ||
              deviceSessionInfo.hostBindingCount !== 0 ||
              deviceSessionInfo.resourceBindingCount !== 2 ||
              deviceSessionInfo.bindingShapeHash === 0n
            ) {
              throw new Error(`unexpected executable standalone logSoftmax WebGPU device-buffer session inspection: ${json(deviceSessionInfo)}`);
            }
            expectCloseWithin(webgpuLogSoftmaxDeviceSession.step(), deviceExpected, 1e-5);
            const deviceProfile = webgpuLogSoftmaxDeviceSession.runtimeProfile();
            if (deviceProfile.callCount !== 1 || deviceProfile.backendOpCount !== 1 || deviceProfile.backendDispatchCount !== 1 || deviceProfile.syncCount !== 0) {
              throw new Error(`unexpected executable standalone logSoftmax WebGPU device-buffer profile: ${json(deviceProfile)}`);
            }
          } finally {
            webgpuLogSoftmaxDeviceSession.free();
          }
        } finally {
          webgpuLogSoftmaxDeviceOutput.free();
          webgpuLogSoftmaxDeviceInput.free();
        }
      } else {
        let webgpuLogSoftmaxBindRejected = false;
        try {
          webgpuLogSoftmaxProgram.bind(standaloneLogSoftmax.bindParameters({ inputShape: [2, 2] }));
        } catch (err) {
          webgpuLogSoftmaxBindRejected = isUnsupportedError(err);
        }
        if (!webgpuLogSoftmaxBindRejected) throw new Error("expected compile-only standalone logSoftmax WebGPU host bind to be unsupported");
      }
    } finally {
      webgpuLogSoftmaxProgram.free();
    }

    const classifierHead = nn.sequential([
      nn.linear(2, 3, {
        weight: [0.5, -1, 1, 1, 0.25, -0.75],
        bias: [0.1, -0.2, 0.3],
      }),
      nn.logSoftmax(),
    ]);
    const classifierHeadSupport = classifierHead.compileSupport({ inputShape: [2, 2] });
    if (
      !classifierHeadSupport.supported ||
      classifierHeadSupport.modelKind !== "module" ||
      classifierHeadSupport.layerCount !== 2 ||
      classifierHeadSupport.inputLen !== 4 ||
      classifierHeadSupport.outputLen !== 6 ||
      classifierHeadSupport.weightsLen !== 6 ||
      classifierHeadSupport.biasLen !== 3 ||
      classifierHeadSupport.trace?.ops.length !== 2 ||
      classifierHeadSupport.trace.ops[0].op !== "linear" ||
      classifierHeadSupport.trace.ops[1].op !== "logSoftmax"
    ) {
      throw new Error(`unexpected TS classifier head compile support: ${json(classifierHeadSupport)}`);
    }
    const classifierHeadProgram = classifierHead.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      if (classifierHeadProgram.inputShape().join(",") !== "2,2" || classifierHeadProgram.outputShape().join(",") !== "2,3") {
        throw new Error(`unexpected TS classifier head Program shapes: ${classifierHeadProgram.inputShape()} -> ${classifierHeadProgram.outputShape()}`);
      }
      const classifierHeadSession = classifierHeadProgram.bind(
        classifierHead.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const classifierSessionLayout = classifierHeadSession.bufferLayout();
        if (classifierHeadSession.inputShape().join(",") !== "2,2" || classifierHeadSession.outputShape().join(",") !== "2,3") {
          throw new Error(`unexpected TS classifier head Session shapes: ${classifierHeadSession.inputShape()} -> ${classifierHeadSession.outputShape()}`);
        }
        if (
          !Object.isFrozen(classifierSessionLayout) ||
          !Object.isFrozen(classifierSessionLayout.slots) ||
          classifierSessionLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== "input:step-input:4:16|output:step-output:6:24|weights:persistent:6:24|bias:persistent:3:12"
        ) {
          throw new Error(`unexpected TS classifier head Session buffer layout: ${json(classifierSessionLayout)}`);
        }
        const input = tensor([1, 2, -1, 0.5], [2, 2]);
        const eager = classifierHead.forward(input);
        expectCloseWithin(classifierHeadSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
        const tensorOutput = classifierHeadSession.stepTensor(input);
        if (!(tensorOutput instanceof Tensor) || tensorOutput.shape.join(",") !== "2,3") {
          throw new Error(`unexpected TS classifier head stepTensor shape: ${tensorOutput.shape}`);
        }
        expectCloseWithin(tensorOutput.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
        const flatTensorOutput = classifierHeadSession.stepTensor(input, { shape: [6] });
        if (flatTensorOutput.shape.join(",") !== "6") {
          throw new Error(`unexpected TS classifier head explicit stepTensor shape: ${flatTensorOutput.shape}`);
        }
        expectCloseWithin(flatTensorOutput.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
        const shapedOutputBuffer = tensor(new Float32Array(6), [2, 3]);
        const shapedTensorOutput = classifierHeadSession.stepTensor(input, { output: shapedOutputBuffer });
        if (shapedTensorOutput.shape.join(",") !== "2,3") {
          throw new Error(`unexpected TS classifier head output Tensor stepTensor shape: ${shapedTensorOutput.shape}`);
        }
        expectCloseWithin(shapedTensorOutput.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
        expectCloseWithin(shapedOutputBuffer.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
        const boundShapedOutput = tensor(new Float32Array(6), [2, 3]);
        const boundOutputSession = classifierHeadProgram.bind({
          ...classifierHead.bindParameters({ inputShape: [2, 2] }),
          output: boundShapedOutput,
        });
        try {
          const boundTensorOutput = boundOutputSession.stepTensor(input);
          if (boundTensorOutput.shape.join(",") !== "2,3") {
            throw new Error(`unexpected TS classifier head bound output Tensor stepTensor shape: ${boundTensorOutput.shape}`);
          }
          expectCloseWithin(boundTensorOutput.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
          expectCloseWithin(boundShapedOutput.toFloat32Array(), Array.from(eager.toFloat32Array()), 1e-5);
        } finally {
          boundOutputSession.free();
        }
      } finally {
        classifierHeadSession.free();
      }
    } finally {
      classifierHeadProgram.free();
    }

    const tokenClassifier = nn.sequential([
      nn.embedding(5, 2, {
        weight: [0, 1, 1, 0, 2, -1, -0.5, 0.25, 0.75, 1.5],
      }),
      nn.linear(2, 3, {
        weight: [1, -1, 0.5, 0.25, 0.75, -0.5],
        bias: [0.1, 0.2, -0.3],
      }),
      nn.logSoftmax(),
    ]);
    const tokenClassifierSupport = tokenClassifier.compileSupport({ inputShape: [3] });
    if (
      !tokenClassifierSupport.supported ||
      tokenClassifierSupport.modelKind !== "module" ||
      tokenClassifierSupport.layerCount !== 3 ||
      tokenClassifierSupport.inputLen !== 3 ||
      tokenClassifierSupport.outputLen !== 9 ||
      tokenClassifierSupport.weightsLen !== 16 ||
      tokenClassifierSupport.biasLen !== 3 ||
      tokenClassifierSupport.trace?.ops.length !== 3 ||
      tokenClassifierSupport.trace.ops[0].op !== "embedding" ||
      tokenClassifierSupport.trace.ops[1].op !== "linear" ||
      tokenClassifierSupport.trace.ops[2].op !== "logSoftmax"
    ) {
      throw new Error(`unexpected TS token classifier compile support: ${json(tokenClassifierSupport)}`);
    }
    const tokenClassifierProgram = tokenClassifier.compile({ backend: "cpu", inputShape: [3] });
    try {
      const tokenClassifierSession = tokenClassifierProgram.bind(
        tokenClassifier.bindParameters({ inputShape: [3] }),
      );
      try {
        const input = new Uint32Array([0, 2, 4]);
        const eager = tokenClassifier.forward(input);
        expectCloseWithin(tokenClassifierSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
      } finally {
        tokenClassifierSession.free();
      }
    } finally {
      tokenClassifierProgram.free();
    }

    const standaloneLayerNorm = nn.layerNorm(2, {
      weight: [1, 1.5],
      bias: [0.1, -0.2],
    });
    const standaloneLayerNormSupport = standaloneLayerNorm.compileSupport({ inputShape: [2] });
    const batchedLayerNormSupport = standaloneLayerNorm.compileSupport({ inputShape: [2, 2] });
    if (
      !standaloneLayerNorm.canCompile({ inputShape: [2] }) ||
      !standaloneLayerNorm.canCompile({ inputShape: [2, 2] }) ||
      !batchedLayerNormSupport.supported ||
      batchedLayerNormSupport.inputLen !== 4 ||
      batchedLayerNormSupport.outputLen !== 4 ||
      !standaloneLayerNormSupport.supported ||
      standaloneLayerNormSupport.modelKind !== "module" ||
      standaloneLayerNormSupport.inputLen !== 2 ||
      standaloneLayerNormSupport.outputLen !== 2 ||
      standaloneLayerNormSupport.weightsLen !== 2 ||
      standaloneLayerNormSupport.biasLen !== 2
    ) {
      throw new Error(`unexpected TS standalone LayerNorm compile support: ${json(standaloneLayerNormSupport)}`);
    }
    const standaloneLayerNormProgram = standaloneLayerNorm.compile({ backend: "cpu", inputShape: [2] });
    try {
      const standaloneLayerNormSession = standaloneLayerNormProgram.bind(
        standaloneLayerNorm.bindParameters({ inputShape: [2] }),
      );
      try {
        const input = tensor([1, 2], [2]);
        const eager = standaloneLayerNorm.forward(input);
        expectCloseWithin(standaloneLayerNormSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
      } finally {
        standaloneLayerNormSession.free();
      }
    } finally {
      standaloneLayerNormProgram.free();
    }
    const batchedLayerNormProgram = standaloneLayerNorm.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const batchedLayerNormSession = batchedLayerNormProgram.bind(
        standaloneLayerNorm.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([1, 2, 3, 1], [2, 2]);
        const eager = standaloneLayerNorm.forward(input);
        expectCloseWithin(batchedLayerNormSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
      } finally {
        batchedLayerNormSession.free();
      }
    } finally {
      batchedLayerNormProgram.free();
    }
    const standaloneRmsNorm = nn.rmsNorm(2, { eps: 0, weights: [1, 2] });
    const batchedRmsNormSupport = standaloneRmsNorm.compileSupport({ inputShape: [2, 2] });
    if (
      !standaloneRmsNorm.canCompile({ inputShape: [2, 2] }) ||
      !batchedRmsNormSupport.supported ||
      batchedRmsNormSupport.modelKind !== "module" ||
      batchedRmsNormSupport.inputLen !== 4 ||
      batchedRmsNormSupport.outputLen !== 4 ||
      batchedRmsNormSupport.weightsLen !== 2 ||
      batchedRmsNormSupport.biasLen !== 0
    ) {
      throw new Error(`unexpected TS batched RMSNorm compile support: ${json(batchedRmsNormSupport)}`);
    }
    const batchedRmsNormProgram = standaloneRmsNorm.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const batchedRmsNormSession = batchedRmsNormProgram.bind(
        standaloneRmsNorm.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const input = tensor([3, 4, 1, 2], [2, 2]);
        const eager = standaloneRmsNorm.forward(input);
        expectCloseWithin(batchedRmsNormSession.step(input), Array.from(eager.toFloat32Array()), 1e-5);
      } finally {
        batchedRmsNormSession.free();
      }
    } finally {
      batchedRmsNormProgram.free();
    }

    const nestedLinear = nn.sequential([
      nn.sequential([
        nn.linear(2, 1, { weights: [2, -1], bias: [0.25] }),
      ]),
    ]);
    const nestedLinearSupport = nestedLinear.compileSupport();
    if (
      !nestedLinear.canCompile() ||
      !nestedLinearSupport.supported ||
      nestedLinearSupport.nativePath !== "tiny-linear" ||
      nestedLinearSupport.modelKind !== "tiny-linear" ||
      nestedLinearSupport.layerCount !== 1 ||
      nestedLinearSupport.trace?.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0.0:linear" ||
      nestedLinearSupport.trace.ops[0].parameters[0].name !== "0.0.weight" ||
      !nestedLinearSupport.ir ||
      nestedLinearSupport.ir.ops[0].path !== "0.0" ||
      !nestedLinearSupport.kernelPlan ||
      nestedLinearSupport.kernelPlan.ops[0].path !== "0.0"
    ) {
      throw new Error(`unexpected TS nested single-linear compile support: ${json(nestedLinearSupport)}`);
    }
    const nestedLinearProgram = nestedLinear.compile({ backend: "cpu" });
    try {
      const nestedLinearEvidence = nestedLinearProgram.compileEvidence();
      if (
        !nestedLinearEvidence ||
        nestedLinearEvidence.kind !== "tiny-linear" ||
        nestedLinearEvidence.trace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0.0:linear" ||
        nestedLinearEvidence.trace.ops[0].parameters[0].name !== "0.0.weight" ||
        nestedLinearProgram.tensorProgramIr() !== nestedLinearEvidence.ir ||
        nestedLinearProgram.kernelPlan() !== nestedLinearEvidence.kernelPlan ||
        nestedLinearProgram.compilerSignatures() !== nestedLinearEvidence.compilerSignatures
      ) {
        throw new Error(`unexpected TS nested single-linear compile evidence: ${json(nestedLinearEvidence)}`);
      }
      const nestedLinearCompatibility = nestedLinearProgram.moduleCompatibility(nestedLinear);
      if (!nestedLinearCompatibility.compatible || nestedLinearProgram.acceptsModule(nestedLinear) !== true) {
        throw new Error(`expected TS nested single-linear Program to accept its source module: ${json(nestedLinearCompatibility)}`);
      }
      const bareLinearSameShape = nn.linear(2, 1, { weights: [2, -1], bias: [0.25] });
      const bareLinearCompatibility = nestedLinearProgram.moduleCompatibility(bareLinearSameShape);
      if (
        bareLinearCompatibility.compatible ||
        bareLinearCompatibility.diagnostics[0]?.code !== "ir-mismatch" ||
        bareLinearCompatibility.diagnostics[0].signatureKind !== "ir" ||
        nestedLinearProgram.acceptsModule(bareLinearSameShape) !== false
      ) {
        throw new Error(`expected TS nested single-linear Program to reject same-shape bare Linear: ${json(bareLinearCompatibility)}`);
      }
      const nestedLinearSession = nestedLinearProgram.bind(nestedLinear.bindParameters());
      try {
        const eager = nestedLinear.forward(frontendInput);
        if (!(eager instanceof Tensor)) {
          throw new Error("expected TS nested single-linear eager output to be a Tensor");
        }
        expectCloseWithin(nestedLinearSession.step(frontendInput), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        nestedLinearSession.free();
      }
    } finally {
      nestedLinearProgram.free();
    }

    const mlpFrontend = nn.sequential([
      nn.linear(2, 3, { weights: [1, 2, 3, 4, 5, 6], bias: [0.5, -0.5, 1] }),
      nn.relu(),
      nn.linear(3, 2, { weights: [1, -1, 0.5, 2, -0.5, 1], bias: [0.25, -0.75] }),
    ]);
    const mlpSupport = mlpFrontend.compileSupport();
    const rootMlpCompilerSignatures = nn.compilerSignatures(mlpFrontend);
    const rootMlpIr = nn.tensorProgramIr(mlpFrontend);
    const rootMlpKernelPlan = nn.kernelPlan(mlpFrontend);
    const rootMlpBufferLayout = nn.bufferLayout(mlpFrontend);
    const rootMlpMemoryLayout = nn.memoryLayout(mlpFrontend);
    const rootMlpInputShape = nn.inputShape(mlpFrontend);
    const rootMlpOutputShape = nn.outputShape(mlpFrontend);
    const rootMlpShapeConstraints = nn.shapeConstraints(mlpFrontend);
    const rootMlpParameterLayout = nn.parameterLayout(mlpFrontend);
    if (
      !mlpFrontend.canCompile() ||
      !mlpSupport.supported ||
      mlpSupport.nativePath !== "device-program" ||
      mlpSupport.modelKind !== "module" ||
      mlpSupport.layerCount !== 3 ||
      typeof mlpSupport.irSignature !== "string" ||
      mlpSupport.irSignature.length === 0 ||
      typeof mlpSupport.kernelPlanSignature !== "string" ||
      mlpSupport.kernelPlanSignature.length === 0 ||
      typeof mlpSupport.memoryLayoutSignature !== "string" ||
      mlpSupport.memoryLayoutSignature.length === 0 ||
      typeof mlpSupport.parameterLayoutSignature !== "string" ||
      mlpSupport.parameterLayoutSignature.length === 0 ||
      typeof mlpSupport.bufferLayoutSignature !== "string" ||
      mlpSupport.bufferLayoutSignature.length === 0 ||
      !mlpSupport.compilerSignatures ||
      !Object.isFrozen(mlpSupport.compilerSignatures) ||
      mlpSupport.compilerSignatures.ir !== mlpSupport.irSignature ||
      mlpSupport.compilerSignatures.kernelPlan !== mlpSupport.kernelPlanSignature ||
      mlpSupport.compilerSignatures.memoryLayout !== mlpSupport.memoryLayoutSignature ||
      mlpSupport.compilerSignatures.parameterLayout !== mlpSupport.parameterLayoutSignature ||
      mlpSupport.compilerSignatures.bufferLayout !== mlpSupport.bufferLayoutSignature ||
      !mlpSupport.ir ||
      !Object.isFrozen(mlpSupport.ir) ||
      !Object.isFrozen(mlpSupport.ir.ops) ||
      mlpSupport.ir.kind !== "tensor-program-ir" ||
      mlpSupport.ir.opCount !== 3 ||
      mlpSupport.ir.inputLen !== 2 ||
      mlpSupport.ir.outputLen !== 2 ||
      mlpSupport.ir.inputValueId !== 0 ||
      mlpSupport.ir.outputValueId !== 7 ||
      mlpSupport.ir.valueCount !== 8 ||
      !Object.isFrozen(mlpSupport.ir.values) ||
      !Object.isFrozen(mlpSupport.ir.values[1].shape) ||
      !Object.isFrozen(mlpSupport.ir.values[1].strides) ||
      !Object.isFrozen(mlpSupport.ir.ops[0].inputValueIds) ||
      mlpSupport.ir.values.map((value) => `${value.id}:${value.role}:${value.name ?? value.op ?? "input"}:${value.scalarCount}:${value.binding ?? ""}`).join("|") !== "0:input:input:2:|1:parameter:0.weight:6:weights|2:parameter:0.bias:3:bias|3:op-output:linear:3:|4:op-output:activation:3:|5:parameter:2.weight:6:weights|6:parameter:2.bias:2:bias|7:op-output:linear:2:" ||
      mlpSupport.ir.values.map((value) => `${value.dtype}:${value.scalarBytes}:${value.rank}:${value.storageLayout}:${value.strides.join("x")}:${value.storageOffset}:${value.dense ? "dense" : "strided"}`).join("|") !== "f32:4:1:row-major:1:0:dense|f32:4:2:row-major:3x1:0:dense|f32:4:1:row-major:1:0:dense|f32:4:1:row-major:1:0:dense|f32:4:1:row-major:1:0:dense|f32:4:2:row-major:2x1:0:dense|f32:4:1:row-major:1:0:dense|f32:4:1:row-major:1:0:dense" ||
      mlpSupport.ir.values.map((value) => value.byteLength).join(",") !== "8,24,12,12,12,24,8,8" ||
      mlpSupport.ir.ops.map((op) => `${op.inputValueIds.join("+")}=>${op.outputValueId}[${op.parameterValueIds.join("+")}]`).join("|") !== "0+1+2=>3[1+2]|3=>4[]|4+5+6=>7[5+6]" ||
      !mlpSupport.kernelPlan ||
      !Object.isFrozen(mlpSupport.kernelPlan) ||
      !Object.isFrozen(mlpSupport.kernelPlan.ops) ||
      !Object.isFrozen(mlpSupport.kernelPlan.ops[0]) ||
      !Object.isFrozen(mlpSupport.kernelPlan.ops[0].inputShape) ||
      !Object.isFrozen(mlpSupport.kernelPlan.ops[0].outputShape) ||
      !Object.isFrozen(mlpSupport.kernelPlan.ops[0].inputValueIds) ||
      "nativeOps" in mlpSupport.kernelPlan ||
      "desc" in mlpSupport.kernelPlan.ops[0] ||
      mlpSupport.kernelPlan.kind !== "native-module-kernel-plan" ||
      mlpSupport.kernelPlan.dispatchCount !== 2 ||
      mlpSupport.kernelPlan.descriptorCount !== 2 ||
      mlpSupport.kernelPlan.weightsLen !== 12 ||
      mlpSupport.kernelPlan.biasLen !== 5 ||
      !Object.isFrozen(mlpSupport.kernelPlan.memoryLayout) ||
      !Object.isFrozen(mlpSupport.kernelPlan.memoryLayout.values) ||
      !Object.isFrozen(mlpSupport.kernelPlan.memoryLayout.values[3].consumerOpIndices) ||
      mlpSupport.kernelPlan.memoryLayout.valueCount !== mlpSupport.ir.valueCount ||
      mlpSupport.kernelPlan.memoryLayout.totalScalarCount !== 27 ||
      mlpSupport.kernelPlan.memoryLayout.totalByteLength !== 108 ||
      mlpSupport.kernelPlan.memoryLayout.scratchScalarCount !== 6 ||
      mlpSupport.kernelPlan.memoryLayout.scratchByteLength !== 24 ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => `${value.id}:${value.role}:${value.scalarType}:${value.scalarBytes}:${value.shape.join("x")}:${value.strides.join("x")}:${value.name ?? value.op ?? "input"}`).join("|") !== "0:input:f32:4:2:1:input|1:parameter:f32:4:2x3:3x1:0.weight|2:parameter:f32:4:3:1:0.bias|3:op-output:f32:4:3:1:linear|4:op-output:f32:4:3:1:activation|5:parameter:f32:4:3x2:2x1:2.weight|6:parameter:f32:4:2:1:2.bias|7:op-output:f32:4:2:1:linear" ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => `${value.storageClass}:${value.buffer}`).join("|") !== "step-input:input|persistent:weights|persistent:bias|scratch:scratch|scratch:scratch|persistent:weights|persistent:bias|step-output:output" ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => `${value.producerOpIndex ?? "ext"}>${value.consumerOpIndices.join("+") || "-"}:${value.firstUseOpIndex ?? "-"}:${value.lastUseOpIndex ?? "-"}:${value.liveStartOpIndex ?? "-"}:${value.liveEndOpIndex ?? "-"}`).join("|") !== "ext>0:0:0:0:0|ext>0:0:0:0:0|ext>0:0:0:0:0|0>1:1:1:0:1|1>2:2:2:1:2|ext>2:2:2:2:2|ext>2:2:2:2:2|2>-:-:-:2:2" ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => `${value.scalarOffset}:${value.byteOffset}`).join("|") !== "0:0|2:8|8:32|11:44|14:56|17:68|23:92|25:100" ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => `${value.bufferScalarOffset}:${value.bufferByteOffset}`).join("|") !== "0:0|0:0|0:0|0:0|3:12|6:24|3:12|0:0" ||
      mlpSupport.kernelPlan.memoryLayout.values.map((value) => value.byteLength).join(",") !== "8,24,12,12,12,24,8,8" ||
      !Object.isFrozen(mlpSupport.kernelPlan.bufferLayout) ||
      !Object.isFrozen(mlpSupport.kernelPlan.bufferLayout.slots) ||
      mlpSupport.kernelPlan.bufferLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== "input:step-input:2:8|output:step-output:2:8|weights:persistent:12:48|bias:persistent:5:20" ||
      mlpSupport.kernelPlan.bufferLayout.weights.elementCount !== 12 ||
      mlpSupport.kernelPlan.bufferLayout.weights.byteLength !== 48 ||
      mlpSupport.kernelPlan.bufferLayout.bias.byteLength !== 20 ||
      !Object.isFrozen(mlpSupport.kernelPlan.parameterLayout) ||
      !Object.isFrozen(mlpSupport.kernelPlan.parameterLayout.parameters) ||
      mlpSupport.kernelPlan.parameterLayout.weightsLen !== 12 ||
      mlpSupport.kernelPlan.parameterLayout.biasLen !== 5 ||
      mlpSupport.kernelPlan.parameterLayout.parameters.map((param) => `${param.name}:${param.binding}:${param.offset}:${param.scalarCount}:${param.shape.join("x")}`).join("|") !== "0.weight:weights:0:6:2x3|0.bias:bias:0:3:3|2.weight:weights:6:6:3x2|2.bias:bias:3:2:2" ||
      mlpSupport.kernelPlan.ops.map((op) => op.kernel).join("|") !== "linear|linear" ||
      mlpSupport.kernelPlan.ops.map((op) => `${op.scalarType}:${op.scalarBytes}`).join("|") !== "f32:4|f32:4" ||
      mlpSupport.kernelPlan.ops.map((op) => `${op.nativeDispatchCount}:${op.nativeDescriptorCount}:${op.nativeKernels.join("+")}`).join("|") !== "1:1:linear+relu|1:1:linear" ||
      mlpSupport.kernelPlan.ops[0].fusedOpCount !== 2 ||
      mlpSupport.kernelPlan.ops[0].fusedOps!.join("+") !== "linear+activation" ||
      mlpSupport.kernelPlan.ops[0].fusedIndices!.join("+") !== "0+1" ||
      mlpSupport.kernelPlan.ops.map((op) => `${op.inputValueIds.join("+")}=>${op.outputValueId}`).join("|") !== "0+1+2=>4|4+5+6=>7" ||
      typeof nn.placeParameters !== "function" ||
      typeof mlpFrontend.placeParameters !== "function"
    ) {
      throw new Error(`unexpected TS MLP compile support: ${json(mlpSupport)}`);
    }
    if (
      !rootMlpCompilerSignatures ||
      !Object.isFrozen(rootMlpCompilerSignatures) ||
      rootMlpCompilerSignatures.ir !== mlpSupport.compilerSignatures.ir ||
      rootMlpCompilerSignatures.kernelPlan !== mlpSupport.compilerSignatures.kernelPlan ||
      rootMlpCompilerSignatures.memoryLayout !== mlpSupport.compilerSignatures.memoryLayout ||
      rootMlpCompilerSignatures.parameterLayout !== mlpSupport.compilerSignatures.parameterLayout ||
      rootMlpCompilerSignatures.bufferLayout !== mlpSupport.compilerSignatures.bufferLayout
    ) {
      throw new Error(`unexpected TS root MLP compiler signatures: ${json({ rootMlpCompilerSignatures, compilerSignatures: mlpSupport.compilerSignatures })}`);
    }
    const rootMlpDerivedCompilerSignatures = nn.compilerSignatures({
      compileSupport() {
        return {
          ...mlpSupport,
          compilerSignatures: undefined,
          irSignature: undefined,
          kernelPlanSignature: undefined,
          memoryLayoutSignature: undefined,
          parameterLayoutSignature: undefined,
          bufferLayoutSignature: undefined,
        };
      },
    } as any);
    if (
      !rootMlpDerivedCompilerSignatures ||
      !Object.isFrozen(rootMlpDerivedCompilerSignatures) ||
      rootMlpDerivedCompilerSignatures.ir !== mlpSupport.compilerSignatures.ir ||
      rootMlpDerivedCompilerSignatures.kernelPlan !== mlpSupport.compilerSignatures.kernelPlan ||
      rootMlpDerivedCompilerSignatures.memoryLayout !== mlpSupport.compilerSignatures.memoryLayout ||
      rootMlpDerivedCompilerSignatures.parameterLayout !== mlpSupport.compilerSignatures.parameterLayout ||
      rootMlpDerivedCompilerSignatures.bufferLayout !== mlpSupport.compilerSignatures.bufferLayout
    ) {
      throw new Error(`unexpected TS derived MLP compiler signatures: ${json({ rootMlpDerivedCompilerSignatures, compilerSignatures: mlpSupport.compilerSignatures })}`);
    }
    if (
      !rootMlpIr ||
      !Object.isFrozen(rootMlpIr) ||
      rootMlpIr.kind !== mlpSupport.ir.kind ||
      rootMlpIr.opCount !== mlpSupport.ir.opCount ||
      rootMlpIr.inputLen !== mlpSupport.ir.inputLen ||
      rootMlpIr.outputLen !== mlpSupport.ir.outputLen ||
      !rootMlpKernelPlan ||
      !Object.isFrozen(rootMlpKernelPlan) ||
      rootMlpKernelPlan.kind !== mlpSupport.kernelPlan.kind ||
      rootMlpKernelPlan.dispatchCount !== mlpSupport.kernelPlan.dispatchCount ||
      rootMlpKernelPlan.descriptorCount !== mlpSupport.kernelPlan.descriptorCount ||
      rootMlpKernelPlan.weightsLen !== mlpSupport.kernelPlan.weightsLen ||
      rootMlpKernelPlan.biasLen !== mlpSupport.kernelPlan.biasLen ||
      !rootMlpMemoryLayout ||
      !Object.isFrozen(rootMlpMemoryLayout) ||
      !Object.isFrozen(rootMlpMemoryLayout.values) ||
      rootMlpMemoryLayout.valueCount !== mlpSupport.kernelPlan.memoryLayout.valueCount ||
      rootMlpMemoryLayout.values[5].name !== mlpSupport.kernelPlan.memoryLayout.values[5].name ||
      rootMlpMemoryLayout.values[5].scalarType !== "f32" ||
      rootMlpMemoryLayout.values[5].scalarBytes !== 4 ||
      rootMlpMemoryLayout.values[5].strides.join(",") !== "2,1" ||
      !rootMlpBufferLayout ||
      !Object.isFrozen(rootMlpBufferLayout) ||
      rootMlpBufferLayout.weights.elementCount !== mlpSupport.kernelPlan.bufferLayout.weights.elementCount ||
      rootMlpBufferLayout.bias.byteLength !== mlpSupport.kernelPlan.bufferLayout.bias.byteLength ||
      rootMlpBufferLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== mlpSupport.kernelPlan.bufferLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") ||
      !Object.isFrozen(rootMlpInputShape) ||
      !Object.isFrozen(rootMlpOutputShape) ||
      rootMlpInputShape.join(",") !== mlpSupport.kernelPlan.inputShape.join(",") ||
      rootMlpOutputShape.join(",") !== mlpSupport.kernelPlan.outputShape.join(",") ||
      !rootMlpShapeConstraints ||
      !Object.isFrozen(rootMlpShapeConstraints) ||
      rootMlpShapeConstraints.inputLen !== mlpSupport.kernelPlan.shapeConstraints.inputLen ||
      rootMlpShapeConstraints.outputLen !== mlpSupport.kernelPlan.shapeConstraints.outputLen ||
      !rootMlpParameterLayout ||
      !Object.isFrozen(rootMlpParameterLayout) ||
      rootMlpParameterLayout.weightsLen !== mlpSupport.kernelPlan.parameterLayout.weightsLen ||
      rootMlpParameterLayout.biasLen !== mlpSupport.kernelPlan.parameterLayout.biasLen ||
      rootMlpParameterLayout.parameters.map((param) => `${param.name}:${param.binding}:${param.offset}`).join("|") !== mlpSupport.kernelPlan.parameterLayout.parameters.map((param) => `${param.name}:${param.binding}:${param.offset}`).join("|")
    ) {
      throw new Error(`unexpected TS root MLP compiler artifacts: ${json({ rootMlpIr, rootMlpKernelPlan, rootMlpMemoryLayout, rootMlpBufferLayout, rootMlpInputShape, rootMlpOutputShape, rootMlpShapeConstraints, rootMlpParameterLayout })}`);
    }
    const constraintsOnlySupport = Object.freeze({
      ...mlpSupport,
      kernelPlan: Object.freeze({
        ...mlpSupport.kernelPlan,
        inputShape: undefined,
        outputShape: undefined,
        inputLen: undefined,
        outputLen: undefined,
      }),
    });
    const constraintsOnlyModule = {
      compileSupport() {
        return constraintsOnlySupport;
      },
    } as any;
    const constraintsOnlyInputShape = nn.inputShape(constraintsOnlyModule);
    const constraintsOnlyOutputShape = nn.outputShape(constraintsOnlyModule);
    if (
      !constraintsOnlyInputShape ||
      !constraintsOnlyOutputShape ||
      constraintsOnlyInputShape.join(",") !== mlpSupport.kernelPlan.shapeConstraints.inputShape.join(",") ||
      constraintsOnlyOutputShape.join(",") !== mlpSupport.kernelPlan.shapeConstraints.outputShape.join(",")
    ) {
      throw new Error(`expected TS nn input/output shape helpers to use KernelPlan shape constraints: ${json({ constraintsOnlyInputShape, constraintsOnlyOutputShape })}`);
    }
    const mlpHostOutput = mlpFrontend.forward(frontendInput);
    if (!(mlpHostOutput instanceof Tensor)) {
      throw new Error("expected TS MLP eager output to be a Tensor");
    }
    expectCloseWithin(mlpHostOutput.toFloat32Array(), [7.5, 28.75], 1e-6);
    const mlpProgram = mlpFrontend.compile({ backend: "cpu" });
    try {
      const mlpRequirements = mlpProgram.requirements();
      const mlpInspection = mlpProgram.inspect();
      const mlpEvidence = mlpProgram.compileEvidence();
      const mlpProgramLayout = mlpProgram.bufferLayout();
      const mlpTrace = mlpProgram.trace();
      const mlpCompilerSignatures = mlpProgram.compilerSignatures();
      const mlpIr = mlpProgram.tensorProgramIr();
      const mlpKernelPlan = mlpProgram.kernelPlan();
      const mlpMemoryLayout = mlpProgram.memoryLayout();
      const mlpShapeConstraints = mlpProgram.shapeConstraints();
      const mlpParameterLayout = mlpProgram.parameterLayout();
      const mlpProgramDesc = (mlpProgram as any).desc;
      const mlpOriginalEvidence = mlpEvidence;
      mlpProgram._withCompileEvidence({
        ...mlpEvidence,
        compilerSignatures: undefined,
        irSignature: undefined,
        kernelPlanSignature: undefined,
        memoryLayoutSignature: undefined,
        parameterLayoutSignature: undefined,
        bufferLayoutSignature: undefined,
      });
      let mlpDerivedEvidence: ProgramCompileEvidence | null;
      let mlpDerivedCompilerSignatures: ModuleCompleteCompilerSignatures | null;
      try {
        mlpDerivedEvidence = mlpProgram.compileEvidence();
        mlpDerivedCompilerSignatures = mlpProgram.compilerSignatures();
      } finally {
        mlpProgram._withCompileEvidence(mlpOriginalEvidence);
      }
      const mlpProgramInputShape = mlpProgram.inputShape();
      const mlpProgramOutputShape = mlpProgram.outputShape();
      const legacyConstructorProgram = new Program((mlpProgram as any).handle, (mlpProgram as any).desc, {
        ...mlpEvidence,
        compilerSignatures: undefined,
        irSignature: undefined,
        kernelPlanSignature: undefined,
        memoryLayoutSignature: undefined,
        parameterLayoutSignature: undefined,
        bufferLayoutSignature: undefined,
      });
      const legacyConstructorEvidence = legacyConstructorProgram.compileEvidence();
      const legacyConstructorSignatures = legacyConstructorProgram.compilerSignatures();
      if (
        mlpRequirements.modelKind !== "module" ||
        mlpRequirements.inputLen !== mlpSupport.kernelPlan.inputLen ||
        mlpRequirements.outputLen !== mlpSupport.kernelPlan.outputLen ||
        mlpRequirements.weightsLen !== mlpSupport.kernelPlan.weightsLen ||
        mlpRequirements.biasLen !== mlpSupport.kernelPlan.biasLen ||
        mlpProgramLayout.slots.map((slot) => `${slot.name}:${slot.role}:${slot.elementCount}:${slot.byteLength}`).join("|") !== "input:step-input:2:8|output:step-output:2:8|weights:persistent:12:48|bias:persistent:5:20" ||
        mlpProgramLayout.output.byteLength !== mlpSupport.kernelPlan.bufferLayout.output.byteLength ||
        mlpInspection.commandCount !== mlpSupport.kernelPlan.dispatchCount ||
        !mlpProgramDesc ||
        mlpProgramDesc.inputShape.join(",") !== mlpSupport.kernelPlan.shapeConstraints.inputShape.join(",") ||
        mlpProgramDesc.outputShape.join(",") !== mlpSupport.kernelPlan.shapeConstraints.outputShape.join(",") ||
        mlpProgramDesc.inputLen !== mlpSupport.kernelPlan.shapeConstraints.inputLen ||
        mlpProgramDesc.outputLen !== mlpSupport.kernelPlan.shapeConstraints.outputLen ||
        mlpProgramDesc.ops.length !== mlpSupport.kernelPlan.descriptorCount ||
        !mlpEvidence ||
        !Object.isFrozen(mlpEvidence) ||
        mlpEvidence.kind !== "module" ||
        mlpEvidence.irSignature !== mlpSupport.irSignature ||
        mlpEvidence.kernelPlanSignature !== mlpSupport.kernelPlanSignature ||
        mlpEvidence.memoryLayoutSignature !== mlpSupport.memoryLayoutSignature ||
        mlpEvidence.parameterLayoutSignature !== mlpSupport.parameterLayoutSignature ||
        mlpEvidence.bufferLayoutSignature !== mlpSupport.bufferLayoutSignature ||
        !mlpEvidence.compilerSignatures ||
        !Object.isFrozen(mlpEvidence.compilerSignatures) ||
        mlpEvidence.compilerSignatures.ir !== mlpSupport.compilerSignatures.ir ||
        mlpEvidence.compilerSignatures.kernelPlan !== mlpSupport.compilerSignatures.kernelPlan ||
        mlpEvidence.compilerSignatures.memoryLayout !== mlpSupport.compilerSignatures.memoryLayout ||
        mlpEvidence.compilerSignatures.parameterLayout !== mlpSupport.compilerSignatures.parameterLayout ||
        mlpEvidence.compilerSignatures.bufferLayout !== mlpSupport.compilerSignatures.bufferLayout ||
        !mlpCompilerSignatures ||
        mlpCompilerSignatures !== mlpEvidence.compilerSignatures ||
        mlpCompilerSignatures.ir !== mlpEvidence.irSignature ||
        mlpCompilerSignatures.kernelPlan !== mlpEvidence.kernelPlanSignature ||
        mlpCompilerSignatures.memoryLayout !== mlpEvidence.memoryLayoutSignature ||
        mlpCompilerSignatures.parameterLayout !== mlpEvidence.parameterLayoutSignature ||
        mlpCompilerSignatures.bufferLayout !== mlpEvidence.bufferLayoutSignature ||
        !mlpDerivedEvidence ||
        !Object.isFrozen(mlpDerivedEvidence) ||
        mlpDerivedEvidence.irSignature !== mlpEvidence.irSignature ||
        mlpDerivedEvidence.kernelPlanSignature !== mlpEvidence.kernelPlanSignature ||
        mlpDerivedEvidence.memoryLayoutSignature !== mlpEvidence.memoryLayoutSignature ||
        mlpDerivedEvidence.parameterLayoutSignature !== mlpEvidence.parameterLayoutSignature ||
        mlpDerivedEvidence.bufferLayoutSignature !== mlpEvidence.bufferLayoutSignature ||
        !mlpDerivedEvidence.compilerSignatures ||
        !Object.isFrozen(mlpDerivedEvidence.compilerSignatures) ||
        !mlpDerivedCompilerSignatures ||
        mlpDerivedCompilerSignatures !== mlpDerivedEvidence.compilerSignatures ||
        mlpDerivedCompilerSignatures.ir !== mlpEvidence.compilerSignatures.ir ||
        mlpDerivedCompilerSignatures.kernelPlan !== mlpEvidence.compilerSignatures.kernelPlan ||
        mlpDerivedCompilerSignatures.memoryLayout !== mlpEvidence.compilerSignatures.memoryLayout ||
        mlpDerivedCompilerSignatures.parameterLayout !== mlpEvidence.compilerSignatures.parameterLayout ||
        mlpDerivedCompilerSignatures.bufferLayout !== mlpEvidence.compilerSignatures.bufferLayout ||
        !legacyConstructorEvidence ||
        !Object.isFrozen(legacyConstructorEvidence) ||
        legacyConstructorEvidence.irSignature !== mlpEvidence.irSignature ||
        legacyConstructorEvidence.kernelPlanSignature !== mlpEvidence.kernelPlanSignature ||
        legacyConstructorEvidence.memoryLayoutSignature !== mlpEvidence.memoryLayoutSignature ||
        legacyConstructorEvidence.parameterLayoutSignature !== mlpEvidence.parameterLayoutSignature ||
        legacyConstructorEvidence.bufferLayoutSignature !== mlpEvidence.bufferLayoutSignature ||
        !legacyConstructorEvidence.compilerSignatures ||
        legacyConstructorSignatures !== legacyConstructorEvidence.compilerSignatures ||
        mlpTrace !== mlpEvidence.trace ||
        mlpTrace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0:linear|1:activation|2:linear" ||
        mlpTrace.ops[1].activation !== "relu" ||
        mlpIr !== mlpEvidence.ir ||
        mlpIr.ops.map((op) => op.op).join("|") !== "linear|activation|linear" ||
        mlpIr.valueCount !== 8 ||
        mlpIr.values[5].name !== "2.weight" ||
        mlpIr.values[5].dtype !== "f32" ||
        mlpIr.values[5].scalarBytes !== 4 ||
        mlpIr.values[5].storageLayout !== "row-major" ||
        mlpIr.values[5].strides.join(",") !== "2,1" ||
        mlpIr.ops.map((op) => `${op.inputValueIds.join("+")}=>${op.outputValueId}`).join("|") !== "0+1+2=>3|3=>4|4+5+6=>7" ||
        mlpIr.ops[1].attrs.activation !== "relu" ||
        mlpMemoryLayout.valueCount !== 8 ||
        mlpMemoryLayout.values[5].name !== "2.weight" ||
        mlpKernelPlan.memoryLayout.values[5].scalarType !== "f32" ||
        mlpKernelPlan.memoryLayout.values[5].scalarBytes !== 4 ||
        mlpKernelPlan.ops.map((op) => op.kernel).join("|") !== "linear|linear" ||
        mlpKernelPlan.ops.map((op) => `${op.scalarType}:${op.scalarBytes}`).join("|") !== "f32:4|f32:4" ||
        mlpKernelPlan.ops.map((op) => `${op.nativeDispatchCount}:${op.nativeDescriptorCount}:${op.nativeKernels.join("+")}`).join("|") !== "1:1:linear+relu|1:1:linear" ||
        mlpKernelPlan.ops.map((op) => `${op.inputValueIds.join("+")}=>${op.outputValueId}`).join("|") !== "0+1+2=>4|4+5+6=>7" ||
        !Object.isFrozen(mlpEvidence.kernelPlan.ops[0]) ||
        !Object.isFrozen(mlpEvidence.kernelPlan.ops[0].inputValueIds) ||
        mlpEvidence.kernelPlan.dispatchCount !== mlpSupport.kernelPlan.dispatchCount ||
        mlpEvidence.kernelPlan.bufferLayout.weights.byteLength !== 48 ||
        !Object.isFrozen(mlpEvidence.kernelPlan.bufferLayout.slots) ||
        !Object.isFrozen(mlpProgramInputShape) ||
        !Object.isFrozen(mlpProgramOutputShape) ||
        mlpShapeConstraints.inputShape.join(",") !== "2" ||
        mlpShapeConstraints.outputShape.join(",") !== "2" ||
        mlpShapeConstraints.inputLen !== 2 ||
        mlpShapeConstraints.outputLen !== 2 ||
        mlpParameterLayout.parameters.map((param) => `${param.name}:${param.binding}:${param.offset}:${param.scalarCount}`).join("|") !== "0.weight:weights:0:6|0.bias:bias:0:3|2.weight:weights:6:6|2.bias:bias:3:2" ||
        mlpEvidence.kernelPlan.parameterLayout.parameters[0].name !== "0.weight" ||
        mlpEvidence.kernelPlan.parameterLayout.parameters[2].offset !== 6 ||
        typeof mlpProgram.bindModule !== "function" ||
        typeof mlpProgram.memoryLayout !== "function" ||
        typeof mlpProgram.moduleCompatibility !== "function" ||
        typeof mlpProgram.acceptsModule !== "function" ||
        "nativeOps" in mlpEvidence.kernelPlan ||
        "desc" in mlpEvidence.kernelPlan.ops[0]
      ) {
        throw new Error(`unexpected TS MLP requirements: ${json(mlpRequirements)}`);
      }
      const mlpCompatibility = mlpProgram.moduleCompatibility(mlpFrontend);
      if (
        !mlpCompatibility.compatible ||
        mlpCompatibility.reason !== null ||
        mlpCompatibility.programKind !== "module" ||
        mlpCompatibility.moduleKind !== "module" ||
        mlpCompatibility.diagnostics.length !== 0 ||
        !Object.isFrozen(mlpCompatibility) ||
        mlpProgram.acceptsModule(mlpFrontend) !== true
      ) {
        throw new Error(`unexpected TS MLP module compatibility: ${json(mlpCompatibility)}`);
      }
      const staleFlatSignatureModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            irSignature: `${mlpSupport.irSignature}|stale-flat`,
            kernelPlanSignature: `${mlpSupport.kernelPlanSignature}|stale-flat`,
            memoryLayoutSignature: `${mlpSupport.memoryLayoutSignature}|stale-flat`,
            parameterLayoutSignature: `${mlpSupport.parameterLayoutSignature}|stale-flat`,
            bufferLayoutSignature: `${mlpSupport.bufferLayoutSignature}|stale-flat`,
            compilerSignatures: mlpSupport.compilerSignatures,
          };
        },
      };
      const staleFlatSignatureCompatibility = mlpProgram.moduleCompatibility(staleFlatSignatureModule);
      if (
        !staleFlatSignatureCompatibility.compatible ||
        staleFlatSignatureCompatibility.diagnostics.length !== 0 ||
        mlpProgram.acceptsModule(staleFlatSignatureModule) !== true
      ) {
        throw new Error(`expected TS canonical compilerSignatures to override stale flat aliases: ${json(staleFlatSignatureCompatibility)}`);
      }
      const irSignatureMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            compilerSignatures: undefined,
            irSignature: `${mlpSupport.irSignature}|mismatch`,
          };
        },
      };
      const irSignatureMismatchCompatibility = mlpProgram.moduleCompatibility(irSignatureMismatchModule);
      const irSignatureMismatchDiagnostic = irSignatureMismatchCompatibility.diagnostics[0];
      if (
        irSignatureMismatchCompatibility.compatible ||
        irSignatureMismatchDiagnostic?.code !== "ir-mismatch" ||
        irSignatureMismatchDiagnostic.signatureKind !== "ir" ||
        irSignatureMismatchDiagnostic.programSignature !== mlpEvidence.irSignature ||
        irSignatureMismatchDiagnostic.moduleSignature !== `${mlpSupport.irSignature}|mismatch` ||
        mlpProgram.acceptsModule(irSignatureMismatchModule) !== false
      ) {
        throw new Error(`expected TS IR signature compatibility mismatch: ${json(irSignatureMismatchCompatibility)}`);
      }
      const nestedIrSignatureMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            irSignature: undefined,
            compilerSignatures: {
              ...mlpSupport.compilerSignatures,
              ir: `${mlpSupport.irSignature}|nested-mismatch`,
            },
          };
        },
      };
      const nestedIrSignatureMismatchCompatibility = mlpProgram.moduleCompatibility(nestedIrSignatureMismatchModule);
      const nestedIrSignatureMismatchDiagnostic = nestedIrSignatureMismatchCompatibility.diagnostics[0];
      if (
        nestedIrSignatureMismatchCompatibility.compatible ||
        nestedIrSignatureMismatchDiagnostic?.code !== "ir-mismatch" ||
        nestedIrSignatureMismatchDiagnostic.signatureKind !== "ir" ||
        nestedIrSignatureMismatchDiagnostic.programSignature !== mlpEvidence.irSignature ||
        nestedIrSignatureMismatchDiagnostic.moduleSignature !== `${mlpSupport.irSignature}|nested-mismatch` ||
        mlpProgram.acceptsModule(nestedIrSignatureMismatchModule) !== false
      ) {
        throw new Error(`expected TS nested IR signature compatibility mismatch: ${json(nestedIrSignatureMismatchCompatibility)}`);
      }
      const descriptorCountMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            kernelPlan: {
              ...mlpSupport.kernelPlan,
              descriptorCount: mlpSupport.kernelPlan.descriptorCount + 1,
            },
            compilerSignatures: mlpSupport.compilerSignatures,
          };
        },
      };
      const descriptorCountMismatchCompatibility = mlpProgram.moduleCompatibility(descriptorCountMismatchModule);
      const descriptorCountMismatchDiagnostic = descriptorCountMismatchCompatibility.diagnostics[0];
      if (
        descriptorCountMismatchCompatibility.compatible ||
        descriptorCountMismatchDiagnostic?.code !== "kernel-plan-mismatch" ||
        descriptorCountMismatchDiagnostic.signatureKind !== "kernel-plan" ||
        descriptorCountMismatchDiagnostic.programSignature !== mlpEvidence.kernelPlanSignature ||
        descriptorCountMismatchDiagnostic.moduleSignature !== mlpSupport.kernelPlanSignature ||
        mlpProgram.acceptsModule(descriptorCountMismatchModule) !== false
      ) {
        throw new Error(`expected TS descriptor-count compatibility mismatch: ${json(descriptorCountMismatchCompatibility)}`);
      }
      const memoryLayoutMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            memoryLayoutSignature: `${mlpSupport.memoryLayoutSignature}|mismatch`,
            compilerSignatures: {
              ...mlpSupport.compilerSignatures,
              memoryLayout: `${mlpSupport.memoryLayoutSignature}|mismatch`,
            },
            kernelPlan: {
              ...mlpSupport.kernelPlan!,
              memoryLayout: {
                ...mlpSupport.kernelPlan!.memoryLayout,
                values: mlpSupport.kernelPlan!.memoryLayout.values.map((value, index) =>
                  index === 5 ? { ...value, storageOffset: value.storageOffset + 1 } : value
                ),
              },
            },
          };
        },
      };
      const memoryLayoutMismatchCompatibility = mlpProgram.moduleCompatibility(memoryLayoutMismatchModule);
      const memoryLayoutMismatchDiagnostic = memoryLayoutMismatchCompatibility.diagnostics[0];
      if (
        memoryLayoutMismatchCompatibility.compatible ||
        memoryLayoutMismatchDiagnostic?.code !== "memory-layout-mismatch" ||
        memoryLayoutMismatchDiagnostic.signatureKind !== "memory-layout" ||
        memoryLayoutMismatchDiagnostic.programSignature !== mlpEvidence.memoryLayoutSignature ||
        memoryLayoutMismatchDiagnostic.moduleSignature !== `${mlpSupport.memoryLayoutSignature}|mismatch` ||
        mlpProgram.acceptsModule(memoryLayoutMismatchModule) !== false
      ) {
        throw new Error(`expected TS memory-layout compatibility mismatch: ${json(memoryLayoutMismatchCompatibility)}`);
      }
      const parameterLayoutMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            compilerSignatures: undefined,
            parameterLayoutSignature: mlpSupport.parameterLayoutSignature!.replace("0.weight", "renamed.weight"),
            kernelPlan: {
              ...mlpSupport.kernelPlan!,
              parameterLayout: {
                ...mlpSupport.kernelPlan!.parameterLayout,
                parameters: mlpSupport.kernelPlan!.parameterLayout.parameters.map((param, index) =>
                  index === 0 ? { ...param, name: "renamed.weight" } : param
                ),
              },
            },
          };
        },
      };
      const parameterLayoutMismatchCompatibility = mlpProgram.moduleCompatibility(parameterLayoutMismatchModule);
      const parameterLayoutMismatchDiagnostic = parameterLayoutMismatchCompatibility.diagnostics[0];
      if (
        parameterLayoutMismatchCompatibility.compatible ||
        parameterLayoutMismatchDiagnostic?.code !== "parameter-layout-mismatch" ||
        parameterLayoutMismatchDiagnostic.signatureKind !== "parameter-layout" ||
        parameterLayoutMismatchDiagnostic.programSignature !== mlpEvidence.parameterLayoutSignature ||
        !parameterLayoutMismatchDiagnostic.moduleSignature.includes("renamed.weight") ||
        mlpProgram.acceptsModule(parameterLayoutMismatchModule) !== false
      ) {
        throw new Error(`expected TS parameter-layout compatibility mismatch: ${json(parameterLayoutMismatchCompatibility)}`);
      }
      const bufferLayoutMismatchModule: any = {
        compileSupport() {
          return {
            ...mlpSupport,
            compilerSignatures: undefined,
            bufferLayoutSignature: `${mlpSupport.bufferLayoutSignature}|mismatch`,
            kernelPlan: {
              ...mlpSupport.kernelPlan!,
              bufferLayout: {
                ...mlpSupport.kernelPlan!.bufferLayout,
                slots: mlpSupport.kernelPlan!.bufferLayout.slots.map((slot) =>
                  slot.name === "weights" ? { ...slot, byteLength: slot.byteLength + 4 } : slot
                ),
                weights: {
                  ...mlpSupport.kernelPlan!.bufferLayout.weights,
                  byteLength: mlpSupport.kernelPlan!.bufferLayout.weights.byteLength + 4,
                },
              },
            },
          };
        },
      };
      const bufferLayoutMismatchCompatibility = mlpProgram.moduleCompatibility(bufferLayoutMismatchModule);
      const bufferLayoutMismatchDiagnostic = bufferLayoutMismatchCompatibility.diagnostics[0];
      if (
        bufferLayoutMismatchCompatibility.compatible ||
        bufferLayoutMismatchDiagnostic?.code !== "buffer-layout-mismatch" ||
        bufferLayoutMismatchDiagnostic.signatureKind !== "buffer-layout" ||
        bufferLayoutMismatchDiagnostic.programSignature !== mlpEvidence.bufferLayoutSignature ||
        bufferLayoutMismatchDiagnostic.moduleSignature !== `${mlpSupport.bufferLayoutSignature}|mismatch` ||
        mlpProgram.acceptsModule(bufferLayoutMismatchModule) !== false
      ) {
        throw new Error(`expected TS buffer-layout compatibility mismatch: ${json(bufferLayoutMismatchCompatibility)}`);
      }
      const incompatibleMlp = nn.sequential([
        nn.linear(2, 4),
        nn.relu(),
        nn.linear(4, 2),
      ]);
      const incompatibleMlpCompatibility = mlpProgram.moduleCompatibility(incompatibleMlp);
      if (
        incompatibleMlpCompatibility.compatible ||
        incompatibleMlpCompatibility.diagnostics.length === 0 ||
        mlpProgram.acceptsModule(incompatibleMlp) !== false
      ) {
        throw new Error(`expected incompatible TS MLP module compatibility: ${json(incompatibleMlpCompatibility)}`);
      }
      let incompatibleBindRejected = false;
      try {
        mlpProgram.bindModule(incompatibleMlp);
      } catch (err) {
        incompatibleBindRejected = String((err as { message?: unknown }).message ?? err).includes("does not match");
      }
      if (!incompatibleBindRejected) throw new Error("expected TS Program.bindModule to reject incompatible module");
      let incompatiblePlacementRejected = false;
      let incompatiblePlacementBindings: { weights?: unknown; bias?: unknown } | null = null;
      try {
        incompatiblePlacementBindings = nn.placeParameters(incompatibleMlp, mlpProgram);
      } catch (err) {
        const message = String((err as { message?: unknown }).message ?? err);
        incompatiblePlacementRejected =
          message.includes("nn.placeParameters module is not compatible with this Program") &&
          message.includes("does not match");
      } finally {
        if (incompatiblePlacementBindings) {
          const weights = incompatiblePlacementBindings.weights;
          const bias = incompatiblePlacementBindings.bias;
          if (weights instanceof NativeBuffer) weights.free();
          if (bias instanceof NativeBuffer) bias.free();
        }
      }
      if (!incompatiblePlacementRejected) throw new Error("expected TS nn.placeParameters to reject incompatible module before placement");
      const mlpSession = mlpProgram.bind(mlpFrontend.bindParameters());
      try {
        expectCloseWithin(mlpSession.step(frontendInput), Array.from(mlpHostOutput.toFloat32Array()), 1e-6);
      } finally {
        mlpSession.free();
      }
      const placedMlpBindings = mlpFrontend.placeParameters(mlpProgram);
      try {
        if (!(placedMlpBindings.weights instanceof NativeBuffer) || !(placedMlpBindings.bias instanceof NativeBuffer)) {
          throw new Error("expected TS module.placeParameters to return NativeBuffer parameter bindings");
        }
        expectCloseWithin(placedMlpBindings.weights.readFloat32(12), Array.from(mlpFrontend.bindParameters().weights), 1e-6);
        expectCloseWithin(placedMlpBindings.bias.readFloat32(5), Array.from(mlpFrontend.bindParameters().bias!), 1e-6);
        const placedMlpSession = mlpProgram.bind(placedMlpBindings);
        try {
          expectCloseWithin(placedMlpSession.step(frontendInput), Array.from(mlpHostOutput.toFloat32Array()), 1e-6);
        } finally {
          placedMlpSession.free();
        }
      } finally {
        if (placedMlpBindings.weights instanceof NativeBuffer) placedMlpBindings.weights.free();
        if (placedMlpBindings.bias instanceof NativeBuffer) placedMlpBindings.bias.free();
      }
      const boundModuleSession = mlpProgram.bindModule(mlpFrontend);
      try {
        expectCloseWithin(boundModuleSession.step(frontendInput), Array.from(mlpHostOutput.toFloat32Array()), 1e-6);
      } finally {
        boundModuleSession.free();
      }
    } finally {
      mlpProgram.free();
    }

    const nestedMlp = nn.sequential([
      nn.sequential([
        nn.linear(2, 3, { weights: [1, 2, 3, 4, 5, 6], bias: [0.5, -0.5, 1] }),
        nn.relu(),
      ]),
      nn.sequential([
        nn.linear(3, 2, { weights: [1, -1, 0.5, 2, -0.5, 1], bias: [0.25, -0.75] }),
      ]),
    ]);
    const nestedMlpSupport = nestedMlp.compileSupport();
    if (
      !nestedMlp.canCompile() ||
      !nestedMlpSupport.supported ||
      nestedMlpSupport.nativePath !== "device-program" ||
      nestedMlpSupport.modelKind !== "module" ||
      nestedMlpSupport.layerCount !== 3
    ) {
      throw new Error(`unexpected TS nested MLP compile support: ${json(nestedMlpSupport)}`);
    }
    const nestedMlpProgram = nestedMlp.compile({ backend: "cpu" });
    try {
      const nestedMlpSession = nestedMlpProgram.bind(nestedMlp.bindParameters());
      try {
        const eager = nestedMlp.forward(frontendInput);
        if (!(eager instanceof Tensor)) {
          throw new Error("expected TS nested MLP eager output to be a Tensor");
        }
        expectCloseWithin(nestedMlpSession.step(frontendInput), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        nestedMlpSession.free();
      }
    } finally {
      nestedMlpProgram.free();
    }

    const siluNoBiasMlp = nn.sequential([
      nn.linear(2, 2, { weights: [1, -2, 0.5, 3], bias: false }),
      nn.silu(),
      nn.linear(2, 1, { weights: [2, -1], bias: false }),
    ]);
    const siluNoBiasSupport = siluNoBiasMlp.compileSupport();
    if (!siluNoBiasSupport.supported || siluNoBiasSupport.modelKind !== "module") {
      throw new Error(`unexpected TS no-bias SiLU MLP compile support: ${json(siluNoBiasSupport)}`);
    }
    const siluNoBiasProgram = siluNoBiasMlp.compile({ backend: "cpu" });
    try {
      const siluNoBiasRequirements = siluNoBiasProgram.requirements();
      if (siluNoBiasRequirements.modelKind !== "module" || siluNoBiasRequirements.weightsLen !== 6 || siluNoBiasRequirements.biasLen !== 0) {
        throw new Error(`unexpected TS no-bias SiLU MLP requirements: ${json(siluNoBiasRequirements)}`);
      }
      const siluNoBiasSession = siluNoBiasProgram.bind(siluNoBiasMlp.bindParameters());
      try {
        const eager = siluNoBiasMlp.forward(frontendInput);
        if (!(eager instanceof Tensor)) {
          throw new Error("expected TS no-bias SiLU MLP eager output to be a Tensor");
        }
        expectCloseWithin(siluNoBiasSession.step(frontendInput), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        siluNoBiasSession.free();
      }
    } finally {
      siluNoBiasProgram.free();
    }

    expectClose(nn.layerNorm(2, { eps: 0 }).forward([1, 3, 2, 4]), [-1, 1, -1, 1]);
    const rms = nn.rmsNorm(2, { eps: 0 }).forward([3, 4]);
    expectCloseWithin(rms, [3 / Math.sqrt(12.5), 4 / Math.sqrt(12.5)], 1e-5);
    const softmaxModule = nn.softmax();
    const softmaxOut = softmaxModule.forward(tensor([1, 2, 3, 1, 1, 1], [2, 3]));
    const exp1 = Math.exp(1);
    const exp2 = Math.exp(2);
    const exp3 = Math.exp(3);
    const expSum = exp1 + exp2 + exp3;
    expectCloseWithin(softmaxOut.toFloat32Array(), [exp1 / expSum, exp2 / expSum, exp3 / expSum, 1 / 3, 1 / 3, 1 / 3], 1e-6);
    const softmaxSupport = softmaxModule.compileSupport();
    if (
      softmaxModule.canCompile() ||
      softmaxSupport.supported ||
      !softmaxSupport.composable ||
      !Array.isArray(softmaxSupport.diagnostics) ||
      !Object.isFrozen(softmaxSupport.diagnostics) ||
      softmaxSupport.diagnostics[0].stage !== "ir" ||
      softmaxSupport.diagnostics[0].code !== "missing-input-shape"
    ) {
      throw new Error(`unexpected TS softmax compile support: ${json(softmaxSupport)}`);
    }
    const lnInput = tensor([1, 2, 4], [3], { requiresGrad: true });
    const ln = nn.layerNorm(3, { eps: 0, weights: [1, 2, 3], bias: [0.5, -0.5, 1] });
    const lnOut = ln.forward(lnInput);
    if (!(lnOut instanceof Tensor)) {
      throw new Error("expected TS layerNorm Tensor input to return a Tensor");
    }
    const sqrt14 = Math.sqrt(14);
    expectCloseWithin(lnOut.toFloat32Array(), [0.5 - 4 / sqrt14, -0.5 - 2 / sqrt14, 1 + 15 / sqrt14], 1e-6);
    train.backward(lnOut.sum());
    const lnParams = ln.parameters();
    expectCloseWithin(lnInput.grad!, [-3 / (7 * sqrt14), 9 / (14 * sqrt14), -3 / (14 * sqrt14)], 1e-6);
    expectCloseWithin(lnParams[0].grad, [-4 / sqrt14, -1 / sqrt14, 5 / sqrt14], 1e-6);
    expectCloseWithin(lnParams[1].grad, [1, 1, 1], 1e-6);
    const rmsInput = tensor([3, 4], [2], { requiresGrad: true });
    const rmsModule = nn.rmsNorm(2, { eps: 0, weights: [1, 2] });
    const rmsOut = rmsModule.forward(rmsInput);
    if (!(rmsOut instanceof Tensor)) {
      throw new Error("expected TS rmsNorm Tensor input to return a Tensor");
    }
    const rmsScale = Math.sqrt(12.5);
    expectCloseWithin(rmsOut.toFloat32Array(), [3 / rmsScale, 8 / rmsScale], 1e-6);
    train.backward(rmsOut.sum());
    expectCloseWithin(rmsInput.grad!, [-8 / (25 * rmsScale), 6 / (25 * rmsScale)], 1e-6);
    expectCloseWithin(rmsModule.parameters()[0].grad, [3 / rmsScale, 4 / rmsScale], 1e-6);
    const silu = nn.silu().forward([-1, 0, 2]);
    expectCloseWithin(silu, [-1 / (1 + Math.exp(1)), 0, 2 / (1 + Math.exp(-2))], 1e-6);
    const sigmoid = nn.sigmoid().forward([-1, 0, 2]);
    expectCloseWithin(sigmoid, [1 / (1 + Math.exp(1)), 0.5, 1 / (1 + Math.exp(-2))], 1e-6);
    const embedding = nn.embedding(4, 3, {
      weights: [
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
      ],
    });
    const embeddingOut = embedding.forward([2, 0, 3]);
    if (!(embeddingOut instanceof Tensor) || embeddingOut.shape.join(",") !== "3,3") {
      throw new Error("expected TS embedding output to be a rank-2 Tensor");
    }
    expectClose(embeddingOut.toFloat32Array(), [
      30, 31, 32,
      10, 11, 12,
      40, 41, 42,
    ]);
    const embeddingSupport = embedding.compileSupport();
    if (
      embedding.canCompile() ||
      embeddingSupport.supported ||
      embeddingSupport.composable ||
      !String(embeddingSupport.reason).includes("requires inputShape") ||
      !embeddingSupport.trace ||
      !Object.isFrozen(embeddingSupport.trace) ||
      !Array.isArray(embeddingSupport.diagnostics) ||
      !Object.isFrozen(embeddingSupport.diagnostics) ||
      embeddingSupport.diagnostics[0].stage !== "ir" ||
      embeddingSupport.diagnostics[0].code !== "missing-input-shape" ||
      embeddingSupport.trace.shapeKnown !== false ||
      embeddingSupport.trace.ops.length !== 1 ||
      embeddingSupport.trace.ops[0].op !== "embedding" ||
      embeddingSupport.trace.ops[0].numEmbeddings !== 4 ||
      embeddingSupport.trace.ops[0].embeddingDim !== 3 ||
      embeddingSupport.trace.ops[0].parameters[0].name !== "0.weight" ||
      embeddingSupport.trace.ops[0].parameters[0].shape.join(",") !== "4,3" ||
      "data" in embeddingSupport.trace.ops[0].parameters[0]
    ) {
      throw new Error(`unexpected TS embedding compile support: ${json(embeddingSupport)}`);
    }
    const shapedEmbeddingSupport = embedding.compileSupport({ inputShape: [3] });
    if (
      !embedding.canCompile({ inputShape: [3] }) ||
      !shapedEmbeddingSupport.supported ||
      shapedEmbeddingSupport.reason !== null ||
      shapedEmbeddingSupport.nativePath !== "device-program" ||
      shapedEmbeddingSupport.modelKind !== "module" ||
      shapedEmbeddingSupport.inputLen !== 3 ||
      shapedEmbeddingSupport.outputLen !== 9 ||
      shapedEmbeddingSupport.weightsLen !== 12 ||
      shapedEmbeddingSupport.biasLen !== 0 ||
      !shapedEmbeddingSupport.trace ||
      !Object.isFrozen(shapedEmbeddingSupport.trace) ||
      !shapedEmbeddingSupport.trace.shapeKnown ||
      shapedEmbeddingSupport.trace.inputShape?.join(",") !== "3" ||
      shapedEmbeddingSupport.trace.outputShape?.join(",") !== "3,3" ||
      shapedEmbeddingSupport.trace.ops[0].inputShape?.join(",") !== "3" ||
      shapedEmbeddingSupport.trace.ops[0].outputShape?.join(",") !== "3,3"
    ) {
      throw new Error(`unexpected TS shaped embedding compile support: ${json(shapedEmbeddingSupport)}`);
    }
    const rootEmbeddingSupport = nn.compileSupport(embedding, { inputShape: [3] });
    const rootEmbeddingTrace = nn.trace(embedding, { inputShape: [3] });
    if (
      !nn.canCompile(embedding, { inputShape: [3] }) ||
      !rootEmbeddingSupport.supported ||
      !rootEmbeddingSupport.trace ||
      rootEmbeddingSupport.trace.ops[0].op !== "embedding" ||
      rootEmbeddingTrace.outputShape?.join(",") !== "3,3"
    ) {
      throw new Error(`unexpected TS root embedding compile helpers: ${json({ rootEmbeddingSupport, rootEmbeddingTrace })}`);
    }
    const embeddingProgram = nn.compile(embedding, { backend: "cpu", inputShape: [3] });
    try {
      const requirements = embeddingProgram.requirements();
      if (
        requirements.modelKind !== "module" ||
        requirements.inputLen !== 3 ||
        requirements.outputLen !== 9 ||
        requirements.weightsLen !== 12 ||
        requirements.biasLen !== 0
      ) {
        throw new Error(`unexpected TS embedding module requirements: ${json(requirements)}`);
      }
      const defaultEmbeddingCompatibility = embeddingProgram.moduleCompatibility(embedding);
      if (!defaultEmbeddingCompatibility.compatible || embeddingProgram.acceptsModule(embedding) !== true) {
        throw new Error(`expected embedding Program evidence to supply default module inputShape: ${json(defaultEmbeddingCompatibility)}`);
      }
      const defaultEmbeddingModuleSession = embeddingProgram.bindModule(embedding);
      try {
        expectClose(defaultEmbeddingModuleSession.step([2, 0, 3]), embeddingOut.toFloat32Array());
      } finally {
        defaultEmbeddingModuleSession.free();
      }
      const defaultEmbeddingPlaced = embedding.placeParameters(embeddingProgram);
      try {
        expectClose(
          defaultEmbeddingPlaced.weights.readFloat32(12),
          nn.bindParameters(embedding, { inputShape: [3] }).weights as Float32Array,
        );
      } finally {
        defaultEmbeddingPlaced.weights.free();
      }
      const rootDefaultEmbeddingPlaced = nn.placeParameters(embedding, embeddingProgram);
      try {
        expectClose(
          rootDefaultEmbeddingPlaced.weights.readFloat32(12),
          nn.bindParameters(embedding, { inputShape: [3] }).weights as Float32Array,
        );
      } finally {
        rootDefaultEmbeddingPlaced.weights.free();
      }
      const embeddingSession = embeddingProgram.bind(nn.bindParameters(embedding, { inputShape: [3] }));
      try {
        expectClose(embeddingSession.step([2, 0, 3]), embeddingOut.toFloat32Array());
        expectClose(embeddingSession.step(new Uint32Array([2, 0, 3])), embeddingOut.toFloat32Array());
      } finally {
        embeddingSession.free();
      }
      const boundTokenSession = embeddingProgram.bind({
        ...nn.bindParameters(embedding, { inputShape: [3] }),
        input: new Uint32Array([2, 0, 3]),
      });
      try {
        expectClose(boundTokenSession.step(), embeddingOut.toFloat32Array());
      } finally {
        boundTokenSession.free();
      }
      const nativeTokenWeights = embeddingProgram.createWeightsBuffer();
      const nativeTokenInput = embeddingProgram.createInputBuffer();
      const nativeTokenOutput = embeddingProgram.createOutputBuffer();
      try {
        nativeTokenWeights.writeFloat32(nn.bindParameters(embedding, { inputShape: [3] }).weights);
        nativeTokenInput.writeFloat32(new Uint32Array([2, 0, 3]));
        const nativeTokenSession = embeddingProgram.bind({
          weights: nativeTokenWeights,
          input: nativeTokenInput,
          output: nativeTokenOutput,
        });
        try {
          expectClose(nativeTokenSession.step(), embeddingOut.toFloat32Array());
          expectClose(nativeTokenOutput.readFloat32(9), embeddingOut.toFloat32Array());
        } finally {
          nativeTokenSession.free();
        }
      } finally {
        nativeTokenOutput.free();
        nativeTokenInput.free();
        nativeTokenWeights.free();
      }
    } finally {
      embeddingProgram.free();
    }
    let rootEmbeddingCompileRejected = false;
    try {
      // @ts-expect-error Embedding compile needs an explicit token-window shape; this block checks the runtime error too.
      nn.compile(embedding);
    } catch (err) {
      rootEmbeddingCompileRejected = String((err as { message?: unknown }).message ?? err).includes("requires inputShape");
    }
    if (!rootEmbeddingCompileRejected) throw new Error("expected TS root nn.compile to reject unshaped embedding");
    const embeddingGradModule = nn.embedding(4, 3, {
      weights: [
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42,
      ],
    });
    train.backward(embeddingGradModule.forward([2, 0, 2]).sum());
    expectClose(embeddingGradModule.parameters()[0].grad, [
      1, 1, 1,
      0, 0, 0,
      2, 2, 2,
      0, 0, 0,
    ]);
    const tokenFrontend = nn.sequential([embedding, nn.silu()]);
    const tokenParams = tokenFrontend.parameters();
    if (tokenParams.length !== 1 || tokenParams[0].name !== "0.weight") {
      throw new Error(`unexpected TS embedding parameter list: ${json(tokenParams.map((param) => param.name))}`);
    }
    if (!(tokenParams[0].grad instanceof Float32Array) || tokenParams[0].grad.length !== tokenParams[0].data.length) {
      throw new Error("TS embedding parameter did not expose a durable gradient buffer");
    }
    const tokenFrontendOut = tokenFrontend.forward([1]);
    if (!(tokenFrontendOut instanceof Tensor)) {
      throw new Error("expected TS embedding sequential output to be a Tensor");
    }
    expectCloseWithin(
      tokenFrontendOut.toFloat32Array(),
      [20, 21, 22].map((x) => x / (1 + Math.exp(-x))),
      1e-5,
    );
    const tokenFrontendSupport = tokenFrontend.compileSupport({ inputShape: [1] });
    if (
      !tokenFrontendSupport.supported ||
      tokenFrontendSupport.modelKind !== "module" ||
      tokenFrontendSupport.inputLen !== 1 ||
      tokenFrontendSupport.outputLen !== 3 ||
      tokenFrontendSupport.weightsLen !== 12 ||
      tokenFrontendSupport.trace?.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0:embedding|1:activation"
    ) {
      throw new Error(`unexpected TS embedding Sequential compile support: ${json(tokenFrontendSupport)}`);
    }
    const tokenFrontendProgram = tokenFrontend.compile({ backend: "cpu", inputShape: [1] });
    try {
      const tokenFrontendSession = tokenFrontendProgram.bind(tokenFrontend.bindParameters({ inputShape: [1] }));
      try {
        expectCloseWithin(tokenFrontendSession.step([1]), tokenFrontendOut.toFloat32Array(), 1e-5);
        expectCloseWithin(tokenFrontendSession.step(new Uint32Array([1])), tokenFrontendOut.toFloat32Array(), 1e-5);
      } finally {
        tokenFrontendSession.free();
      }
    } finally {
      tokenFrontendProgram.free();
    }
    const ce = loss.crossEntropy([2, 0, 1, 0, 3, 1], loss.classTargets([0, 1]), { classes: 3 });
    const ceExpected = -(
      Math.log(Math.exp(2) / (Math.exp(2) + Math.exp(0) + Math.exp(1))) +
      Math.log(Math.exp(3) / (Math.exp(0) + Math.exp(3) + Math.exp(1)))
    ) / 2;
    if (typeof ce !== "number") {
      throw new Error("expected TS crossEntropy over raw logits to return a number");
    }
    expectCloseWithin(Float32Array.of(ce), [ceExpected], 1e-6);
    const ceLogits = tensor([2, 0, 1, 0, 3, 1], [2, 3], { requiresGrad: true });
    const ceLoss = loss.crossEntropy(ceLogits, [0, 1], { classes: 3 });
    if (!(ceLoss instanceof Tensor) || Math.abs(ceLoss.item() - ceExpected) > 1e-6) {
      throw new Error("expected TS crossEntropy over Tensor logits to return a scalar Tensor loss");
    }
    train.backward(ceLoss);
    const row0Denom = Math.exp(2) + Math.exp(0) + Math.exp(1);
    const row1Denom = Math.exp(0) + Math.exp(3) + Math.exp(1);
    expectCloseWithin(ceLogits.grad!, [
      (Math.exp(2) / row0Denom - 1) / 2,
      (Math.exp(0) / row0Denom) / 2,
      (Math.exp(1) / row0Denom) / 2,
      (Math.exp(0) / row1Denom) / 2,
      (Math.exp(3) / row1Denom - 1) / 2,
      (Math.exp(1) / row1Denom) / 2,
    ], 1e-6);
    const classifier = nn.linear(2, 3, { weights: [0, 0, 0, 0, 0, 0], bias: [0, 0, 0] });
    const classifierLoss = loss.crossEntropy(classifier.forward(tensor([1, 2], [2])), [2]);
    if (!(classifierLoss instanceof Tensor)) {
      throw new Error("expected TS classifier crossEntropy loss to be a Tensor");
    }
    train.step(optim.sgd(classifier, { lr: 0.3 }), { loss: classifierLoss });
    expectCloseWithin(classifier.weight, [-0.1, -0.1, 0.2, -0.2, -0.2, 0.4], 1e-6);
    expectCloseWithin(classifier.bias!, [-0.1, -0.1, 0.2], 1e-6);
    expectClose(classifier.parameters()[0].grad, [0, 0, 0, 0, 0, 0]);
    const autogradLinear = nn.linear(2, 1, { weights: [0.5, -1], bias: [0] });
    const autogradLoss = loss.mse(autogradLinear.forward(tensor([2, -1], [2])), tensor([1], [1]));
    if (!(autogradLoss instanceof Tensor) || Math.abs(autogradLoss.item() - 1) > 1e-6) {
      throw new Error("expected TS MSE over tensors to return a scalar Tensor loss");
    }
    train.backward(autogradLoss);
    const autogradParams = autogradLinear.parameters();
    expectCloseWithin(autogradParams[0].grad, [4, -2], 1e-6);
    expectCloseWithin(autogradParams[1].grad, [2], 1e-6);
    const autogradStep = nn.linear(2, 1, { weights: [0.5, -1], bias: [0] });
    const autogradStepLoss = loss.mse(autogradStep.forward(tensor([2, -1], [2])), tensor([1], [1]));
    if (!(autogradStepLoss instanceof Tensor)) {
      throw new Error("expected TS train.step loss to be a Tensor");
    }
    train.step(optim.sgd(autogradStep, { lr: 0.1 }), { loss: autogradStepLoss });
    expectCloseWithin(autogradStep.weight, [0.1, -0.8], 1e-6);
    expectCloseWithin(autogradStep.bias!, [-0.2], 1e-6);
    expectClose(autogradStep.parameters()[0].grad, [0, 0]);
    const trainable = nn.linear(2, 1, { weights: [1, -1], bias: [0.5] });
    const trainParams = trainable.parameters();
    trainParams[0].grad.set([0.25, -0.5]);
    trainParams[1].grad.set([1]);
    const sgd = optim.sgd(trainParams, { lr: 0.1 });
    train.step(sgd);
    expectCloseWithin(trainable.weight, [0.975, -0.95], 1e-6);
    expectCloseWithin(trainable.bias!, [0.4], 1e-6);
    expectClose(trainParams[0].grad, [0, 0]);
    expectClose(trainParams[1].grad, [0]);
    trainParams[0].grad.set([0.2, -0.4]);
    trainParams[1].grad.set([0.5]);
    const momentum = optim.sgd(trainable, { lr: 0.1, momentum: 0.5 });
    momentum.step();
    const momentumState = momentum.stateDict();
    if (momentumState.kind !== "sgd" || momentumState.paramCount !== 2 || momentumState.entries.length !== 2) {
      throw new Error(`unexpected TS SGD optimizer state: ${json(momentumState)}`);
    }
    if (momentumState.entries[0].name !== "velocity.0" || momentumState.entries[0].layout !== "row-major:linear.weight[in_features,out_features]") {
      throw new Error(`unexpected TS SGD optimizer state entry: ${json(momentumState.entries[0])}`);
    }
    const restoredMomentum = optim.sgd(trainable, { lr: 0.1, momentum: 0.5 });
    restoredMomentum.loadStateDict(momentumState);
    expectClose(restoredMomentum.stateDict().entries[0].data, momentumState.entries[0].data);
    let optimizerLayoutRejected = false;
    try {
      restoredMomentum.loadStateDict({
        ...momentumState,
        entries: [
          { ...momentumState.entries[0], layout: "feature-major:linear.weight[out_features,in_features]" },
          momentumState.entries[1],
        ],
      });
    } catch (err) {
      optimizerLayoutRejected = String((err as { message?: unknown }).message ?? err).includes("layout must be row-major:linear.weight[in_features,out_features]");
    }
    if (!optimizerLayoutRejected) throw new Error("expected TS optimizer state layout rejection");
    trainParams[0].grad.set([0.1, 0.1]);
    const adamw = optim.adamW(trainable, { lr: 0.01, weightDecay: 0.1 });
    adamw.step();
    const adamwState = optim.stateDict(adamw);
    if (adamwState.kind !== "adamw" || adamwState.step !== 1 || adamwState.entries.length !== 4) {
      throw new Error(`unexpected TS AdamW optimizer state: ${json(adamwState)}`);
    }
    const restoredAdamW = optim.adamW(trainable, { lr: 0.01, weightDecay: 0.1 });
    optim.loadStateDict(restoredAdamW, adamwState);
    expectClose(restoredAdamW.stateDict().entries[0].data, adamwState.entries[0].data);
    if (!(trainable.weight[0] < 0.975)) {
      throw new Error("expected TS AdamW optimizer to update module parameters");
    }
    const multiLayer = nn.sequential([
      nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: false }),
      nn.softmax(),
      nn.relu(),
    ]);
    const multiLayerSupport = multiLayer.compileSupport();
    if (
      !multiLayer.canCompile() ||
      !multiLayerSupport.supported ||
      multiLayerSupport.reason !== null ||
      multiLayerSupport.nativePath !== "device-program" ||
      multiLayerSupport.modelKind !== "module" ||
      multiLayerSupport.layerCount !== 3 ||
      !multiLayerSupport.trace ||
      !Object.isFrozen(multiLayerSupport.trace) ||
      !Object.isFrozen(multiLayerSupport.trace.ops) ||
      !multiLayerSupport.trace.shapeKnown ||
      multiLayerSupport.trace.inputShape?.join(",") !== "2" ||
      multiLayerSupport.trace.outputShape?.join(",") !== "2" ||
      multiLayerSupport.trace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0:linear|1:softmax|2:activation" ||
      multiLayerSupport.trace.ops[0].parameters[0].name !== "0.weight" ||
      "data" in multiLayerSupport.trace.ops[0].parameters[0]
    ) {
      throw new Error(`unexpected TS multi-layer compile support: ${json(multiLayerSupport)}`);
    }
    const multiLayerProgram = multiLayer.compile({ backend: "cpu" });
    try {
      const requirements = multiLayerProgram.requirements();
      if (
        requirements.modelKind !== "module" ||
        requirements.inputLen !== 2 ||
        requirements.outputLen !== 2 ||
        requirements.weightsLen !== 4 ||
        requirements.biasLen !== 0
      ) {
        throw new Error(`unexpected TS multi-layer module requirements: ${json(requirements)}`);
      }
      const multiLayerSession = multiLayerProgram.bind(multiLayer.bindParameters());
      try {
        const eager = multiLayer.forward(frontendInput);
        expectCloseWithin(multiLayerSession.step(frontendInput), Array.from(eager.toFloat32Array()), 1e-6);
        const sessionInfo = multiLayerSession.inspect();
        if (sessionInfo.modelKind !== "module" || sessionInfo.persistentBindingCount !== 1) {
          throw new Error(`unexpected TS multi-layer module session inspection: ${json(sessionInfo)}`);
        }
      } finally {
        multiLayerSession.free();
      }
      const moduleNativeWeights = multiLayerProgram.createWeightsBuffer();
      const moduleNativeInput = multiLayerProgram.createInputBuffer();
      const moduleNativeOutput = multiLayerProgram.createOutputBuffer();
      try {
        moduleNativeWeights.writeFloat32(multiLayer.bindParameters().weights);
        moduleNativeInput.writeFloat32(frontendInput.toFloat32Array());
        const moduleNativeSession = multiLayerProgram.bind({
          weights: moduleNativeWeights,
          input: moduleNativeInput,
          output: moduleNativeOutput,
        });
        try {
          const nativeInfo = moduleNativeSession.inspect();
          if (
            nativeInfo.modelKind !== "module" ||
            nativeInfo.outputStorage !== "host" ||
            nativeInfo.persistentBindingCount !== 1 ||
            nativeInfo.stepInputCount !== 1 ||
            nativeInfo.stepOutputCount !== 1 ||
            nativeInfo.hostBindingCount !== 3 ||
            nativeInfo.resourceBindingCount !== 0 ||
            nativeInfo.bindingShapeHash === 0n
          ) {
            throw new Error(`unexpected TS native-buffer module session inspection: ${json(nativeInfo)}`);
          }
          const eager = multiLayer.forward(frontendInput);
          expectCloseWithin(moduleNativeSession.step(), Array.from(eager.toFloat32Array()), 1e-6);
          expectCloseWithin(moduleNativeOutput.readFloat32(2), Array.from(eager.toFloat32Array()), 1e-6);
          const nextInput = tensor([3, 1], [2]);
          moduleNativeInput.writeFloat32(nextInput.toFloat32Array());
          const nextEager = multiLayer.forward(nextInput);
          expectCloseWithin(moduleNativeSession.step(), Array.from(nextEager.toFloat32Array()), 1e-6);
        } finally {
          moduleNativeSession.free();
        }
      } finally {
        moduleNativeOutput.free();
        moduleNativeInput.free();
        moduleNativeWeights.free();
      }
    } finally {
      multiLayerProgram.free();
    }
    const batchedMultiLayerSupport = multiLayer.compileSupport({ inputShape: [2, 2] });
    if (
      !multiLayer.canCompile({ inputShape: [2, 2] }) ||
      !batchedMultiLayerSupport.supported ||
      batchedMultiLayerSupport.reason !== null ||
      batchedMultiLayerSupport.nativePath !== "device-program" ||
      batchedMultiLayerSupport.modelKind !== "module" ||
      batchedMultiLayerSupport.inputLen !== 4 ||
      batchedMultiLayerSupport.outputLen !== 4 ||
      !batchedMultiLayerSupport.trace ||
      batchedMultiLayerSupport.trace.inputShape?.join(",") !== "2,2" ||
      batchedMultiLayerSupport.trace.outputShape?.join(",") !== "2,2" ||
      batchedMultiLayerSupport.trace.ops[0].inputShape?.join(",") !== "2,2" ||
      batchedMultiLayerSupport.trace.ops[0].outputShape?.join(",") !== "2,2"
    ) {
      throw new Error(`unexpected TS batched module compile support: ${json(batchedMultiLayerSupport)}`);
    }
    const batchedMultiLayerInput = tensor([1, 2, 3, 1], [2, 2]);
    const batchedMultiLayerProgram = multiLayer.compile({ backend: "cpu", inputShape: [2, 2] });
    try {
      const requirements = batchedMultiLayerProgram.requirements();
      if (
        requirements.modelKind !== "module" ||
        requirements.inputLen !== 4 ||
        requirements.outputLen !== 4 ||
        requirements.weightsLen !== 4 ||
        requirements.biasLen !== 0
      ) {
        throw new Error(`unexpected TS batched module requirements: ${json(requirements)}`);
      }
      const batchedCompatibility = batchedMultiLayerProgram.moduleCompatibility(multiLayer);
      if (!batchedCompatibility.compatible || batchedMultiLayerProgram.acceptsModule(multiLayer) !== true) {
        throw new Error(`expected batched Program evidence to supply default module inputShape: ${json(batchedCompatibility)}`);
      }
      const batchedBoundModuleSession = batchedMultiLayerProgram.bindModule(multiLayer);
      try {
        const eager = multiLayer.forward(batchedMultiLayerInput);
        expectCloseWithin(
          batchedBoundModuleSession.step(batchedMultiLayerInput),
          Array.from(eager.toFloat32Array()),
          1e-6,
        );
      } finally {
        batchedBoundModuleSession.free();
      }
      const batchedMultiLayerSession = batchedMultiLayerProgram.bind(
        multiLayer.bindParameters({ inputShape: [2, 2] }),
      );
      try {
        const eager = multiLayer.forward(batchedMultiLayerInput);
        expectCloseWithin(
          batchedMultiLayerSession.step(batchedMultiLayerInput),
          Array.from(eager.toFloat32Array()),
          1e-6,
        );
      } finally {
        batchedMultiLayerSession.free();
      }
    } finally {
      batchedMultiLayerProgram.free();
    }
    const nestedModule = nn.sequential([
      nn.sequential([
        nn.linear(2, 2, { weights: [1, 0, 0, 1], bias: false }),
        nn.softmax(),
      ]),
      nn.relu(),
    ]);
    const nestedModuleSupport = nestedModule.compileSupport();
    if (
      !nestedModule.canCompile() ||
      !nestedModuleSupport.supported ||
      nestedModuleSupport.reason !== null ||
      nestedModuleSupport.nativePath !== "device-program" ||
      nestedModuleSupport.modelKind !== "module" ||
      nestedModuleSupport.layerCount !== 3 ||
      !nestedModuleSupport.trace ||
      nestedModuleSupport.trace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0.0:linear|0.1:softmax|1:activation" ||
      nestedModuleSupport.trace.ops[0].parameters[0].name !== "0.0.weight"
    ) {
      throw new Error(`unexpected TS nested module compile support: ${json(nestedModuleSupport)}`);
    }
    const nestedModuleProgram = nestedModule.compile({ backend: "cpu" });
    try {
      const nestedModuleSession = nestedModuleProgram.bind(nestedModule.bindParameters());
      try {
        const eager = nestedModule.forward(frontendInput);
        expectCloseWithin(nestedModuleSession.step(frontendInput), Array.from(eager.toFloat32Array()), 1e-6);
      } finally {
        nestedModuleSession.free();
      }
    } finally {
      nestedModuleProgram.free();
    }
    const nestedModuleTrace = nestedModule.trace();
    if (
      nestedModuleTrace.kind !== "sequential" ||
      !nestedModuleTrace.normalized ||
      nestedModuleTrace.layerCount !== 3 ||
      nestedModuleTrace.opCount !== 3 ||
      nestedModuleTrace.parameterCount !== 1 ||
      nestedModuleTrace.parameterScalarCount !== 4 ||
      nestedModuleTrace.ops.map((op) => `${op.path}:${op.op}`).join("|") !== "0.0:linear|0.1:softmax|1:activation" ||
      nestedModuleTrace.ops[0].parameters.length !== 1 ||
      nestedModuleTrace.ops[0].parameters[0].name !== "0.0.weight" ||
      nestedModuleTrace.ops[0].parameters[0].layout !== "row-major:linear.weight[in_features,out_features]" ||
      nestedModuleTrace.ops[0].parameters[0].scalarCount !== 4 ||
      "data" in nestedModuleTrace.ops[0].parameters[0]
    ) {
      throw new Error(`unexpected TS nested module trace: ${json(nestedModuleTrace)}`);
    }
    const nestedModuleShapedTrace = nestedModule.trace({ inputShape: [2] });
    if (
      !nestedModuleShapedTrace.shapeKnown ||
      nestedModuleShapedTrace.inputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.outputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[0].inputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[0].outputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[1].inputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[1].outputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[2].inputShape?.join(",") !== "2" ||
      nestedModuleShapedTrace.ops[2].outputShape?.join(",") !== "2"
    ) {
      throw new Error(`unexpected TS shaped nested module trace: ${json(nestedModuleShapedTrace)}`);
    }
    let traceShapeRejected = false;
    try {
      nestedModule.trace({ inputShape: [3] });
    } catch (err) {
      traceShapeRejected = String((err as { message?: unknown }).message ?? err).includes("linear trace input shape");
    }
    if (!traceShapeRejected) throw new Error("expected TS trace shape mismatch rejection");

    const rebound = program.bind({
      weights: [2, 0, 0, 2, 1, 1],
      bias: [0, 1, -1],
    });
    try {
      const output = rebound.step([3, 4]);
      expectClose(output, [14, 5, 3]);
    } finally {
      rebound.free();
    }

    const boundInput = new Float32Array([2, 3]);
    const boundOutput = new Float32Array(4);
    boundOutput.fill(-999);
    const bound = program.bind({
      weights: [1, 0, 0, 1, 1, 1],
      input: boundInput,
      output: boundOutput,
    });
    try {
      expectClose(bound.step(), [5, 3, 3]);
      if (boundOutput[3] !== -999) throw new Error("bound tiny linear step wrote past output");
      const hostReadTarget = new Float32Array(4).fill(-222);
      const hostReadInto = bound.readOutputInto(hostReadTarget);
      if (hostReadInto !== hostReadTarget || hostReadTarget[3] !== -222) {
        throw new Error("host-bound readOutputInto did not preserve caller storage");
      }
      expectClose(hostReadTarget.subarray(0, 3), [5, 3, 3]);
      let hostReadTargetRejected = false;
      try {
        bound.readOutputInto(new Uint8Array(3) as unknown as Float32Array);
      } catch (err) {
        hostReadTargetRejected = String((err as { message?: unknown }).message ?? err).includes("Float32Array target");
      }
      if (!hostReadTargetRejected) throw new Error("expected host-bound readOutputInto to reject non-Float32Array target");
      let hostReadAlignmentRejected = false;
      try {
        bound.readOutputInto(new Float32Array(3), 3, 2);
      } catch (err) {
        hostReadAlignmentRejected = String((err as { message?: unknown }).message ?? err).includes("aligned to f32");
      }
      if (!hostReadAlignmentRejected) throw new Error("expected host-bound readOutputInto to reject unaligned byteOffset");
      let hostReadRangeRejected = false;
      try {
        bound.readOutputInto(new Float32Array(3), 3, 8);
      } catch (err) {
        hostReadRangeRejected = String((err as { message?: unknown }).message ?? err).includes("source range is out of bounds");
      }
      if (!hostReadRangeRejected) throw new Error("expected host-bound readOutputInto to reject out-of-range source");
      const hostReadTensor = bound.readOutputTensor();
      if (hostReadTensor.shape.join(",") !== "3") {
        throw new Error(`unexpected host-bound readOutputTensor shape: ${hostReadTensor.shape}`);
      }
      expectClose(hostReadTensor.toFloat32Array(), [5, 3, 3]);
      boundInput.set([4, 5]);
      boundOutput.fill(-111);
      expectClose(bound.step(), [9, 5, 5]);
      if (boundOutput[3] !== -111) throw new Error("second bound tiny linear step wrote past output");
    } finally {
      bound.free();
    }

    const tensorInput = tensor([2, 3], [2]);
    const tensorOutput = tensor([-999, -999, -999, -777], [4]);
    const tensorBound = program.bind({
      weights: tensor([1, 0, 0, 1, 1, 1], [3, 2]),
      input: tensorInput,
      output: tensorOutput,
    });
    try {
      expectClose(tensorBound.step(), [5, 3, 3]);
      if (tensorOutput.data[3] !== -777) throw new Error("tensor-bound tiny linear step wrote past output");
      tensorInput.data.set([4, 5]);
      expectClose(tensorBound.step(), [9, 5, 5]);
      const tensorOverride = tensor([-1, -1, -1, -8], [4]);
      expectClose(tensorBound.step(undefined, tensorOverride), [9, 5, 5]);
      if (tensorOverride.data[3] !== -8) throw new Error("tensor output override wrote past output");
      if (tensorOutput.data[0] !== 9 || tensorOutput.data[3] !== -777) {
        throw new Error("tensor output override mutated bound tensor output");
      }
    } finally {
      tensorBound.free();
    }

    const nativeWeights = program.createBuffer("weights");
    nativeWeights.writeFloat32([1, 0, 0, 1, 1, 1]);
    const nativeInput = program.createBuffer("input");
    nativeInput.writeFloat32([2, 3]);
    const nativeOutput = NativeBuffer.fromFloat32([-999, -999, -999, -777]);
    const factoryOutput = program.createBuffer("output");
    if (factoryOutput.byteLength !== requirements.outputByteLength) {
      throw new Error(`unexpected factory output byte length: ${factoryOutput.byteLength}`);
    }
    const factoryOutputInfo = factoryOutput.inspect();
    if (
      !Object.isFrozen(factoryOutputInfo) ||
      factoryOutputInfo.storage !== "host" ||
      factoryOutputInfo.byteLength !== requirements.outputByteLength ||
      factoryOutputInfo.placement !== null ||
      factoryOutputInfo.access !== null ||
      factoryOutputInfo.handle !== 0 ||
      factoryOutputInfo.resourceByteLength !== 0
    ) {
      throw new Error(`unexpected host output inspection: ${json(factoryOutputInfo)}`);
    }
    const nativeOutputInfo = nativeOutput.inspect();
    if (!Object.isFrozen(nativeOutputInfo) || nativeOutputInfo.storage !== "host" || nativeOutputInfo.byteLength !== nativeOutput.byteLength) {
      throw new Error(`unexpected native output inspection: ${json(nativeOutputInfo)}`);
    }
    const autoPlacedInput = new Float32Array([1, 2]);
    const autoPlacedOutput = NativeBuffer.fromFloat32([-909, -909, -909, -808]);
    const autoPlacedBound = program.bind({
      weights: [1, 2, 3, 4, 5, 6],
      bias: [0.5, -0.5, 1.0],
      input: autoPlacedInput,
      output: autoPlacedOutput,
    });
    try {
      expectClose(autoPlacedBound.step(), [9.5, 11.5, 16]);
      if (autoPlacedOutput.readFloat32(4)[3] !== -808) throw new Error("auto-placed native output wrote past output");
      autoPlacedInput.set([2, 3]);
      expectClose(autoPlacedBound.step(), [14.5, 18.5, 25]);
      if (autoPlacedOutput.readFloat32(4)[3] !== -808) throw new Error("second auto-placed native output wrote past output");
    } finally {
      autoPlacedBound.free();
      autoPlacedOutput.free();
    }
    const mixedHostInput = new Float32Array([2, 3]);
    const mixedHostOutput = new Float32Array(4).fill(-707);
    const mixedHostBound = program.bind({
      weights: nativeWeights,
      input: mixedHostInput,
      output: mixedHostOutput,
    });
    try {
      expectClose(mixedHostBound.step(), [5, 3, 3]);
      if (mixedHostOutput[3] !== -707) throw new Error("mixed native/host tiny linear step wrote past output");
      mixedHostInput.set([4, 5]);
      mixedHostOutput.fill(-606);
      expectClose(mixedHostBound.step(), [9, 5, 5]);
      if (mixedHostOutput[3] !== -606) throw new Error("second mixed native/host tiny linear step wrote past output");
    } finally {
      mixedHostBound.free();
    }
    const byteBuffer = NativeBuffer.fromBytes([1, 2, 3, 4, 5, 6, 7, 8]);
    try {
      byteBuffer.writeBytes(new Uint8Array([90, 91]), 3);
      const byteSlice = byteBuffer.readBytes(4, 2);
      if (Array.from(byteSlice).join(",") !== "3,90,91,6") {
        throw new Error(`unexpected NativeBuffer byte readback: ${Array.from(byteSlice)}`);
      }
      const byteAll = byteBuffer.readBytes();
      if (Array.from(byteAll).join(",") !== "1,2,3,90,91,6,7,8") {
        throw new Error(`unexpected NativeBuffer default byte readback: ${Array.from(byteAll)}`);
      }
      let byteWriteRejected = false;
      try {
        byteBuffer.writeBytes(new Uint8Array([1, 2]), 7);
      } catch (err) {
        byteWriteRejected = true;
        if (!String(err && (err as Error).message || err).includes("exceeds buffer byteLength")) throw err;
      }
      if (!byteWriteRejected) throw new Error("NativeBuffer.writeBytes accepted an overlarge byte range");
      const byteTarget = new Uint8Array(6).fill(231);
      const byteInto = byteBuffer.readBytesInto(byteTarget, 4, 2);
      if (byteInto !== byteTarget || byteTarget[4] !== 231 || byteTarget[5] !== 231 || Array.from(byteTarget.subarray(0, 4)).join(",") !== "3,90,91,6") {
        throw new Error(`NativeBuffer.readBytesInto did not preserve caller storage: ${Array.from(byteTarget)}`);
      }
      let byteReadRejected = false;
      try {
        byteBuffer.readBytesInto(new Uint8Array(2), 2, 7);
      } catch (err) {
        byteReadRejected = true;
        if (!String(err && (err as Error).message || err).includes("exceeds buffer byteLength")) throw err;
      }
      if (!byteReadRejected) throw new Error("NativeBuffer.readBytesInto accepted an overlarge byte range");
    } finally {
      byteBuffer.free();
    }
    const floatBuffer = NativeBuffer.fromFloat32([1.25, -2.5, 3.75]);
    try {
      expectClose(floatBuffer.readFloat32(), [1.25, -2.5, 3.75]);
      let floatWriteRejected = false;
      try {
        floatBuffer.writeFloat32([9], floatBuffer.byteLength);
      } catch (err) {
        floatWriteRejected = true;
        if (!String(err && (err as Error).message || err).includes("exceeds buffer byteLength")) throw err;
      }
      if (!floatWriteRejected) throw new Error("NativeBuffer.writeFloat32 accepted an overlarge byte range");
      let floatReadRejected = false;
      try {
        floatBuffer.readFloat32Into(new Float32Array(1), 1, floatBuffer.byteLength);
      } catch (err) {
        floatReadRejected = true;
        if (!String(err && (err as Error).message || err).includes("exceeds buffer byteLength")) throw err;
      }
      if (!floatReadRejected) throw new Error("NativeBuffer.readFloat32Into accepted an overlarge byte range");
    } finally {
      floatBuffer.free();
    }
    const nativeBound = program.bind({
      weights: nativeWeights,
      input: nativeInput,
      output: nativeOutput,
    });
    const factoryBound = program.bind({
      weights: nativeWeights,
      input: nativeInput,
      output: factoryOutput,
    });
    try {
      const nativeBoundInfo = nativeBound.inspect();
      if (
        nativeBoundInfo.modelKind !== "tiny-linear" ||
        nativeBoundInfo.backend !== "cpu" ||
        nativeBoundInfo.outputStorage !== "host" ||
        nativeBoundInfo.kvCacheStorage !== "none" ||
        nativeBoundInfo.persistentBindingCount !== 2 ||
        nativeBoundInfo.stepInputCount !== 1 ||
        nativeBoundInfo.stepOutputCount !== 1 ||
        nativeBoundInfo.hostBindingCount !== 4 ||
        nativeBoundInfo.resourceBindingCount !== 0 ||
        nativeBoundInfo.bindingShapeHash === 0n
      ) {
        throw new Error(`unexpected tiny linear session inspection: ${json(nativeBoundInfo)}`);
      }
      expectClose(nativeBound.step(), [5, 3, 3]);
      expectClose(factoryBound.step(), [5, 3, 3]);
      if (nativeOutput.readFloat32(4)[3] !== -777) throw new Error("native tiny linear step wrote past output");
      expectClose(factoryOutput.readFloat32(3), [5, 3, 3]);
      nativeWeights.writeFloat32([2, 0, 0, 0, 2, 0]);
      expectClose(nativeBound.step(), [5, 3, 3]);
      nativeBound.uploadParameterRange(0, 1);
      expectClose(nativeBound.step(), [4, 6, 0]);
      expectClose(factoryBound.step(), [5, 3, 3]);
      factoryBound.uploadParameters();
      expectClose(factoryBound.step(), [4, 6, 0]);
      let uploadRangeRejected = false;
      try {
        nativeBound.uploadParameterRange(2, 1);
      } catch (err) {
        uploadRangeRejected = true;
        if (!String(err && (err as Error).message || err).includes("shape_mismatch")) throw err;
      }
      if (!uploadRangeRejected) throw new Error("Session.uploadParameterRange accepted an invalid persistent range");
      nativeWeights.writeFloat32([1, 0, 0, 1, 1, 1]);
      nativeBound.uploadParameters();
      factoryBound.uploadParameters();
      expectClose(nativeBound.step(), [5, 3, 3]);
      expectClose(factoryBound.step(), [5, 3, 3]);
      const nativeReadTarget = new Float32Array(4).fill(-888);
      const nativeReadInto = nativeBound.readOutputInto(nativeReadTarget);
      if (nativeReadInto !== nativeReadTarget || nativeReadTarget[3] !== -888) {
        throw new Error("native-bound readOutputInto did not preserve caller storage");
      }
      expectClose(nativeReadTarget.subarray(0, 3), [5, 3, 3]);
      const nativeReadTensor = nativeBound.readOutputTensor();
      if (nativeReadTensor.shape.join(",") !== "3") {
        throw new Error(`unexpected native-bound readOutputTensor shape: ${nativeReadTensor.shape}`);
      }
      expectClose(nativeReadTensor.toFloat32Array(), [5, 3, 3]);
      const factoryReadTarget = new Float32Array(4).fill(-889);
      const factoryReadInto = factoryBound.readOutputInto(factoryReadTarget);
      if (factoryReadInto !== factoryReadTarget || factoryReadTarget[3] !== -889) {
        throw new Error("factory-bound readOutputInto did not preserve caller storage");
      }
      expectClose(factoryReadTarget.subarray(0, 3), [5, 3, 3]);
      const factoryReadTensor = factoryBound.readOutputTensor({ shape: [1, 3] });
      if (factoryReadTensor.shape.join(",") !== "1,3") {
        throw new Error(`unexpected factory-bound readOutputTensor shape: ${factoryReadTensor.shape}`);
      }
      expectClose(factoryReadTensor.toFloat32Array(), [5, 3, 3]);
      nativeInput.writeFloat32([4, 5]);
      nativeOutput.writeFloat32([-111, -111, -111, -222]);
      expectClose(nativeBound.step(), [9, 5, 5]);
      if (nativeOutput.readFloat32(4)[3] !== -222) throw new Error("second native tiny linear step wrote past output");
      nativeOutput.writeFloat32([-333, -333, -333, -444]);
      const overrideNativeOutput = new Float32Array(4).fill(-555);
      expectClose(nativeBound.step(undefined, overrideNativeOutput), [9, 5, 5]);
      if (overrideNativeOutput[3] !== -555) throw new Error("native tiny linear override wrote past output");
      if (nativeOutput.readFloat32(4)[3] !== -444) throw new Error("native tiny linear override wrote bound output");
    } finally {
      factoryBound.free();
      nativeBound.free();
      factoryOutput.free();
      nativeOutput.free();
      nativeInput.free();
      nativeWeights.free();
    }

    let mismatchedPlacementRejected = false;
    try {
      tensor([1], [1]).place(program, "input");
    } catch (err) {
      mismatchedPlacementRejected = String((err as { message?: unknown }).message ?? err).includes("Program input slot length 2");
    }
    if (!mismatchedPlacementRejected) {
      throw new Error("expected Tensor.place to reject mismatched Program input slot length");
    }
    const placedWeights = tensor([1, 0, 0, 1, 1, 1], [3, 2]).place(program, "weights");
    const placedInputTensor = tensor([2, 3], [2]);
    const placedInput = placedInputTensor.place(program, "input");
    const placedOutput = zeros([3]).place(program, "output");
    const placedRoundTrip = placedInputTensor.toNativeBuffer();
    const placedBound = program.bind({
      weights: placedWeights,
      input: placedInput,
      output: placedOutput,
    });
    try {
      expectClose(placedBound.step(), [5, 3, 3]);
      const placedTensor = Tensor.fromNativeBuffer(placedOutput, [3]);
      if (placedTensor.shape.join(",") !== "3") {
        throw new Error(`unexpected placed output tensor shape: ${placedTensor.shape}`);
      }
      expectClose(placedTensor.toFloat32Array(), [5, 3, 3]);
      const roundTripTensor = Tensor.fromNativeBuffer(placedRoundTrip, { length: 2 });
      expectClose(roundTripTensor.toFloat32Array(), [2, 3]);
    } finally {
      placedBound.free();
      placedRoundTrip.free();
      placedOutput.free();
      placedInput.free();
      placedWeights.free();
    }

    const wrappedWeightsBacking = new Float32Array([1, 0, 0, 1, 1, 1]);
    const wrappedInputBacking = new Float32Array([2, 3]);
    const wrappedOutputBacking = new Float32Array([-999, -999, -999, -777]);
    const wrappedWeights = NativeBuffer.wrapFloat32(wrappedWeightsBacking);
    const wrappedInput = NativeBuffer.wrapFloat32(wrappedInputBacking);
    const wrappedOutput = NativeBuffer.wrapFloat32(wrappedOutputBacking);
    const wrappedBound = program.bind({
      weights: wrappedWeights,
      input: wrappedInput,
      output: wrappedOutput,
    });
    try {
      expectClose(wrappedBound.step(), [5, 3, 3]);
      if (wrappedOutputBacking[3] !== -777) throw new Error("wrapped tiny linear step wrote past output");
      wrappedInputBacking.set([4, 5]);
      wrappedOutputBacking.set([-111, -111, -111, -222]);
      expectClose(wrappedBound.step(), [9, 5, 5]);
      if (wrappedOutputBacking[3] !== -222) throw new Error("second wrapped tiny linear step wrote past output");
    } finally {
      wrappedBound.free();
      wrappedOutput.free();
      wrappedInput.free();
      wrappedWeights.free();
    }
    if (wrappedOutputBacking[0] !== 9 || wrappedOutputBacking[1] !== 5 || wrappedOutputBacking[3] !== -222) {
      throw new Error("wrapped backing storage did not remain caller-owned after free");
    }

    const resourceSlotAudit: string[] = [];
    const resourceWeights = program.createBuffer("weights", {
      resource: (slot) => {
        if (
          !Object.isFrozen(slot) ||
          !Object.isFrozen(slot.requirements) ||
          !Object.isFrozen(slot.layout) ||
          !Object.isFrozen(slot.slot) ||
          slot.kind !== "weights" ||
          slot.name !== "weights" ||
          slot.role !== "persistent" ||
          slot.elementCount !== 6 ||
          slot.elementLength !== 6 ||
          slot.byteLength !== 24 ||
          slot.layout.weights.byteLength !== 24 ||
          slot.slot.name !== "weights"
        ) {
          throw new Error(`unexpected weights resource slot: ${json(slot)}`);
        }
        resourceSlotAudit.push(`${slot.kind}:${slot.role}:${slot.elementCount}:${slot.byteLength}`);
        return NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 1,
        byteOffset: 16,
        byteLength: 16 + slot.byteLength,
        access: "read",
        });
      },
    });
    const resourceOutput = program.createBuffer("output", {
      resource: (slot) => {
        if (
          !Object.isFrozen(slot) ||
          !Object.isFrozen(slot.requirements) ||
          !Object.isFrozen(slot.layout) ||
          !Object.isFrozen(slot.slot) ||
          slot.kind !== "output" ||
          slot.name !== "output" ||
          slot.role !== "step-output" ||
          slot.elementCount !== 3 ||
          slot.elementLength !== 3 ||
          slot.byteLength !== 12 ||
          slot.layout.output.byteLength !== 12 ||
          slot.slot.name !== "output"
        ) {
          throw new Error(`unexpected output resource slot: ${json(slot)}`);
        }
        resourceSlotAudit.push(`${slot.kind}:${slot.role}:${slot.elementCount}:${slot.byteLength}`);
        return NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 2,
        byteLength: slot.byteLength,
        access: "write",
        });
      },
    });
    const resourceInput = program.createBuffer("input", {
      resource: (slot) => {
        if (
          !Object.isFrozen(slot) ||
          !Object.isFrozen(slot.requirements) ||
          !Object.isFrozen(slot.layout) ||
          !Object.isFrozen(slot.slot) ||
          slot.kind !== "input" ||
          slot.name !== "input" ||
          slot.role !== "step-input" ||
          slot.elementCount !== 2 ||
          slot.elementLength !== 2 ||
          slot.byteLength !== 8 ||
          slot.layout.input.byteLength !== 8 ||
          slot.slot.name !== "input"
        ) {
          throw new Error(`unexpected input resource slot: ${json(slot)}`);
        }
        resourceSlotAudit.push(`${slot.kind}:${slot.role}:${slot.elementCount}:${slot.byteLength}`);
        return NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 3,
        byteLength: slot.byteLength,
        access: "read",
        });
      },
    });
    if (resourceSlotAudit.join("|") !== "weights:persistent:6:24|output:step-output:3:12|input:step-input:2:8") {
      throw new Error(`unexpected resource slot audit: ${resourceSlotAudit.join("|")}`);
    }
    try {
      const resourceWeightsInfo = resourceWeights.inspect();
      if (
        !Object.isFrozen(resourceWeightsInfo) ||
        !Object.isFrozen(resourceWeightsInfo.access) ||
        resourceWeightsInfo.storage !== "external-resource" ||
        resourceWeightsInfo.placement !== "webgpu" ||
        resourceWeightsInfo.access?.read !== true ||
        resourceWeightsInfo.access?.write !== false ||
        resourceWeightsInfo.handle !== 1 ||
        resourceWeightsInfo.byteOffset !== 16 ||
        resourceWeightsInfo.byteLength !== 16 + requirements.weightsLen * requirements.scalarBytes ||
        resourceWeightsInfo.resourceByteLength !== 32 + requirements.weightsLen * requirements.scalarBytes
      ) {
        throw new Error(`unexpected external weights inspection: ${json(resourceWeightsInfo)}`);
      }
      const resourceOutputInfo = resourceOutput.inspect();
      if (
        !Object.isFrozen(resourceOutputInfo) ||
        !Object.isFrozen(resourceOutputInfo.access) ||
        resourceOutputInfo.storage !== "external-resource" ||
        resourceOutputInfo.placement !== "webgpu" ||
        resourceOutputInfo.access?.read !== false ||
        resourceOutputInfo.access?.write !== true ||
        resourceOutputInfo.handle !== 2 ||
        resourceOutputInfo.byteOffset !== 0 ||
        resourceOutputInfo.byteLength !== requirements.outputByteLength ||
        resourceOutputInfo.resourceByteLength !== requirements.outputByteLength
      ) {
        throw new Error(`unexpected external output inspection: ${json(resourceOutputInfo)}`);
      }
      const resourceInputInfo = resourceInput.inspect();
      if (
        !Object.isFrozen(resourceInputInfo) ||
        !Object.isFrozen(resourceInputInfo.access) ||
        resourceInputInfo.storage !== "external-resource" ||
        resourceInputInfo.placement !== "webgpu" ||
        resourceInputInfo.access?.read !== true ||
        resourceInputInfo.access?.write !== false ||
        resourceInputInfo.handle !== 3 ||
        resourceInputInfo.byteLength !== requirements.inputLen * requirements.scalarBytes
      ) {
        throw new Error(`unexpected external input inspection: ${json(resourceInputInfo)}`);
      }
      if (resourceWeights.size() !== 16 + requirements.weightsLen * requirements.scalarBytes) throw new Error("unexpected external resource byte length");
      let readRejected = false;
      try {
        resourceWeights.readFloat32(1);
      } catch (err) {
        readRejected = isUnsupportedError(err);
      }
      if (!readRejected) throw new Error("expected external resource readback to be unsupported");

      let resourceBindRejected = false;
      try {
        program.bind({
          weights: resourceWeights,
          input: resourceInput,
          output: resourceOutput,
        });
      } catch (err) {
        resourceBindRejected = isUnsupportedError(err);
      }
      if (!resourceBindRejected) throw new Error("expected host-only tiny linear resource binding to be unsupported");
    } finally {
      resourceInput.free();
      resourceOutput.free();
      resourceWeights.free();
    }
  } finally {
    program.free();
  }

  const webgpuProgram = model.compile({ backend: "webgpu" });
  try {
    const webgpuInspection = webgpuProgram.inspect();
    const webgpuCapabilities = webgpuProgram.capabilities();
    if (
      webgpuInspection.backend !== "webgpu" ||
      webgpuInspection.executionSupported !== runtimeReportsNativeWgpu ||
      !webgpuInspection.externalResourcesSupported ||
      webgpuInspection.bufferCount !== 5 ||
      webgpuInspection.bufferElementCount !== 17 ||
      webgpuInspection.bufferByteLength !== 68 ||
      webgpuInspection.initialUploadCount !== 1 ||
      webgpuInspection.qweightCount !== 0 ||
      webgpuInspection.opCount === 0 ||
      webgpuInspection.runtimePatchMaxCacheWritePos !== u32Max ||
      webgpuInspection.runtimePatchMaxAttentionSeqKv !== u32Max ||
      webgpuInspection.commandCount === 0 ||
      webgpuInspection.commandStencilHash === 0n ||
      commandCategoryTotal(webgpuInspection) === 0
    ) {
      throw new Error(`unexpected tiny linear WebGPU inspection: ${json(webgpuInspection)}`);
    }
    if (runtimeReportsNativeWgpu) {
      if (
        webgpuInspection.backendDispatchCount === 0 ||
        !webgpuInspection.dispatchPlanSupported ||
        webgpuInspection.dispatchPlanCoveredOpCount !== webgpuInspection.opCount ||
        webgpuInspection.dispatchPlanFirstUnsupportedOp !== null ||
        webgpuInspection.dispatchPlanProjectionCount === 0
      ) {
        throw new Error(`unexpected native tiny linear WebGPU dispatch plan: ${json(webgpuInspection)}`);
      }
    } else if (
      webgpuInspection.backendDispatchCount !== 0 ||
      webgpuInspection.dispatchPlanSupported ||
      webgpuInspection.dispatchPlanCoveredOpCount !== 0 ||
      webgpuInspection.dispatchPlanFirstUnsupportedOp !== null
    ) {
      throw new Error(`unexpected compile-only tiny linear WebGPU dispatch plan: ${json(webgpuInspection)}`);
    }
    if (
      webgpuCapabilities.backend !== "webgpu" ||
      webgpuCapabilities.mode !== (runtimeReportsNativeWgpu ? "executable" : "resource-probe") ||
      webgpuCapabilities.canExecute !== runtimeReportsNativeWgpu ||
      webgpuProgram.canExecute() !== runtimeReportsNativeWgpu ||
      webgpuProgram.executionMode() !== (runtimeReportsNativeWgpu ? "executable" : "resource-probe") ||
      !webgpuCapabilities.canBindExternalResources ||
      !webgpuProgram.canBindExternalResources() ||
      webgpuCapabilities.hasFullDispatchPlan !== runtimeReportsNativeWgpu ||
      webgpuProgram.hasFullDispatchPlan() !== runtimeReportsNativeWgpu
    ) {
      throw new Error(`unexpected WebGPU tiny linear capabilities: ${json(webgpuCapabilities)}`);
    }
    expectCapabilityDispatchPlan(webgpuCapabilities, webgpuInspection, "WebGPU tiny linear");

    if (webgpuInspection.executionSupported) {
      const executableWebgpu = webgpuProgram.bind({
        weights: [1, 0, 0, 1, 1, 1],
      });
      try {
        expectClose(executableWebgpu.step([2, 3]), [5, 3, 3]);
      } finally {
        executableWebgpu.free();
      }

      const nativeWebgpuWeights = webgpuProgram.createBuffer("weights");
      const nativeWebgpuInput = webgpuProgram.createBuffer("input");
      const nativeWebgpuOutput = webgpuProgram.createBuffer("output");
      try {
        nativeWebgpuWeights.writeFloat32([1, 0, 0, 1, 1, 1]);
        nativeWebgpuInput.writeFloat32([2, 3]);
        const nativeWebgpu = webgpuProgram.bind({
          weights: nativeWebgpuWeights,
          input: nativeWebgpuInput,
          output: nativeWebgpuOutput,
        });
        try {
          expectClose(nativeWebgpu.step(), [5, 3, 3]);
          nativeWebgpuWeights.writeFloat32([9, 9, 9, 9, 9, 9]);
          nativeWebgpuInput.writeFloat32([4, 5]);
          expectClose(nativeWebgpu.step(), [9, 5, 5]);
          nativeWebgpu.resetRuntimeProfile();
          nativeWebgpuInput.writeFloat32([6, 7]);
          nativeWebgpu.advance();
          expectClose(nativeWebgpuOutput.readFloat32(3), [9, 5, 5]);
          const noOutputProfile = nativeWebgpu.runtimeProfile();
          if (noOutputProfile.callCount !== 1 || noOutputProfile.syncCount !== 0) {
            throw new Error(`expected no-readback tiny linear WebGPU advance profile, got ${json(noOutputProfile)}`);
          }
        } finally {
          nativeWebgpu.free();
        }
      } finally {
        nativeWebgpuOutput.free();
        nativeWebgpuInput.free();
        nativeWebgpuWeights.free();
      }

      const webgpuDevice = webgpuProgram.device("webgpu");
      const deviceWebgpuWeights = webgpuDevice.createWeightsBuffer();
      const deviceWebgpuBias = webgpuDevice.createBiasBuffer();
      const deviceWebgpuInput = webgpuDevice.createInputBuffer();
      const deviceWebgpuOutput = webgpuDevice.createOutputBuffer();
      try {
        deviceWebgpuWeights.writeFloat32([1, 0, 0, 1, 1, 1]);
        deviceWebgpuBias.writeFloat32([10, -1, 2]);
        deviceWebgpuInput.writeFloat32([2, 3]);
        const deviceWebgpu = webgpuProgram.bind({
          weights: deviceWebgpuWeights,
          bias: deviceWebgpuBias,
          input: deviceWebgpuInput,
          output: deviceWebgpuOutput,
        });
        try {
          const deviceInfo = deviceWebgpu.inspect();
          if (
            deviceInfo.backend !== "webgpu" ||
            deviceInfo.outputStorage !== "external-resource" ||
            deviceInfo.hostBindingCount !== 0 ||
            deviceInfo.resourceBindingCount !== 4 ||
            deviceInfo.bindingShapeHash === 0n
          ) {
            throw new Error(`unexpected executable tiny linear WebGPU device-buffer session inspection: ${json(deviceInfo)}`);
          }
          expectClose(deviceWebgpu.step(), [15, 2, 5]);
          deviceWebgpu.resetRuntimeProfile();
          deviceWebgpuInput.writeFloat32([6, 7]);
          deviceWebgpu.advance();
          expectClose(deviceWebgpuOutput.readFloat32(3), [23, 6, 9]);
          const deviceNoOutputProfile = deviceWebgpu.runtimeProfile();
          if (deviceNoOutputProfile.callCount !== 1 || deviceNoOutputProfile.syncCount !== 0) {
            throw new Error(`expected no-readback tiny linear WebGPU device-buffer advance profile, got ${json(deviceNoOutputProfile)}`);
          }
        } finally {
          deviceWebgpu.free();
        }

        const deviceHandle = webgpuProgram.deviceHandle("webgpu");
        if (!Number.isSafeInteger(deviceHandle) || deviceHandle <= 0) {
          throw new Error(`expected tiny linear WebGPU device handle, got ${deviceHandle}`);
        }
        if (webgpuDevice.handle !== deviceHandle || webgpuDevice.placement !== "webgpu") {
          throw new Error(`unexpected tiny linear WebGPU ProgramDevice: ${json({ handle: webgpuDevice.handle, placement: webgpuDevice.placement })}`);
        }
        let cpuDeviceRejected = false;
        try {
          webgpuProgram.deviceHandle("cpu");
        } catch (err) {
          cpuDeviceRejected = isUnsupportedError(err);
        }
        if (!cpuDeviceRejected) throw new Error("expected tiny linear WebGPU CPU device handle query to be unsupported");

        const weightsInfo = deviceWebgpuWeights.inspect();
        const biasInfo = deviceWebgpuBias.inspect();
        const inputInfo = deviceWebgpuInput.inspect();
        const outputInfo = deviceWebgpuOutput.inspect();
        let wrongDeviceRejected = false;
        try {
          const badImport = webgpuProgram.importDeviceBuffer("weights", {
            deviceHandle: deviceHandle + 1,
            bufferHandle: weightsInfo.handle,
            byteLength: weightsInfo.byteLength,
          });
          badImport.free();
        } catch (err) {
          wrongDeviceRejected = isUnsupportedError(err);
        }
        if (!wrongDeviceRejected) throw new Error("expected tiny linear WebGPU wrong-device buffer import to be unsupported");

        let unalignedImportRejected = false;
        try {
          const badImport = webgpuProgram.importDeviceBuffer("weights", {
            deviceHandle,
            bufferHandle: weightsInfo.handle,
            byteOffset: 4,
            byteLength: weightsInfo.byteLength,
          });
          badImport.free();
        } catch (err) {
          unalignedImportRejected = isImportRejectedError(err);
        }
        if (!unalignedImportRejected) throw new Error("expected tiny linear WebGPU unaligned buffer import to be rejected");

        let oversizedImportRejected = false;
        try {
          const badImport = webgpuProgram.importDeviceBuffer("weights", {
            deviceHandle,
            bufferHandle: weightsInfo.handle,
            byteLength: weightsInfo.byteLength + 4,
          });
          badImport.free();
        } catch (err) {
          oversizedImportRejected = isImportRejectedError(err);
        }
        if (!oversizedImportRejected) throw new Error("expected tiny linear WebGPU oversized buffer import to be rejected");

        const fakeExternalResource = NativeBuffer.externalResource({
          placement: "webgpu",
          handle: 12345,
          byteLength: weightsInfo.byteLength,
          access: "read",
        });
        let fakeObjectImportRejected = false;
        try {
          const badImport = webgpuProgram.importDeviceBuffer("weights", fakeExternalResource);
          badImport.free();
        } catch (err) {
          fakeObjectImportRejected = String(err).includes("zgml-created device buffer");
        } finally {
          fakeExternalResource.free();
        }
        if (!fakeObjectImportRejected) throw new Error("expected fake external resource object import to be rejected by the wrapper");

        const inferredBias = webgpuDevice.importBuffer("bias", {
          bufferHandle: biasInfo.handle,
        });
        if (inferredBias.byteLength !== biasInfo.byteLength) {
          throw new Error("expected inferred imported bias buffer size to match program requirement");
        }
        inferredBias.free();

        const symbolImportedBias = webgpuDevice.importBuffer("bias", {
          [webgpuInterop.bufferHandle]: biasInfo.handle,
          [webgpuInterop.byteLength]: biasInfo.byteLength,
        });
        if (symbolImportedBias.byteLength !== biasInfo.byteLength) {
          throw new Error("expected symbol imported bias buffer size to match program requirement");
        }
        symbolImportedBias.free();

        const sourceImportedInput = webgpuDevice.importBuffer("input", {
          [webgpuInterop.importSource]() {
            return {
              [webgpuInterop.bufferHandle]: inputInfo.handle,
              [webgpuInterop.byteLength]: inputInfo.byteLength,
            };
          },
        });
        if (sourceImportedInput.byteLength !== inputInfo.byteLength) {
          throw new Error("expected source-method imported input buffer size to match program requirement");
        }
        sourceImportedInput.free();

        const directSymbolImportedWeights = webgpuProgram.importDeviceBuffer("weights", {
          [webgpuInterop.deviceHandle]: deviceHandle,
          [webgpuInterop.bufferHandle]: weightsInfo.handle,
          [webgpuInterop.byteLength]: weightsInfo.byteLength,
        });
        if (directSymbolImportedWeights.byteLength !== weightsInfo.byteLength) {
          throw new Error("expected direct symbol imported weights buffer size to match program requirement");
        }
        directSymbolImportedWeights.free();

        const importedWeights = webgpuDevice.importBuffer("weights", deviceWebgpuWeights);
        if (importedWeights.byteLength !== weightsInfo.byteLength) {
          throw new Error("expected direct imported weights buffer size to match program requirement");
        }
        const importedBias = webgpuDevice.importBuffer("bias", deviceWebgpuBias);
        if (importedBias.byteLength !== biasInfo.byteLength) {
          throw new Error("expected direct imported bias buffer size to match program requirement");
        }
        const importedInput = webgpuDevice.importBuffer("input", deviceWebgpuInput);
        const importedOutput = webgpuDevice.importBuffer("output", deviceWebgpuOutput);
        if (importedInput.byteLength !== inputInfo.byteLength || importedOutput.byteLength !== outputInfo.byteLength) {
          throw new Error("expected direct imported input/output sizes to match program requirements");
        }
        deviceWebgpuOutput.free();
        deviceWebgpuInput.free();
        deviceWebgpuBias.free();
        deviceWebgpuWeights.free();
        try {
          importedInput.writeFloat32([1, 4]);
          const importedWebgpu = webgpuProgram.bind({
            weights: importedWeights,
            bias: importedBias,
            input: importedInput,
            output: importedOutput,
          });
          try {
            expectClose(importedWebgpu.step(), [15, 3, 6]);
          } finally {
            importedWebgpu.free();
          }
        } finally {
          importedOutput.free();
          importedInput.free();
          importedBias.free();
          importedWeights.free();
        }
      } finally {
        deviceWebgpuOutput.free();
        deviceWebgpuInput.free();
        deviceWebgpuBias.free();
        deviceWebgpuWeights.free();
      }
    } else {
      let webgpuHostBindRejected = false;
      try {
        webgpuProgram.bind({
          weights: [1, 0, 0, 1, 1, 1],
        });
      } catch (err) {
        webgpuHostBindRejected = isUnsupportedError(err);
      }
      if (!webgpuHostBindRejected) throw new Error("expected tiny linear WebGPU host bind to be unsupported");

      let webgpuDeviceRejected = false;
      try {
        webgpuProgram.device("webgpu");
      } catch (err) {
        webgpuDeviceRejected = isUnsupportedError(err);
      }
      if (!webgpuDeviceRejected) throw new Error("expected compile-only tiny linear WebGPU ProgramDevice to be unsupported");

      let webgpuDeviceBufferRejected = false;
      try {
        const badBuffer = webgpuProgram.createBuffer("weights", { placement: "webgpu" });
        badBuffer.free();
      } catch (err) {
        webgpuDeviceBufferRejected = isUnsupportedError(err);
      }
      if (!webgpuDeviceBufferRejected) throw new Error("expected compile-only tiny linear WebGPU device buffer creation to be unsupported");

      let webgpuDeviceImportRejected = false;
      try {
        const badImport = webgpuProgram.importDeviceBuffer("weights", {
          deviceHandle: 77,
          bufferHandle: 88,
        });
        badImport.free();
      } catch (err) {
        webgpuDeviceImportRejected = isUnsupportedError(err);
      }
      if (!webgpuDeviceImportRejected) throw new Error("expected compile-only tiny linear WebGPU device buffer import to be unsupported");
    }

    const webgpuWeights = webgpuProgram.createWeightsBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({ placement: "webgpu", handle: 91, byteLength, access: "read" }),
    });
    const webgpuBias = webgpuProgram.createBiasBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({ placement: "webgpu", handle: 92, byteLength, access: "read" }),
    });
    const webgpuInput = webgpuProgram.createInputBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({ placement: "webgpu", handle: 93, byteLength, access: "read" }),
    });
    const webgpuOutput = webgpuProgram.createOutputBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({ placement: "webgpu", handle: 94, byteLength, access: "write" }),
    });
    try {
      if (webgpuInspection.executionSupported) {
        let webgpuResourceBindRejected = false;
        try {
          const webgpuResourceSession = webgpuProgram.bind({
            weights: webgpuWeights,
            bias: webgpuBias,
            input: webgpuInput,
            output: webgpuOutput,
          });
          webgpuResourceSession.free();
        } catch (err) {
          webgpuResourceBindRejected = isUnsupportedError(err);
        }
        if (!webgpuResourceBindRejected) throw new Error("expected executable tiny linear WebGPU resource bind to be unsupported");
      } else {
        const webgpuSession = webgpuProgram.bind({
          weights: webgpuWeights,
          bias: webgpuBias,
          input: webgpuInput,
          output: webgpuOutput,
        });
        try {
          const sessionInfo = webgpuSession.inspect();
          if (
            sessionInfo.backend !== "webgpu" ||
            sessionInfo.outputStorage !== "external-resource" ||
            sessionInfo.persistentBindingCount !== 2 ||
            sessionInfo.stepInputCount !== 1 ||
            sessionInfo.stepOutputCount !== 1 ||
            sessionInfo.hostBindingCount !== 0 ||
            sessionInfo.resourceBindingCount !== 4 ||
            sessionInfo.bindingShapeHash === 0n
          ) {
            throw new Error(`unexpected tiny linear WebGPU resource session inspection: ${json(sessionInfo)}`);
          }
          let webgpuStepRejected = false;
          try {
            webgpuSession.step();
          } catch (err) {
            webgpuStepRejected = isUnsupportedError(err);
          }
          if (!webgpuStepRejected) throw new Error("expected tiny linear WebGPU resource step to be unsupported");
          let webgpuAdvanceRejected = false;
          try {
            webgpuSession.advance();
          } catch (err) {
            webgpuAdvanceRejected = isUnsupportedError(err);
          }
          if (!webgpuAdvanceRejected) throw new Error("expected tiny linear WebGPU resource advance to be unsupported");
          const webgpuResourceProfile = webgpuSession.runtimeProfile();
          if (
            webgpuResourceProfile.callCount !== 0 ||
            webgpuResourceProfile.backendOpCount !== 0 ||
            webgpuResourceProfile.fallbackOpCount !== 0 ||
            webgpuResourceProfile.syncCount !== 0 ||
            webgpuResourceProfile.runtimePatchCallCount !== 0
          ) {
            throw new Error(`tiny linear WebGPU resource probe recorded runtime work: ${json(webgpuResourceProfile)}`);
          }
        } finally {
          webgpuSession.free();
        }
      }
    } finally {
      webgpuOutput.free();
      webgpuInput.free();
      webgpuBias.free();
      webgpuWeights.free();
    }
  } finally {
    webgpuProgram.free();
  }

  const webgpuMlp = nn.sequential([
    nn.linear(2, 3, { weights: [1, 2, 3, 4, 5, 6], bias: [0.5, -0.5, 1] }),
    nn.relu(),
    nn.linear(3, 2, { weights: [1, -1, 0.5, 2, -0.5, 1], bias: [0.25, -0.75] }),
  ]);
  const webgpuMlpProgram = webgpuMlp.compile({ backend: "webgpu" });
  try {
    const webgpuMlpInspection = webgpuMlpProgram.inspect();
    const webgpuMlpCapabilities = webgpuMlpProgram.capabilities();
    if (
      webgpuMlpInspection.backend !== "webgpu" ||
      webgpuMlpInspection.executionSupported !== runtimeReportsNativeWgpu ||
      !webgpuMlpInspection.externalResourcesSupported ||
      webgpuMlpInspection.opCount === 0 ||
      webgpuMlpInspection.commandCount === 0 ||
      webgpuMlpInspection.commandStencilHash === 0n ||
      commandCategoryTotal(webgpuMlpInspection) === 0
    ) {
      throw new Error(`unexpected traced MLP WebGPU inspection: ${json(webgpuMlpInspection)}`);
    }
    const expectedWebgpuMlpMode = runtimeReportsNativeWgpu ? "executable" : "resource-probe";
    if (
      webgpuMlpCapabilities.backend !== "webgpu" ||
      webgpuMlpCapabilities.mode !== expectedWebgpuMlpMode ||
      webgpuMlpCapabilities.canExecute !== runtimeReportsNativeWgpu ||
      webgpuMlpProgram.canExecute() !== runtimeReportsNativeWgpu ||
      webgpuMlpProgram.executionMode() !== expectedWebgpuMlpMode ||
      !webgpuMlpCapabilities.canBindExternalResources ||
      !webgpuMlpProgram.canBindExternalResources() ||
      webgpuMlpCapabilities.hasFullDispatchPlan !== runtimeReportsNativeWgpu ||
      webgpuMlpProgram.hasFullDispatchPlan() !== runtimeReportsNativeWgpu
    ) {
      throw new Error(`unexpected WebGPU traced MLP capabilities: ${json(webgpuMlpCapabilities)}`);
    }
    expectCapabilityDispatchPlan(webgpuMlpCapabilities, webgpuMlpInspection, "WebGPU traced MLP");

    if (webgpuMlpInspection.executionSupported) {
      const webgpuMlpSession = webgpuMlpProgram.bind(webgpuMlp.bindParameters());
      try {
        expectCloseWithin(webgpuMlpSession.step([1, 2]), [7.5, 28.75], 1e-5);
        const sessionInfo = webgpuMlpSession.inspect();
        if (
          sessionInfo.modelKind !== "module" ||
          sessionInfo.backend !== "webgpu" ||
          sessionInfo.outputStorage !== "host" ||
          sessionInfo.persistentBindingCount !== 4 ||
          sessionInfo.stepInputCount !== 1 ||
          sessionInfo.stepOutputCount !== 1 ||
          sessionInfo.hostBindingCount !== 6 ||
          sessionInfo.resourceBindingCount !== 0 ||
          sessionInfo.bindingShapeHash === 0n
        ) {
          throw new Error(`unexpected executable traced MLP WebGPU session inspection: ${json(sessionInfo)}`);
        }
      } finally {
        webgpuMlpSession.free();
      }
    } else {
      let webgpuMlpBindRejected = false;
      try {
        webgpuMlpProgram.bind(webgpuMlp.bindParameters());
      } catch (err) {
        webgpuMlpBindRejected = isUnsupportedError(err);
      }
      if (!webgpuMlpBindRejected) throw new Error("expected compile-only traced MLP WebGPU host bind to be unsupported");
    }
  } finally {
    webgpuMlpProgram.free();
  }
} finally {
  model.free();
}

const llama = TinyLlama.create();
try {
  const modelInspection = llama.inspect();
  if (
    !Object.isFrozen(modelInspection) ||
    modelInspection.modelKind !== "tiny-llama" ||
    modelInspection.vocabSize !== 8 ||
    modelInspection.maxSeqLen !== 8 ||
    modelInspection.dModel !== 4 ||
    modelInspection.nLayers !== 1 ||
    modelInspection.nHeads !== 1 ||
    modelInspection.nKvHeads !== 1 ||
    modelInspection.dFF !== 8 ||
    Math.abs(modelInspection.ropeBase - 10000) > 1e-5 ||
    Math.abs(modelInspection.rmsNormEps - 1e-6) > 1e-12 ||
    modelInspection.tiedLmHead !== false
  ) {
    throw new Error(`unexpected tiny llama model inspection: ${json(modelInspection)}`);
  }

  let badContextRejected = false;
  try {
    llama.compile({ backend: "cpu", contextLength: -1 });
  } catch (err) {
    badContextRejected = err instanceof Error && err.message.includes("contextLength");
  }
  if (!badContextRejected) throw new Error("expected invalid contextLength to be rejected before FFI");
  let badBatchRejected = false;
  try {
    llama.compile({ backend: "cpu", batch: Number.NaN });
  } catch (err) {
    badBatchRejected = err instanceof Error && err.message.includes("batch");
  }
  if (!badBatchRejected) throw new Error("expected invalid batch to be rejected before FFI");
  let unsupportedBatchRejected = false;
  try {
    llama.compile({ backend: "cpu", batch: 2 });
  } catch (err) {
    unsupportedBatchRejected =
      typeof err === "object" &&
      err !== null &&
      ((err as { status?: unknown }).status === "unsupported" ||
        (err as { code?: unknown }).code === 5 ||
        String((err as { message?: unknown }).message).includes("unsupported"));
  }
  if (!unsupportedBatchRejected) throw new Error("expected unsupported LLaMA batch envelope to come from native ABI");

  const program = llama.compile({ backend: "cpu", contextLength: 4 });
  try {
    const inspection = program.inspect();
    if (
      !Object.isFrozen(inspection) ||
      inspection.vocabSize !== 8 ||
      inspection.maxSeqLen !== 8 ||
      inspection.contextLen !== 4 ||
      inspection.batch !== 1 ||
      inspection.nLayers !== 1 ||
      inspection.nHeads !== 1 ||
      inspection.nKvHeads !== 1 ||
      inspection.semanticTokenCount !== 1 ||
      inspection.semanticStageCount === 0 ||
      inspection.semanticRuntimePatchHoles === 0
    ) {
      throw new Error(`unexpected llama program inspection: ${JSON.stringify(inspection)}`);
    }
    if (program.vocabSize !== inspection.vocabSize) {
      throw new Error(`expected program vocab ${inspection.vocabSize}, got ${program.vocabSize}`);
    }
    const requirements = program.requirements();
    if (
      requirements.modelKind !== "tiny-llama" ||
      requirements.logitsLen !== inspection.vocabSize ||
      requirements.outputLen !== inspection.vocabSize ||
      requirements.outputByteLength !== inspection.vocabSize * 4 ||
      requirements.contextLen !== inspection.contextLen ||
      requirements.batch !== inspection.batch ||
      requirements.maxTokenWindow !== inspection.contextLen
    ) {
      throw new Error(`unexpected tiny LLaMA requirements: ${json(requirements)}`);
    }
    const llamaProgramOutputShape = program.outputShape();
    if (
      llamaProgramOutputShape.join(",") !== `${inspection.vocabSize}` ||
      !Object.isFrozen(llamaProgramOutputShape)
    ) {
      throw new Error(`unexpected tiny LLaMA Program output shape: ${llamaProgramOutputShape}`);
    }
    const bufferLayout = program.bufferLayout();
    if (
      !Object.isFrozen(bufferLayout) ||
      !Object.isFrozen(bufferLayout.slots) ||
      bufferLayout.output.elementCount !== inspection.vocabSize ||
      bufferLayout.output.byteLength !== requirements.outputByteLength ||
      bufferLayout.output.role !== "step-output"
    ) {
      throw new Error(`unexpected tiny LLaMA buffer layout: ${json(bufferLayout)}`);
    }
    const executable = program.inspectExecutable();
    if (
      executable.commandCount === 0 ||
      executable.backend !== "cpu" ||
      !executable.executionSupported ||
      executable.externalResourcesSupported ||
      executable.bufferCount === 0 ||
      executable.bufferElementCount === 0 ||
      executable.bufferByteLength === 0 ||
      executable.opCount === 0 ||
      executable.runtimePatchMaxCacheWritePos !== 3 ||
      executable.runtimePatchMaxAttentionSeqKv !== 4 ||
      executable.commandStencilHash === 0n ||
      (executable.commandProjectionCount === 0 && executable.commandAttentionCount === 0) ||
      executable.runtimePatchStencilHash === 0n ||
      executable.runtimePatchHoles !== inspection.semanticRuntimePatchHoles ||
      executable.runtimePatchCacheWritePosHoles !== inspection.semanticRuntimePatchCacheWritePosHoles ||
      executable.runtimePatchAttentionSeqKvHoles !== inspection.semanticRuntimePatchAttentionSeqKvHoles
    ) {
      throw new Error(`unexpected llama executable inspection: ${json(executable)}`);
    }
    const executableCapabilities = program.capabilities();
    if (
      executableCapabilities.backend !== "cpu" ||
      executableCapabilities.mode !== "executable" ||
      !executableCapabilities.canExecute ||
      !program.canExecute() ||
      program.executionMode() !== "executable" ||
      executableCapabilities.canBindExternalResources ||
      program.canBindExternalResources() ||
      executableCapabilities.hasFullDispatchPlan ||
      program.hasFullDispatchPlan()
    ) {
      throw new Error(`unexpected llama executable capabilities: ${json(executableCapabilities)}`);
    }
    expectCapabilityDispatchPlan(executableCapabilities, executable, "tiny LLaMA");

    const a = program.bind();
    const b = program.bind();
    const advanced = program.bind();
    const bulkAdvanced = program.bind();
    const bulkExpected = program.bind();
    const prefilled = program.bind();
    const scalarStep = program.bind();
    const windowStep = program.bind();
    const sessionInfo = a.inspect();
    if (
      !Object.isFrozen(sessionInfo) ||
      sessionInfo.modelKind !== "tiny-llama" ||
      sessionInfo.backend !== "cpu" ||
      sessionInfo.outputStorage !== "host" ||
      sessionInfo.kvCacheStorage !== "host" ||
      sessionInfo.position !== 0 ||
      sessionInfo.contextLen !== inspection.contextLen ||
      sessionInfo.persistentBindingCount === 0 ||
      sessionInfo.stepInputCount === 0 ||
      sessionInfo.stepOutputCount === 0 ||
      sessionInfo.hostBindingCount === 0 ||
      sessionInfo.resourceBindingCount !== 0 ||
      sessionInfo.bindingShapeHash === 0n
    ) {
      throw new Error(`unexpected tiny LLaMA session inspection: ${json(sessionInfo)}`);
    }
    const llamaSessionOutputShape = a.outputShape();
    if (
      llamaSessionOutputShape.join(",") !== `${inspection.vocabSize}` ||
      !Object.isFrozen(llamaSessionOutputShape)
    ) {
      throw new Error(`unexpected tiny LLaMA Session output shape: ${llamaSessionOutputShape}`);
    }
    const boundOutput = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
    boundOutput.fill(-666);
    let hostBindRejected = false;
    try {
      program.bind({ output: boundOutput as unknown as NativeBuffer });
    } catch (err) {
      hostBindRejected = err instanceof Error && err.message.includes("NativeBuffer");
    }
    if (!hostBindRejected) {
      throw new Error("expected LLaMA Float32Array persistent output binding to be rejected");
    }
    const bound = program.bind();
    const nativeBoundOutput = NativeBuffer.fromFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-4444));
    const nativeBound = program.bind({ output: nativeBoundOutput });
    const autoNativeBound = program.bind({ output: "native" });
    const factoryBoundOutput = program.createBuffer("output");
    if (factoryBoundOutput.byteLength !== requirements.outputByteLength) {
      throw new Error(`unexpected llama factory output byte length: ${factoryBoundOutput.byteLength}`);
    }
    const factoryBound = program.bind({ output: factoryBoundOutput });
    const compatibleModel = TinyLlama.create();
    const wrongFamilyModel = TinyLinear.create({ inputLen: 1, outputLen: 1 });
    try {
      const compatible = program.modelCompatibility(compatibleModel);
      if (
        compatible.programModelKind !== "tiny-llama" ||
        compatible.modelKind !== "tiny-llama" ||
        compatible.compatible !== true ||
        program.acceptsModel(compatibleModel) !== true
      ) {
        throw new Error(`unexpected compatible LLaMA preflight: ${json(compatible)}`);
      }
      const incompatible = program.modelCompatibility(wrongFamilyModel);
      if (
        incompatible.programModelKind !== "tiny-llama" ||
        incompatible.modelKind !== "tiny-linear" ||
        incompatible.compatible !== false ||
        program.acceptsModel(wrongFamilyModel) !== false
      ) {
        throw new Error(`unexpected incompatible LLaMA preflight: ${json(incompatible)}`);
      }
    } finally {
      wrongFamilyModel.free();
    }
    const resourceBoundOutput = program.createOutputBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 33,
        byteLength,
        access: "write",
      }),
    });
    try {
      let resourceBindRejected = false;
      try {
        const unexpected = program.bind({ output: resourceBoundOutput });
        unexpected.free();
      } catch (err) {
        resourceBindRejected = isUnsupportedError(err);
      }
      if (!resourceBindRejected) {
        throw new Error("expected host-only LLaMA resource output binding to be unsupported");
      }

      let modelResourceBindRejected = false;
      try {
        const unexpected = program.bind({ model: compatibleModel, output: resourceBoundOutput });
        unexpected.free();
      } catch (err) {
        modelResourceBindRejected = isUnsupportedError(err);
      }
      if (!modelResourceBindRejected) {
        throw new Error("expected model-bound LLaMA resource output binding to be unsupported");
      }
    } finally {
      resourceBoundOutput.free();
    }
    const kvReq = program.kvCacheRequirements();
    if (!Object.isFrozen(kvReq) || kvReq.modelKind !== "tiny-llama" || kvReq.scalarBytes !== 4 || kvReq.layers !== 1 || kvReq.contextLength !== inspection.contextLen || kvReq.bufferByteLength !== 64) {
      throw new Error(`unexpected tiny LLaMA KV cache requirements: ${json(kvReq)}`);
    }
    const kvLayout = program.kvCacheLayout();
    if (
      !Object.isFrozen(kvLayout) ||
      !Object.isFrozen(kvLayout.slots) ||
      !Object.isFrozen(kvLayout.k) ||
      !Object.isFrozen(kvLayout.v) ||
      kvLayout.layers !== kvReq.layers ||
      kvLayout.k[0].name !== "kv-k" ||
      kvLayout.v[0].name !== "kv-v" ||
      kvLayout.k[0].byteLength !== kvReq.kBufferByteLength ||
      kvLayout.v[0].byteLength !== kvReq.vBufferByteLength ||
      kvLayout.slots.map((slot) => `${slot.kind}:${slot.name}:${slot.layer}:${slot.byteLength}`).join("|") !== `k:kv-k:0:${kvReq.kBufferByteLength}|v:kv-v:0:${kvReq.vBufferByteLength}`
    ) {
      throw new Error(`unexpected tiny LLaMA KV cache layout: ${json(kvLayout)}`);
    }
    const hostKvCache = program.createKvCache();
    let hostKvSession = null;
    try {
      hostKvSession = program.bind({ kvCache: hostKvCache });
      const hostSessionKvLayout = hostKvSession.kvCacheLayout();
      if (
        !Object.isFrozen(hostSessionKvLayout) ||
        hostSessionKvLayout.layers !== kvLayout.layers ||
        hostSessionKvLayout.k[0].byteLength !== kvLayout.k[0].byteLength ||
        hostSessionKvLayout.v[0].byteLength !== kvLayout.v[0].byteLength
      ) {
        throw new Error(`unexpected host-KV LLaMA Session KV layout: ${json(hostSessionKvLayout)}`);
      }
      const logits = hostKvSession.step(1);
      if (logits.length !== inspection.vocabSize || hostKvSession.position() !== 1) {
        throw new Error(`unexpected host-KV LLaMA step: logits=${logits.length} pos=${hostKvSession.position()}`);
      }
    } finally {
      hostKvSession?.free();
      hostKvCache.free();
    }
    let nextResourceHandle = 44;
    const kvSlotAudit: string[] = [];
    const kvCache = program.createKvCache({
      resource: (slot) => {
        if (
          !Object.isFrozen(slot) ||
          !Object.isFrozen(slot.requirements) ||
          !Object.isFrozen(slot.layout) ||
          !Object.isFrozen(slot.slot) ||
          slot.role !== "persistent" ||
          slot.scalarBytes !== kvReq.scalarBytes ||
          slot.layout.layers !== kvReq.layers ||
          slot.slot.byteLength !== slot.byteLength ||
          slot.byteOffset !== 0 ||
          (slot.kind === "k" && (slot.name !== "kv-k" || slot.byteLength !== kvReq.kBufferByteLength)) ||
          (slot.kind === "v" && (slot.name !== "kv-v" || slot.byteLength !== kvReq.vBufferByteLength))
        ) {
          throw new Error(`unexpected KV resource slot: ${json(slot)}`);
        }
        kvSlotAudit.push(`${slot.kind}:${slot.name}:${slot.layer}:${slot.byteLength}`);
        return NativeBuffer.externalResource({
        placement: "webgpu",
        handle: nextResourceHandle++,
        byteLength: slot.byteLength,
        access: "readwrite",
        });
      },
    });
    if (kvSlotAudit.join("|") !== `k:kv-k:0:${kvReq.kBufferByteLength}|v:kv-v:0:${kvReq.vBufferByteLength}`) {
      throw new Error(`unexpected KV slot audit: ${kvSlotAudit.join("|")}`);
    }
    try {
      let kvResourceBindRejected = false;
      try {
        const unexpected = program.bind({ kvCache });
        unexpected.free();
      } catch (err) {
        kvResourceBindRejected = isUnsupportedError(err);
      }
      if (!kvResourceBindRejected) {
        throw new Error("expected host-only LLaMA KV resource binding to be unsupported");
      }

      let modelKvResourceBindRejected = false;
      try {
        const unexpected = program.bind({ model: compatibleModel, kvCache });
        unexpected.free();
      } catch (err) {
        modelKvResourceBindRejected = isUnsupportedError(err);
      }
      if (!modelKvResourceBindRejected) {
        throw new Error("expected model-bound LLaMA KV resource binding to be unsupported");
      }
    } finally {
      kvCache.free();
    }
    const modelBound = program.bind({ model: compatibleModel });
    const modelFactoryOutput = program.createBuffer("output");
    const modelFactoryBound = program.bind({ model: compatibleModel, output: modelFactoryOutput });
    try {
      if (a.vocabSize !== inspection.vocabSize || b.vocabSize !== inspection.vocabSize) {
        throw new Error(`unexpected bound llama vocab sizes: a=${a.vocabSize} b=${b.vocabSize}`);
      }
      if (a.outputShape().join(",") !== `${inspection.vocabSize}` || b.outputShape().join(",") !== `${inspection.vocabSize}`) {
        throw new Error(`unexpected bound LLaMA output shapes: a=${a.outputShape()} b=${b.outputShape()}`);
      }
      const sessionLayout = a.bufferLayout();
      if (
        !Object.isFrozen(sessionLayout) ||
        !Object.isFrozen(sessionLayout.slots) ||
        sessionLayout.output.elementCount !== bufferLayout.output.elementCount ||
        sessionLayout.output.byteLength !== bufferLayout.output.byteLength ||
        sessionLayout.output.role !== "step-output"
      ) {
        throw new Error(`unexpected bound LLaMA Session buffer layout: ${json(sessionLayout)}`);
      }
      const sessionKvLayout = a.kvCacheLayout();
      if (
        !Object.isFrozen(sessionKvLayout) ||
        !Object.isFrozen(sessionKvLayout.slots) ||
        sessionKvLayout.layers !== kvLayout.layers ||
        sessionKvLayout.bufferByteLength !== kvLayout.bufferByteLength
      ) {
        throw new Error(`unexpected bound LLaMA Session KV layout: ${json(sessionKvLayout)}`);
      }
      const outA = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outA.fill(-999);
      const logitsA = a.step(0, outA);
      if (logitsA.length !== 8) throw new Error(`expected 8 llama logits, got ${logitsA.length}`);
      if (a.position() !== 1 || b.position() !== 0) {
        throw new Error(`expected independent llama positions, got a=${a.position()} b=${b.position()}`);
      }
      const scalarStepOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      const windowStepOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      const scalarStepLogits = scalarStep.step(0, scalarStepOut);
      const windowStepLogits = windowStep.executeTokens(new Uint32Array([0]), { output: windowStepOut });
      expectClose(scalarStepLogits, Array.from(windowStepLogits!));
      if (scalarStep.position() !== 1 || windowStep.position() !== 1) {
        throw new Error(`unexpected scalar/window step positions: scalar=${scalarStep.position()} window=${windowStep.position()}`);
      }
      windowStep.reset();
      const executeWindowOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-717);
      const executeWindowLogits = windowStep.execute({ tokens: new Uint32Array([0]), output: executeWindowOut });
      expectClose(executeWindowLogits!, Array.from(scalarStepLogits));
      if (executeWindowLogits!.length !== 8 || windowStep.position() !== 1 || executeWindowOut[8] !== -717) {
        throw new Error(`unexpected StepParams token-window execute: logits=${executeWindowLogits!.length} pos=${windowStep.position()}`);
      }
      const windowStepPatched = windowStep as unknown as {
        scalarTokenScratch: Uint32Array;
        scalarTokenExecuteOptionsScratch: ExecuteTokensOptions;
        outputTokenWindowOptionsScratch: ExecuteTokensOptions;
        executeTokens: (tokens: TokenIds, options?: ExecuteTokensOptions) => Float32Array | undefined;
      };
      const windowStepScalarExecuteOptionsScratch = windowStepPatched.scalarTokenExecuteOptionsScratch;
      const windowStepOutputOptionsScratch = windowStepPatched.outputTokenWindowOptionsScratch;
      const originalWindowStepExecuteTokens = windowStepPatched.executeTokens;
      const capturedWindowStepExecutes: Array<{ tokens: TokenIds; options: ExecuteTokensOptions; output?: Float32Array | false; tokensLen?: number }> = [];
      windowStepPatched.executeTokens = function captureWindowStepExecute(tokens: TokenIds, options: ExecuteTokensOptions = {}) {
        capturedWindowStepExecutes.push({
          tokens,
          options,
          output: options.output,
          tokensLen: options.tokensLen,
        });
        return originalWindowStepExecuteTokens.call(this, tokens, options);
      };
      windowStep.reset();
      const executeScalarOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-818);
      const executeScalarLogits = windowStep.execute({ token: 0, output: executeScalarOut });
      try {
        expectClose(executeScalarLogits!, Array.from(scalarStepLogits));
        if (executeScalarLogits!.length !== 8 || windowStep.position() !== 1 || executeScalarOut[8] !== -818) {
          throw new Error(`unexpected StepParams scalar execute: logits=${executeScalarLogits!.length} pos=${windowStep.position()}`);
        }
        windowStep.reset();
        const stepIntoOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-919);
        const stepIntoLogits = windowStep.stepInto(stepIntoOut, 0);
        expectClose(stepIntoLogits, Array.from(scalarStepLogits));
        if (stepIntoLogits.length !== 8 || windowStep.position() !== 1 || stepIntoOut[8] !== -919) {
          throw new Error(`unexpected stepInto logits: logits=${stepIntoLogits.length} pos=${windowStep.position()}`);
        }
        windowStep.reset();
        const executeIntoOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-929);
        const executeIntoParams = { tokens: [0] };
        const executeIntoLogits = windowStep.executeInto(executeIntoOut, executeIntoParams);
        expectClose(executeIntoLogits, Array.from(scalarStepLogits));
        if (executeIntoLogits.length !== 8 || windowStep.position() !== 1 || executeIntoOut[8] !== -929) {
          throw new Error(`unexpected executeInto logits: logits=${executeIntoLogits.length} pos=${windowStep.position()}`);
        }
        windowStep.reset();
        const prefillIntoOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-939);
        const prefillIntoOptions = { tokensLen: 1 };
        const prefillIntoLogits = windowStep.prefillInto(prefillIntoOut, [0, 99], prefillIntoOptions);
        expectClose(prefillIntoLogits, Array.from(scalarStepLogits));
        if (prefillIntoLogits.length !== 8 || windowStep.position() !== 1 || prefillIntoOut[8] !== -939) {
          throw new Error(`unexpected prefillInto logits: logits=${prefillIntoLogits.length} pos=${windowStep.position()}`);
        }
        if (
          Object.prototype.hasOwnProperty.call(executeIntoParams, "output") ||
          Object.prototype.hasOwnProperty.call(prefillIntoOptions, "output")
        ) {
          throw new Error("caller-output helpers must not mutate caller option objects");
        }
      } finally {
        windowStepPatched.executeTokens = originalWindowStepExecuteTokens;
      }
      windowStep.reset();
      const stepTensorOutput = tensor(new Float32Array(inspection.vocabSize).fill(-949), [inspection.vocabSize]);
      const stepTensorLogits = windowStep.stepTensor(0, { output: stepTensorOutput });
      if (stepTensorLogits.shape.join(",") !== `${inspection.vocabSize}` || windowStep.position() !== 1) {
        throw new Error(`unexpected LLaMA stepTensor result: shape=${stepTensorLogits.shape} pos=${windowStep.position()}`);
      }
      expectClose(stepTensorLogits.toFloat32Array(), Array.from(scalarStepLogits));
      expectClose(stepTensorOutput.toFloat32Array(), Array.from(scalarStepLogits));
      windowStep.reset();
      const executeTensorOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-959);
      const executeTensorLogits = windowStep.executeTensor({ tokens: [0], output: executeTensorOut });
      if (executeTensorLogits.shape.join(",") !== `${inspection.vocabSize}` || windowStep.position() !== 1 || executeTensorOut[inspection.vocabSize] !== -959) {
        throw new Error(`unexpected LLaMA executeTensor result: shape=${executeTensorLogits.shape} pos=${windowStep.position()}`);
      }
      expectClose(executeTensorLogits.toFloat32Array(), Array.from(scalarStepLogits));
      windowStep.reset();
      const prefillTensorLogits = windowStep.prefillTensor([0, 99], { tokensLen: 1, shape: [1, inspection.vocabSize] });
      if (prefillTensorLogits.shape.join(",") !== `1,${inspection.vocabSize}` || windowStep.position() !== 1) {
        throw new Error(`unexpected LLaMA prefillTensor result: shape=${prefillTensorLogits.shape} pos=${windowStep.position()}`);
      }
      expectClose(prefillTensorLogits.toFloat32Array(), Array.from(scalarStepLogits));
      const llamaProgramProfile = program.runtimeProfile();
      if (
        !Object.isFrozen(llamaProgramProfile) ||
        llamaProgramProfile.callCount !== 0 ||
        llamaProgramProfile.runtimePatchCallCount !== 0 ||
        llamaProgramProfile.commandCount === 0
      ) {
        throw new Error(`expected cold LLaMA program profile, got ${json(llamaProgramProfile)}`);
      }
      const llamaSessionProfile = a.runtimeProfile();
      if (
        !Object.isFrozen(llamaSessionProfile) ||
        llamaSessionProfile.callCount !== 1 ||
        llamaSessionProfile.runtimePatchCallCount !== 1 ||
        llamaSessionProfile.commandCount === 0
      ) {
        throw new Error(`unexpected LLaMA session profile: ${json(llamaSessionProfile)}`);
      }
      a.resetRuntimeProfile();
      const resetLlamaProfile = a.runtimeProfile();
      if (
        !Object.isFrozen(resetLlamaProfile) ||
        resetLlamaProfile.callCount !== 0 ||
        resetLlamaProfile.runtimePatchCallCount !== 0 ||
        resetLlamaProfile.commandCount === 0
      ) {
        throw new Error(`unexpected reset LLaMA session profile: ${json(resetLlamaProfile)}`);
      }
      let unboundArgmaxRejected = false;
      try {
        a.stepArgmax(0);
      } catch (err) {
        unboundArgmaxRejected = err instanceof Error && err.message.includes("NativeBuffer");
      }
      if (!unboundArgmaxRejected) {
        throw new Error("expected unbound LLaMA stepArgmax to require NativeBuffer output binding");
      }

      a.reset();
      if (a.position() !== 0) {
        throw new Error(`expected reset llama position 0, got ${a.position()}`);
      }
      const outReset = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outReset.fill(-888);
      const logitsReset = a.step(0, outReset);
      expectClose(logitsReset, Array.from(logitsA));
      if (outReset[8] !== -888) {
        throw new Error("reset llama step wrote past vocab logits");
      }

      const logitsBound = bound.step(0, boundOutput);
      expectClose(logitsBound, Array.from(logitsA));
      if (boundOutput[8] !== -666) {
        throw new Error("bound-output llama step wrote past vocab logits");
      }
      const logitsNativeBound = nativeBound.step(0);
      expectClose(logitsNativeBound, Array.from(logitsA));
      const outModelBound = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outModelBound.fill(-1212);
      expectClose(modelBound.step(0, outModelBound), Array.from(logitsA));
      if (outModelBound[8] !== -1212) {
        throw new Error("model-bound llama step wrote past vocab logits");
      }
      expectClose(modelFactoryBound.step(0), Array.from(logitsA));
      expectClose(modelFactoryOutput.readFloat32(inspection.vocabSize), Array.from(logitsA));
      const modelFactoryReadTarget = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1313);
      const modelFactoryReadInto = modelFactoryOutput.readFloat32Into(modelFactoryReadTarget, inspection.vocabSize);
      if (modelFactoryReadInto !== modelFactoryReadTarget || modelFactoryReadTarget[inspection.vocabSize] !== -1313) {
        throw new Error("NativeBuffer.readFloat32Into did not preserve caller-owned target storage");
      }
      expectClose(modelFactoryReadTarget.subarray(0, inspection.vocabSize), Array.from(logitsA));
      const modelFactoryReadTensor = modelFactoryBound.readOutputTensor({ shape: [1, inspection.vocabSize] });
      if (modelFactoryReadTensor.shape.join(",") !== `1,${inspection.vocabSize}`) {
        throw new Error(`unexpected model-bound LLaMA readOutputTensor shape: ${modelFactoryReadTensor.shape}`);
      }
      expectClose(modelFactoryReadTensor.toFloat32Array(), Array.from(logitsA));
      const expectedArgmax = argmax(logitsA);
      const callerArgmax = a.argmaxToken(logitsA);
      if (callerArgmax.token !== expectedArgmax.token || callerArgmax.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected caller logits argmax: ${JSON.stringify(callerArgmax)}`);
      }
      const callerSample = a.sampleToken(logitsA, { topK: 1, seed: 123, temperature: 1 });
      if (callerSample.token !== expectedArgmax.token || callerSample.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected caller logits sample: ${JSON.stringify(callerSample)}`);
      }
      const nativeArgmax = nativeBound.argmaxToken();
      if (nativeArgmax.token !== expectedArgmax.token || nativeArgmax.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected native logits argmax: ${JSON.stringify(nativeArgmax)}`);
      }
      const autoNativeArgmax = autoNativeBound.stepArgmax(0);
      if (autoNativeArgmax.token !== expectedArgmax.token || autoNativeArgmax.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected auto-native step argmax: ${JSON.stringify(autoNativeArgmax)}`);
      }
      if (autoNativeBound.position() !== 1) {
        throw new Error(`expected auto-native stepArgmax position 1, got ${autoNativeBound.position()}`);
      }
      const autoNativeReadTarget = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-5151);
      const autoNativeRead = autoNativeBound.readOutputInto(autoNativeReadTarget);
      if (autoNativeRead !== autoNativeReadTarget || autoNativeReadTarget[inspection.vocabSize] !== -5151) {
        throw new Error("auto-native readOutputInto did not preserve caller-owned target storage");
      }
      expectClose(autoNativeReadTarget.subarray(0, inspection.vocabSize), Array.from(logitsA));
      const autoNativeTensor = autoNativeBound.readOutputTensor();
      if (autoNativeTensor.shape.join(",") !== `${inspection.vocabSize}`) {
        throw new Error(`unexpected auto-native readOutputTensor shape: ${autoNativeTensor.shape}`);
      }
      expectClose(autoNativeTensor.toFloat32Array(), Array.from(logitsA));
      const nativeSample = nativeBound.sampleToken(undefined, { topK: 1, seed: 123, temperature: 1 });
      if (nativeSample.token !== expectedArgmax.token || nativeSample.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected native logits sample: ${JSON.stringify(nativeSample)}`);
      }
      if (nativeBoundOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel)[8] !== -4444) {
        throw new Error("native bound-output llama step wrote past vocab logits");
      }
      nativeBound.reset();
      nativeBoundOutput.writeFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-3333));
      const nativeOverrideOutput = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-7777);
      const logitsNativeOverride = nativeBound.step(0, nativeOverrideOutput);
      expectClose(logitsNativeOverride, Array.from(logitsA));
      if (nativeOverrideOutput[8] !== -7777) {
        throw new Error("native bound-output llama override wrote past vocab logits");
      }
      expectClose(nativeBoundOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel), Array(TinyLlamaSessionVocabPlusSentinel).fill(-3333));
      nativeBound.reset();
      nativeBoundOutput.writeFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-2222));
      const nativeStepArgmax = nativeBound.stepArgmax(0);
      if (nativeStepArgmax.token !== expectedArgmax.token || nativeStepArgmax.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected native step argmax: ${JSON.stringify(nativeStepArgmax)}`);
      }
      if (nativeBound.position() !== 1) {
        throw new Error(`expected native stepArgmax position 1, got ${nativeBound.position()}`);
      }
      if (nativeBoundOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel)[8] !== -2222) {
        throw new Error("native stepArgmax wrote past vocab logits");
      }
      nativeBound.reset();
      nativeBoundOutput.writeFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1111));
      const nativeStepSample = nativeBound.stepSample(0, { topK: 1, seed: 123, temperature: 1 });
      if (nativeStepSample.token !== expectedArgmax.token || nativeStepSample.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected native step sample: ${JSON.stringify(nativeStepSample)}`);
      }
      if (nativeBound.position() !== 1) {
        throw new Error(`expected native stepSample position 1, got ${nativeBound.position()}`);
      }
      if (nativeBoundOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel)[8] !== -1111) {
        throw new Error("native stepSample wrote past vocab logits");
      }
      const factoryStepSample = factoryBound.stepSample(0, { topK: 1, seed: 123, temperature: 1 });
      if (factoryStepSample.token !== expectedArgmax.token || factoryStepSample.logit !== expectedArgmax.logit) {
        throw new Error(`unexpected factory step sample: ${JSON.stringify(factoryStepSample)}`);
      }
      expectClose(factoryBoundOutput.readFloat32(inspection.vocabSize), Array.from(logitsA));
      const scalarScratch = (factoryBound as unknown as { scalarTokenScratch: Uint32Array }).scalarTokenScratch;
      if (!(scalarScratch instanceof Uint32Array) || scalarScratch.length !== 1) {
        throw new Error("expected native scalar helpers to own a one-token scratch buffer");
      }
      const scalarWindowOptionsScratch = (factoryBound as unknown as { scalarTokenWindowOptionsScratch: TokenWindowOptions }).scalarTokenWindowOptionsScratch;
      const scalarSampleOptionsScratch = (factoryBound as unknown as { scalarTokenSampleOptionsScratch: TokenSampleOptions }).scalarTokenSampleOptionsScratch;
      const noOutputWindowOptionsScratch = (factoryBound as unknown as { noOutputTokenWindowOptionsScratch: ExecuteTokensOptions }).noOutputTokenWindowOptionsScratch;
      if (
        scalarWindowOptionsScratch?.tokensLen !== 1 ||
        scalarSampleOptionsScratch?.tokensLen !== 1 ||
        noOutputWindowOptionsScratch?.output !== false
      ) {
        throw new Error("expected native scalar/no-output helpers to own reusable option scratch");
      }
      const scalarPatched = factoryBound as unknown as {
        executeTokens: (tokens: TokenIds, options?: ExecuteTokensOptions) => Float32Array | undefined;
        executeTokensArgmax: (tokens: TokenIds, options?: TokenWindowOptions) => TokenArgmaxResult;
        executeTokensSample: (tokens: TokenIds, options?: TokenSampleOptions) => TokenSampleResult;
      };
      const originalExecuteTokens = scalarPatched.executeTokens;
      const originalExecuteTokensArgmax = scalarPatched.executeTokensArgmax;
      const originalExecuteTokensSample = scalarPatched.executeTokensSample;
      let capturedScalarAdvance: { tokens: TokenIds; options: ExecuteTokensOptions } | null = null;
      let capturedScalarArgmax: { tokens: TokenIds; options: TokenWindowOptions } | null = null;
      let capturedScalarSample: { tokens: TokenIds; options: TokenSampleOptions } | null = null;
      try {
        scalarPatched.executeTokens = function captureScalarExecute(tokens: TokenIds, options: ExecuteTokensOptions = {}) {
          capturedScalarAdvance = { tokens, options };
          return originalExecuteTokens.call(this, tokens, options);
        };
        scalarPatched.executeTokensArgmax = function captureScalarArgmax(tokens: TokenIds, options: TokenWindowOptions = {}) {
          capturedScalarArgmax = { tokens, options };
          return originalExecuteTokensArgmax.call(this, tokens, options);
        };
        scalarPatched.executeTokensSample = function captureScalarSample(tokens: TokenIds, options: TokenSampleOptions = {}) {
          capturedScalarSample = { tokens, options };
          return originalExecuteTokensSample.call(this, tokens, options);
        };
        factoryBound.reset();
        factoryBound.advance(0);
        const scalarAdvancePosition = factoryBound.position();
        factoryBound.reset();
        const scalarArgmax = factoryBound.stepArgmax(0);
        factoryBound.reset();
        const scalarSampleOptions = { topK: 1, seed: 123, temperature: 1 };
        const scalarSample = factoryBound.stepSample(0, scalarSampleOptions);
        if (
          capturedScalarAdvance !== null ||
          scalarAdvancePosition !== 1 ||
          capturedScalarArgmax?.tokens !== scalarScratch ||
          capturedScalarArgmax?.options !== scalarWindowOptionsScratch ||
          capturedScalarArgmax?.options?.tokensLen !== 1 ||
          capturedScalarSample?.tokens !== scalarScratch ||
          capturedScalarSample?.options !== scalarSampleOptionsScratch ||
          capturedScalarSample?.options?.tokensLen !== 1 ||
          capturedScalarSample?.options?.topK !== 1 ||
          capturedScalarSample?.options?.seed !== 123 ||
          capturedScalarSample?.options?.temperature !== 1 ||
          Object.prototype.hasOwnProperty.call(scalarSampleOptions, "tokensLen") ||
          scalarArgmax.token !== expectedArgmax.token ||
          scalarSample.token !== expectedArgmax.token
        ) {
          throw new Error(`native scalar helpers did not reuse one-token scratch: ${json({
            advance: capturedScalarAdvance && { same: capturedScalarAdvance.tokens === scalarScratch, options: capturedScalarAdvance.options },
            advancePosition: scalarAdvancePosition,
            argmax: capturedScalarArgmax && {
              same: capturedScalarArgmax.tokens === scalarScratch,
              sameOptions: capturedScalarArgmax.options === scalarWindowOptionsScratch,
              options: capturedScalarArgmax.options,
            },
            sample: capturedScalarSample && {
              same: capturedScalarSample.tokens === scalarScratch,
              sameOptions: capturedScalarSample.options === scalarSampleOptionsScratch,
              options: capturedScalarSample.options,
              callerOptions: scalarSampleOptions,
            },
          })}`);
        }
      } finally {
        scalarPatched.executeTokens = originalExecuteTokens;
        scalarPatched.executeTokensArgmax = originalExecuteTokensArgmax;
        scalarPatched.executeTokensSample = originalExecuteTokensSample;
      }
      nativeBound.reset();
      const expectedFirst = nativeBound.stepSample(0, { topK: 1, seed: 123, temperature: 1 });
      const expectedSecond = nativeBound.stepSample(expectedFirst.token, { topK: 1, seed: 124, temperature: 1 });
      const borrowedPrompt = new Uint32Array([0, inspection.vocabSize + 1]);
      factoryBound.reset();
      const borrowedExecuteOutput = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      borrowedExecuteOutput.fill(-616);
      const borrowedExecute = factoryBound.executeTokens(borrowedPrompt, { output: borrowedExecuteOutput, tokensLen: 1 });
      if (borrowedExecute!.length !== inspection.vocabSize || factoryBound.position() !== 1 || borrowedExecuteOutput[inspection.vocabSize] !== -616) {
        throw new Error(`unexpected borrowed execute token window: logits=${borrowedExecute!.length} pos=${factoryBound.position()}`);
      }
      expectClose(borrowedExecute!, Array.from(logitsA));
      factoryBound.reset();
      const borrowedArgmax = factoryBound.executeTokensArgmax(borrowedPrompt, { tokensLen: 1 });
      if (borrowedArgmax.token !== expectedArgmax.token || borrowedArgmax.logit !== expectedArgmax.logit || factoryBound.position() !== 1) {
        throw new Error(`unexpected borrowed argmax token window: ${json({ borrowedArgmax, position: factoryBound.position() })}`);
      }
      factoryBound.reset();
      const borrowedSample = factoryBound.executeTokensSample(borrowedPrompt, { tokensLen: 1, topK: 1, seed: 123, temperature: 1 });
      if (borrowedSample.token !== expectedFirst.token || borrowedSample.logit !== expectedFirst.logit || factoryBound.position() !== 1) {
        throw new Error(`unexpected borrowed sampled token window: ${json({ borrowedSample, position: factoryBound.position() })}`);
      }
      factoryBound.reset();
      const borrowedPrefillOutput = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      borrowedPrefillOutput.fill(-617);
      const borrowedPrefill = factoryBound.prefill(borrowedPrompt, borrowedPrefillOutput, { tokensLen: 1 });
      if (borrowedPrefill.length !== inspection.vocabSize || factoryBound.position() !== 1 || borrowedPrefillOutput[inspection.vocabSize] !== -617) {
        throw new Error(`unexpected borrowed prefill token window: logits=${borrowedPrefill.length} pos=${factoryBound.position()}`);
      }
      expectClose(borrowedPrefill, Array.from(logitsA));
      factoryBound.reset();
      const borrowedAdvanceOptions = { tokensLen: 1 };
      const originalBorrowedExecuteTokens = scalarPatched.executeTokens;
      let capturedBorrowedAdvance: { tokens: TokenIds; options: ExecuteTokensOptions } | null = null;
      try {
        scalarPatched.executeTokens = function captureBorrowedAdvance(tokens: TokenIds, options: ExecuteTokensOptions = {}) {
          capturedBorrowedAdvance = { tokens, options };
          return originalBorrowedExecuteTokens.call(this, tokens, options);
        };
        factoryBound.advanceTokens(borrowedPrompt, borrowedAdvanceOptions);
      } finally {
        scalarPatched.executeTokens = originalBorrowedExecuteTokens;
      }
      if (Object.prototype.hasOwnProperty.call(borrowedAdvanceOptions, "output")) {
        throw new Error("native advanceTokens must not mutate caller option objects");
      }
      if (factoryBound.position() !== 1) {
        throw new Error(`expected borrowed advance token window position 1, got ${factoryBound.position()}`);
      }
      factoryBound.reset();
      const generated = factoryBound.generateTokensSample([0], 2, { topK: 1, seed: 123, temperature: 1 });
      if (
        generated.tokens.length !== 2 ||
        generated.tokens[0] !== expectedFirst.token ||
        generated.tokens[1] !== expectedSecond.token ||
        generated.lastToken !== expectedSecond.token ||
        generated.lastLogit !== expectedSecond.logit
      ) {
        throw new Error(`unexpected native generated sample: ${json({
          tokens: Array.from(generated.tokens),
          lastToken: generated.lastToken,
          lastLogit: generated.lastLogit,
        })}`);
      }
      if (factoryBound.position() !== 2) {
        throw new Error(`expected generated sample position 2, got ${factoryBound.position()}`);
      }
      factoryBound.reset();
      const generatedIntoOutput = new Uint32Array([777, 777]);
      const generatedInto = factoryBound.generateTokensSampleInto([0], generatedIntoOutput, { topK: 1, seed: 123, temperature: 1 });
      if (
        generatedInto.tokens.buffer !== generatedIntoOutput.buffer ||
        generatedInto.tokens.length !== 2 ||
        generatedIntoOutput[0] !== expectedFirst.token ||
        generatedIntoOutput[1] !== expectedSecond.token ||
        generatedInto.lastToken !== expectedSecond.token ||
        generatedInto.lastLogit !== expectedSecond.logit
      ) {
        throw new Error(`unexpected native generated sample into: ${json({
          tokens: Array.from(generatedInto.tokens),
          output: Array.from(generatedIntoOutput),
          lastToken: generatedInto.lastToken,
          lastLogit: generatedInto.lastLogit,
        })}`);
      }
      if (factoryBound.position() !== 2) {
        throw new Error(`expected generated sample-into position 2, got ${factoryBound.position()}`);
      }
      factoryBound.reset();
      const borrowedGeneratedSample = factoryBound.generateTokensSample(borrowedPrompt, 2, { tokensLen: 1, topK: 1, seed: 123, temperature: 1 });
      if (
        borrowedGeneratedSample.tokens.length !== 2 ||
        borrowedGeneratedSample.tokens[0] !== expectedFirst.token ||
        borrowedGeneratedSample.tokens[1] !== expectedSecond.token ||
        borrowedGeneratedSample.lastToken !== expectedSecond.token ||
        borrowedGeneratedSample.lastLogit !== expectedSecond.logit ||
        factoryBound.position() !== 2
      ) {
        throw new Error(`unexpected borrowed native generated sample: ${json({
          tokens: Array.from(borrowedGeneratedSample.tokens),
          lastToken: borrowedGeneratedSample.lastToken,
          lastLogit: borrowedGeneratedSample.lastLogit,
          position: factoryBound.position(),
        })}`);
      }
      factoryBound.reset();
      const greedyGenerated = factoryBound.generateTokensArgmax([0], 2);
      if (
        greedyGenerated.tokens.length !== 2 ||
        greedyGenerated.tokens[0] !== expectedFirst.token ||
        greedyGenerated.tokens[1] !== expectedSecond.token ||
        greedyGenerated.lastToken !== expectedSecond.token ||
        greedyGenerated.lastLogit !== expectedSecond.logit
      ) {
        throw new Error(`unexpected native generated argmax: ${json({
          tokens: Array.from(greedyGenerated.tokens),
          lastToken: greedyGenerated.lastToken,
          lastLogit: greedyGenerated.lastLogit,
        })}`);
      }
      if (factoryBound.position() !== 2) {
        throw new Error(`expected generated argmax position 2, got ${factoryBound.position()}`);
      }
      factoryBound.reset();
      const greedyIntoOutput = new Uint32Array([777, 777]);
      const greedyInto = factoryBound.generateTokensArgmaxInto([0], greedyIntoOutput);
      if (
        greedyInto.tokens.buffer !== greedyIntoOutput.buffer ||
        greedyInto.tokens.length !== 2 ||
        greedyIntoOutput[0] !== expectedFirst.token ||
        greedyIntoOutput[1] !== expectedSecond.token ||
        greedyInto.lastToken !== expectedSecond.token ||
        greedyInto.lastLogit !== expectedSecond.logit
      ) {
        throw new Error(`unexpected native generated argmax into: ${json({
          tokens: Array.from(greedyInto.tokens),
          output: Array.from(greedyIntoOutput),
          lastToken: greedyInto.lastToken,
          lastLogit: greedyInto.lastLogit,
        })}`);
      }
      if (factoryBound.position() !== 2) {
        throw new Error(`expected generated argmax-into position 2, got ${factoryBound.position()}`);
      }
      factoryBound.reset();
      const borrowedGreedy = factoryBound.generateTokensArgmax(borrowedPrompt, 2, { tokensLen: 1 });
      if (
        borrowedGreedy.tokens.length !== 2 ||
        borrowedGreedy.tokens[0] !== expectedFirst.token ||
        borrowedGreedy.tokens[1] !== expectedSecond.token ||
        borrowedGreedy.lastToken !== expectedSecond.token ||
        borrowedGreedy.lastLogit !== expectedSecond.logit ||
        factoryBound.position() !== 2
      ) {
        throw new Error(`unexpected borrowed native generated argmax: ${json({
          tokens: Array.from(borrowedGreedy.tokens),
          lastToken: borrowedGreedy.lastToken,
          lastLogit: borrowedGreedy.lastLogit,
          position: factoryBound.position(),
        })}`);
      }
      autoNativeBound.reset();
      const autoNativeGreedy = autoNativeBound.generateTokensArgmax([0], 2);
      if (
        autoNativeGreedy.tokens.length !== 2 ||
        autoNativeGreedy.tokens[0] !== expectedFirst.token ||
        autoNativeGreedy.tokens[1] !== expectedSecond.token ||
        autoNativeGreedy.lastToken !== expectedSecond.token ||
        autoNativeGreedy.lastLogit !== expectedSecond.logit
      ) {
        throw new Error(`unexpected auto-native generated argmax: ${json({
          tokens: Array.from(autoNativeGreedy.tokens),
          lastToken: autoNativeGreedy.lastToken,
          lastLogit: autoNativeGreedy.lastLogit,
        })}`);
      }
      bound.reset();
      boundOutput.fill(-555);
      const logitsBoundPrefill = bound.prefill([0, 1], boundOutput);

      const outB = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outB.fill(-777);
      const logitsB = b.step(0, outB);
      expectClose(logitsB, Array.from(logitsA));
      if (outA[8] !== -999 || outB[8] !== -777) {
        throw new Error("llama step wrote past vocab logits");
      }

      advanced.advance(0);
      if (advanced.position() !== 1) {
        throw new Error(`expected advanced llama position 1, got ${advanced.position()}`);
      }
      const outAdvanced = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outAdvanced.fill(-333);
      const logitsAdvanced = advanced.step(1, outAdvanced);
      if (logitsAdvanced.length !== 8 || advanced.position() !== 2) {
        throw new Error(`unexpected advanced llama step: logits=${logitsAdvanced.length} pos=${advanced.position()}`);
      }
      if (outAdvanced[8] !== -333) {
        throw new Error("advanced llama step wrote past vocab logits");
      }
      expectClose(logitsBoundPrefill, Array.from(logitsAdvanced));
      if (boundOutput[8] !== -555) {
        throw new Error("bound-output llama prefill wrote past vocab logits");
      }

      let tooLongRejected = false;
      try {
        bulkAdvanced.advanceTokens([0, 1, 2, 3, 4]);
      } catch (err) {
        tooLongRejected = typeof err === "object" && err !== null && (err as { code?: unknown }).code === 3;
      }
      if (!tooLongRejected || bulkAdvanced.position() !== 0) {
        throw new Error(`expected bulk advance rejection without position mutation, pos=${bulkAdvanced.position()}`);
      }
      const outTooLongPrefill = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outTooLongPrefill.fill(-909);
      let tooLongPrefillRejected = false;
      try {
        prefilled.prefill([0, 1, 2, 3, 4], outTooLongPrefill);
      } catch (err) {
        tooLongPrefillRejected = typeof err === "object" && err !== null && (err as { code?: unknown }).code === 3;
      }
      if (!tooLongPrefillRejected || prefilled.position() !== 0 || outTooLongPrefill[8] !== -909) {
        throw new Error(`expected prefill rejection without state/output mutation, pos=${prefilled.position()}`);
      }
      bulkAdvanced.advanceTokens([0, 1]);
      if (bulkAdvanced.position() !== 2) {
        throw new Error(`expected bulk-advanced llama position 2, got ${bulkAdvanced.position()}`);
      }
      const outBulkAdvanced = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outBulkAdvanced.fill(-444);
      const logitsBulkAdvanced = bulkAdvanced.step(2, outBulkAdvanced);
      if (logitsBulkAdvanced.length !== 8 || bulkAdvanced.position() !== 3) {
        throw new Error(`unexpected bulk-advanced llama step: logits=${logitsBulkAdvanced.length} pos=${bulkAdvanced.position()}`);
      }
      if (outBulkAdvanced[8] !== -444) {
        throw new Error("bulk-advanced llama step wrote past vocab logits");
      }
      const outBulkExpected = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outBulkExpected.fill(-123);
      const logitsBulkExpected = bulkExpected.prefill([0, 1, 2], outBulkExpected);
      if (logitsBulkExpected.length !== 8 || bulkExpected.position() !== 3) {
        throw new Error(`unexpected bulk-expected llama prefill: logits=${logitsBulkExpected.length} pos=${bulkExpected.position()}`);
      }
      if (outBulkExpected[8] !== -123) {
        throw new Error("bulk-expected llama prefill wrote past vocab logits");
      }
      expectClose(logitsBulkAdvanced, Array.from(logitsBulkExpected));

      const outPrefill = new Float32Array(TinyLlamaSessionVocabPlusSentinel);
      outPrefill.fill(-222);
      const logitsPrefill = prefilled.prefill([0, 1], outPrefill);
      if (logitsPrefill.length !== 8 || prefilled.position() !== 2) {
        throw new Error(`unexpected prefill llama step: logits=${logitsPrefill.length} pos=${prefilled.position()}`);
      }
      expectClose(logitsPrefill, Array.from(logitsAdvanced));
      if (outPrefill[8] !== -222) {
        throw new Error("prefill llama step wrote past vocab logits");
      }
      console.log(`zgml bun ffi tiny llama smoke ok: ${logitsA.length} logits`);
    } finally {
      modelFactoryBound.free();
      modelFactoryOutput.free();
      modelBound.free();
      compatibleModel.free();
      factoryBound.free();
      factoryBoundOutput.free();
      autoNativeBound.free();
      nativeBound.free();
      nativeBoundOutput.free();
      bound.free();
      prefilled.free();
      windowStep.free();
      scalarStep.free();
      bulkExpected.free();
      bulkAdvanced.free();
      advanced.free();
      a.free();
      b.free();
    }
  } finally {
    program.free();
  }
} finally {
  llama.free();
}

const webgpuLlama = TinyLlama.create();
try {
  const program = webgpuLlama.compile({ backend: "webgpu", contextLength: 4 });
  try {
    const inspection = program.inspect();
    const executable = program.inspectExecutable();
    const capabilities = program.capabilities();
    const llamaWebgpuExecutable = executable.executionSupported;
    if (
      inspection.contextLen !== 4 ||
      executable.backend !== "webgpu" ||
      llamaWebgpuExecutable !== runtimeReportsExperimentalLlamaWgpu ||
      (llamaWebgpuExecutable && !runtimeReportsNativeWgpu) ||
      !executable.externalResourcesSupported ||
      executable.bufferCount === 0 ||
      executable.bufferElementCount === 0 ||
      executable.bufferByteLength === 0 ||
      executable.opCount === 0 ||
      executable.runtimePatchMaxCacheWritePos !== 3 ||
      executable.runtimePatchMaxAttentionSeqKv !== 4 ||
      executable.commandCount === 0 ||
      executable.commandStencilHash === 0n ||
      (executable.commandProjectionCount === 0 && executable.commandAttentionCount === 0) ||
      executable.runtimePatchHoles !== inspection.semanticRuntimePatchHoles
    ) {
      throw new Error(`unexpected WebGPU LLaMA inspection: ${json({ inspection, executable })}`);
    }
    if (llamaWebgpuExecutable) {
      if (
        executable.backendDispatchCount === 0 ||
        !executable.dispatchPlanSupported ||
        executable.dispatchPlanCoveredOpCount !== executable.opCount ||
        executable.dispatchPlanFirstUnsupportedOp !== null ||
        executable.dispatchPlanProjectionCount === 0 ||
        executable.dispatchPlanAttentionCount === 0
      ) {
        throw new Error(`unexpected executable WebGPU LLaMA dispatch plan: ${json(executable)}`);
      }
    } else if (
      executable.backendDispatchCount !== 0 ||
      executable.dispatchPlanSupported ||
      executable.dispatchPlanCoveredOpCount !== 0 ||
      executable.dispatchPlanFirstUnsupportedOp !== null ||
      executable.dispatchPlanProjectionCount !== 0 ||
      executable.dispatchPlanAttentionCount !== 0
    ) {
      throw new Error(`unexpected WebGPU compile-only LLaMA dispatch plan: ${json(executable)}`);
    }
    if (
      capabilities.backend !== "webgpu" ||
      capabilities.mode !== (llamaWebgpuExecutable ? "executable" : "resource-probe") ||
      capabilities.canExecute !== llamaWebgpuExecutable ||
      program.canExecute() !== llamaWebgpuExecutable ||
      program.executionMode() !== (llamaWebgpuExecutable ? "executable" : "resource-probe") ||
      !capabilities.canBindExternalResources ||
      !program.canBindExternalResources() ||
      capabilities.hasFullDispatchPlan !== llamaWebgpuExecutable ||
      program.hasFullDispatchPlan() !== llamaWebgpuExecutable
    ) {
      throw new Error(`unexpected WebGPU LLaMA capabilities: ${json(capabilities)}`);
    }
    expectCapabilityDispatchPlan(capabilities, executable, "WebGPU LLaMA");

    if (llamaWebgpuExecutable) {
      const cpuProgram = webgpuLlama.compile({ backend: "cpu", contextLength: 4 });
      try {
        const cpuSession = cpuProgram.bind();
        const webgpuSession = program.bind();
        try {
          const cpuOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-6060);
          const webgpuOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-7070);
          const cpuLogits = cpuSession.step(0, cpuOut);
          const webgpuLogits = webgpuSession.step(0, webgpuOut);
          if (
            cpuLogits.length !== 8 ||
            webgpuLogits.length !== 8 ||
            cpuSession.position() !== 1 ||
            webgpuSession.position() !== 1 ||
            cpuOut[8] !== -6060 ||
            webgpuOut[8] !== -7070
          ) {
            throw new Error(`unexpected executable WebGPU LLaMA step state: cpu=${cpuSession.position()} webgpu=${webgpuSession.position()}`);
          }
          expectCloseWithin(webgpuLogits, Array.from(cpuLogits), 1e-3);
          const webgpuProfile = webgpuSession.runtimeProfile();
          if (
            webgpuProfile.callCount !== 1 ||
            webgpuProfile.backendOpCount === 0 ||
            webgpuProfile.backendDispatchCount !== executable.backendDispatchCount ||
            webgpuProfile.fallbackOpCount !== 0 ||
            webgpuProfile.runtimePatchCallCount !== 1 ||
            webgpuProfile.commandCount === 0
          ) {
            throw new Error(`unexpected executable WebGPU LLaMA session profile: ${json(webgpuProfile)}`);
          }

          const cpuPrefillSession = cpuProgram.bind();
          const webgpuPrefillSession = program.bind();
          try {
            const cpuPrefillOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-606);
            const webgpuPrefillOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-505);
            const cpuPrefill = cpuPrefillSession.prefill([0, 1], cpuPrefillOut);
            const webgpuPrefill = webgpuPrefillSession.prefill([0, 1], webgpuPrefillOut);
            if (
              cpuPrefill.length !== 8 ||
              webgpuPrefill.length !== 8 ||
              cpuPrefillSession.position() !== 2 ||
              webgpuPrefillSession.position() !== 2 ||
              cpuPrefillOut[8] !== -606 ||
              webgpuPrefillOut[8] !== -505
            ) {
              throw new Error(`unexpected executable WebGPU LLaMA prefill state: cpu=${cpuPrefillSession.position()} webgpu=${webgpuPrefillSession.position()}`);
            }
            expectCloseWithin(webgpuPrefill, Array.from(cpuPrefill), 1e-3);
            const cpuAfterPrefillOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-404);
            const webgpuAfterPrefillOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-303);
            const cpuAfterPrefill = cpuPrefillSession.step(2, cpuAfterPrefillOut);
            const webgpuAfterPrefill = webgpuPrefillSession.step(2, webgpuAfterPrefillOut);
            if (
              cpuAfterPrefill.length !== 8 ||
              webgpuAfterPrefill.length !== 8 ||
              cpuPrefillSession.position() !== 3 ||
              webgpuPrefillSession.position() !== 3 ||
              cpuAfterPrefillOut[8] !== -404 ||
              webgpuAfterPrefillOut[8] !== -303
            ) {
              throw new Error(`unexpected executable WebGPU LLaMA decode-after-prefill state: cpu=${cpuPrefillSession.position()} webgpu=${webgpuPrefillSession.position()}`);
            }
            expectCloseWithin(webgpuAfterPrefill, Array.from(cpuAfterPrefill), 1e-3);
          } finally {
            webgpuPrefillSession.free();
            cpuPrefillSession.free();
          }

          const cpuNoOutputSession = cpuProgram.bind();
          const webgpuNoOutputSession = program.bind();
          try {
            webgpuNoOutputSession.resetRuntimeProfile();
            cpuNoOutputSession.advanceTokens([0, 1]);
            webgpuNoOutputSession.advanceTokens([0, 1]);
            if (cpuNoOutputSession.position() !== 2 || webgpuNoOutputSession.position() !== 2) {
              throw new Error(`unexpected executable WebGPU LLaMA no-output prompt positions: cpu=${cpuNoOutputSession.position()} webgpu=${webgpuNoOutputSession.position()}`);
            }
            const cpuNoOutputOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-202);
            const webgpuNoOutputOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-101);
            const cpuNoOutput = cpuNoOutputSession.step(2, cpuNoOutputOut);
            const webgpuNoOutput = webgpuNoOutputSession.step(2, webgpuNoOutputOut);
            if (
              cpuNoOutput.length !== 8 ||
              webgpuNoOutput.length !== 8 ||
              cpuNoOutputOut[8] !== -202 ||
              webgpuNoOutputOut[8] !== -101
            ) {
              throw new Error("unexpected executable WebGPU LLaMA no-output decode result");
            }
            expectCloseWithin(webgpuNoOutput, Array.from(cpuNoOutput), 1e-3);
            const noOutputProfile = webgpuNoOutputSession.runtimeProfile();
            if (
              noOutputProfile.callCount < 2 ||
              noOutputProfile.backendOpCount === 0 ||
              noOutputProfile.backendDispatchCount === 0 ||
              noOutputProfile.fallbackOpCount !== 0
            ) {
              throw new Error(`unexpected executable WebGPU LLaMA no-output profile: ${json(noOutputProfile)}`);
            }
          } finally {
            webgpuNoOutputSession.free();
            cpuNoOutputSession.free();
          }

          const device = program.device("webgpu");
          if (device.handle <= 0 || device.handle !== program.deviceHandle("webgpu")) {
            throw new Error(`unexpected executable WebGPU LLaMA device handle: ${device.handle}`);
          }
          const deviceOutput = device.createOutputBuffer();
          const importedDeviceOutput = device.importBuffer("output", deviceOutput);
          let deviceKvCache: ReturnType<typeof program.createKvCache> | null = null;
          let importedDeviceKvCache: ReturnType<typeof program.createKvCache> | null = null;
          try {
            deviceKvCache = device.createKvCache();
            const deviceKvReq = program.kvCacheRequirements();
            if (
              deviceKvCache.k.length !== deviceKvReq.layers ||
              deviceKvCache.v.length !== deviceKvReq.layers ||
              deviceKvCache.k.some((buffer) => buffer.byteLength !== deviceKvReq.kBufferByteLength) ||
              deviceKvCache.v.some((buffer) => buffer.byteLength !== deviceKvReq.vBufferByteLength)
            ) {
              throw new Error(`unexpected executable WebGPU LLaMA ProgramDevice KV cache layout: ${json({
                layers: deviceKvReq.layers,
                kBufferByteLength: deviceKvReq.kBufferByteLength,
                vBufferByteLength: deviceKvReq.vBufferByteLength,
                k: deviceKvCache.k.map((buffer) => buffer.byteLength),
                v: deviceKvCache.v.map((buffer) => buffer.byteLength),
              })}`);
            }
            importedDeviceKvCache = {
              k: deviceKvCache.k.map((buffer) => device.importBuffer("kv-k", buffer)),
              v: deviceKvCache.v.map((buffer) => device.importBuffer("kv-v", buffer)),
              free() {
                for (const buffer of this.k) buffer.free();
                for (const buffer of this.v) buffer.free();
              },
              dispose() {
                this.free();
              },
              [Symbol.dispose]() {
                this.free();
              },
            };
            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }

            const cpuResourceSession = cpuProgram.bind();
            const webgpuResourceSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const cpuResourceOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-909);
              const cpuResourceLogits = cpuResourceSession.step(0, cpuResourceOut);
              const webgpuResourceLogits = webgpuResourceSession.step(0);
              if (
                cpuResourceLogits.length !== 8 ||
                webgpuResourceLogits.length !== 8 ||
                cpuResourceSession.position() !== 1 ||
                webgpuResourceSession.position() !== 1 ||
                cpuResourceOut[8] !== -909
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource step state: cpu=${cpuResourceSession.position()} webgpu=${webgpuResourceSession.position()}`);
              }
              expectCloseWithin(webgpuResourceLogits, Array.from(cpuResourceLogits), 1e-3);
              const resourceProfile = webgpuResourceSession.runtimeProfile();
              if (
                resourceProfile.callCount !== 1 ||
                resourceProfile.backendOpCount === 0 ||
                resourceProfile.backendDispatchCount !== executable.backendDispatchCount ||
                resourceProfile.fallbackOpCount !== 0
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource step profile: ${json(resourceProfile)}`);
              }
              const resourceInfo = webgpuResourceSession.inspect();
              if (
                resourceInfo.outputStorage !== "external-resource" ||
                resourceInfo.kvCacheStorage !== "external-resource" ||
                resourceInfo.resourceBindingCount < 3
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource session inspection: ${json(resourceInfo)}`);
              }
            } finally {
              webgpuResourceSession.free();
              cpuResourceSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const cpuSelectSession = cpuProgram.bind();
            const webgpuSelectSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const cpuSelectLogits = cpuSelectSession.step(0, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1717));
              const expectedSelect = argmax(cpuSelectLogits);
              const selected = webgpuSelectSession.executeTokensArgmax([0]);
              if (
                selected.token !== expectedSelect.token ||
                Math.abs(selected.logit - expectedSelect.logit) > 1e-3 ||
                webgpuSelectSession.position() !== 1
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource argmax: ${json({ selected, expectedSelect, position: webgpuSelectSession.position() })}`);
              }
            } finally {
              webgpuSelectSession.free();
              cpuSelectSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const webgpuSampleSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const sampled = webgpuSampleSession.executeTokensSample([0], { topK: 1, seed: 123, temperature: 1 });
              const cpuSampleSession = cpuProgram.bind();
              try {
                const expectedSample = argmax(cpuSampleSession.step(0, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1616)));
                if (
                  sampled.token !== expectedSample.token ||
                  Math.abs(sampled.logit - expectedSample.logit) > 1e-3 ||
                  webgpuSampleSession.position() !== 1
                ) {
                  throw new Error(`unexpected executable WebGPU LLaMA resource top-k sample: ${json({ sampled, expectedSample, position: webgpuSampleSession.position() })}`);
                }
              } finally {
                cpuSampleSession.free();
              }
            } finally {
              webgpuSampleSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const cpuGenerateSession = cpuProgram.bind();
            const webgpuGenerateSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const expectedFirst = argmax(cpuGenerateSession.step(0, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1515)));
              const expectedSecond = argmax(cpuGenerateSession.step(expectedFirst.token, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1414)));
              const generated = webgpuGenerateSession.generateTokensArgmax([0], 2);
              if (
                generated.tokens.length !== 2 ||
                generated.tokens[0] !== expectedFirst.token ||
                generated.tokens[1] !== expectedSecond.token ||
                generated.lastToken !== expectedSecond.token ||
                Math.abs(generated.lastLogit - expectedSecond.logit) > 1e-3 ||
                webgpuGenerateSession.position() !== 2
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource greedy generation: ${json({ generated, expectedFirst, expectedSecond, position: webgpuGenerateSession.position() })}`);
              }
            } finally {
              webgpuGenerateSession.free();
              cpuGenerateSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const webgpuGenerateSampleSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const generatedSample = webgpuGenerateSampleSession.generateTokensSample([0], 2, { topK: 1, seed: 321, temperature: 1 });
              const cpuExpectedSampleSession = cpuProgram.bind();
              try {
                const expectedFirst = argmax(cpuExpectedSampleSession.step(0, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1313)));
                const expectedSecond = argmax(cpuExpectedSampleSession.step(expectedFirst.token, new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-1212)));
                if (
                  generatedSample.tokens.length !== 2 ||
                  generatedSample.tokens[0] !== expectedFirst.token ||
                  generatedSample.tokens[1] !== expectedSecond.token ||
                  generatedSample.lastToken !== expectedSecond.token ||
                  Math.abs(generatedSample.lastLogit - expectedSecond.logit) > 1e-3 ||
                  webgpuGenerateSampleSession.position() !== 2
                ) {
                  throw new Error(`unexpected executable WebGPU LLaMA resource sampled generation: ${json({ generatedSample, expectedFirst, expectedSecond, position: webgpuGenerateSampleSession.position() })}`);
                }
              } finally {
                cpuExpectedSampleSession.free();
              }
            } finally {
              webgpuGenerateSampleSession.free();
            }

            for (const buffer of [...importedDeviceKvCache.k, ...importedDeviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const cpuImportedResourceSession = cpuProgram.bind();
            const webgpuImportedResourceSession = program.bind({ output: importedDeviceOutput, kvCache: importedDeviceKvCache });
            try {
              const cpuImportedOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-818);
              const cpuImportedLogits = cpuImportedResourceSession.step(0, cpuImportedOut);
              const webgpuImportedLogits = webgpuImportedResourceSession.step(0);
              if (
                cpuImportedLogits.length !== 8 ||
                webgpuImportedLogits.length !== 8 ||
                cpuImportedResourceSession.position() !== 1 ||
                webgpuImportedResourceSession.position() !== 1 ||
                cpuImportedOut[8] !== -818
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA imported-resource step state: cpu=${cpuImportedResourceSession.position()} webgpu=${webgpuImportedResourceSession.position()}`);
              }
              expectCloseWithin(webgpuImportedLogits, Array.from(cpuImportedLogits), 1e-3);
            } finally {
              webgpuImportedResourceSession.free();
              cpuImportedResourceSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            const cpuResourcePrefillSession = cpuProgram.bind();
            const webgpuResourcePrefillSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              const cpuResourcePrefillOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-808);
              const cpuResourcePrefill = cpuResourcePrefillSession.prefill([0, 1], cpuResourcePrefillOut);
              const webgpuResourcePrefill = webgpuResourcePrefillSession.prefill([0, 1]);
              if (
                cpuResourcePrefill.length !== 8 ||
                webgpuResourcePrefill.length !== 8 ||
                cpuResourcePrefillSession.position() !== 2 ||
                webgpuResourcePrefillSession.position() !== 2 ||
                cpuResourcePrefillOut[8] !== -808
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource prefill state: cpu=${cpuResourcePrefillSession.position()} webgpu=${webgpuResourcePrefillSession.position()}`);
              }
              expectCloseWithin(webgpuResourcePrefill, Array.from(cpuResourcePrefill), 1e-3);

              const cpuResourceAfterOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-707);
              const cpuResourceAfter = cpuResourcePrefillSession.step(2, cpuResourceAfterOut);
              const webgpuResourceAfter = webgpuResourcePrefillSession.step(2);
              if (
                cpuResourceAfter.length !== 8 ||
                webgpuResourceAfter.length !== 8 ||
                cpuResourcePrefillSession.position() !== 3 ||
                webgpuResourcePrefillSession.position() !== 3 ||
                cpuResourceAfterOut[8] !== -707
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource decode-after-prefill state: cpu=${cpuResourcePrefillSession.position()} webgpu=${webgpuResourcePrefillSession.position()}`);
              }
              expectCloseWithin(webgpuResourceAfter, Array.from(cpuResourceAfter), 1e-3);
            } finally {
              webgpuResourcePrefillSession.free();
              cpuResourcePrefillSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            deviceOutput.writeFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-4141));
            const cpuResourceNoOutputSession = cpuProgram.bind();
            const webgpuResourceNoOutputSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              webgpuResourceNoOutputSession.resetRuntimeProfile();
              cpuResourceNoOutputSession.advanceTokens([0, 1]);
              webgpuResourceNoOutputSession.advanceTokens([0, 1]);
              if (cpuResourceNoOutputSession.position() !== 2 || webgpuResourceNoOutputSession.position() !== 2) {
                throw new Error(`unexpected executable WebGPU LLaMA resource no-output positions: cpu=${cpuResourceNoOutputSession.position()} webgpu=${webgpuResourceNoOutputSession.position()}`);
              }
              expectCloseWithin(
                deviceOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel - 1),
                Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-4141),
                0
              );
              const noOutputResourceProfile = webgpuResourceNoOutputSession.runtimeProfile();
              if (
                noOutputResourceProfile.callCount < 1 ||
                noOutputResourceProfile.backendOpCount === 0 ||
                noOutputResourceProfile.fallbackOpCount !== 0 ||
                noOutputResourceProfile.syncCount !== 0
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource no-output profile: ${json(noOutputResourceProfile)}`);
              }
              const cpuResourceNoOutputAfterOut = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-616);
              const cpuResourceNoOutputAfter = cpuResourceNoOutputSession.step(2, cpuResourceNoOutputAfterOut);
              const webgpuResourceNoOutputAfter = webgpuResourceNoOutputSession.step(2);
              if (
                cpuResourceNoOutputAfter.length !== 8 ||
                webgpuResourceNoOutputAfter.length !== 8 ||
                cpuResourceNoOutputSession.position() !== 3 ||
                webgpuResourceNoOutputSession.position() !== 3 ||
                cpuResourceNoOutputAfterOut[8] !== -616
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA resource no-output decode state: cpu=${cpuResourceNoOutputSession.position()} webgpu=${webgpuResourceNoOutputSession.position()}`);
              }
              expectCloseWithin(webgpuResourceNoOutputAfter, Array.from(cpuResourceNoOutputAfter), 1e-3);
            } finally {
              webgpuResourceNoOutputSession.free();
              cpuResourceNoOutputSession.free();
            }

            for (const buffer of [...deviceKvCache.k, ...deviceKvCache.v]) {
              buffer.writeFloat32(new Float32Array(buffer.byteLength / 4));
            }
            deviceOutput.writeFloat32(new Float32Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-8181));
            const webgpuOverContextSession = program.bind({ output: deviceOutput, kvCache: deviceKvCache });
            try {
              webgpuOverContextSession.advanceTokens([0, 1, 2, 3]);
              if (webgpuOverContextSession.position() !== 4) {
                throw new Error(`unexpected executable WebGPU LLaMA resource over-context fill position: ${webgpuOverContextSession.position()}`);
              }
              expectCloseWithin(
                deviceOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel - 1),
                Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-8181),
                0
              );
              const beforeOverContextReject = webgpuOverContextSession.runtimeProfile();
              if (
                beforeOverContextReject.callCount === 0 ||
                beforeOverContextReject.backendOpCount === 0 ||
                beforeOverContextReject.fallbackOpCount !== 0 ||
                beforeOverContextReject.syncCount !== 0
              ) {
                throw new Error(`unexpected executable WebGPU LLaMA over-context fill profile: ${json(beforeOverContextReject)}`);
              }
              let overContextRejected = false;
              try {
                webgpuOverContextSession.step(0);
              } catch (err) {
                overContextRejected = isShapeMismatchError(err);
              }
              if (!overContextRejected || webgpuOverContextSession.position() !== 4) {
                throw new Error(`expected executable WebGPU LLaMA over-context resource step rejection without position mutation, pos=${webgpuOverContextSession.position()}`);
              }
              expectCloseWithin(
                deviceOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel - 1),
                Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-8181),
                0
              );
              expectRuntimeProfileStable(
                beforeOverContextReject,
                webgpuOverContextSession.runtimeProfile(),
                "executable WebGPU LLaMA over-context resource step"
              );

              const overContextGreedyTokens = new Uint32Array([1234, 5678]);
              let overContextGreedyRejected = false;
              try {
                webgpuOverContextSession.generateTokensArgmaxInto([0], overContextGreedyTokens);
              } catch (err) {
                overContextGreedyRejected = isShapeMismatchError(err);
              }
              if (
                !overContextGreedyRejected ||
                webgpuOverContextSession.position() !== 4 ||
                overContextGreedyTokens[0] !== 1234 ||
                overContextGreedyTokens[1] !== 5678
              ) {
                throw new Error(`expected executable WebGPU LLaMA over-context resource greedy generation rejection without state mutation, pos=${webgpuOverContextSession.position()} tokens=${json(Array.from(overContextGreedyTokens))}`);
              }
              expectCloseWithin(
                deviceOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel - 1),
                Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-8181),
                0
              );
              expectRuntimeProfileStable(
                beforeOverContextReject,
                webgpuOverContextSession.runtimeProfile(),
                "executable WebGPU LLaMA over-context resource greedy generation"
              );

              const overContextSampleTokens = new Uint32Array([8765, 4321]);
              let overContextSampleRejected = false;
              try {
                webgpuOverContextSession.generateTokensSampleInto([0], overContextSampleTokens, {
                  topK: 1,
                  seed: 123,
                  temperature: 1,
                });
              } catch (err) {
                overContextSampleRejected = isShapeMismatchError(err);
              }
              if (
                !overContextSampleRejected ||
                webgpuOverContextSession.position() !== 4 ||
                overContextSampleTokens[0] !== 8765 ||
                overContextSampleTokens[1] !== 4321
              ) {
                throw new Error(`expected executable WebGPU LLaMA over-context resource sampled generation rejection without state mutation, pos=${webgpuOverContextSession.position()} tokens=${json(Array.from(overContextSampleTokens))}`);
              }
              expectCloseWithin(
                deviceOutput.readFloat32(TinyLlamaSessionVocabPlusSentinel - 1),
                Array(TinyLlamaSessionVocabPlusSentinel - 1).fill(-8181),
                0
              );
              expectRuntimeProfileStable(
                beforeOverContextReject,
                webgpuOverContextSession.runtimeProfile(),
                "executable WebGPU LLaMA over-context resource sampled generation"
              );
            } finally {
              webgpuOverContextSession.free();
            }
          } finally {
            importedDeviceKvCache?.free();
            importedDeviceOutput.free();
            deviceKvCache?.free();
            deviceOutput.free();
          }
        } finally {
          webgpuSession.free();
          cpuSession.free();
        }
      } finally {
        cpuProgram.free();
      }
    } else {
      let bindUnsupported = false;
      try {
        const session = program.bind();
        session.free();
      } catch (err) {
        bindUnsupported = isUnsupportedError(err);
      }
      if (!bindUnsupported) throw new Error("expected WebGPU LLaMA host bind to be unsupported");

      let deviceUnsupported = false;
      try {
        program.device("webgpu");
      } catch (err) {
        deviceUnsupported = isUnsupportedError(err);
      }
      if (!deviceUnsupported) throw new Error("expected WebGPU LLaMA ProgramDevice to be unsupported");

      let deviceBufferUnsupported = false;
      try {
        const badDeviceBuffer = program.createBuffer("output", { placement: "webgpu" });
        badDeviceBuffer.free();
      } catch (err) {
        deviceBufferUnsupported = isUnsupportedError(err);
      }
      if (!deviceBufferUnsupported) throw new Error("expected WebGPU LLaMA device output buffer to be unsupported");

      let deviceImportUnsupported = false;
      try {
        const badImport = program.importDeviceBuffer("output", {
          deviceHandle: 77,
          bufferHandle: 88,
        });
        badImport.free();
      } catch (err) {
        deviceImportUnsupported = isUnsupportedError(err);
      }
      if (!deviceImportUnsupported) throw new Error("expected WebGPU LLaMA device output import to be unsupported");
    }

    const output = program.createOutputBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 201,
        byteLength,
        access: "write",
      }),
    });
    let nextResourceHandle = 202;
    const kvCache = program.createKvCache({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: nextResourceHandle++,
        byteLength,
        access: "readwrite",
      }),
    });
    const readOnlyOutput = program.createOutputBuffer({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: 211,
        byteLength,
        access: "read",
      }),
    });
    try {
      let badOutputRejected = false;
      try {
        const unexpected = program.bind({ output: readOnlyOutput, kvCache });
        unexpected.free();
      } catch (err) {
        badOutputRejected = isUnsupportedError(err);
      }
      if (!badOutputRejected) throw new Error("expected WebGPU LLaMA read-only output resource bind to be unsupported");
    } finally {
      readOnlyOutput.free();
    }
    const writeOnlyKvCache = program.createKvCache({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: nextResourceHandle++,
        byteLength,
        access: "write",
      }),
    });
    try {
      let badKvRejected = false;
      try {
        const unexpected = program.bind({ output, kvCache: writeOnlyKvCache });
        unexpected.free();
      } catch (err) {
        badKvRejected = isUnsupportedError(err);
      }
      if (!badKvRejected) throw new Error("expected WebGPU LLaMA write-only KV resource bind to be unsupported");
    } finally {
      writeOnlyKvCache.free();
    }
    const readOnlyKvCache = program.createKvCache({
      resource: ({ byteLength }) => NativeBuffer.externalResource({
        placement: "webgpu",
        handle: nextResourceHandle++,
        byteLength,
        access: "read",
      }),
    });
    try {
      let readOnlyKvRejected = false;
      try {
        const unexpected = program.bind({ output, kvCache: readOnlyKvCache });
        unexpected.free();
      } catch (err) {
        readOnlyKvRejected = isUnsupportedError(err);
      }
      if (!readOnlyKvRejected) throw new Error("expected WebGPU LLaMA read-only KV resource bind to be unsupported");
    } finally {
      readOnlyKvCache.free();
    }
    try {
      if (llamaWebgpuExecutable) {
        let resourceBindRejected = false;
        try {
          const unexpected = program.bind({ output, kvCache });
          unexpected.free();
        } catch (err) {
          resourceBindRejected = isUnsupportedError(err);
        }
        if (!resourceBindRejected) throw new Error("expected executable WebGPU LLaMA fake resource bind to be unsupported");

        const runtimeLlama = TinyLlama.create();
        try {
          let modelResourceBindRejected = false;
          try {
            const unexpected = program.bind({ model: runtimeLlama, output, kvCache });
            unexpected.free();
          } catch (err) {
            modelResourceBindRejected = isUnsupportedError(err);
          }
          if (!modelResourceBindRejected) throw new Error("expected model-bound executable WebGPU LLaMA fake resource bind to be unsupported");
        } finally {
          runtimeLlama.free();
        }
      } else {
        let resourceBindingShapeHash = 0n;
        const resourceSession = program.bind({ output, kvCache });
        try {
          const sessionInfo = resourceSession.inspect();
          resourceBindingShapeHash = sessionInfo.bindingShapeHash;
          if (
            sessionInfo.backend !== "webgpu" ||
            sessionInfo.outputStorage !== "external-resource" ||
            sessionInfo.kvCacheStorage !== "external-resource" ||
            sessionInfo.position !== 0 ||
            sessionInfo.contextLen !== 4 ||
            sessionInfo.persistentBindingCount === 0 ||
            sessionInfo.stepInputCount === 0 ||
            sessionInfo.stepOutputCount !== 1 ||
            sessionInfo.hostBindingCount === 0 ||
            sessionInfo.resourceBindingCount !== 3 ||
            sessionInfo.bindingShapeHash === 0n
          ) {
            throw new Error(`unexpected WebGPU LLaMA resource session inspection: ${json(sessionInfo)}`);
          }
          let stepUnsupported = false;
          const resourceLogits = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-8080);
          try {
            resourceSession.step(0, resourceLogits);
          } catch (err) {
            stepUnsupported = isUnsupportedError(err);
          }
          if (!stepUnsupported) throw new Error("expected WebGPU LLaMA resource step to be unsupported");
          if (resourceSession.position() !== 0) throw new Error("WebGPU LLaMA unsupported step advanced position");
          if (resourceLogits[0] !== -8080) throw new Error("WebGPU LLaMA unsupported step mutated caller logits");
          let advanceUnsupported = false;
          try {
            resourceSession.advance(0);
          } catch (err) {
            advanceUnsupported = isUnsupportedError(err);
          }
          if (!advanceUnsupported) throw new Error("expected WebGPU LLaMA resource advance to be unsupported");
          if (resourceSession.position() !== 0) throw new Error("WebGPU LLaMA unsupported advance advanced position");
          const resourceProfile = resourceSession.runtimeProfile();
          if (
            resourceProfile.callCount !== 0 ||
            resourceProfile.backendOpCount !== 0 ||
            resourceProfile.fallbackOpCount !== 0 ||
            resourceProfile.syncCount !== 0 ||
            resourceProfile.runtimePatchCallCount !== 0
          ) {
            throw new Error(`WebGPU LLaMA unsupported step recorded runtime work: ${json(resourceProfile)}`);
          }
        } finally {
          resourceSession.free();
        }
        const runtimeLlama = TinyLlama.create();
        try {
          const modelResourceSession = program.bind({ model: runtimeLlama, output, kvCache });
          try {
            const modelSessionInfo = modelResourceSession.inspect();
            if (
              modelSessionInfo.backend !== "webgpu" ||
              modelSessionInfo.outputStorage !== "external-resource" ||
              modelSessionInfo.kvCacheStorage !== "external-resource" ||
              modelSessionInfo.position !== 0 ||
              modelSessionInfo.contextLen !== 4 ||
              modelSessionInfo.persistentBindingCount === 0 ||
              modelSessionInfo.stepInputCount === 0 ||
              modelSessionInfo.stepOutputCount !== 1 ||
              modelSessionInfo.hostBindingCount === 0 ||
              modelSessionInfo.resourceBindingCount !== 3 ||
              modelSessionInfo.bindingShapeHash === 0n ||
              modelSessionInfo.bindingShapeHash !== resourceBindingShapeHash
            ) {
              throw new Error(`unexpected model-bound WebGPU LLaMA resource session inspection: ${json(modelSessionInfo)}`);
            }
            let modelStepUnsupported = false;
            const modelResourceLogits = new Float32Array(TinyLlamaSessionVocabPlusSentinel).fill(-9090);
            try {
              modelResourceSession.step(0, modelResourceLogits);
            } catch (err) {
              modelStepUnsupported = isUnsupportedError(err);
            }
            if (!modelStepUnsupported) throw new Error("expected model-bound WebGPU LLaMA resource step to be unsupported");
            if (modelResourceSession.position() !== 0) throw new Error("model-bound WebGPU LLaMA unsupported step advanced position");
            if (modelResourceLogits[0] !== -9090) throw new Error("model-bound WebGPU LLaMA unsupported step mutated caller logits");
            let modelAdvanceUnsupported = false;
            try {
              modelResourceSession.advance(0);
            } catch (err) {
              modelAdvanceUnsupported = isUnsupportedError(err);
            }
            if (!modelAdvanceUnsupported) throw new Error("expected model-bound WebGPU LLaMA resource advance to be unsupported");
            if (modelResourceSession.position() !== 0) throw new Error("model-bound WebGPU LLaMA unsupported advance advanced position");
            const modelResourceProfile = modelResourceSession.runtimeProfile();
            if (
              modelResourceProfile.callCount !== 0 ||
              modelResourceProfile.backendOpCount !== 0 ||
              modelResourceProfile.fallbackOpCount !== 0 ||
              modelResourceProfile.syncCount !== 0 ||
              modelResourceProfile.runtimePatchCallCount !== 0
            ) {
              throw new Error(`model-bound WebGPU LLaMA unsupported step recorded runtime work: ${json(modelResourceProfile)}`);
            }
          } finally {
            modelResourceSession.free();
          }
        } finally {
          runtimeLlama.free();
        }
      }
    } finally {
      kvCache.free();
      output.free();
    }
    console.log(`zgml bun ffi ${llamaWebgpuExecutable ? "executable" : "resource-probe"} webgpu tiny llama smoke ok`);
  } finally {
    program.free();
  }
} finally {
  webgpuLlama.free();
}

const smollmPath =
  process.env.ZGML_SMOLLM_MODEL ??
  process.env.ZGML_SMOLLM_GGUF ??
  process.env.ZGML_SMOLLM_SAFETENSORS;
if (smollmPath) {
  const autoProbe = probeModel(smollmPath);
  if (
    autoProbe.modelKind !== "smollm-135m" ||
    autoProbe.vocabSize !== 49152 ||
    autoProbe.maxSeqLen !== 2048 ||
    autoProbe.dModel !== 576 ||
    autoProbe.nLayers !== 30 ||
    autoProbe.nHeads !== 9 ||
    autoProbe.nKvHeads !== 3 ||
    autoProbe.dFF !== 1536 ||
    Math.abs(autoProbe.rmsNormEps - 1e-5) > 1e-12 ||
    autoProbe.tiedLmHead !== true
  ) {
    throw new Error(`unexpected auto LLaMA model probe: ${json(autoProbe)}`);
  }
  const fixedProbe = SmolLM135M.probe(smollmPath);
  if (fixedProbe.modelKind !== autoProbe.modelKind || fixedProbe.vocabSize !== autoProbe.vocabSize) {
    throw new Error(`unexpected fixed SmolLM model probe: ${json(fixedProbe)}`);
  }
  const auto = loadModel(smollmPath);
  try {
    const modelInspection = auto.inspect();
    if (
      modelInspection.modelKind !== "smollm-135m" ||
      modelInspection.vocabSize !== 49152 ||
      modelInspection.maxSeqLen !== 2048 ||
      modelInspection.dModel !== 576 ||
      modelInspection.nLayers !== 30 ||
      modelInspection.nHeads !== 9 ||
      modelInspection.nKvHeads !== 3 ||
      modelInspection.dFF !== 1536 ||
      Math.abs(modelInspection.rmsNormEps - 1e-5) > 1e-12 ||
      modelInspection.tiedLmHead !== true
    ) {
      throw new Error(`unexpected auto LLaMA model inspection: ${json(modelInspection)}`);
    }

    const program = auto.compile({ backend: "cpu", contextLength: 16 });
    try {
      const inspection = program.inspect();
      if (
        inspection.vocabSize !== 49152 ||
        inspection.maxSeqLen !== 2048 ||
        inspection.contextLen !== 16 ||
        program.vocabSize !== inspection.vocabSize
      ) {
        throw new Error(`unexpected auto LLaMA program inspection: ${JSON.stringify(inspection)}`);
      }
      const session = program.bind();
      try {
        if (session.vocabSize !== inspection.vocabSize) {
          throw new Error(`unexpected auto LLaMA session vocab size: ${session.vocabSize}`);
        }
        const out = new Float32Array(session.vocabSize + 1);
        out.fill(-333);
        const logits = session.step(0, out);
        if (logits.length !== inspection.vocabSize || session.position() !== 1) {
          throw new Error(`unexpected auto LLaMA step: logits=${logits.length} pos=${session.position()}`);
        }
        if (out[inspection.vocabSize] !== -333) throw new Error("auto LLaMA step wrote past vocab logits");
        session.reset();
        if (session.position() !== 0) {
          throw new Error(`expected reset auto LLaMA position 0, got ${session.position()}`);
        }
        console.log(`zgml bun ffi auto llama smoke ok: ${logits.length} logits`);
      } finally {
        session.free();
      }
    } finally {
      program.free();
    }
  } finally {
    auto.free();
  }

  const smollm = loadModel(smollmPath, { kind: "smollm-135m" });
  try {
    const modelInspection = smollm.inspect();
    if (
      modelInspection.modelKind !== "smollm-135m" ||
      modelInspection.vocabSize !== 49152 ||
      modelInspection.maxSeqLen !== 2048 ||
      modelInspection.dModel !== 576 ||
      modelInspection.nLayers !== 30 ||
      modelInspection.nHeads !== 9 ||
      modelInspection.nKvHeads !== 3 ||
      modelInspection.dFF !== 1536 ||
      Math.abs(modelInspection.rmsNormEps - 1e-5) > 1e-12 ||
      modelInspection.tiedLmHead !== true
    ) {
      throw new Error(`unexpected SmolLM model inspection: ${json(modelInspection)}`);
    }

    const program = smollm.compile();
    try {
      const inspection = program.inspect();
      if (
        inspection.vocabSize !== 49152 ||
        inspection.maxSeqLen !== 2048 ||
        inspection.nLayers !== 30 ||
        inspection.nHeads !== 9 ||
        inspection.nKvHeads !== 3 ||
        inspection.semanticTokenCount !== 1 ||
        inspection.semanticStageCount === 0 ||
        inspection.semanticRuntimePatchHoles === 0
      ) {
        throw new Error(`unexpected SmolLM program inspection: ${JSON.stringify(inspection)}`);
      }
      if (program.vocabSize !== inspection.vocabSize) {
        throw new Error(`expected SmolLM program vocab ${inspection.vocabSize}, got ${program.vocabSize}`);
      }
      const executable = program.inspectExecutable();
      if (
        executable.commandCount === 0 ||
        executable.commandStencilHash === 0n ||
        executable.runtimePatchMaxCacheWritePos !== 2047 ||
        executable.runtimePatchMaxAttentionSeqKv !== 2048 ||
        executable.runtimePatchStencilHash === 0n ||
        executable.runtimePatchHoles !== inspection.semanticRuntimePatchHoles ||
        executable.runtimePatchCacheWritePosHoles !== inspection.semanticRuntimePatchCacheWritePosHoles ||
        executable.runtimePatchAttentionSeqKvHoles !== inspection.semanticRuntimePatchAttentionSeqKvHoles
      ) {
        throw new Error(`unexpected SmolLM executable inspection: ${json(executable)}`);
      }

      const session = program.bind();
      const prefillSession = program.bind();
      try {
        if (session.vocabSize !== inspection.vocabSize || prefillSession.vocabSize !== inspection.vocabSize) {
          throw new Error(`unexpected SmolLM session vocab sizes: step=${session.vocabSize} prefill=${prefillSession.vocabSize}`);
        }
        const out = new Float32Array(SmolLM135MVocabPlusSentinel);
        out.fill(-555);
        const logits = session.step(0, out);
        if (logits.length !== 49152) throw new Error(`expected 49152 SmolLM logits, got ${logits.length}`);
        if (session.position() !== 1) throw new Error(`expected SmolLM position 1, got ${session.position()}`);
        if (out[49152] !== -555) throw new Error("SmolLM step wrote past vocab logits");

        const prefillOut = new Float32Array(SmolLM135MVocabPlusSentinel);
        prefillOut.fill(-444);
        const prefillLogits = prefillSession.prefill([0, 1], prefillOut);
        if (prefillLogits.length !== 49152) throw new Error(`expected 49152 SmolLM prefill logits, got ${prefillLogits.length}`);
        if (prefillSession.position() !== 2) {
          throw new Error(`expected SmolLM prefill position 2, got ${prefillSession.position()}`);
        }
        if (prefillOut[49152] !== -444) throw new Error("SmolLM prefill wrote past vocab logits");
        console.log(`zgml bun ffi SmolLM smoke ok: ${logits.length} logits`);
      } finally {
        prefillSession.free();
        session.free();
      }
    } finally {
      program.free();
    }
  } finally {
    smollm.free();
  }
}
