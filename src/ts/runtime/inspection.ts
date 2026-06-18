"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  backendName,
  bufferStorageIds,
  bufferStorageName,
  accessFlagsToObject,
  modelKindName,
} from "./abi.js";
import {
  acceptsRuntimeProfile,
  assert_runtime_profile,
  assertRuntimeProfile,
  matchesRuntimeProfileSignature,
  requireNoFallbackRuntimeProfile,
  requireNoSyncRuntimeProfile,
  requireHotRuntimeProfile,
  requireRuntimePatchValidProfile,
  requireRuntimeProfile,
  runtimeProfileExpectation,
  runtimeProfileHasNoFallback,
  runtimeProfileHasNoInvalidRuntimePatches,
  runtimeProfileHasNoSync,
} from "./runtime_profile_expectation.js";
import type {
  ProgramExecutionCapabilities,
  ProgramExecutionMode,
  ProgramInspection,
  ProgramRequirements,
  ProgramRuntimeDiagnostic,
} from "../public_api.js";

type AnyRecord = Record<string, any>;
type AbiWords = BigUint64Array;
type Field = Readonly<{ name: string; record: string; word: number; kind: string }>;

export const inspectionManifest = Object.freeze({
  kind: "zgml-inspection",
  ...tsRuntimeManifestPolicy("src/ts/runtime/inspection.ts", "Program -> Session -> inspection evidence"),
});

const runtimeProfileFields = Object.freeze([
  Object.freeze({ name: "callCount", record: "call_count", word: 0, kind: "number" }),
  Object.freeze({ name: "backendOpCount", record: "backend_op_count", word: 1, kind: "number" }),
  Object.freeze({ name: "fallbackOpCount", record: "fallback_op_count", word: 2, kind: "number" }),
  Object.freeze({ name: "backendDispatchCount", record: "backend_dispatch_count", word: 3, kind: "number" }),
  Object.freeze({ name: "syncCount", record: "sync_count", word: 4, kind: "number" }),
  Object.freeze({ name: "runtimePatchCallCount", record: "runtime_patch_call_count", word: 5, kind: "number" }),
  Object.freeze({ name: "runtimePatchChangedCount", record: "runtime_patch_changed_count", word: 6, kind: "number" }),
  Object.freeze({ name: "runtimePatchInvalidCount", record: "runtime_patch_invalid_count", word: 7, kind: "number" }),
  Object.freeze({ name: "runtimePatchHoles", record: "runtime_patch_holes", word: 8, kind: "number" }),
  Object.freeze({ name: "runtimePatchCacheWritePosHoles", record: "runtime_patch_cache_write_pos_holes", word: 9, kind: "number" }),
  Object.freeze({ name: "runtimePatchAttentionSeqKvHoles", record: "runtime_patch_attention_seq_kv_holes", word: 10, kind: "number" }),
  Object.freeze({ name: "runtimePatchStencilHash", record: "runtime_patch_stencil_hash", word: 11, kind: "bigint" }),
  Object.freeze({ name: "commandCount", record: "command_count", word: 12, kind: "number" }),
  Object.freeze({ name: "commandStencilHash", record: "command_stencil_hash", word: 13, kind: "bigint" }),
  Object.freeze({ name: "commandOpCount", record: "command_op_count", word: 14, kind: "number" }),
  Object.freeze({ name: "commandRowCount", record: "command_row_count", word: 15, kind: "number" }),
  Object.freeze({ name: "commandProjectionCount", record: "command_projection_count", word: 16, kind: "number" }),
  Object.freeze({ name: "commandAttentionCount", record: "command_attention_count", word: 17, kind: "number" }),
  Object.freeze({ name: "commandMovementCount", record: "command_movement_count", word: 18, kind: "number" }),
  Object.freeze({ name: "commandElementwiseCount", record: "command_elementwise_count", word: 19, kind: "number" }),
  Object.freeze({ name: "commandRopeCount", record: "command_rope_count", word: 20, kind: "number" }),
]) satisfies readonly Field[];

function runtimeProfileFromReader(read: (field: Field) => unknown) {
  const profile: AnyRecord = {};
  profile.kind = "zgml.runtime.profile";
  for (const field of runtimeProfileFields) {
    const value = read(field);
    profile[field.name] = field.kind === "bigint" ? BigInt(value as any) : Number(value);
  }
  profile.signature = [
    "runtime-profile",
    ...runtimeProfileFields.map((field) => `${field.name}=${String(profile[field.name])}`),
  ].join("|");
  return Object.freeze(profile);
}

export function runtimeProfileFromAbiRecord(record: AnyRecord) {
  return runtimeProfileFromReader((field) => record[field.record]);
}

export function runtimeProfileFromAbiWords(words: AbiWords) {
  return runtimeProfileFromReader((field) => words[field.word]);
}

export {
  acceptsRuntimeProfile,
  assert_runtime_profile,
  assertRuntimeProfile,
  matchesRuntimeProfileSignature,
  requireNoFallbackRuntimeProfile,
  requireNoSyncRuntimeProfile,
  requireHotRuntimeProfile,
  requireRuntimePatchValidProfile,
  requireRuntimeProfile,
  runtimeProfileExpectation,
  runtimeProfileHasNoFallback,
  runtimeProfileHasNoInvalidRuntimePatches,
  runtimeProfileHasNoSync,
};

const optionalU64Sentinel = 0xffffffffffffffffn;

function evidenceSignature(prefix: string, source: AnyRecord, fields: readonly string[]) {
  return [
    prefix,
    ...fields.map((field) => `${field}=${String(source[field] ?? "null")}`),
  ].join("|");
}

function optionalU64InspectionValue(value: unknown) {
  const normalized = BigInt(value as any);
  return normalized === optionalU64Sentinel ? null : Number(normalized);
}

const programInspectionFields = Object.freeze([
  Object.freeze({ name: "backend", record: "backend", word: 0, kind: "backend" }),
  Object.freeze({ name: "executionSupported", record: "execution_supported", word: 1, kind: "bool" }),
  Object.freeze({ name: "externalResourcesSupported", record: "external_resources_supported", word: 2, kind: "bool" }),
  Object.freeze({ name: "bufferCount", record: "buffer_count", word: 3, kind: "number" }),
  Object.freeze({ name: "bufferElementCount", record: "buffer_element_count", word: 21, kind: "number" }),
  Object.freeze({ name: "bufferByteLength", record: "buffer_byte_len", word: 4, kind: "number" }),
  Object.freeze({ name: "initialUploadCount", record: "initial_upload_count", word: 5, kind: "number" }),
  Object.freeze({ name: "qweightCount", record: "qweight_count", word: 6, kind: "number" }),
  Object.freeze({ name: "opCount", record: "op_count", word: 20, kind: "number" }),
  Object.freeze({ name: "commandCount", record: "command_count", word: 7, kind: "number" }),
  Object.freeze({ name: "commandStencilHash", record: "command_stencil_hash", word: 8, kind: "bigint" }),
  Object.freeze({ name: "runtimePatchHoles", record: "runtime_patch_holes", word: 9, kind: "number" }),
  Object.freeze({ name: "runtimePatchCacheWritePosHoles", record: "runtime_patch_cache_write_pos_holes", word: 10, kind: "number" }),
  Object.freeze({ name: "runtimePatchAttentionSeqKvHoles", record: "runtime_patch_attention_seq_kv_holes", word: 11, kind: "number" }),
  Object.freeze({ name: "runtimePatchMaxCacheWritePos", record: "runtime_patch_max_cache_write_pos", word: 22, kind: "optionalU64" }),
  Object.freeze({ name: "runtimePatchMaxAttentionSeqKv", record: "runtime_patch_max_attention_seq_kv", word: 23, kind: "optionalU64" }),
  Object.freeze({ name: "runtimePatchStencilHash", record: "runtime_patch_stencil_hash", word: 12, kind: "bigint" }),
  Object.freeze({ name: "commandOpCount", record: "command_op_count", word: 13, kind: "number" }),
  Object.freeze({ name: "commandRowCount", record: "command_row_count", word: 14, kind: "number" }),
  Object.freeze({ name: "commandProjectionCount", record: "command_projection_count", word: 15, kind: "number" }),
  Object.freeze({ name: "commandAttentionCount", record: "command_attention_count", word: 16, kind: "number" }),
  Object.freeze({ name: "commandMovementCount", record: "command_movement_count", word: 17, kind: "number" }),
  Object.freeze({ name: "commandElementwiseCount", record: "command_elementwise_count", word: 18, kind: "number" }),
  Object.freeze({ name: "commandRopeCount", record: "command_rope_count", word: 19, kind: "number" }),
  Object.freeze({ name: "backendDispatchCount", record: "backend_dispatch_count", word: 24, kind: "number" }),
  Object.freeze({ name: "dispatchPlanSupported", record: "dispatch_plan_supported", word: 25, kind: "bool" }),
  Object.freeze({ name: "dispatchPlanCoveredOpCount", record: "dispatch_plan_covered_op_count", word: 26, kind: "number" }),
  Object.freeze({ name: "dispatchPlanFirstUnsupportedOp", record: "dispatch_plan_first_unsupported_op", word: 27, kind: "optionalU64" }),
  Object.freeze({ name: "dispatchPlanProjectionCount", record: "dispatch_plan_projection_count", word: 28, kind: "number" }),
  Object.freeze({ name: "dispatchPlanRowCount", record: "dispatch_plan_row_count", word: 29, kind: "number" }),
  Object.freeze({ name: "dispatchPlanAttentionCount", record: "dispatch_plan_attention_count", word: 30, kind: "number" }),
  Object.freeze({ name: "dispatchPlanMovementCount", record: "dispatch_plan_movement_count", word: 31, kind: "number" }),
  Object.freeze({ name: "dispatchPlanElementwiseCount", record: "dispatch_plan_elementwise_count", word: 32, kind: "number" }),
  Object.freeze({ name: "dispatchPlanRopeCount", record: "dispatch_plan_rope_count", word: 33, kind: "number" }),
  Object.freeze({ name: "dispatchPlanQuantizedProjectionCount", record: "dispatch_plan_quantized_projection_count", word: 34, kind: "number" }),
  Object.freeze({ name: "bindingRequirementHash", record: "binding_requirement_hash", word: 35, kind: "bigint" }),
  Object.freeze({ name: "persistentRequirementCount", record: "persistent_requirement_count", word: 36, kind: "number" }),
  Object.freeze({ name: "stepInputRequirementCount", record: "step_input_requirement_count", word: 37, kind: "number" }),
  Object.freeze({ name: "stepOutputRequirementCount", record: "step_output_requirement_count", word: 38, kind: "number" }),
]) satisfies readonly Field[];

function programInspectionFromReader(read: (field: Field) => unknown) {
  const inspection: AnyRecord = {};
  for (const field of programInspectionFields) {
    const value = read(field);
    if (field.kind === "backend") {
      inspection[field.name] = backendName(value);
    } else if (field.kind === "bool") {
      inspection[field.name] = Number(value) !== 0;
    } else if (field.kind === "bigint") {
      inspection[field.name] = BigInt(value as any);
    } else if (field.kind === "optionalU64") {
      inspection[field.name] = optionalU64InspectionValue(value);
    } else {
      inspection[field.name] = Number(value);
    }
  }
  inspection.signature = evidenceSignature("program-inspection", inspection, programInspectionFields.map((field) => field.name));
  return Object.freeze({
    ...inspection,
    kind: "zgml.program.inspection",
    signature: inspection.signature,
    diagnostics: programRuntimeDiagnostics(inspection, programExecutionMode(inspection)),
  });
}

export function programInspectionFromAbiRecord(record: AnyRecord) {
  return programInspectionFromReader((field) => record[field.record]);
}

export function programInspectionFromAbiWords(words: AbiWords) {
  return programInspectionFromReader((field) => words[field.word]);
}

export function sessionInspectionFromAbiRecord(record: AnyRecord) {
  const inspection = {
    kind: "zgml.session.inspection",
    modelKind: modelKindName(record.model_kind),
    backend: backendName(record.backend),
    outputStorage: bufferStorageName(record.output_storage),
    kvCacheStorage: bufferStorageName(record.kv_cache_storage),
    position: Number(record.position),
    contextLen: Number(record.context_len),
    persistentBindingCount: Number(record.persistent_binding_count),
    stepInputCount: Number(record.step_input_count),
    stepOutputCount: Number(record.step_output_count),
    hostBindingCount: Number(record.host_binding_count),
    resourceBindingCount: Number(record.resource_binding_count),
    bindingShapeHash: BigInt(record.binding_shape_hash),
  };
  return Object.freeze({
    ...inspection,
    signature: evidenceSignature("session-inspection", inspection, [
      "modelKind", "backend", "outputStorage", "kvCacheStorage", "position", "contextLen",
      "persistentBindingCount", "stepInputCount", "stepOutputCount", "hostBindingCount", "resourceBindingCount",
      "bindingShapeHash",
    ]),
  });
}

export function sessionInspectionFromAbiWords(words: AbiWords) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const inspection = {
    kind: "zgml.session.inspection",
    modelKind: modelKindName(view.getUint32(0, true)),
    backend: backendName(BigInt(view.getUint32(4, true))),
    outputStorage: bufferStorageName(view.getUint32(8, true)),
    kvCacheStorage: bufferStorageName(view.getUint32(12, true)),
    position: Number(words[2]),
    contextLen: Number(words[3]),
    persistentBindingCount: Number(words[4]),
    stepInputCount: Number(words[5]),
    stepOutputCount: Number(words[6]),
    hostBindingCount: Number(words[7]),
    resourceBindingCount: Number(words[8]),
    bindingShapeHash: BigInt(words[9]),
  };
  return Object.freeze({
    ...inspection,
    signature: evidenceSignature("session-inspection", inspection, [
      "modelKind", "backend", "outputStorage", "kvCacheStorage", "position", "contextLen",
      "persistentBindingCount", "stepInputCount", "stepOutputCount", "hostBindingCount", "resourceBindingCount",
      "bindingShapeHash",
    ]),
  });
}

function bufferInspectionFromFields(storageId: unknown, placementId: unknown, accessFlags: unknown, byteLength: unknown, handle: unknown, byteOffset: unknown, resourceByteLength: unknown) {
  const external = Number(storageId) === bufferStorageIds.externalResource;
  const flags = Number(accessFlags);
  const inspection = {
    kind: "zgml.buffer.inspection",
    storage: bufferStorageName(storageId),
    byteLength: Number(byteLength),
    placement: external ? backendName(placementId) : null,
    accessFlags: flags,
    access: external ? accessFlagsToObject(flags) : null,
    handle: Number(handle),
    byteOffset: Number(byteOffset),
    resourceByteLength: Number(resourceByteLength),
  };
  return Object.freeze({
    ...inspection,
    signature: evidenceSignature("buffer-inspection", inspection, [
      "storage", "byteLength", "placement", "accessFlags", "handle", "byteOffset", "resourceByteLength",
    ]),
  });
}

export function bufferInspectionFromAbiRecord(record: AnyRecord) {
  return bufferInspectionFromFields(
    record.storage,
    record.placement,
    record.access_flags,
    record.byte_len,
    record.handle,
    record.byte_offset,
    record.resource_byte_len,
  );
}

export function bufferInspectionFromAbiWords(words: AbiWords) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  return bufferInspectionFromFields(
    view.getUint32(0, true),
    BigInt(view.getUint32(4, true)),
    view.getUint32(8, true),
    words[2],
    words[3],
    words[4],
    words[5],
  );
}

export function modelInspectionFromAbiRecord(record: AnyRecord) {
  const inspection = {
    kind: "zgml.model.inspection",
    modelKind: modelKindName(record.model_kind),
    inputLen: Number(record.input_len),
    outputLen: Number(record.output_len),
    vocabSize: Number(record.vocab_size),
    maxSeqLen: Number(record.max_seq_len),
    dModel: Number(record.d_model),
    nLayers: Number(record.n_layers),
    nHeads: Number(record.n_heads),
    nKvHeads: Number(record.n_kv_heads),
    dFF: Number(record.d_ff),
    ropeBase: Number(record.rope_base),
    rmsNormEps: Number(record.rms_norm_eps),
    tiedLmHead: Number(record.tied_lm_head) !== 0,
  };
  return Object.freeze({
    ...inspection,
    signature: evidenceSignature("model-inspection", inspection, [
      "modelKind", "inputLen", "outputLen", "vocabSize", "maxSeqLen", "dModel", "nLayers",
      "nHeads", "nKvHeads", "dFF", "ropeBase", "rmsNormEps", "tiedLmHead",
    ]),
  });
}

export function modelInspectionFromAbiWords(words: AbiWords) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const inspection = {
    kind: "zgml.model.inspection",
    modelKind: modelKindName(view.getUint32(0, true)),
    inputLen: Number(words[1]),
    outputLen: Number(words[2]),
    vocabSize: Number(words[3]),
    maxSeqLen: Number(words[4]),
    dModel: Number(words[5]),
    nLayers: Number(words[6]),
    nHeads: Number(words[7]),
    nKvHeads: Number(words[8]),
    dFF: Number(words[9]),
    ropeBase: view.getFloat64(80, true),
    rmsNormEps: view.getFloat64(88, true),
    tiedLmHead: words[12] !== 0n,
  };
  return Object.freeze({
    ...inspection,
    signature: evidenceSignature("model-inspection", inspection, [
      "modelKind", "inputLen", "outputLen", "vocabSize", "maxSeqLen", "dModel", "nLayers",
      "nHeads", "nKvHeads", "dFF", "ropeBase", "rmsNormEps", "tiedLmHead",
    ]),
  });
}

export function programRequirementsFromAbiRecord(record: AnyRecord) {
  const requirements = {
    kind: "zgml.program.requirements" as const,
    modelKind: modelKindName(record.model_kind),
    scalarBytes: Number(record.scalar_bytes),
    tokenIdBytes: Number(record.token_id_bytes),
    inputLen: Number(record.input_len),
    inputByteLength: Number(record.input_byte_len),
    outputLen: Number(record.output_len),
    weightsLen: Number(record.weights_len),
    weightsByteLength: Number(record.weights_byte_len),
    biasLen: Number(record.bias_len),
    biasByteLength: Number(record.bias_byte_len),
    parameterLen: Number(record.parameter_len),
    parameterByteLength: Number(record.parameter_byte_len),
    logitsLen: Number(record.logits_len),
    outputByteLength: Number(record.output_byte_len),
    contextLen: Number(record.context_len),
    batch: Number(record.batch),
    maxTokenWindow: Number(record.max_token_window),
  };
  return Object.freeze({
    ...requirements,
    signature: evidenceSignature("program-requirements", requirements, [
      "modelKind", "scalarBytes", "tokenIdBytes", "inputLen", "inputByteLength", "outputLen",
      "weightsLen", "weightsByteLength", "biasLen", "biasByteLength", "parameterLen",
      "parameterByteLength", "logitsLen", "outputByteLength", "contextLen", "batch", "maxTokenWindow",
    ]),
  });
}

export function programRequirementsFromAbiWords(words: AbiWords): ProgramRequirements {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const requirements = {
    kind: "zgml.program.requirements" as const,
    modelKind: modelKindName(view.getUint32(0, true)) as ProgramRequirements["modelKind"],
    scalarBytes: view.getUint32(4, true),
    tokenIdBytes: view.getUint32(8, true),
    inputLen: Number(words[2]),
    inputByteLength: Number(words[3]),
    outputLen: Number(words[4]),
    weightsLen: Number(words[5]),
    weightsByteLength: Number(words[6]),
    biasLen: Number(words[7]),
    biasByteLength: Number(words[8]),
    parameterLen: Number(words[9]),
    parameterByteLength: Number(words[10]),
    logitsLen: Number(words[11]),
    outputByteLength: Number(words[12]),
    contextLen: Number(words[13]),
    batch: Number(words[14]),
    maxTokenWindow: Number(words[15]),
  };
  return Object.freeze({
    ...requirements,
    signature: evidenceSignature("program-requirements", requirements, [
      "modelKind", "scalarBytes", "tokenIdBytes", "inputLen", "inputByteLength", "outputLen",
      "weightsLen", "weightsByteLength", "biasLen", "biasByteLength", "parameterLen",
      "parameterByteLength", "logitsLen", "outputByteLength", "contextLen", "batch", "maxTokenWindow",
    ]),
  });
}

export function llamaKvCacheRequirementsFromAbiRecord(record: AnyRecord) {
  const requirements = {
    kind: "zgml.llama.kv-cache.requirements",
    modelKind: modelKindName(record.model_kind),
    scalarBytes: Number(record.scalar_bytes),
    layers: Number(record.n_layers),
    kBufferByteLength: Number(record.k_buffer_byte_len),
    vBufferByteLength: Number(record.v_buffer_byte_len),
    bufferByteLength: Number(record.buffer_byte_len),
    contextLength: Number(record.context_len),
  };
  return Object.freeze({
    ...requirements,
    signature: evidenceSignature("llama-kv-cache-requirements", requirements, [
      "modelKind", "scalarBytes", "layers", "kBufferByteLength", "vBufferByteLength", "bufferByteLength", "contextLength",
    ]),
  });
}

export function llamaKvCacheRequirementsFromAbiWords(words: AbiWords) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const requirements = {
    kind: "zgml.llama.kv-cache.requirements",
    modelKind: modelKindName(view.getUint32(0, true)),
    scalarBytes: view.getUint32(4, true),
    layers: view.getUint32(8, true),
    contextLength: Number(words[2]),
    kBufferByteLength: Number(words[3]),
    vBufferByteLength: Number(words[4]),
    bufferByteLength: Number(words[5]),
  };
  return Object.freeze({
    ...requirements,
    signature: evidenceSignature("llama-kv-cache-requirements", requirements, [
      "modelKind", "scalarBytes", "layers", "kBufferByteLength", "vBufferByteLength", "bufferByteLength", "contextLength",
    ]),
  });
}

export function programModelCompatibilityFromAbiRecord(record: AnyRecord) {
  const programModelKind = modelKindName(record.program_model_kind);
  const modelKind = modelKindName(record.model_kind);
  const compatible = Number(record.compatible) !== 0;
  return Object.freeze({
    kind: "zgml.program.model-compatibility",
    signature: `program-model-compatibility|program=${programModelKind}|model=${modelKind}|compatible=${compatible ? 1 : 0}`,
    programModelKind,
    modelKind,
    compatible,
  });
}

export function programModelCompatibilityFromAbiWords(words: AbiWords) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const programModelKind = modelKindName(view.getUint32(0, true));
  const modelKind = modelKindName(view.getUint32(4, true));
  const compatible = words[1] !== 0n;
  return Object.freeze({
    kind: "zgml.program.model-compatibility",
    signature: `program-model-compatibility|program=${programModelKind}|model=${modelKind}|compatible=${compatible ? 1 : 0}`,
    programModelKind,
    modelKind,
    compatible,
  });
}

function freezeProgramRuntimeDiagnostics(diagnostics: readonly ProgramRuntimeDiagnostic[]): readonly ProgramRuntimeDiagnostic[] {
  return Object.freeze(diagnostics.map((diagnostic) => Object.freeze(diagnostic)));
}

export function programRuntimeDiagnostics(inspection: AnyRecord, mode: ProgramExecutionMode): readonly ProgramRuntimeDiagnostic[] {
  const diagnostics: ProgramRuntimeDiagnostic[] = [];
  if (!inspection.executionSupported) {
    diagnostics.push({
      code: "execution-unavailable",
      message: inspection.externalResourcesSupported
        ? "Program can bind external resources for inspection but cannot execute on this backend"
        : "Program was compiled for inspection only and cannot execute on this backend",
      backend: inspection.backend,
      mode,
    });
  }
  if (!inspection.dispatchPlanSupported) {
    diagnostics.push({
      code: "dispatch-plan-unavailable",
      message: "Program backend did not provide a dispatch plan",
      backend: inspection.backend,
      opCount: inspection.opCount,
      coveredOpCount: inspection.dispatchPlanCoveredOpCount,
      firstUnsupportedOp: inspection.dispatchPlanFirstUnsupportedOp,
    });
  } else if (inspection.dispatchPlanFirstUnsupportedOp !== null) {
    diagnostics.push({
      code: "dispatch-plan-unsupported-op",
      message: "Program dispatch plan does not cover every op",
      backend: inspection.backend,
      opCount: inspection.opCount,
      coveredOpCount: inspection.dispatchPlanCoveredOpCount,
      firstUnsupportedOp: inspection.dispatchPlanFirstUnsupportedOp,
    });
  } else if (inspection.dispatchPlanCoveredOpCount !== inspection.opCount) {
    diagnostics.push({
      code: "dispatch-plan-incomplete",
      message: "Program dispatch plan covered op count does not match op count",
      backend: inspection.backend,
      opCount: inspection.opCount,
      coveredOpCount: inspection.dispatchPlanCoveredOpCount,
      firstUnsupportedOp: inspection.dispatchPlanFirstUnsupportedOp,
    });
  }
  return freezeProgramRuntimeDiagnostics(diagnostics);
}

export function programExecutionMode(inspection: AnyRecord): ProgramExecutionMode {
  return inspection.executionSupported
    ? "executable"
    : (inspection.externalResourcesSupported ? "resource-probe" : "compile-only");
}

export function programExecutionCapabilitiesFromInspection(inspection: AnyRecord): ProgramExecutionCapabilities {
  const mode = programExecutionMode(inspection);
  const hasFullDispatchPlan =
    inspection.dispatchPlanSupported &&
    inspection.dispatchPlanCoveredOpCount === inspection.opCount &&
    inspection.dispatchPlanFirstUnsupportedOp === null;
  const signature = [
    "program-capabilities",
    `backend=${inspection.backend}`,
    `mode=${mode}`,
    `canExecute=${inspection.executionSupported ? 1 : 0}`,
    `externalResources=${inspection.externalResourcesSupported ? 1 : 0}`,
    `fullDispatchPlan=${hasFullDispatchPlan ? 1 : 0}`,
    `backendDispatches=${inspection.backendDispatchCount}`,
    `dispatchPlan=${inspection.dispatchPlanSupported ? 1 : 0}`,
    `coveredOps=${inspection.dispatchPlanCoveredOpCount}`,
    `opCount=${inspection.opCount}`,
    `firstUnsupportedOp=${inspection.dispatchPlanFirstUnsupportedOp === null ? "none" : inspection.dispatchPlanFirstUnsupportedOp}`,
    `projection=${inspection.dispatchPlanProjectionCount}`,
    `row=${inspection.dispatchPlanRowCount}`,
    `attention=${inspection.dispatchPlanAttentionCount}`,
    `movement=${inspection.dispatchPlanMovementCount}`,
    `elementwise=${inspection.dispatchPlanElementwiseCount}`,
    `rope=${inspection.dispatchPlanRopeCount}`,
    `qprojection=${inspection.dispatchPlanQuantizedProjectionCount}`,
  ].join("|");
  return Object.freeze({
    kind: "zgml.program.capabilities",
    backend: inspection.backend as ProgramInspection["backend"],
    mode,
    signature,
    canExecute: inspection.executionSupported,
    canBindExternalResources: inspection.externalResourcesSupported,
    hasFullDispatchPlan,
    backendDispatchCount: inspection.backendDispatchCount,
    dispatchPlanSupported: inspection.dispatchPlanSupported,
    dispatchPlanCoveredOpCount: inspection.dispatchPlanCoveredOpCount,
    dispatchPlanFirstUnsupportedOp: inspection.dispatchPlanFirstUnsupportedOp,
    dispatchPlanProjectionCount: inspection.dispatchPlanProjectionCount,
    dispatchPlanRowCount: inspection.dispatchPlanRowCount,
    dispatchPlanAttentionCount: inspection.dispatchPlanAttentionCount,
    dispatchPlanMovementCount: inspection.dispatchPlanMovementCount,
    dispatchPlanElementwiseCount: inspection.dispatchPlanElementwiseCount,
    dispatchPlanRopeCount: inspection.dispatchPlanRopeCount,
    dispatchPlanQuantizedProjectionCount: inspection.dispatchPlanQuantizedProjectionCount,
    diagnostics: inspection.diagnostics ?? programRuntimeDiagnostics(inspection, mode),
  });
}

export function programCompatibilityAccepted(compatibility: AnyRecord | null | undefined): boolean {
  return !!(compatibility && compatibility.compatible);
}

export function programCapabilityCanExecute(capabilities: AnyRecord | null | undefined): boolean {
  return !!(capabilities && capabilities.canExecute);
}

export function programCapabilityCanBindExternalResources(capabilities: AnyRecord | null | undefined): boolean {
  return !!(capabilities && capabilities.canBindExternalResources);
}

export function programCapabilityHasFullDispatchPlan(capabilities: AnyRecord | null | undefined): boolean {
  return !!(capabilities && capabilities.hasFullDispatchPlan);
}

export function programExecutionModeFromCapabilities(capabilities: AnyRecord | null | undefined): string {
  return capabilities ? capabilities.mode : "compile-only";
}
