import {
  setUSize,
} from "./bun_abi_words.js";
import {
  llamaProgramInspectionFromWords,
} from "../runtime/llama_program_inspection.js";
import {
  compileDescFieldsFromOptions,
  isDefaultCompileEnvelope,
  type CompileDescFields,
  type NormalizedCompileOptions,
} from "../runtime/abi.js";
import type {
  ProgramExecutionCapabilities,
  ProgramModelCompatibility,
  ProgramRequirements,
  RuntimeProfile,
} from "../public_api.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;
type EvidenceRecord = Readonly<Record<string, unknown>>;
type ProgramExecutionCapabilitiesEvidence = ProgramExecutionCapabilities;
type ProgramModelCompatibilityEvidence = ProgramModelCompatibility | EvidenceRecord;
type ProgramRequirementsEvidence = ProgramRequirements;
type RuntimeProfileEvidence = RuntimeProfile | EvidenceRecord;

type BunInspectionSymbols = Readonly<{
  modelInspect(handle: NativeHandle, out: BigUint64Array): number;
  programCompile(handle: NativeHandle, desc: BigUint64Array | NativeHandle, out: NativeOut): number;
  bufferInspect(handle: NativeHandle, out: BigUint64Array): number;
  sessionInspect(handle: NativeHandle, out: BigUint64Array): number;
  sessionPosition(handle: NativeHandle, out: BigUint64Array): number;
  sessionReset(handle: NativeHandle): number;
  sessionRuntimeProfile(handle: NativeHandle, out: BigUint64Array): number;
  sessionResetRuntimeProfile(handle: NativeHandle): number;
  programInspect(handle: NativeHandle, out: BigUint64Array): number;
  programGetRequirements(handle: NativeHandle, out: BigUint64Array): number;
  programCheckModelCompatibility(handle: NativeHandle, model: NativeHandle, out: BigUint64Array): number;
  llamaProgramInspect(handle: NativeHandle, out: BigUint64Array): number;
  programRuntimeProfile(handle: NativeHandle, out: BigUint64Array): number;
}>;

export type BunInspectionOpsOptions = Readonly<{
  symbols: BunInspectionSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  normalizeCompileOptions(options: unknown): NormalizedCompileOptions;
  modelHandleForCompatibility(model: unknown): NativeHandle;
  modelInspectionFromAbiWords(out: BigUint64Array): unknown;
  bufferInspectionFromAbiWords(out: BigUint64Array): unknown;
  sessionInspectionFromAbiWords(out: BigUint64Array): unknown;
  programInspectionFromAbiWords(out: BigUint64Array): unknown;
  programExecutionCapabilitiesFromInspection(inspection: unknown): ProgramExecutionCapabilitiesEvidence;
  programRequirementsFromAbiWords(out: BigUint64Array): ProgramRequirementsEvidence;
  programModelCompatibilityFromAbiWords(out: BigUint64Array): ProgramModelCompatibilityEvidence;
  runtimeProfileFromAbiWords(out: BigUint64Array): RuntimeProfileEvidence;
}>;

export function createBunInspectionOps(options: BunInspectionOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    normalizeCompileOptions,
    modelHandleForCompatibility,
    modelInspectionFromAbiWords,
    bufferInspectionFromAbiWords,
    sessionInspectionFromAbiWords,
    programInspectionFromAbiWords,
    programExecutionCapabilitiesFromInspection,
    programRequirementsFromAbiWords,
    programModelCompatibilityFromAbiWords,
    runtimeProfileFromAbiWords,
  } = options;

  function compileDescWords(fields: CompileDescFields): BigUint64Array {
    const buf = new BigUint64Array(3);
    const view = new DataView(buf.buffer);
    view.setUint32(0, fields.backend, true);
    view.setUint32(4, fields.reserved, true);
    setUSize(view, 8, fields.contextLen);
    setUSize(view, 16, fields.batch);
    return buf;
  }

  function compileDesc(compileOptions: unknown = {}): BigUint64Array | NativeHandle {
    const normalized = normalizeCompileOptions(compileOptions);
    if (isDefaultCompileEnvelope(normalized)) return 0;
    const fields = compileDescFieldsFromOptions(normalized);
    return compileDescWords(fields);
  }

  function inspectModelHandle(handle: NativeHandle): unknown {
    const out = new BigUint64Array(13);
    check(symbols.modelInspect(handle, out));
    return modelInspectionFromAbiWords(out);
  }

  function compileModelProgramHandle(handle: NativeHandle, compileOptions: unknown = {}): NativeHandle {
    const out = handleOut();
    check(symbols.programCompile(handle, compileDesc(compileOptions), out));
    return readHandle(out);
  }

  function inspectBufferHandle(handle: NativeHandle): unknown {
    const out = new BigUint64Array(6);
    check(symbols.bufferInspect(handle, out));
    return bufferInspectionFromAbiWords(out);
  }

  function inspectSessionHandle(handle: NativeHandle): unknown {
    const out = new BigUint64Array(10);
    check(symbols.sessionInspect(handle, out));
    return sessionInspectionFromAbiWords(out);
  }

  function llamaSessionPosition(handle: NativeHandle): number {
    const out = new BigUint64Array(1);
    check(symbols.sessionPosition(handle, out));
    return Number(out[0]);
  }

  function resetSessionHandle(handle: NativeHandle): void {
    check(symbols.sessionReset(handle));
  }

  function resetSessionRuntimeProfileHandle(handle: NativeHandle): void {
    check(symbols.sessionResetRuntimeProfile(handle));
  }

  function inspectExecutableProgram(handle: NativeHandle): unknown {
    const out = new BigUint64Array(39);
    check(symbols.programInspect(handle, out));
    return programInspectionFromAbiWords(out);
  }

  function programExecutionCapabilities(handle: NativeHandle): ProgramExecutionCapabilitiesEvidence {
    return programExecutionCapabilitiesFromInspection(inspectExecutableProgram(handle));
  }

  function programRequirements(handle: NativeHandle): ProgramRequirementsEvidence {
    const out = new BigUint64Array(16);
    check(symbols.programGetRequirements(handle, out));
    return programRequirementsFromAbiWords(out);
  }

  function programModelCompatibility(handle: NativeHandle, model: unknown): ProgramModelCompatibilityEvidence {
    const out = new BigUint64Array(2);
    check(symbols.programCheckModelCompatibility(handle, modelHandleForCompatibility(model), out));
    return programModelCompatibilityFromAbiWords(out);
  }

  function inspectLlamaProgram(handle: NativeHandle) {
    const out = new BigUint64Array(15);
    check(symbols.llamaProgramInspect(handle, out));
    return llamaProgramInspectionFromWords(out);
  }

  function programRuntimeProfile(handle: NativeHandle): RuntimeProfileEvidence {
    const out = new BigUint64Array(21);
    check(symbols.programRuntimeProfile(handle, out));
    return runtimeProfileFromAbiWords(out);
  }

  function sessionRuntimeProfile(handle: NativeHandle): RuntimeProfileEvidence {
    const out = new BigUint64Array(21);
    check(symbols.sessionRuntimeProfile(handle, out));
    return runtimeProfileFromAbiWords(out);
  }

  return Object.freeze({
    compileDesc,
    inspectModelHandle,
    compileModelProgramHandle,
    inspectBufferHandle,
    inspectSessionHandle,
    llamaSessionPosition,
    resetSessionHandle,
    resetSessionRuntimeProfileHandle,
    inspectExecutableProgram,
    programExecutionCapabilities,
    programRequirements,
    programModelCompatibility,
    inspectLlamaProgram,
    programRuntimeProfile,
    sessionRuntimeProfile,
  });
}
