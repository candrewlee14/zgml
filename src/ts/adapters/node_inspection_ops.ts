import {
  llamaProgramInspectionFromRecord,
} from "../runtime/llama_program_inspection.js";
import {
  compileDescRecordFromOptions,
  type NormalizedCompileOptions,
} from "../runtime/abi.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type NodeInspectionSymbols = {
  modelInspect(handle: NativeHandle, out: Record<string, unknown>): number;
  programCompile(handle: NativeHandle, desc: Record<string, unknown>, out: NativeOut): number;
  bufferInspect(handle: NativeHandle, out: Record<string, unknown>): number;
  sessionInspect(handle: NativeHandle, out: Record<string, unknown>): number;
  sessionPosition(handle: NativeHandle, out: [number]): number;
  sessionReset(handle: NativeHandle): number;
  sessionRuntimeProfile(handle: NativeHandle, out: Record<string, unknown>): number;
  sessionResetRuntimeProfile(handle: NativeHandle): number;
  programInspect(handle: NativeHandle, out: Record<string, unknown>): number;
  programGetRequirements(handle: NativeHandle, out: Record<string, unknown>): number;
  programCheckModelCompatibility(
    handle: NativeHandle,
    model: NativeHandle | null,
    out: Record<string, unknown>,
  ): number;
  llamaProgramInspect(handle: NativeHandle, out: Record<string, unknown>): number;
  programRuntimeProfile(handle: NativeHandle, out: Record<string, unknown>): number;
};

export type NodeInspectionOpsOptions = Readonly<{
  symbols: NodeInspectionSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  normalizeCompileOptions(options: unknown): NormalizedCompileOptions;
  modelHandleForCompatibility(model: unknown): NativeHandle | null;
  modelInspectionFromAbiRecord(record: Record<string, unknown>): unknown;
  bufferInspectionFromAbiRecord(record: Record<string, unknown>): unknown;
  sessionInspectionFromAbiRecord(record: Record<string, unknown>): unknown;
  programInspectionFromAbiRecord(record: Record<string, unknown>): unknown;
  programExecutionCapabilitiesFromInspection(inspection: unknown): unknown;
  programRequirementsFromAbiRecord(record: Record<string, unknown>): unknown;
  programModelCompatibilityFromAbiRecord(record: Record<string, unknown>): unknown;
  runtimeProfileFromAbiRecord(record: Record<string, unknown>): unknown;
}>;

export function createNodeInspectionOps(options: NodeInspectionOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    normalizeCompileOptions,
    modelHandleForCompatibility,
    modelInspectionFromAbiRecord,
    bufferInspectionFromAbiRecord,
    sessionInspectionFromAbiRecord,
    programInspectionFromAbiRecord,
    programExecutionCapabilitiesFromInspection,
    programRequirementsFromAbiRecord,
    programModelCompatibilityFromAbiRecord,
    runtimeProfileFromAbiRecord,
  } = options;

  function compileDesc(compileOptions: unknown = {}): Record<string, unknown> {
    return compileDescRecordFromOptions(normalizeCompileOptions(compileOptions));
  }

  function modelInspectionFromNative(out: Record<string, unknown>): unknown {
    return modelInspectionFromAbiRecord(out);
  }

  function inspectModelHandle(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.modelInspect(handle, out));
    return modelInspectionFromNative(out);
  }

  function compileModelProgramHandle(handle: NativeHandle, compileOptions: unknown = {}): NativeHandle {
    const out = handleOut();
    check(symbols.programCompile(handle, compileDesc(compileOptions), out));
    return readHandle(out, "program");
  }

  function inspectBufferHandle(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.bufferInspect(handle, out));
    return bufferInspectionFromAbiRecord(out);
  }

  function inspectSessionHandle(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.sessionInspect(handle, out));
    return sessionInspectionFromAbiRecord(out);
  }

  function llamaSessionPosition(handle: NativeHandle): number {
    const out: [number] = [0];
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
    const out: Record<string, unknown> = {};
    check(symbols.programInspect(handle, out));
    return programInspectionFromAbiRecord(out);
  }

  function programExecutionCapabilities(handle: NativeHandle): unknown {
    return programExecutionCapabilitiesFromInspection(inspectExecutableProgram(handle));
  }

  function programRequirements(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.programGetRequirements(handle, out));
    return programRequirementsFromAbiRecord(out);
  }

  function programModelCompatibility(handle: NativeHandle, model: unknown): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.programCheckModelCompatibility(handle, modelHandleForCompatibility(model), out));
    return programModelCompatibilityFromAbiRecord(out);
  }

  function inspectLlamaProgram(handle: NativeHandle) {
    const out: Record<string, unknown> = {};
    check(symbols.llamaProgramInspect(handle, out));
    return llamaProgramInspectionFromRecord(out);
  }

  function programRuntimeProfile(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.programRuntimeProfile(handle, out));
    return runtimeProfileFromAbiRecord(out);
  }

  function sessionRuntimeProfile(handle: NativeHandle): unknown {
    const out: Record<string, unknown> = {};
    check(symbols.sessionRuntimeProfile(handle, out));
    return runtimeProfileFromAbiRecord(out);
  }

  return Object.freeze({
    compileDesc,
    modelInspectionFromNative,
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
