import {
  outputLenFromStepResultRecord,
  sessionStepIoRecord,
  sessionStepNoOutputRecord,
  type SessionStepArrayLike,
} from "../runtime/session_step_io.js";

type NativeHandle = unknown;

type NodeSessionSymbols = {
  sessionUploadPersistent(handle: NativeHandle): number;
  sessionUploadPersistentRange(handle: NativeHandle, first: number, len: number): number;
  sessionStep(handle: NativeHandle, desc: Record<string, unknown> | null, result: Record<string, unknown>): number;
  sessionStepDirect(
    handle: NativeHandle,
    input: SessionStepArrayLike,
    inputLen: number,
    output: SessionStepArrayLike,
    outputLen: number,
  ): number;
  sessionStepNoOutput(handle: NativeHandle, desc: Record<string, unknown> | null, result: Record<string, unknown>): number;
};

export type NodeSessionOpsOptions = Readonly<{
  symbols: NodeSessionSymbols;
  check(code: number): void;
}>;

export function createNodeSessionOps(options: NodeSessionOpsOptions) {
  const {
    symbols,
    check,
  } = options;
  const resultByHandle = new Map<NativeHandle, Record<string, unknown>>();
  const stepRecordByInput = new WeakMap<object, WeakMap<object, Record<string, unknown>>>();
  const stepNoOutputRecordByInput = new WeakMap<object, Record<string, unknown>>();

  function resultRecord(handle: NativeHandle): Record<string, unknown> {
    let result = resultByHandle.get(handle);
    if (!result) {
      result = {};
      resultByHandle.set(handle, result);
    }
    return result;
  }

  function cachedStepRecord(
    input: SessionStepArrayLike | null | undefined,
    output: SessionStepArrayLike | null | undefined,
  ): Record<string, unknown> | null {
    if (!input || !output || typeof input !== "object" || typeof output !== "object") {
      return sessionStepIoRecord(input, output);
    }
    let byOutput = stepRecordByInput.get(input);
    if (!byOutput) {
      byOutput = new WeakMap<object, Record<string, unknown>>();
      stepRecordByInput.set(input, byOutput);
    }
    let record = byOutput.get(output);
    if (!record) {
      record = sessionStepIoRecord(input, output)!;
      byOutput.set(output, record);
    }
    return record;
  }

  function cachedStepNoOutputRecord(input: SessionStepArrayLike | null | undefined): Record<string, unknown> | null {
    if (!input || typeof input !== "object") return sessionStepNoOutputRecord(input);
    let record = stepNoOutputRecordByInput.get(input);
    if (!record) {
      record = sessionStepNoOutputRecord(input)!;
      stepNoOutputRecordByInput.set(input, record);
    }
    return record;
  }

  function sessionUploadPersistent(handle: NativeHandle): void {
    check(symbols.sessionUploadPersistent(handle));
  }

  function sessionUploadPersistentRange(handle: NativeHandle, first: number, len: number): void {
    check(symbols.sessionUploadPersistentRange(handle, first, len));
  }

  function stepSession(
    handle: NativeHandle,
    input: SessionStepArrayLike | null | undefined,
    output: SessionStepArrayLike | null | undefined,
  ): number {
    if (input && output) {
      check(symbols.sessionStepDirect(handle, input, input.length, output, output.length));
      return output.length;
    }
    const result = resultRecord(handle);
    check(symbols.sessionStep(handle, cachedStepRecord(input, output), result));
    return outputLenFromStepResultRecord(result);
  }

  function stepNoOutput(handle: NativeHandle, input: SessionStepArrayLike | null | undefined): number {
    const result = resultRecord(handle);
    check(symbols.sessionStepNoOutput(handle, cachedStepNoOutputRecord(input), result));
    return outputLenFromStepResultRecord(result);
  }

  return Object.freeze({
    sessionUploadPersistent,
    sessionUploadPersistentRange,
    stepSession,
    stepNoOutput,
  });
}
