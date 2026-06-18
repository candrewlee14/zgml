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
    const result: Record<string, unknown> = {};
    check(symbols.sessionStep(handle, sessionStepIoRecord(input, output), result));
    return outputLenFromStepResultRecord(result);
  }

  function stepNoOutput(handle: NativeHandle, input: SessionStepArrayLike | null | undefined): number {
    const result: Record<string, unknown> = {};
    check(symbols.sessionStepNoOutput(handle, sessionStepNoOutputRecord(input), result));
    return outputLenFromStepResultRecord(result);
  }

  return Object.freeze({
    sessionUploadPersistent,
    sessionUploadPersistentRange,
    stepSession,
    stepNoOutput,
  });
}
