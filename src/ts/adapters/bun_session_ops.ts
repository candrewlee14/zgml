import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import {
  outputLenFromStepResultWords,
  sessionStepIoFields,
  sessionStepNoOutputFields,
  type SessionStepIoFields,
  type SessionStepArrayLike,
} from "../runtime/session_step_io.js";

type NativeHandle = number;

type BunSessionSymbols = Readonly<{
  sessionUploadPersistent(handle: NativeHandle): number;
  sessionUploadPersistentRange(handle: NativeHandle, first: bigint, len: bigint): number;
  sessionStep(handle: NativeHandle, desc: BigUint64Array | NativeHandle, result: BigUint64Array): number;
  sessionStepNoOutput(handle: NativeHandle, desc: BigUint64Array | NativeHandle, result: BigUint64Array): number;
}>;

export type BunSessionOpsOptions = Readonly<{
  symbols: BunSessionSymbols;
  check(code: number): void;
  pointerFor(value: SessionStepArrayLike): NativeHandle;
}>;

export function createBunSessionOps(options: BunSessionOpsOptions) {
  const {
    symbols,
    check,
    pointerFor,
  } = options;

  function sessionUploadPersistent(handle: NativeHandle): void {
    check(symbols.sessionUploadPersistent(handle));
  }

  function sessionUploadPersistentRange(handle: NativeHandle, first: number, len: number): void {
    check(symbols.sessionUploadPersistentRange(handle, BigInt(first), BigInt(len)));
  }

  function stepDesc(fields: SessionStepIoFields): BigUint64Array {
    const buf = new BigUint64Array(4);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, fields.input ? pointerFor(fields.input) : 0);
    setUSize(view, 8, fields.inputLen);
    setPtr(view, 16, fields.output ? pointerFor(fields.output) : 0);
    setUSize(view, 24, fields.outputLen);
    return buf;
  }

  function stepSession(handle: NativeHandle, input: SessionStepArrayLike | null, output: SessionStepArrayLike | null): number {
    const result = new BigUint64Array(1);
    const fields = sessionStepIoFields(input, output);
    check(symbols.sessionStep(handle, fields ? stepDesc(fields) : 0, result));
    return outputLenFromStepResultWords(result);
  }

  function prepareStepSession(handle: NativeHandle, input: SessionStepArrayLike | null, output: SessionStepArrayLike | null): () => number {
    const result = new BigUint64Array(1);
    const fields = sessionStepIoFields(input, output);
    const desc = fields ? stepDesc(fields) : 0;
    return function preparedStepSession() {
      check(symbols.sessionStep(handle, desc, result));
      return outputLenFromStepResultWords(result);
    };
  }

  function stepNoOutput(handle: NativeHandle, input: SessionStepArrayLike | null): number {
    const result = new BigUint64Array(1);
    const fields = sessionStepNoOutputFields(input);
    check(symbols.sessionStepNoOutput(handle, fields ? stepDesc(fields) : 0, result));
    return outputLenFromStepResultWords(result);
  }

  return Object.freeze({
    sessionUploadPersistent,
    sessionUploadPersistentRange,
    stepSession,
    prepareStepSession,
    stepNoOutput,
  });
}
